from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import math
import torch
import torch.nn.utils as nn_utils
import torch.optim as optim
from .utils import safe_clip_p
from .models import PolyValue, POEMVPolicy, value_fn

@dataclass
class TrainConfig:
    T_years: float = 10.0
    dt: float = 1/252
    x0: float = 1.0
    s0: float = 1.0
    p0: float = 0.5
    a_max: float | None = None
    cap_mode: str = "component_tanh"  # one of: none, component_tanh, l1_radial
    z: float = 2.0
    Lambda: float = 1.0
    r: float = 0.01
    mu1: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.18], dtype=float))
    mu2: np.ndarray = field(default_factory=lambda: np.array([-0.73, -0.40], dtype=float))
    Sigma: np.ndarray = field(default_factory=lambda: np.array([[0.22**2, 0.22 * 0.18 * 0.3], [0.22 * 0.18 * 0.3, 0.18**2]], dtype=float))
    m_poly: int = 2
    mg_poly: int = 2
    g_p_dep: bool = False
    actor_mix_tail: float = 0.5
    cov_scale: float = 1.0
    lr_step_every: int = 0
    lr_gamma: float = 1.0
    alpha_w: float = 5e-3
    alpha_theta: float = 3e-5
    alpha_phi: float = 1e-4
    omega_init: float = 0.0
    omega_update_every: int = 10
    critic_steps: int = 10
    advantage_norm_eps: float = 1e-8
    omega_ema_beta: float = 0.9
    grad_clip: float = 1.0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64
    episodes_per_iter: int = 10


def _squash_action_np(
    z: np.ndarray,
    x: float,
    a_max: float | None,
    cap_mode: str = "none",
) -> np.ndarray:
    z = np.asarray(z, dtype=float).reshape(2,)
    if (a_max is None) or (cap_mode == "none"):
        return z
    scale = max(abs(float(x)), 1e-12) * float(a_max)
    if cap_mode == "component_tanh":
        return scale * np.tanh(z / max(scale, 1e-12))
    if cap_mode == "l1_radial":
        abs_sum = float(np.sum(np.abs(z)))
        if abs_sum <= 1e-12:
            return z.copy()
        radius = scale * np.tanh(abs_sum / max(scale, 1e-12))
        return (radius / abs_sum) * z
    raise ValueError(f"Unknown cap_mode={cap_mode}")


def _squash_log_prob(
    dist: torch.distributions.MultivariateNormal,
    z_raw: torch.Tensor,
    x: torch.Tensor,
    a_max: float | None,
    cap_mode: str = "none",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-prob / entropy proxy for action squashing.

    cap_mode="none":
        no squash, exact Gaussian log-prob.
    cap_mode="component_tanh":
        component-wise tanh squash with diagonal Jacobian correction.
    cap_mode="l1_radial":
        smooth L1-ball radial squash:
            a = s * tanh(||z||_1 / s) * z / ||z||_1.
        This exactly respects the gross leverage cap in risky-dollar units,
        but the full Jacobian is cumbersome. We therefore use the raw Gaussian
        log-prob / entropy as a practical proxy in this mode.
    """
    if (a_max is None) or (cap_mode == "none"):
        return dist.log_prob(z_raw), dist.entropy()
    if cap_mode == "component_tanh":
        scale = torch.clamp(torch.abs(x), min=1e-12).unsqueeze(-1) * float(a_max)
        y = z_raw / scale
        log_det = torch.log(1.0 - torch.tanh(y).pow(2) + 1e-12).sum(dim=-1)
        log_prob = dist.log_prob(z_raw) - log_det
        entropy_proxy = dist.entropy() + log_det.detach()
        return log_prob, entropy_proxy
    if cap_mode == "l1_radial":
        return dist.log_prob(z_raw), dist.entropy()
    raise ValueError(f"Unknown cap_mode={cap_mode}")

class POEMVAgent:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.vf = PolyValue(T=cfg.T_years, m=cfg.m_poly, mg=cfg.mg_poly, g_p_dep=cfg.g_p_dep).to(cfg.device)
        Sigma = np.asarray(cfg.Sigma, dtype=float)
        L = np.linalg.cholesky(Sigma + 1e-12 * np.eye(Sigma.shape[0]))
        rho1_init = np.linalg.solve(L, np.asarray(cfg.mu1, float) - float(cfg.r))
        rho2_init = np.linalg.solve(L, np.asarray(cfg.mu2, float) - float(cfg.r))
        self.pi = POEMVPolicy(n_assets=2, rho1_init=rho1_init, rho2_init=rho2_init).to(cfg.device)
        self.omega = torch.tensor(cfg.omega_init, dtype=cfg.dtype, device=cfg.device)
        self.opt_theta = optim.Adam(self.vf.parameters(), lr=cfg.alpha_theta)
        self.opt_phi = optim.Adam(self.pi.parameters(), lr=cfg.alpha_phi)
        self.sched_theta = None
        self.sched_phi = None
        if int(cfg.lr_step_every) > 0 and float(cfg.lr_gamma) < 1.0:
            self.sched_theta = optim.lr_scheduler.StepLR(
                self.opt_theta,
                step_size=int(cfg.lr_step_every),
                gamma=float(cfg.lr_gamma),
            )
            self.sched_phi = optim.lr_scheduler.StepLR(
                self.opt_phi,
                step_size=int(cfg.lr_step_every),
                gamma=float(cfg.lr_gamma),
            )
        self.mu1_t = torch.as_tensor(cfg.mu1, dtype=cfg.dtype, device=cfg.device)
        self.mu2_t = torch.as_tensor(cfg.mu2, dtype=cfg.dtype, device=cfg.device)
        self.Sigma_t = torch.as_tensor(cfg.Sigma, dtype=cfg.dtype, device=cfg.device)
        self.mean_xT_ema = None

    def _h_terminal(self, xT: torch.Tensor) -> torch.Tensor:
        return (xT - self.omega)**2 - (self.omega - self.cfg.z)**2

    def _h_terminal_with_omega(self, xT: torch.Tensor, omega: torch.Tensor) -> torch.Tensor:
        return (xT - omega)**2 - (omega - self.cfg.z)**2

    def _policy_features(self, t: torch.Tensor, p: torch.Tensor):
        f = self.vf.f(t, p)
        dlnf = self.vf.dlnf_dp(t, p)
        return f, dlnf
    
    def _policy_dist_from_features(
        self,
        x: torch.Tensor,
        p: torch.Tensor,
        f: torch.Tensor,
        dlnf: torch.Tensor,
        omega: torch.Tensor | None = None,
    ):
        return self.pi.dist(
            x=x,
            omega=self.omega if omega is None else omega,
            p=p,
            dlnf_dp=dlnf,
            f=f,
            Lambda=self.cfg.Lambda,
            cov_scale=self.cfg.cov_scale,
            mu1=self.mu1_t, mu2=self.mu2_t, Sigma=self.Sigma_t, r=self.cfg.r,
        )

    def _policy_dist(self, t: torch.Tensor, x: torch.Tensor, p: torch.Tensor):
        #f = self.vf.f(t, p)
        #dlnf = self.vf.dlnf_dp(t, p)
        f, dlnf = self._policy_features(t, p)
        return self.pi.dist(
            x=x,
            omega=self.omega,
            p=p,
            dlnf_dp=dlnf,
            f=f,
            Lambda=self.cfg.Lambda,
            cov_scale=self.cfg.cov_scale,
            mu1=self.mu1_t,
            mu2=self.mu2_t,
            Sigma=self.Sigma_t,
            r=self.cfg.r,
        )

    def policy_mean(self, t: float, x: float, p: float) -> np.ndarray:
        cfg = self.cfg
        t_t = torch.tensor([t], dtype=cfg.dtype, device=cfg.device)
        x_t = torch.tensor([x], dtype=cfg.dtype, device=cfg.device)
        p_t = torch.tensor([p], dtype=cfg.dtype, device=cfg.device)
        with torch.no_grad():
            dist = self._policy_dist(t_t, x_t, p_t)
            z = dist.mean.squeeze(0).detach().cpu().numpy().reshape(2,)
        return _squash_action_np(z, x=float(x), a_max=cfg.a_max, cap_mode=cfg.cap_mode).astype(float)


    def act(self, t: float, x: float, p: float, deterministic: bool = False):
        cfg = self.cfg
        t_t = torch.tensor([t], dtype=cfg.dtype, device=cfg.device)
        x_t = torch.tensor([x], dtype=cfg.dtype, device=cfg.device)
        p_t = torch.tensor([p], dtype=cfg.dtype, device=cfg.device)
        with torch.no_grad():
            f_t, dlnf_t = self._policy_features(t_t, p_t)
            dist = self._policy_dist_from_features(x_t, p_t, f_t, dlnf_t, omega=self.omega)
            if deterministic:
                z_raw_t = dist.mean.squeeze(0)
            else:
                z_raw_t = dist.rsample().squeeze(0)
        z_raw = z_raw_t.detach().cpu().numpy().reshape(2,)
        u = _squash_action_np(z_raw, x=float(x), a_max=cfg.a_max,cap_mode=cfg.cap_mode)
        info = {
            "f_roll": float(f_t.squeeze(0).detach().cpu().item()),
            "dlnf_roll": float(dlnf_t.squeeze(0).detach().cpu().item()),
            "omega_roll": float(self.omega.detach().cpu().item()),
        }
        return u.astype(float), z_raw.astype(float), info

    def _episode_losses(self, traj):
        cfg = self.cfg
        t = torch.tensor(traj["t"], dtype=cfg.dtype, device=cfg.device)
        x = torch.tensor(traj["x"], dtype=cfg.dtype, device=cfg.device)
        p = torch.tensor(traj["p"], dtype=cfg.dtype, device=cfg.device)
        z_raw = torch.tensor(traj["u_raw"], dtype=cfg.dtype, device=cfg.device)
        xT = x[-1]

        dt = torch.as_tensor(cfg.dt, dtype=cfg.dtype, device=cfg.device)
        t0, x0, p0 = t[:-1], x[:-1], p[:-1]
        t1, x1, p1 = t[1:], x[1:], p[1:]

        f_roll = traj.get("f_roll", None)
        dlnf_roll = traj.get("dlnf_roll", None)
        omega_roll = traj.get("omega_roll", None)

        if f_roll is None or dlnf_roll is None:
            # backward-compatible fallback
            with torch.no_grad():
                f_roll_t, dlnf_roll_t = self._policy_features(t0, p0)
        else:
            f_roll_t = torch.tensor(f_roll, dtype=cfg.dtype, device=cfg.device)
            dlnf_roll_t = torch.tensor(dlnf_roll, dtype=cfg.dtype, device=cfg.device)

        if omega_roll is None:
            omega_roll_t = self.omega
        else:
            omega_roll_t = torch.tensor(float(omega_roll), dtype=cfg.dtype, device=cfg.device)
        #omega_term = omega_roll_t[-1] if omega_roll_t.ndim > 0 else omega_roll_t
        omega_term = omega_roll_t
        hT = self._h_terminal_with_omega(xT, omega_term)
        
        with torch.no_grad():
            #dist_b = self._policy_dist(t0, x0, p0)
            dist_b = self.pi.dist(
                x=x0,
                omega=omega_roll_t,
                p=p0,
                dlnf_dp=dlnf_roll_t,
                f=f_roll_t,
                Lambda=cfg.Lambda,
                cov_scale=cfg.cov_scale,
                mu1=self.mu1_t,
                mu2=self.mu2_t,
                Sigma=self.Sigma_t,
                r=cfg.r,
            )
            _, entropy_b = _squash_log_prob(dist_b, z_raw, x0, cfg.a_max, cfg.cap_mode)
            entropy_tail = torch.flip(torch.cumsum(torch.flip(entropy_b * dt, dims=[0]), dim=0), dims=[0])
            target = hT - cfg.Lambda * entropy_tail

        V_pred, _ = value_fn(t0, x0, p0, omega_roll_t, cfg.z, self.vf)
        loss_c = 0.5 * ((V_pred - target.detach()) ** 2).mean()

        with torch.no_grad():
            V_now, _ = value_fn(t0, x0, p0, omega_roll_t, cfg.z, self.vf)
            #A = target - V_now
            #A = (A - A.mean()) / (A.std(unbiased=False) + cfg.advantage_norm_eps)
            V_next, _ = value_fn(t1, x1, p1, omega_roll_t, cfg.z, self.vf)
            # paper-style one-step martingale increment:
            #   V_{k+1} - V_k - Lambda * H_k * dt
            martingale_inc = V_next - V_now - cfg.Lambda * entropy_b * dt
            martingale_inc = (
                martingale_inc - martingale_inc.mean()
            ) / (martingale_inc.std(unbiased=False) + cfg.advantage_norm_eps)
            tail_adv = target - V_now
            tail_adv = (
                tail_adv - tail_adv.mean()
            ) / (tail_adv.std(unbiased=False) + cfg.advantage_norm_eps)
            mix = float(cfg.actor_mix_tail)
            actor_signal = (1.0 - mix) * martingale_inc + mix * tail_adv

        #f_actor = self.vf.f(t0, p0).detach()
        #dlnf_actor = self.vf.dlnf_dp(t0, p0).detach()
        f_actor = f_roll_t.detach()
        dlnf_actor = dlnf_roll_t.detach()
        dist_actor = self.pi.dist(
            x=x0,
            omega=omega_roll_t,
            p=p0,
            dlnf_dp=dlnf_actor,
            f=f_actor,
            Lambda=cfg.Lambda,
            cov_scale=cfg.cov_scale,
            mu1=self.mu1_t,
            mu2=self.mu2_t,
            Sigma=self.Sigma_t,
            r=cfg.r,
        )
        logp_actor, entropy_actor = _squash_log_prob(dist_actor, z_raw, x0, cfg.a_max,cfg.cap_mode)
        #loss_a = -(logp_actor * A.detach()).mean() - (cfg.Lambda * entropy_actor * dt).mean()
        # Martingale-increment policy gradient surrogate:
        # maximize   log pi(a_k|s_k) * (V_{k+1} - V_k - Lambda H_k dt) - Lambda H_k dt
        # with V terms detached (critic treated as baseline / target here)
        loss_a = (
            -(logp_actor * actor_signal.detach()).mean()
            - (cfg.Lambda * entropy_actor * dt).mean()
        )
        return loss_c, loss_a

    def update_from_episode(self, traj):
        return self.update_from_episodes([traj])

    def update_from_episodes(self, trajs):
        cfg = self.cfg
        if len(trajs) == 0:
            raise ValueError("trajs must be non-empty")

        loss_critic = 0.0
        for _ in range(int(cfg.critic_steps)):
            critic_losses = [self._episode_losses(traj)[0] for traj in trajs]
            loss_c = torch.stack(critic_losses).mean()
            self.opt_theta.zero_grad(set_to_none=True)
            loss_c.backward()
            nn_utils.clip_grad_norm_(self.vf.parameters(), max_norm=cfg.grad_clip)
            self.opt_theta.step()
            loss_critic = float(loss_c.detach().cpu())

        actor_losses = [self._episode_losses(traj)[1] for traj in trajs]
        loss_actor = torch.stack(actor_losses).mean()
        self.opt_phi.zero_grad(set_to_none=True)
        loss_actor.backward()
        nn_utils.clip_grad_norm_(self.pi.parameters(), max_norm=cfg.grad_clip)
        self.opt_phi.step()

        return float(loss_critic), float(loss_actor.detach().cpu())
    
    def step_schedulers(self):
        if self.sched_theta is not None:
            self.sched_theta.step()
        if self.sched_phi is not None:
            self.sched_phi.step()

    def current_lrs(self):
        return {"lr_theta": float(self.opt_theta.param_groups[0]["lr"]), "lr_phi": float(self.opt_phi.param_groups[0]["lr"])}

    def update_omega(self, mean_xT: float):
        cfg = self.cfg
        if self.mean_xT_ema is None:
            self.mean_xT_ema = float(mean_xT)
        else:
            beta = float(cfg.omega_ema_beta)
            self.mean_xT_ema = beta * self.mean_xT_ema + (1.0 - beta) * float(mean_xT)
        self.omega = self.omega - cfg.alpha_w * torch.tensor(
            (self.mean_xT_ema - cfg.z), dtype=cfg.dtype, device=cfg.device)

    def policy_values(self):
        r1 = self.pi.rho1.detach().cpu().numpy()
        r2 = self.pi.rho2.detach().cpu().numpy()
        return {
            "rho1_asset1": float(r1[0]),
            "rho2_asset1": float(r2[0]),
            "rho1_asset2": float(r1[1]),
            "rho2_asset2": float(r2[1]),
        }
