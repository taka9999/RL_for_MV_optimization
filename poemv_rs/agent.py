from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import math
import torch
import torch.nn.utils as nn_utils
import torch.optim as optim
from .utils import safe_clip_p
from .models import PolyValue, POEMVPolicy,value_fn

@dataclass
class TrainConfig:
    T_years: float = 10.0
    dt: float = 1/252
    x0: float = 1.0
    s0: float = 1.0
    p0: float = 0.5
    a_max: float = 2.0
    z: float = 2.0
    Lambda: float = 1.0
    r: float = 0.01
    mu1: np.ndarray = field(default_factory=lambda: np.array([0.25, 0.18], dtype=float))
    mu2: np.ndarray = field(default_factory=lambda: np.array([-0.73, -0.40], dtype=float))
    Sigma1: np.ndarray = field(default_factory=lambda: np.array([[0.22**2, 0.22 * 0.18 * 0.3], [0.22 * 0.18 * 0.3, 0.18**2]], dtype=float))
    Sigma2: np.ndarray = field(default_factory=lambda: np.array([[0.22**2, 0.22 * 0.18 * 0.5], [0.22 * 0.18 * 0.5, 0.18**2]], dtype=float))
    m_poly: int = 2
    mg_poly: int = 2
    alpha_w: float = 1e-3
    alpha_theta: float = 1e-7
    alpha_phi: float = 1e-4
    omega_init: float = 0.0
    omega_update_every: int = 10
    grad_clip: float = 1.0
    #phi3_init: float = -3.0
    #phi3_min: float = -10.0
    #phi3_max: float = 10.0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

class POEMVAgent:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.vf = PolyValue(T=cfg.T_years, m=cfg.m_poly, mg=cfg.mg_poly).to(cfg.device)
        #with torch.no_grad():
        #    # set per-asset phi3 init
        #    self.pi.phi[:,2].fill_(cfg.phi3_init)
        self.pi = POEMVPolicy(n_assets=2).to(cfg.device)
        self.omega = torch.tensor(cfg.omega_init, dtype=cfg.dtype, device=cfg.device)
        self.opt_theta = optim.Adam(self.vf.parameters(), lr=cfg.alpha_theta)
        self.opt_phi = optim.Adam(self.pi.parameters(), lr=cfg.alpha_phi)

        self.mu1_t = torch.as_tensor(cfg.mu1, dtype=cfg.dtype, device=cfg.device)
        self.mu2_t = torch.as_tensor(cfg.mu2, dtype=cfg.dtype, device=cfg.device)
        self.Sigma1_t = torch.as_tensor(cfg.Sigma1, dtype=cfg.dtype, device=cfg.device)
        self.Sigma2_t = torch.as_tensor(cfg.Sigma2, dtype=cfg.dtype, device=cfg.device)
    
    def _h_terminal(self, xT: torch.Tensor) -> torch.Tensor:
        return (xT - self.omega)**2 - (self.omega - self.cfg.z)**2
    
    def _policy_dist(self, t: torch.Tensor, x: torch.Tensor, p: torch.Tensor):
        f = self.vf.f(t, p)
        dlnf = self.vf.dlnf_dp(t, p)
        return self.pi.dist(
            x=x,
            omega=self.omega,
            p=p,
            dlnf_dp=dlnf,
            f=f,
            Lambda=self.cfg.Lambda,
            mu1=self.mu1_t,
            mu2=self.mu2_t,
            Sigma1=self.Sigma1_t,
            Sigma2=self.Sigma2_t,
            r=self.cfg.r,
        )

    def act(self, t: float, x: float, p: float):
        cfg = self.cfg
        t_t = torch.tensor([t], dtype=cfg.dtype, device=cfg.device)
        x_t = torch.tensor([x], dtype=cfg.dtype, device=cfg.device)
        p_t = torch.tensor([p], dtype=cfg.dtype, device=cfg.device)
        with torch.no_grad():
            dist = self._policy_dist(t_t, x_t, p_t)
            u_raw_t = dist.sample().squeeze(0)
        u_raw = u_raw_t.detach().cpu().numpy().reshape(2,)
        a = (cfg.a_max * np.tanh(u_raw)).astype(float)
        return a, u_raw, None

    def update_from_episode(self, traj):
        cfg = self.cfg
        t = torch.tensor(traj["t"], dtype=cfg.dtype, device=cfg.device)   # (n+1,)
        t = torch.tensor(traj["t"], dtype=cfg.dtype, device=cfg.device)
        x = torch.tensor(traj["x"], dtype=cfg.dtype, device=cfg.device)
        p = torch.tensor(traj["p"], dtype=cfg.dtype, device=cfg.device)
        u_raw = torch.tensor(traj["u_raw"], dtype=cfg.dtype, device=cfg.device)
        a_max = float(traj.get("a_max", cfg.a_max))
        a_max_t = torch.tensor(a_max, dtype=cfg.dtype, device=cfg.device)

        dt = torch.as_tensor(cfg.dt, dtype=cfg.dtype, device=cfg.device)

        # states at decision times k=0..n-1
        t0, x0, p0 = t[:-1], x[:-1], p[:-1]

        with torch.no_grad():
            f = self.vf.f(t0, p0)                      # (n,)
            dlnf = self.vf.dlnf_dp(t0, p0)             # (n,)
        dist_det = self.pi.dist(
            x=x0,
            omega=self.omega,
            p=p0,
            dlnf_dp=dlnf,
            f=f,
            Lambda=cfg.Lambda,
            mu1=self.mu1_t,
            mu2=self.mu2_t,
            Sigma1=self.Sigma1_t,
            Sigma2=self.Sigma2_t,
            r=cfg.r,
        )
        entropy = dist_det.entropy()
        logprob_raw = dist_det.log_prob(u_raw)
        tanh_u = torch.tanh(u_raw)
        log_det = (torch.log(a_max_t) + torch.log((1.0 - tanh_u**2).clamp(min=1e-12))).sum(dim=-1)
        logprob = logprob_raw - log_det  # (n,)

        xT = x[-1]
        hT = self._h_terminal(xT)
        G = torch.flip(torch.cumsum(torch.flip(entropy, dims=[0]), dim=0), dims=[0]) * dt
        Y = hT - cfg.Lambda * G

        V0, _ = value_fn(t0, x0, p0, self.omega, cfg.z, self.vf)          # (n,)
        self.opt_theta.zero_grad(set_to_none=True)
        loss_critic = 0.5 * ((Y.detach() - V0)**2).mean()
        loss_critic.backward()
        # gradient clipping to prevent explosions
        nn_utils.clip_grad_norm_(self.vf.parameters(), max_norm=cfg.grad_clip)
        self.opt_theta.step()

        # --- actor: REINFORCE + baseline ---
        # advantage A_k = Y_k - V_theta(t_k, ...)
        with torch.no_grad():
            Vb, _ = value_fn(t0, x0, p0, self.omega, cfg.z, self.vf)
            A = (Y - Vb)                            # (n,)
        
        f_actor = self.vf.f(t0, p0).detach()
        dlnf_actor = self.vf.dlnf_dp(t0, p0).detach()
        dist_actor = self.pi.dist(
            x=x0,
            omega=self.omega,
            p=p0,
            dlnf_dp=dlnf_actor,
            f=f_actor,
            Lambda=cfg.Lambda,
            mu1=self.mu1_t,
            mu2=self.mu2_t,
            Sigma1=self.Sigma1_t,
            Sigma2=self.Sigma2_t,
            r=cfg.r,
        )
        entropy_actor = dist_actor.entropy()
        logprob_raw_actor = dist_actor.log_prob(u_raw)
        logprob_actor = logprob_raw_actor - log_det

        self.opt_phi.zero_grad(set_to_none=True)
        loss_actor = -(logprob_actor * A.detach()).mean() - (cfg.Lambda * entropy_actor * dt).mean()
        loss_actor.backward()
        nn_utils.clip_grad_norm_(self.pi.parameters(), max_norm=cfg.grad_clip)
        self.opt_phi.step()

        return float(loss_critic.detach().cpu()), float(loss_actor.detach().cpu())

    def update_omega(self, mean_terminal_wealth: float):
        # dual ascent: omega <- omega + alpha_w (E[X_T]-z)
        cfg = self.cfg
        self.omega = self.omega - cfg.alpha_w * torch.tensor(
            (mean_terminal_wealth - cfg.z), dtype=cfg.dtype, device=cfg.device
        )
    
    def phi_values(self):
        v = self.pi.phi.detach().cpu().numpy()
        return {
            "phi1_asset1": float(v[0, 0]),
            "phi2_asset1": float(v[0, 1]),
            "phi1_asset2": float(v[1, 0]),
            "phi2_asset2": float(v[1, 1]),
        }
