from __future__ import annotations
import numpy as np
import math
import torch
import torch.nn.utils as nn_utils
import torch.optim as optim
from dataclasses import dataclass
from .models import PolyValue, POEMVPolicy, value_fn
from .filtering import FilterParams, wonham_filter_q_update
from .utils import safe_clip_p

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
    sigma: float = 0.22
    m_poly: int = 2
    mg_poly: int = 2
    alpha_w: float = 1e-3
    alpha_theta: float = 1e-7
    alpha_phi: float = 1e-4
    omega_init: float = 0.0
    omega_update_every: int = 10
    grad_clip: float = 1.0
    phi3_init: float = -3.0
    phi3_min: float = -10.0
    phi3_max: float = 10.0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

class POEMVAgent:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.vf = PolyValue(T=cfg.T_years, m=cfg.m_poly, mg=cfg.mg_poly).to(cfg.device)
        self.pi = POEMVPolicy().to(cfg.device)
        self.pi = POEMVPolicy().to(cfg.device)
        with torch.no_grad():
            self.pi.phi[2].fill_(cfg.phi3_init)
        self.omega = torch.tensor(cfg.omega_init, dtype=cfg.dtype, device=cfg.device)
        #self.opt_theta = optim.SGD(self.vf.parameters(), lr=cfg.alpha_theta)
        #self.opt_phi = optim.SGD(self.pi.parameters(), lr=cfg.alpha_phi)
        # Adam is much more stable than SGD for this problem (prevents one-shot explosions)
        self.opt_theta = optim.Adam(self.vf.parameters(), lr=cfg.alpha_theta)
        self.opt_phi = optim.Adam(self.pi.parameters(), lr=cfg.alpha_phi)
    
    def _h_terminal(self, xT: torch.Tensor) -> torch.Tensor:
        # h(X_T) = (X_T - omega)^2 - (omega - z)^2   (paper's quadratic terminal form)
        return (xT - self.omega)**2 - (self.omega - self.cfg.z)**2

    def act(self, t: float, x: float, p: float):
        cfg = self.cfg
        t_t = torch.tensor([t], dtype=cfg.dtype, device=cfg.device)
        x_t = torch.tensor([x], dtype=cfg.dtype, device=cfg.device)
        p_t = torch.tensor([p], dtype=cfg.dtype, device=cfg.device)

        f = self.vf.f(t_t, p_t)
        dlnf = self.vf.dlnf_dp(t_t, p_t)
        #mean, std = self.pi.dist_params(x_t, self.omega, p_t, dlnf, f, cfg.Lambda)
        mean, std = self.pi.dist_params(x_t, self.omega, p_t, dlnf, f, cfg.Lambda, cfg.sigma)
        eps = torch.randn_like(std)
        #u = (mean + std*eps).detach().cpu().numpy().item()
        #logprob = (-0.5*(((torch.tensor([u], dtype=cfg.dtype, device=cfg.device)-mean)/std)**2)
        #           - torch.log(std) - 0.5*np.log(2*np.pi)).squeeze()
        #entropy = (0.5*torch.log(2*np.pi*np.e*std*std)).squeeze()
        u_raw = (mean + std * eps).detach().cpu().numpy().item()
        a = float(cfg.a_max * np.tanh(u_raw))
        return a, u_raw, None


    def update_from_episode(self, traj):
        cfg = self.cfg
        t = torch.tensor(traj["t"], dtype=cfg.dtype, device=cfg.device)   # (n+1,)
        x = torch.tensor(traj["x"], dtype=cfg.dtype, device=cfg.device)   # (n+1,)
        p = torch.tensor(traj["p"], dtype=cfg.dtype, device=cfg.device)   # (n+1,)
        u = torch.tensor(traj["u"], dtype=cfg.dtype, device=cfg.device)   # (n,)
        u_raw = torch.tensor(traj["u_raw"], dtype=cfg.dtype, device=cfg.device)  # (n,)
        a_max = float(traj.get("a_max", cfg.a_max))
        a_max_t = torch.tensor(a_max, dtype=cfg.dtype, device=cfg.device)

        dt = torch.as_tensor(cfg.dt, dtype=cfg.dtype, device=cfg.device)

        # states at decision times k=0..n-1
        t0, x0, p0 = t[:-1], x[:-1], p[:-1]  # (n,)

        # --- policy moments---
        #f = self.vf.f(t0, p0)                      # (n,)
        #dlnf = self.vf.dlnf_dp(t0, p0)             # (n,)
        # --- policy moments ---
        # IMPORTANT: keep actor gradients out of the value-function (vf)
        # We treat f, dlnf as constants when updating the policy.
        with torch.no_grad():
            f = self.vf.f(t0, p0)                      # (n,)
            dlnf = self.vf.dlnf_dp(t0, p0)             # (n,)
        mean, std = self.pi.dist_params(x0, self.omega, p0, dlnf, f, cfg.Lambda, cfg.sigma)
        std = std.clamp(min=1e-12)

        # entropy H_k = 0.5*log(2*pi*e*std^2) = 0.5*log(2*pi*e) + log(std)
        ENT_CONST = 0.5 * math.log(2.0 * math.pi * math.e)
        entropy = ENT_CONST + torch.log(std)       # (n,)

        # logprob under current policy for *executed* (squashed) action
        # First compute log p(u_raw) under Normal(mean,std)
        LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
        #z = (u - mean) / std
        #logprob = (-0.5 * z**2 - torch.log(std) - LOG_SQRT_2PI)  # (n,)
        z = (u_raw - mean) / std
        logprob_raw = (-0.5 * z**2 - torch.log(std) - LOG_SQRT_2PI)  # (n,)
        # squash: a = a_max * tanh(u_raw)
        tanh_u = torch.tanh(u_raw)
        # Jacobian: |da/du| = a_max * (1 - tanh^2(u))
        log_det = torch.log(a_max_t) + torch.log((1.0 - tanh_u**2).clamp(min=1e-12))
        logprob = logprob_raw - log_det

        # (Optional) enforce consistency between stored executed action and squash(u_raw)
        # Comment out if you intentionally store something else in traj["u"].
        #u_from_raw = a_max_t * tanh_u
        #if torch.max(torch.abs(u - u_from_raw)).item() > 1e-6:
        #    raise ValueError("Mismatch between executed action u and a_max*tanh(u_raw).") 

        # --- critic: Martingale Loss ---
        xT = x[-1]
        hT = self._h_terminal(xT)                  # scalar tensor
        #G = torch.flip(torch.cumsum(torch.flip(entropy, dims=[0]), dim=0), dims=[0]) * dt  # (n,)
        #Y = hT - cfg.Lambda * G                    # (n,)
        Y = hT.expand_as(t0)

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

        self.opt_phi.zero_grad(set_to_none=True)
        #loss_actor = -(logprob * A).mean()
        # entropy bonus (encourage exploration): subtract beta*H*dt from the loss
        # because we minimize loss, this pushes entropy UP.
        loss_actor = -(logprob * A).mean() - (cfg.Lambda * entropy * dt).mean()
        loss_actor.backward()
        nn_utils.clip_grad_norm_(self.pi.parameters(), max_norm=cfg.grad_clip)
        self.opt_phi.step()

        # clamp phi3 (log-variance control) after update to prevent variance blow-up
        with torch.no_grad():
            self.pi.phi[2].clamp_(cfg.phi3_min, cfg.phi3_max)

        return float(loss_critic.detach().cpu()), float(loss_actor.detach().cpu())

    def update_omega(self, mean_terminal_wealth: float):
        # dual ascent: omega <- omega + alpha_w (E[X_T]-z)
        cfg = self.cfg
        self.omega = self.omega + cfg.alpha_w * torch.tensor((mean_terminal_wealth - cfg.z),
                                                             dtype=cfg.dtype, device=cfg.device)

    def phi_values(self):
        v = self.pi.phi.detach().cpu().numpy()
        return dict(phi1=float(v[0]), phi2=float(v[1]), phi3=float(v[2]))
