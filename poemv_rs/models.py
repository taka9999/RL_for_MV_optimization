from __future__ import annotations
import torch
import torch.nn as nn

class PolyValue(nn.Module):
    """Parametrize ln f(t,p) and g(t,p) with low-order polynomials.

    ln f(t,p) = sum_{i=0..m} sum_{j=0..m} theta[i,j] * p^i * (T-t)^j
    if g_p_dep:
        g(t,p) = sum_{i=0..mg} sum_{j=0..mg} theta_g[i,j] * p^i * (T-t)^j
    else:
        g(t)   = sum_{j=0..mg} theta_g[j] * (T-t)^j
    """
    def __init__(self, T: float, m: int = 2, mg: int = 2, g_p_dep: bool = False):
        super().__init__()
        self.T = float(T)
        self.m = int(m)
        self.mg = int(mg)
        self.g_p_dep = bool(g_p_dep)
        self.theta = nn.Parameter(torch.zeros((m+1, m+1), dtype=torch.float64))
        self.theta_g = nn.Parameter(torch.zeros((mg+1, mg+1), dtype=torch.float64) if self.g_p_dep else torch.zeros((mg+1,), dtype=torch.float64))

    def _tau(self, t: torch.Tensor) -> torch.Tensor:
        return (self.T - t).clamp(min=0.0)

    def ln_f(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        tau = self._tau(t)
        out = torch.zeros_like(p, dtype=torch.float64)
        for i in range(self.m+1):
            for j in range(self.m+1):
                out = out + self.theta[i, j] * (p**i) * (tau**j)
        return out

    def f(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.ln_f(t, p))

    def dlnf_dp(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        tau = self._tau(t)
        out = torch.zeros_like(p, dtype=torch.float64)
        for i in range(1, self.m+1):
            for j in range(self.m+1):
                out = out + self.theta[i, j] * i * (p**(i-1)) * (tau**j)
        return out  # equals f_p / f

    def g(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        tau = self._tau(t)
        out = torch.zeros_like(p, dtype=torch.float64)
        if self.g_p_dep:
            for i in range(self.mg+1):
                for j in range(self.mg+1):
                    out = out + self.theta_g[i, j] * (p**i) * (tau**j)
        else:
            for j in range(self.mg+1):
                out = out + self.theta_g[j] * (tau**j)
        return out

class POEMVPolicy(nn.Module):
    """Wu-Li-style policy with common Sigma.

    mean = -(x-omega) * Sigma^{-1/2} [rho2 + (rho1-rho2)(p - p(1-p) f_p/f)]
    cov  = (Lambda / 2) * f(t,p) * Sigma^{-1}
    """
    def __init__(self, n_assets: int = 2, rho1_init=None, rho2_init=None):
        super().__init__()
        self.n_assets = int(n_assets)
        if rho1_init is None:
            rho1_init = torch.zeros((self.n_assets,), dtype=torch.float64)
        if rho2_init is None:
            rho2_init = torch.zeros((self.n_assets,), dtype=torch.float64)
        self.rho1 = nn.Parameter(torch.as_tensor(rho1_init, dtype=torch.float64).clone())
        self.rho2 = nn.Parameter(torch.as_tensor(rho2_init, dtype=torch.float64).clone())

    def dist(
        self,
        x: torch.Tensor,
        omega: torch.Tensor,
        p: torch.Tensor,
        dlnf_dp: torch.Tensor,
        f: torch.Tensor,
        Lambda: float,
        cov_scale: float,
        mu1: torch.Tensor,
        mu2: torch.Tensor,
        Sigma: torch.Tensor,
        r: float,
    ):
        if x.ndim == 0:
            x = x.unsqueeze(0)
        if p.ndim == 0:
            p = p.unsqueeze(0)
        if dlnf_dp.ndim == 0:
            dlnf_dp = dlnf_dp.unsqueeze(0)
        if f.ndim == 0:
            f = f.unsqueeze(0)

        Sigma = torch.as_tensor(Sigma, dtype=x.dtype, device=x.device).view(1, self.n_assets, self.n_assets)
        eye = torch.eye(self.n_assets, dtype=x.dtype, device=x.device).unsqueeze(0)
        Sigma = 0.5 * (Sigma + Sigma.transpose(-1, -2)) + 1e-10 * eye
        chol = torch.linalg.cholesky(Sigma)

        adj = p - dlnf_dp * p * (1.0 - p)
        signal = self.rho2.view(1, self.n_assets) + (self.rho1 - self.rho2).view(1, self.n_assets) * adj.unsqueeze(-1)
        sigma_invhalf_signal = torch.linalg.solve_triangular(
            chol.transpose(-1, -2), signal.unsqueeze(-1), upper=True
        ).squeeze(-1)
        mean = -(x - omega).unsqueeze(-1) * sigma_invhalf_signal

        sigma_inv = torch.cholesky_inverse(chol)
        lam = torch.as_tensor(Lambda, dtype=x.dtype, device=x.device)
        cov_scale_t = torch.as_tensor(cov_scale, dtype=x.dtype, device=x.device)
        cov = cov_scale_t * 0.5 * lam * f.view(-1, 1, 1) * sigma_inv
        cov = 0.5 * (cov + cov.transpose(-1, -2)) + 1e-10 * eye

        return torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)

def value_fn(t: torch.Tensor, x: torch.Tensor, p: torch.Tensor, omega: torch.Tensor, z: float,
             vf: PolyValue):
    f = vf.f(t, p)
    g = vf.g(t, p)
    V = (x - omega)**2 / f + g - (omega - z)**2
    return V, f
