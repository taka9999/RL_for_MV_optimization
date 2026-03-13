from __future__ import annotations
import torch
import torch.nn as nn

class PolyValue(nn.Module):
    """Parametrize ln f(t,p) and g(t,p) with low-order polynomials.

    ln f(t,p) = sum_{i=0..m} sum_{j=0..m} theta[i,j] * p^i * (T-t)^j
    g(t,p)    = sum_{j=1..mg} theta_g[j] * (T-t)^j   (paper simplification)
    """
    def __init__(self, T: float, m: int = 2, mg: int = 2):
        super().__init__()
        self.T = float(T)
        self.m = int(m)
        self.mg = int(mg)
        self.theta = nn.Parameter(torch.zeros((m+1, m+1), dtype=torch.float64))
        self.theta_g = nn.Parameter(torch.zeros((mg+1,), dtype=torch.float64))  # index 0 unused

    def _tau(self, t: torch.Tensor) -> torch.Tensor:
        return (self.T - t).clamp(min=0.0)

    def ln_f(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        tau = self._tau(t)
        out = torch.zeros_like(p, dtype=torch.float64)
        # naive loops are fine (m=2)
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
        for j in range(1, self.mg+1):
            out = out + self.theta_g[j] * (tau**j)
        return out

class POEMVPolicy(nn.Module):
    """2-asset independent Gaussian policy; per-asset (phi1, phi2, phi3).
    Mean uses a regime-belief-weighted excess-drift signal passed through (sigma^T)^-1.
    Covariance is scaled as 0.5 * Lambda * f(t,p) * (sigma sigma^T)^-1 when the
    learnable SPD matrix is initialized to identity, matching the Appendix-A structure.
    """
    def __init__(self, n_assets: int = 2):
        super().__init__()
        self.n_assets = int(n_assets)
        # rows are assets, cols are (phi1, phi2)
        self.phi = nn.Parameter(torch.zeros((self.n_assets, 2), dtype=torch.float64))
        # unconstrained lower-triangular parameter for a learnable SPD covariance core
        self.cov_tril_unconstrained = nn.Parameter(torch.eye(self.n_assets, dtype=torch.float64))

    def _posterior_moments(
        self,
        p: torch.Tensor,
        mu1: torch.Tensor,
        mu2: torch.Tensor,
        Sigma1: torch.Tensor,
        Sigma2: torch.Tensor,
        r: float,
    ):
        p_col = p.unsqueeze(-1)
        mu_bar = p_col * mu1 + (1.0 - p_col) * mu2
        excess_mu = mu_bar - float(r)
        p_mat = p.unsqueeze(-1).unsqueeze(-1)
        Sigma_bar = p_mat * Sigma1 + (1.0 - p_mat) * Sigma2
        eye = torch.eye(self.n_assets, dtype=p.dtype, device=p.device).unsqueeze(0)
        Sigma_bar = Sigma_bar + 1e-10 * eye
        return excess_mu, Sigma_bar

    def _spd_core(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        raw = torch.tril(self.cov_tril_unconstrained.to(dtype=dtype, device=device))
        diag = torch.diagonal(raw, dim1=-2, dim2=-1)
        tril = raw - torch.diag_embed(diag) + torch.diag_embed(torch.exp(diag))
        return tril @ tril.transpose(-1, -2)

    def dist(
        self,
        x: torch.Tensor,
        omega: torch.Tensor,
        p: torch.Tensor,
        dlnf_dp: torch.Tensor,
        f: torch.Tensor,
        Lambda: float,
        mu1: torch.Tensor,
        mu2: torch.Tensor,
        Sigma1: torch.Tensor,
        Sigma2: torch.Tensor,
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

        mu1 = torch.as_tensor(mu1, dtype=x.dtype, device=x.device).view(1, self.n_assets)
        mu2 = torch.as_tensor(mu2, dtype=x.dtype, device=x.device).view(1, self.n_assets)
        Sigma1 = torch.as_tensor(Sigma1, dtype=x.dtype, device=x.device).view(1, self.n_assets, self.n_assets)
        Sigma2 = torch.as_tensor(Sigma2, dtype=x.dtype, device=x.device).view(1, self.n_assets, self.n_assets)

        excess_mu, Sigma_bar = self._posterior_moments(p, mu1, mu2, Sigma1, Sigma2, r)
        chol = torch.linalg.cholesky(Sigma_bar)
        adj = p - dlnf_dp * p * (1.0 - p)
        signal = self.phi[:, 0].view(1, self.n_assets) * (adj.unsqueeze(-1) * excess_mu) + self.phi[:, 1].view(1, self.n_assets) * excess_mu
        sigma_t_inv_signal = torch.linalg.solve_triangular(
            chol.transpose(-1, -2), signal.unsqueeze(-1), upper=True
        ).squeeze(-1)
        mean = -(x - omega).unsqueeze(-1) * sigma_t_inv_signal

        spd_core = self._spd_core(dtype=x.dtype, device=x.device)
        inv_sigma_t = torch.linalg.solve_triangular(
            chol.transpose(-1, -2),
            torch.eye(self.n_assets, dtype=x.dtype, device=x.device).unsqueeze(0).expand_as(Sigma_bar),
            upper=True,
        )

        lam = torch.as_tensor(Lambda, dtype=x.dtype, device=x.device)
        cov = 0.5 * lam * f.view(-1, 1, 1) * (inv_sigma_t @ spd_core.unsqueeze(0) @ inv_sigma_t.transpose(-1, -2))
        eye = torch.eye(self.n_assets, dtype=x.dtype, device=x.device).unsqueeze(0)
        cov = 0.5 * (cov + cov.transpose(-1, -2)) + 1e-10 * eye

        return torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)

def value_fn(t: torch.Tensor, x: torch.Tensor, p: torch.Tensor, omega: torch.Tensor, z: float,
             vf: PolyValue):
    f = vf.f(t, p)
    g = vf.g(t, p)
    V = (x - omega)**2 / f + g - (omega - z)**2
    return V, f
