from __future__ import annotations

import torch
import torch.nn as nn


class PolyValue(nn.Module):
    """Parametrize ln f(t,p) and g(t,p) with low-order polynomials.

    The value ansatz is
        V(t,x,p) = (x-omega)^2 / f(t,p) + g(t,p) - (omega-z)^2.

    This file keeps the same ansatz as the original implementation, while
    allowing the Stage-1 Gaussian policy to switch between:
      - common covariance mode
      - belief-averaged regime-covariance mode
    """

    def __init__(self, T: float, m: int = 2, mg: int = 2, g_p_dep: bool = False):
        super().__init__()
        self.T = float(T)
        self.m = int(m)
        self.mg = int(mg)
        self.g_p_dep = bool(g_p_dep)
        self.theta = nn.Parameter(torch.zeros((m + 1, m + 1), dtype=torch.float64))
        if self.g_p_dep:
            self.theta_g = nn.Parameter(torch.zeros((mg + 1, mg + 1), dtype=torch.float64))
        else:
            self.theta_g = nn.Parameter(torch.zeros((mg + 1,), dtype=torch.float64))

    def _tau(self, t: torch.Tensor) -> torch.Tensor:
        return (self.T - t).clamp(min=0.0)

    def ln_f(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        tau = self._tau(t)
        out = torch.zeros_like(p, dtype=torch.float64)
        for i in range(self.m + 1):
            for j in range(self.m + 1):
                out = out + self.theta[i, j] * (p ** i) * (tau ** j)
        return out

    def f(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.ln_f(t, p))

    def dlnf_dp(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        tau = self._tau(t)
        out = torch.zeros_like(p, dtype=torch.float64)
        for i in range(1, self.m + 1):
            for j in range(self.m + 1):
                out = out + self.theta[i, j] * i * (p ** (i - 1)) * (tau ** j)
        return out

    def g(self, t: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
        tau = self._tau(t)
        out = torch.zeros_like(p, dtype=torch.float64)
        if self.g_p_dep:
            for i in range(self.mg + 1):
                for j in range(self.mg + 1):
                    out = out + self.theta_g[i, j] * (p ** i) * (tau ** j)
        else:
            for j in range(self.mg + 1):
                out = out + self.theta_g[j] * (tau ** j)
        return out


def belief_avg_covariance(
    p: torch.Tensor,
    Sigma1: torch.Tensor,
    Sigma2: torch.Tensor,
) -> torch.Tensor:
    """Return belief-averaged covariance p*Sigma1 + (1-p)*Sigma2.

    Parameters
    ----------
    p : (B,) tensor in [0,1]
    Sigma1, Sigma2 : (d,d) tensors
    """
    if p.ndim == 0:
        p = p.unsqueeze(0)
    B = p.shape[0]
    Sigma1 = Sigma1.view(1, *Sigma1.shape).expand(B, -1, -1)
    Sigma2 = Sigma2.view(1, *Sigma2.shape).expand(B, -1, -1)
    return p.view(B, 1, 1) * Sigma1 + (1.0 - p).view(B, 1, 1) * Sigma2


class Stage1GaussianPolicyCovModes(nn.Module):
    """Stage-1 Gaussian policy with covariance mode switch.

    Modes
    -----
    common
        Recover the original common-Sigma Wu-Li-style structure:
            mean = -(x-omega) * Sigma^{-1/2} * signal(p, dlnf)
            cov  = (Lambda/2) * f(t,p) * Sigma^{-1}

    belief_avg_regime
        Generalized belief-state Gaussian policy based on the practical
        approximation
            Sigma_bar(p) = p Sigma1 + (1-p) Sigma2
        and
            mean = -(x-omega) * Sigma_bar(p)^{-1} * signal(p, dlnf)
            cov  = (Lambda/2) * f(t,p) * Sigma_bar(p)^{-1}

        This keeps the Gaussian family and the quadratic-in-x value ansatz,
        while replacing the exact common-Sigma closed form by a
        belief-averaged-covariance approximation.
    """

    def __init__(self, n_assets: int = 2, signal1_init=None, signal2_init=None):
        super().__init__()
        self.n_assets = int(n_assets)
        if signal1_init is None:
            signal1_init = torch.zeros((self.n_assets,), dtype=torch.float64)
        if signal2_init is None:
            signal2_init = torch.zeros((self.n_assets,), dtype=torch.float64)
        self.signal1 = nn.Parameter(torch.as_tensor(signal1_init, dtype=torch.float64).clone())
        self.signal2 = nn.Parameter(torch.as_tensor(signal2_init, dtype=torch.float64).clone())

    def _effective_covariance(
        self,
        p: torch.Tensor,
        Sigma: torch.Tensor | None,
        Sigma1: torch.Tensor | None,
        Sigma2: torch.Tensor | None,
        cov_mode: str,
    ) -> torch.Tensor:
        if cov_mode == "common":
            if Sigma is None:
                raise ValueError("cov_mode='common' requires Sigma.")
            if p.ndim == 0:
                p = p.unsqueeze(0)
            B = p.shape[0]
            return Sigma.view(1, self.n_assets, self.n_assets).expand(B, -1, -1)
        if cov_mode == "belief_avg_regime":
            if Sigma1 is None or Sigma2 is None:
                raise ValueError("cov_mode='belief_avg_regime' requires Sigma1 and Sigma2.")
            return belief_avg_covariance(p, Sigma1, Sigma2)
        raise ValueError(f"Unknown cov_mode={cov_mode}")

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
        Sigma: torch.Tensor | None,
        Sigma1: torch.Tensor | None,
        Sigma2: torch.Tensor | None,
        r: float,
        cov_mode: str = "common",
    ) -> torch.distributions.MultivariateNormal:
        if x.ndim == 0:
            x = x.unsqueeze(0)
        if p.ndim == 0:
            p = p.unsqueeze(0)
        if dlnf_dp.ndim == 0:
            dlnf_dp = dlnf_dp.unsqueeze(0)
        if f.ndim == 0:
            f = f.unsqueeze(0)

        device = x.device
        dtype = x.dtype
        eye = torch.eye(self.n_assets, dtype=dtype, device=device).unsqueeze(0)

        Sigma_eff = self._effective_covariance(
            p=p,
            Sigma=None if Sigma is None else torch.as_tensor(Sigma, dtype=dtype, device=device),
            Sigma1=None if Sigma1 is None else torch.as_tensor(Sigma1, dtype=dtype, device=device),
            Sigma2=None if Sigma2 is None else torch.as_tensor(Sigma2, dtype=dtype, device=device),
            cov_mode=cov_mode,
        )
        Sigma_eff = 0.5 * (Sigma_eff + Sigma_eff.transpose(-1, -2)) + 1e-10 * eye
        chol = torch.linalg.cholesky(Sigma_eff)
        sigma_inv = torch.cholesky_inverse(chol)

        # Belief-adjusted signal inherited from the common-Sigma derivation:
        #   p - p(1-p) d ln f / dp
        adj = p - dlnf_dp * p * (1.0 - p)
        sig1 = self.signal1.view(1, self.n_assets)
        sig2 = self.signal2.view(1, self.n_assets)
        signal = sig2 + (sig1 - sig2) * adj.unsqueeze(-1)

        if cov_mode == "common":
            sigma_invhalf_signal = torch.linalg.solve_triangular(
                chol.transpose(-1, -2), signal.unsqueeze(-1), upper=True
            ).squeeze(-1)
            mean = -(x - omega).unsqueeze(-1) * sigma_invhalf_signal
        else:
            # Practical generalized form:
            #   mean = -(x-omega) * Sigma_bar(p)^{-1} * signal(p, dlnf)
            mean = -(x - omega).unsqueeze(-1) * torch.matmul(
                sigma_inv, signal.unsqueeze(-1)
            ).squeeze(-1)

        lam = torch.as_tensor(Lambda, dtype=dtype, device=device)
        cov_scale_t = torch.as_tensor(cov_scale, dtype=dtype, device=device)
        cov = cov_scale_t * 0.5 * lam * f.view(-1, 1, 1) * sigma_inv
        cov = 0.5 * (cov + cov.transpose(-1, -2)) + 1e-10 * eye
        return torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)


def value_fn(
    t: torch.Tensor,
    x: torch.Tensor,
    p: torch.Tensor,
    omega: torch.Tensor,
    z: float,
    vf: PolyValue,
):
    f = vf.f(t, p)
    g = vf.g(t, p)
    V = (x - omega) ** 2 / f + g - (omega - z) ** 2
    return V, f
