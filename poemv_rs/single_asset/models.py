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
    """Gaussian policy with scalar parameters phi1, phi2, phi3."""
    def __init__(self):
        super().__init__()
        self.phi = nn.Parameter(torch.zeros((3,), dtype=torch.float64))

    def dist_params(self, x: torch.Tensor, omega: torch.Tensor, p: torch.Tensor, dlnf_dp: torch.Tensor,
                    f: torch.Tensor, Lambda: float,sigma: float):
        phi1, phi2, phi3 = self.phi[0], self.phi[1], self.phi[2]
        inv_sigma2 = 1.0 / (sigma * sigma)
        # mean = -(x-omega) * [phi1*(p - (f_p/f) p(1-p)) + phi2]
        adj = p - dlnf_dp * p * (1.0 - p)
        #mean = -(x - omega) * (phi1 * adj + phi2)
        #var = torch.as_tensor(Lambda, dtype=torch.float64) * torch.exp(phi3) * f
        mean = inv_sigma2 * (-(x - omega) * (phi1 * adj + phi2))
        var = inv_sigma2 * (torch.as_tensor(Lambda, dtype=x.dtype, device=x.device) * torch.exp(phi3) * f)
        var = var.clamp(min=1e-24, max=100.0)   # std in [1e-12, 10]
        std = torch.sqrt(var)
        return mean, std

def value_fn(t: torch.Tensor, x: torch.Tensor, p: torch.Tensor, omega: torch.Tensor, z: float,
             vf: PolyValue):
    f = vf.f(t, p)
    g = vf.g(t, p)
    V = (x - omega)**2 / f + g - (omega - z)**2
    return V, f
