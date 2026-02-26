from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .utils import safe_clip_p

@dataclass
class FilterParams:
    mu1: float
    mu2: float
    sigma: float
    lam1: float
    lam2: float
    r: float = 0.0

def wonham_filter_q_update(p_prev: float, log_return: float, dt: float, fp: FilterParams):
    """Discrete Wonham filter using q-transform:
      q1 = log p, q2 = log(1-p)
    Update follows the paper's Euler scheme in q-space to keep p in (0,1).
    """
    p_prev = safe_clip_p(p_prev)
    q1 = np.log(p_prev)
    q2 = np.log(1.0 - p_prev)

    # rho_i = (mu_i - r)/sigma
    rho1 = (fp.mu1 - fp.r) / fp.sigma
    rho2 = (fp.mu2 - fp.r) / fp.sigma

    # estimated mu_hat(p) and sigma_tilde
    mu_hat = fp.mu2 + (fp.mu1 - fp.mu2) * p_prev
    sigma_tilde = fp.sigma

    # innovation increment Δ B_hat
    dB_hat = (log_return - (mu_hat - 0.5*sigma_tilde*sigma_tilde)*dt) / sigma_tilde

    # drift terms for q updates (derived from the filter SDE)
    # Using the paper's discretization (q-space Euler).
    # Note: p_prev used in denominators; safe-clipped.
    q1_next = q1 + (fp.lam2*(1-p_prev)/p_prev - fp.lam1 - 0.5*(rho1 - (rho1-rho2)*p_prev)**2)*dt                   + (rho1 - (rho1-rho2)*p_prev) * dB_hat
    q2_next = q2 + (fp.lam1*p_prev/(1-p_prev) - fp.lam2 - 0.5*(rho2 + (rho1-rho2)*p_prev)**2)*dt                   + (rho2 + (rho1-rho2)*p_prev) * dB_hat

    # back-transform
    a = np.exp(q1_next); b = np.exp(q2_next)
    p_next = a/(a+b)
    return safe_clip_p(p_next), dB_hat
