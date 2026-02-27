from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .utils import safe_clip_p

@dataclass
class FilterParams:
    mu1: np.ndarray        # (2,)
    mu2: np.ndarray        # (2,)
    Sigma1: np.ndarray     # (2,2)
    Sigma2: np.ndarray     # (2,2)
    lam1: float
    lam2: float
    r: float = 0.0

def _mvn_logpdf(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    mean = np.asarray(mean, dtype=float).reshape(-1)
    cov = np.asarray(cov, dtype=float)
    d = y.shape[0]
    # stable logpdf
    cov = cov + 1e-12*np.eye(d)
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, (y - mean))
    log_det = 2.0*np.sum(np.log(np.diag(L)))
    return -0.5*(d*np.log(2*np.pi) + log_det + np.dot(z, z))

def wonham_filter_q_update(p_prev: float, log_return: np.ndarray, dt: float, fp: FilterParams):
    """2-asset discrete filter update (Bayes + CTMC transition).

    We use:
      p^- = p(1-lam1 dt) + (1-p) lam2 dt
      y | I=i ~ N(m_i, V_i) with m_i=(mu_i-0.5 diag(Sigma_i))dt, V_i=Sigma_i dt
      p^+ ∝ p^- * L1, (1-p^-) * L2
    Returns (p_next, innovation_dummy)
    """
    p_prev = safe_clip_p(p_prev)
    y = np.asarray(log_return, dtype=float).reshape(2,)

    # predict via CTMC transition (first-order)
    p_pred = p_prev*(1.0 - fp.lam1*dt) + (1.0 - p_prev)*(fp.lam2*dt)
    p_pred = safe_clip_p(p_pred)

    mu1 = np.asarray(fp.mu1, dtype=float).reshape(2,)
    mu2 = np.asarray(fp.mu2, dtype=float).reshape(2,)
    S1 = np.asarray(fp.Sigma1, dtype=float)
    S2 = np.asarray(fp.Sigma2, dtype=float)

    m1 = (mu1 - 0.5*np.diag(S1)) * dt
    m2 = (mu2 - 0.5*np.diag(S2)) * dt
    V1 = S1 * dt
    V2 = S2 * dt

    ll1 = _mvn_logpdf(y, m1, V1)
    ll2 = _mvn_logpdf(y, m2, V2)
    # posterior
    a = np.log(p_pred) + ll1
    b = np.log(1.0 - p_pred) + ll2
    # log-sum-exp
    mx = max(a, b)
    denom = mx + np.log(np.exp(a-mx) + np.exp(b-mx))
    p_next = np.exp(a - denom)
    return safe_clip_p(float(p_next)), 0.0

def _logpdf_mvn(y: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Stable log N(y; mean, cov) for 2D."""
    y = np.asarray(y, dtype=float).reshape(2,)
    mean = np.asarray(mean, dtype=float).reshape(2,)
    cov = np.asarray(cov, dtype=float).reshape(2,2)
    # add tiny jitter for numerical stability
    cov = cov + 1e-12*np.eye(2)
    L = np.linalg.cholesky(cov)
    z = np.linalg.solve(L, y - mean)
    quad = float(z.T @ z)
    logdet = 2.0 * float(np.log(np.diag(L)).sum())
    return -0.5*(2*np.log(2*np.pi) + logdet + quad)