from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from .utils import safe_clip_p


@dataclass
class FilterParamsRegimeCov:
    mu1: np.ndarray
    mu2: np.ndarray
    Sigma1: np.ndarray
    Sigma2: np.ndarray
    lam1: float
    lam2: float
    r: float = 0.01


def hmm_filter_regime_cov_q_update(
    p_prev: float,
    log_return: np.ndarray,
    dt: float,
    fp: FilterParamsRegimeCov,
) -> tuple[float, float]:
    """Two-state HMM/Bayes filter update with regime-dependent Gaussian covariance.

    Observation model:
        y_k = d log S_k | I_k=i ~ N(m_i, Sigma_i * dt)
        m_i = (mu_i - 0.5 diag(Sigma_i)) dt

    Returns
    -------
    p_next : float
        Updated posterior P(I_k = 1 | y_{1:k}).
    innovation_scalar : float
        Diagnostic scalar whitening the residual by the belief-averaged covariance.
        This is not the exact Wonham innovation; it is a practical diagnostic.
    """
    p_prev = safe_clip_p(float(p_prev))
    y = np.asarray(log_return, dtype=float).reshape(-1)

    # CTMC one-step prediction
    p_pred = p_prev * (1.0 - fp.lam1 * dt) + (1.0 - p_prev) * (fp.lam2 * dt)
    p_pred = safe_clip_p(float(p_pred))

    mu1 = np.asarray(fp.mu1, dtype=float).reshape(-1)
    mu2 = np.asarray(fp.mu2, dtype=float).reshape(-1)
    Sigma1 = np.asarray(fp.Sigma1, dtype=float)
    Sigma2 = np.asarray(fp.Sigma2, dtype=float)
    d = y.shape[0]
    eye = np.eye(d)
    Sigma1 = 0.5 * (Sigma1 + Sigma1.T) + 1e-12 * eye
    Sigma2 = 0.5 * (Sigma2 + Sigma2.T) + 1e-12 * eye

    drift1 = (mu1 - 0.5 * np.diag(Sigma1)) * dt
    drift2 = (mu2 - 0.5 * np.diag(Sigma2)) * dt

    err1 = y - drift1
    err2 = y - drift2

    Sigma1_inv = np.linalg.inv(Sigma1)
    Sigma2_inv = np.linalg.inv(Sigma2)
    _, logdet1 = np.linalg.slogdet(Sigma1)
    _, logdet2 = np.linalg.slogdet(Sigma2)

    ll1 = -0.5 * logdet1 - 0.5 * float(err1 @ Sigma1_inv @ err1) / max(dt, 1e-12)
    ll2 = -0.5 * logdet2 - 0.5 * float(err2 @ Sigma2_inv @ err2) / max(dt, 1e-12)

    q1 = np.log(p_pred) + ll1
    q2 = np.log(1.0 - p_pred) + ll2
    qmax = max(q1, q2)
    a1 = np.exp(q1 - qmax)
    a2 = np.exp(q2 - qmax)
    p_next = float(a1 / (a1 + a2))
    p_next = safe_clip_p(p_next)

    # practical diagnostic innovation under belief-averaged covariance
    Sigma_bar = p_pred * Sigma1 + (1.0 - p_pred) * Sigma2
    Sigma_bar = 0.5 * (Sigma_bar + Sigma_bar.T) + 1e-12 * eye
    Lbar = np.linalg.cholesky(Sigma_bar)
    muhat = p_pred * mu1 + (1.0 - p_pred) * mu2
    drift_bar = (muhat - 0.5 * np.diag(Sigma_bar)) * dt
    whitened = np.linalg.solve(Lbar, y - drift_bar)
    innovation_scalar = float(np.linalg.norm(whitened))
    return p_next, innovation_scalar
