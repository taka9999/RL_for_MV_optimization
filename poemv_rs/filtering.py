from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .utils import safe_clip_p

@dataclass
class FilterParams:
    mu1: np.ndarray        # (2,)
    mu2: np.ndarray        # (2,)
    Sigma: np.ndarray      # (2,2), common covariance
    lam1: float
    lam2: float
    r: float = 0.01


def wonham_filter_q_update(p_prev: float, log_return: np.ndarray, dt: float, fp: FilterParams):
    """Discretized two-state Wonham/Bayes filter update with common Sigma.

    This version updates two log-unnormalized regime scores separately:
        q1 = log(alpha1), q2 = log(alpha2),
        p = alpha1 / (alpha1 + alpha2),
    rather than updating a single logit q = log(p/(1-p)).
 
    We use:
      1) first-order CTMC prediction on probabilities
      2) Gaussian observation likelihood under each regime
      3) normalization in log-space for numerical stability

    Observation model for one step:
        dlogS ≈ (mu_i - 0.5 diag(Sigma)) dt + L eps,
        eps ~ N(0, dt I),  Sigma = L L^T.

    Returns
    -------
    p_next : float
        Updated belief P(Y_{t+dt}=1 | F_{t+dt}^S).
    innovation_scalar : float
        Scalar diagnostic innovation along the paper-style beta direction.
    """
    p_prev = safe_clip_p(p_prev)
    y = np.asarray(log_return, dtype=float).reshape(2,)

    # CTMC prediction (first-order)
    p_pred = p_prev*(1.0 - fp.lam1*dt) + (1.0 - p_prev)*(fp.lam2*dt)
    p_pred = safe_clip_p(p_pred)

    mu1 = np.asarray(fp.mu1, dtype=float).reshape(2,)
    mu2 = np.asarray(fp.mu2, dtype=float).reshape(2,)
    Sigma = np.asarray(fp.Sigma, dtype=float).reshape(2, 2)
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-12*np.eye(2)
    L = np.linalg.cholesky(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)

    #muhat = p_pred * mu1 + (1.0 - p_pred) * mu2
    #drift_obs = (muhat - 0.5*np.diag(Sigma)) * dt
    # 2) regime-wise Gaussian observation likelihood for dlogS
    drift1 = (mu1 - 0.5*np.diag(Sigma)) * dt
    drift2 = (mu2 - 0.5*np.diag(Sigma)) * dt

    err1 = y - drift1
    err2 = y - drift2

    # log-likelihood up to an additive common constant
    ll1 = -0.5 * float(err1 @ Sigma_inv @ err1) / max(dt, 1e-12)
    ll2 = -0.5 * float(err2 @ Sigma_inv @ err2) / max(dt, 1e-12)

    # 3) q1, q2 update in log-space
    q1_next = np.log(p_pred) + ll1
    q2_next = np.log(1.0 - p_pred) + ll2

    # normalize stably
    q_max = max(q1_next, q2_next)
    a1 = np.exp(q1_next - q_max)
    a2 = np.exp(q2_next - q_max)
    p_next = a1 / (a1 + a2)

    # diagnostic innovation in the beta direction
    muhat = p_pred * mu1 + (1.0 - p_pred) * mu2
    drift_obs = (muhat - 0.5 * np.diag(Sigma)) * dt
    dBhat = np.linalg.solve(L, (y - drift_obs))

    beta = np.linalg.solve(L, (mu1 - mu2))
    #beta2 = float(beta @ beta)

    #q_pred = np.log(p_pred) - np.log(1.0 - p_pred)
    #drift_p = (-(fp.lam1 + fp.lam2) * p_pred + fp.lam2)
    #drift_q = drift_p / max(p_pred * (1.0 - p_pred), 1e-12) - 0.5 * (1.0 - 2.0 * p_pred) * beta2
    #q_next = q_pred + drift_q * dt + float(beta @ dBhat)

    #p_next = 1.0 / (1.0 + np.exp(-q_next))
    return safe_clip_p(float(p_next)), float(beta @ dBhat)
