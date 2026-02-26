from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .utils import annualize_mean, annualize_std

@dataclass
class HeuristicThresholds:
    Y_l: float = 0.19  # bear threshold (decline)
    Y_u: float = 0.24  # bull threshold (increase)

def label_bull_bear_from_drawdowns(S: np.ndarray, thr: HeuristicThresholds):
    """Heuristic used in Dai et al. (2010, 2016): identify bull/bear using % moves from local extrema.

    This is a simple and transparent implementation:
    - Track current 'trend' (bull or bear) and the running extreme (peak for bull, trough for bear).
    - Switch to bear if price falls >= Y_l from the last peak.
    - Switch to bull if price rises >= Y_u from the last trough.

    Returns:
      regime[k] in {1(bull),2(bear)} aligned with S[k].
    """
    n = len(S)
    reg = np.ones(n, dtype=int)  # start as bull
    peak = S[0]
    trough = S[0]
    state = 1
    for k in range(1, n):
        price = S[k]
        if state == 1:
            peak = max(peak, price)
            if price <= peak * (1.0 - thr.Y_l):
                state = 2
                trough = price
        else:
            trough = min(trough, price)
            if price >= trough * (1.0 + thr.Y_u):
                state = 1
                peak = price
        reg[k] = state
    return reg

def estimate_env_params_from_labeled_returns(returns: np.ndarray, reg: np.ndarray, dt: float):
    """Estimate mu_i and lambda_i from labeled regimes; sigma from all returns.

    - mu_i: annualized mean of returns in regime i divided by dt (approx)
    - sigma: annualized std of returns / sqrt(dt) (approx)
    - lambda_i: reciprocal of average time spent in regime i (years)
    """
    # log-returns are more natural; for small dt either works. We'll use log(1+r) if needed.
    daily = returns
    days_per_year = int(round(1/dt))
    # mu approx from E[dS/S]/dt
    mu1 = annualize_mean(daily[reg[:-1]==1].mean(), days_per_year)
    mu2 = annualize_mean(daily[reg[:-1]==2].mean(), days_per_year)
    sigma = annualize_std(daily.std(ddof=1), days_per_year)

    # estimate average sojourn time in each regime (in years)
    durations = {1: [], 2: []}
    cur = reg[0]; run = 1
    for k in range(1, len(reg)):
        if reg[k] == cur:
            run += 1
        else:
            durations[cur].append(run * dt)
            cur = reg[k]
            run = 1
    durations[cur].append(run * dt)
    avg1 = np.mean(durations[1]) if len(durations[1]) else dt
    avg2 = np.mean(durations[2]) if len(durations[2]) else dt
    lam1 = 1.0 / avg1  # 1->2 intensity approx reciprocal of time in 1
    lam2 = 1.0 / avg2  # 2->1 intensity approx reciprocal of time in 2
    return dict(mu1=float(mu1), mu2=float(mu2), sigma=float(sigma), lam1=float(lam1), lam2=float(lam2))
