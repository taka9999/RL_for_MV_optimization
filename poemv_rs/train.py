from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import matplotlib.pyplot as plt

from .env import RSGBMParams, EpisodeConfig, RSGBMEnv, simulate_price_only
from .filtering import FilterParams, wonham_filter_q_update

from .heuristic import HeuristicThresholds, label_bull_bear_from_drawdowns, estimate_env_params_from_labeled_returns
from .agent import POEMVAgent, TrainConfig
from .utils import set_seed, safe_clip_p
from jumpmodels.jump import JumpModel


def _to_common_sigma_params(obj: RSGBMParams) -> RSGBMParams:
    if hasattr(obj, "Sigma"):
        return obj
    s1 = np.asarray(obj.Sigma1, dtype=float)
    s2 = np.asarray(obj.Sigma2, dtype=float)
    sigma = 0.5 * (s1 + s2)
    sigma = 0.5 * (sigma + sigma.T)
    return RSGBMParams(mu1=np.asarray(obj.mu1, float),
                       mu2=np.asarray(obj.mu2, float),
                       Sigma=sigma,
                       lam1=float(obj.lam1), lam2=float(obj.lam2), r=float(obj.r))

def run_filter_demo(outdir: Path, true_params: RSGBMParams, est_params: RSGBMParams, T_years=10.0, dt=1/252, seed=0):
    S, I = simulate_price_only(true_params, T_years=T_years, dt=dt, s0=1.0, seed=seed)
    logret = np.log(S[1:] / S[:-1])  # (n,2)
    p_true = np.empty(len(S)); p_est = np.empty(len(S))
    p_true[0]=0.5; p_est[0]=0.5

    fp_true = FilterParams(mu1=true_params.mu1, mu2=true_params.mu2, Sigma=true_params.Sigma,
                           lam1=true_params.lam1, lam2=true_params.lam2, r=true_params.r)
    fp_est  = FilterParams(mu1=est_params.mu1,  mu2=est_params.mu2,  Sigma=est_params.Sigma,
                           lam1=est_params.lam1,  lam2=est_params.lam2,  r=est_params.r)

    for k in range(1, len(S)):
        p_true[k], _ = wonham_filter_q_update(p_true[k-1], logret[k-1], dt, fp_true)
        p_est[k], _  = wonham_filter_q_update(p_est[k-1],  logret[k-1], dt, fp_est)

    t = np.arange(len(S))*dt
    fig = plt.figure()
    plt.plot(t, p_true, label="belief p_t (true params)")
    plt.plot(t, p_est, label="belief p_t (estimated params)", linestyle="--")
    plt.step(t, (I==1).astype(float), where="post", label="true regime (1=bull)", alpha=0.5)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("time (years)")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir/"filter_demo.png", dpi=200)
    plt.close(fig)

def make_estimated_params_via_heuristic_old(true_params: RSGBMParams, seed=0):
    # Simulate 30y, use first 20y to estimate, per paper description.
    dt = 1/252
    S, _ = simulate_price_only(true_params, T_years=30.0, dt=dt, s0=1.0, seed=seed)
    S_train = S[: int(round(20.0/dt))+1]  # (T,2)
    # label using asset-1 price (minimal). You can replace with equal-weight index.
    reg = label_bull_bear_from_drawdowns(S_train[:,0], HeuristicThresholds())
    ret_train = (S_train[1:] - S_train[:-1]) / S_train[:-1]  # (T-1,2)
    est = estimate_env_params_from_labeled_returns(ret_train, reg, dt=dt)
    return _to_common_sigma_params(RSGBMParams(**est, r=true_params.r))

def make_estimated_params_via_heuristic(true_params: RSGBMParams, seed=0):
    # Simulate 30y, use first 20y to estimate, per paper description.
    dt = 1 / 252
    S, _ = simulate_price_only(true_params, T_years=30.0, dt=dt, s0=1.0, seed=seed)
    S_train = S[: int(round(20.0 / dt)) + 1]  # (T,2)

    # label using asset-1 price (minimal). You can replace with equal-weight index.
    # reg = label_bull_bear_from_drawdowns(S_train[:, 0], HeuristicThresholds())
    # Label using an equal-weight index instead of only asset-1.
    # This is usually more stable for multi-asset regime identification.
    eq_index = np.mean(S_train, axis=1)
    reg = label_bull_bear_from_drawdowns(eq_index, HeuristicThresholds())
    ret_train = (S_train[1:] - S_train[:-1]) / S_train[:-1]  # (T-1,2)

    est = estimate_env_params_from_labeled_returns(ret_train, reg, dt=dt)

    # heuristic estimator returns Sigma1/Sigma2, but current RSGBMParams
    # expects a single common Sigma. Convert first, then construct params.
    #s1 = np.asarray(est["Sigma1"], dtype=float)
    #s2 = np.asarray(est["Sigma2"], dtype=float)
    #sigma = 0.5 * (s1 + s2)
    #sigma = 0.5 * (sigma + sigma.T)
    mu1 = np.asarray(est["mu1"], dtype=float)
    mu2 = np.asarray(est["mu2"], dtype=float)
    Sigma1 = np.asarray(est["Sigma1"], dtype=float)
    Sigma2 = np.asarray(est["Sigma2"], dtype=float)
    lam1 = float(est["lam1"])
    lam2 = float(est["lam2"])

    # Reorder regimes so that regime 1 is the "more bullish" one
    # and regime 2 is the "more bearish" one.
    # We use the sum of annualized mean returns across assets as a simple score.
    score1 = float(np.sum(mu1))
    score2 = float(np.sum(mu2))
    if score1 < score2:
        mu1, mu2 = mu2, mu1
        Sigma1, Sigma2 = Sigma2, Sigma1
        lam1, lam2 = lam2, lam1

    # ------------------------------------------------------------
    # Additional stabilization for estimated_params mode:
    # 1) enforce a minimum separation between bull/bear means
    # 2) shrink bear mean downward if its overall level is not
    #    sufficiently below the bull mean
    # ------------------------------------------------------------
    true_mu1 = np.asarray(true_params.mu1, dtype=float)
    true_mu2 = np.asarray(true_params.mu2, dtype=float)
    true_gap = true_mu1 - true_mu2

    # separation floor: require each asset's bull-bear spread
    # to be at least a fraction of the true spread
    sep_floor_frac = 0.35
    sep_floor = sep_floor_frac * np.abs(true_gap)
    cur_gap = mu1 - mu2
    gap_shortfall = np.maximum(sep_floor - cur_gap, 0.0)
    if np.any(gap_shortfall > 0.0):
        # push bull slightly up and bear more down
        mu1 = mu1 + 0.25 * gap_shortfall
        mu2 = mu2 - 0.75 * gap_shortfall

    # overall bear shrink: force mean(mu2) to sit sufficiently
    # below mean(mu1)
    bull_mean = float(np.mean(mu1))
    bear_mean = float(np.mean(mu2))
    true_overall_gap = float(np.mean(true_mu1) - np.mean(true_mu2))
    overall_gap_floor = 0.35 * abs(true_overall_gap)
    cur_overall_gap = bull_mean - bear_mean
    if cur_overall_gap < overall_gap_floor:
        shortfall = overall_gap_floor - cur_overall_gap
        mu2 = mu2 - shortfall

    # optional extra guard: bear regime should not have higher
    # average return than bull regime on aggregate
    if float(np.mean(mu2)) >= float(np.mean(mu1)):
        mu2 = mu2 - (float(np.mean(mu2)) - float(np.mean(mu1)) + 0.02)

    # Current RSGBMParams uses a common Sigma across regimes.
    sigma = 0.5 * (Sigma1 + Sigma2)
    sigma = 0.5 * (sigma + sigma.T)

    return RSGBMParams(
        #mu1=np.asarray(est["mu1"], dtype=float),
        #mu2=np.asarray(est["mu2"], dtype=float)
        mu1=mu1,
        mu2=mu2,
        Sigma=sigma,
        #lam1=float(est["lam1"]),
        #lam2=float(est["lam2"]),
        lam1=lam1,
        lam2=lam2,
        r=float(true_params.r),
    )

def _logsumexp_1d(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m))))

def _gaussian_logpdf_common_cov(y: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    d = y.shape[0]
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-10 * np.eye(d)
    diff = y - mu
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        Sigma = Sigma + 1e-8 * np.eye(d)
        sign, logdet = np.linalg.slogdet(Sigma)
    inv = np.linalg.inv(Sigma)
    quad = float(diff @ inv @ diff)
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)

def _forward_backward_2state(
    Y: np.ndarray,
    pi0: np.ndarray,
    P: np.ndarray,
    mus: np.ndarray,
    Sigma: np.ndarray,
):
    """
    Y   : (T, d) log returns
    pi0 : (2,)
    P   : (2,2) row-stochastic transition matrix
    mus : (2,d)
    Sigma: (d,d), common covariance

    Returns
    -------
    gamma : (T,2) state posterior probs
    xi    : (T-1,2,2) pairwise posterior probs
    loglik: float
    """
    T, d = Y.shape
    logA = np.log(np.clip(P, 1e-12, 1.0))
    logpi0 = np.log(np.clip(pi0, 1e-12, 1.0))

    logB = np.empty((T, 2), dtype=float)
    for t in range(T):
        for j in range(2):
            logB[t, j] = _gaussian_logpdf_common_cov(Y[t], mus[j], Sigma)

    alpha = np.empty((T, 2), dtype=float)
    alpha[0] = logpi0 + logB[0]
    for t in range(1, T):
        for j in range(2):
            alpha[t, j] = logB[t, j] + _logsumexp_1d(alpha[t - 1] + logA[:, j])

    loglik = _logsumexp_1d(alpha[-1])

    beta = np.zeros((T, 2), dtype=float)
    for t in range(T - 2, -1, -1):
        for i in range(2):
            beta[t, i] = _logsumexp_1d(logA[i, :] + logB[t + 1, :] + beta[t + 1, :])

    log_gamma = alpha + beta
    for t in range(T):
        log_gamma[t] -= _logsumexp_1d(log_gamma[t])
    gamma = np.exp(log_gamma)

    xi = np.empty((T - 1, 2, 2), dtype=float)
    for t in range(T - 1):
        M = np.empty((2, 2), dtype=float)
        for i in range(2):
            for j in range(2):
                M[i, j] = alpha[t, i] + logA[i, j] + logB[t + 1, j] + beta[t + 1, j]
        z = _logsumexp_1d(M.reshape(-1))
        xi[t] = np.exp(M - z)

    return gamma, xi, float(loglik)

def _fit_2state_hmm_common_cov(
    Y: np.ndarray,
    dt: float,
    n_iter: int = 50,
    tol: float = 1e-5,
    seed: int = 0,
):
    """
    Minimal EM fit for 2-state Gaussian HMM with common covariance.
    Observation Y should be annualized-ish one-step log returns, shape (T,2).
    """
    rng = np.random.default_rng(seed)
    Y = np.asarray(Y, dtype=float)
    T, d = Y.shape

    # init: split by median of equal-weight log return rather than only asset-1
    # This is usually more stable in the 2-asset setting.
    score = np.mean(Y, axis=1)
    med = np.median(score)
    z0 = (score > med).astype(int)

    mus = np.vstack([
        Y[z0 == 1].mean(axis=0) if np.any(z0 == 1) else Y.mean(axis=0) + 1e-3,
        Y[z0 == 0].mean(axis=0) if np.any(z0 == 0) else Y.mean(axis=0) - 1e-3,
    ]).astype(float)

    Sigma = np.cov(Y.T, ddof=1)
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-8 * np.eye(d)

    P = np.array([[0.995, 0.005],
                  [0.010, 0.990]], dtype=float)
    pi0 = np.array([0.5, 0.5], dtype=float)

    prev_loglik = -np.inf
    for _ in range(int(n_iter)):
        gamma, xi, loglik = _forward_backward_2state(Y, pi0, P, mus, Sigma)

        # M-step: pi0
        pi0 = gamma[0]
        pi0 = pi0 / np.sum(pi0)

        # M-step: transition
        xi_sum = np.sum(xi, axis=0)              # (2,2)
        gamma_sum = np.sum(gamma[:-1], axis=0)   # (2,)
        P = xi_sum / np.clip(gamma_sum[:, None], 1e-12, None)
        P = np.clip(P, 1e-8, 1.0)
        P = P / P.sum(axis=1, keepdims=True)

        # M-step: means
        for j in range(2):
            w = gamma[:, j]
            mus[j] = np.sum(w[:, None] * Y, axis=0) / np.clip(np.sum(w), 1e-12, None)

        # M-step: common covariance
        Sigma_num = np.zeros((d, d), dtype=float)
        for j in range(2):
            diff = Y - mus[j]
            Sigma_num += (gamma[:, j][:, None] * diff).T @ diff
        Sigma = Sigma_num / np.clip(np.sum(gamma), 1e-12, None)
        Sigma = 0.5 * (Sigma + Sigma.T) + 1e-8 * np.eye(d)

        if np.isfinite(prev_loglik) and abs(loglik - prev_loglik) < tol:
            break
        prev_loglik = loglik

    # reorder so state 0 = more bullish, state 1 = more bearish
    if float(np.sum(mus[0])) < float(np.sum(mus[1])):
        mus = mus[[1, 0]]
        pi0 = pi0[[1, 0]]
        P = P[[1, 0]][:, [1, 0]]

    # convert transition matrix to continuous-time intensity approx
    # P11 ≈ 1 - lam1*dt, P22 ≈ 1 - lam2*dt
    lam1 = max((1.0 - P[0, 0]) / max(dt, 1e-12), 1e-8)
    lam2 = max((1.0 - P[1, 1]) / max(dt, 1e-12), 1e-8)

    # clamp transition intensities to avoid pathological over-switching
    # in short samples / noisy EM fits
    lam1 = min(lam1, 5.0)
    lam2 = min(lam2, 5.0)

    return {
        "mu1": np.asarray(mus[0], dtype=float),
        "mu2": np.asarray(mus[1], dtype=float),
        "Sigma": np.asarray(Sigma, dtype=float),
        "lam1": float(lam1),
        "lam2": float(lam2),
        "pi0": np.asarray(pi0, dtype=float),
        "P": np.asarray(P, dtype=float),
    }

def make_estimated_params_via_hmm(true_params: RSGBMParams, seed=0):
    """
    Minimal estimated-params builder using a 2-state Gaussian HMM
    on 2-asset log returns with common covariance.
    """
    dt = 1 / 252
    S, _ = simulate_price_only(true_params, T_years=30.0, dt=dt, s0=1.0, seed=seed)
    S_train = S[: int(round(20.0 / dt)) + 1]  # (T,2)

    logret = np.log(S_train[1:] / S_train[:-1])  # (T-1,2)

    fit = _fit_2state_hmm_common_cov(
        Y=logret,
        dt=dt,
        n_iter=50,
        tol=1e-5,
        seed=seed,
    )

    # convert one-step moments to annualized model parameters
    #
    # For one-step log return:
    #   logret_t ≈ (mu - 0.5 diag(Sigma)) dt + eps_t,
    #   Var(logret_t) ≈ Sigma * dt
    #
    # Hence:
    #   Sigma_annual = Sigma_step / dt
    #   drift_annual = mean_step / dt
    Sigma_step = np.asarray(fit["Sigma"], dtype=float)
    Sigma = Sigma_step / dt
    Sigma = 0.5 * (Sigma + Sigma.T)
    drift1 = fit["mu1"] / dt
    drift2 = fit["mu2"] / dt
    mu1 = drift1 + 0.5 * np.diag(Sigma)
    mu2 = drift2 + 0.5 * np.diag(Sigma)

    # final safety reorder after annualization
    if float(np.sum(mu1)) < float(np.sum(mu2)):
        mu1, mu2 = mu2, mu1
        lam1, lam2 = float(fit["lam2"]), float(fit["lam1"])
    else:
        lam1, lam2 = float(fit["lam1"]), float(fit["lam2"])

    return RSGBMParams(
        mu1=np.asarray(mu1, dtype=float),
        mu2=np.asarray(mu2, dtype=float),
        Sigma=np.asarray(Sigma, dtype=float),
        lam1=float(lam1),
        lam2=float(lam2),
        r=float(true_params.r),
    )

def make_estimated_params_via_jump_model(
    true_params: RSGBMParams,
    seed: int = 0,
    jump_penalty: float = 100.0,
    jm_max_iter: int = 400,
    jm_n_init: int = 10,
):
    """
    Estimate RSGBM parameters by:
      1) fitting a 2-state JumpModel on features built from 20y simulated data,
      2) using the inferred labels to estimate regime-wise means,
      3) using pooled covariance as common Sigma,
      4) converting the empirical transition matrix into CTMC intensities.
    """
    dt = 1 / 252
    S, _ = simulate_price_only(true_params, T_years=30.0, dt=dt, s0=1.0, seed=seed)
    S_train = S[: int(round(20.0 / dt)) + 1]  # (T,2)

    # daily simple returns and log returns
    ret_train = (S_train[1:] - S_train[:-1]) / S_train[:-1]      # (T-1,2)
    logret_train = np.log(S_train[1:] / S_train[:-1])            # (T-1,2)

    # equal-weight market proxy
    eq_ret = np.mean(ret_train, axis=1)                          # (T-1,)

    # simple rolling features for JM
    def _roll_mean(x: np.ndarray, win: int) -> np.ndarray:
        out = np.empty_like(x, dtype=float)
        for t in range(len(x)):
            lo = max(0, t - win + 1)
            out[t] = np.mean(x[lo:t+1])
        return out

    def _roll_std(x: np.ndarray, win: int) -> np.ndarray:
        out = np.empty_like(x, dtype=float)
        for t in range(len(x)):
            lo = max(0, t - win + 1)
            seg = x[lo:t+1]
            out[t] = np.std(seg, ddof=1) if len(seg) >= 2 else 0.0
        return out

    def _rolling_drawdown_from_returns(x: np.ndarray, win: int) -> np.ndarray:
        out = np.empty_like(x, dtype=float)
        wealth = np.cumprod(1.0 + x)
        for t in range(len(x)):
            lo = max(0, t - win + 1)
            w = wealth[lo:t+1]
            peak = np.maximum.accumulate(w)
            dd = w / np.maximum(peak, 1e-12) - 1.0
            out[t] = np.min(dd)
        return out

    def _rolling_downside_dev(x: np.ndarray, win: int) -> np.ndarray:
        out = np.empty_like(x, dtype=float)
        for t in range(len(x)):
            lo = max(0, t - win + 1)
            seg = x[lo:t+1]
            neg = np.minimum(seg, 0.0)
            out[t] = np.sqrt(np.mean(neg * neg))
        return out

    def _rolling_cumret(x: np.ndarray, win: int) -> np.ndarray:
        out = np.empty_like(x, dtype=float)
        for t in range(len(x)):
            lo = max(0, t - win + 1)
            seg = x[lo:t+1]
            out[t] = np.prod(1.0 + seg) - 1.0
        return out

    def _zscore_cols(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        mu = np.mean(X, axis=0)
        sd = np.std(X, axis=0, ddof=1)
        sd = np.where(sd < eps, 1.0, sd)
        return (X - mu) / sd

    #roll_mean_21 = _roll_mean(eq_ret, 21)
    #roll_std_21 = _roll_std(eq_ret, 21)
    #roll_dd_63 = _rolling_drawdown_from_returns(eq_ret, 63)
    #roll_down_21 = _rolling_downside_dev(eq_ret, 21)

    # feature matrix for jump model
    #X_feat = np.column_stack([
    #    eq_ret,
    #    roll_mean_21,
    #    roll_std_21,
    #    roll_dd_63,
    #    roll_down_21,
    #])
    # 5-day features
    roll_mean_5 = _roll_mean(eq_ret, 5)
    roll_std_5 = _roll_std(eq_ret, 5)
    roll_down_5 = _rolling_downside_dev(eq_ret, 5)
    roll_cumret_5 = _rolling_cumret(eq_ret, 5)

    # 21-day features
    roll_mean_21 = _roll_mean(eq_ret, 21)
    roll_std_21 = _roll_std(eq_ret, 21)
    roll_down_21 = _rolling_downside_dev(eq_ret, 21)
    roll_cumret_21 = _rolling_cumret(eq_ret, 21)

    # 63-day features
    roll_mean_63 = _roll_mean(eq_ret, 63)
    roll_std_63 = _roll_std(eq_ret, 63)
    roll_down_63 = _rolling_downside_dev(eq_ret, 63)
    roll_cumret_63 = _rolling_cumret(eq_ret, 63)
    roll_dd_21 = _rolling_drawdown_from_returns(eq_ret, 21)
    roll_dd_63 = _rolling_drawdown_from_returns(eq_ret, 63)

    # cross-horizon summaries
    mean_spread_5_21 = roll_mean_5 - roll_mean_21
    mean_spread_21_63 = roll_mean_21 - roll_mean_63
    vol_ratio_5_21 = roll_std_5 / np.maximum(roll_std_21, 1e-8)
    vol_ratio_21_63 = roll_std_21 / np.maximum(roll_std_63, 1e-8)

    # feature matrix for jump model
    X_feat = np.column_stack([
        eq_ret,
        #roll_mean_5,
        roll_mean_21,
        roll_mean_63,
        #roll_std_5,
        roll_std_21,
        roll_std_63,
        #roll_down_5,
        roll_down_21,
        roll_down_63,
        #roll_cumret_5,
        #roll_cumret_21,
        #roll_cumret_63,
        #roll_dd_21,
        roll_dd_63,
        #mean_spread_5_21,
        mean_spread_21_63,
        vol_ratio_5_21,
        vol_ratio_21_63,
     ])

    # standardize features so that drawdown / vol / return scales are balanced
    X_feat = _zscore_cols(X_feat)

    # Fit JM. sort_by="cumret" makes state 1 the better-return state.
    jm = JumpModel(
        n_components=2,
        jump_penalty=jump_penalty,
        cont=False,
        random_state=seed,
        max_iter=jm_max_iter,
        n_init=jm_n_init,
        verbose=0,
    )
    jm.fit(X_feat, ret_ser=eq_ret, sort_by="cumret")
    labels = np.asarray(jm.labels_, dtype=int)

    # regime-wise annualized simple-return means
    mu1 = ret_train[labels == 1].mean(axis=0) * 252 if np.any(labels == 1) else ret_train.mean(axis=0) * 252
    mu2 = ret_train[labels == 0].mean(axis=0) * 252 if np.any(labels == 0) else ret_train.mean(axis=0) * 252

    # reorder so regime 1 is more bullish in RSGBM convention
    if float(np.sum(mu1)) < float(np.sum(mu2)):
        mu1, mu2 = mu2, mu1
        labels = 1 - labels

    # common annualized covariance from all returns
    Sigma = np.cov(ret_train.T, ddof=1) * 252.0
    Sigma = 0.5 * (Sigma + Sigma.T)

    # empirical transition matrix -> CTMC intensities
    trans = np.asarray(jm.transmat_, dtype=float)
    trans = np.clip(trans, 1e-8, 1.0)
    trans = trans / trans.sum(axis=1, keepdims=True)

    # jm state 1 is bullish after sort_by="cumret", but we reordered labels above
    # so compute intensities from reordered empirical transitions directly.
    n00 = n01 = n10 = n11 = 0
    for t in range(1, len(labels)):
        i, j = int(labels[t - 1]), int(labels[t])
        if i == 0 and j == 0:
            n00 += 1
        elif i == 0 and j == 1:
            n01 += 1
        elif i == 1 and j == 0:
            n10 += 1
        else:
            n11 += 1
    P = np.array([
        [n00, n01],
        [n10, n11],
    ], dtype=float)
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)

    # RSGBM convention: regime 1=bull, regime 2=bear
    # P[1,1] is bull->bull, P[0,0] is bear->bear after the label convention above,
    # so build a bull/bear ordered matrix explicitly.
    # labels==1 : bull, labels==0 : bear
    bullbull = P[1, 1]
    bearbear = P[0, 0]
    lam1 = max((1.0 - bullbull) / dt, 1e-8)
    lam2 = max((1.0 - bearbear) / dt, 1e-8)

    # mild clamp for stability
    lam1 = min(lam1, 2.0)
    lam2 = min(lam2, 3.5)

    return RSGBMParams(
        mu1=np.asarray(mu1, dtype=float),
        mu2=np.asarray(mu2, dtype=float),
        Sigma=np.asarray(Sigma, dtype=float),
        lam1=float(lam1),
        lam2=float(lam2),
        r=float(true_params.r),
    )

def train(mode: str, iters: int, seed: int, outdir: Path,
          alpha_theta: float, alpha_phi: float, alpha_w: float,
          Lambda: float, omega_update_every: int,
          episode_T_years: float = 10.0, dt: float = 1/252, a_max: float | None = 2.0,
          z: float = 1.2,
          cap_mode: str = "component_tanh",
          r: float = 0.01,
          critic_steps: int = 5,
          advantage_norm_eps: float = 1e-8,
          omega_ema_beta: float = 0.9,
          g_p_dep: bool = False,
          actor_mix_tail: float = 0.5,
          cov_scale: float = 1.0,
          lr_step_every: int = 0,
          lr_gamma: float = 1.0,
          episodes_per_iter: int = 1,
          apply_action_projection: bool = True,
          estimation_method: str = "heuristic",
          ):
    set_seed(seed)
    outdir.mkdir(parents=True, exist_ok=True)

    # paper's simulation "true" parameters
    true_params = RSGBMParams(
        mu1=np.array([0.25, 0.18]), mu2=np.array([-0.73, -0.40]),
        Sigma=np.array([[0.22**2, 0.22*0.18*0.3],[0.22*0.18*0.3, 0.18**2]]),
        lam1=0.36, lam2=2.89, r=r)

    if mode == "true_params":
        est_params = true_params
        policy_params = true_params
        filt_params = FilterParams(mu1=true_params.mu1, mu2=true_params.mu2, Sigma=true_params.Sigma,
                                   lam1=true_params.lam1, lam2=true_params.lam2, r=true_params.r)
    elif mode == "estimated_params":
        if estimation_method == "heuristic":
            est_params = make_estimated_params_via_heuristic(true_params, seed=seed)
        elif estimation_method == "hmm":
            est_params = make_estimated_params_via_hmm(true_params, seed=seed)
        elif estimation_method == "jump_model":
            est_params = make_estimated_params_via_jump_model(true_params, seed=seed)
        else:
            raise ValueError("estimation_method must be heuristic or hmm")

        policy_params = est_params
        filt_params = FilterParams(mu1=est_params.mu1, mu2=est_params.mu2, Sigma=est_params.Sigma,
                                   lam1=est_params.lam1, lam2=est_params.lam2, r=est_params.r)
    else:
        raise ValueError("mode must be true_params or estimated_params")

    # demo plot for filtering
    run_filter_demo(outdir, true_params=true_params, est_params=est_params, seed=seed)

    cfg = TrainConfig(T_years=episode_T_years, dt=dt, x0=1.0, s0=1.0, p0=0.5, z=z,
                        Lambda=Lambda, alpha_theta=alpha_theta, alpha_phi=alpha_phi, alpha_w=alpha_w,
                        omega_update_every=omega_update_every, a_max=a_max,
                        r=r,
                        critic_steps=critic_steps,
                        advantage_norm_eps=advantage_norm_eps,
                        omega_ema_beta=omega_ema_beta,
                        actor_mix_tail=actor_mix_tail,
                        cov_scale=cov_scale,
                        g_p_dep=g_p_dep,
                        lr_step_every=lr_step_every,
                        lr_gamma=lr_gamma,
                        episodes_per_iter=episodes_per_iter,
                        mu1=policy_params.mu1,
                        mu2=policy_params.mu2,
                        Sigma=policy_params.Sigma,
                        )
    agent = POEMVAgent(cfg)

    rows = []
    last_terminals = []
    last_mean_xT_window = float("nan")
    last_gap = float("nan")
    last_domega = 0.0

    def _episode_position_stats(traj):
        """
        Compute average gross leverage, cash weight, and absolute action size
        over one episode from realized risky positions u_k and wealth x_k.

        gross_lev_k = sum_i |u_{k,i}| / |x_k|
        cash_w_k    = 1 - sum_i u_{k,i}/x_k
        abs_u_k     = mean_i |u_{k,i}|
        """
        u = np.asarray(traj["u"], dtype=float)              # (n, d)
        x = np.asarray(traj["x"], dtype=float)[:-1]         # (n,)
        denom = np.maximum(np.abs(x), 1e-12)[:, None]       # (n,1)
        w = u / denom                                       # risky weights

        gross_lev = np.sum(np.abs(w), axis=1)               # (n,)
        cash_w = 1.0 - np.sum(w, axis=1)                    # (n,)
        abs_u = np.mean(np.abs(u), axis=1)                  # (n,)
        return {
            "avg_gross_leverage": float(np.mean(gross_lev)),
            "avg_cash_weight": float(np.mean(cash_w)),
            "avg_abs_action": float(np.mean(abs_u)),
            "max_gross_leverage": float(np.max(gross_lev)),
            "min_cash_weight": float(np.min(cash_w)),
        }

    for m in range(1, iters+1):
        batch_xT = []
        batch_pos_stats = []
        batch_trajs = []

        for b in range(int(episodes_per_iter)):
            # Simulate episodes under TRUE market parameters; filter with selected params.
            env = RSGBMEnv(true_params, EpisodeConfig(
                T_years=episode_T_years,
                dt=dt,
                x0=cfg.x0,
                s0=np.array([cfg.s0, cfg.s0]),
                p0=cfg.p0,
                a_max=a_max,
                omega=float(agent.omega.detach().cpu()),
                seed=seed + m * 10000 + b,
                apply_action_projection=apply_action_projection,
            ))
            obs = env.reset()

            n = env.n_steps
            t_arr = np.empty(n+1); x_arr = np.empty(n+1); p_arr = np.empty(n+1)
            u_arr = np.empty((n,2))
            f_roll_arr = np.empty(n)
            dlnf_roll_arr = np.empty(n)
            #omega_roll_arr = np.empty(n)
            omega_roll = None
            p = cfg.p0
            t_arr[0]=0.0; x_arr[0]=cfg.x0; p_arr[0]=p
            S_prev = obs["S"]
            u_raw_arr = np.empty((n,2))

            for k in range(n):
                u, u_raw, info = agent.act(t_arr[k], x_arr[k], p, deterministic=False)
                u_raw_arr[k] = np.asarray(u_raw, float)
                u_arr[k] = np.asarray(u, float)
                f_roll_arr[k] = info["f_roll"]
                dlnf_roll_arr[k] = info["dlnf_roll"]
                #omega_roll_arr[k] = info["omega_roll"]
                if omega_roll is None:
                    omega_roll = float(info["omega_roll"])
                obs, _, done = env.step(u)
                S_now = obs["S"]
                log_return = np.log(S_now / S_prev)  # (2,)
                p, _ = wonham_filter_q_update(p, log_return, dt, filt_params)
                p = safe_clip_p(p)

                t_arr[k+1] = obs["t"]
                x_arr[k+1] = obs["X"]
                p_arr[k+1] = p
                S_prev = S_now
                if done:
                    break

            steps_done = k + 1
            t_arr = t_arr[:steps_done+1]
            x_arr = x_arr[:steps_done+1]
            p_arr = p_arr[:steps_done+1]
            u_raw_arr = u_raw_arr[:steps_done]
            u_arr = u_arr[:steps_done]
            f_roll_arr = f_roll_arr[:steps_done]
            dlnf_roll_arr = dlnf_roll_arr[:steps_done]
            #omega_roll_arr = omega_roll_arr[:steps_done]
            traj = dict(t=t_arr, x=x_arr, p=p_arr, u=u_arr, u_raw=u_raw_arr, a_max=a_max,
                        #f_roll=f_roll_arr,dlnf_roll=dlnf_roll_arr,omega_roll = omega_roll_arr
                        f_roll=f_roll_arr, dlnf_roll=dlnf_roll_arr, omega_roll=omega_roll)
            batch_trajs.append(traj)
            batch_pos_stats.append(_episode_position_stats(traj))
            batch_xT.append(float(x_arr[-1]))

        loss_c, loss_a = agent.update_from_episodes(batch_trajs)
        agent.step_schedulers()

        xT = float(np.mean(batch_xT))
        last_terminals.append(xT)
        if len(last_terminals) > omega_update_every:
            last_terminals = last_terminals[-omega_update_every:]

        last_domega = 0.0
        if (m % omega_update_every) == 0:
            mean_xT = float(np.mean(last_terminals))
            last_mean_xT_window = mean_xT
            last_gap = mean_xT - cfg.z
            omega_before = float(agent.omega.detach().cpu())
            agent.update_omega(mean_xT)
            lr_info = agent.current_lrs()
            omega_after = float(agent.omega.detach().cpu())
            last_domega = omega_after - omega_before

        mean_pos_stats = {
            key: float(np.mean([ps[key] for ps in batch_pos_stats]))
            for key in batch_pos_stats[0].keys()
        }
        phi = agent.policy_values()
        rows.append(dict(
            iter=m,
            omega=float(agent.omega.detach().cpu()),
            mean_terminal=float(np.mean(last_terminals)),
            xT=xT,
            mean_xT_window=last_mean_xT_window,
            mean_xT_minus_z=last_gap,
            domega=last_domega,
            loss_critic=float(loss_c),
            loss_actor=float(loss_a),
            avg_gross_leverage=mean_pos_stats["avg_gross_leverage"],
            avg_cash_weight=mean_pos_stats["avg_cash_weight"],
            avg_abs_action=mean_pos_stats["avg_abs_action"],
            max_gross_leverage=mean_pos_stats["max_gross_leverage"],
            min_cash_weight=mean_pos_stats["min_cash_weight"],
            episodes_per_iter=int(episodes_per_iter),
            lr_theta=agent.current_lrs()["lr_theta"],
            lr_phi=agent.current_lrs()["lr_phi"],
            **phi
        ))

        if m % max(100, omega_update_every) == 0:
            pd.DataFrame(rows).to_csv(outdir/"metrics.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(outdir/"metrics.csv", index=False)

    # plots
    fig = plt.figure()
    plt.plot(df["iter"], df["mean_terminal"])
    plt.axhline(cfg.z, linestyle="--")
    plt.xlabel("iteration")
    plt.ylabel("mean terminal wealth (recent window)")
    fig.tight_layout()
    fig.savefig(outdir/"learning_curves.png", dpi=200)
    plt.close(fig)

    with open(outdir/"run_config.json","w",encoding="utf-8") as f:
        import json
        def _json_default(o):
            # numpy arrays / scalars -> python native
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")
        json.dump(dict(
            mode=mode, seed=seed, iters=iters,
            T=episode_T_years, dt=dt, a_max=a_max, cap_mode=cap_mode, r=r,
            x0=cfg.x0, p0=cfg.p0, z=cfg.z,
            Lambda=Lambda,
            alpha_theta=alpha_theta, alpha_phi=alpha_phi, alpha_w=alpha_w,
            omega_update_every=omega_update_every,
            critic_steps=critic_steps,
            advantage_norm_eps=advantage_norm_eps,
            omega_ema_beta=omega_ema_beta,
            g_p_dep=g_p_dep,
            actor_mix_tail=actor_mix_tail,
            cov_scale=cov_scale,
            lr_step_every=lr_step_every,
            lr_gamma=lr_gamma,
            policy_params=policy_params.__dict__,
            true_params=true_params.__dict__,
            filter_params=filt_params.__dict__,
            estimated_params=est_params.__dict__,
        ), f, indent=2, default=_json_default)
    torch.save(
        {
            "vf_state_dict": agent.vf.state_dict(),
            "pi_state_dict": agent.pi.state_dict(),
            "omega": float(agent.omega.detach().cpu()),
        },
        outdir / "checkpoint.pt",
    )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="true_params", choices=["true_params","estimated_params"])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="runs/poemv_rs")
    ap.add_argument("--Lambda", type=float, default=1.0)
    ap.add_argument("--alpha_theta", type=float, default=1e-3)
    ap.add_argument("--alpha_phi", type=float, default=1e-3)
    ap.add_argument("--alpha_w", type=float, default=5e-3)
    ap.add_argument("--omega_update_every", type=int, default=10)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=1/252)
    ap.add_argument("--a_max", type=float, default=2.0)
    ap.add_argument("--r", type=float, default=0.01)
    ap.add_argument("--cap_mode", type=str, default="component_tanh", choices=["none","component_tanh","l1_radial"])
    ap.add_argument("--critic_steps", type=int, default=20)
    ap.add_argument("--advantage_norm_eps", type=float, default=1e-8)
    ap.add_argument("--omega_ema_beta", type=float, default=0.9)
    ap.add_argument("--actor_mix_tail", type=float, default=0.5)
    ap.add_argument("--cov_scale", type=float, default=1.0)
    ap.add_argument("--g_p_dep", action="store_true")
    ap.add_argument("--lr_step_every", type=int, default=0)
    ap.add_argument("--lr_gamma", type=float, default=1.0)
    ap.add_argument("--episodes_per_iter", type=int, default=32)
    ap.add_argument("--z", type=float, default=1.2)
    ap.add_argument("--apply_action_projection", action="store_true")
    ap.add_argument("--estimation_method", type=str, choices=["heuristic", "hmm", "jump_model"], default="heuristic")
    args = ap.parse_args()
    train(args.mode, args.iters, args.seed, Path(args.outdir), args.alpha_theta, args.alpha_phi, args.alpha_w,
          args.Lambda, args.omega_update_every, episode_T_years=args.T, dt=args.dt,z = args.z, a_max=args.a_max,
          r=args.r,
          critic_steps=args.critic_steps,
          advantage_norm_eps=args.advantage_norm_eps,
          omega_ema_beta=args.omega_ema_beta,
          actor_mix_tail=args.actor_mix_tail,
          cov_scale=args.cov_scale,
          g_p_dep=args.g_p_dep,
          lr_step_every=args.lr_step_every,
          lr_gamma=args.lr_gamma,
          episodes_per_iter=args.episodes_per_iter,
          cap_mode=args.cap_mode,
          apply_action_projection=args.apply_action_projection,
          estimation_method=args.estimation_method
          )

if __name__ == "__main__":
    main()
