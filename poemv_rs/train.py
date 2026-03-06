from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import math
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torch.optim as optim

from .env import RSGBMParams,EpisodeConfig, RSGBMEnv,simulate_price_only,HistoricalLogReturnEnv
from .filtering import FilterParams, wonham_filter_q_update

from .heuristic import HeuristicThresholds, label_bull_bear_from_drawdowns, estimate_env_params_from_labeled_returns
from .agent import POEMVAgent, TrainConfig
from .utils import set_seed, safe_clip_p

def _parse_z_list(s: str) -> list[float]:
    # examples: "1.05,1.10,1.20" or "1.05:1.55:0.05"
    s = s.strip()
    if ":" in s:
        a, b, step = s.split(":")
        a = float(a); b = float(b); step = float(step)
        z = []
        x = a
        while x <= b + 1e-12:
            z.append(float(x))
            x += step
        return z
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _weights_minvar(Sigma: np.ndarray) -> np.ndarray:
    n = Sigma.shape[0]
    one = np.ones(n)
    w = np.linalg.solve(Sigma + 1e-12*np.eye(n), one)
    w = w / (one @ w)
    return w

def _weights_tangency(Sigma: np.ndarray, mu: np.ndarray) -> np.ndarray:
    w = np.linalg.solve(Sigma + 1e-12*np.eye(len(mu)), mu)
    s = np.sum(w)
    if abs(s) < 1e-12:
        return _weights_minvar(Sigma)
    return w / s

def _weights_target_mean(Sigma: np.ndarray, mu: np.ndarray, m_target: float) -> np.ndarray:
    """
    Solve: min w^T Sigma w  s.t. 1^T w = 1,  mu^T w = m_target
    Closed form: w = Sigma^{-1}(a*1 + b*mu), where [a,b] solves 2x2 system.
    """
    n = Sigma.shape[0]
    one = np.ones(n)
    #Sinv = np.linalg.inv(Sigma + 1e-12*np.eye(n))
    #A = one @ (Sinv @ one)
    #B = one @ (Sinv @ mu)
    #C = mu  @ (Sinv @ mu)
    Sigma_reg = Sigma + 1e-12*np.eye(n)
    # x = Sigma^{-1} 1,  y = Sigma^{-1} mu  via solve
    x = np.linalg.solve(Sigma_reg, one)
    y = np.linalg.solve(Sigma_reg, mu)
    A = float(one @ x)
    B = float(one @ y)
    C = float(mu  @ y)
    det = A*C - B*B
    if abs(det) < 1e-12:
        # fallback (near-collinear): tangency or minvar
        return _weights_tangency(Sigma, mu)
    a = (C*1.0 - B*m_target) / det
    b = (A*m_target - B*1.0) / det
    #w = Sinv @ (a*one + b*mu)
    # w = Sigma^{-1}(a*1 + b*mu) = a*x + b*y
    w = a*x + b*y
    return w

def _backtest_monthly_MV_EW(logret_df_test: pd.DataFrame, lookback_days: int, mv_kind: str, x0: float = 1.0,
                            target_mu: float | None = None,
                            gross_lev_max: float | None = None,
                            tcost: float = 0.0):
    """
    Monthly rebalance at month start using rolling estimates from past lookback_days.
    Wealth: X_{t+1}=X_t + u^T r_t where r_t = exp(logret)-1, and u = w*X (fully invested).
    Returns: dates, X_MV_path, X_EW_path
    """
    df = logret_df_test.copy().dropna(how="any")
    dates = pd.to_datetime(df.index)
    logret = df.values.astype(float)
    T, d = logret.shape
    ret = np.exp(logret) - 1.0

    month = dates.to_period("M")
    is_month_start = np.zeros(T, dtype=bool)
    is_month_start[0] = True
    is_month_start[1:] = (month[1:] != month[:-1])

    X_mv = np.empty(T+1); X_ew = np.empty(T+1)
    X_mv[0] = x0; X_ew[0] = x0
    gross_mv = np.empty(T+1); gross_ew = np.empty(T+1)
    gross_mv[0] = 0.0; gross_ew[0] = 0.0
    u_mv = np.zeros(d); u_ew = np.zeros(d)

    for t in range(T):
        if is_month_start[t]:
            start = max(0, t - lookback_days)
            if (t - start) >= 20:
                lr_hist = logret[start:t]
                mu_hat = lr_hist.mean(axis=0)
                Sigma_hat = np.cov(lr_hist.T, ddof=1)
                if mv_kind == "tangency":
                    w = _weights_tangency(Sigma_hat, mu_hat)
                elif mv_kind == "target":
                    if target_mu is None:
                        raise ValueError("mv_kind='target' requires target_mu")
                    w = _weights_target_mean(Sigma_hat, mu_hat, target_mu)
                else:
                    w = _weights_minvar(Sigma_hat)
            else:
                w = np.ones(d) / d
             # ---- gross leverage cap (important for MV(target)) ----
            if (gross_lev_max is not None) and (gross_lev_max > 0):
                lev = float(np.sum(np.abs(w)))
                if lev > gross_lev_max:
                    w = w * (gross_lev_max / lev)
            w_ew = np.ones(d) / d
            u_new_mv = w * X_mv[t]
            u_new_ew = w_ew * X_ew[t]

            # ---- transaction cost on rebalance: kappa * sum |Δu| ----
            if tcost > 0:
                X_mv[t] -= float(tcost * np.sum(np.abs(u_new_mv - u_mv)))
                X_ew[t] -= float(tcost * np.sum(np.abs(u_new_ew - u_ew)))
            u_mv = u_new_mv
            u_ew = u_new_ew
            #u_mv = w * X_mv[t]
            #u_ew = w_ew * X_ew[t]

        X_mv[t+1] = X_mv[t] + float(u_mv @ ret[t])
        X_ew[t+1] = X_ew[t] + float(u_ew @ ret[t])
        denom_mv = X_mv[t+1] if abs(X_mv[t+1]) > 1e-12 else 1e-12
        denom_ew = X_ew[t+1] if abs(X_ew[t+1]) > 1e-12 else 1e-12
        gross_mv[t+1] = float(np.sum(np.abs(u_mv / denom_mv)))
        gross_ew[t+1] = float(np.sum(np.abs(u_ew / denom_ew)))

    # create a (T+1,) time index for plotting
    dates_path = pd.Index(dates).insert(0, dates[0] - pd.Timedelta(days=1))
    return dates_path, X_mv, X_ew, gross_mv, gross_ew
def _eval_rl_on_env(agent: POEMVAgent, filt_params: FilterParams, logret: np.ndarray, dt: float,
                    n_steps: int, a_max: float, n_eval: int, seed0: int):
    """Evaluate RL on either simulated (logret generated outside) or historical (given logret).
    Here logret is (T,d). We sample random windows of length n_steps from it.
    Returns array of terminal wealths.
    """
    rng = np.random.default_rng(seed0)
    T = logret.shape[0]
    if T <= n_steps + 1:
        raise ValueError("eval logret too short for n_steps")
    xT = []
    for j in range(n_eval):
        start = int(rng.integers(0, T - n_steps))
        env = HistoricalLogReturnEnv(logret[start:start+n_steps], dt=dt, x0=agent.cfg.x0, seed=seed0 + j)
        obs = env.reset(0)
        p = agent.cfg.p0
        S_prev = obs["S"]
        for k in range(env.n_steps):
            u, _, _ = agent.act(obs["t"], obs["X"], p)
            obs, _, done = env.step(u)
            S_now = obs["S"]
            lr = np.log(S_now / S_prev)
            p, _ = wonham_filter_q_update(p, lr, dt, filt_params)
            p = safe_clip_p(p)
            S_prev = S_now
            if done:
                break
        xT.append(float(obs["X"]))
    return np.asarray(xT, float)

def _eval_bench_monthly_on_logret(logret_df: pd.DataFrame, dt: float, n_steps: int, n_eval: int, seed0: int,
                                  lookback_days: int = 252):
    """Evaluate MV(minvar), MV(tangency), EW with monthly rebalance on random windows of the logret_df.
    Returns dict of terminal wealth arrays.
    """
    df = logret_df.dropna(how="any")
    rng = np.random.default_rng(seed0)
    T = df.shape[0]
    if T <= n_steps + 1:
        raise ValueError("benchmark logret too short for n_steps")
    out = {"MV_minvar": [], "MV_tangency": [], "EW": []}
    for j in range(n_eval):
        start = int(rng.integers(0, T - n_steps))
        win = df.iloc[start:start+n_steps]
        # reuse helper: it returns full path; take terminal
        _, X_mv1, X_ew = _backtest_monthly_MV_EW(win, lookback_days=lookback_days, mv_kind="minvar", x0=1.0)
        _, X_mv2, _    = _backtest_monthly_MV_EW(win, lookback_days=lookback_days, mv_kind="tangency", x0=1.0)
        out["MV_minvar"].append(float(X_mv1[-1]))
        out["MV_tangency"].append(float(X_mv2[-1]))
        out["EW"].append(float(X_ew[-1]))
    for k in out:
        out[k] = np.asarray(out[k], float)
    return out
def _backtest_rl_on_logret(agent: POEMVAgent, filt_params: FilterParams,
                           logret_df: pd.DataFrame, dt: float,
                           rebalance: str = "daily",
                           tcost: float = 0.0,
                           gross_lev_max: float | None = None):
    """
    Backtest RL on a fixed logret_df window.
    - rebalance="daily": update action every step
    - rebalance="monthly": update action only at month start, hold u otherwise
    Transaction cost: kappa * sum |Δu| charged when u is updated.
    Returns: dates_path (T+1), X_path (T+1)
    """
    df = logret_df.copy().dropna(how="any")
    dates = pd.to_datetime(df.index)
    logret = df.values.astype(float)   # (T,d)
    ret = np.exp(logret) - 1.0
    T, d = logret.shape

    month = dates.to_period("M")
    is_month_start = np.zeros(T, dtype=bool)
    is_month_start[0] = True
    is_month_start[1:] = (month[1:] != month[:-1])

    # price path used only for filtering
    S = np.ones(d, dtype=float)
    p = float(agent.cfg.p0)

    X = np.empty(T+1, dtype=float)
    X[0] = float(agent.cfg.x0)
    gross = np.empty(T+1, dtype=float)
    gross[0] = 0.0
    u_prev = np.zeros(d, dtype=float)
    u = np.zeros(d, dtype=float)

    for t in range(T):
        do_trade = (rebalance == "daily") or (rebalance == "monthly" and is_month_start[t])
        if do_trade:
            u_new, _, _ = agent.act(t*dt, float(X[t]), p)
            u_new = np.asarray(u_new, float).reshape(d,)
            # Optional gross leverage cap: sum_i |w_i| <= gross_lev_max
            # where w = u / X at the trade time.
            if (gross_lev_max is not None) and (gross_lev_max > 0):
                denom = X[t] if abs(X[t]) > 1e-12 else 1e-12
                lev = float(np.sum(np.abs(u_new / denom)))
                if lev > gross_lev_max:
                    u_new = u_new * (gross_lev_max / lev)
            if tcost > 0:
                X[t] -= float(tcost * np.sum(np.abs(u_new - u_prev)))
            u_prev = u_new
            u = u_new

        # wealth update
        X[t+1] = X[t] + float(u @ ret[t])
        # gross leverage at end-of-day (based on current holdings u and wealth X[t+1])
        denom = X[t+1] if abs(X[t+1]) > 1e-12 else 1e-12
        gross[t+1] = float(np.sum(np.abs(u / denom)))

        # filtering update (using prices)
        S_next = S * np.exp(logret[t])
        lr = np.log(S_next / S)
        p, _ = wonham_filter_q_update(p, lr, dt, filt_params)
        p = safe_clip_p(p)
        S = S_next

    dates_path = pd.Index(dates).insert(0, dates[0] - pd.Timedelta(days=1))
    return dates_path, X, gross

def run_filter_demo(outdir: Path, true_params: RSGBMParams, est_params: RSGBMParams, T_years=10.0, dt=1/252, seed=0):
    S, I = simulate_price_only(true_params, T_years=T_years, dt=dt, s0=1.0, seed=seed)
    logret = np.log(S[1:] / S[:-1])  # (n,2)
    p_true = np.empty(len(S)); p_est = np.empty(len(S))
    p_true[0]=0.5; p_est[0]=0.5

    fp_true = FilterParams(mu1=true_params.mu1, mu2=true_params.mu2, Sigma1=true_params.Sigma1, Sigma2=true_params.Sigma2,
                           lam1=true_params.lam1, lam2=true_params.lam2, r=true_params.r)
    fp_est  = FilterParams(mu1=est_params.mu1,  mu2=est_params.mu2,  Sigma1=est_params.Sigma1,  Sigma2=est_params.Sigma2,
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

def make_estimated_params_via_heuristic(true_params: RSGBMParams, seed=0):
    # Simulate 30y, use first 20y to estimate, per paper description.
    dt = 1/252
    S, _ = simulate_price_only(true_params, T_years=30.0, dt=dt, s0=1.0, seed=seed)
    S_train = S[: int(round(20.0/dt))+1]  # (T,2)
    # label using asset-1 price (minimal). You can replace with equal-weight index.
    reg = label_bull_bear_from_drawdowns(S_train[:,0], HeuristicThresholds())
    ret_train = (S_train[1:] - S_train[:-1]) / S_train[:-1]  # (T-1,2)
    est = estimate_env_params_from_labeled_returns(ret_train, reg, dt=dt)
    return RSGBMParams(**est, r=true_params.r)

def _export_center_wstar_p(agent: POEMVAgent, out_path: Path, p_bins: int = 99, t0: float = 0.0, x0: float = 1.0):
    """
    Export center w*(p) computed from deterministic mean action:
      a*(p) = agent.mean_action(t0, x0, p)
      w*(p) = a*(p) / x0
    Saved as .npz: p_grid (G,), w_star (G,d)
    """
    p_grid = np.linspace(0.01, 0.99, int(p_bins))
    ws = []
    for p in p_grid:
        a = agent.mean_action(t0, x0, float(p))  # (d,)
        w = np.asarray(a, float) / float(x0)
        ws.append(w)
    w_star = np.stack(ws, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, p_grid=p_grid, w_star=w_star)

def _load_center_wstar(path: Path):
    z = np.load(path, allow_pickle=False)
    return z["p_grid"].astype(float), z["w_star"].astype(float)

def _interp_wstar(p: float, p_grid: np.ndarray, w_star: np.ndarray) -> np.ndarray:
    # linear interpolation in p (w_star: (G,d))
    p = float(np.clip(p, p_grid[0], p_grid[-1]))
    j = int(np.searchsorted(p_grid, p))
    if j <= 0:
        return w_star[0].copy()
    if j >= len(p_grid):
        return w_star[-1].copy()
    p0, p1 = p_grid[j-1], p_grid[j]
    a = (p - p0) / (p1 - p0 + 1e-12)
    return (1-a)*w_star[j-1] + a*w_star[j]

def _backtest_band_on_logret(logret_df: pd.DataFrame,
                             filt_params: FilterParams,
                             p_grid: np.ndarray, w_star_grid: np.ndarray,
                             delta: np.ndarray,
                             dt: float,
                             x0: float = 1.0,
                             tcost: float = 0.0,
                             gross_lev_max: float | None = None):
    """
    Band (width-only, projection) backtest.
    State uses p_t (Wonham filter on observed logret).
    Holdings stored as dollar position u. Weight is w=u/X.
    Rule:
      if |w - w*| <= delta : no trade
      else : project to boundary: w_new = w* + clip(w-w*, -delta, +delta)
    Cost on trade: kappa * sum |Δu|.
    """
    df = logret_df.copy().dropna(how="any")
    dates = pd.to_datetime(df.index)
    logret = df.values.astype(float)   # (T,d)
    ret = np.exp(logret) - 1.0
    T, d = logret.shape
    delta = np.asarray(delta, float).reshape(d,)

    X = np.empty(T+1, dtype=float)
    X[0] = float(x0)
    gross = np.empty(T+1, dtype=float)
    gross[0] = 0.0
    u = np.zeros(d, dtype=float)

    # price for filtering
    S = np.ones(d, dtype=float)
    p = 0.5

    for t in range(T):
        # current weights
        w = (u / X[t]) if abs(X[t]) > 1e-12 else np.zeros(d)
        w_star = _interp_wstar(p, p_grid, w_star_grid)
        # check band
        dev = w - w_star
        inside = np.all(np.abs(dev) <= delta)
        if not inside:
            w_new = w_star + np.clip(dev, -delta, +delta)
            if (gross_lev_max is not None) and (gross_lev_max > 0):
                lev = float(np.sum(np.abs(w_new)))
                if lev > gross_lev_max:
                    w_new = w_new * (gross_lev_max / lev)
            u_new = w_new * X[t]
            if tcost > 0:
                X[t] -= float(tcost * np.sum(np.abs(u_new - u)))
            u = u_new

        # wealth update
        X[t+1] = X[t] + float(u @ ret[t])
        denom = X[t+1] if abs(X[t+1]) > 1e-12 else 1e-12
        gross[t+1] = float(np.sum(np.abs(u / denom)))

        # filter update
        S_next = S * np.exp(logret[t])
        lr = np.log(S_next / S)
        p, _ = wonham_filter_q_update(p, lr, dt, filt_params)
        p = safe_clip_p(p)
        S = S_next

    dates_path = pd.Index(dates).insert(0, dates[0] - pd.Timedelta(days=1))
    return dates_path, X, gross

def _train_band_delta(logret_train_df: pd.DataFrame,
                      filt_params: FilterParams,
                      p_grid: np.ndarray, w_star_grid: np.ndarray,
                      dt: float, T_years: float,
                      z_target: float,
                      iters: int,
                      tcost: float,
                      delta_init: float,
                      lr: float,
                      gross_lev_max: float | None,
                      seed: int,
                      outdir: Path):
    """
    Learn band width delta (per-asset) under transaction costs.
    Center w*(p) fixed.
    Objective: maximize quadratic terminal utility h(X_T) with dual omega (same as RL style).
    We do simple stochastic gradient on delta via reparameterization:
      delta = softplus(theta)   (theta trainable)
    We estimate gradient through Monte Carlo episodes using torch autograd by simulating band updates in torch.
    (This is a minimal, stable learner for width-only.)
    """
    df = logret_train_df.copy().dropna(how="any")
    logret_all = df.values.astype(np.float64)
    T_all, d = logret_all.shape
    n_steps = int(round(T_years / dt))
    if T_all <= n_steps + 5:
        raise ValueError("train data too short for band training episode length")

    rng = np.random.default_rng(seed)
    device = torch.device("cpu")
    theta = torch.nn.Parameter(torch.full((d,), float(delta_init), dtype=torch.float64, device=device))
    opt = optim.Adam([theta], lr=float(lr))

    omega = torch.tensor(0.0, dtype=torch.float64, device=device)
    alpha_w = 1e-2  # keep simple; can expose if needed

    rows = []
    for it in range(1, iters+1):
        start = int(rng.integers(0, T_all - n_steps))
        logret = torch.tensor(logret_all[start:start+n_steps], dtype=torch.float64, device=device)  # (n,d)
        ret = torch.exp(logret) - 1.0

        delta = torch.nn.functional.softplus(theta)  # (d,) positive
        # Use scalar state x (no in-place writes on a tensor needed for autograd)
        x = torch.tensor(1.0, dtype=torch.float64, device=device)        
        u = torch.zeros(d, dtype=torch.float64, device=device)
        S = torch.ones(d, dtype=torch.float64, device=device)
        p = 0.5

        # pre-load numpy for filter to keep it minimal (filter is scalar anyway)
        for t in range(n_steps):
            # weights
            w = u / (x + 1e-12)
            w_star = torch.tensor(_interp_wstar(p, p_grid, w_star_grid), dtype=torch.float64, device=device)
            dev = w - w_star
            inside = torch.all(torch.abs(dev) <= delta)
            if not bool(inside.item()):
                w_new = w_star + torch.clamp(dev, -delta, +delta)
                if (gross_lev_max is not None) and (gross_lev_max > 0):
                    lev = torch.sum(torch.abs(w_new))
                    w_new = torch.where(lev > gross_lev_max, w_new * (gross_lev_max / (lev + 1e-12)), w_new)
                u_new = w_new * x
                if tcost > 0:
                    x = x - float(tcost) * torch.sum(torch.abs(u_new - u))
                u = u_new
            # wealth update (no in-place)
            x = x + torch.dot(u, ret[t])

            # filter update (numpy)
            S_next = (S * torch.exp(logret[t])).detach().cpu().numpy()
            lr_np = np.log(S_next / S.detach().cpu().numpy())
            p, _ = wonham_filter_q_update(p, lr_np, dt, filt_params)
            p = float(safe_clip_p(p))
            S = torch.tensor(S_next, dtype=torch.float64, device=device)

        xT = x
        # quadratic terminal utility (same shape as your RL): h = (xT-omega)^2 - (omega-z)^2
        h = (xT - omega)**2 - (omega - float(z_target))**2
        # maximize h => minimize -h
        loss = -h

        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        # dual ascent for omega to meet mean target (very rough but works as a stabilizer)
        with torch.no_grad():
            omega = omega + alpha_w * (xT.detach() - float(z_target))

        if (it % 50) == 0 or it == 1:
            rows.append({
                "iter": it,
                "loss": float(loss.detach().cpu()),
                "xT": float(xT.detach().cpu()),
                "omega": float(omega.detach().cpu()),
                "delta_mean": float(delta.detach().mean().cpu()),
                "delta": delta.detach().cpu().numpy().tolist(),
            })
            pd.DataFrame(rows).to_csv(outdir/"band_train_metrics.csv", index=False)

    # final save
    delta_final = torch.nn.functional.softplus(theta).detach().cpu().numpy()
    np.savez(outdir/"band_delta.npz", delta=delta_final)
    return delta_final

def train(mode: str, iters: int, seed: int, outdir: Path,
          alpha_theta: float, alpha_phi: float, alpha_w: float,
          Lambda: float, omega_update_every: int,
          episode_T_years: float = 10.0, dt: float = 1/252, a_max: float = 2.0,
          train_data_pkl: str | None = None,
          test_data_pkl: str | None = None,
          cols: list[str] | None = None,
          train_start: str | None = None, train_end: str | None = None,
          test_start: str | None = None, test_end: str | None = None,
          sim_test_seed: int | None = None,
          z_target: float = 1.5,
          mv_target_mu: float | None = None,
          rl_gross_lev_max: float | None = None,
          tcost: float = 0.0,
          export_center: bool = False,
          center_bins: int = 99,
          center_path: str | None = None,
          band_overlay: bool = False,
          band_delta: float = 0.05,
          band_tcost: float = 0.0,
          band_gross_lev_max: float | None = None,
          band_train: bool = False,
          band_train_iters: int = 300,
          band_train_lr: float = 3e-3,
          band_delta_init: float = 0.05,
          
          ):
    set_seed(seed)
    outdir.mkdir(parents=True, exist_ok=True)

    # paper's simulation "true" parameters
    true_params = RSGBMParams(
        mu1=np.array([0.25, 0.18]), mu2=np.array([-0.73, -0.40]),
        Sigma1=np.array([[0.22**2, 0.22*0.18*0.3],[0.22*0.18*0.3, 0.18**2]]),
        Sigma2=np.array([[0.22**2, 0.22*0.18*0.5],[0.22*0.18*0.5, 0.18**2]]),
        lam1=0.36, lam2=2.89, r=0.0)
    use_hist_train = (train_data_pkl is not None)
    use_hist_test  = (test_data_pkl is not None)

    if mode == "true_params":
        filt_params = FilterParams(mu1=true_params.mu1, mu2=true_params.mu2, Sigma1=true_params.Sigma1, Sigma2=true_params.Sigma2,
                                   lam1=true_params.lam1, lam2=true_params.lam2, r=true_params.r)
        est_params = true_params
    elif mode == "estimated_params":
        est_params = make_estimated_params_via_heuristic(true_params, seed=seed)
        filt_params = FilterParams(mu1=est_params.mu1, mu2=est_params.mu2, Sigma1=est_params.Sigma1, Sigma2=est_params.Sigma2,
                                   lam1=est_params.lam1, lam2=est_params.lam2, r=est_params.r)
    else:
        raise ValueError("mode must be true_params or estimated_params")

     # demo plot for filtering (only for simulation mode)
    if not use_hist_train:
        run_filter_demo(outdir, true_params=true_params, est_params=est_params, seed=seed)

    cfg = TrainConfig(T_years=episode_T_years, dt=dt, x0=1.0, s0=1.0, p0=0.5, z=float(z_target),
                        Lambda=Lambda, alpha_theta=alpha_theta, alpha_phi=alpha_phi, alpha_w=alpha_w,
                        omega_update_every=omega_update_every, a_max=a_max,
                        )
    agent = POEMVAgent(cfg)

    # ----- historical data setup -----
    df_train = None
    df_test = None
    logret_train = None
    logret_test = None
    if use_hist_train:
        df_train = _load_logret_pkl(train_data_pkl, cols, train_start, train_end)
        logret_train = df_train.values.astype(float)
        with open(outdir/"historical_cols.txt","w",encoding="utf-8") as f:
            f.write(",".join(df_train.columns.tolist()))
    if use_hist_test:
        df_test = _load_logret_pkl(test_data_pkl, cols, test_start, test_end)
        logret_test = df_test.values.astype(float)

    rows = []
    last_terminals = []
    last_mean_xT_window = float("nan")
    last_gap = float("nan")
    last_domega = 0.0

    for m in range(1, iters+1):
        # One episode:
        # - simulation: RSGBMEnv as before
        # - historical: sample a random window from train log-returns
        if not use_hist_train:
            env = RSGBMEnv(true_params, EpisodeConfig(
                T_years=episode_T_years, dt=dt, x0=cfg.x0, s0=cfg.s0, p0=cfg.p0,
                a_max=a_max,
                omega=float(agent.omega.detach().cpu()), seed=seed + m
            ))
            obs = env.reset()
            n = env.n_steps
            # infer asset dimension from the agent action (more reliable than obs["S"])
            u_tmp, u_raw_tmp, _ = agent.act(0.0, cfg.x0, cfg.p0)
            d = int(np.asarray(u_tmp).size)            
        else:
            # sample start so that we can take n steps (n = episode length)
            n = int(round(episode_T_years / dt))
            if logret_train.shape[0] <= n + 1:
                raise ValueError("train period too short for the requested episode length")
            start = np.random.default_rng(seed + m).integers(0, logret_train.shape[0] - n)
            env = HistoricalLogReturnEnv(logret_train[start:start+n], dt=dt, x0=cfg.x0, seed=seed+m)
            obs = env.reset(0)
            d = env.d

        #n = env.n_steps
        t_arr = np.empty(n+1); x_arr = np.empty(n+1); p_arr = np.empty(n+1)
        #logp_arr = np.empty(n); ent_arr = np.empty(n)
        #u_arr = np.empty((n, getattr(env, "d", 1))) if use_historical else np.empty(n)
        # always store vector actions/logits (d assets)
        u_arr = np.empty((n, d), dtype=float)

        p = cfg.p0
        t_arr[0]=0.0; x_arr[0]=cfg.x0; p_arr[0]=p
        S_prev = obs["S"]
        #u_raw_arr = np.empty((n,2))
        #u_raw_arr = np.empty((n, getattr(env, "d", 1))) if use_historical else np.empty(n)
        u_raw_arr = np.empty((n, d), dtype=float)

        for k in range(n):
            u, u_raw, _ = agent.act(t_arr[k], x_arr[k], p)
            #u_raw_arr[k] = np.asarray(u_raw, float)
            #u_arr[k] = np.asarray(u, float)
            #if use_historical:
            #    u_raw_arr[k] = np.asarray(u_raw, float)
            #    u_arr[k] = np.asarray(u, float)
            #else:
            #    u_raw_arr[k] = float(u_raw)
            #    u_arr[k] = float(u)
            u_raw_arr[k] = np.asarray(u_raw, float).reshape(d,)
            u_arr[k]     = np.asarray(u, float).reshape(d,)
            obs, _, done = env.step(u)
            #S_now = obs["S"]
            #log_return = np.log(S_now / S_prev)  # (2,)
            # observed log-return for filtering
            #if use_historical:
            #    # in Historical env, S is updated using the log-return already;
            #    # recover it from prices (stable enough) OR store it in env if you prefer
            #    S_now = obs["S"]
            #    log_return = np.log(S_now / S_prev)        # (d,)
            #else:
            #    S_now = obs["S"]
            #    log_return = float(np.log(S_now / S_prev)) # scalar
            # observed log-return for filtering (always vector now)
            S_now = obs["S"]
            log_return = np.log(np.asarray(S_now, float) / np.asarray(S_prev, float))  # (d,)
            p, _ = wonham_filter_q_update(p, log_return, dt, filt_params)
            p = safe_clip_p(p)

            t_arr[k+1] = obs["t"]
            x_arr[k+1] = obs["X"]
            p_arr[k+1] = p
            #logp_arr[k] = float(logprob.detach().cpu())
            #ent_arr[k] = float(entropy.detach().cpu())
            S_prev = S_now
            if done:
                break

        steps_done = k + 1
        t_arr = t_arr[:steps_done+1]
        x_arr = x_arr[:steps_done+1]
        p_arr = p_arr[:steps_done+1]
        u_raw_arr = u_raw_arr[:steps_done]
        u_arr = u_arr[:steps_done]

        traj = dict(t=t_arr, x=x_arr, p=p_arr, u=u_arr, u_raw=u_raw_arr, a_max=a_max)
        loss_c, loss_a = agent.update_from_episode(traj)

        xT = float(x_arr[-1])
        last_terminals.append(xT)
        #if len(last_terminals) > omega_update_every:
        #    last_terminals = last_terminals[-omega_update_every:]

        #if (m % omega_update_every) == 0:
        #    agent.update_omega(float(np.mean(last_terminals)))
        # keep a rolling window used for omega update
        if len(last_terminals) > omega_update_every:
            last_terminals = last_terminals[-omega_update_every:]

        last_domega = 0.0
        if (m % omega_update_every) == 0:
            mean_xT = float(np.mean(last_terminals))
            last_mean_xT_window = mean_xT
            last_gap = mean_xT - cfg.z
            omega_before = float(agent.omega.detach().cpu())
            agent.update_omega(mean_xT)
            omega_after = float(agent.omega.detach().cpu())
            last_domega = omega_after - omega_before

        phi = agent.phi_values()
        rows.append(dict(
            iter=m,
            omega=float(agent.omega.detach().cpu()),
            mean_terminal=float(np.mean(last_terminals)),
            xT=xT,
            mean_xT_window=last_mean_xT_window,
            mean_xT_minus_z=last_gap,
            domega=last_domega,
            loss_critic=loss_c,
            loss_actor=loss_a,
            **phi
        ))

        if m % max(100, omega_update_every) == 0:
            pd.DataFrame(rows).to_csv(outdir/"metrics.csv", index=False)
    # ----- historical TEST backtest (deterministic mean action is recommended) -----
    if use_hist_test:
        eval_logret = logret_test
        eval_df = df_test
    else:
        te_seed = (seed + 10_000) if sim_test_seed is None else int(sim_test_seed)
        S_te, _ = simulate_price_only(true_params, T_years=episode_T_years, dt=dt, s0=1.0, seed=te_seed)
        eval_logret = np.log(S_te[1:] / S_te[:-1])
        idx = pd.date_range(start="2000-01-03", periods=len(eval_logret), freq="B")
        eval_df = pd.DataFrame(eval_logret, index=idx)

    if eval_logret is not None and (eval_logret.shape[0] > 5):
        # (1) export center w*(p) after training (cost-free center)
        if export_center:
            cpath = Path(center_path) if center_path is not None else (outdir/"center_wstar.npz")
            _export_center_wstar_p(agent, cpath, p_bins=int(center_bins), t0=0.0, x0=1.0)

        # RL backtest on the whole eval_df (same period used by MV/EW)
        # NOTE: we will output BOTH daily-rebalance and monthly-rebalance RL paths.
        # Transaction cost is applied on action updates: kappa * sum |Δu|.
        dates_path, X_rl_daily, gross_rl_daily = _backtest_rl_on_logret(agent, filt_params, eval_df, dt=dt, rebalance="daily", tcost=tcost,gross_lev_max=rl_gross_lev_max)
        _,          X_rl_month, gross_rl_month = _backtest_rl_on_logret(agent, filt_params, eval_df, dt=dt, rebalance="monthly", tcost=tcost,gross_lev_max=rl_gross_lev_max)

        pd.DataFrame({"date": dates_path.astype(str), "X_RL_daily": X_rl_daily, "X_RL_monthly": X_rl_month}).to_csv(
            outdir/"test_backtest_X.csv", index=False
        )
        
        # --- Benchmarks: Monthly MV(minvar), MV(tangency), EW on the SAME test period ---
        # Use the same df_test that was loaded for test period (keeps dates).
        lookback_days = 252
        n_steps = int(round(episode_T_years / dt))

        target_mu_step = float(mv_target_mu) if mv_target_mu is not None else float(np.log(cfg.z / cfg.x0) / max(n_steps, 1))
        dates_path, X_mv_minvar, X_ew, gross_mv_minvar, gross_ew = _backtest_monthly_MV_EW(eval_df, lookback_days=lookback_days, mv_kind="minvar",
                                                                x0=cfg.x0, tcost=tcost)
        _, X_mv_tan, _Xew2, gross_mv_tan, _gross_ew2     = _backtest_monthly_MV_EW(eval_df, lookback_days=lookback_days, mv_kind="tangency",
                                                                x0=cfg.x0, tcost=tcost)
        _, X_mv_tgt, _Xew3, gross_mv_tgt, _gross_ew3     = _backtest_monthly_MV_EW(eval_df, lookback_days=lookback_days, mv_kind="target",
                                                                x0=cfg.x0, target_mu=target_mu_step, gross_lev_max=1.5, tcost=tcost)


        # Save benchmark paths
        bench_df = pd.DataFrame({
            "date": dates_path.astype(str),
            "X_RL_daily": X_rl_daily,
            "X_RL_monthly": X_rl_month,
            #"X_RL": np.asarray(X_path, float),
            "X_MV_minvar": X_mv_minvar,
            "X_MV_tangency": X_mv_tan,
            "X_MV_target": X_mv_tgt,
            "X_EW": X_ew,
             "gross_RL_daily": gross_rl_daily,
            "gross_RL_monthly": gross_rl_month,
            "gross_MV_minvar": gross_mv_minvar,
            "gross_MV_tangency": gross_mv_tan,
            "gross_MV_target": gross_mv_tgt,
            "gross_EW": gross_ew,
        })
        # (2) band overlay (width-only projection) on the same test tape
        if band_overlay:
            cpath = Path(center_path) if center_path is not None else (outdir/"center_wstar.npz")
            if not cpath.exists():
                # if center wasn't exported yet, export now
                _export_center_wstar_p(agent, cpath, p_bins=int(center_bins), t0=0.0, x0=1.0)
            p_grid, w_star_grid = _load_center_wstar(cpath)
            # delta can be learned delta file
            delta_vec = np.full((eval_df.shape[1],), float(band_delta), dtype=float)
            # if band training produced band_delta.npz, prefer it
            learned = outdir/"band_delta.npz"
            if learned.exists():
                try:
                    delta_vec = np.load(learned)["delta"].astype(float).reshape(-1,)
                except Exception:
                    pass
            _, X_band, gross_band = _backtest_band_on_logret(
                eval_df, filt_params, p_grid, w_star_grid,
                delta=delta_vec, dt=dt, x0=cfg.x0,
                tcost=float(band_tcost),
                gross_lev_max=band_gross_lev_max
            )
            bench_df["X_BAND"] = X_band
            bench_df["gross_BAND"] = gross_band
        bench_df.to_csv(outdir/"test_backtest_compare.csv", index=False)
        # --- gross leverage time-series plot (separate png, no subplots) ---
        fig = plt.figure()
        t_years = np.arange(len(bench_df)) * dt

        def _plot_if(col: str, label: str):
            if col in bench_df.columns:
                y = bench_df[col].values.astype(float)
                if np.all(np.isfinite(y)):
                    plt.plot(t_years, y, label=label)

        _plot_if("gross_RL_daily",   "RL(daily)")
        _plot_if("gross_RL_monthly", "RL(monthly)")
        _plot_if("gross_MV_minvar",  "MV(minvar, monthly)")
        #_plot_if("gross_MV_tangency","MV(tangency, monthly)")
        _plot_if("gross_MV_target",  "MV(target, monthly)")
        _plot_if("gross_EW",         "EW(monthly)")
        _plot_if("gross_BAND",       "BAND(width-only)")

        plt.xlabel("time (years)")
        plt.ylabel("gross leverage (sum |w|)")
        plt.legend()
        fig.tight_layout()
        fig.savefig(outdir/"gross_leverage_compare.png", dpi=200)
        plt.close(fig)

        # Plot all on one figure
        fig = plt.figure()
        #t_years = np.arange(len(X_path)) * dt
        #plt.plot(t_years, X_path, label="RL")
        t_years = np.arange(len(X_rl_daily)) * dt
        plt.plot(t_years, X_rl_daily, label="RL(daily)")
        plt.plot(t_years, X_rl_month, label="RL(monthly)")
        plt.plot(t_years, X_mv_minvar, label="MV(minvar, monthly)")
        #plt.plot(t_years, X_mv_tan, label="MV(tangency, monthly)")
        plt.plot(t_years, X_mv_tgt, label="MV(target, monthly)")
        plt.plot(t_years, X_ew, label="EW(monthly)")
        if "X_BAND" in bench_df.columns:
            plt.plot(t_years, bench_df["X_BAND"].values, label="BAND(width-only)")
        plt.xlabel("time (years)")
        plt.ylabel("wealth X (test)")
        plt.legend()
        fig.tight_layout()
        fig.savefig(outdir/"test_backtest_compare.png", dpi=200)
        plt.close(fig)

        # (3) band learning under transaction cost on TRAIN tape (center fixed)
        if band_train:
            if df_train is None:
                raise ValueError("band_train requires historical train_data_pkl (df_train) in current minimal implementation")
            cpath = Path(center_path) if center_path is not None else (outdir/"center_wstar.npz")
            if not cpath.exists():
                _export_center_wstar_p(agent, cpath, p_bins=int(center_bins), t0=0.0, x0=1.0)
            p_grid, w_star_grid = _load_center_wstar(cpath)
            delta_final = _train_band_delta(
                df_train, filt_params,
                p_grid, w_star_grid,
                dt=dt, T_years=float(episode_T_years),
                z_target=float(cfg.z),
                iters=int(band_train_iters),
                tcost=float(band_tcost),
                delta_init=float(band_delta_init),
                lr=float(band_train_lr),
                gross_lev_max=band_gross_lev_max,
                seed=int(seed),
                outdir=outdir
            )
            # re-run overlay with learned delta
            _, X_band2, gross_band2 = _backtest_band_on_logret(eval_df, filt_params, p_grid, w_star_grid, delta_final, dt=dt, x0=cfg.x0,
                                                 tcost=float(band_tcost), gross_lev_max=band_gross_lev_max)
            bench_df["X_BAND"] = X_band2
            bench_df["gross_BAND"] = gross_band2
            bench_df.to_csv(outdir/"test_backtest_compare.csv", index=False)

            fig = plt.figure()
            plt.plot(t_years, X_rl_daily, label="RL(daily)")
            plt.plot(t_years, X_rl_month, label="RL(monthly)")
            plt.plot(t_years, X_band2, label="BAND(width-only, learned)")
            plt.plot(t_years, X_mv_minvar, label="MV(minvar, monthly)")
            #plt.plot(t_years, X_mv_tan, label="MV(tangency, monthly)")
            plt.plot(t_years, X_mv_tgt, label="MV(target, monthly)")
            plt.plot(t_years, X_ew, label="EW(monthly)")
            plt.xlabel("time (years)")
            plt.ylabel("wealth X (test)")
            plt.legend()
            fig.tight_layout()
            fig.savefig(outdir/"test_backtest_compare.png", dpi=200)
            plt.close(fig)
    
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
        json.dump(dict(mode=mode, seed=seed, iters=iters, Lambda=Lambda,
                       alpha_theta=alpha_theta, alpha_phi=alpha_phi, alpha_w=alpha_w,
                       omega_update_every=omega_update_every,
                       true_params=true_params.__dict__, filter_params=filt_params.__dict__,
                       estimated_params=est_params.__dict__), f, indent=2, default=_json_default)

def _load_logret_pkl(pkl_path: str, cols: list[str] | None,
                     start: str | None, end: str | None) -> pd.DataFrame:
    df = pd.read_pickle(pkl_path)
    # index may be strings; parse to datetime
    try:
        df.index = pd.to_datetime(df.index)
    except Exception:
        pass
    if cols is not None and len(cols) > 0:
        df = df[cols]
    if start is not None:
        df = df[df.index >= pd.to_datetime(start)]
    if end is not None:
        df = df[df.index <= pd.to_datetime(end)]
    # drop rows where any selected asset is NaN
    df = df.dropna(how="any")
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="true_params", choices=["true_params","estimated_params"])
    ap.add_argument("--iters", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="runs/poemv_rs")
    ap.add_argument("--Lambda", type=float, default=1.0)
    ap.add_argument("--alpha_theta", type=float, default=1e-13)
    ap.add_argument("--alpha_phi", type=float, default=1e-4)
    ap.add_argument("--alpha_w", type=float, default=1e-3)
    ap.add_argument("--omega_update_every", type=int, default=10)
    ap.add_argument("--T", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=1/252)
    ap.add_argument("--a_max", type=float, default=4.0)
    # Backward compatible: --data_pkl sets BOTH train and test to historical
    ap.add_argument("--data_pkl", type=str, default=None, help="(deprecated) Path to pkl for both train/test")
    ap.add_argument("--train_data_pkl", type=str, default=None, help="Path to pkl of log returns for TRAIN")
    ap.add_argument("--test_data_pkl", type=str, default=None, help="Path to pkl of log returns for TEST")
    ap.add_argument("--cols", type=str, default=None, help="Comma-separated column names, e.g., LargeCap,Gold")
    ap.add_argument("--train_start", type=str, default=None)
    ap.add_argument("--train_end", type=str, default=None)
    ap.add_argument("--test_start", type=str, default=None)
    ap.add_argument("--test_end", type=str, default=None)
    ap.add_argument("--z", type=float, default=3.5)
    ap.add_argument("--sim_test_seed", type=int, default=None, help="Seed for simulated TEST tape (if no test_data_pkl)")
    ap.add_argument("--mv_target_mu", type=float, default=None,
                    help="Target per-step mean log-return for MV(target). Default aligns with RL z: log(z/x0)/n_steps")
    ap.add_argument("--rl_gross_lev_max", type=float, default=None,
                    help="Gross leverage cap for RL backtests: sum_i |w_i| <= cap at trade times.")
    ap.add_argument("--tcost", type=float, default=0.0,
                    help="Proportional transaction cost kappa charged on rebalance: kappa * sum |Δu| (dollar turnover). Example: 0.0005 (5bps)")
    # center export / band
    ap.add_argument("--export_center", action="store_true", help="Export center w*(p) to npz after training")
    ap.add_argument("--center_bins", type=int, default=99, help="Number of p-grid points for center export")
    ap.add_argument("--center_path", type=str, default=None, help="Path to center_wstar.npz (default: outdir/center_wstar.npz)")
    ap.add_argument("--band_overlay", action="store_true", help="Overlay Band(width-only projection) on test plot")
    ap.add_argument("--band_delta", type=float, default=0.05, help="Initial/fixed band half-width (per asset). Used if no learned delta exists.")
    ap.add_argument("--band_tcost", type=float, default=0.0, help="Transaction cost kappa for band backtest/train: kappa * sum|Δu|")
    ap.add_argument("--band_gross_lev_max", type=float, default=None, help="Gross leverage cap for band projected weights (sum|w| <= cap)")
    ap.add_argument("--band_train", action="store_true", help="Train band width delta on TRAIN tape under transaction costs")
    ap.add_argument("--band_train_iters", type=int, default=300, help="Band training iterations (episodes)")
    ap.add_argument("--band_train_lr", type=float, default=3e-3, help="Band delta learning rate")
    ap.add_argument("--band_delta_init", type=float, default=0.05, help="Band delta init for training (softplus parameter init)")

    args = ap.parse_args()
    cols = None if args.cols is None else [c.strip() for c in args.cols.split(",") if c.strip()]
    train_data_pkl = args.train_data_pkl
    test_data_pkl = args.test_data_pkl
    if args.data_pkl is not None:
        train_data_pkl = args.data_pkl
        test_data_pkl = args.data_pkl
    train(args.mode, args.iters, args.seed, Path(args.outdir), args.alpha_theta, args.alpha_phi, args.alpha_w,
              args.Lambda, args.omega_update_every, episode_T_years=args.T, dt=args.dt, a_max=args.a_max,
              train_data_pkl=train_data_pkl, test_data_pkl=test_data_pkl, cols=cols,
              train_start=args.train_start, train_end=args.train_end,
              test_start=args.test_start, test_end=args.test_end,
              sim_test_seed=args.sim_test_seed,
              z_target=args.z,
              mv_target_mu=args.mv_target_mu,
              rl_gross_lev_max=args.rl_gross_lev_max,
              tcost=args.tcost,
              export_center=args.export_center,
              center_bins=args.center_bins,
              center_path=args.center_path,
              band_overlay=args.band_overlay,
              band_delta=args.band_delta,
              band_tcost=args.band_tcost,
              band_gross_lev_max=args.band_gross_lev_max,
              band_train=args.band_train,
              band_train_iters=args.band_train_iters,
              band_train_lr=args.band_train_lr,
              band_delta_init=args.band_delta_init,
              )

if __name__ == "__main__":
    main()
