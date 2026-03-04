from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import math
from pathlib import Path
import matplotlib.pyplot as plt

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

def _backtest_monthly_MV_EW(logret_df_test: pd.DataFrame, lookback_days: int, mv_kind: str, x0: float = 1.0):
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
    u_mv = np.zeros(d); u_ew = np.zeros(d)

    for t in range(T):
        if is_month_start[t]:
            start = max(0, t - lookback_days)
            if (t - start) >= 20:
                lr_hist = logret[start:t]
                mu_hat = lr_hist.mean(axis=0)
                Sigma_hat = np.cov(lr_hist.T, ddof=1)
                w = _weights_tangency(Sigma_hat, mu_hat) if mv_kind == "tangency" else _weights_minvar(Sigma_hat)
            else:
                w = np.ones(d) / d
            w_ew = np.ones(d) / d
            u_mv = w * X_mv[t]
            u_ew = w_ew * X_ew[t]

        X_mv[t+1] = X_mv[t] + float(u_mv @ ret[t])
        X_ew[t+1] = X_ew[t] + float(u_ew @ ret[t])

    # create a (T+1,) time index for plotting
    dates_path = pd.Index(dates).insert(0, dates[0] - pd.Timedelta(days=1))
    return dates_path, X_mv, X_ew

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

def train(mode: str, iters: int, seed: int, outdir: Path,
          alpha_theta: float, alpha_phi: float, alpha_w: float,
          Lambda: float, omega_update_every: int,
          episode_T_years: float = 10.0, dt: float = 1/252, a_max: float = 2.0,
          data_pkl: str | None = None, cols: list[str] | None = None,
          train_start: str | None = None, train_end: str | None = None,
          test_start: str | None = None, test_end: str | None = None,
          z_target: float = 1.5,
          ):

    set_seed(seed)
    outdir.mkdir(parents=True, exist_ok=True)

    # paper's simulation "true" parameters
    true_params = RSGBMParams(
        mu1=np.array([0.25, 0.18]), mu2=np.array([-0.73, -0.40]),
        Sigma1=np.array([[0.22**2, 0.22*0.18*0.3],[0.22*0.18*0.3, 0.18**2]]),
        Sigma2=np.array([[0.22**2, 0.22*0.18*0.5],[0.22*0.18*0.5, 0.18**2]]),
        lam1=0.36, lam2=2.89, r=0.0)
    use_historical = (data_pkl is not None)

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
    if not use_historical:
        run_filter_demo(outdir, true_params=true_params, est_params=est_params, seed=seed)

    cfg = TrainConfig(T_years=episode_T_years, dt=dt, x0=1.0, s0=1.0, p0=0.5, z=float(z_target),
                        Lambda=Lambda, alpha_theta=alpha_theta, alpha_phi=alpha_phi, alpha_w=alpha_w,
                        omega_update_every=omega_update_every, a_max=a_max,
                        )
    agent = POEMVAgent(cfg)

    # ----- historical data setup -----
    if use_historical:
        df_train = _load_logret_pkl(data_pkl, cols, train_start, train_end)
        df_test  = _load_logret_pkl(data_pkl, cols, test_start, test_end)
        logret_train = df_train.values.astype(float)  # (Ttr, d)
        logret_test  = df_test.values.astype(float)   # (Tte, d)
        d = logret_train.shape[1]
        # if your policy/env are 2-asset fixed, enforce d=2
        # assert d == 2
        # optional: save which columns used
        with open(outdir/"historical_cols.txt","w",encoding="utf-8") as f:
            f.write(",".join(df_train.columns.tolist()))

    rows = []
    last_terminals = []
    last_mean_xT_window = float("nan")
    last_gap = float("nan")
    last_domega = 0.0

    for m in range(1, iters+1):
        # One episode:
        # - simulation: RSGBMEnv as before
        # - historical: sample a random window from train log-returns
        if not use_historical:
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
    if use_historical and (logret_test.shape[0] > 5):
        env_te = HistoricalLogReturnEnv(logret_test, dt=dt, x0=cfg.x0, seed=seed+12345)
        obs = env_te.reset(0)
        p = cfg.p0
        S_prev = obs["S"]
        X_path = [obs["X"]]
        for k in range(env_te.n_steps):
            # stochastic action:
            # u, _, _ = agent.act(k*dt, obs["X"], p)            # deterministic center (if you added agent.mean_action):
            u, _, _ = agent.act(k*dt, obs["X"], p)
            obs, _, done = env_te.step(u)
            S_now = obs["S"]
            log_return = np.log(S_now / S_prev)
            p, _ = wonham_filter_q_update(p, log_return, dt, filt_params)
            p = safe_clip_p(p)
            S_prev = S_now
            X_path.append(obs["X"])
            if done:
                break
        pd.DataFrame({"k": np.arange(len(X_path)), "X": X_path}).to_csv(outdir/"test_backtest_X.csv", index=False)

        #fig = plt.figure()
        #plt.plot(np.arange(len(X_path))*dt, X_path)
        #plt.xlabel("time (years)")
        #plt.ylabel("wealth X (test)")
        #fig.tight_layout()
        #fig.savefig(outdir/"test_backtest_X.png", dpi=200)
        #plt.close(fig)
        # --- Benchmarks: Monthly MV(minvar), MV(tangency), EW on the SAME test period ---
        # Use the same df_test that was loaded for test period (keeps dates).
        lookback_days = 252  # you can make this a CLI arg later
        dates_path, X_mv_minvar, X_ew = _backtest_monthly_MV_EW(df_test, lookback_days=lookback_days, mv_kind="minvar", x0=cfg.x0)
        _,          X_mv_tan,   _     = _backtest_monthly_MV_EW(df_test, lookback_days=lookback_days, mv_kind="tangency", x0=cfg.x0)

        # Save benchmark paths
        bench_df = pd.DataFrame({
            "date": dates_path.astype(str),
            "X_RL": np.asarray(X_path, float),
            "X_MV_minvar": X_mv_minvar,
            "X_MV_tangency": X_mv_tan,
            "X_EW": X_ew,
        })
        bench_df.to_csv(outdir/"test_backtest_compare.csv", index=False)

        # Plot all on one figure
        fig = plt.figure()
        t_years = np.arange(len(X_path)) * dt
        plt.plot(t_years, X_path, label="RL")
        plt.plot(t_years, X_mv_minvar, label="MV(minvar, monthly)")
        #plt.plot(t_years, X_mv_tan, label="MV(tangency, monthly)")
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

    return agent, filt_params, (df_train if use_historical else None), (df_test if use_historical else None), true_params

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
    ap.add_argument("--data_pkl", type=str, default=None, help="Path to pkl of log returns (DataFrame)")
    ap.add_argument("--cols", type=str, default=None, help="Comma-separated column names, e.g., LargeCap,Gold")
    ap.add_argument("--train_start", type=str, default=None)
    ap.add_argument("--train_end", type=str, default=None)
    ap.add_argument("--test_start", type=str, default=None)
    ap.add_argument("--test_end", type=str, default=None)
    ap.add_argument("--z", type=float, default=1.5)
    ap.add_argument("--frontier", action="store_true")
    ap.add_argument("--z_list", type=str, default="1.05:1.55:0.05")
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--lookback_days", type=int, default=252)
    args = ap.parse_args()
    cols = None if args.cols is None else [c.strip() for c in args.cols.split(",") if c.strip()]
    #train(args.mode, args.iters, args.seed, Path(args.outdir), args.alpha_theta, args.alpha_phi, args.alpha_w,
    #      args.Lambda, args.omega_update_every, episode_T_years=args.T, dt=args.dt, a_max=args.a_max,
    #      data_pkl=args.data_pkl, cols=cols,
    #      train_start=args.train_start, train_end=args.train_end,
    #      test_start=args.test_start, test_end=args.test_end
    #      )
    if not args.frontier:
        train(args.mode, args.iters, args.seed, Path(args.outdir), args.alpha_theta, args.alpha_phi, args.alpha_w,
              args.Lambda, args.omega_update_every, episode_T_years=args.T, dt=args.dt, a_max=args.a_max,
              data_pkl=args.data_pkl, cols=cols,
              train_start=args.train_start, train_end=args.train_end,
              test_start=args.test_start, test_end=args.test_end,
              z_target=args.z)
        return

    # --- Frontier mode: loop over z_list, train+eval, save frontier_rl.csv/png ---
    z_list = _parse_z_list(args.z_list)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    frontier_rows = []

    for z in z_list:
        subdir = outdir / f"z_{z:.4f}"
        agent, filt_params, df_train, df_test, true_params = train(
            args.mode, args.iters, args.seed, subdir, args.alpha_theta, args.alpha_phi, args.alpha_w,
            args.Lambda, args.omega_update_every, episode_T_years=args.T, dt=args.dt, a_max=args.a_max,
            data_pkl=args.data_pkl, cols=cols,
            train_start=args.train_start, train_end=args.train_end,
            test_start=args.test_start, test_end=args.test_end,
            z_target=float(z)
        )

        n_steps = int(round(args.T / args.dt))
        # eval dataset:
        if df_test is not None:
            logret_eval = df_test.values.astype(float)
            logret_df_eval = df_test
        else:
            # simulation: generate a long logret tape and sample windows from it
            S, _ = simulate_price_only(true_params, T_years=max(30.0, args.T*5), dt=args.dt, s0=1.0, seed=args.seed+123)
            logret_eval = np.log(S[1:] / S[:-1])
            #logret_df_eval = pd.DataFrame(logret_eval)
            # IMPORTANT: attach a datetime index so monthly rebalance works (otherwise falls back to EW)
            idx = pd.date_range(start="2000-01-03", periods=len(logret_eval), freq="B")
            logret_df_eval = pd.DataFrame(logret_eval, index=idx)

        xT_rl = _eval_rl_on_env(agent, filt_params, logret_eval, dt=args.dt, n_steps=n_steps,
                                a_max=args.a_max, n_eval=args.n_eval, seed0=args.seed+999)
        m_rl = float(xT_rl.mean()); s_rl = float(xT_rl.std(ddof=1))

        bench = _eval_bench_monthly_on_logret(logret_df_eval, dt=args.dt, n_steps=n_steps, n_eval=args.n_eval,
                                              seed0=args.seed+1999, lookback_days=args.lookback_days)
        frontier_rows.append({
            "z": float(z),
            "RL_mean": m_rl, "RL_std": s_rl,
            "MV_minvar_mean": float(bench["MV_minvar"].mean()), "MV_minvar_std": float(bench["MV_minvar"].std(ddof=1)),
            "MV_tangency_mean": float(bench["MV_tangency"].mean()), "MV_tangency_std": float(bench["MV_tangency"].std(ddof=1)),
            "EW_mean": float(bench["EW"].mean()), "EW_std": float(bench["EW"].std(ddof=1)),
        })

    frontier = pd.DataFrame(frontier_rows)
    frontier.to_csv(outdir/"frontier_rl.csv", index=False)

    # plot: std on x, mean on y (classic frontier)
    fig = plt.figure()
    plt.plot(frontier["RL_std"], frontier["RL_mean"], marker="o", label="RL (vary z)")
    # add benchmark reference points (use average across z; they don't depend on z structurally)
    plt.scatter([frontier["MV_minvar_std"].mean()], [frontier["MV_minvar_mean"].mean()], label="MV(minvar, monthly)")
    plt.scatter([frontier["MV_tangency_std"].mean()], [frontier["MV_tangency_mean"].mean()], label="MV(tangency, monthly)")
    plt.scatter([frontier["EW_std"].mean()], [frontier["EW_mean"].mean()], label="EW(monthly)")
    plt.xlabel("Std[X_T]")
    plt.ylabel("E[X_T]")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir/"frontier.png", dpi=200)
    plt.close(fig)

if __name__ == "__main__":
    main()
