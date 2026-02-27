from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import math
from pathlib import Path
import matplotlib.pyplot as plt

from .env import RSGBMParams, EpisodeConfig, RSGBMEnv, simulate_price_only
from .filtering import FilterParams, wonham_filter_q_update
from .heuristic import HeuristicThresholds, label_bull_bear_from_drawdowns, estimate_env_params_from_labeled_returns
from .agent import POEMVAgent, TrainConfig
from .utils import set_seed, safe_clip_p

def run_filter_demo(outdir: Path, true_params: RSGBMParams, est_params: RSGBMParams, T_years=10.0, dt=1/252, seed=0):
    S, I = simulate_price_only(true_params, T_years=T_years, dt=dt, s0=1.0, seed=seed)
    logret = np.log(S[1:] / S[:-1])
    p_true = np.empty(len(S)); p_est = np.empty(len(S))
    p_true[0]=0.5; p_est[0]=0.5

    fp_true = FilterParams(mu1=true_params.mu1, mu2=true_params.mu2, sigma=true_params.sigma, lam1=true_params.lam1, lam2=true_params.lam2, r=true_params.r)
    fp_est  = FilterParams(mu1=est_params.mu1,  mu2=est_params.mu2,  sigma=est_params.sigma,  lam1=est_params.lam1,  lam2=est_params.lam2,  r=est_params.r)

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
    S_train = S[: int(round(20.0/dt))+1]
    ret_train = (S_train[1:] - S_train[:-1]) / S_train[:-1]
    reg = label_bull_bear_from_drawdowns(S_train, HeuristicThresholds())
    est = estimate_env_params_from_labeled_returns(ret_train, reg, dt=dt)
    return RSGBMParams(**est, r=true_params.r)

def train(mode: str, iters: int, seed: int, outdir: Path,
          alpha_theta: float, alpha_phi: float, alpha_w: float,
          Lambda: float, omega_update_every: int,
          episode_T_years: float = 10.0, dt: float = 1/252, a_max: float = 2.0):

    set_seed(seed)
    outdir.mkdir(parents=True, exist_ok=True)

    # paper's simulation "true" parameters
    true_params = RSGBMParams(mu1=0.25, mu2=-0.73, sigma=0.22, lam1=0.36, lam2=2.89, r=0.0)

    if mode == "true_params":
        filt_params = FilterParams(mu1=true_params.mu1, mu2=true_params.mu2, sigma=true_params.sigma, lam1=true_params.lam1, lam2=true_params.lam2, r=true_params.r)
        est_params = true_params
    elif mode == "estimated_params":
        est_params = make_estimated_params_via_heuristic(true_params, seed=seed)
        filt_params = FilterParams(mu1=est_params.mu1, mu2=est_params.mu2, sigma=est_params.sigma, lam1=est_params.lam1, lam2=est_params.lam2, r=est_params.r)
    else:
        raise ValueError("mode must be true_params or estimated_params")

    # demo plot for filtering
    run_filter_demo(outdir, true_params=true_params, est_params=est_params, seed=seed)

    cfg = TrainConfig(T_years=episode_T_years, dt=dt, x0=1.0, s0=1.0, p0=0.5, z=1.5,
                      Lambda=Lambda, alpha_theta=alpha_theta, alpha_phi=alpha_phi, alpha_w=alpha_w,
                      omega_update_every=omega_update_every, a_max = a_max)
    agent = POEMVAgent(cfg)

    rows = []
    last_terminals = []
    last_mean_xT_window = float("nan")
    last_gap = float("nan")
    last_domega = 0.0

    for m in range(1, iters+1):
        # One episode: simulate environment with TRUE parameters (market), but filter with selected parameters.
        env = RSGBMEnv(true_params, EpisodeConfig(
            T_years=episode_T_years, dt=dt, x0=cfg.x0, s0=cfg.s0, p0=cfg.p0,
            a_max=a_max,
            omega=float(agent.omega.detach().cpu()), seed=seed + m
        ))
        obs = env.reset()

        n = env.n_steps
        t_arr = np.empty(n+1); x_arr = np.empty(n+1); p_arr = np.empty(n+1)
        #logp_arr = np.empty(n); ent_arr = np.empty(n)
        u_arr = np.empty(n)

        p = cfg.p0
        t_arr[0]=0.0; x_arr[0]=cfg.x0; p_arr[0]=p
        S_prev = obs["S"]
        u_raw_arr = np.empty(n)

        for k in range(n):
            u, u_raw, _ = agent.act(t_arr[k], x_arr[k], p)
            u_raw_arr[k] = float(u_raw)
            #u, logprob, entropy = agent.act(t_arr[k], x_arr[k], p)
            u_arr[k] = float(u)
            obs, _, done = env.step(u)
            S_now = obs["S"]
            log_return = float(np.log(S_now / S_prev))
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
        json.dump(dict(mode=mode, seed=seed, iters=iters, Lambda=Lambda,
                       alpha_theta=alpha_theta, alpha_phi=alpha_phi, alpha_w=alpha_w,
                       omega_update_every=omega_update_every,
                       true_params=true_params.__dict__, filter_params=filt_params.__dict__,
                       estimated_params=est_params.__dict__), f, indent=2)

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
    args = ap.parse_args()
    train(args.mode, args.iters, args.seed, Path(args.outdir), args.alpha_theta, args.alpha_phi, args.alpha_w,
          args.Lambda, args.omega_update_every, episode_T_years=args.T, dt=args.dt,a_max=args.a_max)

if __name__ == "__main__":
    main()
