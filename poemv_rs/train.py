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

def make_estimated_params_via_heuristic(true_params: RSGBMParams, seed=0):
    # Simulate 30y, use first 20y to estimate, per paper description.
    dt = 1/252
    S, _ = simulate_price_only(true_params, T_years=30.0, dt=dt, s0=1.0, seed=seed)
    S_train = S[: int(round(20.0/dt))+1]  # (T,2)
    # label using asset-1 price (minimal). You can replace with equal-weight index.
    reg = label_bull_bear_from_drawdowns(S_train[:,0], HeuristicThresholds())
    ret_train = (S_train[1:] - S_train[:-1]) / S_train[:-1]  # (T-1,2)
    est = estimate_env_params_from_labeled_returns(ret_train, reg, dt=dt)
    return _to_common_sigma_params(RSGBMParams(**est, r=true_params.r))

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
          episodes_per_iter: int = 1,
          apply_action_projection: bool = True,
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
        est_params = make_estimated_params_via_heuristic(true_params, seed=seed)
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

            p = cfg.p0
            t_arr[0]=0.0; x_arr[0]=cfg.x0; p_arr[0]=p
            S_prev = obs["S"]
            u_raw_arr = np.empty((n,2))

            for k in range(n):
                u, u_raw, _ = agent.act(t_arr[k], x_arr[k], p, deterministic=False)
                u_raw_arr[k] = np.asarray(u_raw, float)
                u_arr[k] = np.asarray(u, float)
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

            traj = dict(t=t_arr, x=x_arr, p=p_arr, u=u_arr, u_raw=u_raw_arr, a_max=a_max)
            batch_trajs.append(traj)
            batch_pos_stats.append(_episode_position_stats(traj))
            batch_xT.append(float(x_arr[-1]))

        loss_c, loss_a = agent.update_from_episodes(batch_trajs)

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
    ap.add_argument("--episodes_per_iter", type=int, default=32)
    ap.add_argument("--z", type=float, default=1.2)
    ap.add_argument("--apply_action_projection", action="store_true")
    args = ap.parse_args()
    train(args.mode, args.iters, args.seed, Path(args.outdir), args.alpha_theta, args.alpha_phi, args.alpha_w,
          args.Lambda, args.omega_update_every, episode_T_years=args.T, dt=args.dt,z = args.z, a_max=args.a_max,
          r=args.r,
          critic_steps=args.critic_steps,
          advantage_norm_eps=args.advantage_norm_eps,
          omega_ema_beta=args.omega_ema_beta,
          episodes_per_iter=args.episodes_per_iter,
          cap_mode=args.cap_mode,
          apply_action_projection=args.apply_action_projection,
          )

if __name__ == "__main__":
    main()
