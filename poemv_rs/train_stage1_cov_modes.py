from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .agent_stage1_cov_modes import POEMVAgentCovModes, TrainConfigCovModes
from .env_regime_cov import RSGBMEnvRegimeCov, RSGBMParamsRegimeCov, EpisodeConfigRegimeCov
from .filtering_regime_cov import FilterParamsRegimeCov, hmm_filter_regime_cov_q_update


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def default_sigma1() -> np.ndarray:
    return np.array(
        [[0.22**2, 0.22 * 0.18 * 0.30],
         [0.22 * 0.18 * 0.30, 0.18**2]],
        dtype=float,
    )


def default_sigma2() -> np.ndarray:
    return np.array(
        [[0.30**2, 0.30 * 0.24 * 0.55],
         [0.30 * 0.24 * 0.55, 0.24**2]],
        dtype=float,
    )


def default_true_params(r: float = 0.01, sigma_mode: str = "regime") -> RSGBMParamsRegimeCov:
    s1 = default_sigma1()
    s2 = default_sigma2() if sigma_mode == "regime" else s1.copy()
    return RSGBMParamsRegimeCov(
        mu1=np.array([0.25, 0.18], dtype=float),
        mu2=np.array([-0.73, -0.40], dtype=float),
        Sigma1=s1,
        Sigma2=s2,
        lam1=0.36,
        lam2=2.89,
        r=r,
    )


def parse_json_array(s: str | None, fallback: np.ndarray) -> np.ndarray:
    if s is None:
        return np.asarray(fallback, dtype=float)
    arr = np.asarray(json.loads(s), dtype=float)
    return arr


def _build_true_params_from_args(args: argparse.Namespace) -> RSGBMParamsRegimeCov:
    base = default_true_params(r=args.r, sigma_mode=args.sigma_mode)
    mu1 = parse_json_array(args.mu1_json, base.mu1)
    mu2 = parse_json_array(args.mu2_json, base.mu2)
    sigma1 = parse_json_array(args.sigma1_json, base.Sigma1)
    sigma2 = parse_json_array(args.sigma2_json, base.Sigma2)
    return RSGBMParamsRegimeCov(
        mu1=mu1,
        mu2=mu2,
        Sigma1=sigma1,
        Sigma2=sigma2,
        lam1=float(args.lam1),
        lam2=float(args.lam2),
        r=float(args.r),
    )


def _build_estimated_filter_params_from_args(args: argparse.Namespace, true_params: RSGBMParamsRegimeCov) -> FilterParamsRegimeCov:
    if args.belief_mode == "true_params":
        return FilterParamsRegimeCov(
            mu1=true_params.mu1.copy(),
            mu2=true_params.mu2.copy(),
            Sigma1=true_params.Sigma1.copy(),
            Sigma2=true_params.Sigma2.copy(),
            lam1=true_params.lam1,
            lam2=true_params.lam2,
            r=true_params.r,
        )

    mu1 = parse_json_array(args.est_mu1_json, true_params.mu1)
    mu2 = parse_json_array(args.est_mu2_json, true_params.mu2)
    sigma1 = parse_json_array(args.est_sigma1_json, true_params.Sigma1)
    sigma2 = parse_json_array(args.est_sigma2_json, true_params.Sigma2)
    lam1 = float(args.est_lam1 if args.est_lam1 is not None else true_params.lam1)
    lam2 = float(args.est_lam2 if args.est_lam2 is not None else true_params.lam2)
    r = float(args.est_r if args.est_r is not None else true_params.r)
    return FilterParamsRegimeCov(
        mu1=mu1,
        mu2=mu2,
        Sigma1=sigma1,
        Sigma2=sigma2,
        lam1=lam1,
        lam2=lam2,
        r=r,
    )


def params_from_args(args: argparse.Namespace) -> Tuple[RSGBMParamsRegimeCov, FilterParamsRegimeCov]:
    true_params = _build_true_params_from_args(args)
    filt_params = _build_estimated_filter_params_from_args(args, true_params)
    return true_params, filt_params


def build_train_config(args: argparse.Namespace, true_params: RSGBMParamsRegimeCov) -> TrainConfigCovModes:
    if args.cov_mode == "common":
        sigma_common = 0.5 * (true_params.Sigma1 + true_params.Sigma2)
        sigma1 = None
        sigma2 = None
    else:
        sigma_common = None
        sigma1 = true_params.Sigma1.copy()
        sigma2 = true_params.Sigma2.copy()

    return TrainConfigCovModes(
        T_years=args.T,
        dt=args.dt,
        x0=args.x0,
        p0=args.p0,
        a_max=args.a_max,
        cap_mode=args.cap_mode,
        z=args.z,
        Lambda=args.Lambda,
        r=args.r,
        mu1=true_params.mu1.copy(),
        mu2=true_params.mu2.copy(),
        Sigma=sigma_common,
        Sigma1=sigma1,
        Sigma2=sigma2,
        cov_mode=args.cov_mode,
        m_poly=args.m_poly,
        mg_poly=args.mg_poly,
        g_p_dep=args.g_p_dep,
        actor_mix_tail=args.actor_mix_tail,
        cov_scale=args.cov_scale,
        lr_step_every=args.lr_step_every,
        lr_gamma=args.lr_gamma,
        alpha_w=args.alpha_w,
        alpha_theta=args.alpha_theta,
        alpha_phi=args.alpha_phi,
        omega_init=args.omega_init,
        omega_update_every=args.omega_update_every,
        critic_steps=args.critic_steps,
        omega_ema_beta=args.omega_ema_beta,
        grad_clip=args.grad_clip,
        device=args.device,
        dtype=torch.float64,
        episodes_per_iter=args.episodes_per_iter,
    )


def rollout_episode(
    agent: POEMVAgentCovModes,
    true_params: RSGBMParamsRegimeCov,
    filt_params: FilterParamsRegimeCov,
    ep_cfg: EpisodeConfigRegimeCov,
    seed: int,
    deterministic: bool = False,
) -> Dict[str, np.ndarray]:
    env = RSGBMEnvRegimeCov(true_params, EpisodeConfigRegimeCov(
        T_years=ep_cfg.T_years,
        dt=ep_cfg.dt,
        x0=ep_cfg.x0,
        s0=np.asarray(ep_cfg.s0, dtype=float).copy(),
        p0=ep_cfg.p0,
        a_max=ep_cfg.a_max,
        seed=seed,
        apply_action_projection=ep_cfg.apply_action_projection,
    ))
    obs = env.reset()
    n = env.n_steps

    t_hist: List[float] = [float(obs["t"])]
    x_hist: List[float] = [float(obs["X"])]
    p_hist: List[float] = [float(ep_cfg.p0)]
    s_hist: List[np.ndarray] = [np.asarray(obs["S"], dtype=float).copy()]
    i_hist: List[int] = [int(obs["I_true"])]
    u_raw_hist: List[np.ndarray] = []
    u_exec_hist: List[np.ndarray] = []
    f_roll_hist: List[float] = []
    dlnf_roll_hist: List[float] = []
    omega_roll_hist: List[float] = []
    innov_hist: List[float] = []

    p_curr = float(ep_cfg.p0)
    prev_S = np.asarray(obs["S"], dtype=float).copy()

    for _k in range(n):
        t_k = float(obs["t"])
        x_k = float(obs["X"])
        u_exec, u_raw, info = agent.act(t_k, x_k, p_curr, deterministic=deterministic)
        obs_next, _disc_ret, done = env.step(u_exec)
        next_S = np.asarray(obs_next["S"], dtype=float).copy()
        logret = np.log(next_S / prev_S)
        p_next, innov = hmm_filter_regime_cov_q_update(p_curr, logret, ep_cfg.dt, filt_params)

        u_raw_hist.append(np.asarray(u_raw, dtype=float).copy())
        u_exec_hist.append(np.asarray(u_exec, dtype=float).copy())
        f_roll_hist.append(float(info["f_roll"]))
        dlnf_roll_hist.append(float(info["dlnf_roll"]))
        omega_roll_hist.append(float(info["omega_roll"]))
        innov_hist.append(float(innov))

        t_hist.append(float(obs_next["t"]))
        x_hist.append(float(obs_next["X"]))
        p_hist.append(float(p_next))
        s_hist.append(next_S)
        i_hist.append(int(obs_next["I_true"]))

        obs = obs_next
        prev_S = next_S
        p_curr = float(p_next)
        if done:
            break

    return {
        "t": np.asarray(t_hist, dtype=float),
        "x": np.asarray(x_hist, dtype=float),
        "p": np.asarray(p_hist, dtype=float),
        "S": np.asarray(s_hist, dtype=float),
        "I_true": np.asarray(i_hist, dtype=int),
        "u_raw": np.asarray(u_raw_hist, dtype=float),
        "u_exec": np.asarray(u_exec_hist, dtype=float),
        "f_roll": np.asarray(f_roll_hist, dtype=float),
        "dlnf_roll": np.asarray(dlnf_roll_hist, dtype=float),
        "omega_roll": np.asarray(omega_roll_hist, dtype=float),
        "innovation": np.asarray(innov_hist, dtype=float),
    }


def evaluate_agent(
    agent: POEMVAgentCovModes,
    true_params: RSGBMParamsRegimeCov,
    filt_params: FilterParamsRegimeCov,
    ep_cfg: EpisodeConfigRegimeCov,
    seed0: int,
    n_paths: int,
) -> Dict[str, float]:
    xT = []
    sharpe = []
    for j in range(n_paths):
        traj = rollout_episode(agent, true_params, filt_params, ep_cfg, seed=seed0 + j, deterministic=True)
        wealth = traj["x"]
        xT.append(float(wealth[-1]))
        ret = np.diff(wealth) / np.maximum(np.abs(wealth[:-1]), 1e-12)
        if len(ret) > 1 and np.std(ret, ddof=1) > 1e-12:
            sharpe.append(float(np.sqrt(252.0) * np.mean(ret) / np.std(ret, ddof=1)))
        else:
            sharpe.append(np.nan)
    xT_arr = np.asarray(xT, dtype=float)
    return {
        "val_mean_xT": float(np.mean(xT_arr)),
        "val_std_xT": float(np.std(xT_arr, ddof=1)) if len(xT_arr) > 1 else 0.0,
        "val_mean_sharpe": float(np.nanmean(np.asarray(sharpe, dtype=float))),
    }


def save_checkpoint(agent: POEMVAgentCovModes, outdir: Path, name: str = "checkpoint.pt") -> Path:
    ckpt = {
        "vf_state_dict": agent.vf.state_dict(),
        "pi_state_dict": agent.pi.state_dict(),
        "omega": float(agent.omega.detach().cpu().item()),
        "policy_values": agent.policy_values(),
        "cov_mode": str(agent.cfg.cov_mode),
    }
    path = outdir / name
    torch.save(ckpt, path)
    return path


def save_learning_curves(metrics: pd.DataFrame, outdir: Path) -> None:
    if metrics.empty:
        return

    fig = plt.figure(figsize=(9, 6))
    plt.plot(metrics["iter"], metrics["mean_xT_batch"], label="train batch mean terminal wealth")
    if "val_mean_xT" in metrics.columns:
        plt.plot(metrics["iter"], metrics["val_mean_xT"], label="validation mean terminal wealth")
    plt.axhline(float(metrics["target_z"].iloc[0]), linestyle="--", linewidth=1.2, label="target z")
    plt.xlabel("iteration")
    plt.ylabel("terminal wealth")
    plt.title("Stage1 terminal wealth learning curve")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "learning_curve.png", dpi=180)
    plt.close(fig)

    fig = plt.figure(figsize=(9, 6))
    plt.plot(metrics["iter"], metrics["loss_critic"], label="critic loss")
    plt.plot(metrics["iter"], metrics["loss_actor"], label="actor loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("Stage1 losses")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_curve.png", dpi=180)
    plt.close(fig)


def save_config(outdir: Path, args: argparse.Namespace, true_params: RSGBMParamsRegimeCov, filt_params: FilterParamsRegimeCov) -> None:
    cfg = vars(args).copy()
    cfg["true_params"] = {
        "mu1": true_params.mu1.tolist(),
        "mu2": true_params.mu2.tolist(),
        "Sigma1": true_params.Sigma1.tolist(),
        "Sigma2": true_params.Sigma2.tolist(),
        "lam1": float(true_params.lam1),
        "lam2": float(true_params.lam2),
        "r": float(true_params.r),
    }
    cfg["estimated_params"] = {
        "mu1": filt_params.mu1.tolist(),
        "mu2": filt_params.mu2.tolist(),
        "Sigma1": filt_params.Sigma1.tolist(),
        "Sigma2": filt_params.Sigma2.tolist(),
        "lam1": float(filt_params.lam1),
        "lam2": float(filt_params.lam2),
        "r": float(filt_params.r),
    }
    cfg["filter_params"] = cfg["estimated_params"]
    cfg["training_modes"] = {
        "sigma_mode": str(getattr(args, "sigma_mode", "regime")),
        "cov_mode": str(getattr(args, "cov_mode", "belief_avg_regime")),
        "belief_mode": str(getattr(args, "belief_mode", "true_params")),
    }
    estimation_method = getattr(args, "estimation_method", None)
    cfg["estimation"] = {
        "estimation_method": estimation_method,
        "jump_penalty": getattr(args, "jump_penalty", None),
        "jm_max_iter": getattr(args, "jm_max_iter", None),
        "jm_n_init": getattr(args, "jm_n_init", None),
        "jm_params_json": getattr(args, "jm_params_json", None),
        "jm_regimes_csv": getattr(args, "jm_regimes_csv", None),
        "index_mode": getattr(args, "index_mode", None),
        "Y_l": getattr(args, "Y_l", None),
        "Y_u": getattr(args, "Y_u", None),
    }
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--episodes_per_iter", type=int, default=16)
    ap.add_argument("--val_every", type=int, default=50)
    ap.add_argument("--val_n_paths", type=int, default=64)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=1/252)
    ap.add_argument("--x0", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.5)
    ap.add_argument("--z", type=float, default=1.2)
    ap.add_argument("--Lambda", type=float, default=1.0)
    ap.add_argument("--r", type=float, default=0.01)
    ap.add_argument("--a_max", type=float, default=2.0)
    ap.add_argument("--cap_mode", type=str, default="component_tanh", choices=["none", "component_tanh", "l1_radial"])
    ap.add_argument("--sigma_mode", type=str, default="regime", choices=["common", "regime"])
    ap.add_argument("--cov_mode", type=str, default="belief_avg_regime", choices=["common", "belief_avg_regime"])
    ap.add_argument("--mu1_json", type=str, default=None)
    ap.add_argument("--mu2_json", type=str, default=None)
    ap.add_argument("--sigma1_json", type=str, default=None)
    ap.add_argument("--sigma2_json", type=str, default=None)
    ap.add_argument("--belief_mode", type=str, default="true_params", choices=["true_params", "estimated_params"])
    ap.add_argument("--est_mu1_json", type=str, default=None)
    ap.add_argument("--est_mu2_json", type=str, default=None)
    ap.add_argument("--est_sigma1_json", type=str, default=None)
    ap.add_argument("--est_sigma2_json", type=str, default=None)
    ap.add_argument("--lam1", type=float, default=0.36)
    ap.add_argument("--lam2", type=float, default=2.89)
    ap.add_argument("--est_lam1", type=float, default=None)
    ap.add_argument("--est_lam2", type=float, default=None)
    ap.add_argument("--est_r", type=float, default=None)
    ap.add_argument("--m_poly", type=int, default=2)
    ap.add_argument("--mg_poly", type=int, default=2)
    ap.add_argument("--g_p_dep", action="store_true")
    ap.add_argument("--actor_mix_tail", type=float, default=0.5)
    ap.add_argument("--cov_scale", type=float, default=1.0)
    ap.add_argument("--alpha_w", type=float, default=5e-3)
    ap.add_argument("--alpha_theta", type=float, default=3e-5)
    ap.add_argument("--alpha_phi", type=float, default=1e-4)
    ap.add_argument("--omega_init", type=float, default=0.0)
    ap.add_argument("--omega_update_every", type=int, default=10)
    ap.add_argument("--critic_steps", type=int, default=10)
    ap.add_argument("--omega_ema_beta", type=float, default=0.9)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lr_step_every", type=int, default=0)
    ap.add_argument("--lr_gamma", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    true_params, filt_params = params_from_args(args)
    train_cfg = build_train_config(args, true_params)
    agent = POEMVAgentCovModes(train_cfg)

    ep_cfg = EpisodeConfigRegimeCov(
        T_years=args.T,
        dt=args.dt,
        x0=args.x0,
        s0=np.array([1.0, 1.0], dtype=float),
        p0=args.p0,
        a_max=args.a_max,
        seed=args.seed,
        apply_action_projection=True,
    )

    save_config(outdir, args, true_params, filt_params)

    rows: List[Dict[str, float]] = []
    best_val = -np.inf

    for it in range(1, args.iters + 1):
        trajs = []
        batch_xT = []
        for j in range(args.episodes_per_iter):
            traj = rollout_episode(
                agent=agent,
                true_params=true_params,
                filt_params=filt_params,
                ep_cfg=ep_cfg,
                seed=args.seed + 100000 * it + j,
                deterministic=False,
            )
            trajs.append(traj)
            batch_xT.append(float(traj["x"][-1]))

        loss_c, loss_a = agent.update_from_episodes(trajs)
        if (it % int(args.omega_update_every)) == 0:
            agent.update_omega(float(np.mean(batch_xT)))
        agent.step_schedulers()

        row: Dict[str, float] = {
            "iter": float(it),
            "loss_critic": float(loss_c),
            "loss_actor": float(loss_a),
            "mean_xT_batch": float(np.mean(batch_xT)),
            "std_xT_batch": float(np.std(batch_xT, ddof=1)) if len(batch_xT) > 1 else 0.0,
            "omega": float(agent.omega.detach().cpu().item()),
            "target_z": float(args.z),
            "sigma_mode": str(args.sigma_mode),
            "cov_mode": str(args.cov_mode),
            "belief_mode": str(args.belief_mode),
            "estimation_method": str(getattr(args, "estimation_method", "none")),
            "jump_penalty": float(getattr(args, "jump_penalty", np.nan)) if getattr(args, "jump_penalty", None) is not None else np.nan,
            "jm_max_iter": float(getattr(args, "jm_max_iter", np.nan)) if getattr(args, "jm_max_iter", None) is not None else np.nan,
            "jm_n_init": float(getattr(args, "jm_n_init", np.nan)) if getattr(args, "jm_n_init", None) is not None else np.nan,
        }
        row.update(agent.current_lrs())
        row.update(agent.policy_values())

        if (it % args.val_every) == 0 or it == 1 or it == args.iters:
            val = evaluate_agent(
                agent=agent,
                true_params=true_params,
                filt_params=filt_params,
                ep_cfg=ep_cfg,
                seed0=args.seed + 500000 + it,
                n_paths=args.val_n_paths,
            )
            row.update(val)
            if val["val_mean_xT"] > best_val:
                best_val = val["val_mean_xT"]
                save_checkpoint(agent, outdir, name="best_checkpoint.pt")

        rows.append(row)
        if it % max(1, args.val_every) == 0:
            print(
                f"iter={it} loss_c={row['loss_critic']:.6f} loss_a={row['loss_actor']:.6f} "
                f"mean_xT={row['mean_xT_batch']:.4f} omega={row['omega']:.4f} "
                f"belief_mode={args.belief_mode} cov_mode={args.cov_mode} "
                f"estimation_method={getattr(args, 'estimation_method', 'none')} "
                f"jm_penalty={getattr(args, 'jump_penalty', 'na')} jm_n_init={getattr(args, 'jm_n_init', 'na')} jm_max_iter={getattr(args, 'jm_max_iter', 'na')}"
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "metrics.csv", index=False)
    save_learning_curves(metrics, outdir)
    save_checkpoint(agent, outdir, name="checkpoint.pt")
    print(f"Saved outputs to {outdir}")


if __name__ == "__main__":
    main()
