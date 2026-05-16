from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .agent_stage1_cov_modes import POEMVAgentCovModes, TrainConfigCovModes
from .env_regime_cov import RSGBMParamsRegimeCov, EpisodeConfigRegimeCov, RSGBMEnvRegimeCov
from .filtering_regime_cov import FilterParamsRegimeCov, hmm_filter_regime_cov_q_update


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------
# params / checkpoint loading
# ---------------------------

def default_true_params(r: float = 0.01, sigma_mode: str = "regime") -> RSGBMParamsRegimeCov:
    sigma1 = np.array(
        [[0.22**2, 0.22 * 0.18 * 0.30],
         [0.22 * 0.18 * 0.30, 0.18**2]],
        dtype=float,
    )
    sigma2 = np.array(
        [[0.30**2, 0.30 * 0.24 * 0.55],
         [0.30 * 0.24 * 0.55, 0.24**2]],
        dtype=float,
    ) if sigma_mode == "regime" else sigma1.copy()
    return RSGBMParamsRegimeCov(
        mu1=np.array([0.25, 0.18], dtype=float),
        mu2=np.array([-0.73, -0.40], dtype=float),
        Sigma1=sigma1,
        Sigma2=sigma2,
        lam1=0.36,
        lam2=2.89,
        r=r,
    )


def load_run_config(run_dir: Path) -> Dict:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.json not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(ckpt_path: Path, device: str = "cpu") -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict.")
    return ckpt


def _params_from_dict(d: Dict, fallback: Optional[RSGBMParamsRegimeCov] = None) -> Dict:
    if fallback is None:
        fallback = default_true_params()
    src = d or {}
    return dict(
        mu1=np.asarray(src.get("mu1", fallback.mu1), dtype=float),
        mu2=np.asarray(src.get("mu2", fallback.mu2), dtype=float),
        Sigma1=np.asarray(src.get("Sigma1", fallback.Sigma1), dtype=float),
        Sigma2=np.asarray(src.get("Sigma2", fallback.Sigma2), dtype=float),
        lam1=float(src.get("lam1", fallback.lam1)),
        lam2=float(src.get("lam2", fallback.lam2)),
        r=float(src.get("r", fallback.r)),
    )


def load_true_and_filter_params(run_dir: Path, belief_mode: str, r: float) -> Tuple[RSGBMParamsRegimeCov, FilterParamsRegimeCov, Dict]:
    run_cfg = load_run_config(run_dir)
    sigma_mode = str(run_cfg.get("sigma_mode", "regime"))
    fallback = default_true_params(r=r, sigma_mode=sigma_mode)

    true_src = _params_from_dict(run_cfg.get("true_params", {}), fallback=fallback)
    true_params = RSGBMParamsRegimeCov(
        mu1=true_src["mu1"],
        mu2=true_src["mu2"],
        Sigma1=true_src["Sigma1"],
        Sigma2=true_src["Sigma2"],
        lam1=true_src["lam1"],
        lam2=true_src["lam2"],
        r=true_src["r"],
    )

    filt_key = "estimated_params" if belief_mode == "estimated_params" else "true_params"
    filt_src = _params_from_dict(run_cfg.get(filt_key, {}), fallback=true_params)
    filt_params = FilterParamsRegimeCov(
        mu1=filt_src["mu1"],
        mu2=filt_src["mu2"],
        Sigma1=filt_src["Sigma1"],
        Sigma2=filt_src["Sigma2"],
        lam1=filt_src["lam1"],
        lam2=filt_src["lam2"],
        r=filt_src["r"],
    )
    return true_params, filt_params, run_cfg


# ---------------------------
# agent reconstruction
# ---------------------------

def build_agent_from_checkpoint_cov_modes(
    ckpt_path: Path,
    run_dir: Path,
    T_years: float,
    dt: float,
    a_max: float,
    z: float,
    r: float,
    device: str = "cpu",
) -> Tuple[POEMVAgentCovModes, Dict]:
    run_cfg = load_run_config(run_dir)
    ckpt = load_checkpoint(ckpt_path, device=device)

    sigma_mode = str(run_cfg.get("sigma_mode", "regime"))
    default_params = default_true_params(r=r, sigma_mode=sigma_mode)
    src = _params_from_dict(run_cfg.get("true_params", {}), fallback=default_params)

    cov_mode = str(run_cfg.get("cov_mode", run_cfg.get("training_modes", {}).get("cov_mode", "belief_avg_regime")))
    sigma_common = None
    if cov_mode == "common":
        sigma_common = 0.5 * (src["Sigma1"] + src["Sigma2"])
        sigma1 = None
        sigma2 = None
    else:
        sigma1 = src["Sigma1"]
        sigma2 = src["Sigma2"]

    cfg = TrainConfigCovModes(
        T_years=float(run_cfg.get("T", T_years)),
        dt=float(run_cfg.get("dt", dt)),
        x0=float(run_cfg.get("x0", 1.0)),
        p0=float(run_cfg.get("p0", 0.5)),
        a_max=float(run_cfg.get("a_max", a_max)),
        cap_mode=str(run_cfg.get("cap_mode", "component_tanh")),
        z=float(run_cfg.get("z", z)),
        Lambda=float(run_cfg.get("Lambda", 1.0)),
        r=float(run_cfg.get("r", r)),
        mu1=src["mu1"],
        mu2=src["mu2"],
        Sigma=sigma_common,
        Sigma1=sigma1,
        Sigma2=sigma2,
        cov_mode=cov_mode,
        m_poly=int(run_cfg.get("m_poly", 2)),
        mg_poly=int(run_cfg.get("mg_poly", 2)),
        g_p_dep=bool(run_cfg.get("g_p_dep", False)),
        actor_mix_tail=float(run_cfg.get("actor_mix_tail", 0.5)),
        cov_scale=float(run_cfg.get("cov_scale", 1.0)),
        lr_step_every=int(run_cfg.get("lr_step_every", 0)),
        lr_gamma=float(run_cfg.get("lr_gamma", 1.0)),
        alpha_w=float(run_cfg.get("alpha_w", 1e-3)),
        alpha_theta=float(run_cfg.get("alpha_theta", 3e-5)),
        alpha_phi=float(run_cfg.get("alpha_phi", 1e-4)),
        omega_init=float(run_cfg.get("omega_init", 0.0)),
        omega_update_every=int(run_cfg.get("omega_update_every", 10)),
        critic_steps=int(run_cfg.get("critic_steps", 10)),
        omega_ema_beta=float(run_cfg.get("omega_ema_beta", 0.9)),
        grad_clip=float(run_cfg.get("grad_clip", 1.0)),
        device=device,
        dtype=torch.float64,
        episodes_per_iter=int(run_cfg.get("episodes_per_iter", 16)),
    )
    agent = POEMVAgentCovModes(cfg)

    vf_sd = ckpt.get("vf_state_dict") or ckpt.get("vf") or ckpt.get("value_state_dict")
    pi_sd = ckpt.get("pi_state_dict") or ckpt.get("pi") or ckpt.get("policy_state_dict")
    omega = ckpt.get("omega")
    if vf_sd is None or pi_sd is None:
        raise KeyError("Checkpoint missing vf_state_dict / pi_state_dict.")

    agent.vf.load_state_dict(vf_sd)
    agent.pi.load_state_dict(pi_sd)
    if omega is not None:
        with torch.no_grad():
            agent.omega = torch.as_tensor(float(omega), dtype=cfg.dtype, device=cfg.device)
    agent.vf.eval()
    agent.pi.eval()
    return agent, run_cfg


# ---------------------------
# baselines
# ---------------------------

def stationary_bull_prob(params: RSGBMParamsRegimeCov) -> float:
    return float(params.lam2 / max(params.lam1 + params.lam2, 1e-12))


def unconditional_moments_regime_cov(params: RSGBMParamsRegimeCov) -> Tuple[np.ndarray, np.ndarray]:
    pbull = stationary_bull_prob(params)
    mu_bar = pbull * np.asarray(params.mu1, float) + (1.0 - pbull) * np.asarray(params.mu2, float)
    sigma_bar = pbull * np.asarray(params.Sigma1, float) + (1.0 - pbull) * np.asarray(params.Sigma2, float)
    sigma_bar = 0.5 * (sigma_bar + sigma_bar.T)
    return mu_bar, sigma_bar


def ew_weights(n_assets: int = 2) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets, dtype=float)


def gmv_weights(params: RSGBMParamsRegimeCov) -> np.ndarray:
    _, sigma_bar = unconditional_moments_regime_cov(params)
    inv = np.linalg.pinv(sigma_bar)
    ones = np.ones(sigma_bar.shape[0], dtype=float)
    w = inv @ ones
    denom = float(ones @ w)
    if abs(denom) < 1e-12:
        return ew_weights(sigma_bar.shape[0])
    return np.asarray(w / denom, dtype=float)


def target_excess_return_per_year(x0: float, z: float, T_years: float) -> float:
    gross = max(z / max(x0, 1e-12), 1e-12)
    return float(gross ** (1.0 / T_years) - 1.0)


def mv_target_weights(params: RSGBMParamsRegimeCov, x0: float, z: float, T_years: float) -> np.ndarray:
    mu_bar, sigma_bar = unconditional_moments_regime_cov(params)
    mu_excess = np.asarray(mu_bar - params.r, dtype=float)
    target = target_excess_return_per_year(x0=x0, z=z, T_years=T_years)

    inv = np.linalg.pinv(sigma_bar)
    ones = np.ones(len(mu_excess), dtype=float)
    A = float(ones @ inv @ ones)
    B = float(ones @ inv @ mu_excess)
    C = float(mu_excess @ inv @ mu_excess)
    D = A * C - B * B

    if abs(D) < 1e-12:
        denom = float(mu_excess @ inv @ mu_excess)
        if denom <= 1e-12:
            return ew_weights(len(mu_excess))
        w_tan = inv @ mu_excess
        scale = target / max(float(mu_excess @ w_tan), 1e-12)
        return np.asarray(scale * w_tan, dtype=float)

    lam1 = (C - B * target) / D
    lam2 = (A * target - B) / D
    w = inv @ (lam1 * ones + lam2 * mu_excess)
    return np.asarray(w, dtype=float)


def apply_leverage_cap_to_weights(weights: np.ndarray, leverage_cap: Optional[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float).copy()
    if leverage_cap is None:
        return w
    gross = float(np.sum(np.abs(w)))
    if gross <= leverage_cap or gross <= 1e-12:
        return w
    return w * (leverage_cap / gross)


def cash_weight_from_weights(weights: np.ndarray) -> float:
    return float(1.0 - np.sum(weights))


def gross_leverage_from_weights(weights: np.ndarray) -> float:
    return float(np.sum(np.abs(weights)))


# ---------------------------
# path generation and policies
# ---------------------------

def generate_test_path(params: RSGBMParamsRegimeCov, T_years: float, dt: float, seed: int) -> Dict[str, np.ndarray]:
    env = RSGBMEnvRegimeCov(
        params,
        EpisodeConfigRegimeCov(
            T_years=T_years,
            dt=dt,
            x0=1.0,
            s0=np.array([1.0, 1.0], dtype=float),
            p0=0.5,
            a_max=10.0,
            seed=seed,
            apply_action_projection=False,
        ),
    )
    obs = env.reset()
    n = env.n_steps
    d = len(obs["S"])
    S = np.empty((n + 1, d), dtype=float)
    I = np.empty(n + 1, dtype=int)
    t = np.arange(n + 1, dtype=float) * dt
    S[0] = obs["S"]
    I[0] = obs["I_true"]
    for k in range(n):
        obs, _, _ = env.step(np.zeros(d, dtype=float))
        S[k + 1] = obs["S"]
        I[k + 1] = obs["I_true"]
    return {"t": t, "S": S, "I": I}

def _project_to_gross_leverage(u: np.ndarray, x: float, a_max: float | None) -> np.ndarray:
    """
    Match the environment-side gross leverage projection:
        sum_i |u_i| / |x| <= a_max
    """
    u = np.asarray(u, dtype=float).copy()
    if a_max is None:
        return u
    denom = max(abs(float(x)), 1e-12)
    gross = float(np.sum(np.abs(u)) / denom)
    if gross <= float(a_max) + 1e-12:
        return u
    scale = float(a_max) / max(gross, 1e-12)
    return u * scale

def simulate_stage1_policy_path(
    agent: POEMVAgentCovModes,
    true_params: RSGBMParamsRegimeCov,
    filt_params: FilterParamsRegimeCov,
    T_years: float,
    dt: float,
    seed: int,
    x0: float = 1.0,
    p0: float = 0.5,
    a_max: float = 1.0,
) -> Dict[str, np.ndarray]:
    env = RSGBMEnvRegimeCov(
        true_params,
        EpisodeConfigRegimeCov(
            T_years=T_years,
            dt=dt,
            x0=x0,
            s0=np.array([1.0, 1.0], dtype=float),
            p0=p0,
            a_max=a_max,
            seed=seed,
            apply_action_projection=True,
        ),
    )
    obs = env.reset()
    n = env.n_steps
    d = len(obs["S"])
    t = np.arange(n + 1, dtype=float) * dt
    X = np.empty(n + 1, dtype=float)
    P = np.empty(n + 1, dtype=float)
    I = np.empty(n + 1, dtype=int)
    W = np.empty((n, d), dtype=float)
    S = np.empty((n + 1, d), dtype=float)
    # Applied risky action after the same gross-leverage projection used by the env
    U_applied = np.zeros((n, d), dtype=float)
    # Optional: keep the raw policy output too, for debugging if needed later
    U_raw = np.zeros((n, d), dtype=float)

    p_curr = float(p0)
    prev_S = np.asarray(obs["S"], dtype=float).copy()
    X[0] = float(obs["X"])
    P[0] = p_curr
    I[0] = int(obs["I_true"])
    S[0] = prev_S

    for k in range(n):
        #u_exec, _u_raw, _info = agent.act(float(obs["t"]), float(obs["X"]), p_curr, deterministic=True)
        #W[k] = np.asarray(u_exec, dtype=float)
        u_exec, u_raw, _info = agent.act(float(obs["t"]), float(obs["X"]), p_curr, deterministic=True)
        u_exec = np.asarray(u_exec, dtype=float)
        u_raw = np.asarray(u_raw, dtype=float)
        u_applied = _project_to_gross_leverage(u_exec, float(obs["X"]), a_max)

        U_raw[k] = u_raw
        U_applied[k] = u_applied
        obs_next, _, _ = env.step(u_applied)
        next_S = np.asarray(obs_next["S"], dtype=float)
        logret = np.log(next_S / np.maximum(prev_S, 1e-12))
        p_next, _innov = hmm_filter_regime_cov_q_update(p_curr, logret, dt, filt_params)

        X[k + 1] = float(obs_next["X"])
        P[k + 1] = float(p_next)
        I[k + 1] = int(obs_next["I_true"])
        S[k + 1] = next_S
        p_curr = float(p_next)
        prev_S = next_S
        obs = obs_next

    return {"t": t, "X": X, "P": P, "I": I, "U_applied": U_applied,"U_raw": U_raw, "S": S}


def simulate_static_weight_path(
    weights: np.ndarray,
    true_params: RSGBMParamsRegimeCov,
    T_years: float,
    dt: float,
    seed: int,
    x0: float = 1.0,
) -> Dict[str, np.ndarray]:
    path = generate_test_path(true_params, T_years=T_years, dt=dt, seed=seed)
    S = path["S"]
    t = path["t"]
    I = path["I"]
    n = S.shape[0] - 1
    d = S.shape[1]

    X = np.empty(n + 1, dtype=float)
    W = np.empty((n, d), dtype=float)
    X[0] = x0
    for k in range(n):
        rel = S[k + 1] / np.maximum(S[k], 1e-12)
        risky_growth = float(np.dot(weights, rel - 1.0))
        cash_w = 1.0 - float(np.sum(weights))
        total_growth = 1.0 + true_params.r * dt + risky_growth
        # If weights sum to 1 this matches fully invested risky allocation with discounted x-process proxy.
        # We keep the same simple comparison logic as the original eval code.
        X[k + 1] = X[k] * total_growth
        W[k] = weights
    return {"t": t, "X": X, "I": I, "W": W, "S": S}


# ---------------------------
# evaluation helpers
# ---------------------------

def summarize_terminal_wealth(name: str, XTs: List[float], z: float) -> Dict[str, float]:
    arr = np.asarray(XTs, dtype=float)
    mean_xT = float(np.mean(arr))
    std_xT = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    var_xT = float(np.var(arr, ddof=1)) if len(arr) > 1 else 0.0
    p_hit = float(np.mean(arr >= float(z)))
    shortfall = float(np.mean(np.maximum(float(z) - arr, 0.0)))
    sharpe_like = (mean_xT - float(z)) / max(std_xT, 1e-12)
    return {
        "method": name,
        "mean_xT": mean_xT,
        "std_xT": std_xT,
        "var_xT": var_xT,
        "median_xT": float(np.median(arr)),
        "p10_xT": float(np.quantile(arr, 0.10)),
        "p90_xT": float(np.quantile(arr, 0.90)),
        "p_hit_target": p_hit,
        "shortfall": shortfall,
        "sharpe_like": float(sharpe_like),
    }

def plot_terminal_histograms(results: Dict[str, List[float]], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    # matplotlib のデフォルト color cycle を使って、
    # histogram と mean line の色を揃える
    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", None)

    for idx, (name, vals) in enumerate(results.items()):
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue

        color = None if color_cycle is None else color_cycle[idx % len(color_cycle)]
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0

        ax.hist(
            arr,
            bins=30,
            alpha=0.35,
            density=False,
            color=color,
            label=None,
        )

        ax.axvline(
            mean_val,
            linestyle="--",
            linewidth=2.0,
            color=color,
            alpha=0.9,
            label=f"{name} mean={mean_val:.3f}, std={std_val:.3f}",
        )

    ax.set_xlabel("terminal wealth")
    ax.set_ylabel("count")
    ax.set_title("Terminal wealth histogram")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mean_paths(path_bank: Dict[str, List[np.ndarray]], t: np.ndarray, out_path: Path) -> None:
    fig = plt.figure(figsize=(9, 6))
    for name, seqs in path_bank.items():
        arr = np.asarray(seqs, dtype=float)
        mean = np.mean(arr, axis=0)
        #std = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)
        plt.plot(t, mean, label=name)
        #plt.fill_between(t, mean - std, mean + std, alpha=0.15)
    plt.xlabel("time")
    plt.ylabel("wealth")
    plt.title("Mean wealth path")
    plt.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_example_stage1_path(traj: Dict[str, np.ndarray], out_path: Path) -> None:
    t = traj["t"]
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(t, traj["X"], label="wealth")
    ax1.set_ylabel("wealth")
    ax1.legend()

    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(t, traj["P"], label="belief p_t")
    ax2.set_ylabel("belief")
    ax2.legend()

    # Panel 3: raw risky action u (this is what the policy/environment actually uses)
    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    u = np.asarray(traj["U_applied"], dtype=float)
    for j in range(u.shape[1]):
        ax3.plot(t[:-1], u[:, j], label=f"u{j+1}")
    ax3.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax3.set_ylabel("applied action")
    ax3.legend(loc="best")

    # Panel 4: normalized portfolio interpretation
    # In the training/env logic, u is a risky dollar position (discounted-wealth units),
    # so the interpretable risky portfolio weights are w = u / X.
    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    x_prev = np.asarray(traj["X"][:-1], dtype=float)
    denom = np.maximum(np.abs(x_prev), 1e-12)[:, None]
    w = u / denom
    total = np.sum(w, axis=1)
    cash = 1.0 - total

    for j in range(w.shape[1]):
        ax4.plot(t[:-1], w[:, j], label=f"w{j+1}")
    ax4.plot(t[:-1], cash, label="cash", linestyle="--")
    ax4.plot(t[:-1], total, label="total", linestyle=":")
    ax4.axhline(1.0, color="gray", linestyle=":", linewidth=1.0, alpha=0.8)
    ax4.axhline(0.0, color="black", linestyle="-", linewidth=0.8, alpha=0.5)
    ax4.set_ylabel("norm. weights")
    ax4.set_xlabel("time")
    ax4.legend(loc="best", ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


# ---------------------------
# main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--belief_mode", type=str, default="estimated_params", choices=["true_params", "estimated_params"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_paths", type=int, default=256)
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--x0", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.5)
    ap.add_argument("--z", type=float, default=None)
    ap.add_argument("--r", type=float, default=0.01)
    ap.add_argument("--a_max", type=float, default=1.0)
    ap.add_argument("--leverage_cap", type=float, default=None)
    ap.add_argument("--include_ew", action="store_true")
    ap.add_argument("--include_gmv", action="store_true")
    ap.add_argument("--include_mv", action="store_true")
    ap.add_argument("--plot_example", action="store_true")
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    run_dir = Path(args.run_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    true_params, filt_params, run_cfg = load_true_and_filter_params(run_dir, args.belief_mode, args.r)
    T = float(args.T if args.T is not None else run_cfg.get("T", 1.0))
    dt = float(args.dt if args.dt is not None else run_cfg.get("dt", 1 / 252))
    z = float(args.z if args.z is not None else run_cfg.get("z", 1.2))
    a_max = float(args.a_max if args.a_max is not None else run_cfg.get("a_max", 1.0))

    ckpt_path = Path(args.checkpoint) if args.checkpoint else (run_dir / "best_checkpoint.pt")
    if not ckpt_path.exists():
        ckpt_path = run_dir / "checkpoint.pt"
    agent, run_cfg_loaded = build_agent_from_checkpoint_cov_modes(
        ckpt_path=ckpt_path,
        run_dir=run_dir,
        T_years=T,
        dt=dt,
        a_max=a_max,
        z=z,
        r=args.r,
        device=args.device,
    )

    methods_terminal: Dict[str, List[float]] = {"RL": []}
    path_bank: Dict[str, List[np.ndarray]] = {"RL": []}
    example_traj = None

    ew = apply_leverage_cap_to_weights(ew_weights(2), args.leverage_cap)
    gmv = apply_leverage_cap_to_weights(gmv_weights(true_params), args.leverage_cap)
    mv = apply_leverage_cap_to_weights(mv_target_weights(true_params, x0=args.x0, z=z, T_years=T), args.leverage_cap)

    if args.include_ew:
        methods_terminal["EW"] = []
        path_bank["EW"] = []
    if args.include_gmv:
        methods_terminal["MinVar"] = []
        path_bank["MinVar"] = []
    if args.include_mv:
        methods_terminal["MV"] = []
        path_bank["MV"] = []

    for j in range(args.n_paths):
        seed_j = args.seed + 1000 * j + 7
        traj = simulate_stage1_policy_path(
            agent=agent,
            true_params=true_params,
            filt_params=filt_params,
            T_years=T,
            dt=dt,
            seed=seed_j,
            x0=args.x0,
            p0=args.p0,
            a_max=a_max,
        )
        methods_terminal["RL"].append(float(traj["X"][-1]))
        path_bank["RL"].append(traj["X"])
        if example_traj is None:
            example_traj = traj

        if args.include_ew:
            tr = simulate_static_weight_path(ew, true_params, T_years=T, dt=dt, seed=seed_j, x0=args.x0)
            methods_terminal["EW"].append(float(tr["X"][-1]))
            path_bank["EW"].append(tr["X"])
        if args.include_gmv:
            tr = simulate_static_weight_path(gmv, true_params, T_years=T, dt=dt, seed=seed_j, x0=args.x0)
            methods_terminal["MinVar"].append(float(tr["X"][-1]))
            path_bank["MinVar"].append(tr["X"])
        if args.include_mv:
            tr = simulate_static_weight_path(mv, true_params, T_years=T, dt=dt, seed=seed_j, x0=args.x0)
            methods_terminal["MV"].append(float(tr["X"][-1]))
            path_bank["MV"].append(tr["X"])

    summary_rows = [summarize_terminal_wealth(name, vals, z=z) for name, vals in methods_terminal.items()]
    summary = pd.DataFrame(summary_rows).sort_values("mean_xT", ascending=False)
    summary.to_csv(outdir / "summary_terminal_wealth.csv", index=False)

    with open(outdir / "eval_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(run_dir),
                "checkpoint": str(ckpt_path),
                "belief_mode": args.belief_mode,
                "n_paths": args.n_paths,
                "T": T,
                "dt": dt,
                "x0": args.x0,
                "p0": args.p0,
                "z": z,
                "r": args.r,
                "a_max": a_max,
                "leverage_cap": args.leverage_cap,
                "include_ew": args.include_ew,
                "include_gmv": args.include_gmv,
                "include_mv": args.include_mv,
                "sigma_mode": run_cfg_loaded.get("sigma_mode", None),
                "cov_mode": run_cfg_loaded.get("cov_mode", run_cfg_loaded.get("training_modes", {}).get("cov_mode", None)),
                "estimation_method": run_cfg_loaded.get("estimation", {}).get("estimation_method", None),
            },
            f,
            indent=2,
        )

    plot_terminal_histograms(methods_terminal, outdir / "terminal_wealth_hist.png")
    plot_mean_paths(path_bank, example_traj["t"] if example_traj is not None else np.arange(2), outdir / "mean_paths.png")
    if args.plot_example and example_traj is not None:
        plot_example_stage1_path(example_traj, outdir / "stage1_example_path.png")

    print(summary.to_string(index=False))
    print(f"Saved evaluation outputs to {outdir}")


if __name__ == "__main__":
    main()
