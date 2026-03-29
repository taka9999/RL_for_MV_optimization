from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .agent import POEMVAgent, TrainConfig
from .env import EpisodeConfig, RSGBMEnv, RSGBMParams
from .filtering import FilterParams, wonham_filter_q_update
from .utils import safe_clip_p, set_seed


def default_true_params(r: float = 0.01) -> RSGBMParams:
    return RSGBMParams(
        mu1=np.array([0.25, 0.18], dtype=float),
        mu2=np.array([-0.73, -0.40], dtype=float),
        Sigma=np.array([[0.22 ** 2, 0.22 * 0.18 * 0.3], [0.22 * 0.18 * 0.3, 0.18 ** 2]], dtype=float),
        lam1=0.36, lam2=2.89, r=r,
    )


def _params_from_run_config(run_cfg: Dict, fallback_r: float) -> RSGBMParams:
    default_params = default_true_params(r=fallback_r)
    src = run_cfg.get("policy_params") or run_cfg.get("true_params") or run_cfg.get("filter_params") or {}
    return RSGBMParams(
        mu1=np.asarray(src.get("mu1", default_params.mu1), dtype=float),
        mu2=np.asarray(src.get("mu2", default_params.mu2), dtype=float),
        Sigma=np.asarray(src.get("Sigma", default_params.Sigma), dtype=float),
        lam1=float(src.get("lam1", default_params.lam1)),
        lam2=float(src.get("lam2", default_params.lam2)),
        r=float(src.get("r", fallback_r)),
    )


def load_run_config(run_dir: Path) -> Dict:
    with open(run_dir / "run_config.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_agent_from_checkpoint(ckpt_path: Path, run_dir: Path, device: str = "cpu") -> tuple[POEMVAgent, Dict]:
    run_cfg = load_run_config(run_dir)
    params = _params_from_run_config(run_cfg, fallback_r=float(run_cfg.get("r", 0.01)))
    cfg = TrainConfig(
        T_years=float(run_cfg.get("T", 1.0)),
        dt=float(run_cfg.get("dt", 1 / 252)),
        x0=float(run_cfg.get("x0", 1.0)),
        p0=float(run_cfg.get("p0", 0.5)),
        a_max=run_cfg.get("a_max", None),
        cap_mode=str(run_cfg.get("cap_mode", "none")),
        z=float(run_cfg.get("z", 1.2)),
        Lambda=float(run_cfg.get("Lambda", 1.0)),
        r=float(run_cfg.get("r", 0.01)),
        alpha_theta=float(run_cfg.get("alpha_theta", 3e-5)),
        alpha_phi=float(run_cfg.get("alpha_phi", 1e-4)),
        alpha_w=float(run_cfg.get("alpha_w", 1e-3)),
        omega_update_every=int(run_cfg.get("omega_update_every", 10)),
        critic_steps=int(run_cfg.get("critic_steps", 10)),
        omega_ema_beta=float(run_cfg.get("omega_ema_beta", 0.9)),
        mu1=params.mu1, mu2=params.mu2, Sigma=params.Sigma,
        device=device, dtype=torch.float64,
    )
    agent = POEMVAgent(cfg)
    ckpt = torch.load(ckpt_path, map_location=device)
    agent.vf.load_state_dict(ckpt["vf_state_dict"])
    agent.pi.load_state_dict(ckpt["pi_state_dict"])
    with torch.no_grad():
        agent.omega = torch.as_tensor(float(ckpt.get("omega", 0.0)), dtype=cfg.dtype, device=cfg.device)
    agent.vf.eval(); agent.pi.eval()
    return agent, run_cfg


def stationary_bull_prob(params: RSGBMParams) -> float:
    return float(params.lam2 / (params.lam1 + params.lam2))


def unconditional_moments(params: RSGBMParams) -> Tuple[np.ndarray, np.ndarray]:
    pbull = stationary_bull_prob(params)
    mu_bar = pbull * np.asarray(params.mu1, float) + (1.0 - pbull) * np.asarray(params.mu2, float)
    sigma_bar = pbull * np.asarray(params.Sigma, float) + (1.0 - pbull) * np.asarray(params.Sigma, float)
    return mu_bar, sigma_bar


def ew_weights(n_assets: int = 2) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets, dtype=float)


def gmv_weights(params: RSGBMParams) -> np.ndarray:
    _, sigma_bar = unconditional_moments(params)
    inv = np.linalg.pinv(sigma_bar)
    ones = np.ones(2, dtype=float)
    w = inv @ ones
    denom = float(ones @ w)
    return ew_weights(2) if abs(denom) < 1e-12 else np.asarray(w / denom, dtype=float)


def target_excess_return_per_year(x0: float, z: float, T_years: float) -> float:
    gross = max(z / max(x0, 1e-12), 1e-12)
    return float(gross ** (1.0 / T_years) - 1.0)


def mv_target_weights(params: RSGBMParams, x0: float, z: float, T_years: float) -> np.ndarray:
    mu_bar, sigma_bar = unconditional_moments(params)
    mu_excess = np.asarray(mu_bar - params.r, dtype=float)
    target = target_excess_return_per_year(x0=x0, z=z, T_years=T_years)
    inv = np.linalg.pinv(sigma_bar)
    ones = np.ones(2, dtype=float)
    A = float(ones @ inv @ ones)
    B = float(ones @ inv @ mu_excess)
    C = float(mu_excess @ inv @ mu_excess)
    D = A * C - B * B
    if abs(D) < 1e-12:
        denom = float(mu_excess @ inv @ mu_excess)
        if denom <= 1e-12:
            return ew_weights(2)
        w_tan = inv @ mu_excess
        scale = target / max(float(mu_excess @ w_tan), 1e-12)
        return np.asarray(scale * w_tan, dtype=float)
    lam1 = (C - B * target) / D
    lam2 = (A * target - B) / D
    return np.asarray(inv @ (lam1 * ones + lam2 * mu_excess), dtype=float)


def apply_leverage_cap_to_weights(weights: np.ndarray, leverage_cap: Optional[float]) -> np.ndarray:
    w = np.asarray(weights, dtype=float).copy()
    if leverage_cap is None:
        return w
    gross = float(np.sum(np.abs(w)))
    return w if gross <= leverage_cap or gross <= 1e-12 else w * (leverage_cap / gross)


def static_action_from_weights(weights: np.ndarray, wealth: float) -> np.ndarray:
    return np.asarray(weights, dtype=float) * float(wealth)


def generate_test_path(params: RSGBMParams, T_years: float, dt: float, seed: int) -> Dict[str, np.ndarray]:
    env = RSGBMEnv(params, EpisodeConfig(T_years=T_years, dt=dt, x0=1.0, s0=np.array([1.0, 1.0]), p0=0.5, a_max=10.0, omega=0.0, seed=seed))
    obs = env.reset()
    n = env.n_steps
    S = np.empty((n + 1, 2), dtype=float)
    I = np.empty(n + 1, dtype=int)
    t = np.arange(n + 1, dtype=float) * dt
    S[0] = obs["S"]; I[0] = obs["I_true"]
    for k in range(n):
        obs, _, _ = env.step(np.zeros(2, dtype=float))
        S[k + 1] = obs["S"]; I[k + 1] = obs["I_true"]
    logret = np.log(S[1:] / S[:-1])
    ret = (S[1:] - S[:-1]) / S[:-1]
    return {"t": t, "S": S, "I": I, "logret": logret, "ret": ret}


def compute_belief_path(logret: np.ndarray, filt_params: FilterParams, dt: float, p0: float = 0.5) -> np.ndarray:
    n = logret.shape[0]
    p = np.empty(n + 1, dtype=float)
    p[0] = p0
    for k in range(n):
        p[k + 1], _ = wonham_filter_q_update(p[k], logret[k], dt, filt_params)
        p[k + 1] = safe_clip_p(p[k + 1])
    return p


def rebalance_steps(label: str) -> int:
    return 1 if label.lower() == "daily" else 21


def rl_action(agent: POEMVAgent, t: float, x: float, p: float, deterministic: bool = True) -> np.ndarray:
    return np.asarray(agent.policy_mean(t, x, p), dtype=float) if deterministic else np.asarray(agent.act(t, x, p, deterministic=False)[0], dtype=float)


def scale_action_to_leverage_cap(u: np.ndarray, wealth: float, leverage_cap: Optional[float]) -> np.ndarray:
    u = np.asarray(u, dtype=float).copy()
    if leverage_cap is None:
        return u
    denom = max(abs(float(wealth)), 1e-12)
    gross = float(np.sum(np.abs(u)) / denom)
    return u if gross <= leverage_cap or gross <= 1e-12 else u * (leverage_cap / gross)


def simulate_method_on_path(method: str, path: Dict[str, np.ndarray], params: RSGBMParams, filt_params: FilterParams,
                            T_years: float, dt: float, z: float, leverage_cap: Optional[float], rebalance_label: str,
                            agent: Optional[POEMVAgent] = None, p0: float = 0.5, x0: float = 1.0) -> Dict[str, np.ndarray]:
    n = path["ret"].shape[0]
    step_reb = rebalance_steps(rebalance_label)
    belief = compute_belief_path(path["logret"], filt_params=filt_params, dt=dt, p0=p0)
    wealth = np.empty(n + 1, dtype=float); wealth[0] = x0
    gross_lev = np.empty(n, dtype=float); cash_w = np.empty(n, dtype=float)
    u_hist = np.empty((n, 2), dtype=float); w_hist = np.empty((n, 2), dtype=float)
    current_u = np.zeros(2, dtype=float)
    w_ew = ew_weights(2)
    w_gmv = gmv_weights(params)
    w_mv = mv_target_weights(params, x0=x0, z=z, T_years=T_years)

    for k in range(n):
        xk = wealth[k]; tk = path["t"][k]; pk = belief[k]
        if (k % step_reb) == 0:
            if method == "RLMean":
                current_u = scale_action_to_leverage_cap(rl_action(agent, tk, xk, pk, deterministic=True), xk, leverage_cap)
            elif method == "RLSample":
                current_u = scale_action_to_leverage_cap(rl_action(agent, tk, xk, pk, deterministic=False), xk, leverage_cap)
            elif method == "EW":
                current_u = static_action_from_weights(apply_leverage_cap_to_weights(w_ew, leverage_cap), xk)
            elif method == "MinVar":
                current_u = static_action_from_weights(apply_leverage_cap_to_weights(w_gmv, leverage_cap), xk)
            elif method == "MeanVar":
                current_u = static_action_from_weights(apply_leverage_cap_to_weights(w_mv, leverage_cap), xk)
            else:
                raise ValueError(method)
        ret_k = path["ret"][k]
        wealth[k + 1] = wealth[k] + float(np.dot(current_u, ret_k - params.r * dt))
        denom = max(abs(wealth[k]), 1e-12)
        w_hist[k] = current_u / denom
        u_hist[k] = current_u
        gross_lev[k] = np.sum(np.abs(w_hist[k]))
        cash_w[k] = 1.0 - np.sum(w_hist[k])
    return {"wealth": wealth, "belief": belief, "u": u_hist, "w": w_hist, "gross_lev": gross_lev, "cash_w": cash_w}


def summarize_paths(results: Dict[str, list[np.ndarray]]) -> pd.DataFrame:
    rows = []
    for key, wealth_list in results.items():
        final_vals = np.array([w[-1] for w in wealth_list], dtype=float)
        rows.append(dict(method=key, mean_terminal=float(final_vals.mean()), std_terminal=float(final_vals.std(ddof=0))))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--n_paths", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--rebalance", type=str, default="daily", choices=["daily", "monthly"])
    ap.add_argument("--eval_leverage_cap", type=float, default=None)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    agent, run_cfg = build_agent_from_checkpoint(Path(args.checkpoint), run_dir, device=args.device)
    params = _params_from_run_config(run_cfg, fallback_r=float(run_cfg.get("r", 0.01)))
    filt_src = run_cfg.get("filter_params") or run_cfg.get("true_params") or {}
    filt_params = FilterParams(mu1=np.asarray(filt_src.get("mu1", params.mu1), float),
                               mu2=np.asarray(filt_src.get("mu2", params.mu2), float),
                               Sigma=np.asarray(filt_src.get("Sigma", params.Sigma), float),
                               lam1=float(filt_src.get("lam1", params.lam1)), lam2=float(filt_src.get("lam2", params.lam2)), r=float(filt_src.get("r", params.r)))
    T_years = float(run_cfg.get("T", 1.0)); dt = float(run_cfg.get("dt", 1 / 252)); z = float(run_cfg.get("z", 1.2))
    lev_cap = args.eval_leverage_cap

    set_seed(args.seed)
    methods = ["RLMean", "EW", "MinVar", "MeanVar"]
    wealth_results = {m: [] for m in methods}
    for i in range(args.n_paths):
        path = generate_test_path(params, T_years=T_years, dt=dt, seed=args.seed + i)
        for m in methods:
            res = simulate_method_on_path(m, path, params, filt_params, T_years, dt, z, lev_cap, args.rebalance, agent if m == "RLMean" else None)
            wealth_results[m].append(res["wealth"])
    summary = summarize_paths(wealth_results)
    summary.to_csv(outdir / "summary.csv", index=False)

    fig = plt.figure()
    for m in methods:
        mean_path = np.mean(np.stack(wealth_results[m], axis=0), axis=0)
        plt.plot(mean_path, label=m)
    plt.legend(); plt.xlabel("step"); plt.ylabel("wealth")
    fig.tight_layout(); fig.savefig(outdir / "wealth_paths.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
