from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .agent import POEMVAgent, TrainConfig
from .env import RSGBMParams, EpisodeConfig, RSGBMEnv
from .filtering import FilterParams, wonham_filter_q_update
from .utils import safe_clip_p, set_seed


def default_true_params(r: float = 0.01) -> RSGBMParams:
    return RSGBMParams(
        mu1=np.array([0.25, 0.18], dtype=float),
        mu2=np.array([-0.73, -0.40], dtype=float),
        Sigma1=np.array(
            [[0.22**2, 0.22 * 0.18 * 0.3],
             [0.22 * 0.18 * 0.3, 0.18**2]],
            dtype=float,
        ),
        Sigma2=np.array(
            [[0.22**2, 0.22 * 0.18 * 0.5],
             [0.22 * 0.18 * 0.5, 0.18**2]],
            dtype=float,
        ),
        lam1=0.36,
        lam2=2.89,
        r=r,
    )


def stationary_bull_prob(params: RSGBMParams) -> float:
    return float(params.lam2 / (params.lam1 + params.lam2))


def unconditional_moments(params: RSGBMParams) -> Tuple[np.ndarray, np.ndarray]:
    pbull = stationary_bull_prob(params)
    mu_bar = pbull * np.asarray(params.mu1, float) + (1.0 - pbull) * np.asarray(params.mu2, float)
    sigma_bar = pbull * np.asarray(params.Sigma1, float) + (1.0 - pbull) * np.asarray(params.Sigma2, float)
    return mu_bar, sigma_bar


def target_excess_return_per_year(x0: float, z: float, T_years: float) -> float:
    gross = max(z / max(x0, 1e-12), 1e-12)
    return float(gross ** (1.0 / T_years) - 1.0)


def load_run_config(run_dir: Path) -> Dict:
    cfg_path = run_dir / "run_config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"run_config.json not found: {cfg_path}")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint(ckpt_path: Path, device: str = "cpu") -> Dict:
    ckpt = torch.load(ckpt_path, map_location=device)
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict with vf/pi/omega fields.")
    return ckpt


def build_agent_from_checkpoint(
    ckpt_path: Path,
    run_dir: Path,
    T_years: float,
    dt: float,
    a_max: float,
    z: float,
    r: float,
    device: str = "cpu",
) -> POEMVAgent:
    run_cfg = load_run_config(run_dir)
    ckpt = load_checkpoint(ckpt_path, device=device)

    train_cfg = TrainConfig(
        T_years=T_years,
        dt=dt,
        x0=1.0,
        s0=1.0,
        p0=0.5,
        a_max=a_max,
        z=z,
        Lambda=float(run_cfg.get("Lambda", 1.0)),
        r=r,
        alpha_theta=float(run_cfg.get("alpha_theta", 1e-5)),
        alpha_phi=float(run_cfg.get("alpha_phi", 1e-4)),
        alpha_w=float(run_cfg.get("alpha_w", 1e-3)),
        omega_update_every=int(run_cfg.get("omega_update_every", 10)),
        device=device,
        dtype=torch.float64,
    )
    agent = POEMVAgent(train_cfg)

    vf_sd = ckpt.get("vf_state_dict", ckpt.get("vf", ckpt.get("value_state_dict")))
    pi_sd = ckpt.get("pi_state_dict", ckpt.get("pi", ckpt.get("policy_state_dict")))
    omega = ckpt.get("omega", ckpt.get("omega_value", ckpt.get("lagrange_multiplier")))

    if vf_sd is None or pi_sd is None:
        raise KeyError(
            "Checkpoint is missing vf/pi state dicts. "
            "Expected keys like vf_state_dict and pi_state_dict."
        )

    agent.vf.load_state_dict(vf_sd)
    agent.pi.load_state_dict(pi_sd)
    if omega is not None:
        with torch.no_grad():
            agent.omega = torch.as_tensor(
                float(omega), dtype=train_cfg.dtype, device=train_cfg.device
            )
    agent.vf.eval()
    agent.pi.eval()
    return agent


def ew_weights(n_assets: int = 2) -> np.ndarray:
    return np.full(n_assets, 1.0 / n_assets, dtype=float)


def gmv_weights(params: RSGBMParams) -> np.ndarray:
    _, sigma_bar = unconditional_moments(params)
    inv = np.linalg.pinv(sigma_bar)
    ones = np.ones(2, dtype=float)
    w = inv @ ones
    denom = float(ones @ w)
    if abs(denom) < 1e-12:
        return ew_weights(2)
    return np.asarray(w / denom, dtype=float)


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


def generate_test_path(params: RSGBMParams, T_years: float, dt: float, seed: int) -> Dict[str, np.ndarray]:
    env = RSGBMEnv(
        params,
        EpisodeConfig(
            T_years=T_years,
            dt=dt,
            x0=1.0,
            s0=np.array([1.0, 1.0], dtype=float),
            p0=0.5,
            a_max=10.0,
            omega=0.0,
            seed=seed,
        ),
    )
    obs = env.reset()
    n = env.n_steps
    S = np.empty((n + 1, 2), dtype=float)
    I = np.empty(n + 1, dtype=int)
    t = np.arange(n + 1, dtype=float) * dt

    S[0] = obs["S"]
    I[0] = obs["I_true"]
    for k in range(n):
        obs, _, _ = env.step(np.zeros(2, dtype=float))
        S[k + 1] = obs["S"]
        I[k + 1] = obs["I_true"]

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
    label = label.lower()
    if label == "daily":
        return 1
    if label == "monthly":
        return 21
    raise ValueError(f"Unknown rebalance label: {label}")


def static_action_from_weights(weights: np.ndarray, wealth: float) -> np.ndarray:
    return np.asarray(weights, dtype=float) * float(wealth)


def rl_action(agent: POEMVAgent, t: float, x: float, p: float) -> np.ndarray:
    u, _, _ = agent.act(t, x, p)
    return np.asarray(u, dtype=float)


def scale_action_to_leverage_cap(u: np.ndarray, wealth: float, leverage_cap: Optional[float]) -> np.ndarray:
    u = np.asarray(u, dtype=float).copy()
    if leverage_cap is None:
        return u
    denom = max(abs(float(wealth)), 1e-12)
    gross = float(np.sum(np.abs(u)) / denom)
    if gross <= leverage_cap or gross <= 1e-12:
        return u
    return u * (leverage_cap / gross)


def simulate_method_on_path(
    method: str,
    path: Dict[str, np.ndarray],
    params: RSGBMParams,
    filt_params: FilterParams,
    T_years: float,
    dt: float,
    z: float,
    leverage_cap: Optional[float],
    rebalance_label: str,
    agent: Optional[POEMVAgent] = None,
    p0: float = 0.5,
    x0: float = 1.0,
) -> Dict[str, np.ndarray]:
    n = path["ret"].shape[0]
    step_reb = rebalance_steps(rebalance_label)
    belief = compute_belief_path(path["logret"], filt_params=filt_params, dt=dt, p0=p0)

    wealth = np.empty(n + 1, dtype=float)
    gross_lev = np.empty(n, dtype=float)
    cash_w = np.empty(n, dtype=float)
    u_hist = np.empty((n, 2), dtype=float)
    w_hist = np.empty((n, 2), dtype=float)

    wealth[0] = x0
    current_u = np.zeros(2, dtype=float)

    w_ew = ew_weights(2)
    w_gmv = gmv_weights(params)
    w_mv = mv_target_weights(params, x0=x0, z=z, T_years=T_years)

    for k in range(n):
        xk = wealth[k]
        tk = path["t"][k]
        pk = belief[k]

        if (k % step_reb) == 0:
            if method == "RL":
                if agent is None:
                    raise ValueError("RL method requires a loaded agent.")
                current_u = rl_action(agent, tk, xk, pk)
                current_u = scale_action_to_leverage_cap(current_u, xk, leverage_cap)
            elif method == "EW":
                current_u = static_action_from_weights(
                    apply_leverage_cap_to_weights(w_ew, leverage_cap), xk
                )
            elif method == "MinVar":
                current_u = static_action_from_weights(
                    apply_leverage_cap_to_weights(w_gmv, leverage_cap), xk
                )
            elif method == "MeanVar":
                current_u = static_action_from_weights(
                    apply_leverage_cap_to_weights(w_mv, leverage_cap), xk
                )
            else:
                raise ValueError(f"Unknown method: {method}")

        ret_k = path["ret"][k]
        wealth[k + 1] = xk + float(np.dot(current_u, ret_k))

        denom = max(abs(xk), 1e-12)
        weights_k = current_u / denom
        gross_lev[k] = gross_leverage_from_weights(weights_k)
        cash_w[k] = cash_weight_from_weights(weights_k)
        u_hist[k] = current_u
        w_hist[k] = weights_k

    wealth_ret = np.diff(wealth) / np.maximum(np.abs(wealth[:-1]), 1e-12)
    if wealth_ret.std(ddof=1) > 1e-12:
        path_sharpe_ann = float(np.sqrt(252.0) * wealth_ret.mean() / wealth_ret.std(ddof=1))
    else:
        path_sharpe_ann = np.nan

    return {
        "wealth": wealth,
        "gross_leverage": gross_lev,
        "cash_weight": cash_w,
        "u": u_hist,
        "weights": w_hist,
        "belief": belief,
        "path_sharpe_ann": path_sharpe_ann,
    }


def summarize_results(results: List[Dict]) -> pd.DataFrame:
    rows = []
    grouped: Dict[Tuple[str, str, str], List[Dict]] = {}

    for r in results:
        key = (r["method"], r["rebalance"], r["leverage_cap_label"])
        grouped.setdefault(key, []).append(r)

    for (method, rebalance, lev_label), items in grouped.items():
        xT = np.array([it["wealth"][-1] for it in items], dtype=float)
        path_sharpes = np.array([it["path_sharpe_ann"] for it in items], dtype=float)
        avg_gross_lev = np.mean([np.mean(it["gross_leverage"]) for it in items])
        avg_cash_w = np.mean([np.mean(it["cash_weight"]) for it in items])

        mean_xT = float(np.mean(xT))
        var_xT = float(np.var(xT, ddof=1)) if len(xT) > 1 else 0.0
        std_xT = float(np.std(xT, ddof=1)) if len(xT) > 1 else 0.0
        terminal_sharpe = float((mean_xT - 1.0) / std_xT) if std_xT > 1e-12 else np.nan

        rows.append(
            {
                "method": method,
                "rebalance": rebalance,
                "leverage_cap": lev_label,
                "n_paths": len(items),
                "mean_terminal": mean_xT,
                "variance_terminal": var_xT,
                "std_terminal": std_xT,
                "terminal_sharpe": terminal_sharpe,
                "avg_path_sharpe_ann": float(np.nanmean(path_sharpes)),
                "p05_terminal": float(np.quantile(xT, 0.05)),
                "p50_terminal": float(np.quantile(xT, 0.50)),
                "p95_terminal": float(np.quantile(xT, 0.95)),
                "avg_gross_leverage": float(avg_gross_lev),
                "avg_cash_weight": float(avg_cash_w),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["rebalance", "leverage_cap", "method"]
    ).reset_index(drop=True)


def save_representative_wealth_plot(
    results: List[Dict],
    outdir: Path,
    rebalance_label: str,
    leverage_cap_label: str,
    representative_path_ids: List[int],
):
    subset = [
        r for r in results
        if r["rebalance"] == rebalance_label
        and r["leverage_cap_label"] == leverage_cap_label
        and r["path_id"] in representative_path_ids
    ]
    if not subset:
        return

    fig = plt.figure(figsize=(11, 6))
    for r in subset:
        plt.plot(r["t"], r["wealth"], label=f'{r["method"]} | path {r["path_id"]}', alpha=0.9)
    plt.xlabel("time (years)")
    plt.ylabel("discounted wealth")
    plt.title(f"Representative wealth paths | {rebalance_label} | lev={leverage_cap_label}")
    plt.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / f"wealth_paths_{rebalance_label}_lev_{leverage_cap_label}.png", dpi=200)
    plt.close(fig)


def save_average_timeseries_plot(
    results: List[Dict],
    outdir: Path,
    rebalance_label: str,
    leverage_cap_label: str,
    key: str,
    ylabel: str,
    filename_stub: str,
):
    subset = [
        r for r in results
        if r["rebalance"] == rebalance_label and r["leverage_cap_label"] == leverage_cap_label
    ]
    if not subset:
        return

    methods = sorted({r["method"] for r in subset})
    fig = plt.figure(figsize=(11, 6))
    for method in methods:
        mats = [r[key] for r in subset if r["method"] == method]
        t = [r["t"][:-1] if key != "wealth" else r["t"] for r in subset if r["method"] == method][0]
        avg_series = np.mean(np.stack(mats, axis=0), axis=0)
        plt.plot(t, avg_series, label=method)

    plt.xlabel("time (years)")
    plt.ylabel(ylabel)
    plt.title(f"Average {ylabel} | {rebalance_label} | lev={leverage_cap_label}")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / f"{filename_stub}_{rebalance_label}_lev_{leverage_cap_label}.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True, help="Training run directory containing run_config.json")
    ap.add_argument("--checkpoint", type=str, required=True, help="Model checkpoint for RL policy")
    ap.add_argument("--outdir", type=str, default="runs/eval_compare")
    ap.add_argument("--n_paths", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=1 / 252)
    ap.add_argument("--z", type=float, default=1.5)
    ap.add_argument("--x0", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.5)
    ap.add_argument("--a_max", type=float, default=2.0)
    ap.add_argument("--r", type=float, default=0.01)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    true_params = default_true_params(r=args.r)
    filt_params = FilterParams(
        mu1=true_params.mu1,
        mu2=true_params.mu2,
        Sigma1=true_params.Sigma1,
        Sigma2=true_params.Sigma2,
        lam1=true_params.lam1,
        lam2=true_params.lam2,
        r=true_params.r,
    )

    agent = build_agent_from_checkpoint(
        ckpt_path=Path(args.checkpoint),
        run_dir=Path(args.run_dir),
        T_years=args.T,
        dt=args.dt,
        a_max=args.a_max,
        z=args.z,
        r=args.r,
        device=args.device,
    )

    leverage_caps: List[Optional[float]] = [None, 2.0, 3.0]
    methods = ["RL", "EW", "MinVar", "MeanVar"]
    rebalances = ["daily", "monthly"]

    results: List[Dict] = []
    representative_path_ids = list(range(min(4, args.n_paths)))

    for path_id in range(args.n_paths):
        path = generate_test_path(
            true_params,
            T_years=args.T,
            dt=args.dt,
            seed=args.seed + 10_000 + path_id,
        )

        for reb in rebalances:
            for lev in leverage_caps:
                lev_label = "none" if lev is None else str(int(lev))
                for method in methods:
                    sim = simulate_method_on_path(
                        method=method,
                        path=path,
                        params=true_params,
                        filt_params=filt_params,
                        T_years=args.T,
                        dt=args.dt,
                        z=args.z,
                        leverage_cap=lev,
                        rebalance_label=reb,
                        agent=agent if method == "RL" else None,
                        p0=args.p0,
                        x0=args.x0,
                    )
                    results.append(
                        {
                            "method": method,
                            "rebalance": reb,
                            "leverage_cap_label": lev_label,
                            "path_id": path_id,
                            "t": path["t"],
                            **sim,
                        }
                    )

    summary = summarize_results(results)
    summary.to_csv(outdir / "eval_summary.csv", index=False)

    terminal_rows = []
    for r in results:
        terminal_rows.append(
            {
                "method": r["method"],
                "rebalance": r["rebalance"],
                "leverage_cap": r["leverage_cap_label"],
                "path_id": r["path_id"],
                "terminal_wealth": float(r["wealth"][-1]),
                "avg_gross_leverage": float(np.mean(r["gross_leverage"])),
                "avg_cash_weight": float(np.mean(r["cash_weight"])),
                "path_sharpe_ann": float(r["path_sharpe_ann"]),
            }
        )
    pd.DataFrame(terminal_rows).to_csv(outdir / "eval_terminal_by_path.csv", index=False)

    for reb in rebalances:
        for lev in leverage_caps:
            lev_label = "none" if lev is None else str(int(lev))
            save_representative_wealth_plot(results, outdir, reb, lev_label, representative_path_ids)
            save_average_timeseries_plot(
                results,
                outdir,
                reb,
                lev_label,
                key="gross_leverage",
                ylabel="gross leverage",
                filename_stub="avg_gross_leverage",
            )
            save_average_timeseries_plot(
                results,
                outdir,
                reb,
                lev_label,
                key="cash_weight",
                ylabel="cash weight",
                filename_stub="avg_cash_weight",
            )

    with open(outdir / "eval_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_dir": str(Path(args.run_dir).resolve()),
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "n_paths": args.n_paths,
                "seed": args.seed,
                "T": args.T,
                "dt": args.dt,
                "z": args.z,
                "x0": args.x0,
                "p0": args.p0,
                "a_max": args.a_max,
                "r": args.r,
                "methods": methods,
                "rebalances": rebalances,
                "leverage_caps": ["none", 2, 3],
            },
            f,
            indent=2,
        )

    print(f"Saved summary to {outdir / 'eval_summary.csv'}")
    print(f"Saved path-level results to {outdir / 'eval_terminal_by_path.csv'}")


if __name__ == "__main__":
    main()