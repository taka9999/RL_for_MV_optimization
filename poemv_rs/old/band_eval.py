from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .utils import set_seed
from .eval_compare import (
    default_true_params,
    build_agent_from_checkpoint,
    generate_test_path,
    compute_belief_path,
    ew_weights,
    gmv_weights,
    mv_target_weights,
    apply_leverage_cap_to_weights,
)
from .filtering import FilterParams
from .band_stage2 import BandPolicy, _obs_vec, _stage2_action, _clip_to_box

def _load_band(path: Optional[str], device: str):
    if path is None:
        return None, None
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["stage2_cfg"]
    policy = BandPolicy(obs_dim=9, hidden=int(cfg["hidden"]), init_halfwidth=float(cfg["init_halfwidth"])).to(device=device, dtype=torch.float64)
    policy.load_state_dict(ckpt["band_policy_state_dict"])
    policy.eval()
    return policy, cfg


def _simulate_band_policy(
    *,
    center_agent,
    band_policy: BandPolicy,
    mode: str,
    path: Dict[str, np.ndarray],
    filt_params: FilterParams,
    T_years: float,
    dt: float,
    z: float,
    x0: float,
    p0: float,
    tcost: float,
    rebalance_steps: int,
    leverage_cap: Optional[float],
    init_halfwidth: float,
    gamma_risk: float,
    qvi_width_floor: float,
    width_mode: str,
    device: str,
) -> Dict:
    n = path["ret"].shape[0]
    belief = compute_belief_path(path["logret"], filt_params=filt_params, dt=dt, p0=p0)
    wealth = np.empty(n + 1, dtype=float)
    wealth[0] = x0
    current_u = np.zeros(2, dtype=float)
    gross_lev = np.empty(n, dtype=float)
    cash_w = np.empty(n, dtype=float)

    for k in range(n):
        xk = wealth[k]
        tk = path["t"][k]
        pk = belief[k]

        if (k % int(rebalance_steps)) == 0:
            center_u = np.asarray(center_agent.policy_mean(tk, xk, pk), dtype=float)
            denom = max(abs(float(xk)), 1e-12)
            center_w = center_u / denom
            w_cur = current_u / denom
            obs_np = _obs_vec(tk / max(T_years, 1e-12), xk, pk, w_cur, center_w)
            center_hat, lower, upper, _, _, _ = _stage2_action(
                band_policy,
                mode=mode,
                progress=1.0,
                center_w=center_w,
                obs_np=obs_np,
                deterministic=True,
                device=device,
                dtype=torch.float64,
                init_halfwidth=float(init_halfwidth),
                Sigma=filt_params.Sigma,
                tcost=tcost,
                gamma_risk=gamma_risk,
                qvi_width_floor=qvi_width_floor,
                width_mode=width_mode,
            )
            w_tgt = w_cur if not np.any((w_cur < lower) | (w_cur > upper)) else _clip_to_box(w_cur, lower, upper)
            if leverage_cap is not None:
                gross = np.sum(np.abs(w_tgt))
                if gross > leverage_cap and gross > 1e-12:
                    w_tgt = w_tgt * (leverage_cap / gross)
            new_u = w_tgt * xk
            tc = tcost * float(np.sum(np.abs(new_u - current_u)))
            current_u = new_u
        else:
            tc = 0.0

        wealth[k + 1] = xk + float(np.dot(current_u, path["ret"][k])) - tc
        w = current_u / max(abs(float(xk)), 1e-12)
        gross_lev[k] = float(np.sum(np.abs(w)))
        cash_w[k] = float(1.0 - np.sum(w))

    return {
        "wealth": wealth,
        "gross_leverage": gross_lev,
        "cash_weight": cash_w,
    }


def _simulate_center_only(center_agent, path, filt_params, T_years, dt, x0, p0, leverage_cap):
    n = path["ret"].shape[0]
    belief = compute_belief_path(path["logret"], filt_params=filt_params, dt=dt, p0=p0)
    wealth = np.empty(n + 1, dtype=float)
    wealth[0] = x0
    gross_lev = np.empty(n, dtype=float)
    cash_w = np.empty(n, dtype=float)

    for k in range(n):
        xk = wealth[k]
        tk = path["t"][k]
        pk = belief[k]
        u = np.asarray(center_agent.policy_mean(tk, xk, pk), dtype=float)
        if leverage_cap is not None:
            gross = np.sum(np.abs(u)) / max(abs(float(xk)), 1e-12)
            if gross > leverage_cap and gross > 1e-12:
                u = u * (leverage_cap / gross)
        wealth[k + 1] = xk + float(np.dot(u, path["ret"][k]))
        w = u / max(abs(float(xk)), 1e-12)
        gross_lev[k] = float(np.sum(np.abs(w)))
        cash_w[k] = float(1.0 - np.sum(w))

    return {"wealth": wealth, "gross_leverage": gross_lev, "cash_weight": cash_w}


def _simulate_static(weights, path, leverage_cap, x0):
    weights = apply_leverage_cap_to_weights(weights, leverage_cap)
    n = path["ret"].shape[0]
    wealth = np.empty(n + 1, dtype=float)
    wealth[0] = x0
    gross_lev = np.empty(n, dtype=float)
    cash_w = np.empty(n, dtype=float)
    for k in range(n):
        xk = wealth[k]
        u = weights * xk
        wealth[k + 1] = xk + float(np.dot(u, path["ret"][k]))
        gross_lev[k] = float(np.sum(np.abs(weights)))
        cash_w[k] = float(1.0 - np.sum(weights))
    return {"wealth": wealth, "gross_leverage": gross_lev, "cash_weight": cash_w}


def _summary(results: List[Dict]) -> pd.DataFrame:
    rows = []
    methods = sorted({r["method"] for r in results})
    for method in methods:
        sub = [r for r in results if r["method"] == method]
        xT = np.asarray([r["wealth"][-1] for r in sub], dtype=float)
        rows.append(
            {
                "method": method,
                "mean_terminal": float(np.mean(xT)),
                "std_terminal": float(np.std(xT, ddof=1)),
                "p05_terminal": float(np.quantile(xT, 0.05)),
                "p50_terminal": float(np.quantile(xT, 0.50)),
                "p95_terminal": float(np.quantile(xT, 0.95)),
                "avg_gross_leverage": float(np.mean([np.mean(r["gross_leverage"]) for r in sub])),
                "avg_cash_weight": float(np.mean([np.mean(r["cash_weight"]) for r in sub])),
            }
        )
    return pd.DataFrame(rows).sort_values("method").reset_index(drop=True)


def _save_plots(results: List[Dict], outdir: Path):
    methods = sorted({r["method"] for r in results})

    fig = plt.figure(figsize=(8, 6))
    for m in methods:
        sub = [r for r in results if r["method"] == m]
        x = np.asarray([r["wealth"][-1] for r in sub], dtype=float)
        plt.scatter(np.std(x, ddof=1), np.mean(x), s=80)
        plt.text(np.std(x, ddof=1), np.mean(x), f" {m}", va="center")
    plt.xlabel("std terminal wealth")
    plt.ylabel("mean terminal wealth")
    plt.title("Mean vs Std terminal wealth")
    fig.tight_layout()
    fig.savefig(outdir / "mean_std_scatter.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    data = [np.asarray([r["wealth"][-1] for r in results if r["method"] == m], dtype=float) for m in methods]
    plt.boxplot(data, labels=methods, showfliers=False)
    plt.ylabel("terminal wealth")
    plt.title("Terminal wealth boxplot")
    fig.tight_layout()
    fig.savefig(outdir / "terminal_boxplot.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    for m in methods:
        x = np.asarray([r["wealth"][-1] for r in results if r["method"] == m], dtype=float)
        plt.hist(x, bins=30, alpha=0.35, label=m)
        plt.axvline(float(np.mean(x)), linestyle="--", linewidth=1.2)
    plt.xlabel("terminal wealth")
    plt.ylabel("count")
    plt.title("Terminal wealth histogram")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "terminal_hist.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(11, 6))
    for m in methods:
        mats = [r["wealth"] for r in results if r["method"] == m]
        t = [r["t"] for r in results if r["method"] == m][0]
        mat = np.stack(mats, axis=0)
        mu = np.mean(mat, axis=0)
        sd = np.std(mat, axis=0, ddof=0)
        plt.plot(t, mu, label=m)
        plt.fill_between(t, mu - sd, mu + sd, alpha=0.15, label="_nolegend_")
    plt.xlabel("time (years)")
    plt.ylabel("discounted wealth")
    plt.title("Average wealth path ±1 std")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "avg_wealth.png", dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_run_dir", type=str, required=True)
    ap.add_argument("--stage1_checkpoint", type=str, required=True)
    ap.add_argument("--fixed_ckpt", type=str, default=None)
    ap.add_argument("--resid_ckpt", type=str, default=None)
    ap.add_argument("--boundary_ckpt", type=str, default=None)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--n_paths", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=1 / 252)
    ap.add_argument("--z", type=float, default=1.1)
    ap.add_argument("--x0", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.5)
    ap.add_argument("--tcost", type=float, default=0.002)
    ap.add_argument("--rebalance_steps", type=int, default=1)
    ap.add_argument("--leverage_cap", type=float, default=None)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    true_params = default_true_params()
    filt_params = FilterParams(
        mu1=true_params.mu1,
        mu2=true_params.mu2,
        Sigma=true_params.Sigma,
        lam1=true_params.lam1,
        lam2=true_params.lam2,
        r=true_params.r,
    )

    center_agent = build_agent_from_checkpoint(
        ckpt_path=Path(args.stage1_checkpoint),
        run_dir=Path(args.stage1_run_dir),
        T_years=args.T,
        dt=args.dt,
        a_max=2.0,
        z=args.z,
        r=true_params.r,
        device=args.device,
    )

    fixed_policy, fixed_cfg = _load_band(args.fixed_ckpt, args.device)
    resid_policy, resid_cfg = _load_band(args.resid_ckpt, args.device)
    boundary_policy, boundary_cfg = _load_band(args.boundary_ckpt, args.device)

    results: List[Dict] = []
    w_ew = np.full(2, 0.5, dtype=float)
    w_gmv = gmv_weights(true_params)
    w_mv = mv_target_weights(true_params, x0=args.x0, z=args.z, T_years=args.T)

    for path_id in range(args.n_paths):
        path = generate_test_path(
            true_params,
            T_years=args.T,
            dt=args.dt,
            seed=args.seed + 10000 + path_id,
        )

        sims = {
            "CenterOnly": _simulate_center_only(center_agent, path, filt_params, args.T, args.dt, args.x0, args.p0, args.leverage_cap),
            "EW": _simulate_static(w_ew, path, args.leverage_cap, args.x0),
            "MinVar": _simulate_static(w_gmv, path, args.leverage_cap, args.x0),
            "MeanVar": _simulate_static(w_mv, path, args.leverage_cap, args.x0),
        }
        if fixed_policy is not None:
            sims["FixedBand"] = _simulate_band_policy(
                center_agent=center_agent,
                band_policy=fixed_policy,
                mode="fixed_center_band",
                path=path,
                filt_params=filt_params,
                T_years=args.T,
                dt=args.dt,
                z=args.z,
                x0=args.x0,
                p0=args.p0,
                tcost=args.tcost,
                rebalance_steps=args.rebalance_steps,
                leverage_cap=args.leverage_cap,
                init_halfwidth=float(fixed_cfg["init_halfwidth"]),
                gamma_risk=float(fixed_cfg.get("gamma_risk", 5.0)),
                qvi_width_floor=float(fixed_cfg.get("qvi_width_floor", 1e-4)),
                width_mode=str(fixed_cfg.get("width_mode", "qvi_scale")),
                device=args.device,
            )
        if resid_policy is not None:
            sims["ResidualBand"] = _simulate_band_policy(
                center_agent=center_agent,
                band_policy=resid_policy,
                mode="residual_center_band",
                path=path,
                filt_params=filt_params,
                T_years=args.T,
                dt=args.dt,
                z=args.z,
                x0=args.x0,
                p0=args.p0,
                tcost=args.tcost,
                rebalance_steps=args.rebalance_steps,
                leverage_cap=args.leverage_cap,
                init_halfwidth=float(resid_cfg["init_halfwidth"]),
                gamma_risk=float(resid_cfg.get("gamma_risk", 5.0)),
                qvi_width_floor=float(resid_cfg.get("qvi_width_floor", 1e-4)),
                width_mode=str(resid_cfg.get("width_mode", "qvi_scale")),
                device=args.device,
            )
        if boundary_policy is not None:
            sims["BoundaryRL"] = _simulate_band_policy(
                center_agent=center_agent,
                band_policy=boundary_policy,
                mode="boundary_rl",
                path=path,
                filt_params=filt_params,
                T_years=args.T,
                dt=args.dt,
                z=args.z,
                x0=args.x0,
                p0=args.p0,
                tcost=args.tcost,
                rebalance_steps=args.rebalance_steps,
                leverage_cap=args.leverage_cap,
                init_halfwidth=float(boundary_cfg["init_halfwidth"]),
                gamma_risk=float(boundary_cfg.get("gamma_risk", 5.0)),
                qvi_width_floor=float(boundary_cfg.get("qvi_width_floor", 1e-4)),
                width_mode=str(boundary_cfg.get("width_mode", "qvi_scale")),
                device=args.device,
            )

        for method, sim in sims.items():
            results.append({"method": method, "path_id": path_id, "t": path["t"], **sim})

    summary = _summary(results)
    summary.to_csv(outdir / "eval_summary.csv", index=False)
    _save_plots(results, outdir)

    rows = []
    for r in results:
        rows.append(
            {
                "method": r["method"],
                "path_id": r["path_id"],
                "terminal_wealth": float(r["wealth"][-1]),
                "avg_gross_leverage": float(np.mean(r["gross_leverage"])),
                "avg_cash_weight": float(np.mean(r["cash_weight"])),
            }
        )
    pd.DataFrame(rows).to_csv(outdir / "eval_terminal_by_path.csv", index=False)

    with open(outdir / "eval_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage1_run_dir": str(Path(args.stage1_run_dir).resolve()),
                "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
                "fixed_ckpt": None if args.fixed_ckpt is None else str(Path(args.fixed_ckpt).resolve()),
                "resid_ckpt": None if args.resid_ckpt is None else str(Path(args.resid_ckpt).resolve()),
                "boundary_ckpt": None if args.boundary_ckpt is None else str(Path(args.boundary_ckpt).resolve()),
                "n_paths": args.n_paths,
                "seed": args.seed,
                "T": args.T,
                "dt": args.dt,
                "z": args.z,
                "x0": args.x0,
                "p0": args.p0,
                "tcost": args.tcost,
                "rebalance_steps": args.rebalance_steps,
                "leverage_cap": args.leverage_cap,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()