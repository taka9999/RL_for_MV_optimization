from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .eval_compare import (
    apply_leverage_cap_to_weights,
    build_agent_from_checkpoint,
    compute_belief_path,
    default_true_params,
    ew_weights,
    generate_test_path,
    gmv_weights,
    mv_target_weights,
)
from .filtering import FilterParams
from .stage2_models import BoundaryCorrectionNet, DirectBoundaryNet, Stage2DNNConfig
from .utils import set_seed


def _obs_vec(
    t: float,
    x: float,
    p: float,
    w_cur: np.ndarray,
    center_w: np.ndarray,
) -> np.ndarray:
    tau = max(0.0, 1.0 - t)
    diff = w_cur - center_w
    return np.asarray(
        [tau, x, p, w_cur[0], w_cur[1], center_w[0], center_w[1], diff[0], diff[1]],
        dtype=float,
    )


def _qvi_base_width(
    center_w: np.ndarray,
    Sigma: np.ndarray,
    kappa: float,
    gamma_risk: float,
    width_floor: float = 1e-4,
) -> np.ndarray:
    w = np.asarray(center_w, dtype=float).reshape(2,)
    Sigma = np.asarray(Sigma, dtype=float).reshape(2, 2)
    Sigma = 0.5 * (Sigma + Sigma.T)
    Sw = Sigma @ w
    quad = float(w @ Sw)
    diag = np.diag(Sigma)
    Dii = (w ** 2) * np.maximum(diag - 2.0 * Sw + quad, 0.0)
    Gammaii = np.maximum(float(gamma_risk) * diag, 1e-12)
    delta = ((max(float(kappa), 1e-12) * Dii) / Gammaii) ** (1.0 / 3.0)
    return np.maximum(delta, float(width_floor))


def _project_to_band(w_cur: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(w_cur, lower), upper)


def _load_stage2_model(path: str, device: str):
    ckpt = torch.load(path, map_location=device)
    model_cfg = Stage2DNNConfig(**ckpt["model_cfg"])
    state = ckpt["model_state_dict"]
    # Detect model type from state_dict keys.
    # BoundaryCorrectionNet:
    #   head_lower.*, head_upper.*
    # DirectBoundaryNet:
    #   head_lower_gap.*, head_upper_gap.*
    if any(k.startswith("head_lower_gap") for k in state.keys()):
        model = DirectBoundaryNet(model_cfg).to(device=device, dtype=torch.float64)
        model_type = "direct_boundary"
    else:
        model = BoundaryCorrectionNet(model_cfg).to(device=device, dtype=torch.float64)
        model_type = "band"

    model.load_state_dict(state)
    model.eval()
    return model, ckpt["train_cfg"], model_type

def _simulate_center_only(center_agent, path, filt_params, T_years, dt, x0):
    n = path["ret"].shape[0]
    belief = compute_belief_path(path["logret"], filt_params=filt_params, dt=dt, p0=0.5)
    wealth = np.empty(n + 1, dtype=float)
    wealth[0] = x0
    gross_lev = np.empty(n, dtype=float)
    cash_w = np.empty(n, dtype=float)
    turnover = np.empty(n, dtype=float)
    prev_u = np.zeros(2, dtype=float)

    for k in range(n):
        xk = wealth[k]
        tk = path["t"][k]
        pk = belief[k]
        u = np.asarray(center_agent.policy_mean(tk, xk, pk), dtype=float)
        denom = max(abs(float(xk)), 1e-12)
        w = u / denom
        gross_lev[k] = float(np.sum(np.abs(w)))
        cash_w[k] = float(1.0 - np.sum(w))
        turnover[k] = float(np.sum(np.abs(u - prev_u)) / denom)
        prev_u = u.copy()
        disc_ret = np.exp(path["logret"][k] - filt_params.r * dt) - 1.0
        wealth[k + 1] = xk + float(np.dot(u, disc_ret))
    return {
        "wealth": wealth,
        "gross_lev": gross_lev,
        "cash_w": cash_w,
        "turnover": turnover,
    }

def _simulate_center_only_with_cost(
    center_agent,
    path,
    filt_params,
    T_years,
    dt,
    x0,
    tcost,
    rebalance_every=1,
):
    n = path["ret"].shape[0]
    belief = compute_belief_path(path["logret"], filt_params=filt_params, dt=dt, p0=0.5)
    wealth = np.empty(n + 1, dtype=float)
    wealth[0] = x0
    current_u = np.zeros(2, dtype=float)
    gross_lev = np.empty(n, dtype=float)
    cash_w = np.empty(n, dtype=float)
    turnover = np.empty(n, dtype=float)

    for k in range(n):
        xk = wealth[k]
        tk = path["t"][k]
        pk = belief[k]
        denom = max(abs(float(xk)), 1e-12)

        if k % rebalance_every == 0:
            target_u = np.asarray(center_agent.policy_mean(tk, xk, pk), dtype=float)
            tc = tcost * float(np.sum(np.abs(target_u - current_u)))
            new_u = target_u
            turnover[k] = float(np.sum(np.abs(new_u - current_u)) / denom)
            current_u = new_u
        else:
            tc = 0.0
            turnover[k] = 0.0

        w = current_u / denom
        gross_lev[k] = float(np.sum(np.abs(w)))
        cash_w[k] = float(1.0 - np.sum(w))

        disc_ret = np.exp(path["logret"][k] - filt_params.r * dt) - 1.0
        wealth[k + 1] = xk + float(np.dot(current_u, disc_ret)) - tc

    return {
        "wealth": wealth,
        "gross_lev": gross_lev,
        "cash_w": cash_w,
        "turnover": turnover,
    }


def _simulate_center_only_fixed_band(
    center_agent,
    path,
    filt_params,
    T_years,
    dt,
    x0,
    tcost,
    halfwidth=0.05,
):
    n = path["ret"].shape[0]
    belief = compute_belief_path(path["logret"], filt_params=filt_params, dt=dt, p0=0.5)
    wealth = np.empty(n + 1, dtype=float)
    wealth[0] = x0
    current_u = np.zeros(2, dtype=float)
    gross_lev = np.empty(n, dtype=float)
    cash_w = np.empty(n, dtype=float)
    turnover = np.empty(n, dtype=float)

    for k in range(n):
        xk = wealth[k]
        tk = path["t"][k]
        pk = belief[k]
        center_u = np.asarray(center_agent.policy_mean(tk, xk, pk), dtype=float)
        denom = max(abs(float(xk)), 1e-12)
        center_w = center_u / denom
        w_cur = current_u / denom

        lower = center_w - halfwidth
        upper = center_w + halfwidth
        w_tgt = np.minimum(np.maximum(w_cur, lower), upper)
        new_u = w_tgt * xk
        tc = tcost * float(np.sum(np.abs(new_u - current_u)))
        turnover[k] = float(np.sum(np.abs(new_u - current_u)) / denom)
        current_u = new_u

        gross_lev[k] = float(np.sum(np.abs(w_tgt)))
        cash_w[k] = float(1.0 - np.sum(w_tgt))

        disc_ret = np.exp(path["logret"][k] - filt_params.r * dt) - 1.0
        wealth[k + 1] = xk + float(np.dot(current_u, disc_ret)) - tc

    return {
        "wealth": wealth,
        "gross_lev": gross_lev,
        "cash_w": cash_w,
        "turnover": turnover,
    }

def _simulate_dnn_band(center_agent, model, train_cfg, path, filt_params, T_years, dt, x0, tcost):
    n = path["ret"].shape[0]
    belief = compute_belief_path(path["logret"], filt_params=filt_params, dt=dt, p0=0.5)
    wealth = np.empty(n + 1, dtype=float)
    wealth[0] = x0
    current_u = np.zeros(2, dtype=float)
    gross_lev = np.empty(n, dtype=float)
    cash_w = np.empty(n, dtype=float)
    turnover = np.empty(n, dtype=float)

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    for k in range(n):
        xk = wealth[k]
        tk = path["t"][k]
        pk = belief[k]

        center_u = np.asarray(center_agent.policy_mean(tk, xk, pk), dtype=float)
        denom = max(abs(float(xk)), 1e-12)
        center_w = center_u / denom
        w_cur = current_u / denom

        obs_np = _obs_vec(tk / max(T_years, 1e-12), xk, pk, w_cur, center_w)
        obs_t = torch.as_tensor(obs_np, dtype=model_dtype, device=model_device).unsqueeze(0)

        lower_scale_t, upper_scale_t = model(obs_t)
        lower_scale = lower_scale_t.squeeze(0)
        upper_scale = upper_scale_t.squeeze(0)

        base_width_t = torch.as_tensor(
            _qvi_base_width(
                center_w=center_w,
                Sigma=filt_params.Sigma,
                kappa=tcost,
                gamma_risk=float(train_cfg["gamma_risk"]),
                width_floor=float(train_cfg["qvi_width_floor"]),
            ),
            dtype=model_dtype,
            device=model_device,
        )
        center_t = torch.as_tensor(center_w, dtype=model_dtype, device=model_device)
        lower = (center_t - base_width_t * lower_scale).detach().cpu().numpy()
        upper = (center_t + base_width_t * upper_scale).detach().cpu().numpy()

        w_tgt = _project_to_band(w_cur, lower, upper)
        new_u = w_tgt * xk
        tc = tcost * float(np.sum(np.abs(new_u - current_u)))
        gross_lev[k] = float(np.sum(np.abs(w_tgt)))
        cash_w[k] = float(1.0 - np.sum(w_tgt))
        turnover[k] = float(np.sum(np.abs(new_u - current_u)) / max(abs(float(xk)), 1e-12))
        current_u = new_u

        disc_ret = np.exp(path["logret"][k] - filt_params.r * dt) - 1.0
        wealth[k + 1] = xk + float(np.dot(current_u, disc_ret)) - tc

    return {
        "wealth": wealth,
        "gross_lev": gross_lev,
        "cash_w": cash_w,
        "turnover": turnover,
    }

def _simulate_direct_boundary(center_agent, model, train_cfg, path, filt_params, T_years, dt, x0, tcost):
    n = path["ret"].shape[0]
    belief = compute_belief_path(path["logret"], filt_params=filt_params, dt=dt, p0=0.5)
    wealth = np.empty(n + 1, dtype=float)
    wealth[0] = x0
    current_u = np.zeros(2, dtype=float)
    gross_lev = np.empty(n, dtype=float)
    cash_w = np.empty(n, dtype=float)
    turnover = np.empty(n, dtype=float)

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    for k in range(n):
        xk = wealth[k]
        tk = path["t"][k]
        pk = belief[k]

        center_u = np.asarray(center_agent.policy_mean(tk, xk, pk), dtype=float)
        denom = max(abs(float(xk)), 1e-12)
        center_w = center_u / denom
        w_cur = current_u / denom

        obs_np = _obs_vec(tk / max(T_years, 1e-12), xk, pk, w_cur, center_w)
        obs_t = torch.as_tensor(obs_np, dtype=model_dtype, device=model_device).unsqueeze(0)

        lower_gap_t, upper_gap_t = model(obs_t)
        lower_gap = lower_gap_t.squeeze(0).detach().cpu().numpy()
        upper_gap = upper_gap_t.squeeze(0).detach().cpu().numpy()

        lower = center_w - lower_gap
        upper = center_w + upper_gap

        # direct-boundary rebalancing rule
        w_tgt = np.minimum(np.maximum(w_cur, lower), upper)
        new_u = w_tgt * xk
        tc = tcost * float(np.sum(np.abs(new_u - current_u)))
        gross_lev[k] = float(np.sum(np.abs(w_tgt)))
        cash_w[k] = float(1.0 - np.sum(w_tgt))
        turnover[k] = float(np.sum(np.abs(new_u - current_u)) / max(abs(float(xk)), 1e-12))
        current_u = new_u

        disc_ret = np.exp(path["logret"][k] - filt_params.r * dt) - 1.0
        wealth[k + 1] = xk + float(np.dot(current_u, disc_ret)) - tc

    return {
        "wealth": wealth,
        "gross_lev": gross_lev,
        "cash_w": cash_w,
        "turnover": turnover,
    }

def _simulate_static(weights, path, filt_params, dt, x0, tcost=0.0, rebalance_every=1):
    n = path["ret"].shape[0]
    wealth = np.empty(n + 1, dtype=float)
    wealth[0] = x0
    gross_lev = np.empty(n, dtype=float)
    cash_w = np.empty(n, dtype=float)
    turnover = np.empty(n, dtype=float)
    #prev_u = np.zeros(2, dtype=float)
    current_u = np.zeros(2, dtype=float)
    for k in range(n):
        xk = wealth[k]
        #u = weights * xk
        denom = max(abs(float(xk)), 1e-12)
        if k % rebalance_every == 0:
            new_u = weights * xk
            tc = tcost * float(np.sum(np.abs(new_u - current_u)))
            turnover[k] = float(np.sum(np.abs(new_u - current_u)) / denom)
            current_u = new_u
        else:
            tc = 0.0
            turnover[k] = 0.0
        u = current_u
        gross_lev[k] = float(np.sum(np.abs(weights)))
        cash_w[k] = float(1.0 - np.sum(weights))
        #turnover[k] = float(np.sum(np.abs(u - prev_u)) / max(abs(float(xk)), 1e-12))
        #prev_u = u.copy()
        disc_ret = np.exp(path["logret"][k] - filt_params.r * dt) - 1.0
        wealth[k + 1] = xk + float(np.dot(u, disc_ret)) - tc
    return {
        "wealth": wealth,
        "gross_lev": gross_lev,
        "cash_w": cash_w,
        "turnover": turnover,
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_run_dir", type=str, required=True)
    ap.add_argument("--stage1_checkpoint", type=str, required=True)
    ap.add_argument("--stage2_checkpoint", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--n_paths", type=int, default=500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=1 / 252)
    ap.add_argument("--z", type=float, default=1.2)
    ap.add_argument("--x0", type=float, default=1.0)
    ap.add_argument("--tcost", type=float, default=0.002)
    ap.add_argument("--monthly_steps", type=int, default=21)
    ap.add_argument("--include_center_fixed_band", action="store_true")
    ap.add_argument("--center_fixed_band_halfwidth", type=float, default=0.05)
    ap.add_argument("--lev_cap", type=float, default=None)
    ap.add_argument("--show_wealth_std", dest="show_wealth_std", action="store_true")
    ap.add_argument("--hide_wealth_std", dest="show_wealth_std", action="store_false")
    ap.set_defaults(show_wealth_std=True)
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
        a_max=1.0,
        z=args.z,
        r=true_params.r,
        device=args.device,
    )
    model, train_cfg, model_type = _load_stage2_model(args.stage2_checkpoint, args.device)

    w_ew = np.full(2, 0.5, dtype=float)
    w_gmv = gmv_weights(true_params)
    w_mv = mv_target_weights(true_params, x0=args.x0, z=args.z, T_years=args.T)

    # Align static baselines to the same leverage cap, if requested.
    if args.lev_cap is not None:
        w_ew = apply_leverage_cap_to_weights(w_ew, args.lev_cap)
        w_gmv = apply_leverage_cap_to_weights(w_gmv, args.lev_cap)
        w_mv = apply_leverage_cap_to_weights(w_mv, args.lev_cap)

    rows = []
    methods = ["CenterOnly_Daily", "CenterOnly_Monthly", "DNNBand", "EW_Monthly", "MinVar_Monthly", "MeanVar_Monthly"]
    if args.include_center_fixed_band:
        methods.append("CenterOnly_FixedBand")
    wealth_by_method = {m: [] for m in methods}
    grosslev_by_method = {m: [] for m in methods}
    cash_by_method = {m: [] for m in methods}
    turnover_by_method = {m: [] for m in methods}

    for i in range(args.n_paths):
        path = generate_test_path(
            true_params,
            T_years=args.T,
            dt=args.dt,
            seed=args.seed + 10000 + i,
        )

        sims = {
            "CenterOnly_Daily": _simulate_center_only_with_cost(
                center_agent, path, filt_params, args.T, args.dt, args.x0, args.tcost, rebalance_every=1
            ),
            "CenterOnly_Monthly": _simulate_center_only_with_cost(
                center_agent, path, filt_params, args.T, args.dt, args.x0, args.tcost, rebalance_every=args.monthly_steps
            ),
            "EW_Monthly": _simulate_static(
                w_ew, path, filt_params, args.dt, args.x0, tcost=args.tcost, rebalance_every=args.monthly_steps
            ),
            "MinVar_Monthly": _simulate_static(
                w_gmv, path, filt_params, args.dt, args.x0, tcost=args.tcost, rebalance_every=args.monthly_steps
            ),
            "MeanVar_Monthly": _simulate_static(
                w_mv, path, filt_params, args.dt, args.x0, tcost=args.tcost, rebalance_every=args.monthly_steps
            ),
        }
        if args.include_center_fixed_band:
            sims["CenterOnly_FixedBand"] = _simulate_center_only_fixed_band(
                center_agent,
                path,
                filt_params,
                args.T,
                args.dt,
                args.x0,
                args.tcost,
                halfwidth=args.center_fixed_band_halfwidth,
            )
        if model_type == "direct_boundary":
            sims["DNNBand"] = _simulate_direct_boundary(
                center_agent, model, train_cfg, path, filt_params, args.T, args.dt, args.x0, args.tcost
            )
        else:
            sims["DNNBand"] = _simulate_dnn_band(
                center_agent, model, train_cfg, path, filt_params, args.T, args.dt, args.x0, args.tcost
            )

        for method, sim in sims.items():
            wealth_by_method[method].append(sim["wealth"])
            grosslev_by_method[method].append(sim["gross_lev"])
            cash_by_method[method].append(sim["cash_w"])
            turnover_by_method[method].append(sim["turnover"])
            rows.append(
                {
                    "method": method,
                    "path_id": i,
                    "terminal_wealth": float(sim["wealth"][-1]),
                    "gross_lev": float(np.mean(sim["gross_lev"])),
                    "cash_w": float(np.mean(sim["cash_w"])),
                    "turnover": float(np.mean(sim["turnover"])),
                    #"gross_lev": float(sim["gross_lev"][-1]),
                    #"cash_w": float(sim["cash_w"][-1]),
                    #"turnover": float(sim["turnover"][-1]),
                }
            )

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("method", as_index=False)
        .agg(
            mean_terminal=("terminal_wealth", "mean"),
            std_terminal=("terminal_wealth", "std"),
            median_terminal=("terminal_wealth", "median"),
            mean_gross_lev=("gross_lev", "mean"),
            mean_cash_w=("cash_w", "mean"),
            mean_turnover=("turnover", "mean"),
        )
    )
    summary.to_csv(outdir / "eval_summary.csv", index=False)
    df.to_csv(outdir / "eval_terminal_by_path.csv", index=False)

    fig = plt.figure(figsize=(8, 6))
    for method, mats in wealth_by_method.items():
        mat = np.stack(mats, axis=0)
        mu = np.mean(mat, axis=0)
        sd = np.std(mat, axis=0, ddof=0)
        t = np.linspace(0.0, args.T, mat.shape[1])
        plt.plot(t, mu, label=method)
        if args.show_wealth_std:
            plt.fill_between(t, mu - sd, mu + sd, alpha=0.15, label="_nolegend_")
    plt.xlabel("time (years)")
    plt.ylabel("discounted wealth")
    plt.title("Average wealth path ±1 std")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "avg_wealth.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    for method, mats in grosslev_by_method.items():
        mat = np.stack(mats, axis=0)
        mu = np.mean(mat, axis=0)
        plt.plot(np.linspace(0.0, args.T, mat.shape[1]), mu, label=method)
    plt.xlabel("time (years)")
    plt.ylabel("gross leverage")
    plt.title("Average gross leverage")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "avg_gross_lev.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    for method, mats in cash_by_method.items():
        mat = np.stack(mats, axis=0)
        mu = np.mean(mat, axis=0)
        plt.plot(np.linspace(0.0, args.T, mat.shape[1]), mu, label=method)
    plt.xlabel("time (years)")
    plt.ylabel("cash weight")
    plt.title("Average cash weight")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "avg_cash_weight.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    for method, mats in turnover_by_method.items():
        mat = np.stack(mats, axis=0)
        mu = np.mean(mat, axis=0)
        plt.plot(np.linspace(0.0, args.T, mat.shape[1]), mu, label=method)
    plt.xlabel("time (years)")
    plt.ylabel("turnover")
    plt.title("Average turnover")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "avg_turnover.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 6))
    for method, grp in df.groupby("method"):
        x = grp["terminal_wealth"].to_numpy()
        n, bins, patches = plt.hist(x, bins=30, alpha=0.35, label="_nolegend_")
        color = patches[0].get_facecolor() if len(patches) > 0 else "C0"
        mean_x = float(np.mean(x))
        plt.axvline(
            mean_x,
            linestyle="--",
            linewidth=1.5,
            color=color,
            label=f"{method} mean={mean_x:.3f}",
        )
    plt.xlabel("terminal wealth")
    plt.ylabel("count")
    plt.title("Terminal wealth histogram")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "terminal_hist.png", dpi=200)
    plt.close(fig)

    with open(outdir / "eval_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage1_run_dir": str(Path(args.stage1_run_dir).resolve()),
                "stage1_checkpoint": str(Path(args.stage1_checkpoint).resolve()),
                "stage2_checkpoint": str(Path(args.stage2_checkpoint).resolve()),
                "stage2_model_type": model_type,
                "n_paths": args.n_paths,
                "seed": args.seed,
                "T": args.T,
                "dt": args.dt,
                "z": args.z,
                "x0": args.x0,
                "tcost": args.tcost,
                "monthly_steps": args.monthly_steps,
                "include_center_fixed_band": args.include_center_fixed_band,
                "center_fixed_band_halfwidth": args.center_fixed_band_halfwidth,
                "lev_cap": args.lev_cap,
                "show_wealth_std": args.show_wealth_std,
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    main()