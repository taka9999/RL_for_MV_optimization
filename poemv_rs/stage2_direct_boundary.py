from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional
from multiprocessing import Pool
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from .eval_compare import (
    build_agent_from_checkpoint,
    compute_belief_path,
    default_true_params,
    generate_test_path,
)
from .filtering import FilterParams
from .stage2_models import DirectBoundaryNet, Stage2DNNConfig
from .utils import set_seed


@dataclass
class TrainDirectBoundaryConfig:
    T_years: float = 1.0
    dt: float = 1 / 252
    z: float = 1.2
    x0: float = 1.0
    p0: float = 0.5
    tcost: float = 0.002
    hidden: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-6
    iters: int = 2000
    episodes_per_iter: int = 32
    gamma_risk: float = 5.0
    #qvi_width_floor: float = 1e-4
    #boundary_anchor_coef: float = 1e-2
    turnover_coef: float = 0.0
    utility_kind: str = "log"      # log | sqrt | power
    utility_gamma: float = 2.0     # used only if utility_kind == "power"
    utility_scale: float = 1.0
    utility_shift: float = 0.0
    lr_step_size: int = 500
    lr_decay: float = 0.5
    gap_l2_coef: float = 0.0
    gross_lev_coef: float = 0.0
    val_every: int = 50
    val_n_paths: int = 128
    precompute_center_path: bool = True
    num_workers: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

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

def _utility_torch(
    w_t: torch.Tensor,
    kind: str,
    gamma: float,
    eps: float = 1e-10,
) -> torch.Tensor:
    x = torch.clamp(w_t, min=eps)
    if kind == "log":
        return torch.log(x)
    if kind == "sqrt":
        return torch.sqrt(x)
    if kind == "power":
        g = float(gamma)
        if abs(g - 1.0) < 1e-12:
            return torch.log(x)
        return (x ** (1.0 - g)) / (1.0 - g)
    raise ValueError(f"Unknown utility_kind: {kind}")

def _make_train_sample(args):
    """
    Worker for parallel path/belief/center prefetch.
    Returns a dict with:
        path, belief, center_w_path
    """
    seed_i, T_years, dt, p0, filt_params_dict, stage1_run_dir_str, stage1_ckpt_str, z, device, precompute_center_path = args
    true_params = default_true_params()
    filt_params = FilterParams(**filt_params_dict)
    path = generate_test_path(
        true_params,
        T_years=T_years,
        dt=dt,
        seed=seed_i,
    )
    belief = compute_belief_path(
        path["logret"],
        filt_params=filt_params,
        dt=dt,
        p0=p0,
    )
    center_w_path = None
    if precompute_center_path:
        center_agent = build_agent_from_checkpoint(
            ckpt_path=Path(stage1_ckpt_str),
            run_dir=Path(stage1_run_dir_str),
            T_years=T_years,
            dt=dt,
            a_max=1.0,
            z=z,
            r=true_params.r,
            device=device,
        )

        # Approximation: u*(t,x,p) ≈ x w*(t,p), so use x=1 path as benchmark center weights.
        center_w_path = []
        for k in range(path["ret"].shape[0]):
            tk = path["t"][k]
            pk = belief[k]
            u_star = np.asarray(center_agent.policy_mean(tk, 1.0, pk), dtype=float)
            center_w_path.append(u_star.copy())
        center_w_path = np.asarray(center_w_path, dtype=float)

    return {"path": path, "belief": belief, "center_w_path": center_w_path}

def _prefetch_train_batch(
    *,
    seeds,
    T_years: float,
    dt: float,
    p0: float,
    filt_params: FilterParams,
    stage1_run_dir: Path,
    stage1_checkpoint: Path,
    z: float,
    device: str,
    precompute_center_path: bool,
    num_workers: int,
):
    filt_params_dict = {
        "mu1": filt_params.mu1,
        "mu2": filt_params.mu2,
        "Sigma": filt_params.Sigma,
        "lam1": filt_params.lam1,
        "lam2": filt_params.lam2,
        "r": filt_params.r,
    }
    jobs = [
        (
            s,
            T_years,
            dt,
            p0,
            filt_params_dict,
            str(stage1_run_dir.resolve()),
            str(stage1_checkpoint.resolve()),
            z,
            device,
            precompute_center_path,
        )
        for s in seeds
    ]

    if num_workers is None or num_workers <= 0:
        return [_make_train_sample(j) for j in jobs]

    with Pool(processes=num_workers) as pool:
        out = pool.map(_make_train_sample, jobs)
    return out

def simulate_episode_direct_boundary(
    *,
    center_agent,
    model: DirectBoundaryNet,
    path: Dict[str, np.ndarray],
    belief: np.ndarray,
    center_w_path: Optional[np.ndarray],
    filt_params: FilterParams,
    cfg: TrainDirectBoundaryConfig,
) -> Dict:
    n = path["ret"].shape[0]
    #belief = compute_belief_path(path["logret"], filt_params=filt_params, dt=cfg.dt, p0=cfg.p0)

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype

    wealth_list = [torch.as_tensor(cfg.x0, dtype=model_dtype, device=model_device)]
    current_u = torch.zeros(2, dtype=model_dtype, device=model_device)
    r_t = torch.as_tensor(filt_params.r, dtype=model_dtype, device=model_device)
    logret_t_all = torch.as_tensor(path["logret"], dtype=model_dtype, device=model_device)

    lower_gap_all = []
    upper_gap_all = []
    turnover_all = []
    gross_lev_all = []
    gap_l2_all = []

    for k in range(n):
        xk_t = wealth_list[-1]
        #xk_float = float(xk_t.detach().cpu())
        #tk = path["t"][k]
        #pk = belief[k]
        #center_u_np = np.asarray(center_agent.policy_mean(tk, xk_float, pk), dtype=float)
        #center_u_t = torch.as_tensor(center_u_np, dtype=model_dtype, device=model_device)
        #denom_t = torch.clamp(torch.abs(xk_t), min=1e-12)
        #center_w_t = center_u_t / denom_t
        #center_w_np = center_w_path[k]
        #center_w_t = torch.as_tensor(center_w_np, dtype=model_dtype, device=model_device)
        if center_w_path is not None:
            center_w_np = center_w_path[k]
            center_w_t = torch.as_tensor(center_w_np, dtype=model_dtype, device=model_device)
        else:
            xk_float = float(xk_t.detach().cpu())
            tk = path["t"][k]
            pk = belief[k]
            center_u_np = np.asarray(center_agent.policy_mean(tk, xk_float, pk), dtype=float)
            center_u_t = torch.as_tensor(center_u_np, dtype=model_dtype, device=model_device)
            center_w_t = center_u_t / torch.clamp(torch.abs(xk_t), min=1e-12)
        w_cur_t = current_u / torch.clamp(torch.abs(xk_t), min=1e-12)

        obs_t = torch.stack(
            [
                #torch.as_tensor(max(0.0, 1.0 - tk / max(cfg.T_years, 1e-12)), dtype=model_dtype, device=model_device),
                torch.as_tensor(max(0.0, 1.0 - path["t"][k] / max(cfg.T_years, 1e-12)), dtype=model_dtype, device=model_device),
                xk_t,
                #torch.as_tensor(pk, dtype=model_dtype, device=model_device),
                torch.as_tensor(belief[k], dtype=model_dtype, device=model_device),
                w_cur_t[0],
                w_cur_t[1],
                center_w_t[0],
                center_w_t[1],
                w_cur_t[0] - center_w_t[0],
                w_cur_t[1] - center_w_t[1],
            ]
        ).unsqueeze(0)

        lower_gap_t, upper_gap_t = model(obs_t)
        lower_gap_t = lower_gap_t.squeeze(0)
        upper_gap_t = upper_gap_t.squeeze(0)

        # QVI prior only as anchor, not as projection rule
        #qvi_gap_np = _qvi_base_width(
        #    center_w=center_u_np / max(abs(xk_float), 1e-12),
        #    Sigma=filt_params.Sigma,
        #    kappa=cfg.tcost,
        #    gamma_risk=cfg.gamma_risk,
        #    width_floor=cfg.qvi_width_floor,
        #)
        #qvi_gap_t = torch.as_tensor(qvi_gap_np, dtype=model_dtype, device=model_device)

        lower_t = center_w_t - lower_gap_t
        upper_t = center_w_t + upper_gap_t

        # direct boundary rebalancing rule: pi^+ = min{u, max{l, pi^-}}
        w_tgt_t = torch.minimum(torch.maximum(w_cur_t, lower_t), upper_t)

        new_u_t = w_tgt_t * xk_t
        tc_t = torch.as_tensor(cfg.tcost, dtype=model_dtype, device=model_device) * torch.sum(torch.abs(new_u_t - current_u))
        current_u = new_u_t

        disc_ret_k_t = torch.exp(logret_t_all[k] - r_t * torch.as_tensor(cfg.dt, dtype=model_dtype, device=model_device)) - 1.0
        pnl_k_t = torch.dot(current_u, disc_ret_k_t)
        x_next_t = xk_t + pnl_k_t - tc_t
        wealth_list.append(x_next_t)

        turnover_all.append(torch.sum(torch.abs(new_u_t - w_cur_t * xk_t)) / torch.clamp(torch.abs(xk_t), min=1e-12))
        gross_lev_all.append(torch.sum(torch.abs(w_tgt_t)))
        lower_gap_all.append(lower_gap_t)
        upper_gap_all.append(upper_gap_t)
        #anchor_pen_all.append(((lower_gap_t - qvi_gap_t) ** 2).mean() + ((upper_gap_t - qvi_gap_t) ** 2).mean())
        gap_l2_all.append(lower_gap_t.pow(2).mean() + upper_gap_t.pow(2).mean())

    xT_t = wealth_list[-1]
    #loss_terminal = (xT_t - torch.as_tensor(cfg.z, dtype=model_dtype, device=model_device)) ** 2
    utility_t = _utility_torch(
        xT_t,
        kind=cfg.utility_kind,
        gamma=cfg.utility_gamma,
    )

    lower_gap_cat = torch.stack(lower_gap_all, dim=0)
    upper_gap_cat = torch.stack(upper_gap_all, dim=0)
    #anchor_pen = torch.stack(anchor_pen_all).mean()
    turnover_pen = torch.stack(turnover_all).mean()
    gross_lev_pen = torch.stack(gross_lev_all).mean()
    gap_l2_pen = torch.stack(gap_l2_all).mean()

    #loss = (
    #    loss_terminal
    #    + cfg.boundary_anchor_coef * anchor_pen
    #    + cfg.turnover_coef * turnover_pen
    #)
    # J0 is theta-independent, so minimizing J0 - U(WT) is equivalent to minimizing -U(WT)
    loss = (
        -cfg.utility_scale * (utility_t - torch.as_tensor(cfg.utility_shift, dtype=model_dtype, device=model_device))
        + cfg.turnover_coef * turnover_pen
        + cfg.gap_l2_coef * gap_l2_pen
        + cfg.gross_lev_coef * gross_lev_pen
    )

    return {
        "loss": loss,
        "terminal": float(xT_t.detach().cpu()),
        "wealth": torch.stack(wealth_list).detach().cpu().numpy(),
        "avg_turnover": float(turnover_pen.detach().cpu()),
        "avg_gross_lev": float(gross_lev_pen.detach().cpu()),
        "gap_l2_pen": float(gap_l2_pen.detach().cpu()),
        "utility": float(utility_t.detach().cpu()),
        "mean_lower_gap": float(lower_gap_cat.mean().detach().cpu()),
        "mean_upper_gap": float(upper_gap_cat.mean().detach().cpu()),
        #"anchor_pen": float(anchor_pen.detach().cpu()),
    }

def _run_validation(
    *,
    center_agent,
    model: DirectBoundaryNet,
    stage1_run_dir: Path,
    stage1_checkpoint: Path,
    filt_params: FilterParams,
    cfg: TrainDirectBoundaryConfig,
    seed: int,
    n_paths: int,
):
    seeds = [seed + 500000 + i for i in range(n_paths)]
    batch_samples = _prefetch_train_batch(
        seeds=seeds,
        T_years=cfg.T_years,
        dt=cfg.dt,
        p0=cfg.p0,
        filt_params=filt_params,
        stage1_run_dir=stage1_run_dir,
        stage1_checkpoint=stage1_checkpoint,
        z=cfg.z,
        device=cfg.device,
        precompute_center_path=cfg.precompute_center_path,
        num_workers=cfg.num_workers,
    )

    vals = []
    with torch.no_grad():
        for sample in batch_samples:
            vals.append(
                simulate_episode_direct_boundary(
                    center_agent=center_agent,
                    model=model,
                    path=sample["path"],
                    belief=sample["belief"],
                    center_w_path=sample["center_w_path"],
                    filt_params=filt_params,
                    cfg=cfg,
                )
            )

    return {
        "val_loss": float(np.mean([v["loss"].detach().cpu().item() if torch.is_tensor(v["loss"]) else float(v["loss"]) for v in vals])),
        "val_terminal": float(np.mean([v["terminal"] for v in vals])),
        "val_utility": float(np.mean([v["utility"] for v in vals])),
        "val_turnover": float(np.mean([v["avg_turnover"] for v in vals])),
        "val_gross_lev": float(np.mean([v["avg_gross_lev"] for v in vals])),
        "val_gap_l2": float(np.mean([v["gap_l2_pen"] for v in vals])),
    }

def train_direct_boundary(
    *,
    stage1_run_dir: Path,
    stage1_checkpoint: Path,
    outdir: Path,
    cfg: TrainDirectBoundaryConfig,
    seed: int,
):
    set_seed(seed)
    outdir.mkdir(parents=True, exist_ok=True)

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
        ckpt_path=stage1_checkpoint,
        run_dir=stage1_run_dir,
        T_years=cfg.T_years,
        dt=cfg.dt,
        a_max=1.0,
        z=cfg.z,
        r=true_params.r,
        device=cfg.device,
    )

    model_cfg = Stage2DNNConfig(obs_dim=9, hidden=cfg.hidden)
    model = DirectBoundaryNet(model_cfg).to(cfg.device, dtype=cfg.dtype)
    #opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(
        opt,
        step_size=cfg.lr_step_size,
        gamma=cfg.lr_decay,
    )

    rows = []
    best_val_utility = -np.inf

    for it in range(1, cfg.iters + 1):
        sims = []
        losses = []

        seeds = [seed + 10000 * it + b for b in range(cfg.episodes_per_iter)]
        batch_samples = _prefetch_train_batch(
            seeds=seeds,
            T_years=cfg.T_years,
            dt=cfg.dt,
            p0=cfg.p0,
            filt_params=filt_params,
            stage1_run_dir=stage1_run_dir,
            stage1_checkpoint=stage1_checkpoint,
            z=cfg.z,
            device=cfg.device,
            precompute_center_path=cfg.precompute_center_path,
            num_workers=cfg.num_workers,
        )

        #for b in range(cfg.episodes_per_iter):
        #    path = generate_test_path(
        #        true_params,
        #        T_years=cfg.T_years,
        #        dt=cfg.dt,
        #        seed=seed + 10000 * it + b,
        #    )
        for sample in batch_samples:
            path = sample["path"]
            belief = sample["belief"]
            center_w_path = sample["center_w_path"]
            sim = simulate_episode_direct_boundary(
                center_agent=center_agent,
                model=model,
                path=path,
                belief=belief,
                center_w_path=center_w_path,
                filt_params=filt_params,
                cfg=cfg,
            )
            sims.append(sim)
            losses.append(sim["loss"])

        loss = torch.stack(losses).mean()

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        scheduler.step()

        val_stats = {
            "val_loss": np.nan,
            "val_terminal": np.nan,
            "val_utility": np.nan,
            "val_turnover": np.nan,
            "val_gross_lev": np.nan,
            "val_gap_l2": np.nan,
        }
        if cfg.val_every > 0 and (it % cfg.val_every == 0 or it == cfg.iters):
            val_stats = _run_validation(
                center_agent=center_agent,
                model=model,
                stage1_run_dir=stage1_run_dir,
                stage1_checkpoint=stage1_checkpoint,
                filt_params=filt_params,
                cfg=cfg,
                seed=seed + 700000 + it,
                n_paths=cfg.val_n_paths,
            )
            if val_stats["val_utility"] > best_val_utility:
                best_val_utility = val_stats["val_utility"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_cfg": asdict(model_cfg),
                        "train_cfg": {
                            "T_years": cfg.T_years,
                            "dt": cfg.dt,
                            "z": cfg.z,
                            "x0": cfg.x0,
                            "p0": cfg.p0,
                            "tcost": cfg.tcost,
                            "hidden": cfg.hidden,
                            "lr": cfg.lr,
                            "weight_decay": cfg.weight_decay,
                            "iters": cfg.iters,
                            "episodes_per_iter": cfg.episodes_per_iter,
                            "gamma_risk": cfg.gamma_risk,
                            "turnover_coef": cfg.turnover_coef,
                            "utility_kind": cfg.utility_kind,
                            "utility_gamma": cfg.utility_gamma,
                            "utility_scale": cfg.utility_scale,
                            "utility_shift": cfg.utility_shift,
                            "lr_step_size": cfg.lr_step_size,
                            "lr_decay": cfg.lr_decay,
                            "gap_l2_coef": cfg.gap_l2_coef,
                            "gross_lev_coef": cfg.gross_lev_coef,
                            "val_every": cfg.val_every,
                            "val_n_paths": cfg.val_n_paths,
                            "precompute_center_path": cfg.precompute_center_path,
                            "num_workers": cfg.num_workers,
                            "best_iter": it,
                            "best_val_utility": best_val_utility,
                        },
                    },
                    outdir / "best_checkpoint.pt",
                )

        row = {
            "iter": it,
            "loss": float(loss.detach().cpu()),
            "mean_terminal": float(np.mean([s["terminal"] for s in sims])),
            "std_terminal": float(np.std([s["terminal"] for s in sims], ddof=0)),
            "avg_turnover": float(np.mean([s["avg_turnover"] for s in sims])),
            "avg_gross_lev": float(np.mean([s["avg_gross_lev"] for s in sims])),
            "gap_l2_pen": float(np.mean([s["gap_l2_pen"] for s in sims])),
            "mean_utility": float(np.mean([s["utility"] for s in sims])),
            "mean_lower_gap": float(np.mean([s["mean_lower_gap"] for s in sims])),
            "mean_upper_gap": float(np.mean([s["mean_upper_gap"] for s in sims])),
            #"anchor_pen": float(np.mean([s["anchor_pen"] for s in sims])),
            "lr": float(scheduler.get_last_lr()[0]),
            **val_stats,
        }
        rows.append(row)

        if it % 100 == 0:
            pd.DataFrame(rows).to_csv(outdir / "metrics.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "metrics.csv", index=False)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["iter"], df["mean_terminal"], label="mean_terminal")
    if "val_terminal" in df.columns:
        m = df["val_terminal"].notna()
        plt.plot(df.loc[m, "iter"], df.loc[m, "val_terminal"], label="val_terminal")
    plt.axhline(cfg.z, linestyle="--")
    plt.xlabel("iteration")
    plt.ylabel("mean terminal wealth")
    plt.title("Stage2 direct-boundary learning")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "learning_curve.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["iter"], df["mean_utility"], label="train_utility")
    if "val_utility" in df.columns:
        m = df["val_utility"].notna()
        plt.plot(df.loc[m, "iter"], df.loc[m, "val_utility"], label="val_utility")
    plt.xlabel("iteration")
    plt.ylabel("utility")
    plt.title("Stage2 direct-boundary utility")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "utility_curve.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["iter"], df["loss"], label="train_loss")
    if "val_loss" in df.columns:
        m = df["val_loss"].notna()
        plt.plot(df.loc[m, "iter"], df.loc[m, "val_loss"], label="val_loss")
    plt.xlabel("iteration")
    plt.ylabel("objective")
    plt.title("Stage2 direct-boundary objective")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "loss_curve.png", dpi=200)
    plt.close(fig)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_cfg": asdict(model_cfg),
            "train_cfg": {
                "T_years": cfg.T_years,
                "dt": cfg.dt,
                "z": cfg.z,
                "x0": cfg.x0,
                "p0": cfg.p0,
                "tcost": cfg.tcost,
                "hidden": cfg.hidden,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "iters": cfg.iters,
                "episodes_per_iter": cfg.episodes_per_iter,
                "gamma_risk": cfg.gamma_risk,
                #"qvi_width_floor": cfg.qvi_width_floor,
                #"boundary_anchor_coef": cfg.boundary_anchor_coef,
                "turnover_coef": cfg.turnover_coef,
                "utility_kind": cfg.utility_kind,
                "utility_gamma": cfg.utility_gamma,
                "utility_scale": cfg.utility_scale,
                "utility_shift": cfg.utility_shift,
                "lr_step_size": cfg.lr_step_size,
                "lr_decay": cfg.lr_decay,
                "gap_l2_coef": cfg.gap_l2_coef,
                "gross_lev_coef": cfg.gross_lev_coef,
                "val_every": cfg.val_every,
                "val_n_paths": cfg.val_n_paths,
                "precompute_center_path": cfg.precompute_center_path,
                "num_workers": cfg.num_workers,
            },
        },
        outdir / "checkpoint.pt",
    )

    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage1_run_dir": str(stage1_run_dir.resolve()),
                "stage1_checkpoint": str(stage1_checkpoint.resolve()),
                "seed": seed,
                "T_years": cfg.T_years,
                "dt": cfg.dt,
                "z": cfg.z,
                "x0": cfg.x0,
                "p0": cfg.p0,
                "tcost": cfg.tcost,
                "hidden": cfg.hidden,
                "lr": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "iters": cfg.iters,
                "episodes_per_iter": cfg.episodes_per_iter,
                "gamma_risk": cfg.gamma_risk,
                #"qvi_width_floor": cfg.qvi_width_floor,
                #"boundary_anchor_coef": cfg.boundary_anchor_coef,
                "turnover_coef": cfg.turnover_coef,
                "utility_kind": cfg.utility_kind,
                "utility_gamma": cfg.utility_gamma,
                "utility_scale": cfg.utility_scale,
                "utility_shift": cfg.utility_shift,
                "lr_step_size": cfg.lr_step_size,
                "lr_decay": cfg.lr_decay,
                "gap_l2_coef": cfg.gap_l2_coef,
                "gross_lev_coef": cfg.gross_lev_coef,
                "val_every": cfg.val_every,
                "val_n_paths": cfg.val_n_paths,
                "precompute_center_path": cfg.precompute_center_path, 
            },
            f,
            indent=2,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_run_dir", type=str, required=True)
    ap.add_argument("--stage1_checkpoint", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--episodes_per_iter", type=int, default=32)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=1 / 252)
    ap.add_argument("--z", type=float, default=1.2)
    ap.add_argument("--x0", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.5)
    ap.add_argument("--tcost", type=float, default=0.002)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)
    ap.add_argument("--gamma_risk", type=float, default=5.0)
    #ap.add_argument("--qvi_width_floor", type=float, default=1e-4)
    #ap.add_argument("--boundary_anchor_coef", type=float, default=1e-2)
    ap.add_argument("--turnover_coef", type=float, default=0.0)
    ap.add_argument("--utility_kind", type=str, choices=["log", "sqrt", "power"], default="log")
    ap.add_argument("--utility_gamma", type=float, default=2.0)
    ap.add_argument("--utility_scale", type=float, default=1.0)
    ap.add_argument("--utility_shift", type=float, default=0.0)
    ap.add_argument("--lr_step_size", type=int, default=500)
    ap.add_argument("--lr_decay", type=float, default=0.5)
    ap.add_argument("--gap_l2_coef", type=float, default=0.0)
    ap.add_argument("--gross_lev_coef", type=float, default=0.0)
    ap.add_argument("--val_every", type=int, default=50)
    ap.add_argument("--val_n_paths", type=int, default=128)
    ap.add_argument("--precompute_center_path", dest="precompute_center_path", action="store_true")
    ap.add_argument("--no_precompute_center_path", dest="precompute_center_path", action="store_false")
    ap.set_defaults(precompute_center_path=True)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    cfg = TrainDirectBoundaryConfig(
        T_years=args.T,
        dt=args.dt,
        z=args.z,
        x0=args.x0,
        p0=args.p0,
        tcost=args.tcost,
        hidden=args.hidden,
        lr=args.lr,
        weight_decay=args.weight_decay,
        iters=args.iters,
        episodes_per_iter=args.episodes_per_iter,
        gamma_risk=args.gamma_risk,
        #qvi_width_floor=args.qvi_width_floor,
        #boundary_anchor_coef=args.boundary_anchor_coef,
        turnover_coef=args.turnover_coef,
        utility_kind=args.utility_kind,
        utility_gamma=args.utility_gamma,
        utility_scale=args.utility_scale,
        utility_shift=args.utility_shift,
        lr_step_size=args.lr_step_size,
        lr_decay=args.lr_decay,
        gap_l2_coef=args.gap_l2_coef,
        gross_lev_coef=args.gross_lev_coef,
        val_every=args.val_every,
        val_n_paths=args.val_n_paths,
        precompute_center_path=args.precompute_center_path,
        num_workers=args.num_workers,
        device=args.device,
    )

    train_direct_boundary(
        stage1_run_dir=Path(args.stage1_run_dir),
        stage1_checkpoint=Path(args.stage1_checkpoint),
        outdir=Path(args.outdir),
        cfg=cfg,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()