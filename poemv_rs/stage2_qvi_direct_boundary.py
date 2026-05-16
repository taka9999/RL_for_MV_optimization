from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.optim as optim

from .eval_compare import build_agent_from_checkpoint
from .filtering import FilterParams
from .stage2_models import DirectBoundaryNet, Stage2DNNConfig
from .stage2_direct_boundary import (
    _load_true_and_filter_params,
    _prefetch_train_batch,
    _qvi_base_width,
    _utility_torch,
)
from .utils import set_seed

def build_center_agent_auto(ckpt_path, run_dir, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    pi_sd = ckpt["pi_state_dict"]

    # old stage1 checkpoint
    if "rho1" in pi_sd or "rho2" in pi_sd:
        return build_agent_from_checkpoint(
            ckpt_path=ckpt_path,
            run_dir=run_dir,
            device=device,
        )

    # new cov-modes checkpoint
    if "signal1" in pi_sd or "signal2" in pi_sd:
        from .agent_stage1_cov_modes import POEMVAgentCovModes, TrainConfigCovModes
        # run_config.json から cfg を再構成
        # agent = POEMVAgentCovModes(cfg)
        # agent.vf.load_state_dict(ckpt["vf_state_dict"])
        # agent.pi.load_state_dict(ckpt["pi_state_dict"])
        # agent.omega.data.fill_(ckpt["omega"])
        # return agent

    raise ValueError("Unknown stage1 checkpoint format.")

@dataclass
class TrainQVIDirectBoundaryConfig:
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

    # qvi-guided band parameterization
    qvi_width_floor: float = 1e-4
    correction_clip: float = 1.5
    asymmetric_band: bool = True

    # main objective / regularizers
    turnover_coef: float = 0.0
    utility_kind: str = "log"      # log | sqrt | power
    utility_gamma: float = 2.0
    utility_scale: float = 1.0
    utility_shift: float = 0.0
    gap_l2_coef: float = 0.0
    gross_lev_coef: float = 0.0

    # qvi-inspired penalties
    qvi_prior_coef: float = 1e-2
    switch_residual_coef: float = 1e-2
    boundary_gap_coef: float = 0.0
    switch_margin_inside: float = 0.0
    switch_margin_outside: float = 0.0
    boundary_gap_tol: float = 0.0

    lr_step_size: int = 500
    lr_decay: float = 0.5
    val_every: int = 50
    val_n_paths: int = 128
    precompute_center_path: bool = True
    num_workers: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64


def _local_trade_vs_continue_scores(
    *,
    xk_t: torch.Tensor,
    current_u_t: torch.Tensor,
    new_u_t: torch.Tensor,
    disc_ret_k_t: torch.Tensor,
    tc_t: torch.Tensor,
    cfg: TrainQVIDirectBoundaryConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_continue_t = xk_t + torch.dot(current_u_t, disc_ret_k_t)
    x_trade_t = xk_t + torch.dot(new_u_t, disc_ret_k_t) - tc_t
    score_continue = _utility_torch(x_continue_t, kind=cfg.utility_kind, gamma=cfg.utility_gamma)
    score_trade = _utility_torch(x_trade_t, kind=cfg.utility_kind, gamma=cfg.utility_gamma)
    return score_continue, score_trade


def simulate_episode_qvi_direct_boundary(
    *,
    center_agent,
    model: DirectBoundaryNet,
    path: Dict[str, np.ndarray],
    belief: np.ndarray,
    center_w_path: Optional[np.ndarray],
    filt_params: FilterParams,
    cfg: TrainQVIDirectBoundaryConfig,
) -> Dict:
    n = path["ret"].shape[0]

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
    qvi_prior_all = []
    switch_residual_all = []
    boundary_gap_all = []

    for k in range(n):
        xk_t = wealth_list[-1]
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
                torch.as_tensor(max(0.0, 1.0 - path["t"][k] / max(cfg.T_years, 1e-12)), dtype=model_dtype, device=model_device),
                xk_t,
                torch.as_tensor(belief[k], dtype=model_dtype, device=model_device),
                w_cur_t[0],
                w_cur_t[1],
                center_w_t[0],
                center_w_t[1],
                w_cur_t[0] - center_w_t[0],
                w_cur_t[1] - center_w_t[1],
            ]
        ).unsqueeze(0)

        lower_raw_t, upper_raw_t = model(obs_t)
        lower_raw_t = lower_raw_t.squeeze(0)
        upper_raw_t = upper_raw_t.squeeze(0)

        qvi_gap_np = _qvi_base_width(
            center_w=center_w_t.detach().cpu().numpy(),
            Sigma=filt_params.Sigma,
            kappa=cfg.tcost,
            gamma_risk=cfg.gamma_risk,
            width_floor=cfg.qvi_width_floor,
        )
        qvi_gap_t = torch.as_tensor(qvi_gap_np, dtype=model_dtype, device=model_device)

        clip_c = float(cfg.correction_clip)
        lower_corr_t = torch.clamp(lower_raw_t, min=-clip_c, max=clip_c)
        upper_corr_t = torch.clamp(upper_raw_t, min=-clip_c, max=clip_c)

        if cfg.asymmetric_band:
            lower_gap_t = qvi_gap_t * torch.exp(lower_corr_t)
            upper_gap_t = qvi_gap_t * torch.exp(upper_corr_t)
        else:
            sym_corr_t = 0.5 * (lower_corr_t + upper_corr_t)
            sym_gap_t = qvi_gap_t * torch.exp(sym_corr_t)
            lower_gap_t = sym_gap_t
            upper_gap_t = sym_gap_t

        lower_t = center_w_t - lower_gap_t
        upper_t = center_w_t + upper_gap_t
        w_tgt_t = torch.minimum(torch.maximum(w_cur_t, lower_t), upper_t)

        new_u_t = w_tgt_t * xk_t
        tc_t = torch.as_tensor(cfg.tcost, dtype=model_dtype, device=model_device) * torch.sum(torch.abs(new_u_t - current_u))

        disc_ret_k_t = torch.exp(logret_t_all[k] - r_t * torch.as_tensor(cfg.dt, dtype=model_dtype, device=model_device)) - 1.0
        score_continue, score_trade = _local_trade_vs_continue_scores(
            xk_t=xk_t,
            current_u_t=current_u,
            new_u_t=new_u_t,
            disc_ret_k_t=disc_ret_k_t,
            tc_t=tc_t,
            cfg=cfg,
        )

        inside_mask_t = ((w_cur_t >= lower_t) & (w_cur_t <= upper_t)).all().to(model_dtype)
        outside_mask_t = 1.0 - inside_mask_t

        switch_inside_t = torch.relu(score_trade - score_continue + cfg.switch_margin_inside)
        switch_outside_t = torch.relu(score_continue - score_trade + cfg.switch_margin_outside)
        switch_residual_t = inside_mask_t * switch_inside_t + outside_mask_t * switch_outside_t

        signed_dev_low = lower_t - w_cur_t
        signed_dev_up = w_cur_t - upper_t
        outside_dist_t = torch.relu(signed_dev_low) + torch.relu(signed_dev_up)
        boundary_gap_t = torch.relu(outside_dist_t.mean() - cfg.boundary_gap_tol)

        qvi_prior_t = ((torch.log(torch.clamp(lower_gap_t, min=1e-12)) - torch.log(torch.clamp(qvi_gap_t, min=1e-12))) ** 2).mean() \
                    + ((torch.log(torch.clamp(upper_gap_t, min=1e-12)) - torch.log(torch.clamp(qvi_gap_t, min=1e-12))) ** 2).mean()

        current_u = new_u_t
        pnl_k_t = torch.dot(current_u, disc_ret_k_t)
        x_next_t = xk_t + pnl_k_t - tc_t
        wealth_list.append(x_next_t)

        turnover_all.append(torch.sum(torch.abs(new_u_t - w_cur_t * xk_t)) / torch.clamp(torch.abs(xk_t), min=1e-12))
        gross_lev_all.append(torch.sum(torch.abs(w_tgt_t)))
        lower_gap_all.append(lower_gap_t)
        upper_gap_all.append(upper_gap_t)
        gap_l2_all.append(lower_gap_t.pow(2).mean() + upper_gap_t.pow(2).mean())
        qvi_prior_all.append(qvi_prior_t)
        switch_residual_all.append(switch_residual_t)
        boundary_gap_all.append(boundary_gap_t)

    xT_t = wealth_list[-1]
    utility_t = _utility_torch(xT_t, kind=cfg.utility_kind, gamma=cfg.utility_gamma)

    lower_gap_cat = torch.stack(lower_gap_all, dim=0)
    upper_gap_cat = torch.stack(upper_gap_all, dim=0)
    turnover_pen = torch.stack(turnover_all).mean()
    gross_lev_pen = torch.stack(gross_lev_all).mean()
    gap_l2_pen = torch.stack(gap_l2_all).mean()
    qvi_prior_pen = torch.stack(qvi_prior_all).mean()
    switch_residual_pen = torch.stack(switch_residual_all).mean()
    boundary_gap_pen = torch.stack(boundary_gap_all).mean()

    loss = (
        -cfg.utility_scale * (utility_t - torch.as_tensor(cfg.utility_shift, dtype=model_dtype, device=model_device))
        + cfg.turnover_coef * turnover_pen
        + cfg.gap_l2_coef * gap_l2_pen
        + cfg.gross_lev_coef * gross_lev_pen
        + cfg.qvi_prior_coef * qvi_prior_pen
        + cfg.switch_residual_coef * switch_residual_pen
        + cfg.boundary_gap_coef * boundary_gap_pen
    )

    return {
        "loss": loss,
        "terminal": float(xT_t.detach().cpu()),
        "wealth": torch.stack(wealth_list).detach().cpu().numpy(),
        "avg_turnover": float(turnover_pen.detach().cpu()),
        "avg_gross_lev": float(gross_lev_pen.detach().cpu()),
        "gap_l2_pen": float(gap_l2_pen.detach().cpu()),
        "qvi_prior_pen": float(qvi_prior_pen.detach().cpu()),
        "switch_residual_pen": float(switch_residual_pen.detach().cpu()),
        "boundary_gap_pen": float(boundary_gap_pen.detach().cpu()),
        "utility": float(utility_t.detach().cpu()),
        "mean_lower_gap": float(lower_gap_cat.mean().detach().cpu()),
        "mean_upper_gap": float(upper_gap_cat.mean().detach().cpu()),
    }


def _run_validation(*, center_agent, model, stage1_run_dir: Path, stage1_checkpoint: Path, true_params,
                    filt_params: FilterParams, cfg: TrainQVIDirectBoundaryConfig, seed: int, filter_mode: str,
                    n_paths: int):
    seeds = [seed + 500000 + i for i in range(n_paths)]
    batch_samples = _prefetch_train_batch(
        seeds=seeds,
        T_years=cfg.T_years,
        dt=cfg.dt,
        p0=cfg.p0,
        true_params=true_params,
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
            vals.append(simulate_episode_qvi_direct_boundary(
                center_agent=center_agent,
                model=model,
                path=sample["path"],
                belief=sample["belief"],
                center_w_path=sample["center_w_path"],
                filt_params=filt_params,
                cfg=cfg,
            ))
    return {
        "val_loss": float(np.mean([v["loss"].detach().cpu().item() if torch.is_tensor(v["loss"]) else float(v["loss"]) for v in vals])),
        "val_terminal": float(np.mean([v["terminal"] for v in vals])),
        "val_utility": float(np.mean([v["utility"] for v in vals])),
        "val_turnover": float(np.mean([v["avg_turnover"] for v in vals])),
        "val_gross_lev": float(np.mean([v["avg_gross_lev"] for v in vals])),
        "val_gap_l2": float(np.mean([v["gap_l2_pen"] for v in vals])),
        "val_qvi_prior": float(np.mean([v["qvi_prior_pen"] for v in vals])),
        "val_switch_residual": float(np.mean([v["switch_residual_pen"] for v in vals])),
        "val_boundary_gap": float(np.mean([v["boundary_gap_pen"] for v in vals])),
    }


def train_qvi_direct_boundary(*, stage1_run_dir: Path, stage1_checkpoint: Path, outdir: Path,
                              cfg: TrainQVIDirectBoundaryConfig, seed: int,
                              filter_mode: str = "true_params"):
    set_seed(seed)
    outdir.mkdir(parents=True, exist_ok=True)

    true_params, filt_params = _load_true_and_filter_params(stage1_run_dir, filter_mode)
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
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(opt, step_size=cfg.lr_step_size, gamma=cfg.lr_decay)

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
            true_params=true_params,
            filt_params=filt_params,
            stage1_run_dir=stage1_run_dir,
            stage1_checkpoint=stage1_checkpoint,
            z=cfg.z,
            device=cfg.device,
            precompute_center_path=cfg.precompute_center_path,
            num_workers=cfg.num_workers,
        )

        for sample in batch_samples:
            sim = simulate_episode_qvi_direct_boundary(
                center_agent=center_agent,
                model=model,
                path=sample["path"],
                belief=sample["belief"],
                center_w_path=sample["center_w_path"],
                filt_params=filt_params,
                cfg=cfg,
            )
            sims.append(sim)
            losses.append(sim["loss"])

        loss = torch.stack(losses).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        scheduler.step()

        val_stats = {}
        if it % cfg.val_every == 0 or it == 1 or it == cfg.iters:
            val_stats = _run_validation(
                center_agent=center_agent,
                model=model,
                stage1_run_dir=stage1_run_dir,
                stage1_checkpoint=stage1_checkpoint,
                true_params=true_params,
                filt_params=filt_params,
                cfg=cfg,
                seed=seed,
                filter_mode=filter_mode,
                n_paths=cfg.val_n_paths,
            )
            if val_stats["val_utility"] > best_val_utility:
                best_val_utility = val_stats["val_utility"]
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "model_cfg": asdict(model_cfg),
                    "train_cfg": asdict(cfg),
                }, outdir / "best_checkpoint.pt")

        row = {
            "iter": it,
            "loss": float(loss.detach().cpu()),
            "mean_terminal": float(np.mean([s["terminal"] for s in sims])),
            "avg_turnover": float(np.mean([s["avg_turnover"] for s in sims])),
            "avg_gross_lev": float(np.mean([s["avg_gross_lev"] for s in sims])),
            "gap_l2_pen": float(np.mean([s["gap_l2_pen"] for s in sims])),
            "qvi_prior_pen": float(np.mean([s["qvi_prior_pen"] for s in sims])),
            "switch_residual_pen": float(np.mean([s["switch_residual_pen"] for s in sims])),
            "boundary_gap_pen": float(np.mean([s["boundary_gap_pen"] for s in sims])),
            "mean_utility": float(np.mean([s["utility"] for s in sims])),
            "mean_lower_gap": float(np.mean([s["mean_lower_gap"] for s in sims])),
            "mean_upper_gap": float(np.mean([s["mean_upper_gap"] for s in sims])),
            "lr": float(scheduler.get_last_lr()[0]),
            **val_stats,
        }
        rows.append(row)

        if it % 50 == 0 or it == 1:
            print(
                f"iter={it} loss={row['loss']:.6f} term={row['mean_terminal']:.4f} util={row['mean_utility']:.6f} "
                f"qvi_prior={row['qvi_prior_pen']:.6f} switch={row['switch_residual_pen']:.6f}"
            )
        if it % 100 == 0:
            pd.DataFrame(rows).to_csv(outdir / "metrics.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "metrics.csv", index=False)

    for col, fname, title in [
        ("mean_terminal", "learning_curve.png", "Stage2 QVI-direct terminal wealth"),
        ("mean_utility", "utility_curve.png", "Stage2 QVI-direct utility"),
        ("loss", "loss_curve.png", "Stage2 QVI-direct objective"),
        ("switch_residual_pen", "switch_residual_curve.png", "Stage2 QVI switch residual"),
    ]:
        fig = plt.figure(figsize=(8, 5))
        plt.plot(df["iter"], df[col], label=f"train_{col}")
        vcol = f"val_{col.replace('mean_', '')}" if col.startswith("mean_") else f"val_{col.replace('_pen','').replace('loss','loss')}"
        if vcol in df.columns:
            m = df[vcol].notna()
            plt.plot(df.loc[m, "iter"], df.loc[m, vcol], label=vcol)
        if col == "mean_terminal":
            plt.axhline(cfg.z, linestyle="--")
        plt.xlabel("iteration")
        plt.ylabel(col)
        plt.title(title)
        plt.legend()
        fig.tight_layout()
        fig.savefig(outdir / fname, dpi=200)
        plt.close(fig)

    torch.save(
        {"model_state_dict": model.state_dict(), "model_cfg": asdict(model_cfg), "train_cfg": asdict(cfg)},
        outdir / "checkpoint.pt",
    )
    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "stage1_run_dir": str(stage1_run_dir.resolve()),
            "stage1_checkpoint": str(stage1_checkpoint.resolve()),
            "seed": seed,
            **asdict(cfg),
            "filter_mode": filter_mode,
            "theory_tags": {
                "method_family": "direct_dnn",
                "control_structure": "qvi_guided_direct_boundary",
                "center": "stage1_frictionless_center_fixed",
                "qvi_layer": "base_width_plus_learned_correction",
                "switch_residual": "local_trade_vs_continue_surrogate",
            },
            "true_params": {
                "mu1": np.asarray(true_params.mu1, dtype=float).tolist(),
                "mu2": np.asarray(true_params.mu2, dtype=float).tolist(),
                "Sigma": np.asarray(true_params.Sigma, dtype=float).tolist(),
                "lam1": float(true_params.lam1),
                "lam2": float(true_params.lam2),
                "r": float(true_params.r),
            },
            "filter_params": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in filt_params.__dict__.items()},
        }, f, indent=2)


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
    ap.add_argument("--qvi_width_floor", type=float, default=1e-4)
    ap.add_argument("--correction_clip", type=float, default=1.5)
    ap.add_argument("--symmetric_band", dest="asymmetric_band", action="store_false")
    ap.add_argument("--asymmetric_band", dest="asymmetric_band", action="store_true")
    ap.set_defaults(asymmetric_band=True)
    ap.add_argument("--turnover_coef", type=float, default=0.0)
    ap.add_argument("--utility_kind", type=str, choices=["log", "sqrt", "power"], default="log")
    ap.add_argument("--utility_gamma", type=float, default=2.0)
    ap.add_argument("--utility_scale", type=float, default=1.0)
    ap.add_argument("--utility_shift", type=float, default=0.0)
    ap.add_argument("--lr_step_size", type=int, default=500)
    ap.add_argument("--lr_decay", type=float, default=0.5)
    ap.add_argument("--gap_l2_coef", type=float, default=0.0)
    ap.add_argument("--gross_lev_coef", type=float, default=0.0)
    ap.add_argument("--qvi_prior_coef", type=float, default=1e-2)
    ap.add_argument("--switch_residual_coef", type=float, default=1e-2)
    ap.add_argument("--boundary_gap_coef", type=float, default=0.0)
    ap.add_argument("--switch_margin_inside", type=float, default=0.0)
    ap.add_argument("--switch_margin_outside", type=float, default=0.0)
    ap.add_argument("--boundary_gap_tol", type=float, default=0.0)
    ap.add_argument("--val_every", type=int, default=50)
    ap.add_argument("--val_n_paths", type=int, default=128)
    ap.add_argument("--precompute_center_path", dest="precompute_center_path", action="store_true")
    ap.add_argument("--no_precompute_center_path", dest="precompute_center_path", action="store_false")
    ap.set_defaults(precompute_center_path=True)
    ap.add_argument("--filter_mode", type=str, choices=["true_params", "estimated_params"], default="true_params")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    cfg = TrainQVIDirectBoundaryConfig(
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
        qvi_width_floor=args.qvi_width_floor,
        correction_clip=args.correction_clip,
        asymmetric_band=args.asymmetric_band,
        turnover_coef=args.turnover_coef,
        utility_kind=args.utility_kind,
        utility_gamma=args.utility_gamma,
        utility_scale=args.utility_scale,
        utility_shift=args.utility_shift,
        lr_step_size=args.lr_step_size,
        lr_decay=args.lr_decay,
        gap_l2_coef=args.gap_l2_coef,
        gross_lev_coef=args.gross_lev_coef,
        qvi_prior_coef=args.qvi_prior_coef,
        switch_residual_coef=args.switch_residual_coef,
        boundary_gap_coef=args.boundary_gap_coef,
        switch_margin_inside=args.switch_margin_inside,
        switch_margin_outside=args.switch_margin_outside,
        boundary_gap_tol=args.boundary_gap_tol,
        val_every=args.val_every,
        val_n_paths=args.val_n_paths,
        precompute_center_path=args.precompute_center_path,
        num_workers=args.num_workers,
        device=args.device,
    )

    train_qvi_direct_boundary(
        stage1_run_dir=Path(args.stage1_run_dir),
        stage1_checkpoint=Path(args.stage1_checkpoint),
        outdir=Path(args.outdir),
        cfg=cfg,
        seed=args.seed,
        filter_mode=args.filter_mode,
    )


if __name__ == "__main__":
    main()
