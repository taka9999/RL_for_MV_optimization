from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

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
from .stage2_models import BoundaryCorrectionNet, Stage2DNNConfig
from .utils import set_seed


@dataclass
class TrainStage2DNNConfig:
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
    qvi_width_floor: float = 1e-4
    width_anchor_coef: float = 1e-2
    turnover_coef: float = 0.0
    softproj_temp: float = 10.0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64


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

def _soft_project_to_band_torch(
    w_cur: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    temp: float,
) -> torch.Tensor:
    """
    Smoothly map w_cur into [lower, upper].
    For each component:
        y = lower + (upper-lower) * sigmoid(temp * z)
    where z is the normalized location of w_cur relative to the interval center.
    """
    width = torch.clamp(upper - lower, min=1e-8)
    mid = 0.5 * (lower + upper)
    z = (w_cur - mid) / width
    s = torch.sigmoid(float(temp) * z)
    return lower + width * s

def simulate_episode_dnn(
    *,
    center_agent,
    model: BoundaryCorrectionNet,
    path: Dict[str, np.ndarray],
    filt_params: FilterParams,
    cfg: TrainStage2DNNConfig,
) -> Dict:
    n = path["ret"].shape[0]
    belief = compute_belief_path(path["logret"], filt_params=filt_params, dt=cfg.dt, p0=cfg.p0)
 
    turnover_list = []
    lower_scales_all = []
    upper_scales_all = []

    model_device = next(model.parameters()).device
    model_dtype = next(model.parameters()).dtype
    
    wealth_list = [torch.as_tensor(cfg.x0, dtype=model_dtype, device=model_device)]
    current_u = torch.zeros(2, dtype=model_dtype, device=model_device)
    Sigma_t = torch.as_tensor(filt_params.Sigma, dtype=model_dtype, device=model_device)
    r_t = torch.as_tensor(filt_params.r, dtype=model_dtype, device=model_device)
    logret_t_all = torch.as_tensor(path["logret"], dtype=model_dtype, device=model_device)
 
    for k in range(n):
        xk_t = wealth_list[-1]
        xk_float = float(xk_t.detach().cpu())
        tk = path["t"][k]
        pk = belief[k]

        center_u_np = np.asarray(center_agent.policy_mean(tk, xk_float, pk), dtype=float)
        center_u_t = torch.as_tensor(center_u_np, dtype=model_dtype, device=model_device)
        denom_t = torch.clamp(torch.abs(xk_t), min=1e-12)
        center_w_t = center_u_t / denom_t
        w_cur_t = current_u / denom_t

        obs_t = torch.stack(
            [
                torch.as_tensor(max(0.0, 1.0 - tk / max(cfg.T_years, 1e-12)), dtype=model_dtype, device=model_device),
                xk_t,
                torch.as_tensor(pk, dtype=model_dtype, device=model_device),
                w_cur_t[0],
                w_cur_t[1],
                center_w_t[0],
                center_w_t[1],
                w_cur_t[0] - center_w_t[0],
                w_cur_t[1] - center_w_t[1],
            ]
        ).unsqueeze(0)

        lower_scale_t, upper_scale_t = model(obs_t)
        lower_scale = lower_scale_t.squeeze(0)
        upper_scale = upper_scale_t.squeeze(0)

        base_width_np = _qvi_base_width(
            center_w=center_u_np / max(abs(xk_float), 1e-12),
            Sigma=filt_params.Sigma,
            kappa=cfg.tcost,
            gamma_risk=cfg.gamma_risk,
            width_floor=cfg.qvi_width_floor,
        )
        base_width_t = torch.as_tensor(base_width_np, dtype=model_dtype, device=model_device)

        center_t = center_w_t
        lower_t = center_t - base_width_t * lower_scale
        upper_t = center_t + base_width_t * upper_scale
        
        w_tgt_t = _soft_project_to_band_torch(
            w_cur=w_cur_t,
            lower=lower_t,
            upper=upper_t,
            temp=cfg.softproj_temp,
        )

        new_u_t = w_tgt_t * xk_t
        tc_t = torch.as_tensor(cfg.tcost, dtype=model_dtype, device=model_device) * torch.sum(torch.abs(new_u_t - current_u))
        current_u = new_u_t

        disc_ret_k_t = torch.exp(logret_t_all[k] - r_t * torch.as_tensor(cfg.dt, dtype=model_dtype, device=model_device)) - 1.0
        pnl_k_t = torch.dot(current_u, disc_ret_k_t)
        x_next_t = xk_t + pnl_k_t - tc_t
        wealth_list.append(x_next_t)

        turnover_list.append(torch.sum(torch.abs(new_u_t - w_cur_t * xk_t)) / torch.clamp(torch.abs(xk_t), min=1e-12))

        lower_scales_all.append(lower_scale)
        upper_scales_all.append(upper_scale)

    xT_t = wealth_list[-1]
    loss_terminal = (xT_t - torch.as_tensor(cfg.z, dtype=model_dtype, device=model_device)) ** 2 

    lower_scales_cat = torch.stack(lower_scales_all, dim=0) if lower_scales_all else torch.zeros((0, 2), dtype=model_dtype, device=model_device)
    upper_scales_cat = torch.stack(upper_scales_all, dim=0) if upper_scales_all else torch.zeros((0, 2), dtype=model_dtype, device=model_device)

    width_anchor = (
        ((lower_scales_cat - 1.0) ** 2).mean() + ((upper_scales_cat - 1.0) ** 2).mean()
        if lower_scales_cat.numel() > 0
        else torch.zeros((), dtype=model_dtype, device=model_device)
    )

    turnover_pen = torch.stack(turnover_list).mean() if turnover_list else torch.zeros((), dtype=model_dtype, device=model_device)

    loss = (
        loss_terminal
        + cfg.width_anchor_coef * width_anchor
        + cfg.turnover_coef * turnover_pen
    )

    return {
        "loss": loss,
        "terminal": float(xT_t.detach().cpu()),
        "wealth": torch.stack(wealth_list).detach().cpu().numpy(),
        "avg_turnover": float(turnover_pen.detach().cpu()),
        "mean_lower_scale": float(lower_scales_cat.mean().detach().cpu()) if lower_scales_cat.numel() > 0 else 1.0,
        "mean_upper_scale": float(upper_scales_cat.mean().detach().cpu()) if upper_scales_cat.numel() > 0 else 1.0,
        "width_anchor": float(width_anchor.detach().cpu()) if lower_scales_cat.numel() > 0 else 0.0,
    }


def train_stage2_dnn(
    *,
    stage1_run_dir: Path,
    stage1_checkpoint: Path,
    outdir: Path,
    cfg: TrainStage2DNNConfig,
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
    model = BoundaryCorrectionNet(model_cfg).to(cfg.device, dtype=cfg.dtype)
    opt = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    rows = []

    for it in range(1, cfg.iters + 1):
        sims = []
        losses = []

        for b in range(cfg.episodes_per_iter):
            path = generate_test_path(
                true_params,
                T_years=cfg.T_years,
                dt=cfg.dt,
                seed=seed + 10000 * it + b,
            )
            sim = simulate_episode_dnn(
                center_agent=center_agent,
                model=model,
                path=path,
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

        row = {
            "iter": it,
            "loss": float(loss.detach().cpu()),
            "mean_terminal": float(np.mean([s["terminal"] for s in sims])),
            "std_terminal": float(np.std([s["terminal"] for s in sims], ddof=0)),
            "avg_turnover": float(np.mean([s["avg_turnover"] for s in sims])),
            "mean_lower_scale": float(np.mean([s["mean_lower_scale"] for s in sims])),
            "mean_upper_scale": float(np.mean([s["mean_upper_scale"] for s in sims])),
            "width_anchor": float(np.mean([s["width_anchor"] for s in sims])),
        }
        rows.append(row)

        if it % 100 == 0:
            pd.DataFrame(rows).to_csv(outdir / "metrics.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "metrics.csv", index=False)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["iter"], df["mean_terminal"])
    plt.axhline(cfg.z, linestyle="--")
    plt.xlabel("iteration")
    plt.ylabel("mean terminal wealth")
    plt.title("Stage2 DNN learning")
    fig.tight_layout()
    fig.savefig(outdir / "learning_curve.png", dpi=200)
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
                "qvi_width_floor": cfg.qvi_width_floor,
                "width_anchor_coef": cfg.width_anchor_coef,
                "turnover_coef": cfg.turnover_coef,
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
                **{
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
                    "qvi_width_floor": cfg.qvi_width_floor,
                    "width_anchor_coef": cfg.width_anchor_coef,
                    "turnover_coef": cfg.turnover_coef,
                },
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
    ap.add_argument("--qvi_width_floor", type=float, default=1e-4)
    ap.add_argument("--width_anchor_coef", type=float, default=1e-2)
    ap.add_argument("--turnover_coef", type=float, default=0.0)
    ap.add_argument("--softproj_temp", type=float, default=10.0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    cfg = TrainStage2DNNConfig(
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
        width_anchor_coef=args.width_anchor_coef,
        turnover_coef=args.turnover_coef,
        softproj_temp=args.softproj_temp,
        device=args.device,
    )
    train_stage2_dnn(
        stage1_run_dir=Path(args.stage1_run_dir),
        stage1_checkpoint=Path(args.stage1_checkpoint),
        outdir=Path(args.outdir),
        cfg=cfg,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()