from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from .utils import set_seed, safe_clip_p
from .eval_compare import (
    default_true_params,
    build_agent_from_checkpoint,
    generate_test_path,
    compute_belief_path,
)
from .filtering import FilterParams

@dataclass
class BandStage2Config:
    mode: str = "fixed_center_band"  # fixed_center_band | residual_center_band | boundary_rl | fixed_center_threshold
    T_years: float = 1.0
    dt: float = 1 / 252
    z: float = 1.1
    x0: float = 1.0
    p0: float = 0.5
    tcost: float = 0.002
    rebalance_steps: int = 1
    hidden: int = 64
    lr: float = 1e-4
    iters: int = 3000
    episodes_per_iter: int = 32
    baseline_beta: float = 0.9
    entropy_coef: float = 1e-3
    grad_clip: float = 1.0
    reward_gamma: float = 1.0
    gae_lambda: float = 0.95
    critic_hidden: int = 64
    critic_lr: float = 3e-4
    critic_coef: float = 1.0
    center_unfreeze_at: float = 0.7
    center_ramp_len: float = 0.2
    center_anchor_coef: float = 10.0
    boundary_warm_start_until: float = 0.4
    boundary_center_anchor_coef: float = 20.0
    boundary_width_anchor_coef: float = 20.0
    width_l2_coef: float = 1e-4
    init_halfwidth: float = 0.05
    gamma_risk: float = 5.0
    step_reward_coef: float = 1.0
    terminal_penalty_coef: float = 1.0
    diff_reward_vs_center: bool = False
    diff_reward_scale: float = 10.0
    qvi_width_floor: float = 1e-4
    width_mode: str = "qvi_scale"   # qvi_scale | free
    device: str = "cpu"
    dtype: torch.dtype = torch.float64

class BandPolicy(nn.Module):
    """
    Minimal RL band actor.

    mode=fixed_center_band:
        center = center_base
        learn only halfwidth

    mode=residual_center_band:
        center = center_base + gate * residual
        learn halfwidth + residual

    mode=boundary_rl:
        center = center_base + residual
        learn halfwidth + residual from start
    """
    def __init__(self, obs_dim: int, hidden: int = 64, init_halfwidth: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.head_center = nn.Linear(hidden, 2)
        self.head_width = nn.Linear(hidden, 2)
        self.head_threshold = nn.Linear(hidden, 1)
        self.log_std_center = nn.Parameter(torch.full((2,), -2.5, dtype=torch.float64))
        self.log_std_width = nn.Parameter(torch.full((2,), -2.0, dtype=torch.float64))
        self.log_std_threshold = nn.Parameter(torch.full((1,), -3.0, dtype=torch.float64))
        self.init_halfwidth = float(init_halfwidth)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor):
        h = self.net(obs)
        center_loc = self.head_center(h)
        width_loc = self.head_width(h)
        threshold_loc = self.head_threshold(h)
        return center_loc, width_loc, threshold_loc

    def dist(self, obs: torch.Tensor):
        center_loc, width_loc, threshold_loc = self.forward(obs)
        std_c = torch.exp(self.log_std_center).unsqueeze(0).expand_as(center_loc)
        std_w = torch.exp(self.log_std_width).unsqueeze(0).expand_as(width_loc)
        dc = torch.distributions.Normal(center_loc, std_c)
        dw = torch.distributions.Normal(width_loc, std_w)
        dt = torch.distributions.Normal(
            threshold_loc,
            torch.exp(self.log_std_threshold).unsqueeze(0).expand_as(threshold_loc),
        )
        return dc, dw, dt

class BandValue(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)

def _rebalance_steps(label: str | int) -> int:
    if isinstance(label, int):
        return int(label)
    if str(label).lower() == "daily":
        return 1
    if str(label).lower() == "monthly":
        return 21
    return int(label)


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

def _clip_to_box(w: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(w, lower), upper)

def _qvi_base_width(
    center_w: np.ndarray,
    Sigma: np.ndarray,
    kappa: float,
    gamma_risk: float,
    width_floor: float = 1e-4,
) -> np.ndarray:
    """
    QVI / small-cost local half-width proxy:
        delta_i ~ (kappa * D_ii / Gamma_ii)^(1/3)
    with
        D_ii = w_i^2 (Sigma_ii - 2 (Sigma w)_i + w' Sigma w)
        Gamma_ii ~ gamma * Sigma_ii

    We use the current center weight as w*.
    """
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

def _center_gate(progress: float, start: float, ramp_len: float, mode: str) -> float:
    if mode == "fixed_center_band":
        return 0.0
    if mode == "fixed_center_threshold":
        return 0.0
    if mode == "boundary_rl":
        return 1.0
    # residual_center_band: exact 0 before start, then linear ramp to 1
    if progress <= start:
        return 0.0
    if ramp_len <= 1e-12:
        return 1.0
    return float(np.clip((progress - start) / ramp_len, 0.0, 1.0))

def _boundary_warm_weight(progress: float, until: float) -> float:
    if until <= 1e-12:
        return 0.0
    if progress >= until:
        return 0.0
    # linearly decay from 1 -> 0 over [0, until]
    return float(np.clip(1.0 - progress / until, 0.0, 1.0))

def _stage2_action(
    policy: BandPolicy,
    mode: str,
    progress: float,
    center_w: np.ndarray,
    obs_np: np.ndarray,
    deterministic: bool,
    device: str,
    dtype: torch.dtype,
    init_halfwidth: float,
    Sigma: np.ndarray,
    tcost: float,
    gamma_risk: float,
    qvi_width_floor: float,
    width_mode: str,
    center_unfreeze_at: float,
    center_ramp_len: float,
):
    obs_t = torch.as_tensor(obs_np, dtype=dtype, device=device).unsqueeze(0)
    dc, dw, dt = policy.dist(obs_t)
    gate = _center_gate(progress, center_unfreeze_at, center_ramp_len, mode)

    if mode == "fixed_center_threshold":
        if deterministic:
            thr_raw = dt.mean.squeeze(0)
        else:
            thr_raw = dt.rsample().squeeze(0)
        threshold_logp = dt.log_prob(thr_raw).sum()
        threshold_entropy = dt.entropy().sum()
        width_raw = None
    else:
        # width branch is always active
        if deterministic:
            width_raw = dw.mean.squeeze(0)
        else:
            width_raw = dw.rsample().squeeze(0)
        threshold_logp = torch.zeros((), dtype=dtype, device=device)
        threshold_entropy = torch.zeros((), dtype=dtype, device=device)

    # center branch is inactive before unfreeze in residual mode, and always inactive in fixed mode
    if gate <= 0.0:
        resid_raw = torch.zeros(2, dtype=dtype, device=device)
        center_logp = torch.zeros((), dtype=dtype, device=device)
        center_entropy = torch.zeros((), dtype=dtype, device=device)
    else:
        if deterministic:
            resid_raw = dc.mean.squeeze(0)
        else:
            resid_raw = dc.rsample().squeeze(0)
        center_logp = gate * dc.log_prob(resid_raw).sum()
        center_entropy = gate * dc.entropy().sum()

    center_resid = gate * resid_raw
    #width = torch.nn.functional.softplus(width_raw) + float(init_halfwidth)
    if mode == "fixed_center_threshold":
        base_width_np = _qvi_base_width(
            center_w=center_w,
            Sigma=Sigma,
            kappa=tcost,
            gamma_risk=gamma_risk,
            width_floor=qvi_width_floor,
        )
        base_width = torch.as_tensor(base_width_np, dtype=dtype, device=device)
        # asset-wise threshold instead of collapsing to one scalar
        threshold_scale = torch.exp(0.25 * thr_raw)
        threshold_vec = threshold_scale * base_width
        center = torch.as_tensor(center_w, dtype=dtype, device=device)
        lower = center.clone()
        upper = center.clone()
        width = threshold_vec
        width_scale = torch.full_like(center, float(threshold_scale.detach().cpu()))
        width_pen_l2 = (threshold_scale - 1.0).pow(2).mean()
        width_logp = threshold_logp
        width_entropy = threshold_entropy
    elif width_mode == "qvi_scale":
        base_width_np = _qvi_base_width(
            center_w=center_w,
            Sigma=Sigma,
            kappa=tcost,
            gamma_risk=gamma_risk,
            width_floor=qvi_width_floor,
        )
        base_width = torch.as_tensor(base_width_np, dtype=dtype, device=device)
        # 1-centered proportional correction: width_scale=1 when width_raw=0
        width_scale = torch.exp(0.25 * width_raw)
        width = width_scale * base_width
        width_pen_l2 = torch.mean((width_scale - 1.0) ** 2)
    elif width_mode == "free":
        width = torch.nn.functional.softplus(width_raw) + float(init_halfwidth)
        width_scale = width / max(float(init_halfwidth), 1e-8)
        width_pen_l2 = torch.mean(width ** 2)
    else:
        raise ValueError(f"Unknown width_mode={width_mode}")

    center = torch.as_tensor(center_w, dtype=dtype, device=device) + center_resid
    lower = center - width
    upper = center + width

    if mode != "fixed_center_threshold":
        width_logp = dw.log_prob(width_raw).sum()
        width_entropy = dw.entropy().sum()
    logp = center_logp + width_logp
    entropy = center_entropy + width_entropy

    aux = {
        "center_resid_l2": float(torch.sum(center_resid.pow(2)).detach().cpu()),
        "mean_halfwidth": float(torch.mean(width).detach().cpu()),
        "mean_width_scale": float(torch.mean(width_scale).detach().cpu()),
        "width_pen_l2": float(width_pen_l2.detach().cpu()),
        "width_anchor_l2": float(torch.mean((width_scale - 1.0) ** 2).detach().cpu()),
        "threshold_0": float(width[0].detach().cpu()),
        "threshold_1": float(width[1].detach().cpu()),
        "threshold_mean": float(torch.mean(width).detach().cpu()),
        "gate": float(gate),
    }
    return (
        center.detach().cpu().numpy(),
        lower.detach().cpu().numpy(),
        upper.detach().cpu().numpy(),
        logp,
        entropy,
        aux,
    )

def simulate_stage2_episode(
    *,
    center_agent,
    band_policy: BandPolicy,
    mode: str,
    path: Dict[str, np.ndarray],
    filt_params: FilterParams,
    cfg: BandStage2Config,
    deterministic: bool,
    progress: float,
) -> Dict:
    n = path["ret"].shape[0]
    belief = compute_belief_path(path["logret"], filt_params=filt_params, dt=cfg.dt, p0=cfg.p0)

    wealth = np.empty(n + 1, dtype=float)
    wealth[0] = cfg.x0
    current_u = np.zeros(2, dtype=float)
    # optional center-only shadow path for difference reward
    wealth_center = np.empty(n + 1, dtype=float)
    wealth_center[0] = cfg.x0
    current_u_center = np.zeros(2, dtype=float)

    logps = []
    entropies = []
    center_resid_l2 = []
    mean_halfwidth = []
    mean_width_scale = []
    width_pen_l2 = []
    width_anchor_l2 = []
    turnover = []
    outside_frac = []
    center_dist = []
    mean_threshold = []
    step_rewards = []
    obs_list = []
    next_obs_list = []
    done_list = []
    pending_interval_reward = 0.0

    for k in range(n):
        xk = wealth[k]
        tk = path["t"][k]
        pk = belief[k]

        if (k % int(cfg.rebalance_steps)) == 0:
            center_u = np.asarray(center_agent.policy_mean(tk, xk, pk), dtype=float)
            denom = max(abs(float(xk)), 1e-12)
            center_w = center_u / denom
            w_cur = current_u / denom

            obs_np = _obs_vec(tk / max(cfg.T_years, 1e-12), xk, pk, w_cur, center_w)
            # close previous action interval at the moment a new action is chosen
            if len(obs_list) > 0:
                next_obs_list.append(obs_np.copy())
                done_list.append(0.0)
                step_rewards.append(pending_interval_reward)
                pending_interval_reward = 0.0
            center_hat, lower, upper, logp, entropy, aux = _stage2_action(
                band_policy,
                mode=mode,
                progress=progress,
                center_w=center_w,
                obs_np=obs_np,
                deterministic=deterministic,
                device=cfg.device,
                dtype=cfg.dtype,
                init_halfwidth=cfg.init_halfwidth,
                Sigma=filt_params.Sigma,
                tcost=cfg.tcost,
                gamma_risk=cfg.gamma_risk,
                qvi_width_floor=cfg.qvi_width_floor,
                width_mode=cfg.width_mode,
                center_unfreeze_at=cfg.center_unfreeze_at,
                center_ramp_len=cfg.center_ramp_len,
            )
            obs_list.append(obs_np.copy())

            if cfg.mode == "fixed_center_threshold":
                threshold_vec = np.array([aux["threshold_0"], aux["threshold_1"]], dtype=float)
                diff_vec = w_cur - center_w
                outside = bool(np.any(np.abs(diff_vec) > threshold_vec))
                # project to threshold boundary instead of snapping all the way to center
                clipped_diff = np.clip(diff_vec, -threshold_vec, threshold_vec)
                w_tgt = center_w + clipped_diff if outside else w_cur
            else:
                outside = np.any((w_cur < lower) | (w_cur > upper))
                w_tgt = w_cur if not outside else _clip_to_box(w_cur, lower, upper)

            new_u = w_tgt * xk
            tc = cfg.tcost * float(np.sum(np.abs(new_u - current_u)))

            current_u = new_u
            logps.append(logp)
            entropies.append(entropy)
            center_resid_l2.append(aux["center_resid_l2"])
            mean_halfwidth.append(aux["mean_halfwidth"])
            mean_width_scale.append(aux["mean_width_scale"])
            width_pen_l2.append(aux["width_pen_l2"])
            width_anchor_l2.append(aux["width_anchor_l2"])
            mean_threshold.append(aux["threshold_mean"])
            turnover.append(float(np.sum(np.abs(new_u - (w_cur * xk))) / max(abs(xk), 1e-12)))
            outside_frac.append(float(outside))
            center_dist.append(float(np.mean(np.abs(w_cur - center_w))))
        else:
            tc = 0.0
        # shadow center-only execution under the same path / belief
        if cfg.diff_reward_vs_center:
            xk_c = wealth_center[k]
            center_u_shadow = np.asarray(center_agent.policy_mean(tk, xk_c, pk), dtype=float)
            tc_center = cfg.tcost * float(np.sum(np.abs(center_u_shadow - current_u_center)))
            current_u_center = center_u_shadow
        else:
            tc_center = 0.0

        # Align Stage 2 with discounted-by-bank update used by Stage 1 logic
        disc_ret_k = np.exp(path["logret"][k] - filt_params.r * cfg.dt) - 1.0
        pnl_k = float(np.dot(current_u, disc_ret_k))
        wealth[k + 1] = xk + pnl_k - tc
        # per-step shaped reward:
        # realized discounted PnL - quadratic MV risk penalty - transaction cost
        w_post = current_u / max(abs(float(xk)), 1e-12)
        risk_pen_k = 0.5 * cfg.gamma_risk * float(xk) * float(w_post @ filt_params.Sigma @ w_post) * cfg.dt
        step_r = cfg.step_reward_coef * (pnl_k - risk_pen_k - tc)
        #step_rewards.append(step_r)
        if cfg.diff_reward_vs_center:
            pnl_center = float(np.dot(current_u_center, disc_ret_k))
            wealth_center[k + 1] = wealth_center[k] + pnl_center - tc_center
            w_center_post = current_u_center / max(abs(float(xk_c)), 1e-12)
            risk_pen_center = 0.5 * cfg.gamma_risk * float(xk_c) * float(w_center_post @ filt_params.Sigma @ w_center_post) * cfg.dt
            step_r_center = cfg.step_reward_coef * (pnl_center - risk_pen_center - tc_center)
            pending_interval_reward += cfg.diff_reward_scale * (step_r - step_r_center)
        else:
            pending_interval_reward += step_r

    xT = float(wealth[-1])
    terminal_pen = cfg.terminal_penalty_coef * ((xT - cfg.z) ** 2)
    #if len(step_rewards) > 0:
    #    step_rewards[-1] = step_rewards[-1] - terminal_pen
    #reward = float(np.sum(step_rewards))
    pending_interval_reward -= terminal_pen
    if cfg.diff_reward_vs_center:
        xT_center = float(wealth_center[-1])
        terminal_pen_center = cfg.terminal_penalty_coef * ((xT_center - cfg.z) ** 2)
        pending_interval_reward += cfg.diff_reward_scale * terminal_pen_center
    else:
        xT_center = np.nan
        terminal_pen_center = 0.0

    # close final action interval at episode end
    if len(obs_list) > 0:
        next_obs_list.append(np.zeros_like(obs_list[-1]))
        done_list.append(1.0)
        step_rewards.append(pending_interval_reward)

    reward = float(np.sum(step_rewards))

    return {
        "wealth": wealth,
        "belief": belief,
        "reward": reward,
        "logp": torch.stack(logps).sum() if len(logps) else torch.tensor(0.0, dtype=cfg.dtype, device=cfg.device),
        "entropy": torch.stack(entropies).mean() if len(entropies) else torch.tensor(0.0, dtype=cfg.dtype, device=cfg.device),
        "step_logps": torch.stack(logps) if len(logps) else torch.zeros((0,), dtype=cfg.dtype, device=cfg.device),
        "step_entropies": torch.stack(entropies) if len(entropies) else torch.zeros((0,), dtype=cfg.dtype, device=cfg.device),
        "obs": np.stack(obs_list, axis=0) if len(obs_list) else np.zeros((0, 9), dtype=float),
        "next_obs": np.stack(next_obs_list, axis=0) if len(next_obs_list) else np.zeros((0, 9), dtype=float),
        "dones": np.asarray(done_list, dtype=float),
        "step_rewards": np.asarray(step_rewards, dtype=float),
        "center_resid_l2": float(np.mean(center_resid_l2)) if center_resid_l2 else 0.0,
        "mean_halfwidth": float(np.mean(mean_halfwidth)) if mean_halfwidth else 0.0,
        "mean_width_scale": float(np.mean(mean_width_scale)) if mean_width_scale else 1.0,
        "width_pen_l2": float(np.mean(width_pen_l2)) if width_pen_l2 else 0.0,
        "width_anchor_l2": float(np.mean(width_anchor_l2)) if width_anchor_l2 else 0.0,
        "mean_threshold": float(np.mean(mean_threshold)) if mean_threshold else 0.0,
        "avg_turnover": float(np.mean(turnover)) if turnover else 0.0,
        "outside_frac": float(np.mean(outside_frac)) if outside_frac else 0.0,
        "center_dist": float(np.mean(center_dist)) if center_dist else 0.0,
        "mean_step_reward": float(np.mean(step_rewards)) if step_rewards else 0.0,
        "terminal_penalty": float(terminal_pen),
        "terminal_center": float(xT_center),
        "terminal_penalty_center": float(terminal_pen_center),
        "terminal": xT,
    }

def train_stage2(
    *,
    stage1_run_dir: Path,
    stage1_checkpoint: Path,
    outdir: Path,
    cfg: BandStage2Config,
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

    obs_dim = 9
    policy = BandPolicy(obs_dim=obs_dim, hidden=cfg.hidden, init_halfwidth=cfg.init_halfwidth).to(cfg.device, dtype=cfg.dtype)
    value_net = BandValue(obs_dim=obs_dim, hidden=cfg.critic_hidden).to(cfg.device, dtype=cfg.dtype)
    opt_actor = optim.Adam(policy.parameters(), lr=cfg.lr)
    opt_critic = optim.Adam(value_net.parameters(), lr=cfg.critic_lr)
    rows = []

    for it in range(1, cfg.iters + 1):
        progress = it / max(cfg.iters, 1)
        sims = []
        for b in range(cfg.episodes_per_iter):
            path = generate_test_path(
                true_params,
                T_years=cfg.T_years,
                dt=cfg.dt,
                seed=seed + 10000 * it + b,
            )
            sims.append(
                simulate_stage2_episode(
                    center_agent=center_agent,
                    band_policy=policy,
                    mode=cfg.mode,
                    path=path,
                    filt_params=filt_params,
                    cfg=cfg,
                    deterministic=False,
                    progress=progress,
                )
            )

        mean_reward = float(np.mean([s["reward"] for s in sims]))
        actor_losses = []
        critic_losses = []
        ent_terms = []

        for s in sims:
            if s["obs"].shape[0] == 0:
                continue
            obs_t = torch.as_tensor(s["obs"], dtype=cfg.dtype, device=cfg.device)
            next_obs_t = torch.as_tensor(s["next_obs"], dtype=cfg.dtype, device=cfg.device)
            done_t = torch.as_tensor(s["dones"], dtype=cfg.dtype, device=cfg.device)
            logp_t = s["step_logps"]
            ent_t = s["step_entropies"]
            rew_t = torch.as_tensor(s["step_rewards"], dtype=cfg.dtype, device=cfg.device)

            v_t = value_net(obs_t)
            with torch.no_grad():
                next_v_t = value_net(next_obs_t) * (1.0 - done_t)

            deltas = rew_t + cfg.reward_gamma * next_v_t - v_t
            
            adv = torch.zeros_like(deltas)
            gae = torch.zeros((), dtype=cfg.dtype, device=cfg.device)
            for i in range(len(deltas) - 1, -1, -1):
                gae = deltas[i] + cfg.reward_gamma * cfg.gae_lambda * (1.0 - done_t[i]) * gae
                adv[i] = gae

            ret_t = adv.detach() + v_t.detach()
            adv_norm = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)

            actor_losses.append(-(logp_t * adv_norm.detach()).mean())
            critic_losses.append(0.5 * torch.mean((v_t - ret_t) ** 2))
            ent_terms.append(ent_t.mean())

        if len(actor_losses) == 0:
            continue

        actor_loss = torch.stack(actor_losses).mean()
        critic_loss = torch.stack(critic_losses).mean()
        ent = torch.stack(ent_terms).mean()
        resid_pen = torch.as_tensor(
            np.mean([s["center_resid_l2"] for s in sims]), dtype=cfg.dtype, device=cfg.device
        )
        width_pen = torch.as_tensor(
            np.mean([s["width_pen_l2"] for s in sims]), dtype=cfg.dtype, device=cfg.device
        )
        width_anchor_pen = torch.as_tensor(
            np.mean([s["width_anchor_l2"] for s in sims]), dtype=cfg.dtype, device=cfg.device
        )

        loss_actor_total = actor_loss - cfg.entropy_coef * ent
        if cfg.mode == "residual_center_band":
            gate = _center_gate(progress, cfg.center_unfreeze_at, cfg.center_ramp_len, cfg.mode)
            loss_actor_total = loss_actor_total + gate * cfg.center_anchor_coef * resid_pen
        elif cfg.mode == "boundary_rl":
            #loss_actor_total = loss_actor_total + 0.1 * cfg.center_anchor_coef * resid_pen
            # paper-like advanced start:
            # keep boundary RL close to Stage1 center + QVI-width early on,
            # then gradually release it.
            warm_w = _boundary_warm_weight(progress, cfg.boundary_warm_start_until)
            loss_actor_total = (
                loss_actor_total
                + warm_w * cfg.boundary_center_anchor_coef * resid_pen
                + warm_w * cfg.boundary_width_anchor_coef * width_anchor_pen
            )
        elif cfg.mode == "fixed_center_threshold":
            # no center residual term; keep threshold close to QVI scale 1 early/always via width_pen
            pass
        loss_actor_total = loss_actor_total + cfg.width_l2_coef * width_pen

        opt_actor.zero_grad(set_to_none=True)
        loss_actor_total.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=cfg.grad_clip)
        opt_actor.step()

        opt_critic.zero_grad(set_to_none=True)
        (cfg.critic_coef * critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=cfg.grad_clip)
        opt_critic.step()

        row = {
            "iter": it,
            "mean_reward": mean_reward,
            "mean_terminal": float(np.mean([s["terminal"] for s in sims])),
            "std_terminal": float(np.std([s["terminal"] for s in sims], ddof=0)),
            "mean_step_reward": float(np.mean([s["mean_step_reward"] for s in sims])),
            "mean_terminal_penalty": float(np.mean([s["terminal_penalty"] for s in sims])),
            "mean_terminal_center": float(np.nanmean([s["terminal_center"] for s in sims])),
            "mean_terminal_gap": float(np.mean([s["terminal"] for s in sims]) - np.nanmean([s["terminal_center"] for s in sims])),
            "avg_turnover": float(np.mean([s["avg_turnover"] for s in sims])),
            "outside_frac": float(np.mean([s["outside_frac"] for s in sims])),
            "center_dist": float(np.mean([s["center_dist"] for s in sims])),
            "mean_halfwidth": float(np.mean([s["mean_halfwidth"] for s in sims])),
            "mean_width_scale": float(np.mean([s["mean_width_scale"] for s in sims])),
            "mean_threshold": float(np.mean([s["mean_threshold"] for s in sims])),
            "center_resid_l2": float(np.mean([s["center_resid_l2"] for s in sims])),
            "width_anchor_l2": float(np.mean([s["width_anchor_l2"] for s in sims])),
            "actor_loss": float(actor_loss.detach().cpu()),
            "critic_loss": float(critic_loss.detach().cpu()),
            "loss": float((loss_actor_total + cfg.critic_coef * critic_loss).detach().cpu()),
        }
        rows.append(row)

        if it % 100 == 0:
            pd.DataFrame(rows).to_csv(outdir / "metrics.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "metrics.csv", index=False)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["iter"], df["mean_terminal"], label="band")
    if "mean_terminal_center" in df.columns:
        plt.plot(df["iter"], df["mean_terminal_center"], label="center_only")
    plt.axhline(cfg.z, linestyle="--")
    plt.xlabel("iteration")
    plt.ylabel("mean terminal wealth")
    plt.title(f"Stage2 learning | {cfg.mode}")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "learning_curve.png", dpi=200)
    plt.close(fig)

    if "mean_terminal_gap" in df.columns:
        fig = plt.figure(figsize=(8, 5))
        plt.plot(df["iter"], df["mean_terminal_gap"], label="band - center_only")
        plt.axhline(0.0, linestyle="--")
        plt.xlabel("iteration")
        plt.ylabel("mean terminal gap")
        plt.title(f"Stage2 gap learning | {cfg.mode}")
        plt.legend()
        fig.tight_layout()
        fig.savefig(outdir / "learning_curve_gap.png", dpi=200)
        plt.close(fig)

    torch.save(
        {
            "band_policy_state_dict": policy.state_dict(),
            "value_net_state_dict": value_net.state_dict(),
            "stage2_cfg": {
                "mode": cfg.mode,
                "T_years": cfg.T_years,
                "dt": cfg.dt,
                "z": cfg.z,
                "x0": cfg.x0,
                "p0": cfg.p0,
                "tcost": cfg.tcost,
                "rebalance_steps": cfg.rebalance_steps,
                "hidden": cfg.hidden,
                "reward_gamma": cfg.reward_gamma,
                "gae_lambda": cfg.gae_lambda,
                "critic_hidden": cfg.critic_hidden,
                "critic_lr": cfg.critic_lr,
                "critic_coef": cfg.critic_coef,
                "center_unfreeze_at": cfg.center_unfreeze_at,
                "center_ramp_len": cfg.center_ramp_len,
                "center_anchor_coef": cfg.center_anchor_coef,
                "boundary_warm_start_until": cfg.boundary_warm_start_until,
                "boundary_center_anchor_coef": cfg.boundary_center_anchor_coef,
                "boundary_width_anchor_coef": cfg.boundary_width_anchor_coef,
                "width_l2_coef": cfg.width_l2_coef,
                "init_halfwidth": cfg.init_halfwidth,
                "gamma_risk": cfg.gamma_risk,
                "step_reward_coef": cfg.step_reward_coef,
                "terminal_penalty_coef": cfg.terminal_penalty_coef,
                "diff_reward_vs_center": cfg.diff_reward_vs_center,
                "diff_reward_scale": cfg.diff_reward_scale,
                "qvi_width_floor": cfg.qvi_width_floor,
                "width_mode": cfg.width_mode,
            },
        },
        outdir / "checkpoint.pt",
    )

    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage1_run_dir": str(stage1_run_dir.resolve()),
                "stage1_checkpoint": str(stage1_checkpoint.resolve()),
                "mode": cfg.mode,
                "seed": seed,
                "T": cfg.T_years,
                "dt": cfg.dt,
                "z": cfg.z,
                "x0": cfg.x0,
                "p0": cfg.p0,
                "tcost": cfg.tcost,
                "rebalance_steps": cfg.rebalance_steps,
                "hidden": cfg.hidden,
                "lr": cfg.lr,
                "iters": cfg.iters,
                "episodes_per_iter": cfg.episodes_per_iter,
                "baseline_beta": cfg.baseline_beta,
                "entropy_coef": cfg.entropy_coef,
                "reward_gamma": cfg.reward_gamma,
                "gae_lambda": cfg.gae_lambda,
                "critic_hidden": cfg.critic_hidden,
                "critic_lr": cfg.critic_lr,
                "critic_coef": cfg.critic_coef,
                "center_unfreeze_at": cfg.center_unfreeze_at,
                "center_ramp_len": cfg.center_ramp_len,
                "center_anchor_coef": cfg.center_anchor_coef,
                "boundary_warm_start_until": cfg.boundary_warm_start_until,
                "boundary_center_anchor_coef": cfg.boundary_center_anchor_coef,
                "boundary_width_anchor_coef": cfg.boundary_width_anchor_coef,
                "width_l2_coef": cfg.width_l2_coef,
                "init_halfwidth": cfg.init_halfwidth,
                "gamma_risk": cfg.gamma_risk,
                "step_reward_coef": cfg.step_reward_coef,
                "terminal_penalty_coef": cfg.terminal_penalty_coef,
                "diff_reward_vs_center": cfg.diff_reward_vs_center,
                "diff_reward_scale": cfg.diff_reward_scale,
                "qvi_width_floor": cfg.qvi_width_floor,
                "width_mode": cfg.width_mode,
            },
            f,
            indent=2,
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_run_dir", type=str, required=True)
    ap.add_argument("--stage1_checkpoint", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--mode", type=str, choices=["fixed_center_band", "residual_center_band", "boundary_rl", "fixed_center_threshold"], default="fixed_center_band")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=3000)
    ap.add_argument("--episodes_per_iter", type=int, default=32)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=1 / 252)
    ap.add_argument("--z", type=float, default=1.1)
    ap.add_argument("--x0", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.5)
    ap.add_argument("--tcost", type=float, default=0.002)
    ap.add_argument("--rebalance_steps", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--baseline_beta", type=float, default=0.9)
    ap.add_argument("--entropy_coef", type=float, default=1e-3)
    ap.add_argument("--reward_gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--critic_hidden", type=int, default=64)
    ap.add_argument("--critic_lr", type=float, default=3e-4)
    ap.add_argument("--critic_coef", type=float, default=1.0)
    ap.add_argument("--center_unfreeze_at", type=float, default=0.7)
    ap.add_argument("--center_ramp_len", type=float, default=0.2)
    ap.add_argument("--center_anchor_coef", type=float, default=10.0)
    ap.add_argument("--boundary_warm_start_until", type=float, default=0.4)
    ap.add_argument("--boundary_center_anchor_coef", type=float, default=20.0)
    ap.add_argument("--boundary_width_anchor_coef", type=float, default=20.0)
    ap.add_argument("--width_l2_coef", type=float, default=1e-4)
    ap.add_argument("--init_halfwidth", type=float, default=0.05)
    ap.add_argument("--gamma_risk", type=float, default=5.0)
    ap.add_argument("--step_reward_coef", type=float, default=1.0)
    ap.add_argument("--terminal_penalty_coef", type=float, default=1.0)
    ap.add_argument("--diff_reward_vs_center", action="store_true")
    ap.add_argument("--diff_reward_scale", type=float, default=10.0)
    ap.add_argument("--qvi_width_floor", type=float, default=1e-4)
    ap.add_argument("--width_mode", type=str, default="qvi_scale", choices=["qvi_scale", "free"])

    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args()

    cfg = BandStage2Config(
        mode=args.mode,
        T_years=args.T,
        dt=args.dt,
        z=args.z,
        x0=args.x0,
        p0=args.p0,
        tcost=args.tcost,
        rebalance_steps=args.rebalance_steps,
        hidden=args.hidden,
        lr=args.lr,
        iters=args.iters,
        episodes_per_iter=args.episodes_per_iter,
        baseline_beta=args.baseline_beta,
        entropy_coef=args.entropy_coef,
        reward_gamma=args.reward_gamma,
        gae_lambda=args.gae_lambda,
        critic_hidden=args.critic_hidden,
        critic_lr=args.critic_lr,
        critic_coef=args.critic_coef,
        center_unfreeze_at=args.center_unfreeze_at,
        center_ramp_len=args.center_ramp_len,
        center_anchor_coef=args.center_anchor_coef,
        boundary_warm_start_until=args.boundary_warm_start_until,
        boundary_center_anchor_coef=args.boundary_center_anchor_coef,
        boundary_width_anchor_coef=args.boundary_width_anchor_coef,
        width_l2_coef=args.width_l2_coef,
        init_halfwidth=args.init_halfwidth,
        gamma_risk=args.gamma_risk,
        step_reward_coef=args.step_reward_coef,
        terminal_penalty_coef=args.terminal_penalty_coef,
        diff_reward_vs_center=args.diff_reward_vs_center,
        diff_reward_scale=args.diff_reward_scale,
        qvi_width_floor=args.qvi_width_floor,
        width_mode=args.width_mode,
        device=args.device,
    )
    train_stage2(
        stage1_run_dir=Path(args.stage1_run_dir),
        stage1_checkpoint=Path(args.stage1_checkpoint),
        outdir=Path(args.outdir),
        cfg=cfg,
        seed=args.seed,
    )

if __name__ == "__main__":
    main()