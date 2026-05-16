# stage2_boundary_rl.py
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .eval_compare import build_agent_from_checkpoint, generate_test_path, load_run_config
from .filtering import FilterParams
from .stage2_direct_boundary import _load_true_and_filter_params
from .utils import set_seed
import matplotlib.pyplot as plt


# =========================
# Helpers
# =========================

def _obs_vec(tau: float, x: float, p: float, w_cur: np.ndarray, center_w: np.ndarray) -> np.ndarray:
    diff = w_cur - center_w
    return np.concatenate([
        np.array([tau, x, p], dtype=np.float32),
        np.asarray(w_cur, dtype=np.float32),
        np.asarray(center_w, dtype=np.float32),
        np.asarray(diff, dtype=np.float32),
    ]).astype(np.float32)


def _project_to_band(w_cur: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(w_cur, lower), upper)


def _apply_leverage_cap_to_weights(w: np.ndarray, lev_cap: float | None) -> np.ndarray:
    if lev_cap is None:
        return w
    gross = float(np.sum(np.abs(w)))
    if gross <= lev_cap or gross <= 1e-12:
        return w
    return w * (lev_cap / gross)


def _qvi_base_width(
    center_w: np.ndarray,
    Sigma: np.ndarray,
    kappa: float,
    gamma_risk: float,
    width_floor: float = 1e-4,
) -> np.ndarray:
    # Minimal diagonal-width proxy
    diag = np.clip(np.diag(Sigma), 1e-10, None)
    # Very simple width proxy; replace if you already have a preferred one
    base = np.power(np.maximum(kappa / (gamma_risk * diag), 1e-12), 1.0 / 3.0)
    return np.maximum(base, width_floor).astype(np.float32)


def _discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    out = np.zeros_like(x, dtype=np.float32)
    acc = 0.0
    for t in reversed(range(len(x))):
        acc = float(x[t]) + gamma * acc
        out[t] = acc
    return out


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    values: length T+1
    rewards,dones: length T
    """
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        nonterminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae = delta + gamma * lam * nonterminal * gae
        adv[t] = gae
    ret = adv + values[:-1]
    return adv, ret


# =========================
# Networks
# =========================

class BoundaryActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), -1.0))

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(obs)
        mu = self.mu(h)
        log_std = torch.clamp(self.log_std, min=-4.0, max=1.0)
        return mu, log_std

    def dist(self, obs: torch.Tensor) -> Normal:
        mu, log_std = self(obs)
        std = torch.exp(log_std)
        return Normal(mu, std)


class BoundaryCritic(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs).squeeze(-1)


# =========================
# Config
# =========================

@dataclass
class TrainBoundaryRLConfig:
    T_years: float = 1.0
    dt: float = 1.0 / 252.0
    z: float = 1.2
    x0: float = 1.0
    p0: float = 0.5
    tcost: float = 0.002

    hidden: int = 64
    episodes_per_iter: int = 32
    iters: int = 1000

    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    weight_decay: float = 1e-6

    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_epochs: int = 5
    minibatch_size: int = 2048
    entropy_coef: float = 1e-3
    value_coef: float = 0.5
    max_grad_norm: float = 1.0

    utility_kind: str = "log"   # log | crra
    utility_gamma: float = 2.0
    utility_scale: float = 1.0
    utility_shift: float = 0.0

    turnover_coef: float = 5e-4
    gross_lev_coef: float = 1e-5
    lev_cap: float | None = None
    lev_penalty_power: float = 2.0

    gap_mode: str = "direct"    # direct | qvi_exp
    gap_limit: float = 2.0      # for direct mode, max gap via sigmoid
    qvi_width_floor: float = 1e-4
    gamma_risk: float = 5.0
    correction_clip: float = 1.5

    precompute_center_path: bool = True
    plot_diagnostic_path: bool = False
    diagnostic_path_id: int = 0

    val_every: int = 25
    val_n_paths: int = 64
    device: str = "cpu"


# =========================
# Utility
# =========================

def _terminal_utility(wT: torch.Tensor, kind: str, gamma: float, scale: float, shift: float) -> torch.Tensor:
    x = torch.clamp(wT + shift, min=1e-8)
    if kind == "log":
        u = torch.log(x)
    elif kind == "crra":
        g = gamma
        if abs(g - 1.0) < 1e-8:
            u = torch.log(x)
        else:
            u = (torch.pow(x, 1.0 - g) - 1.0) / (1.0 - g)
    else:
        raise ValueError(f"Unknown utility_kind: {kind}")
    return scale * u

def _terminal_utility_np(wT: float, kind: str, gamma: float, scale: float, shift: float) -> float:
    x = max(wT + shift, 1e-8)
    if kind == "log":
        u = math.log(x)
    elif kind == "crra":
        g = gamma
        if abs(g - 1.0) < 1e-8:
            u = math.log(x)
        else:
            u = (x ** (1.0 - g) - 1.0) / (1.0 - g)
    else:
        raise ValueError(f"Unknown utility_kind: {kind}")
    return float(scale * u)

def _utility_increment(x_now: float, x_next: float, cfg: TrainBoundaryRLConfig) -> float:
    return _terminal_utility_np(
        x_next, cfg.utility_kind, cfg.utility_gamma, cfg.utility_scale, cfg.utility_shift
    ) - _terminal_utility_np(
        x_now, cfg.utility_kind, cfg.utility_gamma, cfg.utility_scale, cfg.utility_shift
    )

def _excess_leverage_penalty(w: np.ndarray, lev_cap: float | None, coef: float, power: float = 2.0) -> float:
    if lev_cap is None or coef <= 0.0:
        return 0.0
    gross = float(np.sum(np.abs(w)))
    excess = max(gross - float(lev_cap), 0.0)
    return float(coef * (excess ** power))

# =========================
# Action -> boundary
# =========================

def _action_to_gaps(
    action_t: torch.Tensor,
    center_w_np: np.ndarray,
    filt_params: FilterParams,
    cfg: TrainBoundaryRLConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    action_t shape: [n_assets*2]
    returns lower_gap_t, upper_gap_t
    """
    n_assets = len(center_w_np)
    raw_lower = action_t[:n_assets]
    raw_upper = action_t[n_assets:]

    if cfg.gap_mode == "direct":
        gap_limit = float(cfg.gap_limit)
        lower_gap_t = gap_limit * torch.sigmoid(raw_lower)
        upper_gap_t = gap_limit * torch.sigmoid(raw_upper)
        return lower_gap_t, upper_gap_t

    if cfg.gap_mode == "qvi_exp":
        qvi_gap_np = _qvi_base_width(
            center_w=center_w_np,
            Sigma=filt_params.Sigma,
            kappa=cfg.tcost,
            gamma_risk=cfg.gamma_risk,
            width_floor=cfg.qvi_width_floor,
        )
        qvi_gap_t = torch.as_tensor(qvi_gap_np, dtype=dtype, device=device)
        clip_c = float(cfg.correction_clip)
        lower_corr = torch.clamp(raw_lower, min=-clip_c, max=clip_c)
        upper_corr = torch.clamp(raw_upper, min=-clip_c, max=clip_c)
        lower_gap_t = qvi_gap_t * torch.exp(lower_corr)
        upper_gap_t = qvi_gap_t * torch.exp(upper_corr)
        return lower_gap_t, upper_gap_t

    raise ValueError(f"Unknown gap_mode: {cfg.gap_mode}")


# =========================
# Rollout
# =========================

def rollout_one_path(
    *,
    path: Dict[str, np.ndarray],
    center_agent,
    actor: BoundaryActor,
    critic: BoundaryCritic,
    filt_params: FilterParams,
    cfg: TrainBoundaryRLConfig,
    deterministic: bool = False,
    center_w_precomputed: np.ndarray | None = None,
) -> Dict[str, np.ndarray]:
    """
    One full episode rollout.
    """
    device = torch.device(cfg.device)
    dtype = torch.float32

    n = path["ret"].shape[0]
    belief = path["belief"]

    obs_list = []
    act_list = []
    logp_list = []
    val_list = []
    rew_list = []
    done_list = []

    wealth = np.empty(n + 1, dtype=np.float32)
    wealth[0] = cfg.x0
    current_u = np.zeros(2, dtype=np.float32)

    gross_lev = np.empty(n, dtype=np.float32)
    turnover = np.empty(n, dtype=np.float32)
    center_w_hist = np.empty((n, 2), dtype=np.float32)
    actual_w_hist = np.empty((n, 2), dtype=np.float32)
    lower_hist = np.empty((n, 2), dtype=np.float32)
    upper_hist = np.empty((n, 2), dtype=np.float32)

    actor.eval()
    critic.eval()

    for k in range(n):
        xk = float(wealth[k])
        tk = float(path["t"][k])
        pk = float(belief[k])

        denom = max(abs(xk), 1e-12)
        if center_w_precomputed is not None:
            center_w = np.asarray(center_w_precomputed[k], dtype=np.float32)
            center_u = center_w * denom
        else:
            center_u = np.asarray(center_agent.policy_mean(tk, xk, pk), dtype=np.float32)
            center_w = center_u / denom
        w_cur = current_u / denom

        tau = tk / max(cfg.T_years, 1e-12)
        obs_np = _obs_vec(tau, xk, pk, w_cur, center_w)
        obs_t = torch.as_tensor(obs_np, dtype=dtype, device=device).unsqueeze(0)

        dist = actor.dist(obs_t)
        value_t = critic(obs_t)

        if deterministic:
            action_t = dist.mean.squeeze(0)
        else:
            action_t = dist.rsample().squeeze(0)

        logp_t = dist.log_prob(action_t.unsqueeze(0)).sum(dim=-1).squeeze(0)

        lower_gap_t, upper_gap_t = _action_to_gaps(
            action_t=action_t,
            center_w_np=center_w,
            filt_params=filt_params,
            cfg=cfg,
            device=device,
            dtype=dtype,
        )

        lower = center_w - lower_gap_t.detach().cpu().numpy()
        upper = center_w + upper_gap_t.detach().cpu().numpy()

        w_tgt = _project_to_band(w_cur, lower, upper)
        w_tgt = _apply_leverage_cap_to_weights(w_tgt, cfg.lev_cap)

        new_u = w_tgt * xk
        trade = float(np.sum(np.abs(new_u - current_u)))
        tc = cfg.tcost * trade

        disc_ret = path["ret"][k]
        x_next = xk + float(np.dot(new_u, disc_ret)) - tc
        wealth[k + 1] = x_next

        # per-step reward: utility increment + penalties
        reward = _utility_increment(xk, x_next, cfg)
        reward -= cfg.turnover_coef * (trade / max(abs(xk), 1e-12))
        reward -= _excess_leverage_penalty(
            w=w_tgt,
            lev_cap=cfg.lev_cap,
            coef=cfg.gross_lev_coef,
            power=cfg.lev_penalty_power,
        )

        obs_list.append(obs_np)
        act_list.append(action_t.detach().cpu().numpy())
        logp_list.append(float(logp_t.detach().cpu().item()))
        val_list.append(float(value_t.detach().cpu().item()))
        rew_list.append(float(reward))
        done_list.append(1.0 if (k == n - 1) else 0.0)

        gross_lev[k] = float(np.sum(np.abs(w_tgt)))
        turnover[k] = trade / max(abs(xk), 1e-12)
        center_w_hist[k] = center_w
        actual_w_hist[k] = w_tgt
        lower_hist[k] = lower
        upper_hist[k] = upper

        current_u = new_u.astype(np.float32)

        # per-step reward: utility increment + penalties
        reward = _utility_increment(xk, x_next, cfg)
        reward -= cfg.turnover_coef * (trade / max(abs(xk), 1e-12))
        reward -= _excess_leverage_penalty(
            w=w_tgt,
            lev_cap=cfg.lev_cap,
            coef=cfg.gross_lev_coef,
            power=cfg.lev_penalty_power,
        )

    terminal_u = _terminal_utility_np(
        float(wealth[-1]),
        cfg.utility_kind,
        cfg.utility_gamma,
        cfg.utility_scale,
        cfg.utility_shift,
    )
    # bootstrap last value = 0 at terminal
    val_list.append(0.0)

    obs_arr = np.asarray(obs_list, dtype=np.float32)
    act_arr = np.asarray(act_list, dtype=np.float32)
    logp_arr = np.asarray(logp_list, dtype=np.float32)
    val_arr = np.asarray(val_list, dtype=np.float32)
    rew_arr = np.asarray(rew_list, dtype=np.float32)
    done_arr = np.asarray(done_list, dtype=np.float32)

    adv_arr, ret_arr = _compute_gae(
        rewards=rew_arr,
        values=val_arr,
        dones=done_arr,
        gamma=cfg.gamma,
        lam=cfg.gae_lambda,
    )

    return {
        "obs": obs_arr,
        "act": act_arr,
        "logp": logp_arr,
        "adv": adv_arr,
        "ret": ret_arr,
        "wealth": wealth,
        "gross_lev": gross_lev,
        "turnover": turnover,
        "belief": belief[:n],
        "center_w": center_w_hist,
        "actual_w": actual_w_hist,
        "lower": lower_hist,
        "upper": upper_hist,
        "terminal_wealth": float(wealth[-1]),
        "terminal_utility": terminal_u,
    }

def _precompute_center_path(
    *,
    path: Dict[str, np.ndarray],
    center_agent,
    T_years: float,
) -> np.ndarray:
    n = path["ret"].shape[0]
    belief = path["belief"]
    center_w = np.empty((n, 2), dtype=np.float32)
    for k in range(n):
        tk = float(path["t"][k])
        pk = float(belief[k])
        # exploit approximate homogeneity by evaluating at x=1
        u = np.asarray(center_agent.policy_mean(tk, 1.0, pk), dtype=np.float32)
        center_w[k] = u
    return center_w

def make_train_path(
    *,
    true_params,
    filt_params: FilterParams,
    seed_i: int,
    T_years: float,
    dt: float,
    p0: float,
) -> Dict[str, np.ndarray]:
    path = generate_test_path(
        true_params,
        T_years=T_years,
        dt=dt,
        seed=seed_i,
    )
    from .eval_compare import compute_belief_path  # avoid circular issues at import time
    belief = compute_belief_path(path["logret"], filt_params=filt_params, dt=dt, p0=p0)
    path["belief"] = belief
    return path


# =========================
# Validation
# =========================

@torch.no_grad()
def evaluate_policy(
    *,
    actor: BoundaryActor,
    critic: BoundaryCritic,
    center_agent,
    true_params,
    filt_params: FilterParams,
    cfg: TrainBoundaryRLConfig,
    seed0: int,
    n_paths: int,
) -> Dict[str, float]:
    terminals = []
    utils = []
    for i in range(n_paths):
        path = make_train_path(
            true_params=true_params,
            filt_params=filt_params,
            seed_i=seed0 + 100000 + i,
            T_years=cfg.T_years,
            dt=cfg.dt,
            p0=cfg.p0,
        )
        center_w_precomputed = _precompute_center_path(
            path=path,
            center_agent=center_agent,
            T_years=cfg.T_years,
        ) if cfg.precompute_center_path else None
        out = rollout_one_path(
            path=path,
            center_agent=center_agent,
            actor=actor,
            critic=critic,
            filt_params=filt_params,
            cfg=cfg,
            deterministic=True,
            center_w_precomputed=center_w_precomputed,
        )
        terminals.append(out["terminal_wealth"])
        utils.append(out["terminal_utility"])

    return {
        "mean_terminal": float(np.mean(terminals)),
        "std_terminal": float(np.std(terminals)),
        "mean_utility": float(np.mean(utils)),
    }

def _plot_training_curves(logs: List[Dict], outdir: Path):
    if not logs:
        return
    it = [r["iter"] for r in logs]

    fig = plt.figure(figsize=(10, 6))
    plt.plot(it, [r.get("train_mean_terminal", np.nan) for r in logs], label="train_mean_terminal")
    plt.plot(it, [r.get("val_mean_terminal", np.nan) for r in logs], label="val_mean_terminal")
    plt.xlabel("iteration")
    plt.ylabel("terminal wealth")
    plt.title("Stage2 boundary-RL terminal wealth")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "learning_terminal_wealth.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(it, [r.get("actor_loss", np.nan) for r in logs], label="actor_loss")
    plt.plot(it, [r.get("critic_loss", np.nan) for r in logs], label="critic_loss")
    plt.xlabel("iteration")
    plt.ylabel("loss")
    plt.title("Stage2 boundary-RL losses")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "learning_losses.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(it, [r.get("train_mean_utility", np.nan) for r in logs], label="train_mean_utility")
    plt.plot(it, [r.get("val_mean_utility", np.nan) for r in logs], label="val_mean_utility")
    plt.xlabel("iteration")
    plt.ylabel("utility")
    plt.title("Stage2 boundary-RL utility")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "learning_utility.png", dpi=200)
    plt.close(fig)

def _plot_diagnostic_path(
    *,
    outdir: Path,
    sim: Dict[str, np.ndarray],
    path: Dict[str, np.ndarray],
):
    t = np.asarray(path["t"][:path["ret"].shape[0]], dtype=float)
    regime = np.asarray(path["I"][:path["ret"].shape[0]], dtype=int) if "I" in path else np.full_like(t, -1, dtype=int)
    belief = np.asarray(sim["belief"], dtype=float)
    center_w = np.asarray(sim["center_w"], dtype=float)
    actual_w = np.asarray(sim["actual_w"], dtype=float)
    lower = np.asarray(sim["lower"], dtype=float)
    upper = np.asarray(sim["upper"], dtype=float)

    for j in range(center_w.shape[1]):
        fig, ax1 = plt.subplots(figsize=(11, 5))

        start = 0
        n = len(t)
        for k in range(1, n + 1):
            if k == n or regime[k] != regime[start]:
                if regime[start] in (0, 1):
                    color = "green" if regime[start] == 1 else "red"
                    ax1.axvspan(t[start], t[k - 1], alpha=0.08, color=color)
                start = k

        ax1.plot(t, center_w[:, j], label=f"center_w[{j}]")
        ax1.plot(t, actual_w[:, j], label=f"actual_w[{j}]")
        ax1.plot(t, lower[:, j], linestyle="--", label=f"lower[{j}]")
        ax1.plot(t, upper[:, j], linestyle="--", label=f"upper[{j}]")
        ax1.set_xlabel("time (years)")
        ax1.set_ylabel(f"asset {j+1} weight")

        ax2 = ax1.twinx()
        ax2.plot(t, belief, linestyle=":", linewidth=1.8, label="belief p_t")
        ax2.set_ylabel("belief / regime")
        ax2.set_ylim(-0.05, 1.05)

        l1, lb1 = ax1.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax1.legend(l1 + l2, lb1 + lb2, loc="best")
        ax1.set_title(f"boundary_rl_diagnostic: asset {j+1}")

        fig.tight_layout()
        fig.savefig(outdir / f"diagnostic_path_asset{j+1}.png", dpi=200)
        plt.close(fig)


# =========================
# PPO update
# =========================

def ppo_update(
    *,
    actor: BoundaryActor,
    critic: BoundaryCritic,
    actor_opt,
    critic_opt,
    batch: Dict[str, np.ndarray],
    cfg: TrainBoundaryRLConfig,
) -> Dict[str, float]:
    device = torch.device(cfg.device)

    obs = torch.as_tensor(batch["obs"], dtype=torch.float32, device=device)
    act = torch.as_tensor(batch["act"], dtype=torch.float32, device=device)
    old_logp = torch.as_tensor(batch["logp"], dtype=torch.float32, device=device)
    adv = torch.as_tensor(batch["adv"], dtype=torch.float32, device=device)
    ret = torch.as_tensor(batch["ret"], dtype=torch.float32, device=device)

    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

    N = obs.shape[0]
    idx_all = np.arange(N)

    actor.train()
    critic.train()

    last_stats = {}

    for _ in range(cfg.ppo_epochs):
        np.random.shuffle(idx_all)
        for start in range(0, N, cfg.minibatch_size):
            idx = idx_all[start:start + cfg.minibatch_size]
            mb_obs = obs[idx]
            mb_act = act[idx]
            mb_old_logp = old_logp[idx]
            mb_adv = adv[idx]
            mb_ret = ret[idx]

            dist = actor.dist(mb_obs)
            new_logp = dist.log_prob(mb_act).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1).mean()

            ratio = torch.exp(new_logp - mb_old_logp)
            surr1 = ratio * mb_adv
            surr2 = torch.clamp(ratio, 1.0 - cfg.ppo_clip, 1.0 + cfg.ppo_clip) * mb_adv
            actor_loss = -torch.min(surr1, surr2).mean() - cfg.entropy_coef * entropy

            value = critic(mb_obs)
            critic_loss = F.mse_loss(value, mb_ret)

            actor_opt.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
            actor_opt.step()

            critic_opt.zero_grad()
            (cfg.value_coef * critic_loss).backward()
            nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
            critic_opt.step()

            last_stats = {
                "actor_loss": float(actor_loss.detach().cpu().item()),
                "critic_loss": float(critic_loss.detach().cpu().item()),
                "entropy": float(entropy.detach().cpu().item()),
            }

    return last_stats


# =========================
# Main train
# =========================

def train_boundary_rl(
    *,
    stage1_run_dir: Path,
    stage1_checkpoint: Path,
    outdir: Path,
    cfg: TrainBoundaryRLConfig,
    seed: int,
    filter_mode: str = "true_params",
):
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)

    true_params, filt_params = _load_true_and_filter_params(stage1_run_dir, filter_mode)

    center_agent = build_agent_from_checkpoint(
        ckpt_path=stage1_checkpoint,
        run_dir=stage1_run_dir,
        T_years=cfg.T_years,
        dt=cfg.dt,
        a_max=1.0,
        z=cfg.z,
        r=float(true_params.r),
        device=cfg.device,
    )

    obs_dim = 3 + 2 + 2 + 2
    act_dim = 4  # lower/upper gap for 2 assets

    actor = BoundaryActor(obs_dim=obs_dim, act_dim=act_dim, hidden=cfg.hidden).to(cfg.device)
    critic = BoundaryCritic(obs_dim=obs_dim, hidden=cfg.hidden).to(cfg.device)

    actor_opt = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.weight_decay)

    logs = []

    best_val = -1e18
    best_path = outdir / "best_checkpoint.pt"

    for it in range(cfg.iters):
        rollouts = []
        for b in range(cfg.episodes_per_iter):
            path = make_train_path(
                true_params=true_params,
                filt_params=filt_params,
                seed_i=seed + 10000 * it + b,
                T_years=cfg.T_years,
                dt=cfg.dt,
                p0=cfg.p0,
            )
            center_w_precomputed = _precompute_center_path(
                path=path,
                center_agent=center_agent,
                T_years=cfg.T_years,
            ) if cfg.precompute_center_path else None
            ro = rollout_one_path(
                path=path,
                center_agent=center_agent,
                actor=actor,
                critic=critic,
                filt_params=filt_params,
                cfg=cfg,
                deterministic=False,
                center_w_precomputed=center_w_precomputed,
            )
            rollouts.append(ro)

        batch = {
            "obs": np.concatenate([r["obs"] for r in rollouts], axis=0),
            "act": np.concatenate([r["act"] for r in rollouts], axis=0),
            "logp": np.concatenate([r["logp"] for r in rollouts], axis=0),
            "adv": np.concatenate([r["adv"] for r in rollouts], axis=0),
            "ret": np.concatenate([r["ret"] for r in rollouts], axis=0),
        }

        stats = ppo_update(
            actor=actor,
            critic=critic,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            batch=batch,
            cfg=cfg,
        )

        train_mean_terminal = float(np.mean([r["terminal_wealth"] for r in rollouts]))
        train_mean_utility = float(np.mean([r["terminal_utility"] for r in rollouts]))

        rec = {
            "iter": it,
            "train_mean_terminal": train_mean_terminal,
            "train_mean_utility": train_mean_utility,
            **stats,
        }

        if (it + 1) % cfg.val_every == 0 or it == 0:
            val = evaluate_policy(
                actor=actor,
                critic=critic,
                center_agent=center_agent,
                true_params=true_params,
                filt_params=filt_params,
                cfg=cfg,
                seed0=seed + 999999,
                n_paths=cfg.val_n_paths,
            )
            rec.update({
                "val_mean_terminal": val["mean_terminal"],
                "val_std_terminal": val["std_terminal"],
                "val_mean_utility": val["mean_utility"],
            })

            if val["mean_utility"] > best_val:
                best_val = val["mean_utility"]
                torch.save(
                    {
                        "actor_state_dict": actor.state_dict(),
                        "critic_state_dict": critic.state_dict(),
                        "cfg": asdict(cfg),
                        "seed": seed,
                        "filter_mode": filter_mode,
                    },
                    best_path,
                )
            if cfg.plot_diagnostic_path and (it + 1) % cfg.val_every == 0:
                diag_path = make_train_path(
                    true_params=true_params,
                    filt_params=filt_params,
                    seed_i=seed + 777777 + cfg.diagnostic_path_id,
                    T_years=cfg.T_years,
                    dt=cfg.dt,
                    p0=cfg.p0,
                )
                diag_center = _precompute_center_path(
                    path=diag_path,
                    center_agent=center_agent,
                    T_years=cfg.T_years,
                ) if cfg.precompute_center_path else None
                diag_sim = rollout_one_path(
                    path=diag_path,
                    center_agent=center_agent,
                    actor=actor,
                    critic=critic,
                    filt_params=filt_params,
                    cfg=cfg,
                    deterministic=True,
                    center_w_precomputed=diag_center,
                )
                _plot_diagnostic_path(outdir=outdir, sim=diag_sim, path=diag_path)

        logs.append(rec)

        if (it + 1) % 10 == 0:
            print(
                f"[it {it+1:4d}] "
                f"train_terminal={train_mean_terminal:.4f} "
                f"train_utility={train_mean_utility:.4f} "
                f"actor_loss={rec.get('actor_loss', float('nan')):.4f} "
                f"critic_loss={rec.get('critic_loss', float('nan')):.4f}"
            )

    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "cfg": asdict(cfg),
            "seed": seed,
            "filter_mode": filter_mode,
        },
        outdir / "last_checkpoint.pt",
    )

    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage1_run_dir": str(stage1_run_dir.resolve()),
                "stage1_checkpoint": str(stage1_checkpoint.resolve()),
                "seed": seed,
                "filter_mode": filter_mode,
                "true_params": {
                    "mu1": np.asarray(true_params.mu1, dtype=float).tolist(),
                    "mu2": np.asarray(true_params.mu2, dtype=float).tolist(),
                    "Sigma": np.asarray(true_params.Sigma, dtype=float).tolist(),
                    "lam1": float(true_params.lam1),
                    "lam2": float(true_params.lam2),
                    "r": float(true_params.r),
                },
                "filter_params": {
                    "mu1": np.asarray(filt_params.mu1, dtype=float).tolist(),
                    "mu2": np.asarray(filt_params.mu2, dtype=float).tolist(),
                    "Sigma": np.asarray(filt_params.Sigma, dtype=float).tolist(),
                    "lam1": float(filt_params.lam1),
                    "lam2": float(filt_params.lam2),
                    "r": float(filt_params.r),
                },
                **asdict(cfg),
            },
            f,
            indent=2,
        )

    try:
        import pandas as pd
        pd.DataFrame(logs).to_csv(outdir / "metrics.csv", index=False)
    except Exception:
        pass
    _plot_training_curves(logs, outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage1_run_dir", type=str, required=True)
    ap.add_argument("--stage1_checkpoint", type=str, required=True)
    ap.add_argument("--outdir", type=str, required=True)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--filter_mode", type=str, choices=["true_params", "estimated_params"], default="true_params")

    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--episodes_per_iter", type=int, default=32)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=1.0 / 252.0)
    ap.add_argument("--z", type=float, default=1.2)
    ap.add_argument("--x0", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.5)
    ap.add_argument("--tcost", type=float, default=0.002)

    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--actor_lr", type=float, default=3e-4)
    ap.add_argument("--critic_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)

    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--ppo_clip", type=float, default=0.2)
    ap.add_argument("--ppo_epochs", type=int, default=5)
    ap.add_argument("--minibatch_size", type=int, default=2048)
    ap.add_argument("--entropy_coef", type=float, default=1e-3)
    ap.add_argument("--value_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--utility_kind", type=str, choices=["log", "crra"], default="log")
    ap.add_argument("--utility_gamma", type=float, default=2.0)
    ap.add_argument("--utility_scale", type=float, default=1.0)
    ap.add_argument("--utility_shift", type=float, default=0.0)

    ap.add_argument("--turnover_coef", type=float, default=5e-4)
    ap.add_argument("--gross_lev_coef", type=float, default=1e-5)
    ap.add_argument("--lev_cap", type=float, default=None)
    ap.add_argument("--lev_penalty_power", type=float, default=2.0)

    ap.add_argument("--gap_mode", type=str, choices=["direct", "qvi_exp"], default="direct")
    ap.add_argument("--gap_limit", type=float, default=2.0)
    ap.add_argument("--qvi_width_floor", type=float, default=1e-4)
    ap.add_argument("--gamma_risk", type=float, default=5.0)
    ap.add_argument("--correction_clip", type=float, default=1.5)

    ap.add_argument("--val_every", type=int, default=25)
    ap.add_argument("--val_n_paths", type=int, default=64)
    ap.add_argument("--precompute_center_path", action="store_true")
    ap.add_argument("--plot_diagnostic_path", action="store_true")
    ap.add_argument("--diagnostic_path_id", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")

    args = ap.parse_args()

    cfg = TrainBoundaryRLConfig(
        T_years=args.T,
        dt=args.dt,
        z=args.z,
        x0=args.x0,
        p0=args.p0,
        tcost=args.tcost,
        hidden=args.hidden,
        episodes_per_iter=args.episodes_per_iter,
        iters=args.iters,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        weight_decay=args.weight_decay,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ppo_clip=args.ppo_clip,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        utility_kind=args.utility_kind,
        utility_gamma=args.utility_gamma,
        utility_scale=args.utility_scale,
        utility_shift=args.utility_shift,
        turnover_coef=args.turnover_coef,
        gross_lev_coef=args.gross_lev_coef,
        lev_cap=args.lev_cap,
        lev_penalty_power=args.lev_penalty_power,
        gap_mode=args.gap_mode,
        gap_limit=args.gap_limit,
        qvi_width_floor=args.qvi_width_floor,
        gamma_risk=args.gamma_risk,
        correction_clip=args.correction_clip,
        precompute_center_path=args.precompute_center_path,
        plot_diagnostic_path=args.plot_diagnostic_path,
        diagnostic_path_id=args.diagnostic_path_id,
        val_every=args.val_every,
        val_n_paths=args.val_n_paths,
        device=args.device,
    )

    train_boundary_rl(
        stage1_run_dir=Path(args.stage1_run_dir),
        stage1_checkpoint=Path(args.stage1_checkpoint),
        outdir=Path(args.outdir),
        cfg=cfg,
        seed=args.seed,
        filter_mode=args.filter_mode,
    )


if __name__ == "__main__":
    main()