from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from .eval_compare import (
    build_agent_from_checkpoint,
    load_run_config,
    compute_belief_path,
    default_true_params,
    generate_test_path,
)
from .filtering import FilterParams
from .utils import set_seed


# =========================
# Config / parameter loading
# =========================

@dataclass
class TrainStage2RLConfig:
    T_years: float = 1.0
    dt: float = 1 / 252
    z: float = 1.2
    x0: float = 1.0
    p0: float = 0.5
    tcost: float = 0.002

    hidden: int = 64
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    weight_decay: float = 1e-6

    iters: int = 800
    episodes_per_iter: int = 64
    gamma: float = 0.995
    gae_lambda: float = 0.95
    ppo_epochs: int = 10
    clip_eps: float = 0.2
    entropy_coef: float = 1e-4
    value_coef: float = 0.5
    max_grad_norm: float = 1.0

    gamma_risk: float = 5.0
    qvi_width_floor: float = 1e-4

    # reward shaping
    reward_scale: float = 1.0
    use_log_wealth_increment: bool = True
    terminal_utility_coef: float = 1.0
    utility_kind: str = "log"  # log | sqrt | power
    utility_gamma: float = 2.0

    turnover_coef: float = 5e-4
    prior_dev_coef: float = 1e-3
    gross_lev_coef: float = 1e-4

    # action parameterization
    correction_scale: float = 0.25
    log_scale_clip: float = 2.0

    val_every: int = 25
    val_n_paths: int = 64

    precompute_center_path: bool = True
    num_workers: int = 0
    device: str = "cpu"
    dtype: torch.dtype = torch.float64


def _params_from_dict(d: Dict, fallback=None):
    if fallback is None:
        fallback = default_true_params()
    src = d or {}
    return dict(
        mu1=np.asarray(src.get("mu1", fallback.mu1), dtype=float),
        mu2=np.asarray(src.get("mu2", fallback.mu2), dtype=float),
        Sigma=np.asarray(src.get("Sigma", fallback.Sigma), dtype=float),
        lam1=float(src.get("lam1", fallback.lam1)),
        lam2=float(src.get("lam2", fallback.lam2)),
        r=float(src.get("r", fallback.r)),
    )


def _load_true_and_filter_params(stage1_run_dir: Path, filter_mode: str):
    run_cfg = load_run_config(stage1_run_dir)
    fallback = default_true_params()

    true_src = _params_from_dict(run_cfg.get("true_params", {}), fallback=fallback)
    true_params = default_true_params()
    true_params.mu1 = true_src["mu1"]
    true_params.mu2 = true_src["mu2"]
    true_params.Sigma = true_src["Sigma"]
    true_params.lam1 = true_src["lam1"]
    true_params.lam2 = true_src["lam2"]
    true_params.r = true_src["r"]

    filt_key = "estimated_params" if filter_mode == "estimated_params" else "true_params"
    filt_src = _params_from_dict(run_cfg.get(filt_key, {}), fallback=true_params)
    filt_params = FilterParams(**filt_src)
    return true_params, filt_params


# =========================
# Common helpers
# =========================

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


def _utility_np(w: float, kind: str, gamma: float, eps: float = 1e-10) -> float:
    x = max(float(w), eps)
    if kind == "log":
        return float(np.log(x))
    if kind == "sqrt":
        return float(np.sqrt(x))
    if kind == "power":
        g = float(gamma)
        if abs(g - 1.0) < 1e-12:
            return float(np.log(x))
        return float((x ** (1.0 - g)) / (1.0 - g))
    raise ValueError(f"Unknown utility_kind: {kind}")


def _discounted_risky_return(logret_k: np.ndarray, r: float, dt: float) -> np.ndarray:
    return np.exp(logret_k - r * dt) - 1.0


def _make_sample(
    seed_i: int,
    T_years: float,
    dt: float,
    p0: float,
    filt_params: FilterParams,
    stage1_run_dir: Path,
    stage1_checkpoint: Path,
    z: float,
    device: str,
    precompute_center_path: bool,
):
    true_params = default_true_params()
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
            ckpt_path=stage1_checkpoint,
            run_dir=stage1_run_dir,
            T_years=T_years,
            dt=dt,
            a_max=1.0,
            z=z,
            r=true_params.r,
            device=device,
        )
        center_w_path = []
        for k in range(path["ret"].shape[0]):
            tk = path["t"][k]
            pk = belief[k]
            u_star = np.asarray(center_agent.policy_mean(tk, 1.0, pk), dtype=float)
            center_w_path.append(u_star.copy())
        center_w_path = np.asarray(center_w_path, dtype=float)
    return {"path": path, "belief": belief, "center_w_path": center_w_path}


# =========================
# Actor / Critic
# =========================

class Stage2RLActor(nn.Module):
    """
    Action:
        a = [log_scale_lower_0, log_scale_lower_1, log_scale_upper_0, log_scale_upper_1]

    We later set:
        lower_gap = base_width * exp(clipped(log_scale_lower))
        upper_gap = base_width * exp(clipped(log_scale_upper))
    """
    def __init__(self, obs_dim: int = 9, hidden: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.mu_head = nn.Linear(hidden, 4)
        self.log_std = nn.Parameter(torch.full((4,), -1.0, dtype=torch.float64))

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.backbone(obs)
        mu = self.mu_head(h)
        log_std = self.log_std.unsqueeze(0).expand_as(mu)
        return mu, log_std


class Stage2RLCritic(nn.Module):
    def __init__(self, obs_dim: int = 9, hidden: int = 64):
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


def _normal_log_prob(x: torch.Tensor, mu: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    var = torch.exp(2.0 * log_std)
    return (-0.5 * (((x - mu) ** 2) / var + 2.0 * log_std + np.log(2.0 * np.pi))).sum(dim=-1)


def _normal_entropy(log_std: torch.Tensor) -> torch.Tensor:
    return (0.5 + 0.5 * np.log(2.0 * np.pi) + log_std).sum(dim=-1)


# =========================
# Trajectory rollout
# =========================

def rollout_episode_rl(
    *,
    center_agent,
    actor: Stage2RLActor,
    critic: Stage2RLCritic,
    path: Dict[str, np.ndarray],
    belief: np.ndarray,
    center_w_path: Optional[np.ndarray],
    filt_params: FilterParams,
    cfg: TrainStage2RLConfig,
    deterministic: bool = False,
) -> Dict[str, np.ndarray]:
    n = path["ret"].shape[0]
    device = next(actor.parameters()).device
    dtype = next(actor.parameters()).dtype

    x = float(cfg.x0)
    current_u = np.zeros(2, dtype=float)

    obs_list = []
    action_list = []
    logp_list = []
    value_list = []
    reward_list = []
    done_list = []

    wealth_path = [x]
    turnover_path = []
    gross_lev_path = []
    mean_lower_scale = []
    mean_upper_scale = []

    for k in range(n):
        tk = float(path["t"][k])
        pk = float(belief[k])

        if center_w_path is not None:
            center_w = np.asarray(center_w_path[k], dtype=float)
        else:
            center_u = np.asarray(center_agent.policy_mean(tk, x, pk), dtype=float)
            center_w = center_u / max(abs(x), 1e-12)

        w_cur = current_u / max(abs(x), 1e-12)
        obs_np = _obs_vec(tk / max(cfg.T_years, 1e-12), x, pk, w_cur, center_w)
        obs_t = torch.as_tensor(obs_np, dtype=dtype, device=device).unsqueeze(0)

        with torch.no_grad():
            mu, log_std = actor(obs_t)
            value = critic(obs_t)

            if deterministic:
                action_t = mu
            else:
                eps = torch.randn_like(mu)
                action_t = mu + torch.exp(log_std) * eps

            logp_t = _normal_log_prob(action_t, mu, log_std)

        action_np = action_t.squeeze(0).detach().cpu().numpy()
        lower_log_scale = np.clip(action_np[:2], -cfg.log_scale_clip, cfg.log_scale_clip)
        upper_log_scale = np.clip(action_np[2:], -cfg.log_scale_clip, cfg.log_scale_clip)

        base_width = _qvi_base_width(
            center_w=center_w,
            Sigma=filt_params.Sigma,
            kappa=cfg.tcost,
            gamma_risk=cfg.gamma_risk,
            width_floor=cfg.qvi_width_floor,
        )

        lower_gap = base_width * np.exp(cfg.correction_scale * lower_log_scale)
        upper_gap = base_width * np.exp(cfg.correction_scale * upper_log_scale)

        lower = center_w - lower_gap
        upper = center_w + upper_gap
        w_tgt = np.minimum(np.maximum(w_cur, lower), upper)

        new_u = w_tgt * x
        tc = cfg.tcost * float(np.sum(np.abs(new_u - current_u)))
        turnover = float(np.sum(np.abs(new_u - current_u)) / max(abs(x), 1e-12))
        gross_lev = float(np.sum(np.abs(w_tgt)))

        disc_ret = _discounted_risky_return(path["logret"][k], filt_params.r, cfg.dt)
        x_next = x + float(np.dot(new_u, disc_ret)) - tc

        if cfg.use_log_wealth_increment:
            step_gain = np.log(max(x_next, 1e-10)) - np.log(max(x, 1e-10))
        else:
            step_gain = (x_next - x) / max(abs(x), 1e-12)

        prior_dev = float(np.mean(lower_log_scale ** 2) + np.mean(upper_log_scale ** 2))
        step_reward = (
            cfg.reward_scale * step_gain
            - cfg.turnover_coef * turnover
            - cfg.prior_dev_coef * prior_dev
            - cfg.gross_lev_coef * gross_lev
        )

        done = 1.0 if (k == n - 1) else 0.0
        if done > 0.5:
            step_reward += cfg.terminal_utility_coef * _utility_np(
                x_next,
                kind=cfg.utility_kind,
                gamma=cfg.utility_gamma,
            )

        obs_list.append(obs_np)
        action_list.append(action_np)
        logp_list.append(float(logp_t.item()))
        value_list.append(float(value.item()))
        reward_list.append(step_reward)
        done_list.append(done)

        wealth_path.append(x_next)
        turnover_path.append(turnover)
        gross_lev_path.append(gross_lev)
        mean_lower_scale.append(float(np.mean(np.exp(cfg.correction_scale * lower_log_scale))))
        mean_upper_scale.append(float(np.mean(np.exp(cfg.correction_scale * upper_log_scale))))

        current_u = new_u
        x = x_next

    # bootstrap value at final state = 0
    value_list.append(0.0)

    return {
        "obs": np.asarray(obs_list, dtype=float),
        "actions": np.asarray(action_list, dtype=float),
        "logp": np.asarray(logp_list, dtype=float),
        "values": np.asarray(value_list, dtype=float),  # len n+1
        "rewards": np.asarray(reward_list, dtype=float),
        "dones": np.asarray(done_list, dtype=float),
        "wealth": np.asarray(wealth_path, dtype=float),
        "terminal": float(wealth_path[-1]),
        "avg_turnover": float(np.mean(turnover_path)),
        "avg_gross_lev": float(np.mean(gross_lev_path)),
        "mean_lower_scale": float(np.mean(mean_lower_scale)),
        "mean_upper_scale": float(np.mean(mean_upper_scale)),
        "utility": float(_utility_np(wealth_path[-1], cfg.utility_kind, cfg.utility_gamma)),
    }


def _compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(rewards)
    adv = np.zeros(n, dtype=float)
    gae = 0.0
    for t in reversed(range(n)):
        nonterminal = 1.0 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * nonterminal - values[t]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        adv[t] = gae
    ret = adv + values[:-1]
    return adv, ret


# =========================
# PPO training
# =========================

def _flatten_batch(rollouts: List[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
    obs = np.concatenate([r["obs"] for r in rollouts], axis=0)
    actions = np.concatenate([r["actions"] for r in rollouts], axis=0)
    logp = np.concatenate([r["logp"] for r in rollouts], axis=0)
    adv = np.concatenate([r["adv"] for r in rollouts], axis=0)
    ret = np.concatenate([r["ret"] for r in rollouts], axis=0)
    return {
        "obs": obs,
        "actions": actions,
        "logp": logp,
        "adv": adv,
        "ret": ret,
    }


def _ppo_update(
    *,
    actor: Stage2RLActor,
    critic: Stage2RLCritic,
    actor_opt,
    critic_opt,
    batch: Dict[str, np.ndarray],
    cfg: TrainStage2RLConfig,
):
    device = next(actor.parameters()).device
    dtype = next(actor.parameters()).dtype

    obs_t = torch.as_tensor(batch["obs"], dtype=dtype, device=device)
    actions_t = torch.as_tensor(batch["actions"], dtype=dtype, device=device)
    old_logp_t = torch.as_tensor(batch["logp"], dtype=dtype, device=device)
    adv_t = torch.as_tensor(batch["adv"], dtype=dtype, device=device)
    ret_t = torch.as_tensor(batch["ret"], dtype=dtype, device=device)

    adv_t = (adv_t - adv_t.mean()) / (adv_t.std(unbiased=False) + 1e-8)

    for _ in range(cfg.ppo_epochs):
        mu, log_std = actor(obs_t)
        new_logp = _normal_log_prob(actions_t, mu, log_std)
        ratio = torch.exp(new_logp - old_logp_t)
        surr1 = ratio * adv_t
        surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * adv_t
        actor_loss = -torch.min(surr1, surr2).mean() - cfg.entropy_coef * _normal_entropy(log_std).mean()

        values = critic(obs_t)
        critic_loss = ((values - ret_t) ** 2).mean()

        actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), cfg.max_grad_norm)
        actor_opt.step()

        critic_opt.zero_grad(set_to_none=True)
        (cfg.value_coef * critic_loss).backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), cfg.max_grad_norm)
        critic_opt.step()

    return float(actor_loss.detach().cpu()), float(critic_loss.detach().cpu())


def _run_validation(
    *,
    center_agent,
    actor,
    critic,
    stage1_run_dir: Path,
    stage1_checkpoint: Path,
    filt_params: FilterParams,
    cfg: TrainStage2RLConfig,
    seed: int,
    n_paths: int,
):
    vals = []
    for i in range(n_paths):
        sample = _make_sample(
            seed_i=seed + 500000 + i,
            T_years=cfg.T_years,
            dt=cfg.dt,
            p0=cfg.p0,
            filt_params=filt_params,
            stage1_run_dir=stage1_run_dir,
            stage1_checkpoint=stage1_checkpoint,
            z=cfg.z,
            device=cfg.device,
            precompute_center_path=cfg.precompute_center_path,
        )
        vals.append(
            rollout_episode_rl(
                center_agent=center_agent,
                actor=actor,
                critic=critic,
                path=sample["path"],
                belief=sample["belief"],
                center_w_path=sample["center_w_path"],
                filt_params=filt_params,
                cfg=cfg,
                deterministic=True,
            )
        )

    return {
        "val_terminal": float(np.mean([v["terminal"] for v in vals])),
        "val_utility": float(np.mean([v["utility"] for v in vals])),
        "val_turnover": float(np.mean([v["avg_turnover"] for v in vals])),
        "val_gross_lev": float(np.mean([v["avg_gross_lev"] for v in vals])),
    }


def train_stage2_rl(
    *,
    stage1_run_dir: Path,
    stage1_checkpoint: Path,
    outdir: Path,
    cfg: TrainStage2RLConfig,
    seed: int,
    filter_mode: str = "true_params",
):
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

    actor = Stage2RLActor(obs_dim=9, hidden=cfg.hidden).to(cfg.device, dtype=cfg.dtype)
    critic = Stage2RLCritic(obs_dim=9, hidden=cfg.hidden).to(cfg.device, dtype=cfg.dtype)

    actor_opt = optim.Adam(actor.parameters(), lr=cfg.actor_lr, weight_decay=cfg.weight_decay)
    critic_opt = optim.Adam(critic.parameters(), lr=cfg.critic_lr, weight_decay=cfg.weight_decay)

    rows = []
    best_val_utility = -np.inf

    for it in range(1, cfg.iters + 1):
        rollouts = []
        for b in range(cfg.episodes_per_iter):
            sample = _make_sample(
                seed_i=seed + 10000 * it + b,
                T_years=cfg.T_years,
                dt=cfg.dt,
                p0=cfg.p0,
                filt_params=filt_params,
                stage1_run_dir=stage1_run_dir,
                stage1_checkpoint=stage1_checkpoint,
                z=cfg.z,
                device=cfg.device,
                precompute_center_path=cfg.precompute_center_path,
            )
            traj = rollout_episode_rl(
                center_agent=center_agent,
                actor=actor,
                critic=critic,
                path=sample["path"],
                belief=sample["belief"],
                center_w_path=sample["center_w_path"],
                filt_params=filt_params,
                cfg=cfg,
                deterministic=False,
            )
            adv, ret = _compute_gae(
                rewards=traj["rewards"],
                values=traj["values"],
                dones=traj["dones"],
                gamma=cfg.gamma,
                gae_lambda=cfg.gae_lambda,
            )
            traj["adv"] = adv
            traj["ret"] = ret
            rollouts.append(traj)

        batch = _flatten_batch(rollouts)
        actor_loss, critic_loss = _ppo_update(
            actor=actor,
            critic=critic,
            actor_opt=actor_opt,
            critic_opt=critic_opt,
            batch=batch,
            cfg=cfg,
        )

        val_stats = {
            "val_terminal": np.nan,
            "val_utility": np.nan,
            "val_turnover": np.nan,
            "val_gross_lev": np.nan,
        }
        if cfg.val_every > 0 and (it % cfg.val_every == 0 or it == cfg.iters):
            val_stats = _run_validation(
                center_agent=center_agent,
                actor=actor,
                critic=critic,
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
                        "actor_state_dict": actor.state_dict(),
                        "critic_state_dict": critic.state_dict(),
                        "train_cfg": asdict(cfg) | {
                            "filter_mode": filter_mode,
                            "best_iter": it,
                            "best_val_utility": best_val_utility,
                        },
                        "model_type": "stage2_rl",
                    },
                    outdir / "best_checkpoint.pt",
                )

        row = {
            "iter": it,
            "actor_loss": actor_loss,
            "critic_loss": critic_loss,
            "mean_terminal": float(np.mean([r["terminal"] for r in rollouts])),
            "std_terminal": float(np.std([r["terminal"] for r in rollouts], ddof=0)),
            "mean_utility": float(np.mean([r["utility"] for r in rollouts])),
            "avg_turnover": float(np.mean([r["avg_turnover"] for r in rollouts])),
            "avg_gross_lev": float(np.mean([r["avg_gross_lev"] for r in rollouts])),
            "mean_lower_scale": float(np.mean([r["mean_lower_scale"] for r in rollouts])),
            "mean_upper_scale": float(np.mean([r["mean_upper_scale"] for r in rollouts])),
            **val_stats,
        }
        rows.append(row)

        if it % 50 == 0:
            pd.DataFrame(rows).to_csv(outdir / "metrics.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "metrics.csv", index=False)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["iter"], df["mean_terminal"], label="train_terminal")
    m = df["val_terminal"].notna()
    if m.any():
        plt.plot(df.loc[m, "iter"], df.loc[m, "val_terminal"], label="val_terminal")
    plt.axhline(cfg.z, linestyle="--")
    plt.xlabel("iteration")
    plt.ylabel("terminal wealth")
    plt.title("Stage2 RL terminal wealth")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "learning_curve.png", dpi=200)
    plt.close(fig)

    fig = plt.figure(figsize=(8, 5))
    plt.plot(df["iter"], df["mean_utility"], label="train_utility")
    m = df["val_utility"].notna()
    if m.any():
        plt.plot(df.loc[m, "iter"], df.loc[m, "val_utility"], label="val_utility")
    plt.xlabel("iteration")
    plt.ylabel("utility")
    plt.title("Stage2 RL utility")
    plt.legend()
    fig.tight_layout()
    fig.savefig(outdir / "utility_curve.png", dpi=200)
    plt.close(fig)

    torch.save(
        {
            "actor_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "train_cfg": asdict(cfg) | {"filter_mode": filter_mode},
            "model_type": "stage2_rl",
        },
        outdir / "checkpoint.pt",
    )

    with open(outdir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "stage1_run_dir": str(stage1_run_dir.resolve()),
                "stage1_checkpoint": str(stage1_checkpoint.resolve()),
                "seed": seed,
                "filter_mode": filter_mode,
                "filter_params": {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in filt_params.__dict__.items()
                },
                **{
                    k: v
                    for k, v in asdict(cfg).items()
                    if k != "dtype"
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
    ap.add_argument("--iters", type=int, default=800)
    ap.add_argument("--episodes_per_iter", type=int, default=64)

    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=1 / 252)
    ap.add_argument("--z", type=float, default=1.2)
    ap.add_argument("--x0", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.5)
    ap.add_argument("--tcost", type=float, default=0.002)

    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--actor_lr", type=float, default=3e-4)
    ap.add_argument("--critic_lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-6)

    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae_lambda", type=float, default=0.95)
    ap.add_argument("--ppo_epochs", type=int, default=10)
    ap.add_argument("--clip_eps", type=float, default=0.2)
    ap.add_argument("--entropy_coef", type=float, default=1e-4)
    ap.add_argument("--value_coef", type=float, default=0.5)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)

    ap.add_argument("--gamma_risk", type=float, default=5.0)
    ap.add_argument("--qvi_width_floor", type=float, default=1e-4)

    ap.add_argument("--reward_scale", type=float, default=1.0)
    ap.add_argument("--no_log_wealth_increment", dest="use_log_wealth_increment", action="store_false")
    ap.add_argument("--terminal_utility_coef", type=float, default=1.0)
    ap.add_argument("--utility_kind", type=str, choices=["log", "sqrt", "power"], default="log")
    ap.add_argument("--utility_gamma", type=float, default=2.0)

    ap.add_argument("--turnover_coef", type=float, default=5e-4)
    ap.add_argument("--prior_dev_coef", type=float, default=1e-3)
    ap.add_argument("--gross_lev_coef", type=float, default=1e-4)

    ap.add_argument("--correction_scale", type=float, default=0.25)
    ap.add_argument("--log_scale_clip", type=float, default=2.0)

    ap.add_argument("--val_every", type=int, default=25)
    ap.add_argument("--val_n_paths", type=int, default=64)

    ap.add_argument("--precompute_center_path", dest="precompute_center_path", action="store_true")
    ap.add_argument("--no_precompute_center_path", dest="precompute_center_path", action="store_false")
    ap.set_defaults(precompute_center_path=True)

    ap.add_argument("--filter_mode", type=str, choices=["true_params", "estimated_params"], default="true_params")
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")

    args = ap.parse_args()

    cfg = TrainStage2RLConfig(
        T_years=args.T,
        dt=args.dt,
        z=args.z,
        x0=args.x0,
        p0=args.p0,
        tcost=args.tcost,
        hidden=args.hidden,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        weight_decay=args.weight_decay,
        iters=args.iters,
        episodes_per_iter=args.episodes_per_iter,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ppo_epochs=args.ppo_epochs,
        clip_eps=args.clip_eps,
        entropy_coef=args.entropy_coef,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        gamma_risk=args.gamma_risk,
        qvi_width_floor=args.qvi_width_floor,
        reward_scale=args.reward_scale,
        use_log_wealth_increment=args.use_log_wealth_increment,
        terminal_utility_coef=args.terminal_utility_coef,
        utility_kind=args.utility_kind,
        utility_gamma=args.utility_gamma,
        turnover_coef=args.turnover_coef,
        prior_dev_coef=args.prior_dev_coef,
        gross_lev_coef=args.gross_lev_coef,
        correction_scale=args.correction_scale,
        log_scale_clip=args.log_scale_clip,
        val_every=args.val_every,
        val_n_paths=args.val_n_paths,
        precompute_center_path=args.precompute_center_path,
        num_workers=args.num_workers,
        device=args.device,
    )

    train_stage2_rl(
        stage1_run_dir=Path(args.stage1_run_dir),
        stage1_checkpoint=Path(args.stage1_checkpoint),
        outdir=Path(args.outdir),
        cfg=cfg,
        seed=args.seed,
        filter_mode=args.filter_mode,
    )


if __name__ == "__main__":
    main()