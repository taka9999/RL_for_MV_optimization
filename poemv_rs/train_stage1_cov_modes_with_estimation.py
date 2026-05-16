from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .env_regime_cov import RSGBMEnvRegimeCov, RSGBMParamsRegimeCov, EpisodeConfigRegimeCov
from .heuristic import HeuristicThresholds, label_bull_bear_from_drawdowns, estimate_env_params_from_labeled_returns
from .train_stage1_cov_modes import (
    POEMVAgentCovModes,
    build_train_config,
    default_true_params,
    evaluate_agent,
    params_from_args,
    rollout_episode,
    save_checkpoint,
    save_config,
    save_learning_curves,
    set_seed,
)

try:
    from jumpmodels.jump import JumpModel
except Exception:  # pragma: no cover - optional dependency
    JumpModel = None


def simulate_price_only_regime_cov(
    params: RSGBMParamsRegimeCov,
    T_years: float,
    dt: float,
    s0: np.ndarray | None = None,
    p0: float = 0.5,
    seed: int = 0,
) -> Dict[str, np.ndarray]:
    if s0 is None:
        s0 = np.array([1.0, 1.0], dtype=float)
    env = RSGBMEnvRegimeCov(
        params,
        EpisodeConfigRegimeCov(
            T_years=T_years,
            dt=dt,
            x0=1.0,
            s0=np.asarray(s0, dtype=float),
            p0=p0,
            a_max=None,
            seed=seed,
            apply_action_projection=False,
        ),
    )
    obs = env.reset()
    n = env.n_steps
    S = np.empty((n + 1, len(obs["S"])), dtype=float)
    I = np.empty(n + 1, dtype=int)
    S[0] = obs["S"]
    I[0] = obs["I_true"]
    for k in range(n):
        obs, _, _ = env.step(np.zeros_like(obs["S"], dtype=float))
        S[k + 1] = obs["S"]
        I[k + 1] = obs["I_true"]
    ret = (S[1:] - S[:-1]) / np.maximum(np.abs(S[:-1]), 1e-12)
    logret = np.log(S[1:] / np.maximum(S[:-1], 1e-12))
    t = np.arange(n + 1, dtype=float) * dt
    return {"t": t, "S": S, "I": I, "ret": ret, "logret": logret}


def build_index_from_prices(S: np.ndarray, mode: str) -> np.ndarray:
    S = np.asarray(S, dtype=float)
    if mode == "asset1":
        return S[:, 0]
    if mode == "asset2":
        return S[:, 1]
    if mode == "equal_weight":
        S_norm = S / np.maximum(S[0:1], 1e-12)
        return np.mean(S_norm, axis=1)
    raise ValueError(f"Unknown index_mode={mode}")


def _reorder_bull_bear(mu1: np.ndarray, mu2: np.ndarray, Sigma1: np.ndarray, Sigma2: np.ndarray,
                       lam1: float, lam2: float, labels: Optional[np.ndarray] = None) -> Dict:
    score1 = float(np.sum(mu1))
    score2 = float(np.sum(mu2))
    if score1 >= score2:
        out = {
            "mu1": np.asarray(mu1, dtype=float),
            "mu2": np.asarray(mu2, dtype=float),
            "Sigma1": np.asarray(Sigma1, dtype=float),
            "Sigma2": np.asarray(Sigma2, dtype=float),
            "lam1": float(lam1),
            "lam2": float(lam2),
        }
        if labels is not None:
            out["regime_labels"] = np.asarray(labels, dtype=int)
        return out

    out = {
        "mu1": np.asarray(mu2, dtype=float),
        "mu2": np.asarray(mu1, dtype=float),
        "Sigma1": np.asarray(Sigma2, dtype=float),
        "Sigma2": np.asarray(Sigma1, dtype=float),
        "lam1": float(lam2),
        "lam2": float(lam1),
    }
    if labels is not None:
        labels = np.asarray(labels, dtype=int)
        out["regime_labels"] = 1 - labels
    return out


def estimate_params_via_heuristic(path: Dict[str, np.ndarray], dt: float, y_l: float, y_u: float, index_mode: str) -> Dict:
    index_series = build_index_from_prices(path["S"], index_mode)
    reg = label_bull_bear_from_drawdowns(index_series, HeuristicThresholds(Y_l=y_l, Y_u=y_u))
    est = estimate_env_params_from_labeled_returns(path["ret"], reg, dt)
    est = _reorder_bull_bear(est["mu1"], est["mu2"], est["Sigma1"], est["Sigma2"], est["lam1"], est["lam2"], reg)
    est["index_series"] = index_series.astype(float)
    return est


def _logsumexp_1d(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    m = np.max(a)
    return float(m + np.log(np.sum(np.exp(a - m))))


def _gaussian_logpdf_common_cov(y: np.ndarray, mu: np.ndarray, Sigma: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).reshape(-1)
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    d = y.shape[0]
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-10 * np.eye(d)
    diff = y - mu
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        Sigma = Sigma + 1e-8 * np.eye(d)
        sign, logdet = np.linalg.slogdet(Sigma)
    inv = np.linalg.inv(Sigma)
    quad = float(diff @ inv @ diff)
    return -0.5 * (d * np.log(2.0 * np.pi) + logdet + quad)


def _forward_backward_2state(
    Y: np.ndarray,
    pi0: np.ndarray,
    P: np.ndarray,
    mus: np.ndarray,
    Sigma: np.ndarray,
):
    T, d = Y.shape
    logA = np.log(np.clip(P, 1e-12, 1.0))
    logpi0 = np.log(np.clip(pi0, 1e-12, 1.0))

    logB = np.empty((T, 2), dtype=float)
    for t in range(T):
        for j in range(2):
            logB[t, j] = _gaussian_logpdf_common_cov(Y[t], mus[j], Sigma)

    alpha = np.empty((T, 2), dtype=float)
    alpha[0] = logpi0 + logB[0]
    for t in range(1, T):
        for j in range(2):
            alpha[t, j] = logB[t, j] + _logsumexp_1d(alpha[t - 1] + logA[:, j])

    loglik = _logsumexp_1d(alpha[-1])

    beta = np.zeros((T, 2), dtype=float)
    for t in range(T - 2, -1, -1):
        for i in range(2):
            beta[t, i] = _logsumexp_1d(logA[i, :] + logB[t + 1, :] + beta[t + 1, :])

    log_gamma = alpha + beta
    for t in range(T):
        log_gamma[t] -= _logsumexp_1d(log_gamma[t])
    gamma = np.exp(log_gamma)

    xi = np.empty((T - 1, 2, 2), dtype=float)
    for t in range(T - 1):
        M = np.empty((2, 2), dtype=float)
        for i in range(2):
            for j in range(2):
                M[i, j] = alpha[t, i] + logA[i, j] + logB[t + 1, j] + beta[t + 1, j]
        z = _logsumexp_1d(M.reshape(-1))
        xi[t] = np.exp(M - z)

    return gamma, xi, float(loglik)


def _fit_2state_hmm_common_cov(
    Y: np.ndarray,
    dt: float,
    n_iter: int = 50,
    tol: float = 1e-5,
    seed: int = 0,
):
    Y = np.asarray(Y, dtype=float)
    T, d = Y.shape

    score = np.mean(Y, axis=1)
    med = np.median(score)
    z0 = (score > med).astype(int)

    mus = np.vstack([
        Y[z0 == 1].mean(axis=0) if np.any(z0 == 1) else Y.mean(axis=0) + 1e-3,
        Y[z0 == 0].mean(axis=0) if np.any(z0 == 0) else Y.mean(axis=0) - 1e-3,
    ]).astype(float)

    Sigma = np.cov(Y.T, ddof=1)
    Sigma = 0.5 * (Sigma + Sigma.T) + 1e-8 * np.eye(d)

    P = np.array([[0.995, 0.005], [0.010, 0.990]], dtype=float)
    pi0 = np.array([0.5, 0.5], dtype=float)

    prev_loglik = -np.inf
    for _ in range(int(n_iter)):
        gamma, xi, loglik = _forward_backward_2state(Y, pi0, P, mus, Sigma)
        pi0 = gamma[0]
        pi0 = pi0 / np.sum(pi0)

        xi_sum = np.sum(xi, axis=0)
        gamma_sum = np.sum(gamma[:-1], axis=0)
        P = xi_sum / np.clip(gamma_sum[:, None], 1e-12, None)
        P = np.clip(P, 1e-8, 1.0)
        P = P / P.sum(axis=1, keepdims=True)

        for j in range(2):
            w = gamma[:, j]
            mus[j] = np.sum(w[:, None] * Y, axis=0) / np.clip(np.sum(w), 1e-12, None)

        Sigma_num = np.zeros((d, d), dtype=float)
        for j in range(2):
            diff = Y - mus[j]
            Sigma_num += (gamma[:, j][:, None] * diff).T @ diff
        Sigma = Sigma_num / np.clip(np.sum(gamma), 1e-12, None)
        Sigma = 0.5 * (Sigma + Sigma.T) + 1e-8 * np.eye(d)

        if np.isfinite(prev_loglik) and abs(loglik - prev_loglik) < tol:
            break
        prev_loglik = loglik

    if float(np.sum(mus[0])) < float(np.sum(mus[1])):
        mus = mus[[1, 0]]
        pi0 = pi0[[1, 0]]
        P = P[[1, 0]][:, [1, 0]]

    lam1 = max((1.0 - P[0, 0]) / max(dt, 1e-12), 1e-8)
    lam2 = max((1.0 - P[1, 1]) / max(dt, 1e-12), 1e-8)
    lam1 = min(lam1, 5.0)
    lam2 = min(lam2, 5.0)

    return {
        "mu1_step": np.asarray(mus[0], dtype=float),
        "mu2_step": np.asarray(mus[1], dtype=float),
        "Sigma_step": np.asarray(Sigma, dtype=float),
        "lam1": float(lam1),
        "lam2": float(lam2),
        "pi0": np.asarray(pi0, dtype=float),
        "P": np.asarray(P, dtype=float),
    }


def estimate_params_via_hmm(path: Dict[str, np.ndarray], dt: float, seed: int = 0) -> Dict:
    fit = _fit_2state_hmm_common_cov(path["logret"], dt=dt, n_iter=50, tol=1e-5, seed=seed)
    Sigma = fit["Sigma_step"] / dt
    Sigma = 0.5 * (Sigma + Sigma.T)
    drift1 = fit["mu1_step"] / dt
    drift2 = fit["mu2_step"] / dt
    mu1 = drift1 + 0.5 * np.diag(Sigma)
    mu2 = drift2 + 0.5 * np.diag(Sigma)
    out = _reorder_bull_bear(mu1, mu2, Sigma, Sigma, fit["lam1"], fit["lam2"])
    return out


def _roll_mean(x: np.ndarray, win: int) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    for t in range(len(x)):
        lo = max(0, t - win + 1)
        out[t] = np.mean(x[lo:t+1])
    return out


def _roll_std(x: np.ndarray, win: int) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    for t in range(len(x)):
        lo = max(0, t - win + 1)
        seg = x[lo:t+1]
        out[t] = np.std(seg, ddof=1) if len(seg) >= 2 else 0.0
    return out


def _rolling_drawdown_from_returns(x: np.ndarray, win: int) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    wealth = np.cumprod(1.0 + x)
    for t in range(len(x)):
        lo = max(0, t - win + 1)
        w = wealth[lo:t+1]
        peak = np.maximum.accumulate(w)
        dd = w / np.maximum(peak, 1e-12) - 1.0
        out[t] = np.min(dd)
    return out


def _rolling_downside_dev(x: np.ndarray, win: int) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    for t in range(len(x)):
        lo = max(0, t - win + 1)
        seg = x[lo:t+1]
        neg = np.minimum(seg, 0.0)
        out[t] = np.sqrt(np.mean(neg * neg))
    return out


def _rolling_cumret(x: np.ndarray, win: int) -> np.ndarray:
    out = np.empty_like(x, dtype=float)
    for t in range(len(x)):
        lo = max(0, t - win + 1)
        seg = x[lo:t+1]
        out[t] = np.prod(1.0 + seg) - 1.0
    return out


def _zscore_cols(X: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    mu = np.mean(X, axis=0)
    sd = np.std(X, axis=0, ddof=1)
    sd = np.where(sd < eps, 1.0, sd)
    return (X - mu) / sd


def estimate_params_via_jump_model(
    path: Dict[str, np.ndarray],
    dt: float,
    seed: int = 0,
    jump_penalty: float = 100.0,
    jm_max_iter: int = 400,
    jm_n_init: int = 10,
) -> Dict:
    if JumpModel is None:
        raise ImportError(
            "jumpmodels.jump.JumpModel could not be imported. Install jumpmodels or use heuristic/hmm."
        )

    ret_train = np.asarray(path["ret"], dtype=float)
    eq_ret = np.mean(ret_train, axis=1)

    roll_mean_5 = _roll_mean(eq_ret, 5)
    roll_std_5 = _roll_std(eq_ret, 5)
    roll_mean_21 = _roll_mean(eq_ret, 21)
    roll_std_21 = _roll_std(eq_ret, 21)
    roll_mean_63 = _roll_mean(eq_ret, 63)
    roll_std_63 = _roll_std(eq_ret, 63)
    roll_down_21 = _rolling_downside_dev(eq_ret, 21)
    roll_down_63 = _rolling_downside_dev(eq_ret, 63)
    roll_dd_63 = _rolling_drawdown_from_returns(eq_ret, 63)
    mean_spread_21_63 = roll_mean_21 - roll_mean_63
    vol_ratio_5_21 = roll_std_5 / np.maximum(roll_std_21, 1e-8)
    vol_ratio_21_63 = roll_std_21 / np.maximum(roll_std_63, 1e-8)

    X_feat = np.column_stack([
        eq_ret,
        roll_mean_21,
        roll_mean_63,
        roll_std_21,
        roll_std_63,
        roll_down_21,
        roll_down_63,
        roll_dd_63,
        mean_spread_21_63,
        vol_ratio_5_21,
        vol_ratio_21_63,
    ])
    X_feat = _zscore_cols(X_feat)

    jm = JumpModel(
        n_components=2,
        jump_penalty=jump_penalty,
        cont=False,
        random_state=seed,
        max_iter=jm_max_iter,
        n_init=jm_n_init,
        verbose=0,
    )
    jm.fit(X_feat, ret_ser=eq_ret, sort_by="cumret")
    labels = np.asarray(jm.labels_, dtype=int)

    # jumpmodels sorts so label 1 is the better state, but we still reorder from estimated mus for safety
    reg1 = labels == 1
    reg0 = labels == 0
    mu1 = ret_train[reg1].mean(axis=0) * 252.0 if np.any(reg1) else ret_train.mean(axis=0) * 252.0
    mu2 = ret_train[reg0].mean(axis=0) * 252.0 if np.any(reg0) else ret_train.mean(axis=0) * 252.0

    logret_train = np.asarray(path["logret"], dtype=float)
    Sigma1 = np.cov(logret_train[reg1].T, ddof=1) * 252.0 if np.sum(reg1) >= 2 else np.cov(logret_train.T, ddof=1) * 252.0
    Sigma2 = np.cov(logret_train[reg0].T, ddof=1) * 252.0 if np.sum(reg0) >= 2 else np.cov(logret_train.T, ddof=1) * 252.0
    Sigma1 = 0.5 * (Sigma1 + Sigma1.T)
    Sigma2 = 0.5 * (Sigma2 + Sigma2.T)

    n00 = n01 = n10 = n11 = 0
    for t in range(1, len(labels)):
        i, j = int(labels[t - 1]), int(labels[t])
        if i == 0 and j == 0:
            n00 += 1
        elif i == 0 and j == 1:
            n01 += 1
        elif i == 1 and j == 0:
            n10 += 1
        else:
            n11 += 1
    P = np.array([[n00, n01], [n10, n11]], dtype=float)
    P = P / np.clip(P.sum(axis=1, keepdims=True), 1e-12, None)

    # labels==1 is bull before reordering
    bullbull = P[1, 1]
    bearbear = P[0, 0]
    lam1 = min(max((1.0 - bullbull) / dt, 1e-8), 2.0)
    lam2 = min(max((1.0 - bearbear) / dt, 1e-8), 3.5)

    out = _reorder_bull_bear(mu1, mu2, Sigma1, Sigma2, lam1, lam2, labels)
    out["jm_transmat"] = np.asarray(getattr(jm, "transmat_", P), dtype=float)
    out["jm_features"] = X_feat
    return out


def load_jm_outputs(jm_params_json: Optional[str], jm_regimes_csv: Optional[str], path: Dict[str, np.ndarray], dt: float) -> Dict:
    if jm_params_json:
        with open(jm_params_json, "r", encoding="utf-8") as f:
            est = json.load(f)
        required = ["mu1", "mu2", "Sigma1", "Sigma2", "lam1", "lam2"]
        missing = [k for k in required if k not in est]
        if missing:
            raise ValueError(f"JM params json is missing keys: {missing}")
        return {
            "mu1": np.asarray(est["mu1"], dtype=float),
            "mu2": np.asarray(est["mu2"], dtype=float),
            "Sigma1": np.asarray(est["Sigma1"], dtype=float),
            "Sigma2": np.asarray(est["Sigma2"], dtype=float),
            "lam1": float(est["lam1"]),
            "lam2": float(est["lam2"]),
        }

    if jm_regimes_csv:
        reg = pd.read_csv(jm_regimes_csv).iloc[:, 0].to_numpy(dtype=int)
        if reg.shape[0] == path["S"].shape[0] - 1:
            reg = np.concatenate([[reg[0]], reg])
        if reg.shape[0] != path["S"].shape[0]:
            raise ValueError(
                f"JM regime labels length {reg.shape[0]} is incompatible with price path length {path['S'].shape[0]}"
            )
        est = estimate_env_params_from_labeled_returns(path["ret"], reg, dt)
        return _reorder_bull_bear(est["mu1"], est["mu2"], est["Sigma1"], est["Sigma2"], est["lam1"], est["lam2"], reg)

    raise ValueError(
        "estimation_method='jm_external' requires either --jm_params_json or --jm_regimes_csv."
    )


def inject_estimated_params_into_args(args: argparse.Namespace, est: Dict) -> None:
    args.belief_mode = "estimated_params"
    args.est_mu1_json = json.dumps(np.asarray(est["mu1"], dtype=float).tolist())
    args.est_mu2_json = json.dumps(np.asarray(est["mu2"], dtype=float).tolist())
    args.est_sigma1_json = json.dumps(np.asarray(est["Sigma1"], dtype=float).tolist())
    args.est_sigma2_json = json.dumps(np.asarray(est["Sigma2"], dtype=float).tolist())
    args.est_lam1 = float(est["lam1"])
    args.est_lam2 = float(est["lam2"])
    if getattr(args, "est_r", None) is None:
        args.est_r = float(args.r)


def save_estimation_artifacts(outdir: Path, path: Dict[str, np.ndarray], estimation_method: str, est: Dict, index_mode: str) -> None:
    payload = {
        "estimation_method": estimation_method,
        "mu1": np.asarray(est["mu1"], dtype=float).tolist(),
        "mu2": np.asarray(est["mu2"], dtype=float).tolist(),
        "Sigma1": np.asarray(est["Sigma1"], dtype=float).tolist(),
        "Sigma2": np.asarray(est["Sigma2"], dtype=float).tolist(),
        "lam1": float(est["lam1"]),
        "lam2": float(est["lam2"]),
        "index_mode": index_mode,
    }
    with open(outdir / "estimated_params_from_wrapper.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    if "regime_labels" in est:
        pd.DataFrame({"regime": np.asarray(est["regime_labels"], dtype=int)}).to_csv(
            outdir / "estimated_regime_labels.csv", index=False
        )
    if "index_series" in est:
        pd.DataFrame({"index_series": np.asarray(est["index_series"], dtype=float)}).to_csv(
            outdir / "estimation_index_series.csv", index=False
        )
    if "jm_transmat" in est:
        pd.DataFrame(np.asarray(est["jm_transmat"], dtype=float)).to_csv(
            outdir / "jm_transition_matrix.csv", index=False
        )
    pd.DataFrame(path["S"], columns=[f"S{i+1}" for i in range(path["S"].shape[1])]).to_csv(
        outdir / "estimation_prices.csv", index=False
    )
    pd.DataFrame(path["ret"], columns=[f"ret{i+1}" for i in range(path["ret"].shape[1])]).to_csv(
        outdir / "estimation_returns.csv", index=False
    )


def run_training(args: argparse.Namespace) -> None:
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    true_params, filt_params = params_from_args(args)
    train_cfg = build_train_config(args, true_params)
    agent = POEMVAgentCovModes(train_cfg)

    ep_cfg = EpisodeConfigRegimeCov(
        T_years=args.T,
        dt=args.dt,
        x0=args.x0,
        s0=np.array([1.0, 1.0], dtype=float),
        p0=args.p0,
        a_max=args.a_max,
        seed=args.seed,
        apply_action_projection=True,
    )

    save_config(outdir, args, true_params, filt_params)

    rows: List[Dict[str, float]] = []
    best_val = -np.inf

    for it in range(1, args.iters + 1):
        trajs = []
        batch_xT = []
        for j in range(args.episodes_per_iter):
            traj = rollout_episode(
                agent=agent,
                true_params=true_params,
                filt_params=filt_params,
                ep_cfg=ep_cfg,
                seed=args.seed + 100000 * it + j,
                deterministic=False,
            )
            trajs.append(traj)
            batch_xT.append(float(traj["x"][-1]))

        loss_c, loss_a = agent.update_from_episodes(trajs)
        if (it % int(args.omega_update_every)) == 0:
            agent.update_omega(float(np.mean(batch_xT)))
        agent.step_schedulers()

        row: Dict[str, float] = {
            "iter": float(it),
            "loss_critic": float(loss_c),
            "loss_actor": float(loss_a),
            "mean_xT_batch": float(np.mean(batch_xT)),
            "std_xT_batch": float(np.std(batch_xT, ddof=1)) if len(batch_xT) > 1 else 0.0,
            "omega": float(agent.omega.detach().cpu().item()),
            "target_z": float(args.z),
        }
        row.update(agent.current_lrs())
        row.update(agent.policy_values())

        if (it % args.val_every) == 0 or it == 1 or it == args.iters:
            val = evaluate_agent(
                agent=agent,
                true_params=true_params,
                filt_params=filt_params,
                ep_cfg=ep_cfg,
                seed0=args.seed + 500000 + it,
                n_paths=args.val_n_paths,
            )
            row.update(val)
            if val["val_mean_xT"] > best_val:
                best_val = val["val_mean_xT"]
                save_checkpoint(agent, outdir, name="best_checkpoint.pt")

        rows.append(row)
        if it % max(1, args.val_every) == 0:
            print(
                f"iter={it} loss_c={row['loss_critic']:.6f} loss_a={row['loss_actor']:.6f} "
                f"mean_xT={row['mean_xT_batch']:.4f} omega={row['omega']:.4f}"
            )

    metrics = pd.DataFrame(rows)
    metrics.to_csv(outdir / "metrics.csv", index=False)
    save_learning_curves(metrics, outdir)
    save_checkpoint(agent, outdir, name="checkpoint.pt")
    print(f"Saved outputs to {outdir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=str, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--episodes_per_iter", type=int, default=16)
    ap.add_argument("--val_every", type=int, default=50)
    ap.add_argument("--val_n_paths", type=int, default=64)
    ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--dt", type=float, default=1/252)
    ap.add_argument("--x0", type=float, default=1.0)
    ap.add_argument("--p0", type=float, default=0.5)
    ap.add_argument("--z", type=float, default=1.2)
    ap.add_argument("--Lambda", type=float, default=1.0)
    ap.add_argument("--r", type=float, default=0.01)
    ap.add_argument("--a_max", type=float, default=2.0)
    ap.add_argument("--cap_mode", type=str, default="component_tanh", choices=["none", "component_tanh", "l1_radial"])
    ap.add_argument("--sigma_mode", type=str, default="regime", choices=["common", "regime"])
    ap.add_argument("--cov_mode", type=str, default="belief_avg_regime", choices=["common", "belief_avg_regime"])
    ap.add_argument("--mu1_json", type=str, default=None)
    ap.add_argument("--mu2_json", type=str, default=None)
    ap.add_argument("--sigma1_json", type=str, default=None)
    ap.add_argument("--sigma2_json", type=str, default=None)
    ap.add_argument("--lam1", type=float, default=0.36)
    ap.add_argument("--lam2", type=float, default=2.89)
    ap.add_argument("--m_poly", type=int, default=2)
    ap.add_argument("--mg_poly", type=int, default=2)
    ap.add_argument("--g_p_dep", action="store_true")
    ap.add_argument("--actor_mix_tail", type=float, default=0.5)
    ap.add_argument("--cov_scale", type=float, default=1.0)
    ap.add_argument("--alpha_w", type=float, default=5e-3)
    ap.add_argument("--alpha_theta", type=float, default=3e-5)
    ap.add_argument("--alpha_phi", type=float, default=1e-4)
    ap.add_argument("--omega_init", type=float, default=0.0)
    ap.add_argument("--omega_update_every", type=int, default=10)
    ap.add_argument("--critic_steps", type=int, default=10)
    ap.add_argument("--omega_ema_beta", type=float, default=0.9)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--lr_step_every", type=int, default=0)
    ap.add_argument("--lr_gamma", type=float, default=1.0)
    ap.add_argument("--device", type=str, default="cpu")

    ap.add_argument("--estimation_method", type=str, default="heuristic", choices=["heuristic", "hmm", "jump_model", "jm_external"])
    ap.add_argument("--estimation_T", type=float, default=20.0, help="Horizon used to generate the estimation sample.")
    ap.add_argument("--estimation_seed", type=int, default=None, help="Seed for estimation sample generation. Defaults to seed+777.")
    ap.add_argument("--index_mode", type=str, default="equal_weight", choices=["asset1", "asset2", "equal_weight"])
    ap.add_argument("--Y_l", type=float, default=0.19)
    ap.add_argument("--Y_u", type=float, default=0.24)
    ap.add_argument("--jm_params_json", type=str, default=None, help="External JM output params json with mu1,mu2,Sigma1,Sigma2,lam1,lam2.")
    ap.add_argument("--jm_regimes_csv", type=str, default=None, help="External JM regime labels csv, one label per row.")
    ap.add_argument("--jump_penalty", type=float, default=100.0)
    ap.add_argument("--jm_max_iter", type=int, default=400)
    ap.add_argument("--jm_n_init", type=int, default=10)

    args = ap.parse_args()

    # fields consumed by underlying trainer config/save_config
    args.belief_mode = "estimated_params"
    args.est_mu1_json = None
    args.est_mu2_json = None
    args.est_sigma1_json = None
    args.est_sigma2_json = None
    args.est_lam1 = None
    args.est_lam2 = None
    args.est_r = None

    true_params = default_true_params(r=args.r, sigma_mode=args.sigma_mode)
    tmp_true, _ = params_from_args(args)
    true_params = tmp_true

    estimation_T = float(args.estimation_T if args.estimation_T is not None else args.T)
    estimation_seed = int(args.estimation_seed if args.estimation_seed is not None else (args.seed + 777))
    path = simulate_price_only_regime_cov(true_params, estimation_T, args.dt, s0=np.array([1.0, 1.0], dtype=float), p0=args.p0, seed=estimation_seed)

    if args.estimation_method == "heuristic":
        est = estimate_params_via_heuristic(path, args.dt, args.Y_l, args.Y_u, args.index_mode)
    elif args.estimation_method == "hmm":
        est = estimate_params_via_hmm(path, args.dt, seed=estimation_seed)
    elif args.estimation_method == "jump_model":
        est = estimate_params_via_jump_model(
            path,
            dt=args.dt,
            seed=estimation_seed,
            jump_penalty=args.jump_penalty,
            jm_max_iter=args.jm_max_iter,
            jm_n_init=args.jm_n_init,
        )
    else:
        est = load_jm_outputs(args.jm_params_json, args.jm_regimes_csv, path, args.dt)

    inject_estimated_params_into_args(args, est)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_estimation_artifacts(outdir, path, args.estimation_method, est, args.index_mode)
    run_training(args)


if __name__ == "__main__":
    main()
