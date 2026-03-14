from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


def _safe_read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _pick_last_valid(series: pd.Series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.iloc[-1])


def _pick_mean_tail(series: pd.Series, tail: int = 100):
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) == 0:
        return np.nan
    return float(s.tail(min(tail, len(s))).mean())


def summarize_one_run(run_dir: Path, tail: int = 100) -> Dict:
    metrics_path = run_dir / "metrics.csv"
    cfg_path = run_dir / "run_config.json"

    if not metrics_path.exists():
        return {
            "run_dir": str(run_dir),
            "error": "metrics.csv not found",
        }

    df = pd.read_csv(metrics_path)
    cfg = _safe_read_json(cfg_path)

    z = cfg.get("z", np.nan)
    alpha_theta = cfg.get("alpha_theta", np.nan)
    alpha_phi = cfg.get("alpha_phi", np.nan)
    alpha_w = cfg.get("alpha_w", np.nan)
    Lambda = cfg.get("Lambda", np.nan)
    a_max = cfg.get("a_max", np.nan)
    T = cfg.get("T", np.nan)
    critic_steps = cfg.get("critic_steps", np.nan)
    omega_ema_beta = cfg.get("omega_ema_beta", np.nan)

    row = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "n_iters": int(len(df)),
        "T": T,
        "z": z,
        "Lambda": Lambda,
        "a_max": a_max,
        "alpha_theta": alpha_theta,
        "alpha_phi": alpha_phi,
        "alpha_w": alpha_w,
        "critic_steps": critic_steps,
        "omega_ema_beta": omega_ema_beta,
    }

    # terminal / target tracking
    if "mean_terminal" in df.columns:
        row["mean_terminal_last"] = _pick_last_valid(df["mean_terminal"])
        row["mean_terminal_tail_mean"] = _pick_mean_tail(df["mean_terminal"], tail=tail)
    else:
        row["mean_terminal_last"] = np.nan
        row["mean_terminal_tail_mean"] = np.nan

    if "mean_xT_window" in df.columns:
        row["mean_xT_window_last"] = _pick_last_valid(df["mean_xT_window"])
        row["mean_xT_window_tail_mean"] = _pick_mean_tail(df["mean_xT_window"], tail=tail)
    else:
        row["mean_xT_window_last"] = np.nan
        row["mean_xT_window_tail_mean"] = np.nan

    if np.isfinite(z):
        row["gap_to_target_last"] = row["mean_terminal_last"] - z if np.isfinite(row["mean_terminal_last"]) else np.nan
        row["gap_to_target_tail_mean"] = row["mean_terminal_tail_mean"] - z if np.isfinite(row["mean_terminal_tail_mean"]) else np.nan
        row["abs_gap_to_target_last"] = abs(row["gap_to_target_last"]) if np.isfinite(row["gap_to_target_last"]) else np.nan
        row["abs_gap_to_target_tail_mean"] = abs(row["gap_to_target_tail_mean"]) if np.isfinite(row["gap_to_target_tail_mean"]) else np.nan
    else:
        row["gap_to_target_last"] = np.nan
        row["gap_to_target_tail_mean"] = np.nan
        row["abs_gap_to_target_last"] = np.nan
        row["abs_gap_to_target_tail_mean"] = np.nan

    # omega
    row["omega_last"] = _pick_last_valid(df["omega"]) if "omega" in df.columns else np.nan
    row["omega_tail_mean"] = _pick_mean_tail(df["omega"], tail=tail) if "omega" in df.columns else np.nan

    # losses
    row["loss_actor_last"] = _pick_last_valid(df["loss_actor"]) if "loss_actor" in df.columns else np.nan
    row["loss_actor_tail_mean"] = _pick_mean_tail(df["loss_actor"], tail=tail) if "loss_actor" in df.columns else np.nan
    row["loss_critic_last"] = _pick_last_valid(df["loss_critic"]) if "loss_critic" in df.columns else np.nan
    row["loss_critic_tail_mean"] = _pick_mean_tail(df["loss_critic"], tail=tail) if "loss_critic" in df.columns else np.nan

    # leverage / cash / action
    row["avg_gross_leverage_last"] = _pick_last_valid(df["avg_gross_leverage"]) if "avg_gross_leverage" in df.columns else np.nan
    row["avg_gross_leverage_tail_mean"] = _pick_mean_tail(df["avg_gross_leverage"], tail=tail) if "avg_gross_leverage" in df.columns else np.nan

    row["avg_cash_weight_last"] = _pick_last_valid(df["avg_cash_weight"]) if "avg_cash_weight" in df.columns else np.nan
    row["avg_cash_weight_tail_mean"] = _pick_mean_tail(df["avg_cash_weight"], tail=tail) if "avg_cash_weight" in df.columns else np.nan

    row["avg_abs_action_last"] = _pick_last_valid(df["avg_abs_action"]) if "avg_abs_action" in df.columns else np.nan
    row["avg_abs_action_tail_mean"] = _pick_mean_tail(df["avg_abs_action"], tail=tail) if "avg_abs_action" in df.columns else np.nan

    # optional extremes
    row["max_gross_leverage_last"] = _pick_last_valid(df["max_gross_leverage"]) if "max_gross_leverage" in df.columns else np.nan
    row["min_cash_weight_last"] = _pick_last_valid(df["min_cash_weight"]) if "min_cash_weight" in df.columns else np.nan

    # ranking score: smaller is better
    score_terms = []
    if np.isfinite(row["abs_gap_to_target_tail_mean"]):
        score_terms.append(row["abs_gap_to_target_tail_mean"])
    if np.isfinite(row["avg_gross_leverage_tail_mean"]):
        score_terms.append(0.05 * row["avg_gross_leverage_tail_mean"])
    row["score"] = float(np.sum(score_terms)) if score_terms else np.nan

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run_dirs",
        nargs="+",
        required=True,
        help="List of run directories containing metrics.csv and run_config.json",
    )
    ap.add_argument(
        "--tail",
        type=int,
        default=100,
        help="Use the last N iterations for tail averages",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="run_metrics_comparison.csv",
        help="Output CSV path",
    )
    args = ap.parse_args()

    rows: List[Dict] = []
    for rd in args.run_dirs:
        rows.append(summarize_one_run(Path(rd), tail=args.tail))

    out = pd.DataFrame(rows)

    sort_cols = []
    if "score" in out.columns:
        sort_cols.append("score")
    if "abs_gap_to_target_tail_mean" in out.columns:
        sort_cols.append("abs_gap_to_target_tail_mean")

    if sort_cols:
        out = out.sort_values(sort_cols, na_position="last").reset_index(drop=True)

    out.to_csv(args.out_csv, index=False)

    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(out)

    print(f"\nSaved comparison CSV to: {args.out_csv}")


if __name__ == "__main__":
    main()