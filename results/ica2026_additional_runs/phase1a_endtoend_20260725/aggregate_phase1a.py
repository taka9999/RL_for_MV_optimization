#!/usr/bin/env python3
"""
Phase 1A aggregation script (read-only over existing evaluation outputs).

Reads ONLY already-produced, gate-passed evaluation outputs
(eval_stage1/eval_stage1_postfix/eval_stage2) and training run_config.json /
metrics.csv files for seeds 0-4. Never launches, resumes, or modifies any
training or evaluation run. Writes new CSVs/tables/figures/reports into
--out-dir only; every input file is opened read-only.

Usage:
    python aggregate_phase1a.py --base-dir <phase1a_endtoend_20260725 dir> \
        --out-dir <phase1a_endtoend_20260725/aggregate dir>

Deterministic: re-running with the same inputs reproduces identical output
files (no randomness anywhere in this script).
"""
from __future__ import annotations
import argparse
import json
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEEDS = [0, 1, 2, 3, 4]
Z_TARGET = 1.2
EVAL_SEED = 999
EVAL_N_PATHS = 1000
A_MAX = 1.0
T_YEARS = 1.0
LEV_NEAR_CAP_THRESHOLD = 1.980
TRADING_DAYS_PER_YEAR = 252
N_SEEDS = len(SEEDS)
T_CRIT_95 = float(stats.t.ppf(0.975, N_SEEDS - 1))  # df = 4 for 5 training seeds

# Stage 1 evaluation directory actually used per seed. Seeds 1/2 use the
# post-quoting-fix re-run (see git commit 04c4f2e); their original
# eval_stage1/seed_1, eval_stage1/seed_2 directories are the empty,
# failed-evaluation artifacts from the unquoted-path shell-splitting bug and
# are explicitly excluded here (never auto-picked up).
STAGE1_EVAL_REL = {
    0: "eval_stage1/seed_0",
    1: "eval_stage1_postfix/seed_1",
    2: "eval_stage1_postfix/seed_2",
    3: "eval_stage1/seed_3",
    4: "eval_stage1/seed_4",
}
STAGE1_EVAL_EXCLUDED_REL = {
    1: "eval_stage1/seed_1",
    2: "eval_stage1/seed_2",
}
STAGE1_EXCLUSION_REASON = (
    "excluded because of known invocation bug (unquoted-path shell-splitting "
    "in run_eval_stage1(), fixed in commit 04c4f2e); directory is empty, "
    "evaluation never completed"
)
STAGE2_EVAL_REL = {s: f"eval_stage2/seed_{s}" for s in SEEDS}
STAGE2_METHODS = ["Center+DNNBand", "CenterOnly_Daily", "CenterOnly_Monthly", "EW_Monthly"]
# Short labels for figure readability only (item 9) - full names remain in all CSVs/tables.
STAGE2_METHOD_SHORT = {
    "Center+DNNBand": "Center+DNNBand",
    "CenterOnly_Daily": "Center-Daily",
    "CenterOnly_Monthly": "Center-Monthly",
    "EW_Monthly": "EW-Monthly",
}
STAGE1_METHOD_SHORT = {"RL": "HJB-guided RL", "EW": "Equal weight", "MinVar": "Min variance"}
POSITIVITY_FLOOR = 1e-10  # matches _utility_torch's eps in poemv_rs/stage2_direct_boundary.py
STAGE2_DAILY_METHODS = {"Center+DNNBand", "CenterOnly_Daily"}  # dwell-fraction only meaningful here
STAGE1_METHODS = ["RL", "EW", "MinVar"]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def jd(v) -> str:
    return json.dumps(v, sort_keys=True)


# ---------------------------------------------------------------------------
# Section 1: validation
# ---------------------------------------------------------------------------

def validate(base: Path) -> tuple[list[str], list[str], dict]:
    """Returns (blocking_issues, warnings, info)."""
    issues: list[str] = []
    warnings: list[str] = []
    info: dict = {}

    for s in SEEDS:
        for d in (f"stage1/seed_{s}", f"stage2/seed_{s}"):
            if not (base / d).exists():
                issues.append(f"MISSING directory: {d}")
        s1d = STAGE1_EVAL_REL[s]
        if not (base / s1d / "eval_summary.csv").exists():
            issues.append(f"MISSING stage1 eval summary for seed {s}: {s1d}")
        s2d = STAGE2_EVAL_REL[s]
        if not (base / s2d / "eval_summary.csv").exists():
            issues.append(f"MISSING stage2 eval summary for seed {s}: {s2d}")
    if issues:
        return issues, warnings, info  # can't safely proceed further

    s1_cfgs = {s: load_json(base / STAGE1_EVAL_REL[s] / "eval_config.json") for s in SEEDS}
    s2_cfgs = {s: load_json(base / STAGE2_EVAL_REL[s] / "eval_config.json") for s in SEEDS}

    for key in ["n_paths", "seed", "T", "dt", "z", "a_max", "x0", "p0", "r", "filter_mode"]:
        vals = {s: s1_cfgs[s].get(key) for s in SEEDS}
        if len({jd(v) for v in vals.values()}) != 1:
            issues.append(f"stage1 eval_config[{key}] differs across seeds: {vals}")
    for key in ["n_paths", "seed", "T", "dt", "z", "x0", "tcost", "monthly_steps",
                "lev_cap", "filter_mode"]:
        vals = {s: s2_cfgs[s].get(key) for s in SEEDS}
        if len({jd(v) for v in vals.values()}) != 1:
            issues.append(f"stage2 eval_config[{key}] differs across seeds: {vals}")

    if s1_cfgs[0].get("T") != T_YEARS or s1_cfgs[0].get("a_max") != A_MAX:
        issues.append(f"stage1 eval T/a_max != expected {T_YEARS}/{A_MAX}: "
                       f"{s1_cfgs[0].get('T')}/{s1_cfgs[0].get('a_max')}")
    if s1_cfgs[0].get("n_paths") != EVAL_N_PATHS or s1_cfgs[0].get("seed") != EVAL_SEED:
        issues.append("stage1 eval n_paths/seed != expected common convention "
                       f"({EVAL_N_PATHS}/{EVAL_SEED})")
    if s2_cfgs[0].get("n_paths") != EVAL_N_PATHS or s2_cfgs[0].get("seed") != EVAL_SEED:
        issues.append("stage2 eval n_paths/seed != expected common convention "
                       f"({EVAL_N_PATHS}/{EVAL_SEED})")

    train_cfgs = {s: load_json(base / f"stage1/seed_{s}/run_config.json") for s in SEEDS}
    for key in ["policy_params", "true_params", "filter_params", "mode",
                "estimation_method", "T", "a_max", "dt", "z", "cap_mode"]:
        vals = {s: train_cfgs[s].get(key) for s in SEEDS}
        if len({jd(v) for v in vals.values()}) != 1:
            issues.append(f"stage1 run_config[{key}] differs across seeds")
    for s in SEEDS:
        if train_cfgs[s].get("seed") != s:
            issues.append(f"stage1 seed {s}: run_config 'seed' field = "
                           f"{train_cfgs[s].get('seed')} (expected {s})")

    stage2_train_cfgs = {s: load_json(base / f"stage2/seed_{s}/run_config.json") for s in SEEDS}
    skip_keys = {"stage1_run_dir", "stage1_checkpoint", "seed"}
    for key in stage2_train_cfgs[0].keys():
        if key in skip_keys:
            continue
        vals = {s: stage2_train_cfgs[s].get(key) for s in SEEDS}
        if len({jd(v) for v in vals.values()}) != 1:
            issues.append(f"stage2 run_config[{key}] differs across seeds")
    for s in SEEDS:
        if stage2_train_cfgs[s].get("seed") != s:
            issues.append(f"stage2 seed {s}: run_config 'seed' field mismatch")
        ckpt_path = Path(stage2_train_cfgs[s]["stage1_checkpoint"])
        if ckpt_path.parent.name != f"seed_{s}":
            issues.append(f"stage2 seed {s}: stage1_checkpoint does not point at seed_{s}: "
                           f"{ckpt_path}")

    def path_series(csv_path, method, col="terminal_wealth"):
        df = pd.read_csv(csv_path)
        sub = df[df["method"] == method].sort_values("path_id")
        return sub[col].to_numpy()

    ref_ew = path_series(base / STAGE1_EVAL_REL[0] / "eval_terminal_by_path.csv", "EW")
    for s in SEEDS[1:]:
        arr = path_series(base / STAGE1_EVAL_REL[s] / "eval_terminal_by_path.csv", "EW")
        if not np.array_equal(arr, ref_ew):
            issues.append(f"stage1 seed {s}: EW baseline terminal_wealth NOT identical "
                           f"to seed 0 - evaluation paths may differ across seeds")
    info["stage1_common_paths_verification"] = (
        "EW terminal_wealth (a method independent of the trained RL policy) is "
        "byte-identical across all 5 seeds' Stage1 evaluations, confirming the "
        "same underlying market scenarios / random paths were used."
    )

    ref_ewm = path_series(base / STAGE2_EVAL_REL[0] / "eval_terminal_by_path.csv", "EW_Monthly")
    for s in SEEDS[1:]:
        arr = path_series(base / STAGE2_EVAL_REL[s] / "eval_terminal_by_path.csv", "EW_Monthly")
        if not np.array_equal(arr, ref_ewm):
            issues.append(f"stage2 seed {s}: EW_Monthly baseline terminal_wealth NOT "
                           f"identical to seed 0 - evaluation paths may differ across seeds")
    info["stage2_common_paths_verification"] = (
        "EW_Monthly terminal_wealth (independent of the trained Stage2 policy) is "
        "byte-identical across all 5 seeds' Stage2 evaluations."
    )
    warnings.append(
        "Stage1-vs-Stage2 path correspondence (used only in the "
        "stage2_net_minus_stage1_terminal paired comparison) is inferred from "
        "identical eval_config parameters (T, dt, x0, eval seed=999, n_paths=1000) "
        "across the two stages, NOT independently cryptographically verified via a "
        "shared baseline method (Stage1's EW and Stage2's EW_Monthly are different "
        "rebalancing rules and cannot be compared directly for this purpose)."
    )

    for s in SEEDS:
        df1 = pd.read_csv(base / STAGE1_EVAL_REL[s] / "eval_terminal_by_path.csv")
        for m in df1["method"].unique():
            sub = df1[df1["method"] == m]
            if sub["path_id"].duplicated().any():
                issues.append(f"stage1 seed {s} method {m}: duplicate path_id rows")
            if sub["path_id"].nunique() != EVAL_N_PATHS:
                issues.append(f"stage1 seed {s} method {m}: path count "
                               f"{sub['path_id'].nunique()} != {EVAL_N_PATHS}")
        expected1 = set(STAGE1_METHODS)
        got1 = set(df1["method"].unique())
        if got1 != expected1:
            issues.append(f"stage1 seed {s}: unexpected method set {got1} "
                           f"(expected {expected1})")

        df2 = pd.read_csv(base / STAGE2_EVAL_REL[s] / "eval_terminal_by_path.csv")
        for m in df2["method"].unique():
            sub = df2[df2["method"] == m]
            if sub["path_id"].duplicated().any():
                issues.append(f"stage2 seed {s} method {m}: duplicate path_id rows")
            if sub["path_id"].nunique() != EVAL_N_PATHS:
                issues.append(f"stage2 seed {s} method {m}: path count "
                               f"{sub['path_id'].nunique()} != {EVAL_N_PATHS}")
        expected2 = set(STAGE2_METHODS)
        got2 = set(df2["method"].unique())
        if got2 != expected2:
            issues.append(f"stage2 seed {s}: unexpected method set {got2} "
                           f"(expected {expected2})")

        for col in ["terminal_wealth", "avg_gross_leverage", "avg_cash_weight", "path_sharpe_ann"]:
            vals = pd.to_numeric(df1[col], errors="coerce")
            if not np.isfinite(vals).all():
                issues.append(f"stage1 seed {s}: {(~np.isfinite(vals)).sum()} "
                               f"non-finite values in column {col}")

        for col in df2.columns:
            if col in ("method",):
                continue
            if col == "mean_band_width":
                dnn = df2[df2["method"] == "Center+DNNBand"]
                if not np.isfinite(pd.to_numeric(dnn[col], errors="coerce")).all():
                    issues.append(f"stage2 seed {s}: non-finite mean_band_width for "
                                   f"Center+DNNBand rows (unexpected)")
                other = df2[df2["method"] != "Center+DNNBand"]
                if not other[col].isna().all():
                    warnings.append(f"stage2 seed {s}: non-DNNBand rows have a non-NaN "
                                     f"mean_band_width value (unexpected but not fatal)")
                continue
            vals = pd.to_numeric(df2[col], errors="coerce")
            if not np.isfinite(vals).all():
                issues.append(f"stage2 seed {s}: {(~np.isfinite(vals)).sum()} "
                               f"non-finite values in column {col}")

        if (df2["cumulative_tc"] < 0).any():
            issues.append(f"stage2 seed {s}: negative cumulative_tc found")
        lo = df2.get("lower_boundary")
        up = df2.get("upper_boundary")
        if lo is not None and up is not None and (lo > up).any():
            issues.append(f"stage2 seed {s}: lower_boundary > upper_boundary found")

        diff = (df2["gross_terminal_wealth"] - df2["net_terminal_wealth"]
                - df2["cumulative_tc"]).abs()
        info.setdefault("gross_net_invariant_max_abs_diff", {})[s] = float(diff.max())

    info["known_gaps"] = [
        "Per-step (time-resolved) leverage trajectories are not persisted by "
        "eval_compare.py/stage2_eval.py, only path-averaged avg_gross_leverage. "
        "share_paths_avg_leverage_near_cap is therefore a coarse proxy (fraction "
        "of paths whose WHOLE-PATH AVERAGE leverage is >= threshold), not a true "
        "per-timestep cap-saturation rate.",
        "Separate lower/upper DNN-band boundary components are not persisted - "
        "only the combined mean_band_width. lower_boundary_avg/upper_boundary_avg "
        "columns in seed_level_results.csv are therefore NaN by construction.",
        "Per-timestep wealth trajectories are not persisted anywhere (only "
        "terminal/summary statistics and pre-rendered per-seed PNG plots) - a "
        "true 5-seed-aggregated mean+-1std wealth path over time cannot be "
        "reconstructed without re-running evaluation, which is out of scope for "
        "this read-only aggregation.",
        "avg_trade_size is not a native output column; reported as a derived "
        "proxy = mean_turnover / mean_num_trades per seed/method.",
        "no_trade_dwell_fraction (1 - num_trades/252) is only computed for the "
        "two daily-rebalancing-frequency methods (Center+DNNBand, "
        "CenterOnly_Daily); it is not meaningful for the fixed monthly-schedule "
        "methods (CenterOnly_Monthly, EW_Monthly) and is left NaN for those.",
    ]
    return issues, warnings, info


# ---------------------------------------------------------------------------
# Section 2 (first stage): per-seed statistics
# ---------------------------------------------------------------------------

def per_seed_stage1(base: Path) -> pd.DataFrame:
    rows = []
    for s in SEEDS:
        d = base / STAGE1_EVAL_REL[s]
        df = pd.read_csv(d / "eval_terminal_by_path.csv")
        df2s = pd.read_csv(d / "eval_terminal_by_path_stage2style.csv")  # has turnover
        for m in STAGE1_METHODS:
            sub = df[df["method"] == m].sort_values("path_id")
            tw = sub["terminal_wealth"].to_numpy()
            gl = sub["avg_gross_leverage"].to_numpy()
            cw = sub["avg_cash_weight"].to_numpy()
            turn_sub = df2s[df2s["method"] == m].sort_values("path_id")
            turnover = turn_sub["turnover"].to_numpy() if len(turn_sub) else np.full(len(tw), np.nan)

            rows.append({
                "stage": "stage1", "seed": s, "method": m, "n_eval_paths": len(tw),
                "mean_terminal_wealth": float(np.mean(tw)),
                "terminal_wealth_std_within_seed": float(np.std(tw, ddof=1)),
                "median_terminal_wealth": float(np.median(tw)),
                "p05_terminal_wealth": float(np.percentile(tw, 5)),
                "target_achievement_probability": float(np.mean(tw >= Z_TARGET)),
                "mean_shortfall_vs_target": float(np.mean(np.maximum(Z_TARGET - tw, 0.0))),
                "target_msd": float(np.mean((tw - Z_TARGET) ** 2)),
                "mean_gross_terminal_wealth": float(np.mean(tw)),  # no tcost in Stage1 eval
                "mean_net_terminal_wealth": float(np.mean(tw)),
                "mean_cumulative_tc": 0.0,
                "mean_gross_minus_net_gap": 0.0,
                "mean_turnover": float(np.nanmean(turnover)) if len(turnover) else np.nan,
                "mean_num_trades": np.nan,
                "avg_trade_size_proxy": np.nan,
                "mean_band_width": np.nan,
                "lower_boundary_avg": np.nan,
                "upper_boundary_avg": np.nan,
                "no_trade_dwell_fraction": np.nan,
                "mean_avg_gross_leverage": float(np.mean(gl)),
                "mean_avg_cash_weight": float(np.mean(cw)),
                "share_paths_avg_leverage_near_cap": float(np.mean(gl >= LEV_NEAR_CAP_THRESHOLD)),
                "runtime_seconds": np.nan,  # filled in by caller from stdout.log
                "runtime_source": "stage1_stdout_log_elapsed_s",
            })
    return pd.DataFrame(rows)


def per_seed_stage2(base: Path) -> pd.DataFrame:
    rows = []
    for s in SEEDS:
        d = base / STAGE2_EVAL_REL[s]
        df = pd.read_csv(d / "eval_terminal_by_path.csv")
        for m in STAGE2_METHODS:
            sub = df[df["method"] == m].sort_values("path_id")
            tw = sub["terminal_wealth"].to_numpy()
            net = sub["net_terminal_wealth"].to_numpy()
            gross = sub["gross_terminal_wealth"].to_numpy()
            tc = sub["cumulative_tc"].to_numpy()
            gap = sub["gross_minus_net_gap"].to_numpy()
            trades = sub["num_trades"].to_numpy()
            band = sub["mean_band_width"].to_numpy()  # NaN for non-DNNBand, structural
            gl = sub["gross_lev"].to_numpy()
            cw = sub["cash_w"].to_numpy()
            turnover = sub["turnover"].to_numpy()

            mean_turnover = float(np.mean(turnover))
            mean_trades = float(np.mean(trades))
            avg_trade_size = (mean_turnover / mean_trades) if mean_trades > 0 else np.nan
            dwell = (1.0 - mean_trades / TRADING_DAYS_PER_YEAR) if m in STAGE2_DAILY_METHODS else np.nan

            rows.append({
                "stage": "stage2", "seed": s, "method": m, "n_eval_paths": len(net),
                "mean_terminal_wealth": float(np.mean(net)),  # net = what investor keeps
                "terminal_wealth_std_within_seed": float(np.std(net, ddof=1)),
                "median_terminal_wealth": float(np.median(net)),
                "p05_terminal_wealth": float(np.percentile(net, 5)),
                "target_achievement_probability": float(np.mean(net >= Z_TARGET)),
                "mean_shortfall_vs_target": float(np.mean(np.maximum(Z_TARGET - net, 0.0))),
                "target_msd": float(np.mean((net - Z_TARGET) ** 2)),
                "mean_gross_terminal_wealth": float(np.mean(gross)),
                "mean_net_terminal_wealth": float(np.mean(net)),
                "mean_cumulative_tc": float(np.mean(tc)),
                "mean_gross_minus_net_gap": float(np.mean(gap)),
                "mean_turnover": mean_turnover,
                "mean_num_trades": mean_trades,
                "avg_trade_size_proxy": float(avg_trade_size) if np.isfinite(avg_trade_size) else np.nan,
                "mean_band_width": float(np.nanmean(band)) if np.isfinite(band).any() else np.nan,
                "lower_boundary_avg": np.nan,
                "upper_boundary_avg": np.nan,
                "no_trade_dwell_fraction": float(dwell) if dwell is not np.nan else np.nan,
                "mean_avg_gross_leverage": float(np.mean(gl)),
                "mean_avg_cash_weight": float(np.mean(cw)),
                "share_paths_avg_leverage_near_cap": float(np.mean(gl >= LEV_NEAR_CAP_THRESHOLD)),
                "runtime_seconds": np.nan,  # filled in by caller
                "runtime_source": "supervisor_log_launch_to_complete (seeds 1-4) / "
                                   "file_mtime_estimate (seed 0)",
            })
    return pd.DataFrame(rows)


def stage1_runtime_seconds(base: Path, seed: int) -> float | None:
    log = base / f"stage1/seed_{seed}/stdout.log"
    if not log.exists():
        return None
    last = None
    for line in log.read_text(errors="replace").splitlines():
        if "elapsed_s=" in line and "iter=9000/9000" in line:
            last = line
    if last is None:
        return None
    for tok in last.split():
        if tok.startswith("elapsed_s="):
            return float(tok.split("=", 1)[1])
    return None


def stage2_runtime_seconds(base: Path, seed: int, sup_log_text: str) -> tuple[float | None, str]:
    import re
    from datetime import datetime, timezone
    launch_re = re.compile(
        rf"\[([\d\-T:.+]+)\] Launched seed {seed} Stage2 session=")
    done_re = re.compile(
        rf"\[([\d\-T:.+]+)\] Seed {seed} Stage2 completed\.")
    launch_m = launch_re.search(sup_log_text)
    done_m = done_re.search(sup_log_text)
    if launch_m and done_m:
        t0 = datetime.fromisoformat(launch_m.group(1))
        t1 = datetime.fromisoformat(done_m.group(1))
        return (t1 - t0).total_seconds(), "supervisor_log_launch_to_complete"

    # Seed 1 special case: its real launch crashed the supervisor (shasum
    # quoting bug, fixed commit 82ba0a3) BEFORE the "Launched seed 1 Stage2"
    # log line was reached, so that line was never written; the job was later
    # adopted from a live tmux session rather than freshly launched. Use the
    # supervisor-restart timestamp immediately preceding the confirmed-running
    # adopted job (seconds-scale proxy for the true launch time) paired with
    # the real "Seed 1 Stage2 completed" timestamp.
    if seed == 1 and done_m is not None:
        t0 = datetime.fromisoformat("2026-07-26T23:06:37.231875+00:00")
        t1 = datetime.fromisoformat(done_m.group(1))
        return (t1 - t0).total_seconds(), "approximate_supervisor_restart_to_complete_seed1_special_case"

    # Fallback (seed 0: launched entirely outside the supervisor by the human
    # coordinator before this pipeline existed, never logged there at all).
    # metrics.csv birth-to-mtime is used as a rough proxy; NOTE this directory
    # lives under a cloud-synced ("My Drive") folder, whose filesystem
    # timestamps (especially birth/ctime) can be perturbed by sync activity
    # and are therefore explicitly flagged as approximate, not authoritative.
    metrics = base / f"stage2/seed_{seed}/metrics.csv"
    if metrics.exists():
        st = metrics.stat()
        birth = getattr(st, "st_birthtime", None)
        if birth is not None and st.st_mtime > birth:
            return (st.st_mtime - birth), "file_birthtime_to_mtime_estimate_approximate_cloud_synced_dir"
    return None, "unavailable"


def train_diagnostics(base: Path) -> pd.DataFrame:
    """avg_abs_action etc from the FINAL training iteration of metrics.csv -
    a training diagnostic, kept in a separate table from evaluation metrics."""
    rows = []
    for s in SEEDS:
        df1 = pd.read_csv(base / f"stage1/seed_{s}/metrics.csv")
        last1 = df1.iloc[-1]
        df2 = pd.read_csv(base / f"stage2/seed_{s}/metrics.csv")
        last2 = df2.iloc[-1]
        rows.append({
            "seed": s, "stage": "stage1", "final_iter": int(last1["iter"]),
            "avg_abs_action": float(last1.get("avg_abs_action", np.nan)),
            "max_gross_leverage": float(last1.get("max_gross_leverage", np.nan)),
            "min_cash_weight": float(last1.get("min_cash_weight", np.nan)),
            "avg_gross_leverage": float(last1.get("avg_gross_leverage", np.nan)),
            "avg_cash_weight": float(last1.get("avg_cash_weight", np.nan)),
        })
        rows.append({
            "seed": s, "stage": "stage2", "final_iter": int(last2["iter"]),
            "avg_abs_action": np.nan,  # stage2 metrics.csv has no avg_abs_action column
            "max_gross_leverage": np.nan,
            "min_cash_weight": np.nan,
            "avg_gross_leverage": float(last2.get("avg_gross_lev", np.nan)),
            "avg_cash_weight": np.nan,
        })
    return pd.DataFrame(rows)


def utility_consistent_metrics(base: Path) -> pd.DataFrame:
    """Mean terminal utility / log-utility certainty equivalent, reconstructed
    from already-computed episode-level net_terminal_wealth (Stage 2) and
    terminal_wealth (Stage 1, RL) using the exact positivity floor from the
    training implementation (poemv_rs/stage2_direct_boundary.py _utility_torch,
    eps=1e-10) and log utility (the utility_kind actually used for Phase 1A).
    No re-simulation - purely a derived statistic over existing episode-level
    data (item 7)."""
    rows = []
    for s in SEEDS:
        df1 = pd.read_csv(base / STAGE1_EVAL_REL[s] / "eval_terminal_by_path.csv")
        sub1 = df1[df1["method"] == "RL"]
        tw1 = sub1["terminal_wealth"].to_numpy()
        u1 = np.log(np.maximum(tw1, POSITIVITY_FLOOR))
        rows.append({
            "stage": "stage1", "seed": s, "method": "RL", "n_eval_paths": len(tw1),
            "mean_terminal_utility": float(np.mean(u1)),
            "log_utility_certainty_equivalent": float(np.exp(np.mean(u1))),
        })

        df2 = pd.read_csv(base / STAGE2_EVAL_REL[s] / "eval_terminal_by_path.csv")
        for m in STAGE2_METHODS:
            sub2 = df2[df2["method"] == m]
            tw2 = sub2["net_terminal_wealth"].to_numpy()
            u2 = np.log(np.maximum(tw2, POSITIVITY_FLOOR))
            rows.append({
                "stage": "stage2", "seed": s, "method": m, "n_eval_paths": len(tw2),
                "mean_terminal_utility": float(np.mean(u2)),
                "log_utility_certainty_equivalent": float(np.exp(np.mean(u2))),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 2 (second stage): cross-seed statistics
# ---------------------------------------------------------------------------

METRIC_COLS = [
    "mean_terminal_wealth", "terminal_wealth_std_within_seed", "median_terminal_wealth",
    "p05_terminal_wealth", "target_achievement_probability", "mean_shortfall_vs_target",
    "target_msd", "mean_gross_terminal_wealth", "mean_net_terminal_wealth",
    "mean_cumulative_tc", "mean_gross_minus_net_gap", "mean_turnover", "mean_num_trades",
    "avg_trade_size_proxy", "mean_band_width", "lower_boundary_avg", "upper_boundary_avg",
    "no_trade_dwell_fraction", "mean_avg_gross_leverage", "mean_avg_cash_weight",
    "share_paths_avg_leverage_near_cap", "runtime_seconds",
    "mean_terminal_utility", "log_utility_certainty_equivalent",
]


def cross_seed_stats(seed_level: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (stage, method), grp in seed_level.groupby(["stage", "method"]):
        for col in METRIC_COLS:
            vals = grp[col].dropna().to_numpy()
            n = len(vals)
            if n == 0:
                rows.append({"stage": stage, "method": method, "metric": col,
                             "mean": np.nan, "std_across_training_seeds": np.nan,
                             "se_across_training_seeds": np.nan,
                             "ci95_low": np.nan, "ci95_high": np.nan,
                             "min": np.nan, "max": np.nan, "n_valid_training_seeds": 0})
                continue
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if n > 1 else np.nan
            se = float(std / np.sqrt(n)) if n > 1 else np.nan
            if n > 1 and np.isfinite(se):
                tcrit = float(stats.t.ppf(0.975, n - 1))
                ci_lo, ci_hi = mean - tcrit * se, mean + tcrit * se
            else:
                ci_lo, ci_hi = np.nan, np.nan
            rows.append({
                "stage": stage, "method": method, "metric": col,
                "mean": mean, "std_across_training_seeds": std,
                "se_across_training_seeds": se,
                "ci95_low": ci_lo, "ci95_high": ci_hi,
                "min": float(np.min(vals)), "max": float(np.max(vals)),
                "n_valid_training_seeds": n,
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Section 4: paired comparisons
# ---------------------------------------------------------------------------

def load_stage1_path_wealth(base: Path, seed: int, method: str) -> pd.Series:
    df = pd.read_csv(base / STAGE1_EVAL_REL[seed] / "eval_terminal_by_path.csv")
    sub = df[df["method"] == method].sort_values("path_id").set_index("path_id")
    return sub["terminal_wealth"]


def load_stage2_path_wealth(base: Path, seed: int, method: str, col: str) -> pd.Series:
    df = pd.read_csv(base / STAGE2_EVAL_REL[seed] / "eval_terminal_by_path.csv")
    sub = df[df["method"] == method].sort_values("path_id").set_index("path_id")
    return sub[col]


def paired_comparisons(base: Path) -> pd.DataFrame:
    comparisons = []  # (name, per_seed_diff_fn, note, descriptive_only)

    def stage2_minus_stage1(seed):
        s2 = load_stage2_path_wealth(base, seed, "Center+DNNBand", "net_terminal_wealth")
        s1 = load_stage1_path_wealth(base, seed, "RL")
        return (s2 - s1)

    comparisons.append((
        "stage2_net_minus_stage1_terminal", stage2_minus_stage1,
        "Stage2 Center+DNNBand net terminal wealth minus Stage1 RL terminal wealth "
        "(same path_id, same seed). Stage1 has no transaction costs; Stage2 is "
        "transaction-cost-aware with a different (tcost-aware) objective - "
        "descriptive comparison only, not a claim that Stage2 'outperforms' Stage1 "
        "on the same objective. Path correspondence across stages is inferred from "
        "matching eval_config (see DATA_QUALITY_REPORT), not independently proven.",
        True,
    ))

    for m in STAGE2_METHODS:
        def f(seed, m=m):
            gross = load_stage2_path_wealth(base, seed, m, "gross_terminal_wealth")
            net = load_stage2_path_wealth(base, seed, m, "net_terminal_wealth")
            return gross - net
        comparisons.append((
            f"stage2_gross_minus_net__{m}", f,
            f"Stage2 {m}: gross terminal wealth minus net terminal wealth "
            f"(= cumulative transaction cost paid, by construction).",
            False,
        ))

    import itertools
    for m1, m2 in itertools.combinations(STAGE2_METHODS, 2):
        def f(seed, m1=m1, m2=m2):
            a = load_stage2_path_wealth(base, seed, m1, "net_terminal_wealth")
            b = load_stage2_path_wealth(base, seed, m2, "net_terminal_wealth")
            return a - b
        comparisons.append((
            f"stage2_net_terminal__{m1}_minus_{m2}", f,
            f"Stage2 net terminal wealth: {m1} minus {m2} (same path_id, same seed). "
            f"Methods differ in rebalancing rule/objective - descriptive comparison only.",
            True,
        ))

    rows = []
    for name, fn, note, descriptive_only in comparisons:
        per_seed_means = []
        n_paths = None
        for s in SEEDS:
            diff = fn(s)
            n_paths = len(diff)
            per_seed_means.append(float(diff.mean()))
        arr = np.array(per_seed_means)
        n = len(arr)
        mean = float(np.mean(arr))
        std = float(np.std(arr, ddof=1))
        se = float(std / np.sqrt(n))
        tcrit = float(stats.t.ppf(0.975, n - 1))
        rows.append({
            "comparison": name,
            "mean_paired_difference": mean,
            "std_across_training_seeds": std,
            "se_across_training_seeds": se,
            "ci95_low": mean - tcrit * se,
            "ci95_high": mean + tcrit * se,
            "n_training_seeds": n,
            "n_eval_paths_per_seed": n_paths,
            "descriptive_only": descriptive_only,
            "note": note,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# episode_level_combined
# ---------------------------------------------------------------------------

def episode_level_combined(base: Path) -> pd.DataFrame:
    frames = []
    for s in SEEDS:
        df1 = pd.read_csv(base / STAGE1_EVAL_REL[s] / "eval_terminal_by_path.csv")
        df1s = pd.read_csv(base / STAGE1_EVAL_REL[s] / "eval_terminal_by_path_stage2style.csv")
        merged = df1.merge(
            df1s[["method", "path_id", "turnover"]], on=["method", "path_id"], how="left")
        merged["stage"] = "stage1"
        merged["seed"] = s
        merged["net_terminal_wealth"] = merged["terminal_wealth"]
        merged["gross_terminal_wealth"] = merged["terminal_wealth"]
        merged["cumulative_tc"] = 0.0
        merged["gross_minus_net_gap"] = 0.0
        merged["num_trades"] = np.nan
        merged["mean_band_width"] = np.nan
        merged = merged.rename(columns={"avg_gross_leverage": "gross_lev",
                                         "avg_cash_weight": "cash_w"})
        frames.append(merged[["stage", "seed", "method", "path_id", "terminal_wealth",
                               "net_terminal_wealth", "gross_terminal_wealth",
                               "cumulative_tc", "gross_minus_net_gap", "num_trades",
                               "mean_band_width", "gross_lev", "cash_w", "turnover",
                               "path_sharpe_ann"]])

        df2 = pd.read_csv(base / STAGE2_EVAL_REL[s] / "eval_terminal_by_path.csv")
        df2 = df2.copy()
        df2["stage"] = "stage2"
        df2["seed"] = s
        df2["path_sharpe_ann"] = np.nan
        frames.append(df2[["stage", "seed", "method", "path_id", "terminal_wealth",
                            "net_terminal_wealth", "gross_terminal_wealth",
                            "cumulative_tc", "gross_minus_net_gap", "num_trades",
                            "mean_band_width", "gross_lev", "cash_w", "turnover",
                            "path_sharpe_ann"]])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# run_manifest
# ---------------------------------------------------------------------------

def build_run_manifest(base: Path, git_commit: str) -> pd.DataFrame:
    rows = []
    for s in SEEDS:
        s1_ckpt = base / f"stage1/seed_{s}/checkpoint.pt"
        rows.append({
            "seed": s, "stage": "stage1", "method": "RL_train", "checkpoint_path": str(s1_ckpt),
            "checkpoint_hash": sha256_file(s1_ckpt) if s1_ckpt.exists() else "",
            "evaluation_directory": str(base / STAGE1_EVAL_REL[s]),
            "config_path": str(base / STAGE1_EVAL_REL[s] / "eval_config.json"),
            "git_commit": git_commit, "evaluation_episode_count": EVAL_N_PATHS,
            "status": "completed", "exclusion_reason": "", "resumed": "",
            "known_historical_bug": "",
        })
        if s in STAGE1_EVAL_EXCLUDED_REL:
            rows.append({
                "seed": s, "stage": "stage1", "method": "RL_train",
                "checkpoint_path": str(s1_ckpt),
                "checkpoint_hash": sha256_file(s1_ckpt) if s1_ckpt.exists() else "",
                "evaluation_directory": str(base / STAGE1_EVAL_EXCLUDED_REL[s]),
                "config_path": "", "git_commit": git_commit,
                "evaluation_episode_count": 0, "status": "excluded",
                "exclusion_reason": STAGE1_EXCLUSION_REASON, "resumed": "no",
                "known_historical_bug": "unquoted-path shell-splitting in "
                                        "run_eval_stage1() (fixed commit 04c4f2e)",
            })
        s2_ckpt = base / f"stage2/seed_{s}/best_checkpoint.pt"
        rows.append({
            "seed": s, "stage": "stage2", "method": "Center+DNNBand_train",
            "checkpoint_path": str(s2_ckpt),
            "checkpoint_hash": sha256_file(s2_ckpt) if s2_ckpt.exists() else "",
            "evaluation_directory": str(base / STAGE2_EVAL_REL[s]),
            "config_path": str(base / STAGE2_EVAL_REL[s] / "eval_config.json"),
            "git_commit": git_commit, "evaluation_episode_count": EVAL_N_PATHS,
            "status": "completed", "exclusion_reason": "",
            "resumed": "yes" if s == 1 else "no",  # seed1 stage2 job was adopted after a supervisor crash, not resumed from a mid-training checkpoint - training itself was never interrupted
            "known_historical_bug": "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ICA report tables
# ---------------------------------------------------------------------------

def pick(agg: pd.DataFrame, stage: str, method: str, metric: str) -> dict:
    row = agg[(agg.stage == stage) & (agg.method == method) & (agg.metric == metric)]
    if row.empty:
        return {"mean": np.nan, "ci95_low": np.nan, "ci95_high": np.nan}
    r = row.iloc[0]
    return {"mean": r["mean"], "ci95_low": r["ci95_low"], "ci95_high": r["ci95_high"]}


def make_table1(agg: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for m in STAGE1_METHODS:
        rows.append({
            "method": m,
            "mean_terminal_wealth": pick(agg, "stage1", m, "mean_terminal_wealth")["mean"],
            "terminal_wealth_std_within_seed": pick(agg, "stage1", m, "terminal_wealth_std_within_seed")["mean"],
            "target_achievement_probability": pick(agg, "stage1", m, "target_achievement_probability")["mean"],
            "target_msd": pick(agg, "stage1", m, "target_msd")["mean"],
            "mean_shortfall_vs_target": pick(agg, "stage1", m, "mean_shortfall_vs_target")["mean"],
            "mean_terminal_wealth_ci95_low": pick(agg, "stage1", m, "mean_terminal_wealth")["ci95_low"],
            "mean_terminal_wealth_ci95_high": pick(agg, "stage1", m, "mean_terminal_wealth")["ci95_high"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tables" / "table1_stage1_robustness.csv", index=False)
    df.to_latex(out_dir / "tables" / "table1_stage1_robustness.tex", index=False, float_format="%.4f")
    return df


def make_table2(agg: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    for m in STAGE2_METHODS:
        rows.append({
            "method": m,
            "net_terminal_wealth": pick(agg, "stage2", m, "mean_net_terminal_wealth")["mean"],
            "gross_terminal_wealth": pick(agg, "stage2", m, "mean_gross_terminal_wealth")["mean"],
            "cumulative_tc": pick(agg, "stage2", m, "mean_cumulative_tc")["mean"],
            "turnover": pick(agg, "stage2", m, "mean_turnover")["mean"],
            "num_trades": pick(agg, "stage2", m, "mean_num_trades")["mean"],
            "band_width": pick(agg, "stage2", m, "mean_band_width")["mean"],
            "target_achievement_probability": pick(agg, "stage2", m, "target_achievement_probability")["mean"],
            "net_terminal_wealth_ci95_low": pick(agg, "stage2", m, "mean_net_terminal_wealth")["ci95_low"],
            "net_terminal_wealth_ci95_high": pick(agg, "stage2", m, "mean_net_terminal_wealth")["ci95_high"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tables" / "table2_stage2_cost_aware_performance.csv", index=False)
    df.to_latex(out_dir / "tables" / "table2_stage2_cost_aware_performance.tex", index=False, float_format="%.4f")
    return df


def make_table3(paired: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    df = paired[["comparison", "mean_paired_difference", "std_across_training_seeds",
                 "se_across_training_seeds", "ci95_low", "ci95_high",
                 "n_training_seeds", "n_eval_paths_per_seed"]].copy()
    df.to_csv(out_dir / "tables" / "table3_paired_comparisons.csv", index=False)
    df.to_latex(out_dir / "tables" / "table3_paired_comparisons.tex", index=False, float_format="%.5f")
    return df


# Item 8: full 5-seed intervals for metrics not already fully tabulated in
# Tables 1-2 (which report only mean_terminal_wealth/net_terminal_wealth
# intervals in the main text). One row per (stage, method); columns are
# "<metric> [ci_low, ci_high]" strings for compact display, plus the raw
# numeric columns for reproducibility.
FULL_INTERVAL_METRICS = [
    "target_achievement_probability", "target_msd", "mean_shortfall_vs_target",
    "mean_cumulative_tc", "mean_turnover", "mean_num_trades", "mean_band_width",
    "mean_terminal_utility", "log_utility_certainty_equivalent",
]


def make_table4_full_intervals(agg: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows = []
    combos = [("stage1", m) for m in STAGE1_METHODS] + [("stage2", m) for m in STAGE2_METHODS]
    for stage, method in combos:
        row = {"stage": stage, "method": method}
        for metric in FULL_INTERVAL_METRICS:
            r = pick(agg, stage, method, metric)
            row[f"{metric}_mean"] = r["mean"]
            row[f"{metric}_ci95_low"] = r["ci95_low"]
            row[f"{metric}_ci95_high"] = r["ci95_high"]
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "tables" / "table4_full_metric_intervals.csv", index=False)

    # Compact display version for the LaTeX appendix: "mean [lo, hi]" strings.
    disp_rows = []
    for stage, method in combos:
        r = {"Stage": "1" if stage == "stage1" else "2", "Method": method}
        for metric in FULL_INTERVAL_METRICS:
            m = pick(agg, stage, method, metric)
            if not np.isfinite(m["mean"]):
                r[metric] = "--"
            elif not np.isfinite(m["ci95_low"]):
                r[metric] = f"{m['mean']:.4g} (seed-invariant)"
            else:
                r[metric] = f"{m['mean']:.4g} [{m['ci95_low']:.4g}, {m['ci95_high']:.4g}]"
        disp_rows.append(r)
    disp_df = pd.DataFrame(disp_rows)
    disp_df.to_latex(out_dir / "tables" / "table4_full_metric_intervals.tex", index=False)
    return df


# ---------------------------------------------------------------------------
# ICA report figures
# ---------------------------------------------------------------------------

def fig1_mean_vs_std(seed_level: pd.DataFrame, agg: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    specs = [("stage1", "RL", "Stage 1 (RL)"), ("stage2", "Center+DNNBand", "Stage 2 (Center+DNNBand)")]
    plot_rows = []
    for ax, (stage, method, title) in zip(axes, specs):
        sub = seed_level[(seed_level.stage == stage) & (seed_level.method == method)]
        ax.scatter(sub["mean_terminal_wealth"], sub["terminal_wealth_std_within_seed"],
                   color="tab:blue", label="individual training seeds", zorder=3)
        for _, r in sub.iterrows():
            ax.annotate(f"seed {int(r['seed'])}", (r["mean_terminal_wealth"], r["terminal_wealth_std_within_seed"]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")
            plot_rows.append({"stage": stage, "method": method, "seed": int(r["seed"]),
                              "mean_terminal_wealth": r["mean_terminal_wealth"],
                              "terminal_wealth_std_within_seed": r["terminal_wealth_std_within_seed"]})
        mx = pick(agg, stage, method, "mean_terminal_wealth")
        my = pick(agg, stage, method, "terminal_wealth_std_within_seed")
        xerr = (mx["mean"] - mx["ci95_low"]) if np.isfinite(mx["ci95_low"]) else 0
        yerr = (my["mean"] - my["ci95_low"]) if np.isfinite(my["ci95_low"]) else 0
        ax.errorbar([mx["mean"]], [my["mean"]], xerr=[[xerr], [xerr]], yerr=[[yerr], [yerr]],
                   fmt="D", color="tab:red", capsize=4,
                   label="5-seed mean\n(95% Student-$t$ training-seed interval)", zorder=4)
        ax.set_xlabel("Mean terminal wealth")
        ax.set_ylabel("Terminal wealth std (within-seed, evaluation-path risk)")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)
    fig.suptitle("Mean terminal wealth vs. terminal wealth dispersion, across 5 training seeds")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "figures" / "fig1_mean_vs_std.pdf")
    fig.savefig(out_dir / "figures" / "fig1_mean_vs_std.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(plot_rows).to_csv(out_dir / "figures" / "fig1_mean_vs_std_data.csv", index=False)


def fig2_terminal_wealth_distribution(combined: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    s1 = combined[combined.stage == "stage1"]
    data1 = [s1[s1.method == m]["terminal_wealth"].to_numpy() for m in STAGE1_METHODS]
    axes[0].violinplot(data1, showmeans=True, showextrema=False)
    axes[0].axhline(Z_TARGET, color="tab:red", linestyle="--", linewidth=1, label=f"target $z={Z_TARGET}$")
    axes[0].set_xticks(range(1, len(STAGE1_METHODS) + 1), [STAGE1_METHOD_SHORT[m] for m in STAGE1_METHODS])
    axes[0].set_ylabel("Terminal wealth")
    axes[0].set_title("Stage 1: pooled across 5 seeds x 1000 paths")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[0].grid(alpha=0.3)

    s2 = combined[combined.stage == "stage2"]
    data2 = [s2[s2.method == m]["net_terminal_wealth"].to_numpy() for m in STAGE2_METHODS]
    axes[1].violinplot(data2, showmeans=True, showextrema=False)
    axes[1].axhline(Z_TARGET, color="tab:red", linestyle="--", linewidth=1, label=f"target $z={Z_TARGET}$")
    axes[1].set_xticks(range(1, len(STAGE2_METHODS) + 1), [STAGE2_METHOD_SHORT[m] for m in STAGE2_METHODS],
                       rotation=20, ha="right")
    axes[1].set_ylabel("Net terminal wealth")
    axes[1].set_title("Stage 2: pooled across 5 seeds x 1000 paths")
    axes[1].legend(fontsize=8, loc="upper left")
    axes[1].grid(alpha=0.3)
    fig.suptitle("Terminal wealth distribution by method (violin, mean shown, target level marked).\n"
                "Descriptive visualization of pooled paths, not treated as 5000 independent observations for inference.",
                fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "figures" / "fig2_terminal_wealth_distribution.pdf")
    fig.savefig(out_dir / "figures" / "fig2_terminal_wealth_distribution.png", dpi=200)
    plt.close(fig)
    combined[["stage", "seed", "method", "path_id", "terminal_wealth", "net_terminal_wealth"]].to_csv(
        out_dir / "figures" / "fig2_terminal_wealth_distribution_data.csv", index=False)


def fig3_note(out_dir: Path):
    note = (
        "Figure 3 (average wealth path over time, mean +/- 1 std band, aggregated "
        "across 5 seeds and all evaluation paths) could NOT be produced.\n\n"
        "Reason: eval_compare.py / stage2_eval.py persist only terminal/summary "
        "statistics per path (final wealth, leverage, turnover, etc.) - the "
        "per-timestep wealth trajectory arrays used to render each individual "
        "seed's own avg_wealth.png at evaluation time were never saved to disk "
        "(no .npy/.npz/CSV with a time axis exists in any eval_stage1, "
        "eval_stage1_postfix, or eval_stage2 directory). Reconstructing this "
        "figure would require re-running evaluation with new trajectory-logging "
        "code, which is out of scope for this read-only, no-new-simulation "
        "aggregation pass.\n\n"
        "Closest available artifacts (per-seed only, NOT a cross-seed aggregate):\n"
    )
    base = out_dir.parent
    for s in SEEDS:
        d = STAGE1_EVAL_REL[s]
        note += f"  - {d}/avg_wealth.png (seed {s}, Stage 1)\n"
    for s in SEEDS:
        note += f"  - eval_stage2/seed_{s}/avg_wealth.png (seed {s}, Stage 2)\n"
    note += (
        "\nRecommendation before Phase 2: add trajectory logging (e.g. save "
        "wealth[t] per path as .npy) to eval_compare.py/stage2_eval.py if this "
        "figure is required for the ICA report.\n"
    )
    (out_dir / "figures" / "fig3_average_wealth_path_NOTE.txt").write_text(note)


def fig4_cost_turnover(agg: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    metrics = [("mean_cumulative_tc", "Cumulative transaction cost", axes[0, 0]),
               ("mean_turnover", "Turnover", axes[0, 1]),
               ("mean_num_trades", "Number of trades", axes[1, 0]),
               ("mean_band_width", "Band width (DNNBand only)", axes[1, 1])]
    rows = []
    for metric, title, ax in metrics:
        means, errs = [], []
        for m in STAGE2_METHODS:
            r = pick(agg, "stage2", m, metric)
            means.append(r["mean"])
            err = (r["mean"] - r["ci95_low"]) if np.isfinite(r["ci95_low"]) else 0
            errs.append(err)
            rows.append({"method": m, "metric": metric, "mean": r["mean"],
                        "ci95_low": r["ci95_low"], "ci95_high": r["ci95_high"]})
        x = np.arange(len(STAGE2_METHODS))
        if metric == "mean_band_width":
            # Band width is only a defined quantity for Center+DNNBand; the
            # comparator policies contain no no-trade band, so their entries
            # are not-applicable rather than zero. Draw only the DNNBand bar
            # and mark the other positions explicitly as "N/A" text (not a
            # zero-height bar), without altering any underlying values.
            dnn_idx = STAGE2_METHODS.index("Center+DNNBand")
            bar_x = [x[dnn_idx]]
            bar_mean = [means[dnn_idx]]
            bar_err = [errs[dnn_idx]]
            ax.bar(bar_x, bar_mean, yerr=bar_err, capsize=4, color="tab:blue", alpha=0.8)
            y_na = 0.12 * max(means[dnn_idx], 1e-9)
            for i, m in enumerate(STAGE2_METHODS):
                if m == "Center+DNNBand":
                    continue
                ax.text(x[i], y_na, "N/A", ha="center", va="bottom",
                        fontsize=9, color="#333333", weight="medium")
            ax.set_ylim(bottom=0)
        else:
            ax.bar(x, means, yerr=errs, capsize=4, color="tab:blue", alpha=0.8)
        ax.set_xticks(x, [STAGE2_METHOD_SHORT[m] for m in STAGE2_METHODS], rotation=20, ha="right", fontsize=8)
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Stage 2 cost and turnover comparison\n(bars: 5-seed mean; error bars: 95% training-seed interval)",
                 fontsize=12)
    fig.tight_layout(rect=[0.02, 0, 0.98, 0.90])
    fig.savefig(out_dir / "figures" / "fig4_cost_turnover.pdf")
    fig.savefig(out_dir / "figures" / "fig4_cost_turnover.png", dpi=200)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(out_dir / "figures" / "fig4_cost_turnover_data.csv", index=False)


def fig5_training_seed_variability(seed_level: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    specs = [("stage1", "RL", "Stage 1 (RL): mean terminal wealth by seed"),
             ("stage2", "Center+DNNBand", "Stage 2 (Center+DNNBand): mean terminal wealth by seed")]
    for ax, (stage, method, title) in zip(axes, specs):
        sub = seed_level[(seed_level.stage == stage) & (seed_level.method == method)].sort_values("seed")
        ax.bar(sub["seed"].astype(str), sub["mean_terminal_wealth"], color="tab:blue", alpha=0.8)
        ax.axhline(sub["mean_terminal_wealth"].mean(), color="tab:red", linestyle="--",
                  label="5-seed mean")
        ax.set_xlabel("Training seed")
        ax.set_ylabel("Mean terminal wealth")
        ax.set_title(title, fontsize=9)
        ax.set_ylim(sub["mean_terminal_wealth"].min() * 0.95, sub["mean_terminal_wealth"].max() * 1.05)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, axis="y")
    fig.suptitle("Mean terminal wealth by training seed")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_dir / "figures" / "fig5_training_seed_variability.pdf")
    fig.savefig(out_dir / "figures" / "fig5_training_seed_variability.png", dpi=200)
    plt.close(fig)
    seed_level[(seed_level.stage.isin(["stage1", "stage2"]))
              & (seed_level.method.isin(["RL", "Center+DNNBand"]))][
        ["stage", "seed", "method", "mean_terminal_wealth"]].to_csv(
        out_dir / "figures" / "fig5_training_seed_variability_data.csv", index=False)


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------

def write_data_quality_report(out: Path, base: Path, warn: list[str], info: dict,
                              seed_level: pd.DataFrame) -> None:
    lines = []
    lines.append("# Phase 1A Data Quality Report\n")
    lines.append(f"Generated by `aggregate_phase1a.py` from `{base}`.\n")
    lines.append("All blocking validation checks in `validate()` PASSED "
                 "(seeds present, eval config consistency, T/a_max, estimated "
                 "parameters identical across seeds, checkpoint-seed correspondence, "
                 "no duplicate/missing paths, no unexpected NaN/Inf, no negative "
                 "transaction cost). See `aggregation_config.json` for the exact "
                 "config values checked.\n")

    lines.append("## Common evaluation paths verification\n")
    lines.append(f"- {info.get('stage1_common_paths_verification')}\n")
    lines.append(f"- {info.get('stage2_common_paths_verification')}\n")

    lines.append("\n## Gross-net invariant (Stage 2)\n")
    lines.append("gross_terminal_wealth - net_terminal_wealth - cumulative_tc, "
                 "max absolute error per seed:\n\n")
    lines.append("| seed | max abs diff |\n|---|---|\n")
    for s, v in sorted(info.get("gross_net_invariant_max_abs_diff", {}).items()):
        lines.append(f"| {s} | {v:.3e} |\n")

    lines.append("\n## Non-blocking warnings\n")
    for w in warn:
        lines.append(f"- {w}\n")

    lines.append("\n## Known gaps in the underlying evaluation outputs\n")
    for g in info.get("known_gaps", []):
        lines.append(f"- {g}\n")

    lines.append("\n## Excluded directories\n")
    for s, d in STAGE1_EVAL_EXCLUDED_REL.items():
        lines.append(f"- seed {s}: `{d}` - {STAGE1_EXCLUSION_REASON}\n")

    lines.append("\n## Runtime data provenance\n")
    rt = seed_level[seed_level.stage == "stage2"][["seed", "runtime_source"]].drop_duplicates()
    lines.append("| seed | stage2 runtime source |\n|---|---|\n")
    for _, r in rt.iterrows():
        lines.append(f"| {int(r['seed'])} | {r['runtime_source']} |\n")

    (out / "DATA_QUALITY_REPORT.md").write_text("".join(lines))


def write_results_summary(out: Path, agg: pd.DataFrame, paired: pd.DataFrame,
                          table1: pd.DataFrame, table2: pd.DataFrame, git_commit: str) -> None:
    lines = []
    lines.append("# Phase 1A Results Summary (5 training seeds, 0-4)\n\n")
    lines.append(f"git commit: `{git_commit}`\n\n")

    lines.append("## Stage 1 (RL policy, no transaction costs)\n\n")
    r = table1[table1.method == "RL"].iloc[0]
    lines.append(f"- Mean terminal wealth: {r['mean_terminal_wealth']:.4f} "
                 f"(95% CI across 5 seeds: [{r['mean_terminal_wealth_ci95_low']:.4f}, "
                 f"{r['mean_terminal_wealth_ci95_high']:.4f}])\n")
    lines.append(f"- Terminal wealth std (within-seed, evaluation-path risk): "
                 f"{r['terminal_wealth_std_within_seed']:.4f}\n")
    lines.append(f"- Target achievement probability: {r['target_achievement_probability']:.3f}\n")
    lines.append(f"- Target MSD: {r['target_msd']:.4f}\n")
    lines.append(f"- Mean shortfall vs target: {r['mean_shortfall_vs_target']:.4f}\n\n")

    lines.append("## Stage 2 (transaction-cost-aware, Center+DNNBand)\n\n")
    r2 = table2[table2.method == "Center+DNNBand"].iloc[0]
    lines.append(f"- Net terminal wealth: {r2['net_terminal_wealth']:.4f} "
                 f"(95% CI across 5 seeds: [{r2['net_terminal_wealth_ci95_low']:.4f}, "
                 f"{r2['net_terminal_wealth_ci95_high']:.4f}])\n")
    lines.append(f"- Gross terminal wealth: {r2['gross_terminal_wealth']:.4f}\n")
    lines.append(f"- Cumulative transaction cost: {r2['cumulative_tc']:.4f}\n")
    lines.append(f"- Turnover: {r2['turnover']:.4f}, trades: {r2['num_trades']:.1f}, "
                 f"band width: {r2['band_width']:.3f}\n")
    lines.append(f"- Target achievement probability: {r2['target_achievement_probability']:.3f}\n\n")

    lines.append("## Cross-seed variability (key finding)\n\n")
    s1_std = agg[(agg.stage == "stage1") & (agg.method == "RL")
                & (agg.metric == "mean_terminal_wealth")]["std_across_training_seeds"].iloc[0]
    s2_std = agg[(agg.stage == "stage2") & (agg.method == "Center+DNNBand")
                & (agg.metric == "mean_terminal_wealth")]["std_across_training_seeds"].iloc[0]
    lines.append(f"- Stage 1 RL mean-terminal-wealth std across training seeds: {s1_std:.6f}\n")
    lines.append(f"- Stage 2 Center+DNNBand mean-terminal-wealth std across training seeds: "
                 f"{s2_std:.6f} (~{s2_std / max(s1_std, 1e-12):.0f}x larger)\n\n")

    lines.append("## Paired comparisons (see paired_comparisons.csv / table3 for full list)\n\n")
    for _, p in paired.iterrows():
        sig = "excludes 0" if (p["ci95_low"] > 0 or p["ci95_high"] < 0) else "includes 0"
        lines.append(f"- `{p['comparison']}`: mean={p['mean_paired_difference']:.4f}, "
                     f"95% CI=[{p['ci95_low']:.4f}, {p['ci95_high']:.4f}] ({sig})"
                     f"{' [descriptive only]' if p['descriptive_only'] else ''}\n")

    (out / "PHASE1A_RESULTS_SUMMARY.md").write_text("".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", required=True, type=Path,
                     help="phase1a_endtoend_20260725 directory (input, read-only)")
    ap.add_argument("--out-dir", required=True, type=Path,
                     help="aggregate output directory (created if missing)")
    args = ap.parse_args()

    base: Path = args.base_dir.resolve()
    out: Path = args.out_dir.resolve()
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=str(base), capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        git_commit = "unknown"

    print("Validating inputs...", file=sys.stderr)
    issues, warn, info = validate(base)
    if issues:
        print("BLOCKING VALIDATION ISSUES - aborting aggregation:", file=sys.stderr)
        for i in issues:
            print(" -", i, file=sys.stderr)
        (out / "VALIDATION_FAILED.txt").write_text("\n".join(issues))
        sys.exit(1)

    print("Computing seed-level (first-stage) statistics...", file=sys.stderr)
    s1 = per_seed_stage1(base)
    s2 = per_seed_stage2(base)

    sup_log_text = (base / "supervisor/supervisor.log").read_text(errors="replace")
    for s in SEEDS:
        rt1 = stage1_runtime_seconds(base, s)
        s1.loc[s1["seed"] == s, "runtime_seconds"] = rt1
        rt2, src2 = stage2_runtime_seconds(base, s, sup_log_text)
        s2.loc[s2["seed"] == s, "runtime_seconds"] = rt2
        s2.loc[s2["seed"] == s, "runtime_source"] = src2

    print("Computing utility-consistent metrics from episode-level data...", file=sys.stderr)
    util = utility_consistent_metrics(base)
    util.to_csv(out / "utility_consistent_metrics.csv", index=False)

    seed_level = pd.concat([s1, s2], ignore_index=True)
    seed_level = seed_level.merge(
        util.drop(columns=["n_eval_paths"]), on=["stage", "seed", "method"], how="left")
    seed_level.to_csv(out / "seed_level_results.csv", index=False)

    print("Computing cross-seed (second-stage) statistics...", file=sys.stderr)
    agg = cross_seed_stats(seed_level)
    agg.to_csv(out / "aggregate_results.csv", index=False)

    print("Computing paired comparisons...", file=sys.stderr)
    paired = paired_comparisons(base)
    paired.to_csv(out / "paired_comparisons.csv", index=False)

    print("Building combined episode-level table...", file=sys.stderr)
    combined = episode_level_combined(base)
    combined.to_parquet(out / "episode_level_combined.parquet", index=False)

    print("Building training diagnostics table...", file=sys.stderr)
    diag = train_diagnostics(base)
    diag.to_csv(out / "training_diagnostics.csv", index=False)

    print("Building run manifest...", file=sys.stderr)
    manifest = build_run_manifest(base, git_commit)
    manifest.to_csv(out / "run_manifest.csv", index=False)

    print("Writing aggregation_config.json...", file=sys.stderr)
    config = {
        "base_dir": str(base), "out_dir": str(out), "git_commit": git_commit,
        "seeds": SEEDS, "z_target": Z_TARGET, "eval_seed": EVAL_SEED,
        "eval_n_paths": EVAL_N_PATHS, "a_max": A_MAX, "T_years": T_YEARS,
        "leverage_near_cap_threshold": LEV_NEAR_CAP_THRESHOLD,
        "trading_days_per_year_for_dwell_fraction": TRADING_DAYS_PER_YEAR,
        "confidence_level": 0.95, "t_distribution_df": N_SEEDS - 1,
        "t_critical_value_95": T_CRIT_95,
        "stage1_eval_dir_per_seed": {str(k): v for k, v in STAGE1_EVAL_REL.items()},
        "stage1_excluded_dirs": {str(k): {"path": v, "reason": STAGE1_EXCLUSION_REASON}
                                  for k, v in STAGE1_EVAL_EXCLUDED_REL.items()},
        "stage2_eval_dir_per_seed": {str(k): v for k, v in STAGE2_EVAL_REL.items()},
        "validation_warnings": warn,
        "validation_info": info,
    }
    (out / "aggregation_config.json").write_text(json.dumps(config, indent=2))

    print("Building ICA report tables...", file=sys.stderr)
    table1 = make_table1(agg, out)
    table2 = make_table2(agg, out)
    make_table3(paired, out)
    make_table4_full_intervals(agg, out)

    print("Building ICA report figures...", file=sys.stderr)
    fig1_mean_vs_std(seed_level, agg, out)
    fig2_terminal_wealth_distribution(combined, out)
    fig3_note(out)
    fig4_cost_turnover(agg, out)
    fig5_training_seed_variability(seed_level, out)

    print("Writing reports...", file=sys.stderr)
    write_data_quality_report(out, base, warn, info, seed_level)
    write_results_summary(out, agg, paired, table1, table2, git_commit)

    print("Done.", file=sys.stderr)
    return base, out, seed_level, agg, paired, combined, diag, manifest, info, warn, git_commit


if __name__ == "__main__":
    main()
