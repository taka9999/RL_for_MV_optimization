#!/usr/bin/env python3
"""
Phase 1A seeds 1-4 queue supervisor.

Runs as a standalone, long-lived process inside its own detached tmux
session, independent of any Claude agent/fork process or session. It does
NOT depend on being "woken up" by anything external - it loops internally
with a blocking sleep, so it keeps working even if nothing is watching it,
which is exactly the property the two previous Claude-fork-based monitors
lacked (they tried to detach their own wait loop and got silently killed).

Scope discipline (hardcoded, not configurable via CLI, on purpose):
  * Only ever touches seeds 1, 2, 3, 4. It never inspects, signals, or
    otherwise interferes with seed 0's tmux sessions/processes.
  * Only ever launches poemv_rs.train / poemv_rs.stage2_direct_boundary /
    poemv_rs.eval_compare / poemv_rs.stage2_eval with the fixed baseline
    hyperparameters already used for seed 0. It never changes a
    hyperparameter, architecture, or loss/update rule. It contains zero
    training-algorithm code of its own.
  * Max 2 concurrent heavy (Stage1 or Stage2) jobs, gated by repeated
    read-only OS resource checks, never more.
  * Stage 1 and Stage 2 for the SAME seed are never run concurrently.
  * Never uses kill -9 except as an explicit last resort after graceful
    (SIGINT then SIGTERM) attempts have failed.
  * Never resumes a job more than once, and never resumes Stage 2 at all
    (poemv_rs.stage2_direct_boundary has no --resume_from mechanism).
"""
from __future__ import annotations
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
BASE_OUT = REPO_ROOT / "results/ica2026_additional_runs/phase1a_endtoend_20260725"
SUP_DIR = BASE_OUT / "supervisor"
STATE_PATH = SUP_DIR / "state.json"
LOG_PATH = SUP_DIR / "supervisor.log"
MANIFEST_PATH = BASE_OUT / "manifest" / "run_manifest.csv"

CONDA_ACTIVATE = "source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh && conda activate JM"

EVAL_SEED = 999
EVAL_N_PATHS = 1000
CHECKPOINT_EVERY = 250
LOOP_INTERVAL_S = 300  # 5 minutes
MIN_GATE_WINDOW_S = 900  # 15 minutes of sustained-green history required before adding a 2nd job
AVAIL_GB_MIN = 2.5
AVAIL_GB_DEGRADE = 1.5
SWAP_GROWTH_MB_MAX_15MIN = 200
LOAD1_MAX = 6.5
DISK_FREE_GB_MIN = 5.0
STALE_METRICS_MIN = 20  # a running job whose metrics.csv hasn't updated in this long is suspect

STAGE1_TOTAL_ITERS = 9000
STAGE2_TOTAL_ITERS = 800

# Argument lists (NOT pre-joined strings) - built into a shell-safe command via
# shlex.join()/shlex.quote() in build_stage1_cmd/build_stage2_cmd below. This is
# Bug B's fix: the previous version embedded literal double-quotes around the
# JSON args inside a Python string that was then wrapped in ANOTHER pair of
# double quotes by tmux_launch(), so the inner quotes prematurely closed the
# outer one and corrupted the command tmux received - every seed 1-4 Stage1
# launch failed before even creating a log file. Building the argv as a list
# and only ever quoting once (via shlex) removes the whole class of bug.
STAGE1_BASE_ARGS = [
    "--mode", "estimated_params", "--iters", str(STAGE1_TOTAL_ITERS), "--episodes_per_iter", "16",
    "--T", "1.0", "--dt", "0.003968253968253968", "--z", "1.2", "--a_max", "1.0",
    "--cap_mode", "component_tanh",
    "--alpha_theta", "3e-4", "--alpha_phi", "1e-4", "--alpha_w", "1e-2",
    "--omega_update_every", "10", "--omega_ema_beta", "0.5",
    "--critic_steps", "1", "--lr_step_every", "2000", "--lr_gamma", "0.5",
    "--actor_mix_tail", "0.3", "--g_p_dep", "--cov_scale", "0.1",
    "--est_mu1_json", "[0.3883,0.2174]",
    "--est_mu2_json", "[-0.4299,-0.3930]",
    "--est_sigma_json", "[[0.0490,0.0117],[0.0117,0.0323]]",
    "--est_lam1", "0.1641", "--est_lam2", "1.75",
]
STAGE2_BASE_ARGS = [
    "--iters", str(STAGE2_TOTAL_ITERS), "--episodes_per_iter", "128",
    "--T", "1.0", "--dt", "0.003968253968253968", "--z", "1.2", "--x0", "1.0", "--p0", "0.5",
    "--tcost", "0.002", "--hidden", "64", "--lr", "0.0005",
    "--weight_decay", "1e-6", "--gamma_risk", "5.0", "--turnover_coef", "0.0005",
    "--utility_kind", "log", "--utility_gamma", "2.0",
    "--lr_step_size", "100", "--lr_decay", "0.5", "--gap_l2_coef", "1e-5", "--gross_lev_coef", "1e-5",
    "--val_every", "25", "--val_n_paths", "64", "--no_precompute_center_path",
    "--filter_mode", "estimated_params", "--num_workers", "6",
]


def build_stage1_cmd(seed: int, outdir: Path, resume: bool = False,
                      iters_override: int | None = None) -> str:
    argv = ["python", "-u", "-m", "poemv_rs.train"] + list(STAGE1_BASE_ARGS)
    if iters_override is not None:
        i = argv.index("--iters")
        argv[i + 1] = str(iters_override)
    argv += ["--checkpoint_every", str(CHECKPOINT_EVERY), "--seed", str(seed), "--outdir", str(outdir)]
    if resume:
        argv += ["--resume_from", str(outdir / "checkpoint_latest.pt")]
    quoted = shlex.join(argv)
    logfile = shlex.quote(str(outdir / "stdout.log"))
    return f"{CONDA_ACTIVATE} && cd {shlex.quote(str(REPO_ROOT))} && {quoted} > {logfile} 2>&1"


def build_stage2_cmd(seed: int, stage1_dir: Path, stage1_ckpt: Path, outdir: Path) -> str:
    argv = (
        ["python", "-u", "-m", "poemv_rs.stage2_direct_boundary",
         "--stage1_run_dir", str(stage1_dir), "--stage1_checkpoint", str(stage1_ckpt),
         "--outdir", str(outdir), "--seed", str(seed)]
        + list(STAGE2_BASE_ARGS)
    )
    quoted = shlex.join(argv)
    logfile = shlex.quote(str(outdir / "stdout.log"))
    return f"{CONDA_ACTIVATE} && cd {shlex.quote(str(REPO_ROOT))} && {quoted} > {logfile} 2>&1"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(msg: str) -> None:
    line = f"[{now_iso()}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def manifest_row(**kw) -> None:
    header = ["timestamp", "seed", "stage", "tmux_session", "pid", "command",
              "start_time", "output_dir", "status", "note"]
    is_new = not MANIFEST_PATH.exists()
    with open(MANIFEST_PATH, "a", encoding="utf-8") as f:
        if is_new:
            f.write(",".join(header) + "\n")
        row = [str(kw.get(h, "")).replace(",", ";").replace("\n", " ") for h in header]
        f.write(",".join(row) + "\n")


# ---------------------------------------------------------------------------
# Read-only OS resource sampling
# ---------------------------------------------------------------------------

def _run(cmd: str) -> str:
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout
    except Exception as e:
        return f"__ERROR__ {e}"


def sample_resources(tracked_pids: list[int]) -> dict:
    out = {"timestamp": now_iso()}
    lvl = _run("sysctl -n kern.memorystatus_vm_pressure_level").strip()
    try:
        out["pressure_level"] = int(lvl)
    except ValueError:
        out["pressure_level"] = None
    vm = _run("vm_stat")
    fields = {}
    for m in re.finditer(r'"?([A-Za-z ]+?)"?:\s+(\d+)\.', vm):
        fields[m.group(1).strip()] = int(m.group(2))
    page_size_m = re.search(r"page size of (\d+) bytes", vm)
    page_size = int(page_size_m.group(1)) if page_size_m else 16384
    free = fields.get("Pages free", 0)
    inactive = fields.get("Pages inactive", 0)
    spec = fields.get("Pages speculative", 0)
    out["avail_gb"] = round((free + inactive + spec) * page_size / 1e9, 3)
    out["compressor_pages"] = fields.get("Pages occupied by compressor", 0)
    out["pageouts"] = fields.get("Pageouts", 0)
    out["swapouts"] = fields.get("Swapouts", 0)
    sw = _run("sysctl -n vm.swapusage")
    m = re.search(r"used\s*=\s*([\d.]+)M", sw)
    out["swap_used_mb"] = float(m.group(1)) if m else None
    try:
        out["load1"] = os.getloadavg()[0]
    except OSError:
        out["load1"] = None
    try:
        du = shutil.disk_usage(str(BASE_OUT))
        out["disk_free_gb"] = round(du.free / 1e9, 2)
    except OSError:
        out["disk_free_gb"] = None
    rss = 0
    for pid in tracked_pids:
        r = _run(f"ps -o rss= -p {pid}").strip()
        if r.isdigit():
            rss += int(r)
    out["tracked_rss_mb"] = round(rss / 1024, 1)
    return out


def resource_is_green(r: dict) -> bool:
    if r.get("pressure_level") != 1:
        return False
    if r.get("avail_gb") is None or r["avail_gb"] < AVAIL_GB_MIN:
        return False
    if r.get("disk_free_gb") is None or r["disk_free_gb"] < DISK_FREE_GB_MIN:
        return False
    if r.get("load1") is not None and r["load1"] > LOAD1_MAX:
        return False
    return True


def sustained_green_for(history: list[dict], seconds: int) -> bool:
    """True if every reading covering at least `seconds` of trailing history is green
    AND swap growth over that window is bounded."""
    if len(history) < 2:
        return False
    t_now = datetime.fromisoformat(history[-1]["timestamp"])
    window = [h for h in history if (t_now - datetime.fromisoformat(h["timestamp"])).total_seconds() <= seconds]
    if not window:
        return False
    span = (t_now - datetime.fromisoformat(window[0]["timestamp"])).total_seconds()
    if span < seconds * 0.9:
        return False  # not enough history yet
    if not all(resource_is_green(h) for h in window):
        return False
    swaps = [h["swap_used_mb"] for h in window if h.get("swap_used_mb") is not None]
    if len(swaps) >= 2 and (swaps[-1] - swaps[0]) > SWAP_GROWTH_MB_MAX_15MIN:
        return False
    return True


def memory_degraded(r: dict) -> bool:
    if r.get("pressure_level") not in (1,) and r.get("pressure_level") is not None:
        return True
    if r.get("avail_gb") is not None and r["avail_gb"] < AVAIL_GB_DEGRADE:
        return True
    return False


# ---------------------------------------------------------------------------
# tmux helpers
# ---------------------------------------------------------------------------

def tmux_session_exists(name: str) -> bool:
    return subprocess.run(["tmux", "has-session", "-t", name],
                           capture_output=True).returncode == 0


def tmux_pane_pid(name: str) -> int | None:
    """Best-effort: find the actual `python -m poemv_rs...` process in this pane's
    process tree, not just the first child (which under --num_workers can be a
    multiprocessing resource_tracker/semaphore helper rather than the real trainer)."""
    out = _run(f"tmux list-panes -t {name} -F '#{{pane_pid}}'").strip()
    if not out.isdigit():
        return None
    root_pid = int(out)
    # BFS the process tree rooted at the pane pid, preferring a `poemv_rs` match.
    frontier = [root_pid]
    seen = set()
    candidates = []
    for _ in range(4):  # a few levels is plenty (pane shell -> python -> workers)
        next_frontier = []
        for p in frontier:
            if p in seen:
                continue
            seen.add(p)
            children = _run(f"pgrep -P {p}").strip().split()
            for c in children:
                if not c.isdigit():
                    continue
                cpid = int(c)
                cmd = _run(f"ps -o command= -p {cpid}").strip()
                if "poemv_rs" in cmd and "resource_tracker" not in cmd and "semaphore_tracker" not in cmd:
                    candidates.append(cpid)
                next_frontier.append(cpid)
        frontier = next_frontier
        if candidates:
            break
    if candidates:
        return candidates[0]
    return root_pid


def tmux_launch(session: str, shell_cmd: str) -> None:
    # Pass argv as a LIST to subprocess.run (no shell=True) so tmux receives
    # `shell_cmd` as a single, un-mangled argument - no outer shell re-parses
    # or re-quotes it. tmux then runs it via the pane's own shell exactly
    # once, which is the only place it needs to be shell-syntax-valid.
    subprocess.run(["tmux", "new-session", "-d", "-s", session, shell_cmd], check=True)


def graceful_stop(session: str, pid: int | None) -> str:
    """SIGINT, then SIGTERM, then (last resort) SIGKILL. Returns method used."""
    if pid is None:
        return "no-pid-found"
    subprocess.run(f"tmux send-keys -t {session} C-c", shell=True)
    for _ in range(6):
        time.sleep(5)
        if subprocess.run(f"kill -0 {pid}", shell=True).returncode != 0:
            return "SIGINT"
    subprocess.run(f"kill -TERM {pid}", shell=True)
    for _ in range(6):
        time.sleep(5)
        if subprocess.run(f"kill -0 {pid}", shell=True).returncode != 0:
            return "SIGTERM"
    subprocess.run(f"kill -9 {pid}", shell=True)
    log(f"WARNING: had to SIGKILL pid={pid} session={session} after SIGINT/SIGTERM did not stop it")
    return "SIGKILL"


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

@dataclass
class JobState:
    seed: int
    stage: str  # "stage1" | "stage2"
    session: str
    pid: int | None
    start_time: str
    output_dir: str
    resume_attempted: bool = False


@dataclass
class SupervisorState:
    pending: list[int] = field(default_factory=lambda: [1, 2, 3, 4])
    running: dict = field(default_factory=dict)   # seed(str) -> JobState dict
    completed: list[int] = field(default_factory=list)
    failed: dict = field(default_factory=dict)    # seed(str) -> reason
    failure_causes: list[str] = field(default_factory=list)
    resource_history: list[dict] = field(default_factory=list)
    global_halt: bool = False
    halt_reason: str = ""


SEED0_STAGE2_SESSION = "ica_phase1_stage2_seed0"


def load_state() -> SupervisorState:
    if STATE_PATH.exists():
        d = json.loads(STATE_PATH.read_text())
        return SupervisorState(**d)
    s = SupervisorState()
    # Seed 0's Stage 1 + Stage 1 eval already completed and passed gate (done by the
    # coordinator before this supervisor was ever started). Seed 0's Stage 2 training is
    # ALREADY RUNNING in tmux session SEED0_STAGE2_SESSION, launched directly by the
    # coordinator, not by this script. On first startup, adopt it into `running` purely
    # for read-only reconciliation (eval + gate + mark-complete once it finishes) - this
    # script must NEVER restart, resume, or signal that job itself; only react once its
    # tmux session ends on its own.
    if tmux_session_exists(SEED0_STAGE2_SESSION):
        pid = tmux_pane_pid(SEED0_STAGE2_SESSION)
        outdir = BASE_OUT / "stage2" / "seed_0"
        job = JobState(seed=0, stage="stage2", session=SEED0_STAGE2_SESSION, pid=pid,
                        start_time=now_iso(), output_dir=str(outdir))
        s.running["0"] = asdict(job)
        log(f"Adopted already-running seed-0 Stage2 job (session={SEED0_STAGE2_SESSION}, "
            f"pid={pid}) for read-only monitoring. Will not touch it; will run Stage2 eval "
            f"and mark seed 0 complete once it finishes on its own.")
    else:
        log("WARNING: seed-0 Stage2 tmux session not found at supervisor startup - "
            "assuming seed 0 already finished and was handled elsewhere. Proceeding "
            "directly to seeds 1-4.")
    return s


def save_state(s: SupervisorState) -> None:
    STATE_PATH.write_text(json.dumps(asdict(s), indent=2))


# ---------------------------------------------------------------------------
# Anomaly checks (cheap, read-only; NOT a substitute for the full aggregate
# analysis done after all seeds finish - this is just a pass/fail gate)
# ---------------------------------------------------------------------------

def check_stage1_metrics(outdir: Path) -> tuple[bool, str]:
    mpath = outdir / "metrics.csv"
    if not mpath.exists():
        return False, "metrics.csv missing"
    try:
        import pandas as pd
        df = pd.read_csv(mpath)
    except Exception as e:
        return False, f"metrics.csv unreadable: {e}"
    if df["iter"].duplicated().any():
        return False, "duplicate iterations in metrics.csv"
    expected_nan_cols = {"mean_xT_window", "mean_xT_minus_z"}
    nan_counts = df.isna().sum()
    for col, cnt in nan_counts.items():
        if cnt == 0:
            continue
        if col in expected_nan_cols and cnt <= 9:
            continue
        return False, f"unexpected NaN in column {col} ({cnt} rows)"
    import numpy as np
    numcols = df.select_dtypes("number")
    if np.isinf(numcols.values).any():
        return False, "Inf present in metrics.csv"
    if len(df) > 0:
        tail = df.tail(500)
        if tail["avg_gross_leverage"].mean() >= 1.98:
            return False, "avg_gross_leverage pinned near cap for last 500 iters"
    return True, "ok"


def _check_checkpoint(path: Path, required_keys: list[str]) -> tuple[bool, str]:
    """Generic loadability + schema check. Stage 1 and Stage 2 checkpoints use
    DIFFERENT schemas (Bug A fix) - callers must pass the right `required_keys`
    for the stage they're checking; there is no shared 'iter' key across both."""
    if not path.exists():
        return False, f"{path} missing"
    code = (
        "import torch\n"
        f"ck = torch.load({str(path)!r}, map_location='cpu', weights_only=False)\n"
        "assert isinstance(ck, dict), 'checkpoint is not a dict'\n"
        f"missing = [k for k in {required_keys!r} if k not in ck]\n"
        "assert not missing, f'missing keys: {missing}'\n"
        "print('OK', ck.get('iter', 'n/a'))\n"
    )
    cmd = f"{CONDA_ACTIVATE} && " + shlex.join(["python3", "-c", code])
    r = subprocess.run(cmd, shell=True, cwd=str(REPO_ROOT),
                        capture_output=True, text=True, executable="/bin/zsh")
    if r.returncode != 0 or "OK" not in r.stdout:
        return False, f"checkpoint load failed: {(r.stderr or r.stdout)[-300:]}"
    return True, r.stdout.strip()


def check_stage1_checkpoint(path: Path) -> tuple[bool, str]:
    return _check_checkpoint(path, ["iter", "vf_state_dict", "pi_state_dict"])


def check_stage2_checkpoint(path: Path) -> tuple[bool, str]:
    return _check_checkpoint(path, ["model_state_dict", "model_cfg", "train_cfg"])


def _last_metrics_iter(outdir: Path) -> tuple[int | None, str]:
    mpath = outdir / "metrics.csv"
    if not mpath.exists():
        return None, "metrics.csv missing"
    try:
        import pandas as pd
        df = pd.read_csv(mpath)
    except Exception as e:
        return None, f"metrics.csv unreadable: {e}"
    if len(df) == 0 or "iter" not in df.columns:
        return None, "metrics.csv empty or missing iter column"
    return int(df["iter"].max()), "ok"


def stage1_training_complete(outdir: Path) -> tuple[bool, str]:
    """Combine metrics.csv's last iteration, the expected total, and final
    checkpoint validity/loadability - NOT a checkpoint-internal 'iter' field
    (Bug A fix)."""
    last_iter, msg = _last_metrics_iter(outdir)
    if last_iter is None:
        return False, msg
    if last_iter < STAGE1_TOTAL_ITERS:
        return False, f"metrics.csv last iter={last_iter} < expected {STAGE1_TOTAL_ITERS}"
    ok, cmsg = check_stage1_checkpoint(outdir / "checkpoint.pt")
    if not ok:
        return False, f"final checkpoint invalid: {cmsg}"
    return True, f"iter={last_iter}/{STAGE1_TOTAL_ITERS}, {cmsg}"


def stage2_training_complete(outdir: Path) -> tuple[bool, str]:
    last_iter, msg = _last_metrics_iter(outdir)
    if last_iter is None:
        return False, msg
    if last_iter < STAGE2_TOTAL_ITERS:
        return False, f"metrics.csv last iter={last_iter} < expected {STAGE2_TOTAL_ITERS}"
    ok, cmsg = check_stage2_checkpoint(outdir / "checkpoint.pt")
    if not ok:
        return False, f"final checkpoint invalid: {cmsg}"
    return True, f"iter={last_iter}/{STAGE2_TOTAL_ITERS}, {cmsg}"


def check_eval_csv_sane(csv_path: Path, wealth_cols: list[str]) -> tuple[bool, str]:
    if not csv_path.exists():
        return False, f"{csv_path} missing"
    try:
        import pandas as pd
        import numpy as np
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"unreadable: {e}"
    for c in wealth_cols:
        if c not in df.columns:
            continue
        vals = df[c].values
        if np.isnan(vals).any() or np.isinf(vals).any():
            return False, f"NaN/Inf in {c}"
        if (np.abs(vals) > 1000).any():
            return False, f"{c} shows numerical blow-up (|value|>1000)"
    return True, "ok"


def check_gross_net_invariant(csv_path: Path) -> tuple[bool, str]:
    try:
        import pandas as pd
        import numpy as np
        df = pd.read_csv(csv_path)
    except Exception as e:
        return False, f"unreadable: {e}"
    if not {"gross_terminal_wealth", "net_terminal_wealth", "cumulative_tc"}.issubset(df.columns):
        return True, "invariant columns not present, skipping (not this run's concern)"
    diff = (df["gross_terminal_wealth"] - df["net_terminal_wealth"] - df["cumulative_tc"]).abs()
    maxdiff = float(diff.max())
    if maxdiff > 1e-6:
        return False, f"gross-net invariant violated, max abs diff={maxdiff}"
    return True, f"invariant holds, max abs diff={maxdiff:.2e}"


def record_failure(state: SupervisorState, seed: int, cause: str, detail: str) -> None:
    """Bug C fix: dedupe by (seed, cause) - `state.failed` is keyed by seed, so a
    given seed can only ever contribute ONE entry no matter how many times this
    is called for it (e.g. if reconcile() somehow saw it twice in one cycle).
    The same-cause count used for global_halt is derived from DISTINCT SEEDS in
    `state.failed` matching `cause`, not from an ever-growing append-only list -
    this is what "2+ RUNS failing with the same cause" should mean, and it can
    no longer inflate from a single seed's dead job being reprocessed repeatedly
    (that reprocessing bug is separately fixed in reconcile())."""
    already_recorded = state.failed.get(str(seed), "").startswith(cause + ":")
    state.failed[str(seed)] = f"{cause}: {detail}"
    if not already_recorded:
        state.failure_causes.append(cause)
        manifest_row(timestamp=now_iso(), seed=seed, stage="?", status="failed", note=f"{cause}: {detail}")
    log(f"SEED {seed} FAILED [{cause}]: {detail}" + (" (duplicate re-record, not re-counted)"
                                                       if already_recorded else ""))
    same_cause_seeds = sum(1 for v in state.failed.values() if v.startswith(cause + ":"))
    if same_cause_seeds >= 2 and not state.global_halt:
        state.global_halt = True
        state.halt_reason = f"2+ distinct seeds failed with the same cause: {cause}"
        log(f"GLOBAL HALT triggered: {state.halt_reason}. No new jobs will be launched. "
            f"Already-running jobs are left alone.")


# ---------------------------------------------------------------------------
# Job launching
# ---------------------------------------------------------------------------

def launch_stage1(state: SupervisorState, seed: int, resume: bool = False) -> None:
    outdir = BASE_OUT / "stage1" / f"seed_{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    session = f"ica_phase1_seed{seed}_stage1" + ("_resume1" if resume else "")
    cmd = build_stage1_cmd(seed, outdir, resume=resume)
    tmux_launch(session, cmd)
    time.sleep(3)
    pid = tmux_pane_pid(session)
    job = JobState(seed=seed, stage="stage1", session=session, pid=pid,
                    start_time=now_iso(), output_dir=str(outdir), resume_attempted=resume)
    state.running[str(seed)] = asdict(job)
    log(f"Launched seed {seed} Stage1 (resume={resume}) session={session} pid={pid}")
    manifest_row(timestamp=now_iso(), seed=seed, stage="stage1", tmux_session=session,
                 pid=pid, command=cmd, start_time=job.start_time, output_dir=str(outdir),
                 status="started", note="resume" if resume else "fresh")


def launch_stage2(state: SupervisorState, seed: int) -> None:
    stage1_dir = BASE_OUT / "stage1" / f"seed_{seed}"
    stage1_ckpt = stage1_dir / "checkpoint.pt"
    outdir = BASE_OUT / "stage2" / f"seed_{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    session = f"ica_phase1_seed{seed}_stage2"
    cmd = build_stage2_cmd(seed, stage1_dir, stage1_ckpt, outdir)
    tmux_launch(session, cmd)
    time.sleep(3)
    pid = tmux_pane_pid(session)
    sha = _run(f"shasum -a 256 {stage1_ckpt}").split()[0] if stage1_ckpt.exists() else ""
    job = JobState(seed=seed, stage="stage2", session=session, pid=pid,
                    start_time=now_iso(), output_dir=str(outdir))
    state.running[str(seed)] = asdict(job)
    log(f"Launched seed {seed} Stage2 session={session} pid={pid} "
        f"stage1_checkpoint={stage1_ckpt} sha256={sha}")
    manifest_row(timestamp=now_iso(), seed=seed, stage="stage2", tmux_session=session,
                 pid=pid, command=cmd, start_time=job.start_time, output_dir=str(outdir),
                 status="started", note=f"stage1_ckpt_sha256={sha}")


def run_eval_stage1(seed: int) -> tuple[bool, str]:
    stage1_dir = BASE_OUT / "stage1" / f"seed_{seed}"
    outdir = BASE_OUT / "eval_stage1" / f"seed_{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = (
        f"{CONDA_ACTIVATE} && cd '{REPO_ROOT}' && python -m poemv_rs.eval_compare "
        f"--checkpoint {stage1_dir}/checkpoint.pt --run_dir {stage1_dir} "
        f"--n_paths {EVAL_N_PATHS} --seed {EVAL_SEED} --filter_mode estimated_params "
        # eval_compare.py's --T/--a_max defaults are 10.0/2.0 (paper's original 10y-horizon
        # demo), which do NOT match this project's 1-year/a_max=1.0 baseline training config.
        # Discovered the hard way during the seed-0 pilot (an eval silently ran with a 10x
        # longer horizon than the model was trained on) - must always override explicitly.
        f"--T 1.0 --dt 0.003968253968253968 --a_max 1.0 --z 1.2 --x0 1.0 --p0 0.5 --r 0.01 "
        f"--outdir {outdir}"
    )
    log(f"Running Stage1 eval for seed {seed} (blocking)...")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/zsh")
    if r.returncode != 0:
        return False, f"eval_compare.py exited {r.returncode}: {r.stderr[-500:]}"
    manifest_row(timestamp=now_iso(), seed=seed, stage="eval_stage1", command=cmd,
                 output_dir=str(outdir), status="done")
    csvs = list(outdir.glob("*.csv"))
    for c in csvs:
        ok, msg = check_eval_csv_sane(c, [col for col in
                                           ["terminal_wealth", "xT", "XT"] ])
        if not ok:
            return False, f"{c.name}: {msg}"
    return True, "ok"


def run_eval_stage2(seed: int) -> tuple[bool, str]:
    stage1_dir = BASE_OUT / "stage1" / f"seed_{seed}"
    stage2_dir = BASE_OUT / "stage2" / f"seed_{seed}"
    outdir = BASE_OUT / "eval_stage2" / f"seed_{seed}"
    outdir.mkdir(parents=True, exist_ok=True)
    stage2_ckpt = stage2_dir / "best_checkpoint.pt"
    if not stage2_ckpt.exists():
        stage2_ckpt = stage2_dir / "checkpoint.pt"
    cmd = (
        f"{CONDA_ACTIVATE} && cd '{REPO_ROOT}' && python -m poemv_rs.stage2_eval "
        f"--stage1_run_dir {stage1_dir} --stage1_checkpoint {stage1_dir}/checkpoint.pt "
        f"--stage2_checkpoint {stage2_ckpt} --outdir {outdir} "
        f"--n_paths {EVAL_N_PATHS} --seed {EVAL_SEED} --T 1.0 --dt 0.003968253968253968 --z 1.2 "
        f"--x0 1.0 --tcost 0.002 --monthly_steps 21 --lev_cap 1.5 --filter_mode estimated_params"
    )
    log(f"Running Stage2 eval for seed {seed} (blocking)...")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, executable="/bin/zsh")
    if r.returncode != 0:
        return False, f"stage2_eval.py exited {r.returncode}: {r.stderr[-500:]}"
    manifest_row(timestamp=now_iso(), seed=seed, stage="eval_stage2", command=cmd,
                 output_dir=str(outdir), status="done")
    csvs = list(outdir.glob("*.csv"))
    for c in csvs:
        ok, msg = check_gross_net_invariant(c)
        if not ok:
            return False, f"{c.name}: {msg}"
    return True, "ok"


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def reconcile(state: SupervisorState) -> None:
    """Check each running job; move to eval/next-stage/failed/completed as appropriate."""
    for seed_str, job_d in list(state.running.items()):
        seed = int(seed_str)
        job = JobState(**job_d)
        alive = tmux_session_exists(job.session)
        if alive:
            # stale-metrics check (only meaningful for stage1, which writes metrics.csv)
            if job.stage == "stage1":
                mpath = Path(job.output_dir) / "metrics.csv"
                if mpath.exists():
                    age_min = (time.time() - mpath.stat().st_mtime) / 60.0
                    if age_min > STALE_METRICS_MIN:
                        log(f"WARNING seed {seed} stage1 metrics.csv stale for {age_min:.0f} min "
                            f"(session still alive) - not stopping automatically, just flagging")
            continue

        # session ended - did it finish cleanly? Bug C fix: `job` is ALWAYS removed
        # from state.running exactly once below, on every branch (completed, crashed,
        # or moved to a resume attempt) - a job entry must never be left in `running`
        # after its tmux session has ended, or reconcile() will reprocess the same
        # dead job (and re-call record_failure) every single loop cycle forever.
        outdir = Path(job.output_dir)
        del state.running[seed_str]
        if job.stage == "stage1":
            ok, msg = stage1_training_complete(outdir)  # Bug A fix: stage-specific check
            if ok:
                log(f"Seed {seed} Stage1 completed cleanly. {msg}")
                manifest_row(timestamp=now_iso(), seed=seed, stage="stage1",
                             output_dir=job.output_dir, status="completed")
                eok, emsg = run_eval_stage1(seed)
                if not eok:
                    record_failure(state, seed, "eval_stage1_anomaly", emsg)
                    continue
                log(f"Seed {seed} Stage1 eval passed gate: {emsg}")
                launch_stage2(state, seed)
            else:
                if not job.resume_attempted:
                    latest = outdir / "checkpoint_latest.pt"
                    if latest.exists():
                        log(f"Seed {seed} Stage1 stopped early ({msg}); attempting ONE resume "
                            f"from checkpoint_latest.pt")
                        launch_stage1(state, seed, resume=True)
                    else:
                        record_failure(state, seed, "stage1_crash_no_checkpoint", msg)
                else:
                    record_failure(state, seed, "stage1_crash_after_resume", msg)
        else:  # stage2
            ok, msg = stage2_training_complete(outdir)  # Bug A fix: stage-specific check
            if ok:
                log(f"Seed {seed} Stage2 completed. {msg}")
                manifest_row(timestamp=now_iso(), seed=seed, stage="stage2",
                             output_dir=job.output_dir, status="completed")
                eok, emsg = run_eval_stage2(seed)
                if not eok:
                    record_failure(state, seed, "eval_stage2_anomaly", emsg)
                    continue
                log(f"Seed {seed} Stage2 eval passed gate: {emsg}")
                state.completed.append(seed)
            else:
                # Stage2 has no resume mechanism - single failure, no retry.
                record_failure(state, seed, "stage2_crash_no_resume", msg)


def maybe_launch_next(state: SupervisorState, resources: dict) -> None:
    if state.global_halt:
        return
    if not state.pending:
        return
    if "0" in state.running:
        # Hard precondition from the user: seed 0's FULL end-to-end pilot (Stage1+eval+
        # Stage2+eval) must pass its anomaly gate before ANY other seed starts - not even
        # as a resource-permitting concurrent job. Do not start seeds 1-4 while seed 0 is
        # still running, regardless of how much headroom the resource check reports.
        return
    n_running = len(state.running)
    if n_running == 0:
        seed = state.pending.pop(0)
        launch_stage1(state, seed)
        return
    if n_running >= 2:
        return
    # would be the 2nd concurrent job - require the strict gate
    if sustained_green_for(state.resource_history, MIN_GATE_WINDOW_S):
        # also require no currently-running job looking suspect
        for seed_str, job_d in state.running.items():
            job = JobState(**job_d)
            if job.stage == "stage1":
                ok, _ = check_stage1_metrics(Path(job.output_dir))
                if not ok:
                    log("Not adding a 2nd job: an existing running job looks anomalous.")
                    return
        seed = state.pending.pop(0)
        log(f"Resource gate passed (green for >= {MIN_GATE_WINDOW_S/60:.0f} min): "
            f"starting 2nd concurrent job for seed {seed}. Latest reading: {resources}")
        launch_stage1(state, seed)
    else:
        pass  # stay serial until sustained green


def check_degrade(state: SupervisorState, resources: dict) -> None:
    if len(state.running) < 2:
        return
    if not memory_degraded(resources):
        return
    log(f"Memory degradation detected with 2 jobs running: {resources}. "
        f"Pausing new launches. Will attempt to gracefully stop the later-started job "
        f"IF it is a Stage1 job (resumable) once it reaches its next periodic checkpoint. "
        f"Stage2 jobs are never preemptively stopped (no resume mechanism) - only new "
        f"launches are paused in that case.")
    # find the later-started job
    jobs = [JobState(**d) for d in state.running.values()]
    jobs.sort(key=lambda j: j.start_time)
    later = jobs[-1]
    if later.stage != "stage1":
        log(f"Later job (seed {later.seed}) is Stage2 - not stopping it. Waiting it out.")
        return
    outdir = Path(later.output_dir)
    mpath = outdir / "metrics.csv"
    if not mpath.exists():
        return
    try:
        import pandas as pd
        last_iter = int(pd.read_csv(mpath)["iter"].max())
    except Exception:
        return
    ckpt = outdir / f"checkpoint_iter_{last_iter - (last_iter % CHECKPOINT_EVERY):07d}.pt"
    if last_iter % CHECKPOINT_EVERY == 0 and ckpt.exists():
        age_s = time.time() - ckpt.stat().st_mtime
        if age_s < 120:  # just landed a fresh periodic checkpoint
            log(f"Seed {later.seed} just reached a periodic checkpoint (iter {last_iter}). "
                f"Gracefully stopping this job now; it can be resumed later via --resume_from.")
            method = graceful_stop(later.session, later.pid)
            manifest_row(timestamp=now_iso(), seed=later.seed, stage="stage1",
                         status="paused_for_memory", note=f"stopped via {method} at iter {last_iter}")
            del state.running[str(later.seed)]
            state.pending.insert(0, later.seed)  # put back at front of queue, not lost


def main() -> None:
    SUP_DIR.mkdir(parents=True, exist_ok=True)
    (BASE_OUT / "manifest").mkdir(parents=True, exist_ok=True)
    state = load_state()
    log(f"Supervisor starting. pending={state.pending} running={list(state.running.keys())} "
        f"completed={state.completed} failed={list(state.failed.keys())} "
        f"global_halt={state.global_halt}")
    while True:
        tracked_pids = [JobState(**d).pid for d in state.running.values() if JobState(**d).pid]
        resources = sample_resources([p for p in tracked_pids if p])
        state.resource_history.append(resources)
        state.resource_history = state.resource_history[-200:]
        log(f"resources: pressure={resources.get('pressure_level')} "
            f"avail_gb={resources.get('avail_gb')} swap_mb={resources.get('swap_used_mb')} "
            f"load1={resources.get('load1')} disk_free_gb={resources.get('disk_free_gb')} "
            f"tracked_rss_mb={resources.get('tracked_rss_mb')} pageouts={resources.get('pageouts')} "
            f"running={list(state.running.keys())} pending={state.pending} "
            f"completed={state.completed} failed={list(state.failed.keys())}")

        reconcile(state)
        check_degrade(state, resources)
        maybe_launch_next(state, resources)

        save_state(state)

        if not state.pending and not state.running:
            log("All seeds resolved (completed or failed). Supervisor exiting.")
            log(f"Final: completed={state.completed} failed={state.failed} "
                f"global_halt={state.global_halt} halt_reason={state.halt_reason}")
            break
        if state.global_halt and not state.running:
            log(f"Global halt with nothing running - stopping (no seeds left to babysit): "
                f"{state.halt_reason}")
            break

        time.sleep(LOOP_INTERVAL_S)


if __name__ == "__main__":
    main()
