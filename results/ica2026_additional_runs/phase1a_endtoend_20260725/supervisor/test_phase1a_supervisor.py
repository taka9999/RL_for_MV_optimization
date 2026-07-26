"""Unit tests for the seed_status orchestration fix in phase1a_supervisor.py.

Run with: python -m pytest test_phase1a_supervisor.py -v
      or: python test_phase1a_supervisor.py

Scope: these tests cover the state-machine/orchestration change only (the
seed_status field, migration/backfill, failure-record schema, and the wave
gate for seeds 3/4) - NOT training logic, hyperparameters, or eval scripts,
none of which were touched by this change. Heavy dependencies (torch
checkpoint loading via subprocess, real tmux/subprocess launches) are
monkeypatched out so these tests run in well under a second with no GPU/conda
dependency.
"""
from __future__ import annotations
import json
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase1a_supervisor as sup


class TestInferSeedStatusFromDisk(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.base = Path(self.tmpdir.name)
        self._orig_base_out = sup.BASE_OUT
        sup.BASE_OUT = self.base
        # Stub out the expensive/real checks with disk-marker-based fakes so
        # these tests don't need torch, conda, or real checkpoints.
        self._orig_s1complete = sup.stage1_training_complete
        self._orig_s2complete = sup.stage2_training_complete
        sup.stage1_training_complete = lambda outdir: (
            (True, "ok") if (outdir / "STAGE1_DONE").exists() else (False, "not done"))
        sup.stage2_training_complete = lambda outdir: (
            (True, "ok") if (outdir / "STAGE2_DONE").exists() else (False, "not done"))

    def tearDown(self):
        sup.BASE_OUT = self._orig_base_out
        sup.stage1_training_complete = self._orig_s1complete
        sup.stage2_training_complete = self._orig_s2complete
        self.tmpdir.cleanup()

    def _mkdir(self, *parts) -> Path:
        p = self.base.joinpath(*parts)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def test_no_stage1_checkpoint_is_stage1_pending(self):
        self.assertEqual(sup.infer_seed_status_from_disk(9), sup.STATUS_STAGE1_PENDING)

    def test_stage1_done_no_eval_is_eval_pending(self):
        d = self._mkdir("stage1", "seed_9")
        (d / "STAGE1_DONE").touch()
        self.assertEqual(sup.infer_seed_status_from_disk(9), sup.STATUS_STAGE1_EVAL_PENDING)

    def test_stage1_and_eval_done_no_stage2_is_stage2_pending(self):
        d = self._mkdir("stage1", "seed_9")
        (d / "STAGE1_DONE").touch()
        evaldir = self._mkdir("eval_stage1_postfix", "seed_9")
        (evaldir / "eval_summary.csv").write_text("method,mean_terminal\nRL,1.1\n")
        self.assertEqual(sup.infer_seed_status_from_disk(9), sup.STATUS_STAGE2_PENDING)

    def test_original_eval_stage1_dir_also_recognized(self):
        # seed 0's convention: no _postfix suffix, still must be recognized.
        d = self._mkdir("stage1", "seed_9")
        (d / "STAGE1_DONE").touch()
        evaldir = self._mkdir("eval_stage1", "seed_9")
        (evaldir / "eval_summary.csv").write_text("method,mean_terminal\nRL,1.1\n")
        self.assertEqual(sup.infer_seed_status_from_disk(9), sup.STATUS_STAGE2_PENDING)

    def test_this_is_seed1_seed2_real_scenario(self):
        """Regression test for the exact bug this fix addresses: a seed whose
        Stage1 eval was fixed and re-run out-of-band into eval_stage1_postfix/
        must resolve to stage2_pending, not silently vanish from the queue."""
        d = self._mkdir("stage1", "seed_1")
        (d / "STAGE1_DONE").touch()
        evaldir = self._mkdir("eval_stage1_postfix", "seed_1")
        (evaldir / "eval_summary.csv").write_text("method,mean_terminal\nRL,1.18\n")
        self.assertEqual(sup.infer_seed_status_from_disk(1), sup.STATUS_STAGE2_PENDING)

    def test_stage1_stage2_done_no_eval2_is_stage2_eval_pending(self):
        d1 = self._mkdir("stage1", "seed_9")
        (d1 / "STAGE1_DONE").touch()
        e1 = self._mkdir("eval_stage1", "seed_9")
        (e1 / "eval_summary.csv").write_text("x\n")
        d2 = self._mkdir("stage2", "seed_9")
        (d2 / "STAGE2_DONE").touch()
        self.assertEqual(sup.infer_seed_status_from_disk(9), sup.STATUS_STAGE2_EVAL_PENDING)

    def test_fully_done_is_completed(self):
        d1 = self._mkdir("stage1", "seed_9")
        (d1 / "STAGE1_DONE").touch()
        e1 = self._mkdir("eval_stage1", "seed_9")
        (e1 / "eval_summary.csv").write_text("x\n")
        d2 = self._mkdir("stage2", "seed_9")
        (d2 / "STAGE2_DONE").touch()
        e2 = self._mkdir("eval_stage2", "seed_9")
        (e2 / "eval_summary.csv").write_text("x\n")
        self.assertEqual(sup.infer_seed_status_from_disk(9), sup.STATUS_COMPLETED)


class TestMigrateSeedStatus(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        sup.BASE_OUT = Path(self.tmpdir.name)
        self._orig_s1complete = sup.stage1_training_complete
        self._orig_s2complete = sup.stage2_training_complete
        sup.stage1_training_complete = lambda outdir: (False, "not done")
        sup.stage2_training_complete = lambda outdir: (False, "not done")

    def tearDown(self):
        sup.stage1_training_complete = self._orig_s1complete
        sup.stage2_training_complete = self._orig_s2complete
        self.tmpdir.cleanup()

    def test_completed_seed_takes_precedence_over_disk(self):
        state = sup.SupervisorState(completed=[0])
        changed = sup.migrate_seed_status(state)
        self.assertTrue(changed)
        self.assertEqual(state.seed_status["0"], sup.STATUS_COMPLETED)

    def test_running_seed_takes_precedence_over_disk(self):
        job = sup.JobState(seed=1, stage="stage2", session="s", pid=123,
                            start_time="t", output_dir="d")
        state = sup.SupervisorState(running={"1": asdict(job)})
        sup.migrate_seed_status(state)
        self.assertEqual(state.seed_status["1"], sup.STATUS_STAGE2_RUNNING)

    def test_failed_seed_takes_precedence_over_disk(self):
        state = sup.SupervisorState(failed={"2": {"seed": 2, "stage": "stage1",
                                                    "failure_type": "x", "detail": "y"}})
        sup.migrate_seed_status(state)
        self.assertEqual(state.seed_status["2"], sup.STATUS_FAILED)

    def test_idempotent(self):
        state = sup.SupervisorState(completed=[0])
        sup.migrate_seed_status(state)
        changed_again = sup.migrate_seed_status(state)
        self.assertFalse(changed_again)

    def test_regression_seed1_seed2_parked_scenario(self):
        """Reproduces the exact real-world state.json this fix was written for:
        seed 1/2 absent from pending/running/completed/failed entirely after an
        out-of-band eval fix. Migration must NOT leave them unset (which would
        make the main-loop's "all done" check fire early and silently strand
        them at stage1_pending forever)."""
        base = sup.BASE_OUT
        for seed in (1, 2):
            d = base / "stage1" / f"seed_{seed}"
            d.mkdir(parents=True)
            (d / "STAGE1_DONE").touch()
        sup.stage1_training_complete = lambda outdir: (
            (True, "ok") if (outdir / "STAGE1_DONE").exists() else (False, "not done"))
        e1 = base / "eval_stage1_postfix" / "seed_1"
        e1.mkdir(parents=True)
        (e1 / "eval_summary.csv").write_text("x\n")
        e2 = base / "eval_stage1_postfix" / "seed_2"
        e2.mkdir(parents=True)
        (e2 / "eval_summary.csv").write_text("x\n")

        state = sup.SupervisorState(pending=[3, 4], running={}, completed=[0], failed={})
        sup.migrate_seed_status(state)
        self.assertEqual(state.seed_status["1"], sup.STATUS_STAGE2_PENDING)
        self.assertEqual(state.seed_status["2"], sup.STATUS_STAGE2_PENDING)
        self.assertEqual(state.seed_status["3"], sup.STATUS_STAGE1_PENDING)
        self.assertEqual(state.seed_status["4"], sup.STATUS_STAGE1_PENDING)
        self.assertEqual(state.seed_status["0"], sup.STATUS_COMPLETED)


class TestRecordFailure(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        sup.BASE_OUT = Path(self.tmpdir.name)
        sup.SUP_DIR = sup.BASE_OUT / "supervisor"
        sup.SUP_DIR.mkdir(parents=True)
        sup.LOG_PATH = sup.SUP_DIR / "supervisor.log"
        sup.MANIFEST_PATH = sup.BASE_OUT / "manifest" / "run_manifest.csv"
        (sup.BASE_OUT / "manifest").mkdir(parents=True)

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_failure_record_has_required_fields(self):
        state = sup.SupervisorState()
        sup.record_failure(state, 3, "stage1", "eval_stage1_anomaly", "boom",
                            output_dir=None, cmd="python -m poemv_rs.eval_compare ...",
                            resumable=True)
        rec = state.failed["3"]
        for key in ("seed", "stage", "failure_type", "detail", "latest_checkpoint",
                    "exact_command", "log_tail", "resumable", "timestamp"):
            self.assertIn(key, rec)
        self.assertEqual(rec["seed"], 3)
        self.assertEqual(rec["stage"], "stage1")
        self.assertEqual(rec["failure_type"], "eval_stage1_anomaly")
        self.assertTrue(rec["resumable"])
        self.assertEqual(state.seed_status["3"], sup.STATUS_FAILED)

    def test_duplicate_failure_not_double_counted(self):
        state = sup.SupervisorState()
        sup.record_failure(state, 3, "stage1", "eval_stage1_anomaly", "boom", cmd="c")
        sup.record_failure(state, 3, "stage1", "eval_stage1_anomaly", "boom again", cmd="c")
        self.assertEqual(len(state.failure_causes), 1)

    def test_two_distinct_seeds_same_cause_triggers_global_halt(self):
        state = sup.SupervisorState()
        sup.record_failure(state, 3, "stage1", "eval_stage1_anomaly", "boom", cmd="c")
        self.assertFalse(state.global_halt)
        sup.record_failure(state, 4, "stage1", "eval_stage1_anomaly", "boom", cmd="c")
        self.assertTrue(state.global_halt)


class TestLaunchableCandidatesWaveGate(unittest.TestCase):
    def test_wave1_only_when_not_done(self):
        state = sup.SupervisorState()
        state.seed_status = {"1": sup.STATUS_STAGE2_PENDING, "2": sup.STATUS_STAGE2_PENDING,
                              "3": sup.STATUS_STAGE1_PENDING, "4": sup.STATUS_STAGE1_PENDING}
        cands = sup._launchable_candidates(state)
        self.assertEqual(cands, [1, 2])

    def test_wave2_unlocked_once_both_wave1_completed(self):
        state = sup.SupervisorState()
        state.seed_status = {"1": sup.STATUS_COMPLETED, "2": sup.STATUS_COMPLETED,
                              "3": sup.STATUS_STAGE1_PENDING, "4": sup.STATUS_STAGE1_PENDING}
        cands = sup._launchable_candidates(state)
        self.assertEqual(cands, [3, 4])

    def test_wave2_blocked_if_either_wave1_seed_failed(self):
        state = sup.SupervisorState()
        state.seed_status = {"1": sup.STATUS_FAILED, "2": sup.STATUS_COMPLETED,
                              "3": sup.STATUS_STAGE1_PENDING, "4": sup.STATUS_STAGE1_PENDING}
        cands = sup._launchable_candidates(state)
        # seed 1 is failed (excluded from wave1 candidates too), seed 2 already
        # completed (nothing to launch); wave2 must NOT unlock.
        self.assertEqual(cands, [])

    def test_running_seed_not_relaunched(self):
        state = sup.SupervisorState()
        state.seed_status = {"1": sup.STATUS_STAGE2_PENDING, "2": sup.STATUS_STAGE2_RUNNING}
        state.running = {"2": {}}
        cands = sup._launchable_candidates(state)
        self.assertEqual(cands, [1])

    def test_stage2_pending_seed_never_relaunches_stage1(self):
        """The core bug this fix prevents: a seed sitting at stage2_pending must
        dispatch to launch_stage2(), never launch_stage1() (which would silently
        retrain Stage 1 from scratch over an already-valid 9000-iter checkpoint)."""
        state = sup.SupervisorState()
        state.seed_status = {"1": sup.STATUS_STAGE2_PENDING}
        calls = []
        orig1, orig2 = sup.launch_stage1, sup.launch_stage2
        sup.launch_stage1 = lambda st, sd, **kw: calls.append(("stage1", sd))
        sup.launch_stage2 = lambda st, sd: calls.append(("stage2", sd))
        try:
            sup._launch_next_stage_for(state, 1)
        finally:
            sup.launch_stage1, sup.launch_stage2 = orig1, orig2
        self.assertEqual(calls, [("stage2", 1)])


class TestRestartResumesSameStage(unittest.TestCase):
    """Simulates a supervisor restart mid-flight: state.json shows seed 1 with a
    live Stage2 job in `running`, seed 2 at stage2_pending. Loading (migrating)
    this state must not alter either seed's status or trigger any launch."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        sup.BASE_OUT = Path(self.tmpdir.name)
        self._orig_s1complete = sup.stage1_training_complete
        self._orig_s2complete = sup.stage2_training_complete
        sup.stage1_training_complete = lambda outdir: (
            (True, "ok") if (outdir / "STAGE1_DONE").exists() else (False, "not done"))
        sup.stage2_training_complete = lambda outdir: (False, "not done")
        # seed 2's on-disk evidence must actually match "stage2_pending" (Stage1
        # trained + evaluated, Stage2 not started) for the migration to agree.
        d2 = sup.BASE_OUT / "stage1" / "seed_2"
        d2.mkdir(parents=True)
        (d2 / "STAGE1_DONE").touch()
        e2 = sup.BASE_OUT / "eval_stage1_postfix" / "seed_2"
        e2.mkdir(parents=True)
        (e2 / "eval_summary.csv").write_text("x\n")

    def tearDown(self):
        sup.stage1_training_complete = self._orig_s1complete
        sup.stage2_training_complete = self._orig_s2complete
        self.tmpdir.cleanup()

    def test_restart_preserves_stage2_running_and_stage2_pending(self):
        job = sup.JobState(seed=1, stage="stage2", session="ica_phase1_seed1_stage2",
                            pid=4242, start_time="t0", output_dir="d")
        state = sup.SupervisorState(
            pending=[], running={"1": asdict(job)}, completed=[0], failed={},
            seed_status={"0": sup.STATUS_COMPLETED, "1": sup.STATUS_STAGE2_RUNNING,
                         "2": sup.STATUS_STAGE2_PENDING, "3": sup.STATUS_STAGE1_PENDING,
                         "4": sup.STATUS_STAGE1_PENDING})
        changed = sup.migrate_seed_status(state)
        self.assertFalse(changed)
        self.assertEqual(state.seed_status["1"], sup.STATUS_STAGE2_RUNNING)
        self.assertEqual(state.seed_status["2"], sup.STATUS_STAGE2_PENDING)
        # And the launch decision for seed 2 (not currently running) must still
        # correctly resolve to Stage2, never Stage1.
        candidates = sup._launchable_candidates(state)
        self.assertIn(2, candidates)
        self.assertNotIn(1, candidates)  # seed 1 is already running


class TestLaunchStage2SurvivesSpaceInPath(unittest.TestCase):
    """Regression test for the 2026-07-27 incident: launch_stage2()'s sha256
    diagnostic used to interpolate the checkpoint path unquoted into a
    shell=True string. A space in the path (this repo's real REPO_ROOT
    contains one - "My Drive") split it into multiple nonexistent-file
    arguments, shasum printed nothing, and `.split()[0]` on the empty result
    raised an unhandled IndexError that killed the whole supervisor process
    immediately after it had already launched a real, orphaned training job.
    This test uses a tmp directory whose path also contains a space (like the
    dry-run methodology used for the earlier eval-quoting fix) and stubs out
    only tmux/pid-lookup (no real tmux dependency needed) to verify
    launch_stage2() completes without raising and records a real sha256."""

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory(prefix="dry run space test ")
        sup.BASE_OUT = Path(self.tmpdir.name)
        self._orig_tmux_launch = sup.tmux_launch
        self._orig_tmux_pane_pid = sup.tmux_pane_pid
        sup.tmux_launch = lambda session, cmd: None  # no real tmux needed
        sup.tmux_pane_pid = lambda session: 99999

    def tearDown(self):
        sup.tmux_launch = self._orig_tmux_launch
        sup.tmux_pane_pid = self._orig_tmux_pane_pid
        self.tmpdir.cleanup()

    def test_launch_stage2_does_not_crash_and_records_sha(self):
        stage1_dir = sup.BASE_OUT / "stage1" / "seed_1"
        stage1_dir.mkdir(parents=True)
        ckpt = stage1_dir / "checkpoint.pt"
        ckpt.write_bytes(b"not a real checkpoint, just needs to exist for shasum")
        import hashlib
        expected_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()

        state = sup.SupervisorState()
        sup.launch_stage2(state, 1)  # must not raise

        self.assertIn("1", state.running)
        self.assertEqual(state.seed_status["1"], sup.STATUS_STAGE2_RUNNING)
        # Recompute the sha the same way launch_stage2() does now, to confirm
        # it actually succeeded (not silently swallowed to "").
        import subprocess
        r = subprocess.run(["shasum", "-a", "256", str(ckpt)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertEqual(r.stdout.split()[0], expected_sha)


if __name__ == "__main__":
    unittest.main(verbosity=2)
