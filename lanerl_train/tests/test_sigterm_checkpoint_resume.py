"""SIGTERM -> checkpoint -> resume, against the real ``python -m lanerl_train``
entrypoint and a real server instance -- not a unit test of the handler in
isolation.

Everyone who touched this today verified it by hand once: a clean run, a
mid-run SIGTERM, and a resume, each timed and eyeballed. That's real
evidence it works today, but it is not a regression test -- nothing stops a
future refactor of ``lanerl_train.__main__`` or ``TrainingLoop.shutdown``
from silently breaking the one property a preemptable ``gpup`` job actually
depends on: a SIGTERM produces a checkpoint fast enough to beat the 60s
preemption budget, and a resume from it picks up forward, not from zero.
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid

import pytest

from lanerl_train import paths

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not paths.server_available(),
        reason="server build not available (or LANERL_SKIP_SERVER_TESTS=1)",
    ),
]

# Generous margins: this asserts SIGTERM->exit is well inside gpup's 60s
# preemption budget, not that it matches the ~56ms measured by hand. Measured
# in practice: with rollout-steps=5 and checkpoint-every=1 on CPU, a real run
# reaches update 400+ within 90s of process start, so a first checkpoint well
# inside 45s is the realistic bound, not the ceiling.
BOOT_AND_CHECKPOINT_TIMEOUT_S = 45
SIGTERM_EXIT_BUDGET_S = 30
RESUME_PROGRESS_TIMEOUT_S = 45


def _run(run_name, *extra_args):
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.Popen(
        [
            sys.executable, "-m", "lanerl_train",
            "--run-name", run_name,
            "--device", "cpu",
            "--num-actors", "1",
            "--envs-per-actor", "1",
            "--rollout-steps", "5",
            "--checkpoint-every", "1",
            "--eval-every", "1000000",
            "--queue-capacity", "1",
            "--max-staleness", "5",
            "--seed", "0",
            *extra_args,
        ],
        cwd=str(paths.repo_root()),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _stop(proc, timeout_s=SIGTERM_EXIT_BUDGET_S):
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)


def _wait_for(predicate, timeout_s, what):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.5)
    pytest.fail(f"timed out after {timeout_s}s waiting for: {what}")


def _state(run_dir):
    p = run_dir / "state.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def _latest_checkpoint_update(run_dir):
    ckpts = sorted((run_dir / "checkpoints").glob("update_*"))
    if not ckpts:
        return None
    # update_00000003.pt -> 3
    return int(ckpts[-1].stem.split("_")[1])


@pytest.fixture
def run_name():
    name = f"sigterm-e2e-{uuid.uuid4().hex[:8]}"
    run_dir = paths.runs_root() / name
    assert not run_dir.exists(), f"{run_dir} already exists -- pick a fresh run name"
    yield name
    shutil.rmtree(run_dir, ignore_errors=True)


def test_sigterm_checkpoints_fast_and_resume_continues_forward(run_name):
    run_dir = paths.runs_root() / run_name
    proc = _run(run_name)
    try:
        _wait_for(
            lambda: (run_dir / "checkpoints").exists()
            and any((run_dir / "checkpoints").glob("update_*")),
            BOOT_AND_CHECKPOINT_TIMEOUT_S,
            "first checkpoint (real server boot + one rollout + one PPO update)",
        )

        proc.send_signal(signal.SIGTERM)
        try:
            rc = proc.wait(timeout=SIGTERM_EXIT_BUDGET_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
            pytest.fail(
                f"process did not exit within {SIGTERM_EXIT_BUDGET_S}s of SIGTERM -- "
                f"a real gpup preemption gives only 60s total"
            )
        assert rc == 0, f"SIGTERM path exited nonzero (rc={rc}); output:\n{proc.stdout.read()}"

        state_after_sigterm = _state(run_dir)
        assert state_after_sigterm is not None, "no state.json after SIGTERM -- resume has nothing to resume from"
        ckpt_after_sigterm = _latest_checkpoint_update(run_dir)
        assert ckpt_after_sigterm is not None
        assert state_after_sigterm["update"] == ckpt_after_sigterm, (
            f"state.json update={state_after_sigterm['update']} does not match the "
            f"checkpoint it points at (update_{ckpt_after_sigterm:08d}) -- a resume "
            f"would load weights for the wrong step"
        )
        assert state_after_sigterm["checkpoint"] is not None, "state.json was not told which checkpoint to resume from"

        no_orphan = subprocess.run(
            ["pgrep", "-f", f"--run-name {run_name}"], capture_output=True, text=True
        )
        assert no_orphan.stdout.strip() == "", (
            f"orphaned process(es) still alive after SIGTERM exit: {no_orphan.stdout!r}"
        )
    finally:
        _stop(proc)

    # -- resume: must pick up forward from the checkpoint, not restart at 0 --
    resumed = _run(run_name, "--resume")
    try:
        _wait_for(
            lambda: (_latest_checkpoint_update(run_dir) or -1) > ckpt_after_sigterm,
            RESUME_PROGRESS_TIMEOUT_S,
            f"a checkpoint past update_{ckpt_after_sigterm:08d} after --resume",
        )
    finally:
        _stop(resumed)

    final_state = _state(run_dir)
    assert final_state["update"] > state_after_sigterm["update"], (
        "resume did not advance past the pre-SIGTERM state -- either it restarted "
        "from scratch, or it never made progress"
    )
