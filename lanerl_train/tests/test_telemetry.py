"""Throughput and resource telemetry: can metrics.jsonl alone diagnose a run?

The bar every test here is written against: take only ``metrics.jsonl`` from a
finished run and answer "was this compute-bound, env-bound or stalled?".  The
first real run could not be asked that question at all, which is why it took
16,200 updates to notice nothing was happening.

The clock is injected everywhere, so these assert exact rates rather than
sleeping and hoping.
"""

from __future__ import annotations

import json

import pytest

from lanerl_train.run import (
    GpuProbe,
    MetricsLog,
    Rollout,
    RunConfig,
    ThroughputMeter,
    TrainingLoop,
)

from .fakes import FakeLearner


class FakeClock:
    """A monotonic clock a test drives by hand."""

    def __init__(self, start: float = 1000.0):
        self.t = float(start)

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> float:
        self.t += float(dt)
        return self.t


# -- ThroughputMeter -------------------------------------------------------


def test_rates_are_over_the_rolling_window_not_the_whole_run():
    clock = FakeClock()
    m = ThroughputMeter(window=4, clock=clock)
    # Ten slow updates, then four fast ones. A lifetime mean would still look
    # slow; the whole point of the window is that it does not.
    for _ in range(10):
        clock.advance(10.0)
        m.record(100)
    for _ in range(4):
        clock.advance(1.0)
        row = m.record(100)
    assert row["window_updates"] == 4
    assert row["updates_per_s"] == pytest.approx(1.0)
    assert row["env_steps_per_s"] == pytest.approx(100.0)
    # The lifetime figure, for contrast, is still dominated by the slow phase.
    assert m.totals()["mean_updates_per_s"] == pytest.approx(14 / 104.0)


def test_decisions_per_second_counts_parallel_envs_and_env_steps_does_not():
    """One buffer row is one decision in every slot; the two must not be conflated."""
    clock = FakeClock()
    m = ThroughputMeter(window=8, clock=clock)
    for _ in range(4):
        clock.advance(2.0)
        row = m.record(255, parallel_envs=8)
    assert row["env_steps_per_s"] == pytest.approx(255 / 2.0)
    assert row["decisions_per_s"] == pytest.approx(255 * 8 / 2.0)
    assert row["total_decisions"] == 255 * 8 * 4


def test_learner_frac_separates_compute_bound_from_env_bound():
    clock = FakeClock()
    m = ThroughputMeter(window=8, clock=clock)

    with m.waiting():
        clock.advance(9.0)
    with m.learning():
        clock.advance(1.0)
    env_bound = m.record(100)
    assert env_bound["learner_frac"] == pytest.approx(0.1)
    assert env_bound["wait_s"] == pytest.approx(9.0)

    with m.waiting():
        clock.advance(0.1)
    with m.learning():
        clock.advance(9.9)
    compute_bound = m.record(100)
    assert compute_bound["learner_frac"] == pytest.approx(0.99)


def test_a_rejected_rollouts_wait_is_billed_separately_not_as_productive_wait():
    """Otherwise a run that throws data away reads as merely env-bound.

    The 31 s still has to be somewhere: the three buckets must account for the
    whole interval, or the log loses the most expensive thing in the run.
    """
    clock = FakeClock()
    m = ThroughputMeter(window=8, clock=clock)
    with m.waiting():
        clock.advance(30.0)
    m.discard_pending()
    with m.waiting():
        clock.advance(1.0)
    with m.learning():
        clock.advance(1.0)
    row = m.record(100)
    assert row["wait_s"] == pytest.approx(1.0)
    assert row["rejected_s"] == pytest.approx(30.0)
    assert row["update_s"] == pytest.approx(32.0)
    assert row["wait_s"] + row["learner_s"] + row["rejected_s"] == pytest.approx(
        row["update_s"]
    )
    # And the buckets reset, so the next update does not inherit them.
    with m.learning():
        clock.advance(2.0)
    nxt = m.record(100)
    assert nxt["rejected_s"] == 0.0
    assert nxt["learner_frac"] == pytest.approx(1.0)


def test_window_must_be_at_least_one_update():
    with pytest.raises(ValueError, match="window must be"):
        ThroughputMeter(window=0)


# -- GpuProbe --------------------------------------------------------------


def test_gpu_probe_returns_the_full_key_set_even_with_no_gpu():
    """A missing key breaks a downstream reader; an explicit None does not."""
    probe = GpuProbe(device="cpu")
    sample = probe.sample()
    assert set(sample) == {"util_pct", "mem_allocated_mb", "mem_reserved_mb"}
    assert not probe.enabled
    assert sample == {"util_pct": None, "mem_allocated_mb": None, "mem_reserved_mb": None}


def test_a_raising_utilization_probe_is_disabled_not_retried(monkeypatch):
    """pynvml is not installed on either node, so this is the REAL path here."""
    calls = {"util": 0, "mem": 0}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def utilization():
            calls["util"] += 1
            raise ModuleNotFoundError("pynvml does not seem to be installed")

        @staticmethod
        def memory_allocated():
            calls["mem"] += 1
            return 512 * 1024 * 1024

        @staticmethod
        def memory_reserved():
            return 1024 * 1024 * 1024

    probe = GpuProbe(device="cuda", utilization_every=1)
    probe._torch = type("T", (), {"cuda": FakeCuda})
    probe._enabled = True

    rows = [probe.sample() for _ in range(20)]
    assert calls["util"] == 1, "a probe that raises must be tried once, not every update"
    assert calls["mem"] == 20
    assert all(r["util_pct"] is None for r in rows)
    assert all(r["mem_allocated_mb"] == pytest.approx(512.0) for r in rows)


def test_utilization_is_sampled_on_a_cadence_and_cached_in_between():
    calls = {"util": 0}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def utilization():
            calls["util"] += 1
            return 5.0

        @staticmethod
        def memory_allocated():
            return 0

        @staticmethod
        def memory_reserved():
            return 0

    probe = GpuProbe(device="cuda", utilization_every=10)
    probe._torch = type("T", (), {"cuda": FakeCuda})
    probe._enabled = True
    rows = [probe.sample() for _ in range(25)]
    assert calls["util"] == 3  # updates 0, 10, 20
    assert all(r["util_pct"] == 5.0 for r in rows)


# -- end to end through the metrics file -----------------------------------


def read_updates(path):
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if json.loads(line)["kind"] == "update"
    ]


def test_metrics_jsonl_alone_answers_compute_bound_or_env_bound(run_dir):
    clock = FakeClock()
    learner = FakeLearner()
    cfg = RunConfig(run_dir=run_dir, num_actors=0, checkpoint_every=0, eval_every=0)
    loop = TrainingLoop(
        cfg,
        learner,
        metrics=MetricsLog(run_dir / "metrics.jsonl"),
        throughput=ThroughputMeter(window=4, clock=clock),
        gpu=GpuProbe(device="cpu"),
    )

    # A learner that burns clock time, against a queue that is never empty:
    # this is the compute-bound shape.
    def slow_update(batch):
        clock.advance(4.0)
        return learner.__class__.update(learner, batch)

    loop.learner.update = slow_update  # type: ignore[method-assign]
    for i in range(6):
        loop.submit(Rollout(actor_id=0, param_version=i, steps=255, parallel_envs=8))
        clock.advance(0.1)  # the queue.get() itself
        loop.step_once(timeout=1.0)

    rows = read_updates(run_dir / "metrics.jsonl")
    assert len(rows) == 6
    last = rows[-1]
    assert last["throughput/learner_frac"] > 0.9, "should read as compute-bound"
    assert last["throughput/decisions_per_s"] == pytest.approx(255 * 8 / 4.1)
    assert last["throughput/wall_s"] == pytest.approx(6 * 4.1)
    assert last["parallel_envs"] == 8
    # Every GPU key present and explicitly null on CPU.
    assert last["gpu/util_pct"] is None
    assert "gpu/mem_allocated_mb" in last


def test_every_update_row_states_rows_and_decisions_separately(run_dir):
    """`total_env_steps` is rows. Nothing in the row used to say so.

    The first run's state.json reads ``total_env_steps: 4131000``, which is a
    perfectly plausible sample count and is not one: with 2 sides x 4
    instances those rows are 33,048,000 decisions.  Both numbers are logged
    now, and ``total_env_steps`` keeps its meaning because the reward's
    zero-sum anneal is denominated in it.
    """
    clock = FakeClock()
    cfg = RunConfig(run_dir=run_dir, num_actors=0, checkpoint_every=0, eval_every=0)
    loop = TrainingLoop(
        cfg,
        FakeLearner(),
        metrics=MetricsLog(run_dir / "metrics.jsonl"),
        throughput=ThroughputMeter(window=4, clock=clock),
        gpu=GpuProbe(device="cpu"),
    )
    for i in range(3):
        loop.submit(Rollout(actor_id=0, param_version=i, steps=255, parallel_envs=8))
        clock.advance(1.0)
        loop.step_once(timeout=1.0)
    last = read_updates(run_dir / "metrics.jsonl")[-1]
    assert last["total_env_steps"] == 255 * 3
    assert last["total_env_rows"] == 255 * 3
    assert last["total_decisions"] == 255 * 8 * 3
    assert last["throughput/total_decisions"] == last["total_decisions"]
    assert loop.state.total_decisions == 255 * 8 * 3


def test_the_decision_count_survives_a_resume(run_dir):
    """It restarted at zero on each of the first run's seven resumes."""
    clock = FakeClock()
    cfg = RunConfig(run_dir=run_dir, num_actors=0, checkpoint_every=1, eval_every=0)
    learner = FakeLearner()
    loop = TrainingLoop(
        cfg,
        learner,
        metrics=MetricsLog(run_dir / "metrics.jsonl"),
        throughput=ThroughputMeter(window=4, clock=clock),
        gpu=GpuProbe(device="cpu"),
    )
    for i in range(3):
        loop.submit(Rollout(actor_id=0, param_version=i, steps=255, parallel_envs=8))
        clock.advance(1.0)
        loop.step_once(timeout=1.0)
    loop.save_state(loop.checkpoints.save(loop.state.update, learner.state_payload()))
    before = loop.state.total_decisions

    clock2 = FakeClock()
    resumed = TrainingLoop(
        cfg,
        FakeLearner(),
        metrics=MetricsLog(run_dir / "metrics.jsonl"),
        throughput=ThroughputMeter(window=4, clock=clock2),
        gpu=GpuProbe(device="cpu"),
    )
    assert resumed.resume() is True
    assert resumed.state.total_decisions == before
    resumed.submit(Rollout(actor_id=0, param_version=99, steps=255, parallel_envs=8))
    clock2.advance(1.0)
    resumed.step_once(timeout=1.0)
    last = read_updates(run_dir / "metrics.jsonl")[-1]
    assert last["total_decisions"] == before + 255 * 8
    assert last["throughput/total_decisions"] == last["total_decisions"]


def test_shutdown_record_carries_the_lifetime_throughput(run_dir):
    clock = FakeClock()
    cfg = RunConfig(run_dir=run_dir, num_actors=0, checkpoint_every=0, eval_every=0)
    loop = TrainingLoop(
        cfg,
        FakeLearner(),
        metrics=MetricsLog(run_dir / "metrics.jsonl"),
        throughput=ThroughputMeter(window=4, clock=clock),
        gpu=GpuProbe(device="cpu"),
    )
    for i in range(3):
        loop.submit(Rollout(actor_id=0, param_version=i, steps=10, parallel_envs=2))
        clock.advance(1.0)
        loop.step_once(timeout=1.0)
    loop.shutdown()
    rec = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text().splitlines()
        if json.loads(line)["kind"] == "shutdown"
    ][0]
    assert rec["total_wall_s"] == pytest.approx(3.0)
    assert rec["mean_decisions_per_s"] == pytest.approx(60 / 3.0)


def test_the_server_rate_and_the_ppo_rate_cannot_disagree():
    """They did, for the whole first run.

    ServerLaunchSpec.step_ticks was hardcoded 4 (15 Hz) while
    lanerl_rl.constants.STEP_TICKS said 2 (30 Hz). PPOConfig derives gamma from
    the constant, so `--horizon-s 30` silently bought a 60 s horizon, and
    global_vec.dt_norm sat at a constant 2.0. Two sources of truth for one
    physical quantity is the bug; this test is the single source.
    """
    from lanerl_train.vec import ServerLaunchSpec
    from lanerl_rl import constants as C
    from lanerl_rl.ppo import PPOConfig

    assert ServerLaunchSpec().step_ticks == C.STEP_TICKS
    assert PPOConfig().decision_hz == C.SERVER_TICK_HZ / ServerLaunchSpec().step_ticks
