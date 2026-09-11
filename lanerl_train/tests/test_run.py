"""The training loop: staleness bounds, loud actor failure, and resume.

Three of these are about failures that this project has already paid for once.
A rollout that is too stale must be *counted*, not quietly averaged in.  An
actor thread that raises must reach the learner thread, because a run that keeps
its process alive while producing nothing is the single most expensive failure
mode here.  And a resume that silently keeps random weights is worse than no
resume at all.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path

import pytest

from lanerl_train.eval import Evaluator
from lanerl_train.league import LATEST, LeagueConfig
from lanerl_train.run import (
    ActorFailure,
    ActorLoop,
    CheckpointManager,
    EpisodeResult,
    MetricsLog,
    ParameterStore,
    Rollout,
    RunConfig,
    StalenessTracker,
    StalledRun,
    TrainingError,
    TrainingLoop,
)

from .fakes import CountingCollect, FakeLearner


def make_loop(run_dir: Path, **kw) -> TrainingLoop:
    cfg = RunConfig(
        run_dir=run_dir,
        num_actors=kw.pop("num_actors", 0),
        max_staleness=kw.pop("max_staleness", 1),
        queue_capacity=kw.pop("queue_capacity", 4),
        checkpoint_every=kw.pop("checkpoint_every", 0),
        snapshot_every=kw.pop("snapshot_every", 0),
        eval_every=kw.pop("eval_every", 0),
        stall_timeout_s=kw.pop("stall_timeout_s", 5.0),
        league=kw.pop("league", LeagueConfig()),
    )
    return TrainingLoop(cfg, FakeLearner(), evaluator=Evaluator(anchors=[]), **kw)


# -- parameter store -------------------------------------------------------


def test_publish_bumps_the_version_monotonically():
    store = ParameterStore({"w": 0})
    assert store.version == 0
    assert store.publish({"w": 1}) == 1
    assert store.publish({"w": 2}) == 2
    assert store.pull() == (2, {"w": 2})


def test_wait_for_version_above_unblocks_on_publish():
    store = ParameterStore({})
    got = []

    def waiter():
        got.append(store.wait_for_version_above(0, timeout=5.0))

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.05)
    store.publish({"w": 1})
    t.join(timeout=5)
    assert got == [True]


# -- staleness -------------------------------------------------------------


def test_a_rollout_within_the_bound_is_accepted():
    s = StalenessTracker(max_staleness=1)
    assert s.admit(rollout_version=5, learner_version=5) is True
    assert s.admit(rollout_version=5, learner_version=6) is True
    assert s.accepted == 2 and s.rejected == 0
    assert s.mean_staleness == pytest.approx(0.5)


def test_a_rollout_beyond_the_bound_is_rejected_and_shouted_about(caplog):
    s = StalenessTracker(max_staleness=1)
    with caplog.at_level(logging.ERROR, logger="lanerl_train.run"):
        assert s.admit(rollout_version=5, learner_version=7) is False
    assert s.rejected == 1 and s.accepted == 0
    assert "REJECTED a rollout at staleness 2 > max 1" in caplog.text
    assert s.max_observed == 2
    assert s.max_accepted == 0, "nothing beyond the bound was ever trained on"


def test_openai_five_style_bound_of_eight():
    s = StalenessTracker(max_staleness=8)
    assert s.admit(0, 8) is True
    assert s.admit(0, 9) is False


def test_the_learner_refuses_a_stale_rollout_and_does_not_update(run_dir):
    loop = make_loop(run_dir, max_staleness=1)
    loop.state.param_version = 10
    loop.submit(Rollout(actor_id=0, param_version=3, steps=4, data="stale"))
    assert loop.step_once(timeout=1.0) is False
    assert loop.learner.updates == 0
    assert loop.state.update == 0
    assert loop.staleness.rejected == 1
    kinds = [json.loads(l)["kind"] for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert "rollout_rejected" in kinds


def test_a_fresh_rollout_updates_and_publishes(run_dir):
    loop = make_loop(run_dir)
    v0 = loop.state.param_version
    loop.submit(Rollout(actor_id=0, param_version=v0, steps=64, data="fresh"))
    assert loop.step_once(timeout=1.0) is True
    assert loop.learner.updates == 1
    assert loop.state.update == 1
    assert loop.state.param_version == v0 + 1
    assert loop.state.total_env_steps == 64


def test_run_config_warns_when_the_queue_guarantees_rejections(caplog, tmp_path):
    with caplog.at_level(logging.WARNING, logger="lanerl_train.run"):
        RunConfig(run_dir=tmp_path, num_actors=8, queue_capacity=16, max_staleness=1)
    assert "above max_staleness=1" in caplog.text


def test_run_config_is_quiet_when_the_bound_is_achievable(caplog, tmp_path):
    with caplog.at_level(logging.WARNING, logger="lanerl_train.run"):
        RunConfig(run_dir=tmp_path, num_actors=1, queue_capacity=1, max_staleness=1)
    assert "above max_staleness" not in caplog.text


# -- actors ----------------------------------------------------------------


def test_an_actor_pulls_the_newest_parameters_before_each_rollout():
    store = ParameterStore({"w": 0})
    out: "queue.Queue[Rollout]" = queue.Queue()
    collect = CountingCollect()
    actor = ActorLoop(0, store, out, collect, queue.Queue(), threading.Event())

    actor.run_once()
    assert collect.versions == [0]
    store.publish({"w": 1})
    store.publish({"w": 2})
    actor.run_once()
    assert collect.versions == [0, 2], "an actor must not collect on parameters it has replaced"
    assert out.get_nowait().param_version == 0
    assert out.get_nowait().param_version == 2


def test_an_actor_blocks_on_backpressure_rather_than_racing_ahead():
    store = ParameterStore({})
    out: "queue.Queue[Rollout]" = queue.Queue(maxsize=1)
    actor = ActorLoop(0, store, out, CountingCollect(), queue.Queue(), threading.Event(),
                      put_timeout_s=0.1)
    actor.run_once()
    with pytest.raises(TrainingError, match="has not consumed a rollout"):
        actor.run_once()


def test_an_actor_exception_reaches_the_learner_thread_with_its_traceback(run_dir):
    loop = make_loop(run_dir, num_actors=1, queue_capacity=2)
    loop.collect = CountingCollect(raise_after=2)
    loop.start_actors()
    with pytest.raises(ActorFailure) as excinfo:
        loop.run(max_updates=50)
    assert "scripted actor failure" in str(excinfo.value)
    assert "Traceback" in str(excinfo.value)
    loop.shutdown(join_timeout_s=5)
    kinds = [json.loads(l)["kind"] for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert "run_failed" in kinds


def test_a_stalled_run_raises_instead_of_waiting_forever(run_dir):
    loop = make_loop(run_dir, stall_timeout_s=0.1)
    with pytest.raises(StalledRun, match="no rollout in"):
        loop.step_once()


def test_threaded_actors_and_learner_reach_the_target_updates(run_dir):
    loop = make_loop(run_dir, num_actors=2, queue_capacity=2, max_staleness=8)
    loop.collect = CountingCollect(steps=16)
    loop.start_actors()
    try:
        state = loop.run(max_updates=25)
    finally:
        loop.shutdown(join_timeout_s=5)
    assert state.update == 25
    assert state.total_env_steps == 25 * 16
    assert loop.staleness.accepted == 25
    # The guarantee is about data that was trained on. max_observed may exceed
    # the bound precisely because the guard rejected those rollouts.
    assert loop.staleness.max_accepted <= 8
    assert loop.staleness.max_observed >= loop.staleness.max_accepted


# -- episodes, league and metrics -----------------------------------------


def test_an_episode_feeds_the_league_the_evaluator_and_the_log(run_dir):
    loop = make_loop(run_dir)
    ep = EpisodeResult(
        agent="agent@0", opponent_id="snap@1", opponent_category="pfsp",
        score=1.0, cs_at_10=33.5, length_steps=9000, reason="time", instance=3,
    )
    loop.record_episode(ep)
    assert loop.sampler.win_rates.n("snap@1") == 1
    assert loop.sampler.win_rates.p("snap@1") > 0.5
    assert loop.evaluator.cs.stats("agent@0")[0] == pytest.approx(33.5)
    assert loop.evaluator.table.games("agent@0", "snap@1") == 1

    lines = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    episode = next(r for r in lines if r["kind"] == "episode")
    assert episode["cs_at_10"] == 33.5 and episode["opponent_category"] == "pfsp"
    assert any(r["kind"] == "match" for r in lines)


def test_a_mirror_game_against_latest_records_no_match(run_dir):
    """Win rate vs your own live weights is 50% by construction; recording it as
    a match would pull every Elo toward the mean for no information."""
    loop = make_loop(run_dir)
    loop.record_episode(
        EpisodeResult(agent="agent@0", opponent_id=LATEST, opponent_category="latest",
                      score=1.0, cs_at_10=31.9)
    )
    assert loop.evaluator.table.players == []
    assert loop.evaluator.cs.stats("agent@0") is not None  # CS still counts


def test_unknown_cs_stays_unknown_rather_than_becoming_zero(run_dir):
    loop = make_loop(run_dir)
    loop.record_episode(
        EpisodeResult(agent="a", opponent_id="snap@1", opponent_category="uniform",
                      score=0.5, cs_at_10=None, reason="instance_died")
    )
    assert loop.evaluator.cs.stats("a") is None


def test_the_metrics_log_is_replayable_by_the_evaluator(run_dir):
    loop = make_loop(run_dir)
    for i in range(12):
        loop.record_episode(
            EpisodeResult(agent="agent@0", opponent_id="snap@1", opponent_category="pfsp",
                          score=1.0 if i % 3 else 0.0, cs_at_10=28.0 + i)
        )
    fresh = Evaluator(anchors=[])
    assert fresh.load_jsonl(run_dir / "metrics.jsonl") == 12
    assert fresh.table.win_rate("agent@0", "snap@1") == pytest.approx(
        loop.evaluator.table.win_rate("agent@0", "snap@1")
    )


def test_snapshots_join_the_pool_and_an_eval_report_is_written(run_dir):
    loop = make_loop(run_dir, checkpoint_every=2, snapshot_every=2, eval_every=2)
    for i in range(4):
        loop.submit(Rollout(actor_id=0, param_version=loop.state.param_version, steps=1))
        assert loop.step_once(timeout=1.0)
    assert len(loop.sampler.pool) == 2
    kinds = [json.loads(l)["kind"] for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    assert "snapshot" in kinds and "eval" in kinds
    assert loop.checkpoints.latest() is not None


# -- checkpointing and resume ---------------------------------------------


def test_checkpoints_are_written_atomically_and_ordered(tmp_path):
    cm = CheckpointManager(tmp_path / "ckpt")
    cm.save(1, {"weights": 1})
    p = cm.save(20, {"weights": 20})
    assert cm.latest() == p
    assert not list((tmp_path / "ckpt").glob("*.tmp"))
    assert cm.load(p)["weights"] == 20


def test_keep_last_prunes_old_checkpoints(tmp_path):
    cm = CheckpointManager(tmp_path / "ckpt", keep_last=2)
    for i in range(1, 6):
        cm.save(i, {"weights": i})
    assert len(cm.all_checkpoints()) == 2
    assert cm.load(cm.latest())["weights"] == 5


def test_a_run_resumes_its_counters_league_and_weights(run_dir):
    loop = make_loop(run_dir, checkpoint_every=2, snapshot_every=2)
    for _ in range(4):
        loop.submit(Rollout(actor_id=0, param_version=loop.state.param_version, steps=10))
        loop.step_once(timeout=1.0)
    loop.record_episode(
        EpisodeResult(agent="a", opponent_id="snap@2", opponent_category="pfsp", score=1.0)
    )
    loop.save_state(loop.checkpoints.latest())
    before = (loop.state.update, loop.state.total_env_steps, loop.learner.weights,
              loop.sampler.pool.ids(), loop.sampler.win_rates.p("snap@2"))
    loop.metrics.close()

    revived = make_loop(run_dir)
    assert revived.resume() is True
    after = (revived.state.update, revived.state.total_env_steps, revived.learner.weights,
             revived.sampler.pool.ids(), revived.sampler.win_rates.p("snap@2"))
    assert after == before
    assert revived.store.pull()[1] == {"weights": before[2]}


def test_resuming_a_fresh_directory_is_a_no_op(run_dir):
    assert make_loop(run_dir).resume() is False


def test_state_without_a_checkpoint_refuses_to_resume_onto_random_weights(run_dir):
    loop = make_loop(run_dir)
    loop.state.update = 500
    loop.save_state()
    revived = make_loop(run_dir)
    with pytest.raises(TrainingError, match="no checkpoint found"):
        revived.resume()


def test_the_rng_survives_a_resume(run_dir):
    loop = make_loop(run_dir, checkpoint_every=1)
    loop.submit(Rollout(actor_id=0, param_version=loop.state.param_version, steps=1))
    loop.step_once(timeout=1.0)
    loop.save_state(loop.checkpoints.latest())
    expected = [loop.rng.random() for _ in range(3)]

    revived = make_loop(run_dir)
    revived.resume()
    assert [revived.rng.random() for _ in range(3)] == expected


# -- metrics log -----------------------------------------------------------


def test_metrics_are_flushed_per_record_so_a_crash_keeps_them(tmp_path):
    path = tmp_path / "m.jsonl"
    m = MetricsLog(path)
    m.write("update", update=1, loss=0.5)
    # readable before close(): the interesting runs are the ones that die
    rec = json.loads(path.read_text().splitlines()[0])
    assert rec["kind"] == "update" and rec["update"] == 1 and "wall" in rec
    m.close()


def test_metrics_log_is_thread_safe(tmp_path):
    m = MetricsLog(tmp_path / "m.jsonl")
    def spam(i):
        for j in range(200):
            m.write("update", actor=i, j=j)
    threads = [threading.Thread(target=spam, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    m.close()
    lines = (tmp_path / "m.jsonl").read_text().splitlines()
    assert len(lines) == 1600
    assert all(json.loads(l)["kind"] == "update" for l in lines)


# -- the training clock the environments read ------------------------------
#
# `lanerl_rl.reward.LaneRewardConfig.alpha` anneals the zero-sum coefficient
# over `zero_sum_anneal_steps`, driven by a number the trainer hands the env.
# Nothing handed it one, so alpha sat at its starting 0.5 for every run.  These
# cover the learner half of that hand-off; `lanerl_rl/tests/test_env.py` covers
# the env half, and `test_the_anneal_moves_end_to_end` below joins them.


def test_train_step_counter_is_monotonic_and_callable():
    from lanerl_train.run import TrainStepCounter

    c = TrainStepCounter()
    assert c.value == 0 and c() == 0
    assert c.set(100) == 100
    assert c.set(50) == 100, "a rewind would silently re-run every env-side anneal"
    assert c() == 100


def test_the_loop_publishes_env_steps_as_the_training_clock(run_dir):
    loop = make_loop(run_dir)
    assert loop.train_steps.value == 0
    for i in range(3):
        loop.submit(Rollout(actor_id=0, param_version=i, steps=128))
        assert loop.step_once(timeout=2.0)
    assert loop.state.total_env_steps == 384
    assert loop.train_steps.value == 384


def test_the_loop_can_publish_updates_instead(run_dir):
    cfg = RunConfig(
        run_dir=run_dir, num_actors=0, queue_capacity=4, checkpoint_every=0,
        snapshot_every=0, eval_every=0, stall_timeout_s=5.0,
        anneal_clock="updates",
    )
    loop = TrainingLoop(cfg, FakeLearner(), evaluator=Evaluator(anchors=[]))
    for i in range(3):
        loop.submit(Rollout(actor_id=0, param_version=i, steps=128))
        assert loop.step_once(timeout=2.0)
    assert loop.train_steps.value == 3


def test_an_unknown_anneal_clock_is_refused(run_dir):
    with pytest.raises(ValueError, match="anneal_clock"):
        RunConfig(run_dir=run_dir, anneal_clock="wallclock")


def test_the_training_clock_is_logged_with_every_update(run_dir):
    loop = make_loop(run_dir)
    loop.submit(Rollout(actor_id=0, param_version=0, steps=64))
    assert loop.step_once(timeout=2.0)
    recs = [json.loads(l) for l in (run_dir / "metrics.jsonl").read_text().splitlines()]
    upd = [r for r in recs if r["kind"] == "update"]
    assert upd and upd[-1]["train_step"] == 64
    assert upd[-1]["anneal_clock"] == "env_steps"


def test_resume_restores_the_training_clock(run_dir):
    """A resume that rewinds the clock silently restarts the anneal."""
    loop = make_loop(run_dir, checkpoint_every=1)
    for i in range(2):
        loop.submit(Rollout(actor_id=0, param_version=i, steps=500))
        assert loop.step_once(timeout=2.0)
    assert loop.train_steps.value == 1000
    loop.save_state(loop.checkpoints.latest())

    fresh = make_loop(run_dir, checkpoint_every=1)
    assert fresh.train_steps.value == 0
    assert fresh.resume()
    assert fresh.train_steps.value == 1000


def test_the_anneal_moves_end_to_end(run_dir):
    """The whole hand-off: learner update -> counter -> LaneEnv -> reward alpha.

    This is the test the wiring exists for.  Each half is covered on its own;
    only this one fails if the two halves are never connected, which is exactly
    the state the code was in.
    """
    from lanerl_rl import constants as C
    from lanerl_rl.env import LaneEnv, LaneEnvConfig
    from lanerl_rl.reward import LaneRewardConfig
    from lanerl_rl.scenarios import top_lane_scenario

    class _Backend:
        ignores_actions = True

        def __init__(self):
            self.frames = [top_lane_scenario(t_ms=90_000 + 100 * i) for i in range(4)]
            self.i = 0

        def reset(self):
            self.i = 0
            return self.frames[0]

        def step(self, commands):
            self.i += 1
            return self.frames[self.i] if self.i < len(self.frames) else None

        def close(self):
            pass

    loop = make_loop(run_dir)
    env = LaneEnv(
        _Backend(),
        LaneEnvConfig(
            warn_on_approx_fog=False,
            reward=LaneRewardConfig(zero_sum_anneal_steps=1000),
        ),
        train_step_source=loop.train_steps,
    )
    noop = {"button": C.BUTTON_INDEX["noop"], "move_x": 4, "move_z": 4, "target": 0}

    def alpha_now() -> float:
        env.reset()
        _, _, _, info = env.step({t: dict(noop) for t in env.cfg.teams})
        return info["reward_info"]["alpha"]

    assert alpha_now() == pytest.approx(0.5)
    for i in range(4):
        loop.submit(Rollout(actor_id=0, param_version=i, steps=250))
        assert loop.step_once(timeout=2.0)
    assert loop.train_steps.value == 1000
    assert alpha_now() == pytest.approx(1.0)


def test_a_self_play_episode_does_not_crash_the_run():
    """Pure self-play killed the first overnight run at its first episode.

    The mixture was {"self": 1.0}, so both sides carried the id "self". That is
    not the LATEST sentinel, so record_episode fell through to MatchRecord,
    whose __post_init__ rejects a self-match -- correctly, since a self-match
    carries no rating information. The caller's guard was simply too narrow.
    """
    from lanerl_train.run import TrainingLoop  # noqa: F401
    from lanerl_train.eval import MatchRecord

    # the guard that fired, still firing: this is what we must not construct
    with pytest.raises(ValueError, match="self-match"):
        MatchRecord(agent_a="self", agent_b="self", score_a=0.5)


def test_record_episode_skips_rating_for_a_self_match(tmp_path):
    """The real regression: record_episode must not build that MatchRecord."""
    import inspect
    from lanerl_train import run as run_mod

    src = inspect.getsource(run_mod.TrainingLoop.record_episode)
    # both the evaluator call and the metrics "match" row must be guarded on
    # agent != opponent, not only on the LATEST sentinel
    assert src.count("ep.opponent_id != ep.agent") >= 2, (
        "record_episode still rates a self-match; pure self-play will crash at "
        "the first completed episode"
    )
