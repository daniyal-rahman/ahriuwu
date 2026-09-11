"""Evaluation against a frozen opponent: the thing that was never wired up.

The bug being closed: ``lanerl_train.eval`` has had anchors, anchor win rates
and an Elo fit pinned to ``scripted_gold`` since before the first real run, and
nothing ever handed it a game.  Every eval report read ``(None, 0)`` against all
four anchors while the other number in it, ``score``, was 0.5 by construction.

So the tests here are about plumbing, not about the bot: does a run refuse to
start when its anchors cannot be played, does the driver really leave RED to the
in-server bot, does the outcome reach the evaluator, and does the evaluator's
actor get the CURRENT weights rather than whatever it was built with.  No game
server is involved -- ``FakeInstance`` speaks the same lockstep.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lanerl_rl import constants as C
from lanerl_train.anchor_eval import (
    AnchorEvalConfig,
    AnchorEvaluator,
    AnchorEvalError,
    anchor_launch_spec,
    play_anchor_episodes,
    score_for_reason,
)
from lanerl_train.eval import (
    AnchorConfigError,
    AnchorSpec,
    Evaluator,
    anchors_for_run,
    default_anchors,
    validate_anchors,
)
from lanerl_train.ports import InstancePorts
from lanerl_train.run import (
    EpisodeResult,
    MetricsLog,
    Rollout,
    RunConfig,
    TrainingError,
    TrainingLoop,
)
from lanerl_train.vec import EpisodeSpec, ServerLaunchSpec, SideAssignment, VecDriver, VecLaneEnv

from .fakes import FakeAdapter, FakeEncoder, FakeInstance, FakeLearner, FakePolicy


# -- startup validation ----------------------------------------------------


def test_the_three_scripted_anchors_are_playable_out_of_the_box():
    anchors = anchors_for_run()
    assert [a.id for a in anchors] == ["scripted_bronze", "scripted_gold", "scripted_diamond"]
    assert all(a.exists() for a in anchors)


def test_bc_policy_is_not_a_default_run_anchor_because_it_has_no_resource():
    """default_anchors() still describes all four; a RUN must not silently skip one."""
    assert "bc_policy" in [a.id for a in default_anchors()]
    assert "bc_policy" not in [a.id for a in anchors_for_run()]


def test_asking_for_bc_policy_without_a_checkpoint_fails_loudly():
    with pytest.raises(AnchorConfigError, match="no resource configured"):
        anchors_for_run(["scripted_gold", "bc_policy"])


def test_a_bc_checkpoint_that_does_not_exist_fails_loudly(tmp_path):
    with pytest.raises(AnchorConfigError, match="does not exist"):
        anchors_for_run(["bc_policy"], bc_checkpoint=tmp_path / "nope.pt")


def test_a_bc_checkpoint_that_does_exist_is_accepted(tmp_path):
    ckpt = tmp_path / "bc.pt"
    ckpt.write_bytes(b"weights")
    anchors = anchors_for_run(["scripted_gold", "bc_policy"], bc_checkpoint=ckpt)
    assert [a.id for a in anchors] == ["scripted_gold", "bc_policy"]


def test_an_unknown_anchor_name_is_an_error_not_a_silent_drop():
    with pytest.raises(AnchorConfigError, match="unknown anchor"):
        anchors_for_run(["scripted_platinum"])


def test_an_empty_anchor_list_is_refused():
    with pytest.raises(AnchorConfigError, match="nothing for evaluation to measure"):
        validate_anchors([])


def test_a_run_with_an_eval_cadence_and_no_anchor_evaluator_refuses_to_start(run_dir):
    cfg = RunConfig(run_dir=run_dir, num_actors=0, eval_every=400)
    loop = TrainingLoop(cfg, FakeLearner(), metrics=MetricsLog(run_dir / "m.jsonl"))
    with pytest.raises(TrainingError, match="no anchor evaluator"):
        loop.require_anchor_eval()


def test_turning_evaluation_off_deliberately_is_allowed(run_dir):
    cfg = RunConfig(run_dir=run_dir, num_actors=0, eval_every=0)
    loop = TrainingLoop(cfg, FakeLearner(), metrics=MetricsLog(run_dir / "m.jsonl"))
    loop.require_anchor_eval()  # must not raise


# -- the launch spec -------------------------------------------------------


def test_the_launch_spec_hands_red_to_the_anchors_own_bot_config():
    anchor = anchors_for_run(["scripted_bronze"])[0]
    spec = anchor_launch_spec(anchor, ServerLaunchSpec(step_ticks=2, toponly=True))
    assert spec.bot_teams == "purple"
    assert Path(spec.bot_config).name == "anchor_bronze.json"
    assert spec.step_ticks == 2, "the eval must run at the training decision rate"
    env = spec.environment(InstancePorts(0, 5000, 5001))
    assert env["LANERL_BOT"] == "purple"
    assert env["LANERL_BOT_CONFIG"].endswith("anchor_bronze.json")


def test_the_three_difficulties_really_are_different_bots():
    """A ladder whose rungs share a config is one rung wearing three hats."""
    configs = {}
    for a in anchors_for_run():
        configs[a.id] = json.loads(Path(anchor_launch_spec(a).bot_config).read_text())
    assert len({c["lastHitAccuracy"] for c in configs.values()}) == 3
    assert configs["scripted_bronze"]["reactionDelayMs"] > configs["scripted_diamond"]["reactionDelayMs"]


def test_a_policy_anchor_cannot_be_played_by_the_in_server_bot(tmp_path):
    ckpt = tmp_path / "bc.pt"
    ckpt.write_bytes(b"w")
    with pytest.raises(AnchorEvalError, match="only 'scripted' anchors"):
        anchor_launch_spec(AnchorSpec("bc_policy", "policy", ckpt))


# -- scoring ---------------------------------------------------------------


def test_the_agent_dying_is_a_loss_and_the_bot_dying_is_a_win():
    assert score_for_reason("death_team_100", C.TEAM_BLUE) == 0.0
    assert score_for_reason("death_team_200", C.TEAM_BLUE) == 1.0
    assert score_for_reason("death_team_100", C.TEAM_RED) == 1.0


def test_reaching_ten_minutes_alive_is_a_draw_not_a_loss():
    """It is the COMMON case here; folding it into a loss makes every rung 0."""
    assert score_for_reason("time", C.TEAM_BLUE) == 0.5
    assert score_for_reason("max_steps", C.TEAM_BLUE) == 0.5


# -- playing, with fakes ---------------------------------------------------


def build_anchor_driver(n=1, max_game_ms=400, instances=None, policy=None):
    """A VecDriver set up exactly as the anchor runner sets one up."""
    insts = instances or [FakeInstance(i, step_ms=66) for i in range(n)]
    ports = [InstancePorts(i, 47000 + 2 * i, 47001 + 2 * i) for i in range(n)]
    env = VecLaneEnv(n, ports=ports, factory=lambda i, p: insts[i], step_timeout_s=2.0)
    pol = policy or FakePolicy("self")
    driver = VecDriver(
        env,
        policies={"self": pol},
        adapter_factory=lambda i, side: FakeAdapter(i, side),
        encoder=FakeEncoder(),
        assignments=[SideAssignment(blue="self", red=None) for _ in range(n)],
        episode=EpisodeSpec(max_game_ms=max_game_ms),
    )
    driver.start()
    return driver, insts, pol


def test_red_is_never_driven_so_the_scripted_bot_keeps_its_own_orders():
    driver, insts, pol = build_anchor_driver(n=2, max_game_ms=10**9)
    driver.step()
    for inst in insts:
        assert "blue" in inst.last_action
        assert "red" not in inst.last_action, (
            "omitting the key is what leaves RED to LanerlControl's bot"
        )
    assert pol.batch_sizes == [2], "only the agent's side is in the policy batch"


def test_a_finished_game_becomes_a_scored_episode_result():
    driver, _insts, _pol = build_anchor_driver(n=1, max_game_ms=400)
    out = play_anchor_episodes(driver, "scripted_gold", "agent@7", n_episodes=1)
    assert len(out) == 1
    ep = out[0]
    assert ep.opponent_id == "scripted_gold"
    assert ep.opponent_category == "anchor"
    assert ep.agent == "agent@7"
    assert ep.reason == "time"
    assert ep.score == 0.5
    assert ep.length_steps > 0


def test_cs_at_10_is_absent_rather_than_zero_when_the_log_has_none():
    """Reporting 0 for an unmeasured CS drags the headline metric down silently."""
    driver, _insts, _pol = build_anchor_driver(n=1, max_game_ms=400)
    out = play_anchor_episodes(driver, "scripted_gold", "agent@1", n_episodes=1)
    assert out[0].cs_at_10 is None


def test_cs_at_10_is_read_for_a_game_that_reached_ten_minutes(monkeypatch):
    driver, _insts, _pol = build_anchor_driver(n=1, max_game_ms=400)
    monkeypatch.setattr(driver, "cs_at_10", lambda i: {C.TEAM_BLUE: 31, C.TEAM_RED: 29})
    out = play_anchor_episodes(driver, "scripted_gold", "agent@1", n_episodes=1)
    assert out[0].cs_at_10 == 31.0


def test_a_server_whose_clock_stops_fails_instead_of_hanging_the_learner():
    driver, _insts, _pol = build_anchor_driver(n=1, max_game_ms=10**9)
    with pytest.raises(AnchorEvalError, match="no episode against scripted_gold finished"):
        play_anchor_episodes(
            driver, "scripted_gold", "agent@1", 1, AnchorEvalConfig(max_steps=5)
        )


def test_several_episodes_can_be_collected_in_one_cycle():
    driver, _insts, _pol = build_anchor_driver(n=2, max_game_ms=400)
    out = play_anchor_episodes(driver, "scripted_bronze", "agent@1", n_episodes=3)
    assert len(out) == 3
    assert {e.instance for e in out} <= {0, 1}


# -- the evaluator ---------------------------------------------------------


class SpyActor:
    """Records the weights it was handed, so a stale-eval bug is visible."""

    def __init__(self):
        self.policy = self
        self.loaded = []

    def load_state_dict(self, sd):
        self.loaded.append(sd)

    # BatchPolicy, minimally
    def initial_state(self, batch):
        return {"batch": batch}

    def act_batch(self, observations, state, resets=None, deterministic=False):
        self.last_deterministic = deterministic
        return [{"instance": o["instance"], "side": o["side"], "policy": "self"}
                for o in observations], state


def make_evaluator(anchor_ids=("scripted_bronze", "scripted_gold"), **cfg_kw):
    anchors = anchors_for_run(list(anchor_ids))
    made = {}

    def factory(anchor):
        actor = SpyActor()
        driver, _insts, _pol = build_anchor_driver(n=1, max_game_ms=400, policy=actor)
        made[anchor.id] = (driver, actor)
        return driver, actor

    ev = AnchorEvaluator(
        anchors, factory, agent_id_fn=lambda: "agent@42",
        config=AnchorEvalConfig(**cfg_kw),
    )
    return ev, made


def test_the_evaluator_loads_the_current_weights_before_every_cycle():
    """Otherwise the ladder measures the eval actor's construction-time weights."""
    ev, made = make_evaluator(("scripted_bronze",), rotate=True)
    ev(100, {"policy": {"w": 1}})
    ev(200, {"policy": {"w": 2}})
    _driver, actor = made["scripted_bronze"]
    assert actor.loaded == [{"w": 1}, {"w": 2}]


def test_an_empty_payload_is_refused_rather_than_silently_evaluating_random_weights():
    ev, _ = make_evaluator(("scripted_bronze",))
    with pytest.raises(AnchorEvalError, match="empty payload"):
        ev(100, {})


def test_rotation_plays_one_anchor_per_cycle_and_covers_them_all():
    ev, _made = make_evaluator(("scripted_bronze", "scripted_gold", "scripted_diamond"),
                               rotate=True)
    seen = []
    for u in range(6):
        seen.extend(e.opponent_id for e in ev(u, {"policy": {"w": u}}))
    assert seen == [
        "scripted_bronze", "scripted_gold", "scripted_diamond",
        "scripted_bronze", "scripted_gold", "scripted_diamond",
    ]


def test_without_rotation_every_anchor_is_played_each_cycle():
    ev, _made = make_evaluator(("scripted_bronze", "scripted_gold"), rotate=False)
    out = ev(100, {"policy": {"w": 1}})
    assert sorted(e.opponent_id for e in out) == ["scripted_bronze", "scripted_gold"]


def test_the_eval_acts_deterministically_by_default():
    """Sampling measures the exploration distribution, not the policy's quality."""
    ev, made = make_evaluator(("scripted_bronze",))
    ev(100, {"policy": {"w": 1}})
    _driver, actor = made["scripted_bronze"]
    assert actor.last_deterministic is True


def test_servers_are_started_once_per_anchor_not_once_per_cycle():
    """A process restart is ~12s against 0.23ms for an in-process episode reset."""
    ev, _made = make_evaluator(("scripted_bronze",), rotate=True)
    calls = []
    inner = ev.driver_factory
    ev.driver_factory = lambda a: (calls.append(a.id), inner(a))[1]
    for u in range(4):
        ev(u, {"policy": {"w": u}})
    assert calls == ["scripted_bronze"]


def test_the_evaluator_refuses_an_unplayable_anchor_at_construction(tmp_path):
    with pytest.raises(AnchorConfigError):
        AnchorEvaluator(
            [AnchorSpec("bc_policy", "policy", None)],
            driver_factory=lambda a: (None, None),
            agent_id_fn=lambda: "agent@0",
        )


# -- end to end into the eval report ---------------------------------------


def test_anchor_results_reach_the_eval_report_and_the_metrics_file(run_dir):
    """The whole point: win_rate_vs_anchor stops being (None, 0)."""
    anchors = anchors_for_run(["scripted_bronze"])
    cfg = RunConfig(run_dir=run_dir, num_actors=0, checkpoint_every=0, snapshot_every=0,
                    eval_every=2)

    def fake_anchor_eval(update, payload):
        from lanerl_train.run import EpisodeResult

        return [
            EpisodeResult(
                agent=f"agent@{update}", opponent_id="scripted_bronze",
                opponent_category="anchor", score=1.0, cs_at_10=24.0, reason="time",
            )
        ]

    loop = TrainingLoop(
        cfg,
        FakeLearner(),
        evaluator=Evaluator(anchors=anchors, min_games_for_min_winrate=1, elo_anchor=None),
        anchor_eval=fake_anchor_eval,
        metrics=MetricsLog(run_dir / "metrics.jsonl"),
    )
    loop.require_anchor_eval()
    for i in range(4):
        loop.submit(Rollout(actor_id=0, param_version=i, steps=8))
        loop.step_once(timeout=1.0)

    recs = [json.loads(ln) for ln in (run_dir / "metrics.jsonl").read_text().splitlines()]
    evals = [r for r in recs if r["kind"] == "eval"]
    assert len(evals) == 2
    wr, games = evals[-1]["win_rate_vs_anchor"]["scripted_bronze"]
    assert games == 2 and wr == 1.0

    anchor_rows = [r for r in recs if r["kind"] == "anchor_eval"]
    assert len(anchor_rows) == 2
    assert anchor_rows[0]["by_anchor"] == {"scripted_bronze": 1}
    assert anchor_rows[0]["elapsed_s"] >= 0.0

    eps = [r for r in recs if r["kind"] == "episode" and r["opponent_category"] == "anchor"]
    assert len(eps) == 2 and all(e["cs_at_10"] == 24.0 for e in eps)


def test_the_agent_id_is_stable_between_snapshots_so_games_accumulate(run_dir):
    """It used to change every update, so no pair ever reached n>1."""
    cfg = RunConfig(run_dir=run_dir, num_actors=0, checkpoint_every=0, snapshot_every=100,
                    eval_every=0)
    loop = TrainingLoop(cfg, FakeLearner(), metrics=MetricsLog(run_dir / "m.jsonl"))
    ids = []
    for _i in range(250):
        loop.state.update += 1
        ids.append(loop.agent_id())
    assert sorted(set(ids)) == ["agent@0", "agent@100", "agent@200"]
    assert ids.count("agent@100") == 100


def test_cs_at_10_recorded_by_the_collector_reaches_the_eval_report(run_dir):
    """179 CS@10 readings in the first run; all 37 eval rows said null."""
    cfg = RunConfig(run_dir=run_dir, num_actors=0, checkpoint_every=0, snapshot_every=0,
                    eval_every=1)
    loop = TrainingLoop(
        cfg,
        FakeLearner(),
        evaluator=Evaluator(anchors=anchors_for_run(["scripted_bronze"])),
        anchor_eval=lambda u, p: [],
        metrics=MetricsLog(run_dir / "metrics.jsonl"),
    )
    # exactly how lane_wiring.collect_rollout labels a self-play episode
    for cs in (30.0, 32.0):
        loop.record_episode(
            EpisodeResult(agent="self", opponent_id="self", opponent_category="self",
                          score=0.5, cs_at_10=cs, reason="time")
        )
    loop.submit(Rollout(actor_id=0, param_version=0, steps=8))
    loop.step_once(timeout=1.0)
    row = [json.loads(ln) for ln in (run_dir / "metrics.jsonl").read_text().splitlines()
           if json.loads(ln)["kind"] == "eval"][-1]
    assert row["cs_at_10"] is not None, "the report asked about an id nothing was filed under"
    assert row["cs_at_10"][0] == pytest.approx(31.0)
    assert row["cs_at_10"][2] == 2


def test_the_report_still_says_no_games_when_the_anchor_has_not_been_played_yet(run_dir):
    """The (None, 0) state must survive -- it is honest before the first game."""
    ev = Evaluator(anchors=anchors_for_run(["scripted_gold"]))
    report = ev.report(0, "agent@0", [])
    assert report.win_rate_vs_anchor["scripted_gold"] == (None, 0)
    assert any("no games against anchor" in n for n in report.notes)
