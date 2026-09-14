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

Three later sections cover the ways the evaluator produced a number that was
not a measurement, rather than no number at all:

* the EPISODE BOUNDARY.  Anchor games ended at the first death while training
  played the full ten minutes, so anchor CS@10 was measured on deathless games
  only -- survivorship bias -- and scoring a played-out game needs the death
  differential, not the end reason.
* the frozen POLICY anchor, whose branch of ``make_anchor_driver_factory`` had
  never been executed by anything and contained an undefined name.
* STATISTICAL POWER.  One ten-minute game per anchor cannot resolve a 29 CS
  difference; the table lives on ``AnchorEvalConfig.episodes_per_anchor`` and
  the tests here pin the default to it.

``eval_vs_bot`` is tested here too rather than in its own file: it is the same
measurement against the same frozen bot, run standalone.
"""

from __future__ import annotations

import json
import math
import socket
import subprocess
import sys
import types
from pathlib import Path

import pytest

from lanerl_rl import constants as C
from lanerl_train import anchor_eval as anchor_eval_mod
from lanerl_train.anchor_eval import (
    GAME_DECISIONS,
    AnchorEvalConfig,
    AnchorEvaluator,
    AnchorEvalError,
    anchor_launch_spec,
    make_anchor_driver_factory,
    play_anchor_episodes,
    score_for_deaths,
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


def test_a_policy_anchor_turns_the_in_server_bot_OFF(tmp_path):
    """A policy anchor is driven by a frozen network on red, not by the bot.

    This used to raise: only scripted anchors could be played, which made
    ``--bc-checkpoint`` a flag whose only reachable effect was to kill the run
    at the first evaluation. It is now supported, and the load-bearing detail
    is ``bot_teams="none"`` -- leave the in-server bot on and it would fight
    the network for the same champion, which is the LANERL_BOT default-to-blue
    bug wearing a different hat.

    The rung matters: scripted anchors measure the agent against a heuristic,
    and only a policy anchor can answer "is the agent better than the BC prior
    it was initialised from".
    """
    ckpt = tmp_path / "bc.pt"
    ckpt.write_bytes(b"w")
    spec = anchor_launch_spec(AnchorSpec("bc_policy", "policy", ckpt))
    assert spec.bot_teams == "none"
    assert spec.bot_config is None


def test_a_policy_anchor_still_needs_its_checkpoint(tmp_path):
    with pytest.raises(AnchorEvalError, match="no usable checkpoint"):
        anchor_launch_spec(AnchorSpec("bc_policy", "policy", tmp_path / "missing.pt"))


def test_an_unknown_anchor_kind_is_refused(tmp_path):
    with pytest.raises(AnchorEvalError, match="expected 'scripted' or 'policy'"):
        anchor_launch_spec(AnchorSpec("weird", "handwritten", tmp_path))


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


def build_anchor_driver(n=1, max_game_ms=400, instances=None, policy=None,
                        end_on_death=False):
    """A VecDriver set up exactly as the anchor runner sets one up.

    ``end_on_death`` defaults to False to MATCH ``make_anchor_driver_factory``.
    It used to inherit ``EpisodeSpec``'s True, which is what the real anchor
    driver did too, and that disagreed with the training drivers launched under
    ``--no-end-on-death``.
    """
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
        episode=EpisodeSpec(max_game_ms=max_game_ms, end_on_death=end_on_death),
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


def test_each_episode_reports_its_own_length_not_the_loop_counter():
    """length_steps was the while-loop index, so the Nth game reported all N."""
    # FakeInstance advances 66 ms a decision; max_game_ms=400 makes that 7.
    driver, _insts, _pol = build_anchor_driver(n=1, max_game_ms=400)
    out = play_anchor_episodes(driver, "scripted_gold", "agent@7", n_episodes=4)
    assert len(out) == 4
    lengths = [ep.length_steps for ep in out]
    assert len(set(lengths)) == 1, (
        f"episode lengths {lengths} grow with the loop counter instead of "
        f"describing each episode"
    )
    assert lengths[0] == 7, lengths


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
    # These tests are about PLUMBING -- which anchor is played, whose weights
    # it is played with -- not about how many games it takes to resolve a CS
    # difference. The real default (35) is a statistical-power decision and is
    # tested as one; asking for 35 fake games here would only make the
    # rotation assertions unreadable.
    cfg_kw.setdefault("episodes_per_anchor", 1)
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


def test_the_eval_SAMPLES_by_default():
    """Inverted on 2026-09-12. It used to assert deterministic is True, on the
    reasoning that "sampling measures the exploration distribution, not the
    policy's quality". That reasoning is wrong for this action space.

    The server pathfinds, so one move order carries the champion thousands of
    units and the bot spends most frames idle: a real 90 s demo game is
    {'noop': 836, 'move': 11}. BC clones that, so the argmax button is noop in
    essentially every early-game state -- 7,199 of 7,199 argmax actions were
    noop on the BC checkpoint. The mode of this policy is not a summary of it,
    it is a strictly worse policy that never moves.

    Measured, same weights, same server, 240 s:
        deterministic=False  ->  10,975 units travelled, 6 CS, level 3
        deterministic=True   ->       0 units travelled, 0 CS, level 1

    Every anchor game before this reported the agent standing in the fountain
    at level 1 with 0 CS, scored 0.50, and the ladder treated it as real.
    """
    ev, made = make_evaluator(("scripted_bronze",))
    ev(100, {"policy": {"w": 1}})
    _driver, actor = made["scripted_bronze"]
    assert actor.last_deterministic is False


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
    """298 CS@10 readings in the first run; all 40 eval rows said null."""
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


# -- the episode boundary: full ten minutes, not the first death ----------


def test_the_anchor_driver_plays_the_full_ten_minutes_by_default(tmp_path, monkeypatch):
    """``make_anchor_driver_factory`` inherited ``EpisodeSpec.end_on_death=True``.

    The training drivers are launched with ``--no-end-on-death`` and the anchor
    driver was not, so the evaluator and the thing it evaluated did not agree
    on what an episode is. Because ``play_anchor_episodes`` records CS@10 only
    for ``reason == "time"``, anchor CS@10 was measured on DEATHLESS GAMES
    ONLY: pure survivorship bias, in which a policy that learns to trade looks
    like a policy that stopped farming. ``runs/rl-bc4-0912`` shows the shape --
    its one anchor game that ended ``death_team_100`` (12,721 steps) reported
    ``cs_at_10 = None`` while every self-play episode of the same run ran the
    full 18,001 steps.
    """
    made: list = []
    _fake_server_env(monkeypatch, made)
    factory = make_anchor_driver_factory(
        build_policy_actor=_AnchorActor,
        policy_key="self",
        port_base=49000,
        log_dir=tmp_path,
        adapter_factory_for=_Adapters,
        envs=1,
    )
    driver, _actor = factory(anchors_for_run(["scripted_bronze"])[0])
    assert driver.episode.end_on_death is False
    assert driver.episode.max_game_ms == 600_000, "CS@10 is defined at ten minutes"


def test_the_old_behaviour_is_still_reachable_deliberately(tmp_path, monkeypatch):
    made: list = []
    _fake_server_env(monkeypatch, made)
    factory = make_anchor_driver_factory(
        build_policy_actor=_AnchorActor, policy_key="self", port_base=49100,
        log_dir=tmp_path, adapter_factory_for=_Adapters, end_on_death=True,
    )
    driver, _actor = factory(anchors_for_run(["scripted_gold"])[0])
    assert driver.episode.end_on_death is True


def test_cs_at_10_is_measured_in_a_game_the_agent_died_in(monkeypatch):
    """The survivorship bias, end to end: a death must not erase the CS."""
    insts = [FakeInstance(0, step_ms=66, champ_dies_at_ms=200)]
    driver, _i, _p = build_anchor_driver(n=1, max_game_ms=400, instances=insts)
    monkeypatch.setattr(driver, "cs_at_10", lambda i: {C.TEAM_BLUE: 31, C.TEAM_RED: 48})
    out = play_anchor_episodes(driver, "scripted_bronze", "agent@1", 1)
    assert out[0].reason == "time", "the game must run the clock out, death or not"
    assert out[0].cs_at_10 == 31.0
    assert out[0].opponent_cs_at_10 == 48.0


def test_a_game_played_through_a_death_is_a_loss_not_a_draw(caplog):
    """Playing through deaths must not turn every anchor score into 0.5.

    ``score_for_reason`` can only read WHY the episode stopped, and a
    full-length game always stops for "time" -- so the fix to the boundary
    would have re-created the 0.5-by-construction score this whole module
    exists to escape. The death differential over the ten minutes replaces it.
    """
    insts = [FakeInstance(0, step_ms=66, champ_dies_at_ms=200)]
    driver, _i, _p = build_anchor_driver(n=1, max_game_ms=400, instances=insts)
    with caplog.at_level("INFO", logger="lanerl_train.anchor_eval"):
        out = play_anchor_episodes(driver, "scripted_bronze", "agent@1", 1)
    assert out[0].reason == "time"
    assert out[0].score == 0.0
    assert (out[0].deaths, out[0].kills) == (1, 0)
    assert "deaths 1-0" in caplog.text, "the differential must be visible in the log"


def test_a_corpse_is_one_death_not_one_per_frame():
    """hp stays <= 0 for every frame until the champion respawns.

    A level-triggered count would score a single death as one per decision --
    ~60 for a 20 s respawn -- and turn an even trade into a rout in the ladder.
    """
    # dies at 66 ms, so 6 of this game's 7 decisions are spent dead
    insts = [FakeInstance(0, step_ms=66, champ_dies_at_ms=66)]
    driver, _i, _p = build_anchor_driver(n=1, max_game_ms=400, instances=insts)
    out = play_anchor_episodes(driver, "scripted_bronze", "agent@1", 1)
    assert out[0].deaths == 1, "one death, not one per frame spent dead"


def test_the_rune_page_canary_is_carried_on_every_anchor_episode():
    """An in-process reset that strips the page (mhp 672 -> 616) is otherwise silent.

    Nothing in the observation, the reward or the CS readout looks wrong while
    it happens, so the only way to notice is to record the stats the episode was
    actually played at. FakeInstance's champion has mhp 600 and sends no ad.
    """
    driver, _i, _p = build_anchor_driver(n=1, max_game_ms=400)
    out = play_anchor_episodes(driver, "scripted_bronze", "agent@1", 2)
    assert [e.first_frame_mhp for e in out] == [600.0, 600.0]
    # absent on the wire stays None; a guessed stat is how this project got
    # three different wrong attack-damage constants
    assert all(e.first_frame_ad is None for e in out)


def test_score_for_deaths_agrees_with_the_end_reason_and_keeps_draws_draws():
    assert score_for_deaths(0, 0) == 0.5, "a deathless lane is the common case"
    assert score_for_deaths(0, 2) == 1.0
    assert score_for_deaths(3, 1) == 0.0
    assert score_for_deaths(2, 2) == 0.5


# -- the frozen POLICY anchor: the branch that had never been executed -----


class _FrozenNet:
    """Just enough of ``LanePolicy`` for the factory's policy branch."""

    def __init__(self):
        self.loaded: list = []
        self.eval_calls = 0

    def load_state_dict(self, sd):
        self.loaded.append(dict(sd))

    def eval(self):
        self.eval_calls += 1
        return self


class _AnchorActor:
    """A BatchPolicy with a ``.policy``, as ``LanePolicyActor`` has."""

    def __init__(self):
        self.policy = _FrozenNet()
        self.last_deterministic = None

    def initial_state(self, batch):
        return {"batch": batch}

    def act_batch(self, observations, state, resets=None, deterministic=False):
        self.last_deterministic = deterministic
        return (
            [{"instance": o["instance"], "side": o["side"], "policy": "x"}
             for o in observations],
            state,
        )


class _Adapters:
    """What ``adapter_factory_for()`` is expected to return."""

    def __init__(self):
        self.adapter_factory = lambda i, side: FakeAdapter(i, side)
        self.encoder = FakeEncoder()


def _fake_server_env(monkeypatch, made):
    """Let ``make_anchor_driver_factory`` build real drivers over FakeInstances.

    The factory is the piece these tests exist for and it was never executed by
    one: the old suite covered ``anchor_launch_spec`` and ``anchors_for_run``
    only, which is why an undefined name survived inside it.
    """
    real = VecLaneEnv

    def wrapper(n, spec=None, ports=None, log_dir=None, **kw):
        insts = [FakeInstance(i, step_ms=66) for i in range(n)]
        made.extend(insts)
        return real(n=n, ports=ports, factory=lambda i, p: insts[i], step_timeout_s=2.0)

    monkeypatch.setattr(anchor_eval_mod, "VecLaneEnv", wrapper)


def test_the_policy_anchor_branch_of_the_factory_runs_at_all(tmp_path, monkeypatch):
    """It referenced ``_ANCHOR_KEY``, which the module never defined.

    A NameError, in a branch reached only by ``--anchors ...,bc_policy``, in a
    factory that :class:`AnchorEvaluator` calls LAZILY -- once, on the first
    cycle that rotates to that anchor. So the run started fine, trained fine,
    and died at ~update 800 from inside ``TrainingLoop.step_once``, an hour of
    GPU time in. Nothing caught it because nothing had ever called the factory.
    """
    import torch

    ckpt = tmp_path / "bc.pt"
    torch.save({"policy": {"w": 1}}, ckpt)
    made: list = []
    _fake_server_env(monkeypatch, made)
    built: list = []

    def build():
        a = _AnchorActor()
        built.append(a)
        return a

    factory = make_anchor_driver_factory(
        build_policy_actor=build,
        policy_key="self",
        port_base=49200,
        log_dir=tmp_path,
        adapter_factory_for=_Adapters,
        envs=1,
        max_game_ms=400,
    )
    driver, actor = factory(AnchorSpec("bc_policy", "policy", ckpt))

    assert actor is built[0]
    red_key = driver.assignments[0].red
    assert red_key is not None, "a policy anchor is a NETWORK on red, not the bot"
    assert red_key != "self", "the anchor must not share the live policy's slot"
    frozen = driver.policies[red_key]
    assert frozen is built[1], "the anchor is a SECOND actor, not the live one"
    assert frozen.policy.loaded == [{"w": 1}], "the checkpoint must be loaded"
    assert frozen.policy.eval_calls == 1


def test_the_frozen_policy_anchor_never_receives_the_current_weights(tmp_path, monkeypatch):
    """A rung that moves with the agent is not a rung."""
    import torch

    ckpt = tmp_path / "bc.pt"
    torch.save({"policy": {"w": 1}}, ckpt)
    _fake_server_env(monkeypatch, [])
    factory = make_anchor_driver_factory(
        build_policy_actor=_AnchorActor, policy_key="self", port_base=49300,
        log_dir=tmp_path, adapter_factory_for=_Adapters, envs=1, max_game_ms=400,
    )
    anchor = AnchorSpec("bc_policy", "policy", ckpt)
    driver, actor = factory(anchor)
    frozen = driver.policies[driver.assignments[0].red]

    ev = AnchorEvaluator(
        [anchor], lambda a: (driver, actor), agent_id_fn=lambda: "agent@800",
        config=AnchorEvalConfig(episodes_per_anchor=1),
    )
    out = ev(800, {"policy": {"w": 2}})
    assert [e.opponent_id for e in out] == ["bc_policy"]
    assert actor.policy.loaded[-1] == {"w": 2}, "the LIVE side gets the new weights"
    assert frozen.policy.loaded == [{"w": 1}], "the anchor stays where it was frozen"


def test_the_live_policy_and_the_frozen_anchor_may_not_share_a_key(tmp_path):
    """Same key in the policy map = the anchor is handed the current weights."""
    with pytest.raises(AnchorEvalError, match="policy map"):
        make_anchor_driver_factory(
            build_policy_actor=_AnchorActor,
            policy_key=anchor_eval_mod._ANCHOR_KEY,
            port_base=49400,
            log_dir=tmp_path,
            adapter_factory_for=_Adapters,
        )


# -- statistical power -----------------------------------------------------


#: Measured over the 144 self-play episodes of runs/rl-bc4-0912/metrics.jsonl,
#: every one of which reached ten minutes: mean 36.44, sd 7.31 (95% CI
#: 6.55-8.26), median 36, min 11, max 53.
MEASURED_CS_SD = 7.31
#: z(0.975) + z(0.80): two-sided alpha = 0.05 at 80% power.
_Z_SUM = 1.959964 + 0.841621


def two_sample_mde(n_per_group: float, sd: float = MEASURED_CS_SD) -> float:
    """Smallest CS difference ``n`` games per group can detect, normal approx."""
    return _Z_SUM * sd * math.sqrt(2.0 / n_per_group)


def test_the_default_episode_count_can_actually_resolve_five_cs():
    """One game per anchor could not resolve 29 CS, let alone 5.

    CS@10 sd is 7.31 (n=144, runs/rl-bc4-0912), so a single game per group
    detects nothing smaller than ~29 CS at 80% power -- the evaluator could not
    tell the 36 CS that run really farmed from 7 CS. Every anchor number this
    project has published was one draw from a distribution three times wider
    than the effects being argued about. See AnchorEvalConfig.episodes_per_anchor
    for the full table.
    """
    n = AnchorEvalConfig().episodes_per_anchor
    assert two_sample_mde(n) <= 5.0, (
        f"{n} games per group resolves only {two_sample_mde(n):.1f} CS; "
        f"{math.ceil(2 * (_Z_SUM * MEASURED_CS_SD / 5.0) ** 2)} are needed for 5"
    )
    # and the table in the docstring must not drift from the formula
    assert round(two_sample_mde(1), 1) == 29.0
    assert round(two_sample_mde(8), 1) == 10.2
    assert round(two_sample_mde(35), 1) == 4.9


def test_the_step_budget_covers_the_whole_cycle_not_one_game():
    """``max_steps`` was a ONE-GAME bound while ``episodes_per_anchor`` was free.

    Asking for n>1 games would have collected one and returned it without an
    error (``play_anchor_episodes`` raises only when it collects NOTHING) -- an
    underpowered result wearing the label of a powered one.
    """
    cfg = AnchorEvalConfig(episodes_per_anchor=35)
    assert cfg.steps_budget(35, n_envs=1) >= 35 * GAME_DECISIONS


def test_the_step_budget_scales_with_server_WAVES_not_with_games():
    """One driver.step() advances every instance, so 8 servers play 8 games at once."""
    cfg = AnchorEvalConfig()
    assert cfg.steps_budget(35, n_envs=8) >= 5 * GAME_DECISIONS  # ceil(35/8) waves
    assert cfg.steps_budget(35, n_envs=8) < 6 * GAME_DECISIONS * 1.2 + 1
    assert cfg.steps_budget(1, n_envs=1) < 2 * GAME_DECISIONS


def test_an_explicit_max_steps_still_bounds_the_cycle():
    """The hang guard must stay overridable, or a stuck server holds the learner."""
    assert AnchorEvalConfig(max_steps=5).steps_budget(35, n_envs=8) == 5


def test_a_cycle_that_comes_up_short_of_the_games_asked_for_says_so(caplog):
    """Silently returning fewer games than asked for is the underpowered-result bug."""
    driver, _insts, _pol = build_anchor_driver(n=1, max_game_ms=400)
    with caplog.at_level("ERROR", logger="lanerl_train.anchor_eval"):
        out = play_anchor_episodes(
            driver, "scripted_gold", "agent@1", 4, AnchorEvalConfig(max_steps=10)
        )
    assert len(out) == 1
    assert "asked for 4 game(s), collected 1" in caplog.text


def test_anchor_eval_samples_rather_than_taking_the_mode():
    """The anchor ladder must not evaluate the argmax policy.

    With server-side pathfinding the bot issues ONE move order and then idles
    while it walks, so a real 90-second demo game is {'noop': 836, 'move': 11}
    and BC clones ~99% noop in the early game. The argmax of that is noop in
    every state: measured on the BC checkpoint, deterministic=True travelled 0
    units and scored 0 CS while the same weights sampled travelled 10,975 units
    and scored 6 CS in 240 s, and 37.3 CS over a full game.

    Anchor games consequently reported the agent at level 1, full HP, 0 CS --
    standing in the fountain for ten minutes -- which the ladder read as a
    legitimate score of 0.50 against every anchor.
    """
    from lanerl_train.anchor_eval import AnchorEvalConfig

    assert AnchorEvalConfig().deterministic is False


# -- eval_vs_bot: one server process per episode ---------------------------
#
# The script's headline "BC = 37.3 CS" was the mean of one runed episode and two
# base-stats ones: the server's IN-PROCESS episode reset strips the rune and
# mastery page (mhp 672 -> 616, ad 78.14 -> 57.88), so episodes 2..N of a shared
# process are played by a weaker champion. A process restart costs ~12 s against
# a ~650 s episode, so every episode gets its own -- and the homogeneity check
# is what makes that a guarantee rather than a comment.


def _champ_frame(t_ms: int, mhp: float = 672.0, ad: float = 78.14, cs: int = 0) -> dict:
    """An observation shaped like ``LanerlControl.BuildObservation``."""
    # Both champions WALK. A fixture whose champions stand on one spot for a
    # whole game is indistinguishable from the harness fault that
    # anchor_eval.UNDRIVEN_MOVE_EPS exists to catch -- an undriven champion --
    # so a stationary fixture would either defeat that check or be rejected by
    # it. 5 units/s puts them ~3,000 units along over a ten-minute game.
    walk = int(t_ms) / 1000.0 * 5.0
    return {
        "t": int(t_ms),
        "u": [
            {"id": 1, "k": "Champion", "tm": 100, "x": 1000 + walk, "y": 12000,
             "hp": 600, "mhp": mhp, "ad": ad, "lvl": 1, "gold": 500, "cs": int(cs)},
            {"id": 2, "k": "Champion", "tm": 200, "x": 3000, "y": 13000 - walk,
             "hp": 600, "mhp": mhp, "ad": ad, "lvl": 1, "gold": 500, "cs": int(cs)},
        ],
    }


class _FakeWire:
    """The control channel's lockstep, in memory: one line out, one line in."""

    def __init__(self, frames):
        self._frames = list(frames)
        self.written: list = []

    def readline(self):
        if not self._frames:
            return b""
        return (json.dumps(self._frames.pop(0)) + "\n").encode()

    def write(self, b):
        self.written.append(b)

    def flush(self):
        pass


class _FakeSock:
    def __init__(self, wire):
        self._wire = wire

    def makefile(self, mode):
        return self._wire


class _FakeProc:
    def __init__(self):
        self.terminated = 0

    def terminate(self):
        self.terminated += 1

    def wait(self, timeout=None):
        return 0


def _arm_eval_vs_bot(monkeypatch, tmp_path, episodes, stats_per_episode):
    """Point ``eval_vs_bot`` at an in-memory server and return (module, procs)."""
    import lanerl_train.eval_vs_bot as evb

    procs: list = []

    def fake_popen(cmd, cwd=None, env=None, stdout=None, stderr=None):
        if stdout is not None:
            stdout.close()
        p = _FakeProc()
        procs.append(p)
        return p

    def fake_connection(addr, timeout=None):
        # Which episode we are in is exactly how many processes have started.
        mhp, ad = stats_per_episode[min(len(procs) - 1, len(stats_per_episode) - 1)]
        frames = [_champ_frame(t, mhp=mhp, ad=ad, cs=t // 66)
                  for t in (0, 66, 132, 198, 264, 330)]
        return _FakeSock(_FakeWire(frames))

    monkeypatch.setattr(evb, "subprocess", types.SimpleNamespace(
        Popen=fake_popen, STDOUT=subprocess.STDOUT,
        TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(evb, "socket", types.SimpleNamespace(
        socket=socket.socket, create_connection=fake_connection))
    monkeypatch.setattr(evb, "_REPO", tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "eval_vs_bot", "--checkpoint", "random",
        "--episodes", str(episodes), "--max-game-ms", "200",
    ])
    return evb, procs


def test_eval_vs_bot_launches_a_fresh_server_process_for_every_episode(tmp_path, monkeypatch):
    """Hoisting the launch out of the loop is what made episodes 2..N base-stats."""
    evb, procs = _arm_eval_vs_bot(monkeypatch, tmp_path, 3, [(672.0, 78.14)])
    assert evb.main() == 0
    assert len(procs) == 3, "one server process per episode, not one per invocation"
    assert all(p.terminated == 1 for p in procs), "and every one is torn down"


def test_eval_vs_bot_refuses_to_average_a_runed_game_with_a_base_stats_one(tmp_path, monkeypatch):
    """The exact shape of "BC = 37.3": one real game plus two weaker ones."""
    evb, _procs = _arm_eval_vs_bot(
        monkeypatch, tmp_path, 3, [(672.0, 78.14), (616.0, 57.88)]
    )
    with pytest.raises(evb.HeterogeneousEpisodes, match="rune/mastery page"):
        evb.main()


def test_start_stats_reads_the_page_off_the_first_frame_and_guesses_nothing():
    from lanerl_train.eval_vs_bot import start_stats

    got = start_stats(_champ_frame(0, mhp=672.0, ad=78.14))
    assert got["mhp"] == 672.0 and got["ad"] == 78.14
    # a stat the server did not send is ABSENT, never defaulted: three wrong
    # attack-damage constants in a row came out of guessing one
    assert "ap" not in got
    assert start_stats(None) == {}


def test_episodes_that_started_from_the_same_stats_are_averaged():
    from lanerl_train.eval_vs_bot import check_homogeneous

    check_homogeneous([{"start_stats": {"mhp": 672.0, "ad": 78.14}}] * 4)  # must not raise
