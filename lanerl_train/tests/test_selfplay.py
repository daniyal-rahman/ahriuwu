"""The seam between opponent sampling, live instances and the learner.

The two things worth testing here are the ones that only exist because of how
the real server behaves: that a scripted anchor's difficulty is process-level
(so it needs a dedicated instance), and that swapping an opponent's weights
keeps the batch slots -- and therefore the recurrent state -- aligned.
"""

from __future__ import annotations

import logging
import random

from lanerl_train.eval import AnchorSpec, default_anchors
from lanerl_train.league import (
    LATEST,
    CheckpointPool,
    LeagueConfig,
    OpponentSampler,
    OpponentSpec,
    Snapshot,
    WinRateTracker,
)
from lanerl_train.ports import InstancePorts
from lanerl_train.selfplay import (
    BLUE_TEAM,
    RED_TEAM,
    SelfPlayCollector,
    SwappableOpponent,
    build_assignments,
    group_assignment,
    lane_outcome,
    plan_instances,
)
from lanerl_train.vec import EpisodeSpec, ServerLaunchSpec, VecDriver, VecLaneEnv

from .fakes import FakeAdapter, FakeEncoder, FakeInstance, FakePolicy, make_obs


# -- outcome ---------------------------------------------------------------


def test_a_dead_champion_loses_regardless_of_gold():
    obs = make_obs(600_000, blue_hp=0)
    obs["u"][0]["gold"] = 9999
    assert lane_outcome(obs) == 0.0


def test_gold_decides_when_both_survive():
    obs = make_obs(600_000)
    obs["u"][0]["gold"] = 3000
    obs["u"][1]["gold"] = 2000
    assert lane_outcome(obs) == 1.0
    obs["u"][0]["gold"] = 1000
    assert lane_outcome(obs) == 0.0


def test_a_negligible_gold_lead_is_a_draw_not_a_win():
    obs = make_obs(600_000)
    obs["u"][0]["gold"] = 2010
    obs["u"][1]["gold"] = 2000
    assert lane_outcome(obs) == 0.5


def test_cs_breaks_a_gold_tie_when_it_is_known():
    obs = make_obs(600_000)
    obs["u"][0]["gold"] = obs["u"][1]["gold"] = 2000
    assert lane_outcome(obs, cs={BLUE_TEAM: 33, RED_TEAM: 21}) == 1.0
    assert lane_outcome(obs, cs={BLUE_TEAM: 21, RED_TEAM: 33}) == 0.0
    assert lane_outcome(obs, cs={BLUE_TEAM: 30}) == 0.5  # only one side known


def test_a_missing_champion_is_a_draw_and_is_shouted_about(caplog):
    obs = {"t": 600_000, "u": [u for u in make_obs(0)["u"] if u["tm"] != RED_TEAM]}
    with caplog.at_level(logging.ERROR, logger="lanerl_train.selfplay"):
        assert lane_outcome(obs) == 0.5
    assert "not usable evidence" in caplog.text


# -- instance planning -----------------------------------------------------


def test_scripted_anchors_get_dedicated_instances_because_difficulty_is_process_level():
    plans = plan_instances(16, ServerLaunchSpec(), default_anchors(), share=0.25)
    anchor_plans = [p for p in plans if p.role == "anchor"]
    assert len(anchor_plans) == 4
    # BC is a network, not a scripted bot: it needs no instance.
    assert {p.anchor.id for p in anchor_plans} <= {
        "scripted_bronze", "scripted_gold", "scripted_diamond"
    }
    for p in anchor_plans:
        env = p.spec.environment(InstancePorts(p.index, 30000, 30001))
        assert env["LANERL_BOT"] == "purple"
        assert env["LANERL_BOT_CONFIG"] == str(p.anchor.resource)
    for p in plans:
        if p.role == "policy":
            assert p.spec.environment(InstancePorts(p.index, 30002, 30003))["LANERL_BOT"] == "none"


def test_too_few_anchor_instances_names_the_unmeasured_rungs(caplog):
    with caplog.at_level(logging.ERROR, logger="lanerl_train.selfplay"):
        plans = plan_instances(16, ServerLaunchSpec(), default_anchors(), share=0.05)
    assert sum(p.role == "anchor" for p in plans) == 1
    assert "leaves" in caplog.text and "scripted_gold" in caplog.text
    assert "12s process restart" in caplog.text


def test_a_missing_anchor_config_is_reported_not_silently_dropped(caplog):
    bad = [AnchorSpec("scripted_ghost", "scripted", None)]
    with caplog.at_level(logging.ERROR, logger="lanerl_train.selfplay"):
        plans = plan_instances(8, ServerLaunchSpec(), bad, share=0.25)
    assert all(p.role == "policy" for p in plans)
    assert "have no config file" in caplog.text


def test_no_anchors_means_every_instance_is_self_play():
    plans = plan_instances(8, ServerLaunchSpec(), [], share=0.05)
    assert all(p.role == "policy" for p in plans)


# -- grouping and assignment ----------------------------------------------


def test_groups_round_robin_and_cap_at_the_instance_count(caplog):
    assert group_assignment(8, 4) == [0, 1, 2, 3, 0, 1, 2, 3]
    with caplog.at_level(logging.WARNING, logger="lanerl_train.selfplay"):
        assert group_assignment(3, 10) == [0, 1, 2]
    assert "batch-1 opponent forwards" in caplog.text


def test_assignments_put_the_learner_on_blue_and_the_bot_on_the_anchor_instances():
    plans = plan_instances(8, ServerLaunchSpec(), default_anchors(), share=0.25)
    groups = group_assignment(8, 2)
    assigns = build_assignments(plans, groups)
    for p, a in zip(plans, assigns):
        assert a.blue == "agent"
        if p.role == "anchor":
            assert a.red is None, "the in-server bot keeps its own orders"
        else:
            assert a.red == f"opp@{groups[p.index]}"


# -- swappable opponent ----------------------------------------------------


def test_swapping_weights_keeps_the_slot_and_records_the_swap():
    loaded = []
    inner = FakePolicy("opp")
    opp = SwappableOpponent(inner, lambda p, payload: loaded.append(dict(payload)))
    snap = Snapshot(id="snap@7", step=700)

    opp.set_opponent(OpponentSpec("pfsp", "snap@7", snapshot=snap), {"w": 1},
                     lambda s: {"w": 7})
    assert loaded == [{"w": 7}] and opp.current == "snap@7" and opp.swaps == 1

    # the same snapshot again is a no-op: reloading weights costs for nothing
    opp.set_opponent(OpponentSpec("uniform", "snap@7", snapshot=snap), {"w": 1},
                     lambda s: {"w": 7})
    assert opp.swaps == 1

    # "latest" always reloads: the live weights have moved
    opp.set_opponent(OpponentSpec("latest", LATEST), {"w": 42}, lambda s: {})
    assert loaded[-1] == {"w": 42} and opp.swaps == 2
    opp.set_opponent(OpponentSpec("latest", LATEST), {"w": 43}, lambda s: {})
    assert loaded[-1] == {"w": 43} and opp.swaps == 3


# -- the collector ---------------------------------------------------------


def build_collector(n=4, groups=2, rollout_steps=3, step_ms=200_000, anchors=()):
    insts = [FakeInstance(i, step_ms=step_ms) for i in range(n)]
    ports = [InstancePorts(i, 46000 + 2 * i, 46001 + 2 * i) for i in range(n)]
    env = VecLaneEnv(n, ports=ports, factory=lambda i, p: insts[i], step_timeout_s=2.0)

    plans = plan_instances(n, ServerLaunchSpec(), anchors, share=0.25)
    group_of = group_assignment(n, groups)
    agent = FakePolicy("agent")
    opp_slots = {g: SwappableOpponent(FakePolicy(f"opp{g}"), lambda p, d: None)
                 for g in set(group_of)}
    policies = {"agent": agent}
    for p in plans:
        if not p.opponent_is_scripted:
            policies[f"opp@{group_of[p.index]}"] = opp_slots[group_of[p.index]]

    driver = VecDriver(
        env,
        policies=policies,
        adapter_factory=lambda i, side: FakeAdapter(i, side),
        encoder=FakeEncoder(),
        assignments=build_assignments(plans, group_of),
        episode=EpisodeSpec(max_game_ms=600_000),
    )
    cfg = LeagueConfig()
    pool = CheckpointPool(cfg)
    for i in range(6):
        pool.add(Snapshot(id=f"snap@{i}", step=i * 100))
    sampler = OpponentSampler(pool, WinRateTracker(cfg), [], cfg, random.Random(0))
    collector = SelfPlayCollector(
        driver=driver,
        sampler=sampler,
        opponents=opp_slots,
        group_of=group_of,
        plans=plans,
        rollout_steps=rollout_steps,
        load_snapshot=lambda spec: {"snapshot": spec.id},
    )
    return collector, driver, env, insts, agent


def test_a_rollout_reports_finished_episodes_with_opponent_and_outcome():
    collector, driver, env, insts, agent = build_collector(n=4, rollout_steps=4)
    rollout = collector(actor_id=0, live_payload={"w": 1}, version=3)
    assert rollout.param_version == 3
    assert rollout.steps == 4 * 4
    assert rollout.episodes, "a 200s-per-step fake must cross ten minutes"
    for ep in rollout.episodes:
        assert ep.opponent_category in {"latest", "pfsp", "uniform", "anchor"}
        assert ep.score in (0.0, 0.5, 1.0)
        assert 0 <= ep.instance < 4
    assert abs(sum(rollout.mixture.values()) - 1.0) < 1e-9


def test_each_episode_boundary_draws_a_new_opponent():
    collector, *_ = build_collector(n=4, rollout_steps=6)
    collector(0, {"w": 1}, 0)
    drawn = sum(collector.sampler.counts.values())
    assert drawn >= 4, "at least the initial assignment for every instance"


def test_the_batch_slots_never_move_across_episode_boundaries():
    """A reindexed GRU column is a silent corruption; the slot count must be fixed."""
    collector, driver, env, insts, agent = build_collector(n=4, groups=2, rollout_steps=6)
    collector(0, {"w": 1}, 0)
    assert len(set(agent.batch_sizes)) == 1, f"agent batch width moved: {agent.batch_sizes}"
    assert agent.batch_sizes[0] == 4


def test_cs_is_absent_rather_than_zero_when_the_log_has_none():
    collector, *_ = build_collector(n=2, rollout_steps=4)
    rollout = collector(0, {"w": 1}, 0)
    assert rollout.episodes
    assert all(ep.cs_at_10 is None for ep in rollout.episodes), (
        "FakeInstance writes no server log, so CS@10 is unknown -- and unknown "
        "must not be reported as zero"
    )


def test_an_anchor_instance_always_reports_its_own_frozen_opponent():
    anchors = [a for a in default_anchors() if a.kind == "scripted"]
    collector, driver, env, insts, agent = build_collector(
        n=4, rollout_steps=4, anchors=anchors
    )
    n_anchor = sum(p.role == "anchor" for p in collector.plans)
    assert n_anchor >= 1
    rollout = collector(0, {"w": 1}, 0)
    anchor_idx = {p.index for p in collector.plans if p.role == "anchor"}
    for ep in rollout.episodes:
        if ep.instance in anchor_idx:
            assert ep.opponent_category == "anchor"
            assert ep.opponent_id.startswith("scripted_")


def test_a_restarted_instance_drops_its_episode_instead_of_scoring_a_crash(caplog):
    insts = [FakeInstance(0, step_ms=1000), FakeInstance(1, step_ms=1000, die_after_sends=1)]
    ports = [InstancePorts(i, 46500 + 2 * i, 46501 + 2 * i) for i in range(2)]
    env = VecLaneEnv(2, ports=ports, factory=lambda i, p: insts[i], step_timeout_s=2.0)
    plans = plan_instances(2, ServerLaunchSpec(), [], share=0.0)
    group_of = group_assignment(2, 1)
    opp = SwappableOpponent(FakePolicy("opp"), lambda p, d: None)
    driver = VecDriver(
        env,
        policies={"agent": FakePolicy("agent"), "opp@0": opp},
        adapter_factory=lambda i, side: FakeAdapter(i, side),
        encoder=FakeEncoder(),
        assignments=build_assignments(plans, group_of),
        episode=EpisodeSpec(max_game_ms=10**9),
    )
    cfg = LeagueConfig()
    sampler = OpponentSampler(CheckpointPool(cfg), WinRateTracker(cfg), [], cfg, random.Random(0))
    collector = SelfPlayCollector(driver, sampler, {0: opp}, group_of, plans, 4,
                                  load_snapshot=lambda s: {})
    with caplog.at_level(logging.ERROR, logger="lanerl_train.selfplay"):
        rollout = collector(0, {"w": 1}, 0)
    assert "restarted mid-episode" in caplog.text
    assert all(ep.instance != 1 for ep in rollout.episodes), (
        "a crashed lane must not be scored -- it would feed noise into the ratings"
    )
