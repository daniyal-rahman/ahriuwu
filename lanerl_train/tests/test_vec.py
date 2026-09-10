"""The vec runner: scatter correctness, batching, and surviving instance death.

Every test here uses :class:`FakeInstance`, which speaks the same lockstep as
``LanerlControl`` -- one observation out, exactly one action line in -- so no
game server is needed and the assertions are about *routing*, which is where the
real bugs live.
"""

from __future__ import annotations

import json
import logging

import pytest

from lanerl_train.ports import InstancePorts
from lanerl_train.vec import (
    RESET_ACTION,
    EpisodeSpec,
    SideAssignment,
    VecDriver,
    VecEnvFailure,
    VecLaneEnv,
    _encode_line,
    episode_done,
)

from .fakes import FakeAdapter, FakeEncoder, FakeInstance, FakePolicy, make_obs


def build_env(n=4, ports_base=42000, **fake_kwargs):
    made = {}

    def factory(i, ports):
        kw = fake_kwargs.get(i, {}) if isinstance(next(iter(fake_kwargs), None), int) else {}
        made[i] = FakeInstance(i, **kw)
        return made[i]

    ports = [InstancePorts(i, ports_base + 2 * i, ports_base + 2 * i + 1) for i in range(n)]
    env = VecLaneEnv(n, ports=ports, factory=factory, step_timeout_s=2.0)
    return env, made


def build_env_with(instances):
    """Build a VecLaneEnv over a prepared list of FakeInstance objects."""
    n = len(instances)
    ports = [InstancePorts(i, 43000 + 2 * i, 43000 + 2 * i + 1) for i in range(n)]
    return VecLaneEnv(
        n,
        ports=ports,
        factory=lambda i, p: instances[i],
        step_timeout_s=2.0,
        max_restarts_per_instance=2,
    )


# -- protocol basics -------------------------------------------------------


def test_start_collects_the_unprompted_first_observation():
    insts = [FakeInstance(i) for i in range(4)]
    env = build_env_with(insts)
    res = env.start()
    assert res.n_alive == 4
    assert all(o is not None and o["t"] == 0 for o in res.obs)


def test_step_sends_exactly_one_action_line_per_instance():
    insts = [FakeInstance(i) for i in range(3)]
    env = build_env_with(insts)
    env.start()
    env.step([{"blue": {"t": "noop"}}] * 3)
    assert all(i.sends == 1 for i in insts)
    env.step([None] * 3)
    assert all(i.sends == 2 for i in insts)


def test_a_none_action_becomes_an_empty_object_so_orders_persist():
    """``ApplyActions`` finds no side key and the standing order continues."""
    insts = [FakeInstance(0)]
    env = build_env_with(insts)
    env.start()
    env.step([None])
    assert insts[0].received[-1] == {}


def test_the_reset_action_is_the_line_the_server_actually_parses():
    """``LanerlWire.Parse`` accepts exactly ``{"cmd":"reset"}``.

    The old ``{"reset": 1}`` dated from when ``OnTick`` did a substring test on
    the raw line.  Under the real parser it is an unknown top-level key, so the
    line is Fatal: the episode does not reset, neither champion is ordered, and
    the trainer sees a perfectly normal step.
    """
    assert RESET_ACTION == {"cmd": "reset"}
    assert json.loads(_encode_line(RESET_ACTION)) == RESET_ACTION


def test_encode_line_refuses_a_line_the_server_would_drop_whole():
    """An unknown top-level key costs BOTH champions their orders, silently."""
    with pytest.raises(VecEnvFailure, match="top-level key"):
        _encode_line({"blue": {"t": "move"}, "reset": 1})
    with pytest.raises(VecEnvFailure, match="top-level key"):
        _encode_line({"green": {"t": "move"}})
    with pytest.raises(VecEnvFailure, match="not a command"):
        _encode_line({"cmd": "restart"})
    # An ordinary order carrying "reset" deeper down is now perfectly fine --
    # the server reads JSON, not substrings.
    assert json.loads(_encode_line({"blue": {"t": "move", "x": 1.0, "y": 2.0}}))
    assert _encode_line({}) == "{}"


def test_reset_episodes_resets_only_the_named_instances():
    insts = [FakeInstance(i) for i in range(4)]
    env = build_env_with(insts)
    env.start()
    for _ in range(3):
        env.step([None] * 4)
    env.reset_episodes([1, 3])
    assert [i.resets for i in insts] == [0, 1, 0, 1]
    # instances not being reset still consumed exactly one action line, because
    # the server is blocked reading one
    assert all(i.sends == 4 for i in insts)


# -- death and restart -----------------------------------------------------


def test_instance_death_is_loud_and_the_instance_is_restarted(caplog):
    insts = [FakeInstance(i) for i in range(3)]
    insts[1].die_after_sends = 1
    env = build_env_with(insts)
    env.start()
    with caplog.at_level(logging.ERROR, logger="lanerl_train.vec"):
        env.step([None] * 3)  # send 1: still fine
        res = env.step([None] * 3)  # send 2: instance 1 dies on the read
    assert 1 in res.died
    assert 1 in res.restarted
    assert env.alive[1] is True
    assert insts[1].starts == 2
    text = caplog.text
    assert "INSTANCE DEATH 1/3" in text
    assert "RESTARTING instance 1" in text
    # the survivors kept stepping
    assert insts[0].sends == 2 and insts[2].sends == 2


def test_a_restarted_instance_reports_a_fresh_first_observation():
    insts = [FakeInstance(i, start_t_ms=0) for i in range(2)]
    insts[0].die_after_sends = 0
    env = build_env_with(insts)
    env.start()
    res = env.step([None] * 2)
    assert 0 in res.restarted
    assert res.obs[0] is not None and res.obs[0]["t"] == 0


def test_exhausting_the_restart_budget_stops_the_run_rather_than_shrinking_it():
    insts = [FakeInstance(0, die_after_sends=0)]
    env = build_env_with(insts)  # max_restarts_per_instance=2
    env.start()
    env.step([None])
    env.step([None])
    with pytest.raises(VecEnvFailure, match="exhausted its restart budget"):
        env.step([None])


def test_an_instance_that_never_answers_is_declared_dead_on_timeout(caplog):
    insts = [FakeInstance(0), FakeInstance(1, stall_forever=True)]
    ports = [InstancePorts(i, 44000 + 2 * i, 44001 + 2 * i) for i in range(2)]
    env = VecLaneEnv(
        2,
        ports=ports,
        factory=lambda i, p: insts[i],
        step_timeout_s=0.05,
        auto_restart=False,
    )
    with caplog.at_level(logging.ERROR, logger="lanerl_train.vec"):
        res = env.start()
    assert res.obs[0] is not None
    assert 1 in res.died and "no observation within" in res.died[1]
    assert env.alive[1] is False


def test_a_shared_port_is_rejected_before_anything_is_spawned():
    from lanerl_train.ports import PortAllocationError

    bad = [InstancePorts(0, 5119, 5120), InstancePorts(1, 5119, 5121)]
    with pytest.raises(PortAllocationError, match="assigned twice"):
        VecLaneEnv(2, ports=bad, factory=lambda i, p: FakeInstance(i))


def test_step_before_start_is_an_error():
    env = build_env_with([FakeInstance(0)])
    with pytest.raises(VecEnvFailure, match="step\\(\\) before start"):
        env.step([None])


# -- batching and scatter --------------------------------------------------


def driver_over(assignments, policies, n=None, episode=None, instances=None):
    n = n if n is not None else len(assignments)
    insts = instances or [FakeInstance(i) for i in range(n)]
    env = build_env_with(insts)
    driver = VecDriver(
        env,
        policies=policies,
        adapter_factory=lambda i, side: FakeAdapter(i, side),
        encoder=FakeEncoder(),
        assignments=assignments,
        episode=episode or EpisodeSpec(max_game_ms=10**9),
    )
    return driver, env, insts


def test_actions_are_scattered_to_the_right_instance_and_side():
    n = 6
    pol = FakePolicy("main")
    assignments = [SideAssignment(blue="main", red="main") for _ in range(n)]
    driver, env, insts = driver_over(assignments, {"main": pol})
    driver.start()
    driver.step()
    for i, inst in enumerate(insts):
        action = inst.last_action
        assert action is not None, f"instance {i} got no action"
        # FakeEncoder writes the slot identity into the order.
        assert action["blue"]["x"] == float(i), f"instance {i} received env {action['blue']['x']}"
        assert action["red"]["x"] == float(i)
        assert action["blue"]["y"] == 1.0
        assert action["red"]["y"] == 2.0


def test_one_forward_per_policy_not_one_per_env():
    """1.70 ms/decision unbatched vs 0.058 ms at batch 24 -- this is the whole point."""
    n = 12
    main, opp = FakePolicy("main"), FakePolicy("opp")
    assignments = [SideAssignment(blue="main", red="opp") for _ in range(n)]
    driver, env, insts = driver_over(assignments, {"main": main, "opp": opp})
    driver.start()
    for _ in range(5):
        driver.step()
    assert main.calls == 5 and opp.calls == 5
    assert main.batch_sizes == [n] * 5
    assert opp.batch_sizes == [n] * 5


def test_a_side_with_no_policy_is_left_to_the_in_server_scripted_bot():
    n = 3
    pol = FakePolicy("main")
    assignments = [SideAssignment(blue="main", red=None) for _ in range(n)]
    driver, env, insts = driver_over(assignments, {"main": pol})
    driver.start()
    driver.step()
    for inst in insts:
        action = inst.last_action
        assert "blue" in action
        assert "red" not in action, "omitting the key is how the bot keeps its own orders"
    assert pol.batch_sizes == [n]  # only the blue slots are in the batch


def test_mixed_assignments_group_slots_by_policy():
    latest, snap = FakePolicy("latest"), FakePolicy("snap")
    assignments = [
        SideAssignment(blue="latest", red="latest"),  # mirror self-play
        SideAssignment(blue="latest", red="snap"),  # vs a pool snapshot
        SideAssignment(blue="latest", red=None),  # vs the scripted anchor
    ]
    driver, env, insts = driver_over(assignments, {"latest": latest, "snap": snap})
    driver.start()
    driver.step()
    assert latest.batch_sizes == [4]  # 3 blue + 1 red
    assert snap.batch_sizes == [1]
    assert insts[1].last_action["red"]["policy"] == "snap"
    assert insts[1].last_action["blue"]["policy"] == "latest"


def test_assigning_an_unknown_policy_fails_at_construction():
    with pytest.raises(VecEnvFailure, match="not in the policy map"):
        driver_over([SideAssignment(blue="ghost", red=None)], {"main": FakePolicy()})


def test_a_policy_returning_the_wrong_number_of_actions_is_caught():
    class Short(FakePolicy):
        def act_batch(self, observations, state, resets=None, deterministic=False):
            actions, state = FakePolicy.act_batch(self, observations, state, resets)
            return actions[:-1], state

    driver, env, insts = driver_over(
        [SideAssignment(blue="p", red="p") for _ in range(2)], {"p": Short("p")}
    )
    driver.start()
    with pytest.raises(VecEnvFailure, match="would silently misroute"):
        driver.step()


def test_reset_flag_is_true_on_the_first_step_and_false_after():
    pol = FakePolicy("p")
    driver, env, insts = driver_over([SideAssignment(blue="p", red=None)], {"p": pol})
    driver.start()
    driver.step()
    driver.step()
    assert pol.seen_resets[0] == [True]
    assert pol.seen_resets[1] == [False]


def test_a_dead_slot_keeps_its_column_but_loses_its_action():
    insts = [FakeInstance(0), FakeInstance(1, die_after_sends=0)]
    pol = FakePolicy("p")
    driver, env, _ = driver_over(
        [SideAssignment(blue="p", red=None) for _ in range(2)],
        {"p": pol},
        instances=insts,
    )
    driver.start()
    driver.step()  # instance 1 dies and is restarted
    driver.step()
    # The batch width never changed, so the recurrent state stayed aligned.
    assert pol.batch_sizes == [2, 2]
    # ...and the restarted instance had its recurrent column reset.
    assert pol.seen_resets[1][1] is True


# -- episode boundaries ----------------------------------------------------


def test_episode_ends_at_ten_minutes_and_triggers_an_in_process_reset():
    insts = [FakeInstance(i, step_ms=200_000) for i in range(2)]
    pol = FakePolicy("p")
    driver, env, _ = driver_over(
        [SideAssignment(blue="p", red=None) for _ in range(2)],
        {"p": pol},
        instances=insts,
        episode=EpisodeSpec(max_game_ms=600_000),
    )
    driver.start()
    for _ in range(3):
        result, dones = driver.step()
    assert all(i.resets >= 1 for i in insts)
    assert driver.episode_index[0] >= 1


def test_episode_done_predicate():
    spec = EpisodeSpec(max_game_ms=600_000)
    assert episode_done(make_obs(0), spec, 0) == (False, "")
    assert episode_done(make_obs(600_000), spec, 0)[0] is True
    dead = make_obs(1000, blue_hp=0)
    ok, reason = episode_done(dead, spec, 0)
    assert ok and reason == "death_team_100"
    assert episode_done(make_obs(0), EpisodeSpec(max_steps=5), 5) == (True, "max_steps")
