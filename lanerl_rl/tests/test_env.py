"""Environment scaffolding, end to end over the real recording."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from lanerl_rl import constants as C
from lanerl_rl.env import (
    ControlBackend,
    JsonlReplayBackend,
    LaneEnv,
    LaneEnvConfig,
    RewardConfig,
    ServerCommand,
    decode_action,
    order_for_command,
)
from lanerl_rl.frame import ApproxFogModel
from lanerl_rl.model import LanePolicy, ModelConfig
from lanerl_rl.obs import ObservationBuilder
from lanerl_rl.ppo import DualClipPPO, PPOConfig, RecurrentRolloutBuffer
from lanerl_rl.reward import LaneRewardConfig, ZeroSumLaneReward
from lanerl_rl.scenarios import reflect_frame_in_lane, top_lane_scenario


@pytest.fixture
def env(recording_path):
    backend = JsonlReplayBackend(recording_path, max_frames=64)
    return LaneEnv(backend, LaneEnvConfig(max_steps=40, warn_on_approx_fog=False))


def test_reset_and_step(env):
    obs = env.reset()
    assert set(obs) == {C.TEAM_BLUE, C.TEAM_RED}
    for o in obs.values():
        assert o.entities.shape == (C.N_SLOTS, C.ENTITY_DIM)

    noop = {"button": C.BUTTON_INDEX["noop"], "move_x": 4, "move_z": 4, "target": 0}
    actions = {t: dict(noop) for t in obs}
    for _ in range(10):
        obs, rew, done, info = env.step(actions)
        assert set(rew) == {C.TEAM_BLUE, C.TEAM_RED}
        assert all(np.isfinite(v) for v in rew.values())
        if done:
            break
    env.close()


def test_zero_sum_reward_is_antisymmetric(env):
    """At alpha = 1 with shaping off, ``ZeroSumLaneReward`` is exactly antisymmetric.

    Both knobs matter: the default anneal starts at alpha = 0.5, and the
    last-hit shaping term is per-agent and deliberately not antisymmetric (it is
    policy-invariant instead).  Leaving either at its default and asserting
    ``r_blue == -r_red`` would be asserting a bug.
    """
    env.cfg.reward = LaneRewardConfig(
        zero_sum_alpha_start=1.0, zero_sum_alpha_end=1.0, last_hit_shaping=False
    )
    env.reward = ZeroSumLaneReward(env.cfg.teams, env.cfg.reward)
    env.reset()
    noop = {"button": 0, "move_x": 4, "move_z": 4, "target": 0}
    for _ in range(20):
        _obs, rew, done, _ = env.step({t: dict(noop) for t in env.cfg.teams})
        a, b = env.cfg.teams
        assert rew[a] == pytest.approx(-rew[b], abs=1e-6)
        if done:
            break
    env.close()


def test_slot_netids_track_the_observation(env):
    obs = env.reset()
    for team, o in obs.items():
        netids = env._slot_netids[team]
        assert len(netids) == C.N_SLOTS
        for s in range(C.N_SLOTS):
            occupied = o.entities[s, C.E_VALID] > 0.5 or o.entities[s, C.E_STALENESS] > 0.0
            if occupied:
                assert netids[s] is not None, f"slot {s} occupied but has no net id"
    env.close()


def test_move_action_decodes_into_world_space_for_both_sides():
    """One lane-local intent -> the same lane-local click for both agents.

    ``move_x`` / ``move_z`` address the lane axes (s, n), and
    ``LaneTransform.vector`` maps that pair back into world space.  The test
    that matters is not "the world displacements are opposite" -- they are
    reflections of one another, not negations -- but that both agents' clicks
    land on the same point of *their own* lane frame.  That is the property
    that lets one set of weights drive both sides.
    """
    fog = ApproxFogModel(warn=False)
    f = top_lane_scenario()
    f2 = reflect_frame_in_lane(f)

    # REAL anchors on both sides -- what a deployed red agent actually has.
    blue = ObservationBuilder(C.TEAM_BLUE, fog_model=fog)
    red = ObservationBuilder(C.TEAM_RED, fog_model=fog)
    ob, orr = blue.build(f), red.build(f2)

    action = {"button": C.BUTTON_INDEX["move"], "move_x": C.N_MOVE_BINS - 1, "move_z": 4, "target": 0}
    me_b = f.champion_of_team(C.TEAM_BLUE)
    me_r = f2.champion_of_team(C.TEAM_RED)
    cmd_b = decode_action(action, blue, ob, me_b, [None] * C.N_SLOTS)
    cmd_r = decode_action(action, red, orr, me_r, [None] * C.N_SLOTS)

    cb = blue.transform.point(cmd_b.x, cmd_b.y)
    cr = red.transform.point(cmd_r.x, cmd_r.y)
    assert cb[0] == pytest.approx(cr[0], abs=1e-6)
    assert cb[1] == pytest.approx(cr[1], abs=1e-6)

    # "move_x at its maximum" means "straight down the lane, towards them", and
    # it must mean that for both agents -- i.e. s increases by the full move
    # distance in each agent's own frame.
    sb = blue.transform.point(me_b.x, me_b.y)
    sr = red.transform.point(me_r.x, me_r.y)
    assert cb[0] - sb[0] == pytest.approx(500.0, abs=1e-6)
    assert cr[0] - sr[0] == pytest.approx(500.0, abs=1e-6)


def test_move_head_axes_are_the_lane_axes():
    """A pure move_z click must not change lane progress at all."""
    fog = ApproxFogModel(warn=False)
    f = top_lane_scenario()
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=fog)
    o = b.build(f)
    me = f.champion_of_team(C.TEAM_BLUE)
    action = {"button": C.BUTTON_INDEX["move"], "move_x": 4, "move_z": C.N_MOVE_BINS - 1, "target": 0}
    cmd = decode_action(action, b, o, me, [None] * C.N_SLOTS)
    s0, n0 = b.transform.point(me.x, me.y)
    s1, n1 = b.transform.point(cmd.x, cmd.y)
    assert s1 - s0 == pytest.approx(0.0, abs=1e-6)
    assert n1 - n0 == pytest.approx(500.0, abs=1e-6)


def test_decode_action_button_kinds():
    fog = ApproxFogModel(warn=False)
    f = top_lane_scenario()
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=fog)
    o = b.build(f)
    me = f.champion_of_team(C.TEAM_BLUE)
    netids = [None] * C.N_SLOTS
    netids[0] = 1002

    for name, kind in (
        ("noop", "noop"),
        ("recall", "recall"),
        ("move", "move"),
        ("attack_move", "attack_move"),
        ("q", "cast"),
        ("r", "cast"),
    ):
        a = {"button": C.BUTTON_INDEX[name], "move_x": 6, "move_z": 2, "target": 0}
        cmd = decode_action(a, b, o, me, netids)
        assert cmd.kind == kind
        if kind == "cast":
            assert cmd.spell_slot is not None
        if kind in ("move", "attack_move", "cast"):
            assert cmd.x is not None and cmd.y is not None


def test_target_netid_is_dropped_for_invalid_slots():
    fog = ApproxFogModel(warn=False)
    f = top_lane_scenario()
    b = ObservationBuilder(C.TEAM_BLUE, fog_model=fog)
    o = b.build(f)
    me = f.champion_of_team(C.TEAM_BLUE)
    invalid = next(s for s in range(C.N_SLOTS) if o.entities[s, C.E_VALID] < 0.5)
    netids = [99 for _ in range(C.N_SLOTS)]
    a = {"button": C.BUTTON_INDEX["attack_move"], "move_x": 4, "move_z": 4, "target": invalid}
    cmd = decode_action(a, b, o, me, netids)
    assert cmd.target_netid is None


def test_rollout_through_env_then_one_ppo_update(recording_path):
    """The whole loop: recording -> obs -> policy -> buffer -> dual-clip PPO."""
    torch.manual_seed(0)
    backend = JsonlReplayBackend(recording_path, max_frames=64, skip=1200)
    env = LaneEnv(backend, LaneEnvConfig(max_steps=1000, warn_on_approx_fog=False))
    policy = LanePolicy(ModelConfig())
    teams = list(env.cfg.teams)
    n_steps = 24
    buf = RecurrentRolloutBuffer(n_steps, len(teams), policy.cfg)
    state = policy.initial_state(len(teams))

    obs = env.reset()

    def to_batch(obs_map):
        def st(attr):
            return torch.from_numpy(np.stack([getattr(obs_map[t], attr) for t in teams])).unsqueeze(1)

        return {
            "entities": st("entities"),
            "entity_pad_mask": st("entity_pad_mask"),
            "self_vec": st("self_vec"),
            "global_vec": st("global_vec"),
            "priv_entities": st("priv_entities"),
            "priv_pad_mask": st("priv_pad_mask"),
            "priv_vec": st("priv_vec"),
        }

    def to_masks(obs_map):
        return {
            k: torch.from_numpy(
                np.stack([getattr(obs_map[t].action_mask, k) for t in teams])
            ).unsqueeze(1)
            for k in ("button", "move_x", "move_z", "target")
        }

    for step in range(n_steps):
        batch = to_batch(obs)
        masks = to_masks(obs)
        action, logp, value, new_state = policy.act(
            {**batch, "action_masks": masks}, state
        )
        env_actions = {
            t: {k: int(action[k][i, 0]) for k in ("button", "move_x", "move_z", "target")}
            for i, t in enumerate(teams)
        }
        nxt, rew, done, _info = env.step(env_actions)
        buf.add(
            obs={k: v[:, 0] for k, v in batch.items()},
            masks={k: v[:, 0] for k, v in masks.items()},
            action={k: action[k][:, 0] for k in action},
            log_prob=logp[:, 0],
            value=value[:, 0],
            reward=torch.tensor([rew[t] for t in teams], dtype=torch.float32),
            done=torch.tensor([float(done)] * len(teams)),
            reset=torch.zeros(len(teams)),
            state=state,
        )
        state = new_state
        obs = nxt
        if done:
            break

    assert buf.step == n_steps, "the replay backend ran out of frames early"
    buf.finish(torch.zeros(len(teams)), gamma=0.99, lam=0.99)
    assert torch.isfinite(buf.advantages[: buf.step]).all()

    before = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    trainer = DualClipPPO(policy, PPOConfig(chunk_len=8, burn_in=4, minibatch_chunks=2, epochs=1))
    stats = trainer.update(buf)
    assert np.isfinite(stats["loss"])
    after = policy.state_dict()
    assert any(
        v.dtype.is_floating_point and not torch.equal(v, before[k]) for k, v in after.items()
    )
    env.close()


# --------------------------------------------------------------------------
# The control-channel wire contract (no server needed)
# --------------------------------------------------------------------------


def test_order_for_command_matches_the_wire_contract():
    assert order_for_command(None) == {"t": "noop"}
    assert order_for_command(ServerCommand(kind="noop")) == {"t": "noop"}
    assert order_for_command(ServerCommand(kind="move", x=1.5, y=-2.5)) == {
        "t": "move",
        "x": 1.5,
        "y": -2.5,
    }
    assert order_for_command(
        ServerCommand(kind="attack_move", x=1.0, y=2.0, target_netid=4242)
    ) == {"t": "attack", "id": 4242}
    cast = order_for_command(
        ServerCommand(kind="cast", spell_slot=2, x=10.0, y=20.0, target_netid=7)
    )
    assert cast == {"t": "cast", "slot": 2, "id": 7, "x": 10.0, "y": 20.0}


def test_targetless_attack_move_becomes_a_move():
    """No target selected -> walk there.  Closing the distance is the policy's job."""
    order = order_for_command(ServerCommand(kind="attack_move", x=3.0, y=4.0))
    assert order == {"t": "move", "x": 3.0, "y": 4.0}


def test_untargeted_cast_still_carries_an_id():
    """``LanerlControl.Num`` returns NaN for a missing key and casts it to uint."""
    order = order_for_command(ServerCommand(kind="cast", spell_slot=0, x=1.0, y=1.0))
    assert order["id"] == 0


def test_recall_becomes_a_real_order_not_a_silent_noop():
    """``LanerlControl`` casts the blue pill for this; it is not dropped."""
    assert order_for_command(ServerCommand(kind="recall")) == {"t": "recall"}


def test_recalling_comes_from_the_server_channel_not_the_button_press():
    """The 8 s channel outlives the order, so the order is not the signal."""
    from lanerl_rl.frame import decode_frame

    raw = {
        "t": 1000,
        "u": [
            {
                "id": 1,
                "k": "Champion",
                "tm": C.TEAM_BLUE,
                "x": 0,
                "y": 0,
                "hp": 100,
                "mhp": 100,
                "rc": 1,
            }
        ],
    }
    assert decode_frame(raw).units[1].recalling is True
    raw["u"][0]["rc"] = 0
    assert decode_frame(raw).units[1].recalling is False
    del raw["u"][0]["rc"]
    assert decode_frame(raw).units[1].recalling is None  # pre-"rc" recording


def test_coordinates_never_reach_the_wire_in_exponent_form():
    """``LanerlControl.Num`` scans digits/.-+ only; an 'e' would truncate the value."""
    order = order_for_command(ServerCommand(kind="move", x=1e-7, y=-1.2345e-9))
    text = json.dumps(order)
    assert "e" not in text.replace('"t"', "").replace("move", "")


def test_no_ordinary_action_line_can_trigger_a_reset():
    """``OnTick`` resets on any line *containing* ``"reset"`` -- a substring test."""
    backend = ControlBackend.__new__(ControlBackend)
    line = json.dumps(
        ControlBackend.encode(
            backend,
            {
                C.TEAM_BLUE: ServerCommand(kind="attack_move", x=1.0, y=2.0, target_netid=9),
                C.TEAM_RED: ServerCommand(kind="cast", spell_slot=3, x=5.0, y=6.0),
            },
        ),
        separators=(",", ":"),
    )
    assert '"reset"' not in line


def test_step_returns_none_when_the_channel_closes_rather_than_hanging():
    """Episode over is a return value, not an exception and never a hang."""
    import socket as _socket

    ours, theirs = _socket.socketpair()
    backend = ControlBackend(config_path=__file__)
    backend.sock = ours
    backend._dead = False
    theirs.close()  # the "server" exits
    try:
        assert backend.step({}) is None
        assert backend.step({}) is None  # and stays None, without touching the socket
    finally:
        ours.close()


def test_the_scripted_bot_is_off_by_default():
    """``LanerlConfig.DriveTeams`` defaults to "blue" -- unset means bot-driven blue."""
    backend = ControlBackend(config_path=__file__)  # nothing is launched here
    backend.control_port = 1
    assert backend._environment()["LANERL_BOT"] == "none"

    frozen = ControlBackend(config_path=__file__, bot_teams="purple")
    frozen.control_port = 1
    assert frozen._environment()["LANERL_BOT"] == "purple"


def test_missing_team_becomes_an_explicit_noop():
    backend = ControlBackend.__new__(ControlBackend)
    encoded = ControlBackend.encode(backend, {})
    assert encoded == {"blue": {"t": "noop"}, "red": {"t": "noop"}}


def test_note_attack_fires_only_on_a_targeted_attack_move(env):
    """Without this wiring the three attack-cycle features are permanently 0."""
    obs = env.reset()
    team = C.TEAM_BLUE
    builder = env.builders[team]
    assert builder.attack_clock.last_attack_ms is None

    noop = {"button": C.BUTTON_INDEX["noop"], "move_x": 4, "move_z": 4, "target": 0}
    env.decode({team: noop})
    assert builder.attack_clock.last_attack_ms is None, "a noop must not note a swing"

    slot = next(
        s
        for s in range(C.N_SLOTS)
        if obs[team].entities[s, C.E_VALID] > 0.5 and env._slot_netids[team][s] is not None
    )
    am = {"button": C.BUTTON_INDEX["attack_move"], "move_x": 4, "move_z": 4, "target": slot}
    cmds = env.decode({team: am})
    assert cmds[team].target_netid is not None
    assert builder.attack_clock.last_attack_ms == env.frame.t_ms

    # ... and the feature it feeds is no longer stuck at "unknown".
    o = builder.build(env.frame)
    assert o.self_vec[C.SELF_FIELD_NAMES.index("attack_timing_known")] == 1.0
    env.close()


def test_lane_env_uses_the_zero_sum_reward():
    """The env must be on ``reward.ZeroSumLaneReward``, not the legacy sum."""
    assert isinstance(LaneEnvConfig().reward, LaneRewardConfig)


def test_replay_backend_ignores_actions_as_documented(recording_path):
    b1 = JsonlReplayBackend(recording_path, max_frames=5)
    b2 = JsonlReplayBackend(recording_path, max_frames=5)
    f1 = b1.reset()
    f2 = b2.reset()
    assert f1.t_ms == f2.t_ms
    n1 = b1.step({})
    n2 = b2.step({C.TEAM_BLUE: ServerCommand(kind="move", x=1.0, y=2.0)})
    assert n1.t_ms == n2.t_ms
    assert b1.ignores_actions is True
    b1.close()
    b2.close()
