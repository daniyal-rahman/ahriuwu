"""The closed loop, against a **real** headless server.

Everything here boots ``GameServerConsole`` and drives it over
``LanerlControl``'s TCP channel.  These are the tests that can tell the
difference between "the plumbing type-checks" and "the policy moved a champion",
so every assertion is on a concrete state change: a position delta, a minion's
hp dropping, the game clock advancing, the champion standing in its fountain.

Two rules this file exists to obey
----------------------------------
*A test that passes when the server never started is worse than no test.*  So a
missing binary is a **failure**, not a skip, unless ``LANERL_SKIP_SERVER_TESTS``
is set deliberately; and every step asserts the frame it got back is not
``None``, which is what the backend returns when the channel dies.

*A broken ``Content/`` script degrades to a silent no-op.*  Those scripts are
compiled at runtime by Roslyn and a failure only ever shows as a "Could not find
script" line in the server log, so :func:`test_server_log_is_clean` reads the log
back and fails on it.

Run them one file at a time, on the Slurm node, never on the login node::

    sbatch -p cpu -w desktop -c 6 -t 40 --chdir=/mnt/nfs/projects/ahriuwu-lanerl \\
      --wrap 'python -m pytest lanerl_rl/tests/test_control_backend.py -x -q -s'
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pytest

from lanerl_rl import constants as C
from lanerl_rl.env import (
    DEFAULT_DOTNET_ROOT,
    DEFAULT_GAME_CONFIG,
    DEFAULT_SERVER_DIR,
    ControlBackend,
    LaneEnv,
    LaneEnvConfig,
    ServerCommand,
)
from lanerl_rl.frame import Frame, Unit

#: 4 ticks of the server's 60 Hz loop -> 15 Hz decisions, 66.67 ms of game time.
STEP_TICKS = 4
MS_PER_STEP = 1000.0 * STEP_TICKS / 60.0

#: Garen's auto-attack range; the engine only swings at targets already in range.
AA_REACH = C.AA_RANGE_GAREN + C.TARGET_RADIUS["minion"]

LOG_DIR = Path(__file__).resolve().parents[2] / "lanerl" / "logs"


def _require_server() -> None:
    """Fail, rather than skip, when the server is missing.

    A silent skip here would let the whole file report green on a node with no
    built binary -- which is precisely the failure mode these tests exist to
    rule out.  ``LANERL_SKIP_SERVER_TESTS=1`` is the deliberate opt-out.
    """
    if os.environ.get("LANERL_SKIP_SERVER_TESTS") == "1":
        pytest.skip("LANERL_SKIP_SERVER_TESTS=1")
    direct = DEFAULT_SERVER_DIR / "GameServerConsole"
    dll = DEFAULT_SERVER_DIR / "GameServerConsole.dll"
    if not direct.exists() and not (dll.exists() and (DEFAULT_DOTNET_ROOT / "dotnet").exists()):
        raise AssertionError(
            f"no built server at {DEFAULT_SERVER_DIR}. Build it before running these "
            f"tests; do not let them skip."
        )
    if not DEFAULT_GAME_CONFIG.exists():
        raise AssertionError(f"missing game config {DEFAULT_GAME_CONFIG}")


@pytest.fixture(scope="module")
def backend():
    """One server for the whole module.  Booting costs ~12 s; resetting costs ~0.1 ms."""
    _require_server()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    b = ControlBackend(step_ticks=STEP_TICKS, log_path=LOG_DIR / "test_control_backend.log")
    try:
        yield b
    finally:
        b.close()


# -- helpers ---------------------------------------------------------------


def _step(backend: ControlBackend, blue=None, red=None) -> Frame:
    frame = backend.step(
        {
            C.TEAM_BLUE: blue if blue is not None else ServerCommand(kind="noop"),
            C.TEAM_RED: red if red is not None else ServerCommand(kind="noop"),
        }
    )
    assert frame is not None, "the server closed the control channel mid-episode"
    return frame


def _drive(backend: ControlBackend, n: int, blue=None, red=None) -> Frame:
    frame = None
    for _ in range(n):
        frame = _step(backend, blue, red)
    assert frame is not None
    return frame


def _me(frame: Frame, team: int = C.TEAM_BLUE) -> Unit:
    u = frame.champion_of_team(team)
    assert u is not None, f"no champion for team {team} in the observation"
    return u


def _enemy_minions(frame: Frame, team: int = C.TEAM_BLUE) -> List[Unit]:
    foe = C.TEAM_RED if team == C.TEAM_BLUE else C.TEAM_BLUE
    return [u for u in frame.units.values() if u.etype == "minion" and u.team == foe and u.alive]


def _dist(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


#: Move orders are a straight line -- ``LanerlControl`` hands the engine a
#: two-point waypoint list, with no pathfinding -- so a champion walked at a
#: wall simply stops.  Which direction out of the fountain is clear is a
#: property of the map, not of this test, so try several and keep the one that
#: makes progress.
_AWAY_DIRECTIONS = ((0.0, 1.0), (1.0, 1.0), (1.0, 0.0), (1.0, -1.0), (0.0, -1.0), (-1.0, 1.0))


def _walk_away(backend: ControlBackend, origin, min_dist: float = 500.0, budget: int = 600) -> Frame:
    """Walk the blue champion at least ``min_dist`` from ``origin``."""
    frame = _step(backend)
    best = 0.0
    for d, (dx, dy) in enumerate(_AWAY_DIRECTIONS):
        target = (origin[0] + 2500.0 * dx, origin[1] + 2500.0 * dy)
        stalled = 0.0
        for i in range(budget // len(_AWAY_DIRECTIONS)):
            frame = _step(backend, blue=ServerCommand(kind="move", x=target[0], y=target[1]))
            me = _me(frame)
            got = _dist(origin, (me.x, me.y))
            best = max(best, got)
            if got >= min_dist:
                return frame
            if i % 40 == 39:
                if got - stalled < 30.0:
                    break  # this direction is into a wall; try the next one
                stalled = got
    raise AssertionError(
        f"could not walk {min_dist:.0f} units from {origin} in {budget} steps (best {best:.0f})"
    )


# -- the tests -------------------------------------------------------------


def test_lockstep_holds_and_the_clock_advances(backend):
    """One observation per action, and the simulation really moves forward."""
    f0 = backend.reset()
    assert f0.champion_of_team(C.TEAM_BLUE) is not None
    assert f0.champion_of_team(C.TEAM_RED) is not None

    n = 30
    f1 = _drive(backend, n)
    dt = f1.t_ms - f0.t_ms
    expected = n * MS_PER_STEP
    print(f"\n[lockstep] {n} steps advanced {dt} ms (expected ~{expected:.0f} ms)")
    assert dt > 0, "the game clock did not advance"
    assert abs(dt - expected) < 0.25 * expected, f"{dt} ms is not ~{expected:.0f} ms"

    # Per-team fog flags, so the actor/critic split is feedable from this channel.
    assert all(u.visible_to is not None for u in f1.units.values())
    # And the champion's own recall-channel state, which is new.
    assert _me(f1).recalling is False


def test_a_move_order_moves_the_champion(backend):
    """The capability that did not exist before the control channel."""
    f0 = backend.reset()
    me0 = _me(f0)
    start = (me0.x, me0.y)
    target = (start[0] + 1200.0, start[1] + 1200.0)

    f1 = _drive(backend, 120, blue=ServerCommand(kind="move", x=target[0], y=target[1]))
    me1 = _me(f1)
    moved = _dist(start, (me1.x, me1.y))
    print(
        f"[move] blue travelled {moved:.0f} units: "
        f"({start[0]:.0f},{start[1]:.0f}) -> ({me1.x:.0f},{me1.y:.0f})"
    )
    assert moved > 200.0, f"the champion did not move (only {moved:.0f} units)"

    # A noop is action-repeat, not a stop: League orders persist, so the champion
    # holds its ground instead of returning.  Not asserted as monotone progress --
    # a champion that has run into terrain settles back a few tens of units.
    me2 = _me(_drive(backend, 10))
    assert _dist(start, (me2.x, me2.y)) > 0.9 * moved, "a noop must not undo the move order"


def test_an_attack_order_damages_a_minion(backend):
    """Attack ordering, over the real engine, against a real minion."""
    frame = backend.reset()
    # The first wave spawns at 90 s; free-run gets there in a few wall seconds.
    while frame.t_ms < 100_000:
        frame = _step(backend)
    minions = _enemy_minions(frame)
    assert minions, f"no enemy minions at t={frame.t_ms} ms; the test would prove nothing"

    me = _me(frame)
    target = min(minions, key=lambda u: _dist((me.x, me.y), (u.x, u.y)))
    tid, hp0 = target.id, target.hp
    print(
        f"[attack] target minion {tid} hp={hp0:.0f} at ({target.x:.0f},{target.y:.0f}); "
        f"champ at ({me.x:.0f},{me.y:.0f}), dist={_dist((me.x, me.y), (target.x, target.y)):.0f}, "
        f"aa reach {AA_REACH:.0f}"
    )

    # Walk into range first: `attack` only swings at targets already in range,
    # which is deliberate -- closing the distance is the policy's job.
    for _ in range(400):
        cur = frame.units.get(tid)
        if cur is None or not cur.alive:
            break
        me = _me(frame)
        if _dist((me.x, me.y), (cur.x, cur.y)) <= AA_REACH:
            break
        frame = _step(backend, blue=ServerCommand(kind="move", x=cur.x, y=cur.y))

    hp_seen = hp0
    killed = False
    for _ in range(300):
        frame = _step(backend, blue=ServerCommand(kind="attack_move", target_netid=tid))
        cur = frame.units.get(tid)
        if cur is None or not cur.alive:
            killed = True
            break
        hp_seen = cur.hp

    print(f"[attack] minion {tid} hp {hp0:.0f} -> {hp_seen:.0f}{' (killed)' if killed else ''}")
    assert killed or hp_seen < hp0, "the attack order did no damage"


def test_recall_channels_then_teleports_to_the_fountain(backend):
    """Recall is a real 8 s channel, not an instant teleport and not a no-op."""
    f0 = backend.reset()
    fountain = (_me(f0).x, _me(f0).y)  # LanerlEpisode.Reset puts champions at spawn

    away = _walk_away(backend, fountain, min_dist=600.0)
    me = _me(away)
    d_away = _dist(fountain, (me.x, me.y))
    print(f"\n[recall] walked {d_away:.0f} units from the fountain {fountain}")
    assert d_away > 400.0, "the champion never left the fountain; the test would prove nothing"
    assert _me(away).recalling is False

    frame = _step(backend, blue=ServerCommand(kind="recall"))
    channelled = False
    home_at: Optional[int] = None
    t_start = frame.t_ms
    for i in range(200):  # 200 steps = 13.3 s of game time; the channel is 8.5 s
        frame = _step(backend)  # noop: a move order would cancel the channel
        if _me(frame).recalling:
            channelled = True
        elif channelled and home_at is None:
            home_at = frame.t_ms
            break

    me = _me(frame)
    d_home = _dist(fountain, (me.x, me.y))
    print(
        f"[recall] channelled={channelled} channel_ms={None if home_at is None else home_at - t_start} "
        f"final dist from fountain={d_home:.0f}"
    )
    assert channelled, "the recall order never started a channel (rc stayed 0)"
    assert d_home < 300.0, f"recall did not return the champion to the fountain ({d_home:.0f} units away)"


def test_a_move_order_cancels_the_recall_channel(backend):
    """The cost of recalling is the window in which it can be interrupted."""
    f0 = backend.reset()
    fountain = (_me(f0).x, _me(f0).y)
    away = _walk_away(backend, fountain, min_dist=600.0)
    here = (_me(away).x, _me(away).y)

    frame = _step(backend, blue=ServerCommand(kind="recall"))
    frame = _drive(backend, 20)
    assert _me(frame).recalling, "precondition: the channel should be running"

    frame = _step(backend, blue=ServerCommand(kind="move", x=here[0] + 300.0, y=here[1] + 300.0))
    frame = _drive(backend, 5)
    print(f"[recall-cancel] recalling after a move order: {_me(frame).recalling}")
    assert not _me(frame).recalling, "a move order must cancel the channel"

    frame = _drive(backend, 200)
    d_home = _dist(fountain, (_me(frame).x, _me(frame).y))
    assert d_home > 400.0, "a cancelled recall must not teleport the champion home"


def test_in_process_reset_is_cheap_and_rewinds_the_clock(backend):
    """A restart costs ~12 s; the whole point of LanerlEpisode.Reset is not paying it."""
    backend.reset()
    frame = _drive(backend, 60)
    assert frame.t_ms > 0

    costs = []
    for _ in range(5):
        t0 = time.perf_counter()
        fresh = backend.reset()
        costs.append((time.perf_counter() - t0) * 1000.0)
        assert fresh.t_ms < MS_PER_STEP * 8, f"the clock did not rewind (t={fresh.t_ms} ms)"
        me = _me(fresh)
        assert me.hp == me.mhp, "reset did not full-heal the champion"
        assert me.lvl == 1, f"reset did not return the champion to level 1 (lvl={me.lvl})"
        frame = _drive(backend, 30)

    print(
        f"\n[reset] {len(costs)} in-process resets over the control channel: "
        f"min={min(costs):.1f} ms median={sorted(costs)[len(costs) // 2]:.1f} ms max={max(costs):.1f} ms"
    )
    # The floor is one step boundary (the server answers the reset with the next
    # observation), not the reset itself, so this bounds the round trip.
    assert max(costs) < 2000.0, f"a reset took {max(costs):.0f} ms; expected well under a second"


def test_a_policy_drives_the_env_end_to_end(backend):
    """The whole loop: server -> observation -> policy -> order -> server."""
    import torch

    from lanerl_rl.model import LanePolicy, ModelConfig

    torch.manual_seed(0)
    env = LaneEnv(
        backend,
        LaneEnvConfig(
            max_steps=400,
            end_on_death=False,
            end_on_turret_loss=False,
            warn_on_approx_fog=False,  # the control channel carries vb/vr
        ),
    )
    policy = LanePolicy(ModelConfig())
    teams = list(env.cfg.teams)

    obs = env.reset()
    start = {t: (env.frame.champion_of_team(t).x, env.frame.champion_of_team(t).y) for t in teams}
    state = policy.initial_state(len(teams))
    rewards = {t: 0.0 for t in teams}
    buttons_used = set()

    for _ in range(200):
        batch = {
            k: torch.from_numpy(np.stack([getattr(obs[t], k) for t in teams])).unsqueeze(1)
            for k in (
                "entities",
                "entity_pad_mask",
                "self_vec",
                "global_vec",
                "priv_entities",
                "priv_pad_mask",
                "priv_vec",
            )
        }
        masks = {
            k: torch.from_numpy(np.stack([getattr(obs[t].action_mask, k) for t in teams])).unsqueeze(1)
            for k in ("button", "move_x", "move_z", "target")
        }
        action, _logp, _value, state = policy.act({**batch, "action_masks": masks}, state)
        env_actions = {
            t: {k: int(action[k][i, 0]) for k in ("button", "move_x", "move_z", "target")}
            for i, t in enumerate(teams)
        }
        for a in env_actions.values():
            buttons_used.add(C.BUTTONS[a["button"]])
        obs, rew, done, info = env.step(env_actions)
        for t in teams:
            rewards[t] += rew[t]
        assert all(np.isfinite(v) for v in rew.values()), f"non-finite reward: {rew}"
        if done:
            break

    moved = {
        t: _dist(start[t], (env.frame.champion_of_team(t).x, env.frame.champion_of_team(t).y))
        for t in teams
    }
    print(
        f"\n[policy] {env.steps} steps, t={env.frame.t_ms} ms, "
        f"blue moved {moved[C.TEAM_BLUE]:.0f} units, red moved {moved[C.TEAM_RED]:.0f} units"
    )
    print(f"[policy] cumulative reward {rewards}, buttons pressed: {sorted(buttons_used)}")
    assert env.steps >= 200 or done
    assert max(moved.values()) > 100.0, "an untrained policy still has to move somebody"
    # The reward the env reports must be the new one, term breakdown and all.
    assert "terms" in info["reward_info"]


def test_server_log_is_clean(backend):
    """A Roslyn failure in Content/ degrades to a silent no-op; catch it here."""
    log = Path(backend.log_path)
    assert log.exists(), f"no server log at {log}"
    text = log.read_text(errors="ignore")
    missing = [ln for ln in text.splitlines() if "Could not find script" in ln]
    print(f"\n[log] {log}: {len(missing)} 'Could not find script' lines")
    for ln in missing[:8]:
        print("  script-load WARN:", ln.strip()[-120:])
    assert "LANERL_CONTROL error" not in text, "the control channel reported an error"
    assert "LANERL_RESET_WARN" not in text, "a champion failed to reset cleanly"
    # The scripted bot re-issues its own orders every tick. If it is attached,
    # every "the policy moved the champion" assertion above is worthless.
    attached = [ln for ln in text.splitlines() if "LANERL_BOT_ATTACH" in ln]
    assert not attached, f"the scripted bot was driving a champion: {attached}"
    # A raw count is not a useful gate: this fork has never shipped the
    # BasicAttack / Passive / turret / minion scripts, and the engine's default
    # behaviour covers them (the minion above still died to auto-attacks).  What
    # must load is exactly the kit the action space can address.
    needed = ("Spells.Recall", "Spells.GarenQ", "Spells.GarenW", "Spells.GarenE", "Spells.GarenR")
    broken = [n for n in needed if any(ln.rstrip().endswith(n) for ln in missing)]
    assert not broken, f"scripts the action space depends on failed to compile: {broken}"
