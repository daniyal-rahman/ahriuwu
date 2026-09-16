"""Turret model inference for the Tier-1 injector.

`inject_snapshot` writes a recorded server state into the JAX sim so a
one-step differential can run against it. Every field it cannot recover
honestly is refused rather than guessed (`infer_minion_model`'s contract);
`infer_turret_model` exists because a turret's TIER used to not need
recovering at all -- every turret shared one profile, so `next(profiles for
this team)` always found the only candidate. Once turrets carry per-tier
profiles that shortcut silently starts returning the OUTER row for every
turret regardless of which one it actually is, which is exactly the kind of
regression a one-step diff would otherwise blame on the wrong mechanic.
"""
from __future__ import annotations

import pytest

from lanerl_jax.parity.inject import (
    infer_turret_model,
    reconstruct_waypoints,
    reconstruct_waypoints_relaxed,
    replay_wave_states,
)
from lanerl_jax.parity.trace import Snapshot
from lanerl_jax.sim.init import ALL_TURRETS, TOP_LANE_PATH
from lanerl_jax.sim.profiles import profile_id
from lanerl_jax.sim.state import Kind, Team
from lanerl_jax.sim.waves import FIRST_WAVE_MS, step_waves


def test_infer_turret_model_recovers_every_tier_not_just_outer():
    """Before this existed, injecting ANY turret entity resolved to
    ``next(r for r, (k, _, t) in enumerate(profiles) if k == TURRET and t ==
    team)`` -- the first turret row for that team, full stop. With one row per
    team that was harmless by construction; with five it silently mislabels
    every inner, inhibitor, nexus and fountain turret as an outer one. This
    walks every real placed turret and checks the recovered row matches its
    OWN tier, not always the first (outer) one.
    """
    for team, x, y, _hp, tier in ALL_TURRETS:
        row, reason = infer_turret_model(x, y, team)
        assert row == profile_id(Kind.TURRET, tier, team), (
            f"({x},{y}) team={team}: got row {row} ({reason}), "
            f"expected tier {tier}'s row")


def test_infer_turret_model_matches_within_the_dumps_own_quantisation():
    """The dump quantises to 1/16 of a unit; a position off by a fraction of
    that must still resolve to the same turret, not the nearest DIFFERENT one.
    """
    team, x, y, _hp, tier = ALL_TURRETS[1]     # blue's top outer turret
    row, _ = infer_turret_model(x + 1 / 32, y - 1 / 16, team)
    assert row == profile_id(Kind.TURRET, tier, team)


def test_infer_turret_model_refuses_rather_than_guessing_when_far_off():
    """No known turret sits at the champion spawn or in the middle of a
    lane -- and this must say so, not hand back its nearest (irrelevant)
    match. `infer_minion_model` refuses the same way for an unrecognised
    minion; this is the turret side of that same contract.
    """
    row, reason = infer_turret_model(6000.0, 6000.0, Team.BLUE)
    assert row is None
    assert "too far" in reason


def test_infer_turret_model_does_not_cross_teams():
    """Blue's outer turret position must not resolve to a RED profile row even
    though `ALL_TURRETS` also has red entries -- the team argument has to
    actually gate the search, not just tiebreak it.
    """
    team, x, y, _hp, tier = ALL_TURRETS[1]     # blue's top outer turret
    assert team == Team.BLUE
    row, _ = infer_turret_model(x, y, Team.BLUE)
    assert row == profile_id(Kind.TURRET, tier, Team.BLUE)
    # Asking for RED at BLUE's outer turret's position must not silently
    # return blue's row -- it must fail to find a nearby RED turret instead
    # (the nearest red turret is thousands of units away).
    row_wrong_team, reason = infer_turret_model(x, y, Team.RED)
    assert row_wrong_team is None, reason


def test_replay_wave_states_does_not_double_count_the_pairing_ticks_own_spawn():
    """Regression for the off-by-one this module's docstring now documents:
    an earlier version paired `trace[i]` with the counters as they stood
    BEFORE `trace[i]`'s own tick, so if `trace[i]` was itself a spawn tick
    (its own population already includes the new minion, since the dump is
    written after that tick's LevelScript.Update ran), `tick()` -- stepping
    FROM the injected `trace[i]` -- would re-evaluate the exact same
    threshold at the exact same game time and spawn AGAIN on top of the unit
    already in the dump.

    `replay_wave_states` must instead return, for index i, counters that
    already reflect `trace[i].t_ms`'s own decision: re-running `step_waves`
    at that SAME game time must spawn nothing more.
    """
    snaps = [Snapshot(t_ms=int(FIRST_WAVE_MS)), Snapshot(t_ms=int(FIRST_WAVE_MS) + 17)]
    states = replay_wave_states(snaps)
    assert len(states) == 2

    # The state paired with the FIRST (spawn) tick must already have consumed
    # that tick's own arrival -- one minion per barrack has already spawned.
    assert states[0].minion_number == 1

    # Re-evaluating step_waves at the SAME game time the state was already
    # advanced through must be a no-op: the bug this guards against would
    # instead spawn a second minion here.
    spawned_again = step_waves(states[0], float(snaps[0].t_ms))
    assert spawned_again == [], (
        "replay_wave_states paired trace[i] with pre-trace[i] counters -- "
        "the sim will spawn on top of a minion the dump already shows")


def test_reconstruct_waypoints_relaxed_never_refuses():
    """Unlike the strict (exact) reconstruction, the relaxed fallback must
    always return a usable waypoint array -- that is its entire purpose: a
    guess for the case the strict version refuses, not a second refusal."""
    # Far off any known corridor vertex -- the strict version refuses this.
    off_x, off_y = 6000.0, 6000.0
    strict = reconstruct_waypoints(off_x, off_y, Team.BLUE)
    assert strict[0] is None, "test fixture should be a case the strict path refuses"

    relaxed = reconstruct_waypoints_relaxed(off_x, off_y, Team.BLUE)
    wp, key, n, reason = relaxed
    assert wp is not None
    assert n == len(TOP_LANE_PATH)
    assert 0 <= key < n
    assert "guess" in reason.lower()


def test_reconstruct_waypoints_relaxed_agrees_with_strict_on_corridor():
    """On the corridor, where the strict path succeeds, the relaxed fallback
    should recover the identical waypoint key -- it is the same projection,
    just without the tolerance gate."""
    x, y = TOP_LANE_PATH[3]
    strict_wp, strict_key, strict_n, _ = reconstruct_waypoints(x, y, Team.BLUE)
    relaxed_wp, relaxed_key, relaxed_n, _ = reconstruct_waypoints_relaxed(
        x, y, Team.BLUE)
    assert strict_wp is not None
    assert relaxed_key == strict_key
    assert relaxed_n == strict_n
