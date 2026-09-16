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

from lanerl_jax.parity.inject import infer_turret_model
from lanerl_jax.sim.init import ALL_TURRETS
from lanerl_jax.sim.profiles import profile_id
from lanerl_jax.sim.state import Kind, Team


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
