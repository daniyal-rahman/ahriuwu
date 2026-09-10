"""Red-side canonicalisation: a REFLECTION in lane coordinates, not a rotation.

The property under test is equivariance::

    obs(F, BLUE)  ==  obs(reflect_frame_in_lane(F), RED)

where the reflection maps blue-lane ``(s, n)`` to red-lane ``(s, n)``.  If that
holds, both agents genuinely draw from one distribution and a single set of
weights can play either side.

Why not the 180 degree rotation this file used to test
-----------------------------------------------------
Summoner's Rift is point-symmetric *as a whole*, so ``rot180`` maps the map
onto itself -- but it maps blue's TOP lane onto red's BOT lane.  A red agent
canonicalised that way is looking at a different corridor, with an oppositely
signed lane normal, from the blue agent it shares weights with.  The old test
did not catch this because it built the RED builder from ``rot180``-ed anchors,
i.e. it configured red to live in the rotated corridor and then checked that
rotation was self-consistent.  It was; it just was not the red side.  Every test
here uses the REAL red configuration (default anchors = red's own top-lane outer
turret), which is what a deployed red agent would have.

We still assert *exact* equality.  The lane basis is irrational, so the world
round trip carries ~1e-11 game units of float64 error -- eight orders of
magnitude below a float32 ulp at map scale, so the observations come out
bit-identical anyway.  Any drift beyond that would be a real bug.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

from lanerl_rl import constants as C
from lanerl_rl.frame import ApproxFogModel, MirrorTransform, rot180_point
from lanerl_rl.obs import ObservationBuilder
from lanerl_rl.scenarios import (
    reflect_frame_in_lane,
    rotate_frame,
    top_lane_lane_frames,
    top_lane_scenario,
    top_lane_sequence,
)


def _real_builders():
    """BLUE and RED with their REAL default anchors -- what deployment uses."""
    fog = ApproxFogModel(warn=False)
    return (
        ObservationBuilder(C.TEAM_BLUE, fog_model=fog),
        ObservationBuilder(C.TEAM_RED, fog_model=fog),
    )


# --------------------------------------------------------------------------
# The geometric facts the design rests on
# --------------------------------------------------------------------------


def test_rot180_maps_blue_top_onto_red_bot():
    """The measurement that rules the rotation out for a same-lane 1v1."""
    got = rot180_point(*C.TOP_OUTER_TURRET[C.TEAM_BLUE])
    assert got == (13409.0, 4227.0)
    # ... which is nowhere near red's TOP outer turret.
    assert np.hypot(*(np.array(got) - np.array(C.TOP_OUTER_TURRET[C.TEAM_RED]))) > 9000.0


def test_rot180_is_an_exact_involution():
    for x, y in [(0, 0), (574, 10220), (13927, 14175), (7000, 7000), (-236, -53)]:
        rx, ry = rot180_point(float(x), float(y))
        bx, by = rot180_point(rx, ry)
        assert bx == float(x)
        assert by == float(y)


def test_the_two_lane_frames_share_a_normal_and_oppose_their_axes():
    """The whole reflection story in two assertions."""
    lanes = top_lane_lane_frames()
    b, r = lanes[C.TEAM_BLUE], lanes[C.TEAM_RED]
    assert np.allclose(b.normal, r.normal), "the two agents must agree on which side is n > 0"
    assert np.allclose(np.array(b.axis), -np.array(r.axis)), "the axes must be antiparallel"
    assert b.length == pytest.approx(r.length)


def test_n_positive_means_the_same_physical_side_for_both_agents():
    """Both nexuses sit at n < 0 in both frames: the top lane is an edge lane.

    This is what makes the handedness rule ("own nexus at n < 0") produce a
    *shared* world normal rather than two conventions.
    """
    lanes = top_lane_lane_frames()
    for team, lane in lanes.items():
        for nexus_team, nexus in C.NEXUS_POSITION.items():
            _s, n = lane.point(*nexus)
            assert n < 0.0, (
                f"{C.TEAM_NAMES[nexus_team]} nexus is at n > 0 in "
                f"{C.TEAM_NAMES[team]}'s lane frame"
            )


def test_the_two_frames_are_related_by_a_reflection_not_a_rotation():
    """Determinant -1: read the same two world directions in both bases.

    A rotation preserves the sign of the 2D cross product; a reflection flips
    it.  This is the assertion that the chirality warning in ``LaneFrame`` is
    about, so it is worth stating as a test rather than a comment.
    """
    lanes = top_lane_lane_frames()
    b, r = lanes[C.TEAM_BLUE], lanes[C.TEAM_RED]
    d1, d2 = (1.0, 0.0), (0.3, 0.9)
    b1, b2 = b.vector(*d1), b.vector(*d2)
    r1, r2 = r.vector(*d1), r.vector(*d2)
    cross_b = b1[0] * b2[1] - b1[1] * b2[0]
    cross_r = r1[0] * r2[1] - r1[1] * r2[0]
    assert np.sign(cross_b) == -np.sign(cross_r)
    assert cross_b == pytest.approx(-cross_r)


def test_the_lane_map_between_the_agents_is_s_to_L_minus_s():
    lanes = top_lane_lane_frames()
    b, r = lanes[C.TEAM_BLUE], lanes[C.TEAM_RED]
    for p in [(2000.0, 11500.0), (574.0, 10220.0), (3911.0, 13654.0), (1500.0, 12800.0)]:
        sb, nb = b.point(*p)
        sr, nr = r.point(*p)
        assert sr == pytest.approx(b.length - sb, abs=1e-6)
        assert nr == pytest.approx(nb, abs=1e-6)


# --------------------------------------------------------------------------
# Observation equivariance, with the REAL red configuration
# --------------------------------------------------------------------------


def test_blue_and_red_observations_are_bit_identical_single_frame():
    blue, red = _real_builders()
    f = top_lane_scenario()
    f2 = reflect_frame_in_lane(f)
    ob = blue.build(f)
    orr = red.build(f2)

    assert np.array_equal(ob.entities, orr.entities)
    assert np.array_equal(ob.entity_pad_mask, orr.entity_pad_mask)
    assert np.array_equal(ob.self_vec, orr.self_vec)
    assert np.array_equal(ob.global_vec, orr.global_vec)
    assert np.array_equal(ob.action_mask.button, orr.action_mask.button)
    assert np.array_equal(ob.action_mask.target, orr.action_mask.target)


def test_blue_and_red_observations_are_bit_identical_over_a_trajectory():
    """Memory, velocity, hp deltas and the ability book are all stateful."""
    blue, red = _real_builders()
    frames = top_lane_sequence(n=40)
    for i, f in enumerate(frames):
        f2 = reflect_frame_in_lane(f)
        ob = blue.build(f)
        orr = red.build(f2)
        assert np.array_equal(ob.entities, orr.entities), f"entities differ at step {i}"
        assert np.array_equal(ob.entity_pad_mask, orr.entity_pad_mask), f"mask differs at step {i}"
        assert np.array_equal(ob.self_vec, orr.self_vec), f"self differs at step {i}"
        assert np.array_equal(ob.global_vec, orr.global_vec), f"global differs at step {i}"
        assert np.array_equal(ob.priv_entities, orr.priv_entities), f"priv entities differ at {i}"
        assert np.array_equal(ob.priv_vec, orr.priv_vec), f"priv differs at step {i}"


def test_mirroring_holds_with_server_supplied_visibility():
    blue, red = _real_builders()
    for i in range(15):
        f = top_lane_scenario(t_ms=90_000 + 100 * i, blue_s=0.3 + 0.01 * i, with_visibility=True)
        f2 = reflect_frame_in_lane(f)
        ob = blue.build(f)
        orr = red.build(f2)
        assert ob.fog_source == orr.fog_source == "server"
        assert np.array_equal(ob.entities, orr.entities), f"step {i}"
        assert np.array_equal(ob.global_vec, orr.global_vec), f"step {i}"


def test_mirroring_holds_for_the_enemy_ability_book():
    """The longest-memory feature in the observation must mirror too."""
    blue, red = _real_builders()
    ob = orr = None
    for i in range(10):
        f = top_lane_scenario(t_ms=100_000 + 100 * i, blue_s=0.45, red_s=0.55, n_minions=0)
        for u in f.units.values():
            if u.etype == "champion" and u.team == C.TEAM_RED:
                u.cooldowns = (0.0, 0.0, 0.0, 0.0) if i < 4 else (8.0, 24.0, 9.0, 160.0)
            elif u.etype == "champion" and u.team == C.TEAM_BLUE:
                u.cooldowns = (1.0, 1.0, 1.0, 1.0)
        ob = blue.build(f)
        orr = red.build(reflect_frame_in_lane(f))
    # The cast at i=4 was witnessed, so the book is populated for both agents.
    assert ob.global_vec[C.G_ENEMY_ABILITY_UNKNOWN][0] == 0.0
    assert np.array_equal(ob.global_vec, orr.global_vec)


def test_velocity_signs_survive_the_reflection():
    """A champion walking down-lane must read as down-lane for both agents."""
    blue, red = _real_builders()
    obs_b = obs_r = None
    for i in range(6):
        f = top_lane_scenario(t_ms=90_000 + 100 * i, blue_s=0.30 + 0.02 * i, red_s=0.70)
        obs_b = blue.build(f)
        obs_r = red.build(reflect_frame_in_lane(f))
    assert obs_b.self_vec[C.S_LANE_S_VEL] > 0.0  # moving towards the enemy
    assert obs_b.self_vec[C.S_LANE_S_VEL] == obs_r.self_vec[C.S_LANE_S_VEL]


# --------------------------------------------------------------------------
# Negative controls: a test that cannot fail is worthless
# --------------------------------------------------------------------------


def test_the_rotation_does_NOT_reproduce_the_reflection():
    """The regression this whole change is about.

    Feed the real red builder the 180-degree-rotated frame -- the old
    canonicalisation -- and its observation must NOT match blue's.  If this ever
    starts passing, the lane frame has silently gone back to being a rotation.
    """
    blue, red = _real_builders()
    f = top_lane_scenario()
    ob = blue.build(f)
    orr = red.build(rotate_frame(f))
    assert not np.array_equal(ob.entities, orr.entities)


def test_a_plain_x_flip_does_not_reproduce_the_reflection():
    """Guards against the observation being symmetric under everything."""
    blue, red = _real_builders()
    f = top_lane_scenario()

    flipped = f.__class__(t_ms=f.t_ms, units={})
    for uid, u in f.units.items():
        v = copy.copy(u)
        v.x = C.MIRROR_X - u.x  # x-only flip: neither our reflection nor a rotation
        if u.team == C.TEAM_BLUE:
            v.team = C.TEAM_RED
        elif u.team == C.TEAM_RED:
            v.team = C.TEAM_BLUE
        flipped.units[uid] = v

    ob = blue.build(f)
    orr = red.build(flipped)
    assert not np.array_equal(ob.entities, orr.entities)


def test_mirror_transform_is_still_a_rotation():
    """``MirrorTransform`` is retained for map reasoning; keep it honest."""
    t = MirrorTransform(C.TEAM_RED)
    a, b = (1.0, 0.0), (0.3, 0.9)
    cross_before = a[0] * b[1] - a[1] * b[0]
    ta, tb = t.vector(*a), t.vector(*b)
    cross_after = ta[0] * tb[1] - ta[1] * tb[0]
    assert cross_before == cross_after
    assert ta == (-a[0], -a[1])
