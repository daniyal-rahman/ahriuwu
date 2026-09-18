"""The observation: layout, fog, and the reflection that makes self-play a mirror."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.data.patch import CONTENT_ROOT, load_patch
from lanerl_jax.obs.builder import (
    ENTITY_DIM,
    GLOBAL_DIM,
    HP_BAR_STEPS,
    N_SLOTS,
    SELF_DIM,
    SLOT_ENEMY_CHAMP,
    SLOT_ENEMY_MINION,
    build_observation,
)
from lanerl_jax.obs.fog import visible_to, visible_to_enemy
from lanerl_jax.obs.frame import make_lane_frame, to_lane
from lanerl_jax.sim.init import TOP_OUTER_TURRET, init_lane, lane_params
from lanerl_jax.sim.combat import growth_sum
from lanerl_jax.sim.spells import BuffId, E_BUFF_SLOT, Q_BUFF_SLOT
from lanerl_jax.sim.state import Kind, Team

pytestmark = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available"
)

BLUE_NEXUS = (1131.8, 1426.3)
RED_NEXUS = (12760.9, 13026.1)


@pytest.fixture(scope="module")
def frames():
    return (make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                            TOP_OUTER_TURRET[Team.RED], BLUE_NEXUS),
            make_lane_frame(TOP_OUTER_TURRET[Team.RED],
                            TOP_OUTER_TURRET[Team.BLUE], RED_NEXUS))


def test_both_nexuses_sit_at_negative_n_in_their_own_frame(frames):
    """The handedness rule, and the reason it works for the top lane.

    The corridor hugs the map's top-left edge, so both nexuses are on the same
    side of it; forcing "own nexus at n < 0" therefore makes both agents adopt
    the **same world normal**, and `n > 0` means "towards the outer wall" for
    each. Without it the plain left-hand normal comes out with opposite signs,
    because red's lane axis is exactly antiparallel to blue's.
    """
    fb, fr = frames
    _, nb = to_lane(fb, jnp.float32(BLUE_NEXUS[0]), jnp.float32(BLUE_NEXUS[1]))
    _, nr = to_lane(fr, jnp.float32(RED_NEXUS[0]), jnp.float32(RED_NEXUS[1]))
    assert float(nb) < 0 and float(nr) < 0


def test_the_two_frames_are_a_reflection_not_a_rotation(frames):
    """``(s, n) -> (L - s, n)``: determinant -1.

    A 180-degree world rotation instead maps blue's TOP outer turret onto red's
    **BOT** one -- a different corridor with an oppositely signed normal -- so
    the two agents would not be sampling one distribution at all.
    """
    fb, fr = frames
    for px, py in ((2500.0, 12000.0), (900.0, 11000.0), (3600.0, 13500.0)):
        sb, nb = to_lane(fb, jnp.float32(px), jnp.float32(py))
        sr, nr = to_lane(fr, jnp.float32(px), jnp.float32(py))
        assert float(fb.length) - float(sb) == pytest.approx(float(sr), abs=1.0)
        assert float(nb) == pytest.approx(float(nr), abs=1.0)


def test_layout_matches_lanerl_rl_constants():
    """The layout is a contract with a policy trained elsewhere."""
    from lanerl_rl import constants as C

    assert N_SLOTS == C.N_SLOTS
    assert ENTITY_DIM == C.ENTITY_DIM
    assert SELF_DIM == C.SELF_DIM
    assert GLOBAL_DIM == C.GLOBAL_DIM
    assert (SLOT_ENEMY_CHAMP, SLOT_ENEMY_MINION) == (C.SLOT_ENEMY_CHAMP,
                                                     C.SLOT_ENEMY_MINION)
    assert C.LAST_HIT_SORT_K == 0, (
        "the HP re-sort is disabled on purpose -- re-enabling it changes the "
        "OBSERVATION, so a policy trained under one value cannot be reused")


def test_own_units_are_always_visible_and_distance_gates_the_rest():
    #        0: my champion   1: my minion   2: near enemy   3: far enemy
    n = 4
    x = jnp.asarray([0.0, 100.0, 300.0, 9000.0])
    y = jnp.zeros(n)
    kind = jnp.asarray([Kind.CHAMPION, Kind.LANE_MINION, Kind.LANE_MINION,
                        Kind.LANE_MINION], jnp.int8)
    team = jnp.asarray([Team.BLUE, Team.BLUE, Team.RED, Team.RED], jnp.int8)
    vis = visible_to(Team.BLUE, x, y, kind, team, jnp.ones(n, bool))
    assert bool(vis[1]), "my own unit is always visible to me"
    assert bool(vis[2]), "an enemy 300 units from my champion is inside 1200"
    assert not bool(vis[3]), "an enemy 9000 away is seen by nobody..."
    # ...until one of my own units walks up to it. The radius belongs to the
    # VIEWER, so a minion (1100) grants vision a champion (1200) would not.
    x2 = x.at[1].set(8500.0)          # 500 from the far enemy, inside 1100
    assert bool(visible_to(Team.BLUE, x2, y, kind, team, jnp.ones(n, bool))[3])


def test_a_turret_is_visible_even_with_no_ally_anywhere_near_it():
    """``BaseTurret.IsAffectedByFoW => false`` (``AI/BaseTurret.cs:36``): a
    turret is exempt from fog outright, not merely "seen by a big radius".

    ``GameObject.IsVisibleByTeam`` is ``!IsAffectedByFoW || _visibleByTeam[team]``
    (``GameObjects/GameObject.cs:328-330``), so for a turret the right-hand side
    never even gets evaluated -- it is visible regardless of who is nearby.
    ``Champion``/``Minion`` inherit ``AttackableUnit.IsAffectedByFoW => true``
    (``AttackableUnits/AttackableUnit.cs:117``) and get no such exemption.

    The bug this catches: folding the turret's 800-unit *viewer* radius (how
    far ITS vision reaches) into the same radius test used for whether the
    turret itself can be SEEN would hide an enemy turret the instant no ally
    stood within 800 units of it -- true for most of a 13,000-unit lane. A
    turret is a permanent, known objective on the server; it must never
    disappear from either team's observation or from ``turret_acquire``.
    """
    #  0: my champion, far away   1: enemy turret, alone on the map
    x = jnp.asarray([0.0, 9000.0])
    y = jnp.asarray([0.0, 9000.0])
    kind = jnp.asarray([Kind.CHAMPION, Kind.TURRET], jnp.int8)
    team = jnp.asarray([Team.BLUE, Team.RED], jnp.int8)
    alive = jnp.ones(2, bool)

    assert bool(visible_to(Team.BLUE, x, y, kind, team, alive)[1]), (
        "an enemy turret is visible even 9000 units from the nearest ally"
    )
    # visible_to_enemy is framed the other way round (per-target, not
    # per-viewing-team) but must agree: the turret's own team is RED, so its
    # "enemy" is BLUE, and BLUE has nothing anywhere near it.
    assert bool(visible_to_enemy(x, y, kind, team, alive)[1])
    # A dead turret is a destroyed turret, and a destroyed turret is not on
    # the map to observe -- the exemption must not defeat the `alive` gate.
    assert not bool(visible_to(Team.BLUE, x, y, kind, team,
                                alive.at[1].set(False))[1])


def test_vision_is_a_team_union_a_minion_can_see_what_its_turret_sees():
    """Vision belongs to the TEAM, not to the individual unit asking.

    A blue minion sitting far from an enemy still "sees" it (i.e. it counts as
    visible to blue, the minion's team) once a friendly TURRET's 800-unit
    bubble reaches that enemy -- the same union rule
    ``test_own_units_are_always_visible_and_distance_gates_the_rest`` exercises
    with a second minion, but the point of a SEPARATE case here is that a
    turret is a different viewer KIND with a different radius
    (``fog.VISION_RADIUS[Kind.TURRET] == 800``), so this cannot be satisfied by
    accidentally reusing one unit's own bubble twice, and it is exactly the
    scenario ``minion_acquire``'s candidate filter and ``turret_acquire``'s own
    (now vision-gated) scan both rely on being a TEAM property.

    The bug this catches: computing "is X visible to blue" from a single
    blue unit's own kind/position (per-unit) rather than as a union over every
    living blue unit would make an enemy standing next to a friendly turret,
    but far from every other blue unit, wrongly stay invisible to blue's
    targeting and observation.
    """
    #  0: enemy minion (the candidate being asked about)
    #  1: blue turret, 700 from the candidate (inside the turret's 800)
    #  2: blue minion, far from the candidate on its own (outside its own 1100)
    x = jnp.asarray([5700.0, 5000.0, 0.0])
    y = jnp.asarray([5000.0, 5000.0, 0.0])
    kind = jnp.asarray([Kind.LANE_MINION, Kind.TURRET, Kind.LANE_MINION], jnp.int8)
    team = jnp.asarray([Team.RED, Team.BLUE, Team.BLUE], jnp.int8)
    alive = jnp.ones(3, bool)

    vis_blue = visible_to(Team.BLUE, x, y, kind, team, alive)
    assert bool(vis_blue[0]), (
        "the enemy minion is inside the blue turret's 800-unit bubble, so "
        "blue (and therefore blue's own minion, via the team union) sees it, "
        "even though blue's minion is nowhere near it"
    )
    # Take the turret out of the picture entirely: no more team union, no
    # more blue vision on the candidate from 5700 units away.
    vis_without_turret = visible_to(
        Team.BLUE, x, y, kind, team, alive.at[1].set(False))
    assert not bool(vis_without_turret[0])


def test_a_fogged_enemy_champion_is_absent_not_at_the_origin(frames):
    """`obs.py`'s rule 3, and the reason it exists.

    A fogged entity must not occupy a slot with ``ds = dn = 0, valid = 1`` --
    that tells the policy "the enemy is standing on top of me", the single worst
    hallucination available. At spawn the two champions are a map apart, so the
    enemy slot must simply be empty.
    """
    fb, _ = frames
    s = init_lane(load_patch())
    ob = build_observation(s, 0, fb, params=lane_params())
    assert int(ob.slot_unit[0]) == -1
    assert bool(ob.entity_pad_mask[0])
    assert float(ob.entities[0].sum()) == 0.0


def test_empty_slots_are_zero_and_masked(frames):
    fb, _ = frames
    ob = build_observation(init_lane(load_patch()), 0, fb, params=lane_params())
    empty = np.asarray(ob.slot_unit) < 0
    assert np.asarray(ob.entity_pad_mask)[empty].all()
    assert float(np.abs(np.asarray(ob.entities)[empty]).sum()) == 0.0


def test_hp_frac_is_quantised_to_bar_resolution(frames):
    """A player reads a health bar, not a float."""
    fb, _ = frames
    p = load_patch()
    s = init_lane(p)
    s = s.replace(hp=s.hp.at[1].set(float(s.max_hp[1]) * 0.5137))
    # put the enemy champion in vision
    s = s.replace(x=s.x.at[1].set(float(s.x[0]) + 200.0),
                  y=s.y.at[1].set(float(s.y[0])))
    ob = build_observation(s, 0, fb, params=lane_params())
    assert int(ob.slot_unit[0]) == 1
    v = float(ob.entities[0, 3])
    assert v == pytest.approx(round(0.5137 * HP_BAR_STEPS) / HP_BAR_STEPS, abs=1e-5)


def test_self_stats_and_cooldowns_come_from_live_state_and_profiles(frames):
    """The four former zero cooldown/stat fields are policy inputs, not pads."""
    fb, _ = frames
    p = load_patch()
    params = lane_params(p)
    s = init_lane(p).replace(
        level=jnp.asarray([9] + [1] * 65, jnp.int8),
        spell_level=jnp.asarray([[2, 1, 5, 2]] + [[0] * 4] * 65, jnp.int8),
        spell_cooldown=jnp.asarray([[4.0, 12.0, 4.5, 60.0]] + [[0.0] * 4] * 65),
    )
    ob = build_observation(s, 0, fb, params=params)
    # Q (8 s), W rank 1 (24 s), E rank 5 (9 s), R rank 2 (120 s).
    np.testing.assert_allclose(np.asarray(ob.self_vec[6:10]), 0.5, atol=1e-6)

    model = int(s.model[0])
    growth = float(growth_sum(9))
    expected_ad = (float(params["attack_damage"][model])
                   + float(params["ad_per_level"][model]) * growth) / 200.0
    expected_armor = (float(params["armor"][model])
                      + float(params["armor_per_level"][model]) * growth) / 200.0
    expected_mr = (float(params["magic_resist"][model])
                   + float(params["mr_per_level"][model]) * growth) / 200.0
    assert float(ob.self_vec[10]) == pytest.approx(expected_ad, abs=1e-6)
    assert float(ob.self_vec[11]) == 0.0, "AP has no state/profile source"
    assert float(ob.self_vec[12]) == pytest.approx(expected_armor, abs=1e-6)
    assert float(ob.self_vec[13]) == pytest.approx(expected_mr, abs=1e-6)


def test_q_and_e_windows_report_unavailable_even_before_cooldown_starts(frames):
    fb, _ = frames
    p = load_patch()
    s = init_lane(p).replace(
        spell_level=jnp.asarray([[1, 0, 1, 0]] + [[0] * 4] * 65, jnp.int8),
        buff_id=init_lane(p).buff_id.at[0, Q_BUFF_SLOT].set(BuffId.GAREN_Q)
        .at[0, E_BUFF_SLOT].set(BuffId.GAREN_E),
    )
    ob = build_observation(s, 0, fb, params=lane_params(p))
    assert float(ob.self_vec[6]) == 1.0
    assert float(ob.self_vec[8]) == 1.0


def test_global_cast_memory_uses_rank_one_bases_and_never_seen_sentinel(frames):
    """The actor gets witnessed event age, never an enemy cooldown/rank."""
    fb, _ = frames
    p = load_patch()
    s = init_lane(p).replace(observed_enemy_cast_ms=jnp.asarray([
        [0.0, 12_000.0, 20_000.0, 160_000.0],
        [-1.0, -1.0, -1.0, -1.0],
    ]))
    blue = build_observation(s, 0, fb, params=lane_params(p))
    # Q/W/E/R rank-one bases are 8/24/13/160 seconds.  E is clipped and a
    # never-witnessed spell saturates rather than exposing a special actor bit.
    np.testing.assert_allclose(np.asarray(blue.global_vec[2:]),
                               [0.0, 0.5, 1.0, 1.0], atol=1e-6)

    # Memory belongs to the observing champion, not the currently visible
    # enemy. Red has never witnessed a blue cast even though raw blue state is
    # present in this simulator state.
    _, fr = frames
    red = build_observation(s, 1, fr, params=lane_params(p))
    np.testing.assert_allclose(np.asarray(red.global_vec[2:]), 1.0, atol=1e-6)


def test_it_jits_and_vmaps(frames):
    fb, _ = frames
    s = init_lane(load_patch())
    params = lane_params()
    f = jax.jit(lambda st: build_observation(st, 0, fb, params=params))
    ob = f(s)
    assert ob.entities.shape == (N_SLOTS, ENTITY_DIM)
    batched = jax.vmap(f)(jax.tree.map(lambda a: jnp.broadcast_to(a, (8,) + a.shape), s))
    assert batched.entities.shape == (8, N_SLOTS, ENTITY_DIM)
