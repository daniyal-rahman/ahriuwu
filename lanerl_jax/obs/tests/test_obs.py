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
from lanerl_jax.obs.fog import visible_to
from lanerl_jax.obs.frame import make_lane_frame, to_lane
from lanerl_jax.sim.init import TOP_OUTER_TURRET, init_lane
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


def test_a_fogged_enemy_champion_is_absent_not_at_the_origin(frames):
    """`obs.py`'s rule 3, and the reason it exists.

    A fogged entity must not occupy a slot with ``ds = dn = 0, valid = 1`` --
    that tells the policy "the enemy is standing on top of me", the single worst
    hallucination available. At spawn the two champions are a map apart, so the
    enemy slot must simply be empty.
    """
    fb, _ = frames
    s = init_lane(load_patch())
    ob = build_observation(s, 0, fb)
    assert int(ob.slot_unit[0]) == -1
    assert bool(ob.entity_pad_mask[0])
    assert float(ob.entities[0].sum()) == 0.0


def test_empty_slots_are_zero_and_masked(frames):
    fb, _ = frames
    ob = build_observation(init_lane(load_patch()), 0, fb)
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
    ob = build_observation(s, 0, fb)
    assert int(ob.slot_unit[0]) == 1
    v = float(ob.entities[0, 3])
    assert v == pytest.approx(round(0.5137 * HP_BAR_STEPS) / HP_BAR_STEPS, abs=1e-5)


def test_it_jits_and_vmaps(frames):
    fb, _ = frames
    s = init_lane(load_patch())
    f = jax.jit(lambda st: build_observation(st, 0, fb))
    ob = f(s)
    assert ob.entities.shape == (N_SLOTS, ENTITY_DIM)
    batched = jax.vmap(f)(jax.tree.map(lambda a: jnp.broadcast_to(a, (8,) + a.shape), s))
    assert batched.entities.shape == (8, N_SLOTS, ENTITY_DIM)
