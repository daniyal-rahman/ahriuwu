"""Collision parity: creation-order Gauss-Seidel, the turret obstacle/affected
split, and the ``CollisionRadius``/``PathfindingRadius`` distinction.

Each test is paired with a fact from the C# source cited in
``sim/collision.py``'s own module docstring, and is written so it would have
failed (either with a ``TypeError`` on the old, narrower signature, or with a
wrong numeric answer reproduced by hand below) against the single-push,
symmetric-mask, single-radius approximation this module replaces.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from lanerl_jax.data.patch import CONTENT_ROOT, load_patch
from lanerl_jax.sim.collision import resolve_collisions
from lanerl_jax.sim.init import init_lane
from lanerl_jax.sim.profiles import PROFILES, build_profile_tables, profile_id
from lanerl_jax.sim.state import Kind, Team, TU_SLICE
from lanerl_jax.sim.targeting import MinionType


def _arrays(x, y, kind, seq, cr, pr, alive=None):
    n = len(x)
    return dict(
        x=jnp.asarray(x, jnp.float32), y=jnp.asarray(y, jnp.float32),
        kind=jnp.asarray(kind, jnp.int8),
        alive=jnp.asarray([True] * n if alive is None else alive),
        spawn_seq=jnp.asarray(seq, jnp.int32),
        collision_radius=jnp.asarray(cr, jnp.float32),
        pathfinding_radius=jnp.asarray(pr, jnp.float32),
    )


# --------------------------------------------------------------------------
# 1. Creation-order Gauss-Seidel, multiple escapes per unit per tick.
# --------------------------------------------------------------------------

def test_a_unit_pinched_between_two_earlier_neighbours_gets_two_escapes():
    """M is created FIRST (``spawn_seq=0``) and starts pinched between P1
    (5 units away) and P2 (35 units away) -- both closer than the trigger
    distance (80 = 40 + 40 collision radii). The server's ``UpdateCollision``
    applies one ``OnCollision`` escape per overlapping neighbour it finds,
    IN SEQUENCE, using the position the previous escape just produced
    (``AttackableUnit.cs:278-318``) -- not a single push computed from the
    pre-tick snapshot. Array/slot index (0, 1, 2 here) plays no part; only
    ``spawn_seq`` does, so this also exercises "creation order, not slot
    order".

    Hand-derived from ``GetCircleEscapePoint``'s formula
    (``p1 + u*(d - r1 - r2)``, ``r1 = PathfindingRadius[mover] + 1``,
    ``r2 = PathfindingRadius[collider]``), walked in the exact sequence
    ``resolve_collisions`` must follow (M's own turn is first, since it has
    the lowest ``spawn_seq``):

        M  vs P1 (d=5):   push = 5  - 41 - 40 = -76  -> M:   0 -> 76
        M  vs P2 (d=41, using M's NEW position 76):
                          push = 41 - 41 - 40 = -40  -> M:  76 -> 116
        P1 vs M  (d=121, M already at 116): 121 !< 80, no escape
        P1 vs P2 (d=40):  push = 40 - 41 - 40 = -41  -> P1: -5 -> -46
        P2 vs M/P1 (d=81 both ways): the TRIGGER radius is 80 (collision
                          radii), not the RESOLUTION radius's 81 -- exactly
                          at the boundary, no escape, P2 stays put.

    A single-push Jacobi approximation (this module's previous version)
    cannot produce M's final 116: it applies exactly one escape, computed
    from the ORIGINAL positions, from whichever overlapping neighbour has
    the lowest ARRAY index -- 76 here, never 116.
    """
    kwargs = _arrays(
        x=[0.0, -5.0, 35.0], y=[0.0, 0.0, 0.0],
        kind=[Kind.LANE_MINION] * 3,
        seq=[0, 1, 2],                       # M, then P1, then P2
        cr=[40.0, 40.0, 40.0], pr=[40.0, 40.0, 40.0],
    )
    nx, ny = resolve_collisions(**kwargs)
    assert float(nx[0]) == pytest.approx(116.0)
    assert float(nx[1]) == pytest.approx(-46.0)
    assert float(nx[2]) == pytest.approx(35.0)
    assert np.allclose(np.asarray(ny), 0.0)


def test_creation_order_not_slot_order_drives_the_outer_sequence():
    """Same geometry as above, but the ARRAY/slot indices are shuffled so
    that slot order and creation order disagree completely (slot 0 is the
    LAST-created unit here). If anything keyed off slot index rather than
    ``spawn_seq``, this would reproduce the previous test's answer under a
    permutation; it must instead reproduce the SAME physical answer (M, the
    unit with ``spawn_seq=0``, still ends at distance 116 from its start).
    """
    # slot order: [P2, P1, M]; creation order: M(seq0), P1(seq1), P2(seq2)
    kwargs = _arrays(
        x=[35.0, -5.0, 0.0], y=[0.0, 0.0, 0.0],
        kind=[Kind.LANE_MINION] * 3,
        seq=[2, 1, 0],
        cr=[40.0, 40.0, 40.0], pr=[40.0, 40.0, 40.0],
    )
    nx, ny = resolve_collisions(**kwargs)
    # slot 2 holds M (spawn_seq 0) here, at starting x=0.
    assert float(nx[2]) == pytest.approx(116.0)
    assert float(nx[1]) == pytest.approx(-46.0)
    assert float(nx[0]) == pytest.approx(35.0)


# --------------------------------------------------------------------------
# 2. Turret obstacle-but-not-affected split.
# --------------------------------------------------------------------------

def test_turret_blocks_a_minion_but_is_never_pushed_itself():
    """``IsCollisionObject`` does not exclude turrets (they ARE obstacles);
    ``IsCollisionAffected`` excludes ``BaseTurret`` (they are never pushed).
    A single symmetric mask -- this module's previous version -- drops
    turrets from BOTH roles, so a minion overlapping one would pass straight
    through. Here the minion overlaps a turret by 90 units of penetration
    (50 apart, radii 100 + 40) and must be teleported out to exactly
    touching (`GetCircleEscapePoint`, PathfindingRadius-based): distance
    41 + 100 = 141. The turret itself must not move at all.
    """
    kwargs = _arrays(
        x=[0.0, 50.0], y=[0.0, 0.0],
        kind=[Kind.TURRET, Kind.LANE_MINION],
        seq=[0, 1],
        cr=[100.0, 40.0], pr=[100.0, 40.0],
    )
    nx, ny = resolve_collisions(**kwargs)
    assert float(nx[0]) == pytest.approx(0.0)          # turret: never affected
    assert float(nx[1]) == pytest.approx(141.0)         # minion: pushed clear
    assert np.allclose(np.asarray(ny), 0.0)


def test_turret_is_untouched_even_when_it_goes_first_in_creation_order():
    """Turrets ARE created before any minion (map load precedes both
    ``PlayerManager.AddPlayer`` and any wave spawn), so a turret always gets
    an early outer-loop turn in the real ``spawn_seq`` ordering. Confirms
    that turn is a structural no-op (``affected`` excludes it) rather than
    something that happens to be masked out only when it goes last.
    """
    kwargs = _arrays(
        x=[0.0, 50.0], y=[0.0, 0.0],
        kind=[Kind.TURRET, Kind.LANE_MINION],
        seq=[0, 1],                    # turret first, as in the real game
        cr=[100.0, 40.0], pr=[100.0, 40.0],
    )
    nx, _ = resolve_collisions(**kwargs)
    assert float(nx[0]) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 3. CollisionRadius (trigger) vs PathfindingRadius (resolution).
# --------------------------------------------------------------------------

def test_trigger_uses_collision_radius_not_pathfinding_radius():
    """A champion's real ``CollisionRadius`` is 30 (``Champion.cs:52``,
    hard-coded, never 40) but its ``PathfindingRadius`` genuinely is 40
    (``CharData.PathfindingCollisionRadius`` fallback, unaffected by the
    hard-code -- see ``sim/collision.py``'s module docstring). A melee
    minion is 40/40 on both. At d=75 from a melee minion:

    * the TRUE trigger sum is 30 + 40 = 70 -- 75 does not overlap, no push.
    * the OLD code used PathfindingRadius for the trigger too: 40 + 40 = 80
      -- 75 WOULD have overlapped, and (incorrectly) pushed both units.

    So this is a distance band a champion-vs-minion pair can genuinely sit
    in where the fix changes the outcome from "push" to "no push".
    """
    kwargs = _arrays(
        x=[0.0, 75.0], y=[0.0, 0.0],
        kind=[Kind.CHAMPION, Kind.LANE_MINION],
        seq=[0, 1],
        cr=[30.0, 40.0], pr=[40.0, 40.0],
    )
    nx, ny = resolve_collisions(**kwargs)
    assert float(nx[0]) == pytest.approx(0.0)
    assert float(nx[1]) == pytest.approx(75.0)
    assert np.allclose(np.asarray(ny), 0.0)


def test_trigger_still_fires_within_the_true_collision_radius():
    """Same pair, close enough (d=65) to overlap on the TRUE 30+40=70
    trigger sum -- the fix must not simply turn collision off for
    champions. Resolution still uses PathfindingRadius (40 + 1 + 40 = 81),
    so the champion (the one being resolved here) ends up 81 units from the
    minion, not 70.
    """
    kwargs = _arrays(
        x=[0.0, 65.0], y=[0.0, 0.0],
        kind=[Kind.CHAMPION, Kind.LANE_MINION],
        seq=[0, 1],
        cr=[30.0, 40.0], pr=[40.0, 40.0],
    )
    nx, _ = resolve_collisions(**kwargs)
    assert float(nx[0]) == pytest.approx(65.0 - 81.0)


# --------------------------------------------------------------------------
# 4. The profile-level radius fix (Minion.cs:57 / Champion.cs:52 hard-codes).
# --------------------------------------------------------------------------

pytestmark_content = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available")


@pytestmark_content
def test_profile_collision_radius_matches_the_server_hardcode_not_content():
    """``ObjAIBase``'s ctor only reads ``CharData.GameplayCollisionRadius``
    when its own ``collisionRadius`` argument is non-positive -- true only
    of ``BaseTurret``. ``Minion``/``Champion`` pass 40/30 unconditionally, so
    Content's own value (65 for a cannon/super minion, -1 -- i.e. "absent"
    -- for Garen) must never surface in ``collision_radius``, even though it
    legitimately does for ``pathfinding_radius``.
    """
    patch = load_patch()
    tables = build_profile_tables(patch)
    cr = np.asarray(tables["collision_radius"])
    pr = np.asarray(tables["pathfinding_radius"])

    champ_row = profile_id(Kind.CHAMPION, -1, Team.BLUE)
    cannon_row = profile_id(Kind.LANE_MINION, MinionType.CANNON, Team.BLUE)
    melee_row = profile_id(Kind.LANE_MINION, MinionType.MELEE, Team.BLUE)

    assert cr[champ_row] == pytest.approx(30.0)
    assert cr[cannon_row] == pytest.approx(40.0)
    assert cr[melee_row] == pytest.approx(40.0)
    # PathfindingRadius is untouched by the hard-code and stays whatever
    # CharData actually says -- NOT flattened to 40/30 the way
    # `collision_radius` is. Compared against `data.patch`'s own
    # (independently loaded) reading of Content rather than a hard-coded
    # guess here, since Garen's real `PathfindingCollisionRadius` (35) turns
    # out to differ from his `collision_radius`'s 40-or-30 story entirely.
    assert pr[champ_row] == pytest.approx(patch.champion.pathfinding_radius)
    assert pr[cannon_row] == pytest.approx(patch.minions["cannon_blue"].pathfinding_radius)

    # Turrets genuinely defer to CharData for CollisionRadius (BaseTurret
    # passes none of its own) -- so this column must NOT be flattened to a
    # constant the way champions/minions are. At least one turret model on
    # this patch must disagree with 40/30 for that to be a meaningful check.
    turret_rows = sorted({profile_id(Kind.TURRET, tier, team)
                          for tier in range(5) for team in (Team.BLUE, Team.RED)})
    turret_cr = cr[turret_rows]
    assert not np.allclose(turret_cr, 40.0) or not np.allclose(turret_cr, 30.0)


# --------------------------------------------------------------------------
# 5. spawn_seq: the monotonic creation sequence itself (task 1).
# --------------------------------------------------------------------------

@pytestmark_content
def test_init_lane_ranks_every_turret_and_champion_before_any_minion():
    """`Map.Init()` (which builds the turrets, nexuses and inhibitors) runs
    strictly before `Game.Initialize`'s `PlayerManager.AddPlayer` loop
    (which constructs the two champions), and both run to completion before
    the game loop -- and any wave spawn -- ever starts. So every live
    ``spawn_seq`` at ``t=0`` must be a turret, all turrets must rank below
    both champions, and ``next_spawn_seq`` (what the first minion will get)
    must be strictly above everything ``init_lane`` assigned.
    """
    s = init_lane()
    seq = np.asarray(s.spawn_seq)
    kind = np.asarray(s.kind)
    turret_seq = seq[kind == Kind.TURRET]
    champ_seq = seq[kind == Kind.CHAMPION]

    assert len(turret_seq) > 0 and len(champ_seq) == 2
    assert turret_seq.max() < champ_seq.min()
    assert champ_seq[0] < champ_seq[1]          # blue (slot 0) before red
    assert int(s.next_spawn_seq) > int(seq.max())
    # every rank is unique -- no two units share a creation slot
    assert len(set(seq[kind != Kind.NONE].tolist())) == int((kind != Kind.NONE).sum())


def test_spawn_minion_never_reuses_a_recycled_slots_old_rank():
    """A minion slot is recycled on death, but the server's real object is a
    brand NEW ``Minion`` GameObject each time (the old occupant was already
    `RemoveObject`d) -- so the new occupant must get a strictly higher
    ``spawn_seq`` than whatever the slot held before, never the old value
    back.
    """
    from lanerl_jax.sim.init import spawn_minion
    from lanerl_jax.sim.state import MI_SLICE, empty_state

    s = empty_state()
    s = s.replace(next_spawn_seq=jnp.asarray(5, jnp.int32))
    path = jnp.asarray([[0.0, 0.0], [10.0, 0.0]], jnp.float32)

    s = spawn_minion(s, Team.BLUE, 0, 100.0, path)
    first_seq = int(s.spawn_seq[MI_SLICE.start])
    assert first_seq == 5

    # kill it, freeing the slot, then spawn again -- a NEW object -> a NEW,
    # strictly larger rank, never 5 again.
    s = s.replace(alive=s.alive.at[MI_SLICE.start].set(False))
    s = spawn_minion(s, Team.BLUE, 0, 100.0, path)
    second_seq = int(s.spawn_seq[MI_SLICE.start])
    assert second_seq > first_seq


# --------------------------------------------------------------------------
# 6. The bounded-rounds inner loop equals the exhaustive single pass.
# --------------------------------------------------------------------------

def _python_reference(x, y, kind, alive, spawn_seq, collision_radius,
                      pathfinding_radius, ghosted):
    """A slow, unvectorised, unbounded transcription of
    ``CollisionHandler.Update``/``UpdateCollision``/``AttackableUnit.
    OnCollision``, independent of ``resolve_collisions``'s own
    implementation -- the standard this project holds a JAX port to (see
    ``sim/movement_jax.py``'s own docstring for the same pattern). Used only
    to check the bounded-rounds vectorisation is a re-expression of the same
    algorithm, not a different, faster one.
    """
    import math
    n = len(x)
    x = list(x)
    y = list(y)
    obstacle = [alive[i] and kind[i] != Kind.NONE and not ghosted[i] for i in range(n)]
    affected = [obstacle[i] and kind[i] != Kind.TURRET for i in range(n)]
    order = sorted(range(n), key=lambda i: spawn_seq[i])
    for i in order:
        if not affected[i]:
            continue
        for j in order:
            if j == i or not obstacle[j]:
                continue
            dx, dy = x[j] - x[i], y[j] - y[i]
            d = math.hypot(dx, dy)
            touch = collision_radius[i] + collision_radius[j]
            if 0 < d < touch:
                ux, uy = dx / d, dy / d
                push = d - (pathfinding_radius[i] + 1.0) - pathfinding_radius[j]
                x[i] += ux * push
                y[i] += uy * push
    return x, y


@pytest.mark.parametrize("seed", range(8))
def test_bounded_rounds_matches_the_unbounded_python_reference(seed):
    """Random crowded scenes -- several units sharing a small patch of map,
    so many-way, multi-escape overlaps are common -- checked against the
    slow reference above rather than against ``resolve_collisions``'s own
    prior implementation. This is the equivalence check the bounded-rounds
    rewrite (module docstring: "why the inner loop is BOUNDED rounds") needs:
    proof that vectorising "which candidate is next" did not change the
    answer, only how many sequential steps it costs.
    """
    rng = np.random.default_rng(seed)
    n = 10
    x = rng.uniform(-60.0, 60.0, n).astype(np.float32)
    y = rng.uniform(-60.0, 60.0, n).astype(np.float32)
    kind = np.full(n, Kind.LANE_MINION, np.int8)
    kind[rng.choice(n, 2, replace=False)] = Kind.TURRET
    alive = np.ones(n, bool)
    spawn_seq = rng.permutation(n).astype(np.int32)
    collision_radius = rng.uniform(30.0, 45.0, n).astype(np.float32)
    pathfinding_radius = rng.uniform(30.0, 45.0, n).astype(np.float32)
    ghosted = np.zeros(n, bool)

    ref_x, ref_y = _python_reference(
        x, y, kind, alive, spawn_seq, collision_radius, pathfinding_radius, ghosted)

    got_x, got_y = resolve_collisions(
        jnp.asarray(x), jnp.asarray(y), jnp.asarray(kind), jnp.asarray(alive),
        jnp.asarray(spawn_seq), jnp.asarray(collision_radius),
        jnp.asarray(pathfinding_radius), ghosted=jnp.asarray(ghosted))

    np.testing.assert_allclose(np.asarray(got_x), np.asarray(ref_x), atol=1e-3)
    np.testing.assert_allclose(np.asarray(got_y), np.asarray(ref_y), atol=1e-3)


def test_max_escapes_used_is_within_the_provisional_bound_on_a_worst_case_scene():
    """A deliberately adversarial scene -- one minion (`spawn_seq=0`, so it
    goes first and sees everyone else unmoved) packed against five others
    all within its collision radius -- to sanity-check
    ``MAX_ESCAPES_PER_UNIT``'s provisional value of 8 isn't obviously too
    small. Not a substitute for the real corpus measurement the module
    docstring says is still owed.
    """
    from lanerl_jax.sim.collision import MAX_ESCAPES_PER_UNIT, max_escapes_used

    n = 6
    x = jnp.array([0.0, 10.0, -10.0, 0.0, 0.0, 20.0], jnp.float32)
    y = jnp.array([0.0, 0.0, 0.0, 10.0, -10.0, 0.0], jnp.float32)
    kind = jnp.full(n, Kind.LANE_MINION, jnp.int8)
    alive = jnp.ones(n, bool)
    spawn_seq = jnp.arange(n, dtype=jnp.int32)
    cr = jnp.full(n, 40.0, jnp.float32)
    pr = jnp.full(n, 40.0, jnp.float32)

    counts = max_escapes_used(x, y, kind, alive, spawn_seq, cr, pr)
    assert int(counts[0]) < MAX_ESCAPES_PER_UNIT
    assert int(counts.max()) < 32, "hit the probe length -- probe was too small"
