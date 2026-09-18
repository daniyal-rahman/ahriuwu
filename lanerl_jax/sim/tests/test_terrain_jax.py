"""Focused device-side tests for terrain collision exits.

These exercise the two call sites in ``AttackableUnit.OnCollision`` as well as
the lower-level grid query.  The host ``NavGrid`` is the independent source
port/oracle; the production functions are all called under ``jax.jit``.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

from lanerl_jax.data.navgrid import NavGrid, NavigationGridCellFlags
from lanerl_jax.sim.collision import resolve_collisions
from lanerl_jax.sim.state import Kind
from lanerl_jax.sim.terrain_jax import (
    TerrainGrid,
    cast_circle_blocked,
    closest_terrain_exit,
    exit_blocked_escape,
    exit_terrain_collision,
    is_walkable,
    map1_terrain,
    repair_collision_terrain_batch,
)


def _tiny_terrain(blocked=()):
    """A 12x12, unit-cell synthetic map plus the host reference equivalent."""
    flags = np.zeros((12, 12), np.uint16)
    for y, x in blocked:
        flags[y, x] = NavigationGridCellFlags.NOT_PASSABLE
    host = NavGrid(flags=flags, cell_size=1.0,
                   min_grid=(0.0, 0.0, 0.0), max_grid=(12.0, 0.0, 12.0))
    return TerrainGrid(jnp.asarray(host.walkable_mask()), 1.0, 0.0, 0.0), host


def test_device_walkability_matches_host_navgrid_for_point_and_radius_queries():
    terrain = map1_terrain()
    host = NavGrid.load()
    # Includes ordinary lane positions, a blocked cell, and the map boundary
    # where GetCell's historical x==width behaviour is relevant.
    points = np.asarray([
        (0.0, 0.0), (1000.0, 1000.0), (5000.0, 5000.0),
        (12000.0, 12000.0), (14300.0, 14500.0),
    ], np.float32)
    radii = np.asarray([0.0, 35.0, 89.4], np.float32)

    for radius in radii:
        fn = jax.jit(jax.vmap(
            lambda xy: is_walkable(xy[0], xy[1], jnp.asarray(radius), terrain)))
        got = np.asarray(fn(jnp.asarray(points)))
        want = np.asarray([
            host.is_walkable_world(float(x), float(y), float(radius))
            for x, y in points
        ])
        np.testing.assert_array_equal(got, want)


def test_device_cast_circle_matches_host_on_ties_blockers_and_map1_routes():
    """The LOS primitive used by server ``SmoothPath`` keeps its corner tie.

    The first three cases exercise a synthetic blocked diagonal, a clear row,
    and a fractional endpoint.  The Map1 cases are the two sides of the
    local-click route fixture: one direct smooth segment and one route corner.
    """
    tiny, host_tiny = _tiny_terrain(blocked=[(5, 5)])
    cases = [
        (host_tiny, tiny, (1.5, 1.5, 9.5, 9.5), 0.1),
        (host_tiny, tiny, (1.5, 2.5, 9.5, 2.5), 0.1),
        (host_tiny, tiny, (1.1, 4.5, 9.2, 4.5), 0.1),
    ]
    host_map = NavGrid.load()
    terrain_map = map1_terrain()
    # Coordinates are NAV CELL coordinates, as CastCircle receives after
    # TranslateToNavGrid; radius remains world units.
    cases.extend([
        (host_map, terrain_map, (7.5, 7.5, 18.5, 46.5), 35.0),
        (host_map, terrain_map, (285.5, 285.5, 273.5, 281.5), 35.0),
    ])
    for host, terrain, xy, radius in cases:
        fn = jax.jit(lambda a, b, c, d: cast_circle_blocked(
            a, b, c, d, jnp.float32(radius), terrain))
        blocked, exhausted = fn(*map(jnp.float32, xy))
        assert not bool(exhausted)
        assert bool(blocked) == bool(host.cast_circle(*xy, radius))


def test_cast_circle_fails_closed_when_its_declared_local_bound_is_exceeded():
    terrain, _ = _tiny_terrain()
    blocked, exhausted = jax.jit(lambda: cast_circle_blocked(
        jnp.float32(1.5), jnp.float32(1.5),
        jnp.float32(10.5), jnp.float32(10.5), jnp.float32(0.1), terrain,
        max_line_steps=2, span_cells=4))()
    assert bool(exhausted)
    assert bool(blocked), "an uninspected CastCircle segment must never smooth"


def test_drifting_spiral_is_jittable_and_matches_host_reference():
    terrain, host = _tiny_terrain(blocked=[(0, 0)])
    start = (np.float32(0.1), np.float32(0.1))
    want = host.closest_terrain_exit(*map(float, start), radius=0.0)
    got_x, got_y, exhausted = jax.jit(
        lambda x, y: closest_terrain_exit(x, y, jnp.float32(0.0), terrain)
    )(*map(jnp.asarray, start))
    assert not bool(exhausted)
    np.testing.assert_allclose(
        np.asarray([got_x, got_y]), want, rtol=0.0, atol=2e-6)


def test_terrain_collision_uses_point_trigger_not_pathfinding_radius_trigger():
    terrain, _ = _tiny_terrain(blocked=[(0, 0)])
    # (1.5, .5) is itself walkable, but a radius-1 circle touches [0, 0].
    # CollisionHandler calls IsWalkable(position) with radius zero before
    # OnCollision chooses PathfindingRadius+1 for the actual exit search.
    x, y, exhausted = jax.jit(
        lambda: exit_terrain_collision(
            jnp.float32(1.5), jnp.float32(0.5), jnp.float32(1.0), terrain)
    )()
    np.testing.assert_allclose(np.asarray([x, y]), [1.5, 0.5])
    assert not bool(exhausted)


def test_object_escape_fallback_uses_radius_guard_then_radius_plus_one_exit():
    terrain, host = _tiny_terrain(blocked=[(3, 2)])
    # The raw circle escape is in the blocked [x=2, y=3] cell.  This is the
    # nested `if (!IsWalkable(exit, radius)) GetClosestTerrainExit(exit,
    # radius+1)` branch, not CollisionHandler's initial point test.
    raw = (np.float32(2.5), np.float32(3.5))
    radius = np.float32(0.1)
    want = host.closest_terrain_exit(*map(float, raw), radius=float(radius + 1.0))
    got_x, got_y, exhausted = jax.jit(
        lambda x, y: exit_blocked_escape(x, y, jnp.float32(radius), terrain)
    )(*map(jnp.asarray, raw))
    assert not bool(exhausted)
    np.testing.assert_allclose(
        np.asarray([got_x, got_y]), want, rtol=0.0, atol=2e-6)


def test_resolve_collisions_applies_terrain_before_object_sweep_and_after_push():
    terrain, _ = _tiny_terrain(blocked=[(3, 3), (3, 2)])
    # Unit 0 starts inside [3,3], proving the pre-sweep terrain branch is
    # threaded through the real collision resolver. Unit 1 then supplies an
    # object candidate; the exact final geometry is less useful than the two
    # source invariants below: neither final position remains blocked at the
    # radius used by its most recent terrain exit, and the compiled call works.
    x = jnp.asarray([3.5, 4.0], jnp.float32)
    y = jnp.asarray([3.5, 3.5], jnp.float32)
    kind = jnp.asarray([Kind.LANE_MINION, Kind.LANE_MINION], jnp.int8)
    alive = jnp.asarray([True, True])
    seq = jnp.asarray([0, 1], jnp.int32)
    cr = jnp.asarray([0.4, 0.4], jnp.float32)
    pr = jnp.asarray([0.1, 0.1], jnp.float32)
    fn = jax.jit(lambda: resolve_collisions(
        x, y, kind, alive, seq, cr, pr, terrain=terrain))
    got_x, got_y = fn()
    assert bool(is_walkable(got_x[0], got_y[0], jnp.float32(1.1), terrain))
    # It landed somewhere else than the original terrain cell, and the actual
    # point query must be walkable too.
    assert bool(is_walkable(got_x[0], got_y[0], jnp.float32(0.0), terrain))


def test_batched_deferred_repair_matches_each_individual_terrain_query():
    terrain, _ = _tiny_terrain(blocked=[(0, 0), (3, 2)])
    x = jnp.asarray([0.1, 2.5, 7.5], jnp.float32)
    y = jnp.asarray([0.1, 3.5, 7.5], jnp.float32)
    radius = jnp.asarray([0.1, 0.1, 0.1], jnp.float32)
    moved = jnp.asarray([False, True, False])

    bx, by, exhausted = jax.jit(
        lambda: repair_collision_terrain_batch(
            x, y, radius, moved, terrain))()

    expected = []
    for i in range(3):
        if bool(moved[i]):
            px, py, _ = exit_blocked_escape(x[i], y[i], radius[i], terrain)
        else:
            px, py, _ = exit_terrain_collision(x[i], y[i], radius[i], terrain)
        expected.append((float(px), float(py)))
    np.testing.assert_allclose(
        np.stack([np.asarray(bx), np.asarray(by)], axis=1),
        np.asarray(expected), rtol=0.0, atol=2e-6)
    assert not np.any(np.asarray(exhausted))
