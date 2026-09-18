import jax
import jax.numpy as jnp
import numpy as np

import pytest

from lanerl_jax.data.route_artifact import DIRECTION_OFFSETS, NO_ROUTE, STAY
from lanerl_jax.sim.local_pathing import (
    LocalRouteStatus,
    LocalRouteTable,
    build_local_waypoints,
    lookup_local_hop,
    smooth_cell_path,
)
from lanerl_jax.sim.orders import OrderKind, Orders, apply_orders
from lanerl_jax.sim.state import Kind, MAX_WAYPOINTS, empty_state
from lanerl_jax.sim.terrain_jax import TerrainGrid, map1_terrain


def _table(width=7, height=7, radius=4, *, covered=None):
    """A tiny deterministic x-then-y local router for contract tests."""
    if covered is None:
        covered = np.ones(width * height, bool)
    cells = np.flatnonzero(covered)
    cell_to_row = np.full(width * height, -1, np.int32)
    cell_to_row[cells] = np.arange(len(cells), dtype=np.int32)
    side = 2 * radius + 1
    hops = np.full((len(cells), side, side), NO_ROUTE, np.uint8)
    code_for = {tuple(map(int, d)): i for i, d in enumerate(DIRECTION_OFFSETS)}
    for row, source in enumerate(cells):
        sy, sx = divmod(int(source), width)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                gx, gy = sx + dx, sy + dy
                if not (0 <= gx < width and 0 <= gy < height):
                    continue
                if dx == 0 and dy == 0:
                    hops[row, dy + radius, dx + radius] = STAY
                elif dx:
                    hops[row, dy + radius, dx + radius] = code_for[(np.sign(dx), 0)]
                else:
                    hops[row, dy + radius, dx + radius] = code_for[(0, np.sign(dy))]
    return LocalRouteTable(jnp.asarray(cell_to_row), jnp.asarray(hops), radius)


def _terrain(width=7, height=7):
    return TerrainGrid(jnp.ones((height, width), bool), 1.0, 0.0, 0.0)


def test_lookup_distinguishes_coverage_window_and_no_route():
    covered = np.ones(49, bool)
    covered[8] = False
    table = _table(covered=covered)

    code, status = lookup_local_hop(0, 3, table, 7)
    assert int(status) == LocalRouteStatus.READY
    assert tuple(DIRECTION_OFFSETS[int(code)]) == (1, 0)

    _, status = lookup_local_hop(8, 9, table, 7)
    assert int(status) == LocalRouteStatus.SOURCE_UNCOVERED

    _, status = lookup_local_hop(0, 6, table, 7)
    assert int(status) == LocalRouteStatus.GOAL_OUTSIDE_WINDOW

    broken = table._replace(next_hop=table.next_hop.at[0, 4, 5].set(NO_ROUTE))
    _, status = lookup_local_hop(0, 1, broken, 7)
    assert int(status) == LocalRouteStatus.NO_ROUTE


def test_packed_uint4_lookup_matches_unpacked_fixture():
    table = _table()
    flat = np.asarray(table.next_hop).reshape(table.next_hop.shape[0], -1)
    nibble = np.where(flat == NO_ROUTE, 0xF, flat).astype(np.uint8)
    if nibble.shape[1] & 1:
        nibble = np.pad(nibble, ((0, 0), (0, 1)), constant_values=0xF)
    packed = nibble[:, 0::2] | (nibble[:, 1::2] << 4)
    packed_table = table._replace(next_hop=jnp.asarray(packed))
    for source, goal in ((0, 3), (8, 24), (30, 9), (48, 48)):
        a = lookup_local_hop(source, goal, table, 7)
        b = lookup_local_hop(source, goal, packed_table, 7)
        assert tuple(map(int, a)) == tuple(map(int, b))


def test_route_reconstruction_compresses_collinear_cells_and_keeps_float_goal():
    # Cell (1,1) -> (4,3): x-then-y has one turn at cell centre (4.5,1.5).
    # `smooth=False` is the pre-SmoothPath control: collinear compression only.
    result = build_local_waypoints(
        jnp.float32(1.25), jnp.float32(1.75),
        jnp.float32(4.2), jnp.float32(3.8), jnp.float32(0.0),
        _table(), _terrain(), max_raw_hops=16, smooth=False)
    assert int(result.status) == LocalRouteStatus.READY
    assert int(result.n_waypoints) == 3
    np.testing.assert_allclose(
        np.asarray(result.waypoints[:3]),
        [[1.25, 1.75], [4.5, 1.5], [4.2, 3.8]], rtol=0, atol=1e-6)


def test_smoothpath_drops_the_corner_the_line_of_sight_can_skip():
    """The same route, with the server's SmoothPath pass on (the default).

    The L-shaped x-then-y itinerary has one turn, and on open terrain the
    source cell can see the goal cell, so the server's greedy pass deletes it.
    This is the whole point of the pass: it is why the server emits 3 waypoints
    for a click where the raw cell path has tens.
    """
    result = build_local_waypoints(
        jnp.float32(1.25), jnp.float32(1.75),
        jnp.float32(4.2), jnp.float32(3.8), jnp.float32(0.0),
        _table(), _terrain(), max_raw_hops=16)
    assert int(result.status) == LocalRouteStatus.READY
    assert int(result.n_waypoints) == 2
    # Endpoints are untouched: smoothing can only ever remove interior cells.
    np.testing.assert_allclose(
        np.asarray(result.waypoints[:2]),
        [[1.25, 1.75], [4.2, 3.8]], rtol=0, atol=1e-6)
    assert not bool(result.smooth_exhausted)


def test_smoothpath_keeps_the_corner_when_terrain_blocks_the_shortcut():
    """The negative case, which is what makes the test above mean anything.

    Block cell (3,2), which the straight line from cell (1,1) to cell (4,3)
    crosses but neither leg of the L does. The corner must survive.
    """
    walkable = np.ones((7, 7), bool)
    walkable[2, 3] = False                      # (row, col) == (y, x)
    terrain = TerrainGrid(jnp.asarray(walkable), 1.0, 0.0, 0.0)
    result = build_local_waypoints(
        jnp.float32(1.25), jnp.float32(1.75),
        jnp.float32(4.2), jnp.float32(3.8), jnp.float32(0.0),
        _table(), terrain, max_raw_hops=16)
    assert int(result.status) == LocalRouteStatus.READY
    assert int(result.n_waypoints) == 3
    np.testing.assert_allclose(
        np.asarray(result.waypoints[:3]),
        [[1.25, 1.75], [4.5, 1.5], [4.2, 3.8]], rtol=0, atol=1e-6)


def test_smooth_cell_path_is_a_noop_below_three_cells():
    """``if (path.Count < 3) return;`` -- reproduced without a special case.

    The loop starts at i = 2 so it never runs, and the epilogue rewrites
    cells[1] with the cell that is already there.
    """
    cells = jnp.asarray([11, 22] + [0] * (MAX_WAYPOINTS - 2), jnp.int32)
    out, n, exhausted = smooth_cell_path(cells, jnp.int32(2), jnp.float32(0.0),
                                         _terrain())
    assert int(n) == 2
    assert [int(v) for v in out[:2]] == [11, 22]
    assert not bool(exhausted)


def test_smooth_cell_path_matches_the_host_port_on_real_map1_routes():
    """Device SmoothPath against the literal host port, on real terrain.

    Map geometry, not a fixture: the fixture grids are too small and too open
    to exercise the greedy's commit branch. Needs the vendored navgrid but not
    the 231 MiB route artifact, so it runs in an ordinary checkout.
    """
    navgrid = pytest.importorskip(
        "lanerl_jax.data.navgrid",
        reason="vendored Map1 navgrid content is not available here")
    grid = navgrid.NavGrid.load()
    terrain = map1_terrain()
    radius = navgrid.GAREN_PATHFINDING_RADIUS
    width = grid.cell_count_x
    rng = np.random.default_rng(0)

    checked = 0
    for _ in range(400):
        if checked >= 12:
            break
        a = grid.cell_center_world(*(int(v) for v in rng.integers(
            (40, 40), (width - 40, grid.cell_count_y - 40))))
        b = (a[0] + float(rng.integers(-1200, 1200)),
             a[1] + float(rng.integers(-1200, 1200)))
        if not (grid.is_walkable_world(*a, radius)
                and grid.is_walkable_world(*b, radius)):
            continue
        route = grid.get_cell_path(a, b, radius)
        if route is None or len(route.cells) < 6:
            continue
        cells = [int(y) * width + int(x) for x, y in route.cells]
        expect = [(c % width, c // width) for c in cells]
        grid._smooth(expect, radius)

        padded = np.zeros(MAX_WAYPOINTS, np.int32)
        padded[:len(cells)] = cells[:MAX_WAYPOINTS]
        out, n, exhausted = smooth_cell_path(
            jnp.asarray(padded), jnp.int32(min(len(cells), MAX_WAYPOINTS)),
            jnp.float32(radius), terrain)
        assert not bool(exhausted), "SmoothPath CastCircle hit its fixed bound"
        got = [(int(c) % width, int(c) // width) for c in out[:int(n)]]
        assert got == expect, f"{a} -> {b}"
        checked += 1

    assert checked >= 6, "corpus generated too few multi-turn routes to be a test"


def test_failed_route_returns_two_point_fallback_and_nonready_status():
    table = _table()
    # The relative cell goal is outside the table's +/-4-cell window.
    result = build_local_waypoints(
        jnp.float32(0.25), jnp.float32(0.25),
        jnp.float32(6.25), jnp.float32(0.25), jnp.float32(0.0),
        table, _terrain(), max_raw_hops=16)
    assert int(result.status) == LocalRouteStatus.GOAL_OUTSIDE_WINDOW
    assert int(result.n_waypoints) == 2
    np.testing.assert_allclose(np.asarray(result.waypoints[:2]),
                               [[0.25, 0.25], [6.25, 0.25]])


def test_fractional_goal_in_uncovered_centre_cell_uses_nearby_anchor():
    covered = np.ones(49, bool)
    covered[3 * 7 + 4] = False  # exact goal's containing cell has no mesh row
    result = build_local_waypoints(
        jnp.float32(1.25), jnp.float32(3.25),
        jnp.float32(4.2), jnp.float32(3.2), jnp.float32(0.0),
        _table(covered=covered), _terrain(), max_raw_hops=16)
    assert int(result.status) == LocalRouteStatus.READY
    points = np.asarray(result.waypoints[:int(result.n_waypoints)])
    np.testing.assert_allclose(points[-1], [4.2, 3.2], atol=1e-6)
    # The penultimate point is the covered cell-centre anchor; the exact float
    # goal is retained rather than quantized to it.
    assert not np.allclose(points[-2], points[-1])


def test_route_reconstruction_jits():
    table, terrain = _table(), _terrain()
    fn = jax.jit(lambda sx, sy, gx, gy: build_local_waypoints(
        sx, sy, gx, gy, jnp.float32(0), table, terrain, max_raw_hops=16))
    result = fn(jnp.float32(1.25), jnp.float32(1.75),
                jnp.float32(4.2), jnp.float32(3.8))
    assert int(result.status) == LocalRouteStatus.READY
    # 2, not 3: SmoothPath runs inside the jit too, and on open terrain the
    # source cell sees the goal cell, so the L's corner goes.
    assert int(result.n_waypoints) == 2


def test_same_direction_run_sidecar_matches_one_hop_route_and_overflow():
    """Jumping a straight raw run must not change turns or the hop cap."""
    table, terrain = _table(), _terrain()
    side = 2 * table.offset_radius + 1
    runs = np.zeros((49, side, side), np.uint8)
    # (1,1) -> (4,3) follows three east then two north hops in `_table`.
    # The sidecar is indexed by the current source row and its relative goal.
    goal = 3 * 7 + 4
    for current, length in ((1 * 7 + 1, 3), (1 * 7 + 4, 2),
                            (2 * 7 + 4, 1)):
        cy, cx = divmod(current, 7)
        gy, gx = divmod(goal, 7)
        runs[current, gy - cy + table.offset_radius,
             gx - cx + table.offset_radius] = length
    jumped = table._replace(run_length=jnp.asarray(runs))
    args = (jnp.float32(1.25), jnp.float32(1.75),
            jnp.float32(4.2), jnp.float32(3.8), jnp.float32(0.0))
    for cap in (16, 2):
        one_hop = build_local_waypoints(*args, table, terrain, max_raw_hops=cap)
        run_hop = build_local_waypoints(*args, jumped, terrain, max_raw_hops=cap)
        for a, b in zip(one_hop, run_hop):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def test_apply_orders_installs_route_and_persists_diagnostic_status():
    state = empty_state().replace(
        kind=jnp.zeros(66, jnp.int8).at[:2].set(Kind.CHAMPION),
        alive=jnp.zeros(66, bool).at[:2].set(True),
        x=jnp.zeros(66, jnp.float32).at[:2].set(
            jnp.asarray([1.25, 1.25], jnp.float32)),
        y=jnp.zeros(66, jnp.float32).at[:2].set(
            jnp.asarray([1.75, 1.75], jnp.float32)))
    orders = Orders(
        kind=jnp.asarray([OrderKind.MOVE, OrderKind.NOOP], jnp.int8),
        x=jnp.asarray([4.2, 1.25], jnp.float32),
        y=jnp.asarray([3.8, 1.75], jnp.float32),
        target=jnp.asarray([-1, -1], jnp.int8))
    params = {
        "pathfinding_radius": jnp.asarray([0.0], jnp.float32),
        "attack_damage": jnp.asarray([78.0], jnp.float32),
        "ad_per_level": jnp.asarray([4.0], jnp.float32),
    }
    routed = apply_orders(state, orders, params,
                          route_table=_table(), terrain=_terrain())
    assert int(routed.route_status[0]) == LocalRouteStatus.READY
    # Smoothed: the open-terrain corner is gone. See
    # test_smoothpath_drops_the_corner_the_line_of_sight_can_skip.
    assert int(routed.n_waypoints[0]) == 2
    # A non-Move action does not overwrite the last route diagnostic.
    assert int(routed.route_status[1]) == LocalRouteStatus.READY
