"""Attack chase must preserve terrain detours instead of grinding into walls."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.data.chase_routes import derive_chase_routes
from lanerl_jax.data.route_artifact import DIRECTION_OFFSETS
from lanerl_jax.sim.local_pathing import LocalRouteStatus, build_local_waypoints
from lanerl_jax.sim.tests.test_local_pathing import _table, _terrain


def test_landmark_graph_routes_distant_goals_and_disconnected_components():
    # A thin wall with its only opening at the bottom. The local table below
    # records just real adjacent edges; no long destination fits its window.
    width, height, radius = 40, 24, 2
    covered = np.ones((height, width), bool)
    covered[:-1, 20] = False
    covered[:, 2] = False  # a second, disconnected component
    cells = np.flatnonzero(covered.ravel()).astype(np.int32)
    rows = np.full(width * height, -1, np.int32)
    rows[cells] = np.arange(len(cells))
    hops = np.full((len(cells), 5, 5), 255, np.uint8)
    for row, cell in enumerate(cells):
        y, x = divmod(int(cell), width)
        hops[row, radius, radius] = 0
        for code, (dx, dy) in enumerate(DIRECTION_OFFSETS[1:], 1):
            # Four-neighbour edges are enough for this fixture.
            if dx and dy:
                continue
            gx, gy = x + int(dx), y + int(dy)
            if 0 <= gx < width and 0 <= gy < height and covered[gy, gx]:
                hops[row, radius + dy, radius + dx] = code
    art = SimpleNamespace(source_cells=cells, cell_to_row=rows, next_hop=hops,
                          manifest=SimpleNamespace(grid_shape=(height, width),
                                                   offset_radius=radius))
    owners, routes = derive_chase_routes(art)
    assert np.all(owners[cells] >= 0)
    source, goal = 5 * width + 15, 5 * width + 25
    owner = owners[goal]
    visited = set()
    while routes[rows[source], owner] != 0:
        assert source not in visited
        visited.add(source)
        code = int(routes[rows[source], owner])
        assert 1 <= code <= 8
        dx, dy = DIRECTION_OFFSETS[code]
        nxt = source + int(dy) * width + int(dx)
        assert covered.ravel()[nxt]
        source = nxt
    # Every goal's attachment is within the local window by graph distance.
    assert abs(source % width - goal % width) < radius
    assert abs(source // width - goal // width) < radius
    assert any(v // width == height - 1 for v in visited)
    assert routes[rows[0], owner] == 255  # disconnected really means no route


def test_tick_routes_attack_around_obstacle_and_retains_route_on_failure():
    from lanerl_jax.sim.config import SimConfig
    from lanerl_jax.sim.init import init_lane
    from lanerl_jax.sim.state import MoveOrder, TU_SLICE
    from lanerl_jax.sim.step import tick

    # Reuse the small x-then-y graph on a 100-unit grid. A wall blocks the
    # direct diagonal, while the graph's east-then-north route is open.
    terrain = _terrain()._replace(cell_size=100., walkable=
        _terrain().walkable.at[3, 3].set(False))
    table = _table(radius=6)
    cfg = SimConfig.unit_test()
    state = init_lane()
    target = TU_SLICE.start + 1
    alive = jnp.zeros_like(state.alive).at[0].set(True).at[target].set(True)
    state = state.replace(alive=alive, team=state.team.at[target].set(1), x=state.x.at[0].set(150.).at[target].set(550.),
        y=state.y.at[0].set(150.).at[target].set(550.),
        target=state.target.at[0].set(target),
        move_order=state.move_order.at[0].set(MoveOrder.ATTACK_TO))
    params = dict(cfg.params, pathfinding_radius=jnp.zeros_like(cfg.params['pathfinding_radius']))
    run = jax.jit(lambda s, rt: tick(s, params, enable_collision=False,
                                   route_table=rt, terrain=terrain))
    out = run(state, table)
    assert int(out.route_status[0]) == LocalRouteStatus.READY
    assert int(out.n_waypoints[0]) >= 3
    assert not np.allclose(out.waypoints[0, 1], [550., 550.])
    # A broken lookup must keep the detour, not replace it with [us,target].
    broken = table._replace(next_hop=jnp.full_like(table.next_hop, 255))
    failed = run(out, broken)
    assert int(failed.route_status[0]) != LocalRouteStatus.READY
    np.testing.assert_array_equal(failed.waypoints[0], out.waypoints[0])
    assert int(failed.n_waypoints[0]) == int(out.n_waypoints[0])


def test_distant_move_uses_global_routes_beyond_local_window():
    from lanerl_jax.sim.init import init_lane, lane_params
    from lanerl_jax.sim.orders import Orders, OrderKind, apply_orders
    from lanerl_jax.sim.local_pathing import lookup_local_hop
    width, height = 150, 3
    table = _table(width=width, height=height, radius=1)
    art = SimpleNamespace(source_cells=np.arange(width*height, dtype=np.int32),
        cell_to_row=np.asarray(table.cell_to_row), next_hop=np.asarray(table.next_hop),
        manifest=SimpleNamespace(grid_shape=(height, width), offset_radius=1))
    owners, routes = derive_chase_routes(art)
    table = table._replace(chase_landmark=jnp.asarray(owners), chase_next_hop=jnp.asarray(routes))
    _, status = lookup_local_hop(width+1, width+140, table, width)
    assert int(status) == LocalRouteStatus.GOAL_OUTSIDE_WINDOW
    state = init_lane()
    state = state.replace(x=state.x.at[:2].set(1.5), y=state.y.at[:2].set(1.5))
    params = lane_params()
    params = dict(params, pathfinding_radius=jnp.zeros_like(params['pathfinding_radius']))
    orders = Orders(jnp.array([OrderKind.MOVE, OrderKind.NOOP], jnp.int8),
                    jnp.array([140.5, 0.]), jnp.array([1.5, 0.]),
                    jnp.full(2, -1, jnp.int8))
    out = jax.jit(lambda s, rt: apply_orders(s, orders, params,
        route_table=rt, terrain=_terrain(width=width, height=height)))(state, table)
    assert int(out.route_status[0]) == LocalRouteStatus.READY
    np.testing.assert_allclose(out.waypoints[0, int(out.n_waypoints[0])-1], [140.5, 1.5])
