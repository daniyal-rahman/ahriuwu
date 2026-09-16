"""The navgrid port, and the server quirks it must reproduce rather than fix.

Several of these assert behaviour that looks like a bug. It is: it is the
*server's* bug, and this module is a reference for what the server does. A
"corrected" reference silently disagrees with the thing it is supposed to
predict, which is the whole failure mode the parity suite exists to prevent.
"""
from __future__ import annotations

import math

import pytest

from lanerl_jax.data.dotnet_pq import DotNetPriorityQueue
from lanerl_jax.data.navgrid import (
    DEFAULT_NGRID,
    GAREN_PATHFINDING_RADIUS,
    NavGrid,
    NavigationGridCellFlags,
)
from lanerl_jax.sim.movement import MoveState, follow, step_move

pytestmark = pytest.mark.skipif(
    not DEFAULT_NGRID.exists(), reason="vendored Map1 navgrid not available"
)


@pytest.fixture(scope="module")
def grid() -> NavGrid:
    return NavGrid.load()


def test_map1_header_matches_the_server(grid):
    assert (grid.cell_count_x, grid.cell_count_y) == (293, 294)
    assert grid.cell_size == 50.0
    assert grid.flags.size == 86_142


def test_walkability_excludes_see_through_not_just_not_passable(grid):
    """``IsWalkable`` tests BOTH flags. Missing SEE_THROUGH costs 516 cells."""
    not_passable_only = int(
        ((grid.flags & NavigationGridCellFlags.NOT_PASSABLE) == 0).sum()
    )
    assert int(grid.walkable_mask().sum()) == 53_135
    assert not_passable_only == 53_651
    assert not_passable_only - int(grid.walkable_mask().sum()) == 516


def test_translation_round_trips_to_the_cell_centre(grid):
    for wx, wy in ((1000.0, 10000.0), (7550.0, 13407.0), (26.0, 280.0)):
        cx, cy = grid.to_nav(wx, wy)
        ix, iy = int(cx), int(cy)
        ox, oy = grid.cell_center_world(ix, iy)
        assert abs(ox - wx) <= grid.cell_size and abs(oy - wy) <= grid.cell_size


def test_short_cast_truncates_toward_zero_not_floor(grid):
    """``(short)(-0.5)`` is 0 in C#; ``floor(-0.5)`` is -1.

    ``GetAllCellsInRange`` casts ``origin - radius``, which goes negative near
    the map edge, so using ``floor`` would enumerate a row the server never
    looks at.
    """
    from lanerl_jax.data.navgrid import _short

    assert _short(-0.5) == 0 and _short(-1.7) == -1 and _short(2.9) == 2
    assert _short(-0.5) != math.floor(-0.5)


def test_an_off_grid_point_reports_walkable_at_radius_but_not_at_zero(grid):
    """A server quirk, reproduced deliberately.

    ``IsWalkable(coords, r>0)`` iterates ``GetAllCellsInRange``, which yields
    nothing off-grid, and the server's loop falls through to ``return true``.
    So an off-map point is "walkable" at radius 35 and unwalkable at radius 0.
    ``GetClosestTerrainExit`` then returns it unchanged and ``GetPath`` returns
    null -- which is exactly what the real server does, and why a move order to
    an off-map point becomes a raw straight line.
    """
    off = (-5000.0, 1000.0)
    assert grid.is_walkable_world(*off, GAREN_PATHFINDING_RADIUS) is True
    assert grid.is_walkable_world(*off, 0.0) is False


def test_closest_terrain_exit_drifts_it_does_not_orbit(grid):
    """The spiral ACCUMULATES into ``location``; it does not sample a ring.

    Getting this wrong (a fixed-centre polar search) moved champion
    trajectories by up to 317 units against the server while agreeing perfectly
    whenever the goal happened to already be walkable.
    """
    inside_wall = None
    for x in range(2000, 6000, 50):
        for y in range(11000, 13000, 50):
            if not grid.is_walkable_world(float(x), float(y), 0.0):
                inside_wall = (float(x), float(y))
                break
        if inside_wall:
            break
    assert inside_wall is not None, "no unwalkable probe point found near lane"
    ex = grid.closest_terrain_exit(*inside_wall, 0.0)
    assert grid.is_walkable_world(*ex, 0.0)
    # a drifting spiral lands further away than the nearest walkable cell would
    assert math.dist(inside_wall, ex) > 0.0


def test_paths_in_open_lane_are_near_straight(grid):
    a, b = (1000.0, 10000.0), (1400.0, 10400.0)
    p = grid.get_path(a, b, radius=GAREN_PATHFINDING_RADIUS)
    assert p is not None and len(p) >= 2
    length = sum(math.dist(p[i], p[i + 1]) for i in range(len(p) - 1))
    assert length / math.dist(a, b) < 1.15


def test_path_from_equals_to_is_none_like_the_server(grid):
    assert grid.get_path((1000.0, 10000.0), (1000.0, 10000.0)) is None


def test_a_path_around_a_corner_has_intermediate_waypoints(grid):
    p = grid.get_path((880.0, 10180.0), (7550.0, 13407.0),
                      radius=GAREN_PATHFINDING_RADIUS)
    assert p is not None and len(p) > 2


# --------------------------------------------------------------- movement --
def test_move_speed_is_per_millisecond(grid):
    """345 units/s over one 16.667 ms tick is 5.75 units, not 345."""
    p = follow((0.0, 0.0), [(0.0, 0.0), (10000.0, 0.0)], 345.0, ticks=1)
    assert p[1][0] == pytest.approx(5.75)
    p = follow((0.0, 0.0), [(0.0, 0.0), (10000.0, 0.0)], 345.0, ticks=60)
    assert p[-1][0] == pytest.approx(345.0, abs=1e-6)


def test_leftover_distance_carries_across_waypoints():
    """Reaching a waypoint does not end the tick -- the remainder is spent.

    A follower that stopped at each waypoint would lag by up to one waypoint
    per tick around a tight corner.
    """
    st = MoveState(x=0.0, y=0.0)
    st.set_waypoints([(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (1000.0, 0.0)])
    step_move(st, 345.0, 1000.0 / 60.0)      # 5.75 units, past two waypoints
    assert st.x == pytest.approx(5.75)
    assert st.key == 3


def test_waypoints_start_at_index_one():
    """``SetWaypoints`` sets ``CurrentWaypointKey = 1``; [0] is where we are."""
    st = MoveState(x=5.0, y=5.0)
    st.set_waypoints([(5.0, 5.0), (105.0, 5.0)])
    assert st.key == 1
    step_move(st, 345.0, 1000.0 / 60.0)
    assert st.x > 5.0


def test_exhausting_the_waypoint_bound_raises_rather_than_stalling():
    from lanerl_jax.sim.movement import step_move as sm

    st = MoveState(x=0.0, y=0.0)
    st.set_waypoints([(0.0, 0.0)] + [(float(i) * 0.01, 0.0) for i in range(1, 40)])
    with pytest.raises(RuntimeError, match="MAX_WAYPOINTS_PER_TICK"):
        sm(st, 345.0, 1000.0 / 60.0, max_waypoints=3)


# ------------------------------------------------------------ .NET heap ----
def test_dotnet_pq_is_a_correct_min_heap():
    import random

    rng = random.Random(0)
    for _ in range(200):
        q = DotNetPriorityQueue()
        prio = {}
        for i in range(rng.randrange(1, 80)):
            p = float(rng.randrange(0, 10))
            prio[i] = p
            q.enqueue(i, p)
        out = []
        while (r := q.try_dequeue()) is not None:
            out.append(prio[r[0]])
        assert out == sorted(out)


def test_dotnet_pq_uses_four_ary_parent_indexing():
    """Arity 4, not 2 -- the sift path differs and so does the tie order."""
    q = DotNetPriorityQueue()
    for i in range(21):
        q.enqueue(i, float(20 - i))
    # index 0's children are 1..4 in a 4-ary heap, 1..2 in a binary one
    assert len(q._nodes) == 21
    root = q.try_dequeue()
    assert root[1] == 0.0


def test_dotnet_pq_rejects_nan_rather_than_corrupting_order():
    q = DotNetPriorityQueue()
    with pytest.raises(ValueError, match="NaN"):
        q.enqueue("x", float("nan"))
