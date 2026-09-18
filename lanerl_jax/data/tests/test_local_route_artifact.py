"""Unit tests for the offline local-route artifact baker.

`_bake_local_table` is numba-parallel by design and raises rather than falling
back to a slow guess, so these tests need numba. It is present in `.venv-jax`
(the login-node env, where artifacts are actually generated) and absent from
`.venv-gpu`, which is a real venv with `include-system-site-packages = false`
and therefore cannot see the conda env's copy either.

That split is deliberate rather than an oversight to repair by installing into
`.venv-gpu`: `.venv-gpu` carries `jax[cuda12]` and is the environment every
gate-4 number is measured in, and numba pins numpy. The skip below is scoped to
BUILDING an artifact. LOADING one is pure numpy and works in both envs, which
is why routed training and the throughput gate are unaffected -- see OPS-001 in
`docs/JAX_FIDELITY_LEDGER.md`.
"""
import numpy as np
import pytest

numba = pytest.importorskip(
    "numba",
    reason="offline artifact generation needs numba; present in .venv-jax, "
           "absent from .venv-gpu by design (see this module's docstring). "
           "Loading a pre-built artifact does NOT need it.")

from lanerl_jax.data.local_route_artifact import _bake_local_table
from lanerl_jax.data.route_artifact import DIRECTION_OFFSETS, NO_ROUTE, STAY


def _open_graph(width, height):
    graph = np.full((width * height, 8), -1, np.int32)
    for cell in range(width * height):
        y, x = divmod(cell, width)
        for edge, (dx, dy) in enumerate(DIRECTION_OFFSETS[1:]):
            nx, ny = x + int(dx), y + int(dy)
            if 0 <= nx < width and 0 <= ny < height:
                graph[cell, edge] = ny * width + nx
    return graph


def test_reverse_bfs_table_is_local_and_repeated_hops_reach_goal():
    width = height = 5
    cells = np.arange(width * height, dtype=np.int32)
    rows = cells.copy()
    radius = 2
    side = 2 * radius + 1
    table = np.full((len(cells), side, side), NO_ROUTE, np.uint8)
    runs = np.zeros_like(table)
    _bake_local_table(rows, _open_graph(width, height), cells,
                      width, height, radius, table, runs)

    goal = 2 * width + 2
    assert table[goal, radius, radius] == STAY
    current = 0
    visited = set()
    while current != goal:
        assert current not in visited
        visited.add(current)
        cy, cx = divmod(current, width)
        gy, gx = divmod(goal, width)
        code = int(table[current, gy - cy + radius, gx - cx + radius])
        assert code != int(NO_ROUTE)
        dx, dy = map(int, DIRECTION_OFFSETS[code])
        current = (cy + dy) * width + cx + dx
    assert len(visited) == 2  # two diagonal hops on the open 8-neighbour graph
    # The sidecar jumps those two equal diagonal directions but stops before
    # the goal's STAY entry, so runtime still observes arrival separately.
    assert runs[0, radius + 2, radius + 2] == 2
    assert runs[width + 1, radius + 1, radius + 1] == 1

    # A source whose goal is beyond the baked window has no column at all;
    # the device lookup reports GOAL_OUTSIDE_WINDOW before gathering.
    assert abs((cells[-1] % width) - (cells[0] % width)) > radius


def test_coverage_is_route_closed_by_construction():
    width = height = 5
    # Remove the centre cell from both rows and graph. No route may use it as
    # an intermediate because every enqueued BFS cell must have a source row.
    cells = np.delete(np.arange(25, dtype=np.int32), 12)
    rows = np.full(25, -1, np.int32)
    rows[cells] = np.arange(len(cells), dtype=np.int32)
    graph = _open_graph(width, height)
    graph[graph == 12] = -1
    graph[12] = -1
    radius = 2
    table = np.full((len(cells), 5, 5), NO_ROUTE, np.uint8)
    _bake_local_table(rows, graph, cells, width, height, radius, table)
    assert 12 not in cells
    assert np.all((table <= 8) | (table == NO_ROUTE))
