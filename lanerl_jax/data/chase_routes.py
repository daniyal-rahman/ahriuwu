"""Derive compact full-map chase routes from a validated local route artifact.

Landmarks cover 16-cell blocks. Reverse BFS attaches every cell to its nearest
landmark by graph distance. A chase follows that landmark's tree until the
existing local route to the actual target becomes available. Both parts use
only edges already certified by the artifact's radius-aware CastCircle graph.
This is navigation, not an attempt to reproduce the server's A* tie breaks.
"""
from __future__ import annotations

import numpy as np

from .route_artifact import DIRECTION_OFFSETS, NO_ROUTE


def derive_chase_routes(artifact):
    from numba import njit

    cells = artifact.source_cells
    rows = artifact.cell_to_row
    height, width = artifact.manifest.grid_shape
    radius = artifact.manifest.offset_radius
    side = 2 * radius + 1
    table = artifact.next_hop
    graph = np.full((len(cells), 8), -1, np.int32)
    # An adjacent destination has a direct route iff the baked first hop is
    # that edge. A detour to a neighbour must not become a fabricated edge.
    for edge, (dx, dy) in enumerate(DIRECTION_OFFSETS[1:], 1):
        logical = (radius + int(dy)) * side + radius + int(dx)
        if table.ndim == 3:
            code = table[:, radius + int(dy), radius + int(dx)]
        else:
            code = (table[:, logical // 2] >> ((logical & 1) * 4)) & 15
        dest = cells + int(dy) * width + int(dx)
        safe = np.clip(dest, 0, len(rows) - 1)
        valid = ((code == edge) & (dest >= 0) & (dest < len(rows))
                 & (rows[safe] >= 0))
        graph[:, edge - 1] = np.where(valid, rows[safe], -1)
    reverse = np.array([8, 7, 6, 5, 4, 3, 2, 1], np.uint8)
    for edge in range(8):
        valid = graph[:, edge] >= 0
        if not np.array_equal(graph[graph[valid, edge], reverse[edge] - 1],
                              np.flatnonzero(valid)):
            raise ValueError('chase route graph must be symmetric')
    x, y = cells % width, cells // width
    block = (y // 16) * ((width + 15) // 16) + x // 16
    # Stable choice nearest each block centre; BFS below also covers small
    # disconnected components with their own landmark.
    score = (x % 16 - 7.5) ** 2 + (y % 16 - 7.5) ** 2
    ordered = np.lexsort((cells, score, block))
    _, first = np.unique(block[ordered], return_index=True)
    seeds = list(ordered[first])

    @njit
    def bfs(graph, seed, reverse):
        count = len(graph)
        hops = np.full(count, 255, np.uint8)
        distance = np.full(count, 32767, np.int32)
        queue = np.empty(count, np.int32)
        queue[0] = seed
        hops[seed] = 0
        distance[seed] = 0
        head, tail = 0, 1
        while head < tail:
            current = queue[head]
            head += 1
            for edge in range(8):
                source = graph[current, edge]
                if source >= 0 and distance[source] == 32767:
                    distance[source] = distance[current] + 1
                    hops[source] = reverse[edge]
                    queue[tail] = source
                    tail += 1
        return hops, distance

    nearest = np.full(len(cells), 32767, np.int32)
    owner = np.full(len(cells), -1, np.int32)
    columns = []
    cursor = 0
    while cursor < len(seeds):
        hops, distance = bfs(graph, seeds[cursor], reverse)
        improve = distance < nearest
        nearest[improve] = distance[improve]
        owner[improve] = cursor
        columns.append(hops)
        cursor += 1
        if cursor == len(seeds) and np.any(nearest >= radius):
            # Guarantees the landmark-to-goal suffix fits wholly within the
            # local BFS window, even for tiny or winding components.
            seeds.append(int(np.argmax(nearest)))
    goal_landmark = np.full(len(rows), -1, np.int32)
    goal_landmark[cells] = owner
    return goal_landmark, np.stack(columns, axis=1)
