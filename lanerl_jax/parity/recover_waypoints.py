"""Recover a unit's waypoint list from the positions it actually walked.

Why this exists
---------------
``LanerlStateDump`` carries ``Waypoints.Count`` but not the waypoints
themselves, so when a path disagreement shows up all the oracle says is "the
server held 6 waypoints and you computed 7".  That is enough to know the
pathfinder disagreed and useless for finding *where*.

But the trajectory is in the dump, once per tick, and
``AttackableUnit.Move`` walks **straight lines between waypoints at constant
speed**.  So the path is a piecewise-linear curve sampled at 60 Hz, and its
breakpoints are the waypoints.  Recovering them turns "the counts differ" into
"your third waypoint is at (2740, 12980) and the server's is at (2690, 13030)",
which is a diagnosis.

Method
------
Walk the sampled positions and greedily extend a straight run for as long as
every sample stays within ``tol`` of the chord.  A break starts a new run.  This
is Ramer-Douglas-Peucker specialised to the case where we know the underlying
curve is exactly piecewise linear, so the tolerance is set by sampling noise
(the dump's 1/16 quantisation) rather than by how much simplification we want.

Limits, stated because a recovered path is evidence and evidence needs error
bars:

* **Collinear waypoints are invisible.** ``SmoothPath`` already removes most of
  them, but three cells in a straight line collapse to one run here and the
  count comes out low. Compare *geometry*, not counts, when using this.
* **A waypoint reached and left within one tick is invisible** -- the champion
  moves 5.75 units per tick at base speed, so a corner cut inside that is lost.
* **A stalled unit yields one run**, which is correct but uninformative.
* The first and last points are the order's true endpoints, not cell centres:
  ``LanerlControl`` sets ``path[0] = champ.Position`` and ``GetPath`` appends the
  snapped goal verbatim.
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

__all__ = ["recover_waypoints", "point_line_distance", "compare_paths"]


def point_line_distance(p: Tuple[float, float], a: Tuple[float, float],
                        b: Tuple[float, float]) -> float:
    """Distance from ``p`` to the segment ``a``-``b``."""
    ax, ay = a
    bx, by = b
    px, py = p
    dx, dy = bx - ax, by - ay
    L2 = dx * dx + dy * dy
    if L2 == 0.0:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / L2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def recover_waypoints(positions: Sequence[Tuple[float, float]],
                      tol: float = 0.25,
                      min_run: int = 2) -> List[Tuple[float, float]]:
    """Piecewise-linear breakpoints of a sampled trajectory.

    ``tol`` defaults to 4x the dump's 1/16 quantisation: tight enough that a
    real corner (which deflects by whole units within a tick or two) is always
    caught, loose enough that quantisation alone never manufactures one.
    """
    pts = [tuple(map(float, p)) for p in positions]
    # Drop the stalled tail: once the unit stops, every further sample is the
    # same point and would otherwise anchor a spurious final run.
    while len(pts) > 1 and math.dist(pts[-1], pts[-2]) < 1e-9:
        pts.pop()
    if len(pts) < 2:
        return list(pts)

    out = [pts[0]]
    anchor = 0
    i = 1
    while i < len(pts):
        # extend while every intermediate sample stays on the chord
        j = i + 1
        while j < len(pts):
            if max(point_line_distance(pts[k], pts[anchor], pts[j])
                   for k in range(anchor + 1, j)) > tol:
                break
            j += 1
        end = j - 1
        if end - anchor >= min_run or end == len(pts) - 1:
            out.append(pts[end])
            anchor = end
            i = end + 1
        else:
            i += 1
    if out[-1] != pts[-1]:
        out.append(pts[-1])
    return out


def compare_paths(mine: Sequence[Tuple[float, float]],
                  recovered: Sequence[Tuple[float, float]]) -> dict:
    """How far the two routes are apart, as geometry rather than as counts.

    **``route_deviation`` is the number to read.** It is the largest distance
    from a *recovered* vertex to the computed polyline: near zero means the
    server walked the route this port computed, whatever the vertex counts say.
    Counts are not comparable directly -- ``SmoothPath`` leaves collinear
    vertices that recovery collapses, and a corner cut inside one 5.75-unit tick
    splits one vertex into two.

    The reverse direction (``mine_to_recovered``) is reported but **confounded**:
    a hold window usually ends before the champion reaches the goal, so the tail
    of the computed path was never walked and scores as a deviation. Use it only
    when the trajectory is known to have completed.
    """
    def dev(pts, poly):
        if len(poly) < 2:
            return float("inf")
        return max(min(point_line_distance(p, poly[k], poly[k + 1])
                       for k in range(len(poly) - 1)) for p in pts)

    fwd = dev(list(recovered), list(mine))
    rev = dev(list(mine), list(recovered))
    return {
        "my_vertices": len(mine),
        "recovered_vertices": len(recovered),
        "route_deviation": fwd,          # <- the diagnostic
        "mine_to_recovered": rev,        # confounded by partial traversal
    }
