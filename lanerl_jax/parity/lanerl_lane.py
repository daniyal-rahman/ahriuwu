"""`GameServerLib/Lanerl/LanerlLane.cs`, ported exactly.

WHY A FAITHFUL PORT AND NOT AN APPROXIMATION
--------------------------------------------
The shared heuristic policy needs lane-local geometry to reproduce the C# bot's
`HoldPoint` and `Retreat`, both of which are expressed in arc length along the
lane. The first version of that policy substituted hand-rolled approximations
-- "back off 500 units from the nearest minion" for `HoldPoint`, "walk directly
away from the wave" for `Retreat` -- and each approximation was wrong in its own
way. The standoff guess alone took the server arm from 12 CS to 5, because the
bot's real hold point is the furthest-forward position where it is still
un-acquired, computed from each minion's own acquisition range, not a fixed
distance from the nearest one.

This is pure geometry over a polyline both engines already share: the C#
`TopLaneDefault` here and `lanerl_jax.sim.init.TOP_LANE_PATH` are the same
eleven points, verified. So there is nothing to get out of sync and no
observation to extend -- which is exactly why it was the wrong thing to
approximate.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

__all__ = ["LanerlLane", "TOP_LANE_DEFAULT", "forward_for"]

#: `LanerlLane.TopLaneDefault` (`LanerlLane.cs:26-39`), identical to
#: `lanerl_jax.sim.init.TOP_LANE_PATH`.
TOP_LANE_DEFAULT: Tuple[Tuple[float, float], ...] = (
    (917.0, 1725.0), (1170.0, 4041.0), (861.0, 6459.0), (880.0, 10180.0),
    (1268.0, 11675.0), (2806.0, 13075.0), (3907.0, 13243.0),
    (7550.0, 13407.0), (10244.0, 13238.0), (10947.0, 13135.0),
    (12511.0, 12776.0),
)


def forward_for(is_blue: bool) -> float:
    """`LanerlLane.Forward`: blue walks up the arc length, purple down it."""
    return 1.0 if is_blue else -1.0


class LanerlLane:
    """A lane polyline with arc-length projection."""

    def __init__(self, pts: Sequence[Tuple[float, float]] = TOP_LANE_DEFAULT):
        if len(pts) < 2:
            raise ValueError("a lane needs at least two points")
        self._pts: List[Tuple[float, float]] = [(float(x), float(y)) for x, y in pts]
        self._cum: List[float] = [0.0]
        for a, b in zip(self._pts, self._pts[1:]):
            self._cum.append(self._cum[-1] + math.dist(a, b))
        self.length = self._cum[-1]

    def project(self, p: Tuple[float, float]):
        """Closest point on the lane, its distance, and its arc length.

        Mirrors `Project` (`LanerlLane.cs:54-75`) including the `len2 <= 1e-6`
        degenerate-segment guard and the clamp of `t` to [0, 1].
        """
        closest = self._pts[0]
        distance = float("inf")
        along = 0.0
        px, py = p
        for i in range(len(self._pts) - 1):
            ax, ay = self._pts[i]
            bx, by = self._pts[i + 1]
            abx, aby = bx - ax, by - ay
            len2 = abx * abx + aby * aby
            t = 0.0 if len2 <= 1e-6 else max(
                0.0, min(1.0, ((px - ax) * abx + (py - ay) * aby) / len2))
            qx, qy = ax + abx * t, ay + aby * t
            d = math.hypot(px - qx, py - qy)
            if d < distance:
                distance = d
                closest = (qx, qy)
                along = self._cum[i] + math.hypot(qx - ax, qy - ay)
        return closest, distance, along

    def distance_to(self, p: Tuple[float, float]) -> float:
        return self.project(p)[1]

    def along_of(self, p: Tuple[float, float]) -> float:
        return self.project(p)[2]

    def point_at(self, along: float) -> Tuple[float, float]:
        """`PointAt` (`LanerlLane.cs:90-104`), clamped to both ends."""
        along = max(0.0, min(self.length, along))
        for i in range(len(self._pts) - 1):
            if along <= self._cum[i + 1]:
                seg = self._cum[i + 1] - self._cum[i]
                t = 0.0 if seg <= 1e-6 else (along - self._cum[i]) / seg
                ax, ay = self._pts[i]
                bx, by = self._pts[i + 1]
                return (ax + (bx - ax) * t, ay + (by - ay) * t)
        return self._pts[-1]
