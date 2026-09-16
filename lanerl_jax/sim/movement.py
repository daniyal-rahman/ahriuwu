"""Waypoint following, ported from ``AttackableUnit.Move``.

The server's integrator, verbatim
---------------------------------
``GameServerLib/GameObjects/AttackableUnits/AttackableUnit.cs:931``::

    if (CurrentWaypointKey < Waypoints.Count) {
        float speed = GetMoveSpeed() * 0.001f;
        var maxDist = speed * delta;
        while (true) {
            var dir  = CurrentWaypoint - Position;
            var dist = dir.Length();
            if (maxDist < dist) { Position += dir / dist * maxDist; return true; }
            Position = CurrentWaypoint;
            maxDist -= dist;
            CurrentWaypointKey++;
            if (CurrentWaypointKey == Waypoints.Count || maxDist == 0) return true;
        }
    }

Three details that a "move towards the target" approximation gets wrong, all of
which matter for position parity:

1. **Leftover distance carries across waypoints.** Reaching a waypoint does not
   end the tick -- the remainder of ``maxDist`` is spent on the next leg. A
   champion rounding a tight corner can cross two or three waypoints in one
   16.67 ms tick. This is why the JAX version needs a *bounded loop* and why
   that bound is a measured number, not a guess.
2. **``SetWaypoints`` starts at index 1.** ``Waypoints[0]`` is where the unit
   already is; ``CurrentWaypointKey`` starts at 1
   (``AttackableUnit.ResetWaypoints``). Off by one here and every path starts
   with a stall.
3. **Speed is per millisecond** (``* 0.001f``), and ``delta`` is milliseconds,
   so a tick moves ``MoveSpeed * 16.667 / 1000`` units -- 5.75 for Garen's base
   345. That is under half a navgrid cell, which is why the greedy entity
   matching in the parity differ is sound at one-tick granularity.

Not modelled here
-----------------
``ObjAIBase.Move`` refuses to move at all under ``CastSpell``/``OrderNone``/
``Stop``/``Taunt`` move orders, and ``MovementParameters`` (dashes) bypass this
path entirely. Both belong to the order layer, not the integrator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Sequence, Tuple

import numpy as np

__all__ = ["MoveState", "step_move", "follow", "MAX_WAYPOINTS_PER_TICK"]

#: Bound for the carry-over loop. The server's ``while(true)`` is unbounded; a
#: JAX port must fix it. 8 is provisional -- J1 must *measure* the true maximum
#: over the corpus before this is frozen, because a too-small bound silently
#: stalls a unit mid-tick and a stall looks exactly like a pathing bug.
MAX_WAYPOINTS_PER_TICK = 8


@dataclass(slots=True)
class MoveState:
    x: float
    y: float
    waypoints: List[Tuple[float, float]] = field(default_factory=list)
    #: ``CurrentWaypointKey``. 1 after ``SetWaypoints``, never 0.
    key: int = 1

    @property
    def path_ended(self) -> bool:
        return self.key >= len(self.waypoints)

    def set_waypoints(self, pts: Sequence[Tuple[float, float]]) -> None:
        self.waypoints = list(pts)
        self.key = 1


def step_move(st: MoveState, move_speed: float, delta_ms: float,
              max_waypoints: int = MAX_WAYPOINTS_PER_TICK) -> bool:
    """One tick of ``AttackableUnit.Move``. Mutates ``st``; returns ``moved``."""
    if st.key >= len(st.waypoints):
        return False
    max_dist = move_speed * 0.001 * delta_ms
    for _ in range(max_waypoints):
        wx, wy = st.waypoints[st.key]
        dx, dy = wx - st.x, wy - st.y
        dist = float(np.hypot(dx, dy))
        if max_dist < dist:
            st.x += dx / dist * max_dist
            st.y += dy / dist * max_dist
            return True
        st.x, st.y = wx, wy
        max_dist -= dist
        st.key += 1
        if st.key == len(st.waypoints) or max_dist == 0:
            return True
    # Falling out of the bound means the unit had more waypoints to consume than
    # `max_waypoints` allowed. Silent truncation is how a unit ends a tick in
    # the wrong place with nothing in the log, so say so.
    raise RuntimeError(
        f"consumed {max_waypoints} waypoints in one tick without exhausting "
        f"maxDist ({max_dist:.3f} left). MAX_WAYPOINTS_PER_TICK is too small "
        "for this path -- raise it and re-measure rather than clamping."
    )


def follow(start: Tuple[float, float], waypoints: Sequence[Tuple[float, float]],
           move_speed: float, ticks: int, delta_ms: float = 1000.0 / 60.0
           ) -> np.ndarray:
    """Integrate ``ticks`` ticks along ``waypoints``. -> (ticks+1, 2) positions."""
    st = MoveState(x=start[0], y=start[1])
    st.set_waypoints([start, *waypoints[1:]] if waypoints and
                     waypoints[0] == start else [start, *waypoints])
    out = np.empty((ticks + 1, 2), dtype=np.float64)
    out[0] = (st.x, st.y)
    for t in range(ticks):
        step_move(st, move_speed, delta_ms)
        out[t + 1] = (st.x, st.y)
    return out
