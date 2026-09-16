"""``AttackableUnit.Move`` in JAX: batched, fixed-shape, no Python control flow.

The validation chain
--------------------
``server  <--measured--  sim/movement.py (numpy)  <--asserted--  this module``

The numpy reference was measured against the real server (median worst-tick
position error 0.062 units against a dump quantised to 0.0625 -- parity at the
oracle's resolution).  This module is then asserted **exactly** equal to the
numpy reference, so it inherits that measurement instead of needing its own
server run.  That is the cheap direction: a server experiment costs minutes and
a boot, an array comparison costs milliseconds.

Turning the server's ``while(true)`` into a fixed loop
-----------------------------------------------------
The server consumes waypoints until the tick's budget runs out::

    while (true) {
        dir = CurrentWaypoint - Position; dist = dir.Length();
        if (maxDist < dist) { Position += dir/dist * maxDist; return; }
        Position = CurrentWaypoint; maxDist -= dist; CurrentWaypointKey++;
        if (CurrentWaypointKey == Waypoints.Count || maxDist == 0) return;
    }

That is unbounded, and JAX needs a fixed trip count.  The loop is therefore
unrolled ``MAX_STEPS_PER_TICK`` times with every iteration masked by "am I still
moving", which is the standard fixed-shape treatment and costs the same work
every tick regardless of how many waypoints are actually crossed.

``MAX_STEPS_PER_TICK`` is a **correctness** bound, not a performance knob: too
small and a unit silently ends the tick short of where the server put it, which
reads as a pathing bug.  At base move speed a tick is 5.75 units and waypoints
are cell centres 50 units apart, so 1 is the common case; the bound exists for
``SmoothPath`` leaving near-coincident waypoints and for haste effects.
:func:`max_steps_used` measures the real requirement over a trajectory so the
bound can be set from data rather than nerve.

No ``lax.cond`` anywhere
------------------------
Every branch here is a ``jnp.where`` over a mask. Under ``vmap`` a ``cond``
lowers to ``select`` and executes both sides anyway, so writing it as a mask is
the same cost and makes the cost visible.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .state import LaneState

__all__ = ["MAX_STEPS_PER_TICK", "TICK_MS", "step_move_units", "max_steps_used"]

#: ``Game.REFRESH_RATE`` = 1000/60 ms, pinned under ``LANERL_FREERUN``.
TICK_MS = 1000.0 / 60.0

#: Waypoints a unit may cross in one tick. See the module docstring.
MAX_STEPS_PER_TICK = 8


def step_move_units(x: jax.Array, y: jax.Array, waypoints: jax.Array,
                    key_idx: jax.Array, n_waypoints: jax.Array,
                    move_speed: jax.Array, can_move: jax.Array,
                    delta_ms: float = TICK_MS,
                    max_steps: int = MAX_STEPS_PER_TICK):
    """One tick of movement for a whole batch of units.

    Args:
      x, y:           (N,) positions
      waypoints:      (N, W, 2)
      key_idx:        (N,) ``CurrentWaypointKey``; 1 after ``SetWaypoints``
      n_waypoints:    (N,) ``Waypoints.Count``
      move_speed:     (N,) units per SECOND (the server stores it this way and
                      multiplies by ``0.001f`` before using it with a ms delta)
      can_move:       (N,) bool. ``ObjAIBase.Move`` refuses under CastSpell /
                      OrderNone / Stop / Taunt, so that gate belongs to the
                      caller, not here.

    Returns:
      ``(x, y, key_idx, moved)``
    """
    dt = jnp.asarray(delta_ms, x.dtype)
    budget = move_speed * jnp.asarray(0.001, x.dtype) * dt
    k = key_idx.astype(jnp.int32)
    n = n_waypoints.astype(jnp.int32)

    # `if (CurrentWaypointKey < Waypoints.Count)` -- the whole method is a no-op
    # otherwise, and `moved` is the method's return value.
    active0 = can_move & (k < n)
    budget = jnp.where(active0, budget, jnp.zeros_like(budget))

    def one(carry, _):
        x, y, k, budget, active = carry
        # Clamp the index so the gather is always in range; `active` is what
        # actually decides whether the result is used.
        kc = jnp.clip(k, 0, waypoints.shape[1] - 1)
        wx = jnp.take_along_axis(waypoints[..., 0], kc[:, None], axis=1)[:, 0]
        wy = jnp.take_along_axis(waypoints[..., 1], kc[:, None], axis=1)[:, 0]
        dx, dy = wx - x, wy - y
        dist = jnp.hypot(dx, dy)

        # Branch A: the waypoint is further than the remaining budget -> move
        # partway and stop for this tick.
        safe = jnp.where(dist > 0, dist, jnp.ones_like(dist))
        partial = active & (budget < dist)
        nx = jnp.where(partial, x + dx / safe * budget, x)
        ny = jnp.where(partial, y + dy / safe * budget, y)

        # Branch B: we reach it -> snap, spend the leg, advance the index.
        reach = active & ~partial
        nx = jnp.where(reach, wx, nx)
        ny = jnp.where(reach, wy, ny)
        nbudget = jnp.where(reach, budget - dist, budget)
        nk = jnp.where(reach, k + 1, k)

        # `if (CurrentWaypointKey == Waypoints.Count || maxDist == 0) return;`
        still = reach & (nk < n) & (nbudget != 0)
        return (nx, ny, nk, nbudget, still), None

    (x, y, k, _, _), _ = jax.lax.scan(
        one, (x, y, k, budget, active0), None, length=max_steps)
    return x, y, k.astype(key_idx.dtype), active0


def max_steps_used(waypoints, key_idx, n_waypoints, move_speed,
                   delta_ms: float = TICK_MS, probe: int = 32) -> jax.Array:
    """How many waypoints a tick would actually consume, per unit.

    Use this to set :data:`MAX_STEPS_PER_TICK` from a real corpus instead of
    guessing. Returns ``(N,)`` counts; if any equals ``probe`` the probe itself
    was too small and the answer is a lower bound.
    """
    k0 = key_idx.astype(jnp.int32)
    x = jnp.zeros_like(move_speed)
    y = jnp.zeros_like(move_speed)
    # walk from the current waypoint, not from an arbitrary origin
    kc = jnp.clip(k0 - 1, 0, waypoints.shape[1] - 1)
    x = jnp.take_along_axis(waypoints[..., 0], kc[:, None], axis=1)[:, 0]
    y = jnp.take_along_axis(waypoints[..., 1], kc[:, None], axis=1)[:, 0]
    _, _, k1, _ = step_move_units(
        x, y, waypoints, key_idx, n_waypoints, move_speed,
        jnp.ones_like(move_speed, dtype=bool), delta_ms, max_steps=probe)
    return k1.astype(jnp.int32) - k0
