"""Champion orders: the sim's action ingress, mirroring ``LanerlControl``.

Semantic orders, not discretised ones
-------------------------------------
``LanerlControl`` describes itself as *"deliberately dumb: it executes semantic
orders (move here / attack that / cast slot) and leaves discretization to
Python, so the action space can change without recompiling C#."*  The same seam
is right here, and for a stronger reason: the policy's action space is a
96x54 screen grid plus an 8-way button and a 32-slot target head
(``lanerl_rl.constants``), and **that mapping is part of the observation
wrapper, not the simulation**.  Keeping it out means the action space can be
re-tuned without touching a single mechanic, and the sim's parity against the
server is not entangled with a discretisation choice.

So this module accepts what the wire accepts::

    noop            League orders persist, so this is action-repeat, not a stop
    move  (x, y)    path to a point
    attack (unit)   SetTargetUnit alone -- the engine only swings at targets
                    already in range, so closing the distance is the policy's job
    stop            clear the order

``attack`` is ``SetTargetUnit`` **alone**, and that is deliberate on the server
side too: ``LanerlControl`` notes that adding ``UpdateMoveOrder(AttackTo)``
there *"clobbers the target and the champion never swings"*.

Pathing, and the one place this differs from the wire
-----------------------------------------------------
The server paths a move order through ``GetPath`` and falls back to a raw
two-point line when that returns null. On device there is no A*, so a move order
becomes the two-point line directly. Measured (§1.5 of the plan): in the laning
region 76% of sub-500-unit paths and 45% of sub-1800-unit paths are already
straight lines, so this is exact most of the time and wrong near terrain.
**Booked, unmeasured**, and the fix when it matters is the baked next-hop table,
not an A* in the step function.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from .spells import cast_e
from .state import Kind, LaneState, MoveOrder

__all__ = ["OrderKind", "Orders", "apply_orders"]


class OrderKind:
    NOOP = 0
    MOVE = 1
    ATTACK = 2
    STOP = 3
    #: ``{"t":"cast","slot":..}``. Only E is implemented; the slot travels in
    #: ``Orders.target`` reinterpreted as a slot index, which keeps the tuple
    #: fixed-width. A cast does NOT clear the move order -- Garen spins while
    #: walking.
    CAST_E = 4


class Orders(NamedTuple):
    """One order per champion slot. Arrays are ``(N_CHAMPIONS,)``."""

    kind: jax.Array      # OrderKind
    x: jax.Array
    y: jax.Array
    target: jax.Array    # unit index, -1 for none


def _ad_placeholder(state):
    """Attack damage for the E snapshot.

    Orders are applied before the tick, and the tick is what holds the stat
    tables, so the caller passes AD in via ``state`` rather than this module
    importing the profile tables. Until champion AD is carried on the state
    this reads the champion profile's base, which is exact while no item or
    buff changes it -- true for the whole laning slice today, and a lie the
    moment Q or an item lands. Marked so it is not forgotten.
    """
    import jax.numpy as _jnp

    return _jnp.full(state.x.shape, 78.134765625, state.x.dtype)


def apply_orders(state: LaneState, orders: Orders) -> LaneState:
    """Write champion orders into the state. Non-champion slots are untouched."""
    n = state.kind.shape[-1]
    n_ch = orders.kind.shape[0]
    idx = jnp.arange(n)
    is_ch = idx < n_ch
    champ = is_ch & (state.kind == Kind.CHAMPION) & state.alive

    def per_unit(a, fill):
        return jnp.concatenate([a, jnp.full((n - n_ch,), fill, a.dtype)])

    kind = per_unit(orders.kind.astype(jnp.int8), OrderKind.NOOP)
    ox = per_unit(orders.x.astype(state.x.dtype), 0)
    oy = per_unit(orders.y.astype(state.y.dtype), 0)
    otgt = per_unit(orders.target.astype(jnp.int8), -1)

    casting_e = champ & (kind == OrderKind.CAST_E)
    moving = champ & (kind == OrderKind.MOVE)
    attacking = champ & (kind == OrderKind.ATTACK) & (otgt >= 0)
    stopping = champ & (kind == OrderKind.STOP)

    # `path[0] = champ.Position` -- SetWaypoints requires the path to start on us
    two = jnp.stack([jnp.stack([state.x, state.y], -1),
                     jnp.stack([ox, oy], -1)], 1)
    waypoints = jnp.where(moving[:, None, None],
                          state.waypoints.at[:, :2].set(two), state.waypoints)

    bid, bel, bdur, bpow, _ = cast_e(
        state.buff_id, state.buff_elapsed, state.buff_duration,
        state.buff_power, state.spell_cooldown, casting_e,
        state.spell_level[:, 2], state.hp * 0 + _ad_placeholder(state))

    return state.replace(
        buff_id=bid, buff_elapsed=bel, buff_duration=bdur, buff_power=bpow,
        waypoints=waypoints,
        n_waypoints=jnp.where(moving, jnp.int8(2), state.n_waypoints),
        waypoint_key=jnp.where(moving, jnp.int8(1), state.waypoint_key),
        # A move order CLEARS the target: the server's MoveTo replaces AttackTo,
        # and a champion that keeps a stale target keeps trying to swing at it.
        target=jnp.where(moving, jnp.int8(-1),
                         jnp.where(attacking, otgt, state.target)),
        move_order=jnp.where(
            moving, jnp.int8(MoveOrder.MOVE_TO),
            jnp.where(stopping, jnp.int8(MoveOrder.STOP), state.move_order)),
    )
