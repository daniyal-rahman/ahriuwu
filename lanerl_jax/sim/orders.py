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
    move  (x, y)    path to a point -- does NOT drop a held target, see below
    attack (unit)   SetTargetUnit alone -- the engine only swings at targets
                    already in range, so closing the distance is the policy's job
    stop            a wire kind of ours with no server receiver -- see below

``attack`` is ``SetTargetUnit`` **alone**, and that is deliberate on the server
side too: ``LanerlControl`` notes that adding ``UpdateMoveOrder(AttackTo)``
there *"clobbers the target and the champion never swings"*.

Sticky targets: a Move order does not clear the target
--------------------------------------------------------
``LanerlControl.cs:349-371`` (``case LanerlOrderKind.Move``) paths and calls
``champ.UpdateMoveOrder(OrderType.MoveTo, publish: false)`` -- it never touches
``TargetUnit`` at all. ``UpdateMoveOrder`` itself only clears the target for
``OrderNone``/``Stop``/``PetHardStop`` (``ObjAIBase.cs:1353-1360``); ``MoveTo``
is not in that set. So on the real training/eval/parity server, issuing a Move
order while holding a target does **not** disengage: ``ObjAIBase.
RefreshWaypoints`` (``ObjAIBase.cs:602-604``) flips ``MoveOrder`` back to
``AttackTo`` and re-paths onto the target the very next tick, for as long as
the target stays alive and visible (``step.py``'s "3b. RefreshWaypoints" block
already reproduces exactly this once a target survives here to be held).
Corroborated by ``LanerlBot.cs:1242-1251``'s dedicated ``ClearTarget()``
helper ("drop the target so the engine stops swinging"), which would be
pointless if ``MoveTo`` already cleared it, and which ``MoveTo()`` never
calls.

This was previously implemented backwards here (a Move order zeroed
``target`` unconditionally) -- fixed per ``docs/PORT_AUDIT_AI.md`` row 3.2, on
an explicit user decision to match the server's sticky-target behaviour rather
than the friendlier "Move disengages" reading. **This changes what a policy
can express**: our sim used to let a champion holding a target cleanly
disengage with a single Move action; the real server never allowed that at
all (a target is dropped only by dying, going untargetable, leaving vision, or
being replaced by a new Attack order) -- see :func:`lanerl_jax.sim.autoattack.
step_autoattack` for the one genuine disengage tool that *does* exist
(a swing cancelled by the target leaving range mid-windup, ``ObjAIBase.cs:
1193-1199``/``:1183-1191``).

No wire-level Stop order
-------------------------
``LanerlWire.cs:11-19``'s ``LanerlOrderKind`` enum is exactly ``{Noop, Move,
Attack, Cast, Level, Recall}`` -- there is no ``Stop`` kind on the wire at
all, and ``LanerlControl.Execute``'s switch has no default case either, so an
order kind it does not recognise is simply never acted on. ``OrderKind.STOP``
below is this sim's own addition with no receiver on the real control plane;
keeping it as a "clear everything" button would train a policy against an
action the deployed server cannot execute (``docs/PORT_AUDIT_AI.md`` row 3.4).
It is therefore a true no-op here -- ``move_order``/``target`` are left
untouched, exactly matching what happens when an order this switch does not
handle is ever sent. Not reachable today regardless:
``train/trainer.py``/``train/benchmark.py``'s action decode never emits it
(only NOOP/MOVE/ATTACK/CAST_E), so this changes no existing training or eval
behaviour.

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

from .spells import R_CAST_RANGE, Slot, cast_e, cast_q, cast_r, cast_w, enemy_champion_index
from .state import Kind, LaneState, MoveOrder

__all__ = ["OrderKind", "Orders", "apply_orders"]


class OrderKind:
    NOOP = 0
    MOVE = 1
    ATTACK = 2
    #: No receiver on the real wire (``LanerlWire.LanerlOrderKind`` has no
    #: ``Stop`` member) -- a true no-op in :func:`apply_orders`, kept only so
    #: an action-space index does not need to be renumbered. See the module
    #: docstring's "No wire-level Stop order" section.
    STOP = 3
    #: ``{"t":"cast","slot":..}``. E was the first implemented; Q/W/R below
    #: follow the same one-kind-per-spell shape rather than a single generic
    #: CAST kind decoding ``Orders.target`` as a slot index, because
    #: ``lanerl_rl.constants``/``train/trainer.py`` already hardcode
    #: ``OrderKind.CAST_E`` for button 5 of the action space, and repurposing
    #: ``target``'s meaning for E would have been a silent breaking change to
    #: a file this task does not own. A cast does NOT clear the move order --
    #: Garen spins/empowers/braces while walking.
    CAST_E = 4
    #: Self-cast, no target. See ``spells.cast_q``.
    CAST_Q = 5
    #: Self-cast, no target. See ``spells.cast_w``.
    CAST_W = 6
    #: Single-target. ``Orders.target`` is reinterpreted as the unit to hit,
    #: exactly like ``ATTACK``'s -- but only the enemy champion is a legal R
    #: target (``GarenR.json`` ``TextFlags``: ``AffectEnemies | AffectHeroes``,
    #: no minions/turrets/buildings/neutral/friends), enforced below.
    CAST_R = 7


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
    casting_q = champ & (kind == OrderKind.CAST_Q)
    casting_w = champ & (kind == OrderKind.CAST_W)
    # R's only legal target is the enemy champion (`GarenR.json` TextFlags --
    # see spells.cast_r's docstring), within CastRange. A minion/turret index,
    # an ally index, or an out-of-range enemy champion all fail this and the
    # order becomes a no-op, the same "fail closed" contract as `cast_r`'s own
    # internal re-check.
    enemy_champ = enemy_champion_index(n)
    r_target_ok = (otgt == enemy_champ) & (enemy_champ >= 0)
    r_mirror = jnp.clip(enemy_champ, 0, n - 1)
    r_d2 = ((state.x - state.x[r_mirror]) ** 2
            + (state.y - state.y[r_mirror]) ** 2)
    r_in_range = r_d2 <= (R_CAST_RANGE * R_CAST_RANGE)
    casting_r = champ & (kind == OrderKind.CAST_R) & r_target_ok & r_in_range
    moving = champ & (kind == OrderKind.MOVE)
    attacking = champ & (kind == OrderKind.ATTACK) & (otgt >= 0)
    # `OrderKind.STOP` has no server receiver at all (module docstring) -- not
    # read anywhere below; kept only as a named no-op rather than removed, so
    # this enum's numbering (and any action-space index built against it)
    # does not shift.

    # `path[0] = champ.Position` -- SetWaypoints requires the path to start on us
    two = jnp.stack([jnp.stack([state.x, state.y], -1),
                     jnp.stack([ox, oy], -1)], 1)
    waypoints = jnp.where(moving[:, None, None],
                          state.waypoints.at[:, :2].set(two), state.waypoints)

    bid, bel, bdur, bpow, _ = cast_e(
        state.buff_id, state.buff_elapsed, state.buff_duration,
        state.buff_power, state.spell_cooldown, casting_e,
        state.spell_level[:, Slot.E], state.hp * 0 + _ad_placeholder(state))
    cd = state.spell_cooldown       # cast_e never touches cooldown; step_buffs does.

    bid, bel, bdur, bpow, cd, _ = cast_q(
        bid, bel, bdur, bpow, cd, casting_q, state.spell_level[:, Slot.Q])
    bid, bel, bdur, bpow, cd, _ = cast_w(
        bid, bel, bdur, bpow, cd, casting_w, state.spell_level[:, Slot.W])
    bid, bel, bdur, bpow, cd, _ = cast_r(
        bid, bel, bdur, bpow, cd, casting_r, state.spell_level[:, Slot.R],
        state.hp, state.max_hp, otgt)

    return state.replace(
        buff_id=bid, buff_elapsed=bel, buff_duration=bdur, buff_power=bpow,
        spell_cooldown=cd,
        waypoints=waypoints,
        n_waypoints=jnp.where(moving, jnp.int8(2), state.n_waypoints),
        waypoint_key=jnp.where(moving, jnp.int8(1), state.waypoint_key),
        # A Move order does NOT clear the target -- `LanerlControl.cs:349-371`
        # never calls `SetTargetUnit`, and `UpdateMoveOrder(MoveTo, ...)`
        # itself only clears the target for OrderNone/Stop/PetHardStop
        # (`ObjAIBase.cs:1353-1360`). A held target survives a Move order and
        # is re-engaged the next tick by `step.py`'s "3b. RefreshWaypoints"
        # block (`ObjAIBase.RefreshWaypoints`, `:602-604`) for as long as it
        # stays alive and visible -- see the module docstring's "Sticky
        # targets" section.
        target=jnp.where(attacking, otgt, state.target),
        move_order=jnp.where(moving, jnp.int8(MoveOrder.MOVE_TO), state.move_order),
    )
