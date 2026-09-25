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

Pathing and its explicit approximation boundary
------------------------------------------------
The server paths a move order through ``GetPath`` and falls back to a raw
two-point line when that returns null. Production training passes a bounded
``LocalRouteTable`` to :func:`apply_orders`: it reconstructs a radius-aware
static-terrain route by repeated local gathers and records a non-ready result in
``state.route_status``. Dynamic minion/champion block remains in
``sim.collision`` where it belongs. A caller that deliberately omits the table
still gets the historical two-point approximation; ``train.run_train`` requires
the production artifact by default and exposes ``--no-route-table`` only as an
explicit comparison/debug switch. The static graph's reverse-BFS choice and
collinear-only smoothing are not bit-identical to server A* + ``SmoothPath``;
the complete boundary is PATH-001--PATH-005 in the fidelity ledger.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..obs.fog import visible_to
from ..obs.frame import make_lane_frame, delta_to_lane
from lanerl_rl.projection import target_on_screen
from .combat import growth_sum
from .spells import (R_CAST_RANGE, R_CAST_TIME_S, Slot, cast_e, cast_q,
                     cast_r, cast_w, enemy_champion_index, status_of)
from .state import Kind, LaneState, MoveOrder, Team

__all__ = ["OrderKind", "Orders", "OBSERVED_CAST_SCREEN_RADIUS", "apply_orders"]


# Legacy diagnostic compatibility only. Production witnessed-cast memory
# below uses the same projected viewport as actor observations.
OBSERVED_CAST_SCREEN_RADIUS = 1800.0


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
    #: ``LanerlOrderKind.Recall``: blue pill's 0.5 s windup followed by its
    #: cancellable 8 s channel.  This is intentionally distinct from the
    #: Garen buff table, whose R-pending mailbox belongs to a different spell.
    RECALL = 8
    #: Screen A-click on ground; normal nearest-enemy acquisition stays in tick.
    ATTACK_MOVE = 9


class Orders(NamedTuple):
    """One order per champion slot. Arrays are ``(N_CHAMPIONS,)``."""

    kind: jax.Array      # OrderKind
    x: jax.Array
    y: jax.Array
    target: jax.Array    # unit index, -1 for none
    # New coordinate-click ingress clears a chase on a ground click. Legacy
    # low-level Move orders omit this and retain their ORDER-001 semantics.
    clear_target: jax.Array | None = None


def _record_observed_enemy_casts(state: LaneState, successful: jax.Array, vision=None) -> jax.Array:
    """Update the two agents' witnessed-enemy-cast clocks at cast ingress.

    ``successful`` is ``(2, 4)`` (caster champion slot, Q/W/E/R), taken from
    the spell helpers' successful result rather than from an order request or
    an enemy cooldown. Each observer learns only about the other champion,
    only when that caster is visible to the observer's team and lies within the
    observer's canonical viewport at this exact pre-tick position.

    A team mate can provide fog visibility, as in the source observer, but does
    not move the camera: the on-screen distance is always from the observing
    champion. Thus a minimap-visible cast never leaks into this memory.
    """
    observer = jnp.arange(2, dtype=jnp.int32)
    observer_team = state.team[:2]
    caster_team = state.team[:2]
    seen_blue = visible_to(Team.BLUE, state.x, state.y, state.kind,
                           state.team, state.alive, vision)
    seen_red = visible_to(Team.RED, state.x, state.y, state.kind,
                          state.team, state.alive, vision)
    # Rows are observers; columns are the two possible champion casters.
    caster_visible = jnp.where(
        observer_team[:, None] == Team.BLUE,
        seen_blue[None, :2], seen_red[None, :2])
    dx = state.x[:2][None, :] - state.x[:2][:, None]
    dy = state.y[:2][None, :] - state.y[:2][:, None]
    from .init import TOP_OUTER_TURRET
    blue_frame = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                                 TOP_OUTER_TURRET[Team.RED], (1131.8, 1426.3))
    red_frame = make_lane_frame(TOP_OUTER_TURRET[Team.RED],
                                TOP_OUTER_TURRET[Team.BLUE], (12760.9, 13026.1))
    blue_ds, blue_dn = delta_to_lane(blue_frame, dx, dy)
    red_ds, red_dn = delta_to_lane(red_frame, dx, dy)
    on_screen = jnp.where(observer_team[:, None] == Team.BLUE,
                          target_on_screen(blue_ds, blue_dn),
                          target_on_screen(red_ds, red_dn))
    observer_live = ((state.kind[:2] == Kind.CHAMPION) & state.alive[:2])
    caster_live = ((state.kind[:2] == Kind.CHAMPION) & state.alive[:2])
    can_witness = (
        observer_live[:, None] & caster_live[None, :]
        & (observer[:, None] != observer[None, :])
        & (observer_team[:, None] != caster_team[None, :])
        & caster_visible & on_screen)
    witnessed = jnp.any(can_witness[:, :, None] & successful[None, :, :], axis=1)
    return jnp.where(witnessed, jnp.zeros_like(state.observed_enemy_cast_ms),
                     state.observed_enemy_cast_ms)


def apply_orders(state: LaneState, orders: Orders, params, *,
                 route_table=None, terrain=None, vision=None) -> LaneState:
    """Write champion orders into the state. Non-champion slots are untouched.

    ``params`` (``lane_params(patch)``) is REQUIRED: E snapshots the caster's
    live, level-scaled AD from it. It used to be optional, with a hard-coded
    level-one AD (``_ad_placeholder``) as a silent fallback that any caller
    forgetting ``params`` got without a word (`STRUCT-003`). Prefer
    :func:`lanerl_jax.sim.step.env_step` with a
    :class:`~lanerl_jax.sim.config.SimConfig`, which also supplies the route
    table.
    """
    if params is None:
        raise TypeError("apply_orders requires params (lane_params(patch)); "
                        "the placeholder-AD fallback was removed (STRUCT-003)")
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

    # ONE availability rule (`spells.status`, `STRUCT-001`), the same one the
    # observation's spell features read -- so the policy is told a spell is
    # available exactly when pressing it does something. `may_cast` is the
    # unit-level half: alive, not silenced, and no ordinary cast in progress
    # (Recall's windup is an ordinary non-instant cast: while `_castingSpell`
    # is live the server refuses movement and every further spell cast; R's
    # windup likewise). The channel, by contrast, is cancellable by a
    # successful ordinary cast. `can_cast` adds each spell's own rule: rank,
    # cooldown, Q's open window, and E's cancel (a press on a spin at least
    # `E_CANCEL_MIN_S` old ends it).
    st = status_of(state)
    can_cast = st.may_cast
    casting_e = champ & st.can_cast[:, Slot.E] & (kind == OrderKind.CAST_E)
    casting_q = champ & st.can_cast[:, Slot.Q] & (kind == OrderKind.CAST_Q)
    casting_w = champ & st.can_cast[:, Slot.W] & (kind == OrderKind.CAST_W)
    # R's only legal target is the enemy champion (`GarenR.json` TextFlags --
    # see spells.cast_r's docstring), within CastRange. A minion/turret index,
    # an ally index, or an out-of-range enemy champion all fail this and the
    # order becomes a no-op, the same "fail closed" contract as `cast_r`'s own
    # internal re-check.
    enemy_champ = enemy_champion_index(n)
    # Targeted SpellData validation rejects a dead unit before the cast state
    # is created.  Keep this distinct from R's *post-cast* target-death rule:
    # a victim dying during its already-live windup does not cancel R.
    r_target_ok = ((otgt == enemy_champ) & (enemy_champ >= 0)
                   & state.alive[jnp.clip(enemy_champ, 0, n - 1)])
    r_mirror = jnp.clip(enemy_champ, 0, n - 1)
    r_d2 = ((state.x - state.x[r_mirror]) ** 2
            + (state.y - state.y[r_mirror]) ** 2)
    r_in_range = r_d2 <= (R_CAST_RANGE * R_CAST_RANGE)
    casting_r = (champ & st.can_cast[:, Slot.R] & (kind == OrderKind.CAST_R)
                 & r_target_ok & r_in_range)
    # SetWaypoints fails while the pill is still winding up (`_castingSpell`),
    # exactly like a server Move packet that cannot pass CanChangeWaypoints.
    moving = (champ & ((kind == OrderKind.MOVE) | (kind == OrderKind.ATTACK_MOVE))
              & (state.recall_windup_ms <= 0) & (state.r_cast_ms <= 0))
    # A live R is an uncancellable ordinary spell cast. Unlike a silence it
    # also prevents target/attack-order changes until FinishCasting.
    attacking = (champ & (kind == OrderKind.ATTACK) & (otgt >= 0)
                 & (state.r_cast_ms <= 0))
    # LanerlControl stops BEFORE it calls `pill.Cast`. A second B press while
    # the pill is already channeling therefore still clears a just-issued
    # MoveTo (the cast itself is refused because that Spell is not READY),
    # rescuing the ongoing channel instead of cancelling it.
    recall_stop = champ & (kind == OrderKind.RECALL) & can_cast
    recalling = recall_stop & (state.recall_channel_ms <= 0)
    # `OrderKind.STOP` has no server receiver at all (module docstring) -- not
    # read anywhere below; kept only as a named no-op rather than removed, so
    # this enum's numbering (and any action-space index built against it)
    # does not shift.

    # `path[0] = champ.Position` -- SetWaypoints requires the path to start on us.
    # With a local table, reconstruct the bounded static-terrain route here.
    # Dynamic bodies deliberately remain absent: CollisionHandler applies
    # minion/champion block later, from their live positions.
    two = jnp.stack([jnp.stack([state.x, state.y], -1),
                     jnp.stack([ox, oy], -1)], 1)
    routed_n = jnp.full((n,), 2, jnp.int8)
    if route_table is None:
        from .local_pathing import LocalRouteStatus
        routed_status = jnp.full((n,), LocalRouteStatus.TABLE_DISABLED, jnp.int8)
    else:
        routed_status = jnp.zeros((n,), jnp.int8)
    candidate_waypoints = state.waypoints.at[:, :2].set(two)
    if route_table is not None:
        if terrain is None or params is None:
            raise ValueError("route_table requires both terrain and profile params")
        from .local_pathing import build_local_waypoints

        path_radius = params["pathfinding_radius"][state.model[:n_ch]]
        # The routed result is committed only under `moving` below.  Giving
        # non-Move orders an identity route keeps those deliberately-discarded
        # lanes from extending the vectorised reconstruction while-loop with a
        # meaningless path to their screen-coordinate placeholder (0, 0).
        # Their visible state/status remains exactly unchanged by the masks in
        # the return value below.
        route_x = jnp.where(moving[:n_ch], ox[:n_ch], state.x[:n_ch])
        route_y = jnp.where(moving[:n_ch], oy[:n_ch], state.y[:n_ch])
        # Long setup moves and screen-edge detours can leave the local
        # table window. Reuse chase's bounded full-map fallback, rather than
        # replacing a reachable destination with a straight failed route.
        routed = jax.vmap(
            lambda sx, sy, gx, gy, radius: build_local_waypoints(
                sx, sy, gx, gy, radius, route_table, terrain,
                max_raw_hops=512, global_chase=True)
        )(state.x[:n_ch], state.y[:n_ch], route_x, route_y, path_radius)
        candidate_waypoints = candidate_waypoints.at[:n_ch].set(routed.waypoints)
        routed_n = routed_n.at[:n_ch].set(routed.n_waypoints)
        routed_status = routed_status.at[:n_ch].set(routed.status)
    waypoints = jnp.where(moving[:, None, None], candidate_waypoints,
                          state.waypoints)

    e_ad = (params["attack_damage"][state.model]
            + params["ad_per_level"][state.model]
            * growth_sum(state.level, jnp))
    # A press of E has three outcomes (start, cancel at >= 1 s, nothing); a
    # cancel ends the spin through `end_e`, which starts the full rank
    # cooldown there and then. The natural 3 s expiry does the same from
    # `step_buffs`. See `cast_e`.
    buffs, cd, cast_e_now = cast_e(
        state.buffs, state.spell_cooldown, casting_e,
        state.spell_level[:, Slot.E], e_ad)
    buffs, cd, cast_q_now = cast_q(
        buffs, cd, casting_q, state.spell_level[:, Slot.Q])
    buffs, cd, cast_w_now = cast_w(
        buffs, cd, casting_w, state.spell_level[:, Slot.W])
    buffs, cd, cast_r_now = cast_r(
        buffs, cd, casting_r, state.spell_level[:, Slot.R], otgt)

    # `Spell.Cast` cancels an existing cancellable channel before it starts
    # the new cast.  Do this only for a spell that really became active; an
    # unavailable Q/W/E/R is a no-op and leaves Recall intact.
    ordinary_cast = cast_e_now | cast_q_now | cast_w_now | cast_r_now
    successful_cast = jnp.stack(
        [cast_q_now, cast_w_now, cast_e_now, cast_r_now], axis=1)[:2]
    observed_enemy_cast_ms = _record_observed_enemy_casts(state, successful_cast, vision)
    cancel_channel = ordinary_cast & (state.recall_channel_ms > 0)
    stop_for_recall = recall_stop
    # `UpdateMoveOrder(Stop)` drops TargetUnit only when the pre-stop path is
    # unfinished. LanerlControl calls StopMovement afterwards regardless, but
    # StopMovement itself does not touch the target; a stationary champion can
    # therefore retain its held target through the pill cast.
    stop_clears_target = stop_for_recall & (state.waypoint_key < state.n_waypoints)
    stop_waypoints = stop_for_recall[:, None, None]
    reset_path = state.waypoints.at[:, 0].set(jnp.stack([state.x, state.y], -1))

    clear_clicked_target = (jnp.zeros_like(moving) if orders.clear_target is None
                            else moving & per_unit(orders.clear_target, False))
    return state.replace(
        buffs=buffs, spell_cooldown=cd,
        observed_enemy_cast_ms=observed_enemy_cast_ms,
        # `GarenQ.OnActivate` calls `CancelAutoAttack(true)` before setting
        # `SkipNextAutoAttack()`.  R is likewise an ordinary non-instant spell
        # and `Spell.Cast` calls `AutoAttackSpell.CastCancelCheck` after it
        # becomes the owner's cast spell.  In either case a pending ordinary
        # swing is discarded and its cooldown reset; otherwise an already
        # started swing could deal damage during R's uncancellable cast lock.
        # The Q skip bit itself is carried by the Q buff (`skip_next`) and
        # consumed by `step_autoattack` at the next swing gate.
        aa_cooldown=jnp.where(cast_q_now | cast_r_now, 0.0, state.aa_cooldown),
        aa_windup=jnp.where(cast_q_now | cast_r_now, 0.0, state.aa_windup),
        is_attacking=jnp.where(cast_q_now | cast_r_now, False, state.is_attacking),
        waypoints=jnp.where(stop_waypoints, reset_path, waypoints),
        n_waypoints=jnp.where(stop_for_recall, jnp.int8(1),
                              jnp.where(moving, routed_n, state.n_waypoints)),
        waypoint_key=jnp.where(stop_for_recall, jnp.int8(1),
                               jnp.where(moving, jnp.int8(1), state.waypoint_key)),
        # A Move order does NOT clear the target -- `LanerlControl.cs:349-371`
        # never calls `SetTargetUnit`, and `UpdateMoveOrder(MoveTo, ...)`
        # itself only clears the target for OrderNone/Stop/PetHardStop
        # (`ObjAIBase.cs:1353-1360`). A held target survives a Move order and
        # is re-engaged the next tick by `step.py`'s "3b. RefreshWaypoints"
        # block (`ObjAIBase.RefreshWaypoints`, `:602-604`) for as long as it
        # stays alive and visible -- see the module docstring's "Sticky
        # targets" section.
        target=jnp.where(stop_clears_target | clear_clicked_target, jnp.int8(-1),
                         jnp.where(attacking, otgt, state.target)),
        move_order=jnp.where(stop_for_recall, jnp.int8(MoveOrder.STOP),
                             jnp.where(moving, jnp.where(kind == OrderKind.ATTACK_MOVE,
                                 jnp.int8(MoveOrder.ATTACK_MOVE), jnp.int8(MoveOrder.MOVE_TO)), state.move_order)),
        route_status=jnp.where(moving, routed_status, state.route_status),
        recall_windup_ms=jnp.where(recalling, jnp.asarray(500.0, state.x.dtype),
                                   state.recall_windup_ms),
        recall_channel_ms=jnp.where(cancel_channel, 0.0, state.recall_channel_ms),
        recall_damage_pending=jnp.where(cancel_channel, False,
                                        state.recall_damage_pending),
        r_cast_ms=jnp.where(cast_r_now,
                            jnp.asarray(R_CAST_TIME_S * 1000.0, state.x.dtype),
                            state.r_cast_ms),
    )
