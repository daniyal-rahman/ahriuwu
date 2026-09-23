"""The lane simulation state: one fixed-shape pytree, struct-of-arrays.

Shapes are fixed and dead units are masked
------------------------------------------
Nothing is ever allocated or removed at runtime.  A dead minion keeps its slot
with ``alive = False``; a wave spawn writes into a free slot.  That is what makes
the whole step function ``vmap``-able over thousands of environments and
``scan``-able over ticks, and it is why the caps below had to be *measured*
before this file could be written -- a cap that is too small truncates silently,
and a silent truncation looks exactly like a mechanic bug.

Caps, and where each number came from
-------------------------------------
All from a real 300 s bot-driven lane recording (see
``lab_notebook/2026-09-16_jax_rewrite.md``), with headroom because bronze-tier
bots at 13 CS are not an upper bound on a contested lane:

=================  =========  =======================================================
constant           measured   note
=================  =========  =======================================================
``N_MINIONS`` 40      max 28  p99 27, median 15
``N_CHAMPIONS`` 2     exactly 2
``N_TURRETS`` 24      exactly 24  every map turret exists as an object even under
                                  ``LANERL_TOPONLY``; only two ever act, and they
                                  cost nothing in a masked array
``MAX_WAYPOINTS`` 64  server SmoothPath worst 19 in the earlier 1800u corpus. The local
                                reverse-BFS router deliberately only removes
                                collinear cells: 110,890 random valid bounded
                                Map1 routes measured p99=24 and max=45. 64 keeps
                                headroom while overflow remains diagnostic.
``MAX_BUFFS`` 8       max 5
``N_MISSILES`` 24     max 16  from a 600 s **idle** top lane (``init_lane`` +
                                ``step_decision``, no orders, no wave-spawn
                                variance beyond the schedule itself) -- NOT the
                                300 s recording above, because missiles did not
                                exist when that one was taken and it carries no
                                missile counts. Every non-melee attacker fires
                                one in this slice -- caster and cannon minions
                                AND both outer turrets, not just the minions
                                (see ``sim/missiles.py``); measured again by
                                ``test_missile_overflow_stays_zero_over_a_long_run``
                                in ``sim/tests/test_missiles.py``, which checks
                                the cap is never actually reached rather than
                                trusting this number to stay true.
=================  =========  =======================================================

Why float32 and not float64
---------------------------
The plan originally said to run the parity suite in float64 "so a disagreement
is a logic disagreement".  That was backwards.  **The server computes in
float32** -- C# ``float`` throughout ``Vector2``, ``Stats`` and the A* priority --
so a float64 sim is not a more accurate version of the server, it is a
*different* simulation.  The same mistake cost real time in the pathfinder,
where double-precision A* priorities had to be replaced with ``_dist32``.

So the default is float32, matching the reference.  ``dtype`` is configurable
because float64 remains useful for one narrow purpose: establishing that a
disagreement is *not* caused by our own rounding, by checking it survives a
precision change.

Unit slots are laid out by kind, and the order is load-bearing
--------------------------------------------------------------
``[champions | minions | turrets]``, contiguous.  Target acquisition in the
server resolves ties by iteration order (``ObjAIBase.cs:1295-1320`` takes the
first strictly-closer unit; ``TurretAI`` notes that "the player to have been
added to the game first will always be targeted"), and ``argmin`` takes the
lowest index.  Those must be the same index, so slot order is a parity
surface, not an implementation detail.
"""
from __future__ import annotations

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax import struct

__all__ = [
    "N_CHAMPIONS", "N_MINIONS", "N_TURRETS", "N_UNITS",
    "MAX_WAYPOINTS", "MAX_BUFFS", "N_MISSILES",
    "CH_SLICE", "MI_SLICE", "TU_SLICE",
    "Kind", "Team", "MoveOrder", "TurretTier",
    "LaneState", "empty_state",
]

N_CHAMPIONS = 2
N_MINIONS = 40
N_TURRETS = 24
N_UNITS = N_CHAMPIONS + N_MINIONS + N_TURRETS      # 66

MAX_WAYPOINTS = 64
MAX_BUFFS = 8
N_MISSILES = 24

CH_SLICE = slice(0, N_CHAMPIONS)
MI_SLICE = slice(N_CHAMPIONS, N_CHAMPIONS + N_MINIONS)
TU_SLICE = slice(N_CHAMPIONS + N_MINIONS, N_UNITS)


class Kind:
    """Unit kind. 0 is "empty slot", so a zeroed state is an empty world."""
    NONE = 0
    CHAMPION = 1
    LANE_MINION = 2
    TURRET = 3


class TurretTier:
    """Which of Map1's five turret models a `Kind.TURRET` unit is.

    Mirrors `MinionType` for turrets: it is the "subtype" half of a profile key
    `(kind, subtype, team)` in `sim.profiles.PROFILES`, the same slot a lane
    minion's `MinionType` occupies. Before this existed every placed turret --
    outer, inner, inhibitor and nexus alike -- was built from the OUTER model's
    Content stats, because `profile_id(Kind.TURRET, ...)` only ever had one row
    per team. That is wrong on two schedules that both fire inside a 10-minute
    episode: `LevelScriptObjects.GetTurretType`
    (`Maps/Map1/LevelScriptObjects.cs:364-393`) assigns a different Content
    model per tier (AD/armour/regen all differ -- see `data.patch.TURRET_MODELS`),
    and `LevelScriptObjects.OnUpdate` ramps the non-outer tiers on a schedule
    that starts at 480 s (`:159-266`) -- INSIDE the episode -- while the outer
    tier's own ramp (already modelled, `combat.outer_turret_ramps`) stops
    mattering by 390 s.

    Order is arbitrary; values are never compared for magnitude, only equality
    and array-indexed lookup, exactly like `MinionType`.
    """
    OUTER = 0
    INNER = 1
    INHIBITOR = 2
    NEXUS = 3
    FOUNTAIN = 4


class Team:
    """Server team ids are 100/200/300; these are compact indices.

    The mapping lives in one place because the two numbering schemes crossing
    silently is exactly the class of bug the canonicalisation review caught.
    """
    BLUE = 0
    RED = 1
    NEUTRAL = 2

    SERVER_ID = (100, 200, 300)


class MoveOrder:
    """``GameServerCore.Enums.OrderType``, the subset the lane uses."""
    NONE = 0
    HOLD = 1
    MOVE_TO = 2
    ATTACK_TO = 3
    ATTACK_MOVE = 4
    STOP = 5
    CAST_SPELL = 6


@struct.dataclass
class LaneState:
    """One lane, one tick. Every field is a leading-axis-``N_UNITS`` array
    unless named otherwise, so ``jax.vmap`` over environments just works."""

    # ---- clock -----------------------------------------------------------
    #: game time in milliseconds. The server advances this by exactly 1000/60
    #: per tick under ``LANERL_FREERUN`` (``Game.cs:333``).
    t_ms: jax.Array
    tick: jax.Array

    # ---- identity --------------------------------------------------------
    kind: jax.Array            # (N,) int8, see Kind
    team: jax.Array            # (N,) int8, see Team
    alive: jax.Array           # (N,) bool
    #: row in :data:`lanerl_jax.sim.profiles.PROFILES` -- the (kind, minion
    #: type, **team**) stat row. Per-unit rather than per-slot because a minion
    #: slot is reused by whatever spawns into it, and per-team because blue and
    #: red minions genuinely differ (cannon range 300 vs 280, gold 35 vs 30).
    model: jax.Array           # (N,) int8
    #: A monotonic creation rank, assigned once at spawn and never reused or
    #: rewritten -- the JAX-side stand-in for `CollisionHandler._objects`'
    #: iteration order, which `GameObject.OnAdded`/`AddObject` fixes once, for
    #: good, at construction (`List<T>.Remove` shifts survivors down but never
    #: reorders them, `GameObject.cs:150-155`). Per-unit rather than per-slot
    #: for exactly the reason `model` is: a minion slot is recycled on death,
    #: so slot index tracks "whoever spawned into this slot most recently",
    #: not creation order, and turrets (created at map load, before ANY
    #: minion, and before the two champions -- `Game.Initialize` runs
    #: `Map.Init()`, which is what actually instantiates them via
    #: `LevelScriptObjects.CreateBuildings`, before its own
    #: `PlayerManager.AddPlayer` loop) sit in the LAST slice of our own
    #: `[champions | minions | turrets]` layout while the server creates them
    #: FIRST. `sim.collision.resolve_collisions` sorts on this field to
    #: reproduce the server's Gauss-Seidel collision order; see its module
    #: docstring for why that order -- and not a separate "quadtree" order --
    #: is the whole story. Assigned by `sim.init.init_lane` (turrets, then the
    #: two champions) and incremented by `sim.init.spawn_minion` via
    #: `next_spawn_seq` below; never touched anywhere else.
    spawn_seq: jax.Array       # (N,) int32
    #: ``GameObject.IsVisibleByTeam`` from the perspective of the team that is
    #: NOT this unit's own -- i.e. can this unit currently be seen (and hence
    #: targeted) by its enemies. Recomputed every tick in ``step.tick`` from
    #: :func:`lanerl_jax.obs.fog.visible_to_enemy` and stored here rather than
    #: recomputed by each consumer, mirroring the server's own cache: `Object
    #: Manager.Update` writes `IsVisibleByTeam` once per tick and everything
    #: downstream -- `ObjAIBase.UpdateTarget`, `LaneMinionAI`, `Spell` -- just
    #: reads the flag (`GameServerLib/Lanerl/LanerlFow.cs`'s "AT THE CACHE"
    #: comment). A single ``(N,)`` field suffices rather than a per-team pair
    #: or an ``(N, N)`` who-sees-whom matrix: a lane has exactly two live
    #: teams, an ally is unconditionally visible to itself, so "visible to
    #: unit u's opponent" already says everything any seeker on either side
    #: needs -- see the function's own docstring for the full argument. That
    #: also keeps the memory cost at 1x an ``(N,)`` leaf rather than 66x (this
    #: state is replicated across thousands of parallel envs, per the module
    #: docstring's caps table).
    visible_to_enemy: jax.Array  # (N,) bool

    # ---- position and movement ------------------------------------------
    x: jax.Array               # (N,)
    y: jax.Array               # (N,)
    #: Positions stored in CollisionHandler's dynamic quadtree. The server
    #: rebuilds these immediately after collision resolution, before units
    #: move, so they intentionally lag ``x``/``y`` by one movement phase.
    collision_x: jax.Array     # (N,)
    collision_y: jax.Array     # (N,)
    #: Whether this slot currently has a node in that quadtree. Newly spawned
    #: objects are inserted immediately by ``GameObject.OnAdded``.
    collision_present: jax.Array  # (N,) bool
    waypoints: jax.Array       # (N, MAX_WAYPOINTS, 2)
    #: ``CurrentWaypointKey``. 1 after ``SetWaypoints``; never 0.
    waypoint_key: jax.Array    # (N,) int8
    n_waypoints: jax.Array     # (N,) int8
    #: ``LaneMinionAI.currentWaypointIndex`` into its immutable
    #: ``PathingWaypoints`` list. This is deliberately distinct from
    #: ``waypoint_key``: the latter indexes a transient movement route and is
    #: overwritten while a minion chases a target; the AI resumes its lane
    #: route at this persistent index after combat.
    lane_waypoint_key: jax.Array  # (N,) int8
    move_order: jax.Array      # (N,) int8, see MoveOrder
    #: Diagnostic for the most recently accepted champion Move route. Zero is
    #: ``LocalRouteStatus.READY``; nonzero values make table coverage,
    #: no-route and fixed-shape overflows visible to rollouts instead of
    #: silently presenting a raw two-point fallback as exact pathing. Minions
    #: do not consume the local player-click table and retain zero here.
    route_status: jax.Array     # (N,) int8

    # ---- combat ----------------------------------------------------------
    hp: jax.Array              # (N,)
    max_hp: jax.Array          # (N,)
    #: unit-index of the current target, or -1. Not a NetId: slot order is the
    #: identity here, and it is chosen to match the server's iteration order.
    target: jax.Array          # (N,) int8
    is_attacking: jax.Array    # (N,) bool
    has_auto_attacked: jax.Array   # (N,) bool
    #: ``_autoAttackCurrentCooldown``, in SECONDS (the server counts down by
    #: ``diff / 1000`` in ``ObjAIBase.Update``).
    aa_cooldown: jax.Array     # (N,)
    #: remaining wind-up, seconds. Zero when not winding up.
    aa_windup: jax.Array       # (N,)
    #: Remaining silence duration in milliseconds. A silenced unit may move
    #: and autoattack but cannot issue Q/W/E/R casts.
    silenced_ms: jax.Array     # (N,)
    #: Garen R's non-instant engine cast. This is held on the caster (unlike
    #: the target-side pending-hit mailbox) because `_castingSpell` locks its
    #: movement, attacks and later casts for the 0.435 s windup.
    r_cast_ms: jax.Array       # (N,), 0 when no R windup is active
    #: Recall has the ordinary 0.5 s spell windup before its 8 s channel.
    #: The windup is deliberately separate: the server only exposes the
    #: latter as ``Champion.ChannelSpell`` / wire ``rc``.
    recall_windup_ms: jax.Array  # (N,), 0 when not winding up
    #: Remaining blue-pill channel time.  A positive value is the exact
    #: ``ChannelSpell != null`` state used by LanerlControl's observation.
    recall_channel_ms: jax.Array  # (N,), 0 when not channeling
    #: ``Buffs/Global/Recall.OnTakeDamage`` sets ``willRemove``; its own
    #: ``OnUpdate`` cancels on the following tick, before Spell.Update.  Keep
    #: that one-tick latch rather than retroactively cancelling the hit tick.
    recall_damage_pending: jax.Array  # (N,) bool

    # ---- progression -----------------------------------------------------
    #: ``Champion.RespawnTimer``, ms. -1 when alive. Champions only.
    respawn_ms: jax.Array      # (N,)
    #: where a champion returns to. Constant per unit, carried in state so the
    #: step function needs no extra table.
    spawn_x: jax.Array         # (N,)
    spawn_y: jax.Array         # (N,)

    level: jax.Array           # (N,) int8
    xp: jax.Array              # (N,)
    gold: jax.Array            # (N,)
    #: ``Champion._goldTimer``, ms. Counts down; on expiry a champion receives
    #: one ambient gold tick. See :mod:`lanerl_jax.sim.rewards`.
    gold_timer: jax.Array      # (N,)
    #: `AttackableUnit._statUpdateTimer` -- the 500 ms accumulator base regen
    #: is applied on. Per unit, because it is per unit on the server.
    stat_timer: jax.Array      # (N,)
    #: `GarenPassiveHeal.healingTimer` -- the passive's own ~1 s accumulator,
    #: independent of the stat clock above.
    heal_timer: jax.Array      # (N,)
    #: ms since this unit last took damage that COUNTS as combat. Ordinary
    #: minion autoattacks deliberately do not reset it -- see `sim/regen.py`.
    #: Starts high so a fresh champion is already out of combat.
    ms_since_damaged: jax.Array  # (N,)
    cs: jax.Array              # (N,) int16
    deaths: jax.Array          # (N,) int16
    #: ``Champion.KillSpree``/``DeathSpree`` (``Champion.cs:31-32``) -- a
    #: champion's OWN consecutive-kills/-deaths counters, read as the VICTIM's
    #: (not the killer's) state inside `Champion.Die`'s gold formula: a fed
    #: victim (high `kill_spree`) is worth a "shutdown" bonus, a feeding one
    #: (`death_spree>=1` with `kill_spree==0`) is worth less. See
    #: `sim.rewards.champion_kill_rewards`. Zero for non-champions, always.
    kill_spree: jax.Array      # (N,) int32
    death_spree: jax.Array     # (N,) int32
    #: ``Champion.GoldFromMinions`` (`Champion.cs:29`) -- gold earned from
    #: minions while on a death spree; crossing 1000 knocks one stack off
    #: `death_spree` (`Champion.OnKill`, `Champion.cs:379-388`). Reset to 0
    #: the instant its owner lands a champion kill (`Champion.cs:490`).
    gold_from_minions: jax.Array  # (N,)
    #: ``ChampionStats.Kills`` -- not read by any gold/XP formula (unlike
    #: `kill_spree`/`death_spree` above), carried only for observability,
    #: symmetric with `deaths` above.
    kills: jax.Array           # (N,) int16
    #: ``Champion._championHitFlagTimer`` (`Champion.cs:21,267-273`) -- ms
    #: remaining since this champion was last hit by ANY source (reset to
    #: 15000 on every `TakeDamage`, decremented every tick, floored at 0; NOT
    #: the same gate as `ms_since_damaged`, which exempts ordinary
    #: melee/caster minion damage for Garen's passive -- this one has no such
    #: exemption, matching `Champion.TakeDamage`'s override applying
    #: unconditionally). Zero for non-champions, always.
    hit_flag_ms: jax.Array     # (N,)
    #: ``Champion._playerHitId`` (`Champion.cs:27,573-574`), translated from a
    #: NetId to this state's own unit index -- whoever last hit this champion,
    #: of any kind (champion/minion/turret). -1 for none yet. See
    #: `sim.rewards.champion_kill_rewards`'s `cKiller` fallback
    #: (`Champion.cs:404-408`).
    hit_flag_by: jax.Array     # (N,) int8
    #: ``IMapScript.HasFirstBloodHappened`` (`Map1/LevelScript.cs:19`) -- a
    #: MAP-level flag, not per-unit, exactly like `next_spawn_ms` below.
    first_blood_done: jax.Array   # () bool

    # ---- buffs -----------------------------------------------------------
    buff_id: jax.Array         # (N, MAX_BUFFS) int8, 0 = empty
    buff_elapsed: jax.Array    # (N, MAX_BUFFS)
    buff_duration: jax.Array   # (N, MAX_BUFFS)
    #: per-buff scalar the script needs. Judgment snapshots its damage at cast
    #: (the AD ratio is NOT recomputed per tick), so it lives here.
    buff_power: jax.Array      # (N, MAX_BUFFS)

    # ---- spells ----------------------------------------------------------
    #: rank per slot, 0 = unlearned. ``Spell.Cast`` does **not** check the
    #: level, so casting an unlearned spell is not a no-op on the server -- it
    #: grants the effect anyway. The action mask is what must forbid it.
    spell_level: jax.Array     # (N, 4) int8
    #: remaining cooldown, SECONDS
    spell_cooldown: jax.Array  # (N, 4)
    #: Per observing champion (blue slot 0, red slot 1) and Q/W/E/R: elapsed
    #: milliseconds since that observer *witnessed* an opposing cast. ``-1``
    #: means never witnessed, deliberately distinct from a long elapsed timer
    #: in state even though both normalize to 1.0 for the policy.  This is
    #: observation memory, not privileged enemy cooldown state.
    observed_enemy_cast_ms: jax.Array  # (N_CHAMPIONS, 4)

    # ---- minion AI -------------------------------------------------------
    #: ``minionActionTimer``; the AI re-evaluates at 250 ms
    #: (``LaneMinionAI.OnUpdate``). Starts at 250 so the first tick evaluates.
    ai_timer: jax.Array        # (N,)
    #: ``targetUnitPriority`` -- a ``ClassifyUnit`` value, 1..14, lower is
    #: higher priority. 14 (DEFAULT) means "no committed target".
    target_priority: jax.Array     # (N,) int8
    #: ``hadTarget`` -- `LaneMinionAI`'s one-tick latch, NOT ``target >= 0``.
    #: It is set true only on the tick AFTER an acquisition (`LaneMinionAI.cs:44`)
    #: and cleared when `TargetJustDied` fires (`:48`), so it disagrees with
    #: ``target >= 0`` on exactly the acquisition tick and on the tick after a
    #: give-up null-out. Measured on the canonical corpus: **556 of 395,486
    #: scored LaneMinion unit-ticks (0.1406%)**, 494 of them ``target`` set
    #: while the latch is clear. Reconstructing it as ``target >= 0`` made the
    #: port fire `TargetJustDied` on those 494 ticks where the server does not.
    #: The server publishes it as ``aihad=`` and `trace.py` has always parsed
    #: it; it simply had nowhere to be stored.
    had_target: jax.Array          # (N,) bool
    #: ``temporaryIgnored``: per (unit, other) the local time until which the
    #: other is ignored. Dense because a dict is not a fixed shape.
    ignore_until: jax.Array    # (N, N)
    #: ``unitsAttackingAllies``: the call-for-help priority another unit carries
    #: for this one, or 14 for none.
    help_priority: jax.Array   # (N, N) int8
    #: ``localTime`` -- each minion AI's own clock, not the game clock.
    ai_local_time: jax.Array   # (N,)
    #: ``timeSinceLastAttack``, ms. Its own clock: it resets whenever the unit
    #: is attacking **or has no target**, and drives the 4 s give-up rule that
    #: makes a minion abandon something it cannot reach. Carried explicitly
    #: because folding it into another timer silently disables that rule.
    time_since_attack: jax.Array   # (N,)

    # ---- missiles --------------------------------------------------------
    missile_alive: jax.Array   # (M,) bool
    missile_x: jax.Array       # (M,)
    missile_y: jax.Array       # (M,)
    missile_tx: jax.Array      # (M,)  target unit index, -1 if none
    missile_source: jax.Array  # (M,) int8
    missile_damage: jax.Array  # (M,)
    missile_speed: jax.Array   # (M,)

    # ---- wave spawner ----------------------------------------------------
    #: ``NextSpawnTime`` and ``_minionNumber`` / ``_cannonMinionCount`` from
    #: ``Maps/Map1/LevelScript.cs``. Scalars, not per-unit.
    next_spawn_ms: jax.Array
    minion_number: jax.Array
    cannon_count: jax.Array
    #: The next value `sim.init.spawn_minion` will assign to a newly created
    #: minion's `spawn_seq`. `init_lane` seeds the first `N_TURRETS +
    #: N_CHAMPIONS` ranks itself (map load, then the two players -- see
    #: `spawn_seq`'s own docstring), so this starts there, not at 0.
    next_spawn_seq: jax.Array

    #: Both Map1 fountains start their independent 1 s heal timers at zero and
    #: receive the same ``diff``, so one scalar represents both exactly.
    fountain_heal_ms: jax.Array

    # ---- rng -------------------------------------------------------------
    key: jax.Array

    # ------------------------------------------------------------------ --
    @property
    def n_units(self) -> int:
        return self.kind.shape[-1]

    def champions(self) -> Tuple[int, int]:
        return (0, 1)

    def is_kind(self, k: int) -> jax.Array:
        return self.kind == k


def empty_state(dtype=jnp.float32, seed: int = 0,
                n_units: int = N_UNITS,
                n_missiles: int = N_MISSILES) -> LaneState:
    """A zeroed world: no live units, clock at zero.

    ``Kind.NONE == 0`` and ``alive == False`` by construction, so "zeroed" and
    "empty" are the same thing -- there is no half-initialised state that looks
    populated.
    """
    z = lambda *s: jnp.zeros(s, dtype=dtype)            # noqa: E731
    zi = lambda *s, t=jnp.int8: jnp.zeros(s, dtype=t)   # noqa: E731
    return LaneState(
        t_ms=jnp.asarray(0.0, dtype=dtype),
        tick=jnp.asarray(0, dtype=jnp.int32),
        kind=zi(n_units),
        team=jnp.full((n_units,), Team.NEUTRAL, dtype=jnp.int8),
        alive=jnp.zeros((n_units,), dtype=bool),
        model=zi(n_units),
        # 0 is a harmless placeholder here: every slot with `spawn_seq == 0`
        # also has `kind == Kind.NONE` / `alive == False` until `init_lane`
        # and `spawn_minion` give it a real one, and collision only ever reads
        # this field through the `alive`/`kind` obstacle mask.
        spawn_seq=jnp.zeros((n_units,), dtype=jnp.int32),
        # An empty world has nothing to see; `tick` recomputes this fresh from
        # real positions before it is ever read this same first tick, so the
        # initial value only matters for a state that is inspected without
        # ever being stepped.
        visible_to_enemy=jnp.zeros((n_units,), dtype=bool),
        x=z(n_units), y=z(n_units),
        collision_x=z(n_units), collision_y=z(n_units),
        collision_present=jnp.zeros((n_units,), dtype=bool),
        waypoints=z(n_units, MAX_WAYPOINTS, 2),
        waypoint_key=jnp.ones((n_units,), dtype=jnp.int8),
        lane_waypoint_key=zi(n_units),
        n_waypoints=zi(n_units),
        move_order=jnp.full((n_units,), MoveOrder.NONE, dtype=jnp.int8),
        route_status=zi(n_units),
        hp=z(n_units), max_hp=z(n_units),
        target=jnp.full((n_units,), -1, dtype=jnp.int8),
        is_attacking=jnp.zeros((n_units,), dtype=bool),
        has_auto_attacked=jnp.zeros((n_units,), dtype=bool),
        aa_cooldown=z(n_units), aa_windup=z(n_units), silenced_ms=z(n_units),
        r_cast_ms=z(n_units),
        recall_windup_ms=z(n_units), recall_channel_ms=z(n_units),
        recall_damage_pending=jnp.zeros((n_units,), dtype=bool),
        respawn_ms=jnp.full((n_units,), -1.0, dtype=dtype),
        spawn_x=z(n_units), spawn_y=z(n_units),
        level=jnp.ones((n_units,), dtype=jnp.int8),
        xp=z(n_units), gold=z(n_units), gold_timer=z(n_units),
        stat_timer=z(n_units), heal_timer=z(n_units),
        ms_since_damaged=jnp.full((n_units,), 1e6, dtype),
        cs=zi(n_units, t=jnp.int16), deaths=zi(n_units, t=jnp.int16),
        kill_spree=zi(n_units, t=jnp.int32), death_spree=zi(n_units, t=jnp.int32),
        gold_from_minions=z(n_units), kills=zi(n_units, t=jnp.int16),
        hit_flag_ms=z(n_units),
        hit_flag_by=jnp.full((n_units,), -1, dtype=jnp.int8),
        first_blood_done=jnp.asarray(False),
        buff_id=zi(n_units, MAX_BUFFS),
        buff_elapsed=z(n_units, MAX_BUFFS),
        buff_duration=z(n_units, MAX_BUFFS),
        buff_power=z(n_units, MAX_BUFFS),
        spell_level=zi(n_units, 4),
        spell_cooldown=z(n_units, 4),
        observed_enemy_cast_ms=jnp.full((N_CHAMPIONS, 4), -1.0, dtype=dtype),
        # 250 so the first tick re-evaluates, as `minionActionTimer = 250f` does
        ai_timer=jnp.full((n_units,), 250.0, dtype=dtype),
        target_priority=jnp.full((n_units,), 14, dtype=jnp.int8),
        had_target=jnp.zeros((n_units,), dtype=bool),
        ignore_until=z(n_units, n_units),
        help_priority=jnp.full((n_units, n_units), 14, dtype=jnp.int8),
        ai_local_time=z(n_units),
        time_since_attack=z(n_units),
        missile_alive=jnp.zeros((n_missiles,), dtype=bool),
        missile_x=z(n_missiles), missile_y=z(n_missiles),
        missile_tx=jnp.full((n_missiles,), -1, dtype=jnp.int8),
        missile_source=jnp.full((n_missiles,), -1, dtype=jnp.int8),
        missile_damage=z(n_missiles), missile_speed=z(n_missiles),
        next_spawn_ms=jnp.asarray(0.0, dtype=dtype),
        minion_number=jnp.asarray(0, dtype=jnp.int32),
        cannon_count=jnp.asarray(0, dtype=jnp.int32),
        next_spawn_seq=jnp.asarray(0, dtype=jnp.int32),
        fountain_heal_ms=jnp.asarray(0.0, dtype=dtype),
        key=jax.random.key(seed),
    )
