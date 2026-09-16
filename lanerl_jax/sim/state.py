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
``MAX_WAYPOINTS`` 24  worst 19  from the ported A* at ``SCREEN_RADIUS`` (1800u)
                                click distance -- NOT from the state dump, whose
                                "max 5" was bot-driven short hops
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

MAX_WAYPOINTS = 24
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
    waypoints: jax.Array       # (N, MAX_WAYPOINTS, 2)
    #: ``CurrentWaypointKey``. 1 after ``SetWaypoints``; never 0.
    waypoint_key: jax.Array    # (N,) int8
    n_waypoints: jax.Array     # (N,) int8
    move_order: jax.Array      # (N,) int8, see MoveOrder

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

    # ---- minion AI -------------------------------------------------------
    #: ``minionActionTimer``; the AI re-evaluates at 250 ms
    #: (``LaneMinionAI.OnUpdate``). Starts at 250 so the first tick evaluates.
    ai_timer: jax.Array        # (N,)
    #: ``targetUnitPriority`` -- a ``ClassifyUnit`` value, 1..14, lower is
    #: higher priority. 14 (DEFAULT) means "no committed target".
    target_priority: jax.Array     # (N,) int8
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
        # An empty world has nothing to see; `tick` recomputes this fresh from
        # real positions before it is ever read this same first tick, so the
        # initial value only matters for a state that is inspected without
        # ever being stepped.
        visible_to_enemy=jnp.zeros((n_units,), dtype=bool),
        x=z(n_units), y=z(n_units),
        waypoints=z(n_units, MAX_WAYPOINTS, 2),
        waypoint_key=jnp.ones((n_units,), dtype=jnp.int8),
        n_waypoints=zi(n_units),
        move_order=jnp.full((n_units,), MoveOrder.NONE, dtype=jnp.int8),
        hp=z(n_units), max_hp=z(n_units),
        target=jnp.full((n_units,), -1, dtype=jnp.int8),
        is_attacking=jnp.zeros((n_units,), dtype=bool),
        has_auto_attacked=jnp.zeros((n_units,), dtype=bool),
        aa_cooldown=z(n_units), aa_windup=z(n_units),
        respawn_ms=jnp.full((n_units,), -1.0, dtype=dtype),
        spawn_x=z(n_units), spawn_y=z(n_units),
        level=jnp.ones((n_units,), dtype=jnp.int8),
        xp=z(n_units), gold=z(n_units), gold_timer=z(n_units),
        stat_timer=z(n_units), heal_timer=z(n_units),
        ms_since_damaged=jnp.full((n_units,), 1e6, dtype),
        cs=zi(n_units, t=jnp.int16), deaths=zi(n_units, t=jnp.int16),
        buff_id=zi(n_units, MAX_BUFFS),
        buff_elapsed=z(n_units, MAX_BUFFS),
        buff_duration=z(n_units, MAX_BUFFS),
        buff_power=z(n_units, MAX_BUFFS),
        spell_level=zi(n_units, 4),
        spell_cooldown=z(n_units, 4),
        # 250 so the first tick re-evaluates, as `minionActionTimer = 250f` does
        ai_timer=jnp.full((n_units,), 250.0, dtype=dtype),
        target_priority=jnp.full((n_units,), 14, dtype=jnp.int8),
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
        key=jax.random.key(seed),
    )
