"""One tick of the lane, in the server's order.

Order is a parity surface
-------------------------
``Game.Update`` -> ``ObjectManager.Update`` -> every object's ``Update(diff)``,
and ``ObjAIBase.Update`` calls ``base.Update`` (i.e.
``AttackableUnit.Update``) **first**.  So within a single tick a unit does::

    UpdateBuffs(diff)                       # AttackableUnit.Update
    Stats.Update every 500 ms               #   (hp/mana regen)
    Move(diff)                              #   <-- MOVEMENT COMES FIRST
    CharScript.OnUpdate(diff)               # ObjAIBase.Update
    AIScript.OnUpdate(diff)                 #   <-- the minion controller
    spells.Update(diff)
    UpdateTarget()                          #   <-- acquisition + the swing gate
    _autoAttackCurrentCooldown -= diff/1000 #   <-- decremented LAST

Two consequences worth stating, because both are easy to get backwards and
neither is visible in a learning curve:

**Targeting sees post-movement positions.** A minion that walks into range this
tick can be acquired this tick, not next. Ordering movement after targeting
delays every acquisition by one tick, which is 5.75 units of travel.

**The cooldown is decremented after the swing gate**, so a swing started this
tick is immediately one tick closer to the next. Both orderings happen to give
the same 97-tick period for a 1.6 s cooldown (the gate at tick T+k sees
``1.6 - k*step`` either way), but that is a coincidence of this particular
arrangement, not a licence to reorder.

Scope
-----
Movement, minion AI, target acquisition and the auto-attack clock, over the
fixed-shape :class:`~lanerl_jax.sim.state.LaneState`. Not yet wired in: buff
ticking, hp/mana regen, spell casts, missiles, wave spawning into free slots,
gold/XP on death, and the champion action decode. Each of those has its own
module or is still to come; this is the spine they hang on.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .autoattack import step_autoattack
from .combat import (
    TURRET_AD_PER_RAMP,
    TURRET_ARMOR_PER_RAMP,
    TURRET_DAMAGE_VS_MINION,
    growth_sum,
    other_turret_ramps,
    outer_turret_ramps,
)
from .collision import resolve_collisions
from ..obs.fog import visible_to_enemy as _visible_to_enemy
from .init import MINION_SPAWN, spawn_minion
from .missiles import step_missiles
from .profiles import PROFILES
from .spells import RANKS_BY_LEVEL, BuffId, Slot, step_buffs
from .waves_jax import step_waves_jax
from .minion_ai import step_minion_ai
from .movement_jax import TICK_MS, step_move_units
from .regen import step_regen
from .rewards import ambient_gold, death_rewards, level_for_xp
from .state import Kind, LaneState, MoveOrder, Team, TurretTier
from .targeting import (MinionType, base_priority, call_for_help_map,
                        nearest_enemy, turret_acquire)

__all__ = ["UnitParams", "tick", "step_decision"]


def _attack_damage_against(attack_damage, attacker_kind, target_kind,
                           model=None, t_ms=None):
    """Raw attack damage, with the attacker/target modifiers the scripts apply.

    Two exist in this slice. One is a property of the *pair*: every lane
    turret's basic-attack script multiplies by 0.7 when the target is a
    Minion, before mitigation -- a turret shooting a champion does full damage,
    so it cannot live in the profile table. The other is a property of the
    attacker's own tier and the game clock: the per-tier AD ramp below.
    """
    vs_minion = target_kind == Kind.LANE_MINION
    from_turret = attacker_kind == Kind.TURRET
    ad = jnp.where(from_turret & vs_minion,
                   attack_damage * TURRET_DAMAGE_VS_MINION, attack_damage)
    if t_ms is not None:
        # `LevelScriptObjects.OnUpdate` runs TWO independent ramps
        # (`combat.py`'s module docstring), selected by TIER, not by "is this a
        # turret": the outer tier ramps +4 AD every 60 s from t=30 s (152 at
        # the start, 180 from 390 s on, capped at 7 applications); every other
        # non-fountain tier ramps the same +4 AD (plus armour, applied
        # separately below) every 60 s from t=480 s, capped at 30. Dispatching
        # on `model` rather than `kind` is what makes this a per-tier lookup
        # instead of the old "every turret gets the outer schedule".
        is_outer = _OUTER_TURRET_ROW[model]
        is_other = _OTHER_TURRET_ROW[model]
        ramp = jnp.where(
            is_outer, TURRET_AD_PER_RAMP * outer_turret_ramps(t_ms, jnp),
            jnp.where(is_other, TURRET_AD_PER_RAMP * other_turret_ramps(t_ms, jnp),
                     jnp.zeros_like(ad)))
        ad = ad + ramp
    return ad


class UnitParams(dict):
    """Per-unit constants baked from the patch table.

    A plain dict of ``(N,)`` arrays rather than a dataclass so it can be closed
    over by ``jit`` without becoming part of the traced state: these never
    change within an episode.
    """


#: Module-level constants, NOT built inside `tick`.
#:
#: Every import in this module is top-level, deliberately. A module whose first
#: import happens **during tracing** has its module-level `jnp` arrays created
#: as *tracers* bound to that trace; they are then cached at module level and
#: reused by the next one, which raises UnexpectedTracerError far from the
#: cause. That is exactly what a lazy `from .waves_jax import ...` inside this
#: function did -- `waves_jax.REGULAR` leaked out of a scan as `int8[9]`.
_MINION_TYPE_TABLE = jnp.asarray([p[1] for p in PROFILES], jnp.int8)
_RANK_TABLE = jnp.asarray(RANKS_BY_LEVEL, jnp.int8)
_WAVE_ROW_BLUE = jnp.asarray(
    [PROFILES.index((Kind.LANE_MINION, m, Team.BLUE)) for m in range(4)], jnp.int8)
_WAVE_ROW_RED = jnp.asarray(
    [PROFILES.index((Kind.LANE_MINION, m, Team.RED)) for m in range(4)], jnp.int8)
#: Which PROFILES row is which turret tier, so the AD/armour ramp can be
#: selected by `state.model` alone rather than by re-deriving the tier from
#: geometry. FOUNTAIN rows are true in neither -- `UpdateTowerStats` excludes
#: it by name and `UpdateOuterTurretStats` never looks it up either
#: (`combat.py`'s `OTHER_TURRET_RAMP_MAX` docstring) -- so a fountain's ramp
#: bonus is 0 from both `jnp.where` branches below, which is the correct
#: "never ramps", not an approximation of it.
_OUTER_TURRET_ROW = jnp.asarray(
    [k == Kind.TURRET and sub == TurretTier.OUTER for k, sub, _ in PROFILES], bool)
_OTHER_TURRET_ROW = jnp.asarray(
    [k == Kind.TURRET and sub in (TurretTier.INNER, TurretTier.INHIBITOR,
                                  TurretTier.NEXUS)
     for k, sub, _ in PROFILES], bool)


#: profile row -> MinionType for a lane minion, or TurretTier for a turret
#: (see `sim.state.TurretTier`) -- `_minion_type_of` is reused for both
#: because every consumer (`base_priority`, `turret_acquire`) dispatches on
#: `kind` FIRST and only trusts this value where `kind == LANE_MINION`, so a
#: turret's tier passing through here as if it were a `MinionType` is inert:
#: it is masked away before it can be read as one. Kept as one lookup rather
#: than two so a turret's tier does not need its own gather in the tick.
def _minion_type_of(state: LaneState) -> jax.Array:
    return _MINION_TYPE_TABLE[state.model]


def _can_move(move_order: jax.Array, alive: jax.Array) -> jax.Array:
    """``ObjAIBase.Move`` refuses under CastSpell / OrderNone / Stop / Taunt.

    ``Hold`` is in the blocked set too: ``RefreshWaypoints`` sets it precisely
    to stop a unit that has walked into attack range, and a unit that keeps
    walking through Hold never stops to fight.
    """
    blocked = (
        (move_order == MoveOrder.CAST_SPELL)
        | (move_order == MoveOrder.NONE)
        | (move_order == MoveOrder.STOP)
        | (move_order == MoveOrder.HOLD)
    )
    return alive & ~blocked


def tick(state: LaneState, params: UnitParams,
         delta_ms: float = TICK_MS, lane_path=None,
         minion_hp=None, enable_call_for_help: bool = False) -> LaneState:
    """Advance one 16.667 ms server tick.

    ``lane_path`` is ``(W, 2)`` -- ``MinionPaths[LANE_L]``, walked forward by
    blue and reversed by red, exactly as ``SetUpLaneMinion`` does
    (``waypoint.Reverse()`` for ``TEAM_PURPLE``). Pass ``None`` to run without
    wave spawning, which is what the unit tests want.

    ``enable_call_for_help`` defaults to ``False``, matching every test and
    every measurement in this tree before ``docs/CALL_FOR_HELP_SWITCH_RATE.md``
    -- flipping it does not change existing behaviour unless a caller opts in.
    See section 5b below and that doc for the measured switch-rate comparison
    and why this stays a toggle rather than either a permanent no-op or a
    permanent wire.
    """
    n = state.kind.shape[-1]
    dtype = state.x.dtype
    # Every stat is a gather through the unit's profile row: a minion slot is
    # reused by whatever spawns into it, so stats cannot be baked per slot.
    P = lambda k: params[k][state.model]          # noqa: E731
    # An INNER/INHIBITOR/NEXUS turret's armour is not its Content value for
    # most of a game either -- see `_attack_damage_against` for the AD half of
    # the same pair of schedules. Computed once, up front, because armour is
    # read in three places below (buffs, the autoattack mitigation target, and
    # missile mitigation) and all three must see the same ramped value.
    base_armor = P("armor")
    armor_now = base_armor + jnp.where(
        _OTHER_TURRET_ROW[state.model],
        TURRET_ARMOR_PER_RAMP * other_turret_ramps(state.t_ms, jnp),
        jnp.zeros_like(base_armor))

    # ---- 0. collision push-apart (Map.Update, FIRST thing in the tick) ----
    # `Game.Update` runs `Map.Update` (Game.cs:481) before `ObjectManager.
    # Update` (:483), and `CollisionHandler.Update()` is the first call inside
    # it (MapScriptHandler.cs:96-100) -- ahead of pathing, ahead of wave
    # spawning, ahead of every unit's own Update. It is the real push-apart: it
    # calls `obj.OnCollision(obj2)` directly (CollisionHandler.cs:121-155), not
    # a quadtree rebuild.
    #
    # So the server separates the positions units came to REST at last tick and
    # only then moves them. This ran after movement here, which meant every
    # phase downstream saw a position with this tick's push already folded in
    # while the server's is still a tick behind on collision.
    #
    # Measured, before this was fixed: over 14,401 one-step injected
    # predictions, minion position was exact on 83.1% of ticks and the
    # disagreements were **98.6% one-sided** -- the sim consistently ended up
    # further along its own heading than the server. Position is fully
    # injectable ground truth, so that bias is not an artifact of the harness.
    #
    # `ghosted` is read from the INCOMING buff state, not from this tick's
    # `step_buffs`, and that is deliberate: `UpdateBuffs` is the first thing
    # inside a unit's own `Update`, which happens after `Map.Update` has
    # already run. Collision therefore sees the Ghosted flag as it stood at the
    # end of last tick. Garen's E sets `StatusFlags.Ghosted`, so a spinning
    # Garen passes through units instead of being shoved out of the wave.
    pre_ghosted = (state.buff_id[:, Slot.E] == BuffId.GAREN_E) & state.alive
    cx, cy = resolve_collisions(state.x, state.y, state.kind, state.alive,
                                P("pathfinding_radius"), ghosted=pre_ghosted)
    state = state.replace(x=cx, y=cy)

    # ---- 1. wave spawning (MapScript.Update, still inside Map.Update) ------
    if lane_path is not None:
        mtype, next_spawn, m_no, c_no = step_waves_jax(
            state.t_ms, state.next_spawn_ms, state.minion_number,
            state.cannon_count)
        mi = jnp.clip(mtype, 0, 3)
        hp_b = params["max_hp"][_WAVE_ROW_BLUE[mi]]
        hp_r = params["max_hp"][_WAVE_ROW_RED[mi]]
        # Spawn at the MEASURED barracks, not at the path's end vertex --
        # they differ by 446 units on the red side. See `spawn_minion`.
        state = spawn_minion(state, Team.BLUE, _WAVE_ROW_BLUE[mi], hp_b,
                             lane_path, enabled=mtype >= 0,
                             spawn_xy=MINION_SPAWN[Team.BLUE])
        state = spawn_minion(state, Team.RED, _WAVE_ROW_RED[mi], hp_r,
                             lane_path[::-1], enabled=mtype >= 0,
                             spawn_xy=MINION_SPAWN[Team.RED])
        state = state.replace(next_spawn_ms=next_spawn, minion_number=m_no,
                              cannon_count=c_no)

    # ---- 2a. buffs (ObjectManager.Update -> AttackableUnit.UpdateBuffs) ----
    # First thing inside the unit's own Update, and therefore AFTER the
    # collision pass above and BEFORE movement below. It reads post-collision,
    # pre-move positions. That distinction only became real once collision
    # moved to the front of the tick: while collision ran last, pre-move and
    # post-collision were the same positions and this was harmless.
    bs = step_buffs(
        buff_id=state.buff_id, buff_elapsed=state.buff_elapsed,
        buff_duration=state.buff_duration, buff_power=state.buff_power,
        spell_cooldown=state.spell_cooldown, spell_level=state.spell_level,
        x=state.x, y=state.y, kind=state.kind, team=state.team,
        alive=state.alive, armor=armor_now,
        magic_resist=P("magic_resist"), delta_ms=delta_ms)

    # ---- Garen's W: the two resist/damage hooks spells.py asks for ---------
    # W's PASSIVE is a permanent +20% Armor and +20% MagicResist, granted once
    # on first rank-up of W (`W.cs:26-46` registers an OnLevelUpSpell listener
    # at spell construction, so it does not require ever pressing W). W's
    # ACTIVE multiplies all incoming post-mitigation damage by 0.7 while the
    # window is open (`GarenW.cs:47-55`).
    #
    # Both are identity when Garen has never levelled or cast W, so this
    # changes nothing in a lane where W is unused.
    #
    # The base is `armor_now`, NOT `P("armor")`: a non-outer turret's armour
    # already grows +1 every 60 s from 480 s (`other_turret_ramps`), and W's
    # bonus is a PERCENTAGE of the current total, so it has to compose with the
    # ramp rather than replace it. Getting this backwards would have silently
    # frozen turret armour at its level-1 value for anyone carrying the buff.
    armor_eff = armor_now * (1.0 + bs.armor_pct_bonus)
    magic_resist_eff = P("magic_resist") * (1.0 + bs.mr_pct_bonus)

    # ---- 2a2. Stats.Update: HP regen (AttackableUnit.Update, after buffs) --
    # Right after UpdateBuffs and before Move, on its own 500 ms accumulator.
    # Not modelling this is why our champion died 7 times in an oracle-driven
    # 600 s episode where the server's died 0 -- see `sim/regen.py`.
    rg = step_regen(
        hp=state.hp, max_hp=state.max_hp, alive=state.alive, kind=state.kind,
        level=state.level, hp_regen=P("hp_regen"),
        stat_timer=state.stat_timer, heal_timer=state.heal_timer,
        ms_since_damaged=state.ms_since_damaged, delta_ms=delta_ms)
    state = state.replace(hp=rg.hp, stat_timer=rg.stat_timer,
                          heal_timer=rg.heal_timer)

    # ---- 2b. movement (AttackableUnit.Move, after UpdateBuffs) -------------
    x, y, wp_key, _ = step_move_units(
        state.x, state.y, state.waypoints, state.waypoint_key,
        state.n_waypoints, P("move_speed"),
        _can_move(state.move_order, state.alive), delta_ms)

    # ---- 2p. fog of war (ObjectManager.Update's vision pass) ---------------
    # The server recomputes `IsVisibleByTeam` once per tick, from that tick's
    # positions, and everything downstream just reads the cached flag
    # (`GameServerLib/Lanerl/LanerlFow.cs`'s "AT THE CACHE" comment; the write
    # side is `ObjectManager.UpdateTeamsVision`, `ObjectManager.cs:196`). Done
    # here, once, on the POST-MOVEMENT `x, y` for the same reason target
    # acquisition below uses them and not `state.x/state.y`: a unit that walks
    # into sight range this tick is seen this tick, matching "targeting sees
    # post-movement positions" in this module's own docstring. Recomputed
    # rather than threaded through unchanged from last tick because it is a
    # pure function of (position, kind, team, alive), all already updated
    # above -- there is no cross-tick memory to preserve, unlike e.g.
    # `ignore_until`.
    visible = _visible_to_enemy(x, y, state.kind, state.team, state.alive)

    # ---- 2. the minion controller (AIScript.OnUpdate) ----------------------
    prio = base_priority(state.kind, _minion_type_of(state))
    ai = step_minion_ai(
        kind=state.kind, alive=state.alive, x=x, y=y, team=state.team,
        targetable=state.alive, visible=visible,
        is_attacking=state.is_attacking,
        acquisition_range=P("acquisition_range"),
        base_prio=prio, help_priority=state.help_priority,
        target=state.target, target_priority=state.target_priority,
        ai_timer=state.ai_timer, ai_local_time=state.ai_local_time,
        time_since_attack=state.time_since_attack,
        ignore_until=state.ignore_until,
        had_target=state.target >= 0, move_order=state.move_order,
        delta_ms=delta_ms)

    # ---- 3. target acquisition for non-minions (ObjAIBase.UpdateTarget) ----
    # Champions on attack-move take the nearest enemy, no priority. Turrets have
    # their own rule (targeting.turret_acquire) and are wired in with buildings.
    # `ObjAIBase.UpdateTarget`'s auto-acquisition runs ONLY under
    # `MoveOrder == OrderType.AttackMove`. A champion on a plain MoveTo does not
    # pick up targets by walking near them -- if it did, every move order would
    # start a swing and the policy would never control when it attacks, which is
    # the whole of last-hitting.
    is_champ = state.kind == Kind.CHAMPION
    attack_moving = is_champ & (state.move_order == MoveOrder.ATTACK_MOVE)
    # The fresh-acquisition scan itself does NOT gate on vision. It is the
    # `MoveOrder == OrderType.AttackMove` branch of `ObjAIBase.UpdateTarget`
    # (ObjAIBase.cs:1288-1320 -- "Acquires the closest target"), and its loop
    # only rejects on `IsDead`, `Team`, `DistanceSquared > range*range` and
    # `!Targetable`; no `IsVisibleByTeam` call appears in it at all. Nor need
    # one: `range` there is `Stats.AcquisitionRange.Total` (400 for Garen,
    # `profiles.py`), smaller than a champion's own `VisionRadius` (1200,
    # `fog.VISION_RADIUS`), so anything the scan can find is already seen by
    # the champion doing the looking regardless of any other teammate --
    # filtering candidates here would be a no-op on this patch's numbers, and
    # skipping it matches the server line-for-line instead of only in effect.
    champ_pick = jnp.where(
        attack_moving,
        nearest_enemy(x, y, state.team, state.alive, state.alive,
                      P("acquisition_range")),
        jnp.int8(-1))
    # What DOES gate on vision -- and matters, because this branch carries no
    # distance cap at all, unlike the minion and turret rules below, so a held
    # target can wander arbitrarily far before this is the only thing that
    # drops it. `ObjAIBase.UpdateTarget` (ObjAIBase.cs:1183):
    #     else if (TargetUnit.IsDead || (...) || !TargetUnit.IsVisibleByTeam(Team))
    #     { ...; SetTargetUnit(null, true); return; }
    # `visible` already ANDs with `alive` (fog.visible_to), so this one gather
    # reproduces both halves of that condition that apply here -- IsDead and
    # !IsVisibleByTeam -- rather than needing a separate alive check.
    cur_champ = jnp.clip(state.target, 0, n - 1)
    keep_champ = is_champ & (state.target >= 0) & visible[cur_champ]

    # Turrets have their own rule entirely: priority first, distance never,
    # plus the dive override. `BaseTurret : ObjAIBase` (BaseTurret.cs:17), so
    # it runs the SAME generic `UpdateTarget` as the champion above: `TurretAI.
    # OnUpdate` (its own AIScript) only handles dropping a target that has
    # left ATTACK range (reproduced below as `left_range`); dropping one that
    # died or went invisible is `ObjAIBase.cs:1183` again, not anything
    # `TurretAI` itself does. `target_gone` reproduces that second path --
    # without it a turret that lands a killing blow keeps "holding" the
    # corpse's now-dead slot (`turret_acquire`'s own holding branch has no
    # aliveness check) until something repositions that exact slot index back
    # out of attack range, going idle in the meantime despite live targets
    # sitting in range. A turret's own sight (800) already covers its own
    # attack range (750, `profiles.py`), so -- as with the champion above --
    # candidate visibility filtering changes nothing here on this patch's
    # numbers; it is passed through `targetable` anyway for a live target that
    # somehow sits in attack range but out of every ally's sight (unreached
    # today, but a real gap in the rule otherwise).
    is_turret = state.kind == Kind.TURRET
    turret_pick = turret_acquire(
        x, y, state.team, state.alive, state.alive & visible, state.kind,
        _minion_type_of(state), P("attack_range"), state.target,
        state.target, P("attack_range"))
    cur = jnp.clip(state.target, 0, n - 1)
    left_range = (state.target >= 0) & (
        ((x[cur] - x) ** 2 + (y[cur] - y) ** 2)
        > P("attack_range") ** 2)
    target_gone = (state.target >= 0) & ~visible[cur]
    turret_target = jnp.where(left_range | target_gone, jnp.int8(-1),
                              turret_pick)

    target = jnp.where(
        is_champ, jnp.where(keep_champ, state.target, champ_pick),
        jnp.where(is_turret, turret_target, ai.target))

    # ---- 3b. RefreshWaypoints -------------------------------------------
    # `ObjAIBase.RefreshWaypoints`: a unit holding a target either stops,
    # because the target is already in range, or re-paths onto it.
    #
    #     if (DistanceSquared(Position, targetPos) <= idealRange^2)
    #          UpdateMoveOrder(OrderType.Hold);
    #     else SetWaypoints(GetPath(Position, targetPos));
    #
    # Without this, units with targets keep walking their old path -- minions
    # stroll past each other trading a hit in passing, waves never clear, and
    # the minion count climbs to the array cap. That is exactly what happened
    # before this block existed: 40/40 slots full by t=210 s against a measured
    # server median of 15.
    #
    # BOOKED APPROXIMATION: the re-path is a straight line, not `GetPath`.
    # Targets are acquired within `AcquisitionRange` (600 for a minion) and
    # lane combat happens in the open corridor, where 76% of paths under 500
    # units are already straight (measured). Chasing across terrain is where
    # this is wrong, and it is unmeasured.
    tgt = jnp.clip(target, 0, n - 1)
    d2 = (x[tgt] - x) ** 2 + (y[tgt] - y) ** 2
    # `idealRange = Stats.Range.Total + TargetUnit.CollisionRadius` -- edge to
    # edge, the attacker's own radius deliberately excluded.
    ideal = P("attack_range") + P("collision_radius")[tgt]
    in_rng = d2 <= ideal * ideal
    has_tgt = target >= 0

    hold = has_tgt & in_rng
    chase = has_tgt & ~in_rng
    wp = state.waypoints
    two = jnp.stack([jnp.stack([x, y], -1), jnp.stack([x[tgt], y[tgt]], -1)], 1)
    wp = jnp.where(chase[:, None, None],
                   wp.at[:, :2].set(two)[:, :, :], wp)
    wp_key = jnp.where(chase, jnp.int8(1), wp_key)
    n_wp = jnp.where(chase, jnp.int8(2), state.n_waypoints)
    # Champion attack damage is NOT static. `Stats.LevelUp`
    # (`GameServerLib/GameObjects/Stats/Stats.cs:270-271`) grows
    # `AttackDamage` every level-up through the same non-linear curve as
    # every other per-level stat (`profiles.py`'s `ad_per_level` column,
    # `combat.growth_sum`). `P("attack_damage")` alone is the level-1(+rune)
    # baseline that was the WHOLE of a champion's attack damage for the
    # entire episode before this -- confirmed against the wire's own
    # ``ad`` field (`LanerlControl.cs`'s ``BuildObservation``, whose own
    # comment records the old Python stack making exactly this mistake:
    # "read 57.88 against a real 73.14, a 21% under-report"). `state.level`
    # here is the level as of the START of this tick, matching the server:
    # a level gained mid-tick updates `Stats.AttackDamage.Total` synchronously
    # at the XP-crossing event (`Champion.AddExperience` -> `Stats.LevelUp`),
    # so by the NEXT tick's `Update` the new value is already live -- exactly
    # what reading the incoming `state.level` (not the value recomputed later
    # in this same tick) reproduces. `ad_per_level` is 0 for every non-champion
    # row, so this is a no-op for minions and turrets regardless of their
    # (always 1, see `state.py`) `level` field.
    ad_now = P("attack_damage") + P("ad_per_level") * growth_sum(state.level, jnp)
    raw_ad = _attack_damage_against(
        ad_now, state.kind, state.kind[tgt], state.model, state.t_ms)
    aa = step_autoattack(
        state.aa_cooldown, state.aa_windup, state.is_attacking,
        state.has_auto_attacked,
        in_range=in_rng,
        # `SetStatus(CanAttack, false)` for Judgment's duration
        can_attack=state.alive & ~bs.suppress_attack,
        has_target=has_tgt,
        attack_period=P("attack_period"),
        windup_time=P("attack_windup"),
        attack_damage=raw_ad,
        target_resist=armor_eff[tgt],
        delta_ms=delta_ms, xp=jnp)

    # ---- 5. apply damage, and attribute the kill --------------------------
    # Who gets the gold is the whole of last-hitting, so attribution is not a
    # detail. The server applies each attacker's damage inside its own
    # `Update`, in ObjectManager iteration order, and `TakeDamage` records the
    # killer only on the transition:
    #
    #     if (!IsDead && Stats.CurrentHealth <= 0) { IsDead = true; _death = ... }
    #
    # So among several attackers landing on the same unit in the same tick, the
    # killer is the **lowest-index** one whose cumulative damage first crosses
    # zero -- later hits still land but cannot re-claim the kill. Reproduced
    # here with a cumulative sum along the attacker axis rather than a
    # scatter-add, which would lose the ordering.
    # A ranged attacker (which in this slice means every non-melee one --
    # casters, cannons AND both outer turrets, see `sim/missiles.py`) launches
    # a MISSILE instead of dealing damage now, at ITS OWN `missile_speed`, and
    # that damage is lost entirely if the target dies before the missile
    # lands. Melee attackers (minions and Garen) are unchanged.
    swings = aa.hit & state.alive & (target >= 0)
    ranged = P("fires_missile") > 0
    launches = swings & ranged
    landed = swings & ~ranged

    ms = step_missiles(
        m_alive=state.missile_alive, m_x=state.missile_x, m_y=state.missile_y,
        m_target=state.missile_tx, m_source=state.missile_source,
        m_damage=state.missile_damage, m_speed=state.missile_speed,
        launches=launches, raw_damage=raw_ad, launch_speed=P("missile_speed"),
        x=x, y=y, alive=state.alive,
        targetable=state.alive, armor=armor_eff, target=target,
        delta_ms=delta_ms)

    dmg_ij = jnp.where(landed[:, None] & (jnp.arange(n)[None, :] == tgt[:, None]),
                       aa.damage[:, None], jnp.zeros((n, n), dtype))
    # A missile that lands this tick is credited to the unit that FIRED it, in
    # that unit's own attacker row, so the lowest-index-crosses-zero rule below
    # sees melee hits and missile hits in one ordering rather than two.
    dmg_ij = dmg_ij + ms.damage_ij
    # W's active scales the victim's TOTAL incoming damage for the tick, so it
    # is applied here rather than per attacker -- one multiply on the sum, not
    # one per source, which is what `TakeDamage`'s post-mitigation hook does.
    dealt = (dmg_ij.sum(axis=0) + bs.damage_dealt) * bs.damage_multiplier
    hp = jnp.maximum(state.hp - dealt, jnp.zeros_like(state.hp))

    # ---- out-of-combat clock, for Garen's passive -------------------------
    # `CharScriptGaren.ShouldPassiveTurnOff` returns FALSE -- the passive keeps
    # running -- when the attacker's UnitTags is one of Minion, Minion_Lane,
    # Minion_Lane_Siege, Minion_Lane_Super or Minion_Summon. So ordinary minion
    # damage does NOT put Garen in combat, which is what lets the server's
    # Garen heal continuously while farming.
    #
    # Cannon minions are the exception, and only by accident. `UnitTag` is a
    # [Flags] enum with NO explicit values, so C# numbers it 0,1,2,...:
    # Minion=2, Minion_Lane=3, Minion_Lane_Siege=4, Monster=7. A melee or
    # caster minion is tagged "Minion | Minion_Lane" = 2|3 = 3 = Minion_Lane,
    # which IS in the exception list. A cannon is
    # "Minion | Minion_Lane | Minion_Lane_Siege" = 2|3|4 = **7 = Monster**,
    # which is not -- so a cannon's autoattack DOES break the passive below
    # level 11 (at 11+ a separate Monster check exempts it again).
    #
    # Reproduced deliberately. It is a bug in the server and parity means
    # matching the server, not the mechanic's evident intention.
    _cannon = (state.kind == Kind.LANE_MINION) & \
        (_minion_type_of(state) == MinionType.CANNON)
    breaks_combat = ~((state.kind == Kind.LANE_MINION) & ~_cannon)
    # `dmg_ij` here is (attacker, victim) and NOT yet the concatenated form
    # that prepends the buff-damage row further down. Judgment's damage is
    # carried separately in `bs.damage_dealt` and comes from a champion, so it
    # always counts as combat.
    hit_by_combat = (
        jnp.where(breaks_combat[:, None], dmg_ij, 0.0).sum(axis=0)
        + bs.damage_dealt) > 0
    ms_since_damaged = jnp.where(
        hit_by_combat, jnp.zeros_like(state.ms_since_damaged),
        state.ms_since_damaged + delta_ms)

    alive = state.alive & (hp > 0)
    died = state.alive & ~alive

    # ---- 5b. call for help: a TOGGLE, not a permanent wire ----------------
    # `targeting.call_for_help_map` implements the broadcast faithfully and is
    # tested. `enable_call_for_help=False` (the default) reproduces every
    # test and every measurement in this tree from before this toggle existed,
    # bit for bit: `help_priority` is simply carried forward unchanged.
    #
    # Why a toggle and not a permanent wire: `docs/CALL_FOR_HELP_SWITCH_RATE.md`
    # retires the switch-RATE objection that blocked this before (idle lane,
    # 600 s: sim 582 isolated cfh switches vs the server's 368, 1.58x -- not
    # the "25x too many" that could not be ruled out previously; the "47" in
    # the earlier commit/task background was itself a mis-citation of
    # `to=Champion` acquisitions, not of `cfh=1` switches, which total 453 in
    # that same server log). What's left is a mechanism-level finding, not a
    # rate one: it helps a champion-in-lane scenario (StandInWave response
    # moves 6/7 metrics toward the server) and hurts a fully idle one (median
    # live minions 22 -> 26, blue outer turret survives -> destroyed, and the
    # final population flips from blue dominant 17-5 to red dominant 1-28,
    # deterministically) because it is a reinforcement mechanic that amplifies
    # this lane's own already-documented unstable equilibrium
    # (`lanerl_jax/sim/tests/test_lane.py`'s own docstring) rather than a rate
    # calibration problem more tuning would fix. Left a toggle, default off.
    if enable_call_for_help:
        # `ObjAIBase.TakeDamage`'s broadcast reacts to every landed hit --
        # melee autoattacks and missiles are both folded into `dmg_ij` above
        # (`:377`). NOT folded in: Judgment's damage (`bs.damage_dealt`), which
        # is a per-VICTIM scalar (one caster, tracked via `bs.dealt_by`
        # separately) rather than an attacker/victim matrix, and reproducing it
        # here would need a one-hot scatter this investigation did not need to
        # build to answer the switch-rate question. Booked, not silently
        # dropped: undercounts calls for help raised by a champion's Judgment
        # specifically, nothing else.
        help_priority = call_for_help_map(
            damage_ij=dmg_ij, x=x, y=y, alive=alive, kind=state.kind,
            team=state.team, acquisition_range=P("acquisition_range"))
    else:
        help_priority = state.help_priority


    # Judgment's damage is applied inside the buff's own update, which runs
    # BEFORE the auto-attack gate, so it is prepended to the attribution order.
    dmg_ij = jnp.concatenate(
        [jnp.zeros((1, n), dtype).at[0].set(bs.damage_dealt), dmg_ij], axis=0)
    cum = jnp.cumsum(dmg_ij, axis=0)                     # (attacker, victim)
    crosses = (cum >= state.hp[None, :]) & (dmg_ij > 0)
    first_row = jnp.argmax(crosses, axis=0)
    # row 0 is the Judgment lane; map it back to whoever cast the spin
    killer = jnp.where(
        jnp.any(crosses, axis=0),
        jnp.where(first_row == 0, bs.dealt_by, first_row - 1),
        -1).astype(jnp.int8)
    killer = jnp.where(died, killer, jnp.int8(-1))

    rw = death_rewards(
        died=died, killer=killer, x=x, y=y, team=state.team, kind=state.kind,
        alive=alive, gold_on_death=P("gold_on_death"),
        xp_on_death=P("xp_on_death"))
    amb, gold_timer = ambient_gold(state.t_ms, state.gold_timer,
                                   state.kind == Kind.CHAMPION)
    gold = state.gold + rw.gold + amb
    xp = state.xp + rw.xp
    cs = state.cs + rw.cs.astype(state.cs.dtype)
    level = jnp.where(state.kind == Kind.CHAMPION,
                      level_for_xp(xp, params["xp_curve"]), state.level)
    # Spell ranks are a pure function of champion level under a fixed skill
    # order, so they need no state of their own. The server spends the points
    # through `AutoLevelUndriven` / `Champion.LevelUpSpell`; the order is the
    # one in `constants.GAREN_SKILL_ORDER`.
    spell_level = jnp.where(
        (state.kind == Kind.CHAMPION)[:, None],
        _RANK_TABLE[jnp.clip(level.astype(jnp.int32), 0, 18)],
        state.spell_level)

    move_order_out = jnp.where(
        hold, jnp.int8(MoveOrder.HOLD),
        jnp.where(chase, jnp.int8(MoveOrder.ATTACK_TO),
                  jnp.where(is_champ, state.move_order, ai.move_order)))

    # ---- 6. champion death and respawn -----------------------------------
    # `Champion.Die` sets RespawnTimer = DeathTimes[Level] * 1000; the timer is
    # decremented in `Champion.Update` and `Respawn()` restores FULL health at
    # the spawn point. Minions and turrets do not come back within an episode
    # (turrets are revived only by `LanerlEpisode.RestoreBuildings`, between
    # episodes -- which is itself a bug this project has paid for; see
    # ObjectManager.Update's comment about a map losing four towers over a run).
    is_ch = state.kind == Kind.CHAMPION
    lvl = jnp.clip(level.astype(jnp.int32), 1, params["death_times"].shape[0]) - 1
    died_ch = died & is_ch
    rt = jnp.where(died_ch, params["death_times"][lvl] * 1000.0, state.respawn_ms)
    rt = jnp.where(rt > 0, rt - jnp.asarray(delta_ms, dtype), rt)
    reborn = is_ch & (state.respawn_ms > 0) & (rt <= 0)
    alive = alive | reborn
    hp = jnp.where(reborn, params["max_hp"][state.model], hp)
    x = jnp.where(reborn, state.spawn_x, x)
    y = jnp.where(reborn, state.spawn_y, y)
    rt = jnp.where(reborn, jnp.asarray(-1.0, dtype), rt)
    deaths = state.deaths + died_ch.astype(state.deaths.dtype)

    return state.replace(
        respawn_ms=rt, deaths=deaths,
        t_ms=state.t_ms + jnp.asarray(delta_ms, dtype),
        tick=state.tick + 1,
        x=x, y=y, waypoint_key=wp_key, waypoints=wp, n_waypoints=n_wp,
        target=target.astype(state.target.dtype),
        target_priority=ai.target_priority,
        move_order=move_order_out,
        ai_timer=ai.ai_timer, ai_local_time=ai.ai_local_time,
        time_since_attack=ai.time_since_attack, ignore_until=ai.ignore_until,
        aa_cooldown=aa.aa_cooldown, aa_windup=aa.aa_windup,
        is_attacking=aa.is_attacking, has_auto_attacked=aa.has_auto_attacked,
        hp=hp, alive=alive,
        # `visible` was computed from this tick's post-movement, PRE-death
        # positions/alive (see the fog-of-war block above) -- exactly what
        # this tick's own targeting needed. ANDed with the tick's final
        # `alive` here before it is stored, so a unit that died or respawned
        # (moved to `spawn_x/y`) THIS tick is never read back next tick as a
        # visible target through the stored field: `visible_to_enemy` implies
        # `alive`, the same invariant `fog.visible_to`/`visible_to_enemy`
        # already hold internally, and downstream readers (e.g. an
        # observation builder) get exactly what they'd get from calling
        # `fog.visible_to_enemy` themselves on the returned state.
        visible_to_enemy=visible & alive,
        gold=gold, xp=xp, cs=cs, level=level,
        gold_timer=gold_timer, ms_since_damaged=ms_since_damaged,
        spell_level=spell_level, buff_id=bs.buff_id,
        buff_elapsed=bs.buff_elapsed, spell_cooldown=bs.spell_cooldown,
        missile_alive=ms.alive, missile_x=ms.x, missile_y=ms.y,
        missile_tx=ms.target.astype(state.missile_tx.dtype),
        missile_source=ms.source.astype(state.missile_source.dtype),
        missile_damage=ms.damage, missile_speed=ms.speed,
        help_priority=help_priority,
    )


def step_decision(state: LaneState, params: UnitParams,
                  step_ticks: int = 2, delta_ms: float = TICK_MS,
                  lane_path=None, minion_hp=None,
                  enable_call_for_help: bool = False) -> LaneState:
    """One agent decision = ``LANERL_STEP_TICKS`` server ticks.

    ``step_ticks`` is 2 in this stack (30 Hz decisions off a 60 Hz sim), set by
    ``lanerl_rl.constants.STEP_TICKS`` and passed to the server as
    ``LANERL_STEP_TICKS``; the two must not drift apart.

    ``enable_call_for_help`` defaults to ``False`` -- see ``tick``'s docstring.
    """
    def one(s, _):
        return tick(s, params, delta_ms, lane_path, minion_hp,
                    enable_call_for_help), None
    out, _ = jax.lax.scan(one, state, None, length=step_ticks)
    return out
