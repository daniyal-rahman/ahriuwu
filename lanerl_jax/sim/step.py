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

The phases of ``tick()``
-----------------------
Numbered exactly as the ``# ---- N.`` markers in :func:`tick`, in the order
they run (`tests/test_step_phase_order.py` fails if the two drift apart). The
server order they reproduce is ``Map.Update`` (collision, fountain, waves),
then per object ``AttackableUnit.Update`` (buffs, regen, move),
``AIScript.OnUpdate``, ``Spell.Update``, ``UpdateTarget``, with deaths and
their rewards resolved after every unit's update.

   1. collision push-apart (Map.Update, FIRST thing in the tick)
   2. fountain healing (LevelScriptObjects.OnUpdate -> Fountain)
   3. wave spawning (MapScript.Update, still inside Map.Update)
   4. buffs (ObjectManager.Update -> AttackableUnit.UpdateBuffs)
   5. Garen's W: the resist/damage hooks spells.py asks for
   6. Stats.Update: HP regen (AttackableUnit.Update, after buffs)
   7. recall damage-buff / cast-windup state
   8. movement (AttackableUnit.Move, after UpdateBuffs)
   9. recall and R Spell.Update (after Move, before targeting)
  10. fog of war (ObjectManager.Update's vision pass)
  11. the minion controller (AIScript.OnUpdate)
  12. LaneMinionAI.WaypointReached
  13. target acquisition (ObjAIBase.UpdateTarget, TurretAI)
  14. RefreshWaypoints
  15. the swing gate and auto-attack clock; Q's hit and silence
  16. apply damage (melee hits, missiles, buff damage)
  17. Champion._championHitFlagTimer / _playerHitId
  18. out-of-combat clock, for Garen's passive
  19. the dead drop target and swing (UpdateTarget; TGT-DEATHTICK)
  20. call for help
  21. kill attribution and death rewards (AttackableUnit.Die)
  22. champion-kill gold/XP (Champion.Die)
  23. turret-destruction gold/XP (LaneTurret.Die)
  24. gold, XP, level-up and spell ranks
  25. move order out, and FinishCasting's Hold
  26. champion death and respawn (Champion.Update)

Known order deviations (`STRUCT-006`), each low impact and each recorded:
the sim computes vision fresh at phase 10 from this tick's positions, while
the server's targeting reads a visibility cache written at the END of the
previous tick (`LanerlFow.cs`); and R's damage is applied at the buff phase
(4) where the server applies it in ``Spell.Update`` after the AI script (one
regen step apart).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .autoattack import step_autoattack
from .combat import (
    TURRET_AD_PER_RAMP,
    TURRET_ARMOR_PER_RAMP,
    TURRET_DAMAGE_VS_MINION,
    garen_passive_exempt,
    growth_sum,
    other_turret_ramps,
    outer_turret_ramps,
    stat_total,
)
from .collision import resolve_collisions
from ..obs.fog import visible_to_enemy as _visible_to_enemy
from .init import MINION_SPAWN, spawn_minion
from .missiles import step_missiles
from .profiles import PROFILES
from .spells import (Q_HASTE_MULTIPLIER, RANKS_BY_LEVEL, Slot,
                     consume_q_skip, end_q, grant_w_passive,
                     q_damage_at_rank, q_silence_duration_at_rank, status,
                     status_of, step_buffs, w_passive_modifiers)
from .terrain_jax import map1_terrain, repair_collision_terrain_batch
from .waves_jax import step_waves_jax
from .minion_ai import LaneWaypointOut, advance_lane_waypoints, step_minion_ai
from .movement_jax import TICK_MS, step_move_units
from .regen import step_regen
from .rewards import (ambient_gold, champion_kill_rewards, death_rewards,
                      level_for_xp, minion_gold_deathspree_decay,
                      turret_kill_rewards, update_hit_flag)
from .state import (AA_TARGET_GONE, Kind, LaneState, MoveOrder, Team,
                    TurretTier, TU_SLICE)
from .targeting import (MinionType, base_priority, call_for_help_map,
                        nearest_enemy, turret_acquire)

__all__ = ["UnitParams", "tick", "step_decision", "env_apply", "env_advance",
           "env_step"]

# One static 293x294 boolean device constant, not per-environment state.
# `CollisionHandler.Update` needs it before the first object update every tick.
_MAP1_TERRAIN = map1_terrain()


def _attack_damage_against(attack_damage, attacker_kind, target_kind,
                           model=None, t_ms=None):
    """Raw attack damage, with the attacker/target modifiers the scripts apply.

    ``TURRET_DAMAGE_VS_MINION`` (`combat.py`) is **1.0** on this map -- a
    no-op -- and this function's `vs_minion`/`from_turret` branch exists only
    so that stops being true the day it should: the 0.7x-vs-minion discount is
    real, but it belongs to `SRUAP_Turret_*`'s `BasicAttack.cs`, a Map11-only
    script. Map1's turrets (`OrderTurret*`/`ChaosTurret*`) have no
    `Characters/<model>/` script folder at all, so `CSharpScriptEngine
    .CreateObjectStatic<ICharScript>("CharScripts", $"CharScript{Model}")`
    falls back to `SpellScriptEmpty` and their basic attack resolves through
    the native `ObjAIBase.AutoAttackHit` at full, undiscounted AD -- confirmed
    directly from the script-resolution code, not merely from the constant
    already being 1.0. See `combat.TURRET_DAMAGE_VS_MINION`'s own docstring
    for the full citation; this docstring previously described the discount as
    live, which it never was on this map.

    What IS real here: a property of the attacker's own tier and the game
    clock, the per-tier AD ramp below.
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

_RECALL_WINDUP_MS = 500.0
_RECALL_CHANNEL_MS = 8_000.0
# ``Characters/Global/Recall.OnSpellChannel`` gives its damage-listener buff
# 7.9 seconds, whereas the Spell channel itself lasts 8.0 seconds.
_RECALL_DAMAGE_BUFF_MS = 7_900.0
_FOUNTAIN_HEAL_PERIOD_MS = 1_000.0
_FOUNTAIN_HEAL_FRAC = 0.15
_FOUNTAIN_RADIUS = 1_000.0


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
         minion_hp=None, enable_call_for_help: bool = True,
         enable_collision: bool = True,
         collision_terrain: bool = True,
         defer_collision_terrain: bool = False) -> LaneState:
    """Advance one 16.667 ms server tick.

    ``lane_path`` is ``(W, 2)`` -- ``MinionPaths[LANE_L]``, walked forward by
    blue and reversed by red, exactly as ``SetUpLaneMinion`` does
    (``waypoint.Reverse()`` for ``TEAM_PURPLE``). Pass ``None`` to run without
    wave spawning, which is what the unit tests want.

    ``enable_call_for_help`` defaults to ``True`` because the reference server
    broadcasts it unconditionally.  The switch remains available for ablation
    and historical parity measurements, but production callers reproduce the
    server unless they explicitly opt out.
    """
    n = state.kind.shape[-1]
    dtype = state.x.dtype
    # Every stat is a gather through the unit's profile row: a minion slot is
    # reused by whatever spawns into it, so stats cannot be baked per slot.
    P = lambda k: params[k][state.model]          # noqa: E731
    # `Game.Update` (`Game.cs:474-497`): `GameTime += diff` runs BEFORE
    # `Map.Update` and `ObjectManager.Update` in the SAME call -- so every
    # per-tick absolute-game-time check the server makes (wave spawning in
    # `LevelScript.Update`, the turret AD/armour ramps in the SAME script's
    # `LevelScriptObjects.OnUpdate`, `Champion.Update`'s ambient-gold gate)
    # reads the POST-increment value, i.e. THIS tick's own outgoing time, not
    # the incoming one `state.t_ms` holds (last tick's outgoing time). Used
    # everywhere below that used to read `state.t_ms` for a ">="-style
    # threshold rather than a duration. See "1. wave spawning" below for the
    # trace evidence this was wrong, not just a theoretical nit.
    t_now = state.t_ms + jnp.asarray(delta_ms, dtype)
    # An INNER/INHIBITOR/NEXUS turret's armour is not its Content value for
    # most of a game either -- see `_attack_damage_against` for the AD half of
    # the same pair of schedules. Computed once, up front, because armour is
    # read in three places below (buffs, the autoattack mitigation target, and
    # missile mitigation) and all three must see the same ramped value.
    level_growth = growth_sum(state.level, jnp)
    base_armor = P("armor") + P("armor_per_level") * level_growth
    magic_resist_now = (
        P("magic_resist") + P("mr_per_level") * level_growth)
    armor_now = base_armor + jnp.where(
        _OTHER_TURRET_ROW[state.model],
        TURRET_ARMOR_PER_RAMP * other_turret_ramps(t_now, jnp),
        jnp.zeros_like(base_armor))

    # ---- 1. collision push-apart (Map.Update, FIRST thing in the tick) ------
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
    # Read through `status` (`STRUCT-001`): this line used to index the buff
    # lane table with `Slot.E` -- 2, the SPELL slot -- where E's buff lived in
    # lane 0, so it compared the W-passive lane against GAREN_E, was
    # identically False, and Garen NEVER ghosted while spinning (`SPELL-005`).
    pre_ghosted = status_of(state).ghosted
    if enable_collision:
        cx, cy = resolve_collisions(state.x, state.y, state.kind, state.alive,
                                    state.spawn_seq, P("collision_radius"),
                                    P("pathfinding_radius"), ghosted=pre_ghosted,
                                    candidate_x=state.collision_x,
                                    candidate_y=state.collision_y,
                                    candidate_present=state.collision_present,
                                    terrain=(_MAP1_TERRAIN if collision_terrain
                                             else None),
                                    mover_indices=jnp.arange(
                                        TU_SLICE.start, dtype=jnp.int32))
    else:
        cx, cy = state.x, state.y
    if enable_collision and defer_collision_terrain:
        moved_by_unit = (cx != state.x) | (cy != state.y)
        # Same eligibility as the inline path's `mover_indices`: champions
        # and minions. Without it the fountain turrets, which sit on
        # unwalkable cells by design, were "repaired" out of terrain every
        # tick and walked across the map in every RL run (`COLL-004`).
        cx, cy, _terrain_exhausted = repair_collision_terrain_batch(
            cx, cy, P("pathfinding_radius"), moved_by_unit, _MAP1_TERRAIN,
            eligible=jnp.arange(state.x.shape[0]) < TU_SLICE.start)
    # UpdateQuadTree runs now, before wave spawns and unit movement. A later
    # spawn is inserted into the same tree immediately by OnAdded; spawn_minion
    # mirrors that insertion in its masked slot write.
    collision_present = state.alive & (state.kind != Kind.NONE)
    state = state.replace(
        x=cx, y=cy, collision_x=cx, collision_y=cy,
        collision_present=collision_present)

    # ---- 2. fountain healing (LevelScriptObjects.OnUpdate -> Fountain) ------
    # Map1 updates its two Fountain objects before ObjectManager. Their timers
    # start together at zero and are reset (not remainder-preserved) together,
    # so one scalar is exact. A recall which completes later in this tick does
    # not receive this already-passed fountain pulse.
    fountain_timer = state.fountain_heal_ms + jnp.asarray(delta_ms, dtype)
    fountain_pulse = fountain_timer >= jnp.asarray(_FOUNTAIN_HEAL_PERIOD_MS, dtype)
    fountain_timer = jnp.where(fountain_pulse, jnp.asarray(0.0, dtype), fountain_timer)
    is_champion = (state.kind == Kind.CHAMPION) & state.alive
    d_spawn2 = ((state.x - state.spawn_x) ** 2
                + (state.y - state.spawn_y) ** 2)
    in_own_fountain = is_champion & (d_spawn2 <= _FOUNTAIN_RADIUS ** 2)
    fountain_heal = jnp.where(
        fountain_pulse & in_own_fountain,
        state.max_hp * jnp.asarray(_FOUNTAIN_HEAL_FRAC, dtype), 0.0)
    state = state.replace(
        # Fountain.Update only calls TakeHeal for its eligible champion
        # targets.  Do not clamp every other unit on a pulse: constructed
        # states may legitimately use hp without filling max_hp, and a
        # zero-heal `min(hp, max_hp)` would manufacture an uncredited death.
        hp=jnp.where(fountain_heal > 0,
                     jnp.minimum(state.hp + fountain_heal, state.max_hp),
                     state.hp),
        fountain_heal_ms=fountain_timer)

    # ---- 3. wave spawning (MapScript.Update, still inside Map.Update) -------
    # `Game.Update` (`Game.cs:474-497`): `GameTime += diff` runs BEFORE
    # `Map.Update(diff)` (which is where `LevelScript.Update`'s spawn check
    # lives) in the SAME call. So the gameTime `LevelScript.Update` reads for
    # THIS tick's decision is already the POST-increment value -- the tick's
    # own outgoing time, not the incoming one `state.t_ms` holds here (that is
    # last tick's outgoing time, i.e. this tick's incoming time). Using
    # `state.t_ms` under-checks by one tick's worth of game time (16.667 ms)
    # on every spawn-eligibility test.
    #
    # Confirmed against the recorded trace, not just read off the source:
    # replaying the injected server state at a tick immediately BEFORE a real
    # spawn (e.g. t=90,798 ms, one red minion), `state.t_ms >= next_spawn_ms +
    # minion_number*800` is false (90,798 < 90,800) and the old code predicted
    # no spawn, while the server's own transition to the NEXT tick (90,815 ms)
    # already shows the new minion -- i.e. the decision was made against
    # 90,815, not 90,798. This is very likely why
    # docs/TIER1_POST_REORDER.md's Tier-1 pass found essentially every
    # spawn-adjacent tick disagreeing on population, in one direction or the
    # other depending on which side of a threshold `state.t_ms` alone landed
    # on.
    if lane_path is not None:
        mtype, next_spawn, m_no, c_no = step_waves_jax(
            t_now, state.next_spawn_ms, state.minion_number, state.cannon_count)
        mi = jnp.clip(mtype, 0, 3)
        hp_b = params["max_hp"][_WAVE_ROW_BLUE[mi]]
        hp_r = params["max_hp"][_WAVE_ROW_RED[mi]]
        # Spawn at the MEASURED barracks, not at the path's end vertex --
        # they differ by 446 units on the red side. See `spawn_minion`.
        # Map1's `SpawnBarracks` is populated from the package's enumeration
        # order.  In TOPONLY that order reaches ``__P_Chaos...L01`` (red) just
        # before ``__P_Order...L01`` (blue), so the red minion's
        # `GameObject.OnAdded` / CollisionHandler insertion happens first.
        # This is observable once the waves meet: collision sweeps creation
        # order, not side or our minion-slot order.  Keep these writes in the
        # server order even though their slots are disjoint.
        state = spawn_minion(state, Team.RED, _WAVE_ROW_RED[mi], hp_r,
                             lane_path[::-1], enabled=mtype >= 0,
                             spawn_xy=MINION_SPAWN[Team.RED])
        state = spawn_minion(state, Team.BLUE, _WAVE_ROW_BLUE[mi], hp_b,
                             lane_path, enabled=mtype >= 0,
                             spawn_xy=MINION_SPAWN[Team.BLUE])
        state = state.replace(next_spawn_ms=next_spawn, minion_number=m_no,
                              cannon_count=c_no)

    # ---- 4. buffs (ObjectManager.Update -> AttackableUnit.UpdateBuffs) ------
    # First thing inside the unit's own Update, and therefore AFTER the
    # collision pass above and BEFORE movement below. It reads post-collision,
    # pre-move positions. That distinction only became real once collision
    # moved to the front of the tick: while collision ran last, pre-move and
    # post-collision were the same positions and this was harmless.
    #
    # The effective resists are computed FIRST, from the incoming buff state
    # with this tick's W-passive grant applied, and E's and R's damage inside
    # `step_buffs` is mitigated against them (`SPELL-007`). They used to be
    # computed after `step_buffs`, so a spin or an R landing on a champion
    # with W ranked saw pre-passive Armor/MR while an auto-attack landing on
    # the same unit in the same tick saw the passive: `TakeDamage` reads
    # `Stats.Armor.Total`, which includes `GarenWPassive`'s modifier, for all
    # of them. (Downstream values are unchanged: the passive `step_buffs`
    # reports is the same grant on the same inputs.)
    buffs_in = grant_w_passive(state.buffs, state.spell_level, state.alive)
    wp = w_passive_modifiers(buffs_in, state.alive, dtype)

    # ---- 5. Garen's W: the resist/damage hooks spells.py asks for -----------
    # W's PASSIVE is granted once on first rank-up of W (`W.cs:26-46` registers
    # an OnLevelUpSpell listener at spell construction, so it does not require
    # ever pressing W) and is NOT a clean +20% to either stat -- see
    # `spells.py`'s W section for the full derivation. W's ACTIVE is meant to
    # multiply all incoming post-mitigation damage by 0.7 while the window is
    # open (`GarenW.cs:47-55`), but a verified server bug means it never
    # actually reaches real HP loss -- `bs.damage_multiplier` is
    # unconditionally 1.0 (see `spells.py`), so the multiply below is inert by
    # construction, not a mistake.
    #
    # Both are identity when Garen has never levelled W, so this changes
    # nothing in a lane where W is never ranked.
    #
    # `Stat.Total = ((BaseValue+BaseBonus)*(1+PercentBaseBonus) + FlatBonus)
    # * (1+PercentBonus)` (`combat.stat_total`) -- NOT a flat
    # `base * (1 + pct)`, which is the bug this replaces (a clean +20% only
    # by coincidence when `FlatBonus == 0` AND `PercentBaseBonus == 0`,
    # neither of which holds once the passive itself sets
    # `PercentBaseBonus = -0.2`). The base is `armor_now`, NOT `P("armor")`:
    # a non-outer turret's armour already grows +1 every 60 s from 480 s
    # (`other_turret_ramps`), which has to compose with the passive rather
    # than be replaced by it -- inert here regardless, since only a champion
    # (Garen) ever carries this buff, but kept for the same reason the
    # ramp-vs-passive ordering mattered before this fix. `P("armor_flat_bonus")`
    # is `Armor.FlatBonus` (the rune page, applied as an item -- see
    # `spells.py`'s W-passive citation): 0 for every non-champion row and for
    # MagicResist entirely (no MR rune/item source is modelled), so
    # subtracting it back out of `armor_now` to recover `BaseValue+BaseBonus`
    # is a no-op wherever the passive itself is also 0.
    armor_flat = P("armor_flat_bonus")
    armor_eff = stat_total(
        armor_now - armor_flat, base_bonus=0.0,
        percent_base_bonus=wp.armor_percent_base_bonus,
        flat_bonus=armor_flat, percent_bonus=wp.armor_percent_bonus)
    magic_resist_eff = stat_total(
        magic_resist_now, base_bonus=0.0,
        percent_base_bonus=wp.mr_percent_base_bonus, flat_bonus=0.0,
        percent_bonus=wp.mr_percent_bonus)

    bs = step_buffs(
        buffs=buffs_in, spell_cooldown=state.spell_cooldown,
        spell_level=state.spell_level,
        x=state.x, y=state.y, kind=state.kind, team=state.team,
        alive=state.alive, armor=armor_eff,
        magic_resist=magic_resist_eff, hp=state.hp, max_hp=state.max_hp,
        # E's 330 is centre-to-EDGE on the server: `GetUnitsInRange` tests the
        # quadtree's per-unit collision circle, so the effective radius is
        # 330 + r_target (370 vs minions, 360 vs champions).
        collision_radius=P("collision_radius"),
        delta_ms=delta_ms)

    # ---- 6. Stats.Update: HP regen (AttackableUnit.Update, after buffs) -----
    # Right after UpdateBuffs and before Move, on its own 500 ms accumulator.
    # Not modelling this is why our champion died 7 times in an oracle-driven
    # 600 s episode where the server's died 0 -- see `sim/regen.py`.
    rg = step_regen(
        hp=state.hp, max_hp=state.max_hp, alive=state.alive, kind=state.kind,
        level=state.level,
        hp_regen=P("hp_regen") + P("hp_regen_per_level") * level_growth,
        stat_timer=state.stat_timer, heal_timer=state.heal_timer,
        ms_since_damaged=state.ms_since_damaged, delta_ms=delta_ms)
    state = state.replace(hp=rg.hp, stat_timer=rg.stat_timer,
                          heal_timer=rg.heal_timer)

    # ---- 7. recall damage-buff / cast-windup state --------------------------
    # Recall's buff observes non-periodic damage and cancels itself on ITS
    # NEXT OnUpdate. This phase is before movement and Spell.Update, matching
    # `AttackableUnit.UpdateBuffs`; it therefore prevents a pending channel
    # from consuming another frame. The 0.5 s windup is a regular cast and is
    # intentionally not affected by the buff (which is added only on channel).
    recall_from_damage = (state.recall_channel_ms > 0) & state.recall_damage_pending
    recall_channel_start = jnp.where(recall_from_damage, 0.0,
                                     state.recall_channel_ms)
    recall_windup_start = state.recall_windup_ms
    r_cast_start = state.r_cast_ms
    state = state.replace(
        recall_channel_ms=recall_channel_start,
        recall_damage_pending=jnp.zeros_like(state.recall_damage_pending),
        move_order=jnp.where(recall_from_damage, jnp.int8(MoveOrder.HOLD),
                             state.move_order))

    # ---- 8. movement (AttackableUnit.Move, after UpdateBuffs) ---------------
    # GarenQHaste writes `MoveSpeed.PercentBonus += .35` on activation. Read
    # the post-UpdateBuffs table so its expiry frame uses the unbuffed speed.
    q_hasted = bs.buffs.q_haste.active
    move_speed = P("move_speed") * jnp.where(
        q_hasted, jnp.asarray(Q_HASTE_MULTIPLIER, dtype), 1.0)
    # `ENT-12`: a DEAD champion keeps walking. `ObjAIBase.CanMove`
    # (`ObjAIBase.cs:302-315`) binds `!IsDead` only to the dash clause;
    # `Champion.Die` stops a dash and nothing else (`Champion.cs:504-505`),
    # and a champion is never removed, so `AttackableUnit.Update` keeps
    # calling `Move` on the corpse. Measured on the server dumps: every
    # champion that died on an unfinished MoveTo moved on every dead tick
    # (598/598, 599/599, 899/899). Orders are still refused while dead
    # (`CanChangeWaypoints` has `!IsDead`; `orders.py` gates on `alive`), so
    # the corpse only finishes the route it had. Dead MINIONS are removed on
    # the server (`SetToRemove` in `AttackableUnit.Die`) and stay frozen here.
    corpse_walks = state.kind == Kind.CHAMPION
    x, y, wp_key, _ = step_move_units(
        state.x, state.y, state.waypoints, state.waypoint_key,
        state.n_waypoints, move_speed,
        (_can_move(state.move_order, state.alive | corpse_walks)
         & (recall_windup_start <= 0) & (recall_channel_start <= 0)
         & (r_cast_start <= 0)), delta_ms)

    # ---- 9. recall and R Spell.Update (after Move, before targeting) --------
    # The BluePill has the engine's ordinary 0.5 s cast time, then the
    # script's 8 s channel. `Spell.Update` decrements a live channel before
    # ChannelCancelCheck; a MoveTo/Attack* order consequently consumes this
    # frame but then stops the channel. Finishing is checked after that call
    # in the source, so an exact-final-frame Move still completes (the source
    # does not re-check Spell.State before FinishChanneling).
    dt = jnp.asarray(delta_ms, dtype)
    windup_live = (recall_windup_start > 0) & state.alive
    recall_windup = jnp.where(
        windup_live, jnp.maximum(recall_windup_start - dt, 0.0),
        recall_windup_start)
    begin_channel = windup_live & (recall_windup <= 0)
    channel_live = (recall_channel_start > 0) & state.alive
    channel_counted = jnp.where(
        channel_live, jnp.maximum(recall_channel_start - dt, 0.0),
        recall_channel_start)
    channel_complete = channel_live & (channel_counted <= 0)
    channel_break_order = (
        (state.move_order == MoveOrder.MOVE_TO)
        | (state.move_order == MoveOrder.ATTACK_MOVE)
        | (state.move_order == MoveOrder.ATTACK_TO))
    channel_cancel_move = channel_live & channel_break_order & ~channel_complete
    recall_channel = jnp.where(
        begin_channel, jnp.asarray(_RECALL_CHANNEL_MS, dtype),
        jnp.where(channel_cancel_move, 0.0, channel_counted))
    recall_move_order = jnp.where(
        channel_cancel_move | channel_complete, jnp.int8(MoveOrder.HOLD),
        state.move_order)
    # `Recall.OnSpellPostChannel -> Champion.Recall -> TeleportTo`; unlike a
    # respawn this does NOT restore health. Fountain's next map-phase pulse is
    # the only healing it receives.
    x = jnp.where(channel_complete, state.spawn_x, x)
    y = jnp.where(channel_complete, state.spawn_y, y)
    state = state.replace(move_order=recall_move_order)

    # R's damage mailbox is held on its victim, but Spell.Update's cast lock
    # belongs to the caster. The lock is uncancellable by ordinary orders
    # (`CantCancelWhileWindingUp=1`) and only clears at FinishCasting or death.
    r_cast_live = (r_cast_start > 0) & state.alive
    r_cast_ms = jnp.where(r_cast_live, jnp.maximum(r_cast_start - dt, 0.0),
                          r_cast_start)
    r_cast_finished = r_cast_live & (r_cast_ms <= 0)

    # ---- 10. fog of war (ObjectManager.Update's vision pass) ----------------
    # The server recomputes `IsVisibleByTeam` once per tick and everything
    # else just reads the cached flag (`GameServerLib/Lanerl/LanerlFow.cs`'s
    # "AT THE CACHE" comment). BUT the write, `UpdateTeamsVision`, runs in
    # `ObjectManager.Update` AFTER every object's `Update` (`ObjectManager.cs`:
    # the `foreach ... obj.Update(diff)` loop, then removals/additions, then
    # `UpdateTeamsVision(obj)`), so this tick's targeting on the server reads
    # the flags written from LAST tick's end positions. The sim instead
    # computes vision here from this tick's post-movement `x, y`: a unit that
    # walks into sight is seen one tick earlier than on the server. Recorded
    # as a known deviation (`STRUCT-006`, and this module's docstring), not
    # changed here: the difference is at most one tick at a sight boundary.
    visible = _visible_to_enemy(x, y, state.kind, state.team, state.alive)

    # ---- 11. the minion controller (AIScript.OnUpdate) ----------------------
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
        had_target=state.had_target, move_order=state.move_order,
        spawn_seq=state.spawn_seq,
        delta_ms=delta_ms)

    # ---- 12. LaneMinionAI.WaypointReached -----------------------------------
    # A lane minion owns TWO waypoint cursors on the server: the transient
    # AttackableUnit movement route, and LaneMinionAI's persistent index into
    # PathingWaypoints. Combat overwrites the first while chasing a target, so
    # `lane_waypoint_key` is deliberately independent and lets it resume the
    # correct lane waypoint once the target is gone.
    lane_key = state.lane_waypoint_key
    lane_stop = jnp.zeros_like(state.alive)
    wp_after_lane = state.waypoints
    wp_key_after_lane = wp_key
    n_wp_after_lane = state.n_waypoints
    if lane_path is not None:
        base_lane = jnp.asarray(lane_path, dtype)
        per_unit_lane = jnp.broadcast_to(base_lane, (n,) + base_lane.shape)
        per_unit_lane = jnp.where(
            (state.team == Team.RED)[:, None, None],
            jnp.broadcast_to(base_lane[::-1], (n,) + base_lane.shape),
            per_unit_lane)
        lane_active = ((state.kind == Kind.LANE_MINION) & state.alive
                       & ai.reevaluated & (ai.target < 0))

        def advance(_):
            return advance_lane_waypoints(
                kind=state.kind, alive=state.alive, x=x, y=y,
                collision_x=state.collision_x, collision_y=state.collision_y,
                collision_present=state.collision_present,
                spawn_seq=state.spawn_seq,
                collision_radius=P("collision_radius"),
                acquisition_range=P("acquisition_range"),
                lane_waypoints=per_unit_lane, lane_waypoint_key=lane_key,
                waypoints=state.waypoints, n_waypoints=state.n_waypoints,
                reevaluated=ai.reevaluated, has_target=ai.target >= 0)

        def no_advance(_):
            return LaneWaypointOut(
                key=lane_key,
                destination=jnp.zeros((n, 2), dtype),
                reset_path=jnp.zeros((n,), dtype=bool),
                stop=jnp.zeros((n,), dtype=bool))

        # The source only enters WaypointReached from a 250-ms controller
        # sweep (or its immediate event branches). Avoid the N² sort/cluster
        # scan on ordinary movement ticks where no lane minion can inspect it.
        lane = jax.lax.cond(jnp.any(lane_active), advance, no_advance,
                            operand=None)
        lane_key = lane.key
        lane_stop = lane.stop
        lane_two = jnp.stack([jnp.stack([x, y], -1), lane.destination], axis=1)
        wp_after_lane = jnp.where(
            lane.reset_path[:, None, None],
            state.waypoints.at[:, :2].set(lane_two), state.waypoints)
        wp_key_after_lane = jnp.where(
            lane.reset_path, jnp.int8(1), wp_key)
        n_wp_after_lane = jnp.where(
            lane.reset_path, jnp.int8(2), state.n_waypoints)

    # ---- 13. target acquisition (ObjAIBase.UpdateTarget, TurretAI) ----------
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
                      P("acquisition_range"), state.spawn_seq),
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
    # `visible` is visible-to-the-ENEMY, which equals `IsVisibleByTeam(Team)`
    # only for a hostile target. An ALLY target (`ENT-01`: accepted and held,
    # the server's disengage) is unconditionally visible to its own team, so
    # it is kept on the team test instead of blinking out on the fog test.
    keep_champ = is_champ & (state.target >= 0) & (
        visible[cur_champ] | (state.team[cur_champ] == state.team))

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
    # `ENT-07`: `if (!baseTurret.IsAttacking) CheckForTargets();`
    # (`TurretAI.cs:23`). The AI script runs before `Spell.Update`
    # (`ObjAIBase.cs:1131-1145`), so the gate reads the `IsAttacking` carried
    # into the tick: a turret mid-swing keeps what it holds and cannot switch
    # to a diver until the swing ends. The retention/`target_gone` drops below
    # still apply -- they are not inside that `if`.
    turret_pick = jnp.where(
        state.is_attacking, state.target,
        turret_acquire(
            x, y, state.team, state.alive, state.alive & visible, state.kind,
            _minion_type_of(state), P("attack_range"), state.target,
            state.target, P("attack_range"), state.spawn_seq,
            collision_radius=P("collision_radius")))
    # `TURRET-001`. The retention test is applied to the target the turret
    # holds AFTER `CheckForTargets`, not to the one it entered the tick with,
    # because that is the order inside a single `TurretAI.OnUpdate`
    # (`TurretAI.cs:22-32`):
    #
    #     if (!baseTurret.IsAttacking) { CheckForTargets(); }      // may SET one
    #     if (TargetUnit != null && DistanceSquared(...) > Range*Range)
    #         baseTurret.SetTargetUnit(null, true);                // may DROP it
    #
    # Acquisition is a quadtree circle-vs-circle query and so reaches
    # `Range + candidate CollisionRadius` (790 for a lane minion); retention is
    # a flat centre-to-centre `Range` (750); selection never consults distance.
    # A best-priority minion in that annulus is therefore picked and dropped in
    # the same update, every tick, and the turret stands idle at cooldown 0
    # with closer minions in range -- 664 consecutive ticks of it, measured.
    # Scoring `left_range` against the INCOMING target (as this did) makes the
    # trap unreachable no matter how wide acquisition is: the sim would simply
    # keep the annulus minion. Both halves are needed, and neither alone is a
    # partial fix -- widening acquisition alone makes turret targeting worse.
    cur = jnp.clip(turret_pick, 0, n - 1)
    left_range = (turret_pick >= 0) & (
        ((x[cur] - x) ** 2 + (y[cur] - y) ** 2)
        > P("attack_range") ** 2)
    target_gone = (turret_pick >= 0) & ~visible[cur]
    turret_target = jnp.where(left_range | target_gone, jnp.int8(-1),
                              turret_pick)

    # `TGT-NULLOUT`: `ObjAIBase.UpdateTarget` runs for EVERY `ObjAIBase`, lane
    # minions included, AFTER the AI script and every tick -- and it nulls a
    # target that is dead, untargetable-and-useable, or no longer visible
    # (`ObjAIBase.cs:1203-1211`). This port applied it to champions and turrets
    # only, so a minion kept a target the server had already dropped until its
    # own 250 ms sweep came round, which can be fifteen ticks later.
    #
    # Measured, not assumed: joining every target disagreement to the server's
    # own branch stream, 21 of 216 rows are cases where the minion's script gate
    # never opened at all -- and all 21 carry the same verdict, `server DROPPED
    # an incumbent the sim kept`. The script cannot have dropped it with the
    # gate shut, so the engine did. The other 194 are `ORDER-003`.
    #
    # Deliberately NOT the script's `IsValidTarget`, which also tests
    # acquisition range: the engine pass has no range term, and copying the
    # wider predicate here would drop targets the server keeps and trade one
    # residual for its mirror image.
    mtgt = jnp.clip(ai.target, 0, n - 1)
    minion_target = jnp.where(
        (ai.target >= 0) & (~state.alive[mtgt] | ~visible[mtgt]),
        jnp.int8(-1), ai.target)

    target = jnp.where(
        is_champ, jnp.where(keep_champ, state.target, champ_pick),
        jnp.where(is_turret, turret_target, minion_target))

    # ---- 14. RefreshWaypoints -----------------------------------------------
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
    # `ENT-01`. `UpdateTarget`'s swing/chase/hold branch sits inside
    #     if (TargetUnit != null && TargetUnit.Team != Team && ...)
    # (`ObjAIBase.cs:1285`). `LanerlControl` accepts an ALLY as a target with
    # no team check, and the server then holds it and does nothing: no
    # swing, no chase, no hold. Its one effect is to release a sticky enemy
    # target, which is the server's only disengage, so the order is kept
    # rather than refused. The port had no team test anywhere on this path:
    # an ATTACK on an allied minion chased it, swung, killed it, and
    # `death_rewards` paid the killer gold and CS for it (measured: +20 gold,
    # +1 CS in 22 ticks). Real League cannot target allies at all.
    hostile = has_tgt & (state.team[tgt] != state.team)
    # `ENT-02`. The hit lands on the unit the SWING STARTED ON, not the
    # current target: `Spell.FinishCasting` deals melee damage to
    # `CastInfo.Targets[0].Unit` (`Spell.cs:1030`), a ranged swing's missile
    # carries its own `TargetUnit`, and `SetTargetUnit` rewrites neither.
    # Only the CANCEL test (`ObjAIBase.cs:1247`, out of range of the current
    # target) reads `TargetUnit` during a wind-up. Resolving the hit against
    # `target` let a policy re-aim a swing on its last frame onto whichever
    # minion had just become killable -- zero-wind-up last-hitting, which the
    # server does not do (measured: minion B died, A untouched). A swing
    # carried in with no recorded target (hand-built test states, and
    # checkpoints from before the field existed) falls back to `target`.
    hit_target = jnp.where(state.aa_target >= 0, state.aa_target, target)
    hit_tgt = jnp.clip(hit_target, 0, n - 1)
    # `AA-007`: the swing's declared target is gone -- it died on an earlier
    # tick (phase 19 replaced the index with `AA_TARGET_GONE` then, so a
    # recycled slot cannot be mistaken for it), or a hand-built / injected
    # state carries a dead index. `step_autoattack` cancels such a swing at
    # `CastCancelCheck`'s position, before the wind-up can complete.
    swing_target_gone = (state.aa_target == AA_TARGET_GONE) | (
        (state.aa_target >= 0) & ~state.alive[jnp.clip(state.aa_target, 0, n - 1)])

    # `UpdateTarget` DOES NOT REACH `RefreshWaypoints` DURING A WINDUP.
    # `ObjAIBase.cs:1192-1205` is an early return that fires on the
    # `IsAttacking` the unit carried INTO this tick:
    #
    #     else if (IsAttacking)
    #     {
    #         if (dist > Range + tgt.CollisionRadius && State == STATE_CASTING
    #             && !CantCancelWhileWindingUp) CancelAutoAttack(...);
    #         if (AutoAttackSpell.State == STATE_READY) IsAttacking = false;
    #         return;                       // <-- never reaches :1239-1242
    #     }
    #
    # so neither the in-range `UpdateMoveOrder(Hold)` (`:651-655`) nor the
    # out-of-range re-path (`:657-669`) happens while a swing is in flight.
    # The order the unit is left with is whatever its AI script wrote earlier
    # in the SAME tick -- and `LaneMinionAI`'s 250 ms timer writes `AttackTo`.
    # That is the whole of the largest gate-1 residual: 5,390 of 5,423 drilled
    # move-order mismatches were `sim=HOLD, server=ATTACK_TO`, and the dumped
    # order for minion 1073743551 flips to AttackTo on exactly the tick
    # `aitimer` resets and stays there for as long as `aastate=1`.
    #
    # `state.is_attacking` is the right value to read: `IsAttacking` is
    # written only inside `UpdateTarget` itself (and by `CancelAutoAttack`,
    # whose callers all null the target first), so the flag this branch tests
    # is the one the tick began with, not the one `step_autoattack` returns
    # below. Reading the post-swing flag instead would re-admit exactly the
    # firing tick, which is the one tick the server DOES write Hold on.
    #
    # Turrets are excluded because `BaseTurret.RefreshWaypoints`
    # (`BaseTurret.cs:108-110`) is an empty override -- "Overridden function
    # unused by turrets" -- so a turret's move order is never touched by this
    # path at all and simply persists.
    #
    # `ENT-10`: nor does it reach one during a recall. `RefreshWaypoints`
    # promotes the order to `AttackTo` only when `_castingSpell == null &&
    # ChannelSpell == null` (`ObjAIBase.cs:622-625`); the recall left the
    # order at `Stop`, so it then finds no `targetPos` and returns
    # (`:635-668`) -- no chase, no Hold, no route reset. Writing `ATTACK_TO`
    # here made the next tick's `ChannelCancelCheck` (phase 9) cancel the
    # recall in the sim only. `recall_windup`/`recall_channel` are the
    # post-`Spell.Update` values, which is what `UpdateTarget` sees.
    in_recall = (recall_windup > 0) | (recall_channel > 0)
    refresh = (hostile & ~state.is_attacking & (state.kind != Kind.TURRET)
               & ~in_recall)
    hold = refresh & in_rng
    chase = refresh & ~in_rng
    wp = wp_after_lane
    two = jnp.stack([jnp.stack([x, y], -1), jnp.stack([x[tgt], y[tgt]], -1)], 1)
    wp = jnp.where(chase[:, None, None],
                   wp.at[:, :2].set(two)[:, :, :], wp)
    wp_key = jnp.where(chase, jnp.int8(1), wp_key_after_lane)
    n_wp = jnp.where(chase, jnp.int8(2), n_wp_after_lane)
    # `HOLD-001`. The in-range branch does not only write an order.
    # `UpdateMoveOrder(OrderType.Hold, true)` (`ObjAIBase.cs:1362-1366`) calls
    # `StopMovement()`, which for a non-dashing unit is
    # `AttackableUnit.ResetWaypoints` (`AttackableUnit.cs:996-1002`):
    #
    #     Waypoints = new List<Vector2> { Position };
    #     CurrentWaypointKey = 1;
    #
    # so the route is DESTROYED, not merely ignored while the order holds.
    # That distinction is invisible for exactly as long as the order stays
    # Hold -- `_can_move` blocks Hold, so the unit does not move either way --
    # and it stops being invisible the moment something writes the order back.
    # Something does, every 250 ms: `LaneMinionAI.ReevaluateBehavior`
    # (`LaneMinionAI.cs:321-331`) returns `AttackTo` for a still-valid target
    # and `UpdateMoveOrder(AttackTo)` touches no waypoints at all. On the
    # server the minion stays where it stopped, because its list is
    # `[Position]`. Without this reset the sim resumed walking down the stale
    # two-point chase path the instant the order flipped back, and kept
    # walking for the whole windup and cooldown, because `UpdateTarget`'s
    # `IsAttacking` early return above never re-paths either.
    #
    # Measured (`parity/tier15.py`, q2 `engaged`, minion 1073744061): from an
    # identical injected state the two sides were bit-identical for 33 ticks;
    # at tick 34 the server stopped for good at (3831.250, 13191.562) with
    # `wps=1` while the sim, agreeing on move order, target, `is_attacking`
    # and the auto-attack clock to four decimals, kept stepping 5.427 u/tick
    # with `wps=2` and was 48.8 u away eight ticks later. Position was the
    # ONLY field that disagreed, which is why no controller-state row ever
    # caught it.
    hold_here = jnp.stack([x, y], -1)
    wp = jnp.where(hold[:, None, None], wp.at[:, 0].set(hold_here), wp)
    wp_key = jnp.where(hold, jnp.int8(1), wp_key)
    n_wp = jnp.where(hold, jnp.int8(1), n_wp)
    # ---- 15. the swing gate and auto-attack clock; Q's hit and silence ------
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
    ad_now = P("attack_damage") + P("ad_per_level") * level_growth
    # `Stats.LevelUp` adds `GrowthAttackSpeed / 100` to
    # `AttackSpeedMultiplier.PercentBaseBonus` through the same non-linear
    # curve. With no reachable items/buffs that alter AS, `Stat.Total` is this
    # multiplier exactly; both period and normal-attack windup scale by it.
    attack_speed_multiplier = (
        1.0 + (P("attack_speed_per_level") / 100.0)
        * level_growth)
    attack_period_now = P("attack_period") / attack_speed_multiplier
    attack_windup_now = P("attack_windup") / attack_speed_multiplier
    raw_ad = _attack_damage_against(
        ad_now, state.kind, state.kind[hit_tgt], state.model, t_now)
    aa = step_autoattack(
        state.aa_cooldown, state.aa_windup, state.is_attacking,
        state.has_auto_attacked,
        in_range=in_rng,
        # `SetStatus(CanAttack, false)` for Judgment's duration, and no
        # swing through a recall or R's windup -- the post-`UpdateBuffs`
        # status, with this tick's recall/R timers.
        can_attack=status(
            bs.buffs, alive=state.alive, spell_level=state.spell_level,
            spell_cooldown=bs.spell_cooldown, silenced_ms=state.silenced_ms,
            recall_windup_ms=recall_windup, recall_channel_ms=recall_channel,
            r_cast_ms=r_cast_ms).can_attack,
        has_target=has_tgt,
        attack_period=attack_period_now,
        windup_time=attack_windup_now,
        attack_damage=raw_ad,
        target_resist=armor_eff[hit_tgt],
        # Q's first post-cast gate is intentionally skipped; the following
        # swing uses GarenQAttack's complete replacement damage, not normal AD
        # plus an extra component.
        empowered_attack=bs.q_empowered,
        empowered_damage=q_damage_at_rank(state.spell_level[:, Slot.Q], ad_now),
        skip_next_autoattack=bs.q_skip_next,
        may_engage=hostile,
        swing_target_gone=swing_target_gone,
        delta_ms=delta_ms, xp=jnp)

    q_landed = aa.hit & bs.q_empowered
    # `GarenQAttack.OnSpellPostCast` -> `OnSpellEnd` deactivates the live
    # GarenQ buff: its `OnDeactivate` is `end_q`, the same one expiry calls.
    # Called AFTER `step_buffs`' countdown: `GarenQAttack` (slot 45) updates
    # after `GarenQ` (slot 0), so the hit row shows exactly 8.0 and the name
    # still LISTED (`SPELL-013`).
    buffs_out,spell_cooldown_out = end_q(bs.buffs, bs.spell_cooldown, q_landed)
    # `SkipNextAutoAttack` is consumed at the swing gate, before the real
    # GarenQAttack swing begins.
    buffs_out = consume_q_skip(buffs_out, aa.consumed_skip)
    # GarenQAttack applies silence to the unit hit. Status duration is carried
    # explicitly so subsequent semantic cast orders fail exactly while the
    # server's CanCast flag is suppressed. Multiple simultaneous Q hits use
    # the longest duration, matching independent status applications.
    silence_left = jnp.maximum(
        state.silenced_ms - jnp.asarray(delta_ms, dtype), 0.0)
    silence_by_attacker = (
        q_silence_duration_at_rank(state.spell_level[:, Slot.Q]) * 1000.0)
    silence_added = jnp.zeros_like(silence_left).at[hit_tgt].max(
        jnp.where(q_landed, silence_by_attacker, 0.0))
    silenced_ms = jnp.maximum(silence_left, silence_added)

    # ---- 16. apply damage (melee hits, missiles, buff damage) ---------------
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
    swings = aa.hit & state.alive & (hit_target >= 0)
    ranged = P("fires_missile") > 0
    launches = swings & ranged
    landed = swings & ~ranged

    ms = step_missiles(
        m_alive=state.missile_alive, m_x=state.missile_x, m_y=state.missile_y,
        m_target=state.missile_tx, m_source=state.missile_source,
        m_damage=state.missile_damage, m_speed=state.missile_speed,
        launches=launches, raw_damage=raw_ad, launch_speed=P("missile_speed"),
        x=x, y=y, alive=state.alive,
        targetable=state.alive, armor=armor_eff, target=hit_target,
        delta_ms=delta_ms,
        m_source_seq=state.missile_source_seq,
        m_source_model=state.missile_source_model,
        spawn_seq=state.spawn_seq, model=state.model)

    dmg_ij = jnp.where(landed[:, None] & (jnp.arange(n)[None, :] == hit_tgt[:, None]),
                       aa.damage[:, None], jnp.zeros((n, n), dtype))
    # A missile that lands this tick is credited to the unit that FIRED it, in
    # that unit's own attacker row, so the lowest-index-crosses-zero rule below
    # sees melee hits and missile hits in one ordering rather than two.
    dmg_ij = dmg_ij + ms.damage_ij
    # `SLOT-003`: a missile whose shooter's slot was recycled while it flew.
    # The server lands it and credits the DEAD owner object (see
    # `step_missiles`). Its damage is real -- hp, W's multiplier, a recall
    # interrupt -- but it has no row: `dmg_ij`'s row for that slot is the NEW
    # occupant's, and routing it there credited the kill, the champion
    # hit-flag and a call for help to a unit that did not fire it (and is
    # usually on the victim's own team). What each consumer should see
    # instead is written where it is read below. The shooter is always a
    # lane minion: only `spawn_minion` recycles a slot.
    orphan_dmg = jnp.zeros((n,), dtype).at[ms.victim].add(ms.orphan_damage)

    # ---- 17. Champion._championHitFlagTimer / _playerHitId ------------------
    # `Champion.TakeDamage` (`Champion.cs:569-575`) resets these on EVERY hit
    # this champion takes, from any source -- no melee/caster-minion exemption
    # like the passive's combat clock below. Read here, off the real (not yet
    # Judgment-prepended) `dmg_ij`, because `champion_kill_rewards` needs it
    # for THIS tick's own death resolution (`Champion.Die` reads whatever the
    # flag holds as of when it runs, which is after every `TakeDamage` call
    # that landed this tick, including the killing one itself).
    hit_flag_ms, hit_flag_by = update_hit_flag(
        kind=state.kind, damage_ij=dmg_ij, buff_damage=bs.damage_dealt,
        buff_dealt_by=bs.dealt_by, hit_flag_ms=state.hit_flag_ms,
        hit_flag_by=state.hit_flag_by, delta_ms=delta_ms,
        orphan_damage=orphan_dmg)

    # W's active scales the victim's TOTAL incoming damage for the tick, so it
    # is applied here rather than per attacker -- one multiply on the sum, not
    # one per source, which is what `TakeDamage`'s post-mitigation hook does.
    dealt = (dmg_ij.sum(axis=0) + orphan_dmg + bs.damage_dealt) \
        * bs.damage_multiplier
    hp = jnp.maximum(state.hp - dealt, jnp.zeros_like(state.hp))

    # `Buffs/Global/Recall` listens only while its 7.9 s buff exists. Auto
    # attacks/missiles are non-periodic; E is periodic and must NOT interrupt
    # a base. R's pending hit is a regular spell hit and does. The listener
    # sets a latch now which its OnUpdate consumes at the top of the next tick.
    r_hit = state.buffs.r_pending.active & (bs.damage_dealt > 0)
    nonperiodic_hit = (dmg_ij.sum(axis=0) > 0) | (orphan_dmg > 0) | r_hit
    recall_listener_live = recall_channel > (_RECALL_CHANNEL_MS - _RECALL_DAMAGE_BUFF_MS)
    recall_damage_pending = (
        state.recall_damage_pending
        | (recall_listener_live & nonperiodic_hit))

    # ---- 18. out-of-combat clock, for Garen's passive -----------------------
    # `CharScriptGaren.ShouldPassiveTurnOff(unit, damageData)` (`unit` = the
    # DEFENDER this passive belongs to, i.e. Garen; `damageData.Attacker` =
    # whoever hit him) returns FALSE -- the passive keeps running, the hit
    # does NOT count as combat -- when EITHER:
    #  (a) `MINION_UNIT_TAG_PASSIVE_EXCEPTIONS.Contains(Attacker.UnitTags)`,
    #      an EXACT-VALUE check against the attacker's raw, already-OR'd
    #      `UnitTags` int; or
    #  (b) `unit.Stats.Level >= 11 && UnitTag.Monster.Equals(Attacker.UnitTags)`
    #      -- the DEFENDER's own level, re-exempting at 11+.
    #
    # `UnitTag` is `[Flags]` with NO explicit values (confirmed directly from
    # `GameServerCore/Enums/UnitTag.cs`), so C# numbers it sequentially:
    # Champion=0, Champion_Clone=1, Minion=2, Minion_Lane=3,
    # Minion_Lane_Siege=4, Minion_Lane_Super=5, Minion_Summon=6, Monster=7.
    # `MINION_UNIT_TAG_PASSIVE_EXCEPTIONS` (`CharScriptGaren.cs:21-28`) lists
    # `{Minion, Minion_Lane, Minion_Lane_Siege, Minion_Lane_Super,
    # Minion_Summon}` BY NAME -- i.e. the raw values `{2,3,4,5,6}` -- but a
    # real minion's `UnitTags` field is the bitwise OR of ALL its tags, not a
    # single one: melee/caster (`Blue_Minion_Basic.json`) is
    # `"Minion | Minion_Lane"` = `2|3` = **3** (in the set, so always exempt,
    # matching `Minion_Lane`'s own raw value by coincidence). Cannon
    # (`Blue_Minion_MechCannon.json`) is
    # `"Minion | Minion_Lane | Minion_Lane_Siege"` = `2|3|4` = **7**, and
    # super (`Blue_Minion_MechMelee.json`) is
    # `"Minion | Minion_Lane | Minion_Lane_Super"` = `2|3|5` = **7** too --
    # BOTH collide with `Monster`'s raw value, and 7 is not in `{2,3,4,5,6}`,
    # so despite `Minion_Lane_Siege`/`Minion_Lane_Super` being named right
    # there in the exceptions list, NEITHER cannon NOR super minions are ever
    # exempted by check (a). Only check (b) -- the Monster-value collision,
    # gated on Garen's OWN level -- can exempt them, and only from level 11.
    #
    # Reproduced deliberately, including the level-11 re-exemption this port
    # previously omitted and the super-minion case this port previously
    # mis-grouped with melee/caster (never breaking combat, at any level) --
    # `docs/PORT_AUDIT_COMBAT.md`'s UnitTag row. It is a bug in the server and
    # parity means matching the server, not the mechanic's evident intention.
    # `combat.garen_passive_exempt` is the (attacker, victim) pair; the
    # exemption is gated on the VICTIM's (defender's) own level, not the
    # attacker's, which is why this can't be a per-attacker vector.
    _minion_type = _minion_type_of(state)
    _is_lane_minion = state.kind == Kind.LANE_MINION
    _is_cannon_or_super = (
        (_minion_type == MinionType.CANNON) | (_minion_type == MinionType.SUPER))
    breaks_combat_pair = ~garen_passive_exempt(
        _is_lane_minion, _is_cannon_or_super, state.level, jnp)
    # `dmg_ij` here is (attacker, victim) and NOT yet the concatenated form
    # that prepends the buff-damage row further down. Judgment's damage is
    # carried separately in `bs.damage_dealt` and comes from a champion, so it
    # always counts as combat.
    # `SLOT-003`: an orphaned missile's `Attacker` is the dead minion, whose
    # `UnitTags` are its own -- read from the model recorded at launch, not
    # from the slot's new occupant. PRE-launch arrays: `ms.orphan_damage` is
    # indexed by the missile slot as it was before this tick's launches.
    o_type = _MINION_TYPE_TABLE[state.missile_source_model]
    o_siege = (o_type == MinionType.CANNON) | (o_type == MinionType.SUPER)
    # `garen_passive_exempt` for a lane-minion attacker, per missile (the
    # helper builds the full attacker x victim matrix): exempt unless it is a
    # cannon/super and the victim is below level 11.
    o_breaks = o_siege & (state.level[ms.victim] < 11)
    orphan_combat = jnp.zeros((n,), bool).at[ms.victim].max(
        (ms.orphan_damage > 0) & o_breaks)
    hit_by_combat = ((
        jnp.where(breaks_combat_pair, dmg_ij, 0.0).sum(axis=0)
        + bs.damage_dealt) > 0) | orphan_combat
    ms_since_damaged = jnp.where(
        hit_by_combat, jnp.zeros_like(state.ms_since_damaged),
        state.ms_since_damaged + delta_ms)

    alive = state.alive & (hp > 0)
    died = state.alive & ~alive

    # ---- 19. the dead drop target and swing (UpdateTarget; TGT-DEATHTICK) ---
    # The null-out in phase 13 above runs before damage, so a unit killed this
    # tick is still `alive` for the whole of this tick's targeting and is only
    # dropped on the NEXT one. The server's order is the reverse, and it is
    # per-unit: `Spell.Update` resolves auto-attack damage and sets `IsDead`
    # synchronously (`AttackableUnit.cs:588-592`), and `UpdateTarget` runs
    # after it inside the same `ObjAIBase.Update` (`ObjAIBase.cs:1142`, then
    # `:1153`). So the port was exactly one tick late on every death-driven
    # target drop.
    #
    # MEASURED, not argued. Each of the three fields has one write site, and
    # those sites now carry `caller=<member>@<line>` (`patch_writesites.py`).
    # Joining the write stream to the death stream on the canonical corpus:
    #
    #   Untarget<StopTargeting<Die   204 events, ALL at +0 ms from the death
    #   UpdateTarget (invalid target) 18 at +0 ms, 13 at -16/-17 ms
    #   ReevaluateBehavior (script)   33 at +0 ms, 20 at -16 ms
    #
    # Nothing lands late. The negative lags are not an error: `IsDead` flips
    # during damage, whereas `Die()` -- and so the broadcast that the +0 ms
    # events come from -- is deferred to the VICTIM's own update, which may
    # be the following tick. Units reading the flag beat the broadcast to it.
    #
    # Why re-apply here rather than move the phase-13 block: the server's
    # interleaving is per-unit (`A.damage, A.target, B.damage, B.target`) and
    # no vectorised tick can reproduce that ordering exactly. Applying the
    # same predicate twice -- once on start-of-tick state, once on
    # post-damage state -- is the closest total order, and it is idempotent,
    # so the phase-13 pass costs nothing where it already fired.
    # MEASURED AND REVERTED. Applying it cost 50 target rows (522 -> 572) and
    # 5 turret rows (30 -> 35) on the canonical corpus, with all 25 other
    # fields bit-identical. Two reasons, both visible in the C# once the
    # result forced a re-read:
    #
    #   * The server's clear is ORDER-DEPENDENT. `IsDead` is set inside the
    #     KILLER's `Spell.Update`, so only units whose own `UpdateTarget` runs
    #     later in the same `foreach` (`ObjectManager.cs:79-82`) see it that
    #     tick. A vectorised pass applies it to everyone, which is strictly
    #     more eager than the server for every unit ordered before the killer.
    #   * It makes `target` DOWNSTREAM of the fire decision. Tier 1 re-injects
    #     state every tick, so the sim can only retire a unit the server keeps
    #     by dealing damage the server did not deal in that same tick -- i.e.
    #     an `AA-004` fire error. Measured: one such tick at t=133343 turned a
    #     single early fire into 75 minions plus 13 turrets all dropping the
    #     same still-alive target.
    #
    # Fidelity to the source is not the criterion; agreement with the server
    # is. Keeping the unconditional form would have traded a measured 522 for
    # a measured 572 in exchange for a better-looking call graph.
    #
    # dtgt = jnp.clip(target, 0, n - 1)
    # target = jnp.where((target >= 0) & ~alive[dtgt], -1, target)

    # `ObjAIBase.UpdateTarget`'s FIRST branch (`ObjAIBase.cs:1219-1227`): a
    # unit that is itself dead drops its own target. 98 calls on the corpus
    # (`caller=UpdateTarget@1224`), and entirely absent from the port -- a
    # dead unit carried its target through death AND through respawn, because
    # the respawn block resets `silenced_ms`/`r_cast_ms`/buffs and never
    # touches `target`.
    #
    # `AA-006`, in full. That branch is
    #     if (TargetUnit != null) { CancelAutoAttack(true, true); SetTargetUnit(null, true); }
    # and `CancelAutoAttack(reset: true, fullCancel: true)` (`ObjAIBase.cs:
    # 469-484`) does four things: spell back to READY, `_autoAttackCurrent
    # Cooldown = 0` and `ResetSpellCast()` (reset), `IsAttacking = false`
    # (fullCancel). It does NOT touch `HasAutoAttacked`. The wind-up is
    # cleared either way: a corpse still `STATE_CASTING` is `ResetSpellCast`
    # by `CastCancelCheck`'s owner-dead branch (`Spell.cs:215-217`) even with
    # no target. Only the cooldown depends on holding a target, and a unit
    # mid-swing always holds one here (`step_autoattack` cancels a swing
    # whose target is null). Applied at the corpse's death tick, as the
    # `target` null is; the stored values are below.
    cancel_on_death = died & (target >= 0)
    target = jnp.where(~alive, -1, target)
    # The swing's own target: fixed at the start tick, held for the wind-up,
    # cleared when the swing ends however it ends (`aa.is_attacking` is
    # false after a hit, a cancel, or a consumed skip) or the unit dies.
    aa_target = jnp.where(
        aa.start, target,
        jnp.where(aa.is_attacking, hit_target, jnp.int8(-1)))
    # `AA-007`: a swing whose declared target is dead at the end of this tick
    # (it died this tick, in practice) drops the index now, on the death
    # tick, for `AA_TARGET_GONE`. The server's reference is
    # to an OBJECT (`CastInfo.Targets[0].Unit`): the corpse stays that object
    # and a recycled slot is a different one. Keeping the slot index let the
    # next wave's spawn -- which runs at the top of the next tick, before
    # the swing is resolved -- turn it into a live unit: a red caster's swing
    # on a dead blue minion then fired at the newborn red caster at its
    # barracks, 8-11k units away (`docs/PLAYTEST_SWEEP.md` (c)1).
    #
    # The CANCEL itself stays on the attacker's next update, as the `target`
    # drop does (`TGT-DEATHTICK`, measured and reverted above): it is
    # `CastCancelCheck` in the attacker's own `Spell.Update`, which sees the
    # death on the same tick only when it is ordered after the killer --
    # and missiles, which land most kills, are created late and update late.
    aa_tgt_c = jnp.clip(aa_target, 0, n - 1)
    aa_target = jnp.where((aa_target >= 0) & ~alive[aa_tgt_c],
                          jnp.int8(AA_TARGET_GONE), aa_target)
    aa_target = jnp.where(alive, aa_target, jnp.int8(-1)).astype(
        state.aa_target.dtype)

    # ---- 20. call for help --------------------------------------------------
    # `targeting.call_for_help_map` implements the broadcast faithfully and is
    # The toggle is retained for ablations.  Production defaults ON: the server
    # has no corresponding switch, so using an aggregate rollout regression to
    # suppress this source-verified mechanic only hid another parity defect.
    if enable_call_for_help:
        # `ObjAIBase.TakeDamage`'s broadcast reacts to every landed hit --
        # melee autoattacks and missiles are both folded into `dmg_ij` above.
        # Orphaned missiles (`SLOT-003`) are not: the server's call names the
        # DEAD shooter, and `LaneMinionAI` never acquires a dead unit
        # (`IsValidTarget`, `LaneMinionAI.cs:125-135`), so it changes nothing.
        # Buff damage is carried victim-wise, so scatter it into its caster's
        # row before broadcasting.  This covers Judgment as well as the E tick.
        buff_src = jnp.clip(bs.dealt_by, 0, n - 1)
        buff_damage_ij = jnp.zeros_like(dmg_ij).at[
            buff_src, jnp.arange(n)
        ].add(jnp.where(bs.dealt_by >= 0, bs.damage_dealt, 0.0))
        help_priority = call_for_help_map(
            damage_ij=dmg_ij + buff_damage_ij,
            x=x, y=y, alive=alive, kind=state.kind,
            team=state.team, acquisition_range=P("acquisition_range"),
            minion_type=_minion_type_of(state))
    else:
        help_priority = state.help_priority


    # ---- 21. kill attribution and death rewards (AttackableUnit.Die) --------
    # Judgment's damage is applied inside the buff's own update, which runs
    # BEFORE the auto-attack gate, so it is prepended to the attribution order.
    # `SLOT-003`: orphaned missile damage (shooter's slot recycled in flight)
    # is APPENDED as a last row. Its killer is a dead minion, which neither
    # `death_rewards` nor `champion_kill_rewards` can pay (both require a
    # champion killer), so it is recorded as -1 rather than as whichever
    # unit now occupies the slot. Last, not in the slot's row, because that
    # row's position is the NEW occupant's; the order among same-tick hits
    # is the sim's own approximation either way (see above).
    dmg_ij = jnp.concatenate(
        [jnp.zeros((1, n), dtype).at[0].set(bs.damage_dealt), dmg_ij,
         orphan_dmg[None, :]], axis=0)
    cum = jnp.cumsum(dmg_ij, axis=0)                     # (attacker, victim)
    crosses = (cum >= state.hp[None, :]) & (dmg_ij > 0)
    first_row = jnp.argmax(crosses, axis=0)
    # row 0 is the Judgment lane; map it back to whoever cast the spin
    killer = jnp.where(
        jnp.any(crosses, axis=0),
        jnp.where(first_row == 0, bs.dealt_by,
                  jnp.where(first_row == n + 1, -1, first_row - 1)),
        -1).astype(jnp.int8)
    killer = jnp.where(died, killer, jnp.int8(-1))

    rw = death_rewards(
        died=died, killer=killer, x=x, y=y, team=state.team, kind=state.kind,
        alive=alive, gold_on_death=P("gold_on_death"),
        xp_on_death=P("xp_on_death"))

    # ---- 22. champion-kill gold/XP (Champion.Die) ---------------------------
    # `death_rewards` above is `AttackableUnit.Die`'s path -- minions (and,
    # numerically inertly, turrets) only. `Champion` overrides `Die` entirely
    # and never calls `base.Die`, so THIS is the only thing that ever pays
    # gold or XP for a champion kill. See `rewards.champion_kill_rewards`'s
    # docstring for the full formula and its citations.
    ckr = champion_kill_rewards(
        died=died, kind=state.kind, level=state.level, killer=killer,
        hit_flag_ms=hit_flag_ms, hit_flag_by=hit_flag_by,
        kill_spree=state.kill_spree, death_spree=state.death_spree,
        gold_from_minions=state.gold_from_minions,
        first_blood_done=state.first_blood_done,
        kill_exp_table=params["champion_kill_exp"])

    # ---- 23. turret-destruction gold/XP (LaneTurret.Die) --------------------
    tk_gold, tk_xp = turret_kill_rewards(
        died=died, kind=state.kind, team=state.team, alive=alive, x=x, y=y,
        local_gold=P("local_gold_on_death"), global_gold=P("global_gold_on_death"),
        global_xp=P("global_xp_on_death"), attack_range=P("attack_range"))

    # ---- 24. gold, XP, level-up and spell ranks -----------------------------
    # `Champion.OnKill`'s minion-kill branch (`Champion.cs:379-388`) is the
    # OTHER place `DeathSpree`/`GoldFromMinions` change; `death_rewards.gold`
    # IS exactly this tick's minion-kill gold (its only source), so it feeds
    # straight in. Applied on top of `ckr`'s own spree/GoldFromMinions output
    # so a champion who lands BOTH a champion kill and a minion kill in the
    # exact same tick sees the champion-kill reset (`GoldFromMinions=0`)
    # first, then this tick's minion gold accumulate from zero -- one
    # deterministic order for an astronomically rare simultaneous case, not a
    # server-verified one.
    gold_from_minions, death_spree = minion_gold_deathspree_decay(
        minion_gold=rw.gold, death_spree=ckr.death_spree,
        gold_from_minions=ckr.gold_from_minions)

    # The clock BEFORE this tick's advance: the server's IsGeneratingGold is
    # set one tick before the first payment (`ambient_gold`).
    amb, gold_timer = ambient_gold(state.t_ms, state.gold_timer,
                                   state.kind == Kind.CHAMPION, delta_ms)
    gold = state.gold + rw.gold + amb + ckr.gold + tk_gold
    xp = state.xp + rw.xp + ckr.xp + tk_xp
    cs = state.cs + rw.cs.astype(state.cs.dtype)
    kills = state.kills + ckr.kills
    level = jnp.where(state.kind == Kind.CHAMPION,
                      level_for_xp(xp, params["xp_to_reach_level"]), state.level)
    # Stats.LevelUp raises both maximum and current HP by the same nonlinear
    # growth increment. The cumulative-curve difference also handles a rare
    # multi-level XP jump without a Python loop.
    hp_growth = P("hp_per_level") * (
        growth_sum(level, jnp) - level_growth)
    max_hp = state.max_hp + hp_growth
    hp = hp + hp_growth
    # Spell ranks are a pure function of champion level under a fixed skill
    # order, so they need no state of their own. The server spends the points
    # through `AutoLevelUndriven` / `Champion.LevelUpSpell`; the order is the
    # one in `constants.GAREN_SKILL_ORDER`, walked forward past entries the
    # `SpellsUpLevels` gate refuses (`spells.ranks_for_level`, `SPELL-011`).
    spell_level = jnp.where(
        (state.kind == Kind.CHAMPION)[:, None],
        _RANK_TABLE[jnp.clip(level.astype(jnp.int32), 0, 18)],
        state.spell_level)
    # `W.OnLevelUpSpell` installs its permanent passive synchronously when W
    # first receives a rank.  ``step_buffs`` correctly applies the passive to
    # incoming ranks, but level-up happens later in this tick after XP is
    # awarded.  Commit the marker with the newly derived ranks so the returned
    # state (and therefore the next policy observation) does not spend one
    # frame reporting level-three W with level-two armor/MR.
    gained_w_passive = ((state.kind == Kind.CHAMPION)
                        & (spell_level[:, Slot.W] >= 1))
    buffs_out = buffs_out.replace(
        w_passive=buffs_out.w_passive | gained_w_passive)

    # ---- 25. move order out, and FinishCasting's Hold -----------------------
    minion_order = jnp.where(lane_stop, jnp.int8(MoveOrder.STOP), ai.move_order)
    move_order_out = jnp.where(
        hold, jnp.int8(MoveOrder.HOLD),
        jnp.where(chase, jnp.int8(MoveOrder.ATTACK_TO),
                  jnp.where(is_champ, state.move_order, minion_order)))

    # `ORDER-005`. **`Spell.FinishCasting` writes the caster's move order, and
    # an auto-attack is a cast.**  The tail of `FinishCasting`
    # (`Spell.cs:1051-1065`) is not spell-specific::
    #
    #     if (SpellData.Flags.HasFlag(SpellDataFlags.InstantCast)) { ...MoveTo/AttackTo... }
    #     else { CastInfo.Owner.UpdateMoveOrder(OrderType.Hold, true); }
    #
    # and it runs for **every** completed cast, after the `IsAutoAttack` branch
    # above it has already set `HasAutoAttacked`, applied the hit (melee) or
    # created the missile (ranged), and put the spell back in `STATE_READY`.
    # `SRU_OrderMinionMeleeBasicAttack` carries `Flags = 232448`; bit 2
    # (`InstantCast`, `SpellDataFlags.cs:13`) is **clear**, so a lane minion
    # takes the `else` -- its move order goes to `Hold` on the tick its wind-up
    # runs out, and `UpdateMoveOrder(Hold)` is `StopMovement()`, i.e.
    # `ResetWaypoints` (`HOLD-001`), so the route is destroyed with it.
    #
    # This tick can never collide with `RefreshWaypoints` above: `aa.hit`
    # implies the unit entered the tick `IsAttacking`, which is exactly the
    # condition `ORDER-002`'s early return fires on, so `hold`/`chase` are
    # already False here.  It *does* override `minion_order`, and must:
    # `AIScript.OnUpdate` runs BEFORE `Spells.Update` in `ObjAIBase.Update`,
    # so a 250 ms sweep that wrote `AttackTo` earlier in this same tick is
    # overwritten by this.
    #
    # The sim already ported this exact tail for Garen's R (`r_cast_finished`
    # below, which is the same three lines of `FinishCasting`) and simply
    # never routed the auto-attack through it.  Measured cost of that:
    # **2,074 of the 4,791 whole-corpus LaneMinion move-order misses** are the
    # single shape `pre=AttackTo, sim=AttackTo, server=Hold` on a tick whose
    # wind-up completes, and 2,050 of them are in the "wind-up COMPLETES this
    # tick" bucket.  The direction is absolute: server-wrote-sim-held 4,769
    # against sim-wrote-server-held 22.
    finish_casting = r_cast_finished | aa.hit

    # ---- 26. champion death and respawn (Champion.Update) -------------------
    # `Champion.Die` sets RespawnTimer = DeathTimes[Level] * 1000; the timer is
    # decremented in `Champion.Update` and `Respawn()` restores FULL health at
    # the spawn point. Minions and turrets do not come back within an episode
    # (turrets are revived only by `LanerlEpisode.RestoreBuildings`, between
    # episodes -- which is itself a bug this project has paid for; see
    # ObjectManager.Update's comment about a map losing four towers over a run).
    is_ch = state.kind == Kind.CHAMPION
    # `Champion.Die` (`Champion.cs:400`) reads `MapData.DeathTimes[Stats.Level]`,
    # and `Package.cs:141-153` fills that list from `i = 1`, so a champion at
    # level L waits `TimeDeadPerLevel.Level(L+1)`. The port once read `Level(L)`
    # and respawned 2.5 s early at every level (`eaa2e77`; measured: the
    # server's level-8 death spanned 27.23 s = Level09, not Level08). Since
    # `STRUCT-005` that shift lives IN the table -- `death_times[L]` is the
    # value for level L, like every level-indexed table -- so the reader
    # indexes with the level itself.
    lvl = jnp.clip(level.astype(jnp.int32), 1, params["death_times"].shape[0] - 1)
    died_ch = died & is_ch
    rt = jnp.where(died_ch, params["death_times"][lvl] * 1000.0, state.respawn_ms)
    rt = jnp.where(rt > 0, rt - jnp.asarray(delta_ms, dtype), rt)
    reborn = is_ch & (state.respawn_ms > 0) & (rt <= 0)
    alive = alive | reborn
    hp = jnp.where(reborn, max_hp, hp)
    x = jnp.where(reborn, state.spawn_x, x)
    y = jnp.where(reborn, state.spawn_y, y)
    # `ENT-12`: `Respawn` moves the champion through `SetPosition(spawnPos)`
    # (`Champion.cs:285-288`), and that setter (`AttackableUnit.cs:195-229`)
    # is not a bare teleport: a route that had ended (`IsPathEnded`,
    # `CurrentWaypointKey >= Waypoints.Count`) is reset to `[Position]`, an
    # unfinished one is re-pathed from the new position to `Waypoints.Last()`.
    # BOOKED APPROXIMATION, as for the chase in 3b: the re-path is a straight
    # two-point route, not `GetPath` + `GetClosestTerrainExit`.
    path_open = wp_key < n_wp
    last_wp = jnp.take_along_axis(
        wp, jnp.clip(n_wp.astype(jnp.int32) - 1, 0, wp.shape[1] - 1)[:, None, None],
        axis=1)[:, 0]
    spawn_xy = jnp.stack([state.spawn_x, state.spawn_y], -1)
    respawn_route = wp.at[:, 0].set(spawn_xy).at[:, 1].set(
        jnp.where(path_open[:, None], last_wp, spawn_xy))
    wp = jnp.where(reborn[:, None, None], respawn_route, wp)
    n_wp = jnp.where(reborn, jnp.where(path_open, jnp.int8(2), jnp.int8(1)),
                     n_wp)
    wp_key = jnp.where(reborn, jnp.int8(1), wp_key)
    rt = jnp.where(reborn, jnp.asarray(-1.0, dtype), rt)
    deaths = state.deaths + died_ch.astype(state.deaths.dtype)
    silenced_ms = jnp.where(died | reborn, 0.0, silenced_ms)
    r_cast_ms = jnp.where(died | reborn, 0.0, r_cast_ms)
    recall_windup = jnp.where(died | reborn, 0.0, recall_windup)
    recall_channel = jnp.where(died | reborn, 0.0, recall_channel)
    recall_damage_pending = jnp.where(died | reborn, False, recall_damage_pending)
    # Death and respawn do NOT touch buffs (`SPELL-006`, fixed). Nothing on
    # the server removes one: `AttackableUnit.UpdateBuffs` keeps ticking a
    # corpse's buffs and neither `Champion.Die` nor `Champion.Respawn`
    # touches them, so a spin or a Q window on a corpse runs out on
    # schedule and its `OnDeactivate` (`end_e`/`end_q`, from `step_buffs`)
    # starts the cooldown -- and `GarenE.OnUpdate` keeps dealing its damage
    # from the corpse. This used to wipe E/Q/W/Q-haste on `died | reborn`
    # with NO cooldown, which made every death a free E and Q reset. R's
    # pending hit is cancelled by its CASTER's death inside `step_buffs`
    # (the generic `CastCancelCheck`), as before.
    # Observation memory records a witnessed cast at ingress (0 ms) and then
    # ages once for every server tick.  The -1 sentinel means "never seen",
    # not a negative elapsed duration, and must survive resets/normal ticking.
    observed_enemy_cast_ms = jnp.where(
        state.observed_enemy_cast_ms >= 0,
        state.observed_enemy_cast_ms + jnp.asarray(delta_ms, dtype),
        state.observed_enemy_cast_ms,
    )

    return state.replace(
        respawn_ms=rt, deaths=deaths,
        t_ms=state.t_ms + jnp.asarray(delta_ms, dtype),
        tick=state.tick + 1,
        x=x, y=y,
        waypoint_key=jnp.where(finish_casting, jnp.int8(1), wp_key),
        lane_waypoint_key=lane_key,
        waypoints=jnp.where(
            finish_casting[:, None, None],
            wp.at[:, 0].set(jnp.stack([x, y], -1)), wp),
        n_waypoints=jnp.where(finish_casting, jnp.int8(1), n_wp),
        target=target.astype(state.target.dtype),
        target_priority=ai.target_priority,
        had_target=ai.had_target,
        move_order=jnp.where(finish_casting, jnp.int8(MoveOrder.HOLD),
                             move_order_out),
        ai_timer=ai.ai_timer, ai_local_time=ai.ai_local_time,
        time_since_attack=ai.time_since_attack, ignore_until=ai.ignore_until,
        # `AA-006`: a unit that died this tick is not mid-swing. `UpdateTarget`'s
        # first branch `CancelAutoAttack(true, true)`s a dead unit's
        # auto-attack on its own update (see `cancel_on_death` above).
        # `target`/`aa_target` were masked above. This used to mask ONLY
        # `is_attacking`: a corpse kept its wind-up (0.461 s, 3e-5 s) and a
        # counting-down cooldown forever (`docs/PLAYTEST_SWEEP.md` (c)4).
        # `has_auto_attacked` is deliberately untouched: `CancelAutoAttack`
        # does not write it.
        aa_cooldown=jnp.where(cancel_on_death,
                              jnp.zeros_like(aa.aa_cooldown), aa.aa_cooldown),
        aa_windup=jnp.where(alive, aa.aa_windup, jnp.zeros_like(aa.aa_windup)),
        is_attacking=aa.is_attacking & alive,
        has_auto_attacked=aa.has_auto_attacked,
        aa_target=aa_target,
        silenced_ms=silenced_ms,
        r_cast_ms=r_cast_ms,
        recall_windup_ms=recall_windup,
        recall_channel_ms=recall_channel,
        recall_damage_pending=recall_damage_pending,
        observed_enemy_cast_ms=observed_enemy_cast_ms,
        hp=hp, max_hp=max_hp, alive=alive,
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
        gold=gold, xp=xp, cs=cs, level=level, kills=kills,
        kill_spree=ckr.kill_spree, death_spree=death_spree,
        gold_from_minions=gold_from_minions,
        hit_flag_ms=hit_flag_ms, hit_flag_by=hit_flag_by,
        first_blood_done=ckr.first_blood_done,
        gold_timer=gold_timer, ms_since_damaged=ms_since_damaged,
        spell_level=spell_level, buffs=buffs_out,
        spell_cooldown=spell_cooldown_out,
        missile_alive=ms.alive, missile_x=ms.x, missile_y=ms.y,
        missile_tx=ms.target.astype(state.missile_tx.dtype),
        missile_source=ms.source.astype(state.missile_source.dtype),
        missile_source_seq=ms.source_seq.astype(state.missile_source_seq.dtype),
        missile_source_model=ms.source_model.astype(
            state.missile_source_model.dtype),
        missile_damage=ms.damage, missile_speed=ms.speed,
        help_priority=help_priority,
    )


def step_decision(state: LaneState, params: UnitParams,
                  step_ticks: int = 2, delta_ms: float = TICK_MS,
                  lane_path=None, minion_hp=None,
                  enable_call_for_help: bool = True,
                  enable_collision: bool = True,
                  collision_terrain: bool = True,
                  defer_collision_terrain: bool = False) -> LaneState:
    """One agent decision = ``LANERL_STEP_TICKS`` server ticks.

    ``step_ticks`` is 2 in this stack (30 Hz decisions off a 60 Hz sim), set by
    ``lanerl_rl.constants.STEP_TICKS`` and passed to the server as
    ``LANERL_STEP_TICKS``; the two must not drift apart.

    ``enable_call_for_help`` defaults to ``True`` -- see ``tick``'s docstring.
    """
    def one(s, _):
        return tick(s, params, delta_ms, lane_path, minion_hp,
                    enable_call_for_help, enable_collision,
                    collision_terrain, defer_collision_terrain), None
    out, _ = jax.lax.scan(one, state, None, length=step_ticks)
    return out


def env_apply(state: LaneState, orders, cfg) -> LaneState:
    """Write ``orders`` into ``state`` under ``cfg`` (a
    :class:`~lanerl_jax.sim.config.SimConfig`): its params (E's live AD
    snapshot) and its route table/terrain (routed Moves). The first half of
    :func:`env_step`; exposed because the trainer reads the post-order
    ``route_status`` and the per-tick gate applies orders once per decision.
    """
    from .orders import apply_orders
    return apply_orders(state, orders, cfg.params,
                        route_table=cfg.route_table, terrain=cfg.terrain)


def env_advance(state: LaneState, cfg) -> LaneState:
    """Run ``cfg.step_ticks`` ticks with every behaviour flag from ``cfg``.
    The second half of :func:`env_step`."""
    return step_decision(
        state, cfg.params, step_ticks=cfg.step_ticks, delta_ms=cfg.delta_ms,
        lane_path=cfg.lane_path, minion_hp=cfg.minion_hp,
        enable_call_for_help=cfg.enable_call_for_help,
        enable_collision=cfg.enable_collision,
        collision_terrain=cfg.collision_terrain,
        defer_collision_terrain=cfg.defer_collision_terrain)


def env_step(state: LaneState, orders, cfg) -> LaneState:
    """THE entry point (`STRUCT-003`): apply ``orders`` with ``cfg``'s params
    and route table, then run ``cfg.step_ticks`` ticks in ``cfg``'s mode.

    Build ``cfg`` with one of :class:`~lanerl_jax.sim.config.SimConfig`'s
    named constructors rather than passing flags to ``tick``/
    ``step_decision`` by hand -- hand-assembled flags are how the training
    mode ended up run by nothing but the trainer (`COLL-004`).
    """
    return env_advance(env_apply(state, orders, cfg), cfg)
