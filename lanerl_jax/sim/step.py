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
    TURRET_DAMAGE_VS_MINION,
    outer_turret_ramps,
)
from .collision import resolve_collisions
from .init import spawn_minion
from .missiles import step_missiles
from .profiles import PROFILES
from .spells import RANKS_BY_LEVEL, step_buffs
from .waves_jax import step_waves_jax
from .minion_ai import step_minion_ai
from .movement_jax import TICK_MS, step_move_units
from .rewards import ambient_gold, death_rewards, level_for_xp
from .state import Kind, LaneState, MoveOrder, Team
from .targeting import base_priority, nearest_enemy, turret_acquire

__all__ = ["UnitParams", "tick", "step_decision"]


def _attack_damage_against(attack_damage, attacker_kind, target_kind,
                           t_ms=None):
    """Raw attack damage, with the attacker/target modifiers the scripts apply.

    Only one exists in this slice: every lane turret's basic-attack script
    multiplies by 0.7 when the target is a Minion, before mitigation. It is a
    property of the *pair*, not of either unit's stats -- a turret shooting a
    champion does full damage -- so it cannot live in the profile table.
    """
    vs_minion = target_kind == Kind.LANE_MINION
    from_turret = attacker_kind == Kind.TURRET
    ad = jnp.where(from_turret & vs_minion,
                   attack_damage * TURRET_DAMAGE_VS_MINION, attack_damage)
    if t_ms is not None:
        # The map script ramps an outer turret +4 AD every 60 s from t=30 s,
        # capped at 7 applications: 152 at the start, 180 from 390 s on. It is
        # a StatsModifier added on a timer and appears in no stat table, so a
        # turret built from Content alone stays at its level-1 damage all game.
        #
        # Applied to every turret because all 24 currently share the outer
        # profile. The other tiers really run a different schedule (from 480 s,
        # and also +1 Armor / +1 MagicResist), so this is right for the two
        # that matter in a top-lane 1v1 and an over-estimate for the rest --
        # booked in `combat.INNER_TURRET_RAMP_START_MS`.
        ad = jnp.where(from_turret,
                       ad + TURRET_AD_PER_RAMP * outer_turret_ramps(t_ms, jnp),
                       ad)
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


#: profile row -> MinionType, for `ClassifyTarget`. Non-minion rows map to -1,
#: which `base_priority` never consults because it dispatches on `kind` first.
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
         minion_hp=None) -> LaneState:
    """Advance one 16.667 ms server tick.

    ``lane_path`` is ``(W, 2)`` -- ``MinionPaths[LANE_L]``, walked forward by
    blue and reversed by red, exactly as ``SetUpLaneMinion`` does
    (``waypoint.Reverse()`` for ``TEAM_PURPLE``). Pass ``None`` to run without
    wave spawning, which is what the unit tests want.
    """
    n = state.kind.shape[-1]
    dtype = state.x.dtype
    # Every stat is a gather through the unit's profile row: a minion slot is
    # reused by whatever spawns into it, so stats cannot be baked per slot.
    P = lambda k: params[k][state.model]          # noqa: E731

    # ---- 0. wave spawning (Map.Update, before ObjectManager.Update) --------
    if lane_path is not None:
        mtype, next_spawn, m_no, c_no = step_waves_jax(
            state.t_ms, state.next_spawn_ms, state.minion_number,
            state.cannon_count)
        mi = jnp.clip(mtype, 0, 3)
        hp_b = params["max_hp"][_WAVE_ROW_BLUE[mi]]
        hp_r = params["max_hp"][_WAVE_ROW_RED[mi]]
        state = spawn_minion(state, Team.BLUE, _WAVE_ROW_BLUE[mi], hp_b,
                             lane_path, enabled=mtype >= 0)
        state = spawn_minion(state, Team.RED, _WAVE_ROW_RED[mi], hp_r,
                             lane_path[::-1], enabled=mtype >= 0)
        state = state.replace(next_spawn_ms=next_spawn, minion_number=m_no,
                              cannon_count=c_no)

    # ---- 1. movement (AttackableUnit.Update, before anything else) ---------
    x, y, wp_key, _ = step_move_units(
        state.x, state.y, state.waypoints, state.waypoint_key,
        state.n_waypoints, P("move_speed"),
        _can_move(state.move_order, state.alive), delta_ms)

    # ---- 1a. buffs (AttackableUnit.Update runs UpdateBuffs first) ---------
    bs = step_buffs(
        buff_id=state.buff_id, buff_elapsed=state.buff_elapsed,
        buff_duration=state.buff_duration, buff_power=state.buff_power,
        spell_cooldown=state.spell_cooldown, spell_level=state.spell_level,
        x=state.x, y=state.y, kind=state.kind, team=state.team,
        alive=state.alive, armor=P("armor"), delta_ms=delta_ms)

    # ---- 1b. collision push-apart -----------------------------------------
    # `CollisionHandler.Update` runs from `Map.Update`, i.e. after the objects
    # have moved. Without it casters are never pushed into melee reach, which
    # is exactly the type that over-survived in the population comparison.
    # Judgment sets `StatusFlags.Ghosted`, so a spinning Garen passes through
    # units instead of being pushed out of the wave he is standing in.
    x, y = resolve_collisions(x, y, state.kind, state.alive,
                              P("pathfinding_radius"), ghosted=bs.ghosted)

    # ---- 2. the minion controller (AIScript.OnUpdate) ----------------------
    prio = base_priority(state.kind, _minion_type_of(state))
    ai = step_minion_ai(
        kind=state.kind, alive=state.alive, x=x, y=y, team=state.team,
        targetable=state.alive, visible=state.alive,
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
    champ_pick = jnp.where(
        attack_moving,
        nearest_enemy(x, y, state.team, state.alive, state.alive,
                      P("acquisition_range")),
        jnp.int8(-1))
    keep_champ = is_champ & (state.target >= 0)

    # Turrets have their own rule entirely: priority first, distance never,
    # plus the dive override. `TurretAI.OnUpdate` also drops a target that has
    # left range, which is the only way a turret ever releases one.
    is_turret = state.kind == Kind.TURRET
    turret_pick = turret_acquire(
        x, y, state.team, state.alive, state.alive, state.kind,
        _minion_type_of(state), P("attack_range"), state.target,
        state.target, P("attack_range"))
    cur = jnp.clip(state.target, 0, n - 1)
    left_range = (state.target >= 0) & (
        ((x[cur] - x) ** 2 + (y[cur] - y) ** 2)
        > P("attack_range") ** 2)
    turret_target = jnp.where(left_range, jnp.int8(-1), turret_pick)

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
    raw_ad = _attack_damage_against(
        P("attack_damage"), state.kind, state.kind[tgt], state.t_ms)
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
        target_resist=P("armor")[tgt],
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
        targetable=state.alive, armor=P("armor"), target=target,
        delta_ms=delta_ms)

    dmg_ij = jnp.where(landed[:, None] & (jnp.arange(n)[None, :] == tgt[:, None]),
                       aa.damage[:, None], jnp.zeros((n, n), dtype))
    # A missile that lands this tick is credited to the unit that FIRED it, in
    # that unit's own attacker row, so the lowest-index-crosses-zero rule below
    # sees melee hits and missile hits in one ordering rather than two.
    dmg_ij = dmg_ij + ms.damage_ij
    dealt = dmg_ij.sum(axis=0) + bs.damage_dealt
    hp = jnp.maximum(state.hp - dealt, jnp.zeros_like(state.hp))
    alive = state.alive & (hp > 0)
    died = state.alive & ~alive

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
        hp=hp, alive=alive, gold=gold, xp=xp, cs=cs, level=level,
        gold_timer=gold_timer, spell_level=spell_level, buff_id=bs.buff_id,
        buff_elapsed=bs.buff_elapsed, spell_cooldown=bs.spell_cooldown,
        missile_alive=ms.alive, missile_x=ms.x, missile_y=ms.y,
        missile_tx=ms.target.astype(state.missile_tx.dtype),
        missile_source=ms.source.astype(state.missile_source.dtype),
        missile_damage=ms.damage, missile_speed=ms.speed,
    )


def step_decision(state: LaneState, params: UnitParams,
                  step_ticks: int = 2, delta_ms: float = TICK_MS,
                  lane_path=None, minion_hp=None) -> LaneState:
    """One agent decision = ``LANERL_STEP_TICKS`` server ticks.

    ``step_ticks`` is 2 in this stack (30 Hz decisions off a 60 Hz sim), set by
    ``lanerl_rl.constants.STEP_TICKS`` and passed to the server as
    ``LANERL_STEP_TICKS``; the two must not drift apart.
    """
    def one(s, _):
        return tick(s, params, delta_ms, lane_path, minion_hp), None
    out, _ = jax.lax.scan(one, state, None, length=step_ticks)
    return out
