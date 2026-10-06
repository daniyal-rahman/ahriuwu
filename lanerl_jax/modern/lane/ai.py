"""Per-tick lane-minion, turret and structure behaviour, plus champion auto-acquisition and attack-move.

The world owns positions, movement, the attack machine, missiles and damage resolution; this module owns the
decisions and rule state: minion targeting (MINIONS §3, §7.4), turret targeting (TOWERS §3), lane walking, Warming
Up, attack packets, structure vulnerability/regen/plates/Overgrowth/rewards (wrapping ``lane.towers``), wave spawning
into per-lane slots, and attack-move (LANES_TERRAIN §4). Shapes: (N,) per unit; (N, N) pairwise ``[i, j]`` = row
unit i about column unit j.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from ..core import damage as D
from ..core.stats import PHYSICAL, TRUE
from ..core.types import (KIND_CHAMPION, KIND_INHIBITOR, KIND_MINION, KIND_MONSTER, KIND_NEXUS, KIND_NONE,
                          KIND_TURRET, KIND_WARD, NEUTRAL, AttackLaunch, AttackState, WorldUnits, is_structure)
from ..map.lanes import BARRACKS, LANE_PATH_LEN, LANE_PATHS, LANE_TOP
from . import minions as M
from . import towers as T

# Minion controller timings (MINIONS §3.1, §3.4).
SWEEP_INTERVAL_S = 0.25
GIVE_UP_S = 4.0
IGNORE_S = 0.5
ATTACK_MEMORY_S = 2.0
WAYPOINT_MARGIN = 25.0          # MINIONS §2.5
FIRST_WAVE_END_S = 59.0         # wave 0 spawns 30.0..34.8, wave 1 at 60
NO_PRIORITY = 99
SUPER_BUILDING_SCALE = 0.125 / 0.60   # supers deal 12.5 % to inhibitor/Nexus (wiki); DMG.45 already applies 0.60
SIEGE_TURRET_BONUS = 1.4        # Anti-tower Socks TurretDamageBonus 0.4
TURRET_GLOBAL_GOLD = (50., 25., 25., 50.)
INHIBITOR_RADIUS, NEXUS_RADIUS = 213.75, 304.0    # client pathfinding radii (TOWERS §1.4)
FIRST_TURRET_GOLD = 300.
INHIBITOR_LAST_HIT_GOLD = 50.
CHAMPION_ACQUISITION_RANGE = 400.0      # Garen and Jax CharacterRecord acquisitionRange
ATTACK_MOVE_ARRIVE = 10.0               # order point reached (INFERRED L)


class MinionStats(NamedTuple):
    hp: Any
    max_hp: Any
    attack_damage: Any
    armor: Any
    magic_resist: Any
    attack_range: Any
    attack_speed: Any
    move_speed: Any
    radius: Any
    gold: Any
    xp: Any
    windup: Any
    missile_speed: Any          # 0 = melee
    acquisition_range: Any
    upgrade: Any                # latched upgrade index U


def minion_spawn_stats(minion_type, now, team=0) -> MinionStats:
    """Stats of a lane minion spawning at ``now`` (MINIONS §1.3). Unit-class damage ratios are left to DMG.45;
    ``move_speed`` is the spawn-time base (``minion_move_speed`` gives the per-tick value)."""
    k = jnp.clip(jnp.asarray(minion_type, jnp.int32), 0, 3)
    u = M.upgrade_index_at(now)
    s = M.minion_upgrade_stats(k, u, team)
    tab = lambda v: jnp.asarray(v, jnp.float32)[k]
    return MinionStats(s.max_hp, s.max_hp, s.attack_damage, s.armor, s.magic_resist,
                       tab(M.ATTACK_RANGE), tab(M.ATTACK_SPEED), M.base_move_speed(now) + 0. * s.max_hp,
                       tab(M.GAMEPLAY_RADIUS), s.gold, s.xp, tab(M.WINDUP_S), tab(M.MISSILE_SPEED),
                       tab(M.ACQUISITION_RANGE), u + 0 * k)


def minion_move_speed(ai, units: WorldUnits, now):
    """(N,) minion move speed: time increases + side-lane bonus, soft-capped; others keep ``units.move_speed``."""
    idx = M.wave_index_at(units.spawn_time)
    bonus = M.sidelane_bonus_move_speed(idx + 1, ai.lane, M.wave_spawn_time(idx),
                                        now - units.spawn_time)
    ms = M.move_speed_soft_cap(M.base_move_speed(now) + bonus)
    return jnp.where(units.kind == KIND_MINION, ms, units.move_speed).astype(jnp.float32)


class LaneAIState(NamedTuple):
    seq: Any                # (N,) spawn_seq this memory belongs to (-1: never seen)
    target: Any             # (N,) held target, -1 none
    target_seq: Any
    target_priority: Any    # (N,) MINIONS §3.1 class of the held target
    sweep_timer: Any        # (N,) seconds since the last regular sweep
    since_attack: Any       # (N,) seconds holding a target without attacking it
    ignore_until: Any       # (N, N) give-up ignore window (§3.4)
    last_attack: Any        # (N, N) last time i attacked/damaged j
    lane: Any               # (N,) minion spawn lane (nearest own barracks), -1 others
    waypoint: Any           # (N,) next lane-path point
    first_wave: Any
    engaged: Any            # first-wave minion has targeted an enemy minion
    champion_aggro: Any     # turret lock came from champion protection
    warm_stacks: Any        # turret Warming Up stacks 0..3
    warm_until: Any         # stacks are 0 once now >= warm_until


def init_lane_ai(n_units) -> LaneAIState:
    n = int(n_units)
    f = lambda v: jnp.full((n,), v, jnp.float32)
    i = lambda v: jnp.full((n,), v, jnp.int32)
    b = jnp.zeros((n,), bool)
    return LaneAIState(i(-1), i(-1), i(0), i(NO_PRIORITY), f(SWEEP_INTERVAL_S), f(0.),
                       jnp.full((n, n), -jnp.inf, jnp.float32), jnp.full((n, n), -jnp.inf, jnp.float32),
                       i(-1), i(0), b, b, b, i(0), f(0.))


def _pairwise(units):
    x = jnp.asarray(units.x, jnp.float32)
    y = jnp.asarray(units.y, jnp.float32)
    return jnp.sqrt((x[:, None] - x[None, :]) ** 2 + (y[:, None] - y[None, :]) ** 2)


def _exists(near, attacking):
    """``[m, c]``: some v with ``near[m, v]`` and ``attacking[c, v]``."""
    return (near.astype(jnp.float32) @ attacking.T.astype(jnp.float32)) > 0.5


def _lex_argmin(mask, primary, secondary):
    """Row-wise argmin of (primary, secondary) over ``mask``; -1 if empty."""
    big = jnp.iinfo(jnp.int32).max
    p = jnp.where(mask, primary, big)
    best_p = jnp.min(p, axis=1)
    sec = jnp.where(mask & (primary == best_p[:, None]), secondary, jnp.inf)
    idx = jnp.argmin(sec, axis=1).astype(jnp.int32)
    found = jnp.any(mask, axis=1)
    return jnp.where(found, idx, -1), jnp.where(found, best_p, NO_PRIORITY).astype(jnp.int32)


def _reset_new_units(ai, units):
    n = units.kind.shape[0]
    new = jnp.asarray(units.spawn_seq, jnp.int32) != ai.seq
    team = jnp.clip(jnp.asarray(units.team, jnp.int32), 0, 1)
    pos = jnp.stack([jnp.asarray(units.x, jnp.float32), jnp.asarray(units.y, jnp.float32)], -1)
    bar = jnp.asarray(BARRACKS)[team]                                  # (N, 3, 2)
    lane = jnp.argmin(jnp.sum((bar - pos[:, None]) ** 2, -1), axis=1).astype(jnp.int32)
    is_minion = units.kind == KIND_MINION
    row = new[:, None] | new[None, :]
    return ai._replace(
        seq=jnp.asarray(units.spawn_seq, jnp.int32),
        target=jnp.where(new, -1, ai.target), target_seq=jnp.where(new, 0, ai.target_seq),
        target_priority=jnp.where(new, NO_PRIORITY, ai.target_priority),
        sweep_timer=jnp.where(new, SWEEP_INTERVAL_S, ai.sweep_timer).astype(jnp.float32),
        since_attack=jnp.where(new, 0., ai.since_attack).astype(jnp.float32),
        ignore_until=jnp.where(new[:, None], -jnp.inf, ai.ignore_until).astype(jnp.float32),
        last_attack=jnp.where(row, -jnp.inf, ai.last_attack).astype(jnp.float32),
        lane=jnp.where(new, jnp.where(is_minion, lane, -1), ai.lane),
        waypoint=jnp.where(new, 0, ai.waypoint),
        first_wave=jnp.where(new, is_minion & (jnp.asarray(units.spawn_time) < FIRST_WAVE_END_S),
                             ai.first_wave),
        engaged=jnp.where(new, False, ai.engaged),
        champion_aggro=jnp.where(new, False, ai.champion_aggro),
        warm_stacks=jnp.where(new | ~units.alive, 0, ai.warm_stacks),
        warm_until=jnp.where(new | ~units.alive, 0., ai.warm_until).astype(jnp.float32)), n


def _lane_goal(ai, units, pos, team):
    """Advance the lane waypoint and return the walking goal (N, 2)."""
    lane = jnp.clip(ai.lane, 0, 2)
    path = jnp.asarray(LANE_PATHS)[team, lane]                         # (N, L, 2)
    length = jnp.asarray(LANE_PATH_LEN)[lane]
    rows = jnp.arange(path.shape[0])
    k = ai.waypoint
    for _ in range(3):  # static unroll: at most three points per tick
        wp = path[rows, jnp.clip(k, 0, path.shape[1] - 1)]
        nxt = path[rows, jnp.clip(k + 1, 0, path.shape[1] - 1)]
        near = jnp.linalg.norm(pos - wp, axis=-1) < WAYPOINT_MARGIN
        # Rejoin after a chase: a point is passed once the minion is closer to the next point than it is.
        passed = (k + 1 < length) & (jnp.linalg.norm(pos - nxt, axis=-1) < jnp.linalg.norm(wp - nxt, axis=-1))
        k = jnp.where((k < length) & (near | passed), k + 1, k)
    is_nexus = units.kind == KIND_NEXUS
    enemy_nexus = is_nexus[None, :] & (units.team[None, :] != units.team[:, None])
    nexus_idx = jnp.argmax(enemy_nexus, axis=1)
    last = path[rows, jnp.clip(length - 1, 0, path.shape[1] - 1)]
    end = jnp.where(jnp.any(enemy_nexus, axis=1)[:, None], pos[nexus_idx], last)
    goal = jnp.where((k < length)[:, None], path[rows, jnp.clip(k, 0, path.shape[1] - 1)], end)
    return k.astype(jnp.int32), goal


def select_targets(ai: LaneAIState, units: WorldUnits, att: AttackState, *, now, dt,
                   champion_attacked_champion, damage_events, visible=None):
    """One AI step for every minion and turret: ``(ai, desired_target, move_goal, stop)``.

    ``champion_attacked_champion[i, j]``: champion i attempted a Call-for-Help hit on enemy champion j this tick.
    ``damage_events[i, j]``: i damaged j (turret impacts on champions advance Warming Up). ``visible``: optional
    (2, N) team vision. Champion damage on minions never aggroes (26.10)."""
    ai, n = _reset_new_units(ai, units)
    now = jnp.asarray(now, jnp.float32)
    dt = jnp.asarray(dt, jnp.float32)
    kind = jnp.asarray(units.kind, jnp.int32)
    sub = jnp.clip(jnp.asarray(units.sub, jnp.int32), 0, 3)
    team = jnp.clip(jnp.asarray(units.team, jnp.int32), 0, 1)
    alive = jnp.asarray(units.alive, bool)
    radius = jnp.asarray(units.radius, jnp.float32)
    pos = jnp.stack([jnp.asarray(units.x, jnp.float32), jnp.asarray(units.y, jnp.float32)], -1)
    d = _pairwise(units)
    cols = jnp.arange(n)
    if visible is None:
        vis = jnp.ones((n, n), bool)
    else:
        vis = jnp.asarray(visible, bool)[team]
    # Relations use raw teams (neutral monsters are nobody's ally); the clipped ``team`` only indexes tables.
    raw_team = jnp.asarray(units.team, jnp.int32)
    enemy = raw_team[:, None] != raw_team[None, :]
    ally = ~enemy & alive[None, :]
    is_champ = kind == KIND_CHAMPION
    is_minion_k = kind == KIND_MINION
    is_struct = is_structure(kind)
    cac = jnp.asarray(champion_attacked_champion, bool) & is_champ[:, None] & is_champ[None, :]
    dmg = jnp.asarray(damage_events, bool)

    # Aggression memory: champions only through champion-on-champion events, others through damage.
    events = jnp.where(is_champ[:, None], cac, dmg)
    last_attack = jnp.where(events, now, ai.last_attack).astype(jnp.float32)
    in_windup = jnp.asarray(att.windup_left, jnp.float32) > 0.
    winding_on = in_windup[:, None] & (jnp.asarray(att.target, jnp.int32)[:, None] == cols[None, :])
    attacking = ((now - last_attack) <= ATTACK_MEMORY_S) | winding_on
    attacking = attacking & alive[:, None]

    # --- minions
    minion = is_minion_k & alive
    unengaged = ai.first_wave & ~ai.engaged
    acq = jnp.asarray(M.ACQUISITION_RANGE, jnp.float32)[sub]
    first_acq = jnp.asarray(M.FIRST_ACQUISITION_RANGE, jnp.float32)[sub]
    wake = jnp.asarray(M.WAKE_UP_RANGE, jnp.float32)[sub]
    scan = jnp.where(is_champ[None, :], jnp.where(unengaged, wake, acq)[:, None],
                     jnp.where(is_minion_k[None, :], jnp.where(unengaged, first_acq, acq)[:, None],
                               acq[:, None] + radius[None, :]))
    # Team-owned summons (Herald, Voidmites) are attacked like minions; neutral camps never are.
    owned_summon = (kind == KIND_MONSTER) & (raw_team < 2)
    targetable_kind = is_champ | is_minion_k | is_struct | owned_summon
    base_valid = (minion[:, None] & enemy & alive[None, :] & jnp.asarray(units.targetable, bool)[None, :]
                  & vis & targetable_kind[None, :] & (d < scan))
    cand = base_valid & (ai.ignore_until <= now)

    a_champ = attacking & is_champ[None, :]
    a_min = attacking & is_minion_k[None, :]
    near_champ_cfh = ally & (d < M.CFH_CHAMPION_RADIUS)
    near_cfh = ally & (d < M.CFH_GENERIC_RADIUS)
    p1 = is_champ[None, :] & _exists(near_champ_cfh, a_champ & is_champ[:, None])
    p2 = is_minion_k[None, :] & _exists(near_cfh, a_champ & is_minion_k[:, None])
    p3 = is_minion_k[None, :] & _exists(near_cfh, a_min & is_minion_k[:, None])
    p4 = (kind == KIND_TURRET)[None, :] & _exists(near_cfh, a_min & (kind == KIND_TURRET)[:, None])
    base_p = jnp.where(is_minion_k, M.TargetPriority.CLOSEST_MINION,
                       jnp.where(is_champ, M.TargetPriority.CLOSEST_CHAMPION, M.TargetPriority.UNPRIORITIZED))
    prio = jnp.select([p1, p2, p3, p4], [1, 2, 3, 4], default=base_p[None, :]).astype(jnp.int32)

    held = ai.target
    has = held >= 0
    safe = jnp.clip(held, 0, n - 1)
    rows = jnp.arange(n)
    held_ok = (has & (jnp.asarray(units.spawn_seq, jnp.int32)[safe] == ai.target_seq)
               & base_valid[rows, safe])
    just_lost = has & ~held_ok & minion
    target = jnp.where(held_ok, held, -1)
    tprio = jnp.where(held_ok, ai.target_priority, NO_PRIORITY)
    safe = jnp.clip(target, 0, n - 1)
    hit_target = (in_windup & (jnp.asarray(att.target, jnp.int32) == target)) | dmg[rows, safe]
    since_attack = jnp.where(target >= 0, jnp.where(hit_target, 0., ai.since_attack + dt), 0.)

    # Call for Help: a strictly better P1-P4 class switches at once, except holding a turret (not first wave) or
    # mid-windup (§3.3).
    cfh_idx, cfh_p = _lex_argmin(cand & (prio <= 4), prio, d)
    held_turret = (target >= 0) & (kind[safe] == KIND_TURRET)
    blocked = held_turret & ~ai.first_wave
    switch = minion & (cfh_idx >= 0) & (cfh_p < tprio) & ~blocked & ~in_windup
    target = jnp.where(switch, cfh_idx, target)
    tprio = jnp.where(switch, cfh_p, tprio)
    since_attack = jnp.where(switch, 0., since_attack)

    # Regular sweep: only acquires when no target is held (hysteresis).
    timer = ai.sweep_timer + dt
    sweep = minion & ~switch & (just_lost | (timer >= SWEEP_INTERVAL_S))
    timer = jnp.where(sweep | switch, 0., timer)
    safe = jnp.clip(target, 0, n - 1)
    give_up = sweep & (target >= 0) & (since_attack >= GIVE_UP_S)
    ignore_until = jnp.where(give_up[:, None] & (cols[None, :] == safe[:, None]), now + IGNORE_S,
                             ai.ignore_until).astype(jnp.float32)
    target = jnp.where(give_up, -1, target)
    tprio = jnp.where(give_up, NO_PRIORITY, tprio)
    acquire = sweep & (target < 0)
    cand = base_valid & (ignore_until <= now)

    # First-wave spread (§2.8): closest-minion picks go to enemy first-wave melee ``k mod 3``, then the
    # least-attacked, then the closest.
    unit_k = jnp.round((jnp.asarray(units.spawn_time, jnp.float32) - M.WAVE_FIRST_S) / M.WAVE_UNIT_GAP_S
                       ).astype(jnp.int32)
    fw_melee = is_minion_k & (sub == M.MinionType.MELEE) & ai.first_wave & alive
    closest_min = prio == M.TargetPriority.CLOSEST_MINION
    restrict = ai.first_wave & jnp.any(cand & fw_melee[None, :] & closest_min, axis=1)
    cand_acq = cand & ~(restrict[:, None] & closest_min & ~fw_melee[None, :])
    onehot = (target[:, None] == cols[None, :]) & minion[:, None]
    # attackers[m, c] = allied (to m) minions currently targeting c
    same_team = (raw_team[:, None] == raw_team[None, :]).astype(jnp.float32)
    attackers = same_team @ onehot.astype(jnp.float32)
    preferred = (unit_k[None, :] == (unit_k % 3)[:, None])
    spread = jnp.where(restrict[:, None] & fw_melee[None, :] & closest_min,
                       jnp.where(preferred, 0., 1e5) + 1e4 * attackers, 0.)
    acq_idx, acq_p = _lex_argmin(cand_acq, prio, d + spread)
    take = acquire & (acq_idx >= 0)
    target = jnp.where(take, acq_idx, target)
    tprio = jnp.where(take, acq_p, tprio)
    since_attack = jnp.where(take, 0., since_attack)
    safe = jnp.clip(target, 0, n - 1)
    engaged = ai.engaged | (minion & (target >= 0) & is_minion_k[safe])

    in_range = d[rows, safe] <= jnp.asarray(units.attack_range, jnp.float32) + radius + radius[safe]
    waypoint, lane_goal = _lane_goal(ai, units, pos, team)
    m_has = minion & (target >= 0)
    m_goal = jnp.where(m_has[:, None], jnp.where(in_range[:, None], pos, pos[safe]), lane_goal)
    m_stop = m_has & in_range

    # --- turrets
    turret = (kind == KIND_TURRET) & alive
    t_range = d <= (jnp.asarray(units.attack_range, jnp.float32)[:, None] + radius[:, None] + radius[None, :])
    t_valid = (turret[:, None] & enemy & alive[None, :] & jnp.asarray(units.targetable, bool)[None, :]
               & vis & (is_champ | is_minion_k | owned_summon)[None, :] & t_range)
    lock = jnp.where((ai.target >= 0) & (jnp.asarray(units.spawn_seq, jnp.int32)[jnp.clip(ai.target, 0, n - 1)]
                                          == ai.target_seq), ai.target, -1)
    hits_ally_champ = cac | (dmg & is_champ[:, None] & is_champ[None, :])
    near_turret = ally & is_champ[None, :] & (d <= T.PROTECTION_RADIUS)
    aggressive = t_valid & is_champ[None, :] & _exists(near_turret, hits_ally_champ)
    t_prio = jnp.where(is_champ, T.CHAMPION,
                       jnp.where(sub >= M.MinionType.CANNON, T.CANNON_SUPER,
                                 jnp.where(sub == M.MinionType.MELEE, T.MELEE, T.CASTER))).astype(jnp.int32)
    t_target = jax.vmap(T.select_target, in_axes=(0, 0, 0, None, 0))(lock, t_valid, d, t_prio, aggressive)
    t_target = jnp.where(turret, t_target, -1).astype(jnp.int32)
    protected = jnp.any(aggressive, axis=1)
    champion_aggro = jnp.where(protected, True, ai.champion_aggro & (t_target == ai.target) & (t_target >= 0))

    # Warming Up advances on impact (damage event) on a champion (§4.1).
    hit_champ = turret & jnp.any(dmg & is_champ[None, :], axis=1)
    stacks_now = jnp.where(now < ai.warm_until, ai.warm_stacks, 0)
    warm_stacks = jnp.where(hit_champ, jnp.minimum(stacks_now + 1, 3), ai.warm_stacks)
    warm_until = jnp.where(hit_champ, now + 5., ai.warm_until)
    warm_stacks = jnp.where(turret, warm_stacks, 0).astype(jnp.int32)
    warm_until = jnp.where(turret, warm_until, 0.).astype(jnp.float32)

    # --- outputs
    desired = jnp.where(minion, target, jnp.where(turret, t_target, -1)).astype(jnp.int32)
    safe_d = jnp.clip(desired, 0, n - 1)
    move_goal = jnp.where(minion[:, None], m_goal, pos).astype(jnp.float32)
    stop = jnp.where(minion, m_stop, ~is_champ & ~is_minion_k)
    new_ai = ai._replace(
        target=desired,
        target_seq=jnp.where(desired >= 0, jnp.asarray(units.spawn_seq, jnp.int32)[safe_d], 0).astype(jnp.int32),
        target_priority=jnp.where(minion & (desired >= 0), tprio, NO_PRIORITY).astype(jnp.int32),
        sweep_timer=jnp.where(minion, timer, ai.sweep_timer).astype(jnp.float32),
        since_attack=jnp.where(minion, since_attack, 0.).astype(jnp.float32),
        ignore_until=ignore_until, last_attack=last_attack,
        waypoint=jnp.where(is_minion_k, waypoint, ai.waypoint).astype(jnp.int32),
        engaged=engaged, champion_aggro=champion_aggro & turret,
        warm_stacks=warm_stacks, warm_until=warm_until)
    return new_ai, desired, move_goal, stop


def attack_packets(units: WorldUnits, launch: AttackLaunch, *, now, ai: LaneAIState,
                   pushing_bonus=None, pushing_divisor=None,
                   turret_minion_shot_mitigated=False) -> D.Packets:
    """(N,) PHYSICAL basic-attack packets of minion and turret attacks launched now, computed at launch.

    Minions: AD + Minion Slayer ``f * target HP`` vs lane minions; siege x1.4 vs turrets; supers x0.125/0.60 vs
    inhibitor/Nexus. Minion Pushing (MINIONS §4.4, minion vs minion): ``pushing_bonus`` joins ``amp``, the
    target's ``pushing_divisor`` divides raw. Turrets: vs champions ``AD(tier, now)`` x Warming Up (the 30 % armor
    pen is ``turret_offense``); vs minions the %-max-HP shot as TRUE damage (MINIONS U-14), or PHYSICAL with
    ``turret_minion_shot_mitigated`` (TOWERS §4.2)."""
    n = units.kind.shape[0]
    src = jnp.arange(n, dtype=jnp.int32)
    tgt = jnp.asarray(launch.target, jnp.int32)
    t = jnp.clip(tgt, 0, n - 1)
    kind = jnp.asarray(units.kind, jnp.int32)
    sub = jnp.clip(jnp.asarray(units.sub, jnp.int32), 0, 3)
    t_kind, t_sub = kind[t], sub[t]
    is_minion, is_turret = kind == KIND_MINION, kind == KIND_TURRET
    t_minion, t_champ = t_kind == KIND_MINION, t_kind == KIND_CHAMPION
    t_building = (t_kind == KIND_INHIBITOR) | (t_kind == KIND_NEXUS)
    now = jnp.asarray(now, jnp.float32)

    slayer = jnp.asarray(M.MINION_SLAYER_FRACTION, jnp.float32)[sub] * jnp.asarray(units.hp, jnp.float32)[t]
    m_raw = jnp.asarray(units.attack_damage, jnp.float32) + jnp.where(t_minion, slayer, 0.)
    m_raw = m_raw * jnp.where((sub == M.MinionType.CANNON) & (t_kind == KIND_TURRET), SIEGE_TURRET_BONUS, 1.)
    m_raw = m_raw * jnp.where((sub == M.MinionType.SUPER) & t_building, SUPER_BUILDING_SCALE, 1.)
    amp = jnp.zeros((n,), jnp.float32)
    if pushing_bonus is not None:
        amp = jnp.where(is_minion & t_minion, jnp.asarray(pushing_bonus, jnp.float32), 0.)
    if pushing_divisor is not None:
        m_raw = m_raw / jnp.where(is_minion & t_minion, jnp.asarray(pushing_divisor, jnp.float32)[t], 1.)

    stacks = jnp.where(now < ai.warm_until, ai.warm_stacks, 0)
    tier = sub
    champ_raw = T.attack_damage(tier, now) * T.warming_multiplier(stacks)
    shot_raw = T.minion_shot_fraction(t_sub, tier) * jnp.asarray(units.max_hp, jnp.float32)[t]
    t_raw = jnp.where(t_champ, champ_raw, shot_raw)
    shot_type = PHYSICAL if turret_minion_shot_mitigated else TRUE
    t_type = jnp.where(t_champ, PHYSICAL, shot_type)

    valid = (jnp.asarray(launch.launched, bool) & (tgt >= 0) & jnp.asarray(units.alive, bool)
             & ((is_minion & (t_kind != KIND_NONE)) | (is_turret & (t_champ | t_minion))))
    raw = jnp.where(is_turret, t_raw, m_raw)
    dtype = jnp.where(is_turret, t_type, PHYSICAL)
    return D.packets(valid, src, t, jnp.where(valid, raw, 0.), dtype, flags=D.BASIC_ATTACK,
                     amp=jnp.where(is_turret, 0., amp), cast_id=jnp.asarray(launch.cast_id, jnp.int32))


def turret_offense(units: WorldUnits, off: D.Offense) -> D.Offense:
    """Turrets get 30 % armor pen (item 1500) and ``is_turret``."""
    t = units.kind == KIND_TURRET
    return off._replace(percent_armor_pen=jnp.where(t, T.ARMOR_PENETRATION, off.percent_armor_pen).astype(jnp.float32),
                        is_turret=off.is_turret | t)


# --- structures --------------------------------------------------------------------------------------------------
class TowersState(NamedTuple):
    """Structure state aligned with world units; the properties are static unit stats (TOWERS §1)."""
    turret: T.TurretState   # (N,) vmapped; tier 0..3 turrets, 4 inhibitor, 5 Nexus
    is_structure: Any
    team: Any
    lane: Any               # -1: nexus/none
    prereq: Any             # structure that must die first, -1 none
    targetable: Any         # vulnerability (TOWERS §2)
    first_turret_taken: Any  # ()

    @property
    def hp(self):
        return self.turret.hp

    @property
    def max_hp(self):
        return self.turret.max_hp

    @property
    def radius(self):
        tier = self.turret.tier
        return jnp.where(tier == T.INHIBITOR_BUILDING, INHIBITOR_RADIUS,
                         jnp.where(tier == T.NEXUS_BUILDING, NEXUS_RADIUS, T.GAMEPLAY_RADIUS)).astype(jnp.float32)

    @property
    def armor(self):
        return jnp.where(self.turret.tier >= T.INHIBITOR_BUILDING, T.BUILDING_ARMOR, 60.).astype(jnp.float32)

    @property
    def magic_resist(self):
        return jnp.where(self.turret.tier >= T.INHIBITOR_BUILDING, T.BUILDING_MR, 60.).astype(jnp.float32)

    @property
    def attack_damage(self):
        """Game-start AD."""
        return jnp.where(self.turret.tier <= T.NEXUS, T.attack_damage(jnp.clip(self.turret.tier, 0, 3), 0.),
                         0.).astype(jnp.float32)

    @property
    def attack_range(self):
        return jnp.where(self.turret.tier <= T.NEXUS, T.ATTACK_RANGE, 0.).astype(jnp.float32)

    @property
    def attack_speed(self):
        return jnp.where(self.turret.tier <= T.NEXUS, T.ATTACK_SPEED, 0.).astype(jnp.float32)

    @property
    def windup(self):
        return jnp.where(self.turret.tier <= T.NEXUS, T.WINDUP_S, 0.).astype(jnp.float32)


def _structure_tier(kind, sub):
    return jnp.where(kind == KIND_TURRET, jnp.clip(sub, 0, 3),
                     jnp.where(kind == KIND_INHIBITOR, T.INHIBITOR_BUILDING,
                               jnp.where(kind == KIND_NEXUS, T.NEXUS_BUILDING, 0))).astype(jnp.int32)


def init_structures(cfg) -> TowersState:
    """``init_towers`` from a ``WorldConfig`` layout (``structure_prereq``, if present, sets the lane chain)."""
    kind = jnp.asarray(cfg.unit_kind, jnp.int32)
    n = kind.shape[0]
    z = jnp.zeros((n,), jnp.float32)
    units = WorldUnits(kind, jnp.asarray(cfg.unit_sub, jnp.int32), jnp.asarray(cfg.unit_team, jnp.int32),
                       jnp.ones((n,), bool), jnp.ones((n,), bool), jnp.asarray(cfg.unit_x, jnp.float32),
                       jnp.asarray(cfg.unit_y, jnp.float32), z, z, z, z, z, z, z, z, z,
                       jnp.arange(n, dtype=jnp.int32), z)
    prereq = getattr(cfg, "structure_prereq", None)
    return init_towers(units, cfg.unit_lane, prereq=prereq)


def init_towers(units: WorldUnits, lane, prereq=None) -> TowersState:
    """Lane chain (TOWERS §2): inner <- outer <- ..., inhibitor <- inhibitor turret of the same team and lane;
    Nexus turrets need an own inhibitor down, the Nexus also both Nexus turrets (``_vulnerable``). Outer Overgrowth
    clocks start at 0:10, other lane turrets' when they become targetable."""
    kind = jnp.asarray(units.kind, jnp.int32)
    sub = jnp.asarray(units.sub, jnp.int32)
    team = jnp.asarray(units.team, jnp.int32)
    lane = jnp.asarray(lane, jnp.int32)
    tier = _structure_tier(kind, sub)
    is_struct = (kind == KIND_TURRET) | (kind == KIND_INHIBITOR) | (kind == KIND_NEXUS)
    need = jnp.where((kind == KIND_TURRET) & (tier == T.INNER), T.OUTER,
                     jnp.where((kind == KIND_TURRET) & (tier == T.INHIBITOR), T.INNER,
                               jnp.where(kind == KIND_INHIBITOR, T.INHIBITOR, -1)))
    match = ((team[:, None] == team[None, :]) & (lane[:, None] == lane[None, :])
             & (kind[None, :] == KIND_TURRET) & (tier[None, :] == need[:, None]) & (need[:, None] >= 0))
    derived = jnp.where(jnp.any(match, axis=1), jnp.argmax(match, axis=1), -1).astype(jnp.int32)
    lane_chain = (kind == KIND_TURRET) & (tier <= T.INHIBITOR) | (kind == KIND_INHIBITOR)
    prereq = derived if prereq is None else jnp.where(lane_chain, jnp.asarray(prereq, jnp.int32), -1)
    prereq = prereq.astype(jnp.int32)
    st = jax.vmap(T.init_turret)(tier)
    max_hp = jnp.where(is_struct & (jnp.asarray(units.max_hp) > 0), jnp.asarray(units.max_hp, jnp.float32), st.max_hp)
    since = jnp.where((kind == KIND_TURRET) & (tier == T.OUTER), 10., jnp.inf).astype(jnp.float32)
    st = st._replace(hp=max_hp.astype(jnp.float32), max_hp=max_hp.astype(jnp.float32), growth_since=since)
    towers = TowersState(st, is_struct, team, lane, prereq, is_struct, jnp.bool_(False))
    return towers._replace(targetable=_vulnerable(towers))


def _vulnerable(towers: TowersState):
    st = towers.turret
    alive = towers.is_structure & (st.hp > 0)
    n = alive.shape[0]
    pre = jnp.clip(towers.prereq, 0, n - 1)
    pre_dead = (towers.prereq < 0) | ~alive[pre]
    team = jnp.clip(towers.team, 0, 1)
    inhib = towers.is_structure & (st.tier == T.INHIBITOR_BUILDING)
    nexus_t = towers.is_structure & (st.tier == T.NEXUS)
    inhib_dead = jnp.stack([jnp.any(inhib & ~alive & (team == k)) for k in (0, 1)])[team]
    nexus_t_alive = jnp.stack([jnp.any(nexus_t & alive & (team == k)) for k in (0, 1)])[team]
    ok = jnp.where(st.tier == T.NEXUS, inhib_dead,
                   jnp.where(st.tier == T.NEXUS_BUILDING, inhib_dead & ~nexus_t_alive, pre_dead))
    return alive & ok


def turret_tick(towers: TowersState, units: WorldUnits, *, now, dt) -> TowersState:
    """Start-of-tick structure bookkeeping (TOWERS §9): regen/respawn, vulnerability, Overgrowth clocks, backdoor
    refresh (enemy lane minion within 1000) and crystal suppression (enemy champion/minion in attack range).
    ``towers.turret.hp`` is authoritative: write ``structure_unit_view`` back to the world after this."""
    now = jnp.asarray(now, jnp.float32)
    n = towers.team.shape[0]
    nowv = jnp.broadcast_to(now, (n,))
    st = jax.vmap(T.regenerate_and_respawn)(towers.turret, nowv, jnp.broadcast_to(jnp.asarray(dt, jnp.float32), (n,)))
    st = st._replace(hp=st.hp.astype(jnp.float32))
    towers = towers._replace(turret=st)
    targetable = _vulnerable(towers)
    lane_turret = towers.is_structure & (st.tier < T.NEXUS)
    st = jax.vmap(T.unlock)(st, jnp.where(targetable & lane_turret, now, jnp.inf).astype(jnp.float32))
    d = _pairwise(units)
    kind = jnp.asarray(units.kind, jnp.int32)
    enemy = (towers.team[:, None] != jnp.asarray(units.team, jnp.int32)[None, :]) & jnp.asarray(units.alive, bool)[None, :]
    minion = kind == KIND_MINION
    minion_near = jnp.any(enemy & minion[None, :] & (d <= T.BACKDOOR_RADIUS), axis=1) & towers.is_structure
    unit_near = jnp.any(enemy & (minion | (kind == KIND_CHAMPION))[None, :]
                        & T.in_attack_range(d, jnp.asarray(units.radius, jnp.float32)[None, :]), axis=1)
    adv = jax.vmap(T.advance)(st, nowv, minion_near, unit_near)
    adv = adv._replace(growth_active=adv.growth_active & lane_turret,
                       backdoor_until=adv.backdoor_until.astype(jnp.float32))
    return towers._replace(turret=adv, targetable=targetable)


def structure_unit_view(towers: TowersState, units: WorldUnits):
    """``(hp, alive, targetable)`` (N,) for the world to write back."""
    s = towers.is_structure
    hp = jnp.where(s, towers.turret.hp, units.hp).astype(jnp.float32)
    return hp, jnp.where(s, towers.turret.hp > 0, units.alive), jnp.where(s, towers.targetable, units.targetable)


def turret_defense(towers: TowersState, units: WorldUnits, *, now):
    """(N,) ``(armor, mr)``: turrets ``T.resistance`` with enemy champions within 850, inhibitor/Nexus 20/0, others
    their unit stats."""
    now = jnp.asarray(now, jnp.float32)
    d = _pairwise(units)
    champs = ((jnp.asarray(units.kind) == KIND_CHAMPION) & jnp.asarray(units.alive, bool))[None, :] \
        & (towers.team[:, None] != jnp.asarray(units.team, jnp.int32)[None, :])
    n850 = jnp.sum(champs & (d <= T.BULWARK_RADIUS), axis=1)
    res = jax.vmap(T.resistance, in_axes=(0, None, 0))(towers.turret, now, n850).astype(jnp.float32)
    kind = jnp.asarray(units.kind)
    building = towers.is_structure & ((kind == KIND_INHIBITOR) | (kind == KIND_NEXUS))
    turret = towers.is_structure & (kind == KIND_TURRET)
    armor = jnp.where(turret, res, jnp.where(building, T.BUILDING_ARMOR, units.armor))
    mr = jnp.where(turret, res, jnp.where(building, T.BUILDING_MR, units.magic_resist))
    return armor.astype(jnp.float32), mr.astype(jnp.float32)


def structure_defense_mods(towers: TowersState, units: WorldUnits, *, now):
    """(N,) ``(received_mult, invulnerable)``: backdoor x0.2 on live turrets without an enemy lane minion near in
    the last 3 s (TOWERS §4.4); untargetable structures take no damage."""
    now = jnp.asarray(now, jnp.float32)
    turret = towers.is_structure & (jnp.asarray(units.kind) == KIND_TURRET) & (towers.turret.hp > 0)
    backdoor = turret & (now >= towers.turret.backdoor_until)
    return (jnp.where(backdoor, 0.2, 1.).astype(jnp.float32),
            towers.is_structure & ~towers.targetable)


class PlateEvents(NamedTuple):
    """Structure reward events this tick, (N,): local gold (``plate_gold + first_turret_gold``) is shared by nearby
    champions of ``rewarded_team``, ``global_gold`` paid to each of them, ``last_hit_gold`` to the last hitter."""
    plates: Any             # claimed this tick (the 5th = destruction)
    plate_gold: Any
    destroyed: Any
    global_gold: Any
    first_turret_gold: Any
    last_hit_gold: Any      # inhibitor
    rewarded_team: Any
    nexus_destroyed: Any


def structure_damage_events(towers: TowersState, hp_before, hp_after, *, now):
    """Plates and destructions between ``hp_before`` and ``hp_after``: ``(towers, PlateEvents)``.

    Plate k falls at ``max_hp * (0.9, 0.75, 0.55, 0.3, 0)[k]``; plates 1-4 each add a 20 s Bulwark stack. Kills pay
    global gold, the first turret 300 local, an inhibitor 50 to its last hitter, and start respawn timers."""
    now = jnp.asarray(now, jnp.float32)
    st = towers.turret
    s = towers.is_structure
    before = jnp.asarray(hp_before, jnp.float32)
    after = jnp.where(s, jnp.maximum(jnp.asarray(hp_after, jnp.float32), 0.), st.hp)
    platable = s & (st.tier < T.NEXUS)
    thresholds = st.max_hp[:, None] * jnp.asarray([.9, .75, .55, .3, 0.], jnp.float32)[None, :]
    plates = jnp.where(platable, jnp.maximum(st.plates, jnp.sum(after[:, None] <= thresholds, axis=1)),
                       st.plates).astype(jnp.int32)
    gained = (plates - st.plates).astype(jnp.int32)
    slots = jnp.arange(4)
    bulwark = jnp.where((slots[None, :] >= st.plates[:, None]) & (slots[None, :] < plates[:, None]),
                        now + 20., st.bulwark_until).astype(jnp.float32)
    destroyed = s & (before > 0) & (after <= 0)
    is_turret = s & (st.tier <= T.NEXUS)
    global_gold = jnp.where(destroyed & is_turret,
                            jnp.asarray(TURRET_GLOBAL_GOLD, jnp.float32)[jnp.clip(st.tier, 0, 3)], 0.)
    turret_kill = destroyed & is_turret
    n = s.shape[0]
    first_idx = jnp.argmax(turret_kill)
    first = turret_kill & (jnp.arange(n) == first_idx) & ~towers.first_turret_taken
    respawn_at = jnp.where(destroyed, now + T.respawn_delay(st.tier), st.respawn_at).astype(jnp.float32)
    st = st._replace(hp=after.astype(jnp.float32), plates=plates, bulwark_until=bulwark, respawn_at=respawn_at,
                     growth_active=st.growth_active & (after > 0),
                     warm_stacks=jnp.where(destroyed, 0, st.warm_stacks).astype(st.warm_stacks.dtype))
    events = PlateEvents(
        plates=gained,
        plate_gold=(gained * jax.vmap(T.plate_value)(jnp.broadcast_to(now, (n,)), st.tier)).astype(jnp.float32),
        destroyed=destroyed, global_gold=global_gold.astype(jnp.float32),
        first_turret_gold=jnp.where(first, FIRST_TURRET_GOLD, 0.).astype(jnp.float32),
        last_hit_gold=jnp.where(destroyed & (st.tier == T.INHIBITOR_BUILDING), INHIBITOR_LAST_HIT_GOLD,
                                0.).astype(jnp.float32),
        rewarded_team=(1 - jnp.clip(towers.team, 0, 1)).astype(jnp.int32),
        nexus_destroyed=destroyed & (st.tier == T.NEXUS_BUILDING))
    towers = towers._replace(turret=st, first_turret_taken=towers.first_turret_taken | jnp.any(turret_kill))
    return towers._replace(targetable=_vulnerable(towers)), events


def overgrowth_packets(towers: TowersState, units: WorldUnits, champion_attack_hit, *, now, team_level):
    """Consume Overgrowth on enemy-champion basic-attack hits ``champion_attack_hit[i, j]``: ``(towers, packets)``.

    One TRUE packet per crystal from the attacker (no modifiers, no vamp), appended after the attack's own packet;
    not consumed under backdoor (TOWERS §6.3). ``team_level`` (2,): average champion level per team."""
    now = jnp.asarray(now, jnp.float32)
    st = towers.turret
    n = st.hp.shape[0]
    hit = (jnp.asarray(champion_attack_hit, bool) & (jnp.asarray(units.kind) == KIND_CHAMPION)[:, None]
           & (jnp.asarray(units.team, jnp.int32)[:, None] != towers.team[None, :]))
    any_hit = jnp.any(hit, axis=0)
    attacker = jnp.argmax(hit, axis=0).astype(jnp.int32)
    backdoor = now >= st.backdoor_until
    proc = (any_hit & towers.is_structure & (st.tier < T.NEXUS) & st.growth_active & ~backdoor
            & (st.hp > 0) & towers.targetable)
    level = jnp.asarray(team_level, jnp.float32)[jnp.clip(jnp.asarray(units.team, jnp.int32)[attacker], 0, 1)]
    lo, hi = T.overgrowth_level_fractions(level)
    dmg = jax.vmap(T.overgrowth_damage, in_axes=(0, None, 0, 0))(st, now, lo, hi)
    st = st._replace(growth_active=st.growth_active & ~proc,
                     growth_since=jnp.where(proc, now, st.growth_since).astype(jnp.float32))
    flags = D.TAG_PROC | D.PROP_NO_DAMAGE_MOD | D.PROP_NO_OMNIVAMP
    pk = D.packets(proc, attacker, jnp.arange(n, dtype=jnp.int32), jnp.where(proc, dmg, 0.), TRUE, flags=flags)
    return towers._replace(turret=st), pk


# --- wave spawning into per-lane slot ranges (LANES_TERRAIN §1) ---------------------------------------------------
def wave_inhibitor_inputs(towers: TowersState) -> dict:
    """``lane_spawn_step`` inputs; ``[t, l]`` describes team ``1-t``'s inhibitor in lane ``l``."""
    st = towers.turret
    inhib = towers.is_structure & (st.tier == T.INHIBITOR_BUILDING)
    dead = inhib & (st.hp <= 0.)
    team = jnp.asarray(towers.team, jnp.int32)
    lane = jnp.asarray(towers.lane, jnp.int32)
    sel = lambda o, l: inhib & (team == o) & (lane == l)
    down = jnp.stack([jnp.stack([jnp.any(dead & sel(o, l)) for l in range(3)]) for o in (0, 1)])
    resp = jnp.stack([jnp.stack([jnp.min(jnp.where(dead & sel(o, l), st.respawn_at, jnp.inf)) for l in range(3)])
                      for o in (0, 1)])
    all_down = jnp.all(down, axis=1)
    enemy = jnp.asarray([1, 0])
    return dict(enemy_inhibitor_down=down[enemy], all_enemy_inhibitors_down=jnp.broadcast_to(
        all_down[enemy][:, None], (2, 3)), inhibitor_respawn_at=resp[enemy].astype(jnp.float32))


def slot_lane(n_units: int, slot0: int, per_lane: int = 40, lanes=(0, 1, 2)):
    """(N,) lane of each minion slot (block b holds ``lanes[b]``), -1 outside the minion blocks."""
    i = jnp.arange(n_units) - slot0
    block = jnp.clip(i // per_lane, 0, len(lanes) - 1)
    return jnp.where((i >= 0) & (i < len(lanes) * per_lane), jnp.asarray(lanes, jnp.int32)[block], -1).astype(jnp.int32)


class SpawnWrite(NamedTuple):
    """Minion slots to spawn this tick, (N,); ``spawn_seq = next_seq + seq_offset``."""
    pick: Any
    team: Any
    sub: Any                # minion type
    lane: Any
    x: Any                  # barracks of (team, lane)
    y: Any
    stats: MinionStats
    seq_offset: Any         # 0..count-1 among picked slots, -1 elsewhere
    count: Any              # ()
    overflow: Any           # () due units without a free slot in their lane


def spawn_lane_minions(spawn: M.LaneSpawnState, towers: TowersState, kind, alive, *, now, slot0: int,
                       per_lane: int = 40, lanes=(0, 1, 2)):
    """Wave spawning (MINIONS §2): ``(spawn_state, SpawnWrite, due)``.

    Spawning lane ``lanes[b]`` (static) owns slots ``[slot0 + per_lane*b, slot0 + per_lane*(b+1))`` for both teams;
    free slots are empty or dead minions, Blue takes the lowest, then Red. Every lane's cursor advances."""
    now = jnp.asarray(now, jnp.float32)
    spawn, due = M.lane_spawn_step(spawn, now, **wave_inhibitor_inputs(towers))
    kind = jnp.asarray(kind, jnp.int32)
    n = kind.shape[0]
    idx = jnp.arange(n)
    free = (kind == KIND_NONE) | ((kind == KIND_MINION) & ~jnp.asarray(alive, bool))
    pick = jnp.zeros((n,), bool)
    team_o = jnp.zeros((n,), jnp.int32)
    sub_o = jnp.zeros((n,), jnp.int32)
    lane_o = jnp.full((n,), -1, jnp.int32)
    overflow = jnp.int32(0)
    for b, l in enumerate(lanes):
        m = free & (idx >= slot0 + per_lane * b) & (idx < slot0 + per_lane * (b + 1))
        for t in (0, 1):
            p = m & ((jnp.cumsum(m) - 1) == 0) & due.due[t, l]
            pick, m = pick | p, m & ~p
            team_o = jnp.where(p, t, team_o)
            sub_o = jnp.where(p, due.minion_type[t, l], sub_o)
            lane_o = jnp.where(p, l, lane_o)
            overflow = overflow + (due.due[t, l] & ~jnp.any(p)).astype(jnp.int32)
    stats = minion_spawn_stats(jnp.clip(sub_o, 0, 3), now, team_o)
    stats = MinionStats(*(jnp.broadcast_to(jnp.asarray(v), (n,)) for v in stats))
    bar = jnp.asarray(BARRACKS)[team_o, jnp.clip(lane_o, 0, 2)]
    seq = jnp.where(pick, jnp.cumsum(pick) - 1, -1).astype(jnp.int32)
    write = SpawnWrite(pick, team_o, sub_o, lane_o, bar[:, 0], bar[:, 1], stats, seq,
                       jnp.sum(pick).astype(jnp.int32), overflow)
    return spawn, write, due


def minion_ghosted(ai: LaneAIState, units: WorldUnits, now):
    """(N,) first-wave minions still ghosted (``minions.first_wave_ghost_s``), for the collision mask."""
    first = (jnp.asarray(units.kind) == KIND_MINION) & jnp.asarray(units.alive, bool) \
        & (M.wave_index_at(units.spawn_time) == 0)
    return first & ((jnp.asarray(now, jnp.float32) - units.spawn_time) < M.first_wave_ghost_s(ai.lane))


def minion_in_lane(ai: LaneAIState, units: WorldUnits, lane: int = LANE_TOP):
    """(N,) lane minions spawned in ``lane`` (ROLE_QUESTS §2.1)."""
    return (jnp.asarray(units.kind) == KIND_MINION) & (ai.lane == lane)


class GameResult(NamedTuple):
    over: Any               # () a Nexus is destroyed
    winner: Any             # () 0 blue / 1 red; -1 while running or on a same-tick double kill


def game_result(towers: TowersState) -> GameResult:
    """The team whose Nexus stands wins (TOWERS §1.4); sticky since Nexuses never respawn."""
    st = towers.turret
    nexus = towers.is_structure & (st.tier == T.NEXUS_BUILDING)
    team = jnp.asarray(towers.team, jnp.int32)
    lost = jnp.stack([jnp.any(nexus & (team == k) & (st.hp <= 0.)) for k in (0, 1)])
    over = jnp.any(lost)
    winner = jnp.where(lost[0] & ~lost[1], 1, jnp.where(lost[1] & ~lost[0], 0, -1)).astype(jnp.int32)
    return GameResult(over, winner)


# --- champion acquisition, idle auto-attack and attack-move (LANES_TERRAIN §4) ----------------------------------
def champion_acquisition_range(attack_range, base_attack_range, base_acquisition=CHAMPION_ACQUISITION_RANGE):
    """The record's acquisition range shifted by attack-range modifiers (wiki Champion statistic)."""
    return jnp.asarray(base_acquisition, jnp.float32) + (jnp.asarray(attack_range, jnp.float32)
                                                         - jnp.asarray(base_attack_range, jnp.float32))


def _hostile(units: WorldUnits, champion_unit, visible, *, monster_aggro=None, champions_only=False,
             wards=True):
    """(C, N) attack-move / auto-acquire candidates: enemy champions, minions, structures, wards and monsters
    only where ``monster_aggro`` (wiki Basic attack)."""
    kind = jnp.asarray(units.kind, jnp.int32)
    team = jnp.asarray(units.team, jnp.int32)
    me = jnp.asarray(champion_unit, jnp.int32)
    c, n = me.shape[0], kind.shape[0]
    kinds = (kind == KIND_CHAMPION) | (kind == KIND_MINION) | (kind == KIND_TURRET) | (kind == KIND_INHIBITOR) \
        | (kind == KIND_NEXUS)
    if wards:
        kinds = kinds | (kind == KIND_WARD)
    if champions_only:
        kinds = kind == KIND_CHAMPION
    enemy = (team[None, :] != team[me][:, None]) & (team[None, :] != NEUTRAL) & kinds[None, :]
    if monster_aggro is not None and not champions_only:
        enemy = enemy | ((kind == KIND_MONSTER)[None, :] & jnp.asarray(monster_aggro, bool))
    vis = jnp.ones((c, n), bool) if visible is None else jnp.asarray(visible, bool)
    ok = enemy & jnp.asarray(units.alive, bool)[None, :] & jnp.asarray(units.targetable, bool)[None, :] & vis
    return ok & (jnp.arange(n)[None, :] != me[:, None])


def _nearest(mask, dist):
    d = jnp.where(mask, dist, jnp.inf)
    i = jnp.argmin(d, axis=1).astype(jnp.int32)
    return jnp.where(jnp.isfinite(jnp.min(d, axis=1)), i, -1)


def idle_acquire(units: WorldUnits, champion_unit, visible, acquisition_range, *, monster_aggro=None):
    """(C,) nearest (centre distance) valid enemy whose hitbox is within an idle champion's acquisition radius
    (wiki; distances INFERRED M)."""
    me = jnp.asarray(champion_unit, jnp.int32)
    x, y, r = (jnp.asarray(v, jnp.float32) for v in (units.x, units.y, units.radius))
    center = jnp.sqrt((x[None, :] - x[me][:, None]) ** 2 + (y[None, :] - y[me][:, None]) ** 2)
    edge = center - r[None, :] - r[me][:, None]
    ok = _hostile(units, me, visible, monster_aggro=monster_aggro, wards=False) \
        & (edge <= jnp.asarray(acquisition_range, jnp.float32)[:, None])
    return _nearest(ok, center)


class AttackMoveOut(NamedTuple):
    target: Any             # (C,) unit to chase; -1 = walk to the order point
    target_seq: Any
    goal_x: Any             # target position or the order point
    goal_y: Any
    active: Any             # False once the point is reached


def attack_move_step(active, point_x, point_y, held, held_seq, units: WorldUnits, champion_unit, visible,
                     acquisition_range, *, cursor_x=None, cursor_y=None, cursor_radius=None,
                     champions_only=False, monster_aggro=None, arrive=ATTACK_MOVE_ARRIVE) -> AttackMoveOut:
    """One tick of the attack-move order (wiki Basic attack "Attack-move").

    A still-valid ``held`` target is kept; otherwise the hostile nearest the champion within its acquisition radius
    is taken (no champion priority; ``champions_only`` is the client toggle). With ``cursor_x/cursor_y`` units within
    ``cursor_radius`` (default the acquisition radius, INFERRED L) of the cursor are tried first. Without a target
    the champion walks to the point; within ``arrive`` of it the order ends."""
    me = jnp.asarray(champion_unit, jnp.int32)
    act = jnp.asarray(active, bool)
    n = jnp.asarray(units.kind).shape[0]
    x, y, r = (jnp.asarray(v, jnp.float32) for v in (units.x, units.y, units.radius))
    hostile = _hostile(units, me, visible, monster_aggro=monster_aggro, champions_only=champions_only)
    held = jnp.asarray(held, jnp.int32)
    hs = jnp.clip(held, 0, n - 1)
    rows = jnp.arange(me.shape[0])
    keep = act & (held >= 0) & hostile[rows, hs] & (jnp.asarray(units.spawn_seq, jnp.int32)[hs]
                                                    == jnp.asarray(held_seq, jnp.int32))
    center = jnp.sqrt((x[None, :] - x[me][:, None]) ** 2 + (y[None, :] - y[me][:, None]) ** 2)
    acq = jnp.asarray(acquisition_range, jnp.float32)[:, None]
    scan = _nearest(hostile & (center - r[None, :] - r[me][:, None] <= acq), center)
    if cursor_x is not None:
        cr = acq[:, 0] if cursor_radius is None else jnp.broadcast_to(jnp.asarray(cursor_radius, jnp.float32), acq[:, 0].shape)
        dc = jnp.sqrt((x[None, :] - jnp.asarray(cursor_x)[:, None]) ** 2 + (y[None, :] - jnp.asarray(cursor_y)[:, None]) ** 2)
        near_cursor = _nearest(hostile & (dc - r[None, :] <= cr[:, None]), dc)
        scan = jnp.where(near_cursor >= 0, near_cursor, scan)
    target = jnp.where(keep, held, jnp.where(act, scan, -1)).astype(jnp.int32)
    ts = jnp.clip(target, 0, n - 1)
    px, py = jnp.asarray(point_x, jnp.float32), jnp.asarray(point_y, jnp.float32)
    gx = jnp.where(target >= 0, x[ts], px)
    gy = jnp.where(target >= 0, y[ts], py)
    arrived = (target < 0) & (jnp.sqrt((x[me] - px) ** 2 + (y[me] - py) ** 2) <= arrive)
    return AttackMoveOut(target, jnp.where(target >= 0, jnp.asarray(units.spawn_seq, jnp.int32)[ts], 0),
                         gx, gy, act & ~arrived)
