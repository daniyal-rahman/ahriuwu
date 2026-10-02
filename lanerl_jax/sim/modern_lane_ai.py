"""Per-tick 26.19 lane-minion and turret behaviour for the modern world tick.

``modern_step`` owns positions, movement, collision, the generic attack timer
(``modern_world_types.AttackState``), missiles and damage resolution
(``modern_damage``). This module owns the *decisions* and the rule state:

* ``select_targets``: minion target choice (MINIONS §3, §7.4: post-26.10
  priority list, Call for Help, hysteresis, 250 ms sweep, 4 s give-up, first
  wave) and turret target choice (TOWERS §3: sticky lock, champion
  protection, priority classes), lane-path walking goals, Warming Up stacks.
* ``attack_packets``: minion and turret basic-attack damage packets.
* ``minion_spawn_stats`` / ``minion_move_speed``: client upgrade formula,
  time and side-lane move speed.
* ``init_towers`` / ``turret_tick`` / ``turret_defense`` /
  ``structure_defense_mods`` / ``structure_damage_events`` /
  ``overgrowth_packets``: structure vulnerability chain, Overgrowth, backdoor,
  regen/respawn, Bulwark, plates and destruction rewards (wrapping
  ``modern_towers``).

Recommended order inside one world tick (MINIONS §7.3, TOWERS §9)::

    towers = turret_tick(towers, units, now=now, dt=dt)        # write back structure_unit_view
    armor, mr = turret_defense(towers, units, now=now)         # Defense for this tick's packets
    ... resolve due packets (world), then for structures:
    towers, plate_events = structure_damage_events(towers, hp_before, hp_after, now=now)
    ai, desired, goal, stop = select_targets(ai, units, att, now=now, dt=dt, ...)
    ... world movement + attack machine -> launch
    packets = attack_packets(units, launch, now=now, ai=ai)

Everything is fixed-shape JAX over the N world units; no Python branches on
traced values. Shapes: (N,) per unit, (N, N) pairwise ``[i, j]`` = row unit i
about column unit j.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from . import modern_damage as D
from . import modern_minions as M
from . import modern_towers as T
from .modern_stats import PHYSICAL, TRUE
from .modern_world_types import (KIND_CHAMPION, KIND_INHIBITOR, KIND_MINION, KIND_NEXUS, KIND_NONE,
                                 KIND_TURRET, AttackLaunch, AttackState, WorldUnits)

__all__ = [
    "LaneAIState", "init_lane_ai", "select_targets", "attack_packets",
    "MinionStats", "minion_spawn_stats", "minion_move_speed",
    "TowersState", "init_towers", "init_structures", "turret_tick", "turret_defense",
    "structure_defense_mods", "structure_unit_view", "PlateEvents",
    "structure_damage_events", "overgrowth_packets", "turret_offense",
    "turret_attack_damage",
]

# Legacy controller timings, the MINIONS §3.4/§3.1 defaults (U-6).
SWEEP_INTERVAL_S = 0.25
GIVE_UP_S = 4.0
IGNORE_S = 0.5
ATTACK_MEMORY_S = 2.0
WAYPOINT_MARGIN = 25.0          # legacy WAYPOINT_MARGIN (MINIONS §2.5, L)
FIRST_WAVE_END_S = 59.0         # wave 0 units spawn 30.0..34.8, wave 1 at 60
NO_PRIORITY = 99
# Super minions deal 12.5 % to non-turret structures (wiki, MINIONS §4.3 L);
# DMG.45 already applies 0.60, so the raw amount is scaled by 0.125/0.60.
SUPER_BUILDING_SCALE = 0.125 / 0.60
SIEGE_TURRET_BONUS = 1.4        # Anti-tower Socks TurretDamageBonus 0.4
TURRET_GLOBAL_GOLD = (50., 25., 25., 50.)
INHIBITOR_RADIUS, NEXUS_RADIUS = 213.75, 304.0    # client pathfinding radii (TOWERS §1.4)
FIRST_TURRET_GOLD = 300.
INHIBITOR_LAST_HIT_GOLD = 50.

# --- lane geometry (client base_srx.materials.bin via geometry.json) --------
_GEOMETRY = json.loads((Path(__file__).resolve().parents[1] / "data" / "modern" / "26.19"
                        / "geometry.json").read_text())
LANE_BOT, LANE_MID, LANE_TOP = 0, 1, 2
_LANE_NAMES = ("bot", "mid", "top")


def _lane_tables():
    raw = [np.asarray(_GEOMETRY["lane_paths"][name], np.float32) for name in _LANE_NAMES]
    length = max(len(p) for p in raw)
    paths = np.zeros((2, 3, length, 2), np.float32)
    for lane, p in enumerate(raw):
        for team, q in enumerate((p, p[::-1])):          # Order->Chaos; Chaos reversed
            paths[team, lane, :len(q)] = q
            paths[team, lane, len(q):] = q[-1]
    barracks = np.zeros((2, 3, 2), np.float32)
    for b in _GEOMETRY["barracks"]:
        barracks[b["team"], b["lane"]] = b["position"]
    return paths, np.asarray([len(p) for p in raw], np.int32), barracks


LANE_PATHS, LANE_PATH_LEN, BARRACKS = _lane_tables()


# --- minion stats -----------------------------------------------------------
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
    missile_speed: Any          # 0 = melee (hits at launch)
    acquisition_range: Any
    upgrade: Any                # latched upgrade index U


def minion_spawn_stats(minion_type, now, team=0) -> MinionStats:
    """Stats of a lane minion spawning at ``now`` (U latched at spawn, U-1).

    HP/AD/armor/gold per the client MinionUpgradeConfig (MINIONS §1.3); the
    0.55/0.60 damage ratios are *not* baked in (applied at DMG.45, README X-2).
    Move speed is the base for the spawn time; use ``minion_move_speed`` each
    tick for the global increases and the side-lane buff.
    """
    k = jnp.clip(jnp.asarray(minion_type, jnp.int32), 0, 3)
    u = M.upgrade_index_at(now)
    s = M.minion_upgrade_stats(k, u, team)
    tab = lambda v: jnp.asarray(v, jnp.float32)[k]
    return MinionStats(s.max_hp, s.max_hp, s.attack_damage, s.armor, s.magic_resist,
                       tab(M.ATTACK_RANGE), tab(M.ATTACK_SPEED), M.base_move_speed(now) + 0. * s.max_hp,
                       tab(M.GAMEPLAY_RADIUS), s.gold, s.xp, tab(M.WINDUP_S), tab(M.MISSILE_SPEED),
                       tab(M.ACQUISITION_RANGE), u + 0 * k)


def minion_move_speed(ai, units: WorldUnits, now):
    """(N,) minion move speed: time increases + side-lane bonus, soft-capped.

    Non-minions keep ``units.move_speed``. Lane comes from the AI's spawn-time
    lane assignment (nearest own barracks).
    """
    idx = M.wave_index_at(units.spawn_time)
    bonus = M.sidelane_bonus_move_speed(idx + 1, ai.lane, M.wave_spawn_time(idx),
                                        now - units.spawn_time)
    ms = M.move_speed_soft_cap(M.base_move_speed(now) + bonus)
    return jnp.where(units.kind == KIND_MINION, ms, units.move_speed).astype(jnp.float32)


# --- AI state ---------------------------------------------------------------
class LaneAIState(NamedTuple):
    seq: Any                # (N,) spawn_seq this memory belongs to (-1: never seen)
    target: Any             # (N,) int32 held target, -1 none
    target_seq: Any         # (N,) spawn_seq of the held target
    target_priority: Any    # (N,) MINIONS §3.1 class of the held target (minions)
    sweep_timer: Any        # (N,) seconds since last regular sweep
    since_attack: Any       # (N,) seconds holding a target without attacking it
    ignore_until: Any       # (N, N) give-up ignore window (MINIONS §3.4)
    last_attack: Any        # (N, N) last time i attacked/damaged j (2 s memory, §3.1)
    lane: Any               # (N,) 0 bot, 1 mid, 2 top (minions), -1 others
    waypoint: Any           # (N,) next lane-path point index
    first_wave: Any         # (N,) bool
    engaged: Any            # (N,) bool: first-wave minion has targeted an enemy minion
    champion_aggro: Any     # (N,) bool: turret lock came from champion protection
    warm_stacks: Any        # (N,) int32 turret Warming Up stacks (0..3)
    warm_until: Any         # (N,) seconds; stacks are 0 once now >= warm_until


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
        # Rejoin after a chase: a point is passed once the minion is closer to
        # the following point than the point itself is (nearest forward point).
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
    """One AI step for every minion and turret.

    ``champion_attacked_champion[i, j]``: champion i attacked/damaged enemy
    champion j this tick with a Call-for-Help source (attempts count, incl.
    blocked/0 damage). ``damage_events[i, j]``: i damaged j this tick (turret
    impacts on champions advance Warming Up; minion/turret hits feed the
    "attacking" memory). ``visible`` (2, N) optional team vision (default all
    visible). Champion damage on minions is ignored by design (26.10).

    Returns ``(ai, desired_target, move_goal, stop)``.
    """
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
    enemy = team[:, None] != team[None, :]
    ally = ~enemy & alive[None, :]
    is_champ = kind == KIND_CHAMPION
    is_minion_k = kind == KIND_MINION
    is_struct = (kind == KIND_TURRET) | (kind == KIND_INHIBITOR) | (kind == KIND_NEXUS)
    cac = jnp.asarray(champion_attacked_champion, bool) & is_champ[:, None] & is_champ[None, :]
    dmg = jnp.asarray(damage_events, bool)

    # Aggression memory: champions only through champion-on-champion events
    # (26.10: champion hits on minions never aggro); others through damage.
    events = jnp.where(is_champ[:, None], cac, dmg)
    last_attack = jnp.where(events, now, ai.last_attack).astype(jnp.float32)
    in_windup = jnp.asarray(att.windup_left, jnp.float32) > 0.
    winding_on = in_windup[:, None] & (jnp.asarray(att.target, jnp.int32)[:, None] == cols[None, :])
    attacking = ((now - last_attack) <= ATTACK_MEMORY_S) | winding_on
    attacking = attacking & alive[:, None]

    # ---------------- minions ----------------
    minion = is_minion_k & alive
    unengaged = ai.first_wave & ~ai.engaged
    acq = jnp.asarray(M.ACQUISITION_RANGE, jnp.float32)[sub]
    first_acq = jnp.asarray(M.FIRST_ACQUISITION_RANGE, jnp.float32)[sub]
    wake = jnp.asarray(M.WAKE_UP_RANGE, jnp.float32)[sub]
    scan = jnp.where(is_champ[None, :], jnp.where(unengaged, wake, acq)[:, None],
                     jnp.where(is_minion_k[None, :], jnp.where(unengaged, first_acq, acq)[:, None],
                               acq[:, None] + radius[None, :]))
    targetable_kind = is_champ | is_minion_k | is_struct
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

    # Call for Help: a strictly better P1-P4 class switches immediately,
    # except when holding a turret (not first wave) or mid-windup (§3.3).
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

    # First-wave spread (§2.8, U-4 default): while enemy first-wave melee are
    # candidates, closest-minion picks go to enemy melee ``k mod 3``, then the
    # least-attacked, then the closest.
    unit_k = jnp.round((jnp.asarray(units.spawn_time, jnp.float32) - M.WAVE_FIRST_S) / M.WAVE_UNIT_GAP_S
                       ).astype(jnp.int32)
    fw_melee = is_minion_k & (sub == M.MinionType.MELEE) & ai.first_wave & alive
    closest_min = prio == M.TargetPriority.CLOSEST_MINION
    restrict = ai.first_wave & jnp.any(cand & fw_melee[None, :] & closest_min, axis=1)
    cand_acq = cand & ~(restrict[:, None] & closest_min & ~fw_melee[None, :])
    onehot = (target[:, None] == cols[None, :]) & minion[:, None]
    # attackers[m, c] = allied (to m) minions currently targeting c
    same_team = (team[:, None] == team[None, :]).astype(jnp.float32)
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

    # ---------------- turrets ----------------
    turret = (kind == KIND_TURRET) & alive
    t_range = d <= (jnp.asarray(units.attack_range, jnp.float32)[:, None] + radius[:, None] + radius[None, :])
    t_valid = (turret[:, None] & enemy & alive[None, :] & jnp.asarray(units.targetable, bool)[None, :]
               & vis & (is_champ | is_minion_k)[None, :] & t_range)
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

    # ---------------- outputs ----------------
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


# --- attack packets ---------------------------------------------------------
def turret_attack_damage(units: WorldUnits, now):
    """(N,) turret AD growth by tier and time (TOWERS §1.1); 0 for others."""
    tier = jnp.clip(jnp.asarray(units.sub, jnp.int32), 0, 3)
    return jnp.where(units.kind == KIND_TURRET, T.attack_damage(tier, jnp.asarray(now, jnp.float32)),
                     0.).astype(jnp.float32)


def attack_packets(units: WorldUnits, launch: AttackLaunch, *, now, ai: LaneAIState,
                   pushing_bonus=None, pushing_divisor=None,
                   turret_minion_shot_mitigated=False) -> D.Packets:
    """(N,) basic-attack packets for minion and turret attacks launched now.

    Minions: ``AD`` (+ Minion Slayer ``f * target current HP`` vs lane
    minions, one combined packet; siege x1.4 vs turrets; super x0.125/0.60 vs
    inhibitor/Nexus). PHYSICAL, ``BASIC_ATTACK``; the 0.55/0.60 class ratio is
    left to DMG.45. Optional Minion Pushing (N,) arrays: ``pushing_bonus`` of
    the attacker joins ``amp`` and the target's ``pushing_divisor`` divides
    raw, minion-vs-minion only (MINIONS §4.4).

    Turrets: vs champions ``AD(tier, now) * (1 + 0.5 * stacks)`` PHYSICAL
    (the world's turret Offense must carry 30 % armor pen, see
    ``turret_offense``); vs minions the %-max-HP shot as TRUE damage so the
    minion loses exactly ``f * max_hp`` (README X-3, MINIONS U-14), or
    PHYSICAL raw with ``turret_minion_shot_mitigated=True`` (TOWERS §4.2).

    Damage is computed at launch: ranged packets ride the missile and land
    unchanged (Warming stacks cannot change between a launch and its impact
    because the period 1.2005 s exceeds the longest flight, 0.75 s).
    """
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
    """Patch a world Offense: turrets get 30 % armor pen (item 1500) and
    ``is_turret`` (bypasses Plated Steelcaps-style basic-attack reduction)."""
    t = units.kind == KIND_TURRET
    return off._replace(percent_armor_pen=jnp.where(t, T.ARMOR_PENETRATION, off.percent_armor_pen).astype(jnp.float32),
                        is_turret=off.is_turret | t)


# --- structures -------------------------------------------------------------
class TowersState(NamedTuple):
    turret: T.TurretState   # (N,) vmapped; tier 0..3 turrets, 4 inhibitor, 5 Nexus
    is_structure: Any       # (N,) bool
    team: Any               # (N,) int32
    lane: Any               # (N,) int32 (-1: nexus/none)
    prereq: Any             # (N,) int32 structure that must die first, -1 none
    targetable: Any         # (N,) bool vulnerability (TOWERS §2)
    first_turret_taken: Any  # () bool

    # Static per-unit structure stats for the world's unit arrays (TOWERS §1).
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
        """Game-start AD; use ``turret_attack_damage(units, now)`` per tick."""
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
    """``init_towers`` from a ``modern_world.WorldConfig`` (static unit layout).

    Uses ``cfg.unit_kind/unit_sub/unit_team/unit_lane`` and, when present,
    ``cfg.structure_prereq`` for the lane chain (team-level Nexus-turret and
    Nexus rules are always derived here).
    """
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
    """Structure state aligned with world units.

    ``lane`` (N,) is each structure's lane (0 bot, 1 mid, 2 top; any value
    for Nexus turrets / Nexus / non-structures), e.g. from geometry.json.
    Prerequisites: inner <- outer, inhibitor turret <- inner, inhibitor <-
    inhibitor turret of the same team and lane. Nexus turrets need any own
    inhibitor down; the Nexus additionally both Nexus turrets (TOWERS §2).
    Outer Overgrowth cooldown starts at 0:10 (U11); other lane turrets start
    theirs when they become targetable.
    """
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
    """Structure bookkeeping at the start of a tick (TOWERS §9 step 1).

    Respawn (Nexus turret 180 s at 40 %, inhibitor 300 s full) and regen
    (inhibitor turret 3/s and Nexus turret 6/s segment-capped, inhibitor
    15/s, Nexus 20/s), vulnerability chain, Overgrowth start on becoming
    targetable, backdoor refresh (enemy lane minion within 1000, U2) and
    Overgrowth appearance (suppressed by enemy champions/lane minions within
    the turret's edge attack range, U2). ``towers.turret.hp`` is
    authoritative for structures: write ``structure_unit_view`` back into the
    world units after this call.
    """
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
    # Untargetable lane turrets cannot grow a crystal (their clock is inf).
    adv = adv._replace(growth_active=adv.growth_active & lane_turret,
                       backdoor_until=adv.backdoor_until.astype(jnp.float32))
    return towers._replace(turret=adv, targetable=targetable)


def structure_unit_view(towers: TowersState, units: WorldUnits):
    """``(hp, alive, targetable)`` (N,) for the world to write back."""
    s = towers.is_structure
    hp = jnp.where(s, towers.turret.hp, units.hp).astype(jnp.float32)
    return hp, jnp.where(s, towers.turret.hp > 0, units.alive), jnp.where(s, towers.targetable, units.targetable)


def turret_defense(towers: TowersState, units: WorldUnits, *, now):
    """(N,) ``(armor, mr)`` for the damage pass.

    Turrets: 60 (outer: -15 per decay step 11:00-14:00) + Bulwark
    ``30 + 5 * (n - 1)`` per live stack, ``n`` = alive enemy champions within
    850 (TOWERS §5.2-5.3). Inhibitor/Nexus: 20 armor / 0 MR (C5/U9).
    Others: their unit stats. Fortification and the Lane Swap Detector do
    not exist on SR in 26.19 (TOWERS §7.4). Backdoor DR is not a resist:
    see ``structure_defense_mods``.
    """
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
    """(N,) ``(received_mult, invulnerable)`` for structures.

    Reinforced Armor (backdoor) x0.2 on alive turrets with no enemy lane
    minion near in the last 3 s (TOWERS §4.4; it also applies to true
    damage, which ``Defense.received_mult`` currently skips: shared change).
    Untargetable structures take no damage (TOWERS §2). Others: 1 / False.
    """
    now = jnp.asarray(now, jnp.float32)
    turret = towers.is_structure & (jnp.asarray(units.kind) == KIND_TURRET) & (towers.turret.hp > 0)
    backdoor = turret & (now >= towers.turret.backdoor_until)
    return (jnp.where(backdoor, 0.2, 1.).astype(jnp.float32),
            towers.is_structure & ~towers.targetable)


class PlateEvents(NamedTuple):
    """Structure reward events this tick, (N,) by structure slot.

    The world splits ``plate_gold + first_turret_gold`` among eligible
    champions of ``rewarded_team`` (``modern_towers.local_reward_eligible``),
    pays ``global_gold`` to every champion of that team (alive or dead) and
    ``last_hit_gold`` to the inhibitor's last hitter. Destroy XP is 0.
    """
    plates: Any             # int32 plates claimed this tick (the 5th = destruction)
    plate_gold: Any         # local plate gold (sum over claimed plates)
    destroyed: Any          # bool
    global_gold: Any        # per champion of rewarded_team
    first_turret_gold: Any  # 300 local (shared) on the game's first turret kill
    last_hit_gold: Any      # inhibitor 50 to the last-hitting champion
    rewarded_team: Any      # int32 team that destroyed / damaged it
    nexus_destroyed: Any    # bool: game over


def structure_damage_events(towers: TowersState, hp_before, hp_after, *, now):
    """Plates broken / structures destroyed between ``hp_before`` and ``hp_after``.

    Plate k is claimed the first time ``hp <= max_hp * (0.9, 0.75, 0.55, 0.3,
    0)[k]``; plates 1-4 each add an independent 20 s Bulwark stack that
    applies from the next tick's packets. Plate gold is ``plate_value(now)``
    (outer decays 120 -> 80 from 11:00). Destruction pays the global gold
    (50/25/25/50), the first turret 300 local, an inhibitor 50 to its last
    hitter, and starts Nexus-turret (180 s) / inhibitor (300 s) respawn.
    """
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
    """Consume Crystalline Overgrowth on enemy-champion basic-attack hits.

    ``champion_attack_hit[i, j]``: champion i's basic attack hit structure j
    this tick. ``team_level`` (2,) average champion level per team (1v1: the
    champion's level; fractional, U6). Emits one TRUE packet per consumed
    crystal (src = attacking champion so plates/assists credit it; no
    damage modifiers, no vamp) to append *after* the attack's own packet
    (U7). Not consumed while backdoor is active (TOWERS §6.3).
    Returns ``(towers, Packets (N,))``.
    """
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
