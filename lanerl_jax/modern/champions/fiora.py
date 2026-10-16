"""Fiora (114), 26.19. Rules FIO.* and their evidence: docs/modern/CHAMPIONS.md; numbers from the pinned client
JSON (``modern/data/26.19/champions``). Vitals are kit state on (C, N) enemy champions: a direction (north, east,
south or west of the target) a hit must come from. Riposte uses the world's parry and debuff-immunity hooks
(``KitDefense.parry``, ``cc_immune``, ``on_parry``); Bladework's second attack uses ``KitAttackMods.force_crit``;
Lunge's stab applies on-hit effects through ``KitOut.on_hit_target``."""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..core import damage as D
from ..core.types import KIND_CHAMPION, KIND_MINION, KIND_MONSTER, KIND_WARD, CCOut, Dash
from ..data.champions import spell, stat
from ..items.catalog import ItemStats
from ..items.effects.core import Debuffs, neutral_debuffs
from ..runes.catalog import ChampionTraits
from .core import (CODE_FIORA_VITAL, NEVER, KitAttackMods, KitDefense, KitOut, cc_matrix, cooldown_row, due,
                   enemies, f32, holder_rows, is_structure, later, make_cast_id, mana_row, out, ranked, scalar,
                   target_dist)

NAME = "Fiora"
ID = 114
SKILL_ORDER = (0, 2, 1, 0, 0, 3, 0, 2, 0, 2, 3, 2, 2, 1, 1, 3, 1, 1)    # default ranks per level: Q > E > W
TRAITS = ChampionTraits(has_immobilize=True, resource="mana")          # rune legality: W stuns
BASE_MS = stat(NAME, "baseMoveSpeedModifiable")                       # 345

_P = spell(NAME, "Passive")["calculations"]["PassiveHealAmount"]["mFormulaParts"][0]
P_HEAL = (_P["mStartValue"], _P["mEndValue"])           # 35 -> 100 by level
P_DAMAGE = scalar(NAME, "Passive", "PassiveDamageBase")  # 3% target max health
P_DAMAGE_AD = scalar(NAME, "Passive", "PassiveDamageADRatio")   # + 0.04% per bonus AD (4% per 100)
P_MS_DURATION = scalar(NAME, "Passive", "MovementSpeedDuration")  # 2 s decay (wiki 1.85)
P_IDENTIFY = 1.75                                       # wiki: a vital becomes targetable 1.75 s after identifying
P_LINGER = 13.25                                        # then lasts 13.25 s (15 in all, tooltip)
P_RANGE = 1200.0                                        # wiki: identified within 1200 (centre range)
P_KEEP = 1350.0                                         # ... and kept while Fiora stays within 1350 (edge range)
P_HALF_ANGLE = 45.0                                     # wiki: a 90 degree arc
N, E, S, W = range(4)                                   # vital directions: +y, +x, -y, -x
_DIR_X = jnp.asarray([0.0, 1.0, 0.0, -1.0], jnp.float32)
_DIR_Y = jnp.asarray([1.0, 0.0, -1.0, 0.0], jnp.float32)

Q_MIN, Q_MAX = 50.0, 400.0                              # wiki dash distance
Q_SPEED = 1100.0                                        # + 350% bonus MS (cap 1600) or + 300% (floor 850)
Q_STAB = 420.0                                          # client castRadiusSecondary (centre range)
Q_REFUND = scalar(NAME, "Q", "CDRefundPercent")         # 0.5
Q_EXECUTE_RADIUS = 0.6                                  # wiki: killable units within 60% of the stab radius

W_PARRY = scalar(NAME, "W", "ParryDuration")            # 0.75
W_POISE = 0.5                                           # wiki: the stab leaves after 0.5 s
W_SPEED = 3200.0                                        # FioraWMissile speed
W_RANGE = 750.0                                         # FioraW castRangeDisplayOverride (missile record: 900)
W_HALF_WIDTH = 70.0                                     # mLineWidth (wiki width 140)
W_CC = scalar(NAME, "W", "CCDuration")                  # 2 s slow + cripple, or stun
W_SLOW = -scalar(NAME, "W", "MSSlowPercent")            # 50%
W_CRIPPLE = -scalar(NAME, "W", "AttackSlowPercent")     # 25% (14.4)

E_WINDOW = scalar(NAME, "E", "BuffDuration")            # 3 s for the two attacks (wiki: 4)
E_SLOW = -scalar(NAME, "E", "SlowPercent")              # 30%
E_SLOW_DURATION = scalar(NAME, "E", "SlowDuration")     # 1
E_RANGE = 25.0                                          # wiki
BASE_CRIT_DAMAGE = 2.0                                  # the world base (stat_pipeline); Bladework adds the bonus above it

R_RANGE = 500.0                                         # castRange (centre range)
R_DELAY = 0.5                                           # wiki: the four vitals appear after 0.5 s
R_DURATION = scalar(NAME, "R", "MarkDuration")          # 8
R_MS_RADIUS = scalar(NAME, "R", "MSRingRadius")         # 500 (edge range; wiki 550)
R_ZONE = scalar(NAME, "R", "HealRingRadius")            # 600 (26.19)
R_ZONE_TIME = scalar(NAME, "R", "HealDuration")         # 5
R_HEAL_AD = scalar(NAME, "R", "Ratio")                  # 0.7 bonus AD per second
R_HEAL_PERIOD = 0.25                                    # V25.14
UNIT_TARGET_RANGE = (0.0, 0.0, 0.0, R_RANGE)

Q_FLAGS = D.TAG_BASIC_ATTACK | D.TAG_ACTIVE_SPELL | D.PROP_LIFESTEAL   # wiki: basic damage, also spell damage
W_FLAGS = D.TAG_ACTIVE_SPELL | D.TAG_AOE
P_FLAGS = D.TAG_PROC


class State(NamedTuple):
    v_dir: Any              # (C, N) int32 vital direction, -1 none
    v_from: Any             # (C, N) targetable from
    v_until: Any            # (C, N) expires
    v_group: Any            # (C, N) int32 next group: 0 south/west, 1 north/east
    v_hit_at: Any           # (C, N) a vital struck this tick (its true damage leaves next tick)
    v_proc: Any             # (C, N) bool: true damage due next tick
    ms_from: Any            # (C,) last vital proc (decaying move speed)
    q_dash: Any             # (C,) bool: the lunge is moving
    q_cast_at: Any
    q_cast_id: Any
    q_half_at: Any          # (C,) the stab struck an enemy this tick: Lunge's cooldown starts halved
    w_until: Any            # (C,) parry end
    w_fire_at: Any          # stab departure
    w_dx: Any
    w_dy: Any
    w_fly: Any
    w_ox: Any
    w_oy: Any
    w_dist: Any
    w_cast_id: Any
    w_parried: Any          # (C,) bool: negated a hostile immobilizing effect
    w_struck: Any           # (C,) int32 champion the stab hit, -1 none
    w_struck_at: Any
    e_left: Any             # (C,) int32 empowered attacks left (2, 1, 0)
    e_until: Any
    e_cast_id: Any
    r_target: Any           # (C,) int32 challenged champion, -1 none
    r_seq: Any
    r_from: Any             # vitals visible from
    r_until: Any
    r_hit: Any              # (C, 4) bool vitals struck
    r_zone_x: Any
    r_zone_y: Any
    r_zone_until: Any
    r_zone_next: Any        # next heal tick
    r_rank: Any
    r_near: Any             # (C,) bool: within the move-speed ring of the challenged champion


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    i = jnp.zeros((c,), jnp.int32)
    zn = jnp.zeros((c, n), jnp.float32)
    return State(jnp.full((c, n), -1, jnp.int32), zn + NEVER, zn - NEVER, jnp.zeros((c, n), jnp.int32), zn - NEVER,
                 jnp.zeros((c, n), bool), z - NEVER,
                 f, z - NEVER, i, z - NEVER,
                 z - NEVER, z + NEVER, z, z, f, z, z, z, i, f, i - 1, z - NEVER,
                 i, z - NEVER, i,
                 i - 1, i, z + NEVER, z - NEVER, jnp.zeros((c, 4), bool), z, z, z - NEVER, z + NEVER, i, f)


def _mine(kctx) -> Any:
    return kctx.champion_id == ID


def by_level(pair, level) -> Any:
    lv = jnp.clip(jnp.asarray(level, jnp.float32), 1.0, 18.0)
    return pair[0] + (pair[1] - pair[0]) * (lv - 1.0) / 17.0


def vital_damage(bonus_ad, max_hp) -> Any:
    """True damage of one vital: 3% (+4% per 100 bonus AD) of the target's maximum health."""
    return (P_DAMAGE + P_DAMAGE_AD * bonus_ad) * max_hp


def facing(dx, dy, direction) -> Any:
    """The source at offset ``(dx, dy)`` from the target lies within the 90 degree arc of ``direction``."""
    d = jnp.sqrt(dx * dx + dy * dy)
    cosang = (dx * _DIR_X[jnp.clip(direction, 0, 3)] + dy * _DIR_Y[jnp.clip(direction, 0, 3)]) \
        / jnp.maximum(d, 1e-3)
    return (direction >= 0) & (d > 1e-3) & (cosang >= jnp.cos(jnp.deg2rad(P_HALF_ANGLE)) - 1e-6)


def lunge_speed(move_speed) -> Any:
    """1100 + 350% of bonus move speed (at most 1600), or + 300% of a negative bonus (at least 850)."""
    bonus = move_speed - BASE_MS
    return jnp.where(bonus >= 0, jnp.minimum(Q_SPEED + 3.5 * bonus, 1600.0),
                     jnp.maximum(Q_SPEED + 3.0 * bonus, 850.0))


def _base_cd(state: State, kctx) -> Any:
    """Base cooldowns; Lunge's halves on the tick its stab struck an enemy (every hook reports the same value)."""
    cd = cooldown_row(NAME, kctx.ranks)
    half = jnp.abs(state.q_half_at - kctx.now) < 0.5 * kctx.dt
    cd = cd.at[:, 0].multiply(jnp.where(half, 1.0 - Q_REFUND, 1.0))
    return jnp.where(_mine(kctx)[:, None], cd, 0.0)


def _unit_dir(dx, dy):
    d = jnp.sqrt(dx * dx + dy * dy)
    ok = d > 1e-3
    return jnp.where(ok, dx / jnp.maximum(d, 1e-3), 1.0), jnp.where(ok, dy / jnp.maximum(d, 1e-3), 0.0), d


def cast(state: State, kctx, units, order) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    r = kctx.ranks
    now = f32(kctx.now)
    parrying = now < state.w_until - 1e-6
    free = m & kctx.alive & ~kctx.stunned & ~kctx.silenced & ~parrying
    rooted = jnp.zeros_like(m) if kctx.rooted is None else kctx.rooted
    cost = mana_row(NAME, r)
    ready = (r > 0) & (kctx.cooldowns <= 0) & (kctx.mana[:, None] >= cost)
    want = lambda s: (order.slot == s) & free          # noqa: E731
    ux, uy, dist = _unit_dir(order.x - kctx.x, order.y - kctx.y)
    ms = jnp.zeros((c,), jnp.float32) + BASE_MS if kctx.move_speed is None else kctx.move_speed

    # Lunge: dash 50-400 toward the cursor; every other ability is castable during it.
    q = want(0) & ready[:, 0] & ~rooted & ~state.q_dash
    q_len = jnp.clip(dist, Q_MIN, Q_MAX)
    dash = Dash(q, f32(kctx.x + ux * q_len), f32(kctx.y + uy * q_len), f32(lunge_speed(ms)),
                jnp.full((c,), -1, jnp.int32), jnp.zeros((c,), bool))
    # Riposte: 0.75 s parry (lockout), the stab after 0.5 s in the direction fixed at cast.
    w = want(1) & ready[:, 1]
    # Bladework: two empowered attacks; attack reset.
    e = want(2) & ready[:, 2] & (state.e_left == 0)
    # Grand Challenge: an enemy champion within 500 (centre range), not one already challenged.
    t = jnp.clip(order.target, 0, n - 1)
    valid_r = (order.target >= 0) & units.alive[t] & (units.kind[t] == KIND_CHAMPION) & (units.team[t] != kctx.team) \
        & (target_dist(kctx, units, order.target) <= R_RANGE) & (state.r_target != order.target)
    rr = want(3) & ready[:, 3] & valid_r

    started = q | w | e | rr
    slot = jnp.where(q, 0, jnp.where(w, 1, jnp.where(e, 2, jnp.where(rr, 3, -1)))).astype(jnp.int32)
    cid = jnp.where(started, make_cast_id(kctx, jnp.maximum(slot, 0)), 0).astype(jnp.int32)
    state = state._replace(
        q_dash=state.q_dash | q, q_cast_at=jnp.where(q, now, state.q_cast_at),
        q_cast_id=jnp.where(q, cid, state.q_cast_id),
        w_until=jnp.where(w, later(kctx, W_PARRY), state.w_until),
        w_fire_at=jnp.where(w, later(kctx, W_POISE), state.w_fire_at),
        w_dx=jnp.where(w, f32(ux), state.w_dx), w_dy=jnp.where(w, f32(uy), state.w_dy),
        w_cast_id=jnp.where(w, cid, state.w_cast_id), w_parried=jnp.where(w, False, state.w_parried),
        w_struck=jnp.where(w, -1, state.w_struck).astype(jnp.int32),
        e_left=jnp.where(e, 2, state.e_left).astype(jnp.int32), e_until=jnp.where(e, later(kctx, E_WINDOW),
                                                                                  state.e_until),
        e_cast_id=jnp.where(e, cid, state.e_cast_id),
        r_target=jnp.where(rr, order.target, state.r_target).astype(jnp.int32),
        r_seq=jnp.where(rr, units.spawn_seq[t], state.r_seq).astype(jnp.int32),
        r_from=jnp.where(rr, later(kctx, R_DELAY), state.r_from),
        r_until=jnp.where(rr, later(kctx, R_DURATION), state.r_until),
        r_hit=jnp.where(rr[:, None], False, state.r_hit), r_rank=jnp.where(rr, r[:, 3], state.r_rank))
    mana = jnp.sum(jnp.where(jnp.stack([q, w, e, rr], -1), cost, 0.0), axis=1)
    # Lunge's cooldown starts when the stab resolves (halved on a hit); Riposte and Bladework post-effect.
    cd = jnp.stack([jnp.zeros_like(q), jnp.zeros_like(q), jnp.zeros_like(q), rr], -1)
    return state, out(c, n, dash=dash, mana_cost=f32(mana), cooldown_start=cd, base_cooldown=_base_cd(state, kctx),
                      attack_reset=e, cast_started=started, cast_slot=slot, cast_id=cid,
                      cast_lockout=f32(jnp.where(w, W_PARRY, 0.0)))


def _stab_target(kctx, units, vital_open, r_col) -> Any:
    """(C,) Lunge's stab target within its radius by the wiki priority, -1 none: a challenged champion with a vital
    facing Fiora, any champion with one, a challenged champion, the lowest-health champion under 20%, a unit Lunge
    kills within 60% of the radius, the nearest champion, a minion it kills, any minion or monster, a structure."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    rx, ry = units.x[None, :] - kctx.x[:, None], units.y[None, :] - kctx.y[:, None]
    dist = jnp.sqrt(rx * rx + ry * ry)
    inside = dist <= Q_STAB
    foe = enemies(kctx, units)
    dmg = _q_damage(kctx)[:, None]
    champ = foe & inside & (units.kind == KIND_CHAMPION)[None, :]
    frac = (units.hp / jnp.maximum(units.max_hp, 1.0))[None, :]
    mm = foe & inside & ((units.kind == KIND_MINION) | (units.kind == KIND_MONSTER))[None, :]
    struct = inside & (units.team[None, :] != kctx.team[:, None]) & units.alive[None, :] \
        & is_structure(units.kind)[None, :]
    tiers = [(champ & vital_open & r_col, dist), (champ & vital_open, dist), (champ & r_col, dist),
             (champ & (frac < 0.2), jnp.broadcast_to(frac, dist.shape)),
             ((champ | mm) & (dist <= Q_EXECUTE_RADIUS * Q_STAB) & (units.hp[None, :] < dmg), dist),
             (champ, dist), (mm & (units.kind == KIND_MINION)[None, :] & (units.hp[None, :] < dmg), dist),
             (mm, dist), (struct, dist)]
    pick = jnp.full((c,), -1, jnp.int32)
    found = jnp.zeros((c,), bool)
    for tier, key in tiers:
        any_ = jnp.any(tier, axis=1)
        pick = jnp.where(~found & any_, jnp.argmin(jnp.where(tier, key, jnp.inf), axis=1), pick).astype(jnp.int32)
        found = found | any_
    return pick


def _q_damage(kctx) -> Any:
    r = kctx.ranks[:, 0]
    return ranked(NAME, "Q", "Damage", r) + ranked(NAME, "Q", "bADRatio", r) * kctx.bonus_ad


def periodic(state: State, kctx, units) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    alive = kctx.alive
    dead = ~alive
    r = kctx.ranks
    now = f32(kctx.now)
    cols = jnp.arange(n)[None, :]
    z = jnp.zeros((c, n), jnp.float32)
    champ_foe = enemies(kctx, units) & (units.kind == KIND_CHAMPION)[None, :]
    rx, ry = units.x[None, :] - kctx.x[:, None], units.y[None, :] - kctx.y[:, None]
    dist = jnp.sqrt(rx * rx + ry * ry)

    # Duelist's Dance: last tick's struck vitals deal their true damage now.
    pid = make_cast_id(kctx, CODE_FIORA_VITAL)
    p_v = D.packets(state.v_proc & units.alive[None, :], kctx.unit[:, None], cols,
                    vital_damage(kctx.bonus_ad[:, None], units.max_hp[None, :]), D.TRUE, P_FLAGS,
                    cast_id=pid[:, None])

    # Grand Challenge state (its target gets no passive vital meanwhile).
    rt = jnp.clip(state.r_target, 0, n - 1)
    r_alive = (state.r_target >= 0) & units.alive[rt] & (units.spawn_seq[rt] == state.r_seq)
    r_on = r_alive & (now < state.r_until)
    r_col = r_on[:, None] & (cols == rt[:, None])
    # A challenged target dying with a vital struck, or all four struck: the Victory Zone.
    died = (state.r_target >= 0) & ~r_alive & (now < state.r_until) & jnp.any(state.r_hit, axis=1)
    all4 = r_on & jnp.all(state.r_hit, axis=1)
    zone = (died | all4) & m
    r_zone_x = jnp.where(zone, units.x[rt], state.r_zone_x)
    r_zone_y = jnp.where(zone, units.y[rt], state.r_zone_y)
    r_zone_until = jnp.where(zone, f32(now + R_ZONE_TIME), state.r_zone_until)
    r_zone_next = jnp.where(zone, now, state.r_zone_next)
    r_target = jnp.where(zone | ~r_alive | (now >= state.r_until), -1, state.r_target).astype(jnp.int32)

    # Passive vitals: identify on enemy champions within 1200; keep while Fiora stays within 1350 (edge).
    has = state.v_dir >= 0
    keep = has & champ_foe & (dist <= P_KEEP + units.radius[None, :]) & (now < state.v_until) & ~r_col
    v_dir = jnp.where(keep, state.v_dir, -1)
    new = m[:, None] & alive[:, None] & ~keep & champ_foe & (dist <= P_RANGE) & ~r_col
    pick = jnp.where(kctx.rand[:, None] < 0.5, 0, 1) if kctx.rand is not None else jnp.zeros((c, 1), jnp.int32)
    group = state.v_group                                                    # 0: south or west, 1: north or east
    d_new = jnp.where(group == 0, jnp.where(pick == 0, S, W), jnp.where(pick == 0, N, E))
    v_dir = jnp.where(new, d_new, v_dir).astype(jnp.int32)
    v_from = jnp.where(new, f32(now + P_IDENTIFY), state.v_from)
    v_until = jnp.where(new, f32(now + P_IDENTIFY + P_LINGER), jnp.where(keep, state.v_until, -NEVER))
    v_group = jnp.where(new, 1 - group, group).astype(jnp.int32)

    # Lunge: the stab when the dash ends (no stab if the dash was interrupted before moving: death).
    landed = state.q_dash & (~kctx.dashing if kctx.dashing is not None else (now > state.q_cast_at + 0.4)) \
        & (now > state.q_cast_at + 0.5 * kctx.dt)
    q_done = state.q_dash & (landed | dead)
    q_fire = q_done & alive
    vital_open = _vital_open(state._replace(v_dir=v_dir, v_from=v_from), kctx, units, r_col, r_on)
    tgt = jnp.where(q_fire, _stab_target(kctx, units, vital_open, r_col), -1).astype(jnp.int32)
    stab = tgt >= 0
    tq = jnp.clip(tgt, 0, n - 1)
    p_q = D.packets(stab, kctx.unit, tq, _q_damage(kctx), D.PHYSICAL, Q_FLAGS, cast_id=state.q_cast_id)
    hit_enemy = stab & ~is_structure(units.kind[tq])

    # Riposte: the stab leaves after 0.5 s and stops on the first enemy champion; earlier units take damage too.
    w_go = m & alive & due(kctx, state.w_fire_at)
    w_fly = (state.w_fly | w_go) & alive
    ox, oy = jnp.where(w_go, kctx.x, state.w_ox), jnp.where(w_go, kctx.y, state.w_oy)
    d0 = jnp.where(w_go, 0.0, state.w_dist)
    d1 = d0 + W_SPEED * kctx.dt
    wx, wy = units.x[None, :] - ox[:, None], units.y[None, :] - oy[:, None]
    w_along = wx * state.w_dx[:, None] + wy * state.w_dy[:, None]
    w_perp = jnp.abs(wy * state.w_dx[:, None] - wx * state.w_dy[:, None])
    rad = units.radius[None, :]
    swept = w_fly[:, None] & enemies(kctx, units) & (w_along - rad <= jnp.minimum(d1, W_RANGE)[:, None]) \
        & (w_along + rad >= d0[:, None]) & (w_perp <= W_HALF_WIDTH + rad)
    champ_hit = swept & (units.kind == KIND_CHAMPION)[None, :]
    first_c = jnp.argmin(jnp.where(champ_hit, w_along, jnp.inf), axis=1)
    stop = jnp.any(champ_hit, axis=1)
    stop_at = jnp.where(stop, jnp.min(jnp.where(champ_hit, w_along, jnp.inf), axis=1), jnp.inf)
    w_mask = swept & (w_along <= stop_at[:, None])
    w_raw = ranked(NAME, "W", "BaseDamage", r[:, 1]) + kctx.ap
    p_w = D.packets(w_mask, kctx.unit[:, None], cols, w_raw[:, None], D.MAGIC, W_FLAGS,
                    cast_id=state.w_cast_id[:, None])
    struck = stop[:, None] & (cols == first_c[:, None])
    stun = struck & state.w_parried[:, None]
    slow = struck & ~state.w_parried[:, None]
    w_done = stop | (d1 >= W_RANGE)
    w_struck = jnp.where(stop, first_c, state.w_struck).astype(jnp.int32)
    w_struck_at = jnp.where(stop, now, state.w_struck_at)
    w_end = m & (state.w_until > -NEVER / 2) & due(kctx, state.w_until)

    # Bladework window.
    e_end = m & (state.e_left > 0) & (due(kctx, state.e_until) | dead)

    # Victory Zone: heals Fiora within 600 of it every 0.25 s for 5 s.
    zone_on = m & (now < r_zone_until) & (now >= r_zone_next - 1e-6)
    in_zone = jnp.sqrt((kctx.x - r_zone_x) ** 2 + (kctx.y - r_zone_y) ** 2) <= R_ZONE
    heal = jnp.where(zone_on & in_zone & alive,
                     (ranked(NAME, "R", "HealPerSecond", state.r_rank) + R_HEAL_AD * kctx.bonus_ad) * R_HEAL_PERIOD,
                     0.0)
    r_zone_next = jnp.where(zone_on, r_zone_next + R_HEAL_PERIOD, r_zone_next)

    state = state._replace(
        v_dir=v_dir, v_from=f32(v_from), v_until=f32(v_until), v_group=v_group,
        v_proc=jnp.zeros_like(state.v_proc),
        q_dash=state.q_dash & ~q_done, q_half_at=jnp.where(hit_enemy, now, state.q_half_at),
        w_fire_at=jnp.where(w_go | dead, NEVER, state.w_fire_at), w_fly=w_fly & ~w_done, w_ox=f32(ox), w_oy=f32(oy),
        w_dist=f32(d1), w_struck=w_struck, w_struck_at=f32(w_struck_at),
        w_until=jnp.where(w_end | dead, -NEVER, state.w_until),
        e_left=jnp.where(e_end, 0, state.e_left).astype(jnp.int32),
        r_target=r_target, r_zone_x=f32(r_zone_x), r_zone_y=f32(r_zone_y), r_zone_until=f32(r_zone_until),
        r_zone_next=f32(r_zone_next),
        r_near=r_on & (dist[jnp.arange(c), rt] <= R_MS_RADIUS + units.radius[rt]))
    sd = jnp.where(stun, W_CC, 0.0)
    cc = CCOut(f32(sd), z, z, z, jnp.where(slow, W_SLOW, 0.0), jnp.where(slow, W_CC, 0.0),
               jnp.where(struck, state.w_cast_id[:, None], 0).astype(jnp.int32))
    cd = jnp.zeros((c, 4), bool).at[:, 0].set(q_done & m).at[:, 1].set(w_end).at[:, 2].set(e_end)
    return state, out(c, n, packets=D.concat_packets(p_v, p_q, p_w), cc=cc, heal=f32(heal), cooldown_start=cd,
                      base_cooldown=_base_cd(state, kctx), on_hit_target=jnp.where(stab, tgt, -1).astype(jnp.int32),
                      on_hit_raw=_q_damage(kctx), on_hit_cast_id=state.q_cast_id)


def _vital_open(state: State, kctx, units, r_col, r_on) -> Any:
    """(C, N) an open vital faces Fiora's current position (passive vital, or any unstruck Grand Challenge vital)."""
    n = units.x.shape[0]
    dx, dy = kctx.x[:, None] - units.x[None, :], kctx.y[:, None] - units.y[None, :]
    passive = (state.v_dir >= 0) & (kctx.now >= state.v_from - 1e-6) & facing(dx, dy, state.v_dir)
    r_vis = r_on & (kctx.now >= state.r_from - 1e-6)
    grand = jnp.zeros(passive.shape, bool)
    for d in range(4):
        grand = grand | (~state.r_hit[:, d:d + 1] & facing(dx, dy, jnp.full(dx.shape, d, jnp.int32)))
    del n
    return jnp.where(r_col, r_vis[:, None] & grand, passive)


def on_attack(state: State, kctx, units, launch) -> tuple[State, KitOut]:
    """Bladework: the first empowered attack slows its target (even if dodged); each consumes a charge."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    target = holder_rows(launch.target, kctx)
    went = holder_rows(launch.launched, kctx) & m & kctx.alive & (target >= 0) & (state.e_left > 0)
    first = went & (state.e_left == 2)
    mask = first[:, None] & (jnp.arange(n)[None, :] == target[:, None])
    slow, sid = cc_matrix(mask, E_SLOW, state.e_cast_id[:, None])
    z = jnp.zeros((c, n), jnp.float32)
    left = jnp.where(went, state.e_left - 1, state.e_left)
    done = went & (left == 0)
    state = state._replace(e_left=left.astype(jnp.int32))
    cd = jnp.zeros((c, 4), bool).at[:, 2].set(done)
    return state, out(c, n, cc=CCOut(z, z, z, z, f32(slow), jnp.where(mask, E_SLOW_DURATION, 0.0), sid),
                      cooldown_start=cd, base_cooldown=_base_cd(state, kctx))


def on_hit(state: State, kctx, units, launch, dodging) -> tuple[State, KitOut]:
    return state, out(kctx.unit.shape[0], units.x.shape[0], base_cooldown=_base_cd(state, kctx))


def on_damage(state: State, kctx, units, report) -> tuple[State, KitOut]:
    """Duelist's Dance: any damage Fiora deals to a champion (not reactive) from the direction of an open vital
    strikes it: heal now, true damage next tick, decaying move speed; Grand Challenge records which of its four."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    p = report.packets
    res = report.resolved                                             # partial reports (tests) fall back
    dealt = getattr(res, "final", None)
    dealt = getattr(res, "health_loss", p.raw) if dealt is None else dealt
    final = jnp.where(p.valid, dealt, 0.0)
    dst = jnp.clip(p.dst, 0, n - 1)
    landed = (p.src[None, :] == kctx.unit[:, None]) & m[:, None] & (final[None, :] > 0) \
        & ((p.flags & D.PROP_REACTIVE) == 0)[None, :] & (p.cast_id[None, :] != make_cast_id(kctx, CODE_FIORA_VITAL)[:, None])
    on = jnp.any(landed[:, :, None] & (dst[None, :, None] == jnp.arange(n)[None, None, :]), axis=1)    # (C, N)
    rt = jnp.clip(state.r_target, 0, n - 1)
    r_on = (state.r_target >= 0) & (kctx.now < state.r_until)
    r_col = r_on[:, None] & (jnp.arange(n)[None, :] == rt[:, None])
    dx, dy = kctx.x[:, None] - units.x[None, :], kctx.y[:, None] - units.y[None, :]
    passive = on & ~r_col & (state.v_dir >= 0) & (kctx.now >= state.v_from - 1e-6) & facing(dx, dy, state.v_dir)
    r_vis = r_on & (kctx.now >= state.r_from - 1e-6)
    r_dirs = jnp.stack([facing(dx[jnp.arange(c), rt], dy[jnp.arange(c), rt], jnp.full((c,), d, jnp.int32))
                        for d in range(4)], -1)
    r_struck = (on[jnp.arange(c), rt] & r_vis)[:, None] & r_dirs & ~state.r_hit
    first_r = jnp.argmax(r_struck, axis=1)
    r_new = jnp.any(r_struck, axis=1)
    r_hit = state.r_hit | (r_new[:, None] & (jnp.arange(4)[None, :] == first_r[:, None]))
    procs = passive | (r_new[:, None] & r_col)
    n_procs = jnp.sum(procs, axis=1).astype(jnp.float32)
    heal = jnp.where(m & kctx.alive, by_level(P_HEAL, kctx.level) * n_procs, 0.0)
    state = state._replace(v_proc=state.v_proc | procs, v_dir=jnp.where(passive, -1, state.v_dir).astype(jnp.int32),
                           ms_from=jnp.where(jnp.any(procs, axis=1), f32(kctx.now), state.ms_from), r_hit=r_hit)
    return state, out(c, n, heal=f32(heal))


def on_parry(state: State, kctx, parried) -> State:
    """Riposte negated a hostile immobilizing effect this tick: the stab will stun instead of slowing."""
    m = _mine(kctx) & parried & (kctx.now < state.w_until)
    return state._replace(w_parried=state.w_parried | m)


def on_takedown(state: State, kctx, units, kills) -> State:
    return state


def stats(state: State, kctx) -> ItemStats:
    """Bladework attack speed; Duelist's Dance move speed (decaying over 2 s); Grand Challenge's while near."""
    m = _mine(kctx)
    e = m & (state.e_left > 0)
    pct = ranked(NAME, "R", "PercentMS", kctx.ranks[:, 3])
    t = kctx.now - state.ms_from
    decay = jnp.where(state.r_near & (kctx.now < state.r_until), 1.0, jnp.clip(1.0 - t / P_MS_DURATION, 0.0, 1.0))
    return ItemStats(attack_speed=f32(jnp.where(e, ranked(NAME, "E", "ASPercent", kctx.ranks[:, 2]), 0.0)),
                     percent_move_speed=f32(jnp.where(m, pct * decay, 0.0)))


def defense(state: State, kctx) -> KitDefense:
    c = kctx.unit.shape[0]
    o = jnp.ones((c,), jnp.float32)
    return KitDefense(o, jnp.zeros((c,), bool), o, jnp.zeros((c,), jnp.float32),
                      parry=_mine(kctx) & kctx.alive & (kctx.now < state.w_until))


def cc_immune(state: State, kctx) -> Any:
    """(C,) Riposte's debuff immunity."""
    return _mine(kctx) & kctx.alive & (kctx.now < state.w_until)


def attack_mods(state: State, kctx) -> KitAttackMods:
    """Bladework: +25 range; the first attack cannot crit; the second always crits for 160-200% (+ bonus crit
    damage)."""
    m = _mine(kctx)
    c = kctx.unit.shape[0]
    z = jnp.zeros((c,), jnp.float32)
    e = m & (state.e_left > 0)
    second = m & (state.e_left == 1)
    mult = ranked(NAME, "E", "AttackTwoPercentTAD", kctx.ranks[:, 2]) + (kctx.crit_damage - BASE_CRIT_DAMAGE)
    return KitAttackMods(f32(jnp.where(e, E_RANGE, 0.0)), jnp.zeros((c,), bool), jnp.zeros((c,), bool),
                         m & (state.e_left == 2), z, z, jnp.zeros((c,), bool),
                         force_crit=second, crit_mult=f32(jnp.where(second, mult, 0.0)))


def debuffs(state: State, kctx, units) -> Debuffs:
    """Riposte's 25% attack-speed cripple on the champion its stab slowed (2 s)."""
    n = units.x.shape[0]
    on = _mine(kctx) & (state.w_struck >= 0) & ~state.w_parried & (kctx.now < state.w_struck_at + W_CC)
    tgt = jnp.where(on, state.w_struck, n)
    cripple = jnp.zeros((n + 1,), jnp.float32).at[tgt].max(jnp.where(on, W_CRIPPLE, 0.0))[:n]
    return neutral_debuffs(n)._replace(attack_speed_cripple=cripple)


__all__ = ["State", "init", "cast", "periodic", "on_attack", "on_hit", "on_damage", "on_parry", "on_takedown",
           "stats", "defense", "cc_immune", "attack_mods", "debuffs", "facing", "vital_damage", "lunge_speed",
           "by_level"]
