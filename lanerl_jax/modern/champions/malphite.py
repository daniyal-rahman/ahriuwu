"""Malphite (54), 26.19. Rules MAL.* and their evidence: docs/modern/CHAMPIONS.md; numbers from the pinned client
JSON (``modern/data/26.19/champions``). Granite Shield is a kit-sourced shield (``D.SOURCE_KIT``) that refills."""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..core import damage as D
from ..core.types import KIND_MONSTER, CCOut, Dash
from ..data.champions import spell
from ..items.catalog import ItemStats
from ..items.effects.core import Debuffs, neutral_debuffs, shield_grants
from ..runes.catalog import ChampionTraits
from .core import (NEVER, KitAttackMods, KitDefense, KitOut, cc_matrix, center_dist, cooldown_row, due, enemies,
                   f32, gather, holder_rows, is_structure, later, make_cast_id, mana_row, out, ranked, scalar,
                   target_dist)

NAME = "Malphite"
ID = 54
SKILL_ORDER = (0, 2, 1, 0, 0, 3, 0, 2, 0, 2, 3, 2, 2, 1, 1, 3, 1, 1)    # default ranks per level: Q > E > W
TRAITS = ChampionTraits(has_immobilize=True, resource="mana")          # rune legality: R knocks up
_P = spell(NAME, "Passive")["calculations"]
P_SHIELD = _P["TotalShield"]["mFormulaParts"][0]["mCoefficient"]       # 10% max health
_PCD = _P["PassiveCooldown"]["mFormulaParts"][0]
P_CD = (_PCD["mLevel1Value"], tuple((b["mLevel"], b["mAdditionalBonusAtThisLevel"]) for b in _PCD["mBreakpoints"]))
P_FULL = 0.999                                        # refill when below this share of the full shield
Q_RANGE = 625.0
Q_CAST_TIME = 0.25
Q_SPEED = 1200.0
Q_AP = 0.6
Q_DURATION = scalar(NAME, "Q", "SlowDuration")        # 3
W_WINDOW = 6.0                                        # wiki: the next attack within 6 s
W_AFTERSHOCK = scalar(NAME, "W", "ThunderclapBuffDuration")   # 5 s of aftershocks
W_RANGE_BONUS = 50.0
W_AP = scalar(NAME, "W", "ThunderclapBaseDamageAPRatio")       # 0.2
W_ARMOR = scalar(NAME, "W", "ThunderclapBaseDamageArmorRatio")  # 0.15
W_SPLASH_AP = scalar(NAME, "W", "ThunderclapSplashAP")         # 0.3
W_SPLASH_ARMOR = scalar(NAME, "W", "ThunderclapSplashArmor")   # 0.15
W_CONE_RANGE = scalar(NAME, "W", "ThunderClapSplashRange")     # 400 (from Malphite's position)
W_CONE_HALF_ANGLE = 45.0                              # no client or wiki value: INFERRED-L
W_MONSTER = scalar(NAME, "W", "MonsterDamageMod")     # 1.8
W_SHIELD_MULT = scalar(NAME, "W", "BonusArmorPassiveMultiplier")   # x3 while Granite Shield holds
E_CAST_TIME = 0.2419                                  # client mCastTime
E_RADIUS = 400.0                                      # client castRadius / wiki (center to edge)
E_ARMOR = 0.4
E_AP = 0.6
E_DURATION = scalar(NAME, "E", "Duration")            # 3
R_RANGE = 1000.0                                      # cast at max range toward the cursor beyond it
R_SPEED = 1500.0                                      # + move speed
R_RADIUS = 325.0                                      # wiki (client castRadius 270)
R_AP = 0.9
R_KNOCKUP = scalar(NAME, "R", "KnockupDuration")      # 1.5
P_SIZE_PER_ARMOR = 0.08 / 100.0                       # wiki Shard of the Monolith: 8% size per 100 armor
P_SIZE_CAP = 0.35
UNIT_TARGET_RANGE = (Q_RANGE, 0.0, 0.0, 0.0)

Q_FLAGS = D.TAG_ACTIVE_SPELL
W_FLAGS = D.TAG_ON_HIT | D.TAG_ACTIVE_SPELL
W_SPLASH_FLAGS = D.TAG_AOE | D.TAG_ON_HIT
E_FLAGS = D.TAG_AOE | D.TAG_ACTIVE_SPELL
R_FLAGS = D.TAG_AOE | D.TAG_ACTIVE_SPELL | D.PROP_ULTIMATE


class State(NamedTuple):
    last_hurt: Any          # (C,) last health loss (Granite Shield clock)
    q_pending: Any          # cast time, then the shard flies
    q_hit_at: Any
    q_target: Any
    q_seq: Any
    q_cast_id: Any
    q_ms: Any               # Malphite's stolen move speed
    q_ms_until: Any
    w_on: Any               # empowered next attack
    w_until: Any
    w_shock_until: Any      # aftershock window
    w_cast_id: Any
    e_pending: Any
    e_fire_at: Any
    e_cast_id: Any
    cripple_until: Any      # (C, N)
    cripple: Any            # (C,) strength at cast rank
    r_pending: Any          # dashing
    r_cast_at: Any
    r_land_at: Any          # planned landing (used when the world gives no dash state)
    r_x: Any
    r_y: Any
    r_cast_id: Any


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    i = jnp.zeros((c,), jnp.int32)
    return State(z - NEVER, f, z + NEVER, i - 1, i, i, z, z - NEVER, f, z, z - NEVER, i,
                 f, z + NEVER, i, jnp.zeros((c, n), jnp.float32) - NEVER, z, f, z, z + NEVER, z, z, i)


def _mine(kctx) -> Any:
    return kctx.champion_id == ID


def _base_cd(kctx) -> Any:
    return jnp.where(_mine(kctx)[:, None], cooldown_row(NAME, kctx.ranks), 0.0)


def shield_cooldown(level: Any) -> Any:
    """Granite Shield's no-damage delay: 8 s, 7 from level 7, 6 from 13 (``PassiveCooldown`` breakpoints)."""
    lv = jnp.asarray(level, jnp.float32)
    base, bps = P_CD
    return base + sum(jnp.where(lv >= l, add, 0.0) for l, add in bps)


def _shielded(kctx) -> Any:
    return (jnp.zeros(kctx.unit.shape, jnp.float32) if kctx.kit_shield is None else kctx.kit_shield) > 0.0


def _cone(kctx, units, tx, ty, half_angle_deg: float, length: float) -> Any:
    """(C, N) units whose hitbox reaches the cone from the holder toward (tx, ty)."""
    dx, dy = tx - kctx.x, ty - kctx.y
    d = jnp.maximum(jnp.sqrt(dx * dx + dy * dy), 1e-3)
    ux, uy = dx / d, dy / d
    rx, ry = units.x[None, :] - kctx.x[:, None], units.y[None, :] - kctx.y[:, None]
    dist = jnp.sqrt(rx * rx + ry * ry)
    ang = jnp.arctan2(jnp.abs(rx * uy[:, None] - ry * ux[:, None]), rx * ux[:, None] + ry * uy[:, None])
    slack = jnp.arcsin(jnp.clip(units.radius[None, :] / jnp.maximum(dist, 1e-3), 0.0, 1.0))
    return (dist <= length + units.radius[None, :]) & ((ang <= jnp.deg2rad(half_angle_deg) + slack)
                                                       | (dist <= units.radius[None, :]))


def cast(state: State, kctx, units, order) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    r = kctx.ranks
    now = f32(kctx.now)
    busy = state.q_pending | state.e_pending | state.r_pending
    free = m & kctx.alive & ~kctx.stunned & ~kctx.silenced & ~busy
    cost = mana_row(NAME, r)
    ready = (r > 0) & (kctx.cooldowns <= 0) & (kctx.mana[:, None] >= cost)
    want = lambda s: (order.slot == s) & free          # noqa: E731
    t = jnp.clip(order.target, 0, n - 1)
    dist_q = target_dist(kctx, units, order.target)
    valid_q = (order.target >= 0) & units.alive[t] & units.targetable[t] & (units.team[t] != kctx.team) \
        & ~is_structure(units.kind[t]) & (dist_q <= Q_RANGE + units.radius[t])
    q = want(0) & ready[:, 0] & valid_q
    w = want(1) & ready[:, 1]
    e = want(2) & ready[:, 2]
    rr = want(3) & ready[:, 3]
    started = q | w | e | rr
    slot = jnp.where(q, 0, jnp.where(w, 1, jnp.where(e, 2, jnp.where(rr, 3, -1)))).astype(jnp.int32)
    cid = jnp.where(started, make_cast_id(kctx, jnp.maximum(slot, 0)), 0).astype(jnp.int32)
    # Unstoppable Force: dash to the point, at most 1000 away, at 1500 + move speed.
    dx, dy = order.x - kctx.x, order.y - kctx.y
    d = jnp.sqrt(dx * dx + dy * dy)
    reach = jnp.minimum(d, R_RANGE)
    k = jnp.where(d > 1e-3, reach / jnp.maximum(d, 1e-3), 0.0)
    rx, ry = kctx.x + dx * k, kctx.y + dy * k
    ms = jnp.zeros((c,), jnp.float32) if kctx.move_speed is None else kctx.move_speed
    speed = R_SPEED + ms
    dash = Dash(rr, f32(rx), f32(ry), f32(speed), jnp.full((c,), -1, jnp.int32), jnp.zeros((c,), bool))
    state = state._replace(
        q_pending=state.q_pending | q,
        q_hit_at=jnp.where(q, f32(now + Q_CAST_TIME + dist_q / Q_SPEED), state.q_hit_at),
        q_target=jnp.where(q, order.target, state.q_target).astype(jnp.int32),
        q_seq=jnp.where(q, units.spawn_seq[t], state.q_seq).astype(jnp.int32),
        q_cast_id=jnp.where(q, cid, state.q_cast_id),
        w_on=state.w_on | w, w_until=jnp.where(w, later(kctx, W_WINDOW), state.w_until),
        w_shock_until=jnp.where(w, later(kctx, W_AFTERSHOCK), state.w_shock_until),
        w_cast_id=jnp.where(w, cid, state.w_cast_id),
        e_pending=state.e_pending | e, e_fire_at=jnp.where(e, later(kctx, E_CAST_TIME), state.e_fire_at),
        e_cast_id=jnp.where(e, cid, state.e_cast_id),
        r_pending=state.r_pending | rr, r_cast_at=jnp.where(rr, now, state.r_cast_at), r_land_at=jnp.where(rr, f32(now + reach / speed), state.r_land_at),
        r_x=jnp.where(rr, f32(rx), state.r_x), r_y=jnp.where(rr, f32(ry), state.r_y),
        r_cast_id=jnp.where(rr, cid, state.r_cast_id))
    mana = jnp.sum(jnp.where(jnp.stack([q, w, e, rr], -1), cost, 0.0), axis=1)
    lock = jnp.where(q, Q_CAST_TIME, jnp.where(e, E_CAST_TIME, 0.0))
    return state, out(c, n, dash=dash, mana_cost=f32(mana), cooldown_start=jnp.stack([q, w, e, rr], -1),
                      base_cooldown=_base_cd(kctx), attack_reset=w, cast_started=started, cast_slot=slot,
                      cast_id=cid, cast_lockout=f32(lock))


def periodic(state: State, kctx, units) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    alive = kctx.alive
    dead = ~alive
    r = kctx.ranks
    now = f32(kctx.now)
    cols = jnp.arange(n)[None, :]
    z = jnp.zeros((c, n), jnp.float32)

    # Granite Shield: refill to 10% max health after the delay without health loss.
    full = P_SHIELD * kctx.max_hp
    have = jnp.zeros((c,), jnp.float32) if kctx.kit_shield is None else kctx.kit_shield
    refill = m & alive & (now - state.last_hurt >= shield_cooldown(kctx.level) - 1e-6) & (have < P_FULL * full)
    shield = shield_grants(jnp.where(refill, full, 0.0), D.SHIELD_ALL + D.SOURCE_SHIFT * D.SOURCE_KIT, jnp.inf)

    # Seismic Shard lands: magic damage, a 3 s slow, and Malphite takes the move speed it removed.
    qt = jnp.clip(state.q_target, 0, n - 1)
    q_ok = units.alive[qt] & (units.spawn_seq[qt] == state.q_seq)
    q_due = state.q_pending & (due(kctx, state.q_hit_at) | ~q_ok)
    q_land = q_due & q_ok
    q_raw = ranked(NAME, "Q", "BaseDamage", r[:, 0]) + Q_AP * kctx.ap
    p_q = D.packets(q_land, kctx.unit, qt, q_raw, D.MAGIC, Q_FLAGS, cast_id=state.q_cast_id)
    q_slow = ranked(NAME, "Q", "SpeedSteal", r[:, 0]) / 100.0
    mask_q = q_land[:, None] & (cols == qt[:, None])
    slow_q, sid_q = cc_matrix(mask_q, q_slow[:, None], state.q_cast_id[:, None])
    q_ms = jnp.where(q_land, q_slow * units.move_speed[qt], state.q_ms)
    q_ms_until = jnp.where(q_land, f32(now + kctx.dt + Q_DURATION), state.q_ms_until)

    # Ground Slam at the end of its cast time: damage and a 3 s attack-speed cripple.
    e_due = state.e_pending & (due(kctx, state.e_fire_at) | dead)
    e_fire = e_due & alive
    e_hit = e_fire[:, None] & enemies(kctx, units) & (center_dist(kctx, units) <= E_RADIUS + units.radius[None, :])
    e_raw = ranked(NAME, "E", "BaseDamage", r[:, 2]) + E_ARMOR * kctx.armor + E_AP * kctx.ap
    p_e = D.packets(e_hit, kctx.unit[:, None], cols, e_raw[:, None], D.MAGIC, E_FLAGS,
                    cast_id=state.e_cast_id[:, None])
    cripple_until = jnp.where(e_hit, f32(now + kctx.dt + E_DURATION), state.cripple_until)
    cripple = jnp.where(e_fire, ranked(NAME, "E", "ASReduction", r[:, 2]) / 100.0, state.cripple)

    # Unstoppable Force lands where the dash ends (terrain can stop it short): damage and a 1.5 s knock-up (not
    # reduced by tenacity).
    if kctx.dashing is None:
        landed, lx, ly = due(kctx, state.r_land_at), state.r_x, state.r_y
    else:
        landed, lx, ly = ~kctx.dashing & (now > state.r_cast_at + 0.5 * kctx.dt), kctx.x, kctx.y
    r_due = state.r_pending & (landed | dead)
    r_fire = r_due & alive
    rdx, rdy = units.x[None, :] - lx[:, None], units.y[None, :] - ly[:, None]
    r_hit = r_fire[:, None] & enemies(kctx, units) \
        & (jnp.sqrt(rdx * rdx + rdy * rdy) <= R_RADIUS + units.radius[None, :])
    r_raw = ranked(NAME, "R", "BaseDamage", r[:, 3]) + R_AP * kctx.ap
    p_r = D.packets(r_hit, kctx.unit[:, None], cols, r_raw[:, None], D.MAGIC, R_FLAGS,
                    cast_id=state.r_cast_id[:, None])
    up = jnp.where(r_hit, R_KNOCKUP, 0.0)

    w_end = state.w_on & (due(kctx, state.w_until) | dead)
    state = state._replace(
        q_pending=state.q_pending & ~q_due, q_ms=f32(q_ms), q_ms_until=f32(q_ms_until),
        w_on=state.w_on & ~w_end, w_shock_until=jnp.where(dead, -NEVER, state.w_shock_until),
        e_pending=state.e_pending & ~e_due, cripple_until=f32(cripple_until), cripple=f32(cripple),
        r_pending=state.r_pending & ~r_due)
    cc = CCOut(z, z, z, f32(up), f32(slow_q), f32(jnp.where(mask_q, Q_DURATION, 0.0)),
               jnp.where(r_hit, state.r_cast_id[:, None], sid_q).astype(jnp.int32))
    return state, out(c, n, packets=D.concat_packets(p_q, p_e, p_r), cc=cc, shield=shield,
                      base_cooldown=_base_cd(kctx))


def on_attack(state: State, kctx, units, launch) -> tuple[State, KitOut]:
    return state, out(kctx.unit.shape[0], units.x.shape[0], base_cooldown=_base_cd(kctx))


def on_hit(state: State, kctx, units, launch, dodging) -> tuple[State, KitOut]:
    """Thunderclap: the empowered attack's bonus, and aftershock cones on every landed attack in the window."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    target = holder_rows(launch.target, kctx)
    t = jnp.clip(target, 0, n - 1)
    hit = holder_rows(launch.launched, kctx) & m & kctx.alive & (target >= 0)
    land = hit & ~gather(dodging, target)
    monster = jnp.where(units.kind == KIND_MONSTER, W_MONSTER, 1.0)
    w = hit & state.w_on
    w_land = w & land
    w_raw = (ranked(NAME, "W", "ThunderclapBaseDamage", kctx.ranks[:, 1]) + W_AP * kctx.ap + W_ARMOR * kctx.armor) \
        * monster[t]
    p_w = D.packets(w_land, kctx.unit, t, w_raw, D.PHYSICAL, W_FLAGS, cast_id=state.w_cast_id)
    shock = land & (kctx.now < state.w_shock_until)
    in_cone = shock[:, None] & enemies(kctx, units) & _cone(kctx, units, units.x[t], units.y[t],
                                                            W_CONE_HALF_ANGLE, W_CONE_RANGE)
    s_raw = ranked(NAME, "W", "ThunderclapSplashDamage", kctx.ranks[:, 1]) + W_SPLASH_AP * kctx.ap \
        + W_SPLASH_ARMOR * kctx.armor
    p_s = D.packets(in_cone, kctx.unit[:, None], jnp.arange(n)[None, :], s_raw[:, None] * monster[None, :],
                    D.PHYSICAL, W_SPLASH_FLAGS, cast_id=state.w_cast_id[:, None])
    state = state._replace(w_on=state.w_on & ~w)
    return state, out(c, n, packets=D.concat_packets(p_w, p_s), base_cooldown=_base_cd(kctx))


def on_damage(state: State, kctx, units, report) -> tuple[State, KitOut]:
    """Granite Shield clock: health lost to anything (shield-absorbed damage does not count)."""
    p = report.packets
    lost = p.raw > 0 if report.resolved is None else report.resolved.health_loss > 0
    hurt = jnp.any((p.valid & lost)[None, :] & (p.dst[None, :] == kctx.unit[:, None]), axis=1)
    state = state._replace(last_hurt=jnp.where(_mine(kctx) & hurt, f32(kctx.now), state.last_hurt))
    return state, out(kctx.unit.shape[0], units.x.shape[0])


def on_takedown(state: State, kctx, units, kills) -> State:
    return state


def stats(state: State, kctx) -> ItemStats:
    """Thunderclap passive armor (tripled while Granite Shield holds) and Seismic Shard's stolen move speed."""
    m = _mine(kctx)
    pct = jnp.where(m, _w_armor_pct(kctx), 0.0)
    ms = jnp.where(m & (kctx.now < state.q_ms_until), state.q_ms, 0.0)
    return ItemStats(percent_armor=f32(pct), move_speed=f32(ms))


def _w_armor_pct(kctx) -> Any:
    pct = jnp.where(kctx.ranks[:, 1] > 0, ranked(NAME, "W", "BonusArmorPassive", kctx.ranks[:, 1]), 0.0)
    return pct * jnp.where(_shielded(kctx), W_SHIELD_MULT, 1.0)


def size(state: State, kctx) -> Any:
    """Shard of the Monolith: 8% size per 100 total armor, at most 35%; Thunderclap's bonus armor does not count
    (wiki), so it is divided back out of the composed armor."""
    armor = kctx.armor / (1.0 + _w_armor_pct(kctx))
    return f32(jnp.where(_mine(kctx), jnp.clip(P_SIZE_PER_ARMOR * armor, 0.0, P_SIZE_CAP), 0.0))


def defense(state: State, kctx) -> KitDefense:
    c = kctx.unit.shape[0]
    o = jnp.ones((c,), jnp.float32)
    return KitDefense(o, jnp.zeros((c,), bool), o, jnp.zeros((c,), jnp.float32))


def attack_mods(state: State, kctx) -> KitAttackMods:
    m = _mine(kctx)
    c = kctx.unit.shape[0]
    z = jnp.zeros((c,), jnp.float32)
    w = m & state.w_on
    return KitAttackMods(f32(jnp.where(w, W_RANGE_BONUS, 0.0)), jnp.zeros((c,), bool), m & state.r_pending,
                         jnp.zeros((c,), bool), z, z, w)


def debuffs(state: State, kctx, units) -> Debuffs:
    """Ground Slam attack-speed cripple, (N,)."""
    n = units.x.shape[0]
    on = _mine(kctx)[:, None] & (kctx.now < state.cripple_until)
    return neutral_debuffs(n)._replace(
        attack_speed_cripple=f32(jnp.max(jnp.where(on, state.cripple[:, None], 0.0), axis=0)))


__all__ = ["State", "init", "cast", "periodic", "on_attack", "on_hit", "on_damage", "on_takedown", "stats",
           "defense", "attack_mods", "debuffs", "shield_cooldown"]
