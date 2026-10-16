"""Yone (777), 26.19. Rules YON.* and their evidence: docs/modern/CHAMPIONS.md; numbers from the pinned client
JSON (``modern/data/26.19/champions``). Yone has no mana: his bar (Flow) shows Soul Unbound's remaining time
(``resource``). Mortal Steel and Spirit Cleave scale their cast times and cooldowns with bonus attack speed and
ignore ability haste (``Trait_NotBenefitFromHaste``); the kit hands the world a base cooldown that cancels haste.
Azakana attacks use ``KitAttackMods.magic_split``; Mortal Steel applies on-attack and on-hit effects to its first
target (``KitOut.on_hit_target`` + ``on_attack_proc``); Fate Sealed's teleport is a kit blink."""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..core import damage as D
from ..core.types import KIND_CHAMPION, KIND_MINION, KIND_MONSTER, CCOut, Dash
from ..data.champions import spell
from ..items.catalog import ItemStats
from ..items.effects.core import Debuffs, ShieldGrant, neutral_debuffs
from ..runes.catalog import ChampionTraits
from .core import (CODE_YONE_E, CODE_YONE_Q3, NEVER, KitAttackMods, KitDefense, KitOut, cc_matrix, cooldown_row,
                   due, enemies, f32, holder_rows, later, make_cast_id, out, ranked, scalar)

NAME = "Yone"
ID = 777
SKILL_ORDER = (0, 1, 2, 0, 0, 3, 0, 1, 0, 1, 3, 1, 1, 2, 2, 3, 2, 2)    # default ranks per level: Q > W > E
TRAITS = ChampionTraits(has_immobilize=True, resource="none")          # rune legality: Q3 knocks up; Flow, no mana
UNIT_TARGET_RANGE = (0.0, 0.0, 0.0, 0.0)

P_SPLIT = scalar(NAME, "Passive", "MagicDamageSplit")                 # 0.5: Azakana attacks are half magic
P_CRIT_TO_AD = scalar(NAME, "Passive", "YoneCritToAD") / 100.0       # 0.5 AD per 1% crit chance above 100%
P_CRIT_MULT = scalar(NAME, "Passive", "CritChanceMultiplier")         # +100%: crit chance doubled
P_CRIT_DAMAGE = scalar(NAME, "Passive", "CritDamageMod")              # crits deal 95%

Q_RANGE = 450.0                                       # thrust, edge range
Q_HALF_WIDTH = scalar(NAME, "Q", "Width") / 2.0       # wiki: thrust width 80 (edge range)
Q_AD = scalar(NAME, "Q", "ADRatio")                   # 1.1 AD (crits)
Q_STACK_TIME = scalar(NAME, "Q", "BuffDuration")      # 6
Q_COOLDOWN = 4.0                                      # static (client cooldownTime 4, NotBenefitFromHaste)
Q_CD_PER_AS = scalar(NAME, "Q", "QAttackSpeedCDPercent")              # 0.6 of bonus AS ...
Q_CD_MAX = scalar(NAME, "Q", "QAttackSpeedCDMax")                    # ... up to 66.7% shorter
Q_CAST = scalar(NAME, "Q", "QBaseCastTime")           # 0.35
Q_CAST_PER_AS = scalar(NAME, "Q", "QAttackSpeedCastReductionPercent")  # 0.41667 of bonus AS ...
Q_CAST_MAX = scalar(NAME, "Q", "QAttackSpeedCastReductionMax")        # ... up to 50% shorter
Q3_DASH = scalar(NAME, "Q", "Q3MaxDistance")          # 450
Q3_SPEED = scalar(NAME, "Q", "Q3DashSpeed")           # 1500 (dash and whirlwind)
Q3_DASH_RADIUS = scalar(NAME, "Q", "Q3AoESize")       # 100, edge range around the dash
Q3_RANGE = 950.0                                      # YoneQ3Missile castRange (wiki whirlwind 1050 centre range)
Q3_HALF_WIDTH = 80.0                                  # YoneQ3Missile mMissileWidth (wiki width 160)
Q3_KNOCKUP = scalar(NAME, "Q", "Q3KnockupDuration")   # 0.75

W_COOLDOWN = 14.0                                     # static base
W_CD_PER_AS = scalar(NAME, "W", "WAttackSpeedCDPercent")            # 0.6
W_CD_MAX = scalar(NAME, "W", "WAttackSpeedCDMax")                  # 57.1% (14 -> 6)
W_CAST = scalar(NAME, "W", "WBaseCastTime")           # 0.5
W_CAST_PER_AS = scalar(NAME, "W", "WAttackSpeedCastReductionPercent")
W_CAST_MAX = scalar(NAME, "W", "WAttackSpeedCastReductionMax")
W_RADIUS = 600.0                                      # wiki: edge range 600 (650 from 50 behind Yone)
W_BACK = 50.0
W_HALF_ANGLE = scalar(NAME, "W", "ConeWidth")         # 40 degrees (wiki 80 degree cone)
W_SHIELD_TIME = scalar(NAME, "W", "ShieldDuration")   # 1.5
W_SHIELD_AD = scalar(NAME, "W", "ShieldbADRatio")     # 0.65 bonus AD
W_FIRST_CHAMP = scalar(NAME, "W", "FirstChampShieldMultiplier")     # +100%
W_MORE_CHAMP = scalar(NAME, "W", "SecondChampShieldMultiplier")     # +50% each further champion
_WS = spell(NAME, "W")["calculations"]["WShield"]["mFormulaParts"][0]
W_SHIELD = (_WS["mStartValue"], _WS["mEndValue"])     # 40 -> 90 by level
_WM = spell(NAME, "W")["calculations"]["MinimumDamageMinions"]["mFormulaParts"][0]
W_MINION_FLOOR = _WM["mLevel1Value"]                  # 30, +10/level to 8, +20/level 9-13, +40/level from 14
W_MONSTER_CAP = tuple(spell(NAME, "W")["calculations"]["MonsterDamageCap"]["mFormulaParts"][0]["values"])

E_DASH = scalar(NAME, "E", "EDashRange")              # 300 out; the body is left 300 behind the cast point
E_SPEED = scalar(NAME, "E", "EDashSpeed")             # 1200
E_TIME = scalar(NAME, "E", "ReturnTimer")             # 5
E_RECAST_LOCK = scalar(NAME, "E", "RecastLockout")    # 0.5
E_MS_START = scalar(NAME, "E", "StartingMS")          # 10% ...
E_MS_END = scalar(NAME, "E", "MovementSpeed")         # ... ramping to 30%
E_MS_RAMP = 3.0                                       # wiki: over the first 3 s
E_RECAST_TIME = 0.225                                 # wiki recast cast time (displacement immune)
E_RETURN_SPEED = 2950.0                               # replay-measured (YON.E3; wiki "1200, pending for test")

R_CAST = 0.75
R_LENGTH = 1000.0                                     # edge range
R_HALF_WIDTH = scalar(NAME, "R", "Width") / 2.0       # 225 wide
R_STUN = 1.0
R_BEHIND = scalar(NAME, "R", "BlinkDistanceBehindTarget")           # 200
R_GUST = 0.3                                          # wiki: the gust 0.3 s after the cast time
R_BLINK_DELAY = 0.05
R_PULL_SPEED = scalar(NAME, "R", "RKnockTowardsSpeed")              # 3000
R_KNOCKUP = scalar(NAME, "R", "RKnockupDuration")    # 0.75, after the pull
R_AD = scalar(NAME, "R", "ADRatio")                   # 0.8 bonus AD
R_LOCK = 0.45                                         # wiki: lockout after the cast time

Q_FLAGS_FIRST = D.TAG_BASIC_ATTACK | D.PROP_LIFESTEAL                    # wiki: basic damage to the first target
Q_FLAGS_REST = D.TAG_AOE | D.TAG_ACTIVE_SPELL
W_FLAGS = D.TAG_AOE | D.TAG_ACTIVE_SPELL
R_FLAGS = D.TAG_AOE | D.TAG_ACTIVE_SPELL | D.PROP_ULTIMATE
E_FLAGS = D.TAG_ACTIVE_SPELL


class State(NamedTuple):
    azakana: Any            # (C,) bool: the next attack is the Azakana sword's
    q_stacks: Any           # (C,) int32 Gathering Storm (0-2)
    q_stacks_until: Any
    q_pending: Any
    q_fire_at: Any
    q_three: Any            # (C,) bool: the wound-up cast is the empowered one
    q_dx: Any
    q_dy: Any
    q_cast_id: Any
    q3_until: Any           # Q3 dash end
    q3_fly: Any             # whirlwind travelling
    q3_ox: Any
    q3_oy: Any
    q3_dist: Any
    q3_hit: Any             # (C, N) bool: already struck by this Q3
    w_pending: Any
    w_fire_at: Any
    w_dx: Any
    w_dy: Any
    w_cast_id: Any
    e_on: Any               # (C,) bool: Spirit Form
    e_from: Any             # cast time
    e_until: Any            # automatic return
    e_body_x: Any
    e_body_y: Any
    e_mark: Any             # (C, N) stored damage per champion
    e_back_at: Any          # recast resolves (end of its cast time)
    e_cast_id: Any
    r_pending: Any
    r_fire_at: Any
    r_dx: Any
    r_dy: Any
    r_ox: Any
    r_oy: Any
    r_cast_id: Any
    r_marked: Any           # (C, N) bool
    r_gust_at: Any
    r_bx: Any               # blink point
    r_by: Any
    r_blink_at: Any
    r_pull_until: Any       # (C, N) pull end
    r_up_at: Any            # (C, N) knock-up start (pull end)
    e_bar: Any              # (C,) Soul Unbound's remaining fraction (Flow)


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    i = jnp.zeros((c,), jnp.int32)
    zn = jnp.zeros((c, n), jnp.float32)
    fn = jnp.zeros((c, n), bool)
    return State(f, i, z - NEVER, f, z + NEVER, f, z, z, i, z - NEVER, f, z, z, z, fn,
                 f, z + NEVER, z, z, i,
                 f, z - NEVER, z + NEVER, z, z, zn, z + NEVER, i,
                 f, z + NEVER, z, z, z, z, i, fn, z + NEVER, z, z, z + NEVER, zn - NEVER, zn + NEVER, z)


def _mine(kctx) -> Any:
    return kctx.champion_id == ID


def by_level(pair, level) -> Any:
    lv = jnp.clip(jnp.asarray(level, jnp.float32), 1.0, 18.0)
    return pair[0] + (pair[1] - pair[0]) * (lv - 1.0) / 17.0


def as_scaled(base, bonus_as, per, cap) -> Any:
    """Mortal Steel / Spirit Cleave cast times and cooldowns: ``base x (1 - min(per x bonus AS, cap))``."""
    return base * (1.0 - jnp.minimum(per * jnp.maximum(bonus_as, 0.0), cap))


def w_minion_floor(level) -> Any:
    lv = jnp.asarray(level, jnp.float32)
    return W_MINION_FLOOR + 10.0 * (jnp.clip(lv, 1, 8) - 1) + 20.0 * jnp.clip(lv - 8, 0, 5) \
        + 40.0 * jnp.maximum(lv - 13, 0)


def w_monster_cap(level) -> Any:
    return jnp.asarray(W_MONSTER_CAP, jnp.float32)[jnp.clip(jnp.asarray(level) - 1, 0, len(W_MONSTER_CAP) - 1)]


def _base_cd(kctx) -> Any:
    """Base cooldowns; Q and W from bonus attack speed, pre-multiplied to cancel the world's ability haste."""
    cd = cooldown_row(NAME, kctx.ranks)
    unhaste = 1.0 + kctx.ability_haste / 100.0
    cd = cd.at[:, 0].set(as_scaled(Q_COOLDOWN, kctx.bonus_attack_speed, Q_CD_PER_AS, Q_CD_MAX) * unhaste)
    cd = cd.at[:, 1].set(as_scaled(W_COOLDOWN, kctx.bonus_attack_speed, W_CD_PER_AS, W_CD_MAX) * unhaste)
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
    busy = state.q_pending | state.w_pending | state.r_pending | (state.e_back_at < NEVER / 2)
    free = m & kctx.alive & ~kctx.stunned & ~kctx.silenced & ~busy
    rooted = jnp.zeros_like(m) if kctx.rooted is None else kctx.rooted
    ready = (r > 0) & (kctx.cooldowns <= 0)
    want = lambda s: (order.slot == s) & free          # noqa: E731
    ux, uy, _ = _unit_dir(order.x - kctx.x, order.y - kctx.y)
    stacks = jnp.where(now < state.q_stacks_until, state.q_stacks, 0)

    # Mortal Steel: the empowered cast at two stacks dashes (not while rooted).
    q = want(0) & ready[:, 0] & ~((stacks >= 2) & rooted)
    q_ct = as_scaled(Q_CAST, kctx.bonus_attack_speed, Q_CAST_PER_AS, Q_CAST_MAX)
    # Spirit Cleave.
    w = want(1) & ready[:, 1]
    w_ct = as_scaled(W_CAST, kctx.bonus_attack_speed, W_CAST_PER_AS, W_CAST_MAX)
    # Soul Unbound: dash out; the recast (after 0.5 s) returns.
    e_out = want(2) & ready[:, 2] & ~state.e_on & ~rooted
    e_back = want(2) & state.e_on & (now >= state.e_from + E_RECAST_LOCK - 1e-6)
    e_dash = Dash(e_out, f32(kctx.x + ux * E_DASH), f32(kctx.y + uy * E_DASH), jnp.full((c,), E_SPEED, jnp.float32),
                  jnp.full((c,), -1, jnp.int32), jnp.zeros((c,), bool))
    # Fate Sealed.
    rr = want(3) & ready[:, 3] & ~rooted

    started = q | w | e_out | e_back | rr
    slot = jnp.where(q, 0, jnp.where(w, 1, jnp.where(e_out | e_back, 2, jnp.where(rr, 3, -1)))).astype(jnp.int32)
    cid = jnp.where(started, make_cast_id(kctx, jnp.maximum(slot, 0)), 0).astype(jnp.int32)
    three = q & (stacks >= 2)
    state = state._replace(
        q_stacks=jnp.where(three, 0, stacks).astype(jnp.int32),
        q_pending=state.q_pending | q, q_fire_at=jnp.where(q, later(kctx, q_ct), state.q_fire_at),
        q_three=jnp.where(q, three, state.q_three), q_dx=jnp.where(q, f32(ux), state.q_dx),
        q_dy=jnp.where(q, f32(uy), state.q_dy), q_cast_id=jnp.where(q, cid, state.q_cast_id),
        w_pending=state.w_pending | w, w_fire_at=jnp.where(w, later(kctx, w_ct), state.w_fire_at),
        w_dx=jnp.where(w, f32(ux), state.w_dx), w_dy=jnp.where(w, f32(uy), state.w_dy),
        w_cast_id=jnp.where(w, cid, state.w_cast_id),
        azakana=jnp.where(e_out, False, state.azakana),
        e_on=state.e_on | e_out, e_from=jnp.where(e_out, now, state.e_from),
        e_until=jnp.where(e_out, later(kctx, E_TIME), state.e_until),
        e_body_x=jnp.where(e_out, kctx.x - ux * E_DASH, state.e_body_x),
        e_body_y=jnp.where(e_out, kctx.y - uy * E_DASH, state.e_body_y),
        e_mark=jnp.where(e_out[:, None], 0.0, state.e_mark),
        e_back_at=jnp.where(e_back, later(kctx, E_RECAST_TIME), state.e_back_at),
        e_cast_id=jnp.where(e_out | e_back, cid, state.e_cast_id),
        r_pending=state.r_pending | rr, r_fire_at=jnp.where(rr, later(kctx, R_CAST), state.r_fire_at),
        r_dx=jnp.where(rr, f32(ux), state.r_dx), r_dy=jnp.where(rr, f32(uy), state.r_dy),
        r_cast_id=jnp.where(rr, cid, state.r_cast_id))
    lock = jnp.where(q, q_ct, jnp.where(w, w_ct, jnp.where(rr, R_CAST, jnp.where(e_back, E_RECAST_TIME, 0.0))))
    cd = jnp.stack([q, w, jnp.zeros_like(q), rr], -1)
    return state, out(c, n, dash=e_dash, cooldown_start=cd, base_cooldown=_base_cd(kctx), cast_started=started,
                      cast_slot=slot, cast_id=cid, cast_lockout=f32(lock))


def _crit_raw(kctx, base, ad_ratio, roll) -> Any:
    """Ability damage whose AD part crits on ``roll`` (C, N): base + ratio x AD x crit damage (with the 95%)."""
    return base[:, None] + (ad_ratio * kctx.total_ad)[:, None] * jnp.where(roll, kctx.crit_damage[:, None], 1.0)


def periodic(state: State, kctx, units) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    alive = kctx.alive
    dead = ~alive
    r = kctx.ranks
    now = f32(kctx.now)
    cols = jnp.arange(n)[None, :]
    z = jnp.zeros((c, n), jnp.float32)
    foe = enemies(kctx, units)
    rad = units.radius[None, :]
    minion = (units.kind == KIND_MINION)[None, :]
    monster = (units.kind == KIND_MONSTER)[None, :]
    roll = (kctx.rand_n if kctx.rand_n is not None else jnp.ones((c, n))) < kctx.crit_chance[:, None]
    rx, ry = units.x[None, :] - kctx.x[:, None], units.y[None, :] - kctx.y[:, None]
    dist = jnp.sqrt(rx * rx + ry * ry)

    # Mortal Steel (thrust): a 450 x 80 line from Yone; the closest enemy takes basic damage (on-hit, on-attack).
    q_due = state.q_pending & (due(kctx, state.q_fire_at) | dead)
    q_fire = q_due & alive
    along = rx * state.q_dx[:, None] + ry * state.q_dy[:, None]
    perp = jnp.abs(ry * state.q_dx[:, None] - rx * state.q_dy[:, None])
    thrust = (q_fire & ~state.q_three)[:, None] & foe & (along + rad >= 0.0) & (along - rad <= Q_RANGE) \
        & (perp <= Q_HALF_WIDTH + rad)
    first = jnp.argmin(jnp.where(thrust, along, jnp.inf), axis=1)
    q_base = ranked(NAME, "Q", "BaseDamage", r[:, 0])
    q_raw = _crit_raw(kctx, q_base, Q_AD, roll)
    is_first = thrust & (cols == first[:, None])
    p_q = D.packets(thrust, kctx.unit[:, None], cols, q_raw, D.PHYSICAL,
                    jnp.where(is_first, Q_FLAGS_FIRST, Q_FLAGS_REST) | jnp.where(roll, D.PROP_CRIT, 0),
                    cast_id=state.q_cast_id[:, None])
    q_hit_any = jnp.any(thrust, axis=1)
    q_stacks = jnp.where(q_hit_any, jnp.minimum(state.q_stacks + 1, 2), state.q_stacks)
    q_stacks_until = jnp.where(q_hit_any, f32(now + Q_STACK_TIME), state.q_stacks_until)

    # Mortal Steel (empowered): a 450 dash and a whirlwind, both knocking up; each enemy struck once.
    q3_go = q_fire & state.q_three
    q3_until = jnp.where(q3_go, f32(now + Q3_DASH / Q3_SPEED), state.q3_until)
    q3_dash = q3_go | (m & (now < state.q3_until))
    q3_fly = (state.q3_fly | q3_go) & alive
    ox, oy = jnp.where(q3_go, kctx.x, state.q3_ox), jnp.where(q3_go, kctx.y, state.q3_oy)
    d0 = jnp.where(q3_go, 0.0, state.q3_dist)
    d1 = d0 + Q3_SPEED * kctx.dt
    wx, wy = units.x[None, :] - ox[:, None], units.y[None, :] - oy[:, None]
    w_along = wx * state.q_dx[:, None] + wy * state.q_dy[:, None]
    w_perp = jnp.abs(wy * state.q_dx[:, None] - wx * state.q_dy[:, None])
    q3_seen = jnp.where(q3_go[:, None], False, state.q3_hit)
    by_dash = q3_dash[:, None] & (dist <= Q3_DASH_RADIUS + rad)
    by_wind = q3_fly[:, None] & (w_along - rad <= jnp.minimum(d1, Q3_RANGE)[:, None]) & (w_along + rad >= d0[:, None]) \
        & (w_perp <= Q3_HALF_WIDTH + rad)
    q3_new = foe & (by_dash | by_wind) & ~q3_seen
    q3_first = jnp.argmin(jnp.where(q3_new, dist, jnp.inf), axis=1)
    q3_is_first = q3_new & (cols == q3_first[:, None]) & ~jnp.any(q3_seen, axis=1)[:, None]
    q3_id = jnp.where(q3_go, state.q_cast_id, make_cast_id(kctx, CODE_YONE_Q3))
    p_q3 = D.packets(q3_new, kctx.unit[:, None], cols, q_raw, D.PHYSICAL,
                     jnp.where(q3_is_first, Q_FLAGS_FIRST, Q_FLAGS_REST) | jnp.where(roll, D.PROP_CRIT, 0),
                     cast_id=q3_id[:, None])
    q3_up = jnp.where(q3_new, Q3_KNOCKUP, 0.0)
    q3_fly = q3_fly & (d1 < Q3_RANGE)
    on_hit = jnp.where(jnp.any(is_first, axis=1), first, jnp.where(jnp.any(q3_is_first, axis=1), q3_first, -1))
    q3_dash_out = Dash(q3_go, f32(kctx.x + state.q_dx * Q3_DASH), f32(kctx.y + state.q_dy * Q3_DASH),
                       jnp.full((c,), Q3_SPEED, jnp.float32), jnp.full((c,), -1, jnp.int32), jnp.zeros((c,), bool))

    # Spirit Cleave: a cone (apex 50 behind Yone, 650 reach, 80 degrees); centres behind Yone are never hit.
    w_due = state.w_pending & (due(kctx, state.w_fire_at) | dead)
    w_fire = w_due & alive
    ax, ay = kctx.x - state.w_dx * W_BACK, kctx.y - state.w_dy * W_BACK
    cx, cy = units.x[None, :] - ax[:, None], units.y[None, :] - ay[:, None]
    cd_ = jnp.sqrt(cx * cx + cy * cy)
    c_along = cx * state.w_dx[:, None] + cy * state.w_dy[:, None]
    c_ang = jnp.arctan2(jnp.abs(cy * state.w_dx[:, None] - cx * state.w_dy[:, None]), c_along)
    slack = jnp.arcsin(jnp.clip(rad / jnp.maximum(cd_, 1e-3), 0.0, 1.0))
    front = rx * state.w_dx[:, None] + ry * state.w_dy[:, None] >= 0.0
    w_hit = w_fire[:, None] & foe & front & (cd_ <= W_RADIUS + W_BACK + rad) \
        & (c_ang <= jnp.deg2rad(W_HALF_ANGLE) + slack)
    total = ranked(NAME, "W", "BaseDamage", r[:, 1])[:, None] \
        + ranked(NAME, "W", "MaxHealthDamage", r[:, 1])[:, None] * units.max_hp[None, :]
    total = jnp.where(minion, jnp.maximum(total, w_minion_floor(kctx.level)[:, None]), total)
    total = jnp.where(monster, jnp.minimum(total, w_monster_cap(kctx.level)[:, None]), total)
    p_w = D.concat_packets(
        D.packets(w_hit, kctx.unit[:, None], cols, 0.5 * total, D.PHYSICAL, W_FLAGS, cast_id=state.w_cast_id[:, None]),
        D.packets(w_hit, kctx.unit[:, None], cols, 0.5 * total, D.MAGIC, W_FLAGS, cast_id=state.w_cast_id[:, None]))
    champs = jnp.sum(w_hit & (units.kind == KIND_CHAMPION)[None, :], axis=1).astype(jnp.float32)
    shield_amt = (by_level(W_SHIELD, kctx.level) + W_SHIELD_AD * kctx.bonus_ad) \
        * (1.0 + jnp.where(champs > 0, W_FIRST_CHAMP + W_MORE_CHAMP * (champs - 1.0), 0.0))
    shield_on = w_fire & jnp.any(w_hit, axis=1)
    sh = jnp.where(shield_on, shield_amt, 0.0)[:, None]
    shield = ShieldGrant(f32(sh), jnp.zeros((c, 1), jnp.int32), jnp.where(shield_on, W_SHIELD_TIME, 0.0)[:, None],
                         jnp.zeros((c, 1), jnp.float32))

    # Soul Unbound: automatic return at 5 s; the return resolves after its cast time: marks detonate, dash home.
    auto = state.e_on & due(kctx, state.e_until) & (state.e_back_at > NEVER / 2) & ~kctx.stunned & ~kctx.silenced \
        & alive & ~state.q_pending & ~state.w_pending & ~state.r_pending
    e_back_at = jnp.where(auto, f32(now + E_RECAST_TIME), state.e_back_at)
    back = state.e_on & (due(kctx, e_back_at) | dead)
    e_id = make_cast_id(kctx, CODE_YONE_E)
    p_e = D.packets(back[:, None] & (state.e_mark > 0) & units.alive[None, :], kctx.unit[:, None], cols, state.e_mark,
                    D.TRUE, E_FLAGS, cast_id=e_id[:, None])
    home = Dash(back & alive, f32(state.e_body_x), f32(state.e_body_y), jnp.full((c,), E_RETURN_SPEED, jnp.float32),
                jnp.full((c,), -1, jnp.int32), jnp.zeros((c,), bool))

    # Fate Sealed: at the end of the cast time, stun everything in the area and pick the blink point.
    r_due = state.r_pending & (due(kctx, state.r_fire_at) | dead)
    r_fire = r_due & alive
    r_along = rx * state.r_dx[:, None] + ry * state.r_dy[:, None]
    r_perp = jnp.abs(ry * state.r_dx[:, None] - rx * state.r_dy[:, None])
    r_area = foe & (r_along + rad >= 0.0) & (r_along - rad <= R_LENGTH) & (r_perp <= R_HALF_WIDTH + rad)
    marked = r_fire[:, None] & r_area
    champ_marked = marked & (units.kind == KIND_CHAMPION)[None, :]
    last = jnp.argmax(jnp.where(champ_marked, r_along, -jnp.inf), axis=1)
    has_c = jnp.any(champ_marked, axis=1)
    reach = jnp.where(has_c, jnp.take_along_axis(r_along, last[:, None], axis=1)[:, 0] + R_BEHIND, R_LENGTH)
    bx, by = kctx.x + state.r_dx * reach, kctx.y + state.r_dy * reach
    stun, sid = cc_matrix(marked, R_STUN, state.r_cast_id[:, None])
    r_marked = jnp.where(r_fire[:, None], marked, state.r_marked)
    r_gust_at = jnp.where(r_fire, f32(now + R_GUST), state.r_gust_at)
    r_blink_at = jnp.where(r_fire, f32(now + R_BLINK_DELAY), state.r_blink_at)
    r_bx, r_by = jnp.where(r_fire, bx, state.r_bx), jnp.where(r_fire, by, state.r_by)
    r_ox, r_oy = jnp.where(r_fire, kctx.x, state.r_ox), jnp.where(r_fire, kctx.y, state.r_oy)
    blink = m & alive & due(kctx, state.r_blink_at)
    blink_out = Dash(blink, f32(state.r_bx), f32(state.r_by), jnp.full((c,), jnp.inf, jnp.float32),
                     jnp.full((c,), -1, jnp.int32), jnp.ones((c,), bool))
    # The gust: marked enemies still in the area take the damage and are pulled to the blink point, then knocked up.
    gust = m & due(kctx, state.r_gust_at)
    gx, gy = units.x[None, :] - state.r_ox[:, None], units.y[None, :] - state.r_oy[:, None]
    g_along = gx * state.r_dx[:, None] + gy * state.r_dy[:, None]
    g_perp = jnp.abs(gy * state.r_dx[:, None] - gx * state.r_dy[:, None])
    g_in = state.r_marked & units.alive[None, :] & (g_along + rad >= 0.0) & (g_along - rad <= R_LENGTH) \
        & (g_perp <= R_HALF_WIDTH + rad)
    g_hit = gust[:, None] & g_in
    r_raw = (ranked(NAME, "R", "BaseDamage", r[:, 3]) + R_AD * kctx.bonus_ad)[:, None] * 0.5
    p_r = D.concat_packets(
        D.packets(g_hit, kctx.unit[:, None], cols, r_raw, D.MAGIC, R_FLAGS, cast_id=state.r_cast_id[:, None]),
        D.packets(g_hit, kctx.unit[:, None], cols, r_raw, D.PHYSICAL, R_FLAGS, cast_id=state.r_cast_id[:, None]))
    pd = jnp.sqrt((units.x[None, :] - state.r_bx[:, None]) ** 2 + (units.y[None, :] - state.r_by[:, None]) ** 2)
    t_pull = pd / R_PULL_SPEED
    r_pull_until = jnp.where(g_hit, f32(now + kctx.dt + t_pull), state.r_pull_until)
    r_up_at = jnp.where(g_hit, f32(now + t_pull), state.r_up_at)
    up_now = (state.r_up_at <= now + 0.5 * kctx.dt) & units.alive[None, :]
    r_up = jnp.where(up_now, R_KNOCKUP, 0.0)
    r_up_at = jnp.where(up_now, NEVER, r_up_at)
    pulled = jnp.where(g_hit, t_pull + 1e-3, 0.0)                     # airborne through the pull

    state = state._replace(
        q_stacks=jnp.where(q_due, q_stacks, state.q_stacks).astype(jnp.int32), q_stacks_until=f32(q_stacks_until),
        q_pending=state.q_pending & ~q_due, q3_until=f32(q3_until), q3_fly=q3_fly, q3_ox=f32(ox), q3_oy=f32(oy),
        q3_dist=f32(d1), q3_hit=q3_seen | q3_new,
        w_pending=state.w_pending & ~w_due,
        e_on=state.e_on & ~back, e_back_at=jnp.where(back, NEVER, e_back_at),
        e_mark=jnp.where(back[:, None], 0.0, state.e_mark),
        r_pending=state.r_pending & ~r_due, r_marked=jnp.where(gust[:, None], False, r_marked),
        r_gust_at=jnp.where(gust, NEVER, r_gust_at), r_blink_at=jnp.where(blink, NEVER, r_blink_at),
        r_bx=f32(r_bx), r_by=f32(r_by), r_ox=f32(r_ox), r_oy=f32(r_oy), r_pull_until=f32(r_pull_until),
        r_up_at=f32(r_up_at),
        e_bar=f32(jnp.where(state.e_on & ~back, jnp.clip((state.e_until - now - kctx.dt) / E_TIME, 0.0, 1.0), 0.0)))
    cc = CCOut(stun, z, z, f32(q3_up + r_up + pulled), z, z,
               jnp.where(stun > 0, sid, jnp.where(q3_new, q3_id[:, None], 0)).astype(jnp.int32))
    dash = Dash(*(jnp.where(blink_out.active, b, jnp.where(home.active, h, q))
                  for q, h, b in zip(q3_dash_out, home, blink_out)))
    cd = jnp.zeros((c, 4), bool).at[:, 2].set(back & m)
    return state, out(c, n, packets=D.concat_packets(p_q, p_q3, p_w, p_e, p_r), cc=cc, dash=dash, shield=shield,
                      cooldown_start=cd, base_cooldown=_base_cd(kctx), cast_lockout=f32(jnp.where(r_fire, R_LOCK, 0.0)),
                      on_hit_target=jnp.where(m, on_hit, -1).astype(jnp.int32), on_attack_proc=m & (on_hit >= 0),
                      on_hit_raw=jnp.take_along_axis(q_raw, jnp.maximum(on_hit, 0)[:, None], axis=1)[:, 0],
                      on_hit_crit=jnp.take_along_axis(roll, jnp.maximum(on_hit, 0)[:, None], axis=1)[:, 0],
                      on_hit_cast_id=jnp.where(jnp.any(is_first, axis=1), state.q_cast_id, q3_id))


def on_attack(state: State, kctx, units, launch) -> tuple[State, KitOut]:
    """Steel and Spirit: each launched attack switches sword."""
    went = holder_rows(launch.launched, kctx) & _mine(kctx)
    return state._replace(azakana=state.azakana ^ went), out(kctx.unit.shape[0], units.x.shape[0],
                                                              base_cooldown=_base_cd(kctx))


def on_hit(state: State, kctx, units, launch, dodging) -> tuple[State, KitOut]:
    return state, out(kctx.unit.shape[0], units.x.shape[0], base_cooldown=_base_cd(kctx))


def on_damage(state: State, kctx, units, report) -> tuple[State, KitOut]:
    """Soul Unbound marks: a share of the health damage Yone's attacks and abilities deal to enemy champions."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    p = report.packets
    res = report.resolved
    lost = getattr(res, "health_loss", None)
    lost = p.raw if lost is None else lost
    lost = jnp.where(p.valid, lost, 0.0)
    dst = jnp.clip(p.dst, 0, n - 1)
    kind = (p.flags & (D.TAG_BASIC_ATTACK | D.TAG_ACTIVE_SPELL)) != 0
    own = (p.src[None, :] == kctx.unit[:, None]) & _mine(kctx)[:, None] & state.e_on[:, None] & kind[None, :] \
        & ((p.flags & D.TAG_ITEM) == 0)[None, :] & (p.dtype != D.TRUE)[None, :] \
        & (units.kind[dst] == KIND_CHAMPION)[None, :] & (units.team[dst][None, :] != kctx.team[:, None])
    share = ranked(NAME, "E", "DeathmarkPercent", kctx.ranks[:, 2])
    add = jnp.sum(jnp.where(own[:, :, None] & (dst[None, :, None] == jnp.arange(n)[None, None, :]),
                            lost[None, :, None], 0.0), axis=1) * share[:, None]
    return state._replace(e_mark=state.e_mark + add), out(c, n)


def on_takedown(state: State, kctx, units, kills) -> State:
    return state


def stats(state: State, kctx) -> ItemStats:
    """Intent: crit chance doubled, 0.5 AD per 1% above 100%, crits at 95%; Spirit Form's ramping move speed."""
    m = _mine(kctx)
    cc = kctx.crit_chance
    excess = jnp.maximum((1.0 + P_CRIT_MULT) * cc - 1.0, 0.0) * 100.0
    ramp = jnp.clip((kctx.now - state.e_from) / E_MS_RAMP, 0.0, 1.0)
    ms = jnp.where(state.e_on, E_MS_START + (E_MS_END - E_MS_START) * ramp, 0.0)
    return ItemStats(crit_chance=f32(jnp.where(m, P_CRIT_MULT * cc, 0.0)),
                     attack_damage=f32(jnp.where(m, P_CRIT_TO_AD * excess, 0.0)),
                     crit_damage=f32(jnp.where(m, (P_CRIT_DAMAGE - 1.0) * kctx.crit_damage, 0.0)),
                     percent_move_speed=f32(jnp.where(m, ms, 0.0)))


def defense(state: State, kctx) -> KitDefense:
    c = kctx.unit.shape[0]
    o = jnp.ones((c,), jnp.float32)
    return KitDefense(o, jnp.zeros((c,), bool), o, jnp.zeros((c,), jnp.float32))


def attack_mods(state: State, kctx) -> KitAttackMods:
    m = _mine(kctx)
    c = kctx.unit.shape[0]
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    return KitAttackMods(z, f, m & (state.q_pending | state.w_pending | state.r_pending), f, z, z, f,
                         magic_split=f32(jnp.where(m & state.azakana, P_SPLIT, 0.0)))


def ghosted(state: State, kctx) -> Any:
    return _mine(kctx) & state.e_on


def displacement(state: State, kctx, n_units: int) -> tuple[Any, Any, Any, Any]:
    """(N,) Fate Sealed's pull to the blink point at 3000 units/s."""
    on = _mine(kctx)[:, None] & (kctx.now < state.r_pull_until)
    k = jnp.argmax(on, axis=0)
    pick = lambda a: jnp.take_along_axis(jnp.broadcast_to(a, on.shape), k[None, :], axis=0)[0]   # noqa: E731
    return jnp.any(on, axis=0), pick(state.r_bx[:, None]), pick(state.r_by[:, None]), \
        jnp.full((n_units,), R_PULL_SPEED, jnp.float32)


def resource(state: State, champion_id: Any) -> Any:
    """(C,) Flow shows Soul Unbound's remaining time; NaN where the holder is another kit."""
    return jnp.where(champion_id == ID, state.e_bar, jnp.nan)


def debuffs(state: State, kctx, units) -> Debuffs:
    return neutral_debuffs(units.x.shape[0])


__all__ = ["State", "init", "cast", "periodic", "on_attack", "on_hit", "on_damage", "on_takedown", "stats",
           "defense", "attack_mods", "ghosted", "displacement", "resource", "debuffs", "as_scaled",
           "w_minion_floor", "w_monster_cap", "by_level"]
