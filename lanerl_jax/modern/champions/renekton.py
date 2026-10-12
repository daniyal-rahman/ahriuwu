"""Renekton (58), 26.19. Rules REN.* and their evidence: docs/modern/CHAMPIONS.md; numbers from the pinned client
JSON (``modern/data/26.19/champions``). Fury is kit state (the world's mana stays 0)."""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..core import damage as D
from ..core.types import KIND_CHAMPION, KIND_MONSTER, CCOut, Dash
from ..items.catalog import ItemStats
from ..items.effects.core import Debuffs, neutral_debuffs
from ..runes.catalog import ChampionTraits
from .core import (NEVER, KitAttackMods, KitDefense, KitOut, cc_matrix, center_dist, cooldown_row, due, enemies,
                   f32, gather, holder_rows, is_structure, later, make_cast_id, out, pulses, ranked, scalar)

NAME = "Renekton"
ID = 58
SKILL_ORDER = (0, 2, 1, 0, 0, 3, 0, 2, 0, 2, 3, 2, 2, 1, 1, 3, 1, 1)    # default ranks per level: Q > E > W
TRAITS = ChampionTraits(has_immobilize=True, resource="none")          # rune legality: W stuns; fury, no mana
UNIT_TARGET_RANGE = (0.0, 0.0, 0.0, 0.0)
FURY_MAX = 100.0                                      # arIncrements 50 x arMaxSegments 2
EMPOWER = 50.0
P_FURY = scalar(NAME, "Passive", "FuryPerAttack")              # 5 per basic attack on-hit
P_LOW_HP = scalar(NAME, "Passive", "LowHealthPercentThreshold")  # 0.5
P_LOW_BONUS = scalar(NAME, "Passive", "FuryIncreasePercent")   # +50% below half health
P_COMBAT = scalar(NAME, "Passive", "InCombatDuration")         # 12 s
P_DECAY_PERIOD = 0.25                                 # wiki: -1 Fury every 0.25 s out of combat
Q_RADIUS = scalar(NAME, "Q", "Range")                 # 400 (center to edge)
Q_R_MULT = scalar(NAME, "Q", "UltimateRangeMod")      # x1.2 during Dominus
Q_LOCK = 0.25                                         # wiki: no attacks, E or R for 0.25 s
Q_FURY_MINION = scalar(NAME, "Q", "MinionFuryGain")   # 2.5
Q_FURY_CHAMP = scalar(NAME, "Q", "ChampionFuryGain")  # 10
Q_FURY_CAP = scalar(NAME, "Q", "FuryGainCap")         # 30
Q_HEAL_MULT = scalar(NAME, "Q", "EmpoweredHealMultiplier")    # 3
W_DURATION = scalar(NAME, "W", "BuffDuration")        # 6
W_RANGE_BONUS = 50.0                                  # wiki: the empowered attack has 175 range (base 125)
W_AD = scalar(NAME, "W", "TotalADPerHit")             # 0.75 AD per strike
W_STUN = scalar(NAME, "W", "StunDuration")            # 0.75
W_STUN_EMP = scalar(NAME, "W", "EnragedStunDuration")  # 1.5
W_FURY_CHAMP = scalar(NAME, "W", "BonusFuryVsChamps")  # 10
W_LOCK = 0.525                                        # wiki: no movement, Q or E after the strikes
E_DISTANCE = 450.0                                    # castRangeDisplayOverride: fixed dash length
E_SPEED = scalar(NAME, "E", "BaseDashSpeed")          # 750 + move speed
E_WIDTH = 50.0                                        # mLineWidth: hit distance from the dash path
E_WINDOW = scalar(NAME, "E", "DiceTimer")             # 4
E_FURY_MINION = scalar(NAME, "E", "MinionRageGeneration")    # 2
E_FURY_CHAMP = scalar(NAME, "E", "ChampionRageGeneration")   # 10
E_FURY_CAP = scalar(NAME, "E", "FuryMax")             # 30
E_SHRED_DURATION = scalar(NAME, "E", "ShredTimer")    # 4
R_CAST_TIME = 0.25
R_DURATION = scalar(NAME, "R", "BuffDuration")        # 15
R_TICK = scalar(NAME, "R", "TickRate")                # 0.5
R_RADIUS = 375.0                                      # wiki aura radius
R_RATIO = 0.1                                         # TotalDamagePerSecond: +10% AP and +10% bonus AD per second
R_FURY_CAST = scalar(NAME, "R", "FuryOnCast")         # 20
R_FURY_RATE = scalar(NAME, "R", "FuryPerSecond")      # 5
R_FURY_LIMIT = 75.0                                   # wiki: Dominus generates Fury up to 75
R_RANGE_BONUS = 25.0

Q_FLAGS = D.TAG_AOE | D.TAG_ACTIVE_SPELL
W_FLAGS = D.TAG_ACTIVE_SPELL | D.PROP_LIFESTEAL       # basic damage, first strike also spell damage (wiki)
E_FLAGS = D.TAG_AOE | D.TAG_ACTIVE_SPELL
R_FLAGS = D.TAG_AOE | D.TAG_ACTIVE_SPELL | D.TAG_PERIODIC | D.PROP_ULTIMATE


class State(NamedTuple):
    fury: Any               # (C,) 0..100
    combat_at: Any          # last attack, damage dealt or taken (Fury decay clock)
    q_lock_until: Any
    w_on: Any               # empowered attack waiting
    w_until: Any
    w_emp: Any              # the empowered (3-strike) version
    w_cast_id: Any
    e_dash: Any             # dashing (Slice or Dice)
    e_until: Any
    e_px: Any               # position at the previous tick (hit path)
    e_py: Any
    e_hit: Any              # (C, N) bool units already hit by this dash
    e_fury: Any             # Fury gained by this dash (cap 30)
    e_emp: Any              # this dash is an empowered Dice
    e_cast_id: Any
    e_dice_until: Any       # Dice may be cast before this (an enemy was hit by Slice)
    e_is_dice: Any
    shred_until: Any        # (C, N) empowered Dice armor reduction end
    e_shred: Any
    r_pending: Any
    r_fire_at: Any
    r_on: Any
    r_start: Any
    r_until: Any
    r_ticks: Any
    r_cast_id: Any
    r_rank: Any


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    i = jnp.zeros((c,), jnp.int32)
    return State(z, z - NEVER, z - NEVER, f, z, f, i, f, z, z, z, jnp.zeros((c, n), bool), z, f, i, z - NEVER, f,
                 jnp.zeros((c, n), jnp.float32), z, f, z + NEVER, f, z, z, i, i, i)


def _mine(kctx) -> Any:
    return kctx.champion_id == ID


def _base_cd(kctx) -> Any:
    return jnp.where(_mine(kctx)[:, None], cooldown_row(NAME, kctx.ranks), 0.0)


def gain(kctx, amount: Any) -> Any:
    """Fury generated from ``amount`` (Reign of Anger: +50% below half health)."""
    low = kctx.hp < P_LOW_HP * kctx.max_hp
    return jnp.asarray(amount, jnp.float32) * jnp.where(low, 1.0 + P_LOW_BONUS, 1.0)


def _add(fury: Any, delta: Any) -> Any:
    return jnp.clip(fury + delta, 0.0, FURY_MAX)


def q_radius(r_on: Any) -> Any:
    return jnp.where(r_on, Q_RADIUS * Q_R_MULT, Q_RADIUS)


def _per_hit_fury(hit: Any, champ: Any, minion_gain: float, champ_gain: float, cap: Any) -> Any:
    raw = jnp.sum(jnp.where(hit, jnp.where(champ[None, :], champ_gain, minion_gain), 0.0), axis=1)
    return jnp.minimum(raw, cap)


def cast(state: State, kctx, units, order) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    r = kctx.ranks
    now = f32(kctx.now)
    cols = jnp.arange(n)[None, :]
    free = m & kctx.alive & ~kctx.stunned & ~kctx.silenced & ~state.r_pending
    ready = (r > 0) & (kctx.cooldowns <= 0)
    want = lambda s: (order.slot == s) & free          # noqa: E731
    q_locked = now < state.q_lock_until - 1e-6
    rooted = jnp.zeros_like(m) if kctx.rooted is None else kctx.rooted
    champ_col = units.kind == KIND_CHAMPION
    emp = state.fury >= EMPOWER - 1e-6

    # Cull the Meek: instant cleave; heals per enemy hit up to a cap; empowered at 50 Fury.
    q = want(0) & ready[:, 0]
    q_emp = q & emp
    hit = q[:, None] & enemies(kctx, units) & (center_dist(kctx, units)
                                               <= q_radius(state.r_on)[:, None] + units.radius[None, :])
    q_raw = jnp.where(q_emp, ranked(NAME, "Q", "EmpoweredBaseDamage", r[:, 0])
                      + ranked(NAME, "Q", "EmpoweredADRatio", r[:, 0]) * kctx.bonus_ad,
                      ranked(NAME, "Q", "BaseDamage", r[:, 0]) + ranked(NAME, "Q", "BaseADRatio", r[:, 0]) * kctx.bonus_ad)
    q_id = make_cast_id(kctx, 0)
    p_q = D.packets(hit, kctx.unit[:, None], cols, q_raw[:, None], D.PHYSICAL, Q_FLAGS, cast_id=q_id[:, None])
    per_minion = ranked(NAME, "Q", "BaseMinionHeal", r[:, 0]) + ranked(NAME, "Q", "MinionHealADRatio", r[:, 0]) \
        * kctx.bonus_ad
    per_champ = ranked(NAME, "Q", "ChampHeal", r[:, 0]) + ranked(NAME, "Q", "ChampionHealADRatio", r[:, 0]) \
        * kctx.bonus_ad
    heal = jnp.sum(jnp.where(hit, jnp.where(champ_col[None, :], per_champ[:, None], per_minion[:, None]), 0.0), axis=1)
    heal = jnp.minimum(heal * jnp.where(q_emp, Q_HEAL_MULT, 1.0),
                       jnp.where(q_emp, ranked(NAME, "Q", "EmpoweredHealCap", r[:, 0]),
                                 ranked(NAME, "Q", "BasicHealCap", r[:, 0])))
    q_fury = jnp.where(q & ~q_emp, gain(kctx, _per_hit_fury(hit, champ_col, Q_FURY_MINION, Q_FURY_CHAMP, Q_FURY_CAP)),
                       0.0)

    # Ruthless Predator: attack reset; the next attack within 6 s strikes 2 (3) times.
    w = want(1) & ready[:, 1] & ~state.w_on
    w_emp = w & emp

    # Slice and Dice: fixed-length dash toward the cursor; Dice while the window from a Slice hit is open.
    dice_ok = m & (now < state.e_dice_until)
    e_any = want(2) & ~q_locked & ~rooted & ~state.e_dash
    slice_ = e_any & ready[:, 2] & ~dice_ok
    dice = e_any & dice_ok
    e = slice_ | dice
    e_emp = dice & emp
    dx, dy = order.x - kctx.x, order.y - kctx.y
    d = jnp.sqrt(dx * dx + dy * dy)
    ux, uy = jnp.where(d > 1e-3, dx / jnp.maximum(d, 1e-3), 1.0), jnp.where(d > 1e-3, dy / jnp.maximum(d, 1e-3), 0.0)
    ms = jnp.zeros((c,), jnp.float32) if kctx.move_speed is None else kctx.move_speed
    speed = E_SPEED + ms
    dash = Dash(e, f32(kctx.x + ux * E_DISTANCE), f32(kctx.y + uy * E_DISTANCE), f32(speed),
                jnp.full((c,), -1, jnp.int32), jnp.zeros((c,), bool))

    # Dominus: 0.25 s cast time.
    rr = want(3) & ready[:, 3] & ~q_locked

    started = q | w | e | rr
    slot = jnp.where(q, 0, jnp.where(w, 1, jnp.where(e, 2, jnp.where(rr, 3, -1)))).astype(jnp.int32)
    cid = jnp.where(started, make_cast_id(kctx, jnp.maximum(slot, 0)), 0).astype(jnp.int32)
    spent = jnp.where(q_emp | w_emp | e_emp, EMPOWER, 0.0)
    state = state._replace(
        fury=_add(state.fury, q_fury - spent),
        combat_at=jnp.where(q & jnp.any(hit, axis=1), now, state.combat_at),
        q_lock_until=jnp.where(q, later(kctx, Q_LOCK), state.q_lock_until),
        w_on=state.w_on | w, w_until=jnp.where(w, later(kctx, W_DURATION), state.w_until),
        w_emp=jnp.where(w, w_emp, state.w_emp), w_cast_id=jnp.where(w, cid, state.w_cast_id),
        e_dash=state.e_dash | e, e_until=jnp.where(e, f32(now + E_DISTANCE / speed), state.e_until),
        e_px=jnp.where(e, kctx.x, state.e_px), e_py=jnp.where(e, kctx.y, state.e_py),
        e_hit=jnp.where(e[:, None], False, state.e_hit), e_fury=jnp.where(e, 0.0, state.e_fury),
        e_emp=jnp.where(e, e_emp, state.e_emp), e_cast_id=jnp.where(e, cid, state.e_cast_id),
        e_is_dice=jnp.where(e, dice, state.e_is_dice),
        e_dice_until=jnp.where(dice, -NEVER, state.e_dice_until),
        r_pending=state.r_pending | rr, r_fire_at=jnp.where(rr, later(kctx, R_CAST_TIME), state.r_fire_at),
        r_cast_id=jnp.where(rr, cid, state.r_cast_id), r_rank=jnp.where(rr, r[:, 3], state.r_rank))
    cd = jnp.stack([q, jnp.zeros_like(q), slice_, rr], -1)
    return state, out(c, n, packets=p_q, dash=dash, heal=f32(heal), cooldown_start=cd, base_cooldown=_base_cd(kctx),
                      attack_reset=w, cast_started=started, cast_slot=slot, cast_id=cid,
                      cast_lockout=f32(jnp.where(rr, R_CAST_TIME, 0.0)))


def periodic(state: State, kctx, units) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    alive = kctx.alive
    dead = ~alive
    r = kctx.ranks
    now = f32(kctx.now)
    cols = jnp.arange(n)[None, :]
    champ_col = units.kind == KIND_CHAMPION

    # Ruthless Predator expires unused: cooldown (post-effect).
    w_end = state.w_on & (due(kctx, state.w_until) | dead)

    # Slice and Dice: enemies within E_WIDTH of the stretch dashed since the last tick take the hit once.
    px, py = state.e_px, state.e_py
    sx, sy = kctx.x - px, kctx.y - py
    seg2 = jnp.maximum(sx * sx + sy * sy, 1e-6)
    tt = jnp.clip(((units.x[None, :] - px[:, None]) * sx[:, None] + (units.y[None, :] - py[:, None]) * sy[:, None])
                  / seg2[:, None], 0.0, 1.0)
    qx, qy = px[:, None] + tt * sx[:, None], py[:, None] + tt * sy[:, None]
    near = jnp.sqrt((units.x[None, :] - qx) ** 2 + (units.y[None, :] - qy) ** 2) <= E_WIDTH + units.radius[None, :]
    e_new = (state.e_dash & alive)[:, None] & enemies(kctx, units) & near & ~state.e_hit
    e_base = ranked(NAME, "E", "BaseDamage", r[:, 2]) + ranked(NAME, "E", "BaseADRatio", r[:, 2]) * kctx.bonus_ad
    e_bonus = ranked(NAME, "E", "EnragedBaseDamage", r[:, 2]) + ranked(NAME, "E", "EnragedADRatio", r[:, 2]) \
        * kctx.bonus_ad
    e_raw = e_base + jnp.where(state.e_emp, e_bonus, 0.0)
    p_e = D.packets(e_new, kctx.unit[:, None], cols, e_raw[:, None], D.PHYSICAL, E_FLAGS,
                    cast_id=state.e_cast_id[:, None])
    raw_fury = jnp.sum(jnp.where(e_new, jnp.where(champ_col[None, :], E_FURY_CHAMP, E_FURY_MINION), 0.0), axis=1)
    e_add = jnp.where(state.e_emp, 0.0, jnp.minimum(raw_fury, jnp.maximum(E_FURY_CAP - state.e_fury, 0.0)))
    shred = e_new & state.e_emp[:, None]
    shred_until = jnp.where(shred, f32(now + kctx.dt + E_SHRED_DURATION), state.shred_until)
    e_shred = jnp.where(jnp.any(shred, axis=1), ranked(NAME, "E", "EnragedArmorShred", r[:, 2]) / 100.0,
                        state.e_shred)
    e_hit = state.e_hit | e_new
    # The world moves the dash after this hook: the stretch ending at e_until is checked on the tick after.
    e_done = state.e_dash & ((now >= state.e_until - 1e-6) | dead)
    landed_slice = e_done & ~state.e_is_dice & jnp.any(e_hit, axis=1) & alive
    dice_until = jnp.where(landed_slice, f32(now + E_WINDOW), state.e_dice_until)

    # Dominus: buff after the cast time; aura every 0.5 s; +5 Fury per second up to 75.
    r_due = state.r_pending & (due(kctx, state.r_fire_at) | dead)
    r_fire = r_due & alive
    r_on = (state.r_on & ~(due(kctx, state.r_until) | dead)) | r_fire
    r_start = jnp.where(r_fire, now, state.r_start)
    r_ticks = jnp.where(r_fire, 0, state.r_ticks)
    storm = state.r_on & alive & ~r_fire & (r_ticks < round(R_DURATION / R_TICK)) \
        & due(kctx, r_start + (r_ticks + 1).astype(jnp.float32) * R_TICK)
    r_hit = storm[:, None] & enemies(kctx, units) & (center_dist(kctx, units) <= R_RADIUS + units.radius[None, :])
    r_raw = (ranked(NAME, "R", "AuraDamagePerSecond", state.r_rank) + R_RATIO * (kctx.ap + kctx.bonus_ad)) * R_TICK
    p_r = D.packets(r_hit, kctx.unit[:, None], cols, r_raw[:, None], D.MAGIC, R_FLAGS, cast_id=state.r_cast_id[:, None])
    r_rate = jnp.where(state.r_on & alive & (state.fury < R_FURY_LIMIT),
                       jnp.minimum(gain(kctx, R_FURY_RATE * kctx.dt), jnp.maximum(R_FURY_LIMIT - state.fury, 0.0)),
                       0.0)

    # Reign of Anger: after 12 s out of combat (Dominus keeps the clock running), -1 Fury every 0.25 s.
    combat_at = jnp.where(jnp.any(e_new, axis=1) | r_on, now, state.combat_at)
    idle = now - combat_at >= P_COMBAT - 1e-6
    decay = jnp.where(idle & alive, pulses(kctx, P_DECAY_PERIOD), 0.0)
    fury = _add(state.fury, gain(kctx, e_add) + jnp.where(r_fire, gain(kctx, R_FURY_CAST), 0.0) + r_rate - decay)
    fury = jnp.where(dead, 0.0, fury)

    state = state._replace(
        fury=f32(fury), combat_at=f32(combat_at),
        w_on=state.w_on & ~w_end,
        e_dash=state.e_dash & ~e_done, e_px=kctx.x, e_py=kctx.y, e_hit=e_hit, e_fury=f32(state.e_fury + e_add),
        e_dice_until=f32(jnp.where(dead, -NEVER, dice_until)), shred_until=f32(shred_until), e_shred=f32(e_shred),
        r_pending=state.r_pending & ~r_due, r_on=r_on, r_start=f32(r_start), r_until=f32(
            jnp.where(r_fire, now + R_DURATION, state.r_until)),
        r_ticks=(r_ticks + storm.astype(jnp.int32)).astype(jnp.int32))
    cd = jnp.zeros((c, 4), bool).at[:, 1].set(w_end & m)
    return state, out(c, n, packets=D.concat_packets(p_e, p_r), cooldown_start=cd, base_cooldown=_base_cd(kctx))


def on_attack(state: State, kctx, units, launch) -> tuple[State, KitOut]:
    go = holder_rows(launch.launched, kctx) & _mine(kctx)
    return state._replace(combat_at=jnp.where(go, f32(kctx.now), state.combat_at)), \
        out(kctx.unit.shape[0], units.x.shape[0], base_cooldown=_base_cd(kctx))


def on_hit(state: State, kctx, units, launch, dodging) -> tuple[State, KitOut]:
    """Fury on basic attacks and the Ruthless Predator strikes. ``dodging`` (N,) bool."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    target = holder_rows(launch.target, kctx)
    t = jnp.clip(target, 0, n - 1)
    hit = holder_rows(launch.launched, kctx) & m & kctx.alive & (target >= 0)
    land = hit & ~gather(dodging, target)
    struct = is_structure(units.kind[t])
    w = hit & state.w_on & ~struct                       # never triggers on structures (stays armed)
    w_land = w & land
    strikes = jnp.where(state.w_emp, 3.0, 2.0)
    # The attack's own AD hit stands for the first strike (it crits for the bonus crit damage).
    w_raw = strikes * (ranked(NAME, "W", "BaseDamagePerHit", kctx.ranks[:, 1]) + W_AD * kctx.total_ad) \
        - kctx.total_ad
    p_w = D.packets(w_land, kctx.unit, t, jnp.maximum(w_raw, 0.0), D.PHYSICAL, W_FLAGS, cast_id=state.w_cast_id)
    mask = w_land[:, None] & (jnp.arange(n)[None, :] == t[:, None])
    stun, sid = cc_matrix(mask, jnp.where(state.w_emp, W_STUN_EMP, W_STUN)[:, None], state.w_cast_id[:, None])
    z = jnp.zeros((c, n), jnp.float32)
    strip = mask & state.w_emp[:, None] & (units.kind[t] != KIND_MONSTER)[:, None]
    # Fury: 5 per landed attack on a non-structure, per strike for Ruthless Predator, +10 vs champions; none from
    # the empowered version.
    per = jnp.where(w_land, jnp.where(state.w_emp, 0.0, 2.0 * P_FURY
                                      + jnp.where(units.kind[t] == KIND_CHAMPION, W_FURY_CHAMP, 0.0)), P_FURY)
    fury = _add(state.fury, jnp.where(land & ~struct, gain(kctx, per), 0.0))
    state = state._replace(fury=f32(fury), w_on=state.w_on & ~w,
                           combat_at=jnp.where(land, f32(kctx.now), state.combat_at))
    cd = jnp.zeros((c, 4), bool).at[:, 1].set(w)
    return state, out(c, n, packets=p_w, cc=CCOut(stun, z, z, z, z, z, sid), cooldown_start=cd,
                      base_cooldown=_base_cd(kctx), cast_lockout=f32(jnp.where(w_land, W_LOCK, 0.0)),
                      strip_shields=strip)


def on_damage(state: State, kctx, units, report) -> tuple[State, KitOut]:
    """Combat clock: damage Renekton dealt to or took from an enemy unit."""
    p = report.packets
    src_team, dst_team = gather(units.team, p.src), gather(units.team, p.dst)
    live = p.valid & (p.raw > 0) & (src_team != dst_team)
    involved = jnp.any(live[None, :] & ((p.src[None, :] == kctx.unit[:, None]) | (p.dst[None, :] == kctx.unit[:, None])),
                       axis=1)
    state = state._replace(combat_at=jnp.where(_mine(kctx) & involved, f32(kctx.now), state.combat_at))
    return state, out(kctx.unit.shape[0], units.x.shape[0])


def on_takedown(state: State, kctx, units, kills) -> State:
    return state


def stats(state: State, kctx) -> ItemStats:
    on = _mine(kctx) & state.r_on
    return ItemStats(health=f32(jnp.where(on, ranked(NAME, "R", "HealthGain", state.r_rank), 0.0)))


def defense(state: State, kctx) -> KitDefense:
    c = kctx.unit.shape[0]
    o = jnp.ones((c,), jnp.float32)
    return KitDefense(o, jnp.zeros((c,), bool), o, jnp.zeros((c,), jnp.float32))


def attack_mods(state: State, kctx) -> KitAttackMods:
    m = _mine(kctx)
    c = kctx.unit.shape[0]
    z = jnp.zeros((c,), jnp.float32)
    rng = jnp.where(m & state.w_on, W_RANGE_BONUS, 0.0) + jnp.where(m & state.r_on, R_RANGE_BONUS, 0.0)
    locked = m & (kctx.now < state.q_lock_until - 1e-6)
    return KitAttackMods(f32(rng), jnp.zeros((c,), bool), locked, jnp.zeros((c,), bool), z, z, jnp.zeros((c,), bool))


def debuffs(state: State, kctx, units) -> Debuffs:
    """Empowered Dice armor reduction, (N,)."""
    n = units.x.shape[0]
    on = _mine(kctx)[:, None] & (kctx.now < state.shred_until)
    return neutral_debuffs(n)._replace(
        percent_armor_reduction=f32(jnp.max(jnp.where(on, state.e_shred[:, None], 0.0), axis=0)))


def resource(state: State, champion_id: Any) -> Any:
    """(C,) Fury as a fraction of its maximum where the holder is Renekton, else NaN (the observation's
    resource bar)."""
    return jnp.where(champion_id == ID, state.fury / FURY_MAX, jnp.nan)


__all__ = ["State", "init", "cast", "periodic", "on_attack", "on_hit", "on_damage", "on_takedown", "stats",
           "defense", "attack_mods", "debuffs", "resource", "gain", "q_radius"]
