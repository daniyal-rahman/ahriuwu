"""Darius (122), 26.19. Rules DAR.* and their evidence: docs/modern/CHAMPIONS.md; numbers from the pinned client
JSON (``modern/data/26.19/champions``). Apprehend's pull is a kit-owned displacement the world moves
(``displacement``)."""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..core import damage as D
from ..core.types import KIND_CHAMPION, KIND_MONSTER, CCOut, Dash
from ..data.champions import spell
from ..items.catalog import ItemStats
from ..items.effects.core import Debuffs, neutral_debuffs
from ..runes.catalog import ChampionTraits
from .core import (NEVER, KitAttackMods, KitDefense, KitOut, cc_matrix, center_dist, cooldown_row, due, enemies,
                   f32, gather, holder_rows, is_structure, later, make_cast_id, mana_row, out, ranked, scalar,
                   target_dist)
from .nasus import _large

NAME = "Darius"
ID = 122
SKILL_ORDER = (0, 2, 1, 0, 0, 3, 0, 1, 0, 1, 3, 1, 1, 2, 2, 3, 2, 2)    # default ranks per level: Q > W > E
TRAITS = ChampionTraits(has_immobilize=True, resource="mana")          # rune legality: E pulls (airborne)
_CALC = spell(NAME, "Passive")["calculations"]
_BLEED = _CALC["BleedDamagePerStack"]["mFormulaParts"]
P_BLEED_LV1, P_BLEED_LV18 = _BLEED[0]["mStartValue"], _BLEED[0]["mEndValue"]   # 13 -> 30 by level
P_BLEED_AD = _BLEED[1]["mCoefficient"]                                         # 0.3 bonus AD
_MIGHT = _CALC["NoxianMightBonusAD"]["mFormulaParts"][0]
P_MAX = int(scalar(NAME, "Passive", "MaxStacks"))     # 5
P_DURATION = scalar(NAME, "Passive", "BleedDuration")  # 5
P_TICK = 1.25                                         # wiki: every 1.25 s, 4 ticks per application
P_TICKS = 4
P_MONSTER = scalar(NAME, "Passive", "MonsterMod")     # 2.5 (26.4)
P_MIGHT_DURATION = 5.0
Q_DELAY = 0.75                                        # wiki: hefts the axe for 0.75 s
Q_GHOST = 1.0
Q_OUTER = 460.0                                       # wiki radii (center to edge / center)
Q_INNER = 240.0
Q_HANDLE = scalar(NAME, "Q", "InnerCircleDamagePercent") / 100.0      # 0.35
Q_HEAL = scalar(NAME, "Q", "MissingHealthHeal") / 100.0               # 17% missing health per target
Q_HEAL_TARGETS = round(scalar(NAME, "Q", "MissingHealPercent") / scalar(NAME, "Q", "MissingHealthHeal"))  # 3
W_WINDOW = 4.0                                        # wiki
W_RANGE_BONUS = 25.0
W_SLOW = scalar(NAME, "W", "SlowPercent") / 100.0     # 0.9
W_SLOW_DURATION = scalar(NAME, "W", "SlowDuration")   # 1
W_REFUND = scalar(NAME, "W", "PercentCDRefund") / 100.0               # 0.5
E_CAST_TIME = 0.25
E_RANGE = 535.0                                       # castRangeDisplayOverride / wiki
E_HALF_ANGLE = 25.0                                   # client castConeAngle (wiki: a 50 degree cone)
E_LOCK = 0.4                                          # wiki: no movement, Q or R after the cast time
# Pull and rebound, measured on 14 pulls of Garen in the 16.9 replay corpus (20 fps positions; DAR.E2): the target
# is dragged at 1784 units/s to about 45 from Darius's centre, then rebounds over about 0.5 s to about 230 from it.
E_PULL_SPEED = 1784.0
E_PULL_STOP = 45.0
E_REBOUND = 230.0
E_REBOUND_TIME = 0.5
E_SLOW = scalar(NAME, "E", "SlowPercent") / 100.0     # 0.4, from the rebound
E_SLOW_DURATION = 1.0
R_RANGE = 475.0
R_CAST_TIME = 0.3667
R_AD = 0.75
R_PER_STACK = scalar(NAME, "R", "RDamagePercentPerHemoStack")         # 0.2
R_KILL_WINDOW = 0.15
R_RECAST = scalar(NAME, "R", "RRecastDuration")       # 20
R_LOCK = 0.25                                         # wiki: no Q or E after the cast time
UNIT_TARGET_RANGE = (0.0, 0.0, 0.0, R_RANGE)

Q_FLAGS = D.TAG_AOE | D.TAG_ACTIVE_SPELL
W_FLAGS = D.TAG_ACTIVE_SPELL | D.PROP_LIFESTEAL       # basic and spell damage (wiki)
BLEED_FLAGS = D.TAG_PERIODIC | D.TAG_PROC
R_FLAGS = D.TAG_ACTIVE_SPELL | D.PROP_ULTIMATE


class State(NamedTuple):
    stacks: Any             # (C, N) int32 Hemorrhage stacks on unit n
    bleed_until: Any        # (C, N)
    tick_at: Any            # (C, N) next bleed tick
    might_until: Any        # (C,) Noxian Might
    q_pending: Any
    q_fire_at: Any
    q_ghost_until: Any
    q_cast_id: Any
    w_on: Any
    w_until: Any
    w_cast_id: Any
    w_hit: Any              # (C,) int32 unit the empowered attack landed on this tick (-1 none)
    w_hit_at: Any
    w_refund_at: Any        # tick of a kill refund (W restarts at half its cooldown, mana back)
    e_pending: Any
    e_fire_at: Any
    e_dx: Any               # sweep direction
    e_dy: Any
    e_cast_id: Any
    e_lock_until: Any       # Q and R blocked
    disp_until: Any         # (C, N) rebound end
    pull_until: Any         # (C, N) pull end (then the rebound)
    disp_ux: Any            # (C, N) unit vector from Darius toward the target at the sweep
    disp_uy: Any
    anchor_x: Any           # (C,) Darius's position at the sweep (he cannot move during the pull)
    anchor_y: Any
    slow_at: Any            # (C, N) Apprehend slow starts (rebound end)
    r_pending: Any
    r_fire_at: Any
    r_target: Any
    r_seq: Any
    r_cast_id: Any
    r_hit_at: Any
    r_lock_until: Any
    r_recast_until: Any     # free recast window after an execution
    r_reset_at: Any         # tick of a rank-3 execution (cooldown reset)


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    i = jnp.zeros((c,), jnp.int32)
    zn = jnp.zeros((c, n), jnp.float32)
    return State(jnp.zeros((c, n), jnp.int32), zn, zn, z - NEVER,
                 f, z + NEVER, z - NEVER, i, f, z, i, i - 1, z - NEVER, z - NEVER,
                 f, z + NEVER, z, z, i, z - NEVER, zn - NEVER, zn - NEVER, zn, zn, z, z, zn + NEVER,
                 f, z + NEVER, i - 1, i, i, z - NEVER, z - NEVER, z - NEVER, z - NEVER)


def _mine(kctx) -> Any:
    return kctx.champion_id == ID


def _this_tick(kctx, t: Any) -> Any:
    return jnp.abs(t - kctx.now) < 0.5 * kctx.dt


def _base_cd(state: State, kctx) -> Any:
    """Base cooldowns; a W kill restarts W at half, a rank-3 execution resets R (DAR.W2, DAR.R2)."""
    cd = cooldown_row(NAME, kctx.ranks)
    cd = cd.at[:, 1].multiply(jnp.where(_this_tick(kctx, state.w_refund_at), 1.0 - W_REFUND, 1.0))
    cd = cd.at[:, 3].multiply(jnp.where(_this_tick(kctx, state.r_reset_at), 0.0, 1.0))
    return jnp.where(_mine(kctx)[:, None], cd, 0.0)


def bleed_per_stack(level: Any, bonus_ad: Any) -> Any:
    """Hemorrhage damage per stack over its 5 s: 13 -> 30 linearly by level (+30% bonus AD)."""
    lv = jnp.clip(jnp.asarray(level, jnp.float32), 1.0, 18.0)
    return P_BLEED_LV1 + (P_BLEED_LV18 - P_BLEED_LV1) * (lv - 1.0) / 17.0 + P_BLEED_AD * bonus_ad


def might_ad(level: Any) -> Any:
    """Noxian Might bonus AD: 30, +5 per level to 10, +10 per level 11-13, +25 per level from 14."""
    lv = jnp.asarray(level, jnp.float32)
    base, per = _MIGHT["mLevel1Value"], _MIGHT["mInitialBonusPerLevel"]
    b11, b14 = (b["mBonusPerLevelAtAndAfter"] for b in _MIGHT["mBreakpoints"])
    return base + per * (jnp.clip(lv, 1, 10) - 1) + b11 * jnp.clip(lv - 10, 0, 3) + b14 * jnp.maximum(lv - 13, 0)


def _apply(state: State, kctx, units, mask) -> State:
    """One Hemorrhage stack (five under Noxian Might) on ``mask`` (C, N) units; a champion reaching five stacks
    grants Noxian Might."""
    mask = mask & ~is_structure(units.kind)[None, :]
    might = (kctx.now < state.might_until)[:, None]
    fresh = mask & (state.stacks == 0)
    stacks = jnp.where(mask, jnp.where(might, P_MAX, jnp.minimum(state.stacks + 1, P_MAX)), state.stacks)
    full = mask & (stacks >= P_MAX) & (units.kind == KIND_CHAMPION)[None, :]
    return state._replace(
        stacks=stacks.astype(jnp.int32),
        bleed_until=jnp.where(mask, f32(kctx.now + P_DURATION), state.bleed_until),
        tick_at=jnp.where(fresh, f32(kctx.now + P_TICK), state.tick_at),
        might_until=jnp.where(jnp.any(full, axis=1), f32(kctx.now + P_MIGHT_DURATION), state.might_until))


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
    rooted = jnp.zeros_like(m) if kctx.rooted is None else kctx.rooted
    e_locked = now < state.e_lock_until - 1e-6
    r_locked = now < state.r_lock_until - 1e-6

    q = want(0) & ready[:, 0] & ~e_locked & ~r_locked
    w = want(1) & ready[:, 1] & ~state.w_on
    e = want(2) & ready[:, 2] & ~r_locked
    t = jnp.clip(order.target, 0, n - 1)
    valid_r = (order.target >= 0) & units.alive[t] & units.targetable[t] & (units.kind[t] == KIND_CHAMPION) \
        & (units.team[t] != kctx.team) & (target_dist(kctx, units, order.target) <= R_RANGE + units.radius[t])
    recast = now < state.r_recast_until
    rr = want(3) & (r[:, 3] > 0) & valid_r & ~rooted & ~e_locked \
        & ((kctx.cooldowns[:, 3] <= 0) & (kctx.mana >= cost[:, 3]) | recast)
    started = q | w | e | rr
    slot = jnp.where(q, 0, jnp.where(w, 1, jnp.where(e, 2, jnp.where(rr, 3, -1)))).astype(jnp.int32)
    cid = jnp.where(started, make_cast_id(kctx, jnp.maximum(slot, 0)), 0).astype(jnp.int32)
    dx, dy = order.x - kctx.x, order.y - kctx.y
    d = jnp.sqrt(dx * dx + dy * dy)
    ux, uy = jnp.where(d > 1e-3, dx / jnp.maximum(d, 1e-3), 1.0), jnp.where(d > 1e-3, dy / jnp.maximum(d, 1e-3), 0.0)
    dist_r = target_dist(kctx, units, order.target)
    dash = Dash(rr, f32(units.x[t]), f32(units.y[t]), f32(jnp.maximum(dist_r, 1.0) / R_CAST_TIME),
                jnp.where(rr, order.target, -1).astype(jnp.int32), jnp.zeros((c,), bool))
    state = state._replace(
        q_pending=state.q_pending | q, q_fire_at=jnp.where(q, later(kctx, Q_DELAY), state.q_fire_at),
        q_ghost_until=jnp.where(q, later(kctx, Q_GHOST), state.q_ghost_until),
        q_cast_id=jnp.where(q, cid, state.q_cast_id),
        w_on=state.w_on | w, w_until=jnp.where(w, later(kctx, W_WINDOW), state.w_until),
        w_cast_id=jnp.where(w, cid, state.w_cast_id),
        e_pending=state.e_pending | e, e_fire_at=jnp.where(e, later(kctx, E_CAST_TIME), state.e_fire_at),
        e_dx=jnp.where(e, f32(ux), state.e_dx), e_dy=jnp.where(e, f32(uy), state.e_dy),
        e_cast_id=jnp.where(e, cid, state.e_cast_id),
        r_pending=state.r_pending | rr, r_fire_at=jnp.where(rr, later(kctx, R_CAST_TIME), state.r_fire_at),
        r_target=jnp.where(rr, order.target, state.r_target).astype(jnp.int32),
        r_seq=jnp.where(rr, units.spawn_seq[t], state.r_seq).astype(jnp.int32),
        r_cast_id=jnp.where(rr, cid, state.r_cast_id),
        r_recast_until=jnp.where(rr & recast, -NEVER, state.r_recast_until))
    pay = jnp.stack([q, w, e, rr & ~recast], -1)
    mana = jnp.sum(jnp.where(pay, cost, 0.0), axis=1)
    lock = jnp.where(e, E_CAST_TIME + E_LOCK, jnp.where(rr, R_CAST_TIME, 0.0))
    cd = jnp.stack([jnp.zeros_like(q), jnp.zeros_like(q), e, rr & ~recast], -1)
    return state, out(c, n, dash=dash, mana_cost=f32(mana), cooldown_start=cd, base_cooldown=_base_cd(state, kctx),
                      attack_reset=w, cast_started=started, cast_slot=slot, cast_id=cid, cast_lockout=f32(lock))


def periodic(state: State, kctx, units) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    alive = kctx.alive
    dead = ~alive
    r = kctx.ranks
    now = f32(kctx.now)
    cols = jnp.arange(n)[None, :]
    z = jnp.zeros((c, n), jnp.float32)

    # Decimate: the swing after the 0.75 s heft; blade and handle, healing per champion or large monster bladed.
    q_due = state.q_pending & (due(kctx, state.q_fire_at) | dead)
    q_fire = q_due & alive
    dist = center_dist(kctx, units)
    in_q = q_fire[:, None] & enemies(kctx, units) & (dist <= Q_OUTER + units.radius[None, :])
    handle = in_q & (dist <= Q_INNER)
    blade = in_q & ~handle
    q_raw = ranked(NAME, "Q", "BaseDamage", r[:, 0]) + ranked(NAME, "Q", "TotalADRatio", r[:, 0]) / 100.0 \
        * kctx.total_ad
    p_q = D.packets(in_q, kctx.unit[:, None], cols, (q_raw[:, None] * jnp.where(handle, Q_HANDLE, 1.0)),
                    D.PHYSICAL, Q_FLAGS, cast_id=state.q_cast_id[:, None])
    big = blade & _large(units, jnp.arange(n))[None, :] & ((units.kind == KIND_CHAMPION)
                                                           | (units.kind == KIND_MONSTER))[None, :]
    n_heal = jnp.minimum(jnp.sum(big, axis=1), Q_HEAL_TARGETS).astype(jnp.float32)
    heal = jnp.where(q_fire, Q_HEAL * n_heal * jnp.maximum(kctx.max_hp - kctx.hp, 0.0), 0.0)
    state = _apply(state, kctx, units, blade)

    # Crippling Strike window expires (cooldown post-effect); a kill on the last tick refunds.
    w_end = state.w_on & (due(kctx, state.w_until) | dead)
    refund = _this_tick(kctx, state.w_refund_at)

    # Apprehend: the sweep at the end of the cast time pulls every enemy in the cone to 150 from Darius.
    e_due = state.e_pending & (due(kctx, state.e_fire_at) | dead)
    e_fire = e_due & alive
    rx, ry = units.x[None, :] - kctx.x[:, None], units.y[None, :] - kctx.y[:, None]
    along = rx * state.e_dx[:, None] + ry * state.e_dy[:, None]
    ang = jnp.arctan2(jnp.abs(rx * state.e_dy[:, None] - ry * state.e_dx[:, None]), along)
    slack = jnp.arcsin(jnp.clip(units.radius[None, :] / jnp.maximum(dist, 1e-3), 0.0, 1.0))
    pulled = e_fire[:, None] & enemies(kctx, units) & (dist <= E_RANGE + units.radius[None, :]) \
        & ((ang <= jnp.deg2rad(E_HALF_ANGLE) + slack) | (dist <= units.radius[None, :]))
    ux = jnp.where(dist > 1e-3, rx / jnp.maximum(dist, 1e-3), state.e_dx[:, None])
    uy = jnp.where(dist > 1e-3, ry / jnp.maximum(dist, 1e-3), state.e_dy[:, None])
    t_pull = jnp.maximum(dist - E_PULL_STOP, 0.0) / E_PULL_SPEED
    t_air = t_pull + E_REBOUND_TIME
    up = jnp.where(pulled, t_air, 0.0)
    disp_until = jnp.where(pulled, f32(now + kctx.dt + t_air), state.disp_until)
    pull_until = jnp.where(pulled, f32(now + kctx.dt + t_pull), state.pull_until)
    disp_ux = jnp.where(pulled, f32(ux), state.disp_ux)
    disp_uy = jnp.where(pulled, f32(uy), state.disp_uy)
    anchor_x = jnp.where(e_fire, kctx.x, state.anchor_x)
    anchor_y = jnp.where(e_fire, kctx.y, state.anchor_y)
    slow_at = jnp.where(pulled, f32(now + t_air), state.slow_at)
    slow_now = (state.slow_at <= now + 0.5 * kctx.dt) & units.alive[None, :]
    slow = jnp.where(slow_now, E_SLOW, 0.0)
    slow_dur = jnp.where(slow_now, E_SLOW_DURATION, 0.0)
    slow_at = jnp.where(slow_now | ~units.alive[None, :], NEVER, slow_at)
    near = jnp.where(pulled, dist - jnp.where(units.kind[None, :] == KIND_CHAMPION, 1e5, 0.0), jnp.inf)
    follow = jnp.where(e_fire & jnp.any(pulled, axis=1), jnp.argmin(near, axis=1), -1).astype(jnp.int32)

    # Noxian Guillotine: true damage at the end of the cast time, then a Hemorrhage stack.
    rt = jnp.clip(state.r_target, 0, n - 1)
    r_due = state.r_pending & (due(kctx, state.r_fire_at) | dead)
    r_fire = r_due & alive & units.alive[rt] & (units.spawn_seq[rt] == state.r_seq)
    stacks_t = jnp.take_along_axis(state.stacks, rt[:, None], axis=1)[:, 0].astype(jnp.float32)
    r_raw = (ranked(NAME, "R", "RBaseDamage", r[:, 3]) + R_AD * kctx.bonus_ad) * (1.0 + R_PER_STACK * stacks_t)
    p_r = D.packets(r_fire, kctx.unit, rt, r_raw, D.TRUE, R_FLAGS, cast_id=state.r_cast_id)
    state = _apply(state, kctx, units, r_fire[:, None] & (cols == rt[:, None]))

    # Hemorrhage ticks: a quarter of the per-stack total per stack, x2.5 vs monsters; dead units drop stacks.
    has = (state.stacks > 0) & units.alive[None, :]
    tick = has & due(kctx, state.tick_at)
    per = bleed_per_stack(kctx.level, kctx.bonus_ad)[:, None] / P_TICKS * state.stacks.astype(jnp.float32)
    raw = per * jnp.where(units.kind == KIND_MONSTER, P_MONSTER, 1.0)[None, :]
    p_b = D.packets(tick, kctx.unit[:, None], cols, raw, D.PHYSICAL, BLEED_FLAGS)
    tick_at = jnp.where(tick, state.tick_at + P_TICK, state.tick_at)
    gone = ~units.alive[None, :] | (due(kctx, state.bleed_until) & ~(tick_at < state.bleed_until - 1e-4))
    stacks = jnp.where(gone, 0, state.stacks)

    state = state._replace(
        stacks=stacks.astype(jnp.int32), tick_at=f32(tick_at),
        q_pending=state.q_pending & ~q_due, w_on=state.w_on & ~w_end,
        e_pending=state.e_pending & ~e_due,
        e_lock_until=jnp.where(e_fire, f32(now + E_LOCK), state.e_lock_until),
        disp_until=f32(disp_until), pull_until=f32(pull_until), disp_ux=disp_ux, disp_uy=disp_uy,
        anchor_x=f32(anchor_x), anchor_y=f32(anchor_y),
        slow_at=f32(slow_at),
        r_pending=state.r_pending & ~r_due,
        r_hit_at=jnp.where(r_fire, now, state.r_hit_at),
        r_lock_until=jnp.where(r_fire, f32(now + R_LOCK), state.r_lock_until))
    cc = CCOut(z, z, z, f32(up), f32(slow), f32(slow_dur),
               jnp.where(pulled | slow_now, state.e_cast_id[:, None], 0).astype(jnp.int32))
    reset = _this_tick(kctx, state.r_reset_at)
    cd = jnp.zeros((c, 4), bool).at[:, 0].set(q_due & m).at[:, 1].set((w_end | refund) & m).at[:, 3].set(reset & m)
    return state, out(c, n, packets=D.concat_packets(p_q, p_r, p_b), cc=cc, heal=f32(heal), cooldown_start=cd,
                      base_cooldown=_base_cd(state, kctx),
                      mana_cost=f32(jnp.where(refund & m, -mana_row(NAME, r)[:, 1], 0.0)), attack_target=follow)


def on_attack(state: State, kctx, units, launch) -> tuple[State, KitOut]:
    return state, out(kctx.unit.shape[0], units.x.shape[0], base_cooldown=_base_cd(state, kctx))


def on_hit(state: State, kctx, units, launch, dodging) -> tuple[State, KitOut]:
    """Crippling Strike and Hemorrhage on landed attacks. ``dodging`` (N,) bool."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    target = holder_rows(launch.target, kctx)
    t = jnp.clip(target, 0, n - 1)
    hit = holder_rows(launch.launched, kctx) & m & kctx.alive & (target >= 0)
    land = hit & ~gather(dodging, target)
    w = hit & state.w_on
    w_land = w & land
    crit = jnp.zeros((c,), bool) if launch.is_crit is None else holder_rows(launch.is_crit, kctx)
    bonus = (ranked(NAME, "W", "ADRatio", kctx.ranks[:, 1]) - 1.0) * kctx.total_ad \
        * jnp.where(crit, kctx.crit_damage, 1.0)
    p_w = D.packets(w_land, kctx.unit, t, bonus, D.PHYSICAL, W_FLAGS, cast_id=state.w_cast_id)
    mask = (jnp.arange(n)[None, :] == t[:, None])
    slow, sid = cc_matrix(w_land[:, None] & mask, W_SLOW, state.w_cast_id[:, None])
    dur = jnp.where(w_land[:, None] & mask, W_SLOW_DURATION, 0.0)
    z = jnp.zeros((c, n), jnp.float32)
    state = _apply(state, kctx, units, land[:, None] & mask)
    state = state._replace(w_on=state.w_on & ~w, w_hit=jnp.where(w_land, target, state.w_hit).astype(jnp.int32),
                           w_hit_at=jnp.where(w_land, f32(kctx.now), state.w_hit_at))
    cd = jnp.zeros((c, 4), bool).at[:, 1].set(w)
    return state, out(c, n, packets=p_w, cc=CCOut(z, z, z, z, f32(slow), f32(dur), sid), cooldown_start=cd,
                      base_cooldown=_base_cd(state, kctx))


def on_damage(state: State, kctx, units, report) -> tuple[State, KitOut]:
    return state, out(kctx.unit.shape[0], units.x.shape[0])


def on_takedown(state: State, kctx, units, kills) -> State:
    """Crippling Strike kill refund; Noxian Guillotine executions (recast window, rank-3 reset, Noxian Might)."""
    m = _mine(kctx)
    killed = jnp.asarray(kills.killed_units, bool)
    wt = jnp.clip(state.w_hit, 0, units.x.shape[0] - 1)
    w_kill = m & (state.w_hit >= 0) & _this_tick(kctx, state.w_hit_at) \
        & jnp.take_along_axis(killed, wt[:, None], axis=1)[:, 0]
    rt = jnp.clip(state.r_target, 0, units.x.shape[0] - 1)
    r_kill = m & (kctx.now - state.r_hit_at <= R_KILL_WINDOW + 1e-6) & (jnp.asarray(kills.champion_kill) > 0) \
        & jnp.take_along_axis(killed, rt[:, None], axis=1)[:, 0]
    rank3 = kctx.ranks[:, 3] >= 3
    now = f32(kctx.now)
    return state._replace(
        w_refund_at=jnp.where(w_kill, now + kctx.dt, state.w_refund_at),
        r_recast_until=jnp.where(r_kill & ~rank3, f32(now + R_RECAST), state.r_recast_until),
        r_reset_at=jnp.where(r_kill & rank3, now + kctx.dt, state.r_reset_at),
        might_until=jnp.where(r_kill, f32(now + P_MIGHT_DURATION), state.might_until),
        r_hit_at=jnp.where(r_kill, -NEVER, state.r_hit_at))


def stats(state: State, kctx) -> ItemStats:
    m = _mine(kctx)
    pen = jnp.where(m & (kctx.ranks[:, 2] > 0), ranked(NAME, "E", "PassivePercentArmorPen", kctx.ranks[:, 2]) / 100.0,
                    0.0)
    might = m & (kctx.now < state.might_until)
    return ItemStats(attack_damage=f32(jnp.where(might, might_ad(kctx.level), 0.0)), percent_armor_pen=f32(pen))


def defense(state: State, kctx) -> KitDefense:
    c = kctx.unit.shape[0]
    o = jnp.ones((c,), jnp.float32)
    return KitDefense(o, jnp.zeros((c,), bool), o, jnp.zeros((c,), jnp.float32))


def attack_mods(state: State, kctx) -> KitAttackMods:
    m = _mine(kctx)
    c = kctx.unit.shape[0]
    z = jnp.zeros((c,), jnp.float32)
    w = m & state.w_on
    return KitAttackMods(f32(jnp.where(w, W_RANGE_BONUS, 0.0)), jnp.zeros((c,), bool), m & state.q_pending,
                         jnp.zeros((c,), bool), z, z, w)


def ghosted(state: State, kctx) -> Any:
    """Decimate ghosts Darius for 1 s."""
    return _mine(kctx) & kctx.alive & (kctx.now < state.q_ghost_until)


def displacement(state: State, kctx, n_units: int) -> tuple[Any, Any, Any, Any]:
    """(N,) Apprehend: ``(active, to_x, to_y, speed)`` for the world to move — the pull to 45 from Darius at 1784,
    then the rebound to 230 over 0.5 s."""
    on = _mine(kctx)[:, None] & (kctx.now < state.disp_until)
    pulling = kctx.now < state.pull_until
    reach = jnp.where(pulling, E_PULL_STOP, E_REBOUND)
    tx = state.anchor_x[:, None] + state.disp_ux * reach
    ty = state.anchor_y[:, None] + state.disp_uy * reach
    sp = jnp.where(pulling, E_PULL_SPEED, (E_REBOUND - E_PULL_STOP) / E_REBOUND_TIME)
    k = jnp.argmax(on, axis=0)
    pick = lambda a: jnp.take_along_axis(jnp.broadcast_to(a, on.shape), k[None, :], axis=0)[0]   # noqa: E731
    return jnp.any(on, axis=0), pick(tx), pick(ty), pick(sp)


def debuffs(state: State, kctx, units) -> Debuffs:
    return neutral_debuffs(units.x.shape[0])


__all__ = ["State", "init", "cast", "periodic", "on_attack", "on_hit", "on_damage", "on_takedown", "stats",
           "defense", "attack_mods", "ghosted", "displacement", "debuffs", "bleed_per_stack", "might_ad"]
