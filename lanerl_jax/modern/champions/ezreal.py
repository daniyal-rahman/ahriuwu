"""Ezreal (81), 26.19. Rules EZR.* and their evidence: docs/modern/CHAMPIONS.md; numbers from the pinned client
JSON (``modern/data/26.19/champions``) and, where marked, the 16.9 replay measurements
(/mnt/nfs/shared/champ-expansion/roster-measurements).

Ezreal is ranged (550 attack range, 2000 units/s attack missiles via ``attack_missile_speed``). His skillshots are
kit missiles: a flat front advancing each tick that touches a unit when it reaches the unit's hitbox and is within
its half-width. Mystic Shot deals basic damage, applies on-hit and on-attack effects (``KitOut.on_hit_target`` +
``on_attack_proc``), triggers spell effects with a 0-damage spell packet, and refunds 1.5 s of every current
cooldown (``KitOut.cooldown_refund``). Essence Flux marks the first champion, epic monster or structure; Ezreal's
next landed attack or ability on the mark detonates it (an ability also restores mana, ``KitOut.mana_gain``).
Arcane Shift is a kit blink; its bolt is aimed from where Ezreal landed, on the next tick."""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..core import damage as D
from ..core.types import KIND_CHAMPION, KIND_MINION, KIND_MONSTER, KIND_WARD, Dash
from ..data.champions import spell
from ..items.catalog import ItemStats
from ..items.effects.core import Debuffs, neutral_debuffs
from ..jungle import objectives as OBJ
from ..runes.catalog import ChampionTraits
from .core import (NEVER, KitAttackMods, KitDefense, KitOut, cooldown_row, due, enemies, f32, gather, holder_rows,
                   is_structure, later, later_after_tick, make_cast_id, mana_row, out, ranked, scalar)
from .nasus import LARGE_EPICS

NAME = "Ezreal"
ID = 81
# LoLalytics 16.19 Emerald+ bottom (level 15 skill order; 16-18 by the Q > E > W priority).
SKILL_ORDER = (0, 2, 1, 0, 0, 3, 0, 2, 0, 2, 3, 2, 2, 1, 1, 3, 1, 1)
TRAITS = ChampionTraits(has_immobilize=False, resource="mana")         # no crowd control in the kit
UNIT_TARGET_RANGE = (0.0, 0.0, 0.0, 0.0)
RADIUS = 65.0                          # no overrideGameplayCollisionRadius on the record
ATTACK_MISSILE = 2000.0                # EzrealBasicAttack movement mSpeed (crit attack the same)

# Rising Spell Force.
P_AS = scalar(NAME, "Passive", "AttackSpeedPerStack")                  # +10% bonus attack speed per stack
P_TIME = scalar(NAME, "Passive", "StackDuration")                      # 6 s, refreshed by each new stack
P_MAX = int(scalar(NAME, "Passive", "MaxStacks"))                      # 5


def _ratio(slot: str, calc: str, part: int) -> float:
    p = spell(NAME, slot)["calculations"][calc]["mFormulaParts"][part]
    return float(p["mCoefficient"]) if "mCoefficient" in p else scalar(NAME, slot, p["mDataValue"])


# Mystic Shot.
Q_CAST = 0.25                          # spellCastTime (replay: hits at 0.25 + d / 2000)
Q_SPEED = 2000.0                       # missile movement mSpeed
Q_RANGE = 1150.0                       # castRangeDisplayOverride (replay: no hit beyond ~1200 center distance = 1150
                                       # + a 65 hitbox; castRange 1200 with the front reaching hitboxes would hit to 1265)
Q_HALF = 60.0                          # mMissileWidth (half-width; wiki width 120)
Q_AD = _ratio("Q", "Damage", 1)        # 1.3 total AD (BaseDamageADRatio)
Q_AP = _ratio("Q", "Damage", 2)        # 0.4 AP (PATCH 26.9)
Q_REFUND = scalar(NAME, "Q", "CDRefund")                               # 1.5 s off every current cooldown

# Essence Flux.
W_CAST = 0.25                          # spellCastTime
W_SPEED = 1700.0                       # missile movement mSpeed = wiki (the spell's missileSpeed 1200 is not used)
W_RANGE = Q_RANGE                      # same castRange / display override as Mystic Shot (INFERRED from EZR.Q1)
W_HALF = 80.0                          # mMissileWidth (wiki width 160)
W_TIME = scalar(NAME, "W", "DetonationTimeout")                       # 4 s
W_AD = _ratio("W", "Damage", 1)        # 1.0 bonus AD
W_AP = _ratio("W", "Damage", 2)        # 0.9 AP (PATCH 26.9)
W_MANA = scalar(NAME, "W", "ManaReturn")                               # 60 + the detonating ability's cost

# Arcane Shift.
E_CAST = 0.25                          # spellCastTime (replay: the blink 0.27-0.33 s after the record)
E_BLINK = 475.0                        # castRangeDisplayOverride = wiki (replay: jumps peak at 470-480)
E_BOLT_RANGE = scalar(NAME, "E", "MissileRange")                       # 750 from where Ezreal lands (center)
E_BOLT_SPEED = 2000.0                  # EzrealEMissile movement mSpeed (homing)
E_AD = _ratio("E", "Damage", 1)        # 0.6 bonus AD (PATCH 26.3)
E_AP = _ratio("E", "Damage", 2)        # 0.75 AP

# Trueshot Barrage.
R_CAST = 1.0                           # spellCastTime (replay: hits at 1.0 + d / 2000)
R_SPEED = 2000.0                       # missile movement mSpeed
R_RANGE = 25000.0                      # castRange (global)
R_HALF = 160.0                         # mMissileWidth (wiki width 320)
R_AD = _ratio("R", "Damage", 1)        # 1.0 bonus AD
R_AP = _ratio("R", "Damage", 2)        # 1.1 AP (PATCH 26.9)
_RM = spell(NAME, "R")["calculations"]["DamageMinionMonster"]["mFormulaParts"]
R_MINION_AD = float(_RM[1]["mCoefficient"])                            # minions / non-epic monsters: 1.0 bonus AD
R_MINION_AP = float(_RM[2]["mCoefficient"])                            # ... and 1.1 AP (PATCH 26.9)

Q_FLAGS = D.BASIC_ATTACK               # wiki: basic damage (life steal), on-hit and on-attack effects
SPELL = D.TAG_ACTIVE_SPELL
R_FLAGS = D.TAG_AOE | D.TAG_ACTIVE_SPELL | D.PROP_ULTIMATE
MARK_FLAGS = D.TAG_PROC                # wiki: the mark's application deals 0 proc damage
CODE_Q, CODE_W, CODE_E, CODE_R = 0, 1, 2, 3


class State(NamedTuple):
    p_stacks: Any           # (C,) int32 Rising Spell Force
    p_until: Any
    q_pending: Any          # Mystic Shot cast time
    q_fire_at: Any
    q_dx: Any
    q_dy: Any
    q_cast_id: Any
    q_fly: Any              # the Mystic Shot missile
    q_ox: Any
    q_oy: Any
    q_mdx: Any
    q_mdy: Any
    q_dist: Any
    q_mid: Any              # its cast id
    w_pending: Any          # Essence Flux cast time
    w_fire_at: Any
    w_dx: Any
    w_dy: Any
    w_cast_id: Any
    w_fly: Any              # the Essence Flux orb
    w_ox: Any
    w_oy: Any
    w_mdx: Any
    w_mdy: Any
    w_dist: Any
    w_mid: Any
    mark_on: Any            # (C,) bool: an Essence Flux mark is up
    mark_unit: Any          # int32
    mark_seq: Any
    mark_until: Any
    mark_cast_id: Any
    e_pending: Any          # Arcane Shift cast time
    e_fire_at: Any
    e_tx: Any               # the target point
    e_ty: Any
    e_cast_id: Any
    e_aim: Any              # (C,) bool: blinked; the bolt picks its target next tick, from the landing point
    e_bolt: Any             # (C,) bool: the homing bolt is flying
    e_tgt: Any
    e_seq: Any
    e_bx: Any
    e_by: Any
    r_pending: Any          # Trueshot Barrage cast time
    r_fire_at: Any
    r_dx: Any
    r_dy: Any
    r_cast_id: Any
    r_fly: Any              # the barrage
    r_ox: Any
    r_oy: Any
    r_mdx: Any
    r_mdy: Any
    r_dist: Any
    r_mid: Any
    r_hit: Any              # (C, N) bool: already struck by this barrage


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    i = jnp.zeros((c,), jnp.int32)
    return State(p_stacks=i, p_until=z - NEVER,
                 q_pending=f, q_fire_at=z + NEVER, q_dx=z, q_dy=z, q_cast_id=i,
                 q_fly=f, q_ox=z, q_oy=z, q_mdx=z, q_mdy=z, q_dist=z, q_mid=i,
                 w_pending=f, w_fire_at=z + NEVER, w_dx=z, w_dy=z, w_cast_id=i,
                 w_fly=f, w_ox=z, w_oy=z, w_mdx=z, w_mdy=z, w_dist=z, w_mid=i,
                 mark_on=f, mark_unit=i - 1, mark_seq=i, mark_until=z - NEVER, mark_cast_id=i,
                 e_pending=f, e_fire_at=z + NEVER, e_tx=z, e_ty=z, e_cast_id=i,
                 e_aim=f, e_bolt=f, e_tgt=i - 1, e_seq=i, e_bx=z, e_by=z,
                 r_pending=f, r_fire_at=z + NEVER, r_dx=z, r_dy=z, r_cast_id=i,
                 r_fly=f, r_ox=z, r_oy=z, r_mdx=z, r_mdy=z, r_dist=z, r_mid=i, r_hit=jnp.zeros((c, n), bool))


def _mine(kctx) -> Any:
    return kctx.champion_id == ID


def _aim(kctx, px, py) -> tuple[Any, Any, Any]:
    """Unit direction from the holder to ``(px, py)`` and the distance (0 -> no direction)."""
    dx, dy = px - kctx.x, py - kctx.y
    d = jnp.sqrt(dx * dx + dy * dy)
    ok = d > 1e-3
    s = jnp.where(ok, d, 1.0)
    return f32(jnp.where(ok, dx / s, 0.0)), f32(jnp.where(ok, dy / s, 0.0)), d


def epic_monster(units) -> Any:
    """(N,) the epic monsters (Voidgrubs, Rift Herald, Baron, drakes, Elder)."""
    epic = jnp.zeros(units.kind.shape, bool)
    for e in LARGE_EPICS:
        epic = epic | (units.sub == OBJ.SUB_BASE + e)
    return (units.kind == KIND_MONSTER) & epic


def _sweep(units, ox, oy, dx, dy, d0, d1, half) -> tuple[Any, Any]:
    """(C, N) units the missile front touches while advancing from ``d0`` to ``d1`` along ``(dx, dy)`` from
    ``(ox, oy)``, and where along the path it first reaches each one."""
    sx, sy = units.x[None, :] - ox[:, None], units.y[None, :] - oy[:, None]
    along = sx * dx[:, None] + sy * dy[:, None]
    perp = jnp.abs(sy * dx[:, None] - sx * dy[:, None])
    rad = units.radius[None, :]
    touch = (perp <= half + rad) & (along - rad <= d1[:, None]) & (along + rad >= d0[:, None])
    return touch, jnp.maximum(along - rad, d0[:, None])


def _first(mask, s) -> tuple[Any, Any, Any]:
    """(C,) whether any unit was touched, the first one, and the (C, N) one-hot."""
    s_m = jnp.where(mask, s, jnp.inf)
    k = jnp.argmin(s_m, axis=1)
    hit = jnp.any(mask, axis=1)
    one = mask & (jnp.arange(mask.shape[1])[None, :] == k[:, None])
    return hit, k.astype(jnp.int32), one


def _fly(state_fly, go, ox_new, oy_new, dx_new, dy_new, ox, oy, dx, dy, dist):
    """A missile's path this tick: launched now (``go``) from the holder, or continuing."""
    fly = state_fly | go
    return (fly, jnp.where(go, ox_new, ox), jnp.where(go, oy_new, oy), jnp.where(go, dx_new, dx),
            jnp.where(go, dy_new, dy), jnp.where(go, 0.0, dist))


def cast(state: State, kctx, units, order) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    r = kctx.ranks
    busy = state.q_pending | state.w_pending | state.e_pending | state.r_pending
    free = m & kctx.alive & ~kctx.stunned & ~kctx.silenced & ~busy
    cost = mana_row(NAME, r)
    ready = (r > 0) & (kctx.cooldowns <= 0) & (kctx.mana[:, None] >= cost)
    rooted = jnp.zeros_like(m) if kctx.rooted is None else kctx.rooted
    ax, ay, ad = _aim(kctx, order.x, order.y)
    aimed = ad > 1e-3
    want = lambda s: (order.slot == s) & free & ready[:, s] & aimed      # noqa: E731
    q, w, e, rr = want(0), want(1), want(2) & ~rooted, want(3)
    started = q | w | e | rr
    slot = jnp.where(q, 0, jnp.where(w, 1, jnp.where(e, 2, jnp.where(rr, 3, -1)))).astype(jnp.int32)
    cid = jnp.where(started, make_cast_id(kctx, jnp.maximum(slot, 0)), 0).astype(jnp.int32)
    state = state._replace(
        q_pending=state.q_pending | q, q_fire_at=jnp.where(q, later(kctx, Q_CAST), state.q_fire_at),
        q_dx=jnp.where(q, ax, state.q_dx), q_dy=jnp.where(q, ay, state.q_dy),
        q_cast_id=jnp.where(q, cid, state.q_cast_id),
        w_pending=state.w_pending | w, w_fire_at=jnp.where(w, later(kctx, W_CAST), state.w_fire_at),
        w_dx=jnp.where(w, ax, state.w_dx), w_dy=jnp.where(w, ay, state.w_dy),
        w_cast_id=jnp.where(w, cid, state.w_cast_id),
        e_pending=state.e_pending | e, e_fire_at=jnp.where(e, later(kctx, E_CAST), state.e_fire_at),
        e_tx=jnp.where(e, f32(order.x), state.e_tx), e_ty=jnp.where(e, f32(order.y), state.e_ty),
        e_cast_id=jnp.where(e, cid, state.e_cast_id),
        r_pending=state.r_pending | rr, r_fire_at=jnp.where(rr, later(kctx, R_CAST), state.r_fire_at),
        r_dx=jnp.where(rr, ax, state.r_dx), r_dy=jnp.where(rr, ay, state.r_dy),
        r_cast_id=jnp.where(rr, cid, state.r_cast_id))
    went = jnp.stack([q, w, e, rr], -1)
    mana = jnp.sum(jnp.where(went, cost, 0.0), axis=1)
    lock = jnp.where(rr, R_CAST, jnp.where(q | w | e, Q_CAST, 0.0))
    return state, out(c, n, mana_cost=f32(mana), cooldown_start=went,
                      base_cooldown=jnp.where(m[:, None], cooldown_row(NAME, r), 0.0), cast_started=started,
                      cast_slot=slot, cast_id=cid, cast_lockout=f32(lock))


def _w_damage(kctx) -> Any:
    return ranked(NAME, "W", "BaseDamage", kctx.ranks[:, 1]) + W_AD * kctx.bonus_ad + W_AP * kctx.ap


def _marked(state: State, units, kctx) -> Any:
    """(C,) the Essence Flux mark is up on its living unit (a reused slot is a new unit)."""
    t = jnp.clip(state.mark_unit, 0, units.x.shape[0] - 1)
    return state.mark_on & (kctx.now < state.mark_until) & units.alive[t] & (units.spawn_seq[t] == state.mark_seq)


def periodic(state: State, kctx, units) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    alive = kctx.alive
    dead = ~alive
    r = kctx.ranks
    now = f32(kctx.now)
    dt = kctx.dt
    cols = jnp.arange(n)[None, :]
    foe = enemies(kctx, units)
    src = kctx.unit[:, None]
    cost = mana_row(NAME, r)
    marked0 = _marked(state, units, kctx)
    mark_t = state.mark_unit

    # Mystic Shot: leaves at the end of the cast time; the first enemy touched takes basic damage.
    q_due = state.q_pending & (due(kctx, state.q_fire_at) | dead)
    q_go = q_due & alive
    q_fly, qox, qoy, qdx, qdy, qd0 = _fly(state.q_fly, q_go, kctx.x, kctx.y, state.q_dx, state.q_dy, state.q_ox,
                                          state.q_oy, state.q_mdx, state.q_mdy, state.q_dist)
    q_mid = jnp.where(q_go, state.q_cast_id, state.q_mid)
    qd1 = jnp.minimum(qd0 + Q_SPEED * dt, Q_RANGE)
    q_touch, q_s = _sweep(units, qox, qoy, qdx, qdy, qd0, qd1, Q_HALF)
    q_hit, q_t, q_one = _first(q_fly[:, None] & foe & q_touch, q_s)
    q_raw = ranked(NAME, "Q", "BaseDamage", r[:, 0]) + Q_AD * kctx.total_ad + Q_AP * kctx.ap
    p_q = D.concat_packets(
        D.packets(q_one, src, cols, q_raw[:, None], D.PHYSICAL, Q_FLAGS, cast_id=q_mid[:, None]),
        D.packets(q_one, src, cols, 0.0, D.PHYSICAL, SPELL, cast_id=q_mid[:, None]))     # wiki: + 0 spell damage
    q_fly = q_fly & ~q_hit & (qd1 < Q_RANGE - 1e-3)

    # Essence Flux: the first enemy champion, epic monster or structure touched is marked for 4 s.
    w_due = state.w_pending & (due(kctx, state.w_fire_at) | dead)
    w_go = w_due & alive
    w_fly, wox, woy, wdx, wdy, wd0 = _fly(state.w_fly, w_go, kctx.x, kctx.y, state.w_dx, state.w_dy, state.w_ox,
                                          state.w_oy, state.w_mdx, state.w_mdy, state.w_dist)
    w_mid = jnp.where(w_go, state.w_cast_id, state.w_mid)
    wd1 = jnp.minimum(wd0 + W_SPEED * dt, W_RANGE)
    w_touch, w_s = _sweep(units, wox, woy, wdx, wdy, wd0, wd1, W_HALF)
    enemy = (units.team[None, :] != kctx.team[:, None]) & units.alive[None, :]
    w_ok = enemy & ((units.targetable & ((units.kind == KIND_CHAMPION) | epic_monster(units)))
                    | is_structure(units.kind))[None, :]
    w_hit, w_t, w_one = _first(w_fly[:, None] & w_ok & w_touch, w_s)
    p_mark = D.packets(w_one, src, cols, 0.0, D.MAGIC, MARK_FLAGS, cast_id=w_mid[:, None])
    w_fly = w_fly & ~w_hit & (wd1 < W_RANGE - 1e-3)

    # Arcane Shift: the blink at the end of the cast time (toward the point, at most 475 from where Ezreal is then).
    e_due = state.e_pending & (due(kctx, state.e_fire_at) | dead)
    e_go = e_due & alive
    bx_, by_, bd = _aim(kctx, state.e_tx, state.e_ty)
    hop = jnp.minimum(bd, E_BLINK)
    blink = Dash(e_go, f32(kctx.x + bx_ * hop), f32(kctx.y + by_ * hop), jnp.full((c,), jnp.inf, jnp.float32),
                 jnp.full((c,), -1, jnp.int32), jnp.ones((c,), bool))
    # ... and on the next tick the bolt picks the nearest enemy within 750 of the landing point (the marked unit
    # first; sight not needed) and homes in at 2000 units/s.
    aim = state.e_aim & alive
    dist = jnp.sqrt((units.x[None, :] - kctx.x[:, None]) ** 2 + (units.y[None, :] - kctx.y[:, None]) ** 2)
    in_reach = foe & (dist <= E_BOLT_RANGE)
    near = jnp.argmin(jnp.where(in_reach, dist, jnp.inf), axis=1).astype(jnp.int32)
    mk = jnp.clip(mark_t, 0, n - 1)
    mark_pick = marked0 & jnp.take_along_axis(in_reach, mk[:, None], axis=1)[:, 0]
    pick = jnp.where(mark_pick, mark_t, near).astype(jnp.int32)
    launch = aim & jnp.any(in_reach, axis=1)
    e_bolt = state.e_bolt | launch
    e_tgt = jnp.where(launch, pick, state.e_tgt).astype(jnp.int32)
    et = jnp.clip(e_tgt, 0, n - 1)
    e_seq = jnp.where(launch, units.spawn_seq[et], state.e_seq).astype(jnp.int32)
    ebx, eby = jnp.where(launch, kctx.x, state.e_bx), jnp.where(launch, kctx.y, state.e_by)
    e_live = units.alive[et] & units.targetable[et] & (units.spawn_seq[et] == e_seq)
    e_bolt = e_bolt & e_live
    gx, gy = units.x[et] - ebx, units.y[et] - eby
    gd = jnp.sqrt(gx * gx + gy * gy)
    step = E_BOLT_SPEED * dt
    e_hit = e_bolt & (gd - units.radius[et] <= step)
    frac = jnp.where(gd > 1e-3, jnp.minimum(step / jnp.maximum(gd, 1e-3), 1.0), 0.0)
    ebx, eby = ebx + gx * frac, eby + gy * frac
    e_one = e_hit[:, None] & (cols == et[:, None])
    e_raw = ranked(NAME, "E", "BaseDamage", r[:, 2]) + E_AD * kctx.bonus_ad + E_AP * kctx.ap
    p_e = D.packets(e_one, src, cols, e_raw[:, None], D.MAGIC, SPELL, cast_id=state.e_cast_id[:, None])
    e_bolt = e_bolt & ~e_hit

    # Trueshot Barrage: leaves at the end of the 1 s cast time and strikes every enemy it passes once.
    r_due = state.r_pending & (due(kctx, state.r_fire_at) | dead)
    r_go = r_due & alive
    r_fly, rox, roy, rdx, rdy, rd0 = _fly(state.r_fly, r_go, kctx.x, kctx.y, state.r_dx, state.r_dy, state.r_ox,
                                          state.r_oy, state.r_mdx, state.r_mdy, state.r_dist)
    r_mid = jnp.where(r_go, state.r_cast_id, state.r_mid)
    rd1 = jnp.minimum(rd0 + R_SPEED * dt, R_RANGE)
    r_touch, _ = _sweep(units, rox, roy, rdx, rdy, rd0, rd1, R_HALF)
    seen = jnp.where(r_go[:, None], False, state.r_hit)
    r_new = r_fly[:, None] & foe & r_touch & ~seen
    small = ((units.kind == KIND_MINION) | ((units.kind == KIND_MONSTER) & ~epic_monster(units)))[None, :]
    r_full = ranked(NAME, "R", "BaseDamage", r[:, 3]) + R_AD * kctx.bonus_ad + R_AP * kctx.ap
    r_less = ranked(NAME, "R", "BaseDamageMinionMonster", r[:, 3]) + R_MINION_AD * kctx.bonus_ad \
        + R_MINION_AP * kctx.ap
    r_raw = jnp.where(small, r_less[:, None], r_full[:, None])
    p_r = D.packets(r_new, src, cols, r_raw, D.MAGIC, R_FLAGS, cast_id=r_mid[:, None])
    r_fly = r_fly & (rd1 < R_RANGE - 1e-3)

    # Essence Flux detonates on the first of this tick's ability hits on the marked unit (Mystic Shot, Arcane
    # Shift, Trueshot Barrage), restoring 60 mana plus that ability's cost; then a new mark replaces the old one.
    on_mark = lambda one: jnp.take_along_axis(one, mk[:, None], axis=1)[:, 0] & marked0      # noqa: E731
    by_q, by_e, by_r = on_mark(q_one), on_mark(e_one), on_mark(r_new)
    det = by_q | by_e | by_r
    refund = W_MANA + jnp.where(by_q, cost[:, 0], jnp.where(by_e, cost[:, 2], cost[:, 3]))
    p_det = D.packets(det[:, None] & (cols == mk[:, None]), src, cols, _w_damage(kctx)[:, None], D.MAGIC, SPELL,
                      cast_id=state.mark_cast_id[:, None])
    keep = marked0 & ~det
    wt = jnp.clip(w_t, 0, n - 1)
    mark_on = jnp.where(w_hit, True, keep)
    mark_unit = jnp.where(w_hit, w_t, state.mark_unit).astype(jnp.int32)
    mark_seq = jnp.where(w_hit, units.spawn_seq[wt], state.mark_seq).astype(jnp.int32)
    mark_until = jnp.where(w_hit, later_after_tick(kctx, W_TIME), state.mark_until)
    mark_cast_id = jnp.where(w_hit, w_mid, state.mark_cast_id)

    # Rising Spell Force: a stack per enemy hit by an ability (the mark's application included).
    hits = q_hit.astype(jnp.int32) + w_hit.astype(jnp.int32) + e_hit.astype(jnp.int32) \
        + jnp.sum(r_new, axis=1).astype(jnp.int32)
    stacks = jnp.where(now < state.p_until, state.p_stacks, 0)
    p_stacks = jnp.where(hits > 0, jnp.minimum(stacks + hits, P_MAX), stacks)
    p_until = jnp.where(hits > 0, later_after_tick(kctx, P_TIME), state.p_until)

    state = state._replace(
        p_stacks=p_stacks.astype(jnp.int32), p_until=f32(p_until),
        q_pending=state.q_pending & ~q_due, q_fly=q_fly, q_ox=f32(qox), q_oy=f32(qoy), q_mdx=f32(qdx),
        q_mdy=f32(qdy), q_dist=f32(qd1), q_mid=q_mid,
        w_pending=state.w_pending & ~w_due, w_fly=w_fly, w_ox=f32(wox), w_oy=f32(woy), w_mdx=f32(wdx),
        w_mdy=f32(wdy), w_dist=f32(wd1), w_mid=w_mid,
        mark_on=mark_on, mark_unit=mark_unit, mark_seq=mark_seq, mark_until=f32(mark_until),
        mark_cast_id=mark_cast_id,
        e_pending=state.e_pending & ~e_due, e_aim=e_go, e_bolt=e_bolt, e_tgt=e_tgt, e_seq=e_seq, e_bx=f32(ebx),
        e_by=f32(eby),
        r_pending=state.r_pending & ~r_due, r_fly=r_fly, r_ox=f32(rox), r_oy=f32(roy), r_mdx=f32(rdx),
        r_mdy=f32(rdy), r_dist=f32(rd1), r_mid=r_mid, r_hit=seen | r_new)
    refund_cd = jnp.where((q_hit & m)[:, None], Q_REFUND, 0.0) * jnp.ones((1, 4), jnp.float32)
    return state, out(c, n, packets=D.concat_packets(p_q, p_mark, p_e, p_r, p_det), dash=blink,
                      base_cooldown=jnp.where(m[:, None], cooldown_row(NAME, r), 0.0),
                      on_hit_target=jnp.where(m & q_hit, q_t, -1).astype(jnp.int32), on_attack_proc=m & q_hit,
                      mana_gain=f32(jnp.where(m & det, refund, 0.0)), cooldown_refund=f32(refund_cd))


def on_attack(state: State, kctx, units, launch) -> tuple[State, KitOut]:
    return state, out(kctx.unit.shape[0], units.x.shape[0])


def on_hit(state: State, kctx, units, launch, dodging) -> tuple[State, KitOut]:
    """A landed basic attack on the marked unit detonates Essence Flux (no mana; dodged or blocked attacks don't)."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    target = holder_rows(launch.target, kctx)
    t = jnp.clip(target, 0, n - 1)
    land = holder_rows(launch.launched, kctx) & m & (target >= 0) & ~gather(dodging, target) \
        & (units.kind[t] != KIND_WARD)
    det = land & _marked(state, units, kctx) & (target == state.mark_unit)
    p = D.packets(det, kctx.unit, t, _w_damage(kctx), D.MAGIC, SPELL, cast_id=state.mark_cast_id)
    return state._replace(mark_on=state.mark_on & ~det), out(c, n, packets=p)


def on_damage(state: State, kctx, units, report) -> tuple[State, KitOut]:
    return state, out(kctx.unit.shape[0], units.x.shape[0])


def on_takedown(state: State, kctx, units, kills) -> State:
    return state


def stats(state: State, kctx) -> ItemStats:
    """Rising Spell Force: +10% bonus attack speed per stack."""
    on = _mine(kctx) & (kctx.now < state.p_until)
    return ItemStats(attack_speed=f32(jnp.where(on, P_AS * state.p_stacks, 0.0)))


def defense(state: State, kctx) -> KitDefense:
    c = kctx.unit.shape[0]
    o = jnp.ones((c,), jnp.float32)
    return KitDefense(o, jnp.zeros((c,), bool), o, jnp.zeros((c,), jnp.float32))


def attack_mods(state: State, kctx) -> KitAttackMods:
    """No attacks during a cast time."""
    m = _mine(kctx)
    c = kctx.unit.shape[0]
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    busy = m & (state.q_pending | state.w_pending | state.e_pending | state.r_pending)
    return KitAttackMods(z, f, busy)


def attack_missile_speed(state: State, kctx) -> Any:
    """(C,) Ezreal's attack missiles (2000 units/s)."""
    return f32(jnp.where(_mine(kctx), ATTACK_MISSILE, 0.0))


def debuffs(state: State, kctx, units) -> Debuffs:
    return neutral_debuffs(units.x.shape[0])


__all__ = ["State", "init", "cast", "periodic", "on_attack", "on_hit", "on_damage", "on_takedown", "stats",
           "defense", "attack_mods", "attack_missile_speed", "debuffs", "epic_monster"]
