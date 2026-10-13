"""Aatrox (266), 26.19. Rules AAT.* and their evidence: docs/modern/CHAMPIONS.md; numbers from the pinned client
JSON (``modern/data/26.19/champions``). Q and W hold Aatrox in place through their cast times without a world cast
lock (``immobile``) so that Umbral Dash stays castable; Infernal Chains' pull is a kit displacement."""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..core import damage as D
from ..core.types import KIND_CHAMPION, KIND_MINION, KIND_MONSTER, KIND_WARD, CCOut, Dash
from ..data.champions import spell
from ..items.catalog import ItemStats
from ..items.effects.core import Debuffs, neutral_debuffs
from ..runes.catalog import ChampionTraits
from .core import (CODE_AATROX_P, CODE_AATROX_W2, NEVER, KitAttackMods, KitDefense, KitOut, cc_matrix, cooldown_row,
                   due, enemies, f32, holder_rows, is_structure, later, make_cast_id, out, ranked, scalar)
from .nasus import _large

NAME = "Aatrox"
ID = 266
SKILL_ORDER = (0, 2, 1, 0, 0, 3, 0, 2, 0, 2, 3, 2, 2, 1, 1, 3, 1, 1)    # default ranks per level: Q > E > W
TRAITS = ChampionTraits(has_immobilize=True, resource="none")          # rune legality: Q knocks up; no mana
UNIT_TARGET_RANGE = (0.0, 0.0, 0.0, 0.0)

_P = spell(NAME, "Passive")["calculations"]
_lv = lambda name: (_P[name]["mFormulaParts"][0]["mStartValue"], _P[name]["mFormulaParts"][0]["mEndValue"])  # noqa: E731
P_DAMAGE = _lv("PDamage")                             # 4% -> 10% target max health by level
P_COOLDOWN = _lv("PCooldown")                         # 22 -> 10 s by level (static)
P_MONSTER_CAP = _lv("MonsterDamageCap")               # 100 -> 320 by level
P_RANGE = scalar(NAME, "Passive", "PBonusAARange")    # +50
P_CDR = scalar(NAME, "Passive", "PChargeRate")        # 2 s per champion / large monster hit (4 s Q sweetspot)
P_HEAL = scalar(NAME, "Passive", "PHealingRatio")     # 100% of the post-mitigation bonus damage
P_HEAL_MINION = scalar(NAME, "Passive", "PHealingMinionMod")   # 25% against minions

Q_CAST_TIME = 0.6                                     # client AatroxQWrapperCast.mCastTime
Q_RECAST_GAP = 1.0                                    # wiki: 1 s static cooldown between casts
Q_WINDOW = scalar(NAME, "Q", "QExtensionTime")        # 4 s to recast
Q_RAMP = scalar(NAME, "Q", "QRampBonus")              # +25% per cast
Q_SWEET = scalar(NAME, "Q", "QSweetSpotBonus")        # +75% (26.12)
Q_KNOCKUP = scalar(NAME, "Q", "QKnockupDuration")     # 0.25 s, doubled against monsters
Q_MONSTER_BONUS = scalar(NAME, "Q", "QMonsterBonus")  # +25 physical against monsters (26.2)
_QM = spell(NAME, "Q")["calculations"]["QMinionDamage"]["mFormulaParts"][0]
Q_MINION_BASE = _QM["mLevel1Value"]                   # 55% against minions, +5% at levels 6, 11 and 16
Q_MINION_STEPS = tuple(b["mLevel"] for b in _QM["mBreakpoints"])
Q_MINION_STEP = _QM["mBreakpoints"][0]["mAdditionalBonusAtThisLevel"]
# Hitboxes (wiki Template:Data Aatrox/The Darkin Blade and File:Aatrox hitbox areas.png): along the cast direction
# from Aatrox's centre. Main areas use edge range for casts 1-2, centre range for cast 3; sweetspots centre range.
Q1_LEN, Q1_HALF = 625.0, 90.0                         # 625 x 180 rectangle from Aatrox's centre
Q1_SWEET_FROM, Q1_SWEET_TO, Q1_SWEET_HALF = 450.0, 755.0, 186.5     # sweetspot box (image, 1 px = 1 unit)
Q2_BACK, Q2_FRONT, Q2_HALF_BACK, Q2_HALF_FRONT = -100.0, 475.0, 150.0, 250.0   # trapezoid 300 -> 500 wide
Q2_SWEET_FROM, Q2_SWEET_TO = 300.0, 675.0             # sweetspot quad (image), half-width 301 -> 342
Q2_SWEET_HALF_FROM, Q2_SWEET_HALF_TO = 301.0, 342.0
Q3_OFFSET, Q3_RADIUS, Q3_SWEET = 200.0, 300.0, 180.0

W_CAST_TIME = 0.25                                    # client mCastTime
W_SPEED = 1800.0                                      # mMissileSpec FixedSpeedMovement
W_RANGE = 765.0                                       # wiki edge range (centre range 825)
W_HALF_WIDTH = 80.0                                   # mMissileWidth (wiki width 160)
W_SLOW_DURATION = scalar(NAME, "W", "WSlowDuration")  # 1.5
W_TETHER = 1.5                                        # wiki: the tether lasts 1.5 s
W_ZONE = 650.0                                        # escape radius around the hit point (DAT.W3; replay bounds)
W_PULL_TIME = 0.25                                    # replay: every pull takes ~0.25 s whatever its length
W_AIRBORNE = 0.5                                      # replay: the target stays down ~0.25 s after arriving

E_MIN = scalar(NAME, "E", "EMinRange")                # 75
E_MAX = scalar(NAME, "E", "EMaxRange")                # 300
E_SPEED = scalar(NAME, "E", "EDashSpeed")             # 800
E_HEAL = scalar(NAME, "E", "ESpellVamp") / 100.0      # 16% of post-mitigation damage to champions
E_HEAL_PER_HP = scalar(NAME, "E", "EVampHPRatio") / 100.0       # 0.013% per bonus health: 1.3% per 100 (26.19)

R_CAST_TIME = 0.25
R_DURATION = scalar(NAME, "R", "RDuration")           # 10
R_EXTEND = scalar(NAME, "R", "RExtension")            # +5 s per champion takedown, up to 10 remaining
R_FEAR_RADIUS = scalar(NAME, "R", "RFearRadius")      # 600
R_FEAR = scalar(NAME, "R", "RMinionFearDuration")     # 3 s on minions and monsters
R_MS_DECAY = 0.9                                      # wiki: -10% of the current bonus every 0.25 s
R_MS_PERIOD = 0.25
R_SIZE = 0.05                                         # wiki: 5% increased size

Q_FLAGS = D.TAG_AOE | D.TAG_ACTIVE_SPELL
W_FLAGS = D.TAG_ACTIVE_SPELL
P_FLAGS = D.TAG_PROC | D.PROP_LIFESTEAL               # wiki: proc, ApplyLifesteal


class State(NamedTuple):
    p_ready_at: Any         # (C,) Deathbringer Stance ready from this time
    p_armed: Any            # (C,) bool: the attack launched this tick is empowered
    p_cast_id: Any          # (C,) instance of the last bonus-damage packet (its heal is read in on_damage)
    p_minion: Any           # (C,) bool: that packet's target was a minion
    q_stage: Any            # (C,) int32 casts used in the current sequence (0..3)
    q_last: Any             # (C,) start of the last cast
    q_pending: Any
    q_fire_at: Any
    q_index: Any            # (C,) int32 cast being wound up (0, 1, 2)
    q_dx: Any
    q_dy: Any
    q_cast_id: Any
    w_pending: Any
    w_fire_at: Any
    w_dx: Any
    w_dy: Any
    w_cast_id: Any
    w_fly: Any              # (C,) bool: the chain is travelling
    w_ox: Any               # chain origin (Aatrox at the end of the cast time)
    w_oy: Any
    w_dist: Any             # distance travelled
    w_target: Any           # (C,) int32 tethered unit, -1 none
    w_seq: Any
    w_cx: Any               # tether centre (the target at the hit)
    w_cy: Any
    w_check_at: Any         # tether end: pull if still inside
    w_pulled: Any           # (C,) int32 unit being pulled
    w_pull_until: Any
    w_pull_speed: Any
    r_pending: Any
    r_fire_at: Any
    r_cast_id: Any
    r_until: Any            # World Ender end
    r_ms_from: Any          # start of the current move-speed decay
    r_rank: Any
    r_bar: Any              # (C,) remaining World Ender fraction (the resource bar)


def init(n_champions: int, n_units: int) -> State:
    c = n_champions
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    i = jnp.zeros((c,), jnp.int32)
    return State(z - NEVER, f, i, f,
                 i, z - NEVER, f, z + NEVER, i, z, z, i,
                 f, z + NEVER, z, z, i, f, z, z, z, i - 1, i, z, z, z + NEVER, i - 1, z - NEVER, z,
                 f, z + NEVER, i, z - NEVER, z - NEVER, i, z)


def _mine(kctx) -> Any:
    return kctx.champion_id == ID


def by_level(pair, level) -> Any:
    """Client ``ByCharLevelInterpolation``: linear from level 1 to 18."""
    lv = jnp.clip(jnp.asarray(level, jnp.float32), 1.0, 18.0)
    return pair[0] + (pair[1] - pair[0]) * (lv - 1.0) / 17.0


def minion_mult(level) -> Any:
    """The Darkin Blade against minions: 55%, +5% at levels 6, 11 and 16."""
    lv = jnp.asarray(level)
    return Q_MINION_BASE + Q_MINION_STEP * sum((lv >= s).astype(jnp.float32) for s in Q_MINION_STEPS)


def _champ_or_large_monster(units, idx) -> Any:
    """Champions and large monsters (camp large monsters and the epics): Deathbringer cooldown refunds and the
    Infernal Chains tether."""
    return _large(units, idx) & (units.kind[idx] != KIND_MINION)


def q_regions(index, along, perp, radius) -> tuple[Any, Any]:
    """``(hit, sweet)`` of The Darkin Blade cast ``index`` (0, 1, 2) for units at ``along``/``perp`` (signed)
    from Aatrox in the cast direction with gameplay ``radius``."""
    a, p = along, jnp.abs(perp)
    # Cast 1: rectangle [0, 625] x [-90, 90], edge range (exact circle-box test).
    dx1 = jnp.maximum(jnp.maximum(-a, a - Q1_LEN), 0.0)
    dy1 = jnp.maximum(p - Q1_HALF, 0.0)
    hit1 = dx1 * dx1 + dy1 * dy1 <= radius * radius
    sweet1 = (a >= Q1_SWEET_FROM) & (a <= Q1_SWEET_TO) & (p <= Q1_SWEET_HALF)
    # Cast 2: trapezoid [-100, 475], half-width 150 -> 250, edge range (the box grown by the radius).
    hw = Q2_HALF_BACK + (Q2_HALF_FRONT - Q2_HALF_BACK) * (jnp.clip(a, Q2_BACK, Q2_FRONT) - Q2_BACK) \
        / (Q2_FRONT - Q2_BACK)
    hit2 = (a >= Q2_BACK - radius) & (a <= Q2_FRONT + radius) & (p <= hw + radius)
    hw_s = Q2_SWEET_HALF_FROM + (Q2_SWEET_HALF_TO - Q2_SWEET_HALF_FROM) \
        * (jnp.clip(a, Q2_SWEET_FROM, Q2_SWEET_TO) - Q2_SWEET_FROM) / (Q2_SWEET_TO - Q2_SWEET_FROM)
    sweet2 = (a >= Q2_SWEET_FROM) & (a <= Q2_SWEET_TO) & (p <= hw_s)
    # Cast 3: circle of 300 around the point 200 ahead, centre range; sweetspot 180.
    d3 = jnp.sqrt((a - Q3_OFFSET) ** 2 + p * p)
    hit3 = d3 <= Q3_RADIUS
    sweet3 = d3 <= Q3_SWEET
    idx = jnp.asarray(index)[..., None] if jnp.ndim(index) else index
    hit = jnp.where(idx == 0, hit1, jnp.where(idx == 1, hit2, hit3))
    sweet = jnp.where(idx == 0, sweet1, jnp.where(idx == 1, sweet2, sweet3))
    return hit, hit & sweet


def _base_cd(kctx) -> Any:
    return jnp.where(_mine(kctx)[:, None], cooldown_row(NAME, kctx.ranks), 0.0)


def _unit_dir(dx, dy) -> tuple[Any, Any, Any]:
    d = jnp.sqrt(dx * dx + dy * dy)
    ok = d > 1e-3
    return jnp.where(ok, dx / jnp.maximum(d, 1e-3), 1.0), jnp.where(ok, dy / jnp.maximum(d, 1e-3), 0.0), d


def cast(state: State, kctx, units, order) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    r = kctx.ranks
    now = f32(kctx.now)
    free = m & kctx.alive & ~kctx.stunned & ~kctx.silenced
    winding = state.q_pending | state.w_pending | state.r_pending
    rooted = jnp.zeros_like(m) if kctx.rooted is None else kctx.rooted
    want = lambda s: (order.slot == s) & free          # noqa: E731
    ready = (r > 0) & (kctx.cooldowns <= 0)
    ux, uy, dist = _unit_dir(order.x - kctx.x, order.y - kctx.y)

    # The Darkin Blade: three casts; each after 1 s and within 4 s of the previous one.
    open_ = (state.q_stage > 0) & (state.q_stage < 3) & (now <= state.q_last + Q_WINDOW + 1e-6)
    stage = jnp.where(open_, state.q_stage, 0)
    q = want(0) & ~winding & (r[:, 0] > 0) \
        & jnp.where(stage > 0, now >= state.q_last + Q_RECAST_GAP - 1e-6, ready[:, 0])
    # Infernal Chains and World Ender; Umbral Dash also during the other cast times (and they during it).
    w = want(1) & ~winding & ready[:, 1]
    rr = want(3) & ~winding & ready[:, 3]
    e = want(2) & ready[:, 2] & ~rooted & ~(kctx.dashing if kctx.dashing is not None else jnp.zeros_like(m))
    e_len = jnp.clip(dist, E_MIN, E_MAX)
    dash = Dash(e, f32(kctx.x + ux * e_len), f32(kctx.y + uy * e_len), jnp.full((c,), E_SPEED, jnp.float32),
                jnp.full((c,), -1, jnp.int32), jnp.zeros((c,), bool))

    started = q | w | e | rr
    slot = jnp.where(q, 0, jnp.where(w, 1, jnp.where(e, 2, jnp.where(rr, 3, -1)))).astype(jnp.int32)
    cid = jnp.where(started, make_cast_id(kctx, jnp.maximum(slot, 0)), 0).astype(jnp.int32)
    state = state._replace(
        q_stage=jnp.where(q, stage + 1, stage).astype(jnp.int32), q_last=jnp.where(q, now, state.q_last),
        q_pending=state.q_pending | q, q_fire_at=jnp.where(q, later(kctx, Q_CAST_TIME), state.q_fire_at),
        q_index=jnp.where(q, stage, state.q_index).astype(jnp.int32),
        q_dx=jnp.where(q, f32(ux), state.q_dx), q_dy=jnp.where(q, f32(uy), state.q_dy),
        q_cast_id=jnp.where(q, cid, state.q_cast_id),
        w_pending=state.w_pending | w, w_fire_at=jnp.where(w, later(kctx, W_CAST_TIME), state.w_fire_at),
        w_dx=jnp.where(w, f32(ux), state.w_dx), w_dy=jnp.where(w, f32(uy), state.w_dy),
        w_cast_id=jnp.where(w, cid, state.w_cast_id),
        r_pending=state.r_pending | rr, r_fire_at=jnp.where(rr, later(kctx, R_CAST_TIME), state.r_fire_at),
        r_cast_id=jnp.where(rr, cid, state.r_cast_id), r_rank=jnp.where(rr, r[:, 3], state.r_rank))
    # Cooldowns start on cast; every Q cast restarts Q's (wiki cdstart).
    cd = jnp.stack([q, w, e, rr], -1)
    return state, out(c, n, dash=dash, cooldown_start=cd, base_cooldown=_base_cd(kctx), attack_reset=e,
                      cast_started=started, cast_slot=slot, cast_id=cid)


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
    big = _champ_or_large_monster(units, jnp.arange(n))[None, :]
    minion = (units.kind == KIND_MINION)[None, :]
    monster = (units.kind == KIND_MONSTER)[None, :]

    # The Darkin Blade fires at the end of its cast time from where Aatrox is then, in the direction fixed at cast.
    q_due = state.q_pending & (due(kctx, state.q_fire_at) | dead)
    q_fire = q_due & alive
    rx, ry = units.x[None, :] - kctx.x[:, None], units.y[None, :] - kctx.y[:, None]
    along = rx * state.q_dx[:, None] + ry * state.q_dy[:, None]
    perp = ry * state.q_dx[:, None] - rx * state.q_dy[:, None]
    hit, sweet = q_regions(state.q_index, along, perp, jnp.broadcast_to(units.radius[None, :], (c, n)))
    hit = hit & foe & q_fire[:, None]
    sweet = sweet & hit
    mult = (1.0 + Q_RAMP * state.q_index.astype(jnp.float32))[:, None] * jnp.where(sweet, 1.0 + Q_SWEET, 1.0)
    q_base = ranked(NAME, "Q", "QBaseDamage", r[:, 0]) + ranked(NAME, "Q", "QTotalADRatio", r[:, 0]) * kctx.total_ad
    q_raw = (q_base[:, None] + jnp.where(monster, Q_MONSTER_BONUS, 0.0)) * mult \
        * jnp.where(minion, minion_mult(kctx.level)[:, None], 1.0)
    p_q = D.packets(hit, kctx.unit[:, None], cols, q_raw, D.PHYSICAL, Q_FLAGS, cast_id=state.q_cast_id[:, None])
    q_up = jnp.where(sweet, Q_KNOCKUP * jnp.where(monster, 2.0, 1.0), 0.0)
    q_cdr = jnp.where(jnp.any(sweet & big, axis=1), 2.0 * P_CDR, jnp.where(jnp.any(hit & big, axis=1), P_CDR, 0.0))
    last_q = q_fire & (state.q_index >= 2)
    expired = (state.q_stage > 0) & (now > state.q_last + Q_WINDOW + 1e-6) & ~state.q_pending
    q_stage = jnp.where(last_q | expired | dead, 0, state.q_stage)

    # Infernal Chains: the chain leaves at the end of the cast time and stops on the first enemy it reaches.
    w_due = state.w_pending & (due(kctx, state.w_fire_at) | dead)
    w_go = w_due & alive
    w_fly = (state.w_fly | w_go) & alive
    ox, oy = jnp.where(w_go, kctx.x, state.w_ox), jnp.where(w_go, kctx.y, state.w_oy)
    d0 = jnp.where(w_go, 0.0, state.w_dist)
    d1 = d0 + W_SPEED * kctx.dt
    wx, wy = units.x[None, :] - ox[:, None], units.y[None, :] - oy[:, None]
    w_along = wx * state.w_dx[:, None] + wy * state.w_dy[:, None]
    w_perp = jnp.abs(wy * state.w_dx[:, None] - wx * state.w_dy[:, None])
    rad = units.radius[None, :]
    reach = w_fly[:, None] & foe & (w_along - rad <= jnp.minimum(d1, W_RANGE)[:, None]) & (w_along + rad >= d0[:, None]) \
        & (w_perp <= W_HALF_WIDTH + rad)
    first = jnp.argmin(jnp.where(reach, w_along, jnp.inf), axis=1)
    w_hit = jnp.any(reach, axis=1)
    w_mask = w_hit[:, None] & (cols == first[:, None])
    w_raw = ranked(NAME, "W", "WBaseDamage", r[:, 1]) + ranked(NAME, "W", "WTotalADRatio", r[:, 1]) * kctx.total_ad
    p_w = D.packets(w_mask, kctx.unit[:, None], cols, w_raw[:, None] * jnp.where(minion, 2.0, 1.0), D.PHYSICAL,
                    W_FLAGS, cast_id=state.w_cast_id[:, None])
    w_slow = -ranked(NAME, "W", "WSlowPercentage", r[:, 1])
    slow1, sid1 = cc_matrix(w_mask, w_slow[:, None], state.w_cast_id[:, None])
    tether = w_hit & big[0, first]
    w_done = w_hit | (d1 >= W_RANGE)
    ft = jnp.clip(first, 0, n - 1)
    w_target = jnp.where(tether, first, state.w_target).astype(jnp.int32)
    w_seq = jnp.where(tether, units.spawn_seq[ft], state.w_seq).astype(jnp.int32)
    w_cx = jnp.where(tether, units.x[ft], state.w_cx)
    w_cy = jnp.where(tether, units.y[ft], state.w_cy)
    w_check_at = jnp.where(tether, f32(now + W_TETHER), state.w_check_at)
    w_cdr = jnp.where(w_hit & big[0, first], P_CDR, 0.0)

    # The tether: still inside at its end -> the damage again and a pull back to the centre.
    tt = jnp.clip(state.w_target, 0, n - 1)
    held = (state.w_target >= 0) & units.alive[tt] & (units.spawn_seq[tt] == state.w_seq) & alive
    t_end = (state.w_target >= 0) & (due(kctx, state.w_check_at) | ~held)
    off = jnp.sqrt((units.x[tt] - state.w_cx) ** 2 + (units.y[tt] - state.w_cy) ** 2)
    pull = t_end & held & due(kctx, state.w_check_at) & (off <= W_ZONE)
    pull_mask = pull[:, None] & (cols == tt[:, None])
    w2_id = make_cast_id(kctx, CODE_AATROX_W2)
    p_w2 = D.packets(pull_mask, kctx.unit[:, None], cols, w_raw[:, None], D.PHYSICAL, W_FLAGS, cast_id=w2_id[:, None])
    up_w, _ = cc_matrix(pull_mask, W_AIRBORNE, w2_id[:, None])
    w_cdr = w_cdr + jnp.where(pull & _champ_or_large_monster(units, tt), P_CDR, 0.0)
    w_target = jnp.where(t_end, -1, w_target).astype(jnp.int32)
    w_pull_until = jnp.where(pull, f32(now + kctx.dt + W_PULL_TIME), state.w_pull_until)
    w_pull_speed = jnp.where(pull, off / W_PULL_TIME, state.w_pull_speed)

    # World Ender: at the end of the cast time; fear (as a stun, Aatrox limitation 6) on nearby minions and monsters.
    r_due = state.r_pending & (due(kctx, state.r_fire_at) | dead)
    r_fire = r_due & alive
    dist = jnp.sqrt(rx * rx + ry * ry)
    feared = r_fire[:, None] & foe & ((units.kind == KIND_MINION) | (units.kind == KIND_MONSTER))[None, :] \
        & (dist <= R_FEAR_RADIUS + rad)
    fear, fid = cc_matrix(feared, R_FEAR, state.r_cast_id[:, None])
    r_until = jnp.where(r_fire, f32(now + R_DURATION), jnp.where(dead, -NEVER, state.r_until))

    # Deathbringer Stance refunds (current cooldown) from this tick's ability hits.
    p_ready_at = jnp.where(m, state.p_ready_at - q_cdr - w_cdr, state.p_ready_at)

    state = state._replace(
        p_ready_at=f32(p_ready_at), q_stage=q_stage.astype(jnp.int32), q_pending=state.q_pending & ~q_due,
        w_pending=state.w_pending & ~w_due, w_fly=w_fly & ~w_done, w_ox=f32(ox), w_oy=f32(oy), w_dist=f32(d1),
        w_target=w_target, w_seq=w_seq, w_cx=f32(w_cx), w_cy=f32(w_cy), w_check_at=f32(w_check_at),
        w_pulled=jnp.where(pull, tt, state.w_pulled).astype(jnp.int32), w_pull_until=f32(w_pull_until),
        w_pull_speed=f32(w_pull_speed), r_pending=state.r_pending & ~r_due, r_until=f32(r_until),
        r_ms_from=jnp.where(r_fire, now, state.r_ms_from),
        r_bar=f32(jnp.where(m, jnp.clip((r_until - now - kctx.dt) / R_DURATION, 0.0, 1.0), 0.0)))
    cc = CCOut(fear, z, z, f32(q_up + up_w), f32(slow1), jnp.where(w_mask, W_SLOW_DURATION, 0.0),
               jnp.where(hit, state.q_cast_id[:, None], jnp.where(w_mask, sid1, jnp.where(pull_mask, w2_id[:, None],
                                                                                        fid))).astype(jnp.int32))
    return state, out(c, n, packets=D.concat_packets(p_q, p_w, p_w2), cc=cc, base_cooldown=_base_cd(kctx))


def on_attack(state: State, kctx, units, launch) -> tuple[State, KitOut]:
    """Deathbringer Stance arms the attack launched while it is ready (not against structures or wards)."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    target = holder_rows(launch.target, kctx)
    t = jnp.clip(target, 0, n - 1)
    went = holder_rows(launch.launched, kctx) & _mine(kctx) & kctx.alive & (target >= 0)
    ok = ~is_structure(units.kind[t]) & (units.kind[t] != KIND_WARD)
    armed = went & ok & (kctx.now >= state.p_ready_at - 1e-6)
    cool = by_level(P_COOLDOWN, kctx.level)
    state = state._replace(p_armed=armed, p_ready_at=jnp.where(armed, f32(kctx.now + cool), state.p_ready_at))
    return state, out(c, n, base_cooldown=_base_cd(kctx))


def on_hit(state: State, kctx, units, launch, dodging) -> tuple[State, KitOut]:
    """The empowered attack's bonus magic damage (max-health %, capped against monsters); attack hits on champions
    and large monsters refund 2 s of Deathbringer Stance."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    target = holder_rows(launch.target, kctx)
    t = jnp.clip(target, 0, n - 1)
    hit = holder_rows(launch.launched, kctx) & m & kctx.alive & (target >= 0)
    land = hit & ~dodging[t]
    emp = land & state.p_armed
    raw = by_level(P_DAMAGE, kctx.level) * units.max_hp[t]
    raw = jnp.where(units.kind[t] == KIND_MONSTER, jnp.minimum(raw, by_level(P_MONSTER_CAP, kctx.level)), raw)
    pid = make_cast_id(kctx, CODE_AATROX_P)
    p = D.packets(emp, kctx.unit, t, raw, D.MAGIC, P_FLAGS, cast_id=pid)
    refund = land & ~state.p_armed & _champ_or_large_monster(units, t)
    state = state._replace(p_armed=state.p_armed & ~hit,
                           p_cast_id=jnp.where(emp, pid, state.p_cast_id).astype(jnp.int32),
                           p_minion=jnp.where(emp, units.kind[t] == KIND_MINION, state.p_minion),
                           p_ready_at=jnp.where(refund, state.p_ready_at - P_CDR, state.p_ready_at))
    return state, out(c, n, packets=p, base_cooldown=_base_cd(kctx))


def on_damage(state: State, kctx, units, report) -> tuple[State, KitOut]:
    """Deathbringer Stance heals its post-mitigation bonus damage (25% against minions); Umbral Dash's passive heals
    16% (+1.3% per 100 bonus health) of non-persistent post-mitigation damage dealt to enemy champions."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    p = report.packets
    res = report.resolved                                             # partial reports (tests) fall back
    dealt = getattr(res, "final", None)
    dealt = getattr(res, "health_loss", p.raw) if dealt is None else dealt
    final = jnp.where(p.valid, dealt, 0.0)
    src = (p.src[None, :] == kctx.unit[:, None]) & _mine(kctx)[:, None]
    dst = jnp.clip(p.dst, 0, n - 1)
    from_p = src & (p.cast_id[None, :] == state.p_cast_id[:, None]) & (state.p_cast_id[:, None] != 0)
    p_heal = jnp.sum(jnp.where(from_p, final[None, :], 0.0), axis=1) * P_HEAL \
        * jnp.where(state.p_minion, P_HEAL_MINION, 1.0)
    persistent = ((p.flags & (D.TAG_PERIODIC | D.TAG_BURN)) != 0)[None, :]
    to_champ = src & (units.kind[dst] == KIND_CHAMPION)[None, :] & (units.team[dst][None, :] != kctx.team[:, None]) \
        & ~persistent
    e_rate = jnp.where(kctx.ranks[:, 2] > 0, E_HEAL + E_HEAL_PER_HP * kctx.bonus_hp, 0.0)
    e_heal = jnp.sum(jnp.where(to_champ, final[None, :], 0.0), axis=1) * e_rate
    heal = jnp.where(_mine(kctx) & kctx.alive, p_heal + e_heal, 0.0)
    return state._replace(p_cast_id=jnp.zeros_like(state.p_cast_id)), out(c, n, heal=f32(heal))


def on_takedown(state: State, kctx, units, kills) -> State:
    """World Ender: each champion takedown adds 5 s, up to its original 10 s remaining, and refreshes the move
    speed."""
    took = _mine(kctx) & ((jnp.asarray(kills.champion_kill) + jnp.asarray(kills.champion_assist)) > 0)
    on = took & (kctx.now < state.r_until)
    now = f32(kctx.now)
    until = jnp.minimum(state.r_until + R_EXTEND, now + R_DURATION)
    return state._replace(r_until=jnp.where(on, f32(until), state.r_until),
                          r_ms_from=jnp.where(on, now, state.r_ms_from))


def r_active(state: State, kctx) -> Any:
    return _mine(kctx) & kctx.alive & (kctx.now < state.r_until)


def stats(state: State, kctx) -> ItemStats:
    """World Ender: +20/30/40% of total attack damage, decaying bonus move speed and +50/75/100% self-healing."""
    on = r_active(state, kctx)
    rk = state.r_rank
    steps = jnp.floor(jnp.maximum(kctx.now - state.r_ms_from, 0.0) / R_MS_PERIOD + 1e-6)
    ms = ranked(NAME, "R", "RMovementSpeedBonus", rk) * R_MS_DECAY ** steps
    return ItemStats(attack_damage=f32(jnp.where(on, ranked(NAME, "R", "RTotalADAmp", rk) * kctx.total_ad, 0.0)),
                     percent_move_speed=f32(jnp.where(on, ms, 0.0)),
                     incoming_heal=f32(jnp.where(on, ranked(NAME, "R", "RHealingAmp", rk), 0.0)))


def defense(state: State, kctx) -> KitDefense:
    c = kctx.unit.shape[0]
    o = jnp.ones((c,), jnp.float32)
    return KitDefense(o, jnp.zeros((c,), bool), o, jnp.zeros((c,), jnp.float32))


def attack_mods(state: State, kctx) -> KitAttackMods:
    """+50 range while Deathbringer Stance is ready; no attacks through the Q, W and R cast times."""
    m = _mine(kctx)
    c = kctx.unit.shape[0]
    z = jnp.zeros((c,), jnp.float32)
    ready = m & (kctx.now >= state.p_ready_at - 1e-6)
    winding = m & (state.q_pending | state.w_pending | state.r_pending)
    return KitAttackMods(f32(jnp.where(ready, P_RANGE, 0.0)), jnp.zeros((c,), bool), winding,
                         jnp.zeros((c,), bool), z, z, jnp.zeros((c,), bool))


def immobile(state: State, kctx) -> Any:
    """(C,) Aatrox cannot walk during the Q and W cast times (Umbral Dash still moves him)."""
    return _mine(kctx) & (state.q_pending | state.w_pending)


def ghosted(state: State, kctx) -> Any:
    return r_active(state, kctx)


def size(state: State, kctx) -> Any:
    return f32(jnp.where(r_active(state, kctx), R_SIZE, 0.0))


def displacement(state: State, kctx, n_units: int) -> tuple[Any, Any, Any, Any]:
    """(N,) Infernal Chains' pull to the tether centre over 0.25 s."""
    on = _mine(kctx) & (kctx.now < state.w_pull_until) & (state.w_pulled >= 0)
    tgt = jnp.where(on, state.w_pulled, n_units)                     # row n_units collects the idle holders
    put = lambda v, z: z.at[tgt].set(v)[:n_units]                    # noqa: E731
    zf = jnp.zeros((n_units + 1,), jnp.float32)
    return put(on, jnp.zeros((n_units + 1,), bool)), put(state.w_cx, zf), put(state.w_cy, zf), \
        put(state.w_pull_speed, zf)


def resource(state: State, champion_id: Any) -> Any:
    """(C,) the bar shows World Ender's remaining duration (wiki); NaN where the holder is another kit."""
    return jnp.where(champion_id == ID, state.r_bar, jnp.nan)


def debuffs(state: State, kctx, units) -> Debuffs:
    return neutral_debuffs(units.x.shape[0])


__all__ = ["State", "init", "cast", "periodic", "on_attack", "on_hit", "on_damage", "on_takedown", "stats",
           "defense", "attack_mods", "immobile", "ghosted", "size", "displacement", "resource", "debuffs",
           "q_regions", "by_level", "minion_mult"]
