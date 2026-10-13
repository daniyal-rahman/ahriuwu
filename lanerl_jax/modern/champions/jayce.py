"""Jayce (126), 26.19. Rules JAY.* and their evidence: docs/modern/CHAMPIONS.md; numbers from the pinned client
JSON (``modern/data/26.19/champions``; the Mercury Cannon spells are pinned as ``Q2``/``W2``/``E2``/``R2``) and, where
marked, the 16.9 replay measurements (/mnt/nfs/shared/champ-expansion/jayce-measurements).

Jayce starts in Mercury Hammer stance with Transform at rank 1 (``INNATE_RANKS``); Q/W/E have 6 ranks shared by both
stances. Each stance keeps its own Q/W/E cooldowns: a transform swaps the world's slot cooldowns with the inactive
stance's (``KitOut.swap_cooldowns``), and Hyper Charge's post-effect cooldown can start while it is the inactive
stance's (``alt_cooldown_start``). Mercury Cannon is ranged: +375 attack range (``ItemStats.attack_range``) and 2000
units/s attack missiles (``attack_missile_speed``); Mercury Hammer is melee."""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..core import damage as D
from ..core.types import KIND_CHAMPION, KIND_MONSTER, KIND_NONE, KIND_WARD, CCOut, Dash
from ..data.champions import cooldowns, max_ranks, spell
from ..items.catalog import ItemStats
from ..items.effects.core import Debuffs, neutral_debuffs
from ..runes.catalog import ChampionTraits
from .core import (NEVER, KitAttackMods, KitDefense, KitOut, due, enemies, f32, gather, holder_rows, is_structure,
                   later, make_cast_id, out, ranked, scalar, target_dist, within_edge)

NAME = "Jayce"
ID = 126
# Q > W > E over 18 points; each rank at its earliest level (Q at 1/3/5/7/9/11). Transform is never ranked.
SKILL_ORDER = (0, 1, 0, 2, 0, 1, 0, 1, 0, 1, 0, 1, 1, 2, 2, 2, 2, 2)
MAX_RANKS = max_ranks(NAME)            # client rank-up table: Q/W/E at levels 1/3/5/7/9/11; R has no rank-ups
INNATE_RANKS = (0, 0, 0, 1)            # Transform from level 1 without a skill point
TRAITS = ChampionTraits(has_immobilize=True, resource="mana", special="Jayce")   # E roots; Axiom Arcanist -> Nimbus
RADIUS = 65.0                          # no overrideGameplayCollisionRadius on the record

# Hextech Capacitor.
P_MS = scalar(NAME, "Passive", "FlatMovementSpeed")                    # 30 (PATCH 26.14)
P_TIME = scalar(NAME, "Passive", "MovementSpeedDuration")              # 0.75 s, and ghosted

# Mercury Hammer: To the Skies!
Q_RANGE = 600.0                        # castRange (center to edge)
_QC = spell(NAME, "Q")["calculations"]["Damage"]["mFormulaParts"]
Q_AD = _QC[1]["mCoefficient"]                                          # 1.35 bonus AD
Q_SLOW_TIME = scalar(NAME, "Q", "SlowDuration")                        # 2 s
Q_MONSTER = scalar(NAME, "Q", "MonsterBonusDamage")                    # +10 vs monsters
Q_RADIUS = 300.0                       # wiki: damage radius 300 around Jayce where he lands (cr)
Q_STOP = 225.0                         # lands 225 short of the target's center (castRadiusSecondary; replay median 230)
Q_LEAP_TIME = 0.5                      # wiki: over 0.5 s, whatever the length (replay: 300-516 units in 0.5-0.6 s)
Q_LOCK_AFTER_E = 0.4                   # wiki: no To the Skies! / Shock Blast for 0.4 s after Thundering Blow's cast

# Lightning Field.
W_RADIUS = 285.0 + RADIUS              # castRange 285 from Jayce's edge = wiki 350 (cr)
W_TIME = scalar(NAME, "W", "Duration")                                 # 4 s
W_TICKS = 4                            # wiki: magic damage every second for 4 s
W_FIRST = 1.0                          # the first tick 1 s after the cast (INFERRED: 4 ticks over the 4 s field)
W_AP = spell(NAME, "W")["calculations"]["Damage"]["mFormulaParts"][1]["mCoefficient"]   # 1.0 AP over the 4 ticks

# Thundering Blow.
E_RANGE = 240.0 + RADIUS               # castRange 240 edge to edge (castRangeUseBoundingBoxes) -> center to edge
E_CAST = 0.25                          # mCastTime; the target is rooted over it (PATCH 14.4)
E_KNOCK_NEAR = 475.0                   # replay (46 knockbacks): pushed away to max(475, distance + 250) from Jayce's
E_KNOCK_MIN = 250.0                    # center; KnockbackDistance 600 / wiki 600 would double the displacement
E_KNOCK_TIME = scalar(NAME, "E", "KnockbackDuration")                  # 0.35 s airborne (replay 0.35)

# Transform: Mercury Cannon (R) / Mercury Hammer (R2).
R_RANGE_UP = scalar(NAME, "R", "RangedFormRangeIncrease")              # +375 -> 500 attack range
R_SHRED_TIME = scalar(NAME, "R", "ShredDuration")                      # 5 s
R_HIT_AD = scalar(NAME, "R", "ADRatio")                                # Mercury Hammer's hit: 0.3 bonus AD
_RC = spell(NAME, "R")["calculations"]
R_RES_AD = _RC["Resists"]["mFormulaParts"][1]["mCoefficient"]          # 7.5% bonus AD as armor and MR


def _breakpoints(calc: dict) -> tuple[float, tuple]:
    part = calc["mFormulaParts"][0]
    return float(part["mLevel1Value"]), tuple((int(b["mLevel"]), float(b["mAdditionalBonusAtThisLevel"]))
                                              for b in part["mBreakpoints"])


R_RES = _breakpoints(_RC["Resists"])                                   # 5 (+7 at 6, 11, 16) armor and MR (26.14)
R_HIT = _breakpoints(_RC["Damage"])                                    # 25 (+35 at 6, 11, 16) magic on hit
R_SHRED = _breakpoints(_RC["RangedFormShred"])                         # 20% (+5% at 6, 11, 16) armor and MR

# Mercury Cannon: Shock Blast.
Q2_CAST = 0.2143                       # JayceShockBlast mCastTime
Q2_SPEED = 1450.0                      # JayceShockBlastMis movement mSpeed
Q2_SPEED_UP = 2350.0                   # JayceShockBlastWallMis movement mSpeed (through the gate)
Q2_RANGE = 1050.0                      # castRangeDisplayOverride = wiki (replay: ungated hits <= 944 flown; not 1300)
Q2_RANGE_UP = 1600.0                   # wiki total range through the gate (replay: <= ~1650 flown; not WallMis 2000)
Q2_HALF = 70.0                         # mMissileWidth (half-width; wiki width 140)
Q2_BLAST = 170.0                       # wiki blast radius 170 (cr; the missile's castRadius is 210)
Q2_BLAST_UP = 250.0                    # wiki 250 (cr)
Q2_AD = scalar(NAME, "Q2", "ADRatio")                                  # 1.3 bonus AD
Q2_UP_MULT = scalar(NAME, "Q2", "EmpowerMultiplier")                   # x1.4 through the gate
Q2_MONSTER = scalar(NAME, "Q2", "MonsterBonusDamage")                  # +10 vs monsters

# Hyper Charge.
W2_HITS = int(scalar(NAME, "W2", "NumAttacks"))                        # 3 attacks ...
W2_TIME = scalar(NAME, "W2", "Duration")                               # ... within 3.6 s (wiki 4)
W2_AS = scalar(NAME, "W2", "PercentIncreasedAS")                       # +360% bonus attack speed (PATCH 25.04)
W2_MISSILE = 1800.0                    # JayceHyperChargeRangedAttack movement mSpeed
ATTACK_MISSILE = 2000.0                # JayceRangedAttack movement mSpeed

# Acceleration Gate.
E2_RANGE = 685.0                       # castRangeDisplayOverride (cast at max beyond it; wiki 650 since 2017)
E2_TIME = scalar(NAME, "E2", "Duration")                               # 4 s
E2_HALF_LEN = scalar(NAME, "E2", "GateWidth") / 2.0                    # 750 wide, across the cast direction
E2_THICK = 100.0                       # castRadius 100 (wiki collision radius 100)
E2_DECAY = scalar(NAME, "E2", "HasteDuration")                         # the haste decays over 3 s

SPELL = D.TAG_ACTIVE_SPELL
AOE = D.TAG_AOE | D.TAG_ACTIVE_SPELL
DOT = D.TAG_AOE | D.TAG_ACTIVE_SPELL | D.TAG_PERIODIC
HIT_FLAGS = D.TAG_ON_HIT | D.TAG_PROC
CODE_Q, CODE_W, CODE_E, CODE_R = 0, 1, 2, 3


class State(NamedTuple):
    cannon: Any             # (C,) bool: Mercury Cannon stance
    hammer_hit: Any         # (C,) bool: the next landed attack deals Mercury Hammer's magic damage
    shred_ready: Any        # (C,) bool: the next attack shreds (Mercury Cannon)
    shred_flying: Any       # (C,) bool: a launched attack will shred when it lands
    shred_on: Any           # int32 unit the shred is on
    shred_until: Any
    shred_pct: Any
    p_until: Any            # Hextech Capacitor haste and ghosting
    q_pending: Any          # (C,) leaping (To the Skies!)
    q_land_at: Any
    q_target: Any
    q_seq: Any
    q_cast_id: Any
    q_lock_until: Any       # no Q in either stance (after Thundering Blow)
    q_lx: Any               # the leap's landing point
    q_ly: Any
    w_until: Any            # Lightning Field
    w_next: Any             # next tick time
    w_left: Any             # int32 ticks left
    w_cast_id: Any
    e_pending: Any          # Thundering Blow cast time
    e_fire_at: Any
    e_target: Any
    e_seq: Any
    e_cast_id: Any
    kb_until: Any           # (C,) knockback in progress on ``e_target``
    kb_x: Any
    kb_y: Any
    kb_speed: Any
    q2_pending: Any         # Shock Blast cast time
    q2_fire_at: Any
    q2_dx: Any
    q2_dy: Any
    q2_cast_id: Any
    m_fly: Any              # (C,) the Shock Blast missile is travelling
    m_ox: Any
    m_oy: Any
    m_dx: Any
    m_dy: Any
    m_dist: Any
    m_up: Any               # (C,) bool: supercharged by the gate
    w2_left: Any            # int32 Hyper Charge attacks left
    w2_until: Any
    w2_cast_id: Any
    gate_until: Any         # Acceleration Gate
    gate_x: Any
    gate_y: Any
    gate_nx: Any            # unit normal (Jayce's cast direction)
    gate_ny: Any
    gate_rank: Any          # int32 E rank at the cast
    haste_until: Any        # the gate's haste: full while inside, then decays to 0 at ``haste_until``
    r_at: Any               # time of the last transform (the stance at the start of that tick is the other one)


def init(n_champions: int, n_units: int) -> State:
    c = n_champions
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    i = jnp.zeros((c,), jnp.int32)
    return State(cannon=f, hammer_hit=f, shred_ready=f, shred_flying=f, shred_on=i - 1, shred_until=z - NEVER, shred_pct=z, p_until=z - NEVER,
                 q_pending=f, q_land_at=z + NEVER, q_target=i - 1, q_seq=i, q_cast_id=i, q_lock_until=z - NEVER,
                 q_lx=z, q_ly=z,
                 w_until=z - NEVER, w_next=z + NEVER, w_left=i, w_cast_id=i,
                 e_pending=f, e_fire_at=z + NEVER, e_target=i - 1, e_seq=i, e_cast_id=i,
                 kb_until=z - NEVER, kb_x=z, kb_y=z, kb_speed=z,
                 q2_pending=f, q2_fire_at=z + NEVER, q2_dx=z, q2_dy=z, q2_cast_id=i,
                 m_fly=f, m_ox=z, m_oy=z, m_dx=z, m_dy=z, m_dist=z, m_up=f,
                 w2_left=i, w2_until=z - NEVER, w2_cast_id=i,
                 gate_until=z - NEVER, gate_x=z, gate_y=z, gate_nx=z, gate_ny=z, gate_rank=i, haste_until=z - NEVER,
                 r_at=z - NEVER)


def _mine(kctx) -> Any:
    return kctx.champion_id == ID


def by_level(table: tuple, level: Any) -> Any:
    """Level-1 value plus each breakpoint bonus reached (``ByCharLevelBreakpointsCalculationPart``)."""
    base, steps = table
    lv = jnp.asarray(level, jnp.float32)
    return f32(base + sum(jnp.where(lv >= at, add, 0.0) for at, add in steps))


def _row(kind: str, slots: str, ranks: Any) -> Any:
    """(C, 4) cooldowns (``kind="cooldown"``) or mana costs of the stance's spells at the holder's ranks."""
    cols = []
    for i, slot in enumerate(slots.split()):
        mana = spell(NAME, slot).get("mana")
        if kind == "mana" and mana is None:
            cols.append(jnp.zeros(ranks.shape[:1], jnp.float32))
            continue
        table = jnp.asarray(cooldowns(NAME, slot) if kind == "cooldown" else mana["values"], jnp.float32)
        cols.append(table[jnp.clip(ranks[:, i] - 1, 0, table.shape[0] - 1)])
    return jnp.stack(cols, -1)


HAMMER, CANNON = "Q W E R", "Q2 W2 E2 R2"


def cooldown_rows(ranks: Any) -> tuple[Any, Any]:
    """(C, 4) base cooldowns of Mercury Hammer's and Mercury Cannon's spells."""
    return _row("cooldown", HAMMER, ranks), _row("cooldown", CANNON, ranks)


def mana_rows(ranks: Any) -> tuple[Any, Any]:
    return _row("mana", HAMMER, ranks), _row("mana", CANNON, ranks)


def _base_cd(state: State, kctx) -> Any:
    """(C, 4) base cooldowns of the stance in use (0 for other champions)."""
    ham, can = cooldown_rows(kctx.ranks)
    return jnp.where(_mine(kctx)[:, None], jnp.where(state.cannon[:, None], can, ham), 0.0)


def _alt_cd(state: State, kctx) -> Any:
    """(C, 3) base Q/W/E cooldowns of the inactive stance."""
    ham, can = cooldown_rows(kctx.ranks)
    return jnp.where(_mine(kctx)[:, None], jnp.where(state.cannon[:, None], ham, can), 0.0)[:, :3]


def gate_haste(state: State, kctx) -> Any:
    """(C,) the gate's bonus move speed: full until 3 s before ``haste_until``, then decaying linearly to 0."""
    left = jnp.clip((state.haste_until - kctx.now) / E2_DECAY, 0.0, 1.0)
    return f32(ranked(NAME, "E2", "MovementSpeed", state.gate_rank) * left)


def _gate_cross(state: State, ox, oy, dx, dy, d0, d1) -> tuple[Any, Any]:
    """Does the path ``o + u * s`` for ``s`` in [d0, d1] enter an active gate (a 750 x 200 band)? Returns
    ``(crossed, s)`` with the first such ``s``."""
    gx, gy = ox - state.gate_x, oy - state.gate_y
    nx, ny = state.gate_nx, state.gate_ny
    h0 = gx * nx + gy * ny                         # signed distance across the gate at s = 0
    l0 = -gx * ny + gy * nx                        # offset along the gate
    hu = dx * nx + dy * ny
    lu = -dx * ny + dy * nx

    def band(v0, vu, half):
        safe = jnp.where(jnp.abs(vu) > 1e-6, vu, 1e-6)
        a, b = (-half - v0) / safe, (half - v0) / safe
        lo, hi = jnp.minimum(a, b), jnp.maximum(a, b)
        inside = jnp.abs(v0) <= half
        flat = jnp.abs(vu) <= 1e-6
        return jnp.where(flat, jnp.where(inside, -jnp.inf, jnp.inf), lo), jnp.where(flat, jnp.where(inside, jnp.inf,
                                                                                                    -jnp.inf), hi)
    h_lo, h_hi = band(h0, hu, E2_THICK)
    l_lo, l_hi = band(l0, lu, E2_HALF_LEN)
    lo = jnp.maximum(jnp.maximum(h_lo, l_lo), d0)
    hi = jnp.minimum(jnp.minimum(h_hi, l_hi), d1)
    return lo <= hi, lo


def stance_at_tick_start(state: State, kctx) -> Any:
    """(C,) bool: Mercury Cannon at the start of this tick (a transform this tick toggled ``cannon``)."""
    return jnp.where(jnp.abs(state.r_at - kctx.now) < 0.5 * kctx.dt, ~state.cannon, state.cannon)


def _foe_target(kctx, units, target, reach) -> Any:
    """(C,) ``target`` is a living, targetable enemy unit (not a structure or ward) within ``reach`` of the holder's
    center (to its edge)."""
    n = units.x.shape[0]
    t = jnp.clip(target, 0, n - 1)
    return (target >= 0) & (target != kctx.unit) & units.alive[t] & units.targetable[t] \
        & (units.team[t] != kctx.team) & (units.kind[t] != KIND_NONE) & (units.kind[t] != KIND_WARD) \
        & ~is_structure(units.kind[t]) & (target_dist(kctx, units, target) <= reach + units.radius[t])


def _aim(kctx, px, py) -> tuple[Any, Any, Any]:
    """Unit direction from the holder to ``(px, py)`` and the distance (0 -> no direction)."""
    dx, dy = px - kctx.x, py - kctx.y
    d = jnp.sqrt(dx * dx + dy * dy)
    ok = d > 1e-3
    s = jnp.where(ok, d, 1.0)
    return f32(jnp.where(ok, dx / s, 0.0)), f32(jnp.where(ok, dy / s, 0.0)), d


def cast(state: State, kctx, units, order) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    j = _mine(kctx)
    r = kctx.ranks
    now = f32(kctx.now)
    cannon = state.cannon
    busy = state.q_pending | state.e_pending | state.q2_pending
    free = j & kctx.alive & ~kctx.stunned & ~kctx.silenced
    ham_cost, can_cost = mana_rows(r)
    cost = jnp.where(cannon[:, None], can_cost, ham_cost)
    ready = (r > 0) & (kctx.cooldowns <= 0) & (kctx.mana[:, None] >= cost)
    want = lambda s: (order.slot == s) & free          # noqa: E731
    rooted = jnp.zeros_like(j) if kctx.rooted is None else kctx.rooted
    q_ok = now >= state.q_lock_until

    # Mercury Hammer. Lightning Field can also be cast during the leap (wiki).
    q = want(0) & ~cannon & ~busy & ready[:, 0] & q_ok & ~rooted & _foe_target(kctx, units, order.target, Q_RANGE)
    w = want(1) & ~cannon & ~state.e_pending & ready[:, 1]
    e = want(2) & ~cannon & ~busy & ready[:, 2] & _foe_target(kctx, units, order.target, E_RANGE)
    # Mercury Cannon.
    ax, ay, ad = _aim(kctx, order.x, order.y)
    q2 = want(0) & cannon & ~busy & ready[:, 0] & q_ok & (ad > 1e-3)
    w2 = want(1) & cannon & ~busy & ready[:, 1]
    e2 = want(2) & cannon & ~busy & ready[:, 2] & (ad > 1e-3)
    # Transform (Transform is not an ability activation for on-cast effects: no ``cast_started``).
    rr = want(3) & ~busy & ready[:, 3]

    started = q | w | e | q2 | w2 | e2
    slot = jnp.where(q | q2, 0, jnp.where(w | w2, 1, jnp.where(e | e2, 2, -1))).astype(jnp.int32)
    cid = jnp.where(started | rr, make_cast_id(kctx, jnp.where(rr, CODE_R, jnp.maximum(slot, 0))), 0).astype(jnp.int32)
    t = jnp.clip(order.target, 0, n - 1)
    dist = target_dist(kctx, units, order.target)

    # Thundering Blow roots its target over the cast time.
    root, rid = jnp.zeros((c, n), jnp.float32), jnp.zeros((c, n), jnp.int32)
    e_mask = e[:, None] & (jnp.arange(n)[None, :] == t[:, None])
    root = jnp.where(e_mask, E_CAST, root)
    rid = jnp.where(e_mask, cid[:, None], rid)
    # To the Skies!: a straight leap to 225 short of the target's center, over 0.5 s.
    lx_, ly_, _ = _aim(kctx, units.x[t], units.y[t])
    leap = jnp.maximum(dist - Q_STOP, 0.0)
    q_lx, q_ly = f32(kctx.x + lx_ * leap), f32(kctx.y + ly_ * leap)
    # Acceleration Gate: centered at the point, clamped to its range, across the cast direction.
    g_len = jnp.minimum(ad, E2_RANGE)
    to_cannon = rr & ~cannon
    to_hammer = rr & cannon
    state = state._replace(
        cannon=cannon ^ rr, r_at=jnp.where(rr, now, state.r_at),
        hammer_hit=(state.hammer_hit & ~rr) | to_hammer, shred_ready=(state.shred_ready & ~rr) | to_cannon,
        p_until=jnp.where(rr, f32(now + P_TIME), state.p_until),
        q_pending=state.q_pending | q,
        q_land_at=jnp.where(q, f32(now + Q_LEAP_TIME), state.q_land_at),
        q_target=jnp.where(q, order.target, state.q_target).astype(jnp.int32),
        q_seq=jnp.where(q, units.spawn_seq[t], state.q_seq).astype(jnp.int32),
        q_cast_id=jnp.where(q, cid, state.q_cast_id),
        q_lx=jnp.where(q, q_lx, state.q_lx), q_ly=jnp.where(q, q_ly, state.q_ly),
        w_until=jnp.where(w, later(kctx, W_TIME), state.w_until),
        w_next=jnp.where(w, later(kctx, W_FIRST), state.w_next),
        w_left=jnp.where(w, W_TICKS, state.w_left).astype(jnp.int32),
        w_cast_id=jnp.where(w, cid, state.w_cast_id),
        e_pending=state.e_pending | e, e_fire_at=jnp.where(e, later(kctx, E_CAST), state.e_fire_at),
        e_target=jnp.where(e, order.target, state.e_target).astype(jnp.int32),
        e_seq=jnp.where(e, units.spawn_seq[t], state.e_seq).astype(jnp.int32),
        e_cast_id=jnp.where(e, cid, state.e_cast_id),
        q2_pending=state.q2_pending | q2, q2_fire_at=jnp.where(q2, later(kctx, Q2_CAST), state.q2_fire_at),
        q2_dx=jnp.where(q2, ax, state.q2_dx), q2_dy=jnp.where(q2, ay, state.q2_dy),
        q2_cast_id=jnp.where(q2, cid, state.q2_cast_id),
        w2_left=jnp.where(w2, W2_HITS, state.w2_left).astype(jnp.int32),
        w2_until=jnp.where(w2, later(kctx, W2_TIME), state.w2_until),
        w2_cast_id=jnp.where(w2, cid, state.w2_cast_id),
        gate_until=jnp.where(e2, later(kctx, E2_TIME), state.gate_until),
        gate_x=jnp.where(e2, f32(kctx.x + ax * g_len), state.gate_x),
        gate_y=jnp.where(e2, f32(kctx.y + ay * g_len), state.gate_y),
        gate_nx=jnp.where(e2, ax, state.gate_nx), gate_ny=jnp.where(e2, ay, state.gate_ny),
        gate_rank=jnp.where(e2, r[:, 2], state.gate_rank).astype(jnp.int32))
    mana = jnp.sum(jnp.where(jnp.stack([q | q2, w | w2, e | e2, jnp.zeros_like(q)], -1), cost, 0.0), axis=1)
    # Hyper Charge's cooldown starts when it ends (post-effect); the others on cast.
    cd_start = jnp.stack([q | q2, w, e | e2, rr], -1)
    z = jnp.zeros((c, n), jnp.float32)
    dash = Dash(q, q_lx, q_ly, f32(jnp.maximum(leap, 1.0) / Q_LEAP_TIME), jnp.full((c,), -1, jnp.int32),
                jnp.zeros((c,), bool))
    lockout = jnp.where(e, E_CAST, jnp.where(q2, Q2_CAST, 0.0))
    return state, out(c, n, cc=CCOut(z, f32(root), z, z, z, z, rid), dash=dash, mana_cost=f32(mana),
                      cooldown_start=cd_start, base_cooldown=jnp.where(j[:, None], jnp.where(
                          cannon[:, None], cooldown_rows(r)[1], cooldown_rows(r)[0]), 0.0),
                      attack_reset=w2, cast_started=started, cast_slot=slot, cast_id=cid,
                      cast_lockout=f32(lockout), swap_cooldowns=rr)


def periodic(state: State, kctx, units) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    j = _mine(kctx)
    alive = kctx.alive
    dead = ~alive
    r = kctx.ranks
    now = f32(kctx.now)
    dt = kctx.dt
    cols = jnp.arange(n)[None, :]
    rad = units.radius[None, :]
    foe = enemies(kctx, units)
    monster = (units.kind == KIND_MONSTER)[None, :]

    # To the Skies!: the slam where Jayce lands strikes and slows enemies around him (the target included, wherever it
    # went; identity checked against a recycled slot for the attack order).
    t = jnp.clip(state.q_target, 0, n - 1)
    target_ok = units.alive[t] & (units.spawn_seq[t] == state.q_seq)
    land = state.q_pending & due(kctx, state.q_land_at) & alive
    qd = jnp.sqrt((units.x[None, :] - state.q_lx[:, None]) ** 2 + (units.y[None, :] - state.q_ly[:, None]) ** 2)
    q_hit = land[:, None] & foe & (qd <= Q_RADIUS + rad)
    q_raw = (ranked(NAME, "Q", "BaseDamage", r[:, 0]) + Q_AD * kctx.bonus_ad)[:, None] + jnp.where(monster, Q_MONSTER,
                                                                                                   0.0)
    p_q = D.packets(q_hit, kctx.unit[:, None], cols, q_raw, D.PHYSICAL, AOE, cast_id=state.q_cast_id[:, None])
    q_slow = jnp.where(q_hit, -ranked(NAME, "Q", "Slow", r[:, 0])[:, None], 0.0)
    q_pending = state.q_pending & ~(land | dead)
    follow = jnp.where(land & target_ok & (units.kind[t] == KIND_CHAMPION), state.q_target, -1)

    # Lightning Field: a tick every second (4) to enemies around Jayce.
    tick = (state.w_left > 0) & due(kctx, state.w_next) & alive
    w_hit = tick[:, None] & foe & within_edge(kctx, units, W_RADIUS)
    w_raw = (ranked(NAME, "W", "BaseDamage", r[:, 1]) + W_AP * kctx.ap) / W_TICKS
    p_w = D.packets(w_hit, kctx.unit[:, None], cols, w_raw[:, None], D.MAGIC, DOT, cast_id=state.w_cast_id[:, None])
    w_left = jnp.where(dead, 0, state.w_left - tick.astype(jnp.int32))
    w_next = jnp.where(tick, f32(state.w_next + 1.0), state.w_next)

    # Thundering Blow: at the end of the cast time, %max-health magic damage (capped vs monsters), knocked back 600
    # from Jayce over 0.35 s (airborne); no Q for 0.4 s. Cancelled if the target is lost.
    et = jnp.clip(state.e_target, 0, n - 1)
    e_ok = units.alive[et] & units.targetable[et] & (units.spawn_seq[et] == state.e_seq)
    e_due = state.e_pending & (due(kctx, state.e_fire_at) | dead)
    e_fire = e_due & alive & e_ok
    e_mask = e_fire[:, None] & (cols == et[:, None])
    e_raw = ranked(NAME, "E", "PercHPDamage", r[:, 2]) * units.max_hp[et] + kctx.bonus_ad
    e_raw = jnp.where(units.kind[et] == KIND_MONSTER, jnp.minimum(e_raw, ranked(NAME, "E", "MonsterCap", r[:, 2])),
                      e_raw)
    p_e = D.packets(e_mask, kctx.unit[:, None], cols, e_raw[:, None], D.MAGIC, SPELL,
                    cast_id=state.e_cast_id[:, None])
    kx, ky, kd = _aim(kctx, units.x[et], units.y[et])
    kx, ky = jnp.where(kd > 1e-3, kx, 1.0), jnp.where(kd > 1e-3, ky, 0.0)
    k_to = jnp.maximum(E_KNOCK_NEAR, kd + E_KNOCK_MIN)
    kb_until = jnp.where(e_fire, f32(now + dt + E_KNOCK_TIME), state.kb_until)
    kb_x = jnp.where(e_fire, f32(kctx.x + kx * k_to), state.kb_x)
    kb_y = jnp.where(e_fire, f32(kctx.y + ky * k_to), state.kb_y)
    kb_speed = jnp.where(e_fire, f32((k_to - kd) / E_KNOCK_TIME), state.kb_speed)
    q_lock_until = jnp.where(e_fire, f32(now + Q_LOCK_AFTER_E), state.q_lock_until)

    # Shock Blast: leaves at the end of the cast time; detonates on the first enemy it touches or at its range.
    q2_due = state.q2_pending & (due(kctx, state.q2_fire_at) | dead)
    go = q2_due & alive
    fly = state.m_fly | go
    ox, oy = jnp.where(go, kctx.x, state.m_ox), jnp.where(go, kctx.y, state.m_oy)
    dx, dy = jnp.where(go, state.q2_dx, state.m_dx), jnp.where(go, state.q2_dy, state.m_dy)
    d0 = jnp.where(go, 0.0, state.m_dist)
    up0 = jnp.where(go, False, state.m_up)
    gate_on = state.gate_until > now
    v0 = jnp.where(up0, Q2_SPEED_UP, Q2_SPEED)
    reach0 = jnp.where(up0, Q2_RANGE_UP, Q2_RANGE)
    d_slow = jnp.minimum(d0 + v0 * dt, reach0)
    crossed, s_g = _gate_cross(state, ox, oy, dx, dy, d0, d_slow)
    crossed = crossed & fly & ~up0 & gate_on
    # Through the gate: the rest of the tick at 2350 units/s and 1600 units total.
    rest = jnp.clip(dt - (s_g - d0) / Q2_SPEED, 0.0, dt)
    d1 = jnp.where(crossed, jnp.minimum(s_g + Q2_SPEED_UP * rest, Q2_RANGE_UP), d_slow)
    sx, sy = units.x[None, :] - ox[:, None], units.y[None, :] - oy[:, None]
    along = sx * dx[:, None] + sy * dy[:, None]
    perp = jnp.abs(sy * dx[:, None] - sx * dy[:, None])
    reach_r = Q2_HALF + rad
    touch = jnp.sqrt(jnp.maximum(reach_r ** 2 - perp ** 2, 0.0))
    s_hit = along - touch                                  # missile center where it first touches the hitbox
    m_hit = fly[:, None] & foe & (perp <= reach_r) & (s_hit <= d1[:, None]) & (along + touch >= d0[:, None])
    s_first = jnp.min(jnp.where(m_hit, jnp.maximum(s_hit, d0[:, None]), jnp.inf), axis=1)
    first = m_hit & (jnp.maximum(s_hit, d0[:, None]) <= s_first[:, None])
    hit_any = jnp.any(m_hit, axis=1)
    reach1 = jnp.where(up0 | crossed, Q2_RANGE_UP, Q2_RANGE)
    boom = fly & (hit_any | (d1 >= reach1 - 1e-3))
    s_boom = jnp.where(hit_any, s_first, d1)
    up = up0 | (crossed & (s_g <= s_boom))
    bx, by = ox + dx * s_boom, oy + dy * s_boom
    bd = jnp.sqrt((units.x[None, :] - bx[:, None]) ** 2 + (units.y[None, :] - by[:, None]) ** 2)
    blast = jnp.where(up, Q2_BLAST_UP, Q2_BLAST)
    b_hit = boom[:, None] & foe & ((bd <= blast[:, None] + rad) | first)
    q2_raw = (ranked(NAME, "Q2", "BaseDamage", r[:, 0]) + Q2_AD * kctx.bonus_ad) * jnp.where(up, Q2_UP_MULT, 1.0)
    q2_raw = q2_raw[:, None] + jnp.where(monster, Q2_MONSTER, 0.0)
    p_q2 = D.packets(b_hit, kctx.unit[:, None], cols, q2_raw, D.PHYSICAL, AOE, cast_id=state.q2_cast_id[:, None])

    # Hyper Charge ends after its attacks or 3.6 s; its cooldown starts then (in whichever stance holds it).
    w2_end = (state.w2_left > 0) & (due(kctx, state.w2_until) | dead)
    pre = stance_at_tick_start(state, kctx)

    # Acceleration Gate: Jayce touching it is hasted (full while inside, decaying over 3 s after).
    gx, gy = kctx.x - state.gate_x, kctx.y - state.gate_y
    in_gate = j & alive & gate_on & (jnp.abs(gx * state.gate_nx + gy * state.gate_ny) <= E2_THICK + RADIUS) \
        & (jnp.abs(-gx * state.gate_ny + gy * state.gate_nx) <= E2_HALF_LEN + RADIUS)
    haste_until = jnp.where(in_gate, f32(now + E2_DECAY), jnp.where(dead, -NEVER, state.haste_until))

    state = state._replace(
        q_pending=q_pending, w_left=w_left.astype(jnp.int32), w_next=f32(w_next),
        w_until=jnp.where(dead, -NEVER, state.w_until),
        e_pending=state.e_pending & ~e_due, kb_until=f32(kb_until), kb_x=f32(kb_x), kb_y=f32(kb_y),
        kb_speed=f32(kb_speed),
        q_lock_until=f32(q_lock_until),
        q2_pending=state.q2_pending & ~q2_due, m_fly=fly & ~boom, m_ox=f32(ox), m_oy=f32(oy), m_dx=f32(dx),
        m_dy=f32(dy), m_dist=f32(d1), m_up=up,
        w2_left=jnp.where(w2_end, 0, state.w2_left).astype(jnp.int32), haste_until=f32(haste_until),
        gate_until=jnp.where(dead, -NEVER, state.gate_until))
    z = jnp.zeros((c, n), jnp.float32)
    up_cc = jnp.where(e_mask, E_KNOCK_TIME, 0.0)
    slow_t = jnp.where(q_hit, Q_SLOW_TIME, 0.0)
    cid = jnp.where(q_hit, state.q_cast_id[:, None], jnp.where(e_mask, state.e_cast_id[:, None], 0))
    cc = CCOut(z, z, z, f32(up_cc), f32(q_slow), f32(slow_t), cid.astype(jnp.int32))
    ham, can = cooldown_rows(r)
    w2_now = w2_end & j & pre                              # Mercury Cannon in use: its W slot
    w2_alt = w2_end & j & ~pre                             # Mercury Hammer in use: the inactive stance's W
    return state, out(c, n, packets=D.concat_packets(p_q, p_w, p_e, p_q2), cc=cc,
                      cooldown_start=jnp.zeros((c, 4), bool).at[:, 1].set(w2_now),
                      base_cooldown=jnp.where(j[:, None], jnp.where(pre[:, None], can, ham), 0.0),
                      alt_cooldown_start=jnp.zeros((c, 3), bool).at[:, 1].set(w2_alt),
                      alt_base_cooldown=jnp.where(j[:, None], jnp.where(pre[:, None], ham, can), 0.0)[:, :3],
                      attack_target=jnp.where(j, follow, -1).astype(jnp.int32))


def on_attack(state: State, kctx, units, launch) -> tuple[State, KitOut]:
    """Hyper Charge spends an attack per launch; Mercury Cannon's shred rides the launched attack."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    j = _mine(kctx)
    target = holder_rows(launch.target, kctx)
    t = jnp.clip(target, 0, n - 1)
    went = holder_rows(launch.launched, kctx) & j & (target >= 0)
    unit_target = ~is_structure(units.kind[t]) & (units.kind[t] != KIND_WARD)
    charge = went & (state.w2_left > 0)
    left = jnp.where(charge, state.w2_left - 1, state.w2_left)
    spent = charge & (left == 0)
    shred = went & state.cannon & state.shred_ready & unit_target
    pre = state.cannon
    ham, can = cooldown_rows(kctx.ranks)
    state = state._replace(w2_left=left.astype(jnp.int32), shred_ready=state.shred_ready & ~shred,
                           shred_flying=state.shred_flying | shred)
    return state, out(c, n, cooldown_start=jnp.zeros((c, 4), bool).at[:, 1].set(spent & pre),
                      base_cooldown=jnp.where(j[:, None], jnp.where(pre[:, None], can, ham), 0.0),
                      alt_cooldown_start=jnp.zeros((c, 3), bool).at[:, 1].set(spent & ~pre),
                      alt_base_cooldown=jnp.where(j[:, None], jnp.where(pre[:, None], ham, can), 0.0)[:, :3])


def on_hit(state: State, kctx, units, launch, dodging) -> tuple[State, KitOut]:
    """Mercury Hammer's empowered hit (magic, structures included, not wards), Lightning Field's mana on hit
    (Mercury Hammer), and Mercury Cannon's shred landing (``debuffs``)."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    j = _mine(kctx)
    target = holder_rows(launch.target, kctx)
    t = jnp.clip(target, 0, n - 1)
    hit = holder_rows(launch.launched, kctx) & j & kctx.alive & (target >= 0)
    land = hit & ~gather(dodging, target)
    ward = units.kind[t] == KIND_WARD
    melee = land & ~state.cannon & ~ward
    hammer = melee & state.hammer_hit
    raw = by_level(R_HIT, kctx.level) + R_HIT_AD * kctx.bonus_ad
    p = D.packets(hammer, kctx.unit, t, raw, D.MAGIC, HIT_FLAGS)
    mana = jnp.where(melee & (kctx.ranks[:, 1] > 0), ranked(NAME, "W", "ManaGain", kctx.ranks[:, 1]), 0.0)
    shred_now = land & state.shred_flying
    shred_until = jnp.where(shred_now, f32(kctx.now + R_SHRED_TIME), state.shred_until)
    shred_on = jnp.where(shred_now, t, state.shred_on)
    state = state._replace(hammer_hit=state.hammer_hit & ~hammer, shred_flying=state.shred_flying & ~land,
                           shred_until=f32(shred_until), shred_on=shred_on.astype(jnp.int32),
                           shred_pct=jnp.where(shred_now, by_level(R_SHRED, kctx.level), state.shred_pct))
    return state, out(c, n, packets=p, mana_gain=f32(mana))


def on_damage(state: State, kctx, units, report) -> tuple[State, KitOut]:
    return state, out(kctx.unit.shape[0], units.x.shape[0])


def on_takedown(state: State, kctx, units, kills) -> State:
    return state


def stats(state: State, kctx) -> ItemStats:
    """Mercury Hammer: bonus armor and MR; Mercury Cannon: +375 attack range; Hyper Charge: +360% attack speed;
    Hextech Capacitor: +30 move speed; the gate: decaying bonus move speed."""
    j = _mine(kctx)
    now = kctx.now
    res = jnp.where(j & ~state.cannon, by_level(R_RES, kctx.level) + R_RES_AD * kctx.bonus_ad, 0.0)
    w = lambda on, v: f32(jnp.where(j & on, v, 0.0))              # noqa: E731
    return ItemStats(armor=f32(res), magic_resist=f32(res), attack_range=w(state.cannon, R_RANGE_UP),
                     attack_speed=w(state.w2_left > 0, W2_AS), move_speed=w(now < state.p_until, P_MS),
                     percent_move_speed=w(state.haste_until > now, gate_haste(state, kctx)))


def defense(state: State, kctx) -> KitDefense:
    c = kctx.unit.shape[0]
    o = jnp.ones((c,), jnp.float32)
    return KitDefense(o, jnp.zeros((c,), bool), o, jnp.zeros((c,), jnp.float32))


def attack_mods(state: State, kctx) -> KitAttackMods:
    """No attacks while leaping or in a cast time; Hyper Charge's attacks deal 70-110% of the attack's damage."""
    j = _mine(kctx)
    c = kctx.unit.shape[0]
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    busy = j & (state.q_pending | state.e_pending | state.q2_pending)
    mult = jnp.where(j & (state.w2_left > 0), ranked(NAME, "W2", "DamagePerc", kctx.ranks[:, 1]), 0.0)
    return KitAttackMods(z, f, busy, damage_mult=f32(mult))


def ghosted(state: State, kctx) -> Any:
    """Hextech Capacitor: ghosted for 0.75 s after a transform."""
    return _mine(kctx) & kctx.alive & (kctx.now < state.p_until)


def attack_missile_speed(state: State, kctx) -> Any:
    """(C,) Mercury Cannon's attack missile speed (Hyper Charge's attacks 1800), 0 = melee."""
    v = jnp.where(state.w2_left > 0, W2_MISSILE, ATTACK_MISSILE)
    return f32(jnp.where(_mine(kctx) & state.cannon, v, 0.0))


def unit_target_range(state: State, champion_id) -> Any:
    """(C, 4) unit-targeted cast ranges of the stance in use (Mercury Cannon has none)."""
    ham = jnp.asarray([Q_RANGE, 0.0, E_RANGE, 0.0], jnp.float32)
    return jnp.where(state.cannon[:, None], 0.0, ham[None, :])


def displacement(state: State, kctx, n_units: int) -> tuple[Any, Any, Any, Any]:
    """(N,) Thundering Blow's knockback: straight away from Jayce to max(475, distance + 250) over 0.35 s."""
    m = _mine(kctx)[:, None]
    cols = jnp.arange(n_units)[None, :]
    act = m & (kctx.now < state.kb_until)[:, None] & (cols == state.e_target[:, None])
    k = jnp.argmax(act, axis=0)
    pick = lambda a: jnp.take_along_axis(jnp.broadcast_to(a, act.shape), k[None, :], axis=0)[0]   # noqa: E731
    return (jnp.any(act, axis=0), f32(pick(state.kb_x[:, None])), f32(pick(state.kb_y[:, None])),
            f32(pick(state.kb_speed[:, None])))


def debuffs(state: State, kctx, units) -> Debuffs:
    """Mercury Cannon's shred: armor and MR reduced by 20-35% for 5 s on the unit its empowered attack hit."""
    n = units.x.shape[0]
    on = _mine(kctx) & (kctx.now < state.shred_until)
    pct = jnp.zeros((n,), jnp.float32).at[jnp.clip(state.shred_on, 0, n - 1)].max(jnp.where(on, state.shred_pct, 0.0))
    d = neutral_debuffs(n)
    return d._replace(percent_armor_reduction=pct, percent_mr_reduction=pct)


__all__ = ["State", "init", "cast", "periodic", "on_attack", "on_hit", "on_damage", "on_takedown", "stats",
           "defense", "attack_mods", "ghosted", "attack_missile_speed", "unit_target_range", "displacement",
           "debuffs", "by_level", "cooldown_rows", "mana_rows", "gate_haste", "stance_at_tick_start"]
