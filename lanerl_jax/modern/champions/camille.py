"""Camille (164), 26.19. Rules CAM.* and their evidence: docs/modern/CHAMPIONS.md; numbers from the pinned client
JSON (``modern/data/26.19/champions``). Hookshot finds walls with ``mechanics.first_wall`` on ``KitCtx.terrain``;
Wall Dive reads move / attack commands like Drakehound's Step; The Hextech Ultimatum leashes its target
(``leash``) and makes Camille untargetable during the leap (``untargetable``); Precision Protocol's recast converts
part of the attack to true damage (``KitAttackMods.true_split``)."""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from .. import mechanics as M
from ..core import damage as D
from ..core.types import KIND_CHAMPION, KIND_MONSTER, CCOut, Dash
from ..data.champions import spell
from ..items.catalog import ItemStats
from ..items.effects.core import Debuffs, ShieldGrant, neutral_debuffs
from ..runes.catalog import ChampionTraits
from .core import (CODE_CAMILLE_W_OUTER, NEVER, KitAttackMods, KitDefense, KitOut, cc_matrix, cooldown_row, due,
                   enemies, f32, gather, holder_rows, later, make_cast_id, mana_row, out, ranked, scalar,
                   target_dist)
from ..jungle import objectives as OBJ
from .nasus import LARGE_EPICS

NAME = "Camille"
ID = 164
SKILL_ORDER = (0, 2, 1, 0, 0, 3, 0, 1, 0, 1, 3, 1, 1, 2, 2, 3, 2, 2)    # default ranks per level: Q > W > E
TRAITS = ChampionTraits(has_immobilize=True, resource="mana")          # rune legality: E stuns
UNIT_TARGET_RANGE = (0.0, 0.0, 0.0, 475.0)

_P = spell(NAME, "Passive")["calculations"]
_PS = _P["ShieldAmount"]["mFormulaParts"][0]["mSubpart"]
P_SHIELD = _PS["mLevel1Value"]                         # 10% max health, +5% at levels 7, 13 and 19
P_SHIELD_STEPS = tuple(b["mLevel"] for b in _PS["mBreakpoints"])
P_SHIELD_STEP = _PS["mBreakpoints"][0]["mAdditionalBonusAtThisLevel"]
_PC = _P["PassiveCooldown"]["mFormulaParts"][0]
P_COOLDOWN = _PC["mLevel1Value"]                       # 14 s, -3 at levels 7 and 13
P_COOLDOWN_STEPS = tuple(b["mLevel"] for b in _PC["mBreakpoints"])
P_COOLDOWN_STEP = _PC["mBreakpoints"][0]["mAdditionalBonusAtThisLevel"]
P_SHIELD_TIME = scalar(NAME, "Passive", "ShieldDuration")            # 2

Q_WINDOW = 4.0                                         # wiki: the next attack within 4 s
Q_RANGE = scalar(NAME, "Q", "BonusAARange")            # +50
Q_MS_TIME = scalar(NAME, "Q", "MSDuration")            # 1
Q_RECAST_FROM = 0.25                                   # wiki: recastable 0.25 s after the empowered attack ...
Q_RECAST_WINDOW = scalar(NAME, "Q", "QTotalRecastTime")              # ... within 3.5 s
Q_RAMP = scalar(NAME, "Q", "QRampUpTime")              # 1.5 s after the first attack: empowered
Q_AMP = scalar(NAME, "Q", "QEmpoweredAmp")             # x2 bonus
_QT = spell(NAME, "Q")["calculations"]["DamageConversionPercentage"]["mFormulaParts"][0]
Q_TRUE_BASE = _QT["mLevel1Value"]                      # 40% true, +4% per level, capped at level 16 (100%)
Q_TRUE_PER = _QT["mInitialBonusPerLevel"]

W_DELAY = 1.1                                          # wiki: the sweep lands 1.1 s after the cast (client ChargeDuration 0.75)
W_LENGTH = scalar(NAME, "W", "BlastLength")            # 650 (centre range)
W_INNER = 325.0                                        # wiki: the outer half starts at 325
W_HALF_ANGLE = scalar(NAME, "W", "ConeAngle")          # 35 degrees (a 70 degree cone)
W_AD_PER_PCT = scalar(NAME, "W", "ADRequiredFor1PercentDamage")      # +1% max health per 40 bonus AD
W_MONSTER_CAP = 300.0                                  # MonsterHealthDamageCap
W_MONSTER_MULT = 1.0 - scalar(NAME, "W", "MonsterDamageReduction") / 100.0   # non-epic monsters take 50%
W_SLOW = scalar(NAME, "W", "SlowPercentage") / 100.0   # 80% decaying over 2 s
W_SLOW_TIME = scalar(NAME, "W", "SlowDuration")
W_HEAL = scalar(NAME, "W", "OuterConeHealingRatio") / 100.0          # 100% of the outer bonus vs champions

E_RANGE = 800.0                                        # Hookshot range (CamilleE castRangeDisplayOverride; wiki)
E_HOOK_SPEED = 1400.0                                  # the grapple
E_SPEED = scalar(NAME, "E", "DashSpeed")               # 1050 + move speed (pull and dive; replay-checked)
E_HANG = scalar(NAME, "E", "WallHangDuration")         # 1 s on the wall (client; wiki 0.75)
E_SHORT = scalar(NAME, "E", "E2ShortDashRange")        # 400
E_LONG = scalar(NAME, "E", "E2LongDashRange")          # 800 toward an enemy champion
E_CHAMP_SEARCH = 1400.0                                # wiki: champions within 1400 extend the dive
E_RADIUS = scalar(NAME, "E", "ECollisionRange")        # 130: collision and landing radius
E_STUN = scalar(NAME, "E", "KnockupDuration")          # 0.75
E_AS_TIME = scalar(NAME, "E", "ASDuration")            # 5

R_RANGE = 475.0                                        # centre range
R_LEAP = 0.5                                           # wiki: lands 0.5 s after the cast
R_ZONE = scalar(NAME, "R", "RCircleRadius")            # 425 (a hexagon live; circle here)
R_KNOCK_SPEED = 1000.0                                 # wiki knock-away speed
R_LEAVE = 1.0                                          # wiki: ends within 1 s of Camille leaving

Q_FLAGS = D.TAG_PROC | D.PROP_LIFESTEAL
AOE = D.TAG_AOE | D.TAG_ACTIVE_SPELL
R_FLAGS = D.TAG_PROC


class State(NamedTuple):
    p_ready_at: Any         # (C,) Adaptive Defenses ready from
    p_until: Any            # (C,) its shield ends (the cooldown starts then, or when broken)
    p_on: Any               # (C,) bool: a shield is up
    dealt_phys: Any         # (C, N) physical damage unit n dealt to champions (shield typing)
    dealt_magic: Any        # (C, N)
    q_armed: Any            # (C,) bool: the next attack is Precision Protocol's
    q_until: Any            # its window
    q_second: Any           # (C,) bool: the armed attack is the recast's
    q_first_at: Any         # time of the first empowered attack
    q_recast_from: Any
    q_recast_until: Any
    q_cast_id: Any
    q_hit: Any              # (C,) bool: the attack launched this tick is empowered
    q_conv: Any             # (C,) true share of that attack (0 if not ramped)
    q_ms_until: Any
    w_fire_at: Any
    w_dx: Any
    w_dy: Any
    w_cast_id: Any
    w_slow_from: Any        # (C, N) decaying slow start
    e_stage: Any            # (C,) int32 0 idle, 1 grapple flying, 2 pulled / hanging, 3 diving
    e_arrive_at: Any        # grapple reaches the wall
    e_wx: Any               # wall point
    e_wy: Any
    e_hang_until: Any
    e_hang_from: Any
    e_dive_from: Any
    e_cast_id: Any
    e_as_until: Any
    r_target: Any           # (C,) int32 challenged champion, -1 none
    r_seq: Any
    r_land_at: Any
    r_cx: Any
    r_cy: Any
    r_until: Any            # zone end
    r_out_since: Any        # Camille outside the zone since
    r_cast_id: Any
    r_knock_until: Any      # (C, N) knock-away end
    r_kx: Any               # (C, N) knock-away point
    r_ky: Any


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    i = jnp.zeros((c,), jnp.int32)
    zn = jnp.zeros((c, n), jnp.float32)
    return State(z - NEVER, z - NEVER, f, zn, zn,
                 f, z - NEVER, f, z - NEVER, z - NEVER, z - NEVER, i, f, z, z - NEVER,
                 z + NEVER, z, z, i, zn - NEVER,
                 i, z + NEVER, z, z, z - NEVER, z - NEVER, z - NEVER, i, z - NEVER,
                 i - 1, i, z - NEVER, z, z, z - NEVER, z + NEVER, i, zn - NEVER, zn, zn)


def _mine(kctx) -> Any:
    return kctx.champion_id == ID


def _steps(level, steps) -> Any:
    lv = jnp.asarray(level)
    return sum((lv >= s).astype(jnp.float32) for s in steps)


def shield_share(level) -> Any:
    return P_SHIELD + P_SHIELD_STEP * _steps(level, P_SHIELD_STEPS)


def passive_cooldown(level) -> Any:
    return P_COOLDOWN + P_COOLDOWN_STEP * _steps(level, P_COOLDOWN_STEPS)


def true_share(level) -> Any:
    """Precision Protocol's empowered recast: 40% + 4% per level after 1, capped at level 16 (100%)."""
    lv = jnp.clip(jnp.asarray(level, jnp.float32), 1.0, 16.0)
    return jnp.minimum(Q_TRUE_BASE + Q_TRUE_PER * (lv - 1.0), 1.0)


def _base_cd(kctx) -> Any:
    return jnp.where(_mine(kctx)[:, None], cooldown_row(NAME, kctx.ranks), 0.0)


def _unit_dir(dx, dy):
    d = jnp.sqrt(dx * dx + dy * dy)
    ok = d > 1e-3
    return jnp.where(ok, dx / jnp.maximum(d, 1e-3), 1.0), jnp.where(ok, dy / jnp.maximum(d, 1e-3), 0.0), d


def cast(state: State, kctx, units, order) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    r = kctx.ranks
    now = f32(kctx.now)
    leaping = now < state.r_land_at
    free = m & kctx.alive & ~kctx.stunned & ~kctx.silenced & ~leaping
    rooted = jnp.zeros_like(m) if kctx.rooted is None else kctx.rooted
    cost = mana_row(NAME, r)
    ready = (r > 0) & (kctx.cooldowns <= 0) & (kctx.mana[:, None] >= cost)
    want = lambda s: (order.slot == s) & free          # noqa: E731
    ux, uy, dist = _unit_dir(order.x - kctx.x, order.y - kctx.y)

    # Precision Protocol: arm the next attack; the recast in its window costs nothing.
    recast = (now >= state.q_recast_from - 1e-6) & (now <= state.q_recast_until)
    q1 = want(0) & ready[:, 0] & ~recast & ~state.q_armed
    q2 = want(0) & recast & ~state.q_armed
    q = q1 | q2
    # Tactical Sweep (the cone lands 1.1 s later; other spells stay castable).
    w = want(1) & ready[:, 1] & (state.w_fire_at > NEVER / 2)
    # Hookshot; Wall Dive while hanging.
    e1 = want(2) & ready[:, 2] & (state.e_stage == 0)
    e2 = want(2) & (state.e_stage == 2) & (now >= state.e_hang_from) & ~rooted
    # The Hextech Ultimatum.
    t = jnp.clip(order.target, 0, n - 1)
    valid_r = (order.target >= 0) & units.alive[t] & units.targetable[t] & (units.kind[t] == KIND_CHAMPION) \
        & (units.team[t] != kctx.team) & (target_dist(kctx, units, order.target) <= R_RANGE)
    rr = want(3) & ready[:, 3] & valid_r & ~rooted & (state.e_stage != 2)

    # The grapple: the first wall within 800 in the cast direction.
    ter = kctx.terrain
    if ter is not None:
        hit, wx, wy = M.first_wall(kctx.x, kctx.y, ux, uy, jnp.full((c,), E_RANGE, jnp.float32), kctx.team, ter)
    else:
        hit, wx, wy = jnp.zeros((c,), bool), kctx.x, kctx.y
    wall_d = jnp.sqrt((wx - kctx.x) ** 2 + (wy - kctx.y) ** 2)
    e_miss = e1 & ~hit

    started = q | w | e1 | e2 | rr
    slot = jnp.where(q, 0, jnp.where(w, 1, jnp.where(e1 | e2, 2, jnp.where(rr, 3, -1)))).astype(jnp.int32)
    cid = jnp.where(started, make_cast_id(kctx, jnp.maximum(slot, 0)), 0).astype(jnp.int32)
    leap = Dash(rr, f32(units.x[t]), f32(units.y[t]),
                f32(jnp.maximum(target_dist(kctx, units, order.target), 1.0) / R_LEAP), jnp.where(rr, order.target, -1)
                .astype(jnp.int32), jnp.zeros((c,), bool))
    state = state._replace(
        q_armed=state.q_armed | q, q_until=jnp.where(q, later(kctx, Q_WINDOW), state.q_until),
        q_second=jnp.where(q, q2, state.q_second), q_cast_id=jnp.where(q, cid, state.q_cast_id),
        q_recast_until=jnp.where(q2, -NEVER, state.q_recast_until),
        w_fire_at=jnp.where(w, later(kctx, W_DELAY), state.w_fire_at),
        w_dx=jnp.where(w, f32(ux), state.w_dx), w_dy=jnp.where(w, f32(uy), state.w_dy),
        w_cast_id=jnp.where(w, cid, state.w_cast_id),
        e_stage=jnp.where(e1 & hit, 1, jnp.where(e2, 3, state.e_stage)).astype(jnp.int32),
        e_arrive_at=jnp.where(e1 & hit, f32(now + wall_d / E_HOOK_SPEED), state.e_arrive_at),
        e_wx=jnp.where(e1 & hit, f32(wx), state.e_wx), e_wy=jnp.where(e1 & hit, f32(wy), state.e_wy),
        e_cast_id=jnp.where(e1 | e2, cid, state.e_cast_id),
        e_dive_from=jnp.where(e2, now, state.e_dive_from),
        r_target=jnp.where(rr, order.target, state.r_target).astype(jnp.int32),
        r_seq=jnp.where(rr, units.spawn_seq[t], state.r_seq).astype(jnp.int32),
        r_land_at=jnp.where(rr, later(kctx, R_LEAP), state.r_land_at), r_cast_id=jnp.where(rr, cid, state.r_cast_id),
        r_until=jnp.where(rr, -NEVER, state.r_until))                 # no zone until the landing
    ms = jnp.zeros((c,), jnp.float32) if kctx.move_speed is None else kctx.move_speed
    dive = _dive(state, kctx, units, e2, ux, uy, ms)
    dash = Dash(*(jnp.where(leap.active, a, b) for a, b in zip(leap, dive)))
    mana = jnp.sum(jnp.where(jnp.stack([q1, w, e1, rr], -1), cost, 0.0), axis=1)
    cd = jnp.stack([jnp.zeros_like(q), w, e_miss, rr], -1)
    return state, out(c, n, dash=dash, mana_cost=f32(mana), cooldown_start=cd, base_cooldown=_base_cd(kctx),
                      attack_reset=q, cast_started=started, cast_slot=slot, cast_id=cid,
                      cast_lockout=f32(jnp.where(rr, R_LEAP, 0.0)))


def _dive(state: State, kctx, units, go, ux, uy, ms) -> Dash:
    """Wall Dive from the wall toward ``(ux, uy)``: 400, or up to 800 toward an enemy champion within 1400 whose
    hitbox the dive line passes."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    rx, ry = units.x[None, :] - kctx.x[:, None], units.y[None, :] - kctx.y[:, None]
    along = rx * ux[:, None] + ry * uy[:, None]
    perp = jnp.abs(ry * ux[:, None] - rx * uy[:, None])
    champ = enemies(kctx, units) & (units.kind == KIND_CHAMPION)[None, :]
    toward = champ & (along > 0) & (along <= jnp.minimum(E_LONG, E_CHAMP_SEARCH)) \
        & (perp <= E_RADIUS + units.radius[None, :])
    reach = jnp.where(jnp.any(toward, axis=1), E_LONG, E_SHORT)
    return Dash(go, f32(kctx.x + ux * reach), f32(kctx.y + uy * reach), f32(E_SPEED + ms), jnp.full((c,), -1, jnp.int32),
                jnp.zeros((c,), bool))


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
    rx, ry = units.x[None, :] - kctx.x[:, None], units.y[None, :] - kctx.y[:, None]
    dist = jnp.sqrt(rx * rx + ry * ry)
    dashing = jnp.zeros_like(m) if kctx.dashing is None else kctx.dashing
    ms = jnp.zeros((c,), jnp.float32) if kctx.move_speed is None else kctx.move_speed

    # Adaptive Defenses: the shield ending (expiry or broken) starts the cooldown.
    kit_shield = jnp.ones((c,), jnp.float32) if kctx.kit_shield is None else kctx.kit_shield
    broken = state.p_on & (now > state.p_until - P_SHIELD_TIME + 1.5 * kctx.dt) & (kit_shield <= 0.0)
    ended = state.p_on & (due(kctx, state.p_until) | broken | dead)
    p_ready_at = jnp.where(ended, f32(now + passive_cooldown(kctx.level)), state.p_ready_at)

    # Precision Protocol window.
    q_lapse = state.q_armed & (due(kctx, state.q_until) | dead)

    # Tactical Sweep.
    w_due = m & due(kctx, state.w_fire_at)
    w_fire = w_due & alive
    along = rx * state.w_dx[:, None] + ry * state.w_dy[:, None]
    ang = jnp.arctan2(jnp.abs(ry * state.w_dx[:, None] - rx * state.w_dy[:, None]), along)
    w_hit = w_fire[:, None] & foe & (dist <= W_LENGTH) & (ang <= jnp.deg2rad(W_HALF_ANGLE)) & (along > 0)
    outer = w_hit & (dist > W_INNER)
    monster = (units.kind == KIND_MONSTER)[None, :]
    epic = jnp.zeros((n,), bool)
    for e_ in LARGE_EPICS:
        epic = epic | (units.sub == OBJ.SUB_BASE + e_)
    small_monster = monster & ~epic[None, :]
    base = ranked(NAME, "W", "BaseDamage", r[:, 1]) + ranked(NAME, "W", "BADRatio", r[:, 1]) * kctx.bonus_ad
    pct = ranked(NAME, "W", "OuterConeMaxHPDamage", r[:, 1]) + kctx.bonus_ad / W_AD_PER_PCT / 100.0
    bonus = pct[:, None] * units.max_hp[None, :]
    bonus = jnp.where(monster, jnp.minimum(bonus, W_MONSTER_CAP), bonus)
    mon_mult = jnp.where(small_monster, W_MONSTER_MULT, 1.0)
    w_out_id = make_cast_id(kctx, CODE_CAMILLE_W_OUTER)
    p_w = D.concat_packets(
        D.packets(w_hit, kctx.unit[:, None], cols, base[:, None] * mon_mult, D.PHYSICAL, AOE,
                  cast_id=state.w_cast_id[:, None]),
        D.packets(outer, kctx.unit[:, None], cols, bonus * mon_mult, D.PHYSICAL, AOE, cast_id=w_out_id[:, None]))
    w_slow_from = jnp.where(outer, now, state.w_slow_from)
    left = 1.0 - (now - w_slow_from) / W_SLOW_TIME
    slowing = m[:, None] & (left > 0.0) & units.alive[None, :]
    w_slow = jnp.where(slowing, W_SLOW * left, 0.0)

    # Hookshot: the grapple reaches the wall, Camille is pulled there, then hangs; a dive or the hang's end.
    arrive = (state.e_stage == 1) & due(kctx, state.e_arrive_at) & alive
    pull = Dash(arrive, state.e_wx, state.e_wy, f32(E_SPEED + ms), jnp.full((c,), -1, jnp.int32),
                jnp.zeros((c,), bool))
    on_wall = (state.e_stage == 1) & (now > state.e_arrive_at + 0.5 * kctx.dt) & ~dashing
    e_stage = jnp.where(on_wall, 2, state.e_stage)
    e_hang_from = jnp.where(on_wall, now, state.e_hang_from)
    e_hang_until = jnp.where(on_wall, f32(now + E_HANG), state.e_hang_until)
    # A move or attack command while hanging dives toward it (wiki interaction table).
    commanded = (kctx.last_order_at if kctx.last_order_at is not None else jnp.full((c,), -NEVER)) \
        >= e_hang_from - 1e-6
    at_ = jnp.full((c,), -1, jnp.int32) if kctx.attack_order is None else kctx.attack_order
    atc = jnp.clip(at_, 0, n - 1)
    goal = jnp.zeros((c, 2)) if kctx.move_goal is None else kctx.move_goal
    gx = jnp.where(at_ >= 0, units.x[atc], goal[:, 0])
    gy = jnp.where(at_ >= 0, units.y[atc], goal[:, 1])
    gux, guy, _ = _unit_dir(gx - kctx.x, gy - kctx.y)
    auto = (e_stage == 2) & ~on_wall & commanded & alive & m
    dive2 = _dive(state, kctx, units, auto, gux, guy, ms)
    hang_over = (e_stage == 2) & due(kctx, e_hang_until) & ~auto
    e_stage = jnp.where(auto, 3, e_stage)
    e_dive_from = jnp.where(auto, now, state.e_dive_from)
    # The dive: stops on the first enemy champion it touches; lands on enemies near the end.
    diving = (e_stage == 3) & (now > e_dive_from + 0.5 * kctx.dt)
    champ_near = foe & (units.kind == KIND_CHAMPION)[None, :] & (dist <= E_RADIUS + rad)
    collide = diving & dashing & jnp.any(champ_near, axis=1)
    land = diving & (~dashing | collide)
    stop = Dash(collide, f32(kctx.x), f32(kctx.y), jnp.full((c,), 1e6, jnp.float32), jnp.full((c,), -1, jnp.int32),
                jnp.zeros((c,), bool))
    e_hit = land[:, None] & foe & (dist <= E_RADIUS + rad)
    e_raw = ranked(NAME, "E", "BaseDamage", r[:, 2]) + ranked(NAME, "E", "BADRatio", r[:, 2]) * kctx.bonus_ad
    p_e = D.packets(e_hit, kctx.unit[:, None], cols, e_raw[:, None], D.PHYSICAL, AOE, cast_id=state.e_cast_id[:, None])
    e_stun = jnp.where(collide[:, None] & champ_near, E_STUN, 0.0)
    e_done = land | hang_over | ((state.e_stage > 0) & dead)
    e_stage = jnp.where(e_done, 0, e_stage)

    # The Hextech Ultimatum: lands after 0.5 s; zone around Camille; other enemies knocked to its edge.
    rt = jnp.clip(state.r_target, 0, n - 1)
    r_valid = (state.r_target >= 0) & units.alive[rt] & (units.spawn_seq[rt] == state.r_seq)
    leaping = m & (state.r_target >= 0) & (now < state.r_land_at)
    landed = leaping & r_valid & due(kctx, state.r_land_at) & alive
    failed = leaping & (~r_valid | dead)
    r_cx = jnp.where(landed, kctx.x, state.r_cx)
    r_cy = jnp.where(landed, kctx.y, state.r_cy)
    r_until = jnp.where(landed, f32(now + ranked(NAME, "R", "RDuration", r[:, 3])), state.r_until)
    others = landed[:, None] & foe & (cols != rt[:, None]) & (dist <= R_ZONE + rad)
    kux = jnp.where(dist > 1e-3, rx / jnp.maximum(dist, 1e-3), 1.0)
    kuy = jnp.where(dist > 1e-3, ry / jnp.maximum(dist, 1e-3), 0.0)
    kx, ky = kctx.x[:, None] + kux * (R_ZONE + rad), kctx.y[:, None] + kuy * (R_ZONE + rad)
    t_knock = jnp.maximum(R_ZONE + rad - dist, 0.0) / R_KNOCK_SPEED
    r_knock_until = jnp.where(others, f32(now + kctx.dt + t_knock), state.r_knock_until)
    r_kx, r_ky = jnp.where(others, kx, state.r_kx), jnp.where(others, ky, state.r_ky)
    knock_up = jnp.where(others, t_knock + 1e-3, 0.0)
    zone_on = m & (now < r_until) & r_valid
    outside = zone_on & (jnp.sqrt((kctx.x - r_cx) ** 2 + (kctx.y - r_cy) ** 2) > R_ZONE)
    r_out_since = jnp.where(outside, jnp.minimum(state.r_out_since, now), NEVER)
    r_end = (m & (state.r_target >= 0) & ~leaping & (~zone_on | dead
                                                                       | (now - r_out_since >= R_LEAVE))) | failed
    r_target = jnp.where(r_end, -1, state.r_target).astype(jnp.int32)
    r_until = jnp.where(r_end, -NEVER, r_until)

    state = state._replace(
        p_ready_at=f32(p_ready_at), p_on=state.p_on & ~ended,
        q_armed=state.q_armed & ~q_lapse,
        w_fire_at=jnp.where(w_due | dead, NEVER, state.w_fire_at), w_slow_from=f32(w_slow_from),
        e_stage=e_stage.astype(jnp.int32), e_hang_from=f32(e_hang_from), e_hang_until=f32(e_hang_until),
        e_dive_from=f32(e_dive_from),
        e_as_until=jnp.where(land, f32(now + E_AS_TIME), state.e_as_until),
        r_cx=f32(r_cx), r_cy=f32(r_cy), r_until=f32(r_until), r_out_since=f32(r_out_since), r_target=r_target,
        r_land_at=jnp.where(r_end | landed, -NEVER, state.r_land_at), r_knock_until=f32(r_knock_until),
        r_kx=f32(r_kx), r_ky=f32(r_ky))
    dash = Dash(*(jnp.where(stop.active, a, jnp.where(dive2.active, b, p)) for p, b, a in zip(pull, dive2, stop)))
    cc = CCOut(f32(e_stun), z, z, f32(knock_up), f32(w_slow), jnp.where(w_slow > 0, 1.5 * kctx.dt, 0.0),
               jnp.where(e_stun > 0, state.e_cast_id[:, None], jnp.where(outer, w_out_id[:, None], 0))
               .astype(jnp.int32))
    cd = jnp.zeros((c, 4), bool).at[:, 0].set(q_lapse & m & ~state.q_second & (state.q_first_at < -NEVER / 2)) \
        .at[:, 2].set(e_done & m & (state.e_stage > 0))
    return state, out(c, n, packets=D.concat_packets(p_w, p_e), cc=cc, dash=dash, base_cooldown=_base_cd(kctx))


def on_attack(state: State, kctx, units, launch) -> tuple[State, KitOut]:
    """Precision Protocol consumes the armed attack at launch; the recast's ramps 1.5 s after the first."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    target = holder_rows(launch.target, kctx)
    went = holder_rows(launch.launched, kctx) & m & (target >= 0)
    emp = went & state.q_armed
    first = emp & ~state.q_second
    ramped = emp & state.q_second & (kctx.now >= state.q_first_at + Q_RAMP - 1e-6)
    now = f32(kctx.now)
    state = state._replace(
        q_armed=state.q_armed & ~emp, q_hit=emp, q_conv=jnp.where(ramped, true_share(kctx.level), 0.0),
        q_first_at=jnp.where(first, now, jnp.where(emp & state.q_second, -NEVER, state.q_first_at)),
        q_recast_from=jnp.where(first, f32(now + Q_RECAST_FROM), state.q_recast_from),
        q_recast_until=jnp.where(first, f32(now + Q_RECAST_FROM + Q_RECAST_WINDOW), state.q_recast_until),
        q_ms_until=jnp.where(emp, f32(now + Q_MS_TIME), state.q_ms_until))
    cd = jnp.zeros((c, 4), bool).at[:, 0].set(first)
    return state, out(c, n, cooldown_start=cd, base_cooldown=_base_cd(kctx))


def on_hit(state: State, kctx, units, launch, dodging) -> tuple[State, KitOut]:
    """Precision Protocol's bonus (x2 and part true when ramped); The Hextech Ultimatum's current-health magic
    damage on its target; Adaptive Defenses' typed shield on hitting an enemy champion."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    target = holder_rows(launch.target, kctx)
    t = jnp.clip(target, 0, n - 1)
    hit = holder_rows(launch.launched, kctx) & m & kctx.alive & (target >= 0)
    land = hit & ~gather(dodging, target)
    now = f32(kctx.now)
    q = land & state.q_hit
    amp = jnp.where(state.q_conv > 0, Q_AMP, 1.0)
    bonus = ranked(NAME, "Q", "TADRatio", kctx.ranks[:, 0]) * kctx.total_ad * amp
    p_q = D.concat_packets(
        D.packets(q, kctx.unit, t, bonus * (1.0 - state.q_conv), D.PHYSICAL, Q_FLAGS, cast_id=state.q_cast_id),
        D.packets(q & (state.q_conv > 0), kctx.unit, t, bonus * state.q_conv, D.TRUE, Q_FLAGS,
                  cast_id=state.q_cast_id))
    in_zone = (state.r_target == target) & (now < state.r_until)
    p_r = D.packets(land & in_zone, kctx.unit, t,
                    ranked(NAME, "R", "RPercentCurrentHPDamage", kctx.ranks[:, 3]) / 100.0 * units.hp[t], D.MAGIC,
                    R_FLAGS, cast_id=state.r_cast_id)
    champ = units.kind[t] == KIND_CHAMPION
    give = land & champ & (now >= state.p_ready_at - 1e-6) & ~state.p_on
    phys = jnp.take_along_axis(state.dealt_phys, t[:, None], axis=1)[:, 0]
    mag = jnp.take_along_axis(state.dealt_magic, t[:, None], axis=1)[:, 0]
    threat_p = phys * 100.0 / (100.0 + jnp.maximum(kctx.armor, 0.0))
    threat_m = mag * 100.0 / (100.0 + jnp.maximum(kctx.magic_resist, 0.0))
    kind = jnp.where(threat_m > threat_p, D.SHIELD_MAGIC, D.SHIELD_PHYSICAL) + D.SOURCE_SHIFT * D.SOURCE_KIT
    amount = jnp.where(give, shield_share(kctx.level) * kctx.max_hp, 0.0)[:, None]
    shield = ShieldGrant(f32(amount), kind[:, None].astype(jnp.int32), jnp.where(give, P_SHIELD_TIME, 0.0)[:, None],
                         jnp.zeros((c, 1), jnp.float32))
    state = state._replace(q_hit=state.q_hit & ~hit, p_on=state.p_on | give,
                           p_until=jnp.where(give, f32(now + P_SHIELD_TIME), state.p_until))
    return state, out(c, n, packets=D.concat_packets(p_q, p_r), shield=shield, base_cooldown=_base_cd(kctx))


def on_damage(state: State, kctx, units, report) -> tuple[State, KitOut]:
    """Records each unit's physical / magic damage to champions (Adaptive Defenses' typing) and heals Tactical
    Sweep's outer bonus dealt to champions."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    p = report.packets
    res = report.resolved
    final = getattr(res, "final", None)
    final = getattr(res, "health_loss", p.raw) if final is None else final
    final = jnp.where(p.valid, final, 0.0)
    src = jnp.clip(p.src, 0, n - 1)
    dst = jnp.clip(p.dst, 0, n - 1)
    to_champ = (units.kind[dst] == KIND_CHAMPION) & (units.kind[src] == KIND_CHAMPION)
    onehot = (src[None, :] == jnp.arange(n)[:, None])                             # (N, P)
    add_p = jnp.sum(jnp.where(onehot & to_champ[None, :] & (p.dtype == D.PHYSICAL)[None, :], final[None, :], 0.0),
                    axis=1)
    add_m = jnp.sum(jnp.where(onehot & to_champ[None, :] & (p.dtype == D.MAGIC)[None, :], final[None, :], 0.0), axis=1)
    m = _mine(kctx)
    outer_id = make_cast_id(kctx, CODE_CAMILLE_W_OUTER)
    mine = (p.src[None, :] == kctx.unit[:, None]) & (p.cast_id[None, :] == outer_id[:, None]) \
        & (units.kind[dst] == KIND_CHAMPION)[None, :]
    heal = jnp.where(m & kctx.alive, jnp.sum(jnp.where(mine, final[None, :], 0.0), axis=1) * W_HEAL, 0.0)
    return state._replace(dealt_phys=state.dealt_phys + jnp.where(m[:, None], add_p[None, :], 0.0),
                          dealt_magic=state.dealt_magic + jnp.where(m[:, None], add_m[None, :], 0.0)), \
        out(c, n, heal=f32(heal))


def on_takedown(state: State, kctx, units, kills) -> State:
    return state


def stats(state: State, kctx) -> ItemStats:
    """Precision Protocol's move speed after an empowered attack; Wall Dive's attack speed."""
    m = _mine(kctx)
    q_ms = m & (kctx.now < state.q_ms_until)
    e_as = m & (kctx.now < state.e_as_until)
    return ItemStats(percent_move_speed=f32(jnp.where(q_ms, ranked(NAME, "Q", "MSBonus", kctx.ranks[:, 0]), 0.0)),
                     attack_speed=f32(jnp.where(e_as, ranked(NAME, "E", "ASBuff", kctx.ranks[:, 2]), 0.0)))


def defense(state: State, kctx) -> KitDefense:
    c = kctx.unit.shape[0]
    o = jnp.ones((c,), jnp.float32)
    return KitDefense(o, jnp.zeros((c,), bool), o, jnp.zeros((c,), jnp.float32))


def attack_mods(state: State, kctx) -> KitAttackMods:
    """Precision Protocol: +50 range, uncancellable, no crit; the ramped recast's true share. No attacks during
    Tactical Sweep, Hookshot or the leap."""
    m = _mine(kctx)
    c = kctx.unit.shape[0]
    z = jnp.zeros((c,), jnp.float32)
    armed = m & state.q_armed
    ramp = armed & state.q_second & (kctx.now >= state.q_first_at + Q_RAMP - 1e-6)
    busy = m & ((state.w_fire_at < NEVER / 2) | (state.e_stage > 0) | (kctx.now < state.r_land_at))
    return KitAttackMods(f32(jnp.where(armed, Q_RANGE, 0.0)), jnp.zeros((c,), bool), busy, armed, z, z, armed,
                         true_split=f32(jnp.where(ramp, true_share(kctx.level), 0.0)))


def ghosted(state: State, kctx) -> Any:
    """Tactical Sweep's wind-up."""
    return _mine(kctx) & (state.w_fire_at < NEVER / 2)


def immobile(state: State, kctx) -> Any:
    """(C,) bound to the wall while hanging (Wall Dive's dash still moves her)."""
    return _mine(kctx) & (state.e_stage == 2)


def untargetable(state: State, champion_id, now) -> Any:
    """The Hextech Ultimatum's leap."""
    return (champion_id == ID) & (now < state.r_land_at)


def leash(state: State, kctx, n_units: int) -> tuple[Any, Any, Any, Any]:
    """(N,) the challenged champion cannot leave the zone."""
    on = _mine(kctx) & (state.r_target >= 0) & (kctx.now < state.r_until)
    tgt = jnp.where(on, state.r_target, n_units)
    put = lambda v, z: z.at[tgt].set(v)[:n_units]                    # noqa: E731
    zf = jnp.zeros((n_units + 1,), jnp.float32)
    return put(on, jnp.zeros((n_units + 1,), bool)), put(state.r_cx, zf), put(state.r_cy, zf), \
        put(jnp.full(on.shape, R_ZONE, jnp.float32), zf)


def displacement(state: State, kctx, n_units: int) -> tuple[Any, Any, Any, Any]:
    """(N,) the other enemies knocked to the zone's edge at 1000 units/s."""
    on = _mine(kctx)[:, None] & (kctx.now < state.r_knock_until)
    k = jnp.argmax(on, axis=0)
    pick = lambda a: jnp.take_along_axis(jnp.broadcast_to(a, on.shape), k[None, :], axis=0)[0]   # noqa: E731
    return jnp.any(on, axis=0), pick(state.r_kx), pick(state.r_ky), jnp.full((n_units,), R_KNOCK_SPEED, jnp.float32)


def debuffs(state: State, kctx, units) -> Debuffs:
    return neutral_debuffs(units.x.shape[0])


__all__ = ["State", "init", "cast", "periodic", "on_attack", "on_hit", "on_damage", "on_takedown", "stats",
           "defense", "attack_mods", "ghosted", "immobile", "untargetable", "leash", "displacement", "debuffs",
           "shield_share", "passive_cooldown", "true_share"]
