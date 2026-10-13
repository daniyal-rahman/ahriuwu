"""K'Sante (897), 26.19. Rules KSA.* and their evidence: docs/modern/CHAMPIONS.md; numbers from the pinned client
JSON (``modern/data/26.19/champions``) and, where marked, other client records or the 16.9 replay measurements
(/mnt/nfs/shared/champ-expansion/ksante-measurements). Path Maker's charge uses ``displacement_immune``, ``immobile``
and the raw cast slot (its recast works while stunned); All Out uses the kit stat buckets ``base_armor_loss`` /
``percent_bonus_armor_pen``, ``hp_cap``, ``mana_override``, ``untargetable`` (the pull over terrain) and a
tenacity-free root (``CCOut.tenacity_free``)."""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from .. import mechanics as M
from ..core import damage as D
from ..core.types import KIND_CHAMPION, KIND_MINION, KIND_MONSTER, KIND_WARD, CCOut, Dash
from ..data.champions import spell
from ..items.catalog import ItemStats
from ..items.effects.core import Debuffs, ShieldGrant, neutral_debuffs
from ..runes.catalog import ChampionTraits
from .core import (NEVER, KitAttackMods, KitDefense, KitOut, cooldown_row, due, enemies, f32, gather, holder_rows,
                   is_structure, later, make_cast_id, mana_row, out, ranked, scalar, target_dist)

NAME = "KSante"
ID = 897
SKILL_ORDER = (0, 2, 1, 0, 0, 3, 0, 1, 0, 1, 3, 1, 1, 2, 2, 3, 2, 2)    # default ranks per level: Q > W > E
TRAITS = ChampionTraits(has_immobilize=True, resource="mana")          # rune legality: Q3 / W stun, R root
UNIT_TARGET_RANGE = (0.0, 0.0, 550.0, 350.0)        # Footwork on an ally; All Out (center to edge)

# Dauntless Instinct.
P_MARK = scalar(NAME, "Passive", "MarkDuration")                       # 4 s
P_FLAT = scalar(NAME, "Passive", "FlatDamage")                         # 12
P_PCT_MIN = scalar(NAME, "Passive", "MarkDamagePercentMin")            # 1% max health at level 1 ...
P_PCT_MAX = scalar(NAME, "Passive", "MarkDamagePercentMax")            # ... to 2% at 18
P_AO_BASE = scalar(NAME, "Passive", "MaxHealthDamagePercentBase")      # All Out: 1% max health
P_AO_PER = scalar(NAME, "Passive", "MaxHealthDamageResistRatio")       # +1% per 100 bonus armor and per 100 bonus MR

# Ntofo Strikes.
_QC = spell(NAME, "Q")["calculations"]["BaseDamage"]["mFormulaParts"]
Q_ARMOR = _QC[1]["mCoefficient"]                                       # 0.4 bonus armor
Q_MR = _QC[2]["mCoefficient"]                                          # 0.4 bonus MR
Q_LENGTH = scalar(NAME, "Q", "RectangleLength")                        # 465 from K'Sante's center
Q_HALF = scalar(NAME, "Q", "RectangleWidth") / 2.0                     # 100 wide
Q_SLOW = scalar(NAME, "Q", "SlowPercent")                              # 80%
Q_SLOW_TIME = scalar(NAME, "Q", "SlowDuration")                        # 0.5 s
Q_STACK_TIME = scalar(NAME, "Q", "RecastWindow")                       # 6 s
Q_CT_MAX = scalar(NAME, "Q", "MaxCastTime")                            # 0.45 s ...
Q_CT_MIN = scalar(NAME, "Q", "MinCastTime")                            # ... to 0.35 s
Q_CD_MAX = scalar(NAME, "Q", "BaseCD")                                 # 3.5 s ...
Q_CD_MIN = scalar(NAME, "Q", "MinCD")                                  # ... to 2 s
Q_RES_CAP = scalar(NAME, "Q", "DefenseCapforCooldown")                 # at 120 bonus armor + MR
Q_AO_CDR = scalar(NAME, "Q", "RCooldownReduction")                     # -33.3% during All Out
Q3_RANGE = 825.0                       # KSanteQ3Missile castRange (mProjectTargetToCastRange)
Q3_SPEED = 1600.0                      # KSanteQ3Missile movementComponent mSpeed (spell missileSpeed 1800)
Q3_HALF = 70.0                         # KSanteQ3Missile mMissileWidth (half-width, as Aatrox / Yone; wiki width 100)
Q3_UP = scalar(NAME, "Q", "KnockupDuration")                           # 0.65 s airborne pull
Q3_PULL_SPEED = scalar(NAME, "Q", "PullSpeed")                         # 500
Q3_PULL = scalar(NAME, "Q", "PullDistance")                            # 300
Q3_STUN = 0.8                          # KSanteQ3 StunDuration; wiki test (the Q record's 1.0 is the tooltip's)

# Path Maker.
W_MIN = scalar(NAME, "W", "MinChargeTime")                             # 0.4 s
W_MAX = scalar(NAME, "W", "MaxDuration")                               # 1 s
W_FULL = scalar(NAME, "W", "TimeToFullCharge")                         # 0.9 s
W_LEN_MIN = scalar(NAME, "W", "MinDashBase")                           # 100 ...
W_LEN_MAX = scalar(NAME, "W", "MaxDashBase")                           # ... to 450
W_SPEED = scalar(NAME, "W", "DashSpeedBase")                           # 1400 (replay 1400; wiki 1500)
W_SPEED_AO = scalar(NAME, "W", "DashSpeedAO")                          # 1800 (replay)
W_VICTIM = scalar(NAME, "W", "VictimDashMod")                          # victims move at 1.3x (replay)
W_STUN_MIN = scalar(NAME, "W", "MinKnockbackDuration")                 # 0.5 s ...
W_STUN_MAX = scalar(NAME, "W", "MaxKnockbackDuration")                 # ... to 1.75 s
W_TRUE_MIN = scalar(NAME, "W", "RDamageIncreaseMin")                   # All Out: 10% ...
W_TRUE_MAX = scalar(NAME, "W", "RDamageIncreaseMax")                   # ... to 80% as bonus true damage
W_DR = scalar(NAME, "W", "DamageReduction")                            # 30% (26.20: 25%)
W_DR_AO = scalar(NAME, "W", "RDamageReduction")                        # 75% (26.20: 70%)
W_PCT = scalar(NAME, "W", "MaxHealthDamage")                           # 8% max health ...
W_PCT_PER = scalar(NAME, "W", "MaxHealthDamageResistRatio")            # ... +2% per 100 bonus armor and MR
W_AO_EXTEND = 2.0                      # wiki: All Out lasts 2 s more if it would end during the charge

# Footwork.
E_RANGE = scalar(NAME, "E", "FreeTargetRangeBase")                     # 250 (also in All Out: PATCH 14.19, replay)
E_SPEED = scalar(NAME, "E", "FreeTargetSpeedBase")                     # 550 + move speed (replay; wiki 500)
E_SPEED_AO = scalar(NAME, "E", "FreeTargetSpeedAO")                    # 1250 + move speed (replay)
E_ALLY_SPEED = scalar(NAME, "E", "AllySpeedBase")                      # 1100 + move speed (replay)
E_ALLY_SPEED_AO = scalar(NAME, "E", "AllySpeedAO")                     # 1400 + move speed
E_ALLY_RANGE = 550.0                   # KSanteEAllyDash castRange
E_SHIELD_HP = scalar(NAME, "E", "ShieldHealthRatio")                   # 13.5% bonus health
E_SHIELD_TIME = scalar(NAME, "E", "ShieldDuration")                    # 2 s
E_AO_CD = scalar(NAME, "E", "CooldownModAO")                           # x0.5 during All Out

# All Out.
R_RANGE = 350.0                        # center to edge (castRange; wiki 300 + 50 forgiveness)
R_CAST = 0.4                           # mCastTime
R_ROOT = 0.5                           # wiki: unaffected by tenacity (PATCH 14.15)
R_TACKLE = scalar(NAME, "R", "TackleRange")                            # 300 (replay 300)
R_WALL_CHECK = 350.0                   # wiki: terrain within 350 of the target in the cast direction
R_WALL_OUT = 450.0                     # wiki: pulled 450 past the far edge of the terrain
R_WALL_MAX = scalar(NAME, "R", "MaxWallCarrySR")                       # 5000 on Summoner's Rift
R_STAND = 35.0                         # the far side must fit a champion (pathing radius 35; replay-checked)
R_LAND = scalar(NAME, "R", "KSanteLandingOffset")                      # K'Sante lands 250 beyond the target (replay)
R_D_MIN = scalar(NAME, "R", "DistanceRatioMin")                        # pull speed 900 at <= 600 ...
R_D_MAX = scalar(NAME, "R", "DistanceRatioMax")                        # ... to 1700 at >= 2000 (replay 900 / 1716)
R_V_MIN = scalar(NAME, "R", "CarrySpeedRatioMin")
R_V_MAX = scalar(NAME, "R", "CarrySpeedRatioMax")
R_STUN = scalar(NAME, "R", "NoWallStunDuration")                       # 0.25 s (wiki 0.3)
R_WALL_STUN = scalar(NAME, "R", "WallStunDuration")                    # 0.5 s
R_WALL_UP = 0.25                       # wiki: airborne 0.25 s after the pull over terrain ...
R_SLAM_DELAY = 0.125                   # ... struck 0.125 s into it
R_VANISH_BLINK = 0.25                  # wiki: K'Sante vanishes ~0.25 s after the cast (replay 0.3)
R_SLAM_HP = spell(NAME, "R")["calculations"]["TotalDamageSlamDown"]["mFormulaParts"][1]["mCoefficient"]   # 5%
AO_TIME = scalar(NAME, "R", "AllOutDuration")                          # 15 s
AO_HP = scalar(NAME, "R", "HealthLost")                                # threshold at 65% max health
AO_RES = scalar(NAME, "R", "DefensesLost")                             # base resists -85% of bonus resists
AO_VAMP = scalar(NAME, "R", "Omnivamp")                                # 20%
AO_PEN = scalar(NAME, "R", "ArmorPenPercent")                          # 50% bonus armor penetration

AOE = D.TAG_AOE | D.TAG_ACTIVE_SPELL
R_FLAGS = D.TAG_ACTIVE_SPELL | D.PROP_ULTIMATE
P_FLAGS = D.TAG_PROC


class State(NamedTuple):
    mark_until: Any         # (C, N) Dauntless Instinct mark
    p_on: Any               # (C,) bool: the attack launched at a marked target consumes the mark on hit
    q_stacks: Any           # (C,) int32 Ntofo Strikes stacks
    q_stacks_until: Any
    q_pending: Any          # (C,) bool: in Q's cast time
    q_fire_at: Any
    q_three: Any            # (C,) bool: the pending cast is the shockwave
    q_dx: Any
    q_dy: Any
    q_cast_id: Any
    q3_fly: Any             # (C,) bool: the shockwave is travelling
    q3_ox: Any
    q3_oy: Any
    q3_dx: Any
    q3_dy: Any
    q3_dist: Any
    q3_hit: Any             # (C, N) bool: already struck
    q3_bonus_used: Any      # (C,) bool: All Out's ability bonus went to its first target
    q3_cast_id: Any
    q3_pull_until: Any      # (C, N)
    q3_px: Any              # (C, N) pull point
    q3_py: Any
    w_stage: Any            # (C,) int32 0 idle, 1 charging, 2 dashing
    w_from: Any             # charge start
    w_dx: Any
    w_dy: Any
    w_release: Any          # (C,) bool: a recast before the minimum charge (released at 0.4 s)
    w_charge: Any           # charge time at release
    w_ao: Any               # (C,) bool: the dash is All Out's
    w_dash_until: Any
    w_ex: Any               # dash end
    w_ey: Any
    w_hit: Any              # (C, N) bool
    w_bonus_used: Any
    w_cast_id: Any
    w_carry_until: Any      # (C, N)
    w_cx: Any               # (C, N) carry point
    w_cy: Any
    e_shield_until: Any
    r_pending: Any          # (C,) bool: in All Out's cast time
    r_fire_at: Any
    r_target: Any           # (C,) int32
    r_seq: Any
    r_cast_id: Any
    r_dx: Any               # cast direction (K'Sante to target at cast)
    r_dy: Any
    r_wall: Any             # (C,) bool: the pull crosses terrain
    r_pull_until: Any
    r_px: Any               # pull end
    r_py: Any
    r_speed: Any
    r_hit_at: Any           # the pull's damage
    r_slam_at: Any          # over terrain: the strike
    r_stun_at: Any
    r_blink_at: Any         # K'Sante's blink (and All Out's start)
    r_lx: Any
    r_ly: Any
    r_vanish_until: Any
    ao_until: Any           # All Out
    ao_from: Any
    ao_mana: Any            # mana before All Out (after its cost), restored after
    ao_cap: Any             # the health threshold
    ao_restore: Any         # (C,) bool: restore the mana this tick


def init(n_champions: int, n_units: int) -> State:
    c, n = n_champions, n_units
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    i = jnp.zeros((c,), jnp.int32)
    zn = jnp.zeros((c, n), jnp.float32)
    fn = jnp.zeros((c, n), bool)
    never = z - NEVER
    return State(zn - NEVER, f, i, never,
                 f, never, f, z, z, i,
                 f, z, z, z, z, z, fn, f, i, zn - NEVER, zn, zn,
                 i, never, z, z, f, z, f, never, z, z, fn, f, i, zn - NEVER, zn, zn,
                 never,
                 f, never, i - 1, i, i, z, z, f, never, z, z, z, never, never, never, never, z, z, never,
                 never, never, z, z, f)


def _mine(kctx) -> Any:
    return kctx.champion_id == ID


def _unit_dir(dx, dy):
    d = jnp.sqrt(dx * dx + dy * dy)
    ok = d > 1e-3
    return jnp.where(ok, dx / jnp.maximum(d, 1e-3), 1.0), jnp.where(ok, dy / jnp.maximum(d, 1e-3), 0.0), d


def _bonus_res(kctx) -> tuple[Any, Any]:
    c = kctx.unit.shape[0]
    z = jnp.zeros((c,), jnp.float32)
    return (z if kctx.bonus_armor is None else f32(kctx.bonus_armor)), (z if kctx.bonus_mr is None
                                                                       else f32(kctx.bonus_mr))


def all_out(state: State, now) -> Any:
    return (now >= state.ao_from) & (now < state.ao_until)


def q_cast_time(bonus_res) -> Any:
    """0.45 s, 1/1200 s shorter per point of bonus armor + MR up to 120."""
    r = jnp.clip(bonus_res, 0.0, Q_RES_CAP)
    return Q_CT_MAX - (Q_CT_MAX - Q_CT_MIN) * r / Q_RES_CAP


def q_cooldown(bonus_res, in_all_out) -> Any:
    """3.5 s, 0.0125 s shorter per point of bonus armor + MR up to 120; a third shorter during All Out."""
    r = jnp.clip(bonus_res, 0.0, Q_RES_CAP)
    cd = Q_CD_MAX - (Q_CD_MAX - Q_CD_MIN) * r / Q_RES_CAP
    return jnp.where(in_all_out, cd * (1.0 - Q_AO_CDR), cd)


def w_length(charge) -> Any:
    return W_LEN_MIN + (W_LEN_MAX - W_LEN_MIN) * jnp.clip(charge / W_FULL, 0.0, 1.0)


def w_stun(charge) -> Any:
    """0.5 s + 0.125 s per 0.05 s charged past 0.4 s, up to 0.9 s (1.75 s)."""
    return W_STUN_MIN + (W_STUN_MAX - W_STUN_MIN) * jnp.clip((charge - W_MIN) / (W_FULL - W_MIN), 0.0, 1.0)


def w_true_share(charge) -> Any:
    """All Out: 10% + 7% per 0.05 s charged past 0.4 s, up to 0.9 s (80%)."""
    return W_TRUE_MIN + (W_TRUE_MAX - W_TRUE_MIN) * jnp.clip((charge - W_MIN) / (W_FULL - W_MIN), 0.0, 1.0)


def r_pull_speed(distance) -> Any:
    """900 at <= 600 units to 1700 at >= 2000 (client ratios; replay 900 for the 300-unit pull, 1716 at ~2000)."""
    f = jnp.clip((distance - R_D_MIN) / (R_D_MAX - R_D_MIN), 0.0, 1.0)
    return R_V_MIN + (R_V_MAX - R_V_MIN) * f


def mark_damage(level, max_hp, kind) -> Any:
    """12 + 1-2% (level 1-18) of max health; the health part is at least 10 + 5 x level against minions and at most
    15 + 5 x level against monsters."""
    lv = jnp.asarray(level, jnp.float32)
    pct = P_PCT_MIN + (P_PCT_MAX - P_PCT_MIN) * jnp.clip((lv - 1.0) / 17.0, 0.0, 1.0)
    part = pct * max_hp
    part = jnp.where(kind == KIND_MINION, jnp.maximum(part, 10.0 + 5.0 * lv), part)
    part = jnp.where(kind == KIND_MONSTER, jnp.minimum(part, 15.0 + 5.0 * lv), part)
    return P_FLAT + part


def ao_bonus_share(bonus_armor, bonus_mr) -> Any:
    """All Out's bonus physical damage: 1% (+1% per 100 bonus armor) (+1% per 100 bonus MR) of max health."""
    return P_AO_BASE + P_AO_PER * (bonus_armor + bonus_mr)


def _base_cd(kctx, state) -> Any:
    ba, bm = _bonus_res(kctx)
    ao = all_out(state, kctx.now)
    row = cooldown_row(NAME, kctx.ranks)
    row = row.at[:, 0].set(q_cooldown(ba + bm, ao)).at[:, 2].set(jnp.where(ao, row[:, 2] * E_AO_CD, row[:, 2]))
    return jnp.where(_mine(kctx)[:, None], row, 0.0)


def _first(mask, dist) -> Any:
    """(C, N) the nearest unit of each row's ``mask`` (All Out's ability bonus: first target only)."""
    k = jnp.argmin(jnp.where(mask, dist, jnp.inf), axis=1)
    return mask & (jnp.arange(mask.shape[1])[None, :] == k[:, None])


def cast(state: State, kctx, units, order) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    r = kctx.ranks
    now = f32(kctx.now)
    ao = all_out(state, now)
    rooted = jnp.zeros_like(m) if kctx.rooted is None else kctx.rooted
    w_dash = state.w_stage == 2
    busy = state.q_pending | state.r_pending | (state.w_stage == 1)
    free = m & kctx.alive & ~kctx.stunned & ~kctx.silenced
    cost = jnp.where(ao[:, None] & (jnp.arange(4) < 3)[None, :], 0.0, mana_row(NAME, r))   # free basics in All Out
    ready = (r > 0) & (kctx.cooldowns <= 0) & (kctx.mana[:, None] >= cost)
    want = lambda s: (order.slot == s) & free          # noqa: E731
    ux, uy, dist = _unit_dir(order.x - kctx.x, order.y - kctx.y)
    ba, bm = _bonus_res(kctx)
    stacks = jnp.where(now <= state.q_stacks_until, state.q_stacks, 0)

    # Ntofo Strikes (also during Footwork's and Path Maker's dashes).
    q = want(0) & ready[:, 0] & ~busy
    three = q & (stacks >= 2)
    # Path Maker: the charge (the recast is read in ``periodic``).
    w = want(1) & ready[:, 1] & ~busy & ~w_dash & ~rooted & (state.w_stage == 0)
    # Footwork: to a location, or to an allied champion or minion within 550.
    t = jnp.clip(order.target, 0, n - 1)
    ally = (order.target >= 0) & (order.target != kctx.unit) & units.alive[t] & units.targetable[t] \
        & (units.team[t] == kctx.team) & ((units.kind[t] == KIND_CHAMPION) | (units.kind[t] == KIND_MINION)) \
        & (target_dist(kctx, units, order.target) <= E_ALLY_RANGE + units.radius[t])
    e = want(2) & ready[:, 2] & ~busy & ~w_dash & ~rooted
    # All Out (also during the dashes; not again while All Out lasts).
    foe_champ = (order.target >= 0) & units.alive[t] & units.targetable[t] & (units.kind[t] == KIND_CHAMPION) \
        & (units.team[t] != kctx.team) & (target_dist(kctx, units, order.target) <= R_RANGE + units.radius[t])
    rr = want(3) & ready[:, 3] & ~busy & ~rooted & foe_champ & ~ao & (state.r_target < 0)

    started = q | w | e | rr
    slot = jnp.where(q, 0, jnp.where(w, 1, jnp.where(e, 2, jnp.where(rr, 3, -1)))).astype(jnp.int32)
    cid = jnp.where(started, make_cast_id(kctx, jnp.maximum(slot, 0)), 0).astype(jnp.int32)

    # Footwork's dash and shield (granted at cast; not while its last shield is up).
    ms = jnp.zeros((c,), jnp.float32) if kctx.move_speed is None else kctx.move_speed
    to_ally = e & ally
    e_len = jnp.minimum(dist, E_RANGE)
    e_speed = jnp.where(to_ally, jnp.where(ao, E_ALLY_SPEED_AO, E_ALLY_SPEED), jnp.where(ao, E_SPEED_AO, E_SPEED)) + ms
    dash = Dash(e, f32(jnp.where(to_ally, units.x[t], kctx.x + ux * e_len)),
                f32(jnp.where(to_ally, units.y[t], kctx.y + uy * e_len)), f32(e_speed),
                jnp.where(to_ally, order.target, -1).astype(jnp.int32), jnp.zeros((c,), bool))
    kit_shield = jnp.zeros((c,), jnp.float32) if kctx.kit_shield is None else kctx.kit_shield
    give = e & ~((now < state.e_shield_until) & (kit_shield > 0.0))
    amount = jnp.where(give, ranked(NAME, "E", "ShieldBaseAmountFast", r[:, 2]) + E_SHIELD_HP * kctx.bonus_hp, 0.0)
    kind = jnp.full((c,), D.SHIELD_ALL + D.SOURCE_SHIFT * D.SOURCE_KIT, jnp.int32)
    ally_champ = to_ally & (units.kind[t] == KIND_CHAMPION) & (order.target < c)
    ally_row = jnp.where(ally_champ, order.target, c)                                   # row c = dropped
    ally_amt = jnp.zeros((c + 1,), jnp.float32).at[ally_row].add(jnp.where(ally_champ, amount, 0.0))[:c]
    shield = ShieldGrant(f32(jnp.stack([amount, ally_amt], -1)), jnp.stack([kind, kind], -1),
                         jnp.stack([jnp.where(give, E_SHIELD_TIME, 0.0), jnp.where(ally_amt > 0, E_SHIELD_TIME, 0.0)],
                                   -1).astype(jnp.float32), jnp.zeros((c, 2), jnp.float32))

    rx, ry, _ = _unit_dir(units.x[t] - kctx.x, units.y[t] - kctx.y)
    q_ct = q_cast_time(ba + bm)
    state = state._replace(
        q_pending=state.q_pending | q, q_fire_at=jnp.where(q, f32(now + q_ct), state.q_fire_at),
        q_three=jnp.where(q, three, state.q_three), q_dx=jnp.where(q, f32(ux), state.q_dx),
        q_dy=jnp.where(q, f32(uy), state.q_dy), q_cast_id=jnp.where(q, cid, state.q_cast_id),
        q_stacks=jnp.where(three, 0, stacks).astype(jnp.int32),
        w_stage=jnp.where(w, 1, state.w_stage).astype(jnp.int32), w_from=jnp.where(w, now, state.w_from),
        w_dx=jnp.where(w, f32(ux), state.w_dx), w_dy=jnp.where(w, f32(uy), state.w_dy),
        w_release=state.w_release & ~w, w_cast_id=jnp.where(w, cid, state.w_cast_id),
        e_shield_until=jnp.where(give, f32(now + E_SHIELD_TIME), state.e_shield_until),
        r_pending=state.r_pending | rr, r_fire_at=jnp.where(rr, later(kctx, R_CAST), state.r_fire_at),
        r_target=jnp.where(rr, order.target, state.r_target).astype(jnp.int32),
        r_seq=jnp.where(rr, units.spawn_seq[t], state.r_seq).astype(jnp.int32),
        r_cast_id=jnp.where(rr, cid, state.r_cast_id), r_dx=jnp.where(rr, f32(rx), state.r_dx),
        r_dy=jnp.where(rr, f32(ry), state.r_dy))
    # All Out's root (0.5 s, tenacity-free) from the start of the cast.
    cols = jnp.arange(n)[None, :]
    root_mask = rr[:, None] & (cols == t[:, None])
    z = jnp.zeros((c, n), jnp.float32)
    cc = CCOut(z, f32(jnp.where(root_mask, R_ROOT, 0.0)), z, z, z, z,
               jnp.where(root_mask, cid[:, None], 0).astype(jnp.int32), tenacity_free=root_mask)
    # Costs at cast; cooldowns: Q, E at cast, W after the dash, R after All Out.
    mana = jnp.sum(jnp.where(jnp.stack([q, w, e, rr], -1), cost, 0.0), axis=1)
    cd = jnp.stack([q, jnp.zeros_like(w), e, jnp.zeros_like(rr)], -1)
    lock = jnp.where(q, q_ct, jnp.where(rr, R_CAST, 0.0))
    return state, out(c, n, dash=dash, shield=shield, cc=cc, mana_cost=f32(mana), cooldown_start=cd,
                      base_cooldown=_base_cd(kctx, state), cast_started=started, cast_slot=slot, cast_id=cid,
                      cast_lockout=f32(lock), dash_stops_at_walls=e & ~to_ally)


def periodic(state: State, kctx, units) -> tuple[State, KitOut]:
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    alive = kctx.alive
    dead = ~alive
    r = kctx.ranks
    now = f32(kctx.now)
    dt = kctx.dt
    cols = jnp.arange(n)[None, :]
    z = jnp.zeros((c, n), jnp.float32)
    zb = jnp.zeros((c, n), bool)
    foe = enemies(kctx, units)
    rad = units.radius[None, :]
    my_r = units.radius[kctx.unit]
    rx, ry = units.x[None, :] - kctx.x[:, None], units.y[None, :] - kctx.y[:, None]
    dist = jnp.sqrt(rx * rx + ry * ry)
    champ = (units.kind == KIND_CHAMPION)[None, :]
    minion = (units.kind == KIND_MINION)[None, :]
    monster = (units.kind == KIND_MONSTER)[None, :]
    max_hp = units.max_hp[None, :]
    ba, bm = _bonus_res(kctx)
    ao = all_out(state, now) & m
    ao_share = ao_bonus_share(ba, bm)
    dashing = jnp.zeros_like(m) if kctx.dashing is None else kctx.dashing
    attack_to = jnp.full((c,), -1, jnp.int32)

    # Ntofo Strikes fires at the end of its cast time from K'Sante's position.
    q_due = state.q_pending & (due(kctx, state.q_fire_at) | dead)
    q_fire = q_due & alive
    q1 = q_fire & ~state.q_three
    along = rx * state.q_dx[:, None] + ry * state.q_dy[:, None]
    perp = jnp.abs(ry * state.q_dx[:, None] - rx * state.q_dy[:, None])
    q_hit = q1[:, None] & foe & (along >= -rad) & (along <= Q_LENGTH + rad) & (perp <= Q_HALF + rad)
    q_raw = ranked(NAME, "Q", "FlatDamage", r[:, 0]) + Q_ARMOR * ba + Q_MR * bm
    q_bonus = jnp.where(ao[:, None] & _first(q_hit, dist), ao_share[:, None] * max_hp, 0.0)
    p_q = D.packets(q_hit, kctx.unit[:, None], cols, q_raw[:, None] + q_bonus, D.PHYSICAL, AOE,
                    cast_id=state.q_cast_id[:, None])
    q_any = jnp.any(q_hit, axis=1)
    stacks = jnp.where(now <= state.q_stacks_until, state.q_stacks, 0)
    q_stacks = jnp.where(q_any, jnp.minimum(stacks + 1, 2), stacks)
    q_stacks_until = jnp.where(q_any, f32(now + Q_STACK_TIME), state.q_stacks_until)
    one_q = jnp.sum(q_hit & champ, axis=1) == 1
    attack_to = jnp.where(one_q, jnp.argmax(q_hit & champ, axis=1), attack_to)

    # The shockwave: leaves at the end of the cast time, strikes every enemy it passes, pulls and stuns.
    go3 = q_fire & state.q_three
    fly = (state.q3_fly | go3) & alive
    ox, oy = jnp.where(go3, kctx.x, state.q3_ox), jnp.where(go3, kctx.y, state.q3_oy)
    dx3, dy3 = jnp.where(go3, state.q_dx, state.q3_dx), jnp.where(go3, state.q_dy, state.q3_dy)
    d0 = jnp.where(go3, 0.0, state.q3_dist)
    d1 = jnp.minimum(d0 + Q3_SPEED * dt, Q3_RANGE)
    sx, sy = units.x[None, :] - ox[:, None], units.y[None, :] - oy[:, None]
    s_along = sx * dx3[:, None] + sy * dy3[:, None]
    s_perp = jnp.abs(sy * dx3[:, None] - sx * dy3[:, None])
    seen = jnp.where(go3[:, None], zb, state.q3_hit)
    q3_hit = fly[:, None] & foe & ~seen & (s_along - rad <= d1[:, None]) & (s_along + rad >= d0[:, None]) \
        & (s_perp <= Q3_HALF + rad)
    used3 = jnp.where(go3, False, state.q3_bonus_used)
    first3 = _first(q3_hit, s_along) & ~used3[:, None]
    q3_bonus = jnp.where(ao[:, None] & first3, ao_share[:, None] * max_hp, 0.0)
    q3_id = jnp.where(go3, state.q_cast_id, state.q3_cast_id)
    p_q3 = D.packets(q3_hit, kctx.unit[:, None], cols, q_raw[:, None] + q3_bonus, D.PHYSICAL, AOE,
                     cast_id=q3_id[:, None])
    q3_seen = seen | q3_hit
    q3_done = fly & (d1 >= Q3_RANGE)
    one_q3 = q3_done & (jnp.sum(q3_seen & champ, axis=1) == 1)
    attack_to = jnp.where(one_q3, jnp.argmax(q3_seen & champ, axis=1), attack_to)
    pux, puy, pd = _unit_dir(-rx, -ry)
    pull_len = jnp.clip(jnp.minimum(Q3_PULL, pd - my_r[:, None] - rad), 0.0, None)
    q3_pull_until = jnp.where(q3_hit, f32(now + dt + pull_len / Q3_PULL_SPEED), state.q3_pull_until)
    q3_px = jnp.where(q3_hit, units.x[None, :] + pux * pull_len, state.q3_px)
    q3_py = jnp.where(q3_hit, units.y[None, :] + puy * pull_len, state.q3_py)

    # Path Maker: charge, release (recast after 0.4 s, buffered before it, attack-move, or at 1 s), dash.
    charging = (state.w_stage == 1) & m
    held = now - state.w_from
    raw = jnp.full((c,), -1, jnp.int32) if kctx.raw_cast_slot is None else kctx.raw_cast_slot
    amove = jnp.zeros((c,), bool) if kctx.attack_move_cmd is None else kctx.attack_move_cmd
    press = charging & (now > state.w_from + 0.5 * dt) & ((raw == 1) | amove)
    w_release = state.w_release | (press & (held < W_MIN - 1e-6))
    go_w = charging & alive & (held >= W_MIN - 1e-6) & (press | w_release | (held >= W_MAX - 1e-6))
    w_cancel = charging & dead
    charge = jnp.minimum(held, W_MAX)
    length = w_length(charge)
    wall = kctx.terrain
    if wall is not None:
        hit_w, wx, wy = M.first_wall(kctx.x, kctx.y, state.w_dx, state.w_dy, f32(length + my_r), kctx.team, wall)
        wall_d = jnp.sqrt((wx - kctx.x) ** 2 + (wy - kctx.y) ** 2)
        length = jnp.where(hit_w, jnp.clip(wall_d - my_r, 0.0, length), length)
    ex, ey = kctx.x + state.w_dx * length, kctx.y + state.w_dy * length
    w_ao_now = jnp.where(go_w, ao, state.w_ao)
    w_speed = jnp.where(w_ao_now, W_SPEED_AO, W_SPEED)
    w_dash = Dash(go_w, f32(ex), f32(ey), f32(w_speed), jnp.full((c,), -1, jnp.int32), jnp.zeros((c,), bool))
    w_charge = jnp.where(go_w, charge, state.w_charge)
    w_dash_until = jnp.where(go_w, f32(now + dt + length / w_speed), state.w_dash_until)
    w_ex, w_ey = jnp.where(go_w, ex, state.w_ex), jnp.where(go_w, ey, state.w_ey)
    in_w = ((state.w_stage == 2) | go_w) & alive & m
    w_seen = jnp.where(go_w[:, None], zb, state.w_hit)
    w_hit = in_w[:, None] & foe & ~w_seen & (dist <= my_r[:, None] + rad)
    w_used = jnp.where(go_w, False, state.w_bonus_used)
    w_pct = W_PCT + W_PCT_PER * (ba + bm)
    w_raw = ranked(NAME, "W", "BaseDamage", r[:, 1])[:, None] + w_pct[:, None] * max_hp
    w_raw = jnp.where(minion, jnp.maximum(w_raw, ranked(NAME, "W", "MinimumMinionDamage", r[:, 1])[:, None]), w_raw)
    w_raw = jnp.where(monster & ~w_ao_now[:, None],
                      jnp.minimum(w_raw, ranked(NAME, "W", "MaxMonsterDamage", r[:, 1])[:, None]), w_raw)
    w_first = _first(w_hit, dist) & ~w_used[:, None]
    w_phys = w_raw + jnp.where(ao[:, None] & w_first, ao_share[:, None] * max_hp, 0.0)
    w_true = jnp.where(w_ao_now[:, None], w_true_share(w_charge)[:, None] * w_raw, 0.0)
    p_w = D.concat_packets(
        D.packets(w_hit, kctx.unit[:, None], cols, w_phys, D.PHYSICAL, AOE, cast_id=state.w_cast_id[:, None]),
        D.packets(w_hit & w_ao_now[:, None], kctx.unit[:, None], cols, w_true, D.TRUE, AOE,
                  cast_id=state.w_cast_id[:, None]))
    carry = w_hit & ~w_ao_now[:, None]
    gap = my_r[:, None] + rad
    cpx, cpy = w_ex[:, None] + state.w_dx[:, None] * gap, w_ey[:, None] + state.w_dy[:, None] * gap
    c_len = jnp.sqrt((cpx - units.x[None, :]) ** 2 + (cpy - units.y[None, :]) ** 2)
    w_carry_until = jnp.where(carry, f32(now + dt + c_len / (W_VICTIM * w_speed[:, None])), state.w_carry_until)
    w_cx, w_cy = jnp.where(carry, cpx, state.w_cx), jnp.where(carry, cpy, state.w_cy)
    w_stun_m = jnp.where(carry, w_stun(w_charge)[:, None], 0.0)
    w_all = w_seen | w_hit
    w_end = (state.w_stage == 2) & ((now >= state.w_dash_until - 1e-6) | dead)
    one_w = w_end & alive & (jnp.sum(w_all & champ, axis=1) == 1)
    attack_to = jnp.where(one_w, jnp.argmax(w_all & champ, axis=1), attack_to)
    w_stage = jnp.where(go_w, 2, jnp.where(w_end | w_cancel, 0, state.w_stage))

    # All Out: at the end of the cast time the target is pulled (over terrain when it is within 350 in the cast
    # direction and has a far side within 5000), K'Sante blinks 250 beyond the pull's end and goes All Out.
    rt = jnp.clip(state.r_target, 0, n - 1)
    r_valid = (state.r_target >= 0) & units.alive[rt] & (units.spawn_seq[rt] == state.r_seq)
    r_due = state.r_pending & (due(kctx, state.r_fire_at) | dead)
    r_fire = r_due & alive & r_valid
    tx, ty = units.x[rt], units.y[rt]
    if wall is not None:
        found, entry, exit_ = M.wall_exit(tx, ty, state.r_dx, state.r_dy, jnp.full((c,), R_WALL_CHECK, jnp.float32),
                                          jnp.full((c,), R_WALL_MAX, jnp.float32), units.team[rt], wall, radius=R_STAND)
        again, nx, ny = M.first_wall(tx + state.r_dx * exit_, ty + state.r_dy * exit_, state.r_dx, state.r_dy,
                                     jnp.full((c,), R_WALL_OUT, jnp.float32), units.team[rt], wall)
        past = jnp.where(again, jnp.minimum(jnp.sqrt((nx - tx) ** 2 + (ny - ty) ** 2) - exit_ - R_STAND, R_WALL_OUT),
                         R_WALL_OUT)                                   # stops short of a second wall
        hit_short, sx_, sy_ = M.first_wall(tx, ty, state.r_dx, state.r_dy, jnp.full((c,), R_TACKLE, jnp.float32),
                                           units.team[rt], wall)
        short = jnp.where(hit_short, jnp.sqrt((sx_ - tx) ** 2 + (sy_ - ty) ** 2), R_TACKLE)
    else:
        found, exit_, past, short = jnp.zeros((c,), bool), jnp.zeros((c,), jnp.float32), 0.0, R_TACKLE
    over = found
    p_len = jnp.where(over, jnp.minimum(exit_ + jnp.maximum(past, 0.0), R_WALL_MAX), short)
    speed = r_pull_speed(p_len)
    p_time = p_len / speed
    px, py = tx + state.r_dx * p_len, ty + state.r_dy * p_len
    t_end = now + p_time
    r_up = jnp.where(r_fire, jnp.where(over, p_time + R_WALL_UP, p_time), 0.0)
    r_mask = r_fire[:, None] & (cols == rt[:, None])
    r_pull_until = jnp.where(r_fire, f32(now + dt + p_time), state.r_pull_until)
    r_hit_at = jnp.where(r_fire, f32(jnp.where(over, now + exit_ / speed, t_end)), state.r_hit_at)
    r_slam_at = jnp.where(r_fire, jnp.where(over, f32(t_end + R_SLAM_DELAY), NEVER), state.r_slam_at)
    r_stun_at = jnp.where(r_fire, f32(jnp.where(over, t_end + R_WALL_UP, t_end)), state.r_stun_at)
    r_blink_at = jnp.where(r_fire, f32(jnp.where(over, now + R_VANISH_BLINK, now)), state.r_blink_at)
    r_lx = jnp.where(r_fire, f32(px + state.r_dx * R_LAND), state.r_lx)
    r_ly = jnp.where(r_fire, f32(py + state.r_dy * R_LAND), state.r_ly)
    r_vanish_until = jnp.where(r_fire & over, f32(t_end), state.r_vanish_until)
    r_wall = jnp.where(r_fire, over, state.r_wall)
    r_cancel = r_due & ~r_fire
    # Its events.
    rank_r = r[:, 3]
    tgt_ok = (state.r_target >= 0) & units.alive[rt] & (units.spawn_seq[rt] == state.r_seq)
    hit_now = m & tgt_ok & due(kctx, state.r_hit_at) & (state.r_hit_at > -NEVER / 2)
    slam_now = m & tgt_ok & due(kctx, state.r_slam_at) & (state.r_slam_at < NEVER / 2)
    stun_now = m & tgt_ok & due(kctx, state.r_stun_at) & (state.r_stun_at > -NEVER / 2)
    on_t = lambda b: b[:, None] & (cols == rt[:, None])                 # noqa: E731
    r_bonus = ao_share[:, None] * max_hp
    p_r = D.concat_packets(
        D.packets(on_t(hit_now), kctx.unit[:, None], cols,
                  ranked(NAME, "R", "BaseDamage", rank_r)[:, None] + jnp.where(on_t(hit_now & ao), r_bonus, 0.0),
                  D.PHYSICAL, R_FLAGS, cast_id=state.r_cast_id[:, None]),
        D.packets(on_t(slam_now), kctx.unit[:, None], cols,
                  (ranked(NAME, "R", "SlamDownStrikeDamage", rank_r) + R_SLAM_HP * kctx.bonus_hp)[:, None]
                  + jnp.where(on_t(slam_now & ao), r_bonus, 0.0), D.PHYSICAL, R_FLAGS,
                  cast_id=state.r_cast_id[:, None]))
    r_stun = jnp.where(on_t(stun_now), jnp.where(state.r_wall, R_WALL_STUN, R_STUN)[:, None], 0.0)
    attack_to = jnp.where(stun_now & state.r_wall, rt, attack_to)
    blink_now = m & alive & due(kctx, state.r_blink_at) & (state.r_blink_at > -NEVER / 2) & ~r_fire \
        | (r_fire & ~over)
    lx_now = jnp.where(r_fire, r_lx, state.r_lx)
    ly_now = jnp.where(r_fire, r_ly, state.r_ly)
    blink = Dash(blink_now, f32(lx_now), f32(ly_now), jnp.full((c,), jnp.inf, jnp.float32),
                 jnp.full((c,), -1, jnp.int32), jnp.ones((c,), bool))
    r_blink_at = jnp.where(blink_now, -NEVER, r_blink_at)
    r_hit_at = jnp.where(hit_now, -NEVER, r_hit_at)
    r_slam_at = jnp.where(slam_now, NEVER, r_slam_at)
    r_stun_at = jnp.where(stun_now, -NEVER, r_stun_at)
    # A target lost mid-pull drops the remaining events (K'Sante still blinks and goes All Out).
    lost = (state.r_target >= 0) & ~state.r_pending & ~r_fire & ~tgt_ok
    r_hit_at = jnp.where(lost, -NEVER, r_hit_at)
    r_slam_at = jnp.where(lost, NEVER, r_slam_at)
    r_stun_at = jnp.where(lost, -NEVER, r_stun_at)
    # Dying before the blink: no All Out; the cooldown starts.
    r_died = m & dead & (r_blink_at > -NEVER / 2)
    r_blink_at = jnp.where(r_died, -NEVER, r_blink_at)
    r_over = (state.r_target >= 0) & ~state.r_pending & ~r_fire & (r_hit_at < -NEVER / 2) \
        & (r_slam_at > NEVER / 2) & (r_stun_at < -NEVER / 2) & (r_blink_at < -NEVER / 2)
    r_target = jnp.where(r_cancel | r_over | dead, -1, state.r_target).astype(jnp.int32)

    # Going All Out (at the blink): health threshold, mana refill, Q stacks reset (Q's cooldown reset at 2
    # stacks), Path Maker's cooldown refreshed. It lasts 15 s (+2 s if it would end during the charge).
    enter = blink_now & (state.ao_until < now)
    ao_from = jnp.where(enter, now, state.ao_from)
    ao_until = jnp.where(enter, f32(now + AO_TIME), state.ao_until)
    ao_until = jnp.where(charging & (ao_until > now) & (ao_until <= now + dt) & ~enter,
                         f32(ao_until + W_AO_EXTEND), ao_until)
    ao_mana = jnp.where(enter, f32(kctx.mana), state.ao_mana)
    ao_cap = jnp.where(enter, f32((1.0 - AO_HP) * kctx.max_hp), state.ao_cap)
    q_reset = enter & (q_stacks >= 2)
    q_stacks = jnp.where(enter, 0, q_stacks)
    ao_end = m & (state.ao_until > -NEVER / 2) & ~enter & (due(kctx, ao_until) | dead) & (now >= state.ao_from)
    ao_restore = ao_end & alive
    ao_until = jnp.where(ao_end, -NEVER, ao_until)
    ao_from = jnp.where(ao_end, NEVER, ao_from)

    cooldowns = jnp.stack([jnp.zeros_like(m), w_end & alive & m, jnp.zeros_like(m), ao_end | r_died], -1)
    state = state._replace(
        q_pending=state.q_pending & ~q_due, q_stacks=q_stacks.astype(jnp.int32), q_stacks_until=f32(q_stacks_until),
        q3_fly=fly & ~q3_done, q3_ox=f32(ox), q3_oy=f32(oy), q3_dx=f32(dx3), q3_dy=f32(dy3), q3_dist=f32(d1),
        q3_hit=q3_seen, q3_bonus_used=used3 | jnp.any(first3, axis=1), q3_cast_id=q3_id.astype(jnp.int32),
        q3_pull_until=f32(q3_pull_until), q3_px=f32(q3_px), q3_py=f32(q3_py),
        w_stage=w_stage.astype(jnp.int32), w_release=w_release & ~go_w & ~w_cancel, w_charge=f32(w_charge),
        w_ao=w_ao_now, w_dash_until=f32(w_dash_until), w_ex=f32(w_ex), w_ey=f32(w_ey), w_hit=w_all,
        w_bonus_used=w_used | jnp.any(w_first, axis=1), w_carry_until=f32(w_carry_until), w_cx=f32(w_cx),
        w_cy=f32(w_cy),
        r_pending=state.r_pending & ~r_due, r_target=r_target, r_wall=r_wall, r_pull_until=f32(r_pull_until),
        r_px=f32(jnp.where(r_fire, px, state.r_px)), r_py=f32(jnp.where(r_fire, py, state.r_py)),
        r_speed=f32(jnp.where(r_fire, speed, state.r_speed)), r_hit_at=f32(r_hit_at), r_slam_at=f32(r_slam_at),
        r_stun_at=f32(r_stun_at), r_blink_at=f32(r_blink_at), r_lx=lx_now, r_ly=ly_now,
        r_vanish_until=f32(jnp.where(dead, -NEVER, r_vanish_until)),
        ao_until=f32(ao_until), ao_from=f32(ao_from), ao_mana=f32(ao_mana), ao_cap=f32(ao_cap),
        ao_restore=ao_restore)
    # Marks from this tick's ability hits (not the strike after a pull over terrain).
    marked = q_hit | q3_hit | w_hit | on_t(hit_now)
    state = state._replace(mark_until=jnp.where(marked & m[:, None], f32(now + P_MARK), state.mark_until))
    stun = jnp.maximum(jnp.maximum(jnp.where(q3_hit, Q3_STUN, 0.0), w_stun_m), r_stun)
    up = jnp.maximum(jnp.where(q3_hit, Q3_UP, 0.0), jnp.where(r_mask, r_up[:, None], 0.0))
    slow = jnp.where(q_hit | q3_hit, Q_SLOW, 0.0)
    cid = jnp.where(q_hit, state.q_cast_id[:, None], jnp.where(q3_hit, q3_id[:, None],
                    jnp.where(w_hit, state.w_cast_id[:, None], jnp.where(r_mask | (r_stun > 0),
                                                                          state.r_cast_id[:, None], 0))))
    cc = CCOut(f32(stun), z, z, f32(up), f32(slow), jnp.where(slow > 0, Q_SLOW_TIME, 0.0).astype(jnp.float32),
               cid.astype(jnp.int32))
    dash = Dash(*(jnp.where(blink.active, b, a) for a, b in zip(w_dash, blink)))
    return state, out(c, n, packets=D.concat_packets(p_q, p_q3, p_w, p_r), cc=cc, dash=dash,
                      cooldown_start=cooldowns, base_cooldown=_base_cd(kctx, state),
                      attack_reset=q_fire & m, attack_target=jnp.where(m, attack_to, -1).astype(jnp.int32),
                      dash_stops_at_walls=go_w)


def on_attack(state: State, kctx, units, launch) -> tuple[State, KitOut]:
    """An attack started at a marked target consumes the mark when it lands, even if the mark runs out first."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    target = holder_rows(launch.target, kctx)
    t = jnp.clip(target, 0, n - 1)
    went = holder_rows(launch.launched, kctx) & _mine(kctx) & (target >= 0)
    marked = jnp.take_along_axis(state.mark_until, t[:, None], axis=1)[:, 0] > kctx.now
    return state._replace(p_on=jnp.where(went, marked, state.p_on)), out(c, n)


def on_hit(state: State, kctx, units, launch, dodging) -> tuple[State, KitOut]:
    """The mark's bonus physical damage (a proc after the attack); All Out's bonus on attacks and on the mark."""
    c, n = kctx.unit.shape[0], units.x.shape[0]
    m = _mine(kctx)
    target = holder_rows(launch.target, kctx)
    t = jnp.clip(target, 0, n - 1)
    hit = holder_rows(launch.launched, kctx) & m & kctx.alive & (target >= 0)
    land = hit & ~gather(dodging, target)
    ba, bm = _bonus_res(kctx)
    ao = all_out(state, kctx.now) & m
    bonus = ao_bonus_share(ba, bm) * units.max_hp[t]
    unit_target = ~is_structure(units.kind[t]) & (units.kind[t] != KIND_WARD)
    consume = land & state.p_on
    raw = mark_damage(kctx.level, units.max_hp[t], units.kind[t]) + jnp.where(ao & unit_target, bonus, 0.0)
    p = D.concat_packets(
        D.packets(consume, kctx.unit, t, raw, D.PHYSICAL, P_FLAGS),
        D.packets(land & ao & unit_target, kctx.unit, t, bonus, D.PHYSICAL, D.TAG_PROC | D.TAG_ON_HIT))
    mark = state.mark_until.at[jnp.arange(c), t].set(jnp.where(consume, -NEVER, state.mark_until[jnp.arange(c), t]))
    return state._replace(mark_until=mark, p_on=state.p_on & ~hit), out(c, n, packets=p)


def on_damage(state: State, kctx, units, report) -> tuple[State, KitOut]:
    return state, out(kctx.unit.shape[0], units.x.shape[0])


def on_takedown(state: State, kctx, units, kills) -> State:
    return state


def stats(state: State, kctx) -> ItemStats:
    """All Out: bonus attack speed, 20% omnivamp, 50% bonus armor penetration, base armor / MR reduced by 85% of
    the bonus armor / MR (``bonus_*`` keeps its value)."""
    on = _mine(kctx) & all_out(state, kctx.now)
    ba, bm = _bonus_res(kctx)
    w = lambda v: f32(jnp.where(on, v, 0.0))                       # noqa: E731
    return ItemStats(attack_speed=w(ranked(NAME, "R", "AttackSpeed", kctx.ranks[:, 3])), omnivamp=w(AO_VAMP),
                     percent_bonus_armor_pen=w(AO_PEN), base_armor_loss=w(AO_RES * ba), base_mr_loss=w(AO_RES * bm))


def defense(state: State, kctx) -> KitDefense:
    """Path Maker's charge: 30% damage reduction (75% during All Out)."""
    c = kctx.unit.shape[0]
    o = jnp.ones((c,), jnp.float32)
    charging = _mine(kctx) & (state.w_stage == 1)
    mult = jnp.where(charging, 1.0 - jnp.where(all_out(state, kctx.now), W_DR_AO, W_DR), 1.0)
    return KitDefense(f32(mult), jnp.zeros((c,), bool), o, jnp.zeros((c,), jnp.float32))


def attack_mods(state: State, kctx) -> KitAttackMods:
    """No attacks during Path Maker (charge and dash) or the cast times."""
    m = _mine(kctx)
    c = kctx.unit.shape[0]
    z = jnp.zeros((c,), jnp.float32)
    f = jnp.zeros((c,), bool)
    busy = m & ((state.w_stage > 0) | state.q_pending | state.r_pending)
    return KitAttackMods(z, f, busy)


def immobile(state: State, kctx) -> Any:
    """Path Maker's charge."""
    return _mine(kctx) & (state.w_stage == 1)


def displacement_immune(state: State, kctx) -> Any:
    """Path Maker's charge and All Out's cast time."""
    return _mine(kctx) & ((state.w_stage == 1) | state.r_pending)


def untargetable(state: State, champion_id, now) -> Any:
    """Vanished while All Out's target is pulled over terrain."""
    return (champion_id == ID) & (now < state.r_vanish_until) & state.r_wall


def hp_cap(state: State, kctx) -> Any:
    on = _mine(kctx) & all_out(state, kctx.now)
    return jnp.where(on, state.ao_cap, jnp.inf).astype(jnp.float32)


def mana_override(state: State, kctx) -> tuple[Any, Any]:
    """All Out: full mana decaying to 0 over its duration; the mana from before it afterwards."""
    m = _mine(kctx)
    now = kctx.now
    on = m & all_out(state, now)
    left = jnp.clip((state.ao_until - now) / jnp.maximum(state.ao_until - state.ao_from, 1e-3), 0.0, 1.0)
    restore = m & state.ao_restore
    return on | restore, f32(jnp.where(on, kctx.max_mana * left, state.ao_mana))


def displacement(state: State, kctx, n_units: int) -> tuple[Any, Any, Any, Any]:
    """(N,) the shockwave's pull (500 units/s), Path Maker's carry (1.3x the dash speed) and All Out's pull."""
    m = _mine(kctx)[:, None]
    now = kctx.now
    q3 = m & (now < state.q3_pull_until)
    wc = m & (now < state.w_carry_until)
    w_speed = jnp.where(state.w_ao, W_SPEED_AO, W_SPEED) * W_VICTIM
    cols = jnp.arange(n_units)[None, :]
    rr = m & (now < state.r_pull_until)[:, None] & (cols == state.r_target[:, None])
    act = q3 | wc | rr
    x = jnp.where(rr, state.r_px[:, None], jnp.where(wc, state.w_cx, state.q3_px))
    y = jnp.where(rr, state.r_py[:, None], jnp.where(wc, state.w_cy, state.q3_py))
    v = jnp.where(rr, state.r_speed[:, None], jnp.where(wc, w_speed[:, None], Q3_PULL_SPEED))
    k = jnp.argmax(act, axis=0)
    pick = lambda a: jnp.take_along_axis(a, k[None, :], axis=0)[0]  # noqa: E731
    return jnp.any(act, axis=0), f32(pick(x)), f32(pick(y)), f32(pick(v))


def debuffs(state: State, kctx, units) -> Debuffs:
    return neutral_debuffs(units.x.shape[0])


__all__ = ["State", "init", "cast", "periodic", "on_attack", "on_hit", "on_damage", "on_takedown", "stats",
           "defense", "attack_mods", "immobile", "displacement_immune", "untargetable", "hp_cap", "mana_override",
           "displacement", "debuffs", "all_out", "q_cast_time", "q_cooldown", "w_length", "w_stun", "w_true_share",
           "r_pull_speed", "mark_damage", "ao_bonus_share"]
