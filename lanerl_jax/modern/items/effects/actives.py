"""26.19 item actives outside the Hydra/consumable modules (ITEMS.md §11; evidence LANES_TERRAIN.md §6).

Passives of these items live in their own modules. World effects (stasis, cleanse, Rocketbelt dash, Actualizer
rates, Seeker's transform) are exposed by ``world`` for the tick to apply. ``with_aim`` stores the order's target
unit/point; without one Gunblade picks the nearest enemy champion in range, Rocketbelt dashes along the facing and
Redemption lands on the holder. 1v1 has no allies: "you and allies" effects reach the holder, ally-only actives
(Mikael's, Knight's Vow) are refused without starting a cooldown.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ...core import types as W
from ...core.damage import (CLASS_CHAMPION, CLASS_STRUCTURE, MAGIC, TAG_ACTIVE_SPELL, TAG_AOE, TAG_ITEM, TRUE,
                            concat_packets, empty_packets, has, packets)
from ..catalog import ItemStats, catalog
from .core import ActiveOut, StatusFlags, dv, effects, enemy_mask, holds, in_circle, shield_grants
from .support import locket_shield, redemption_heal

ZHONYAS, SEEKERS, SHATTERED = 3157, 2420, 2421
QUICKSILVER, MERCURIAL = 3140, 3139
YOUMUU, RANDUINS, GUNBLADE, ROCKETBELT = 3142, 3143, 3146, 3152
SHURELYA, LOCKET, REDEMPTION, ACTUALIZER = 2065, 3190, 3107, 2522
MIKAELS, KNIGHTS_VOW = 3222, 3109

ACTIVE_ITEMS = (ZHONYAS, SEEKERS, QUICKSILVER, MERCURIAL, YOUMUU, RANDUINS, GUNBLADE, ROCKETBELT,
                SHURELYA, LOCKET, REDEMPTION, ACTUALIZER)
INERT_ALLY_ONLY = (MIKAELS, KNIGHTS_VOW)
_K = {iid: k for k, iid in enumerate(ACTIVE_ITEMS)}

STASIS_S = dv(ZHONYAS, "Duration")
ROCKET_DASH = 275.0
ROCKET_RANGE = 1050.0
ROCKET_HALF_ANGLE = jnp.deg2rad(30.0)    # INFERRED L: 7 rockets at 10 degree spacing
ROCKET_PATH_WIDTH = 85.0
ROCKET_DASH_SPEED = 1500.0               # INFERRED L
GUNBLADE_RANGE = 700.0
RANDUIN_RADIUS = dv(RANDUINS, "Radius")
REDEMPTION_DELAY = 2.5


def _cd(iid: int) -> float:
    if iid == SEEKERS:
        return float("inf")
    if iid == SHURELYA:
        return dv(iid, "ActiveCooldown")
    return dv(iid, "Cooldown")


class State(NamedTuple):
    cd_until: Any           # (C, K) per active item
    stasis_until: Any       # (C,)
    shurelya_until: Any
    mercurial_until: Any
    youmuu_until: Any
    youmuu_ms: Any
    actualizer_until: Any
    red_at: Any             # (C,) pending Intervention landing time (inf none)
    red_x: Any
    red_y: Any
    aim_unit: Any           # (C,) int32 order target (-1 none)
    aim_x: Any
    aim_y: Any
    aim_set: Any            # (C,) bool: aim_x/aim_y valid
    cleanse_now: Any        # (C,) bool pulses of the last active() call
    dash_now: Any
    dash_x: Any
    dash_y: Any
    shatter_now: Any        # (C,) bool: Seeker's used -> transform to Shattered Armguard


def init(n_champions: int, n_units: int) -> State:
    c = n_champions
    z, b = jnp.zeros((c,), jnp.float32), jnp.zeros((c,), bool)
    m = jnp.full((c,), -1e9, jnp.float32)
    return State(jnp.zeros((c, len(ACTIVE_ITEMS)), jnp.float32), m, m, m, m, z, m,
                 jnp.full((c,), jnp.inf, jnp.float32), z, z, jnp.full((c,), -1, jnp.int32), z, z, b, b, b, z, z, b)


def with_aim(state: State, unit=None, x=None, y=None) -> State:
    """Store this tick's order target for targeted actives (before ``combat_tick``)."""
    c = state.aim_unit.shape[0]
    u = jnp.full((c,), -1, jnp.int32) if unit is None else jnp.asarray(unit, jnp.int32)
    if x is None:
        return state._replace(aim_unit=u, aim_set=jnp.zeros((c,), bool))
    return state._replace(aim_unit=u, aim_x=jnp.asarray(x, jnp.float32), aim_y=jnp.asarray(y, jnp.float32),
                          aim_set=jnp.ones((c,), bool))


def actualizer_amp(state: State, ctx) -> Any:
    """(C,) Mana Made Real ability damage and heal-shield power bonus (max-mana basis INFERRED M)."""
    on = ctx.now < state.actualizer_until
    return jnp.where(on, (15.0 + 0.005 * ctx.max_mana) * 0.01, 0.0)


def stats(state: State, own, ctx) -> ItemStats:
    now = ctx.now
    ms = (jnp.where(now < state.shurelya_until, dv(SHURELYA, "ActiveMoveSpeed"), 0.0)
          + jnp.where(now < state.mercurial_until, dv(MERCURIAL, "MoveSpeed"), 0.0)
          + jnp.where(now < state.youmuu_until, state.youmuu_ms, 0.0))
    return ItemStats(percent_move_speed=ms, heal_shield_power=actualizer_amp(state, ctx))


def status(state: State, own, ctx) -> StatusFlags:
    return StatusFlags(ctx.now < state.youmuu_until)


def packet_amp(state: State, own, ctx, units, p) -> Any:
    ability = has(p.flags, TAG_ACTIVE_SPELL) & ~has(p.flags, TAG_ITEM)
    src_is = p.src[:, None] == ctx.unit[None, :]
    return jnp.sum(jnp.where(src_is & ability[:, None], actualizer_amp(state, ctx)[None, :], 0.0), axis=1)


def _lerp_level(level, lo, hi):
    lv = jnp.clip(jnp.asarray(level, jnp.float32), 1.0, 18.0)
    return lo + (lv - 1.0) / 17.0 * (hi - lo)


def active(state: State, own, ctx, units, request):
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    req = jnp.asarray(request, jnp.int32)
    in_stasis = now < state.stasis_until
    cd = state.cd_until
    go = {}
    for iid in ACTIVE_ITEMS:
        k = _K[iid]
        alive_ok = jnp.ones((c,), bool) if iid == REDEMPTION else ctx.alive
        go[iid] = (req == iid) & holds(own, iid) & (now >= cd[:, k]) & ~in_stasis & alive_ok
    enemies = enemy_mask(ctx, units) & (units.cls[None, :] != CLASS_STRUCTURE)
    champs = enemies & (units.cls[None, :] == CLASS_CHAMPION)
    hx, hy, hr = units.x[ctx.unit], units.y[ctx.unit], units.radius[ctx.unit]
    dist = jnp.sqrt((units.x[None, :] - hx[:, None]) ** 2 + (units.y[None, :] - hy[:, None]) ** 2)

    # Gunblade: the aimed enemy champion, else the nearest in range.
    g_in = champs & (dist <= GUNBLADE_RANGE + hr[:, None] + units.radius[None, :])
    aim = jnp.clip(state.aim_unit, 0, n - 1)
    aimed = (state.aim_unit >= 0) & g_in[jnp.arange(c), aim]
    near = jnp.argmin(jnp.where(g_in, dist, jnp.inf), axis=1)
    g_tgt = jnp.where(aimed, aim, near).astype(jnp.int32)
    go[GUNBLADE] = go[GUNBLADE] & jnp.any(g_in, axis=1)
    used = jnp.zeros((c,), bool)
    for iid in ACTIVE_ITEMS:
        used = used | go[iid]
        cd = cd.at[:, _K[iid]].set(jnp.where(go[iid], now + _cd(iid), cd[:, _K[iid]]))

    stasis_until = jnp.where(go[ZHONYAS] | go[SEEKERS], now + STASIS_S, state.stasis_until)
    cleanse = go[QUICKSILVER] | go[MERCURIAL]
    mercurial_until = jnp.where(go[MERCURIAL], now + dv(MERCURIAL, "MSDuration"), state.mercurial_until)
    ranged = ctx.is_ranged
    y_dur = jnp.where(ranged, dv(YOUMUU, "DurationNDV") * 0.667, dv(YOUMUU, "DurationNDV"))
    y_ms = jnp.where(ranged, dv(YOUMUU, "RangedItemCalcValueB"), dv(YOUMUU, "MeleeItemCalcValueB")) * 0.01
    youmuu_until = jnp.where(go[YOUMUU], now + y_dur, state.youmuu_until)
    youmuu_ms = jnp.where(go[YOUMUU], y_ms, state.youmuu_ms)
    shurelya_until = jnp.where(go[SHURELYA], now + dv(SHURELYA, "BuffDuration"), state.shurelya_until)
    actualizer_until = jnp.where(go[ACTUALIZER], now + dv(ACTUALIZER, "Duration"), state.actualizer_until)

    all_p = []
    slow = jnp.zeros((n,), jnp.float32)
    slow_d = jnp.zeros((n,), jnp.float32)
    r_hit = in_circle(units, hx, hy, jnp.full((c,), RANDUIN_RADIUS)) & enemies & go[RANDUINS][:, None]
    r_any = jnp.any(r_hit, axis=0)
    slow = jnp.where(r_any, dv(RANDUINS, "SlowAmount"), slow)
    slow_d = jnp.where(r_any, dv(RANDUINS, "SlowDuration"), slow_d)
    g_hit = (jnp.arange(n)[None, :] == g_tgt[:, None]) & go[GUNBLADE][:, None]
    g_raw = _lerp_level(ctx.level, 175.0, 253.0) + 0.3 * ctx.ap
    all_p.append(packets(g_hit, ctx.unit[:, None], jnp.arange(n)[None, :], g_raw[:, None], MAGIC,
                         TAG_ACTIVE_SPELL | TAG_ITEM, item=GUNBLADE))
    g_any = jnp.any(g_hit, axis=0)
    stronger = g_any & (dv(GUNBLADE, "SlowAmount") > slow)
    slow = jnp.where(stronger, dv(GUNBLADE, "SlowAmount"), slow)
    slow_d = jnp.where(stronger, dv(GUNBLADE, "SlowDuration"), slow_d)
    # Rocketbelt: dash toward the aim point (else facing); rockets in an arc from the dash end, or along the path.
    ax = jnp.where(state.aim_set, state.aim_x - hx, ctx.facing_x)
    ay = jnp.where(state.aim_set, state.aim_y - hy, ctx.facing_y)
    norm = jnp.sqrt(ax ** 2 + ay ** 2)
    ux = jnp.where(norm > 1e-6, ax / jnp.maximum(norm, 1e-6), 1.0)
    uy = jnp.where(norm > 1e-6, ay / jnp.maximum(norm, 1e-6), 0.0)
    ex, ey = hx + ROCKET_DASH * ux, hy + ROCKET_DASH * uy
    rx, ry = units.x[None, :] - ex[:, None], units.y[None, :] - ey[:, None]
    rd = jnp.sqrt(rx ** 2 + ry ** 2)
    cosang = (rx * ux[:, None] + ry * uy[:, None]) / jnp.maximum(rd, 1e-6)
    arc = (rd - units.radius[None, :] <= ROCKET_RANGE) \
        & ((cosang >= jnp.cos(ROCKET_HALF_ANGLE)) | (rd <= units.radius[None, :]))
    sx, sy = units.x[None, :] - hx[:, None], units.y[None, :] - hy[:, None]
    along = jnp.clip(sx * ux[:, None] + sy * uy[:, None], 0.0, ROCKET_DASH)
    across = jnp.sqrt((sx - along * ux[:, None]) ** 2 + (sy - along * uy[:, None]) ** 2)
    path = across <= ROCKET_PATH_WIDTH + units.radius[None, :]
    rb_hit = (arc | path) & enemies & go[ROCKETBELT][:, None]
    rb_raw = dv(ROCKETBELT, "BaseDamage") + dv(ROCKETBELT, "APRatio") * ctx.ap
    all_p.append(packets(rb_hit, ctx.unit[:, None], jnp.arange(n)[None, :], rb_raw[:, None], MAGIC,
                         TAG_AOE | TAG_ACTIVE_SPELL | TAG_ITEM, item=ROCKETBELT))
    shield = jnp.where(go[LOCKET], locket_shield(ctx.level), 0.0)
    # Redemption: schedule within cast range, land after the delay.
    px = jnp.where(state.aim_set, state.aim_x, hx)
    py = jnp.where(state.aim_set, state.aim_y, hy)
    dx, dy = px - hx, py - hy
    dd = jnp.sqrt(dx ** 2 + dy ** 2)
    scale = jnp.minimum(1.0, dv(REDEMPTION, "CastRange") / jnp.maximum(dd, 1e-6))
    red_at = jnp.where(go[REDEMPTION], now + REDEMPTION_DELAY, state.red_at)
    red_x = jnp.where(go[REDEMPTION], hx + dx * scale, state.red_x)
    red_y = jnp.where(go[REDEMPTION], hy + dy * scale, state.red_y)
    land = now >= red_at
    area = in_circle(units, red_x, red_y, jnp.full((c,), dv(REDEMPTION, "AOESize")))
    red_hit = area & champs & land[:, None]
    all_p.append(packets(red_hit, ctx.unit[:, None], jnp.arange(n)[None, :],
                         dv(REDEMPTION, "DamageToChampions") * units.max_hp[None, :], TRUE,
                         TAG_AOE | TAG_ACTIVE_SPELL | TAG_ITEM, item=REDEMPTION))
    self_in = area[jnp.arange(c), ctx.unit] & ctx.alive & land
    heal = jnp.where(self_in, redemption_heal(ctx.level), 0.0)
    red_at = jnp.where(land, jnp.inf, red_at)

    new = State(cd, stasis_until, shurelya_until, mercurial_until, youmuu_until, youmuu_ms, actualizer_until,
                red_at, red_x, red_y, state.aim_unit, state.aim_x, state.aim_y, jnp.zeros((c,), bool),
                cleanse, go[ROCKETBELT], ex, ey, go[SEEKERS])
    eff = effects(c, n, packets=concat_packets(empty_packets(0), *all_p), slow=slow, slow_duration=slow_d,
                  heal=heal, shields=shield_grants(shield, duration=dv(LOCKET, "ShieldDuration"), decay_hold=0.0),
                  attack_reset=go[ROCKETBELT])
    out = ActiveOut(used, jnp.zeros((c,), jnp.float32), jnp.ones((c,), bool), go[ROCKETBELT])
    return new, eff, out


class ActiveWorld(NamedTuple):
    """World effects of item actives (C,)."""
    stasis: Any             # untargetable, invulnerable, no move/attack/cast/summoner/item
    stasis_until: Any
    cleanse: Any            # pulse: remove all CC except airborne
    dash: W.Dash            # Rocketbelt (terrain-blocked, not a blink)
    mana_cost_mult: Any     # Actualizer
    basic_cd_rate: Any      # Actualizer: Q/W/E cooldowns tick faster
    transform: tuple        # (from_row, to_row, do): Seeker's -> Shattered Armguard


def world(state: State, now) -> ActiveWorld:
    on = now < state.actualizer_until
    c = state.stasis_until.shape[0]
    cat = catalog()
    dash = W.Dash(state.dash_now, state.dash_x, state.dash_y, jnp.full((c,), ROCKET_DASH_SPEED, jnp.float32),
                  jnp.full((c,), -1, jnp.int32), jnp.zeros((c,), bool))
    return ActiveWorld(now < state.stasis_until, state.stasis_until, state.cleanse_now, dash,
                       jnp.where(on, 1.0 + dv(ACTUALIZER, "ManaCostIncrease"), 1.0),
                       jnp.where(on, 1.0 + dv(ACTUALIZER, "CooldownTick"), 1.0),
                       (jnp.full((c,), cat.row(SEEKERS), jnp.int32), jnp.full((c,), cat.row(SHATTERED), jnp.int32),
                        state.shatter_now))


def request_allowed(request, *, disabled, in_stasis=None):
    """Zero the requests a champion cannot make: only cleanse items while ``disabled`` (INFERRED M; silence
    does not block items), nothing in stasis (wiki Zhonya's)."""
    req = jnp.asarray(request, jnp.int32)
    qss = (req == QUICKSILVER) | (req == MERCURIAL)
    ok = ~jnp.asarray(disabled, bool) | qss
    if in_stasis is not None:
        ok = ok & ~jnp.asarray(in_stasis, bool)
    return jnp.where(ok, req, 0)


COVERAGE: dict = {}      # passives belong to the items' own modules
