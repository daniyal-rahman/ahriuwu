"""Hydra line and Stridebreaker: Cleave passives and all four actives (ITEMS.md §8).

Values are read from the 16.19.8230722 item data (``dv``); geometry rules
not encoded in data follow ITEMS.md §8 defaults (edge-inclusive radii U-2,
nearest-10 splash cap U-11, Titanic cone U-3, Stridebreaker MS decay U-5).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ...core.damage import (CLASS_STRUCTURE, ON_HIT_ITEM, PHYSICAL, PROP_LIFESTEAL, TAG_ACTIVE_SPELL, TAG_AOE,
                            TAG_ITEM, TAG_PROC, concat_packets, packets)
from ..catalog import ItemStats
from .core import (ActiveOut, Effects, dv, effects, enemy_mask, holds, in_circle, nearest_k, onehot_units,
                   target_class, unit_pos)

TIAMAT, RAVENOUS, TITANIC, PROFANE, STRIDEBREAKER = 3077, 3074, 3748, 6698, 6631
CLEAVE_ITEMS = (TIAMAT, RAVENOUS, STRIDEBREAKER, PROFANE)
CLEAVE_RATIO_MELEE = 0.40     # MeleeItemCalcValue (all four)
CLEAVE_RATIO_RANGED = 0.20    # RangedItemCalcValue
MAX_SPLASH = int(dv(TIAMAT, "MaxProcPerAuto"))
CLEAVE_RADIUS = dv(TIAMAT, "CleaveRadius")
ACTIVE_OFFSET = 100.0         # spell castConeDistance
# Titanic cone default (U-3): from the primary target, along attacker->target.
TITANIC_CONE_LENGTH = 300.0
TITANIC_CONE_HALF_WIDTH = 210.0

COVERAGE = {
    TIAMAT: "Cleave; Crescent active (0.75 AD, r450 offset 100, cd 10 from cast end)",
    RAVENOUS: "Cleave with life steal; Ravenous Crescent (0.8 AD, VampAmp 1.0, cd 10 from cast end)",
    TITANIC: "Cleave on-hit 1% max HP + cone 3% max HP; Titanic Crescent empowered attack, attack reset",
    PROFANE: "Cleave (not on 0-damage attacks); Heretical Cleave (0.8 AD, cd 10 from cast start)",
    STRIDEBREAKER: "Cleave; Breaking Shockwave (0.8 AD, 35% slow 3 s, +35% MS per champion decaying 3 s)",
}

# Active table: item -> (ratio, radius, cooldown, base cast time, cd starts at cast start).
_ACTIVES = {
    TIAMAT: (dv(TIAMAT, "ActiveADRatio"), dv(TIAMAT, "Radius"), dv(TIAMAT, "Cooldown"), 0.2, False),
    RAVENOUS: (dv(RAVENOUS, "ActiveADRatio"), dv(RAVENOUS, "Radius"), dv(RAVENOUS, "Cooldown"), 0.2, False),
    PROFANE: (0.8, dv(PROFANE, "ActiveRadius"), dv(PROFANE, "Cooldown"), 0.2, True),
    STRIDEBREAKER: (dv(STRIDEBREAKER, "ADRatio"), dv(STRIDEBREAKER, "CleaveRadius"),
                    dv(STRIDEBREAKER, "Cooldown"), 0.25, True),
}


class State(NamedTuple):
    cd_until: Any           # (C,) shared Hydra-group active cooldown (one Hydra ownable)
    cast_item: Any          # (C,) int32 item id of the pending cast, 0 = none
    cast_end: Any           # (C,) seconds
    titanic_until: Any      # (C,) empowered-attack window end
    stride_ms: Any          # (C,) bonus MS fraction at grant
    stride_ms_start: Any    # (C,) seconds


def init(n_champions: int, n_units: int) -> State:
    z = jnp.zeros((n_champions,), jnp.float32)
    return State(z - 1e9, jnp.zeros((n_champions,), jnp.int32), z, z - 1e9, z, z - 1e9)


STRIDE_DECAY = dv(STRIDEBREAKER, "Duration")


def stats(state: State, own, ctx):
    left = jnp.clip(1.0 - (ctx.now - state.stride_ms_start) / STRIDE_DECAY, 0.0, 1.0)
    return ItemStats(percent_move_speed=state.stride_ms * left)


def _holder_ratio(ctx):
    return jnp.where(ctx.is_ranged, CLEAVE_RATIO_RANGED, CLEAVE_RATIO_MELEE)


def on_hit(state: State, own, ctx, units, attack) -> tuple[State, Effects]:
    c, n = ctx.level.shape[0], units.x.shape[0]
    hit = attack.hit & ctx.alive
    tcls = target_class(units, attack.target)
    not_structure = tcls != CLASS_STRUCTURE
    tx, ty = unit_pos(units, attack.target)
    primary = onehot_units(attack.target, n)
    enemies = enemy_mask(ctx, units) & (units.cls[None, :] != CLASS_STRUCTURE) & ~primary
    dist = jnp.sqrt((units.x[None, :] - tx[:, None]) ** 2 + (units.y[None, :] - ty[:, None]) ** 2)

    # Cleave (Tiamat/Ravenous/Stridebreaker/Profane): 350 around the primary.
    cleave_holder = holds(own, TIAMAT) | holds(own, RAVENOUS) | holds(own, STRIDEBREAKER) | holds(own, PROFANE)
    profane_zero = holds(own, PROFANE) & (attack.raw <= 0.0)
    do_cleave = hit & not_structure & cleave_holder & ~profane_zero
    near = in_circle(units, tx, ty, jnp.full((c,), CLEAVE_RADIUS)) & enemies
    splash = nearest_k(dist, near, MAX_SPLASH) & do_cleave[:, None]
    cleave_dmg = _holder_ratio(ctx) * ctx.total_ad
    cleave_flags = TAG_AOE | TAG_PROC | TAG_ITEM | jnp.where(holds(own, RAVENOUS), PROP_LIFESTEAL, 0)
    cleave_item = jnp.where(holds(own, RAVENOUS), RAVENOUS, jnp.where(holds(own, STRIDEBREAKER), STRIDEBREAKER,
                            jnp.where(holds(own, PROFANE), PROFANE, TIAMAT)))
    p_cleave = packets(splash, ctx.unit[:, None], jnp.arange(n)[None, :], cleave_dmg[:, None], PHYSICAL,
                       cleave_flags[:, None], item=cleave_item[:, None])

    # Titanic: on-hit %max HP to the primary (also vs structures) + cone splash.
    titanic = hit & holds(own, TITANIC)
    empowered = titanic & (ctx.now < state.titanic_until)
    rmult = jnp.where(ctx.is_ranged, dv(TITANIC, "RangedEffectiveness"), 1.0)
    onhit_ratio = jnp.where(empowered, dv(TITANIC, "ActivePrimaryTargetHPRatio"), dv(TITANIC, "PrimaryTargetHPRatio"))
    cone_ratio = jnp.where(empowered, dv(TITANIC, "ActiveSplashHPRatio"), dv(TITANIC, "SplashHPRatio"))
    p_titanic_hit = packets(titanic, ctx.unit, jnp.maximum(attack.target, 0), onhit_ratio * rmult * ctx.max_hp,
                            PHYSICAL, ON_HIT_ITEM | PROP_LIFESTEAL, item=TITANIC)
    ax, ay = unit_pos(units, ctx.unit)
    dirx, diry = tx - ax, ty - ay
    norm = jnp.maximum(jnp.sqrt(dirx ** 2 + diry ** 2), 1e-6)
    ux, uy = dirx / norm, diry / norm
    rx, ry = units.x[None, :] - tx[:, None], units.y[None, :] - ty[:, None]
    along = rx * ux[:, None] + ry * uy[:, None]
    across = jnp.abs(-rx * uy[:, None] + ry * ux[:, None])
    reach = units.radius[None, :]
    in_cone = (along >= -reach) & (along <= TITANIC_CONE_LENGTH + reach) \
        & (across <= TITANIC_CONE_HALF_WIDTH * jnp.clip(along, 0.0, TITANIC_CONE_LENGTH) / TITANIC_CONE_LENGTH + reach)
    cone = nearest_k(dist, in_cone & enemies, MAX_SPLASH) & (titanic & not_structure)[:, None]
    p_cone = packets(cone, ctx.unit[:, None], jnp.arange(n)[None, :], (cone_ratio * rmult * ctx.max_hp)[:, None],
                     PHYSICAL, TAG_AOE | TAG_PROC | TAG_ITEM, item=TITANIC)
    state = state._replace(titanic_until=jnp.where(empowered, -1e9, state.titanic_until),
                           cd_until=jnp.where(empowered, ctx.now + dv(TITANIC, "Cooldown"), state.cd_until))
    return state, effects(c, n, packets=concat_packets(p_titanic_hit, p_cleave, p_cone))


def active(state: State, own, ctx, units, request) -> tuple[State, Effects, ActiveOut]:
    """Start a Hydra-family cast; resolve casts whose cast time ends this tick."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    ready = (ctx.now >= state.cd_until) & (state.cast_item == 0) & ctx.alive
    out_used = jnp.zeros((c,), bool)
    cast_time = jnp.zeros((c,), jnp.float32)
    can_move = jnp.ones((c,), bool)
    reset = jnp.zeros((c,), bool)
    cast_item, cast_end, cd_until = state.cast_item, state.cast_end, state.cd_until
    titanic_until = state.titanic_until

    # Titanic Crescent: instant, empowers the next attack and resets the attack timer.
    t_go = ready & (request == TITANIC) & holds(own, TITANIC)
    titanic_until = jnp.where(t_go, ctx.now + dv(TITANIC, "Cooldown"), titanic_until)
    cd_until = jnp.where(t_go, jnp.inf, cd_until)   # starts when the empowered attack is used
    out_used, reset = out_used | t_go, reset | t_go
    # Expired unused empowerment starts the cooldown at window end (INFERRED M).
    expired = jnp.isinf(cd_until) & (ctx.now >= titanic_until)
    cd_until = jnp.where(expired, titanic_until + dv(TITANIC, "Cooldown"), cd_until)

    for item, (ratio, radius, cooldown, base_cast, cd_at_start) in _ACTIVES.items():
        go = ready & (request == item) & holds(own, item)
        t = jnp.minimum(base_cast, ctx.attack_windup)
        cast_item = jnp.where(go, item, cast_item)
        cast_end = jnp.where(go, ctx.now + t, cast_end)
        cd_until = jnp.where(go, jnp.where(cd_at_start, ctx.now + cooldown, jnp.inf), cd_until)
        out_used = out_used | go
        cast_time = jnp.where(go, t, cast_time)
        can_move = jnp.where(go, item == STRIDEBREAKER, can_move)

    # Resolve casts ending this tick (zero-length casts resolve immediately).
    finishing = (cast_item != 0) & (ctx.now >= cast_end) & ctx.alive
    cx = ctx.x + ACTIVE_OFFSET * ctx.facing_x
    cy = ctx.y + ACTIVE_OFFSET * ctx.facing_y
    enemies = enemy_mask(ctx, units) & (units.cls[None, :] != CLASS_STRUCTURE)
    all_packets = []
    slow = jnp.zeros((n,), jnp.float32)
    slow_duration = jnp.zeros((n,), jnp.float32)
    stride_ms, stride_ms_start = state.stride_ms, state.stride_ms_start
    for item, (ratio, radius, cooldown, base_cast, cd_at_start) in _ACTIVES.items():
        fire = finishing & (cast_item == item)
        hit = in_circle(units, cx, cy, jnp.full((c,), radius)) & enemies & fire[:, None]
        flags = TAG_AOE | TAG_ACTIVE_SPELL | TAG_ITEM | (PROP_LIFESTEAL if item == RAVENOUS else 0)
        all_packets.append(packets(hit, ctx.unit[:, None], jnp.arange(n)[None, :],
                                   (ratio * ctx.total_ad)[:, None], PHYSICAL, flags, item=item))
        cd_until = jnp.where(fire & ~cd_at_start, ctx.now + cooldown, cd_until)
        if item == STRIDEBREAKER:
            any_hit = jnp.any(hit, axis=0)
            slow = jnp.where(any_hit, -dv(STRIDEBREAKER, "MSSlow"), slow)
            slow_duration = jnp.where(any_hit, dv(STRIDEBREAKER, "Duration"), slow_duration)
            champs = jnp.sum(hit & (units.cls[None, :] == 0), axis=1).astype(jnp.float32)
            gain = fire & (champs > 0)
            stride_ms = jnp.where(gain, dv(STRIDEBREAKER, "ActiveMS") * champs, stride_ms)
            stride_ms_start = jnp.where(gain, ctx.now, stride_ms_start)
    cast_item = jnp.where(finishing, 0, cast_item)
    state = State(cd_until, cast_item, cast_end, titanic_until, stride_ms, stride_ms_start)
    eff = effects(c, n, packets=concat_packets(*all_packets), slow=slow, slow_duration=slow_duration,
                  attack_reset=reset)
    return state, eff, ActiveOut(out_used, cast_time, can_move, reset)
