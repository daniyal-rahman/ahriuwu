"""Spellblade group and Phage Rage (ITEMS.md §6.1).

One Spellblade per champion (group max 1): armed at ability cast start when its cooldown is over, for a 10 s window
a recast refreshes; consumed by the next basic attack that lands (any target, structures included), which starts the
cooldown. Damage reads base AD, is not crit-scaled and is an on-hit proc. Dusk and Dawn re-applies on-hits 0.2 s
later (wiki): ``periodic`` sets ``dd_extra_due`` and the world re-runs ``on_hit`` with ``extra_on_hit_attack``; the
Spellblade is then on cooldown, so only the other on-hits apply again.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ...core.damage import (CLASS_CHAMPION, CLASS_MONSTER, CLASS_STRUCTURE, MAGIC, ON_HIT_ITEM, PHYSICAL,
                            PROP_LIFESTEAL, packets)
from ..catalog import ItemStats, catalog
from .core import (Attack, Debuffs, Effects, dv, effects, enemy_mask, holds, holds_any, in_circle,
                   neutral_debuffs, onehot_units, target_class, unit_pos)

SHEEN, TRINITY, ICEBORN, LICH_BANE, ESSENCE_REAVER, DUSK_DAWN, BLOODSONG = (
    3057, 3078, 6662, 3100, 3508, 2510, 3877)
PHAGE = 3044
SPELLBLADE_ITEMS = (SHEEN, TRINITY, ICEBORN, LICH_BANE, ESSENCE_REAVER, DUSK_DAWN, BLOODSONG)


def _part(item_id: int, calc: str, i: int) -> dict:
    return catalog()[item_id].calculations[calc]["mFormulaParts"][i]


SB_COOLDOWN = dv(SHEEN, "SpellbladeCooldown")            # static (§6.1 INFERRED M)
SB_WINDOW = dv(LICH_BANE, "SpellBladeDuration")          # shared by the group
SHEEN_AD = _part(SHEEN, "SpellbladeDamage", 0)["mCoefficient"]
TRINITY_AD = dv(TRINITY, "SpellbladeMultiplier")
ICEBORN_AD = dv(ICEBORN, "SpellbladeMultiplier")
LICH_AD, LICH_AP = dv(LICH_BANE, "SpellbladeADRatio"), dv(LICH_BANE, "LichBaneAPValue")
ER_AD, ER_CRIT = dv(ESSENCE_REAVER, "BaseADRatio"), dv(ESSENCE_REAVER, "CritChanceMultiplier")
ER_MANA = catalog()[ESSENCE_REAVER].calculations["TotalManaRefund"]["mMultiplier"]["mNumber"]
DD_AD = _part(DUSK_DAWN, "SpellbladeDamage", 0)["mCoefficient"]
DD_AP = _part(DUSK_DAWN, "SpellbladeDamage", 1)["mCoefficient"]
DD_HEAL_AP = _part(DUSK_DAWN, "SpellbladeHealing", 0)["mCoefficient"]
DD_HEAL_BONUS_HP = _part(DUSK_DAWN, "SpellbladeHealing", 1)["mCoefficient"]
DD_EXTRA_DELAY = 0.2      # wiki
BS_AD = dv(BLOODSONG, "SheenMult")
FIELD_RADIUS = dv(ICEBORN, "AoERadius")
FIELD_DURATION = dv(ICEBORN, "SlowFieldDuration")
FIELD_SLOW_MELEE, FIELD_SLOW_RANGED = dv(ICEBORN, "SlowAmount"), dv(ICEBORN, "RangedSlowAmount")
ICEBORN_MONSTER = dv(ICEBORN, "MonsterMod")   # INFERRED L: applied to proc damage vs monsters
FIELD_TICK_LINGER = 1.5   # x dt: the per-tick field slow bridges to the next re-application in any hook order
LICH_AS = dv(LICH_BANE, "SheenASBuff")
# Quicken (Trinity) and Rage (Phage) are different named passives: both apply.
QUICKEN_MS, QUICKEN_DURATION = dv(TRINITY, "MoveSpeedBonus"), dv(TRINITY, "MSDuration")
RAGE_MS, RAGE_DURATION, RAGE_RANGED = (dv(PHAGE, "MoveSpeedBonus"), dv(PHAGE, "MoveSpeedDuration"),
                                       dv(PHAGE, "RangedMod"))
BS_AMP_MELEE, BS_AMP_RANGED = dv(BLOODSONG, "MeleeDamageAmp"), dv(BLOODSONG, "RangedDamageAmp")
BS_DEBUFF_DURATION = dv(BLOODSONG, "DebuffDuration")
BS_GP10 = dv(BLOODSONG, "GP10")

COVERAGE = {
    SHEEN: "Spellblade physical on-hit", TRINITY: "Spellblade; Quicken MS on attack hit",
    ICEBORN: "Spellblade (more vs monsters, INFERRED); frost field slowing enemies inside, one per holder",
    LICH_BANE: "Spellblade magic; AS while armed",
    ESSENCE_REAVER: "Spellblade with crit scaling, no life steal; mana from proc damage",
    DUSK_DAWN: "Spellblade magic; HSP heal; delayed on-hit re-application (extra_on_hit_attack)",
    BLOODSONG: "Spellblade; Expose Weakness on champions (strongest holder); gold; ward active deferred (vision)",
    PHAGE: "Rage MS on attack hit",
}


class State(NamedTuple):
    armed_until: Any        # (C,) Spellblade window end
    cd_until: Any           # (C,) shared SheenDelay end
    quicken_until: Any      # (C,)
    rage_until: Any         # (C,)
    field_until: Any        # (C,) Iceborn frost field end (one live field per holder)
    field_x: Any
    field_y: Any
    field_slow: Any         # (C,) slow strength of that field
    maim_until: Any         # (C, N) Bloodsong Expose Weakness end per target
    maim_amp: Any           # (C, N)
    dd_extra_at: Any        # (C,) pending Dusk and Dawn re-application time (inf = none)
    dd_extra_target: Any    # (C,) int32
    dd_extra_due: Any       # (C,) bool: re-application fires this tick (set by periodic)
    dd_due_target: Any      # (C,) int32


def init(n_champions: int, n_units: int) -> State:
    z = jnp.zeros((n_champions,), jnp.float32)
    zn = jnp.zeros((n_champions, n_units), jnp.float32)
    neg = z - 1e9
    m1 = jnp.full((n_champions,), -1, jnp.int32)
    return State(neg, neg, neg, neg, neg, z, z, z, zn - 1e9, zn, z + jnp.inf, m1,
                 jnp.zeros((n_champions,), bool), m1)


def _armed(state: State, own, ctx) -> Any:
    return holds_any(own, SPELLBLADE_ITEMS) & (ctx.now < state.armed_until)


def stats(state: State, own, ctx) -> ItemStats:
    quicken = holds(own, TRINITY) & (ctx.now < state.quicken_until)
    rage = holds(own, PHAGE) & (ctx.now < state.rage_until)
    rage_ms = RAGE_MS * jnp.where(ctx.is_ranged, RAGE_RANGED, 1.0)
    lich = holds(own, LICH_BANE) & _armed(state, own, ctx)
    return ItemStats(move_speed=jnp.where(quicken, QUICKEN_MS, 0.0) + jnp.where(rage, rage_ms, 0.0),
                     attack_speed=jnp.where(lich, LICH_AS, 0.0))


def on_cast(state: State, own, ctx, units, cast) -> tuple[State, Effects]:
    c, n = ctx.level.shape[0], units.x.shape[0]
    arm = cast.started & ctx.alive & holds_any(own, SPELLBLADE_ITEMS) & (ctx.now >= state.cd_until)
    state = state._replace(armed_until=jnp.where(arm, ctx.now + SB_WINDOW, state.armed_until))
    return state, effects(c, n)


def _which(own) -> Any:
    """(C,) item id of the held Spellblade item (group max 1)."""
    out = jnp.zeros(own.shape[:1], jnp.int32)
    for iid in reversed((TRINITY, ICEBORN, LICH_BANE, ESSENCE_REAVER, DUSK_DAWN, BLOODSONG, SHEEN)):
        out = jnp.where(holds(own, iid), iid, out)
    return out


def proc_damage(own, ctx) -> tuple[Any, Any]:
    """(C,) Spellblade raw damage and dtype for the held item (0 if none)."""
    item = _which(own)
    crit = jnp.clip(ctx.crit_chance, 0.0, 1.0)
    b, ap = ctx.base_ad, ctx.ap
    dmg = jnp.select(
        [item == TRINITY, item == ICEBORN, item == LICH_BANE, item == ESSENCE_REAVER,
         item == DUSK_DAWN, item == BLOODSONG, item == SHEEN],
        [TRINITY_AD * b, ICEBORN_AD * b, LICH_AD * b + LICH_AP * ap, ER_AD * b + ER_CRIT * crit,
         DD_AD * b + DD_AP * ap, BS_AD * b, SHEEN_AD * b], 0.0)
    dtype = jnp.where((item == LICH_BANE) | (item == DUSK_DAWN), MAGIC, PHYSICAL)
    return dmg, dtype


def on_hit(state: State, own, ctx, units, attack: Attack) -> tuple[State, Effects]:
    c, n = ctx.level.shape[0], units.x.shape[0]
    hit = attack.hit & ctx.alive & (attack.target >= 0)
    now = ctx.now
    item = _which(own)
    proc = hit & _armed(state, own, ctx)
    tgt = jnp.maximum(attack.target, 0)
    tcls = target_class(units, attack.target)

    dmg, dtype = proc_damage(own, ctx)
    dmg = jnp.where((item == ICEBORN) & (tcls == CLASS_MONSTER), dmg * ICEBORN_MONSTER, dmg)
    flags = ON_HIT_ITEM | jnp.where(item == ESSENCE_REAVER, 0, PROP_LIFESTEAL)
    p = packets(proc & (dmg > 0.0), ctx.unit, tgt, dmg, dtype, flags, item=item)
    mana = jnp.where(proc & (item == ESSENCE_REAVER), ER_MANA * dmg, 0.0)
    heal = jnp.where(proc & (item == DUSK_DAWN), DD_HEAL_AP * ctx.ap + DD_HEAL_BONUS_HP * ctx.bonus_hp, 0.0)

    # Iceborn field at the target; slows enemies inside from this tick.
    ice = proc & (item == ICEBORN)
    tx, ty = unit_pos(units, attack.target)
    slow_val = jnp.where(ctx.is_ranged, FIELD_SLOW_RANGED, FIELD_SLOW_MELEE)
    field_until = jnp.where(ice, now + FIELD_DURATION, state.field_until)
    field_x = jnp.where(ice, tx, state.field_x)
    field_y = jnp.where(ice, ty, state.field_y)
    field_slow = jnp.where(ice, slow_val, state.field_slow)
    inside = _field_hits(units, ctx, field_x, field_y) & ice[:, None]
    slow = jnp.max(jnp.where(inside, slow_val[:, None], 0.0), axis=0)
    slow_duration = jnp.where(slow > 0.0, FIELD_TICK_LINGER * ctx.dt, 0.0)

    bs = proc & (item == BLOODSONG) & (tcls == CLASS_CHAMPION)
    on_t = onehot_units(attack.target, n) & bs[:, None]
    amp = jnp.where(ctx.is_ranged, BS_AMP_RANGED, BS_AMP_MELEE)
    maim_until = jnp.where(on_t, now + BS_DEBUFF_DURATION, state.maim_until)
    maim_amp = jnp.where(on_t, amp[:, None], state.maim_amp)

    dd = proc & (item == DUSK_DAWN)
    dd_at = jnp.where(dd, now + DD_EXTRA_DELAY, state.dd_extra_at)
    dd_target = jnp.where(dd, attack.target, state.dd_extra_target)

    quick = hit & holds(own, TRINITY)
    rage = hit & holds(own, PHAGE)
    state = state._replace(
        armed_until=jnp.where(proc, -1e9, state.armed_until),
        cd_until=jnp.where(proc, now + SB_COOLDOWN, state.cd_until),
        quicken_until=jnp.where(quick, now + QUICKEN_DURATION, state.quicken_until),
        rage_until=jnp.where(rage, now + RAGE_DURATION, state.rage_until),
        field_until=field_until, field_x=field_x, field_y=field_y, field_slow=field_slow,
        maim_until=maim_until, maim_amp=maim_amp, dd_extra_at=dd_at, dd_extra_target=dd_target)
    return state, effects(c, n, packets=p, mana=mana, heal=heal, slow=slow, slow_duration=slow_duration)


def _field_hits(units, ctx, fx, fy) -> Any:
    """(C, N) enemy non-structures touching each holder's field (U-2)."""
    c = fx.shape[0]
    enemies = enemy_mask(ctx, units) & (units.cls[None, :] != CLASS_STRUCTURE)
    return in_circle(units, fx, fy, jnp.full((c,), FIELD_RADIUS)) & enemies


def periodic(state: State, own, ctx, units) -> tuple[State, Effects]:
    c, n = ctx.level.shape[0], units.x.shape[0]
    now = ctx.now
    live = holds(own, ICEBORN) & (now < state.field_until)
    inside = _field_hits(units, ctx, state.field_x, state.field_y) & live[:, None]
    slow = jnp.max(jnp.where(inside, state.field_slow[:, None], 0.0), axis=0)
    slow_duration = jnp.where(slow > 0.0, FIELD_TICK_LINGER * ctx.dt, 0.0)

    t = jnp.clip(state.dd_extra_target, 0, n - 1)
    fire = now >= state.dd_extra_at
    due = fire & holds(own, DUSK_DAWN) & ctx.alive & (state.dd_extra_target >= 0) & units.alive[t]
    gold = jnp.where(holds(own, BLOODSONG), BS_GP10 / 10.0 * ctx.dt, 0.0)
    state = state._replace(dd_extra_due=due, dd_due_target=jnp.where(due, state.dd_extra_target, -1),
                           dd_extra_at=jnp.where(fire, jnp.inf, state.dd_extra_at))
    return state, effects(c, n, slow=slow, slow_duration=slow_duration, gold=gold)


def extra_on_hit_attack(state: State) -> Attack:
    """Attack for the Dusk and Dawn re-application due this tick."""
    z = jnp.zeros(state.dd_extra_due.shape, jnp.float32)
    return Attack(jnp.zeros_like(state.dd_extra_due), state.dd_extra_due, state.dd_due_target, z,
                  jnp.zeros_like(state.dd_extra_due))


def debuffs(state: State, own, ctx, units) -> Debuffs:
    n = units.x.shape[0]
    live = (ctx.now < state.maim_until) & holds(own, BLOODSONG)[:, None]
    # Same named debuff from several holders: strongest.
    amp = jnp.max(jnp.where(live, state.maim_amp, 0.0), axis=0)
    return neutral_debuffs(n)._replace(received_amp=amp)
