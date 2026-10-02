"""Registry and dispatch for every 26.19 SR item effect.

``MODULES`` lists effect modules in canonical emission order (README hook
crosswalk: main hit -> item on-hits in this order). Dispatch calls each
module's hook if it defines one and merges results; per-module state lives
in one field of ``ItemEffectState`` named after the module.

Coverage contract: every catalog item is exactly one of
  * implemented by a module (``module.COVERAGE``),
  * ``STATS_ONLY`` (no behaviour beyond static stats), or
  * ``DEFERRED`` with a reason (MODERN-009: jungle, vision, non-Hydra actives).
``coverage_report()`` and the tests enforce this, so a newly added item can
never silently run as stats only.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..modern_item_data import ItemStats, catalog, combine_stats, zero_stats
from .core import (CC, ActiveOut, Attack, AttackMods, Cast, Ctx, Debuffs, Effects, HolderDefense,
                   Kills, Report, StatusFlags, Units, combine_debuffs, combine_defense, merge_effects,
                   neutral_debuffs, no_effects)
from . import (consumables, starters, spellblade, hydra, fighter, defense, mage, marksman,
               support, boots)

MODULES = (consumables, starters, spellblade, hydra, fighter, defense, mage, marksman, support, boots)

# Items whose entire 26.19 effect is their static stat line: no client data
# values, calculations or spell (items_client.json), minus Phantom Dancer,
# whose ghosting passive has no data values.
STATS_ONLY = {
    1001, 1004, 1006, 1011, 1018, 1026, 1027, 1028, 1029, 1031, 1033, 1036, 1037, 1038,
    1042, 1052, 1053, 1055, 1057, 1058, 2021, 2022, 2421, 3024, 3031, 3035, 3051, 3066,
    3067, 3086, 3108, 3113, 3114, 3133, 3135, 3801, 4630, 4642, 6690,
    2422,   # Slightly Magical Footwear (Magical Footwear's +10 MS is the rune's)
}
DEFERRED = {
    1101: "jungle pet (MODERN-009)", 1102: "jungle pet (MODERN-009)", 1103: "jungle pet (MODERN-009)",
    2055: "Control Ward: vision (MODERN-009)", 3340: "Stealth Ward trinket: vision (MODERN-009)",
    3363: "Farsight Alteration trinket: vision (MODERN-009)", 3364: "Oracle Lens trinket: vision (MODERN-009)",
    3330: "Scarecrow Effigy: Fiddlesticks-only trinket", 3599: "Kalista's Black Spear: Kalista-only",
    3600: "Kalista's Black Spear (Sylas copy): Kalista-only",
}


class ItemEffectState(NamedTuple):
    consumables: Any
    starters: Any
    spellblade: Any
    hydra: Any
    fighter: Any
    defense: Any
    mage: Any
    marksman: Any
    support: Any
    boots: Any


def _name(m) -> str:
    return m.__name__.rsplit(".", 1)[-1]


assert tuple(_name(m) for m in MODULES) == ItemEffectState._fields


def init(n_champions: int, n_units: int) -> ItemEffectState:
    return ItemEffectState(*(m.init(n_champions, n_units) for m in MODULES))


def coverage_report() -> dict[int, str]:
    """Item id -> provenance; raises on gaps or double coverage."""
    out: dict[int, str] = {}
    for m in MODULES:
        for iid, what in m.COVERAGE.items():
            if iid in out:
                raise RuntimeError(f"item {iid} covered twice: {out[iid]} / {_name(m)}")
            out[iid] = f"{_name(m)}: {what}"
    for iid in STATS_ONLY:
        if iid in out:
            raise RuntimeError(f"item {iid} is both STATS_ONLY and {out[iid]}")
        out[iid] = "stats only"
    for iid, why in DEFERRED.items():
        if iid in out:
            raise RuntimeError(f"item {iid} is both DEFERRED and {out[iid]}")
        out[iid] = f"DEFERRED: {why}"
    missing = sorted(set(catalog().ids) - set(out))
    if missing:
        names = ", ".join(f"{i} {catalog()[i].name}" for i in missing)
        raise RuntimeError(f"items without an effect classification: {names}")
    extra = sorted(set(out) - set(catalog().ids))
    if extra:
        raise RuntimeError(f"coverage lists non-catalog items: {extra}")
    return out


def _each(hook: str):
    for m in MODULES:
        fn = getattr(m, hook, None)
        if fn is not None:
            yield m, fn


def _sub(state: ItemEffectState, m):
    return getattr(state, _name(m))


def _put(state: ItemEffectState, m, sub) -> ItemEffectState:
    return state._replace(**{_name(m): sub})


def dynamic_stats(state: ItemEffectState, own, ctx: Ctx) -> ItemStats:
    parts = [fn(_sub(state, m), own, ctx) for m, fn in _each("stats")]
    return combine_stats(zero_stats(ctx.level.shape), *parts)


def holder_defense(state: ItemEffectState, own, ctx: Ctx) -> HolderDefense:
    parts = [fn(_sub(state, m), own, ctx) for m, fn in _each("defense")]
    return combine_defense(parts, ctx.level.shape[0])


def status(state: ItemEffectState, own, ctx: Ctx) -> StatusFlags:
    out = StatusFlags(jnp.zeros(ctx.level.shape, bool))
    for m, fn in _each("status"):
        r = fn(_sub(state, m), own, ctx)
        out = StatusFlags(out.ghosted | r.ghosted)
    return out


def target_debuffs(state: ItemEffectState, own, ctx: Ctx, units: Units) -> Debuffs:
    parts = [fn(_sub(state, m), own, ctx, units) for m, fn in _each("debuffs")]
    return combine_debuffs(parts, units.x.shape[0])


def dealt_amp(state: ItemEffectState, own, ctx: Ctx, units: Units):
    amp = jnp.zeros((ctx.level.shape[0], units.x.shape[0]), jnp.float32)
    for m, fn in _each("dealt_amp"):
        amp = amp + fn(_sub(state, m), own, ctx, units)
    return amp


def packet_amp(state: ItemEffectState, own, ctx: Ctx, units: Units, packets):
    """(P,) additive DMG.40 amps: per-target ``dealt_amp`` plus tag-filtered ``packet_amp``."""
    amp = dealt_amp(state, own, ctx, units)
    p = packets
    src_is = p.src[:, None] == ctx.unit[None, :]
    per = amp[:, jnp.clip(p.dst, 0, amp.shape[1] - 1)].T
    out = jnp.sum(jnp.where(src_is, per, 0.0), axis=1)
    for m, fn in _each("packet_amp"):
        out = out + fn(_sub(state, m), own, ctx, units, p)
    return out


def attack_mods(state: ItemEffectState, own, ctx: Ctx, units: Units, target) -> AttackMods:
    c = ctx.level.shape[0]
    out = AttackMods(jnp.zeros((c,), bool), jnp.ones((c,), jnp.float32))
    for m, fn in _each("attack_mods"):
        r = fn(_sub(state, m), own, ctx, units, target)
        out = AttackMods(out.force_crit | r.force_crit,
                         jnp.where(r.force_crit, r.crit_scale, out.crit_scale))
    return out


def _event(hook: str, state: ItemEffectState, own, ctx: Ctx, units: Units, *event):
    c, n = ctx.level.shape[0], units.x.shape[0]
    parts = []
    for m, fn in _each(hook):
        sub, eff = fn(_sub(state, m), own, ctx, units, *event)
        state = _put(state, m, sub)
        parts.append(eff)
    return state, merge_effects(parts, c, n)


def on_attack(state, own, ctx, units, attack: Attack):
    return _event("on_attack", state, own, ctx, units, attack)


def on_hit(state, own, ctx, units, attack: Attack):
    return _event("on_hit", state, own, ctx, units, attack)


def on_cast(state, own, ctx, units, cast: Cast):
    return _event("on_cast", state, own, ctx, units, cast)


def on_cc(state, own, ctx, units, cc: CC):
    return _event("on_cc", state, own, ctx, units, cc)


def on_damage(state, own, ctx, units, report: Report):
    return _event("on_damage", state, own, ctx, units, report)


def periodic(state, own, ctx, units):
    return _event("periodic", state, own, ctx, units)


def on_takedown(state, own, ctx, units, kills: Kills):
    return _event("on_takedown", state, own, ctx, units, kills)


def on_shop(state, own, ctx):
    for m, fn in _each("on_shop"):
        state = _put(state, m, fn(_sub(state, m), own, ctx))
    return state


def active(state: ItemEffectState, own, ctx: Ctx, units: Units, request):
    """``request`` (C,) int32 item id whose active is pressed (0 = none)."""
    c, n = ctx.level.shape[0], units.x.shape[0]
    parts = []
    out = ActiveOut(jnp.zeros((c,), bool), jnp.zeros((c,), jnp.float32),
                    jnp.ones((c,), bool), jnp.zeros((c,), bool))
    for m, fn in _each("active"):
        sub, eff, r = fn(_sub(state, m), own, ctx, units, request)
        state = _put(state, m, sub)
        parts.append(eff)
        out = ActiveOut(out.used | r.used, jnp.where(r.used, r.cast_time, out.cast_time),
                        jnp.where(r.used, r.can_move, out.can_move), out.attack_reset | r.attack_reset)
    return state, merge_effects(parts, c, n), out


__all__ = ["MODULES", "STATS_ONLY", "DEFERRED", "ItemEffectState", "init", "coverage_report",
           "dynamic_stats", "holder_defense", "status", "target_debuffs", "dealt_amp", "attack_mods",
           "packet_amp", "on_attack", "on_hit", "on_cast", "on_cc", "on_damage", "periodic", "on_takedown", "on_shop",
           "active", "no_effects"]
