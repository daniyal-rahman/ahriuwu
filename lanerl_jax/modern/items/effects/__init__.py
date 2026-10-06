"""Registry and dispatch of every 26.19 SR item effect (ITEMS.md §10-11; hook protocol in ``core``).

``MODULES`` is the canonical emission order (main hit, then item on-hits in this order). Dispatch calls each
module's hook if defined and merges the results; module state lives in the ``ItemEffectState`` field named after
the module. Every catalog item is exactly one of: a module's ``COVERAGE``, ``STATS_ONLY``, ``WORLD`` (run by a world
subsystem) or ``DEFERRED``; ``coverage_report`` enforces this so a new item never silently runs as stats only.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ..catalog import ItemStats, catalog, combine_stats, zero_stats
from . import (actives, boots, consumables, defense, fighter, hydra, jungle, mage, marksman, spellblade, starters,
               support)
from .core import (ActiveOut, AttackMods, Ctx, Debuffs, HolderDefense, StatusFlags, Units, combine_debuffs,
                   combine_defense, merge_effects)

MODULES = (consumables, starters, spellblade, hydra, fighter, defense, mage, marksman, support, boots,
           actives, jungle)

# No client data values, calculations or spell: the stat line is the whole effect (Phantom Dancer's ghosting
# has no data values and lives in marksman).
STATS_ONLY = {
    1001, 1004, 1006, 1011, 1018, 1026, 1027, 1028, 1029, 1031, 1033, 1036, 1037, 1038,
    1042, 1052, 1053, 1055, 1057, 1058, 2021, 2022, 2421, 3024, 3031, 3035, 3051, 3066,
    3067, 3086, 3108, 3113, 3114, 3133, 3135, 3801, 4630, 4642, 6690,
    2422,   # Slightly Magical Footwear (the +10 MS belongs to the Magical Footwear rune)
}
WORLD = {   # docs/modern/WARDS.md
    2055: "wards: Control Ward", 3340: "wards: Stealth Ward trinket", 3363: "wards: Farsight Alteration trinket",
    3364: "wards: Oracle Lens trinket",
}
DEFERRED = {
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
    actives: Any
    jungle: Any


def _name(m) -> str:
    return m.__name__.rsplit(".", 1)[-1]


assert tuple(_name(m) for m in MODULES) == ItemEffectState._fields


def init(n_champions: int, n_units: int) -> ItemEffectState:
    return ItemEffectState(*(m.init(n_champions, n_units) for m in MODULES))


def coverage_report() -> dict[int, str]:
    """Item id -> provenance; raises on gaps or double coverage."""
    out: dict[int, str] = {}
    sources = [(m.COVERAGE, f"{_name(m)}: ") for m in MODULES]
    sources += [(dict.fromkeys(STATS_ONLY, "stats only"), ""), (WORLD, "WORLD "), (DEFERRED, "DEFERRED: ")]
    for table, prefix in sources:
        for iid, what in table.items():
            if iid in out:
                raise RuntimeError(f"item {iid} covered twice: {out[iid]} / {prefix}{what}")
            out[iid] = prefix + what
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
        out = StatusFlags(out.ghosted | fn(_sub(state, m), own, ctx).ghosted)
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
        out = AttackMods(out.force_crit | r.force_crit, jnp.where(r.force_crit, r.crit_scale, out.crit_scale))
    return out


def _event(hook: str):
    def run(state: ItemEffectState, own, ctx: Ctx, units: Units, *event):
        parts = []
        for m, fn in _each(hook):
            sub, eff = fn(_sub(state, m), own, ctx, units, *event)
            state = _put(state, m, sub)
            parts.append(eff)
        return state, merge_effects(parts, ctx.level.shape[0], units.x.shape[0])
    run.__name__ = hook
    return run


on_attack, on_hit, on_cast, on_cc, on_damage, periodic, on_takedown = map(
    _event, ("on_attack", "on_hit", "on_cast", "on_cc", "on_damage", "periodic", "on_takedown"))


def on_shop(state: ItemEffectState, own, ctx: Ctx) -> ItemEffectState:
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
