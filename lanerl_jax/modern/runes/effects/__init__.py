"""Registry and dispatch of every 26.19 SR rune effect (RUNES.md §9).

``MODULES`` (one per tree) run in this order after all item hooks (main hit -> item on-hits -> rune on-hits,
U-18); each module's state is the ``RuneEffectState`` field of its name. ``coverage_report`` enforces that
every catalog perk is implemented by exactly one module, run by the world (``WORLD``), ``DEFERRED`` with a
reason, or a ``STATIC`` stat shard (``items.loadout.stat_shard_stats``).
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp

from ...items.catalog import ItemStats, catalog, combine_stats, zero_stats
from ...items.effects.core import Debuffs, combine_debuffs, merge_effects
from ..catalog import rune_catalog
from . import domination, inspiration, precision, resolve, sorcery
from .core import RuneEvents, RuneOutputs, merge_outputs, no_outputs

MODULES = (precision, domination, sorcery, resolve, inspiration)

WORLD = {
    8137: "Sixth Sense: wards.ward_step via domination.sixth_sense",
    8141: "Deep Ward: wards.ward_step via domination.deep_ward",
}
DEFERRED: dict[int, str] = {}
STATIC = {
    5008: "shard: +9 adaptive force", 5005: "shard: +10% attack speed", 5007: "shard: +8 ability haste",
    5010: "shard: +2.5% move speed", 5001: "shard: +10·level health (extrapolated past 18)",
    5011: "shard: +65 health", 5013: "shard: +15% tenacity and slow resist",
}


class RuneEffectState(NamedTuple):
    precision: Any
    domination: Any
    sorcery: Any
    resolve: Any
    inspiration: Any


def _name(m) -> str:
    return m.__name__.rsplit(".", 1)[-1]


assert tuple(_name(m) for m in MODULES) == RuneEffectState._fields


def init(n_champions: int, n_units: int) -> RuneEffectState:
    return RuneEffectState(*(m.init(n_champions, n_units) for m in MODULES))


def coverage_report() -> dict[int, str]:
    """Perk id -> provenance; raises on gaps or double coverage."""
    out: dict[int, str] = {}
    tables = [(_name(m), m.COVERAGE) for m in MODULES] + [("WORLD", WORLD), ("DEFERRED", DEFERRED),
                                                          ("STATIC", STATIC)]
    for label, table in tables:
        for pid, what in table.items():
            if pid in out:
                raise RuntimeError(f"rune {pid} is both {label} and {out[pid]}")
            out[pid] = f"{label}: {what}"
    cat = rune_catalog()
    missing = sorted(set(cat.ids) - set(out))
    if missing:
        raise RuntimeError("runes without an effect classification: "
                           + ", ".join(f"{p} {cat[p].name}" for p in missing))
    extra = sorted(set(out) - set(cat.ids))
    if extra:
        raise RuntimeError(f"coverage lists non-catalog perks: {extra}")
    return out


def _each(hook: str):
    """(module name, hook function) for the modules that define ``hook``."""
    return [(_name(m), getattr(m, hook)) for m in MODULES if hasattr(m, hook)]


def stats(state, page, ctx, ev: RuneEvents) -> ItemStats:
    parts = [fn(getattr(state, m), page, ctx, ev) for m, fn in _each("stats")]
    return combine_stats(zero_stats(ctx.level.shape), *parts)


def debuffs(state, page, ctx, units, ev) -> Debuffs:
    parts = [fn(getattr(state, m), page, ctx, units, ev) for m, fn in _each("debuffs")]
    return combine_debuffs(parts, units.x.shape[0])


def _packet_sum(hook: str):
    def run(state, page, ctx, units, ev, packets):
        out = jnp.zeros(packets.valid.shape, jnp.float32)
        for m, fn in _each(hook):
            out = out + fn(getattr(state, m), page, ctx, units, ev, packets)
        return out
    return run


packet_amp, packet_block = _packet_sum("packet_amp"), _packet_sum("packet_block")


def heal_mult(state, page, ctx, ev):
    out = jnp.ones(ctx.level.shape, jnp.float32)
    for m, fn in _each("heal_mult"):
        out = out * fn(getattr(state, m), page, ctx, ev)
    return out


def _event(hook: str):
    def run(state, page, ctx, units, ev):
        parts = []
        for m, fn in _each(hook):
            sub, eff = fn(getattr(state, m), page, ctx, units, ev)
            state = state._replace(**{m: sub})
            parts.append(eff)
        return state, merge_effects(parts, ctx.level.shape[0], units.x.shape[0])
    return run


on_cast, on_attack, on_hit, on_cc, periodic, on_damage, on_takedown = map(
    _event, ("on_cast", "on_attack", "on_hit", "on_cc", "periodic", "on_damage", "on_takedown"))


def post_tick(state, page, ctx, units, ev):
    for m, fn in _each("post_tick"):
        state = state._replace(**{m: fn(getattr(state, m), page, ctx, units, ev)})
    return state


def outputs(state, page, ctx, ev) -> RuneOutputs:
    c, i = ctx.level.shape[0], len(catalog().ids)
    parts = [fn(getattr(state, m), page, ctx, ev) for m, fn in _each("outputs")]
    return merge_outputs(parts, c, i) if parts else no_outputs(c, i)
