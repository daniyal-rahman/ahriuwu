"""Registry and dispatch for every 26.19 SR rune effect.

``MODULES`` (one per tree) run in this order after all item hooks, matching
the README default "main hit -> item on-hits -> rune on-hits" (RUNES U-18).
Per-module state lives in one ``RuneEffectState`` field named after the
module.

Coverage contract: every selectable rune is exactly one of
  * implemented by a module (``module.COVERAGE``),
  * ``WORLD``: run by a world subsystem (Sixth Sense, Deep Ward: domination
    kernels called by ``wards.ward_step``), or
  * ``DEFERRED`` with a reason (currently none),
and the seven stat shards are ``STATIC`` (``items.loadout.stat_shard_stats``).
``coverage_report()`` and the tests enforce this.
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

# Vision runes run by the world's ward system: kernels ``domination.sixth_sense`` /
# ``domination.deep_ward``, called from ``wards.ward_step`` (docs/modern/WARDS.md).
WORLD = {
    8137: "Sixth Sense (wards via domination.sixth_sense): alive and off cd, track the nearest "
          "untracked enemy ward within 900 unseen by the holder's team; from level 11 reveal it 10 s; cd 250 s",
    8141: "Deep Ward (wards via domination.deep_ward): trinket Totem Wards placed in the enemy jungle "
          "(river too from level 9) get +1 HP and +lin(45, 150, avg level) s",
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
    for m in MODULES:
        for pid, what in m.COVERAGE.items():
            if pid in out:
                raise RuntimeError(f"rune {pid} covered twice: {out[pid]} / {_name(m)}")
            out[pid] = f"{_name(m)}: {what}"
    for table, label in ((WORLD, "WORLD"), (DEFERRED, "DEFERRED"), (STATIC, "STATIC")):
        for pid, why in table.items():
            if pid in out:
                raise RuntimeError(f"rune {pid} is both {label} and {out[pid]}")
            out[pid] = f"{label}: {why}"
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
    for m in MODULES:
        fn = getattr(m, hook, None)
        if fn is not None:
            yield m, fn


def _sub(state, m):
    return getattr(state, _name(m))


def _put(state, m, sub):
    return state._replace(**{_name(m): sub})


def stats(state, page, ctx, ev: RuneEvents) -> ItemStats:
    parts = [fn(_sub(state, m), page, ctx, ev) for m, fn in _each("stats")]
    return combine_stats(zero_stats(ctx.level.shape), *parts)


def debuffs(state, page, ctx, units, ev) -> Debuffs:
    parts = [fn(_sub(state, m), page, ctx, units, ev) for m, fn in _each("debuffs")]
    return combine_debuffs(parts, units.x.shape[0])


def packet_amp(state, page, ctx, units, ev, packets):
    out = jnp.zeros(packets.valid.shape, jnp.float32)
    for m, fn in _each("packet_amp"):
        out = out + fn(_sub(state, m), page, ctx, units, ev, packets)
    return out


def packet_block(state, page, ctx, units, ev, packets):
    out = jnp.zeros(packets.valid.shape, jnp.float32)
    for m, fn in _each("packet_block"):
        out = out + fn(_sub(state, m), page, ctx, units, ev, packets)
    return out


def heal_mult(state, page, ctx, ev):
    out = jnp.ones(ctx.level.shape, jnp.float32)
    for m, fn in _each("heal_mult"):
        out = out * fn(_sub(state, m), page, ctx, ev)
    return out


def _event(hook: str, state, page, ctx, units, ev):
    c, n = ctx.level.shape[0], units.x.shape[0]
    parts = []
    for m, fn in _each(hook):
        sub, eff = fn(_sub(state, m), page, ctx, units, ev)
        state = _put(state, m, sub)
        parts.append(eff)
    return state, merge_effects(parts, c, n)


def on_cast(state, page, ctx, units, ev):
    return _event("on_cast", state, page, ctx, units, ev)


def on_attack(state, page, ctx, units, ev):
    return _event("on_attack", state, page, ctx, units, ev)


def on_hit(state, page, ctx, units, ev):
    return _event("on_hit", state, page, ctx, units, ev)


def on_cc(state, page, ctx, units, ev):
    return _event("on_cc", state, page, ctx, units, ev)


def periodic(state, page, ctx, units, ev):
    return _event("periodic", state, page, ctx, units, ev)


def on_damage(state, page, ctx, units, ev):
    return _event("on_damage", state, page, ctx, units, ev)


def on_takedown(state, page, ctx, units, ev):
    return _event("on_takedown", state, page, ctx, units, ev)


def post_tick(state, page, ctx, units, ev):
    for m, fn in _each("post_tick"):
        state = _put(state, m, fn(_sub(state, m), page, ctx, units, ev))
    return state


def outputs(state, page, ctx, ev) -> RuneOutputs:
    c, i = ctx.level.shape[0], len(catalog().ids)
    parts = [fn(_sub(state, m), page, ctx, ev) for m, fn in _each("outputs")]
    return merge_outputs(parts, c, i) if parts else no_outputs(c, i)


__all__ = ["MODULES", "WORLD", "DEFERRED", "STATIC", "RuneEffectState", "init", "coverage_report", "stats",
           "debuffs", "packet_amp", "packet_block", "heal_mult", "on_cast", "on_attack", "on_hit", "on_cc",
           "periodic", "on_damage", "on_takedown", "post_tick", "outputs"]
