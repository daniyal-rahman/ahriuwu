"""Small fixed-shape helpers for rune-effect unit tests.

Builds on ``item_harness`` (units, ctx, attack, cast, kills, resolve): unit 0
and 1 are the two champions (holders 0 and 1, teams 0 and 1).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.runes import catalog as RD
from lanerl_jax.modern.items.effects.core import Report
from lanerl_jax.modern.runes.effects.core import CombatClocks, rune_events
from lanerl_jax.modern.tests import item_harness as H


def perks(*lists):
    """(C, R) int32 page counts from raw perk id lists (no legality checks)."""
    cat = RD.rune_catalog()
    out = np.zeros((len(lists), len(cat.ids)), np.int32)
    for c, ids in enumerate(lists):
        for pid in ids:
            out[c, cat.row(pid)] += 1
    return jnp.asarray(out)


def ev(ctx, n_units, **kw):
    return rune_events(ctx, n_units, **kw)


def clocks(n=2, *, last_combat=-1e9, last_champion_combat=-1e9, last_hit_by_champion=-1e9,
           champion_combat_start=-1e9, struck_first=False):
    v = lambda x: jnp.broadcast_to(jnp.asarray(x, jnp.float32), (n,))
    return CombatClocks(v(last_combat), v(last_champion_combat), v(last_hit_by_champion),
                        v(champion_combat_start), jnp.broadcast_to(jnp.asarray(struck_first), (n,)))


def report(p, u, **kw) -> Report:
    """Resolve packets against units ``u`` (item_harness.resolve) and return the Report."""
    rep, _ = H.resolve(p, u, **kw)
    return rep


def hit_packet(src, dst, raw, dtype=D.PHYSICAL, flags=D.BASIC_ATTACK, cast_id=0, item=0):
    return D.packets(jnp.ones(1, bool), src, dst, raw, dtype, flags, item=item, cast_id=cast_id)


def total(p, *, rune=None, dst=None, src=None):
    """Sum of valid raw packet damage with rune provenance ``rune``."""
    return H.packet_total(p, dst=dst, item=None if rune is None else -rune, src=src)
