"""One shared world and ONE compiled tick program for the full-tick test modules.

Compiling ``step`` takes minutes and ~8 GB on CPU, and the config is closed over, so every distinct jit/scan is
another full compile. ``test_step``, ``test_world_rules``, ``test_vision`` and ``obs/tests/test_obs`` share
``world()`` (Jax runs Precision + Resolve with Overgrowth), ``advance`` (one jitted ``fori_loop`` whose tick count
is traced; ``step``/``run`` wrap it) and ``refresh``. ``fast_world`` is the ``fog="fast"`` variant.
"""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp

from lanerl_jax.modern.runes import catalog as RD
from lanerl_jax.modern.world import config as MW

ITEMS = (1055, 2003)                  # Doran's Blade, Health Potion
JAX_PAGE = RD.RunePage(RD.PRECISION, 8010, (9111, 9104, 8299), RD.RESOLVE, (8444, 8451), (5005, 5008, 5001))


def artifacts_present() -> bool:
    return MW.DEFAULT_MAP.exists() and MW.DEFAULT_ROUTES.exists()


@lru_cache(maxsize=1)
def world():
    return MW.build_config((MW.Loadout("Garen", items=ITEMS, rune_page=RD.GAREN_DEFAULT_PAGE),
                            MW.Loadout("Jax", items=ITEMS, rune_page=JAX_PAGE)))


@lru_cache(maxsize=1)
def _programs():
    from lanerl_jax.modern import world as MS
    cfg = world()
    tick = jax.jit(lambda s, o: MS.step(s, o, cfg))

    def advance(s, o, ticks):
        ev0 = jax.tree.map(lambda a: jnp.zeros(a.shape, a.dtype), jax.eval_shape(tick, s, o)[1])
        z = jnp.zeros((), ev0.packet_overflow.dtype), jnp.zeros((), ev0.missile_overflow.dtype)

        def body(_, carry):
            s, _, po, mo = carry
            s, e = tick(s, o)
            return s, e, jnp.maximum(po, e.packet_overflow.max()), jnp.maximum(mo, e.missile_overflow.max())
        return jax.lax.fori_loop(0, ticks, body, (s, ev0, *z))

    return jax.jit(advance), jax.jit(lambda s: MS.refresh_visibility(s, cfg))


def advance(s, o, ticks: int):
    """``(state, last-tick events, max packet overflow, max missile overflow)`` after ``ticks`` ticks of ``o``."""
    return _programs()[0](s, o, jnp.asarray(ticks, jnp.int32))


def step(s, o):
    """One tick: ``(state, events)`` (same program as ``run``)."""
    s, e, _, _ = advance(s, o, 1)
    return s, e


def run(s, o, ticks: int):
    """``ticks`` ticks with the same orders: ``(state, (max packet overflow, max missile overflow))``."""
    s, _, po, mo = advance(s, o, ticks)
    return s, (po, mo)


def refresh(s):
    return _programs()[1](s)


@lru_cache(maxsize=1)
def fast_world():
    """Same world with ``fog="fast"`` (brush lookup, walls ignored); visibility only, no step."""
    from lanerl_jax.modern import world as MS
    fast = MW.build_config(world().loadouts, fog="fast")
    return fast, jax.jit(lambda s: MS.refresh_visibility(s, fast))


def lane_mid(cfg):
    """(x, y) of the top-lane midpoint."""
    lane = cfg.lane_path
    return tuple(float(v) for v in lane[len(lane) // 2])


def orders(**kw):
    from lanerl_jax.modern import world as MS
    o = MS.no_orders()._asdict()
    for k, v in kw.items():
        o[k] = jnp.asarray(v, o[k].dtype)
    return MS.ModernOrders(**o)
