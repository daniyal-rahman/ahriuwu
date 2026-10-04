"""One shared modern world and ONE compiled tick program for the full-tick test modules.

Compiling ``modern_step.step`` takes minutes and ~8 GB on CPU, and the config
is closed over (its arrays become program constants), so every distinct
``jit``/scan -- a different closure, a different ``length=``, a different rune
page -- is another full compile. ``test_modern_step``, ``test_modern_world_rules``,
``test_modern_vision`` and ``obs/tests/test_modern_obs`` therefore share:

* ``world()``: one config (Garen default page; Jax Precision + Resolve with
  Second Wind / Overgrowth, which ``test_modern_world_rules`` needs);
* ``advance(s, orders, ticks)``: one jitted ``fori_loop`` whose tick count is a
  traced argument, so a single step and a 3000-tick run are the same program.
  ``step``/``run`` are thin wrappers with the old call shapes;
* ``refresh(s)``: one jitted ``refresh_visibility``.

Only what needs a genuinely different program builds one (``fast_world`` for
``fog="fast"``, the observation builder/decoder in ``test_modern_obs``).
"""
from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp

from lanerl_jax.sim import modern_rune_data as RD
from lanerl_jax.sim import modern_world as MW

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
    from lanerl_jax.sim import modern_step as MS
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
    from lanerl_jax.sim import modern_step as MS
    fast = MW.build_config(world().loadouts, fog="fast")
    return fast, jax.jit(lambda s: MS.refresh_visibility(s, fast))


def orders(**kw):
    from lanerl_jax.sim import modern_step as MS
    o = MS.no_orders()._asdict()
    for k, v in kw.items():
        o[k] = jnp.asarray(v, o[k].dtype)
    return MS.ModernOrders(**o)
