"""Malphite through the real compiled tick: Granite Shield in the shield store (one instance, refilled after the
delay), Ground Slam's cripple on Garen's attack speed and Unstoppable Force's
dash and knock-up (one top-lane world, own compile)."""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.tests import world_harness as H

if not H.artifacts_present():
    pytest.skip("modern map/route artifacts not present", allow_module_level=True)

from lanerl_jax.modern import world as MS  # noqa: E402
from lanerl_jax.modern.data import loadouts as DL  # noqa: E402
from lanerl_jax.modern.runes import catalog as RD  # noqa: E402
from lanerl_jax.modern.world import config as MW  # noqa: E402

orders = H.orders


@lru_cache(maxsize=1)
def world():
    cfg = MW.build_config((MW.Loadout("Malphite", items=H.ITEMS, rune_page=DL.rune_pages("Malphite")[0]),
                           MW.Loadout("Garen", items=H.ITEMS, rune_page=RD.GAREN_DEFAULT_PAGE)),
                          fog=False, lanes=(2,), jungle=False, objectives=False)
    tick = jax.jit(lambda s, o: MS.step(s, o, cfg))

    @jax.jit
    def run(s, o, n):
        return jax.lax.fori_loop(0, n, lambda _, s: tick(s, o)[0], s)
    return cfg, tick, run


def step(s, o):
    return world()[1](s, o)[0]


def run(s, o, n):
    return world()[2](s, o, jnp.asarray(n, jnp.int32))


def granite(s):
    v = D.shield_value(s.shields, s.t)[0]
    return float(jnp.sum(jnp.where(s.shields.source[0] == D.SOURCE_KIT, v, 0.0)))


def placed(mal_xy, garen_xy, ranks=(1, 1, 1, 1)):
    s = MS.init_state(world()[0])
    return s._replace(x=s.x.at[0].set(mal_xy[0]).at[1].set(garen_xy[0]),
                      y=s.y.at[0].set(mal_xy[1]).at[1].set(garen_xy[1]), t=jnp.float32(70.0),
                      champ=s.champ._replace(ranks=s.champ.ranks.at[0].set(jnp.asarray(ranks, jnp.int32))))


def test_granite_shield_one_instance_refilled_after_the_delay():
    cfg = world()[0]
    mx, my = H.lane_mid(cfg)
    s = placed((mx, my), (mx + 3000.0, my))
    s = run(s, MS.no_orders(), 3)
    full = granite(s)
    np.testing.assert_allclose(full, 0.1 * float(s.max_hp[0]), rtol=1e-4)
    s = s._replace(shields=s.shields._replace(amount=jnp.where(s.shields.source == D.SOURCE_KIT, 0.0,
                                                               s.shields.amount)))   # broken
    s = run(s, MS.no_orders(), 2)
    np.testing.assert_allclose(granite(s), full, rtol=1e-4)            # never hurt: refilled at once
    assert int(jnp.sum((s.shields.source[0] == D.SOURCE_KIT) & (s.shields.amount[0] > 0))) == 1


def test_ground_slam_cripples_garens_attack_speed():
    cfg = world()[0]
    mx, my = H.lane_mid(cfg)
    s = placed((mx, my), (mx + 200.0, my))
    s = step(s, MS.no_orders())
    as0 = float(s.attack_speed[1])
    s = step(s, orders(cast_slot=[2, -1]))
    s = run(s, MS.no_orders(), 9)
    np.testing.assert_allclose(float(s.attack_speed[1]), as0 * (1.0 - 0.30), rtol=1e-3)
    s = run(s, MS.no_orders(), 95)
    np.testing.assert_allclose(float(s.attack_speed[1]), as0, rtol=1e-4)


def test_unstoppable_force_dashes_and_knocks_up():
    sx, sy = 5600.0, 13620.0                                          # straight top-lane stretch toward +x
    s = placed((sx, sy), (6450.0, 13640.0))
    hp_g = float(s.hp[1])
    s = step(s, orders(cast_slot=[3, -1], cast_x=[7600.0, 0.0], cast_y=[13650.0, 0.0]))
    s = run(s, MS.no_orders(), 20)                                    # 1000 / (1500 + 335) ~ 0.55 s
    moved = float(np.hypot(s.x[0] - sx, s.y[0] - sy))
    assert 900.0 < moved <= 1010.0, moved
    assert float(s.hp[1]) < hp_g and float(s.cc.knockup_until[1]) > float(s.t)
