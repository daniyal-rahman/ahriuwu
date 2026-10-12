"""Darius through the real compiled tick: Apprehend pulls Garen next to him (knocked up, then slowed),
Hemorrhage ticks through the damage pipeline, Decimate heals, and the Guillotine resets on a rank-3 execution
(one top-lane world, own compile)."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

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
    cfg = MW.build_config((MW.Loadout("Darius", items=H.ITEMS, rune_page=DL.rune_pages("Darius")[0]),
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


def placed(dar_xy, garen_xy, ranks=(1, 1, 1, 1)):
    s = MS.init_state(world()[0])
    return s._replace(x=s.x.at[0].set(dar_xy[0]).at[1].set(garen_xy[0]),
                      y=s.y.at[0].set(dar_xy[1]).at[1].set(garen_xy[1]), t=jnp.float32(70.0),
                      champ=s.champ._replace(ranks=s.champ.ranks.at[0].set(jnp.asarray(ranks, jnp.int32))))


def test_record_radius_and_pathing():
    cfg = world()[0]
    s = MS.init_state(cfg)
    np.testing.assert_allclose(float(s.radius[0]), 65.0)
    np.testing.assert_allclose(np.asarray(cfg.champion_base.pathing_radius), [25.7666, 35.0], rtol=1e-4)
    np.testing.assert_allclose(np.asarray(cfg.champion_base.acquisition_range), [600.0, 400.0])


def test_apprehend_pulls_garen_in_and_knocks_him_up():
    cfg = world()[0]
    mx, my = H.lane_mid(cfg)
    s = placed((mx, my), (mx + 450.0, my))
    s = step(s, orders(cast_slot=[2, -1], cast_x=[mx + 1000.0, 0.0], cast_y=[my, 0.0]))
    s = run(s, MS.no_orders(), 9)                                     # past the 0.25 s cast time
    assert float(s.cc.knockup_until[1]) > float(s.t)
    s = run(s, MS.no_orders(), 25)                                    # 0.7 s pull + rebound
    d = float(np.hypot(s.x[1] - s.x[0], s.y[1] - s.y[0]))
    assert 200.0 <= d <= 260.0, d                                     # measured: ~230 after the rebound
    s = run(s, MS.no_orders(), 3)
    assert float(s.cc.slow[1]) == pytest.approx(0.4) and float(s.cc.slow_until[1]) > float(s.t)


def test_hemorrhage_and_decimate_through_the_damage_pipeline():
    cfg = world()[0]
    mx, my = H.lane_mid(cfg)
    s = placed((mx, my), (mx + 350.0, my))
    s = s._replace(hp=s.hp.at[0].set(300.0))
    hp_g = float(s.hp[1])
    s = step(s, orders(cast_slot=[0, -1]))
    s = run(s, MS.no_orders(), 24)                                    # the 0.75 s heft
    assert float(s.hp[1]) < hp_g and int(s.kits.darius.stacks[0, 1]) == 1
    assert float(s.hp[0]) > 300.0 + 0.17 * 100.0                      # healed 17% of missing health (+ regen)
    hp_g = float(s.hp[1])
    s = run(s, MS.no_orders(), 45)                                    # one bleed tick at 1.25 s
    assert float(s.hp[1]) < hp_g


def test_rank3_guillotine_execution_resets():
    cfg = world()[0]
    mx, my = H.lane_mid(cfg)
    s = placed((mx, my), (mx + 300.0, my), ranks=(1, 1, 1, 3))
    s = s._replace(hp=s.hp.at[1].set(100.0))
    s = step(s, orders(cast_slot=[3, -1], cast_target=[1, -1]))
    for _ in range(20):
        s = step(s, MS.no_orders())
    assert not bool(s.alive[1])
    assert float(s.champ.cooldowns[0, 3]) == 0.0
