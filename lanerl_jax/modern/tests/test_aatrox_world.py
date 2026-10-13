"""Aatrox through the real compiled tick: The Darkin Blade holds him in place through its wind-up and hits Garen
in the sweetspot, Umbral Dash moves him mid wind-up, Infernal Chains pulls Garen back into the zone, Deathbringer
Stance's empowered attack, and World Ender's speed and ghosting (one top-lane world, own compile)."""

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
SX, SY = 5600.0, 13620.0                                              # straight top-lane stretch toward +x


@lru_cache(maxsize=1)
def world():
    cfg = MW.build_config((MW.Loadout("Aatrox", items=H.ITEMS, rune_page=DL.rune_pages("Aatrox")[0]),
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


def placed(aat_xy, garen_xy, ranks=(1, 1, 1, 1), passive_ready=False):
    s = MS.init_state(world()[0])
    aat = s.kits.aatrox
    ready = jnp.float32(0.0 if passive_ready else 1e6)
    kits = s.kits._replace(aatrox=aat._replace(p_ready_at=aat.p_ready_at.at[0].set(ready)))
    return s._replace(x=s.x.at[0].set(aat_xy[0]).at[1].set(garen_xy[0]),
                      y=s.y.at[0].set(aat_xy[1]).at[1].set(garen_xy[1]), t=jnp.float32(70.0), kits=kits,
                      champ=s.champ._replace(ranks=s.champ.ranks.at[0].set(jnp.asarray(ranks, jnp.int32))))


def test_record_radius_pathing_and_no_mana():
    cfg = world()[0]
    s = MS.init_state(cfg)
    np.testing.assert_allclose(float(s.radius[0]), 65.0)
    np.testing.assert_allclose(np.asarray(cfg.champion_base.pathing_radius), [35.0, 35.0])
    np.testing.assert_allclose(np.asarray(cfg.champion_base.acquisition_range), [475.0, 400.0])
    assert float(s.champ.mana[0]) == 0.0


def test_darkin_blade_holds_him_then_sweetspots_garen():
    s = placed((SX, SY), (SX + 550.0, SY + 20.0))
    hp = float(s.hp[1])
    move = dict(move=[True, False], move_x=[SX - 1000.0, 0.0], move_y=[SY, 0.0])
    s = step(s, orders(cast_slot=[0, -1], cast_x=[SX + 1000.0, 0.0], cast_y=[SY, 0.0]))
    s = run(s, orders(**move), 15)                                     # 0.5 s of the 0.6 s wind-up
    assert abs(float(s.x[0]) - SX) < 1.0 and float(s.hp[1]) == hp     # held, nothing yet
    s = run(s, orders(**move), 4)
    assert float(s.hp[1]) < hp and float(s.cc.knockup_until[1]) > float(s.t)
    s = run(s, orders(**move), 5)
    assert float(s.x[0]) < SX - 20.0                                  # free to walk once it fired


def test_umbral_dash_mid_wind_up_moves_the_blade():
    s = placed((SX, SY), (SX + 900.0, SY + 20.0))
    hp = float(s.hp[1])
    s = step(s, orders(cast_slot=[0, -1], cast_x=[SX + 1000.0, 0.0], cast_y=[SY, 0.0]))
    s = run(s, MS.no_orders(), 3)
    s = step(s, orders(cast_slot=[2, -1], cast_x=[SX + 1000.0, 0.0], cast_y=[SY, 0.0]))
    s = run(s, MS.no_orders(), 20)
    assert float(s.x[0]) > SX + 295.0                                 # the 300 dash
    assert float(s.hp[1]) < hp                                         # 900 away: only reachable after the dash


def test_infernal_chains_pulls_garen_back_into_the_zone():
    s = placed((SX, SY), (SX + 500.0, SY + 20.0), ranks=(1, 1, 1, 1))
    s = step(s, orders(cast_slot=[1, -1], cast_x=[SX + 1000.0, 0.0], cast_y=[SY + 20.0, 0.0]))
    s = run(s, MS.no_orders(), 18)                                     # 0.25 s cast + ~435 / 1800 flight
    assert int(s.kits.aatrox.w_target[0]) == 1
    cx, cy = float(s.kits.aatrox.w_cx[0]), float(s.kits.aatrox.w_cy[0])
    hp = float(s.hp[1])
    s = run(s, orders(move=[False, True], move_x=[0.0, SX + 2000.0], move_y=[0.0, SY + 20.0]), 40)
    away = float(np.hypot(s.x[1] - cx, s.y[1] - cy))
    assert 150.0 < away < 650.0, away                                  # slowed Garen is still inside
    s = run(s, orders(move=[False, True], move_x=[0.0, SX + 2000.0], move_y=[0.0, SY + 20.0]), 16)
    back = float(np.hypot(s.x[1] - cx, s.y[1] - cy))
    assert back < 40.0 and float(s.hp[1]) < hp, back


def test_deathbringer_empowered_attack():
    s = placed((SX, SY), (SX + 170.0, SY), passive_ready=True)
    hp = float(s.hp[1])
    s = run(s, orders(attack=[1, -1]), 25)
    assert float(s.kits.aatrox.p_ready_at[0]) > float(s.t) + 15.0     # consumed; level 1 cooldown 22 s
    assert hp - float(s.hp[1]) > 0.04 * float(s.max_hp[1])            # more than the 4% bonus alone


def test_world_ender_speed_and_ghost():
    s = placed((SX, SY), (SX + 3000.0, SY), ranks=(1, 1, 1, 1))
    s = run(s, orders(move=[True, False], move_x=[SX + 2000.0, 0.0], move_y=[SY, 0.0]), 10)
    x0 = float(s.x[0])
    s = run(s, orders(move=[True, False], move_x=[SX + 2000.0, 0.0], move_y=[SY, 0.0]), 10)
    walk = float(s.x[0]) - x0
    s = step(s, orders(cast_slot=[3, -1]))
    s = run(s, orders(move=[True, False], move_x=[SX + 2000.0, 0.0], move_y=[SY, 0.0]), 9)
    x1 = float(s.x[0])
    s = run(s, orders(move=[True, False], move_x=[SX + 2000.0, 0.0], move_y=[SY, 0.0]), 10)
    assert float(s.x[0]) - x1 > 1.3 * walk                            # +60% decaying move speed
    assert float(s.kits.aatrox.r_until[0]) > float(s.t) + 8.0
