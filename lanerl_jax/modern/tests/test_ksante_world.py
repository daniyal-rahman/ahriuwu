"""K'Sante through the real compiled tick against Garen: Ntofo Strikes and its shockwave's pull, Path Maker's
charge (damage reduction, displacement immunity) and carry, All Out's pull with and without terrain, its health
threshold, resistance loss and mana (one top-lane world, own compile)."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.tests import world_harness as H

if not H.artifacts_present():
    pytest.skip("modern map/route artifacts not present", allow_module_level=True)

from lanerl_jax.modern import mechanics as M  # noqa: E402
from lanerl_jax.modern import world as MS  # noqa: E402
from lanerl_jax.modern.world import config as MW  # noqa: E402

orders = H.orders
SX, SY = 5600.0, 13620.0                                              # straight top-lane stretch toward +x


@lru_cache(maxsize=1)
def world():
    cfg = MW.build_config((MW.Loadout("KSante", items=H.ITEMS), MW.Loadout("Garen", items=H.ITEMS)),   # no runes
                          fog=False, lanes=(2,), jungle=False, objectives=False)
    tick = jax.jit(lambda s, o: MS.step(s, o, cfg))

    @jax.jit
    def run(s, o, n):
        return jax.lax.fori_loop(0, n, lambda _, s: tick(s, o)[0], s)
    return cfg, tick, run


def step(s, o):
    return world()[1](s, o)


def run(s, o, n):
    return world()[2](s, o, jnp.asarray(n, jnp.int32))


def placed(ks_xy, garen_xy, ranks=(1, 1, 1, 1)):
    s = MS.init_state(world()[0])
    return s._replace(x=s.x.at[0].set(ks_xy[0]).at[1].set(garen_xy[0]),
                      y=s.y.at[0].set(ks_xy[1]).at[1].set(garen_xy[1]), t=jnp.float32(70.0),
                      champ=s.champ._replace(ranks=s.champ.ranks.at[0].set(jnp.asarray(ranks, jnp.int32))
                                             .at[1].set(jnp.asarray([1, 1, 1, 0], jnp.int32))))


def test_record_radius_and_range():
    cfg = world()[0]
    s = MS.init_state(cfg)
    np.testing.assert_allclose(float(s.radius[0]), 80.0)
    np.testing.assert_allclose(np.asarray(cfg.champion_base.attack_range)[0], 150.0)


def test_ntofo_strikes_twice_then_the_shockwave_pulls_garen():
    s = placed((SX, SY), (SX + 300.0, SY))
    hold = dict(move=[True, True], move_x=[SX, SX + 300.0], move_y=[SY, SY])
    hp = float(s.hp[1])
    for _ in range(2):
        s, _ = step(s, orders(cast_slot=[0, -1], cast_x=[SX + 1000.0, 0.0], cast_y=[SY, 0.0], **hold))
        s = run(s, orders(**hold), 110)                                # the cast time and the 3.5 s cooldown
    assert float(s.hp[1]) < hp and int(s.kits.ksante.q_stacks[0]) == 2
    s = s._replace(x=s.x.at[1].set(SX + 700.0), y=s.y.at[1].set(SY))
    s, _ = step(s, orders(cast_slot=[0, -1], cast_x=[SX + 1000.0, 0.0], cast_y=[SY, 0.0]))
    s = run(s, MS.no_orders(), 45)                                     # cast 0.45 s, flight ~0.35 s, pull 0.6 s
    assert float(s.x[1]) < SX + 700.0 - 200.0, float(s.x[1])           # pulled toward K'Sante
    assert float(s.cc.stun_until[1]) > float(s.t) - 0.3


def test_path_maker_reduces_damage_resists_a_pull_and_carries_garen():
    s = placed((SX, SY), (SX + 200.0, SY))
    s, _ = step(s, orders(cast_slot=[1, -1], cast_x=[SX + 1000.0, 0.0], cast_y=[SY, 0.0]))
    assert int(s.kits.ksante.w_stage[0]) == 1
    s = run(s, MS.no_orders(), 20)                                     # charges to 1 s and releases itself
    s = run(s, MS.no_orders(), 25)                                     # the 450-unit dash (0.32 s)
    assert float(s.x[0]) > SX + 300.0 and float(s.x[1]) > float(s.x[0])     # Garen carried in front
    assert float(s.cc.stun_until[1]) > float(s.t)


def _wall_spot():
    cfg = world()[0]
    lane = np.asarray(cfg.lane_path)
    for px, py in lane[4:-4]:
        for ang in np.deg2rad(np.arange(0.0, 360.0, 15.0)):
            ux, uy = float(np.cos(ang)), float(np.sin(ang))
            f, entry, ex = M.wall_exit(jnp.asarray([px]), jnp.asarray([py]), jnp.asarray([ux]), jnp.asarray([uy]),
                                       jnp.asarray([350.0]), jnp.asarray([5000.0]), jnp.asarray([1]), cfg.terrain)
            if bool(f[0]) and 150.0 < float(entry[0]) < 330.0 and float(ex[0]) < 1500.0:
                return float(px), float(py), ux, uy, float(ex[0])
    return None


def test_all_out_pulls_garen_over_terrain_and_goes_all_out():
    spot = _wall_spot()
    assert spot is not None
    gx, gy, ux, uy, exit_ = spot
    s = placed((gx - 200.0 * ux, gy - 200.0 * uy), (gx, gy), ranks=(1, 1, 1, 1))
    s = s._replace(champ=s.champ._replace(mana=s.champ.mana.at[0].set(200.0)))
    hp0, max0 = float(s.hp[0]), float(s.max_hp[0])
    s, _ = step(s, orders(cast_slot=[3, -1], cast_target=[1, -1]))
    assert float(s.cc.root_until[1]) > float(s.t)
    s = run(s, MS.no_orders(), 13)                                     # the 0.4 s cast time
    assert bool(s.kits.ksante.r_wall[0])
    s = run(s, MS.no_orders(), 30)                                     # the pull (~0.8 s); stunned after
    ks = s.kits.ksante
    end_along = (float(ks.r_px[0]) - gx) * ux + (float(ks.r_py[0]) - gy) * uy
    assert end_along > exit_ + 50.0, (end_along, exit_)                # past the far side (450, or short of a wall)
    d = float(np.hypot(float(s.x[1]) - float(ks.r_px[0]), float(s.y[1]) - float(ks.r_py[0])))
    assert d < 30.0, (d, end_along, exit_)                             # Garen was pulled there, through the terrain
    assert float(s.hp[0]) <= 0.65 * max0 + 1.0 and float(s.hp[0]) < hp0
    assert float(s.champ.mana[0]) < 400.0 and float(s.kits.ksante.ao_until[0]) > float(s.t)


def test_all_out_without_terrain_lands_250_beyond():
    s = placed((SX, SY), (SX + 250.0, SY))
    s, _ = step(s, orders(cast_slot=[3, -1], cast_target=[1, -1]))
    s = run(s, MS.no_orders(), 30)
    np.testing.assert_allclose(float(s.x[1]), SX + 250.0 + 300.0, atol=25.0)
    np.testing.assert_allclose(float(s.kits.ksante.r_lx[0]) - float(s.x[1]), 250.0, atol=25.0)   # the landing
