"""Nasus through the real compiled tick: record radii, walk-in Spirit Fire, Wither's cripple on the target's attack
speed, Siphoning Strike stacks from a lane last hit and Fury of the Sands health (one top-lane world, own compile)."""
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core import types as W
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
    cfg = MW.build_config((MW.Loadout("Nasus", items=H.ITEMS, rune_page=DL.rune_pages("Nasus")[0]),
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


def placed(nasus_xy, garen_xy, t=70.0, ranks=(1, 1, 1, 1)):
    cfg = world()[0]
    s = MS.init_state(cfg)
    return s._replace(x=s.x.at[0].set(nasus_xy[0]).at[1].set(garen_xy[0]),
                      y=s.y.at[0].set(nasus_xy[1]).at[1].set(garen_xy[1]), t=jnp.float32(t),
                      champ=s.champ._replace(ranks=s.champ.ranks.at[0].set(jnp.asarray(ranks, jnp.int32))))


def test_record_radius_and_pathing():
    cfg = world()[0]
    s = MS.init_state(cfg)
    np.testing.assert_allclose(np.asarray(s.radius[:2]), [80.0, 65.0])
    np.testing.assert_allclose(np.asarray(cfg.champion_base.pathing_radius), [50.0, 35.0])
    np.testing.assert_allclose(np.asarray(cfg.champion_base.acquisition_range), [350.0, 400.0])


def test_out_of_range_spirit_fire_walks_in_and_burns():
    cfg = world()[0]
    mx, my = H.lane_mid(cfg)
    s = placed((mx - 1000.0, my), (mx, my))
    s = step(s, orders(cast_slot=[2, -1], cast_x=[mx, 0.0], cast_y=[my, 0.0]))
    assert int(s.champ.queued_cast.slot[0]) == 2 and float(s.champ.cooldowns[0, 2]) == 0.0
    hp0, mana0 = float(s.hp[1]), float(s.champ.mana[0])
    s = run(s, MS.no_orders(), 45)
    assert float(s.champ.cooldowns[0, 2]) > 0.0 and int(s.champ.queued_cast.slot[0]) == -1
    d = float(np.hypot(s.x[0] - mx, s.y[0] - my))
    assert 600.0 < d <= 650.0                                           # cast from just inside its range
    assert float(s.champ.mana[0]) < mana0 - 59.0
    s = run(s, MS.no_orders(), 60)
    assert float(s.hp[1]) < hp0                                         # Garen stood in the fire


def test_wither_slows_and_cripples_garens_attack_speed():
    cfg = world()[0]
    mx, my = H.lane_mid(cfg)
    s = placed((mx - 400.0, my), (mx, my))
    s = step(s, MS.no_orders())
    as0, slow0 = float(s.attack_speed[1]), float(s.cc.slow[1])
    s = step(s, orders(cast_slot=[1, -1], cast_target=[1, -1]))
    s = run(s, MS.no_orders(), 10)                                      # past the 0.25 s cast time
    np.testing.assert_allclose(float(s.cc.slow[1]), 0.35, rtol=1e-5)
    assert slow0 == 0.0
    s = step(s, MS.no_orders())
    np.testing.assert_allclose(float(s.attack_speed[1]), as0 * (1.0 - 0.75 * 0.35), rtol=1e-3)
    s = run(s, MS.no_orders(), 75)                                      # ~2.8 s in: two steps
    np.testing.assert_allclose(float(s.cc.slow[1]), 0.35 + 2 * 0.03, rtol=1e-4)
    s = run(s, MS.no_orders(), 75)                                      # ~5.3 s: over
    assert float(s.attack_speed[1]) == pytest.approx(as0, rel=1e-4)


def test_siphoning_strike_last_hit_grants_stacks_in_world():
    cfg, _, _ = world()
    s = run(MS.init_state(cfg), MS.no_orders(), 40 * 30)                # first wave walking, nobody fighting it
    k, a, team = np.asarray(s.kind), np.asarray(s.alive), np.asarray(s.team)
    m = int(np.flatnonzero((k == W.KIND_MINION) & a & (team == 1) & (np.asarray(s.sub) == 0))[0])
    mx, my = float(s.x[m]), float(s.y[m])
    s = s._replace(x=s.x.at[0].set(mx - 150.0), y=s.y.at[0].set(my), hp=s.hp.at[m].set(5.0),
                   champ=s.champ._replace(ranks=s.champ.ranks.at[0].set(jnp.asarray([1, 0, 0, 0], jnp.int32))))
    s = step(s, orders(cast_slot=[0, -1]))
    for _ in range(45):
        s = step(s, orders(attack=[m, -1]))
        if not bool(s.alive[m]):
            break
    assert not bool(s.alive[m])
    assert float(s.kits.nasus.q_stacks[0]) == 4.0 and float(s.champ.cooldowns[0, 0]) > 0.0


def test_fury_of_the_sands_raises_max_health():
    cfg = world()[0]
    mx, my = H.lane_mid(cfg)
    s = placed((mx, my), (mx + 2000.0, my))
    base = run(s, MS.no_orders(), 11)
    s = step(s, orders(cast_slot=[3, -1]))
    s = run(s, MS.no_orders(), 10)
    np.testing.assert_allclose(float(s.max_hp[0]), float(base.max_hp[0]) + 300.0, rtol=1e-5)
    assert float(s.champ.cooldowns[0, 3]) > 100.0
