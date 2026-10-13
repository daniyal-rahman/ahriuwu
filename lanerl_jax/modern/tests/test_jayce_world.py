"""Jayce through the real compiled tick against Garen: the innate Transform and 6-rank skill table, per-stance
cooldowns across transforms, Mercury Cannon's ranged attack (range and missile), Hyper Charge, and Shock Blast
(one top-lane world, own compile)."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.tests import world_harness as H

if not H.artifacts_present():
    pytest.skip("modern map/route artifacts not present", allow_module_level=True)

from lanerl_jax.modern import economy as E  # noqa: E402
from lanerl_jax.modern import world as MS  # noqa: E402
from lanerl_jax.modern.champions import jayce as J  # noqa: E402
from lanerl_jax.modern.world import config as MW  # noqa: E402

orders = H.orders
DT = 1.0 / 30.0
SX, SY = 5600.0, 13620.0                                              # straight top-lane stretch toward +x


@lru_cache(maxsize=1)
def world():
    cfg = MW.build_config((MW.Loadout("Jayce", items=H.ITEMS), MW.Loadout("Garen", items=H.ITEMS)),   # no runes
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


def placed(jay_xy, garen_xy, ranks=(1, 1, 1, 1)):
    s = MS.init_state(world()[0])
    return s._replace(x=s.x.at[0].set(jay_xy[0]).at[1].set(garen_xy[0]),
                      y=s.y.at[0].set(jay_xy[1]).at[1].set(garen_xy[1]), t=jnp.float32(70.0),
                      champ=s.champ._replace(ranks=s.champ.ranks.at[0].set(jnp.asarray(ranks, jnp.int32))
                                             .at[1].set(jnp.asarray([1, 1, 1, 0], jnp.int32))))


def test_innate_transform_and_six_ranks():
    s = MS.init_state(world()[0])
    np.testing.assert_array_equal(np.asarray(s.champ.ranks[0]), [1, 0, 0, 1])      # Q from the order, R innate
    np.testing.assert_allclose(float(s.radius[0]), J.RADIUS)
    # Level 11 with every point placed by the default order: Q reaches rank 6, R stays 1.
    xp11 = float(E.table("need")[11])
    s = s._replace(econ=s.econ._replace(level=s.econ.level.at[0].set(11), xp=s.econ.xp.at[0].set(xp11 + 1.0)))
    s = run(s, MS.no_orders(), 12)                                                  # one point per tick
    r = np.asarray(s.champ.ranks[0])
    np.testing.assert_array_equal(r, [6, 4, 1, 1])                                 # Q > W > E
    s, _ = step(s, orders(level_up=[3, -1]))                                        # R cannot be ranked
    assert int(s.champ.ranks[0, 3]) == 1


def test_stance_cooldowns_survive_transforms():
    s = placed((SX, SY), (SX + 400.0, SY))
    s, _ = step(s, orders(cast_slot=[0, -1], cast_target=[1, -1]))                  # To the Skies! (16 s)
    s = run(s, MS.no_orders(), 20)
    cd_q = float(s.champ.cooldowns[0, 0])
    assert cd_q > 14.0
    s, _ = step(s, orders(cast_slot=[3, -1]))                                       # to Mercury Cannon
    assert float(s.champ.cooldowns[0, 0]) == 0.0 and float(s.champ.alt_cooldowns[0, 0]) > 14.0
    s = run(s, MS.no_orders(), 186)                                                 # the 6 s transform cooldown (30 Hz)
    s, _ = step(s, orders(cast_slot=[3, -1]))                                       # back to Mercury Hammer
    assert not bool(s.kits.jayce.cannon[0])
    np.testing.assert_allclose(float(s.champ.cooldowns[0, 0]), cd_q - DT * 188, atol=0.1)    # kept ticking


def test_mercury_cannon_attacks_from_500_with_a_missile():
    s = placed((SX, SY), (SX + 480.0, SY))
    s, _ = step(s, orders(cast_slot=[3, -1]))
    hp0 = float(s.hp[1])
    s, _ = step(s, orders(attack=[1, -1]))
    s = run(s, orders(attack=[1, -1]), 40)
    assert float(s.hp[1]) < hp0                                                     # hit from 480 without walking
    assert abs(float(s.x[0]) - SX) < 30.0
    # Mercury Hammer at the same distance walks in first.
    s = placed((SX, SY), (SX + 480.0, SY))
    s = run(s, orders(attack=[1, -1]), 10)
    assert float(s.x[0]) > SX + 100.0


def test_shock_blast_hits_garen_at_900():
    s = placed((SX, SY), (SX + 900.0, SY))
    s, _ = step(s, orders(cast_slot=[3, -1]))
    s = run(s, MS.no_orders(), 2)
    hp0 = float(s.hp[1])
    s, _ = step(s, orders(cast_slot=[0, -1], cast_x=[SX + 2000.0, 0.0], cast_y=[SY, 0.0]))
    s = run(s, MS.no_orders(), 30)                                                  # 0.21 s + ~0.53 s of flight
    assert float(s.hp[1]) < hp0
    assert float(s.champ.cooldowns[0, 0]) > 6.0
