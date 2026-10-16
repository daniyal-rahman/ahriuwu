"""Ezreal through the real compiled tick against Garen: the ranged attack (range and missile), Mystic Shot's hit and
its cooldown refund, and Arcane Shift's blink (one top-lane world, own compile)."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.tests import world_harness as H

if not H.artifacts_present():
    pytest.skip("modern map/route artifacts not present", allow_module_level=True)

from lanerl_jax.modern import world as MS  # noqa: E402
from lanerl_jax.modern.champions import ezreal as Z  # noqa: E402
from lanerl_jax.modern.world import config as MW  # noqa: E402

orders = H.orders
DT = 1.0 / 30.0
SX, SY = 5600.0, 13620.0                                              # straight top-lane stretch toward +x


@lru_cache(maxsize=1)
def world():
    cfg = MW.build_config((MW.Loadout("Ezreal", items=H.ITEMS), MW.Loadout("Garen", items=H.ITEMS)),   # no runes
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


def placed(ez_xy, garen_xy, ranks=(1, 1, 1, 1)):
    s = MS.init_state(world()[0])
    return s._replace(x=s.x.at[0].set(ez_xy[0]).at[1].set(garen_xy[0]),
                      y=s.y.at[0].set(ez_xy[1]).at[1].set(garen_xy[1]), t=jnp.float32(70.0),
                      champ=s.champ._replace(ranks=s.champ.ranks.at[0].set(jnp.asarray(ranks, jnp.int32))
                                             .at[1].set(jnp.asarray([1, 1, 1, 0], jnp.int32))))


def test_ranged_attack_from_540_with_a_missile():
    s = placed((SX, SY), (SX + 540.0, SY))
    hp0 = float(s.hp[1])
    s = run(s, orders(attack=[1, -1]), 40)
    assert float(s.hp[1]) < hp0                                                     # hit without walking
    assert abs(float(s.x[0]) - SX) < 30.0


def test_mystic_shot_hits_and_refunds_every_cooldown():
    s = placed((SX, SY), (SX + 900.0, SY))
    s, _ = step(s, orders(cast_slot=[1, -1], cast_x=[SX - 2000.0, 0.0], cast_y=[SY, 0.0]))   # W away from Garen
    s = run(s, MS.no_orders(), 10)
    w0 = float(s.champ.cooldowns[0, 1])
    hp0 = float(s.hp[1])
    s, _ = step(s, orders(cast_slot=[0, -1], cast_x=[SX + 2000.0, 0.0], cast_y=[SY, 0.0]))
    q0 = float(s.champ.cooldowns[0, 0])
    np.testing.assert_allclose(q0, 5.5, atol=0.05)
    n = 25                                                                          # 0.25 s + ~0.42 s of flight
    s = run(s, MS.no_orders(), n)
    assert float(s.hp[1]) < hp0
    elapsed = DT * (n + 1)
    np.testing.assert_allclose(float(s.champ.cooldowns[0, 1]), w0 - elapsed - Z.Q_REFUND, atol=0.05)
    np.testing.assert_allclose(float(s.champ.cooldowns[0, 0]), q0 - DT * n - Z.Q_REFUND, atol=0.05)


def test_arcane_shift_blinks_475_toward_the_point():
    s = placed((SX, SY), (SX + 1500.0, SY + 1500.0))
    s, _ = step(s, orders(cast_slot=[2, -1], cast_x=[SX + 2000.0, 0.0], cast_y=[SY, 0.0]))
    s = run(s, MS.no_orders(), 10)
    np.testing.assert_allclose(float(s.x[0]) - SX, Z.E_BLINK, atol=5.0)
    np.testing.assert_allclose(float(s.y[0]), SY, atol=5.0)
