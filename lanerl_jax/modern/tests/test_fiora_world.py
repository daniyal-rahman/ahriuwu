"""Fiora through the real compiled tick against Darius: Riposte negates his damage and his Apprehend pull and stuns
him with the stab, Bladework's second attack crits, Lunge dashes and stabs (one top-lane world, own compile)."""

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.tests import world_harness as H

if not H.artifacts_present():
    pytest.skip("modern map/route artifacts not present", allow_module_level=True)

from lanerl_jax.modern import world as MS  # noqa: E402
from lanerl_jax.modern.world import config as MW  # noqa: E402

orders = H.orders
SX, SY = 5600.0, 13620.0                                              # straight top-lane stretch toward +x


@lru_cache(maxsize=1)
def world():
    cfg = MW.build_config((MW.Loadout("Fiora", items=H.ITEMS), MW.Loadout("Darius", items=H.ITEMS)),   # no runes
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


def placed(fio_xy, dar_xy, ranks=(1, 1, 1, 1), dar_ranks=(1, 1, 1, 1)):
    s = MS.init_state(world()[0])
    return s._replace(x=s.x.at[0].set(fio_xy[0]).at[1].set(dar_xy[0]),
                      y=s.y.at[0].set(fio_xy[1]).at[1].set(dar_xy[1]), t=jnp.float32(70.0),
                      champ=s.champ._replace(ranks=s.champ.ranks.at[0].set(jnp.asarray(ranks, jnp.int32))
                                             .at[1].set(jnp.asarray(dar_ranks, jnp.int32))))


def test_record_radius_pathing_and_windup():
    cfg = world()[0]
    s = MS.init_state(cfg)
    np.testing.assert_allclose(float(s.radius[0]), 65.0)
    np.testing.assert_allclose(np.asarray(cfg.champion_base.pathing_radius)[0], 35.0)
    np.testing.assert_allclose(np.asarray(cfg.champion_base.acquisition_range)[0], 600.0)
    np.testing.assert_allclose(np.asarray(cfg.champion_base.windup_percent)[0], 0.2 / 1.45, rtol=1e-5)


def test_riposte_negates_the_pull_and_stuns_darius():
    s = placed((SX, SY), (SX + 400.0, SY))
    hp = float(s.hp[0])
    # Fiora parries; Darius's Apprehend (0.25 s cast time) lands inside the 0.75 s window.
    s = step(s, orders(cast_slot=[1, 2], cast_x=[SX + 1000.0, SX - 1000.0], cast_y=[SY, SY]))
    s = run(s, MS.no_orders(), 12)
    # Not pulled (the cast tick itself still lets idle acquisition step her ~11.5 toward Darius).
    assert abs(float(s.x[0]) - SX) < 15.0
    assert float(s.cc.knockup_until[0]) <= float(s.t)
    s = run(s, MS.no_orders(), 6)
    assert float(s.cc.stun_until[1]) > float(s.t) + 1.5               # the stab stunned (2 s)
    assert float(s.hp[0]) >= hp - 1.0


def test_riposte_without_immobilize_slows_and_blocks_damage():
    s = placed((SX, SY), (SX + 150.0, SY))
    hp = float(s.hp[0])
    s = step(s, orders(cast_slot=[1, -1], cast_x=[SX + 1000.0, 0.0], cast_y=[SY, 0.0], attack=[-1, 0]))
    s = run(s, orders(attack=[-1, 0]), 20)
    assert float(s.hp[0]) >= hp - 1.0                                 # Darius's attacks negated
    assert float(s.cc.slow[1]) == pytest.approx(0.5) and float(s.cc.stun_until[1]) <= float(s.t)


def test_bladework_second_attack_crits():
    s = placed((SX, SY), (SX + 160.0, SY), ranks=(1, 1, 1, 1))
    s = step(s, orders(cast_slot=[2, -1]))
    drops, hp = [], float(s.hp[1])
    for _ in range(60):
        s = step(s, orders(attack=[1, -1]))
        if float(s.hp[1]) < hp - 1.0:
            drops.append(hp - float(s.hp[1]))
        hp = float(s.hp[1])
        if len(drops) == 2:
            break
    assert len(drops) == 2
    np.testing.assert_allclose(drops[1] / drops[0], 1.6, rtol=0.03)    # rank 1: 160% of the attack
    assert float(s.cc.slow[1]) == pytest.approx(0.3)


def test_lunge_dashes_and_stabs():
    s = placed((SX, SY), (SX + 600.0, SY))
    hp = float(s.hp[1])
    s = step(s, orders(cast_slot=[0, -1], cast_x=[SX + 1000.0, 0.0], cast_y=[SY, 0.0]))
    s = run(s, MS.no_orders(), 15)
    assert 395.0 < float(s.x[0]) - SX <= 405.0                        # the full 400 (first-tick dash fix)
    assert float(s.hp[1]) < hp
    assert 6.0 < float(s.champ.cooldowns[0, 0]) <= 6.5                # halved at the stab (~0.37 s after the cast)
