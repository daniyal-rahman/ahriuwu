"""Yone through the real compiled tick against Garen: the Azakana sword's half-magic attack, Mortal Steel's
empowered dash and knock-up, Soul Unbound out and back with the mark detonation, Fate Sealed's blink and pull
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
from lanerl_jax.modern.core import damage as D  # noqa: E402
from lanerl_jax.modern.world import config as MW  # noqa: E402

orders = H.orders
SX, SY = 5600.0, 13620.0                                              # straight top-lane stretch toward +x


@lru_cache(maxsize=1)
def world():
    cfg = MW.build_config((MW.Loadout("Yone", items=H.ITEMS), MW.Loadout("Garen", items=H.ITEMS)),   # no runes
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


def placed(yone_xy, garen_xy, ranks=(1, 1, 1, 1)):
    s = MS.init_state(world()[0])
    return s._replace(x=s.x.at[0].set(yone_xy[0]).at[1].set(garen_xy[0]),
                      y=s.y.at[0].set(yone_xy[1]).at[1].set(garen_xy[1]), t=jnp.float32(70.0),
                      champ=s.champ._replace(ranks=s.champ.ranks.at[0].set(jnp.asarray(ranks, jnp.int32))
                                             .at[1].set(jnp.asarray([1, 1, 1, 0], jnp.int32))))


def test_record_radius_pathing_and_flow():
    cfg = world()[0]
    s = MS.init_state(cfg)
    np.testing.assert_allclose(float(s.radius[0]), 65.0)
    np.testing.assert_allclose(np.asarray(cfg.champion_base.pathing_radius)[0], 32.0)
    np.testing.assert_allclose(np.asarray(cfg.champion_base.acquisition_range)[0], 400.0)
    assert float(s.champ.mana[0]) == 0.0


def test_azakana_attack_is_half_magic():
    s = placed((SX, SY), (SX + 170.0, SY))
    kinds = []
    for _ in range(90):
        s, ev = step(s, orders(attack=[1, -1]))
        p = ev.report.packets
        mine = np.asarray(p.valid) & (np.asarray(p.src) == 0) & (np.asarray(p.dst) == 1) \
            & ((np.asarray(p.flags) & D.TAG_BASIC_ATTACK) != 0)
        if mine.any():
            kinds.append(sorted(np.asarray(p.dtype)[mine].tolist()))
        if len(kinds) == 2:
            break
    assert kinds == [[D.PHYSICAL], [D.PHYSICAL, D.MAGIC]]


def test_empowered_mortal_steel_dashes_and_knocks_up():
    s = placed((SX, SY), (SX + 350.0, SY))
    yone = s.kits.yone
    s = s._replace(kits=s.kits._replace(yone=yone._replace(q_stacks=yone.q_stacks.at[0].set(2),
                                                          q_stacks_until=yone.q_stacks_until.at[0].set(80.0))))
    hp = float(s.hp[1])
    s, _ = step(s, orders(cast_slot=[0, -1], cast_x=[SX + 1000.0, 0.0], cast_y=[SY, 0.0]))
    s = run(s, MS.no_orders(), 18)
    assert float(s.cc.knockup_until[1]) > float(s.t) and float(s.hp[1]) < hp
    s = run(s, MS.no_orders(), 6)
    assert 445.0 < float(s.x[0]) - SX <= 465.0                        # 450 (+ the cast tick's idle step)


def test_soul_unbound_out_and_back_detonates():
    s = placed((SX, SY), (SX + 600.0, SY))
    s, _ = step(s, orders(cast_slot=[2, -1], cast_x=[SX + 1000.0, 0.0], cast_y=[SY, 0.0]))
    s = run(s, MS.no_orders(), 12)
    assert 290.0 < float(s.x[0]) - SX <= 305.0
    s = run(s, orders(attack=[1, -1]), 60)                             # attack Garen in Spirit Form
    mark = float(s.kits.yone.e_mark[0, 1])
    assert mark > 0.0
    hp = float(s.hp[1])
    s, _ = step(s, orders(cast_slot=[2, -1], stop=[True, False]))
    s = run(s, orders(stop=[True, False]), 12)                         # 0.225 s recast + ~600 at 2950
    assert abs(float(s.x[0]) - (SX - 300.0)) < 15.0                   # back at the body, 300 behind the cast
    assert hp - float(s.hp[1]) >= mark - 1.0


def test_fate_sealed_blinks_behind_and_pulls():
    s = placed((SX, SY), (SX + 500.0, SY))
    s, _ = step(s, orders(cast_slot=[3, -1], cast_x=[SX + 1000.0, 0.0], cast_y=[SY, 0.0]))
    s = run(s, MS.no_orders(), 23)                                     # 0.75 s cast + the 0.05 s blink delay
    np.testing.assert_allclose(float(s.x[0]) - float(s.x[1]), 200.0, atol=5.0)   # 200 behind Garen (who walked in)
    s = run(s, MS.no_orders(), 12)
    assert abs(float(s.x[1]) - float(s.x[0])) < 60.0                  # Garen pulled to Yone
    assert float(s.cc.knockup_until[1]) > float(s.t)
