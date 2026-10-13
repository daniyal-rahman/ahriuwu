"""Ambessa through the real compiled tick against Garen: energy, Cunning Sweep followed by a Drakehound's Step dash
on a move command, the empowered chained attack, Repudiation's shield and smash, Public Execution's seize and slam
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
    cfg = MW.build_config((MW.Loadout("Ambessa", items=H.ITEMS), MW.Loadout("Garen", items=H.ITEMS)),   # no runes
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


def placed(amb_xy, garen_xy, ranks=(1, 1, 1, 1)):
    s = MS.init_state(world()[0])
    return s._replace(x=s.x.at[0].set(amb_xy[0]).at[1].set(garen_xy[0]),
                      y=s.y.at[0].set(amb_xy[1]).at[1].set(garen_xy[1]), t=jnp.float32(70.0),
                      champ=s.champ._replace(ranks=s.champ.ranks.at[0].set(jnp.asarray(ranks, jnp.int32))
                                             .at[1].set(jnp.asarray([1, 1, 1, 0], jnp.int32))))


def test_record_radius_and_energy():
    cfg = world()[0]
    s = MS.init_state(cfg)
    np.testing.assert_allclose(float(s.radius[0]), 65.0)
    np.testing.assert_allclose(np.asarray(cfg.champion_base.acquisition_range)[0], 600.0)
    assert float(s.champ.mana[0]) == 200.0 and bool(np.asarray(cfg.uses_energy)[0])


def test_sweep_then_drakehounds_step_on_a_move_command():
    s = placed((SX, SY), (SX + 350.0, SY + 30.0))
    hp = float(s.hp[1])
    s, _ = step(s, orders(cast_slot=[0, -1], cast_x=[SX + 1000.0, 0.0], cast_y=[SY, 0.0]))
    s, _ = step(s, orders(move=[True, False], move_x=[SX - 1000.0, 0.0], move_y=[SY, 0.0]))   # a command in the lockout
    s = run(s, MS.no_orders(), 6)
    assert float(s.hp[1]) < hp                                         # the sweep landed
    assert float(s.champ.mana[0]) < 200.0 - 60.0
    x0 = float(s.x[0])
    s = run(s, MS.no_orders(), 12)
    assert SX - float(s.x[0]) > 300.0, (x0, float(s.x[0]))           # dashed ~350 toward the command
    assert int(s.kits.ambessa.stacks[0]) == 1


def test_empowered_chained_attack_is_quick():
    s = placed((SX, SY), (SX + 180.0, SY))
    amb = s.kits.ambessa
    s = s._replace(kits=s.kits._replace(ambessa=amb._replace(stacks=amb.stacks.at[0].set(2),
                                                            stacks_until=amb.stacks_until.at[0].set(80.0))))
    hits = []
    for i in range(45):
        s, ev = step(s, orders(attack=[1, -1]))
        p = ev.report.packets
        mine = np.asarray(p.valid) & (np.asarray(p.src) == 0) & (np.asarray(p.dst) == 1) \
            & ((np.asarray(p.flags) & D.TAG_BASIC_ATTACK) != 0)
        if mine.any():
            hits.append(i)
    assert len(hits) >= 2 and hits[1] - hits[0] <= 12                  # 0.35 s chained attack time (~11 ticks)


def test_repudiation_shield_and_smash():
    s = placed((SX, SY), (SX + 250.0, SY))
    hp = float(s.hp[1])
    s, _ = step(s, orders(cast_slot=[1, -1]))
    assert float(D.total_shield(s.shields, s.t)[0]) > 50.0
    s = run(s, MS.no_orders(), 18)
    assert float(s.hp[1]) < hp


def test_public_execution_seizes_and_slams():
    s = placed((SX, SY), (SX + 900.0, SY))
    s, _ = step(s, orders(cast_slot=[3, -1], cast_x=[SX + 1500.0, 0.0], cast_y=[SY, 0.0]))
    s = run(s, MS.no_orders(), 22)                                     # 0.7 s cast
    assert float(s.x[0]) > float(s.x[1])                               # blinked behind Garen
    assert float(s.cc.stun_until[1]) > float(s.t)
    hp = float(s.hp[1])
    s = run(s, MS.no_orders(), 26)                                     # 0.75 s suppression, then the slam
    assert float(s.hp[1]) < hp - 100.0
