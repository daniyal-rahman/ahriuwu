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


def arrival_state(launching):
    from lanerl_jax.modern.core import damage as D
    from lanerl_jax.modern.core import types as W
    from lanerl_jax.modern.items.catalog import catalog

    s = placed((SX, SY), (SX + 540.0, SY))
    target = world()[0].layout.minion0
    s = s._replace(
        kind=s.kind.at[target].set(W.KIND_MINION), team=s.team.at[target].set(1),
        alive=s.alive.at[target].set(True), targetable=s.targetable.at[target].set(True),
        visible=s.visible.at[:, target].set(True),
        x=s.x.at[target].set(SX + 300.0), y=s.y.at[target].set(SY),
        hp=s.hp.at[target].set(1000.0), max_hp=s.max_hp.at[target].set(1000.0),
        radius=s.radius.at[target].set(48.0),
        att=s.att._replace(target=s.att.target.at[0].set(target),
                           target_seq=s.att.target_seq.at[0].set(s.spawn_seq[target]),
                           windup_left=s.att.windup_left.at[0].set(DT / 2 if launching else 0.0),
                           cooldown_left=s.att.cooldown_left.at[0].set(1.0)))
    inv = s.champ.inventory
    s = s._replace(champ=s.champ._replace(inventory=inv._replace(
        item=inv.item.at[0, 0].set(catalog().row(6698)))))               # Profane Hydra
    kit = s.kits.ezreal
    kit = kit._replace(mark_on=kit.mark_on.at[0].set(True), mark_unit=kit.mark_unit.at[0].set(1),
                       mark_seq=kit.mark_seq.at[0].set(s.spawn_seq[1]),
                       mark_until=kit.mark_until.at[0].set(74.0), mark_cast_id=kit.mark_cast_id.at[0].set(123))
    ms = s.missiles
    ms = ms._replace(alive=ms.alive.at[0].set(True), src=ms.src.at[0].set(0), dst=ms.dst.at[0].set(1),
                     dst_seq=ms.dst_seq.at[0].set(s.spawn_seq[1]), x=ms.x.at[0].set(s.x[1]),
                     y=ms.y.at[0].set(s.y[1]), speed=ms.speed.at[0].set(Z.ATTACK_MISSILE),
                     raw=ms.raw.at[0].set(100.0), flags=ms.flags.at[0].set(D.BASIC_ATTACK))
    s = s._replace(kits=s.kits._replace(ezreal=kit), missiles=ms)
    return s, target


@pytest.mark.parametrize("launching", [False, True])
def test_arriving_attack_detonates_mark_while_launching_at_another_unit(launching):
    s, target = arrival_state(launching)
    s, ev = step(s, orders(attack=[target, -1]))
    assert bool(ev.launched[0]) == launching
    assert int(s.att.target[0]) == target
    assert not bool(s.kits.ezreal.mark_on[0])
    p = ev.report.packets
    det = np.asarray(p.valid & (p.cast_id == 123))
    assert det.sum() == 1
    assert int(np.asarray(p.dst)[det][0]) == 1
    assert bool(jnp.any(p.valid & (p.item == 6698) & (p.dst == target)))  # nonzero landed raw enables Cleave


@pytest.mark.parametrize("collision", ["basics_different", "basics_same", "q_basic"])
@pytest.mark.parametrize("launching", [False, True])
def test_every_colliding_landing_runs_on_hits(collision, launching):
    from lanerl_jax.modern.core import damage as D
    from lanerl_jax.modern.items.catalog import catalog
    from lanerl_jax.modern.items.effects import marksman as M

    s, target = arrival_state(launching)
    inv = s.champ.inventory
    s = s._replace(champ=s.champ._replace(inventory=inv._replace(
        item=inv.item.at[0, 0].set(catalog().row(M.RECURVE)).at[0, 1].set(catalog().row(M.YUNTAL))
                     .at[0, 2].set(catalog().row(6698)), stack=inv.stack.at[0, 2].set(1))))
    marksman = s.combat.items.marksman._replace(yt_cd_until=s.combat.items.marksman.yt_cd_until.at[0].set(90.0))
    s = s._replace(combat=s.combat._replace(items=s.combat.items._replace(marksman=marksman)))
    second = 1 if collision == "basics_same" else target
    kit = s.kits.ezreal._replace(mark_unit=s.kits.ezreal.mark_unit.at[0].set(second),
                                mark_seq=s.kits.ezreal.mark_seq.at[0].set(s.spawn_seq[second]))
    ms = s.missiles._replace(raw=s.missiles.raw.at[0].set(110.0), crit=s.missiles.crit.at[0].set(True),
                             flags=s.missiles.flags.at[0].set(D.BASIC_ATTACK | D.PROP_CRIT),
                             cast_id=s.missiles.cast_id.at[0].set(101))
    if collision == "q_basic":
        kit = kit._replace(q_fly=kit.q_fly.at[0].set(True), q_ox=kit.q_ox.at[0].set(s.x[target] - 60.0),
                           q_oy=kit.q_oy.at[0].set(s.y[target]), q_mdx=kit.q_mdx.at[0].set(1.0),
                           q_mdy=kit.q_mdy.at[0].set(0.0), q_dist=kit.q_dist.at[0].set(0.0),
                           q_mid=kit.q_mid.at[0].set(333))
    else:
        ms = ms._replace(alive=ms.alive.at[1].set(True), src=ms.src.at[1].set(0), dst=ms.dst.at[1].set(second),
                         dst_seq=ms.dst_seq.at[1].set(s.spawn_seq[second]), x=ms.x.at[1].set(s.x[second]),
                         y=ms.y.at[1].set(s.y[second]), speed=ms.speed.at[1].set(Z.ATTACK_MISSILE),
                         raw=ms.raw.at[1].set(70.0), flags=ms.flags.at[1].set(D.BASIC_ATTACK),
                         cast_id=ms.cast_id.at[1].set(102))
    s, ev = step(s._replace(kits=s.kits._replace(ezreal=kit), missiles=ms), orders(attack=[target, -1]))
    assert bool(ev.launched[0]) == launching
    assert int(s.att.target[0]) == target
    p = ev.report.packets
    primary = p.valid & (p.src == 0) & ((p.cast_id == 101) | (p.cast_id == (333 if collision == "q_basic" else 102)))
    assert int(jnp.sum(primary & (p.raw > 0))) == 2
    assert bool(jnp.any(primary & (p.cast_id == 101) & (p.dst == 1) & (p.raw == 110.0)))
    onhit = p.valid & (p.item == M.RECURVE) & (p.src == 0)
    np.testing.assert_array_equal(np.sort(np.asarray(p.dst)[np.asarray(onhit)]), np.sort([1, second]))
    assert int(jnp.sum(p.valid & (p.item == 6698))) == 2  # both landed raw values enable Profane Cleave
    assert not bool(s.kits.ezreal.mark_on[0])
    assert int(jnp.sum(p.valid & (p.cast_id == 123) & (p.dst == second))) == 1
    np.testing.assert_allclose(s.combat.items.marksman.yt_cd_until[0], 90.0 - M.YT_CRIT_CDR - M.YT_AA_CDR)
    launches = int(launching) + int(collision == "q_basic")
    np.testing.assert_allclose(s.combat.items.marksman.yt_crit[0], launches * M.YT_CRIT_PER * M.YT_RANGED)
    assert int(ev.packet_overflow) == int(ev.missile_overflow) == 0


def test_hit_batch_padding_cannot_launch_another_ability():
    from lanerl_jax.modern.items.effects.core import Attack
    from lanerl_jax.modern.world.phases.attack import _pack_hits

    hits = Attack(jnp.asarray([[False, False], [True, False], [False, False]]),
                  jnp.asarray([[False, True], [True, False], [False, True]]),
                  jnp.asarray([[-1, 0], [2, -1], [-1, 3]]), jnp.asarray([[0., 80.], [90., 0.], [0., 100.]]),
                  jnp.asarray([[False, False], [True, False], [False, True]]),
                  cast_id=jnp.asarray([[0, 11], [12, 0], [0, 13]]))
    packed, _ = jax.jit(_pack_hits)(hits)
    np.testing.assert_array_equal(packed.hit[:2], [[True, True], [False, True]])
    np.testing.assert_array_equal(packed.launched, [[True, False], [False, False], [False, False]])
    np.testing.assert_array_equal(packed.target[0], [2, 0])
    np.testing.assert_array_equal(packed.raw[0], [90., 80.])
    np.testing.assert_array_equal(packed.is_crit[0], [True, False])
    np.testing.assert_array_equal(packed.cast_id[0], [12, 11])
