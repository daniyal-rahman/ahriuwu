"""Symptom tests for the remaining 26.19 item actives (actives)."""
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.items import effects as E
from lanerl_jax.modern.items.effects import actives as A
from lanerl_jax.modern.tests import item_harness as H


def _world(extra=(), **kw):
    rows = H.champions(**kw) + list(extra)
    return H.units(rows)


def _press(item, *, now=10.0, state=None, own_items=None, ctx_kw=None, units=None, other=0):
    own = H.own(own_items or [item], [])
    u = units if units is not None else _world()
    c = H.ctx(now=now, **(ctx_kw or {}))
    st = A.init(2, u.x.shape[0]) if state is None else state
    req = jnp.asarray([item, other], jnp.int32)
    return A.active(st, own, c, u, req), own, c, u


def test_registered_and_coverage_points_to_module():
    assert "actives" in E.ItemEffectState._fields
    cov = E.coverage_report()
    for iid in A.ACTIVE_ITEMS:
        assert "DEFERRED" not in cov[iid], (iid, cov[iid])


def test_zhonyas_stasis_blocks_other_actives_and_sets_cooldown():
    (st, eff, out), own, c, u = _press(A.ZHONYAS, own_items=[A.ZHONYAS, A.RANDUINS])
    assert bool(out.used[0])
    w = A.world(st, jnp.float32(11.0))
    assert bool(w.stasis[0]) and not bool(w.stasis[1])
    assert float(w.stasis_until[0]) == pytest.approx(12.5)
    assert not bool(A.world(st, jnp.float32(12.6)).stasis[0])
    # During stasis Randuin's cannot be used.
    st2, _, out2 = A.active(st, own, H.ctx(now=11.0), u, jnp.asarray([A.RANDUINS, 0], jnp.int32))
    assert not bool(out2.used[0])
    # Zhonya's on cooldown for 120 s.
    _, _, out3 = A.active(st, own, H.ctx(now=100.0), u, jnp.asarray([A.ZHONYAS, 0], jnp.int32))
    assert not bool(out3.used[0])
    _, _, out4 = A.active(st, own, H.ctx(now=130.1), u, jnp.asarray([A.ZHONYAS, 0], jnp.int32))
    assert bool(out4.used[0])


def test_seekers_is_single_use_and_requests_the_shattered_transform():
    (st, _, out), own, c, u = _press(A.SEEKERS)
    assert bool(out.used[0])
    frm, to, do = A.world(st, jnp.float32(10.0)).transform
    assert bool(do[0]) and not bool(do[1])
    from lanerl_jax.modern.items.catalog import catalog
    assert int(frm[0]) == catalog().row(A.SEEKERS) and int(to[0]) == catalog().row(A.SHATTERED)
    _, _, again = A.active(st, own, H.ctx(now=10000.0), u, jnp.asarray([A.SEEKERS, 0], jnp.int32))
    assert not bool(again.used[0])


def test_quicksilver_pulses_cleanse_and_mercurial_adds_ms():
    (st, _, out), own, c, u = _press(A.QUICKSILVER)
    assert bool(A.world(st, c.now).cleanse[0])
    (st, _, _), own, c, u = _press(A.MERCURIAL)
    assert bool(A.world(st, c.now).cleanse[0])
    assert float(A.stats(st, own, H.ctx(now=11.0)).percent_move_speed[0]) == pytest.approx(0.5)
    assert float(A.stats(st, own, H.ctx(now=12.1)).percent_move_speed[0]) == pytest.approx(0.0)


def test_youmuu_ms_and_ghost_melee_vs_ranged():
    (st, _, _), own, _, _ = _press(A.YOUMUU)
    assert float(A.stats(st, own, H.ctx(now=15.9)).percent_move_speed[0]) == pytest.approx(0.20)
    assert bool(A.status(st, own, H.ctx(now=15.9)).ghosted[0])
    assert not bool(A.status(st, own, H.ctx(now=16.1)).ghosted[0])
    (st, _, _), own, _, _ = _press(A.YOUMUU, ctx_kw=dict(ranged=True))
    assert float(A.stats(st, own, H.ctx(now=13.9)).percent_move_speed[0]) == pytest.approx(0.15)
    assert not bool(A.status(st, own, H.ctx(now=14.1)).ghosted[0])


def test_randuins_slows_enemies_within_500_edge():
    near = dict(x=400.0, y=0.0, team=1, radius=48.0)      # edge 400-48 < 500
    far = dict(x=600.0, y=0.0, team=1, radius=48.0)       # edge 552 > 500
    ally = dict(x=100.0, y=0.0, team=0)
    u = _world([near, far, ally], x1=2000.0)
    (st, eff, out), *_ = _press(A.RANDUINS, units=u)
    slow = np.asarray(eff.slow)
    assert slow[2] == pytest.approx(0.7) and slow[3] == 0.0 and slow[4] == 0.0
    assert float(eff.slow_duration[2]) == pytest.approx(2.0)


def test_gunblade_needs_an_enemy_champion_in_range():
    (st, eff, out), *_ = _press(A.GUNBLADE, units=_world(x1=1200.0))
    assert not bool(out.used[0])
    assert float(st.cd_until[0, A._K[A.GUNBLADE]]) == 0.0
    (st, eff, out), *_ = _press(A.GUNBLADE, units=_world(x1=600.0), ctx_kw=dict(level=18, ap=100.0))
    assert bool(out.used[0])
    assert H.packet_total(eff.packets, dst=1, item=A.GUNBLADE) == pytest.approx(253.0 + 30.0)
    assert float(eff.slow[1]) == pytest.approx(0.25)


def test_rocketbelt_dashes_toward_aim_hits_arc_and_resets_attack():
    minion = dict(x=900.0, y=0.0, team=1)
    behind = dict(x=-600.0, y=0.0, team=1)
    u = _world([minion, behind], x1=5000.0)
    st0 = A.with_aim(A.init(2, u.x.shape[0]), jnp.asarray([-1, -1]), jnp.asarray([1000.0, 0.0]), jnp.asarray([0.0, 0.0]))
    (st, eff, out), *_ = _press(A.ROCKETBELT, state=st0, units=u, ctx_kw=dict(ap=50.0))
    assert bool(out.attack_reset[0])
    w = A.world(st, jnp.float32(10.0))
    assert bool(w.dash.active[0]) and float(w.dash.to_x[0]) == pytest.approx(275.0) and not bool(w.dash.blink[0])
    assert H.packet_targets(eff.packets, item=A.ROCKETBELT) == [2]
    assert H.packet_total(eff.packets, dst=2, item=A.ROCKETBELT) == pytest.approx(105.0)


def test_locket_and_shurelya_reach_the_holder():
    (st, eff, out), own, _, _ = _press(A.LOCKET, ctx_kw=dict(level=10))
    assert float(eff.shields.amount[0, 0]) == pytest.approx(290.0 + 2 * 7.0)
    assert float(eff.shields.duration[0, 0]) == pytest.approx(2.5)
    (st, _, _), own, _, _ = _press(A.SHURELYA)
    assert float(A.stats(st, own, H.ctx(now=13.9)).percent_move_speed[0]) == pytest.approx(0.3)


def test_redemption_lands_after_2_5_s_even_if_cast_while_dead():
    u = _world(x1=300.0)
    st0 = A.with_aim(A.init(2, 2), jnp.asarray([-1, -1]), jnp.asarray([150.0, 0.0]), jnp.asarray([0.0, 0.0]))
    (st, eff, out), own, _, _ = _press(A.REDEMPTION, state=st0, units=u, ctx_kw=dict(alive=False))
    assert bool(out.used[0]) and H.packet_total(eff.packets) == 0.0
    st1, eff1, _ = A.active(st, own, H.ctx(now=11.0), u, jnp.asarray([0, 0], jnp.int32))
    assert H.packet_total(eff1.packets) == 0.0
    st2, eff2, _ = A.active(st1, own, H.ctx(now=12.5, level=18), u, jnp.asarray([0, 0], jnp.int32))
    assert H.packet_total(eff2.packets, dst=1, item=A.REDEMPTION) == pytest.approx(100.0)   # 10% of 1000
    assert float(eff2.heal[0]) == pytest.approx(350.0)
    st3, eff3, _ = A.active(st2, own, H.ctx(now=12.6), u, jnp.asarray([0, 0], jnp.int32))
    assert H.packet_total(eff3.packets) == 0.0


def test_actualizer_amplifies_ability_damage_and_doubles_mana_costs():
    (st, _, _), own, _, u = _press(A.ACTUALIZER, ctx_kw=dict(max_mana=1000.0))
    c = H.ctx(now=12.0, max_mana=1000.0)
    p = D.concat_packets(D.packets(True, 0, 1, 100.0, D.MAGIC, D.TAG_ACTIVE_SPELL),
                         D.packets(True, 0, 1, 100.0, D.MAGIC, D.TAG_ACTIVE_SPELL | D.TAG_ITEM),
                         D.packets(True, 1, 0, 100.0, D.MAGIC, D.TAG_ACTIVE_SPELL))
    amp = np.asarray(A.packet_amp(st, own, c, u, p))
    assert amp[0] == pytest.approx(0.20) and amp[1] == 0.0 and amp[2] == 0.0
    w = A.world(st, jnp.float32(12.0))
    assert float(w.mana_cost_mult[0]) == 2.0 and float(w.basic_cd_rate[0]) == pytest.approx(1.3)
    assert float(A.world(st, jnp.float32(18.1)).mana_cost_mult[0]) == 1.0


def test_ally_only_actives_are_inert():
    for iid in A.INERT_ALLY_ONLY:
        (st, eff, out), *_ = _press(iid)
        assert not bool(out.used[0])


def test_request_gating_under_cc():
    req = jnp.asarray([A.ZHONYAS, A.QUICKSILVER], jnp.int32)
    out = A.request_allowed(req, disabled=jnp.asarray([True, True]))
    assert out.tolist() == [0, A.QUICKSILVER]
    assert A.request_allowed(req, disabled=jnp.asarray([False, False]), in_stasis=jnp.asarray([True, False])).tolist() \
        == [0, A.QUICKSILVER]
