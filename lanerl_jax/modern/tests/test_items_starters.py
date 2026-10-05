"""Starters, Glory, Cull, Manaflow line and consumables (items.effects.starters/consumables)."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lanerl_jax.modern.items.effects as E
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.items import inventory as inv_mod
from lanerl_jax.modern.items.catalog import catalog
from lanerl_jax.modern.items.effects import consumables as CO
from lanerl_jax.modern.items.effects import starters as S
from lanerl_jax.modern.tests import item_harness as H


def world(extra=()):
    return H.units(H.champions(x1=300.0) + [dict(x=500, y=0, team=1, cls=D.CLASS_MINION)] + list(extra))


def pk(src, dst, raw, dtype=D.PHYSICAL, flags=0, item=0):
    return D.packets(jnp.ones((len(src),), bool), jnp.asarray(src), jnp.asarray(dst), jnp.asarray(raw, jnp.float32),
                     jnp.asarray(dtype), jnp.asarray(flags), item=jnp.asarray(item))


def test_coverage_registered():
    starters = {1054, 1056, 1082, 3041, 1083, 1086, 1120, 3070, 3003, 3040, 3004, 3042, 3119, 3121, 2526, 2530}
    assert set(S.COVERAGE) == starters and set(CO.COVERAGE) == {2003, 2031, 2138, 2139, 2140, 2010, 2150, 2151, 2152}
    for iid in list(S.COVERAGE) + list(CO.COVERAGE):
        assert iid in catalog() and iid not in E.STATS_ONLY and iid not in E.DEFERRED


# ---- Helping Hand -------------------------------------------------------------

def test_helping_hand_minion_only_once():
    u = world()
    own = H.own([1054, 3070], [])
    st = S.init(2, 3)
    _, eff = S.on_hit(st, own, H.ctx(), u, H.attack(target=(2, 0)))
    assert H.packet_total(eff.packets, dst=2) == pytest.approx(5.0)
    p = eff.packets
    sel = np.asarray(p.valid)
    assert not np.any(np.asarray(p.flags)[sel] & D.PROP_LIFESTEAL)
    _, eff = S.on_hit(st, own, H.ctx(), u, H.attack(target=(1, 0)))     # champion: none
    assert H.packet_total(eff.packets) == 0.0
    _, eff = S.on_hit(st, H.own([1055], []), H.ctx(), u, H.attack(target=(2, 0)))
    assert H.packet_total(eff.packets) == 0.0


# ---- Doran's Shield (F11) -----------------------------------------------------

def _shield_trigger(flags):
    u = world()
    own = H.own([1054], [])
    st = S.init(2, 3)
    p = pk([1], [0], [100.0], flags=flags)
    rep, _ = H.resolve(p, u)
    ctx = H.ctx(max_hp=1000.0, hp=400.0, base_hp=600.0)
    st, _ = S.on_damage(st, own, ctx, u, rep)
    return st, own, ctx


def test_dorans_shield_f11():
    st, own, ctx = _shield_trigger(D.BASIC_ATTACK)
    assert float(S.stats(st, own, ctx).health_regen[0]) == pytest.approx(4.0, rel=1e-5)
    assert float(S.stats(st, own, ctx._replace(now=jnp.float32(8.1))).health_regen[0]) == 0.0
    st, own, ctx = _shield_trigger(D.TAG_AOE)
    assert float(S.stats(st, own, ctx).health_regen[0]) == pytest.approx(2.64, rel=1e-5)
    # Minion damage does not trigger.
    u = world()
    rep, _ = H.resolve(pk([2], [0], [100.0]), u)
    st2, _ = S.on_damage(S.init(2, 3), own, ctx, u, rep)
    assert float(S.stats(st2, own, ctx).health_regen[0]) == 0.0


# ---- Doran's Ring ---------------------------------------------------------------

def test_dorans_ring_drain():
    u = world()
    own = H.own([1056], [1056])
    st = S.init(2, 3)
    ctx = H.ctx(max_mana=jnp.asarray([300.0, 0.0]))
    s = S.stats(st, own, ctx)
    assert float(s.mana_regen[0]) == 1.0 and float(s.health_regen[1]) == pytest.approx(0.45)
    rep, _ = H.resolve(pk([0], [1], [50.0], D.MAGIC), u)
    st, _ = S.on_damage(st, own, ctx, u, rep)
    assert float(S.stats(st, own, ctx).mana_regen[0]) == 2.0
    assert float(S.stats(st, own, ctx._replace(now=jnp.float32(5.1))).mana_regen[0]) == 1.0


# ---- Glory ------------------------------------------------------------------------

def test_glory_dark_seal_to_mejais():
    u = world()
    st = S.init(2, 3)
    seal = H.own([1082], [])
    st, _ = S.on_takedown(st, seal, H.ctx(), u, H.kills(3, champion_kill=(4, 0), champion_assist=(3, 0)))
    assert float(st.glory[0]) == 10.0          # 8 + 3 capped at 10
    assert float(S.stats(st, seal, H.ctx()).ability_power[0]) == 40.0
    mej = H.own([3041], [])
    s = S.stats(st, mej, H.ctx())
    assert float(s.ability_power[0]) == 50.0 and float(s.percent_move_speed[0]) == pytest.approx(0.1)
    st, _ = S.on_takedown(st, mej, H.ctx(), u, H.kills(3, champion_kill=(5, 0)))
    assert float(st.glory[0]) == 25.0
    st, _ = S.on_takedown(st, mej, H.ctx(), u, H.kills(3, died=(True, False)))
    assert float(st.glory[0]) == 15.0
    assert float(st.glory[1]) == 0.0


# ---- Cull --------------------------------------------------------------------------

def test_cull_gold_and_heal():
    u = world()
    own = H.own([1083], [])
    st = S.init(2, 3)
    _, eff = S.on_hit(st, own, H.ctx(), u, H.attack(target=(2, 0)))
    assert float(eff.heal[0]) == 3.0 and float(eff.heal[1]) == 0.0
    st, eff = S.on_takedown(st, own, H.ctx(), u, H.kills(3, minion_kill=(99, 5)))
    assert float(eff.gold[0]) == 99.0 and float(eff.gold[1]) == 0.0
    st, eff = S.on_takedown(st, own, H.ctx(), u, H.kills(3, minion_kill=(3, 0)))
    assert float(eff.gold[0]) == 1.0 + 350.0
    st, eff = S.on_takedown(st, own, H.ctx(), u, H.kills(3, minion_kill=(3, 0)))
    assert float(eff.gold[0]) == 0.0


# ---- Manaflow and transforms ---------------------------------------------------------

def test_manaflow_charges_and_ability_trigger():
    u = world()
    own = H.own([3070], [])
    st = S.init(2, 3)
    for t in np.arange(0.0, 33.0, 0.5):
        st, _ = S.periodic(st, own, H.ctx(now=float(t)), u, )
    assert float(st.charges[0]) == 4.0 and float(st.charges[1]) == 0.0
    ctx = H.ctx(now=33.0)
    st, _ = S.on_cast(st, own, ctx, u, H.cast())
    rep, _ = H.resolve(pk([0, 0], [1, 2], [50.0, 50.0], D.MAGIC, D.TAG_ACTIVE_SPELL), u)
    st, _ = S.on_damage(st, own, ctx, u, rep)
    assert float(st.tear_mana[0]) == 6.0 and float(st.charges[0]) == 3.0
    st, _ = S.on_damage(st, own, ctx, u, rep)            # same cast instance: no second charge
    assert float(st.tear_mana[0]) == 6.0
    assert float(S.stats(st, own, ctx).mana[0]) == 6.0
    # Tear does not charge on attacks; Manamune does (x1 vs minion).
    st2, _ = S.on_hit(st, own, ctx, u, H.attack(target=(2, 0)))
    assert float(st2.tear_mana[0]) == 6.0
    mm = H.own([3004], [])
    st3, _ = S.on_hit(st, mm, ctx, u, H.attack(target=(2, 0)))
    assert float(st3.tear_mana[0]) == 9.0


def test_transform_and_awe():
    st = S.init(2, 3)._replace(tear_mana=jnp.asarray([360.0, 100.0]))
    own = H.own([3004], [3003])
    frm, to, do = S.pending_transforms(st, own)
    assert bool(do[0]) and not bool(do[1])
    cat = catalog()
    assert int(frm[0]) == cat.row(3004) and int(to[0]) == cat.row(3042)
    inv = inv_mod.inventory_from_ids([[3004], [3003]])
    new = jax.vmap(inv_mod.replace_item)(inv, frm, to, do)
    assert inv_mod.owns(new, 3042)[0] and inv_mod.owns(new, 3003)[1]
    # Manamune AD = 2% of max mana (ctx max mana + stacks).
    ctx = H.ctx(max_mana=800.0)
    s = S.stats(st, own, ctx)
    assert float(s.attack_damage[0]) == pytest.approx(0.02 * 1160.0, rel=1e-5)
    # Archangel's AP = 1% bonus mana (600 item + 100 stacks).
    assert float(s.ability_power[1]) == pytest.approx(7.0, rel=1e-5)
    # After transform the stacks stop counting.
    s2 = S.stats(st, H.own([3042], [3040]), H.ctx(max_mana=1300.0))
    assert float(s2.attack_damage[0]) == pytest.approx(26.0, rel=1e-5)
    assert float(s2.ability_power[1]) == pytest.approx(20.0, rel=1e-5)
    st2, _ = S.periodic(st, H.own([3042], [3040]), H.ctx(), H.units(H.champions()))
    assert float(st2.tear_mana[0]) == 0.0


def test_winters_and_circlet_awe():
    st = S.init(2, 3)._replace(tear_mana=jnp.asarray([100.0, 100.0]))
    s = S.stats(st, H.own([3119], [2526]), H.ctx())
    assert float(s.health[0]) == pytest.approx(0.15 * 600.0, rel=1e-5)
    assert float(s.heal_shield_power[1]) == pytest.approx(0.005 * 400.0 / 100.0, rel=1e-5)


def test_muramana_shock():
    u = world()
    own = H.own([3042], [])
    st = S.init(2, 3)
    ctx = H.ctx(max_mana=1500.0)
    _, eff = S.on_hit(st, own, ctx, u, H.attack(target=(1, 0)))
    assert H.packet_total(eff.packets, item=3042) == pytest.approx(18.0, rel=1e-5)
    _, eff = S.on_hit(st, own, ctx, u, H.attack(target=(2, 0)))
    assert H.packet_total(eff.packets, item=3042) == 0.0
    st, _ = S.on_cast(st, own, ctx, u, H.cast())
    rep, _ = H.resolve(pk([0, 0], [1, 2], [50.0, 50.0], D.MAGIC, D.TAG_ACTIVE_SPELL), u)
    st, eff = S.on_damage(st, own, ctx, u, rep)
    assert H.packet_total(eff.packets, item=3042, dst=1) == pytest.approx(60.0, rel=1e-5)
    assert H.packet_targets(eff.packets, item=3042) == [1]
    _, eff = S.on_damage(st, own, ctx, u, rep)
    assert H.packet_total(eff.packets, item=3042) == 0.0
    st_r, eff = S.on_damage(S.on_cast(S.init(2, 3), own, ctx, u, H.cast())[0], own,
                            ctx._replace(is_ranged=jnp.asarray([True, True])), u, rep)
    assert H.packet_total(eff.packets, item=3042) == pytest.approx(45.0, rel=1e-5)


def test_seraphs_lifeline():
    u = H.units(H.champions())
    own = H.own([3040], [])
    st = S.init(2, 2)
    ctx = H.ctx(max_mana=1000.0, base_hp=1000.0, hp=400.0)
    dfn = S.defense(st, own, ctx)
    assert bool(dfn.lifeline_ready[0]) and not bool(dfn.lifeline_ready[1])
    assert float(dfn.lifeline_shield[0]) == pytest.approx(180.0, rel=1e-5)
    dd = D.default_defense(2)._replace(unit_class=u.cls, lifeline_ready=dfn.lifeline_ready,
                                       lifeline_shield=dfn.lifeline_shield, lifeline_duration=dfn.lifeline_duration,
                                       lifeline_decay_hold=dfn.lifeline_decay_hold)
    rep, res = H.resolve(pk([1], [0], [200.0]), u._replace(hp=jnp.asarray([400.0, 1000.0])), defense=dd)
    assert bool(res.lifeline_fired[0])
    st, _ = S.on_damage(st, own, ctx, u, rep)
    assert float(st.seraph_cd[0]) == pytest.approx(90.0)
    assert not bool(S.defense(st, own, ctx).lifeline_ready[0])


def test_everlasting_helper_and_diadem():
    u = H.units(H.champions(x1=300.0) + [dict(x=400, y=0, team=1, cls=D.CLASS_CHAMPION)])
    own = H.own([3121], [])
    st = S.init(2, 3)
    st, eff = S.everlasting(st, own, H.ctx(mana=1000.0), u, jnp.asarray([True, False]), jnp.zeros(2, bool))
    assert float(eff.shields.amount[0, 0]) == pytest.approx((100 + 45.0) * 1.8, rel=1e-5)
    _, eff = S.everlasting(st, own, H.ctx(mana=1000.0), u, jnp.asarray([True, False]), jnp.zeros(2, bool))
    assert float(eff.shields.amount[0, 0]) == 0.0
    own = H.own([2530], [])
    st, eff = S.periodic(S.init(2, 3), own, H.ctx(max_mana=1000.0, in_combat=True), u)
    assert float(eff.heal[0]) == pytest.approx(8.0, rel=1e-5)
    _, eff = S.periodic(st, own, H.ctx(now=0.5, max_mana=1000.0, in_combat=True), u)
    assert float(eff.heal[0]) == 0.0


def test_starters_jit():
    u = world()
    own = H.own([1054, 3004], [1083])
    st = S.init(2, 3)
    f = jax.jit(S.on_hit)
    st2, eff = f(st, own, H.ctx(), u, H.attack(target=(2, 2), hit=(True, True)))
    assert H.packet_total(eff.packets, dst=2) == pytest.approx(5.0)
    jax.jit(S.stats)(st2, own, H.ctx())
    jax.jit(S.periodic)(st2, own, H.ctx(), u)


# ---- Consumables ---------------------------------------------------------------------------

def _run_hots(st, own, t_end, dt=1 / 30, gw=False):
    u = H.units(H.champions())
    total = 0.0
    t = 0.0
    while t < t_end - 1e-9:
        t += dt
        st, eff = CO.periodic(st, own, H.ctx(now=t), u)
        total += float(eff.heal_plain[0])
    return st, total


def test_health_potion_f18_and_consume():
    u = H.units(H.champions())
    own = H.own([2003, 2003], [])
    st = CO.init(2, 2)
    st, eff, out = CO.active(st, own, H.ctx(), u, jnp.asarray([2003, 2003]))
    assert bool(out.used[0]) and not bool(out.used[1])
    assert int(st.consume_row[0]) == catalog().row(2003) and int(st.consume_row[1]) == -1
    st1, eff = CO.periodic(st, own, H.ctx(now=0.5), u)
    assert float(eff.heal_plain[0]) == pytest.approx(4.0)
    assert float(D.heal_amount(eff.heal_plain[0], grievous=True)) == pytest.approx(2.4)
    _, total = _run_hots(st, own, 16.0)
    assert total == pytest.approx(120.0, rel=1e-5)
    # dt-agnostic: one big step.
    _, eff = CO.periodic(st, own, H.ctx(now=20.0), u)
    assert float(eff.heal_plain[0]) == pytest.approx(120.0)
    inv = inv_mod.inventory_from_ids([[2003, 2003], []])
    slot = jnp.argmax(inv.item[0] == st.consume_row[0])
    after = inv_mod.consume_one(jax.tree_util.tree_map(lambda a: a[0], inv), slot)
    assert int(after.stack[slot]) == 1


def test_potions_stack_independently_and_cooldown():
    u = H.units(H.champions())
    own = H.own([2003, 2003], [])
    st = CO.init(2, 2)
    st, _, out = CO.active(st, own, H.ctx(), u, jnp.asarray([2003, 0]))
    st, _, out = CO.active(st, own, H.ctx(now=0.5), u, jnp.asarray([2003, 0]))
    assert not bool(out.used[0])                     # 1 s cooldown
    st, _, out = CO.active(st, own, H.ctx(now=1.0), u, jnp.asarray([2003, 0]))
    assert bool(out.used[0])
    _, eff = CO.periodic(st, own, H.ctx(now=30.0), u)
    assert float(eff.heal_plain[0]) == pytest.approx(240.0)


def test_refillable_charges_and_refill():
    u = H.units(H.champions())
    own = H.own([2031], [])
    st = CO.init(2, 2)
    for t in (0.0, 2.0, 4.0):
        st, _, out = CO.active(st, own, H.ctx(now=t), u, jnp.asarray([2031, 0]))
        assert int(st.consume_row[0]) == -1
        assert bool(out.used[0]) == (t < 3.0)
    assert float(st.refill_charges[0]) == 0.0
    _, eff = CO.periodic(st, own, H.ctx(now=30.0), u)
    assert float(eff.heal_plain[0]) == pytest.approx(200.0, rel=1e-5)
    st = CO.on_shop(st, own, H.ctx(in_shop=True))
    assert float(st.refill_charges[0]) == 2.0
    _, eff = CO.periodic(CO.init(2, 2), own, H.ctx(now=0.0), u)
    st0, _, _ = CO.active(CO.init(2, 2), own, H.ctx(), u, jnp.asarray([2031, 0]))
    _, eff = CO.periodic(st0, own, H.ctx(now=0.5), u)
    assert float(eff.heal_plain[0]) == pytest.approx(100.0 / 24.0, rel=1e-5)


def test_elixirs_replace_and_expire():
    u = H.units(H.champions())
    own = H.own([2138, 2140], [])
    st = CO.init(2, 2)
    st, _, out = CO.active(st, own, H.ctx(alive=False), u, jnp.asarray([2138, 0]))
    assert bool(out.used[0]) and int(st.consume_row[0]) == catalog().row(2138)
    s = CO.stats(st, own, H.ctx())
    assert float(s.health[0]) == 300.0 and float(s.tenacity[0]) == pytest.approx(0.25)
    assert float(s.health[1]) == 0.0
    st, _, _ = CO.active(st, own, H.ctx(now=10.0), u, jnp.asarray([2140, 0]))
    s = CO.stats(st, own, H.ctx(now=10.0))
    assert float(s.health[0]) == 0.0 and float(s.attack_damage[0]) == 30.0
    assert float(CO.stats(st, own, H.ctx(now=190.1)).attack_damage[0]) == 0.0
    st_s, _, _ = CO.active(CO.init(2, 2), H.own([2139], []), H.ctx(), u, jnp.asarray([2139, 0]))
    s = CO.stats(st_s, own, H.ctx())
    assert float(s.ability_power[0]) == 50.0 and float(s.mana_regen[0]) == 3.0


def test_wrath_drain_and_sorcery_proc():
    u = H.units(H.champions(x1=300.0) + [dict(x=500, y=0, team=1, cls=D.CLASS_STRUCTURE),
                                          dict(x=600, y=0, team=1, cls=D.CLASS_MINION)])
    st = CO.init(2, 4)._replace(elixir=jnp.asarray([2140, 2139], jnp.int32),
                                elixir_until=jnp.asarray([180.0, 180.0]))
    own = H.own([], [])
    p = pk([0, 0, 0], [1, 1, 3], [100.0, 100.0, 100.0], flags=[0, D.TAG_AOE, 0])
    rep, _ = H.resolve(p, u)
    _, eff = CO.on_damage(st, own, H.ctx(), u, rep)
    assert float(eff.heal[0]) == pytest.approx(0.12 * (100 + 33.0), rel=1e-5)
    # Sorcery holder is unit 1 (team 1): damages champion 0 and its own-team structure (ignored).
    p = pk([1, 1], [0, 2], [100.0, 100.0], D.MAGIC)
    rep, _ = H.resolve(p, u)
    st2, eff = CO.on_damage(st, own, H.ctx(), u, rep)
    assert H.packet_targets(eff.packets, item=2139) == [0]
    assert H.packet_total(eff.packets, item=2139) == 25.0
    _, eff = CO.on_damage(st2, own, H.ctx(now=4.0), u, rep)
    assert H.packet_total(eff.packets, item=2139) == 0.0
    _, eff = CO.on_damage(st2, own, H.ctx(now=5.0), u, rep)
    assert H.packet_total(eff.packets, item=2139) == 25.0


def test_consumables_jit():
    u = H.units(H.champions())
    own = H.own([2003], [2031])
    st = CO.init(2, 2)
    st, _, out = jax.jit(CO.active)(st, own, H.ctx(), u, jnp.asarray([2003, 2031]))
    assert bool(out.used[0]) and bool(out.used[1])
    _, eff = jax.jit(CO.periodic)(st, own, H.ctx(now=0.5), u)
    assert float(eff.heal_plain[1]) == pytest.approx(100.0 / 24.0, rel=1e-5)
