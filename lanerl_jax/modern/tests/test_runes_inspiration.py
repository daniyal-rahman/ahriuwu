"""Inspiration tree (runes.effects.inspiration): RUNES.md §7 rules and §13 fixtures."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import combat as M
from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.items import inventory as INV
from lanerl_jax.modern.items.catalog import catalog
from lanerl_jax.modern.items.effects import consumables as CO
from lanerl_jax.modern.items.effects import runtime as RT
from lanerl_jax.modern.items.effects.core import CC
from lanerl_jax.modern.runes.catalog import rune_catalog
from lanerl_jax.modern.runes.effects import inspiration as IN
from lanerl_jax.modern.tests import item_harness as H
from lanerl_jax.modern.tests import rune_harness as RH

IDS = (8351, 8360, 8369, 8306, 8304, 8321, 8313, 8352, 8345, 8347, 8410, 8316)


def world(x1=300.0, extra=()):
    return H.units(H.champions(x1=x1) + [dict(x=600, y=0, team=1, cls=D.CLASS_MINION)] + list(extra))


def page(*ids):
    """Holder 0 has ``ids``; holder 1 has an empty page (holder isolation)."""
    return RH.perks(list(ids), [])


def tick(state, pg, ctx, u, ev, hook):
    return getattr(IN, hook)(state, pg, ctx, u, ev)


def test_coverage_lists_the_whole_tree():
    assert set(IN.COVERAGE) == set(IDS)
    cat = rune_catalog()
    for pid in IDS:
        assert cat[pid].style == 8300 and IN.COVERAGE[pid]


# ---- grant queue: Biscuit Delivery (F-32), Triple Tonic, Magical Footwear --------------------------

def run_periodic(state, pg, u, *, now, level=1, own=None, granted=(0, 0), ranged=False, **kw):
    ctx = H.ctx(now=now, level=level, ranged=ranged)
    ev = RH.ev(ctx, u.x.shape[0], game_time=jnp.float32(now), own=own,
               granted=jnp.asarray(granted, jnp.int32), **kw)
    return IN.periodic(state, pg, ctx, u, ev) + (ctx, ev)


def test_biscuits_delivered_at_2_4_6_minutes_until_acknowledged():
    u, pg = world(), page(8345)
    st = IN.init(2, 3)
    st, *_ = run_periodic(st, pg, u, now=119.9)
    assert int(st.grant_q[0, 0]) == 0
    st, _, ctx, ev = run_periodic(st, pg, u, now=120.0)
    out = IN.outputs(st, pg, ctx, ev)
    assert [int(x) for x in out.grant_item] == [2010, 0]          # holder 1 has no rune
    st, _, ctx, ev = run_periodic(st, pg, u, now=125.0)            # inventory full: still pending
    assert int(IN.outputs(st, pg, ctx, ev).grant_item[0]) == 2010
    st, _, ctx, ev = run_periodic(st, pg, u, now=126.0, granted=(2010, 0))
    assert int(IN.outputs(st, pg, ctx, ev).grant_item[0]) == 0
    for t in (240.0, 360.0, 480.0, 600.0):
        st, *_ = run_periodic(st, pg, u, now=t)
    assert int(st.biscuits_sched[0]) == 3 and list(np.asarray(st.grant_q[0, :3])) == [2010, 2010, 0]
    assert int(st.biscuits_sched[1]) == 0


def test_biscuit_sold_gives_silent_permanent_health_and_item_heal_fixture():
    u, pg = world(), page(8345)
    st = IN.init(2, 3)
    st, *_ = run_periodic(st, pg, u, now=10.0, sold=jnp.asarray([2010, 2010], jnp.int32))
    s = IN.stats(st, pg, H.ctx(), RH.ev(H.ctx(), 3))
    assert float(s.health[0]) == 30.0 and float(s.silent_health[0]) == 30.0 and float(s.health[1]) == 0.0
    # F-32: the item heal (consumables) at 1000 max HP: 35 / 52.5 / 70 over 5 s.
    own = H.own([2010], [])
    for hp, want in ((1000.0, 35.0), (650.0, 52.5), (300.0, 70.0)):
        cst = CO.init(2, 3)
        ctx = H.ctx(max_hp=1000.0, hp=hp, base_hp=1000.0)
        cst, _, _ = CO.active(cst, own, ctx, u, jnp.asarray([2010, 0], jnp.int32))
        total = 0.0
        for k in range(12):
            cst, eff = CO.periodic(cst, own, ctx._replace(now=jnp.float32(0.5 * k)), u)
            total += float(eff.heal_plain[0])
        assert total == pytest.approx(want, rel=1e-5)


def test_triple_tonic_levels_and_full_inventory_skill():
    u, pg = world(), page(8313)
    st = IN.init(2, 3)
    st, *_ = run_periodic(st, pg, u, now=1.0, level=2)
    assert int(st.grant_q[0, 0]) == 0
    st, *_ = run_periodic(st, pg, u, now=2.0, level=3)
    st, *_ = run_periodic(st, pg, u, now=3.0, level=6)
    st, *_ = run_periodic(st, pg, u, now=4.0, level=9)
    assert list(np.asarray(st.grant_q[0, :4])) == [2151, 2152, 2150, 0]
    st, *_ = run_periodic(st, pg, u, now=5.0, level=12)            # each only once
    assert list(np.asarray(st.grant_q[0, :4])) == [2151, 2152, 2150, 0]
    # Level 9 with a full inventory: Elixir of Skill is consumed at once.
    full = H.own([1036, 1036, 1036, 1036, 1036, 1036], [])
    st = IN.init(2, 3)
    st, _, ctx, ev = run_periodic(st, pg, u, now=1.0, level=9, own=full)
    out = IN.outputs(st, pg, ctx, ev)
    assert list(np.asarray(st.grant_q[0, :3])) == [2151, 2152, 0] and int(out.skill_points[0]) == 1
    st, _, ctx, ev = run_periodic(st, pg, u, now=1.1, level=9, own=full)
    assert int(IN.outputs(st, pg, ctx, ev).skill_points[0]) == 0


def test_magical_footwear_takedowns_forbid_and_move_speed():
    u, pg = world(), page(8304)
    st = IN.init(2, 3)
    boots_rows = [catalog().row(i) for i in (1001, 3047, 3006)]
    st, _, ctx, ev = run_periodic(st, pg, u, now=60.0, kills=H.kills(3, champion_kill=(1, 0), champion_assist=(1, 0)))
    out = IN.outputs(st, pg, ctx, ev)
    assert all(bool(out.forbid_purchase[0, r]) for r in boots_rows)
    assert not bool(out.forbid_purchase[0, catalog().row(1036)]) and not bool(jnp.any(out.forbid_purchase[1]))
    st, *_ = run_periodic(st, pg, u, now=629.9)                    # due 720 - 2 x 45 = 630
    assert int(st.grant_q[0, 0]) == 0
    st, *_ = run_periodic(st, pg, u, now=630.0)
    assert int(st.grant_q[0, 0]) == 2422
    st, _, ctx, ev = run_periodic(st, pg, u, now=631.0, granted=(2422, 0))
    out = IN.outputs(st, pg, ctx, ev)
    assert not bool(jnp.any(out.forbid_purchase)) and int(out.grant_item[0]) == 0
    s = IN.stats(st, pg, ctx, ev._replace(own=H.own([3047], [3047])))
    assert [float(x) for x in s.move_speed] == [10.0, 0.0]
    s = IN.stats(st, pg, ctx, ev._replace(own=H.own([1036], [])))
    assert float(s.move_speed[0]) == 0.0


# ---- Cash Back, Time Warp Tonic, Cosmic Insight --------------------------------------

def test_cash_back_legendary_refund_and_sell_back():
    u, pg = world(), page(8321)
    st = IN.init(2, 3)
    buy = lambda st, iid, now: run_periodic(st, pg, u, now=now, purchased=jnp.asarray([iid, iid], jnp.int32))
    sell = lambda st, iid, now: run_periodic(st, pg, u, now=now, sold=jnp.asarray([iid, iid], jnp.int32))
    st, eff, *_ = buy(st, 3078, 1.0)                               # Trinity Force, 3333 total
    assert float(eff.gold[0]) == pytest.approx(0.075 * 3333, rel=1e-6) and float(eff.gold[1]) == 0.0
    st, eff, *_ = buy(st, 1036, 2.0)                               # Long Sword: not Legendary
    assert float(eff.gold[0]) == 0.0
    st, eff, *_ = buy(st, 3047, 3.0)                               # tier-2 boots: epic
    assert float(eff.gold[0]) == 0.0
    st, eff, *_ = buy(st, 3170, 3.5)                               # tier-3 boots: epicness 7
    assert float(eff.gold[0]) == 0.0
    st, eff, *_ = sell(st, 3078, 4.0)
    assert float(eff.gold[0]) == pytest.approx(-0.075 * 3333, rel=1e-6)
    st, eff, *_ = sell(st, 3078, 5.0)                              # no refund left on that row
    assert float(eff.gold[0]) == 0.0
    assert IN._tables()["legendary"][catalog().row(3026)]          # Guardian Angel is a normal Legendary


def test_time_warp_tonic_instant_heal():
    u, pg = world(), page(8352)
    st = IN.init(2, 3)
    for pot, want in ((2003, 48.0), (2031, 40.0), (2010, 0.0)):
        _, eff, *_ = run_periodic(st, pg, u, now=1.0, potion_drunk=jnp.asarray([pot, pot], jnp.int32))
        assert float(eff.heal_plain[0]) == pytest.approx(want, rel=1e-6) and float(eff.heal_plain[1]) == 0.0


def test_cosmic_insight_haste():
    pg = page(8347)
    ctx = H.ctx()
    s = IN.stats(IN.init(2, 3), pg, ctx, RH.ev(ctx, 3))
    assert [float(x) for x in s.summoner_haste] == [18.0, 0.0] and [float(x) for x in s.item_haste] == [10.0, 0.0]


# ---- Jack of All Trades ------------------------------------------------------

def own_rows(*ids):
    """(2, I) owned counts for holder 0 without inventory-slot limits."""
    out = np.zeros((2, len(catalog().ids)), np.int32)
    for iid in ids:
        out[0, catalog().row(iid)] += 1
    return jnp.asarray(out)


def test_jack_of_all_trades_stacks_and_thresholds():
    pg = page(8316)
    ctx = H.ctx()
    st = IN.init(2, 3)
    stats = lambda own: IN.stats(st, pg, ctx, RH.ev(ctx, 3, own=own))
    s = stats(own_rows(3078))                                       # Trinity Force: AD, AS, AH, HP
    assert float(s.ability_haste[0]) == 4.0 and float(s.adaptive_force[0]) == 0.0
    s = stats(own_rows(3078, 1029))                                 # + Cloth Armor = 5 types
    assert float(s.ability_haste[0]) == 5.0 and float(s.adaptive_force[0]) == 8.0
    s = stats(own_rows(3078, 1029, 1033, 1026, 1027, 1018))         # + MR, AP, mana, crit = 9
    assert float(s.ability_haste[0]) == 9.0 and float(s.adaptive_force[0]) == 8.0
    s = stats(own_rows(3078, 1029, 1033, 1026, 1027, 1018, 1001))   # + boots flat MS = 10
    assert float(s.ability_haste[0]) == 10.0 and float(s.adaptive_force[0]) == 20.0
    assert float(s.ability_haste[1]) == 0.0
    s = stats(own_rows(3078, 3078))                                 # duplicates do not stack
    assert float(s.ability_haste[0]) == 4.0
    s = stats(own_rows(3006))                                       # Berserker's: AS + MS; slow resist never counts
    assert float(s.ability_haste[0]) == 2.0
    assert float(stats(own_rows(3009)).ability_haste[0]) == 1.0      # Swiftness: MS (+ slow resist, ineligible)


# ---- Approach Velocity ---------------------------------------------------------

def test_approach_velocity_own_and_other_impairments():
    pg = page(8410)
    st = IN.init(2, 3)

    def av(x1, *, facing=(1.0, 0.0), own=False, other=False, visible=True):
        u = world(x1=x1)
        ctx = H.ctx(facing=facing)
        ibh = jnp.zeros((2, 3), bool).at[0, 1].set(own)
        ev = RH.ev(ctx, 3, impaired_by_holder=ibh, movement_impaired=jnp.zeros((3,), bool).at[1].set(other | own),
                   visible=jnp.ones((2, 3), bool).at[0, 1].set(visible))
        s2 = IN.post_tick(st, pg, ctx, u, ev)
        return float(IN.stats(s2, pg, ctx, ev).percent_move_speed[0])

    assert av(3000.0, own=True, visible=False) == pytest.approx(0.15)
    assert av(900.0, other=True) == pytest.approx(0.075)
    assert av(1200.0, other=True) == 0.0
    assert av(900.0, other=True, visible=False) == 0.0
    assert av(900.0, other=True, facing=(-1.0, 0.0)) == 0.0
    assert av(900.0, other=True, facing=(0.0, 1.0)) == pytest.approx(0.075)   # 90 deg edge of the arc
    assert av(900.0) == 0.0


# ---- First Strike --------------------------------------------------------------

def fs_damage(st, pg, u, *, now, raw, start, struck_first=True, level=1, ranged=False, src=0, dst=1, item=0):
    ctx = H.ctx(now=now, level=level, ranged=ranged)
    p = RH.hit_packet(src, dst, raw, dtype=D.TRUE, item=item)
    rep = RH.report(p, u)
    clk = RH.clocks(champion_combat_start=start, last_champion_combat=now, struck_first=struck_first)
    ev = RH.ev(ctx, 3, report=rep, clocks=clk)
    return IN.on_damage(st, pg, ctx, u, ev)


def test_first_strike_cooldown_formula():
    for lvl, want in ((1, 25.0), (9, 25 - 10 * 8 / 17), (18, 15.0), (20, 15.0)):
        assert float(IN.first_strike_cooldown(lvl)) == pytest.approx(want, abs=1e-4)


def test_first_strike_activation_bonus_delay_and_gold():
    u, pg = world(), page(8369)
    st = IN.init(2, 3)
    st, eff = fs_damage(st, pg, u, now=10.0, raw=100.0, start=9.8)
    assert float(eff.gold[0]) == 10.0 and float(st.fs_until[0]) == pytest.approx(13.0)
    assert float(st.fs_cd_until[0]) == pytest.approx(35.0)
    # Not emitted before 0.4 s.
    st, eff, *_ = run_periodic(st, pg, u, now=10.3)
    assert RH.total(eff.packets, rune=8369) == 0.0
    st, eff, *_ = run_periodic(st, pg, u, now=10.4)
    p = eff.packets
    sel = np.asarray(p.valid) & (np.asarray(p.item) == -8369)
    assert RH.total(p, rune=8369, dst=1) == pytest.approx(7.0, rel=1e-6)
    assert np.all(np.asarray(p.dtype)[sel] == D.TRUE)
    assert np.all((np.asarray(p.flags)[sel] & (D.TAG_PROC | D.TAG_INDIRECT)) == (D.TAG_PROC | D.TAG_INDIRECT))
    # The bonus packet resolving is not itself amplified, but enters the gold ledger.
    st, eff = fs_damage(st, pg, u, now=10.4, raw=7.0, start=9.8, item=-8369)
    assert int(jnp.sum(st.fs_due[0] < 1e8)) == 0 and float(st.fs_gold_acc[0]) == pytest.approx(7.0)
    # A later hit inside the buff also gets 7%.
    st, _ = fs_damage(st, pg, u, now=12.0, raw=200.0, start=9.8)
    st, eff, *_ = run_periodic(st, pg, u, now=12.4)
    assert RH.total(eff.packets, rune=8369) == pytest.approx(14.0, rel=1e-6)
    st, _ = fs_damage(st, pg, u, now=12.4, raw=14.0, start=9.8, item=-8369)
    # Damage after the buff is not amplified; gold (50% of 21) is paid once it is over.
    st, _ = fs_damage(st, pg, u, now=13.5, raw=100.0, start=9.8)
    assert int(jnp.sum(st.fs_due[0] < 1e8)) == 0
    st, eff, ctx, ev = run_periodic(st, pg, u, now=13.6)
    assert float(eff.gold[0]) == pytest.approx(10.5, rel=1e-5)
    assert float(IN.outputs(st, pg, ctx, ev).first_strike_gold[0]) == pytest.approx(10.5, rel=1e-5)
    st, eff, *_ = run_periodic(st, pg, u, now=13.7)
    assert float(eff.gold[0]) == 0.0


def test_first_strike_ranged_gold_and_grace_window():
    u, pg = world(), page(8369)
    st = IN.init(2, 3)
    late, _ = fs_damage(st, pg, u, now=10.3, raw=100.0, start=10.0)          # 0.3 s > 0.25 s grace
    assert float(late.fs_until[0]) < 0
    st, _ = fs_damage(st, pg, u, now=10.25, raw=100.0, start=10.0, ranged=True)
    st, eff, *_ = run_periodic(st, pg, u, now=10.65)
    assert RH.total(eff.packets, rune=8369) == pytest.approx(7.0, rel=1e-6)
    st, _ = fs_damage(st, pg, u, now=10.65, raw=7.0, start=10.0, item=-8369)
    st, eff, *_ = run_periodic(st, pg, u, now=13.5, ranged=True)
    assert float(eff.gold[0]) == pytest.approx(0.35 * 7.0, rel=1e-5)


def test_first_strike_struck_first_by_enemy_locks_out():
    u, pg = world(), page(8369)
    st = IN.init(2, 3)
    # Enemy champion opened the episode this tick: full cooldown, no buff, no gold.
    st, eff = fs_damage(st, pg, u, now=10.0, raw=50.0, start=10.0, struck_first=False, src=1, dst=0)
    assert float(st.fs_cd_until[0]) == pytest.approx(35.0) and float(eff.gold[0]) == 0.0
    st2, _ = fs_damage(st, pg, u, now=10.1, raw=100.0, start=10.0, struck_first=False)
    assert float(st2.fs_until[0]) < 0
    # Holder 1 (no First Strike) is unaffected throughout.
    assert float(st2.fs_cd_until[1]) < 0


def test_first_strike_through_combat_tick_jit():
    u = H.units(H.champions(x1=150.) + [dict(x=2000, y=0, team=1)])
    n = u.x.shape[0]
    inv = INV.inventory_from_ids([[], []])
    own, item = INV.owned_counts(inv), INV.inventory_stats(inv)
    hp = jnp.asarray([1000., 1000., 500.], jnp.float32)
    u = u._replace(hp=hp, max_hp=hp)
    dfn = D.default_defense(n)._replace(unit_class=u.cls)
    off = D.default_offense(n)._replace(unit_class=u.cls)
    pg = page(8369)

    def run(state, hp, now, base):
        ctx = H.ctx(base_hp=1000., max_hp=1000.)._replace(now=jnp.asarray(now, jnp.float32), hp=hp[:2])
        uu = u._replace(hp=hp)
        return M.combat_tick(state, own, pg, ctx, uu, attack=H.attack(hit=(False, False)),
                             cast=H.cast(started=(False, False)), request=jnp.zeros((2,), jnp.int32),
                             base_packets=base, base_offense=off, base_defense=dfn, hp=hp, max_hp=u.max_hp,
                             shields=D.init_shields(n), status=RT.init_status(n), kills=H.kills(n),
                             holder_stats=item)

    f = jax.jit(run)
    hit = D.packets(jnp.ones(1, bool), 0, 1, 100.0, D.TRUE, D.BASIC_ATTACK)
    none = D.packets(jnp.zeros(1, bool), 0, 1, 0.0, D.TRUE, 0)
    out = f(M.init_combat(2, n), hp, 20.0, hit)
    assert float(out.effects.gold[0]) == pytest.approx(10.0)
    assert float(out.rune_outputs.first_strike_gold[0]) == pytest.approx(10.0)
    hp1 = out.hp
    out = f(out.state, out.hp, 20.2, none)
    assert float(out.hp[1]) == pytest.approx(float(hp1[1]))
    out = f(out.state, out.hp, 20.5, none)                           # 7 true damage lands
    assert float(hp1[1]) - float(out.hp[1]) == pytest.approx(7.0, rel=1e-5)
    out = f(out.state, out.hp, 23.5, none)
    assert float(out.effects.gold[0]) == pytest.approx(3.5, rel=1e-5)
    assert int(out.packet_overflow) == 0


# ---- Glacial Augment ------------------------------------------------------------

def test_glacial_augment_zones_slow_cooldown_and_ally_reduction():
    extra = [dict(x=600, y=20, team=1, cls=D.CLASS_MINION),       # unit 3 behind the zone start (x = 400)
             dict(x=300, y=500, team=1, cls=D.CLASS_MINION),      # unit 4 far off the rays
             dict(x=-200, y=0, team=0, cls=D.CLASS_CHAMPION, radius=65.0)]   # unit 5: holder's ally
    u = H.units(H.champions(x1=300.0) + [dict(x=5000, y=0, team=1)] + extra)
    n = u.x.shape[0]
    pg = page(8351)
    st = IN.init(2, n)
    ctx = H.ctx(now=5.0, bonus_ad=100.0)
    imm = jnp.zeros((2, n), bool).at[0, 1].set(True)
    ccd = jnp.zeros((2, n), jnp.float32).at[0, 1].set(1.5)
    ev = RH.ev(ctx, n, cc=CC(jnp.zeros((2, n), bool), imm), cc_duration=ccd)
    st, eff = IN.on_cc(st, pg, ctx, u, ev)
    assert float(st.ga_until[0]) == pytest.approx(5.0 + 3.0 + 1.5) and float(st.ga_cd_until[0]) == pytest.approx(30.0)
    assert float(st.ga_until[1]) < 0
    slow = np.asarray(eff.slow)
    assert slow[1] == pytest.approx(0.27, rel=1e-5)                 # 20% + 7% per 100 bonus AD
    assert slow[3] == 0.0 and slow[4] == 0.0 and slow[0] == 0.0 and slow[5] == 0.0      # off-ray enemy, holder, ally
    # Ray toward the holder runs from 100 behind the target (x=400) to x=-300; the ally ray hits unit 5.
    mask = np.asarray(IN.zone_mask(st, ctx, u))[0]
    assert mask[1] and mask[0] and mask[5] and not mask[4]
    # Still slowing next tick; nothing new triggers on cooldown.
    ctx2 = ctx._replace(now=jnp.float32(9.0))
    st2, eff2 = IN.on_cc(st, pg, ctx2, u, RH.ev(ctx2, n, cc=CC(jnp.zeros((2, n), bool), imm), cc_duration=ccd))
    assert float(eff2.slow[1]) > 0 and float(st2.ga_until[0]) == pytest.approx(9.5)
    ctx3 = ctx._replace(now=jnp.float32(9.6))
    _, eff3 = IN.on_cc(st2, pg, ctx3, u, RH.ev(ctx3, n))
    assert float(jnp.max(eff3.slow)) == 0.0
    # -15% for the zoned enemy's damage to the holder's ally only (the holder itself is excluded).
    p = D.packets(jnp.ones(3, bool), jnp.asarray([1, 1, 4]), jnp.asarray([5, 0, 5]), 100.0, D.PHYSICAL)
    amp = np.asarray(IN.packet_amp(st, pg, ctx, u, RH.ev(ctx, n), p))
    assert amp[0] == pytest.approx(-0.15) and amp[1] == 0.0 and amp[2] == 0.0


def test_glacial_slow_formula():
    assert float(IN.glacial_slow(0.0, 0.0, 0.0)) == pytest.approx(0.20)
    assert float(IN.glacial_slow(0.0, 100.0, 0.1)) == pytest.approx(0.20 + 0.06 + 0.09)


# ---- Unsealed Spellbook ------------------------------------------------------------

def test_unsealed_spellbook_availability_and_cooldowns():
    u, pg = world(), page(8360)
    st = IN.init(2, 3)

    def step(st, now, req=0, last_combat=-1e9):
        ctx = H.ctx(now=now)
        ev = RH.ev(ctx, 3, game_time=jnp.float32(now), spellbook_request=jnp.asarray([req, req], jnp.int32),
                   clocks=RH.clocks(last_combat=last_combat))
        st, _ = IN.periodic(st, pg, ctx, u, ev)
        return st, IN.outputs(st, pg, ctx, ev)

    st, out = step(st, 359.0, req=14)
    assert not bool(out.spellbook_swap_ready[0]) and int(st.sb_recent[0, 0]) == 0
    st, out = step(st, 360.0, last_combat=356.0)                     # in combat 4 s ago
    assert not bool(out.spellbook_swap_ready[0])
    st, out = step(st, 361.0)
    assert bool(out.spellbook_swap_ready[0]) and not bool(out.spellbook_swap_ready[1])
    st, out = step(st, 362.0, req=14)
    assert float(st.sb_ready_at[0]) == pytest.approx(362.0 + 245.0) and not bool(out.spellbook_swap_ready[0])
    st, _ = step(st, 607.0, req=3)                                    # unique 2 -> 220 s
    assert float(st.sb_ready_at[0]) == pytest.approx(827.0)
    st, _ = step(st, 827.0, req=7)                                    # unique 3 -> 195 s
    assert float(st.sb_ready_at[0]) == pytest.approx(1022.0)
    st, _ = step(st, 1022.0, req=14)                                  # 14 is within the last 3 picks: refused
    assert list(np.asarray(st.sb_recent[0])) == [7, 3, 14] and float(st.sb_ready_at[0]) == pytest.approx(1022.0)
    st, _ = step(st, 1022.0, req=4)                                   # unique 4 -> 170 s
    assert list(np.asarray(st.sb_recent[0])) == [4, 7, 3] and float(st.sb_ready_at[0]) == pytest.approx(1192.0)
    assert not bool(IN.spellbook_can_select(st, jnp.int32(3))[0]) and bool(IN.spellbook_can_select(st, jnp.int32(1))[0])
    assert float(IN.spellbook_cooldown(6)) == 120.0 and float(IN.spellbook_cooldown(9)) == 120.0


# ---- Hextech Flashtraption -----------------------------------------------------------

def hx_step(st, pg, u, now, req=0, flash_cd=100.0, combat=False):
    ctx = H.ctx(now=now)
    ev = RH.ev(ctx, 3, hexflash_request=jnp.asarray([req, req], jnp.int32),
               flash_cooldown=jnp.full((2,), flash_cd, jnp.float32),
               clocks=RH.clocks(last_champion_combat=now if combat else -1e9))
    st = IN.post_tick(st, pg, ctx, u, ev)
    return st, IN.outputs(st, pg, ctx, ev)


def test_hexflash_channel_release_and_cooldowns():
    u, pg = world(), page(8306)
    st = IN.init(2, 3)
    st, out = hx_step(st, pg, u, 1.0, req=1, flash_cd=2.0)            # Flash nearly up: no Hexflash
    assert not bool(out.move_locked[0])
    st, out = hx_step(st, pg, u, 1.0, req=1)
    assert bool(out.move_locked[0]) and not bool(out.move_locked[1])
    st, out = hx_step(st, pg, u, 2.5, req=2)                          # 1.5 s: 400 range
    assert bool(out.blink[0]) and float(out.blink_range[0]) == pytest.approx(400.0) and not bool(out.move_locked[0])
    assert float(st.hx_cd_until[0]) == pytest.approx(22.5)
    s = IN.stats(st, pg, H.ctx(now=2.6), RH.ev(H.ctx(now=2.6), 3))
    assert float(s.percent_move_speed[0]) == pytest.approx(0.5)
    st, out = hx_step(st, pg, u, 3.0, req=0)
    assert not bool(out.blink[0])
    # Released at 1.0 s: 200 + 3 x 40.
    st, _ = hx_step(st, pg, u, 30.0, req=1)
    st, out = hx_step(st, pg, u, 31.0, req=2)
    assert bool(out.blink[0]) and float(out.blink_range[0]) == pytest.approx(320.0)
    # Early release: 10 s cooldown, no blink.
    st, _ = hx_step(st, pg, u, 60.0, req=1)
    st, out = hx_step(st, pg, u, 60.5, req=2)
    assert not bool(out.blink[0]) and float(st.hx_cd_until[0]) == pytest.approx(70.5)
    # Auto release at 2 s.
    st, _ = hx_step(st, pg, u, 80.0, req=1)
    st, out = hx_step(st, pg, u, 82.0)
    assert bool(out.blink[0]) and float(out.blink_range[0]) == pytest.approx(400.0)
    # Champion combat during the channel cancels it with a 10 s cooldown.
    st, _ = hx_step(st, pg, u, 110.0, req=1)
    st, out = hx_step(st, pg, u, 110.5, combat=True)
    assert not bool(out.move_locked[0]) and not bool(out.blink[0])
    assert float(st.hx_cd_until[0]) == pytest.approx(120.5)


def test_hexflash_cooldown_scaled_by_cosmic_insight():
    u, pg = world(), page(8306, 8347)
    st = IN.init(2, 3)
    st, _ = hx_step(st, pg, u, 1.0, req=1)
    st, _ = hx_step(st, pg, u, 2.5, req=2)
    assert float(st.hx_cd_until[0]) == pytest.approx(2.5 + 20.0 * 100.0 / 118.0, rel=1e-5)


def test_periodic_and_post_tick_jit():
    u, pg = world(), page(*IDS)
    st = IN.init(2, 3)
    ctx = H.ctx(now=130.0, level=9)
    ev = RH.ev(ctx, 3, game_time=jnp.float32(130.0), own=H.own([3078], []),
               purchased=jnp.asarray([3078, 0], jnp.int32))
    st2, eff = jax.jit(lambda s: IN.periodic(s, pg, ctx, u, ev))(st)
    assert float(eff.gold[0]) == pytest.approx(0.075 * 3333, rel=1e-6)
    assert int(st2.grant_q[0, 0]) == 2010
    st3 = jax.jit(lambda s: IN.post_tick(s, pg, ctx, u, ev))(st2)
    out = jax.jit(lambda s: IN.outputs(s, pg, ctx, ev))(st3)
    assert int(out.grant_item[0]) == 2010
