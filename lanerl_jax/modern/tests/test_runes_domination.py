"""Domination tree (RUNES.md §4, fixtures §13 F-26/27/28/34)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.items.effects.core import CC
from lanerl_jax.modern.runes.effects import domination as M
from lanerl_jax.modern.runes.effects.core import rune_item
from lanerl_jax.modern.tests import item_harness as H
from lanerl_jax.modern.tests import rune_harness as RH

N = 2


def world(hp1=1000.0):
    u = H.units(H.champions())
    return u._replace(hp=u.hp.at[1].set(jnp.float32(hp1)))


def page(*ids):
    return RH.perks(list(ids), [])


def rune_packets(eff, perk):
    p = eff.packets
    sel = np.asarray(p.valid) & (np.asarray(p.item) == rune_item(perk))
    return np.asarray(p.raw)[sel], np.asarray(p.dtype)[sel], np.asarray(p.dst)[sel]


def damage(st, pg, ctx, u, p, **kw):
    rep = RH.report(p, u)
    return M.on_damage(st, pg, ctx, u, RH.ev(ctx, N, report=rep, **kw))


def hit(raw=50.0, cast_id=0, flags=D.BASIC_ATTACK, src=0, dst=1, item=0):
    return RH.hit_packet(src, dst, raw, flags=flags, cast_id=cast_id, item=item)


def test_coverage_lists_tree_without_deferred_vision():
    assert set(M.COVERAGE) == {8112, 8128, 9923, 8126, 8139, 8143, 8140, 8135, 8105, 8106}


# ---- Electrocute ---------------------------------------------------------------

def _electrocute(level=1, bonus_ad=0.0, ap=0.0):
    u, pg, st = world(), page(M.ELECTROCUTE), M.init(2, N)
    for t in (0.0, 1.0, 2.0):
        ctx = H.ctx(now=t, level=level, bonus_ad=bonus_ad, ap=ap)
        st, eff = damage(st, pg, ctx, u, hit())
    return st, pg, u


@pytest.mark.parametrize("level,expected", [(1, 70.0), (9, 150.0), (18, 240.0)])
def test_f26_electrocute_values_and_delay(level, expected):
    st, pg, u = _electrocute(level)
    assert float(st.elec_due[0]) == pytest.approx(2.25)
    _, eff = M.periodic(st, pg, H.ctx(now=2.2, level=level), u, RH.ev(H.ctx(now=2.2), N))
    assert rune_packets(eff, M.ELECTROCUTE)[0].size == 0
    ctx = H.ctx(now=2.25, level=level)
    st2, eff = M.periodic(st, pg, ctx, u, RH.ev(ctx, N))
    raw, dt, dst = rune_packets(eff, M.ELECTROCUTE)
    assert raw.tolist() == pytest.approx([expected], abs=1e-4)
    assert dt.tolist() == [D.MAGIC] and dst.tolist() == [1]
    assert float(st2.elec_due[0]) > 1e8
    # Holder 1 has no rune: no stacks, nothing queued.
    assert int(st.elec_stacks[1].sum()) == 0 and float(st.elec_due[1]) > 1e8


def test_electrocute_variable_damage_physical_with_bonus_ad():
    st, _, _ = _electrocute(1, bonus_ad=50.0)
    assert float(st.elec_raw[0]) == pytest.approx(75.0) and int(st.elec_dtype[0]) == D.PHYSICAL


def test_electrocute_one_stack_per_cast_instance():
    u, pg, st = world(), page(M.ELECTROCUTE), M.init(2, N)
    ctx = H.ctx(now=0.0)
    three = D.concat_packets(hit(cast_id=7), hit(cast_id=7), hit(cast_id=7))
    st, _ = damage(st, pg, ctx, u, three)
    assert int(st.elec_stacks[0, 1]) == 1
    st, _ = damage(st, pg, H.ctx(now=0.5), u, hit(cast_id=7))     # same instance, later tick (DoT)
    assert int(st.elec_stacks[0, 1]) == 1
    st, _ = damage(st, pg, H.ctx(now=0.6), u, D.concat_packets(hit(cast_id=8), hit(cast_id=0)))
    assert float(st.elec_due[0]) == pytest.approx(0.85)          # 3rd stack -> trigger


def test_electrocute_proc_damage_does_not_stack_but_pet_does():
    u, pg, st = world(), page(M.ELECTROCUTE), M.init(2, N)
    st, _ = damage(st, pg, H.ctx(now=0.0), u, hit(flags=D.ON_HIT_ITEM))
    assert int(st.elec_stacks[0, 1]) == 0
    st, _ = damage(st, pg, H.ctx(now=0.0), u, hit(flags=D.TAG_PROC | D.TAG_PET))
    assert int(st.elec_stacks[0, 1]) == 1


def test_electrocute_window_does_not_refresh_and_cooldown():
    u, pg, st = world(), page(M.ELECTROCUTE), M.init(2, N)
    for t in (0.0, 1.0, 3.5):
        st, _ = damage(st, pg, H.ctx(now=t), u, hit())
    assert float(st.elec_due[0]) > 1e8 and int(st.elec_stacks[0, 1]) == 1   # window from t=0 expired
    for t in (4.0, 5.0):
        st, _ = damage(st, pg, H.ctx(now=t), u, hit())
    assert float(st.elec_cd_until[0]) == pytest.approx(25.0)
    for t in (6.0, 7.0, 8.0):
        st, _ = damage(st, pg, H.ctx(now=t), u, hit())
    assert int(st.elec_stacks[0, 1]) == 0                         # no stacks on cooldown
    for t in (25.0, 26.0, 27.0):
        st, _ = damage(st, pg, H.ctx(now=t), u, hit())
    assert float(st.elec_cd_until[0]) == pytest.approx(47.0)


def test_electrocute_cc_stack_pairs_with_same_tick_damage():
    u, pg, st = world(), page(M.ELECTROCUTE), M.init(2, N)
    ctx = H.ctx(now=0.0)
    cc = CC(jnp.zeros((2, N), bool), jnp.zeros((2, N), bool).at[0, 1].set(True))
    st, _ = M.on_cc(st, pg, ctx, u, RH.ev(ctx, N, cc=cc))
    assert int(st.elec_stacks[0, 1]) == 1
    st, _ = damage(st, pg, ctx, u, hit(cast_id=3))
    assert int(st.elec_stacks[0, 1]) == 1
    st, _ = damage(st, pg, H.ctx(now=0.5), u, hit(cast_id=4))
    assert int(st.elec_stacks[0, 1]) == 2


# ---- Dark Harvest --------------------------------------------------------------

def test_dark_harvest_trigger_soul_cooldown_and_reset():
    pg, st = page(M.DARK_HARVEST), M.init(2, N)
    st, eff = damage(st, pg, H.ctx(now=0.0), world(600.0), hit(100.0))
    assert rune_packets(eff, M.DARK_HARVEST)[0].size == 0         # 600/1000 is not below 50%
    u = world(400.0)
    st, eff = damage(st, pg, H.ctx(now=0.0, bonus_ad=20.0), u, hit(100.0))
    raw, dt, _ = rune_packets(eff, M.DARK_HARVEST)
    assert raw.tolist() == pytest.approx([32.0]) and dt.tolist() == [D.PHYSICAL]
    st, _ = damage(st, pg, H.ctx(now=0.5), u, hit(1.5))
    for t, souls in ((1.7, 0.0), (1.75, 1.0)):
        ctx = H.ctx(now=t)
        st, _ = M.periodic(st, pg, ctx, u, RH.ev(ctx, N))
        assert float(st.dh_souls[0]) == souls
    # Proc damage does not trigger; cooldown 35 s.
    st2, eff = damage(st, pg, H.ctx(now=40.0), u, hit(100.0, flags=D.ON_HIT_ITEM))
    assert rune_packets(eff, M.DARK_HARVEST)[0].size == 0
    st2, eff = damage(st, pg, H.ctx(now=10.0), u, hit(100.0))
    assert rune_packets(eff, M.DARK_HARVEST)[0].size == 0
    # Takedown resets the remaining cooldown to 1 s; then 30 + 11 souls.
    ctx = H.ctx(now=10.0)
    st, _ = M.on_takedown(st, pg, ctx, u, RH.ev(ctx, N, kills=H.kills(N, champion_kill=(1, 0),
                          killed_units=jnp.zeros((2, N), bool).at[0, 1].set(True))))
    assert float(st.dh_cd_until[0]) == pytest.approx(11.0)
    st, eff = damage(st, pg, H.ctx(now=11.0), u, hit(100.0))
    assert rune_packets(eff, M.DARK_HARVEST)[0].tolist() == pytest.approx([41.0])


def test_dark_harvest_execute_credit_soul_while_ready():
    pg, st, u = page(M.DARK_HARVEST), M.init(2, N), world()
    ctx = H.ctx(now=1.0)
    st, _ = M.on_takedown(st, pg, ctx, u, RH.ev(ctx, N, execute_credit=jnp.asarray([1.0, 1.0])))
    assert st.dh_souls.tolist() == [1.0, 0.0]


# ---- Hail of Blades ---------------------------------------------------------------

def _hob_ev(ctx, *, start=False, launch=False, hit_=False, cancel=False, reset=False, target=1):
    att = H.Attack(jnp.asarray([launch, False]), jnp.asarray([hit_, False]), jnp.asarray([target, 0], jnp.int32),
                   jnp.asarray([100.0, 0.0], jnp.float32), jnp.zeros(2, bool))
    return RH.ev(ctx, N, attack=att, attack_started=jnp.asarray([start, False]),
                 attack_start_target=jnp.asarray([target, 0], jnp.int32),
                 attack_cancelled=jnp.asarray([cancel, False]), attack_reset=jnp.asarray([reset, False]))


def _hob_attack(st, pg, u, t, **kw):
    ctx = H.ctx(now=t)
    st, _ = M.on_attack(st, pg, ctx, u, _hob_ev(ctx, launch=True, hit_=True, **kw))
    st, eff = M.on_hit(st, pg, ctx, u, _hob_ev(ctx, launch=True, hit_=True, **kw))
    return st, rune_packets(eff, M.HAIL_OF_BLADES)


def test_f34_hail_of_blades_three_empowered_attacks():
    u, pg, st = world(), page(M.HAIL_OF_BLADES), M.init(2, N)
    ctx = H.ctx(now=0.0)
    st, _ = M.on_attack(st, pg, ctx, u, _hob_ev(ctx, start=True))
    s = M.stats(st, pg, ctx, RH.ev(ctx, N))
    assert float(s.attack_speed[0]) == 0.0                         # not active until the windup completes
    for i, t in enumerate((0.2, 0.8, 1.4)):
        if i:
            s = M.stats(st, pg, H.ctx(now=t), RH.ev(H.ctx(now=t), N))
            assert float(s.attack_speed[0]) == pytest.approx(0.9) and float(s.attack_speed_cap_lift[0]) == 1.0
            assert float(s.attack_speed[1]) == 0.0
        st, (raw, dt, dst) = _hob_attack(st, pg, u, t)
        assert raw.tolist() == pytest.approx([2.0]) and dt.tolist() == [D.TRUE] and dst.tolist() == [1]
    assert not bool(st.hob_active[0]) and float(st.hob_cd_until[0]) == pytest.approx(11.4)
    st, (raw, _, _) = _hob_attack(st, pg, u, 2.0)
    assert raw.size == 0
    s = M.stats(st, pg, H.ctx(now=2.0), RH.ev(H.ctx(now=2.0), N))
    assert float(s.attack_speed[0]) == 0.0
    # Ranged value and damage packet flags.
    ctx_r = H.ctx(now=0.5, ranged=True)
    assert float(M.stats(M.init(2, N)._replace(hob_active=jnp.asarray([True, False]), hob_stacks=jnp.asarray([2, 0]),
                                               hob_expire=jnp.asarray([3.0, 0.0])), pg, ctx_r,
                         RH.ev(ctx_r, N)).attack_speed[0]) == pytest.approx(0.6)


def test_hail_of_blades_resets_timeout_and_cancel():
    u, pg = world(), page(M.HAIL_OF_BLADES)
    st = M.init(2, N)
    ctx = H.ctx(now=0.0)
    st, _ = M.on_attack(st, pg, ctx, u, _hob_ev(ctx, start=True))
    st, _ = _hob_attack(st, pg, u, 0.2)
    for t in (0.3, 0.4, 0.5):                                      # 3 resets, only 2 count
        ctx = H.ctx(now=t)
        st, _ = M.on_attack(st, pg, ctx, u, _hob_ev(ctx, reset=True))
    assert int(st.hob_stacks[0]) == 4
    # Timeout: 3 s without an attack ends it; cooldown from the end.
    ctx = H.ctx(now=3.3)
    st, _ = M.on_attack(st, pg, ctx, u, _hob_ev(ctx))
    assert not bool(st.hob_active[0]) and float(st.hob_cd_until[0]) == pytest.approx(13.2)
    # Cancelled windup: no stacks, brief lockout.
    st = M.init(2, N)
    ctx = H.ctx(now=0.0)
    st, _ = M.on_attack(st, pg, ctx, u, _hob_ev(ctx, start=True))
    ctx = H.ctx(now=0.1)
    st, _ = M.on_attack(st, pg, ctx, u, _hob_ev(ctx, cancel=True))
    assert not bool(st.hob_pending[0]) and not bool(st.hob_active[0])
    assert float(st.hob_cd_until[0]) == pytest.approx(0.1 + M.HOB_CANCEL_LOCKOUT)
    # Windup on a minion does not trigger.
    u3 = H.units(H.champions() + [dict(x=100, y=0, team=1)])
    st = M.init(2, 3)
    ctx = H.ctx(now=0.0)
    ev = RH.ev(ctx, 3, attack_started=jnp.asarray([True, False]), attack_start_target=jnp.asarray([2, 0], jnp.int32))
    st, _ = M.on_attack(st, pg, ctx, u3, ev)
    assert not bool(st.hob_pending[0])


# ---- Cheap Shot / Taste of Blood / Sudden Impact ---------------------------------------

@pytest.mark.parametrize("level,expected", [(1, 10.0), (18, 45.0)])
def test_cheap_shot(level, expected):
    u, pg, st = world(), page(M.CHEAP_SHOT), M.init(2, N)
    imp = jnp.asarray([False, True])
    st0, eff = damage(st, pg, H.ctx(now=0.0, level=level), u, hit())
    assert rune_packets(eff, M.CHEAP_SHOT)[0].size == 0               # not impaired
    st1, eff = damage(st, pg, H.ctx(now=0.0, level=level), u, hit(flags=D.ON_HIT_ITEM), impaired=imp)
    assert rune_packets(eff, M.CHEAP_SHOT)[0].size == 0               # proc damage
    st, eff = damage(st, pg, H.ctx(now=0.0, level=level), u, hit(), impaired=imp)
    raw, dt, _ = rune_packets(eff, M.CHEAP_SHOT)
    assert raw.tolist() == pytest.approx([expected], abs=1e-4) and dt.tolist() == [D.TRUE]
    _, eff = damage(st, pg, H.ctx(now=3.9, level=level), u, hit(), impaired=imp)
    assert rune_packets(eff, M.CHEAP_SHOT)[0].size == 0
    _, eff = damage(st, pg, H.ctx(now=4.0, level=level), u, hit(), impaired=imp)
    assert rune_packets(eff, M.CHEAP_SHOT)[0].size == 1


@pytest.mark.parametrize("level,expected", [(1, 16.0), (9, 27.2941), (18, 40.0)])
def test_f27_taste_of_blood(level, expected):
    u, pg, st = world(), page(M.TASTE_OF_BLOOD), M.init(2, N)
    _, eff = damage(st, pg, H.ctx(now=0.0, level=level), u, hit())
    assert float(eff.heal[0]) == 0.0                                  # full HP
    st, eff = damage(st, pg, H.ctx(now=0.0, level=level, hp=500.0), u, hit(0.0))
    assert float(eff.heal[0]) == pytest.approx(expected, abs=1e-4) and float(eff.heal[1]) == 0.0
    _, eff = damage(st, pg, H.ctx(now=19.9, level=level, hp=500.0), u, hit())
    assert float(eff.heal[0]) == 0.0
    _, eff = damage(st, pg, H.ctx(now=20.0, level=level, hp=500.0), u, hit())
    assert float(eff.heal[0]) == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize("level,expected", [(1, 20.0), (9, 48.2353), (18, 80.0)])
def test_f28_sudden_impact(level, expected):
    u, pg, st = world(), page(M.SUDDEN_IMPACT), M.init(2, N)
    blink = jnp.asarray([True, True])
    ctx = H.ctx(now=0.0, level=level)
    st, _ = M.on_cast(st, pg, ctx, u, RH.ev(ctx, N, blinked=blink))
    assert float(st.si_armed_until[0]) == pytest.approx(4.0) and float(st.si_armed_until[1]) < -1e8
    st2, eff = damage(st, pg, H.ctx(now=3.9, level=level), u, hit(0.0))
    raw, dt, _ = rune_packets(eff, M.SUDDEN_IMPACT)
    assert raw.tolist() == pytest.approx([expected], abs=1e-4) and dt.tolist() == [D.TRUE]
    assert float(st2.si_cd_until[0]) == pytest.approx(13.9)
    _, eff = damage(st2, pg, H.ctx(now=4.0, level=level), u, hit())
    assert rune_packets(eff, M.SUDDEN_IMPACT)[0].size == 0            # consumed
    # Expiry starts the cooldown; no re-arm during it.
    ctx = H.ctx(now=4.1, level=level)
    st3, _ = M.on_cast(st, pg, ctx, u, RH.ev(ctx, N, blinked=blink))
    assert float(st3.si_cd_until[0]) == pytest.approx(14.0) and float(st3.si_armed_until[0]) < -1e8
    _, eff = damage(st3, pg, H.ctx(now=4.2, level=level), u, hit())
    assert rune_packets(eff, M.SUDDEN_IMPACT)[0].size == 0
    ctx = H.ctx(now=14.0, level=level)
    st4, _ = M.on_cast(st3, pg, ctx, u, RH.ev(ctx, N, blinked=blink))
    assert float(st4.si_armed_until[0]) == pytest.approx(18.0)


# ---- Bounty Hunter row and Grisly Mementos ------------------------------------------

def test_bounty_hunter_row_and_mementos():
    u = world()
    pg = RH.perks([M.TREASURE_HUNTER, M.GRISLY_MEMENTOS], [M.ULTIMATE_HUNTER, M.RELENTLESS_HUNTER])
    st = M.init(2, N)
    ctx = H.ctx(now=100.0)
    clocks = RH.clocks(last_combat=96.0)
    s = M.stats(st, pg, ctx, RH.ev(ctx, N, clocks=clocks))
    assert s.ultimate_haste.tolist() == [0.0, 6.0] and s.trinket_haste.tolist() == [0.0, 0.0]
    kill0 = jnp.zeros((2, N), bool).at[0, 1].set(True).at[1, 0].set(True)
    ev = RH.ev(ctx, N, kills=H.kills(N, champion_kill=(1, 1), killed_units=kill0))
    st, eff = M.on_takedown(st, pg, ctx, u, ev)
    assert eff.gold.tolist() == [50.0, 0.0]
    st, eff = M.on_takedown(st, pg, ctx, u, ev)                      # same champion: no new stack
    assert eff.gold.tolist() == [0.0, 0.0]
    s = M.stats(st, pg, ctx, RH.ev(ctx, N, clocks=clocks))
    assert s.ultimate_haste.tolist() == [0.0, 11.0] and s.move_speed.tolist() == [0.0, 0.0]   # in combat
    s = M.stats(st, pg, ctx, RH.ev(ctx, N, clocks=RH.clocks(last_combat=95.0)))
    assert s.move_speed.tolist() == [0.0, 8.0]
    assert s.trinket_haste.tolist() == [12.0, 0.0]                    # 2 takedowns -> 2 mementos
    # Gold sequence for stacks 2..5 (multi-enemy world) and the cap.
    u6 = H.units(H.champions() + [dict(x=0, y=0, team=1, cls=D.CLASS_CHAMPION) for _ in range(5)])
    st = M.init(2, 7)
    ctx = H.ctx(now=0.0)
    ku = jnp.zeros((2, 7), bool).at[0, 1:].set(True)
    st, eff = M.on_takedown(st, pg, ctx, u6, RH.ev(ctx, 7, kills=H.kills(7, killed_units=ku)))
    assert float(eff.gold[0]) == pytest.approx(450.0)
    assert float(M._bounty_stacks(st)[0]) == 5.0


# ---- jit ----------------------------------------------------------------------

def test_on_damage_and_periodic_under_jit():
    u, pg, st = world(400.0), page(M.ELECTROCUTE, M.DARK_HARVEST, M.TASTE_OF_BLOOD), M.init(2, N)
    on_damage, periodic = jax.jit(M.on_damage), jax.jit(M.periodic)
    for t in (0.0, 1.0, 2.0):
        ctx = H.ctx(now=t, hp=500.0)
        st, eff = on_damage(st, pg, ctx, u, RH.ev(ctx, N, report=RH.report(hit(), u)))
        if t == 0.0:
            assert rune_packets(eff, M.DARK_HARVEST)[0].tolist() == pytest.approx([30.0])
            assert float(eff.heal[0]) == pytest.approx(16.0)
    ctx = H.ctx(now=2.25)
    st, eff = periodic(st, pg, ctx, u, RH.ev(ctx, N))
    assert rune_packets(eff, M.ELECTROCUTE)[0].tolist() == pytest.approx([70.0])
