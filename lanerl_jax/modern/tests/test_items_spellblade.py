"""Spellblade group + Phage (items.effects.spellblade)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.items import effects as E
from lanerl_jax.modern.items.effects import spellblade as S
from lanerl_jax.modern.tests import item_harness as H


def world(extra=()):
    return H.units(H.champions(x1=300.0) + list(extra))


def arm_and_hit(item, *, cast_t=0.0, hit_t=0.6, target=1, u=None, **ctxkw):
    u = world() if u is None else u
    n = u.x.shape[0]
    own = H.own([item], [])
    st = S.init(2, n)
    st, _ = S.on_cast(st, own, H.ctx(now=cast_t, **ctxkw), u, H.cast())
    st, eff = S.on_hit(st, own, H.ctx(now=hit_t, **ctxkw), u, H.attack(target=(target, 0)))
    return st, eff, own, u


def test_f8_trinity_force_proc_and_cooldown():
    st, eff, own, u = arm_and_hit(S.TRINITY, base_ad=80.0)
    assert H.packet_total(eff.packets, item=S.TRINITY, dst=1) == pytest.approx(160.0)
    p = eff.packets
    sel = np.asarray(p.valid)
    assert np.all(np.asarray(p.dtype)[sel] == D.PHYSICAL)
    assert np.all(np.asarray(p.flags)[sel] & D.PROP_LIFESTEAL)
    assert np.all(np.asarray(p.flags)[sel] & D.TAG_ON_HIT)
    assert float(st.cd_until[0]) == pytest.approx(2.1)
    # Recast at t = 1.0 cannot re-arm; the hit at 1.2 does not proc.
    st2, _ = S.on_cast(st, own, H.ctx(now=1.0, base_ad=80.0), u, H.cast())
    _, eff2 = S.on_hit(st2, own, H.ctx(now=1.2, base_ad=80.0), u, H.attack())
    assert H.packet_total(eff2.packets) == 0.0
    # At 2.1 arming works again.
    st3, _ = S.on_cast(st, own, H.ctx(now=2.1, base_ad=80.0), u, H.cast())
    _, eff3 = S.on_hit(st3, own, H.ctx(now=2.3, base_ad=80.0), u, H.attack())
    assert H.packet_total(eff3.packets) == pytest.approx(160.0)


def test_sheen_base_ad_only_and_window():
    _, eff, _, _ = arm_and_hit(S.SHEEN, base_ad=70.0, bonus_ad=100.0)
    assert H.packet_total(eff.packets, item=S.SHEEN) == pytest.approx(70.0)
    _, eff, _, _ = arm_and_hit(S.SHEEN, hit_t=10.5, base_ad=70.0)
    assert H.packet_total(eff.packets) == 0.0
    _, eff, _, _ = arm_and_hit(S.SHEEN, hit_t=9.9, base_ad=70.0)
    assert H.packet_total(eff.packets) == pytest.approx(70.0)


def test_no_arm_without_cast_or_item_and_non_holder_isolated():
    u = world()
    own = H.own([S.SHEEN], [])
    st = S.init(2, 2)
    _, eff = S.on_hit(st, own, H.ctx(now=0.5), u, H.attack())
    assert H.packet_total(eff.packets) == 0.0
    # Holder 1 (no items) casts and hits: nothing.
    st, _ = S.on_cast(st, own, H.ctx(), u, H.cast(started=(False, True)))
    _, eff = S.on_hit(st, own, H.ctx(now=0.5), u, H.attack(hit=(False, True), target=(1, 0)))
    assert H.packet_total(eff.packets) == 0.0
    assert float(st.armed_until[1]) < 0


def test_miss_does_not_consume():
    u = world()
    own = H.own([S.SHEEN], [])
    st = S.init(2, 2)
    st, _ = S.on_cast(st, own, H.ctx(), u, H.cast())
    st, eff = S.on_hit(st, own, H.ctx(now=0.3), u, H.attack(hit=(False, False)))
    assert H.packet_total(eff.packets) == 0.0
    _, eff = S.on_hit(st, own, H.ctx(now=0.5), u, H.attack())
    assert H.packet_total(eff.packets) == pytest.approx(60.0)


def test_procs_on_structures():
    u = world([dict(x=200.0, y=0.0, team=1, cls=D.CLASS_STRUCTURE, radius=88.0)])
    _, eff, _, _ = arm_and_hit(S.SHEEN, target=2, u=u, base_ad=60.0)
    assert H.packet_total(eff.packets, dst=2) == pytest.approx(60.0)


def test_iceborn_damage_and_frost_field():
    extra = [dict(x=500.0, y=0.0, team=1),      # 200 from target: inside
             dict(x=700.0, y=0.0, team=1),      # 400 from target: outside (edge 300 + 48)
             dict(x=400.0, y=0.0, team=0)]      # ally: never slowed
    u = world(extra)
    st, eff, own, _ = arm_and_hit(S.ICEBORN, u=u, base_ad=100.0)
    assert H.packet_total(eff.packets, item=S.ICEBORN) == pytest.approx(150.0)
    slow = np.asarray(eff.slow)
    assert slow[1] == pytest.approx(0.25) and slow[2] == pytest.approx(0.25)
    assert slow[3] == 0.0 and slow[4] == 0.0 and slow[0] == 0.0
    assert np.asarray(eff.slow_duration)[1] > 0.0
    # Field persists 2 s at the target's position; re-applied each tick.
    _, e1 = S.periodic(st, own, H.ctx(now=1.5), u)
    assert np.asarray(e1.slow)[2] == pytest.approx(0.25)
    _, e2 = S.periodic(st, own, H.ctx(now=2.7), u)
    assert float(jnp.max(e2.slow)) == 0.0
    # Ranged holder: 12.5%.
    _, eff, _, _ = arm_and_hit(S.ICEBORN, u=u, ranged=True)
    assert np.asarray(eff.slow)[1] == pytest.approx(0.125)


def test_lich_bane_magic_and_attack_speed():
    u = world()
    own = H.own([S.LICH_BANE], [])
    st = S.init(2, 2)
    c = H.ctx(base_ad=60.0, ap=200.0)
    assert float(S.stats(st, own, c).attack_speed[0]) == 0.0
    st, _ = S.on_cast(st, own, c, u, H.cast())
    asp = S.stats(st, own, c).attack_speed
    assert float(asp[0]) == pytest.approx(0.5) and float(asp[1]) == 0.0
    st, eff = S.on_hit(st, own, H.ctx(now=0.4, base_ad=60.0, ap=200.0), u, H.attack())
    assert H.packet_total(eff.packets, item=S.LICH_BANE) == pytest.approx(0.75 * 60 + 0.45 * 200)
    assert int(np.asarray(eff.packets.dtype)[np.asarray(eff.packets.valid)][0]) == D.MAGIC
    assert float(S.stats(st, own, H.ctx(now=0.4)).attack_speed[0]) == 0.0


def test_essence_reaver_damage_mana_no_lifesteal():
    _, eff, _, _ = arm_and_hit(S.ESSENCE_REAVER, base_ad=80.0, crit_chance=0.25)
    dmg = 1.25 * 80 + 50 * 0.25
    assert H.packet_total(eff.packets, item=S.ESSENCE_REAVER) == pytest.approx(dmg)
    assert float(eff.mana[0]) == pytest.approx(0.5 * dmg) and float(eff.mana[1]) == 0.0
    flags = np.asarray(eff.packets.flags)[np.asarray(eff.packets.valid)]
    assert not np.any(flags & D.PROP_LIFESTEAL)


def test_dusk_and_dawn_heal_and_extra_on_hit():
    st, eff, own, u = arm_and_hit(S.DUSK_DAWN, base_ad=80.0, ap=100.0, base_hp=600.0, max_hp=1100.0)
    assert H.packet_total(eff.packets, item=S.DUSK_DAWN) == pytest.approx(0.75 * 80 + 0.1 * 100)
    assert float(eff.heal[0]) == pytest.approx(0.1 * 100 + 0.03 * 500, rel=1e-5)
    st1, _ = S.periodic(st, own, H.ctx(now=0.7), u)
    assert not bool(st1.dd_extra_due[0])
    st2, _ = S.periodic(st1, own, H.ctx(now=0.8), u)
    assert bool(st2.dd_extra_due[0]) and int(st2.dd_due_target[0]) == 1
    atk = S.extra_on_hit_attack(st2)
    assert bool(atk.hit[0]) and not bool(atk.launched[0]) and float(atk.raw[0]) == 0.0
    # Re-running on_hit for the extra application does not re-proc Spellblade.
    full = E.init(2, 2)._replace(spellblade=st2)
    _, e = E.on_hit(full, own, H.ctx(now=0.8, base_ad=80.0), u, atk)
    assert H.packet_total(e.packets, item=S.DUSK_DAWN) == 0.0
    st3, _ = S.periodic(st2, own, H.ctx(now=0.85), u)
    assert not bool(st3.dd_extra_due[0])


def test_bloodsong_expose_weakness_and_gold():
    st, eff, own, u = arm_and_hit(S.BLOODSONG, base_ad=50.0)
    assert H.packet_total(eff.packets, item=S.BLOODSONG) == pytest.approx(50.0)
    d = S.debuffs(st, own, H.ctx(now=3.0), u)
    assert float(d.received_amp[1]) == pytest.approx(0.08) and float(d.received_amp[0]) == 0.0
    assert float(S.debuffs(st, own, H.ctx(now=4.7), u).received_amp[1]) == 0.0
    st, _, _, _ = arm_and_hit(S.BLOODSONG, ranged=True)
    assert float(S.debuffs(st, own, H.ctx(now=1.0), u).received_amp[1]) == pytest.approx(0.05)
    # Minion target: no debuff.
    um = world([dict(x=200.0, y=0.0, team=1)])
    st, _, _, _ = arm_and_hit(S.BLOODSONG, target=2, u=um)
    assert float(jnp.max(S.debuffs(st, own, H.ctx(now=1.0), um).received_amp)) == 0.0
    # 9 gold per 10 s, dt-agnostic.
    _, e = S.periodic(S.init(2, 2), own, H.ctx(dt=0.5), u)
    assert float(e.gold[0]) == pytest.approx(0.45) and float(e.gold[1]) == 0.0


def test_trinity_quicken_and_phage_rage():
    u = world()
    for item, ranged, ms in ((S.TRINITY, False, 20.0), (S.PHAGE, False, 20.0), (S.PHAGE, True, 10.0)):
        own = H.own([item], [])
        st, _ = S.on_hit(S.init(2, 2), own, H.ctx(now=1.0, ranged=ranged), u, H.attack())
        assert float(S.stats(st, own, H.ctx(now=2.9, ranged=ranged)).move_speed[0]) == pytest.approx(ms)
        assert float(S.stats(st, own, H.ctx(now=3.1, ranged=ranged)).move_speed[0]) == 0.0
        assert float(S.stats(st, own, H.ctx(now=2.0)).move_speed[1]) == 0.0


def test_coverage_lists_all_assigned():
    assert set(S.COVERAGE) == {3057, 3078, 6662, 3100, 3508, 2510, 3877, 3044}
    assert not set(S.COVERAGE) & (E.STATS_ONLY | set(E.DEFERRED))


def test_jit_dispatch():
    u = world()
    own = H.own([S.TRINITY], [])
    state = E.init(2, 2)
    cast_j = jax.jit(E.on_cast)
    hit_j = jax.jit(E.on_hit)
    state, _ = cast_j(state, own, H.ctx(base_ad=80.0), u, H.cast())
    state, eff = hit_j(state, own, H.ctx(now=0.6, base_ad=80.0), u, H.attack())
    assert H.packet_total(eff.packets, item=S.TRINITY) == pytest.approx(160.0)
    stats = jax.jit(E.dynamic_stats)(state, own, H.ctx(now=1.0))
    assert float(stats.move_speed[0]) == pytest.approx(20.0)
    jax.jit(E.periodic)(state, own, H.ctx(now=1.0), u)
    jax.jit(E.target_debuffs)(state, own, H.ctx(now=1.0), u)
