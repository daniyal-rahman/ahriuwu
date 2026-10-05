"""Mage item effects (items.effects.mage), client 16.19.8230722 values."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.items import effects as E
from lanerl_jax.modern.items.effects import mage as M
from lanerl_jax.modern.tests import item_harness as H

MON = D.CLASS_MONSTER
CH = D.CLASS_CHAMPION


def world(extra=(), x1=300.0, **kw):
    return H.units(H.champions(x1=x1, **kw) + list(extra))


def ability(dst, raw=100.0, dtype=D.MAGIC, src=0, flags=D.TAG_ACTIVE_SPELL):
    dst = jnp.atleast_1d(jnp.asarray(dst, jnp.int32))
    return D.packets(jnp.ones(dst.shape, bool), src, dst, raw, dtype, flags)


def report(p, u):
    return H.resolve(p, u)[0]


def hit(st, own, u, p, now=0.0, **ctx_kw):
    return M.on_damage(st, own, H.ctx(now=now, **ctx_kw), u, report(p, u))


def run(st, own, u, t0, t1, dt, **ctx_kw):
    """Run periodic from t0 (exclusive) to t1; return state and all packets."""
    out = []
    steps = int(round((t1 - t0) / dt))
    for i in range(1, steps + 1):
        st, eff = M.periodic(st, own, H.ctx(now=t0 + i * dt, dt=dt, **ctx_kw), u)
        out.append(eff.packets)
    return st, D.concat_packets(*out)


# ---- static/dynamic stats --------------------------------------------------------

def test_rabadon_riftmaker_roa_stats():
    st = M.init(2, 2)
    s = M.stats(st, H.own([3089], []), H.ctx(ap=200.))
    assert float(s.ability_power[0]) == pytest.approx(60.0, rel=1e-5)
    assert float(s.ability_power[1]) == 0.0
    # Riftmaker: 2% of bonus HP (1000) as AP.
    s = M.stats(st, H.own([4633], []), H.ctx(ap=100., base_hp=1000., max_hp=2000.))
    assert float(s.ability_power[0]) == pytest.approx(20.0, rel=1e-5)
    # Rabadon multiplies Riftmaker's AP too: (100 + 20) * 0.3 + 20.
    s = M.stats(st, H.own([4633, 3089], []), H.ctx(ap=100., base_hp=1000., max_hp=2000.))
    assert float(s.ability_power[0]) == pytest.approx(56.0, rel=1e-5)
    # Malignance ultimate haste.
    s = M.stats(st, H.own([3118], []), H.ctx())
    assert float(s.ultimate_haste[0]) == 20.0


def test_rod_of_ages_timeless():
    u = world()
    own = H.own([6657], [])
    st = M.init(2, 2)
    st, _ = run(st, own, u, 0.0, 650.0, 5.0)
    s = M.stats(st, own, H.ctx())
    assert float(s.ability_power[0]) == pytest.approx(30.0)
    assert float(s.health[0]) == pytest.approx(100.0)
    assert float(s.mana[0]) == pytest.approx(300.0)
    assert float(s.ability_power[1]) == 0.0
    st, _ = run(st, own, u, 650.0, 700.0, 5.0)        # capped at 10
    assert float(M.stats(st, own, H.ctx()).ability_power[0]) == pytest.approx(30.0)
    st, _ = run(st, H.own([], []), u, 700.0, 705.0, 5.0)   # sold -> reset
    assert float(st.roa_elapsed[0]) == 0.0


# ---- on-hit ------------------------------------------------------------------------

def test_nashors_on_hit():
    u = world()
    st, eff = M.on_hit(M.init(2, 2), H.own([3115], [3115]), H.ctx(ap=200.), u, H.attack())
    p = eff.packets
    assert H.packet_total(p, item=3115, src=0) == pytest.approx(45.0)
    assert H.packet_total(p, src=1) == 0.0
    assert int(p.dtype[0]) == D.MAGIC and bool(p.flags[0] & D.PROP_LIFESTEAL)


# ---- burns ---------------------------------------------------------------------------

@pytest.mark.parametrize("dt", [1 / 30, 0.5, 1.0])
def test_fated_ashes_burn_is_dt_agnostic(dt):
    u = world([dict(x=500, y=0, team=1, cls=MON)])
    own = H.own([2508], [])
    st, _ = hit(M.init(2, 3), own, u, ability([1, 2]))
    st, p = run(st, own, u, 0.0, 4.0, dt)
    assert H.packet_total(p, dst=1, item=2508) == pytest.approx(15.0, rel=1e-5)
    assert H.packet_total(p, dst=2, item=2508) == pytest.approx(15.0 + 45.0, rel=1e-5)
    assert np.all(np.asarray(p.flags)[np.asarray(p.valid)] & D.TAG_PERIODIC)


def test_burn_refresh_extends_and_basic_attacks_do_not_trigger():
    u = world()
    own = H.own([2508], [])
    st, _ = hit(M.init(2, 2), own, u, ability(1, flags=D.BASIC_ATTACK))
    assert float(st.ashes_until[0, 1]) < 0
    st, _ = hit(st, own, u, ability(1))
    st, p1 = run(st, own, u, 0.0, 1.0, 0.1)
    st, _ = hit(st, own, u, ability(1), now=1.0)       # refresh at t=1 -> ends at 4
    st, p2 = run(st, own, u, 1.0, 5.0, 0.1)
    assert H.packet_total(p1, item=2508) + H.packet_total(p2, item=2508) == pytest.approx(20.0, rel=1e-4)


def test_blackfire_burn_and_ap_stacks():
    u = world([dict(x=500, y=0, team=1), dict(x=500, y=50, team=1, cls=MON)])
    own = H.own([2503], [])
    st, _ = hit(M.init(2, 4), own, u, ability([1, 2, 3]), ap=100.)
    s = M.stats(st, own, H.ctx(ap=100.))
    assert float(s.ability_power[0]) == pytest.approx(8.0, rel=1e-5)   # champion + monster: 2 x 4%
    st, p = run(st, own, u, 0.0, 3.0, 0.25, ap=100.)
    assert H.packet_total(p, dst=1) == pytest.approx(3 * 22.0, rel=1e-5)
    assert H.packet_total(p, dst=2) == pytest.approx(3 * 22.0, rel=1e-5)   # minion: MinionDPS 20 + 2% AP
    assert H.packet_total(p, dst=3) == pytest.approx(3 * 42.0, rel=1e-5)   # monster: MonsterDPS 40 + 2% AP
    st, _ = run(st, own, u, 3.0, 3.5, 0.25, ap=100.)
    assert float(M.stats(st, own, H.ctx(now=3.5, ap=100.)).ability_power[0]) == 0.0


def test_liandry_burn_and_monster_cap():
    u = world([dict(x=500, y=0, team=1, cls=MON, max_hp=5000., hp=5000.)], max_hp=2000., hp=2000.)
    own = H.own([6653], [])
    st, _ = hit(M.init(2, 3), own, u, ability([1, 2]))
    st, p = run(st, own, u, 0.0, 3.0, 1 / 30)
    assert H.packet_total(p, dst=1, item=6653) == pytest.approx(120.0, rel=1e-4)
    assert H.packet_total(p, dst=2, item=6653) == pytest.approx(120.0, rel=1e-4)   # 40/s cap


def test_burn_stops_when_target_dies():
    rows = H.champions()
    u = world()
    own = H.own([6653], [])
    st, _ = hit(M.init(2, 2), own, u, ability(1))
    rows[1]["alive"] = False
    st, p = run(st, own, H.units(rows), 0.0, 3.0, 0.5)
    assert H.packet_total(p) == 0.0


# ---- champion-combat damage amps ----------------------------------------------------

def test_guise_liandry_riftmaker_stacks():
    u = world()
    st = M.init(2, 2)
    own = H.own([3147, 4633], [])
    for t in (0.0, 1.0, 2.0, 3.0, 4.0):
        st, _ = hit(st, own, u, ability(1, src=0, flags=D.BASIC_ATTACK), now=t)
    amp = M.dealt_amp(st, own, H.ctx(now=2.5), u)
    assert float(amp[0, 1]) == pytest.approx(0.04 + 0.04, rel=1e-5)
    amp = M.dealt_amp(st, own, H.ctx(now=4.0), u)
    assert float(amp[0, 1]) == pytest.approx(0.06 + 0.08, rel=1e-5)
    s = M.stats(st, own, H.ctx(now=4.0))
    assert float(s.omnivamp[0]) == pytest.approx(0.10, rel=1e-5)
    s = M.stats(st, own, H.ctx(now=4.0, ranged=True))
    assert float(s.omnivamp[0]) == pytest.approx(0.06, rel=1e-5)
    # Out of champion combat: Guise ends after 3 s, Riftmaker after 4 s.
    assert float(M.dealt_amp(st, own, H.ctx(now=7.5), u)[0, 1]) == pytest.approx(0.08, rel=1e-5)
    assert float(M.dealt_amp(st, own, H.ctx(now=8.5), u)[0, 1]) == 0.0
    assert float(jnp.abs(M.dealt_amp(st, own, H.ctx(now=2.5), u)[1]).sum()) == 0.0


def test_combat_does_not_start_from_minions():
    u = world([dict(x=500, y=0, team=1)])
    own = H.own([3147], [])
    st, _ = hit(M.init(2, 3), own, u, ability(2), now=0.0)
    assert float(M.dealt_amp(st, own, H.ctx(now=2.0), u)[0, 2]) == 0.0


# ---- on-damage procs ---------------------------------------------------------------

def test_luden_echo_targets_and_cooldown():
    u = world([dict(x=400, y=0, team=1), dict(x=300, y=600, team=1), dict(x=300, y=900, team=1),
               dict(x=350, y=0, team=0)])
    own = H.own([6655], [])
    st, eff = hit(M.init(2, 6), own, u, ability(1), ap=200.)
    p = eff.packets
    dmg = 75 + 0.05 * 200
    assert H.packet_targets(p, item=6655) == [1, 2, 3]
    assert H.packet_total(p, dst=2) == pytest.approx(dmg)
    # 5 echoes beyond the first, 2 found targets -> 3 repeats at 20% on the primary.
    assert H.packet_total(p, dst=1) == pytest.approx(dmg * (1 + 3 * 0.2), rel=1e-5)
    st, eff = hit(st, own, u, ability(1), now=11.9, ap=200.)
    assert H.packet_total(eff.packets) == 0.0
    st, eff = hit(st, own, u, ability(1), now=12.0, ap=200.)
    assert H.packet_total(eff.packets, dst=1) > 0


def test_luden_single_target_max_is_double():
    u = world()
    _, eff = hit(M.init(2, 2), H.own([6655], []), u, ability(1), ap=0.)
    assert H.packet_total(eff.packets) == pytest.approx(2 * 75.0, rel=1e-5)


def test_rylai_slow_and_grievous_wounds():
    u = world([dict(x=500, y=0, team=1)])
    _, eff = hit(M.init(2, 3), H.own([3116], []), u, ability([1, 2]))
    np.testing.assert_allclose(eff.slow, [0, 0.3, 0.3], rtol=1e-6)
    np.testing.assert_allclose(eff.slow_duration, [0, 1, 1])
    for item in (3165, 3916):
        _, eff = hit(M.init(2, 3), H.own([item], []), u, ability([1, 2]))
        np.testing.assert_allclose(eff.grievous, [0, 3, 0])
        _, eff = hit(M.init(2, 3), H.own([item], []), u, ability([1, 2], dtype=D.PHYSICAL))
        assert float(eff.grievous.sum()) == 0.0
    # Non-holder: holder 1 deals magic damage to holder 0 without items.
    _, eff = hit(M.init(2, 3), H.own([3165], []), u, ability(0, src=1))
    assert float(eff.grievous.sum()) == 0.0


def test_alternator_and_cosmic_drive():
    u = world()
    own = H.own([3145, 4629], [])
    st, eff = hit(M.init(2, 2), own, u, ability(1, flags=D.BASIC_ATTACK, dtype=D.PHYSICAL))
    assert H.packet_total(eff.packets, item=3145) == pytest.approx(65.0)
    assert float(st.cosmic_until[0]) < 0           # physical -> no Spelldance
    st, eff = hit(st, own, u, ability(1), now=39.0)
    assert H.packet_total(eff.packets, item=3145) == 0.0
    assert float(M.stats(st, own, H.ctx(now=42.9)).move_speed[0]) == 20.0
    assert float(M.stats(st, own, H.ctx(now=43.1)).move_speed[0]) == 0.0
    _, eff = hit(st, own, u, ability(1), now=40.0)
    assert H.packet_total(eff.packets, item=3145) == pytest.approx(65.0)


def test_shadowflame_cinderbloom():
    rows = H.champions()
    rows[1].update(hp=300., max_hp=1000.)
    u = H.units(rows)
    own = H.own([4645], [])
    _, eff = hit(M.init(2, 2), own, u, ability(1, raw=100.))
    assert H.packet_total(eff.packets, item=4645) == pytest.approx(20.0, rel=1e-5)
    _, eff = hit(M.init(2, 2), own, u, ability(1, raw=100., dtype=D.PHYSICAL))
    assert H.packet_total(eff.packets, item=4645) == 0.0
    _, eff = hit(M.init(2, 2), own, world(), ability(1, raw=100.))
    assert H.packet_total(eff.packets, item=4645) == 0.0


def test_stormsurge_squall():
    u = world()
    own = H.own([4646], [])
    st, _ = hit(M.init(2, 2), own, u, ability(1, raw=150.), now=0.0, ap=100.)
    assert int(st.storm_target[0]) == -1
    st, _ = hit(st, own, u, ability(1, raw=120.), now=1.0, ap=100.)
    assert int(st.storm_target[0]) == 1
    st, p = run(st, own, u, 1.0, 2.9, 0.1, ap=100.)
    assert H.packet_total(p) == 0.0
    st, p = run(st, own, u, 2.9, 3.2, 0.1, ap=100.)
    assert H.packet_total(p, dst=1, item=4646) == pytest.approx(135.0)
    # Window expires: 150 at t=10, 120 at t=13 -> no trigger (also on cd until 31).
    st2, _ = hit(M.init(2, 2), own, u, ability(1, raw=150.), now=10.0)
    st2, _ = hit(st2, own, u, ability(1, raw=120.), now=13.0)
    assert int(st2.storm_target[0]) == -1


def test_malignance_hatefog():
    u = world([dict(x=400, y=0, team=1)])
    own = H.own([3118], [])
    st = M.init(2, 3)
    st, _ = hit(st, own, u, ability(1, raw=100.), ap=100.)
    assert float(st.mal_until[0, 1]) < 0                  # no R cast -> no zone
    st, _ = M.on_cast(st, own, H.ctx(), u, H.cast(slot=(3, 0)))
    st, _ = hit(st, own, u, ability(1, raw=100.), ap=100.)
    assert float(st.mal_r[0, 1]) == pytest.approx(252.0, rel=1e-5)   # 250 + 2^(100/100)
    d = M.debuffs(st, own, H.ctx(now=0.1), u)
    np.testing.assert_allclose(d.flat_mr_reduction, [0, 10, 10])
    st, p = run(st, own, u, 0.0, 3.0, 1 / 30, ap=100.)
    assert H.packet_total(p, dst=1, item=3118) == pytest.approx(3 * 65.0, rel=1e-4)
    assert H.packet_total(p, dst=2, item=3118) == pytest.approx(3 * 65.0, rel=1e-4)
    assert float(M.debuffs(st, own, H.ctx(now=3.1), u).flat_mr_reduction.sum()) == 0.0


def test_bloodletters_stacks_and_icd():
    u = world()
    own = H.own([8010], [])
    st = M.init(2, 2)
    for t in (0.0, 0.1, 0.4, 0.8, 1.2, 1.6):
        st, _ = hit(st, own, u, ability(1), now=t)
    d = M.debuffs(st, own, H.ctx(now=1.7), u)
    assert float(d.percent_mr_reduction[1]) == pytest.approx(0.30, rel=1e-5)
    st2 = M.init(2, 2)
    for t in (0.0, 0.1):
        st2, _ = hit(st2, own, u, ability(1), now=t)
    assert float(M.debuffs(st2, own, H.ctx(now=0.2), u).percent_mr_reduction[1]) == pytest.approx(0.075)
    assert float(M.debuffs(st2, own, H.ctx(now=6.1), u).percent_mr_reduction[1]) == 0.0


def test_horizon_focus_mark():
    u = world([dict(x=900, y=500, team=1, cls=CH)], x1=700.)
    own = H.own([4628], [])
    st, _ = hit(M.init(2, 3), own, u, ability(1))
    amp = M.dealt_amp(st, own, H.ctx(now=1.0), u)
    np.testing.assert_allclose(amp[0], [0, 0.1, 0.1], rtol=1e-6)
    np.testing.assert_allclose(M.dealt_amp(st, own, H.ctx(now=3.5), u)[0], [0, 0.1, 0], rtol=1e-6)
    st3, _ = hit(M.init(2, 2), own, world(x1=500.), ability(1))
    assert float(M.dealt_amp(st3, own, H.ctx(now=1.0), world(x1=500.))[0, 1]) == 0.0


def test_eternity_mana():
    u = world([dict(x=100, y=0, team=1)])
    own = H.own([3803], [6657])
    p = D.concat_packets(ability(0, raw=200., src=1, dtype=D.PHYSICAL), ability(0, raw=500., src=2))
    _, eff = hit(M.init(2, 3), own, u, p)
    assert float(eff.mana[0]) == pytest.approx(20.0)      # minion damage ignored
    assert float(eff.mana[1]) == 0.0


def test_cryptbloom_takedown_heal():
    u = world()
    own = H.own([3137], [])
    st, _ = hit(M.init(2, 2), own, u, ability(1))
    ku = jnp.asarray([[False, True], [False, False]])
    st, eff = M.on_takedown(st, own, H.ctx(now=2.0, ap=100.), u, H.kills(2, killed_units=ku))
    assert float(eff.heal[0]) == pytest.approx(120.0)
    st, eff = M.on_takedown(st, own, H.ctx(now=2.5, ap=100.), u, H.kills(2, killed_units=ku))
    assert float(eff.heal[0]) == 0.0                     # cd 60
    st2, _ = hit(M.init(2, 2), own, u, ability(1))
    _, eff = M.on_takedown(st2, own, H.ctx(now=3.5), u, H.kills(2, killed_units=ku))
    assert float(eff.heal[0]) == 0.0                     # outside 3 s window


def test_lost_chapter_enlighten():
    u = world()
    own = H.own([3802], [])
    st = M.init(2, 2)
    st, eff = M.periodic(st, own, H.ctx(level=1, max_mana=300.), u)
    assert float(eff.mana.sum()) == 0.0                  # first observation is not a level-up
    total = 0.0
    for i in range(1, 121):
        st, eff = M.periodic(st, own, H.ctx(now=i / 30, level=2, max_mana=300.), u)
        total += float(eff.mana[0])
        assert float(eff.mana[1]) == 0.0
    assert total == pytest.approx(60.0, rel=1e-4)


# ---- registry / JIT -------------------------------------------------------------

def test_coverage_entries_and_dispatch_jit():
    for iid in (2503, 2508, 2522, 3089, 3115, 3116, 3118, 3137, 3145, 3146, 3147, 3152, 3165, 3802,
                3803, 3916, 4628, 4629, 4633, 4645, 4646, 6653, 6655, 6657, 8010):
        assert iid in M.COVERAGE
    u = world()
    own = H.own([6653, 2503], [6653])
    rep = report(ability(1), u)

    @jax.jit
    def step(state, now):
        ctx = H.ctx(now=now, ap=100.)
        state, e1 = E.on_damage(state, own, ctx, u, rep)
        state, e2 = E.periodic(state, own, ctx._replace(now=now + 0.5), u)
        return state, e2.packets
    st, p = step(E.init(2, 2), jnp.float32(0.0))
    assert H.packet_total(p, dst=1, item=6653) == pytest.approx(0.02 * 1000 * 0.5, rel=1e-5)
    assert H.packet_total(p, dst=1, item=2503) == pytest.approx(11.0, rel=1e-5)
    assert H.packet_total(p, src=1) == 0.0
