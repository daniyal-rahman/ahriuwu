"""Precision tree rune kernels (RUNES.md §3, fixtures §13)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core import damage as D
from lanerl_jax.modern.runes.effects import precision as PR
from lanerl_jax.modern.tests import item_harness as H
from lanerl_jax.modern.tests import rune_harness as R
from lanerl_jax.modern.tests.rune_harness import page, world

approx = lambda v: pytest.approx(v, rel=1e-5, abs=1e-4)


def at(c, t):
    return c._replace(now=jnp.float32(t))


def step_damage(st, pg, c, u, p, **ev):
    return PR.on_damage(st, pg, c, u, R.ev(c, u.x.shape[0], report=R.report(p, u), **ev))


def test_coverage_lists_the_tree():
    assert set(PR.COVERAGE) == {8005, 8008, 8021, 8010, 9101, 9111, 8009, 9104, 9105, 9103, 8014, 8017, 8299}


# ---- Conqueror ----------------------------------------------------------------

@pytest.mark.parametrize("level,per,ad", [(1, 1.8, 12.96), (6, 2.4471, 17.6188), (9, 2.8353, 20.4141),
                                          (13, 3.3529, 24.1412), (18, 4.0, 28.8)])
def test_f4_conqueror_af_per_stack(level, per, ad):
    c = H.ctx(level=level)
    st = PR.init(2, 2)._replace(conq_stacks=jnp.full((2,), 12.0), conq_expire=jnp.full((2,), 1e9),
                                conq_level=jnp.full((2,), float(level)))
    s = PR.stats(st, page(PR.CONQUEROR), c, R.ev(c, 2))
    assert float(s.adaptive_force[0]) / 12 == pytest.approx(per, abs=1e-4)
    assert float(s.adaptive_force[0]) * 0.6 == approx(ad)
    assert float(s.adaptive_force[1]) == 0.0


def test_f5_conqueror_melee_autos_heal_and_expiry():
    u, pg = world(), page(PR.CONQUEROR)
    st = PR.init(2, 2)
    for k in range(6):
        c = at(H.ctx(), float(k))
        st, eff = step_damage(st, pg, c, u, R.hit_packet(0, 1, 100.0))
        assert float(st.conq_stacks[0]) == 2 * (k + 1)
        assert float(eff.heal[0]) == approx(8.0 if k == 5 else 0.0)
    # Holder 1 hitting holder 0 without the rune gets nothing.
    st2, eff = step_damage(st, pg, at(H.ctx(), 5.0), u, R.hit_packet(1, 0, 100.0))
    assert float(st2.conq_stacks[1]) == 0.0 and float(eff.heal[1]) == 0.0
    # Still active just before 5 s after the last hit, gone after.
    c = at(H.ctx(), 9.9)
    assert float(PR.stats(st, pg, c, R.ev(c, 2)).adaptive_force[0]) > 0
    c = at(H.ctx(), 10.0)
    st, _ = PR.periodic(st, pg, c, u, R.ev(c, 2))
    assert float(st.conq_stacks[0]) == 0.0
    assert float(PR.stats(st, pg, c, R.ev(c, 2)).adaptive_force[0]) == 0.0


def test_conqueror_ranged_basic_gives_one_and_proc_gives_none():
    u, pg = world(), page(PR.CONQUEROR)
    c = H.ctx(ranged=True)
    st, _ = step_damage(PR.init(2, 2), pg, c, u, R.hit_packet(0, 1, 100.0))
    assert float(st.conq_stacks[0]) == 1.0
    st, _ = step_damage(st, pg, c, u, R.hit_packet(0, 1, 50.0, D.MAGIC, D.TAG_PROC, item=-8005))
    assert float(st.conq_stacks[0]) == 1.0
    # Minion damage does not stack.
    u3 = world([dict(x=100.0, y=0.0, team=1, cls=D.CLASS_MINION)])
    st, _ = step_damage(st, pg, c, u3, R.hit_packet(0, 2, 50.0))
    assert float(st.conq_stacks[0]) == 1.0


def test_f6_special_cased_dot_stacks_every_tick():
    """Garen E-like ticks each their own instance (cast_id 0)."""
    u, pg = world(), page(PR.CONQUEROR)
    st = PR.init(2, 2)
    for k in range(8):
        c = at(H.ctx(), 0.5 * k)
        st, _ = step_damage(st, pg, c, u, R.hit_packet(0, 1, 10.0, D.PHYSICAL, D.TAG_PERIODIC))
        assert float(st.conq_stacks[0]) == min(2 * (k + 1), 12)


def test_f7_same_cast_instance_restacks_after_4s():
    u, pg = world(), page(PR.CONQUEROR)
    st = PR.init(2, 2)
    seen = []
    for k in range(12):                                 # one instance, a tick every 0.5 s for 6 s
        c = at(H.ctx(), 0.5 * k)
        st, _ = step_damage(st, pg, c, u, R.hit_packet(0, 1, 10.0, D.MAGIC, D.TAG_PERIODIC, cast_id=7))
        seen.append(float(st.conq_stacks[0]))
    assert seen[0] == 2.0 and seen[7] == 2.0 and seen[8] == 4.0 and seen[-1] == 4.0
    # Two packets of one instance in one pass count once.
    p = D.concat_packets(R.hit_packet(0, 1, 10.0, D.MAGIC, D.TAG_AOE, cast_id=9),
                         R.hit_packet(0, 1, 10.0, D.MAGIC, D.TAG_AOE, cast_id=9))
    st, _ = step_damage(st, pg, at(H.ctx(), 6.0), u, p)
    assert float(st.conq_stacks[0]) == 6.0


def test_conqueror_level_lock_u03():
    u, pg = world(), page(PR.CONQUEROR)
    st, _ = step_damage(PR.init(2, 2), pg, H.ctx(level=5), u, R.hit_packet(0, 1, 10.0))
    c = H.ctx(level=6)
    st, _ = step_damage(st, pg, c, u, R.hit_packet(0, 1, 10.0))
    af = float(PR.stats(st, pg, c, R.ev(c, 2)).adaptive_force[0])
    assert af == approx(4 * (1.8 + 2.2 * 4 / 17))


# ---- Press the Attack ---------------------------------------------------------

@pytest.mark.parametrize("level,proc", [(1, 40.0), (9, 96.4706), (18, 160.0)])
def test_f8_pta_proc_and_amp(level, proc):
    u, pg = world(), page(PR.PTA)
    st = PR.init(2, 2)
    atk = H.attack(hit=(True, True), target=(1, 0))
    for k in range(3):
        c = at(H.ctx(level=level), float(k))
        st, eff = PR.on_hit(st, pg, c, u, R.ev(c, 2, attack=atk))
        assert R.total(eff.packets, rune=PR.PTA, src=0) == approx(proc if k == 2 else 0.0)
        assert R.total(eff.packets, rune=PR.PTA, src=1) == 0.0          # holder 1 has no PTA
    hit = R.hit_packet(0, 1, 100.0)
    # Same tick as the gain: unamped. Later: +8% until 5 s after champion combat.
    clocks = R.clocks(last_champion_combat=2.0)
    assert float(PR.packet_amp(st, pg, c, u, R.ev(c, 2, clocks=clocks), hit)[0]) == 0.0
    c = at(c, 3.0)
    assert float(PR.packet_amp(st, pg, c, u, R.ev(c, 2, clocks=clocks), hit)[0]) == approx(0.08)
    true_hit = R.hit_packet(0, 1, 100.0, D.TRUE, D.TAG_ACTIVE_SPELL)
    assert float(PR.packet_amp(st, pg, c, u, R.ev(c, 2, clocks=clocks), true_hit)[0]) == approx(0.08)
    c = at(c, 7.0)
    st, _ = PR.periodic(st, pg, c, u, R.ev(c, 2, clocks=clocks))
    assert float(PR.packet_amp(st, pg, c, u, R.ev(c, 2, clocks=clocks), hit)[0]) == 0.0
    # Cooldown 6 s from consumption: no stacks at t=7.9, stacks from t=8.
    st, _ = PR.on_hit(st, pg, at(c, 7.9), u, R.ev(at(c, 7.9), 2, attack=atk))
    assert float(st.pta_stacks[0]) == 0.0
    st, _ = PR.on_hit(st, pg, at(c, 8.0), u, R.ev(at(c, 8.0), 2, attack=atk))
    assert float(st.pta_stacks[0]) == 1.0


def test_f9_pta_switching_target_clears_stacks():
    u = world([dict(x=200.0, y=100.0, team=1, cls=D.CLASS_CHAMPION)])
    pg, st = page(PR.PTA), PR.init(2, 3)
    for k, tgt in enumerate((1, 1, 2)):
        c = at(H.ctx(), float(k))
        st, eff = PR.on_hit(st, pg, c, u, R.ev(c, 3, attack=H.attack(target=(tgt, 0))))
        assert R.total(eff.packets, rune=PR.PTA) == 0.0
    assert int(st.pta_target[0]) == 2 and float(st.pta_stacks[0]) == 1.0
    # Stacks expire 4 s after the last one.
    c = at(H.ctx(), 6.0)
    st, _ = PR.periodic(st, pg, c, u, R.ev(c, 3))
    assert float(st.pta_stacks[0]) == 0.0


# ---- Lethal Tempo -------------------------------------------------------------

def test_f10_lethal_tempo_stacks_bolt_and_decay():
    u, pg, st = world(), page(PR.LETHAL_TEMPO), PR.init(2, 2)
    atk = H.attack(hit=(True, True), target=(1, 0))
    for k in range(6):
        c = at(H.ctx(bonus_as=0.10), 0.5 * k)
        bonus = 0.10 + 0.06 * min(k, 6)
        ev = R.ev(c, 2, attack=atk, bonus_attack_speed=jnp.asarray([bonus, 0.1], jnp.float32))
        st, _ = PR.on_attack(st, pg, c, u, ev)
        st, eff = PR.on_hit(st, pg, c, u, ev)
    s = PR.stats(st, pg, c, R.ev(c, 2))
    assert float(s.attack_speed[0]) == approx(0.36) and float(s.attack_speed[1]) == 0.0
    # Bolt with +10% shard and LT's own 36%: 9 * 1.46.
    c = at(c, 3.0)
    ev = R.ev(c, 2, attack=atk, bonus_attack_speed=jnp.asarray([0.46, 0.1], jnp.float32))
    st, _ = PR.on_attack(st, pg, c, u, ev)
    st, eff = PR.on_hit(st, pg, c, u, ev)
    assert R.total(eff.packets, rune=PR.LETHAL_TEMPO, src=0) == approx(13.14)
    assert R.total(eff.packets, rune=PR.LETHAL_TEMPO, src=1) == 0.0
    # Decay: expiry at 9.0 drops 1 now, then 1 per 0.3 s.
    as_at = lambda t: float(PR.stats(st, pg, at(c, t), R.ev(at(c, t), 2)).attack_speed[0])
    assert as_at(8.99) == approx(0.36)
    assert as_at(9.0) == approx(0.30)
    assert as_at(9.29) == approx(0.30)
    assert as_at(9.31) == approx(0.24)
    assert as_at(20.0) == 0.0


def test_lethal_tempo_ranged_values():
    u, pg = world(), page(PR.LETHAL_TEMPO)
    c = H.ctx(ranged=True)
    st = PR.init(2, 2)._replace(lt_stacks=jnp.full((2,), 5.0), lt_expire=jnp.full((2,), 6.0))
    s = PR.stats(st, pg, c, R.ev(c, 2))
    assert float(s.attack_speed[0]) == approx(5 * 0.048)
    ev = R.ev(c, 2, attack=H.attack(target=(1, 0)), bonus_attack_speed=jnp.zeros(2, jnp.float32))
    st, _ = PR.on_attack(st, pg, c, u, ev)
    st, eff = PR.on_hit(st, pg, c, u, ev)
    assert R.total(eff.packets, rune=PR.LETHAL_TEMPO) == approx(9.0 * 0.667)


# ---- Fleet Footwork -----------------------------------------------------------

@pytest.mark.parametrize("level,heal", [(1, 15.0), (6, 48.691), (9, 72.488), (13, 108.397), (18, 160.0)])
def test_f11_fleet_heal(level, heal):
    u = world([dict(x=100.0, y=0.0, team=1, cls=D.CLASS_MINION)])
    pg = page(PR.FLEET)
    c = H.ctx(level=level)
    armed = PR.init(2, 3)._replace(fleet_armed=jnp.asarray([True, True]), fleet_energy=jnp.full((2,), 100.0))
    st, eff = PR.on_hit(armed, pg, c, u, R.ev(c, 3, attack=H.attack(hit=(True, True), target=(1, 0))))
    assert float(eff.heal[0]) == pytest.approx(heal, abs=1e-3)
    assert float(eff.heal[1]) == 0.0
    assert float(st.fleet_energy[0]) == 0.0
    assert float(PR.stats(st, pg, c, R.ev(c, 3)).percent_move_speed[0]) == approx(0.2)
    _, eff = PR.on_hit(armed, pg, c, u, R.ev(c, 3, attack=H.attack(target=(2, 0))))
    assert float(eff.heal[0]) == pytest.approx(heal * 0.15, abs=1e-3)


def test_f12_fleet_energy_from_walking_and_attacks():
    u, pg = world(), page(PR.FLEET)
    c = H.ctx(moved=2400.0)
    st, _ = PR.periodic(PR.init(2, 2), pg, c, u, R.ev(c, 2))
    assert float(st.fleet_energy[0]) == approx(100.0) and float(st.fleet_energy[1]) == 0.0
    st = PR.init(2, 2)
    c = H.ctx()
    launch = R.ev(c, 2, attack=H.attack(hit=(False, False), launched=(True, False), target=(1, 0)))
    for _ in range(17):
        st, _ = PR.on_attack(st, pg, c, u, launch)
        assert not bool(st.fleet_armed[0])
    assert float(st.fleet_energy[0]) == 100.0
    st, _ = PR.on_attack(st, pg, c, u, launch)
    assert bool(st.fleet_armed[0])


# ---- Row 1 --------------------------------------------------------------------

@pytest.mark.parametrize("level,heal", [(1, 1.0), (5, 2.0), (6, 3.0), (10, 7.0), (11, 9.0), (18, 23.0), (20, 27.0)])
def test_f33_absorb_life(level, heal):
    u, pg = world(), page(PR.ABSORB_LIFE)
    c = H.ctx(level=level)
    _, eff = PR.on_takedown(PR.init(2, 2), pg, c, u, R.ev(c, 2, kills=H.kills(2, minion_kill=(1, 1))))
    assert float(eff.heal[0]) == approx(heal) and float(eff.heal[1]) == 0.0


def test_f20_triumph_delayed_heal_and_gold():
    u, pg = world(), page(PR.TRIUMPH)
    c = at(H.ctx(base_hp=1000.0, hp=400.0), 10.0)
    st, eff = PR.on_takedown(PR.init(2, 2), pg, c, u, R.ev(c, 2, kills=H.kills(2, champion_kill=(1, 1))))
    assert float(eff.heal[0]) == 0.0
    c = at(c, 10.99)
    st, eff = PR.periodic(st, pg, c, u, R.ev(c, 2))
    assert float(eff.heal[0]) == 0.0 and float(eff.gold[0]) == 0.0
    c = at(c, 11.0)
    st, eff = PR.periodic(st, pg, c, u, R.ev(c, 2))
    assert float(eff.heal[0]) == approx(55.0) and float(eff.gold[0]) == approx(20.0)
    assert float(eff.heal[1]) == 0.0 and float(eff.gold[1]) == 0.0
    st, eff = PR.periodic(st, pg, at(c, 12.0), u, R.ev(at(c, 12.0), 2))
    assert float(eff.gold[0]) == 0.0


def test_presence_of_mind_mana_cooldown_and_takedown():
    u, pg = world(), page(PR.PRESENCE_OF_MIND)
    hit = R.hit_packet(0, 1, 50.0)
    for level, mana in ((1, 6.0), (18, 44.0)):
        _, eff = step_damage(PR.init(2, 2), pg, H.ctx(level=level), u, hit)
        assert float(eff.mana[0]) == approx(mana)
    _, eff = step_damage(PR.init(2, 2), pg, H.ctx(level=18, ranged=True), u, hit)
    assert float(eff.mana[0]) == approx(44.0 * 0.8)
    _, eff = step_damage(PR.init(2, 2), pg, H.ctx(), u, hit, uses_energy=jnp.asarray([True, False]))
    assert float(eff.mana[0]) == approx(6.0)
    st, _ = step_damage(PR.init(2, 2), pg, H.ctx(), u, hit)
    _, eff = step_damage(st, pg, at(H.ctx(), 7.9), u, hit)
    assert float(eff.mana[0]) == 0.0
    st, eff = step_damage(st, pg, at(H.ctx(), 8.0), u, hit)
    assert float(eff.mana[0]) == approx(6.0)
    # Takedown: 15% max mana after 1 s.
    c = H.ctx(max_mana=300.0)
    st, _ = PR.on_takedown(st, pg, c, u, R.ev(c, 2, kills=H.kills(2, champion_assist=(1, 0))))
    _, eff = PR.periodic(st, pg, at(c, 1.0), u, R.ev(at(c, 1.0), 2))
    assert float(eff.mana[0]) == approx(45.0)


# ---- Row 2 (Legend) -----------------------------------------------------------

def test_f21_legend_haste_and_others():
    u = world()
    c = H.ctx()
    kills = H.kills(2, minion_kill=(49, 49), champion_kill=(1, 1))
    pg = page(PR.HASTE)
    st, _ = PR.on_takedown(PR.init(2, 2), pg, c, u, R.ev(c, 2, kills=kills))
    assert float(st.legend_points[0]) == 296.0 and float(st.legend_points[1]) == 0.0
    s = PR.stats(st, pg, c, R.ev(c, 2))
    assert float(s.basic_ability_haste[0]) == approx(3.0) and float(s.basic_ability_haste[1]) == 0.0
    s = PR.stats(st, page(PR.ALACRITY), c, R.ev(c, 2))
    assert float(s.attack_speed[0]) == approx(0.03 + 2 * 0.015)
    full = PR.init(2, 2)._replace(legend_points=jnp.full((2,), 5000.0))
    s = PR.stats(full, page(PR.BLOODLINE), c, R.ev(c, 2))
    assert float(s.life_steal[0]) == approx(0.0675) and float(s.health[0]) == approx(85.0)
    assert float(PR.stats(st, page(PR.BLOODLINE), c, R.ev(c, 2)).health[0]) == 0.0
    s = PR.stats(full, page(PR.HASTE), c, R.ev(c, 2))
    assert float(s.basic_ability_haste[0]) == approx(15.0)


# ---- Row 3 (DMG.40 amps) ------------------------------------------------------

def _amp(pg, c, u, p, st=None):
    return float(PR.packet_amp(PR.init(2, u.x.shape[0]) if st is None else st, pg, c, u, R.ev(c, u.x.shape[0]), p)[0])


@pytest.mark.parametrize("own,amp", [(0.59, 0.052), (0.45, 0.08), (0.30, 0.11), (0.10, 0.11), (0.61, 0.0)])
def test_f22_last_stand(own, amp):
    c = H.ctx(base_hp=1000.0, hp=1000.0 * own)
    assert _amp(page(PR.LAST_STAND), c, world(), R.hit_packet(0, 1, 100.0)) == approx(amp)


def test_f23_f24_coup_and_cut_down():
    c, hit = H.ctx(), R.hit_packet(0, 1, 100.0)
    assert _amp(page(PR.COUP_DE_GRACE), c, world(hp=410.0), hit) == 0.0
    assert _amp(page(PR.COUP_DE_GRACE), c, world(hp=390.0), hit) == approx(0.08)
    assert _amp(page(PR.CUT_DOWN), c, world(hp=610.0), hit) == approx(0.08)
    assert _amp(page(PR.CUT_DOWN), c, world(hp=600.0), hit) == 0.0
    # Holder 1 without the rune, minion targets, non-ampable and summoner packets: no amp.
    pg, u = page(PR.CUT_DOWN), world(hp=900.0)
    assert float(PR.packet_amp(PR.init(2, 2), pg, c, u, R.ev(c, 2), R.hit_packet(1, 0, 100.0))[0]) == 0.0
    assert _amp(pg, c, u, R.hit_packet(0, 1, 100.0, D.TRUE, D.TAG_NON_AMPABLE)) == 0.0
    assert _amp(pg, c, u, R.hit_packet(0, 1, 100.0, D.TRUE, D.PROP_SUMMONER)) == 0.0
    assert _amp(pg, c, u, R.hit_packet(0, 1, 100.0, D.TRUE, D.PROP_NO_DAMAGE_MOD)) == 0.0
    u3 = world([dict(x=100.0, y=0.0, team=1, cls=D.CLASS_MINION, hp=900.0)])
    assert _amp(pg, c, u3, R.hit_packet(0, 2, 100.0)) == 0.0


def test_f25_pta_plus_coup_is_additive():
    u, c = world(hp=300.0), at(H.ctx(), 1.0)
    st = PR.init(2, 2)._replace(pta_amp=jnp.asarray([True, True]), pta_amp_since=jnp.zeros(2, jnp.float32))
    assert _amp(page(PR.PTA, PR.COUP_DE_GRACE), c, u, R.hit_packet(0, 1, 100.0), st) == approx(0.16)


# ---- jit ------------------------------------------------------------------------

def test_hooks_under_jit():
    u, pg = world(hp=300.0), page(PR.CONQUEROR, PR.COUP_DE_GRACE, PR.PRESENCE_OF_MIND)
    c = H.ctx()
    p = D.concat_packets(R.hit_packet(0, 1, 100.0), R.hit_packet(0, 1, 30.0, D.MAGIC, D.TAG_ACTIVE_SPELL, cast_id=3))
    ev = R.ev(c, 2, report=R.report(p, u))
    st, eff = jax.jit(PR.on_damage)(PR.init(2, 2), pg, c, u, ev)
    assert float(st.conq_stacks[0]) == 4.0 and float(eff.mana[0]) == approx(6.0)
    amp = jax.jit(PR.packet_amp)(st, pg, c, u, R.ev(c, 2), p)
    np.testing.assert_allclose(np.asarray(amp), [0.08, 0.08], atol=1e-6)
