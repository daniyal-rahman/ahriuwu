"""Economy/progression and Top quest against ECONOMY_PROGRESSION §18 and
ROLE_QUESTS §10 fixtures (spec-derived; the replay oracle tests check the
spec itself against real games)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern import economy as E
from lanerl_jax.modern import role_quest as Q
from lanerl_jax.modern.core import damage as D


def test_client_tables():
    e = E.econ()
    assert e["constants"]["ai_StartingGold"] == 500 and e["constants"]["mission_AmbientGoldStartTime"] == 65
    assert float(E.table("need")[20]) == 22420 and float(E.table("need")[19]) == 20340
    assert float(E.table("kill_xp")[1]) == 42 and float(E.table("death")[18]) == 52.5
    assert float(E.table("death")[20]) == 52.5 and float(E.table("base_gold")[20]) == 420   # U-E-6 clamp
    assert E._tables()["minion_xp_radius"] == 1500


def test_ambient_gold():                                                            # fixture 1 (oracle phase)
    # Payments at 65.0, 65.5, ... (replay oracle; the doc fixture assumed 65.5).
    assert 500 + float(E.ambient_payments(0.0, 120.0)) == pytest.approx(613.22)
    assert 500 + float(E.ambient_payments(0.0, 600.0)) == pytest.approx(1592.42)
    assert float(E.ambient_payments(0.0, 64.99)) == 0.0
    assert float(E.ambient_payments(64.99, 65.0)) == pytest.approx(1.02)


def test_levels():                                                                  # fixture 2
    xp = jnp.asarray([279., 280., 18359., 18360., 25000.])
    np.testing.assert_array_equal(E.level_for_xp(xp, 18), [1, 2, 17, 18, 18])
    assert int(E.level_for_xp(25000., 20)) == 20 and int(E.level_for_xp(20339., 20)) == 18
    assert float(E.decimal_level(2000.)) == pytest.approx(5.4118, abs=1e-4)
    assert int(E.skill_points(20)) == 18
    np.testing.assert_array_equal(E.max_rank(jnp.asarray([1, 3, 9, 18])), [1, 2, 5, 5])
    np.testing.assert_array_equal(E.max_rank(jnp.asarray([5, 6, 11, 16]), ultimate=True), [0, 1, 2, 3])


def deaths(**kw):
    base = dict(valid=jnp.ones((1,), bool), x=jnp.zeros(1), y=jnp.zeros(1), team=jnp.asarray([1]),
                gold=jnp.asarray([21.]), xp=jnp.asarray([62.]), level=jnp.asarray([1]),
                last_hitter=jnp.asarray([0], jnp.int32))
    base.update(kw)
    return E.MinionDeaths(**base)


def test_minion_xp_split_and_comeback():                                            # fixture 3
    cx, cy, team = jnp.asarray([0., 100.]), jnp.asarray([0., 0.]), jnp.asarray([0, 0])
    alive = jnp.asarray([True, True])
    gold, xp, _ = E.minion_rewards(deaths(), cx[:1], cy[:1], team[:1], alive[:1], jnp.asarray([1.]))
    assert float(xp[0]) == pytest.approx(62.) and float(gold[0]) == 21.
    gold, xp, _ = E.minion_rewards(deaths(), cx, cy, team, alive, jnp.asarray([1., 1.]))
    np.testing.assert_allclose(xp, [40.3, 40.3], rtol=1e-6)
    np.testing.assert_allclose(gold, [21., 0.])
    # Out of 1500 range: only the last hitter.
    far = deaths(x=jnp.asarray([0.]))
    _, xp, _ = E.minion_rewards(far, jnp.asarray([0., 1600.]), cy, team, alive, jnp.asarray([1., 1.]))
    np.testing.assert_allclose(xp, [62., 0.])
    assert float(E.minion_comeback_mult(7, 4.0)) == pytest.approx(2.2)
    assert float(E.minion_comeback_mult(6, 4.5)) == pytest.approx(1.3)
    assert float(E.minion_comeback_mult(5, 1.0)) == 1.0


def test_champion_kill_xp():                                                        # fixture 4
    one = jnp.asarray([True])
    k = lambda v, rec, n=1: float(E.kill_xp(v, float(v), jnp.asarray([rec] * n), jnp.asarray([True] * n))[0])
    assert k(6, 6.0) == pytest.approx(234.)
    assert k(6, 4.0) == pytest.approx(280.8)
    assert k(6, 8.5) == pytest.approx(163.8)
    assert k(6, 12.0) == pytest.approx(93.6)
    assert k(8, 8.0, 2) == pytest.approx(160.72, rel=1e-5)
    assert k(1, 1.0) == pytest.approx(42.)
    assert float(E.kill_xp(6, 6.0, jnp.asarray([6.0]), one, quest_flat=80.0)[0]) == pytest.approx(314.)


def test_kill_gold_bounty_cycle():                                                  # fixture 5–8
    b = E.init_bounty(3)
    # V8 victim (holder 2), killer 0, assister 1, first blood, t = 150 s.
    pay = E.champion_kill(b, 2, 8, jnp.int32(0), jnp.asarray([False, True, False]), 150.0, jnp.asarray(False),
                          jnp.asarray(True))
    # Assist: (min(160, 160) + 0.5·100)·early(150) = 210·0.8958 (FB shared, replay oracle).
    np.testing.assert_allclose(pay.gold, [420., 210 * (0.5 + 0.5 * 95 / 120), 0.], rtol=1e-5)
    paid = 320 + 210 * (0.5 + 0.5 * 95 / 120)
    assert float(pay.bounty.b[2]) == pytest.approx(-paid / 3.5, rel=1e-4)
    assert float(E.kill_gold(pay.bounty.b[2], 8)) == pytest.approx(320 - paid / 3.5, rel=1e-4)
    applied = E.bounty_apply_pending(pay.bounty, jnp.asarray([5.0, 5.0, 5.0]))
    assert float(applied.b[0]) == pytest.approx(40.) and float(applied.buf[0]) == 100.
    held = E.bounty_apply_pending(pay.bounty, jnp.asarray([4.9, 4.9, 4.9]))
    assert float(held.b[0]) == 0.0
    assert float(E.kill_gold(500., 10)) == 840.0 and float(E.kill_gold(900., 10)) == 1040.0
    v = E.bounty_on_death(E.Bounty(*(jnp.asarray([x]) for x in (900., 100., 0., 0.))), 10, 1040., jnp.asarray([True]))
    assert float(v.b[0]) == 0 and float(v.carry[0]) == 200
    assert float(E.bounty_on_respawn(v, jnp.asarray([True])).b[0]) == 200
    assert float(E.kill_gold(-280., 1)) == 50.0
    np.testing.assert_allclose(E.early_assist_factor(jnp.asarray([40., 115., 175., 300.])), [0.5, 0.75, 1., 1.])


def test_death_timer():                                                             # fixture 9 (continuous)
    t = lambda lv, s: float(E.death_time(lv, s))
    assert t(5, 540.) == pytest.approx(14.0) and t(9, 870.) == pytest.approx(28.0)   # no scaling < 15:00
    assert t(6, 1200.) == pytest.approx(16.68, rel=1e-5)                         # on 30 s marks, same as steps
    assert t(9, 1860.) == pytest.approx(31.738, rel=1e-5)
    assert t(18, 3360.) == pytest.approx(78.75, rel=1e-5)
    assert t(9, 900.) == pytest.approx(28.0)
    assert t(9, 915.) == pytest.approx(28 * (1 + 0.00425 / 2), rel=1e-6)        # half a step mid-interval


def test_fountain_and_level_up():                                                   # fixtures 10, 12
    hp, _ = E.fountain_regen(500., 2000., 0., 0., True, 0.0, 0.25)
    assert float(hp) == pytest.approx(540.)
    hp, _ = E.fountain_regen(500., 2000., 0., 0., True, 0.0, 0.5, homeguard=True)
    assert float(hp) == pytest.approx(693.6)
    hp, _ = E.level_up_sync(400., 600., 672.)
    assert float(hp) == pytest.approx(472.)


def test_structure_gold_eligibility():                                              # fixture 11
    credit = E.init_credit(3, 5)
    credit = credit._replace(last_structure_damage=credit.last_structure_damage.at[2, 4].set(91.0))
    cx, cy = jnp.asarray([0., 500., 3000.]), jnp.zeros(3)
    team, alive = jnp.asarray([0, 0, 0]), jnp.asarray([True, True, True])
    g = E.structure_gold(120., 0., 4, 0., 0., 1, credit, 100.0, cx, cy, team, alive)
    np.testing.assert_allclose(g, [40., 40., 40.])          # 9 s ago, 3000 away: still eligible
    g = E.structure_gold(120., 0., 4, 0., 0., 1, credit, 102.0, cx, cy, team, alive)
    np.testing.assert_allclose(g, [60., 60., 0.])


def test_recall_and_homeguard():
    r = E.init_recall(1)
    kw = dict(cancel_action=jnp.asarray([False]), health_damage=jnp.asarray([False]),
              disabled=jnp.asarray([False]), dead=jnp.asarray([False]))
    r, done = E.recall_step(r, 10.0, request=jnp.asarray([True]), **kw)
    r, done = E.recall_step(r, 18.0, request=jnp.asarray([False]), **kw)
    assert bool(r.channeling[0]) and not bool(done[0])          # 0.5 s cast + 8 s channel
    r, done = E.recall_step(r, 18.45, request=jnp.asarray([False]), **{**kw, "health_damage": jnp.asarray([True])})
    assert bool(r.channeling[0]) and not bool(done[0])          # last 0.1 s: damage does not interrupt
    r, done = E.recall_step(r, 18.5, request=jnp.asarray([False]), **kw)
    assert bool(done[0]) and not bool(r.channeling[0])
    e = E.init_recall(1)                                        # Empowered Recall: 0.5 + 4 s, same grace
    e, _ = E.recall_step(e, 10.0, request=jnp.asarray([True]), channel=jnp.asarray([4.0]), **kw)
    e, done = E.recall_step(e, 14.45, request=jnp.asarray([False]), channel=jnp.asarray([4.0]),
                            **{**kw, "health_damage": jnp.asarray([True])})
    assert bool(e.channeling[0]) and not bool(done[0])
    e, done = E.recall_step(e, 14.5, request=jnp.asarray([False]), channel=jnp.asarray([4.0]), **kw)
    assert bool(done[0])
    assert float(E.homeguard_bonus_ms(300., 2.0)) == pytest.approx(0.6)
    assert float(E.homeguard_bonus_ms(900., 10.0)) == pytest.approx(0.65)


def test_quest_fixtures():                                                          # ROLE_QUESTS §10
    st = Q.init_quest([Q.ROLE_TOP])
    ev = Q.no_quest_events(1)
    kw = dict(in_lane=jnp.asarray([True]), alive=jnp.asarray([True]), level=jnp.asarray([5]),
              recalled=jnp.asarray([False]))
    out = Q.quest_step(st, ev, now=865.0, dt=800.0, **kw)          # passive from 65 s in lane
    assert bool(out.completed_now[0]) and int(out.level_cap[0]) == 20
    half = st._replace(points=jnp.asarray([600.]))
    ev2 = ev._replace(minions_out=jnp.asarray([1.]))
    out = Q.quest_step(half, ev2, now=10.0, dt=0.0, **kw)
    assert float(out.state.points[0]) == pytest.approx(601.25)
    np.testing.assert_allclose(Q.unleashed_tp_cooldown(jnp.asarray([5, 9, 12]), True), [260., 220., 210.])
    assert float(Q.unleashed_tp_cooldown(12)) == 240.
    assert float(Q.tp_arrival_shield(2000., True, True)) == 700.


def test_economy_step_first_blood_jit():
    """Holder 1 dies at 150 s to holder 0 (credited by a hit this tick): first
    blood gold, kill XP, death timer, bounty deferral; jitted."""
    c, n = 2, 4
    st = E.init_economy(c, n, [Q.ROLE_TOP, Q.ROLE_TOP])
    st = st._replace(last_t=jnp.float32(149.9), level=jnp.asarray([3, 3]), xp=jnp.asarray([700., 700.]))
    hit = D.packets(jnp.ones(1, bool), 0, 1, 100.0, D.PHYSICAL, D.BASIC_ATTACK)
    from lanerl_jax.modern.tests import item_harness as H
    u = H.units(H.champions(x1=200.) + [dict(x=5000, y=0, team=0), dict(x=5000, y=0, team=1)])
    rep, _ = H.resolve(hit, u)
    z = jnp.zeros((c,), bool)
    md = E.MinionDeaths(jnp.zeros((1,), bool), jnp.zeros(1), jnp.zeros(1), jnp.asarray([1]), jnp.zeros(1),
                        jnp.zeros(1), jnp.ones(1, jnp.int32), jnp.asarray([-1], jnp.int32))
    se = E.StructureEvents(jnp.zeros((1,), bool), jnp.asarray([3]), jnp.zeros(1), jnp.zeros(1), jnp.asarray([1]),
                           jnp.zeros(1), jnp.zeros(1), jnp.zeros((1,), bool), jnp.zeros((1,), bool))
    inp = E.EconomyInputs(now=150.0, unit=jnp.asarray([0, 1]), x=jnp.asarray([0., 200.]), y=jnp.zeros(2),
                          team=jnp.asarray([0, 1]), hp=jnp.asarray([500., 0.]), max_hp=jnp.asarray([800., 800.]),
                          report=rep, cc=None, final_blow=jnp.asarray([-1, 0], jnp.int32), minion_deaths=md,
                          minion_in_lane=jnp.zeros((1,), bool), structures=se,
                          last_champion_combat=jnp.asarray([150., 150.]), in_fountain=z, in_quest_lane=~z,
                          recall_request=z, cancel_action=z, health_damage=z, disabled=z, reached_endpoint=z,
                          in_jungle=z, teleported=z)
    out = jax.jit(E.economy_step)(st, inp)
    ambient = float(E.ambient_payments(149.9, 150.0))
    assert float(out.gold_gained[0]) == pytest.approx(400. + ambient)
    assert float(out.xp_gained[0]) == pytest.approx(144.)          # V3 solo, equal levels
    assert float(out.death_duration[1]) == pytest.approx(12.0)
    assert bool(out.state.first_blood_done) and float(out.kills.champion_kill[0]) == 1
    assert float(out.state.bounty.b[0]) == 0.0                    # deferred: still in champion combat
    assert float(out.state.bounty.b[1]) < 0.0


def test_structure_damage_credit_without_a_plate_falling():
    """§7: a hit on a structure stamps ``last_structure_damage`` every tick (``is_structure``), not
    only on the tick a plate/turret falls (``valid``), so the 10 s share window can see it."""
    c, n = 2, 4
    st = E.init_economy(c, n, [Q.ROLE_TOP, Q.ROLE_TOP])
    hit = D.packets(jnp.ones(1, bool), 0, 3, 100.0, D.PHYSICAL, D.BASIC_ATTACK)
    from lanerl_jax.modern.tests import item_harness as H
    u = H.units(H.champions(x1=200.) + [dict(x=5000, y=0, team=0), dict(x=300, y=0, team=1)])
    rep, _ = H.resolve(hit, u)
    z = jnp.zeros((c,), bool)
    md = E.MinionDeaths(jnp.zeros((1,), bool), jnp.zeros(1), jnp.zeros(1), jnp.asarray([1]), jnp.zeros(1),
                        jnp.zeros(1), jnp.ones(1, jnp.int32), jnp.asarray([-1], jnp.int32))
    se = E.StructureEvents(jnp.zeros((1,), bool), jnp.asarray([3]), jnp.zeros(1), jnp.zeros(1), jnp.asarray([1]),
                           jnp.zeros(1), jnp.zeros(1), jnp.zeros((1,), bool), jnp.zeros((1,), bool))
    inp = E.EconomyInputs(now=100.0, unit=jnp.asarray([0, 1]), x=jnp.asarray([0., 200.]), y=jnp.zeros(2),
                          team=jnp.asarray([0, 1]), hp=jnp.asarray([500., 500.]), max_hp=jnp.asarray([800., 800.]),
                          report=rep, cc=None, final_blow=jnp.asarray([-1, -1], jnp.int32), minion_deaths=md,
                          minion_in_lane=jnp.zeros((1,), bool), structures=se,
                          last_champion_combat=jnp.asarray([0., 0.]), in_fountain=z, in_quest_lane=~z,
                          recall_request=z, cancel_action=z, health_damage=z, disabled=z, reached_endpoint=z,
                          in_jungle=z, teleported=z)
    plain = E.economy_step(st, inp).state.credit.last_structure_damage
    marked = E.economy_step(st, inp._replace(structures=se._replace(is_structure=jnp.ones((1,), bool))))
    assert float(plain[0, 3]) < 0.0                                # legacy: valid=False hides the structure
    assert float(marked.state.credit.last_structure_damage[0, 3]) == 100.0
