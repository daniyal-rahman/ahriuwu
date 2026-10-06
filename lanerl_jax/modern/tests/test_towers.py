import json

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.modern.core.types import (KIND_CHAMPION, KIND_INHIBITOR, KIND_NEXUS, KIND_TURRET, WorldUnits,
                                          init_attack_state)
from lanerl_jax.modern.data import PATCH_DIR
from lanerl_jax.modern.lane import ai as L
from lanerl_jax.modern.lane import towers as t
from lanerl_jax.modern.map.lanes import LANE_NAMES


def exposed(now=100.):
    return t.advance(t.init_turret(), now, True, False)


def test_resistance_decay_and_independent_bulwark_stacks():
    s = t.init_turret()._replace(bulwark_until=jnp.asarray([120., 130., 0., 0.], jnp.float32))
    assert t.resistance(s, 119., 1) == 120
    assert t.resistance(s, 120., 1) == 90
    assert t.resistance(s, 130., 1) == 60
    assert t.resistance(s, 119., 5) == 160
    states = jax.vmap(t.init_turret)(jnp.arange(4))
    np.testing.assert_allclose(jax.vmap(t.resistance, in_axes=(0, None, None))(states, 1000., 1), [0, 60, 60, 60])


def test_decay_and_ad_boundaries():
    times = jnp.array([0., 29.99, 30., 90., 810., 900.])
    np.testing.assert_allclose(jax.jit(t.outer_attack_damage)(times), [182, 182, 194, 206, 350, 350])
    np.testing.assert_allclose(t.plate_value(jnp.array([659.9, 660., 720., 780., 840., 1000.])),
                               [120, 110, 100, 90, 80, 80])
    np.testing.assert_allclose(jax.vmap(t.attack_damage, in_axes=(0, None))(jnp.arange(4), 180.), [218, 203, 203, 181])


def test_backdoor_grace_refresh():
    s = t.advance(t.init_turret(), 10., True, True)
    assert s.backdoor_until == 13.
    assert t.advance(s, 12., False, True).backdoor_until == 13.
    assert t.advance(s, 13., True, True).backdoor_until == 16.


def test_growth_suppression_and_regrowth_after_consumption():
    s = t.advance(t.init_turret(), 100., False, True)
    assert not s.growth_active
    s = t.advance(s, 400., True, False)
    assert s.growth_active
    assert t.advance(s, 401., False, True).growth_active                  # an active crystal is not suppressed
    used = s._replace(growth_active=jnp.bool_(False), growth_since=jnp.float32(400.))
    assert not t.advance(used, 489.99, False, False).growth_active
    assert t.advance(used, 490., False, False).growth_active
    assert not t.advance(t.init_turret(t.NEXUS), 1000., True, False).growth_active


def test_locked_lane_growth_clock():
    inner = t.init_turret(t.INNER, jnp.inf)
    assert not t.advance(inner, 500., False, False).growth_active
    inner = t.unlock(inner, 500.)
    assert not t.advance(inner, 589., False, False).growth_active
    assert t.advance(inner, 590., False, False).growth_active


def test_ap_attack_uses_both_contributions():
    damage, magic = t.champion_structure_attack(60., 40., 100.)
    assert damage == 160 and magic
    assert not t.champion_structure_attack(60., 60., 100.)[1]


def test_minion_shot_fractions_and_target_lock():
    np.testing.assert_allclose(t.minion_shot_fraction(jnp.arange(4)), [.45, .7, .14, .07])
    np.testing.assert_allclose(t.minion_shot_fraction(2, jnp.arange(4)), [.14, .11, .08, .08])
    e = jnp.array([True, True, True]); d = jnp.array([100., 200., 300.])
    p = jnp.array([t.CHAMPION, t.MELEE, t.CANNON_SUPER])
    none = jnp.zeros(3, bool)
    assert t.select_target(-1, e, d, p, none) == 2
    assert t.select_target(1, e, d, p, none) == 1
    assert t.select_target(1, e, d, p, jnp.array([True, False, False])) == 0
    assert t.select_target(1, ~e, d, p, none) == -1


def test_regen_segments_and_respawns():
    states = jax.vmap(t.init_turret)(jnp.arange(4))
    np.testing.assert_allclose(states.hp, [9000, 5000, 4750, 3500])
    base = t.init_turret(t.INHIBITOR)
    for frac, cap in [(.2, .3), (.5, .75), (.9, 1.)]:
        assert t.regenerate_and_respawn(base._replace(hp=base.max_hp * frac), 500., 10000.).hp == base.max_hp * cap
    nexus = t.init_turret(t.NEXUS)
    dead = nexus._replace(hp=jnp.float32(0.), respawn_at=1000. + t.respawn_delay(nexus.tier))
    assert t.regenerate_and_respawn(dead, 1179., 1.).hp == 0
    returned = t.regenerate_and_respawn(dead, 1180., 1.)
    assert returned.hp == 1400 and jnp.isinf(returned.respawn_at)
    assert t.regenerate_and_respawn(returned, 1181., 1000.).hp == 1400   # 40 % segment cap


def test_buildings_regen_and_respawn():
    inhib, nexus = t.init_turret(t.INHIBITOR_BUILDING), t.init_turret(t.NEXUS_BUILDING)
    assert inhib.max_hp == 4000 and nexus.max_hp == 5500
    np.testing.assert_allclose(t.regenerate_and_respawn(inhib._replace(hp=jnp.float32(1000.)), 0., 10.).hp, 1150.)
    np.testing.assert_allclose(t.regenerate_and_respawn(nexus._replace(hp=jnp.float32(1000.)), 0., 10.).hp, 1200.)
    dead = inhib._replace(hp=jnp.float32(0.), respawn_at=100. + t.respawn_delay(inhib.tier))
    assert dead.respawn_at == 400.
    assert t.regenerate_and_respawn(dead, 399., 1.).hp == 0
    assert t.regenerate_and_respawn(dead, 400., 1.).hp == 4000
    assert jnp.isinf(t.respawn_delay(t.OUTER))
    np.testing.assert_allclose(1 / t.ATTACK_SPEED, 1.20048, rtol=1e-5)
    np.testing.assert_allclose(t.WINDUP_S, .1669, rtol=1e-3)


def test_overgrowth_clock_and_level_curve():
    s = exposed(100.)
    np.testing.assert_allclose([t.overgrowth_damage(s, x, .02, .033) for x in [100., 160., 280., 400.]],
                               [180, 180, 238.5, 297])
    # Client item-1524 curve (TOWERS §6.2/D1/D2): L9.5 high 0.1026; L20 extends the minimum.
    low, high = jax.jit(t.overgrowth_level_fractions)(jnp.array([1., 9.5, 18., 20.]))
    np.testing.assert_allclose(low, [.02, .054, .088, .096], rtol=1e-6)
    np.testing.assert_allclose(high, [.033, .1026, .1892, .2064], rtol=1e-5)
    np.testing.assert_allclose(t.overgrowth_damage(s, 100., *t.overgrowth_level_fractions(9.5)), 486., rtol=1e-5)


def test_overgrowth_client_curve_fixtures():
    s = t.init_turret()                    # TOWERS §13.3: outer 9000 HP; clock starts 10 -> appears 100
    for level, lo, mid, hi in [(1, 180., 238.5, 297.), (3, 252., 341.3, 430.6), (6, 360., 503.5, 646.9),
                               (9, 468., 675.2, 882.3), (12, 576., 856.4, 1136.8), (18, 792., 1247.4, 1702.8),
                               (20, 864., 1360.8, 1857.6)]:
        a, b = t.overgrowth_level_fractions(float(level))
        got = [t.overgrowth_damage(s, 100. + g, a, b) for g in (60., 180., 300.)]
        np.testing.assert_allclose(got, [lo, mid, hi], atol=.06)
    inner = t.init_turret(t.INNER, 100.)
    a, b = t.overgrowth_level_fractions(6.)
    np.testing.assert_allclose([t.overgrowth_damage(inner, 250., a, b), t.overgrowth_damage(inner, 490., a, b)],
                               [200., 359.4], atol=.06)


# --- every lane: chain, plates, Nexus turrets, inhibitor respawn, game end ----------------------------------------
_GEO = json.loads((PATCH_DIR / "geometry.json").read_text())
_TIER = {"outer": 0, "inner": 1, "inhibitor": 2, "nexus": 3}


def _map_structures():
    """Champions + all 30 structures from the client geometry."""
    from lanerl_jax.modern.world.config import INHIBITORS
    rows = [(KIND_CHAMPION, 0, 0, 500., 500., -1), (KIND_CHAMPION, 0, 1, 14000., 14000., -1)]
    for o in _GEO["turrets"]:
        rows.append((KIND_TURRET, _TIER[o["tier"]], o["team"], *o["position"], o["lane"]))
    for (team, lane), p in INHIBITORS.items():
        rows.append((KIND_INHIBITOR, 0, team, p[0], p[1], LANE_NAMES.index(lane)))
    rows += [(KIND_NEXUS, 0, 0, 1549., 1658., -1), (KIND_NEXUS, 0, 1, 13240., 13235., -1)]
    a = np.asarray(rows, np.float64)
    n = len(rows)
    z = jnp.zeros((n,), jnp.float32)
    units = WorldUnits(jnp.asarray(a[:, 0], jnp.int32), jnp.asarray(a[:, 1], jnp.int32),
                       jnp.asarray(a[:, 2], jnp.int32), jnp.ones((n,), bool), jnp.ones((n,), bool),
                       jnp.asarray(a[:, 3], jnp.float32), jnp.asarray(a[:, 4], jnp.float32), z + 88.4, z, z, z, z,
                       z, z, z, z, jnp.arange(n, dtype=jnp.int32), z)
    return units, jnp.asarray(a[:, 5], jnp.int32)


def _slot(units, lane_arr, kind, team, lane=None, tier=None):
    k, tm, sb, ln = (np.asarray(v) for v in (units.kind, units.team, units.sub, lane_arr))
    m = (k == kind) & (tm == team)
    if lane is not None:
        m &= ln == lane
    if tier is not None:
        m &= sb == tier
    return np.nonzero(m)[0].tolist()


def _kill(towers, slots, now):
    hp = towers.turret.hp
    return L.structure_damage_events(towers, hp, hp.at[jnp.asarray(slots)].set(0.0), now=now)


def test_vulnerability_chain_and_plates_in_every_lane():
    units, lane = _map_structures()
    tw = L.init_towers(units, lane)
    for team in (0, 1):
        for ln in (0, 1, 2):
            outer, = _slot(units, lane, KIND_TURRET, team, ln, 0)
            inner, = _slot(units, lane, KIND_TURRET, team, ln, 1)
            inhib_t, = _slot(units, lane, KIND_TURRET, team, ln, 2)
            inhib, = _slot(units, lane, KIND_INHIBITOR, team, ln)
            assert bool(tw.targetable[outer]) and not bool(tw.targetable[inner])
            t2, ev = _kill(tw, [outer], 300.)
            assert int(ev.plates[outer]) == 5 and float(ev.global_gold[outer]) == 50.
            assert bool(t2.targetable[inner]) and not bool(t2.targetable[inhib_t])
            hp = t2.turret.hp                                                  # 10 % of an inner: one 120 g plate
            t3, ev3 = L.structure_damage_events(t2, hp, hp.at[inner].add(-500.), now=900.)
            assert int(ev3.plates[inner]) == 1 and float(ev3.plate_gold[inner]) == 120.
            t4, _ = _kill(t3, [inner], 901.)
            assert bool(t4.targetable[inhib_t]) and not bool(t4.targetable[inhib])
            t5, ev5 = _kill(t4, [inhib_t], 902.)
            assert int(ev5.plates[inhib_t]) == 5 and bool(t5.targetable[inhib])


def test_nexus_turrets_need_an_inhibitor_down_have_no_plates_and_respawn():
    units, lane = _map_structures()
    tw = L.init_towers(units, lane)
    nt = _slot(units, lane, KIND_TURRET, 1, tier=3)
    nexus, = _slot(units, lane, KIND_NEXUS, 1)
    assert len(nt) == 2 and not any(bool(tw.targetable[i]) for i in nt)
    tw, _ = _kill(tw, [_slot(units, lane, KIND_TURRET, 1, 0, tier)[0] for tier in (0, 1, 2)], 1000.)
    inhib, = _slot(units, lane, KIND_INHIBITOR, 1, 0)
    tw, ev = _kill(tw, [inhib], 1001.)
    assert float(ev.last_hit_gold[inhib]) == 50.
    assert all(bool(tw.targetable[i]) for i in nt) and not bool(tw.targetable[nexus])
    tw, ev = _kill(tw, nt, 1002.)
    assert int(np.sum(np.asarray(ev.plates)[nt])) == 0 and float(ev.global_gold[nt[0]]) == 50.
    assert bool(tw.targetable[nexus])
    tw = L.turret_tick(tw, units, now=1301.0, dt=1 / 30)                     # inhibitor back: Nexus locks
    assert float(tw.turret.hp[inhib]) == 4000. and not bool(tw.targetable[nexus])
    tw = L.turret_tick(tw, units, now=1182.5, dt=1 / 30)                     # Nexus turrets back at 40 %
    assert float(tw.turret.hp[nt[0]]) == 1400. and not bool(tw.targetable[nt[0]])


def test_nexus_destruction_ends_the_game_for_the_other_team():
    units, lane = _map_structures()
    tw = L.init_towers(units, lane)
    assert not bool(L.game_result(tw).over) and int(L.game_result(tw).winner) == -1
    nexus, = _slot(units, lane, KIND_NEXUS, 1)
    tw, ev = _kill(tw, [nexus], 2000.)
    assert bool(ev.nexus_destroyed[nexus])
    res = L.game_result(tw)
    assert bool(res.over) and int(res.winner) == 0


def test_turrets_of_every_lane_shoot_enemy_minions_in_range():
    units, lane = _map_structures()
    n0 = units.kind.shape[0]
    outers = _slot(units, lane, KIND_TURRET, 0, tier=0) + _slot(units, lane, KIND_TURRET, 1, tier=0)
    m = len(outers)                                                          # an enemy melee 400 from each
    xs = [float(units.x[s]) + 400. for s in outers]
    ys = [float(units.y[s]) for s in outers]
    teams = [1 - int(units.team[s]) for s in outers]
    cat = lambda a, b: jnp.concatenate([jnp.asarray(a), jnp.asarray(b, jnp.asarray(a).dtype)])
    u = WorldUnits(cat(units.kind, [2] * m), cat(units.sub, [0] * m), cat(units.team, teams),
                   cat(units.alive, [True] * m), cat(units.targetable, [True] * m), cat(units.x, xs), cat(units.y, ys),
                   cat(units.radius, [48.] * m), cat(units.hp, [500.] * m), cat(units.max_hp, [500.] * m),
                   cat(units.armor, [0.] * m), cat(units.magic_resist, [0.] * m), cat(units.attack_damage, [0.] * m),
                   cat(units.attack_range, [110.] * m), cat(units.attack_speed, [1.] * m),
                   cat(units.move_speed, [350.] * m), jnp.arange(n0 + m, dtype=jnp.int32),
                   cat(units.spawn_time, [100.] * m))
    u = u._replace(attack_range=jnp.where(u.kind == KIND_TURRET, 750., u.attack_range))
    n = n0 + m
    z = jnp.zeros((n, n), bool)
    _, desired, _, _ = L.select_targets(L.init_lane_ai(n), u, init_attack_state(n), now=100., dt=1 / 30,
                                        champion_attacked_champion=z, damage_events=z)
    for k, s in enumerate(outers):
        assert int(desired[s]) == n0 + k
