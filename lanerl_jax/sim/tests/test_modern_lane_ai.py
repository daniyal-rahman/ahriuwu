import functools

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.sim import modern_damage as D
from lanerl_jax.sim import modern_lane_ai as L
from lanerl_jax.sim import modern_towers as T
from lanerl_jax.sim.modern_stats import PHYSICAL, TRUE
from lanerl_jax.sim.modern_world_types import (
    KIND_CHAMPION, KIND_INHIBITOR, KIND_MINION, KIND_NEXUS, KIND_TURRET,
    AttackLaunch, WorldUnits, damage_class, init_attack_state)

MELEE, CASTER, SIEGE, SUPER = range(4)
BLUE, RED = 0, 1


def champ(team, x, y=0., **kw):
    return dict(kind=KIND_CHAMPION, sub=0, team=team, x=x, y=y, radius=65., attack_range=125.,
                hp=600., max_hp=600., **kw)


def minion(team, x, y=0., sub=MELEE, spawn_time=100., **kw):
    s = L.minion_spawn_stats(sub, spawn_time, team)
    base = dict(kind=KIND_MINION, sub=sub, team=team, x=x, y=y, radius=float(s.radius),
                attack_range=float(s.attack_range), hp=float(s.hp), max_hp=float(s.max_hp),
                attack_damage=float(s.attack_damage), armor=float(s.armor), spawn_time=spawn_time)
    base.update(kw)
    return base


def turret(team, x, y=0., tier=0, **kw):
    hp = T.TIER_MAX_HP[tier]
    return dict(kind=KIND_TURRET, sub=tier, team=team, x=x, y=y, radius=88.4, attack_range=750.,
                hp=hp, max_hp=hp, armor=60., magic_resist=60., **kw)


def building(kind, team, x, y=0.):
    hp = 4000. if kind == KIND_INHIBITOR else 5500.
    return dict(kind=kind, sub=0, team=team, x=x, y=y, radius=200., attack_range=0., hp=hp, max_hp=hp)


def world(specs):
    defaults = dict(alive=True, targetable=True, armor=0., magic_resist=0., attack_damage=0.,
                    attack_speed=1., move_speed=350., spawn_time=0.)
    rows = [{**defaults, **s} for s in specs]
    col = lambda k, dt: jnp.asarray([r[k] for r in rows], dt)
    f, i = jnp.float32, jnp.int32
    return WorldUnits(col('kind', i), col('sub', i), col('team', i), col('alive', bool),
                      col('targetable', bool), col('x', f), col('y', f), col('radius', f), col('hp', f),
                      col('max_hp', f), col('armor', f), col('magic_resist', f), col('attack_damage', f),
                      col('attack_range', f), col('attack_speed', f), col('move_speed', f),
                      jnp.arange(1, len(rows) + 1, dtype=i), col('spawn_time', f))


def step(ai, units, now, dt=.25, cac=None, dmg=None, att=None, visible=None):
    n = units.kind.shape[0]
    z = jnp.zeros((n, n), bool)
    att = init_attack_state(n) if att is None else att
    return L.select_targets(ai, units, att, now=now, dt=dt,
                            champion_attacked_champion=z if cac is None else cac,
                            damage_events=z if dmg is None else dmg, visible=visible)


def pair(n, *ij):
    m = np.zeros((n, n), bool)
    for i, j in ij:
        m[i, j] = True
    return jnp.asarray(m)


def launch(n, attacker, target):
    on = np.zeros(n, bool); tg = np.full(n, -1, np.int32)
    on[attacker] = True; tg[attacker] = target
    return AttackLaunch(jnp.asarray(on), jnp.asarray(tg), jnp.zeros(n, bool), jnp.zeros(n, bool),
                        jnp.ones(n, jnp.int32))


def final_damage(units, packets, *, off=None, armor=None):
    n = units.kind.shape[0]
    cls = damage_class(units.kind)
    off = D.default_offense(n)._replace(unit_class=cls) if off is None else off
    dfn = D.default_defense(n)._replace(unit_class=cls, armor=units.armor if armor is None else armor,
                                         magic_resist=units.magic_resist)
    return D.premitigation_to_final(packets, off, dfn)


# ---------------------------------------------------------------- stats
def test_minion_spawn_stats_upgrade_formula_at_several_times():
    def row(kind, t, team=0):
        s = L.minion_spawn_stats(kind, t, team)
        return [float(s.max_hp), float(s.attack_damage), float(s.armor), float(s.gold), int(s.upgrade)]
    np.testing.assert_allclose(row(MELEE, 30.), [465, 11, 0, 20, 1])
    np.testing.assert_allclose(row(MELEE, 480.), [640, 14, 0, 20, 6])
    np.testing.assert_allclose(row(MELEE, 570.), [675, 17, .085, 20, 7], rtol=1e-6)
    np.testing.assert_allclose(row(MELEE, 840.), [780, 26, .85, 20, 10], rtol=1e-6)
    np.testing.assert_allclose(row(CASTER, 90.8), [284, 21, 0, 14, 1])
    np.testing.assert_allclose(row(SIEGE, 92.4), [835, 37.5, 0, 50, 1])
    np.testing.assert_allclose(row(SIEGE, 450.), [1175, 43.5, 0, 54, 5])
    np.testing.assert_allclose(row(SIEGE, 840.), [1600, 63.5, 0, 59, 10])
    np.testing.assert_allclose(row(SUPER, 450., team=1), [2000, 205, 100, 49, 5])
    np.testing.assert_allclose(row(SUPER, 450., team=0)[3], 54)
    s = L.minion_spawn_stats(jnp.arange(4), 630.)
    np.testing.assert_allclose(s.attack_range, [110, 550, 300, 170])
    np.testing.assert_allclose(s.attack_speed, [1.25, .667, 1., .85])
    np.testing.assert_allclose(s.move_speed, [375] * 4)
    np.testing.assert_allclose(s.radius, [48, 48, 65, 65])
    np.testing.assert_allclose(s.xp, [62, 31, 75, 75])
    np.testing.assert_allclose(s.windup, [.393, .47, .3, .4085], atol=1e-3)
    np.testing.assert_allclose(s.missile_speed, [0, 650, 1200, 0])
    np.testing.assert_allclose(s.acquisition_range, [750, 700, 750, 600])
    assert float(L.minion_spawn_stats(MELEE, 29.)[0]) == 430   # U=0 before the first upgrade


def test_minion_move_speed_sidelane_and_time():
    top = L.BARRACKS[BLUE, L.LANE_TOP]
    mid = L.BARRACKS[BLUE, L.LANE_MID]
    u = world([minion(BLUE, *top, spawn_time=60.), minion(BLUE, *mid, spawn_time=60.),
               champ(RED, 9000., 9000.)])
    ai, *_ = step(L.init_lane_ai(3), u, 60.)
    assert list(np.asarray(ai.lane[:2])) == [L.LANE_TOP, L.LANE_MID]
    np.testing.assert_allclose(L.minion_move_speed(ai, u, 60.)[:2], [451.8, 350.], rtol=1e-6)
    np.testing.assert_allclose(L.minion_move_speed(ai, u, 86.)[:2], [350., 350.])


# ---------------------------------------------------------------- minion targeting
def test_priority_and_acquisition_ranges():
    # Closest enemy minion (P5) beats a nearer enemy champion (P6).
    u = world([minion(BLUE, 0.), minion(RED, 700.), champ(RED, 300.)])
    _, desired, _, _ = step(L.init_lane_ai(3), u, 100.)
    assert int(desired[0]) == 1
    # Melee acquisition range is 750 (strict <): a minion at 760 is not a
    # candidate, so the champion is taken.
    u = world([minion(BLUE, 0.), minion(RED, 760.), champ(RED, 300.)])
    _, desired, _, _ = step(L.init_lane_ai(3), u, 100.)
    assert int(desired[0]) == 2
    # Caster acquisition range is 700.
    u = world([minion(BLUE, 0., sub=CASTER), minion(RED, 720.)])
    _, desired, _, _ = step(L.init_lane_ai(2), u, 100.)
    assert int(desired[0]) == -1
    # Within P5, pure distance (no cannon > caster > melee type preference, U-5).
    u = world([minion(BLUE, 0.), minion(RED, 400., sub=SIEGE), minion(RED, 300., sub=CASTER)])
    _, desired, _, _ = step(L.init_lane_ai(3), u, 100.)
    assert int(desired[0]) == 2
    # Structures only when nothing else is in range; invisible units ignored.
    u = world([minion(BLUE, 0.), turret(RED, 500.), champ(RED, 200.)])
    vis = jnp.asarray([[True, True, False], [True, True, True]])
    _, desired, _, _ = step(L.init_lane_ai(3), u, 100., visible=vis)
    assert int(desired[0]) == 1
    _, desired, _, _ = step(L.init_lane_ai(3), u, 100.)
    assert int(desired[0]) == 2
    # Champions and structures get no lane-AI target.
    assert int(desired[2]) == -1


def test_hysteresis_equal_priority_never_steals():
    u = world([minion(BLUE, 0.), minion(RED, 300.), minion(RED, 200., alive=False)])
    ai, desired, _, _ = step(L.init_lane_ai(3), u, 100.)
    assert int(desired[0]) == 1
    u = u._replace(alive=jnp.ones(3, bool))       # a closer P5 candidate appears
    for k in range(4):
        ai, desired, _, _ = step(ai, u, 100.25 + .25 * k)
        assert int(desired[0]) == 1


def test_call_for_help_switch_and_26_10_removal():
    # 0 blue melee (listener), 1 red minion (held, P5), 2 blue champion
    # (victim, 800 from the listener), 3 red champion (aggressor, 500 away).
    specs = [minion(BLUE, 0.), minion(RED, 300.), champ(BLUE, 0., 800.), champ(RED, 500.)]
    u = world(specs)
    ai, desired, _, _ = step(L.init_lane_ai(4), u, 100.)
    assert int(desired[0]) == 1 and int(ai.target_priority[0]) == 5
    held = ai
    # Enemy champion attacks the allied champion: switch to it as P1.
    ai, desired, _, _ = step(held, u, 100.05, dt=.05, cac=pair(4, (3, 2)))
    assert int(desired[0]) == 3 and int(ai.target_priority[0]) == 1
    # 26.10: an enemy champion damaging an allied MINION does not aggro.
    ai, desired, _, _ = step(held, u, 100.05, dt=.05, dmg=pair(4, (3, 0)))
    assert int(desired[0]) == 1
    # Victim beyond 1000 of the listener: no Call for Help.
    far = world([minion(BLUE, 0.), minion(RED, 300.), champ(BLUE, 0., 1100.), champ(RED, 500.)])
    _, desired, _, _ = step(held, far, 100.05, dt=.05, cac=pair(4, (3, 2)))
    assert int(desired[0]) == 1
    # Mid-windup: finish the windup first; the 2 s memory switches next tick.
    att = init_attack_state(4)._replace(target=jnp.asarray([1, -1, -1, -1], jnp.int32),
                                        windup_left=jnp.asarray([.2, 0, 0, 0], jnp.float32))
    ai, desired, _, _ = step(held, u, 100.05, dt=.05, cac=pair(4, (3, 2)), att=att)
    assert int(desired[0]) == 1
    ai, desired, _, _ = step(ai, u, 100.10, dt=.05)
    assert int(desired[0]) == 3
    # ...but the memory expires after 2 s.
    _, desired, _, _ = step(held._replace(last_attack=held.last_attack.at[3, 2].set(98.)), u, 100.1, dt=.05)
    assert int(desired[0]) == 1
    # An enemy minion attacking an allied minion within 500 outranks P5/P6.
    u3 = world([minion(BLUE, 0.), minion(RED, 300.), minion(BLUE, 0., 400.), minion(RED, 450., 400.)])
    held3, desired, _, _ = step(L.init_lane_ai(4), u3, 100.)
    assert int(desired[0]) == 1
    ai, desired, _, _ = step(held3, u3, 100.05, dt=.05, dmg=pair(4, (3, 2)))
    assert int(desired[0]) == 3 and int(ai.target_priority[0]) == 3


def test_minion_holding_turret_ignores_call_for_help():
    u = world([minion(BLUE, 0.), turret(RED, 600.), champ(BLUE, 0., 300.), champ(RED, 300., 300.,
                                                                                targetable=False)])
    ai, desired, _, _ = step(L.init_lane_ai(4), u, 100.)
    assert int(desired[0]) == 1
    u = u._replace(targetable=jnp.ones(4, bool))
    _, desired, _, _ = step(ai, u, 100.05, dt=.05, cac=pair(4, (3, 2)))
    assert int(desired[0]) == 1
    # First-wave minions are exempt from that exception.
    fw = ai._replace(first_wave=ai.first_wave.at[0].set(True), engaged=ai.engaged.at[0].set(True))
    _, desired, _, _ = step(fw, u, 100.05, dt=.05, cac=pair(4, (3, 2)))
    assert int(desired[0]) == 3


def test_chase_stop_leash_and_give_up():
    u = world([minion(BLUE, 0.), champ(RED, 600.)])
    ai, desired, goal, stop = step(L.init_lane_ai(2), u, 100.)
    assert int(desired[0]) == 1 and not bool(stop[0])
    np.testing.assert_allclose(goal[0], [600., 0.])                  # chase
    near = u._replace(x=jnp.asarray([0., 220.], jnp.float32))        # 110 + 48 + 65 = 223
    _, _, goal, stop = step(ai, near, 100.05, dt=.05)
    assert bool(stop[0])
    np.testing.assert_allclose(goal[0], [0., 0.])
    # Leaving acquisition range drops the target and resumes the lane walk.
    gone = u._replace(x=jnp.asarray([0., 800.], jnp.float32))
    ai2, desired, goal, stop = step(ai, gone, 100.05, dt=.05)
    assert int(desired[0]) == -1 and not bool(stop[0])
    assert not np.allclose(goal[0], [800., 0.])
    # Give-up: 4 s holding a target without attacking -> ignore it for 0.5 s.
    t, a = 100., ai
    seen = []
    for k in range(24):
        t += .25
        a, desired, _, _ = step(a, u, t)
        seen.append(int(desired[0]))
    first_drop = seen.index(-1)
    assert 14 <= first_drop <= 16                                    # ~4 s after acquisition
    assert seen[first_drop:first_drop + 2] == [-1, -1]
    assert seen[first_drop + 2] == 1                                 # re-acquired after 0.5 s
    # Attacking (windup on the target) keeps resetting the give-up timer.
    att = init_attack_state(2)._replace(target=jnp.asarray([1, -1], jnp.int32),
                                        windup_left=jnp.asarray([.1, 0.], jnp.float32))
    a = ai
    for k in range(24):
        a, desired, _, _ = step(a, u, 100. + .25 * (k + 1), att=att)
        assert int(desired[0]) == 1


def test_lane_path_walking():
    top = L.BARRACKS[BLUE, L.LANE_TOP]
    path = L.LANE_PATHS[BLUE, L.LANE_TOP]
    u = world([minion(BLUE, *top), champ(RED, 14000., 14000.)])
    ai, desired, goal, stop = step(L.init_lane_ai(2), u, 100.)
    assert int(desired[0]) == -1 and int(ai.lane[0]) == L.LANE_TOP
    np.testing.assert_allclose(goal[0], path[0])
    at_wp = u._replace(x=u.x.at[0].set(path[0, 0] + 10.), y=u.y.at[0].set(path[0, 1]))
    ai, _, goal, _ = step(ai, at_wp, 100.25)
    np.testing.assert_allclose(goal[0], path[1])
    # Chaos minions walk the same spline reversed.
    red_top = L.BARRACKS[RED, L.LANE_TOP]
    u = world([minion(RED, *red_top), champ(BLUE, 0., 0.)])
    _, _, goal, _ = step(L.init_lane_ai(2), u, 100.)
    np.testing.assert_allclose(goal[0], L.LANE_PATHS[BLUE, L.LANE_TOP, L.LANE_PATH_LEN[L.LANE_TOP] - 1])


def test_first_wave_spreads_onto_enemy_melee():
    blue = [minion(BLUE, 0., 100. * k, spawn_time=30. + .8 * k) for k in range(3)]
    red = [minion(RED, 950., 100. * k, spawn_time=30. + .8 * k) for k in range(3)]
    u = world(blue + red)
    ai, desired, _, _ = step(L.init_lane_ai(6), u, 40.)
    assert bool(ai.first_wave[0]) and bool(ai.engaged[0])
    # 950 is beyond the normal 750 scan but inside firstAcquisitionRange 1000.
    assert list(np.asarray(desired)) == [3, 4, 5, 0, 1, 2]
    # A blue champion wanders within 600 of a red first-wave melee: ignored
    # until within wakeUpRange (450).
    u = world([minion(RED, 0., spawn_time=30.), champ(BLUE, 600.)])
    _, desired, _, _ = step(L.init_lane_ai(2), u, 40.)
    assert int(desired[0]) == -1
    u = world([minion(RED, 0., spawn_time=30.), champ(BLUE, 400.)])
    _, desired, _, _ = step(L.init_lane_ai(2), u, 40.)
    assert int(desired[0]) == 1


# ---------------------------------------------------------------- turrets
def test_turret_priority_lock_and_champion_protection():
    specs = [turret(BLUE, 0.), champ(RED, 300.), minion(RED, 600.), minion(RED, 700., sub=SIEGE),
             champ(BLUE, 1300.)]
    u = world(specs)
    ai, desired, goal, stop = step(L.init_lane_ai(5), u, 100.)
    assert int(desired[0]) == 3 and bool(stop[0])                    # cannon first
    np.testing.assert_allclose(goal[0], [0., 0.])
    # Locked on a melee minion; a cannon walking into range does not steal.
    u_m = world([turret(BLUE, 0.), champ(RED, 300.), minion(RED, 600.), minion(RED, 2000., sub=SIEGE),
                 champ(BLUE, 1300.)])
    ai, desired, _, _ = step(L.init_lane_ai(5), u_m, 100.)
    assert int(desired[0]) == 2
    ai, desired, _, _ = step(ai, u, 100.25)
    assert int(desired[0]) == 2
    # Enemy champion (in range) attacks the allied champion 1300 from the turret.
    ai2, desired, _, _ = step(ai, u, 100.5, cac=pair(5, (1, 4)))
    assert int(desired[0]) == 1 and bool(ai2.champion_aggro[0])
    # Lock persists on the champion after the attack ends.
    ai2, desired, _, _ = step(ai2, u, 100.75)
    assert int(desired[0]) == 1
    # Victim at 1450: no switch.
    far = u._replace(x=u.x.at[4].set(1450.))
    _, desired, _, _ = step(ai, far, 100.5, cac=pair(5, (1, 4)))
    assert int(desired[0]) == 2
    # Aggressor out of turret range: no switch (no memory, U1).
    out = u._replace(x=u.x.at[1].set(1000.))
    _, desired, _, _ = step(ai, out, 100.5, cac=pair(5, (1, 4)))
    assert int(desired[0]) == 2
    # Champion attacking an allied minion: no switch.
    u_b = world([turret(BLUE, 0.), champ(RED, 300.), minion(RED, 600.), minion(RED, 2000., sub=SIEGE),
                 champ(BLUE, 1300.), minion(BLUE, 200.)])
    ai_b, desired, _, _ = step(L.init_lane_ai(6), u_b, 100.)
    assert int(desired[0]) == 2
    _, desired, _, _ = step(ai_b, u_b, 100.25, dmg=pair(6, (1, 5)))
    assert int(desired[0]) == 2
    # Lock released when the target leaves range -> re-acquire by priority.
    gone = u._replace(x=u.x.at[2].set(2000.))
    _, desired, _, _ = step(ai, gone, 100.5)
    assert int(desired[0]) == 3


def test_warming_up_ramp_and_reset():
    u = world([turret(BLUE, 0.), champ(RED, 300.)])
    ai, desired, _, _ = step(L.init_lane_ai(2), u, 30.)
    assert int(desired[0]) == 1
    raws = []
    t = 30.
    for k in range(5):
        raws.append(float(L.attack_packets(u, launch(2, 0, 1), now=t, ai=ai).raw[0]))
        ai, *_ = step(ai, u, t + .4, dmg=pair(2, (0, 1)))            # impact
        t += 1.
    np.testing.assert_allclose(raws, [194, 291, 388, 485, 485])
    # 5 s after the last champion hit the ramp resets.
    np.testing.assert_allclose(L.attack_packets(u, launch(2, 0, 1), now=t + 5., ai=ai).raw[0], 194)
    # Switching between champions keeps the stacks.
    u2 = world([turret(BLUE, 0.), champ(RED, 300.), champ(RED, 400.)])
    ai2, *_ = step(L.init_lane_ai(3), u2, 30.)
    ai2, *_ = step(ai2, u2, 30.4, dmg=pair(3, (0, 1)))
    ai2, *_ = step(ai2, u2, 31.6, dmg=pair(3, (0, 1)))
    np.testing.assert_allclose(L.attack_packets(u2, launch(3, 0, 2), now=32.8, ai=ai2).raw[0], 194 * 2.)
    # Minion shots neither add stacks nor refresh the window (U10).
    u3 = world([turret(BLUE, 0.), champ(RED, 300.), minion(RED, 500.)])
    ai3, *_ = step(L.init_lane_ai(3), u3, 30.)
    ai3, *_ = step(ai3, u3, 30.4, dmg=pair(3, (0, 1)))
    ai3, *_ = step(ai3, u3, 33.0, dmg=pair(3, (0, 2)))
    assert int(ai3.warm_stacks[0]) == 1
    np.testing.assert_allclose(L.attack_packets(u3, launch(3, 0, 1), now=35.5, ai=ai3).raw[0], 194)
    # Mitigation with the turret's 30 % armor pen (TOWERS §13.2).
    u4 = world([turret(BLUE, 0.), champ(RED, 300., armor=40.)])
    ai4, *_ = step(L.init_lane_ai(2), u4, 600.)
    pk = L.attack_packets(u4, launch(2, 0, 1), now=600., ai=ai4)
    off = L.turret_offense(u4, D.default_offense(2)._replace(unit_class=damage_class(u4.kind)))
    np.testing.assert_allclose(final_damage(u4, pk, off=off)[0], 235.9375, rtol=1e-5)
    assert int(pk.dtype[0]) == PHYSICAL and bool(pk.flags[0] & D.TAG_BASIC_ATTACK)


def test_turret_percent_health_shots_vs_minions():
    u = world([turret(BLUE, 0., tier=t) for t in range(4)]
              + [minion(RED, 300., max_hp=477., hp=477.), minion(RED, 300., sub=SIEGE, max_hp=1000., hp=1000.),
                 minion(RED, 300., sub=CASTER), minion(RED, 300., sub=SUPER, max_hp=2000., hp=2000., armor=135.)])
    ai = L.init_lane_ai(8)
    n = 8

    def shot(src, dst):
        pk = L.attack_packets(u, launch(n, src, dst), now=100., ai=ai)
        return float(pk.raw[src]), int(pk.dtype[src]), float(final_damage(u, pk)[src])
    raw, dtype, fin = shot(0, 4)
    assert dtype == TRUE
    np.testing.assert_allclose([raw, fin], [214.65, 214.65], rtol=1e-5)
    np.testing.assert_allclose([shot(t, 5)[2] for t in range(4)], [140, 110, 80, 80], rtol=1e-5)
    caster_hp = float(u.max_hp[6])
    np.testing.assert_allclose(shot(0, 6)[2], .7 * caster_hp, rtol=1e-5)
    # Super: 7 % per shot regardless of its armor (README X-3) -> 15 shots.
    np.testing.assert_allclose(shot(0, 7)[2], 140., rtol=1e-5)
    pk = L.attack_packets(u, launch(n, 0, 7), now=100., ai=ai, turret_minion_shot_mitigated=True)
    off = L.turret_offense(u, D.default_offense(n)._replace(unit_class=damage_class(u.kind)))
    np.testing.assert_allclose(final_damage(u, pk, off=off, armor=u.armor.at[7].set(100.))[0], 82.3529, rtol=1e-5)


def test_minion_attack_packets_and_ratios():
    u = world([minion(BLUE, 0., sub=CASTER, spawn_time=90.8), minion(RED, 300., hp=465.),
               champ(RED, 300., armor=30.), turret(RED, 300.),
               minion(BLUE, 0., sub=SIEGE, spawn_time=92.4), minion(BLUE, 0., spawn_time=90.),
               minion(RED, 200., sub=CASTER, spawn_time=90.8)])
    n = 7
    ai = L.init_lane_ai(n)
    fin = lambda a, t, **kw: float(final_damage(u, L.attack_packets(u, launch(n, a, t), now=100., ai=ai, **kw))[a])
    np.testing.assert_allclose(fin(0, 1), 37.275, rtol=1e-5)          # 21 + 0.035 * 465
    np.testing.assert_allclose(fin(0, 2), 8.8846, rtol=1e-4)          # 21 * 0.55 * 100/130 (DMG.45)
    np.testing.assert_allclose(fin(4, 3) * 1., 37.5 * .6 * 1.4 * 100 / 160, rtol=1e-5)
    np.testing.assert_allclose(fin(5, 6), 11 + .02 * 284, rtol=1e-5)  # melee -> caster 16.68
    pk = L.attack_packets(u, launch(n, 0, 1), now=100., ai=ai)
    assert int(pk.dtype[0]) == PHYSICAL
    np.testing.assert_allclose(pk.raw[0], 21 + .035 * 465, rtol=1e-6)
    # Minion Pushing: attacker bonus joins amp, target divisor divides raw.
    np.testing.assert_allclose(fin(0, 1, pushing_bonus=jnp.full(n, .1)), 41.0025, rtol=1e-5)
    np.testing.assert_allclose(fin(0, 1, pushing_divisor=jnp.full(n, 3.)), 12.425, rtol=1e-5)
    # Champion launches are not this module's packets.
    assert not bool(L.attack_packets(u, launch(n, 2, 1), now=100., ai=ai).valid.any())


# ---------------------------------------------------------------- structures
def red_base():
    # 0 blue champ, 1 red outer, 2 red inner, 3 red inhib turret, 4 red
    # inhibitor, 5/6 red Nexus turrets, 7 red Nexus, 8 blue minion (far).
    specs = [champ(BLUE, -1500.), turret(RED, 500., tier=0), turret(RED, 3000., tier=1),
             turret(RED, 5000., tier=2), building(KIND_INHIBITOR, RED, 5500.),
             turret(RED, 6500., tier=3), turret(RED, 6600., tier=3), building(KIND_NEXUS, RED, 7000.),
             minion(BLUE, -5000.)]
    lane = jnp.asarray([-1, 2, 2, 2, 2, 1, 1, 1, -1], jnp.int32)
    u = world(specs)
    return u, L.init_towers(u, lane)


def test_vulnerability_chain_and_nexus_turrets_untargetable_at_start():
    u, tw = red_base()
    tw = L.turret_tick(tw, u, now=0., dt=.05)
    assert list(np.asarray(tw.targetable[1:8])) == [True, False, False, False, False, False, False]
    assert list(np.asarray(tw.prereq[1:5])) == [-1, 1, 2, 3]
    hp = tw.turret.hp
    for k, expected in [(1, [False, True, False, False, False, False, False]),
                        (2, [False, False, True, False, False, False, False]),
                        (3, [False, False, False, True, False, False, False]),
                        (4, [False, False, False, False, True, True, False])]:
        tw, ev = L.structure_damage_events(tw, hp, hp.at[k].set(0.), now=1000.)
        hp = tw.turret.hp
        assert bool(ev.destroyed[k])
        assert list(np.asarray(tw.targetable[1:8])) == expected
    tw, _ = L.structure_damage_events(tw, hp, hp.at[5].set(0.).at[6].set(0.), now=1000.)
    assert bool(tw.targetable[7])
    # Untargetable structures are invulnerable; the inhibitor respawns at 300 s
    # and the Nexus turrets (respawning at 180 s, 40 % HP) lock again.
    _, inv = L.structure_defense_mods(tw, u, now=1000.)
    assert not bool(inv[7]) and bool(inv[1])
    tw2 = L.turret_tick(tw, u, now=1180., dt=.05)
    np.testing.assert_allclose(tw2.turret.hp[5:7], [1400, 1400])
    assert not bool(tw2.targetable[7]) and bool(tw2.targetable[5])
    tw3 = L.turret_tick(tw2, u, now=1300., dt=.05)
    assert float(tw3.turret.hp[4]) == 4000 and not bool(tw3.targetable[5])
    hp_view, alive_view, _ = L.structure_unit_view(tw3, u)
    assert bool(alive_view[4]) and float(hp_view[4]) == 4000


def test_plates_and_turret_kill_events_with_gold():
    u, tw = red_base()
    tw = L.turret_tick(tw, u, now=100., dt=.05)
    hp = tw.turret.hp
    tw, ev = L.structure_damage_events(tw, hp, hp.at[1].set(8100.), now=100.)
    assert int(ev.plates[1]) == 1 and float(ev.plate_gold[1]) == 120 and not bool(ev.destroyed[1])
    assert int(ev.rewarded_team[1]) == BLUE
    # Bulwark from the claimed plate applies to subsequent packets.
    near = u._replace(x=u.x.at[0].set(400.))
    armor, mr = L.turret_defense(tw, near, now=110.)
    assert float(armor[1]) == 90 and float(mr[1]) == 90
    assert float(L.turret_defense(tw, near, now=120.)[0][1]) == 60
    assert float(L.turret_defense(tw, u, now=110.)[0][4]) == 20       # inhibitor 20 / 0
    # Outer at 5000 (2 plates) takes a packet down to 2600 at 11:40: plates
    # 3 and 4 at 110 g each.
    st = tw.turret._replace(hp=tw.turret.hp.at[1].set(5000.), plates=tw.turret.plates.at[1].set(2))
    tw = tw._replace(turret=st)
    tw, ev = L.structure_damage_events(tw, st.hp, st.hp.at[1].set(2600.), now=700.)
    assert int(ev.plates[1]) == 2 and float(ev.plate_gold[1]) == 220
    # Kill: 5th plate gold, 50 global, first turret 300.
    hp = tw.turret.hp
    tw, ev = L.structure_damage_events(tw, hp, hp.at[1].set(-30.), now=700.)
    assert bool(ev.destroyed[1]) and int(ev.plates[1]) == 1
    np.testing.assert_allclose([ev.plate_gold[1], ev.global_gold[1], ev.first_turret_gold[1]], [110, 50, 300])
    assert bool(tw.first_turret_taken) and float(tw.turret.hp[1]) == 0
    # A one-packet kill of a fresh inner turret: 5 plates, 600 g, 25 global, no
    # second first-turret bonus.
    hp = tw.turret.hp
    tw, ev = L.structure_damage_events(tw, hp, hp.at[2].set(0.), now=900.)
    np.testing.assert_allclose([ev.plates[2], ev.plate_gold[2], ev.global_gold[2], ev.first_turret_gold[2]],
                               [5, 600, 25, 0])
    # Inhibitor: 50 to the last hitter, no plates.
    hp = tw.turret.hp
    tw, ev = L.structure_damage_events(tw, hp, hp.at[3].set(0.), now=950.)
    hp = tw.turret.hp
    tw, ev = L.structure_damage_events(tw, hp, hp.at[4].set(0.), now=960.)
    np.testing.assert_allclose([ev.plates[4], ev.plate_gold[4], ev.last_hit_gold[4]], [0, 0, 50])
    assert float(tw.turret.respawn_at[4]) == 1260.
    hp = tw.turret.hp
    tw, ev = L.structure_damage_events(tw, hp, hp.at[5].set(0.), now=970.)
    np.testing.assert_allclose([ev.plates[5], ev.plate_gold[5], ev.global_gold[5]], [0, 0, 50])


def test_backdoor_overgrowth_appearance_and_consumption():
    u, tw = red_base()
    tw = L.turret_tick(tw, u, now=99.95, dt=.05)
    assert not bool(tw.turret.growth_active[1])
    tw = L.turret_tick(tw, u, now=100., dt=.05)
    assert bool(tw.turret.growth_active[1])
    mult, _ = L.structure_defense_mods(tw, u, now=100.)
    assert np.isclose(float(mult[1]), .2)                                     # no enemy minion near
    # Backdoor active: the crystal is not consumed.
    hit = pair(9, (0, 1))
    tw2, pk = L.overgrowth_packets(tw, u, hit, now=100., team_level=jnp.asarray([1., 1.]))
    assert not bool(pk.valid.any()) and bool(tw2.turret.growth_active[1])
    # Enemy lane minion within 1000 disables it (3 s grace after it leaves).
    with_minion = u._replace(x=u.x.at[8].set(-400.))
    tw = L.turret_tick(tw, with_minion, now=100., dt=.05)
    assert float(L.structure_defense_mods(tw, u, now=102.9)[0][1]) == 1.
    assert np.isclose(float(L.structure_defense_mods(tw, u, now=103.)[0][1]), .2)
    tw2, pk = L.overgrowth_packets(tw, u, hit, now=100., team_level=jnp.asarray([1., 1.]))
    assert bool(pk.valid[1]) and int(pk.src[1]) == 0 and int(pk.dst[1]) == 1 and int(pk.dtype[1]) == TRUE
    np.testing.assert_allclose(pk.raw[1], 180.)
    assert not bool(tw2.turret.growth_active[1]) and float(tw2.turret.growth_since[1]) == 100.
    # Fully grown at L9 (outer 9000): 882.3 (TOWERS U6 fixture).
    tw400 = L.turret_tick(tw, with_minion, now=400., dt=.05)
    _, pk = L.overgrowth_packets(tw400, u, hit, now=400., team_level=jnp.asarray([9., 1.]))
    np.testing.assert_allclose(pk.raw[1], 882.3, atol=.05)
    # Suppression: an enemy champion in range at the 90 s mark defers it.
    u2, tw = red_base()
    close = u2._replace(x=u2.x.at[0].set(700.))
    tw = L.turret_tick(tw, close, now=100., dt=.05)
    assert not bool(tw.turret.growth_active[1])
    tw = L.turret_tick(tw, u2, now=250., dt=.05)
    assert bool(tw.turret.growth_active[1])
    tw = L.turret_tick(tw, with_minion._replace(x=with_minion.x.at[0].set(0.)), now=250., dt=.05)  # crystal stays once active
    _, pk = L.overgrowth_packets(tw, u2, hit, now=250., team_level=jnp.asarray([1., 1.]))
    np.testing.assert_allclose(pk.raw[1], 223.875, rtol=1e-5)
    # Nexus turrets never grow a crystal.
    assert not bool(jnp.any(tw.turret.growth_active[5:8]))


# ---------------------------------------------------------------- jit
def test_select_targets_jits_at_world_size():
    rng = np.random.default_rng(0)
    specs = [champ(BLUE, 1000., 1000.), champ(RED, 1300., 1200.)]
    for k in range(40):
        specs.append(minion(k % 2, float(rng.uniform(500, 2000)), float(rng.uniform(500, 2000)),
                            sub=k % 3, spawn_time=60. + k))
    for k in range(24):
        specs.append(turret(k % 2, float(rng.uniform(0, 3000)), float(rng.uniform(0, 3000)), tier=k % 4))
    u = world(specs)
    n = 66
    dmg = jnp.asarray(rng.uniform(size=(n, n)) < .02)
    cac = pair(n, (1, 0))
    fn = jax.jit(functools.partial(L.select_targets, dt=.0333))
    att = init_attack_state(n)
    ai = L.init_lane_ai(n)
    eager = L.select_targets(ai, u, att, now=100., dt=.0333, champion_attacked_champion=cac, damage_events=dmg)
    out = fn(ai, u, att, now=100., champion_attacked_champion=cac, damage_events=dmg)
    np.testing.assert_array_equal(out[1], eager[1])
    np.testing.assert_allclose(out[2], eager[2])
    assert out[1].shape == (n,) and out[1].dtype == jnp.int32 and out[2].shape == (n, 2)
    assert out[2].dtype == jnp.float32 and out[3].dtype == bool
    for a, b in zip(jax.tree_util.tree_leaves(out[0]), jax.tree_util.tree_leaves(ai)):
        assert a.shape == b.shape and a.dtype == b.dtype, (a.dtype, b.dtype)
    # A second step reuses the compiled function with the new state.
    out2 = fn(out[0], u, att, now=100.0333, champion_attacked_champion=cac, damage_events=dmg)
    assert out2[1].shape == (n,)
    pk = jax.jit(lambda u, l, ai: L.attack_packets(u, l, now=100., ai=ai))(u, launch(n, 2, 3), out[0])
    assert pk.valid.shape == (n,)
    lane = jnp.zeros((n,), jnp.int32)
    tw = jax.jit(lambda u: L.turret_tick(L.init_towers(u, lane), u, now=100., dt=.0333))(u)
    assert tw.turret.hp.dtype == jnp.float32


def test_init_structures_from_world_config_layout():
    from types import SimpleNamespace
    u, tw = red_base()
    cfg = SimpleNamespace(unit_kind=u.kind, unit_sub=u.sub, unit_team=u.team, unit_x=u.x, unit_y=u.y,
                          unit_lane=jnp.asarray([-1, 2, 2, 2, 2, 1, 1, 1, -1]),
                          structure_prereq=jnp.asarray([-1, -1, 1, 2, 3, 4, 4, 5, -1]))
    s = L.init_structures(cfg)
    np.testing.assert_array_equal(s.prereq, tw.prereq)          # Nexus prereqs are team-level
    np.testing.assert_array_equal(s.targetable, tw.targetable)
    np.testing.assert_allclose(s.hp[1:8], [9000, 5000, 4750, 4000, 3500, 3500, 5500])
    np.testing.assert_allclose(s.armor[1:8], [60, 60, 60, 20, 60, 60, 20])
    np.testing.assert_allclose(s.magic_resist[1:8], [60, 60, 60, 0, 60, 60, 0])
    np.testing.assert_allclose(s.attack_damage[1:8], [182, 187, 187, 0, 165, 165, 0])
    np.testing.assert_allclose(s.attack_range[1:8], [750, 750, 750, 0, 750, 750, 0])
    np.testing.assert_allclose(s.radius[1:5], [88.4, 88.4, 88.4, 213.75], rtol=1e-6)
    np.testing.assert_allclose(s.windup[1], .1669, rtol=1e-3)
