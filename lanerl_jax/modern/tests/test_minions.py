import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.lane.minions import (MinionType, LaneSpawnState, base_move_speed, cannon_wave, gold_bounty,
                                            init_lane_spawn, lane_spawn_step, minion_pushing_modifiers,
                                            minion_upgrade_stats, move_speed_soft_cap, sidelane_bonus_move_speed,
                                            super_count, upgrade_index_at, wave_index_at, wave_interval_s,
                                            wave_spawn_time, wave_unit_type)


def composition(i, supers=0):
    """``[supers, melee, cannon, casters]`` of wave ``i``."""
    types = [int(wave_unit_type(i, u, supers)[0]) for u in range(10)]
    return [types.count(t) for t in (MinionType.SUPER, MinionType.MELEE, MinionType.CANNON, MinionType.CASTER)]


def test_wave_schedule_including_breakpoint_gap_and_cannon_sequence():
    np.testing.assert_allclose(wave_spawn_time(jnp.asarray([0, 26, 27, 28, 65, 66, 67])),
                               [30, 810, 840, 865, 1790, 1810, 1830])
    times = jnp.asarray([29.9, 30., 839.9, 840., 1799.9, 1800., 1809.9, 1810.])
    np.testing.assert_array_equal(wave_index_at(times), [-1, 0, 26, 27, 65, 65, 65, 66])
    np.testing.assert_array_equal(wave_interval_s(jnp.asarray([839., 840., 1765., 1790.])), [30., 25., 25., 20.])
    np.testing.assert_array_equal(cannon_wave(jnp.asarray([1, 2, 26, 27, 28, 30, 52, 53, 54, 65])),
                                  [False, True, True, False, True, True, True, False, True, True])


def test_composition_and_spawn_order():
    # Index 28 (14:25): cannon wave with one fewer melee; a super replaces the cannon, melee follows the rotation.
    assert composition(28) == [0, 2, 1, 3]
    assert composition(28, 1) == [1, 2, 0, 3]
    assert composition(29, 1) == [1, 3, 0, 3]
    assert composition(54) == [0, 2, 1, 3]
    assert composition(54, 2) == [2, 2, 0, 3]
    for i, expected in [(0, [0, 3, 0, 3]), (2, [0, 3, 1, 3]), (26, [0, 3, 1, 3]), (27, [0, 3, 0, 3]),
                        (29, [0, 3, 0, 3]), (53, [0, 3, 0, 3]), (65, [0, 2, 1, 3]), (66, [0, 2, 1, 2])]:
        assert composition(i) == expected, i                              # MINIONS §10 fixtures
    assert [int(wave_unit_type(2, u, 0)[0]) for u in range(7)] == [0, 0, 0, 2, 1, 1, 1]
    assert not bool(wave_unit_type(2, 7, 0)[1])
    assert int(wave_unit_type(-1, 0, 0)[0]) == MinionType.NONE
    fn = jax.jit(jax.vmap(lambda i, u: wave_unit_type(i, u, 0)[0]))
    np.testing.assert_array_equal(fn(jnp.asarray([2, 28]), jnp.asarray([3, 2])), [2, 2])


def test_gold_xp_pushing_and_upgrade_index():
    # Siege/super gold is 49 + U (54 at U=5); Chaos supers stay at 49.
    np.testing.assert_allclose(gold_bounty(jnp.asarray([0, 1, 2, 3]), 5), [20, 14, 54, 54])
    np.testing.assert_allclose(gold_bounty(2, jnp.asarray([1, 5, 10, 41, 50])), [50, 54, 59, 90, 90])
    assert float(gold_bounty(3, 5, team=1)) == 49
    np.testing.assert_array_equal(upgrade_index_at(jnp.asarray([29.9, 30, 119.9, 120, 210])), [0, 1, 1, 2, 3])
    bonus, divisor = minion_pushing_modifiers(team_level_advantage=4., lane_turret_advantage=2., time_s=210.)
    np.testing.assert_allclose([bonus, divisor], [.45, 7.])
    fn = jax.jit(jax.vmap(wave_index_at))
    np.testing.assert_array_equal(fn(jnp.asarray([30., 840., 1810.])), [0, 27, 66])


def test_upgrade_stats_match_client_formula_fixtures():
    def stats(kind, u):                                                   # MINIONS §10 "Stats at U"
        r = minion_upgrade_stats(kind, u)
        return [float(r.max_hp), float(r.attack_damage), float(r.armor)]
    np.testing.assert_allclose(stats(0, 1), [465, 11, 0])
    np.testing.assert_allclose(stats(0, 6), [640, 14, 0])
    np.testing.assert_allclose(stats(0, 7), [675, 17, .085], rtol=1e-6)
    np.testing.assert_allclose(stats(0, 10), [780, 26, .85], rtol=1e-6)
    assert stats(0, 32)[0] == 1550 and stats(0, 33)[0] == 1550
    np.testing.assert_allclose(minion_upgrade_stats(1, jnp.asarray([5, 6, 29, 30])).attack_damage,
                               [27, 31, 123, 125])
    np.testing.assert_allclose(minion_upgrade_stats(2, jnp.asarray([5, 6, 26])).attack_damage, [43.5, 47.5, 126])
    sup = minion_upgrade_stats(3, 1)
    np.testing.assert_allclose([sup.max_hp, sup.attack_damage, sup.armor, sup.magic_resist], [1600, 185, 100, -30])
    np.testing.assert_array_equal(upgrade_index_at(jnp.asarray([450., 480., 570., 840., 1500.])), [5, 6, 7, 10, 17])


def test_move_speed_timing_and_sidelane_buff():
    np.testing.assert_allclose(base_move_speed(jnp.asarray([629., 630., 930., 1530., 2000.])),
                               [350, 375, 400, 450, 450])
    tau = jnp.asarray([0., 7., 14., 21., 25.])
    np.testing.assert_allclose(sidelane_bonus_move_speed(2, 2, 60., tau), [111, 96, 81, 66, 0])
    assert float(sidelane_bonus_move_speed(10, 0, 300., 0.)) == 75
    np.testing.assert_allclose(sidelane_bonus_move_speed(26, 2, 780., jnp.asarray([0., 7.])), [3, 0])
    assert float(sidelane_bonus_move_speed(27, 2, 810., 0.)) == 0
    assert float(sidelane_bonus_move_speed(1, 2, 30., 0.)) == 0
    assert float(sidelane_bonus_move_speed(2, 1, 60., 0.)) == 0
    np.testing.assert_allclose(move_speed_soft_cap(350. + 111.), 451.8, rtol=1e-6)


# --- all-lane spawn schedule ------------------------------------------------------------------------------------
def _cursor(wave):
    return LaneSpawnState(jnp.full((2, 3), wave, jnp.int32), jnp.zeros((2, 3), jnp.int32),
                          jnp.full((2, 3), -1, jnp.int32))


def _run(st, t, t_end, **kw):
    """Step at 30 Hz from ``t`` to ``t_end``: ``(state, [(t, team, lane, type)])``."""
    log = []
    while t < t_end:
        t += 1 / 30
        st, due = lane_spawn_step(st, jnp.float32(t), **kw)
        for team, lane in zip(*np.nonzero(np.asarray(due.due))):
            log.append((round(t, 3), int(team), int(lane), int(due.minion_type[team, lane])))
    return st, log


def test_every_lane_and_team_spawns_the_same_first_wave():
    st, log = _run(init_lane_spawn(), 0.0, 35.0)
    for team in (0, 1):
        for lane in (0, 1, 2):
            assert [k for (_, tm, ln, k) in log if tm == team and ln == lane] == [0, 0, 0, 1, 1, 1]
    first = sorted({t for (t, *_) in log})
    assert first[0] == pytest.approx(30.0, abs=1 / 30) and first[-1] == pytest.approx(34.0, abs=1 / 30)
    np.testing.assert_array_equal(st.wave, np.ones((2, 3)))


def test_lane_with_enemy_inhibitor_down_gets_a_super_instead_of_the_cannon():
    down = jnp.zeros((2, 3), bool).at[0, 2].set(True)       # red's top inhibitor down -> blue top supers
    _, log = _run(_cursor(2), 89.9, 96.0, enemy_inhibitor_down=down)
    types = lambda team, lane: [k for (_, tm, ln, k) in log if (tm, ln) == (team, lane)]
    assert types(0, 2) == [3, 0, 0, 0, 1, 1, 1]
    assert types(0, 1) == [0, 0, 0, 2, 1, 1, 1]
    assert types(1, 2) == [0, 0, 0, 2, 1, 1, 1]


def test_super_counts_all_down_and_respawn_cutoff():
    assert int(super_count(True, False, jnp.inf, 600.)) == 1
    assert int(super_count(False, True, jnp.inf, 600.)) == 2
    assert int(super_count(True, True, jnp.inf, 600.)) == 2
    assert int(super_count(True, False, 655., 600.)) == 0                 # within two 30 s intervals
    assert int(super_count(True, False, 661., 600.)) == 1
    assert int(super_count(True, False, 900., 865.)) == 0                 # two 25 s intervals
    assert int(super_count(True, False, 915., 865.)) == 1


def test_super_count_is_latched_for_the_whole_wave():
    down = jnp.zeros((2, 3), bool).at[1, 0].set(True)       # blue bot inhibitor down -> red bot supers
    st, due = lane_spawn_step(_cursor(4), jnp.float32(150.0), enemy_inhibitor_down=down)
    assert int(due.minion_type[1, 0]) == 3 and int(st.supers[1, 0]) == 1
    # The inhibitor respawns mid-wave: the wave completes with its latched list.
    out, t = [], 150.0
    while int(st.wave[1, 0]) == 4:
        t += 1 / 30
        st, due = lane_spawn_step(st, jnp.float32(t))
        if bool(due.due[1, 0]):
            out.append(int(due.minion_type[1, 0]))
    assert out == [0, 0, 0, 1, 1, 1] and int(st.supers[1, 0]) == -1
