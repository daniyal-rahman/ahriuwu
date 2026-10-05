import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.modern.lane.minions import (CANNON_PROFILE, CASTER_PROFILE, MELEE_PROFILE, SUPER_PROFILE,
                                            MinionType, TargetKind, base_move_speed, call_for_help_applies,
                                            call_for_help_trigger, cannon_wave, gold_bounty,
                                            lane_minion_current_hp_bonus, minion_pushing_modifiers,
                                            minion_upgrade_stats, move_speed_soft_cap, select_target,
                                            shared_xp_fraction, sidelane_bonus_move_speed, spawn_event,
                                            target_priority, upgrade_index_at, wave_composition, wave_index_at,
                                            wave_interval_s, wave_spawn_time)


def test_wave_schedule_including_breakpoint_gap_and_cannon_sequence():
    indices = jnp.asarray([0, 26, 27, 28, 65, 66, 67])
    np.testing.assert_allclose(
        wave_spawn_time(indices), [30, 810, 840, 865, 1790, 1810, 1830])
    times = jnp.asarray([29.9, 30., 839.9, 840., 1799.9, 1800., 1809.9, 1810.])
    np.testing.assert_array_equal(wave_index_at(times), [-1, 0, 26, 27, 65, 65, 65, 66])
    np.testing.assert_array_equal(wave_interval_s(jnp.asarray([839., 840., 1765., 1790.])),
                                  [30., 25., 25., 20.])
    np.testing.assert_array_equal(
        cannon_wave(jnp.asarray([1, 2, 26, 27, 28, 30, 52, 53, 54, 65])),
        [False, True, True, False, True, True, True, False, True, True])


def test_composition_and_spawn_order_and_offsets():
    # Index 28 is 14:25: cannon wave after the tempo transition, with one
    # fewer melee. A destroyed top inhibitor replaces that cannon with a super.
    np.testing.assert_array_equal(wave_composition(28, 865.), [0, 2, 1, 3])
    # Changed: the melee count follows the client rotation [2,3] regardless of
    # whether a super replaced the cannon (MINIONS §2.3/§9, U-12); the old
    # expectation [1,3,0,3] tied the melee count to cannon presence.
    np.testing.assert_array_equal(wave_composition(28, 865., True), [1, 2, 0, 3])
    np.testing.assert_array_equal(wave_composition(29, 890., True), [1, 3, 0, 3])
    np.testing.assert_array_equal(wave_composition(54, 1515.), [0, 2, 1, 3])
    # Changed for the same reason (was [2, 3, 0, 3]).
    np.testing.assert_array_equal(wave_composition(54, 1515., False, True), [2, 2, 0, 3])
    # MINIONS §10 composition fixtures.
    for i, expected in [(0, [0, 3, 0, 3]), (2, [0, 3, 1, 3]), (26, [0, 3, 1, 3]),
                        (27, [0, 3, 0, 3]), (29, [0, 3, 0, 3]), (53, [0, 3, 0, 3]),
                        (65, [0, 2, 1, 3]), (66, [0, 2, 1, 2])]:
        np.testing.assert_array_equal(wave_composition(i, wave_spawn_time(i)), expected)
    # No supers within two waves of the inhibitor respawning.
    np.testing.assert_array_equal(wave_composition(28, 865., True, inhibitor_respawn_at=900.),
                                  [0, 2, 1, 3])
    np.testing.assert_array_equal(wave_composition(28, 865., True, inhibitor_respawn_at=915.),
                                  [1, 2, 0, 3])
    ev = [spawn_event(2, i) for i in range(7)]
    assert [int(x.minion_type) for x in ev] == [0, 0, 0, 2, 1, 1, 1]
    # Changed: client MinionSpawnIntervalSecs 0.8 (was 0.792, unsourced; U-13).
    np.testing.assert_allclose([float(x.spawn_time_s) for x in ev],
                               [90 + .8 * i for i in range(7)], rtol=1e-6)
    assert not bool(spawn_event(2, 7).valid)
    assert int(spawn_event(-1, 0).minion_type) == MinionType.NONE
    np.testing.assert_array_equal(wave_composition(66, 1810.), [0, 2, 1, 2])


def test_target_ranks_26_10_and_incumbent_only_yields_to_strict_upgrade():
    kinds = jnp.asarray([TargetKind.CHAMPION, TargetKind.MINION,
                         TargetKind.MINION, TargetKind.TURRET,
                         TargetKind.MINION, TargetKind.CHAMPION])
    victims = jnp.asarray([TargetKind.CHAMPION, TargetKind.CHAMPION,
                           TargetKind.MINION, TargetKind.MINION, 0,
                           TargetKind.MINION])
    np.testing.assert_array_equal(
        target_priority(kinds, victims), [1, 2, 3, 4, 5, 6])
    # 26.10 removed the champion-attacking-allied-minion special priority.
    assert int(target_priority(TargetKind.CHAMPION, TargetKind.MINION)) == 6

    p = jnp.asarray([5, 6, 2, 1])
    d2 = jnp.asarray([1., 1., 50., 80.])
    valid = jnp.ones((4,), bool)
    assert int(select_target(p, d2, valid, current_target=0)) == 3
    # Same priority never displaces the valid incumbent, even if nearer.
    assert int(select_target(p, d2, valid.at[2].set(False).at[3].set(False),
                             current_target=0)) == 0
    # Once the incumbent is invalid, nearest candidate at the best available
    # priority is selected. Equal-distance ties retain collection order.
    assert int(select_target(p, d2, valid.at[0].set(False).at[3].set(False),
                             current_target=0)) == 2


def test_call_for_help_rules_and_distance_gate():
    assert bool(call_for_help_trigger(
        champion_hit_enemy_champion=False, champion_in_path=True,
        no_other_target_in_attack_range=True, outside_turret_range=True,
        minion_attacking_turret=True, is_first_wave=True))
    assert not bool(call_for_help_trigger(
        champion_hit_enemy_champion=True, champion_in_path=False,
        no_other_target_in_attack_range=False, outside_turret_range=False,
        minion_attacking_turret=True, is_first_wave=False))
    assert bool(call_for_help_trigger(
        champion_hit_enemy_champion=True, champion_in_path=False,
        no_other_target_in_attack_range=False, outside_turret_range=False,
        minion_attacking_turret=False, is_first_wave=False))
    assert bool(call_for_help_applies(900., True, 500.))
    assert not bool(call_for_help_applies(1001., True, 500.))
    assert not bool(call_for_help_applies(501., False, 500.))


def test_profile_facts_xp_and_upgrades_and_jax_transforms():
    assert MELEE_PROFILE.health_base == 465 and MELEE_PROFILE.health_cap == 1550
    assert CASTER_PROFILE.attack_range == 550 and CANNON_PROFILE.attack_range == 300
    assert SUPER_PROFILE.armor_base == 100 and SUPER_PROFILE.magic_resist == -30
    # Changed: client mPlayerMinionSplitXp floats (was 13/30 and 13/60).
    np.testing.assert_allclose(shared_xp_fraction(jnp.arange(7)),
                               [0, 1, .65, .433, .325, .26, .217], rtol=1e-6)
    # Changed: siege/super gold is 49 + U (50 at U=1), not 50 + U, so U=5
    # pays 54 (MINIONS §1.3/§9); Chaos supers stay at 49.
    np.testing.assert_allclose(gold_bounty(jnp.asarray([0, 1, 2, 3]), 5),
                               [20, 14, 54, 54])
    np.testing.assert_allclose(gold_bounty(2, jnp.asarray([1, 5, 10, 41, 50])),
                               [50, 54, 59, 90, 90])
    assert float(gold_bounty(3, 5, team=1)) == 49
    np.testing.assert_array_equal(upgrade_index_at(jnp.asarray([29.9, 30, 119.9, 120, 210])),
                                  [0, 1, 1, 2, 3])
    np.testing.assert_allclose(
        lane_minion_current_hp_bonus(jnp.asarray([0, 1, 2, 3]), 1000.),
        [20, 35, 50, 0])
    bonus, divisor = minion_pushing_modifiers(
        team_level_advantage=4., lane_turret_advantage=2., time_s=210.)
    np.testing.assert_allclose([bonus, divisor], [.45, 7.])
    fn = jax.jit(jax.vmap(wave_index_at))
    np.testing.assert_array_equal(fn(jnp.asarray([30., 840., 1810.])), [0, 27, 66])
    comp = jax.jit(jax.vmap(lambda idx, t: wave_composition(idx, t)))
    result = comp(jnp.asarray([2, 28]), jnp.asarray([90., 865.]))
    np.testing.assert_array_equal(jnp.stack(result, axis=-1),
                                  [[0, 3, 1, 3], [0, 2, 1, 3]])


def test_upgrade_stats_match_client_formula_fixtures():
    # MINIONS §10 "Stats at U".
    def stats(kind, u):
        r = minion_upgrade_stats(kind, u)
        return [float(r.max_hp), float(r.attack_damage), float(r.armor)]
    np.testing.assert_allclose(stats(0, 1), [465, 11, 0])
    np.testing.assert_allclose(stats(0, 6), [640, 14, 0])
    np.testing.assert_allclose(stats(0, 7), [675, 17, .085], rtol=1e-6)
    np.testing.assert_allclose(stats(0, 10), [780, 26, .85], rtol=1e-6)
    assert stats(0, 32)[0] == 1550 and stats(0, 33)[0] == 1550
    np.testing.assert_allclose(minion_upgrade_stats(1, jnp.asarray([5, 6, 29, 30])).attack_damage,
                               [27, 31, 123, 125])
    np.testing.assert_allclose(minion_upgrade_stats(2, jnp.asarray([5, 6, 26])).attack_damage,
                               [43.5, 47.5, 126])
    sup = minion_upgrade_stats(3, 1)
    np.testing.assert_allclose([sup.max_hp, sup.attack_damage, sup.armor, sup.magic_resist],
                               [1600, 185, 100, -30])
    assert SUPER_PROFILE.attack_damage_base == 185   # was 180 (the U=0 value)
    np.testing.assert_array_equal(
        upgrade_index_at(jnp.asarray([450., 480., 570., 840., 1500.])), [5, 6, 7, 10, 17])


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


# --- all-lane spawn schedule -------------------------------------------------
from lanerl_jax.modern.lane.minions import LaneSpawnState, init_lane_spawn, lane_spawn_step, super_count, wave_unit_type


def _run_schedule(t_end, dt=1 / 30, **kw):
    st, t, log = init_lane_spawn(), 0.0, []
    while t < t_end:
        t += dt
        st, due = lane_spawn_step(st, jnp.float32(t), **kw)
        for team, lane in zip(*np.nonzero(np.asarray(due.due))):
            log.append((round(t, 3), int(team), int(lane), int(due.minion_type[team, lane])))
    return st, log


def test_every_lane_and_team_spawns_the_same_first_wave():
    st, log = _run_schedule(35.0)
    for team in (0, 1):
        for lane in (0, 1, 2):
            types = [k for (_, tm, ln, k) in log if tm == team and ln == lane]
            assert types == [0, 0, 0, 1, 1, 1]
    first = sorted({t for (t, *_) in log})
    assert first[0] == pytest_approx(30.0, 1 / 30) and first[-1] == pytest_approx(34.0, 1 / 30)
    np.testing.assert_array_equal(st.wave, np.ones((2, 3)))


def pytest_approx(v, tol):
    import pytest
    return pytest.approx(v, abs=tol)


def test_wave_unit_type_matches_spawn_event_without_supers():
    for i in (0, 2, 27, 28, 53, 54, 66):
        for u in range(8):
            k, ok = wave_unit_type(i, u, 0)
            ev = spawn_event(i, u)
            assert int(k) == int(ev.minion_type) and bool(ok) == bool(ev.valid)


def test_lane_with_enemy_inhibitor_down_gets_a_super_instead_of_the_cannon():
    # Red's (team 1) top inhibitor down -> blue's top waves get one super; nothing else changes.
    down = jnp.zeros((2, 3), bool).at[0, 2].set(True)
    st = LaneSpawnState(jnp.full((2, 3), 2, jnp.int32), jnp.zeros((2, 3), jnp.int32), jnp.full((2, 3), -1, jnp.int32))
    types = {}
    t = 89.9
    while t < 96.0:
        t += 1 / 30
        st, due = lane_spawn_step(st, jnp.float32(t), enemy_inhibitor_down=down)
        for team, lane in zip(*np.nonzero(np.asarray(due.due))):
            types.setdefault((int(team), int(lane)), []).append(int(due.minion_type[team, lane]))
    assert types[(0, 2)] == [3, 0, 0, 0, 1, 1, 1]          # super first, no cannon
    assert types[(0, 1)] == [0, 0, 0, 2, 1, 1, 1]          # mid keeps its cannon
    assert types[(1, 2)] == [0, 0, 0, 2, 1, 1, 1]


def test_super_counts_all_down_and_respawn_cutoff():
    assert int(super_count(True, False, jnp.inf, 600.)) == 1
    assert int(super_count(False, True, jnp.inf, 600.)) == 2
    assert int(super_count(True, True, jnp.inf, 600.)) == 2
    # Inhibitor respawns at 655: wave at 600 is within two 30-s intervals -> no super.
    assert int(super_count(True, False, 655., 600.)) == 0
    assert int(super_count(True, False, 661., 600.)) == 1


def test_super_count_is_latched_for_the_whole_wave():
    down = jnp.zeros((2, 3), bool).at[1, 0].set(True)       # blue bot inhibitor down -> red bot supers
    st = LaneSpawnState(jnp.full((2, 3), 4, jnp.int32), jnp.zeros((2, 3), jnp.int32), jnp.full((2, 3), -1, jnp.int32))
    st, due = lane_spawn_step(st, jnp.float32(150.0), enemy_inhibitor_down=down)
    assert int(due.minion_type[1, 0]) == 3 and int(st.supers[1, 0]) == 1
    # The inhibitor respawns mid-wave: the wave still completes with its latched list.
    out = []
    t = 150.0
    while int(st.wave[1, 0]) == 4:
        t += 1 / 30
        st, due = lane_spawn_step(st, jnp.float32(t))
        if bool(due.due[1, 0]):
            out.append(int(due.minion_type[1, 0]))
    assert out == [0, 0, 0, 1, 1, 1]
    assert int(st.supers[1, 0]) == -1
