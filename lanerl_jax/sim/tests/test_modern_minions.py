import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.sim.modern_minions import (
    CANNON_PROFILE, CASTER_PROFILE, MELEE_PROFILE, SUPER_PROFILE,
    MinionType, TargetKind, TargetPriority, call_for_help_applies,
    call_for_help_trigger, cannon_wave, gold_bounty,
    lane_minion_current_hp_bonus, minion_pushing_modifiers,
    select_target, shared_xp_fraction,
    spawn_event, target_priority, upgrade_index_at, wave_composition,
    wave_index_at, wave_interval_s, wave_spawn_time,
)


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
    np.testing.assert_array_equal(wave_composition(28, 865., True), [1, 3, 0, 3])
    np.testing.assert_array_equal(wave_composition(54, 1515.), [0, 2, 1, 3])
    np.testing.assert_array_equal(wave_composition(54, 1515., False, True), [2, 3, 0, 3])
    ev = [spawn_event(2, i) for i in range(7)]
    assert [int(x.minion_type) for x in ev] == [0, 0, 0, 2, 1, 1, 1]
    np.testing.assert_allclose([float(x.spawn_time_s) for x in ev],
                               [90 + .792 * i for i in range(7)], rtol=1e-6)
    assert not bool(spawn_event(2, 7).valid)
    assert int(spawn_event(-1, 0).minion_type) == MinionType.NONE
    np.testing.assert_array_equal(wave_composition(66, 1810.), [0, 3, 0, 2])


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
    assert int(select_target(p, d2, valid.at[2].set(False), current_target=0)) == 0
    # Once the incumbent is invalid, nearest candidate at the best available
    # priority is selected. Equal-distance ties retain collection order.
    assert int(select_target(p, d2, valid.at[0].set(False), current_target=0)) == 2


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
    np.testing.assert_allclose(shared_xp_fraction(jnp.arange(7)),
                               [0, 1, .65, 13 / 30, .325, .26, 13 / 60], rtol=1e-6)
    np.testing.assert_allclose(gold_bounty(jnp.asarray([1, 2, 3, 4]), 5),
                               [20, 14, 55, 55])
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
    np.testing.assert_array_equal(comp(jnp.asarray([2, 28]), jnp.asarray([90., 865.])),
                                  [[0, 3, 1, 3], [0, 2, 1, 3]])
