"""Simulation timing units and scripted ability workload."""
import math
from types import SimpleNamespace
import unittest

from ops.modern.bench import sample_summary, scripted_orders


class SampleSummaryTests(unittest.TestCase):
    def test_median_is_normalized_by_environment_ticks(self):
        result = sample_summary([4.0, 1.0, 2.0], 100, 30, 1 / 30)
        self.assertEqual(result['steady_s'], 2.0)
        self.assertEqual(result['env_ticks_per_s'], 1500.0)
        self.assertEqual(result['simulated_game_seconds_per_wall_s'], 50.0)
        self.assertEqual((result['min_s'], result['max_s']), (1.0, 4.0))

    def test_invalid_samples_are_rejected(self):
        for samples in ([], [0], [-1], [math.inf], [math.nan]):
            with self.subTest(samples=samples), self.assertRaises(ValueError):
                sample_summary(samples, 1, 1, 1 / 30)

    def test_invalid_counts_or_tick_duration_are_rejected(self):
        for envs, ticks, dt in ((0, 1, .1), (1, 0, .1), (1, 1, 0), (1, 1, math.nan)):
            with self.subTest(values=(envs, ticks, dt)), self.assertRaises(ValueError):
                sample_summary([1.0], envs, ticks, dt)


class BaselineKeyTests(unittest.TestCase):
    def test_ability_workload_has_an_independent_baseline(self):
        from unittest.mock import patch
        from ops.modern.perf_guard import config_key
        args = SimpleNamespace(envs=1, ticks=30, warm_ticks=60, fog='rays', no_jungle=True,
                               no_objectives=True, lanes=[2], packet_capacity=0, allowlist=False,
                               lane_structures=False, champions=['Garen', 'Jax'], roster=[], orders='lane')
        device = SimpleNamespace(device_kind='test')
        with patch('ops.modern.perf_guard.jax.default_backend', return_value='test'):
            lane = config_key(args, device)
            args.orders = 'abilities'
            self.assertEqual(config_key(args, device), lane + '|orders=abilities')


class ScriptedOrderTests(unittest.TestCase):
    def test_abilities_are_staggered_without_changing_lane_orders(self):
        import jax.numpy as jnp
        import numpy as np
        from lanerl_jax.modern.core import types as W
        state = SimpleNamespace(x=jnp.asarray([0., 500.]), y=jnp.zeros(2),
                                team=jnp.asarray([0, 1]), alive=jnp.ones(2, bool),
                                kind=jnp.full((2,), W.KIND_CHAMPION), tick=jnp.int32(30))
        lane = scripted_orders(state, jnp.asarray([1000., 0.]))
        active = scripted_orders(state, jnp.asarray([1000., 0.]), abilities=True)
        np.testing.assert_array_equal(lane.cast_slot, [-1, -1])
        np.testing.assert_array_equal(active.cast_slot, [1, -1])
        np.testing.assert_array_equal(active.cast_target, [1, 0])
        np.testing.assert_array_equal(active.attack, lane.attack)
        np.testing.assert_array_equal(active.move, lane.move)
        state.tick = jnp.int32(45)
        active = scripted_orders(state, jnp.asarray([1000., 0.]), abilities=True)
        np.testing.assert_array_equal(active.cast_slot, [-1, 2])


if __name__ == '__main__':
    unittest.main()
