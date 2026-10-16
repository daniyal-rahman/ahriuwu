"""Units and denominators in MODERN-043's training comparison; no JAX needed."""
import unittest

from ops.modern.training_bench import rates


class RateTests(unittest.TestCase):
    def test_game_time_is_per_environment_not_per_agent(self):
        r = rates(256, 128, 4, 30, 3, 2, 2)
        self.assertEqual(r['env_decisions_per_s'], 8192)
        self.assertEqual(r['agent_decisions_per_s'], 16384)
        self.assertAlmostEqual(r['simulated_game_seconds_per_wall_s'], 819.2)

    def test_tick_rate_normalization_and_afk_learning(self):
        modern = rates(1024, 128, 8, 30, 3, 2, 2)
        old = rates(1024, 128, 8, 60, 6, 2, 1)
        self.assertEqual(modern['simulated_game_seconds_per_wall_s'], old['simulated_game_seconds_per_wall_s'])
        self.assertEqual(old['learner_agent_decisions_per_s'], old['env_decisions_per_s'])
        self.assertEqual(modern['learner_agent_decisions_per_s'], 2 * old['learner_agent_decisions_per_s'])

    def test_different_decision_rate(self):
        r = rates(10, 20, 2, 60, 3, 2, 1)
        self.assertEqual(r['simulated_game_seconds_per_wall_s'], 5)

    def test_invalid_denominators_and_counts(self):
        base = [256, 128, 4, 30, 3, 2, 2]
        for i in range(len(base)):
            for bad in (0, -1, float('nan'), float('inf')):
                args = base.copy()
                args[i] = bad
                with self.subTest(index=i, value=bad), self.assertRaises(ValueError):
                    rates(*args)


class MatrixFailureTests(unittest.TestCase):
    def test_oom_cells_preserve_results_and_continue(self):
        import argparse
        import contextlib
        import io
        import json
        from pathlib import Path
        import tempfile
        from ops.modern.training_bench import matrix

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text('{"files": {}}')
            for engine in ('modern', 'legacy'):
                (root / engine).mkdir()
            (root / 'training_bench.py').write_text(
                'import sys\nprint("RESOURCE_EXHAUSTED", file=sys.stderr)\nsys.exit(1)\n')
            args = argparse.Namespace(root=root, output=root / 'results', presets=['matched'],
                env_counts=[256], layouts=['top'], samples=1, rollout=128, seed=0, backend='gpu')
            with contextlib.redirect_stdout(io.StringIO()):
                matrix(args)
            rows = [json.loads(x) for x in (args.output / 'results.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows), 4)
            self.assertTrue(all(x['status'] == 'oom' for x in rows))
            self.assertEqual(len({x['case'] for x in rows}), 4)


if __name__ == '__main__':
    unittest.main()
