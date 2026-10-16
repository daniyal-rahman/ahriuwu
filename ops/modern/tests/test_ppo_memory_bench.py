"""Diagnostic failures must retain memory evidence and allow the bounded cases to finish."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from ops.modern.ppo_memory_bench import run


class FailureTests(unittest.TestCase):
    def test_timeout_and_oom_preserve_evidence_and_allocator_controls(self):
        seen = []

        def worker(cmd, **kw):
            seen.append(kw['env'])
            kw['stdout'].write('{"event":"component_memory","component":"minibatch_value_and_grad"}\n')
            kw['stdout'].write('{"event":"compiled","compiled_memory":{"temp_size_in_bytes":123}}\n')
            if len(seen) == 1:
                raise subprocess.TimeoutExpired(cmd, 1)
            kw['stderr'].write('RESOURCE_EXHAUSTED')
            return subprocess.CompletedProcess(cmd, 1)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / 'manifest.json').write_text('{"files":{}}')
            args = argparse.Namespace(root=root, output=root / 'results', cell_timeout=1)
            with patch('ops.modern.ppo_memory_bench.subprocess.run', side_effect=worker), \
                    patch.dict('os.environ', {'TF_GPU_ALLOCATOR': 'other', 'XLA_PYTHON_CLIENT_ALLOCATOR': 'platform'}), \
                    contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                run(args)
            rows = [json.loads(x) for x in (args.output / 'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 8)
        self.assertEqual(rows[0]['status'], 'timeout')
        self.assertTrue(all(x['status'] == 'oom' for x in rows[1:]))
        self.assertTrue(all(x['compiled']['compiled_memory']['temp_size_in_bytes'] == 123 for x in rows))
        self.assertTrue(all(len(x['components']) == 1 for x in rows))
        self.assertEqual({x['XLA_PYTHON_CLIENT_MEM_FRACTION'] for x in seen}, {'.75', '.90'})
        self.assertTrue(all('TF_GPU_ALLOCATOR' not in x and 'XLA_PYTHON_CLIENT_ALLOCATOR' not in x for x in seen))


if __name__ == '__main__':
    unittest.main()
