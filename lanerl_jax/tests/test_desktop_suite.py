"""A test process failing must fail the suite, even when later files pass."""
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("failed", ["", "test_light.py", "test_local_pathing.py"])
def test_desktop_suite_propagates_failures(tmp_path, failed):
    repo = tmp_path / "repo"
    (repo / "ops").mkdir(parents=True)
    tests = repo / "lanerl_jax/sim/tests"
    tests.mkdir(parents=True)
    for name in ("test_light.py", "test_local_pathing.py"):
        (tests / name).touch()
    script = repo / "ops/desktop_suite.sh"
    script.write_text((ROOT / "ops/desktop_suite.sh").read_text())
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    commands = {
        "ssh": 'shift 3; exec "$@"',
        "free": 'echo "Mem: 64 0 0 0 0 64"',
        "systemd-run": 'case " $* " in *"/$FAILED_FILE "*) '
                      '[ -z "$FAILED_FILE" ] || exit 1;; esac; echo "1 passed"',
    }
    for name, body in commands.items():
        path = bin_dir / name
        path.write_text("#!/usr/bin/env bash\n" + body + "\n")
        path.chmod(0o755)
    result = subprocess.run(
        ["bash", str(script), "1", "5G"], text=True, capture_output=True,
        env={**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}",
             "FAILED_FILE": failed}, timeout=10,
    )
    assert (result.returncode == 0) == (not failed), result.stdout + result.stderr
    assert "test_local_pathing.py" in result.stdout, "must finish after a failure"
    assert ("FAIL " in result.stdout) == bool(failed)
