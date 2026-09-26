"""ops/launch.py refuses the launch mistakes of 2026-09-26 and dry-runs every spec."""
import json, subprocess, sys
from pathlib import Path
import pytest
ROOT = Path(__file__).resolve().parents[2]
SPECS = sorted(ROOT.glob("experiments/*.json"))


def run(*args):
    return subprocess.run([sys.executable, "ops/launch.py", *args, "--dry-run"], cwd=ROOT,
                          capture_output=True, text=True)


@pytest.mark.parametrize("spec", SPECS, ids=lambda p: p.stem)
def test_every_spec_dry_runs(spec):
    r = run(spec.stem, "--no-canary")
    assert r.returncode == 0, r.stdout + r.stderr
    assert "command:" in r.stdout


def test_missing_init_checkpoint_is_refused():
    r = run(SPECS[0].stem, "--init-from", "/mnt/nfs/nope.msgpack")
    assert r.returncode != 0 and "REFUSED" in r.stdout + r.stderr


def test_resume_and_init_are_exclusive():
    spec = json.loads(SPECS[0].read_text())
    if "init_from" in spec:
        r = run(SPECS[0].stem, "--resume", str(ROOT / "README.md"))
        assert r.returncode != 0 and "exclusive" in r.stdout + r.stderr


def test_ephemeral_port_base_is_refused(tmp_path, monkeypatch):
    bad = json.loads(SPECS[0].read_text()); bad["id"] = "ZZ_bad_port"; bad["port_base"] = 49700
    (ROOT / "experiments" / "ZZ_bad_port.json").write_text(json.dumps(bad))
    try:
        r = run("ZZ_bad_port", "--no-canary")
        assert r.returncode != 0 and "OPS-003" in r.stdout + r.stderr
    finally:
        (ROOT / "experiments" / "ZZ_bad_port.json").unlink()
