"""Bounded MODERN-045 full-PPO diagnosis, using an immutable MODERN-043 source snapshot."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from ops.modern.training_bench import sha


# Small trainable batches first; repeat 256 only to isolate the allocator cap.
CASES = (
    ("modern", "matched", 64, ".75", False),
    ("legacy", "matched", 64, ".75", False),
    ("modern", "matched", 128, ".75", False),
    ("modern", "natural", 128, ".75", False),
    ("modern", "matched", 256, ".75", True),
    ("modern", "matched", 256, ".90", False),
    ("modern", "natural", 256, ".75", True),
    ("modern", "natural", 256, ".90", False),
)


def run(a):
    manifest = json.loads((a.root / "manifest.json").read_text())
    for name, digest in manifest["files"].items():
        if sha(a.root / name) != digest:
            raise RuntimeError(f"snapshot changed: {name}")
    a.output.mkdir(parents=True, exist_ok=False)
    shutil.copy2(a.root / "manifest.json", a.output / "manifest.json")
    errors = []
    for engine, preset, envs, fraction, diagnose in CASES:
        name = f"{engine}-{preset}-{envs}-fraction{fraction}"
        work = a.output / name
        work.mkdir()
        env = dict(os.environ, XLA_PYTHON_CLIENT_MEM_FRACTION=fraction,
                   XLA_PYTHON_CLIENT_PREALLOCATE="false")
        env.pop("XLA_PYTHON_CLIENT_ALLOCATOR", None)
        env.pop("TF_GPU_ALLOCATOR", None)
        cmd = [sys.executable, str(a.root / "training_bench.py"), "worker", "--root", str(a.root),
               "--engine", engine, "--preset", preset, "--envs", str(envs), "--layout", "top",
               "--phase", "ppo", "--rollout", "128", "--samples", "3", "--work", str(work)]
        if diagnose:
            cmd.append("--diagnose-memory")
        print(json.dumps(dict(event="start", case=name, command=cmd)), flush=True)
        with (work / "stdout.jsonl").open("w") as out, (work / "stderr.log").open("w") as err:
            try:
                proc = subprocess.run(cmd, cwd=a.root / engine, env=env, stdout=out, stderr=err,
                                      timeout=a.cell_timeout)
                returncode = proc.returncode
                status = "error"
            except subprocess.TimeoutExpired:
                returncode, status = None, "timeout"
        rows = [json.loads(x) for x in (work / "stdout.jsonl").read_text().splitlines() if x.startswith("{")]
        result = next((x for x in rows if x.get("event") == "result"), None)
        if returncode != 0 or result is None:
            error = (work / "stderr.log").read_text()
            if "RESOURCE_EXHAUSTED" in error or "out of memory" in error.lower():
                status = "oom"
            result = dict(event="failure", status=status, returncode=returncode,
                          compiled=next((x for x in rows if x.get("event") == "compiled"), None),
                          components=[x for x in rows if x.get("event") == "component_memory"])
            errors.append(status)
        result = dict(result, case=name, allocator_fraction=fraction)
        with (a.output / "results.jsonl").open("a") as out:
            out.write(json.dumps(result) + "\n")
        print(json.dumps(result), flush=True)
    if any(x != "oom" for x in errors):
        raise SystemExit(1)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--cell-timeout", type=int, default=720)
    a = p.parse_args()
    if a.cell_timeout < 1:
        p.error("cell timeout must be positive")
    run(a)


if __name__ == "__main__":
    main()
