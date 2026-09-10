#!/usr/bin/env python
"""Measure what a full server process restart costs.

This is the number an in-process reset has to beat. We time three points:

  boot     -- process spawn -> "Game is ready." (config parse, content load,
              Roslyn compile of every Content script, map/nav-grid load)
  start    -- process spawn -> game actually running (first recorded tick)
  ready10  -- process spawn -> 10 minutes of *game* time simulated, under
              LANERL_FREERUN

Only `start` is the per-episode cost you would pay if you restarted the
process between episodes; `boot` is the part an in-process reset skips.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import statistics
import subprocess
import time
from pathlib import Path

# The NFS export is mounted at /srv/nfs on danilogin and /mnt/nfs on desktop.
# Never hardcode either: resolve everything from this file's own location.
# (This block was added to the three sibling files and missed here, so every
# call raised NameError: _VENDOR -- the module had never actually run.)
_PROJECTS = Path(__file__).resolve().parents[3]
_VENDOR = str(_PROJECTS / "lanerl-vendor")

SERVER_DIR = Path(
    _VENDOR + "/LoLServer/GameServerConsole/bin/Release/net6.0"
)
DOTNET_ROOT = Path(_VENDOR + "/dotnet")


def run_once(config: Path, workdir: Path, trial: int, freerun: bool) -> dict:
    log = workdir / f"boot{trial}.log"
    rec = workdir / f"boot{trial}.jsonl"
    for p in (log, rec):
        p.unlink(missing_ok=True)

    env = dict(os.environ)
    env["DOTNET_ROOT"] = str(DOTNET_ROOT)
    env["LANERL_HEADLESS"] = "1"
    env["LANERL_RECORD"] = str(rec)
    if freerun:
        env["LANERL_FREERUN"] = "1"
    # a distinct port per trial so a lingering socket cannot stall the next run
    port = str(5300 + trial)

    t0 = time.monotonic()
    with log.open("wb") as fh:
        proc = subprocess.Popen(
            [str(DOTNET_ROOT / "dotnet"), "./GameServerConsole.dll",
             "--config", str(config), "--port", port],
            cwd=SERVER_DIR, env=env, stdout=fh, stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    t_ready = t_first = None
    try:
        deadline = t0 + 180
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                break
            if t_ready is None and log.exists():
                if "Game is ready." in log.read_text(errors="replace"):
                    t_ready = time.monotonic() - t0
            if t_ready is not None and t_first is None and rec.exists():
                if rec.stat().st_size > 0:
                    t_first = time.monotonic() - t0
                    break
            time.sleep(0.01)
    finally:
        pass

    # keep running until 10 game-minutes are simulated, then stop
    t_10min = None
    deadline = t0 + 900
    while time.monotonic() < deadline and proc.poll() is None:
        try:
            with rec.open("rb") as fh:
                fh.seek(max(0, rec.stat().st_size - 4096))
                tail = fh.read().decode(errors="replace").strip().split("\n")
            gt = json.loads(tail[-1])["t"] if tail and tail[-1].startswith("{") else 0
        except Exception:
            gt = 0
        if gt >= 600_000:
            t_10min = time.monotonic() - t0
            break
        time.sleep(0.25)

    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        proc.kill()
    proc.wait(timeout=30)

    return {"boot_s": t_ready, "start_s": t_first, "sim10min_s": t_10min}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--freerun", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(args.trials):
        r = run_once(args.config, args.out, i, args.freerun)
        rows.append(r)
        print(f"trial {i}: {r}", flush=True)

    summary = {"trials": rows, "freerun": args.freerun}
    for k in ("boot_s", "start_s", "sim10min_s"):
        vals = [r[k] for r in rows if r[k] is not None]
        if vals:
            summary[k] = {
                "median": statistics.median(vals),
                "min": min(vals),
                "max": max(vals),
                "n": len(vals),
            }
    (args.out / "process_restart.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
