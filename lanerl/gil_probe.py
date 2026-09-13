#!/usr/bin/env python
"""Is the rollout loop actually parallel, or is it one GIL-bound thread?

``--num-actors N`` reads like it buys N-way parallelism.  It does not
necessarily: ``ActorLoop.start`` is ``threading.Thread``, so every actor and
the learner live in ONE process sharing ONE GIL, and the work an actor does --
``ObservationBuilder.build`` -- is pure Python, which *holds* the GIL for its
whole duration.  If that is what is happening, then no number of actors and no
number of cores can make observation building go faster, and the scaling curve
measured from outside (throughput falling 498 -> 209 decisions/s from 8 to 72
instances, on a 16-core box showing load 0.55) is contention, not load.

That is a hypothesis about thread scheduling.  This measures it, three ways,
without touching the training code:

  cores_used      total process CPU time / wall time.  The whole claim in one
                  number: a genuinely parallel N-actor run uses ~N cores; a
                  GIL-bound one pins near 1.0 no matter what N is.

  per-thread CPU  sampled from /proc/<pid>/task/<tid>/stat, grouped by thread
                  name.  Shows WHICH thread holds the time -- actor threads
                  starving while the learner runs looks completely different
                  from every actor getting an equal, tiny slice.

  throughput      decisions/s from the run's own metrics.jsonl, so the cost is
                  expressed in the unit that matters rather than in CPU alone.

Run it over a sweep that holds TOTAL INSTANCES FIXED and varies only the actor
count.  That is the controlled comparison: same servers, same decisions, same
work -- only the number of Python threads changes.  If throughput is flat or
falls across that sweep, ``--num-actors`` is buying nothing.

    python lanerl/gil_probe.py --seconds 180 -- <training command...>
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

CLK_TCK = os.sysconf("SC_CLK_TCK")


def _thread_cpu(pid: int) -> Dict[str, float]:
    """Per-thread CPU seconds so far, keyed by ``name/tid``.

    Keyed by name AND tid because thread names are not unique (every actor is
    ``lanerl-actor-<i>``, but the pool of them is the whole question) and
    because a thread that dies mid-run must not have its time silently merged
    into a later thread that reused the tid.
    """
    out: Dict[str, float] = {}
    task = Path(f"/proc/{pid}/task")
    try:
        tids = list(task.iterdir())
    except OSError:
        return out
    for tdir in tids:
        try:
            stat = (tdir / "stat").read_text()
            name = (tdir / "comm").read_text().strip()
        except OSError:
            continue  # thread exited between listing and reading
        # comm can contain spaces and parens, so split on the LAST ')'.
        try:
            rest = stat[stat.rindex(")") + 2:].split()
            utime, stime = int(rest[11]), int(rest[12])
        except (ValueError, IndexError):
            continue
        out[f"{name}/{tdir.name}"] = (utime + stime) / CLK_TCK
    return out


def _machine_busy_seconds() -> float:
    """CPU-seconds the whole machine has burned, across every core.

    The Python process is only half the bill: each instance is a separate C#
    game server process, and none of them appear under /proc/<pid>/task. Sizing
    "how many instances fit on this box" needs the MACHINE number, or the
    answer is off by however much the simulator costs -- which is the part that
    actually scales with instance count.
    """
    with open("/proc/stat") as fh:
        parts = fh.readline().split()
    # user nice system idle iowait irq softirq steal ... -- everything except
    # idle and iowait is a core doing work.
    vals = [int(v) for v in parts[1:9]]
    busy = sum(vals) - vals[3] - vals[4]
    return busy / CLK_TCK


def _decisions_per_s(run_dir: Optional[Path]) -> Optional[float]:
    if run_dir is None:
        return None
    m = run_dir / "metrics.jsonl"
    if not m.exists():
        return None
    vals = []
    for line in m.read_text(errors="ignore").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        v = r.get("throughput/decisions_per_s")
        if isinstance(v, (int, float)):
            vals.append(float(v))
    if not vals:
        return None
    # The tail only: early updates include server startup and the first
    # episode's allocation churn, which is not what steady-state costs.
    tail = vals[len(vals) // 2:] or vals
    return sum(tail) / len(tail)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--seconds", type=float, default=180.0)
    ap.add_argument("--interval", type=float, default=2.0)
    ap.add_argument("--run-dir", default=None, help="to read decisions/s from")
    ap.add_argument("--label", default="")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    args = ap.parse_args()

    cmd = args.cmd
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("no command given")

    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    pid = proc.pid
    t0 = time.time()
    machine0 = _machine_busy_seconds()
    peak: Dict[str, float] = defaultdict(float)
    try:
        while time.time() - t0 < args.seconds:
            if proc.poll() is not None:
                break
            for k, v in _thread_cpu(pid).items():
                # Max, not last: a thread that exits before the final sample
                # would otherwise contribute nothing at all.
                if v > peak[k]:
                    peak[k] = v
            time.sleep(args.interval)
        # One last sample BEFORE the kill. Without it, everything the process
        # earned between the final loop sample and the end of the window is
        # lost while `wall` still counts that time, so cores_used reads low by
        # up to one full interval's worth.
        for k, v in _thread_cpu(pid).items():
            if v > peak[k]:
                peak[k] = v
    finally:
        wall = time.time() - t0
        machine = _machine_busy_seconds() - machine0
        if proc.poll() is None:
            os.killpg(os.getpgid(pid), signal.SIGTERM)
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(pid), signal.SIGKILL)

    total = sum(peak.values())
    groups: Dict[str, float] = defaultdict(float)
    for k, v in peak.items():
        groups[k.rsplit("/", 1)[0]] += v

    dps = _decisions_per_s(Path(args.run_dir) if args.run_dir else None)
    ncpu = os.cpu_count() or 1
    print(f"\n=== {args.label or ' '.join(shlex.quote(c) for c in cmd)[:70]} ===")
    print(f"wall {wall:.0f}s   cpu {total:.0f}s   CORES USED {total / wall:.2f}"
          f"   MACHINE {machine / wall:.2f} of {ncpu}"
          f"   (servers ~{max(0.0, machine - total) / wall:.2f})")
    if dps is not None:
        print(f"throughput {dps:,.0f} decisions/s")
    print("  per-thread-name cores:")
    for name, secs in sorted(groups.items(), key=lambda kv: -kv[1])[:10]:
        n = sum(1 for k in peak if k.rsplit("/", 1)[0] == name)
        print(f"    {name:24s} {secs / wall:5.2f} cores  ({n} thread(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
