#!/usr/bin/env python
"""One line describing a live run, for polling from a Monitor.

`run_watch.py` prints a full report; this prints a single line, because each
line a Monitor emits becomes a separate notification and a full report per poll
is unreadable. It carries exactly what decides the next action on THIS run:
whether CS@10 is still declining, and which reward term the policy is chasing.

    python lanerl/run_summary.py <run_dir>
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path


def main() -> int:
    run = Path(sys.argv[1])
    m = run / "metrics.jsonl"
    if not m.exists():
        print(f"{run.name}: no metrics.jsonl yet (still booting)")
        return 0

    rows = []
    for line in m.read_text(errors="ignore").splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except Exception:
                pass

    upd = [r for r in rows if r.get("kind") == "update"]
    rej = [r for r in rows if r.get("kind") == "rollout_rejected"]
    eps = [r for r in rows if r.get("kind") == "episode"]
    cs = [r["cs_at_10"] for r in eps if r.get("cs_at_10") is not None]

    parts = [f"{run.name}: upd={len(upd)} eps={len(eps)}"]
    if upd or rej:
        parts.append(f"rej={100 * len(rej) / (len(upd) + len(rej)):.0f}%")
    d = [r["throughput/decisions_per_s"] for r in upd if r.get("throughput/decisions_per_s")]
    if d:
        parts.append(f"{statistics.mean(d[len(d) // 2:]):,.0f}dec/s")

    if cs:
        parts.append(f"CS@10 n={len(cs)} mean={statistics.mean(cs):.1f}")
        # The whole question: is it going up or down? Thirds, like the
        # rl-bc4-0912 analysis this has to be comparable with.
        if len(cs) >= 12:
            t = len(cs) // 3
            parts.append(f"[{statistics.mean(cs[:t]):.1f} -> {statistics.mean(cs[-t:]):.1f}]")
    else:
        parts.append("CS@10 none yet")

    # Which term the policy is actually being paid by. Never been visible on
    # any run before today; the surviving hypotheses all resolve on it.
    wt = [r for r in eps if r.get("reward_terms")]
    if wt:
        agg = {}
        for r in wt[-40:]:
            for k, v in r["reward_terms"].items():
                agg[k] = agg.get(k, 0.0) + float(v)
        tot = sum(abs(v) for v in agg.values()) or 1.0
        top = sorted(agg.items(), key=lambda kv: -abs(kv[1]))[:4]
        parts.append("terms " + " ".join(f"{k}={100 * v / tot:+.0f}%" for k, v in top))

    print("  ".join(parts))
    return 0


if __name__ == "__main__":
    sys.exit(main())
