#!/usr/bin/env python3
"""Scan every (PID, offset) for f32 values that look like HP / HP_max — i.e.
fall in [50, 5000] for the majority of blocks. Cheap unsupervised filter."""
from __future__ import annotations

import argparse
import struct
from collections import Counter, defaultdict

from parse_rofl import parse_rofl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rofl",
                    default="/mnt/nfs/datasets/lol_replays_16_9_772/NA1_5552884026/rofl/replay.rofl")
    ap.add_argument("--min-blocks", type=int, default=50)
    ap.add_argument("--lo", type=float, default=30.0)
    ap.add_argument("--hi", type=float, default=8000.0)
    ap.add_argument("--frac-thr", type=float, default=0.85)
    args = ap.parse_args()

    blocks = parse_rofl(args.rofl)
    print(f"loaded {len(blocks)} blocks")

    # Index: (pid, modal_payload_size) → blocks
    by_pid_size: dict[tuple[int, int], list[bytes]] = defaultdict(list)
    for b in blocks:
        by_pid_size[(b.pid, len(b.payload))].append(b.payload)

    rows = []
    for (pid, sz), pls in by_pid_size.items():
        if len(pls) < args.min_blocks:
            continue
        if sz < 4:
            continue
        for off in range(0, sz - 3):
            in_range = 0
            uniq = set()
            for pl in pls:
                try:
                    v = struct.unpack_from("<f", pl, off)[0]
                except Exception:
                    continue
                if v != v:
                    continue
                if args.lo <= v <= args.hi:
                    in_range += 1
                    uniq.add(round(v, 1))
            frac = in_range / len(pls)
            if frac >= args.frac_thr and len(uniq) >= 5:
                rows.append((pid, sz, off, len(pls), in_range, frac, len(uniq)))

    rows.sort(key=lambda r: -(r[3] * r[5]))
    print(f"\n{'pid':>5} {'sz':>3} {'off':>3} {'n':>7} {'in_rng':>7} {'frac':>5} {'unique':>6}")
    print("-" * 50)
    for pid, sz, off, n, ir, frac, u in rows[:60]:
        print(f"{pid:5d} {sz:3d} {off:3d} {n:7d} {ir:7d} {frac:5.2f} {u:6d}")


if __name__ == "__main__":
    main()
