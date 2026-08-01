#!/usr/bin/env python3
"""Find PIDs whose blocks are keyed on hero entity ids (0x400000ae..0x400000b7
in NA1_5552884026, more generally any entity that matches the 10 heroes
present in the raw_mem ground truth).

For each hero-PID, report:
  - block count per entity
  - payload size distribution
  - rough cadence (blocks-per-game-second)

This narrows the search space for HP correlation: stat updates should be
hero-entity-keyed, small-ish payload, and steady cadence."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict

from parse_rofl import parse_rofl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--rofl",
        default="/mnt/nfs/datasets/lol_replays_16_9_772/NA1_5552884026/rofl/replay.rofl",
    )
    ap.add_argument(
        "--raw-mem",
        default="/mnt/nfs/datasets/lol_replays_16_9_772/NA1_5552884026/raw_mem.json",
    )
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--max-payload", type=int, default=64,
                    help="cap payload size — HP candidates are small (<=64B)")
    args = ap.parse_args()

    rm = json.load(open(args.raw_mem))
    n_heroes = len(rm[0]["heroes"])
    game_length = rm[-1]["gt"]
    print(f"raw_mem: {len(rm)} ticks, {n_heroes} heroes, gt range [0, {game_length:.1f}s]")

    blocks = parse_rofl(args.rofl)
    print(f"rofl:    {len(blocks)} blocks")

    # Hero entity ids look like 0x400000xx. We want PIDs whose param set is
    # dominated by exactly n_heroes such ids.
    HERO_MASK = 0xFFFFFF00
    HERO_PREFIX = 0x40000000

    # PID → (param → block count)
    pid_param_count: dict[int, Counter] = defaultdict(Counter)
    pid_sizes: dict[int, list[int]] = defaultdict(list)

    for b in blocks:
        pid_param_count[b.pid][b.param] += 1
        pid_sizes[b.pid].append(len(b.payload))

    # For each PID, count how many of its blocks land on hero-shaped entity ids
    candidates = []
    for pid, pcount in pid_param_count.items():
        sizes = pid_sizes[pid]
        if max(sizes) > args.max_payload:
            continue
        hero_blocks = 0
        hero_params: Counter = Counter()
        for param, c in pcount.items():
            if (param & HERO_MASK) == HERO_PREFIX and 0x400000a0 <= param <= 0x400000ff:
                hero_blocks += c
                hero_params[param] = c
        if hero_blocks < 50:
            continue
        # Pure hero PIDs only — at least 50% of blocks land on hero entities
        # and at least n_heroes // 2 distinct hero entities present.
        total = sum(pcount.values())
        if hero_blocks / total < 0.4:
            continue
        if len(hero_params) < n_heroes // 2:
            continue
        candidates.append((pid, hero_blocks, hero_params, sizes, total))

    candidates.sort(key=lambda c: -c[1])

    print(f"\n{'PID':>5} {'hero_blocks':>11} {'%hero':>6} {'distinct':>8} "
          f"{'sz_min':>6} {'sz_max':>6} {'sz_med':>6} {'cadence/s':>9}")
    print("-" * 75)
    for pid, hero_blocks, hero_params, sizes, total in candidates[: args.top]:
        sizes_sorted = sorted(sizes)
        sz_med = sizes_sorted[len(sizes_sorted) // 2]
        cadence = hero_blocks / max(game_length, 1) / max(len(hero_params), 1)
        print(
            f"{pid:5d} {hero_blocks:11d} {hero_blocks/total*100:5.1f}% "
            f"{len(hero_params):8d} {min(sizes):6d} {max(sizes):6d} "
            f"{sz_med:6d} {cadence:9.2f}"
        )

    # Also show per-entity breakdown for the top 6 hero-keyed PIDs
    print("\nPer-entity block count (top 6 hero PIDs):")
    for pid, hero_blocks, hero_params, sizes, total in candidates[:6]:
        per = sorted(hero_params.items())
        items = ", ".join(f"0x{p:08x}={c}" for p, c in per)
        print(f"  pid={pid}: {items}")


if __name__ == "__main__":
    main()
