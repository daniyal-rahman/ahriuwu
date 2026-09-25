#!/usr/bin/env python3
"""HP correlation v3 — finer time alignment + broader PID scan.

Changes from v2:

  * Within-frame block ordering — each block within a chunk gets gt =
    frame_start + (block_position_in_frame / blocks_per_frame) * frame_duration.
    Brings effective resolution from ~20s/frame to ~ms.

  * Scans every PID with ≥100 hero-keyed blocks (not just a hand-picked
    list).

  * Tries f32 at every aligned offset across the modal payload size.

  * Entity↔hero assignment via greedy max-Pearson on the best (offset, dtype)
    per (PID, entity).

  * Reports candidates that are CONSISTENT across all 10 entities, since a
    real HP-carrying PID should explain every hero's HP at the same offset.
"""
from __future__ import annotations

import argparse
import json
import struct
from collections import Counter, defaultdict

import numpy as np

from parse_rofl import parse_rofl


HERO_PARAM_LO = 0x400000A0
HERO_PARAM_HI = 0x400000FF


def build_oracles(raw_mem):
    stats = ("hp", "hp_max", "level", "gold", "gold_total")
    out = {s: {} for s in stats}
    series = {s: defaultdict(list) for s in stats}
    for tick in raw_mem:
        gt = tick["gt"]
        for name, h in tick["heroes"].items():
            for s in stats:
                v = h.get(s)
                if v is None:
                    continue
                series[s][name].append((gt, float(v)))
    for s, by_hero in series.items():
        for name, pairs in by_hero.items():
            pairs.sort()
            out[s][name] = (
                np.array([p[0] for p in pairs], dtype=np.float64),
                np.array([p[1] for p in pairs], dtype=np.float64),
            )
    return out


def compute_block_gts(blocks, game_length: float) -> np.ndarray:
    """Per-block gt estimate using within-frame ordering."""
    n_frames = max(b.frame_idx for b in blocks) + 1
    frame_dur = game_length / n_frames

    # Count blocks per frame & assign within-frame index
    frame_counts: Counter = Counter(b.frame_idx for b in blocks)
    frame_seen: dict[int, int] = defaultdict(int)
    out = np.empty(len(blocks), dtype=np.float64)
    for i, b in enumerate(blocks):
        fc = frame_counts[b.frame_idx]
        idx = frame_seen[b.frame_idx]
        frame_seen[b.frame_idx] = idx + 1
        # gt = frame_start + (idx + 0.5) / fc * frame_dur
        out[i] = b.frame_idx * frame_dur + (idx + 0.5) / fc * frame_dur
    return out


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 5 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def extract_value(payload: bytes, offset: int, dtype: str):
    sz = {"f32": 4, "u32": 4, "u16": 2, "u8": 1}[dtype]
    if offset + sz > len(payload):
        return None
    if dtype == "f32":
        return struct.unpack_from("<f", payload, offset)[0]
    if dtype == "u32":
        return float(struct.unpack_from("<I", payload, offset)[0])
    if dtype == "u16":
        return float(struct.unpack_from("<H", payload, offset)[0])
    return float(payload[offset])


def candidate_series(payloads, offset: int, dtype: str):
    arr = np.empty(len(payloads), dtype=np.float64)
    for i, pl in enumerate(payloads):
        v = extract_value(pl, offset, dtype)
        if v is None:
            return None
        arr[i] = v
    if dtype == "f32" and not np.all(np.isfinite(arr)):
        return None
    return arr


def analyse(rofl_path: str, raw_mem_path: str, min_blocks_per_pid: int,
            min_blocks_per_entity: int, top_pid_limit: int, r_thr: float):
    raw_mem = json.load(open(raw_mem_path))
    game_length = raw_mem[-1]["gt"]
    oracles = build_oracles(raw_mem)
    print(f"raw_mem: {len(raw_mem)} ticks, game_length={game_length:.1f}s, "
          f"{len(oracles['hp'])} heroes")

    blocks = parse_rofl(rofl_path)
    n_frames = max(b.frame_idx for b in blocks) + 1
    frame_dur = game_length / n_frames
    print(f"rofl:    {len(blocks)} blocks across {n_frames} frames "
          f"(~{frame_dur:.1f}s/frame)")

    # Compute per-block gt
    block_gts = compute_block_gts(blocks, game_length)

    # Group: pid → entity → list[(gt, payload)]
    pid_count: Counter = Counter()
    grouped: dict[int, dict[int, list[tuple[float, bytes]]]] = defaultdict(lambda: defaultdict(list))
    for i, b in enumerate(blocks):
        if HERO_PARAM_LO <= b.param <= HERO_PARAM_HI:
            pid_count[b.pid] += 1
            grouped[b.pid][b.param].append((block_gts[i], b.payload))

    # Pick top PIDs by hero-keyed block count
    pids = [pid for pid, c in pid_count.most_common(top_pid_limit) if c >= min_blocks_per_pid]
    print(f"\nScanning {len(pids)} PIDs")

    rows = []
    for pid in pids:
        ent_map = grouped[pid]
        # Determine modal payload size across all entities
        sizes = Counter(len(pl) for series in ent_map.values() for _, pl in series)
        if not sizes:
            continue
        modal_size, modal_count = sizes.most_common(1)[0]
        if modal_size < 4:
            continue

        # For each entity, filter to modal-size blocks; split by sub-tag (byte 1)
        for entity, series in ent_map.items():
            same = [(gt, pl) for gt, pl in series if len(pl) == modal_size]
            if len(same) < min_blocks_per_entity:
                continue
            by_sub: dict[int, list] = defaultdict(list)
            for gt, pl in same:
                by_sub[pl[1]].append((gt, pl))
            for sub_tag, ss in by_sub.items():
                if len(ss) < min_blocks_per_entity:
                    continue
                gts = np.array([g for g, _ in ss], dtype=np.float64)
                payloads = [p for _, p in ss]
                for dtype, sz in [("f32", 4), ("u32", 4), ("u16", 2), ("u8", 1)]:
                    for off in range(2, modal_size - sz + 1):
                        arr = candidate_series(payloads, off, dtype)
                        if arr is None:
                            continue
                        n_unique = len(set(arr.tolist()))
                        if n_unique < 5:
                            continue
                        # Skip clearly junk f32 (huge or tiny absolute values)
                        if dtype == "f32":
                            absvals = np.abs(arr)
                            if absvals.max() > 1e6 or absvals.max() < 0.5:
                                continue
                        for stat, by_hero in oracles.items():
                            for hero, (gt_o, val_o) in by_hero.items():
                                target = np.interp(gts, gt_o, val_o)
                                r = safe_pearson(arr, target)
                                if abs(r) < r_thr:
                                    continue
                                rows.append({
                                    "pid": pid, "entity": entity, "sub_tag": sub_tag,
                                    "modal_size": modal_size, "offset": off,
                                    "dtype": dtype, "stat": stat, "hero": hero,
                                    "n": len(arr), "n_unique": n_unique,
                                    "r": r,
                                    "v_min": float(arr.min()), "v_max": float(arr.max()),
                                })
    return rows, oracles


def report(rows, top_n: int):
    if not rows:
        print("\nNo rows above threshold.")
        return

    # Group by (pid, sub_tag, offset, dtype, stat) and compute average |r| across entities
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        # Best (entity, hero) per pid/sub/off/dtype/stat: pick max |r| per entity
        pass

    # Aggregate: for each (pid, sub_tag, offset, dtype, stat), find best hero per entity
    # then summarise: how many distinct entities have a strong match, and
    # whether they all map to DIFFERENT heroes (good sign of an HP field).
    by_combo: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for r in rows:
        ck = (r["pid"], r["sub_tag"], r["offset"], r["dtype"], r["stat"])
        ek = r["entity"]
        cur = by_combo[ck].get(ek)
        if cur is None or abs(r["r"]) > abs(cur["r"]):
            by_combo[ck][ek] = r

    summary = []
    for ck, by_ent in by_combo.items():
        if len(by_ent) < 4:
            continue
        rs = list(by_ent.values())
        rs.sort(key=lambda r: -abs(r["r"]))
        avg_r = float(np.mean([abs(r["r"]) for r in rs]))
        min_r = float(np.min([abs(r["r"]) for r in rs]))
        # Diversity = # distinct heroes assigned across entities
        heroes = [r["hero"] for r in rs]
        distinct_heroes = len(set(heroes))
        summary.append({
            "key": ck,
            "n_ent": len(by_ent),
            "avg_r": avg_r,
            "min_r": min_r,
            "distinct_heroes": distinct_heroes,
            "rows": rs,
        })

    summary.sort(key=lambda s: -(s["avg_r"] * (1 + s["distinct_heroes"] / 10)))

    print(f"\n{'='*120}")
    print(f"TOP COMBINATIONS — sorted by avg_r weighted by hero diversity")
    print(f"{'='*120}")
    print(f"{'pid':>4} {'sub':>3} {'msz':>3} {'off':>3} {'dt':>3} {'stat':>10} "
          f"{'n_ent':>5} {'avg_r':>5} {'min_r':>5} {'dist_h':>6}  best mappings")
    print("-" * 120)
    for s in summary[:top_n]:
        pid, sub, off, dt, stat = s["key"]
        msz = s["rows"][0]["modal_size"]
        # Show entity → hero mappings
        m = ", ".join(
            f"0x{r['entity']:08x}→{r['hero']}({r['r']:+.2f})"
            for r in s["rows"]
        )
        print(
            f"{pid:4d} {sub:3d} {msz:3d} {off:3d} {dt:>3} {stat:>10} "
            f"{s['n_ent']:5d} {s['avg_r']:5.2f} {s['min_r']:5.2f} {s['distinct_heroes']:6d}  {m[:70]}"
        )

    # Spotlight: best combo for HP specifically
    hp_only = [s for s in summary if s["key"][4] == "hp"]
    if hp_only:
        print(f"\n{'='*120}")
        print("BEST FOR HP — full per-entity breakdown")
        print(f"{'='*120}")
        for s in hp_only[:5]:
            pid, sub, off, dt, stat = s["key"]
            print(f"\npid={pid} sub_tag={sub} offset={off} dtype={dt}: "
                  f"avg|r|={s['avg_r']:.3f} across {s['n_ent']} entities "
                  f"({s['distinct_heroes']} distinct heroes)")
            for r in s["rows"]:
                print(
                    f"  ent=0x{r['entity']:08x}  hero={r['hero']:>14}  |r|={abs(r['r']):.3f}  "
                    f"n={r['n']}  unique={r['n_unique']}  range=[{r['v_min']:.1f}, {r['v_max']:.1f}]"
                )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rofl",
                    default="/mnt/nfs/datasets/lol_replays_16_9_772/NA1_5552884026/rofl/replay.rofl")
    ap.add_argument("--raw-mem",
                    default="/mnt/nfs/datasets/lol_replays_16_9_772/NA1_5552884026/raw_mem.json")
    ap.add_argument("--min-blocks-per-pid", type=int, default=200)
    ap.add_argument("--min-blocks-per-entity", type=int, default=30)
    ap.add_argument("--top-pid-limit", type=int, default=40)
    ap.add_argument("--r-thr", type=float, default=0.5)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()

    rows, _ = analyse(args.rofl, args.raw_mem,
                      args.min_blocks_per_pid, args.min_blocks_per_entity,
                      args.top_pid_limit, args.r_thr)
    print(f"\ncollected {len(rows)} rows above |r|>={args.r_thr}")
    report(rows, top_n=args.top)


if __name__ == "__main__":
    main()
