#!/usr/bin/env python3
"""HP correlation v2 — more careful about structure.

Improvements over v1:
  * Splits payloads by sub-tag (the byte at offset 1 takes a small set of
    values; treating different sub-tags as one series mixes signals).
  * Strips the leading checksum byte from consideration as a value.
  * pattern_consistency penalises values that occur only once (which were
    trivially "consistent" in v1).
  * Correlates against ALL ground-truth stats (hp, hp_max, level, gold,
    gold_total) so we can see whether a candidate is HP or something else.
  * Builds the entity↔hero assignment as a Hungarian-ish greedy match
    using the best Pearson r per (entity, hero) pair on a chosen
    (PID, sub-tag, offset, dtype).
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
    """{stat_name: {hero: (gt_arr, val_arr)}}."""
    stats = ("hp", "hp_max", "level", "gold", "gold_total")
    out: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {s: {} for s in stats}
    series: dict[str, dict[str, list[tuple[float, float]]]] = {
        s: defaultdict(list) for s in stats
    }
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
            gts = np.array([p[0] for p in pairs], dtype=np.float64)
            vals = np.array([p[1] for p in pairs], dtype=np.float64)
            out[s][name] = (gts, vals)
    return out


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 5 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def pattern_consistency_v2(values: np.ndarray, target: np.ndarray, bin_size: float) -> tuple[float, int]:
    """For values that REPEAT, how often does the same value map to the same
    target bin? Returns (consistency_fraction, n_repeated_value_observations).
    Singleton values are excluded from both numerator and denominator."""
    if len(values) < 10:
        return 0.0, 0
    bins = np.round(target / max(bin_size, 1e-9)).astype(np.int64)
    by_value: dict[float, list[int]] = defaultdict(list)
    for v, b in zip(values, bins):
        by_value[v].append(b)
    hits = 0
    total = 0
    for v, bs in by_value.items():
        if len(bs) < 2:
            continue  # singleton — skip
        c = Counter(bs).most_common(1)[0][1]
        hits += c
        total += len(bs)
    if total == 0:
        return 0.0, 0
    return hits / total, total


def extract_value(payload: bytes, offset: int, dtype: str):
    if dtype == "f32":
        if offset + 4 > len(payload):
            return None
        return struct.unpack_from("<f", payload, offset)[0]
    if dtype == "u32":
        if offset + 4 > len(payload):
            return None
        return float(struct.unpack_from("<I", payload, offset)[0])
    if dtype == "u16":
        if offset + 2 > len(payload):
            return None
        return float(struct.unpack_from("<H", payload, offset)[0])
    if dtype == "u8":
        if offset + 1 > len(payload):
            return None
        return float(payload[offset])
    return None


def analyse_pid(pid: int, hero_blocks: dict[int, list], oracles, frame_to_gt: float,
                stat_bins: dict[str, float], r_thr: float, pc_thr: float):
    """For one PID, scan (sub_tag, offset, dtype) × entity × hero × stat."""
    rows = []
    # Determine modal payload size per entity, then split by sub-tag (byte 1 value)
    for entity, blks in hero_blocks.items():
        if len(blks) < 30:
            continue
        sizes = Counter(len(b.payload) for b in blks)
        modal_size, _ = sizes.most_common(1)[0]
        same = [b for b in blks if len(b.payload) == modal_size]
        if len(same) < 30 or modal_size < 3:
            continue

        # Sub-tag = payload[1]. Split.
        by_subtag: dict[int, list] = defaultdict(list)
        for b in same:
            by_subtag[b.payload[1]].append(b)

        for sub_tag, sub_blocks in by_subtag.items():
            if len(sub_blocks) < 30:
                continue
            payloads = [b.payload for b in sub_blocks]
            frame_ids = np.array([b.frame_idx for b in sub_blocks], dtype=np.float64)
            gts = frame_ids * frame_to_gt

            # Try value extraction at every offset for several dtypes.
            for dtype, sz in [("f32", 4), ("u32", 4), ("u16", 2), ("u8", 1)]:
                for off in range(2, modal_size - sz + 1):  # skip checksum byte 0 + sub-tag byte 1
                    vals = []
                    bad = False
                    for pl in payloads:
                        v = extract_value(pl, off, dtype)
                        if v is None or (isinstance(v, float) and not np.isfinite(v)):
                            bad = True
                            break
                        vals.append(v)
                    if bad:
                        continue
                    arr = np.asarray(vals, dtype=np.float64)
                    n_unique = len(set(vals))
                    if n_unique < 4:
                        continue

                    for stat, by_hero in oracles.items():
                        bin_sz = stat_bins[stat]
                        for hero, (gt_o, val_o) in by_hero.items():
                            target = np.interp(gts, gt_o, val_o)
                            r = safe_pearson(arr, target)
                            pc, n_rep = pattern_consistency_v2(arr, target, bin_sz)
                            if abs(r) < r_thr and (pc < pc_thr or n_rep < 20):
                                continue
                            rows.append({
                                "pid": pid, "entity": entity, "sub_tag": sub_tag,
                                "modal_size": modal_size, "offset": off, "dtype": dtype,
                                "stat": stat, "hero": hero,
                                "n": len(arr), "n_unique": n_unique,
                                "n_rep": n_rep,
                                "r": r, "pc": pc,
                                "v_min": float(arr.min()), "v_max": float(arr.max()),
                            })
    return rows


def best_per_entity_for_stat(rows, stat: str, key=lambda r: max(abs(r["r"]), r["pc"])):
    """For a given stat, find the best (PID, sub_tag, offset, dtype, hero) per entity."""
    by_e: dict[int, dict] = {}
    for r in rows:
        if r["stat"] != stat:
            continue
        cur = by_e.get(r["entity"])
        if cur is None or key(r) > key(cur):
            by_e[r["entity"]] = r
    return by_e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rofl",
                    default="/mnt/nfs/datasets/lol_replays_16_9_772/NA1_5552884026/rofl/replay.rofl")
    ap.add_argument("--raw-mem",
                    default="/mnt/nfs/datasets/lol_replays_16_9_772/NA1_5552884026/raw_mem.json")
    ap.add_argument("--pids", nargs="+", type=int,
                    default=[132, 89, 758, 104, 452, 828, 781, 785])
    ap.add_argument("--r-thr", type=float, default=0.4)
    ap.add_argument("--pc-thr", type=float, default=0.6)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    raw_mem = json.load(open(args.raw_mem))
    game_length = raw_mem[-1]["gt"]
    oracles = build_oracles(raw_mem)
    print(f"raw_mem: {len(raw_mem)} ticks, {len(oracles['hp'])} heroes, "
          f"game_length={game_length:.1f}s")

    blocks = parse_rofl(args.rofl)
    n_frames = max(b.frame_idx for b in blocks) + 1
    frame_to_gt = game_length / n_frames
    print(f"rofl:    {len(blocks)} blocks across {n_frames} frames "
          f"(~{frame_to_gt:.1f}s/frame)")

    # Group: pid → entity → blocks
    grouped: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for b in blocks:
        if b.pid in args.pids and HERO_PARAM_LO <= b.param <= HERO_PARAM_HI:
            grouped[b.pid][b.param].append(b)

    # Bin sizes for pattern consistency: tolerate small fluctuations
    stat_bins = {"hp": 25.0, "hp_max": 5.0, "level": 1.0, "gold": 25.0, "gold_total": 25.0}

    all_rows = []
    for pid in args.pids:
        rows = analyse_pid(pid, grouped[pid], oracles, frame_to_gt,
                           stat_bins, args.r_thr, args.pc_thr)
        if rows:
            print(f"  PID {pid}: {len(rows)} interesting rows")
        all_rows.extend(rows)

    print(f"\ntotal interesting rows: {len(all_rows)}")

    # === Top by Pearson |r| ===
    by_r = sorted(all_rows, key=lambda r: -abs(r["r"]))
    print(f"\n{'='*110}")
    print("TOP MATCHES BY |Pearson r|")
    print(f"{'='*110}")
    print(f"{'pid':>4} {'ent':>10} {'st':>3} {'msz':>3} {'off':>3} {'dt':>3} "
          f"{'stat':>10} {'hero':>14} {'n':>5} {'unq':>4} {'|r|':>5} {'pc':>4} "
          f"{'rng':>22}")
    print("-" * 110)
    for r in by_r[: args.top]:
        print(
            f"{r['pid']:4d} 0x{r['entity']:08x} {r['sub_tag']:3d} {r['modal_size']:3d} "
            f"{r['offset']:3d} {r['dtype']:>3} {r['stat']:>10} {r['hero']:>14} "
            f"{r['n']:5d} {r['n_unique']:4d} {abs(r['r']):5.2f} {r['pc']:4.2f} "
            f"[{r['v_min']:9.1f},{r['v_max']:9.1f}]"
        )

    # === Top by pattern consistency (cipher-tolerant) ===
    by_pc = sorted(
        [r for r in all_rows if r["n_rep"] >= 30],
        key=lambda r: -(r["pc"] * np.log2(max(r["n_unique"], 2))),
    )
    print(f"\n{'='*110}")
    print("TOP MATCHES BY PATTERN CONSISTENCY (cipher-tolerant; weighted by log unique)")
    print(f"{'='*110}")
    print(f"{'pid':>4} {'ent':>10} {'st':>3} {'msz':>3} {'off':>3} {'dt':>3} "
          f"{'stat':>10} {'hero':>14} {'n':>5} {'unq':>4} {'rep':>5} {'|r|':>5} {'pc':>4}")
    print("-" * 110)
    for r in by_pc[: args.top]:
        print(
            f"{r['pid']:4d} 0x{r['entity']:08x} {r['sub_tag']:3d} {r['modal_size']:3d} "
            f"{r['offset']:3d} {r['dtype']:>3} {r['stat']:>10} {r['hero']:>14} "
            f"{r['n']:5d} {r['n_unique']:4d} {r['n_rep']:5d} {abs(r['r']):5.2f} {r['pc']:4.2f}"
        )

    # === Per-PID best HP candidate ===
    print(f"\n{'='*110}")
    print("BEST HP CANDIDATE PER (PID, sub_tag, offset, dtype) — ranked by avg(|r|, pc) across entities")
    print(f"{'='*110}")
    by_combo: dict[tuple, list] = defaultdict(list)
    for r in all_rows:
        if r["stat"] != "hp":
            continue
        # Best hero per (pid, subtag, off, dtype, entity)
        k = (r["pid"], r["sub_tag"], r["offset"], r["dtype"], r["entity"])
        by_combo[k].append(r)
    # Best hero per (pid, subtag, off, dtype) by averaging best per entity
    combo_scores: dict[tuple, list] = defaultdict(list)
    combo_best_hero: dict[tuple, dict[int, str]] = defaultdict(dict)
    for k, rs in by_combo.items():
        best = max(rs, key=lambda r: max(abs(r["r"]), r["pc"]))
        ck = k[:4]
        combo_scores[ck].append(max(abs(best["r"]), best["pc"]))
        combo_best_hero[ck][k[4]] = best["hero"]
    for ck, scores in sorted(combo_scores.items(),
                              key=lambda kv: -(np.mean(kv[1]) if kv[1] else 0)):
        if len(scores) < 4:
            continue
        avg = float(np.mean(scores))
        if avg < 0.4:
            continue
        pid, st, off, dt = ck
        hm = combo_best_hero[ck]
        hm_str = ", ".join(f"0x{e:x}→{h}" for e, h in sorted(hm.items()))
        print(f"  pid={pid} st={st} off={off} {dt} avg={avg:.2f} n_ent={len(scores)}  {hm_str}")


if __name__ == "__main__":
    main()
