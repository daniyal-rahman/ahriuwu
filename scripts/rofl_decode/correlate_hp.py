#!/usr/bin/env python3
"""Brute-force HP-correlation across PIDs / offsets / entity↔hero mappings.

Strategy
--------
For each candidate hero-keyed PID, for each entity that PID targets, for each
4-byte aligned offset in the payload, extract a time series of values
interpreted as f32 / u32 / u16. Correlate each series against every hero's
ground-truth `hp_current` from raw_mem.json. Report the strongest matches.

Two correlation modes:

  R²  — Pearson r squared on direct values. Catches the easy case of
        plaintext f32 / u32 (or any monotone-preserving cipher).

  PM  — Pattern matching on encrypted values: how many distinct values
        each map to a consistent HP bin. Catches deterministic per-position
        ciphers where same plaintext → same ciphertext but the value space
        is permuted.

The rofl has no per-block timestamp we trust, so we approximate gt with
frame_idx × (game_length / n_frames). Coarse but adequate for HP, which
varies on a multi-second scale."""
from __future__ import annotations

import argparse
import json
import struct
from collections import Counter, defaultdict

import numpy as np

from parse_rofl import parse_rofl


HERO_MASK = 0xFFFFFF00
HERO_PREFIX = 0x40000000


def _is_hero_param(p: int) -> bool:
    return (p & HERO_MASK) == HERO_PREFIX and 0x400000a0 <= p <= 0x400000ff


def build_hp_oracle(raw_mem):
    """Return {hero_name: (gt_array, hp_array)} sorted by gt."""
    hero_series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for tick in raw_mem:
        gt = tick["gt"]
        for name, h in tick["heroes"].items():
            hp = h.get("hp")
            if hp is None:
                continue
            hero_series[name].append((gt, hp))
    out = {}
    for name, pairs in hero_series.items():
        pairs.sort()
        gts = np.array([p[0] for p in pairs], dtype=np.float64)
        hps = np.array([p[1] for p in pairs], dtype=np.float64)
        out[name] = (gts, hps)
    return out


def interp_hp(gts_query: np.ndarray, gt_oracle: np.ndarray, hp_oracle: np.ndarray) -> np.ndarray:
    """Look up HP at query times. Outside the oracle range we extrapolate
    with edge-clamp; HP is bounded so this is fine."""
    return np.interp(gts_query, gt_oracle, hp_oracle)


def candidate_value_series(payloads: list[bytes], offset: int, dtype: str) -> np.ndarray | None:
    """Extract value at (offset, dtype) from each payload. Returns None if any
    payload is too short."""
    if dtype == "f32":
        sz = 4
        unpack = lambda b: struct.unpack_from("<f", b, offset)[0]
    elif dtype == "u32":
        sz = 4
        unpack = lambda b: struct.unpack_from("<I", b, offset)[0]
    elif dtype == "u16":
        sz = 2
        unpack = lambda b: struct.unpack_from("<H", b, offset)[0]
    else:
        return None
    out = np.empty(len(payloads), dtype=np.float64)
    for i, pl in enumerate(payloads):
        if len(pl) < offset + sz:
            return None
        try:
            out[i] = float(unpack(pl))
        except Exception:
            return None
    # Drop NaN/inf for f32
    if dtype == "f32":
        if not np.all(np.isfinite(out)):
            return None
    return out


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson r, NaN-safe. Returns 0.0 if either side has zero variance."""
    if len(x) < 5 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def pattern_consistency(values: np.ndarray, hp: np.ndarray, hp_bin: float = 50.0) -> float:
    """How well does each unique `values[i]` map to a consistent HP bin?
    Returns the fraction of (value, bin) pairs that hit the modal bin for
    that value, weighted by occurrences. 1.0 = perfect deterministic cipher."""
    if len(values) < 10:
        return 0.0
    bins = np.round(hp / hp_bin).astype(np.int64)
    by_value: dict[float, list[int]] = defaultdict(list)
    for v, b in zip(values, bins):
        by_value[v].append(b)
    weighted_hits = 0
    weighted_total = 0
    for v, bs in by_value.items():
        if len(bs) < 2:
            continue
        c = Counter(bs).most_common(1)[0][1]
        weighted_hits += c
        weighted_total += len(bs)
    if weighted_total == 0:
        return 0.0
    return weighted_hits / weighted_total


def analyse(rofl_path: str, raw_mem_path: str, top_pids: list[int], min_blocks: int,
            max_payload: int):
    raw_mem = json.load(open(raw_mem_path))
    hp_oracle = build_hp_oracle(raw_mem)
    game_length = raw_mem[-1]["gt"]
    print(f"raw_mem: {len(raw_mem)} ticks, game_length={game_length:.1f}s, "
          f"{len(hp_oracle)} heroes")

    blocks = parse_rofl(rofl_path)
    n_frames = max(b.frame_idx for b in blocks) + 1
    gt_per_frame = game_length / n_frames
    print(f"rofl:    {len(blocks)} blocks across {n_frames} frames "
          f"(~{gt_per_frame:.1f}s/frame)")

    # Group: pid → entity → list[(frame_idx, payload)]
    grouped: dict[int, dict[int, list[tuple[int, bytes]]]] = defaultdict(lambda: defaultdict(list))
    for b in blocks:
        if b.pid not in top_pids:
            continue
        if not _is_hero_param(b.param):
            continue
        if len(b.payload) > max_payload:
            continue
        grouped[b.pid][b.param].append((b.frame_idx, b.payload))

    results = []
    hero_names = list(hp_oracle.keys())

    for pid in top_pids:
        ent_map = grouped.get(pid, {})
        if not ent_map:
            continue
        # Find consistent payload size for this PID — pick the most common
        all_sizes = Counter(len(pl) for series in ent_map.values() for _, pl in series)
        modal_size, _ = all_sizes.most_common(1)[0]

        for entity, series in ent_map.items():
            if len(series) < min_blocks:
                continue
            # Filter to modal size only — mixing sizes muddies offset semantics
            series_f = [(fi, pl) for fi, pl in series if len(pl) == modal_size]
            if len(series_f) < min_blocks:
                continue

            frame_ids = np.array([fi for fi, _ in series_f], dtype=np.float64)
            gts = frame_ids * gt_per_frame
            payloads = [pl for _, pl in series_f]

            for dtype, sz in [("f32", 4), ("u32", 4), ("u16", 2)]:
                for off in range(0, modal_size - sz + 1):
                    vals = candidate_value_series(payloads, off, dtype)
                    if vals is None:
                        continue
                    if len(set(vals.tolist())) < 5:
                        continue  # no meaningful variation

                    for hero_name in hero_names:
                        gt_o, hp_o = hp_oracle[hero_name]
                        hp_at = interp_hp(gts, gt_o, hp_o)
                        r = safe_pearson(vals, hp_at)
                        pc = pattern_consistency(vals, hp_at)

                        # Filter for interesting matches only
                        if abs(r) < 0.4 and pc < 0.5:
                            continue
                        results.append({
                            "pid": pid,
                            "entity": entity,
                            "offset": off,
                            "dtype": dtype,
                            "hero": hero_name,
                            "n": len(vals),
                            "r": r,
                            "pc": pc,
                            "v_min": float(vals.min()),
                            "v_max": float(vals.max()),
                            "v_unique": len(set(vals.tolist())),
                        })

    return results


def report(results, top_n: int = 30):
    # Sort by max(|r|, pc) descending
    def key(d):
        return max(abs(d["r"]), d["pc"])
    results = sorted(results, key=key, reverse=True)

    print(f"\n{'='*100}")
    print(f"TOP MATCHES — sorted by max(|r|, pc)  (showing {top_n})")
    print(f"{'='*100}")
    print(f"{'pid':>5} {'entity':>10} {'off':>3} {'dt':>3} {'hero':>14} "
          f"{'n':>5} {'|r|':>6} {'pc':>5} {'unique':>6} {'value range':>22}")
    print("-" * 100)
    for d in results[:top_n]:
        print(
            f"{d['pid']:5d} 0x{d['entity']:08x} {d['offset']:3d} {d['dtype']:>3} "
            f"{d['hero']:>14} {d['n']:5d} {abs(d['r']):6.3f} {d['pc']:5.2f} "
            f"{d['v_unique']:6d} [{d['v_min']:9.1f}, {d['v_max']:9.1f}]"
        )

    # Also: per-PID best, to see whether HP lives in one PID
    print(f"\n{'='*100}")
    print("BEST PER (pid, entity)  — how strongly does any (offset, dtype) map "
          "to *some* hero's HP")
    print(f"{'='*100}")
    by_pe: dict[tuple[int, int], dict] = {}
    for d in results:
        k = (d["pid"], d["entity"])
        prev = by_pe.get(k)
        if prev is None or key(d) > key(prev):
            by_pe[k] = d
    for k in sorted(by_pe.keys(), key=lambda kk: -key(by_pe[kk]))[:30]:
        d = by_pe[k]
        print(
            f"  pid={d['pid']:4d} ent=0x{d['entity']:08x}  →  "
            f"hero={d['hero']:>14}  off={d['offset']:2d} {d['dtype']} "
            f"|r|={abs(d['r']):.3f}  pc={d['pc']:.2f}  n={d['n']}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rofl",
                    default="/mnt/nfs/datasets/lol_replays_16_9_772/NA1_5552884026/rofl/replay.rofl")
    ap.add_argument("--raw-mem",
                    default="/mnt/nfs/datasets/lol_replays_16_9_772/NA1_5552884026/raw_mem.json")
    ap.add_argument("--pids", nargs="+", type=int,
                    default=[132, 89, 758, 104, 452, 828, 785, 1104, 781, 1196, 1045, 306])
    ap.add_argument("--min-blocks", type=int, default=30)
    ap.add_argument("--max-payload", type=int, default=64)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    results = analyse(args.rofl, args.raw_mem, args.pids,
                      args.min_blocks, args.max_payload)
    report(results, top_n=args.top)


if __name__ == "__main__":
    main()
