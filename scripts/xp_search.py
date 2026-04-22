#!/usr/bin/env python3
"""
Search for XP values in ROFL frames around entity b2 markers.

LoL XP-to-level thresholds (cumulative XP needed for each level):
Level 2: 280, Level 3: 660, Level 4: 1140, Level 5: 1720,
Level 6: 2400, Level 7: 3180, Level 8: 4060, Level 9: 5040,
Level 10: 6120, Level 11: 7300, Level 12: 8580, Level 13: 9960,
Level 14: 11440, Level 15: 13020, Level 16: 14700, Level 17: 16480, Level 18: 18360

Search approaches:
1. u16 LE XP values in separator-format records (raw frame search for entity b2 marker)
2. f32 XP values in block payloads
3. u16/u32 XP values in block payloads
"""
import struct
import json
import zstandard as zstd
from collections import defaultdict
import os
import numpy as np

ROFL_PATH = 'data/replays/NA1-5528069928.rofl'
ORACLE_PATH = '/tmp/oracle_data.json'
GAREN_ENTITY_BYTES = b'\xb2\x00\x00\x40'
BLOCK_MARKERS = {0x91, 0xf1, 0xb1, 0x31, 0x11}

# Cumulative XP thresholds for levels 2-18
XP_THRESHOLDS = {
    1: 0, 2: 280, 3: 660, 4: 1140, 5: 1720, 6: 2400, 7: 3180,
    8: 4060, 9: 5040, 10: 6120, 11: 7300, 12: 8580, 13: 9960,
    14: 11440, 15: 13020, 16: 14700, 17: 16480, 18: 18360
}

def level_to_xp_range(level):
    """Return (min_xp, max_xp) for a given level.
    min = threshold for that level, max = threshold for next level - 1."""
    min_xp = XP_THRESHOLDS.get(level, 0)
    max_xp = XP_THRESHOLDS.get(level + 1, 99999) - 1
    return min_xp, max_xp

def extract_frames(rofl_path):
    with open(rofl_path, 'rb') as f:
        data = f.read()
    dctx = zstd.ZstdDecompressor()
    frames = []
    pos = 0
    while pos < len(data):
        idx = data.find(b'\x28\xb5\x2f\xfd', pos)
        if idx < 0:
            break
        try:
            dec = dctx.decompress(data[idx:idx + 0x100000], max_output_size=0x400000)
            frames.append(dec)
        except:
            pass
        pos = idx + 1
    return frames

def main():
    with open(ORACLE_PATH) as f:
        oracle = json.load(f)
    game_length = oracle['game_length']
    garen_snaps = []
    for snap in oracle['snapshots']:
        if 'Garen' in snap['players']:
            g = snap['players']['Garen']
            garen_snaps.append({
                'time': snap['actual_time'], 'level': g['level'],
                'cs': g['scores']['creepScore'],
            })

    def get_stat(t, stat):
        v = None
        for snap in garen_snaps:
            if snap['time'] <= t + 1.0: v = snap[stat]
            else: break
        return v

    print("Extracting frames...")
    frames = extract_frames(ROFL_PATH)
    n_frames = len(frames)
    frame_dur = game_length / max(n_frames - 1, 1)
    print(f"{n_frames} frames, {frame_dur:.1f}s per frame")

    # ══════════════════════════════════════════════════════
    # PART 1: Separator format - search for entity b2 in raw bytes
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 1: SEPARATOR FORMAT - entity b2 in raw frame bytes")
    print("=" * 70)

    # Find all occurrences of b2 00 00 40 in raw frames
    all_hits = []
    for fi, fd in enumerate(frames):
        pos = 0
        while pos < len(fd):
            idx = fd.find(GAREN_ENTITY_BYTES, pos)
            if idx < 0:
                break
            # Read context: 20 bytes before, 40 bytes after
            start = max(0, idx - 20)
            end = min(len(fd), idx + 4 + 40)
            context = fd[start:end]
            entity_offset_in_context = idx - start
            all_hits.append({
                'frame': fi,
                'pos': idx,
                'context': context,
                'ent_off': entity_offset_in_context,
            })
            pos = idx + 1

    print(f"Found {len(all_hits)} entity b2 markers across {n_frames} frames")

    # For each hit, read u16 values at various offsets relative to entity marker
    # The 4 bytes before entity marker are (size_u16, pid_u16) according to lead
    # But actually it's the block header: marker(1) + channel(1) + size(1) + PID(2) + entity(4)
    # So entity is at +5 from block start

    # Let's read u16 values at offsets -20 to +36 from entity marker
    # and check if any consistently match XP range for the known level
    print("\nSearching u16 values near entity markers for XP correlation...")

    offset_xp_matches = defaultdict(list)  # offset → [(frame, u16_val, level, xp_min, xp_max)]

    for hit in all_hits:
        ft = hit['frame'] * frame_dur
        level = get_stat(ft, 'level')
        if level is None:
            continue
        xp_min, xp_max = level_to_xp_range(level)

        # Read u16 at various offsets from entity marker position in raw frame
        fd = frames[hit['frame']]
        entity_pos = hit['pos']

        for off in range(-20, 38, 2):
            abs_pos = entity_pos + off
            if abs_pos >= 0 and abs_pos + 2 <= len(fd):
                v16 = struct.unpack_from('<H', fd, abs_pos)[0]
                if xp_min <= v16 <= xp_max:
                    offset_xp_matches[off].append((hit['frame'], v16, level, xp_min, xp_max))

    # Report offsets with high match rate
    total_with_level = sum(1 for h in all_hits if get_stat(h['frame'] * frame_dur, 'level') is not None)
    print(f"Total hits with known level: {total_with_level}")

    for off in sorted(offset_xp_matches.keys()):
        entries = offset_xp_matches[off]
        rate = len(entries) / total_with_level
        if rate > 0.3:
            print(f"\n  Offset {off:+3d}: {len(entries)}/{total_with_level} ({rate:.0%}) values in XP range")
            # Show samples grouped by level
            by_level = defaultdict(list)
            for fi, v16, lev, xmin, xmax in entries:
                by_level[lev].append(v16)
            for lev in sorted(by_level.keys()):
                vals = by_level[lev]
                xmin, xmax = level_to_xp_range(lev)
                print(f"    L={lev:2d} (XP {xmin}-{xmax}): {sorted(set(vals))[:8]} (n={len(vals)})")

    # ══════════════════════════════════════════════════════
    # PART 2: Search for XP as f32 in block payloads
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 2: XP as f32 in block payloads (all PIDs)")
    print("=" * 70)

    # Extract blocks
    pid_blocks = defaultdict(list)
    for fi, fd in enumerate(frames):
        pos = 0
        while pos < len(fd):
            idx = fd.find(GAREN_ENTITY_BYTES, pos)
            if idx < 0:
                break
            block_start = idx - 5
            if block_start >= 0 and fd[block_start] in BLOCK_MARKERS:
                size = fd[block_start + 2]
                pid = struct.unpack_from('<H', fd, block_start + 3)[0]
                end = block_start + 9 + size
                if end <= len(fd) and size > 0:
                    pid_blocks[pid].append({
                        'frame': fi, 'payload': bytes(fd[block_start + 9:end]),
                        'size': size,
                    })
            pos = idx + 1

    for pid in sorted(pid_blocks.keys()):
        blocks = pid_blocks[pid]
        if len(blocks) < 5:
            continue

        max_payload = max(len(b['payload']) for b in blocks)
        if max_payload < 4:
            continue

        # For each 4-byte offset, read as f32 and check if in XP range for level
        for off in range(max_payload - 3):
            pairs = []
            for b in blocks:
                if off + 3 < len(b['payload']):
                    ft = b['frame'] * frame_dur
                    level = get_stat(ft, 'level')
                    if level is None or level < 2:
                        continue
                    v32 = struct.unpack_from('<I', b['payload'], off)[0]
                    try:
                        fval = struct.unpack('<f', struct.pack('<I', v32))[0]
                    except:
                        continue
                    if fval != fval:  # NaN
                        continue
                    xp_min, xp_max = level_to_xp_range(level)
                    pairs.append((fval, level, xp_min, xp_max))

            if len(pairs) < 5:
                continue

            matches = sum(1 for fval, lev, xmin, xmax in pairs if xmin <= fval <= xmax)
            rate = matches / len(pairs)
            if rate > 0.5:
                print(f"  PID {pid:4d} offset={off:3d} f32: {matches}/{len(pairs)} ({rate:.0%}) in XP range")
                for fval, lev, xmin, xmax in pairs[:8]:
                    hit = "Y" if xmin <= fval <= xmax else "N"
                    print(f"    L={lev:2d} f32={fval:.1f} range=({xmin},{xmax}) {hit}")

    # ══════════════════════════════════════════════════════
    # PART 3: Search for XP as u16 in block payloads
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 3: XP as u16 in block payloads (all PIDs)")
    print("=" * 70)

    for pid in sorted(pid_blocks.keys()):
        blocks = pid_blocks[pid]
        if len(blocks) < 5:
            continue

        max_payload = max(len(b['payload']) for b in blocks)
        if max_payload < 2:
            continue

        for off in range(max_payload - 1):
            pairs = []
            for b in blocks:
                if off + 1 < len(b['payload']):
                    ft = b['frame'] * frame_dur
                    level = get_stat(ft, 'level')
                    if level is None or level < 2:
                        continue
                    v16 = struct.unpack_from('<H', b['payload'], off)[0]
                    xp_min, xp_max = level_to_xp_range(level)
                    pairs.append((v16, level, xp_min, xp_max))

            if len(pairs) < 5:
                continue

            matches = sum(1 for v16, lev, xmin, xmax in pairs if xmin <= v16 <= xmax)
            rate = matches / len(pairs)
            if rate > 0.5:
                print(f"  PID {pid:4d} offset={off:3d} u16: {matches}/{len(pairs)} ({rate:.0%}) in XP range")
                # Check if values increase monotonically with level
                by_level = defaultdict(list)
                for v16, lev, xmin, xmax in pairs:
                    by_level[lev].append(v16)
                means = [(lev, np.mean(vals)) for lev, vals in sorted(by_level.items())]
                if len(means) >= 3:
                    monotonic = all(means[i][1] <= means[i+1][1] for i in range(len(means)-1))
                    if monotonic:
                        print(f"    *** MONOTONICALLY INCREASING with level! ***")
                        for lev, mean_val in means:
                            xmin, xmax = level_to_xp_range(lev)
                            print(f"    L={lev:2d} mean={mean_val:.0f} range=({xmin},{xmax})")

    # ══════════════════════════════════════════════════════
    # PART 4: u16 values near entity markers - check ALL values, correlate with level
    # ══════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PART 4: Correlation of u16 values near entity markers with level")
    print("=" * 70)

    for off in range(-20, 40, 2):
        vals = []
        levels = []
        for hit in all_hits:
            ft = hit['frame'] * frame_dur
            level = get_stat(ft, 'level')
            if level is None:
                continue
            fd = frames[hit['frame']]
            abs_pos = hit['pos'] + off
            if abs_pos >= 0 and abs_pos + 2 <= len(fd):
                v16 = struct.unpack_from('<H', fd, abs_pos)[0]
                vals.append(v16)
                levels.append(level)

        if len(vals) < 20 and len(set(levels)) < 3:
            continue

        try:
            r = np.corrcoef(vals, levels)[0, 1]
            if abs(r) > 0.5:
                print(f"  Offset {off:+3d}: r={r:.4f} (n={len(vals)})")
                by_level = defaultdict(list)
                for v, l in zip(vals, levels):
                    by_level[l].append(v)
                for lev in sorted(by_level.keys())[:6]:
                    vs = by_level[lev]
                    print(f"    L={lev:2d}: mean={np.mean(vs):.0f} std={np.std(vs):.0f} n={len(vs)}")
        except:
            pass


if __name__ == '__main__':
    main()
