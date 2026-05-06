#!/usr/bin/env python3
"""
Search for level/CS appearing ANYWHERE in stat PID payloads.
Also: look at PID 842 repeated byte patterns to find the structure.
Try: payload size correlates with stat? Fixed bytes vs variable bytes?
"""
import struct
import json
import zstandard as zstd
from collections import defaultdict
import os

ROFL_PATH = 'data/replays/NA1-5528069928.rofl'
ORACLE_PATH = '/tmp/oracle_data.json'
GAREN_ENTITY = 0x400000b2
BLOCK_MARKERS = {0x91, 0xf1, 0xb1, 0x31, 0x11}

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

def extract_all_entity_blocks(frames, entity):
    entity_bytes = struct.pack('<I', entity)
    pid_blocks = defaultdict(list)
    for fi, fd in enumerate(frames):
        pos = 0
        while pos < len(fd):
            idx = fd.find(entity_bytes, pos)
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
                        'channel': fd[block_start + 1], 'size': size,
                    })
            pos = idx + 1
    return pid_blocks

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
                'cs': g['scores']['creepScore'], 'kills': g['scores']['kills'],
            })

    def get_stat(t, stat):
        v = None
        for snap in garen_snaps:
            if snap['time'] <= t + 1.0: v = snap[stat]
            else: break
        return v

    frames = extract_frames(ROFL_PATH)
    n_frames = len(frames)
    frame_dur = game_length / max(n_frames - 1, 1)

    pid_blocks = extract_all_entity_blocks(frames, GAREN_ENTITY)

    # ── PID 842 structural analysis ──
    print("=" * 70)
    print("PID 842: STRUCTURAL PATTERN ANALYSIS")
    print("=" * 70)

    blocks842 = sorted(pid_blocks.get(842, []), key=lambda x: x['frame'])
    if blocks842:
        # Find bytes that are CONSTANT across all blocks
        min_len = min(len(b['payload']) for b in blocks842)
        print(f"\n  {len(blocks842)} blocks, min payload size={min_len}")

        # Align all payloads to min_len and find constant positions
        constant_pos = []
        variable_pos = []
        for i in range(min_len):
            vals = set(b['payload'][i] for b in blocks842)
            if len(vals) == 1:
                constant_pos.append((i, list(vals)[0]))
            else:
                variable_pos.append((i, len(vals)))

        print(f"  Constant positions: {len(constant_pos)}/{min_len}")
        print(f"  Variable positions: {len(variable_pos)}/{min_len}")

        # Show constant byte map
        const_map = {}
        for pos, val in constant_pos:
            const_map[pos] = val

        print(f"\n  Constant bytes (offset: value):")
        for pos, val in constant_pos:
            print(f"    +{pos:3d}: 0x{val:02x}", end="")
            if (pos + 1) in [p for p, _ in constant_pos]:
                print("", end="")
            print()

        # Show variability of variable positions
        print(f"\n  Variable positions (offset: #distinct_values):")
        for pos, ndist in sorted(variable_pos, key=lambda x: -x[1])[:30]:
            # Also show sample values at this position across levels
            samples = []
            seen_lev = set()
            for b in blocks842:
                ft = b['frame'] * frame_dur
                lev = get_stat(ft, 'level')
                if lev and lev not in seen_lev and pos < len(b['payload']):
                    seen_lev.add(lev)
                    samples.append((lev, b['payload'][pos]))
            samples.sort()
            sample_str = " ".join(f"L{l}={v:02x}" for l, v in samples[:8])
            print(f"    +{pos:3d}: {ndist:3d} distinct  {sample_str}")

        # Look for the "aeae59ae" pattern
        pattern = bytes.fromhex('aeae59ae')
        for b in blocks842[:3]:
            idx = b['payload'].find(pattern)
            if idx >= 0:
                print(f"\n  Pattern 'aeae59ae' found at offset {idx}")
                break

        # Look for "f5f5f5f5" pattern
        pattern2 = bytes.fromhex('f5f5f5f5')
        for b in blocks842[:3]:
            idx = b['payload'].find(pattern2)
            if idx >= 0:
                print(f"  Pattern 'f5f5f5f5' found at offset {idx}")
                break

    # ── For ALL PIDs: search for level byte anywhere in payload ──
    print("\n" + "=" * 70)
    print("SEARCH: Level value appearing ANYWHERE in payload")
    print("=" * 70)

    for pid in sorted(pid_blocks.keys()):
        blocks = pid_blocks[pid]
        if len(blocks) < 5:
            continue

        # For each block, check if the level byte appears in the payload
        found_offsets = defaultdict(int)  # offset → count of matches
        total_with_level = 0

        for b in blocks:
            ft = b['frame'] * frame_dur
            level = get_stat(ft, 'level')
            if level is None or level < 2:  # skip level 1 (too common byte)
                continue
            total_with_level += 1

            for off in range(len(b['payload'])):
                if b['payload'][off] == level:
                    found_offsets[off] += 1

        if total_with_level < 5:
            continue

        # Check if any offset has level appearing >60% of the time
        for off, count in sorted(found_offsets.items()):
            rate = count / total_with_level
            if rate > 0.5:
                print(f"  PID {pid:4d} offset={off:3d}: level byte present {count}/{total_with_level} ({rate:.0%})")

    # ── For ALL PIDs: search for level as u16 LE anywhere ──
    print("\n" + "=" * 70)
    print("SEARCH: Level as u16 LE anywhere in payload")
    print("=" * 70)

    for pid in sorted(pid_blocks.keys()):
        blocks = pid_blocks[pid]
        if len(blocks) < 5:
            continue

        found_offsets = defaultdict(int)
        total_with_level = 0

        for b in blocks:
            ft = b['frame'] * frame_dur
            level = get_stat(ft, 'level')
            if level is None or level < 2:
                continue
            total_with_level += 1
            level_bytes = struct.pack('<H', level)

            for off in range(len(b['payload']) - 1):
                if b['payload'][off:off+2] == level_bytes:
                    found_offsets[off] += 1

        if total_with_level < 5:
            continue

        for off, count in sorted(found_offsets.items()):
            rate = count / total_with_level
            if rate > 0.4:
                print(f"  PID {pid:4d} offset={off:3d}: level u16 LE present {count}/{total_with_level} ({rate:.0%})")

    # ── Payload SIZE correlation with level ──
    print("\n" + "=" * 70)
    print("SEARCH: Payload SIZE correlation with level/CS")
    print("=" * 70)

    import numpy as np
    for pid in sorted(pid_blocks.keys()):
        blocks = pid_blocks[pid]
        if len(blocks) < 10:
            continue

        sizes = []
        levels = []
        css = []
        for b in blocks:
            ft = b['frame'] * frame_dur
            level = get_stat(ft, 'level')
            cs = get_stat(ft, 'cs')
            if level is not None:
                sizes.append(len(b['payload']))
                levels.append(level)
                css.append(cs if cs else 0)

        if len(set(sizes)) < 3 or len(set(levels)) < 3:
            continue

        r_level = np.corrcoef(sizes, levels)[0, 1]
        r_cs = np.corrcoef(sizes, css)[0, 1]
        if abs(r_level) > 0.5 or abs(r_cs) > 0.5:
            print(f"  PID {pid:4d}: size~level r={r_level:.3f}, size~cs r={r_cs:.3f} (n={len(sizes)})")

    # ── XOR adjacent blocks to find changing bits ──
    print("\n" + "=" * 70)
    print("PID 368: XOR between level transitions")
    print("=" * 70)

    blocks368 = sorted(pid_blocks.get(368, []), key=lambda x: x['frame'])
    # Find pairs where level changes
    prev = None
    prev_level = None
    for b in blocks368:
        ft = b['frame'] * frame_dur
        level = get_stat(ft, 'level')
        if level is None:
            continue
        if prev is not None and prev_level != level:
            min_l = min(len(prev['payload']), len(b['payload']))
            xor = bytes(a ^ b_ for a, b_ in zip(prev['payload'][:min_l], b['payload'][:min_l]))
            changed = [(i, prev['payload'][i], b['payload'][i]) for i in range(min_l) if xor[i] != 0]
            if len(changed) <= 15:
                print(f"\n  Level {prev_level}→{level}: {len(changed)} bytes changed")
                for i, old, new in changed:
                    print(f"    offset {i:2d}: 0x{old:02x}→0x{new:02x} ({old}→{new})")
        prev = b
        prev_level = level

    # ── PID 224: XOR between level transitions ──
    print("\n" + "=" * 70)
    print("PID 224: XOR between level transitions")
    print("=" * 70)

    blocks224 = sorted(pid_blocks.get(224, []), key=lambda x: x['frame'])
    prev = None
    prev_level = None
    for b in blocks224:
        ft = b['frame'] * frame_dur
        level = get_stat(ft, 'level')
        if level is None:
            continue
        if prev is not None and prev_level != level:
            min_l = min(len(prev['payload']), len(b['payload']))
            xor = bytes(a ^ b_ for a, b_ in zip(prev['payload'][:min_l], b['payload'][:min_l]))
            changed = [(i, prev['payload'][i], b['payload'][i]) for i in range(min_l) if xor[i] != 0]
            if len(changed) <= 15:
                print(f"\n  Level {prev_level}→{level}: {len(changed)} bytes changed")
                for i, old, new in changed:
                    print(f"    offset {i:2d}: 0x{old:02x}→0x{new:02x} ({old}→{new})")
        prev = b
        prev_level = level


if __name__ == '__main__':
    main()
