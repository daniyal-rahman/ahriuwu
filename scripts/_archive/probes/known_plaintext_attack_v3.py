#!/usr/bin/env python3
"""
Known-plaintext attack v3: Deep structural analysis.
1. Full-frame raw byte scanning (bypass block parser entirely)
2. PID 842 structural analysis (entropy, byte-pair combinations)
3. Strict injective permutation test
4. Try all entities (not just b2) to verify entity mapping
5. Use PID-targeted search (scan for PID bytes anywhere in frame)
"""
import struct
import json
from collections import defaultdict
import zstandard as zstd
import numpy as np

ROFL_PATH = 'data/replays/NA1-5528069928.rofl'
ORACLE_PATH = '/tmp/oracle_data.json'
PID_MAP_PATH = '/tmp/pid_full_map.json'
RDATA_PATH = '/tmp/pe_dump_16.7/rdata.bin'
GAREN_ENTITY = 0x400000b2
BLOCK_MARKERS = {0x91, 0xf1, 0xb1, 0x31, 0x11}
PLAYER_ID_START = 0x400000ae


def load_oracle(oracle_path):
    with open(oracle_path) as f:
        data = json.load(f)
    results = []
    for snap in data['snapshots']:
        if 'Garen' in snap['players']:
            g = snap['players']['Garen']
            results.append({
                'time': snap['actual_time'],
                'level': g['level'],
                'cs': g['scores']['creepScore'],
                'kills': g['scores']['kills'],
            })
    return results, data['game_length']


def load_valid_pids(pid_map_path):
    with open(pid_map_path) as f:
        pid_map = json.load(f)
    return set(int(k) for k in pid_map.keys())


def parse_rofl_frames(rofl_path):
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


def get_stat_at_time(oracle_data, t, stat_name):
    val = None
    for snap in oracle_data:
        if snap['time'] <= t + 1.0:
            val = snap[stat_name]
        else:
            break
    return val


def extract_blocks_targeted(frame_data, pid, valid_pids):
    """PID-targeted block extraction: scan for PID bytes anywhere in frame."""
    fd = frame_data
    pid_bytes = struct.pack('<H', pid)
    blocks = []
    pos = 3  # PID is at offset 3 in the 9-byte header
    while pos < len(fd) - 6:
        idx = fd.find(pid_bytes, pos)
        if idx < 0:
            break
        block_start = idx - 3
        if block_start >= 0 and fd[block_start] in BLOCK_MARKERS:
            size = fd[block_start + 2]
            param = struct.unpack_from('<I', fd, block_start + 5)[0]
            end = block_start + 9 + size
            if end <= len(fd) and size > 0:
                # Validate chain
                valid_chain = (end >= len(fd) - 9 or
                              (end < len(fd) and fd[end] in BLOCK_MARKERS))
                if valid_chain:
                    blocks.append({
                        'pid': pid,
                        'param': param,
                        'payload': bytes(fd[block_start + 9:end]),
                        'marker': fd[block_start],
                        'channel': fd[block_start + 1],
                        'size': size,
                    })
        pos = idx + 1
    return blocks


def main():
    oracle_data, game_length = load_oracle(ORACLE_PATH)
    valid_pids = load_valid_pids(PID_MAP_PATH)
    frames = parse_rofl_frames(ROFL_PATH)
    n_frames = len(frames)
    frame_duration = game_length / max(n_frames - 1, 1)
    print(f"Frames: {n_frames}, duration: {frame_duration:.2f}s, game: {game_length:.1f}s")

    # ═══════════════════════════════════════════════════════════════
    # ATTACK A: Full-frame byte scanning (no block parser)
    # For each position in the frame, check if byte == level across frames
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ATTACK A: Full-frame raw byte scan for level values")
    print("=" * 70)

    # For each XOR key (0-255), for each frame, the expected ciphertext byte is level ^ key
    # Find positions where this byte appears in ALL frames
    # This is equivalent to: find position p such that frame[p] ^ key == level for all frames

    # More efficient approach: for each position p, compute level_candidate = frame[p]
    # across all frames. Check if the sequence of candidates matches the oracle levels.

    # First, build per-frame level array
    frame_levels = []
    for fi in range(n_frames):
        ft = fi * frame_duration
        level = get_stat_at_time(oracle_data, ft, 'level')
        frame_levels.append(level if level is not None else 0)

    # Find the minimum frame size (to limit scan range)
    min_frame_size = min(len(f) for f in frames)
    print(f"Min frame size: {min_frame_size} bytes")
    print(f"Level sequence: {frame_levels[:10]}...{frame_levels[-5:]}")

    # For efficiency, use numpy
    # Build a matrix: frames × positions, where each cell is the byte value
    # Then for each position, check if XOR with a constant produces the level sequence
    MAX_SCAN = min(min_frame_size, 5000)  # scan first 5000 bytes of each frame
    print(f"Scanning first {MAX_SCAN} bytes of each frame...")

    # Build byte matrix
    byte_matrix = np.zeros((n_frames, MAX_SCAN), dtype=np.uint8)
    for fi, fd in enumerate(frames):
        byte_matrix[fi, :] = np.frombuffer(fd[:MAX_SCAN], dtype=np.uint8)

    levels_arr = np.array(frame_levels, dtype=np.uint8)

    # For each position, try XOR with constant k
    # byte_matrix[fi, pos] ^ k == levels_arr[fi] for all fi
    # => k = byte_matrix[fi, pos] ^ levels_arr[fi] for all fi
    # => all (byte_matrix[:, pos] ^ levels_arr[:]) must be the same value

    xor_candidates = byte_matrix ^ levels_arr[:, np.newaxis]  # shape: (n_frames, MAX_SCAN)

    # Check where all rows have the same value (the XOR key is constant)
    # Quick check: if xor_candidates[0, pos] == xor_candidates[fi, pos] for all fi
    ref = xor_candidates[0, :]  # reference row
    matches_xor = np.all(xor_candidates == ref[np.newaxis, :], axis=0)
    xor_hits = np.where(matches_xor)[0]

    if len(xor_hits) > 0:
        print(f"\n*** XOR MATCHES at {len(xor_hits)} positions ***")
        for pos in xor_hits[:20]:
            k = int(ref[pos])
            vals = byte_matrix[:, pos]
            print(f"  pos={pos:5d} XOR_key=0x{k:02x} byte_vals={vals[:10].tolist()}...")
    else:
        print("No XOR matches across full frame.")

    # Same for ADD: (byte + k) % 256 == level => k = (level - byte) % 256
    add_candidates = (levels_arr[:, np.newaxis].astype(np.int16) - byte_matrix.astype(np.int16)) % 256
    add_candidates = add_candidates.astype(np.uint8)
    ref_add = add_candidates[0, :]
    matches_add = np.all(add_candidates == ref_add[np.newaxis, :], axis=0)
    add_hits = np.where(matches_add)[0]

    if len(add_hits) > 0:
        print(f"\n*** ADD MATCHES at {len(add_hits)} positions ***")
        for pos in add_hits[:20]:
            k = int(ref_add[pos])
            print(f"  pos={pos:5d} ADD_key=0x{k:02x}")
    else:
        print("No ADD matches across full frame.")

    # ═══════════════════════════════════════════════════════════════
    # ATTACK B: PID 842 deep structural analysis
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ATTACK B: PID 842 structural analysis")
    print("=" * 70)

    pid842_samples = []
    for fi, fd in enumerate(frames):
        blocks = extract_blocks_targeted(fd, 842, valid_pids)
        b2_blocks = [b for b in blocks if b['param'] == GAREN_ENTITY]
        if b2_blocks:
            ft = fi * frame_duration
            level = get_stat_at_time(oracle_data, ft, 'level')
            cs = get_stat_at_time(oracle_data, ft, 'cs')
            kills = get_stat_at_time(oracle_data, ft, 'kills')
            for blk in b2_blocks:
                pid842_samples.append({
                    'frame': fi,
                    'level': level,
                    'cs': cs,
                    'kills': kills,
                    'payload': blk['payload'],
                    'channel': blk['channel'],
                })

    print(f"PID 842 entity b2: {len(pid842_samples)} blocks")
    if pid842_samples:
        min_size = min(len(s['payload']) for s in pid842_samples)
        print(f"Min payload size: {min_size}")
        channels = set(s['channel'] for s in pid842_samples)
        print(f"Channels: {sorted(channels)}")

        # Take ONE block per frame (first occurrence)
        seen_frames = set()
        unique_samples = []
        for s in pid842_samples:
            if s['frame'] not in seen_frames:
                seen_frames.add(s['frame'])
                unique_samples.append(s)

        print(f"Unique frames: {len(unique_samples)}")
        distinct_levels = len(set(s['level'] for s in unique_samples))
        print(f"Distinct levels: {distinct_levels}")

        # Entropy analysis: for each byte offset, compute entropy of values
        print(f"\nByte entropy analysis (min_size={min_size}):")
        entropies = []
        for off in range(min_size):
            vals = [s['payload'][off] for s in unique_samples]
            counts = np.bincount(vals, minlength=256)
            probs = counts / counts.sum()
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log2(probs))
            entropies.append((off, entropy, len(set(vals))))

        # Show bytes with low entropy (few unique values = likely flags/constants)
        print("\n  Low-entropy bytes (possible flags/constants):")
        for off, ent, nuniq in sorted(entropies, key=lambda x: x[1])[:10]:
            vals = [s['payload'][off] for s in unique_samples]
            unique_vals = sorted(set(vals))[:5]
            print(f"    offset={off:3d} entropy={ent:.3f} unique={nuniq:3d} vals={unique_vals}")

        # Show bytes with moderate entropy (could be stats like level)
        print("\n  Moderate-entropy bytes (possible stat values):")
        for off, ent, nuniq in entropies:
            if 3.0 < ent < 5.0 and 5 <= nuniq <= 30:
                vals = [s['payload'][off] for s in unique_samples]
                levels = [s['level'] for s in unique_samples]
                rho = np.corrcoef(vals, levels)[0, 1] if len(set(vals)) > 1 else 0
                print(f"    offset={off:3d} entropy={ent:.3f} unique={nuniq:3d} corr={rho:.4f}")

        # Strict injective permutation test on PID 842
        print("\n  Strict injective permutation test:")
        for off in range(min_size):
            vals = [s['payload'][off] for s in unique_samples]
            levels = [s['level'] for s in unique_samples]

            # Forward: byte → level (must be consistent)
            fwd = {}
            fwd_ok = True
            for v, l in zip(vals, levels):
                if v in fwd:
                    if fwd[v] != l:
                        fwd_ok = False
                        break
                else:
                    fwd[v] = l

            if not fwd_ok:
                continue

            # Reverse: level → byte (must be consistent for injectivity)
            rev = {}
            rev_ok = True
            for v, l in zip(vals, levels):
                if l in rev:
                    if rev[l] != v:
                        rev_ok = False
                        break
                else:
                    rev[l] = v

            if fwd_ok and rev_ok and len(fwd) >= 8:
                print(f"    *** INJECTIVE MATCH offset={off}: {len(fwd)} mappings")
                for k, v in sorted(fwd.items(), key=lambda x: x[1]):
                    print(f"      byte=0x{k:02x}({k:3d}) → level={v}")

        # Try u16 LE pairs with injective test
        print("\n  u16 LE injective test:")
        for off in range(min_size - 1):
            vals = [struct.unpack_from('<H', s['payload'], off)[0] for s in unique_samples]
            levels = [s['level'] for s in unique_samples]

            fwd = {}
            fwd_ok = True
            for v, l in zip(vals, levels):
                if v in fwd:
                    if fwd[v] != l:
                        fwd_ok = False
                        break
                else:
                    fwd[v] = l

            rev = {}
            rev_ok = True
            for v, l in zip(vals, levels):
                if l in rev:
                    if rev[l] != v:
                        rev_ok = False
                        break
                else:
                    rev[l] = v

            if fwd_ok and rev_ok and len(fwd) >= 8:
                print(f"    *** u16 INJECTIVE MATCH offset={off}: {len(fwd)} mappings")
                for k, v in sorted(fwd.items(), key=lambda x: x[1]):
                    print(f"      u16=0x{k:04x}({k:5d}) → level={v}")

        # Try u32 LE for CS (which goes 0→300+)
        print("\n  u32 LE check for CS (creep score):")
        for off in range(min_size - 3):
            vals = [struct.unpack_from('<I', s['payload'], off)[0] for s in unique_samples]
            cs_vals = [s['cs'] for s in unique_samples]

            fwd = {}
            fwd_ok = True
            for v, c in zip(vals, cs_vals):
                if v in fwd:
                    if fwd[v] != c:
                        fwd_ok = False
                        break
                else:
                    fwd[v] = c

            rev = {}
            rev_ok = True
            for v, c in zip(vals, cs_vals):
                if c in rev:
                    if rev[c] != v:
                        rev_ok = False
                        break
                else:
                    rev[c] = v

            if fwd_ok and rev_ok and len(fwd) >= 10:
                print(f"    *** u32 CS INJECTIVE MATCH offset={off}: {len(fwd)} mappings")
                for k, v in sorted(fwd.items(), key=lambda x: x[1])[:15]:
                    print(f"      u32=0x{k:08x}({k:10d}) → cs={v}")

    # ═══════════════════════════════════════════════════════════════
    # ATTACK C: Verify entity mapping by checking ALL entities
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ATTACK C: Check block counts per entity per PID")
    print("=" * 70)

    # For key PIDs, show entity distribution
    key_pids = [842, 183, 318, 813, 705, 539, 368, 305, 169, 487]
    for pid in key_pids:
        entity_counts = defaultdict(int)
        for fi, fd in enumerate(frames):
            blocks = extract_blocks_targeted(fd, pid, valid_pids)
            for b in blocks:
                entity_counts[b['param']] += 1

        hero_entries = {p: c for p, c in entity_counts.items()
                        if PLAYER_ID_START <= p < PLAYER_ID_START + 10}
        if hero_entries:
            dist = " ".join(f"{(p & 0xFF):02x}:{c}" for p, c in sorted(hero_entries.items()))
            total = sum(hero_entries.values())
            print(f"  PID {pid:4d}: {total:4d} hero blocks - {dist}")

    # ═══════════════════════════════════════════════════════════════
    # ATTACK D: Check if blocks with SAME channel have consistent bytes
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ATTACK D: Per-channel analysis for PID 842")
    print("=" * 70)

    if pid842_samples:
        by_channel = defaultdict(list)
        for s in pid842_samples:
            by_channel[s['channel']].append(s)

        for ch in sorted(by_channel.keys()):
            ch_samples = by_channel[ch]
            # One per frame
            seen = set()
            uniq = []
            for s in ch_samples:
                if s['frame'] not in seen:
                    seen.add(s['frame'])
                    uniq.append(s)

            distinct_levels = len(set(s['level'] for s in uniq))
            if len(uniq) < 5 or distinct_levels < 3:
                continue

            min_sz = min(len(s['payload']) for s in uniq)
            print(f"\n  Channel 0x{ch:02x}: {len(uniq)} frames, {distinct_levels} levels, min_sz={min_sz}")

            # Try injective permutation on each byte
            for off in range(min_sz):
                vals = [s['payload'][off] for s in uniq]
                levels = [s['level'] for s in uniq]

                fwd = {}
                fwd_ok = True
                for v, l in zip(vals, levels):
                    if v in fwd:
                        if fwd[v] != l:
                            fwd_ok = False
                            break
                    else:
                        fwd[v] = l

                rev = {}
                rev_ok = True
                for v, l in zip(vals, levels):
                    if l in rev:
                        if rev[l] != v:
                            rev_ok = False
                            break
                    else:
                        rev[l] = v

                if fwd_ok and rev_ok and len(fwd) >= 5:
                    print(f"    INJECTIVE offset={off}: {len(fwd)} mappings")
                    for k, v in sorted(fwd.items(), key=lambda x: x[1]):
                        print(f"      0x{k:02x} → level={v}")

    # ═══════════════════════════════════════════════════════════════
    # ATTACK E: Look at blocks for ALL entities to find global stat blocks
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ATTACK E: Non-entity blocks (param outside hero range)")
    print("=" * 70)

    # Check which params appear that are NOT hero entities
    non_hero_params = defaultdict(int)
    for fi in range(0, n_frames, 10):  # sample every 10th frame
        fd = frames[fi]
        pos = 0
        while pos + 9 <= len(fd):
            if fd[pos] in BLOCK_MARKERS:
                size = fd[pos + 2]
                pid = struct.unpack_from('<H', fd, pos + 3)[0]
                param = struct.unpack_from('<I', fd, pos + 5)[0]
                end = pos + 9 + size
                if end <= len(fd):
                    valid_chain = (end >= len(fd) - 9 or
                                  (end < len(fd) and fd[end] in BLOCK_MARKERS))
                    if valid_chain and pid in valid_pids:
                        if not (PLAYER_ID_START <= param < PLAYER_ID_START + 10):
                            non_hero_params[(pid, param)] += 1
                    if valid_chain:
                        pos = end
                        continue
            pos += 1

    print(f"Non-hero (pid, param) pairs (sampled from every 10th frame):")
    for (pid, param), count in sorted(non_hero_params.items(), key=lambda x: -x[1])[:20]:
        print(f"  PID={pid:4d} param=0x{param:08x} count={count}")

    # ═══════════════════════════════════════════════════════════════
    # ATTACK F: Hex dump comparison for PID 842 at different levels
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ATTACK F: PID 842 hex dump at different levels")
    print("=" * 70)

    if pid842_samples:
        # Pick one block per level transition
        seen_levels = set()
        level_examples = []
        for s in sorted(pid842_samples, key=lambda x: x['frame']):
            if s['level'] not in seen_levels and s['level'] is not None:
                seen_levels.add(s['level'])
                level_examples.append(s)

        for s in level_examples[:12]:
            p = s['payload']
            print(f"\n  Level={s['level']:2d} frame={s['frame']:3d} ch=0x{s['channel']:02x} sz={len(p)}")
            # Print first 64 bytes in hex with offset markers
            for row_start in range(0, min(len(p), 64), 16):
                hex_bytes = ' '.join(f'{p[row_start+i]:02x}' for i in range(min(16, len(p) - row_start)))
                print(f"    {row_start:3d}: {hex_bytes}")

    # ═══════════════════════════════════════════════════════════════
    # ATTACK G: Sliding window XOR/ADD on per-PID blocks
    # Try matching across VARIABLE skip lengths (1-7 bytes)
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("ATTACK G: Variable-offset injective test (sliding skip)")
    print("=" * 70)

    for pid in sorted(pid_data_cache.keys()) if 'pid_data_cache' in dir() else [842, 813, 705, 539, 368, 305, 169, 224]:
        pid_samples = []
        for fi, fd in enumerate(frames):
            blocks = extract_blocks_targeted(fd, pid, valid_pids)
            b2_blocks = [b for b in blocks if b['param'] == GAREN_ENTITY]
            if b2_blocks:
                ft = fi * frame_duration
                level = get_stat_at_time(oracle_data, ft, 'level')
                if level is not None:
                    pid_samples.append({
                        'frame': fi,
                        'level': level,
                        'payload': b2_blocks[0]['payload'],
                    })

        # One per frame
        seen = set()
        uniq = []
        for s in pid_samples:
            if s['frame'] not in seen:
                seen.add(s['frame'])
                uniq.append(s)

        distinct_levels = len(set(s['level'] for s in uniq))
        if len(uniq) < 10 or distinct_levels < 5:
            continue

        min_sz = min(len(s['payload']) for s in uniq)

        # For each block, try skip lengths 0-7, then check post-skip bytes
        for skip_len in range(0, min(8, min_sz - 1)):
            post_sz = min_sz - skip_len
            for off in range(min(post_sz, 40)):
                abs_off = skip_len + off
                vals = [s['payload'][abs_off] for s in uniq]
                levels = [s['level'] for s in uniq]

                # Injective check
                fwd = {}
                rev = {}
                ok = True
                for v, l in zip(vals, levels):
                    if v in fwd and fwd[v] != l:
                        ok = False
                        break
                    if l in rev and rev[l] != v:
                        ok = False
                        break
                    fwd[v] = l
                    rev[l] = v

                if ok and len(fwd) >= 8:
                    print(f"  PID={pid:4d} skip={skip_len} off={off} (abs={abs_off}): "
                          f"{len(fwd)} injective mappings")
                    for k, v in sorted(fwd.items(), key=lambda x: x[1]):
                        print(f"    0x{k:02x}({k:3d}) → level={v}")

    print("\nDone.")


if __name__ == '__main__':
    main()
