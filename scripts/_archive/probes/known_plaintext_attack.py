#!/usr/bin/env python3
"""
Known-plaintext attack on ROFL replay files.
Uses oracle data (level, CS, kills from Live Client API) to find which bytes
in the ROFL encode game stats.

For each frame, we know Garen's level from oracle data. We brute-force every
single-byte transform on every payload byte of every valid-PID block for entity
0x400000b2, looking for transforms that produce the correct level across ALL frames.
"""
import struct
import json
import sys
from collections import defaultdict
import zstandard as zstd
import numpy as np

# ── Config ──
ROFL_PATH = 'data/replays/NA1-5528069928.rofl'
ORACLE_PATH = '/tmp/oracle_data.json'
PID_MAP_PATH = '/tmp/pid_full_map.json'
RDATA_PATH = '/tmp/pe_dump_16.7/rdata.bin'
GAREN_ENTITY = 0x400000b2
BLOCK_MARKERS = {0x91, 0xf1, 0xb1, 0x31, 0x11}

# S-box offsets in rdata (file-relative, since rdata.bin starts at rdata section)
SBOX_OFFSETS = {
    'sbox_D_1056b0': 0x1056b0,
    'sbox_B_1057b0': 0x1057b0,
    'sbox_1059a0': 0x1059a0,
    'sbox_105400': 0x105400,
    'sbox_fa330': 0xfa330,
    'sbox_SKIP_1199f0': 0x1199f0,
}


def load_sboxes(rdata_path):
    """Load all known S-box tables from rdata."""
    with open(rdata_path, 'rb') as f:
        rdata = f.read()
    sboxes = {}
    for name, off in SBOX_OFFSETS.items():
        sbox = list(rdata[off:off + 256])
        sboxes[name] = sbox
        # Also build inverse S-box (for sbox_inv[level] → expected input byte)
        inv = [0] * 256
        for i, v in enumerate(sbox):
            inv[v] = i
        sboxes[name + '_inv'] = inv
    return sboxes


def load_oracle(oracle_path):
    """Load oracle data, return list of (time, level, cs, kills)."""
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
                'deaths': g['scores']['deaths'],
            })
    return results, data['game_length']


def load_valid_pids(pid_map_path):
    """Load set of valid PIDs from dispatch table."""
    with open(pid_map_path) as f:
        pid_map = json.load(f)
    return set(int(k) for k in pid_map.keys())


def parse_rofl_frames(rofl_path):
    """Parse ROFL file into list of decompressed frames."""
    with open(rofl_path, 'rb') as f:
        data = f.read()
    ZSTD_MAGIC = b'\x28\xb5\x2f\xfd'
    dctx = zstd.ZstdDecompressor()
    frames = []
    pos = 0
    while pos < len(data):
        idx = data.find(ZSTD_MAGIC, pos)
        if idx < 0:
            break
        try:
            dec = dctx.decompress(data[idx:idx + 0x100000], max_output_size=0x400000)
            frames.append(dec)
        except:
            pass
        pos = idx + 1
    return frames


def extract_entity_blocks(frame_data, entity_id, valid_pids):
    """Extract all blocks for a specific entity from a decompressed frame.
    Returns list of (pid, payload_bytes) tuples.
    Uses sequential chain scanning with PID validation.
    """
    fd = frame_data
    blocks = []
    pos = 0
    while pos + 9 <= len(fd):
        if fd[pos] in BLOCK_MARKERS:
            size = fd[pos + 2]
            pid = struct.unpack_from('<H', fd, pos + 3)[0]
            param = struct.unpack_from('<I', fd, pos + 5)[0]
            end = pos + 9 + size
            if end <= len(fd):
                # Check chain: next byte should be a marker or near end
                valid_chain = (end >= len(fd) - 9 or
                              (end < len(fd) and fd[end] in BLOCK_MARKERS))
                if valid_chain and pid in valid_pids and param == entity_id:
                    blocks.append((pid, bytes(fd[pos + 9:end])))
                if valid_chain:
                    pos = end
                    continue
        pos += 1
    return blocks


def get_level_at_time(oracle_data, t):
    """Get oracle level at time t (use most recent snapshot before t)."""
    level = None
    for snap in oracle_data:
        if snap['time'] <= t + 1.0:  # small tolerance
            level = snap['level']
        else:
            break
    return level


def get_stat_at_time(oracle_data, t, stat_name):
    """Get any oracle stat at time t."""
    val = None
    for snap in oracle_data:
        if snap['time'] <= t + 1.0:
            val = snap[stat_name]
        else:
            break
    return val


def brute_force_single_byte(frames_data, oracle_data, game_length, valid_pids, sboxes):
    """
    For each frame, get expected level, then for each entity-b2 block,
    try every single-byte transform on every payload byte.

    Approach: For efficiency, compute per (PID, byte_offset) what XOR key / ADD key
    would be needed. If the key is constant across ALL frames with distinct levels,
    we have a match.
    """
    n_frames = len(frames_data)
    frame_duration = game_length / max(n_frames - 1, 1)

    print(f"Frames: {n_frames}, frame_duration: {frame_duration:.2f}s")
    print(f"Valid PIDs: {len(valid_pids)}")
    print(f"S-boxes: {len([k for k in sboxes if not k.endswith('_inv')])} tables")

    # ── Phase 1: Collect all (pid, byte_offset) → [(frame_idx, byte_val, expected_level)] ──
    # Track per-PID block data indexed by frame
    pid_frame_data = defaultdict(list)  # pid → [(frame_idx, payload_bytes)]

    for frame_idx, fd in enumerate(frames_data):
        blocks = extract_entity_blocks(fd, GAREN_ENTITY, valid_pids)
        for pid, payload in blocks:
            pid_frame_data[pid].append((frame_idx, payload))

    print(f"\nEntity b2 blocks by PID:")
    for pid in sorted(pid_frame_data.keys()):
        n_blocks = len(pid_frame_data[pid])
        sizes = set(len(p) for _, p in pid_frame_data[pid])
        print(f"  PID {pid:4d}: {n_blocks:4d} blocks, sizes: {sorted(sizes)[:10]}{'...' if len(sizes)>10 else ''}")

    # ── Phase 2: For each PID, try brute-force transforms ──
    results = []
    stats_to_check = ['level', 'cs', 'kills']

    for pid in sorted(pid_frame_data.keys()):
        blocks_list = pid_frame_data[pid]
        if len(blocks_list) < 3:
            continue

        # Get min payload size across all blocks for this PID
        min_size = min(len(p) for _, p in blocks_list)
        if min_size == 0:
            continue

        # Build (frame_idx, expected_values, payload) tuples
        samples = []
        for frame_idx, payload in blocks_list:
            frame_time = frame_idx * frame_duration
            level = get_level_at_time(oracle_data, frame_time)
            cs = get_stat_at_time(oracle_data, frame_time, 'cs')
            kills = get_stat_at_time(oracle_data, frame_time, 'kills')
            if level is not None:
                samples.append((frame_idx, {'level': level, 'cs': cs, 'kills': kills}, payload))

        if len(samples) < 3:
            continue

        # Check how many distinct levels we see
        distinct_levels = len(set(s[1]['level'] for s in samples))
        if distinct_levels < 3:
            continue

        # For each byte offset, try transforms
        for byte_off in range(min_size):
            for stat_name in stats_to_check:
                distinct_stat_vals = len(set(s[1][stat_name] for s in samples))
                if distinct_stat_vals < 3:
                    continue

                # Extract byte values and expected stat values
                byte_vals = [s[2][byte_off] for s in samples]
                expected = [s[1][stat_name] for s in samples]

                # Skip if stat value > 255 (can't be single byte)
                if max(expected) > 255:
                    continue

                # ── Transform 1: RAW (byte == stat) ──
                if all(b == e for b, e in zip(byte_vals, expected)):
                    results.append({
                        'pid': pid, 'offset': byte_off, 'transform': 'RAW',
                        'stat': stat_name, 'distinct': distinct_stat_vals,
                        'n_samples': len(samples),
                    })

                # ── Transform 2: XOR with constant k ──
                # k = byte ^ expected; check if constant
                xor_keys = set((b ^ e) & 0xFF for b, e in zip(byte_vals, expected))
                if len(xor_keys) == 1:
                    k = xor_keys.pop()
                    results.append({
                        'pid': pid, 'offset': byte_off, 'transform': f'XOR_0x{k:02x}',
                        'stat': stat_name, 'distinct': distinct_stat_vals,
                        'n_samples': len(samples),
                    })

                # ── Transform 3: ADD constant k ──
                # (byte + k) % 256 == expected → k = (expected - byte) % 256
                add_keys = set((e - b) % 256 for b, e in zip(byte_vals, expected))
                if len(add_keys) == 1:
                    k = add_keys.pop()
                    if k != 0 or 'RAW' not in [r['transform'] for r in results
                                                 if r['pid'] == pid and r['offset'] == byte_off and r['stat'] == stat_name]:
                        results.append({
                            'pid': pid, 'offset': byte_off, 'transform': f'ADD_0x{k:02x}',
                            'stat': stat_name, 'distinct': distinct_stat_vals,
                            'n_samples': len(samples),
                        })

                # ── Transform 4: S-box lookups ──
                for sbox_name, sbox in sboxes.items():
                    if sbox_name.endswith('_inv'):
                        continue
                    if all(sbox[b] == e for b, e in zip(byte_vals, expected)):
                        results.append({
                            'pid': pid, 'offset': byte_off,
                            'transform': f'SBOX_{sbox_name}',
                            'stat': stat_name, 'distinct': distinct_stat_vals,
                            'n_samples': len(samples),
                        })

                # ── Transform 5: NOT then check ──
                not_vals = [(~b) & 0xFF for b in byte_vals]
                if all(n == e for n, e in zip(not_vals, expected)):
                    results.append({
                        'pid': pid, 'offset': byte_off, 'transform': 'NOT',
                        'stat': stat_name, 'distinct': distinct_stat_vals,
                        'n_samples': len(samples),
                    })

                # ── Transform 6: ROR by 1-7 bits ──
                for rot in range(1, 8):
                    ror_vals = [((b >> rot) | (b << (8 - rot))) & 0xFF for b in byte_vals]
                    ror_keys = set((r ^ e) & 0xFF for r, e in zip(ror_vals, expected))
                    if len(ror_keys) == 1:
                        k = ror_keys.pop()
                        results.append({
                            'pid': pid, 'offset': byte_off,
                            'transform': f'ROR{rot}_XOR_0x{k:02x}',
                            'stat': stat_name, 'distinct': distinct_stat_vals,
                            'n_samples': len(samples),
                        })

    return results


def brute_force_u16(frames_data, oracle_data, game_length, valid_pids):
    """Try u16 LE pairs for stats that might exceed 255 (CS, gold)."""
    n_frames = len(frames_data)
    frame_duration = game_length / max(n_frames - 1, 1)

    pid_frame_data = defaultdict(list)
    for frame_idx, fd in enumerate(frames_data):
        blocks = extract_entity_blocks(fd, GAREN_ENTITY, valid_pids)
        for pid, payload in blocks:
            pid_frame_data[pid].append((frame_idx, payload))

    results = []
    stats_to_check = ['cs', 'level']  # CS can go up to ~300

    for pid in sorted(pid_frame_data.keys()):
        blocks_list = pid_frame_data[pid]
        if len(blocks_list) < 3:
            continue

        min_size = min(len(p) for _, p in blocks_list)
        if min_size < 2:
            continue

        samples = []
        for frame_idx, payload in blocks_list:
            frame_time = frame_idx * frame_duration
            level = get_level_at_time(oracle_data, frame_time)
            cs = get_stat_at_time(oracle_data, frame_time, 'cs')
            if level is not None:
                samples.append((frame_idx, {'level': level, 'cs': cs}, payload))

        if len(samples) < 3:
            continue

        for stat_name in stats_to_check:
            distinct_vals = len(set(s[1][stat_name] for s in samples))
            if distinct_vals < 3:
                continue

            for byte_off in range(min_size - 1):
                expected = [s[1][stat_name] for s in samples]

                # u16 LE raw
                u16_vals = [struct.unpack_from('<H', s[2], byte_off)[0] for s in samples]
                if all(v == e for v, e in zip(u16_vals, expected)):
                    results.append({
                        'pid': pid, 'offset': byte_off, 'transform': 'U16_LE_RAW',
                        'stat': stat_name, 'distinct': distinct_vals,
                        'n_samples': len(samples),
                    })

                # u16 LE + constant k (mod 65536)
                add_keys = set((e - v) % 65536 for v, e in zip(u16_vals, expected))
                if len(add_keys) == 1:
                    k = add_keys.pop()
                    results.append({
                        'pid': pid, 'offset': byte_off, 'transform': f'U16_LE_ADD_0x{k:04x}',
                        'stat': stat_name, 'distinct': distinct_vals,
                        'n_samples': len(samples),
                    })

                # u16 BE raw
                u16be_vals = [struct.unpack_from('>H', s[2], byte_off)[0] for s in samples]
                if all(v == e for v, e in zip(u16be_vals, expected)):
                    results.append({
                        'pid': pid, 'offset': byte_off, 'transform': 'U16_BE_RAW',
                        'stat': stat_name, 'distinct': distinct_vals,
                        'n_samples': len(samples),
                    })

                # u16 XOR constant
                xor_keys = set((v ^ e) & 0xFFFF for v, e in zip(u16_vals, expected))
                if len(xor_keys) == 1:
                    k = xor_keys.pop()
                    results.append({
                        'pid': pid, 'offset': byte_off, 'transform': f'U16_LE_XOR_0x{k:04x}',
                        'stat': stat_name, 'distinct': distinct_vals,
                        'n_samples': len(samples),
                    })

    return results


def brute_force_bitfield(frames_data, oracle_data, game_length, valid_pids):
    """Try extracting N-bit fields at various bit offsets.
    Level fits in 5 bits (0-31), CS fits in 10 bits (0-1023).
    """
    n_frames = len(frames_data)
    frame_duration = game_length / max(n_frames - 1, 1)

    pid_frame_data = defaultdict(list)
    for frame_idx, fd in enumerate(frames_data):
        blocks = extract_entity_blocks(fd, GAREN_ENTITY, valid_pids)
        for pid, payload in blocks:
            pid_frame_data[pid].append((frame_idx, payload))

    results = []

    def extract_bits(payload, bit_offset, n_bits):
        """Extract n_bits starting at bit_offset from payload bytes."""
        val = 0
        for i in range(n_bits):
            byte_idx = (bit_offset + i) // 8
            bit_idx = (bit_offset + i) % 8
            if byte_idx >= len(payload):
                return None
            val |= ((payload[byte_idx] >> bit_idx) & 1) << i
        return val

    for pid in sorted(pid_frame_data.keys()):
        blocks_list = pid_frame_data[pid]
        if len(blocks_list) < 5:
            continue

        min_size = min(len(p) for _, p in blocks_list)
        if min_size < 2:
            continue

        samples = []
        for frame_idx, payload in blocks_list:
            frame_time = frame_idx * frame_duration
            level = get_level_at_time(oracle_data, frame_time)
            cs = get_stat_at_time(oracle_data, frame_time, 'cs')
            if level is not None:
                samples.append((frame_idx, {'level': level, 'cs': cs}, payload))

        distinct_levels = len(set(s[1]['level'] for s in samples))
        if distinct_levels < 5:
            continue

        max_bit = min_size * 8

        # Try 5-bit fields for level (values 1-20)
        for bit_off in range(max_bit - 5):
            vals = []
            valid = True
            for _, stats, payload in samples:
                v = extract_bits(payload, bit_off, 5)
                if v is None:
                    valid = False
                    break
                vals.append(v)
            if not valid:
                continue
            expected = [s[1]['level'] for s in samples]
            if all(v == e for v, e in zip(vals, expected)):
                results.append({
                    'pid': pid, 'bit_offset': bit_off, 'n_bits': 5,
                    'transform': f'BITS[{bit_off}:{bit_off+5}]',
                    'stat': 'level', 'distinct': distinct_levels,
                    'n_samples': len(samples),
                })
            # Also try XOR on the extracted value
            xor_keys = set((v ^ e) & 0x1F for v, e in zip(vals, expected))
            if len(xor_keys) == 1 and xor_keys != {0}:
                k = xor_keys.pop()
                results.append({
                    'pid': pid, 'bit_offset': bit_off, 'n_bits': 5,
                    'transform': f'BITS[{bit_off}:{bit_off+5}]_XOR_0x{k:02x}',
                    'stat': 'level', 'distinct': distinct_levels,
                    'n_samples': len(samples),
                })

        # Try 8-bit fields at non-byte boundaries for level
        for bit_off in range(1, max_bit - 8):  # skip byte-aligned (already covered above)
            if bit_off % 8 == 0:
                continue
            vals = []
            valid = True
            for _, stats, payload in samples:
                v = extract_bits(payload, bit_off, 8)
                if v is None:
                    valid = False
                    break
                vals.append(v)
            if not valid:
                continue
            expected = [s[1]['level'] for s in samples]
            add_keys = set((e - v) % 256 for v, e in zip(vals, expected))
            if len(add_keys) == 1:
                k = add_keys.pop()
                results.append({
                    'pid': pid, 'bit_offset': bit_off, 'n_bits': 8,
                    'transform': f'BITS[{bit_off}:{bit_off+8}]_ADD_0x{k:02x}',
                    'stat': 'level', 'distinct': distinct_levels,
                    'n_samples': len(samples),
                })

    return results


def main():
    print("=" * 70)
    print("KNOWN-PLAINTEXT ATTACK ON ROFL")
    print("=" * 70)

    # Load data
    print("\nLoading oracle data...")
    oracle_data, game_length = load_oracle(ORACLE_PATH)
    print(f"  {len(oracle_data)} snapshots, game length: {game_length:.1f}s")

    print("Loading valid PIDs...")
    valid_pids = load_valid_pids(PID_MAP_PATH)
    print(f"  {len(valid_pids)} valid PIDs")

    print("Loading S-box tables...")
    sboxes = load_sboxes(RDATA_PATH)
    print(f"  {len([k for k in sboxes if not k.endswith('_inv')])} S-boxes loaded")

    print("Parsing ROFL frames...")
    frames = parse_rofl_frames(ROFL_PATH)
    print(f"  {len(frames)} frames")

    # ── Phase 1: Single-byte transforms ──
    print("\n" + "=" * 70)
    print("PHASE 1: Single-byte transforms (RAW, XOR, ADD, SBOX, NOT, ROR)")
    print("=" * 70)
    results1 = brute_force_single_byte(frames, oracle_data, game_length, valid_pids, sboxes)

    if results1:
        print(f"\n*** FOUND {len(results1)} MATCHES ***")
        for r in sorted(results1, key=lambda x: (-x['distinct'], x['pid'], x['offset'])):
            print(f"  PID={r['pid']:4d} offset={r['offset']:3d} transform={r['transform']:20s} "
                  f"stat={r['stat']:6s} distinct_vals={r['distinct']:2d} samples={r['n_samples']}")
    else:
        print("\nNo single-byte matches found.")

    # ── Phase 2: u16 transforms ──
    print("\n" + "=" * 70)
    print("PHASE 2: u16 transforms (LE/BE, RAW, ADD, XOR)")
    print("=" * 70)
    results2 = brute_force_u16(frames, oracle_data, game_length, valid_pids)

    if results2:
        print(f"\n*** FOUND {len(results2)} MATCHES ***")
        for r in sorted(results2, key=lambda x: (-x['distinct'], x['pid'], x['offset'])):
            print(f"  PID={r['pid']:4d} offset={r['offset']:3d} transform={r['transform']:20s} "
                  f"stat={r['stat']:6s} distinct_vals={r['distinct']:2d} samples={r['n_samples']}")
    else:
        print("\nNo u16 matches found.")

    # ── Phase 3: Bit-field transforms ──
    print("\n" + "=" * 70)
    print("PHASE 3: Bit-field transforms (5-bit level, 8-bit non-aligned)")
    print("=" * 70)
    results3 = brute_force_bitfield(frames, oracle_data, game_length, valid_pids)

    if results3:
        print(f"\n*** FOUND {len(results3)} MATCHES ***")
        for r in sorted(results3, key=lambda x: (-x['distinct'], x['pid'], x['offset'])):
            extra = f"bit_off={r.get('bit_offset', '?')}" if 'bit_offset' in r else ''
            print(f"  PID={r['pid']:4d} {extra:12s} transform={r['transform']:30s} "
                  f"stat={r['stat']:6s} distinct_vals={r['distinct']:2d} samples={r['n_samples']}")
    else:
        print("\nNo bit-field matches found.")

    # ── Summary ──
    all_results = results1 + results2 + results3
    print("\n" + "=" * 70)
    print(f"TOTAL MATCHES: {len(all_results)}")
    print("=" * 70)

    if all_results:
        # Save results
        with open('/tmp/kpa_results.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        print("Results saved to /tmp/kpa_results.json")

    # ── Phase 4: Diagnostic — show raw byte values at promising PIDs ──
    if not all_results:
        print("\n" + "=" * 70)
        print("DIAGNOSTIC: Showing byte value distributions for top PIDs")
        print("=" * 70)
        n_frames = len(frames)
        frame_duration = game_length / max(n_frames - 1, 1)

        pid_frame_data = defaultdict(list)
        for frame_idx, fd in enumerate(frames):
            blocks = extract_entity_blocks(fd, GAREN_ENTITY, valid_pids)
            for pid, payload in blocks:
                pid_frame_data[pid].append((frame_idx, payload))

        # For each PID with many blocks, show if any byte correlates with level
        for pid in sorted(pid_frame_data.keys(), key=lambda p: -len(pid_frame_data[p]))[:15]:
            blocks_list = pid_frame_data[pid]
            min_size = min(len(p) for _, p in blocks_list)

            samples = []
            for frame_idx, payload in blocks_list:
                frame_time = frame_idx * frame_duration
                level = get_level_at_time(oracle_data, frame_time)
                if level is not None:
                    samples.append((frame_idx, level, payload))

            if len(samples) < 5:
                continue

            distinct_levels = len(set(s[1] for s in samples))
            print(f"\nPID {pid}: {len(samples)} samples, {distinct_levels} distinct levels, min_size={min_size}")

            # For each byte, compute correlation with level
            best_corr = 0
            best_off = -1
            for byte_off in range(min(min_size, 60)):
                byte_vals = [s[2][byte_off] for s in samples]
                levels = [s[1] for s in samples]
                if len(set(byte_vals)) < 2:
                    continue
                # Pearson correlation
                bv = np.array(byte_vals, dtype=float)
                lv = np.array(levels, dtype=float)
                corr = np.corrcoef(bv, lv)[0, 1]
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_off = byte_off

            if best_off >= 0:
                print(f"  Best correlation: offset={best_off}, r={best_corr:.4f}")
                # Show values at that offset
                byte_vals = [s[2][best_off] for s in samples]
                levels = [s[1] for s in samples]
                # Show a few samples
                for i in range(0, len(samples), max(1, len(samples) // 10)):
                    print(f"    frame={samples[i][0]:3d} level={levels[i]:2d} byte=0x{byte_vals[i]:02x}({byte_vals[i]:3d})")


if __name__ == '__main__':
    main()
