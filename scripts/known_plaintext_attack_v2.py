#!/usr/bin/env python3
"""
Known-plaintext attack v2 on ROFL replay files.
Improvements over v1:
1. Decode and strip variable-length SKIP varint from each payload
2. Test arbitrary permutation matches (not just XOR/ADD)
3. Try correlating CHANGES in byte values with level CHANGES
4. Group by channel byte to handle per-channel keying
5. Try monotonic correlation (Spearman rank)
"""
import struct
import json
import sys
from collections import defaultdict
import zstandard as zstd
import numpy as np
from scipy import stats as scipy_stats

# ── Config ──
ROFL_PATH = 'data/replays/NA1-5528069928.rofl'
ORACLE_PATH = '/tmp/oracle_data.json'
PID_MAP_PATH = '/tmp/pid_full_map.json'
RDATA_PATH = '/tmp/pe_dump_16.7/rdata.bin'
GAREN_ENTITY = 0x400000b2
BLOCK_MARKERS = {0x91, 0xf1, 0xb1, 0x31, 0x11}

# Load S-box for SKIP decoder
def load_skip_sbox(rdata_path):
    with open(rdata_path, 'rb') as f:
        rdata = f.read()
    return list(rdata[0x1199f0:0x1199f0 + 256])

def bit_shuffle(val):
    """Common bit-shuffle used by all ciphers."""
    al = val & 0xD5
    dl = (val >> 1) & 0x55
    return ((al << 1) & 0xFF) | dl

def decode_skip_byte(byte_val, sbox):
    """Decode one byte through the SKIP cipher A.
    Algorithm: NOT → sbox → +0x54 → double_recombine → sbox → -0x4C → sbox → take 7 bits
    """
    v = (~byte_val) & 0xFF          # NOT
    v = sbox[v]                     # sbox lookup
    v = (v + 0x54) & 0xFF           # add 0x54
    v = bit_shuffle(v)              # recombine (shuffle)
    v = bit_shuffle(v)              # double recombine
    v = sbox[v]                     # sbox lookup
    v = (v - 0x4C) & 0xFF           # subtract 0x4C
    v = sbox[v]                     # sbox lookup
    return v

def decode_skip_varint(payload, sbox):
    """Decode the SKIP varint at the start of the payload.
    Returns (decoded_value, bytes_consumed).
    """
    value = 0
    shift = 0
    for i, byte_val in enumerate(payload):
        decoded = decode_skip_byte(byte_val, sbox)
        value |= (decoded & 0x7F) << shift
        shift += 7
        if not (decoded & 0x80):  # no continuation bit
            return value, i + 1
        if i > 5:  # safety: varints shouldn't be longer than 5 bytes
            return value, i + 1
    return value, len(payload)


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
                'deaths': g['scores']['deaths'],
            })
    return results, data['game_length']


def load_valid_pids(pid_map_path):
    with open(pid_map_path) as f:
        pid_map = json.load(f)
    return set(int(k) for k in pid_map.keys())


def parse_rofl_frames(rofl_path):
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
    """Extract all blocks for a specific entity, including channel byte."""
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
                valid_chain = (end >= len(fd) - 9 or
                              (end < len(fd) and fd[end] in BLOCK_MARKERS))
                if valid_chain and pid in valid_pids and param == entity_id:
                    blocks.append({
                        'pid': pid,
                        'payload': bytes(fd[pos + 9:end]),
                        'marker': fd[pos],
                        'channel': fd[pos + 1],
                    })
                if valid_chain:
                    pos = end
                    continue
        pos += 1
    return blocks


def get_stat_at_time(oracle_data, t, stat_name):
    val = None
    for snap in oracle_data:
        if snap['time'] <= t + 1.0:
            val = snap[stat_name]
        else:
            break
    return val


def test_arbitrary_permutation(byte_vals, expected_vals):
    """Test if there exists ANY permutation (bijective mapping) from byte_vals to expected_vals.
    Returns (is_match, mapping_dict) if consistent.
    """
    mapping = {}
    for b, e in zip(byte_vals, expected_vals):
        if b in mapping:
            if mapping[b] != e:
                return False, {}
        else:
            mapping[b] = e
    return True, mapping


def main():
    print("=" * 70)
    print("KNOWN-PLAINTEXT ATTACK v2 ON ROFL")
    print("=" * 70)

    # Load data
    oracle_data, game_length = load_oracle(ORACLE_PATH)
    valid_pids = load_valid_pids(PID_MAP_PATH)
    skip_sbox = load_skip_sbox(RDATA_PATH)

    print(f"Oracle: {len(oracle_data)} snapshots, game: {game_length:.1f}s")
    print(f"Valid PIDs: {len(valid_pids)}")

    frames = parse_rofl_frames(ROFL_PATH)
    n_frames = len(frames)
    frame_duration = game_length / max(n_frames - 1, 1)
    print(f"Frames: {n_frames}, duration: {frame_duration:.2f}s")

    # ── Collect all blocks per frame with SKIP decoded ──
    print("\nExtracting blocks and decoding SKIP varints...")
    pid_data = defaultdict(list)  # pid → [(frame_idx, skip_val, skip_len, post_skip_payload, channel)]

    for frame_idx, fd in enumerate(frames):
        blocks = extract_entity_blocks(fd, GAREN_ENTITY, valid_pids)
        for blk in blocks:
            pid = blk['pid']
            payload = blk['payload']
            if len(payload) < 1:
                continue
            skip_val, skip_len = decode_skip_varint(payload, skip_sbox)
            post_skip = payload[skip_len:]
            pid_data[pid].append({
                'frame': frame_idx,
                'skip_val': skip_val,
                'skip_len': skip_len,
                'post_skip': post_skip,
                'raw_payload': payload,
                'channel': blk['channel'],
            })

    # Print PID summary with SKIP info
    print(f"\nPID summary (entity b2):")
    for pid in sorted(pid_data.keys()):
        entries = pid_data[pid]
        skip_vals = set(e['skip_val'] for e in entries)
        skip_lens = set(e['skip_len'] for e in entries)
        post_sizes = set(len(e['post_skip']) for e in entries)
        channels = set(e['channel'] for e in entries)
        frames_covered = len(set(e['frame'] for e in entries))
        print(f"  PID {pid:4d}: {len(entries):4d} blocks, {frames_covered:3d} frames, "
              f"skip_vals={sorted(skip_vals)[:5]}, skip_lens={sorted(skip_lens)}, "
              f"post_sizes={sorted(post_sizes)[:5]}{'...' if len(post_sizes)>5 else ''}, "
              f"channels={sorted(channels)[:5]}")

    # ── ATTACK 1: Arbitrary permutation on post-SKIP bytes ──
    print("\n" + "=" * 70)
    print("ATTACK 1: Arbitrary permutation on post-SKIP bytes")
    print("=" * 70)

    results = []

    for pid in sorted(pid_data.keys()):
        entries = pid_data[pid]
        if len(entries) < 5:
            continue

        # Get one entry per frame (first block in that frame)
        frame_to_entries = defaultdict(list)
        for e in entries:
            frame_to_entries[e['frame']].append(e)

        # Use first block per frame to avoid multi-block ambiguity
        samples = []
        for frame_idx in sorted(frame_to_entries.keys()):
            entry = frame_to_entries[frame_idx][0]
            frame_time = frame_idx * frame_duration
            level = get_stat_at_time(oracle_data, frame_time, 'level')
            cs = get_stat_at_time(oracle_data, frame_time, 'cs')
            kills = get_stat_at_time(oracle_data, frame_time, 'kills')
            if level is not None:
                samples.append({
                    'frame': frame_idx,
                    'level': level,
                    'cs': cs,
                    'kills': kills,
                    'post_skip': entry['post_skip'],
                    'raw': entry['raw_payload'],
                    'channel': entry['channel'],
                    'skip_len': entry['skip_len'],
                })

        if len(samples) < 10:
            continue

        for stat_name in ['level', 'kills']:
            expected = [s[stat_name] for s in samples]
            distinct = len(set(expected))
            if distinct < 5:
                continue
            if max(expected) > 255:
                continue

            # Post-SKIP bytes
            min_post_size = min(len(s['post_skip']) for s in samples)
            for byte_off in range(min(min_post_size, 80)):
                byte_vals = [s['post_skip'][byte_off] for s in samples]
                is_perm, mapping = test_arbitrary_permutation(byte_vals, expected)
                if is_perm and len(mapping) >= min(distinct, 5):
                    results.append({
                        'pid': pid, 'offset': f'post_skip+{byte_off}',
                        'transform': 'ARBITRARY_PERM',
                        'stat': stat_name, 'distinct': len(mapping),
                        'total_distinct': distinct,
                        'n_samples': len(samples),
                        'mapping': {f'0x{k:02x}': v for k, v in sorted(mapping.items())},
                    })

            # Raw payload bytes (before SKIP stripping)
            min_raw_size = min(len(s['raw']) for s in samples)
            for byte_off in range(min(min_raw_size, 80)):
                byte_vals = [s['raw'][byte_off] for s in samples]
                is_perm, mapping = test_arbitrary_permutation(byte_vals, expected)
                if is_perm and len(mapping) >= min(distinct, 5):
                    results.append({
                        'pid': pid, 'offset': f'raw+{byte_off}',
                        'transform': 'ARBITRARY_PERM',
                        'stat': stat_name, 'distinct': len(mapping),
                        'total_distinct': distinct,
                        'n_samples': len(samples),
                        'mapping': {f'0x{k:02x}': v for k, v in sorted(mapping.items())},
                    })

    if results:
        print(f"\n*** FOUND {len(results)} PERMUTATION MATCHES ***")
        for r in sorted(results, key=lambda x: (-x['distinct'], x['pid'])):
            print(f"  PID={r['pid']:4d} {r['offset']:15s} stat={r['stat']:6s} "
                  f"distinct_mapped={r['distinct']}/{r['total_distinct']} "
                  f"samples={r['n_samples']}")
            if r['distinct'] >= 5:
                print(f"    Mapping: {r['mapping']}")
    else:
        print("No permutation matches found.")

    # ── ATTACK 2: Spearman rank correlation on post-SKIP bytes ──
    print("\n" + "=" * 70)
    print("ATTACK 2: Spearman rank correlation on post-SKIP bytes")
    print("=" * 70)

    corr_results = []

    for pid in sorted(pid_data.keys()):
        entries = pid_data[pid]
        if len(entries) < 10:
            continue

        frame_to_entries = defaultdict(list)
        for e in entries:
            frame_to_entries[e['frame']].append(e)

        samples = []
        for frame_idx in sorted(frame_to_entries.keys()):
            entry = frame_to_entries[frame_idx][0]
            frame_time = frame_idx * frame_duration
            level = get_stat_at_time(oracle_data, frame_time, 'level')
            if level is not None:
                samples.append({
                    'frame': frame_idx,
                    'level': level,
                    'post_skip': entry['post_skip'],
                    'raw': entry['raw_payload'],
                })

        distinct_levels = len(set(s['level'] for s in samples))
        if distinct_levels < 5 or len(samples) < 10:
            continue

        min_post_size = min(len(s['post_skip']) for s in samples)
        levels = [s['level'] for s in samples]

        for byte_off in range(min(min_post_size, 80)):
            byte_vals = [s['post_skip'][byte_off] for s in samples]
            if len(set(byte_vals)) < 3:
                continue
            rho, pval = scipy_stats.spearmanr(byte_vals, levels)
            if abs(rho) > 0.7 and pval < 0.001:
                corr_results.append({
                    'pid': pid, 'offset': f'post_skip+{byte_off}',
                    'rho': rho, 'pval': pval,
                    'n_samples': len(samples),
                })

        # Also raw bytes
        min_raw_size = min(len(s['raw']) for s in samples)
        for byte_off in range(min(min_raw_size, 80)):
            byte_vals = [s['raw'][byte_off] for s in samples]
            if len(set(byte_vals)) < 3:
                continue
            rho, pval = scipy_stats.spearmanr(byte_vals, levels)
            if abs(rho) > 0.7 and pval < 0.001:
                corr_results.append({
                    'pid': pid, 'offset': f'raw+{byte_off}',
                    'rho': rho, 'pval': pval,
                    'n_samples': len(samples),
                })

    if corr_results:
        print(f"\n*** FOUND {len(corr_results)} HIGH-CORRELATION BYTES ***")
        for r in sorted(corr_results, key=lambda x: -abs(x['rho'])):
            print(f"  PID={r['pid']:4d} {r['offset']:15s} rho={r['rho']:.4f} "
                  f"p={r['pval']:.2e} samples={r['n_samples']}")
    else:
        print("No high-correlation bytes found (threshold |rho| > 0.7)")

    # ── ATTACK 3: Differential attack — look at CHANGES between consecutive blocks ──
    print("\n" + "=" * 70)
    print("ATTACK 3: Differential attack — byte changes vs level changes")
    print("=" * 70)

    diff_results = []

    for pid in sorted(pid_data.keys()):
        entries = pid_data[pid]
        if len(entries) < 15:
            continue

        frame_to_entries = defaultdict(list)
        for e in entries:
            frame_to_entries[e['frame']].append(e)

        samples = []
        for frame_idx in sorted(frame_to_entries.keys()):
            entry = frame_to_entries[frame_idx][0]
            frame_time = frame_idx * frame_duration
            level = get_stat_at_time(oracle_data, frame_time, 'level')
            if level is not None:
                samples.append({
                    'frame': frame_idx,
                    'level': level,
                    'post_skip': entry['post_skip'],
                })

        if len(samples) < 15:
            continue

        min_post_size = min(len(s['post_skip']) for s in samples)

        # Compute deltas between consecutive samples
        for byte_off in range(min(min_post_size, 60)):
            # Check: byte CHANGES when and only when level CHANGES
            level_changes = 0
            byte_changes = 0
            co_changes = 0
            for i in range(1, len(samples)):
                lev_changed = samples[i]['level'] != samples[i-1]['level']
                byte_changed = samples[i]['post_skip'][byte_off] != samples[i-1]['post_skip'][byte_off]
                if lev_changed:
                    level_changes += 1
                if byte_changed:
                    byte_changes += 1
                if lev_changed and byte_changed:
                    co_changes += 1

            if level_changes > 5 and byte_changes > 0:
                # Jaccard similarity of change events
                union = level_changes + byte_changes - co_changes
                jaccard = co_changes / union if union > 0 else 0
                if jaccard > 0.5:
                    diff_results.append({
                        'pid': pid, 'offset': f'post_skip+{byte_off}',
                        'jaccard': jaccard,
                        'level_changes': level_changes,
                        'byte_changes': byte_changes,
                        'co_changes': co_changes,
                        'n_samples': len(samples),
                    })

    if diff_results:
        print(f"\n*** FOUND {len(diff_results)} DIFFERENTIAL MATCHES ***")
        for r in sorted(diff_results, key=lambda x: -x['jaccard']):
            print(f"  PID={r['pid']:4d} {r['offset']:15s} jaccard={r['jaccard']:.4f} "
                  f"co_changes={r['co_changes']}/{r['level_changes']}(lev)/{r['byte_changes']}(byte) "
                  f"samples={r['n_samples']}")
    else:
        print("No differential matches found (threshold jaccard > 0.5)")

    # ── ATTACK 4: Multi-byte varint after SKIP ──
    print("\n" + "=" * 70)
    print("ATTACK 4: Try decoding post-SKIP as varints and check against level")
    print("=" * 70)

    # Load all S-boxes for varint decoding attempts
    with open(RDATA_PATH, 'rb') as f:
        rdata = f.read()
    sbox_offsets = {
        'sbox_1056b0': 0x1056b0,
        'sbox_1057b0': 0x1057b0,
        'sbox_1059a0': 0x1059a0,
        'sbox_105400': 0x105400,
        'sbox_fa330': 0xfa330,
    }
    all_sboxes = {}
    for name, off in sbox_offsets.items():
        all_sboxes[name] = list(rdata[off:off + 256])

    varint_results = []

    for pid in sorted(pid_data.keys()):
        entries = pid_data[pid]
        if len(entries) < 10:
            continue

        frame_to_entries = defaultdict(list)
        for e in entries:
            frame_to_entries[e['frame']].append(e)

        samples = []
        for frame_idx in sorted(frame_to_entries.keys()):
            entry = frame_to_entries[frame_idx][0]
            frame_time = frame_idx * frame_duration
            level = get_stat_at_time(oracle_data, frame_time, 'level')
            if level is not None:
                samples.append({
                    'frame': frame_idx,
                    'level': level,
                    'post_skip': entry['post_skip'],
                })

        distinct_levels = len(set(s['level'] for s in samples))
        if distinct_levels < 5 or len(samples) < 10:
            continue

        min_post_size = min(len(s['post_skip']) for s in samples)

        # For each starting byte offset, try to decode a varint using each S-box
        # The per-PID cipher might be: sbox[byte] → 7-bit varint
        for sbox_name, sbox in all_sboxes.items():
            for start_off in range(min(min_post_size, 30)):
                levels = []
                decoded = []
                valid = True
                for s in samples:
                    ps = s['post_skip']
                    val = 0
                    shift = 0
                    ok = False
                    for i in range(start_off, min(start_off + 5, len(ps))):
                        d = sbox[ps[i]]
                        val |= (d & 0x7F) << shift
                        shift += 7
                        if not (d & 0x80):
                            ok = True
                            break
                    if not ok:
                        valid = False
                        break
                    decoded.append(val)
                    levels.append(s['level'])

                if not valid or len(decoded) < 10:
                    continue

                # Check if decoded values match level
                is_perm, mapping = test_arbitrary_permutation(decoded, levels)
                if is_perm and len(mapping) >= 5:
                    varint_results.append({
                        'pid': pid, 'sbox': sbox_name, 'start': start_off,
                        'transform': 'SBOX_VARINT',
                        'distinct': len(mapping),
                        'total_distinct': distinct_levels,
                        'n_samples': len(samples),
                        'mapping': {str(k): v for k, v in sorted(mapping.items())[:20]},
                    })

                # Also try: varint where decoded == level directly
                if all(d == l for d, l in zip(decoded, levels)):
                    varint_results.append({
                        'pid': pid, 'sbox': sbox_name, 'start': start_off,
                        'transform': 'SBOX_VARINT_EXACT',
                        'distinct': distinct_levels,
                        'total_distinct': distinct_levels,
                        'n_samples': len(samples),
                    })

        # Also try raw varint (no S-box, just 7-bit extraction)
        for start_off in range(min(min_post_size, 30)):
            decoded = []
            valid = True
            for s in samples:
                ps = s['post_skip']
                val = 0
                shift = 0
                ok = False
                for i in range(start_off, min(start_off + 5, len(ps))):
                    d = ps[i]
                    val |= (d & 0x7F) << shift
                    shift += 7
                    if not (d & 0x80):
                        ok = True
                        break
                if not ok:
                    valid = False
                    break
                decoded.append(val)

            if not valid or len(decoded) < 10:
                continue

            levels = [s['level'] for s in samples]
            is_perm, mapping = test_arbitrary_permutation(decoded, levels)
            if is_perm and len(mapping) >= 5:
                varint_results.append({
                    'pid': pid, 'sbox': 'NONE', 'start': start_off,
                    'transform': 'RAW_VARINT',
                    'distinct': len(mapping),
                    'total_distinct': distinct_levels,
                    'n_samples': len(samples),
                    'mapping': {str(k): v for k, v in sorted(mapping.items())[:20]},
                })

    if varint_results:
        print(f"\n*** FOUND {len(varint_results)} VARINT MATCHES ***")
        for r in sorted(varint_results, key=lambda x: (-x['distinct'], x['pid'])):
            print(f"  PID={r['pid']:4d} sbox={r['sbox']:15s} start={r['start']} "
                  f"type={r['transform']:20s} distinct={r['distinct']}/{r['total_distinct']} "
                  f"samples={r['n_samples']}")
            if 'mapping' in r and r['distinct'] >= 5:
                print(f"    Mapping: {r['mapping']}")
    else:
        print("No varint matches found.")

    # ── ATTACK 5: Check PID 318 timing against level-ups ──
    print("\n" + "=" * 70)
    print("ATTACK 5: PID 318 block timing vs level-up events")
    print("=" * 70)

    if 318 in pid_data:
        entries_318 = pid_data[318]
        print(f"PID 318: {len(entries_318)} blocks")
        for e in entries_318:
            ft = e['frame'] * frame_duration
            level = get_stat_at_time(oracle_data, ft, 'level')
            print(f"  frame={e['frame']:3d} time={ft:7.1f}s level={level:2d} "
                  f"skip_val={e['skip_val']} skip_len={e['skip_len']} "
                  f"post_skip={e['post_skip'].hex()[:40]} "
                  f"ch=0x{e['channel']:02x}")

    # ── ATTACK 6: Check if payload SIZE correlates with level ──
    print("\n" + "=" * 70)
    print("ATTACK 6: Payload size vs level correlation")
    print("=" * 70)

    for pid in sorted(pid_data.keys()):
        entries = pid_data[pid]
        if len(entries) < 10:
            continue

        frame_to_entries = defaultdict(list)
        for e in entries:
            frame_to_entries[e['frame']].append(e)

        sizes = []
        levels = []
        for frame_idx in sorted(frame_to_entries.keys()):
            entry = frame_to_entries[frame_idx][0]
            frame_time = frame_idx * frame_duration
            level = get_stat_at_time(oracle_data, frame_time, 'level')
            if level is not None:
                sizes.append(len(entry['raw_payload']))
                levels.append(level)

        if len(set(sizes)) < 3 or len(set(levels)) < 5:
            continue

        rho, pval = scipy_stats.spearmanr(sizes, levels)
        if abs(rho) > 0.5 and pval < 0.01:
            print(f"  PID {pid:4d}: size-level rho={rho:.4f} p={pval:.2e} "
                  f"sizes={sorted(set(sizes))[:10]}")

    # ── Summary ──
    print("\n" + "=" * 70)
    total = len(results) + len(corr_results) + len(diff_results) + len(varint_results)
    print(f"TOTAL MATCHES ACROSS ALL ATTACKS: {total}")
    print("=" * 70)

    all_data = {
        'permutation': results,
        'correlation': corr_results,
        'differential': diff_results,
        'varint': varint_results,
    }
    with open('/tmp/kpa_v2_results.json', 'w') as f:
        json.dump(all_data, f, indent=2, default=str)
    print("Results saved to /tmp/kpa_v2_results.json")


if __name__ == '__main__':
    main()
