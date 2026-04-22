#!/usr/bin/env python3
"""
Focused search: For STAT PIDs (842, 368, 224) in entity b2 blocks,
try every decoding of the payload bytes to find level/CS.

Approach: For each block, decode the payload with:
- Raw bytes at each offset (u8, u16 LE, u32 LE, float32)
- Varint decoding starting at each offset
- XOR/ADD u8 with every constant 0-255
- S-box lookups from known rdata tables
- The SKIP varint cipher chain
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
STAT_PIDS = {842, 368, 224}

def load_sbox(rdata_raw, offset):
    return list(rdata_raw[offset:offset+256])

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

def extract_blocks_for_entity(frames, entity, pids=None):
    """Extract blocks matching entity and optionally filter by PIDs."""
    entity_bytes = struct.pack('<I', entity)
    pid_blocks = defaultdict(list)

    for fi, fd in enumerate(frames):
        pos = 3
        while pos < len(fd) - 6:
            # Find any of our target PIDs
            best_idx = len(fd)
            for pid in (pids or range(1200)):
                idx = fd.find(struct.pack('<H', pid), pos)
                if 0 <= idx < best_idx:
                    best_idx = idx

            if best_idx >= len(fd):
                break

            idx = best_idx
            block_start = idx - 3
            if block_start >= 0 and fd[block_start] in BLOCK_MARKERS:
                pid = struct.unpack_from('<H', fd, idx)[0]
                size = fd[block_start + 2]
                param = struct.unpack_from('<I', fd, block_start + 5)[0]
                end = block_start + 9 + size
                if end <= len(fd) and size > 0 and param == entity:
                    if pids is None or pid in pids:
                        pid_blocks[pid].append({
                            'frame': fi,
                            'payload': bytes(fd[block_start + 9:end]),
                            'channel': fd[block_start + 1],
                            'marker': fd[block_start],
                            'size': size,
                        })
            pos = best_idx + 1

    return pid_blocks

def extract_all_entity_blocks(frames, entity):
    """Faster: just search for entity bytes and backtrack to find block header."""
    entity_bytes = struct.pack('<I', entity)
    pid_blocks = defaultdict(list)

    for fi, fd in enumerate(frames):
        pos = 0
        while pos < len(fd):
            idx = fd.find(entity_bytes, pos)
            if idx < 0:
                break
            # Entity is at offset +5 in block header
            block_start = idx - 5
            if block_start >= 0 and fd[block_start] in BLOCK_MARKERS:
                size = fd[block_start + 2]
                pid = struct.unpack_from('<H', fd, block_start + 3)[0]
                end = block_start + 9 + size
                if end <= len(fd) and size > 0:
                    pid_blocks[pid].append({
                        'frame': fi,
                        'payload': bytes(fd[block_start + 9:end]),
                        'channel': fd[block_start + 1],
                        'marker': fd[block_start],
                        'size': size,
                    })
            pos = idx + 1

    return pid_blocks

def read_varint(data, offset):
    """Read 7-bit varint starting at offset. Returns (value, bytes_consumed)."""
    result = 0
    shift = 0
    i = offset
    while i < len(data):
        b = data[i]
        result |= (b & 0x7F) << shift
        i += 1
        if (b & 0x80) == 0:
            return result, i - offset
        shift += 7
        if shift > 35:
            break
    return None, 0

def zigzag_decode(n):
    return (n >> 1) ^ -(n & 1)

def main():
    # Load oracle
    with open(ORACLE_PATH) as f:
        oracle = json.load(f)
    game_length = oracle['game_length']
    garen_snaps = []
    for snap in oracle['snapshots']:
        if 'Garen' in snap['players']:
            g = snap['players']['Garen']
            garen_snaps.append({
                'time': snap['actual_time'],
                'level': g['level'],
                'cs': g['scores']['creepScore'],
                'kills': g['scores']['kills'],
                'deaths': g['scores']['deaths'],
                'assists': g['scores']['assists'],
            })

    # Load S-boxes from rdata
    pe_dir = '/tmp/pe_dump_16.7'
    rdata_raw = open(os.path.join(pe_dir, 'rdata.bin'), 'rb').read()
    sbox_offsets = [0x1056b0, 0x1057b0, 0x1059a0, 0x105400, 0xfa330, 0x1199f0]
    # Adjust for rdata RVA
    rdata_rva = 0x18fd000
    sboxes = {}
    for off in sbox_offsets:
        adj = off  # These are rdata-relative offsets
        if adj + 256 <= len(rdata_raw):
            sboxes[off] = load_sbox(rdata_raw, adj)

    print(f"Loaded {len(sboxes)} S-boxes, {len(garen_snaps)} oracle snapshots")

    # Extract frames
    print("Extracting frames...")
    frames = extract_frames(ROFL_PATH)
    n_frames = len(frames)
    frame_dur = game_length / max(n_frames - 1, 1)
    print(f"{n_frames} frames, {frame_dur:.1f}s per frame")

    # Extract ALL blocks for entity b2
    print("Extracting entity b2 blocks...")
    pid_blocks = extract_all_entity_blocks(frames, GAREN_ENTITY)

    print(f"\nPIDs found for entity b2:")
    for pid in sorted(pid_blocks.keys()):
        blocks = pid_blocks[pid]
        sizes = set(b['size'] for b in blocks)
        print(f"  PID {pid:4d}: {len(blocks):3d} blocks, sizes={sorted(sizes)}")

    # Focus on stat PIDs
    print("\n" + "=" * 70)
    print("STAT PID ANALYSIS")
    print("=" * 70)

    def get_stat(t, stat):
        v = None
        for snap in garen_snaps:
            if snap['time'] <= t + 1.0:
                v = snap[stat]
            else:
                break
        return v

    for pid in sorted(STAT_PIDS):
        blocks = pid_blocks.get(pid, [])
        if not blocks:
            print(f"\nPID {pid}: NO BLOCKS for entity b2")
            continue

        print(f"\n{'='*70}")
        print(f"PID {pid}: {len(blocks)} blocks")
        print(f"{'='*70}")

        # Show raw hex for first few blocks at different levels
        seen_levels = set()
        sample_blocks = []
        for b in sorted(blocks, key=lambda x: x['frame']):
            ft = b['frame'] * frame_dur
            level = get_stat(ft, 'level')
            if level and level not in seen_levels:
                seen_levels.add(level)
                sample_blocks.append((b, level, ft))

        print(f"  Distinct levels covered: {sorted(seen_levels)}")
        print(f"\n  Raw payloads at different levels:")
        for b, level, ft in sample_blocks[:18]:
            cs = get_stat(ft, 'cs')
            kills = get_stat(ft, 'kills')
            deaths = get_stat(ft, 'deaths')
            assists = get_stat(ft, 'assists')
            payload = b['payload']
            hex_str = payload.hex()
            print(f"    L={level:2d} cs={cs:3d} k={kills} d={deaths} a={assists} "
                  f"ch=0x{b['channel']:02x} [{len(payload):2d}B]: {hex_str}")

        # Try decoding each byte offset
        print(f"\n  === BYTE-BY-BYTE ANALYSIS ===")
        max_payload = max(len(b['payload']) for b in blocks)

        # For each offset, collect (value, level) pairs
        for stat_name in ['level', 'cs', 'kills']:
            stat_idx = {'level': 'level', 'cs': 'cs', 'kills': 'kills'}[stat_name]

            # u8 raw match
            for off in range(max_payload):
                pairs = []
                for b in blocks:
                    if off < len(b['payload']):
                        ft = b['frame'] * frame_dur
                        sv = get_stat(ft, stat_idx)
                        if sv is not None:
                            pairs.append((b['payload'][off], sv))

                if len(pairs) < 5:
                    continue

                # Check exact match
                matches = sum(1 for v, s in pairs if v == s)
                if matches > len(pairs) * 0.6:
                    print(f"    u8 offset={off}: {matches}/{len(pairs)} exact match {stat_name}")
                    for v, s in pairs[:10]:
                        print(f"      byte=0x{v:02x}({v:3d}) {stat_name}={s}")

                # Check XOR with each constant
                for xk in range(1, 256):
                    matches = sum(1 for v, s in pairs if (v ^ xk) == s)
                    if matches > len(pairs) * 0.7:
                        print(f"    u8 offset={off} XOR 0x{xk:02x}: {matches}/{len(pairs)} match {stat_name}")

                # Check ADD/SUB
                for ak in range(1, 256):
                    matches = sum(1 for v, s in pairs if ((v + ak) & 0xFF) == s)
                    if matches > len(pairs) * 0.7:
                        print(f"    u8 offset={off} ADD 0x{ak:02x}: {matches}/{len(pairs)} match {stat_name}")
                    matches = sum(1 for v, s in pairs if ((v - ak) & 0xFF) == s)
                    if matches > len(pairs) * 0.7:
                        print(f"    u8 offset={off} SUB 0x{ak:02x}: {matches}/{len(pairs)} match {stat_name}")

                # S-box lookups
                for soff, sbox in sboxes.items():
                    matches = sum(1 for v, s in pairs if sbox[v] == s)
                    if matches > len(pairs) * 0.7:
                        print(f"    u8 offset={off} SBOX[0x{soff:x}]: {matches}/{len(pairs)} match {stat_name}")

            # u16 LE raw match
            for off in range(max_payload - 1):
                pairs = []
                for b in blocks:
                    if off + 1 < len(b['payload']):
                        ft = b['frame'] * frame_dur
                        sv = get_stat(ft, stat_idx)
                        if sv is not None:
                            v16 = struct.unpack_from('<H', b['payload'], off)[0]
                            pairs.append((v16, sv))

                if len(pairs) < 5:
                    continue

                matches = sum(1 for v, s in pairs if v == s)
                if matches > len(pairs) * 0.6:
                    print(f"    u16LE offset={off}: {matches}/{len(pairs)} exact match {stat_name}")

                # XOR u16
                for xk in range(1, 256):
                    matches = sum(1 for v, s in pairs if (v ^ xk) == s)
                    if matches > len(pairs) * 0.7:
                        print(f"    u16LE offset={off} XOR 0x{xk:02x}: {matches}/{len(pairs)} match {stat_name}")

            # u32 LE as float
            for off in range(max_payload - 3):
                pairs = []
                for b in blocks:
                    if off + 3 < len(b['payload']):
                        ft = b['frame'] * frame_dur
                        sv = get_stat(ft, stat_idx)
                        if sv is not None:
                            v32 = struct.unpack_from('<I', b['payload'], off)[0]
                            try:
                                fval = struct.unpack('<f', struct.pack('<I', v32))[0]
                                if not (fval != fval):  # not NaN
                                    pairs.append((fval, sv))
                            except:
                                pass

                if len(pairs) < 5:
                    continue

                matches = sum(1 for v, s in pairs if abs(v - s) < 0.01)
                if matches > len(pairs) * 0.6:
                    print(f"    f32 offset={off}: {matches}/{len(pairs)} exact match {stat_name}")

            # Varint at each offset
            for off in range(max_payload):
                pairs = []
                for b in blocks:
                    if off < len(b['payload']):
                        ft = b['frame'] * frame_dur
                        sv = get_stat(ft, stat_idx)
                        if sv is not None:
                            vint, consumed = read_varint(b['payload'], off)
                            if vint is not None:
                                pairs.append((vint, sv))
                                # Also try zigzag

                if len(pairs) < 5:
                    continue

                # Raw varint
                matches = sum(1 for v, s in pairs if v == s)
                if matches > len(pairs) * 0.6:
                    print(f"    varint offset={off}: {matches}/{len(pairs)} exact match {stat_name}")

                # Zigzag varint
                pairs_zz = [(zigzag_decode(v), s) for v, s in pairs]
                matches = sum(1 for v, s in pairs_zz if v == s)
                if matches > len(pairs_zz) * 0.6:
                    print(f"    zigzag_varint offset={off}: {matches}/{len(pairs_zz)} exact match {stat_name}")

        # Differential: look for bytes that CHANGE when level changes
        print(f"\n  === DIFFERENTIAL ANALYSIS (bytes that change with level) ===")
        prev_payload = None
        prev_level = None
        changes_at_level_change = defaultdict(int)
        changes_at_no_level_change = defaultdict(int)
        level_transitions = 0
        same_level_count = 0

        sorted_blocks = sorted(blocks, key=lambda x: x['frame'])
        for b in sorted_blocks:
            ft = b['frame'] * frame_dur
            level = get_stat(ft, 'level')
            if level is None:
                continue

            if prev_payload is not None and prev_level is not None:
                min_len = min(len(b['payload']), len(prev_payload))
                if level != prev_level:
                    level_transitions += 1
                    for i in range(min_len):
                        if b['payload'][i] != prev_payload[i]:
                            changes_at_level_change[i] += 1
                else:
                    same_level_count += 1
                    for i in range(min_len):
                        if b['payload'][i] != prev_payload[i]:
                            changes_at_no_level_change[i] += 1

            prev_payload = b['payload']
            prev_level = level

        print(f"    Level transitions: {level_transitions}, same-level pairs: {same_level_count}")
        if level_transitions > 0:
            for off in sorted(changes_at_level_change.keys()):
                lc = changes_at_level_change[off]
                nlc = changes_at_no_level_change.get(off, 0)
                lc_rate = lc / level_transitions
                nlc_rate = nlc / max(same_level_count, 1)
                # Bytes that change MORE at level transitions than otherwise
                if lc_rate > 0.3 and lc_rate > nlc_rate * 1.5:
                    print(f"    offset {off}: changes {lc}/{level_transitions} at level change, "
                          f"{nlc}/{same_level_count} otherwise (ratio={lc_rate/max(nlc_rate,0.01):.1f}x)")

        # Also check: does the PAYLOAD SIZE correlate with level?
        size_level_pairs = []
        for b in blocks:
            ft = b['frame'] * frame_dur
            level = get_stat(ft, 'level')
            if level is not None:
                size_level_pairs.append((len(b['payload']), level))

        if len(size_level_pairs) > 5:
            sizes_by_level = defaultdict(list)
            for sz, lev in size_level_pairs:
                sizes_by_level[lev].append(sz)
            print(f"\n  Payload size by level:")
            for lev in sorted(sizes_by_level.keys()):
                szs = sizes_by_level[lev]
                print(f"    L={lev:2d}: sizes={sorted(set(szs))}, count={len(szs)}")


if __name__ == '__main__':
    main()
