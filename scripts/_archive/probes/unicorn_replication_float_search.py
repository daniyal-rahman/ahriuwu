#!/usr/bin/env python3
"""
Run PID 842 (Replication) deserializer through Unicorn WITHOUT SKIP patch.
Check if any 4-byte writes decode to float32 values matching oracle stats.

maknee's Replication format: {primary_index, secondary_index, name, data: {Float: value}}
The deserializer should output (property_id, float_value) pairs.
"""
import struct
import json
import os
import zstandard as zstd
from collections import defaultdict

ROFL_PATH = 'data/replays/NA1-5528069928.rofl'
ORACLE_PATH = '/tmp/oracle_data.json'
GAREN_ENTITY = 0x400000b2
BLOCK_MARKERS = {0x91, 0xf1, 0xb1, 0x31, 0x11}

from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_PROT_ALL, UC_PROT_READ, UC_PROT_WRITE
from unicorn import UC_HOOK_MEM_READ_UNMAPPED, UC_HOOK_MEM_WRITE_UNMAPPED, UC_HOOK_MEM_FETCH_UNMAPPED, UC_HOOK_CODE, UC_HOOK_MEM_WRITE
from unicorn.x86_const import *

BASE_ADDR    = 0x140000000
STACK_BASE   = 0x7FFFFFFF0000
STACK_SIZE   = 0x4000
HEAP_BASE    = 0x7FFFFFFFA000
ALLOC_BASE   = 0x7FFFFFF80000
ALLOC_SIZE   = 0x40000
SCRATCH_BASE = 0x7FFFFFF00000
STOP_ADDR    = 0xDEAD0000

PID842_DESER_RVA = 0x101e8b0
PID842_VTABLE = 0x19fde98
PID842_CTOR = 0xe03ef0

# XP thresholds
XP_THRESHOLDS = {1: 0, 2: 280, 3: 660, 4: 1140, 5: 1720, 6: 2400, 7: 3180,
    8: 4060, 9: 5040, 10: 6120, 11: 7300, 12: 8580, 13: 9960, 14: 11440,
    15: 13020, 16: 14700, 17: 16480, 18: 18360}

def align_down(a, s=0x1000): return a & ~(s-1)
def align_up(s, p=0x1000): return (s+p-1) & ~(p-1)

def extract_frames(rofl_path):
    with open(rofl_path, 'rb') as f:
        data = f.read()
    dctx = zstd.ZstdDecompressor()
    frames = []
    pos = 0
    while pos < len(data):
        idx = data.find(b'\x28\xb5\x2f\xfd', pos)
        if idx < 0: break
        try:
            dec = dctx.decompress(data[idx:idx + 0x100000], max_output_size=0x400000)
            frames.append(dec)
        except: pass
        pos = idx + 1
    return frames

def extract_entity_blocks(frames, entity, pid):
    entity_bytes = struct.pack('<I', entity)
    blocks = []
    for fi, fd in enumerate(frames):
        pos = 0
        while pos < len(fd):
            idx = fd.find(entity_bytes, pos)
            if idx < 0: break
            bs = idx - 5
            if bs >= 0 and fd[bs] in BLOCK_MARKERS:
                sz = fd[bs + 2]
                p = struct.unpack_from('<H', fd, bs + 3)[0]
                end = bs + 9 + sz
                if end <= len(fd) and sz > 0 and p == pid:
                    blocks.append({'frame': fi, 'payload': bytes(fd[bs+9:end]),
                                  'channel': fd[bs+1], 'size': sz})
            pos = idx + 1
    return blocks

def setup_emulator():
    pe_dir = '/tmp/pe_dump_16.7'
    text_raw = open(os.path.join(pe_dir, 'text.bin'), 'rb').read()
    rdata_raw = open(os.path.join(pe_dir, 'rdata.bin'), 'rb').read()
    data_raw = open(os.path.join(pe_dir, 'data.bin'), 'rb').read()

    mu = Uc(UC_ARCH_X86, UC_MODE_64)
    mu.mem_map(STACK_BASE, STACK_SIZE, UC_PROT_READ | UC_PROT_WRITE)
    mu.mem_map(HEAP_BASE, align_up(0x20000), UC_PROT_READ | UC_PROT_WRITE)
    mu.mem_map(SCRATCH_BASE, 0x10000, UC_PROT_READ | UC_PROT_WRITE)
    mu.mem_map(ALLOC_BASE, ALLOC_SIZE, UC_PROT_ALL)

    pe_start = align_down(BASE_ADDR + 0x1000)
    pe_end = align_up(BASE_ADDR + 0x1d21000 + len(data_raw) + 0x1000)
    mu.mem_map(pe_start, pe_end - pe_start, UC_PROT_ALL)
    mu.mem_write(BASE_ADDR + 0x1000, text_raw)
    mu.mem_write(BASE_ADDR + 0x18fd000, rdata_raw)
    mu.mem_write(BASE_ADDR + 0x1d21000, data_raw)

    # Do NOT patch SKIP - this is maknee's key insight
    # mu.mem_write(BASE_ADDR + 0x118b120, b'\x48\xC7\xC0\x01\x00\x00\x00\xC3')
    mu.mem_write(BASE_ADDR + 0x10fa150, b'\xC3')  # free → ret

    mu.mem_map(align_down(STOP_ADDR), 0x1000, UC_PROT_ALL)
    mu.mem_write(STOP_ADDR, b'\xF4')

    def on_unmap(mu, access, addr, sz, val, ud):
        try:
            mu.mem_map(align_down(addr), 0x1000, UC_PROT_ALL)
            return True
        except: return False
    mu.hook_add(UC_HOOK_MEM_READ_UNMAPPED | UC_HOOK_MEM_WRITE_UNMAPPED |
                UC_HOOK_MEM_FETCH_UNMAPPED, on_unmap)

    state = {'alloc_cursor': 0}
    def malloc_hook(mu, address, size, user_data):
        if address == BASE_ADDR + 0x10fa120:
            sz = mu.reg_read(UC_X86_REG_RCX)
            aligned = max((sz + 15) & ~15, 0x200)
            ptr = ALLOC_BASE + state['alloc_cursor']
            state['alloc_cursor'] += aligned
            mu.reg_write(UC_X86_REG_RAX, ptr)
            rsp = mu.reg_read(UC_X86_REG_RSP)
            ret = struct.unpack('<Q', mu.mem_read(rsp, 8))[0]
            mu.reg_write(UC_X86_REG_RSP, rsp + 8)
            mu.reg_write(UC_X86_REG_RIP, ret)
    mu.hook_add(UC_HOOK_CODE, malloc_hook,
                begin=BASE_ADDR + 0x10fa120, end=BASE_ADDR + 0x10fa121)

    # Run constructor
    mu.mem_write(HEAP_BASE, b'\x00' * 0x400)
    mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
    mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
    rsp = mu.reg_read(UC_X86_REG_RSP) - 8
    mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
    mu.reg_write(UC_X86_REG_RSP, rsp)
    mu.emu_start(BASE_ADDR + PID842_CTOR, STOP_ADDR, timeout=5000000)
    ctor_data = bytes(mu.mem_read(HEAP_BASE, 0x400))

    return mu, ctor_data, state

def decode_block(mu, ctor_data, state, payload):
    state['alloc_cursor'] = 0
    mu.mem_write(ALLOC_BASE, b'\x00' * min(0x8000, ALLOC_SIZE))
    mu.mem_write(HEAP_BASE, b'\x00' * 0x400)
    mu.mem_write(HEAP_BASE, ctor_data)
    mu.mem_write(HEAP_BASE, struct.pack('<Q', BASE_ADDR + PID842_VTABLE))

    pay_addr = SCRATCH_BASE + 0x100
    mu.mem_write(pay_addr, payload + b'\x00' * 128)
    pp = SCRATCH_BASE + 0x200
    mu.mem_write(pp, struct.pack('<Q', pay_addr))
    mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
    mu.reg_write(UC_X86_REG_RDX, pp)
    mu.reg_write(UC_X86_REG_R8, pay_addr + len(payload))
    mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
    rsp = mu.reg_read(UC_X86_REG_RSP) - 8
    mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
    mu.reg_write(UC_X86_REG_RSP, rsp)

    writes = []
    def track(mu, access, address, size, value, user_data):
        val = value & ((1 << (size * 8)) - 1)
        writes.append((address, size, val))
        return True

    h1 = mu.hook_add(UC_HOOK_MEM_WRITE, track,
                     begin=ALLOC_BASE, end=ALLOC_BASE + ALLOC_SIZE)
    h2 = mu.hook_add(UC_HOOK_MEM_WRITE, track,
                     begin=HEAP_BASE, end=HEAP_BASE + 0x2000)
    try:
        mu.emu_start(BASE_ADDR + PID842_DESER_RVA, STOP_ADDR, timeout=10000000)
        ok = True
    except:
        ok = False
    mu.hook_del(h1)
    mu.hook_del(h2)
    return writes, ok

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
                'deaths': g['scores']['deaths'], 'assists': g['scores']['assists'],
                'gold': g.get('currentGold', 0),
                'ward_score': g['scores'].get('wardScore', 0),
            })

    def get_snap(t):
        best = None
        for snap in garen_snaps:
            if snap['time'] <= t + 1.0: best = snap
            else: break
        return best

    print("Extracting frames...")
    frames = extract_frames(ROFL_PATH)
    n_frames = len(frames)
    frame_dur = game_length / max(n_frames - 1, 1)

    print("Extracting PID 842 blocks...")
    blocks = extract_entity_blocks(frames, GAREN_ENTITY, 842)
    print(f"  {len(blocks)} PID 842 blocks")

    print("Setting up emulator (NO SKIP patch)...")
    mu, ctor_data, state = setup_emulator()
    print("  Ready.\n")

    # Take one block per frame, at different levels
    frame_blocks = {}
    for b in blocks:
        if b['frame'] not in frame_blocks:
            frame_blocks[b['frame']] = b

    # Decode all blocks and collect float32 values
    print(f"Decoding {len(frame_blocks)} blocks...")

    # Track: for each (address_offset, size), collect [(frame, float_val, snap)]
    float_history = defaultdict(list)  # (region, rel_offset) → [(frame, fval, snap)]
    errors = 0

    for i, (fi, blk) in enumerate(sorted(frame_blocks.items())):
        ft = fi * frame_dur
        snap = get_snap(ft)
        if snap is None:
            continue

        writes, ok = decode_block(mu, ctor_data, state, blk['payload'])
        if not ok:
            errors += 1
            continue

        # Extract all 4-byte writes as float32
        for addr, sz, val in writes:
            if sz == 4:
                try:
                    fval = struct.unpack('<f', struct.pack('<I', val & 0xFFFFFFFF))[0]
                except:
                    continue
                if fval != fval:  # NaN
                    continue

                if addr >= ALLOC_BASE and addr < ALLOC_BASE + ALLOC_SIZE:
                    region = 'ALLOC'
                    rel = addr - ALLOC_BASE
                elif addr >= HEAP_BASE and addr < HEAP_BASE + 0x2000:
                    region = 'HEAP'
                    rel = addr - HEAP_BASE
                else:
                    continue

                float_history[(region, rel)].append((fi, fval, snap))

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(frame_blocks)}...")

    print(f"\nDecoded {len(frame_blocks) - errors} blocks ({errors} errors)")
    print(f"Unique float write positions: {len(float_history)}")

    # Check float values against oracle stats
    print("\n" + "=" * 70)
    print("SEARCH: Float values matching oracle gold")
    print("=" * 70)

    for key, entries in sorted(float_history.items()):
        if len(entries) < 5:
            continue
        # Gold is integer but could be stored as float
        matches = sum(1 for _, fval, snap in entries
                     if snap and snap['gold'] > 0 and abs(fval - snap['gold']) < 5.0)
        if matches > len(entries) * 0.3:
            region, off = key
            print(f"  {region}+0x{off:04x}: {matches}/{len(entries)} match gold")
            for fi, fval, snap in entries[:8]:
                print(f"    frame={fi:3d} fval={fval:.1f} gold={snap['gold']}")

    print("\n" + "=" * 70)
    print("SEARCH: Float values matching CS (creepScore)")
    print("=" * 70)

    for key, entries in sorted(float_history.items()):
        if len(entries) < 5:
            continue
        matches = sum(1 for _, fval, snap in entries
                     if snap and abs(fval - snap['cs']) < 1.5)
        if matches > len(entries) * 0.3:
            region, off = key
            print(f"  {region}+0x{off:04x}: {matches}/{len(entries)} match CS")
            for fi, fval, snap in entries[:8]:
                print(f"    frame={fi:3d} fval={fval:.1f} cs={snap['cs']}")

    print("\n" + "=" * 70)
    print("SEARCH: Float values matching XP thresholds for level")
    print("=" * 70)

    for key, entries in sorted(float_history.items()):
        if len(entries) < 5:
            continue
        xp_matches = 0
        for _, fval, snap in entries:
            if snap:
                lev = snap['level']
                xp_min = XP_THRESHOLDS.get(lev, 0)
                xp_max = XP_THRESHOLDS.get(lev + 1, 99999) - 1
                if xp_min <= fval <= xp_max:
                    xp_matches += 1
        if xp_matches > len(entries) * 0.5:
            region, off = key
            print(f"  {region}+0x{off:04x}: {xp_matches}/{len(entries)} match XP range")
            for fi, fval, snap in entries[:8]:
                lev = snap['level']
                xp_min = XP_THRESHOLDS.get(lev, 0)
                xp_max = XP_THRESHOLDS.get(lev + 1, 99999) - 1
                hit = "Y" if xp_min <= fval <= xp_max else "N"
                print(f"    frame={fi:3d} fval={fval:.1f} level={lev} xp_range=({xp_min},{xp_max}) {hit}")

    print("\n" + "=" * 70)
    print("SEARCH: Float == level (mLevelRef)")
    print("=" * 70)

    for key, entries in sorted(float_history.items()):
        if len(entries) < 5:
            continue
        matches = sum(1 for _, fval, snap in entries
                     if snap and abs(fval - snap['level']) < 0.5)
        if matches > len(entries) * 0.3:
            region, off = key
            print(f"  {region}+0x{off:04x}: {matches}/{len(entries)} match level as float")
            for fi, fval, snap in entries[:8]:
                print(f"    frame={fi:3d} fval={fval:.3f} level={snap['level']}")

    # Show ALL distinct float values for each write position
    print("\n" + "=" * 70)
    print("ALL FLOAT WRITES: Summary of distinct values per position")
    print("=" * 70)

    for key in sorted(float_history.keys()):
        entries = float_history[key]
        if len(entries) < 3:
            continue
        region, off = key
        fvals = [fval for _, fval, _ in entries]
        distinct = len(set(round(f, 2) for f in fvals))
        if distinct > 1:
            vmin, vmax = min(fvals), max(fvals)
            # Show representative values
            samples = list(set(round(f, 2) for f in fvals))[:10]
            print(f"  {region}+0x{off:04x}: {len(entries)} writes, {distinct} distinct, "
                  f"range=[{vmin:.2f}, {vmax:.2f}]  samples={sorted(samples)[:8]}")


if __name__ == '__main__':
    main()
