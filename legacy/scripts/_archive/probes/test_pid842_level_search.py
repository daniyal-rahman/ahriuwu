#!/usr/bin/env python3
"""
Run PID 842 deserializer on ALL entity b2 blocks and find which decoded
write offset consistently equals Garen's level.
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

def align_down(a, s=0x1000): return a & ~(s-1)
def align_up(s, p=0x1000): return (s+p-1) & ~(p-1)


def extract_blocks(rofl_path, pid, entity):
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

    pid_bytes = struct.pack('<H', pid)
    all_blocks = []
    for fi, fd in enumerate(frames):
        pos_s = 3
        while pos_s < len(fd) - 6:
            idx = fd.find(pid_bytes, pos_s)
            if idx < 0:
                break
            bs = idx - 3
            if bs >= 0 and fd[bs] in BLOCK_MARKERS:
                size = fd[bs + 2]
                param = struct.unpack_from('<I', fd, bs + 5)[0]
                end = bs + 9 + size
                if end <= len(fd) and size > 0:
                    vc = (end >= len(fd) - 9 or (end < len(fd) and fd[end] in BLOCK_MARKERS))
                    if vc and param == entity:
                        all_blocks.append({'frame': fi, 'payload': bytes(fd[bs + 9:end]),
                                          'channel': fd[bs + 1]})
            pos_s = idx + 1
    return all_blocks, len(frames)


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

    mu.mem_write(BASE_ADDR + 0x118b120, b'\x48\xC7\xC0\x01\x00\x00\x00\xC3')  # SKIP → ret 1
    mu.mem_write(BASE_ADDR + 0x10fa150, b'\xC3')  # free → ret

    mu.mem_map(align_down(STOP_ADDR), 0x1000, UC_PROT_ALL)
    mu.mem_write(STOP_ADDR, b'\xF4')

    def on_unmap(mu, access, addr, sz, val, ud):
        try:
            mu.mem_map(align_down(addr), 0x1000, UC_PROT_ALL)
            return True
        except:
            return False
    mu.hook_add(UC_HOOK_MEM_READ_UNMAPPED | UC_HOOK_MEM_WRITE_UNMAPPED |
                UC_HOOK_MEM_FETCH_UNMAPPED, on_unmap)

    state = {'alloc_cursor': 0, 'alloc_regions': []}
    def malloc_hook(mu, address, size, user_data):
        if address == BASE_ADDR + 0x10fa120:
            sz = mu.reg_read(UC_X86_REG_RCX)
            aligned = max((sz + 15) & ~15, 0x200)
            ptr = ALLOC_BASE + state['alloc_cursor']
            state['alloc_regions'].append((state['alloc_cursor'], sz))
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
    ctor_data = bytes(mu.mem_read(HEAP_BASE, 0x200))

    return mu, ctor_data, state


def decode_block(mu, ctor_data, state, payload):
    """Decode one PID 842 block. Returns dict of {(region, offset, size): value}."""
    state['alloc_cursor'] = 0
    state['alloc_regions'] = []
    mu.mem_write(ALLOC_BASE, b'\x00' * min(0x8000, ALLOC_SIZE))
    mu.mem_write(HEAP_BASE, b'\x00' * 0x200)
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

    writes = {}
    def track(mu, access, address, size, value, user_data):
        val = value & ((1 << (size * 8)) - 1)
        off = address - ALLOC_BASE
        if 0 <= off < ALLOC_SIZE:
            writes[('ALLOC', off, size)] = val
        soff = address - HEAP_BASE
        if 0 <= soff < 0x400:
            writes[('HEAP', soff, size)] = val
        return True

    h1 = mu.hook_add(UC_HOOK_MEM_WRITE, track,
                     begin=ALLOC_BASE, end=ALLOC_BASE + ALLOC_SIZE)
    h2 = mu.hook_add(UC_HOOK_MEM_WRITE, track,
                     begin=HEAP_BASE, end=HEAP_BASE + 0x400)
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
            garen_snaps.append({'time': snap['actual_time'], 'level': g['level'],
                                'cs': g['scores']['creepScore'], 'kills': g['scores']['kills']})

    def get_stat(t, s):
        v = None
        for snap in garen_snaps:
            if snap['time'] <= t + 1.0: v = snap[s]
            else: break
        return v

    blocks, n_frames = extract_blocks(ROFL_PATH, 842, GAREN_ENTITY)
    frame_dur = game_length / max(n_frames - 1, 1)
    print(f"{len(blocks)} PID 842 blocks for entity b2, {n_frames} frames")

    mu, ctor_data, state = setup_emulator()
    print("Emulator ready.\n")

    # Decode ALL blocks and collect (write_key → [(frame, level, value)])
    write_history = defaultdict(list)  # (region, offset, size) → [(frame, level, cs, kills, value)]
    errors = 0

    # Take one block per frame
    frame_blocks = {}
    for b in blocks:
        if b['frame'] not in frame_blocks:
            frame_blocks[b['frame']] = b

    print(f"Processing {len(frame_blocks)} unique frames...")

    for i, (fi, blk) in enumerate(sorted(frame_blocks.items())):
        ft = fi * frame_dur
        level = get_stat(ft, 'level')
        cs = get_stat(ft, 'cs')
        kills = get_stat(ft, 'kills')
        if level is None:
            continue

        writes, ok = decode_block(mu, ctor_data, state, blk['payload'])
        if not ok:
            errors += 1
            continue

        for key, val in writes.items():
            write_history[key].append((fi, level, cs, kills, val))

        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(frame_blocks)}...")

    print(f"\nDecoded {len(frame_blocks) - errors} blocks ({errors} errors)")
    print(f"Unique write keys: {len(write_history)}")

    # ── Find writes where value == level ──
    print("\n" + "=" * 70)
    print("SEARCH: Write offsets where decoded value EQUALS level")
    print("=" * 70)

    level_matches = []
    for key, entries in write_history.items():
        if len(entries) < 5:
            continue
        matches = sum(1 for _, lev, _, _, val in entries if val == lev)
        total = len(entries)
        if matches > total * 0.7:  # >70% match rate
            level_matches.append((key, matches, total, entries))

    if level_matches:
        print(f"\n*** {len(level_matches)} offsets match level >70% of the time ***")
        for key, matches, total, entries in sorted(level_matches, key=lambda x: -x[1]):
            region, off, sz = key
            print(f"\n  {region} +0x{off:04x} [{sz}B]: {matches}/{total} match level")
            distinct = len(set(e[1] for e in entries))
            print(f"  Distinct levels covered: {distinct}")
            for fi, lev, cs, kills, val in entries[:15]:
                match = "✓" if val == lev else "✗"
                print(f"    frame={fi:3d} level={lev:2d} val={val:6d} {match}")
    else:
        print("No offset matches level >70%")

    # ── Find writes where value == CS ──
    print("\n" + "=" * 70)
    print("SEARCH: Write offsets where decoded value EQUALS CS")
    print("=" * 70)

    cs_matches = []
    for key, entries in write_history.items():
        if len(entries) < 5:
            continue
        matches = sum(1 for _, _, cs, _, val in entries if cs is not None and val == cs)
        total = len(entries)
        if matches > total * 0.5:
            cs_matches.append((key, matches, total, entries))

    if cs_matches:
        print(f"\n*** {len(cs_matches)} offsets match CS >50% of the time ***")
        for key, matches, total, entries in sorted(cs_matches, key=lambda x: -x[1])[:5]:
            region, off, sz = key
            print(f"  {region} +0x{off:04x} [{sz}B]: {matches}/{total} match CS")
            for fi, lev, cs, kills, val in entries[:10]:
                match = "✓" if val == cs else "✗"
                print(f"    frame={fi:3d} cs={cs:4d} val={val:6d} {match}")
    else:
        print("No offset matches CS >50%")

    # ── Find writes where value == kills ──
    print("\n" + "=" * 70)
    print("SEARCH: Write offsets where decoded value EQUALS kills")
    print("=" * 70)

    kill_matches = []
    for key, entries in write_history.items():
        if len(entries) < 5:
            continue
        matches = sum(1 for _, _, _, kills, val in entries if kills is not None and val == kills)
        total = len(entries)
        if matches > total * 0.5:
            kill_matches.append((key, matches, total, entries))

    if kill_matches:
        for key, matches, total, entries in sorted(kill_matches, key=lambda x: -x[1])[:5]:
            region, off, sz = key
            print(f"  {region} +0x{off:04x} [{sz}B]: {matches}/{total} match kills")
    else:
        print("No offset matches kills >50%")

    # ── Monotonic correlation check ──
    print("\n" + "=" * 70)
    print("SEARCH: Decoded values with high correlation to level/CS")
    print("=" * 70)

    import numpy as np
    for stat_name in ['level', 'cs']:
        stat_idx = 1 if stat_name == 'level' else 2
        best = []
        for key, entries in write_history.items():
            if len(entries) < 10:
                continue
            vals = [e[4] for e in entries]
            stats = [e[stat_idx] for e in entries]
            if len(set(vals)) < 3 or len(set(stats)) < 3:
                continue
            r = np.corrcoef(vals, stats)[0, 1]
            if abs(r) > 0.7:
                best.append((key, r, len(entries)))

        if best:
            print(f"\n  High correlation with {stat_name}:")
            for key, r, n in sorted(best, key=lambda x: -abs(x[1]))[:10]:
                region, off, sz = key
                print(f"    {region} +0x{off:04x} [{sz}B]: r={r:.4f} (n={n})")
                entries = write_history[key]
                for fi, lev, cs, kills, val in entries[:8]:
                    s = lev if stat_name == 'level' else cs
                    print(f"      frame={fi:3d} {stat_name}={s:4d} val={val}")
        else:
            print(f"\n  No high correlation with {stat_name}")


if __name__ == '__main__':
    main()
