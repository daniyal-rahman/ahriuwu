#!/usr/bin/env python3
"""
Run PID 842 deserializer WITHOUT SKIP patch to get correctly decoded values.
Then search for level/CS/kills in the writes.
Also try PID 183, 318, 866 deserializers.
"""
import struct
import json
import os
import zstandard as zstd
from collections import defaultdict
import numpy as np

from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_PROT_ALL, UC_PROT_READ, UC_PROT_WRITE
from unicorn import UC_HOOK_MEM_READ_UNMAPPED, UC_HOOK_MEM_WRITE_UNMAPPED, UC_HOOK_MEM_FETCH_UNMAPPED
from unicorn import UC_HOOK_CODE, UC_HOOK_MEM_WRITE
from unicorn.x86_const import *

BASE_ADDR    = 0x140000000
STACK_BASE   = 0x7FFFFFFF0000
STACK_SIZE   = 0x4000
HEAP_BASE    = 0x7FFFFFFFA000
ALLOC_BASE   = 0x7FFFFFF80000
ALLOC_SIZE   = 0x40000
SCRATCH_BASE = 0x7FFFFFF00000
STOP_ADDR    = 0xDEAD0000
BLOCK_MARKERS = {0x91, 0xf1, 0xb1, 0x31, 0x11}

PIDS = {
    842: {'deser': 0x101e8b0, 'vtable': 0x19fde98, 'ctor': 0xe03ef0, 'name': 'GameStats'},
    183: {'deser': 0x101d170, 'vtable': 0x19fda10, 'ctor': 0xe03b10, 'name': 'StatsHiFreq'},
    318: {'deser': 0xf53aa0, 'vtable': 0x19faff8, 'ctor': 0xe05430, 'name': 'LevelChange'},
    866: {'deser': 0xf8b910, 'vtable': 0x19fb5d0, 'ctor': 0xe1e4e0, 'name': 'StatsMulti'},
}

align_down = lambda a, s=0x1000: a & ~(s-1)
align_up = lambda s, p=0x1000: (s+p-1) & ~(p-1)

ORACLE_PATH = '/tmp/oracle_data.json'
ROFL_PATH = 'data/replays/NA1-5528069928.rofl'
GAREN_ENTITY = 0x400000b2
PLAYER_ID_START = 0x400000ae


def load_oracle():
    with open(ORACLE_PATH) as f:
        data = json.load(f)
    snaps = []
    for s in data['snapshots']:
        if 'Garen' in s['players']:
            g = s['players']['Garen']
            snaps.append({'time': s['actual_time'], 'level': g['level'],
                         'cs': g['scores']['creepScore'], 'kills': g['scores']['kills']})
    return snaps, data['game_length']


def get_stat(oracle, t, s):
    v = None
    for snap in oracle:
        if snap['time'] <= t + 1.0: v = snap[s]
        else: break
    return v


def load_frames():
    with open(ROFL_PATH, 'rb') as f:
        data = f.read()
    dctx = zstd.ZstdDecompressor()
    frames = []
    pos = 0
    while pos < len(data):
        idx = data.find(b'\x28\xb5\x2f\xfd', pos)
        if idx < 0: break
        try:
            dec = dctx.decompress(data[idx:idx+0x100000], max_output_size=0x400000)
            frames.append(dec)
        except: pass
        pos = idx + 1
    return frames


def extract_pid_blocks(frames, pid, entity=None):
    pid_bytes = struct.pack('<H', pid)
    all_blocks = []
    for fi, fd in enumerate(frames):
        pos = 3
        while pos < len(fd) - 6:
            idx = fd.find(pid_bytes, pos)
            if idx < 0: break
            bs = idx - 3
            if bs >= 0 and fd[bs] in BLOCK_MARKERS:
                size = fd[bs + 2]
                param = struct.unpack_from('<I', fd, bs + 5)[0]
                end = bs + 9 + size
                if end <= len(fd) and size > 0:
                    vc = (end >= len(fd) - 9 or (end < len(fd) and fd[end] in BLOCK_MARKERS))
                    if vc and (entity is None or param == entity):
                        all_blocks.append({'frame': fi, 'payload': bytes(fd[bs+9:end]),
                                          'channel': fd[bs+1], 'param': param})
            pos = idx + 1
    return all_blocks


def create_emulator(pid_info, patch_skip=False):
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

    if patch_skip:
        mu.mem_write(BASE_ADDR + 0x118b120, b'\x48\xC7\xC0\x01\x00\x00\x00\xC3')
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

    state = {'ac': 0, 'ar': []}
    def malloc_hook(mu, addr, size, ud):
        if addr == BASE_ADDR + 0x10fa120:
            sz = mu.reg_read(UC_X86_REG_RCX)
            aligned = max((sz + 15) & ~15, 0x200)
            ptr = ALLOC_BASE + state['ac']
            state['ar'].append((state['ac'], sz))
            state['ac'] += aligned
            mu.reg_write(UC_X86_REG_RAX, ptr)
            rsp = mu.reg_read(UC_X86_REG_RSP)
            ret = struct.unpack('<Q', mu.mem_read(rsp, 8))[0]
            mu.reg_write(UC_X86_REG_RSP, rsp + 8)
            mu.reg_write(UC_X86_REG_RIP, ret)
    mu.hook_add(UC_HOOK_CODE, malloc_hook,
                begin=BASE_ADDR + 0x10fa120, end=BASE_ADDR + 0x10fa121)

    # Constructor
    mu.mem_write(HEAP_BASE, b'\x00' * 0x400)
    mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
    mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
    rsp = mu.reg_read(UC_X86_REG_RSP) - 8
    mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
    mu.reg_write(UC_X86_REG_RSP, rsp)
    try:
        mu.emu_start(BASE_ADDR + pid_info['ctor'], STOP_ADDR, timeout=5000000)
    except:
        pass
    ctor_data = bytes(mu.mem_read(HEAP_BASE, 0x400))
    return mu, ctor_data, state


def decode_block(mu, ctor_data, state, pid_info, payload):
    state['ac'] = 0
    state['ar'] = []
    mu.mem_write(ALLOC_BASE, b'\x00' * min(0x8000, ALLOC_SIZE))
    mu.mem_write(HEAP_BASE, b'\x00' * 0x400)
    mu.mem_write(HEAP_BASE, ctor_data)
    mu.mem_write(HEAP_BASE, struct.pack('<Q', BASE_ADDR + pid_info['vtable']))

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
    def track(mu, access, address, size, value, ud):
        val = value & ((1 << (size * 8)) - 1)
        off_a = address - ALLOC_BASE
        off_h = address - HEAP_BASE
        if 0 <= off_a < ALLOC_SIZE:
            writes.append(('A', off_a, size, val))
        if 0 <= off_h < 0x400:
            writes.append(('H', off_h, size, val))
        return True

    h1 = mu.hook_add(UC_HOOK_MEM_WRITE, track, begin=ALLOC_BASE, end=ALLOC_BASE+ALLOC_SIZE)
    h2 = mu.hook_add(UC_HOOK_MEM_WRITE, track, begin=HEAP_BASE, end=HEAP_BASE+0x400)

    inst_count = [0]
    def limiter(mu, addr, sz, ud):
        inst_count[0] += 1
        if inst_count[0] > 1000000:
            mu.emu_stop()
    h3 = mu.hook_add(UC_HOOK_CODE, limiter)

    try:
        mu.emu_start(BASE_ADDR + pid_info['deser'], STOP_ADDR, timeout=30000000)
        ok = True
    except:
        ok = False

    mu.hook_del(h1); mu.hook_del(h2); mu.hook_del(h3)
    return writes, ok, inst_count[0]


def main():
    oracle, game_length = load_oracle()
    frames = load_frames()
    n_frames = len(frames)
    frame_dur = game_length / max(n_frames - 1, 1)
    print(f"Frames: {n_frames}, game: {game_length:.1f}s\n")

    for pid, pid_info in PIDS.items():
        print("=" * 70)
        print(f"PID {pid} ({pid_info['name']})")
        print("=" * 70)

        # Extract blocks
        blocks_b2 = extract_pid_blocks(frames, pid, GAREN_ENTITY)
        # Also try with all hero entities
        hero_blocks = extract_pid_blocks(frames, pid)
        hero_b2 = [b for b in hero_blocks if b['param'] == GAREN_ENTITY]

        print(f"  Entity b2 blocks: {len(blocks_b2)}")
        print(f"  All blocks: {len(hero_blocks)}")
        entity_counts = defaultdict(int)
        for b in hero_blocks:
            entity_counts[b['param']] += 1
        top_entities = sorted(entity_counts.items(), key=lambda x: -x[1])[:5]
        print(f"  Top entities: {[(f'0x{e:08x}', c) for e, c in top_entities]}")

        if len(blocks_b2) < 3:
            print("  Too few blocks, skipping\n")
            continue

        # Create emulator WITHOUT skip patch
        mu, ctor_data, state = create_emulator(pid_info, patch_skip=False)

        # Take one block per frame
        frame_blocks = {}
        for b in blocks_b2:
            if b['frame'] not in frame_blocks:
                frame_blocks[b['frame']] = b

        print(f"  Unique frames: {len(frame_blocks)}")

        # Decode and search
        write_history = defaultdict(list)
        errors = 0
        for fi, blk in sorted(frame_blocks.items()):
            ft = fi * frame_dur
            level = get_stat(oracle, ft, 'level')
            cs = get_stat(oracle, ft, 'cs')
            kills = get_stat(oracle, ft, 'kills')
            if level is None:
                continue

            writes, ok, inst = decode_block(mu, ctor_data, state, pid_info, blk['payload'])
            if not ok:
                errors += 1
                continue

            # Check for level value in writes
            for region, off, sz, val in writes:
                write_history[(region, off, sz)].append((fi, level, cs, kills, val))

        print(f"  Decoded: {len(frame_blocks) - errors}/{len(frame_blocks)} ({errors} errors)")
        print(f"  Unique write keys: {len(write_history)}")

        # Search for level
        print(f"\n  --- Level matches ---")
        found_any = False
        for key, entries in write_history.items():
            if len(entries) < 3:
                continue
            matches = sum(1 for _, lev, _, _, val in entries if val == lev)
            if matches >= len(entries) * 0.5 and matches >= 3:
                region, off, sz = key
                distinct_levels = len(set(e[1] for e in entries if e[4] == e[1]))
                print(f"  {region}+0x{off:04x} [{sz}B]: {matches}/{len(entries)} match level "
                      f"(distinct_matched={distinct_levels})")
                for _, lev, _, _, val in entries[:8]:
                    m = "✓" if val == lev else "✗"
                    print(f"    level={lev:2d} val={val:6d} {m}")
                found_any = True

        if not found_any:
            print("  No level matches found")

        # Search for CS
        print(f"\n  --- CS matches ---")
        found_cs = False
        for key, entries in write_history.items():
            if len(entries) < 3:
                continue
            matches = sum(1 for _, _, cs, _, val in entries if cs is not None and val == cs)
            if matches >= len(entries) * 0.3 and matches >= 3:
                region, off, sz = key
                print(f"  {region}+0x{off:04x} [{sz}B]: {matches}/{len(entries)} match CS")
                for _, _, cs, _, val in entries[:5]:
                    m = "✓" if val == cs else "✗"
                    print(f"    cs={cs:4d} val={val:6d} {m}")
                found_cs = True
        if not found_cs:
            print("  No CS matches found")

        # Correlation analysis
        print(f"\n  --- Correlation ---")
        for stat_name, stat_idx in [('level', 1), ('cs', 2), ('kills', 3)]:
            best_r = 0
            best_key = None
            for key, entries in write_history.items():
                if len(entries) < 8:
                    continue
                vals = [e[4] for e in entries]
                stats = [e[stat_idx] for e in entries]
                if len(set(vals)) < 3 or len(set(stats)) < 3:
                    continue
                try:
                    r = np.corrcoef(vals, stats)[0, 1]
                    if abs(r) > abs(best_r):
                        best_r = r
                        best_key = key
                except:
                    pass
            if best_key:
                region, off, sz = best_key
                entries = write_history[best_key]
                print(f"  Best {stat_name} corr: {region}+0x{off:04x} [{sz}B] r={best_r:.4f} (n={len(entries)})")
                for _, lev, cs, kills, val in entries[:5]:
                    s = lev if stat_name == 'level' else (cs if stat_name == 'cs' else kills)
                    print(f"    {stat_name}={s:4d} val={val}")

        print()


if __name__ == '__main__':
    main()
