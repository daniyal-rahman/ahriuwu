#!/usr/bin/env python3
"""
Quick test: run PID 842 (stat) deserializer through Unicorn emulator.
Uses the existing framework from decode_replay_movement.py but with PID 842 addresses.
"""
import struct
import json
import zstandard as zstd
from collections import defaultdict

# Import the existing decoder framework
import sys
sys.path.insert(0, 'scripts')
from decode_replay_movement import (
    MovementDecoder, parse_rofl_blocks, BLOCK_MARKERS,
    BASE_ADDR, HEAP_BASE, ALLOC_BASE, ALLOC_SIZE, STACK_BASE, STACK_SIZE,
    SCRATCH_BASE, STOP_ADDR
)
from unicorn import UC_HOOK_MEM_WRITE
from unicorn.x86_const import *

ROFL_PATH = 'data/replays/NA1-5528069928.rofl'
ORACLE_PATH = '/tmp/oracle_data.json'
GAREN_ENTITY = 0x400000b2

# PID 842 addresses
PID842_DESER_RVA = 0x101e8b0
PID842_VTABLE = 0x19fde98
PID842_CTOR = 0xe03ef0

def extract_pid842_blocks(rofl_path, entity_filter=None):
    """Extract PID 842 blocks from ROFL."""
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

    pid_bytes = struct.pack('<H', 842)
    all_blocks = []
    for fi, fd in enumerate(frames):
        pos = 3
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
                    valid_chain = (end >= len(fd) - 9 or
                                  (end < len(fd) and fd[end] in BLOCK_MARKERS))
                    if valid_chain:
                        if entity_filter is None or param == entity_filter:
                            all_blocks.append({
                                'frame': fi,
                                'param': param,
                                'payload': bytes(fd[block_start + 9:end]),
                                'channel': fd[block_start + 1],
                                'size': size,
                            })
            pos = idx + 1
    return all_blocks, len(frames)


def main():
    print("Loading oracle data...")
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
            })

    print("Extracting PID 842 blocks for entity b2...")
    blocks, n_frames = extract_pid842_blocks(ROFL_PATH, GAREN_ENTITY)
    frame_duration = game_length / max(n_frames - 1, 1)
    print(f"Found {len(blocks)} blocks across {n_frames} frames")

    # Get level for each frame
    def get_level(t):
        level = None
        for s in garen_snaps:
            if s['time'] <= t + 1.0:
                level = s['level']
            else:
                break
        return level

    print("\nInitializing Unicorn emulator for PID 842...")
    # Create decoder with PID 842 addresses
    import os
    pe_dir = '/tmp/pe_dump_16.7'
    text_raw = open(os.path.join(pe_dir, 'text.bin'), 'rb').read()
    rdata_raw = open(os.path.join(pe_dir, 'rdata.bin'), 'rb').read()
    data_raw = open(os.path.join(pe_dir, 'data.bin'), 'rb').read()

    from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_PROT_ALL, UC_PROT_READ, UC_PROT_WRITE
    from unicorn import UC_HOOK_MEM_READ_UNMAPPED, UC_HOOK_MEM_WRITE_UNMAPPED, UC_HOOK_MEM_FETCH_UNMAPPED, UC_HOOK_CODE

    def align_down(a, s=0x1000): return a & ~(s-1)
    def align_up(s, p=0x1000): return (s+p-1) & ~(p-1)

    addrs = {
        'TEXT_RVA': 0x1000,
        'RDATA_RVA': 0x18fd000,
        'DATA_RVA': 0x1d21000,
        'SKIP_RVA': 0x118b120,
        'MALLOC_RVA': 0x10fa120,
        'FREE_RVA': 0x10fa150,
        'DESER_RVA': PID842_DESER_RVA,
        'MAIN_VTABLE': PID842_VTABLE,
        'CONSTRUCTOR': PID842_CTOR,
        'STRUCT_SIZE': 0x100,
    }

    mu = Uc(UC_ARCH_X86, UC_MODE_64)
    mu.mem_map(STACK_BASE, STACK_SIZE, UC_PROT_READ | UC_PROT_WRITE)
    mu.mem_map(HEAP_BASE, align_up(0x20000), UC_PROT_READ | UC_PROT_WRITE)
    mu.mem_map(SCRATCH_BASE, 0x10000, UC_PROT_READ | UC_PROT_WRITE)
    mu.mem_map(ALLOC_BASE, ALLOC_SIZE, UC_PROT_ALL)

    pe_start = align_down(BASE_ADDR + addrs['TEXT_RVA'])
    pe_end = align_up(BASE_ADDR + addrs['DATA_RVA'] + len(data_raw) + 0x1000)
    mu.mem_map(pe_start, pe_end - pe_start, UC_PROT_ALL)
    mu.mem_write(BASE_ADDR + addrs['TEXT_RVA'], text_raw)
    mu.mem_write(BASE_ADDR + addrs['RDATA_RVA'], rdata_raw)
    mu.mem_write(BASE_ADDR + addrs['DATA_RVA'], data_raw)

    # Patch SKIP → mov rax, 1; ret (skip validation)
    mu.mem_write(BASE_ADDR + addrs['SKIP_RVA'], b'\x48\xC7\xC0\x01\x00\x00\x00\xC3')
    # Patch free → ret
    mu.mem_write(BASE_ADDR + addrs['FREE_RVA'], b'\xC3')

    # Stop address
    mu.mem_map(align_down(STOP_ADDR), 0x1000, UC_PROT_ALL)
    mu.mem_write(STOP_ADDR, b'\xF4')

    # Unmapped memory handler
    def on_unmap(mu, access, addr, sz, val, ud):
        try:
            mu.mem_map(align_down(addr), 0x1000, UC_PROT_ALL)
            return True
        except:
            return False
    mu.hook_add(UC_HOOK_MEM_READ_UNMAPPED | UC_HOOK_MEM_WRITE_UNMAPPED |
                UC_HOOK_MEM_FETCH_UNMAPPED, on_unmap)

    # Malloc hook
    alloc_cursor = [0]
    alloc_regions = []

    def malloc_hook(mu, address, size, user_data):
        if address == BASE_ADDR + addrs['MALLOC_RVA']:
            sz = mu.reg_read(UC_X86_REG_RCX)
            aligned = max((sz + 15) & ~15, 0x200)
            ptr = ALLOC_BASE + alloc_cursor[0]
            alloc_regions.append((alloc_cursor[0], sz))
            alloc_cursor[0] += aligned
            mu.reg_write(UC_X86_REG_RAX, ptr)
            rsp = mu.reg_read(UC_X86_REG_RSP)
            ret = struct.unpack('<Q', mu.mem_read(rsp, 8))[0]
            mu.reg_write(UC_X86_REG_RSP, rsp + 8)
            mu.reg_write(UC_X86_REG_RIP, ret)
    mu.hook_add(UC_HOOK_CODE, malloc_hook,
                begin=BASE_ADDR + addrs['MALLOC_RVA'],
                end=BASE_ADDR + addrs['MALLOC_RVA'] + 1)

    # Run constructor
    print("Running PID 842 constructor...")
    mu.mem_write(HEAP_BASE, b'\x00' * 0x400)
    mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
    mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
    rsp = mu.reg_read(UC_X86_REG_RSP) - 8
    mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
    mu.reg_write(UC_X86_REG_RSP, rsp)
    try:
        mu.emu_start(BASE_ADDR + addrs['CONSTRUCTOR'], STOP_ADDR, timeout=5000000)
        print("Constructor completed successfully")
    except Exception as e:
        print(f"Constructor error: {e}")

    ctor_data = bytes(mu.mem_read(HEAP_BASE, 0x100))

    # Set vtable
    mu.mem_write(HEAP_BASE, struct.pack('<Q', BASE_ADDR + addrs['MAIN_VTABLE']))

    # Test with first few blocks
    print("\n" + "=" * 70)
    print("Testing PID 842 deserializer on entity b2 blocks")
    print("=" * 70)

    # Pick blocks at different levels
    seen_levels = set()
    test_blocks = []
    for b in sorted(blocks, key=lambda x: x['frame']):
        ft = b['frame'] * frame_duration
        level = get_level(ft)
        if level and level not in seen_levels and len(test_blocks) < 15:
            seen_levels.add(level)
            test_blocks.append((b, level))

    for blk, level in test_blocks:
        payload = blk['payload']
        ft = blk['frame'] * frame_duration

        # Reset state
        alloc_cursor[0] = 0
        alloc_regions.clear()
        mu.mem_write(ALLOC_BASE, b'\x00' * min(0x8000, ALLOC_SIZE))
        mu.mem_write(HEAP_BASE, b'\x00' * 0x100)
        mu.mem_write(HEAP_BASE, ctor_data)
        mu.mem_write(HEAP_BASE, struct.pack('<Q', BASE_ADDR + addrs['MAIN_VTABLE']))

        # Set up payload
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

        # Track writes
        write_log = []
        def track_writes(mu, access, address, size, value, user_data):
            val = value & ((1 << (size * 8)) - 1)
            off = address - ALLOC_BASE
            if 0 <= off < ALLOC_SIZE:
                write_log.append(('ALLOC', off, size, val))
            soff = address - HEAP_BASE
            if 0 <= soff < 0x400:
                write_log.append(('HEAP', soff, size, val))
            return True

        h1 = mu.hook_add(UC_HOOK_MEM_WRITE, track_writes,
                         begin=ALLOC_BASE, end=ALLOC_BASE + ALLOC_SIZE)
        h2 = mu.hook_add(UC_HOOK_MEM_WRITE, track_writes,
                         begin=HEAP_BASE, end=HEAP_BASE + 0x400)

        try:
            mu.emu_start(BASE_ADDR + addrs['DESER_RVA'], STOP_ADDR, timeout=10000000)
            status = "OK"
        except Exception as e:
            status = f"ERROR: {e}"

        mu.hook_del(h1)
        mu.hook_del(h2)

        print(f"\nLevel={level:2d} frame={blk['frame']:3d} t={ft:.0f}s sz={len(payload)} "
              f"ch=0x{blk['channel']:02x} status={status}")
        print(f"  Writes: {len(write_log)}, Allocs: {len(alloc_regions)}")

        # Show interesting writes (look for level-sized values)
        if write_log:
            # Show all 4-byte writes that could be stats
            for region, off, sz, val in write_log:
                if sz >= 4:
                    fval = struct.unpack('<f', struct.pack('<I', val & 0xFFFFFFFF))[0]
                    if 0 < val < 1000 or (0.5 < fval < 100 and fval == int(fval)):
                        print(f"    {region} +0x{off:04x} [{sz}B] = {val} "
                              f"(f32={fval:.1f})")
                elif sz == 1 and 1 <= val <= 20:
                    print(f"    {region} +0x{off:04x} [1B] = {val}")
                elif sz == 2 and 1 <= val <= 300:
                    print(f"    {region} +0x{off:04x} [2B] = {val}")

        if len(write_log) == 0:
            # Show raw alloc buffer
            if alloc_regions:
                off, sz = alloc_regions[0]
                buf = bytes(mu.mem_read(ALLOC_BASE + off, min(sz, 64)))
                print(f"  Alloc[0] raw: {buf.hex()}")


if __name__ == '__main__':
    main()
