#!/usr/bin/env python3
"""
Unicorn Emulator V2 — Run any PID's deserializer with SKIP NOT PATCHED.

Key differences from V1 (decode_replay_movement.py):
  - Does NOT patch SKIP (0x118b0c0 or 0x118b160) — these are real varint decoders
  - Supports any PID via pid_full_map.json
  - Loops over sub-messages within each payload
  - Captures all memory writes to struct and alloc buffer
  - Validates against oracle data
"""
import struct
import json
import sys
import os
import zstandard as zstd
from collections import defaultdict
from unicorn import (Uc, UC_ARCH_X86, UC_MODE_64, UC_PROT_ALL,
                     UC_PROT_READ, UC_PROT_WRITE,
                     UC_HOOK_MEM_WRITE, UC_HOOK_MEM_READ_UNMAPPED,
                     UC_HOOK_MEM_WRITE_UNMAPPED, UC_HOOK_MEM_FETCH_UNMAPPED,
                     UC_HOOK_CODE)
from unicorn.x86_const import *

# ── Memory layout ──────────────────────────────────────────────
BASE_ADDR    = 0x140000000
STACK_BASE   = 0x7FFFFFFF0000
STACK_SIZE   = 0x4000
HEAP_BASE    = 0x7FFFFFFFA000   # struct lives here
HEAP_SIZE    = 0x20000
SCRATCH_BASE = 0x7FFFFFF00000   # payload + pointer
SCRATCH_SIZE = 0x10000
ALLOC_BASE   = 0x200000000000   # bump-allocated memory (far from stack/heap)
ALLOC_SIZE   = 0x100000          # 1MB
STOP_ADDR    = 0xDEAD0000

# ── 16.7 addresses ────────────────────────────────────────────
TEXT_RVA    = 0x1000
RDATA_RVA   = 0x18fd000
DATA_RVA    = 0x1d21000
SKIP_RVA_1  = 0x118b0c0   # short varint reader (1-2 byte)
SKIP_RVA_2  = 0x118b160   # full SKIP varint with S-box cipher
MALLOC_RVA  = 0x10fa120
FREE_RVA    = 0x10fa150
FREE_TARGET = 0x10fe1b0   # FREE jmps here

# ── Block parsing (from V1) ───────────────────────────────────
BLOCK_MARKERS = {0x91, 0xf1, 0xb1, 0x31, 0x11}

def align_down(a, s=0x1000): return a & ~(s-1)
def align_up(s, p=0x1000): return (s + p - 1) & ~(p - 1)


class EmulatorV2:
    def __init__(self, pe_dir='/tmp/pe_dump_16.7', pid_map_path='/tmp/pid_full_map.json'):
        self.text_raw = open(os.path.join(pe_dir, 'text.bin'), 'rb').read()
        self.rdata_raw = open(os.path.join(pe_dir, 'rdata.bin'), 'rb').read()
        self.data_raw = open(os.path.join(pe_dir, 'data.bin'), 'rb').read()
        self.pid_map = json.load(open(pid_map_path))
        self._setup_emu()

    def _setup_emu(self):
        mu = Uc(UC_ARCH_X86, UC_MODE_64)

        # Map memory regions
        mu.mem_map(STACK_BASE, STACK_SIZE, UC_PROT_READ | UC_PROT_WRITE)
        mu.mem_map(HEAP_BASE, align_up(HEAP_SIZE), UC_PROT_READ | UC_PROT_WRITE)
        mu.mem_map(SCRATCH_BASE, SCRATCH_SIZE, UC_PROT_ALL)
        mu.mem_map(ALLOC_BASE, ALLOC_SIZE, UC_PROT_ALL)

        # Map PE sections
        pe_start = align_down(BASE_ADDR + TEXT_RVA)
        pe_end = align_up(BASE_ADDR + DATA_RVA + len(self.data_raw) + 0x1000)
        mu.mem_map(pe_start, pe_end - pe_start, UC_PROT_ALL)
        mu.mem_write(BASE_ADDR + TEXT_RVA, self.text_raw)
        mu.mem_write(BASE_ADDR + RDATA_RVA, self.rdata_raw)
        mu.mem_write(BASE_ADDR + DATA_RVA, self.data_raw)

        # ── KEY FIX: Do NOT patch SKIP! ──
        # Only patch FREE → ret (can't actually free bump-allocated memory)
        mu.mem_write(BASE_ADDR + FREE_RVA, b'\xC3')
        # Also patch the jmp target of FREE in case something calls it directly
        mu.mem_write(BASE_ADDR + FREE_TARGET, b'\xC3')

        # Stop address
        mu.mem_map(align_down(STOP_ADDR), 0x1000, UC_PROT_ALL)
        mu.mem_write(STOP_ADDR, b'\xF4')  # HLT

        # Unmapped memory handler — auto-map on access
        def on_unmap(mu, access, addr, sz, val, ud):
            page = align_down(addr)
            try:
                mu.mem_map(page, 0x1000, UC_PROT_ALL)
                return True
            except:
                return False
        mu.hook_add(UC_HOOK_MEM_READ_UNMAPPED | UC_HOOK_MEM_WRITE_UNMAPPED |
                    UC_HOOK_MEM_FETCH_UNMAPPED, on_unmap)

        # Malloc hook — bump allocator
        self._alloc_cursor = 0
        self._alloc_regions = []

        def malloc_hook(mu, address, size, user_data):
            if address == BASE_ADDR + MALLOC_RVA:
                sz = mu.reg_read(UC_X86_REG_RCX)
                aligned = max((sz + 15) & ~15, 0x200)
                if self._alloc_cursor + aligned > ALLOC_SIZE:
                    # Out of alloc space — return NULL
                    mu.reg_write(UC_X86_REG_RAX, 0)
                else:
                    ptr = ALLOC_BASE + self._alloc_cursor
                    self._alloc_regions.append((self._alloc_cursor, sz))
                    self._alloc_cursor += aligned
                    mu.reg_write(UC_X86_REG_RAX, ptr)
                # Simulate ret
                rsp = mu.reg_read(UC_X86_REG_RSP)
                ret = struct.unpack('<Q', mu.mem_read(rsp, 8))[0]
                mu.reg_write(UC_X86_REG_RSP, rsp + 8)
                mu.reg_write(UC_X86_REG_RIP, ret)

        mu.hook_add(UC_HOOK_CODE, malloc_hook,
                    begin=BASE_ADDR + MALLOC_RVA,
                    end=BASE_ADDR + MALLOC_RVA + 1)

        self.mu = mu
        self._write_log = []
        self._mapped_pages = set()

    def _track_writes(self, mu, access, address, size, value, user_data):
        val = value & ((1 << (size * 8)) - 1)
        # Track alloc buffer writes
        aoff = address - ALLOC_BASE
        if 0 <= aoff < ALLOC_SIZE:
            self._write_log.append(('ALLOC', aoff, size, val))
        # Track struct (heap) writes
        hoff = address - HEAP_BASE
        if 0 <= hoff < 0x1000:
            self._write_log.append(('HEAP', hoff, size, val))
        return True

    def _run_constructor(self, pid):
        """Run the constructor for a PID to initialize the struct."""
        pid_info = self.pid_map.get(str(pid))
        if not pid_info:
            return False
        ctor_rva = pid_info['ctor']    # already an RVA
        vtable_rva = pid_info['vtable']  # already an RVA

        # Clear struct
        self.mu.mem_write(HEAP_BASE, b'\x00' * 0x1000)

        # Run constructor: RCX = struct pointer
        self.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
        self.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
        rsp = self.mu.reg_read(UC_X86_REG_RSP) - 8
        self.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
        self.mu.reg_write(UC_X86_REG_RSP, rsp)

        try:
            self.mu.emu_start(BASE_ADDR + ctor_rva, STOP_ADDR, timeout=5_000_000)
        except Exception as e:
            # Constructor may fail on some external calls — that's OK
            pass

        # Make sure vtable is set
        self.mu.mem_write(HEAP_BASE, struct.pack('<Q', BASE_ADDR + vtable_rva))
        return True

    def decode_payload(self, pid, payload, verbose=False):
        """Decode a single payload for a given PID.

        Returns list of dicts, one per sub-message decoded.
        """
        pid_info = self.pid_map.get(str(pid))
        if not pid_info:
            return []

        deser_rva = pid_info['deser']  # RVA from pid_map
        deser_addr = BASE_ADDR + deser_rva
        if deser_rva < TEXT_RVA or deser_rva > TEXT_RVA + len(self.text_raw):
            if verbose:
                print(f"  PID {pid}: deser RVA 0x{deser_rva:x} out of text range")
            return []

        results = []
        pay_addr = SCRATCH_BASE + 0x100
        pp_addr = SCRATCH_BASE + 0x200  # pointer-to-pointer
        pay_end = pay_addr + len(payload)

        # Write payload
        self.mu.mem_write(pay_addr, payload + b'\x00' * 128)

        # Initialize current position to start of payload
        current_pos = pay_addr

        iteration = 0
        max_iterations = 10  # safety limit

        while current_pos < pay_end and iteration < max_iterations:
            iteration += 1

            # Reset tracking
            self._write_log = []
            self._alloc_cursor = 0
            self._alloc_regions = []
            self.mu.mem_write(ALLOC_BASE, b'\x00' * min(0x10000, ALLOC_SIZE))

            # Reset struct via constructor
            self._run_constructor(pid)

            # Save clean struct state
            clean_struct = bytes(self.mu.mem_read(HEAP_BASE, 0x100))

            # Write current payload pointer
            self.mu.mem_write(pp_addr, struct.pack('<Q', current_pos))

            # Set up registers for deserializer call:
            # RCX = struct, RDX = &payload_ptr, R8 = payload_end
            self.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
            self.mu.reg_write(UC_X86_REG_RDX, pp_addr)
            self.mu.reg_write(UC_X86_REG_R8, pay_end)
            self.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)

            # Push return address
            rsp = self.mu.reg_read(UC_X86_REG_RSP) - 8
            self.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
            self.mu.reg_write(UC_X86_REG_RSP, rsp)

            # Add write tracking hooks
            h1 = self.mu.hook_add(UC_HOOK_MEM_WRITE, self._track_writes,
                                  begin=ALLOC_BASE, end=ALLOC_BASE + ALLOC_SIZE)
            h2 = self.mu.hook_add(UC_HOOK_MEM_WRITE, self._track_writes,
                                  begin=HEAP_BASE, end=HEAP_BASE + 0x1000)

            # Run!
            success = False
            try:
                self.mu.emu_start(deser_addr, STOP_ADDR, timeout=10_000_000)
                al = self.mu.reg_read(UC_X86_REG_RAX) & 0xFF
                success = (al != 0)
            except Exception as e:
                if verbose:
                    print(f"  Iter {iteration}: emulation error: {e}")

            self.mu.hook_del(h1)
            self.mu.hook_del(h2)

            # Check how far payload pointer advanced
            new_pos = struct.unpack('<Q', self.mu.mem_read(pp_addr, 8))[0]
            bytes_consumed = new_pos - current_pos

            if verbose:
                remaining_before = pay_end - current_pos
                print(f"  Iter {iteration}: success={success}, consumed={bytes_consumed}/{remaining_before}, "
                      f"writes={len(self._write_log)}, allocs={len(self._alloc_regions)}")

            if bytes_consumed <= 0:
                # No progress — stop
                if verbose:
                    print(f"  No payload progress, stopping loop")
                break

            # Extract results from this iteration
            result = self._extract_result(clean_struct, verbose)
            result['_iteration'] = iteration
            result['_bytes_consumed'] = bytes_consumed
            result['_success'] = success
            results.append(result)

            # Advance for next iteration
            current_pos = new_pos

        return results

    def _extract_result(self, clean_struct, verbose=False):
        """Extract meaningful values from write log."""
        result = {}

        # Categorize writes
        heap_writes = defaultdict(list)  # struct offset → [(size, val)]
        alloc_writes = defaultdict(list)  # alloc offset → [(size, val)]

        for region, off, sz, val in self._write_log:
            if region == 'HEAP':
                heap_writes[off].append((sz, val))
            elif region == 'ALLOC':
                alloc_writes[off].append((sz, val))

        result['heap_write_count'] = sum(len(v) for v in heap_writes.values())
        result['alloc_write_count'] = sum(len(v) for v in alloc_writes.values())
        result['alloc_regions'] = len(self._alloc_regions)

        # Extract struct fields — look for interesting values
        struct_fields = {}
        for off, writes in sorted(heap_writes.items()):
            last_sz, last_val = writes[-1]
            # Check if this offset was changed from constructor state
            if off < len(clean_struct):
                orig_bytes = clean_struct[off:off + last_sz]
                if len(orig_bytes) == last_sz:
                    orig_val = int.from_bytes(orig_bytes, 'little')
                    if last_val == orig_val:
                        continue  # Same as constructor — skip
            struct_fields[off] = (last_sz, last_val)

        result['struct_fields'] = struct_fields

        # Reconstruct alloc buffer
        if self._alloc_regions:
            for idx, (aoff, asz) in enumerate(self._alloc_regions):
                buf = bytearray(max(asz, 64))
                for off, writes in alloc_writes.items():
                    rel = off - aoff
                    if 0 <= rel < len(buf):
                        for sz, val in writes:
                            for b in range(sz):
                                if rel + b < len(buf):
                                    buf[rel + b] = (val >> (b * 8)) & 0xFF
                result[f'alloc_{idx}_raw'] = buf[:asz].hex()
                result[f'alloc_{idx}_size'] = asz

                # Try to interpret as game values
                game_vals = self._scan_for_game_values(buf, asz)
                if game_vals:
                    result[f'alloc_{idx}_game_vals'] = game_vals

        # Also scan struct for game values
        struct_game_vals = {}
        for off, (sz, val) in struct_fields.items():
            if sz == 4:
                # As float
                fval = struct.unpack('<f', struct.pack('<I', val & 0xFFFFFFFF))[0]
                if 0 < abs(fval) < 20000 and not (fval != fval):  # not NaN
                    struct_game_vals[f'struct+0x{off:x}_f32'] = round(fval, 2)
                # As int
                if 1 <= val <= 20:
                    struct_game_vals[f'struct+0x{off:x}_level?'] = val
                elif 0 < val < 500:
                    struct_game_vals[f'struct+0x{off:x}_cs?'] = val
                elif 500 <= val <= 15000:
                    struct_game_vals[f'struct+0x{off:x}_coord?'] = val
            elif sz == 2:
                if 0 < val < 15000:
                    struct_game_vals[f'struct+0x{off:x}_u16'] = val

        if struct_game_vals:
            result['struct_game_vals'] = struct_game_vals

        return result

    def _scan_for_game_values(self, buf, size):
        """Scan a buffer for values in game coordinate or stat range."""
        vals = {}
        for off in range(0, min(size, 256) - 3, 4):
            # Float interpretation
            fval = struct.unpack_from('<f', buf, off)[0]
            if 100 < abs(fval) < 15500 and fval == fval:  # game coord range, not NaN
                vals[f'+0x{off:x}_f32'] = round(fval, 2)
            elif 0 < fval < 20.5 and fval == fval:  # level range
                int_fval = round(fval)
                if abs(fval - int_fval) < 0.01:
                    vals[f'+0x{off:x}_lvl?'] = int_fval

            # U32 interpretation
            uval = struct.unpack_from('<I', buf, off)[0]
            if 500 <= uval <= 15000:
                vals[f'+0x{off:x}_u32_coord?'] = uval
            elif 1 <= uval <= 20:
                vals[f'+0x{off:x}_u32_lvl?'] = uval

            # 14-bit packed coords (like V1 found for PID 487)
            if uval != 0 and uval != 0xFFFFFFFF:
                x14 = uval & 0x3FFF
                y14 = (uval >> 14) & 0x3FFF
                MAP_SCALE = 14914.0 / 16384.0
                xc = x14 * MAP_SCALE
                yc = y14 * MAP_SCALE
                if 100 < xc < 15000 and 100 < yc < 15000:
                    vals[f'+0x{off:x}_14bit'] = (round(xc, 1), round(yc, 1))

        return vals if vals else None


def parse_rofl_blocks(rofl_path, pid_filter=None, param_filter=None):
    """Parse blocks from ROFL2 replay. Returns list of block dicts."""
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

    blocks = []
    for fd in frames:
        blocks.extend(_scan_blocks(fd, pid_filter, param_filter))
    return blocks


def _scan_blocks(fd, pid_filter=None, param_filter=None):
    """Scan decompressed frame for blocks."""
    blocks = []
    if pid_filter is not None:
        pid_bytes = struct.pack('<H', pid_filter)
        pos = 3
        while pos < len(fd) - 5:
            idx = fd.find(pid_bytes, pos)
            if idx < 0:
                break
            bs = idx - 3
            if bs >= 0 and fd[bs] in BLOCK_MARKERS:
                size = fd[bs + 2]
                param = struct.unpack_from('<I', fd, bs + 5)[0]
                end = bs + 9 + size
                if end <= len(fd):
                    if param_filter is None or param in param_filter:
                        valid = (end >= len(fd) - 9 or
                                 (end < len(fd) and fd[end] in BLOCK_MARKERS))
                        if valid:
                            blocks.append({
                                'pid': pid_filter, 'param': param,
                                'payload': fd[bs + 9:end],
                                'marker': fd[bs], 'channel': fd[bs + 1],
                                'size': size,
                            })
            pos = idx + 1
    else:
        pos = 0
        while pos + 9 <= len(fd):
            if fd[pos] in BLOCK_MARKERS:
                size = fd[pos + 2]
                pid = struct.unpack_from('<H', fd, pos + 3)[0]
                param = struct.unpack_from('<I', fd, pos + 5)[0]
                end = pos + 9 + size
                if end <= len(fd):
                    valid = (end >= len(fd) - 9 or
                             (end < len(fd) and fd[end] in BLOCK_MARKERS))
                    if valid:
                        if param_filter is None or param in param_filter:
                            blocks.append({
                                'pid': pid, 'param': param,
                                'payload': fd[pos + 9:end], 'marker': fd[pos],
                                'channel': fd[pos + 1], 'size': size,
                            })
                        pos = end
                        continue
            pos += 1
    return blocks


def test_pid(emu, rofl_path, pid, entity=0x400000b2, max_blocks=10, verbose=True):
    """Test decoding blocks for a specific PID and entity."""
    print(f"\n{'='*70}")
    print(f"PID {pid} — entity 0x{entity:08x}")
    print(f"{'='*70}")

    pid_info = emu.pid_map.get(str(pid))
    if not pid_info:
        print(f"  PID {pid} not in map!")
        return []

    print(f"  deser=0x{pid_info['deser']:x}, ctor=0x{pid_info['ctor']:x}, vtable=0x{pid_info['vtable']:x}")

    blocks = parse_rofl_blocks(rofl_path, pid_filter=pid, param_filter={entity})
    print(f"  Found {len(blocks)} blocks for entity")

    if not blocks:
        return []

    # Show payload size distribution
    sizes = [b['size'] for b in blocks]
    print(f"  Payload sizes: min={min(sizes)}, max={max(sizes)}, "
          f"unique={sorted(set(sizes))[:10]}")

    all_results = []
    for i, block in enumerate(blocks[:max_blocks]):
        payload = block['payload']
        print(f"\n  Block {i}: size={len(payload)}, payload={payload[:32].hex()}")

        results = emu.decode_payload(pid, payload, verbose=verbose)
        if not results:
            print(f"    → No results (decode failed)")
            continue

        for r in results:
            print(f"    → iter={r['_iteration']}, consumed={r['_bytes_consumed']}, "
                  f"success={r['_success']}, heap_writes={r['heap_write_count']}, "
                  f"alloc_writes={r['alloc_write_count']}, alloc_regions={r['alloc_regions']}")

            if r.get('struct_game_vals'):
                print(f"    Struct game vals: {r['struct_game_vals']}")
            for key in sorted(r.keys()):
                if key.startswith('alloc_') and key.endswith('_game_vals'):
                    print(f"    {key}: {r[key]}")
                elif key.startswith('alloc_') and key.endswith('_raw'):
                    print(f"    {key}: {r[key][:80]}...")

            all_results.append(r)

    return all_results


def validate_against_oracle(oracle_path, results_by_time):
    """Compare decoded values against oracle data."""
    oracle = json.load(open(oracle_path))
    snapshots = oracle['snapshots']

    print(f"\n{'='*70}")
    print("Oracle validation")
    print(f"{'='*70}")

    garen_snapshots = []
    for s in snapshots:
        g = s['players'].get('Garen', {})
        garen_snapshots.append({
            'time': s['actual_time'],
            'level': g.get('level'),
            'cs': g.get('scores', {}).get('creepScore'),
            'kills': g.get('scores', {}).get('kills'),
            'deaths': g.get('scores', {}).get('deaths'),
        })

    # Print oracle data at key timestamps
    for gs in garen_snapshots[::10]:
        print(f"  t={gs['time']:.0f}s: level={gs['level']}, CS={gs['cs']}, "
              f"K/D={gs['kills']}/{gs['deaths']}")

    return garen_snapshots


def deep_test_pid(emu, rofl_path, pid, entity=0x400000b2, max_blocks=5):
    """Detailed diagnostic test for a single PID — dumps ALL struct writes."""
    print(f"\n{'='*70}")
    print(f"DEEP TEST: PID {pid} — entity 0x{entity:08x}")
    print(f"{'='*70}")

    pid_info = emu.pid_map.get(str(pid))
    if not pid_info:
        print(f"  PID {pid} not in map!")
        return

    blocks = parse_rofl_blocks(rofl_path, pid_filter=pid, param_filter={entity})
    print(f"  Found {len(blocks)} blocks for entity")
    if not blocks:
        return

    # Test ONE block in full detail — no looping, just one call
    for bi, block in enumerate(blocks[:max_blocks]):
        payload = block['payload']
        print(f"\n  Block {bi}: size={len(payload)}, hex={payload.hex()}")

        # Reset everything
        emu._write_log = []
        emu._alloc_cursor = 0
        emu._alloc_regions = []
        emu.mu.mem_write(ALLOC_BASE, b'\x00' * min(0x10000, ALLOC_SIZE))
        emu._run_constructor(pid)

        # Save clean struct
        clean_struct = bytes(emu.mu.mem_read(HEAP_BASE, 0x100))

        pay_addr = SCRATCH_BASE + 0x100
        pp_addr = SCRATCH_BASE + 0x200
        pay_end = pay_addr + len(payload)

        emu.mu.mem_write(pay_addr, payload + b'\x00' * 128)
        emu.mu.mem_write(pp_addr, struct.pack('<Q', pay_addr))

        # Set up registers
        emu.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
        emu.mu.reg_write(UC_X86_REG_RDX, pp_addr)
        emu.mu.reg_write(UC_X86_REG_R8, pay_end)
        emu.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
        rsp = emu.mu.reg_read(UC_X86_REG_RSP) - 8
        emu.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
        emu.mu.reg_write(UC_X86_REG_RSP, rsp)

        h1 = emu.mu.hook_add(UC_HOOK_MEM_WRITE, emu._track_writes,
                              begin=ALLOC_BASE, end=ALLOC_BASE + ALLOC_SIZE)
        h2 = emu.mu.hook_add(UC_HOOK_MEM_WRITE, emu._track_writes,
                              begin=HEAP_BASE, end=HEAP_BASE + 0x1000)

        try:
            emu.mu.emu_start(BASE_ADDR + pid_info['deser'], STOP_ADDR, timeout=10_000_000)
            al = emu.mu.reg_read(UC_X86_REG_RAX) & 0xFF
            print(f"    Return AL={al}")
        except Exception as e:
            print(f"    Emulation error: {e}")
            al = 0

        emu.mu.hook_del(h1)
        emu.mu.hook_del(h2)

        # Check payload consumption
        new_pos = struct.unpack('<Q', emu.mu.mem_read(pp_addr, 8))[0]
        consumed = new_pos - pay_addr
        print(f"    Consumed {consumed}/{len(payload)} bytes")

        # Dump ALL struct changes
        print(f"    Struct writes ({len([w for w in emu._write_log if w[0]=='HEAP'])}):")
        for region, off, sz, val in emu._write_log:
            if region == 'HEAP':
                # Check if changed from constructor
                if off < len(clean_struct) and off + sz <= len(clean_struct):
                    orig = int.from_bytes(clean_struct[off:off + sz], 'little')
                    changed = '***' if val != orig else '   '
                else:
                    changed = '???'
                if sz == 4:
                    fval = struct.unpack('<f', struct.pack('<I', val & 0xFFFFFFFF))[0]
                    print(f"      {changed} HEAP+0x{off:04x} [{sz}B] = 0x{val:08x} (i32={val if val < 0x80000000 else val-0x100000000}, f32={fval:.4f})")
                elif sz == 8:
                    print(f"      {changed} HEAP+0x{off:04x} [{sz}B] = 0x{val:016x}")
                elif sz == 2:
                    print(f"      {changed} HEAP+0x{off:04x} [{sz}B] = 0x{val:04x} (u16={val})")
                elif sz == 1:
                    print(f"      {changed} HEAP+0x{off:04x} [{sz}B] = 0x{val:02x}")

        # Dump alloc writes
        alloc_writes = [w for w in emu._write_log if w[0] == 'ALLOC']
        if alloc_writes:
            print(f"    Alloc writes ({len(alloc_writes)}):")
            for region, off, sz, val in alloc_writes[:50]:
                if sz == 4:
                    fval = struct.unpack('<f', struct.pack('<I', val & 0xFFFFFFFF))[0]
                    print(f"      ALLOC+0x{off:04x} [{sz}B] = 0x{val:08x} (f32={fval:.4f})")
                elif sz == 1:
                    print(f"      ALLOC+0x{off:04x} [{sz}B] = 0x{val:02x}")
                else:
                    print(f"      ALLOC+0x{off:04x} [{sz}B] = 0x{val:0{sz*2}x}")

        # Read final struct state
        final_struct = bytes(emu.mu.mem_read(HEAP_BASE, 0x100))
        diffs = []
        for i in range(0, min(len(clean_struct), 0x100), 4):
            c = int.from_bytes(clean_struct[i:i+4], 'little')
            f = int.from_bytes(final_struct[i:i+4], 'little')
            if c != f:
                fval = struct.unpack('<f', struct.pack('<I', f & 0xFFFFFFFF))[0]
                diffs.append((i, c, f, fval))
        if diffs:
            print(f"    Struct diff (final vs ctor):")
            for off, old, new, fval in diffs:
                print(f"      +0x{off:04x}: 0x{old:08x} → 0x{new:08x} (f32={fval:.4f})")


def oracle_correlation_test(emu, rofl_path, oracle_path, entity=0x400000b2):
    """Full looping decode for ALL blocks, track field_tag→struct changes, correlate with oracle."""
    oracle = json.load(open(oracle_path))
    snapshots = oracle['snapshots']
    game_length = oracle['game_length']

    # Build oracle timeline for Garen
    oracle_timeline = []
    for s in snapshots:
        g = s['players'].get('Garen', {})
        oracle_timeline.append({
            'time': s['actual_time'],
            'level': g.get('level', 0),
            'cs': g.get('scores', {}).get('creepScore', 0),
            'kills': g.get('scores', {}).get('kills', 0),
        })

    target_pids = [368, 305, 224, 169, 539, 487]

    for pid in target_pids:
        print(f"\n{'='*70}")
        print(f"FULL DECODE: PID {pid}")
        print(f"{'='*70}")

        pid_info = emu.pid_map.get(str(pid))
        if not pid_info:
            print("  Not in pid_map")
            continue

        blocks = parse_rofl_blocks(rofl_path, pid_filter=pid, param_filter={entity})
        if not blocks:
            print("  No blocks")
            continue

        n_blocks = len(blocks)
        time_per_block = game_length / max(n_blocks, 1)
        print(f"  {n_blocks} blocks, ~{time_per_block:.1f}s per block")

        # For each block, do FULL looping decode and track:
        # - The SKIP varint (field tag) for each sub-message
        # - The struct changes per sub-message
        # - Accumulated final struct per block

        # Collect: for each (field_tag, struct_offset) → time series of values
        tag_offset_series = defaultdict(list)  # (tag, off) → [(block_idx, val)]
        # Also collect final struct per block (all sub-messages applied)
        final_struct_series = defaultdict(list)  # off → [(block_idx, val)]

        for bi, block in enumerate(blocks):
            payload = block['payload']
            pay_addr = SCRATCH_BASE + 0x100
            pp_addr = SCRATCH_BASE + 0x200
            pay_end = pay_addr + len(payload)
            emu.mu.mem_write(pay_addr, payload + b'\x00' * 128)

            # Run all sub-messages, accumulating changes to a single struct
            emu._alloc_cursor = 0
            emu._alloc_regions = []
            emu.mu.mem_write(ALLOC_BASE, b'\x00' * min(0x10000, ALLOC_SIZE))
            emu._run_constructor(pid)
            clean_struct = bytes(emu.mu.mem_read(HEAP_BASE, 0x80))

            current_pos = pay_addr
            for iteration in range(20):
                if current_pos >= pay_end:
                    break

                emu._write_log = []
                # DON'T reset struct between sub-messages — accumulate

                emu.mu.mem_write(pp_addr, struct.pack('<Q', current_pos))
                emu.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
                emu.mu.reg_write(UC_X86_REG_RDX, pp_addr)
                emu.mu.reg_write(UC_X86_REG_R8, pay_end)
                emu.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
                rsp = emu.mu.reg_read(UC_X86_REG_RSP) - 8
                emu.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
                emu.mu.reg_write(UC_X86_REG_RSP, rsp)

                h1 = emu.mu.hook_add(UC_HOOK_MEM_WRITE, emu._track_writes,
                                      begin=ALLOC_BASE, end=ALLOC_BASE + ALLOC_SIZE)
                h2 = emu.mu.hook_add(UC_HOOK_MEM_WRITE, emu._track_writes,
                                      begin=HEAP_BASE, end=HEAP_BASE + 0x1000)
                try:
                    emu.mu.emu_start(BASE_ADDR + pid_info['deser'], STOP_ADDR, timeout=10_000_000)
                except:
                    pass
                emu.mu.hook_del(h1)
                emu.mu.hook_del(h2)

                new_pos = struct.unpack('<Q', emu.mu.mem_read(pp_addr, 8))[0]
                if new_pos <= current_pos:
                    break

                # Extract field tag from struct+0xC
                tag_raw = struct.unpack('<I', emu.mu.mem_read(HEAP_BASE + 0xC, 4))[0]
                tag = tag_raw ^ (1 << 30)  # undo btc bit 30

                # Track per-tag struct changes
                for region, off, sz, val in emu._write_log:
                    if region == 'HEAP' and off != 0xC and sz >= 2:
                        tag_offset_series[(tag, off, sz)].append((bi, val))

                current_pos = new_pos

            # Read final accumulated struct
            final_struct = bytes(emu.mu.mem_read(HEAP_BASE, 0x80))
            for off in range(0, 0x78, 2):
                fv = int.from_bytes(final_struct[off:off+2], 'little')
                cv = int.from_bytes(clean_struct[off:off+2], 'little') if off+2 <= len(clean_struct) else 0
                if fv != cv:
                    final_struct_series[off].append((bi, fv))
            for off in range(0, 0x78, 4):
                fv = int.from_bytes(final_struct[off:off+4], 'little')
                cv = int.from_bytes(clean_struct[off:off+4], 'little') if off+4 <= len(clean_struct) else 0
                if fv != cv:
                    final_struct_series[off + 0x2000].append((bi, fv))

        # Analysis: find field tags with values that correlate with oracle
        print(f"\n  Field tag analysis ({len(tag_offset_series)} unique tag/offset combos):")

        # Group by tag
        tags = set(t for (t, o, s) in tag_offset_series.keys())
        for tag in sorted(tags):
            tag_entries = {(o, s): v for (t, o, s), v in tag_offset_series.items() if t == tag}
            if not tag_entries:
                continue
            total_writes = sum(len(v) for v in tag_entries.values())
            if total_writes < 10:
                continue
            offsets_str = ", ".join(f"+0x{o:x}[{s}B]" for (o, s) in sorted(tag_entries.keys()))
            print(f"    tag={tag} (0x{tag:x}): {total_writes} writes at {offsets_str}")

            # For each offset, check value range
            for (off, sz), series in sorted(tag_entries.items()):
                if len(series) < 5:
                    continue
                vals = [v for _, v in series]
                unique_vals = sorted(set(vals))

                # Level check (1-20)
                if all(1 <= v <= 20 for v in unique_vals) and len(unique_vals) >= 3:
                    sample = [(bi, v) for bi, v in series[::max(1, len(series)//5)]][:5]
                    print(f"      *** LEVEL? +0x{off:x}[{sz}B]: {unique_vals}")
                    for bi, v in sample:
                        t = bi * time_per_block
                        closest = min(oracle_timeline, key=lambda o: abs(o['time'] - t))
                        print(f"          block {bi} (~{t:.0f}s): decoded={v}, oracle_level={closest['level']}")

                # CS check (0-500, monotonic)
                elif sz >= 2 and all(0 <= v <= 500 for v in unique_vals) and max(vals) > 30:
                    increases = sum(1 for i in range(1, len(vals)) if vals[i] > vals[i-1])
                    if increases > len(vals) * 0.4:
                        sample = [(bi, v) for bi, v in series[::max(1, len(series)//5)]][:5]
                        print(f"      *** CS? +0x{off:x}[{sz}B]: range [{min(vals)}-{max(vals)}], {len(unique_vals)} unique")
                        for bi, v in sample:
                            t = bi * time_per_block
                            closest = min(oracle_timeline, key=lambda o: abs(o['time'] - t))
                            print(f"          block {bi} (~{t:.0f}s): decoded={v}, oracle_CS={closest['cs']}")

                # Coordinate check (500-15000 game units)
                elif sz >= 2:
                    in_range = sum(1 for v in vals if 500 <= v <= 15000)
                    if in_range > len(vals) * 0.5 and len(unique_vals) > 5:
                        print(f"      COORD? +0x{off:x}[{sz}B]: {in_range}/{len(vals)} in range, "
                              f"{len(unique_vals)} unique [{min(vals)}-{max(vals)}]")

        # Also check final accumulated struct
        print(f"\n  Final struct analysis (accumulated across sub-messages):")
        for off in sorted(final_struct_series.keys()):
            series = final_struct_series[off]
            if len(series) < 20:
                continue
            vals = [v for _, v in series]
            unique = sorted(set(vals))
            is_u32 = off >= 0x2000
            real_off = off - 0x2000 if is_u32 else off
            label = f"u32+0x{real_off:04x}" if is_u32 else f"u16+0x{real_off:04x}"

            # Level
            if all(1 <= v <= 20 for v in unique) and len(unique) >= 3:
                increases = sum(1 for i in range(1, len(vals)) if vals[i] > vals[i-1])
                print(f"    *** LEVEL? {label}: {unique}, inc={increases}")
                for si in [0, len(series)//4, len(series)//2, 3*len(series)//4, len(series)-1]:
                    bi, v = series[si]
                    t = bi * time_per_block
                    closest = min(oracle_timeline, key=lambda o: abs(o['time'] - t))
                    print(f"        block {bi} (~{t:.0f}s): val={v}, oracle={closest['level']}")

            # CS
            elif not is_u32 and all(0 <= v <= 500 for v in unique) and max(vals) > 50 and len(unique) > 10:
                increases = sum(1 for i in range(1, len(vals)) if vals[i] > vals[i-1])
                if increases > len(vals) * 0.3:
                    print(f"    *** CS? {label}: range [{min(vals)}-{max(vals)}], {len(unique)} unique, inc={increases}")
                    for si in [0, len(series)//4, len(series)//2, 3*len(series)//4, len(series)-1]:
                        bi, v = series[si]
                        t = bi * time_per_block
                        closest = min(oracle_timeline, key=lambda o: abs(o['time'] - t))
                        print(f"        block {bi} (~{t:.0f}s): val={v}, oracle={closest['cs']}")


def extract_intermediate_values(emu, rofl_path, pid, entity=0x400000b2):
    """Extract INTERMEDIATE values from the bit reader BEFORE the per-PID cipher encrypts them.

    Key discovery: the deserializer has two phases per field:
      1. Bit reader (RVA 0xfe0000-0xff1000) reads plaintext game data from payload,
         accumulates 4 bytes big-endian, bswaps to LE, writes to struct.
      2. Per-PID cipher (handler-specific, 48+ unique cipher chains for PID 842)
         reads from struct, encrypts byte-by-byte, writes back to same offset.

    The INTERMEDIATE values (after bit reader, before cipher) ARE the game data.
    This function captures them via write hooks.
    """
    from unicorn import UC_HOOK_MEM_WRITE

    blocks = parse_rofl_blocks(rofl_path, pid_filter=pid, param_filter={entity})
    n_blocks = len(blocks)
    if not blocks:
        print(f"No blocks for PID {pid}")
        return {}

    pid_info = emu.pid_map.get(str(pid))
    if not pid_info:
        return {}

    deser_addr = BASE_ADDR + pid_info['deser']
    game_length = 2532.0  # default, override with oracle
    tpb = game_length / n_blocks

    print(f"PID {pid}: {n_blocks} blocks, extracting intermediate values...")

    # (struct_offset) → [(block_idx, game_time, f32_value)]
    all_intermediate = defaultdict(list)

    for bi in range(n_blocks):
        block = blocks[bi]
        payload = block['payload']

        emu._alloc_cursor = 0
        emu._alloc_regions = []
        emu._run_constructor(pid)

        pay_addr = SCRATCH_BASE + 0x100
        pp_addr = SCRATCH_BASE + 0x200
        pay_end = pay_addr + len(payload)
        emu.mu.mem_write(pay_addr, payload + b'\x00' * 128)

        block_vals = []

        def write_hook(mu, access, address, size, value, ud):
            val = value & ((1 << (size * 8)) - 1)
            hoff = address - HEAP_BASE
            if 0 <= hoff < 0x200 and size == 4:
                rip = mu.reg_read(UC_X86_REG_RIP)
                rva = rip - BASE_ADDR
                # Bit reader functions are in RVA range 0xfe0000-0xff1000
                if 0xfe0000 <= rva <= 0xff1000:
                    fval = struct.unpack('<f', struct.pack('<I', val))[0]
                    if fval == fval and abs(fval) < 20000 and abs(fval) > 0.5:
                        block_vals.append((hoff, fval))
            return True

        current_pos = pay_addr
        for iteration in range(20):
            if current_pos >= pay_end:
                break
            block_vals.clear()

            emu.mu.mem_write(pp_addr, struct.pack('<Q', current_pos))
            emu.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
            emu.mu.reg_write(UC_X86_REG_RDX, pp_addr)
            emu.mu.reg_write(UC_X86_REG_R8, pay_end)
            emu.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
            rsp = emu.mu.reg_read(UC_X86_REG_RSP) - 8
            emu.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
            emu.mu.reg_write(UC_X86_REG_RSP, rsp)

            h = emu.mu.hook_add(UC_HOOK_MEM_WRITE, write_hook,
                                begin=HEAP_BASE, end=HEAP_BASE + 0x200)
            try:
                emu.mu.emu_start(deser_addr, STOP_ADDR, timeout=10_000_000)
            except:
                pass
            emu.mu.hook_del(h)

            new_pos = struct.unpack('<Q', emu.mu.mem_read(pp_addr, 8))[0]
            if new_pos <= current_pos:
                break

            for hoff, fval in block_vals:
                all_intermediate[hoff].append((bi, bi * tpb, fval))

            current_pos = new_pos

        if bi % 50 == 0:
            print(f"  Block {bi}/{n_blocks}...")

    # Print summary
    print(f"\nIntermediate values extracted: {len(all_intermediate)} struct offsets")
    for off in sorted(all_intermediate.keys()):
        series = all_intermediate[off]
        if len(series) < 5:
            continue
        vals = [v for _, _, v in series]
        unique = len(set(round(v, 1) for v in vals))
        if unique < 3:
            continue
        coord_pct = sum(1 for v in vals if 500 < abs(v) < 15000) / len(vals) * 100
        print(f"  +0x{off:04x}: {len(series)} vals, {unique} unique, "
              f"[{min(vals):.0f}, {max(vals):.0f}], {coord_pct:.0f}% in coord range")

    return all_intermediate


def comprehensive_extraction(emu, rofl_path, pid, entity=0x400000b2):
    """Capture ALL writes to struct from ALL code regions, tagged by source RVA.

    Unlike extract_intermediate_values which only gets f32 from bit reader,
    this captures u8/u16/u32/u64 from any code path — including the short
    varint reader that may carry level/CS as u16.
    """
    from unicorn import UC_HOOK_MEM_WRITE

    oracle = json.load(open('/tmp/oracle_data.json'))
    snapshots = oracle['snapshots']
    game_length = oracle['game_length']

    oracle_timeline = []
    for s in snapshots:
        g = s['players'].get('Garen', {})
        oracle_timeline.append({
            'time': s['actual_time'],
            'level': g.get('level', 0),
            'cs': g.get('scores', {}).get('creepScore', 0),
        })

    blocks = parse_rofl_blocks(rofl_path, pid_filter=pid, param_filter={entity})
    n_blocks = len(blocks)
    if not blocks:
        print(f"No blocks for PID {pid}")
        return
    tpb = game_length / n_blocks

    pid_info = emu.pid_map.get(str(pid))
    if not pid_info:
        return
    deser_addr = BASE_ADDR + pid_info['deser']

    print(f"PID {pid}: {n_blocks} blocks, {tpb:.1f}s/block, comprehensive extraction...")

    # (struct_offset, write_size) → [(block_idx, game_time, raw_value, source_rva)]
    all_writes = defaultdict(list)
    # Separate: track pre-cipher f32 from bit reader range specifically
    bitreader_f32 = defaultdict(list)

    for bi in range(n_blocks):
        block = blocks[bi]
        payload = block['payload']
        gt = bi * tpb

        emu._alloc_cursor = 0
        emu._alloc_regions = []
        emu._run_constructor(pid)

        pay_addr = SCRATCH_BASE + 0x100
        pp_addr = SCRATCH_BASE + 0x200
        pay_end = pay_addr + len(payload)
        emu.mu.mem_write(pay_addr, payload + b'\x00' * 128)

        block_writes = []

        def write_hook(mu, access, address, size, value, ud):
            val = value & ((1 << (size * 8)) - 1)
            hoff = address - HEAP_BASE
            if 0x08 <= hoff < 0x200:  # skip vtable ptr at +0x00
                rip = mu.reg_read(UC_X86_REG_RIP)
                rva = rip - BASE_ADDR
                block_writes.append((hoff, size, val, rva))
            return True

        current_pos = pay_addr
        for iteration in range(20):
            if current_pos >= pay_end:
                break
            block_writes.clear()

            emu.mu.mem_write(pp_addr, struct.pack('<Q', current_pos))
            emu.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
            emu.mu.reg_write(UC_X86_REG_RDX, pp_addr)
            emu.mu.reg_write(UC_X86_REG_R8, pay_end)
            emu.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
            rsp = emu.mu.reg_read(UC_X86_REG_RSP) - 8
            emu.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
            emu.mu.reg_write(UC_X86_REG_RSP, rsp)

            h = emu.mu.hook_add(UC_HOOK_MEM_WRITE, write_hook,
                                begin=HEAP_BASE, end=HEAP_BASE + 0x200)
            try:
                emu.mu.emu_start(deser_addr, STOP_ADDR, timeout=10_000_000)
            except:
                pass
            emu.mu.hook_del(h)

            new_pos = struct.unpack('<Q', emu.mu.mem_read(pp_addr, 8))[0]
            if new_pos <= current_pos:
                break

            # Record all writes from this sub-message
            for hoff, sz, val, rva in block_writes:
                all_writes[(hoff, sz)].append((bi, gt, val, rva))
                # Also capture bit reader f32 specifically
                if sz == 4 and 0xfe0000 <= rva <= 0xff1000:
                    fval = struct.unpack('<f', struct.pack('<I', val))[0]
                    if fval == fval and abs(fval) < 20000:
                        bitreader_f32[hoff].append((bi, gt, fval))

            current_pos = new_pos

        if bi % 100 == 0:
            print(f"  Block {bi}/{n_blocks}...")

    # ── Analysis ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"ANALYSIS: {len(all_writes)} unique (offset, size) combos")
    print(f"{'='*70}")

    # 1. Find fields with values matching level range (1-20) that change over time
    print(f"\n── Level candidates (values 1-20, increasing over game) ──")
    for (off, sz), series in sorted(all_writes.items()):
        if len(series) < 20:
            continue
        vals = [v for _, _, v, _ in series]
        unique = sorted(set(vals))
        if all(1 <= v <= 20 for v in unique) and len(unique) >= 3:
            # Check if generally increasing
            increases = sum(1 for i in range(1, len(vals)) if vals[i] > vals[i-1])
            if increases > len(vals) * 0.2:
                # Sample against oracle
                sample_idxs = [0, len(series)//4, len(series)//2, 3*len(series)//4, len(series)-1]
                print(f"\n  +0x{off:03x} [{sz}B]: {len(unique)} unique vals {unique}, {increases} increases")
                rvas = set(rva for _, _, _, rva in series)
                print(f"    Source RVAs: {', '.join(f'0x{r:x}' for r in sorted(rvas))}")
                for si in sample_idxs:
                    bi, gt, val, rva = series[si]
                    closest = min(oracle_timeline, key=lambda o: abs(o['time'] - gt))
                    print(f"    block {bi} (~{gt:.0f}s): val={val}, oracle_level={closest['level']}")

    # 2. Find fields with values matching CS range (0-400, mostly increasing)
    print(f"\n── CS candidates (0-400, increasing trend) ──")
    for (off, sz), series in sorted(all_writes.items()):
        if len(series) < 20:
            continue
        vals = [v for _, _, v, _ in series]
        unique = sorted(set(vals))
        if max(vals) <= 500 and max(vals) >= 30 and min(vals) >= 0:
            increases = sum(1 for i in range(1, len(vals)) if vals[i] > vals[i-1])
            if increases > len(vals) * 0.3:
                sample_idxs = [0, len(series)//4, len(series)//2, 3*len(series)//4, len(series)-1]
                print(f"\n  +0x{off:03x} [{sz}B]: range [{min(vals)}-{max(vals)}], "
                      f"{len(unique)} unique, {increases} inc")
                rvas = set(rva for _, _, _, rva in series)
                print(f"    Source RVAs: {', '.join(f'0x{r:x}' for r in sorted(rvas))}")
                for si in sample_idxs:
                    bi, gt, val, rva = series[si]
                    closest = min(oracle_timeline, key=lambda o: abs(o['time'] - gt))
                    print(f"    block {bi} (~{gt:.0f}s): val={val}, oracle_CS={closest['cs']}")

    # 3. Trajectory analysis for bit reader f32 coordinates
    print(f"\n── Coordinate trajectory (bit reader f32, 500-15000 range) ──")
    for off in sorted(bitreader_f32.keys()):
        series = bitreader_f32[off]
        if len(series) < 20:
            continue
        vals = [v for _, _, v in series]
        in_coord = sum(1 for v in vals if 500 < abs(v) < 15000)
        if in_coord < len(vals) * 0.3:
            continue
        # Check smoothness: median absolute difference between consecutive values
        diffs = [abs(vals[i] - vals[i-1]) for i in range(1, len(vals))]
        if not diffs:
            continue
        median_diff = sorted(diffs)[len(diffs)//2]
        max_diff = max(diffs)
        unique = len(set(round(v, 1) for v in vals))

        print(f"\n  +0x{off:03x} f32: {len(series)} vals, {unique} unique, "
              f"{in_coord}/{len(vals)} in coord range")
        print(f"    Range: [{min(vals):.1f}, {max(vals):.1f}], "
              f"median_step={median_diff:.1f}, max_step={max_diff:.1f}")
        # Smooth = small median step relative to range
        val_range = max(vals) - min(vals)
        if val_range > 0 and median_diff < val_range * 0.1:
            print(f"    *** SMOOTH TRAJECTORY (step/range = {median_diff/val_range:.3f})")
        # Sample values
        for si in [0, len(series)//4, len(series)//2, 3*len(series)//4, len(series)-1]:
            bi, gt, fval = series[si]
            print(f"    block {bi} (~{gt:.0f}s): {fval:.1f}")

    # 4. Overview: all (offset, size) combos with high write counts, grouped by source RVA
    print(f"\n── Top write targets by source code region ──")
    rva_groups = defaultdict(list)
    for (off, sz), series in all_writes.items():
        for _, _, val, rva in series[:3]:  # sample first few
            rva_groups[rva].append((off, sz, len(series)))
    for rva in sorted(rva_groups.keys()):
        targets = rva_groups[rva]
        total = sum(n for _, _, n in targets)
        if total < 20:
            continue
        offsets_str = ", ".join(f"+0x{o:x}[{s}B]({n})" for o, s, n in sorted(set(targets))[:10])
        is_bitreader = 0xfe0000 <= rva <= 0xff1000
        is_cipher = 0x1002000 <= rva <= 0x1007000
        label = " [bitreader]" if is_bitreader else " [cipher]" if is_cipher else ""
        print(f"  RVA 0x{rva:07x}{label}: {offsets_str}")

    return all_writes, bitreader_f32


def dump_bitreader_values(emu, rofl_path, pid, entity=0x400000b2):
    """Dump ALL bit reader 4-byte writes (pre and post cipher) to understand value ranges."""
    from unicorn import UC_HOOK_MEM_WRITE

    blocks = parse_rofl_blocks(rofl_path, pid_filter=pid, param_filter={entity})
    n_blocks = len(blocks)
    if not blocks:
        print(f"No blocks for PID {pid}")
        return

    pid_info = emu.pid_map.get(str(pid))
    deser_addr = BASE_ADDR + pid_info['deser']
    tpb = 2532.0 / n_blocks

    print(f"PID {pid}: {n_blocks} blocks, dumping ALL writes to key offsets...")

    # Track: offset → [(block_idx, [(size, value, rva)])]
    # Focus on offsets +0x38 (known coord), +0x50 (high count), +0x14 (high count)
    target_offsets = {0x14, 0x20, 0x24, 0x38, 0x3c, 0x40, 0x44, 0x48, 0x50, 0x68, 0x74, 0x90}
    # offset → [(block, write_seq, size, val, rva)]
    offset_writes = defaultdict(list)

    sample_blocks = [0, 1, 2, 10, 50, 100, 150, 200, 269]
    sample_blocks = [b for b in sample_blocks if b < n_blocks]

    for bi in sample_blocks:
        block = blocks[bi]
        payload = block['payload']

        emu._alloc_cursor = 0
        emu._alloc_regions = []
        emu._run_constructor(pid)

        pay_addr = SCRATCH_BASE + 0x100
        pp_addr = SCRATCH_BASE + 0x200
        pay_end = pay_addr + len(payload)
        emu.mu.mem_write(pay_addr, payload + b'\x00' * 128)

        write_seq = [0]  # mutable counter

        def write_hook(mu, access, address, size, value, ud):
            val = value & ((1 << (size * 8)) - 1)
            hoff = address - HEAP_BASE
            if hoff in target_offsets or (hoff & ~3) in target_offsets:
                rip = mu.reg_read(UC_X86_REG_RIP)
                rva = rip - BASE_ADDR
                offset_writes[hoff].append((bi, write_seq[0], size, val, rva))
                write_seq[0] += 1
            return True

        current_pos = pay_addr
        for iteration in range(20):
            if current_pos >= pay_end:
                break

            emu.mu.mem_write(pp_addr, struct.pack('<Q', current_pos))
            emu.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
            emu.mu.reg_write(UC_X86_REG_RDX, pp_addr)
            emu.mu.reg_write(UC_X86_REG_R8, pay_end)
            emu.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
            rsp = emu.mu.reg_read(UC_X86_REG_RSP) - 8
            emu.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
            emu.mu.reg_write(UC_X86_REG_RSP, rsp)

            h = emu.mu.hook_add(UC_HOOK_MEM_WRITE, write_hook,
                                begin=HEAP_BASE, end=HEAP_BASE + 0x200)
            try:
                emu.mu.emu_start(deser_addr, STOP_ADDR, timeout=10_000_000)
            except:
                pass
            emu.mu.hook_del(h)

            new_pos = struct.unpack('<Q', emu.mu.mem_read(pp_addr, 8))[0]
            if new_pos <= current_pos:
                break
            current_pos = new_pos

    # Dump write sequences for key offsets per block
    print(f"\n{'='*70}")
    print(f"WRITE SEQUENCES AT TARGET OFFSETS")
    print(f"{'='*70}")

    for off in sorted(target_offsets):
        writes = offset_writes.get(off, [])
        if not writes:
            continue
        print(f"\n── +0x{off:03x} ({len(writes)} total writes) ──")
        for bi in sample_blocks:
            bwrites = [(seq, sz, val, rva) for (b, seq, sz, val, rva) in writes if b == bi]
            if not bwrites:
                continue
            gt = bi * tpb
            print(f"  Block {bi} (~{gt:.0f}s): {len(bwrites)} writes")
            for seq, sz, val, rva in bwrites:
                is_br = 0xfe0000 <= rva <= 0xff1000
                label = "BR" if is_br else "CI" if 0x1002000 <= rva <= 0x1007000 else "??"
                if sz == 4:
                    fval = struct.unpack('<f', struct.pack('<I', val))[0]
                    print(f"    #{seq:3d} [{sz}B] RVA=0x{rva:07x} [{label}] = 0x{val:08x} (f32={fval:.4f})")
                elif sz == 2:
                    print(f"    #{seq:3d} [{sz}B] RVA=0x{rva:07x} [{label}] = 0x{val:04x} (u16={val})")
                elif sz == 1:
                    print(f"    #{seq:3d} [{sz}B] RVA=0x{rva:07x} [{label}] = 0x{val:02x}")
                else:
                    print(f"    #{seq:3d} [{sz}B] RVA=0x{rva:07x} [{label}] = 0x{val:x}")

    # Also dump: for +0x38, track the FIRST 4B write (pre-cipher plaintext) across ALL blocks
    print(f"\n{'='*70}")
    print(f"FIRST 4B WRITE TO +0x38 ACROSS ALL BLOCKS (trajectory)")
    print(f"{'='*70}")

    first_f32_38 = []
    for bi in range(n_blocks):
        block = blocks[bi]
        payload = block['payload']

        emu._alloc_cursor = 0
        emu._alloc_regions = []
        emu._run_constructor(pid)

        pay_addr = SCRATCH_BASE + 0x100
        pp_addr = SCRATCH_BASE + 0x200
        pay_end = pay_addr + len(payload)
        emu.mu.mem_write(pay_addr, payload + b'\x00' * 128)

        captured = [None]

        def write_hook_38(mu, access, address, size, value, ud):
            val = value & ((1 << (size * 8)) - 1)
            hoff = address - HEAP_BASE
            if hoff == 0x38 and size == 4 and captured[0] is None:
                rip = mu.reg_read(UC_X86_REG_RIP)
                rva = rip - BASE_ADDR
                fval = struct.unpack('<f', struct.pack('<I', val))[0]
                captured[0] = (fval, rva)
            return True

        current_pos = pay_addr
        for iteration in range(20):
            if current_pos >= pay_end:
                break
            emu.mu.mem_write(pp_addr, struct.pack('<Q', current_pos))
            emu.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
            emu.mu.reg_write(UC_X86_REG_RDX, pp_addr)
            emu.mu.reg_write(UC_X86_REG_R8, pay_end)
            emu.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
            rsp = emu.mu.reg_read(UC_X86_REG_RSP) - 8
            emu.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
            emu.mu.reg_write(UC_X86_REG_RSP, rsp)

            h = emu.mu.hook_add(UC_HOOK_MEM_WRITE, write_hook_38,
                                begin=HEAP_BASE + 0x38, end=HEAP_BASE + 0x3C)
            try:
                emu.mu.emu_start(deser_addr, STOP_ADDR, timeout=10_000_000)
            except:
                pass
            emu.mu.hook_del(h)

            new_pos = struct.unpack('<Q', emu.mu.mem_read(pp_addr, 8))[0]
            if new_pos <= current_pos:
                break
            if captured[0] is not None:
                break  # got it from first sub-message
            current_pos = new_pos

        if captured[0] is not None:
            fval, rva = captured[0]
            first_f32_38.append((bi, bi * tpb, fval))
        if bi % 50 == 0:
            print(f"  Block {bi}/{n_blocks}...")

    print(f"\n  Captured {len(first_f32_38)} values at +0x38:")
    for bi, gt, fval in first_f32_38[::max(1, len(first_f32_38)//20)]:
        print(f"    block {bi:3d} (~{gt:6.0f}s): f32 = {fval:.2f}")

    # Check smoothness
    if len(first_f32_38) > 5:
        vals = [v for _, _, v in first_f32_38]
        diffs = [abs(vals[i] - vals[i-1]) for i in range(1, len(vals))]
        print(f"\n  Range: [{min(vals):.1f}, {max(vals):.1f}]")
        print(f"  Median step: {sorted(diffs)[len(diffs)//2]:.1f}")
        print(f"  Mean step: {sum(diffs)/len(diffs):.1f}")


def extract_plaintext_values(emu, rofl_path, pid, entity=0x400000b2):
    """Extract plaintext game values by capturing the LAST multi-byte bit reader write
    before each cipher 1B overwrite.

    The deserializer pattern per field:
      1. Bit reader (RVA 0xfe0000-0xff1000): accumulate bytes → bswap → write f32/u16
      2. Cipher (various RVAs): read → encrypt → overwrite byte-by-byte

    The BSWAP write is the plaintext game value. We identify it as the last ≥2B write
    from bit reader range before a 1B cipher write to the same offset neighborhood.
    """
    from unicorn import UC_HOOK_MEM_WRITE

    oracle = json.load(open('/tmp/oracle_data.json'))
    game_length = oracle['game_length']
    oracle_timeline = []
    for s in oracle['snapshots']:
        g = s['players'].get('Garen', {})
        oracle_timeline.append({
            'time': s['actual_time'],
            'level': g.get('level', 0),
            'cs': g.get('scores', {}).get('creepScore', 0),
        })

    blocks = parse_rofl_blocks(rofl_path, pid_filter=pid, param_filter={entity})
    n_blocks = len(blocks)
    if not blocks:
        print(f"No blocks for PID {pid}")
        return
    tpb = game_length / n_blocks

    pid_info = emu.pid_map.get(str(pid))
    deser_addr = BASE_ADDR + pid_info['deser']

    print(f"PID {pid}: {n_blocks} blocks, extracting plaintext values...")

    # For each struct offset aligned to 4 bytes, store plaintext time series
    # (offset) → [(block_idx, game_time, f32_value, u32_value)]
    plaintext_series = defaultdict(list)

    for bi in range(n_blocks):
        block = blocks[bi]
        payload = block['payload']
        gt = bi * tpb

        emu._alloc_cursor = 0
        emu._alloc_regions = []
        emu._run_constructor(pid)

        pay_addr = SCRATCH_BASE + 0x100
        pp_addr = SCRATCH_BASE + 0x200
        pay_end = pay_addr + len(payload)
        emu.mu.mem_write(pay_addr, payload + b'\x00' * 128)

        # Track write sequence globally (across sub-messages)
        # (offset_aligned_4, [(size, value, rva)])
        offset_history = defaultdict(list)

        def write_hook(mu, access, address, size, value, ud):
            val = value & ((1 << (size * 8)) - 1)
            hoff = address - HEAP_BASE
            if 0x08 <= hoff < 0x200:
                rip = mu.reg_read(UC_X86_REG_RIP)
                rva = rip - BASE_ADDR
                # Align to 4-byte boundary for grouping
                off4 = hoff & ~3
                offset_history[off4].append((hoff, size, val, rva))
            return True

        current_pos = pay_addr
        for iteration in range(20):
            if current_pos >= pay_end:
                break
            emu.mu.mem_write(pp_addr, struct.pack('<Q', current_pos))
            emu.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
            emu.mu.reg_write(UC_X86_REG_RDX, pp_addr)
            emu.mu.reg_write(UC_X86_REG_R8, pay_end)
            emu.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
            rsp = emu.mu.reg_read(UC_X86_REG_RSP) - 8
            emu.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
            emu.mu.reg_write(UC_X86_REG_RSP, rsp)

            h = emu.mu.hook_add(UC_HOOK_MEM_WRITE, write_hook,
                                begin=HEAP_BASE, end=HEAP_BASE + 0x200)
            try:
                emu.mu.emu_start(deser_addr, STOP_ADDR, timeout=10_000_000)
            except:
                pass
            emu.mu.hook_del(h)

            new_pos = struct.unpack('<Q', emu.mu.mem_read(pp_addr, 8))[0]
            if new_pos <= current_pos:
                break
            current_pos = new_pos

        # Post-process: for each offset group, find plaintext values
        # Pattern: bit reader writes (RVA in 0xfe0000-0xff1000) accumulate bytes,
        # then a cipher write (1B, from non-bit-reader RVA) encrypts.
        # The LAST ≥2B write from bit reader range is the bswap plaintext.
        for off4, writes in offset_history.items():
            # Split into groups: each group ends with 1B writes from cipher
            # Find all "plaintext candidates": multi-byte writes from bit reader
            # that are followed by 1B writes from non-bit-reader
            last_multibyte_br = None
            for i, (hoff, sz, val, rva) in enumerate(writes):
                is_br = 0xfe0000 <= rva <= 0xff1000
                if is_br and sz >= 2:
                    last_multibyte_br = (hoff, sz, val, rva)
                elif sz == 1 and not is_br and last_multibyte_br is not None:
                    # This 1B write is from cipher — the last_multibyte_br was plaintext
                    pt_hoff, pt_sz, pt_val, pt_rva = last_multibyte_br
                    if pt_sz == 4:
                        fval = struct.unpack('<f', struct.pack('<I', pt_val))[0]
                        plaintext_series[off4].append((bi, gt, fval, pt_val))
                    elif pt_sz == 2:
                        plaintext_series[off4].append((bi, gt, float(pt_val), pt_val))
                    last_multibyte_br = None  # consumed

        if bi % 50 == 0:
            print(f"  Block {bi}/{n_blocks}...")

    # ── Analysis ──────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"PLAINTEXT ANALYSIS: {len(plaintext_series)} offsets with data")
    print(f"{'='*70}")

    for off in sorted(plaintext_series.keys()):
        series = plaintext_series[off]
        if len(series) < 10:
            continue
        vals_f32 = [v for _, _, v, _ in series]
        vals_u32 = [v for _, _, _, v in series]

        unique_f32 = len(set(round(v, 1) for v in vals_f32))
        unique_u32 = len(set(vals_u32))
        if unique_f32 < 3:
            continue

        # Classification
        label = ""

        # Check level (f32 1.0-18.0, integers, increasing)
        int_vals = [round(v) for v in vals_f32 if 0.5 < v < 20.5 and abs(v - round(v)) < 0.01]
        if len(int_vals) > len(series) * 0.3:
            unique_ints = sorted(set(int_vals))
            if all(1 <= i <= 20 for i in unique_ints) and len(unique_ints) >= 3:
                increases = sum(1 for i in range(1, len(int_vals)) if int_vals[i] > int_vals[i-1])
                if increases > len(int_vals) * 0.15:
                    label = "*** LEVEL ***"

        # Check CS (0-500 as float, mostly increasing)
        cs_vals = [v for v in vals_f32 if 0 <= v <= 500 and abs(v - round(v)) < 0.01]
        if len(cs_vals) > len(series) * 0.3 and max(cs_vals) > 30:
            increases = sum(1 for i in range(1, len(cs_vals)) if cs_vals[i] > cs_vals[i-1])
            if increases > len(cs_vals) * 0.3 and not label:
                label = "*** CS ***"

        # Check coord trajectory
        coord_vals = [v for v in vals_f32 if 100 < abs(v) < 15500]
        if len(coord_vals) > len(series) * 0.5:
            diffs = [abs(coord_vals[i] - coord_vals[i-1]) for i in range(1, len(coord_vals))]
            if diffs:
                med = sorted(diffs)[len(diffs)//2]
                rng = max(coord_vals) - min(coord_vals)
                if rng > 100 and med < rng * 0.15:
                    label = "*** SMOOTH COORD ***"
                elif rng > 500:
                    label = "COORD?"

        if not label:
            # Skip uninteresting
            if unique_f32 < 10 and all(abs(v) < 5 or abs(v) > 100000 for v in vals_f32):
                continue

        print(f"\n  +0x{off:03x}: {len(series)} pts, {unique_f32} unique f32, "
              f"range [{min(vals_f32):.1f}, {max(vals_f32):.1f}]  {label}")

        # Sample values with oracle comparison
        sample_idxs = [0, len(series)//4, len(series)//2, 3*len(series)//4, len(series)-1]
        for si in sample_idxs:
            bi, gt, fval, u32 = series[si]
            closest = min(oracle_timeline, key=lambda o: abs(o['time'] - gt))
            print(f"    block {bi:3d} (~{gt:6.0f}s): f32={fval:12.2f}, u32=0x{u32:08x}, "
                  f"oracle: L={closest['level']}, CS={closest['cs']}")

    return plaintext_series


def multi_pid_survey(emu, rofl_path, entity=0x400000b2):
    """Quick survey of many PIDs — look for plaintext-like values in struct."""
    oracle = json.load(open('/tmp/oracle_data.json'))
    game_length = oracle['game_length']

    oracle_timeline = []
    for s in oracle['snapshots']:
        g = s['players'].get('Garen', {})
        oracle_timeline.append({
            'time': s['actual_time'],
            'level': g.get('level', 0),
            'cs': g.get('scores', {}).get('creepScore', 0),
        })

    # Get ALL PIDs that have blocks for this entity
    print("Scanning all blocks for entity...")
    all_blocks = parse_rofl_blocks(rofl_path, param_filter={entity})
    pid_counts = defaultdict(list)
    for b in all_blocks:
        pid_counts[b['pid']].append(b)

    print(f"Found {len(pid_counts)} PIDs with blocks for entity 0x{entity:08x}")
    for pid in sorted(pid_counts.keys(), key=lambda p: -len(pid_counts[p]))[:30]:
        cnt = len(pid_counts[pid])
        has_deser = str(pid) in emu.pid_map
        print(f"  PID {pid:4d}: {cnt:4d} blocks {'✓' if has_deser else '✗'}")

    # Test PIDs with enough blocks
    interesting = []
    for pid in sorted(pid_counts.keys(), key=lambda p: -len(pid_counts[p])):
        blocks = pid_counts[pid]
        cnt = len(blocks)
        if cnt < 10:
            continue
        pid_info = emu.pid_map.get(str(pid))
        if not pid_info:
            continue

        tpb = game_length / cnt

        # Decode 5 sample blocks across game timeline
        sample_idxs = [0, cnt//4, cnt//2, 3*cnt//4, cnt-1]
        struct_snapshots = []

        for si in sample_idxs:
            block = blocks[si]
            payload = block['payload']
            gt = si * tpb

            results = emu.decode_payload(pid, payload, verbose=False)
            if not results:
                continue

            # Read final struct
            final = bytes(emu.mu.mem_read(HEAP_BASE, 0x200))

            # Collect ALL u32/f32 values at each offset
            fields = {}
            for off in range(8, 0x1F8, 4):
                f = int.from_bytes(final[off:off+4], 'little')
                if f != 0:
                    fval = struct.unpack('<f', struct.pack('<I', f))[0]
                    fields[off] = (f, fval)
            # Also u16
            for off in range(8, 0x1F8, 2):
                v = int.from_bytes(final[off:off+2], 'little')
                if v != 0:
                    fields[(off, 'u16')] = (v, float(v))

            struct_snapshots.append((si, gt, fields))

        if len(struct_snapshots) < 3:
            continue

        # Look for game-like patterns across samples
        details = []

        # Check ALL u32 offsets
        all_offsets = set()
        for _, _, d in struct_snapshots:
            all_offsets.update(k for k in d.keys() if isinstance(k, int))

        for off in sorted(all_offsets):
            vals_u32 = []
            vals_f32 = []
            for si, gt, d in struct_snapshots:
                if off in d:
                    u32, f32 = d[off]
                    vals_u32.append((si, gt, u32))
                    vals_f32.append((si, gt, f32))

            if len(vals_u32) < 3:
                continue

            u32s = [v[2] for v in vals_u32]
            f32s = [v[2] for v in vals_f32]

            # LEVEL: values 1-20, generally increasing
            if all(1 <= u <= 20 for u in u32s) and len(set(u32s)) >= 3:
                if u32s[-1] >= u32s[0]:
                    # Check against oracle
                    matches = 0
                    for si, gt, u in vals_u32:
                        closest = min(oracle_timeline, key=lambda o: abs(o['time'] - gt))
                        if u == closest['level']:
                            matches += 1
                    details.append((off, f'LEVEL (match={matches}/{len(vals_u32)})', vals_u32, 'u32'))

            # COORD f32: 100-15000, changes
            elif all(100 < abs(f) < 15500 and f == f for f in f32s) and len(set(round(f, 0) for f in f32s)) >= 2:
                details.append((off, 'COORD_F32', [(si, gt, f) for si, gt, f in vals_f32], 'f32'))

            # COORD u32: 500-15000
            elif all(500 <= u <= 15000 for u in u32s) and len(set(u32s)) >= 2:
                details.append((off, 'COORD_U32', vals_u32, 'u32'))

            # CS: 0-500, increasing
            elif all(0 <= u <= 500 for u in u32s) and u32s[-1] > u32s[0] + 20 and len(set(u32s)) >= 3:
                matches = 0
                for si, gt, u in vals_u32:
                    closest = min(oracle_timeline, key=lambda o: abs(o['time'] - gt))
                    if abs(u - closest['cs']) <= 10:
                        matches += 1
                if matches >= 2:
                    details.append((off, f'CS (match={matches}/{len(vals_u32)})', vals_u32, 'u32'))

        if details:
            interesting.append((pid, cnt, details))
            print(f"\n{'='*60}")
            print(f"*** PID {pid}: {cnt} blocks, {tpb:.1f}s/block ***")
            print(f"{'='*60}")
            for off, label, vals, fmt in details:
                off_str = f"+0x{off:03x}" if isinstance(off, int) else f"+0x{off[0]:03x}[u16]"
                print(f"  {off_str} [{label}]:")
                for si, gt, v in vals:
                    closest = min(oracle_timeline, key=lambda o: abs(o['time'] - gt))
                    if fmt == 'f32':
                        print(f"    block {si:3d} (~{gt:6.0f}s): f32={v:12.2f}  oracle: L={closest['level']}, CS={closest['cs']}")
                    else:
                        print(f"    block {si:3d} (~{gt:6.0f}s): val={v:8d} (0x{v:08x})  oracle: L={closest['level']}, CS={closest['cs']}")

    if not interesting:
        print("\nNo PIDs found with plaintext game-like values.")

    return interesting


def pattern_matching_oracle(emu, rofl_path, pid, entity=0x400000b2):
    """Match encrypted value PATTERNS against oracle, without needing to decrypt.

    Key insight: cipher is deterministic, so same level → same encrypted u32.
    Level transitions appear as step changes in the encrypted time series.
    We correlate the pattern of changes with oracle level/CS.
    """
    oracle = json.load(open('/tmp/oracle_data.json'))
    game_length = oracle['game_length']

    oracle_timeline = []
    for s in oracle['snapshots']:
        g = s['players'].get('Garen', {})
        oracle_timeline.append({
            'time': s['actual_time'],
            'level': g.get('level', 0),
            'cs': g.get('scores', {}).get('creepScore', 0),
        })

    blocks = parse_rofl_blocks(rofl_path, pid_filter=pid, param_filter={entity})
    n_blocks = len(blocks)
    if not blocks:
        print(f"No blocks for PID {pid}")
        return
    tpb = game_length / n_blocks

    pid_info = emu.pid_map.get(str(pid))
    if not pid_info:
        return
    deser_addr = BASE_ADDR + pid_info['deser']

    print(f"PID {pid}: {n_blocks} blocks, {tpb:.1f}s/block, decoding all blocks...")

    # For each block, decode and capture the FINAL encrypted struct
    struct_series = []  # list of (block_idx, game_time, struct_bytes)

    for bi in range(n_blocks):
        block = blocks[bi]
        payload = block['payload']
        gt = bi * tpb

        results = emu.decode_payload(pid, payload, verbose=False)
        if not results:
            struct_series.append((bi, gt, None))
            continue

        final = bytes(emu.mu.mem_read(HEAP_BASE, 0x200))
        struct_series.append((bi, gt, final))

        if bi % 100 == 0:
            print(f"  Block {bi}/{n_blocks}...")

    # Now analyze each struct offset
    print(f"\n{'='*70}")
    print(f"PATTERN ANALYSIS: PID {pid}")
    print(f"{'='*70}")

    # Build oracle level change points
    oracle_level_at = {}  # time → level
    for ot in oracle_timeline:
        oracle_level_at[round(ot['time'])] = ot['level']

    # For each struct offset (4-byte aligned), build encrypted value time series
    best_level_matches = []
    best_cs_matches = []

    for off in range(8, 0x1F8, 1):  # check every byte offset for 4-byte values
        if off + 4 > 0x200:
            continue

        # Build time series of u32 values
        time_series = []
        for bi, gt, sb in struct_series:
            if sb is None:
                continue
            val = int.from_bytes(sb[off:off+4], 'little')
            time_series.append((bi, gt, val))

        if len(time_series) < 20:
            continue

        vals = [v for _, _, v in time_series]
        unique = sorted(set(vals))
        n_unique = len(unique)

        # Skip if all same or too many unique (noise)
        if n_unique < 3:
            continue

        # ── Level pattern matching ──
        # Level has ~18 unique values with step-function behavior
        if 5 <= n_unique <= 25:
            # Build "segments" — runs of constant value
            segments = []
            seg_start = 0
            for i in range(1, len(vals)):
                if vals[i] != vals[seg_start]:
                    segments.append((seg_start, i - 1, vals[seg_start]))
                    seg_start = i
            segments.append((seg_start, len(vals) - 1, vals[seg_start]))

            # Map each encrypted value to what oracle level was at that time
            val_to_levels = defaultdict(list)
            for bi, gt, val in time_series:
                closest = min(oracle_timeline, key=lambda o: abs(o['time'] - gt))
                val_to_levels[val].append(closest['level'])

            # Check if each encrypted value maps to exactly ONE oracle level
            one_to_one = 0
            total_mappings = 0
            for val, levels in val_to_levels.items():
                if len(levels) >= 2:  # enough samples
                    total_mappings += 1
                    # Check if >80% of occurrences map to the same level
                    from collections import Counter
                    mc = Counter(levels).most_common(1)[0]
                    if mc[1] >= len(levels) * 0.8:
                        one_to_one += 1

            if total_mappings >= 5 and one_to_one >= total_mappings * 0.7:
                # Build the mapping
                val_to_level = {}
                for val, levels in val_to_levels.items():
                    if len(levels) >= 2:
                        mc = Counter(levels).most_common(1)[0][0]
                        val_to_level[val] = mc

                # Validate: check if the mapped levels form a valid progression
                mapped = sorted(set(val_to_level.values()))
                if len(mapped) >= 5 and mapped[0] <= 3 and mapped[-1] >= 15:
                    # Score: what % of blocks have correct level via this mapping?
                    correct = 0
                    total = 0
                    for bi, gt, val in time_series:
                        if val in val_to_level:
                            closest = min(oracle_timeline, key=lambda o: abs(o['time'] - gt))
                            if val_to_level[val] == closest['level']:
                                correct += 1
                            total += 1

                    accuracy = correct / max(total, 1)
                    best_level_matches.append((off, accuracy, n_unique, len(segments),
                                               one_to_one, total_mappings, val_to_level))

        # ── CS pattern matching ──
        # CS has many unique values, monotonically increasing
        if n_unique >= 30:
            # Check monotonicity
            increases = sum(1 for i in range(1, len(vals)) if vals[i] > vals[i-1])
            decreases = sum(1 for i in range(1, len(vals)) if vals[i] < vals[i-1])
            same = sum(1 for i in range(1, len(vals)) if vals[i] == vals[i-1])

            # CS should be mostly increasing with some same
            if increases > decreases * 2 and increases > len(vals) * 0.3:
                best_cs_matches.append((off, n_unique, increases, decreases, same))

    # Report level matches
    best_level_matches.sort(key=lambda x: -x[1])
    print(f"\n── Level pattern matches (sorted by accuracy) ──")
    for off, acc, n_uniq, n_seg, oto, total, mapping in best_level_matches[:10]:
        print(f"\n  +0x{off:03x}: accuracy={acc:.1%}, {n_uniq} unique vals, {n_seg} segments, "
              f"{oto}/{total} 1:1 mappings")
        # Show the mapping
        for val, level in sorted(mapping.items(), key=lambda x: x[1]):
            count = sum(1 for _, _, v in time_series if v == val)
            print(f"    encrypted 0x{val:08x} → level {level:2d} ({count} blocks)")
        # Show sample validation
        for si in [0, len(time_series)//4, len(time_series)//2, 3*len(time_series)//4, len(time_series)-1]:
            bi, gt, val = time_series[si]
            closest = min(oracle_timeline, key=lambda o: abs(o['time'] - gt))
            mapped_level = mapping.get(val, '?')
            match = '✓' if mapped_level == closest['level'] else '✗'
            print(f"    block {bi:3d} (~{gt:6.0f}s): enc=0x{val:08x} → L={mapped_level}, "
                  f"oracle L={closest['level']} {match}")

    # Report CS matches
    best_cs_matches.sort(key=lambda x: -x[2])
    print(f"\n── CS pattern matches (by increasing count) ──")
    for off, n_uniq, inc, dec, same in best_cs_matches[:5]:
        print(f"  +0x{off:03x}: {n_uniq} unique, {inc} inc / {dec} dec / {same} same")

    return best_level_matches, best_cs_matches


def per_tag_pattern_match(emu, rofl_path, pid, entity=0x400000b2):
    """Decode ALL sub-messages, group by field tag, then pattern-match against oracle.

    Each sub-message has a field tag (read by SKIP varint at struct+0x0C).
    By grouping encrypted struct values by field tag, we get consistent
    time series that can be correlated with oracle.
    """
    oracle = json.load(open('/tmp/oracle_data.json'))
    game_length = oracle['game_length']

    oracle_timeline = []
    for s in oracle['snapshots']:
        g = s['players'].get('Garen', {})
        oracle_timeline.append({
            'time': s['actual_time'],
            'level': g.get('level', 0),
            'cs': g.get('scores', {}).get('creepScore', 0),
        })

    blocks = parse_rofl_blocks(rofl_path, pid_filter=pid, param_filter={entity})
    n_blocks = len(blocks)
    if not blocks:
        print(f"No blocks for PID {pid}")
        return
    tpb = game_length / n_blocks

    pid_info = emu.pid_map.get(str(pid))
    if not pid_info:
        return
    deser_addr = BASE_ADDR + pid_info['deser']

    print(f"PID {pid}: {n_blocks} blocks, {tpb:.1f}s/block")
    print(f"Decoding all sub-messages individually...")

    # (field_tag, struct_offset) → [(block_idx, game_time, encrypted_u32)]
    tag_offset_series = defaultdict(list)
    # Track how many sub-messages per block and which tags appear
    tag_counts = defaultdict(int)

    for bi in range(n_blocks):
        block = blocks[bi]
        payload = block['payload']
        gt = bi * tpb

        pay_addr = SCRATCH_BASE + 0x100
        pp_addr = SCRATCH_BASE + 0x200
        pay_end = pay_addr + len(payload)
        emu.mu.mem_write(pay_addr, payload + b'\x00' * 128)

        current_pos = pay_addr
        for iteration in range(20):
            if current_pos >= pay_end:
                break

            # Reset struct for each sub-message
            emu._alloc_cursor = 0
            emu._alloc_regions = []
            emu._run_constructor(pid)
            clean_struct = bytes(emu.mu.mem_read(HEAP_BASE, 0x100))

            emu.mu.mem_write(pp_addr, struct.pack('<Q', current_pos))
            emu.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
            emu.mu.reg_write(UC_X86_REG_RDX, pp_addr)
            emu.mu.reg_write(UC_X86_REG_R8, pay_end)
            emu.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
            rsp = emu.mu.reg_read(UC_X86_REG_RSP) - 8
            emu.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
            emu.mu.reg_write(UC_X86_REG_RSP, rsp)

            try:
                emu.mu.emu_start(deser_addr, STOP_ADDR, timeout=10_000_000)
            except:
                pass

            new_pos = struct.unpack('<Q', emu.mu.mem_read(pp_addr, 8))[0]
            if new_pos <= current_pos:
                break

            # Read field tag from struct+0x0C
            tag_raw = struct.unpack('<I', emu.mu.mem_read(HEAP_BASE + 0xC, 4))[0]
            tag = tag_raw  # keep as-is for consistency

            tag_counts[tag] += 1

            # Read changed struct fields
            final_struct = bytes(emu.mu.mem_read(HEAP_BASE, 0x100))
            for off in range(0x10, 0x100, 4):
                c = int.from_bytes(clean_struct[off:off+4], 'little')
                f = int.from_bytes(final_struct[off:off+4], 'little')
                if f != c and f != 0:
                    tag_offset_series[(tag, off)].append((bi, gt, f))

            # Also check 2-byte fields
            for off in range(0x10, 0x100, 2):
                c = int.from_bytes(clean_struct[off:off+2], 'little')
                f = int.from_bytes(final_struct[off:off+2], 'little')
                if f != c and f != 0:
                    tag_offset_series[(tag, off + 0x1000)].append((bi, gt, f))  # +0x1000 = u16 flag

            current_pos = new_pos

        if bi % 100 == 0:
            print(f"  Block {bi}/{n_blocks}...")

    print(f"\n  {len(tag_counts)} unique field tags, "
          f"{len(tag_offset_series)} unique (tag, offset) combos")

    # Show top tags by frequency
    print(f"\n  Top field tags:")
    for tag, cnt in sorted(tag_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"    tag=0x{tag:08x}: {cnt} sub-messages")

    # ── Pattern matching against oracle ──
    print(f"\n{'='*70}")
    print(f"LEVEL PATTERN MATCHING")
    print(f"{'='*70}")

    from collections import Counter

    best_matches = []

    for (tag, off_raw), series in tag_offset_series.items():
        if len(series) < 20:
            continue

        is_u16 = off_raw >= 0x1000
        off = off_raw - 0x1000 if is_u16 else off_raw
        width = 'u16' if is_u16 else 'u32'

        vals = [v for _, _, v in series]
        unique = sorted(set(vals))
        n_unique = len(unique)

        # Level check: 5-25 unique values, step-function pattern
        if 5 <= n_unique <= 25:
            # Map each encrypted value to oracle level at that time
            val_to_levels = defaultdict(list)
            for bi, gt, val in series:
                closest = min(oracle_timeline, key=lambda o: abs(o['time'] - gt))
                val_to_levels[val].append(closest['level'])

            # Check 1:1 mapping quality
            val_to_level = {}
            good_mappings = 0
            total_mappings = 0
            for val, levels in val_to_levels.items():
                if len(levels) >= 2:
                    total_mappings += 1
                    mc = Counter(levels).most_common(1)[0]
                    purity = mc[1] / len(levels)
                    if purity >= 0.7:
                        val_to_level[val] = mc[0]
                        good_mappings += 1

            if total_mappings < 5 or good_mappings < total_mappings * 0.5:
                continue

            # Check if mapped levels form valid progression (1-18 range, most present)
            mapped_levels = sorted(set(val_to_level.values()))
            if len(mapped_levels) < 5 or mapped_levels[0] > 5 or mapped_levels[-1] < 12:
                continue

            # Score accuracy
            correct = 0
            total = 0
            for bi, gt, val in series:
                if val in val_to_level:
                    closest = min(oracle_timeline, key=lambda o: abs(o['time'] - gt))
                    if val_to_level[val] == closest['level']:
                        correct += 1
                    total += 1

            accuracy = correct / max(total, 1)
            best_matches.append((tag, off, width, accuracy, n_unique,
                                 good_mappings, total_mappings, val_to_level, series))

    best_matches.sort(key=lambda x: -x[3])
    for tag, off, width, acc, n_uniq, good, total, mapping, series in best_matches[:5]:
        print(f"\n  tag=0x{tag:08x} +0x{off:03x} [{width}]: "
              f"accuracy={acc:.1%}, {n_uniq} unique, {good}/{total} clean mappings")
        for val, level in sorted(mapping.items(), key=lambda x: x[1]):
            cnt = sum(1 for _, _, v in series if v == val)
            print(f"    enc=0x{val:08x} → level {level:2d} ({cnt} occurrences)")
        # Validate
        print(f"  Validation samples:")
        for si in [0, len(series)//4, len(series)//2, 3*len(series)//4, len(series)-1]:
            bi, gt, val = series[si]
            closest = min(oracle_timeline, key=lambda o: abs(o['time'] - gt))
            mapped = mapping.get(val, '?')
            match = '✓' if mapped == closest['level'] else '✗'
            print(f"    block {bi:3d} (~{gt:6.0f}s): enc=0x{val:08x} → L={mapped}, "
                  f"oracle L={closest['level']} {match}")

    if not best_matches:
        print("  No level pattern matches found.")

    # ── Also look for fields where encrypted values simply CHANGE with level ──
    # Even if we can't build a 1:1 mapping, check if value changes coincide
    # with oracle level changes
    print(f"\n{'='*70}")
    print(f"CHANGE-POINT CORRELATION")
    print(f"{'='*70}")

    # Build oracle level change blocks
    oracle_level_changes = set()
    prev_level = 0
    for bi in range(n_blocks):
        gt = bi * tpb
        closest = min(oracle_timeline, key=lambda o: abs(o['time'] - gt))
        if closest['level'] != prev_level:
            oracle_level_changes.add(bi)
            prev_level = closest['level']

    print(f"  Oracle level changes at {len(oracle_level_changes)} block boundaries")

    change_correlations = []
    for (tag, off_raw), series in tag_offset_series.items():
        if len(series) < 50:
            continue

        is_u16 = off_raw >= 0x1000
        off = off_raw - 0x1000 if is_u16 else off_raw
        width = 'u16' if is_u16 else 'u32'

        # Find blocks where this value changes
        prev_val = None
        value_changes = set()
        for bi, gt, val in series:
            if prev_val is not None and val != prev_val:
                value_changes.add(bi)
            prev_val = val

        if len(value_changes) < 5:
            continue

        # Compute overlap with oracle level changes
        overlap = len(value_changes & oracle_level_changes)
        # How many oracle changes are captured?
        recall = overlap / max(len(oracle_level_changes), 1)
        # How many of our changes are actual level changes?
        precision = overlap / max(len(value_changes), 1)

        if recall > 0.3 and precision > 0.1:
            change_correlations.append((tag, off, width, recall, precision,
                                        len(value_changes), overlap, series))

    change_correlations.sort(key=lambda x: -(x[3] * x[4]))
    for tag, off, width, recall, precision, n_changes, overlap, series in change_correlations[:10]:
        print(f"\n  tag=0x{tag:08x} +0x{off:03x} [{width}]: "
              f"recall={recall:.1%}, precision={precision:.1%}, "
              f"{n_changes} changes, {overlap} overlap with level")

    return best_matches, change_correlations


def verify_cipher_scope(emu, rofl_path, pid, entity=0x400000b2):
    """Verify theory: cipher only overwrites byte 0 of each f32 field.

    If true, bytes 1-3 of the encrypted struct are plaintext,
    and the f32 values are approximately correct (error < 0.01%).
    """
    from unicorn import UC_HOOK_MEM_WRITE

    blocks = parse_rofl_blocks(rofl_path, pid_filter=pid, param_filter={entity})
    n_blocks = len(blocks)
    if not blocks:
        return

    pid_info = emu.pid_map.get(str(pid))
    deser_addr = BASE_ADDR + pid_info['deser']
    tpb = 2532.0 / n_blocks

    print(f"PID {pid}: {n_blocks} blocks, verifying cipher scope...")

    # For a few sample blocks, capture the COMPLETE write sequence to key offsets
    # Track: for each aligned-4 offset, which BYTE positions get 1B cipher writes?
    # cipher_byte_positions[offset] = set of byte positions (0-3) that get cipher writes
    cipher_byte_positions = defaultdict(set)
    # Also: for each cipher write, record (offset, byte_pos, plaintext_byte, cipher_byte)
    cipher_pairs = defaultdict(list)

    sample_blocks = list(range(min(50, n_blocks)))

    for bi in sample_blocks:
        block = blocks[bi]
        payload = block['payload']

        emu._alloc_cursor = 0
        emu._alloc_regions = []
        emu._run_constructor(pid)

        pay_addr = SCRATCH_BASE + 0x100
        pp_addr = SCRATCH_BASE + 0x200
        pay_end = pay_addr + len(payload)
        emu.mu.mem_write(pay_addr, payload + b'\x00' * 128)

        # Track: last_4B_write[offset] = value (the bit reader's 4B write)
        last_4B_write = {}
        # cipher_1B_writes[offset_byte] = value
        write_log = []

        def write_hook(mu, access, address, size, value, ud):
            val = value & ((1 << (size * 8)) - 1)
            hoff = address - HEAP_BASE
            if 0x10 <= hoff < 0x200:
                rip = mu.reg_read(UC_X86_REG_RIP)
                rva = rip - BASE_ADDR
                write_log.append((hoff, size, val, rva))
            return True

        current_pos = pay_addr
        for iteration in range(20):
            if current_pos >= pay_end:
                break
            write_log.clear()
            last_4B_write.clear()

            emu.mu.mem_write(pp_addr, struct.pack('<Q', current_pos))
            emu.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
            emu.mu.reg_write(UC_X86_REG_RDX, pp_addr)
            emu.mu.reg_write(UC_X86_REG_R8, pay_end)
            emu.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
            rsp = emu.mu.reg_read(UC_X86_REG_RSP) - 8
            emu.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
            emu.mu.reg_write(UC_X86_REG_RSP, rsp)

            h = emu.mu.hook_add(UC_HOOK_MEM_WRITE, write_hook,
                                begin=HEAP_BASE, end=HEAP_BASE + 0x200)
            try:
                emu.mu.emu_start(deser_addr, STOP_ADDR, timeout=10_000_000)
            except:
                pass
            emu.mu.hook_del(h)

            new_pos = struct.unpack('<Q', emu.mu.mem_read(pp_addr, 8))[0]
            if new_pos <= current_pos:
                break

            # Analyze write sequence: find 4B writes followed by 1B overwrites
            for i, (hoff, sz, val, rva) in enumerate(write_log):
                if sz == 4 and 0xfe0000 <= rva <= 0xff1000:
                    # Bit reader 4B write — store as potential plaintext
                    last_4B_write[hoff] = val
                elif sz == 1:
                    # 1B write — check if it overwrites a recent 4B write
                    off4 = hoff & ~3
                    byte_pos = hoff & 3
                    if off4 in last_4B_write:
                        pt_u32 = last_4B_write[off4]
                        # Plaintext byte at this position
                        pt_byte = (pt_u32 >> (byte_pos * 8)) & 0xFF
                        ct_byte = val
                        cipher_byte_positions[off4].add(byte_pos)
                        cipher_pairs[(off4, byte_pos)].append((bi, pt_byte, ct_byte, rva))

            current_pos = new_pos

    # Report findings
    print(f"\n{'='*70}")
    print(f"CIPHER SCOPE ANALYSIS")
    print(f"{'='*70}")

    print(f"\nByte positions overwritten by cipher, per struct offset:")
    for off in sorted(cipher_byte_positions.keys()):
        positions = sorted(cipher_byte_positions[off])
        n_pairs = sum(len(cipher_pairs.get((off, bp), [])) for bp in positions)
        print(f"  +0x{off:03x}: cipher overwrites byte(s) {positions} ({n_pairs} samples)")

        # Show cipher transformation for each byte position
        for bp in positions:
            pairs = cipher_pairs.get((off, bp), [])
            if not pairs:
                continue
            # Build lookup: plaintext → ciphertext
            pt_to_ct = {}
            for bi, pt, ct, rva in pairs:
                if pt in pt_to_ct and pt_to_ct[pt] != ct:
                    # Different ciphers for same plaintext byte!
                    pass  # expected if different handlers
                pt_to_ct[pt] = ct

            # Check if it's consistent (same pt → same ct)
            consistent = len(set((pt, ct) for _, pt, ct, _ in pairs)) == len(pt_to_ct)
            rvas = set(rva for _, _, _, rva in pairs)
            print(f"    byte {bp}: {len(pairs)} samples, {len(pt_to_ct)} unique pt values, "
                  f"consistent={consistent}, handlers={len(rvas)}")
            if len(pairs) <= 10:
                for bi, pt, ct, rva in pairs[:5]:
                    print(f"      block {bi}: 0x{pt:02x} → 0x{ct:02x} (RVA 0x{rva:x})")

    # KEY QUESTION: are bytes 1-3 left intact?
    bytes_encrypted = set()
    for off, positions in cipher_byte_positions.items():
        for bp in positions:
            bytes_encrypted.add(bp)

    print(f"\n  Encrypted byte positions (across all offsets): {sorted(bytes_encrypted)}")
    if bytes_encrypted == {0}:
        print("  *** CONFIRMED: Only byte 0 is encrypted! Bytes 1-3 are plaintext ***")
    elif len(bytes_encrypted) <= 2:
        print(f"  Partial encryption: only bytes {sorted(bytes_encrypted)} are modified")
    else:
        print(f"  Full encryption: {len(bytes_encrypted)} byte positions affected")

    # Now test: read final struct f32 values and see if they're approximately correct
    if 0 in bytes_encrypted and len(bytes_encrypted) <= 2:
        print(f"\n{'='*70}")
        print(f"APPROXIMATE F32 EXTRACTION (ignoring encrypted bytes)")
        print(f"{'='*70}")

        oracle = json.load(open('/tmp/oracle_data.json'))
        oracle_tl = []
        for s in oracle['snapshots']:
            g = s['players'].get('Garen', {})
            oracle_tl.append({
                'time': s['actual_time'],
                'level': g.get('level', 0),
                'cs': g.get('scores', {}).get('creepScore', 0),
            })

        # For select blocks, decode and read struct f32 values (ignoring byte 0)
        # This gives approximately correct values with < 0.01% error
        offsets_with_data = sorted(cipher_byte_positions.keys())

        # offset → [(block, game_time, raw_f32, approx_f32)]
        approx_series = defaultdict(list)

        for bi in range(n_blocks):
            block = blocks[bi]
            payload = block['payload']
            gt = bi * tpb

            results = emu.decode_payload(pid, payload, verbose=False)
            if not results:
                continue

            final = bytes(emu.mu.mem_read(HEAP_BASE, 0x200))
            for off in offsets_with_data:
                u32 = int.from_bytes(final[off:off+4], 'little')
                if u32 == 0:
                    continue
                # Zero out encrypted byte(s) for approximate reading
                u32_approx = u32
                for bp in cipher_byte_positions[off]:
                    u32_approx &= ~(0xFF << (bp * 8))
                fval = struct.unpack('<f', struct.pack('<I', u32_approx))[0]
                if fval == fval and abs(fval) < 20000:  # not NaN
                    approx_series[off].append((bi, gt, fval, u32))

            if bi % 100 == 0:
                print(f"  Block {bi}/{n_blocks}...")

        # Analyze approximate f32 series
        print(f"\nApproximate f32 values per offset:")
        for off in sorted(approx_series.keys()):
            series = approx_series[off]
            if len(series) < 10:
                continue
            vals = [v for _, _, v, _ in series]
            unique = len(set(round(v, 0) for v in vals))
            if unique < 2:
                continue

            # Classification
            label = ""
            # Coord trajectory
            coord_vals = [v for v in vals if 500 < abs(v) < 15000]
            if len(coord_vals) > len(vals) * 0.3:
                diffs = [abs(coord_vals[i] - coord_vals[i-1]) for i in range(1, len(coord_vals))]
                if diffs:
                    med_diff = sorted(diffs)[len(diffs)//2]
                    val_range = max(coord_vals) - min(coord_vals)
                    if val_range > 500:
                        label = f"COORD (range={val_range:.0f}, med_step={med_diff:.0f})"
                        if med_diff < val_range * 0.1:
                            label = f"*** SMOOTH COORD *** (range={val_range:.0f}, step={med_diff:.0f})"

            # Level: integers 1-20
            int_vals = [round(v) for v in vals if 0 < v < 21 and abs(v - round(v)) < 0.5]
            if len(int_vals) > len(vals) * 0.5:
                if max(int_vals) <= 20 and min(int_vals) >= 1:
                    label = f"LEVEL? ({sorted(set(int_vals))})"

            if not label and unique < 5:
                continue

            print(f"\n  +0x{off:03x}: {len(series)} pts, {unique} unique, "
                  f"[{min(vals):.1f}, {max(vals):.1f}]  {label}")
            # Samples
            for si in [0, len(series)//4, len(series)//2, 3*len(series)//4, len(series)-1]:
                bi, gt, fval, raw = series[si]
                closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                print(f"    block {bi:3d} (~{gt:6.0f}s): f32≈{fval:12.2f} (raw=0x{raw:08x})  "
                      f"oracle: L={closest['level']}, CS={closest['cs']}")

    return cipher_byte_positions, cipher_pairs


def build_sbox_and_decrypt(emu, rofl_path, pid, entity=0x400000b2):
    """Build cipher S-box inverses from (plaintext, ciphertext) pairs captured
    during emulation, then decrypt all blocks.

    For each (handler_RVA, byte_position), we build a 256-entry lookup table.
    """
    from unicorn import UC_HOOK_MEM_WRITE

    oracle = json.load(open('/tmp/oracle_data.json'))
    game_length = oracle['game_length']
    oracle_tl = []
    for s in oracle['snapshots']:
        g = s['players'].get('Garen', {})
        oracle_tl.append({
            'time': s['actual_time'],
            'level': g.get('level', 0),
            'cs': g.get('scores', {}).get('creepScore', 0),
        })

    blocks = parse_rofl_blocks(rofl_path, pid_filter=pid, param_filter={entity})
    n_blocks = len(blocks)
    if not blocks:
        return
    tpb = game_length / n_blocks

    pid_info = emu.pid_map.get(str(pid))
    deser_addr = BASE_ADDR + pid_info['deser']

    print(f"PID {pid}: {n_blocks} blocks, building cipher S-boxes from ALL blocks...")

    # Phase 1: Collect (plaintext_byte, ciphertext_byte) pairs
    # Key: (handler_rva, byte_pos) → {pt: ct}
    sbox_data = defaultdict(dict)
    # Also track: for each block, the handler and encrypted struct per offset
    block_results = []  # [(block_idx, gt, {off4: (handler_rva, enc_u32)})]

    for bi in range(n_blocks):
        block = blocks[bi]
        payload = block['payload']
        gt = bi * tpb

        pay_addr = SCRATCH_BASE + 0x100
        pp_addr = SCRATCH_BASE + 0x200
        pay_end = pay_addr + len(payload)
        emu.mu.mem_write(pay_addr, payload + b'\x00' * 128)

        # Per sub-message, track write sequences
        all_sub_results = {}  # off4 → (handler_rva, enc_u32, pt_u32)

        current_pos = pay_addr
        for iteration in range(20):
            if current_pos >= pay_end:
                break

            emu._alloc_cursor = 0
            emu._alloc_regions = []
            emu._run_constructor(pid)

            write_log = []

            def write_hook(mu, access, address, size, value, ud):
                val = value & ((1 << (size * 8)) - 1)
                hoff = address - HEAP_BASE
                if 0x10 <= hoff < 0x200:
                    rip = mu.reg_read(UC_X86_REG_RIP)
                    rva = rip - BASE_ADDR
                    write_log.append((hoff, size, val, rva))
                return True

            emu.mu.mem_write(pp_addr, struct.pack('<Q', current_pos))
            emu.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
            emu.mu.reg_write(UC_X86_REG_RDX, pp_addr)
            emu.mu.reg_write(UC_X86_REG_R8, pay_end)
            emu.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
            rsp = emu.mu.reg_read(UC_X86_REG_RSP) - 8
            emu.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
            emu.mu.reg_write(UC_X86_REG_RSP, rsp)

            h = emu.mu.hook_add(UC_HOOK_MEM_WRITE, write_hook,
                                begin=HEAP_BASE, end=HEAP_BASE + 0x200)
            try:
                emu.mu.emu_start(deser_addr, STOP_ADDR, timeout=10_000_000)
            except:
                pass
            emu.mu.hook_del(h)

            new_pos = struct.unpack('<Q', emu.mu.mem_read(pp_addr, 8))[0]
            if new_pos <= current_pos:
                break

            # Extract (plaintext, ciphertext) pairs from write log
            # Strategy: for each aligned-4 offset, find the LAST 4B write
            # followed by 1B writes to the same 4-byte region
            last_4B = {}  # off4 → (val, rva)
            cipher_bytes = defaultdict(dict)  # off4 → {byte_pos: (ct, rva)}

            for hoff, sz, val, rva in write_log:
                off4 = hoff & ~3
                if sz >= 4:
                    last_4B[off4] = (val, rva)
                    # Reset cipher bytes when a new 4B write happens
                    cipher_bytes[off4] = {}
                elif sz == 1:
                    bp = hoff & 3
                    cipher_bytes[off4][bp] = (val, rva)

            # For offsets that have both a 4B write AND 4 cipher bytes,
            # extract the S-box entries
            for off4 in last_4B:
                pt_u32, pt_rva = last_4B[off4]
                cb = cipher_bytes.get(off4, {})
                if len(cb) < 1:
                    continue

                # Determine the handler RVA from the first cipher byte
                handler_rva = None
                for bp in sorted(cb.keys()):
                    handler_rva = cb[bp][1]
                    break

                # Build encrypted u32 from cipher bytes overlaid on plaintext
                enc_u32 = pt_u32
                for bp in range(4):
                    if bp in cb:
                        ct, rva = cb[bp]
                        enc_u32 = (enc_u32 & ~(0xFF << (bp * 8))) | (ct << (bp * 8))
                        # S-box entry: plaintext byte → ciphertext byte
                        pt_byte = (pt_u32 >> (bp * 8)) & 0xFF
                        sbox_data[(handler_rva, bp)][pt_byte] = ct

                all_sub_results[off4] = (handler_rva, enc_u32, pt_u32)

            current_pos = new_pos

        block_results.append((bi, gt, all_sub_results))

        if bi % 100 == 0:
            print(f"  Block {bi}/{n_blocks}...")

    # Phase 2: Build S-box inverses
    print(f"\n{'='*70}")
    print(f"S-BOX ANALYSIS")
    print(f"{'='*70}")

    # handler_rva → {byte_pos: {ct: pt}}
    inverse_sboxes = {}
    handler_coverage = defaultdict(lambda: [0, 0, 0, 0])  # rva → [coverage per byte_pos]

    unique_handlers = set(rva for (rva, bp) in sbox_data.keys())
    print(f"\nFound {len(unique_handlers)} unique cipher handlers")

    for handler_rva in sorted(unique_handlers):
        inverse_sboxes[handler_rva] = {}
        for bp in range(4):
            fwd = sbox_data.get((handler_rva, bp), {})
            if fwd:
                inv = {ct: pt for pt, ct in fwd.items()}
                inverse_sboxes[handler_rva][bp] = inv
                handler_coverage[handler_rva][bp] = len(fwd)

        covs = handler_coverage[handler_rva]
        print(f"  Handler 0x{handler_rva:07x}: coverage = {covs} / 256 per byte")

    # Phase 3: Decrypt all blocks using S-box inverses
    print(f"\n{'='*70}")
    print(f"DECRYPTION")
    print(f"{'='*70}")

    # offset → [(block_idx, game_time, decrypted_f32)]
    decrypted_series = defaultdict(list)

    for bi, gt, sub_results in block_results:
        for off4, (handler_rva, enc_u32, pt_u32) in sub_results.items():
            # We already have the plaintext! It's pt_u32 from the bit reader write.
            # But let's also verify by inverting the cipher.
            fval = struct.unpack('<f', struct.pack('<I', pt_u32))[0]
            if fval == fval:  # not NaN
                decrypted_series[off4].append((bi, gt, fval, pt_u32))

    # Analyze decrypted time series
    print(f"\n{len(decrypted_series)} offsets with decrypted data")

    for off in sorted(decrypted_series.keys()):
        series = decrypted_series[off]
        if len(series) < 10:
            continue
        vals = [v for _, _, v, _ in series]
        unique = len(set(round(v, 1) for v in vals))
        if unique < 3:
            continue

        # Classification
        label = ""

        # Coordinate: 100-15000
        coord_vals = [v for v in vals if 100 < abs(v) < 15500 and v == v]
        if len(coord_vals) > len(vals) * 0.3:
            diffs = [abs(coord_vals[i] - coord_vals[i-1]) for i in range(1, len(coord_vals))]
            if diffs:
                med = sorted(diffs)[len(diffs)//2]
                rng = max(coord_vals) - min(coord_vals)
                if rng > 500 and med < rng * 0.15:
                    label = f"*** SMOOTH COORD *** (range={rng:.0f}, step={med:.0f})"
                elif rng > 200:
                    label = f"COORD? (range={rng:.0f})"

        # Level: f32 integers 1-20
        int_vals = [round(v) for v in vals if 0.5 < v < 20.5 and abs(v - round(v)) < 0.01]
        if len(int_vals) > len(vals) * 0.3 and not label:
            unique_ints = sorted(set(int_vals))
            if all(1 <= i <= 20 for i in unique_ints) and len(unique_ints) >= 3:
                # Check against oracle
                correct = 0
                for bi, gt, fval, _ in series:
                    if 0.5 < fval < 20.5 and abs(fval - round(fval)) < 0.01:
                        closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                        if round(fval) == closest['level']:
                            correct += 1
                label = f"*** LEVEL *** (match={correct}/{len(series)}, vals={unique_ints})"

        # CS: 0-500, increasing
        cs_vals = [v for v in vals if 0 <= v <= 500 and abs(v - round(v)) < 0.5]
        if len(cs_vals) > len(vals) * 0.3 and max(cs_vals) > 30 and not label:
            increases = sum(1 for i in range(1, len(cs_vals)) if cs_vals[i] > cs_vals[i-1])
            if increases > len(cs_vals) * 0.3:
                label = f"CS? (max={max(cs_vals):.0f}, inc={increases})"

        # Skip low-interest
        if not label:
            if all(abs(v) < 3 or abs(v) > 50000 for v in vals):
                continue
            if unique < 10:
                continue

        print(f"\n  +0x{off:03x}: {len(series)} pts, {unique} unique, "
              f"[{min(vals):.1f}, {max(vals):.1f}]  {label}")
        # Samples
        for si in [0, len(series)//4, len(series)//2, 3*len(series)//4, len(series)-1]:
            bi, gt, fval, raw = series[si]
            closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
            print(f"    block {bi:3d} (~{gt:6.0f}s): f32={fval:12.4f} (0x{raw:08x})  "
                  f"oracle: L={closest['level']}, CS={closest['cs']}")


def targeted_bswap_extraction(emu, rofl_path, pid, entity=0x400000b2):
    """Extract plaintext by capturing ONLY the bswap write (specific RVA).

    The deserializer bit reader accumulates 4 bytes big-endian, then does a
    BSWAP to little-endian. The bswap write is from a specific RVA per offset.
    We first discover the bswap RVA by looking at byte-reversal patterns,
    then track it across all blocks.
    """
    from unicorn import UC_HOOK_MEM_WRITE

    oracle = json.load(open('/tmp/oracle_data.json'))
    game_length = oracle['game_length']

    blocks = parse_rofl_blocks(rofl_path, pid_filter=pid, param_filter={entity})
    n_blocks = len(blocks)
    if not blocks:
        return
    tpb = game_length / n_blocks

    pid_info = emu.pid_map.get(str(pid))
    deser_addr = BASE_ADDR + pid_info['deser']

    print(f"PID {pid}: {n_blocks} blocks")

    # Phase 1: Discover bswap RVAs from first 5 blocks
    # The bswap write immediately follows the last accumulation write,
    # and its value is the byte-reversal of the previous write's value.
    print("Phase 1: Discovering bswap RVAs...")

    bswap_rvas = {}  # offset → set of bswap RVAs
    all_write_logs = []

    for bi in range(min(5, n_blocks)):
        block = blocks[bi]
        payload = block['payload']

        pay_addr = SCRATCH_BASE + 0x100
        pp_addr = SCRATCH_BASE + 0x200
        pay_end = pay_addr + len(payload)
        emu.mu.mem_write(pay_addr, payload + b'\x00' * 128)

        current_pos = pay_addr
        for iteration in range(20):
            if current_pos >= pay_end:
                break

            emu._alloc_cursor = 0
            emu._alloc_regions = []
            emu._run_constructor(pid)

            writes = []

            def write_hook(mu, access, address, size, value, ud):
                val = value & ((1 << (size * 8)) - 1)
                hoff = address - HEAP_BASE
                if 0x10 <= hoff < 0x200 and size == 4:
                    rip = mu.reg_read(UC_X86_REG_RIP)
                    rva = rip - BASE_ADDR
                    if 0xfe0000 <= rva <= 0xff1000:
                        writes.append((hoff, val, rva))
                return True

            emu.mu.mem_write(pp_addr, struct.pack('<Q', current_pos))
            emu.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
            emu.mu.reg_write(UC_X86_REG_RDX, pp_addr)
            emu.mu.reg_write(UC_X86_REG_R8, pay_end)
            emu.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
            rsp = emu.mu.reg_read(UC_X86_REG_RSP) - 8
            emu.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
            emu.mu.reg_write(UC_X86_REG_RSP, rsp)

            h = emu.mu.hook_add(UC_HOOK_MEM_WRITE, write_hook,
                                begin=HEAP_BASE, end=HEAP_BASE + 0x200)
            try:
                emu.mu.emu_start(deser_addr, STOP_ADDR, timeout=10_000_000)
            except:
                pass
            emu.mu.hook_del(h)

            new_pos = struct.unpack('<Q', emu.mu.mem_read(pp_addr, 8))[0]
            if new_pos <= current_pos:
                break

            # Find bswap writes: value is byte-reversal of previous write to same offset
            prev_at_off = {}
            for hoff, val, rva in writes:
                if hoff in prev_at_off:
                    prev_val = prev_at_off[hoff]
                    # Check byte reversal
                    bswapped = struct.unpack('>I', struct.pack('<I', prev_val))[0]
                    if val == bswapped and val != 0 and val != prev_val:
                        if hoff not in bswap_rvas:
                            bswap_rvas[hoff] = set()
                        bswap_rvas[hoff].add(rva)
                prev_at_off[hoff] = val

            current_pos = new_pos

    print(f"  Found bswap RVAs for {len(bswap_rvas)} offsets:")
    for off in sorted(bswap_rvas.keys()):
        rvas = bswap_rvas[off]
        print(f"    +0x{off:03x}: {', '.join(f'0x{r:x}' for r in sorted(rvas))}")

    if not bswap_rvas:
        print("  No bswap patterns found!")
        return

    # Phase 2: Track bswap values across ALL blocks
    print(f"\nPhase 2: Extracting bswap values across all {n_blocks} blocks...")

    # offset → [(block_idx, game_time, f32, u32)]
    bswap_series = defaultdict(list)

    # Build lookup: offset → set of bswap RVAs
    target_bswap = {}
    for off, rvas in bswap_rvas.items():
        target_bswap[off] = rvas

    for bi in range(n_blocks):
        block = blocks[bi]
        payload = block['payload']
        gt = bi * tpb

        pay_addr = SCRATCH_BASE + 0x100
        pp_addr = SCRATCH_BASE + 0x200
        pay_end = pay_addr + len(payload)
        emu.mu.mem_write(pay_addr, payload + b'\x00' * 128)

        captured = defaultdict(list)  # offset → [f32 values from this block]

        def write_hook_bswap(mu, access, address, size, value, ud):
            if size != 4:
                return True
            val = value & 0xFFFFFFFF
            hoff = address - HEAP_BASE
            if hoff in target_bswap:
                rip = mu.reg_read(UC_X86_REG_RIP)
                rva = rip - BASE_ADDR
                if rva in target_bswap[hoff]:
                    fval = struct.unpack('<f', struct.pack('<I', val))[0]
                    captured[hoff].append((fval, val))
            return True

        current_pos = pay_addr
        for iteration in range(20):
            if current_pos >= pay_end:
                break

            emu._alloc_cursor = 0
            emu._alloc_regions = []
            emu._run_constructor(pid)

            emu.mu.mem_write(pp_addr, struct.pack('<Q', current_pos))
            emu.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
            emu.mu.reg_write(UC_X86_REG_RDX, pp_addr)
            emu.mu.reg_write(UC_X86_REG_R8, pay_end)
            emu.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
            rsp = emu.mu.reg_read(UC_X86_REG_RSP) - 8
            emu.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
            emu.mu.reg_write(UC_X86_REG_RSP, rsp)

            h = emu.mu.hook_add(UC_HOOK_MEM_WRITE, write_hook_bswap,
                                begin=HEAP_BASE, end=HEAP_BASE + 0x200)
            try:
                emu.mu.emu_start(deser_addr, STOP_ADDR, timeout=10_000_000)
            except:
                pass
            emu.mu.hook_del(h)

            new_pos = struct.unpack('<Q', emu.mu.mem_read(pp_addr, 8))[0]
            if new_pos <= current_pos:
                break
            current_pos = new_pos

        # Store LAST bswap value per offset for this block
        for off, vals in captured.items():
            if vals:
                fval, u32 = vals[-1]  # last value
                bswap_series[off].append((bi, gt, fval, u32))

        if bi % 50 == 0:
            print(f"  Block {bi}/{n_blocks}...")

    # Phase 3: Analyze trajectories
    print(f"\n{'='*70}")
    print(f"BSWAP PLAINTEXT ANALYSIS")
    print(f"{'='*70}")

    oracle_tl = []
    for s in oracle['snapshots']:
        g = s['players'].get('Garen', {})
        oracle_tl.append({
            'time': s['actual_time'],
            'level': g.get('level', 0),
            'cs': g.get('scores', {}).get('creepScore', 0),
        })

    for off in sorted(bswap_series.keys()):
        series = bswap_series[off]
        if len(series) < 5:
            continue
        vals = [v for _, _, v, _ in series]
        unique = len(set(round(v, 1) for v in vals if v == v))

        # Filter NaN
        valid = [(bi, gt, v, u) for bi, gt, v, u in series if v == v]
        if len(valid) < 5:
            continue
        vals = [v for _, _, v, _ in valid]

        # Classify
        label = ""
        coord_vals = [v for v in vals if 100 < abs(v) < 15500]
        if len(coord_vals) > len(vals) * 0.3:
            diffs = [abs(coord_vals[i] - coord_vals[i-1]) for i in range(1, len(coord_vals))]
            if diffs:
                med = sorted(diffs)[len(diffs)//2]
                rng = max(coord_vals) - min(coord_vals)
                if rng > 500 and med < rng * 0.15:
                    label = f"*** SMOOTH COORD *** (range={rng:.0f}, med_step={med:.0f})"
                elif rng > 100:
                    label = f"COORD? (range={rng:.0f}, step={med:.0f})"

        # Level
        int_vals = [round(v) for v in vals if 0.5 < v < 20.5 and abs(v - round(v)) < 0.01]
        if len(int_vals) > len(vals) * 0.3 and not label:
            unique_ints = sorted(set(int_vals))
            if all(1 <= i <= 20 for i in unique_ints) and len(unique_ints) >= 3:
                correct = 0
                for bi, gt, fval, _ in valid:
                    if 0.5 < fval < 20.5 and abs(fval - round(fval)) < 0.01:
                        closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                        if round(fval) == closest['level']:
                            correct += 1
                label = f"*** LEVEL *** (match={correct}/{len(valid)}, vals={unique_ints})"

        print(f"\n  +0x{off:03x}: {len(valid)} pts, {unique} unique f32, "
              f"[{min(vals):.1f}, {max(vals):.1f}]  {label}")

        # Samples
        step = max(1, len(valid) // 15)
        for vi in range(0, len(valid), step):
            bi, gt, fval, u32 = valid[vi]
            closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
            print(f"    block {bi:3d} (~{gt:6.0f}s): f32={fval:12.2f}  "
                  f"oracle: L={closest['level']}, CS={closest['cs']}")

    return bswap_series


def decrypt_per_tag_bswap(emu, rofl_path, pid, entity=0x400000b2):
    """Extract plaintext game values by capturing bswap writes PER FIELD TAG.

    Previous approaches failed because different sub-messages write different
    game data to the same struct offset. By grouping bswap plaintext by field
    tag, we get clean time series that can match oracle data.

    For each sub-message:
      1. SKIP varint reads field tag → struct+0x0C
      2. Bit reader accumulates bytes → bswap write (PLAINTEXT f32) to struct
      3. Cipher overwrites with encrypted bytes

    We capture step 2 via write hooks (4B writes from bit reader RVA range),
    and read the field tag from struct+0x0C after decode.
    """
    from unicorn import UC_HOOK_MEM_WRITE

    oracle = json.load(open('/tmp/oracle_data.json'))
    game_length = oracle['game_length']
    oracle_tl = []
    for s in oracle['snapshots']:
        g = s['players'].get('Garen', {})
        oracle_tl.append({
            'time': s['actual_time'],
            'level': g.get('level', 0),
            'cs': g.get('scores', {}).get('creepScore', 0),
        })

    blocks = parse_rofl_blocks(rofl_path, pid_filter=pid, param_filter={entity})
    n_blocks = len(blocks)
    if not blocks:
        print(f"No blocks for PID {pid}")
        return
    tpb = game_length / n_blocks

    pid_info = emu.pid_map.get(str(pid))
    if not pid_info:
        return
    deser_addr = BASE_ADDR + pid_info['deser']

    print(f"PID {pid}: {n_blocks} blocks, {tpb:.1f}s/block")
    print(f"Extracting bswap plaintext per field tag...")

    # (field_tag, struct_offset) → [(block_idx, game_time, f32, u32)]
    tag_series = defaultdict(list)
    # Also collect: struct_offset → [(block_idx, game_time, f32, u32, field_tag)]
    # for cross-tag analysis
    offset_all = defaultdict(list)
    tag_freq = defaultdict(int)

    BIT_READER_LO = 0xfe0000
    BIT_READER_HI = 0xff1000
    debug_bswap_count = [0]
    debug_fallback_count = [0]

    for bi in range(n_blocks):
        block = blocks[bi]
        payload = block['payload']
        gt = bi * tpb

        pay_addr = SCRATCH_BASE + 0x100
        pp_addr = SCRATCH_BASE + 0x200
        pay_end = pay_addr + len(payload)
        emu.mu.mem_write(pay_addr, payload + b'\x00' * 128)

        current_pos = pay_addr
        # Shared mutable state for write hook — reset per sub-message
        hook_state = {'prev': {}, 'bswap': {}, 'all_br': defaultdict(list)}

        def write_hook(mu, access, address, size, value, ud):
            val = value & ((1 << (size * 8)) - 1)
            hoff = address - HEAP_BASE
            if 0x10 <= hoff < 0x200 and size == 4:
                rip = mu.reg_read(UC_X86_REG_RIP)
                rva = rip - BASE_ADDR
                if BIT_READER_LO <= rva <= BIT_READER_HI:
                    hook_state['all_br'][hoff].append((val, rva))
                    # Bswap detection: byte-reversal of previous write
                    if hoff in hook_state['prev']:
                        prev_val = hook_state['prev'][hoff]
                        bswapped = struct.unpack('>I', struct.pack('<I', prev_val))[0]
                        if val == bswapped and val != 0 and val != prev_val:
                            hook_state['bswap'][hoff] = (val, rva)
                    hook_state['prev'][hoff] = val
            return True

        for iteration in range(20):
            if current_pos >= pay_end:
                break

            emu._alloc_cursor = 0
            emu._alloc_regions = []
            emu._run_constructor(pid)

            # Reset tracking state for this sub-message
            hook_state['prev'].clear()
            hook_state['bswap'].clear()
            hook_state['all_br'].clear()

            emu.mu.mem_write(pp_addr, struct.pack('<Q', current_pos))
            emu.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
            emu.mu.reg_write(UC_X86_REG_RDX, pp_addr)
            emu.mu.reg_write(UC_X86_REG_R8, pay_end)
            emu.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
            rsp = emu.mu.reg_read(UC_X86_REG_RSP) - 8
            emu.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
            emu.mu.reg_write(UC_X86_REG_RSP, rsp)

            h = emu.mu.hook_add(UC_HOOK_MEM_WRITE, write_hook,
                                begin=HEAP_BASE, end=HEAP_BASE + 0x200)
            try:
                emu.mu.emu_start(deser_addr, STOP_ADDR, timeout=10_000_000)
            except:
                pass
            emu.mu.hook_del(h)

            new_pos = struct.unpack('<Q', emu.mu.mem_read(pp_addr, 8))[0]
            if new_pos <= current_pos:
                break

            # Read field tag
            tag_raw = struct.unpack('<I', emu.mu.mem_read(HEAP_BASE + 0xC, 4))[0]
            tag_freq[tag_raw] += 1

            # Record bswap-confirmed plaintext values
            # Also fall back to last bit reader write if no bswap detected
            bswap = hook_state['bswap']
            all_br = hook_state['all_br']
            for hoff in set(list(bswap.keys()) + list(all_br.keys())):
                if hoff in bswap:
                    val, rva = bswap[hoff]
                    debug_bswap_count[0] += 1
                elif hoff in all_br and len(all_br[hoff]) >= 2:
                    # Fallback: only use if we have multiple writes (last MIGHT be bswap)
                    # But skip single-write offsets as those are just accumulation
                    val, rva = all_br[hoff][-1]
                    debug_fallback_count[0] += 1
                else:
                    continue
                fval = struct.unpack('<f', struct.pack('<I', val))[0]
                if fval != fval:  # NaN
                    continue
                tag_series[(tag_raw, hoff)].append((bi, gt, fval, val))
                offset_all[hoff].append((bi, gt, fval, val, tag_raw))

            current_pos = new_pos

        if bi % 50 == 0:
            print(f"  Block {bi}/{n_blocks}...")

    # ── Analysis ──────────────────────────────────────────────
    n_points = sum(len(s) for s in tag_series.values())
    print(f"\n{'='*70}")
    print(f"RESULTS: {len(tag_series)} unique (tag, offset) combos, {n_points} total data points")
    print(f"         {len(tag_freq)} unique field tags")
    print(f"         bswap_confirmed={debug_bswap_count[0]}, fallback={debug_fallback_count[0]}")
    print(f"{'='*70}")

    # Show tag frequency
    print(f"\nTop field tags by frequency:")
    for tag, cnt in sorted(tag_freq.items(), key=lambda x: -x[1])[:20]:
        print(f"  tag=0x{tag:08x}: {cnt} sub-messages")

    # ── Search for LEVEL ──
    print(f"\n{'='*70}")
    print(f"LEVEL SEARCH")
    print(f"{'='*70}")

    level_candidates = []
    for (tag, off), series in tag_series.items():
        if len(series) < 20:
            continue
        vals_f32 = [v for _, _, v, _ in series]

        # Level as f32: exact integers 1.0-18.0
        int_vals = [(bi, gt, round(v), u) for bi, gt, v, u in series
                    if 0.5 < v < 20.5 and abs(v - round(v)) < 0.01]
        if len(int_vals) < len(series) * 0.5:
            continue
        unique_ints = sorted(set(iv for _, _, iv, _ in int_vals))
        if len(unique_ints) < 3 or not all(1 <= i <= 20 for i in unique_ints):
            continue

        # Check monotonicity (level only increases)
        levels = [iv for _, _, iv, _ in int_vals]
        increases = sum(1 for i in range(1, len(levels)) if levels[i] > levels[i-1])
        decreases = sum(1 for i in range(1, len(levels)) if levels[i] < levels[i-1])
        if decreases > increases:
            continue

        # Match against oracle
        correct = 0
        total = 0
        for bi, gt, lv, _ in int_vals:
            closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
            if lv == closest['level']:
                correct += 1
            total += 1
        accuracy = correct / max(total, 1)

        level_candidates.append((tag, off, accuracy, unique_ints, len(series), int_vals))

    level_candidates.sort(key=lambda x: -x[2])
    for tag, off, acc, uniq, n, int_vals in level_candidates[:10]:
        print(f"\n  tag=0x{tag:08x} +0x{off:03x}: accuracy={acc:.1%}, "
              f"{n} pts, levels={uniq}")
        # Show samples
        step = max(1, len(int_vals) // 8)
        for i in range(0, len(int_vals), step):
            bi, gt, lv, u32 = int_vals[i]
            closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
            match = '✓' if lv == closest['level'] else '✗'
            print(f"    block {bi:3d} (~{gt:6.0f}s): level={lv:2d}  "
                  f"oracle={closest['level']:2d} {match}")

    if not level_candidates:
        print("  No level candidates found.")

    # ── Search for CS ──
    print(f"\n{'='*70}")
    print(f"CS SEARCH")
    print(f"{'='*70}")

    cs_candidates = []
    for (tag, off), series in tag_series.items():
        if len(series) < 20:
            continue
        vals_f32 = [v for _, _, v, _ in series]

        # CS as f32 integer: 0-500, mostly increasing
        cs_vals = [(bi, gt, round(v), u) for bi, gt, v, u in series
                   if 0 <= v <= 500 and abs(v - round(v)) < 0.5]
        if len(cs_vals) < len(series) * 0.3:
            continue
        raw_cs = [c for _, _, c, _ in cs_vals]
        if max(raw_cs) < 30:
            continue

        increases = sum(1 for i in range(1, len(raw_cs)) if raw_cs[i] > raw_cs[i-1])
        if increases < len(raw_cs) * 0.2:
            continue

        # Match against oracle
        correct = 0
        total = 0
        for bi, gt, cs, _ in cs_vals:
            closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
            if abs(cs - closest['cs']) <= 5:
                correct += 1
            total += 1
        accuracy = correct / max(total, 1)

        cs_candidates.append((tag, off, accuracy, max(raw_cs), len(series), cs_vals))

    cs_candidates.sort(key=lambda x: -x[2])
    for tag, off, acc, max_cs, n, cs_vals in cs_candidates[:10]:
        print(f"\n  tag=0x{tag:08x} +0x{off:03x}: accuracy={acc:.1%}, "
              f"{n} pts, max_cs={max_cs}")
        step = max(1, len(cs_vals) // 8)
        for i in range(0, len(cs_vals), step):
            bi, gt, cs, u32 = cs_vals[i]
            closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
            match = '✓' if abs(cs - closest['cs']) <= 5 else '✗'
            print(f"    block {bi:3d} (~{gt:6.0f}s): cs={cs:4d}  "
                  f"oracle={closest['cs']:4d} {match}")

    if not cs_candidates:
        print("  No CS candidates found.")

    # ── Search for COORDINATES ──
    print(f"\n{'='*70}")
    print(f"COORDINATE SEARCH")
    print(f"{'='*70}")

    coord_candidates = []
    for (tag, off), series in tag_series.items():
        if len(series) < 20:
            continue
        vals = [v for _, _, v, _ in series if v == v]  # filter NaN
        if len(vals) < 20:
            continue

        # Coords: f32 in 100-15500 range, smooth trajectory
        coord_vals = [v for v in vals if 100 < abs(v) < 15500]
        if len(coord_vals) < len(vals) * 0.3:
            continue

        diffs = [abs(coord_vals[i] - coord_vals[i-1]) for i in range(1, len(coord_vals))]
        if not diffs:
            continue
        med_diff = sorted(diffs)[len(diffs) // 2]
        val_range = max(coord_vals) - min(coord_vals)

        if val_range < 200:
            continue

        smoothness = med_diff / val_range if val_range > 0 else 999
        coord_candidates.append((tag, off, smoothness, val_range, med_diff,
                                 len(series), series))

    coord_candidates.sort(key=lambda x: x[2])  # lower smoothness = smoother
    for tag, off, smooth, rng, med, n, series in coord_candidates[:15]:
        label = "*** SMOOTH ***" if smooth < 0.05 else "smooth" if smooth < 0.15 else ""
        print(f"\n  tag=0x{tag:08x} +0x{off:03x}: {label} smoothness={smooth:.3f}, "
              f"range={rng:.0f}, med_step={med:.0f}, {n} pts")
        step = max(1, len(series) // 10)
        for i in range(0, len(series), step):
            bi, gt, fval, u32 = series[i]
            print(f"    block {bi:3d} (~{gt:6.0f}s): {fval:10.2f}")

    if not coord_candidates:
        print("  No coordinate candidates found.")

    # ── Summary ──
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    if level_candidates:
        best = level_candidates[0]
        print(f"  Best level: tag=0x{best[0]:08x} +0x{best[1]:03x}, accuracy={best[2]:.1%}")
    if cs_candidates:
        best = cs_candidates[0]
        print(f"  Best CS:    tag=0x{best[0]:08x} +0x{best[1]:03x}, accuracy={best[2]:.1%}")
    if coord_candidates and coord_candidates[0][2] < 0.15:
        best = coord_candidates[0]
        print(f"  Best coord: tag=0x{best[0]:08x} +0x{best[1]:03x}, smoothness={best[2]:.3f}")

    return tag_series, tag_freq


def scan_all_pids_bswap(emu, rofl_path, entity=0x400000b2):
    """Scan ALL PIDs with blocks for entity, extract bswap plaintext, match oracle.

    Tests each PID quickly: only bswap-confirmed values, compare to oracle.
    """
    from unicorn import UC_HOOK_MEM_WRITE

    oracle = json.load(open('/tmp/oracle_data.json'))
    game_length = oracle['game_length']
    oracle_tl = []
    for s in oracle['snapshots']:
        g = s['players'].get('Garen', {})
        oracle_tl.append({
            'time': s['actual_time'],
            'level': g.get('level', 0),
            'cs': g.get('scores', {}).get('creepScore', 0),
        })

    # Find all PIDs with blocks for this entity
    print("Scanning all blocks for entity...")
    all_blocks = parse_rofl_blocks(rofl_path, param_filter={entity})
    pid_blocks = defaultdict(list)
    for b in all_blocks:
        pid_blocks[b['pid']].append(b)

    # Sort by block count, filter to PIDs with deserializers
    pid_list = []
    for pid in sorted(pid_blocks.keys(), key=lambda p: -len(pid_blocks[p])):
        if str(pid) in emu.pid_map and len(pid_blocks[pid]) >= 20:
            pid_list.append(pid)

    print(f"Found {len(pid_list)} PIDs with ≥20 blocks and deserializers")
    for pid in pid_list[:30]:
        print(f"  PID {pid:4d}: {len(pid_blocks[pid]):4d} blocks")

    BIT_READER_LO = 0xfe0000
    BIT_READER_HI = 0xff1000

    results_summary = []

    for pid in pid_list:
        blocks = pid_blocks[pid]
        n_blocks = len(blocks)
        tpb = game_length / n_blocks
        pid_info = emu.pid_map[str(pid)]
        deser_addr = BASE_ADDR + pid_info['deser']

        # Per (field_tag, struct_offset) → [(block_idx, game_time, f32, u32)]
        tag_series = defaultdict(list)
        # Per struct_offset (no tag) → [(block_idx, game_time, f32, u32)]
        offset_series = defaultdict(list)

        for bi in range(n_blocks):
            block = blocks[bi]
            payload = block['payload']
            gt = bi * tpb

            pay_addr = SCRATCH_BASE + 0x100
            pp_addr = SCRATCH_BASE + 0x200
            pay_end = pay_addr + len(payload)
            emu.mu.mem_write(pay_addr, payload + b'\x00' * 128)

            hook_state = {'prev': {}, 'bswap': {}}

            def write_hook(mu, access, address, size, value, ud):
                val = value & ((1 << (size * 8)) - 1)
                hoff = address - HEAP_BASE
                if 0x10 <= hoff < 0x200 and size == 4:
                    rip = mu.reg_read(UC_X86_REG_RIP)
                    rva = rip - BASE_ADDR
                    if BIT_READER_LO <= rva <= BIT_READER_HI:
                        if hoff in hook_state['prev']:
                            prev_val = hook_state['prev'][hoff]
                            bswapped = struct.unpack('>I', struct.pack('<I', prev_val))[0]
                            if val == bswapped and val != 0 and val != prev_val:
                                hook_state['bswap'][hoff] = val
                        hook_state['prev'][hoff] = val
                return True

            current_pos = pay_addr
            for iteration in range(20):
                if current_pos >= pay_end:
                    break

                emu._alloc_cursor = 0
                emu._alloc_regions = []
                emu._run_constructor(pid)

                hook_state['prev'].clear()
                hook_state['bswap'].clear()

                emu.mu.mem_write(pp_addr, struct.pack('<Q', current_pos))
                emu.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
                emu.mu.reg_write(UC_X86_REG_RDX, pp_addr)
                emu.mu.reg_write(UC_X86_REG_R8, pay_end)
                emu.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
                rsp = emu.mu.reg_read(UC_X86_REG_RSP) - 8
                emu.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
                emu.mu.reg_write(UC_X86_REG_RSP, rsp)

                h = emu.mu.hook_add(UC_HOOK_MEM_WRITE, write_hook,
                                    begin=HEAP_BASE, end=HEAP_BASE + 0x200)
                try:
                    emu.mu.emu_start(deser_addr, STOP_ADDR, timeout=10_000_000)
                except:
                    pass
                emu.mu.hook_del(h)

                new_pos = struct.unpack('<Q', emu.mu.mem_read(pp_addr, 8))[0]
                if new_pos <= current_pos:
                    break

                tag = struct.unpack('<I', emu.mu.mem_read(HEAP_BASE + 0xC, 4))[0]

                for hoff, val in hook_state['bswap'].items():
                    fval = struct.unpack('<f', struct.pack('<I', val))[0]
                    if fval != fval:  # NaN
                        continue
                    tag_series[(tag, hoff)].append((bi, gt, fval, val))
                    offset_series[hoff].append((bi, gt, fval, val))

                current_pos = new_pos

        # Quick analysis: check for level, CS, coord matches
        pid_hits = []

        for (tag, off), series in tag_series.items():
            if len(series) < 10:
                continue
            vals = [v for _, _, v, _ in series]

            # Level: f32 integers 1-18, increasing
            int_vals = [(bi, gt, round(v)) for bi, gt, v, _ in series
                        if 0.5 < v < 20.5 and abs(v - round(v)) < 0.02]
            if len(int_vals) >= max(5, len(series) * 0.4):
                levels = [iv for _, _, iv in int_vals]
                unique_levels = sorted(set(levels))
                if len(unique_levels) >= 3 and all(1 <= l <= 20 for l in unique_levels):
                    correct = 0
                    for bi, gt, lv in int_vals:
                        closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                        if lv == closest['level']:
                            correct += 1
                    acc = correct / len(int_vals)
                    if acc > 0.3:
                        pid_hits.append(('LEVEL', tag, off, acc, len(int_vals),
                                         unique_levels, int_vals))

            # CS: f32 0-500, mostly increasing
            cs_vals = [(bi, gt, round(v)) for bi, gt, v, _ in series
                       if 0 <= v <= 500 and abs(v - round(v)) < 1.0]
            if len(cs_vals) >= max(5, len(series) * 0.3) and max(c for _, _, c in cs_vals) > 30:
                raw_cs = [c for _, _, c in cs_vals]
                inc = sum(1 for i in range(1, len(raw_cs)) if raw_cs[i] > raw_cs[i-1])
                if inc > len(raw_cs) * 0.2:
                    correct = 0
                    for bi, gt, cs in cs_vals:
                        closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                        if abs(cs - closest['cs']) <= 5:
                            correct += 1
                    acc = correct / len(cs_vals)
                    if acc > 0.2:
                        pid_hits.append(('CS', tag, off, acc, len(cs_vals),
                                         max(raw_cs), cs_vals))

        # Also check aggregated per-offset (all tags mixed)
        for off, series in offset_series.items():
            if len(series) < 20:
                continue
            vals = [v for _, _, v, _ in series]

            # Smooth coordinate trajectory
            coord_vals = [(bi, gt, v) for bi, gt, v, _ in series
                          if 100 < abs(v) < 15500]
            if len(coord_vals) > len(series) * 0.5:
                cv = [v for _, _, v in coord_vals]
                diffs = [abs(cv[i] - cv[i-1]) for i in range(1, len(cv))]
                if diffs:
                    med = sorted(diffs)[len(diffs) // 2]
                    rng = max(cv) - min(cv)
                    if rng > 500 and med < rng * 0.05:
                        pid_hits.append(('COORD', 'all', off, med/rng, len(coord_vals),
                                         rng, coord_vals))

        if pid_hits:
            results_summary.append((pid, n_blocks, pid_hits))
            print(f"\n{'='*60}")
            print(f"*** PID {pid}: {n_blocks} blocks, {tpb:.1f}s/block ***")
            print(f"{'='*60}")
            for hit in pid_hits:
                kind = hit[0]
                if kind == 'LEVEL':
                    _, tag, off, acc, n, unique_lvls, int_vals = hit
                    print(f"  LEVEL tag=0x{tag:08x} +0x{off:03x}: acc={acc:.0%}, "
                          f"{n} pts, levels={unique_lvls}")
                    for bi, gt, lv in int_vals[::max(1, len(int_vals)//5)]:
                        closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                        m = '✓' if lv == closest['level'] else '✗'
                        print(f"    block {bi:3d} (~{gt:6.0f}s): L={lv:2d} oracle={closest['level']:2d} {m}")
                elif kind == 'CS':
                    _, tag, off, acc, n, max_cs, cs_vals = hit
                    print(f"  CS tag=0x{tag:08x} +0x{off:03x}: acc={acc:.0%}, "
                          f"{n} pts, max={max_cs}")
                    for bi, gt, cs in cs_vals[::max(1, len(cs_vals)//5)]:
                        closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                        m = '✓' if abs(cs - closest['cs']) <= 5 else '✗'
                        print(f"    block {bi:3d} (~{gt:6.0f}s): CS={cs:4d} oracle={closest['cs']:4d} {m}")
                elif kind == 'COORD':
                    _, tag, off, smooth, n, rng, cv = hit
                    print(f"  COORD +0x{off:03x}: smoothness={smooth:.3f}, range={rng:.0f}, {n} pts")
                    for bi, gt, v in cv[::max(1, len(cv)//8)]:
                        print(f"    block {bi:3d} (~{gt:6.0f}s): {v:10.2f}")

        if not pid_hits:
            # Still print a brief status
            n_bswap = sum(len(s) for s in tag_series.values())
            n_tags = len(set(t for t, _ in tag_series.keys()))
            if n_bswap > 0:
                print(f"  PID {pid:4d}: {n_blocks} blocks, {n_bswap} bswap vals, "
                      f"{n_tags} tags — no matches")

    # Final summary
    print(f"\n{'='*70}")
    print(f"FINAL SUMMARY")
    print(f"{'='*70}")
    if results_summary:
        for pid, n_blocks, hits in results_summary:
            for h in hits:
                kind = h[0]
                if kind == 'LEVEL':
                    print(f"  PID {pid}: LEVEL accuracy={h[3]:.0%} at tag=0x{h[1]:08x} +0x{h[2]:03x}")
                elif kind == 'CS':
                    print(f"  PID {pid}: CS accuracy={h[3]:.0%} at tag=0x{h[1]:08x} +0x{h[2]:03x}")
                elif kind == 'COORD':
                    print(f"  PID {pid}: COORD smoothness={h[3]:.3f} at +0x{h[2]:03x}")
    else:
        print("  No matches found across any PID.")

    return results_summary


def encrypted_pattern_scan(emu, rofl_path, entity=0x400000b2):
    """Scan ALL PIDs: decode all blocks, read encrypted struct, pattern-match against oracle.

    The cipher is deterministic: same plaintext → same ciphertext. So:
    - Level (1→18 over game) maps to ~18 unique encrypted u32 values
    - The pattern of changes (step function) should correlate with oracle level timeline
    - CS (monotonically increasing) maps to many unique values, always increasing

    For each PID, we decode each block's FIRST sub-message, read the final struct,
    and check every 4-byte offset for these patterns.
    """
    oracle = json.load(open('/tmp/oracle_data.json'))
    game_length = oracle['game_length']
    oracle_tl = []
    for s in oracle['snapshots']:
        g = s['players'].get('Garen', {})
        oracle_tl.append({
            'time': s['actual_time'],
            'level': g.get('level', 0),
            'cs': g.get('scores', {}).get('creepScore', 0),
        })

    # Find all PIDs
    print("Scanning all blocks for entity...")
    all_blocks = parse_rofl_blocks(rofl_path, param_filter={entity})
    pid_blocks = defaultdict(list)
    for b in all_blocks:
        pid_blocks[b['pid']].append(b)

    pid_list = []
    for pid in sorted(pid_blocks.keys(), key=lambda p: -len(pid_blocks[p])):
        if str(pid) in emu.pid_map and len(pid_blocks[pid]) >= 30:
            pid_list.append(pid)

    print(f"Testing {len(pid_list)} PIDs...")

    from collections import Counter
    best_results = []

    for pid in pid_list:
        blocks = pid_blocks[pid]
        n_blocks = len(blocks)
        tpb = game_length / n_blocks
        pid_info = emu.pid_map[str(pid)]

        # Decode all blocks, capture final struct per block
        # For speed: only decode first sub-message per block
        struct_snapshots = []  # [(block_idx, game_time, struct_bytes)]

        for bi in range(n_blocks):
            block = blocks[bi]
            payload = block['payload']
            gt = bi * tpb

            results = emu.decode_payload(pid, payload, verbose=False)
            if results:
                final = bytes(emu.mu.mem_read(HEAP_BASE, 0x200))
                struct_snapshots.append((bi, gt, final))

        if len(struct_snapshots) < 20:
            continue

        # For each 4-byte offset, check the time series of encrypted values
        for off in range(0x10, 0x180, 4):
            time_series = []
            for bi, gt, sb in struct_snapshots:
                val = int.from_bytes(sb[off:off+4], 'little')
                if val != 0:
                    time_series.append((bi, gt, val))

            if len(time_series) < 20:
                continue

            vals = [v for _, _, v in time_series]
            unique = sorted(set(vals))
            n_unique = len(unique)

            # ── Level pattern: 5-25 unique values, step function ──
            if 5 <= n_unique <= 25:
                # Map each encrypted value to oracle level at that time
                val_to_levels = defaultdict(list)
                for bi, gt, val in time_series:
                    closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                    val_to_levels[val].append(closest['level'])

                # Check 1:1 mapping quality
                val_to_level = {}
                good = 0
                total_map = 0
                for val, levels in val_to_levels.items():
                    if len(levels) >= 2:
                        total_map += 1
                        mc = Counter(levels).most_common(1)[0]
                        if mc[1] >= len(levels) * 0.7:
                            val_to_level[val] = mc[0]
                            good += 1

                if total_map < 5 or good < total_map * 0.5:
                    continue

                mapped_levels = sorted(set(val_to_level.values()))
                if len(mapped_levels) < 5 or mapped_levels[0] > 5 or mapped_levels[-1] < 12:
                    continue

                # Score accuracy
                correct = 0
                total = 0
                for bi, gt, val in time_series:
                    if val in val_to_level:
                        closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                        if val_to_level[val] == closest['level']:
                            correct += 1
                        total += 1

                accuracy = correct / max(total, 1)
                if accuracy > 0.5:
                    best_results.append(('LEVEL', pid, off, accuracy, n_unique,
                                         mapped_levels, val_to_level, time_series))

            # ── Also check 2-byte at this offset ──
            for sub_off in [off, off + 2]:
                ts2 = []
                for bi, gt, sb in struct_snapshots:
                    v16 = int.from_bytes(sb[sub_off:sub_off+2], 'little')
                    if v16 != 0:
                        ts2.append((bi, gt, v16))
                if len(ts2) < 20:
                    continue
                vals16 = [v for _, _, v in ts2]
                uniq16 = sorted(set(vals16))
                n16 = len(uniq16)

                if 5 <= n16 <= 25:
                    val_to_levels = defaultdict(list)
                    for bi, gt, v in ts2:
                        closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                        val_to_levels[v].append(closest['level'])

                    val_to_level = {}
                    good = 0
                    total_map = 0
                    for v, levels in val_to_levels.items():
                        if len(levels) >= 2:
                            total_map += 1
                            mc = Counter(levels).most_common(1)[0]
                            if mc[1] >= len(levels) * 0.7:
                                val_to_level[v] = mc[0]
                                good += 1

                    if total_map < 5 or good < total_map * 0.5:
                        continue
                    mapped_levels = sorted(set(val_to_level.values()))
                    if len(mapped_levels) < 5 or mapped_levels[0] > 5 or mapped_levels[-1] < 12:
                        continue

                    correct = 0
                    total = 0
                    for bi, gt, v in ts2:
                        if v in val_to_level:
                            closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                            if val_to_level[v] == closest['level']:
                                correct += 1
                            total += 1
                    accuracy = correct / max(total, 1)
                    if accuracy > 0.5:
                        best_results.append(('LEVEL_U16', pid, sub_off, accuracy, n16,
                                             mapped_levels, val_to_level, ts2))

        print(f"  PID {pid:4d}: {n_blocks} blocks, {len(struct_snapshots)} decoded"
              f" — {sum(1 for r in best_results if r[1]==pid)} hits so far")

    # Report
    best_results.sort(key=lambda x: -x[3])
    print(f"\n{'='*70}")
    print(f"RESULTS: {len(best_results)} matches")
    print(f"{'='*70}")

    for kind, pid, off, acc, n_uniq, mapped_levels, mapping, ts in best_results[:10]:
        print(f"\n*** PID {pid} +0x{off:03x} [{kind}]: accuracy={acc:.1%}, "
              f"{n_uniq} unique, levels={mapped_levels} ***")

        # Show the encrypted→level mapping
        for val, level in sorted(mapping.items(), key=lambda x: x[1]):
            cnt = sum(1 for _, _, v in ts if v == val)
            if kind == 'LEVEL_U16':
                print(f"  encrypted 0x{val:04x} → level {level:2d} ({cnt} blocks)")
            else:
                print(f"  encrypted 0x{val:08x} → level {level:2d} ({cnt} blocks)")

        # Validation
        print(f"  Validation:")
        step = max(1, len(ts) // 8)
        for i in range(0, len(ts), step):
            bi, gt, val = ts[i]
            closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
            mapped = mapping.get(val, '?')
            m = '✓' if mapped == closest['level'] else '✗'
            if kind == 'LEVEL_U16':
                print(f"    block {bi:3d} (~{gt:6.0f}s): enc=0x{val:04x} → L={mapped}, "
                      f"oracle={closest['level']} {m}")
            else:
                print(f"    block {bi:3d} (~{gt:6.0f}s): enc=0x{val:08x} → L={mapped}, "
                      f"oracle={closest['level']} {m}")

    return best_results


def per_submsg_pattern_scan(emu, rofl_path, entity=0x400000b2):
    """For each PID, decode ALL blocks, extract per-sub-message struct_fields,
    group by field tag, and pattern-match against oracle.

    Unlike encrypted_pattern_scan which only saw the LAST sub-message's struct,
    this checks EVERY sub-message individually.
    """
    oracle = json.load(open('/tmp/oracle_data.json'))
    game_length = oracle['game_length']
    oracle_tl = []
    for s in oracle['snapshots']:
        g = s['players'].get('Garen', {})
        oracle_tl.append({
            'time': s['actual_time'],
            'level': g.get('level', 0),
            'cs': g.get('scores', {}).get('creepScore', 0),
        })

    all_blocks = parse_rofl_blocks(rofl_path, param_filter={entity})
    pid_blocks = defaultdict(list)
    for b in all_blocks:
        pid_blocks[b['pid']].append(b)

    pid_list = [p for p in sorted(pid_blocks.keys(), key=lambda p: -len(pid_blocks[p]))
                if str(p) in emu.pid_map and len(pid_blocks[p]) >= 30]
    print(f"Testing {len(pid_list)} PIDs with per-sub-message analysis...")

    from collections import Counter
    all_hits = []

    for pid in pid_list:
        blocks = pid_blocks[pid]
        n_blocks = len(blocks)
        tpb = game_length / n_blocks

        # (field_tag, struct_offset) → [(block_idx, game_time, value)]
        tag_off_series = defaultdict(list)
        tag_counter = defaultdict(int)

        for bi in range(n_blocks):
            block = blocks[bi]
            payload = block['payload']
            gt = bi * tpb

            results = emu.decode_payload(pid, payload, verbose=False)
            for r in results:
                sf = r.get('struct_fields', {})
                if not sf:
                    continue

                # Get field tag from offset 0x0C
                tag_entry = sf.get(0x0C)
                if tag_entry:
                    tag = tag_entry[1]  # (size, value)
                else:
                    tag = 0
                tag_counter[tag] += 1

                # Record all struct changes except vtable and tag itself
                for off, (sz, val) in sf.items():
                    if off < 0x10:  # skip vtable+tag
                        continue
                    tag_off_series[(tag, off)].append((bi, gt, val, sz))

        n_combos = len(tag_off_series)
        n_tags = len(tag_counter)
        total_pts = sum(len(s) for s in tag_off_series.values())

        if total_pts == 0:
            continue

        # Pattern match each (tag, offset) series
        pid_hits = []

        for (tag, off), series in tag_off_series.items():
            if len(series) < 10:
                continue

            vals = [v for _, _, v, _ in series]
            sizes = set(s for _, _, _, s in series)
            unique = sorted(set(vals))
            n_unique = len(unique)

            # ── Level: 5-25 unique values, step-function matching oracle ──
            if 5 <= n_unique <= 30:
                val_to_levels = defaultdict(list)
                for bi, gt, val, _ in series:
                    closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                    val_to_levels[val].append(closest['level'])

                val_to_level = {}
                good = 0
                total_map = 0
                for v, levels in val_to_levels.items():
                    if len(levels) >= 2:
                        total_map += 1
                        mc = Counter(levels).most_common(1)[0]
                        if mc[1] >= len(levels) * 0.7:
                            val_to_level[v] = mc[0]
                            good += 1

                if total_map >= 3 and good >= total_map * 0.5:
                    mapped = sorted(set(val_to_level.values()))
                    if len(mapped) >= 3:
                        # Score
                        correct = total = 0
                        for bi, gt, val, _ in series:
                            if val in val_to_level:
                                closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                                if val_to_level[val] == closest['level']:
                                    correct += 1
                                total += 1
                        acc = correct / max(total, 1)
                        if acc > 0.4:
                            pid_hits.append(('LEVEL', tag, off, acc, len(series),
                                             mapped, val_to_level, series, sizes))

            # ── CS: many unique values, mostly increasing, monotonic trend ──
            if n_unique >= 15 and 4 in sizes:
                # Check if values increase over game time
                increases = sum(1 for i in range(1, len(vals)) if vals[i] > vals[i-1])
                decreases = sum(1 for i in range(1, len(vals)) if vals[i] < vals[i-1])
                if increases > decreases * 2 and increases > len(vals) * 0.3:
                    pid_hits.append(('CS_PATTERN', tag, off, increases / len(vals),
                                     len(series), n_unique, None, series, sizes))

        if pid_hits:
            all_hits.extend(pid_hits)
            print(f"\n{'='*60}")
            print(f"*** PID {pid}: {n_blocks} blocks, {n_tags} tags, {n_combos} combos ***")
            print(f"{'='*60}")
            for h in pid_hits:
                kind = h[0]
                if kind == 'LEVEL':
                    _, tag, off, acc, n, mapped, mapping, series, sizes = h
                    print(f"  LEVEL tag=0x{tag:08x} +0x{off:03x} [{sizes}]: "
                          f"acc={acc:.0%}, {n} pts, levels={mapped}")
                    step = max(1, len(series) // 6)
                    for i in range(0, len(series), step):
                        bi, gt, val, sz = series[i]
                        closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                        ml = mapping.get(val, '?')
                        m = '✓' if ml == closest['level'] else '✗'
                        print(f"    block {bi:3d} (~{gt:6.0f}s): enc=0x{val:08x} → L={ml},"
                              f" oracle={closest['level']} {m}")
                elif kind == 'CS_PATTERN':
                    _, tag, off, ratio, n, n_uniq, _, series, sizes = h
                    print(f"  CS tag=0x{tag:08x} +0x{off:03x} [{sizes}]: "
                          f"inc_ratio={ratio:.0%}, {n} pts, {n_uniq} unique")
        else:
            print(f"  PID {pid:4d}: {n_blocks} blocks, {n_tags} tags, "
                  f"{total_pts} pts — no matches")

    # Summary
    print(f"\n{'='*70}")
    print(f"FINAL: {len(all_hits)} matches")
    print(f"{'='*70}")
    for h in sorted(all_hits, key=lambda x: -x[3])[:10]:
        kind = h[0]
        if kind == 'LEVEL':
            _, tag, off, acc, n, mapped, _, _, _ = h
            print(f"  LEVEL: PID ? tag=0x{tag:08x} +0x{off:03x}: acc={acc:.0%}, levels={mapped}")
        elif kind == 'CS_PATTERN':
            _, tag, off, ratio, n, n_uniq, _, _, _ = h
            print(f"  CS: tag=0x{tag:08x} +0x{off:03x}: inc={ratio:.0%}, {n_uniq} unique")

    return all_hits


def deep_level_analysis(emu, rofl_path, entity=0x400000b2):
    """Deep analysis on PIDs 577 and 487 which showed level correlations.

    Relax all thresholds: map every encrypted value to its oracle level,
    check if mapping is 1:1 and monotonically increasing.
    """
    oracle = json.load(open('/tmp/oracle_data.json'))
    game_length = oracle['game_length']
    oracle_tl = []
    for s in oracle['snapshots']:
        g = s['players'].get('Garen', {})
        oracle_tl.append({
            'time': s['actual_time'],
            'level': g.get('level', 0),
            'cs': g.get('scores', {}).get('creepScore', 0),
        })

    from collections import Counter

    # Test specific PIDs that showed promise
    target_pids = [577, 487, 169, 834, 490, 194, 266, 813, 705]

    for pid in target_pids:
        blocks = parse_rofl_blocks(rofl_path, pid_filter=pid, param_filter={entity})
        n_blocks = len(blocks)
        if n_blocks < 20:
            continue
        tpb = game_length / n_blocks

        pid_info = emu.pid_map.get(str(pid))
        if not pid_info:
            continue

        # Decode ALL blocks, capturing per-sub-message struct_fields
        # (field_tag, offset) → [(block_idx, game_time, value, size)]
        tag_off_series = defaultdict(list)

        for bi in range(n_blocks):
            block = blocks[bi]
            gt = bi * tpb
            results = emu.decode_payload(pid, block['payload'], verbose=False)
            for r in results:
                sf = r.get('struct_fields', {})
                tag_entry = sf.get(0x0C)
                tag = tag_entry[1] if tag_entry else 0
                for off, (sz, val) in sf.items():
                    if off < 0x10:
                        continue
                    tag_off_series[(tag, off)].append((bi, gt, val, sz))

        if not tag_off_series:
            continue

        print(f"\n{'='*70}")
        print(f"DEEP LEVEL ANALYSIS: PID {pid} ({n_blocks} blocks, {tpb:.1f}s/block)")
        print(f"{'='*70}")

        # For EVERY (tag, offset) combo with enough data, try to build a level mapping
        best = []
        for (tag, off), series in tag_off_series.items():
            if len(series) < 5:
                continue

            # Build mapping: for each encrypted value, what oracle level(s) appear?
            val_levels = defaultdict(list)
            for bi, gt, val, sz in series:
                closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                val_levels[val].append(closest['level'])

            # For each value, assign its most common oracle level
            full_mapping = {}
            for val, levels in val_levels.items():
                mc = Counter(levels).most_common(1)[0]
                full_mapping[val] = mc[0]

            # Check quality: is the mapping mostly 1:1? (different vals → different levels)
            level_to_vals = defaultdict(set)
            for val, level in full_mapping.items():
                level_to_vals[level].add(val)

            # Injective check: how many levels map to exactly 1 encrypted value?
            injective = sum(1 for l, vs in level_to_vals.items() if len(vs) == 1)
            total_levels = len(level_to_vals)

            if total_levels < 3:
                continue

            # Check temporal consistency: when we apply the mapping to the time series,
            # does the decoded level match oracle at each point?
            correct = 0
            for bi, gt, val, sz in series:
                closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                if full_mapping.get(val) == closest['level']:
                    correct += 1
            acc = correct / len(series)

            # Also check: does the mapped level sequence show monotonic increase?
            mapped_seq = [full_mapping[val] for _, _, val, _ in series]
            mono_inc = sum(1 for i in range(1, len(mapped_seq))
                          if mapped_seq[i] >= mapped_seq[i-1])
            mono_ratio = mono_inc / max(len(mapped_seq) - 1, 1)

            if acc > 0.3 or (total_levels >= 5 and mono_ratio > 0.7):
                best.append((tag, off, acc, total_levels, injective,
                             mono_ratio, len(series), full_mapping, series))

        best.sort(key=lambda x: -(x[2] * x[5]))  # sort by acc * monotonicity

        for tag, off, acc, n_levels, injective, mono, n, mapping, series in best[:5]:
            sizes = set(s for _, _, _, s in series)
            print(f"\n  tag=0x{tag:08x} +0x{off:03x} [{sizes}]: "
                  f"acc={acc:.0%}, {n_levels} levels, {injective}/{n_levels} injective, "
                  f"mono={mono:.0%}, {n} pts")

            # Show the mapping sorted by level
            for level in sorted(set(mapping.values())):
                vals = [v for v, l in mapping.items() if l == level]
                counts = {v: sum(1 for _, _, val, _ in series if val == v) for v in vals}
                vals_str = ", ".join(f"0x{v:02x}({counts[v]})" if v < 256
                                    else f"0x{v:08x}({counts[v]})" for v in vals)
                print(f"    level {level:2d} ← {vals_str}")

            # Show time series with oracle comparison
            print(f"  Time series:")
            step = max(1, len(series) // 12)
            for i in range(0, len(series), step):
                bi, gt, val, sz = series[i]
                closest = min(oracle_tl, key=lambda o: abs(o['time'] - gt))
                ml = mapping[val]
                m = '✓' if ml == closest['level'] else f'✗({closest["level"]})'
                if val < 256:
                    print(f"    block {bi:3d} (~{gt:6.0f}s): enc=0x{val:02x} → L={ml:2d} {m}")
                else:
                    print(f"    block {bi:3d} (~{gt:6.0f}s): enc=0x{val:08x} → L={ml:2d} {m}")

        if not best:
            print(f"  No level patterns found")

    return


def main():
    rofl_path = 'data/replays/NA1-5528069928.rofl'
    entity_garen = 0x400000b2

    print("Initializing Emulator V2...")
    emu = EmulatorV2()
    print("  PE sections loaded, SKIP NOT patched, malloc hooked")

    # Per-sub-message pattern scan: use decode_payload's struct_fields per result
    results = per_submsg_pattern_scan(emu, rofl_path, entity=entity_garen)

    # Deep analysis on promising PIDs
    deep_level_analysis(emu, rofl_path, entity_garen)


if __name__ == '__main__':
    main()
