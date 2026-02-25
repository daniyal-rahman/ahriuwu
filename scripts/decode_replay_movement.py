#!/usr/bin/env python3
"""
Decode movement/waypoint data from League of Legends replay (netid=762).
Extracts: entity_id, current_pos, destination, speed, timestamp, champion_name
"""
import struct
import pickle
import json
from collections import defaultdict
from unicorn import Uc, UC_ARCH_X86, UC_MODE_64, UC_PROT_ALL, UC_PROT_READ, UC_PROT_WRITE
from unicorn import UC_HOOK_MEM_WRITE, UC_HOOK_MEM_READ_UNMAPPED, UC_HOOK_MEM_WRITE_UNMAPPED
from unicorn import UC_HOOK_MEM_FETCH_UNMAPPED, UC_HOOK_CODE
from unicorn.x86_const import *

BASE_ADDR    = 0x140000000
STACK_BASE   = 0x7FFFFFFF0000
STACK_SIZE   = 0x4000
HEAP_BASE    = 0x7FFFFFFFA000
HEAP_SIZE    = 0x20000
SCRATCH_BASE = 0x7FFFFFF00000
SCRATCH_SIZE = 0x10000
ALLOC_BASE   = 0x7FFFFFF80000
ALLOC_SIZE   = 0x40000
TEXT_RVA, RDATA_RVA, DATA_RVA = 0x1000, 0x18e9000, 0x1d0c000
SKIP_RVA     = 0x1186e30
MALLOC_RVA   = 0x10f98f0
FREE_RVA     = 0x10f9920
DESER_RVA    = 0x1002930
SUB_INIT_RVA = 0xe03db0
MAIN_VTABLE  = 0x19eb6c0
STOP_ADDR    = 0xDEAD0000

def align_down(a, s=0x1000): return a & ~(s-1)
def align_up(s, p=0x1000): return (s+p-1) & ~(p-1)


class MovementDecoder:
    def __init__(self, text_path='/tmp/pe_dump/text.bin',
                 rdata_path='/tmp/pe_dump/rdata.bin',
                 data_path='/tmp/pe_dump/data.bin'):
        self.text_raw = open(text_path, 'rb').read()
        self.rdata_raw = open(rdata_path, 'rb').read()
        self.data_raw = open(data_path, 'rb').read()
        self._setup_emu()

    def _setup_emu(self):
        mu = Uc(UC_ARCH_X86, UC_MODE_64)
        mu.mem_map(STACK_BASE, STACK_SIZE, UC_PROT_READ | UC_PROT_WRITE)
        mu.mem_map(HEAP_BASE, align_up(HEAP_SIZE), UC_PROT_READ | UC_PROT_WRITE)
        mu.mem_map(SCRATCH_BASE, SCRATCH_SIZE, UC_PROT_READ | UC_PROT_WRITE)
        mu.mem_map(ALLOC_BASE, ALLOC_SIZE, UC_PROT_ALL)
        pe_start = align_down(BASE_ADDR + TEXT_RVA)
        pe_end = align_up(BASE_ADDR + DATA_RVA + len(self.data_raw) + 0x1000)
        mu.mem_map(pe_start, pe_end - pe_start, UC_PROT_ALL)
        mu.mem_write(BASE_ADDR + TEXT_RVA, self.text_raw)
        mu.mem_write(BASE_ADDR + RDATA_RVA, self.rdata_raw)
        mu.mem_write(BASE_ADDR + DATA_RVA, self.data_raw)
        # Patch skip, free, mutexes
        mu.mem_write(BASE_ADDR + SKIP_RVA, b'\x48\xC7\xC0\x01\x00\x00\x00\xC3')
        mu.mem_write(BASE_ADDR + FREE_RVA, b'\xC3')
        for rva in [0x1858c98, 0x1858d04, 0x1858fd4]:
            mu.mem_write(BASE_ADDR + rva, b'\xC3')
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

        # Malloc hook
        self._alloc_cursor = 0
        self._alloc_regions = []

        def malloc_hook(mu, address, size, user_data):
            if address == BASE_ADDR + MALLOC_RVA:
                sz = mu.reg_read(UC_X86_REG_RCX)
                aligned = max((sz + 15) & ~15, 0x200)
                ptr = ALLOC_BASE + self._alloc_cursor
                self._alloc_regions.append((self._alloc_cursor, sz))
                self._alloc_cursor += aligned
                mu.reg_write(UC_X86_REG_RAX, ptr)
                rsp = mu.reg_read(UC_X86_REG_RSP)
                ret = struct.unpack('<Q', mu.mem_read(rsp, 8))[0]
                mu.reg_write(UC_X86_REG_RSP, rsp + 8)
                mu.reg_write(UC_X86_REG_RIP, ret)
        mu.hook_add(UC_HOOK_CODE, malloc_hook,
                    begin=BASE_ADDR + MALLOC_RVA, end=BASE_ADDR + MALLOC_RVA + 1)

        self.mu = mu
        self._write_log = []

    def _init_struct(self):
        """Initialize the packet struct with sub-object."""
        self.mu.mem_write(HEAP_BASE, b'\x00' * 0x400)
        self.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE + 0x10)
        self.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
        rsp = self.mu.reg_read(UC_X86_REG_RSP) - 8
        self.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
        self.mu.reg_write(UC_X86_REG_RSP, rsp)
        try:
            self.mu.emu_start(BASE_ADDR + SUB_INIT_RVA, STOP_ADDR, timeout=5000000)
        except:
            pass
        self.mu.mem_write(HEAP_BASE, struct.pack('<Q', BASE_ADDR + MAIN_VTABLE))

    def _track_writes(self, mu, access, address, size, value, user_data):
        val = value & ((1 << (size * 8)) - 1)
        off = address - ALLOC_BASE
        if 0 <= off < ALLOC_SIZE:
            self._write_log.append((off, size, val))
        # Also track struct writes for small packets
        soff = address - HEAP_BASE
        if 0 <= soff < 0x200:
            self._write_log.append((soff + 0x100000, size, val))  # mark with offset
        return True

    def decode(self, payload):
        """Decode a netid=762 payload. Returns dict with movement data."""
        self._write_log = []
        self._alloc_cursor = 0
        self._alloc_regions = []
        self.mu.mem_write(ALLOC_BASE, b'\x00' * min(0x8000, ALLOC_SIZE))
        self._init_struct()

        pay_addr = SCRATCH_BASE + 0x100
        self.mu.mem_write(pay_addr, payload + b'\x00' * 128)
        pp = SCRATCH_BASE + 0x200
        self.mu.mem_write(pp, struct.pack('<Q', pay_addr))
        self.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
        self.mu.reg_write(UC_X86_REG_RDX, pp)
        self.mu.reg_write(UC_X86_REG_R8, pay_addr + len(payload))
        self.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
        rsp = self.mu.reg_read(UC_X86_REG_RSP) - 8
        self.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
        self.mu.reg_write(UC_X86_REG_RSP, rsp)

        h1 = self.mu.hook_add(UC_HOOK_MEM_WRITE, self._track_writes,
                              begin=ALLOC_BASE, end=ALLOC_BASE + ALLOC_SIZE)
        h2 = self.mu.hook_add(UC_HOOK_MEM_WRITE, self._track_writes,
                              begin=HEAP_BASE, end=HEAP_BASE + 0x200)
        try:
            self.mu.emu_start(BASE_ADDR + DESER_RVA, STOP_ADDR, timeout=60000000)
        except:
            pass
        self.mu.hook_del(h1)
        self.mu.hook_del(h2)

        return self._extract_fields()

    def _extract_fields(self):
        """Extract pre-S-box values from write log."""
        result = {}

        if self._alloc_regions:
            # Large packet: data in allocated buffer
            main_off, main_sz = self._alloc_regions[0]
            by_rel = defaultdict(list)
            for off, sz, val in self._write_log:
                if off >= 0x100000:
                    continue  # struct marker
                rel = off - main_off
                if 0 <= rel < main_sz + 32:
                    by_rel[rel].append((sz, val))

            def get_u32(offset):
                if offset in by_rel:
                    mb = [(sz, val) for sz, val in by_rel[offset] if sz >= 4]
                    if mb:
                        return mb[-1][1]
                return None

            def get_f32(offset):
                v = get_u32(offset)
                if v is not None:
                    return struct.unpack('<f', struct.pack('<I', v & 0xFFFFFFFF))[0]
                return None

            eid = get_u32(0x008)
            result['entity_id'] = eid
            result['current_x'] = get_f32(0x01c)
            result['current_z'] = get_f32(0x020)
            result['current_y'] = get_f32(0x024)
            result['speed'] = get_f32(0x02c)
            result['dest_x'] = get_f32(0x054)
            result['dest_z'] = get_f32(0x058)
            result['dest_y'] = get_f32(0x05c)
            result['game_time'] = get_f32(0x10c)
            result['sequence'] = get_u32(0x028)
            result['has_destination'] = True

            # Extract champion name from 2nd buffer
            if len(self._alloc_regions) >= 2:
                buf2_off, buf2_sz = self._alloc_regions[1]
                name_bytes = []
                name_writes = [(off - buf2_off, sz, val)
                               for off, sz, val in self._write_log
                               if 0 <= off - buf2_off < buf2_sz and off < 0x100000]
                # Get last 1-byte write per offset
                by_byte = defaultdict(list)
                for rel, sz, val in name_writes:
                    if sz == 1:
                        by_byte[rel].append(val)
                name = ''
                for i in range(buf2_sz):
                    if i in by_byte:
                        c = by_byte[i][-1]
                        if 32 <= c < 127:
                            name += chr(c)
                result['champion_name'] = name
        else:
            # Small packet: data in sub-object (struct+0x10)
            by_rel = defaultdict(list)
            for off, sz, val in self._write_log:
                if off >= 0x100000:
                    soff = off - 0x100000
                    sub_rel = soff - 0x10  # sub-object starts at struct+0x10
                    if 0 <= sub_rel < 0x80:
                        by_rel[sub_rel].append((sz, val))

            def get_f32_sub(offset):
                if offset in by_rel:
                    mb = [(sz, val) for sz, val in by_rel[offset] if sz >= 4]
                    if mb:
                        v = mb[-1][1]
                        return struct.unpack('<f', struct.pack('<I', v & 0xFFFFFFFF))[0]
                return None

            result['current_x'] = get_f32_sub(0x24)
            result['current_z'] = get_f32_sub(0x28)
            result['current_y'] = get_f32_sub(0x2c)
            result['has_destination'] = False

        return result


def main():
    import sys

    print("Loading replay data...")
    with open('/tmp/rofl_blocks_cache.pkl', 'rb') as f:
        blocks = pickle.load(f)

    PLAYER_ID_START = 0x400000ae
    hero_params = set(range(PLAYER_ID_START, PLAYER_ID_START + 10))

    # Filter movement blocks
    mov_blocks = [b for b in blocks if b['packet_id'] == 762 and b['param'] in hero_params]
    mov_blocks.sort(key=lambda b: b['timestamp'])
    print(f"Found {len(mov_blocks)} movement blocks for heroes")
    print(f"  Large (>100B): {sum(1 for b in mov_blocks if len(b['payload']) > 100)}")
    print(f"  Small (<=100B): {sum(1 for b in mov_blocks if len(b['payload']) <= 100)}")

    # Decode all
    decoder = MovementDecoder()
    movements = []
    errors = 0

    for i, b in enumerate(mov_blocks):
        if i % 100 == 0:
            print(f"  Decoding {i}/{len(mov_blocks)}...", end='\r')
        try:
            result = decoder.decode(b['payload'])
            result['block_timestamp'] = b['timestamp']
            result['block_param'] = b['param']
            result['payload_size'] = len(b['payload'])
            movements.append(result)
        except Exception as e:
            errors += 1

    print(f"\nDecoded {len(movements)} movements ({errors} errors)")

    # Summary stats
    by_entity = defaultdict(list)
    for m in movements:
        eid = m.get('entity_id', m.get('block_param'))
        by_entity[eid].append(m)

    print(f"\nPer-entity summary:")
    for eid in sorted(by_entity.keys()):
        ems = by_entity[eid]
        with_dest = sum(1 for m in ems if m.get('has_destination'))
        champ = next((m['champion_name'] for m in ems if m.get('champion_name')), '?')
        xs = [m['current_x'] for m in ems if m.get('current_x') is not None]
        ys = [m['current_y'] for m in ems if m.get('current_y') is not None]
        coord_str = (f"X=[{min(xs):.0f}-{max(xs):.0f}], Y=[{min(ys):.0f}-{max(ys):.0f}]"
                     if xs and ys else "NO COORDS")
        print(f"  0x{eid:x} ({champ:15s}): {len(ems):4d} updates, "
              f"{with_dest:3d} with dest, {coord_str}")

    # Save
    out_path = '/tmp/movement_data.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(movements, f)
    print(f"\nSaved to {out_path}")

    # Also save as JSON for inspection
    json_path = '/tmp/movement_data_sample.json'
    sample = movements[:50]
    with open(json_path, 'w') as f:
        json.dump(sample, f, indent=2, default=str)
    print(f"Sample saved to {json_path}")


if __name__ == '__main__':
    main()
