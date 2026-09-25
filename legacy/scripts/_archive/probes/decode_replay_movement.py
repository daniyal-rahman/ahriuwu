#!/usr/bin/env python3
"""
Decode movement/waypoint data from League of Legends replay.
Supports patch 16.3 (netid=762), 16.4 (netid=437), and 16.7 (netid=487).
Uses Unicorn Engine to emulate the game's deserializer functions.
"""
import struct
import pickle
import json
import zstandard as zstd
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
STOP_ADDR    = 0xDEAD0000

# Patch-specific addresses
PATCH_16_3 = {
    'TEXT_RVA':    0x1000,
    'RDATA_RVA':   0x18e9000,
    'DATA_RVA':    0x1d0c000,
    'SKIP_RVA':    0x1186e30,
    'MALLOC_RVA':  0x10f98f0,
    'FREE_RVA':    0x10f9920,
    'DESER_RVA':   0x1002930,
    'SUB_INIT_RVA': 0xe03db0,
    'MAIN_VTABLE': 0x19eb6c0,
    'MUTEXES':     [0x1858c98, 0x1858d04, 0x1858fd4],
    'MOVEMENT_NETID': 762,
}

PATCH_16_4 = {
    'TEXT_RVA':    0x1000,
    'RDATA_RVA':   0x1927000,
    'DATA_RVA':    0x1d4e000,
    'SKIP_RVA':    0x11bcec0,
    'MALLOC_RVA':  0x112cb00,
    'FREE_RVA':    0x112cb30,
    'DESER_RVA':   0x1033400,   # pid=437 movement deserializer
    'SUB_INIT_RVA': 0xe2a490,   # sub-object init for struct+0x18
    'SUB_INIT_OFF': 0x18,       # offset within struct to init
    'MAIN_VTABLE': 0x1a2afe8,   # pid=437 vtable
    'MUTEXES':     [],
    'MOVEMENT_NETID': 437,
}

PATCH_16_7 = {
    'TEXT_RVA':    0x1000,
    'RDATA_RVA':   0x18fd000,
    'DATA_RVA':    0x1d21000,
    'SKIP_RVA':    0x118b120,   # validation check called at start of deserializer
    'MALLOC_RVA':  0x10fa120,
    'FREE_RVA':    0x10fa150,   # jmp trampoline to 0x10fe1b0
    'DESER_RVA':   0xf53050,    # pid=487 movement deserializer (vtable entry[1])
    'SUB_INIT_RVA': None,       # 16.7: no separate sub-init needed
    'SUB_INIT_OFF': None,
    'MAIN_VTABLE': 0x19fc1c8,   # pid=487 vtable
    'MUTEXES':     [],
    'MOVEMENT_NETID': 487,
    'CONSTRUCTOR':  0xe050a0,    # movement object constructor
    'STRUCT_SIZE':  0x28,        # smaller struct in 16.7
}

def align_down(a, s=0x1000): return a & ~(s-1)
def align_up(s, p=0x1000): return (s+p-1) & ~(p-1)


class MovementDecoder:
    def __init__(self, text_path='/tmp/pe_dump/text.bin',
                 rdata_path='/tmp/pe_dump/rdata.bin',
                 data_path='/tmp/pe_dump/data.bin',
                 patch='16.4'):
        self.text_raw = open(text_path, 'rb').read()
        self.rdata_raw = open(rdata_path, 'rb').read()
        self.data_raw = open(data_path, 'rb').read()
        self.patch = patch
        if patch == '16.7':
            self.addrs = PATCH_16_7
        elif patch == '16.4':
            self.addrs = PATCH_16_4
        else:
            self.addrs = PATCH_16_3
        self._setup_emu()
        self._init_struct()  # one-time init, saves sub-object state for reuse

    def _setup_emu(self):
        a = self.addrs
        mu = Uc(UC_ARCH_X86, UC_MODE_64)
        mu.mem_map(STACK_BASE, STACK_SIZE, UC_PROT_READ | UC_PROT_WRITE)
        mu.mem_map(HEAP_BASE, align_up(HEAP_SIZE), UC_PROT_READ | UC_PROT_WRITE)
        mu.mem_map(SCRATCH_BASE, SCRATCH_SIZE, UC_PROT_READ | UC_PROT_WRITE)
        mu.mem_map(ALLOC_BASE, ALLOC_SIZE, UC_PROT_ALL)
        pe_start = align_down(BASE_ADDR + a['TEXT_RVA'])
        pe_end = align_up(BASE_ADDR + a['DATA_RVA'] + len(self.data_raw) + 0x1000)
        mu.mem_map(pe_start, pe_end - pe_start, UC_PROT_ALL)
        mu.mem_write(BASE_ADDR + a['TEXT_RVA'], self.text_raw)
        mu.mem_write(BASE_ADDR + a['RDATA_RVA'], self.rdata_raw)
        mu.mem_write(BASE_ADDR + a['DATA_RVA'], self.data_raw)

        # Patch skip → mov rax, 1; ret
        mu.mem_write(BASE_ADDR + a['SKIP_RVA'], b'\x48\xC7\xC0\x01\x00\x00\x00\xC3')
        # Patch free → ret
        mu.mem_write(BASE_ADDR + a['FREE_RVA'], b'\xC3')
        # Patch mutexes → ret
        for rva in a['MUTEXES']:
            mu.mem_write(BASE_ADDR + rva, b'\xC3')

        # Stop address
        mu.mem_map(align_down(STOP_ADDR), 0x1000, UC_PROT_ALL)
        mu.mem_write(STOP_ADDR, b'\xF4')

        # Unmapped memory handler — auto-map on access
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
            if address == BASE_ADDR + a['MALLOC_RVA']:
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
                    begin=BASE_ADDR + a['MALLOC_RVA'],
                    end=BASE_ADDR + a['MALLOC_RVA'] + 1)

        self.mu = mu
        self._write_log = []

    def _init_struct(self):
        """Initialize the packet struct before deserialization."""
        a = self.addrs
        struct_size = a.get('STRUCT_SIZE', 0x400)
        self.mu.mem_write(HEAP_BASE, b'\x00' * max(struct_size, 0x400))

        if a.get('CONSTRUCTOR') is not None:
            # 16.7+: run the full constructor
            self.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE)
            self.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
            rsp = self.mu.reg_read(UC_X86_REG_RSP) - 8
            self.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
            self.mu.reg_write(UC_X86_REG_RSP, rsp)
            try:
                self.mu.emu_start(BASE_ADDR + a['CONSTRUCTOR'], STOP_ADDR, timeout=5000000)
            except:
                pass
            # Save constructor state for reuse
            self._ctor_data = bytes(self.mu.mem_read(HEAP_BASE, struct_size))
        elif a['SUB_INIT_RVA'] is not None:
            sub_off = a.get('SUB_INIT_OFF', 0x10)  # 16.3=0x10, 16.4=0x18
            self.mu.reg_write(UC_X86_REG_RCX, HEAP_BASE + sub_off)
            self.mu.reg_write(UC_X86_REG_RSP, STACK_BASE + STACK_SIZE - 0x200)
            rsp = self.mu.reg_read(UC_X86_REG_RSP) - 8
            self.mu.mem_write(rsp, struct.pack('<Q', STOP_ADDR))
            self.mu.reg_write(UC_X86_REG_RSP, rsp)
            try:
                self.mu.emu_start(BASE_ADDR + a['SUB_INIT_RVA'], STOP_ADDR, timeout=5000000)
            except:
                pass
            # Save init state so we can restore it between decodes
            self._sub_init_data = bytes(self.mu.mem_read(HEAP_BASE + sub_off, 0x30))
            self._sub_init_off = sub_off

        # Set main vtable at struct+0x00
        self.mu.mem_write(HEAP_BASE, struct.pack('<Q', BASE_ADDR + a['MAIN_VTABLE']))

    def _track_writes(self, mu, access, address, size, value, user_data):
        val = value & ((1 << (size * 8)) - 1)
        off = address - ALLOC_BASE
        if 0 <= off < ALLOC_SIZE:
            self._write_log.append((off, size, val))
        soff = address - HEAP_BASE
        if 0 <= soff < 0x400:
            self._write_log.append((soff + 0x100000, size, val))
        return True

    def decode(self, payload):
        """Decode a movement payload. Returns dict with movement data."""
        self._write_log = []
        self._alloc_cursor = 0
        self._alloc_regions = []
        self.mu.mem_write(ALLOC_BASE, b'\x00' * min(0x8000, ALLOC_SIZE))

        # Reset struct
        struct_size = self.addrs.get('STRUCT_SIZE', 0x100)
        self.mu.mem_write(HEAP_BASE, b'\x00' * max(struct_size, 0x100))
        if hasattr(self, '_ctor_data'):
            # 16.7+: restore full constructor state
            self.mu.mem_write(HEAP_BASE, self._ctor_data)
        else:
            self.mu.mem_write(HEAP_BASE, struct.pack('<Q', BASE_ADDR + self.addrs['MAIN_VTABLE']))
            if hasattr(self, '_sub_init_data'):
                self.mu.mem_write(HEAP_BASE + self._sub_init_off, self._sub_init_data)

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
                              begin=HEAP_BASE, end=HEAP_BASE + 0x400)
        try:
            self.mu.emu_start(BASE_ADDR + self.addrs['DESER_RVA'], STOP_ADDR, timeout=60000000)
        except Exception as e:
            print(f"  Emulation error: {e}")
        self.mu.hook_del(h1)
        self.mu.hook_del(h2)

        return self._extract_fields()

    def _extract_fields(self):
        """Extract decoded fields from write log.

        Struct layouts differ between patches:
        - 16.3: alloc buf offsets: entity=+0x8, curX=+0x1c, curZ=+0x20, curY=+0x24,
                 seq=+0x28, speed=+0x2c, destX=+0x54, destZ=+0x58, destY=+0x5c,
                 time=+0x10c. Champion name in alloc[1].
        - 16.4: HEAP offsets: curX=+0x10, curY=+0x14. Sub-object: destX=+0x20,
                 destZ=+0x24, destY=+0x28. Alloc buf offsets: entity=+0x94,
                 curZ=+0x10, speed=+0x60, time=+0x98. Champion name in alloc[2].
                 Small packets: HEAP +0x10=X, +0x14=Y only (no alloc).
        """
        result = {}
        patch_16_4 = self.addrs.get('SUB_INIT_OFF') == 0x18
        patch_16_7 = self.patch == '16.7'

        # Helper to get last 4-byte write at an offset from a write dict
        def _get_u32(writes_by_off, offset):
            if offset in writes_by_off:
                mb = [(sz, val) for sz, val in writes_by_off[offset] if sz >= 4]
                if mb:
                    return mb[-1][1]
            return None

        def _get_f32(writes_by_off, offset):
            v = _get_u32(writes_by_off, offset)
            if v is not None:
                return struct.unpack('<f', struct.pack('<I', v & 0xFFFFFFFF))[0]
            return None

        # Build write maps for HEAP and ALLOC regions
        heap_writes = defaultdict(list)  # struct offset -> [(size, val)]
        alloc_writes = defaultdict(list)  # absolute alloc offset -> [(size, val)]
        for off, sz, val in self._write_log:
            if off >= 0x100000:
                heap_writes[off - 0x100000].append((sz, val))
            else:
                alloc_writes[off].append((sz, val))

        if self._alloc_regions:
            main_off, main_sz = self._alloc_regions[0]
            # Build relative offset map for main alloc buffer
            buf_writes = defaultdict(list)
            for off, writes in alloc_writes.items():
                rel = off - main_off
                if 0 <= rel < main_sz + 64:
                    buf_writes[rel].extend(writes)

            if patch_16_7:
                # 16.7: packed binary buffer (52 bytes), byte-by-byte writes.
                # Reconstruct full buffer from individual byte writes.
                buf = bytearray(max(main_sz, 64))
                for rel_off, writes in buf_writes.items():
                    if 0 <= rel_off < len(buf):
                        for sz, val in writes:
                            for byte_i in range(sz):
                                if rel_off + byte_i < len(buf):
                                    buf[rel_off + byte_i] = (val >> (byte_i * 8)) & 0xFF

                # 16.7 buffer layout (52 bytes):
                # +0x00: u16 flags (0x01C8=456 normal, 0x0056=86 spawn)
                # +0x04: u32 entity_id
                # +0x08: u32 packed position: x=bits[0:13], y=bits[14:27], flags=bits[28:31]
                #        map_coord = grid_val * (14914.0 / 16384)
                #        pos_flags: 0=normal movement, 4=spawn/init position
                # +0x0C: u32 destination/waypoint data (same 14-bit packing)
                # +0x10: u32 entity_id (source)
                # +0x14: u32 entity_id (target)
                # +0x18: f32 game_time
                # +0x1C: u32 waypoint/dest data (14-bit packed) or f32 speed (spawn packets)
                # +0x20: u32 waypoint_count / flags
                # +0x24: u32 dest entity
                # +0x28: 4B destination data (variant, same 14-bit packing)
                # +0x2C: u32 flags (0xFFFFFFFF = sentinel)
                # +0x30: u32 sequence/type
                MAP_SIZE = 14914.0
                GRID_SCALE = MAP_SIZE / 16384.0

                result['entity_id'] = struct.unpack_from('<I', buf, 0x04)[0]
                result['game_time'] = struct.unpack_from('<f', buf, 0x18)[0]
                result['flags'] = struct.unpack_from('<H', buf, 0x00)[0]

                # Position data at +0x08: 14-bit packed u32
                # x = u32 & 0x3FFF, y = (u32 >> 14) & 0x3FFF, flags = (u32 >> 28) & 0xF
                pos_u32 = struct.unpack_from('<I', buf, 0x08)[0]
                if pos_u32 != 0:
                    result['current_x'] = (pos_u32 & 0x3FFF) * GRID_SCALE
                    result['current_y'] = ((pos_u32 >> 14) & 0x3FFF) * GRID_SCALE
                    result['pos_flags'] = (pos_u32 >> 28) & 0xF
                    result['has_destination'] = True
                else:
                    result['current_x'] = None
                    result['current_y'] = None
                    result['pos_flags'] = None
                    result['has_destination'] = False

                # Destination at +0x0C (same 14-bit packing)
                dest_u32 = struct.unpack_from('<I', buf, 0x0C)[0]
                result['dest_raw'] = dest_u32
                if dest_u32 != 0 and dest_u32 != 0xFFFFFFFF:
                    result['dest_x'] = (dest_u32 & 0x3FFF) * GRID_SCALE
                    result['dest_y'] = ((dest_u32 >> 14) & 0x3FFF) * GRID_SCALE
                else:
                    result['dest_x'] = None
                    result['dest_y'] = None

                # Destination variant at +0x28 (same 14-bit packing)
                dest2_u32 = struct.unpack_from('<I', buf, 0x28)[0]
                result['dest2_raw'] = dest2_u32

                # +0x1C: speed (f32) for spawn packets, or waypoint data for normal
                if result.get('pos_flags') == 4:
                    result['speed'] = struct.unpack_from('<f', buf, 0x1C)[0]
                else:
                    wp_u32 = struct.unpack_from('<I', buf, 0x1C)[0]
                    if wp_u32 != 0:
                        result['waypoint_x'] = (wp_u32 & 0x3FFF) * GRID_SCALE
                        result['waypoint_y'] = ((wp_u32 >> 14) & 0x3FFF) * GRID_SCALE
                    result['speed'] = None

                result['waypoint_count'] = struct.unpack_from('<I', buf, 0x20)[0]
                result['sequence'] = struct.unpack_from('<I', buf, 0x30)[0]

                # Raw buffer for further analysis
                result['raw_buffer'] = buf[:main_sz].hex()
                result['current_z'] = None
                result['dest_z'] = None
                result['champion_name'] = None
                return result

            elif patch_16_4:
                # 16.4 layout
                result['entity_id'] = _get_u32(buf_writes, 0x094)
                result['current_x'] = _get_f32(heap_writes, 0x10)
                result['current_z'] = _get_f32(buf_writes, 0x010)
                result['current_y'] = _get_f32(heap_writes, 0x14)
                result['speed'] = _get_f32(buf_writes, 0x060)
                result['dest_x'] = _get_f32(heap_writes, 0x20)
                result['dest_z'] = _get_f32(heap_writes, 0x24)
                result['dest_y'] = _get_f32(heap_writes, 0x28)
                result['game_time'] = _get_f32(buf_writes, 0x098)
                result['sequence'] = _get_u32(buf_writes, 0x0f0)
                # Champion name in alloc[2]
                name_alloc_idx = 2
            else:
                # 16.3 layout
                result['entity_id'] = _get_u32(buf_writes, 0x008)
                result['current_x'] = _get_f32(buf_writes, 0x01c)
                result['current_z'] = _get_f32(buf_writes, 0x020)
                result['current_y'] = _get_f32(buf_writes, 0x024)
                result['speed'] = _get_f32(buf_writes, 0x02c)
                result['dest_x'] = _get_f32(buf_writes, 0x054)
                result['dest_z'] = _get_f32(buf_writes, 0x058)
                result['dest_y'] = _get_f32(buf_writes, 0x05c)
                result['game_time'] = _get_f32(buf_writes, 0x10c)
                result['sequence'] = _get_u32(buf_writes, 0x028)
                # Champion name in alloc[1]
                name_alloc_idx = 1

            result['has_destination'] = True

            if len(self._alloc_regions) > name_alloc_idx:
                name_off, name_sz = self._alloc_regions[name_alloc_idx]
                by_byte = defaultdict(list)
                for off, writes in alloc_writes.items():
                    rel = off - name_off
                    if 0 <= rel < name_sz:
                        for sz, val in writes:
                            if sz == 1:
                                by_byte[rel].append(val)
                name = ''
                for i in range(name_sz):
                    if i in by_byte:
                        c = by_byte[i][-1]
                        if 32 <= c < 127:
                            name += chr(c)
                result['champion_name'] = name
        else:
            # Small packet (no alloc) — position only
            if patch_16_7:
                # 16.7: small packet, HEAP-only data
                result['has_destination'] = False
                result['current_x'] = None
                result['current_y'] = None
                result['entity_id'] = None
                result['game_time'] = None
                return result
            elif patch_16_4:
                # 16.4: position in HEAP struct +0x10 (X) and +0x14 (Y)
                result['current_x'] = _get_f32(heap_writes, 0x10)
                result['current_y'] = _get_f32(heap_writes, 0x14)
                # Sub-object at +0x20/+0x24/+0x28 may have coords too
                result['dest_x'] = _get_f32(heap_writes, 0x20)
                result['dest_z'] = _get_f32(heap_writes, 0x24)
                result['dest_y'] = _get_f32(heap_writes, 0x28)
            else:
                # 16.3: position in sub-object relative to struct+0x10
                result['current_x'] = _get_f32(heap_writes, 0x10 + 0x24)
                result['current_z'] = _get_f32(heap_writes, 0x10 + 0x28)
                result['current_y'] = _get_f32(heap_writes, 0x10 + 0x2c)
            result['has_destination'] = False

        return result

    def dump_writes(self, payload):
        """Decode and dump raw write log for structure analysis."""
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
                              begin=HEAP_BASE, end=HEAP_BASE + 0x400)
        try:
            self.mu.emu_start(BASE_ADDR + self.addrs['DESER_RVA'], STOP_ADDR, timeout=60000000)
        except Exception as e:
            print(f"  Emulation error: {e}")
        self.mu.hook_del(h1)
        self.mu.hook_del(h2)

        print(f"\n  Alloc regions: {len(self._alloc_regions)}")
        for i, (off, sz) in enumerate(self._alloc_regions):
            print(f"    [{i}] offset=0x{off:x} size={sz} (0x{sz:x})")

        print(f"\n  Write log ({len(self._write_log)} entries):")
        for off, sz, val in self._write_log:
            region = "HEAP" if off >= 0x100000 else "ALLOC"
            real_off = off - 0x100000 if off >= 0x100000 else off
            if sz == 4:
                fval = struct.unpack('<f', struct.pack('<I', val & 0xFFFFFFFF))[0]
                print(f"    {region} +0x{real_off:04x} [{sz}B] = 0x{val:08x}  (f32={fval:.3f})")
            elif sz == 8:
                print(f"    {region} +0x{real_off:04x} [{sz}B] = 0x{val:016x}")
            else:
                print(f"    {region} +0x{real_off:04x} [{sz}B] = 0x{val:0{sz*2}x}")


# --- Block parser v2 (variable-header, self-healing) ---
# Block format: contiguous blocks with NO marker bytes. ALL byte values are valid
# block types. Header size is determined by bits 5-6 of byte[0]:
#   b6b5=00: 9 bytes (TYPE + CH + SIZE + PID(2) + PARAM(4))
#   b6b5=01: 6 bytes (TYPE + CH + SIZE + PID(2) + FLAG(1))
#   b6b5=10: 7 bytes (TYPE + CH + SIZE + PID(2) + FLAGS(2))
#   b6b5=11: 4 bytes (TYPE + CH + SIZE + FLAG(1))
# Channel byte (byte[1]) determines block category:
#   0x00: stat/property data (primary channel, well-chained)
#   0x21: position/movement data (cleartext coordinates in payloads)
#   0x22: additional game data
# PID dispatch table has 820 entries, max PID = 1164.
VALID_CHANNELS = {0x00, 0x21, 0x22}
MAX_VALID_PID = 1164

def _get_block_header_size(type_byte):
    """Return block header size based on bits 5-6 of the type byte."""
    return {0: 9, 1: 6, 2: 7, 3: 4}[(type_byte >> 5) & 3]


def _parse_frame_blocks(frame_data, start=15, max_bad_streak=5):
    """Parse blocks from a decompressed ROFL frame using self-healing resync.

    Walks contiguous variable-header blocks. When a streak of invalid blocks
    (channel != 0x00 or PID out of range) is hit, scans forward up to 500 bytes
    looking for a valid resynchronization point (two consecutive blocks with
    channel=0x00).

    Skips 0xfa padding regions that separate sub-sections in large frames.

    Returns (blocks, resync_count).
    """
    fd = frame_data
    pos = start
    blocks = []
    bad_streak = 0
    resyncs = 0

    while pos + 4 <= len(fd):
        # Skip 0xfa padding regions
        if fd[pos] == 0xfa and pos + 1 < len(fd) and fd[pos + 1] == 0xfa:
            while pos < len(fd) and fd[pos] == 0xfa:
                pos += 1
            continue

        tb = fd[pos]
        b6b5 = (tb >> 5) & 3
        hdr = _get_block_header_size(tb)
        ch = fd[pos + 1] if pos + 1 < len(fd) else 0xFF

        if pos + hdr > len(fd):
            break
        size = fd[pos + 2]
        end = pos + hdr + size
        if end > len(fd):
            break

        is_valid = (ch == 0x00)
        if b6b5 in (0, 1, 2) and ch == 0x00:
            pid = struct.unpack_from('<H', fd, pos + 3)[0]
            if pid > MAX_VALID_PID:
                is_valid = False

        if is_valid:
            bad_streak = 0
            pid = None
            if b6b5 in (0, 1, 2):
                pid = struct.unpack_from('<H', fd, pos + 3)[0]
            blocks.append({
                'pos': pos, 'type': tb, 'b6b5': b6b5,
                'hdr': hdr, 'sz': size, 'pid': pid,
                'payload': fd[pos + hdr:end],
            })
            pos = end
        else:
            bad_streak += 1
            if bad_streak >= max_bad_streak:
                # Resync: scan forward for two consecutive valid blocks
                found = False
                for scan in range(pos + 1, min(pos + 500, len(fd) - 4)):
                    if fd[scan + 1] == 0x00:
                        tb2 = fd[scan]
                        hdr2 = _get_block_header_size(tb2)
                        if scan + hdr2 <= len(fd):
                            sz2 = fd[scan + 2]
                            end2 = scan + hdr2 + sz2
                            if end2 <= len(fd) and end2 + 1 < len(fd) and fd[end2 + 1] == 0x00:
                                pos = scan
                                bad_streak = 0
                                resyncs += 1
                                found = True
                                break
                if not found:
                    pos += 1
                    bad_streak = 0
            else:
                pos = end

    return blocks, resyncs


def _parse_frame_blocks_multichannel(frame_data, start=15, channels=None):
    """Parse blocks accepting multiple channel values.

    Two-pass approach:
    1. Parse ch=0x00 blocks with self-healing (high confidence)
    2. Scan gaps for blocks with other valid channels (ch=0x21, 0x22)

    Returns (all_blocks, resync_count).
    """
    if channels is None:
        channels = VALID_CHANNELS

    # Pass 1: ch=0x00 blocks (self-healing)
    ch0_blocks, resyncs = _parse_frame_blocks(frame_data, start)

    if channels == {0x00}:
        return ch0_blocks, resyncs

    # Build coverage set from ch=0x00 blocks
    fd = frame_data
    covered_ends = []
    for b in ch0_blocks:
        covered_ends.append((b['pos'], b['pos'] + b['hdr'] + b['sz']))

    # Pass 2: scan gaps for other-channel blocks
    other_channels = channels - {0x00}
    gap_blocks = []
    gap_regions = []

    prev_end = start
    for bstart, bend in covered_ends:
        if bstart > prev_end:
            gap_regions.append((prev_end, bstart))
        prev_end = bend
    if prev_end < len(fd):
        gap_regions.append((prev_end, len(fd)))

    for gap_start, gap_end in gap_regions:
        pos = gap_start
        while pos + 4 <= gap_end:
            if fd[pos] == 0xfa and pos + 1 < len(fd) and fd[pos + 1] == 0xfa:
                while pos < len(fd) and fd[pos] == 0xfa:
                    pos += 1
                continue

            tb = fd[pos]
            b6b5 = (tb >> 5) & 3
            hdr = _get_block_header_size(tb)
            ch = fd[pos + 1] if pos + 1 < len(fd) else 0xFF

            if pos + hdr > len(fd):
                pos += 1
                continue
            size = fd[pos + 2]
            end = pos + hdr + size
            if end > len(fd):
                pos += 1
                continue

            if ch in other_channels:
                pid = None
                if b6b5 in (0, 1, 2) and pos + 5 <= len(fd):
                    pid = struct.unpack_from('<H', fd, pos + 3)[0]
                if pid is None or pid <= MAX_VALID_PID:
                    gap_blocks.append({
                        'pos': pos, 'type': tb, 'b6b5': b6b5,
                        'hdr': hdr, 'sz': size, 'pid': pid,
                        'ch': ch, 'payload': fd[pos + hdr:end],
                    })
                    pos = end
                    continue
            pos += 1

    # Add channel info to ch=0x00 blocks and merge
    for b in ch0_blocks:
        b['ch'] = 0x00
    all_blocks = ch0_blocks + gap_blocks
    all_blocks.sort(key=lambda b: b['pos'])
    return all_blocks, resyncs


def parse_rofl_blocks(rofl_path, netid_filter=None, param_filter=None):
    """Parse blocks from a ROFL2 replay file.

    Uses the v2 variable-header self-healing parser. Reads ROFL container
    framing (per-frame metadata) to correctly locate zstd compressed data.
    When netid_filter/param_filter are given, filters the parsed blocks.
    """
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

    print(f"  Found {len(frames)} zstd frames")

    blocks = []
    for frame_data in frames:
        frame_blocks, _ = _parse_frame_blocks(frame_data)
        for blk in frame_blocks:
            pid = blk['pid']
            # For b6b5=00 blocks, extract param (4 bytes at offset +5)
            if blk['b6b5'] == 0 and blk['hdr'] == 9 and blk['pos'] + 9 <= len(frame_data):
                param = struct.unpack_from('<I', frame_data, blk['pos'] + 5)[0]
            else:
                param = None

            if netid_filter is not None and pid != netid_filter:
                continue
            if param_filter is not None and param not in param_filter:
                continue

            blocks.append({
                'packet_id': pid,
                'param': param,
                'payload': blk['payload'],
                'marker': blk['type'],  # compat: old code called this 'marker'
                'channel': 0x00,  # all valid blocks have ch=0x00
            })

    return blocks


def analyze_pids(replay_path):
    """Analyze all PIDs in a replay to find movement candidates."""
    PLAYER_ID_START = 0x400000ae
    hero_params = set(range(PLAYER_ID_START, PLAYER_ID_START + 10))

    print(f"Analyzing PIDs in: {replay_path}")
    all_blocks = parse_rofl_blocks(replay_path)
    print(f"Total blocks parsed: {len(all_blocks)}")

    # Count hero blocks by PID
    pid_counts = defaultdict(int)
    pid_entities = defaultdict(set)
    pid_sizes = defaultdict(list)
    for b in all_blocks:
        if b['param'] in hero_params:
            pid = b['packet_id']
            pid_counts[pid] += 1
            pid_entities[pid].add(b['param'])
            pid_sizes[pid].append(len(b['payload']))

    # Movement characteristics:
    # - Should have blocks for MOST/ALL 10 entities (not skewed to one)
    # - Should have high total count (thousands)
    # - Should have variable payload sizes (small ~31B for sync, large ~100-147B with dest)
    print(f"\n{'PID':>6} {'Count':>6} {'Entities':>8} {'Sizes':>20}  Notes")
    print("-" * 80)
    for pid in sorted(pid_counts, key=pid_counts.get, reverse=True)[:30]:
        cnt = pid_counts[pid]
        n_ent = len(pid_entities[pid])
        sizes = pid_sizes[pid]
        size_str = f"min={min(sizes)} max={max(sizes)} med={sorted(sizes)[len(sizes)//2]}"
        notes = []
        if n_ent >= 8:
            notes.append("ALL_ENTITIES")
        if min(sizes) < 40 and max(sizes) > 80:
            notes.append("VARIABLE_SIZE")
        if cnt > 1000:
            notes.append("HIGH_COUNT")
        note_str = " ".join(notes)
        if n_ent >= 8 or cnt > 500:
            note_str = "  *** " + note_str
        print(f"{pid:>6} {cnt:>6} {n_ent:>8} {size_str:>20}  {note_str}")

    # Also show per-entity distribution for top candidates
    print("\n--- Per-entity distribution for top candidates ---")
    for pid in sorted(pid_counts, key=pid_counts.get, reverse=True)[:10]:
        if len(pid_entities[pid]) < 5:
            continue
        ent_counts = defaultdict(int)
        for b in all_blocks:
            if b['param'] in hero_params and b['packet_id'] == pid:
                ent_counts[b['param']] += 1
        dist = " ".join(f"{e & 0xFF:02x}:{c}" for e, c in sorted(ent_counts.items()))
        print(f"  pid={pid}: {dist}")


def main():
    import sys

    replay_path = sys.argv[1] if len(sys.argv) > 1 else 'data/replays/NA1-5528508560.rofl'
    patch = sys.argv[2] if len(sys.argv) > 2 else '16.7'

    # If --analyze flag, just do PID analysis
    if '--analyze' in sys.argv:
        analyze_pids(replay_path)
        return

    addrs = PATCH_16_7 if patch == '16.7' else (PATCH_16_4 if patch == '16.4' else PATCH_16_3)
    movement_netid = int(sys.argv[3]) if len(sys.argv) > 3 else addrs['MOVEMENT_NETID']

    print(f"Replay: {replay_path}")
    print(f"Patch: {patch}, movement netid: {movement_netid}")

    PLAYER_ID_START = 0x400000ae
    hero_params = set(range(PLAYER_ID_START, PLAYER_ID_START + 10))

    print("Parsing replay blocks...")
    mov_blocks = parse_rofl_blocks(replay_path, netid_filter=movement_netid,
                                    param_filter=hero_params)
    print(f"Found {len(mov_blocks)} movement blocks for heroes")

    if not mov_blocks:
        print("No movement blocks found! Run with --analyze to see all PIDs.")
        return

    # Per-entity count
    ent_counts = defaultdict(int)
    for b in mov_blocks:
        ent_counts[b['param']] += 1
    print(f"Per entity: {' '.join(f'{e & 0xFF:02x}:{c}' for e, c in sorted(ent_counts.items()))}")

    # Dump first few blocks raw to understand structure
    # Select PE dump directory based on patch
    pe_dir = f'/tmp/pe_dump_{patch}' if patch >= '16.7' else '/tmp/pe_dump'
    import os
    if not os.path.exists(os.path.join(pe_dir, 'text.bin')):
        pe_dir = '/tmp/pe_dump'  # fallback
    print(f"\nInitializing decoder (patch {patch}, PE dumps: {pe_dir})...")
    decoder = MovementDecoder(
        text_path=os.path.join(pe_dir, 'text.bin'),
        rdata_path=os.path.join(pe_dir, 'rdata.bin'),
        data_path=os.path.join(pe_dir, 'data.bin'),
        patch=patch,
    )

    print("\n--- Raw write dump for first 3 large blocks ---")
    large = [b for b in mov_blocks if len(b['payload']) > 50]
    for i, b in enumerate(large[:3]):
        print(f"\nBlock {i}: param=0x{b['param']:x} payload={len(b['payload'])}B ch=0x{b['channel']:02x}")
        print(f"  Payload hex: {b['payload'][:64].hex()}")
        decoder.dump_writes(b['payload'])

    # Decode all
    print(f"\n--- Decoding all {len(mov_blocks)} blocks ---")
    movements = []
    errors = 0

    for i, b in enumerate(mov_blocks):
        if i % 100 == 0:
            print(f"  Decoding {i}/{len(mov_blocks)}...", end='\r')
        try:
            result = decoder.decode(b['payload'])
            result['block_param'] = b['param']
            result['payload_size'] = len(b['payload'])
            movements.append(result)
        except Exception as e:
            errors += 1

    print(f"\nDecoded {len(movements)} movements ({errors} errors)")

    by_entity = defaultdict(list)
    for m in movements:
        eid = m.get('entity_id', m.get('block_param'))
        by_entity[eid].append(m)

    print(f"\nPer-entity summary:")
    for eid in sorted(by_entity.keys()):
        ems = by_entity[eid]
        with_dest = sum(1 for m in ems if m.get('has_destination'))
        champ = next((m.get('champion_name', '') for m in ems if m.get('champion_name')), '?')
        xs = [m['current_x'] for m in ems if m.get('current_x') is not None]
        ys = [m['current_y'] for m in ems if m.get('current_y') is not None]
        coord_str = (f"X=[{min(xs):.0f}-{max(xs):.0f}], Y=[{min(ys):.0f}-{max(ys):.0f}]"
                     if xs and ys else "NO COORDS")
        print(f"  0x{eid:x} ({champ:15s}): {len(ems):4d} updates, "
              f"{with_dest:3d} with dest, {coord_str}")

    out_path = '/tmp/movement_data.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(movements, f)
    print(f"\nSaved to {out_path}")

    json_path = '/tmp/movement_data_sample.json'
    sample = movements[:50]
    with open(json_path, 'w') as f:
        json.dump(sample, f, indent=2, default=str)
    print(f"Sample saved to {json_path}")


if __name__ == '__main__':
    main()
