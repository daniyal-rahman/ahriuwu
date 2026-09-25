"""
Approach the AiManager from two angles:
1. Disassemble GetAiManager function at module+0x292420 to understand pointer resolution
2. Raw dump hero+0x41F0 chain to see what's actually there
3. Scan hero struct for pointers to structs containing velocity-like floats (300-500)
4. Try calling GetAiManager via the vfunc/wrapper pattern
"""

import ctypes
import ctypes.wintypes
import struct
import sys
import time
from ctypes import wintypes

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
TH32CS_SNAPPROCESS = 0x00000002
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
MAX_MODULE_NAME32 = 255
MAX_PATH = 260
HERO_ARRAY_RVA = 0x1DBEEE8

class PROCESSENTRY32(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD),("cntUsage", wintypes.DWORD),("th32ProcessID", wintypes.DWORD),("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),("th32ModuleID", wintypes.DWORD),("cntThreads", wintypes.DWORD),("th32ParentProcessID", wintypes.DWORD),("pcPriClassBase", ctypes.c_long),("dwFlags", wintypes.DWORD),("szExeFile", ctypes.c_char * MAX_PATH)]

class MODULEENTRY32(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD),("th32ModuleID", wintypes.DWORD),("th32ProcessID", wintypes.DWORD),("GlblcntUsage", wintypes.DWORD),("ProccntUsage", wintypes.DWORD),("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),("modBaseSize", wintypes.DWORD),("hModule", wintypes.HMODULE),("szModule", ctypes.c_char * (MAX_MODULE_NAME32 + 1)),("szExePath", ctypes.c_char * MAX_PATH)]

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

def find_league():
    s = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    pe = PROCESSENTRY32(); pe.dwSize = ctypes.sizeof(PROCESSENTRY32)
    kernel32.Process32First(s, ctypes.byref(pe))
    while True:
        if b"League of Legends" in pe.szExeFile:
            pid = pe.th32ProcessID; kernel32.CloseHandle(s); return pid
        if not kernel32.Process32Next(s, ctypes.byref(pe)): break
    kernel32.CloseHandle(s); return None

def find_base(pid):
    s = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    me = MODULEENTRY32(); me.dwSize = ctypes.sizeof(MODULEENTRY32)
    kernel32.Module32First(s, ctypes.byref(me))
    while True:
        if b"league of legends" in me.szModule.lower():
            b = ctypes.cast(me.modBaseAddr, ctypes.c_void_p).value; sz = me.modBaseSize; kernel32.CloseHandle(s); return b, sz
        if not kernel32.Module32Next(s, ctypes.byref(me)): break
    kernel32.CloseHandle(s); return None, None

class Mem:
    def __init__(s, pid):
        s.h = kernel32.OpenProcess(PROCESS_VM_READ|PROCESS_QUERY_INFORMATION, False, pid)
    def read(s, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = kernel32.ReadProcessMemory(s.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u32(s,a): d=s.read(a,4); return struct.unpack("<I",d)[0] if d else None
    def u64(s,a): d=s.read(a,8); return struct.unpack("<Q",d)[0] if d else None
    def f32(s,a): d=s.read(a,4); return struct.unpack("<f",d)[0] if d else None
    def vec3(s,a): d=s.read(a,12); return struct.unpack("<fff",d) if d else None
    def string(s,a,n=64):
        d=s.read(a,n)
        if not d: return None
        return d.split(b'\x00')[0].decode('ascii',errors='replace') or None
    def close(s): kernel32.CloseHandle(s.h)

def is_heap(v): return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF
def is_map_pos(v):
    if v is None: return False
    x, y, z = v
    return (100 < x < 15000 and -500 < y < 1000 and 100 < z < 15000)

def disasm_simple(data, base_addr):
    """Very basic x86-64 disassembly - just enough to identify mov/lea patterns."""
    lines = []
    i = 0
    while i < len(data) and len(lines) < 40:
        addr = base_addr + i
        b = data[i]

        # Common x86-64 patterns
        if b == 0xC3:
            lines.append(f"  0x{addr:X}: ret")
            break
        elif b == 0xCC:
            lines.append(f"  0x{addr:X}: int3")
            break

        # Just show raw bytes with hex
        chunk = data[i:i+8]
        hex_str = ' '.join(f'{x:02X}' for x in chunk)
        lines.append(f"  0x{addr:X}: {hex_str}")
        i += 1

    return '\n'.join(lines)

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("=" * 70)
    print("AiManager Deep Dive - Function Disasm + Raw Pointer Analysis")
    print("=" * 70)

    pid = find_league()
    if not pid: print("League not found!"); return
    base, mod_size = find_base(pid)
    if not base: print("Module not found!"); return
    print(f"PID={pid} Base=0x{base:X} ModSize=0x{mod_size:X}")

    m = Mem(pid)
    assert m.read(base, 2) == b'MZ'

    arr_ptr = m.u64(base + HERO_ARRAY_RVA)
    heroes = []
    for i in range(10):
        hp = m.u64(arr_ptr + i * 8)
        if not is_heap(hp): continue
        name = m.string(hp + 0x4328, 32) or f"h{i}"
        pos = m.vec3(hp + 0x25C)
        heroes.append({"ptr": hp, "name": name, "pos": pos, "idx": i})

    print(f"Found {len(heroes)} heroes")

    # ================================================================
    # STEP 1: Dump GetAiManager function bytes
    # ================================================================
    print(f"\n{'='*70}")
    print("STEP 1: GetAiManager function at module+0x292420")
    print("=" * 70)

    # GetAiManager = 0x292420 (SEED offset)
    func_addr = base + 0x292420
    func_bytes = m.read(func_addr, 128)
    if func_bytes:
        print(f"  Address: 0x{func_addr:X}")
        for i in range(0, min(128, len(func_bytes)), 16):
            hex_str = ' '.join(f'{b:02X}' for b in func_bytes[i:i+16])
            print(f"  +{i:3d}: {hex_str}")

    # Also GetAiManagerInner = 0x293A10
    func2_addr = base + 0x293A10
    func2_bytes = m.read(func2_addr, 128)
    if func2_bytes:
        print(f"\n  GetAiManagerInner at 0x{func2_addr:X}:")
        for i in range(0, min(128, len(func2_bytes)), 16):
            hex_str = ' '.join(f'{b:02X}' for b in func2_bytes[i:i+16])
            print(f"  +{i:3d}: {hex_str}")

    # ================================================================
    # STEP 2: Raw dump of hero+0x41F0 chain
    # ================================================================
    print(f"\n{'='*70}")
    print("STEP 2: hero+0x41F0 raw chain analysis")
    print("=" * 70)

    for hero in heroes[:3]:
        hname = hero['name']
        hptr = hero['ptr']
        hpos = hero['pos']
        print(f"\n  {hname} (pos={hpos[0]:.0f},{hpos[2]:.0f}):")

        # Read the raw 8 bytes at hero+0x41F0
        raw = m.u64(hptr + 0x41F0)
        print(f"    hero+0x41F0 = 0x{raw:016X}" if raw else f"    hero+0x41F0 = NULL")

        if is_heap(raw):
            # Dump first 128 bytes
            data = m.read(raw, 128)
            if data:
                print(f"    Dump of ptr (0x{raw:X}):")
                for i in range(0, 128, 16):
                    hex_str = ' '.join(f'{b:02X}' for b in data[i:i+16])
                    # Also as pointers/floats
                    ptrs = []
                    for j in range(0, 16, 8):
                        if i+j+8 <= len(data):
                            p = struct.unpack("<Q", data[i+j:i+j+8])[0]
                            if is_heap(p):
                                ptrs.append(f"+0x{i+j:02X}:PTR")
                    floats = []
                    for j in range(0, 16, 4):
                        if i+j+4 <= len(data):
                            f = struct.unpack("<f", data[i+j:i+j+4])[0]
                            if 50 < abs(f) < 16000:
                                floats.append(f"+0x{i+j:02X}:{f:.0f}")
                    extra = " | ".join(ptrs + floats)
                    print(f"      +0x{i:03X}: {hex_str}  {extra}")

                # Try inner deref at +0x10
                inner = m.u64(raw + 0x10)
                if is_heap(inner):
                    print(f"\n    Inner at raw+0x10 = 0x{inner:X}:")
                    idata = m.read(inner, 256)
                    if idata:
                        for i in range(0, 256, 16):
                            hex_str = ' '.join(f'{b:02X}' for b in idata[i:i+16])
                            floats = []
                            for j in range(0, 16, 4):
                                if i+j+4 <= len(idata):
                                    f = struct.unpack("<f", idata[i+j:i+j+4])[0]
                                    if 50 < abs(f) < 16000:
                                        floats.append(f"+0x{i+j:02X}:{f:.0f}")
                            extra = " | ".join(floats)
                            print(f"      +0x{i:03X}: {hex_str}  {extra}")

    # ================================================================
    # STEP 3: Search hero struct for pointers to structs with velocity floats
    # ================================================================
    print(f"\n{'='*70}")
    print("STEP 3: Search for structs with move speed (300-500) float")
    print("=" * 70)

    test_hero = heroes[0]
    print(f"Testing: {test_hero['name']}")

    # Read the entire hero struct in one go
    hero_data = m.read(test_hero['ptr'], 0x6000)
    if not hero_data:
        print("Failed to read hero struct"); m.close(); return

    speed_candidates = []

    for off in range(0, len(hero_data) - 8, 8):
        ptr = struct.unpack("<Q", hero_data[off:off+8])[0]
        if not is_heap(ptr): continue

        # Read the pointed-to struct
        target = m.read(ptr, 0x500)
        if not target: continue

        # Scan for velocity-like float (300-500)
        for sub in range(0, len(target) - 4, 4):
            vel = struct.unpack("<f", target[sub:sub+4])[0]
            if 250 < vel < 600:
                # Also check if there's a map position nearby
                has_pos = False
                for pos_off in [sub-12, sub+4, sub+16, 0, 0x34, 0x330, 0x33C, 0x474]:
                    if 0 <= pos_off < len(target) - 12:
                        v = struct.unpack("<fff", target[pos_off:pos_off+12])
                        if is_map_pos(v):
                            has_pos = True
                            break

                if has_pos:
                    speed_candidates.append((off, ptr, sub, vel))

        # Also try inner deref
        if len(target) >= 0x18:
            for inner_off in [0x08, 0x10, 0x18]:
                inner = struct.unpack("<Q", target[inner_off:inner_off+8])[0]
                if not is_heap(inner): continue
                inner_data = m.read(inner, 0x500)
                if not inner_data: continue

                for sub in range(0, len(inner_data) - 4, 4):
                    vel = struct.unpack("<f", inner_data[sub:sub+4])[0]
                    if 250 < vel < 600:
                        has_pos = False
                        for pos_off in [sub-12, sub+4, sub+16, 0, 0x34, 0x330, 0x33C, 0x474]:
                            if 0 <= pos_off < len(inner_data) - 12:
                                v = struct.unpack("<fff", inner_data[pos_off:pos_off+12])
                                if is_map_pos(v):
                                    has_pos = True
                                    break
                        if has_pos:
                            speed_candidates.append((off, inner, sub, vel, f"via+0x{inner_off:X}"))

    # Deduplicate by target ptr
    seen = set()
    unique = []
    for c in speed_candidates:
        key = (c[1], c[2])
        if key not in seen:
            seen.add(key)
            unique.append(c)

    print(f"\nFound {len(unique)} candidates with speed float + nearby map position:")
    for c in unique[:30]:
        via = c[4] if len(c) > 4 else "direct"
        print(f"  hero+0x{c[0]:04X} -> 0x{c[1]:X} +0x{c[2]:03X} vel={c[3]:.1f} [{via}]")

        # Show surrounding context
        target_data = m.read(c[1], 0x500)
        if target_data:
            # Show the velocity and nearby floats
            sub = c[2]
            context_start = max(0, sub - 32)
            context_end = min(len(target_data), sub + 32)
            floats = []
            for j in range(context_start, context_end, 4):
                f = struct.unpack("<f", target_data[j:j+4])[0]
                marker = " <<VEL" if j == sub else ""
                if abs(f) < 100000:
                    floats.append(f"+0x{j:03X}={f:.1f}{marker}")
            print(f"    context: {', '.join(floats)}")

    # ================================================================
    # STEP 4: Alternative - read hero positions over time and compute velocity
    # ================================================================
    print(f"\n{'='*70}")
    print("STEP 4: Position-based velocity computation (2 samples, 0.5s apart)")
    print("=" * 70)

    snap1 = {}
    for h in heroes:
        snap1[h['name']] = m.vec3(h['ptr'] + 0x25C)

    time.sleep(0.5)

    snap2 = {}
    for h in heroes:
        snap2[h['name']] = m.vec3(h['ptr'] + 0x25C)

    for h in heroes:
        p1 = snap1[h['name']]
        p2 = snap2[h['name']]
        if p1 and p2:
            dx = p2[0] - p1[0]
            dz = p2[2] - p1[2]
            dist = (dx*dx + dz*dz)**0.5
            speed = dist / 0.5
            moving = "MOVING" if speed > 50 else "static"
            if speed > 50:
                # Extrapolate target (rough, assumes straight-line movement)
                # Project 2 seconds ahead
                proj_x = p2[0] + dx * 4  # 2s ahead at current velocity
                proj_z = p2[2] + dz * 4
                print(f"  {h['name']:12s} speed={speed:6.0f} dir=({dx:.0f},{dz:.0f}) pos=({p2[0]:.0f},{p2[2]:.0f}) proj2s=({proj_x:.0f},{proj_z:.0f})")
            else:
                print(f"  {h['name']:12s} {moving}")

    # ================================================================
    # STEP 5: ActiveSpell — try hero+0x3120 direct (SEED offset)
    # ================================================================
    print(f"\n{'='*70}")
    print("STEP 5: ActiveSpell via hero+0x3120 (SEED: SpellBook+ActiveSpellCast)")
    print("=" * 70)

    for hero in heroes:
        # hero + 0x3120 should be the ActiveSpellCast ptr
        active_ptr = m.u64(hero['ptr'] + 0x3120)
        if not is_heap(active_ptr):
            continue

        # Dump the active spell struct
        spell_data = m.read(active_ptr, 0x250)
        if not spell_data: continue

        # Scan ALL offsets for name strings
        names_found = []
        for sub_off in range(0, 0x100, 8):
            ptr_val = struct.unpack("<Q", spell_data[sub_off:sub_off+8])[0]
            if not is_heap(ptr_val): continue

            # Try as direct string
            name = m.string(ptr_val, 48)
            if name and len(name) >= 4 and all(c.isalnum() or c in '_' for c in name):
                names_found.append((sub_off, "direct", name))

            # Try deref -> string at +0x28 (SpellInfo -> SpellName)
            for n_off in [0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0x40]:
                ptr2 = m.u64(ptr_val + n_off)
                if is_heap(ptr2):
                    name2 = m.string(ptr2, 48)
                    if name2 and len(name2) >= 4 and all(c.isalnum() or c in '_' for c in name2):
                        names_found.append((sub_off, f"deref+0x{n_off:X}", name2))
                # Also try as inline string
                name3 = m.string(ptr_val + n_off, 48)
                if name3 and len(name3) >= 4 and all(c.isalnum() or c in '_' for c in name3):
                    names_found.append((sub_off, f"inline+0x{n_off:X}", name3))

        if names_found:
            print(f"\n  {hero['name']:12s} ActiveSpell at 0x{active_ptr:X}:")
            for sub, mode, name in names_found[:10]:
                print(f"    +0x{sub:03X} [{mode}]: '{name}'")

            # Also look for Vec3 positions in the spell struct
            for pos_off in [0xD0, 0xDC, 0xE8, 0x108, 0x114, 0x120, 0x1E0, 0x1E8, 0x1F4]:
                if pos_off + 12 <= len(spell_data):
                    v = struct.unpack("<fff", spell_data[pos_off:pos_off+12])
                    if is_map_pos(v):
                        print(f"    +0x{pos_off:03X}: pos=({v[0]:.0f},{v[1]:.0f},{v[2]:.0f})")

            # Check game time float
            for t_off in [0x1E8, 0x1F4, 0x200, 0x0C0, 0x0C4, 0x0C8]:
                if t_off + 4 <= len(spell_data):
                    t_val = struct.unpack("<f", spell_data[t_off:t_off+4])[0]
                    if 10 < t_val < 5000:
                        print(f"    +0x{t_off:03X}: time={t_val:.2f}s")

    # ================================================================
    # STEP 6: Try INCOMING spell at hero+0x4578 with more thorough chain
    # ================================================================
    print(f"\n{'='*70}")
    print("STEP 6: Incoming spell (hero+0x4578) deep analysis")
    print("=" * 70)

    for hero in heroes[:5]:
        spell_ptr = m.u64(hero['ptr'] + 0x4578)
        if not is_heap(spell_ptr):
            continue

        spell_data = m.read(spell_ptr, 0x250)
        if not spell_data: continue

        print(f"\n  {hero['name']:12s} hero+0x4578 -> 0x{spell_ptr:X}")

        # Dump first 128 bytes
        for i in range(0, 128, 16):
            hex_str = ' '.join(f'{b:02X}' for b in spell_data[i:i+16])
            print(f"    +0x{i:03X}: {hex_str}")

        # Scan for strings
        for sub_off in range(0, 0x100, 8):
            ptr_val = struct.unpack("<Q", spell_data[sub_off:sub_off+8])[0]
            if not is_heap(ptr_val): continue
            # Direct string
            name = m.string(ptr_val, 48)
            if name and len(name) >= 4 and all(c.isalnum() or c in '_' for c in name):
                print(f"    +0x{sub_off:03X} -> '{name}'")
            # Deref
            for n_off in [0x00, 0x08, 0x10, 0x18, 0x20, 0x28, 0x30]:
                ptr2 = m.u64(ptr_val + n_off)
                if is_heap(ptr2):
                    name2 = m.string(ptr2, 48)
                    if name2 and len(name2) >= 4 and all(c.isalnum() or c in '_' for c in name2):
                        print(f"    +0x{sub_off:03X} -> +0x{n_off:X} -> '{name2}'")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
