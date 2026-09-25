"""
AiManager inner struct offset scanner.

Scans offsets 0x00 to 0x500 in the AiManager inner struct for Vec3 values
that look like map positions and CHANGE over time (indicating movement target,
path waypoints, etc.)

Known working chain: hero + 0x4628 -> deref -> +0x10 -> inner struct
  inner + 0x000 = server/current position (confirmed)
  inner + 0x030 = reads (1.0, 1.0, 1.0) - WRONG for this build

We scan for the REAL movement target offset.
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
    """Valid LoL map coordinate: X in -500..16000, Y in -500..1000, Z in -500..16000"""
    if v is None: return False
    x, y, z = v
    return (-500 < x < 16000 and -500 < y < 1000 and -500 < z < 16000
            and not (x == 0 and z == 0)
            and not (abs(x - 1.0) < 0.01 and abs(z - 1.0) < 0.01))  # filter (1,1,1)

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    print("=" * 70)
    print("AiManager Inner Struct Offset Scanner")
    print("=" * 70)

    pid = find_league()
    if not pid: print("League not found!"); return
    base, mod_size = find_base(pid)
    if not base: print("Module not found!"); return
    print(f"PID={pid} Base=0x{base:X} ModSize=0x{mod_size:X}")

    m = Mem(pid)
    assert m.read(base, 2) == b'MZ', "RPM failed"

    # Read hero array
    arr_ptr = m.u64(base + HERO_ARRAY_RVA)
    if not is_heap(arr_ptr): print("Hero array NULL!"); return

    heroes = []
    for i in range(10):
        hp = m.u64(arr_ptr + i * 8)
        if not is_heap(hp): continue
        name = m.string(hp + 0x4328, 32) or f"h{i}"
        pos = m.vec3(hp + 0x25C)
        heroes.append({"ptr": hp, "name": name, "pos": pos, "idx": i})

    print(f"Found {len(heroes)} heroes")
    for h in heroes:
        p = h['pos']
        print(f"  [{h['idx']}] {h['name']:12s} pos=({p[0]:.0f},{p[1]:.0f},{p[2]:.0f})" if p else f"  [{h['idx']}] {h['name']:12s} pos=N/A")

    # Get AiManager inner ptrs for all heroes
    print(f"\n--- Resolving AiManager inner pointers ---")
    inner_ptrs = {}
    for h in heroes:
        ai_raw = m.u64(h['ptr'] + 0x4628)
        if not is_heap(ai_raw):
            print(f"  {h['name']}: ai_raw NULL"); continue
        inner = m.u64(ai_raw + 0x10)
        if not is_heap(inner):
            print(f"  {h['name']}: inner NULL"); continue
        inner_ptrs[h['name']] = inner
        # Verify inner+0x0 = server pos
        sp = m.vec3(inner)
        hp = h['pos']
        if sp and hp:
            dist = ((sp[0]-hp[0])**2 + (sp[2]-hp[2])**2)**0.5
            print(f"  {h['name']:12s} inner=0x{inner:X}  srv=({sp[0]:.0f},{sp[2]:.0f}) pos=({hp[0]:.0f},{hp[2]:.0f}) delta={dist:.1f}")

    if not inner_ptrs:
        print("No valid AiManager inner pointers found!"); m.close(); return

    # ================================================================
    # SCAN 1: Read full inner struct dump for all heroes, find map positions
    # ================================================================
    SCAN_SIZE = 0x500
    print(f"\n--- SCAN 1: All map-position Vec3 in inner struct (0x{SCAN_SIZE:X} bytes) ---")

    for hname, inner in list(inner_ptrs.items())[:3]:
        hero = [h for h in heroes if h['name'] == hname][0]
        hp = hero['pos']
        data = m.read(inner, SCAN_SIZE)
        if not data: continue

        print(f"\n  {hname} (pos={hp[0]:.0f},{hp[2]:.0f}):")
        for off in range(0, len(data) - 12, 4):
            v = struct.unpack("<fff", data[off:off+12])
            if is_map_pos(v):
                dx = v[0] - hp[0]; dz = v[2] - hp[2]
                dist = (dx*dx + dz*dz)**0.5
                tag = ""
                if dist < 5: tag = " <== SAME AS POS (server pos?)"
                elif dist < 50: tag = " <== VERY CLOSE"
                elif dist < 500: tag = " <== NEARBY"
                elif dist < 2000: tag = " <== MEDIUM RANGE"
                else: tag = " <== FAR"
                print(f"    +0x{off:03X}: ({v[0]:8.1f}, {v[1]:6.1f}, {v[2]:8.1f})  dist={dist:7.1f}{tag}")

    # ================================================================
    # SCAN 2: Temporal analysis — take snapshots 2s apart, find CHANGING fields
    # ================================================================
    print(f"\n\n{'='*70}")
    print("SCAN 2: Temporal analysis — 6 snapshots over 10 seconds")
    print("=" * 70)
    print("(Looking for Vec3 fields that CHANGE, indicating movement target/path)")

    snapshots = []
    for t in range(6):
        snap = {}
        for hname, inner in inner_ptrs.items():
            hero = [h for h in heroes if h['name'] == hname][0]
            hero_pos = m.vec3(hero['ptr'] + 0x25C)
            inner_data = m.read(inner, SCAN_SIZE)
            snap[hname] = {"pos": hero_pos, "data": inner_data}
        snapshots.append(snap)
        if t < 5:
            print(f"  Snapshot {t} taken, waiting 2s...")
            time.sleep(2)
        else:
            print(f"  Snapshot {t} taken (final)")

    # Analyze which offsets changed
    print(f"\n--- Fields that CHANGED over time ---")
    for hname in list(inner_ptrs.keys())[:5]:
        hero = [h for h in heroes if h['name'] == hname][0]
        first_pos = snapshots[0][hname]["pos"]
        last_pos = snapshots[-1][hname]["pos"]
        if first_pos and last_pos:
            pos_dist = ((last_pos[0]-first_pos[0])**2 + (last_pos[2]-first_pos[2])**2)**0.5
            moved = "MOVED" if pos_dist > 10 else "STATIC"
        else:
            pos_dist = 0; moved = "?"

        print(f"\n  {hname} [{moved}, dist={pos_dist:.0f}]:")

        all_data = [snapshots[t][hname]["data"] for t in range(6)]
        if not all(d is not None for d in all_data):
            print("    (missing data)"); continue

        changing_map = []
        changing_any = []

        for off in range(0, SCAN_SIZE - 12, 4):
            vals = [struct.unpack("<fff", d[off:off+12]) for d in all_data]
            # Check if any component changed significantly
            max_delta = max(
                abs(vals[t][c] - vals[0][c])
                for t in range(1, 6) for c in range(3)
            )

            if max_delta > 0.5:
                is_map = all(is_map_pos(v) for v in vals)
                if is_map:
                    changing_map.append((off, vals, max_delta))
                elif max_delta > 1.0:
                    # Show non-map changes too but summarize
                    changing_any.append((off, vals[0], vals[-1], max_delta))

        if changing_map:
            print(f"    CHANGING MAP POSITIONS ({len(changing_map)}):")
            for off, vals, delta in changing_map:
                first = vals[0]; last = vals[-1]
                # Check if this is the server position (matches hero pos)
                hp0 = snapshots[0][hname]["pos"]
                dist_from_hero = ((first[0]-hp0[0])**2 + (first[2]-hp0[2])**2)**0.5 if hp0 else 999
                tag = ""
                if dist_from_hero < 5: tag = " [SERVER POS - tracks hero]"
                elif all(abs(v[0]-vals[0][0])<5 and abs(v[2]-vals[0][2])<5 for v in vals):
                    tag = " [STABLE DESTINATION - barely moves]"
                else:
                    tag = " [CANDIDATE TARGET]"

                # Show all 6 values
                coords = " | ".join(f"({v[0]:.0f},{v[2]:.0f})" for v in vals)
                print(f"      +0x{off:03X}: {coords}  delta={delta:.1f}{tag}")
        else:
            print(f"    No changing map positions found")

        if changing_any:
            print(f"    OTHER CHANGING FIELDS ({len(changing_any)}):")
            for off, v0, vn, delta in changing_any[:10]:
                print(f"      +0x{off:03X}: ({v0[0]:.1f},{v0[1]:.1f},{v0[2]:.1f}) -> ({vn[0]:.1f},{vn[1]:.1f},{vn[2]:.1f})  delta={delta:.1f}")

    # ================================================================
    # SCAN 3: Also check ActiveSpell at hero+0x4578
    # ================================================================
    print(f"\n\n{'='*70}")
    print("SCAN 3: ActiveSpell check (hero + 0x4578)")
    print("=" * 70)

    for h in heroes:
        as_ptr = m.u64(h['ptr'] + 0x4578)
        if not is_heap(as_ptr):
            continue

        # Read SpellInfo chain: +0x038 -> SpellInfo -> +0x28 -> name
        spell_info = m.u64(as_ptr + 0x038)
        spell_name = None
        if is_heap(spell_info):
            spell_name = m.string(spell_info + 0x28, 48)

        cast_pos = m.vec3(as_ptr + 0x108)
        cast_time = m.f32(as_ptr + 0x1E8)

        cast_str = f"({cast_pos[0]:.0f},{cast_pos[2]:.0f})" if cast_pos and is_map_pos(cast_pos) else "N/A"
        time_str = f"{cast_time:.2f}" if cast_time and 0 < cast_time < 5000 else "N/A"

        if spell_name:
            print(f"  {h['name']:12s} ActiveSpell='{spell_name}' castPos={cast_str} castTime={time_str}")

        # Also scan nearby offsets for the spell name in case the chain is different
        if not spell_name:
            # Try direct name at various offsets
            for name_off in [0x28, 0x30, 0x38, 0x40, 0x48, 0x50, 0x58, 0x60, 0x68, 0x70, 0x78, 0x80]:
                n = m.string(as_ptr + name_off, 32)
                if n and len(n) > 4 and n[0].isupper():
                    print(f"  {h['name']:12s} ActiveSpell direct+0x{name_off:X}='{n}'")
                    break

    # ================================================================
    # SCAN 4: Dump raw hex of first hero's AiManager inner for manual inspection
    # ================================================================
    print(f"\n\n{'='*70}")
    print("SCAN 4: Raw hex dump of first hero's AiManager inner (0x00..0x100)")
    print("=" * 70)

    first_hero = heroes[0]
    first_inner = inner_ptrs.get(first_hero['name'])
    if first_inner:
        data = m.read(first_inner, 0x100)
        if data:
            print(f"  Hero: {first_hero['name']}, inner=0x{first_inner:X}")
            for i in range(0, len(data), 16):
                hex_str = ' '.join(f'{b:02X}' for b in data[i:i+16])
                # Also interpret as floats
                floats = []
                for j in range(0, 16, 4):
                    if i+j+4 <= len(data):
                        f = struct.unpack("<f", data[i+j:i+j+4])[0]
                        floats.append(f"{f:.1f}" if abs(f) < 100000 else "---")
                float_str = " | ".join(floats)
                print(f"  +0x{i:03X}: {hex_str:<48s}  [{float_str}]")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
