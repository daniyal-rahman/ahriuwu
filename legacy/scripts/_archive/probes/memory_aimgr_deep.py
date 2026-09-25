"""
Deep AiManager analysis — scan candidates over TIME to identify
which fields track hero movement target vs static vs changing.
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
        if b"League of Legends" in pe.szExeFile: pid = pe.th32ProcessID; kernel32.CloseHandle(s); return pid
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
    def __init__(s, pid): s.h = kernel32.OpenProcess(PROCESS_VM_READ|PROCESS_QUERY_INFORMATION, False, pid)
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
def is_map(v): return v and -500<v[0]<16000 and -500<v[1]<1000 and -500<v[2]<16000 and (v[0]!=0 or v[2]!=0)

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    pid = find_league(); base, _ = find_base(pid); m = Mem(pid)
    arr_ptr = m.u64(base + HERO_ARRAY_RVA)
    heroes = []
    for i in range(10):
        hp = m.u64(arr_ptr + i*8)
        if not is_heap(hp): continue
        heroes.append({"ptr":hp, "name": m.string(hp+0x4328,32) or f"h{i}"})

    # Test both AiManager candidates on ALL heroes
    # Candidate A: hero+0x4010 (direct, had destination-like positions)
    # Candidate B: hero+0x4628 -> inner+0x10 (had exact position match)
    # Also try the hacker_logs offsets: 0x4038, 0x41F0

    ai_candidates = [0x4010, 0x4038, 0x41F0, 0x4628]

    print("=== AiManager Candidate Analysis ===")
    print(f"Heroes: {[h['name'] for h in heroes]}\n")

    for ai_off in ai_candidates:
        print(f"\n--- Testing hero + 0x{ai_off:X} ---")
        for hero in heroes[:3]:
            ptr = m.u64(hero["ptr"] + ai_off)
            if not is_heap(ptr):
                print(f"  {hero['name']}: NULL"); continue

            hero_pos = m.vec3(hero["ptr"] + 0x25C)
            hx, hy, hz = hero_pos if hero_pos else (0,0,0)

            # Try direct
            t34 = m.vec3(ptr + 0x34)
            t330 = m.vec3(ptr + 0x330)
            t33c = m.vec3(ptr + 0x33C)
            v318 = m.f32(ptr + 0x318)
            s350 = m.u32(ptr + 0x350)
            narr = m.u64(ptr + 0x348)
            sp474 = m.vec3(ptr + 0x474)

            d_direct = "d34={} d330={} d33c={} v={} s={} sp={}".format(
                "({:.0f},{:.0f})".format(t34[0],t34[2]) if t34 and is_map(t34) else "X",
                "({:.0f},{:.0f})".format(t330[0],t330[2]) if t330 and is_map(t330) else "X",
                "({:.0f},{:.0f})".format(t33c[0],t33c[2]) if t33c and is_map(t33c) else "X",
                "{:.0f}".format(v318) if v318 and 0<v318<1000 else "X",
                s350 if s350 and 0<s350<50 else "X",
                "({:.0f},{:.0f})".format(sp474[0],sp474[2]) if sp474 and is_map(sp474) else "X",
            )

            # Try inner +0x10
            inner = m.u64(ptr + 0x10)
            d_inner = "N/A"
            if is_heap(inner):
                t34i = m.vec3(inner + 0x34)
                t330i = m.vec3(inner + 0x330)
                t33ci = m.vec3(inner + 0x33C)
                v318i = m.f32(inner + 0x318)
                s350i = m.u32(inner + 0x350)
                narri = m.u64(inner + 0x348)
                sp474i = m.vec3(inner + 0x474)
                p0i = m.vec3(inner + 0x0)

                d_inner = "p0={} d34={} d330={} d33c={} v={} s={} sp={}".format(
                    "({:.0f},{:.0f})".format(p0i[0],p0i[2]) if p0i and is_map(p0i) else "X",
                    "({:.0f},{:.0f})".format(t34i[0],t34i[2]) if t34i and is_map(t34i) else "X",
                    "({:.0f},{:.0f})".format(t330i[0],t330i[2]) if t330i and is_map(t330i) else "X",
                    "({:.0f},{:.0f})".format(t33ci[0],t33ci[2]) if t33ci and is_map(t33ci) else "X",
                    "{:.0f}".format(v318i) if v318i and 0<v318i<1000 else "X",
                    s350i if s350i and 0<s350i<50 else "X",
                    "({:.0f},{:.0f})".format(sp474i[0],sp474i[2]) if sp474i and is_map(sp474i) else "X",
                )

                # Also scan inner for ALL map positions
                data = m.read(inner, 0x500)
                if data:
                    close_positions = []
                    for off in range(0, len(data)-12, 4):
                        v = struct.unpack("<fff", data[off:off+12])
                        if is_map(v):
                            dx = v[0]-hx; dz = v[2]-hz; dist = (dx*dx+dz*dz)**0.5
                            if dist < 3000:
                                close_positions.append((off, v, dist))
                    if close_positions:
                        close_positions.sort(key=lambda x: x[2])
                        d_inner += f"\n      close_pos: " + ", ".join(
                            f"+0x{o:03X}:({v[0]:.0f},{v[2]:.0f})d={d:.0f}"
                            for o,v,d in close_positions[:8]
                        )

            hpos_str = "({:.0f},{:.0f})".format(hx,hz)
            print(f"  {hero['name']:10s} pos={hpos_str}")
            print(f"    direct: {d_direct}")
            print(f"    inner:  {d_inner}")

    # Now do temporal analysis on best candidate
    print("\n\n=== Temporal Analysis: hero+0x4628 -> inner+0x10 ===")
    print("Tracking all map-position fields over 5 seconds...")

    test_hero = heroes[0]  # First hero
    ai_raw = m.u64(test_hero["ptr"] + 0x4628)
    inner_ptr = m.u64(ai_raw + 0x10) if is_heap(ai_raw) else None

    if inner_ptr and is_heap(inner_ptr):
        # Take 5 snapshots
        snapshots = []
        for t in range(5):
            hero_pos = m.vec3(test_hero["ptr"] + 0x25C)
            data = m.read(inner_ptr, 0x500)
            snapshots.append((hero_pos, data))
            time.sleep(1)

        # Find fields that CHANGE over time
        print(f"\n  Hero: {test_hero['name']}")
        h0 = snapshots[0][0]
        h4 = snapshots[4][0]
        if h0 and h4:
            print(f"  Position: ({h0[0]:.0f},{h0[2]:.0f}) -> ({h4[0]:.0f},{h4[2]:.0f})")

        changing_fields = []
        for off in range(0, 0x4FC, 4):
            vals = []
            for _, data in snapshots:
                if data and off+12 <= len(data):
                    v = struct.unpack("<fff", data[off:off+12])
                    vals.append(v)

            if len(vals) >= 2:
                # Check if any component changed
                changed = any(
                    abs(vals[i][c] - vals[0][c]) > 0.1
                    for i in range(1, len(vals))
                    for c in range(3)
                )
                if changed and is_map(vals[0]):
                    changing_fields.append((off, vals))

        if changing_fields:
            print(f"\n  CHANGING map-position fields ({len(changing_fields)}):")
            for off, vals in changing_fields:
                print(f"    +0x{off:03X}: ", end="")
                for v in vals:
                    print(f"({v[0]:.0f},{v[2]:.0f}) ", end="")
                print()
        else:
            print("\n  No changing map-position fields found in inner struct")
            print("  (Hero might not be moving, or this isn't the AiManager)")

    # Also check hero+0x4010 temporally
    print("\n\n=== Temporal Analysis: hero+0x4010 (direct) ===")
    ai_direct = m.u64(test_hero["ptr"] + 0x4010)
    if is_heap(ai_direct):
        snapshots2 = []
        for t in range(5):
            hero_pos = m.vec3(test_hero["ptr"] + 0x25C)
            data = m.read(ai_direct, 0x500)
            snapshots2.append((hero_pos, data))
            time.sleep(1)

        h0 = snapshots2[0][0]
        h4 = snapshots2[4][0]
        if h0 and h4:
            print(f"  Hero: {test_hero['name']} ({h0[0]:.0f},{h0[2]:.0f}) -> ({h4[0]:.0f},{h4[2]:.0f})")

        changing2 = []
        for off in range(0, 0x4FC, 4):
            vals = []
            for _, data in snapshots2:
                if data and off+12 <= len(data):
                    v = struct.unpack("<fff", data[off:off+12])
                    vals.append(v)
            if len(vals) >= 2:
                changed = any(abs(vals[i][c]-vals[0][c])>0.1 for i in range(1,len(vals)) for c in range(3))
                if changed and is_map(vals[0]):
                    changing2.append((off, vals))

        if changing2:
            print(f"\n  CHANGING map-position fields ({len(changing2)}):")
            for off, vals in changing2:
                print(f"    +0x{off:03X}: ", end="")
                for v in vals:
                    print(f"({v[0]:.0f},{v[2]:.0f}) ", end="")
                print()
        else:
            print("  No changing map-position fields")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
