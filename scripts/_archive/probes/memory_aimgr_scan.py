"""
AiManager offset finder — scan the AiManager struct for position-like values.

Strategy: Take a MOVING hero, read its position from +0x25C, then scan
the AiManager object for Vec3 values that are:
1. Near the hero's current position (= path start)
2. In valid map range (= target position / waypoints)
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
    _fields_ = [
        ("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_char * MAX_PATH),
    ]

class MODULEENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD), ("th32ModuleID", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD), ("GlblcntUsage", wintypes.DWORD),
        ("ProccntUsage", wintypes.DWORD),
        ("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),
        ("modBaseSize", wintypes.DWORD), ("hModule", wintypes.HMODULE),
        ("szModule", ctypes.c_char * (MAX_MODULE_NAME32 + 1)),
        ("szExePath", ctypes.c_char * MAX_PATH),
    ]

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

def find_league():
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    pe = PROCESSENTRY32(); pe.dwSize = ctypes.sizeof(PROCESSENTRY32)
    kernel32.Process32First(snap, ctypes.byref(pe))
    while True:
        if b"League of Legends" in pe.szExeFile:
            pid = pe.th32ProcessID; kernel32.CloseHandle(snap); return pid
        if not kernel32.Process32Next(snap, ctypes.byref(pe)): break
    kernel32.CloseHandle(snap); return None

def find_base(pid):
    snap = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    me = MODULEENTRY32(); me.dwSize = ctypes.sizeof(MODULEENTRY32)
    kernel32.Module32First(snap, ctypes.byref(me))
    while True:
        if b"league of legends" in me.szModule.lower():
            b = ctypes.cast(me.modBaseAddr, ctypes.c_void_p).value
            s = me.modBaseSize; kernel32.CloseHandle(snap); return b, s
        if not kernel32.Module32Next(snap, ctypes.byref(me)): break
    kernel32.CloseHandle(snap); return None, None

class Mem:
    def __init__(self, pid):
        self.h = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz)
        n = ctypes.c_size_t(0)
        ok = kernel32.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u32(self, a): d = self.read(a, 4); return struct.unpack("<I", d)[0] if d else None
    def u64(self, a): d = self.read(a, 8); return struct.unpack("<Q", d)[0] if d else None
    def f32(self, a): d = self.read(a, 4); return struct.unpack("<f", d)[0] if d else None
    def vec3(self, a): d = self.read(a, 12); return struct.unpack("<fff", d) if d else None
    def string(self, a, n=64):
        d = self.read(a, n)
        if not d: return None
        return d.split(b'\x00')[0].decode('ascii', errors='replace') or None
    def close(self): kernel32.CloseHandle(self.h)

def is_heap(v): return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF
def is_map_pos(v): return v is not None and all(-500 < c < 16000 for c in v) and (v[0] != 0 or v[2] != 0)

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    pid = find_league()
    base, _ = find_base(pid)
    m = Mem(pid)
    print(f"PID={pid} Base=0x{base:X}")

    # Read hero array
    arr_ptr = m.u64(base + HERO_ARRAY_RVA)
    heroes = []
    for i in range(10):
        hp = m.u64(arr_ptr + i * 8)
        if not is_heap(hp): continue
        name = m.string(hp + 0x4328, 32) or f"hero_{i}"
        pos = m.vec3(hp + 0x25C)
        heroes.append({"ptr": hp, "name": name, "pos": pos})

    # Pick a moving hero — wait and check positions
    print("\nWaiting 1s to find a moving hero...")
    time.sleep(1)
    moving_hero = None
    for h in heroes:
        pos2 = m.vec3(h["ptr"] + 0x25C)
        if pos2 and h["pos"]:
            dx = pos2[0] - h["pos"][0]
            dz = pos2[2] - h["pos"][2]
            dist = (dx*dx + dz*dz)**0.5
            if dist > 10:
                moving_hero = h
                moving_hero["pos2"] = pos2
                print(f"  Moving: {h['name']} pos=({h['pos'][0]:.0f},{h['pos'][2]:.0f}) "
                      f"-> ({pos2[0]:.0f},{pos2[2]:.0f}) dist={dist:.0f}")
                break

    if not moving_hero:
        print("  No moving hero found, using first hero")
        moving_hero = heroes[0]
        moving_hero["pos2"] = m.vec3(moving_hero["ptr"] + 0x25C)

    hero_ptr = moving_hero["ptr"]
    hero_pos = moving_hero.get("pos2") or moving_hero["pos"]
    print(f"\nUsing {moving_hero['name']} at ({hero_pos[0]:.0f}, {hero_pos[1]:.0f}, {hero_pos[2]:.0f})")

    # ================================================================
    # STEP 1: Scan hero struct for AiManager pointer candidates
    # ================================================================
    print("\n--- Scanning hero struct for AiManager pointer candidates ---")
    ai_candidates = []

    for off in range(0x3F00, 0x4800, 8):
        ptr = m.u64(hero_ptr + off)
        if not is_heap(ptr): continue

        # Check if this object contains position-like vec3 values
        # Read a large chunk from the candidate
        data = m.read(ptr, 0x500)
        if not data: continue

        pos_matches = []
        for i in range(0, len(data) - 12, 4):
            v = struct.unpack("<fff", data[i:i+12])
            if is_map_pos(v):
                # Check if close to hero position
                dx = v[0] - hero_pos[0]
                dz = v[2] - hero_pos[2]
                dist = (dx*dx + dz*dz)**0.5
                pos_matches.append((i, v, dist))

        if pos_matches:
            # Sort by distance to hero
            pos_matches.sort(key=lambda x: x[2])
            best = pos_matches[0]
            ai_candidates.append((off, ptr, pos_matches))
            if best[2] < 2000:  # Within 2000 units of hero
                print(f"  hero+0x{off:X} -> 0x{ptr:X}: {len(pos_matches)} pos-like values")
                for inner_off, v, d in pos_matches[:5]:
                    marker = " <-- CLOSE" if d < 500 else ""
                    print(f"    +0x{inner_off:03X}: ({v[0]:.0f},{v[1]:.0f},{v[2]:.0f}) dist={d:.0f}{marker}")

        # Also try inner dereference
        inner = m.u64(ptr + 0x10)
        if is_heap(inner):
            data2 = m.read(inner, 0x500)
            if data2:
                pos_matches2 = []
                for i in range(0, len(data2) - 12, 4):
                    v = struct.unpack("<fff", data2[i:i+12])
                    if is_map_pos(v):
                        dx = v[0] - hero_pos[0]
                        dz = v[2] - hero_pos[2]
                        dist = (dx*dx + dz*dz)**0.5
                        pos_matches2.append((i, v, dist))

                if pos_matches2:
                    pos_matches2.sort(key=lambda x: x[2])
                    best2 = pos_matches2[0]
                    if best2[2] < 2000:
                        print(f"  hero+0x{off:X} -> inner+0x10 -> 0x{inner:X}: "
                              f"{len(pos_matches2)} pos-like values")
                        for inner_off, v, d in pos_matches2[:5]:
                            marker = " <-- CLOSE" if d < 500 else ""
                            print(f"    +0x{inner_off:03X}: ({v[0]:.0f},{v[1]:.0f},{v[2]:.0f}) dist={d:.0f}{marker}")

    # ================================================================
    # STEP 2: For the best AiManager candidate, identify offsets
    # ================================================================
    if ai_candidates:
        # Use the candidate with the most position values close to hero
        best_cand = None
        best_close_count = 0
        for off, ptr, matches in ai_candidates:
            close_count = sum(1 for _, _, d in matches if d < 1000)
            if close_count > best_close_count:
                best_close_count = close_count
                best_cand = (off, ptr, matches)

        if best_cand:
            off, ptr, matches = best_cand
            print(f"\n--- Best AiManager candidate: hero+0x{off:X} -> 0x{ptr:X} ---")
            print(f"  {best_close_count} position values within 1000 units of hero")

            # Read full struct and identify key fields
            data = m.read(ptr, 0x500)
            if data:
                print("\n  Identified position fields:")
                for inner_off, v, d in matches:
                    label = ""
                    if d < 100: label = "  <-- CURRENT POS / PATH START"
                    elif d < 500: label = "  <-- TARGET / PATH END"
                    elif d < 2000: label = "  <-- NEARBY (waypoint?)"
                    print(f"    +0x{inner_off:03X}: ({v[0]:.1f}, {v[1]:.1f}, {v[2]:.1f}) "
                          f"dist={d:.0f}{label}")

                # Check for int values that could be segment count (small positive int)
                print("\n  Small int fields (potential segment count):")
                for i in range(0, min(len(data), 0x400), 4):
                    val = struct.unpack("<i", data[i:i+4])[0]
                    if 1 <= val <= 20:
                        print(f"    +0x{i:03X}: {val}")

                # Check for pointers (potential waypoint array)
                print("\n  Heap pointer fields (potential waypoint array):")
                for i in range(0, min(len(data), 0x400), 8):
                    val = struct.unpack("<Q", data[i:i+8])[0]
                    if is_heap(val):
                        # Read first waypoint from array
                        wp = m.vec3(val)
                        wp_valid = is_map_pos(wp) if wp else False
                        wp_str = f"({wp[0]:.0f},{wp[1]:.0f},{wp[2]:.0f})" if wp else "?"
                        if wp_valid:
                            print(f"    +0x{i:03X}: 0x{val:X} -> wp0={wp_str} VALID")

            # Verify with second sample
            print("\n  Verifying with 1s delay...")
            time.sleep(1)
            new_pos = m.vec3(hero_ptr + 0x25C)
            if new_pos:
                print(f"  Hero now at ({new_pos[0]:.0f},{new_pos[1]:.0f},{new_pos[2]:.0f})")
                # Re-check the identified offsets
                for inner_off, v_old, _ in matches[:5]:
                    v_new = m.vec3(ptr + inner_off)
                    if v_new:
                        changed = any(abs(a - b) > 1 for a, b in zip(v_old, v_new))
                        print(f"    +0x{inner_off:03X}: ({v_new[0]:.0f},{v_new[1]:.0f},{v_new[2]:.0f}) "
                              f"{'CHANGED' if changed else 'static'}")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
