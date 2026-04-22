"""
League Memory Reader — Waypoint + Auto Attack extraction.

AiManager offsets (from hacker_logs.md):
  AiManager ptr: hero + 0x4038 (or 0x41F0 with inner deref at +0x10)
  TargetPosition:  +0x34  (Vec3: click destination)
  Velocity:        +0x318 (float: move speed)
  IsMoving:        +0x31C (bool)
  CurrentSegment:  +0x320 (int)
  PathStart:       +0x330 (Vec3)
  PathEnd:         +0x33C (Vec3)
  NavArray:        +0x348 (ptr to Vec3[] waypoints)
  SegmentsCount:   +0x350 (int: waypoint count)
  IsDashing:       +0x384 (bool)
  DashSpeed:       +0x360 (float)
  ServerPos:       +0x474 (Vec3)

SpellBook / Active Spell:
  SpellBook:       hero + 0x30E8
  BasicAttackBase: hero + 0x2C68
  ActiveSpell:     hero + 0x2A70 (from pandoras)
  OnCastSpellName: active_spell + 0x28
  OnCastStartPos:  active_spell + 0xD0
  OnCastTargetPos: active_spell + 0xDC
"""

import ctypes
import ctypes.wintypes
import struct
import sys
import time
import json
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
        ("dwSize", wintypes.DWORD),
        ("cntUsage", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("th32DefaultHeapID", ctypes.POINTER(ctypes.c_ulong)),
        ("th32ModuleID", wintypes.DWORD),
        ("cntThreads", wintypes.DWORD),
        ("th32ParentProcessID", wintypes.DWORD),
        ("pcPriClassBase", ctypes.c_long),
        ("dwFlags", wintypes.DWORD),
        ("szExeFile", ctypes.c_char * MAX_PATH),
    ]

class MODULEENTRY32(ctypes.Structure):
    _fields_ = [
        ("dwSize", wintypes.DWORD),
        ("th32ModuleID", wintypes.DWORD),
        ("th32ProcessID", wintypes.DWORD),
        ("GlblcntUsage", wintypes.DWORD),
        ("ProccntUsage", wintypes.DWORD),
        ("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),
        ("modBaseSize", wintypes.DWORD),
        ("hModule", wintypes.HMODULE),
        ("szModule", ctypes.c_char * (MAX_MODULE_NAME32 + 1)),
        ("szExePath", ctypes.c_char * MAX_PATH),
    ]

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

def find_league():
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    pe = PROCESSENTRY32()
    pe.dwSize = ctypes.sizeof(PROCESSENTRY32)
    kernel32.Process32First(snapshot, ctypes.byref(pe))
    while True:
        if b"League of Legends" in pe.szExeFile:
            pid = pe.th32ProcessID
            kernel32.CloseHandle(snapshot)
            return pid
        if not kernel32.Process32Next(snapshot, ctypes.byref(pe)):
            break
    kernel32.CloseHandle(snapshot)
    return None

def find_base(pid):
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    me = MODULEENTRY32()
    me.dwSize = ctypes.sizeof(MODULEENTRY32)
    kernel32.Module32First(snapshot, ctypes.byref(me))
    while True:
        if b"league of legends" in me.szModule.lower():
            base = ctypes.cast(me.modBaseAddr, ctypes.c_void_p).value
            size = me.modBaseSize
            kernel32.CloseHandle(snapshot)
            return base, size
        if not kernel32.Module32Next(snapshot, ctypes.byref(me)):
            break
    kernel32.CloseHandle(snapshot)
    return None, None

class Mem:
    def __init__(self, pid):
        self.h = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
    def read(self, addr, sz):
        buf = ctypes.create_string_buffer(sz)
        n = ctypes.c_size_t(0)
        ok = kernel32.ReadProcessMemory(self.h, ctypes.c_void_p(addr), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u8(self, a):
        d = self.read(a, 1); return d[0] if d else None
    def u32(self, a):
        d = self.read(a, 4); return struct.unpack("<I", d)[0] if d else None
    def i32(self, a):
        d = self.read(a, 4); return struct.unpack("<i", d)[0] if d else None
    def u64(self, a):
        d = self.read(a, 8); return struct.unpack("<Q", d)[0] if d else None
    def f32(self, a):
        d = self.read(a, 4); return struct.unpack("<f", d)[0] if d else None
    def vec3(self, a):
        d = self.read(a, 12); return struct.unpack("<fff", d) if d else None
    def string(self, a, n=64):
        d = self.read(a, n)
        if not d: return None
        return d.split(b'\x00')[0].decode('ascii', errors='replace') or None
    def close(self):
        kernel32.CloseHandle(self.h)

def is_heap(v):
    return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF

def is_valid_pos(v):
    """Check if a vec3 looks like a valid LoL map position."""
    if v is None: return False
    x, y, z = v
    return -500 < x < 16000 and -500 < y < 1000 and -500 < z < 16000

def try_read_ai_manager(m, hero_ptr, ai_offset):
    """Try to read AiManager data from a hero at a given offset."""
    ai_ptr = m.u64(hero_ptr + ai_offset)
    if not is_heap(ai_ptr):
        return None

    # Try direct access
    target_pos = m.vec3(ai_ptr + 0x34)
    if is_valid_pos(target_pos):
        return ai_ptr, False  # direct, no inner deref

    # Try with inner dereference (LeagueObfuscation pattern)
    inner = m.u64(ai_ptr + 0x10)
    if is_heap(inner):
        target_pos = m.vec3(inner + 0x34)
        if is_valid_pos(target_pos):
            return inner, True  # inner deref needed

    return None

def read_ai_manager(m, ai_ptr):
    """Read full AiManager state."""
    result = {}

    result["target_pos"] = m.vec3(ai_ptr + 0x34)
    result["velocity"] = m.f32(ai_ptr + 0x318)
    result["is_moving"] = m.f32(ai_ptr + 0x31C)
    result["current_segment"] = m.i32(ai_ptr + 0x320)
    result["path_start"] = m.vec3(ai_ptr + 0x330)
    result["path_end"] = m.vec3(ai_ptr + 0x33C)
    result["segments_count"] = m.i32(ai_ptr + 0x350)
    result["has_path"] = m.i32(ai_ptr + 0x354)
    result["dash_speed"] = m.f32(ai_ptr + 0x360)
    result["is_dashing"] = m.u8(ai_ptr + 0x384)
    result["server_pos"] = m.vec3(ai_ptr + 0x474)

    # Read waypoint array
    nav_arr_ptr = m.u64(ai_ptr + 0x348)
    seg_count = result["segments_count"]
    waypoints = []
    if is_heap(nav_arr_ptr) and seg_count and 0 < seg_count <= 50:
        for i in range(seg_count):
            wp = m.vec3(nav_arr_ptr + i * 12)  # Vec3 = 12 bytes
            if wp:
                waypoints.append(wp)
    result["waypoints"] = waypoints

    return result

def read_active_spell(m, hero_ptr):
    """Try to read active spell / auto attack data."""
    results = {}

    # Try SpellBook approach: hero + 0x30E8 -> spellbook
    sb_ptr = m.u64(hero_ptr + 0x30E8)
    if is_heap(sb_ptr):
        # Active spell at spellbook + 0x38 (common offset)
        active = m.u64(sb_ptr + 0x38)
        if is_heap(active):
            spell_name = m.string(active + 0x28, 32)
            start_pos = m.vec3(active + 0xD0)
            target_pos = m.vec3(active + 0xDC)
            results["spellbook_active"] = {
                "name": spell_name,
                "start_pos": start_pos,
                "target_pos": target_pos
            }

    # Try BasicAttack approach: hero + 0x2C68
    ba_ptr = m.u64(hero_ptr + 0x2C68)
    if is_heap(ba_ptr):
        # BasicAttack -> +0x2C0 -> +0x70
        ba2 = m.u64(ba_ptr + 0x2C0)
        if is_heap(ba2):
            ba3 = m.u64(ba2 + 0x70)
            results["basic_attack_chain"] = f"0x{ba_ptr:X} -> 0x{ba2:X} -> {ba3}"

    # Try ActiveSpell at hero + 0x2A70 (pandoras)
    as_ptr = m.u64(hero_ptr + 0x2A70)
    if is_heap(as_ptr):
        spell_name = m.string(as_ptr + 0x28, 32)
        results["active_spell_pandoras"] = {
            "ptr": f"0x{as_ptr:X}",
            "name": spell_name,
        }

    return results

def vec3_str(v):
    if v is None: return "N/A"
    return f"({v[0]:.0f},{v[1]:.0f},{v[2]:.0f})"

def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 70)
    print("League Memory Reader - Waypoints & Auto Attacks")
    print("=" * 70)

    pid = find_league()
    if not pid:
        print("League not found!"); return
    base, _ = find_base(pid)
    if not base:
        print("Module not found!"); return
    print(f"PID={pid} Base=0x{base:X}")

    m = Mem(pid)

    # Read hero array
    arr_ptr = m.u64(base + HERO_ARRAY_RVA)
    if not is_heap(arr_ptr):
        print("Hero array not found!"); return

    heroes = []
    for i in range(10):
        hp = m.u64(arr_ptr + i * 8)
        if not is_heap(hp): continue
        name = m.string(hp + 0x4328, 32) or f"hero_{i}"
        pos = m.vec3(hp + 0x25C)
        heroes.append({"ptr": hp, "name": name, "pos": pos, "index": i})

    print(f"Found {len(heroes)} heroes\n")

    # ================================================================
    # STEP 1: Find working AiManager offset
    # ================================================================
    print("--- STEP 1: Finding AiManager offset ---")
    ai_offset = None
    needs_inner = False

    for test_hero in heroes[:3]:
        for test_off in [0x4038, 0x41F0, 0x4030]:
            result = try_read_ai_manager(m, test_hero["ptr"], test_off)
            if result:
                ai_ptr_val, inner = result
                target = m.vec3(ai_ptr_val + 0x34)
                print(f"  {test_hero['name']}: AiManager at hero+0x{test_off:X} "
                      f"{'(inner +0x10)' if inner else '(direct)'} "
                      f"target={vec3_str(target)}")
                ai_offset = test_off
                needs_inner = inner
                break

    if ai_offset is None:
        # Brute force scan for AiManager pointer in hero struct
        print("\n  Standard offsets failed. Scanning hero struct for AiManager...")
        test_hero = heroes[0]
        for off in range(0x3F00, 0x4500, 8):
            ptr = m.u64(test_hero["ptr"] + off)
            if not is_heap(ptr): continue
            target = m.vec3(ptr + 0x34)
            if is_valid_pos(target):
                print(f"  FOUND: hero+0x{off:X} -> 0x{ptr:X} target={vec3_str(target)}")
                ai_offset = off
                break
            # Try inner deref
            inner = m.u64(ptr + 0x10)
            if is_heap(inner):
                target = m.vec3(inner + 0x34)
                if is_valid_pos(target):
                    print(f"  FOUND: hero+0x{off:X} -> inner+0x10 target={vec3_str(target)}")
                    ai_offset = off
                    needs_inner = True
                    break

    if ai_offset is None:
        print("  FAILED: Could not find AiManager offset")
        # Still try to get spell data
    else:
        print(f"\n  Using: hero + 0x{ai_offset:X} {'-> inner+0x10' if needs_inner else ''}")

    # ================================================================
    # STEP 2: Read AiManager for all heroes
    # ================================================================
    if ai_offset:
        print(f"\n--- STEP 2: AiManager state for all heroes ---")
        for hero in heroes:
            ai_raw = m.u64(hero["ptr"] + ai_offset)
            if not is_heap(ai_raw):
                print(f"  {hero['name']:12s} AiManager=NULL")
                continue

            ai_ptr = ai_raw
            if needs_inner:
                inner = m.u64(ai_raw + 0x10)
                if is_heap(inner):
                    ai_ptr = inner
                else:
                    print(f"  {hero['name']:12s} inner deref failed")
                    continue

            ai = read_ai_manager(m, ai_ptr)

            tp = ai["target_pos"]
            sp = ai["server_pos"]
            moving = "Y" if ai["is_moving"] and ai["is_moving"] > 0 else "N"
            dashing = "Y" if ai["is_dashing"] else "N"
            seg = ai["segments_count"] or 0
            vel = ai["velocity"] or 0

            print(f"  {hero['name']:12s} mov={moving} dash={dashing} "
                  f"vel={vel:.0f} segs={seg} "
                  f"target={vec3_str(tp)} "
                  f"server={vec3_str(sp)}")

            if ai["waypoints"]:
                for j, wp in enumerate(ai["waypoints"]):
                    print(f"    wp[{j}] = {vec3_str(wp)}")

    # ================================================================
    # STEP 3: Check active spells / auto attacks
    # ================================================================
    print(f"\n--- STEP 3: Active spells / auto attacks ---")
    for hero in heroes:
        spells = read_active_spell(m, hero["ptr"])
        if spells:
            print(f"  {hero['name']:12s}: {spells}")

    # ================================================================
    # STEP 4: Monitor waypoints over time
    # ================================================================
    if ai_offset:
        print(f"\n--- STEP 4: Waypoint monitoring (10 samples, 0.25s) ---")
        all_samples = []
        for t in range(10):
            time.sleep(0.25)
            sample = {"t": t}
            for hero in heroes:
                ai_raw = m.u64(hero["ptr"] + ai_offset)
                if not is_heap(ai_raw): continue
                ai_ptr = ai_raw
                if needs_inner:
                    inner = m.u64(ai_raw + 0x10)
                    if is_heap(inner): ai_ptr = inner
                    else: continue

                pos = m.vec3(hero["ptr"] + 0x25C)
                target = m.vec3(ai_ptr + 0x34)
                seg_count = m.i32(ai_ptr + 0x350)
                is_moving = m.f32(ai_ptr + 0x31C)

                sample[hero["name"]] = {
                    "pos": [pos[0], pos[2]] if pos else None,
                    "target": [target[0], target[2]] if target else None,
                    "segs": seg_count,
                    "moving": is_moving and is_moving > 0,
                }

            all_samples.append(sample)

            # Print compact
            parts = [f"t={t}"]
            for hero in heroes[:5]:
                s = sample.get(hero["name"])
                if s and s["pos"]:
                    mov = "*" if s["moving"] else " "
                    parts.append(f"{hero['name'][:6]}{mov}({s['pos'][0]:.0f},{s['pos'][1]:.0f})"
                                 f"->({s['target'][0]:.0f},{s['target'][1]:.0f})" if s["target"] else "")
            print(f"  {'  '.join(parts)}")

        # Save samples
        out = {
            "ai_offset": f"0x{ai_offset:X}",
            "needs_inner_deref": needs_inner,
            "heroes": [{"name": h["name"], "ptr": f"0x{h['ptr']:X}"} for h in heroes],
            "samples": all_samples,
        }
        out_path = "C:\\Users\\daniz\\waypoint_output.json"
        with open(out_path, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f"\nSaved to {out_path}")

    m.close()
    print("Done!")

if __name__ == "__main__":
    main()
