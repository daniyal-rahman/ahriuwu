"""
League Memory Reader v3 — Using offsets from hacker_logs.md (patch 26.S7).

CORRECT offsets from UnknownCheats thread (hacker_logs.md):
  Module base note: 0x7ff692ca0000 / size 0x202d000
  Our module: 0x7FF75BAC0000 / size 0x207C000
  (Different build - offsets may need adjustment)
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

# CORRECTED offsets from hacker_logs.md namespace Offset::Global
GLOBAL = {
    "LocalPlayer":    0x1DAB760,
    "HeroManager":    0x1D7A470,
    "GameTime":       0x1D88580,
    "MissileManager": 0x1D7DD90,
    "ObjectManager":  0x1D7A418,
    "MinionManager":  0x1D7A468,
}

# GameObject struct offsets
OBJ = {
    "Index":          0x10,
    "Team":           0x3C,
    "Name":           0x58,     # short name
    "NetId":          0xCC,
    "Dead":           0x250,
    "Position":       0x25C,    # Vec3 (x, y, z)
    "Visibility":     0x2E0,
    "Visible":        0x308,
    "Radius":         0x6F8,
    "CharacterName":  0x4330,   # champion name string
    "AiManager":      0x4038,
    "CharacterData":  0x40C8,
    "Direction":      0x21D8,   # facing Vec3
    "ItemList":       0x4D20,
}

# Health offsets (LeagueObfuscation<float>)
HP = {
    "HP":       0x1080,
    "MaxHP":    0x10A8,
}

# Mana
MANA = {
    "MP":    0x360,
    "MaxMP": 0x388,
}

# AI Manager
AI = {
    "StartPath":  0x88,
    "EndPath":    0x88,
    "TargetPos":  0x34,  # may need validation
}

# SpellBook
SPELL = {
    "SpellBook": 0x30E8,
}

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

def find_league_pid():
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    pe = PROCESSENTRY32()
    pe.dwSize = ctypes.sizeof(PROCESSENTRY32)
    if not kernel32.Process32First(snapshot, ctypes.byref(pe)):
        kernel32.CloseHandle(snapshot)
        return None
    while True:
        name = pe.szExeFile.decode("utf-8", errors="ignore")
        if "League of Legends" in name:
            pid = pe.th32ProcessID
            kernel32.CloseHandle(snapshot)
            return pid
        if not kernel32.Process32Next(snapshot, ctypes.byref(pe)):
            break
    kernel32.CloseHandle(snapshot)
    return None

def find_module_base(pid):
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid)
    if snapshot == -1:
        return None, None
    me = MODULEENTRY32()
    me.dwSize = ctypes.sizeof(MODULEENTRY32)
    if not kernel32.Module32First(snapshot, ctypes.byref(me)):
        kernel32.CloseHandle(snapshot)
        return None, None
    while True:
        mod_name = me.szModule.decode("utf-8", errors="ignore")
        if "league of legends" in mod_name.lower():
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
        if not self.h:
            raise OSError(f"OpenProcess failed: {ctypes.get_last_error()}")

    def read(self, addr, sz):
        buf = ctypes.create_string_buffer(sz)
        n = ctypes.c_size_t(0)
        ok = kernel32.ReadProcessMemory(self.h, ctypes.c_void_p(addr), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None

    def u32(self, addr):
        d = self.read(addr, 4)
        return struct.unpack("<I", d)[0] if d else None

    def u64(self, addr):
        d = self.read(addr, 8)
        return struct.unpack("<Q", d)[0] if d else None

    def f32(self, addr):
        d = self.read(addr, 4)
        return struct.unpack("<f", d)[0] if d else None

    def vec3(self, addr):
        d = self.read(addr, 12)
        return struct.unpack("<fff", d) if d else None

    def string(self, addr, n=64):
        d = self.read(addr, n)
        if d is None:
            return None
        try:
            s = d.split(b'\x00')[0].decode('utf-8', errors='replace')
            return s if s and len(s) > 0 else None
        except:
            return None

    def close(self):
        kernel32.CloseHandle(self.h)

def is_valid_ptr(v):
    return v is not None and 0x10000 < v < 0x7FFFFFFFFFFF

def dump_hex(m, addr, size=64, label=""):
    data = m.read(addr, size)
    if not data:
        print(f"  [{label}] FAILED to read 0x{addr:X}")
        return
    print(f"  [{label}] 0x{addr:X}:")
    for i in range(0, min(len(data), size), 16):
        hex_str = ' '.join(f'{b:02X}' for b in data[i:i+16])
        print(f"    +0x{i:02X}: {hex_str}")

def main():
    print("=" * 60)
    print("League Memory Reader v3")
    print("Offsets: hacker_logs.md (UC thread)")
    print("=" * 60)

    pid = find_league_pid()
    if not pid:
        print("League not found!")
        return
    print(f"PID: {pid}")

    base, mod_size = find_module_base(pid)
    if not base:
        print("Module not found!")
        return
    print(f"Base: 0x{base:X}, Size: 0x{mod_size:X} ({mod_size/1024/1024:.1f} MB)")
    print(f"Expected size: 0x202D000. Ours: 0x{mod_size:X}. Delta: 0x{mod_size - 0x202D000:X}")

    m = Mem(pid)

    # Verify
    mz = m.read(base, 2)
    assert mz == b'MZ', f"MZ check failed"
    print("ReadProcessMemory: OK")

    # ================================================================
    # STEP 1: Read global pointers
    # ================================================================
    print("\n--- GLOBAL POINTERS ---")
    ptrs = {}
    for name, offset in GLOBAL.items():
        val = m.u64(base + offset)
        valid = is_valid_ptr(val)
        ptrs[name] = val if valid else None
        status = f"0x{val:X}" if val else "NULL"
        if valid:
            status += " [VALID]"
        elif val and val != 0:
            status += " [INVALID - not a pointer]"
        print(f"  {name:20s} @ base+0x{offset:X} = {status}")

    # ================================================================
    # STEP 2: Read game time
    # ================================================================
    print("\n--- GAME TIME ---")
    gt = m.f32(base + GLOBAL["GameTime"])
    if gt and 0 < gt < 10000:
        print(f"  GameTime = {gt:.2f}s ({gt/60:.1f} min)")
    else:
        print(f"  GameTime raw = {gt} (unexpected)")
        # Try reading as pointer-to-float
        gt_ptr = m.u64(base + GLOBAL["GameTime"])
        if is_valid_ptr(gt_ptr):
            gt2 = m.f32(gt_ptr)
            print(f"  As ptr (0x{gt_ptr:X}) -> {gt2}")

    # ================================================================
    # STEP 3: Read local player
    # ================================================================
    print("\n--- LOCAL PLAYER ---")
    lp = ptrs.get("LocalPlayer")
    if lp:
        name_short = m.string(lp + OBJ["Name"])
        name_champ = m.string(lp + OBJ["CharacterName"])
        net_id = m.u32(lp + OBJ["NetId"])
        team = m.u32(lp + OBJ["Team"])
        pos = m.vec3(lp + OBJ["Position"])
        hp = m.f32(lp + HP["HP"])
        maxhp = m.f32(lp + HP["MaxHP"])

        print(f"  Name:     {name_short}")
        print(f"  Champion: {name_champ}")
        print(f"  NetID:    {net_id:#x if net_id else 'N/A'}")
        print(f"  Team:     {team}")
        pos_str = f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})" if pos else "N/A"
        print(f"  Position: {pos_str}")
        print(f"  HP:       {hp:.0f}/{maxhp:.0f}" if hp and maxhp else f"  HP: raw={hp}")
    else:
        print("  LocalPlayer pointer is NULL/invalid")
        # Dump the region to see what's there
        dump_hex(m, base + GLOBAL["LocalPlayer"], 32, "LocalPlayer offset")

    # ================================================================
    # STEP 4: Read hero list
    # ================================================================
    print("\n--- HERO MANAGER ---")
    hm = ptrs.get("HeroManager")
    if not hm:
        print("  HeroManager pointer is NULL/invalid")
        dump_hex(m, base + GLOBAL["HeroManager"], 64, "HeroManager offset")
    else:
        print(f"  HeroManager ptr: 0x{hm:X}")
        dump_hex(m, hm, 64, "HeroManager struct")

        # Common ManagerTemplate layout:
        # +0x00: vtable
        # +0x08: array_ptr (pointer to array of hero pointers)
        # +0x0C: count (u32)
        # +0x10: capacity (u32) or array_end ptr

        # Try several offsets for the array
        for arr_off in [0x08, 0x10, 0x18]:
            arr_ptr = m.u64(hm + arr_off)
            if not is_valid_ptr(arr_ptr):
                continue

            print(f"\n  Trying hero array at HeroMgr+0x{arr_off:X} -> 0x{arr_ptr:X}")
            heroes = []

            for i in range(12):
                hp_ptr = m.u64(arr_ptr + i * 8)
                if not is_valid_ptr(hp_ptr):
                    continue

                # Read hero data
                name = m.string(hp_ptr + OBJ["Name"])
                champ = m.string(hp_ptr + OBJ["CharacterName"])
                pos = m.vec3(hp_ptr + OBJ["Position"])
                net_id = m.u32(hp_ptr + OBJ["NetId"])
                team = m.u32(hp_ptr + OBJ["Team"])
                health = m.f32(hp_ptr + HP["HP"])

                # Check if this looks like a real champion
                pos_valid = pos and 0 < pos[0] < 16000 and 0 < pos[2] < 16000
                has_name = champ and len(champ) > 2 and champ[0].isupper()

                if has_name or pos_valid:
                    pos_str = f"({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})" if pos else "?"
                    print(f"    [{i:2d}] {champ or name or '???':15s} team={team} "
                          f"netId={net_id:#010x if net_id else 0:#x} "
                          f"pos={pos_str} hp={health:.0f}" if health else
                          f"    [{i:2d}] {champ or name or '???':15s} team={team} "
                          f"netId={net_id:#010x if net_id else 0:#x} pos={pos_str}")
                    heroes.append({
                        "ptr": hp_ptr,
                        "name": champ or name,
                        "team": team,
                        "pos": pos,
                        "net_id": net_id,
                    })

            if heroes:
                print(f"\n  SUCCESS! Found {len(heroes)} heroes via HeroMgr+0x{arr_off:X}")
                break
        else:
            # No heroes found - try scanning the manager more broadly
            print("\n  Standard array offsets didn't work. Scanning manager struct...")
            for off in range(0, 0x100, 8):
                ptr = m.u64(hm + off)
                if is_valid_ptr(ptr):
                    # Check if this pointer leads to champion-like objects
                    first_obj = m.u64(ptr)
                    if is_valid_ptr(first_obj):
                        champ = m.string(first_obj + OBJ["CharacterName"])
                        if champ and len(champ) > 2:
                            print(f"    HeroMgr+0x{off:X} -> arr -> obj with name='{champ}'")

    # ================================================================
    # STEP 5: Continuous monitoring
    # ================================================================
    if lp and ptrs.get("LocalPlayer"):
        print("\n--- CONTINUOUS MONITORING (5s) ---")
        for t in range(5):
            time.sleep(1)
            pos = m.vec3(lp + OBJ["Position"])
            gt = m.f32(base + GLOBAL["GameTime"])
            hp = m.f32(lp + HP["HP"])
            pos_str = f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})" if pos else "?"
            print(f"  t={gt:.1f}s pos={pos_str} hp={hp:.0f}" if hp and gt else
                  f"  gt={gt} pos={pos_str} hp={hp}")

    # ================================================================
    # STEP 6: If offsets don't match, try scanning
    # ================================================================
    if not ptrs.get("LocalPlayer") and not ptrs.get("HeroManager"):
        print("\n--- OFFSET SCAN ---")
        print("  Neither LocalPlayer nor HeroManager valid.")
        print(f"  Module size mismatch: expected 0x202D000, got 0x{mod_size:X}")
        print("  Offsets are likely for a different game build.")
        print("\n  Scanning for GameTime float (should change over 2s)...")

        # Quick scan around expected offset +/- 0x50000
        gt_off = GLOBAL["GameTime"]
        for delta in range(-0x50000, 0x50001, 0x10000):
            test_addr = base + gt_off + delta
            v1 = m.f32(test_addr)
            if v1 and 10 < v1 < 5000:
                time.sleep(1)
                v2 = m.f32(test_addr)
                if v2 and 0 < v2 - v1 < 50:
                    adj = gt_off + delta - GLOBAL["GameTime"]
                    print(f"  GameTime at base+0x{gt_off+delta:X} "
                          f"(offset adjustment: {'+' if adj >= 0 else ''}{adj:#x})")
                    print(f"  v1={v1:.2f} v2={v2:.2f} delta={v2-v1:.2f}")
                    print(f"\n  Apply this adjustment to all offsets:")
                    print(f"    LocalPlayer = 0x{GLOBAL['LocalPlayer']+adj:X}")
                    print(f"    HeroManager = 0x{GLOBAL['HeroManager']+adj:X}")

    m.close()
    print("\n" + "=" * 60)
    print("Done!")

if __name__ == "__main__":
    main()
