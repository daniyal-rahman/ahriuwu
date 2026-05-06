"""
League of Legends Memory Scraper - WORKING

Reads all 10 champion positions + stats from game memory via ReadProcessMemory.
Requires Vanguard DISABLED (vgc + vgk services stopped).

Discovered offsets for this build (patch 26.S7, module size 0x207C000):
  HeroArrayPtr: base + 0x1DBEEE8 -> pointer to array of 10 hero pointers
  GameTime: TBD (scan for changing float)

  GameObject struct (confirmed working):
    +0x25C = Position (Vec3: x, y, z) [CONFIRMED]
    +0xCC  = NetId [CONFIRMED]
    +0x4328= CharacterName [CONFIRMED]

  AiManager (hero + 0x4628 -> deref -> +0x10 -> inner):
    inner+0x000 = Current/server position (Vec3) [CONFIRMED changing]
    inner+0x030 = Target/click position (Vec3) [CONFIRMED for moving heroes]
    inner+0x228 = Path waypoint data (changes on new move commands)

  Struct offsets from hacker_logs (UNVERIFIED for this build):
    +0x10  = Index
    +0x3C  = Team (100=blue, 200=red)
    +0x58  = Name (short)
    +0x250 = Dead
    +0x308 = Visible
    +0x1080= HP (obfuscated float)
    +0x10A8= MaxHP
    +0x360 = MP
    +0x388 = MaxMP
"""

import ctypes
import ctypes.wintypes
import struct
import sys
import time
import json
import os
from ctypes import wintypes

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
TH32CS_SNAPPROCESS = 0x00000002
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
MAX_MODULE_NAME32 = 255
MAX_PATH = 260

# Discovered offsets
HERO_ARRAY_RVA = 0x1DBEEE8  # .data RVA -> ptr to hero pointer array

# Alternative hero arrays found (for validation)
HERO_ARRAY_ALT1 = 0x1DBFB58  # second copy
HERO_ARRAY_ALT2 = 0x1DC1EE8  # red team only (partial)

# .data section for scanning
DATA_RVA = 0x1D21000
DATA_SIZE = 0x172B00

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
        if not self.h:
            raise OSError(f"OpenProcess failed: {ctypes.get_last_error()}")
        self._cache = {}

    def read(self, addr, sz):
        buf = ctypes.create_string_buffer(sz)
        n = ctypes.c_size_t(0)
        ok = kernel32.ReadProcessMemory(self.h, ctypes.c_void_p(addr), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None

    def u32(self, a):
        d = self.read(a, 4); return struct.unpack("<I", d)[0] if d else None
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


def find_game_time(m, base):
    """Scan .data for game time float."""
    # Read two snapshots 2s apart
    print("  Scanning for game time (2s wait)...")

    candidates = {}
    chunk_size = 0x40000
    for off in range(0, DATA_SIZE, chunk_size):
        sz = min(chunk_size, DATA_SIZE - off)
        data = m.read(base + DATA_RVA + off, sz)
        if not data:
            continue
        for i in range(0, len(data) - 4, 4):
            val = struct.unpack("<f", data[i:i+4])[0]
            if 1.0 < val < 5000.0:
                candidates[DATA_RVA + off + i] = val

    time.sleep(2)

    for rva, v1 in list(candidates.items()):
        v2 = m.f32(base + rva)
        if v2 is not None:
            diff = v2 - v1
            if 0.5 < diff < 20.0:
                print(f"  GameTime at RVA=0x{rva:X}: {v2:.2f}s (rate={diff/2:.1f}x)")
                return rva

    print("  GameTime not found (game may be paused)")
    return None


def read_heroes(m, base, hero_array_rva):
    """Read all heroes from the hero array."""
    arr_ptr = m.u64(base + hero_array_rva)
    if not is_heap(arr_ptr):
        return []

    heroes = []
    for i in range(10):
        hp = m.u64(arr_ptr + i * 8)
        if not is_heap(hp):
            continue

        name = m.string(hp + 0x4328, 32)
        pos = m.vec3(hp + 0x25C)
        team = m.u32(hp + 0x3C)
        net_id = m.u32(hp + 0xCC)
        health = m.f32(hp + 0x1080)
        max_health = m.f32(hp + 0x10A8)
        mana = m.f32(hp + 0x360)
        max_mana = m.f32(hp + 0x388)
        dead = m.u32(hp + 0x250)
        visible = m.u32(hp + 0x308)

        heroes.append({
            "index": i,
            "ptr": hp,
            "name": name or f"unknown_{i}",
            "team": team,
            "net_id": net_id,
            "position": {"x": pos[0], "y": pos[1], "z": pos[2]} if pos else None,
            "hp": health,
            "max_hp": max_health,
            "mana": mana,
            "max_mana": max_mana,
            "dead": dead,
            "visible": visible,
        })

    return heroes


def snapshot(m, base, hero_ptrs, game_time_rva=None):
    """Take a single snapshot of all game state."""
    gt = None
    if game_time_rva:
        gt = m.f32(base + game_time_rva)

    heroes = []
    for i, hp in enumerate(hero_ptrs):
        pos = m.vec3(hp + 0x25C)
        health = m.f32(hp + 0x1080)
        dead = m.u32(hp + 0x250)
        visible = m.u32(hp + 0x308)

        heroes.append({
            "position": [pos[0], pos[1], pos[2]] if pos else None,
            "hp": health,
            "dead": dead,
            "visible": visible,
        })

    return {
        "game_time": gt,
        "heroes": heroes,
    }


def main():
    # Force UTF-8 output
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 60)
    print("League of Legends Memory Scraper")
    print("=" * 60)

    pid = find_league()
    if not pid:
        print("League not found!"); return
    base, mod_size = find_base(pid)
    if not base:
        print("Module not found!"); return

    print(f"PID: {pid}")
    print(f"Base: 0x{base:X}")
    print(f"Module size: 0x{mod_size:X}")

    m = Mem(pid)

    # Verify RPM
    mz = m.read(base, 2)
    assert mz == b'MZ', "RPM failed!"
    print("ReadProcessMemory: WORKING")

    # Read heroes
    print("\n--- Reading hero array ---")
    heroes = read_heroes(m, base, HERO_ARRAY_RVA)

    if not heroes:
        print("No heroes found at primary RVA, trying alternatives...")
        for alt in [HERO_ARRAY_ALT1, HERO_ARRAY_ALT2]:
            heroes = read_heroes(m, base, alt)
            if heroes:
                print(f"Found heroes at RVA 0x{alt:X}")
                break

    if not heroes:
        print("FATAL: No heroes found in any array!")
        m.close()
        return

    # Print hero table
    print(f"\n{'='*80}")
    print(f"{'#':>2} {'Champion':15s} {'Team':>4} {'NetID':>10} {'Position':>25} {'HP':>15} {'Dead':>4} {'Vis':>3}")
    print(f"{'='*80}")
    for h in heroes:
        pos = h['position']
        pos_str = f"({pos['x']:.0f}, {pos['y']:.0f}, {pos['z']:.0f})" if pos else "N/A"
        hp_str = f"{h['hp']:.0f}/{h['max_hp']:.0f}" if h['hp'] and h['max_hp'] else "?"
        net_str = f"0x{h['net_id']:08X}" if h['net_id'] else "?"
        print(f"{h['index']:2d} {h['name']:15s} {h['team'] or 0:4d} {net_str:>10} {pos_str:>25} {hp_str:>15} {h['dead'] or 0:4d} {h['visible'] or 0:3d}")

    # Get hero pointers for monitoring
    hero_ptrs = [h['ptr'] for h in heroes]
    hero_names = [h['name'] for h in heroes]

    # Find game time
    print("\n--- Finding game time ---")
    gt_rva = find_game_time(m, base)

    # Monitor positions
    print("\n--- Position monitoring (10 samples, 0.5s apart) ---")
    samples = []
    for t in range(10):
        time.sleep(0.5)
        snap = snapshot(m, base, hero_ptrs, gt_rva)
        samples.append(snap)

        gt_str = f"gt={snap['game_time']:.1f}s" if snap['game_time'] else "gt=?"
        parts = [gt_str]
        for i, h in enumerate(snap['heroes']):
            pos = h['position']
            if pos:
                parts.append(f"{hero_names[i]}=({pos[0]:.0f},{pos[2]:.0f})")
        print(f"  [{t}] {' '.join(parts[:6])}...")

    # Check if positions changed
    print("\n--- Position change analysis ---")
    if len(samples) >= 2:
        first = samples[0]
        last = samples[-1]
        for i in range(len(hero_names)):
            p1 = first['heroes'][i]['position']
            p2 = last['heroes'][i]['position']
            if p1 and p2:
                dx = p2[0] - p1[0]
                dz = p2[2] - p1[2]
                dist = (dx*dx + dz*dz)**0.5
                moved = "MOVED" if dist > 1 else "static"
                print(f"  {hero_names[i]:15s} delta=({dx:.1f},{dz:.1f}) dist={dist:.1f} [{moved}]")

    # Save data
    output = {
        "module_base": f"0x{base:X}",
        "module_size": f"0x{mod_size:X}",
        "hero_array_rva": f"0x{HERO_ARRAY_RVA:X}",
        "game_time_rva": f"0x{gt_rva:X}" if gt_rva else None,
        "heroes": heroes,
        "offsets": {
            "CharacterName": "0x4328",
            "Position": "0x25C",
            "Team": "0x3C",
            "NetId": "0xCC",
            "Dead": "0x250",
            "HP": "0x1080",
            "MaxHP": "0x10A8",
            "MP": "0x360",
            "MaxMP": "0x388",
            "AiManager": "0x4038",
        },
        "samples": samples,
    }

    out_path = os.path.join(os.path.dirname(__file__) or '.', 'scraper_output.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    m.close()
    print("Done!")


if __name__ == "__main__":
    main()
