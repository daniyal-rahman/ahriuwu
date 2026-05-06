"""
League Memory Reader v2 — Diagnostic deep-scan.

ReadProcessMemory confirmed working. Now we need to:
1. Understand the hero list structure (it's not a simple pointer array)
2. Find valid champion objects on the heap
3. Scan for position-like floats near known offsets

Known from v1:
  module_base = 0x7FF75BAC0000
  base + oHeroList (0x1dbef80) → 0x1369FFCDC50 (valid heap pointer)
  base + oLocalPlayer (0x1df65a8) → NULL (offset wrong or replay quirk)
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

# ibrahimcelik offsets (patch 26.7)
O = {
    "oLocalPlayer":       0x1df65a8,
    "oHeroList":          0x1dbef80,
    "oObjPosition":       0x25C,
    "oObjNetId":          0xCC,
    "oObjName":           0x4328,
    "oObjAiManager":      0x4030,
    "oAiManagerTargetPos": 0x34,
    "oAiManagerStartPath": 0x88,
    "oAiManagerEndPath":   0x88,
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
        if d is None: return None
        return d.split(b'\x00')[0].decode('utf-8', errors='replace')

    def close(self):
        kernel32.CloseHandle(self.h)

def is_valid_ptr(v):
    """Check if value looks like a valid 64-bit pointer."""
    return v is not None and 0x10000 < v < 0x7FFFFFFFFFFF

def dump_region(m, addr, name, size=128):
    """Dump a region of memory as hex + pointers."""
    print(f"\n{'='*60}")
    print(f"Dump: {name} @ 0x{addr:X}")
    print(f"{'='*60}")
    data = m.read(addr, size)
    if not data:
        print("  FAILED TO READ")
        return

    # Print hex dump
    for i in range(0, len(data), 16):
        hex_str = ' '.join(f'{b:02X}' for b in data[i:i+16])
        ascii_str = ''.join(chr(b) if 32 <= b < 127 else '.' for b in data[i:i+16])
        print(f"  +0x{i:03X}: {hex_str:<48s} {ascii_str}")

    # Print as pointers
    print(f"\n  As u64 pointers:")
    for i in range(0, min(size, 64), 8):
        val = struct.unpack("<Q", data[i:i+8])[0]
        marker = " <-- VALID PTR" if is_valid_ptr(val) else ""
        as_float = struct.unpack("<f", data[i:i+4])[0]
        float_str = f" (f32={as_float:.1f})" if abs(as_float) < 100000 and as_float != 0 else ""
        print(f"    +0x{i:02X}: 0x{val:016X}{marker}{float_str}")

def explore_hero_list(m, base, hero_list_raw_ptr):
    """Try multiple hero list structure formats."""
    print(f"\n{'='*60}")
    print(f"Exploring hero list structure at 0x{hero_list_raw_ptr:X}")
    print(f"{'='*60}")

    # Dump the hero list region
    dump_region(m, hero_list_raw_ptr, "HeroList raw", 256)

    # Common ManagerTemplate patterns:
    # Pattern A: [vtable, array_ptr, array_end, size]
    # Pattern B: [vtable, size, array_ptr]
    # Pattern C: direct array of pointers
    # Pattern D: [vtable, list_start, list_end, ...]

    for offset in [0x0, 0x8, 0x10, 0x18, 0x20, 0x28]:
        ptr = m.u64(hero_list_raw_ptr + offset)
        if not is_valid_ptr(ptr):
            continue

        print(f"\n  --- Trying array at hero_list+0x{offset:X} → 0x{ptr:X} ---")

        # Read first 10 potential hero pointers
        found_any = False
        for i in range(10):
            hp = m.u64(ptr + i * 8)
            if not is_valid_ptr(hp):
                continue

            # Try reading name at various offsets
            for name_off in [O["oObjName"], 0x60, 0x68, 0x70, 0x2E64, 0x3024]:
                name = m.string(hp + name_off, 32)
                if name and len(name) >= 3 and name.isascii() and name[0].isupper():
                    pos = m.vec3(hp + O["oObjPosition"])
                    pos_str = f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})" if pos else "?"
                    print(f"    [hero {i}] ptr=0x{hp:X} name@+0x{name_off:X}='{name}' pos@+0x{O['oObjPosition']:X}={pos_str}")
                    found_any = True
                    break

            if not found_any:
                # Scan for champion name strings
                for scan_off in range(0, 0x5000, 8):
                    name = m.string(hp + scan_off, 32)
                    if name and any(champ in name for champ in
                        ["Garen", "Ahri", "Aatrox", "Annie", "Ashe", "Jinx",
                         "Lux", "Yasuo", "Zed", "Irelia", "Darius", "Riven",
                         "Fizz", "Mundo", "Corki", "Milio", "Volibear", "Ryze",
                         "Caitlyn", "Bard", "Lissandra", "Varus", "Sona",
                         "Thresh", "Renekton", "Orianna", "KSante", "Smolder",
                         "Draven", "Kayn", "Nasus", "Teemo"]):
                        print(f"    [hero {i}] ptr=0x{hp:X} FOUND NAME at +0x{scan_off:X}: '{name}'")
                        found_any = True
                        break
                    if scan_off > 0x1000 and not found_any:
                        break  # Don't scan too far

        if not found_any:
            # Try the first pointer and dump it
            hp = m.u64(ptr)
            if is_valid_ptr(hp):
                print(f"    First obj at 0x{hp:X} - dumping...")
                dump_region(m, hp, f"First hero obj via +0x{offset:X}", 128)

def scan_for_game_time(m, base, mod_size):
    """Scan module data sections for a float that looks like game time (increases each second)."""
    print(f"\n{'='*60}")
    print(f"Scanning for game time float...")
    print(f"{'='*60}")

    # Read two samples 2 seconds apart
    # Game time should be a float that increases by ~2.0
    candidates = []

    # Scan .data section (usually at higher offsets in the module)
    # The module is 32.5MB. Data section is typically in the last ~5MB
    scan_ranges = [
        (0x1D00000, 0x2070000, "high data section"),  # Where our offsets point
    ]

    for start, end, label in scan_ranges:
        print(f"\n  Scanning {label} (0x{start:X} - 0x{end:X})...")
        chunk_size = 0x10000  # 64KB chunks

        # First pass
        t1_values = {}
        for chunk_start in range(start, end, chunk_size):
            data = m.read(base + chunk_start, chunk_size)
            if not data:
                continue
            for i in range(0, len(data) - 4, 4):
                val = struct.unpack("<f", data[i:i+4])[0]
                # Game time in a replay: 0 to ~2400 (40 min)
                if 10.0 < val < 3000.0 and val == int(val * 10) / 10:  # rough check
                    offset = chunk_start + i
                    t1_values[offset] = val

        if not t1_values:
            print(f"    No candidates in first pass")
            continue

        print(f"    {len(t1_values)} candidates in first pass, waiting 2s...")
        time.sleep(2)

        # Second pass
        for offset, v1 in list(t1_values.items()):
            v2 = m.f32(base + offset)
            if v2 is None:
                continue
            diff = v2 - v1
            # Game time should increase by ~2.0 (or by replay speed * 2.0)
            if 0.5 < diff < 20.0:
                candidates.append((offset, v1, v2, diff))
                if len(candidates) <= 20:
                    print(f"    CANDIDATE: offset=0x{offset:X} v1={v1:.2f} v2={v2:.2f} delta={diff:.2f}")

    print(f"\n  Total candidates: {len(candidates)}")
    return candidates

def scan_for_champion_strings(m, base, mod_size):
    """Scan module memory for pointers to champion name strings."""
    print(f"\n{'='*60}")
    print(f"Scanning for champion name pointers in module data...")
    print(f"{'='*60}")

    # Known champion names that might be in this game
    champ_names = [b"Garen", b"Irelia", b"Ahri", b"Darius", b"Mundo",
                   b"Fizz", b"Corki", b"Milio", b"Aatrox", b"Volibear",
                   b"Ryze", b"Caitlyn", b"Bard"]

    # Scan the data sections
    scan_start = 0x1D00000
    scan_end = min(0x2070000, mod_size)
    chunk_size = 0x100000  # 1MB

    found_ptrs = []
    for chunk_start in range(scan_start, scan_end, chunk_size):
        data = m.read(base + chunk_start, chunk_size)
        if not data:
            continue

        # Look for 8-byte values that are valid pointers
        for i in range(0, len(data) - 8, 8):
            ptr = struct.unpack("<Q", data[i:i+8])[0]
            if not is_valid_ptr(ptr):
                continue

            # Read what this pointer points to
            target = m.read(ptr, 32)
            if target is None:
                continue

            for cname in champ_names:
                if target.startswith(cname):
                    offset = chunk_start + i
                    name_str = target.split(b'\x00')[0].decode('utf-8', errors='replace')
                    print(f"  0x{offset:X}: ptr=0x{ptr:X} → '{name_str}'")
                    found_ptrs.append((offset, ptr, name_str))
                    break

    print(f"\n  Found {len(found_ptrs)} champion name pointers")
    return found_ptrs

def main():
    print("League Memory Reader v2 — Deep Diagnostic")
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
    print(f"Base: 0x{base:X}, Size: 0x{mod_size:X}")

    m = Mem(pid)

    # Verify read works
    mz = m.read(base, 2)
    assert mz == b'MZ', f"MZ check failed: {mz}"
    print("ReadProcessMemory: CONFIRMED WORKING")

    # Dump the regions around our offsets
    print("\n" + "=" * 60)
    print("STEP 1: Raw pointer reads at ibrahimcelik offsets")
    print("=" * 60)

    lp = m.u64(base + O["oLocalPlayer"])
    hl = m.u64(base + O["oHeroList"])
    print(f"  base + oLocalPlayer (0x{O['oLocalPlayer']:X}) = {'0x'+format(lp,'X') if lp else 'NULL'}")
    print(f"  base + oHeroList    (0x{O['oHeroList']:X}) = {'0x'+format(hl,'X') if hl else 'NULL'}")

    # Dump memory around the offset locations
    dump_region(m, base + O["oLocalPlayer"] - 0x20, "Around oLocalPlayer", 128)
    dump_region(m, base + O["oHeroList"] - 0x20, "Around oHeroList", 128)

    # Explore hero list structure
    if hl and is_valid_ptr(hl):
        explore_hero_list(m, base, hl)

    # Scan for game time
    game_time_candidates = scan_for_game_time(m, base, mod_size)

    # If we found game time candidates, the offsets are in the right ballpark
    if game_time_candidates:
        print("\n  Game time found! Offsets are in the right region.")
        # Use the best candidate to estimate replay speed
        off, v1, v2, delta = game_time_candidates[0]
        print(f"  Best: offset=0x{off:X}, game_time≈{v2:.1f}s, speed≈{delta/2:.1f}x")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
