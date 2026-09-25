"""
League Memory Scanner — Find correct offsets by scanning for patterns.

Strategy:
1. Parse PE headers to find actual .data section boundaries
2. Scan .data for game time float (changes in real-time)
3. Scan .data for valid heap pointers near each other (global pointer table)
4. For each heap pointer, check if target looks like a champion object
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
            return s if s else None
        except:
            return None

    def close(self):
        kernel32.CloseHandle(self.h)

def is_valid_heap_ptr(v):
    """Check if value looks like a valid heap pointer (high address space)."""
    return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF

def parse_pe_sections(m, base):
    """Parse PE headers to find section boundaries."""
    # DOS header: e_lfanew at offset 0x3C
    e_lfanew = m.u32(base + 0x3C)
    if not e_lfanew or e_lfanew > 0x1000:
        return []

    pe_sig = m.u32(base + e_lfanew)
    if pe_sig != 0x00004550:  # "PE\0\0"
        return []

    # COFF header at PE + 4
    coff = base + e_lfanew + 4
    num_sections = m.u32(coff) & 0xFFFF  # NumberOfSections is u16 at offset 2
    num_sections = struct.unpack("<H", m.read(coff + 2, 2))[0]
    opt_header_size = struct.unpack("<H", m.read(coff + 16, 2))[0]

    # Section headers start after optional header
    section_start = coff + 20 + opt_header_size
    sections = []

    for i in range(num_sections):
        sec_addr = section_start + i * 40
        sec_data = m.read(sec_addr, 40)
        if not sec_data:
            break

        name = sec_data[:8].rstrip(b'\x00').decode('ascii', errors='replace')
        virt_size = struct.unpack("<I", sec_data[8:12])[0]
        virt_addr = struct.unpack("<I", sec_data[12:16])[0]
        raw_size = struct.unpack("<I", sec_data[16:20])[0]
        chars = struct.unpack("<I", sec_data[36:40])[0]

        sections.append({
            "name": name,
            "virt_addr": virt_addr,
            "virt_size": virt_size,
            "raw_size": raw_size,
            "characteristics": chars,
            "readable": bool(chars & 0x40000000),
            "writable": bool(chars & 0x80000000),
            "executable": bool(chars & 0x20000000),
        })

    return sections

def scan_for_game_time(m, base, sections):
    """Scan writable sections for a float that increases over time."""
    print("\n=== SCANNING FOR GAME TIME ===")

    # Find writable sections (typically .data)
    data_sections = [s for s in sections if s["writable"] and not s["executable"]]
    if not data_sections:
        print("  No writable sections found, scanning all non-code sections")
        data_sections = [s for s in sections if not s["executable"]]

    for sec in data_sections:
        print(f"  Section '{sec['name']}': RVA=0x{sec['virt_addr']:X}, "
              f"Size=0x{sec['virt_size']:X} "
              f"({'RW' if sec['writable'] else 'R'}{'+X' if sec['executable'] else ''})")

    # First pass: read all float candidates
    print("\n  Pass 1: Reading candidate floats...")
    candidates = {}
    for sec in data_sections:
        start = sec["virt_addr"]
        size = sec["virt_size"]
        chunk_size = 0x40000  # 256KB chunks for speed

        for chunk_off in range(0, size, chunk_size):
            read_size = min(chunk_size, size - chunk_off)
            data = m.read(base + start + chunk_off, read_size)
            if not data:
                continue

            for i in range(0, len(data) - 4, 4):
                val = struct.unpack("<f", data[i:i+4])[0]
                # Game time: 0 to ~3600s (60 min). Be generous.
                if 1.0 < val < 5000.0:
                    offset = start + chunk_off + i
                    candidates[offset] = val

    print(f"  Found {len(candidates)} candidate floats in range [1, 5000]")

    if len(candidates) > 50000:
        print("  Too many candidates, narrowing to [30, 3000]...")
        candidates = {k: v for k, v in candidates.items() if 30 < v < 3000}
        print(f"  Narrowed to {len(candidates)} candidates")

    # Wait and re-read
    print("  Waiting 2 seconds...")
    time.sleep(2)

    print("  Pass 2: Checking which floats changed...")
    game_time_candidates = []
    for offset, v1 in candidates.items():
        v2 = m.f32(base + offset)
        if v2 is None:
            continue
        diff = v2 - v1
        # Game time increases at 1x-8x speed during replay
        if 1.0 < diff < 20.0:
            game_time_candidates.append((offset, v1, v2, diff))

    game_time_candidates.sort(key=lambda x: abs(x[3] - 2.0))  # closest to 2s increase

    print(f"\n  Found {len(game_time_candidates)} game time candidates:")
    for off, v1, v2, d in game_time_candidates[:20]:
        print(f"    offset=0x{off:X} v1={v1:.2f} v2={v2:.2f} delta={d:.3f}s/2s")

    return game_time_candidates

def scan_for_pointer_clusters(m, base, sections):
    """Scan writable sections for clusters of valid heap pointers (global pointer table)."""
    print("\n=== SCANNING FOR POINTER CLUSTERS ===")

    data_sections = [s for s in sections if s["writable"] and not s["executable"]]

    clusters = []
    for sec in data_sections:
        start = sec["virt_addr"]
        size = sec["virt_size"]
        chunk_size = 0x40000

        for chunk_off in range(0, size, chunk_size):
            read_size = min(chunk_size, size - chunk_off)
            data = m.read(base + start + chunk_off, read_size)
            if not data:
                continue

            # Find runs of valid heap pointers
            run_start = None
            run_count = 0

            for i in range(0, len(data) - 8, 8):
                val = struct.unpack("<Q", data[i:i+8])[0]
                if is_valid_heap_ptr(val):
                    if run_start is None:
                        run_start = start + chunk_off + i
                    run_count += 1
                else:
                    if run_count >= 5:  # At least 5 consecutive pointers
                        clusters.append((run_start, run_count))
                    run_start = None
                    run_count = 0

            if run_count >= 5:
                clusters.append((run_start, run_count))

    clusters.sort(key=lambda x: -x[1])  # largest clusters first
    print(f"  Found {len(clusters)} pointer clusters (>=5 consecutive pointers)")

    for off, count in clusters[:15]:
        print(f"    offset=0x{off:X} count={count}")

        # Check first few pointers for champion names
        for i in range(min(count, 5)):
            ptr = m.u64(base + off + i * 8)
            if not is_valid_heap_ptr(ptr):
                continue

            # Check for champion name at known offset
            for name_off in [0x58, 0x4330, 0x4328]:
                name = m.string(ptr + name_off, 32)
                if name and len(name) > 2 and name[0].isupper() and name.isalpha():
                    print(f"      [ptr {i}] -> 0x{ptr:X} + 0x{name_off:X} = '{name}'")

    return clusters

def probe_champion_at_ptr(m, ptr):
    """Check multiple name offsets to find champion name."""
    for name_off in [0x58, 0x4330, 0x4328, 0x60, 0x68]:
        name = m.string(ptr + name_off, 32)
        if name and len(name) >= 3 and name[0].isupper() and all(c.isalpha() for c in name):
            return name, name_off
    return None, None

def deep_scan_hero_pointers(m, base, sections, known_champs=None):
    """Scan data sections for pointers to objects containing champion names."""
    print("\n=== DEEP SCAN FOR HERO POINTERS ===")
    if known_champs is None:
        known_champs = {"Garen", "Ahri", "Irelia", "Darius", "Mundo", "Fizz",
                        "Corki", "Milio", "Aatrox", "Volibear", "Ryze",
                        "Caitlyn", "Bard", "Lissandra", "Varus", "Sona",
                        "Thresh", "Renekton", "Orianna", "KSante", "Smolder",
                        "Draven", "Kayn", "Nasus", "Teemo", "Jinx", "Lux",
                        "Yasuo", "Zed", "Annie", "Ashe", "Riven", "Malphite",
                        "Sett", "Yone", "Jhin", "Xerath", "Syndra", "Viego",
                        "Samira", "Yuumi"}

    data_sections = [s for s in sections if s["writable"] and not s["executable"]]

    hero_ptrs = {}  # ptr -> name
    print("  Scanning data sections for heap pointers to champion objects...")

    for sec in data_sections:
        start = sec["virt_addr"]
        size = sec["virt_size"]
        chunk_size = 0x40000

        for chunk_off in range(0, size, chunk_size):
            read_size = min(chunk_size, size - chunk_off)
            data = m.read(base + start + chunk_off, read_size)
            if not data:
                continue

            for i in range(0, len(data) - 8, 8):
                val = struct.unpack("<Q", data[i:i+8])[0]
                if not is_valid_heap_ptr(val):
                    continue

                # Quick check: try reading champion name at known offset
                name, name_off = probe_champion_at_ptr(m, val)
                if name and name in known_champs:
                    rva = start + chunk_off + i
                    if val not in hero_ptrs:
                        hero_ptrs[val] = name
                        pos = m.vec3(val + 0x25C)
                        pos_str = f"({pos[0]:.0f}, {pos[1]:.0f}, {pos[2]:.0f})" if pos else "?"
                        print(f"    RVA=0x{rva:X}: ptr=0x{val:X} name='{name}' "
                              f"(at +0x{name_off:X}) pos={pos_str}")

    print(f"\n  Total unique champion objects found: {len(hero_ptrs)}")
    return hero_ptrs

def main():
    print("=" * 60)
    print("League Memory Scanner")
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
    mz = m.read(base, 2)
    assert mz == b'MZ'
    print("RPM: OK\n")

    # Parse PE sections
    sections = parse_pe_sections(m, base)
    print("PE Sections:")
    for s in sections:
        flags = ('R' if s['readable'] else '') + ('W' if s['writable'] else '') + ('X' if s['executable'] else '')
        print(f"  {s['name']:8s} RVA=0x{s['virt_addr']:08X} Size=0x{s['virt_size']:08X} [{flags}]")

    # Scan for game time
    gt_candidates = scan_for_game_time(m, base, sections)

    # Scan for pointer clusters
    clusters = scan_for_pointer_clusters(m, base, sections)

    # Deep scan for champion pointers
    hero_ptrs = deep_scan_hero_pointers(m, base, sections)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if gt_candidates:
        off, v1, v2, d = gt_candidates[0]
        print(f"  GameTime offset: 0x{off:X} (value={v2:.1f}s, rate={d/2:.1f}x)")

    if hero_ptrs:
        print(f"  Champion objects: {len(hero_ptrs)}")
        for ptr, name in hero_ptrs.items():
            pos = m.vec3(ptr + 0x25C)
            team = m.u32(ptr + 0x3C)
            net_id = m.u32(ptr + 0xCC)
            hp = m.f32(ptr + 0x1080)
            maxhp = m.f32(ptr + 0x10A8)
            pos_str = f"({pos[0]:.0f},{pos[1]:.0f},{pos[2]:.0f})" if pos else "?"
            hp_str = f"{hp:.0f}/{maxhp:.0f}" if hp and maxhp else "?"
            print(f"    {name:15s} team={team} netId={net_id:#x if net_id else 0} "
                  f"pos={pos_str} hp={hp_str}")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
