"""
League of Legends Memory Reader — Direct ReadProcessMemory approach.

Reads champion positions, stats, and game state from League process memory.
Requires Vanguard to be DISABLED (vgc + vgk services stopped).

Offsets from ibrahimcelik (patch 26.7, April 2026):
  oLocalPlayer  = 0x1df65a8
  oHeroList     = 0x1dbef80
  oObjPosition  = 0x25C
  oObjNetId     = 0xCC
  oObjName      = 0x4328
  oObjAiManager = 0x4030
  oAiManagerTargetPos  = 0x34
  oAiManagerStartPath  = 0x88
  oAiManagerEndPath    = 0x88

Pattern (from tlol-scraper-pandoras):
  module_base = base address of "League of Legends.exe" module
  local_player_ptr = read_u64(module_base + oLocalPlayer)
  hero_list_ptr    = read_u64(module_base + oHeroList)

  For each hero in hero_list:
    hero_ptr = read_u64(hero_list_ptr + i * 8)
    pos_x    = read_f32(hero_ptr + oObjPosition)
    pos_y    = read_f32(hero_ptr + oObjPosition + 4)
    pos_z    = read_f32(hero_ptr + oObjPosition + 8)
    net_id   = read_u32(hero_ptr + oObjNetId)
    name     = read_str(hero_ptr + oObjName)
"""

import ctypes
import ctypes.wintypes
import struct
import sys
import time
import json
from ctypes import wintypes

# Windows API constants
PROCESS_ALL_ACCESS = 0x1F0FFF
PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
TH32CS_SNAPPROCESS = 0x00000002
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
MAX_MODULE_NAME32 = 255
MAX_PATH = 260

# ibrahimcelik's offsets (patch 26.7)
OFFSETS = {
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

# Structures for Windows API
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

# Load Windows DLLs
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
psapi = ctypes.WinDLL("psapi", use_last_error=True)

def find_league_pid():
    """Find League of Legends.exe process ID."""
    snapshot = kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0)
    if snapshot == -1:
        raise OSError(f"CreateToolhelp32Snapshot failed: {ctypes.get_last_error()}")

    pe = PROCESSENTRY32()
    pe.dwSize = ctypes.sizeof(PROCESSENTRY32)

    try:
        if not kernel32.Process32First(snapshot, ctypes.byref(pe)):
            raise OSError("Process32First failed")

        while True:
            name = pe.szExeFile.decode("utf-8", errors="ignore")
            if "League of Legends" in name:
                return pe.th32ProcessID
            if not kernel32.Process32Next(snapshot, ctypes.byref(pe)):
                break
    finally:
        kernel32.CloseHandle(snapshot)

    raise RuntimeError("League of Legends.exe not found")

def find_module_base(pid, module_name="League of Legends.exe"):
    """Find the base address of a module in the target process."""
    snapshot = kernel32.CreateToolhelp32Snapshot(
        TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, pid
    )
    if snapshot == -1:
        err = ctypes.get_last_error()
        raise OSError(f"CreateToolhelp32Snapshot(MODULE) failed: err={err}")

    me = MODULEENTRY32()
    me.dwSize = ctypes.sizeof(MODULEENTRY32)

    try:
        if not kernel32.Module32First(snapshot, ctypes.byref(me)):
            err = ctypes.get_last_error()
            raise OSError(f"Module32First failed: err={err}")

        while True:
            mod_name = me.szModule.decode("utf-8", errors="ignore")
            if module_name.lower() in mod_name.lower():
                base = ctypes.cast(me.modBaseAddr, ctypes.c_void_p).value
                size = me.modBaseSize
                print(f"[+] Found module: {mod_name}")
                print(f"    Base: 0x{base:X}")
                print(f"    Size: 0x{size:X} ({size / 1024 / 1024:.1f} MB)")
                return base
            if not kernel32.Module32Next(snapshot, ctypes.byref(me)):
                break
    finally:
        kernel32.CloseHandle(snapshot)

    raise RuntimeError(f"Module {module_name} not found in process {pid}")

class MemoryReader:
    def __init__(self, pid):
        self.pid = pid
        self.handle = kernel32.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
        if not self.handle:
            err = ctypes.get_last_error()
            raise OSError(f"OpenProcess failed: err={err}")
        print(f"[+] Opened process {pid}, handle=0x{self.handle:X}")

    def read_bytes(self, address, size):
        """Read raw bytes from process memory."""
        buf = ctypes.create_string_buffer(size)
        bytes_read = ctypes.c_size_t(0)
        ok = kernel32.ReadProcessMemory(
            self.handle,
            ctypes.c_void_p(address),
            buf,
            size,
            ctypes.byref(bytes_read)
        )
        if not ok:
            err = ctypes.get_last_error()
            return None
        return buf.raw[:bytes_read.value]

    def read_u32(self, address):
        data = self.read_bytes(address, 4)
        if data is None or len(data) < 4:
            return None
        return struct.unpack("<I", data)[0]

    def read_u64(self, address):
        data = self.read_bytes(address, 8)
        if data is None or len(data) < 8:
            return None
        return struct.unpack("<Q", data)[0]

    def read_f32(self, address):
        data = self.read_bytes(address, 4)
        if data is None or len(data) < 4:
            return None
        return struct.unpack("<f", data)[0]

    def read_vec3(self, address):
        """Read 3 floats (x, y, z) from address."""
        data = self.read_bytes(address, 12)
        if data is None or len(data) < 12:
            return None
        return struct.unpack("<fff", data)

    def read_string(self, address, max_len=64):
        data = self.read_bytes(address, max_len)
        if data is None:
            return None
        try:
            return data.split(b'\x00')[0].decode('utf-8', errors='ignore')
        except:
            return None

    def close(self):
        if self.handle:
            kernel32.CloseHandle(self.handle)
            self.handle = None

def scan_for_positions(reader, base_addr, hero_ptr):
    """Scan around common offset ranges for position-like float triplets."""
    print(f"\n  Scanning hero_ptr 0x{hero_ptr:X} for position floats...")
    candidates = []

    # Scan a range of offsets where position might be
    for offset in range(0x200, 0x400, 4):
        vec = reader.read_vec3(hero_ptr + offset)
        if vec is None:
            continue
        x, y, z = vec
        # League map is roughly 0-15000 on X and Z, Y (height) is usually 50-500
        if 0 < x < 16000 and -500 < y < 1000 and 0 < z < 16000:
            candidates.append((offset, x, y, z))

    if candidates:
        print(f"  Found {len(candidates)} position candidates:")
        for off, x, y, z in candidates:
            print(f"    offset 0x{off:03X}: ({x:.1f}, {y:.1f}, {z:.1f})")
    else:
        print("  No position candidates found in 0x200-0x400 range")

    return candidates

def main():
    print("=" * 60)
    print("League of Legends Memory Reader")
    print("Offsets: ibrahimcelik patch 26.7")
    print("=" * 60)

    # Step 1: Find League process
    print("\n[1] Finding League of Legends process...")
    pid = find_league_pid()
    print(f"[+] League PID: {pid}")

    # Step 2: Find module base
    print("\n[2] Finding module base address...")
    try:
        module_base = find_module_base(pid)
    except OSError as e:
        print(f"[-] Module enumeration failed: {e}")
        print("    This usually means Vanguard is still protecting the process.")
        print("    Make sure vgc and vgk services are STOPPED.")
        return

    # Step 3: Open process for reading
    print("\n[3] Opening process for memory reading...")
    reader = MemoryReader(pid)

    # Step 4: Read base pointers
    print("\n[4] Reading base pointers...")
    local_player_ptr = reader.read_u64(module_base + OFFSETS["oLocalPlayer"])
    hero_list_ptr = reader.read_u64(module_base + OFFSETS["oHeroList"])

    print(f"  LocalPlayer ptr: ", end="")
    if local_player_ptr:
        print(f"0x{local_player_ptr:X}")
    else:
        print("FAILED TO READ")

    print(f"  HeroList ptr:    ", end="")
    if hero_list_ptr:
        print(f"0x{hero_list_ptr:X}")
    else:
        print("FAILED TO READ")

    if not local_player_ptr or not hero_list_ptr:
        print("\n[-] Base pointer reads failed. Possible causes:")
        print("    1. Offsets are wrong for this patch version")
        print("    2. Vanguard kernel hooks still active (try reboot)")
        print("    3. Need administrator privileges")

        # Try a basic read test
        print("\n[*] Basic read test - reading first 16 bytes at module base...")
        data = reader.read_bytes(module_base, 16)
        if data:
            print(f"    Success! Data: {data.hex()}")
            if data[:2] == b'MZ':
                print("    [+] MZ header confirmed - ReadProcessMemory works!")
                print("    [*] Offsets may just be wrong. Trying offset scan...")
        else:
            print("    FAILED - ReadProcessMemory is blocked")
            print("    Vanguard kernel driver may still have hooks loaded.")
            print("    Try: reboot with vgc/vgk disabled, THEN launch League")
            reader.close()
            return

    # Step 5: Read hero list
    print("\n[5] Reading hero list...")
    heroes = []

    if hero_list_ptr:
        # Hero list is typically: [count/size, ptr1, ptr2, ...]
        # Or it could be a pointer to an array struct
        # Try reading as direct array of pointers first
        for i in range(10):  # 10 champions max
            hero_ptr = reader.read_u64(hero_list_ptr + i * 8)
            if hero_ptr is None or hero_ptr == 0 or hero_ptr > 0x7FFFFFFFFFFF:
                continue

            # Try reading position
            pos = reader.read_vec3(hero_ptr + OFFSETS["oObjPosition"])
            net_id = reader.read_u32(hero_ptr + OFFSETS["oObjNetId"])
            name = reader.read_string(hero_ptr + OFFSETS["oObjName"])

            hero = {
                "index": i,
                "ptr": f"0x{hero_ptr:X}",
                "net_id": f"0x{net_id:X}" if net_id else None,
                "name": name,
                "position": pos,
            }
            heroes.append(hero)

            pos_str = f"({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})" if pos else "FAILED"
            print(f"  Hero[{i}]: ptr=0x{hero_ptr:X} name={name or '?':15s} "
                  f"netId={net_id or 0:#x} pos={pos_str}")

    # Step 5b: Try alternative hero list format (pointer to struct with size + array)
    if not heroes and hero_list_ptr:
        print("\n  Trying alternative hero list format (struct with header)...")
        # Some versions have: [vtable, size, ptr_array_start, ptr_array_end]
        for header_offset in [0, 8, 0x10, 0x18]:
            arr_ptr = reader.read_u64(hero_list_ptr + header_offset)
            if arr_ptr and 0 < arr_ptr < 0x7FFFFFFFFFFF:
                hero_ptr = reader.read_u64(arr_ptr)
                if hero_ptr and 0 < hero_ptr < 0x7FFFFFFFFFFF:
                    name = reader.read_string(hero_ptr + OFFSETS["oObjName"])
                    pos = reader.read_vec3(hero_ptr + OFFSETS["oObjPosition"])
                    if name and len(name) > 2:
                        print(f"  Found via header+0x{header_offset:X}: "
                              f"name={name} pos={pos}")

    # Step 6: Read local player
    print("\n[6] Reading local player...")
    if local_player_ptr and 0 < local_player_ptr < 0x7FFFFFFFFFFF:
        pos = reader.read_vec3(local_player_ptr + OFFSETS["oObjPosition"])
        name = reader.read_string(local_player_ptr + OFFSETS["oObjName"])
        net_id = reader.read_u32(local_player_ptr + OFFSETS["oObjNetId"])

        print(f"  Name:   {name}")
        print(f"  NetID:  {net_id:#x if net_id else 'FAILED'}")
        print(f"  Pos:    {pos}")

        # Try AI manager for movement path
        ai_mgr_ptr = reader.read_u64(local_player_ptr + OFFSETS["oObjAiManager"])
        if ai_mgr_ptr and 0 < ai_mgr_ptr < 0x7FFFFFFFFFFF:
            target_pos = reader.read_vec3(ai_mgr_ptr + OFFSETS["oAiManagerTargetPos"])
            print(f"  AiMgr:  0x{ai_mgr_ptr:X}")
            print(f"  Target: {target_pos}")

        # Scan for position if offset didn't work
        if pos is None or not (0 < pos[0] < 16000):
            scan_for_positions(reader, module_base, local_player_ptr)

    # Step 7: Game time
    print("\n[7] Checking known game time offsets...")
    # pandoras used 0x21FE6F8 for patch 13.23
    # We don't have a game time offset from ibrahimcelik, try common patterns
    for name_hint, test_offset in [
        ("ibrahimcelik LocalPlayer-area", OFFSETS["oLocalPlayer"] - 0x100),
        ("ibrahimcelik LocalPlayer+area", OFFSETS["oLocalPlayer"] + 0x100),
    ]:
        val = reader.read_f32(module_base + test_offset)
        if val and 0 < val < 10000:
            print(f"  {name_hint} (0x{test_offset:X}): {val:.2f}s")

    # Step 8: Continuous monitoring mode
    if heroes or (local_player_ptr and 0 < local_player_ptr < 0x7FFFFFFFFFFF):
        print("\n[8] Continuous position monitoring (5 samples, 1s apart)...")
        for sample in range(5):
            time.sleep(1)
            positions = {}

            if local_player_ptr and 0 < local_player_ptr < 0x7FFFFFFFFFFF:
                pos = reader.read_vec3(local_player_ptr + OFFSETS["oObjPosition"])
                if pos:
                    positions["local_player"] = pos

            for hero in heroes:
                hero_ptr = int(hero["ptr"], 16)
                pos = reader.read_vec3(hero_ptr + OFFSETS["oObjPosition"])
                if pos:
                    positions[hero.get("name", f"hero_{hero['index']}")] = pos

            print(f"  t+{sample+1}s: {positions}")

    print("\n" + "=" * 60)
    print("Done!")
    reader.close()

if __name__ == "__main__":
    main()
