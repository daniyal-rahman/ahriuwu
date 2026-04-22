"""
League Memory Trace — Found 4 hero objects, now find the hero manager array.

Known hero object addresses (from scan):
  0x1372A5C7FD0 = Smolder, pos=(1984, 96, 1218)
  0x13734818020 = Ashe, pos=(5147, 50, 4683)
  0x1372D074DD0 = Garen, pos=(4646, 51, 4180)
  0x13732A60020 = Malphite, pos=(2148, 96, 1974)

Found in .data at RVAs: 0x1DFA800, 0x1DFAC20, 0x1DFAE30, 0x1DFAF38

Strategy:
1. Read the .data section around these RVAs to find the hero manager structure
2. Search heap for an array containing multiple hero pointers
3. Dump full hero state once found
"""

import ctypes
import ctypes.wintypes
import struct
import time
from ctypes import wintypes

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
TH32CS_SNAPPROCESS = 0x00000002
TH32CS_SNAPMODULE = 0x00000008
TH32CS_SNAPMODULE32 = 0x00000010
MAX_MODULE_NAME32 = 255
MAX_PATH = 260

# Confirmed offsets
OBJ = {
    "Index":          0x10,
    "Team":           0x3C,
    "Name":           0x58,
    "NetId":          0xCC,
    "Dead":           0x250,
    "Position":       0x25C,
    "Visible":        0x308,
    "Radius":         0x6F8,
    "CharacterName":  0x4328,
    "AiManager":      0x4038,
}

HP_OFFSET = 0x1080
MAXHP_OFFSET = 0x10A8
MP_OFFSET = 0x360
MAXMP_OFFSET = 0x388

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
        return d.split(b'\x00')[0].decode('utf-8', errors='replace') or None
    def close(self):
        kernel32.CloseHandle(self.h)

def is_heap(v):
    return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF

KNOWN_CHAMPS = {"Garen","Ahri","Irelia","Darius","Mundo","Fizz","Corki","Milio",
                "Aatrox","Volibear","Ryze","Caitlyn","Bard","Lissandra","Varus",
                "Sona","Thresh","Renekton","Orianna","KSante","Smolder","Draven",
                "Kayn","Nasus","Teemo","Jinx","Lux","Yasuo","Zed","Annie","Ashe",
                "Riven","Malphite","Sett","Yone","Jhin","Xerath","Syndra","Viego",
                "Samira","Yuumi","Senna","Seraphine","Zeri","Nilah","Naafiri",
                "Briar","Hwei","Aurora","Ambessa","Milio","Mel"}

def read_hero(m, ptr):
    """Read full hero state from a champion object pointer."""
    name = m.string(ptr + OBJ["CharacterName"])
    if not name or name not in KNOWN_CHAMPS:
        name = m.string(ptr + OBJ["Name"])
    pos = m.vec3(ptr + OBJ["Position"])
    team = m.u32(ptr + OBJ["Team"])
    net_id = m.u32(ptr + OBJ["NetId"])
    hp = m.f32(ptr + HP_OFFSET)
    maxhp = m.f32(ptr + MAXHP_OFFSET)
    dead = m.u32(ptr + OBJ["Dead"])

    return {
        "ptr": ptr,
        "name": name,
        "team": team,
        "net_id": net_id,
        "position": pos,
        "hp": hp,
        "max_hp": maxhp,
        "dead": dead,
    }

def main():
    print("=" * 60)
    print("League Memory Trace - Hero Finder")
    print("=" * 60)

    pid = find_league()
    if not pid:
        print("League not found!"); return
    base, mod_size = find_base(pid)
    if not base:
        print("Module not found!"); return
    print(f"PID={pid} Base=0x{base:X} Size=0x{mod_size:X}")

    m = Mem(pid)

    # .data section: RVA=0x1D21000 Size=0x172B00
    data_rva = 0x1D21000
    data_size = 0x172B00
    data_start = base + data_rva
    data_end = data_start + data_size

    # ================================================================
    # STEP 1: Find all champion objects in .data section
    # ================================================================
    print("\n--- STEP 1: Find all champion objects ---")
    hero_objs = {}  # ptr -> name
    hero_rvas = {}  # rva -> ptr

    chunk_size = 0x40000
    for off in range(0, data_size, chunk_size):
        sz = min(chunk_size, data_size - off)
        data = m.read(data_start + off, sz)
        if not data:
            continue
        for i in range(0, len(data) - 8, 8):
            val = struct.unpack("<Q", data[i:i+8])[0]
            if not is_heap(val):
                continue
            # Check +0x4328 for champion name
            cname = m.string(val + 0x4328, 32)
            if cname and cname in KNOWN_CHAMPS and val not in hero_objs:
                hero_objs[val] = cname
                rva = data_rva + off + i
                hero_rvas[rva] = val
                pos = m.vec3(val + 0x25C)
                pos_str = "({:.0f},{:.0f},{:.0f})".format(*pos) if pos else "?"
                print(f"  RVA=0x{rva:X} ptr=0x{val:X} {cname:15s} pos={pos_str}")

    print(f"\n  Found {len(hero_objs)} unique champion objects")

    if not hero_objs:
        print("  No champions found! The game might not be loaded.")
        # Try the +0x4330 offset too
        print("\n  Trying CharacterName at +0x4330...")
        for off in range(0, data_size, chunk_size):
            sz = min(chunk_size, data_size - off)
            data = m.read(data_start + off, sz)
            if not data: continue
            for i in range(0, len(data) - 8, 8):
                val = struct.unpack("<Q", data[i:i+8])[0]
                if not is_heap(val): continue
                cname = m.string(val + 0x4330, 32)
                if cname and cname in KNOWN_CHAMPS and val not in hero_objs:
                    hero_objs[val] = cname
                    pos = m.vec3(val + 0x25C)
                    pos_str = "({:.0f},{:.0f},{:.0f})".format(*pos) if pos else "?"
                    print(f"    ptr=0x{val:X} {cname:15s} pos={pos_str}")

    # ================================================================
    # STEP 2: Search for an array of hero pointers on the heap
    # ================================================================
    print("\n--- STEP 2: Search for hero pointer array ---")

    # The hero manager on heap should contain a contiguous array of 10 hero pointers
    # Strategy: for each hero ptr, search .data for pointers-to-pointers
    hero_ptr_set = set(hero_objs.keys())

    if len(hero_objs) >= 2:
        # Take two known hero pointers and find where they are stored contiguously
        hero_list = list(hero_objs.keys())

        print("  Searching .data for pointers that reference hero arrays...")
        for off in range(0, data_size, chunk_size):
            sz = min(chunk_size, data_size - off)
            data = m.read(data_start + off, sz)
            if not data: continue

            for i in range(0, len(data) - 8, 8):
                val = struct.unpack("<Q", data[i:i+8])[0]
                if not is_heap(val):
                    continue

                # Read 10 pointers from this address
                arr_data = m.read(val, 80)  # 10 * 8 bytes
                if not arr_data:
                    continue

                # Count how many point to known heroes
                matches = 0
                for j in range(10):
                    arr_val = struct.unpack("<Q", arr_data[j*8:(j+1)*8])[0]
                    if arr_val in hero_ptr_set:
                        matches += 1

                if matches >= 3:
                    rva = data_rva + off + i
                    print(f"\n  HERO ARRAY FOUND at RVA=0x{rva:X}")
                    print(f"  Array ptr: 0x{val:X} ({matches}/10 matches)")

                    # Read and print all 10 entries
                    for j in range(10):
                        arr_val = struct.unpack("<Q", arr_data[j*8:(j+1)*8])[0]
                        if is_heap(arr_val):
                            cname = m.string(arr_val + 0x4328, 32)
                            pos = m.vec3(arr_val + 0x25C)
                            pos_str = "({:.0f},{:.0f},{:.0f})".format(*pos) if pos else "?"
                            print(f"    [{j}] 0x{arr_val:X} {cname or '???':15s} pos={pos_str}")
                        else:
                            print(f"    [{j}] 0x{arr_val:016X} (not a valid ptr)")

                    # Now trace back: what points to this array?
                    # The hero manager global should be: base + offset -> manager_ptr
                    # manager_ptr + X -> array_ptr (= val)
                    # We found array_ptr in .data at rva
                    # But the actual hero manager is a heap object
                    # Let's check nearby .data for the manager pointer
                    print(f"\n  Manager search: array at 0x{val:X}, "
                          f"found via .data RVA=0x{rva:X}")

                    # The RVA where we found the pointer to the array
                    # This is likely: base + HeroManagerOffset -> manager_ptr
                    # And manager_ptr + 0x08 (or similar) -> array_ptr
                    # So we need to find: what heap object contains 0x{val:X} at +0x08?

                    # Search .data for any pointer that, when dereferenced + small offset, gives val
                    print("  Searching for manager object...")
                    for search_off in range(0, data_size, chunk_size):
                        ssz = min(chunk_size, data_size - search_off)
                        sdata = m.read(data_start + search_off, ssz)
                        if not sdata: continue
                        for si in range(0, len(sdata) - 8, 8):
                            mgr_candidate = struct.unpack("<Q", sdata[si:si+8])[0]
                            if not is_heap(mgr_candidate):
                                continue
                            # Check +0x08, +0x10, +0x18 for array pointer
                            for mgr_off in [0x08, 0x10, 0x18]:
                                arr_check = m.u64(mgr_candidate + mgr_off)
                                if arr_check == val:
                                    mgr_rva = data_rva + search_off + si
                                    print(f"    Manager at 0x{mgr_candidate:X} "
                                          f"(.data RVA=0x{mgr_rva:X}), "
                                          f"array at +0x{mgr_off:X}")
                                    # Check count
                                    for cnt_off in [mgr_off + 8, mgr_off + 4, 0x04]:
                                        cnt = m.u32(mgr_candidate + cnt_off)
                                        if cnt and 0 < cnt <= 12:
                                            print(f"    Count at +0x{cnt_off:X} = {cnt}")

    # ================================================================
    # STEP 3: Full hero state dump
    # ================================================================
    print("\n--- STEP 3: Full hero state ---")
    for ptr, name in sorted(hero_objs.items(), key=lambda x: x[1]):
        h = read_hero(m, ptr)
        pos = h["position"]
        pos_str = "({:.0f},{:.0f},{:.0f})".format(*pos) if pos else "N/A"
        hp_str = "{:.0f}/{:.0f}".format(h["hp"], h["max_hp"]) if h["hp"] and h["max_hp"] else "?"
        net_str = "0x{:08X}".format(h["net_id"]) if h["net_id"] else "?"
        print(f"  {h['name'] or '???':15s} team={h['team']} netId={net_str} "
              f"pos={pos_str} hp={hp_str} dead={h['dead']}")

    # ================================================================
    # STEP 4: Monitor positions over time
    # ================================================================
    if hero_objs:
        print("\n--- STEP 4: Position monitoring (5 samples, 1s apart) ---")
        for t in range(5):
            time.sleep(1)
            line = f"  t+{t+1}s: "
            for ptr, name in sorted(hero_objs.items(), key=lambda x: x[1]):
                pos = m.vec3(ptr + 0x25C)
                if pos:
                    line += f"{name}=({pos[0]:.0f},{pos[2]:.0f}) "
            print(line)

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
