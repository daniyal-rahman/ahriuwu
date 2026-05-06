"""
Find SpellBook using deeper chain + inline struct approach.

hacker_logs structure:
  hero + oObjSpellBook (0x30E8) -> SpellBook (inline struct on hero)
  SpellBook + oSpellSlotArray (0xAE0) -> array of SpellSlot*
  SpellSlot -> SpellInfo -> SpellData -> name string

Our build shifts offsets. Try:
1. For every 8-byte aligned offset in hero struct, treat it as SpellSlot pointer
2. Follow 4 levels deep to find spell names like "GarenQ"
3. Look for clusters of consecutive slots with valid spell names = SpellBook

Also try treating regions of the hero struct as INLINE SpellBook and
reading the slot array within it.
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
    _fields_ = [("dwSize",wintypes.DWORD),("cntUsage",wintypes.DWORD),("th32ProcessID",wintypes.DWORD),("th32DefaultHeapID",ctypes.POINTER(ctypes.c_ulong)),("th32ModuleID",wintypes.DWORD),("cntThreads",wintypes.DWORD),("th32ParentProcessID",wintypes.DWORD),("pcPriClassBase",ctypes.c_long),("dwFlags",wintypes.DWORD),("szExeFile",ctypes.c_char*MAX_PATH)]
class MODULEENTRY32(ctypes.Structure):
    _fields_ = [("dwSize",wintypes.DWORD),("th32ModuleID",wintypes.DWORD),("th32ProcessID",wintypes.DWORD),("GlblcntUsage",wintypes.DWORD),("ProccntUsage",wintypes.DWORD),("modBaseAddr",ctypes.POINTER(ctypes.c_byte)),("modBaseSize",wintypes.DWORD),("hModule",wintypes.HMODULE),("szModule",ctypes.c_char*(MAX_MODULE_NAME32+1)),("szExePath",ctypes.c_char*MAX_PATH)]

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

def find_league():
    s=kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS,0);pe=PROCESSENTRY32();pe.dwSize=ctypes.sizeof(PROCESSENTRY32);kernel32.Process32First(s,ctypes.byref(pe))
    while True:
        if b"League of Legends" in pe.szExeFile: pid=pe.th32ProcessID;kernel32.CloseHandle(s);return pid
        if not kernel32.Process32Next(s,ctypes.byref(pe)): break
    kernel32.CloseHandle(s);return None
def find_base(pid):
    s=kernel32.CreateToolhelp32Snapshot(TH32CS_SNAPMODULE|TH32CS_SNAPMODULE32,pid);me=MODULEENTRY32();me.dwSize=ctypes.sizeof(MODULEENTRY32);kernel32.Module32First(s,ctypes.byref(me))
    while True:
        if b"league of legends" in me.szModule.lower(): b=ctypes.cast(me.modBaseAddr,ctypes.c_void_p).value;sz=me.modBaseSize;kernel32.CloseHandle(s);return b,sz
        if not kernel32.Module32Next(s,ctypes.byref(me)): break
    kernel32.CloseHandle(s);return None,None

class Mem:
    def __init__(s,pid): s.h=kernel32.OpenProcess(PROCESS_VM_READ|PROCESS_QUERY_INFORMATION,False,pid)
    def read(s,a,sz):
        buf=ctypes.create_string_buffer(sz);n=ctypes.c_size_t(0)
        ok=kernel32.ReadProcessMemory(s.h,ctypes.c_void_p(a),buf,sz,ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value==sz else None
    def u64(s,a): d=s.read(a,8); return struct.unpack("<Q",d)[0] if d else None
    def string(s,a,n=128):
        d=s.read(a,n)
        if not d: return None
        return d.split(b'\x00')[0].decode('ascii',errors='replace') or None
    def close(s): kernel32.CloseHandle(s.h)

def is_heap(v): return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF

def try_spell_name_deep(m, slot_ptr, max_depth=4):
    """Try many chain patterns to find a spell name from a slot pointer."""
    if not is_heap(slot_ptr):
        return None

    # Read the slot object
    slot_data = m.read(slot_ptr, 0x60)
    if not slot_data:
        return None

    # Try every pointer in slot as SpellInfo, then every pointer in SpellInfo as name
    for off1 in range(0, 0x58, 8):
        p1 = struct.unpack("<Q", slot_data[off1:off1+8])[0]
        if not is_heap(p1): continue

        # Check if p1 points to string directly
        s = m.string(p1, 64)
        if s and len(s) > 3 and s[0].isalpha() and not s.startswith("H\x89"):
            return (f"slot+0x{off1:02X}->str", s)

        # p1 -> SpellInfo, check sub-offsets for name
        si_data = m.read(p1, 0x50)
        if not si_data: continue

        for off2 in range(0, 0x48, 8):
            p2 = struct.unpack("<Q", si_data[off2:off2+8])[0]
            if not is_heap(p2): continue

            s2 = m.string(p2, 64)
            if s2 and len(s2) > 3 and s2[0].isalpha() and not s2.startswith("H\x89"):
                return (f"slot+0x{off1:02X}->+0x{off2:02X}->str", s2)

            # One more level: p2 -> SpellData -> name
            if max_depth >= 4:
                sd_data = m.read(p2, 0x40)
                if not sd_data: continue
                for off3 in [0x0, 0x8, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38]:
                    p3 = struct.unpack("<Q", sd_data[off3:off3+8])[0]
                    if is_heap(p3):
                        s3 = m.string(p3, 64)
                        if s3 and len(s3) > 3 and s3[0].isalpha() and not s3.startswith("H\x89"):
                            return (f"slot+0x{off1:02X}->+0x{off2:02X}->+0x{off3:02X}->str", s3)

    return None


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    pid = find_league(); base, _ = find_base(pid); m = Mem(pid)
    print(f"PID={pid} Base=0x{base:X}")

    arr_ptr = m.u64(base + HERO_ARRAY_RVA)
    heroes = []
    for i in range(10):
        hp = m.u64(arr_ptr + i*8)
        if not is_heap(hp): continue
        name = m.string(hp + 0x4328, 32) or f"h{i}"
        heroes.append({"ptr": hp, "name": name})

    garen = next((h for h in heroes if "Garen" in h['name']), None)
    if not garen:
        print("Garen not found!"); return
    print(f"Garen at 0x{garen['ptr']:X}")

    # ================================================================
    # APPROACH 1: Treat every heap pointer in hero struct as a possible
    # SpellSlot and try deep chain resolution.
    # Scan 0x2800-0x6000 (wider than before, deeper chains)
    # ================================================================
    print("\n=== Approach 1: Deep chain spell name search (4 levels) ===")
    print("Scanning hero+0x2800 to hero+0x6000...")

    slot_hits = []  # (hero_off, chain, name)

    for hero_off in range(0x2800, 0x6000, 8):
        ptr = m.u64(garen["ptr"] + hero_off)
        if not is_heap(ptr): continue

        result = try_spell_name_deep(m, ptr)
        if result:
            chain, name = result
            # Filter out known noise
            if name.startswith("H\x89") or name.startswith("H\x83"):
                continue
            slot_hits.append((hero_off, chain, name))

        if hero_off % 0x800 == 0:
            print(f"  ...0x{hero_off:04X} ({len(slot_hits)} hits)")

    print(f"\n  Total hits: {len(slot_hits)}")
    for off, chain, name in slot_hits:
        print(f"    hero+0x{off:04X} ({chain}): '{name[:50]}'")

    # ================================================================
    # APPROACH 2: Find inline SpellBook by looking for a pointer in
    # hero struct that, when followed, reveals a large object containing
    # an array of spell slot pointers at various sub-offsets.
    #
    # For each heap pointer P in hero struct:
    #   Read P+0x000 to P+0x1200 (big object)
    #   Scan for regions of 4+ consecutive heap pointers
    #   For each such region, try resolving spell names
    # ================================================================
    print("\n\n=== Approach 2: Large object spell slot array search ===")
    print("Looking for SpellBook-like objects with consecutive slot arrays...")

    for hero_off in range(0x2800, 0x5000, 8):
        ptr = m.u64(garen["ptr"] + hero_off)
        if not is_heap(ptr): continue

        # Read a big chunk
        big = m.read(ptr, 0x1200)
        if not big: continue

        # Scan for runs of 4+ consecutive heap pointers
        for arr_start in range(0, 0x1100, 8):
            consecutive = 0
            for i in range(0, 0x80, 8):  # Check up to 16 slots
                val = struct.unpack("<Q", big[arr_start+i:arr_start+i+8])[0]
                if is_heap(val):
                    consecutive += 1
                else:
                    break

            if consecutive >= 4:
                # Try to resolve spell names from these slots
                names = []
                for i in range(consecutive):
                    slot_ptr = struct.unpack("<Q", big[arr_start+i*8:arr_start+i*8+8])[0]
                    result = try_spell_name_deep(m, slot_ptr, max_depth=4)
                    if result:
                        names.append((i, result[0], result[1]))

                # Check if any are champion spells (not "H\x89" garbage)
                real_names = [(i, c, n) for i, c, n in names
                              if not n.startswith("H\x89") and not n.startswith("H\x83")]

                if len(real_names) >= 2:
                    print(f"\n  hero+0x{hero_off:04X} -> +0x{arr_start:04X}: "
                          f"{consecutive} ptrs, {len(real_names)} spell names")
                    for i, chain, name in real_names[:8]:
                        print(f"    slot[{i}] ({chain}): '{name[:50]}'")

    # ================================================================
    # APPROACH 3: Also look at hero struct beyond 0x5000
    # Some builds have the SpellBook at very high offsets
    # ================================================================
    print("\n\n=== Approach 3: Extended search hero+0x5000-0x8000 ===")

    for hero_off in range(0x5000, 0x8000, 8):
        ptr = m.u64(garen["ptr"] + hero_off)
        if not is_heap(ptr): continue

        result = try_spell_name_deep(m, ptr)
        if result:
            chain, name = result
            if name.startswith("H\x89") or name.startswith("H\x83"):
                continue
            # Only print if it looks like a real spell name
            if any(c.isupper() for c in name[:5]) and len(name) > 4:
                print(f"    hero+0x{hero_off:04X} ({chain}): '{name[:50]}'")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
