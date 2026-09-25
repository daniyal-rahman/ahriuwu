"""
Find Garen's SpellBook by brute-force searching for "GarenQ" string.

The SpellBook contains spell slots for Q/W/E/R/D/F + basic attack.
Search hero struct 0x2800-0x6000 for pointer chains leading to "GarenQ".
Also search for "SummonerFlash", "SummonerDot" (ignite), etc.
"""

import ctypes
import ctypes.wintypes
import struct
import sys
import time
import urllib.request
import ssl
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
    def u32(s,a): d=s.read(a,4); return struct.unpack("<I",d)[0] if d else None
    def u64(s,a): d=s.read(a,8); return struct.unpack("<Q",d)[0] if d else None
    def f32(s,a): d=s.read(a,4); return struct.unpack("<f",d)[0] if d else None
    def string(s,a,n=128):
        d=s.read(a,n)
        if not d: return None
        return d.split(b'\x00')[0].decode('ascii',errors='replace') or None
    def close(s): kernel32.CloseHandle(s.h)

def is_heap(v): return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF

# Target spell names to search for
TARGETS = [b"GarenQ", b"GarenW", b"GarenE", b"GarenR",
           b"SummonerFlash", b"SummonerDot", b"SummonerHeal",
           b"SummonerTeleport", b"SummonerSmite", b"SummonerBarrier",
           b"GarenPassive", b"GarenBasicAttack"]

def check_string_at(m, addr, targets):
    """Check if the string at addr matches any target."""
    s = m.string(addr, 64)
    if not s or len(s) < 4:
        return None
    sb = s.encode('ascii', errors='replace')
    for t in targets:
        if sb.startswith(t):
            return s
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
    # SEARCH 1: Read hero struct in chunks, find all heap pointers,
    # follow 1-3 levels of deref, check for target spell names
    # ================================================================
    print("\n=== Search: Hero struct -> chains -> spell names ===")
    print("Searching hero+0x0000 to hero+0x6000...")

    hits = []
    checked = 0

    for hero_off in range(0x0, 0x6000, 8):
        p1 = m.u64(garen["ptr"] + hero_off)
        if not is_heap(p1): continue
        checked += 1

        # Level 1: p1 might directly point to a string
        for str_off in [0x0, 0x8, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38]:
            name_ptr = m.u64(p1 + str_off)
            if is_heap(name_ptr):
                s = check_string_at(m, name_ptr, TARGETS)
                if s:
                    hits.append((hero_off, f"+0x{str_off:02X}", s))

        # Level 2: p1 -> p2 -> string
        for sub1 in [0x0, 0x8, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38, 0x40, 0x48]:
            p2 = m.u64(p1 + sub1)
            if not is_heap(p2): continue

            for str_off in [0x0, 0x8, 0x10, 0x18, 0x20, 0x28, 0x30, 0x38]:
                name_ptr = m.u64(p2 + str_off)
                if is_heap(name_ptr):
                    s = check_string_at(m, name_ptr, TARGETS)
                    if s:
                        hits.append((hero_off, f"+0x{sub1:02X}->+0x{str_off:02X}", s))

            # Level 3: p1 -> p2 -> p3 -> string (but only check +0x28 at end)
            for sub2 in [0x0, 0x8, 0x10, 0x18, 0x20, 0x28]:
                p3 = m.u64(p2 + sub2)
                if not is_heap(p3): continue
                name_ptr = m.u64(p3 + 0x28)
                if is_heap(name_ptr):
                    s = check_string_at(m, name_ptr, TARGETS)
                    if s:
                        hits.append((hero_off, f"+0x{sub1:02X}->+0x{sub2:02X}->+0x28", s))

        if hero_off % 0x1000 == 0:
            print(f"  ...0x{hero_off:04X} ({checked} ptrs checked, {len(hits)} hits)")

    # Deduplicate and display
    print(f"\n  Total: {len(hits)} hits from {checked} pointer checks")

    # Group by spell name
    by_spell = {}
    for off, chain, name in hits:
        if name not in by_spell:
            by_spell[name] = []
        by_spell[name].append((off, chain))

    for spell, locs in sorted(by_spell.items()):
        print(f"\n  '{spell}' found at {len(locs)} locations:")
        # Show unique hero offsets
        unique_offsets = sorted(set(off for off, _ in locs))
        for off in unique_offsets[:10]:
            chains = [c for o, c in locs if o == off]
            print(f"    hero+0x{off:04X}: {chains[0]}")

    # ================================================================
    # Cross-validate: check if same offsets have different spell names
    # on other heroes (confirms these are SpellBook slots)
    # ================================================================
    if by_spell:
        print(f"\n\n=== Cross-validation on other heroes ===")
        # Get all unique hero offsets that had spell names
        all_offsets = sorted(set(off for off, _, _ in hits))

        # For each offset, check what spell name we get on other heroes
        for hero_off in all_offsets[:15]:
            chains_for_off = [(c, n) for o, c, n in hits if o == hero_off]
            chain = chains_for_off[0][0]  # Use first chain found

            print(f"\n  hero+0x{hero_off:04X} (chain {chain}):")
            for h in heroes[:6]:
                p1 = m.u64(h["ptr"] + hero_off)
                if not is_heap(p1):
                    print(f"    {h['name']:12s}: NULL")
                    continue

                # Follow the same chain
                parts = chain.split("->")
                current = p1
                for part in parts[:-1]:
                    off_val = int(part.replace("+0x", ""), 16)
                    current = m.u64(current + off_val)
                    if not is_heap(current):
                        current = None
                        break
                if current:
                    last_off = int(parts[-1].replace("+0x", ""), 16)
                    name_ptr = m.u64(current + last_off)
                    if is_heap(name_ptr):
                        s = m.string(name_ptr, 64)
                        print(f"    {h['name']:12s}: '{s}'")
                    else:
                        print(f"    {h['name']:12s}: (no string)")
                else:
                    print(f"    {h['name']:12s}: (chain broken)")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
