"""
Find SpellBook by searching for INLINE spell name strings (SSO).

League uses Riot's string type with Small String Optimization:
- Strings <= 15 chars stored INLINE (no pointer!)
- Strings > 15 chars stored as heap pointer

"GarenQ" = 6 chars -> INLINE
"SummonerFlash" = 13 chars -> INLINE
"GarenBasicAttack" = 16 chars -> POINTER (why we found it before)

Search strategy: Read hero struct objects and scan raw bytes for inline
spell name strings like "GarenQ", "GarenW", "SummonerFlash", etc.
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
    def string(s,a,n=64):
        d=s.read(a,n)
        if not d: return None
        return d.split(b'\x00')[0].decode('ascii',errors='replace') or None
    def close(s): kernel32.CloseHandle(s.h)

def is_heap(v): return v is not None and 0x100000000 < v < 0x7FFFFFFFFFFF

# Inline spell names to search for (all <= 15 chars)
INLINE_TARGETS = [
    b"GarenQ", b"GarenW", b"GarenE", b"GarenR",
    b"GarenPassive",    # 12 chars
    b"SummonerFlash",   # 13 chars
    b"SummonerDot",     # 11 chars
    b"SummonerHeal",    # 12 chars
    b"SummonerSmite",   # 13 chars
    b"SummonerHaste",   # 13 chars
    b"S5_SummonerSmiteDuel",  # 20 chars -> pointer (skip)
]
# Only keep <= 15 chars
INLINE_TARGETS = [t for t in INLINE_TARGETS if len(t) <= 15]

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
    print(f"Searching for inline strings: {[t.decode() for t in INLINE_TARGETS]}")

    # ================================================================
    # SEARCH 1: Scan hero struct raw bytes for inline spell names
    # Read entire hero struct in chunks and search for byte patterns
    # ================================================================
    print("\n=== Search 1: Inline strings in hero struct (direct) ===")
    print("Reading hero+0x0000 to hero+0x8000...")

    for chunk_start in range(0, 0x8000, 0x1000):
        data = m.read(garen["ptr"] + chunk_start, 0x1000)
        if not data: continue

        for target in INLINE_TARGETS:
            idx = 0
            while True:
                idx = data.find(target, idx)
                if idx < 0: break
                hero_off = chunk_start + idx
                # Read surrounding context
                ctx_start = max(0, idx - 8)
                ctx_end = min(len(data), idx + 32)
                ctx = data[ctx_start:ctx_end]
                # Check if it's a proper null-terminated string
                end = data.find(b'\x00', idx)
                if end > idx:
                    full_str = data[idx:end].decode('ascii', errors='replace')
                else:
                    full_str = target.decode()
                print(f"  hero+0x{hero_off:04X}: '{full_str}'")
                idx += 1

    # ================================================================
    # SEARCH 2: For each heap pointer in hero struct, read target object
    # and search for inline spell names
    # ================================================================
    print("\n=== Search 2: Inline strings in pointed-to objects ===")
    print("For each ptr in hero struct, read 0x200 bytes and search...")

    hits = []  # (hero_off, obj_internal_off, spell_name)

    for hero_off in range(0, 0x6000, 8):
        ptr = m.u64(garen["ptr"] + hero_off)
        if not is_heap(ptr): continue

        data = m.read(ptr, 0x200)
        if not data: continue

        for target in INLINE_TARGETS:
            idx = data.find(target)
            if idx >= 0:
                end = data.find(b'\x00', idx)
                full_str = data[idx:end].decode('ascii', errors='replace') if end > idx else target.decode()
                hits.append((hero_off, idx, full_str))
                print(f"  hero+0x{hero_off:04X} -> obj+0x{idx:03X}: '{full_str}'")

        if hero_off % 0x1000 == 0 and hero_off > 0:
            print(f"  ...checked up to hero+0x{hero_off:04X} ({len(hits)} hits)")

    # ================================================================
    # SEARCH 3: Two-level deref — for each ptr P in hero struct, read
    # each sub-ptr Q in P's object, then search Q's object for inline strings
    # ================================================================
    print(f"\n=== Search 3: Two-level deref inline string search ===")
    print(f"hero struct -> ptr P -> sub-ptr Q -> search for inline strings")

    deep_hits = []

    for hero_off in range(0x2800, 0x5000, 8):
        ptr = m.u64(garen["ptr"] + hero_off)
        if not is_heap(ptr): continue

        # Read P's object
        p_data = m.read(ptr, 0x100)
        if not p_data: continue

        for sub_off in range(0, 0xF8, 8):
            q = struct.unpack("<Q", p_data[sub_off:sub_off+8])[0]
            if not is_heap(q): continue

            q_data = m.read(q, 0x200)
            if not q_data: continue

            for target in INLINE_TARGETS:
                idx = q_data.find(target)
                if idx >= 0:
                    end = q_data.find(b'\x00', idx)
                    full_str = q_data[idx:end].decode('ascii', errors='replace') if end > idx else target.decode()
                    deep_hits.append((hero_off, sub_off, idx, full_str))
                    print(f"  hero+0x{hero_off:04X} -> +0x{sub_off:02X} -> obj+0x{idx:03X}: '{full_str}'")

        if hero_off % 0x800 == 0 and hero_off > 0x2800:
            sys.stdout.flush()

    # ================================================================
    # SEARCH 4: Three-level deref (hero -> P -> Q -> R -> inline search)
    # Only for promising regions (0x3800-0x4800)
    # ================================================================
    print(f"\n=== Search 4: Three-level deref (0x3800-0x4800 only) ===")

    for hero_off in range(0x3800, 0x4800, 8):
        ptr = m.u64(garen["ptr"] + hero_off)
        if not is_heap(ptr): continue

        p_data = m.read(ptr, 0x60)
        if not p_data: continue

        for sub1 in range(0, 0x58, 8):
            q = struct.unpack("<Q", p_data[sub1:sub1+8])[0]
            if not is_heap(q): continue

            q_data = m.read(q, 0x60)
            if not q_data: continue

            for sub2 in range(0, 0x58, 8):
                r = struct.unpack("<Q", q_data[sub2:sub2+8])[0]
                if not is_heap(r): continue

                r_data = m.read(r, 0x200)
                if not r_data: continue

                for target in INLINE_TARGETS:
                    idx = r_data.find(target)
                    if idx >= 0:
                        end = r_data.find(b'\x00', idx)
                        full_str = r_data[idx:end].decode('ascii', errors='replace') if end > idx else target.decode()
                        print(f"  hero+0x{hero_off:04X} -> +0x{sub1:02X} -> +0x{sub2:02X} -> obj+0x{idx:03X}: '{full_str}'")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n\n=== Summary ===")
    print(f"  Direct hero struct hits: see Search 1")
    print(f"  One-level deref hits: {len(hits)}")
    print(f"  Two-level deref hits: {len(deep_hits)}")

    m.close()
    print("\nDone!")

if __name__ == "__main__":
    main()
