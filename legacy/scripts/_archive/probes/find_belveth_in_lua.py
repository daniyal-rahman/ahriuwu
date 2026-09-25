"""Scan for 'Belveth' interned as a Lua string and find what references it.

Lua interns strings — each distinct string exists once in the string pool.
If "Belveth" is a Lua TString, it'll appear as a null-terminated ASCII byte
sequence in the heap. References to it from Lua tables are u64 pointers to
the TString's data (or a small fixed offset before it, depending on the
Lua version — typically 24 bytes before payload for Lua 5.1/5.2/5.3, or
the TString start).

Writes C:\\tmp\\belveth_lua_refs.json with findings.
"""
import ctypes, ctypes.wintypes as wt
import sys, subprocess, time, json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

_k = ctypes.windll.kernel32
class MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", ctypes.c_ulong), ("__a", ctypes.c_ulong),
                ("RegionSize", ctypes.c_size_t), ("State", ctypes.c_ulong),
                ("Protect", ctypes.c_ulong), ("Type", ctypes.c_ulong),
                ("__b", ctypes.c_ulong)]
_k.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
_k.VirtualQueryEx.restype = ctypes.c_size_t
MEM_COMMIT = 0x1000
PAGE_READABLE = 0x02 | 0x04 | 0x08 | 0x10 | 0x20 | 0x40 | 0x80

NEEDLES = [
    b"Belveth\x00",         # Riot's internal champion id (no apostrophe)
    b"Bel'Veth\x00",        # display name with apostrophe
    b"belveth\x00",
    b"BelvethBasicAttack",  # a known spell name (definitely Lua-interned)
    b"BelvethQ",
    b"BelvethW",
    b"BelvethR",
    b"OnClick",
    b"OnMove",
    b"IssueOrder",
    b"MoveTo",
    b"ClientCommand",
    b"AiManager",
    b"GetHero",
    b"ObjectManager",
]

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])
    return None

def readable_regions(h):
    addr = 0; mbi = MBI()
    while addr < 0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
            break
        b = mbi.BaseAddress or 0; s = mbi.RegionSize
        if mbi.State == MEM_COMMIT and (mbi.Protect & PAGE_READABLE):
            yield b, s, mbi.Type
        addr = b + s
        if addr <= b: break

def read_region(h, base, size):
    if size > 512 * 1024 * 1024: return None
    out = bytearray(size); v = memoryview(out); o = 0; CH = 4 * 1024 * 1024
    while o < size:
        n = min(CH, size - o)
        buf = (ctypes.c_char * n)(); r = ctypes.c_size_t(0)
        if not _k.ReadProcessMemory(h, ctypes.c_void_p(base + o), buf, n, ctypes.byref(r)) or r.value == 0:
            return None if o == 0 else bytes(v[:o])
        v[o:o+r.value] = buf[:r.value]; o += r.value
    return bytes(out)

def find_all(needle, hay):
    i = 0
    while True:
        j = hay.find(needle, i)
        if j == -1: break
        yield j
        i = j + 1

def main():
    pid = find_pid()
    if not pid:
        print("ERR: League not running"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    print(f"pid={pid}")

    # Load all regions once
    print("Loading regions...")
    t0 = time.time()
    regions = []
    for base, size, typ in readable_regions(h):
        data = read_region(h, base, size)
        if data: regions.append((base, size, typ, data))
    print(f"  {len(regions)} regions in {time.time()-t0:.1f}s")

    # Phase 1: string search
    print("\n== String presence ==")
    string_addrs = {}
    for needle in NEEDLES:
        addrs = []
        for base, _, _, data in regions:
            for off in find_all(needle, data):
                addrs.append(base + off)
        string_addrs[needle] = addrs
        name = needle.rstrip(b"\x00").decode(errors="replace")
        sample = [f"0x{a:X}" for a in addrs[:3]]
        print(f"  {name:<22}  {len(addrs):>4} hits  {sample}")

    # Phase 2: find references to Belveth strings.
    # Lua TString layout (5.1/5.3):
    #   offset 0:   next GCObject* (8)
    #   offset 8:   tt (byte) = 4 (LUA_TSTRING)
    #   offset 9:   marked (byte)
    #   offset A:   reserved (byte)
    #   offset B:   (pad)
    #   offset C:   hash (4)
    #   offset 10:  len (size_t, 8)
    #   offset 18:  data (chars)
    # So a TString* pointer points 0x18 bytes BEFORE the actual char data.
    # A reference from a Lua table (TValue.value.gc) stores the TString*, not
    # the char pointer. So pointers to Belveth TStrings target (addr_of_chars - 0x18).
    # We search for pointers into [addr_of_chars - 0x20, addr_of_chars].
    primary = []
    for needle in (b"Belveth\x00", b"Bel'Veth\x00"):
        primary.extend(string_addrs.get(needle, []))
    if not primary:
        print("\nNo 'Belveth' or 'Bel'Veth' string found — name interned differently?")
        return 1

    print(f"\n== References to 'Belveth' strings ({len(primary)} locations) ==")
    # Ranges to look for pointers: every primary address's surrounding 0x20 bytes
    target_ranges = set()
    for a in primary:
        for off in range(-0x30, 0x4, 4):
            target_ranges.add(a + off)
    lo = min(target_ranges); hi = max(target_ranges) + 1

    ptr_hits = []
    for base, size, typ, data in regions:
        usable = len(data) & ~7
        if usable < 8: continue
        for align in (0, 4):
            if align + 8 > usable: continue
            use = (usable - align) & ~7
            arr = np.frombuffer(data, dtype=np.uint64, count=use // 8, offset=align)
            mask = (arr >= lo) & (arr < hi)
            if not mask.any(): continue
            for i in np.nonzero(mask)[0]:
                v = int(arr[i])
                if v in target_ranges:
                    src = base + align + int(i) * 8
                    ptr_hits.append((src, v))

    # Dedupe
    ptr_hits = list(set(ptr_hits))
    print(f"  {len(ptr_hits)} unique references to Belveth-strings (header-relative)")
    for src, tgt in ptr_hits[:30]:
        # find which primary string this targets
        off = 0; which = 0
        for a in primary:
            if a - 0x30 <= tgt <= a:
                off = tgt - a; which = a; break
        if which:
            print(f"    0x{src:X} -> 0x{tgt:X}  (to Belveth@0x{which:X}{off:+d})")
        else:
            print(f"    0x{src:X} -> 0x{tgt:X}")

    out = {
        "pid": pid,
        "hit_counts": {k.rstrip(b"\x00").decode(errors="replace"): len(v)
                       for k, v in string_addrs.items()},
        "belveth_string_addrs": [hex(a) for a in primary],
        "belveth_string_refs": [{"src": hex(s), "tgt": hex(t)} for s, t in ptr_hits],
    }
    with open(r"C:\tmp\belveth_lua_refs.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote C:\\tmp\\belveth_lua_refs.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
