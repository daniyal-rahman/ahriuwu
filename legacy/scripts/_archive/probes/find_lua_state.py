"""Hunt for Lua state inside League of Legends' memory.

Phase 1: Scan for well-known Lua metatable method name strings
  (__index, __newindex, __gc, __call, __tostring, __metatable, __add, __sub,
   __mul, __div, __mod, __pow, __unm, __concat, __len, __eq, __lt, __le, __mode)
  If League embeds Lua (stock or close variant), these appear many times —
  mostly in the string intern pool. Near-zero hits means LuaJIT or a custom
  fork.

Phase 2: For each string found, scan for u64 pointers pointing at it. Those
  are references from Lua strings/tables/metatables. Clustering of these
  pointers into a single small region suggests the Lua string table.

Phase 3: Look for lua_State signatures. Stock Lua 5.x has a `global_State`
  struct with predictable fields (strt pointer, total bytes, GC config).
  Print candidate addresses.

Output: C:\\tmp\\lua_state_hunt.json
"""
import ctypes, ctypes.wintypes as wt
import sys, subprocess, time, json, struct
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

LUA_METAMETHODS = [
    b"__index\x00", b"__newindex\x00", b"__gc\x00", b"__call\x00",
    b"__tostring\x00", b"__metatable\x00", b"__mode\x00",
    b"__add\x00", b"__sub\x00", b"__mul\x00", b"__div\x00",
    b"__mod\x00", b"__pow\x00", b"__unm\x00", b"__concat\x00",
    b"__len\x00", b"__eq\x00", b"__lt\x00", b"__le\x00",
]

# Distinctive Lua API strings likely present in any Lua-using program
LUA_HALLMARKS = [
    b"attempt to index", b"stack overflow", b"_LOADED", b"_PRELOAD",
    b"LuaJIT", b"Lua 5.", b"PUC-Rio",
]

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])
    return None

def module_range(pid):
    psapi = ctypes.WinDLL("psapi.dll")
    h = _k.OpenProcess(0x0410, False, pid)
    HMODULE = wt.HMODULE
    psapi.EnumProcessModulesEx.argtypes = [wt.HANDLE, ctypes.POINTER(HMODULE), wt.DWORD, ctypes.POINTER(wt.DWORD), wt.DWORD]
    psapi.GetModuleFileNameExW.argtypes = [wt.HANDLE, HMODULE, wt.LPWSTR, wt.DWORD]
    class MINFO(ctypes.Structure):
        _fields_ = [("lpBaseOfDll", ctypes.c_void_p), ("SizeOfImage", wt.DWORD), ("EntryPoint", ctypes.c_void_p)]
    psapi.GetModuleInformation.argtypes = [wt.HANDLE, HMODULE, ctypes.POINTER(MINFO), wt.DWORD]
    mods = (HMODULE * 1024)(); needed = wt.DWORD(0)
    psapi.EnumProcessModulesEx(h, mods, ctypes.sizeof(mods), ctypes.byref(needed), 3)
    n = needed.value // ctypes.sizeof(HMODULE)
    for i in range(n):
        name = ctypes.create_unicode_buffer(260)
        psapi.GetModuleFileNameExW(h, mods[i], name, 260)
        if name.value.lower().endswith("league of legends.exe"):
            mi = MINFO()
            psapi.GetModuleInformation(h, mods[i], ctypes.byref(mi), ctypes.sizeof(mi))
            return mi.lpBaseOfDll, mi.SizeOfImage
    return None, None

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
    """Yield all byte offsets where needle appears in hay."""
    i = 0; L = len(needle)
    while True:
        j = hay.find(needle, i)
        if j == -1: break
        yield j
        i = j + 1

def classify(addr, mod_base, mod_size):
    if mod_base and mod_base <= addr < mod_base + mod_size:
        return "MODULE"
    if 0x10000000000 < addr < 0x80000000000:
        return "HEAP"
    return "?"

def main():
    pid = find_pid()
    if not pid:
        print("ERR: League not running"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    mod_base, mod_size = module_range(pid)
    print(f"pid={pid}  module=0x{mod_base:X}  size=0x{mod_size:X}")

    # Phase 1: scan for metamethod strings across ALL readable memory
    print("\n== Phase 1: hunt for Lua metamethod strings ==")
    hits = {needle: [] for needle in LUA_METAMETHODS + LUA_HALLMARKS}
    regions_data = []
    t0 = time.time()
    for base, size, typ in readable_regions(h):
        data = read_region(h, base, size)
        if not data: continue
        regions_data.append((base, size, typ, data))
        for needle in hits:
            for off in find_all(needle, data):
                hits[needle].append(base + off)
    print(f"scanned {len(regions_data)} regions in {time.time()-t0:.1f}s")

    for needle, addrs in hits.items():
        if not addrs: continue
        mod_n = sum(1 for a in addrs if classify(a, mod_base, mod_size) == "MODULE")
        heap_n = len(addrs) - mod_n
        # show first 3 addresses of each
        sample = [f"0x{a:X}({classify(a, mod_base, mod_size)[0]})" for a in addrs[:3]]
        name = needle.rstrip(b"\x00").decode(errors="replace")
        print(f"  {name:<16}  {len(addrs):>5} hits  (mod={mod_n} heap={heap_n})  {sample}")

    # Phase 2: identify the string table region — the heap region with the
    # most metamethod-string hits is likely where the Lua string pool lives.
    # Group hits by containing region.
    print("\n== Phase 2: densest region holding metamethod strings ==")
    region_scores = {}
    for needle, addrs in hits.items():
        if not needle.startswith(b"__"): continue
        for a in addrs:
            for base, size, typ, _ in regions_data:
                if base <= a < base + size:
                    key = (base, typ)
                    region_scores[key] = region_scores.get(key, 0) + 1
                    break
    top = sorted(region_scores.items(), key=lambda kv: -kv[1])[:5]
    for (base, typ), n in top:
        tag = "MODULE" if typ == 0x1000000 else ("HEAP" if typ == 0x20000 else "MAP")
        print(f"  region 0x{base:X}  ({tag})  holds {n} metamethod strings")

    # Phase 3: find addresses that reference __index (well-known anchor) —
    # these are likely Lua TString pointers or metatable entries.
    print("\n== Phase 3: who references '__index' ? ==")
    index_addrs = hits.get(b"__index\x00", [])
    if not index_addrs:
        print("  no __index found — League does NOT embed stock Lua")
    else:
        # Scan for pointers pointing at any __index occurrence.
        tset = set(index_addrs)
        # Also consider that Lua TString stores the string 16-24 bytes after the
        # TString header (depending on version). So scan for any pointer
        # pointing within [addr - 32, addr + 0] for each.
        ptr_targets = set()
        for a in index_addrs:
            for off in range(-32, 0, 4):
                ptr_targets.add(a + off)
        lo, hi = min(ptr_targets), max(ptr_targets) + 1
        ptr_hits = []
        for base, size, typ, data in regions_data:
            arr = np.frombuffer(data[:len(data) & ~7], dtype=np.uint64)
            mask = (arr >= lo) & (arr < hi)
            if not mask.any(): continue
            for i in np.nonzero(mask)[0]:
                v = int(arr[i])
                if v in ptr_targets or v in tset:
                    ptr_hits.append((base + int(i) * 8, v))
        print(f"  {len(ptr_hits)} references to __index (header-relative) found")
        # Group by source region type
        src_mod = sum(1 for s, _ in ptr_hits if classify(s, mod_base, mod_size) == "MODULE")
        print(f"    in module: {src_mod}   in heap: {len(ptr_hits) - src_mod}")
        for s, t in ptr_hits[:15]:
            cls = classify(s, mod_base, mod_size)
            tag_m = ""
            if mod_base <= s < mod_base + mod_size:
                tag_m = f" (module+0x{s-mod_base:X})"
            print(f"    0x{s:X} -> 0x{t:X}  [{cls}]{tag_m}")

    out = {
        "pid": pid,
        "module_base": hex(mod_base),
        "hit_counts": {k.rstrip(b"\x00").decode(): len(v) for k, v in hits.items()},
        "top_regions": [{"base": hex(b), "type": t, "count": n}
                         for (b, t), n in top],
    }
    with open(r"C:\tmp\lua_state_hunt.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote C:\\tmp\\lua_state_hunt.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
