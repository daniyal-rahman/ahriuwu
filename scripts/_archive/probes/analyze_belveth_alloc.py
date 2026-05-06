"""Deep-dive a confirmed Bel'Veth click-destination allocation.

Usage:
    python analyze_belveth_alloc.py --alloc 0x1F37A7793A4

Does four things:
  1. Dumps 0x500 bytes starting at alloc-0x40, annotating each u64 as
     (hex, 2 x f32, ascii, module-pointer?)
  2. Walks every 8/4-byte-aligned field and flags anything that's a pointer
     into another heap region, another module, or back into this alloc.
  3. Compares key field positions with UC-forum / tlol-scraper-pandoras
     struct layouts (AiManager, SpellInput, ActiveCastSpell) and calls out
     matches.
  4. Scans process-wide for pointers into [alloc, alloc+0x400) at depth-1
     so we can see which heap objects/struct reference it.
"""
import ctypes, ctypes.wintypes as wt
import argparse, struct, subprocess, sys, time, json
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
MEM_COMMIT = 0x1000; MEM_PRIVATE = 0x20000
PAGE_READABLE = 0x02 | 0x04 | 0x08 | 0x10 | 0x20 | 0x40 | 0x80

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

def read_bytes(h, addr, n):
    buf = (ctypes.c_char * n)()
    r = ctypes.c_size_t(0)
    if not _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, n, ctypes.byref(r)): return None
    return bytes(buf[:r.value])

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

def classify_ptr(val, mod_base, mod_size, alloc_base, alloc_end):
    if mod_base is not None and mod_base <= val < mod_base + mod_size:
        return f"MODULE+0x{val - mod_base:X}"
    if alloc_base <= val < alloc_end:
        return f"SELF+0x{val - alloc_base:X}"
    if 0x10000000000 < val < 0x80000000000:
        return "HEAP"
    return ""

def annotate_dump(h, alloc_base, alloc_size, mod_base, mod_size):
    before = 0x40
    total = before + alloc_size
    data = read_bytes(h, alloc_base - before, total)
    if not data:
        print("Read failed"); return
    alloc_end = alloc_base + alloc_size
    print(f"\n== Annotated dump of alloc 0x{alloc_base:X} (+0x{alloc_size:X}) ==")
    print(f"   module=[0x{mod_base:X},0x{mod_base+mod_size:X})")
    print(f"   off    hex              f32a    f32b       ascii/pointer")
    for i in range(0, len(data) - 7, 8):
        off = (alloc_base - before) + i - alloc_base
        u64 = struct.unpack_from("<Q", data, i)[0]
        fa = struct.unpack_from("<f", data, i)[0]
        fb = struct.unpack_from("<f", data, i + 4)[0]
        cls = classify_ptr(u64, mod_base, mod_size, alloc_base, alloc_end)
        ascii_repr = ""
        raw = data[i:i+8]
        if all(32 <= b <= 126 or b in (0, 9, 10) for b in raw):
            ascii_repr = "".join(chr(b) if 32 <= b <= 126 else "." for b in raw)
            if ascii_repr.strip("\x00. "):
                ascii_repr = f"'{ascii_repr.rstrip(chr(0))}'"
            else:
                ascii_repr = ""
        tag = ""
        if alloc_base + 0x000 == alloc_base + off: tag = "<-- mirror A"
        elif off == 0x308: tag = "<-- mirror B"
        elif off == 0x374: tag = "<-- mirror C"
        fmt_float = lambda f: f"{f:>10.2f}" if abs(f) < 1e7 and abs(f) > 0.01 or f == 0 else f"{f:>10.2e}"
        line = f"  [+{off:>+5d}]  {u64:016X}  {fmt_float(fa)} {fmt_float(fb)}  {cls:<18} {ascii_repr}{tag}"
        print(line)

def reverse_ptr_scan(h, alloc_base, alloc_size):
    """Find every u64 in the process pointing into [alloc_base, alloc_end)."""
    lo = alloc_base; hi = alloc_base + alloc_size
    hits = []
    for base, size, _ in readable_regions(h):
        data = read_region(h, base, size)
        if not data: continue
        arr = np.frombuffer(data[:len(data) & ~7], dtype=np.uint64)
        mask = (arr >= lo) & (arr < hi)
        if not mask.any(): continue
        for i in np.nonzero(mask)[0]:
            hits.append((base + int(i) * 8, int(arr[i])))
    return hits

# Known field offsets from tlol-scraper-pandoras / UC forum traditions.
# These are LIVE-mode AiManager-style offsets; our alloc likely has different
# layout (this is a click-callback buffer, not AiManager itself). We check
# anyway in case some fields happen to line up.
KNOWN_AIMANAGER = {
    0x10: "ServerPos (Vec3)",
    0x40: "Waypoint count",
    0x48: "Waypoint ptr",
    0x1C: "IsMoving",
    0x24: "MovementSpeed",
    0x474: "ServerPos alt",
}

def compare_known_offsets(h, alloc_base):
    data = read_bytes(h, alloc_base, 0x500)
    if not data: return
    print("\n== Field-offset alignment with known struct conventions ==")
    for off, name in KNOWN_AIMANAGER.items():
        if off + 8 > len(data): continue
        u64 = struct.unpack_from("<Q", data, off)[0]
        fa = struct.unpack_from("<f", data, off)[0]
        fb = struct.unpack_from("<f", data, off + 4)[0]
        note = ""
        if abs(fa) < 20000 and abs(fa) > 1 and abs(fb) < 20000:
            note = f"(looks like coords Vec3)"
        print(f"  alloc+0x{off:03X}  {u64:016X}  f=({fa:.2f},{fb:.2f}) — {name} {note}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alloc", required=True, help="hex address e.g. 0x1F37A7793A4")
    ap.add_argument("--size", default="0x400")
    args = ap.parse_args()

    alloc = int(args.alloc, 16)
    alloc_size = int(args.size, 16)
    pid = find_pid()
    if not pid:
        print("ERR: League not running"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    mod_base, mod_size = module_range(pid)
    print(f"pid={pid}  module=0x{mod_base:X}  alloc=0x{alloc:X}")

    # Sanity: read current Vec3 so we know alloc is still valid
    data12 = read_bytes(h, alloc, 12)
    if not data12:
        print("ERR: could not read alloc"); return 1
    vec = struct.unpack("<fff", data12)
    print(f"Current Vec3 @ alloc: ({vec[0]:.2f}, {vec[1]:.2f}, {vec[2]:.2f})")

    annotate_dump(h, alloc, alloc_size, mod_base, mod_size)
    compare_known_offsets(h, alloc)

    print("\n== Reverse pointer scan (who points into this alloc?) ==")
    t0 = time.time()
    hits = reverse_ptr_scan(h, alloc, alloc_size)
    print(f"Scan took {time.time() - t0:.1f}s — {len(hits)} pointer hits")
    # Classify sources
    ext = []; self_ = []; module = []
    for src, tgt in hits:
        if alloc <= src < alloc + alloc_size:
            self_.append((src, tgt))
        elif mod_base is not None and mod_base <= src < mod_base + mod_size:
            module.append((src, tgt))
        else:
            ext.append((src, tgt))
    print(f"  self (internal): {len(self_)}")
    print(f"  module-static:   {len(module)}")
    print(f"  external heap:   {len(ext)}")
    for src, tgt in module[:10]:
        print(f"    MODULE+0x{src-mod_base:X} -> alloc+0x{tgt-alloc:X}")
    for src, tgt in ext[:20]:
        print(f"    0x{src:X} -> alloc+0x{tgt-alloc:X}")

    out = {
        "alloc": hex(alloc), "size": alloc_size,
        "module_base": hex(mod_base), "module_size": mod_size,
        "reverse_ptr_hits": {
            "self": [[hex(s), hex(t)] for s, t in self_],
            "module": [[hex(s), hex(t)] for s, t in module],
            "external": [[hex(s), hex(t)] for s, t in ext],
        },
    }
    with open(r"C:\tmp\belveth_alloc_analysis.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote C:\\tmp\\belveth_alloc_analysis.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
