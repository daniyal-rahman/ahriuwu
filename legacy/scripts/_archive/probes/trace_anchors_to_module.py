"""Given a set of 'level-1' anchor addresses that point into the Bel'Veth
click-dest allocation, reverse-walk through pointer levels looking for a
chain that reaches a module-static pointer in League.exe.

Each level-k frontier is the set of addresses that point to some address in
the previous level. Stops when any module-static is found or depth cap is
reached. Records the shortest chain per winner.
"""
import ctypes, ctypes.wintypes as wt
import argparse, sys, time, json, subprocess
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
            yield b, s
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

def pointer_scan(all_regions_data, targets):
    """Given preloaded region data, find pointers whose value is in targets
    (exact match at 8-byte and 4-byte alignments)."""
    if not targets: return []
    tset = set(targets)
    lo = min(tset); hi = max(tset) + 1
    hits = []
    for base, size, data in all_regions_data:
        if not data: continue
        for align_off in (0, 4):
            start = (base + (align_off + 7 - ((base + align_off) & 7)) & 7)
            off = align_off
            if off + 8 > len(data): continue
            usable = (len(data) - off) & ~7
            if usable < 8: continue
            arr = np.frombuffer(data, dtype=np.uint64, count=usable // 8, offset=off)
            mask = (arr >= lo) & (arr < hi)
            if not mask.any(): continue
            for i in np.nonzero(mask)[0]:
                v = int(arr[i])
                if v in tset:
                    src = base + off + int(i) * 8
                    hits.append((src, v))
    return list({(s, t): (s, t) for s, t in hits}.values())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchors", nargs="+", required=True,
                    help="hex addresses to trace back from")
    ap.add_argument("--depth", type=int, default=5)
    args = ap.parse_args()

    targets = [int(a, 16) for a in args.anchors]
    pid = find_pid()
    if not pid:
        print("ERR: no league"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    mod_base, mod_size = module_range(pid)
    print(f"pid={pid} module=0x{mod_base:X} anchors={[hex(t) for t in targets]}")

    # Preload all readable regions once (costly — one time)
    print("Loading regions...")
    t0 = time.time()
    regions = []
    total_bytes = 0
    for base, size in readable_regions(h):
        data = read_region(h, base, size)
        if not data: continue
        regions.append((base, size, data))
        total_bytes += len(data)
    print(f"  {len(regions)} regions, {total_bytes/(1024*1024):.0f}MB in {time.time()-t0:.1f}s")

    # Parent chains: parent[addr] = (target_pointed_to, depth)
    parent = {}
    for t in targets: parent[t] = (None, 0)
    frontier = list(targets)
    winners = []

    for depth in range(1, args.depth + 1):
        if not frontier: break
        print(f"\ndepth={depth}  frontier_size={len(frontier)}")
        hits = pointer_scan(regions, frontier)
        print(f"  {len(hits)} pointer hits")
        next_frontier = []
        mods = 0
        for src, tgt in hits:
            if src in parent: continue
            parent[src] = (tgt, depth)
            if mod_base <= src < mod_base + mod_size:
                mods += 1
                chain = []
                s, t = src, tgt
                while t is not None:
                    chain.append((s, t))
                    p = parent.get(t)
                    if p is None: break
                    s, t = t, p[0]
                winners.append({"depth": depth,
                                "module_rva": src - mod_base,
                                "chain": [[hex(x), hex(y)] for x, y in chain]})
            else:
                next_frontier.append(src)
        print(f"  new external: {len(next_frontier)}, module hits: {mods}")
        if winners:
            print(f"  FOUND {len(winners)} module-static chain(s) at depth={depth}")
            break
        # Cap frontier
        if len(next_frontier) > 4000:
            print(f"  truncating frontier {len(next_frontier)} -> 4000")
            next_frontier = next_frontier[:4000]
        frontier = next_frontier

    out = {"pid": pid, "mod_base": hex(mod_base), "anchors": [hex(t) for t in targets],
           "winners": winners}
    with open(r"C:\tmp\anchor_trace.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nTotal module chains: {len(winners)}")
    for w in winners[:10]:
        print(f"  depth={w['depth']}  module+0x{w['module_rva']:X}  chain:")
        for s, t in w['chain']:
            print(f"    {s} -> {t}")
    print("\nwrote C:\\tmp\\anchor_trace.json")
    return 0 if winners else 2

if __name__ == "__main__":
    sys.exit(main())
