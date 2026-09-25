"""Trace a stable pointer chain from a module-static pointer down to the
click-destination allocation in League of Legends replay memory.

Why: click_destination_memory_found.md identifies a 1KB allocation with 3 Vec3
mirrors of the click destination (alloc+0, +0x308, +0x374). The allocation base
is ephemeral — a new session puts it at a new address. We cannot simply
remember the address. We need a reproducible FINDER: a chain of pointer-hops
from a known-stable root (module-static pointer in League.exe's image) down
to the allocation.

Algorithm:
  1. Locate the allocation fresh this session by running process_scan on known
     click destinations (same logic as belveth_process_scan.py).
  2. Reverse pointer walk, bounded depth:
       level-0 = { alloc_base .. alloc_base + 0x400 }
       level-k = process-wide u64 scan for pointers whose target ∈ level-(k-1)
     At each level, classify sources: if the source is in the League.exe image
     range, it's a candidate module-static anchor — record the chain.
  3. Validate: the same chain (module RVAs + struct offsets) must reach a
     similar-shape allocation on a SECOND run. Only chains that survive count.

Assumes:
  - League replay running, Bel'Veth selected (API selectionName sets the
    focused-hero pointer even when rendered cam isn't keyboard-locked).
  - Expected click destinations known (from prior keylog ground truth).

Writes results to C:\\tmp\\click_anchor_trace.json.
"""
import ctypes, ctypes.wintypes as wt
import struct, subprocess, sys, json, time, os, ssl, urllib.request

sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

EXPECTED_CLICKS = [
    # (label, pause_gt, expected_x, expected_z)
    ("win1", 40.0, 3124.0, 8122.0),
    ("win2", 44.5, 3736.0, 8358.0),
    ("win3", 49.0, 4398.0, 8444.0),
]
VEC_TOL = 5.0

MAX_CHAIN_DEPTH = 6          # 0..6 hops from module-static to alloc
MAX_L1_SOURCES = 5000        # cap per-level to keep the walk tractable
CHUNK = 4 * 1024 * 1024

# ─── Win32 plumbing ──────────────────────────────────────────────────────
_k = ctypes.windll.kernel32
class MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", ctypes.c_ulong), ("__a", ctypes.c_ulong),
                ("RegionSize", ctypes.c_size_t), ("State", ctypes.c_ulong),
                ("Protect", ctypes.c_ulong), ("Type", ctypes.c_ulong),
                ("__b", ctypes.c_ulong)]
_k.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
_k.VirtualQueryEx.restype = ctypes.c_size_t
MEM_COMMIT = 0x1000; MEM_PRIVATE = 0x20000; MEM_IMAGE = 0x1000000
PAGE_READABLE = 0x02 | 0x04 | 0x08 | 0x10 | 0x20 | 0x40 | 0x80

def open_proc(pid):
    h = _k.OpenProcess(0x0410, False, pid)
    if not h: raise RuntimeError(f"OpenProcess({pid}) failed: {ctypes.get_last_error()}")
    return h

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])
    return None

def enumerate_regions(h, rw_private_only=False):
    """Yield (base, size, state, type, protect)."""
    addr = 0
    mbi = MBI()
    while addr < 0x7FFFFFFFFFFF:
        ok = _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi))
        if not ok: break
        base = mbi.BaseAddress or 0
        size = mbi.RegionSize
        if mbi.State == MEM_COMMIT and (mbi.Protect & PAGE_READABLE):
            if (not rw_private_only) or (mbi.Type == MEM_PRIVATE and (mbi.Protect & 0x0C)):
                yield base, size, mbi.State, mbi.Type, mbi.Protect
        addr = base + size
        if addr <= base: break

def read_region(h, base, size):
    if size > 512 * 1024 * 1024: return None
    out = bytearray(size)
    view = memoryview(out)
    off = 0
    while off < size:
        n = min(CHUNK, size - off)
        buf = (ctypes.c_char * n)()
        read = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(h, ctypes.c_void_p(base + off), buf, n, ctypes.byref(read))
        if not ok or read.value == 0:
            return None if off == 0 else bytes(view[:off])
        view[off:off + read.value] = buf[:read.value]
        off += read.value
    return bytes(out)

def module_range(pid):
    """Locate League of Legends.exe's image base + size via PSAPI."""
    psapi = ctypes.WinDLL("psapi.dll")
    h = open_proc(pid)
    HMODULE = wt.HMODULE
    psapi.EnumProcessModulesEx.argtypes = [wt.HANDLE, ctypes.POINTER(HMODULE),
                                           wt.DWORD, ctypes.POINTER(wt.DWORD),
                                           wt.DWORD]
    psapi.EnumProcessModulesEx.restype = wt.BOOL
    psapi.GetModuleFileNameExW.argtypes = [wt.HANDLE, HMODULE,
                                           wt.LPWSTR, wt.DWORD]
    psapi.GetModuleFileNameExW.restype = wt.DWORD
    class MINFO(ctypes.Structure):
        _fields_ = [("lpBaseOfDll", ctypes.c_void_p),
                    ("SizeOfImage", wt.DWORD),
                    ("EntryPoint", ctypes.c_void_p)]
    psapi.GetModuleInformation.argtypes = [wt.HANDLE, HMODULE,
                                           ctypes.POINTER(MINFO), wt.DWORD]
    psapi.GetModuleInformation.restype = wt.BOOL

    mods = (HMODULE * 1024)(); needed = wt.DWORD(0)
    ok = psapi.EnumProcessModulesEx(h, mods, ctypes.sizeof(mods),
                                    ctypes.byref(needed), 3)
    if not ok:
        raise RuntimeError(f"EnumProcessModulesEx failed: {ctypes.get_last_error()}")
    n = needed.value // ctypes.sizeof(HMODULE)
    for i in range(n):
        name = ctypes.create_unicode_buffer(260)
        psapi.GetModuleFileNameExW(h, mods[i], name, 260)
        if name.value.lower().endswith("league of legends.exe"):
            mi = MINFO()
            psapi.GetModuleInformation(h, mods[i], ctypes.byref(mi), ctypes.sizeof(mi))
            return mi.lpBaseOfDll, mi.SizeOfImage
    return None, None

# ─── Replay API ──────────────────────────────────────────────────────────
_ctx = ssl.create_default_context(); _ctx.check_hostname = False; _ctx.verify_mode = ssl.CERT_NONE
def api_post(path, obj):
    req = urllib.request.Request(f"https://127.0.0.1:2999{path}",
            data=json.dumps(obj).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return json.loads(r.read())
def api_get(path):
    with urllib.request.urlopen(f"https://127.0.0.1:2999{path}",
                                context=_ctx, timeout=3) as r: return json.loads(r.read())

def wait_till(gt, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = api_get("/replay/playback")
        if st["time"] >= gt: return st
        time.sleep(0.1)
    return api_get("/replay/playback")

def play_and_pause_at(gt, speed=0.25):
    api_post("/replay/playback", {"speed": speed, "paused": False})
    wait_till(gt)
    api_post("/replay/playback", {"speed": speed, "paused": True})
    time.sleep(0.3)
    return api_get("/replay/playback")

# ─── Scan: find click-dest allocation ────────────────────────────────────
def scan_vec3(data, base, ex, ez, tol):
    import numpy as np
    arr = np.frombuffer(data, dtype=np.float32)
    n = len(arr)
    if n < 3: return []
    m_x = np.abs(arr[:-2] - ex) < tol
    if not m_x.any(): return []
    m_xz = m_x & (np.abs(arr[2:] - ez) < tol)
    m_xz &= np.isfinite(arr[:-2]) & np.isfinite(arr[2:])
    idx = np.nonzero(m_xz)[0]
    return [base + int(i) * 4 for i in idx]

def find_click_alloc(h):
    """Returns (alloc_base, addrs_in_alloc) where addrs = 3 mirrors we matched."""
    api_post("/replay/playback", {"time": 36.0, "speed": 0.25, "paused": True})
    for _ in range(30):
        time.sleep(0.3)
        st = api_get("/replay/playback")
        if not st["seeking"] and st["paused"]: break

    per_window_hits = []
    for label, target_gt, ex, ez in EXPECTED_CLICKS:
        print(f"[scan] {label}: play to gt={target_gt}")
        st = play_and_pause_at(target_gt)
        print(f"       paused at gt={st['time']:.2f}, scanning for ({ex},?,{ez})")
        hits = []
        for base, size, _, typ, _ in enumerate_regions(h, rw_private_only=True):
            data = read_region(h, base, size)
            if not data: continue
            for a in scan_vec3(data, base, ex, ez, VEC_TOL):
                hits.append(a)
        print(f"       {len(hits)} Vec3 hits")
        per_window_hits.append(set(hits))

    common = set.intersection(*per_window_hits) if per_window_hits else set()
    print(f"[scan] addresses matching all 3 windows: {len(common)}")
    if not common:
        return None, []
    # The 3 mirror addresses should share a ~0x400 span. Group by alloc base.
    # Strategy: find the densest cluster.
    sorted_addrs = sorted(common)
    groups = []
    cur = [sorted_addrs[0]]
    for a in sorted_addrs[1:]:
        if a - cur[0] < 0x800:
            cur.append(a)
        else:
            groups.append(cur); cur = [a]
    groups.append(cur)
    # Pick the group with ≥3 hits closest to matching the +0/+0x308/+0x374 stride
    best = max(groups, key=lambda g: (len(g), -(max(g) - min(g))))
    alloc_base = min(best)
    print(f"[scan] best alloc_base candidate: 0x{alloc_base:X}  (mirrors={[hex(a) for a in best]})")
    return alloc_base, best

# ─── Reverse pointer walk ───────────────────────────────────────────────
def pointer_scan(h, tgt_range, all_regions):
    """Find every u64 (checked at 4- and 8-byte alignments) whose value falls
    within tgt_range=(lo, hi). Returns list of (src_addr, target_addr)."""
    import numpy as np
    lo, hi = tgt_range
    hits = []
    for base, size in all_regions:
        data = read_region(h, base, size)
        if not data: continue
        for align in (8, 4):
            trim = (base + align - 1) & ~(align - 1)
            skip = trim - base
            if skip >= len(data): continue
            # Number of 8-byte values we can read starting at alignment
            usable = (len(data) - skip) & ~7
            if usable < 8: continue
            # Read u64 at each `align`-byte offset (overlapping if align=4)
            if align == 8:
                arr = np.frombuffer(data, dtype=np.uint64,
                                    count=usable // 8, offset=skip)
                mask = (arr >= lo) & (arr < hi)
                if not mask.any(): continue
                for i in np.nonzero(mask)[0]:
                    src = trim + int(i) * 8
                    hits.append((src, int(arr[i])))
            else:
                # 4-byte aligned but 8-byte value — offset of 4 from the 8-aligned grid
                off4 = skip + 4
                usable4 = (len(data) - off4) & ~7
                if usable4 < 8: continue
                arr = np.frombuffer(data, dtype=np.uint64,
                                    count=usable4 // 8, offset=off4)
                mask = (arr >= lo) & (arr < hi)
                if not mask.any(): continue
                for i in np.nonzero(mask)[0]:
                    src = off4 + base - base + int(i) * 8 + base  # = base + off4 + i*8 — but base already in off4? no, off4 is offset in data
                    src = base + off4 + int(i) * 8
                    hits.append((src, int(arr[i])))
    # Dedup (4-byte scan can overlap 8-byte hits offset by 4)
    return list({(s, t): (s, t) for s, t in hits}.values())

def classify(src, mod_base, mod_size):
    """Return 'module' if in League image, else 'heap'."""
    if mod_base is not None and mod_base <= src < mod_base + mod_size:
        return "module"
    return "heap"

def reverse_walk(h, alloc_base, alloc_size, mod_base, mod_size,
                 max_depth=MAX_CHAIN_DEPTH):
    """Walk backwards from [alloc_base, alloc_base+alloc_size) looking for a
    chain of pointers that ends at a module-static pointer. Excludes sources
    that fall within the alloc itself (internal self-pointers aren't anchors)."""
    all_regions = [(b, s) for b, s, _, _, _ in enumerate_regions(h, rw_private_only=False)]
    print(f"[walk] {len(all_regions)} readable regions")
    alloc_end = alloc_base + alloc_size
    def is_internal(addr):
        return alloc_base <= addr < alloc_end

    parent = {}   # addr -> (prev_addr, depth); prev is what addr points to (downstream)
    # Level 0 range = alloc
    cur_ranges = [(alloc_base, alloc_base + alloc_size)]
    found_module_chains = []

    for depth in range(1, max_depth + 1):
        # Collapse ranges into one (lo,hi) for pointer_scan; hits will be filtered
        lo = min(r[0] for r in cur_ranges); hi = max(r[1] for r in cur_ranges)
        print(f"[walk] depth={depth}  range=[0x{lo:X},0x{hi:X})")
        hits = pointer_scan(h, (lo, hi), all_regions)
        # Filter: keep hits whose tgt actually falls in one of cur_ranges
        def in_any(v):
            for rlo, rhi in cur_ranges:
                if rlo <= v < rhi: return True
            return False
        hits = [(s, t) for s, t in hits if in_any(t)]
        print(f"[walk] depth={depth}  {len(hits)} pointer hits")
        if not hits: break
        # Dedup by src address (a src may show up multiple times via overlap)
        seen = set()
        uhits = []
        for s, t in hits:
            if s in seen: continue
            seen.add(s); uhits.append((s, t))
        print(f"[walk] depth={depth}  unique sources: {len(uhits)}  (first 10: "
              f"{[hex(s) for s,_ in uhits[:10]]})")
        hits = uhits
        new_srcs = []
        n_already = 0
        n_module = 0
        for src, tgt in hits:
            if src in parent:
                n_already += 1
                continue
            parent[src] = (tgt, depth)
            if classify(src, mod_base, mod_size) == "module":
                n_module += 1
                # Reconstruct full chain
                chain = [(src, tgt)]
                t = tgt
                while True:
                    p = parent.get(t)
                    if p is None or p[0] is None: break
                    chain.append((t, p[0])); t = p[0]
                found_module_chains.append({
                    "depth": depth,
                    "module_rva": src - mod_base,
                    "chain": [[hex(s), hex(tt)] for s, tt in chain],
                })
            else:
                # Skip alloc-internal self-pointers — they aren't anchors
                if depth == 1 and is_internal(src):
                    continue
                new_srcs.append(src)
        print(f"[walk] depth={depth}  new_external={len(new_srcs)}  module={n_module}  already_seen={n_already}")
        if found_module_chains:
            print(f"[walk] FOUND {len(found_module_chains)} module chain(s) at depth={depth}")
            break
        # Next depth targets: the set of pointer source addresses we just found.
        # Bucket them into small ranges (pages) to make the range check cheap.
        new_srcs.sort()
        ranges = []
        if new_srcs:
            a = b = new_srcs[0]
            for s in new_srcs[1:]:
                if s - b <= 0x1000:
                    b = s
                else:
                    ranges.append((a, b + 8)); a = b = s
            ranges.append((a, b + 8))
        cur_ranges = ranges
        print(f"[walk] depth={depth}  next-level source ranges: {len(cur_ranges)}")
        if len(cur_ranges) > MAX_L1_SOURCES:
            print(f"[walk] truncating ranges {len(cur_ranges)} -> {MAX_L1_SOURCES}")
            cur_ranges = cur_ranges[:MAX_L1_SOURCES]
        if not cur_ranges: break

    return found_module_chains

# ─── Main ────────────────────────────────────────────────────────────────
def main():
    pid = find_pid()
    if not pid:
        print("ERR: League of Legends.exe not running"); return 1
    h = open_proc(pid)
    mod_base, mod_size = module_range(pid)
    print(f"[main] pid={pid}  module_base=0x{mod_base:X}  size=0x{mod_size:X}")

    # Make sure Bel'Veth is selected for focused-hero pointer
    try:
        api_post("/replay/render", {"interfaceAll": True, "selectionName": "Belveth"})
    except Exception as e:
        print(f"[main] render api note: {e}")

    alloc_base, mirrors = find_click_alloc(h)
    if alloc_base is None:
        print("ERR: click-dest allocation not found — check click destinations and replay state")
        return 1
    print(f"[main] click-dest allocation base = 0x{alloc_base:X}  RVA vs module = 0x{alloc_base - mod_base:X}")

    # Reverse walk — target range is [alloc_base, alloc_base + 0x400)
    chains = reverse_walk(h, alloc_base, 0x400, mod_base, mod_size)

    out = {
        "pid": pid,
        "module_base": hex(mod_base),
        "module_size": hex(mod_size),
        "alloc_base": hex(alloc_base),
        "alloc_rva_if_static": hex(alloc_base - mod_base),
        "mirrors": [hex(a) for a in mirrors],
        "chains": chains,
    }
    with open(r"C:\tmp\click_anchor_trace.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[main] wrote C:\\tmp\\click_anchor_trace.json")
    print(f"[main] module chains found: {len(chains)}")
    for c in chains[:5]:
        print(f"  depth={c['depth']}  module_rva=0x{c['module_rva']:X}  chain_len={len(c['chain'])}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
