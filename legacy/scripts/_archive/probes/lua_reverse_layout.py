"""Reverse-engineer Riot's custom Lua Table/Node layout empirically.

Strategy (layout-agnostic):
  1. Collect TString* addresses for a basket of well-known Lua strings
     (metamethods __index/__newindex/__gc/__call/__mode, plus Belveth* names).
     These are addresses that a Lua Table node would store as keys.
  2. For each candidate GC object (from lua_belveth_objects.json) and for
     each 'promising' heap page, scan for 8-byte-aligned u64 slots whose
     value is in the TString* set.
  3. Cluster the hits by proximity — a run of hits with uniform stride is
     a Node array. The stride is the Node size. The offset between
     consecutive key hits gives us the field offset of the key-pointer
     within the Node.
  4. Report (stride, key_offset) distribution. The dominant (stride,
     key_offset) tuple IS Riot's Lua Node layout.
  5. For the top cluster, dump a few entries showing (val8, key_ptr ->
     string, next_ofs) for verification.

Usage: python lua_reverse_layout.py
  Reads C:\\tmp\\belveth_lua_refs.json and C:\\tmp\\lua_belveth_objects.json.
  Writes C:\\tmp\\lua_layout_reversed.json.
"""
import ctypes, ctypes.wintypes as wt
import sys, subprocess, struct, json, time
from collections import Counter, defaultdict
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

# Strings we'll look for as TStrings. Their pointers from nodes are what
# we want to find. Include metamethods (very common), generic Lua keys,
# and Belveth-specific spell names (we know these are interned).
KEY_STRINGS = [
    b"__index\x00", b"__newindex\x00", b"__gc\x00", b"__call\x00",
    b"__tostring\x00", b"__metatable\x00", b"__mode\x00",
    b"__add\x00", b"__sub\x00", b"__mul\x00", b"__eq\x00",
    b"__len\x00", b"__lt\x00", b"__le\x00", b"__concat\x00",
    b"Belveth\x00", b"BelvethQ\x00", b"BelvethW\x00", b"BelvethE\x00",
    b"BelvethR\x00", b"BelvethBasicAttack\x00", b"BelvethPassive\x00",
    # common Lua/game API names
    b"OnUpdate\x00", b"OnTick\x00", b"OnSpellCast\x00", b"OnPreAttack\x00",
    b"name\x00", b"value\x00", b"mana\x00", b"health\x00",
    b"position\x00", b"owner\x00", b"target\x00", b"spell\x00",
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

def read_bytes(h, addr, n):
    buf = (ctypes.c_char * n)(); r = ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, n, ctypes.byref(r))
    return bytes(buf[:r.value]) if ok else None

def find_all(needle, hay):
    i = 0
    while True:
        j = hay.find(needle, i)
        if j == -1: break
        yield j
        i = j + 1

def main():
    pid = find_pid()
    if not pid: print("no league"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    print(f"pid={pid}")

    # Step 1: load all heap regions, find addresses of our key strings.
    print("\n== Loading regions + locating key strings ==")
    t0 = time.time()
    regions = []
    for base, size in readable_regions(h):
        data = read_region(h, base, size)
        if data: regions.append((base, data))
    print(f"  {len(regions)} regions in {time.time()-t0:.1f}s")

    # Find each key string's byte offset. The TString* pointer we'd see in
    # a Lua Node points to the TString header — which is 16..24 bytes
    # BEFORE the string's ASCII bytes (depending on Lua version).
    # We'll collect the char-address and accept pointers in [char-40, char].
    # Then we also store the "primary" chr address so we can map back.
    key_char_addrs = {}  # char_addr -> string (bytes, no \0)
    for needle in KEY_STRINGS:
        s = needle.rstrip(b"\x00")
        for base, data in regions:
            for off in find_all(needle, data):
                key_char_addrs[base + off] = s
    print(f"  {len(key_char_addrs)} key-string instances found")
    by_name = Counter(key_char_addrs.values())
    for name, n in by_name.most_common(10):
        print(f"    {name.decode(errors='replace'):<22} {n}")

    # Build target range set: for each char_addr, accept pointers into
    # [char_addr - 40, char_addr]. We'll store a dict tgt_ptr -> key_name
    # (lazy: we don't enumerate all 40 pre-bytes; instead we'll check
    # candidate pointers against the char_addr set at read time).
    sorted_char_addrs = sorted(key_char_addrs.keys())
    char_arr = np.array(sorted_char_addrs, dtype=np.uint64)

    def is_tstring_ptr(v):
        # True if v in [char_addr - 40, char_addr] for any char_addr.
        # Use binary search.
        idx = np.searchsorted(char_arr, np.uint64(v))
        if idx < len(char_arr) and int(char_arr[idx]) - v <= 40:
            return int(char_arr[idx])
        if idx > 0 and int(char_arr[idx-1]) - v <= 40 and v <= int(char_arr[idx-1]):
            return int(char_arr[idx-1])
        return 0

    # Step 2: scan all heap u64 slots for pointers into our key-string
    # targets. Record (abs_addr, ptr_value, string_name).
    print("\n== Scanning heap for TString* pointers to our key strings ==")
    t0 = time.time()
    # Build bitmap ranges: for each char_addr in char_arr, accept [c-40, c].
    # Use vectorized search: compute mask in one pass.
    lo_bound = int(char_arr.min()) - 40 if len(char_arr) else 0
    hi_bound = int(char_arr.max()) + 1 if len(char_arr) else 0
    ptr_hits = []  # list of (abs_addr, tgt_ptr, char_addr)
    for base, data in regions:
        usable = len(data) & ~7
        if usable < 8: continue
        arr = np.frombuffer(data[:usable], dtype=np.uint64)
        # rough filter
        mask = (arr >= lo_bound) & (arr < hi_bound)
        if not mask.any(): continue
        for i in np.nonzero(mask)[0]:
            v = int(arr[i])
            c = is_tstring_ptr(v)
            if c:
                ptr_hits.append((base + int(i) * 8, v, c))
    print(f"  {len(ptr_hits)} TString*-pointing slots found in {time.time()-t0:.1f}s")

    # Step 3: cluster hits by proximity to find Node arrays.
    # A Node array will have consecutive hits with uniform stride (the
    # Node size). We bin ptr_hits by region-local address and then scan
    # for runs with stride in {24, 32, 40, 48, 56, 64}.
    ptr_hits.sort()
    print("\n== Clustering consecutive hits to identify Node stride ==")
    strides = Counter()
    runs = []  # list of (start_addr, stride, count, samples[])
    i = 0
    # Group by region bucket (4MB) to keep clustering local
    BUCKET = 1 << 22
    by_bucket = defaultdict(list)
    for a, v, c in ptr_hits:
        by_bucket[a >> 22].append((a, v, c))

    for bucket, hits in by_bucket.items():
        hits.sort()
        # For each starting hit, look for up to 8 following hits with
        # consistent stride AND key diversity (distinct char_addrs).
        used = set()
        for i in range(len(hits)):
            if i in used: continue
            a0, v0, c0 = hits[i]
            # candidate strides: gap to next few hits
            for j in range(i + 1, min(i + 8, len(hits))):
                d = hits[j][0] - a0
                if d <= 0 or d > 256: continue
                if d not in (24, 32, 40, 48, 56, 64, 72, 80): continue
                hit_addrs = {hh[0]: hh for hh in hits}
                matches = [hits[i], hits[j]]
                for k in range(2, 32):
                    want = a0 + k * d
                    if want in hit_addrs:
                        matches.append(hit_addrs[want])
                    else:
                        break
                # require >=3 matches AND >=3 distinct keys (char_addr)
                distinct_keys = len(set(m[2] for m in matches))
                if len(matches) >= 3 and distinct_keys >= 3:
                    strides[d] += 1
                    runs.append((a0, d, len(matches), matches[:8], distinct_keys))
                    for m in matches:
                        ma = m[0]
                        for k, hh in enumerate(hits):
                            if hh[0] == ma: used.add(k)
                    break
    print("  Stride frequency among runs of >=3 consecutive hits:")
    for d, n in strides.most_common():
        print(f"    stride {d:>3d}: {n} runs")

    # Pick dominant stride
    if not strides:
        print("  no clear stride found — giving up"); return 0
    dominant_stride = strides.most_common(1)[0][0]
    dom_runs = [r for r in runs if r[1] == dominant_stride]
    print(f"\n== Top 15 runs (sorted by distinct_keys then count) ==")
    # Favor runs with many distinct keys (real hash Node arrays)
    runs.sort(key=lambda r: (-r[4], -r[2]))
    for a0, d, cnt, samples, distinct in runs[:15]:
        tag = "HEAP" if 0x10000000000 < a0 < 0x80000000000 else "MODULE"
        print(f"  start=0x{a0:X} ({tag})  stride={d}  entries={cnt}  distinct_keys={distinct}")
        for (addr, v, c) in samples:
            key = key_char_addrs.get(c, b"?").decode("ascii", errors="replace")
            print(f"    +0x{addr-a0:04X}  -> 0x{v:X}  key='{key}'")

    # Key pointer offset within Node — filter to HEAP runs only.
    heap_runs = [r for r in runs if 0x10000000000 < r[0] < 0x80000000000]
    key_offsets = Counter()
    for a0, d, cnt, samples, distinct in heap_runs:
        for (addr, v, c) in samples:
            key_offsets[(addr - a0) % d] += 1
    print(f"\n  Key-pointer offsets within Node (HEAP runs only):")
    for ofs, n in key_offsets.most_common(5):
        print(f"    offset +0x{ofs:X}: {n} hits")

    # Step 4: for the top dom_runs, inspect 1 Node in full and dump all
    # u64 slots. This shows the TValue for the VALUE associated with each
    # key, which will include pointers to spell scripts, sub-tables, and
    # eventually the click-dest struct.
    print(f"\n== Full Node dump for top 3 HEAP runs (first 8 Nodes each) ==")
    for a0, d, cnt, samples, distinct in [r for r in runs if 0x10000000000 < r[0] < 0x80000000000][:3]:
        n_nodes = min(cnt, 8)
        # Also try reading a bit BEFORE the first hit — maybe key is not
        # at offset 0 of the Node, in which case the true Node start is
        # earlier. We'll dump 16 bytes before as context.
        raw = read_bytes(h, a0 - 16, d * n_nodes + 16)
        if not raw: continue
        print(f"\n  RUN: start=0x{a0:X} stride={d} distinct_keys={distinct}")
        # Pre-context
        for off in range(0, 16, 8):
            u64 = struct.unpack_from("<Q", raw, off)[0]
            print(f"    (pre) -0x{16-off:02X}  0x{u64:016X}")
        for ni in range(n_nodes):
            node_addr = a0 + ni*d
            print(f"  Node[{ni}] at 0x{node_addr:X}:")
            for off in range(0, d, 8):
                if 16 + ni*d + off + 8 > len(raw): break
                u64 = struct.unpack_from("<Q", raw, 16 + ni*d + off)[0]
                note = ""
                c = is_tstring_ptr(u64)
                if c:
                    note = f"  KEY='{key_char_addrs[c].decode(errors='replace')}'"
                elif 0x10000000000 < u64 < 0x80000000000:
                    note = "  HEAP"
                # interpret as float too
                b8 = raw[16 + ni*d + off:16 + ni*d + off + 8]
                f1, f2 = struct.unpack("<ff", b8)
                if -1e6 < f1 < 1e6 and -1e6 < f2 < 1e6 and (abs(f1) > 0.01 or abs(f2) > 0.01):
                    note += f"  float=({f1:.2f}, {f2:.2f})"
                print(f"    +{off:02X}  0x{u64:016X}{note}")

    # Step 5: for the top HEAP stride-64 runs (Riot's apparent Node size),
    # follow the "value" ptr at +0x20 and check if it points near the
    # click-dest alloc (0x1F37A7793A4). Look within ±0x1000000 heap vicinity.
    CLICK_ALLOC = 0x1F37A7793A4
    print(f"\n== Checking value ptrs for proximity to click alloc 0x{CLICK_ALLOC:X} ==")
    heap_64 = [r for r in runs if r[1] == 64 and 0x10000000000 < r[0] < 0x80000000000]
    for a0, d, cnt, samples, distinct in heap_64:
        n_nodes = min(cnt * 2, 16)  # also look a bit past the detected run
        raw = read_bytes(h, a0, d * n_nodes)
        if not raw: continue
        print(f"  run at 0x{a0:X}  distinct_keys={distinct}")
        for ni in range(n_nodes):
            if ni * d + 48 > len(raw): break
            key_ptr = struct.unpack_from("<Q", raw, ni*d + 0)[0]
            val_ptr = struct.unpack_from("<Q", raw, ni*d + 0x20)[0]
            c = is_tstring_ptr(key_ptr)
            key_name = key_char_addrs.get(c, b"<?>").decode(errors='replace') if c else "<not-key>"
            dist = val_ptr - CLICK_ALLOC if val_ptr else 0
            note = ""
            if val_ptr and 0x10000000000 < val_ptr < 0x80000000000:
                if abs(dist) < 0x1000000:
                    note = f"  NEAR click_alloc ({dist:+d})"
                elif abs(dist) < 0x10000000:
                    note = f"  same-16MB vicinity ({dist:+d})"
            print(f"    Node[{ni:2d}] key='{key_name:<24}'  val=0x{val_ptr:X}{note}")

    out = {
        "pid": pid,
        "stride_histogram": {str(k): v for k, v in strides.most_common()},
        "top_runs": [{"start": hex(a0), "stride": d, "count": cnt, "distinct_keys": dk}
                      for (a0, d, cnt, _, dk) in sorted(runs, key=lambda r:(-r[4],-r[2]))[:30]],
    }
    with open(r"C:\tmp\lua_layout_reversed.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote C:\\tmp\\lua_layout_reversed.json")
    return 0

if __name__ == "__main__":
    sys.exit(main())
