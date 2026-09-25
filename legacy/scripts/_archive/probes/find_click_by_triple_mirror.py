"""Find the click-destination allocation by the triple-Vec3-mirror signature.

The click-dest struct stores the same Vec3 at three offsets inside a ~1KB
allocation: +0x000, +0x308, +0x374. This is unique enough to scan for
directly — no anchor needed, no knowledge of the click value in advance.

Scan invariant per candidate address X:
    Vec3(X+0x000) == Vec3(X+0x308) == Vec3(X+0x374)
    x, z are in the map coord range [0, 16000]
    y is in the terrain height range [0, 1000]

Emits hits as (addr, vec3). Finds the click alloc fresh on every session.

Usage:   python find_click_by_triple_mirror.py [--verify-with-known]
"""
import ctypes, ctypes.wintypes as wt
import sys, subprocess, time, json

sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

# ── Win32 plumbing (same as trace_click_anchor) ─────────────────────────
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
PAGE_RW = 0x04 | 0x08 | 0x40

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower():
            return int(l.strip('"').split('","')[1])
    return None

def rw_regions(h):
    addr = 0; mbi = MBI()
    while addr < 0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
            break
        base = mbi.BaseAddress or 0; size = mbi.RegionSize
        if (mbi.State == MEM_COMMIT and mbi.Type == MEM_PRIVATE and
            (mbi.Protect & PAGE_RW)):
            yield base, size
        addr = base + size
        if addr <= base: break

def read_region(h, base, size):
    if size > 512 * 1024 * 1024: return None
    out = bytearray(size); view = memoryview(out)
    off = 0
    CHUNK = 4 * 1024 * 1024
    while off < size:
        n = min(CHUNK, size - off)
        buf = (ctypes.c_char * n)()
        read = ctypes.c_size_t(0)
        if not _k.ReadProcessMemory(h, ctypes.c_void_p(base + off), buf, n,
                                    ctypes.byref(read)) or read.value == 0:
            return None if off == 0 else bytes(view[:off])
        view[off:off + read.value] = buf[:read.value]
        off += read.value
    return bytes(out)

# ── The scan ────────────────────────────────────────────────────────────
MIRROR_OFFSETS = (0x000, 0x308, 0x374)
ALLOC_SIZE_MAX = 0x400  # struct we care about ends ~0x3C0

def scan_for_triple_mirror(data, base):
    """Find all offsets within `data` (region starting at `base`) where the
    Vec3 at +0, +0x308, +0x374 all match and are plausible click coords."""
    import numpy as np
    # Interpret the whole region as float32 array
    if len(data) < ALLOC_SIZE_MAX + 16:
        return []
    n = len(data) // 4
    arr = np.frombuffer(data, dtype=np.float32, count=n)

    # For a candidate starting at float-index i (i.e. byte offset i*4):
    #   mirror_0 = arr[i], arr[i+1], arr[i+2]   (bytes +0, +4, +8)
    #   mirror_1 = arr[i + 0x308/4], arr[i + 0x308/4 + 1], arr[i + 0x308/4 + 2]
    #   mirror_2 = arr[i + 0x374/4], arr[i + 0x374/4 + 1], arr[i + 0x374/4 + 2]

    off1 = 0x308 // 4
    off2 = 0x374 // 4

    max_i = n - off2 - 3
    if max_i <= 0:
        return []

    x0 = arr[:max_i]; y0 = arr[1:max_i + 1]; z0 = arr[2:max_i + 2]
    x1 = arr[off1:off1 + max_i]; y1 = arr[off1 + 1:off1 + 1 + max_i]; z1 = arr[off1 + 2:off1 + 2 + max_i]
    x2 = arr[off2:off2 + max_i]; y2 = arr[off2 + 1:off2 + 1 + max_i]; z2 = arr[off2 + 2:off2 + 2 + max_i]

    # Plausible click coords: x, z in [100, 16000], y in [-500, 1500]
    valid = (
        np.isfinite(x0) & np.isfinite(y0) & np.isfinite(z0) &
        (x0 > 100) & (x0 < 16000) &
        (z0 > 100) & (z0 < 16000) &
        (y0 > -500) & (y0 < 1500)
    )

    # Triple equality — exact bitwise would be strongest, but allow 1e-3
    same = (
        (x0 == x1) & (x0 == x2) &
        (y0 == y1) & (y0 == y2) &
        (z0 == z1) & (z0 == z2)
    )

    # Every float index (4-byte aligned). The real allocation in a prior scan
    # landed at an address ending in 0x...4, i.e. 4-byte but not 8-byte
    # aligned, so we must not drop odd float indices.
    hits_mask = valid & same
    hits = []
    NIL_PATTERN = b"<nil>"   # ASCII at byte +0x0C of click-dest struct
    for i in np.nonzero(hits_mask)[0]:
        addr = base + int(i) * 4
        # Check the ASCII disambiguator: bytes +0x0C..+0x10 should be "<nil>"
        byte_off = int(i) * 4 + 0x0C
        has_nil = (byte_off + 5 < len(data) and
                   data[byte_off:byte_off + 5] == NIL_PATTERN)
        hits.append((addr, float(x0[i]), float(y0[i]), float(z0[i]), has_nil))
    return hits

def main():
    pid = find_pid()
    if not pid:
        print("ERR: League not running"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    print(f"pid={pid}")

    t0 = time.time()
    all_hits = []
    n_regions = 0
    total_bytes = 0
    for base, size in rw_regions(h):
        n_regions += 1
        data = read_region(h, base, size)
        if not data: continue
        total_bytes += len(data)
        hits = scan_for_triple_mirror(data, base)
        all_hits.extend(hits)

    dt = time.time() - t0
    print(f"scanned {n_regions} regions, {total_bytes/(1024*1024):.1f} MB in {dt:.1f}s")
    print(f"triple-mirror hits total: {len(all_hits)}")
    nil_hits = [h for h in all_hits if h[4]]
    print(f"with '<nil>' ASCII at +0x0C: {len(nil_hits)}")
    print("\nHits WITH <nil> (likely click-dest structs):")
    for addr, x, y, z, has_nil in nil_hits[:20]:
        print(f"  0x{addr:X}  Vec3=({x:.2f}, {y:.2f}, {z:.2f})")
    print("\nHits WITHOUT <nil> (other triple-mirror structs), sample:")
    for addr, x, y, z, has_nil in [h for h in all_hits if not h[4]][:10]:
        print(f"  0x{addr:X}  Vec3=({x:.2f}, {y:.2f}, {z:.2f})")

    if all_hits:
        out = {
            "pid": pid,
            "scan_duration_s": round(dt, 2),
            "total_hits": len(all_hits),
            "nil_hits": len(nil_hits),
            "hits": [
                {"addr": hex(a), "x": x, "y": y, "z": z, "has_nil": n}
                for a, x, y, z, n in all_hits
            ],
        }
        with open(r"C:\tmp\triple_mirror_hits.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote C:\\tmp\\triple_mirror_hits.json")

    return 0 if nil_hits else (2 if all_hits else 3)

if __name__ == "__main__":
    sys.exit(main())
