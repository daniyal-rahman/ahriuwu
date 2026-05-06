"""Process-wide memory scan for click-destination Vec3 fields.

Plays the replay at 0.25x and pauses at the midpoint of each of three click
windows. In each pause, walks all committed RW private heap regions in the
League process and records every 4-byte-aligned offset where:
    float32(addr)   ≈ expected_x  (tolerance ±5u)
    float32(addr+8) ≈ expected_z  (tolerance ±5u)

Writes per-window absolute-address lists to C:\\tmp\\scan_winN.json.
Offline intersection identifies addresses that held the right destination
in all three windows — candidate click-location memory fields.

Note: only works if heap allocations are stable across pauses (no major
re-allocation between windows). Pauses without seek should preserve this.
"""
import ctypes, ctypes.wintypes as wt, struct, subprocess, sys, json, time, os
import ssl, urllib.request
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

EXPECTED = [
    ("win1", 40.0, 3124.0, 8122.0),  # pause at gt=40.0, search for click-1 dest
    ("win2", 44.5, 3736.0, 8358.0),  # pause at gt=44.5, search for click-2 dest
    ("win3", 49.0, 4398.0, 8444.0),  # pause at gt=49.0, search for click-3 dest
]
TOL = 5.0

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
PAGE_RW_FLAGS = 0x04 | 0x08 | 0x40  # PAGE_READWRITE | PAGE_WRITECOPY | PAGE_EXECUTE_READWRITE

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])

def enumerate_regions(h):
    """Yield (base, size) for every MEM_COMMIT + MEM_PRIVATE + RW page."""
    addr = 0
    mbi = MBI()
    while addr < 0x7FFFFFFFFFFF:
        ok = _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi))
        if not ok: break
        base = mbi.BaseAddress or 0
        size = mbi.RegionSize
        if (mbi.State == MEM_COMMIT and
            mbi.Type == MEM_PRIVATE and
            (mbi.Protect & PAGE_RW_FLAGS)):
            yield base, size
        addr = base + size
        if addr <= base: break

def read_region(h, base, size, max_chunk=4*1024*1024):
    """Read `size` bytes starting at `base`. Returns bytes or None on error."""
    if size > 512*1024*1024:  # cap absurd regions
        return None
    out = bytearray(size)
    view = memoryview(out)
    offset = 0
    while offset < size:
        n = min(max_chunk, size - offset)
        buf = (ctypes.c_char * n)()
        read = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(h, ctypes.c_void_p(base + offset), buf, n, ctypes.byref(read))
        if not ok or read.value == 0:
            return None if offset == 0 else bytes(view[:offset])
        view[offset:offset + read.value] = buf[:read.value]
        offset += read.value
    return bytes(out)

def scan_for_vec3(data, base_addr, ex, ez, tol):
    """Yield absolute addresses where float(addr) ≈ ex and float(addr+8) ≈ ez."""
    # Use numpy if available for speed
    try:
        import numpy as np
        arr = np.frombuffer(data, dtype=np.float32)
        n = len(arr)
        if n < 3: return
        # mask: arr[i] ≈ ex and arr[i+2] ≈ ez  (indices advance by 1 float = 4 bytes)
        mask_x = np.abs(arr[:-2] - ex) < tol
        if not mask_x.any(): return
        mask_xz = mask_x & (np.abs(arr[2:] - ez) < tol)
        # Also filter NaN: np.isfinite
        mask_xz &= np.isfinite(arr[:-2]) & np.isfinite(arr[2:])
        # Additionally: skip where ex==0 or ez==0 would trivially hit; require non-zero
        idx = np.nonzero(mask_xz)[0]
        for i in idx:
            yield base_addr + int(i) * 4
    except ImportError:
        # fallback, slower
        for off in range(0, len(data) - 12, 4):
            x = struct.unpack_from("<f", data, off)[0]
            if abs(x - ex) >= tol: continue
            z = struct.unpack_from("<f", data, off + 8)[0]
            if abs(z - ez) < tol and z == z and x == x:
                yield base_addr + off

_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
def _post(obj):
    req = urllib.request.Request("https://127.0.0.1:2999/replay/playback",
            data=json.dumps(obj).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return json.loads(r.read())
def _get():
    with urllib.request.urlopen("https://127.0.0.1:2999/replay/playback", context=_ctx, timeout=2) as r:
        return json.loads(r.read())

def wait_till_at(target_gt, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = _get()
        if st["time"] >= target_gt:
            return st
        time.sleep(0.1)
    return _get()

def play_slow_and_pause_at(gt, speed=0.25):
    """Play at `speed` until game_time >= gt, then pause."""
    _post({"speed": speed, "paused": False})
    st = wait_till_at(gt)
    _post({"speed": speed, "paused": True})
    time.sleep(0.3)
    return _get()

def main():
    pid = find_pid()
    h = _k.OpenProcess(0x0410, False, pid)
    print(f"PID={pid} handle=0x{h:X}")

    # Seek once to before all windows
    print(f"\nSeeking to gt=36, paused")
    _post({"time": 36.0, "speed": 0.25, "paused": True})
    for _ in range(30):
        time.sleep(0.3)
        st = _get()
        if not st["seeking"] and st["paused"]: break

    results = {}
    for label, target_gt, ex, ez in EXPECTED:
        print(f"\n--- {label}: play at 0.25x to gt={target_gt}, then scan for ({ex},?,{ez}) ---")
        st = play_slow_and_pause_at(target_gt)
        print(f"  paused at gt={st['time']:.2f}")
        regions = list(enumerate_regions(h))
        total_bytes = sum(sz for _, sz in regions)
        print(f"  {len(regions)} RW private regions, {total_bytes/(1024*1024):.1f} MB total")
        hits = []
        scanned = 0
        t0 = time.time()
        for base, size in regions:
            data = read_region(h, base, size)
            if not data: continue
            for a in scan_for_vec3(data, base, ex, ez, TOL):
                hits.append(a)
            scanned += len(data)
        dt = time.time() - t0
        print(f"  scanned {scanned/(1024*1024):.1f} MB in {dt:.1f}s, {len(hits)} Vec3 hits")
        results[label] = {"gt_paused": st["time"], "expected": [ex, ez], "hits": [hex(a) for a in hits]}
        out_path = fr"C:\tmp\scan_{label}.json"
        with open(out_path, "w") as f: json.dump(results[label], f)
        print(f"  -> {out_path}")

    # Intersection
    print("\n--- intersection ---")
    addr_sets = [set(int(a, 16) for a in results[label]["hits"]) for label, *_ in EXPECTED]
    # First: same absolute address must be a hit in all 3 windows
    common = set.intersection(*addr_sets)
    print(f"Addresses that matched in all 3 windows: {len(common)}")
    for a in sorted(common)[:40]:
        print(f"  0x{a:X}")
    with open(r"C:\tmp\scan_intersection.json", "w") as f:
        json.dump({"common": [hex(a) for a in common]}, f, indent=2)

    # Also count pair intersections for diagnostics
    for i in range(3):
        for j in range(i+1, 3):
            inter = addr_sets[i] & addr_sets[j]
            print(f"  win{i+1} ∩ win{j+1}: {len(inter)}")

if __name__ == "__main__":
    main()
