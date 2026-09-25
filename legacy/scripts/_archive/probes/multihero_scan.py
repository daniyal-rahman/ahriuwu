"""Multi-hero click-destination memory scan.

Deep-seeks to reset heap, plays forward through the test window to let both
Bel'Veth's and Illaoi's clicks fire, pauses at gt=51, then scans memory for
both destinations:
  Bel'Veth  (4398, ?, 8444) — her click-3 arrival, held gt 47.86..50.95
  Illaoi    (11988, ?, 4337) — her arrival observed at gt~49.8

For each, counts hits and checks whether the 3-mirror-stride (Δ=0x308, 0x6C)
pattern appears. If both have their own 3-mirror allocation, per-hero click
storage is confirmed.
"""
import ctypes, ctypes.wintypes as wt, struct, subprocess, sys, json, time, ssl, urllib.request
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

TARGETS = [
    ("Belveth_click3", 4398.0, 8444.0),
    ("Illaoi_arrival", 11988.0, 4337.0),
]
TOL = 5.0
PROBE_GT = 51.0  # pause here — Belveth and Illaoi both at their latest dest

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
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])

def enum_rw(h):
    addr = 0; mbi = MBI()
    while addr < 0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)): break
        base = mbi.BaseAddress or 0; size = mbi.RegionSize
        if (mbi.State == MEM_COMMIT and mbi.Type == MEM_PRIVATE and (mbi.Protect & PAGE_RW)):
            yield base, size
        addr = base + size
        if addr <= base: break

def read_region(h, base, size, chunk=4*1024*1024):
    if size > 512*1024*1024: return None
    out = bytearray(size); view = memoryview(out); off = 0
    while off < size:
        n = min(chunk, size - off)
        buf = (ctypes.c_char * n)()
        rd = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(h, ctypes.c_void_p(base + off), buf, n, ctypes.byref(rd))
        if not ok or rd.value == 0: return None if off == 0 else bytes(view[:off])
        view[off:off+rd.value] = buf[:rd.value]
        off += rd.value
    return bytes(out)

def scan_for_vec3(data, base_addr, ex, ez, tol):
    import numpy as np
    arr = np.frombuffer(data, dtype=np.float32)
    if len(arr) < 3: return []
    mask_x = np.abs(arr[:-2] - ex) < tol
    if not mask_x.any(): return []
    mask = mask_x & (np.abs(arr[2:] - ez) < tol) & np.isfinite(arr[:-2]) & np.isfinite(arr[2:])
    idx = np.nonzero(mask)[0]
    return [base_addr + int(i) * 4 for i in idx]

_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
def _post(o):
    req = urllib.request.Request("https://127.0.0.1:2999/replay/playback",
            data=json.dumps(o).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return json.loads(r.read())
def _get():
    with urllib.request.urlopen("https://127.0.0.1:2999/replay/playback", context=_ctx, timeout=2) as r:
        return json.loads(r.read())

def wait_seek_done(timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(0.3)
        st = _get()
        if not st["seeking"]: return st
    return _get()

def main():
    pid = find_pid(); h = _k.OpenProcess(0x0410, False, pid)
    print(f"PID={pid}")

    # Deep-seek to reset heap, then seek to 34
    print("Deep-seek to gt=200 (heap reset)...")
    _post({"time": 200.0, "speed": 1.0, "paused": True})
    wait_seek_done()
    time.sleep(1.0)
    print("Seek to gt=34...")
    _post({"time": 34.0, "speed": 1.0, "paused": True})
    wait_seek_done()

    # Play forward to PROBE_GT, then pause
    print(f"Playing forward 1.0x to gt={PROBE_GT}...")
    _post({"speed": 1.0, "paused": False})
    while True:
        time.sleep(0.1)
        st = _get()
        if st["time"] >= PROBE_GT: break
    _post({"speed": 1.0, "paused": True})
    time.sleep(0.4)
    st = _get()
    print(f"Paused at gt={st['time']:.2f}")

    # One scan pass, collect hits for each target
    regions = list(enum_rw(h))
    total = sum(s for _, s in regions)
    print(f"\nScanning {len(regions)} RW private regions ({total/(1024*1024):.0f} MB) for {len(TARGETS)} targets...")
    results = {name: [] for name, _, _ in TARGETS}
    t0 = time.time()
    scanned = 0
    for base, size in regions:
        data = read_region(h, base, size)
        if not data: continue
        scanned += len(data)
        for name, ex, ez in TARGETS:
            for a in scan_for_vec3(data, base, ex, ez, TOL):
                results[name].append(a)
    print(f"Scanned {scanned/(1024*1024):.0f} MB in {time.time()-t0:.1f}s")

    # Analyze: for each target, group addresses by their enclosing allocation and look for the 3-mirror stride
    def virtual_query(addr):
        mbi = MBI()
        _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi))
        return (mbi.AllocationBase or 0), (mbi.BaseAddress or 0), mbi.RegionSize

    print()
    for name, ex, ez in TARGETS:
        hits = sorted(results[name])
        print(f"--- {name} dest=({ex},?,{ez}): {len(hits)} raw hits ---")
        # Group by allocation base
        by_alloc = {}
        for a in hits:
            alloc, _, _ = virtual_query(a)
            by_alloc.setdefault(alloc, []).append(a)
        for alloc, addrs in sorted(by_alloc.items()):
            # compute pairwise deltas
            deltas = []
            for i in range(1, len(addrs)):
                deltas.append(addrs[i] - addrs[0])
            stride_match = (len(addrs) >= 3 and deltas[0] == 0x308 and deltas[1] == 0x374)
            tag = " <<< 3-MIRROR-STRIDE" if stride_match else ""
            addrs_s = ", ".join(f"0x{a:X}" for a in addrs)
            print(f"  alloc=0x{alloc:X}  hits={len(addrs)}  [{addrs_s}]{tag}")

if __name__ == "__main__":
    main()
