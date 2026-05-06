"""Locate the camera Vec3 (cx, cy, cz) in League's process memory.

Strategy:
  1. Poll /replay/render → ground-truth cam triple.
  2. Scan all RW private regions of League for 3 consecutive f32s matching.
  3. Move cam (let game advance), repeat scan, intersect candidates.
  4. Iterate until 1-3 addresses survive. Each scan eliminates ~99%.

Once located: read this address directly during recording (where /replay/render
goes stale). Bypass cam-thread polling entirely.

Usage:
    # Game must be running with a replay open and cam-locked to a champion.
    python scripts/find_cam_addr.py [--rounds 4] [--motion-wait 4]
"""
import os, sys, time, ctypes, json, argparse, subprocess, struct
import http.client, ssl, numpy as np
from ctypes import wintypes as wt

# ─── Win32 RPM plumbing (same as click_buffer_shape_scan.py) ───
_k = ctypes.WinDLL('kernel32', use_last_error=True)
PROCESS_VM_READ  = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400
class MBI(ctypes.Structure):
    _fields_ = [("BaseAddress", ctypes.c_void_p), ("AllocationBase", ctypes.c_void_p),
                ("AllocationProtect", wt.DWORD), ("__align", wt.DWORD),
                ("RegionSize", ctypes.c_size_t), ("State", wt.DWORD),
                ("Protect", wt.DWORD), ("Type", wt.DWORD)]
_k.OpenProcess.restype = wt.HANDLE
_k.OpenProcess.argtypes = [wt.DWORD, wt.BOOL, wt.DWORD]
_k.ReadProcessMemory.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)]
_k.ReadProcessMemory.restype = wt.BOOL
_k.VirtualQueryEx.argtypes = [wt.HANDLE, ctypes.c_void_p, ctypes.POINTER(MBI), ctypes.c_size_t]
_k.VirtualQueryEx.restype = ctypes.c_size_t
_k.CloseHandle.argtypes = [wt.HANDLE]
MEM_COMMIT = 0x1000; MEM_PRIVATE = 0x20000; MEM_IMAGE = 0x1000000
PAGE_R = 0x02 | 0x04 | 0x08 | 0x40 | 0x20 | 0x10  # any readable

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])
    return None

def enum_readable(h):
    """Yield (base, size) for all readable regions (private + image)."""
    addr = 0; mbi = MBI()
    while addr < 0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)): break
        base = mbi.BaseAddress or 0; size = mbi.RegionSize
        if mbi.State == MEM_COMMIT and (mbi.Protect & PAGE_R):
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
        if not ok or rd.value == 0:
            return None if off == 0 else bytes(view[:off])
        view[off:off+rd.value] = buf[:rd.value]
        off += rd.value
    return bytes(out)

def read_at(h, addr, n):
    buf = ctypes.create_string_buffer(n); rd = ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, n, ctypes.byref(rd))
    return buf.raw[:rd.value] if ok else b""

# ─── Replay API ───
_ctx = ssl.create_default_context(); _ctx.check_hostname = False; _ctx.verify_mode = ssl.CERT_NONE
def replay_get(ep):
    c = http.client.HTTPSConnection("127.0.0.1", 2999, context=_ctx, timeout=5)
    c.request("GET", ep); r = c.getresponse(); d = json.loads(r.read()); c.close(); return d
def replay_post(ep, data):
    c = http.client.HTTPSConnection("127.0.0.1", 2999, context=_ctx, timeout=5)
    body = json.dumps(data).encode()
    c.request("POST", ep, body=body, headers={"Content-Type":"application/json"})
    r = c.getresponse(); d = r.read(); c.close()
    try: return json.loads(d)
    except: return {}

def get_cam():
    cp = replay_get("/replay/render").get("cameraPosition", {})
    return float(cp.get("x", 0)), float(cp.get("y", 0)), float(cp.get("z", 0))

# ─── Float-triple scan ───
def find_triple(data, x, y, z, tol=0.01):
    """Find offsets in `data` where 3 consecutive f32s match (x,y,z) within tol."""
    arr = np.frombuffer(data, dtype=np.float32)
    # candidate offsets where arr[i] ≈ x AND arr[i+1] ≈ y AND arr[i+2] ≈ z
    with np.errstate(invalid="ignore"):
        mx = np.abs(arr - x) < tol
    if not mx.any(): return []
    idxs = np.where(mx)[0]
    out = []
    for i in idxs:
        if i + 2 >= len(arr): continue
        v1, v2 = arr[i+1], arr[i+2]
        if np.isnan(v1) or np.isnan(v2): continue
        if abs(v1 - y) < tol and abs(v2 - z) < tol:
            out.append(int(i) * 4)  # byte offset
    return out

def scan_for_cam(h, x, y, z, tol=0.01):
    """Scan all readable regions, return list of absolute addresses."""
    hits = []
    n_regions = 0
    n_bytes = 0
    for base, size in enum_readable(h):
        n_regions += 1
        if size < 12: continue
        data = read_region(h, base, size)
        if not data: continue
        n_bytes += len(data)
        for off in find_triple(data, x, y, z, tol):
            hits.append(base + off)
    print(f"  scanned {n_regions} regions, {n_bytes/1e9:.2f}GB -> {len(hits)} hits", flush=True)
    return hits

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=4,
                    help="number of scan rounds (each round narrows hits via intersection)")
    ap.add_argument("--motion-wait", type=float, default=4.0,
                    help="seconds between rounds (cam should move during this)")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="float-match tolerance (game state may settle slightly between API call and scan)")
    args = ap.parse_args()

    pid = find_pid()
    if not pid: print("League not running"); return 1
    print(f"PID = {pid}")
    h = _k.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
    if not h: print(f"OpenProcess failed: {ctypes.get_last_error()}"); return 1

    candidates = None
    snapshots = []   # list of (round_idx, x, y, z, hit_count)
    try:
        for r in range(args.rounds):
            # Pause so cam value is stable during the multi-second scan.
            replay_post("/replay/playback", {"paused": True})
            time.sleep(0.4)
            x, y, z = get_cam()
            print(f"\n--- round {r+1}/{args.rounds}: cam=({x:.1f}, {y:.1f}, {z:.1f}) (paused) ---", flush=True)
            t0 = time.time()
            hits = scan_for_cam(h, x, y, z, tol=args.tol)
            print(f"  scan took {time.time()-t0:.1f}s")
            snapshots.append((r, x, y, z, len(hits)))

            if candidates is None:
                candidates = set(hits)
            else:
                # Intersect: prior candidates whose CURRENT 12 bytes match (x,y,z)
                new_cands = set()
                for a in candidates:
                    b = read_at(h, a, 12)
                    if len(b) != 12: continue
                    cx, cy, cz = struct.unpack("<fff", b)
                    if abs(cx-x) < args.tol and abs(cy-y) < args.tol and abs(cz-z) < args.tol:
                        new_cands.add(a)
                candidates = new_cands
            print(f"  surviving candidates: {len(candidates)}")
            if len(candidates) <= 5:
                for a in sorted(candidates):
                    print(f"    0x{a:X}")

            if r < args.rounds - 1 and len(candidates) > 1:
                # Unpause and let cam move, then we'll re-pause next round.
                replay_post("/replay/playback", {"paused": False, "speed": 1.0})
                print(f"  unpaused, waiting {args.motion_wait}s for cam to move...")
                time.sleep(args.motion_wait)

        print(f"\n=== final ===")
        print(f"  candidates: {len(candidates)}")
        # Find League module base for RVA computation
        mod_base = None
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            import pipeline as P
            mod_base, _ = P.find_module_base(pid)
        except Exception as e:
            print(f"  (could not get module base: {e})")
        annotated = []
        for a in sorted(candidates):
            b = read_at(h, a, 12)
            cx, cy, cz = struct.unpack("<fff", b)
            tag = ""
            if mod_base:
                rva = a - mod_base
                if 0 <= rva < 0x10000000:
                    tag = f"  module+0x{rva:X}"
            print(f"    0x{a:X}{tag}  -> ({cx:.1f}, {cy:.1f}, {cz:.1f})")
            annotated.append({"addr": hex(a), "rva": (hex(a-mod_base) if mod_base else None),
                              "value": [cx, cy, cz]})
        x, y, z = get_cam()
        print(f"  api cam:  ({x:.1f}, {y:.1f}, {z:.1f})")
        if mod_base: print(f"  module base: 0x{mod_base:X}")

        out = {"pid": pid, "module_base": (hex(mod_base) if mod_base else None),
               "candidates": annotated, "snapshots": snapshots}
        with open(r"C:\tmp\cam_addr_candidates.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote C:\\tmp\\cam_addr_candidates.json")
    finally:
        _k.CloseHandle(h)
    return 0

if __name__ == "__main__":
    sys.exit(main())
