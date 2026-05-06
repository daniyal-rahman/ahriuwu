"""Identify Bel'Veth's click-destination allocation by intersecting
process-wide Vec3 scans across multiple known click windows.

Rationale: at each click time t, the click-dest alloc for the clicking unit
holds that click's Vec3. We pause the replay at t+delta and scan heap for
that Vec3. Only the alloc belonging to Bel'Veth will match ALL click
windows.

Click set for NA1_5545727197 (Bel'Veth, first 60s):
  gt=18.00  dest ≈ (2821, 7081)   [inferred from trajectory]
  gt=27.90  dest ≈ (3852, 7878)   [inferred from trajectory]
  gt=33.90  dest ≈ (3772, 8502)   [inferred from trajectory]
  gt=38.68  dest =  (3124, 8122)  [keylog ground truth]
  gt=43.50  dest =  (3736, 8358)  [keylog ground truth]
  gt=47.86  dest =  (4398, 8444)  [keylog ground truth]

Writes C:\\tmp\\belveth_click_alloc.json with the intersection.
"""
import ctypes, ctypes.wintypes as wt
import sys, subprocess, ssl, time, json, urllib.request
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

CLICKS = [
    # (label, pause_gt, expected_x, expected_z, tolerance_units)
    # Pause mid-walk for each click so the alloc holds the destination
    # while Bel'Veth is still moving toward it (sanity-check: not stationary).
    # Walks: 18.0 -> arr~27.9; 27.9 -> arr~33.9; 33.9 -> arr~38.68;
    #        38.68 -> arr~43.5; 43.5 -> arr~47.86; 47.86 -> arr~51
    ("w18",  22.0,  2821.0, 7081.0, 60.0),  # inferred, wider tol
    ("w28",  30.5,  3852.0, 7878.0, 60.0),  # inferred
    ("w34",  36.0,  3772.0, 8502.0, 60.0),  # inferred
    ("w39",  41.0,  3124.0, 8122.0, 10.0),  # keylog
    ("w44",  45.5,  3736.0, 8358.0, 10.0),  # keylog
    ("w48",  49.5,  4398.0, 8444.0, 10.0),  # keylog
]

_k = ctypes.windll.kernel32
_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE

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
        b = mbi.BaseAddress or 0; s = mbi.RegionSize
        if (mbi.State == MEM_COMMIT and mbi.Type == MEM_PRIVATE and
            (mbi.Protect & PAGE_RW)):
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

def scan_vec3_in_region(data, base, ex, ez, tol):
    arr = np.frombuffer(data, dtype=np.float32)
    if len(arr) < 3: return []
    mx = np.abs(arr[:-2] - ex) < tol
    if not mx.any(): return []
    mxz = mx & (np.abs(arr[2:] - ez) < tol)
    mxz &= np.isfinite(arr[:-2]) & np.isfinite(arr[2:])
    return [base + int(i) * 4 for i in np.nonzero(mxz)[0]]

def api_post(ep, body):
    req = urllib.request.Request(f"https://127.0.0.1:2999{ep}",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return json.loads(r.read())
def api_get(ep):
    with urllib.request.urlopen(f"https://127.0.0.1:2999{ep}", context=_ctx, timeout=3) as r:
        return json.loads(r.read())

def wait_until(gt, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = api_get("/replay/playback")
        if st["time"] >= gt: return st
        time.sleep(0.1)
    return api_get("/replay/playback")

def play_and_pause_at(gt, speed=1.0):
    api_post("/replay/playback", {"speed": speed, "paused": False})
    wait_until(gt)
    api_post("/replay/playback", {"speed": speed, "paused": True})
    time.sleep(0.4)
    return api_get("/replay/playback")

def scan_all(h, ex, ez, tol):
    hits = []
    nb = 0
    for base, size in rw_regions(h):
        data = read_region(h, base, size)
        if not data: continue
        nb += len(data)
        hits.extend(scan_vec3_in_region(data, base, ex, ez, tol))
    return hits, nb

def main():
    pid = find_pid()
    if not pid:
        print("ERR: League not running"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    print(f"pid={pid}  windows={len(CLICKS)}")

    # Seek to before first window
    print(f"\nSeeking to gt=12 (before first click)")
    api_post("/replay/playback", {"time": 12.0, "speed": 1.0, "paused": True})
    for _ in range(30):
        time.sleep(0.3)
        st = api_get("/replay/playback")
        if not st["seeking"] and st["paused"]: break

    window_hits = []
    for label, gt, ex, ez, tol in CLICKS:
        print(f"\n--- {label}: play -> gt={gt}, scan for ({ex},?,{ez}) tol={tol} ---")
        # Get cam pos RIGHT BEFORE pausing (measures motion while playing)
        try:
            cp0 = api_get("/replay/render").get("cameraPosition", {})
            t0_before = api_get("/replay/playback")["time"]
        except Exception:
            cp0, t0_before = {}, gt
        st = play_and_pause_at(gt)
        try:
            cp1 = api_get("/replay/render").get("cameraPosition", {})
            t1 = api_get("/replay/playback")["time"]
            if cp0 and cp1 and t1 > t0_before:
                dx = (cp1.get("x",0)) - (cp0.get("x",0))
                dz = (cp1.get("z",0)) - (cp0.get("z",0))
                speed = (dx*dx + dz*dz)**0.5 / (t1 - t0_before)
                print(f"    paused at gt={st['time']:.2f}  cam_speed={speed:.0f} u/s "
                      f"({'MOVING' if speed > 100 else 'STATIONARY!'})")
            else:
                print(f"    paused at gt={st['time']:.2f}")
        except Exception:
            print(f"    paused at gt={st['time']:.2f}")
        t0 = time.time()
        hits, nb = scan_all(h, ex, ez, tol)
        print(f"    {len(hits)} hits in {nb/(1024*1024):.0f} MB in {time.time()-t0:.1f}s")
        window_hits.append({"label": label, "gt": gt, "ex": ex, "ez": ez,
                            "tol": tol, "hits": [hex(a) for a in hits]})

    # Intersection: for each hit in window 1, check nearby addresses in
    # subsequent windows (allow slight address drift; since allocation itself
    # is stable, should be EXACT match).
    sets = [set(int(a, 16) for a in w["hits"]) for w in window_hits]
    common = set.intersection(*sets) if all(sets) else set()
    print(f"\n=== INTERSECTION ===")
    print(f"addresses matching ALL {len(CLICKS)} windows: {len(common)}")
    for a in sorted(common)[:20]:
        print(f"  0x{a:X}")

    # Also compute pairwise intersections for diagnostics
    print("\nPairwise intersections:")
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            n = len(sets[i] & sets[j])
            print(f"  {window_hits[i]['label']} ∩ {window_hits[j]['label']}: {n}")

    # Rank candidates: addresses matching the most windows
    from collections import Counter
    cnt = Counter()
    for s in sets:
        for a in s: cnt[a] += 1
    print(f"\nTop candidates by # windows matched:")
    for a, n in cnt.most_common(15):
        print(f"  0x{a:X}  matched {n}/{len(CLICKS)} windows")

    out = {
        "pid": pid,
        "windows": window_hits,
        "common_all": [hex(a) for a in common],
        "top_candidates": [{"addr": hex(a), "match_count": n}
                           for a, n in cnt.most_common(30)],
    }
    with open(r"C:\tmp\belveth_click_alloc.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote C:\\tmp\\belveth_click_alloc.json")

    # Pick the best candidate (most matches, and the mirror-base if it's
    # part of a triple-mirror trio).
    if not cnt:
        return 2
    top_addrs = [a for a, _ in cnt.most_common(20)]
    # Look for a triple-mirror set: base, base+0x308, base+0x374 all in top
    winner = None
    for a in top_addrs:
        if (a + 0x308) in set(top_addrs) and (a + 0x374) in set(top_addrs):
            winner = a; break
    if winner is None:
        winner = top_addrs[0]
    print(f"\n=== Best candidate: 0x{winner:X} ===")

    # Phase 2: continue forward playback without seeking, poll the alloc,
    # record click changes and Bel'Veth position simultaneously so we can
    # validate arrival-at-destination.
    print(f"\nContinuing forward playback, polling 0x{winner:X} ...")
    events = []
    samples = []
    prev = (0.0, 0.0, 0.0)
    api_post("/replay/playback", {"speed": 2.0, "paused": False})
    end_gt = 100.0
    t0_wall = time.time()
    last_poll = 0
    while True:
        now = time.time()
        if now - last_poll < 0.03:
            time.sleep(0.005); continue
        last_poll = now
        try:
            st = api_get("/replay/playback")
            cp = api_get("/replay/render").get("cameraPosition", {})
        except Exception:
            continue
        gt = st["time"]
        if gt >= end_gt or time.time() - t0_wall > 120: break
        buf = (ctypes.c_char * 12)(); rr = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(h, ctypes.c_void_p(winner), buf, 12, ctypes.byref(rr))
        if not ok or rr.value != 12: continue
        import struct
        v = struct.unpack("<fff", bytes(buf))
        samples.append({
            "gt": round(gt, 3), "x": v[0], "y": v[1], "z": v[2],
            "cam_x": cp.get("x", 0), "cam_z": cp.get("z", 0),
        })
        dx = v[0] - prev[0]; dz = v[2] - prev[2]
        if (dx * dx + dz * dz) ** 0.5 > 100.0:
            events.append({"gt": round(gt, 3), "x": v[0], "y": v[1], "z": v[2],
                           "cam_x": cp.get("x", 0), "cam_z": cp.get("z", 0)})
            print(f"  gt={gt:.2f}  click -> ({v[0]:.0f},{v[2]:.0f})  "
                  f"cam @ ({cp.get('x',0):.0f},{cp.get('z',0):.0f})")
        prev = v
    api_post("/replay/playback", {"paused": True})
    print(f"\n{len(events)} click changes over gt=[{samples[0]['gt']:.1f},"
          f"{samples[-1]['gt']:.1f}]")
    out["winner"] = hex(winner)
    out["forward_events"] = events
    out["forward_samples"] = samples
    with open(r"C:\tmp\belveth_click_alloc.json", "w") as f:
        json.dump(out, f, indent=2)
    return 0 if events else 2

if __name__ == "__main__":
    sys.exit(main())
