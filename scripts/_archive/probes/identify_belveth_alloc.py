"""Identify Bel'Veth's click-dest alloc by correlating candidate trajectories
against her actual position during replay playback.

Setup (done by this script):
  - Seeks replay to START_GT (default 30s), unpauses at 1x
  - Sets selection to Bel'Veth via /replay/render (no keyboard lock needed)
  - Bel'Veth's live world position = cameraPosition (x, z+CAM_Z_OFFSET_INV)

How it works:
  1. Every ~1.0s: GET /replay/render → extract cameraPosition + correct offset
     = Bel'Veth world position sample
  2. Every ~4s: scan heap for triple-mirror + y-in-floor-range candidates
  3. Track each candidate's Vec3 history
  4. Score each candidate: how well its Vec3 sequence PREDICTS Bel'Veth's
     future position (destination leads arrival by 3-10s)
  5. Best-scoring = her alloc

Runs for DURATION_S seconds (default 90) then outputs winner.
Output: C:\\tmp\\belveth_alloc_id.json
"""
import ctypes, ctypes.wintypes as wt
import sys, subprocess, struct, json, time, ssl, urllib.request
from collections import defaultdict
import numpy as np

sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
import builtins
_orig_print = builtins.print
def print(*a, **k):
    k.setdefault("flush", True); _orig_print(*a, **k)
builtins.print = print

_k = ctypes.windll.kernel32
_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
REPLAY_HOST = "https://127.0.0.1:2999"
FOCUSED_HERO_RVA = 0x1E13490
HERO_POS_OFF = 0x200
# Cam-position → champion world-position correction (from pipeline.py)
CAM_Y_INV = 1912     # camera Y is constant; cam.z = world.z - 1292
CAM_Z_OFFSET_INV = 1292   # world.z = cam.z + 1292
START_GT = 30.0
DURATION_S = 90
CHAMPION = "Bel'Veth"

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

def api_get(path):
    try:
        with urllib.request.urlopen(f"{REPLAY_HOST}{path}", context=_ctx, timeout=2) as r:
            return json.loads(r.read())
    except Exception as e:
        return None

def api_post(path, body):
    try:
        req = urllib.request.Request(f"{REPLAY_HOST}{path}",
            data=json.dumps(body).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, context=_ctx, timeout=2) as r:
            return r.read()
    except Exception as e:
        return str(e).encode()

_belveth_hero_ptr = None
def find_belveth_hero(h):
    """Scan heap for 'Belveth\0' substrings; any such addr MAY be at hero+0x4360.
    Test by reading hero+0x200 as Vec3; if it's a plausible world pos, found it."""
    global _belveth_hero_ptr
    if _belveth_hero_ptr:
        return _belveth_hero_ptr
    for base, size in readable_regions(h):
        data = read_region(h, base, size)
        if not data: continue
        needle = b"Belveth\x00"
        off = 0
        while True:
            j = data.find(needle, off)
            if j == -1: break
            # candidate hero_base = (base + j) - 0x4360
            cand = base + j - 0x4360
            pos = read_vec3(h, cand + HERO_POS_OFF)
            if pos and 100 < pos[0] < 15000 and 100 < pos[2] < 15000 and 45 < pos[1] < 65:
                # verify champion_name still == Belveth (not a copy)
                nm = read_bytes(h, cand + 0x4360, 16)
                if nm and nm.split(b"\x00")[0] == b"Belveth":
                    _belveth_hero_ptr = cand
                    return cand
            off = j + 1
    return None

def get_belveth_pos(h):
    """Live Bel'Veth world position from her hero struct."""
    ptr = find_belveth_hero(h)
    if not ptr: return None
    return read_vec3(h, ptr + HERO_POS_OFF)

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])
    return None

def module_base(pid, h):
    psapi = ctypes.WinDLL("psapi.dll")
    HMODULE = wt.HMODULE
    psapi.EnumProcessModulesEx.argtypes = [wt.HANDLE, ctypes.POINTER(HMODULE), wt.DWORD, ctypes.POINTER(wt.DWORD), wt.DWORD]
    psapi.GetModuleFileNameExW.argtypes = [wt.HANDLE, HMODULE, wt.LPWSTR, wt.DWORD]
    class MI(ctypes.Structure):
        _fields_ = [("lpBaseOfDll", ctypes.c_void_p), ("SizeOfImage", wt.DWORD), ("EntryPoint", ctypes.c_void_p)]
    psapi.GetModuleInformation.argtypes = [wt.HANDLE, HMODULE, ctypes.POINTER(MI), wt.DWORD]
    mods = (HMODULE * 1024)(); needed = wt.DWORD(0)
    psapi.EnumProcessModulesEx(h, mods, ctypes.sizeof(mods), ctypes.byref(needed), 3)
    for i in range(needed.value // ctypes.sizeof(HMODULE)):
        name = ctypes.create_unicode_buffer(260)
        psapi.GetModuleFileNameExW(h, mods[i], name, 260)
        if name.value.lower().endswith("league of legends.exe"):
            mi = MI(); psapi.GetModuleInformation(h, mods[i], ctypes.byref(mi), ctypes.sizeof(mi))
            return mi.lpBaseOfDll
    return None

def read_bytes(h, addr, n):
    buf = (ctypes.c_char * n)(); r = ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, n, ctypes.byref(r))
    return bytes(buf[:r.value]) if ok else None

def read_u64(h, addr):
    b = read_bytes(h, addr, 8); return struct.unpack("<Q", b)[0] if b and len(b)==8 else None

def read_vec3(h, addr):
    b = read_bytes(h, addr, 12); return struct.unpack("<fff", b) if b and len(b)==12 else None

def readable_regions(h):
    addr = 0; mbi = MBI()
    while addr < 0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
            break
        b = mbi.BaseAddress or 0; s = mbi.RegionSize
        if (mbi.State == MEM_COMMIT and mbi.Type == MEM_PRIVATE
                and (mbi.Protect & PAGE_RW) and s <= 32 * 1024 * 1024):
            yield b, s
        addr = b + s
        if addr <= b: break

def read_region(h, base, size):
    if size > 32 * 1024 * 1024: return None
    out = bytearray(size); v = memoryview(out); o = 0; CH = 4 * 1024 * 1024
    while o < size:
        n = min(CH, size - o)
        buf = (ctypes.c_char * n)(); r = ctypes.c_size_t(0)
        if not _k.ReadProcessMemory(h, ctypes.c_void_p(base + o), buf, n, ctypes.byref(r)) or r.value == 0:
            return None if o == 0 else bytes(v[:o])
        v[o:o+r.value] = buf[:r.value]; o += r.value
    return bytes(out)

def scan_triple_mirrors(h):
    candidates = []
    SB = 0x308 // 4; SC = 0x374 // 4
    for base, size in readable_regions(h):
        data = read_region(h, base, size)
        if not data or len(data) < 0x400: continue
        n_f = len(data) // 4
        if n_f <= SC + 3: continue
        arr = np.frombuffer(data, dtype=np.float32, count=n_f)
        L = n_f - SC - 3
        if L <= 0: continue
        x0 = arr[0:L]; y0 = arr[1:L+1]; z0 = arr[2:L+2]
        xb = arr[SB:SB+L]; yb = arr[SB+1:SB+1+L]; zb = arr[SB+2:SB+2+L]
        xc = arr[SC:SC+L]; yc = arr[SC+1:SC+1+L]; zc = arr[SC+2:SC+2+L]
        with np.errstate(invalid='ignore', over='ignore'):
            mask = ((x0 > 100) & (x0 < 15000) & (z0 > 100) & (z0 < 15000)
                    & (y0 > 45) & (y0 < 65)
                    & (np.abs(x0 - xb) < 1e-2) & (np.abs(y0 - yb) < 1e-2) & (np.abs(z0 - zb) < 1e-2)
                    & (np.abs(x0 - xc) < 1e-2) & (np.abs(z0 - zc) < 1e-2))
        for i in np.nonzero(mask)[0]:
            candidates.append((base + int(i) * 4, (float(x0[i]), float(y0[i]), float(z0[i]))))
    return candidates

def main():
    pid = find_pid()
    if not pid: print("ERR no league"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    mod = module_base(pid, h)
    print(f"pid={pid} module=0x{mod:X}")

    # Set up replay: seek to START_GT, select Bel'Veth, unpause 1x
    api_post("/replay/playback", {"time": START_GT, "paused": True, "speed": 1.0})
    time.sleep(1.5)
    api_post("/replay/render", {"selectionName": CHAMPION, "interfaceAll": True})
    time.sleep(0.5)
    api_post("/replay/playback", {"paused": False, "speed": 1.0})
    time.sleep(1.0)
    # Confirm selection + initial position
    print("searching for Bel'Veth hero struct...")
    pos0 = get_belveth_pos(h)
    print(f"initial Bel'Veth pos = {pos0}  (hero_ptr = {hex(_belveth_hero_ptr) if _belveth_hero_ptr else None})")
    if not pos0:
        print("ERR couldn't find Bel'Veth hero struct"); return 1

    # Track histories: addr -> [(t, vec3)]
    cand_hist = defaultdict(list)
    hero_hist = []  # [(t, vec3)]

    t_start = time.time()
    next_scan = t_start
    last_pos_t = 0
    print(f"running {DURATION_S}s...")
    while time.time() - t_start < DURATION_S:
        now = time.time()
        t_rel = now - t_start
        # Poll hero position every 1.0s via replay API
        if now - last_pos_t >= 1.0:
            pos = get_belveth_pos(h)
            if pos and 100 < pos[0] < 15000 and 100 < pos[2] < 15000:
                hero_hist.append((t_rel, pos))
            last_pos_t = now
        # Scan for candidates every ~4s
        if now >= next_scan:
            cands = scan_triple_mirrors(h)
            # For each candidate in this scan, record its Vec3 at t_rel
            for addr, v in cands:
                cand_hist[addr].append((t_rel, v))
            print(f"  t={t_rel:5.1f}s  hero_samples={len(hero_hist)}  cands_this_scan={len(cands)}  tracked={len(cand_hist)}")
            next_scan = now + 4.0
        else:
            time.sleep(0.25)

    # ---- Analyze ----
    print(f"\n== Analysis: {len(hero_hist)} hero samples, {len(cand_hist)} candidates tracked ==")
    # Keep candidates seen in >=5 scans with distinct Vec3s
    useful = {a: h for a, h in cand_hist.items()
              if len(h) >= 5 and len(set(v for _, v in h)) >= 3}
    print(f"  candidates with >=5 samples and >=3 distinct Vec3s: {len(useful)}")

    # For each useful candidate, compute "prediction score":
    # For each candidate Vec3 sample (t_c, v_c), find the hero sample at t_h >= t_c
    # whose position is closest to v_c. Good click-dest candidates have hero ARRIVING
    # AT v_c within LOOKAHEAD seconds. Score = average min-distance, lower = better.
    LOOKAHEAD = 15.0
    scores = []
    for addr, hist in useful.items():
        distances = []
        for t_c, v_c in hist:
            # hero samples in [t_c, t_c + LOOKAHEAD]
            future = [(t, p) for t, p in hero_hist if t_c <= t <= t_c + LOOKAHEAD]
            if not future: continue
            # min distance in xz plane between hero and v_c during lookahead
            min_d = min(((p[0]-v_c[0])**2 + (p[2]-v_c[2])**2)**0.5 for _, p in future)
            distances.append(min_d)
        if len(distances) < 3: continue
        # Also compute: how often does the candidate match current hero pos within 50 units?
        # (if candidate equals current hero pos, it's NOT click-dest — it's hero pos itself)
        simultaneous = 0
        for t_c, v_c in hist:
            same_t = [p for t, p in hero_hist if abs(t - t_c) < 1.0]
            if same_t:
                pos = same_t[0]
                if ((pos[0]-v_c[0])**2 + (pos[2]-v_c[2])**2)**0.5 < 50:
                    simultaneous += 1
        avg = sum(distances) / len(distances)
        scores.append((addr, avg, simultaneous, len(distances), len(set(v for _, v in hist))))

    # Sort by avg distance (ascending = better)
    scores.sort(key=lambda r: r[1])
    print(f"\n== Top 10 candidates by trajectory-match score (lower = click-dest leads arrival) ==")
    print(f"{'addr':<20} {'avg_d':<8} {'simult':<8} {'n_obs':<6} {'n_vecs':<6}")
    for addr, avg, simul, n, nv in scores[:10]:
        print(f"  0x{addr:<18X} {avg:<8.0f} {simul:<8} {n:<6} {nv:<6}")

    # Winner heuristic: lowest avg_d AND simultaneous ratio < 50%
    winners = [(a, s, sim, n, nv) for (a, s, sim, n, nv) in scores
               if sim / max(n, 1) < 0.5 and s < 500]
    winner = winners[0] if winners else (scores[0] if scores else None)

    result = {
        "pid": pid, "module_base": hex(mod),
        "champion": CHAMPION, "duration_s": DURATION_S,
        "start_gt": START_GT,
        "n_hero_samples": len(hero_hist), "n_candidates_tracked": len(cand_hist),
        "top_candidates": [{"addr": hex(a), "avg_dist": s, "simultaneous": sim,
                            "n_obs": n, "n_distinct_vecs": nv}
                           for a, s, sim, n, nv in scores[:30]],
        "winner": {"addr": hex(winner[0]), "avg_dist": winner[1],
                   "simultaneous": winner[2], "n_obs": winner[3],
                   "n_distinct_vecs": winner[4]} if winner else None,
        "hero_trajectory": [{"t": t, "pos": [round(p, 1) for p in v]} for t, v in hero_hist],
    }
    with open(r"C:\tmp\belveth_alloc_id.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nwrote C:\\tmp\\belveth_alloc_id.json")
    if winner:
        print(f"  WINNER: 0x{winner[0]:X}  avg_dist={winner[1]:.0f}u  simult={winner[2]}/{winner[3]}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
