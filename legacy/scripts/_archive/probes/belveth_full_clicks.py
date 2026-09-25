"""Extract Bel'Veth's clicks for an entire replay via:
  1. Continuous tracking of her live hero-struct position.
  2. Continuous scan-all for triple-mirror candidates (every SCAN_INTERVAL_S).
  3. Per-window analysis: split the run into 60s windows, and for each window
     pick the candidate whose Vec3 sequence best predicts Bel'Veth's future
     position. Clicks in that window come from that window's winner.

Prereqs:
  - Replay is playing forward; seek to start BEFORE running this script.
  - DO NOT seek during the run.

Run plan:
  - Polls hero position every 1s
  - Scans heap every ~4s
  - Runs for up to RUN_DURATION_S real-time seconds
  - Saves progress to a pickle every 60s so we can recover from crashes

Outputs:
  C:\\tmp\\belveth_full_state.pkl    (raw state, for iteration)
  C:\\tmp\\belveth_full_clicks.json  (extracted per-window clicks)
"""
import ctypes, ctypes.wintypes as wt
import sys, subprocess, struct, json, time, ssl, urllib.request, pickle
from collections import defaultdict
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
import builtins
_orig_print = builtins.print
def print(*a, **k):
    k.setdefault("flush", True); _orig_print(*a, **k)
builtins.print = print

CHAMPION = b"Belveth"
HERO_POS_OFF = 0x200
POS_POLL_S = 1.0
SCAN_INTERVAL_S = 4.0
# RUN_DURATION_S governs wall-clock. At replay speed 2x, 11 min of wall → 22 min of game.
RUN_DURATION_S = 900  # 15 min wall → 30 min game at 2x
WINDOW_S = 60.0
CLICK_DELTA = 50.0

_k = ctypes.windll.kernel32
_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
REPLAY = "https://127.0.0.1:2999"
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

def api_get(p):
    try:
        with urllib.request.urlopen(f"{REPLAY}{p}", context=_ctx, timeout=2) as r:
            return json.loads(r.read())
    except: return None

def api_post(p, b):
    try:
        req = urllib.request.Request(f"{REPLAY}{p}",
            data=json.dumps(b).encode(), method="POST",
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, context=_ctx, timeout=2) as r: return r.read()
    except: return None

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])
    return None

def read_bytes(h, addr, n):
    buf = (ctypes.c_char * n)(); r = ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, n, ctypes.byref(r))
    return bytes(buf[:r.value]) if ok else None

def read_vec3(h, addr):
    b = read_bytes(h, addr, 12); return struct.unpack("<fff", b) if b and len(b)==12 else None

def readable_regions(h, max_size=32*1024*1024):
    addr = 0; mbi = MBI()
    while addr < 0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(h, ctypes.c_void_p(addr), ctypes.byref(mbi), ctypes.sizeof(mbi)):
            break
        b = mbi.BaseAddress or 0; s = mbi.RegionSize
        if (mbi.State == MEM_COMMIT and mbi.Type == MEM_PRIVATE
                and (mbi.Protect & PAGE_RW) and s <= max_size):
            yield b, s
        addr = b + s
        if addr <= b: break

def read_region(h, base, size):
    out = bytearray(size); v = memoryview(out); o = 0; CH = 4 * 1024 * 1024
    while o < size:
        n = min(CH, size - o)
        buf = (ctypes.c_char * n)(); r = ctypes.c_size_t(0)
        if not _k.ReadProcessMemory(h, ctypes.c_void_p(base + o), buf, n, ctypes.byref(r)) or r.value == 0:
            return None if o == 0 else bytes(v[:o])
        v[o:o+r.value] = buf[:r.value]; o += r.value
    return bytes(out)

def scan_triple_mirrors(h):
    cands = []
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
            cands.append((base + int(i) * 4, (float(x0[i]), float(y0[i]), float(z0[i]))))
    return cands

def find_belveth_hero(h):
    for base, size in readable_regions(h):
        data = read_region(h, base, size)
        if not data: continue
        off = 0
        while True:
            j = data.find(CHAMPION + b"\x00", off)
            if j == -1: break
            cand = base + j - 0x4360
            pos = read_vec3(h, cand + HERO_POS_OFF)
            if pos and 100 < pos[0] < 15000 and 100 < pos[2] < 15000 and 45 < pos[1] < 65:
                nm = read_bytes(h, cand + 0x4360, 16)
                if nm and nm.split(b"\x00")[0] == CHAMPION:
                    return cand
            off = j + 1
    return None

def main():
    pid = find_pid()
    if not pid: print("ERR"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    print(f"pid={pid}")
    hero_ptr = find_belveth_hero(h)
    if not hero_ptr: print("ERR no Belveth"); return 1
    print(f"Belveth hero struct at 0x{hero_ptr:X}")

    pb = api_get("/replay/playback")
    start_gt = pb.get("time", 0) if pb else 0
    total_len = pb.get("length", 1554) if pb else 1554
    print(f"replay start_gt={start_gt:.1f}s  total_len={total_len:.1f}s")

    # Histories
    hero_hist = []            # [(wall_t, game_t, (x,y,z))]
    cand_hist = defaultdict(list)  # addr -> [(wall_t, game_t, (x,y,z))]

    t_start = time.time()
    next_scan = t_start
    last_pos_t = 0
    last_save_t = t_start
    last_print_t = t_start

    def save_state():
        with open(r"C:\tmp\belveth_full_state.pkl", "wb") as f:
            pickle.dump({"hero_hist": hero_hist,
                         "cand_hist": dict(cand_hist),
                         "hero_ptr": hero_ptr, "start_gt": start_gt}, f)

    while time.time() - t_start < RUN_DURATION_S:
        now = time.time()
        # Read game_t from replay
        pb = api_get("/replay/playback")
        gt = pb.get("time", 0) if pb else 0
        if gt >= total_len - 1:
            print(f"  reached end of replay game_t={gt:.1f}"); break

        # Poll hero pos
        if now - last_pos_t >= POS_POLL_S:
            pos = read_vec3(h, hero_ptr + HERO_POS_OFF)
            if pos and 100 < pos[0] < 15000:
                hero_hist.append((now - t_start, gt, pos))
            last_pos_t = now

        # Scan candidates
        if now >= next_scan:
            cands = scan_triple_mirrors(h)
            for a, v in cands:
                cand_hist[a].append((now - t_start, gt, v))
            print(f"  wall={now-t_start:6.1f}s  game_t={gt:7.1f}s  hero_samples={len(hero_hist)}  scan_cands={len(cands)}  tracked={len(cand_hist)}")
            next_scan = now + SCAN_INTERVAL_S

        # Periodic state save
        if now - last_save_t >= 60.0:
            save_state()
            last_save_t = now

        time.sleep(0.2)

    save_state()
    print(f"\n== Run complete. {len(hero_hist)} hero samples, {len(cand_hist)} candidates tracked ==")

    # ---------- Offline analysis: windowed best-candidate + click extraction ----------
    print(f"\n== Windowed analysis (WINDOW_S={WINDOW_S}s) ==")
    if not hero_hist:
        print("no hero samples — abort"); return 1
    gt_min = hero_hist[0][1]
    gt_max = hero_hist[-1][1]
    windows = []
    t = gt_min
    while t < gt_max:
        windows.append((t, min(t + WINDOW_S, gt_max)))
        t += WINDOW_S

    # For each window: compute best candidate + clicks
    all_clicks = []
    for wi, (wa, wb) in enumerate(windows):
        hero_w = [(t, gt, p) for t, gt, p in hero_hist if wa <= gt < wb]
        if len(hero_w) < 3: continue
        cand_w = {}
        for addr, hist in cand_hist.items():
            h_in = [(t, gt, v) for t, gt, v in hist if wa <= gt < wb]
            if len(h_in) < 3: continue
            if len(set(v for _, _, v in h_in)) < 2: continue
            cand_w[addr] = h_in

        # Score each candidate: min dist from future hero position
        best = None; best_score = 1e9; best_simul = 0
        scores = []
        for addr, hist in cand_w.items():
            dists = []
            simul = 0
            for t_c, gt_c, v_c in hist:
                future_hero = [p for t, gt, p in hero_hist if gt_c <= gt <= gt_c + 15.0]
                if future_hero:
                    md = min(((p[0]-v_c[0])**2 + (p[2]-v_c[2])**2)**0.5 for p in future_hero)
                    dists.append(md)
                same_hero = [p for t, gt, p in hero_hist if abs(gt - gt_c) < 1.0]
                if same_hero and ((same_hero[0][0]-v_c[0])**2 + (same_hero[0][2]-v_c[2])**2)**0.5 < 50:
                    simul += 1
            if not dists: continue
            avg = sum(dists) / len(dists)
            simul_ratio = simul / len(hist)
            scores.append((addr, avg, simul_ratio, len(hist)))
            if simul_ratio < 0.4 and avg < best_score:
                best_score = avg; best = addr; best_simul = simul_ratio

        if not best:
            print(f"  window[{wi}] gt={wa:.0f}-{wb:.0f}  no winner")
            continue
        # Extract clicks from best candidate in this window
        hist = cand_w[best]
        w_clicks = [{"game_t": hist[0][1], "x": hist[0][2][0], "y": hist[0][2][1], "z": hist[0][2][2], "window": wi}]
        last_v = hist[0][2]
        for t, gt, v in hist[1:]:
            d = ((v[0]-last_v[0])**2 + (v[2]-last_v[2])**2)**0.5
            if d > CLICK_DELTA:
                w_clicks.append({"game_t": gt, "x": v[0], "y": v[1], "z": v[2], "window": wi, "delta": round(d, 1)})
                last_v = v
        print(f"  window[{wi}] gt={wa:.0f}-{wb:.0f}  winner=0x{best:X} avg_d={best_score:.0f} simul={best_simul:.2f}  clicks={len(w_clicks)}")
        all_clicks.extend(w_clicks)

    out = {
        "start_gt": start_gt,
        "n_hero_samples": len(hero_hist),
        "n_candidates": len(cand_hist),
        "n_windows": len(windows),
        "total_clicks": len(all_clicks),
        "clicks": all_clicks,
        "hero_trajectory": [{"game_t": gt, "pos": [round(p, 1) for p in v]}
                             for t, gt, v in hero_hist],
    }
    with open(r"C:\tmp\belveth_full_clicks.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote C:\\tmp\\belveth_full_clicks.json ({len(all_clicks)} clicks)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
