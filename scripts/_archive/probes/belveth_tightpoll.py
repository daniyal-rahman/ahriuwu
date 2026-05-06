"""Single-shot: identify Bel'Veth's click-dest alloc, then tight-poll it at
~10Hz for the entire replay, capturing every click.

Phases (all in one continuous run, no seeks):
  P1 (first 90s wall): scan + identify → top 5 candidate addrs
  P2 (rest of replay): every POLL_MS poll those 5 addrs' Vec3s; detect
      changes > DELTA_UNITS; attribute to the address whose Vec3 sequence
      best correlates with hero position (re-evaluated every REVERIFY_S).
      If all 5 go invalid, do a fresh P1-style identify.

Prereqs: replay started, playing from t≥30, 2x speed OK.
Output: C:\\tmp\\belveth_tightpoll_clicks.json
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

CHAMPION = b"Belveth"
HERO_POS_OFF = 0x200
POS_POLL_S = 0.5       # hero position sample cadence
POLL_MS = 100          # tight-poll cadence for watched addrs (10Hz wall → 5Hz game at 2x)
SCAN_INTERVAL_S = 4.0
P1_DURATION_S = 90     # identify phase duration
REVERIFY_S = 180       # every 3 min redo identify to keep watched-set current
DELTA_UNITS = 50.0     # min Vec3 change to count as a click
RUN_DURATION_S = 900   # safety cap

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

def valid_vec(v):
    return v and 100 < v[0] < 15000 and 45 < v[1] < 65 and 100 < v[2] < 15000

def identify_top_k(h, hero_ptr, duration_s, k=5):
    """Scan + track for duration_s, return top-k candidates by trajectory match."""
    hero_hist = []
    cand_hist = defaultdict(list)
    t0 = time.time()
    next_scan = t0
    last_pos_t = 0
    print(f"  [identify] running {duration_s}s...")
    while time.time() - t0 < duration_s:
        now = time.time()
        t_rel = now - t0
        pb = api_get("/replay/playback"); gt = pb.get("time", 0) if pb else 0
        if now - last_pos_t >= 1.0:
            pos = read_vec3(h, hero_ptr + HERO_POS_OFF)
            if valid_vec(pos):
                hero_hist.append((t_rel, gt, pos))
            last_pos_t = now
        if now >= next_scan:
            cands = scan_triple_mirrors(h)
            for a, v in cands:
                cand_hist[a].append((t_rel, gt, v))
            next_scan = now + SCAN_INTERVAL_S
        time.sleep(0.2)

    # Score
    scores = []
    for addr, hist in cand_hist.items():
        if len(hist) < 3 or len(set(v for _, _, v in hist)) < 2: continue
        dists = []
        simul = 0
        for t_c, gt_c, v_c in hist:
            future = [p for t, gt, p in hero_hist if gt_c <= gt <= gt_c + 15.0]
            if future:
                dists.append(min(((p[0]-v_c[0])**2 + (p[2]-v_c[2])**2)**0.5 for p in future))
            same = [p for t, gt, p in hero_hist if abs(gt - gt_c) < 1.0]
            if same and ((same[0][0]-v_c[0])**2 + (same[0][2]-v_c[2])**2)**0.5 < 50:
                simul += 1
        if not dists: continue
        avg = sum(dists) / len(dists)
        sr = simul / len(hist)
        if sr < 0.4 and avg < 1200:
            scores.append((addr, avg, sr, len(hist)))
    scores.sort(key=lambda r: r[1])
    return scores[:k], hero_hist

def main():
    pid = find_pid()
    if not pid: print("ERR"); return 1
    h = _k.OpenProcess(0x0410, False, pid)
    hero_ptr = find_belveth_hero(h)
    if not hero_ptr: print("ERR no Belveth"); return 1
    print(f"pid={pid}  belveth_hero=0x{hero_ptr:X}")

    # -------- P1: initial identify --------
    top, hero_hist = identify_top_k(h, hero_ptr, P1_DURATION_S, k=5)
    if not top:
        print("ERR no candidates from identify"); return 1
    print(f"\n  top-5 after P1:")
    for a, avg, sr, n in top:
        print(f"    0x{a:X}  avg_d={avg:.0f}  simul={sr:.2f}  n={n}")
    watched = [a for a, *_ in top]
    prev_vec = {a: read_vec3(h, a) for a in watched}

    # -------- P2: tight-poll --------
    all_clicks = []     # list of dicts
    t_loop0 = time.time()
    next_reverify = t_loop0 + REVERIFY_S
    last_print = t_loop0
    last_gt = 0
    total_len = api_get("/replay/playback").get("length", 1554)
    print(f"\n== P2: tight-poll 5 addrs at {1000//POLL_MS}Hz ==")

    while True:
        now = time.time()
        if now - t_loop0 > RUN_DURATION_S: break
        pb = api_get("/replay/playback")
        gt = pb.get("time", last_gt) if pb else last_gt
        last_gt = gt
        if gt >= total_len - 1: print("reached end of replay"); break

        hero_pos = read_vec3(h, hero_ptr + HERO_POS_OFF)

        for addr in watched:
            v = read_vec3(h, addr)
            if not valid_vec(v):
                continue
            prev = prev_vec.get(addr)
            if prev:
                dx = v[0] - prev[0]; dz = v[2] - prev[2]
                d = (dx*dx + dz*dz) ** 0.5
                if d > DELTA_UNITS:
                    # Record click; attribute candidate-addr so later we can filter
                    all_clicks.append({
                        "game_t": gt, "addr": hex(addr),
                        "x": v[0], "y": v[1], "z": v[2], "delta": round(d, 1),
                        "hero_x": hero_pos[0] if hero_pos else None,
                        "hero_z": hero_pos[2] if hero_pos else None,
                    })
            prev_vec[addr] = v

        # Periodic re-verify
        if now >= next_reverify:
            # Check all watched valid; if any died, full reidentify
            alive = [a for a in watched if valid_vec(read_vec3(h, a))]
            if len(alive) < max(1, len(watched) // 2):
                print(f"\n  reverify at gt={gt:.0f}: {len(alive)}/{len(watched)} alive, redoing identify")
                top, hero_hist = identify_top_k(h, hero_ptr, 60, k=5)
                if top:
                    watched = [a for a, *_ in top]
                    prev_vec = {a: read_vec3(h, a) for a in watched}
                    print(f"  new watched: {[hex(a) for a in watched]}")
            else:
                print(f"  reverify at gt={gt:.0f}: {len(alive)}/{len(watched)} alive OK")
            next_reverify = now + REVERIFY_S

        if now - last_print > 30.0:
            print(f"  wall={now-t_loop0:6.1f}s  gt={gt:7.1f}s  clicks_so_far={len(all_clicks)}")
            last_print = now
        time.sleep(POLL_MS / 1000.0)

    out = {
        "total_clicks": len(all_clicks),
        "watched_addrs": [hex(a) for a in watched],
        "clicks": all_clicks,
    }
    with open(r"C:\tmp\belveth_tightpoll_clicks.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote C:\\tmp\\belveth_tightpoll_clicks.json ({len(all_clicks)} clicks)")
    return 0

if __name__ == "__main__":
    sys.exit(main())
