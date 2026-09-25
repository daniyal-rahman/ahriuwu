"""Drift instrumentation: measure where click-vs-camera misalignment comes from.

Designed to run on Windows alongside an active League replay. Locates Garen +
click-dest via the same recipes as garen_clicks_vtable.py, then samples
high-frequency for 60s capturing for each iteration:

  t0  -- iter start (perf_counter)
  gt_pre  -- /replay/playback time BEFORE other reads
  t1
  hero_pos, click_pos -- memory reads
  t2
  cam_pos -- /replay/render
  t3
  gt_post -- /replay/playback time AFTER other reads
  t4

Then dumps the raw log to JSON and prints a summary of:

  (a) HTTP round-trip latencies (playback + render)
  (b) game_t advance per iteration (= the granularity of click event timestamps)
  (c) wall->gt interpolation jitter (using mem timing)
  (d) hero->cam cross-correlation lag (camera smoothing)
  (e) click event apparent gt error vs true wall time of detected delta

Run on Windows in same session as a running replay (preferably 1x speed for
cleaner timing).
"""
import ctypes, ctypes.wintypes as wt
import sys, subprocess, struct, json, time, ssl, urllib.request, base64, os
import numpy as np
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

CHAMPION = b"Garen"
HERO_POS_OFF = 0x200
VTABLE_RVA = 0x192BF90
VEC3_OFFSET_FROM_VPTR = 0x14
SB = 0x308; SC = 0x374
PROBE_DURATION_S = 60.0
TARGET_HZ = 200

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
MEM_COMMIT=0x1000; MEM_PRIVATE=0x20000; PAGE_RW=0x04|0x08|0x40

import http.client
_pb_conn = None
_render_conn = None

def _persistent_get(conn_attr, path):
    """Reuse a persistent HTTPS connection for low-overhead polling."""
    global _pb_conn, _render_conn
    conn = globals()[conn_attr]
    for attempt in range(2):
        if conn is None:
            conn = http.client.HTTPSConnection("127.0.0.1", 2999, context=_ctx, timeout=2)
            globals()[conn_attr] = conn
        try:
            conn.request("GET", path)
            return json.loads(conn.getresponse().read())
        except (http.client.RemoteDisconnected, ConnectionResetError, OSError):
            try: conn.close()
            except: pass
            globals()[conn_attr] = None
            conn = None
    return None

def get_playback():
    d = _persistent_get('_pb_conn', '/replay/playback')
    return d.get("time") if d else None

def get_render():
    d = _persistent_get('_render_conn', '/replay/render')
    cp = (d or {}).get("cameraPosition") if d else None
    return (cp.get("x"), cp.get("y"), cp.get("z")) if cp else None

def find_pid():
    r=subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                     capture_output=True,text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])

def module_range(pid):
    psapi = ctypes.WinDLL("psapi.dll")
    h = _k.OpenProcess(0x0410, False, pid)
    HMODULE = wt.HMODULE
    psapi.EnumProcessModulesEx.argtypes = [wt.HANDLE, ctypes.POINTER(HMODULE), wt.DWORD, ctypes.POINTER(wt.DWORD), wt.DWORD]
    psapi.GetModuleFileNameExW.argtypes = [wt.HANDLE, HMODULE, wt.LPWSTR, wt.DWORD]
    class MINFO(ctypes.Structure):
        _fields_ = [("lpBaseOfDll", ctypes.c_void_p), ("SizeOfImage", wt.DWORD), ("EntryPoint", ctypes.c_void_p)]
    psapi.GetModuleInformation.argtypes = [wt.HANDLE, HMODULE, ctypes.POINTER(MINFO), wt.DWORD]
    mods = (HMODULE * 1024)(); needed = wt.DWORD(0)
    psapi.EnumProcessModulesEx(h, mods, ctypes.sizeof(mods), ctypes.byref(needed), 3)
    n = needed.value // ctypes.sizeof(HMODULE)
    for i in range(n):
        name = ctypes.create_unicode_buffer(260)
        psapi.GetModuleFileNameExW(h, mods[i], name, 260)
        if name.value.lower().endswith("league of legends.exe"):
            mi = MINFO()
            psapi.GetModuleInformation(h, mods[i], ctypes.byref(mi), ctypes.sizeof(mi))
            _k.CloseHandle(h)
            return mi.lpBaseOfDll, mi.SizeOfImage
    _k.CloseHandle(h); return None, None

def read_bytes(h, addr, n):
    buf=(ctypes.c_char*n)(); r=ctypes.c_size_t(0)
    ok = _k.ReadProcessMemory(h, ctypes.c_void_p(addr), buf, n, ctypes.byref(r))
    return bytes(buf[:r.value]) if ok else None

def read_vec3(h, addr):
    b=read_bytes(h, addr, 12)
    return struct.unpack("<fff", b) if b and len(b)==12 else None

def regions(h, max_size=64*1024*1024):
    addr=0; mbi=MBI()
    while addr<0x7FFFFFFFFFFF:
        if not _k.VirtualQueryEx(h,ctypes.c_void_p(addr),ctypes.byref(mbi),ctypes.sizeof(mbi)): break
        b=mbi.BaseAddress or 0; s=mbi.RegionSize
        if (mbi.State==MEM_COMMIT and mbi.Type==MEM_PRIVATE
                and (mbi.Protect&PAGE_RW) and s<=max_size):
            yield b,s
        addr=b+s
        if addr<=b: break

def read_region(h, base, size):
    out=bytearray(size); v=memoryview(out); o=0; CH=4*1024*1024
    while o<size:
        n=min(CH,size-o)
        buf=(ctypes.c_char*n)(); r=ctypes.c_size_t(0)
        if not _k.ReadProcessMemory(h,ctypes.c_void_p(base+o),buf,n,ctypes.byref(r)) or r.value==0:
            return None if o==0 else bytes(v[:o])
        v[o:o+r.value]=buf[:r.value]; o+=r.value
    return bytes(out)

def find_hero_by_name(h, name_bytes):
    needle = name_bytes + b"\x00"
    for base, size in regions(h):
        data = read_region(h, base, size)
        if not data: continue
        off = 0
        while True:
            j = data.find(needle, off)
            if j == -1: break
            cand = base + j - 0x4360
            pos = read_vec3(h, cand + HERO_POS_OFF)
            if pos and 100 < pos[0] < 15000 and 100 < pos[2] < 15000 and 45 < pos[1] < 65:
                nm = read_bytes(h, cand + 0x4360, len(name_bytes)+1)
                if nm and nm.split(b"\x00")[0] == name_bytes:
                    return cand
            off = j + 1
    return None

def vtable_scan(h, target_vptr):
    target_bytes = struct.pack("<Q", target_vptr)
    out = []
    for base, size in regions(h):
        data = read_region(h, base, size)
        if not data: continue
        n = len(data) // 8 * 8
        if n < 8: continue
        arr = np.frombuffer(data[:n], dtype=np.uint64)
        idxs = np.nonzero(arr == np.uint64(target_vptr))[0]
        for i in idxs:
            vptr_addr = base + int(i)*8
            vec3_addr = vptr_addr + VEC3_OFFSET_FROM_VPTR
            bv = read_bytes(h, vec3_addr, 12)
            if not bv or len(bv) < 12: continue
            x, y, z = struct.unpack("<fff", bv)
            if not (100 < x < 15000 and 100 < z < 15000 and 45 < y < 65): continue
            bb = read_bytes(h, vec3_addr + SB, 12)
            bc = read_bytes(h, vec3_addr + SC, 12)
            if not (bb and bc and len(bb)==12 and len(bc)==12): continue
            xb,yb,zb = struct.unpack("<fff", bb)
            xc,yc,zc = struct.unpack("<fff", bc)
            if (abs(x-xb)<0.01 and abs(y-yb)<0.01 and abs(z-zb)<0.01
                and abs(x-xc)<0.01 and abs(z-zc)<0.01):
                out.append((vptr_addr, vec3_addr, (x,y,z)))
    return out

def pick_click_addr(h, hero_ptr, target_vptr, secs=15):
    """Identify the real click-dest among vtable hits using avg-distance to hero."""
    from collections import defaultdict
    hero_hist = []
    cand_hist = defaultdict(list)
    t0 = time.time(); next_scan = t0
    while time.time() - t0 < secs:
        gt = get_playback() or 0
        pos = read_vec3(h, hero_ptr + HERO_POS_OFF)
        if pos: hero_hist.append((gt, pos))
        if time.time() >= next_scan:
            for vp, va, v in vtable_scan(h, target_vptr):
                cand_hist[va].append((gt, v))
            next_scan = time.time() + 4.0
        time.sleep(0.2)
    scores = []
    for addr, hist in cand_hist.items():
        if len(hist) < 3 or len(set(v for _,v in hist)) < 2: continue
        ds = []
        for gt_c, v_c in hist:
            future = [p for gt,p in hero_hist if gt_c <= gt <= gt_c + 15]
            if future: ds.append(min(((p[0]-v_c[0])**2+(p[2]-v_c[2])**2)**0.5 for p in future))
        if ds:
            scores.append((addr, sum(ds)/len(ds), len(hist)))
    scores.sort(key=lambda r: r[1])
    return scores[0][0] if scores else None

def main():
    pid = find_pid()
    if not pid: print("ERR: no League"); return 1
    base, _ = module_range(pid)
    target_vptr = base + VTABLE_RVA
    print(f"pid={pid}  module=0x{base:X}  target_vptr=0x{target_vptr:X}")
    h = _k.OpenProcess(0x0410, False, pid)
    print("finding Garen hero...")
    hero_ptr = None
    for attempt in range(15):
        hero_ptr = find_hero_by_name(h, CHAMPION)
        if hero_ptr: break
        time.sleep(2)
    if not hero_ptr: print("ERR: no Garen"); return 1
    print(f"  hero=0x{hero_ptr:X}")
    print("identifying click-dest addr (~15s)...")
    click_addr = pick_click_addr(h, hero_ptr, target_vptr, secs=15)
    if not click_addr: print("ERR: no click_addr"); return 1
    print(f"  click_addr=0x{click_addr:X}")

    print(f"\n=== probing for {PROBE_DURATION_S}s at target {TARGET_HZ}Hz ===")
    log = []
    t_start = time.perf_counter()
    target_dt = 1.0 / TARGET_HZ
    next_t = t_start
    while True:
        t0 = time.perf_counter()
        if t0 - t_start >= PROBE_DURATION_S: break
        gt_pre = get_playback()
        t1 = time.perf_counter()
        hero = read_vec3(h, hero_ptr + HERO_POS_OFF)
        click = read_vec3(h, click_addr)
        t2 = time.perf_counter()
        cam = get_render()
        t3 = time.perf_counter()
        gt_post = get_playback()
        t4 = time.perf_counter()
        log.append({
            "t0": t0 - t_start, "t1": t1 - t_start, "t2": t2 - t_start,
            "t3": t3 - t_start, "t4": t4 - t_start,
            "gt_pre": gt_pre, "gt_post": gt_post,
            "hero": hero, "click": click, "cam": cam,
        })
        next_t += target_dt
        sleep_for = next_t - time.perf_counter()
        if sleep_for > 0: time.sleep(sleep_for)
        else: next_t = time.perf_counter()  # don't accumulate debt

    out_path = r"C:\tmp\drift_probe.json"
    os.makedirs(r"C:\tmp", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"hero_addr": hex(hero_ptr), "click_addr": hex(click_addr),
                   "duration_s": PROBE_DURATION_S, "target_hz": TARGET_HZ,
                   "samples": log}, f)
    print(f"wrote {out_path}  ({len(log)} samples)")

    analyze(log)
    return 0

def analyze(log):
    """Print summary metrics."""
    import statistics as st
    n = len(log)
    if n < 10: print("not enough samples"); return
    print(f"\n=== analysis ({n} samples) ===")

    # Sample rate achieved
    dts = [log[i+1]["t0"] - log[i]["t0"] for i in range(n-1)]
    print(f"sample dt: p50={1000*st.median(dts):.2f}ms  p95={1000*sorted(dts)[int(0.95*len(dts))]:.2f}ms  achieved_hz={1.0/st.median(dts):.0f}")

    # (a) HTTP RTT
    pb_rtts = [(s["t1"] - s["t0"]) for s in log if s["gt_pre"] is not None]
    pb_rtts2 = [(s["t4"] - s["t3"]) for s in log if s["gt_post"] is not None]
    cam_rtts = [(s["t3"] - s["t2"]) for s in log if s["cam"] is not None]
    pb_all = pb_rtts + pb_rtts2
    print(f"\n(a) HTTP RTT:")
    if pb_all:
        print(f"  /replay/playback : p50={1000*st.median(pb_all):.2f}ms  p95={1000*sorted(pb_all)[int(0.95*len(pb_all))]:.2f}ms  max={1000*max(pb_all):.2f}ms")
    if cam_rtts:
        print(f"  /replay/render   : p50={1000*st.median(cam_rtts):.2f}ms  p95={1000*sorted(cam_rtts)[int(0.95*len(cam_rtts))]:.2f}ms  max={1000*max(cam_rtts):.2f}ms")

    # (b) game_t advance per iter — granularity of click timestamps
    gt_advances = []
    for s in log:
        if s["gt_pre"] is not None and s["gt_post"] is not None:
            gt_advances.append(s["gt_post"] - s["gt_pre"])
    if gt_advances:
        print(f"\n(b) game_t advance per iter (= click-stamp granularity ceiling):")
        print(f"  p50={1000*st.median(gt_advances):.2f}ms  p95={1000*sorted(gt_advances)[int(0.95*len(gt_advances))]:.2f}ms  max={1000*max(gt_advances):.2f}ms")

    # (c) gt-vs-wall jitter: how non-linear is gt(wall)?
    valid = [(s["t0"], s["gt_pre"]) for s in log if s["gt_pre"] is not None]
    if len(valid) > 50:
        ws = np.array([v[0] for v in valid])
        gs = np.array([v[1] for v in valid])
        # Fit a linear gt = a*wall + b, look at residual
        a, b = np.polyfit(ws, gs, 1)
        resid = gs - (a*ws + b)
        print(f"\n(c) wall->gt linearity (resid from linear fit):")
        print(f"  slope={a:.3f}  (1.0=1x speed, 2.0=2x)  resid_std={1000*resid.std():.2f}ms  max_abs={1000*max(abs(resid.min()), abs(resid.max())):.2f}ms")

    # (d) hero->cam lag via cross-correlation
    valid = [s for s in log if s["hero"] and s["cam"]]
    if len(valid) > 100:
        hx = np.array([s["hero"][0] for s in valid])
        hz = np.array([s["hero"][2] for s in valid])
        cx = np.array([s["cam"][0] for s in valid])
        cz = np.array([s["cam"][2] for s in valid])
        ts = np.array([s["t2"] for s in valid])
        dt = float(np.median(np.diff(ts)))
        # Cross-correlate hero(t) with cam(t-tau); positive tau = cam lags hero
        max_lag_s = 0.5
        max_lag = int(max_lag_s / dt)
        h_signal = np.concatenate([hx - hx.mean(), hz - hz.mean()])
        c_signal = np.concatenate([cx - cx.mean(), cz - cz.mean()])
        # Normalize per-segment
        N = len(hx)
        best_corr = -1; best_lag = 0
        for lag in range(-max_lag, max_lag+1):
            if lag >= 0:
                a = hx[:N-lag] - hx[:N-lag].mean()
                b = cx[lag:] - cx[lag:].mean()
                if a.std() < 1 or b.std() < 1: continue
                corr = float(np.corrcoef(a, b)[0,1])
            else:
                a = hx[-lag:] - hx[-lag:].mean()
                b = cx[:N+lag] - cx[:N+lag].mean()
                if a.std() < 1 or b.std() < 1: continue
                corr = float(np.corrcoef(a, b)[0,1])
            if corr > best_corr:
                best_corr = corr; best_lag = lag
        print(f"\n(d) hero→cam lag (cross-correlation peak):")
        print(f"  best_lag={best_lag} samples = {1000*best_lag*dt:.0f}ms  (positive = cam lags hero)  corr={best_corr:.3f}")

    # (e) click event timing: when did vec3 change, and how big is the gt window?
    click_events = []
    for i in range(1, len(log)):
        a = log[i-1]["click"]; b = log[i]["click"]
        if not a or not b: continue
        d = ((a[0]-b[0])**2 + (a[2]-b[2])**2)**0.5
        if d > 50:  # click delta threshold
            gt_window = (log[i]["gt_pre"] or 0) - (log[i-1]["gt_post"] or 0)
            wall_gap = log[i]["t2"] - log[i-1]["t2"]
            click_events.append({"i": i, "gt_window_s": gt_window, "wall_gap_s": wall_gap,
                                 "delta_units": d})
    if click_events:
        gws = [e["gt_window_s"] for e in click_events]
        wgs = [e["wall_gap_s"] for e in click_events]
        print(f"\n(e) detected click events ({len(click_events)} total):")
        print(f"  gt window per click: p50={1000*st.median(gws):.1f}ms  p95={1000*sorted(gws)[int(0.95*len(gws))]:.1f}ms")
        print(f"  wall gap per click : p50={1000*st.median(wgs):.1f}ms  p95={1000*sorted(wgs)[int(0.95*len(wgs))]:.1f}ms")
        print(f"  -> at TARGET_HZ={TARGET_HZ}, the extractor's 10Hz polling has ~10x worse precision")
    else:
        print(f"\n(e) no click events detected during probe (Garen idle?)")

if __name__ == "__main__":
    sys.exit(main())
