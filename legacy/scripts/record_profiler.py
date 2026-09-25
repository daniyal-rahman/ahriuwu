"""Profile a /replay/recording pass to localize the wall-fps bottleneck.

Runs a 60s game-time recording with cam locked to Garen, while a sampler thread
captures every 500ms:
  - League process: CPU%, mem MB, per-thread CPU breakdown (top 3)
  - Python process (this script): CPU%, mem
  - System CPU% per logical core
  - Disk write rate to staging dir (bytes/s + new files/s)
  - GPU%, GPU mem, GPU encoder %, GPU power (via nvidia-smi if present)
  - /replay/playback time (game-time progress vs wall progress)

Output: C:\\tmp\\record_profile.json + console summary.

Designed to be the smallest reproducible test of the slow-recording phenomenon
(22fps wall when target is 40fps at 720p).
"""
import sys, os, time, json, ssl, subprocess, urllib.request, base64
import threading, glob, statistics
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

try:
    import psutil
except ImportError:
    print("ERR: psutil missing — `pip install psutil`"); sys.exit(2)

GAME_ID = "5547184086"
LOCKFILE = r"C:\Riot Games\League of Legends\lockfile"
STAGING = r"C:\tmp\record_profile_pngs"
PROFILE_JSON = r"C:\tmp\record_profile.json"
REC_DUR_GT = 60.0       # 60s game time
FPS = 20                # match pipeline.py
SPEED = 2.0             # match pipeline.py
SAMPLE_DT = 0.5

_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
REPLAY = "https://127.0.0.1:2999"

def lcu_post(ep, body=None):
    parts = open(LOCKFILE).read().strip().split(":")
    port = parts[2]; auth = base64.b64encode(f"riot:{parts[3]}".encode()).decode()
    req = urllib.request.Request(
        f"https://127.0.0.1:{port}{ep}", method="POST",
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, context=_ctx, timeout=10) as r:
        raw = r.read(); return json.loads(raw) if raw else None

def replay_post(p, body):
    req = urllib.request.Request(f"{REPLAY}{p}", method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return r.read()

def api_get(p):
    try:
        with urllib.request.urlopen(f"{REPLAY}{p}", context=_ctx, timeout=2) as r:
            return json.loads(r.read())
    except: return None

def find_league():
    for p in psutil.process_iter(['name','pid']):
        if p.info['name'] and 'League of Legends.exe' in p.info['name']:
            return psutil.Process(p.info['pid'])
    return None

def gpu_sample():
    """Returns dict from nvidia-smi or None if unavailable."""
    try:
        out = subprocess.run([
            "nvidia-smi",
            "--query-gpu=utilization.gpu,utilization.memory,memory.used,power.draw,encoder.stats.sessionCount,encoder.stats.averageFps",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, timeout=3).stdout.strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "gpu_pct": float(parts[0]) if parts[0] not in ("", "[N/A]") else None,
            "mem_pct": float(parts[1]) if parts[1] not in ("", "[N/A]") else None,
            "mem_used_mb": float(parts[2]) if parts[2] not in ("", "[N/A]") else None,
            "power_w": float(parts[3]) if parts[3] not in ("", "[N/A]") else None,
            "enc_sessions": int(parts[4]) if parts[4] not in ("", "[N/A]") else None,
            "enc_fps": float(parts[5]) if parts[5] not in ("", "[N/A]") else None,
        }
    except Exception:
        return None

def disk_io_sample():
    """Per-disk write bytes/sec via psutil.disk_io_counters."""
    try:
        ctrs = psutil.disk_io_counters(perdisk=False)
        return {"write_bytes": ctrs.write_bytes, "write_count": ctrs.write_count,
                "read_bytes": ctrs.read_bytes, "read_count": ctrs.read_count}
    except: return None

def lock_cam_recipe(key="1"):
    """pause→select→unpause→focus+lock×2. Reuses scancode SendInput directly."""
    import ctypes, ctypes.wintypes as wt
    PUL = ctypes.POINTER(ctypes.c_ulong)
    class KeyBdInput(ctypes.Structure):
        _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                    ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                    ("dwExtraInfo", PUL)]
    class _II(ctypes.Union):
        _fields_ = [("ki", KeyBdInput)]
    class Input(ctypes.Structure):
        _fields_ = [("type", ctypes.c_ulong), ("ii", _II)]
    SCAN = {'1':0x02,'2':0x03,'3':0x04,'4':0x05,'5':0x06,'q':0x10,'w':0x11,'e':0x12,'r':0x13,'t':0x14}
    def send(scan, up=False):
        flags = 0x0008 | (0x0002 if up else 0)
        extra = ctypes.c_ulong(0)
        ki = KeyBdInput(wVk=0, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=ctypes.pointer(extra))
        ctypes.windll.user32.SendInput(1, ctypes.byref(Input(type=1, ii=_II(ki=ki))), ctypes.sizeof(Input))
    def tap(s): send(s); time.sleep(0.06); send(s, up=True)
    def find_hwnd():
        u32 = ctypes.windll.user32; hwnds=[]
        def cb(h, _):
            if not u32.IsWindowVisible(h): return True
            cls = ctypes.create_unicode_buffer(256); u32.GetClassNameW(h, cls, 256)
            if cls.value == "RiotWindowClass": hwnds.append(h)
            return True
        u32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)(cb), 0)
        return hwnds[0] if hwnds else None
    def focus():
        u32 = ctypes.windll.user32; k = ctypes.windll.kernel32
        h = find_hwnd()
        if not h: return False
        u32.SystemParametersInfoW(0x2001, 0, None, 0)
        fg = u32.GetForegroundWindow()
        ft = u32.GetWindowThreadProcessId(fg, None); ct = k.GetCurrentThreadId()
        u32.AttachThreadInput(ct, ft, True)
        u32.keybd_event(0x12, 0, 0, 0); u32.keybd_event(0x12, 0, 2, 0)
        u32.ShowWindow(h, 9); u32.BringWindowToTop(h); u32.SetForegroundWindow(h)
        u32.AttachThreadInput(ct, ft, False); time.sleep(0.3); return True
    scan = SCAN[key]
    replay_post("/replay/playback", {"paused": True, "time": 120.0}); time.sleep(0.5)
    replay_post("/replay/render", {"interfaceAll": True, "selectionName": "Garen"}); time.sleep(0.5)
    replay_post("/replay/playback", {"paused": False, "speed": SPEED}); time.sleep(1.0)
    focus(); time.sleep(0.3); tap(scan); time.sleep(0.28); tap(scan); time.sleep(0.5)
    focus(); time.sleep(0.3); tap(scan); time.sleep(0.28); tap(scan); time.sleep(0.5)
    print("  cam locked", flush=True)

def launch():
    print(f"launching replay {GAME_ID}...", flush=True)
    try: lcu_post(f"/lol-replays/v1/rofls/{GAME_ID}/watch", {"componentType": "replay"})
    except Exception as e: print(f"  LCU: {e}", flush=True)
    for i in range(120):
        gs = api_get("/liveclientdata/gamestats")
        if gs and gs.get("gameTime") is not None:
            print(f"  loaded ({i*2}s)", flush=True); return True
        time.sleep(2)
    return False

def sampler_loop(stop, samples, league_proc, this_proc):
    last_disk = disk_io_sample()
    last_t = time.perf_counter()
    last_png_count = 0
    while not stop.is_set():
        t0 = time.perf_counter()
        # League
        try:
            with league_proc.oneshot():
                l_cpu = league_proc.cpu_percent(interval=None)
                l_mem = league_proc.memory_info().rss / 1024 / 1024
                threads = league_proc.threads()
                l_n_threads = len(threads)
        except Exception:
            l_cpu=l_mem=None; l_n_threads=0
        # this script
        py_cpu = this_proc.cpu_percent(interval=None)
        py_mem = this_proc.memory_info().rss / 1024 / 1024
        # System per-cpu
        sys_cpu = psutil.cpu_percent(interval=None, percpu=True)
        # Disk
        disk = disk_io_sample()
        if disk and last_disk:
            dt = t0 - last_t
            wbps = (disk["write_bytes"] - last_disk["write_bytes"]) / dt
            wips = (disk["write_count"] - last_disk["write_count"]) / dt
        else:
            wbps = wips = None
        last_disk = disk; last_t = t0
        # PNG count growth
        png_count = len(glob.glob(os.path.join(STAGING, "**", "*.png"), recursive=True))
        png_delta = png_count - last_png_count
        last_png_count = png_count
        # GPU
        gpu = gpu_sample()
        # Replay engine clock
        pb = api_get("/replay/playback")
        rec = api_get("/replay/recording")
        samples.append({
            "t": t0,
            "league_cpu_pct": l_cpu, "league_mem_mb": round(l_mem or 0, 1),
            "league_threads": l_n_threads,
            "py_cpu_pct": py_cpu, "py_mem_mb": round(py_mem or 0, 1),
            "sys_cpu_pct": sys_cpu,
            "sys_cpu_total": round(sum(sys_cpu)/len(sys_cpu), 1),
            "disk_write_mbps": round(wbps/1048576, 2) if wbps is not None else None,
            "disk_write_iops": round(wips, 0) if wips is not None else None,
            "png_count": png_count, "png_delta": png_delta,
            "wall_pngs_per_s": round(png_delta / SAMPLE_DT, 1),
            "gpu": gpu,
            "playback_time": pb.get("time") if pb else None,
            "recording_active": rec.get("recording") if rec else None,
        })
        sleep_for = SAMPLE_DT - (time.perf_counter() - t0)
        if sleep_for > 0: time.sleep(sleep_for)

def main():
    print("=== record profiler ===", flush=True)
    if not launch(): print("ERR launch"); return 1

    # Clean staging
    os.makedirs(STAGING, exist_ok=True)
    for f in glob.glob(os.path.join(STAGING, "**", "*.png"), recursive=True):
        try: os.remove(f)
        except: pass

    # Cam lock + speed
    lock_cam_recipe(key="1")

    # Find League proc + pin to all cores (matches pipeline.py pass2)
    league = find_league()
    if not league: print("ERR no League proc"); return 1
    print(f"  League pid={league.pid}", flush=True)
    n_cores = psutil.cpu_count(logical=True) or 6
    try: league.cpu_affinity(list(range(n_cores)))
    except: pass
    print(f"  League pinned to {n_cores} cores", flush=True)
    league.cpu_percent(interval=None)  # prime
    me = psutil.Process(); me.cpu_percent(interval=None)

    # Start recording
    rec_start_gt = 120.0
    rec_end_gt = rec_start_gt + REC_DUR_GT
    print(f"\nstarting recording: gt {rec_start_gt}→{rec_end_gt} ({REC_DUR_GT}s game-time)", flush=True)
    t_rec_start = time.perf_counter()
    res = replay_post("/replay/recording", {
        "recording": True,
        "path": STAGING.replace("\\", "/"),
        "codec": "png",
        "framesPerSecond": FPS * 2,
        "startTime": rec_start_gt,
        "endTime": rec_end_gt,
        "enforceFrameRate": True,
    })
    print(f"  recording started: {res}", flush=True)

    # Sampler thread
    stop = threading.Event(); samples = []
    sampler = threading.Thread(target=sampler_loop, args=(stop, samples, league, me), daemon=True)
    sampler.start()

    # Wait for recording to finish (or 5min wall timeout)
    deadline = time.perf_counter() + 300
    while time.perf_counter() < deadline:
        time.sleep(2)
        r = api_get("/replay/recording")
        if r is not None and not r.get("recording"):
            print("  recording complete", flush=True); break
    else:
        print("  TIMEOUT", flush=True)
    t_rec_end = time.perf_counter()
    stop.set(); sampler.join(timeout=2)

    wall = t_rec_end - t_rec_start
    n = len(glob.glob(os.path.join(STAGING, "**", "*.png"), recursive=True))
    eff_fps = n / wall if wall > 0 else 0
    print(f"\n  wall={wall:.1f}s  pngs={n}  effective_fps={eff_fps:.1f}", flush=True)

    # Save
    with open(PROFILE_JSON, "w") as f:
        json.dump({
            "wall_s": round(wall, 2), "n_pngs": n, "effective_fps": round(eff_fps, 2),
            "rec_dur_gt": REC_DUR_GT, "speed": SPEED, "fps_target": FPS*2,
            "samples": samples,
        }, f, indent=1)
    print(f"  wrote {PROFILE_JSON}", flush=True)

    # Summary
    print("\n=== summary ===", flush=True)
    if samples:
        active = [s for s in samples if s.get("recording_active")]
        if not active: active = samples
        def med(key, default=None):
            xs = [s[key] for s in active if s.get(key) is not None]
            return statistics.median(xs) if xs else default
        def p95(key):
            xs = sorted([s[key] for s in active if s.get(key) is not None])
            return xs[int(0.95*len(xs))] if xs else None
        print(f"  League CPU%   : p50={med('league_cpu_pct'):.0f}%  p95={p95('league_cpu_pct')}%")
        print(f"  System total  : p50={med('sys_cpu_total'):.0f}%")
        print(f"  Disk write    : p50={med('disk_write_mbps')}MB/s  p95={p95('disk_write_mbps')}MB/s")
        print(f"  Disk IOPS     : p50={med('disk_write_iops')}  p95={p95('disk_write_iops')}")
        print(f"  PNGs/s        : p50={med('wall_pngs_per_s')}  p95={p95('wall_pngs_per_s')}")
        gpus = [s["gpu"] for s in active if s.get("gpu")]
        if gpus:
            gpu_pct = [g["gpu_pct"] for g in gpus if g.get("gpu_pct") is not None]
            if gpu_pct:
                print(f"  GPU%          : p50={statistics.median(gpu_pct):.0f}%  p95={sorted(gpu_pct)[int(0.95*len(gpu_pct))]:.0f}%")
            enc_pct = [g.get("mem_pct") for g in gpus if g.get("mem_pct") is not None]
            print(f"  GPU mem%      : p50={statistics.median(enc_pct) if enc_pct else 'N/A'}")
        # Per-core hot core analysis
        all_pcc = [s["sys_cpu_pct"] for s in active if s.get("sys_cpu_pct")]
        if all_pcc:
            ncores = len(all_pcc[0])
            per_core_med = [statistics.median([row[i] for row in all_pcc]) for i in range(ncores)]
            print(f"  Per-core CPU% (median): {[round(c,0) for c in per_core_med]}")
            print(f"    -> hottest core: {max(per_core_med):.0f}%, coldest: {min(per_core_med):.0f}%")

    return 0

if __name__ == "__main__":
    sys.exit(main())
