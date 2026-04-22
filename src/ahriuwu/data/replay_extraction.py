"""
Garen Replay Data Pipeline — 2 Pass + Overlapped Post-Processing
=================================================================
Pass 1: Memory scrape (50Hz) + camera API poll (93Hz persistent HTTPS), 2x speed
        → determines real game duration from game time stall
        → cam samples timestamped via wall→gt interpolation from mem thread
Pass 2: Record video (720p PNG, 40fps enforced at 2x = 20fps game-time)
        → uses real duration from pass 1 as endTime
Post:   Resize 720→352² (cv2, capped at 2 threads), build labels.json, delete originals
        → runs in a separate process, overlapped with the next game's pass 1

Runs on Windows with:
  - League client open (for LCU replay launch)
  - Vanguard disabled (for ReadProcessMemory)
  - game.cfg: Width=1280, Height=720, WindowMode=1, EnableReplayApi=1

Usage:
    python pipeline.py --manifest E:\\garen_manifest_300.json --index 0
    python pipeline.py --manifest E:\\garen_manifest_300.json --batch
    python pipeline.py --game-id 5528069928 --match-id NA1_5528069928 --team red --slot 5
    python pipeline.py --benchmark   # 3×3 scenario test for overlap contention

Output per game:  E:\\replay_data\\{match_id}\\
    frames\\  → 352×352 PNGs at 20fps
    labels.json → per-frame labels (720p screen coords, null for unlabeled)

Structured log: E:\\replay_data\\pipeline.jsonl
Memory offsets: patch 26.7, module size 0x207C000
"""
import argparse, base64, bisect, ctypes, ctypes.wintypes as wt
import glob, http.client, json, math, multiprocessing as mp, os, shutil
import socket, struct, subprocess, sys, threading, time, traceback
import urllib.request, urllib.error, ssl
from pynput.keyboard import Controller as KbController

try:
    import psutil
except ImportError:
    psutil = None

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
SCREEN_W, SCREEN_H = 1280, 720
FRAME_SZ = 352
FPS = 20

# Camera projection (locked to Garen, empirically measured)
CAM_Y = 1912.0
CAM_Z_OFFSET = -1292.0
FLOOR_Y = 52.0
TILT = math.radians(56.0)
FOV_V = math.radians(40.0)
FOV_H = 2 * math.atan(math.tan(FOV_V / 2) * SCREEN_W / SCREEN_H)
TAN_H, TAN_V = math.tan(FOV_H / 2), math.tan(FOV_V / 2)
COS_T, SIN_T = math.cos(TILT), math.sin(TILT)

# ── Memory offsets (patch 16.8, module size 0x2097000) ──
# 16.8 shifts from 16.7: RVAs +0x18240, hero fields 0x2858..0x30E8 +0x38,
# hero fields 0x4CE8..0x55D8 +0x40. API-verified 2026-04-16.
EXPECTED_MOD_SIZE = 0x2097000
OFFSETS = {
    # base-relative RVAs
    "hero_array":     0x1DD7128,
    "game_time":      0x1DE5420,

    # hero struct — inline fields
    "position":       0x25C,     # unchanged (GameObject base)
    "hp_current":     0x1080,    # f32 (unchanged)
    "hp_max":         0x10A8,    # f32 (unchanged)
    "gold_current":   0x2890,    # f32 (+0x38 from 0x2858)
    "gold_total":     0x4D28,    # f32 (+0x40 from 0x4CE8)
    "active_spell":   0x3158,    # ptr (+0x38 from 0x3120)
    "champion_name":  0x4360,    # ASCII str (+0x38 from 0x4328)
    "level":          0x4D50,    # u32 (+0x40 from 0x4D10)
    "kills":          0x5468,    # u32 (+0x40 tentative — NOT in hero struct per stats_offset_research)
    "assists":        0x54C8,    # u32 (+0x40 tentative — same caveat)

    # ActiveSpellCast struct (unchanged)
    "spell_info":     0x008,
    "cast_target":    0x0DC,

    # SpellInfo struct (unchanged)
    "spell_name_ptr": 0x28,
}

OUTPUT_BASE  = r"E:\replay_data"
TEMP_BASE    = r"E:\tmp\_pipeline"
LOCKFILE     = r"C:\Riot Games\League of Legends\lockfile"
JSONL_PATH   = os.path.join(OUTPUT_BASE, "pipeline.jsonl")

# ── Alarm thresholds ──
ALARM_MIN_SPEED    = 1.5   # effective replay speed
ALARM_MIN_REC_FPS  = 25    # effective recording fps
ALARM_MIN_MEM_HZ   = 40    # mem thread Hz
MAX_MEM_GAP        = 0.1   # 100ms — for label matching
MAX_CAM_GAP        = 0.1

# ── Core allocation (6-core i5-8600K) ──
# Pass 1 (scrape at 2x): League uses ~2 cores + Python threads ~0.5 → 3 cores
# Post workers: parallelized resize, 2 cores
# Core 5: buffer for overflow
PASS1_CORES = [0, 1, 2, 5]    # League pinned here during pass1; core 5 is buffer
POST_CORES  = [3, 4]          # Post workers pinned here (runs during pass 1)
ALL_CORES   = [0, 1, 2, 3, 4, 5]  # pass 2 recording uses everything

_ctx = ssl.create_default_context()
_ctx.check_hostname = False; _ctx.verify_mode = ssl.CERT_NONE
_k = ctypes.windll.kernel32
_kb = KbController()

# ═══════════════════════════════════════════════════════════════
# Structured Logging
# ═══════════════════════════════════════════════════════════════
def log_event(path, **fields):
    fields["ts"] = time.time()
    fields["host"] = socket.gethostname()
    with open(path, "a") as f:
        f.write(json.dumps(fields) + "\n")

def check_alarm(label, value, threshold, op=">="):
    """Print loud alarm if value crosses threshold. Returns True if alarmed."""
    if op == ">=" and value >= threshold: return False
    if op == "<" and value >= threshold: return False
    if op == "<" and value < threshold:
        print(f"\n  *** ALARM: {label} = {value:.1f} (threshold {threshold}) ***\n", flush=True)
        return True
    if op == ">=" and value < threshold:
        print(f"\n  *** ALARM: {label} = {value:.1f} (threshold {threshold}) ***\n", flush=True)
        return True
    return False

# ═══════════════════════════════════════════════════════════════
# API Helpers
# ═══════════════════════════════════════════════════════════════
def replay_get(ep):
    with urllib.request.urlopen(
        urllib.request.Request(f"https://127.0.0.1:2999{ep}"),
        context=_ctx, timeout=5
    ) as r:
        return json.loads(r.read())

def replay_post(ep, data):
    with urllib.request.urlopen(
        urllib.request.Request(f"https://127.0.0.1:2999{ep}",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"}),
        context=_ctx, timeout=5
    ) as r:
        return json.loads(r.read())

def lcu_auth():
    parts = open(LOCKFILE).read().strip().split(":")
    return parts[2], f"Basic {base64.b64encode(f'riot:{parts[3]}'.encode()).decode()}"

def lcu_post(ep, body=None):
    port, auth = lcu_auth()
    req = urllib.request.Request(
        f"https://127.0.0.1:{port}{ep}", method="POST",
        data=json.dumps(body).encode() if body else None,
        headers={"Authorization": auth, "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, context=_ctx, timeout=10) as r:
        raw = r.read()
        return json.loads(raw) if raw else None

# ═══════════════════════════════════════════════════════════════
# Memory Reader
# ═══════════════════════════════════════════════════════════════
class Mem:
    def __init__(self, pid):
        self.h = _k.OpenProcess(0x0410, False, pid)
        if not self.h: raise OSError(f"OpenProcess({pid}) failed")
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u64(self, a):
        d = self.read(a, 8); return struct.unpack('<Q', d)[0] if d else None
    def u32(self, a):
        d = self.read(a, 4); return struct.unpack('<I', d)[0] if d else None
    def f32(self, a):
        d = self.read(a, 4); return struct.unpack('<f', d)[0] if d else None
    def vec3(self, a):
        d = self.read(a, 12); return struct.unpack('<fff', d) if d else None
    def string(self, a, n=32):
        d = self.read(a, n)
        if not d: return None
        t = d.split(b'\x00')[0]
        try: return t.decode('ascii')
        except: return None
    def close(self): _k.CloseHandle(self.h)

def find_league_pid():
    r = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq League of Legends.exe',
                        '/FO', 'CSV', '/NH'], capture_output=True, text=True)
    for line in r.stdout.strip().split('\n'):
        if 'league' in line.lower():
            return int(line.strip('"').split('","')[1])
    return None

def find_module_base(pid):
    class ME(ctypes.Structure):
        _fields_ = [("dwSize", ctypes.c_ulong), ("th32ModuleID", ctypes.c_ulong),
            ("th32ProcessID", ctypes.c_ulong), ("GlblcntUsage", ctypes.c_ulong),
            ("ProccntUsage", ctypes.c_ulong), ("modBaseAddr", ctypes.POINTER(ctypes.c_byte)),
            ("modBaseSize", ctypes.c_ulong), ("hModule", ctypes.c_void_p),
            ("szModule", ctypes.c_char * 256), ("szExePath", ctypes.c_char * 260)]
    snap = _k.CreateToolhelp32Snapshot(0x18, pid)
    me = ME(); me.dwSize = ctypes.sizeof(ME)
    if _k.Module32First(snap, ctypes.byref(me)):
        while True:
            if b'league' in me.szModule.lower():
                base = ctypes.cast(me.modBaseAddr, ctypes.c_void_p).value
                size = me.modBaseSize
                _k.CloseHandle(snap)
                return base, size
            if not _k.Module32Next(snap, ctypes.byref(me)): break
    _k.CloseHandle(snap)
    return None, None

def init_heroes(m, base):
    arr_ptr = m.u64(base + OFFSETS["hero_array"])
    if not arr_ptr: return {}
    heroes = {}
    for i in range(10):
        hp = m.u64(arr_ptr + i * 8)
        if not hp or hp < 0x10000: continue
        name = m.string(hp + OFFSETS["champion_name"])
        if name and len(name) >= 2 and name[0].isalpha():
            heroes[name] = {"ptr": hp, "slot": i, "team": "blue" if i < 5 else "red"}
    return heroes

def verify_game_time(m, base):
    gt_rva = OFFSETS["game_time"]
    gt = m.f32(base + gt_rva)
    if gt is not None and 0 < gt < 10000:
        return gt_rva
    return None

# ═══════════════════════════════════════════════════════════════
# Window Focus & Camera Lock
# ═══════════════════════════════════════════════════════════════
def find_game_hwnd():
    user32 = ctypes.windll.user32; hwnds = []
    def cb(hwnd, _):
        n = user32.GetWindowTextLengthW(hwnd)
        if n > 0:
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if "league of legends" in buf.value.lower(): hwnds.append(hwnd)
        return True
    user32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)(cb), 0)
    return hwnds[0] if hwnds else None

def focus_game():
    user32 = ctypes.windll.user32
    hwnd = find_game_hwnd()
    if not hwnd: return False
    user32.SystemParametersInfoW(0x2001, 0, None, 0)
    fg = user32.GetForegroundWindow()
    ft = user32.GetWindowThreadProcessId(fg, None)
    ct = _k.GetCurrentThreadId()
    user32.AttachThreadInput(ct, ft, True)
    user32.keybd_event(0x12, 0, 0, 0); user32.keybd_event(0x12, 0, 2, 0)
    user32.ShowWindow(hwnd, 9); user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    user32.AttachThreadInput(ct, ft, False)
    time.sleep(0.3)
    return True

def click_center():
    user32 = ctypes.windll.user32
    hwnd = find_game_hwnd()
    if not hwnd: return
    rect = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    cx = (rect.left + rect.right) // 2
    cy = (rect.top + rect.bottom) // 2
    user32.SetCursorPos(cx, cy)
    time.sleep(0.1)
    user32.mouse_event(0x0002, 0, 0, 0, 0)
    time.sleep(0.05)
    user32.mouse_event(0x0004, 0, 0, 0, 0)
    time.sleep(0.2)

def lock_camera(key):
    click_center()
    _kb.press(key); time.sleep(0.05); _kb.release(key)
    time.sleep(0.15)
    _kb.press(key); time.sleep(0.05); _kb.release(key)
    time.sleep(0.3)

def cam_key_for(team, slot):
    if team == "blue": return str(slot + 1)
    return "qwert"[slot - 5]

# ═══════════════════════════════════════════════════════════════
# Projection
# ═══════════════════════════════════════════════════════════════
def project(wx, wz, cx, cy, cz):
    dx = wx - cx; dy = FLOOR_Y - cy; dz = wz - cz
    vy = dy * COS_T + dz * SIN_T
    vz = -dy * SIN_T + dz * COS_T
    if vz <= 10: return None
    sx = 0.5 + (dx / vz) / TAN_H * 0.5
    sy = 0.5 - (vy / vz) / TAN_V * 0.5
    px, py = int(sx * SCREEN_W), int(sy * SCREEN_H)
    return [px, py] if 0 <= px < SCREEN_W and 0 <= py < SCREEN_H else None

def classify_spell(name):
    """Classify a spell name into action type.
    Order matters: BasicAttack must be checked before Q/W/E/R since
    abilities like 'GarenQAttack' contain 'Attack'."""
    if not name: return "idle"
    if "BasicAttack" in name: return "attack"
    if "Recall" in name: return "recall"
    if name.startswith("Summoner"): return "summoner"
    # Garen abilities: GarenQ, GarenQAttack, GarenW, GarenE, GarenR
    if name.startswith("Garen"):
        suffix = name[5:]
        if suffix[:1] in ("Q", "W", "E", "R"):
            return "ability"
    # Generic ability detection for other champions
    return "other"

# ═══════════════════════════════════════════════════════════════
# Pipeline Utilities
# ═══════════════════════════════════════════════════════════════
def kill_game():
    os.system('taskkill /F /IM "League of Legends.exe" >nul 2>&1')
    time.sleep(3)

def pin_league(cores):
    """Set League process CPU affinity. No-op if psutil missing or process not found."""
    if psutil is None: return
    pid = find_league_pid()
    if not pid: return
    try:
        psutil.Process(pid).cpu_affinity(cores)
        print(f"  Pinned League to cores {cores}", flush=True)
    except Exception as e:
        print(f"  Could not pin League: {e}", flush=True)

def launch_replay(game_id):
    print(f"  Launching replay {game_id}...", flush=True)
    try:
        lcu_post(f"/lol-replays/v1/rofls/{game_id}/watch",
                 {"componentType": "replay"})
    except Exception as e:
        print(f"  LCU launch failed: {e}", flush=True)
        return False
    for i in range(120):
        try:
            replay_get("/liveclientdata/gamestats")
            print(f"  Game loaded ({i*2}s)", flush=True)
            return True
        except: time.sleep(2)
    print("  TIMEOUT: game did not load", flush=True)
    return False

def _nearest(arr, keys, gt):
    i = bisect.bisect_right(keys, gt)
    best, best_gap = None, float("inf")
    for j in (i-1, i):
        if 0 <= j < len(arr):
            gap = abs(arr[j]["gt"] - gt)
            if gap < best_gap:
                best, best_gap = arr[j], gap
    return best, best_gap

# ═══════════════════════════════════════════════════════════════
# Pass 1: Memory + Camera Scrape
# ═══════════════════════════════════════════════════════════════
def _mem_loop(m, base, hero_ptrs, gt_rva, stop, results):
    o = OFFSETS
    interval = 1.0 / 50
    while not stop.is_set():
        tick = time.perf_counter()
        gt = m.f32(base + gt_rva) if gt_rva else 0
        heroes = {}
        for name, hinfo in hero_ptrs.items():
            hp = hinfo["ptr"]
            pos = m.vec3(hp + o["position"])
            if not pos: continue
            entry = {
                "pos": [round(pos[0], 1), round(pos[2], 1)],
                "hp":         round(m.f32(hp + o["hp_current"]) or 0, 1),
                "hp_max":     round(m.f32(hp + o["hp_max"]) or 0, 1),
                "gold":       round(m.f32(hp + o["gold_current"]) or 0, 1),
                "gold_total": round(m.f32(hp + o["gold_total"]) or 0, 1),
                "level":      m.u32(hp + o["level"]) or 0,
                "kills":      m.u32(hp + o["kills"]) or 0,
                "assists":    m.u32(hp + o["assists"]) or 0,
            }
            # Spell detection for ALL heroes (not just Garen)
            sp = m.u64(hp + o["active_spell"])
            if sp and sp > 0x10000:
                si = m.u64(sp + o["spell_info"])
                if si and si > 0x10000:
                    np_ = m.u64(si + o["spell_name_ptr"])
                    if np_ and np_ > 0x10000:
                        sn = m.string(np_)
                        if sn and len(sn) > 2:
                            entry["spell"] = sn
                            ct = m.vec3(sp + o["cast_target"])
                            if ct:
                                if abs(ct[0]) < 20000 and abs(ct[2]) < 20000:
                                    entry["cast_target"] = [round(ct[0], 1), round(ct[2], 1)]
                                else:
                                    # Self-cast / no valid target → use own position
                                    entry["cast_target"] = [round(pos[0], 1), round(pos[2], 1)]
                                    entry["self_cast"] = True
            heroes[name] = entry
        results.append({"wall": time.perf_counter(), "gt": round(gt, 3), "heroes": heroes})
        elapsed = time.perf_counter() - tick
        if elapsed < interval: time.sleep(interval - elapsed)

def _cam_loop(stop, results):
    conn = http.client.HTTPSConnection("127.0.0.1", 2999, context=_ctx, timeout=5)
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 50
    try:
        while not stop.is_set():
            try:
                wall = time.perf_counter()
                conn.request("GET", "/replay/render")
                data = json.loads(conn.getresponse().read())
                cp = data.get("cameraPosition", {})
                results.append({
                    "wall": wall,
                    "cx": round(cp.get("x", 0), 1),
                    "cy": round(cp.get("y", 0), 1),
                    "cz": round(cp.get("z", 0), 1),
                })
                consecutive_errors = 0
            except (http.client.RemoteDisconnected, ConnectionResetError):
                conn = http.client.HTTPSConnection("127.0.0.1", 2999, context=_ctx, timeout=5)
                consecutive_errors += 1
            except Exception:
                consecutive_errors += 1
                time.sleep(0.01)
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"  CAM LOOP: {MAX_CONSECUTIVE_ERRORS} consecutive errors, bailing", flush=True)
                break
    finally:
        conn.close()

def pass1_scrape(game_id, cam_key, duration, force_patch=False):
    """Memory scrape (50Hz) + camera poll (93Hz) at 2x speed. Returns (mem_data, cam_data, stats)."""
    print("\n--- Pass 1: Memory + Camera Scrape (2x) ---", flush=True)
    scrape_start = time.time()
    kill_game()
    if not launch_replay(game_id):
        return [], [], {}
    time.sleep(5)

    pid = find_league_pid()
    if not pid:
        print("  No game PID", flush=True); kill_game(); return [], [], {}
    base, mod_size = find_module_base(pid)
    if not base:
        print("  No module base", flush=True); kill_game(); return [], [], {}
    if mod_size != EXPECTED_MOD_SIZE:
        msg = f"Module size 0x{mod_size:X} != expected 0x{EXPECTED_MOD_SIZE:X} — patch changed?"
        if force_patch:
            print(f"  WARNING: {msg} (--force-patch)", flush=True)
        else:
            print(f"  ABORT: {msg}", flush=True)
            print(f"  Update OFFSETS or pass --force-patch.", flush=True)
            kill_game(); return [], [], {}

    m = Mem(pid)
    if m.read(base, 2) != b'MZ':
        print("  RPM verify failed", flush=True); m.close(); kill_game(); return [], [], {}
    print(f"  Memory: PID={pid} base=0x{base:X}", flush=True)

    gt_rva = verify_game_time(m, base)
    if not gt_rva:
        print(f"  ABORT: GameTime RVA 0x{OFFSETS['game_time']:X} reads garbage", flush=True)
        m.close(); kill_game(); return [], [], {}
    print(f"  GameTime: 0x{gt_rva:X} = {m.f32(base + gt_rva):.1f}s", flush=True)

    hero_ptrs = init_heroes(m, base)
    if "Garen" not in hero_ptrs:
        print(f"  Garen not found (heroes: {list(hero_ptrs.keys())})", flush=True)
        m.close(); kill_game(); return [], [], {}
    print(f"  Heroes: {list(hero_ptrs.keys())}", flush=True)

    # Pin League to pass1 cores — leave [3,4] free for concurrent post workers
    pin_league(PASS1_CORES)

    # Lock camera, play at 2x
    replay_post("/replay/render", {"interfaceAll": False, "selectionName": "Garen"})
    time.sleep(0.5)
    focus_game(); time.sleep(0.3)
    lock_camera(cam_key)
    time.sleep(0.5)
    replay_post("/replay/playback", {"speed": 2.0})
    time.sleep(0.5)
    focus_game(); lock_camera(cam_key)
    print(f"  Camera locked, 2x speed", flush=True)

    stop = threading.Event()
    mem_data, cam_data = [], []
    mt = threading.Thread(target=_mem_loop, args=(m, base, hero_ptrs, gt_rva, stop, mem_data), daemon=True)
    ct = threading.Thread(target=_cam_loop, args=(stop, cam_data), daemon=True)
    mt.start(); ct.start()

    last_gt = 0; stall = 0
    while not stop.is_set():
        time.sleep(5)
        if mem_data:
            cur_gt = mem_data[-1]["gt"]
            if cur_gt > 0 and abs(cur_gt - last_gt) < 2.0:
                stall += 1
                if stall >= 3:
                    print(f"  Game ended at gt={cur_gt:.0f}s", flush=True)
                    stop.set()
            else:
                stall = 0
            last_gt = cur_gt
            if cur_gt >= duration + 30:
                print(f"  Past duration, stopping at gt={cur_gt:.0f}s", flush=True)
                stop.set()
        if find_league_pid() is None:
            print("  Game process exited", flush=True)
            stop.set()

    mt.join(timeout=5); ct.join(timeout=5)
    scrape_end = time.time()
    wall_span = scrape_end - scrape_start

    # Compute stats
    stats = {"scrape_start": scrape_start, "scrape_end": scrape_end}
    if len(mem_data) >= 2:
        mem_walls = [s["wall"] for s in mem_data]
        mem_span = mem_walls[-1] - mem_walls[0]
        mem_hz = len(mem_data) / mem_span if mem_span > 0 else 0
        mem_gaps = [mem_walls[i+1] - mem_walls[i] for i in range(len(mem_walls)-1)]
        gt_span = mem_data[-1]["gt"] - mem_data[0]["gt"]
        effective_speed = gt_span / mem_span if mem_span > 0 else 0
        stats.update(mem_n=len(mem_data), mem_hz=round(mem_hz, 1),
                     mem_max_gap=round(max(mem_gaps), 4), gt_span=round(gt_span, 1),
                     wall_span=round(wall_span, 1), effective_speed=round(effective_speed, 2))
        check_alarm("effective_speed", effective_speed, ALARM_MIN_SPEED, "<")
        check_alarm("mem_hz", mem_hz, ALARM_MIN_MEM_HZ, "<")
    if len(cam_data) >= 2:
        cam_walls = [s["wall"] for s in cam_data]
        cam_span = cam_walls[-1] - cam_walls[0]
        cam_hz = len(cam_data) / cam_span if cam_span > 0 else 0
        cam_gaps = [cam_walls[i+1] - cam_walls[i] for i in range(len(cam_walls)-1)]
        stats.update(cam_n=len(cam_data), cam_hz=round(cam_hz, 1),
                     cam_max_gap=round(max(cam_gaps), 4))

    print(f"  Memory: {len(mem_data)} @ {stats.get('mem_hz',0)}Hz (max_gap={stats.get('mem_max_gap',0)*1000:.0f}ms)", flush=True)
    print(f"  Camera: {len(cam_data)} @ {stats.get('cam_hz',0)}Hz (max_gap={stats.get('cam_max_gap',0)*1000:.0f}ms)", flush=True)
    if mem_data:
        print(f"  GT: {mem_data[0]['gt']:.0f}-{mem_data[-1]['gt']:.0f}s  speed={stats.get('effective_speed',0):.2f}x", flush=True)
    m.close(); kill_game()
    return mem_data, cam_data, stats

# ═══════════════════════════════════════════════════════════════
# Pass 2: Video Recording
# ═══════════════════════════════════════════════════════════════
def pass2_record(game_id, cam_key, duration, staging_dir):
    """Record full game as 720p PNGs at 2x (40fps enforced = 20fps game-time).
    Writes to staging_dir. Returns (frame_count, stats)."""
    print("\n--- Pass 2: Video Recording (2x, 40fps enforced) ---", flush=True)
    record_start = time.time()
    kill_game()
    if not launch_replay(game_id):
        return 0, {}
    time.sleep(5)

    replay_post("/replay/render", {"interfaceAll": False, "selectionName": "Garen"})
    time.sleep(1)
    focus_game(); time.sleep(0.5)
    lock_camera(cam_key)
    time.sleep(1)
    print(f"  Camera locked (key={cam_key})", flush=True)

    # Pass 2 gets all cores — PNG encoder saturates everything
    pin_league(ALL_CORES)

    # Unpause to 2x
    replay_post("/replay/playback", {"speed": 2.0})
    time.sleep(0.5)
    focus_game()
    lock_camera(cam_key)
    time.sleep(0.5)

    # Prepare staging dir
    os.makedirs(staging_dir, exist_ok=True)
    for f in glob.glob(os.path.join(staging_dir, "**", "*.png"), recursive=True):
        os.remove(f)

    rec = replay_post("/replay/recording", {
        "recording": True,
        "path": staging_dir.replace("\\", "/"),
        "codec": "png",
        "framesPerSecond": FPS * 2,
        "startTime": 1.0,
        "endTime": duration + 60,
        "enforceFrameRate": True,
    })
    print(f"  Recording started: {rec.get('width')}x{rec.get('height')}", flush=True)

    time.sleep(1.5)
    focus_game(); time.sleep(0.3)
    lock_camera(cam_key)
    print(f"  Re-locked after recording start", flush=True)

    max_wait = duration + 120
    t0 = time.time()
    while time.time() - t0 < max_wait:
        time.sleep(10)
        try:
            r = replay_get("/replay/recording")
            if not r.get("recording", False):
                print("  Recording complete", flush=True)
                break
        except:
            if find_league_pid() is None:
                print("  Game exited", flush=True)
                break
    else:
        print(f"  Recording timeout ({max_wait}s)", flush=True)

    record_end = time.time()
    wall_time = record_end - record_start
    pngs = sorted(glob.glob(os.path.join(staging_dir, "**", "*.png"), recursive=True))
    n_frames = len(pngs)
    effective_fps = n_frames / wall_time if wall_time > 0 else 0
    expected_frames = int(duration * FPS)

    stats = {
        "record_start": record_start, "record_end": record_end,
        "frames_recorded": n_frames, "duration_requested": round(duration, 1),
        "wall_time": round(wall_time, 1), "effective_fps": round(effective_fps, 1),
        "expected_frames": expected_frames,
    }
    check_alarm("effective_fps", effective_fps, ALARM_MIN_REC_FPS, "<")

    print(f"  Frames: {n_frames} (expected ~{expected_frames}), {effective_fps:.1f} wall-fps, {wall_time:.0f}s wall", flush=True)
    kill_game()
    return n_frames, stats

# ═══════════════════════════════════════════════════════════════
# Post-Processing (runs in worker process)
# ═══════════════════════════════════════════════════════════════
def _resize_worker_init():
    """Pool worker initializer — pin to POST_CORES, cap cv2 at 1 thread each."""
    import cv2
    cv2.setNumThreads(1)
    if psutil is not None:
        try:
            psutil.Process().cpu_affinity(POST_CORES)
        except Exception:
            pass

def _resize_one(args):
    """Worker function: read PNG, resize to FRAME_SZ², write to dst. Returns (idx, ok)."""
    import cv2
    src, dst = args
    img = cv2.imread(src)
    if img is None:
        try:
            shutil.copy2(src, dst)
            return (dst, False)  # fallback copy
        except Exception:
            return (dst, False)
    out = cv2.resize(img, (FRAME_SZ, FRAME_SZ), interpolation=cv2.INTER_AREA)
    cv2.imwrite(dst, out)
    return (dst, True)


def post_process(match_id, mem_data, cam_data, game_info, staging_dir):
    """Resize frames, build labels, clean up. Returns stats dict."""
    post_start = time.time()
    game_dir = os.path.join(OUTPUT_BASE, match_id)
    frames_dir = os.path.join(game_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    pngs = sorted(glob.glob(os.path.join(staging_dir, "**", "*.png"), recursive=True))
    n_frames = len(pngs)
    print(f"\n--- Post-Processing: {n_frames} frames ({match_id}) ---", flush=True)
    if n_frames == 0:
        print("  ERROR: no frames", flush=True); return None

    # ── Resize 720p → 352×352 (parallel, 2 workers pinned to POST_CORES) ──
    print(f"  Resizing to {FRAME_SZ}x{FRAME_SZ} with 2 workers on cores {POST_CORES}...", flush=True)
    jobs = [(p, os.path.join(frames_dir, f"{i:06d}.png")) for i, p in enumerate(pngs)]
    n_resize_fail = 0
    t_resize_start = time.time()
    with mp.Pool(2, initializer=_resize_worker_init) as pool:
        done = 0
        for dst, ok in pool.imap_unordered(_resize_one, jobs, chunksize=50):
            done += 1
            if not ok:
                n_resize_fail += 1
            if done % 5000 == 0:
                print(f"    {done}/{n_frames}", flush=True)
    resize_wall = time.time() - t_resize_start
    print(f"  Resize complete in {resize_wall:.1f}s ({n_frames/resize_wall:.1f} fps)", flush=True)
    if n_resize_fail > 0:
        print(f"  WARNING: {n_resize_fail} frames failed imread — raw copies used", flush=True)

    # ── Build wall→gt map from mem samples ──
    print("  Building labels...", flush=True)
    start_gt = 1.0
    mem_walls = [s["wall"] for s in mem_data]
    mem_gts   = [s["gt"]   for s in mem_data]

    def wall_to_gt(w):
        i = bisect.bisect_right(mem_walls, w)
        if i == 0 or i >= len(mem_walls): return None
        w0, w1 = mem_walls[i-1], mem_walls[i]
        g0, g1 = mem_gts[i-1], mem_gts[i]
        if w1 == w0: return g0
        t = (w - w0) / (w1 - w0)
        return g0 + t * (g1 - g0)

    # ── Assign gt to cam samples ──
    cam_with_gt = []
    n_cam_dropped = 0
    for cs in cam_data:
        gt = wall_to_gt(cs["wall"])
        if gt is None:
            n_cam_dropped += 1; continue
        cam_with_gt.append({"gt": gt, "cx": cs["cx"], "cy": cs["cy"], "cz": cs["cz"]})
    cam_with_gt.sort(key=lambda s: s["gt"])
    cam_gt_keys = [s["gt"] for s in cam_with_gt]

    if n_cam_dropped > 0:
        print(f"  Cam outside mem range: {n_cam_dropped} dropped", flush=True)

    mem_sorted = sorted(mem_data, key=lambda s: s["gt"])
    mem_gt_keys = [s["gt"] for s in mem_sorted]

    # ── Per-frame labels ──
    frames_out = []
    n_unlabeled = 0
    n_cam_fallback = 0
    mem_gaps_ms = []

    for fi in range(n_frames):
        gt = start_gt + fi / FPS
        bm, mem_gap = _nearest(mem_sorted, mem_gt_keys, gt)
        mem_gaps_ms.append(mem_gap * 1000)

        if mem_gap > MAX_MEM_GAP:
            n_unlabeled += 1
            frames_out.append({"frame": fi, "gt": round(gt, 3), "label": None})
            continue

        heroes = bm.get("heroes", {})
        garen = heroes.get("Garen", {})
        gp = garen.get("pos", [0, 0])

        bc, cam_gap = _nearest(cam_with_gt, cam_gt_keys, gt)
        if bc and cam_gap <= MAX_CAM_GAP:
            cx, cy, cz = bc["cx"], bc["cy"], bc["cz"]
        else:
            n_cam_fallback += 1
            cx, cy, cz = gp[0], CAM_Y, gp[1] + CAM_Z_OFFSET

        garen_screen = project(gp[0], gp[1], cx, cy, cz)
        visible = []
        for name, hd in heroes.items():
            p = hd.get("pos", [0, 0])
            sp = project(p[0], p[1], cx, cy, cz)
            if sp:
                visible.append({
                    "name": name,
                    "screen": sp,
                    "hp": hd.get("hp", 0),
                    "hp_max": hd.get("hp_max", 0),
                    "level": hd.get("level", 0),
                })

        spell = garen.get("spell")
        cast_target = garen.get("cast_target")
        action_screen = project(cast_target[0], cast_target[1], cx, cy, cz) if cast_target else None

        # Velocity / heading from position delta (0.5s ahead via future lookup)
        # We'll patch these post-loop since we need the next frame's garen pos
        frames_out.append({
            "frame": fi, "gt": round(gt, 3),
            "_garen_world": gp,  # temporary for velocity calc
            "_cam": (cx, cy, cz),  # temporary for projection
            "label": {
                "garen_screen": garen_screen,
                "garen_world": gp,
                "garen_stats": {
                    "hp":       garen.get("hp", 0),
                    "hp_max":   garen.get("hp_max", 0),
                    "gold":     garen.get("gold", 0),
                    "gold_total": garen.get("gold_total", 0),
                    "level":    garen.get("level", 0),
                    "kills":    garen.get("kills", 0),
                    "assists":  garen.get("assists", 0),
                },
                "visible_heroes": visible,
                "action": {"type": classify_spell(spell), "spell": spell, "screen": action_screen},
            },
        })

    # ── Post-loop: compute velocity and heading from position deltas ──
    # Look ahead ~10 frames (0.5s game time) to get a stable movement direction
    LOOKAHEAD = 10
    for fi in range(len(frames_out)):
        fd = frames_out[fi]
        if fd.get("label") is None:
            continue
        gp = fd.get("_garen_world")
        if gp is None: continue
        cam = fd.get("_cam")
        # Find the next labeled frame ~LOOKAHEAD ahead
        future_gp = None
        for j in range(fi + 1, min(fi + LOOKAHEAD + 5, len(frames_out))):
            fj = frames_out[j]
            if fj.get("label") and fj.get("_garen_world") is not None:
                if j - fi >= LOOKAHEAD:
                    future_gp = fj["_garen_world"]
                    break
                elif future_gp is None:
                    future_gp = fj["_garen_world"]
        if future_gp is None or gp is None:
            continue
        dx = future_gp[0] - gp[0]
        dz = future_gp[1] - gp[1]
        speed = (dx * dx + dz * dz) ** 0.5
        if speed > 5 and cam:  # moving
            # Project the future position to screen
            heading_screen = project(future_gp[0], future_gp[1], cam[0], cam[1], cam[2])
            fd["label"]["movement"] = {
                "heading_world": [round(future_gp[0], 1), round(future_gp[1], 1)],
                "heading_screen": heading_screen,
                "speed": round(speed, 1),
            }
        else:
            fd["label"]["movement"] = None

    # Clean up temporary fields
    for fd in frames_out:
        fd.pop("_garen_world", None)
        fd.pop("_cam", None)

    # Stats
    n_labeled = sum(1 for f in frames_out if f["label"] is not None)
    act_counts = {}
    for fr in frames_out:
        if fr["label"] is None: continue
        t = fr["label"]["action"]["type"]
        act_counts[t] = act_counts.get(t, 0) + 1

    mem_gaps_ms.sort()
    p50 = mem_gaps_ms[len(mem_gaps_ms)//2] if mem_gaps_ms else 0
    p99 = mem_gaps_ms[int(len(mem_gaps_ms)*0.99)] if mem_gaps_ms else 0

    print(f"  Labeled: {n_labeled}/{n_frames} ({n_unlabeled} unlabeled, {n_cam_fallback} cam fallbacks)", flush=True)
    print(f"  Mem gap p50={p50:.1f}ms p99={p99:.1f}ms", flush=True)
    if n_unlabeled > 0:
        pct = n_unlabeled / n_frames * 100
        print(f"  WARNING: {n_unlabeled} frames ({pct:.1f}%) unlabeled", flush=True)

    labels = {
        "match_id": match_id, "champion": "Garen",
        "garen_team": game_info.get("garen_team"),
        "garen_slot": game_info.get("garen_slot"),
        "fps": FPS, "screen_resolution": [SCREEN_W, SCREEN_H],
        "frame_resolution": [FRAME_SZ, FRAME_SZ],
        "total_frames": len(frames_out),
        "projection": {"fov_v_deg": 40.0, "fov_h_deg": round(math.degrees(FOV_H), 1),
                        "tilt_deg": 56.0, "cam_y": CAM_Y, "cam_z_offset": CAM_Z_OFFSET},
        "action_distribution": act_counts,
        "frames": frames_out,
    }

    with open(os.path.join(game_dir, "labels.json"), "w") as f:
        json.dump(labels, f)
    with open(os.path.join(game_dir, "raw_mem.json"), "w") as f:
        json.dump(mem_data, f)
    if cam_data:
        with open(os.path.join(game_dir, "raw_cam.json"), "w") as f:
            json.dump(cam_data, f)

    # Delete staging PNGs
    print(f"  Cleaning staging dir...", flush=True)
    shutil.rmtree(staging_dir, ignore_errors=True)

    post_end = time.time()
    print(f"  Done: {n_labeled} frames, {act_counts}  ({post_end - post_start:.0f}s)", flush=True)

    return {
        "post_start": post_start, "post_end": post_end,
        "wall_time": round(post_end - post_start, 1),
        "n_frames": n_frames, "n_labeled": n_labeled,
        "n_unlabeled": n_unlabeled, "n_cam_fallback": n_cam_fallback,
        "mem_gap_p50": round(p50, 2), "mem_gap_p99": round(p99, 2),
        "action_distribution": act_counts,
    }

# ═══════════════════════════════════════════════════════════════
# Post-Process Worker (separate process)
# ═══════════════════════════════════════════════════════════════
def _postprocess_worker(queue, log_path):
    """Long-lived worker that consumes post-process jobs from queue."""
    while True:
        job = queue.get()
        if job is None:
            break
        try:
            stats = post_process(**job)
            if stats:
                log_event(log_path, phase="post", match_id=job["match_id"], **stats)
        except Exception:
            traceback.print_exc()

# ═══════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════
def process_game(game_info, force_patch=False, post_queue=None):
    match_id = game_info["match_id"]
    game_id = game_info["game_id"]
    team = game_info["garen_team"]
    slot = game_info["garen_slot"]
    duration = game_info.get("duration", 1800)
    key = cam_key_for(team, slot)

    out_dir = os.path.join(OUTPUT_BASE, match_id)
    if os.path.exists(os.path.join(out_dir, "labels.json")):
        print(f"SKIP {match_id} (already done)", flush=True)
        return True

    # Per-game log redirect
    log_dir = os.path.join(OUTPUT_BASE, "logs")
    os.makedirs(log_dir, exist_ok=True)
    game_log = open(os.path.join(log_dir, f"{match_id}.log"), "w")
    old_stdout = sys.stdout
    sys.stdout = game_log

    try:
        print(f"\n{'='*60}", flush=True)
        print(f"GAME: {match_id}  team={team} slot={slot} key={key} dur={duration}s", flush=True)
        print(f"{'='*60}", flush=True)

        # Pass 1: Memory + Camera
        mem_data, cam_data, scrape_stats = pass1_scrape(game_id, key, duration, force_patch=force_patch)
        if len(mem_data) < 100:
            print(f"FAIL {match_id}: only {len(mem_data)} mem samples", flush=True)
            return False
        log_event(JSONL_PATH, phase="scrape", match_id=match_id, **scrape_stats)

        real_duration = mem_data[-1]["gt"]
        print(f"  Real game duration: {real_duration:.0f}s ({real_duration/60:.1f}min)", flush=True)

        # Pass 2: Video recording → per-game staging dir
        staging_dir = os.path.join(TEMP_BASE, match_id)
        n_frames, record_stats = pass2_record(game_id, key, real_duration, staging_dir)
        if n_frames < 100:
            print(f"FAIL {match_id}: only {n_frames} frames", flush=True)
            return False
        log_event(JSONL_PATH, phase="record", match_id=match_id, **record_stats)

        # Post-process: queue to worker or run inline
        job = {
            "match_id": match_id, "mem_data": mem_data, "cam_data": cam_data,
            "game_info": game_info, "staging_dir": staging_dir,
        }
        if post_queue is not None:
            post_queue.put(job)
            print(f"  Post-process queued", flush=True)
        else:
            stats = post_process(**job)
            if stats:
                log_event(JSONL_PATH, phase="post", match_id=match_id, **stats)
            if not stats:
                print(f"FAIL {match_id}: post-processing failed", flush=True)
                return False

        print(f"\nOK {match_id}: {n_frames} frames recorded", flush=True)
        return True

    finally:
        sys.stdout = old_stdout
        game_log.close()
        # Echo result to terminal
        last_line = open(os.path.join(log_dir, f"{match_id}.log")).readlines()[-1].strip()
        print(f"  [{match_id}] {last_line}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Garen Replay Data Pipeline")
    parser.add_argument("--manifest", help="Path to manifest JSON")
    parser.add_argument("--index", type=int, help="Process single game at index")
    parser.add_argument("--batch", action="store_true", help="Process all games")
    parser.add_argument("--start", type=int, default=0, help="Start index for batch")
    parser.add_argument("--game-id", help="Numeric game ID")
    parser.add_argument("--match-id", help="Full match ID (e.g. NA1_5528069928)")
    parser.add_argument("--team", choices=["blue", "red"])
    parser.add_argument("--slot", type=int)
    parser.add_argument("--duration", type=int, default=1800)
    parser.add_argument("--force-patch", action="store_true")
    parser.add_argument("--no-overlap", action="store_true",
                        help="Run post-process inline instead of in worker process")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run 3x3 scenario test for overlap contention")
    args = parser.parse_args()

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    # ── Benchmark mode ──
    if args.benchmark:
        if not args.game_id:
            parser.error("--benchmark requires --game-id, --team, --slot")
        gid = args.game_id
        mid = args.match_id or f"NA1_{gid}"
        gi = {"match_id": mid, "game_id": gid, "garen_team": args.team,
              "garen_slot": args.slot, "duration": 120}

        scenarios = [
            ("serial (no overlap)", True),
            ("overlap, cv2 unrestricted", False),
            ("overlap, cv2 capped", False),
        ]
        print("=== BENCHMARK: 3 scenarios, 1 game each ===", flush=True)
        for name, no_overlap in scenarios:
            # Clean previous
            out_dir = os.path.join(OUTPUT_BASE, mid)
            if os.path.exists(out_dir): shutil.rmtree(out_dir)

            print(f"\n--- Scenario: {name} ---", flush=True)
            if no_overlap:
                process_game(gi, force_patch=args.force_patch, post_queue=None)
            else:
                q = mp.Queue()
                log_p = JSONL_PATH
                worker = mp.Process(target=_postprocess_worker, args=(q, log_p))
                worker.start()
                process_game(gi, force_patch=args.force_patch, post_queue=q)
                q.put(None); worker.join()
        print("\n=== Check pipeline.jsonl for timing comparison ===", flush=True)
        return

    # ── Normal mode ──
    post_queue = None
    worker = None
    if not args.no_overlap:
        post_queue = mp.Queue()
        worker = mp.Process(target=_postprocess_worker, args=(post_queue, JSONL_PATH))
        worker.start()

    try:
        if args.manifest:
            manifest = json.load(open(args.manifest))
            games = manifest.get("matches", [])
            games_ok = [g for g in games if g.get("version", "").startswith("16.")]

            if args.index is not None:
                process_game(games_ok[args.index], force_patch=args.force_patch, post_queue=post_queue)
            elif args.batch:
                ok = fail = consecutive_alarms = 0
                for i in range(args.start, len(games_ok)):
                    g = games_ok[i]
                    print(f"\n[{i+1}/{len(games_ok)}] {g['match_id']}", flush=True)
                    try:
                        if process_game(g, force_patch=args.force_patch, post_queue=post_queue):
                            ok += 1; consecutive_alarms = 0
                        else:
                            fail += 1; consecutive_alarms += 1
                    except Exception:
                        traceback.print_exc(); fail += 1; consecutive_alarms += 1
                        kill_game()
                    if consecutive_alarms >= 3:
                        print(f"\n*** 3 consecutive failures — stopping batch ***", flush=True)
                        break
                print(f"\nBATCH: {ok} ok, {fail} fail / {len(games_ok)} total", flush=True)
            else:
                parser.error("--manifest needs --index or --batch")

        elif args.game_id or args.match_id:
            if not args.team or args.slot is None:
                parser.error("Need --team and --slot")
            gid = args.game_id or args.match_id.split("_")[-1]
            mid = args.match_id or f"NA1_{gid}"
            process_game({
                "match_id": mid, "game_id": gid,
                "garen_team": args.team, "garen_slot": args.slot,
                "duration": args.duration,
            }, force_patch=args.force_patch, post_queue=post_queue)
        else:
            parser.error("Provide --manifest, --game-id, or --benchmark")

    finally:
        if post_queue is not None:
            post_queue.put(None)
        if worker is not None:
            worker.join(timeout=600)


if __name__ == "__main__":
    main()
