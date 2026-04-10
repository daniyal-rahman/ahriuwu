"""
Garen Replay Data Pipeline — 2 Pass
=====================================
Pass 1: Memory scrape (50Hz) + camera API poll (93Hz persistent HTTPS), 2x speed
        → determines real game duration from game time stall
        → cam samples timestamped via wall→gt interpolation from mem thread
Pass 2: Record video (720p PNG, 40fps enforced at 2x = 20fps game-time)
        → uses real duration from pass 1 as endTime
Post:   Resize 720→352², build labels.json, delete originals

Runs on Windows with:
  - League client open (for LCU replay launch)
  - Vanguard disabled (for ReadProcessMemory)
  - game.cfg: Width=1280, Height=720, WindowMode=1, EnableReplayApi=1

Usage:
    # Single game (from manifest):
    python pipeline.py --manifest E:\\garen_manifest_300.json --index 0

    # Batch (all games):
    python pipeline.py --manifest E:\\garen_manifest_300.json --batch

    # Single game (manual):
    python pipeline.py --game-id 5528069928 --match-id NA1_5528069928 --team red --slot 5 --duration 1800

Output per game:  E:\\replay_data\\{match_id}\\
    frames\\  → 352×352 PNGs at 20fps
    labels.json → per-frame labels (720p screen coords)

Memory offsets: patch 26.7, module size 0x207C000
"""
import argparse, base64, bisect, ctypes, ctypes.wintypes as wt
import glob, http.client, json, math, os, struct, subprocess, sys
import threading, time, traceback
import urllib.request, urllib.error, ssl
from pynput.keyboard import Controller as KbController

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ═══════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════
SCREEN_W, SCREEN_H = 1280, 720
FRAME_SZ = 352
FPS = 20

# Camera projection (locked to Garen, empirically measured)
CAM_Y = 1912.0
CAM_Z_OFFSET = -1292.0  # cam_z = garen_z + offset (camera is south of champion)
FLOOR_Y = 52.0          # ground-level height for champion feet
TILT = math.radians(56.0)
FOV_V = math.radians(40.0)
FOV_H = 2 * math.atan(math.tan(FOV_V / 2) * SCREEN_W / SCREEN_H)
TAN_H, TAN_V = math.tan(FOV_H / 2), math.tan(FOV_V / 2)
COS_T, SIN_T = math.cos(TILT), math.sin(TILT)

# ── Memory offsets (patch 26.7, module size 0x207C000) ──
# Update these together when the patch changes.
EXPECTED_MOD_SIZE = 0x207C000

OFFSETS = {
    # RVAs (relative to module base)
    "hero_array":     0x1DBEEE8,  # ptr → array of 10 hero pointers
    "game_time":      0x1DCD1E0,  # f32, confirmed via ibrahimcelik

    # GameObject struct (relative to hero pointer)
    "position":       0x25C,      # Vec3 (x, y, z)
    "champion_name":  0x4328,     # ASCII string
    "active_spell":   0x3120,     # ptr → ActiveSpellCast

    # ActiveSpellCast struct
    "spell_info":     0x008,      # ptr → SpellInfo
    "cast_target":    0x0DC,      # Vec3 — target position

    # SpellInfo struct
    "spell_name_ptr": 0x28,       # ptr → ptr → spell name string
}

OUTPUT_BASE  = r"E:\replay_data"
TEMP_PNG_DIR = r"E:\tmp\_pipeline"
LOCKFILE     = r"C:\Riot Games\League of Legends\lockfile"

_ctx = ssl.create_default_context()
_ctx.check_hostname = False; _ctx.verify_mode = ssl.CERT_NONE
_k = ctypes.windll.kernel32
_kb = KbController()

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
    """Verify the known GameTime RVA reads a sane value. Returns RVA or None."""
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
    """Move mouse to center of game window and click to clear any UI focus."""
    user32 = ctypes.windll.user32
    hwnd = find_game_hwnd()
    if not hwnd: return
    rect = wt.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    cx = (rect.left + rect.right) // 2
    cy = (rect.top + rect.bottom) // 2
    user32.SetCursorPos(cx, cy)
    time.sleep(0.1)
    user32.mouse_event(0x0002, 0, 0, 0, 0)  # left down
    time.sleep(0.05)
    user32.mouse_event(0x0004, 0, 0, 0, 0)  # left up
    time.sleep(0.2)

def lock_camera(key):
    click_center()
    _kb.press(key); time.sleep(0.05); _kb.release(key)
    time.sleep(0.15)
    _kb.press(key); time.sleep(0.05); _kb.release(key)
    time.sleep(0.3)

def cam_key_for(team, slot):
    if team == "blue": return str(slot + 1)         # '1'-'5'
    return "qwert"[slot - 5]                         # 'q'-'t'

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
    if not name: return "idle"
    nl = name.lower()
    if "attack" in nl: return "attack"
    if "recall" in nl: return "recall"
    return "ability"

# ═══════════════════════════════════════════════════════════════
# Pipeline Utilities
# ═══════════════════════════════════════════════════════════════
def kill_game():
    os.system('taskkill /F /IM "League of Legends.exe" >nul 2>&1')
    time.sleep(3)

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

# ═══════════════════════════════════════════════════════════════
# Pass 2: Video Recording
# ═══════════════════════════════════════════════════════════════
def pass2_record(game_id, cam_key, duration):
    """Record full game as 720p PNGs at 2x speed (40fps enforced = 20fps game-time). Returns frame count."""
    print("\n--- Pass 2: Video Recording (2x, 40fps enforced) ---", flush=True)
    kill_game()
    if not launch_replay(game_id):
        return 0
    time.sleep(5)

    # Disable HUD, select Garen
    replay_post("/replay/render", {"interfaceAll": False, "selectionName": "Garen"})
    time.sleep(1)

    # Lock camera (game starts paused at t=0)
    focus_game(); time.sleep(0.5)
    lock_camera(cam_key)
    time.sleep(1)
    print(f"  Camera locked (key={cam_key})", flush=True)

    # Unpause to 2x
    replay_post("/replay/playback", {"speed": 2.0})
    time.sleep(0.5)

    # Re-lock while playing
    focus_game()
    lock_camera(cam_key)
    time.sleep(0.5)

    # Clean temp dir
    os.makedirs(TEMP_PNG_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(TEMP_PNG_DIR, "**", "*.png"), recursive=True):
        os.remove(f)

    # Start recording — full game
    rec = replay_post("/replay/recording", {
        "recording": True,
        "path": TEMP_PNG_DIR.replace("\\", "/"),
        "codec": "png",
        "framesPerSecond": FPS * 2,  # 40 wall-fps at 2x = 20 game-fps
        "startTime": 1.0,
        "endTime": duration + 60,
        "enforceFrameRate": True,
    })
    print(f"  Recording started: {rec.get('width')}x{rec.get('height')}", flush=True)

    # Re-lock after recording seek
    time.sleep(1.5)
    focus_game(); time.sleep(0.3)
    lock_camera(cam_key)
    print(f"  Re-locked after recording start", flush=True)

    # Wait for recording to finish — duration is known from memory pass
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
        print(f"  Recording timeout ({max_wait}s) — frames may be truncated", flush=True)

    pngs = sorted(glob.glob(os.path.join(TEMP_PNG_DIR, "**", "*.png"), recursive=True))
    print(f"  Frames recorded: {len(pngs)}", flush=True)
    kill_game()
    return len(pngs)

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
        for name, info in hero_ptrs.items():
            hp = info["ptr"]
            pos = m.vec3(hp + o["position"])
            if not pos: continue
            entry = {"pos": [round(pos[0], 1), round(pos[2], 1)]}
            # Spell detection for Garen only
            if name == "Garen":
                sp = m.u64(hp + o["active_spell"])
                if sp and sp > 0x10000:
                    info = m.u64(sp + o["spell_info"])
                    if info and info > 0x10000:
                        np_ = m.u64(info + o["spell_name_ptr"])
                        if np_ and np_ > 0x10000:
                            sn = m.string(np_)
                            if sn and len(sn) > 2:
                                entry["spell"] = sn
                                ct = m.vec3(sp + o["cast_target"])
                                if ct and abs(ct[0]) < 20000:
                                    entry["cast_target"] = [round(ct[0], 1), round(ct[2], 1)]
            heroes[name] = entry
        results.append({"wall": time.perf_counter(), "gt": round(gt, 3), "heroes": heroes})
        elapsed = time.perf_counter() - tick
        if elapsed < interval: time.sleep(interval - elapsed)

def _cam_loop(stop, results):
    """Poll camera via persistent HTTPS connection (~93Hz).
    Timestamps are wall-clock (perf_counter); converted to game time at merge."""
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
    """Memory scrape (50Hz) + camera poll (93Hz) at 2x speed. Returns (mem_data, cam_data)."""
    print("\n--- Pass 1: Memory + Camera Scrape (2x) ---", flush=True)
    kill_game()
    if not launch_replay(game_id):
        return [], []
    time.sleep(5)

    # Init memory reader
    pid = find_league_pid()
    if not pid:
        print("  No game PID", flush=True); kill_game(); return [], []
    base, mod_size = find_module_base(pid)
    if not base:
        print("  No module base", flush=True); kill_game(); return [], []
    if mod_size != EXPECTED_MOD_SIZE:
        msg = f"  Module size 0x{mod_size:X} != expected 0x{EXPECTED_MOD_SIZE:X} — patch changed?"
        if force_patch:
            print(f"  WARNING: {msg} (--force-patch, continuing)", flush=True)
        else:
            print(f"  ABORT: {msg}", flush=True)
            print(f"  Update OFFSETS for the new patch or pass --force-patch to override.", flush=True)
            kill_game(); return [], []

    m = Mem(pid)
    mz = m.read(base, 2)
    if mz != b'MZ':
        print("  RPM verify failed", flush=True); m.close(); kill_game(); return [], []
    print(f"  Memory: PID={pid} base=0x{base:X}", flush=True)

    gt_rva = verify_game_time(m, base)
    if not gt_rva:
        print(f"  ABORT: GameTime RVA 0x{OFFSETS['game_time']:X} reads garbage — offsets are wrong for this build", flush=True)
        m.close(); kill_game(); return [], []
    print(f"  GameTime: 0x{gt_rva:X} = {m.f32(base + gt_rva):.1f}s", flush=True)

    hero_ptrs = init_heroes(m, base)
    if "Garen" not in hero_ptrs:
        print(f"  Garen not found (heroes: {list(hero_ptrs.keys())})", flush=True)
        m.close(); kill_game(); return [], []
    print(f"  Heroes: {list(hero_ptrs.keys())}", flush=True)

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

    # Start scrape threads
    stop = threading.Event()
    mem_data, cam_data = [], []
    mt = threading.Thread(target=_mem_loop, args=(m, base, hero_ptrs, gt_rva, stop, mem_data), daemon=True)
    ct = threading.Thread(target=_cam_loop, args=(stop, cam_data), daemon=True)
    mt.start(); ct.start()

    # Monitor until game ends
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
    print(f"  Memory: {len(mem_data)} samples, Camera: {len(cam_data)} samples", flush=True)
    if mem_data:
        gt_range = f"{mem_data[0]['gt']:.0f}-{mem_data[-1]['gt']:.0f}s"
        print(f"  GT range: {gt_range}", flush=True)
    m.close(); kill_game()
    return mem_data, cam_data

# ═══════════════════════════════════════════════════════════════
# Post-Processing
# ═══════════════════════════════════════════════════════════════
def post_process(match_id, mem_data, cam_data, game_info):
    """Resize frames to 352², build labels, clean up 720p originals."""
    import cv2

    game_dir = os.path.join(OUTPUT_BASE, match_id)
    frames_dir = os.path.join(game_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    pngs = sorted(glob.glob(os.path.join(TEMP_PNG_DIR, "**", "*.png"), recursive=True))
    n_frames = len(pngs)
    print(f"\n--- Post-Processing: {n_frames} frames ---", flush=True)
    if n_frames == 0:
        print("  ERROR: no frames", flush=True); return None

    # Resize 720p → 352×352
    print(f"  Resizing to {FRAME_SZ}x{FRAME_SZ}...", flush=True)
    n_resize_fail = 0
    for i, p in enumerate(pngs):
        img = cv2.imread(p)
        if img is None:
            n_resize_fail += 1
            # Copy raw file as fallback so frame index stays aligned
            import shutil
            shutil.copy2(p, os.path.join(frames_dir, f"{i:06d}.png"))
            continue
        out = cv2.resize(img, (FRAME_SZ, FRAME_SZ), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(frames_dir, f"{i:06d}.png"), out)
        if (i + 1) % 5000 == 0:
            print(f"    {i+1}/{n_frames}", flush=True)
    if n_resize_fail > 0:
        print(f"  WARNING: {n_resize_fail} frames failed cv2.imread — raw copies used", flush=True)

    # Build labels
    print("  Building labels...", flush=True)
    start_gt = 1.0  # recording starts at gt=1.0

    # ── Build wall→gt piecewise-linear map from mem samples ──
    mem_walls = [s["wall"] for s in mem_data]
    mem_gts   = [s["gt"]   for s in mem_data]

    def wall_to_gt(w):
        """Interpolate game time from wall-clock time using mem samples."""
        i = bisect.bisect_right(mem_walls, w)
        if i == 0 or i >= len(mem_walls):
            return None  # outside mem coverage — drop
        w0, w1 = mem_walls[i-1], mem_walls[i]
        g0, g1 = mem_gts[i-1], mem_gts[i]
        if w1 == w0:
            return g0
        t = (w - w0) / (w1 - w0)
        return g0 + t * (g1 - g0)

    # ── Assign game times to cam samples via interpolation ──
    cam_with_gt = []
    n_cam_dropped = 0
    for cs in cam_data:
        gt = wall_to_gt(cs["wall"])
        if gt is None:
            n_cam_dropped += 1
            continue
        cam_with_gt.append({"gt": gt, "cx": cs["cx"], "cy": cs["cy"], "cz": cs["cz"]})
    cam_with_gt.sort(key=lambda s: s["gt"])
    cam_gt_keys = [s["gt"] for s in cam_with_gt]

    if n_cam_dropped > 0:
        print(f"  Cam samples outside mem range: {n_cam_dropped} dropped", flush=True)
    print(f"  Cam samples with interpolated gt: {len(cam_with_gt)}", flush=True)

    # ── Sort mem by gt for frame lookup ──
    mem_sorted = sorted(mem_data, key=lambda s: s["gt"])
    mem_gt_keys = [s["gt"] for s in mem_sorted]

    MAX_MEM_GAP = 0.1   # 100ms — mem at 50Hz, so gap >100ms = real problem
    MAX_CAM_GAP = 0.1   # 100ms — cam at 93Hz, same logic

    def _nearest(arr, keys, gt):
        i = bisect.bisect_right(keys, gt)
        best, best_gap = None, float("inf")
        for j in (i-1, i):
            if 0 <= j < len(arr):
                gap = abs(arr[j]["gt"] - gt)
                if gap < best_gap:
                    best, best_gap = arr[j], gap
        return best, best_gap

    # ── Per-frame label building ──
    frames_out = []
    n_unlabeled = 0
    n_cam_fallback = 0

    for fi in range(n_frames):
        gt = start_gt + fi / FPS

        bm, mem_gap = _nearest(mem_sorted, mem_gt_keys, gt)

        if mem_gap > MAX_MEM_GAP:
            # No usable mem data — emit null label, skip in training
            n_unlabeled += 1
            frames_out.append({
                "frame": fi, "gt": round(gt, 3), "label": None,
            })
            continue

        heroes = bm.get("heroes", {})
        garen = heroes.get("Garen", {})
        gp = garen.get("pos", [0, 0])

        # Camera: prefer API data, fallback to derived from Garen pos
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
                visible.append({"name": name, "screen": sp})

        spell = garen.get("spell")
        cast_target = garen.get("cast_target")
        action_screen = project(cast_target[0], cast_target[1], cx, cy, cz) if cast_target else None

        frames_out.append({
            "frame": fi,
            "gt": round(gt, 3),
            "label": {
                "garen_screen": garen_screen,
                "visible_heroes": visible,
                "action": {
                    "type": classify_spell(spell),
                    "spell": spell,
                    "screen": action_screen,
                },
            },
        })

    # Stats — only from labeled frames
    n_labeled = sum(1 for f in frames_out if f["label"] is not None)
    act_counts = {}
    for fr in frames_out:
        if fr["label"] is None:
            continue
        t = fr["label"]["action"]["type"]
        act_counts[t] = act_counts.get(t, 0) + 1

    print(f"  Labeled: {n_labeled}/{n_frames} ({n_unlabeled} unlabeled, {n_cam_fallback} cam fallbacks)", flush=True)
    if n_unlabeled > 0:
        pct = n_unlabeled / n_frames * 100
        print(f"  WARNING: {n_unlabeled} frames ({pct:.1f}%) have no label — investigate mem gaps", flush=True)

    labels = {
        "match_id": match_id,
        "champion": "Garen",
        "garen_team": game_info.get("garen_team"),
        "garen_slot": game_info.get("garen_slot"),
        "fps": FPS,
        "screen_resolution": [SCREEN_W, SCREEN_H],
        "frame_resolution": [FRAME_SZ, FRAME_SZ],
        "total_frames": len(frames_out),
        "projection": {
            "fov_v_deg": 40.0, "fov_h_deg": round(math.degrees(FOV_H), 1),
            "tilt_deg": 56.0, "cam_y": CAM_Y, "cam_z_offset": CAM_Z_OFFSET,
        },
        "action_distribution": act_counts,
        "frames": frames_out,
    }

    with open(os.path.join(game_dir, "labels.json"), "w") as f:
        json.dump(labels, f)

    # Save raw scrape data (debugging)
    with open(os.path.join(game_dir, "raw_mem.json"), "w") as f:
        json.dump(mem_data, f)
    if cam_data:
        with open(os.path.join(game_dir, "raw_cam.json"), "w") as f:
            json.dump(cam_data, f)

    print(f"  Frames: {len(frames_out)}", flush=True)
    print(f"  Actions: {act_counts}", flush=True)

    # Delete 720p originals
    print(f"  Deleting {n_frames} temp PNGs...", flush=True)
    for f in pngs:
        os.remove(f)

    print(f"  Output: {game_dir}", flush=True)
    return labels

# ═══════════════════════════════════════════════════════════════
# Main Orchestrator
# ═══════════════════════════════════════════════════════════════
def process_game(game_info, force_patch=False):
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

    print(f"\n{'='*60}", flush=True)
    print(f"GAME: {match_id}  team={team} slot={slot} key={key} dur={duration}s", flush=True)
    print(f"{'='*60}", flush=True)

    # Pass 1: Memory + Camera (determines real game duration)
    mem_data, cam_data = pass1_scrape(game_id, key, duration, force_patch=force_patch)
    if len(mem_data) < 100:
        print(f"FAIL {match_id}: only {len(mem_data)} mem samples", flush=True)
        return False

    # Real game duration from memory data
    real_duration = mem_data[-1]["gt"]
    print(f"  Real game duration: {real_duration:.0f}s ({real_duration/60:.1f}min)", flush=True)

    # Pass 2: Video recording (using real duration as endTime)
    n_frames = pass2_record(game_id, key, real_duration)
    if n_frames < 100:
        print(f"FAIL {match_id}: only {n_frames} frames recorded", flush=True)
        return False

    # Post-process
    labels = post_process(match_id, mem_data, cam_data, game_info)
    if not labels:
        print(f"FAIL {match_id}: post-processing failed", flush=True)
        return False

    print(f"\nOK {match_id}: {labels['total_frames']} frames", flush=True)
    return True


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
    parser.add_argument("--force-patch", action="store_true",
                        help="Continue even if module size doesn't match expected patch")
    args = parser.parse_args()

    os.makedirs(OUTPUT_BASE, exist_ok=True)

    if args.manifest:
        manifest = json.load(open(args.manifest))
        games = manifest.get("matches", [])
        # Filter to current patch only (downloadable replays)
        games_ok = [g for g in games if g.get("version", "").startswith("16.")]

        if args.index is not None:
            process_game(games_ok[args.index], force_patch=args.force_patch)
        elif args.batch:
            ok = fail = skip = 0
            for i in range(args.start, len(games_ok)):
                g = games_ok[i]
                print(f"\n[{i+1}/{len(games_ok)}] {g['match_id']}", flush=True)
                try:
                    if process_game(g, force_patch=args.force_patch): ok += 1
                    else: fail += 1
                except Exception:
                    traceback.print_exc(); fail += 1; kill_game()
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
        }, force_patch=args.force_patch)
    else:
        parser.error("Provide --manifest or --game-id")


if __name__ == "__main__":
    main()
