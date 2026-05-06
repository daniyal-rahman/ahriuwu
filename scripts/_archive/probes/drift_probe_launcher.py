"""Launch a replay, lock the camera to Garen, then exec drift_probe.py."""
import json, base64, ssl, time, urllib.request, sys, subprocess, os
import ctypes, ctypes.wintypes as wt

GAME_ID = "5547184086"
LOCKFILE = r"C:\Riot Games\League of Legends\lockfile"
_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
REPLAY = "https://127.0.0.1:2999"
CAM_KEY = "1"   # Garen on blue team slot 0

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
    req = urllib.request.Request(
        f"{REPLAY}{p}", method="POST",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return r.read()

def api_get(p):
    try:
        with urllib.request.urlopen(f"{REPLAY}{p}", context=_ctx, timeout=2) as r:
            return json.loads(r.read())
    except: return None

# ── SendInput scancodes (DirectInput-compatible) ─────────────────────
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort), ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong), ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]
class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long), ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong), ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong), ("dwExtraInfo", PUL)]
class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong), ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]
class _II(ctypes.Union):
    _fields_ = [("ki", KeyBdInput), ("mi", MouseInput), ("hi", HardwareInput)]
class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong), ("ii", _II)]
INPUT_KEYBOARD = 1
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008
VK_TO_SCAN = {'1':0x02, '2':0x03, '3':0x04, '4':0x05, '5':0x06,
              'q':0x10, 'w':0x11, 'e':0x12, 'r':0x13, 't':0x14}

def send_scan(scan, up=False):
    extra = ctypes.c_ulong(0)
    flags = KEYEVENTF_SCANCODE | (KEYEVENTF_KEYUP if up else 0)
    ki = KeyBdInput(wVk=0, wScan=scan, dwFlags=flags, time=0, dwExtraInfo=ctypes.pointer(extra))
    inp = Input(type=INPUT_KEYBOARD, ii=_II(ki=ki))
    ctypes.windll.user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

def tap(scan):
    send_scan(scan, up=False); time.sleep(0.06); send_scan(scan, up=True)

def find_game_hwnd():
    u32 = ctypes.windll.user32
    hwnds = []
    def cb(h, _):
        if not u32.IsWindowVisible(h): return True
        cls = ctypes.create_unicode_buffer(256)
        u32.GetClassNameW(h, cls, 256)
        if cls.value == "RiotWindowClass":
            hwnds.append(h)
        return True
    u32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)(cb), 0)
    return hwnds[0] if hwnds else None

def focus_game():
    u32 = ctypes.windll.user32; k = ctypes.windll.kernel32
    h = find_game_hwnd()
    if not h: return False
    u32.SystemParametersInfoW(0x2001, 0, None, 0)
    fg = u32.GetForegroundWindow()
    ft = u32.GetWindowThreadProcessId(fg, None)
    ct = k.GetCurrentThreadId()
    u32.AttachThreadInput(ct, ft, True)
    u32.keybd_event(0x12, 0, 0, 0); u32.keybd_event(0x12, 0, 2, 0)
    u32.ShowWindow(h, 9); u32.BringWindowToTop(h); u32.SetForegroundWindow(h)
    u32.AttachThreadInput(ct, ft, False)
    time.sleep(0.3)
    return True

def lock_cam():
    scan = VK_TO_SCAN[CAM_KEY]
    focus_game(); time.sleep(0.3)
    tap(scan); time.sleep(0.28); tap(scan)
    time.sleep(0.5)

def launch():
    print(f"launching replay {GAME_ID}...", flush=True)
    try: lcu_post(f"/lol-replays/v1/rofls/{GAME_ID}/watch", {"componentType": "replay"})
    except Exception as e: print(f"  LCU: {e}", flush=True)
    for i in range(120):
        gs = api_get("/liveclientdata/gamestats")
        if gs and gs.get("gameTime") is not None:
            print(f"  loaded ({i*2}s, gt={gs['gameTime']:.1f})", flush=True)
            return True
        time.sleep(2)
    return False

def setup_with_lock():
    """pause → select → unpause → focus+lock → focus+lock (the recipe)."""
    print("running cam lock recipe...", flush=True)
    # 1. pause
    replay_post("/replay/playback", {"paused": True, "time": 120.0})
    time.sleep(0.5)
    # 2. selectionName
    replay_post("/replay/render", {"interfaceAll": True, "selectionName": "Garen"})
    time.sleep(0.5)
    # 3. unpause at 1x for cleanest timing
    replay_post("/replay/playback", {"paused": False, "speed": 1.0})
    time.sleep(1.0)
    # 4-5. double-lock
    lock_cam()
    lock_cam()
    # Verify
    cp = api_get("/replay/render") or {}
    cam = cp.get("cameraPosition", {})
    print(f"  cam after lock: ({cam.get('x',0):.0f}, {cam.get('y',0):.0f}, {cam.get('z',0):.0f})", flush=True)
    return True

if __name__ == "__main__":
    if not launch(): print("ERR launch"); sys.exit(1)
    if not setup_with_lock(): print("ERR setup"); sys.exit(1)
    sys.argv = ["drift_probe.py"]
    import importlib.util
    spec = importlib.util.spec_from_file_location("drift_probe",
            os.path.join(os.path.dirname(__file__), "drift_probe.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.exit(mod.main())
