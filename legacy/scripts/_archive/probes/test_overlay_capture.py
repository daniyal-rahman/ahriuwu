"""Quick overlay test: record 30s of video (HUD ON) + mem scrape + cam poll.
Uses pipeline.py infrastructure directly. Outputs to C:\\tmp\\overlay_test\\

Usage (run via schtasks /IT on Windows):
    python scripts/test_overlay_capture.py
"""
import ctypes, ctypes.wintypes as wt, glob, http.client, json, math, os, ssl
import struct, subprocess, sys, threading, time
sys.stdout.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)

# ── Borrow from pipeline.py ──
_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
_k = ctypes.windll.kernel32
from pynput.keyboard import Controller as KbController
_kb = KbController()

def replay_post(ep, d):
    return json.loads(urllib.request.urlopen(urllib.request.Request(
        f"https://127.0.0.1:2999{ep}", data=json.dumps(d).encode(),
        headers={"Content-Type":"application/json"}), context=_ctx, timeout=5).read())
def replay_get(ep):
    return json.loads(urllib.request.urlopen(urllib.request.Request(
        f"https://127.0.0.1:2999{ep}"), context=_ctx, timeout=5).read())
import urllib.request

def find_game_hwnd():
    user32 = ctypes.windll.user32; hwnds = []
    def cb(hwnd, _):
        n = user32.GetWindowTextLengthW(hwnd)
        if n > 0:
            buf = ctypes.create_unicode_buffer(n+1); user32.GetWindowTextW(hwnd, buf, n+1)
            if "league of legends" in buf.value.lower(): hwnds.append(hwnd)
        return True
    user32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)(cb), 0)
    return hwnds[0] if hwnds else None

def focus_game():
    user32 = ctypes.windll.user32; hwnd = find_game_hwnd()
    if not hwnd: return False
    user32.SystemParametersInfoW(0x2001, 0, None, 0)
    fg = user32.GetForegroundWindow()
    ft = user32.GetWindowThreadProcessId(fg, None); ct = _k.GetCurrentThreadId()
    user32.AttachThreadInput(ct, ft, True)
    user32.keybd_event(0x12,0,0,0); user32.keybd_event(0x12,0,2,0)
    user32.ShowWindow(hwnd,9); user32.BringWindowToTop(hwnd); user32.SetForegroundWindow(hwnd)
    user32.AttachThreadInput(ct, ft, False); time.sleep(0.3)
    return True

def click_center():
    user32 = ctypes.windll.user32; hwnd = find_game_hwnd()
    if not hwnd: return
    rect = wt.RECT(); user32.GetWindowRect(hwnd, ctypes.byref(rect))
    cx=(rect.left+rect.right)//2; cy=(rect.top+rect.bottom)//2
    user32.SetCursorPos(cx,cy); time.sleep(0.1)
    user32.mouse_event(0x0002,0,0,0,0); time.sleep(0.05)
    user32.mouse_event(0x0004,0,0,0,0); time.sleep(0.2)

def lock_camera(key):
    click_center()
    _kb.press(key); time.sleep(0.05); _kb.release(key)
    time.sleep(0.15)
    _kb.press(key); time.sleep(0.05); _kb.release(key)
    time.sleep(0.3)

class Mem:
    def __init__(self, pid): self.h = _k.OpenProcess(0x0410, False, pid)
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok and n.value == sz else None
    def u64(self, a): d=self.read(a,8); return struct.unpack('<Q',d)[0] if d else None
    def u32(self, a): d=self.read(a,4); return struct.unpack('<I',d)[0] if d else None
    def f32(self, a): d=self.read(a,4); return struct.unpack('<f',d)[0] if d else None
    def vec3(self, a):
        d=self.read(a,12); return struct.unpack('<fff',d) if d and len(d)==12 else None
    def string(self, a, n=80):
        d=self.read(a,n)
        if not d: return None
        try: return d.split(b'\x00')[0].decode('ascii')
        except: return None

def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'], capture_output=True, text=True)
    for line in r.stdout.strip().split('\n'):
        if 'league' in line.lower(): return int(line.strip('"').split('","')[1])
    return None
def find_base(pid):
    class ME(ctypes.Structure):
        _fields_=[("dwSize",ctypes.c_ulong),("th32ModuleID",ctypes.c_ulong),("th32ProcessID",ctypes.c_ulong),
            ("GlblcntUsage",ctypes.c_ulong),("ProccntUsage",ctypes.c_ulong),("modBaseAddr",ctypes.POINTER(ctypes.c_byte)),
            ("modBaseSize",ctypes.c_ulong),("hModule",ctypes.c_void_p),("szModule",ctypes.c_char*256),("szExePath",ctypes.c_char*260)]
    snap=_k.CreateToolhelp32Snapshot(0x18,pid); me=ME(); me.dwSize=ctypes.sizeof(ME)
    if _k.Module32First(snap, ctypes.byref(me)):
        while True:
            if b'league' in me.szModule.lower():
                _k.CloseHandle(snap); return ctypes.cast(me.modBaseAddr, ctypes.c_void_p).value
            if not _k.Module32Next(snap, ctypes.byref(me)): break
    return None

# ── Offsets (from pipeline.py) ──
O = {
    "hero_array": 0x1DBEEE8, "game_time": 0x1DCD1E0,
    "position": 0x25C, "hp": 0x1080, "hp_max": 0x10A8,
    "gold": 0x2830, "gold_earned": 0x2858, "level": 0x4D10,
    "name": 0x4328, "waypoint": 0x4220, "active_spell": 0x3120,
    "spell_info": 0x008, "spell_name": 0x28, "vision": 0x55D8,
    "sb_base": 0x30E8, "sb_slots": 0xAE0, "slot_cd": 0x30,
}
LO, HI = 0x10000000, 0x7FFFFFFFFFFF

# ── Config ──
OUT_DIR = r"C:\tmp\overlay_test"
GT_START = 810.0
DURATION_WALL = 30  # seconds real time
CAM_KEY = "q"  # Garen is red team slot 0

# ═══════════════════════════════════════════════════════════════
print("=== Overlay capture test ===", flush=True)
os.makedirs(OUT_DIR, exist_ok=True)

pid = find_pid()
if not pid: print("ERROR: League not running"); sys.exit(1)
base = find_base(pid)
m = Mem(pid)

def get_heroes():
    arr = m.u64(base + O["hero_array"])
    heroes = {}
    for i in range(10):
        hp = m.u64(arr + i*8)
        if hp:
            name = m.string(hp + O["name"])
            if name: heroes[name] = hp
    return heroes

# 1. Seek + setup
print("Seeking to gt=810, enabling HUD, locking camera...", flush=True)
replay_post("/replay/playback", {"time": GT_START, "speed": 0.0, "paused": True})
time.sleep(2)
replay_post("/replay/render", {"interfaceAll": True, "selectionName": "Garen"})
time.sleep(1)
focus_game(); time.sleep(0.5)
lock_camera(CAM_KEY)
time.sleep(1)
print("  Camera locked", flush=True)

# 2. Start recording (PNGs, HUD visible)
png_dir = os.path.join(OUT_DIR, "frames")
os.makedirs(png_dir, exist_ok=True)
for f in glob.glob(os.path.join(png_dir, "*.png")):
    os.remove(f)

replay_post("/replay/playback", {"speed": 2.0, "paused": False})
time.sleep(0.5)
focus_game(); lock_camera(CAM_KEY)
time.sleep(0.5)

rec = replay_post("/replay/recording", {
    "recording": True,
    "path": png_dir.replace("\\", "/"),
    "codec": "png",
    "framesPerSecond": 40,  # 40 real fps = 20 game fps at 2x
    "startTime": GT_START,
    "endTime": GT_START + 120,
    "enforceFrameRate": True,
})
print(f"  Recording started: {rec.get('width')}x{rec.get('height')}", flush=True)
time.sleep(1)
focus_game(); lock_camera(CAM_KEY)

# 3. Parallel mem + cam capture
mem_data, cam_data = [], []
cam_stop = threading.Event()

def cam_loop():
    conn = http.client.HTTPSConnection("127.0.0.1", 2999, context=_ctx, timeout=5)
    while not cam_stop.is_set():
        try:
            wall = time.perf_counter()
            conn.request("GET", "/replay/render")
            d = json.loads(conn.getresponse().read())
            cp = d.get("cameraPosition", {})
            cam_data.append({"wall": wall, "cx": round(cp.get("x",0),1),
                             "cy": round(cp.get("y",0),1), "cz": round(cp.get("z",0),1)})
        except:
            try: conn = http.client.HTTPSConnection("127.0.0.1", 2999, context=_ctx, timeout=5)
            except: pass
        time.sleep(0.005)
ct = threading.Thread(target=cam_loop, daemon=True); ct.start()

print(f"  Scraping mem+cam for {DURATION_WALL}s...", flush=True)
t0 = time.perf_counter()
while time.perf_counter() - t0 < DURATION_WALL:
    tick = time.perf_counter()
    gt = m.f32(base + O["game_time"]) or 0
    heroes = get_heroes()
    sample = {"wall": tick, "gt": round(gt, 3), "heroes": {}}
    for name, hp in heroes.items():
        pos = m.vec3(hp + O["position"])
        if not pos: continue
        wp = m.vec3(hp + O["waypoint"])
        entry = {
            "pos": [round(pos[0],1), round(pos[2],1)],
            "hp": round(m.f32(hp + O["hp"]) or 0, 1),
            "hp_max": round(m.f32(hp + O["hp_max"]) or 0, 1),
            "gold": round(m.f32(hp + O["gold"]) or 0, 0),
            "gold_earned": round(m.f32(hp + O["gold_earned"]) or 0, 0),
            "level": m.u32(hp + O["level"]) or 0,
            "vision": round(m.f32(hp + O["vision"]) or 0, 2),
        }
        if wp and 0 < wp[0] < 16000 and 0 < wp[2] < 16000:
            entry["waypoint"] = [round(wp[0],1), round(wp[2],1)]
        sp = m.u64(hp + O["active_spell"])
        if sp and sp > 0x10000:
            si = m.u64(sp + O["spell_info"])
            if si and si > 0x10000:
                np_ = m.u64(si + O["spell_name"])
                if np_ and np_ > 0x10000:
                    sn = m.string(np_)
                    if sn and len(sn) > 2: entry["spell"] = sn
        sb = hp + O["sb_base"]
        cds = []
        for slot in range(4):
            slot_ptr = m.u64(sb + O["sb_slots"] + slot*8)
            if slot_ptr and LO < slot_ptr < HI:
                cd_exp = m.f32(slot_ptr + O["slot_cd"]) or 0
                remaining = round(cd_exp - gt, 1) if cd_exp > gt else 0
                cds.append(remaining)
            else: cds.append(0)
        entry["cooldowns"] = cds
        sample["heroes"][name] = entry
    mem_data.append(sample)
    elapsed = time.perf_counter() - tick
    if elapsed < 1/30: time.sleep(1/30 - elapsed)

# 4. Stop
cam_stop.set()
replay_post("/replay/playback", {"speed": 0.0, "paused": True})
replay_post("/replay/recording", {"recording": False})
time.sleep(1)

# Add gt to cam via interpolation
import bisect
mem_walls = [s["wall"] for s in mem_data]
mem_gts = [s["gt"] for s in mem_data]
for c in cam_data:
    i = bisect.bisect_left(mem_walls, c["wall"])
    i = max(1, min(i, len(mem_walls)-1))
    w0, w1 = mem_walls[i-1], mem_walls[i]
    g0, g1 = mem_gts[i-1], mem_gts[i]
    frac = (c["wall"] - w0) / (w1 - w0) if w1 != w0 else 0
    c["gt"] = round(g0 + frac*(g1-g0), 3)

pngs = sorted(glob.glob(os.path.join(png_dir, "**", "*.png"), recursive=True))

print(f"\n=== Results ===", flush=True)
print(f"  PNGs: {len(pngs)}", flush=True)
print(f"  Mem:  {len(mem_data)} samples, gt {mem_data[0]['gt']:.1f}-{mem_data[-1]['gt']:.1f}", flush=True)
print(f"  Cam:  {len(cam_data)} samples", flush=True)

# Verify mid-sample
s = mem_data[len(mem_data)//2]
g = s["heroes"].get("Garen", {})
print(f"  Mid: gt={s['gt']:.1f} Garen wp={g.get('waypoint')} hp={g.get('hp')}", flush=True)

with open(os.path.join(OUT_DIR, "mem.json"), "w") as f: json.dump(mem_data, f)
with open(os.path.join(OUT_DIR, "cam.json"), "w") as f: json.dump(cam_data, f)
print(f"\nSaved to {OUT_DIR}", flush=True)
