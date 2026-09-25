"""Record first 60s of Bel'Veth replay + memory reads, then build overlay video.

Composes existing pieces:
  - focus_game() pattern from fresh_record.py
  - focused-hero pointer (module+0x1E13490) from belveth_pos_sample.py  -> live Bel'Veth pos
  - mss + cv2 video capture
  - map_to_screen() projection from create_overlay_video.py

One script. Produces:
  C:\\tmp\\belveth_run\\video.avi
  C:\\tmp\\belveth_run\\frame_data.json   (wall, gt, belveth_xyz, cam_xyz per ~50ms tick)
  C:\\tmp\\belveth_run\\overlay.mp4       (video with per-frame HUD + belveth marker)
"""
import ctypes, ctypes.wintypes as wt, struct, subprocess, sys, json, time, os, threading
import ssl, urllib.request, base64, math
import numpy as np
import cv2, mss
from pynput.keyboard import Controller as KbController
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

OUT_DIR        = r"C:\tmp\belveth_run"
os.makedirs(OUT_DIR, exist_ok=True)
DURATION_GT    = 60.0    # 60 seconds of GAME TIME
RECORD_FPS     = 20
FOCUSED_HERO_PTR_RVA = 0x1E13490
POSITION_OFF   = 0x200
CHAMP_NAME_OFF = 0x4360
SELECTION_NAME = "Belveth"
CAM_KEY        = '2'     # blue-side jungle

# mss capture: 1280x720 matches pipeline.py's assumed resolution
CAPTURE_W, CAPTURE_H = 1280, 720

# --- win32 primitives ---
_k = ctypes.WinDLL("kernel32", use_last_error=True)
class ME32(ctypes.Structure):
    _fields_ = [("dwSize",wt.DWORD),("a",wt.DWORD),("pid",wt.DWORD),
                ("b",wt.DWORD),("c",wt.DWORD),
                ("modBase",ctypes.POINTER(ctypes.c_byte)),("modSize",wt.DWORD),
                ("hMod",wt.HMODULE),("szMod",ctypes.c_char*256),
                ("szPath",ctypes.c_char*260)]
def find_pid():
    r = subprocess.run(['tasklist','/FI','IMAGENAME eq League of Legends.exe','/FO','CSV','/NH'],
                       capture_output=True, text=True)
    for l in r.stdout.strip().split('\n'):
        if 'league' in l.lower(): return int(l.strip('"').split('","')[1])
def find_base(pid):
    snap = _k.CreateToolhelp32Snapshot(0x18, pid)
    me = ME32(); me.dwSize = ctypes.sizeof(ME32)
    if _k.Module32First(snap, ctypes.byref(me)):
        while True:
            if b'league' in me.szMod.lower():
                b = ctypes.cast(me.modBase, ctypes.c_void_p).value
                _k.CloseHandle(snap); return b, me.modSize
            if not _k.Module32Next(snap, ctypes.byref(me)): break
    _k.CloseHandle(snap); return None, None

class Mem:
    def __init__(self, pid): self.h = _k.OpenProcess(0x0410, False, pid)
    def read(self, a, sz):
        buf = ctypes.create_string_buffer(sz); n = ctypes.c_size_t(0)
        ok = _k.ReadProcessMemory(self.h, ctypes.c_void_p(a), buf, sz, ctypes.byref(n))
        return buf.raw[:n.value] if ok else b""
    def u64(self, a): d = self.read(a, 8); return struct.unpack("<Q", d)[0] if len(d)==8 else None
    def vec3(self, a): d = self.read(a, 12); return struct.unpack("<fff", d) if len(d)==12 else (0,0,0)

_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
def api_get(ep):
    with urllib.request.urlopen(f"https://127.0.0.1:2999{ep}", context=_ctx, timeout=2) as r:
        return json.loads(r.read())
def api_post(ep, data):
    req = urllib.request.Request(f"https://127.0.0.1:2999{ep}",
        data=json.dumps(data).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r: return json.loads(r.read())

# --- focus+lock from fresh_record.py pattern ---
def focus_game():
    user32 = ctypes.windll.user32
    hwnds = []
    def cb(hwnd, _):
        n = user32.GetWindowTextLengthW(hwnd)
        if n > 0:
            buf = ctypes.create_unicode_buffer(n + 1)
            user32.GetWindowTextW(hwnd, buf, n + 1)
            if "league of legends" in buf.value.lower(): hwnds.append(hwnd)
        return True
    user32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, wt.HWND, wt.LPARAM)(cb), 0)
    if not hwnds: return None
    hwnd = hwnds[0]
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
    return hwnd

def lock_camera(key):
    kb = KbController()
    kb.press(key); time.sleep(0.05); kb.release(key)
    time.sleep(0.15)
    kb.press(key); time.sleep(0.05); kb.release(key)
    time.sleep(0.3)

# --- projection (from create_overlay_video.py) ---
MAP_VIEW_WIDTH_DEFAULT = 14830.0 * (1280 / 2160)  # scale for 720p-ish view; tune
# Empirically for cam-locked replay at 1280x720: one pixel ≈ map_width / screen_w
# The default from create_overlay_video is map_view_width=6500 (tuned for 352x352 crop)
# For a full 1280x720 view we use ~12000 map units visible horizontally
DEFAULT_MAP_VIEW_WIDTH = 12000
CAMERA_TILT_DEG = 56.0

def map_to_screen(mx, mz, cx, cz, screen_w, screen_h, map_view_width=DEFAULT_MAP_VIEW_WIDTH):
    scale = screen_w / map_view_width
    sx = (mx - cx) * scale + screen_w / 2
    tilt = math.cos(math.radians(CAMERA_TILT_DEG))
    sy = (cz - mz) * scale * tilt + screen_h / 2
    return int(sx), int(sy)

# --- main ---
def main():
    pid = find_pid(); base, _ = find_base(pid); m = Mem(pid)
    print(f"PID={pid} base=0x{base:X}")

    # Ensure replay is at gt=0, paused
    print("\nSeeking to gt=0 paused")
    api_post("/replay/playback", {"time": 0.5, "speed": 1.0, "paused": True})
    for _ in range(30):
        time.sleep(0.3)
        st = api_get("/replay/playback")
        if not st["seeking"] and st["paused"]: break
    # Set selection name & lock camera
    print(f"Selecting {SELECTION_NAME} and locking camera (key '{CAM_KEY}')")
    api_post("/replay/render", {"interfaceAll": True, "selectionName": SELECTION_NAME})
    time.sleep(0.5)
    focus_game(); time.sleep(0.3)
    lock_camera(CAM_KEY)
    time.sleep(0.5)

    # Verify focused-hero pointer resolves to Bel'Veth
    hero = m.u64(base + FOCUSED_HERO_PTR_RVA)
    if not hero:
        print("ERR: focused-hero pointer null after lock"); return
    name = m.read(hero + CHAMP_NAME_OFF, 16).split(b"\x00",1)[0].decode('ascii','replace')
    print(f"Focused hero: 0x{hero:X} name='{name}'")
    if SELECTION_NAME.lower() not in name.lower():
        print(f"WARN: locked hero is '{name}' not Bel'Veth — lock may have failed");

    # (VideoWriter created after capture region is known)

    # Find capture region. Prefer the game window but fall back to primary monitor.
    focus_game()
    time.sleep(0.2)
    # Simple: primary monitor (mss.monitors[1]) — grabs whatever is foregrounded.
    with mss.mss() as _probe:
        mon = dict(_probe.monitors[1])  # {left, top, width, height}
    # Override output res to match primary monitor so video matches capture
    CAPTURE_W_ACTUAL = mon["width"]
    CAPTURE_H_ACTUAL = mon["height"]
    print(f"Capture region (primary monitor): {mon}")

    # Prepare video writer at ACTUAL resolution
    video_path = os.path.join(OUT_DIR, "video.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    vw = cv2.VideoWriter(video_path, fourcc, RECORD_FPS, (CAPTURE_W_ACTUAL, CAPTURE_H_ACTUAL))
    if not vw.isOpened():
        print("ERR: VideoWriter failed to open"); return

    # Shared data
    stop_evt = threading.Event()
    frame_data = []  # list of {frame_idx, wall, gt, belveth_xyz, cam_xyz}

    def poll_thread():
        """Poll memory + replay render at ~50 Hz. Data is picked up by capture loop
        each frame via the most-recent sample."""
        last_sample = {}
        while not stop_evt.is_set():
            t0 = time.time()
            try:
                pb = api_get("/replay/playback"); gt = pb["time"]
                rd = api_get("/replay/render")
                cam = rd["cameraPosition"]
                hp = m.u64(base + FOCUSED_HERO_PTR_RVA) or hero
                bx, by, bz = m.vec3(hp + POSITION_OFF)
                last_sample.update({"wall": round(t0,4), "gt": gt,
                    "bvx": bx, "bvy": by, "bvz": bz,
                    "cx": cam["x"], "cy": cam["y"], "cz": cam["z"]})
                poll_thread.last = dict(last_sample)
            except Exception:
                pass
            sl = 0.02 - (time.time() - t0)
            if sl > 0: time.sleep(sl)
    poll_thread.last = None
    pt_th = threading.Thread(target=poll_thread, daemon=True); pt_th.start()

    # Unpause and capture
    print(f"\nStarting capture for {DURATION_GT}s game-time...")
    api_post("/replay/playback", {"speed": 1.0, "paused": False})
    # Create mss INSIDE the loop thread (mss is not thread-safe, and we want
    # captures in the main thread here)
    sct = mss.mss()
    frame_idx = 0
    capture_start_wall = time.time()
    last_frame_wall = capture_start_wall
    frame_interval = 1.0 / RECORD_FPS
    end_gt = None

    try:
        while True:
            # Time-step to the next frame
            now = time.time()
            sleep_for = (last_frame_wall + frame_interval) - now
            if sleep_for > 0: time.sleep(sleep_for)
            last_frame_wall = time.time()

            # Grab frame
            img = np.array(sct.grab(mon))
            img = img[:, :, :3]  # BGRA -> BGR
            if img.shape[1] != CAPTURE_W_ACTUAL or img.shape[0] != CAPTURE_H_ACTUAL:
                img = cv2.resize(img, (CAPTURE_W_ACTUAL, CAPTURE_H_ACTUAL))
            vw.write(img)

            # Snapshot latest poll data
            s = poll_thread.last
            rec = {"frame": frame_idx, "wall": round(last_frame_wall, 4)}
            if s:
                rec.update({"gt": s["gt"], "bv": [s["bvx"], s["bvy"], s["bvz"]],
                            "cam": [s["cx"], s["cy"], s["cz"]]})
                if end_gt is None and s["gt"] >= DURATION_GT:
                    end_gt = s["gt"]
                    print(f"Reached gt={end_gt:.2f}, stopping")
                    break
                if frame_idx % 20 == 0:
                    print(f"  frame={frame_idx} gt={s['gt']:.2f} bv=({s['bvx']:.0f},{s['bvz']:.0f})")
            frame_data.append(rec)
            frame_idx += 1
            # safety bound
            if time.time() - capture_start_wall > DURATION_GT + 30:
                print("Timeout — stopping capture"); break
    finally:
        stop_evt.set()
        vw.release()
        api_post("/replay/playback", {"speed": 1.0, "paused": True})

    # Save frame data
    fd_path = os.path.join(OUT_DIR, "frame_data.json")
    with open(fd_path, "w") as f: json.dump(frame_data, f)
    print(f"\nSaved {frame_idx} frames -> {video_path}")
    print(f"Saved frame_data -> {fd_path}")

    # ─── Build overlay ───
    print("\nBuilding overlay...")
    overlay_path = os.path.join(OUT_DIR, "overlay.mp4")
    cap = cv2.VideoCapture(video_path)
    fourcc_out = cv2.VideoWriter_fourcc(*"mp4v")
    ovw = cv2.VideoWriter(overlay_path, fourcc_out, RECORD_FPS, (CAPTURE_W_ACTUAL, CAPTURE_H_ACTUAL))
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break
        rec = frame_data[idx] if idx < len(frame_data) else None
        if rec and "cam" in rec:
            gt = rec["gt"]; bv = rec["bv"]; cam = rec["cam"]
            # Project Bel'Veth world pos to screen (cam locked on her → roughly center)
            sx, sy = map_to_screen(bv[0], bv[2], cam[0], cam[2], CAPTURE_W_ACTUAL, CAPTURE_H_ACTUAL)
            # Draw green cross at projected Bel'Veth position
            cv2.line(frame, (sx-12, sy), (sx+12, sy), (0,255,0), 2)
            cv2.line(frame, (sx, sy-12), (sx, sy+12), (0,255,0), 2)
            cv2.circle(frame, (sx, sy), 16, (0,255,0), 2)
            # HUD text top-left
            lines = [
                f"gt={gt:6.2f}s  frame={idx}",
                f"BelVeth world: ({bv[0]:7.1f}, {bv[1]:6.1f}, {bv[2]:7.1f})",
                f"Camera  world: ({cam[0]:7.1f}, {cam[1]:6.1f}, {cam[2]:7.1f})",
                f"Delta bv-cam : ({bv[0]-cam[0]:+6.1f}, {bv[2]-cam[2]:+6.1f})",
            ]
            for i, line in enumerate(lines):
                cv2.putText(frame, line, (10, 24 + i*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame, line, (10, 24 + i*22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,255,255), 1, cv2.LINE_AA)
        ovw.write(frame)
        idx += 1
    cap.release(); ovw.release()
    print(f"Overlay -> {overlay_path}")
    print(f"\nDone. {frame_idx} frames, {len(frame_data)} data records.")

if __name__ == "__main__":
    main()
