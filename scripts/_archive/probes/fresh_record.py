"""Fresh recording: 1x PNG + camera poll, then 0.25x camera poll. No reuse."""
import urllib.request, ssl, json, time, os, base64, threading, glob, sys

LOG = r"C:\tmp\fresh_log.txt"
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG, 'a') as f: f.write(line + "\n")
with open(LOG, 'w') as f: f.write("")

ctx = ssl.create_default_context()
ctx.check_hostname = False; ctx.verify_mode = ssl.CERT_NONE

def api_get(ep):
    with urllib.request.urlopen(urllib.request.Request(
        f"https://127.0.0.1:2999{ep}"), context=ctx, timeout=5) as r:
        return json.loads(r.read())

def api_post(ep, data):
    with urllib.request.urlopen(urllib.request.Request(
        f"https://127.0.0.1:2999{ep}",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"}), context=ctx, timeout=5) as r:
        return json.loads(r.read())

def focus_game():
    import ctypes, ctypes.wintypes
    user32 = ctypes.windll.user32; kernel32 = ctypes.windll.kernel32
    rl = []
    def cb(hwnd, _):
        l = user32.GetWindowTextLengthW(hwnd)
        if l > 0:
            buf = ctypes.create_unicode_buffer(l + 1)
            user32.GetWindowTextW(hwnd, buf, l + 1)
            if "league of legends" in buf.value.lower(): rl.append(hwnd)
        return True
    user32.EnumWindows(ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.wintypes.HWND,
                                           ctypes.wintypes.LPARAM)(cb), 0)
    if rl:
        hwnd = rl[0]
        user32.SystemParametersInfoW(0x2001, 0, None, 0)
        fg = user32.GetForegroundWindow()
        ft = user32.GetWindowThreadProcessId(fg, None)
        ct = kernel32.GetCurrentThreadId()
        user32.AttachThreadInput(ct, ft, True)
        user32.keybd_event(0x12, 0, 0, 0); user32.keybd_event(0x12, 0, 2, 0)
        user32.ShowWindow(hwnd, 9); user32.BringWindowToTop(hwnd)
        user32.SetForegroundWindow(hwnd)
        user32.AttachThreadInput(ct, ft, False)
        time.sleep(0.3)

def lock_camera():
    from pynput.keyboard import Controller
    kb = Controller()
    kb.press('q'); time.sleep(0.05); kb.release('q')
    time.sleep(0.15)
    kb.press('q'); time.sleep(0.05); kb.release('q')
    time.sleep(0.3)

def launch_replay():
    os.system('taskkill /F /IM "League of Legends.exe" >nul 2>&1')
    time.sleep(3)
    lf = open(r"C:\Riot Games\League of Legends\lockfile").read().strip().split(":")
    auth = base64.b64encode(f"riot:{lf[3]}".encode()).decode()
    urllib.request.urlopen(urllib.request.Request(
        f"https://127.0.0.1:{lf[2]}/lol-replays/v1/rofls/5528069928/watch",
        method="POST", data=json.dumps({"componentType": "replay"}).encode(),
        headers={"Authorization": f"Basic {auth}", "Content-Type": "application/json"}
    ), context=ctx, timeout=10)
    for i in range(90):
        try: api_get("/liveclientdata/gamestats"); return True
        except: time.sleep(3)
    return False

def setup_camera():
    """Seek to 800, select Garen, lock camera."""
    focus_game()
    api_post("/replay/render", {"interfaceAll": False, "selectionName": "Garen"})
    time.sleep(1)
    api_post("/replay/playback", {"speed": 0.0, "time": 799})
    time.sleep(4)
    focus_game()
    lock_camera()
    time.sleep(1)

def poll_camera(stop_event, results, rate=0.03):
    """Poll camera in a thread."""
    while not stop_event.is_set():
        try:
            pb = api_get("/replay/playback")
            rd = api_get("/replay/render")
            results.append({
                "gt": round(pb["time"], 4),
                "cx": round(rd["cameraPosition"]["x"], 2),
                "cy": round(rd["cameraPosition"]["y"], 2),
                "cz": round(rd["cameraPosition"]["z"], 2),
                "fov": rd.get("fieldOfView", 0),
                "tilt": rd.get("cameraRotation", {}).get("y", 0),
                "sel": rd.get("selectionName", ""),
                "wall": round(time.time(), 4),
            })
        except: pass
        time.sleep(rate)

try:
    # ============================================================
    # PASS 1: 1x speed PNG recording + camera polling
    # ============================================================
    log("=== PASS 1: 1x PNG recording + polling ===")
    launch_replay()
    log("Game loaded")
    time.sleep(5)
    setup_camera()

    # Start playback and re-lock
    api_post("/replay/playback", {"speed": 1.0})
    time.sleep(0.3)
    focus_game()
    lock_camera()
    time.sleep(0.5)

    rd = api_get("/replay/render")
    log(f"Camera: ({rd['cameraPosition']['x']:.0f}, {rd['cameraPosition']['z']:.0f}) "
        f"sel='{rd.get('selectionName','')}' fov={rd.get('fieldOfView')}")

    # Prepare PNG dir
    png_dir = r"E:\tmp\fresh_png"
    os.makedirs(png_dir, exist_ok=True)
    for f in glob.glob(os.path.join(png_dir, "*", "*.png")): os.remove(f)
    for d in glob.glob(os.path.join(png_dir, "*")):
        if os.path.isdir(d):
            try: os.rmdir(d)
            except: pass

    # Start poller
    cam_1x = []
    stop1 = threading.Event()
    t1 = threading.Thread(target=poll_camera, args=(stop1, cam_1x, 0.025), daemon=True)
    t1.start()
    time.sleep(0.3)

    # Start recording
    rec = api_post("/replay/recording", {
        "recording": True, "path": "E:/tmp/fresh_png", "codec": "png",
        "framesPerSecond": 20, "startTime": 800, "endTime": 830,
        "enforceFrameRate": True,
    })
    log(f"Recording started: {rec.get('width')}x{rec.get('height')}")
    time.sleep(0.5)
    focus_game()

    # Wait for recording to finish
    t0 = time.time()
    while True:
        time.sleep(2)
        try:
            if not api_get("/replay/recording").get("recording", False):
                log(f"Recording done, elapsed={time.time()-t0:.0f}s")
                break
        except: pass
        if time.time()-t0 > 300: log("Timeout"); break

    stop1.set(); t1.join(timeout=3)
    png_count = len(glob.glob(os.path.join(png_dir, "*", "*.png")))
    log(f"Pass 1: {png_count} PNGs, {len(cam_1x)} camera samples")
    if cam_1x:
        gts = [c['gt'] for c in cam_1x]
        log(f"  GT: {min(gts):.2f}-{max(gts):.2f}")
        log(f"  CX: {min(c['cx'] for c in cam_1x):.0f}-{max(c['cx'] for c in cam_1x):.0f}")
        log(f"  CZ: {min(c['cz'] for c in cam_1x):.0f}-{max(c['cz'] for c in cam_1x):.0f}")

    with open(r"E:\tmp\fresh_cam_1x.json", 'w') as f:
        json.dump(cam_1x, f)

    # Encode PNGs to video
    import subprocess
    sub_dir = glob.glob(os.path.join(png_dir, "*"))[0] if glob.glob(os.path.join(png_dir, "*")) else ""
    if sub_dir and os.path.isdir(sub_dir):
        subprocess.run([
            "ffmpeg", "-y", "-framerate", "20", "-start_number", "1",
            "-i", os.path.join(sub_dir, "%06d.png"),
            "-c:v", "libx264", "-preset", "fast", "-crf", "18",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart",
            r"E:\tmp\fresh_video.mp4"
        ], capture_output=True, timeout=120)
        log("Video encoded")

    # ============================================================
    # PASS 2: 0.25x speed camera polling only (same replay)
    # ============================================================
    log("\n=== PASS 2: 0.25x polling only ===")
    os.system('taskkill /F /IM "League of Legends.exe" >nul 2>&1')
    time.sleep(3)
    launch_replay()
    log("Game loaded")
    time.sleep(5)
    setup_camera()

    # Start at 0.25x
    api_post("/replay/playback", {"speed": 0.25})
    time.sleep(0.3)
    focus_game()
    lock_camera()
    time.sleep(0.5)

    rd = api_get("/replay/render")
    log(f"Camera: ({rd['cameraPosition']['x']:.0f}, {rd['cameraPosition']['z']:.0f}) "
        f"sel='{rd.get('selectionName','')}'")

    cam_025x = []
    stop2 = threading.Event()
    t2 = threading.Thread(target=poll_camera, args=(stop2, cam_025x, 0.015), daemon=True)
    t2.start()

    # Wait until game_time > 831
    t0 = time.time()
    while True:
        try:
            pb = api_get("/replay/playback")
            if pb["time"] > 831:
                log(f"Reached gt={pb['time']:.1f}")
                break
        except: pass
        time.sleep(0.5)
        if time.time()-t0 > 300: log("Timeout"); break

    stop2.set(); t2.join(timeout=3)
    log(f"Pass 2: {len(cam_025x)} camera samples")
    if cam_025x:
        gts = [c['gt'] for c in cam_025x]
        log(f"  GT: {min(gts):.2f}-{max(gts):.2f}")
        log(f"  CX: {min(c['cx'] for c in cam_025x):.0f}-{max(c['cx'] for c in cam_025x):.0f}")
        log(f"  CZ: {min(c['cz'] for c in cam_025x):.0f}-{max(c['cz'] for c in cam_025x):.0f}")

    with open(r"E:\tmp\fresh_cam_025x.json", 'w') as f:
        json.dump(cam_025x, f)

    os.system('taskkill /F /IM "League of Legends.exe" >nul 2>&1')
    log("\n=== ALL DONE ===")

except Exception as e:
    log(f"ERROR: {e}")
    import traceback; log(traceback.format_exc())
