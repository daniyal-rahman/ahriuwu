"""A/B test of camera-lock methods, using PNG pixel-diff as ground truth.

For each method: kill game -> relaunch -> seek to a moment where Bel'Veth is
walking -> apply the lock -> record 4 game-seconds of frames -> compute mean
absolute pixel-diff between first and last frames.

A STATIC cam (not locked) will have very low diff (only small foreground motion).
A CAM FOLLOWING the champion will have HIGH diff (entire scene shifts).

Methods tested:
  0. baseline — no lock (cam stays fountain)
  1. selection only
  2. cameraAttached=true
  3. cameraLockX/Y/Z=true
  4. cameraMode='follow'
  5. schtasks keyboard double-tap
  6. schtasks keyboard + all API flags
"""
import os, sys, time, json, glob, shutil, subprocess
import ssl, urllib.request, base64
import ctypes, struct
import ctypes.wintypes as wt
import numpy as np
import cv2
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)

GAME_ID = "5545727197"
STAGING_ROOT = r"C:\tmp\camtest"
LOCKFILE = r"C:\Riot Games\League of Legends\lockfile"
CAM_KEY = "2"
SEEK_TO_GT = 30.0   # Bel'Veth is walking through jungle around this time
RECORD_SEC_GT = 4.0 # 4 game-seconds at 2x = 2s wall
SPEED = 2.0

_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
def api_get(ep, timeout=3):
    with urllib.request.urlopen(f"https://127.0.0.1:2999{ep}", context=_ctx, timeout=timeout) as r:
        return json.loads(r.read())
def api_post(ep, data, timeout=5):
    req = urllib.request.Request(f"https://127.0.0.1:2999{ep}",
        data=json.dumps(data).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_ctx, timeout=timeout) as r: return json.loads(r.read())

def kill_game():
    subprocess.run(['taskkill','/F','/IM','League of Legends.exe'],
                   capture_output=True, text=True, timeout=10)
    time.sleep(3)

def launch_replay(game_id, timeout=120):
    """Use the proven helper that knows how to retry LCU + wait for game load."""
    try:
        r = subprocess.run(['python', r'C:\Users\daniz\launch_replay_only.py', str(game_id)],
                           capture_output=True, text=True, timeout=timeout)
        print(f"  launch stdout tail: {r.stdout.strip().splitlines()[-1] if r.stdout.strip() else '(empty)'}", flush=True)
        # Wait for live data
        t0 = time.time()
        while time.time() - t0 < 30:
            try: api_get("/liveclientdata/gamestats"); return True
            except: time.sleep(2)
        return False
    except Exception as e:
        print(f"  launch exception: {e}", flush=True)
        return False

def seek_and_ready(gt):
    api_post("/replay/playback", {"time": gt, "speed": 1.0, "paused": True})
    for _ in range(30):
        time.sleep(0.3)
        st = api_get("/replay/playback")
        if not st["seeking"] and st["paused"]: break

def schtasks_cam_lock(key):
    subprocess.run(['schtasks', '/End', '/TN', 'LockCam'], capture_output=True)
    subprocess.run(['schtasks', '/Delete', '/TN', 'LockCam', '/F'], capture_output=True)
    subprocess.run(['schtasks', '/Create', '/TN', 'LockCam',
                    '/TR', f'cmd.exe /c python C:\\Users\\daniz\\lock_cam_once.py {key}',
                    '/SC', 'ONCE', '/ST', '23:59', '/IT', '/F'], capture_output=True)
    subprocess.run(['schtasks', '/Run', '/TN', 'LockCam'], capture_output=True)
    time.sleep(2.0)

def apply_method(method):
    if method == 0:
        pass  # baseline: do nothing
    elif method == 1:
        api_post("/replay/render", {"selectionName": "Belveth"})
    elif method == 2:
        api_post("/replay/render", {"selectionName": "Belveth", "cameraAttached": True})
    elif method == 3:
        api_post("/replay/render", {"selectionName": "Belveth",
                                     "cameraLockX": True, "cameraLockY": True, "cameraLockZ": True})
    elif method == 4:
        api_post("/replay/render", {"selectionName": "Belveth", "cameraMode": "follow"})
    elif method == 5:
        api_post("/replay/render", {"selectionName": "Belveth"})
        time.sleep(0.3)
        schtasks_cam_lock(CAM_KEY)
    elif method == 6:
        api_post("/replay/render", {"selectionName": "Belveth",
                                     "cameraAttached": True,
                                     "cameraLockX": True, "cameraLockY": True, "cameraLockZ": True,
                                     "cameraMode": "follow"})
        time.sleep(0.3)
        schtasks_cam_lock(CAM_KEY)

def record_and_diff(out_dir, speed=SPEED, rec_sec_gt=RECORD_SEC_GT):
    """Record `rec_sec_gt` game-seconds starting from the current game_time,
    write PNGs to out_dir, then compute mean abs diff first vs last frame."""
    os.makedirs(out_dir, exist_ok=True)
    for f in glob.glob(os.path.join(out_dir, "**", "*.png"), recursive=True):
        os.remove(f)
    st = api_get("/replay/playback")
    start_gt = st["time"]
    api_post("/replay/recording", {
        "recording": True,
        "path": out_dir.replace("\\","/"),
        "codec": "png",
        "framesPerSecond": 10,   # few frames, manageable
        "startTime": start_gt,
        "endTime": start_gt + rec_sec_gt,
        "enforceFrameRate": True,
    })
    # Unpause to trigger recording
    api_post("/replay/playback", {"speed": speed, "paused": False})
    # Wait for recording to complete
    t0 = time.time()
    while time.time() - t0 < rec_sec_gt / speed + 30:
        time.sleep(1.5)
        try:
            r = api_get("/replay/recording")
            if not r.get("recording"): break
        except: pass
    api_post("/replay/playback", {"speed": 1.0, "paused": True})

    pngs = sorted(glob.glob(os.path.join(out_dir, "**", "*.png"), recursive=True))
    if len(pngs) < 2:
        return {"n": len(pngs), "diff": None, "note": "insufficient frames"}
    first = cv2.imread(pngs[0], cv2.IMREAD_GRAYSCALE)
    last = cv2.imread(pngs[-1], cv2.IMREAD_GRAYSCALE)
    if first is None or last is None:
        return {"n": len(pngs), "diff": None, "note": "read fail"}
    # downscale for speed & noise immunity
    first = cv2.resize(first, (400, 225))
    last = cv2.resize(last, (400, 225))
    diff = np.mean(np.abs(first.astype(int) - last.astype(int)))
    return {"n": len(pngs), "diff": round(float(diff), 2),
            "first": os.path.basename(pngs[0]), "last": os.path.basename(pngs[-1])}

def main():
    method_names = {
        0: "baseline (no lock)",
        1: "selectionName only",
        2: "cameraAttached=true",
        3: "cameraLockX/Y/Z=true",
        4: "cameraMode='follow'",
        5: "schtasks keyboard double-tap",
        6: "schtasks + all API flags",
    }
    methods = int(os.environ.get("CAM_METHOD", "-1"))
    if methods < 0:
        runs = list(method_names.keys())
    else:
        runs = [methods]

    results = []
    for method in runs:
        print(f"\n================ METHOD {method}: {method_names[method]} ================", flush=True)
        try:
            kill_game()
            if not launch_replay(GAME_ID):
                results.append({"method": method, "name": method_names[method], "n": 0, "diff": None, "note": "launch failed"})
                print("  LAUNCH FAILED"); continue
            time.sleep(3)
            seek_and_ready(SEEK_TO_GT)
            apply_method(method)
            time.sleep(1.0)
            out_dir = os.path.join(STAGING_ROOT, f"method_{method}")
            if os.path.isdir(out_dir): shutil.rmtree(out_dir)
            res = record_and_diff(out_dir)
        except Exception as e:
            import traceback; traceback.print_exc()
            res = {"n": 0, "diff": None, "note": f"exception: {type(e).__name__}: {e}"}
        res["method"] = method; res["name"] = method_names[method]
        results.append(res)
        print(f"  result: n_frames={res.get('n')}  pixel_diff={res.get('diff')}  note={res.get('note','')}")

    # Summary
    print("\n\n================ SUMMARY (high diff = cam followed champ) ================")
    print(f"{'method':>2}  {'name':<32}  {'n':>3}  {'pixel_diff':>10}")
    for r in results:
        d = r.get('diff'); d_s = f"{d:>10.2f}" if d is not None else f"{'-':>10}"
        print(f"{r['method']:>2}  {r['name']:<32}  {r.get('n',0):>3}  {d_s}  {r.get('note','')}")
    with open(r"C:\tmp\camtest\summary.json", "w") as f: json.dump(results, f, indent=2)

if __name__ == "__main__":
    main()
