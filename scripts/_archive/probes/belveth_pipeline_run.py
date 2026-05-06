"""Thin orchestrator that reuses scripts/pipeline.py for the proper recording
path (kill game → LCU relaunch → /replay/recording PNG dump), but targets
Bel'Veth (blue jungle, cam key '2') and only the first 60s.

No modification to pipeline.py. Just imports its functions.
Produces:
  C:\\tmp\\belveth_frames\\*.png   (720p PNGs from League's own recorder)
  C:\\tmp\\belveth_run2\\samples.json
  C:\\tmp\\belveth_run2\\overlay.mp4
"""
import os, sys, time, json, glob, threading, bisect, math
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all pipeline machinery
import pipeline as P  # reuses kill_game, launch_replay, focus_game, lock_camera,
                      # replay_get/post, Mem, find_league_pid, find_module_base

import ctypes, struct

GAME_ID      = "5545727197"   # NA1-5545727197 Bel'Veth replay
MATCH_ID     = "NA1_5545727197"
CHAMPION     = "Belveth"
CAM_KEY      = "2"            # blue-side slot 1 = jungle
DURATION_GT  = 60.0           # first minute of game-time
SPEED        = 2.0            # play at 2x
REC_FPS      = 20             # 20 PNG fps / 2x = 10 game-fps (lighter I/O, survives 4K)

STAGING  = r"C:\tmp\belveth_frames"
OUT_DIR  = r"C:\tmp\belveth_run2"
FOCUSED_HERO_RVA = 0x1E13490
POSITION_OFF     = 0x200
CHAMP_NAME_OFF   = 0x4360

def main():
    os.makedirs(STAGING, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    for f in glob.glob(os.path.join(STAGING, "*.png")): os.remove(f)

    # Kill + relaunch
    P.kill_game()
    time.sleep(3)
    print("Launching replay...")
    if not P.launch_replay(GAME_ID):
        print("ABORT: replay launch failed"); return
    time.sleep(5)

    pid = P.find_league_pid()
    base, mod_size = P.find_module_base(pid)
    print(f"PID={pid} base=0x{base:X} mod_size=0x{mod_size:X}")
    m = P.Mem(pid)

    # Seek to 0 + select champion
    P.replay_post("/replay/render", {"interfaceAll": False, "selectionName": CHAMPION})
    time.sleep(0.5)
    # Camera lock via schtasks /IT — runs in session 1 so keypress reaches game.
    import subprocess
    def cam_lock_via_schtasks(key):
        subprocess.run(['schtasks', '/End', '/TN', 'LockCam'], capture_output=True)
        subprocess.run(['schtasks', '/Delete', '/TN', 'LockCam', '/F'], capture_output=True)
        subprocess.run(['schtasks', '/Create', '/TN', 'LockCam',
                        '/TR', f'cmd.exe /c python C:\\Users\\daniz\\lock_cam_once.py {key}',
                        '/SC', 'ONCE', '/ST', '23:59', '/IT', '/F'], capture_output=True)
        subprocess.run(['schtasks', '/Run', '/TN', 'LockCam'], capture_output=True)
        time.sleep(2.0)  # give task time to run

    hero = m.u64(base + FOCUSED_HERO_RVA)
    if not hero:
        print("ABORT: focused-hero pointer null"); return
    name = m.string(hero + CHAMP_NAME_OFF) if hasattr(m, 'string') else m.read(hero+CHAMP_NAME_OFF,16).split(b"\x00",1)[0].decode('ascii','replace')
    print(f"focused hero: 0x{hero:X}  name='{name}'")

    # Play at 2x first, lock cam while playing (cam lock requires game not paused)
    P.replay_post("/replay/playback", {"time": 0.5, "speed": SPEED, "paused": False})
    time.sleep(2.0)
    cam_lock_via_schtasks(CAM_KEY)
    time.sleep(1.0)
    # Verify lock took
    try:
        rd = P.replay_get("/replay/render")
        cp = rd["cameraPosition"]
        print(f"cam after lock: ({cp['x']:.0f},{cp['y']:.0f},{cp['z']:.0f}) — expect near Bel'Veth fountain ~(-700, 200)")
    except Exception as e:
        print(f"cam check err: {e}")

    # NOW start recording (without touching focus again)
    rec = P.replay_post("/replay/recording", {
        "recording": True,
        "path": STAGING.replace("\\", "/"),
        "codec": "png",
        "framesPerSecond": REC_FPS,
        "startTime": 0.5,
        "endTime": DURATION_GT + 0.5,
        "enforceFrameRate": True,
    })
    print(f"Recording started: {rec.get('width')}x{rec.get('height')} @ {REC_FPS}fps")

    # Poll memory + camera at high rate
    samples = []
    stop = threading.Event()
    def poll_loop():
        while not stop.is_set():
            t0 = time.time()
            try:
                pb = P.replay_get("/replay/playback")
                rd = P.replay_get("/replay/render")
                hp = m.u64(base + FOCUSED_HERO_RVA) or hero
                bx, by, bz = m.vec3(hp + POSITION_OFF)
                cam = rd["cameraPosition"]
                samples.append({
                    "wall": round(t0, 4), "gt": pb["time"],
                    "bv": [bx, by, bz],
                    "cam": [cam["x"], cam["y"], cam["z"]],
                })
            except Exception:
                pass
            sl = 0.03 - (time.time() - t0)
            if sl > 0: time.sleep(sl)
    poller = threading.Thread(target=poll_loop, daemon=True); poller.start()

    # Wait for recording to finish (2x speed → 30s wall for 60s game)
    max_wait = DURATION_GT / SPEED + 60
    t0 = time.time()
    while time.time() - t0 < max_wait:
        time.sleep(3)
        try:
            r = P.replay_get("/replay/recording")
            rec_on = r.get("recording", False)
            gt = P.replay_get("/replay/playback").get("time", 0)
            print(f"  rec_on={rec_on} gt={gt:.1f}s samples={len(samples)}")
            if not rec_on: break
        except Exception as e:
            if P.find_league_pid() is None:
                print("Game exited"); break
    stop.set(); poller.join(timeout=2)

    # Save samples
    sp = os.path.join(OUT_DIR, "samples.json")
    with open(sp, "w") as f: json.dump(samples, f)
    pngs = sorted(glob.glob(os.path.join(STAGING, "*.png")))
    print(f"\nGot {len(pngs)} PNG frames, {len(samples)} mem/cam samples")

    if not pngs:
        print("No PNG frames captured — recording may have failed"); return

    # ─── Build overlay ───
    import cv2, numpy as np
    first = cv2.imread(pngs[0])
    if first is None:
        print(f"Could not read first PNG: {pngs[0]}"); return
    H, W = first.shape[:2]
    out_video = os.path.join(OUT_DIR, "overlay.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # Playback fps = game-time fps = REC_FPS / SPEED
    play_fps = REC_FPS / SPEED
    vw = cv2.VideoWriter(out_video, fourcc, play_fps, (W, H))

    sample_gts = [s["gt"] for s in samples]
    # Projection params from create_overlay_video.py (empirical for 720p locked cam)
    MAP_VIEW_WIDTH = 12000.0
    TILT_FACTOR = math.cos(math.radians(56.0))
    scale = W / MAP_VIEW_WIDTH

    for i, png in enumerate(pngs):
        frame = cv2.imread(png)
        if frame is None: continue
        # Estimate game time for this frame — League records at framesPerSecond at SPEED,
        # so game_time advances at (REC_FPS / SPEED) per real-second.
        # PNG index i → game_time = startTime + i / (REC_FPS / SPEED)
        gt = 0.5 + i / (REC_FPS / SPEED)
        # Find nearest sample by gt
        idx = bisect.bisect_left(sample_gts, gt) if sample_gts else -1
        if idx < len(samples) and idx >= 0:
            s = samples[min(idx, len(samples)-1)]
            bv, cam = s["bv"], s["cam"]
            # Project world (x, z) relative to camera (locked) → screen pixel
            sx = int((bv[0] - cam[0]) * scale + W / 2)
            sy = int((cam[2] - bv[2]) * scale * TILT_FACTOR + H / 2)
            cv2.circle(frame, (sx, sy), 18, (0,255,0), 2)
            cv2.line(frame, (sx-14,sy), (sx+14,sy), (0,255,0), 2)
            cv2.line(frame, (sx,sy-14), (sx,sy+14), (0,255,0), 2)
            lines = [
                f"gt={gt:6.2f}s  frame={i}",
                f"BelVeth world: ({bv[0]:7.1f}, {bv[1]:6.1f}, {bv[2]:7.1f})",
                f"Camera  world: ({cam[0]:7.1f}, {cam[1]:6.1f}, {cam[2]:7.1f})",
                f"delta bv-cam:  ({bv[0]-cam[0]:+6.1f}, {bv[2]-cam[2]:+6.1f})",
            ]
            for j, line in enumerate(lines):
                cv2.putText(frame, line, (10, 24 + j*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,0), 3, cv2.LINE_AA)
                cv2.putText(frame, line, (10, 24 + j*22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100,255,255), 1, cv2.LINE_AA)
        vw.write(frame)
    vw.release()
    print(f"\nOverlay -> {out_video}")
    print(f"Samples -> {sp}")

if __name__ == "__main__":
    main()
