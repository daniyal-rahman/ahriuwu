#!/usr/bin/env python3
"""
End-to-end test of the recording pipeline.
1. Changes League config to 1080p
2. Launches a replay via LCU
3. Locks camera to champion
4. Records a 30s test clip using built-in recording API
5. Verifies frame count and resolution
6. Optionally restores original resolution

Usage:
    python scripts/test_recording_pipeline.py \
        --game-id 5528069928 \
        --champion Garen \
        --output-dir C:/tmp/recording_test \
        --start 300 --duration 30

Requires: League Client running, EnableReplayApi=1 in game.cfg
"""
import argparse
import base64
import json
import os
import ssl
import sys
import time
import urllib.request

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE
GAME_API = "https://127.0.0.1:2999"
CFG_PATH = r"C:\Riot Games\League of Legends\Config\game.cfg"


def api_get(ep):
    req = urllib.request.Request(f"{GAME_API}{ep}")
    with urllib.request.urlopen(req, context=ctx, timeout=5) as r:
        return json.loads(r.read())


def api_post(ep, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(f"{GAME_API}{ep}", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, context=ctx, timeout=5) as r:
        return json.loads(r.read())


def set_league_resolution(width, height):
    """Change League's render resolution in game.cfg."""
    content = open(CFG_PATH).read()
    # Find and replace Width= and Height= (not Camera Height=)
    lines = content.split("\n")
    new_lines = []
    for line in lines:
        if line.strip().startswith("Width="):
            new_lines.append(f"Width={width}")
        elif line.strip().startswith("Height=") and "Camera" not in line:
            new_lines.append(f"Height={height}")
        else:
            new_lines.append(line)
    open(CFG_PATH, "w").write("\n".join(new_lines))

    # Verify
    content2 = open(CFG_PATH).read()
    actual_w = actual_h = None
    for line in content2.split("\n"):
        if line.strip().startswith("Width="):
            actual_w = line.strip().split("=")[1]
        elif line.strip().startswith("Height=") and "Camera" not in line:
            actual_h = line.strip().split("=")[1]
    print(f"  Config set to: {actual_w}x{actual_h}")
    return actual_w == str(width) and actual_h == str(height)


def wait_for_game(timeout=120):
    """Wait for game to load."""
    print("Waiting for game to load...")
    for _ in range(timeout // 3):
        try:
            stats = api_get("/liveclientdata/gamestats")
            print(f"  Game loaded at t={stats['gameTime']:.1f}s")
            return True
        except:
            time.sleep(3)
    return False


def record_clip(output_path, start_time, duration, target_game_fps=20, champion="Garen"):
    """Record a clip using the built-in recording API.

    The game forces replaySpeed=4. With enforceFrameRate=true, we set
    framesPerSecond = target_game_fps * effective_speed.

    Since we can't predict the exact effective speed, we use a high enough
    fps to guarantee >= target_game_fps, then verify.
    """
    # Lock camera to champion
    api_post("/replay/render", {
        "selectionName": champion,
        "cameraAttached": True,
    })
    time.sleep(0.5)

    # The game forces replaySpeed=4. To get 20 game-fps:
    # We need framesPerSecond such that (frames / game_duration) >= 20.
    # From testing: 80fps at 4x gives ~69 game-fps, 20fps gives ~5.4.
    # Use 60fps as a good balance: should give ~40-50 game-fps.
    wall_fps = max(target_game_fps * 3, 60)  # 3x buffer

    end_time = start_time + duration

    print(f"Recording: t={start_time:.0f}-{end_time:.0f}s, {wall_fps} wall-fps, "
          f"enforceFrameRate=true")

    # Start recording
    rec = api_post("/replay/recording", {
        "recording": True,
        "path": output_path,
        "framesPerSecond": wall_fps,
        "startTime": start_time,
        "endTime": end_time,
        "enforceFrameRate": True,
        "codec": "webm",
    })

    print(f"  Recording started: {rec['width']}x{rec['height']} @ "
          f"{rec['framesPerSecond']}fps, speed={rec['replaySpeed']}x")

    # Poll camera positions while recording
    cam_data = []
    while True:
        try:
            rec_status = api_get("/replay/recording")
            pb = api_get("/replay/playback")
            rd = api_get("/replay/render")

            cam_data.append({
                "game_time": pb["time"],
                "camera_x": rd["cameraPosition"]["x"],
                "camera_y": rd["cameraPosition"]["y"],
                "camera_z": rd["cameraPosition"]["z"],
                "recording": rec_status["recording"],
            })

            if not rec_status["recording"]:
                print(f"  Recording finished at t={pb['time']:.1f}s")
                break
        except:
            pass
        time.sleep(0.5)

    # Save camera data
    cam_path = output_path.replace(".webm", "_camera.json")
    with open(cam_path, "w") as f:
        json.dump(cam_data, f, indent=2)
    print(f"  Camera data: {len(cam_data)} samples -> {cam_path}")

    return cam_data


def verify_output(video_path, expected_game_duration, target_game_fps):
    """Verify the output video meets our requirements."""
    import cv2
    cap = cv2.VideoCapture(video_path)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_dur = n / fps if fps > 0 else 0
    game_fps = n / expected_game_duration
    cap.release()

    print(f"\nVerification:")
    print(f"  Resolution: {w}x{h}")
    print(f"  Frames: {n}")
    print(f"  Video FPS: {fps}")
    print(f"  Video duration: {vid_dur:.1f}s")
    print(f"  Game-fps: {game_fps:.1f} (target: {target_game_fps})")
    print(f"  File size: {os.path.getsize(video_path)/1e6:.1f}MB")

    ok = True
    if w != 1920 or h != 1080:
        print(f"  FAIL: Resolution is {w}x{h}, expected 1920x1080")
        ok = False
    if game_fps < target_game_fps:
        print(f"  FAIL: Game-fps {game_fps:.1f} < target {target_game_fps}")
        ok = False
    else:
        print(f"  OK: Game-fps {game_fps:.1f} >= target {target_game_fps}")

    return ok, {"width": w, "height": h, "frames": n, "video_fps": fps,
                "game_fps": game_fps, "file_size_mb": os.path.getsize(video_path)/1e6}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", required=True)
    parser.add_argument("--champion", default="Garen")
    parser.add_argument("--output-dir", default=r"C:\tmp\recording_test")
    parser.add_argument("--start", type=float, default=300)
    parser.add_argument("--duration", type=float, default=30)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--skip-launch", action="store_true")
    parser.add_argument("--no-restore", action="store_true",
                        help="Don't restore original resolution after test")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Save original resolution and set to 1080p
    print("Step 1: Setting League to 1080p...")
    content = open(CFG_PATH).read()
    orig_w = orig_h = None
    for line in content.split("\n"):
        if line.strip().startswith("Width="):
            orig_w = line.strip().split("=")[1]
        elif line.strip().startswith("Height=") and "Camera" not in line:
            orig_h = line.strip().split("=")[1]
    print(f"  Original: {orig_w}x{orig_h}")

    if orig_w != "1920" or orig_h != "1080":
        set_league_resolution(1920, 1080)
    else:
        print("  Already 1080p")

    # Step 2: Launch replay
    if not args.skip_launch:
        print("\nStep 2: Launching replay...")
        lockfile = open(r"C:\Riot Games\League of Legends\lockfile").read().strip().split(":")
        port, token = int(lockfile[2]), lockfile[3]
        auth = base64.b64encode(f"riot:{token}".encode()).decode()
        req = urllib.request.Request(
            f"https://127.0.0.1:{port}/lol-replays/v1/rofls/{args.game_id}/watch",
            method="POST",
            data=json.dumps({"componentType": "replay"}).encode(),
            headers={"Authorization": f"Basic {auth}",
                     "Content-Type": "application/json"})
        urllib.request.urlopen(req, context=ctx, timeout=10)
        print("  Launched")

    # Step 3: Wait for game
    print("\nStep 3: Waiting for game...")
    if not wait_for_game():
        print("  FAILED: Game didn't load")
        return

    # Check if replay API is available
    try:
        pb = api_get("/replay/playback")
        print(f"  Replay API: OK (length={pb['length']:.0f}s)")
    except:
        print("  FAILED: Replay API not available")
        return

    # Step 4: Record
    print(f"\nStep 4: Recording {args.duration}s clip...")
    video_path = os.path.join(args.output_dir, "test_clip.webm")
    cam_data = record_clip(video_path, args.start, args.duration,
                            args.fps, args.champion)

    # Step 5: Verify
    print("\nStep 5: Verifying output...")
    ok, stats = verify_output(video_path, args.duration, args.fps)

    # Step 6: Restore resolution
    if not args.no_restore and orig_w and orig_h:
        if orig_w != "1920" or orig_h != "1080":
            print(f"\nStep 6: Restoring {orig_w}x{orig_h}...")
            # Kill game first
            os.system('taskkill /F /IM "League of Legends.exe" >nul 2>&1')
            time.sleep(2)
            set_league_resolution(int(orig_w), int(orig_h))
    else:
        print("\nStep 6: Skipping restore (was already 1080p or --no-restore)")

    # Summary
    print("\n" + "=" * 50)
    print("RESULT:", "PASS" if ok else "FAIL")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
