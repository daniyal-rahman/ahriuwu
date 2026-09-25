#!/usr/bin/env python3
"""
Record a League replay at exact 20 game-fps using the Replay API.

Uses pause → screenshot → seek approach. No monitor refresh dependency.
Captures camera position every frame for pixel-perfect action overlay.

Usage:
    python scripts/record_replay_api.py \
        --game-id 5528069928 \
        --champion Garen \
        --output-dir data/processed_replays/NA1-5528069928 \
        --start 0 --end 0 --fps 20

Requirements:
    - Windows with League client running
    - EnableReplayApi=1 in Config/game.cfg under [General]
    - pip install dxcam opencv-python
"""

import argparse
import base64
import json
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Replay API client
# ---------------------------------------------------------------------------

GAME_API = "https://127.0.0.1:2999"
_ctx = ssl.create_default_context()
_ctx.check_hostname = False
_ctx.verify_mode = ssl.CERT_NONE


def api_get(endpoint):
    req = urllib.request.Request(f"{GAME_API}{endpoint}")
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r:
        return json.loads(r.read())


def api_post(endpoint, data):
    body = json.dumps(data).encode()
    req = urllib.request.Request(f"{GAME_API}{endpoint}", data=body,
                                headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, context=_ctx, timeout=5) as r:
        return json.loads(r.read())


def lcu_post(port, token, endpoint, data=None):
    auth = base64.b64encode(f"riot:{token}".encode()).decode()
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        f"https://127.0.0.1:{port}{endpoint}",
        method="POST", data=body,
        headers={"Authorization": f"Basic {auth}",
                 "Content-Type": "application/json"})
    with urllib.request.urlopen(req, context=_ctx, timeout=10) as r:
        return r.read()


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------

def wait_for_game(timeout=180):
    """Wait for the game to load (poll Live Client Data API)."""
    print("Waiting for game to load...")
    for _ in range(timeout // 3):
        try:
            stats = api_get("/liveclientdata/gamestats")
            print(f"  Game loaded! time={stats['gameTime']:.1f}s")
            return stats
        except Exception:
            time.sleep(3)
    raise TimeoutError("Game did not load")


def find_champion_slot(champion_name):
    """Find which player slot a champion is in."""
    players = api_get("/liveclientdata/playerlist")
    for i, p in enumerate(players):
        if p.get("championName", "").lower() == champion_name.lower():
            team = p.get("team", "")
            return i, team, p.get("riotIdGameName", "?")
    return None, None, None


def setup_camera(champion_name):
    """Lock camera to champion and hide UI for clean recording."""
    api_post("/replay/render", {
        "selectionName": champion_name,
        "cameraAttached": True,
        # Hide HUD elements for cleaner recording
        "interfaceTimeline": False,
        "interfaceChat": False,
        "interfaceScoreboard": False,
        "floatingText": True,
        "healthBarMinions": True,
        "healthBarChampions": True,
    })
    time.sleep(0.5)
    render = api_get("/replay/render")
    print(f"  Camera attached to {champion_name}")
    print(f"  Position: ({render['cameraPosition']['x']:.0f}, "
          f"{render['cameraPosition']['y']:.0f}, "
          f"{render['cameraPosition']['z']:.0f})")
    return render


def record(output_dir, start_time, end_time, fps, champion_name, speed=4.0):
    """Record frames using Replay API with high-speed playback.

    Strategy: play at 4-8x speed, capture as fast as dxcam allows (~25 wall-fps),
    tag each raw frame with game_time. Then in post-processing, upsample to
    target fps by duplicating frames to fill 20 game-fps slots.

    Camera position is polled every few frames and interpolated for the rest.
    """
    import dxcam
    import cv2
    import numpy as np

    frame_step = 1.0 / fps
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Init screen capture
    cam = dxcam.create(output_idx=0)
    time.sleep(0.5)
    f = cam.grab()
    while f is None:
        time.sleep(0.1)
        f = cam.grab()
    h, w = f.shape[:2]
    print(f"Capture: {w}x{h}")

    # Seek to start
    print(f"Seeking to {start_time:.1f}s...")
    api_post("/replay/playback", {"time": start_time, "paused": True, "speed": 0})
    for _ in range(100):
        time.sleep(0.1)
        try:
            if not api_get("/replay/playback").get("seeking", False):
                break
        except Exception:
            pass

    # Setup camera
    setup_camera(champion_name)
    time.sleep(0.5)

    # Start playback at target speed
    api_post("/replay/playback", {"paused": False, "speed": speed})
    time.sleep(0.3)

    duration = end_time - start_time
    # Write raw frames directly to a temp video + metadata file (no RAM buffering)
    raw_video_path = str(output_dir / "raw_capture.avi")
    raw_writer = cv2.VideoWriter(raw_video_path, cv2.VideoWriter_fourcc(*"MJPG"),
                                  30, (w, h))  # arbitrary fps for raw
    raw_meta = []
    raw_idx = 0
    last_cam = {"x": 0, "y": 0, "z": 0}
    cam_poll_interval = 5  # poll camera every N frames

    print(f"\nCapturing at {speed}x speed ({duration:.0f}s game = "
          f"{duration/speed:.0f}s wall)")
    t_wall_start = time.perf_counter()

    while True:
        # Capture frame
        frame = cam.grab()
        if frame is None:
            time.sleep(0.005)
            continue

        frame_bgr = frame[:, :, ::-1].copy()

        # Get game time
        try:
            pb = api_get("/replay/playback")
            game_time = pb["time"]
        except Exception:
            continue

        if game_time >= end_time:
            break

        # Poll camera position periodically
        if raw_idx % cam_poll_interval == 0:
            try:
                rd = api_get("/replay/render")
                last_cam = rd["cameraPosition"]
            except Exception:
                pass

        # Write frame to disk immediately (no RAM accumulation)
        raw_writer.write(frame_bgr)
        raw_meta.append({
            "idx": raw_idx,
            "game_time": game_time,
            "cam_x": last_cam["x"],
            "cam_y": last_cam["y"],
            "cam_z": last_cam["z"],
        })
        raw_idx += 1

        # Progress
        if raw_idx % 50 == 0:
            elapsed = time.perf_counter() - t_wall_start
            pct = (game_time - start_time) / duration * 100
            wall_fps = raw_idx / elapsed
            game_fps = raw_idx / max(game_time - start_time, 0.1)
            print(f"  Raw {raw_idx}: t={game_time:.1f}s ({pct:.0f}%) "
                  f"[{wall_fps:.1f} wall-fps, {game_fps:.1f} game-fps]")

    api_post("/replay/playback", {"paused": True})
    raw_writer.release()
    elapsed = time.perf_counter() - t_wall_start
    game_covered = raw_meta[-1]["game_time"] - raw_meta[0]["game_time"] if raw_meta else 0
    raw_game_fps = len(raw_meta) / game_covered if game_covered > 0 else 0

    print(f"\nRaw capture: {len(raw_meta)} frames, {game_covered:.1f}s game, "
          f"{raw_game_fps:.1f} game-fps, {elapsed:.1f}s wall")

    # ================================================================
    # Post-process: upsample to target fps by nearest-frame selection
    # Read from raw video, write to output video
    # ================================================================
    print(f"\nUpsampling to {fps} game-fps...")

    raw_reader = cv2.VideoCapture(raw_video_path)
    n_output_frames = int(game_covered * fps)
    video_path = str(output_dir / "replay.avi")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"),
                              fps, (w, h))

    # Stream: read raw frames one at a time, duplicate to fill output fps
    # Build a map: for each output frame, which raw frame index to use
    output_start = raw_meta[0]["game_time"]
    output_raw_map = []  # output_frame_idx -> raw_frame_idx
    raw_i = 0
    for out_i in range(n_output_frames):
        target_t = output_start + out_i * frame_step
        while raw_i < len(raw_meta) - 1 and raw_meta[raw_i + 1]["game_time"] <= target_t:
            raw_i += 1
        output_raw_map.append(raw_i)

    # Stream through raw video, writing duplicated frames
    print(f"  Streaming {len(raw_meta)} raw -> {n_output_frames} output frames...")
    frame_data = []
    current_raw_frame = None
    current_raw_idx = -1
    raw_read_idx = 0

    for out_i in range(n_output_frames):
        needed_raw = output_raw_map[out_i]

        # Read forward in raw video until we reach the needed frame
        while raw_read_idx <= needed_raw:
            ret, frm = raw_reader.read()
            if ret:
                current_raw_frame = frm
                current_raw_idx = raw_read_idx
            raw_read_idx += 1

        if current_raw_frame is not None:
            writer.write(current_raw_frame)

        # Camera position (interpolated)
        rm = raw_meta[needed_raw]
        cam_x = rm["cam_x"]
        cam_y = rm["cam_y"]
        cam_z = rm["cam_z"]
        target_t = output_start + out_i * frame_step

        if needed_raw < len(raw_meta) - 1:
            rm_next = raw_meta[needed_raw + 1]
            dt = rm_next["game_time"] - rm["game_time"]
            if dt > 0:
                t_frac = (target_t - rm["game_time"]) / dt
                t_frac = max(0, min(1, t_frac))
                cam_x += (rm_next["cam_x"] - cam_x) * t_frac
                cam_z += (rm_next["cam_z"] - cam_z) * t_frac

        frame_data.append({
            "frame": out_i,
            "game_time": target_t,
            "camera_x": cam_x,
            "camera_y": cam_y,
            "camera_z": cam_z,
            "source_raw_frame": rm["idx"],
        })

        if (out_i + 1) % 2000 == 0:
            print(f"    {out_i+1}/{n_output_frames}")

    writer.release()
    raw_reader.release()

    # Clean up raw video
    try:
        os.remove(raw_video_path)
    except Exception:
        pass

    print(f"Output: {n_output_frames} frames at {fps}fps")

    # Save metadata
    meta_path = str(output_dir / "frame_data.json")
    with open(meta_path, "w") as f:
        json.dump(frame_data, f, indent=2)

    info = {
        "video_path": video_path,
        "fps": fps,
        "resolution": [w, h],
        "start_time": start_time,
        "end_time": end_time,
        "n_output_frames": n_output_frames,
        "n_raw_frames": len(raw_frames),
        "raw_game_fps": raw_game_fps,
        "playback_speed": speed,
        "champion": champion_name,
        "wall_time": elapsed,
    }
    with open(str(output_dir / "recording_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    print(f"Video: {video_path} ({os.path.getsize(video_path)/1e6:.1f}MB)")
    print(f"Metadata: {meta_path}")

    return frame_data


def main():
    parser = argparse.ArgumentParser(description="Record replay via Replay API")
    parser.add_argument("--game-id", required=True, help="Game ID to replay")
    parser.add_argument("--champion", default="Garen", help="Champion to follow")
    parser.add_argument("--output-dir", "-o", required=True)
    parser.add_argument("--start", type=float, default=0,
                        help="Start time in seconds (0=game start)")
    parser.add_argument("--end", type=float, default=0,
                        help="End time in seconds (0=full game)")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--speed", type=float, default=4.0,
                        help="Playback speed (4.0 or 8.0)")
    parser.add_argument("--skip-launch", action="store_true",
                        help="Skip launching the replay (game already running)")

    args = parser.parse_args()

    if not args.skip_launch:
        # Launch replay via LCU
        lockfile = open(r"C:\Riot Games\League of Legends\lockfile").read().strip().split(":")
        lcu_port, lcu_token = int(lockfile[2]), lockfile[3]
        print(f"Launching replay {args.game_id} via LCU (port {lcu_port})...")
        lcu_post(lcu_port, lcu_token,
                 f"/lol-replays/v1/rofls/{args.game_id}/watch",
                 {"componentType": "replay"})

    # Wait for game
    stats = wait_for_game()

    # Get game length
    playback = api_get("/replay/playback")
    game_length = playback["length"]
    print(f"Game length: {game_length:.1f}s ({game_length/60:.1f}min)")

    start = args.start if args.start > 0 else 1.0  # skip first second
    end = args.end if args.end > 0 else game_length - 1.0

    # Find and attach to champion
    slot, team, summoner = find_champion_slot(args.champion)
    if slot is not None:
        print(f"Found {args.champion}: slot {slot}, {team}, {summoner}")
    else:
        print(f"WARNING: {args.champion} not found, using free camera")

    setup_camera(args.champion)

    # Record
    record(args.output_dir, start, end, args.fps, args.champion, speed=args.speed)

    # Kill game
    os.system('taskkill /F /IM "League of Legends.exe" >nul 2>&1')


if __name__ == "__main__":
    main()
