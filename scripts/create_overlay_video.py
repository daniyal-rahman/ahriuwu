#!/usr/bin/env python3
"""
Create an overlay video with action labels on recorded replay footage.

Uses camera position from the Replay API to compute exact screen positions
for movement clicks and ability casts. Shows a key HUD that lights up on press.

Usage:
    python scripts/create_overlay_video.py \
        --video data/processed_replays/NA1-5528069928/replay.avi \
        --frame-data data/processed_replays/NA1-5528069928/frame_data.json \
        --actions /tmp/garen_12_0_actions.json \
        --output /tmp/garen_overlay.mp4
"""

import argparse
import json
import math
import cv2
import numpy as np
from collections import defaultdict

# ---------------------------------------------------------------------------
# Map-to-screen projection for locked camera
# ---------------------------------------------------------------------------
# LoL camera: top-down at ~56° tilt, 40° FOV
# Camera position from API: x = map_x, y = height, z = map_z
# In LoL's coordinate system: x = left-right, z = up-down on map
# Screen: x = left-right, y = top-bottom
#
# With locked camera, champion is near screen center.
# The visible map area depends on camera height and FOV.
# At default zoom (height ~1912), the visible area is roughly:
#   ~3200 map units wide, ~2400 map units tall (at 16:9)
# This gives a scale of ~0.6 pixels per map unit at 1920x1080.
#
# We calibrate empirically: scale = screen_width / visible_map_width

DEFAULT_MAP_VIEW_WIDTH = 3200.0  # map units visible horizontally at default zoom
CAMERA_TILT_DEG = 56.0  # camera tilt angle from horizontal


def map_to_screen(map_x, map_z, cam_x, cam_z, screen_w, screen_h,
                  map_view_width=DEFAULT_MAP_VIEW_WIDTH):
    """Convert map coordinates to screen pixel coordinates.

    Args:
        map_x, map_z: action position in map coordinates
        cam_x, cam_z: camera position (= champion position for locked cam)
        screen_w, screen_h: video resolution
    Returns:
        (screen_x, screen_y) or None if off-screen
    """
    scale = screen_w / map_view_width
    # X axis: map_x increases rightward, screen_x increases rightward
    sx = (map_x - cam_x) * scale + screen_w / 2
    # Z axis: map_z increases upward in LoL, screen_y increases downward
    # Also account for camera tilt (foreshortening on vertical axis)
    tilt_factor = math.cos(math.radians(CAMERA_TILT_DEG))
    sy = (cam_z - map_z) * scale * tilt_factor + screen_h / 2

    # Check bounds (with some margin)
    margin = 50
    if -margin <= sx <= screen_w + margin and -margin <= sy <= screen_h + margin:
        return int(sx), int(sy)
    return None


# ---------------------------------------------------------------------------
# Action snapping: assign each action to its nearest frame (floor)
# ---------------------------------------------------------------------------

def snap_actions_to_frames(actions, frame_data, fps):
    """Assign each action to a frame index using floor rounding.

    An action at game_time t maps to frame = floor((t - start_time) * fps).
    Multiple actions can map to the same frame.
    """
    if not frame_data:
        return {}

    start_time = frame_data[0]["game_time"]
    end_time = frame_data[-1]["game_time"]
    frame_step = 1.0 / fps

    # Build frame_index -> [actions] mapping
    frame_actions = defaultdict(list)

    for action in actions:
        t = action.get("game_time") or action.get("est_time", 0)
        if t < start_time or t > end_time:
            continue
        frame_idx = int((t - start_time) / frame_step)
        frame_idx = max(0, min(frame_idx, len(frame_data) - 1))
        frame_actions[frame_idx].append(action)

    return dict(frame_actions)


# ---------------------------------------------------------------------------
# Drawing functions
# ---------------------------------------------------------------------------

# Colors (BGR)
SPELL_COLORS = {
    'Q': (0, 200, 255),     # orange
    'W': (255, 200, 0),     # cyan
    'E': (0, 255, 0),       # green
    'R': (0, 0, 255),       # red
    'D': (255, 100, 255),   # pink
    'F': (255, 255, 0),     # yellow
    'B': (200, 200, 200),   # gray
}

KEY_ORDER = ['Q', 'W', 'E', 'R', 'D', 'F', 'B']
KEY_INACTIVE = (60, 60, 60)
KEY_BG = (30, 30, 30)


def draw_key_hud(frame, active_keys, screen_w, screen_h):
    """Draw a persistent key bar at the bottom showing Q W E R D F B.
    Active keys light up with their color.
    """
    n_keys = len(KEY_ORDER)
    key_w = 70
    key_h = 50
    gap = 8
    total_w = n_keys * key_w + (n_keys - 1) * gap
    x_start = (screen_w - total_w) // 2
    y_start = screen_h - key_h - 20

    # Background bar
    cv2.rectangle(frame,
                  (x_start - 10, y_start - 10),
                  (x_start + total_w + 10, y_start + key_h + 10),
                  KEY_BG, -1)

    font = cv2.FONT_HERSHEY_SIMPLEX

    for i, key in enumerate(KEY_ORDER):
        x = x_start + i * (key_w + gap)
        y = y_start

        is_active = key in active_keys
        color = SPELL_COLORS.get(key, (255, 255, 255)) if is_active else KEY_INACTIVE

        # Key box
        thickness = -1 if is_active else 2
        cv2.rectangle(frame, (x, y), (x + key_w, y + key_h), color, thickness)

        # Key label
        text_color = (0, 0, 0) if is_active else (120, 120, 120)
        text_scale = 1.2 if is_active else 0.9
        (tw, th), _ = cv2.getTextSize(key, font, text_scale, 2)
        tx = x + (key_w - tw) // 2
        ty = y + (key_h + th) // 2
        cv2.putText(frame, key, (tx, ty), font, text_scale, text_color, 2, cv2.LINE_AA)


def draw_click_marker(frame, sx, sy, color, radius=18, thickness=2):
    """Draw a click indicator (circle + crosshair)."""
    sx = max(0, min(sx, frame.shape[1] - 1))
    sy = max(0, min(sy, frame.shape[0] - 1))
    cv2.circle(frame, (sx, sy), radius, color, thickness, cv2.LINE_AA)
    cv2.circle(frame, (sx, sy), 4, color, -1, cv2.LINE_AA)  # center dot
    arm = radius + 6
    cv2.line(frame, (sx - arm, sy), (sx - radius - 2, sy), color, 1)
    cv2.line(frame, (sx + radius + 2, sy), (sx + arm, sy), color, 1)
    cv2.line(frame, (sx, sy - arm), (sx, sy - radius - 2), color, 1)
    cv2.line(frame, (sx, sy + radius + 2), (sx, sy + arm), color, 1)


def draw_info(frame, frame_idx, game_time, n_frames, screen_w):
    """Draw frame number and game time in the corner."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    mins = int(game_time // 60)
    secs = game_time % 60

    # Game time
    time_text = f"{mins}:{secs:04.1f}"
    cv2.putText(frame, time_text, (15, 40), font, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, time_text, (15, 40), font, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    # Frame number
    frame_text = f"F{frame_idx}/{n_frames}"
    cv2.putText(frame, frame_text, (15, 75), font, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, frame_text, (15, 75), font, 0.6, (180, 180, 180), 1, cv2.LINE_AA)


def draw_action_log(frame, recent_actions, screen_w):
    """Draw recent actions in the top-right corner."""
    if not recent_actions:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    x = screen_w - 300
    y_start = 30

    # Semi-transparent background
    n_show = min(len(recent_actions), 8)
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - 10, y_start - 20),
                  (screen_w - 10, y_start + n_show * 28 + 5),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, (t, spell, color) in enumerate(recent_actions[-n_show:]):
        y = y_start + i * 28
        mins = int(t // 60)
        secs = t % 60
        text = f"{mins}:{secs:04.1f}  {spell}"
        cv2.putText(frame, text, (x, y + 18), font, 0.55, color, 1, cv2.LINE_AA)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Create action overlay video")
    parser.add_argument("--video", required=True, help="Input replay video")
    parser.add_argument("--frame-data", required=True,
                        help="Frame metadata JSON (from record_replay_api.py)")
    parser.add_argument("--actions", required=True, help="Action data JSON")
    parser.add_argument("--output", "-o", default="/tmp/garen_overlay.mp4")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--map-view-width", type=float, default=DEFAULT_MAP_VIEW_WIDTH,
                        help="Visible map width in game units (calibration)")
    args = parser.parse_args()

    # Load data
    print("Loading data...")
    with open(args.frame_data) as f:
        frame_data = json.load(f)

    with open(args.actions) as f:
        actions = json.load(f)

    movements = actions.get("movements", [])
    abilities = actions.get("abilities", [])
    recalls = actions.get("recalls", [])

    print(f"  Frame data: {len(frame_data)} frames")
    print(f"  Movements: {len(movements)}, Abilities: {len(abilities)}, "
          f"Recalls: {len(recalls)}")

    fps = 20  # assumed from recording
    game_duration = actions.get("game_duration", 2525)

    # Estimate game_time for abilities that don't have it
    total_blocks = max((a.get("block_index", 0) for a in abilities), default=1) + 1
    for a in abilities:
        if not a.get("game_time"):
            a["est_time"] = (a.get("block_index", 0) / total_blocks) * game_duration

    # Estimate game_time for recalls
    for r in recalls:
        if not r.get("game_time"):
            r["est_time"] = r.get("approx_time", 0)

    # Build timed action list
    timed_actions = []

    for m in movements:
        t = m.get("game_time", 0)
        if t > 0 and m.get("current_x") is not None:
            timed_actions.append({
                "game_time": t, "type": "move",
                "map_x": m.get("dest_x") or m.get("current_x"),
                "map_z": m.get("dest_y") or m.get("current_y"),
                "spell": "CLICK",
            })

    for a in abilities:
        t = a.get("game_time") or a.get("est_time", 0)
        spell = a.get("ability", a.get("spell", "?"))
        timed_actions.append({
            "game_time": t, "type": "ability", "spell": spell,
            "map_x": a.get("cast_x", 0), "map_z": a.get("cast_y", 0),
        })

    for r in recalls:
        t = r.get("game_time") or r.get("est_time", 0)
        timed_actions.append({
            "game_time": t, "type": "recall", "spell": "B",
            "map_x": 0, "map_z": 0,
        })

    timed_actions.sort(key=lambda a: a["game_time"])
    print(f"  Total timed actions: {len(timed_actions)}")

    # Snap actions to frames
    frame_actions = snap_actions_to_frames(timed_actions, frame_data, fps)
    frames_with_actions = len(frame_actions)
    print(f"  Frames with actions: {frames_with_actions}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open {args.video}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"  Video: {width}x{height} @ {vid_fps}fps, {total_frames} frames")

    # Output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    if not out.isOpened():
        print(f"ERROR: Cannot create {args.output}")
        return

    max_frames = args.max_frames if args.max_frames > 0 else min(total_frames, len(frame_data))
    recent_log = []  # (time, spell, color) for action log
    # Track active keys — stay lit for a short duration
    active_keys = {}  # key -> frames_remaining

    KEY_ACTIVE_FRAMES = 6  # stay lit for 6 frames (0.3s at 20fps)

    print(f"\nProcessing {max_frames} frames...")

    for frame_idx in range(max_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Get frame metadata
        if frame_idx < len(frame_data):
            fd = frame_data[frame_idx]
            game_time = fd["game_time"]
            cam_x = fd["camera_x"]
            cam_z = fd["camera_z"]
        else:
            game_time = 0
            cam_x = cam_z = 0

        # Process actions for this frame
        new_keys = set()
        if frame_idx in frame_actions:
            for action in frame_actions[frame_idx]:
                spell = action["spell"]
                atype = action["type"]
                color = SPELL_COLORS.get(spell, (255, 255, 255))

                # Mark key as active
                if spell in KEY_ORDER:
                    new_keys.add(spell)
                    active_keys[spell] = KEY_ACTIVE_FRAMES

                # Draw click marker at action position
                mx = action.get("map_x", 0)
                mz = action.get("map_z", 0)
                if mx and mz and cam_x and cam_z:
                    pos = map_to_screen(mx, mz, cam_x, cam_z, width, height,
                                        args.map_view_width)
                    if pos:
                        sx, sy = pos
                        marker_color = color if atype == "ability" else (255, 255, 255)
                        draw_click_marker(frame, sx, sy, marker_color)
                        # Label
                        if atype == "ability":
                            cv2.putText(frame, spell, (sx + 22, sy - 8),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        color, 2, cv2.LINE_AA)

                # Add to log
                recent_log.append((game_time, spell, color))

        # Decay active keys
        expired = []
        for key in active_keys:
            if key not in new_keys:
                active_keys[key] -= 1
                if active_keys[key] <= 0:
                    expired.append(key)
        for key in expired:
            del active_keys[key]

        # Draw overlays
        draw_key_hud(frame, set(active_keys.keys()), width, height)
        draw_info(frame, frame_idx, game_time, max_frames, width)
        draw_action_log(frame, recent_log, width)

        out.write(frame)

        if (frame_idx + 1) % 200 == 0:
            print(f"  Frame {frame_idx+1}/{max_frames} "
                  f"(t={game_time:.1f}s, {len(recent_log)} actions so far)")

    cap.release()
    out.release()
    print(f"\nDone! {frame_idx + 1} frames written to {args.output}")


if __name__ == "__main__":
    main()
