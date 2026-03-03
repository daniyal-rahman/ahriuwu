#!/usr/bin/env python3
"""Generate a video with detected button presses overlaid on the footage."""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahriuwu.data.keylog_extractor import (
    HUDRegions, HUDRegionsNormalized, GarenHUDTracker, AbilityBarDetector
)


def draw_detection_regions(frame: np.ndarray, regions: HUDRegions) -> np.ndarray:
    """Draw boxes showing where detection is happening."""
    vis = frame.copy()

    # Draw game area for optical flow (cyan, thin)
    x, y, w, h = regions.game_area
    cv2.rectangle(vis, (x, y), (x+w, y+h), (255, 255, 0), 1)
    cv2.putText(vis, "WASD detection area", (x+5, y+15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

    # Draw ability detection boxes (different colors for each)
    ability_colors = {
        'Q': (0, 255, 255),    # Yellow
        'W': (0, 165, 255),    # Orange
        'E': (255, 0, 255),    # Magenta
        'R': (0, 0, 255),      # Red
        'D': (255, 255, 255),  # White
        'F': (255, 255, 255),  # White
    }

    ability_regions = {
        'Q': regions.ability_q,
        'W': regions.ability_w,
        'E': regions.ability_e,
        'R': regions.ability_r,
        'D': regions.summoner_d,
        'F': regions.summoner_f,
    }

    for name, region in ability_regions.items():
        x, y, w, h = region
        color = ability_colors[name]
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        cv2.putText(vis, name, (x+8, y+17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Label the ability detection area
    cv2.putText(vis, "Ability detection", (40, 900),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    return vis


def draw_key_overlay(frame: np.ndarray, wasd: list[str], abilities: dict[str, bool],
                     dx: float, dy: float, frame_idx: int, fps: float,
                     regions: HUDRegions) -> np.ndarray:
    """Draw button press overlay on frame. Scales with resolution."""
    # First draw detection regions
    vis = draw_detection_regions(frame, regions)

    h, w = vis.shape[:2]

    # Scale factor based on reference resolution (1080p)
    scale = min(w / 1920, h / 1080)

    # Scaled dimensions
    box_w = int(290 * scale)
    box_h = int(190 * scale)
    margin = int(10 * scale)
    key_size = int(35 * scale)
    key_gap = int(5 * scale)
    font_scale_title = 0.6 * scale
    font_scale_key = 0.7 * scale
    font_scale_small = 0.4 * scale
    thickness = max(1, int(2 * scale))
    thickness_thin = max(1, int(1 * scale))

    # Draw semi-transparent overlay box in bottom-right
    overlay_x = w - box_w - margin
    overlay_y = h - box_h - margin
    cv2.rectangle(vis, (overlay_x, overlay_y), (w - margin, h - margin), (0, 0, 0), -1)
    cv2.rectangle(vis, (overlay_x, overlay_y), (w - margin, h - margin), (255, 255, 255), thickness)

    # Title
    cv2.putText(vis, "Detected Inputs", (overlay_x + margin, overlay_y + int(25 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_title, (255, 255, 255), thickness)

    # WASD display - keyboard layout style
    wasd_x = overlay_x + int(30 * scale)
    wasd_y = overlay_y + int(60 * scale)

    # Draw WASD keys
    key_positions = {
        'W': (wasd_x + key_size + key_gap, wasd_y),
        'A': (wasd_x, wasd_y + key_size + key_gap),
        'S': (wasd_x + key_size + key_gap, wasd_y + key_size + key_gap),
        'D': (wasd_x + 2 * (key_size + key_gap), wasd_y + key_size + key_gap),
    }

    for key, (kx, ky) in key_positions.items():
        is_pressed = key in wasd
        color = (0, 255, 0) if is_pressed else (50, 50, 50)
        cv2.rectangle(vis, (kx, ky), (kx + key_size, ky + key_size), color, -1)
        cv2.rectangle(vis, (kx, ky), (kx + key_size, ky + key_size), (255, 255, 255), thickness_thin)
        text_color = (0, 0, 0) if is_pressed else (150, 150, 150)
        text_x = kx + int(10 * scale)
        text_y = ky + int(25 * scale)
        cv2.putText(vis, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_key, text_color, thickness)

    # Abilities display (QWER + DF)
    ability_x = overlay_x + int(150 * scale)
    ability_y = overlay_y + int(60 * scale)
    ability_keys = ['Q', 'W', 'E', 'R', 'D', 'F']

    for i, key in enumerate(ability_keys):
        kx = ability_x + (i % 4) * (key_size + key_gap)
        ky = ability_y + (i // 4) * (key_size + key_gap)
        is_pressed = abilities.get(key, False)
        color = (0, 200, 255) if is_pressed else (50, 50, 50)
        cv2.rectangle(vis, (kx, ky), (kx + key_size, ky + key_size), color, -1)
        cv2.rectangle(vis, (kx, ky), (kx + key_size, ky + key_size), (255, 255, 255), thickness_thin)
        text_color = (0, 0, 0) if is_pressed else (150, 150, 150)
        text_x = kx + int(10 * scale)
        text_y = ky + int(25 * scale)
        cv2.putText(vis, key, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_key, text_color, thickness)

    # Movement vector
    cv2.putText(vis, f"dx:{dx:+.1f} dy:{dy:+.1f}", (overlay_x + margin, overlay_y + int(175 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (200, 200, 200), thickness_thin)

    # Frame info - show timestamp in MM:SS format
    total_seconds = frame_idx / fps
    minutes = int(total_seconds // 60)
    seconds = total_seconds % 60
    cv2.putText(vis, f"Frame {frame_idx} ({minutes}:{seconds:05.2f})", (overlay_x + margin, h - margin - int(5 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (200, 200, 200), thickness_thin)

    return vis


def generate_overlay_video(
    video_path: Path,
    output_path: Path,
    start_sec: float = 920,  # 15:20
    duration_sec: float = 30,
    output_fps: int = 20,
    output_resolution: tuple[int, int] = None,  # (width, height) or None for original
):
    """Generate overlay video from segment."""
    cap = cv2.VideoCapture(str(video_path))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use output resolution if specified, otherwise use input resolution
    if output_resolution is not None:
        out_width, out_height = output_resolution
    else:
        out_width, out_height = frame_width, frame_height

    print(f"Input: {input_fps} FPS, {frame_width}x{frame_height}")
    print(f"Output: {output_fps} FPS, {out_width}x{out_height}, {duration_sec}s segment starting at {start_sec}s ({int(start_sec//60)}:{start_sec%60:02.0f})")

    # Calculate frame sampling
    start_frame = int(start_sec * input_fps)
    end_frame = int((start_sec + duration_sec) * input_fps)
    frame_skip = int(input_fps / output_fps)  # Sample every Nth frame

    print(f"Frames {start_frame} to {end_frame}, sampling every {frame_skip} frames")

    # Initialize video writer with output resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (out_width, out_height))

    # Initialize detectors using normalized regions at OUTPUT resolution
    # This ensures detection boxes scale correctly with the output
    normalized_regions = HUDRegionsNormalized()
    regions = normalized_regions.to_pixels(out_width, out_height)
    tracker = GarenHUDTracker(
        normalized_regions=normalized_regions,
        frame_width=out_width,
        frame_height=out_height,
    )
    ability_detector = AbilityBarDetector(
        normalized_regions=normalized_regions,
        frame_width=out_width,
        frame_height=out_height,
        fps=input_fps,
    )

    # === PASS 1: Detect all abilities and movement ===
    print("Pass 1: Detecting abilities and movement...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Store movement data per frame (indexed by frame number relative to start)
    movement_data = {}  # frame_idx -> (dx, dy, wasd)
    frame_count = 0
    needs_resize = (out_width != frame_width or out_height != frame_height)

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        absolute_frame = start_frame + frame_count

        # Resize frame if needed (detection happens on output resolution)
        if needs_resize:
            frame = cv2.resize(frame, (out_width, out_height))

        # Detect movement on every frame
        dx, dy, conf = tracker.detect_movement(frame)

        # Detect abilities (stores in buffer with time-corrected frame indices)
        ability_detector.detect_ability_usage(frame, frame_idx=absolute_frame)

        # Store movement for output frames
        if frame_count % frame_skip == 0:
            wasd = tracker.infer_wasd(dx, dy)
            movement_data[absolute_frame] = (dx, dy, wasd)

        frame_count += 1

    # Get all time-corrected ability detections
    all_detections = ability_detector.get_all_detections()
    print(f"  Found {len(all_detections)} ability activations (time-corrected)")

    # Build a map of output_frame -> abilities that should show
    # For each detection, find the nearest output frame
    output_frames = sorted(movement_data.keys())
    ability_by_frame = {f: {k: False for k in ['Q', 'W', 'E', 'R', 'D', 'F']} for f in output_frames}

    for corrected_frame, ability in all_detections:
        # Find the nearest output frame >= corrected_frame
        for out_frame in output_frames:
            if out_frame >= corrected_frame:
                ability_by_frame[out_frame][ability] = True
                break

    # === PASS 2: Render overlay with corrected detections ===
    print("Pass 2: Rendering overlay...")
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_count = 0
    total_with_movement = 0
    total_abilities = 0
    frame_count = 0

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        absolute_frame = start_frame + frame_count

        # Only output every Nth frame
        if frame_count % frame_skip == 0:
            # Resize frame if needed
            if needs_resize:
                frame = cv2.resize(frame, (out_width, out_height))

            dx, dy, wasd = movement_data[absolute_frame]
            abilities = ability_by_frame[absolute_frame]

            # Stats
            if wasd:
                total_with_movement += 1
            if any(abilities.values()):
                total_abilities += 1

            # Draw overlay
            vis = draw_key_overlay(frame, wasd, abilities, dx, dy,
                                   absolute_frame, input_fps, regions)

            out.write(vis)
            output_count += 1

            if output_count % 100 == 0:
                print(f"  Rendered {output_count} frames...")

        frame_count += 1

    cap.release()
    out.release()

    print(f"\nDone! Generated {output_count} frames")
    print(f"  Frames with movement: {total_with_movement} ({100*total_with_movement/output_count:.1f}%)")
    print(f"  Frames with abilities: {total_abilities}")
    print(f"  Output: {output_path}")


if __name__ == "__main__":
    video_path = Path("data/keylog_extraction/garen_replay.mp4")

    # Generate at original resolution (1920x1080)
    print("=== Generating 1080p version ===")
    generate_overlay_video(
        video_path=video_path,
        output_path=Path("data/keylog_extraction/overlay_1520_1550_1080p.mp4"),
        start_sec=920,  # 15:20
        duration_sec=30,
        output_fps=20,
        output_resolution=None,  # Original resolution
    )

    # Generate at 256x256
    print("\n=== Generating 256x256 version ===")
    generate_overlay_video(
        video_path=video_path,
        output_path=Path("data/keylog_extraction/overlay_1520_1550_256.mp4"),
        start_sec=920,  # 15:20
        duration_sec=30,
        output_fps=20,
        output_resolution=(256, 256),
    )
