#!/usr/bin/env python3
"""Generate a video showing item usage detection."""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahriuwu.data.keylog_extractor import ItemUsageDetector


def generate_item_tracking_video(
    video_path: Path,
    output_path: Path,
    start_sec: float = 920,
    duration_sec: float = 30,
    output_fps: int = 20,
    team_side: str = "blue",
):
    """Generate video showing item usage detection."""
    cap = cv2.VideoCapture(str(video_path))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Input: {input_fps} FPS, {frame_width}x{frame_height}")
    print(f"Output: {output_fps} FPS, {duration_sec}s segment starting at {start_sec}s")
    print(f"Team side: {team_side}")

    start_frame = int(start_sec * input_fps)
    end_frame = int((start_sec + duration_sec) * input_fps)
    frame_skip = int(input_fps / output_fps)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (frame_width, frame_height))

    # Initialize item detector
    item_detector = ItemUsageDetector(
        frame_width=frame_width,
        frame_height=frame_height,
        team_side=team_side,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_count = 0
    frame_count = 0
    total_item_uses = 0
    recent_uses = []  # (frame_idx, slot_idx) for display

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        absolute_frame = start_frame + frame_count

        if frame_count % frame_skip == 0:
            # Detect item usage
            used_slots = item_detector.detect_item_usage(frame, absolute_frame)

            for slot_idx in used_slots:
                recent_uses.append((absolute_frame, slot_idx))
                total_item_uses += 1
                print(f"  Frame {absolute_frame}: Item slot {slot_idx + 1} used!")

            # Remove old uses from display (older than 2 seconds)
            recent_uses = [(f, s) for f, s in recent_uses if absolute_frame - f < 40]

            # Draw visualization
            vis = frame.copy()

            # Draw item slot boxes
            for slot_idx, (x, y, w, h) in enumerate(item_detector.item_slots):
                # Check if this slot was recently used
                recently_used = any(s == slot_idx for _, s in recent_uses)

                if recently_used:
                    color = (0, 0, 255)  # Red for recently used
                    thickness = 3
                else:
                    color = (0, 255, 0)  # Green for ready
                    thickness = 2

                cv2.rectangle(vis, (x, y), (x + w, y + h), color, thickness)

                # Show slot number
                cv2.putText(vis, str(slot_idx + 1), (x + 2, y - 3),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)

                # Show saturation value
                sat = item_detector.get_item_saturation(frame, slot_idx)
                cv2.putText(vis, f"{sat:.0f}", (x, y + h + 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, (200, 200, 200), 1)

            # Draw info overlay
            overlay_x, overlay_y = 10, 10
            cv2.rectangle(vis, (overlay_x, overlay_y), (overlay_x + 200, overlay_y + 60), (0, 0, 0), -1)
            cv2.putText(vis, f"Item Usage Detection", (overlay_x + 10, overlay_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(vis, f"Total uses: {total_item_uses}", (overlay_x + 10, overlay_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            # Frame info
            total_seconds = absolute_frame / input_fps
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            cv2.putText(vis, f"Frame {absolute_frame} ({minutes}:{seconds:05.2f})",
                       (10, frame_height - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            out.write(vis)
            output_count += 1

            if output_count % 100 == 0:
                print(f"  Rendered {output_count} frames...")

        frame_count += 1

    cap.release()
    out.release()

    print(f"\nDone! Generated {output_count} frames")
    print(f"Total item uses detected: {total_item_uses}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    video_path = Path("data/keylog_extraction/garen_replay.mp4")
    output_path = Path("data/keylog_extraction/item_tracking_demo.mp4")

    generate_item_tracking_video(
        video_path=video_path,
        output_path=output_path,
        start_sec=920,
        duration_sec=30,
        output_fps=20,
        team_side="blue",  # Garen is on blue side
    )
