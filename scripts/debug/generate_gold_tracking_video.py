#!/usr/bin/env python3
"""Generate a video showing dynamic health bar tracking and gold detection."""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahriuwu.data.keylog_extractor import GoldTextDetector, HUDRegionsNormalized


def generate_gold_tracking_video(
    video_path: Path,
    output_path: Path,
    start_sec: float = 920,  # 15:20
    duration_sec: float = 30,
    output_fps: int = 20,
    output_resolution: tuple[int, int] = None,  # (width, height) or None for original
):
    """Generate video showing health bar tracking and gold detections."""
    cap = cv2.VideoCapture(str(video_path))
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Use output resolution if specified
    if output_resolution is not None:
        out_width, out_height = output_resolution
    else:
        out_width, out_height = frame_width, frame_height

    needs_resize = (out_width != frame_width or out_height != frame_height)

    print(f"Input: {input_fps} FPS, {frame_width}x{frame_height}")
    print(f"Output: {output_fps} FPS, {out_width}x{out_height}, {duration_sec}s segment starting at {start_sec}s ({int(start_sec//60)}:{start_sec%60:02.0f})")

    # Calculate frame sampling
    start_frame = int(start_sec * input_fps)
    end_frame = int((start_sec + duration_sec) * input_fps)
    frame_skip = int(input_fps / output_fps)

    print(f"Frames {start_frame} to {end_frame}, sampling every {frame_skip} frames")

    # Initialize video writer with output resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, output_fps, (out_width, out_height))

    # Initialize gold detector at OUTPUT resolution (detection happens on resized frame)
    normalized_regions = HUDRegionsNormalized()
    gold_detector = GoldTextDetector(
        normalized_regions=normalized_regions,
        frame_width=out_width,
        frame_height=out_height,
        use_gpu=False,  # CPU for compatibility
    )

    # Load curated templates for stable tracking
    template_dir = Path(__file__).parent.parent / "data/keylog_extraction/health_bar_templates"
    if template_dir.exists():
        gold_detector.load_curated_templates(template_dir)
        print(f"Loaded {len(gold_detector._curated_templates)} curated templates")

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    output_count = 0
    frame_count = 0
    total_gold_detected = 0
    recent_gold_display = []  # (frame_idx, amount, conf) for displaying recently detected gold

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        absolute_frame = start_frame + frame_count

        # Only process every Nth frame
        if frame_count % frame_skip == 0:
            # Resize frame if needed
            if needs_resize:
                frame = cv2.resize(frame, (out_width, out_height))

            # Detect gold text (also gets health bar position)
            gold_gains, health_bar = gold_detector.detect_gold_text(frame, frame_idx=absolute_frame)

            # Track recent gold for display
            for amount, conf in gold_gains:
                recent_gold_display.append((absolute_frame, amount, conf))
                total_gold_detected += amount
                print(f"  Frame {absolute_frame}: Detected +{amount} gold (conf={conf:.2f})")

            # Remove old gold displays (older than 2 seconds = 40 frames at 20fps)
            recent_gold_display = [(f, a, c) for f, a, c in recent_gold_display
                                   if absolute_frame - f < 40]

            # Draw visualization
            vis = frame.copy()

            # Scale factor for text/line thickness based on resolution
            scale = min(out_width / 1920, out_height / 1080)
            thickness = max(1, int(2 * scale))
            font_scale = max(0.3, 0.5 * scale)

            # Draw search area bounds (magenta) - where we look for health bars
            sa_x = int(0.20 * out_width)
            sa_y = int(0.10 * out_height)  # Start higher (10% instead of 18%)
            sa_w = int(0.60 * out_width)
            sa_h = int(0.63 * out_height)  # Extended to 73%
            cv2.rectangle(vis, (sa_x, sa_y), (sa_x + sa_w, sa_y + sa_h), (255, 0, 255), 1)
            cv2.putText(vis, "Search Area", (sa_x + 5, sa_y + int(15 * scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 255), 1)

            # Draw health bar tracking box (green)
            if health_bar is not None:
                hb_x, hb_y, hb_w, hb_h = health_bar
                # Draw health bar outline
                cv2.rectangle(vis, (hb_x, hb_y), (hb_x + hb_w, hb_y + hb_h), (0, 255, 0), thickness)

                # Draw the ROI box (extends from health bar down)
                roi_h = int(0.18 * out_height)
                cv2.rectangle(vis, (hb_x, hb_y), (hb_x + hb_w, min(hb_y + roi_h, out_height)), (0, 255, 255), 1)

                # Label
                cv2.putText(vis, "Gold ROI", (hb_x, max(hb_y - 3, 10)),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), 1)

            # Draw gold detection overlay (scaled)
            overlay_x = int(10 * scale)
            overlay_y = int(10 * scale)
            overlay_w = int(200 * scale)
            overlay_h = int(80 * scale)

            # Semi-transparent background
            cv2.rectangle(vis, (overlay_x, overlay_y),
                         (overlay_x + overlay_w, overlay_y + overlay_h), (0, 0, 0), -1)
            cv2.rectangle(vis, (overlay_x, overlay_y),
                         (overlay_x + overlay_w, overlay_y + overlay_h), (255, 255, 255), 1)

            # Title
            cv2.putText(vis, "Gold Detection", (overlay_x + int(10 * scale), overlay_y + int(20 * scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1)

            # Total gold
            cv2.putText(vis, f"Total: +{total_gold_detected}", (overlay_x + int(10 * scale), overlay_y + int(45 * scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 215, 255), 1)  # Gold color

            # Recent detection
            if recent_gold_display:
                latest = recent_gold_display[-1]
                cv2.putText(vis, f"Last: +{latest[1]} ({latest[2]:.0%})", (overlay_x + int(10 * scale), overlay_y + int(65 * scale)),
                           cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (200, 200, 200), 1)

            # Frame info
            total_seconds = absolute_frame / input_fps
            minutes = int(total_seconds // 60)
            seconds = total_seconds % 60
            cv2.putText(vis, f"Frame {absolute_frame} ({minutes}:{seconds:05.2f})",
                       (int(10 * scale), out_height - int(10 * scale)),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale * 0.8, (200, 200, 200), 1)

            out.write(vis)
            output_count += 1

            if output_count % 100 == 0:
                print(f"  Rendered {output_count} frames...")

        frame_count += 1

    cap.release()
    out.release()

    print(f"\nDone! Generated {output_count} frames")
    print(f"Total gold detected: +{total_gold_detected}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    import sys
    video_path = Path("data/keylog_extraction/garen_replay.mp4")

    mode = sys.argv[1] if len(sys.argv) > 1 else "clip"

    if mode == "full":
        # Full game at 1080p
        print("=== Generating FULL GAME (1080p) ===")
        generate_gold_tracking_video(
            video_path=video_path,
            output_path=Path("data/keylog_extraction/gold_tracking_full_1080p.mp4"),
            start_sec=0,
            duration_sec=1424,  # Full 24 min
            output_fps=20,
        )
    elif mode == "256":
        # 30 sec clip at 256x256
        print("=== Generating 30s clip (256x256) ===")
        generate_gold_tracking_video(
            video_path=video_path,
            output_path=Path("data/keylog_extraction/gold_tracking_30s_256.mp4"),
            start_sec=920,  # 15:20
            duration_sec=30,
            output_fps=20,
            output_resolution=(256, 256),
        )
    else:
        # Default: 30 sec clip at 1080p
        print("=== Generating 30s clip (1080p) ===")
        generate_gold_tracking_video(
            video_path=video_path,
            output_path=Path("data/keylog_extraction/gold_tracking_demo.mp4"),
            start_sec=920,  # 15:20
            duration_sec=30,
            output_fps=20,
        )
