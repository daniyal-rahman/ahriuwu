#!/usr/bin/env python3
"""Extract health bar candidates for manual curation."""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahriuwu.data.keylog_extractor import GoldTextDetector, HUDRegionsNormalized


def extract_templates(
    video_path: Path,
    output_dir: Path,
    start_sec: float = 0,
    duration_sec: float = 60,
    sample_every_n_frames: int = 20,  # Don't need every frame
):
    """Extract health bar candidates from video."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + duration_sec) * fps)

    print(f"Extracting from {start_sec}s to {start_sec + duration_sec}s")
    print(f"Frames {start_frame} to {end_frame}, sampling every {sample_every_n_frames} frames")

    # Initialize detector (disable template matching to use color detection)
    normalized_regions = HUDRegionsNormalized()
    detector = GoldTextDetector(
        normalized_regions=normalized_regions,
        frame_width=frame_width,
        frame_height=frame_height,
        use_gpu=False,
    )
    detector._use_template_matching = False  # Force color detection

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    template_count = 0
    frame_count = 0

    while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        absolute_frame = start_frame + frame_count

        # Only process every Nth frame
        if frame_count % sample_every_n_frames == 0:
            # Find health bars using color detection
            health_bars = detector._find_teal_health_bars(frame)

            for i, (hb_x, hb_y, hb_w, hb_h) in enumerate(health_bars):
                # Level box is ~25px left of teal bar
                # Teal bar is ~105px wide, level box is ~25px
                # Total health bar width: ~130px
                level_box_width = 25
                full_bar_width = 130

                # Crop tightly: level box + health bar
                x1 = max(0, hb_x - level_box_width)  # No left margin
                y1 = max(0, hb_y - 3)  # 3px top margin
                x2 = min(frame_width, hb_x - level_box_width + full_bar_width + 2)  # 2px right margin
                y2 = min(frame_height, hb_y + hb_h + 3)  # 3px bottom margin

                template = frame[y1:y2, x1:x2]

                # Save template
                filename = f"hb_{template_count:04d}_frame{absolute_frame}_idx{i}.jpg"
                cv2.imwrite(str(output_dir / filename), template)
                template_count += 1

            if frame_count % 100 == 0:
                print(f"  Frame {absolute_frame}: found {len(health_bars)} health bars, total templates: {template_count}")

        frame_count += 1

    cap.release()

    print(f"\nDone! Saved {template_count} templates to {output_dir}")
    print(f"Review and delete bad ones, then we'll use the rest.")


if __name__ == "__main__":
    video_path = Path("data/keylog_extraction/garen_replay.mp4")
    output_dir = Path("data/keylog_extraction/health_bar_templates")

    # Extract from multiple segments to get variety
    extract_templates(
        video_path=video_path,
        output_dir=output_dir,
        start_sec=900,  # 15:00
        duration_sec=120,  # 2 minutes
        sample_every_n_frames=30,  # Every 30 frames (~0.5 sec)
    )
