#!/usr/bin/env python3
"""Check gold popup visual timing vs OCR detection.

For a gold event at frame t, check frames t-5 to t+5 to see:
1. When gold popup FIRST appears visually (by looking at the popup region)
2. When OCR detects it (the gold_gained field)
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image


def analyze_gold_popup_region(video_id: str, frame_idx: int, features_dir: str, window: int = 5):
    """Analyze the gold popup region across frames."""
    features_dir = Path(features_dir)
    video_dir = features_dir / video_id
    frames_dir = video_dir / "frames"

    # Load features
    with open(video_dir / "features.json") as f:
        features = json.load(f)

    frames_data = features.get("frames", [])

    # Gold popups typically appear near the character health bar
    # The health bar position is tracked in features
    center_frame = frames_data[frame_idx]
    hb_x = center_frame.get("health_bar_x", 640)
    hb_y = center_frame.get("health_bar_y", 400)

    # Gold popup region is typically above/around the health bar
    # Define a region of interest (ROI) for the gold text
    # Gold text appears above the character, roughly 100-200 pixels above health bar
    roi_x1 = max(0, hb_x - 100)
    roi_x2 = min(1280, hb_x + 100)
    roi_y1 = max(0, hb_y - 150)  # Above health bar
    roi_y2 = max(0, hb_y - 20)

    print(f"\nVideo: {video_id}, Frame: {frame_idx}")
    print(f"Health bar: ({hb_x}, {hb_y})")
    print(f"Gold popup ROI: ({roi_x1}, {roi_y1}) to ({roi_x2}, {roi_y2})")
    print()

    print(f"{'t':>4} | {'gold_ocr':>8} | {'roi_brightness':>14} | notes")
    print(f"{'-'*4}-+-{'-'*8}-+-{'-'*14}-+{'-'*20}")

    prev_brightness = None
    for t in range(frame_idx - window, frame_idx + window + 1):
        if t < 0 or t >= len(frames_data):
            continue

        frame_path = frames_dir / f"frame_{t:06d}.jpg"
        if not frame_path.exists():
            continue

        img = Image.open(frame_path)
        img_array = np.array(img)

        # Extract ROI and compute brightness (gold text is bright yellow)
        roi = img_array[roi_y1:roi_y2, roi_x1:roi_x2]

        # Look for yellow pixels (high R, high G, low B)
        r, g, b = roi[:, :, 0], roi[:, :, 1], roi[:, :, 2]
        yellow_mask = (r > 150) & (g > 150) & (b < 100)
        yellow_pixels = yellow_mask.sum()

        gold_ocr = frames_data[t].get("gold_gained", 0)
        rel_t = t - frame_idx

        notes = ""
        if rel_t == 0:
            notes = " <-- OCR detection"
        elif prev_brightness is not None and yellow_pixels > prev_brightness * 1.5:
            notes = " <-- brightness spike"

        print(f"{rel_t:+4} | {gold_ocr:>8} | {yellow_pixels:>14} | {notes}")
        prev_brightness = yellow_pixels


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="data/processed")
    parser.add_argument("--n-events", type=int, default=5)
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    # Find clean gold events (not consecutive)
    features_dir = Path(args.features_dir)
    events = []

    for video_dir in sorted(features_dir.iterdir()):
        if not video_dir.is_dir():
            continue

        features_path = video_dir / "features.json"
        if not features_path.exists():
            continue

        with open(features_path) as f:
            features = json.load(f)

        frames = features.get("frames", [])
        for i, frame in enumerate(frames):
            gold = frame.get("gold_gained", 0)
            if 15 <= gold <= 30:  # Typical minion kill
                # Check it's isolated (no gold in adjacent frames)
                prev_gold = frames[i-1].get("gold_gained", 0) if i > 0 else 0
                next_gold = frames[i+1].get("gold_gained", 0) if i < len(frames)-1 else 0
                if prev_gold == 0 and next_gold == 0 and i > 10:
                    events.append({
                        "video_id": video_dir.name,
                        "frame_idx": i,
                        "gold": gold,
                    })

    print(f"Found {len(events)} isolated minion kill events")

    import random
    random.seed(42)
    selected = random.sample(events, min(args.n_events, len(events)))

    for event in selected:
        analyze_gold_popup_region(
            event["video_id"],
            event["frame_idx"],
            args.features_dir,
            window=args.window
        )


if __name__ == "__main__":
    main()
