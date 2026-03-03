#!/usr/bin/env python3
"""Debug gold OCR by visualizing what OCR sees."""

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def debug_frame(video_id: str, frame_idx: int, features_dir: str):
    """Debug OCR on a specific frame."""
    import easyocr

    features_dir = Path(features_dir)
    video_dir = features_dir / video_id
    frames_dir = video_dir / "frames"

    # Load features
    with open(video_dir / "features.json") as f:
        features = json.load(f)

    frames_data = features.get("frames", [])

    frame_path = frames_dir / f"frame_{frame_idx:06d}.jpg"
    img = cv2.imread(str(frame_path))

    print(f"Frame shape: {img.shape}")

    # Get health bar position
    hb_x = frames_data[frame_idx].get("health_bar_x", 640)
    hb_y = frames_data[frame_idx].get("health_bar_y", 400)
    gold = frames_data[frame_idx].get("gold_gained", 0)

    print(f"Health bar: ({hb_x}, {hb_y})")
    print(f"Recorded gold_gained: {gold}")

    # Use same ROI logic as keylog_extractor
    # Actually let's check what the extractor uses
    height, width = img.shape[:2]

    # The extractor uses a dynamic ROI around the health bar
    # Let's try multiple ROI sizes
    rois = [
        ("small", max(0, hb_x - 100), min(width, hb_x + 100), max(0, hb_y - 150), hb_y),
        ("medium", max(0, hb_x - 200), min(width, hb_x + 200), max(0, hb_y - 200), hb_y + 50),
        ("large", max(0, hb_x - 300), min(width, hb_x + 300), max(0, hb_y - 300), hb_y + 100),
        ("full_width", 0, width, max(0, hb_y - 200), hb_y + 50),
    ]

    reader = easyocr.Reader(['en'], gpu=True, verbose=False)

    for name, x1, x2, y1, y2 in rois:
        print(f"\n--- ROI: {name} ({x1},{y1}) to ({x2},{y2}) ---")

        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            print("  Empty ROI")
            continue

        # Try without color filter first
        print(f"  ROI shape: {roi.shape}")

        try:
            results = reader.readtext(roi)
            print(f"  OCR results (raw): {len(results)} texts")
            for (bbox, text, conf) in results:
                if conf > 0.3:
                    print(f"    '{text}' (conf={conf:.2f})")
        except Exception as e:
            print(f"  OCR error: {e}")

        # Now with gold color filter
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Gold color (yellow-orange range)
        gold_lower = np.array([15, 80, 150])
        gold_upper = np.array([35, 255, 255])
        gold_mask = cv2.inRange(hsv, gold_lower, gold_upper)

        # Also try white (for the + sign)
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)

        combined_mask = cv2.bitwise_or(gold_mask, white_mask)
        filtered = cv2.bitwise_and(roi, roi, mask=combined_mask)

        # Check pixel counts
        gold_pixels = cv2.countNonZero(gold_mask)
        white_pixels = cv2.countNonZero(white_mask)
        print(f"  Gold pixels: {gold_pixels}, White pixels: {white_pixels}")

        if gold_pixels > 0 or white_pixels > 0:
            try:
                results = reader.readtext(filtered)
                print(f"  OCR results (filtered): {len(results)} texts")
                for (bbox, text, conf) in results:
                    if conf > 0.3:
                        print(f"    '{text}' (conf={conf:.2f})")
            except Exception as e:
                print(f"  OCR error: {e}")

    # Save debug image
    debug_img = img.copy()
    for name, x1, x2, y1, y2 in rois[:3]:
        color = {'small': (0, 255, 0), 'medium': (0, 255, 255), 'large': (255, 0, 0)}[name]
        cv2.rectangle(debug_img, (x1, y1), (x2, y2), color, 2)
    cv2.circle(debug_img, (hb_x, hb_y), 10, (0, 0, 255), -1)  # Health bar center

    output_path = f"/tmp/gold_debug_{video_id}_{frame_idx}.jpg"
    cv2.imwrite(output_path, debug_img)
    print(f"\nSaved debug image: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="data/processed")
    parser.add_argument("--video", required=True)
    parser.add_argument("--frame", type=int, required=True)
    args = parser.parse_args()

    debug_frame(args.video, args.frame, args.features_dir)


if __name__ == "__main__":
    main()
