#!/usr/bin/env python3
"""Measure actual gold detection delay by re-running OCR without consensus.

Compares:
1. Raw OCR detection (first frame that sees "+XXg")
2. Consensus-confirmed detection (gold_gained field)
"""

import json
from pathlib import Path

import cv2
import numpy as np
import easyocr


def measure_delays(video_id: str, features_dir: str, n_events: int = 5, window: int = 10):
    """Measure delay between first OCR hit and consensus confirmation."""
    features_dir = Path(features_dir)
    video_dir = features_dir / video_id
    frames_dir = video_dir / "frames"

    # Load features
    with open(video_dir / "features.json") as f:
        features = json.load(f)

    frames_data = features.get("frames", [])

    # Find gold events
    gold_frames = []
    for i, frame in enumerate(frames_data):
        gold = frame.get("gold_gained", 0)
        if 15 <= gold <= 30:  # Typical minion kills
            # Check isolated
            prev_gold = frames_data[i-1].get("gold_gained", 0) if i > 0 else 0
            next_gold = frames_data[i+1].get("gold_gained", 0) if i < len(frames_data)-1 else 0
            if prev_gold == 0 and next_gold == 0 and i > window:
                gold_frames.append((i, gold))

    print(f"Found {len(gold_frames)} isolated gold events")

    if not gold_frames:
        return

    # Initialize OCR
    reader = easyocr.Reader(['en'], gpu=True, verbose=False)

    # HSV ranges for gold color
    gold_lower = np.array([15, 80, 150])
    gold_upper = np.array([35, 255, 255])

    delays = []

    import random
    random.seed(42)
    selected = random.sample(gold_frames, min(n_events, len(gold_frames)))

    for confirmed_frame, gold_amount in selected:
        print(f"\n{'='*60}")
        print(f"Confirmed at frame {confirmed_frame}: +{gold_amount}g")
        print(f"{'='*60}")

        # Scan backwards to find first raw OCR detection
        first_detection = None

        for t in range(confirmed_frame - window, confirmed_frame + 3):
            if t < 0 or t >= len(frames_data):
                continue

            frame_path = frames_dir / f"frame_{t:06d}.jpg"
            if not frame_path.exists():
                continue

            img = cv2.imread(str(frame_path))
            if img is None:
                continue

            # Get health bar position for ROI
            hb_x = frames_data[t].get("health_bar_x", 640)
            hb_y = frames_data[t].get("health_bar_y", 400)

            # Define ROI around character
            roi_x1 = max(0, hb_x - 150)
            roi_x2 = min(img.shape[1], hb_x + 150)
            roi_y1 = max(0, hb_y - 200)
            roi_y2 = max(0, hb_y + 50)

            roi = img[roi_y1:roi_y2, roi_x1:roi_x2]

            if roi.size == 0:
                continue

            # Filter for gold color
            hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask = cv2.inRange(hsv, gold_lower, gold_upper)
            filtered = cv2.bitwise_and(roi, roi, mask=mask)

            # Run OCR
            try:
                results = reader.readtext(filtered)
            except:
                results = []

            # Check for target gold amount
            detected = None
            for (bbox, text, conf) in results:
                text = text.strip()
                if text.startswith('+'):
                    try:
                        amount_str = ''.join(c for c in text[1:] if c.isdigit())
                        if amount_str:
                            detected_amount = int(amount_str)
                            # Check if it matches or is close to target
                            if detected_amount == gold_amount or abs(detected_amount - gold_amount) <= 2:
                                detected = (detected_amount, conf)
                                break
                    except:
                        pass

            rel_t = t - confirmed_frame
            status = ""
            if detected:
                status = f"DETECTED +{detected[0]}g (conf={detected[1]:.2f})"
                if first_detection is None:
                    first_detection = t
            if t == confirmed_frame:
                status += " <-- CONSENSUS"

            print(f"  t={rel_t:+3}: {status}")

        if first_detection is not None:
            delay = confirmed_frame - first_detection
            delays.append(delay)
            print(f"\n  >> First detection at t-{delay}, delay = {delay} frames")
        else:
            print(f"\n  >> Could not find first detection (OCR missed)")

    if delays:
        print(f"\n{'='*60}")
        print(f"SUMMARY: {len(delays)} events measured")
        print(f"  Mean delay: {sum(delays)/len(delays):.1f} frames")
        print(f"  Min delay:  {min(delays)} frames")
        print(f"  Max delay:  {max(delays)} frames")
        print(f"{'='*60}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--features-dir", default="data/processed")
    parser.add_argument("--video", default=None, help="Specific video to analyze")
    parser.add_argument("--n-events", type=int, default=5)
    parser.add_argument("--window", type=int, default=10)
    args = parser.parse_args()

    features_dir = Path(args.features_dir)

    if args.video:
        videos = [args.video]
    else:
        # Pick a video with many gold events
        videos = []
        for video_dir in features_dir.iterdir():
            if not video_dir.is_dir():
                continue
            features_path = video_dir / "features.json"
            if not features_path.exists():
                continue
            with open(features_path) as f:
                features = json.load(f)
            gold_count = sum(1 for f in features.get("frames", []) if f.get("gold_gained", 0) > 0)
            if gold_count > 100:
                videos.append(video_dir.name)
                if len(videos) >= 1:
                    break

    for video_id in videos:
        print(f"\n{'#'*60}")
        print(f"Analyzing: {video_id}")
        print(f"{'#'*60}")
        measure_delays(video_id, args.features_dir, args.n_events, args.window)


if __name__ == "__main__":
    main()
