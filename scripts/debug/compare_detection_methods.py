#!/usr/bin/env python3
"""Compare color-based vs OCR-based health bar detection (fair test).

OCR detection is validated by checking for health bar color at expected position.
"""

import cv2
import numpy as np
import sys
sys.path.insert(0, 'src')

from ahriuwu.data.keylog_extractor import GoldTextDetector, HUDRegionsNormalized


def has_health_bar_at_position(frame: np.ndarray, name_x: int, name_y: int, team_side: str) -> bool:
    """Check if there's a health bar at the expected position below name text."""
    h, w = frame.shape[:2]

    # Health bar should be ~5-10px below name, centered on name_x
    hb_w = int(105 * (w / 1920))
    hb_h = int(12 * (h / 1080))
    hb_x = name_x - hb_w // 2
    hb_y = name_y + int(5 * (h / 1080))

    # Clamp to frame bounds
    hb_x = max(0, min(hb_x, w - hb_w))
    hb_y = max(0, min(hb_y, h - hb_h))

    if hb_x < 0 or hb_y < 0 or hb_x + hb_w > w or hb_y + hb_h > h:
        return False

    roi = frame[hb_y:hb_y+hb_h, hb_x:hb_x+hb_w]
    if roi.size == 0:
        return False

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    if team_side == "red":
        # Red health bar (hue wraps around)
        mask1 = cv2.inRange(hsv, np.array([0, 150, 80]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 150, 80]), np.array([180, 255, 255]))
        health_mask = mask1 | mask2
    else:
        # Teal health bar
        health_mask = cv2.inRange(hsv, np.array([80, 80, 80]), np.array([100, 255, 255]))

    health_pixels = cv2.countNonZero(health_mask)
    total_pixels = roi.shape[0] * roi.shape[1]
    health_ratio = health_pixels / total_pixels if total_pixels > 0 else 0

    # Need at least 15% health-colored pixels
    return health_ratio > 0.15


def test_color_based(video_path: str, team_side: str, num_frames: int, start_frame: int):
    """Test color-based detection."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = GoldTextDetector(
        normalized_regions=HUDRegionsNormalized(),
        frame_width=width,
        frame_height=height,
        team_side=team_side,
    )

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    detected = 0
    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        health_bar = detector._find_health_bar(frame)
        if health_bar is not None:
            detected += 1

        if (i + 1) % 300 == 0:
            pct = 100 * detected / (i + 1)
            print(f"  Color: {i+1}/{num_frames} - {pct:.1f}%", flush=True)

    cap.release()
    return detected, num_frames


def test_ocr_based(video_path: str, team_side: str, num_frames: int, start_frame: int):
    """Test OCR-based detection with health bar validation."""
    try:
        import easyocr
    except ImportError:
        print("  OCR: easyocr not available, skipping")
        return 0, num_frames

    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("  OCR: Loading model...", flush=True)
    reader = easyocr.Reader(['en'], gpu=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    detected = 0
    validated = 0  # OCR found AND health bar confirmed

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Search in game area (center of screen where Garen would be)
        ga_x = int(0.25 * width)
        ga_y = int(0.15 * height)
        ga_w = int(0.50 * width)
        ga_h = int(0.45 * height)

        roi = frame[ga_y:ga_y+ga_h, ga_x:ga_x+ga_w]
        results = reader.readtext(roi)

        found_garen = False
        for (bbox, text, conf) in results:
            if 'garen' in text.lower():
                # Get center bottom of text box
                pts = bbox
                cx = int((pts[0][0] + pts[2][0]) / 2) + ga_x
                cy = int(max(p[1] for p in pts)) + ga_y

                found_garen = True
                detected += 1

                # Validate: check if health bar exists at expected position
                if has_health_bar_at_position(frame, cx, cy, team_side):
                    validated += 1
                break

        if (i + 1) % 100 == 0:
            det_pct = 100 * detected / (i + 1)
            val_pct = 100 * validated / (i + 1)
            print(f"  OCR: {i+1}/{num_frames} - found: {det_pct:.1f}%, validated: {val_pct:.1f}%", flush=True)

    cap.release()
    return detected, validated, num_frames


def main():
    video_path = 'data/raw/videos/qYm3MNvtzBU.mp4'
    team_side = 'red'  # Garen is on red team in this video

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    start_frame = int(90 * fps)  # Skip pre-game
    num_frames = int(60 * fps)   # Test 1 minute

    print(f"Testing on {num_frames} frames (1 minute of gameplay)")
    print(f"Team side: {team_side}")
    print()

    print("Testing COLOR-BASED detection...")
    color_detected, color_total = test_color_based(video_path, team_side, num_frames, start_frame)
    color_rate = 100 * color_detected / color_total
    print(f"Color-based: {color_detected}/{color_total} ({color_rate:.1f}%)")
    print()

    print("Testing OCR-BASED detection (with validation)...")
    ocr_detected, ocr_validated, ocr_total = test_ocr_based(video_path, team_side, num_frames, start_frame)
    ocr_found_rate = 100 * ocr_detected / ocr_total
    ocr_valid_rate = 100 * ocr_validated / ocr_total
    print(f"OCR found 'Garen': {ocr_detected}/{ocr_total} ({ocr_found_rate:.1f}%)")
    print(f"OCR validated (with health bar): {ocr_validated}/{ocr_total} ({ocr_valid_rate:.1f}%)")
    print()

    print("=" * 50)
    print(f"RESULTS:")
    print(f"  Color-based:        {color_rate:.1f}%")
    print(f"  OCR (found text):   {ocr_found_rate:.1f}%")
    print(f"  OCR (validated):    {ocr_valid_rate:.1f}%")


if __name__ == "__main__":
    main()
