#!/usr/bin/env python3
"""Compare color-based vs OCR-based health bar detection."""

import cv2
import sys
sys.path.insert(0, 'src')

from ahriuwu.data.keylog_extractor import GoldTextDetector, HUDRegionsNormalized

def test_color_based(video_path: str, team_side: str, num_frames: int, start_frame: int):
    """Test color-based detection."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
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
    """Test OCR-based detection (simulated - need OCR code)."""
    # For this test, we'll use a simple OCR-based detector
    # Import easyocr only if available
    try:
        import easyocr
    except ImportError:
        print("  OCR: easyocr not available, skipping")
        return 0, num_frames

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print("  OCR: Loading model...", flush=True)
    reader = easyocr.Reader(['en'], gpu=False)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    detected = 0
    last_name_pos = None
    ocr_interval = 15  # Run OCR every 15 frames

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        # Only run OCR every N frames
        if i % ocr_interval == 0 or last_name_pos is None:
            # Search in game area
            ga_x = int(0.20 * width)
            ga_y = int(0.10 * height)
            ga_w = int(0.60 * width)
            ga_h = int(0.50 * height)

            # Narrow search if we have a previous position
            if last_name_pos is not None:
                lx, ly = last_name_pos
                expand = 100
                ga_x = max(0, lx - expand)
                ga_y = max(0, ly - expand)
                ga_w = min(width - ga_x, 2 * expand)
                ga_h = min(height - ga_y, 2 * expand)

            roi = frame[ga_y:ga_y+ga_h, ga_x:ga_x+ga_w]
            results = reader.readtext(roi)

            found = False
            for (bbox, text, conf) in results:
                if 'garen' in text.lower():
                    # Get center bottom of text box
                    pts = bbox
                    cx = int((pts[0][0] + pts[2][0]) / 2) + ga_x
                    cy = int(max(p[1] for p in pts)) + ga_y
                    last_name_pos = (cx, cy)
                    found = True
                    break

            if found:
                detected += 1
            elif last_name_pos is not None:
                # Use cached position
                detected += 1
        else:
            # Use cached position
            if last_name_pos is not None:
                detected += 1

        if (i + 1) % 300 == 0:
            pct = 100 * detected / (i + 1)
            print(f"  OCR: {i+1}/{num_frames} - {pct:.1f}%", flush=True)

    cap.release()
    return detected, num_frames


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

    print("Testing OCR-BASED detection...")
    ocr_detected, ocr_total = test_ocr_based(video_path, team_side, num_frames, start_frame)
    ocr_rate = 100 * ocr_detected / ocr_total
    print(f"OCR-based: {ocr_detected}/{ocr_total} ({ocr_rate:.1f}%)")
    print()

    print("=" * 50)
    print(f"RESULTS:")
    print(f"  Color-based: {color_rate:.1f}%")
    print(f"  OCR-based:   {ocr_rate:.1f}%")
    print(f"  Difference:  {ocr_rate - color_rate:+.1f}%")


if __name__ == "__main__":
    main()
