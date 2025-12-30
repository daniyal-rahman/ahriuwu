#!/usr/bin/env python3
"""Test the keylog extractor and visualize HUD regions."""

import cv2
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahriuwu.data.keylog_extractor import (
    KeylogExtractor, HUDRegions1080p, visualize_regions,
    GarenHUDTracker, AbilityBarDetector
)


def analyze_frame(frame_path: Path, output_path: Path = None):
    """Analyze a single frame and visualize HUD regions."""
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"Failed to load: {frame_path}")
        return

    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    regions = HUDRegions1080p()

    # Visualize regions
    vis = visualize_regions(frame, regions)

    # Save visualization
    if output_path is None:
        output_path = frame_path.parent / f"{frame_path.stem}_regions.jpg"

    cv2.imwrite(str(output_path), vis)
    print(f"Saved visualization to: {output_path}")

    # Analyze ability bar regions
    print("\nAbility bar analysis:")
    detector = AbilityBarDetector(regions)

    for name, region in [
        ('Q', regions.ability_q),
        ('W', regions.ability_w),
        ('E', regions.ability_e),
        ('R', regions.ability_r),
        ('D', regions.summoner_d),
        ('F', regions.summoner_f),
    ]:
        x, y, w, h = region
        ability_img = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(ability_img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1].mean()
        bright = hsv[:, :, 2].mean()
        print(f"  {name}: saturation={sat:.1f}, brightness={bright:.1f}")

        # Save cropped ability
        cv2.imwrite(str(output_path.parent / f"ability_{name}.jpg"), ability_img)


def test_movement_detection(video_path: Path, start_sec: float = 60, duration_sec: float = 5):
    """Test movement detection on a video segment."""
    print(f"\nTesting movement detection: {start_sec}s to {start_sec + duration_sec}s")

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_sec * fps)
    end_frame = int((start_sec + duration_sec) * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    regions = HUDRegions1080p()
    tracker = GarenHUDTracker(regions)

    movements = []
    frame_idx = start_frame

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        dx, dy, conf = tracker.detect_movement(frame)
        wasd = tracker.infer_wasd(dx, dy)

        movements.append({
            'frame': frame_idx,
            'dx': dx,
            'dy': dy,
            'conf': conf,
            'wasd': wasd
        })

        frame_idx += 1

    cap.release()

    # Print stats
    total = len(movements)
    with_movement = sum(1 for m in movements if m['wasd'])
    avg_dx = np.mean([m['dx'] for m in movements])
    avg_dy = np.mean([m['dy'] for m in movements])
    avg_conf = np.mean([m['conf'] for m in movements])

    print(f"  Total frames: {total}")
    print(f"  Frames with movement: {with_movement} ({100*with_movement/total:.1f}%)")
    print(f"  Average dx: {avg_dx:.2f}, dy: {avg_dy:.2f}")
    print(f"  Average confidence: {avg_conf:.3f}")

    # Print first few movements
    print("\n  First 20 frames with movement:")
    count = 0
    for m in movements:
        if m['wasd'] and count < 20:
            print(f"    Frame {m['frame']}: dx={m['dx']:.2f}, dy={m['dy']:.2f}, keys={m['wasd']}")
            count += 1

    return movements


def find_garen_hud(frame_path: Path):
    """Try to locate Garen's overhead HUD using template matching."""
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h, w = frame.shape[:2]

    print(f"\nSearching for Garen HUD in {w}x{h} frame...")

    # Look for the level box (small blue rectangle with number)
    # It should be near the center of the screen, slightly above middle

    # Search in center region
    search_region = frame[200:400, 600:800]  # Roughly where Garen's head would be

    # Look for blue-ish color (the level box background)
    hsv = cv2.cvtColor(search_region, cv2.COLOR_BGR2HSV)

    # Blue range
    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"  Found {len(contours)} blue regions")

    for i, cnt in enumerate(contours[:5]):
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        print(f"    Region {i}: x={x+600}, y={y+200}, w={cw}, h={ch}, area={area}")

    # Save debug image
    debug = frame.copy()
    cv2.rectangle(debug, (600, 200), (800, 400), (0, 255, 255), 2)  # Search region
    cv2.imwrite(str(frame_path.parent / "hud_search_region.jpg"), debug)

    # Save mask
    cv2.imwrite(str(frame_path.parent / "blue_mask.jpg"), mask)


if __name__ == "__main__":
    frame_dir = Path("data/keylog_extraction/frames_1080p")
    video_path = Path("data/keylog_extraction/garen_replay.mp4")

    # Analyze a sample frame
    sample_frame = frame_dir / "frame_0015.jpg"
    if sample_frame.exists():
        analyze_frame(sample_frame)
        find_garen_hud(sample_frame)

    # Test movement detection on video
    if video_path.exists():
        test_movement_detection(video_path, start_sec=60, duration_sec=10)
