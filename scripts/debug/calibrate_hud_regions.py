#!/usr/bin/env python3
"""Calibrate HUD region positions by cropping and saving specific areas."""

import cv2
import numpy as np
from pathlib import Path


def crop_and_save_regions(frame_path: Path, output_dir: Path):
    """Crop various regions from a frame to help calibrate HUD positions."""
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"Failed to load: {frame_path}")
        return

    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Define regions to crop for inspection
    regions = {
        # === Garen's overhead HUD (moves with character, but locked camera keeps it near center) ===
        # Level box is the teal square with "1" - good tracking target
        "garen_level_box": (670, 276, 20, 16),  # Teal level indicator
        "garen_name_hp": (668, 264, 100, 30),   # Name + level + health bar combined

        # === Bottom-left player HUD (fixed position) ===
        "bottom_left_full": (0, 620, 240, 160),  # Full bottom-left HUD area
        "garen_portrait": (10, 638, 50, 58),     # Garen's portrait icon
        "health_mana_bars": (62, 640, 90, 20),   # Health/mana bar area with text
        "stats_row": (62, 660, 150, 20),         # CS, gold stats row

        # === Ability bar - the key detection target ===
        # In domisumReplay, abilities are shown in bottom-left area
        # Looking for Q/W/E/R icons + summoner spells
        "ability_area_wide": (0, 690, 220, 40),  # Wide search for ability icons
        "ability_icons_row": (65, 695, 160, 30), # Where QWER should be

        # === Items row ===
        "items_row": (75, 695, 120, 25),         # Item icons below portrait

        # === Scorecard (bottom center) ===
        "scorecard_garen_row": (455, 655, 200, 35),  # Garen's row in scorecard
    }

    for name, (x, y, w, h) in regions.items():
        crop = frame[y:y+h, x:x+w]
        output_path = output_dir / f"{name}.jpg"
        cv2.imwrite(str(output_path), crop)
        print(f"Saved {name}: ({x}, {y}, {w}, {h}) -> {output_path}")

    # Also save a grid visualization
    vis = frame.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i, (name, (x, y, w, h)) in enumerate(regions.items()):
        color = colors[i % len(colors)]
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        cv2.putText(vis, name[:10], (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    cv2.imwrite(str(output_dir / "calibration_grid.jpg"), vis)
    print(f"\nSaved calibration grid to {output_dir / 'calibration_grid.jpg'}")


def find_garen_level_box(frame_path: Path):
    """Try to find Garen's level box (blue rectangle with number) using color detection."""
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return

    # Convert to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Look for the blue/teal color of the level box
    # The level box appears to be a teal/cyan color
    lower_teal = np.array([85, 100, 100])
    upper_teal = np.array([100, 255, 255])
    mask = cv2.inRange(hsv, lower_teal, upper_teal)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"\nSearching for teal regions (level box color)...")
    print(f"Found {len(contours)} teal regions")

    # Filter for small rectangles (level box is ~20x18 pixels)
    candidates = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        aspect = w / h if h > 0 else 0

        # Level box criteria: small, roughly square, in center-ish region
        if 100 < area < 2000 and 0.8 < aspect < 1.5 and 200 < y < 400:
            candidates.append((x, y, w, h, area))

    print(f"Candidates (small teal rectangles in center region): {len(candidates)}")
    for x, y, w, h, area in candidates[:10]:
        print(f"  ({x}, {y}, {w}, {h}), area={area}")

    # Save visualization
    vis = frame.copy()
    for x, y, w, h, area in candidates:
        cv2.rectangle(vis, (x, y), (x+w, y+h), (0, 255, 0), 2)

    output_dir = frame_path.parent
    cv2.imwrite(str(output_dir / "level_box_candidates.jpg"), vis)


if __name__ == "__main__":
    frame_dir = Path("data/keylog_extraction/frames_1080p")
    output_dir = Path("data/keylog_extraction/calibration")

    sample_frame = frame_dir / "frame_0015.jpg"
    if sample_frame.exists():
        crop_and_save_regions(sample_frame, output_dir)
        find_garen_level_box(sample_frame)
    else:
        print(f"Frame not found: {sample_frame}")
