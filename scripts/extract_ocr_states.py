#!/usr/bin/env python3
"""Extract game state from frames using OCR.

Processes all frames and extracts:
- Game clock (timestamp)
- Garen health (OCR text: "current/total")
- Enemy health (color detection on side panel)

Results are saved as JSON per video for later use in training.

Usage:
    python scripts/extract_ocr_states.py --frames-dir data/processed/frames
    python scripts/extract_ocr_states.py --frames-dir data/processed/frames --sample-rate 20
    python scripts/extract_ocr_states.py --video LKfZG99Qbv0 --visualize  # Test on single video
"""

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np


def get_ocr_reader():
    """Lazy load EasyOCR reader."""
    import easyocr
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def read_game_clock_from_crop(crop: np.ndarray, reader) -> int | None:
    """Read game clock from cropped region.

    Args:
        crop: Cropped image of game clock region
        reader: EasyOCR reader instance

    Returns:
        Total seconds or None if failed
    """
    # Preprocess
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # OCR
    results = reader.readtext(thresh, detail=0)
    text = " ".join(results).strip()

    # Parse MM:SS or MM.SS
    match = re.search(r"(\d{1,2})[:.:](\d{2})", text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return minutes * 60 + seconds

    return None


def read_garen_health_from_crop(crop: np.ndarray, reader) -> dict | None:
    """Read Garen's health from text OCR (format: "current/total").

    Args:
        crop: Cropped image of health text region
        reader: EasyOCR reader instance

    Returns:
        Dict with current, total, percentage or None if failed
    """
    # Scale up 3x for better OCR accuracy on small text
    scaled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)

    # OCR with digit allowlist for better accuracy
    results = reader.readtext(scaled, detail=0, allowlist="0123456789/ ")
    text = " ".join(results).strip()

    # Parse "current/total" format (e.g., "1234/2500" or "1234 / 2500")
    match = re.search(r"(\d+)\s*/\s*(\d+)", text)
    if match:
        current = int(match.group(1))
        total = int(match.group(2))
        if total > 0:
            return {
                "current": current,
                "total": total,
                "percentage": current / total,
            }

    return None


def read_enemy_health_from_crop(crop: np.ndarray) -> float | None:
    """Estimate enemy health percentage from health bar via color detection.

    The side panel health bars fill horizontally with green.
    We detect where the green ends to calculate fill percentage.

    Args:
        crop: Cropped image of health bar region

    Returns:
        Health percentage 0.0-1.0 or None
    """
    if crop is None or crop.size == 0:
        return None

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Green health bar range (adjust if needed)
    green_lower = np.array([35, 100, 100])
    green_upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, green_lower, green_upper)

    # Find the rightmost column with green pixels
    # Health bars fill left-to-right, so rightmost green = current health
    col_has_green = np.any(mask > 0, axis=0)  # True for each column with green
    green_cols = np.where(col_has_green)[0]

    if len(green_cols) == 0:
        return 0.0  # No green = 0% health

    rightmost_green = green_cols[-1]
    bar_width = mask.shape[1]

    # Return fill percentage (rightmost green column / total width)
    return (rightmost_green + 1) / bar_width


def load_video_metadata(metadata_dir: Path, video_id: str) -> dict | None:
    """Load video metadata to get side info.

    Args:
        metadata_dir: Directory containing metadata JSON files
        video_id: Video ID

    Returns:
        Metadata dict or None if not found
    """
    metadata_path = metadata_dir / f"{video_id}.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return None


def process_frame(
    frame_path: Path,
    regions: dict,
    reader,
    garen_side: str,
) -> dict:
    """Process a single frame and extract state.

    Args:
        frame_path: Path to frame image
        regions: Dict of HUD regions
        reader: EasyOCR reader
        garen_side: Which side Garen is on ("red" or "blue")

    Returns:
        Dict with extracted state
    """
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return {"error": "Could not read frame"}

    state = {
        "frame": frame_path.name,
        "game_time_seconds": None,
        "garen_health": None,
        "garen_health_pct": None,
        "enemy_health_pct": None,
    }

    # Extract game clock
    if regions.get("game_clock"):
        x, y, w, h = regions["game_clock"]
        crop = frame[y:y+h, x:x+w]
        state["game_time_seconds"] = read_game_clock_from_crop(crop, reader)

    # Extract Garen's health (OCR text)
    if regions.get("garen_health_text"):
        x, y, w, h = regions["garen_health_text"]
        crop = frame[y:y+h, x:x+w]
        health_data = read_garen_health_from_crop(crop, reader)
        if health_data:
            state["garen_health"] = f"{health_data['current']}/{health_data['total']}"
            state["garen_health_pct"] = health_data["percentage"]

    # Extract enemy health (color detection on appropriate side)
    # If Garen is RED -> enemies on BLUE side (left panel)
    # If Garen is BLUE -> enemies on RED side (right panel)
    if garen_side == "red" and regions.get("enemy_health_blue_side"):
        x, y, w, h = regions["enemy_health_blue_side"]
        crop = frame[y:y+h, x:x+w]
        state["enemy_health_pct"] = read_enemy_health_from_crop(crop)
    elif garen_side == "blue" and regions.get("enemy_health_red_side"):
        x, y, w, h = regions["enemy_health_red_side"]
        crop = frame[y:y+h, x:x+w]
        state["enemy_health_pct"] = read_enemy_health_from_crop(crop)

    return state


def visualize_regions(
    frame_path: Path,
    regions: dict,
    garen_side: str,
    output_path: Path,
):
    """Draw rectangles on frame showing what regions are being captured.

    Args:
        frame_path: Path to frame image
        regions: Dict of HUD regions
        garen_side: Which side Garen is on
        output_path: Path to save visualization
    """
    frame = cv2.imread(str(frame_path))
    if frame is None:
        print(f"Could not read frame: {frame_path}")
        return

    # Draw each region
    colors = {
        "game_clock": (0, 255, 255),  # Yellow
        "garen_health_text": (0, 255, 0),  # Green
        "enemy_health_blue_side": (255, 0, 0),  # Blue
        "enemy_health_red_side": (0, 0, 255),  # Red
    }

    regions_to_draw = ["game_clock", "garen_health_text"]
    if garen_side == "red":
        regions_to_draw.append("enemy_health_blue_side")
    else:
        regions_to_draw.append("enemy_health_red_side")

    for region_name in regions_to_draw:
        if region_name in regions and regions[region_name]:
            x, y, w, h = regions[region_name]
            color = colors.get(region_name, (255, 255, 255))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(
                frame, region_name, (x, y-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
            )

    # Also create crops
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    print(f"Saved visualization to {output_path}")

    # Save individual crops
    for region_name in regions_to_draw:
        if region_name in regions and regions[region_name]:
            x, y, w, h = regions[region_name]
            crop = cv2.imread(str(frame_path))[y:y+h, x:x+w]
            crop_path = output_path.parent / f"crop_{region_name}.png"
            cv2.imwrite(str(crop_path), crop)
            print(f"Saved crop to {crop_path}")


def process_video_frames(
    video_dir: Path,
    output_dir: Path,
    metadata_dir: Path,
    regions: dict,
    sample_rate: int = 20,
    file_ext: str = "jpg",
) -> dict:
    """Process all frames in a video directory.

    Args:
        video_dir: Directory containing frames
        output_dir: Directory to save results
        metadata_dir: Directory containing video metadata
        regions: HUD region coordinates
        sample_rate: Process every Nth frame (20 = 1 per second at 20 FPS)
        file_ext: Frame file extension

    Returns:
        Stats dict
    """
    video_id = video_dir.name
    frames = sorted(video_dir.glob(f"frame_*.{file_ext}"))

    if not frames:
        return {"video_id": video_id, "status": "no_frames", "count": 0}

    # Load metadata to get Garen's side
    metadata = load_video_metadata(metadata_dir, video_id)
    garen_side = "unknown"
    if metadata:
        garen_side = metadata.get("side", "unknown")

    if garen_side == "unknown":
        print(f"WARNING: Unknown side for {video_id}, enemy health detection may not work")

    # Initialize OCR reader for this process
    reader = get_ocr_reader()

    # Sample frames
    sampled_frames = frames[::sample_rate]

    states = []
    clock_success = 0
    health_success = 0

    print(f"Processing {video_id}: {len(sampled_frames)} frames (sampled from {len(frames)})")
    print(f"  Garen side: {garen_side}")

    for i, frame_path in enumerate(sampled_frames):
        state = process_frame(frame_path, regions, reader, garen_side)

        # Add frame index (original, not sampled)
        frame_num = int(frame_path.stem.split("_")[1])
        state["frame_index"] = frame_num

        if state.get("game_time_seconds") is not None:
            clock_success += 1
        if state.get("garen_health_pct") is not None:
            health_success += 1

        states.append(state)

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  {video_id}: {i + 1}/{len(sampled_frames)} frames processed")

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{video_id}_states.json"

    with open(output_path, "w") as f:
        json.dump({
            "video_id": video_id,
            "garen_side": garen_side,
            "total_frames": len(frames),
            "sampled_frames": len(sampled_frames),
            "sample_rate": sample_rate,
            "clock_success_rate": clock_success / len(sampled_frames) if sampled_frames else 0,
            "health_success_rate": health_success / len(sampled_frames) if sampled_frames else 0,
            "states": states,
        }, f, indent=2)

    print(f"  {video_id}: Saved to {output_path}")
    print(f"    Clock OCR: {clock_success}/{len(sampled_frames)}")
    print(f"    Health OCR: {health_success}/{len(sampled_frames)}")

    return {
        "video_id": video_id,
        "status": "success",
        "garen_side": garen_side,
        "total_frames": len(frames),
        "sampled_frames": len(sampled_frames),
        "clock_success": clock_success,
        "health_success": health_success,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract game state via OCR")
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="data/processed/frames",
        help="Directory containing video subdirs with frames",
    )
    parser.add_argument(
        "--metadata-dir",
        type=str,
        default="data/raw/metadata",
        help="Directory containing video metadata JSON files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed/states",
        help="Directory to save extracted states",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=20,
        help="Process every Nth frame (20 = 1 per second at 20 FPS)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Process only this video ID (for testing)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Save visualization of detected regions (for testing)",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["red", "blue"],
        default=None,
        help="Override Garen's side (for testing when metadata is missing)",
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)

    # Get HUD regions (using replay HUD for domisumReplay videos)
    from ahriuwu.data.hud_regions import REPLAY_HUD_1080P
    regions = REPLAY_HUD_1080P

    # Find video directories
    if args.video:
        video_dirs = [frames_dir / args.video]
    else:
        video_dirs = [d for d in frames_dir.iterdir() if d.is_dir()]

    print("=" * 60)
    print("OCR State Extraction")
    print("=" * 60)
    print(f"Frames dir: {frames_dir}")
    print(f"Metadata dir: {metadata_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Sample rate: every {args.sample_rate} frames")
    print(f"Videos to process: {len(video_dirs)}")
    print("=" * 60)

    # If visualize mode, just show regions on first frame
    if args.visualize and video_dirs:
        video_dir = video_dirs[0]
        frames = sorted(video_dir.glob("frame_*.jpg")) or sorted(video_dir.glob("frame_*.png"))
        if frames:
            garen_side = args.side or "red"  # Default to red for visualization
            vis_path = Path("data/samples") / f"{video_dir.name}_regions.png"
            visualize_regions(frames[0], regions, garen_side, vis_path)
            print("\nVisualization complete. Check the output images.")
            return

    start_time = time.time()
    results = []

    # Process videos
    for video_dir in video_dirs:
        if not video_dir.is_dir():
            continue

        result = process_video_frames(
            video_dir,
            output_dir,
            metadata_dir,
            regions,
            sample_rate=args.sample_rate,
        )
        results.append(result)

    # Summary
    elapsed = time.time() - start_time
    total_sampled = sum(r.get("sampled_frames", 0) for r in results)
    total_clock = sum(r.get("clock_success", 0) for r in results)
    total_health = sum(r.get("health_success", 0) for r in results)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Videos processed: {len(results)}")
    print(f"Total frames sampled: {total_sampled}")
    if total_sampled > 0:
        print(f"Clock OCR success: {total_clock}/{total_sampled} ({100*total_clock/total_sampled:.1f}%)")
        print(f"Health OCR success: {total_health}/{total_sampled} ({100*total_health/total_sampled:.1f}%)")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
