#!/usr/bin/env python3
"""Extract game state from frames using OCR.

Processes all frames and extracts:
- Game clock (timestamp)
- Health percentages (via color detection)

Results are saved as JSON per video for later use in training.

Usage:
    python scripts/extract_ocr_states.py --frames-dir data/processed/frames
    python scripts/extract_ocr_states.py --frames-dir data/processed/frames --sample-rate 20
"""

import argparse
import json
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import cv2
import numpy as np

# Import OCR functions - these will be imported in worker processes
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
    import re

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
    match = re.search(r"(\d{1,2})[:.](\d{2})", text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return minutes * 60 + seconds

    return None


def read_health_from_crop(crop: np.ndarray) -> float | None:
    """Estimate health percentage from health bar region via color detection.

    Args:
        crop: Cropped image of health bar region

    Returns:
        Health percentage 0.0-1.0 or None
    """
    if crop is None or crop.size == 0:
        return None

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Green health bar range
    green_lower = np.array([35, 100, 100])
    green_upper = np.array([85, 255, 255])

    mask = cv2.inRange(hsv, green_lower, green_upper)

    # Find rightmost green pixel column
    col_sums = np.sum(mask, axis=0)
    green_cols = np.where(col_sums > 0)[0]

    if len(green_cols) == 0:
        return 0.0

    rightmost = green_cols[-1]
    return rightmost / mask.shape[1]


def process_frame(frame_path: Path, regions: dict, reader) -> dict:
    """Process a single frame and extract state.

    Args:
        frame_path: Path to frame image
        regions: Dict of HUD regions
        reader: EasyOCR reader

    Returns:
        Dict with extracted state
    """
    frame = cv2.imread(str(frame_path))
    if frame is None:
        return {"error": "Could not read frame"}

    state = {
        "frame": frame_path.name,
        "game_time_seconds": None,
        "player_health": None,
        "enemy_health": None,
    }

    # Extract game clock
    if regions.get("game_clock"):
        x, y, w, h = regions["game_clock"]
        crop = frame[y:y+h, x:x+w]
        state["game_time_seconds"] = read_game_clock_from_crop(crop, reader)

    # Extract health (if regions defined)
    if regions.get("player_health"):
        x, y, w, h = regions["player_health"]
        crop = frame[y:y+h, x:x+w]
        state["player_health"] = read_health_from_crop(crop)

    if regions.get("enemy_health"):
        x, y, w, h = regions["enemy_health"]
        crop = frame[y:y+h, x:x+w]
        state["enemy_health"] = read_health_from_crop(crop)

    return state


def process_video_frames(
    video_dir: Path,
    output_dir: Path,
    regions: dict,
    sample_rate: int = 20,
    file_ext: str = "jpg",
) -> dict:
    """Process all frames in a video directory.

    Args:
        video_dir: Directory containing frames
        output_dir: Directory to save results
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

    # Initialize OCR reader for this process
    reader = get_ocr_reader()

    # Sample frames
    sampled_frames = frames[::sample_rate]

    states = []
    success_count = 0

    print(f"Processing {video_id}: {len(sampled_frames)} frames (sampled from {len(frames)})")

    for i, frame_path in enumerate(sampled_frames):
        state = process_frame(frame_path, regions, reader)

        # Add frame index (original, not sampled)
        frame_num = int(frame_path.stem.split("_")[1])
        state["frame_index"] = frame_num

        if state.get("game_time_seconds") is not None:
            success_count += 1

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
            "total_frames": len(frames),
            "sampled_frames": len(sampled_frames),
            "sample_rate": sample_rate,
            "ocr_success_rate": success_count / len(sampled_frames) if sampled_frames else 0,
            "states": states,
        }, f, indent=2)

    print(f"  {video_id}: Saved to {output_path} (OCR success: {success_count}/{len(sampled_frames)})")

    return {
        "video_id": video_id,
        "status": "success",
        "total_frames": len(frames),
        "sampled_frames": len(sampled_frames),
        "ocr_success": success_count,
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
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (each loads its own OCR model)",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Process only this video ID (for testing)",
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
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
    print(f"Output dir: {output_dir}")
    print(f"Sample rate: every {args.sample_rate} frames")
    print(f"Videos to process: {len(video_dirs)}")
    print("=" * 60)

    start_time = time.time()
    results = []

    # Process videos (sequential for now - OCR is CPU-bound and each worker needs its own model)
    for video_dir in video_dirs:
        if not video_dir.is_dir():
            continue

        result = process_video_frames(
            video_dir,
            output_dir,
            regions,
            sample_rate=args.sample_rate,
        )
        results.append(result)

    # Summary
    elapsed = time.time() - start_time
    total_sampled = sum(r.get("sampled_frames", 0) for r in results)
    total_success = sum(r.get("ocr_success", 0) for r in results)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Videos processed: {len(results)}")
    print(f"Total frames sampled: {total_sampled}")
    print(f"OCR success rate: {total_success}/{total_sampled} ({100*total_success/total_sampled:.1f}%)" if total_sampled > 0 else "N/A")
    print(f"Time elapsed: {elapsed/60:.1f} minutes")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
