#!/usr/bin/env python3
"""Extract features using new consensus-based gold OCR and Garen tracking.

Uses the new GoldTextDetector with:
- OCR-based "Garen" text tracking (99.7% detection rate)
- Consensus-based gold filtering (3+ detections in 45 frames)
- Team-side aware health bar detection

Usage:
    # Process all videos
    python scripts/extract_features_v2.py

    # Process specific video
    python scripts/extract_features_v2.py --video 0Ax8Pudr5-o

    # Use GPU for faster OCR
    python scripts/extract_features_v2.py --gpu
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahriuwu.data.keylog_extractor import (
    GoldTextDetector,
    GarenHUDTracker,
    HUDRegionsNormalized,
)


def load_video_metadata(metadata_path: Path) -> dict | None:
    """Load video metadata."""
    if metadata_path.exists():
        with open(metadata_path) as f:
            return json.load(f)
    return None


def get_video_path(video_id: str, videos_dir: Path) -> Path | None:
    """Find video file for a given video ID."""
    # Try common extensions
    for ext in ['.mp4', '.mkv', '.webm']:
        # Try direct match
        path = videos_dir / f"{video_id}{ext}"
        if path.exists():
            return path
        # Try with format suffix (e.g., video_id.f299.mp4)
        for p in videos_dir.glob(f"{video_id}*{ext}"):
            if p.exists():
                return p
    return None


def process_video(
    video_path: Path,
    metadata: dict,
    output_dir: Path,
    use_gpu: bool = False,
    output_fps: int = 20,
    output_resolution: tuple[int, int] = (256, 256),
) -> dict:
    """Process a single video and extract features.

    Args:
        video_path: Path to video file
        metadata: Video metadata dict
        output_dir: Directory to save features and frames
        use_gpu: Use GPU for OCR
        output_fps: Output frame rate for extracted frames
        output_resolution: Resolution for saved frames (for tokenizer)

    Returns:
        Stats dict
    """
    video_id = metadata["video_id"]
    team_side = metadata.get("side", "blue")
    if team_side == "unknown":
        team_side = "blue"  # Default assumption

    print(f"\nProcessing {video_id} (team_side={team_side})")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video_id": video_id, "status": "error", "error": "Could not open video"}

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Source: {source_width}x{source_height} @ {source_fps:.1f} fps, {total_frames} frames")

    # Frame sampling: extract at output_fps
    frame_interval = max(1, int(source_fps / output_fps))

    # Initialize detectors
    gold_detector = GoldTextDetector(
        normalized_regions=HUDRegionsNormalized(),
        frame_width=source_width,
        frame_height=source_height,
        team_side=team_side,
        use_gpu=use_gpu,
    )

    hud_tracker = GarenHUDTracker(HUDRegionsNormalized())

    # Create output directories
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    # Process frames
    features = {
        "video_id": video_id,
        "source_resolution": f"{source_width}x{source_height}",
        "source_fps": source_fps,
        "output_fps": output_fps,
        "output_resolution": f"{output_resolution[0]}x{output_resolution[1]}",
        "team_side": team_side,
        "frames": [],
    }

    frame_idx = 0
    output_frame_idx = 0
    prev_frame = None

    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process frames at output_fps rate
        if frame_idx % frame_interval == 0:
            # Get movement from optical flow
            movement_dx, movement_dy, movement_conf = 0.0, 0.0, 0.0
            if prev_frame is not None:
                movement_dx, movement_dy, movement_conf = hud_tracker.compute_movement(
                    prev_frame, frame
                )

            # Get movement slice (0-17 for 18 directions)
            movement_slice = hud_tracker.movement_to_wasd(movement_dx, movement_dy)
            if movement_slice is None:
                movement_slice = 0
            else:
                # Convert WASD to slice index
                wasd_to_slice = {
                    'D': 1, 'WD': 2, 'W': 5, 'WA': 8, 'A': 10,
                    'SA': 12, 'S': 14, 'SD': 16,
                }
                movement_slice = wasd_to_slice.get(movement_slice, 0)

            # Run gold detection
            gold_gains, health_bar = gold_detector.detect_gold_text(frame, output_frame_idx)

            # Get health bar position
            hb_x, hb_y = 0, 0
            if health_bar is not None:
                hb_x, hb_y = health_bar[0], health_bar[1]

            # Sum gold gains for this frame
            total_gold = sum(g[0] for g in gold_gains) if gold_gains else 0

            # Build frame features
            frame_features = {
                "frame_idx": output_frame_idx,
                "timestamp_ms": (frame_idx / source_fps) * 1000,
                "movement_dx": float(movement_dx),
                "movement_dy": float(movement_dy),
                "movement_slice": movement_slice,
                "movement_confidence": float(movement_conf),
                "ability_q": False,  # TODO: detect from keylog overlay
                "ability_w": False,
                "ability_e": False,
                "ability_r": False,
                "summoner_d": False,
                "summoner_f": False,
                "item_used": False,
                "recall_b": False,
                "gold_gained": total_gold,
                "health_bar_x": hb_x,
                "health_bar_y": hb_y,
            }
            features["frames"].append(frame_features)

            # Save resized frame for tokenizer
            resized = cv2.resize(frame, output_resolution, interpolation=cv2.INTER_AREA)
            frame_path = frames_dir / f"frame_{output_frame_idx:06d}.jpg"
            cv2.imwrite(str(frame_path), resized, [cv2.IMWRITE_JPEG_QUALITY, 95])

            prev_frame = frame.copy()
            output_frame_idx += 1

            # Progress
            if output_frame_idx % 500 == 0:
                elapsed = time.time() - start_time
                fps = output_frame_idx / elapsed
                eta = (total_frames / frame_interval - output_frame_idx) / fps
                print(f"  {output_frame_idx} frames ({fps:.1f} fps, ETA: {eta/60:.1f} min)")

        frame_idx += 1

    cap.release()

    # Update frame count
    features["num_frames"] = output_frame_idx

    # Save features
    features_path = output_dir / "features.json"
    with open(features_path, "w") as f:
        json.dump(features, f)

    # Stats
    gold_frames = sum(1 for f in features["frames"] if f["gold_gained"] > 0)
    elapsed = time.time() - start_time

    print(f"  Done: {output_frame_idx} frames in {elapsed/60:.1f} min")
    print(f"  Gold events: {gold_frames} frames ({100*gold_frames/output_frame_idx:.2f}%)")

    return {
        "video_id": video_id,
        "status": "success",
        "num_frames": output_frame_idx,
        "gold_frames": gold_frames,
        "time_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Extract features with new gold OCR")
    parser.add_argument(
        "--videos-dir",
        type=str,
        default="data/raw/videos",
        help="Directory containing video files",
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
        default="data/processed",
        help="Directory to save extracted features",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Process only this video ID",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU for OCR (faster)",
    )
    parser.add_argument(
        "--output-fps",
        type=int,
        default=20,
        help="Output frame rate",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have features.json",
    )
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    metadata_dir = Path(args.metadata_dir)
    output_dir = Path(args.output_dir)

    # Find videos to process
    if args.video:
        metadata_files = [metadata_dir / f"{args.video}.json"]
    else:
        metadata_files = sorted(metadata_dir.glob("*.json"))

    print("=" * 60)
    print("Feature Extraction v2 (Consensus Gold OCR)")
    print("=" * 60)
    print(f"Videos dir: {videos_dir}")
    print(f"Metadata dir: {metadata_dir}")
    print(f"Output dir: {output_dir}")
    print(f"GPU: {args.gpu}")
    print(f"Output FPS: {args.output_fps}")
    print(f"Videos to process: {len(metadata_files)}")
    print("=" * 60)

    results = []
    skipped = 0

    for metadata_path in metadata_files:
        if not metadata_path.exists():
            print(f"Metadata not found: {metadata_path}")
            continue

        metadata = load_video_metadata(metadata_path)
        if not metadata:
            continue

        video_id = metadata["video_id"]
        video_output_dir = output_dir / video_id

        # Skip if already processed
        if args.skip_existing and (video_output_dir / "features.json").exists():
            skipped += 1
            continue

        # Find video file
        video_path = get_video_path(video_id, videos_dir)
        if not video_path:
            print(f"Video not found: {video_id}")
            continue

        # Process
        result = process_video(
            video_path=video_path,
            metadata=metadata,
            output_dir=video_output_dir,
            use_gpu=args.gpu,
            output_fps=args.output_fps,
        )
        results.append(result)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    successful = [r for r in results if r["status"] == "success"]
    print(f"Processed: {len(successful)}/{len(results)}")
    print(f"Skipped (existing): {skipped}")
    if successful:
        total_frames = sum(r["num_frames"] for r in successful)
        total_gold = sum(r["gold_frames"] for r in successful)
        total_time = sum(r["time_seconds"] for r in successful)
        print(f"Total frames: {total_frames:,}")
        print(f"Total gold events: {total_gold:,} ({100*total_gold/total_frames:.2f}%)")
        print(f"Total time: {total_time/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
