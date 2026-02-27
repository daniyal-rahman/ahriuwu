#!/usr/bin/env python3
"""Re-extract gold using new consensus OCR system.

Only updates gold_gained in existing features.json files.
Does NOT re-extract frames or other features.

Usage:
    python scripts/reextract_gold.py --gpu
    python scripts/reextract_gold.py --gpu --resume
    python scripts/reextract_gold.py --gpu --max-minutes 15  # laning phase only
    python scripts/reextract_gold.py --video 0Ax8Pudr5-o --gpu
"""

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahriuwu.data.keylog_extractor import GoldTextDetector, HUDRegionsNormalized

PROGRESS_FILE = Path(__file__).parent.parent / "data" / ".reextract_gold_progress.json"


def load_progress() -> set[str]:
    """Load set of completed video IDs from progress file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return set(json.load(f).get("completed", []))
    return set()


def save_progress(completed: set[str]):
    """Save completed video IDs to progress file."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"completed": sorted(completed)}, f)


def get_video_path(video_id: str, videos_dir: Path) -> Path | None:
    """Find video file for a given video ID."""
    for ext in ['.mp4', '.mkv', '.webm']:
        path = videos_dir / f"{video_id}{ext}"
        if path.exists():
            return path
        for p in videos_dir.glob(f"{video_id}*{ext}"):
            if p.exists() and not str(p).endswith('.part'):
                return p
    return None


def reextract_gold(
    video_path: Path,
    features_path: Path,
    use_gpu: bool = False,
    max_minutes: float | None = None,
) -> dict:
    """Re-extract gold from video and update features.json.

    Args:
        video_path: Path to video file
        features_path: Path to existing features.json
        use_gpu: Use GPU for OCR

    Returns:
        Stats dict
    """
    # Load existing features
    with open(features_path) as f:
        features = json.load(f)

    video_id = features["video_id"]
    team_side = features.get("team_side", "blue")
    if team_side == "unknown":
        team_side = "blue"

    frames_data = features.get("frames", [])
    if not frames_data:
        return {"video_id": video_id, "status": "error", "error": "No frames in features.json"}

    print(f"\nRe-extracting gold for {video_id} (team_side={team_side})")

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"video_id": video_id, "status": "error", "error": "Could not open video"}

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    source_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_fps = features.get("output_fps", 20)
    frame_interval = max(1, int(source_fps / output_fps))

    # Limit to max_minutes if specified
    total_frames_count = len(frames_data)
    if max_minutes is not None:
        max_frame_idx = int(max_minutes * 60 * output_fps)
        frames_data = [f for f in frames_data if f["frame_idx"] < max_frame_idx]

    print(f"  Source: {source_width}x{source_height} @ {source_fps:.1f} fps")
    if max_minutes is not None:
        print(f"  Limiting to first {max_minutes} minutes ({len(frames_data)}/{total_frames_count} frames)")
    else:
        print(f"  Frames to process: {len(frames_data)}")

    # Initialize detector
    gold_detector = GoldTextDetector(
        normalized_regions=HUDRegionsNormalized(),
        frame_width=source_width,
        frame_height=source_height,
        team_side=team_side,
        use_gpu=use_gpu,
    )

    start_time = time.time()
    gold_events = 0
    processed = 0

    # Process each frame that exists in features
    for i, frame_data in enumerate(frames_data):
        output_frame_idx = frame_data["frame_idx"]
        source_frame_idx = output_frame_idx * frame_interval

        # Seek to frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        # Run gold detection
        gold_gains, health_bar = gold_detector.detect_gold_text(frame, output_frame_idx)

        # Update gold_gained
        total_gold = sum(g[0] for g in gold_gains) if gold_gains else 0
        frames_data[i]["gold_gained"] = total_gold

        if total_gold > 0:
            gold_events += 1

        # Update health bar position if detected
        if health_bar is not None:
            frames_data[i]["health_bar_x"] = health_bar[0]
            frames_data[i]["health_bar_y"] = health_bar[1]

        processed += 1

        if processed % 500 == 0:
            elapsed = time.time() - start_time
            fps = processed / elapsed
            eta = (len(frames_data) - processed) / fps
            print(f"  {processed}/{len(frames_data)} frames ({fps:.1f} fps, ETA: {eta/60:.1f} min)")

    cap.release()

    # Save updated features
    features["frames"] = frames_data
    with open(features_path, "w") as f:
        json.dump(features, f)

    elapsed = time.time() - start_time
    print(f"  Done: {processed} frames in {elapsed/60:.1f} min")
    print(f"  Gold events: {gold_events} ({100*gold_events/processed:.2f}%)")

    return {
        "video_id": video_id,
        "status": "success",
        "frames": processed,
        "gold_events": gold_events,
        "time_seconds": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="Re-extract gold with new OCR system")
    parser.add_argument("--videos-dir", type=str, default="data/raw/videos")
    parser.add_argument("--processed-dir", type=str, default="data/processed")
    parser.add_argument("--video", type=str, default=None, help="Process only this video ID")
    parser.add_argument("--gpu", action="store_true", help="Use GPU for OCR")
    parser.add_argument("--resume", action="store_true", help="Resume from last run, skipping completed videos")
    parser.add_argument("--max-minutes", type=float, default=None, help="Only process first N minutes of each video (e.g., 15 for laning phase)")
    args = parser.parse_args()

    videos_dir = Path(args.videos_dir)
    processed_dir = Path(args.processed_dir)

    # Find videos to process
    if args.video:
        video_dirs = [processed_dir / args.video]
    else:
        video_dirs = sorted(processed_dir.iterdir())

    # Load progress if resuming
    completed = load_progress() if args.resume else set()

    print("=" * 60)
    print("Gold Re-extraction (Consensus OCR)")
    print("=" * 60)
    print(f"GPU: {args.gpu}")
    print(f"Resume: {args.resume}")
    if args.max_minutes:
        print(f"Max minutes: {args.max_minutes}")
    if args.resume and completed:
        print(f"Already completed: {len(completed)} videos")
    print(f"Videos to process: {len(video_dirs)}")
    print("=" * 60)

    results = []

    for video_dir in video_dirs:
        if not video_dir.is_dir():
            continue

        video_id = video_dir.name
        features_path = video_dir / "features.json"

        # Skip if already completed (resume mode)
        if video_id in completed:
            print(f"Skipping {video_id} (already completed)")
            continue

        if not features_path.exists():
            print(f"No features.json for {video_id}, skipping")
            continue

        # Find video file
        video_path = get_video_path(video_id, videos_dir)
        if not video_path:
            print(f"Video not found: {video_id}")
            continue

        result = reextract_gold(
            video_path=video_path,
            features_path=features_path,
            use_gpu=args.gpu,
            max_minutes=args.max_minutes,
        )
        results.append(result)

        # Save progress after each successful video
        if result["status"] == "success":
            completed.add(video_id)
            save_progress(completed)

    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    successful = [r for r in results if r["status"] == "success"]
    print(f"Processed: {len(successful)}/{len(results)}")
    if successful:
        total_frames = sum(r["frames"] for r in successful)
        total_gold = sum(r["gold_events"] for r in successful)
        total_time = sum(r["time_seconds"] for r in successful)
        print(f"Total frames: {total_frames:,}")
        print(f"Total gold events: {total_gold:,} ({100*total_gold/total_frames:.2f}%)")
        print(f"Total time: {total_time/60:.1f} min")
    print("=" * 60)


if __name__ == "__main__":
    main()
