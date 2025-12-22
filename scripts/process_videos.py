#!/usr/bin/env python3
"""Process downloaded videos: extract frames, run OCR, compute rewards.

Usage:
    # Process all downloaded videos
    python scripts/process_videos.py

    # Process a specific video
    python scripts/process_videos.py --video VIDEO_ID

    # Only extract frames (skip OCR)
    python scripts/process_videos.py --frames-only
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahriuwu.data import (
    extract_frames,
    is_extracted,
    get_hud_regions,
    VideoMetadata,
)
from ahriuwu.ocr import GameStateReader
from ahriuwu.rewards import RewardExtractor
import cv2


def process_video(
    video_path: Path,
    metadata: VideoMetadata,
    output_dir: Path,
    frames_only: bool = False,
) -> dict:
    """Process a single video.

    Returns dict with processing stats.
    """
    video_id = metadata.video_id
    frames_dir = output_dir / "frames" / video_id
    states_file = output_dir / "states" / f"{video_id}_states.json"

    stats = {"video_id": video_id, "frames": 0, "states": 0}

    # Extract frames
    if not is_extracted(frames_dir):
        print(f"  Extracting frames...")
        num_frames = extract_frames(video_path, frames_dir, fps=20)
        stats["frames"] = num_frames
        print(f"  Extracted {num_frames} frames")
    else:
        stats["frames"] = len(list(frames_dir.glob("frame_*.png")))
        print(f"  Frames already extracted ({stats['frames']} frames)")

    if frames_only:
        return stats

    # Run OCR and compute rewards
    if not states_file.exists():
        print(f"  Running OCR...")
        states_file.parent.mkdir(parents=True, exist_ok=True)

        try:
            hud_regions = get_hud_regions(metadata.channel)
        except ValueError:
            print(f"  Warning: Unknown channel {metadata.channel}, skipping OCR")
            return stats

        reader = GameStateReader(hud_regions)
        reward_extractor = RewardExtractor()

        states = []
        frame_files = sorted(frames_dir.glob("frame_*.png"))

        for i, frame_path in enumerate(frame_files):
            frame = cv2.imread(str(frame_path))
            if frame is None:
                continue

            state = reader.read_frame(frame)
            reward_info = reward_extractor.compute_reward(state)
            state["reward"] = reward_info.total
            state["reward_breakdown"] = {
                "gold": reward_info.gold_reward,
                "health": reward_info.health_reward,
                "death": reward_info.death_reward,
            }
            states.append(state)

            if (i + 1) % 500 == 0:
                print(f"    Processed {i + 1}/{len(frame_files)} frames")

        states_file.write_text(json.dumps(states, indent=2))
        stats["states"] = len(states)
        print(f"  Saved {len(states)} states")
    else:
        existing = json.loads(states_file.read_text())
        stats["states"] = len(existing)
        print(f"  States already exist ({stats['states']} states)")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Process downloaded videos")
    parser.add_argument(
        "--data-dir", "-d",
        default="data/raw",
        help="Data directory (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="data/processed",
        help="Output directory (default: data/processed)",
    )
    parser.add_argument(
        "--video", "-v",
        help="Process only this video ID",
    )
    parser.add_argument(
        "--frames-only", "-f",
        action="store_true",
        help="Only extract frames, skip OCR",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos_dir = data_dir / "videos"
    metadata_dir = data_dir / "metadata"

    if not metadata_dir.exists():
        print("No metadata found. Run download_youtube.py first.")
        sys.exit(1)

    # Get videos to process
    metadata_files = list(metadata_dir.glob("*.json"))
    if args.video:
        metadata_files = [f for f in metadata_files if f.stem == args.video]

    if not metadata_files:
        print("No videos to process.")
        sys.exit(1)

    print(f"Processing {len(metadata_files)} videos...")

    total_stats = {"videos": 0, "frames": 0, "states": 0}

    for meta_file in metadata_files:
        metadata = VideoMetadata.from_json(meta_file.read_text())
        video_path = videos_dir / f"{metadata.video_id}.mp4"

        if not video_path.exists():
            print(f"Skipping {metadata.video_id}: video file not found")
            continue

        print(f"\nProcessing: {metadata.video_id}")
        print(f"  Title: {metadata.title[:60]}...")

        stats = process_video(video_path, metadata, output_dir, args.frames_only)

        total_stats["videos"] += 1
        total_stats["frames"] += stats["frames"]
        total_stats["states"] += stats["states"]

    print(f"\n--- Done ---")
    print(f"Videos: {total_stats['videos']}")
    print(f"Frames: {total_stats['frames']}")
    print(f"States: {total_stats['states']}")


if __name__ == "__main__":
    main()
