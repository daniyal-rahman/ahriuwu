#!/usr/bin/env python3
"""Extract frames from replay .avi recordings and apply HUD mask.

Replay recordings are captured at 8x game speed. This script extracts
one frame per game tick (every 8th video frame) to reconstruct 20fps
game-time footage, applies the same HUD mask used for YouTube videos,
and saves as 352x352 JPGs.

Usage:
    python scripts/prepare_data/extract_replay_frames.py \
        --replays-dir /mnt/storage/ahriuwu-data/replays \
        --output-dir /mnt/storage/ahriuwu-data/frames/replay_frames

    # Single replay
    python scripts/prepare_data/extract_replay_frames.py \
        --replay /mnt/storage/ahriuwu-data/replays/NA1-5496350713/replay.avi \
        --output-dir /mnt/storage/ahriuwu-data/frames/replay_frames
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm


# HUD mask regions at 1080p (scaled from 360p mask used for YouTube).
# YouTube pipeline uses 640x360 coordinates; replay .avi is 1920x1080.
# Scale factor: 1920/640 = 3x horizontal, 1080/360 = 3x vertical.
# Each is (y1, y2, x1, x2) for direct numpy slicing.
MASK_REGIONS_1080P = [
    (0, 105, 0, 1920),        # top scoreboard (full width)
    (105, 645, 0, 105),       # left champion cards (blue team)
    (105, 645, 1815, 1920),   # right champion cards (red team)
    (771, 1080, 0, 315),      # bottom-left: garen HUD + watermark
    (846, 1080, 315, 1650),   # bottom-center: scorecard + items
]

# Same mask at 360p for reference (matches download_yt_frames.py)
MASK_REGIONS_360P = [
    (0, 35, 0, 640),
    (35, 215, 0, 35),
    (35, 215, 605, 640),
    (257, 360, 0, 105),
    (282, 360, 105, 550),
]

TARGET_SIZE = (352, 352)
REPLAY_SPEED = 8  # Game runs at 8x during recording
VIDEO_FPS = 20    # Recording FPS
GAME_FPS = 20     # Target game-time FPS


def apply_mask_1080p(frame: np.ndarray) -> np.ndarray:
    """Apply HUD mask to a 1920x1080 frame. Zeros out HUD regions."""
    for y1, y2, x1, x2 in MASK_REGIONS_1080P:
        frame[y1:y2, x1:x2] = 0
    return frame


def extract_replay(
    video_path: Path,
    output_dir: Path,
    replay_speed: int = REPLAY_SPEED,
    skip_start_s: float = 5.0,
    skip_end_s: float = 5.0,
) -> int:
    """Extract game-time frames from a sped-up replay recording.

    Since the replay runs at `replay_speed`x, we sample every
    `replay_speed`th video frame to get 1 frame per game tick.

    Args:
        video_path: Path to replay .avi file
        output_dir: Output directory for extracted frames
        replay_speed: Game speed multiplier during recording (default 8)
        skip_start_s: Skip first N seconds of VIDEO time (loading screen)
        skip_end_s: Skip last N seconds of VIDEO time (end screen)

    Returns:
        Number of frames extracted
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  ERROR: Cannot open {video_path}")
        return 0

    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or VIDEO_FPS
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Skip start/end
    start_frame = int(skip_start_s * video_fps)
    end_frame = total_video_frames - int(skip_end_s * video_fps)

    # Sample every Nth frame (N = replay_speed) to get game-time fps
    frame_step = replay_speed

    output_dir.mkdir(parents=True, exist_ok=True)
    match_id = video_path.parent.name

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    game_frame_num = 0
    extracted = 0

    pbar = tqdm(
        total=(end_frame - start_frame) // frame_step,
        desc=f"  {match_id}",
        unit="frame",
    )

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if (frame_idx - start_frame) % frame_step == 0:
            # Apply HUD mask based on resolution
            if height >= 1080:
                frame = apply_mask_1080p(frame)
            else:
                # Scale mask for non-1080p (shouldn't happen with current data)
                scale_y = height / 1080
                scale_x = width / 1920
                for y1, y2, x1, x2 in MASK_REGIONS_1080P:
                    sy1, sy2 = int(y1 * scale_y), int(y2 * scale_y)
                    sx1, sx2 = int(x1 * scale_x), int(x2 * scale_x)
                    frame[sy1:sy2, sx1:sx2] = 0

            # Resize to target
            frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)

            # Save
            output_path = output_dir / f"frame_{game_frame_num:06d}.jpg"
            cv2.imwrite(str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            game_frame_num += 1
            extracted += 1
            pbar.update(1)

        frame_idx += 1

    pbar.close()
    cap.release()

    game_duration_s = game_frame_num / GAME_FPS
    print(f"  {match_id}: {extracted} frames ({game_duration_s:.0f}s game time)")
    return extracted


def main():
    parser = argparse.ArgumentParser(description="Extract masked frames from replay recordings")
    parser.add_argument("--replays-dir", type=str, default="/mnt/storage/ahriuwu-data/replays",
                        help="Directory containing replay subdirs (NA1-*/replay.avi)")
    parser.add_argument("--replay", type=str, default=None,
                        help="Single replay .avi to process (overrides --replays-dir)")
    parser.add_argument("--output-dir", type=str, default="/mnt/storage/ahriuwu-data/frames/replay_frames",
                        help="Output directory for extracted frames")
    parser.add_argument("--replay-speed", type=int, default=8,
                        help="Replay speed multiplier (default 8)")
    parser.add_argument("--skip-start", type=float, default=5.0,
                        help="Skip first N seconds of video (loading)")
    parser.add_argument("--skip-end", type=float, default=5.0,
                        help="Skip last N seconds of video (end screen)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-extracted replays")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.replay:
        # Single replay
        replay_path = Path(args.replay)
        match_id = replay_path.parent.name
        extract_replay(replay_path, output_dir / match_id,
                       replay_speed=args.replay_speed,
                       skip_start_s=args.skip_start,
                       skip_end_s=args.skip_end)
    else:
        # All replays
        replays_dir = Path(args.replays_dir)
        replay_dirs = sorted(replays_dir.glob("NA1-*"))
        print(f"Found {len(replay_dirs)} replay directories")

        total_extracted = 0
        for replay_dir in replay_dirs:
            video_path = replay_dir / "replay.avi"
            if not video_path.exists():
                print(f"  Skipping {replay_dir.name}: no replay.avi")
                continue

            match_output = output_dir / replay_dir.name
            if args.resume and match_output.exists() and any(match_output.iterdir()):
                print(f"  Skipping {replay_dir.name}: already extracted")
                continue

            n = extract_replay(video_path, match_output,
                               replay_speed=args.replay_speed,
                               skip_start_s=args.skip_start,
                               skip_end_s=args.skip_end)
            total_extracted += n

        print(f"\nTotal: {total_extracted} frames extracted")


if __name__ == "__main__":
    main()
