#!/usr/bin/env python3
"""Pre-cache resized frames as uint8 numpy arrays for faster training.

This eliminates cv2.imread + resize overhead during training by pre-processing
all frames once and saving them as .npy files.

Storage: 256x256x3 bytes = 196KB per frame (vs ~33KB JPEG)
Auto-stops when disk space runs low (keeps 100GB buffer by default).

Usage:
    python scripts/precache_frames.py --frames-dir data/processed/frames
    python scripts/precache_frames.py --frames-dir data/processed/frames --buffer-gb 50
"""

import argparse
import shutil
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import cv2
import numpy as np
from tqdm import tqdm


def get_free_space_gb(path: Path) -> float:
    """Get free disk space in GB for the given path."""
    total, used, free = shutil.disk_usage(path)
    return free / (1024**3)


def process_frame(args: tuple) -> tuple[str, bool, int]:
    """Process a single frame: load, resize, save as .npy.

    Returns:
        Tuple of (frame_path, success, bytes_written)
    """
    frame_path, output_path, target_size = args

    try:
        output_path = Path(output_path)

        # Skip if already exists
        if output_path.exists():
            return (frame_path, True, 0)

        # Load and resize
        frame = cv2.imread(frame_path)
        if frame is None:
            return (frame_path, False, 0)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, target_size, interpolation=cv2.INTER_AREA)

        # Save as uint8 numpy (no compression for speed)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, frame)

        return (frame_path, True, frame.nbytes)
    except Exception:
        return (frame_path, False, 0)


def main():
    parser = argparse.ArgumentParser(description="Pre-cache resized frames as numpy arrays")
    parser.add_argument(
        "--frames-dir",
        type=str,
        default="data/processed/frames",
        help="Directory containing video subdirs with frames",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: frames_dir/../frames_cache)",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Target size (width, height)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of workers (default: CPU count)",
    )
    parser.add_argument(
        "--buffer-gb",
        type=float,
        default=100,
        help="Keep this much free disk space in GB (default: 100)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip frames that already have cached files",
    )
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    output_dir = Path(args.output_dir) if args.output_dir else frames_dir.parent / "frames_cache"
    target_size = tuple(args.target_size)
    num_workers = args.num_workers or cpu_count()

    # Calculate limits
    output_dir.mkdir(parents=True, exist_ok=True)
    free_space_gb = get_free_space_gb(output_dir)
    usable_space_gb = max(0, free_space_gb - args.buffer_gb)
    frame_size_bytes = target_size[0] * target_size[1] * 3 + 128  # +128 for npy header
    max_frames = int((usable_space_gb * 1024**3) / frame_size_bytes)

    print("=" * 60)
    print("Frame Pre-caching")
    print("=" * 60)
    print(f"Frames dir: {frames_dir}")
    print(f"Output dir: {output_dir}")
    print(f"Target size: {target_size}")
    print(f"Workers: {num_workers}")
    print(f"Free space: {free_space_gb:.1f} GB")
    print(f"Buffer: {args.buffer_gb:.0f} GB")
    print(f"Usable space: {usable_space_gb:.1f} GB")
    print(f"Max frames: {max_frames:,}")
    print("=" * 60)

    # Index all frames
    print("\nIndexing frames...")
    frame_items = []

    for video_dir in sorted(frames_dir.iterdir()):
        if not video_dir.is_dir():
            continue

        video_id = video_dir.name
        output_video_dir = output_dir / video_id

        for frame_path in sorted(video_dir.glob("frame_*.jpg")):
            output_path = output_video_dir / f"{frame_path.stem}.npy"

            if args.resume and output_path.exists():
                continue

            frame_items.append((str(frame_path), str(output_path), target_size))

            if len(frame_items) >= max_frames:
                break

        if len(frame_items) >= max_frames:
            break

    print(f"Will process {len(frame_items):,} frames (limited by disk space)")

    if len(frame_items) == 0:
        print("No frames to process!")
        return

    # Process frames
    print(f"\nProcessing with {num_workers} workers...")
    start_time = time.time()

    success_count = 0
    fail_count = 0
    bytes_written = 0

    with Pool(num_workers) as pool:
        for frame_path, success, nbytes in tqdm(
            pool.imap_unordered(process_frame, frame_items),
            total=len(frame_items),
            desc="Caching",
        ):
            if success:
                success_count += 1
                bytes_written += nbytes
            else:
                fail_count += 1

    elapsed = time.time() - start_time
    fps = success_count / elapsed if elapsed > 0 else 0

    print("\n" + "=" * 60)
    print("Pre-caching Complete")
    print("=" * 60)
    print(f"Frames cached: {success_count:,}")
    print(f"Frames failed: {fail_count}")
    print(f"Data written: {bytes_written / 1024**3:.1f} GB")
    print(f"Time: {elapsed / 60:.1f} minutes ({fps:.0f} frames/sec)")
    print(f"Output: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
