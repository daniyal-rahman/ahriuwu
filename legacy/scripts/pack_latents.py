#!/usr/bin/env python3
"""Pack per-frame latent files into chunked files for faster I/O.

Converts individual latent_XXXXXX.npy files into packed video files.
This dramatically reduces file open/close overhead and enables sequential reads.

Usage:
    # Pack CNN latents (256-dim)
    python scripts/pack_latents.py --input-dir data/processed/latents_cnn --output-dir data/processed/latents_cnn_packed

    # Pack transformer latents (32-dim)
    python scripts/pack_latents.py --input-dir data/processed/latents --output-dir data/processed/latents_packed

Output format:
    {output_dir}/{video_id}.npz containing:
        - 'latents': (num_frames, C, H, W) float16 array
        - 'frame_indices': (num_frames,) int32 array of original frame numbers

Expected speedup: 10-50x for sequence loading.
"""

import argparse
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm


def pack_video_latents_worker(args: tuple) -> tuple:
    """Worker function for multiprocessing."""
    video_dir, output_path = args
    stats = pack_video_latents(video_dir, output_path)
    return (video_dir, stats)


def pack_video_latents(video_dir: Path, output_path: Path) -> dict:
    """Pack all latent files from a video directory into a single file.

    Returns:
        dict with 'num_frames', 'shape', 'size_mb' for stats
    """
    # Find all latent files
    latent_files = sorted(video_dir.glob("latent_*.npy"))
    if not latent_files:
        return None

    # Extract frame numbers and sort
    frame_data = []
    for f in latent_files:
        try:
            frame_num = int(f.stem.split("_")[1])
            frame_data.append((frame_num, f))
        except (ValueError, IndexError):
            continue

    if not frame_data:
        return None

    # Sort by frame number
    frame_data.sort(key=lambda x: x[0])

    # Load all latents
    frame_indices = []
    latents = []

    for frame_num, latent_path in frame_data:
        latent = np.load(latent_path)
        latents.append(latent)
        frame_indices.append(frame_num)

    # Stack into single array
    latents_array = np.stack(latents, axis=0)  # (num_frames, C, H, W)
    frame_indices_array = np.array(frame_indices, dtype=np.int32)

    # Save as compressed npz
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        latents=latents_array,
        frame_indices=frame_indices_array,
    )

    # Return stats
    size_mb = output_path.stat().st_size / (1024 * 1024)
    return {
        'num_frames': len(latents),
        'shape': latents_array.shape,
        'size_mb': size_mb,
    }


def main():
    parser = argparse.ArgumentParser(description="Pack per-frame latents into video files")
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing per-frame latents (video_id/latent_XXXXXX.npy)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save packed latent files",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos that already have packed files",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (default: 1)",
    )
    parser.add_argument(
        "--delete-source",
        action="store_true",
        help="Delete source directory after successful packing (saves disk space)",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print("=" * 60)
    print("Packing Latent Files")
    print("=" * 60)
    print(f"Input dir: {input_dir}")
    print(f"Output dir: {output_dir}")
    print("=" * 60)

    # Find all video directories
    video_dirs = sorted([d for d in input_dir.iterdir() if d.is_dir()])
    print(f"Found {len(video_dirs)} video directories")

    output_dir.mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_size_mb = 0

    # Build work list
    work_items = []
    for video_dir in video_dirs:
        video_id = video_dir.name
        output_path = output_dir / f"{video_id}.npz"

        # Skip if exists and --skip-existing
        if args.skip_existing and output_path.exists():
            continue

        work_items.append((video_dir, output_path))

    print(f"Videos to process: {len(work_items)} (skipped {len(video_dirs) - len(work_items)} existing)")
    if args.delete_source:
        print("WARNING: Will delete source directories after packing!")

    deleted_count = 0

    if args.workers > 1:
        # Parallel processing
        print(f"Using {args.workers} parallel workers")
        with Pool(args.workers) as pool:
            for video_dir, stats in tqdm(
                pool.imap_unordered(pack_video_latents_worker, work_items),
                total=len(work_items),
                desc="Packing videos",
            ):
                if stats:
                    total_frames += stats['num_frames']
                    total_size_mb += stats['size_mb']
                    # Delete source after successful pack
                    if args.delete_source and video_dir.exists():
                        shutil.rmtree(video_dir)
                        deleted_count += 1
    else:
        # Sequential processing
        for video_dir, output_path in tqdm(work_items, desc="Packing videos"):
            stats = pack_video_latents(video_dir, output_path)
            if stats:
                total_frames += stats['num_frames']
                total_size_mb += stats['size_mb']
                # Delete source after successful pack
                if args.delete_source and video_dir.exists():
                    shutil.rmtree(video_dir)
                    deleted_count += 1

    print("\n" + "=" * 60)
    print("Packing Complete")
    print("=" * 60)
    print(f"Total frames packed: {total_frames:,}")
    print(f"Total size: {total_size_mb / 1024:.2f} GB")
    print(f"Output directory: {output_dir}")
    if args.delete_source:
        print(f"Deleted {deleted_count} source directories")

    # Compare with original size
    original_files = list(input_dir.rglob("latent_*.npy"))
    if original_files:
        original_size_mb = sum(f.stat().st_size for f in original_files) / (1024 * 1024)
        compression = (1 - total_size_mb / original_size_mb) * 100 if original_size_mb > 0 else 0
        print(f"Original size: {original_size_mb / 1024:.2f} GB")
        print(f"Compression: {compression:.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
