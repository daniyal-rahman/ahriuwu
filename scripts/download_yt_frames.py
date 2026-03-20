#!/usr/bin/env python3
"""Download domisumReplay-Garen videos and extract masked+resized frames.

Pipeline:
1. Get video IDs from channel (most recent N)
2. Download each at 360p
3. Extract frames at target FPS, apply HUD mask, resize to 352x352
4. Save frames as JPG, delete raw video to save space

Usage:
    python scripts/download_yt_frames.py --num-videos 2000 --output-dir /mnt/storage/ahriuwu/frames
    python scripts/download_yt_frames.py --num-videos 10 --output-dir /mnt/storage/ahriuwu/frames --keep-videos
"""

import argparse
import json
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np


# HUD mask regions at 640x360 (domisumReplay format)
# Each is (y1, y2, x1, x2) for direct numpy slicing
MASK_REGIONS_360P = [
    (0, 35, 0, 640),       # top scoreboard (full width)
    (35, 215, 0, 35),      # left champion cards (blue team)
    (35, 215, 605, 640),   # right champion cards (red team)
    (257, 360, 0, 105),    # bottom-left: garen HUD + watermark
    (282, 360, 105, 550),  # bottom-center: scorecard + items
]

TARGET_SIZE = (352, 352)
CHANNEL_URL = "https://www.youtube.com/@domisumReplay-Garen/videos"

# Global shutdown event — set by SIGTERM handler
_shutdown = threading.Event()


def get_video_ids(num_videos: int) -> list[str]:
    """Get most recent video IDs from channel."""
    print(f"Fetching {num_videos} most recent video IDs...")
    result = subprocess.run(
        [
            "yt-dlp", "--flat-playlist", "--print", "id",
            "--playlist-end", str(num_videos),
            CHANNEL_URL,
        ],
        capture_output=True, text=True,
    )
    ids = [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]
    print(f"Got {len(ids)} video IDs")
    return ids


def download_video(video_id: str, output_dir: Path) -> Path | None:
    """Download a single video at 360p. Returns path or None on failure."""
    output_path = output_dir / f"{video_id}.mp4"
    if output_path.exists():
        return output_path

    # yt-dlp writes to a .part file then renames on completion,
    # so output_path only exists if the download fully succeeded.
    result = subprocess.run(
        [
            "yt-dlp",
            "-f", "worst[ext=mp4]/worst",
            "-o", str(output_path),
            "--no-warnings",
            "-q",
            f"https://www.youtube.com/watch?v={video_id}",
        ],
        capture_output=True, text=True, timeout=300,
    )
    if _shutdown.is_set():
        # Clean up partial download
        for p in output_dir.glob(f"{video_id}.mp4*"):
            p.unlink(missing_ok=True)
        return None
    if result.returncode != 0 or not output_path.exists():
        # Try alternative format
        result = subprocess.run(
            [
                "yt-dlp",
                "-f", "18",  # 360p mp4 format code
                "-o", str(output_path),
                "--no-warnings",
                "-q",
                f"https://www.youtube.com/watch?v={video_id}",
            ],
            capture_output=True, text=True, timeout=300,
        )
        if _shutdown.is_set():
            for p in output_dir.glob(f"{video_id}.mp4*"):
                p.unlink(missing_ok=True)
            return None
    return output_path if output_path.exists() else None


def apply_mask(frame: np.ndarray) -> np.ndarray:
    """Apply HUD mask to a 640x360 frame."""
    for y1, y2, x1, x2 in MASK_REGIONS_360P:
        frame[y1:y2, x1:x2] = 0
    return frame


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: float = 4.0,
    skip_start_s: float = 30.0,
    skip_end_s: float = 30.0,
) -> int:
    """Extract frames from video, apply mask, resize, save as JPG.

    Args:
        video_path: Path to input video
        output_dir: Directory to save frames
        fps: Target extraction FPS
        skip_start_s: Skip first N seconds (loading screen)
        skip_end_s: Skip last N seconds (end screen)

    Returns:
        Number of frames extracted
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s = total_frames / video_fps if video_fps > 0 else 0

    if duration_s < skip_start_s + skip_end_s + 60:
        # Video too short (< 2 min of gameplay)
        cap.release()
        return 0

    # Calculate frame interval
    frame_interval = int(video_fps / fps) if video_fps > 0 else 8
    start_frame = int(skip_start_s * video_fps)
    end_frame = int((duration_s - skip_end_s) * video_fps)

    output_dir.mkdir(parents=True, exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_idx = start_frame
    saved = 0

    while frame_idx < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            h, w = frame.shape[:2]
            # Resize to 640x360 if not already (shouldn't need this for 360p)
            if (w, h) != (640, 360):
                frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)

            # Apply HUD mask
            frame = apply_mask(frame)

            # Resize to 352x352
            frame = cv2.resize(frame, TARGET_SIZE, interpolation=cv2.INTER_AREA)

            # Save
            out_path = output_dir / f"frame_{saved:06d}.jpg"
            cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved += 1

        frame_idx += 1

    cap.release()
    return saved


def load_progress(progress_path: Path) -> dict:
    """Load processing progress."""
    if progress_path.exists():
        with open(progress_path) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "total_frames": 0}


def save_progress(progress_path: Path, progress: dict):
    """Save processing progress."""
    with open(progress_path, "w") as f:
        json.dump(progress, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Download and extract masked frames")
    parser.add_argument("--num-videos", type=int, default=2000)
    parser.add_argument("--output-dir", type=str, default="/mnt/storage/ahriuwu/frames")
    parser.add_argument("--fps", type=float, default=4.0, help="Frame extraction rate")
    parser.add_argument("--keep-videos", action="store_true", help="Don't delete raw videos")
    parser.add_argument("--skip-start", type=float, default=30.0, help="Skip first N seconds")
    parser.add_argument("--skip-end", type=float, default=30.0, help="Skip last N seconds")
    parser.add_argument("--workers", type=int, default=1, help="Parallel downloads (be gentle)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "raw_videos"
    frames_dir = output_dir / "frames"
    videos_dir.mkdir(parents=True, exist_ok=True)
    frames_dir.mkdir(parents=True, exist_ok=True)

    progress_path = output_dir / "progress.json"
    progress = load_progress(progress_path)

    # Get video IDs
    ids_cache = output_dir / "video_ids.json"
    if ids_cache.exists():
        with open(ids_cache) as f:
            video_ids = json.load(f)
        print(f"Loaded {len(video_ids)} cached video IDs")
    else:
        video_ids = get_video_ids(args.num_videos)
        with open(ids_cache, "w") as f:
            json.dump(video_ids, f)

    already = set(progress["completed"] + progress["failed"])
    remaining = [vid for vid in video_ids if vid not in already]
    print(f"Progress: {len(progress['completed'])} done, {len(progress['failed'])} failed, {len(remaining)} remaining")
    print(f"Total frames so far: {progress['total_frames']:,}")
    print(f"Workers: {args.workers}")

    progress_lock = threading.Lock()

    # Semaphore to stagger downloads (only 1 downloading at a time, extraction parallel)
    dl_semaphore = threading.Semaphore(1)
    dl_delay = 3  # seconds between download starts

    def process_one(video_id: str) -> tuple[str, int]:
        """Download and extract frames for one video. Returns (video_id, n_frames)."""
        if _shutdown.is_set():
            return (video_id, -2)  # -2 = skipped due to shutdown

        # Stagger downloads to avoid YouTube throttling
        with dl_semaphore:
            if _shutdown.is_set():
                return (video_id, -2)
            try:
                video_path = download_video(video_id, videos_dir)
            except Exception as e:
                print(f"  [{video_id}] Download failed: {e}", flush=True)
                return (video_id, -1)
            time.sleep(dl_delay)

        if _shutdown.is_set() or video_path is None:
            if video_path is None:
                print(f"  [{video_id}] Download failed", flush=True)
            return (video_id, -1 if video_path is None else -2)

        # Extract frames (can run in parallel with other extractions)
        vid_frames_dir = frames_dir / video_id
        n_frames = extract_frames(
            video_path, vid_frames_dir,
            fps=args.fps,
            skip_start_s=args.skip_start,
            skip_end_s=args.skip_end,
        )

        if _shutdown.is_set() and n_frames == 0:
            # Extraction was likely interrupted — clean up partial frames
            if vid_frames_dir.exists():
                shutil.rmtree(vid_frames_dir)
            return (video_id, -2)

        # Delete raw video to save space
        if not args.keep_videos and video_path.exists():
            video_path.unlink()

        return (video_id, n_frames)

    def handle_sigterm(signum, frame):
        print("\nSIGTERM received — finishing current video and saving progress...", flush=True)
        _shutdown.set()

    signal.signal(signal.SIGTERM, handle_sigterm)
    signal.signal(signal.SIGINT, handle_sigterm)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_one, vid): vid for vid in remaining}

        for future in as_completed(futures):
            video_id, n_frames = future.result()

            with progress_lock:
                if n_frames == -2:
                    pass  # shutdown skip — don't record as failed
                elif n_frames < 0:
                    progress["failed"].append(video_id)
                elif n_frames == 0:
                    print(f"  [{video_id}] No frames (video too short?)", flush=True)
                    progress["failed"].append(video_id)
                else:
                    progress["completed"].append(video_id)
                    progress["total_frames"] += n_frames

                done = len(progress["completed"])
                total = len(video_ids)
                if n_frames >= 0:
                    print(
                        f"[{done}/{total}] {video_id}: {n_frames} frames "
                        f"(total: {progress['total_frames']:,})",
                        flush=True,
                    )
                save_progress(progress_path, progress)

            if _shutdown.is_set():
                pool.shutdown(wait=False, cancel_futures=True)
                break

    print(f"\n{'Interrupted' if _shutdown.is_set() else 'Done'}! "
          f"{len(progress['completed'])} videos, {progress['total_frames']:,} total frames")
    print(f"Frames at: {frames_dir}")
    if _shutdown.is_set():
        sys.exit(0)


if __name__ == "__main__":
    main()
