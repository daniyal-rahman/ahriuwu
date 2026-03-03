#!/usr/bin/env python3
"""Monitor frame extraction progress with a live progress bar."""

import time
from pathlib import Path

VIDEOS_DIR = Path("data/raw/videos")
FRAMES_DIR = Path("data/processed/frames")

def count_frames(frames_dir: Path) -> int:
    """Count total extracted frames."""
    return sum(1 for _ in frames_dir.rglob("*.jpg"))

def main():
    total_videos = len(list(VIDEOS_DIR.glob("*.mp4")))

    print("\n🎬 Frame Extraction Progress")
    print("=" * 50)

    try:
        while True:
            # Count completed directories (with at least 1 frame)
            completed = sum(
                1 for d in FRAMES_DIR.iterdir()
                if d.is_dir() and any(d.glob("*.jpg"))
            )

            # Count total frames
            total_frames = count_frames(FRAMES_DIR)

            # Progress bar
            pct = completed / total_videos if total_videos > 0 else 0
            bar_len = 30
            filled = int(bar_len * pct)
            bar = "█" * filled + "░" * (bar_len - filled)

            # Clear line and print
            print(f"\r[{bar}] {completed}/{total_videos} videos ({pct*100:.1f}%) | {total_frames:,} frames", end="", flush=True)

            if completed >= total_videos:
                print("\n✅ Done!")
                break

            time.sleep(5)

    except KeyboardInterrupt:
        print("\n\nStopped.")

if __name__ == "__main__":
    main()
