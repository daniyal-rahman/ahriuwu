#!/usr/bin/env python3
"""Download LoL gameplay videos from YouTube.

Usage:
    # Download from a specific channel (max 10 videos)
    python scripts/download_youtube.py --channel GarenChallenger --max 10

    # Download from all known channels
    python scripts/download_youtube.py --all --max 50

    # List already downloaded videos
    python scripts/download_youtube.py --list
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ahriuwu.data.youtube_downloader import KNOWN_CHANNELS, YouTubeDownloader


def main():
    parser = argparse.ArgumentParser(description="Download LoL gameplay videos from YouTube")
    parser.add_argument(
        "--output", "-o",
        default="data/raw",
        help="Output directory (default: data/raw)",
    )
    parser.add_argument(
        "--channel", "-c",
        choices=list(KNOWN_CHANNELS.keys()),
        help="Channel to download from",
    )
    parser.add_argument(
        "--all", "-a",
        action="store_true",
        help="Download from all known channels",
    )
    parser.add_argument(
        "--max", "-m",
        type=int,
        default=None,
        help="Maximum number of videos to download (per channel)",
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List already downloaded videos",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-download even if video exists",
    )

    args = parser.parse_args()

    downloader = YouTubeDownloader(args.output)

    if args.list:
        videos = downloader.list_downloaded()
        print(f"Downloaded videos: {len(videos)}")
        for v in videos:
            elo = v.elo_tier or "?"
            side = v.side if v.side != "unknown" else "?"
            print(f"  [{v.channel}] {v.video_id}: {v.title[:50]}... ({elo}, {side} side)")
        return

    if not args.channel and not args.all:
        parser.print_help()
        print("\nError: Must specify --channel or --all")
        sys.exit(1)

    skip_existing = not args.no_skip

    if args.all:
        results = downloader.download_all_channels(args.max, skip_existing)
        for channel, videos in results.items():
            print(f"\n{channel}: {len(videos)} videos downloaded")
    else:
        videos = downloader.download_channel(args.channel, args.max, skip_existing)
        print(f"\n{args.channel}: {len(videos)} videos downloaded")


if __name__ == "__main__":
    main()
