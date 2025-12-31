"""Feature extraction pipeline for LoL replay videos.

Pipeline:
1. Download video at 1080p from YouTube
2. Process video at 1080p to extract features (movement, abilities, items, gold)
3. Save features as JSON per video
4. Convert frames to 256x256 for training
5. Delete source 1080p video to save space
6. Log video metadata to manifest

Features are extracted at 1080p for accuracy (HUD elements are small),
then frames are downscaled for the vision tokenizer.
"""

import cv2
import json
import numpy as np
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

from .keylog_extractor import (
    GarenHUDTracker,
    GoldTextDetector,
    ItemUsageDetector,
    HUDRegionsNormalized,
    MousePositionEstimator,
)


# Manifest file tracking all processed videos
MANIFEST_FILENAME = "video_manifest.json"


def load_manifest(output_dir: Path) -> dict:
    """Load or create video manifest."""
    manifest_path = output_dir / MANIFEST_FILENAME
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {"videos": []}


def save_manifest(output_dir: Path, manifest: dict):
    """Save video manifest."""
    manifest_path = output_dir / MANIFEST_FILENAME
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


EXTRACTION_VERSION = "1.0"


def add_to_manifest(
    output_dir: Path,
    video_id: str,
    youtube_url: str,
    channel: str,
    team_side: str,
    stats: dict,
    has_mouse_data: bool = False,
    has_item_data: bool = False,
):
    """Add a processed video to the manifest."""
    manifest = load_manifest(output_dir)

    entry = {
        "video_id": video_id,
        "youtube_url": youtube_url,
        "channel": channel,
        "team_side": team_side,
        "processed_at": datetime.now().isoformat(),
        "extraction_version": EXTRACTION_VERSION,
        "has_mouse_data": has_mouse_data,  # True only for your gameplay with actual mouse
        "has_item_data": has_item_data,    # True if item detection is reliable
        **stats,
    }

    # Check if already exists
    for i, v in enumerate(manifest["videos"]):
        if v["video_id"] == video_id:
            manifest["videos"][i] = entry
            save_manifest(output_dir, manifest)
            return

    # Add new entry
    manifest["videos"].append(entry)
    save_manifest(output_dir, manifest)


@dataclass
class FrameFeatures:
    """Features extracted from a single frame."""
    frame_idx: int
    timestamp_ms: float

    # Movement (from optical flow)
    movement_dx: float
    movement_dy: float
    movement_slice: int  # 0-17, direction of movement
    movement_confidence: float

    # Abilities (True if just used this frame)
    ability_q: bool
    ability_w: bool
    ability_e: bool
    ability_r: bool
    summoner_d: bool
    summoner_f: bool

    # Item usage
    item_used: bool

    # Gold (from OCR)
    gold_gained: int  # Amount of gold gained this frame (0 if none)

    # Health bar position (for reference)
    health_bar_x: Optional[int] = None
    health_bar_y: Optional[int] = None


class FeatureExtractionPipeline:
    """Extract features from 1080p video and save 256x256 frames."""

    def __init__(
        self,
        output_dir: Path,
        target_fps: int = 20,
        target_size: tuple[int, int] = (256, 256),
        team_side: str = "blue",
        use_gpu_ocr: bool = False,
    ):
        """Initialize pipeline.

        Args:
            output_dir: Directory to save frames and features
            target_fps: Target FPS for extraction (default 20)
            target_size: Output frame size (default 256x256)
            team_side: Which side Garen is on ("blue" or "red")
            use_gpu_ocr: Use GPU for OCR (requires CUDA)
        """
        self.output_dir = Path(output_dir)
        self.target_fps = target_fps
        self.target_size = target_size
        self.team_side = team_side
        self.use_gpu_ocr = use_gpu_ocr

    def process_video(
        self,
        video_path: Path,
        video_id: str,
        youtube_url: str = "",
        channel: str = "",
        start_sec: float = 0,
        end_sec: Optional[float] = None,
        delete_source: bool = True,
    ) -> dict:
        """Process a video: extract features at 1080p, save 256x256 frames.

        Args:
            video_path: Path to input video (should be 1080p)
            video_id: Unique identifier for this video
            youtube_url: Source YouTube URL (for manifest)
            channel: Source channel name (for manifest)
            start_sec: Start time in seconds
            end_sec: End time in seconds (None = end of video)
            delete_source: Delete source video after processing to save space

        Returns:
            Dict with processing stats
        """
        # Create output directories
        video_output_dir = self.output_dir / video_id
        frames_dir = video_output_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if frame_width != 1920 or frame_height != 1080:
            print(f"Warning: Video is {frame_width}x{frame_height}, expected 1920x1080")

        print(f"Processing {video_id}: {frame_width}x{frame_height} @ {input_fps} FPS")
        print(f"  Total frames: {total_frames}, Duration: {total_frames/input_fps:.1f}s")

        # Calculate frame indices
        start_frame = int(start_sec * input_fps)
        end_frame = int(end_sec * input_fps) if end_sec else total_frames
        frame_skip = max(1, int(input_fps / self.target_fps))

        print(f"  Processing frames {start_frame} to {end_frame}, every {frame_skip} frames")

        # Initialize detectors at 1080p
        normalized_regions = HUDRegionsNormalized()

        movement_tracker = GarenHUDTracker(
            normalized_regions=normalized_regions,
            frame_width=frame_width,
            frame_height=frame_height,
        )

        gold_detector = GoldTextDetector(
            normalized_regions=normalized_regions,
            frame_width=frame_width,
            frame_height=frame_height,
            use_gpu=self.use_gpu_ocr,
        )

        item_detector = ItemUsageDetector(
            frame_width=frame_width,
            frame_height=frame_height,
            team_side=self.team_side,
        )

        mouse_estimator = MousePositionEstimator(num_slices=18)

        # Seek to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # Process frames
        features_list = []
        frame_count = 0
        output_frame_idx = 0

        while cap.get(cv2.CAP_PROP_POS_FRAMES) < end_frame:
            ret, frame = cap.read()
            if not ret:
                break

            absolute_frame = start_frame + frame_count

            # Only process every Nth frame
            if frame_count % frame_skip == 0:
                timestamp_ms = absolute_frame / input_fps * 1000

                # === Extract features at 1080p ===

                # Movement detection
                dx, dy, move_conf = movement_tracker.detect_movement(frame)
                movement_slice = mouse_estimator.movement_to_slice(dx, dy)

                # Ability detection
                abilities = movement_tracker.detect_ability_usage(frame, absolute_frame)

                # Gold detection
                gold_gains, health_bar = gold_detector.detect_gold_text(frame, absolute_frame)
                total_gold = sum(amount for amount, _ in gold_gains)

                # Item detection
                item_used = item_detector.detect_item_usage(frame, absolute_frame)

                # Create feature record
                features = FrameFeatures(
                    frame_idx=output_frame_idx,
                    timestamp_ms=timestamp_ms,
                    movement_dx=dx,
                    movement_dy=dy,
                    movement_slice=movement_slice,
                    movement_confidence=move_conf,
                    ability_q=abilities.get('Q', False),
                    ability_w=abilities.get('W', False),
                    ability_e=abilities.get('E', False),
                    ability_r=abilities.get('R', False),
                    summoner_d=abilities.get('D', False),
                    summoner_f=abilities.get('F', False),
                    item_used=item_used,
                    gold_gained=total_gold,
                    health_bar_x=health_bar[0] if health_bar else None,
                    health_bar_y=health_bar[1] if health_bar else None,
                )
                features_list.append(asdict(features))

                # === Save 256x256 frame ===
                frame_256 = cv2.resize(frame, self.target_size, interpolation=cv2.INTER_AREA)
                frame_path = frames_dir / f"frame_{output_frame_idx:06d}.jpg"
                cv2.imwrite(str(frame_path), frame_256, [cv2.IMWRITE_JPEG_QUALITY, 95])

                output_frame_idx += 1

                if output_frame_idx % 500 == 0:
                    print(f"  Processed {output_frame_idx} frames...")

            frame_count += 1

        cap.release()

        # Save features JSON
        features_path = video_output_dir / "features.json"
        with open(features_path, 'w') as f:
            json.dump({
                "video_id": video_id,
                "source_resolution": f"{frame_width}x{frame_height}",
                "source_fps": input_fps,
                "output_fps": self.target_fps,
                "output_resolution": f"{self.target_size[0]}x{self.target_size[1]}",
                "team_side": self.team_side,
                "num_frames": output_frame_idx,
                "frames": features_list,
            }, f, indent=2)

        # Compute stats
        stats = {
            "video_id": video_id,
            "num_frames": output_frame_idx,
            "duration_sec": output_frame_idx / self.target_fps,
            "abilities_detected": sum(
                1 for f in features_list
                if f["ability_q"] or f["ability_w"] or f["ability_e"] or f["ability_r"]
            ),
            "items_used": sum(1 for f in features_list if f["item_used"]),
            "gold_events": sum(1 for f in features_list if f["gold_gained"] > 0),
            "total_gold": sum(f["gold_gained"] for f in features_list),
        }

        print(f"\nDone processing {video_id}:")
        print(f"  Frames: {stats['num_frames']} ({stats['duration_sec']:.1f}s)")
        print(f"  Abilities: {stats['abilities_detected']}")
        print(f"  Items: {stats['items_used']}")
        print(f"  Gold events: {stats['gold_events']} (total: {stats['total_gold']})")

        # Determine quality flags
        # has_mouse_data: False for replay footage (no actual mouse cursor)
        # has_item_data: True if we detected any item usage (suggests detection is working)
        is_replay = "replay" in channel.lower() or "domisum" in channel.lower()
        has_mouse_data = not is_replay  # Only true for live gameplay recordings
        has_item_data = stats["items_used"] > 0  # If we detected items, extraction is working

        # Add to manifest
        add_to_manifest(
            output_dir=self.output_dir,
            video_id=video_id,
            youtube_url=youtube_url,
            channel=channel,
            team_side=self.team_side,
            stats=stats,
            has_mouse_data=has_mouse_data,
            has_item_data=has_item_data,
        )

        # Delete source video to save space
        if delete_source and video_path.exists():
            video_path.unlink()
            print(f"  Deleted source video: {video_path}")

        return stats


def process_video_cli():
    """CLI entry point for processing a single video."""
    import argparse

    parser = argparse.ArgumentParser(description="Extract features from LoL replay video")
    parser.add_argument("video_path", type=Path, help="Path to input video")
    parser.add_argument("--output", "-o", type=Path, default=Path("data/processed"),
                       help="Output directory")
    parser.add_argument("--video-id", type=str, default=None,
                       help="Video ID (default: video filename)")
    parser.add_argument("--youtube-url", type=str, default="",
                       help="Source YouTube URL (for manifest)")
    parser.add_argument("--channel", type=str, default="",
                       help="Source channel name (for manifest)")
    parser.add_argument("--start", type=float, default=0,
                       help="Start time in seconds")
    parser.add_argument("--end", type=float, default=None,
                       help="End time in seconds")
    parser.add_argument("--fps", type=int, default=20,
                       help="Target FPS")
    parser.add_argument("--side", choices=["blue", "red"], default="blue",
                       help="Team side")
    parser.add_argument("--keep-source", action="store_true",
                       help="Keep source video after processing (default: delete)")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU for OCR")

    args = parser.parse_args()

    video_id = args.video_id or args.video_path.stem

    pipeline = FeatureExtractionPipeline(
        output_dir=args.output,
        target_fps=args.fps,
        team_side=args.side,
        use_gpu_ocr=args.gpu,
    )

    pipeline.process_video(
        video_path=args.video_path,
        video_id=video_id,
        youtube_url=args.youtube_url,
        channel=args.channel,
        start_sec=args.start,
        end_sec=args.end,
        delete_source=not args.keep_source,
    )


if __name__ == "__main__":
    process_video_cli()
