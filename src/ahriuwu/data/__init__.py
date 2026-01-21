"""Data collection and processing modules."""

from .youtube_downloader import YouTubeDownloader, VideoMetadata
from .frame_extractor import extract_frames, is_extracted, get_video_duration
from .hud_regions import get_hud_regions, crop_region, REPLAY_HUD_1080P, LIVE_HUD_1080P
from .dataset import (
    SingleFrameDataset,
    FrameSequenceDataset,
    FrameWithStateDataset,
    LatentSequenceDataset,
    PackedLatentSequenceDataset,
)

__all__ = [
    "YouTubeDownloader",
    "VideoMetadata",
    "extract_frames",
    "is_extracted",
    "get_video_duration",
    "get_hud_regions",
    "crop_region",
    "REPLAY_HUD_1080P",
    "LIVE_HUD_1080P",
    "SingleFrameDataset",
    "FrameSequenceDataset",
    "FrameWithStateDataset",
    "LatentSequenceDataset",
    "PackedLatentSequenceDataset",
]
