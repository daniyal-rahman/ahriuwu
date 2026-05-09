"""Data loading and preprocessing modules.

`ReplayLatentSequenceDataset` is intentionally NOT re-exported from this
package's __init__: it imports from ``ahriuwu.rewards``, which itself
imports from ``ahriuwu.data.lane_opponent``. Eager re-export here would
create a circular import. Consumers should:

    from ahriuwu.data.replay_dataset import ReplayLatentSequenceDataset
"""

from .youtube_downloader import YouTubeDownloader, VideoMetadata
from .frame_extractor import extract_frames, is_extracted, get_video_duration
from .hud_regions import get_hud_regions, crop_region, REPLAY_HUD_1080P
from .dataset import (
    SingleFrameDataset,
    FrameSequenceDataset,
    VideoGroupedSampler,
    VideoShuffleSampler,
)
from .lane_opponent import (
    identify_teams,
    find_lane_opponent,
    resolve_lane_opponent,
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
    "SingleFrameDataset",
    "FrameSequenceDataset",
    "VideoGroupedSampler",
    "VideoShuffleSampler",
    "identify_teams",
    "find_lane_opponent",
    "resolve_lane_opponent",
]
