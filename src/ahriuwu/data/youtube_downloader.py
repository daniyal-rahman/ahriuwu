"""YouTube video downloader for LoL gameplay footage."""

import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class VideoMetadata:
    """Metadata for a downloaded video."""

    video_id: str
    channel: str
    title: str
    upload_date: str
    duration_seconds: int
    url: str

    # Detected/labeled
    hud_type: str  # "replay" | "spectator" | "live" | "unknown"
    side: str  # "blue" | "red" | "unknown"
    patch_version: str | None
    elo_tier: str | None

    # Processing state
    downloaded: bool = False
    frames_extracted: bool = False
    ocr_complete: bool = False

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "VideoMetadata":
        return cls(**json.loads(json_str))


# Known channels and their characteristics
KNOWN_CHANNELS = {
    "GarenChallenger": {
        "url": "https://www.youtube.com/@GarenChallenger",
        "hud_type": "live",  # Appears to be live gameplay
    },
    "domisumReplay-Garen": {
        "url": "https://www.youtube.com/@domisumReplay-Garen",
        "hud_type": "replay",  # Replay recordings
    },
}

# Regex patterns for parsing video titles
ELO_PATTERNS = [
    (r"\bchallenger\b", "challenger"),
    (r"\bgrandmaster\b", "grandmaster"),
    (r"\bmaster\b", "master"),
    (r"\bdiamond\b", "diamond"),
    (r"\bemerald\b", "emerald"),
    (r"\bplatinum\b", "platinum"),
    (r"\bplat\b", "platinum"),
    (r"\bgold\b", "gold"),
    (r"\bsilver\b", "silver"),
    (r"\bbronze\b", "bronze"),
    (r"\biron\b", "iron"),
]

PATCH_PATTERN = re.compile(r"\b(\d{1,2}\.\d{1,2})\b")
SIDE_PATTERNS = [
    (r"\bblue\s*side\b", "blue"),
    (r"\bred\s*side\b", "red"),
]

# Pattern for domisumReplay descriptions: "EUW-7646979958-TOP-RED" or "TOP-BLUE"
SIDE_FROM_DESC_PATTERN = re.compile(r"-(RED|BLUE)\s*$", re.IGNORECASE)


def parse_elo_from_title(title: str) -> str | None:
    """Extract elo tier from video title."""
    title_lower = title.lower()
    for pattern, elo in ELO_PATTERNS:
        if re.search(pattern, title_lower):
            return elo
    return None


def parse_patch_from_title(title: str) -> str | None:
    """Extract patch version from video title (e.g., '14.23')."""
    match = PATCH_PATTERN.search(title)
    if match:
        version = match.group(1)
        major, minor = version.split(".")
        # Sanity check: LoL patches are typically 1-14.x
        if 1 <= int(major) <= 25 and 1 <= int(minor) <= 30:
            return version
    return None


def parse_side_from_title(title: str, description: str = "") -> str:
    """Extract side (blue/red) from video title or description.

    Checks:
    1. Title for "blue side" / "red side"
    2. Description for domisumReplay format: "EUW-xxx-TOP-RED" or "-BLUE"
    """
    title_lower = title.lower()
    for pattern, side in SIDE_PATTERNS:
        if re.search(pattern, title_lower):
            return side

    # Check description for domisumReplay format (e.g., "TOP-RED" at end)
    if description:
        match = SIDE_FROM_DESC_PATTERN.search(description)
        if match:
            return match.group(1).lower()

    return "unknown"


class YouTubeDownloader:
    """Download LoL gameplay videos from YouTube channels."""

    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.videos_dir = self.output_dir / "videos"
        self.videos_dir.mkdir(exist_ok=True)
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)

    def get_channel_videos(self, channel_url: str, max_videos: int | None = None) -> list[dict]:
        """Get list of all videos from a channel using yt-dlp."""
        cmd = [
            "yt-dlp",
            "--flat-playlist",
            "--dump-json",
            channel_url + "/videos",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr}")

        videos = []
        for line in result.stdout.strip().split("\n"):
            if line:
                videos.append(json.loads(line))

        if max_videos:
            videos = videos[:max_videos]

        return videos

    def get_video_info(self, video_id: str) -> dict:
        """Get detailed info for a single video."""
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            f"https://www.youtube.com/watch?v={video_id}",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"yt-dlp failed: {result.stderr}")

        return json.loads(result.stdout)

    def create_metadata(self, video_info: dict, channel_name: str) -> VideoMetadata:
        """Create VideoMetadata from yt-dlp video info."""
        video_id = video_info.get("id", video_info.get("url", "").split("=")[-1])
        title = video_info.get("title", "")
        description = video_info.get("description", "")

        # Get HUD type from known channels
        hud_type = KNOWN_CHANNELS.get(channel_name, {}).get("hud_type", "unknown")

        return VideoMetadata(
            video_id=video_id,
            channel=channel_name,
            title=title,
            upload_date=video_info.get("upload_date", ""),
            duration_seconds=video_info.get("duration", 0),
            url=f"https://www.youtube.com/watch?v={video_id}",
            hud_type=hud_type,
            side=parse_side_from_title(title, description),
            patch_version=parse_patch_from_title(title),
            elo_tier=parse_elo_from_title(title),
        )

    def is_downloaded(self, video_id: str) -> bool:
        """Check if video is already downloaded."""
        video_path = self.videos_dir / f"{video_id}.mp4"
        metadata_path = self.metadata_dir / f"{video_id}.json"
        return video_path.exists() and metadata_path.exists()

    def download_video(
        self,
        video_id: str,
        channel_name: str,
        skip_existing: bool = True,
    ) -> VideoMetadata | None:
        """Download a single video and save metadata."""
        if skip_existing and self.is_downloaded(video_id):
            print(f"Skipping {video_id} (already downloaded)")
            # Load existing metadata
            metadata_path = self.metadata_dir / f"{video_id}.json"
            return VideoMetadata.from_json(metadata_path.read_text())

        print(f"Downloading {video_id}...")

        # Get full video info
        video_info = self.get_video_info(video_id)
        metadata = self.create_metadata(video_info, channel_name)

        # Download video at 1080p
        output_path = self.videos_dir / f"{video_id}.mp4"
        cmd = [
            "yt-dlp",
            "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080][ext=mp4]/best",
            "--merge-output-format", "mp4",
            "-o", str(output_path),
            metadata.url,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Failed to download {video_id}: {result.stderr}")
            return None

        # Save metadata
        metadata.downloaded = True
        metadata_path = self.metadata_dir / f"{video_id}.json"
        metadata_path.write_text(metadata.to_json())

        print(f"Downloaded {video_id}: {metadata.title}")
        return metadata

    def download_channel(
        self,
        channel_name: str,
        max_videos: int | None = None,
        skip_existing: bool = True,
    ) -> list[VideoMetadata]:
        """Download all videos from a known channel."""
        if channel_name not in KNOWN_CHANNELS:
            raise ValueError(f"Unknown channel: {channel_name}. Known: {list(KNOWN_CHANNELS.keys())}")

        channel_url = KNOWN_CHANNELS[channel_name]["url"]
        print(f"Fetching video list from {channel_name}...")

        videos = self.get_channel_videos(channel_url, max_videos)
        print(f"Found {len(videos)} videos")

        downloaded = []
        for video in videos:
            video_id = video.get("id", video.get("url", "").split("=")[-1])
            metadata = self.download_video(video_id, channel_name, skip_existing)
            if metadata:
                downloaded.append(metadata)

        return downloaded

    def download_all_channels(
        self,
        max_videos_per_channel: int | None = None,
        skip_existing: bool = True,
    ) -> dict[str, list[VideoMetadata]]:
        """Download from all known channels."""
        results = {}
        for channel_name in KNOWN_CHANNELS:
            results[channel_name] = self.download_channel(
                channel_name, max_videos_per_channel, skip_existing
            )
        return results

    def list_downloaded(self) -> list[VideoMetadata]:
        """List all downloaded videos with their metadata."""
        metadata_files = self.metadata_dir.glob("*.json")
        return [
            VideoMetadata.from_json(f.read_text())
            for f in metadata_files
        ]
