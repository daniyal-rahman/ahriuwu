"""Extract frames from video files at 20 FPS using ffmpeg."""

import subprocess
from pathlib import Path


def get_video_duration(video_path: Path) -> float:
    """Get video duration in seconds."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip())


def extract_frames(
    video_path: Path,
    output_dir: Path,
    fps: int = 20,
    format: str = "jpg",
    quality: int = 2,
    use_gpu: bool = True,
    resolution: int = 256,
) -> int:
    """Extract frames from video at target FPS.

    Args:
        video_path: Path to video file
        output_dir: Directory to save frames
        fps: Target frames per second (default 20)
        format: Output format (jpg or png). jpg is ~10x smaller.
        quality: JPEG quality 2-31 (2=best, only used for jpg)
        use_gpu: Use NVIDIA GPU acceleration (NVDEC) if available
        resolution: Output resolution (default 256 for tokenizer input)

    Returns:
        Number of frames extracted
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / f"frame_%06d.{format}"

    cmd = ["ffmpeg"]

    # Add NVIDIA hardware acceleration for decoding
    if use_gpu:
        cmd.extend(["-hwaccel", "cuda"])

    cmd.extend([
        "-i", str(video_path),
        "-vf", f"fps={fps},scale={resolution}:{resolution}",
        "-start_number", "0",
        "-y",
    ])

    # Add JPEG quality setting (2=best, 31=worst)
    if format == "jpg":
        cmd.extend(["-qscale:v", str(quality)])

    cmd.append(str(output_pattern))

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr}")

    frame_count = len(list(output_dir.glob(f"frame_*.{format}")))
    return frame_count


def is_extracted(output_dir: Path) -> bool:
    """Check if frames already exist."""
    return output_dir.exists() and len(list(output_dir.glob("frame_*"))) > 0
