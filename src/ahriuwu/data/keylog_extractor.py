"""Extract pseudo-keylog data from LoL replay videos.

This module extracts action labels from replay footage where:
- WASD movement is inferred from camera/terrain shift or Garen HUD tracking
- Ability usage (QWER + FG) is detected from cooldown state changes in ability bar
- Mouse position is estimated as 18 slices around Garen's ability range circle

Key insight: In replay footage with locked camera on Garen, the terrain/pixels
shift when Garen moves, and we can track his overhead HUD to infer WASD inputs.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class HUDRegionsNormalized:
    """HUD region coordinates as normalized values (0-1) for resolution independence.

    Format: (x, y, width, height) where (x,y) is top-left corner.
    All values are fractions of screen width/height.
    Calibrated from 1920x1080 domisumReplay footage.
    """
    # Reference resolution used for calibration
    ref_width: int = 1920
    ref_height: int = 1080

    # Individual ability icons (QWER) - normalized coordinates
    # Calibrated at 1920x1080: Q=(56,925,26,24), W=(96,925,26,24), etc.
    ability_q: tuple[float, float, float, float] = (0.02917, 0.85648, 0.01354, 0.02222)
    ability_w: tuple[float, float, float, float] = (0.05000, 0.85648, 0.01354, 0.02222)
    ability_e: tuple[float, float, float, float] = (0.06979, 0.85648, 0.01354, 0.02222)
    ability_r: tuple[float, float, float, float] = (0.09115, 0.85648, 0.01354, 0.02222)

    # Summoner spells (D, F)
    summoner_d: tuple[float, float, float, float] = (0.11667, 0.85648, 0.01354, 0.02222)
    summoner_f: tuple[float, float, float, float] = (0.13333, 0.85648, 0.01354, 0.02222)

    # Screen center region for optical flow (exclude HUD areas)
    game_area: tuple[float, float, float, float] = (0.27604, 0.18519, 0.44792, 0.46296)

    # Replay timing offsets in milliseconds (positive = detection is late, shift earlier)
    # Calibrated at 60fps: R=24 frames late (400ms), Q=1 frame late (16.67ms)
    offset_q_ms: float = 16.67
    offset_w_ms: float = 0.0
    offset_e_ms: float = 0.0
    offset_r_ms: float = 400.0
    offset_d_ms: float = 0.0
    offset_f_ms: float = 0.0

    def to_pixels(self, width: int, height: int) -> "HUDRegions":
        """Convert normalized coordinates to pixel coordinates for given resolution."""
        def convert(norm: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
            x, y, w, h = norm
            return (int(x * width), int(y * height), int(w * width), int(h * height))

        return HUDRegions(
            ability_q=convert(self.ability_q),
            ability_w=convert(self.ability_w),
            ability_e=convert(self.ability_e),
            ability_r=convert(self.ability_r),
            summoner_d=convert(self.summoner_d),
            summoner_f=convert(self.summoner_f),
            game_area=convert(self.game_area),
        )

    def get_offset_frames(self, ability: str, fps: float) -> int:
        """Get the frame offset for an ability at given FPS."""
        offsets_ms = {
            'Q': self.offset_q_ms,
            'W': self.offset_w_ms,
            'E': self.offset_e_ms,
            'R': self.offset_r_ms,
            'D': self.offset_d_ms,
            'F': self.offset_f_ms,
        }
        ms = offsets_ms.get(ability, 0.0)
        return int(round(ms * fps / 1000.0))


@dataclass
class HUDRegions:
    """HUD region coordinates in pixels for a specific resolution."""
    ability_q: tuple[int, int, int, int] = (56, 925, 26, 24)
    ability_w: tuple[int, int, int, int] = (96, 925, 26, 24)
    ability_e: tuple[int, int, int, int] = (134, 925, 26, 24)
    ability_r: tuple[int, int, int, int] = (175, 925, 26, 24)
    summoner_d: tuple[int, int, int, int] = (224, 925, 26, 24)
    summoner_f: tuple[int, int, int, int] = (256, 925, 26, 24)
    game_area: tuple[int, int, int, int] = (530, 200, 860, 500)


# Backwards compatibility alias
HUDRegions1080p = HUDRegions


@dataclass
class ActionLabels:
    """Extracted action labels for a single frame."""
    frame_idx: int
    timestamp_ms: float

    # WASD movement (inferred from camera movement)
    movement_dx: float  # Horizontal camera shift (positive = right = D key)
    movement_dy: float  # Vertical camera shift (positive = down = S key)
    wasd_keys: list[str]  # Inferred keys: ['W'], ['A', 'W'], etc.

    # Ability states (True if ability was just used this frame)
    ability_q_used: bool = False
    ability_w_used: bool = False
    ability_e_used: bool = False
    ability_r_used: bool = False

    # Summoner spells
    summoner_d_used: bool = False
    summoner_f_used: bool = False

    # Mouse position estimate (18 slices: 0-17, where 0 = right, going CCW)
    mouse_slice: int = 0  # Which of 18 sectors around Garen
    mouse_distance: float = 0.5  # 0-1, where 1 = max ability range

    # Confidence scores
    movement_confidence: float = 0.0
    ability_confidence: float = 0.0


class GarenHUDTracker:
    """Track camera movement to detect WASD inputs.

    The locked camera follows Garen, so when Garen moves, the entire screen
    shifts. We detect this using dense optical flow (Farneback) on the game area.

    Key insight: When Garen moves RIGHT, terrain shifts LEFT on screen (negative dx).
    So we INVERT the flow direction to get Garen's movement direction.
    """

    def __init__(
        self,
        regions: HUDRegions = None,
        normalized_regions: HUDRegionsNormalized = None,
        frame_width: int = 1920,
        frame_height: int = 1080,
    ):
        """Initialize tracker.

        Args:
            regions: Pixel-based regions (for backwards compatibility)
            normalized_regions: Resolution-independent regions (preferred)
            frame_width: Video frame width (used with normalized_regions)
            frame_height: Video frame height (used with normalized_regions)
        """
        if normalized_regions is not None:
            self.regions = normalized_regions.to_pixels(frame_width, frame_height)
        elif regions is not None:
            self.regions = regions
        else:
            self.regions = HUDRegions()

        self.prev_gray = None

        # Movement threshold - lowered significantly for sensitivity
        # At 60fps, even fast movement is only ~2-4 pixels per frame
        self.movement_threshold = 0.5  # Minimum pixels to register as movement

    def detect_movement(self, frame: np.ndarray) -> tuple[float, float, float]:
        """Detect camera movement using dense optical flow (Farneback).

        Returns:
            (dx, dy, confidence) where dx/dy are Garen's movement direction
            Positive dx = moving right (D), positive dy = moving down (S)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0, 0.0, 0.0

        # Compute dense optical flow on game area (exclude HUD)
        x, y, w, h = self.regions.game_area
        prev_game = self.prev_gray[y:y+h, x:x+w]
        curr_game = gray[y:y+h, x:x+w]

        # Farneback dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_game, curr_game, None,
            pyr_scale=0.5, levels=3, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0
        )

        # Get flow magnitude
        mag = np.sqrt(flow[:,:,0]**2 + flow[:,:,1]**2)

        # Focus on pixels with significant motion (filter noise)
        motion_threshold = 0.3
        moving_mask = mag > motion_threshold

        if moving_mask.sum() < 100:
            # Very little motion detected
            self.prev_gray = gray
            return 0.0, 0.0, 0.0

        # Get median of moving pixels (robust to outliers from minions/champions)
        dx_terrain = np.median(flow[:,:,0][moving_mask])
        dy_terrain = np.median(flow[:,:,1][moving_mask])

        # INVERT: terrain moves opposite to Garen's movement
        # If terrain shifts left (negative dx), Garen moved right
        dx_garen = -dx_terrain
        dy_garen = -dy_terrain

        # Confidence based on how consistent the motion is
        std_x = np.std(flow[:,:,0][moving_mask])
        std_y = np.std(flow[:,:,1][moving_mask])
        confidence = 1.0 / (1.0 + std_x + std_y)

        self.prev_gray = gray
        return dx_garen, dy_garen, confidence

    def infer_wasd(self, dx: float, dy: float) -> list[str]:
        """Convert Garen's movement direction to WASD keys.

        dx > 0 = moving right = D key
        dy > 0 = moving down = S key
        """
        keys = []

        # Horizontal movement
        if dx > self.movement_threshold:
            keys.append('D')  # Moving right
        elif dx < -self.movement_threshold:
            keys.append('A')  # Moving left

        # Vertical movement
        if dy > self.movement_threshold:
            keys.append('S')  # Moving down
        elif dy < -self.movement_threshold:
            keys.append('W')  # Moving up

        return keys


class AbilityBarDetector:
    """Detect ability usage from the bottom-left HUD.

    We only care about detecting when an ability is USED:
    - Color (high saturation) → Low saturation = ability was activated

    We do NOT care about:
    - Low saturation → Color (coming off cooldown or skilling up)
    - Whether ability is skilled or not

    Supports time offset correction for replay delay.
    """

    def __init__(
        self,
        regions: HUDRegions = None,
        normalized_regions: HUDRegionsNormalized = None,
        frame_width: int = 1920,
        frame_height: int = 1080,
        fps: float = 60.0,
    ):
        """Initialize detector.

        Args:
            regions: Pixel-based regions (for backwards compatibility)
            normalized_regions: Resolution-independent regions (preferred)
            frame_width: Video frame width (used with normalized_regions)
            frame_height: Video frame height (used with normalized_regions)
            fps: Video frame rate (used for time offset calculation)
        """
        self.normalized_regions = normalized_regions or HUDRegionsNormalized()
        self.fps = fps

        if normalized_regions is not None:
            self.regions = normalized_regions.to_pixels(frame_width, frame_height)
        elif regions is not None:
            self.regions = regions
        else:
            self.regions = self.normalized_regions.to_pixels(frame_width, frame_height)

        # Previous frame saturation values for each ability
        self.prev_saturation = {
            'Q': None, 'W': None, 'E': None, 'R': None,
            'D': None, 'F': None
        }

        # Thresholds for detecting ability activation
        self.high_sat_threshold = 80   # Above this = ability available (colored)
        self.low_sat_threshold = 60    # Below this = on cooldown (desaturated)
        self.sat_drop_threshold = 30   # Minimum drop in saturation to count as activation

        # Detection buffer for time offset correction
        # Stores (frame_idx, ability) tuples for pending detections
        self._detection_buffer: list[tuple[int, str]] = []
        self._current_frame_idx = 0

    def get_ability_saturation(self, frame: np.ndarray, region: tuple) -> float:
        """Get mean saturation of ability icon region."""
        x, y, w, h = region
        ability_img = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(ability_img, cv2.COLOR_BGR2HSV)
        return float(hsv[:, :, 1].mean())

    def get_offset_frames(self, ability: str) -> int:
        """Get frame offset for an ability based on fps and configured ms offset."""
        return self.normalized_regions.get_offset_frames(ability, self.fps)

    def detect_ability_usage(self, frame: np.ndarray, frame_idx: int = None) -> dict[str, bool]:
        """Detect which abilities were just used this frame.

        Only detects: high saturation → low saturation transitions
        Returns dict of ability -> True if just activated

        Note: Detections are stored with time offset correction. Use
        get_corrected_detections() to retrieve detections for a specific frame.
        """
        if frame_idx is not None:
            self._current_frame_idx = frame_idx
        else:
            self._current_frame_idx += 1

        ability_regions = {
            'Q': self.regions.ability_q,
            'W': self.regions.ability_w,
            'E': self.regions.ability_e,
            'R': self.regions.ability_r,
            'D': self.regions.summoner_d,
            'F': self.regions.summoner_f,
        }

        usage = {k: False for k in ability_regions}

        for ability, region in ability_regions.items():
            curr_sat = self.get_ability_saturation(frame, region)
            prev_sat = self.prev_saturation[ability]

            # Detect color → low saturation transition
            if prev_sat is not None:
                sat_drop = prev_sat - curr_sat
                was_colored = prev_sat > self.high_sat_threshold
                now_desaturated = curr_sat < self.low_sat_threshold

                # Ability used if: was colored AND (now desaturated OR big drop)
                if was_colored and (now_desaturated or sat_drop > self.sat_drop_threshold):
                    usage[ability] = True
                    # Store with corrected frame index (shifted earlier by offset)
                    offset = self.get_offset_frames(ability)
                    corrected_frame = self._current_frame_idx - offset
                    self._detection_buffer.append((corrected_frame, ability))

            self.prev_saturation[ability] = curr_sat

        return usage

    def get_corrected_detections(self, frame_idx: int) -> dict[str, bool]:
        """Get detections corrected for time offset at a specific frame.

        Call this after processing all frames to get the time-corrected
        detection for each frame.
        """
        result = {'Q': False, 'W': False, 'E': False, 'R': False, 'D': False, 'F': False}
        for det_frame, ability in self._detection_buffer:
            if det_frame == frame_idx:
                result[ability] = True
        return result

    def get_all_detections(self) -> list[tuple[int, str]]:
        """Get all detections with corrected frame indices."""
        return list(self._detection_buffer)

    def clear_buffer(self):
        """Clear the detection buffer."""
        self._detection_buffer = []
        self._current_frame_idx = 0


class MousePositionEstimator:
    """Estimate mouse position relative to Garen.

    Divides the area around Garen into 18 angular slices (20 degrees each).
    The mouse is estimated based on:
    - Direction Garen is facing
    - Direction of attacks/abilities
    - Movement direction

    Slice 0 = East (right), increasing counter-clockwise.
    """

    def __init__(self, num_slices: int = 18):
        self.num_slices = num_slices
        self.slice_angle = 360 / num_slices  # 20 degrees per slice

        # Garen's longest range ability (E - Judgment) is melee range
        # For ranged champions, this would be their max attack range
        self.max_range_pixels = 200  # Approximate pixels for max ability range

    def angle_to_slice(self, angle_degrees: float) -> int:
        """Convert angle (0=right, CCW positive) to slice index."""
        # Normalize to 0-360
        angle = angle_degrees % 360
        return int(angle / self.slice_angle) % self.num_slices

    def movement_to_slice(self, dx: float, dy: float) -> int:
        """Convert movement direction to mouse slice.

        Assumption: Mouse is often in the direction of movement.
        """
        if abs(dx) < 1 and abs(dy) < 1:
            return 0  # No movement, default to right

        # atan2 gives angle in radians, CCW from positive x-axis
        angle_rad = np.arctan2(-dy, dx)  # -dy because screen y is inverted
        angle_deg = np.degrees(angle_rad)
        return self.angle_to_slice(angle_deg)


class KeylogExtractor:
    """Main class for extracting pseudo-keylogs from replay video."""

    def __init__(self, video_path: Path, output_dir: Path = None):
        self.video_path = Path(video_path)
        self.output_dir = output_dir or self.video_path.parent / "keylogs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.regions = HUDRegions1080p()
        self.hud_tracker = GarenHUDTracker(self.regions)
        self.ability_detector = AbilityBarDetector(self.regions)
        self.mouse_estimator = MousePositionEstimator(num_slices=18)

        # Video info
        self.cap = None
        self.fps = 60
        self.total_frames = 0

    def open_video(self):
        """Open video file and get metadata."""
        self.cap = cv2.VideoCapture(str(self.video_path))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return self.cap.isOpened()

    def close_video(self):
        """Close video file."""
        if self.cap:
            self.cap.release()

    def extract_frame_actions(self, frame: np.ndarray, frame_idx: int) -> ActionLabels:
        """Extract action labels from a single frame."""
        timestamp_ms = (frame_idx / self.fps) * 1000

        # Detect movement
        dx, dy, move_conf = self.hud_tracker.detect_movement(frame)
        wasd = self.hud_tracker.infer_wasd(dx, dy)

        # Detect ability usage
        abilities = self.ability_detector.detect_ability_usage(frame)

        # Estimate mouse position from movement direction
        mouse_slice = self.mouse_estimator.movement_to_slice(dx, dy)

        return ActionLabels(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            movement_dx=dx,
            movement_dy=dy,
            wasd_keys=wasd,
            ability_q_used=abilities['Q'],
            ability_w_used=abilities['W'],
            ability_e_used=abilities['E'],
            ability_r_used=abilities['R'],
            summoner_d_used=abilities['D'],
            summoner_f_used=abilities['F'],
            mouse_slice=mouse_slice,
            mouse_distance=min(1.0, np.sqrt(dx**2 + dy**2) / 50),
            movement_confidence=move_conf,
            ability_confidence=1.0 if any(abilities.values()) else 0.0
        )

    def extract_video(self, start_frame: int = 0, end_frame: int = None,
                      sample_rate: int = 1) -> list[ActionLabels]:
        """Extract action labels from video.

        Args:
            start_frame: Frame to start extraction
            end_frame: Frame to end extraction (None = end of video)
            sample_rate: Process every Nth frame (1 = all frames)

        Returns:
            List of ActionLabels for processed frames
        """
        if not self.open_video():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        if end_frame is None:
            end_frame = self.total_frames

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        labels = []
        frame_idx = start_frame

        while frame_idx < end_frame:
            ret, frame = self.cap.read()
            if not ret:
                break

            if (frame_idx - start_frame) % sample_rate == 0:
                action = self.extract_frame_actions(frame, frame_idx)
                labels.append(action)

            frame_idx += 1

        self.close_video()
        return labels

    def save_labels(self, labels: list[ActionLabels], filename: str = "keylogs.json"):
        """Save extracted labels to JSON file."""
        output_path = self.output_dir / filename

        data = []
        for label in labels:
            data.append({
                'frame_idx': label.frame_idx,
                'timestamp_ms': float(label.timestamp_ms),
                'movement': {
                    'dx': float(label.movement_dx),
                    'dy': float(label.movement_dy),
                    'wasd': label.wasd_keys,
                    'confidence': float(label.movement_confidence)
                },
                'abilities': {
                    'Q': label.ability_q_used,
                    'W': label.ability_w_used,
                    'E': label.ability_e_used,
                    'R': label.ability_r_used,
                    'D': label.summoner_d_used,
                    'F': label.summoner_f_used,
                    'confidence': label.ability_confidence
                },
                'mouse': {
                    'slice': int(label.mouse_slice),
                    'distance': float(label.mouse_distance)
                }
            })

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        return output_path


def visualize_regions(frame: np.ndarray, regions: HUDRegions1080p) -> np.ndarray:
    """Draw HUD regions on frame for debugging."""
    vis = frame.copy()

    # Draw regions with different colors for different purposes
    region_configs = [
        # Ability icons (green)
        (regions.ability_q, (0, 255, 0), 'Q'),
        (regions.ability_w, (0, 255, 0), 'W'),
        (regions.ability_e, (0, 255, 0), 'E'),
        (regions.ability_r, (0, 255, 0), 'R'),
        # Summoner spells (blue)
        (regions.summoner_d, (255, 0, 0), 'D'),
        (regions.summoner_f, (255, 0, 0), 'F'),
        # Full ability bar area (cyan)
        (regions.ability_bar_full, (255, 255, 0), 'AbilityBar'),
        # Game area for optical flow (yellow)
        (regions.game_area, (0, 255, 255), 'Game'),
        # Garen overhead HUD (red - most important for movement tracking)
        (regions.garen_level_box, (0, 0, 255), 'Lvl'),
        (regions.garen_name_area, (0, 0, 255), 'Name'),
        (regions.garen_health_bar, (0, 100, 255), 'HP'),
        # Portrait and stats (magenta)
        (regions.garen_portrait, (255, 0, 255), 'Portrait'),
        (regions.stats_area, (200, 0, 200), 'Stats'),
    ]

    for region, color, label in region_configs:
        x, y, w, h = region
        cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)
        cv2.putText(vis, label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    return vis


if __name__ == "__main__":
    import sys

    # Test on sample video
    video_path = Path("data/keylog_extraction/garen_replay.mp4")

    if video_path.exists():
        extractor = KeylogExtractor(video_path)

        # Extract first 600 frames (10 seconds at 60fps)
        print("Extracting actions from first 10 seconds...")
        labels = extractor.extract_video(start_frame=3600, end_frame=4200)  # 1 min to 1:10

        # Save results
        output_path = extractor.save_labels(labels)
        print(f"Saved {len(labels)} frame labels to {output_path}")

        # Print some stats
        wasd_frames = sum(1 for l in labels if l.wasd_keys)
        ability_frames = sum(1 for l in labels if any([
            l.ability_q_used, l.ability_w_used,
            l.ability_e_used, l.ability_r_used
        ]))

        print(f"\nStats:")
        print(f"  Frames with movement: {wasd_frames}/{len(labels)}")
        print(f"  Frames with abilities: {ability_frames}/{len(labels)}")
    else:
        print(f"Video not found: {video_path}")
