"""OCR reader for extracting game state from frames using EasyOCR."""

import re
from functools import lru_cache

import cv2
import numpy as np

# Lazy import easyocr to avoid slow startup
_reader = None


def get_reader():
    """Get or create EasyOCR reader (lazy init)."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def crop_region(frame: np.ndarray, region: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a region from frame. Region is (x, y, w, h)."""
    x, y, w, h = region
    return frame[y:y+h, x:x+w]


def preprocess_for_ocr(crop: np.ndarray) -> np.ndarray:
    """Preprocess image crop for better OCR accuracy."""
    # Convert to grayscale
    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    # Increase contrast
    gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)

    # Threshold to get clean black/white
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return thresh


def read_text(frame: np.ndarray, region: tuple[int, int, int, int]) -> str:
    """Read text from a region of the frame."""
    crop = crop_region(frame, region)
    processed = preprocess_for_ocr(crop)

    reader = get_reader()
    results = reader.readtext(processed, detail=0)

    return " ".join(results).strip()


def read_number(frame: np.ndarray, region: tuple[int, int, int, int]) -> int | None:
    """Read a number from a region. Returns None if not found."""
    text = read_text(frame, region)

    # Extract digits
    digits = re.sub(r"[^\d]", "", text)

    if digits:
        return int(digits)
    return None


def read_gold(frame: np.ndarray, region: tuple[int, int, int, int]) -> int | None:
    """Read gold value from HUD region."""
    return read_number(frame, region)


def read_cs(frame: np.ndarray, region: tuple[int, int, int, int]) -> int | None:
    """Read CS count from HUD region."""
    return read_number(frame, region)


def read_game_clock(frame: np.ndarray, region: tuple[int, int, int, int]) -> int | None:
    """Read game clock and return total seconds.

    Expects format like "12:34" or "1:23".
    Returns total seconds (e.g., 754 for 12:34).
    """
    text = read_text(frame, region)

    # Try to parse MM:SS format
    match = re.search(r"(\d{1,2}):(\d{2})", text)
    if match:
        minutes = int(match.group(1))
        seconds = int(match.group(2))
        return minutes * 60 + seconds

    return None


def read_health_percent(frame: np.ndarray, region: tuple[int, int, int, int]) -> float | None:
    """Estimate health percentage from health bar region.

    Uses color detection rather than OCR - looks for green/red pixels.
    Returns 0.0 to 1.0, or None if detection fails.
    """
    crop = crop_region(frame, region)

    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

    # Green health bar range (adjust as needed)
    green_lower = np.array([35, 100, 100])
    green_upper = np.array([85, 255, 255])

    # Create mask for green pixels
    mask = cv2.inRange(hsv, green_lower, green_upper)

    # Calculate percentage of green pixels in the bar
    total_pixels = mask.shape[0] * mask.shape[1]
    green_pixels = np.count_nonzero(mask)

    if total_pixels == 0:
        return None

    # Rough estimate - health bars are typically horizontal
    # Find leftmost and rightmost green pixel columns
    col_sums = np.sum(mask, axis=0)
    green_cols = np.where(col_sums > 0)[0]

    if len(green_cols) == 0:
        return 0.0

    # Health percent based on how far the green extends
    rightmost = green_cols[-1]
    return rightmost / mask.shape[1]


class GameStateReader:
    """Read full game state from a frame."""

    def __init__(self, hud_regions: dict):
        """Initialize with HUD region coordinates.

        Args:
            hud_regions: Dict with keys like 'gold', 'cs', 'game_clock', etc.
                         Values are (x, y, w, h) tuples.
        """
        self.regions = hud_regions

    def read_frame(self, frame: np.ndarray) -> dict:
        """Extract all game state values from a frame."""
        state = {}

        if self.regions.get("gold"):
            state["gold"] = read_gold(frame, self.regions["gold"])

        if self.regions.get("cs"):
            state["cs"] = read_cs(frame, self.regions["cs"])

        if self.regions.get("game_clock"):
            state["game_time_seconds"] = read_game_clock(frame, self.regions["game_clock"])

        if self.regions.get("player_health"):
            state["player_health"] = read_health_percent(frame, self.regions["player_health"])

        if self.regions.get("enemy_health"):
            state["enemy_health"] = read_health_percent(frame, self.regions["enemy_health"])

        return state
