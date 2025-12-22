"""OCR modules for extracting game state from frames."""

from .gold_reader import GoldReader
from .cs_reader import CSReader
from .health_detector import HealthDetector
from .game_clock import GameClockReader

__all__ = ["GoldReader", "CSReader", "HealthDetector", "GameClockReader"]
