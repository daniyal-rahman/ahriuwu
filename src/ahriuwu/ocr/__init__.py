"""OCR modules for extracting game state from frames."""

from .reader import (
    GameStateReader,
    read_gold,
    read_cs,
    read_game_clock,
    read_health_percent,
    read_text,
    read_number,
)

__all__ = [
    "GameStateReader",
    "read_gold",
    "read_cs",
    "read_game_clock",
    "read_health_percent",
    "read_text",
    "read_number",
]
