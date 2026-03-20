"""Shared constants and paths for the ahriuwu package.

Constants live here to avoid cross-package imports between models and data.
Paths are configurable via environment variables with sensible defaults.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #

MOVEMENT_DIM = 2  # Continuous movement (x, y)
ABILITY_KEYS = ['Q', 'W', 'E', 'R', 'D', 'F', 'item', 'B']

# --------------------------------------------------------------------------- #
# Project paths — override via environment variables
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(os.environ.get(
    "AHRIUWU_ROOT", Path(__file__).resolve().parent.parent.parent
))
DATA_ROOT = Path(os.environ.get(
    "AHRIUWU_DATA", "/mnt/storage/ahriuwu-data"
))
OUTPUT_ROOT = Path(os.environ.get(
    "AHRIUWU_OUTPUT", "/mnt/storage/ahriuwu-data"
))
LATENTS_DIR = Path(os.environ.get(
    "AHRIUWU_LATENTS", "/opt/ahriuwu/latents_pt"
))

# Derived paths
FRAMES_DIR = DATA_ROOT / "frames"
PROCESSED_DIR = DATA_ROOT / "processed"
REPLAYS_DIR = DATA_ROOT / "replays"
CHECKPOINTS_DIR = OUTPUT_ROOT / "checkpoints"
LOGS_DIR = OUTPUT_ROOT / "logs"
