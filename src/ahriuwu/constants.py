"""Shared constants and paths for the ahriuwu package.

Constants live here to avoid cross-package imports between models and data.
Paths are configurable via environment variables with sensible defaults.
"""

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Domain constants
# --------------------------------------------------------------------------- #

MOVEMENT_DIM = 2  # Continuous (x, y) cursor location in [0, 1] screen coords.
# Action keys (binary per frame):
#   Q W E R - champion abilities
#   D F     - summoner spells
#   B       - recall
#   C       - attack-move-click (AA initiation; covers all AAs whether keyed
#             via attack-move or right-click on enemy)
# `item` was tracked here previously; dropped because pipeline doesn't capture
# item activations and the signal is sparse (~20/game).
ABILITY_KEYS = ['Q', 'W', 'E', 'R', 'D', 'F', 'B', 'C']

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
