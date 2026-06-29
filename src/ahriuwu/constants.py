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
# Action keys: binary per-frame "fired this frame". Garen v1 action space,
# parsed in ahriuwu.data.replay_dataset from clicks.json casts (by spell_name) +
# per-frame label.action + inventory:
#   Q W E R   champion abilities (spell_name GarenQ/W/E/R). NOTE: the pipeline
#             logs Garen's E as EITHER 'GarenE' or 'GarenECancel' inconsistently
#             (per-match one or the other, never both, across 131 matches) — they
#             are ALIASES for the same "E used", so both map to E (no cancel key).
#   Flash     SummonerFlash           (only Flash/Ignite kept of the summoners)
#   Ignite    SummonerDot
#   AA        auto-attack / attack-move initiation (label.action.type -> attack)
#   Recall    spell_name 'recall'
#   Stride    Stridebreaker active (item 6631), from inventory `lf` jumps (sparse)
# Dropped — no clean signal in the labels: pots (lf dead / uc garbage), tiamat
# (lf dead), ward (lf ~1-4/game vs ~15, no location), TP, super-recall.
ABILITY_KEYS = ['Q', 'W', 'E', 'R', 'Flash', 'Ignite', 'AA', 'Recall', 'Stride']

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
