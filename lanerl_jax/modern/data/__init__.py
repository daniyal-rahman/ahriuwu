"""Pinned patch data of the 26.19 modern world and the loaders/builders that make it.

``PATCH_DIR`` holds the client-derived tables every rule module reads
(items, runes, economy, jungle, objectives, geometry, champion records);
``ORACLE_DIR`` holds the recorded-game oracles the fidelity tests compare with.
The ``build_*`` modules regenerate the tables from the client research dumps.
"""
from pathlib import Path

PATCH = "26.19"
PATCH_DIR = Path(__file__).resolve().parent / PATCH
ORACLE_DIR = Path(__file__).resolve().parent / "oracle"
