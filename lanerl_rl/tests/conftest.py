"""Shared fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

from lanerl_rl import constants as C
from lanerl_rl.frame import ApproxFogModel

#: The recorded game shipped with the repo.  4846 ticks of a real headless run.
RECORDING = Path(__file__).resolve().parents[2] / "lanerl" / "logs" / "state.jsonl"


@pytest.fixture(scope="session")
def recording_path() -> Path:
    if not RECORDING.exists():
        pytest.skip(f"recording not found at {RECORDING}")
    return RECORDING


@pytest.fixture
def quiet_fog() -> ApproxFogModel:
    """A fog model that does not emit the once-per-process fallback warning."""
    return ApproxFogModel(warn=False)
