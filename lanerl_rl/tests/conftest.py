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


#: 400 frames subsampled from a real 173 s bot-vs-bot game, recorded
#: 2026-09-12 AFTER the wire started carrying ad/ap/ar/mr. The older
#: lanerl/logs/state.jsonl predates those fields, so a guard that checks "is
#: every declared feature ever written" would flag them as dead against it --
#: for the wrong reason.
FRAMES_V2 = Path(__file__).resolve().parent / "data" / "frames_v2.jsonl"


@pytest.fixture(scope="session")
def lane_frames():
    """Decoded Frames from the recording, for tests that need real variety."""
    import json

    from lanerl_rl.frame import decode_frame

    if not FRAMES_V2.exists():
        pytest.skip(f"frame fixture not found at {FRAMES_V2}")
    out = []
    for line in FRAMES_V2.read_text().splitlines():
        if line.strip():
            out.append(decode_frame(json.loads(line)))
    return out
