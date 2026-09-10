"""Test setup.

The repo root goes on ``sys.path`` because ``lanerl_train`` is used as a plain
top-level package here (like ``lanerl_rl`` and ``lanerl_bot``), not an installed
distribution.  Resolved from ``__file__`` -- the NFS export is at a different
absolute path on each node, so a hardcoded root would work on exactly one of
them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: needs a real game server")


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "run"
    d.mkdir()
    return d
