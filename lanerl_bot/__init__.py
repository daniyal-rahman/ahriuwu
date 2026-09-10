"""lanerl_bot -- the scripted Garen lane bot and fast episode reset for the headless
League 1v1 RL environment.

The bot itself runs *inside* the C# server (GameServerLib/Lanerl/), because it has to
be the frozen opponent for self-play and an IPC round trip per tick would both slow
that down and make the anchor depend on a Python process being alive. This package is
the other half:

  damage.py    the last-hit model as a pure Python reference, kept honest against the
               C# implementation by a parity test rather than by good intentions
  content.py   champion/minion numbers read out of the server's own JSON content, so
               nothing here is a magic number that can silently go stale
  build.py     Season-4 skill order and item path, with the justification attached
  telemetry.py parsing of the server's JSONL and log output
"""

from lanerl_bot.damage import (  # noqa: F401
    LastHitModel,
    post_mitigation,
    is_last_hit,
    windup_seconds,
)
from lanerl_bot.build import GAREN_SKILL_ORDER, GAREN_BUILD_PATH, skill_at  # noqa: F401

__all__ = [
    "LastHitModel",
    "post_mitigation",
    "is_last_hit",
    "windup_seconds",
    "GAREN_SKILL_ORDER",
    "GAREN_BUILD_PATH",
    "skill_at",
]
