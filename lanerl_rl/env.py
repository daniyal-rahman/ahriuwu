"""Two-agent environment scaffolding for the 1v1 Garen lane.

Layering
--------
``LaneEnv`` owns the RL semantics -- observations, action decoding, reward,
termination -- and talks to a :class:`ServerBackend` for the actual simulation.
Three backends ship here:

``JsonlReplayBackend``
    Replays a ``LANERL_RECORD`` JSONL dump.  Fully working, runs offline, and
    is what the test suite uses.  **It ignores actions**: it is a fixed
    trajectory, useful for exercising the observation/model/PPO plumbing and
    for offline analysis, not for learning.

``HeadlessServerBackend``
    Boots the vendored server and tails its ``LANERL_RECORD`` file.
    Observation-only: the record file is a one-way stream, so actions have
    nowhere to go.  Superseded by :class:`ControlBackend`; kept because reading
    an existing dump needs no control port and no lockstep.

``ControlBackend``
    **The closed loop.**  Boots the server with ``LANERL_CONTROL_PORT`` set and
    speaks ``GameServerLib/Lanerl/LanerlControl.cs``: newline-delimited JSON
    over TCP, strict lockstep, one observation per action, ``LANERL_STEP_TICKS``
    server ticks apart.  Actions really drive the champions and
    :meth:`ControlBackend.reset` restarts the episode *in process* (sub-ms)
    rather than respawning the server (~12 s).

Nothing here silently pretends to close the loop.
"""

from __future__ import annotations

import json
import math
import os
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Protocol, Sequence, Tuple

import numpy as np

from . import constants as C
from . import projection
from .frame import ApproxFogModel, Frame, MirrorTransform, Unit, decode_frame
from .obs import AgentObservation, ObservationBuilder
from .reward import LaneRewardConfig, ZeroSumLaneReward

#: What survives here after 2026-09-12. `lanerl_train.lane_wiring` imports
#: exactly these two functions; everything else this module used to export --
#: LaneEnv, LaneEnvConfig, ControlBackend, the three backends, and the legacy
#: RewardConfig/LaneReward pair -- was a SECOND implementation of the obs /
#: action / reward wiring with no production caller, and it owned the unit
#: tests while lane_wiring.py owned the runtime. That is how EpisodeSpec and
#: LaneEnvConfig drifted to different max_steps (20,000 vs 6,000) and different
#: end_on_death without anything failing. Deleted; recoverable at cec588a. The
#: live path keeps its real-server coverage in lanerl_train/tests/
#: test_live_server.py and test_e2e_real_server.py.
__all__ = [
    "ServerCommand",
    "decode_action",
    "order_for_command",
]


# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ServerCommand:
    """A decoded champion order, in **raw world** coordinates.

    ``kind`` is one of ``noop``, ``move``, ``attack_move``, ``cast``, ``recall``.
    ``spell_slot`` is 0..3 for Q/W/E/R.  ``target_netid`` is the server net id
    used only by explicit legacy diagnostic commands; policy decoding never sets it.
    """

    kind: str
    x: Optional[float] = None
    y: Optional[float] = None
    spell_slot: Optional[int] = None
    target_netid: Optional[int] = None


def decode_action(
    action: Dict[str, int],
    builder: ObservationBuilder,
    observation: AgentObservation,
    self_unit: Unit,
    slot_netids: Sequence[Optional[int]],
) -> ServerCommand:
    """Turn button and screen-coordinate heads into one server order.

    The move heads live in the agent's **canonical** frame, so the chosen
    direction is rotated back into world space through the agent's mirror
    transform before it becomes a click position.  That inverse is what makes a
    red-side policy trained in canonical space produce correct red-side orders.
    """
    for name, limit in (("button", C.N_BUTTONS), ("screen_x", C.N_SCREEN_X),
                        ("screen_y", C.N_SCREEN_Y)):
        if not 0 <= int(action[name]) < limit:
            raise ValueError(f"{name} index is outside the action grid")
    button = C.BUTTONS[int(action["button"])]
    if button == "noop":
        return ServerCommand(kind="noop")
    if button == "recall":
        return ServerCommand(kind="recall")

    # WHERE ON SCREEN the mouse is. No move_distance: the click names a
    # POINT, the way a human's does, so "step 150 units back" is expressible.
    # The old space picked a 9x9 lane-local DIRECTION and travelled a
    # hardcoded 500 units, which cannot express any small adjustment -- and a
    # small adjustment is exactly what melee last-hitting is (attack range
    # 125, minion aggro several hundred).
    sx = float(C.SCREEN_X_VALUES[int(action["screen_x"])])
    sy = float(C.SCREEN_Y_VALUES[int(action["screen_y"])])
    if (button in ("move", "attack_move", "r")
            and sx >= projection.MINIMAP_X_MIN and sy >= projection.MINIMAP_Y_MIN):
        return ServerCommand(kind="noop")
    try:
        # The screen offset from the champion, in a CANONICAL champion-centred
        # view. Taken about the origin so it is a pure offset, then mapped
        # through the lane frame.
        #
        # Why not straight to world: the observation is lane-canonical -- blue
        # and red see the same picture via the (s,n) -> (L-s,n) reflection --
        # but a world-aligned camera makes one screen position mean OPPOSITE
        # lane directions for the two sides. The policy would read a canonical
        # frame and act in a non-canonical one, which is the same class of bug
        # as the rot180 canonicalisation that mapped blue's top lane onto red's
        # bot. test_the_decoded_ORDER_is_equivariant_under_the_reflection
        # catches it.
        #
        # So the screen axes ARE the lane axes: +x is down-lane towards the
        # enemy, +y is across it. A deployed vision+mouse shim would rotate
        # this back into true screen space, which is a display concern, not a
        # policy one.
        ds, dn = projection.screen_to_world_centred(0.0, 0.0, sx, sy)
    except ValueError:
        # The horizon: that screen row has no ground point at all. Standing
        # still is the honest response -- inventing a destination would send
        # the champion somewhere the action never named.
        return ServerCommand(kind="noop")
    wx, wy = builder.transform.vector(ds, dn)
    px = self_unit.x + wx
    py = self_unit.y + wy

    # Entity hit-testing belongs to the server. Observation slots and any legacy
    # ``target`` action field cannot alter the chosen screen location.
    if button == "move":
        return ServerCommand(kind="move", x=px, y=py)
    if button == "attack_move":
        return ServerCommand(kind="attack_move", x=px, y=py)
    spell = {"q": 0, "w": 1, "e": 2, "r": 3}[button]
    return ServerCommand(kind="cast", spell_slot=spell, x=px, y=py)


# --------------------------------------------------------------------------
# Reward (LEGACY -- `LaneEnv` uses `reward.ZeroSumLaneReward`)
# --------------------------------------------------------------------------
#
# The pair below is the original raw delta-hp / delta-gold / delta-xp sum.  It
# is superseded by :mod:`lanerl_rl.reward`, whose HP term is a potential
# *difference* and whose gold term subtracts the 1.9 gold/s ambient trickle --
# two corrections worth roughly half the reward budget between them.  Kept
# because the mirror and decode tests build one directly, and because it is the
# baseline the new reward is compared against.


def order_for_command(cmd: Optional[ServerCommand]) -> Dict[str, object]:
    """Serialize coordinate-only policy commands.

    The server resolves clicks against visible, alive units. Entity-ID attack
    and cast orders remain available directly on the diagnostic wire, but this
    production adapter never emits them.
    """
    if cmd is None or cmd.kind == "noop":
        return {"t": "noop"}
    if cmd.kind == "recall":
        return {"t": "recall"}
    if cmd.kind == "move":
        return {"t": "click", "button": "move", "x": _coord(cmd.x), "y": _coord(cmd.y)}
    if cmd.target_netid is not None:
        raise ValueError("policy commands cannot select an entity ID; use screen coordinates")
    if cmd.kind == "attack_move":
        return {"t": "click", "button": "attack_move", "x": _coord(cmd.x), "y": _coord(cmd.y)}
    if cmd.kind == "cast":
        if cmd.spell_slot not in (0, 1, 2, 3):
            raise ValueError("a cast command needs spell_slot 0..3")
        return {"t": "click", "button": ("q", "w", "e", "r")[cmd.spell_slot],
                "x": _coord(cmd.x), "y": _coord(cmd.y)}
    raise ValueError(f"unknown ServerCommand kind {cmd.kind!r}")


def _coord(v: Optional[float]) -> float:
    """Round a world coordinate to 2 dp.

    ``LanerlControl.Num`` scans digits, ``-``, ``+`` and ``.`` only, so a float
    that ``json.dumps`` renders in exponent form (``1e-05``) would be truncated
    to its mantissa.  Rounding puts every value we can produce -- champion
    position +/- ~500 units -- in plain decimal, and collapses the sub-0.01
    values that would have used an exponent to 0.0.
    """
    if v is None:
        raise ValueError("a move/cast order needs both x and y")
    return round(float(v), 2)


#: The in-process episode reset line.  ``LanerlWire.Parse`` accepts exactly
#: this object and nothing else: a ``"cmd"`` line may carry no other key, and
#: an unknown top-level key makes the whole line Fatal -- no reset, and both
#: champions' orders dropped for that step.
#:
#: ``OnTick`` used to decide this with ``line.Contains("\"reset\"")``, so any
#: order that happened to carry the token restarted the episode.  That is gone,
#: but :meth:`ControlBackend._send` still refuses the token in an ordinary
#: order: this package is also run against older server builds, and the failure
#: it prevented is silent.
RESET_LINE = '{"cmd":"reset"}'
_RESET_TOKEN = '"reset"'

