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
    of the unit the target head selected, or ``None``.
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
    move_distance: float = 500.0,
) -> ServerCommand:
    """Turn the four categorical heads into one server order.

    The move heads live in the agent's **canonical** frame, so the chosen
    direction is rotated back into world space through the agent's mirror
    transform before it becomes a click position.  That inverse is what makes a
    red-side policy trained in canonical space produce correct red-side orders.
    """
    button = C.BUTTONS[int(action["button"])]
    if button == "noop":
        return ServerCommand(kind="noop")
    if button == "recall":
        return ServerCommand(kind="recall")

    tx = float(C.MOVE_BIN_VALUES[int(action["move_x"])])
    tz = float(C.MOVE_BIN_VALUES[int(action["move_z"])])
    norm = math.hypot(tx, tz)
    if norm > 1e-6:
        tx, tz = tx / norm, tz / norm
    # Canonical direction -> world direction (a reflection, so it IS its own inverse -- but LaneTransform.vector is lane->world and to_lane_vector is the inverse; do not swap them).
    wx, wy = builder.transform.vector(tx, tz)
    px = self_unit.x + wx * move_distance
    py = self_unit.y + wy * move_distance

    slot_idx = int(action["target"])
    target_netid = slot_netids[slot_idx] if slot_idx < len(slot_netids) else None
    if observation.entities[slot_idx, C.E_VALID] < 0.5:
        target_netid = None

    if button == "move":
        return ServerCommand(kind="move", x=px, y=py)
    if button == "attack_move":
        return ServerCommand(kind="attack_move", x=px, y=py, target_netid=target_netid)
    spell = {"q": 0, "w": 1, "e": 2, "r": 3}[button]
    return ServerCommand(kind="cast", spell_slot=spell, x=px, y=py, target_netid=target_netid)


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
    """One :class:`ServerCommand` -> one ``LanerlControl.Execute`` order object.

    The mapping is the wire contract in ``LanerlControl.cs``:

    ==============  ==========================================================
    ``kind``        order
    ==============  ==========================================================
    ``noop``        ``{"t":"noop"}`` -- League orders persist, so this is
                    action-repeat, not a stop
    ``move``        ``{"t":"move","x":..,"y":..}``
    ``cast``        ``{"t":"cast","slot":..}`` plus ``id``/``x``/``y``
    ``attack_move`` ``{"t":"attack","id":..}`` when a target was selected,
                    otherwise a plain ``move``.  The engine only swings at
                    targets already in range, so closing the distance is the
                    policy's job -- that is the behaviour we want learned.
    ``recall``      ``{"t":"recall"}`` -- casts the blue pill in
                    ``SpellSlotType.BluePillSlot`` (13), a 0.5 s windup plus an
                    8 s channel that a move order or any non-periodic damage
                    cancels.  Not an instant teleport; see ``LanerlControl.cs``.
    ==============  ==========================================================
    """
    if cmd is None or cmd.kind == "noop":
        return {"t": "noop"}
    if cmd.kind == "recall":
        return {"t": "recall"}
    if cmd.kind == "move":
        return {"t": "move", "x": _coord(cmd.x), "y": _coord(cmd.y)}
    if cmd.kind == "attack_move":
        if cmd.target_netid is not None:
            return {"t": "attack", "id": int(cmd.target_netid)}
        return {"t": "move", "x": _coord(cmd.x), "y": _coord(cmd.y)}
    if cmd.kind == "cast":
        if cmd.spell_slot is None:
            raise ValueError("a cast command needs a spell_slot")
        # `id` is always sent, though LanerlWire.ParseOrder treats it as optional
        # for "cast": an absent key is fine, and 0 is the documented "no target"
        # value that FindUnit short-circuits on. Sending it explicitly just keeps
        # every cast order the same shape.
        order: Dict[str, object] = {
            "t": "cast",
            "slot": int(cmd.spell_slot),
            "id": 0 if cmd.target_netid is None else int(cmd.target_netid),
        }
        if cmd.x is not None and cmd.y is not None:
            order["x"] = _coord(cmd.x)
            order["y"] = _coord(cmd.y)
        return order
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

