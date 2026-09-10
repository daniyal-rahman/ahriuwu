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

__all__ = [
    "ServerCommand",
    "decode_action",
    "RewardConfig",
    "LaneReward",
    "ServerBackend",
    "JsonlReplayBackend",
    "HeadlessServerBackend",
    "ControlBackend",
    "order_for_command",
    "LaneEnvConfig",
    "LaneEnv",
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
    # Canonical direction -> world direction (the rotation is its own inverse).
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


@dataclass
class RewardConfig:
    """Dense lane-phase reward weights.  **Legacy**; see :mod:`lanerl_rl.reward`.

    The reward is computed from the *unfogged* frame.  That is legitimate --
    it is a training-time signal, not an observation -- but it is the reason
    the reward function lives here and not in ``obs.py``.
    """

    damage_dealt: float = 1.0        # per unit of enemy hp fraction removed
    damage_taken: float = 1.0        # per unit of own hp fraction lost
    gold: float = 1.0 / 200.0        # per gold earned
    xp: float = 1.0 / 800.0          # per xp earned
    kill: float = 3.0
    death: float = 3.0
    turret_damage: float = 2.0       # per unit of enemy turret hp fraction removed
    turret_lost: float = 2.0
    time_penalty: float = 0.0
    zero_sum: bool = True            # subtract the opponent's raw reward


class LaneReward:
    """Per-agent dense reward from consecutive frames.  **Legacy** -- see above."""

    def __init__(self, team: int, cfg: Optional[RewardConfig] = None):
        self.team = team
        self.enemy_team = C.TEAM_RED if team == C.TEAM_BLUE else C.TEAM_BLUE
        self.cfg = cfg or RewardConfig()
        self._prev: Optional[Dict[str, float]] = None

    def reset(self) -> None:
        self._prev = None

    def _snapshot(self, frame: Frame) -> Dict[str, float]:
        me = frame.champion_of_team(self.team)
        foe = frame.champion_of_team(self.enemy_team)
        own_turret = enemy_turret = 0.0
        own_n = enemy_n = 0
        for u in frame.units.values():
            if u.etype != "turret":
                continue
            frac = 0.0 if u.mhp <= 0 else max(0.0, min(1.0, u.hp / u.mhp))
            if u.team == self.team:
                own_turret += frac
                own_n += 1
            elif u.team == self.enemy_team:
                enemy_turret += frac
                enemy_n += 1
        return {
            "my_hp": 0.0 if me is None or me.mhp <= 0 else max(0.0, me.hp / me.mhp),
            "foe_hp": 0.0 if foe is None or foe.mhp <= 0 else max(0.0, foe.hp / foe.mhp),
            "my_gold": 0.0 if me is None else float(me.gold or 0.0),
            "my_xp": 0.0 if me is None else float(me.xp or 0.0),
            "my_alive": 0.0 if me is None else float(me.hp > 0),
            "foe_alive": 0.0 if foe is None else float(foe.hp > 0),
            "own_turret": own_turret / max(own_n, 1),
            "enemy_turret": enemy_turret / max(enemy_n, 1),
        }

    def raw(self, frame: Frame) -> float:
        cur = self._snapshot(frame)
        if self._prev is None:
            self._prev = cur
            return 0.0
        p, c = self._prev, cur
        cfg = self.cfg
        r = 0.0
        r += cfg.damage_dealt * max(0.0, p["foe_hp"] - c["foe_hp"])
        r -= cfg.damage_taken * max(0.0, p["my_hp"] - c["my_hp"])
        r += cfg.gold * max(0.0, c["my_gold"] - p["my_gold"])
        r += cfg.xp * max(0.0, c["my_xp"] - p["my_xp"])
        if p["foe_alive"] > 0.5 >= c["foe_alive"]:
            r += cfg.kill
        if p["my_alive"] > 0.5 >= c["my_alive"]:
            r -= cfg.death
        r += cfg.turret_damage * max(0.0, p["enemy_turret"] - c["enemy_turret"])
        r -= cfg.turret_lost * max(0.0, p["own_turret"] - c["own_turret"])
        r -= cfg.time_penalty
        self._prev = cur
        return float(r)


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------


class ServerBackend(Protocol):
    """Anything that can produce a stream of :class:`Frame`."""

    def reset(self) -> Frame: ...
    def step(self, commands: Dict[int, ServerCommand]) -> Optional[Frame]: ...
    def close(self) -> None: ...


class JsonlReplayBackend:
    """Replays a recorded game.  Actions are accepted and **ignored**.

    Use for tests, offline observation checks and profiling.  Learning against
    this backend is meaningless: the trajectory does not respond to the policy.
    """

    ignores_actions = True

    def __init__(self, path: str | Path, max_frames: Optional[int] = None, skip: int = 0):
        self.path = Path(path)
        self.max_frames = max_frames
        self.skip = skip
        self._fh = None
        self._n = 0

    def reset(self) -> Frame:
        self.close()
        self._fh = open(self.path, "r")
        self._n = 0
        for _ in range(self.skip):
            if not self._fh.readline():
                break
        first = self._read()
        if first is None:
            raise RuntimeError(f"{self.path} is empty")
        return first

    def _read(self) -> Optional[Frame]:
        if self._fh is None:
            return None
        if self.max_frames is not None and self._n >= self.max_frames:
            return None
        line = self._fh.readline()
        if not line:
            return None
        self._n += 1
        return decode_frame(json.loads(line))

    def step(self, commands: Dict[int, ServerCommand]) -> Optional[Frame]:
        return self._read()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


# Derived from __file__, never hardcoded: the NFS export is mounted at
# /srv/nfs on the login node and /mnt/nfs on the Slurm node, and a literal for
# either one turns a job on the other into an instant exit-53 with no output.
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENDOR_ROOT = _REPO_ROOT.parent / "lanerl-vendor"
DEFAULT_SERVER_DIR = DEFAULT_VENDOR_ROOT / "LoLServer/GameServerConsole/bin/Release/net6.0"
DEFAULT_DOTNET_ROOT = DEFAULT_VENDOR_ROOT / "dotnet"
DEFAULT_GAME_CONFIG = _REPO_ROOT / "lanerl/cfg/garen1v1.json"


class HeadlessServerBackend:
    """Boots the vendored server and streams its recorded frames live.

    Observation streaming is real and works.  Action injection is **not
    implemented**, because the vendored server has no ingress for it and this
    change may not patch the vendored tree.  See :meth:`send_action`.
    """

    ignores_actions = True

    def __init__(
        self,
        config_path: str | Path,
        record_path: str | Path,
        server_dir: Path = DEFAULT_SERVER_DIR,
        dotnet_root: Path = DEFAULT_DOTNET_ROOT,
        port: int = 5119,
        freerun: bool = True,
        boot_timeout_s: float = 180.0,
        log_path: Optional[str | Path] = None,
    ):
        self.config_path = Path(config_path)
        self.record_path = Path(record_path)
        self.server_dir = Path(server_dir)
        self.dotnet_root = Path(dotnet_root)
        self.port = int(port)
        self.freerun = freerun
        self.boot_timeout_s = boot_timeout_s
        self.log_path = Path(log_path) if log_path else self.record_path.with_suffix(".log")
        self.proc: Optional[subprocess.Popen] = None
        self._fh = None
        self._pending = ""

    # -- process -----------------------------------------------------------

    def _spawn(self) -> None:
        env = dict(os.environ)
        env["DOTNET_ROOT"] = str(self.dotnet_root)
        env["LANERL_HEADLESS"] = "1"
        env["LANERL_RECORD"] = str(self.record_path)
        if self.freerun:
            env["LANERL_FREERUN"] = "1"
        self.record_path.parent.mkdir(parents=True, exist_ok=True)
        self.record_path.unlink(missing_ok=True)
        self._log_fh = open(self.log_path, "wb")
        self.proc = subprocess.Popen(
            [
                str(self.dotnet_root / "dotnet"),
                "./GameServerConsole.dll",
                "--config",
                str(self.config_path),
                "--port",
                str(self.port),
            ],
            cwd=str(self.server_dir),
            env=env,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def reset(self) -> Frame:
        self.close()
        self._spawn()
        deadline = time.monotonic() + self.boot_timeout_s
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited during boot (rc={self.proc.returncode}); see {self.log_path}"
                )
            if self.record_path.exists() and self.record_path.stat().st_size > 0:
                self._fh = open(self.record_path, "r")
                frame = self._read_blocking(timeout_s=deadline - time.monotonic())
                if frame is not None:
                    return frame
            time.sleep(0.05)
        raise TimeoutError(f"server did not produce a frame within {self.boot_timeout_s}s")

    def _read_blocking(self, timeout_s: float) -> Optional[Frame]:
        """Read one complete JSONL record, waiting for the writer if needed."""
        deadline = time.monotonic() + max(timeout_s, 0.0)
        while time.monotonic() < deadline:
            if self._fh is None:
                return None
            chunk = self._fh.readline()
            if chunk:
                self._pending += chunk
                if self._pending.endswith("\n"):
                    line, self._pending = self._pending.strip(), ""
                    if line:
                        return decode_frame(json.loads(line))
                continue
            if self.proc is not None and self.proc.poll() is not None:
                return None
            time.sleep(0.002)
        return None

    def step(self, commands: Dict[int, ServerCommand]) -> Optional[Frame]:
        if commands:
            self.send_action(commands)
        return self._read_blocking(timeout_s=10.0)

    def send_action(self, commands: Dict[int, ServerCommand]) -> None:
        """Not supported by *this* backend -- use :class:`ControlBackend`.

        ``LANERL_RECORD`` is a one-way dump: the server writes frames into a
        file and never reads anything back, so there is no direction in which an
        order could travel.  That is a property of this transport, not of the
        server.  The order ingress **exists** -- ``LanerlControl.cs`` accepts a
        TCP connection on ``LANERL_CONTROL_PORT`` and applies one JSON order per
        champion per step -- and :class:`ControlBackend` speaks it.

        Kept as a hard failure rather than a silent no-op so that a caller who
        picked the wrong backend finds out on the first action instead of after
        a training run that never learned anything.
        """
        raise NotImplementedError(
            "HeadlessServerBackend is an observation-only tail of LANERL_RECORD and "
            "cannot inject actions. Use ControlBackend, which speaks the server's "
            "LANERL_CONTROL_PORT lockstep channel."
        )

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=10)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        self.proc = None
        fh = getattr(self, "_log_fh", None)
        if fh is not None:
            fh.close()
            self._log_fh = None


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


class ControlBackend:
    """The closed loop: a headless server driven over ``LanerlControl``'s TCP channel.

    Protocol (``GameServerLib/Lanerl/LanerlControl.cs``), strict lockstep::

        server -> {"t": <game_ms>, "u": [...]}          every LANERL_STEP_TICKS ticks
        client -> {"blue": {order}, "red": {order}}

    The server blocks on the read, so the simulation advances exactly one
    decision per action and there is no way for the two sides to drift.

    Two ports, both allocated dynamically
    -------------------------------------
    The control channel (``LANERL_CONTROL_PORT``) and the game server's own
    ENet listener (``--port``) are different sockets.  Both are bound to port 0
    and read back before launch, because ``LanerlControl`` calls
    ``TcpListener.Start()`` in its constructor: a collision there kills the
    process while the trainer sits blocked on ``accept``.  Fixed ports have
    caused exactly that here.

    Who drives the champions
    ------------------------
    ``bot_teams`` maps to ``LANERL_BOT`` and defaults to ``"none"``, which is
    **not** the server's own default: ``LanerlConfig.DriveTeams`` is ``"blue"``,
    so an unset ``LANERL_BOT`` attaches the scripted bot to the blue champion and
    it overrides the policy's orders every tick.  Pass ``"purple"`` to keep red
    as a frozen scripted opponent.

    Resetting
    ---------
    :meth:`reset` sends ``{"cmd":"reset"}`` in place of an order line, which
    runs ``LanerlEpisode.Reset`` in process -- sub-millisecond, against ~12 s to
    restart the process.  The server does not answer that line immediately; it
    writes the post-reset observation at the *next* step boundary, so from here
    the exchange stays one-line-out / one-line-in like any other step.
    """

    ignores_actions = False

    def __init__(
        self,
        config_path: str | Path = DEFAULT_GAME_CONFIG,
        server_dir: Path = DEFAULT_SERVER_DIR,
        dotnet_root: Path = DEFAULT_DOTNET_ROOT,
        step_ticks: int = 4,
        toponly: bool = True,
        freerun: bool = True,
        bot_teams: str = "none",
        boot_timeout_s: float = 180.0,
        step_timeout_s: float = 60.0,
        log_path: Optional[str | Path] = None,
        extra_env: Optional[Dict[str, str]] = None,
    ):
        self.config_path = Path(config_path)
        self.server_dir = Path(server_dir)
        self.dotnet_root = Path(dotnet_root)
        self.step_ticks = int(step_ticks)
        self.toponly = bool(toponly)
        self.freerun = bool(freerun)
        self.bot_teams = str(bot_teams)
        self.boot_timeout_s = float(boot_timeout_s)
        self.step_timeout_s = float(step_timeout_s)
        self._log_path_arg = Path(log_path) if log_path else None
        #: Where the current server's stdout went.  Read it on any failure: a
        #: Content script that fails to compile only ever says so here.
        self.log_path: Optional[Path] = self._log_path_arg
        self.extra_env = dict(extra_env or {})
        self.proc: Optional[subprocess.Popen] = None
        self.sock: Optional[socket.socket] = None
        self.control_port: Optional[int] = None
        self.game_port: Optional[int] = None
        self.resets = 0
        self._buf = bytearray()
        self._log_fh = None
        self._dead = False

    # -- ports -------------------------------------------------------------

    @staticmethod
    def _free_port() -> int:
        """Ask the kernel for a port, then hand it over.

        There is an unavoidable race between closing this socket and the server
        binding it, which is why :meth:`_connect` fails loudly on a dead process
        instead of retrying forever.
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])
        finally:
            s.close()

    # -- process -----------------------------------------------------------

    def _command(self) -> List[str]:
        direct = self.server_dir / "GameServerConsole"
        if direct.exists() and os.access(direct, os.X_OK):
            exe = [str(direct)]
        else:
            dll = self.server_dir / "GameServerConsole.dll"
            dotnet = self.dotnet_root / "dotnet"
            if not dll.exists() or not dotnet.exists():
                raise RuntimeError(
                    f"no runnable server: neither {direct} nor {dll} + {dotnet}. "
                    f"Build the vendored server first."
                )
            exe = [str(dotnet), str(dll)]
        return exe + ["--config", str(self.config_path), "--port", str(self.game_port)]

    def _environment(self) -> Dict[str, str]:
        env = dict(os.environ)
        env["DOTNET_ROOT"] = str(self.dotnet_root)
        env["LANERL_HEADLESS"] = "1"
        if self.freerun:
            env["LANERL_FREERUN"] = "1"
        if self.toponly:
            env["LANERL_TOPONLY"] = "1"
        env["LANERL_CONTROL_PORT"] = str(self.control_port)
        env["LANERL_STEP_TICKS"] = str(self.step_ticks)
        # MUST be set. LanerlConfig.DriveTeams defaults to "blue", so leaving
        # LANERL_BOT unset attaches the in-server scripted bot to the blue
        # champion, where it re-issues its own orders every tick and quietly
        # overrides the policy -- the loop looks closed and is not. Set it to
        # "purple" for a frozen scripted opponent, "both" for a bot-vs-bot
        # baseline.
        env["LANERL_BOT"] = self.bot_teams
        env.update({k: str(v) for k, v in self.extra_env.items()})
        return env

    def _spawn(self) -> None:
        if not self.config_path.exists():
            raise RuntimeError(f"game config {self.config_path} does not exist")
        self.control_port = self._free_port()
        self.game_port = self._free_port()
        # Recomputed per spawn so a restart does not truncate the log that
        # explains why the previous server died.
        self.log_path = self._log_path_arg or (
            _REPO_ROOT / "lanerl" / "logs" / f"control_{self.control_port}.log"
        )
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_fh = open(self.log_path, "wb")
        self._dead = False
        self.proc = subprocess.Popen(
            self._command(),
            cwd=str(self.server_dir),
            env=self._environment(),
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # so close() can kill the whole group
        )

    def _connect(self) -> None:
        """Wait for ``LanerlControl`` to accept.

        The listener only exists once the map has loaded, so a cold start
        legitimately takes tens of seconds.  Process liveness is rechecked every
        round: a server that dies while we wait must surface as its own error,
        not as a full connect timeout.
        """
        deadline = time.monotonic() + self.boot_timeout_s
        last_err: Optional[BaseException] = None
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(
                    f"server exited with rc={self.proc.returncode} before accepting on "
                    f"control port {self.control_port}; see {self.log_path}"
                )
            try:
                s = socket.create_connection(("127.0.0.1", self.control_port), timeout=2.0)
            except OSError as exc:
                last_err = exc
                time.sleep(0.25)
                continue
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            s.settimeout(self.step_timeout_s)
            self.sock = s
            self._buf = bytearray()
            return
        raise TimeoutError(
            f"no connection to control port {self.control_port} within "
            f"{self.boot_timeout_s:.0f}s (last error: {last_err}); see {self.log_path}"
        )

    # -- transport ---------------------------------------------------------

    def _send(self, line: str) -> None:
        if self.sock is None:
            raise RuntimeError("control channel is closed")
        if line != RESET_LINE and _RESET_TOKEN in line:
            # Belt and braces against an older server build, whose OnTick tested
            # for this token as a substring of the whole line and would have
            # restarted the episode instead of moving the champion.
            raise ValueError(f"order line contains the reset token: {line!r}")
        self.sock.sendall((line + "\n").encode("ascii"))

    def _recv_line(self) -> Optional[str]:
        """One complete line, or ``None`` when the server closed the channel."""
        if self.sock is None:
            return None
        while True:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                line = self._buf[:nl].decode("ascii", errors="replace")
                del self._buf[: nl + 1]
                return line
            try:
                chunk = self.sock.recv(65536)
            except socket.timeout:
                raise TimeoutError(
                    f"no observation from the server within {self.step_timeout_s:.0f}s "
                    f"(control port {self.control_port}); see {self.log_path}"
                ) from None
            except OSError:
                self._dead = True
                return None
            if not chunk:  # EOF: the episode is over, do not block waiting for more
                self._dead = True
                return None
            self._buf.extend(chunk)

    def _exchange(self, line: str) -> Optional[Frame]:
        if self._dead:
            return None
        try:
            self._send(line)
        except OSError:
            # The server exited between our last read and this write. That is an
            # episode ending, not an exception the caller should handle: `step`
            # is documented to return None, and a raise here would abort a
            # rollout over a server that simply finished.
            self._dead = True
            return None
        raw = self._recv_line()
        if raw is None:
            return None
        return decode_frame(json.loads(raw))

    # -- ServerBackend -----------------------------------------------------

    def reset(self) -> Frame:
        """Start (or restart) an episode and return its first observation."""
        if (
            not self._dead
            and self.sock is not None
            and self.proc is not None
            and self.proc.poll() is None
        ):
            frame = self._exchange(RESET_LINE)
            if frame is not None:
                self.resets += 1
                return frame
            # The channel died mid-reset; fall through to a cold restart rather
            # than hand back a stale frame.
        self.close()
        self._spawn()
        self._connect()
        raw = self._recv_line()  # the server writes the first observation unprompted
        if raw is None:
            raise RuntimeError(
                f"control channel closed before the first observation; see {self.log_path}"
            )
        return decode_frame(json.loads(raw))

    def step(self, commands: Dict[int, ServerCommand]) -> Optional[Frame]:
        return self._exchange(json.dumps(self.encode(commands), separators=(",", ":")))

    def encode(self, commands: Dict[int, ServerCommand]) -> Dict[str, Dict[str, object]]:
        """Per-team commands -> the ``{"blue": ..., "red": ...}`` action line.

        A team with no command gets an explicit ``noop`` rather than being left
        out: ``ApplyActions`` skips a missing key, which is the same *effect*
        but makes a dropped command indistinguishable from an intended hold.
        """
        return {
            "blue": order_for_command(commands.get(C.TEAM_BLUE)),
            "red": order_for_command(commands.get(C.TEAM_RED)),
        }

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
            self.sock = None
        self._buf = bytearray()
        self._dead = True
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=10)
            except Exception:
                # A server wedged inside its tick loop ignores SIGTERM; leaking
                # it would hold both ports and the next run would fail to bind.
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    self.proc.wait(timeout=10)
                except Exception:
                    pass
        self.proc = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------


@dataclass
class LaneEnvConfig:
    teams: Tuple[int, int] = (C.TEAM_BLUE, C.TEAM_RED)
    max_steps: int = 6000
    end_on_death: bool = True
    end_on_turret_loss: bool = True
    #: The training reward.  ``reward.LaneRewardConfig``, not the legacy
    #: ``env.RewardConfig``: the HP term has to be a potential *difference* and
    #: the gold term has to have the ambient trickle removed.  See ``reward.py``.
    reward: LaneRewardConfig = field(default_factory=LaneRewardConfig)
    move_distance: float = 500.0
    warn_on_approx_fog: bool = True


class LaneEnv:
    """Two-agent 1v1 lane environment.

    ``reset`` / ``step`` return dicts keyed by team id, so a self-play loop can
    treat the two sides symmetrically -- which is the whole point of the
    mirroring in :mod:`lanerl_rl.obs`.
    """

    def __init__(
        self,
        backend: ServerBackend,
        cfg: Optional[LaneEnvConfig] = None,
        train_step_source: Optional[Callable[[], int]] = None,
    ):
        self.backend = backend
        self.cfg = cfg or LaneEnvConfig()
        fog = ApproxFogModel(warn=self.cfg.warn_on_approx_fog)
        self.builders: Dict[int, ObservationBuilder] = {
            t: ObservationBuilder(t, fog_model=fog) for t in self.cfg.teams
        }
        self.reward = ZeroSumLaneReward(self.cfg.teams, self.cfg.reward)
        #: Live source for :attr:`train_step`.  Pass
        #: ``lanerl_train.run.TrainingLoop.train_steps`` here and the anneal
        #: follows the learner without anybody having to push a value in.
        self._train_step_source = train_step_source
        self._train_step = 0
        self.frame: Optional[Frame] = None
        self.steps = 0
        self._slot_netids: Dict[int, List[Optional[int]]] = {
            t: [None] * C.N_SLOTS for t in self.cfg.teams
        }
        self._last_obs: Dict[int, AgentObservation] = {}

    # -- the training clock ------------------------------------------------

    @property
    def train_step(self) -> int:
        """How far along the run is; feeds ``LaneRewardConfig.alpha``'s anneal.

        The environment has no idea how far along a run is, so the trainer owns
        this.  Either push it (``env.train_step = n``) or, better, construct the
        env with ``train_step_source=loop.train_steps`` and it tracks the
        learner by itself -- a pushed value that somebody forgets to push is how
        the anneal came to be pinned at its starting alpha for every run so far.
        """
        if self._train_step_source is not None:
            return int(self._train_step_source())
        return self._train_step

    @train_step.setter
    def train_step(self, value: int) -> None:
        if self._train_step_source is not None:
            raise RuntimeError(
                "this LaneEnv reads train_step from a source callable; assigning to it "
                "would be silently ignored on the next step. Drop the assignment or "
                "drop the source."
            )
        self._train_step = int(value)

    def set_train_step_source(self, source: Optional[Callable[[], int]]) -> None:
        """Attach (or detach) the live training-clock source."""
        self._train_step_source = source

    # -- api ---------------------------------------------------------------

    def reset(self) -> Dict[int, AgentObservation]:
        for b in self.builders.values():
            b.reset()
        self.reward.reset()
        self.steps = 0
        self.frame = self.backend.reset()
        return self._observe()

    def step(
        self, actions: Dict[int, Dict[str, int]]
    ) -> Tuple[Dict[int, AgentObservation], Dict[int, float], bool, Dict[str, object]]:
        if self.frame is None:
            raise RuntimeError("call reset() first")
        commands = self.decode(actions)
        nxt = self.backend.step(commands)
        self.steps += 1
        if nxt is None:
            obs = self._observe()
            return obs, {t: 0.0 for t in self.cfg.teams}, True, {"reason": "stream_end"}
        self.frame = nxt
        obs = self._observe()
        rew, rinfo = self.reward.step(self.frame, self.train_step)
        done, reason = self._terminated()
        info = {
            "reason": reason,
            "t_ms": self.frame.t_ms,
            "raw_reward": rinfo["raw"],
            "reward_info": rinfo,
        }
        return obs, rew, done, info

    def decode(self, actions: Dict[int, Dict[str, int]]) -> Dict[int, ServerCommand]:
        """Decode per-team action dicts into server orders (world coordinates)."""
        out: Dict[int, ServerCommand] = {}
        if self.frame is None:
            return out
        for team, action in actions.items():
            builder = self.builders[team]
            me = self.frame.champion_of_team(team)
            if me is None:
                continue
            obs = self._last_obs.get(team)
            if obs is None:
                continue
            cmd = decode_action(
                action,
                builder,
                obs,
                me,
                self._slot_netids[team],
                move_distance=self.cfg.move_distance,
            )
            if cmd.kind == "cast" and cmd.spell_slot is not None:
                builder.note_cast(cmd.spell_slot, self.frame.t_ms)
            if cmd.kind == "attack_move" and cmd.target_netid is not None:
                # AttackClock has no server-side source -- the control channel
                # reports spell cooldowns but not the auto-attack timer -- so the
                # swing has to be told to it here.  Only a *targeted* attack_move
                # counts: without a target the order goes on the wire as a plain
                # move (see order_for_command), no swing is issued, and noting
                # one would fabricate an attack phase out of nothing.
                builder.note_attack(self.frame.t_ms)
            if me.recalling is None:
                # Old recordings carry no "rc" flag, so the best available
                # signal is the order we just issued.  Live, `_observe`
                # overwrites this from the server's own channel state, which is
                # the only thing that knows the 8 s channel is still running.
                builder.set_recalling(cmd.kind == "recall")
            out[team] = cmd
        return out

    def close(self) -> None:
        self.backend.close()

    # -- internals ---------------------------------------------------------

    def _observe(self) -> Dict[int, AgentObservation]:
        assert self.frame is not None
        out: Dict[int, AgentObservation] = {}
        for team, builder in self.builders.items():
            me = self.frame.champion_of_team(team)
            if me is not None and me.recalling is not None:
                # The agent's OWN recall channel: HUD information, and the only
                # honest source for it -- the order that started the channel was
                # issued ~120 decisions ago and may have been cancelled since.
                builder.set_recalling(bool(me.recalling))
            o = builder.build(self.frame)
            out[team] = o
            self._slot_netids[team] = self._slot_netids_for(builder, self.frame, team)
        self._last_obs = out
        return out

    def _slot_netids_for(
        self, builder: ObservationBuilder, frame: Frame, team: int
    ) -> List[Optional[int]]:
        """Recover the net id behind each slot, so the target head can address units.

        This mirrors ``ObservationBuilder._slot_entities`` exactly; it lives on
        the *environment* side, not in the observation, so that no slot->netid
        mapping ever becomes an input feature.
        """
        me = frame.champion_of_team(team)
        if me is None:
            return [None] * C.N_SLOTS
        from .frame import visible_ids_for

        visible, _ = visible_ids_for(frame, team, builder.fog_model)
        ax, ay = builder.transform.point(me.x, me.y)
        slots = builder._slot_entities(frame.t_ms, ax, ay, me.id, visible)
        return [None if e is None else e.uid for e in slots]

    def _terminated(self) -> Tuple[bool, str]:
        assert self.frame is not None
        if self.steps >= self.cfg.max_steps:
            return True, "max_steps"
        if self.cfg.end_on_death:
            for t in self.cfg.teams:
                ch = self.frame.champion_of_team(t)
                if ch is not None and not ch.alive:
                    return True, f"death_team_{t}"
        if self.cfg.end_on_turret_loss:
            for u in self.frame.units.values():
                if u.etype == "turret" and u.hp <= 0:
                    return True, f"turret_lost_team_{u.team}"
        return False, ""
