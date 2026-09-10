"""N headless servers, one batched policy forward, actions scattered back.

Why batching is structural, not an optimisation
----------------------------------------------
Measured on this stack: 1.70 ms per decision when the policy is called once per
env, 0.058 ms per decision at batch 24.  The simulator runs 16 instances at
~688x real time aggregate (43x each; 120x at N=1 -- per-instance throughput
degrades with instance count because the node is memory/L3 bound).  At 15 Hz
decisions, the unbatched policy retains 41% of that throughput and the batched
one retains 96%.  So :class:`VecDriver` performs exactly **one forward pass per
distinct policy per step**, never one per env.

Why the send/recv split
-----------------------
``LanerlControl`` blocks on ``_in.ReadLine()`` after emitting each observation,
so a server only advances once it has our action.  Reading instance 0 to
completion before writing to instance 1 would serialise 16 simulators behind one
another.  :meth:`VecLaneEnv.step` therefore writes *all* actions first and then
multiplexes the reads with ``selectors``, so all 16 processes simulate
concurrently while we wait.

Why the noise
-------------
Every death path here logs at ``ERROR`` and every unrecoverable one raises.  The
server is not perfectly stable, and a swallowed instance death shows up as a
quietly smaller batch, a quietly lower CS average, and days of confusion.  There
is no silent degradation in this module by construction: if an instance cannot
be brought back within its restart budget, the runner stops.
"""

from __future__ import annotations

import errno
import json
import logging
import os
import selectors
import signal
import socket
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

from . import paths
from .ports import InstancePorts, PortAllocator, assert_unique
from .protocols import BLUE, SIDES, ControlAction, RawObs, Side
from .serverlog import LogTail, cs_at

__all__ = [
    "InstanceDied",
    "VecEnvFailure",
    "ServerLaunchSpec",
    "InstanceHandle",
    "ServerInstance",
    "StepResult",
    "VecLaneEnv",
    "SideAssignment",
    "EpisodeSpec",
    "VecDriver",
    "RESET_ACTION",
]

log = logging.getLogger("lanerl_train.vec")

#: ``LanerlControl.OnTick`` triggers an in-process reset on any action line
#: containing the literal ``"reset"`` -- 0.23 ms, against 12.08 s for a process
#: restart.  Because the check is a substring test, no ordinary action payload
#: may ever contain that token; :func:`_encode_line` enforces it.
RESET_ACTION = {"reset": 1}
_RESET_TOKEN = '"reset"'


class InstanceDied(RuntimeError):
    """One server instance is gone.  Recoverable by restart, never by ignoring."""


class VecEnvFailure(RuntimeError):
    """The vec runner cannot continue.  Always fatal."""


# --------------------------------------------------------------------------
# Launching one instance
# --------------------------------------------------------------------------


@dataclass
class ServerLaunchSpec:
    """Everything that turns into a server process.

    Defaults come from :mod:`lanerl_train.paths`, i.e. from ``__file__``, so the
    same spec is valid on either node without a path edit.
    """

    config_path: Optional[Path] = None
    server_dir: Optional[Path] = None
    dotnet_root: Optional[Path] = None
    step_ticks: int = 4  # 4 ticks at the server's 60 Hz == 15 Hz decisions
    toponly: bool = True
    freerun: bool = True
    #: Which champions the *in-server scripted bot* drives: "none" for
    #: policy-vs-policy, "purple" to make red a frozen scripted anchor.
    bot_teams: str = "none"
    bot_config: Optional[Path] = None
    bot_seed: Optional[int] = None
    extra_env: Dict[str, str] = field(default_factory=dict)
    connect_timeout_s: float = 180.0
    #: Killing a process group is best-effort; this bounds the wait.
    shutdown_timeout_s: float = 10.0

    def resolved_config(self) -> Path:
        return Path(self.config_path) if self.config_path else paths.default_game_config()

    def resolved_server_dir(self) -> Path:
        return Path(self.server_dir) if self.server_dir else paths.server_dir()

    def resolved_dotnet_root(self) -> Path:
        return Path(self.dotnet_root) if self.dotnet_root else paths.dotnet_root()

    def command(self, game_port: int) -> List[str]:
        d = self.resolved_server_dir()
        direct = d / "GameServerConsole"
        cfg = self.resolved_config()
        if not cfg.exists():
            raise VecEnvFailure(f"game config {cfg} does not exist")
        if direct.exists() and os.access(direct, os.X_OK):
            exe = [str(direct)]
        else:
            dll = d / "GameServerConsole.dll"
            dotnet = self.resolved_dotnet_root() / "dotnet"
            if not dll.exists() or not dotnet.exists():
                raise VecEnvFailure(
                    f"no runnable server: neither {direct} nor {dll} + {dotnet}. "
                    f"Build the vendored server before starting a run."
                )
            exe = [str(dotnet), str(dll)]
        return exe + ["--config", str(cfg), "--port", str(game_port)]

    def environment(self, ports: InstancePorts) -> Dict[str, str]:
        env = dict(os.environ)
        env["DOTNET_ROOT"] = str(self.resolved_dotnet_root())
        env["LANERL_HEADLESS"] = "1"
        if self.freerun:
            env["LANERL_FREERUN"] = "1"
        if self.toponly:
            env["LANERL_TOPONLY"] = "1"
        env["LANERL_STEP_TICKS"] = str(int(self.step_ticks))
        env["LANERL_BOT"] = self.bot_teams
        if self.bot_config is not None:
            cfg = Path(self.bot_config)
            if not cfg.exists():
                raise VecEnvFailure(f"bot config {cfg} does not exist")
            env["LANERL_BOT_CONFIG"] = str(cfg)
        if self.bot_seed is not None:
            env["LANERL_BOT_SEED"] = str(int(self.bot_seed))
        env.update(ports.as_env())
        env.update({k: str(v) for k, v in self.extra_env.items()})
        return env


class InstanceHandle(Protocol):
    """The transport the vec runner needs.  Fakeable without a live server."""

    index: int

    def start(self) -> None: ...
    def send_line(self, line: str) -> None: ...
    def read_line(self) -> Optional[str]:
        """A complete line, or ``None`` if one is not buffered yet.

        Raises :class:`InstanceDied` on EOF or transport error.
        """
        ...

    def fileno(self) -> int:
        """A selectable fd, or ``-1`` when the handle cannot be selected on."""
        ...

    def is_alive(self) -> bool: ...
    def diagnostics(self) -> str:
        """Human-readable context for a loud death message."""
        ...

    def close(self) -> None: ...


class ServerInstance:
    """One headless server process plus its TCP control channel."""

    def __init__(
        self,
        index: int,
        ports: InstancePorts,
        spec: ServerLaunchSpec,
        log_dir: Path,
        logger: Optional[logging.Logger] = None,
    ):
        self.index = int(index)
        self.ports = ports
        self.spec = spec
        self.log_dir = Path(log_dir)
        self.log = logger or log
        self.log_path = self.log_dir / f"instance{self.index:03d}.log"
        self.proc: Optional[subprocess.Popen] = None
        self.sock: Optional[socket.socket] = None
        self.tail = LogTail(self.log_path)
        self._buf = bytearray()
        self._log_fh = None
        self._started_at: float = 0.0
        self.starts = 0

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        self.close()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.unlink(missing_ok=True)
        self.tail.rewind()
        cmd = self.spec.command(self.ports.game)
        env = self.spec.environment(self.ports)
        self._log_fh = self.log_path.open("wb")
        self._started_at = time.monotonic()
        self.starts += 1
        self.proc = subprocess.Popen(
            cmd,
            cwd=str(self.spec.resolved_server_dir()),
            env=env,
            stdout=self._log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,  # so close() can kill the whole group
        )
        self.log.info(
            "instance %d: spawned pid=%d control=%d game=%d log=%s",
            self.index,
            self.proc.pid,
            self.ports.control,
            self.ports.game,
            self.log_path,
        )
        self._connect()

    def _connect(self) -> None:
        """Wait for ``LanerlControl`` to accept.

        The listener is created on the first tick of ``LanerlHooks``, after the
        map loads, so this legitimately takes tens of seconds on a cold start.
        The loop checks process liveness every round: a server that dies while
        we wait must not turn into a full connect timeout.
        """
        deadline = time.monotonic() + self.spec.connect_timeout_s
        last_err: Optional[BaseException] = None
        while time.monotonic() < deadline:
            self.tail.poll()
            if self.proc is not None and self.proc.poll() is not None:
                raise InstanceDied(
                    f"instance {self.index}: server exited with rc="
                    f"{self.proc.returncode} before accepting on control port "
                    f"{self.ports.control}. {self.diagnostics()}"
                )
            try:
                s = socket.create_connection(("127.0.0.1", self.ports.control), timeout=2.0)
            except OSError as exc:
                last_err = exc
                time.sleep(0.25)
                continue
            s.setblocking(False)
            s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock = s
            self._buf = bytearray()
            self.log.info(
                "instance %d: control channel up on %d after %.1fs",
                self.index,
                self.ports.control,
                time.monotonic() - self._started_at,
            )
            return
        raise InstanceDied(
            f"instance {self.index}: no connection to control port {self.ports.control} "
            f"within {self.spec.connect_timeout_s:.0f}s (last error: {last_err}). "
            f"{self.diagnostics()}"
        )

    # -- transport ---------------------------------------------------------

    def send_line(self, line: str) -> None:
        if self.sock is None:
            raise InstanceDied(f"instance {self.index}: send on a closed channel")
        data = (line + "\n").encode("ascii")
        view = memoryview(data)
        deadline = time.monotonic() + 30.0
        while view:
            try:
                sent = self.sock.send(view)
            except BlockingIOError:
                if time.monotonic() > deadline:
                    raise InstanceDied(
                        f"instance {self.index}: control socket not writable for 30s. "
                        f"{self.diagnostics()}"
                    ) from None
                time.sleep(0.001)
                continue
            except OSError as exc:
                raise InstanceDied(
                    f"instance {self.index}: send failed ({exc}). {self.diagnostics()}"
                ) from exc
            if sent == 0:
                raise InstanceDied(f"instance {self.index}: peer closed. {self.diagnostics()}")
            view = view[sent:]

    def read_line(self) -> Optional[str]:
        if self.sock is None:
            raise InstanceDied(f"instance {self.index}: read on a closed channel")
        while True:
            nl = self._buf.find(b"\n")
            if nl >= 0:
                line = self._buf[:nl].decode("ascii", errors="replace")
                del self._buf[: nl + 1]
                return line
            try:
                chunk = self.sock.recv(65536)
            except (BlockingIOError, InterruptedError):
                return None
            except OSError as exc:
                if exc.errno == errno.EAGAIN:
                    return None
                raise InstanceDied(
                    f"instance {self.index}: recv failed ({exc}). {self.diagnostics()}"
                ) from exc
            if not chunk:
                raise InstanceDied(
                    f"instance {self.index}: control channel closed by the server "
                    f"(the server sets SetToExit on any control error). {self.diagnostics()}"
                )
            self._buf.extend(chunk)

    def fileno(self) -> int:
        return self.sock.fileno() if self.sock is not None else -1

    def is_alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None and self.sock is not None

    def diagnostics(self) -> str:
        rc = None if self.proc is None else self.proc.poll()
        ev = self.tail.poll()
        fatal = (self.tail.fatal_seen + ev.fatal_lines)[-3:]
        tail_lines = ev.lines[-3:]
        return (
            f"[instance {self.index} rc={rc} starts={self.starts} log={self.log_path}"
            + (f" fatal={fatal}" if fatal else "")
            + (f" tail={tail_lines}" if tail_lines else "")
            + "]"
        )

    def close(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:  # already gone; nothing to recover
                pass
            self.sock = None
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=self.spec.shutdown_timeout_s)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                    self.proc.wait(timeout=self.spec.shutdown_timeout_s)
                except Exception as exc:
                    self.log.error(
                        "instance %d: could not kill pid %s (%s); it may be leaking a port",
                        self.index,
                        getattr(self.proc, "pid", "?"),
                        exc,
                    )
        self.proc = None
        if self._log_fh is not None:
            self._log_fh.close()
            self._log_fh = None
        self.tail.close()


# --------------------------------------------------------------------------
# The vector runner
# --------------------------------------------------------------------------


def _encode_line(action: Mapping[str, Any]) -> str:
    """Serialise one action line, guarding the reset sentinel.

    ``LanerlControl`` resets the episode on *any* line containing ``"reset"``,
    so an action that happened to carry that token would silently wipe the
    episode instead of moving the champion.
    """
    line = json.dumps(action, separators=(",", ":"))
    if _RESET_TOKEN in line and action != RESET_ACTION:
        raise VecEnvFailure(
            f"action line contains the reset sentinel {_RESET_TOKEN} and would silently "
            f"reset the episode: {line}"
        )
    if "\n" in line:  # pragma: no cover - json never emits raw newlines
        raise VecEnvFailure("action line contains a newline")
    return line


@dataclass
class StepResult:
    """What one vec step produced.

    ``obs[i] is None`` means instance ``i`` produced nothing this step; its
    trajectory must be dropped, not padded.
    """

    obs: List[Optional[RawObs]]
    alive: List[bool]
    #: Instances that died this step (index -> reason).  Already logged loudly.
    died: Dict[int, str] = field(default_factory=dict)
    #: Instances restarted this step.  Their recurrent state must be reset and
    #: any in-flight trajectory discarded.
    restarted: List[int] = field(default_factory=list)

    @property
    def n_alive(self) -> int:
        return sum(self.alive)


class VecLaneEnv:
    """N server instances behind one send-all / receive-all interface."""

    def __init__(
        self,
        n: int,
        spec: Optional[ServerLaunchSpec] = None,
        log_dir: Optional[Path] = None,
        ports: Optional[Sequence[InstancePorts]] = None,
        factory: Optional[Callable[[int, InstancePorts], InstanceHandle]] = None,
        step_timeout_s: float = 60.0,
        auto_restart: bool = True,
        max_restarts_per_instance: int = 5,
        max_total_restarts: int = 50,
        logger: Optional[logging.Logger] = None,
    ):
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        self.n = int(n)
        self.spec = spec or ServerLaunchSpec()
        self.log = logger or log
        self.log_dir = Path(log_dir) if log_dir else paths.runs_root() / "vec_logs"
        if ports is None:
            ports = PortAllocator().allocate(self.n)
        if len(ports) != self.n:
            raise ValueError(f"got {len(ports)} port blocks for {self.n} instances")
        assert_unique(ports)  # a shared port kills everything but the first instance
        self.ports = list(ports)
        self._factory = factory or self._default_factory
        self.step_timeout_s = float(step_timeout_s)
        self.auto_restart = bool(auto_restart)
        self.max_restarts_per_instance = int(max_restarts_per_instance)
        self.max_total_restarts = int(max_total_restarts)

        self.handles: List[InstanceHandle] = [
            self._factory(i, self.ports[i]) for i in range(self.n)
        ]
        self.alive: List[bool] = [False] * self.n
        self.last_obs: List[Optional[RawObs]] = [None] * self.n
        self.restarts: List[int] = [0] * self.n
        self.total_restarts = 0
        self.deaths: List[int] = [0] * self.n
        self._started = False

    def _default_factory(self, index: int, ports: InstancePorts) -> InstanceHandle:
        return ServerInstance(index, ports, self.spec, self.log_dir, self.log)

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> StepResult:
        """Spawn every instance and collect the unprompted first observation."""
        for i, h in enumerate(self.handles):
            try:
                h.start()
                self.alive[i] = True
            except Exception as exc:
                self.alive[i] = False
                self.log.error("instance %d: FAILED TO START: %s", i, exc, exc_info=True)
        if not any(self.alive):
            raise VecEnvFailure(
                "no server instance started. This is a setup failure (build, config or "
                "ports), not a transient -- see the per-instance logs in "
                f"{self.log_dir}."
            )
        self._started = True
        # LanerlControl writes one observation before reading any action.
        result = self._collect(list(range(self.n)))
        self._restart_dead(result)
        return result

    def close(self) -> None:
        for h in self.handles:
            try:
                h.close()
            except Exception as exc:  # never let one bad handle strand the rest
                self.log.error("instance %d: error during close: %s", h.index, exc)
        self.alive = [False] * self.n
        self._started = False

    def __enter__(self) -> "VecLaneEnv":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- stepping ----------------------------------------------------------

    def step(self, actions: Sequence[Optional[Mapping[str, Any]]]) -> StepResult:
        """Send one action line per live instance, then read one observation each.

        ``actions[i] is None`` sends a bare ``{}``: ``LanerlControl.ApplyActions``
        finds no side key, so both champions keep their standing orders.  That is
        real action-repeat, not a no-op hack.
        """
        if not self._started:
            raise VecEnvFailure("step() before start()")
        if len(actions) != self.n:
            raise ValueError(f"got {len(actions)} actions for {self.n} instances")
        pending = self._send_all(actions)
        result = self._collect(pending)
        self._restart_dead(result)
        return result

    def reset_episodes(self, indices: Sequence[int]) -> StepResult:
        """In-process episode reset (0.23 ms) on the named instances.

        Every live instance still needs exactly one action line this step --
        the server is blocked reading one -- so instances not in ``indices``
        receive an empty action and keep their orders.
        """
        want = set(int(i) for i in indices)
        bad = want - set(range(self.n))
        if bad:
            raise ValueError(f"reset_episodes: no such instances {sorted(bad)}")
        actions: List[Optional[Mapping[str, Any]]] = [
            dict(RESET_ACTION) if i in want else None for i in range(self.n)
        ]
        return self.step(actions)

    def _send_all(self, actions: Sequence[Optional[Mapping[str, Any]]]) -> List[int]:
        pending: List[int] = []
        for i, action in enumerate(actions):
            if not self.alive[i]:
                continue
            try:
                self.handles[i].send_line(_encode_line(action if action is not None else {}))
                pending.append(i)
            except InstanceDied as exc:
                self._mark_dead(i, str(exc))
            except VecEnvFailure:
                raise  # a malformed action is our bug, not the server's
        return pending

    def _collect(self, pending: Sequence[int]) -> StepResult:
        obs: List[Optional[RawObs]] = [None] * self.n
        died: Dict[int, str] = {}
        remaining = {i for i in pending if self.alive[i]}
        deadline = time.monotonic() + self.step_timeout_s
        while remaining:
            progressed = False
            for i in sorted(remaining):
                try:
                    line = self.handles[i].read_line()
                except InstanceDied as exc:
                    died[i] = str(exc)
                    self._mark_dead(i, str(exc))
                    remaining.discard(i)
                    progressed = True
                    continue
                if line is None:
                    continue
                remaining.discard(i)
                progressed = True
                try:
                    parsed = json.loads(line)
                except ValueError as exc:
                    reason = (
                        f"instance {i}: unparseable observation ({exc}); "
                        f"first 200 chars: {line[:200]!r}"
                    )
                    died[i] = reason
                    self._mark_dead(i, reason)
                    continue
                obs[i] = parsed
                self.last_obs[i] = parsed
            if not remaining:
                break
            left = deadline - time.monotonic()
            if left <= 0:
                for i in sorted(remaining):
                    reason = (
                        f"instance {i}: no observation within {self.step_timeout_s:.1f}s "
                        f"{self._diag(i)}"
                    )
                    died[i] = reason
                    self._mark_dead(i, reason)
                break
            if not progressed:
                self._wait_readable(remaining, min(left, 0.5))
        self._check_logs(died)
        return StepResult(obs=obs, alive=list(self.alive), died=died)

    def _wait_readable(self, indices, timeout_s: float) -> None:
        fds = {i: self.handles[i].fileno() for i in indices}
        if any(fd is None or fd < 0 for fd in fds.values()):
            # A handle that cannot be selected on (a fake, or one mid-restart).
            time.sleep(min(timeout_s, 0.0005))
            return
        sel = selectors.DefaultSelector()
        try:
            for i, fd in fds.items():
                sel.register(fd, selectors.EVENT_READ, i)
            sel.select(timeout_s)
        except (OSError, ValueError) as exc:
            # A closed fd between the fileno() call and register(); the next
            # read_line() will surface the real cause.
            self.log.warning("select failed (%s); falling back to a poll", exc)
            time.sleep(min(timeout_s, 0.0005))
        finally:
            sel.close()

    def _diag(self, i: int) -> str:
        try:
            return self.handles[i].diagnostics()
        except Exception as exc:  # diagnostics must never mask the real failure
            return f"[diagnostics unavailable: {exc}]"

    def _check_logs(self, died: Dict[int, str]) -> None:
        """Surface server-side fatals even when the socket is still open."""
        for i, h in enumerate(self.handles):
            tail = getattr(h, "tail", None)
            if tail is None or not self.alive[i]:
                continue
            try:
                ev = tail.poll()
            except Exception as exc:  # pragma: no cover - defensive
                self.log.error("instance %d: log tail failed: %s", i, exc)
                continue
            for line in ev.fatal_lines:
                reason = f"instance {i}: server logged a fatal line: {line}"
                self.log.error("%s", reason)
                died.setdefault(i, reason)
                self._mark_dead(i, reason)

    def _mark_dead(self, i: int, reason: str) -> None:
        if self.alive[i]:
            self.deaths[i] += 1
            self.log.error("INSTANCE DEATH %d/%d: %s", i, self.n, reason)
        self.alive[i] = False
        try:
            self.handles[i].close()
        except Exception as exc:
            self.log.error("instance %d: close after death failed: %s", i, exc)

    # -- restart -----------------------------------------------------------

    def _restart_dead(self, result: StepResult) -> None:
        if not self.auto_restart:
            result.alive = list(self.alive)
            return
        for i in range(self.n):
            if self.alive[i]:
                continue
            if self.restarts[i] >= self.max_restarts_per_instance:
                raise VecEnvFailure(
                    f"instance {i} died {self.deaths[i]} times and has exhausted its restart "
                    f"budget ({self.max_restarts_per_instance}). Stopping rather than training "
                    f"on a silently smaller batch. Log: {self._diag(i)}"
                )
            if self.total_restarts >= self.max_total_restarts:
                raise VecEnvFailure(
                    f"{self.total_restarts} instance restarts across the run have exhausted the "
                    f"global budget ({self.max_total_restarts}); the server is unstable enough "
                    f"that the run is not worth continuing."
                )
            self.restarts[i] += 1
            self.total_restarts += 1
            self.log.error(
                "RESTARTING instance %d (restart %d/%d for this instance, %d total). "
                "A process restart costs ~12s against 0.23ms for an in-process reset, so "
                "this is a real throughput hit, not a free retry.",
                i,
                self.restarts[i],
                self.max_restarts_per_instance,
                self.total_restarts,
            )
            try:
                self.handles[i].start()
            except Exception as exc:
                self.log.error("instance %d: restart FAILED: %s", i, exc, exc_info=True)
                continue
            self.alive[i] = True
            # The freshly booted server emits its first observation unprompted.
            fresh = self._collect([i])
            result.died.update(fresh.died)
            if fresh.obs[i] is not None:
                result.obs[i] = fresh.obs[i]
                result.restarted.append(i)
            else:
                self.log.error(
                    "instance %d: restarted but produced no first observation", i
                )
        result.alive = list(self.alive)


# --------------------------------------------------------------------------
# Batched policy driving
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SideAssignment:
    """Which policy drives each champion in one instance.

    ``None`` means *the in-server scripted bot drives that side*: the action
    line simply omits the key, ``LanerlControl.ApplyActions`` finds no object
    for it, and the bot's own orders stand.  That is how a frozen scripted
    anchor is played without a second network.
    """

    blue: Optional[str]
    red: Optional[str]

    def key_for(self, side: Side) -> Optional[str]:
        return self.blue if side == BLUE else self.red


@dataclass
class EpisodeSpec:
    """When an episode is over.

    ``max_game_ms`` defaults to ten minutes because CS@10 is the headline
    absolute metric -- an episode that stops earlier cannot contribute one.
    """

    max_game_ms: int = 600_000
    end_on_death: bool = True
    #: Guard against a server whose clock stops advancing.
    max_steps: int = 20_000


def episode_done(raw: RawObs, spec: EpisodeSpec, steps: int) -> Tuple[bool, str]:
    if steps >= spec.max_steps:
        return True, "max_steps"
    t = int(raw.get("t", 0))
    if t >= spec.max_game_ms:
        return True, "time"
    if spec.end_on_death:
        for u in raw.get("u", ()):
            if u.get("k") == "Champion" and int(u.get("hp", 1)) <= 0:
                return True, f"death_team_{u.get('tm')}"
    return False, ""


class VecDriver:
    """Owns the batched forward and the scatter.

    Slot bookkeeping is the whole job.  Each ``(instance, side)`` pair driven by
    a network is a *slot* in that policy's batch; the slot list is fixed for the
    lifetime of an assignment, so the recurrent state stays column-aligned.  A
    dead instance keeps its slot and is fed its last observation -- padding a
    fixed-width batch is cheaper than reindexing a GRU hidden state, and its
    action is discarded on the way out.
    """

    def __init__(
        self,
        env: VecLaneEnv,
        policies: Mapping[str, Any],
        adapter_factory: Callable[[int, Side], Any],
        encoder: Any,
        assignments: Sequence[SideAssignment],
        episode: Optional[EpisodeSpec] = None,
        logger: Optional[logging.Logger] = None,
    ):
        self.env = env
        self.policies = dict(policies)
        self.adapter_factory = adapter_factory
        self.encoder = encoder
        self.episode = episode or EpisodeSpec()
        self.log = logger or log
        self.adapters: Dict[Tuple[int, Side], Any] = {}
        self.steps_in_episode: List[int] = [0] * env.n
        self.episode_index: List[int] = [0] * env.n
        self._pending_resets: List[bool] = [True] * env.n
        self.set_assignments(assignments)

    # -- assignment --------------------------------------------------------

    def set_assignments(self, assignments: Sequence[SideAssignment]) -> None:
        if len(assignments) != self.env.n:
            raise ValueError(
                f"got {len(assignments)} assignments for {self.env.n} instances"
            )
        self.assignments = list(assignments)
        slots: Dict[str, List[Tuple[int, Side]]] = {}
        for i, a in enumerate(self.assignments):
            for side in SIDES:
                key = a.key_for(side)
                if key is None:
                    continue
                if key not in self.policies:
                    raise VecEnvFailure(
                        f"instance {i} side {side} is assigned to policy {key!r}, which is "
                        f"not in the policy map {sorted(self.policies)}"
                    )
                slots.setdefault(key, []).append((i, side))
                self.adapters.setdefault((i, side), self.adapter_factory(i, side))
        self.slots = slots
        self.states = {k: self.policies[k].initial_state(len(v)) for k, v in slots.items()}
        self._pending_resets = [True] * self.env.n

    # -- rollout -----------------------------------------------------------

    def start(self) -> StepResult:
        result = self.env.start()
        missing = [i for i in range(self.env.n) if self.env.last_obs[i] is None]
        if missing:
            raise VecEnvFailure(
                f"instances {missing} produced no first observation after start (and any "
                f"restarts). Every slot needs one before the first batched forward -- this "
                f"is a setup failure, not something to pad around. Logs: "
                f"{[self.env._diag(i) for i in missing]}"
            )
        self._on_new_observations(result)
        return result

    def step(self, deterministic: bool = False) -> Tuple[StepResult, Dict[int, str]]:
        """One decision for every instance.  Returns ``(step result, done reasons)``.

        Exactly ``len(self.slots)`` policy forwards happen here -- one per
        distinct policy, not one per env.
        """
        actions_per_slot = self._forward(deterministic)
        lines = self._scatter(actions_per_slot)
        result = self.env.step(lines)
        self._on_new_observations(result)
        dones = self._episode_boundaries(result)
        if dones:
            reset = self.env.reset_episodes(sorted(dones))
            self._on_new_observations(reset)
            for i in dones:
                self.episode_index[i] += 1
                self.steps_in_episode[i] = 0
                self._pending_resets[i] = True
                for side in SIDES:
                    ad = self.adapters.get((i, side))
                    if ad is not None:
                        ad.reset()
            # The reset step carries the fresh first observation of the next
            # episode; hand it back so the caller is never a step behind.
            for i in range(self.env.n):
                if reset.obs[i] is not None:
                    result.obs[i] = reset.obs[i]
            result.alive = list(reset.alive)
            result.died.update(reset.died)
            result.restarted.extend(reset.restarted)
        return result, dones

    def _forward(self, deterministic: bool) -> Dict[Tuple[int, Side], Any]:
        out: Dict[Tuple[int, Side], Any] = {}
        for key, slot_list in self.slots.items():
            batch = []
            resets = []
            for i, side in slot_list:
                raw = self.env.last_obs[i]
                if raw is None:
                    raise VecEnvFailure(
                        f"instance {i} has no observation to act on; start() must have "
                        f"produced one for every live instance"
                    )
                batch.append(self.adapters[(i, side)].build(raw, side))
                resets.append(self._pending_resets[i])
            actions, self.states[key] = self.policies[key].act_batch(
                batch, self.states[key], resets=resets, deterministic=deterministic
            )
            if len(actions) != len(slot_list):
                raise VecEnvFailure(
                    f"policy {key!r} returned {len(actions)} actions for {len(slot_list)} "
                    f"slots; the scatter would silently misroute"
                )
            for (i, side), action in zip(slot_list, actions):
                out[(i, side)] = action
        self._pending_resets = [False] * self.env.n
        return out

    def _scatter(
        self, actions_per_slot: Mapping[Tuple[int, Side], Any]
    ) -> List[Optional[Dict[str, ControlAction]]]:
        lines: List[Optional[Dict[str, ControlAction]]] = [None] * self.env.n
        for (i, side), action in actions_per_slot.items():
            if not self.env.alive[i]:
                continue  # dead slots keep their column but their action is dropped
            raw = self.env.last_obs[i]
            if raw is None:  # pragma: no cover - guarded in _forward
                continue
            encoded = self.encoder.encode(action, raw, side)
            slot = lines[i]
            if slot is None:
                slot = {}
                lines[i] = slot
            slot[side] = encoded
        return lines

    def _on_new_observations(self, result: StepResult) -> None:
        for i in range(self.env.n):
            if result.obs[i] is not None:
                self.steps_in_episode[i] += 1
        for i in result.restarted:
            self.steps_in_episode[i] = 0
            self.episode_index[i] += 1
            self._pending_resets[i] = True
            for side in SIDES:
                ad = self.adapters.get((i, side))
                if ad is not None:
                    ad.reset()

    def _episode_boundaries(self, result: StepResult) -> Dict[int, str]:
        dones: Dict[int, str] = {}
        for i in range(self.env.n):
            raw = result.obs[i]
            if raw is None or not self.env.alive[i]:
                continue
            done, reason = episode_done(raw, self.episode, self.steps_in_episode[i])
            if done:
                dones[i] = reason
        return dones

    # -- telemetry ---------------------------------------------------------

    def cs_at_10(self, index: int) -> Dict[int, int]:
        """CS per team at ten minutes for instance ``index``, from its own log.

        Empty when the instance has not logged a ``LANERL_CS`` row yet.  A
        missing team is *absent*, never zero: reporting 0 for a crashed episode
        would drag the headline metric down with no warning.
        """
        tail = getattr(self.env.handles[index], "tail", None)
        if tail is None:
            return {}
        tail.poll()
        return {team: row.cs for team, row in cs_at(tail.cs_rows, 600_000).items()}
