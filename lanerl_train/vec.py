"""N headless servers, one batched policy forward, actions scattered back.

Why batching is structural, not an optimisation
----------------------------------------------
Measured on this stack: 1.70 ms per decision when the policy is called once per
env, 0.058 ms per decision at batch 24.  The simulator runs 16 instances at
~688x real time aggregate (43x each; 120x at N=1 -- per-instance throughput
degrades with instance count because the node is memory/L3 bound).  Both figures
reproduce from ``lanerl/logs/scaling/n*_i*.log``: the final ``LANERL_TPS``
speedup averages 43.0x over the 16 instances, 687.9x summed.

Retention, with the formula written out because the number here used to be
unreproducible.  43x is 2580 ticks/s, and a decision is ``STEP_TICKS`` ticks, so
the simulator spends ``2 / 2580 = 0.78 ms`` of wall per decision at the current
30 Hz.  ``sim / (sim + policy)`` is then **31% unbatched and 93% batched** (it
was 48% / 96% at 15 Hz, where a decision cost 1.55 ms of sim).  The "41% and
96%" that stood here for a while was two different formulas in one sentence --
41% is ``16 x 1.70 / 66.7 ms`` of budget consumed, 96% is per-decision
retention -- so it could not be checked either way.  The conclusion is
unchanged and got stronger: :class:`VecDriver` performs exactly **one forward
pass per distinct policy per step**, never one per env.

Caveat on all of the above: the scaling logs are ``LANERL_FREERUN`` servers with
no control channel and no policy, so 43x is an upper bound on simulator
throughput and the retention figures are optimistic.

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

#: The in-process episode reset -- 0.23 ms, against 12.08 s for a process
#: restart.
#:
#: This used to be ``{"reset": 1}``, from the era when ``LanerlControl.OnTick``
#: tested ``line.Contains("\"reset\"")``.  ``LanerlWire.Parse`` now reads the
#: line as JSON and a reset is *exactly* ``{"cmd":"reset"}``; under that parser
#: the old payload is an unknown top-level key, which makes the whole line
#: Fatal, so the episode did not reset **and** neither champion moved.  Silent
#: in the logs, visible only as a game clock that never rewinds.
from lanerl_rl import constants as _C

RESET_ACTION = {"cmd": "reset"}

#: What ``LanerlWire.Parse`` accepts at the top level of an action line.
#: Anything else makes the line Fatal and drops both champions' orders.
_TOP_LEVEL_KEYS = frozenset({"blue", "red"})


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
    #: Server ticks per decision. DERIVED, never a second literal: this was
    #: hardcoded to 4 (15 Hz) while lanerl_rl.constants.STEP_TICKS said 2
    #: (30 Hz), so PPOConfig computed gamma for 30 Hz while the servers ran at
    #: 15. `--horizon-s 30` therefore bought a 60 s horizon -- double what was
    #: asked for -- and global_vec.dt_norm sat at a constant 2.0. Confirmed
    #: from the run's own logs: 111.6 ticks/s / 27.7 decisions/s = 4.03.
    step_ticks: int = _C.STEP_TICKS
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


def _die_with_parent() -> None:
    """Ask the kernel to SIGKILL this child when its parent dies.

    Runs in the forked child between fork and exec.

    Why this is not paranoia: a game server outlives EVERY external kill of the
    trainer -- `scancel`, an OOM, a plain `kill` on the parent python. The
    trainer's own shutdown path closes them, but that path does not run when
    the trainer is killed rather than asked to stop. The orphans keep their
    ports, and because `PORTS_PER_ACTOR_STRIDE` is 64 an actor's block spans
    768 ports, so a single survivor anywhere in that band kills a LATER run
    with `SocketException (98): Address already in use` at a completely
    unrelated instance index.

    Exactly that happened on 2026-09-13: job 743 was stopped with `kill -TERM`
    on its parent, left a server holding 37067, and job 745 -- a different
    port base entirely -- died at actor1/instance8 because 37067 fell inside
    actor1's 768-port block. This is the likeliest explanation for the
    long-standing "the server just would not start" failures, which never
    reproduced because reproducing them needed a PREVIOUS run to have been
    killed the wrong way.

    `start_new_session=True` above is what makes this necessary as well as
    sufficient: it detaches the child into its own process group so close() can
    signal the whole group, which also means it is no longer killed by signals
    sent to the trainer's group.

    Linux-only, and best-effort: on any other platform, or if ctypes cannot
    reach prctl, the child simply starts as before.
    """
    try:
        import ctypes
        import signal as _signal

        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, _signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass


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
            preexec_fn=_die_with_parent,
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
    """Serialise one action line, refusing anything the server would reject.

    ``LanerlWire.Parse`` marks a line Fatal on an unknown top-level key and
    then executes *nothing* -- both champions lose their orders for that step,
    with only a ``LANERL_CONTROL_BADACTION`` line in the server log to say so.
    From the trainer's side that is invisible: the step still returns an
    observation, the rollout still fills, and the actions simply had no effect.
    So the same rule is enforced here, where it is a stack trace.
    """
    keys = set(action)
    if keys and keys != {"cmd"} and not keys <= _TOP_LEVEL_KEYS:
        raise VecEnvFailure(
            f"action line has top-level key(s) {sorted(keys - _TOP_LEVEL_KEYS)}; "
            f"LanerlWire.Parse accepts only {sorted(_TOP_LEVEL_KEYS)} or a bare "
            f"{RESET_ACTION!r}, and would drop the WHOLE line -- both champions' "
            f"orders -- without failing the step"
        )
    if keys == {"cmd"} and dict(action) != RESET_ACTION:
        raise VecEnvFailure(
            f"{dict(action)!r} is not a command the server knows; the only one is "
            f"{RESET_ACTION!r}"
        )
    line = json.dumps(action, separators=(",", ":"))
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
    #: ``instance -> the observation that ENDED its episode this step``.
    #:
    #: ``VecDriver.step`` resets in the same call it detects the boundary in,
    #: and the reset step's observation then overwrites ``obs[i]`` (and
    #: ``VecLaneEnv.last_obs[i]``) so the caller is never a step behind.  That
    #: made the terminal frame unreachable, and everything that needs it read
    #: the post-reset frame instead: CS@10 came back 0 for every episode of
    #: ``runs/rl-overnight-0911-0608``, and the reward of the final transition
    #: -- the frame the champion dies on, under ``end_on_death`` -- was 0 by
    #: construction.  Empty on any step with no boundary.
    terminal_obs: Dict[int, RawObs] = field(default_factory=dict)

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
        specs: Optional[Sequence[ServerLaunchSpec]] = None,
    ):
        """``specs`` gives each instance its OWN launch spec, instead of ``spec``.

        Needed because everything that distinguishes one scripted opponent
        from another -- ``LANERL_BOT_SEED`` and ``LANERL_BOT_CONFIG`` -- is a
        process environment variable, read once by ``LanerlConfig.FromEnv`` at
        start-up.  With a single shared spec every instance of a run launches
        the *same* bot from the *same* seed (the server's default, 1234:
        ``LanerlConfig.cs:57``), so N parallel envs draw the same reaction
        jitters, the same last-hit coin flips and the same ability rolls, and
        N-fold parallelism buys N copies of one game rather than N samples.
        """
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        self.n = int(n)
        if specs is not None:
            if spec is not None:
                raise ValueError(
                    "pass spec= or specs=, not both: silently preferring one of them is "
                    "how half the instances end up launched from a config nobody chose"
                )
            if len(specs) != self.n:
                raise ValueError(f"got {len(specs)} launch specs for {self.n} instances")
            self.specs: List[ServerLaunchSpec] = list(specs)
        else:
            # Deliberately N references to ONE spec, which is exactly what
            # every instance shared before this: a spec is read-only after
            # construction, and copying it would invite two instances to be
            # "the same" while differing.
            self.specs = [spec or ServerLaunchSpec()] * self.n
        #: The FIRST instance's spec.  Kept because it was the public attribute
        #: before instances could differ; read ``specs[i]`` for anything that
        #: can vary per instance.
        self.spec = self.specs[0]
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
        return ServerInstance(index, ports, self.specs[index], self.log_dir, self.log)

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
        _t0 = time.perf_counter()
        pending = self._send_all(actions)
        _t1 = time.perf_counter()
        result = self._collect(pending)
        _t2 = time.perf_counter()
        if _PhaseTimer.ENABLED:
            _ENV_TIMER.add("env_send", _t1 - _t0)
            _ENV_TIMER.add("env_wait_recv", _t2 - _t1)
        self._restart_dead(result)
        return result

    def reset_episodes(self, indices: Sequence[int]) -> StepResult:
        """In-process episode reset (0.23 ms) on the named instances.

        Every live instance still needs exactly one action line this step --
        the server is blocked reading one -- so instances not in ``indices``
        receive an empty action and keep their orders.

        Known cost, measured rather than guessed: that empty action IS a
        decision for every instance not being reset, and no transition is
        recorded for it, so the reward the collector next attributes to their
        previous action spans two env steps instead of one.  At the production
        shape -- 4 instances, ~18,000 decisions an episode at
        :data:`lanerl_rl.constants.DECISION_HZ` -- that is four extra steps per
        18,000, i.e. 0.02% of transitions, each one an action-repeat of the
        action it is charged to.  Removing it means sending the reset as the
        NEXT step's action line instead of an extra one, which moves the
        recurrent-state reset and the "never a step behind" contract with it;
        not worth that for 0.02%, but it should not be discovered twice.
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


class _PhaseTimer:
    """Per-phase wall clock for one decision, on LANERL_TIME_PHASES=1.

    Three different AGGREGATES have each implied a different bottleneck --
    learner_frac 0.98 said "the learner", 4% GPU and load 1.25/16 said "nothing
    is saturated", and throughput falling as instances rose said "stragglers".
    None of them timed a phase. This does, and it costs a perf_counter call per
    phase when enabled and nothing when not.
    """

    ENABLED = os.environ.get("LANERL_TIME_PHASES") == "1"
    EVERY = int(os.environ.get("LANERL_TIME_PHASES_EVERY", "200"))

    def __init__(self) -> None:
        self.acc: Dict[str, float] = {}
        self.n = 0

    def add(self, name: str, dt: float) -> None:
        self.acc[name] = self.acc.get(name, 0.0) + dt

    def tick(self, n_instances: int) -> None:
        self.n += 1
        if not self.ENABLED or self.n % self.EVERY:
            return
        tot = sum(self.acc.values()) or 1e-9
        parts = "  ".join(
            f"{k}={1000*v/self.n:6.2f}ms({100*v/tot:4.1f}%)"
            for k, v in sorted(self.acc.items(), key=lambda kv: -kv[1])
        )
        log.info("PHASE n=%d instances=%d per-decision %.2fms | %s",
                 self.n, n_instances, 1000 * tot / self.n, parts)


_ENV_TIMER = _PhaseTimer()

_OBS_PROF = None
if os.environ.get("LANERL_PROFILE_OBS") == "1":
    import atexit as _atexit
    import cProfile as _cProfile
    import pstats as _pstats

    _OBS_PROF = _cProfile.Profile()

    def _dump_obs_profile():
        import io as _io
        buf = _io.StringIO()
        st = _pstats.Stats(_OBS_PROF, stream=buf).sort_stats("tottime")
        st.print_stats(30)
        for line in buf.getvalue().splitlines():
            log.info("OBSPROF %s", line)

    _atexit.register(_dump_obs_profile)


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
        #: Per-instance counts of the two ways _scatter drops an order on the
        #: floor WITHOUT SENDING ANYTHING. A dropped line is not a no-op on
        #: the server: LanerlControl.ApplyActions skips a null side entirely,
        #: so the champion keeps whatever order it last had -- and if it never
        #: had one, it stands in the fountain for the whole game at level 1,
        #: full hp, which is exactly what three of four anchor instances did
        #: in rl-screen-bc-0914c while nothing logged a thing.
        self.skipped_no_obs: List[int] = [0] * env.n
        self.skipped_not_alive: List[int] = [0] * env.n
        #: Orders actually written, per instance. The denominator: "0 sent"
        #: and "sent but ignored" are different bugs and used to look alike.
        self.orders_sent: List[int] = [0] * env.n
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
        # _forward times its own two halves (obs_build / policy_forward), so
        # there is nothing to measure around it from out here.
        actions_per_slot = self._forward(deterministic)
        _t1 = time.perf_counter()
        lines = self._scatter(actions_per_slot)
        _t2 = time.perf_counter()
        result = self.env.step(lines)
        _t3 = time.perf_counter()
        self._on_new_observations(result)
        _t4 = time.perf_counter()
        if _PhaseTimer.ENABLED:
            # _forward is observation-build + policy forward; _scatter is action
            # encode; env.step is already split into send and wait-for-reply.
            # build+forward is now split into obs_build / policy_forward
            # by the instrumentation inside _forward itself.
            _ENV_TIMER.add("encode", _t2 - _t1)
            _ENV_TIMER.add("on_new_obs", _t4 - _t3)
            _ENV_TIMER.tick(self.env.n)
        dones = self._episode_boundaries(result)
        if dones:
            # Keep the frame the episode ended on before the reset step
            # overwrites it, and show it to the adapters.  Both halves matter:
            # the dict is how a caller reads the final CS off the episode it
            # just finished, and the build() is how anything stateful behind
            # the adapter -- the reward model, above all -- gets to see the
            # transition that ended the game.  Without it the death that ends
            # an episode is worth exactly 0 to the learner, because the only
            # frame it appears on is thrown away here.
            result.terminal_obs = {i: result.obs[i] for i in dones if result.obs[i] is not None}
            for i in sorted(result.terminal_obs):
                for side in SIDES:
                    ad = self.adapters.get((i, side))
                    if ad is not None:
                        ad.build(result.terminal_obs[i], side)
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
                _b0 = time.perf_counter()
                if _OBS_PROF is not None:
                    _OBS_PROF.enable()
                batch.append(self.adapters[(i, side)].build(raw, side))
                if _OBS_PROF is not None:
                    _OBS_PROF.disable()
                if _PhaseTimer.ENABLED:
                    _ENV_TIMER.add("obs_build", time.perf_counter() - _b0)
                resets.append(self._pending_resets[i])
            _f0 = time.perf_counter()
            actions, self.states[key] = self.policies[key].act_batch(
                batch, self.states[key], resets=resets, deterministic=deterministic
            )
            if _PhaseTimer.ENABLED:
                _ENV_TIMER.add("policy_forward", time.perf_counter() - _f0)
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
                self.skipped_not_alive[i] += 1
                continue  # dead slots keep their column but their action is dropped
            raw = self.env.last_obs[i]
            if raw is None:  # pragma: no cover - guarded in _forward
                self.skipped_no_obs[i] += 1
                continue
            encoded = self.encoder.encode(action, raw, side)
            slot = lines[i]
            if slot is None:
                slot = {}
                lines[i] = slot
            slot[side] = encoded
            self.orders_sent[i] += 1
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
