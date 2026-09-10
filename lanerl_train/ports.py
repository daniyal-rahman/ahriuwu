"""One port block per server instance, proven free before anything is spawned.

Why this is a module and not two lines inline: every instance needs *two* ports
-- the game server's own listener (``--port``) and the RL control channel
(``LANERL_CONTROL_PORT``) -- and a shared default is not a slow degradation, it
is a silent kill.  ``LanerlControl`` calls ``TcpListener.Start()`` in its
constructor; the second instance to reach that line throws, the exception is
caught where the server treats it as fatal, and the process disappears while the
trainer sits blocked on ``accept``.  That has already happened here.

So: allocate a distinct block per instance, and *probe* each port by binding it
before handing it over.  The bind is closed immediately, so a race with an
unrelated process remains possible in principle -- which is why
:class:`ServerInstance` also verifies the connect and fails loudly rather than
retrying forever.
"""

from __future__ import annotations

import errno
import os
import socket
from dataclasses import dataclass
from typing import List, Optional, Sequence

__all__ = ["PortAllocationError", "InstancePorts", "is_port_free", "PortAllocator"]

#: Deliberately clear of every port already used by this repo's scripts
#: (5119 in ``lanerl_bot/bench/run_server.py``, 5810/5910 in
#: ``lanerl_rl/control_smoke.py``, 5901/5902 in ``lanerl_bot/tests/conftest.py``).
DEFAULT_BASE_PORT = 21000
PORTS_PER_INSTANCE = 2
MAX_PORT = 65535


class PortAllocationError(RuntimeError):
    """No free block could be found.  Fatal: never fall back to a shared port."""


@dataclass(frozen=True)
class InstancePorts:
    """The ports belonging to one server instance."""

    index: int
    control: int  # TCP, LANERL_CONTROL_PORT -- the RL action/observation channel
    game: int  # the server's own --port

    def as_env(self) -> dict:
        return {"LANERL_CONTROL_PORT": str(self.control)}


def is_port_free(port: int, host: str = "127.0.0.1") -> bool:
    """True if both a TCP and a UDP bind on ``port`` succeed right now.

    Both protocols are checked because the control channel is TCP while the game
    server's ENet listener is UDP, and a block is only safe if neither half
    collides.  ``SO_REUSEADDR`` is deliberately **not** set: we want to know
    whether a real listener is there, not whether we could share the address.
    """
    for family in (socket.SOCK_STREAM, socket.SOCK_DGRAM):
        s = socket.socket(socket.AF_INET, family)
        try:
            s.bind((host, port))
        except OSError as exc:
            if exc.errno in (errno.EADDRINUSE, errno.EACCES, errno.EADDRNOTAVAIL):
                return False
            raise
        finally:
            s.close()
    return True


class PortAllocator:
    """Hands out disjoint, verified-free port blocks.

    ``base`` defaults to ``LANERL_PORT_BASE`` or :data:`DEFAULT_BASE_PORT`.  Two
    trainers on one node must not share a base; the env var is how you separate
    them, and the free-port probe catches it if you forget.
    """

    def __init__(
        self,
        base: Optional[int] = None,
        stride: int = PORTS_PER_INSTANCE,
        max_scan: int = 4096,
        host: str = "127.0.0.1",
    ):
        if stride < PORTS_PER_INSTANCE:
            raise ValueError(f"stride must be >= {PORTS_PER_INSTANCE}, got {stride}")
        env_base = os.environ.get("LANERL_PORT_BASE")
        self.base = int(base if base is not None else (env_base or DEFAULT_BASE_PORT))
        if not (1024 <= self.base <= MAX_PORT - stride):
            raise ValueError(f"base port {self.base} out of range")
        self.stride = int(stride)
        self.max_scan = int(max_scan)
        self.host = host
        self._taken: set = set()

    def allocate(self, n: int, probe: bool = True) -> List[InstancePorts]:
        """Return ``n`` distinct blocks.  Raises if it cannot find them."""
        if n <= 0:
            raise ValueError(f"n must be positive, got {n}")
        out: List[InstancePorts] = []
        cursor = self.base
        scanned = 0
        while len(out) < n:
            if scanned >= self.max_scan or cursor + self.stride > MAX_PORT:
                raise PortAllocationError(
                    f"could not find {n} free port blocks of {self.stride} starting at "
                    f"{self.base} after scanning {scanned} candidates. Something is already "
                    f"listening on this range -- check for a leftover server "
                    f"(pgrep -af GameServerConsole) or move LANERL_PORT_BASE."
                )
            control, game = cursor, cursor + 1
            cursor += self.stride
            scanned += 1
            if control in self._taken or game in self._taken:
                continue
            if probe and not (
                is_port_free(control, self.host) and is_port_free(game, self.host)
            ):
                continue
            self._taken.update((control, game))
            out.append(InstancePorts(index=len(out), control=control, game=game))
        assert_unique(out)
        return out

    def release(self, ports: Sequence[InstancePorts]) -> None:
        for p in ports:
            self._taken.discard(p.control)
            self._taken.discard(p.game)


def assert_unique(ports: Sequence[InstancePorts]) -> None:
    """Fail loudly if any port is shared between instances.

    Cheap, and it turns the exact failure this module exists to prevent into an
    exception at the call site instead of a hung ``accept``.
    """
    seen: dict = {}
    for p in ports:
        for role, value in (("control", p.control), ("game", p.game)):
            if value in seen:
                raise PortAllocationError(
                    f"port {value} assigned twice: instance {seen[value][0]} "
                    f"({seen[value][1]}) and instance {p.index} ({role}). "
                    f"A shared port kills every instance but the first."
                )
            seen[value] = (p.index, role)
