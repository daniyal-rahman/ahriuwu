"""Incremental parsing of a server instance's stdout.

The control channel carries state, not bookkeeping: CS is not in the observation
(``LanerlControl.BuildObservation`` emits position/hp/gold/xp/level/cooldowns),
but the server already prints it.  So the log is the source for CS@10 -- the one
absolute, non-zero-sum metric self-play cannot fake -- and it is also where the
server says it is dying.

Both matter enough to tail rather than to read at the end:

* ``LANERL_CS t=.. name=.. team=.. cs=.. gold=.. lvl=.. hp=../.. deaths=..``
* ``LANERL_EPISODE n <reset report>``
* ``LANERL_CONTROL error: ..``, ``FATAL``, ``Unhandled exception``

A fatal line that nobody reads is how a run turns into hours of a trainer
blocked on a socket, so :meth:`LogTail.poll` classifies them and the vec runner
escalates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

__all__ = ["CsRow", "LogEvents", "LogTail", "parse_cs_line", "cs_at"]

CS_RE = re.compile(
    r"LANERL_CS t=(\d+) name=(\S+) team=(\d+) cs=(\d+) gold=(\d+) lvl=(\d+) "
    r"hp=(\d+)/(\d+) deaths=(\d+)"
)

#: Substrings that mean the instance is finished, whatever the exit code says.
FATAL_MARKERS = (
    "LANERL_CONTROL error:",
    "Unhandled exception",
    " FATAL ",
)


@dataclass(frozen=True)
class CsRow:
    t_ms: int
    name: str
    team: int
    cs: int
    gold: int
    level: int
    hp: int
    max_hp: int
    deaths: int


@dataclass
class LogEvents:
    """What appeared in the log since the last poll."""

    cs_rows: List[CsRow] = field(default_factory=list)
    episode_lines: List[str] = field(default_factory=list)
    fatal_lines: List[str] = field(default_factory=list)
    control_attached: bool = False
    lines: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:  # pragma: no cover - convenience
        return bool(self.lines)


def parse_cs_line(line: str) -> Optional[CsRow]:
    m = CS_RE.search(line)
    if m is None:
        return None
    return CsRow(
        t_ms=int(m.group(1)),
        name=m.group(2),
        team=int(m.group(3)),
        cs=int(m.group(4)),
        gold=int(m.group(5)),
        level=int(m.group(6)),
        hp=int(m.group(7)),
        max_hp=int(m.group(8)),
        deaths=int(m.group(9)),
    )


def cs_at(rows: List[CsRow], t_ms: int, tolerance_ms: int = 1000) -> Dict[int, CsRow]:
    """Latest row per team at or just before ``t_ms``, keyed by team id.

    Returns an empty dict for a team that has no row yet -- callers must treat a
    missing team as "unknown", never as zero CS.  Reporting a silent 0 for a
    crashed episode would drag the headline metric down without any warning,
    which is precisely the class of bug this project keeps paying for.
    """
    out: Dict[int, CsRow] = {}
    for r in rows:
        if r.t_ms <= t_ms + tolerance_ms:
            prev = out.get(r.team)
            if prev is None or r.t_ms >= prev.t_ms:
                out[r.team] = r
    return out


class LogTail:
    """Follows a log file that another process is appending to.

    Holds no file handle between polls when the file is absent yet, so it can be
    constructed before the server is spawned.  Partial trailing lines are kept
    buffered rather than parsed, so a half-written ``LANERL_CS`` never yields a
    truncated number.
    """

    def __init__(self, path: Path, max_kept_fatal: int = 20):
        self.path = Path(path)
        self.max_kept_fatal = int(max_kept_fatal)
        self._fh = None
        self._pending = ""
        self.fatal_seen: List[str] = []
        self.cs_rows: List[CsRow] = []
        self.attached = False

    def _open(self) -> bool:
        if self._fh is not None:
            return True
        if not self.path.exists():
            return False
        self._fh = self.path.open("r", errors="replace")
        return True

    def rewind(self) -> None:
        """Reopen from the top.  Used when the instance restarts onto a fresh log."""
        self.close()
        self._pending = ""
        self.cs_rows = []
        self.attached = False

    def poll(self) -> LogEvents:
        ev = LogEvents()
        if not self._open():
            return ev
        assert self._fh is not None
        chunk = self._fh.read()
        if not chunk:
            return ev
        self._pending += chunk
        if "\n" not in self._pending:
            return ev
        head, _, self._pending = self._pending.rpartition("\n")
        for line in head.split("\n"):
            if not line:
                continue
            ev.lines.append(line)
            row = parse_cs_line(line)
            if row is not None:
                ev.cs_rows.append(row)
                self.cs_rows.append(row)
            if line.startswith("LANERL_EPISODE"):
                ev.episode_lines.append(line)
            if "LANERL_CONTROL client attached" in line:
                ev.control_attached = True
                self.attached = True
            if any(marker in line for marker in FATAL_MARKERS):
                ev.fatal_lines.append(line)
                if len(self.fatal_seen) < self.max_kept_fatal:
                    self.fatal_seen.append(line)
        return ev

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
