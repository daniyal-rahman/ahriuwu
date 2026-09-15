"""Parsing of what the server emits: CS lines, reset reports, and state JSONL."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

CS_RE = re.compile(
    r"LANERL_CS t=(\d+) name=(\S+) team=(\d+) cs=(\d+) gold=(\d+) lvl=(\d+) "
    r"hp=(\d+)/(\d+) deaths=(\d+)"
)

RESET_RE = re.compile(
    r"ms=([\d.]+) minions=(\d+) missiles=(\d+) champs=(\d+) buildings=(\d+) "
    r"dead_buildings=(\d+) waves=(\d+) t_before=([-\d]+) t_after=([-\d]+)"
)

#: Parsed separately and by NAME, not appended to RESET_RE's fixed run of
#: groups: the server's comment warns that inserting a field between the
#: existing ones makes the whole regex miss and parse_reset_reports return an
#: empty list -- silently, because it `continue`s on a non-match.
RESET_TAIL_RE = re.compile(r"pages=(\d+) no_page=(\d+) items=(\d+)")

BENCH_SUMMARY_RE = re.compile(
    r"LANERL_RESET_BENCH_SUMMARY n=(\d+) median_ms=([\d.]+) mean_ms=([\d.]+) "
    r"min_ms=([\d.]+) max_ms=([\d.]+) played_ms_per_episode=(\d+)"
)


@dataclass
class CsRow:
    t: int
    name: str
    team: int
    cs: int
    gold: int
    lvl: int
    hp: int
    mhp: int
    deaths: int


@dataclass
class ResetReport:
    duration_ms: float
    minions_removed: int
    missiles_removed: int
    champions_reset: int
    buildings_restored: int
    buildings_unrevivable: int
    wave_timers_reset: bool
    t_before: int
    t_after: int
    #: The rune/mastery page counters. pages_missing MUST be 0: a champion
    #: reset without its page plays the rest of the process at base stats
    #: (ad 78.14 -> 57.88, mhp 672 -> 616). The server grew these fields
    #: specifically to make that regression impossible to miss, and this
    #: parser never picked them up -- so the one integration test whose
    #: stated job is "does reset hand back a clean game" could not assert
    #: the single thing the feature exists for. Optional so an older log
    #: still parses.
    pages_restored: int | None = None
    pages_missing: int | None = None
    items_removed: int | None = None


def parse_cs_lines(text: str) -> list[CsRow]:
    out = []
    for m in CS_RE.finditer(text):
        out.append(CsRow(
            t=int(m.group(1)), name=m.group(2), team=int(m.group(3)),
            cs=int(m.group(4)), gold=int(m.group(5)), lvl=int(m.group(6)),
            hp=int(m.group(7)), mhp=int(m.group(8)), deaths=int(m.group(9)),
        ))
    return out


def parse_reset_reports(text: str) -> list[ResetReport]:
    out = []
    for line in text.splitlines():
        if "LANERL_RESET" not in line:
            continue
        m = RESET_RE.search(line)
        if not m:
            continue
        tail = RESET_TAIL_RE.search(line)
        out.append(ResetReport(
            duration_ms=float(m.group(1)), minions_removed=int(m.group(2)),
            missiles_removed=int(m.group(3)), champions_reset=int(m.group(4)),
            buildings_restored=int(m.group(5)), buildings_unrevivable=int(m.group(6)),
            wave_timers_reset=m.group(7) == "1",
            t_before=int(m.group(8)), t_after=int(m.group(9)),
            pages_restored=int(tail.group(1)) if tail else None,
            pages_missing=int(tail.group(2)) if tail else None,
            items_removed=int(tail.group(3)) if tail else None,
        ))
    return out


def parse_bench_summary(text: str) -> dict | None:
    for line in text.splitlines():
        m = BENCH_SUMMARY_RE.search(line)
        if m:
            return {
                "n": int(m.group(1)), "median_ms": float(m.group(2)),
                "mean_ms": float(m.group(3)), "min_ms": float(m.group(4)),
                "max_ms": float(m.group(5)),
                "played_ms_per_episode": int(m.group(6)),
            }
    return None


def cs_at(rows: list[CsRow], t_ms: int, name: str) -> CsRow | None:
    """The last CS reading for a champion at or before a game time.

    ONLY WITHIN THE CURRENT EPISODE. An in-process reset rewinds the game
    clock to 0 and the log keeps appending, so a whole-list scan returns the
    highest ``t`` ever written -- which after the first reset is always an
    OLD episode's row, and it is a longer episode with more CS. The reading
    looks entirely plausible and belongs to a different game.

    ``lanerl_train.serverlog.cs_at`` already fixed exactly this; the fix never
    reached this copy, which is a second implementation of the same parse.
    Today's callers happen to use single-episode logs (LANERL_EXIT_AT), so it
    was not biting -- but ``lanerl_bot/tests/test_reset.py`` drives multi-
    episode logs, and this function's whole job is to be the yardstick.
    """
    start = 0
    for i in range(1, len(rows)):
        if rows[i].t < rows[i - 1].t:
            start = i          # a clock that went backwards is a reset
    best = None
    for r in rows[start:]:
        if r.name == name and r.t <= t_ms + 1000:
            best = r
    return best


def load_state_jsonl(path: str | Path) -> list[dict]:
    """The LANERL_RECORD dump: one object per recorded tick."""
    rows = []
    with Path(path).open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # the writer is line-buffered; a killed server can leave a partial tail
                continue
    return rows


def units_of_kind(tick: dict, kind: str) -> list[dict]:
    return [u for u in tick.get("u", []) if u.get("k") == kind]
