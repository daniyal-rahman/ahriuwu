"""Parse the server's own state dump into a typed trace.

Where the fixture comes from
----------------------------
``GameServerLib/Lanerl/LanerlStateDump.cs`` already emits, **once per server
tick** -- not once per decision::

    LANERL_STATEHASH t=<gametime_ms> n=<entities> h=<fnv1a64 hex>
    LANERL_STATEROW  t=<gametime_ms> <one canonical entity row>     # _FULL=1 only

It was written to catch state leaking across an episode reset.  It happens to
be exactly the fixture a second implementation needs: a canonical, quantised,
complete snapshot of the whole simulation.  So the parity harness does not have
to be built -- it has to be *parsed*.

``Emit`` is called from ``LanerlHooks.OnUpdate`` (``LanerlHooks.cs:124``), which
runs every tick, *after* that tick's actions have been applied.  Measured: 7,202
snapshots over 120,010 ms of game time, i.e. one per 16.67 ms tick.  That is
finer than the decision rate (``LANERL_STEP_TICKS=2`` -> 30 Hz) and it is the
better granularity to diff at: a one-tick differential is the tightest possible
isolation of a logic error from accumulated float drift.

The row grammar, read off ``LanerlStateDump.Describe``
-----------------------------------------------------
Fields are ``|``-separated.  The number of fields identifies the C# type, because
each subclass appends its own block and nothing is optional within a block:

===== ============= =========================================================
parts kind          appended by
===== ============= =========================================================
    2 GameObject    ``Name|x,y``
    5 AttackableUnit  ``+team|x,y|hp/maxhp|A|D``  (the 2-part pos is replaced)
   11 ObjAIBase     ``+moveOrder|waypoints|castSpell|channelSpell|canMove|buffs``
   21 Champion      ``+ad,ar,mr,ms,as|level|gold|cs|deaths|skillPoints|`` 4x ``lvl:cd``
===== ============= =========================================================

Any other count is a dump-format change, and this module raises rather than
guessing.  A parity harness that silently mis-parses its own oracle is worse
than no harness: it produces green diffs over garbage.

Quantisation is the server's, not ours
--------------------------------------
Positions arrive as ``round(v * 16)`` and stats as ``round(v * 1024)``
(``LanerlStateDump.PosQ`` / ``StatQ``), game time as integer milliseconds.  We
keep the **integers** as the wire values and expose floats as a convenience.
Comparing in quantised integer space is the point: it is the tolerance the
server itself chose as "any difference that could change behaviour", so an exact
integer match is a meaningful, sharp assertion rather than a float epsilon
argument.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

__all__ = [
    "PosQ",
    "StatQ",
    "Entity",
    "ChampionBlock",
    "AIBlock",
    "Snapshot",
    "Trace",
    "parse_row",
    "parse_stream",
    "load_trace",
    "TraceFormatError",
]

#: ``LanerlStateDump.PosQ`` -- positions are quantised to 1/16 of a world unit.
PosQ = 16.0
#: ``LanerlStateDump.StatQ`` -- stats are quantised to 1/1024.
StatQ = 1024.0

HASH_RE = re.compile(r"LANERL_STATEHASH t=(-?\d+) n=(\d+) h=([0-9a-f]{16})")
ROW_RE = re.compile(r"LANERL_STATEROW t=(-?\d+) (.*)$")

#: field counts -> kind, from ``Describe``.  See the module docstring.
_PARTS_GAMEOBJECT = 2
_PARTS_ATTACKABLE = 5
_PARTS_AI = 11
_PARTS_CHAMPION = 21


class TraceFormatError(ValueError):
    """The dump did not look like ``LanerlStateDump`` thinks it should.

    Raised, never warned.  If ``Describe`` grows a field, every parity number
    computed after that point is meaningless until this parser is updated, so
    the run must stop.
    """


@dataclass(slots=True, frozen=True)
class AIBlock:
    """The ``ObjAIBase`` tail of a row.

    ``cast_spell`` and ``channel_spell`` are the two fields whose silent
    persistence froze champions for entire runs -- twice.  They are in the dump
    for that reason and they are compared for that reason.
    """

    move_order: int
    waypoints: int          # count only; -1 when Waypoints was null
    cast_spell: str         # "-" when none
    channel_spell: str      # "-" when none
    can_move: bool
    buffs: Tuple[str, ...]  # sorted ordinal by the server; () when none


@dataclass(slots=True, frozen=True)
class ChampionBlock:
    """The ``Champion`` tail of a row.  All stats in StatQ integer units."""

    q_ad: int
    q_armor: int
    q_mr: int
    q_move_speed: int
    q_attack_speed_mult: int
    level: int
    q_gold: int
    minions_killed: int
    deaths: int
    skill_points: int
    #: per slot 0..3 (Q/W/E/R): (spell level, cooldown in StatQ units).
    #: level -1 and cooldown -1 mean the slot holds no spell at all.
    spells: Tuple[Tuple[int, int], ...]


@dataclass(slots=True, frozen=True)
class Entity:
    """One canonical entity row.

    Positions and stats are the server's quantised **integers** (see the module
    docstring).  ``NetId`` is deliberately absent from the dump -- a reset
    renumbers minions, so anything keyed on allocation order would make every
    reset comparison fail for a reason that is not a bug.  That is also why
    entity correspondence between two traces has to be *recovered* rather than
    read off (see :mod:`lanerl_jax.parity.diff`).
    """

    kind: str
    q_x: int
    q_y: int
    team: Optional[int] = None
    q_hp: Optional[int] = None
    q_max_hp: Optional[int] = None
    dead: Optional[bool] = None
    ai: Optional[AIBlock] = None
    champ: Optional[ChampionBlock] = None

    # -- convenience views; the integers above remain the comparison surface --
    @property
    def x(self) -> float:
        return self.q_x / PosQ

    @property
    def y(self) -> float:
        return self.q_y / PosQ

    @property
    def hp(self) -> Optional[float]:
        return None if self.q_hp is None else self.q_hp / StatQ

    @property
    def is_champion(self) -> bool:
        return self.champ is not None

    def group_key(self) -> Tuple[str, int]:
        """The bucket within which two traces' entities may be matched.

        Type and team only.  Position must *not* be in the key: the whole point
        of matching is to compare positions that are allowed to differ slightly.
        """
        return (self.kind, -1 if self.team is None else self.team)


@dataclass(slots=True)
class Snapshot:
    """Every entity at the end of one server tick."""

    t_ms: int
    entities: List[Entity] = field(default_factory=list)
    #: the server's own FNV-1a 64 over the sorted rows, when the HASH line was
    #: present.  Two snapshots with equal hashes are identical by construction,
    #: which makes it a free fast path before any field-by-field work.
    state_hash: Optional[str] = None
    #: the ``n=`` the server reported, cross-checked against the rows parsed.
    expected_n: Optional[int] = None

    def by_group(self) -> Dict[Tuple[str, int], List[Entity]]:
        out: Dict[Tuple[str, int], List[Entity]] = {}
        for e in self.entities:
            out.setdefault(e.group_key(), []).append(e)
        return out

    def champion(self, team: int) -> Optional[Entity]:
        for e in self.entities:
            if e.is_champion and e.team == team:
                return e
        return None


@dataclass(slots=True)
class Trace:
    """A sequence of snapshots, one per server tick, in game-time order."""

    snapshots: List[Snapshot] = field(default_factory=list)
    source: Optional[Path] = None

    def __len__(self) -> int:
        return len(self.snapshots)

    def __iter__(self) -> Iterator[Snapshot]:
        return iter(self.snapshots)

    def __getitem__(self, i: int) -> Snapshot:
        return self.snapshots[i]

    @property
    def hashes(self) -> List[Optional[str]]:
        return [s.state_hash for s in self.snapshots]


def _int(tok: str, what: str) -> int:
    try:
        return int(tok)
    except ValueError as exc:  # pragma: no cover - a format change, not a path
        raise TraceFormatError(f"{what}: expected an integer, got {tok!r}") from exc


def _pos(tok: str) -> Tuple[int, int]:
    parts = tok.split(",")
    if len(parts) != 2:
        raise TraceFormatError(f"position field should be 'x,y', got {tok!r}")
    return _int(parts[0], "position x"), _int(parts[1], "position y")


def parse_row(body: str) -> Entity:
    """One ``LANERL_STATEROW`` body (everything after ``t=<n> ``) -> an Entity."""
    parts = body.split("|")
    n = len(parts)
    kind = parts[0]

    if n == _PARTS_GAMEOBJECT:
        qx, qy = _pos(parts[1])
        return Entity(kind=kind, q_x=qx, q_y=qy)

    if n not in (_PARTS_ATTACKABLE, _PARTS_AI, _PARTS_CHAMPION):
        raise TraceFormatError(
            f"{kind!r} row has {n} fields; LanerlStateDump.Describe emits "
            f"{_PARTS_GAMEOBJECT}, {_PARTS_ATTACKABLE}, {_PARTS_AI} or "
            f"{_PARTS_CHAMPION}. The dump format changed -- update this parser "
            f"before trusting any parity number. Row: {body!r}"
        )

    team = _int(parts[1], "team")
    qx, qy = _pos(parts[2])
    hp_tok = parts[3].split("/")
    if len(hp_tok) != 2:
        raise TraceFormatError(f"health field should be 'hp/maxhp', got {parts[3]!r}")
    q_hp = _int(hp_tok[0], "hp")
    q_max_hp = _int(hp_tok[1], "max hp")
    if parts[4] not in ("A", "D"):
        raise TraceFormatError(f"alive flag should be 'A' or 'D', got {parts[4]!r}")
    dead = parts[4] == "D"

    ai = None
    champ = None
    if n >= _PARTS_AI:
        buffs_tok = parts[10]
        ai = AIBlock(
            move_order=_int(parts[5], "move order"),
            waypoints=_int(parts[6], "waypoint count"),
            cast_spell=parts[7],
            channel_spell=parts[8],
            can_move=parts[9] == "1",
            # "" is the no-buffs case; "?" is Describe's own catch block, which
            # means GetBuffs() threw. Keep it verbatim rather than normalising
            # it away -- a run where the oracle could not read its own buffs is
            # a fact the diff should surface, not hide.
            buffs=tuple(buffs_tok.split("+")) if buffs_tok not in ("", "?") else
                  (("?",) if buffs_tok == "?" else ()),
        )

    if n == _PARTS_CHAMPION:
        stats = parts[11].split(",")
        if len(stats) != 5:
            raise TraceFormatError(
                f"champion stat block should be 'ad,ar,mr,ms,as', got {parts[11]!r}"
            )
        spells = []
        for slot, tok in enumerate(parts[17:21]):
            lv_cd = tok.split(":")
            if len(lv_cd) != 2:
                raise TraceFormatError(
                    f"spell slot {slot} should be 'level:cooldown', got {tok!r}"
                )
            spells.append((_int(lv_cd[0], "spell level"), _int(lv_cd[1], "spell cooldown")))
        champ = ChampionBlock(
            q_ad=_int(stats[0], "ad"),
            q_armor=_int(stats[1], "armor"),
            q_mr=_int(stats[2], "mr"),
            q_move_speed=_int(stats[3], "move speed"),
            q_attack_speed_mult=_int(stats[4], "attack speed multiplier"),
            level=_int(parts[12], "level"),
            q_gold=_int(parts[13], "gold"),
            minions_killed=_int(parts[14], "cs"),
            deaths=_int(parts[15], "deaths"),
            skill_points=_int(parts[16], "skill points"),
            spells=tuple(spells),
        )

    return Entity(
        kind=kind, q_x=qx, q_y=qy, team=team,
        q_hp=q_hp, q_max_hp=q_max_hp, dead=dead, ai=ai, champ=champ,
    )


def parse_stream(lines: Sequence[str] | Iterator[str]) -> Trace:
    """Pull every STATEHASH/STATEROW out of a server log.

    Tolerant of interleaved server chatter (the dump shares stdout with
    ``LANERL_TPS``, ``LANERL_CS``, log4net output and everything else), strict
    about the rows themselves.

    Rows are grouped by their ``t=``.  ``Emit`` writes the hash line first and
    then its rows, all carrying the same ``t``, so a row arriving for a ``t``
    whose snapshot is already closed means the log was interleaved by another
    writer.  That would corrupt a snapshot silently, so it raises.

    The ``n=`` on the hash line is the server's own row count and is checked
    against the rows actually parsed.  Without that check a truncated snapshot
    -- a log rotated mid-``Emit``, a dropped line -- scores its missing entities
    as *agreement*, which is the most dangerous possible failure for a parity
    harness.
    """
    trace = Trace()
    current: Optional[Snapshot] = None
    closed: set[int] = set()

    def close(snap: Snapshot) -> None:
        if snap.expected_n is not None and len(snap.entities) != snap.expected_n:
            raise TraceFormatError(
                f"t={snap.t_ms}: STATEHASH said n={snap.expected_n} entities but "
                f"{len(snap.entities)} STATEROW lines were parsed. Rows were dropped "
                "or interleaved; a diff over a truncated snapshot would silently "
                "score the missing entities as agreement."
            )
        trace.snapshots.append(snap)
        closed.add(snap.t_ms)

    for line in lines:
        m = HASH_RE.search(line)
        if m is not None:
            if current is not None:
                close(current)
            current = Snapshot(
                t_ms=int(m.group(1)),
                state_hash=m.group(3),
                expected_n=int(m.group(2)),
            )
            continue

        m = ROW_RE.search(line)
        if m is None:
            continue
        t = int(m.group(1))
        if current is None or t != current.t_ms:
            if t in closed:
                raise TraceFormatError(
                    f"a STATEROW for t={t} arrived after that snapshot was closed; "
                    "the log is interleaved and snapshots cannot be trusted"
                )
            # STATE_DUMP_FULL without a parseable hash line is still usable.
            if current is not None:
                close(current)
            current = Snapshot(t_ms=t)
        current.entities.append(parse_row(m.group(2)))

    if current is not None:
        close(current)
    return trace


def load_trace(path: Path | str) -> Trace:
    """Parse a server log file into a :class:`Trace`."""
    path = Path(path)
    with path.open(errors="replace") as fh:
        trace = parse_stream(fh)
    trace.source = path
    return trace
