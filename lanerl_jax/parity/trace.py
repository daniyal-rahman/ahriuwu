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
    "AIInternal",
    "MissileInternal",
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
INTERNAL_RE = re.compile(r"LANERL_INTERNAL t=(-?\d+) (ai|missile) (.*)$")

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
class AIInternal:
    """Opt-in, behaviour-neutral controller state excluded from the hash."""

    net_id: int
    kind: str
    team: int
    q_x: int
    q_y: int
    x_bits: Optional[int]
    y_bits: Optional[int]
    target_net_id: int
    target_kind: str
    target_team: int
    target_q_x: int
    target_q_y: int
    waypoint_key: int
    waypoints: Tuple[Tuple[int, int], ...]
    #: False for legacy diagnostic lines written before the collision-cache
    #: field existed.  This differs materially from ``coll=none``: absent
    #: means "unknown, retain the temporal proxy"; explicit none means the
    #: server says this unit was absent from the quadtree.
    collision_observed: bool
    collision_q_x: Optional[int]
    collision_q_y: Optional[int]
    collision_x_bits: Optional[int]
    collision_y_bits: Optional[int]
    q_aa_cooldown: int
    aa_state: int
    q_aa_cast: int
    q_aa_delay: int
    q_aa_windup: int
    is_attacking: bool
    has_auto_attacked: bool
    q_ai_timer: Optional[int]
    q_ai_local: Optional[int]
    q_time_since_attack: Optional[int]
    target_priority: Optional[int]
    lane_waypoint_key: Optional[int]
    had_target: Optional[bool]
    ignored: Tuple[Tuple[int, int], ...]
    help: Tuple[Tuple[int, int], ...]
    #: `aacdbits` -- the float32 BIT PATTERN of the UNCLAMPED remaining
    #: auto-attack cooldown, added 2026-09-21 for `AA-004`. `aacd` publishes
    #: `Q(Math.Max(0f, remaining), StatQ)`: the clamp runs before the
    #: quantisation, so a still-positive sub-quantum residue is flattened to a
    #: flat 0 and becomes indistinguishable from a genuinely ready swing. The
    #: quantised `aacdraw` does not fix that either -- rounding destroys the
    #: same information. Only the exact bits can separate "+1.3 us remaining,
    #: server will NOT fire" from "0" and from "-1 tick, already ready".
    #: `None` on every recording made before the field existed.
    aa_cooldown_bits: Optional[int] = None
    #: `aagate` -- every gate on the auto-attack swing path, packed one bit per
    #: CONDITION, added 2026-09-22 for `AA-005`. The residual after `AA-004` is
    #: 50 unit-ticks that were ready by every gate the dump published and still
    #: did not swing; the remaining gates (`SpellToCast`, `MovementParameters`,
    #: the status flags behind `CanAttack()`, `_castingSpell`, `ChannelSpell`,
    #: `_skipNextAutoAttack`) were all unpublished. Three inferred explanations
    #: were refuted by the first direct measurement of them, so this publishes
    #: the gates rather than their shadows. Bit layout: see
    #: `LanerlAim.AutoAttackGateBits`, and `GATE_BITS` below. `None` on every
    #: recording made before the field existed.
    aa_gate_bits: Optional[int] = None
    #: `status` -- the raw `StatusFlags` word. The packed bits above say that a
    #: gate was shut; this says WHICH flag shut it.
    status_flags: Optional[int] = None


#: Sentinel for "this recording predates `aagate=`", distinct from "no gate
#: was shut". See `first_shut_gate`.
GATE_UNRECORDED = "<unrecorded>"

#: `AA-005`: bit -> (name, what being SET means for the swing).
#: Mirrors `LanerlAim.AutoAttackGateBits`; keep the two in step. The bits are
#: CONDITIONS, not outcomes, so the chain can be re-evaluated from them rather
#: than trusting the server's idea of which gate mattered. `blocks` is True when
#: the bit being SET is what stops the swing, False when the bit being CLEAR is.
GATE_BITS: Tuple[Tuple[int, str, bool], ...] = (
    (0, "target_present", False),
    (1, "target_is_enemy", False),
    (2, "move_order_is_cast_spell", True),
    (3, "movement_parameters_set", True),
    (4, "aa_spell_ready", False),
    (5, "can_attack", False),
    (6, "casting_spell_set", True),
    (7, "channel_spell_set", True),
    (8, "skip_next_auto_attack", True),
    (9, "spell_to_cast_set", True),
    (10, "is_attacking", True),
    (11, "has_made_initial_attack", False),
    (12, "target_in_range", False),
    (13, "target_invalid", True),
    (14, "can_move", False),
    (15, "can_change_waypoints", False),
    (16, "cooldown_elapsed", False),
)

#: The gates in the order `ObjAIBase.Update` evaluates them, so "which gate shut
#: FIRST" is a well-defined question. Anything after the first shut gate was
#: never reached and its value is not evidence about the swing.
GATE_CHAIN: Tuple[str, ...] = (
    "target_present",
    "target_invalid",
    "is_attacking",
    "spell_to_cast_set",
    "target_is_enemy",
    "move_order_is_cast_spell",
    "target_in_range",
    "movement_parameters_set",
    "aa_spell_ready",
    "can_attack",
    "cooldown_elapsed",
    "skip_next_auto_attack",
)


def gate_flags(bits: Optional[int]) -> Dict[str, bool]:
    """Unpack an `aagate` word into named booleans (empty when not recorded)."""
    if bits is None or bits < 0:
        return {}
    return {name: bool(bits >> shift & 1) for shift, name, _ in GATE_BITS}


def first_shut_gate(bits: Optional[int]) -> Optional[str]:
    """The FIRST gate on the swing path that was closed, in server order.

    `None` means every gate was open -- i.e. the server should have swung, and a
    row where it did not is a genuine unexplained residual rather than a gate we
    simply could not see. `UNRECORDED` means the opposite: this recording predates
    the field, so nothing is known. Collapsing those two into one `None` is the
    same absent-vs-observed-empty confusion that `collision_observed` exists to
    avoid, and it would score old recordings as clean.
    """
    flags = gate_flags(bits)
    if not flags:
        return GATE_UNRECORDED
    blocks = {name: b for _, name, b in GATE_BITS}
    for name in GATE_CHAIN:
        if name not in flags:
            continue
        if flags[name] == blocks[name]:
            return name
    return None


@dataclass(slots=True, frozen=True)
class MissileInternal:
    """One in-flight targeted missile from the diagnostic stream."""

    net_id: int
    kind: str
    q_x: int
    q_y: int
    x_bits: Optional[int]
    y_bits: Optional[int]
    owner_net_id: int
    target_net_id: int
    q_speed: int
    q_damage: int


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
    #: set by :func:`load_trace_window` on a snapshot whose rows were skipped
    #: at parse time because it fell outside the requested window.  It exists
    #: only to carry ``t_ms`` forward, because `inject.replay_wave_states`
    #: needs an entry per tick and reads nothing else.  It is POISONED rather
    #: than merely empty: an empty snapshot handed to a diff scores every
    #: missing entity as agreement, which is the exact failure `parse_stream`
    #: raises `TraceFormatError` over for a truncated snapshot.  Reading any
    #: entity view of one raises instead.
    placeholder: bool = False
    #: the server's own FNV-1a 64 over the sorted rows, when the HASH line was
    #: present.  Two snapshots with equal hashes are identical by construction,
    #: which makes it a free fast path before any field-by-field work.
    state_hash: Optional[str] = None
    #: the ``n=`` the server reported, cross-checked against the rows parsed.
    expected_n: Optional[int] = None
    ai_internals: List[AIInternal] = field(default_factory=list)
    missile_internals: List[MissileInternal] = field(default_factory=list)

    def _require_rows(self, what: str) -> None:
        if self.placeholder:
            raise TraceFormatError(
                f"t={self.t_ms}: {what} was read from a snapshot parsed as a "
                "window placeholder, so its rows were never read off the log. "
                "It has no entities because none were PARSED, not because the "
                "server reported none -- scoring it would count every entity "
                "as agreeing. Widen load_trace_window's window to cover it.")

    def by_group(self) -> Dict[Tuple[str, int], List[Entity]]:
        self._require_rows("by_group()")
        out: Dict[Tuple[str, int], List[Entity]] = {}
        for e in self.entities:
            out.setdefault(e.group_key(), []).append(e)
        return out

    def champion(self, team: int) -> Optional[Entity]:
        self._require_rows("champion()")
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


def _kv(body: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for token in body.split():
        if "=" not in token:
            raise TraceFormatError(f"internal token should be key=value, got {token!r}")
        key, value = token.split("=", 1)
        out[key] = value
    return out


def _optional_int(token: str, what: str) -> Optional[int]:
    return None if token == "-" else _int(token, what)


def _pairs(token: str, what: str) -> Tuple[Tuple[int, int], ...]:
    if token in ("", "none"):
        return ()
    if token == "-":
        return ()
    pairs = []
    for item in token.split(","):
        bits = item.split(":")
        if len(bits) != 2:
            raise TraceFormatError(f"{what}: expected id:value pairs, got {token!r}")
        pairs.append((_int(bits[0], f"{what} id"), _int(bits[1], f"{what} value")))
    return tuple(pairs)


def _waypoints(token: str) -> Tuple[Tuple[int, int], ...]:
    if token == "none":
        return ()
    out = []
    for item in token.split(";"):
        xy = item.split(",")
        if len(xy) != 2:
            raise TraceFormatError(f"waypoints: expected x,y pairs, got {token!r}")
        out.append((_int(xy[0], "waypoint x"), _int(xy[1], "waypoint y")))
    return tuple(out)


def _collision_position(token: str) -> Tuple[Optional[int], Optional[int]]:
    if token == "none":
        return None, None
    xy = token.split(",")
    if len(xy) != 2:
        raise TraceFormatError(
            f"collision position: expected x,y or none, got {token!r}")
    return (_int(xy[0], "collision x"), _int(xy[1], "collision y"))


def _optional_xy(token: Optional[str], what: str) -> Tuple[Optional[int], Optional[int]]:
    if token is None or token == "none":
        return None, None
    xy = token.split(",")
    if len(xy) != 2:
        raise TraceFormatError(f"{what}: expected x,y or none, got {token!r}")
    return _int(xy[0], f"{what} x"), _int(xy[1], f"{what} y")


def parse_internal(kind: str, body: str) -> AIInternal | MissileInternal:
    values = _kv(body)
    try:
        if kind == "missile":
            return MissileInternal(
                net_id=_int(values["id"], "missile id"), kind=values["kind"],
                q_x=_int(values["x"], "missile x"), q_y=_int(values["y"], "missile y"),
                x_bits=_optional_int(values.get("xbits", "-"), "missile x bits"),
                y_bits=_optional_int(values.get("ybits", "-"), "missile y bits"),
                owner_net_id=_int(values["owner"], "missile owner"),
                target_net_id=_int(values["target"], "missile target"),
                q_speed=_int(values["speed"], "missile speed"),
                q_damage=_int(values["damage"], "missile damage"),
            )
        target = values["target"].split(",")
        if len(target) != 5:
            raise TraceFormatError(f"AI target should have five comma fields, got {values['target']!r}")
        collision_observed = "coll" in values
        collision_q_x, collision_q_y = (
            _collision_position(values["coll"])
            if collision_observed else (None, None))
        collision_x_bits, collision_y_bits = _optional_xy(
            values.get("collbits"), "collision bits")
        return AIInternal(
            net_id=_int(values["id"], "AI id"), kind=values["kind"],
            team=_int(values["team"], "AI team"), q_x=_int(values["x"], "AI x"),
            q_y=_int(values["y"], "AI y"),
            x_bits=_optional_int(values.get("xbits", "-"), "AI x bits"),
            y_bits=_optional_int(values.get("ybits", "-"), "AI y bits"),
            target_net_id=_int(target[0], "target id"),
            target_kind=target[1], target_team=_int(target[2], "target team"),
            target_q_x=_int(target[3], "target x"), target_q_y=_int(target[4], "target y"),
            # These fields were added after the first internal fixtures.  An
            # absent pair means "no diagnostic override" in inject.py, not an
            # observed empty route.
            waypoint_key=_int(values.get("wpkey", "0"), "waypoint key"),
            waypoints=_waypoints(values.get("wps", "none")),
            collision_observed=collision_observed,
            collision_q_x=collision_q_x,
            collision_q_y=collision_q_y,
            collision_x_bits=collision_x_bits,
            collision_y_bits=collision_y_bits,
            q_aa_cooldown=_int(values["aacd"], "AA cooldown"),
            aa_cooldown_bits=_optional_int(
                values.get("aacdbits", "-"), "AA cooldown bits"),
            aa_gate_bits=_optional_int(
                values.get("aagate", "-"), "AA gate bits"),
            status_flags=_optional_int(
                values.get("status", "-"), "status flags"),
            aa_state=_int(values["aastate"], "AA state"),
            q_aa_cast=_int(values["aacast"], "AA cast time"),
            q_aa_delay=_int(values["aadelay"], "AA delay"),
            q_aa_windup=_int(values["aawindup"], "AA windup"),
            is_attacking=values["attacking"] == "1",
            has_auto_attacked=values["hasaa"] == "1",
            q_ai_timer=_optional_int(values["aitimer"], "AI timer"),
            q_ai_local=_optional_int(values["ailocal"], "AI local time"),
            q_time_since_attack=_optional_int(values["aitsa"], "time since attack"),
            target_priority=_optional_int(values["aiprio"], "target priority"),
            lane_waypoint_key=_optional_int(values["aiwp"], "AI waypoint"),
            had_target=(None if values["aihad"] == "-" else values["aihad"] == "1"),
            ignored=_pairs(values["aiignore"], "ignore map"),
            help=_pairs(values["aihelp"], "help map"),
        )
    except KeyError as exc:
        raise TraceFormatError(f"internal {kind} line missing {exc.args[0]!r}: {body!r}") from exc


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

        m = INTERNAL_RE.search(line)
        if m is not None:
            t = int(m.group(1))
            if current is None or t != current.t_ms:
                raise TraceFormatError(
                    f"an INTERNAL row for t={t} has no matching open snapshot")
            value = parse_internal(m.group(2), m.group(3))
            if isinstance(value, AIInternal):
                current.ai_internals.append(value)
            else:
                current.missile_internals.append(value)
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


def load_trace_window(
    path: Path | str,
    from_ms: Optional[int] = None,
    to_ms: Optional[int] = None,
    max_snapshots: Optional[int] = None,
) -> Trace:
    """`load_trace`, but materialising entity rows only inside a window.

    A 600 s dump is ~575 MB of STATEROW, and `load_trace` builds an `Entity`
    for every row in it.  That is where the one-step differential's 26 GB
    goes, and why it OOMs on anything smaller than the compute node -- even
    when the caller asked for a 2,000-pair slice, because ``--max-pairs``
    bounded the WORK and the parse had already happened.  This bounds the
    parse instead: snapshots outside ``[from_ms, to_ms]`` become
    ``placeholder=True`` stubs carrying only ``t_ms``, which is the only field
    `inject.replay_wave_states` reads off them -- so the wave replay, the one
    thing that genuinely needs every tick from ``FIRST_WAVE_MS`` forward,
    stays exact rather than being approximated by an early stop.

    ``max_snapshots`` caps how many snapshots are kept in full even if the
    window is wide, so a caller that only wants N pairs gets O(N) memory
    without having to convert N into a time bound it cannot know in advance.

    One snapshot past ``to_ms`` is always kept in full: every consumer here
    diffs ``snaps[i]`` against ``snaps[i + 1]``, so a window whose last tick
    had no successor would silently drop its own final pair.

    Index alignment with `load_trace` is preserved exactly -- same snapshots,
    same order, same count -- so a windowed trace and a full one are
    interchangeable anywhere that respects the placeholder poison.
    """
    lo = -(2 ** 62) if from_ms is None else int(from_ms)
    hi = 2 ** 62 if to_ms is None else int(to_ms)
    path = Path(path)
    trace = Trace()
    buf: List[str] = []
    cur_t: Optional[int] = None
    keep = False
    kept = 0
    tail_used = False

    def flush() -> None:
        nonlocal kept
        if cur_t is None:
            return
        if keep:
            sub = parse_stream(buf)
            if len(sub.snapshots) != 1:
                raise TraceFormatError(
                    f"t={cur_t}: windowed parse produced "
                    f"{len(sub.snapshots)} snapshots for one STATEHASH")
            trace.snapshots.append(sub.snapshots[0])
            kept += 1
        else:
            trace.snapshots.append(Snapshot(t_ms=cur_t, placeholder=True))
        buf.clear()

    with path.open(errors="replace") as fh:
        for line in fh:
            m = HASH_RE.search(line)
            if m is not None:
                flush()
                cur_t = int(m.group(1))
                keep = lo <= cur_t <= hi
                if not keep and lo <= cur_t and not tail_used and cur_t > hi:
                    # the successor of the window's last tick; see docstring
                    keep = True
                    tail_used = True
                if keep and max_snapshots is not None and kept >= max_snapshots:
                    keep = False
                if keep:
                    buf.append(line)
                continue
            if cur_t is None and ROW_RE.search(line) is not None:
                raise TraceFormatError(
                    "a STATEROW arrived before any STATEHASH line. This log "
                    "uses the hashless STATE_DUMP_FULL form, which a windowed "
                    "parse cannot delimit; use load_trace instead.")
            if keep:
                buf.append(line)
    flush()
    trace.source = path
    return trace
