"""Target selection, recovered from outputs the server already produces.

Why this module exists
----------------------
``LanerlStateDump.Describe`` carries no ``TargetUnit``.  Target selection is
what minion aggro and last-hitting turn on, and it has the strictest parity
target in the plan (*exact*) -- so the mechanic we most need to measure looked
unobservable, and the obvious move was to add a field to the C# dump.

It is not needed, and the vendored server is shared with every other worktree
(and had a commit land in it today), so not touching it is worth a little work
on this side.  Three existing, env-gated, behaviour-neutral outputs cover it:

=========================  ==================================================
champion target            the control-channel observation already carries
                           ``tgt`` (the target's NetId), ``atk``
                           (``IsAttacking``) and ``mo`` (``MoveOrder``) per
                           champion -- ``LanerlControl.cs:232-234``.  Every
                           unit's ``id`` is in the same message, so the NetId
                           resolves within the observation.  **Full identity.**
turret target              ``LANERL_TURRET_TRACE=1`` ->
                           ``LANERL_TURRET t= turret= team= target= ttype=
                           tteam= d=`` on change, polled every 250 ms
                           (``LanerlHooks.cs:553``).  Identity up to
                           (type, team, distance), which is near enough to
                           unique when cross-referenced with the same tick's
                           state dump.
minion retarget            ``LANERL_AGGRO_TRACE=1`` ->
                           ``MRT id= lt= from= to= cfh= held= fromprio=
                           toprio=`` (``LaneMinionAI.cs:250``).  ``from``/``to``
                           are ``GetType().Name``, **not identities**.
=========================  ==================================================

So minion-target *identity* is the one thing still missing.  What remains is
still most of what the minion AI's parity turns on: the 250 ms re-evaluation
cadence, the computed priority on both sides of a switch, how long the previous
target was held, and whether the switch came from a call for help.  A JAX sim
that reproduces all four and picks a different same-priority minion is wrong in
a way worth knowing about but not worth blocking J0 for.

Both switches are read once into a ``static readonly`` / field at construction
and used only to guard a ``Console.WriteLine`` (aggro) or a read-only poll
(turret), so neither perturbs the simulation.  Checked, because a "diagnostic"
that changes what it measures is worse than no diagnostic -- and this project
has a commit titled *a diagnostic must not be able to kill the run*.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

__all__ = [
    "TurretTarget",
    "MinionRetarget",
    "ChampionTarget",
    "TargetTraces",
    "parse_target_traces",
    "champion_targets_from_obs",
    "TRACE_ENV",
]

#: Turn these on when recording a fixture.  Both are behaviour-neutral.
TRACE_ENV: Dict[str, str] = {
    "LANERL_AGGRO_TRACE": "1",
    "LANERL_TURRET_TRACE": "1",
}

TURRET_RE = re.compile(
    r"LANERL_TURRET t=(-?\d+) turret=(\S+) team=(\d+) target=(\S+)"
    r"(?: ttype=(\S+) tteam=(\d+) d=(\d+))?"
)
MRT_RE = re.compile(
    r"MRT id=(\d+) lt=(-?\d+) from=(\S+) to=(\S+) cfh=(\d) held=(-?\d+) "
    r"fromprio=(-?\d+) toprio=(-?\d+)"
)


@dataclass(slots=True, frozen=True)
class TurretTarget:
    """One turret target *change*, not a per-tick sample.

    ``TurretTrace`` only prints when the target differs from the previous poll,
    so a parity check reconstructs the held target by carrying the last value
    forward -- and must not mistake "no line" for "no target".
    """

    t_ms: int
    turret: str
    team: int
    target: Optional[str]          # None when the line said ``target=none``
    target_type: Optional[str]
    target_team: Optional[int]
    distance: Optional[int]


@dataclass(slots=True, frozen=True)
class MinionRetarget:
    """One minion target switch.

    ``from_kind``/``to_kind`` are C# type names, not identities -- see the
    module docstring.  ``held_ms`` is how long the previous target survived,
    which is the sharpest single check on the 250 ms re-evaluation cadence.
    """

    net_id: int
    local_time_ms: int
    from_kind: str                 # "none" when it had no target
    to_kind: str
    from_call_for_help: bool
    held_ms: int
    from_priority: int
    to_priority: int


@dataclass(slots=True, frozen=True)
class ChampionTarget:
    """A champion's target at one decision, read off the observation wire."""

    t_ms: int
    team: int
    target_net_id: Optional[int]   # None when ``tgt`` was 0
    is_attacking: bool
    move_order: int
    #: resolved from the same observation's unit list, when the id is present
    target_kind: Optional[str] = None
    target_team: Optional[int] = None


@dataclass(slots=True)
class TargetTraces:
    turret: List[TurretTarget] = field(default_factory=list)
    minion: List[MinionRetarget] = field(default_factory=list)

    def turret_target_at(self, t_ms: int, turret: str) -> Optional[TurretTarget]:
        """The last change at or before ``t_ms`` -- the held target."""
        best: Optional[TurretTarget] = None
        for e in self.turret:
            if e.turret == turret and e.t_ms <= t_ms:
                if best is None or e.t_ms >= best.t_ms:
                    best = e
        return best

    def retargets_between(self, t0_ms: int, t1_ms: int) -> List[MinionRetarget]:
        return [m for m in self.minion if t0_ms <= m.local_time_ms < t1_ms]


def parse_target_traces(lines: Sequence[str] | Iterator[str]) -> TargetTraces:
    """Pull the turret and minion target traces out of a server log."""
    out = TargetTraces()
    for line in lines:
        m = TURRET_RE.search(line)
        if m is not None:
            tgt = m.group(4)
            none = tgt == "none"
            out.turret.append(TurretTarget(
                t_ms=int(m.group(1)),
                turret=m.group(2),
                team=int(m.group(3)),
                target=None if none else tgt,
                target_type=m.group(5),
                target_team=int(m.group(6)) if m.group(6) else None,
                distance=int(m.group(7)) if m.group(7) else None,
            ))
            continue
        m = MRT_RE.search(line)
        if m is not None:
            out.minion.append(MinionRetarget(
                net_id=int(m.group(1)),
                local_time_ms=int(m.group(2)),
                from_kind=m.group(3),
                to_kind=m.group(4),
                from_call_for_help=m.group(5) == "1",
                held_ms=int(m.group(6)),
                from_priority=int(m.group(7)),
                to_priority=int(m.group(8)),
            ))
    return out


def champion_targets_from_obs(obs: dict) -> List[ChampionTarget]:
    """Read every champion's target out of one control-channel observation.

    The NetId in ``tgt`` is resolved against the same observation's unit list.
    An unresolvable id is kept rather than dropped: it means the target is not
    in the observation, which is itself a fact (fog, or a unit that died between
    the target being set and the observation being built).
    """
    units = {u["id"]: u for u in obs.get("u", []) if "id" in u}
    t_ms = int(obs.get("t", -1))
    out: List[ChampionTarget] = []
    for u in obs.get("u", []):
        if u.get("k") != "Champion" or "tgt" not in u:
            continue
        tgt = int(u["tgt"])
        ref = units.get(tgt)
        out.append(ChampionTarget(
            t_ms=t_ms,
            team=int(u["tm"]),
            target_net_id=None if tgt == 0 else tgt,
            is_attacking=bool(u.get("atk", 0)),
            move_order=int(u.get("mo", -1)),
            target_kind=None if ref is None else ref.get("k"),
            target_team=None if ref is None else ref.get("tm"),
        ))
    return out
