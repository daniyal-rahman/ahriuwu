"""Diff two simulation snapshots and say exactly which field moved.

This is the instrument the whole rewrite is measured with, so it is built to
fail loudly in every direction that could make a disagreement look like
agreement.

Recovering entity correspondence
--------------------------------
``LanerlStateDump`` deliberately omits ``NetId`` -- an episode reset renumbers
minions, so an id-keyed comparison would fail for a reason that is not a bug --
and sorts rows by their *content*.  Content-sorting means a row's index moves
when any of its fields moves, so **row i in one trace is not entity i in the
other**.  Correspondence has to be recovered.

We recover it within ``(kind, team)`` groups by nearest position, greedily,
closest pair first.  That is sound precisely when the two simulations are close,
which is the regime a one-step differential is run in: state is injected from
the reference, both step once, and positions cannot have moved more than one
tick of movement.  It is *not* sound deep into a free-running divergence -- so
:func:`diff_snapshots` reports its matching distances, and a Tier-2 harness is
expected to stop trusting the matching once those grow past a champion's
per-tick travel (``345 units/s / 60 Hz`` is under 6 units).

Unmatched entities on either side are reported, never dropped.  An entity that
exists in one simulation and not the other is the single most important thing
this differ can find -- a minion that should have died and did not, a turret
that vanished over a run -- and it is exactly what a naive zip would hide.

What this differ cannot see
---------------------------
The dump was written to catch state leaking across a reset, not to support a
second implementation, so it carries no ``TargetUnit`` -- which looked fatal,
since target selection is what minion aggro and last-hitting turn on.

It is not: :mod:`lanerl_jax.parity.targets` recovers targeting from three
outputs the server already produces, with **no change to the vendored server**
(which is shared with every other worktree).  Champion targeting comes back with
full identity, turret targeting up to (type, team, distance).

What is left, and named in :data:`UNOBSERVABLE` so it appears in every report as
a known blind spot rather than as silent agreement:

* **minion target identity** -- ``LANERL_AGGRO_TRACE`` gives the *kind* switched
  to, the priority on both sides, the hold time and whether it was a call for
  help, but not *which* minion;
* **the auto-attack clock** -- ``_autoAttackCurrentCooldown`` and
  ``HasAutoAttacked``.  ``IsAttacking`` is on the observation wire; the cast
  spell name shows a wind-up is happening but not when the next swing is due;
* **waypoint positions** (only ``Waypoints.Count``) and **buff time remaining**
  (only sorted names).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from .trace import Entity, PosQ, Snapshot, StatQ, Trace

__all__ = [
    "Tolerance",
    "DEFAULT_TOLERANCE",
    "UNOBSERVABLE",
    "FieldDiff",
    "EntityDiff",
    "SnapshotDiff",
    "diff_snapshots",
    "diff_traces",
]

#: Fields parity needs that the server does not currently dump.  Reported in
#: every diff so the blind spot is never mistaken for agreement.
UNOBSERVABLE: Tuple[str, ...] = (
    "minion_target_identity",
    "auto_attack_cooldown",
    "has_auto_attacked",
    "waypoint_positions",
    "buff_time_remaining",
)


#: The entity kinds a lane parity diff is *about*.  Measured from a bot-driven
#: 300 s fixture, a snapshot also carries 46 ``Region``, 24 ``LaneTurret``,
#: 6 ``Inhibitor``, 4 ``LevelProp`` and 2 ``Nexus`` objects, nearly all of them
#: static and irrelevant to a top-lane 1v1.  Diffing them unscoped buries the
#: handful of rows that matter under ~80 that never change.
#:
#: Scoping is a reporting choice, never a correctness one: :func:`diff_snapshots`
#: still *counts* what it skipped, so a turret quietly vanishing over a run --
#: which has happened here -- cannot hide behind the filter.
LANE_KINDS: Tuple[str, ...] = ("Champion", "LaneMinion", "LaneTurret")


@dataclass(slots=True, frozen=True)
class Tolerance:
    """Per-field slack, in the server's own quantised integer units.

    Zero means exact.  The defaults come from
    ``docs/JAX_REWRITE_PLAN.md`` section 3: position to the dump's own
    quantisation (1 unit of PosQ = 1/16 world unit), everything else exact,
    because the quantisation *is* the tolerance the server already chose as
    "any difference that could change behaviour".
    """

    q_pos: int = 1
    q_hp: int = 0
    q_stat: int = 0
    #: how far apart two entities may be and still be considered the same one.
    #: 16 PosQ units = 1 world unit; a champion travels < 6 world units per tick.
    match_radius_q: int = 16 * 8
    #: restrict the field-by-field diff to these kinds; None means everything.
    #: Skipped entities are still counted and reported (see :data:`LANE_KINDS`).
    kinds: Optional[Tuple[str, ...]] = None
    #: field names the sim does not model **yet**. Excluded from the diff and
    #: listed in every report, so "not implemented" never reads as "agrees".
    #: Distinct from :data:`UNOBSERVABLE`, which is what the *oracle* cannot
    #: show us; this is what *we* have not built.
    ignore_fields: Tuple[str, ...] = ()


DEFAULT_TOLERANCE = Tolerance()


@dataclass(slots=True, frozen=True)
class FieldDiff:
    name: str
    left: object
    right: object
    delta: Optional[float] = None

    def __str__(self) -> str:
        d = "" if self.delta is None else f"  (delta {self.delta:+g})"
        return f"{self.name}: {self.left!r} != {self.right!r}{d}"


@dataclass(slots=True)
class EntityDiff:
    """One matched pair that disagreed, or one unmatched entity."""

    kind: str
    team: Optional[int]
    #: quantised distance between the two matched positions; None when unmatched
    match_distance_q: Optional[float]
    fields: List[FieldDiff] = field(default_factory=list)
    only_in: Optional[str] = None       # "left" | "right" when unmatched
    entity: Optional[Entity] = None     # the unmatched one, for the report

    @property
    def is_unmatched(self) -> bool:
        return self.only_in is not None

    def __str__(self) -> str:
        head = f"{self.kind}(team={self.team})"
        if self.is_unmatched:
            e = self.entity
            where = "" if e is None else f" at ({e.x:.1f},{e.y:.1f})"
            return f"{head}{where}: present only in {self.only_in}"
        body = "; ".join(str(f) for f in self.fields)
        return f"{head}: {body}"


@dataclass(slots=True)
class SnapshotDiff:
    t_ms_left: int
    t_ms_right: int
    entity_diffs: List[EntityDiff] = field(default_factory=list)
    hash_equal: Optional[bool] = None
    #: largest position distance among *matched* pairs; the signal that the
    #: greedy matching is about to stop being trustworthy (see module docstring)
    max_match_distance_q: float = 0.0
    time_mismatch: bool = False
    #: entities excluded by ``Tolerance.kinds``, and the kinds they were.
    #: Counted, never silently dropped: a scoped diff that loses track of how
    #: much it is not looking at is how a vanishing turret goes unnoticed.
    skipped: Dict[str, int] = field(default_factory=dict)
    #: field names excluded by ``Tolerance.ignore_fields``, echoed so an
    #: unimplemented mechanic is never mistaken for an agreeing one.
    ignored_fields: Tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.entity_diffs and not self.time_mismatch

    @property
    def unmatched(self) -> List[EntityDiff]:
        return [d for d in self.entity_diffs if d.is_unmatched]

    def report(self, limit: int = 20) -> str:
        skip = ""
        if self.ignored_fields:
            skip += f", {len(self.ignored_fields)} fields not modelled " \
                    f"({', '.join(sorted(self.ignored_fields))})"
        if self.skipped:
            n = sum(self.skipped.values())
            skip = f", {n} entities out of scope ({', '.join(sorted(self.skipped))})"
        if self.clean:
            return (f"t={self.t_ms_left}: clean "
                    f"({len(UNOBSERVABLE)} fields unobservable{skip})")
        lines = [f"t={self.t_ms_left} vs {self.t_ms_right}: "
                 f"{len(self.entity_diffs)} entity diffs "
                 f"({len(self.unmatched)} unmatched), "
                 f"max match distance {self.max_match_distance_q / PosQ:.2f} units{skip}"]
        if self.time_mismatch:
            lines.append("  !! game time differs -- these snapshots are not comparable")
        for d in self.entity_diffs[:limit]:
            lines.append(f"  {d}")
        if len(self.entity_diffs) > limit:
            lines.append(f"  ... {len(self.entity_diffs) - limit} more")
        return "\n".join(lines)


def _match_group(
    left: Sequence[Entity], right: Sequence[Entity], radius_q: int
) -> Tuple[List[Tuple[Entity, Entity, float]], List[Entity], List[Entity]]:
    """Greedy closest-pair matching within one ``(kind, team)`` bucket.

    Greedy rather than optimal (Hungarian) on purpose: in the regime this runs
    in, pairs are within a few units of each other and greedy and optimal agree.
    If they ever stop agreeing, the ``max_match_distance_q`` in the report will
    have grown first, which is the signal to stop trusting the matching at all
    rather than to reach for a better matcher.
    """
    pairs: List[Tuple[float, int, int]] = []
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            d = math.hypot(a.q_x - b.q_x, a.q_y - b.q_y)
            if d <= radius_q:
                pairs.append((d, i, j))
    pairs.sort()

    used_l: set[int] = set()
    used_r: set[int] = set()
    matched: List[Tuple[Entity, Entity, float]] = []
    for d, i, j in pairs:
        if i in used_l or j in used_r:
            continue
        used_l.add(i)
        used_r.add(j)
        matched.append((left[i], right[j], d))

    only_l = [e for i, e in enumerate(left) if i not in used_l]
    only_r = [e for j, e in enumerate(right) if j not in used_r]
    return matched, only_l, only_r


def _compare(a: Entity, b: Entity, tol: Tolerance) -> List[FieldDiff]:
    out: List[FieldDiff] = []
    skip = set(tol.ignore_fields)

    def num(name: str, x, y, slack: int, scale: float) -> None:
        if name in skip:
            return
        if x is None and y is None:
            return
        if x is None or y is None:
            out.append(FieldDiff(name, x, y))
            return
        if abs(x - y) > slack:
            out.append(FieldDiff(name, x / scale, y / scale, (y - x) / scale))

    def exact(name: str, x, y) -> None:
        if name in skip:
            return
        if x != y:
            out.append(FieldDiff(name, x, y))

    num("pos.x", a.q_x, b.q_x, tol.q_pos, PosQ)
    num("pos.y", a.q_y, b.q_y, tol.q_pos, PosQ)
    num("hp", a.q_hp, b.q_hp, tol.q_hp, StatQ)
    num("max_hp", a.q_max_hp, b.q_max_hp, tol.q_stat, StatQ)
    exact("dead", a.dead, b.dead)

    if (a.ai is None) != (b.ai is None) and "ai_block" not in skip:
        out.append(FieldDiff("ai_block", a.ai is not None, b.ai is not None))
    elif a.ai is not None and b.ai is not None:
        exact("move_order", a.ai.move_order, b.ai.move_order)
        exact("waypoints", a.ai.waypoints, b.ai.waypoints)
        exact("cast_spell", a.ai.cast_spell, b.ai.cast_spell)
        exact("channel_spell", a.ai.channel_spell, b.ai.channel_spell)
        exact("can_move", a.ai.can_move, b.ai.can_move)
        if a.ai.buffs != b.ai.buffs and "buffs" not in skip:
            la, lb = set(a.ai.buffs), set(b.ai.buffs)
            out.append(FieldDiff(
                "buffs",
                "+".join(sorted(la - lb)) or "-",
                "+".join(sorted(lb - la)) or "-",
            ))

    if (a.champ is None) != (b.champ is None) and "champion_block" not in skip:
        out.append(FieldDiff("champion_block", a.champ is not None,
                             b.champ is not None))
    elif a.champ is not None and b.champ is not None:
        ca, cb = a.champ, b.champ
        num("ad", ca.q_ad, cb.q_ad, tol.q_stat, StatQ)
        num("armor", ca.q_armor, cb.q_armor, tol.q_stat, StatQ)
        num("mr", ca.q_mr, cb.q_mr, tol.q_stat, StatQ)
        num("move_speed", ca.q_move_speed, cb.q_move_speed, tol.q_stat, StatQ)
        num("attack_speed", ca.q_attack_speed_mult, cb.q_attack_speed_mult, tol.q_stat, StatQ)
        num("gold", ca.q_gold, cb.q_gold, tol.q_stat, StatQ)
        exact("level", ca.level, cb.level)
        exact("cs", ca.minions_killed, cb.minions_killed)
        exact("deaths", ca.deaths, cb.deaths)
        exact("skill_points", ca.skill_points, cb.skill_points)
        for slot, (sa, sb) in enumerate(zip(ca.spells, cb.spells)):
            name = "QWER"[slot]
            exact(f"spell.{name}.level", sa[0], sb[0])
            num(f"spell.{name}.cooldown", sa[1], sb[1], tol.q_stat, StatQ)
    return out


def diff_snapshots(
    left: Snapshot, right: Snapshot, tol: Tolerance = DEFAULT_TOLERANCE
) -> SnapshotDiff:
    """Compare two snapshots field by field, recovering entity correspondence."""
    out = SnapshotDiff(t_ms_left=left.t_ms, t_ms_right=right.t_ms,
                       ignored_fields=tuple(tol.ignore_fields))
    out.time_mismatch = left.t_ms != right.t_ms
    if left.state_hash is not None and right.state_hash is not None:
        out.hash_equal = left.state_hash == right.state_hash
        # Equal hashes mean identical states by construction (FNV-1a over the
        # canonical rows), so there is nothing to find. Still fall through when
        # they differ -- the whole job is saying *what* differs.
        if out.hash_equal and not out.time_mismatch:
            return out

    lg, rg = left.by_group(), right.by_group()
    for key in sorted(set(lg) | set(rg)):
        kind, team = key
        if tol.kinds is not None and kind not in tol.kinds:
            n = len(lg.get(key, [])) + len(rg.get(key, []))
            out.skipped[kind] = out.skipped.get(kind, 0) + n
            continue
        matched, only_l, only_r = _match_group(
            lg.get(key, []), rg.get(key, []), tol.match_radius_q
        )
        for a, b, d in matched:
            out.max_match_distance_q = max(out.max_match_distance_q, d)
            fields = _compare(a, b, tol)
            if fields:
                out.entity_diffs.append(
                    EntityDiff(kind=kind, team=None if team < 0 else team,
                               match_distance_q=d, fields=fields)
                )
        for e in only_l:
            out.entity_diffs.append(EntityDiff(kind, None if team < 0 else team,
                                               None, only_in="left", entity=e))
        for e in only_r:
            out.entity_diffs.append(EntityDiff(kind, None if team < 0 else team,
                                               None, only_in="right", entity=e))
    return out


def diff_traces(
    left: Trace, right: Trace, tol: Tolerance = DEFAULT_TOLERANCE
) -> List[SnapshotDiff]:
    """Snapshot-by-snapshot diff, aligned on game time.

    Aligning on ``t_ms`` rather than on index: a dropped snapshot on one side
    would otherwise shift every subsequent comparison by one decision and turn a
    single missing line into a wall of spurious diffs.
    """
    by_t = {s.t_ms: s for s in right}
    out: List[SnapshotDiff] = []
    for a in left:
        b = by_t.get(a.t_ms)
        if b is None:
            d = SnapshotDiff(t_ms_left=a.t_ms, t_ms_right=-1, time_mismatch=True)
            out.append(d)
            continue
        out.append(diff_snapshots(a, b, tol))
    return out
