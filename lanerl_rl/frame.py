"""Decoding of server state frames, the lane frame, the fog gate, and memory.

Wire format
-----------
``GameServerLib/Lanerl/LanerlControl.cs::BuildObservation`` writes one JSON
object per ``LANERL_STEP_TICKS`` server ticks (4 -> 15 Hz)::

    {"t": <game_time_ms:int>,
     "u": [{"id":<netid>, "k":<GetType().Name>, "tm":<(int)Team>,
            "x":<int>, "y":<int>, "hp":<int>, "mhp":<int>,
            "vb":<0|1>, "vr":<0|1>,
            # champions only:
            "gold":<int>, "xp":<int>, "lvl":<int>,
            "cd0":<ms|-1>, "cd1":..., "cd2":..., "cd3":...}, ...]}

``"vb"`` / ``"vr"``  (per unit)
    ``GameObject.IsVisibleByTeam(TEAM_BLUE / TEAM_PURPLE)`` evaluated
    server-side.  This is the authoritative fog signal and includes the terrain
    line-of-sight test.  The older ``"vis": [<team_id>, ...]`` list form is
    still understood.  When neither is present we fall back to
    :class:`ApproxFogModel`, which reimplements ``ObjectManager.UnitHasVisionOn``
    *minus* ``NavigationGrid.IsAnythingBetween``, because we do not load the
    navgrid.  The fallback is strictly more permissive than the server: it can
    only ever reveal a unit the server would have hidden behind terrain, never
    hide one the server showed.  ``fog_source`` reports which path was taken.

``"cd0".."cd3"``  (per champion)
    Remaining cooldown in **milliseconds**, ``-1`` when the slot has no spell.
    The legacy ``"cd": [q, w, e, r]`` list form, in **seconds**, is also
    understood.  For the agent's OWN champion this is HUD information and is
    used directly.  For the ENEMY it is privileged: the actor only ever sees
    :class:`EnemyAbilityIntel`'s witnessed-cast estimate.

``"rc": <0|1>``  (per champion)
    1 while the champion is channelling the blue pill (``SpellSlotType.
    BluePillSlot``).  Recall is a 0.5 s windup plus an 8 s channel -- about 120
    decisions at 15 Hz -- so "am I recalling right now" is a state, not an
    event, and the environment cannot reconstruct it from the order it issued.
    Absent from older recordings, in which case it decodes to ``None``.

``"sl": [q, w, e, r]``  (per champion, optional)
    Spell ranks.  Absent from the control channel; the standard availability
    rule (Q at 1, W at 2, E at 3, R at 6) is assumed for the agent's own kit.

``"cs": <int>`` (per champion, optional)
    Creep score.  Absent; estimated by :class:`CreepScoreEstimator` from
    enemy/neutral minion deaths inside the champion's attack range.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import constants as C

__all__ = [
    "Unit",
    "Frame",
    "decode_frame",
    "iter_jsonl",
    "rot180_point",
    "MirrorTransform",
    "LaneFrame",
    "LaneTransform",
    "ApproxFogModel",
    "UnitMemory",
    "AgentMemory",
    "EnemyAbilityIntel",
    "CreepScoreEstimator",
    "visible_ids_for",
]


# --------------------------------------------------------------------------
# Frame decoding
# --------------------------------------------------------------------------


@dataclass(slots=True)
class Unit:
    """One attackable unit in one frame, in raw (unmirrored) world space."""

    id: int
    kind: str
    etype: str
    team: int
    x: float
    y: float
    hp: float
    mhp: float
    gold: Optional[float] = None
    xp: Optional[float] = None
    lvl: Optional[int] = None
    cs: Optional[int] = None
    #: Remaining cooldown per spell slot, in SECONDS.  ``None`` for a slot the
    #: champion does not have, ``None`` for the whole tuple when unreported.
    cooldowns: Optional[Tuple[Optional[float], ...]] = None
    spell_levels: Optional[Tuple[int, int, int, int]] = None
    #: Teams (server ids) that can see this unit, or ``None`` when the recorder
    #: did not supply it.
    visible_to: Optional[frozenset] = None
    #: Champions only: is this champion currently channelling its recall?
    #: ``None`` when the frame predates the ``"rc"`` field.  This is HUD
    #: information for the agent's own champion; for the enemy it is privileged
    #: and must not reach the actor path.
    recalling: Optional[bool] = None

    @property
    def alive(self) -> bool:
        return self.hp > 0.0

    @property
    def is_champion(self) -> bool:
        return self.etype == "champion"

    @property
    def affected_by_fow(self) -> bool:
        """Mirrors ``GameObject.IsAffectedByFoW``.

        ``ObjBuilding`` (inhibitor, nexus) and ``BaseTurret`` both override it
        to ``false``; every other ``AttackableUnit`` leaves it ``true``.
        """
        return self.etype not in ("turret", "inhibitor", "nexus")


@dataclass(slots=True)
class Frame:
    t_ms: int
    units: Dict[int, Unit]

    @property
    def t_s(self) -> float:
        return self.t_ms / 1000.0

    def champions(self) -> List[Unit]:
        return [u for u in self.units.values() if u.is_champion]

    def champion_of_team(self, team: int) -> Optional[Unit]:
        for u in self.units.values():
            if u.is_champion and u.team == team:
                return u
        return None


def _as_tuple4(v) -> Optional[Tuple]:
    if v is None:
        return None
    seq = tuple(v)
    if len(seq) != 4:
        raise ValueError(f"expected 4 entries, got {seq!r}")
    return seq


def _decode_cooldowns(ru: dict) -> Optional[Tuple[Optional[float], ...]]:
    """Read ``cd0..cd3`` (ms, -1 = no such spell) or the legacy ``cd`` list (s)."""
    if "cd0" in ru:
        out: List[Optional[float]] = []
        for i in range(4):
            raw = ru.get(f"cd{i}")
            if raw is None:
                out.append(None)
                continue
            ms = float(raw)
            out.append(None if ms < 0.0 else ms / 1000.0)
        return tuple(out)
    legacy = _as_tuple4(ru.get("cd"))
    if legacy is None:
        return None
    return tuple(None if c is None else float(c) for c in legacy)


def _decode_visibility(ru: dict) -> Optional[frozenset]:
    """Read ``vb``/``vr`` flags, or the legacy ``vis`` team list."""
    if "vb" in ru or "vr" in ru:
        teams = set()
        if int(ru.get("vb", 0)):
            teams.add(C.TEAM_BLUE)
        if int(ru.get("vr", 0)):
            teams.add(C.TEAM_RED)
        return frozenset(teams)
    vis = ru.get("vis")
    return None if vis is None else frozenset(int(t) for t in vis)


def decode_frame(raw: dict) -> Frame:
    """Decode one JSONL / control-channel record into a :class:`Frame`."""
    units: Dict[int, Unit] = {}
    for ru in raw["u"]:
        kind = ru["k"]
        etype = C.KIND_TO_TYPE.get(kind, "other")
        units[ru["id"]] = Unit(
            id=int(ru["id"]),
            kind=kind,
            etype=etype,
            team=int(ru["tm"]),
            x=float(ru["x"]),
            y=float(ru["y"]),
            hp=float(ru["hp"]),
            mhp=float(ru["mhp"]),
            gold=None if ru.get("gold") is None else float(ru["gold"]),
            xp=None if ru.get("xp") is None else float(ru["xp"]),
            lvl=None if ru.get("lvl") is None else int(ru["lvl"]),
            cs=None if ru.get("cs") is None else int(ru["cs"]),
            cooldowns=_decode_cooldowns(ru),
            spell_levels=_as_tuple4(ru.get("sl")),
            visible_to=_decode_visibility(ru),
            recalling=None if ru.get("rc") is None else bool(int(ru["rc"])),
        )
    return Frame(t_ms=int(raw["t"]), units=units)


def iter_jsonl(path: str | Path, limit: Optional[int] = None) -> Iterator[Frame]:
    """Stream frames out of a ``LANERL_RECORD`` JSONL dump."""
    n = 0
    with open(path, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield decode_frame(json.loads(line))
            n += 1
            if limit is not None and n >= limit:
                return


# --------------------------------------------------------------------------
# Map symmetry (retained for reasoning and for the negative-control test; the
# observation builder no longer uses it -- see LaneFrame below for why)
# --------------------------------------------------------------------------


def rot180_point(x: float, y: float) -> Tuple[float, float]:
    """180 degree rotation about the map centre.

    Summoner's Rift is *point*-symmetric as a whole::

        (x, y) -> (MIRROR_X - x, MIRROR_Y - y)

    This is the transform that maps the map onto itself.  It is **not** the
    right canonicalisation for a same-lane 1v1: it maps blue's TOP lane onto
    red's BOT lane.  Verified numerically::

        rot180(574, 10220) = (13409, 4227)  ~= red's bot outer turret

    so a rotation-canonicalised red agent would be looking at a different
    corridor from the blue agent, with an oppositely-signed lane normal.  See
    :class:`LaneFrame`.
    """
    return C.MIRROR_X - x, C.MIRROR_Y - y


class MirrorTransform:
    """180 degree rotation of world space for RED, identity for BLUE.

    Kept for map-geometry reasoning and for the negative-control test in
    ``tests/test_mirror.py``.  It is **no longer** the observation
    canonicalisation; :class:`LaneTransform` is.
    """

    __slots__ = ("team", "flip")

    def __init__(self, team: int):
        if team not in (C.TEAM_BLUE, C.TEAM_RED):
            raise ValueError(f"team must be BLUE({C.TEAM_BLUE}) or RED({C.TEAM_RED}), got {team}")
        self.team = team
        self.flip = team == C.TEAM_RED

    def point(self, x: float, y: float) -> Tuple[float, float]:
        if self.flip:
            return rot180_point(x, y)
        return x, y

    def vector(self, dx: float, dy: float) -> Tuple[float, float]:
        if self.flip:
            return -dx, -dy
        return dx, dy

    def points_array(self, xy: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`point` over an ``(N, 2)`` array."""
        if not self.flip:
            return xy
        out = np.empty_like(xy)
        out[:, 0] = C.MIRROR_X - xy[:, 0]
        out[:, 1] = C.MIRROR_Y - xy[:, 1]
        return out


# --------------------------------------------------------------------------
# The lane frame -- the canonicalisation the observation actually uses
# --------------------------------------------------------------------------


class LaneFrame:
    r"""Signed-progress / perpendicular-offset coordinates along one lane.

    Built from two world anchors: the agent's own outer turret and the enemy's.
    ``s`` is progress from mine towards theirs; ``n`` is the perpendicular
    offset.  The normal is flipped, if necessary, so that
    ``handedness_ref_world`` (default: the agent's own nexus) lies at ``n < 0``.

    Why this replaces the 180 degree rotation
    -----------------------------------------
    In a same-lane 1v1 both champions occupy the **same physical corridor**;
    they merely enter it from opposite ends.  Concretely, for the top lane::

        blue: origin (574, 10220)   axis  u = +(3337, 3434)/|.|
        red:  origin (3911, 13654)  axis  u = -(3337, 3434)/|.|

    so red's lane axis is exactly antiparallel to blue's, and the plain
    left-hand normal therefore comes out with the opposite sign for the two
    agents.  The handedness rule fixes that: the top lane hugs the map's
    top-left edge, so *both* nexuses sit on the same side of it (verified:
    blue nexus n = -1.90e6/L, red nexus n = -3.25e7/L in blue's frame), and
    forcing "own nexus at n < 0" makes both agents adopt the **same world
    normal**.  ``n > 0`` then means "towards the outer wall, away from both
    bases" for BLUE and for RED alike -- a physically consistent meaning.

    The resulting map between the two agents' frames is
    ``(s, n) -> (L - s, n)``: a **reflection** across the lane's perpendicular
    bisector, determinant -1.  That is the correct statement of the symmetry
    for a same-lane duel, and it is what ``tests/test_mirror.py`` asserts.

    .. warning::

       **CHIRALITY ASSUMPTION.**  A reflection swaps left and right.  It is a
       valid canonicalisation only for a champion whose kit is chirally
       symmetric.  Garen qualifies exactly: Q is a self-buff, W is a self-shield,
       E is a symmetric AoE spin, R is point-target, and there are no
       skillshots, no directional dashes, and no cone or line abilities.  For
       ANY other champion -- anyone with a skillshot, a directional dash, a
       cone, or a wall -- a reflected policy would aim its abilities into the
       mirror image of where it means to, and this class must be replaced by a
       true rotation plus a lane-specific correction.  The assumption is
       re-stated at the call site in ``obs.ObservationBuilder.__init__``.
    """

    __slots__ = ("origin", "axis", "normal", "length")

    def __init__(
        self,
        own_anchor_world: Sequence[float],
        enemy_anchor_world: Sequence[float],
        handedness_ref_world: Optional[Sequence[float]] = None,
    ):
        ox, oy = float(own_anchor_world[0]), float(own_anchor_world[1])
        ex, ey = float(enemy_anchor_world[0]), float(enemy_anchor_world[1])
        dx, dy = ex - ox, ey - oy
        length = math.hypot(dx, dy)
        if length <= 0.0:
            raise ValueError("lane anchors coincide")
        self.origin = (ox, oy)
        self.length = length
        self.axis = (dx / length, dy / length)
        # Left-hand normal of the axis (rotate axis +90 deg).
        self.normal = (-dy / length, dx / length)
        if handedness_ref_world is not None:
            rx, ry = float(handedness_ref_world[0]), float(handedness_ref_world[1])
            if self.point(rx, ry)[1] > 0.0:
                self.normal = (-self.normal[0], -self.normal[1])

    # -- world -> lane -----------------------------------------------------

    def point(self, x: float, y: float) -> Tuple[float, float]:
        """World position -> lane-local ``(s, n)``, both in **game units**."""
        px, py = x - self.origin[0], y - self.origin[1]
        return (
            px * self.axis[0] + py * self.axis[1],
            px * self.normal[0] + py * self.normal[1],
        )

    def vector(self, dx: float, dy: float) -> Tuple[float, float]:
        """World direction -> lane-local ``(ds, dn)`` (no origin shift)."""
        return (
            dx * self.axis[0] + dy * self.axis[1],
            dx * self.normal[0] + dy * self.normal[1],
        )

    def sn(self, x: float, y: float) -> Tuple[float, float]:
        """``(s, n)`` with ``s`` normalised to lane-lengths, ``n`` in units."""
        s, n = self.point(x, y)
        return s / self.length, n

    def sn_vector(self, dx: float, dy: float) -> Tuple[float, float]:
        """Alias of :meth:`vector`, kept for readability at the call sites."""
        return self.vector(dx, dy)

    # -- lane -> world -----------------------------------------------------

    def to_world_point(self, s: float, n: float) -> Tuple[float, float]:
        return (
            self.origin[0] + s * self.axis[0] + n * self.normal[0],
            self.origin[1] + s * self.axis[1] + n * self.normal[1],
        )

    def to_world_vector(self, ds: float, dn: float) -> Tuple[float, float]:
        return (
            ds * self.axis[0] + dn * self.normal[0],
            ds * self.axis[1] + dn * self.normal[1],
        )


class LaneTransform:
    """The environment-facing view of a :class:`LaneFrame`.

    .. note::

       ``point`` and ``vector`` deliberately run in **opposite directions**,
       because that is what the two call sites in ``env.py`` need and this class
       exists to be a drop-in for the old :class:`MirrorTransform` (which, being
       an involution, hid the distinction):

       * ``point(x, y)``    WORLD position  -> lane-local ``(s, n)``.
         Used by ``LaneEnv._slot_netids_for`` to place the agent.
       * ``vector(ds, dn)`` lane-local direction -> WORLD direction.
         Used by ``env.decode_action`` to turn the policy's ``(move_x, move_z)``
         pair into a world click.

       Use :meth:`to_lane_vector` / :meth:`to_world_point` when you want the
       other direction; they are named unambiguously.
    """

    __slots__ = ("lane", "team")

    def __init__(self, lane: LaneFrame, team: int):
        self.lane = lane
        self.team = int(team)

    def point(self, x: float, y: float) -> Tuple[float, float]:
        return self.lane.point(x, y)

    def vector(self, ds: float, dn: float) -> Tuple[float, float]:
        return self.lane.to_world_vector(ds, dn)

    def to_lane_vector(self, dx: float, dy: float) -> Tuple[float, float]:
        return self.lane.vector(dx, dy)

    def to_world_point(self, s: float, n: float) -> Tuple[float, float]:
        return self.lane.to_world_point(s, n)

    def points_array(self, xy: np.ndarray) -> np.ndarray:
        """Vectorised :meth:`point` over an ``(N, 2)`` array."""
        rel = np.asarray(xy, dtype=np.float64) - np.asarray(self.lane.origin)
        ax = np.asarray(self.lane.axis)
        nz = np.asarray(self.lane.normal)
        return np.stack([rel @ ax, rel @ nz], axis=-1)


# --------------------------------------------------------------------------
# Fog of war
# --------------------------------------------------------------------------

_FOG_WARNED = False


class ApproxFogModel:
    """Fallback fog model, used only when the frame carries no visibility field.

    Reimplements ``ObjectManager.TeamHasVisionOn`` / ``UnitHasVisionOn``:

    * a unit with ``IsAffectedByFoW == false`` (turret, inhibitor, nexus) is
      always visible;
    * a unit on the observing team is always visible;
    * otherwise it is visible iff some *alive* unit of the observing team is
      within that observer's ``VisionRadius``.

    The one omitted term is ``NavigationGrid.IsAnythingBetween`` (terrain
    line-of-sight / brush).  Omitting it can only *add* visibility, never
    remove it, so it never causes the actor to be trained on something the
    server would have hidden behind a wall -- but it does mean brush control is
    not modelled.  ``LanerlControl`` now emits ``vb``/``vr``, so this path
    should only ever be taken by offline replays of the old recorder.
    """

    def __init__(self, warn: bool = True):
        global _FOG_WARNED
        if warn and not _FOG_WARNED:
            _FOG_WARNED = True
            warnings.warn(
                "lanerl_rl: frame has no per-unit visibility field ('vb'/'vr'); "
                "falling back to ApproxFogModel (radius-only, no terrain "
                "line-of-sight).",
                RuntimeWarning,
                stacklevel=3,
            )

    @staticmethod
    def _vision_radius(u: Unit) -> float:
        if u.etype == "champion":
            return C.VISION_RADIUS_CHAMPION
        if u.etype == "minion":
            return C.VISION_RADIUS_MINION
        if u.etype == "turret":
            return C.VISION_RADIUS_TURRET
        if u.etype in ("inhibitor", "nexus"):
            return C.VISION_RADIUS_BUILDING
        return 0.0

    def visible_ids(self, frame: Frame, team: int) -> set:
        """Return the set of unit ids visible to ``team``."""
        units = list(frame.units.values())
        if not units:
            return set()

        visible = set()
        tested: List[Unit] = []
        for u in units:
            if not u.affected_by_fow or u.team == team:
                visible.add(u.id)
            else:
                tested.append(u)
        if not tested:
            return visible

        prov = [u for u in units if u.team == team and u.alive and self._vision_radius(u) > 0.0]
        if not prov:
            return visible

        pxy = np.fromiter(
            (c for u in prov for c in (u.x, u.y)), dtype=np.float64, count=2 * len(prov)
        ).reshape(-1, 2)
        prad2 = np.fromiter(
            (self._vision_radius(u) ** 2 for u in prov), dtype=np.float64, count=len(prov)
        )
        txy = np.fromiter(
            (c for u in tested for c in (u.x, u.y)), dtype=np.float64, count=2 * len(tested)
        ).reshape(-1, 2)

        # (n_tested, n_providers) squared distances against per-provider radii.
        d2 = (txy[:, None, 0] - pxy[None, :, 0]) ** 2 + (txy[:, None, 1] - pxy[None, :, 1]) ** 2
        seen = (d2 < prad2[None, :]).any(axis=1)
        for u, ok in zip(tested, seen):
            if ok:
                visible.add(u.id)
        return visible


def visible_ids_for(frame: Frame, team: int, fog: Optional[ApproxFogModel] = None) -> Tuple[set, str]:
    """Resolve visibility for ``team``; returns ``(ids, source)``.

    ``source`` is ``"server"`` when every unit carried a visibility field and
    ``"approx"`` when we had to fall back.
    """
    units = list(frame.units.values())
    if units and all(u.visible_to is not None for u in units):
        return {u.id for u in units if (team in u.visible_to or not u.affected_by_fow)}, "server"
    model = fog if fog is not None else ApproxFogModel()
    return model.visible_ids(frame, team), "approx"


# --------------------------------------------------------------------------
# Per-agent memory (velocity, hp deltas, staleness, last-known position)
# --------------------------------------------------------------------------


@dataclass(slots=True)
class UnitMemory:
    """What one agent remembers about one unit.

    Only ever fed with observations the agent was actually allowed to make, so
    nothing fogged can leak in through here.
    """

    etype: str
    team: int
    #: (t_ms, x, y, hp_frac) samples, newest last.
    history: Deque[Tuple[float, float, float, float]] = field(default_factory=lambda: deque(maxlen=64))
    last_seen_ms: float = -1e18
    last_x: float = 0.0
    last_y: float = 0.0
    last_hp_frac: float = 0.0
    last_mhp: float = 1.0
    #: Unit-length world heading at the moment of the last sighting, or ``None``
    #: if the unit was stationary / only ever seen once.
    last_heading: Optional[Tuple[float, float]] = None
    ever_seen: bool = False

    def observe(self, t_ms: float, u: Unit) -> None:
        hp_frac = 0.0 if u.mhp <= 0 else max(0.0, min(1.0, u.hp / u.mhp))
        self.history.append((t_ms, u.x, u.y, hp_frac))
        self.last_seen_ms = t_ms
        self.last_x = u.x
        self.last_y = u.y
        self.last_hp_frac = hp_frac
        self.last_mhp = u.mhp
        self.etype = u.etype
        self.team = u.team
        self.ever_seen = True
        vx, vy = self.velocity(t_ms)
        speed = math.hypot(vx, vy)
        if speed > 1e-6:
            self.last_heading = (vx / speed, vy / speed)

    def _sample_at_or_before(self, t_ms: float) -> Optional[Tuple[float, float, float, float]]:
        best = None
        for sample in self.history:
            if sample[0] <= t_ms:
                best = sample
            else:
                break
        return best

    def velocity(self, t_ms: float) -> Tuple[float, float]:
        """Finite-difference velocity in world units/second."""
        if len(self.history) < 2:
            return 0.0, 0.0
        past = self._sample_at_or_before(self.last_seen_ms - C.VEL_WINDOW_MS)
        if past is None:
            past = self.history[0]
        dt = (self.last_seen_ms - past[0]) / 1000.0
        if dt <= 1e-6:
            return 0.0, 0.0
        return (self.last_x - past[1]) / dt, (self.last_y - past[2]) / dt

    def hp_delta(self, window_ms: float) -> float:
        """``hp_frac(now) - hp_frac(now - window)``; 0.0 without enough history."""
        past = self._sample_at_or_before(self.last_seen_ms - window_ms)
        if past is None:
            return 0.0
        return self.last_hp_frac - past[3]

    def age_s(self, t_ms: float) -> float:
        return max(0.0, (t_ms - self.last_seen_ms) / 1000.0)

    def reachability_radius(self, t_ms: float, move_speed: float = C.GAREN_MOVE_SPEED) -> float:
        """How far the unit could have travelled since we last saw it.

        A remembered position is not a position: it is the centre of a disc of
        radius ``age * move_speed``.  Handing the policy the age and the radius
        is what lets it reason "they have been gone 4 s, they could be on top of
        me" instead of trusting a stale dot.
        """
        return self.age_s(t_ms) * float(move_speed)


class EnemyAbilityIntel:
    """Per-enemy-ability memory, from **witnessed casts only**.

    At 15 Hz, Garen's W (20-24 s) is 300-360 decisions.  No BPTT window this
    stack could afford reaches that, so the memory is computed exactly here
    rather than hoped for from a recurrent core.

    How a cast is witnessed
    -----------------------
    :meth:`observe` is called only for an enemy champion that is *visible and
    on screen*.  A cast is recorded when the observed remaining cooldown of a
    slot **rises** by more than :data:`constants.CAST_DETECT_RISE_S` between two
    sightings at most :data:`constants.CAST_OBSERVE_MAX_GAP_MS` apart.  That is
    exactly the event a human sees as a cast animation, and the ONLY thing kept
    is the timestamp -- never the observed cooldown value itself.  Casts made
    out of vision, or across a vision gap, are not observed at all, which is
    correct: you cannot read an enemy's cooldown off a screenshot.

    The remaining-cooldown estimate uses the **rank-1** base cooldown
    (:data:`constants.ENEMY_COOLDOWN_ASSUMED_RANK1`), so nothing here depends on
    the enemy's spell ranks or level.  Exact for Q and E, conservative for W/R.
    """

    __slots__ = ("last_cast_ms", "_prev_cd", "_prev_ms")

    def __init__(self) -> None:
        self.last_cast_ms: List[Optional[float]] = [None, None, None, None]
        self._prev_cd: Optional[Tuple[Optional[float], ...]] = None
        self._prev_ms: float = -1e18

    def reset(self) -> None:
        self.last_cast_ms = [None, None, None, None]
        self._prev_cd = None
        self._prev_ms = -1e18

    def observe(self, t_ms: float, cooldowns: Optional[Sequence[Optional[float]]]) -> None:
        """Record one *witnessed* sighting of the enemy's ability bar."""
        if cooldowns is None:
            self._prev_cd = None
            self._prev_ms = float(t_ms)
            return
        cur = tuple(cooldowns)
        gap = float(t_ms) - self._prev_ms
        if self._prev_cd is not None and 0.0 <= gap <= C.CAST_OBSERVE_MAX_GAP_MS:
            for i in range(min(4, len(cur))):
                a, b = self._prev_cd[i], cur[i]
                if a is None or b is None:
                    continue
                if b - a > C.CAST_DETECT_RISE_S:
                    self.last_cast_ms[i] = float(t_ms)
        self._prev_cd = cur
        self._prev_ms = float(t_ms)

    def note_gap(self) -> None:
        """Called when the enemy is not observable this tick."""
        self._prev_cd = None

    def features(self, t_ms: float) -> Tuple[List[float], List[float], List[float]]:
        """``(time_since_cast, cd_estimate, unknown)``, each 4 floats in [0, 1].

        ``time_since_cast`` saturates at the assumed base cooldown, so 1.0 means
        "long enough ago that it is certainly back up".  ``cd_estimate`` is the
        clamped remaining fraction.  ``unknown`` is 1.0 for a slot whose cast we
        have never seen -- which is genuinely different from "off cooldown", and
        conflating the two is how an agent learns to walk into an unseen W.
        """
        since, est, unknown = [], [], []
        for i in range(4):
            base = C.ENEMY_COOLDOWN_ASSUMED_RANK1[i]
            last = self.last_cast_ms[i]
            if last is None:
                since.append(1.0)
                est.append(0.0)
                unknown.append(1.0)
                continue
            dt = max(0.0, (float(t_ms) - last) / 1000.0)
            since.append(min(dt / base, 1.0))
            est.append(max(0.0, min(1.0, (base - dt) / base)))
            unknown.append(0.0)
        return since, est, unknown


class AgentMemory:
    """All of one agent's unit memories, plus its own KDA / last-seen book."""

    def __init__(self) -> None:
        self.units: Dict[int, UnitMemory] = {}
        self.kills = 0
        self.deaths = 0
        self.assists = 0
        self.enemy_last_seen_ms: float = -1e18
        self.enemy_intel = EnemyAbilityIntel()
        self._enemy_was_alive: Optional[bool] = None
        self._self_was_alive: Optional[bool] = None

    def reset(self) -> None:
        self.__init__()

    def update(
        self,
        frame: Frame,
        visible: set,
        self_id: int,
        enemy_id: Optional[int],
        enemy_on_screen: bool = False,
    ) -> None:
        t = float(frame.t_ms)
        for uid in visible:
            u = frame.units.get(uid)
            if u is None:
                continue
            mem = self.units.get(uid)
            if mem is None:
                mem = UnitMemory(etype=u.etype, team=u.team)
                self.units[uid] = mem
            mem.observe(t, u)

        # Enemy ability bar: only readable while they are visible AND on screen.
        if enemy_id is not None and enemy_id in visible and enemy_on_screen:
            self.enemy_last_seen_ms = t
            enemy_u = frame.units.get(enemy_id)
            self.enemy_intel.observe(t, None if enemy_u is None else enemy_u.cooldowns)
        else:
            if enemy_id is not None and enemy_id in visible:
                self.enemy_last_seen_ms = t
            self.enemy_intel.note_gap()

        # Forget stale entries so memory does not grow without bound.
        drop = [uid for uid, m in self.units.items() if m.age_s(t) > C.FORGET_S]
        for uid in drop:
            del self.units[uid]

        # KDA bookkeeping.  A death is a visible alive->dead transition of a
        # champion.  Both champions are always in the frame, but the enemy's
        # death is only counted when we could see it -- exactly what a player
        # observes (the kill banner aside, which we do not model).
        self_u = frame.units.get(self_id)
        if self_u is not None:
            alive = self_u.alive
            if self._self_was_alive is True and not alive:
                self.deaths += 1
            self._self_was_alive = alive
        if enemy_id is not None:
            enemy_u = frame.units.get(enemy_id)
            if enemy_u is not None:
                alive = enemy_u.alive
                if self._enemy_was_alive is True and not alive and enemy_id in visible:
                    self.kills += 1
                self._enemy_was_alive = alive


class CreepScoreEstimator:
    """Estimates a champion's CS when the recorder does not report it.

    A last hit is credited when an enemy or neutral minion that was alive on
    the previous tick is gone/dead on this one *and* it was inside the
    champion's attack range at its last known position.  This is an estimate;
    prefer a server-side ``cs`` field.

    ``last_hits_this_step`` is the per-tick count, which the reward's
    ``last_hit`` term consumes.
    """

    def __init__(self, champion_team: int, aa_range: float = C.AA_RANGE_GAREN):
        self.team = champion_team
        self.reach = aa_range + C.TARGET_RADIUS["minion"] + 25.0
        self.cs = 0
        self.last_hits_this_step = 0
        self._prev: Dict[int, Tuple[float, float, int]] = {}

    def reset(self) -> None:
        self.cs = 0
        self.last_hits_this_step = 0
        self._prev = {}

    def update(self, frame: Frame, champ: Optional[Unit]) -> int:
        current: Dict[int, Tuple[float, float, int]] = {}
        for u in frame.units.values():
            if u.etype != "minion" or u.team == self.team:
                continue
            if u.alive:
                current[u.id] = (u.x, u.y, u.team)
        got = 0
        if champ is not None and champ.alive:
            for uid, (mx, my, _team) in self._prev.items():
                if uid in current:
                    continue
                if math.hypot(mx - champ.x, my - champ.y) <= self.reach:
                    got += 1
        self.cs += got
        self.last_hits_this_step = got
        self._prev = current
        return self.cs
