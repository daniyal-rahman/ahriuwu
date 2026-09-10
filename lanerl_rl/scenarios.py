"""Synthetic state frames, used by the audit and by the tests.

Everything here builds :class:`~lanerl_rl.frame.Frame` objects that are wire
compatible with what ``Game.cs::LanerlRecord`` writes, so a scenario and a
recorded game go through exactly the same code path.
"""

from __future__ import annotations

import copy
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import constants as C
from .frame import Frame, LaneFrame, Unit, rot180_point

__all__ = [
    "unit",
    "make_frame",
    "rotate_frame",
    "top_lane_lane_frames",
    "reflect_frame_in_lane",
    "top_lane_scenario",
    "top_lane_sequence",
]

_ETYPE_KIND = {
    "champion": "Champion",
    "minion": "LaneMinion",
    "turret": "LaneTurret",
    "inhibitor": "Inhibitor",
    "nexus": "Nexus",
    "other": "Monster",
}


def unit(
    uid: int,
    etype: str,
    team: int,
    x: float,
    y: float,
    hp: float = 100.0,
    mhp: float = 100.0,
    **extra,
) -> Unit:
    return Unit(
        id=uid,
        kind=_ETYPE_KIND[etype],
        etype=etype,
        team=team,
        x=float(x),
        y=float(y),
        hp=float(hp),
        mhp=float(mhp),
        **extra,
    )


def make_frame(t_ms: int, units: Iterable[Unit]) -> Frame:
    return Frame(t_ms=int(t_ms), units={u.id: u for u in units})


def rotate_frame(frame: Frame, swap_teams: bool = True) -> Frame:
    """Rotate a frame 180 degrees about the map centre and swap BLUE/RED.

    This is the exact counterpart of :class:`~lanerl_rl.frame.MirrorTransform`:
    if ``F2 = rotate_frame(F)``, then the RED agent's canonical view of ``F2``
    must be bit-identical to the BLUE agent's canonical view of ``F``.  Unit
    ids are preserved so that both agents' memories key the same way.
    """
    out: Dict[int, Unit] = {}
    for uid, u in frame.units.items():
        v = copy.copy(u)
        v.x, v.y = rot180_point(u.x, u.y)
        if swap_teams:
            if u.team == C.TEAM_BLUE:
                v.team = C.TEAM_RED
            elif u.team == C.TEAM_RED:
                v.team = C.TEAM_BLUE
            if u.visible_to is not None:
                mapping = {C.TEAM_BLUE: C.TEAM_RED, C.TEAM_RED: C.TEAM_BLUE}
                v.visible_to = frozenset(mapping.get(t, t) for t in u.visible_to)
        out[uid] = v
    return Frame(t_ms=frame.t_ms, units=out)


def top_lane_lane_frames(
    lane: str = "top",
) -> Dict[int, LaneFrame]:
    """The default per-team :class:`~lanerl_rl.frame.LaneFrame` pair.

    Same construction ``ObservationBuilder`` uses, exposed so tests and
    scenario helpers can talk about the two agents' frames without building a
    whole observation pipeline.
    """
    if lane != "top":
        raise ValueError("only the top lane is anchored in constants.py")
    out = {}
    for team in (C.TEAM_BLUE, C.TEAM_RED):
        enemy = C.TEAM_RED if team == C.TEAM_BLUE else C.TEAM_BLUE
        out[team] = LaneFrame(
            C.TOP_OUTER_TURRET[team],
            C.TOP_OUTER_TURRET[enemy],
            C.NEXUS_POSITION[team],
        )
    return out


def reflect_frame_in_lane(
    frame: Frame,
    lanes: Optional[Dict[int, LaneFrame]] = None,
    swap_teams: bool = True,
) -> Frame:
    r"""Map a BLUE-side situation onto the equivalent RED-side one.

    This is the counterpart of the observation's canonicalisation, and the
    reason it is a *reflection* rather than a rotation is explained at
    :class:`~lanerl_rl.frame.LaneFrame`: in a same-lane 1v1 both champions
    occupy one corridor and enter it from opposite ends, so the map between
    their lane frames is ``(s, n) -> (L - s, n)``.

    Concretely, every unit is read out in BLUE's lane frame and written back at
    the world point with the *same* ``(s, n)`` in RED's lane frame.  If
    ``F2 = reflect_frame_in_lane(F)`` then RED's observation of ``F2`` must
    equal BLUE's observation of ``F``.  Unit ids are preserved so both agents'
    memories key the same way.

    The world coordinates come back through an irrational basis, so the round
    trip carries ~1e-11 game units of float64 error.  That is roughly eight
    orders of magnitude below a float32 ulp at map scale, so the *observations*
    still come out bit-identical and ``tests/test_mirror.py`` asserts exactly
    that -- but do not rely on the float64 world coordinates matching bitwise
    the way :func:`rotate_frame`'s integer arithmetic does.
    """
    lanes = lanes or top_lane_lane_frames()
    src = lanes[C.TEAM_BLUE]
    dst = lanes[C.TEAM_RED]
    out: Dict[int, Unit] = {}
    for uid, u in frame.units.items():
        v = copy.copy(u)
        s, n = src.point(u.x, u.y)
        v.x, v.y = dst.to_world_point(s, n)
        if swap_teams:
            if u.team == C.TEAM_BLUE:
                v.team = C.TEAM_RED
            elif u.team == C.TEAM_RED:
                v.team = C.TEAM_BLUE
            if u.visible_to is not None:
                mapping = {C.TEAM_BLUE: C.TEAM_RED, C.TEAM_RED: C.TEAM_BLUE}
                v.visible_to = frozenset(mapping.get(t, t) for t in u.visible_to)
        out[uid] = v
    return Frame(t_ms=frame.t_ms, units=out)


def _lerp(a: Sequence[float], b: Sequence[float], t: float) -> Tuple[float, float]:
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def top_lane_scenario(
    t_ms: int = 90_000,
    blue_s: float = 0.35,
    red_s: float = 0.65,
    n_minions: int = 6,
    blue_hp_frac: float = 1.0,
    red_hp_frac: float = 1.0,
    blue_gold: float = 620.0,
    red_gold: float = 640.0,
    blue_lvl: int = 3,
    red_lvl: int = 3,
    jitter: float = 0.0,
    with_visibility: bool = False,
) -> Frame:
    """A plausible top-lane 1v1 tick, in raw world coordinates.

    ``blue_s`` / ``red_s`` are positions along the blue->red top-lane axis in
    [0, 1] (0 = blue outer turret, 1 = red outer turret).  All coordinates are
    rounded to integers, exactly as the server recorder emits them, which keeps
    the 180 degree rotation an exact involution.
    """
    a = C.TOP_OUTER_TURRET[C.TEAM_BLUE]
    b = C.TOP_OUTER_TURRET[C.TEAM_RED]
    units: List[Unit] = []

    def R(p):  # noqa: N802 - integerise like the recorder does
        return (round(p[0]), round(p[1]))

    bx, by = R(_lerp(a, b, blue_s))
    rx, ry = R(_lerp(a, b, red_s))
    bx += round(jitter)
    rx -= round(jitter)

    units.append(
        unit(
            1001, "champion", C.TEAM_BLUE, bx, by,
            hp=round(671 * blue_hp_frac), mhp=671,
            gold=blue_gold, xp=float(blue_lvl * 280), lvl=blue_lvl,
        )
    )
    units.append(
        unit(
            1002, "champion", C.TEAM_RED, rx, ry,
            hp=round(671 * red_hp_frac), mhp=671,
            gold=red_gold, xp=float(red_lvl * 280), lvl=red_lvl,
        )
    )

    # Minion waves meeting near the middle of the lane.
    for i in range(n_minions):
        s = 0.46 + 0.012 * i
        mx, my = R(_lerp(a, b, s))
        units.append(unit(2000 + i, "minion", C.TEAM_BLUE, mx, my + 40 * (i % 3 - 1), hp=455 - 30 * i, mhp=455))
    for i in range(n_minions):
        s = 0.54 - 0.012 * i
        mx, my = R(_lerp(a, b, s))
        units.append(unit(3000 + i, "minion", C.TEAM_RED, mx, my - 40 * (i % 3 - 1), hp=455 - 22 * i, mhp=455))

    # The two outer turrets and both nexuses (turrets and buildings are never
    # fogged: ObjBuilding/BaseTurret override IsAffectedByFoW to false).
    units.append(unit(4001, "turret", C.TEAM_BLUE, a[0], a[1], hp=1550, mhp=1550))
    units.append(unit(4002, "turret", C.TEAM_RED, b[0], b[1], hp=1400, mhp=1550))
    units.append(
        unit(5001, "nexus", C.TEAM_BLUE, *C.NEXUS_POSITION[C.TEAM_BLUE], hp=5500, mhp=5500)
    )
    units.append(
        unit(5002, "nexus", C.TEAM_RED, *C.NEXUS_POSITION[C.TEAM_RED], hp=5500, mhp=5500)
    )

    frame = make_frame(t_ms, units)
    if with_visibility:
        _annotate_visibility(frame)
    return frame


def _annotate_visibility(frame: Frame) -> None:
    """Fill each unit's ``visible_to`` using the same radius rule the server uses."""
    from .frame import ApproxFogModel

    model = ApproxFogModel(warn=False)
    blue = model.visible_ids(frame, C.TEAM_BLUE)
    red = model.visible_ids(frame, C.TEAM_RED)
    for uid, u in frame.units.items():
        teams = set()
        if uid in blue:
            teams.add(C.TEAM_BLUE)
        if uid in red:
            teams.add(C.TEAM_RED)
        u.visible_to = frozenset(teams)


def top_lane_sequence(
    n: int = 30,
    dt_ms: int = 100,
    t0_ms: int = 90_000,
    **kwargs,
) -> List[Frame]:
    """A short trajectory with both champions walking towards each other."""
    frames = []
    for i in range(n):
        blue_s = 0.30 + 0.004 * i
        red_s = 0.70 - 0.004 * i
        frames.append(
            top_lane_scenario(
                t_ms=t0_ms + i * dt_ms,
                blue_s=blue_s,
                red_s=red_s,
                blue_hp_frac=max(0.35, 1.0 - 0.01 * i),
                red_hp_frac=max(0.40, 1.0 - 0.008 * i),
                **kwargs,
            )
        )
    return frames
