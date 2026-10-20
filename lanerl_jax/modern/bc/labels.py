"""The replay champion's received actions as the modern policy's actions: (button, screen column, screen row).

The policy clicks a cell of the 96 x 54 grid over the locked camera centred on the champion
(`modern.screen`), in the champion's own lane frame (`modern.actions`). A replay has no clicks, only what the
server received or executed (`replay.Action`). One decision is labelled from the actions received in
``[t, t + 0.1 s)``:

* a movement path with two or more points: **move**, the click on the path's end. A path that ends on an
  enemy unit is a right-click on it, and `modern_orders_from` reads it as an attack, as the game does.
* an executed basic attack on a unit the champion was not already attacking: **move** with the click on the
  target. Repeats on the same target are the standing order, so **noop**.
* a cast in slot 0-3: **q**, **w**, **e**, **r**; slot 4/5: **summoner_d** / **summoner_f** (Flash always D,
  `replay.canonical_summoners`); slot 12 (the trinket): **ward**; slot 13: **recall**. The click is the cast's target unit or point where the replay has
  one; a self cast has no click, and the click loss is masked for it (`click_valid`), even though the trainer
  counts the click heads for every cast (`actions.SCREEN_BUTTONS`). A targeted or aimed cast whose target
  the replay does not give is marked not representable. Item slots (6-11) and other slots have no button
  (the trainer masks buy/sell/use_item): not representable.
* a one-point path at the champion's own position, received while it was still walking an earlier path and
  not followed by an attack within 0.6 s: **stop** (the S key). A heuristic: the replay does not mark the key,
  and most one-point paths are the server holding the champion where it stops to attack, which the attack
  rule excludes. Other one-point paths: noop.
* nothing received: **noop** (the standing order continues, which is what not clicking does).

Several actions in one decision: a cast beats an attack, which beats a stop, which beats a path; among equals
the latest wins. A click beyond the screen is pulled back along the line from the champion to the screen's
edge (`screen_cell`). Attack-move is never labelled: the server cannot tell it from a right-click.
"""
from __future__ import annotations

import math
from typing import Dict, List, NamedTuple

import numpy as np

from ..actions import MODERN_BUTTON_INDEX, screen_usage
from ..screen import CAM_Y, FLOOR_Y, FOV_V_DEG, N_SCREEN_X, N_SCREEN_Y, RESOLUTION, TILT_DEG, target_on_screen
from .replay import DECISION_DT, Action

__all__ = ["table_cast_rules", "CAST_BUTTON", "SELF_CAST_SLOTS", "CastRules", "cast_rules", "Labels", "lane_to_screen", "screen_cell", "label_actions"]

#: Cast slot -> button. Slots not listed (items, unknown) have none.
CAST_BUTTON = {0: "q", 1: "w", 2: "e", 3: "r", 4: "summoner_d", 5: "summoner_f", 12: "ward", 13: "recall"}
#: Casts that take no cursor for the cloned champion (Garen: Q, W, E are self casts; recall).
SELF_CAST_SLOTS = {0, 1, 2, 13}


class CastRules(NamedTuple):
    """How a champion's Q/W/E/R take their target, from the modern kit (``cast_rules``)."""
    self_cast: frozenset       # no cursor (the click loss is masked)
    unit_target: frozenset     # the cursor must be on the target unit; a non-enemy target is not representable


def cast_rules(champion: str) -> CastRules:
    """``CastRules`` of a champion. A registered modern kit: ``UNIT_TARGET_RANGE[s] > 0`` means unit-targeted
    (Garen R, Jax Q); ``POINT_TARGET_RANGE[s] > 0`` a ground cast (Nasus E); the rest cast on the champion itself.
    Any other champion: the champion table's cast types (``self`` / ``unit`` / ``point``, docs/modern/MULTICHAMP.md)."""
    from .. import champions as K
    if champion in K.BY_NAME:
        kit = K.BY_NAME[champion]
        unit = tuple(getattr(kit, "UNIT_TARGET_RANGE", (0.0,) * 4))
        point = tuple(getattr(kit, "POINT_TARGET_RANGE", (0.0,) * 4))
        return CastRules(self_cast=frozenset({s for s in range(4) if unit[s] <= 0 and point[s] <= 0} | {13}),
                         unit_target=frozenset(s for s in range(4) if unit[s] > 0))
    return table_cast_rules(champion)


def table_cast_rules(champion: str) -> CastRules:
    """``CastRules`` from the champion table's cast types (any champion of the build; unknown -> all ground casts)."""
    from ..data import champion_table as CT
    idx = CT.index_of(champion)
    if idx == 0:
        raise ValueError(f"{champion!r} is not in the champion table")
    types = [CT.CAST_TYPES[i] for i in CT.cast_types()[idx]]
    return CastRules(self_cast=frozenset({s for s in range(4) if types[s] == "self"} | {13}),
                     unit_target=frozenset(s for s in range(4) if types[s] == "unit"))
#: Label a cast on the champion itself (target = ``me``: Annie E, Karma E) or one the decoder reads as
#: self-targeted (``Action.mode`` "self", or "unit" with no target: recasts such as Lux E, Zed W, Aurora Q) as
#: its button with no cursor, whatever the champion table says. False reproduces the labels of datasets converted before 2026-10-20 (not representable).
MODE_SELF_CASTS = True

#: Label a cast on an allied unit (a shield, heal or dash target) as its button with the click on that unit,
#: as an enemy target is; False reproduces the labels of datasets converted before 2026-10-20 (not
#: representable). Whether the world can act on an ally click is the world's concern.
ALLY_TARGET_CLICKS = True

_PRIORITY = {"move": 0, "stop": 1, "attack": 2, "cast": 3}


def _rank(kind: str, a) -> tuple:
    """Which of two actions in one decision is kept: casts over attacks over stops over moves, and among casts
    one with a button (``CAST_BUTTON``) over a record on an unknown slot (Gnar's Q arrives with a second record
    on its mega-form slot 47; that record must not displace the Q)."""
    return (_PRIORITY[kind], int(kind == "cast" and a.slot in CAST_BUTTON))
_HOLD_STILL_S = 0.3           # an active move whose champion has not moved for this long is over
_HOLD_STILL_UNITS = 5.0       # per decision (0.1 s): below this the champion is standing
_HOLD_ATTACK_RANGE = 1500.0   # an attack target further than this is no longer being attacked
_STOP_RADIUS = 80.0           # a stop is at the champion (within this), while an earlier path's end is farther
_STOP_NO_ATTACK_S = 0.6       # ... and no attack follows within this
_EDGE = 0.97                  # how far inside the screen's edge a pulled-back click lands


class Labels(NamedTuple):
    button: np.ndarray        # (T,) int32, `actions.MODERN_BUTTONS`
    screen_x: np.ndarray      # (T,) int32
    screen_y: np.ndarray      # (T,) int32
    uses_screen: np.ndarray   # (T,) float32, the trainer's `actions.screen_usage`
    click_valid: np.ndarray   # (T,) bool: the click is a real target (the click loss applies)
    click_on_unit: np.ndarray  # (T,) bool: the click is on a unit (attack, targeted cast), not ground
    representable: np.ndarray  # (T,) bool: the decision's action has a button
    clamped: np.ndarray       # (T,) bool: the click was pulled back onto the screen
    source: np.ndarray        # (T,) index into the action list, -1 none
    held: np.ndarray          # (T,) bool: filled from the order active at the time, not an event
    n_superseded: int
    n_repeat_attacks: int


def lane_to_screen(ds, dn):
    """Ground offset from the champion in its lane frame -> normalised screen point (the inverse of
    `screen.screen_to_lane`)."""
    tilt = math.radians(TILT_DEG)
    ct, st = math.cos(tilt), math.sin(tilt)
    dy = FLOOR_Y - CAM_Y
    tv = math.tan(math.radians(FOV_V_DEG) / 2.0)
    th = tv * RESOLUTION[0] / RESOLUTION[1]
    dz = dn - dy * ct / st
    vy, vz = dy * ct + dz * st, -dy * st + dz * ct
    return 0.5 + ds / (2.0 * th * vz), 0.5 - vy / (2.0 * tv * vz)


def screen_cell(ds: float, dn: float):
    """The cell under a ground offset (along, across the lane), and whether it was pulled back on screen."""
    clamped = not bool(target_on_screen(ds, dn))
    if clamped:
        lo, hi = 0.0, 1.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if bool(target_on_screen(mid * ds, mid * dn)) else (lo, mid)
        ds, dn = _EDGE * lo * ds, _EDGE * lo * dn
    sx, sy = lane_to_screen(ds, dn)
    col = min(max(int(math.floor(sx * N_SCREEN_X)), 0), N_SCREEN_X - 1)
    row = min(max(int(math.floor(sy * N_SCREEN_Y)), 0), N_SCREEN_Y - 1)
    return col, row, clamped


def label_actions(t: np.ndarray, actions: List[Action], me_xy: np.ndarray, unit_xy: Dict[int, np.ndarray],
                  axis: np.ndarray, normal: np.ndarray, enemies: set, summoner_slot=None,
                  hold: bool = True, rules: CastRules | None = None, me: int | None = None) -> Labels:
    """Label the decisions at times `t`. `me_xy` (T, 2) is the champion's position at each decision;
    `unit_xy[id]` (T, 2) for every unit an action can target (NaN where unknown); `axis`/`normal` the
    champion's lane frame; `enemies` the ids of enemy units; `summoner_slot` maps replay cast slots 4/5 to the
    policy's summoner slots (`replay.canonical_summoners`; identity by default). With `hold`, decisions
    between events are labelled with the order still active: a move to the destination the champion is
    walking to, or a click on the unit it is auto-attacking (`held`); they stop when the champion arrives,
    stands still, loses the target, is given a new order, stops, recalls or ults. ``rules`` (``cast_rules``)
    says which ability slots cast on the champion itself and which need the cursor on the target unit
    (default: Garen's); ``me`` is the champion's unit id (a cast on it is a self cast, ``MODE_SELF_CASTS``)."""
    summoner_slot = summoner_slot or {4: 0, 5: 1}
    rules = rules or CastRules(frozenset(SELF_CAST_SLOTS), frozenset({3}))
    n = len(t)
    b = MODERN_BUTTON_INDEX
    button = np.full(n, b["noop"], np.int32)
    sx = np.zeros(n, np.int32)
    sy = np.zeros(n, np.int32)
    representable = np.ones(n, bool)
    click_valid = np.zeros(n, bool)
    on_unit = np.zeros(n, bool)
    clamped = np.zeros(n, bool)
    source = np.full(n, -1, np.int64)

    attack_times = np.asarray([a.t for a in actions if a.kind == "attack"])
    kinds: Dict[int, str] = {}
    chosen: Dict[int, int] = {}
    n_superseded = 0
    walking_to = None
    for i, a in enumerate(actions):
        k = int(math.floor((a.t - t[0]) / DECISION_DT + 1e-9))
        kind = a.kind
        if a.kind == "path" and a.n_waypoints > 1:
            kind, walking_to = "move", a.point
        elif a.kind == "path":
            at_me = 0 <= k < n and math.dist(a.point, me_xy[k]) < _STOP_RADIUS
            walking = walking_to is not None and math.dist(a.point, walking_to) > _STOP_RADIUS
            attacks = np.any((attack_times > a.t) & (attack_times <= a.t + _STOP_NO_ATTACK_S))
            kind, walking_to = ("stop" if at_me and walking and not attacks else None), None
        if kind is None or not 0 <= k < n:
            continue
        kinds[i] = kind
        if k in chosen:
            n_superseded += 1
            if _rank(kind, a) < _rank(kinds[chosen[k]], actions[chosen[k]]):
                continue
        chosen[k] = i

    def click(k, x, y, unit: bool):
        d = np.array([x, y]) - me_xy[k]
        col, row, c = screen_cell(float(d @ axis), float(d @ normal))
        sx[k], sy[k], clamped[k], click_valid[k], on_unit[k] = col, row, c, True, unit

    def unit_point(k, target):
        xy = unit_xy.get(target)
        return None if xy is None or not np.all(np.isfinite(xy[k])) else xy[k]

    attacking = None
    n_repeat = 0
    for k in sorted(chosen):
        a = actions[chosen[k]]
        kind = kinds[chosen[k]]
        if kind == "move":
            click(k, *a.point, unit=False)
            button[k] = b["move"]
            attacking = None
        elif kind == "stop":
            button[k] = b["stop"]
            attacking = None
        elif kind == "attack":
            p = unit_point(k, a.target)
            if a.target == attacking or p is None:
                n_repeat += a.target == attacking
                continue
            click(k, *p, unit=True)
            button[k] = b["move"]
            attacking = a.target
        elif a.slot in CAST_BUTTON:
            slot = 4 + summoner_slot[a.slot] if a.slot in (4, 5) else a.slot
            button[k] = b[CAST_BUTTON[slot]]
            if a.slot in rules.self_cast:
                pass                                   # no cursor; the click loss is masked
            elif a.target is not None and a.target in enemies and unit_point(k, a.target) is not None:
                click(k, *unit_point(k, a.target), unit=True)
            elif MODE_SELF_CASTS and me is not None and a.target == me and a.slot < 4:
                pass                                   # cast on itself: no cursor
            elif ALLY_TARGET_CLICKS and a.target is not None and a.slot < 4 and unit_point(k, a.target) is not None:
                click(k, *unit_point(k, a.target), unit=True)      # an allied target
            elif a.slot in rules.unit_target:
                representable[k] = False               # a unit cast on an ally, a ward or an unseen unit
                button[k] = b["noop"]
            elif a.target is not None and unit_point(k, a.target) is not None and a.slot in (4, 5):
                click(k, *unit_point(k, a.target), unit=True)      # summoners may target any unit
            elif a.point is not None:
                click(k, *a.point, unit=False)
            elif MODE_SELF_CASTS and a.target is None and getattr(a, "mode", None) in ("self", "unit"):
                pass                                   # a recast or self cast: no cursor
            else:
                representable[k] = False
        else:
            representable[k] = False
        source[k] = chosen[k]
    held = np.zeros(n, bool)
    if hold:
        active = None                                       # ("move", point) or ("attack", target)
        still = 0
        for k in range(n):
            moved = k > 0 and math.dist(me_xy[k], me_xy[k - 1]) > _HOLD_STILL_UNITS
            still = 0 if moved else still + 1
            if source[k] >= 0:
                a, kind = actions[source[k]], kinds[source[k]]
                if kind == "move":
                    active = ("move", a.point)
                elif kind == "attack":
                    active = ("attack", a.target) if button[k] == b["move"] else active
                elif kind == "stop" or (kind == "cast" and a.slot in (3, 13)):
                    active = None
                continue
            if active is None:
                continue
            if active[0] == "move":
                if math.dist(active[1], me_xy[k]) < _STOP_RADIUS or still * DECISION_DT >= _HOLD_STILL_S:
                    active = None
                    continue
                click(k, *active[1], unit=False)
            else:
                p = unit_point(k, active[1])
                if p is None or math.dist(p, me_xy[k]) > _HOLD_ATTACK_RANGE:
                    active = None
                    continue
                click(k, *p, unit=True)
            button[k] = b["move"]
            held[k] = True
    uses = np.asarray(screen_usage(button), np.float32)
    return Labels(button, sx, sy, uses, click_valid & representable, on_unit, representable, clamped, source,
                  held, n_superseded, n_repeat)
