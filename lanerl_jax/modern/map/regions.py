"""Map11 region masks from the 26.19 navgrid region bytes (lanes, jungle, river, bases).

Source (CLIENT H): the v7.1 NGRID stores four region bytes per cell
(``ModernMapGrid.regions``); layout and enum names from
FrankTheBoxMonster/LoL-NGRID-converter ``NGridFileReader.cs`` /
``NavGridCell.cs`` @92943ed:

  byte 0  RiverRegionFlags (bitfield: 1 jungle quadrant, 2 baron pit, 16 river, 64 river entrance)
  byte 1  low nibble JungleQuadrant, high nibble **MainRegion** (equality):
          0 Spawn (fountain platform), 1 Base, 2 TopLane, 3 MidLane, 4 BotLane,
          5 TopSideJungle, 6 BotSideJungle, 7 TopSideRiver, 8 BotSideRiver,
          9 TopSideBasePerimeter, 10 BotSideBasePerimeter, 11 TopSideLaneAlcove, 12 BotSideLaneAlcove
  byte 2  low nibble NearestLane (0-5 blue/red top/mid/bot lane, 6-11 neutral zones), high nibble POI
  byte 3  low nibble Ring (0-4 blue spawn..outer->neutral, 5-9 red), high nibble SRX

On the 26.19 grid MainRegion takes exactly the values 0..12 and agrees with
the other layers (walkable cells): jungle 5/6 vs river-byte bit 0, 14041 of
14346 cells; river 7/8 vs bit 64, 4901 of 4946; dragon pit POI centroid
(9870, 4409). Team side comes from the Ring nibble (0-4 Order, 5-9 Chaos).

Definitions used by the world (LANES_TERRAIN.md §3):

* **lane l** (0 bot, 1 mid, 2 top, geometry ids): MainRegion BotLane/MidLane/
  TopLane plus that side's lane alcove (11 top, 12 bot). The lane regions
  stop at both base edges.
* **in the quest lane** (ROLE_QUESTS U-RQ-1, 26.9 "anywhere in the lane,
  outside of your base"): ``lane_of(x, y) == quest lane``; the base is never
  a lane region, so "outside your base" holds by construction.
* **jungle** (Homeguard): MainRegion 5/6. River, base perimeter and alcoves
  are not jungle (INFERRED M: Homeguard "entering the jungle").
* **river** (Waterwalking ``in_river``): MainRegion 7/8, which include the
  dragon/Baron pit floors.
* **base / fountain**: MainRegion 1 (+0) / 0, side from the Ring nibble.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from ..core.types import KIND_INHIBITOR, KIND_MINION, KIND_TURRET
from .lanes import LANE_PATH_LEN, LANE_PATHS

SPAWN, BASE, TOP_LANE, MID_LANE, BOT_LANE, TOP_JUNGLE, BOT_JUNGLE, TOP_RIVER, BOT_RIVER, \
    TOP_BASE_PERIMETER, BOT_BASE_PERIMETER, TOP_ALCOVE, BOT_ALCOVE = range(13)
OUTSIDE = -1
REGION_NAMES = ("spawn", "base", "top_lane", "mid_lane", "bot_lane", "top_jungle", "bot_jungle", "top_river",
                "bot_river", "top_base_perimeter", "bot_base_perimeter", "top_alcove", "bot_alcove")
LANE_BOT, LANE_MID, LANE_TOP = 0, 1, 2
# MainRegion -> geometry lane id (-1 not a lane).
_REGION_LANE = np.full(13, -1, np.int32)
_REGION_LANE[[BOT_LANE, BOT_ALCOVE]] = LANE_BOT
_REGION_LANE[MID_LANE] = LANE_MID
_REGION_LANE[[TOP_LANE, TOP_ALCOVE]] = LANE_TOP


class MapRegions(NamedTuple):
    main: Any               # (Z, X) int32 MainRegion code
    side: Any               # (Z, X) int32 0 Order / 1 Chaos (Ring nibble)
    cell_size: float
    min_x: float
    min_z: float


def build_regions(grid) -> MapRegions:
    """Host builder from a ``data.modern_map.ModernMapGrid`` (fails on unknown codes)."""
    r = np.asarray(grid.regions)
    main = (r[..., 1] >> 4).astype(np.int32)
    ring = (r[..., 3] & 15).astype(np.int32)
    if main.max() > BOT_ALCOVE or ring.max() > 9:
        raise ValueError("navgrid region codes outside the 26.19 MainRegion/Ring enums")
    return MapRegions(jnp.asarray(main), jnp.asarray((ring >= 5).astype(np.int32)), float(grid.cell_size),
                      float(grid.min_bounds[0]), float(grid.min_bounds[2]))


def _cell(regions: MapRegions, x, y):
    x, y = jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.float32)
    ix = jnp.floor((x - regions.min_x) / regions.cell_size).astype(jnp.int32)
    iz = jnp.floor((y - regions.min_z) / regions.cell_size).astype(jnp.int32)
    h, w = regions.main.shape
    inside = (ix >= 0) & (ix < w) & (iz >= 0) & (iz < h) & jnp.isfinite(x) & jnp.isfinite(y)
    return jnp.clip(iz, 0, h - 1), jnp.clip(ix, 0, w - 1), inside


def region_of(x, y, regions: MapRegions):
    """MainRegion code at world XZ points (any shape); ``OUTSIDE`` off the grid."""
    iz, ix, inside = _cell(regions, x, y)
    return jnp.where(inside, regions.main[iz, ix], OUTSIDE).astype(jnp.int32)


def side_of(x, y, regions: MapRegions):
    """0 Order (blue) / 1 Chaos (red) half of the map at the points; -1 off the grid."""
    iz, ix, inside = _cell(regions, x, y)
    return jnp.where(inside, regions.side[iz, ix], -1).astype(jnp.int32)


def lane_of(x, y, regions: MapRegions):
    """Geometry lane id (0 bot, 1 mid, 2 top) whose lane region contains the point, else -1."""
    code = region_of(x, y, regions)
    return jnp.where(code >= 0, jnp.asarray(_REGION_LANE)[jnp.clip(code, 0, 12)], -1).astype(jnp.int32)


def in_quest_lane(x, y, quest_lane, regions: MapRegions):
    """ROLE_QUESTS 1.4 / U-RQ-1: anywhere in the quest lane's region, outside the bases."""
    return lane_of(x, y, regions) == jnp.asarray(quest_lane, jnp.int32)


def in_jungle(x, y, regions: MapRegions):
    code = region_of(x, y, regions)
    return (code == TOP_JUNGLE) | (code == BOT_JUNGLE)


def in_river(x, y, regions: MapRegions):
    code = region_of(x, y, regions)
    return (code == TOP_RIVER) | (code == BOT_RIVER)


def in_base(x, y, team, regions: MapRegions):
    """Inside ``team``'s base (MainRegion Base or Spawn on that team's side)."""
    code = region_of(x, y, regions)
    return ((code == BASE) | (code == SPAWN)) & (side_of(x, y, regions) == jnp.asarray(team, jnp.int32))


def in_spawn_platform(x, y, team, regions: MapRegions):
    """On ``team``'s fountain platform (MainRegion Spawn)."""
    return (region_of(x, y, regions) == SPAWN) & (side_of(x, y, regions) == jnp.asarray(team, jnp.int32))


# --- lane progress and the Homeguard endpoint (ECONOMY_PROGRESSION §11.2) -----
HOMEGUARD_TURRET_MARGIN = 500.0     # endpoint "just before" the outermost living allied turret (WIKI M)
HOMEGUARD_MINION_LEAD = 2000.0      # after 14:00 / a lane turret down: 2000 before the furthest minion (WIKI M)
HOMEGUARD_LATE_S = 840.0


def _paths():
    return jnp.asarray(LANE_PATHS), jnp.asarray(LANE_PATH_LEN)


def lane_progress(x, y, team, lane):
    """Arc length along ``team``'s minion path of ``lane`` (from its own side) of the point's
    projection; broadcasts over matching shapes."""
    paths, _ = _paths()
    team = jnp.clip(jnp.asarray(team, jnp.int32), 0, 1)
    lane = jnp.clip(jnp.asarray(lane, jnp.int32), 0, 2)
    p = paths[team, lane]                                   # (..., L, 2)
    a, b = p[..., :-1, :], p[..., 1:, :]
    seg = b - a
    sl = jnp.sqrt(jnp.sum(seg ** 2, -1))
    cum = jnp.concatenate([jnp.zeros(sl.shape[:-1] + (1,), sl.dtype), jnp.cumsum(sl, -1)[..., :-1]], -1)
    q = jnp.stack([jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.float32)], -1)[..., None, :]
    t = jnp.clip(jnp.sum((q - a) * seg, -1) / jnp.maximum(sl ** 2, 1e-6), 0., 1.)
    d = jnp.sum((a + t[..., None] * seg - q) ** 2, -1)
    d = jnp.where(sl > 1e-6, d, jnp.inf)
    k = jnp.argmin(d, -1)
    cum, sl = jnp.broadcast_to(cum, t.shape), jnp.broadcast_to(sl, t.shape)
    pick = lambda v: jnp.take_along_axis(v, k[..., None], -1)[..., 0]
    return pick(cum) + pick(t) * pick(sl)


def homeguard_endpoint(team, lane, now, units, structure_lane, minion_lane):
    """(C,) lane-progress threshold past which Homeguard ends, for champions of ``team`` (C,)
    in ``lane`` (C,) (ECONOMY_PROGRESSION 11.2.3, WIKI M):

    ``max(progress(outermost living allied lane turret) - 500, progress(own inhibitor))``;
    after 14:00 or once any allied turret of that lane is down, at least
    ``progress(furthest living allied minion of the lane) - 2000``.
    ``units`` is a ``WorldUnits``; ``structure_lane`` (N,) the structure lanes
    (``WorldConfig.unit_lane``); ``minion_lane`` (N,) the minions' lanes (``LaneAIState.lane``).
    """
    team = jnp.asarray(team, jnp.int32)
    lane = jnp.asarray(lane, jnp.int32)
    c = team.shape[0]
    kind, uteam = jnp.asarray(units.kind), jnp.asarray(units.team, jnp.int32)
    x = jnp.broadcast_to(jnp.asarray(units.x, jnp.float32)[None, :], (c, kind.shape[0]))
    y = jnp.broadcast_to(jnp.asarray(units.y, jnp.float32)[None, :], (c, kind.shape[0]))
    prog = lane_progress(x, y, team[:, None], lane[:, None])        # (C, N)
    own = uteam[None, :] == team[:, None]
    slane = jnp.asarray(structure_lane, jnp.int32)[None, :] == lane[:, None]
    lane_turret = own & slane & (kind == KIND_TURRET)[None, :] & (jnp.asarray(units.sub)[None, :] <= 2)
    alive = jnp.asarray(units.alive, bool)[None, :]
    tur = jnp.max(jnp.where(lane_turret & alive, prog, -jnp.inf), axis=1) - HOMEGUARD_TURRET_MARGIN
    inhib = jnp.max(jnp.where(own & slane & (kind == KIND_INHIBITOR)[None, :], prog, -jnp.inf), axis=1)
    end = jnp.maximum(tur, inhib)
    late = (jnp.asarray(now) >= HOMEGUARD_LATE_S) | jnp.any(lane_turret & ~alive, axis=1)
    mins = own & alive & (kind == KIND_MINION)[None, :] & (jnp.asarray(minion_lane, jnp.int32)[None, :] == lane[:, None])
    front = jnp.max(jnp.where(mins, prog, -jnp.inf), axis=1) - HOMEGUARD_MINION_LEAD
    return jnp.where(late, jnp.maximum(end, front), end)


def homeguard_flags(x, y, team, now, units, structure_lane, minion_lane, regions: MapRegions):
    """``(reached_endpoint, in_jungle)`` (C,) for ``EconomyInputs`` (Homeguard 11.2.1).

    ``reached_endpoint``: the champion stands in a lane region and its
    progress along its team's path of that lane is at or past
    ``homeguard_endpoint``. Champions outside lanes (base, river, jungle)
    never reach it; ``in_jungle`` is the jungle mask."""
    lane = lane_of(x, y, regions)
    end = homeguard_endpoint(team, jnp.maximum(lane, 0), now, units, structure_lane, minion_lane)
    prog = lane_progress(x, y, team, jnp.maximum(lane, 0))
    return (lane >= 0) & (prog >= end), in_jungle(x, y, regions)


__all__ = ["SPAWN", "BASE", "TOP_LANE", "MID_LANE", "BOT_LANE", "TOP_JUNGLE", "BOT_JUNGLE", "TOP_RIVER",
           "BOT_RIVER", "TOP_BASE_PERIMETER", "BOT_BASE_PERIMETER", "TOP_ALCOVE", "BOT_ALCOVE", "OUTSIDE",
           "REGION_NAMES", "MapRegions", "build_regions", "region_of", "side_of", "lane_of", "in_quest_lane",
           "in_jungle", "in_river", "in_base", "in_spawn_platform", "lane_progress", "homeguard_endpoint",
           "homeguard_flags"]
