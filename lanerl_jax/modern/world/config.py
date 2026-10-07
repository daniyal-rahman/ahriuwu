"""Host-side build of the 26.19 world: Map11 terrain and routes, structure geometry, champion loadouts and the
fixed unit layout (``Layout``; docs/modern/WORLD_IMPLEMENTATION.md "Layout"). Missing assets raise.

Positions: turrets and barracks from the client ``base_srx.materials.bin`` placements (``geometry.json``),
inhibitors from ``SRUAP_*_Inhibitor_Idle``, fountains from ``__Spawn_T1/T2``. The Nexus has no placement record:
it is the centre of its navgrid structure pad (STRUCTURE cells, LANES_TERRAIN).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from .. import champions as K
from .. import vision as MV
from .. import wards as WD
from ..core import types as W
from ..core.stat_pipeline import champion_base
from ..data import PATCH_DIR
from ..data.navgrid import load_patch_map
from ..data.routes import load_routes
from ..items.inventory import validate_item_loadout
from ..items.loadout import acquirable_rows, validate_rune_page
from ..jungle import camps as J
from ..jungle import objectives as OBJ
from ..lane.ai import AISlots
from ..map import dynamic_terrain as DTR
from ..map import regions as REG
from ..map import rift as RIFT
from ..map.lanes import LANE_NAMES
from ..runes import catalog as RD

DEFAULT_MAP = Path("/mnt/nfs/shared/modern-world-map-research/grid-26.19-base")
DEFAULT_ROUTES = Path("/mnt/nfs/shared/WORLD001_map_routes/routes")
N_CHAMPIONS = 2
N_STRUCTURES = 30                   # 22 turrets, 6 inhibitors, 2 Nexuses
BASE_STRUCTURES = 3                 # per team: 2 Nexus turrets and the Nexus
NEXUS_POSITIONS = ((1549.0, 1658.0), (13240.0, 13235.0))    # navgrid Nexus pad centres (blue, red)
FOUNTAINS = ((394.0, 461.0), (14340.0, 14391.0))            # __Spawn_T1 / __Spawn_T2
INHIBITORS = {  # (team, lane) -> position, from SRUAP_*_Inhibitor_Idle placements
    (0, "top"): (1171.6, 3569.7), (0, "mid"): (3198.6, 3218.5), (0, "bot"): (3456.0, 1200.1),
    (1, "top"): (11256.2, 13675.8), (1, "mid"): (11600.0, 11671.4), (1, "bot"): (13598.7, 11311.9),
}
TIERS = {"outer": 0, "inner": 1, "inhibitor": 2, "nexus": 3}
PREREQ = {"inner": "outer", "inhibitor": "inner", "inhib": "inhibitor"}   # lane chain; Nexus rules live in lane.ai


@dataclass(frozen=True)
class Loadout:
    """One champion's game setup (host-side, validated at build time)."""
    champion: str                       # a kit name in ``champions.KITS`` ("Garen", "Jax")
    items: tuple[int, ...] = ()
    rune_page: Any = None               # runes.catalog.RunePage or None (no-runes ruleset)
    summoners: tuple[int, int] = (4, 12)  # Flash, Teleport
    role: int = 1                       # role_quest.ROLE_TOP
    skill_order: tuple[int, ...] = ()   # slot per level 1..18; empty = champion default
    auto_skill: bool = True             # spend unassigned skill points by skill_order (False: only level_up orders)
    allowed_items: tuple[int, ...] = () # the only items this champion may buy (empty: the whole shop)


@dataclass(frozen=True)
class Layout:
    """Start index of every unit block: champions, a minion block per spawning lane, camps, epics, wards (blue's 8
    then red's 8), structures (last: the unfogged block). Disabled systems get no slots."""
    lanes: tuple = (0, 1, 2)            # lanes whose waves spawn, one minion block each, in this order
    jungle: bool = True
    objectives: bool = True
    packets: int = 0                    # main-pass packet capacity; 0 = two per unit (``packet_capacity``)
    lane_structures: bool = False       # only the spawning lanes' turrets and inhibitors (plus the base)

    @property
    def minion0(self) -> int:
        return N_CHAMPIONS

    @property
    def monster0(self) -> int:
        return self.minion0 + len(self.lanes) * W.MINION_SLOTS_PER_LANE

    @property
    def epic0(self) -> int:
        return self.monster0 + (W.JUNGLE_SLOTS if self.jungle else 0)

    @property
    def ward0(self) -> int:
        return self.epic0 + (W.EPIC_SLOTS if self.objectives else 0)

    @property
    def struct0(self) -> int:
        return self.ward0 + 2 * W.MAX_WARDS_PER_TEAM

    @property
    def n_structures(self) -> int:
        """All 30, or per team 3 turrets and an inhibitor per spawning lane plus the base."""
        return 2 * (4 * len(self.lanes) + BASE_STRUCTURES) if self.lane_structures else N_STRUCTURES

    @property
    def n_units(self) -> int:
        return self.struct0 + self.n_structures

    @property
    def ai_slots(self) -> AISlots:
        """Lane AI rows (minion and structure blocks) and columns (champions, minions, objectives, structures)."""
        n = self.n_units
        return AISlots(np.arange(self.minion0, self.monster0), np.arange(self.struct0, n),
                       np.r_[0:self.monster0, self.epic0:self.ward0, self.struct0:n])

    @property
    def packet_capacity(self) -> int:
        """Damage packets per tick after compaction: ``packets``, else the power of two >= 2 per unit (512 on the
        full map). Overflowing packets are dropped and counted (``packet_overflow``)."""
        return self.packets or 1 << (2 * self.n_units - 1).bit_length()

    @property
    def follow_up_capacity(self) -> int:
        """Packets of the follow-up (trigger) pass: half the main pass."""
        return self.packet_capacity // 2

    @property
    def ray_capacity(self) -> int:
        """Sight rays per tick after compaction (enemy pairs in sight range): the power of two >= 16 per unit
        (4096 on the full map; chaos play peaks near 1050 there, 640 on the top-lane world)."""
        return 1 << (16 * self.n_units - 1).bit_length()


@dataclass(frozen=True)
class WorldConfig:
    terrain: tuple                      # per-team StaticTerrain (gate masks)
    routes: Any                         # FlowRoutes
    loadouts: tuple
    champion_ids: Any                   # (C,) int32
    champion_base: Any                  # core.stat_pipeline.ChampionBase (C,)
    rune_pages: Any                     # (C, R) int32 page counts (prepared pages)
    adaptive_physical: Any              # (C,) bool
    uses_energy: Any                    # (C,) bool
    unit_kind: Any                      # (N,) int32 static kind per slot
    unit_team: Any                      # (N,) int32
    unit_sub: Any                       # (N,) int32 (turret tier for structures)
    unit_x: Any
    unit_y: Any
    unit_lane: Any                      # (N,) int32 lane index (structures), -1 otherwise
    structure_prereq: Any               # (N,) int32 unit that must die first (-1 none)
    lane_path: Any                      # (W, 2) top-lane path, blue -> red
    fountain: Any                       # (2, 2)
    dt: float = 1.0 / 30.0
    profile: dict = field(default_factory=dict)
    vision: Any = None                  # obs.vision.VisionGrid over the navgrid (vision); None = no fog
    jungle: Any = None                  # jungle.camps.JungleTable (camp slots)
    objectives: Any = None              # jungle.objectives.ObjectiveTable (epic slots)
    rift: Any = None                    # map.rift.RiftTerrain (elemental/Baron-pit variants)
    regions: Any = None                 # map.regions.MapRegions
    footprints: Any = None              # map.dynamic_terrain.Footprints (structure pads)
    ward_grid: Any = None               # wards.WardGrid
    layout: Layout = Layout()           # unit blocks (static)
    item_allowed: Any = None            # (C, I) numpy bool: items each champion can ever hold (None: any)

    @property
    def n_units(self) -> int:
        return int(self.unit_kind.shape[0])



def _item_allowed(loadouts):
    """(C, I) bool items each champion can hold (its allow-list and starting items, closed over transforms and rune
    grants); None when no champion is restricted."""
    if not any(lo.allowed_items for lo in loadouts):
        return None
    return np.stack([acquirable_rows((*lo.allowed_items, *lo.items)) if lo.allowed_items
                     else np.ones_like(acquirable_rows(())) for lo in loadouts])

def build_config(loadouts, *, map_path=DEFAULT_MAP, route_path=DEFAULT_ROUTES, dt=1.0 / 30.0,
                 fog: bool | str = "rays", lanes=(0, 1, 2), jungle: bool = True,
                 objectives: bool = True, packet_capacity: int = 0, lane_structures: bool = False) -> WorldConfig:
    """Validate loadouts and build the static world (host-side).

    ``fog``: ``"rays"`` (default; walls and brush along the ray), ``"fast"`` (brush lookup, walls
    ignored) or ``False`` (every unit visible). See ``vision``. ``lanes``: lanes whose minion
    waves spawn (0 bot, 1 mid, 2 top). ``jungle`` / ``objectives``: spawn camps / epic monsters.
    ``packet_capacity``: damage packets per tick (0: ``Layout.packet_capacity`` default). ``lane_structures``: only
    the spawning lanes' turrets and inhibitors, plus the Nexus turrets and Nexuses (a one-lane game never reaches the
    others).
    """
    if fog not in (False, "fast", "rays"):
        raise ValueError("fog must be 'fast', 'rays' or False")
    lay = Layout(tuple(int(v) for v in lanes), bool(jungle), bool(objectives), int(packet_capacity),
                 bool(lane_structures))
    if len(loadouts) != N_CHAMPIONS:
        raise ValueError("the modern lane world has exactly two champions")
    pages = []
    for lo in loadouts:
        traits = K.kit(lo.champion).TRAITS
        validate_item_loadout(tuple(lo.items))
        traits = RD.ChampionTraits(traits.has_immobilize, traits.resource, traits.special,
                                   flash_equipped=4 in lo.summoners, adaptive_physical=traits.adaptive_physical)
        pages.append(validate_rune_page(lo.rune_page, traits))
    grid, manifest = load_patch_map(Path(map_path))
    routes, route_meta = load_routes(route_path, grid)
    geometry = json.loads((PATCH_DIR / "geometry.json").read_text())
    terrain = tuple(grid.as_jax(team) for team in (0, 1))

    kinds, teams, subs, xs, ys, lanes = [], [], [], [], [], []

    def add(kind, team, sub=0, pos=(0.0, 0.0), lane=-1) -> int:
        kinds.append(kind); teams.append(team); subs.append(sub); xs.append(pos[0]); ys.append(pos[1])
        lanes.append(lane)
        return len(kinds) - 1
    for c in range(N_CHAMPIONS):
        add(W.KIND_CHAMPION, c, pos=FOUNTAINS[c])
    for _ in range(lay.minion0, lay.monster0):                     # spawn slots stay KIND_NONE until written
        add(W.KIND_NONE, 0)
    for _ in range(lay.monster0, lay.ward0):
        add(W.KIND_NONE, W.NEUTRAL)
    for k in range(lay.struct0 - lay.ward0):
        add(W.KIND_NONE, k // W.MAX_WARDS_PER_TEAM)
    index = {}
    kept = lambda lane, tier: not lay.lane_structures or tier == "nexus" or lane in lay.lanes   # noqa: E731
    for o in geometry["turrets"]:
        lane = o["lane"] if isinstance(o["lane"], int) else LANE_NAMES.index(o["lane"])
        if kept(lane, o["tier"]):
            index[(o["team"], lane, o["tier"])] = add(W.KIND_TURRET, o["team"], TIERS[o["tier"]], o["position"], lane)
    for (team, lane_name), pos in INHIBITORS.items():
        lane = LANE_NAMES.index(lane_name)
        if kept(lane, "inhib"):
            index[(team, lane, "inhib")] = add(W.KIND_INHIBITOR, team, 0, pos, lane)
    for team in (0, 1):
        add(W.KIND_NEXUS, team, 0, NEXUS_POSITIONS[team])
    n = len(kinds)
    assert n == lay.n_units, (n, lay)
    prereq = np.full(n, -1, np.int32)
    for (team, lane, tier), slot in index.items():
        if tier in PREREQ:
            prereq[slot] = index[(team, lane, PREREQ[tier])]
    champion_names = [lo.champion for lo in loadouts]
    traits = [K.kit(nm).TRAITS for nm in champion_names]
    profile = {"patch": "26.19", "mode": "CLASSIC", "map_id": 11, "lane": "top", "dt": dt,
               "map_arrays_sha256": manifest["arrays_sha256"], "routes": route_meta,
               "nexus_position": "navgrid Nexus pad centre (STRUCTURE cells, LANES_TERRAIN)",
               "lanes": lay.lanes, "jungle": jungle, "objectives": objectives, "lane_structures": lay.lane_structures,
               "vision": f"fog={fog!r}: sight radii, brush{', walls' if fog == 'rays' else ''}, structures always "
                         "visible, attack reveal",
               "deferred": ["allied champions (ally-targeted effects inert)", "champions without a kit module"]}
    return WorldConfig(
        terrain=terrain, routes=routes, loadouts=tuple(loadouts),
        champion_ids=jnp.asarray([K.kit(nm).ID for nm in champion_names], jnp.int32),
        champion_base=champion_base(champion_names),
        rune_pages=jnp.asarray(RD.page_counts(pages)),
        adaptive_physical=jnp.asarray([t.adaptive_physical for t in traits]),
        uses_energy=jnp.asarray([t.resource == "energy" for t in traits]),
        unit_kind=jnp.asarray(kinds, jnp.int32), unit_team=jnp.asarray(teams, jnp.int32),
        unit_sub=jnp.asarray(subs, jnp.int32), unit_x=jnp.asarray(xs, jnp.float32),
        unit_y=jnp.asarray(ys, jnp.float32), unit_lane=jnp.asarray(lanes, jnp.int32),
        structure_prereq=jnp.asarray(prereq),
        lane_path=jnp.asarray(np.asarray(geometry["lane_paths"]["top"], np.float32)),
        fountain=jnp.asarray(FOUNTAINS, jnp.float32), dt=dt, profile=profile,
        vision=MV.vision_grid(grid, rays=fog == "rays") if fog else None,
        jungle=J.build_table(lay.monster0) if jungle else None,
        objectives=OBJ.load_table(lay.epic0) if objectives else None,
        rift=RIFT.load_rift_terrain() if objectives else None,
        regions=REG.build_regions(grid), footprints=DTR.build_footprints(grid, np.asarray(kinds), np.asarray(xs),
                                                                         np.asarray(ys)),
        ward_grid=WD.ward_grid(grid), layout=lay, item_allowed=_item_allowed(loadouts))
