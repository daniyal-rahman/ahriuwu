"""Patch-26.19 modern world: static configuration and the fixed unit layout.

Host-side construction of everything the modern tick (``modern_step``)
closes over: Map11 terrain per team and the route graph, structure and lane
geometry, champion records, loadouts (items, rune pages, summoner spells,
roles). No Map1/legacy fallback: missing assets raise.

Unit layout (fixed shapes, N = 2 + MAX_MINIONS + structures):
  0..1                      champions (holder c = unit c; team c)
  2 .. 2+MAX_MINIONS-1      lane-minion slots (top lane, both teams)
  then                      22 turrets, 6 inhibitors, 2 Nexuses

Geometry sources (``data/modern/26.19/geometry.json``, client
``base_srx.materials.bin``): turrets and minion barracks from placement
records; inhibitors from the ``SRUAP_*_Inhibitor_Idle*`` placements (one per
lane per team); fountains/spawn platforms ``__Spawn_T1/T2``. The Nexus has no
placement record in the decoded materials bin: its position is the midpoint
of the two Nexus turrets pushed 350 units toward the fountain (INFERRED-L,
flagged in ``profile``; a top-lane game never reaches it before 20 min).
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import numpy as np

from . import modern_world_types as W

DATA = Path(__file__).resolve().parents[1] / "data" / "modern" / "26.19"
DEFAULT_MAP = Path("/mnt/nfs/shared/modern-world-map-research/grid-26.19-base")
DEFAULT_ROUTES = Path("/mnt/nfs/shared/WORLD001_map_routes/routes")
MAX_MINIONS = 40
N_CHAMPIONS = 2
FOUNTAINS = ((394.0, 461.0), (14340.0, 14391.0))            # __Spawn_T1 / __Spawn_T2
SHOP_CENTERS = ((412.9, 416.2), (14297.2, 14388.3))         # Order/ChaosShopAreaCenter
INHIBITORS = {  # (team, lane) -> position, from SRUAP_*_Inhibitor_Idle placements
    (0, "top"): (1171.6, 3569.7), (0, "mid"): (3198.6, 3218.5), (0, "bot"): (3456.0, 1200.1),
    (1, "top"): (11256.2, 13675.8), (1, "mid"): (11600.0, 11671.4), (1, "bot"): (13598.7, 11311.9),
}
LANES = ("bot", "mid", "top")         # geometry.json lane ids: 0 bot, 1 mid, 2 top
TIERS = {"outer": 0, "inner": 1, "inhibitor": 2, "nexus": 3}
# Structure stats not in towers.json (TOWERS §1.3–1.4, D4).
INHIBITOR_HP, INHIBITOR_ARMOR, INHIBITOR_REGEN, INHIBITOR_RESPAWN = 4000.0, 20.0, 15.0, 300.0
NEXUS_HP, NEXUS_ARMOR, NEXUS_REGEN = 5500.0, 20.0, 20.0


@dataclass(frozen=True)
class Loadout:
    """One champion's game setup (host-side, validated at build time)."""
    champion: str                       # "Garen" | "Jax"
    items: tuple[int, ...] = ()
    rune_page: Any = None               # modern_rune_data.RunePage or None (no-runes ruleset)
    summoners: tuple[int, int] = (4, 12)  # Flash, Teleport
    role: int = 1                       # modern_role_quest.ROLE_TOP
    skill_order: tuple[int, ...] = ()   # slot per level 1..18; empty = champion default


@dataclass(frozen=True)
class WorldConfig:
    terrain: tuple                      # per-team StaticTerrain (gate masks)
    routes: Any                         # FlowRoutes
    loadouts: tuple
    champion_ids: Any                   # (C,) int32
    champion_base: Any                  # modern_stat_pipeline.ChampionBase (C,)
    rune_pages: Any                     # (C, R) int32 page counts (prepared pages)
    adaptive_physical: Any              # (C,) bool
    uses_energy: Any                    # (C,) bool
    unit_kind: Any                      # (N,) int32 static kind per slot
    unit_team: Any                      # (N,) int32
    unit_sub: Any                       # (N,) int32 (turret tier for structures)
    unit_x: Any
    unit_y: Any
    unit_lane: Any                      # (N,) int32 lane index (structures), -1 otherwise
    structure_prereq: Any               # (N,) int32 unit that must die first (-1 none), second prereq in prereq2
    structure_prereq2: Any
    lane_path: Any                      # (W, 2) top-lane path, blue -> red
    minion_spawn: Any                   # (2, 2) barracks position per team (top)
    fountain: Any                       # (2, 2)
    dt: float = 1.0 / 30.0
    profile: dict = field(default_factory=dict)

    @property
    def n_units(self) -> int:
        return int(self.unit_kind.shape[0])


def unit_ranges():
    m0 = N_CHAMPIONS
    s0 = m0 + MAX_MINIONS
    return slice(0, N_CHAMPIONS), slice(m0, s0), s0


def build_config(loadouts, *, map_path=DEFAULT_MAP, route_path=DEFAULT_ROUTES, dt=1.0 / 30.0) -> WorldConfig:
    """Validate loadouts and build the static world (host-side)."""
    from ..data.modern_map import load_patch_map
    from ..data.modern_routes import load_routes
    from . import modern_rune_data as RD
    from .modern_inventory import validate_item_loadout
    from .modern_stat_pipeline import champion_base
    from .modern_items import validate_rune_page
    if len(loadouts) != N_CHAMPIONS:
        raise ValueError("the modern lane world has exactly two champions")
    ids = {"Garen": 86, "Jax": 24}
    pages = []
    for lo in loadouts:
        if lo.champion not in ids:
            raise ValueError(f"unsupported modern champion {lo.champion!r}")
        validate_item_loadout(tuple(lo.items))
        traits = RD.CHAMPION_TRAITS[lo.champion]
        traits = RD.ChampionTraits(traits.has_immobilize, traits.resource, traits.special,
                                   flash_equipped=4 in lo.summoners, adaptive_physical=traits.adaptive_physical)
        pages.append(validate_rune_page(lo.rune_page, traits))
    grid, manifest = load_patch_map(Path(map_path))
    routes, route_meta = load_routes(route_path, grid)
    geometry = json.loads((DATA / "geometry.json").read_text())
    terrain = tuple(grid.as_jax(team) for team in (0, 1))

    kinds, teams, subs, xs, ys, lanes = [], [], [], [], [], []
    for c in range(N_CHAMPIONS):
        kinds.append(W.KIND_CHAMPION); teams.append(c); subs.append(0)
        xs.append(FOUNTAINS[c][0]); ys.append(FOUNTAINS[c][1]); lanes.append(-1)
    for _ in range(MAX_MINIONS):
        kinds.append(W.KIND_NONE); teams.append(0); subs.append(0); xs.append(0.0); ys.append(0.0); lanes.append(-1)
    s0 = len(kinds)
    index = {}
    for o in geometry["turrets"]:
        lane = o["lane"] if isinstance(o["lane"], int) else LANES.index(o["lane"])
        index[(o["team"], lane, o["tier"])] = len(kinds)
        kinds.append(W.KIND_TURRET); teams.append(o["team"]); subs.append(TIERS[o["tier"]])
        xs.append(o["position"][0]); ys.append(o["position"][1]); lanes.append(lane)
    for (team, lane_name), pos in INHIBITORS.items():
        index[(team, LANES.index(lane_name), "inhib")] = len(kinds)
        kinds.append(W.KIND_INHIBITOR); teams.append(team); subs.append(0)
        xs.append(pos[0]); ys.append(pos[1]); lanes.append(LANES.index(lane_name))
    nexus_turrets = {t: [o["position"] for o in geometry["turrets"] if o["team"] == t and o["tier"] == "nexus"]
                     for t in (0, 1)}
    for team in (0, 1):
        mid = np.mean(np.asarray(nexus_turrets[team], float), axis=0)
        to_fountain = np.asarray(FOUNTAINS[team]) - mid
        pos = mid + 350.0 * to_fountain / np.linalg.norm(to_fountain)
        index[(team, -1, "nexus")] = len(kinds)
        kinds.append(W.KIND_NEXUS); teams.append(team); subs.append(0)
        xs.append(float(pos[0])); ys.append(float(pos[1])); lanes.append(-1)
    n = len(kinds)
    prereq = np.full(n, -1, np.int32)
    prereq2 = np.full(n, -1, np.int32)
    for (team, lane, tier), slot in index.items():
        if tier == "inner":
            prereq[slot] = index[(team, lane, "outer")]
        elif tier == "inhibitor":
            prereq[slot] = index[(team, lane, "inner")]
        elif tier == "inhib":
            prereq[slot] = index[(team, lane, "inhibitor")]
        # Nexus turrets and the Nexus: team-level rules derived by modern_lane_ai (TOWERS §2).
    lane_path = np.asarray(geometry["lane_paths"]["top"], np.float32)
    spawns = np.asarray([next(o["position"] for o in geometry["barracks"]
                              if o["team"] == t and o["lane"] in (2, "top")) for t in (0, 1)], np.float32)
    champion_names = [lo.champion for lo in loadouts]
    traits = [RD.CHAMPION_TRAITS[nm] for nm in champion_names]
    profile = {"patch": "26.19", "mode": "CLASSIC", "map_id": 11, "lane": "top", "dt": dt,
               "map_arrays_sha256": manifest["arrays_sha256"], "routes": route_meta,
               "nexus_position": "INFERRED-L: Nexus-turret midpoint + 350 toward fountain",
               "deferred": ["dynamic terrain", "full vision and wards", "jungle and objectives",
                            "item actives except Tiamat line/Stridebreaker", "other lanes' waves"]}
    return WorldConfig(
        terrain=terrain, routes=routes, loadouts=tuple(loadouts),
        champion_ids=jnp.asarray([ids[nm] for nm in champion_names], jnp.int32),
        champion_base=champion_base(champion_names),
        rune_pages=jnp.asarray(RD.page_counts(pages)),
        adaptive_physical=jnp.asarray([t.adaptive_physical for t in traits]),
        uses_energy=jnp.asarray([t.resource == "energy" for t in traits]),
        unit_kind=jnp.asarray(kinds, jnp.int32), unit_team=jnp.asarray(teams, jnp.int32),
        unit_sub=jnp.asarray(subs, jnp.int32), unit_x=jnp.asarray(xs, jnp.float32),
        unit_y=jnp.asarray(ys, jnp.float32), unit_lane=jnp.asarray(lanes, jnp.int32),
        structure_prereq=jnp.asarray(prereq), structure_prereq2=jnp.asarray(prereq2),
        lane_path=jnp.asarray(lane_path), minion_spawn=jnp.asarray(spawns),
        fountain=jnp.asarray(FOUNTAINS, jnp.float32), dt=dt, profile=profile)
