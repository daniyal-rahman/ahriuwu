# LANES_TERRAIN.md — all three lanes, dynamic terrain, map regions, attack-move, game end, item actives (26.19)

Patch 26.19, client 16.19.8230722, Summoner's Rift CLASSIC. Evidence levels: **CLIENT** (client data), **WIKI**
(wiki revision cited), **PATCH** (Riot notes), **INFERRED-M / INFERRED-L** (our reading, medium / low confidence).

Code: `lanerl_jax/sim/modern_minions.py` (schedule), `modern_lane_ai.py` (spawn writes, AI, attack-move, game end),
`modern_towers.py` (structure rules), `modern_dynamic_terrain.py`, `modern_map_regions.py`, `modern_item_actives.py`.
Tests: `tests/test_modern_{minions,lane_ai,towers,dynamic_terrain,map_regions,item_actives}.py`.

## 1. All three lanes

### 1.1 Spawning

| Rule | Value | Evidence |
|---|---|---|
| Wave clock | same for every lane and team: `wave_spawn_time(i)` (30 s, 25 s from 14:00, 20 s from 30:10) | PATCH 26.1, WIKI (MINIONS §2.2) |
| Composition | per lane and team; melee/cannon/caster rotation is global (`wave_unit_type`) | CLIENT (MINIONS §2.3) |
| Supers | per `(team, lane)` from the *enemy* inhibitor of that lane: 1 if it is down, 2 in every lane when all three are down, 0 within two wave intervals of its respawn; a super replaces the cannon | CLIENT `SpawnCountPerInhibitorDown [1,1,2]`, WIKI (inhibitor: "next 8 waves") |
| Latching | the super count is fixed when a wave's first unit spawns, so the unit list of a wave never changes mid-wave | INFERRED-M |
| Stagger | 0.8 s between units of a wave, one unit per `(team, lane)` per tick | CLIENT `MinionSpawnIntervalSecs` |
| Position | the barracks of `(team, lane)` (`geometry.json`, `modern_lane_ai.BARRACKS[team, lane]`) | CLIENT `base_srx.materials.bin` |

State: `modern_minions.LaneSpawnState` `(wave, unit, supers)`, each `(2, 3)` `[team, lane]`; `init_lane_spawn()`.
`lane_spawn_step(state, now, **wave_inhibitor_inputs(towers))` returns the units due this tick.

Slot layout (types contract): lane `l` owns slots `[C + 40·l, C + 40·(l+1))`, shared by both teams (Blue takes the
lowest free slot of the lane, then Red). Free = `KIND_NONE` or a dead minion. A due unit with no free slot in its
lane is dropped and counted in `SpawnWrite.overflow` (it must stay 0; 40 slots hold about five full waves).
`modern_lane_ai.spawn_lane_minions(spawn, towers, kind, alive, now=, slot0=C, lanes=(0, 1, 2))` returns
`(spawn_state, SpawnWrite, due)`. `lanes=(2,)` keeps the top-only scenario; the other cursors still advance.

### 1.2 Lane AI per lane

- **Lane assignment.** `LaneAIState.lane` is set when a minion first appears: the nearest own barracks (exact at
  spawn); `slot_lane(n, C)` gives the same answer from the slot index.
- **Paths.** `LANE_PATHS[team, lane]`: client `MinionPath_*` splines, reversed for Chaos. After the last point,
  minions walk to the enemy Nexus (MINIONS §2.5).
- **Side-lane speed.** The buff applies to lanes 0 and 2 only (MINIONS §2.7, WIKI-M).
- **First wave.** It is ghosted for 28 s (side lanes) or 18 s (mid) after spawn: `minion_ghosted` (MINIONS §2.8,
  WIKI-M).
- **Neutral units.** Team relations use the raw team. Jungle monsters (team 2) are nobody's ally, so they cannot
  fake Call-for-Help ally checks. Minions and turrets never target monsters or wards (TOWERS §3.1, WIKI-H).

### 1.3 Structures in every lane

Already lane-generic, now verified for all lanes and both teams (`test_modern_towers.py`):

- **Vulnerability chain.** Outer, then inner, then inhibitor turret, then inhibitor, using lane prerequisites
  (TOWERS §2, WIKI-H).
- **Nexus turrets** are targetable only while ≥1 own inhibitor is down. They have no plates, respawn 180 s after
  death at 40 % HP, and stay untargetable after respawning until an inhibitor is down (CLIENT 1505, WIKI).
- **Nexus** needs an inhibitor down and both Nexus turrets dead.
- **Inhibitor respawn** is 300 s, at full HP (WIKI-H). It re-locks the Nexus and its turrets.
- **Plates** exist on outer, inner and inhibitor turrets. Plate gold is 120; on outer turrets it decays by 10 per
  step from 11:00 down to 80.
  - Evidence: PATCH 26.1 "all turrets (except Nexus turrets) now have turret plates"; CLIENT item 1515
    `Tier1SRPalisadeCount 5`, `Tier2PalisadeCount 5`, `Tier2PalisadeValue 120`.
  - This contradicts "plates only on outer turrets"; the code follows the client and patch data.
  - The 14:00 fall-off was removed in 26.1 (PATCH-H).

## 2. Dynamic terrain

Evidence:

- The navgrid marks every structure pad with `StructureWall` (0x4). It always appears together with
  WALL|TRANSPARENT (flag 70). Pad sizes match the pathfinding radii (CLIENT):

  | Pads | Count | Cells each | Radius |
  |---|---|---|---|
  | Turret | 22 | 18–21 (5×5) | 125 |
  | Inhibitor | 6 | 44–61 | 213.75 |
  | Nexus | 2 | 157/158 | 304 |
  | Fountain-platform pieces (not structures) | 2 | small | — |

- **Turret pads stay blocked after destruction.** Wiki Turret: "This terrain remains even after the turret is
  destroyed" (WIKI-H). The client also has a `TurretRubble` character with pathfinding radius 100 (CLIENT).
- **Inhibitor pads: no direct source.** Map11 ships navgrid overlays only for the Baron pit and dragon-soul
  terrain. Structure cells are in the static base grid. Default: inhibitor pads stay blocked as well
  (INFERRED-M).
- **Nexus pad.** Its centroid is (1549, 1658) / (13240, 13235). `modern_world` infers the Nexus at
  (1716, 1790) / (12998, 12950), about 214 units off. This is CLIENT evidence for the Nexus position.

API (`modern_dynamic_terrain`):

- `build_footprints(grid, unit_kind, unit_x, unit_y)` (host). Each 8-connected StructureWall component goes to the
  nearest structure unit within 450. It raises if a structure gets no pad.
- `walkable_masks(cfg.terrain, footprints, alive, release_mask(kind, policy))` (JAX). Returns per-team
  `StaticTerrain`s with dead, released pads opened for both teams.
- `RELEASE_ON_DEATH` keeps every pad blocked, so with the default policy `walkable_masks` returns the base masks.
- `eject(x, y, team, radius, terrain)` pushes units off cells that became blocked (inhibitor respawn) to the
  nearest walkable ring point within 450.

Route graph (`/mnt/nfs/shared/WORLD001_map_routes/routes`):

- It was baked on the static grid, so no node lies inside a pad. Under the default policy nothing changes.
- If pads are released, the step should pass the dynamic masks to `move_step`, `blink_point` and the flash/dash
  clamps. `route_next` already prefers a direct segment when `segment_clear` passes on the given terrain (≤ 600
  units), and a pad is ≤ 750 units wide. Units therefore steer straight across an opened pad whenever the straight
  segment is clear, and fall back to graph nodes around it otherwise.
- No graph rebake is needed. A detour of at most one pad diameter remains when the goal is more than 600 units
  past the pad (INFERRED-M). Vision is unaffected, because pads are TRANSPARENT.

## 3. Map regions

Source: the NGRID v7.1 region bytes (CLIENT). The enum names come from FrankTheBoxMonster/LoL-NGRID-converter
`NavGridCell.cs` @92943ed. The MainRegion codes are 0 Spawn, 1 Base, 2–4 Top/Mid/Bot lane, 5–6 Top/Bot-side jungle,
7–8 Top/Bot-side river, 9–10 base perimeter and 11–12 lane alcoves. The Ring nibble gives the side (0–4 Order,
5–9 Chaos).

Cross-checks on walkable cells:
- jungle 5/6 vs river-byte bit 0: 14041 of 14346 cells agree;
- river 7/8 vs bit 64: 4901 of 4946 cells agree;
- dragon-pit POI centroid: (9870, 4409).

| Query | Definition | Evidence |
|---|---|---|
| `lane_of(x, y)` | lane region (+ that side's alcove), else −1 | CLIENT regions; alcove in lane INFERRED-M |
| `in_quest_lane(x, y, lane)` (ROLE_QUESTS U-RQ-1) | `lane_of == quest lane`. 26.9: "anywhere in the lane, outside of your base"; lane regions stop at both bases | PATCH 26.9 + CLIENT |
| `in_jungle` (Homeguard) | MainRegion 5/6 (not river, base perimeter or alcoves) | CLIENT; Homeguard semantics INFERRED-M |
| `in_river` (Waterwalking) | MainRegion 7/8, which includes the dragon and Baron pit floors | CLIENT; pits count INFERRED-M |
| `in_base(team)` / `in_spawn_platform(team)` | MainRegion 0/1 on that side / MainRegion 0 on that side | CLIENT |
| `homeguard_flags(...) -> (reached_endpoint, in_jungle)` | see below | ECONOMY 11.2.3, WIKI-M |

`reached_endpoint`: the champion is in a lane region and its arc-length progress along its team's path of that lane
is ≥ the endpoint. The endpoint is
`max(progress(outermost living allied lane turret) − 500, progress(own inhibitor))`. After 14:00, or once an
allied turret of that lane is down, it is at least `progress(furthest living allied minion of the lane) − 2000`.

Minion lane for the quest's minion rules: `modern_lane_ai.minion_in_lane(ai, units, lane=2)`, i.e. the spawn lane
(ROLE_QUESTS U-RQ-2 default).

## 4. Attack-move and idle acquisition

Source: wiki Basic attack (rev 4052292, WIKI-H) and the CharacterRecord `acquisitionRange` (CLIENT, 400 for Garen
and Jax). "Attack range modifiers affect this value by the same amount" (wiki Champion statistic):
`champion_acquisition_range(range, base_range)`.

Rules:
- **Candidates.** Enemy champions, minions, structures and wards, plus monsters only where
  `monster_aggro[c, j]` ("attacked monsters"). Each must be alive, targetable and visible to the team.
- **Attack-move.**
  - A held target stays while it remains valid (same `spawn_seq`); acquisition is not re-scanned.
  - Otherwise the scan picks the valid hostile nearest the **champion** whose hitbox is within the acquisition
    radius. There is no champion-over-minion priority.
  - "Target Champions Only" restricts candidates to champions.
  - The option "Attack move on cursor" (off by default) first tries units within `cursor_radius` of the cursor,
    nearest to the cursor. The radius is unpublished; the default is the acquisition radius (INFERRED-L).
  - With no target, the champion walks to the point. Within 10 units of it the order ends, and idle acquisition
    takes over.
- **Idle.** `idle_acquire` returns the nearest valid enemy, excluding wards and unaggroed monsters, within the
  acquisition radius. The champion then chases it like an attack order.
  - The current step instead uses the attack range and never chases, which is a known gap.
- Edge distance is used for the radius test and center distance for ordering (INFERRED-M).

API: `attack_move_step(active, point_x, point_y, held, held_seq, units, champion_unit, visible, acq, ...)` returns
`AttackMoveOut(target, target_seq, goal_x, goal_y, active)`.

## 5. Game end

`modern_lane_ai.game_result(towers) -> GameResult(over, winner)`. The team whose Nexus still stands wins. Nexuses
never respawn, so the result is sticky. `PlateEvents.nexus_destroyed` flags the tick it happens. A same-tick double
Nexus kill gives `over=True, winner=-1`.

## 6. Item actives (`modern_item_actives`, registered in `modern_item_effects.MODULES`)

All 26.19 items with a non-vision active, other than consumables and the Hydra line.
Values come from `items_client.json` (CLIENT). Rules come from tooltips and wiki ItemData (WIKI).

| Item | Active | World effect needed |
|---|---|---|
| 3157 Zhonya's, 2420 Seeker's (single use → 2421) | stasis 2.5 s; cd 120 / once | `world.stasis`: untargetable + invulnerable; cannot move, attack, cast, use summoners or use items; Seeker's transform `world.transform` |
| 3140 Quicksilver, 3139 Mercurial | cleanse all CC except airborne (Mercurial +50 % MS 2 s); cd 90 | `world.cleanse` pulse → `M.apply_cc(..., cleansed=)` without clearing knockups |
| 3142 Youmuu's | +20 % (ranged 15 %) MS and ghosting 6 s (ranged 4 s); cd 45 | ghost via the `status` hook (already folded into `out.status`?) → collision `ghosted` |
| 3143 Randuin's | 70 % slow 2 s, enemies within 500 (edge); cd 90 | none (Effects.slow) |
| 3146 Gunblade | target enemy champion ≤ 700 (edge): 175–253 + 30 % AP magic, 25 % slow 1.5 s; cd 60 | target from `with_aim` |
| 3152 Rocketbelt | 275 dash (no terrain crossing) + rocket arc to 1050 (±30° INFERRED-L), 100 + 10 % AP magic once per enemy, attack reset; cd 50 | `world.dash` (speed 1500 INFERRED-L, not a blink) clamped by terrain |
| 2065 Shurelya's | +30 % MS 4 s to "you and all allies"; cd 75 | none |
| 3190 Locket | shield 290 (+7/level from 9) decaying over 2.5 s to "you and allied champions" in 850; cd 90 | none |
| 3107 Redemption | point ≤ 5500, lands after 2.5 s: allied units in 550 healed 150–350 (holder when inside, INFERRED-M); enemy champions 10 % max-HP true; usable while dead; cd 90 | aim point from `with_aim`; allied minions are not healed (no world heal path) |
| 2522 Actualizer | 8 s: spell mana costs ×2, +(15 + 0.5 % max mana) % ability damage and heal/shield power, basic cooldowns ×1.3 speed; cd 60 | `world.mana_cost_mult`, `world.basic_cd_rate` |
| 3222 Mikael's, 3109 Knight's Vow | ally-only (Purify targets an ally champion, Pledge binds an ally) | inert: request refused, no cooldown |

Notes:
- Support quest items (3867, 3869–3877) have ward-placement actives, which belong to vision/wards (`modern_wards`).
- Item-active haste (`ItemStats.item_haste`, Cosmic Insight) is not applied, because the hook context lacks it.
- Diminishing returns on repeated Locket/Intervention cannot occur in a 1v1.
- Actualizer overflow rules (`OverflowAddition/Revert`) are not modelled.
- Gating: `request_allowed(request, disabled=, in_stasis=)` lets only Quicksilver/Mercurial through under disabling
  CC, and nothing during stasis (INFERRED-M; silence does not block items).
