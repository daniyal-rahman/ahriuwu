# WORLD_IMPLEMENTATION.md — the 26.19 modern world tick

**Status (2026-10-03).** `lanerl_jax/modern/world/tick.py` runs a modern Summoner's Rift game end to end (two
champions, Garen and Jax), with every modern system composed in one JAX tick:
- minion waves in all three lanes and minion/turret AI;
- jungle camps, Scuttle Crabs, Smite and jungle pets ([JUNGLE.md](JUNGLE.md));
- epic objectives, team buffs and the Elemental Rift / Baron-pit terrain ([OBJECTIVES.md](OBJECTIVES.md));
- fog of war, wards, trinkets, stealth and true sight ([VISION.md](VISION.md), [WARDS.md](WARDS.md));
- destroyed-structure pads, map regions, attack-move, item actives and game end
  ([LANES_TERRAIN.md](LANES_TERRAIN.md));
- Garen and Jax kits;
- summoner spells;
- items and runes through the damage pipeline;
- crowd control with tenacity;
- economy, progression and the top role quest;
- shop, Flash, Teleport and Recall movement, collision, attacks and missiles.

The legacy C#-parity `step.py` is unchanged and remains the legacy ruleset, so its parity tests are not
affected.

## Layout

| File | Role |
|---|---|
| `world/config.py` | Host config `build_config(loadouts)`. It loads the Map11 terrain (per-team gate masks) and the route graph from the pinned artifacts (`/mnt/nfs/shared/modern-world-map-research/grid-26.19-base`, `/mnt/nfs/shared/WORLD001_map_routes/routes`). It sets the fixed unit layout and the structure chain, and validates loadouts (items, rune pages with client substitutions, summoners, role, skill order). |
| `core/types.py` | Shared contract: `WorldUnits`, `AttackState`, `AttackLaunch`, `CastOrder`, `CCOut` (with cast ids), `Dash`. |
| `mechanics.py` | Basic-attack machine (windup → launch → period; any cancel resets the timer; edge-to-edge range; tick rounding). Missiles, CC timers (tenacity applied at application with the 0.3 s floor, strongest slow, champion-CC clock), capability flags (DAMAGE §10.3), route movement with terrain clamp, Flash landing. |
| `lane/ai.py` | Minion and turret targeting, Call for Help, movement goals, their attack packets, minion spawn stats, structure tick, defense and plates (agent-built, see its docstring). |
| `champions/` | Garen and Jax kits on the packet/CC/stat contract (agent-built). |
| `champions/summoners.py` | Flash, Teleport and Unleashed Teleport (incl. quest rewards), Ignite, Exhaust, Barrier, Heal, Ghost, Cleanse (agent-built). Smite lives in `jungle.camps.smite_step`. |
| `jungle/camps.py` | Camps and Scuttle (spawns, AI, leash/reset, rewards, buffs), Smite, jungle pets. |
| `jungle/objectives.py`, `map/rift.py` | Voidgrubs, Rift Herald (+ Mercenary), Drakes/Soul/Elder, Baron, team buffs; 21 Rift × Baron-pit terrain variants from client navgrid overlays. |
| `wards.py`, `vision.py` | Ward slots and trinkets; fog with sight radii, walls, brush, stealth and true sight. |
| `map/dynamic_terrain.py`, `map/regions.py`, `items/effects/actives.py` | Structure pads (policy: pads stay, wiki), lane/jungle/river/base regions, item actives (stasis, cleanse, dashes, …). |
| `world/tick.py` | `ModernState`, `ModernOrders`, `init_state(cfg)`, `step(state, orders, cfg) -> (state, TickEvents)`. |

Units (216, `world.config.layout()`): 2 champions (unit c = holder c = team c), 120 lane-minion slots (40 per lane,
both teams), 48 monster slots (38 camp slots, 8 epic slots), 16 ward slots (8 per team), then 22 turrets,
6 inhibitors and 2 Nexuses. Structures come last so the fogged units are exactly the slots before them.

Positions come from:
- turrets and barracks: the client `base_srx.materials.bin` placements (`geometry.json`);
- inhibitors: the `SRUAP_*_Inhibitor_Idle` placements;
- fountains: `__Spawn_T1/T2`;
- **Nexus**: no placement exists in the decoded bin; it is the centre of its structure pad in the 26.19 navgrid
  (STRUCTURE cells, LANES_TERRAIN), about 214 units from the earlier inferred position.

## Tick order (`step`)

`step` is a pipeline of phase functions `(s, orders, cfg, sc) -> (s, sc)`. `sc` is a `TickScratch` NamedTuple;
its field list (each field commented with the phase that writes it) is the tick's data-flow map. Phases write
`s` only where the table says so. Everything else travels in `sc`, and `_commit` assembles the next state. Then
`step` freezes the world if a Nexus had already fallen and casts the state back to the incoming dtypes, so
`lax.scan` carries stay stable.

| Phase | Function | Main reads | Main writes |
|---|---|---|---|
| 1. INPUT | `_input` | orders, `s.visible`, `s.champ`, `s.amove`, `s.cc`, item stasis, `s.terrain_variant` | Fog-filtered orders: a target in fog is dropped. Move/attack/stop intents and attack-move orders (`ModernOrders.attack_move`). Stasis gates orders and `targetable`. Skill points from `level_up` or the default order. Minion waves (`lane.minions.spawn_event`, 0.8 s apart, `minion_spawn_stats`) and camp spawns go into free slots. Writes `s.champ`, `s.amove`, the spawn slots and `sc.caps`. `sc.terrain` is this tick's walkable mask: the Rift variant, then the structure pads. |
| 2. STATS | `_stats` | `s.champ.inventory`, `s.econ.level`, jungle/objective buffs, kits | `sc.static` and `sc.st_static`: items, shards, monster buffs (Blue, Red, shrine MS, drake stacks, soul, Hand of Baron) and kit stats. STAT.70 syncs champion HP to static max-HP changes (`s.hp`, `s.max_hp`, `s.champ.static_max_hp`). |
| 2b. OBJ | `_objectives` | `units_view(s)`, `s.damage_matrix`, `s.obj` | `objectives_step`: epic spawns, abilities, Rift transformation. Writes `s.obj`, the epic slots, `s.terrain_variant` and `sc.so`. |
| 3. CASTS | `_casts` | orders, `sc.st_static`, `s.champ.dyn`, slows | Shop: buy/sell in the shop area or while dead, with rune purchase blocks (`s.econ`, `s.champ`). `sc.st`: static stats plus last tick's dynamic stats and slows. Kit `cast` and `periodic`, `champions.summoners.step`, Smite (`smite_step`), kit attack mods and reach. |
| 4. AI | `_ai` | `s.towers`, `s.lane_ai`, `s.att`, last tick's `s.damage_matrix` (aggro), `s.visible` | Structure tick, then minion/turret targets and goals. Jungle and epic monster AI. Hand of Baron minion buffs. Team summons are minion-like targets. Champions: the ordered target, else attack-move (`LA.attack_move_step`), else idle acquisition (`LA.idle_acquire`, range 400, chases). Writes structure/monster rows of `s` and `s.obj`. Desired targets and goals go to `sc`. |
| 5. MOVE | `_move` | `sc` goals, `sc.st`, `sc.s_out`, `sc.kit_out`, `sc.terrain`, `s.pending_dash` | Route-graph steering from a cached per-unit route anchor (`s.route_anchor`; `map.pathing.route_follow`, at most 16 full nearest-node searches per tick for units that lost theirs, only for the slots before the wards) with a swept terrain clamp. Champion speed: Ghost/Heal, Gustwalker and Homeguard are bonus % MS in the STAT pipeline, before the soft caps. Non-champion slows count slow resist. Monsters walk to their AI goals. Kit dashes follow their target (dash state persists across ticks); the Rocketbelt dash starts the tick after its active. Flash lands on walkable terrain; Teleport arrives. Collision: Ghost, dashes and first-wave minions are ghosted; wards and structures don't collide (structures block through their navgrid pads). Writes `s.x/y`, `s.route_anchor`, `s.lane_ai`, `s.towers`, the dash state and facing (`s.champ` and the working `champ`) and `sc.ms/sres/units/ictx`. |
| 6. ATTACK | `_attack` | `sc.desired`, `sc.units`, `sc.kmods`, CC and lock gates | Attack machine for every unit (gated by kit `cannot_attack`, cast lockout, Teleport and dashes). Champion crit roll at launch (X-8) with item forced crits; the structure formula and melee ×1.2 vs turrets. Lane, jungle and epic attack packets; Baron siege multiplier vs structures. Ranged attacks become missiles; champion arrivals count as on-hit. Champion hits on wards become ward hits (1 HP each). Kit `on_attack`/`on_hit`, Overgrowth, jungle combat effects (Red, Scorchclaw, Gustwalker). Writes `s.obj` and `sc` packets, `att`, `missiles`, `kit_all`. |
| 7. DAMAGE | `_damage` | all `sc` packets, `s.hp`, `s.shields`, `s.status`, `s.kills` | `combat.combat_tick` runs items and runes around `core.damage`, after `objectives_packet_mods`. Defense: champion armor/MR, turret resists with Bulwark, backdoor ×0.2, untargetable structures, Teleport and stasis invulnerability, kit DR/dodge/AoE reduction, Garen E shred, Baron's Void Corruption. Offense: turret 30% armor pen, Exhaust as dealt reduction. `RuneEvents` from world state: windups, cast ids, CC, summoner casts, blinks, deaths, purchases, grants, river, epic/large kills. Then kit `on_damage`. Writes `sc.out`, `sc.hp`, `sc.shields`, `sc.status`. |
| 8. CC / HEAL | `_cc_heal` | `sc.kit_all`, `sc.s_eff`, `sc.jfx`, `sc.so`, `s.cc` | Kit, summoner, jungle and objective heals, mana and shields (HSP, incoming heal, Grievous Wounds). CC with tenacity (stat, Garen W, Cleanse) and slow resist. Scuttle is slow-immune with −100% tenacity. Monster CC is not champion-sourced. Exhaust and item/rune slows; Cleanse and item cleanse. Writes `sc.cc`. |
| 9. DEATH | `_death` | `sc.out` resolved packets, `sc.hp`, start-of-tick `s.sight` and `s.cc` | Killer per unit, damage matrix, `death_seen`. Plates and destruction (`structure_damage_events`). `MinionDeaths` (last hitter; gold/XP/level fixed at spawn). Camp and objective rewards (`extra_gold/xp`, `epic`). Map regions give the quest lane, Homeguard endpoint, jungle and river flags. Hand of Baron's 4 s Empowered Recall. `economy_step`: gold, XP, levels, bounty, kill credit, death timers, respawn, Recall, Homeguard, quest. Kit `on_takedown`. Writes `s.obj` and `sc.econ/eco/died`. |
| 10. TIMERS | `_timers` | `sc.eco`, `sc.out`, `sc.kit_all`, `sc.item_cleanse` | Respawn and Recall move the champion to the fountain. Kit cooldowns start hasted (basic vs ultimate), then rune refunds and item-active cooldown rate. Mana costs (item-active multiplier) and regen. HP regen in 0.5 s pulses, plus the fountain and Homeguard heal. Inventory: Tear and Armguard transforms, consumption, pet egg, rune grants into a free slot (acknowledged next tick). `ward_step`: ward gold, Control Ward use. Ward unit rows; dead minions free their slot. Ejection from terrain that closed. Results go to `sc` only. |
| 11. FOG | `_fog` | `sc` final positions/units, `s0.visible` | Attack-reveal circles, then `visible`/`sight` for the next tick and the observation, with wards, the Rift brush and Scuttle shrines (`vision`). `seen_cast` records casts the enemy saw. |
| COMMIT | `_commit` | `s`, `sc` | Clears CC on units that died, builds `TickEvents` and `LA.game_result`. Champion stat columns mirror `sc.st`. Returns the next state. |

One-tick lags (the value is produced in this tick and read in the next):
- `damage_matrix`: lane AI aggro, Scorchclaw, objective AI.
- `death_seen`: Overgrowth.
- `epic_prev` and `large_prev`: rune events.
- `pending_dash`: the Rocketbelt dash.
- `champ.reset_next`: item attack resets.
- `champ.dyn`: dynamic stats.
- `champ.granted`: the rune-grant acknowledgement.
- `visible` and `sight`: fog for orders and runes.
- Gustwalker's brush entry, through jungle state.

## Verified

- **`modern/tests/test_step.py`** covers:
  - layout and starting structure vulnerability (only outer turrets);
  - 100 s with no orders: passive gold equals `500 + ambient_payments`, both teams' waves are alive, no packet
    or missile overflow;
  - a shop purchase in the fountain (gold and inventory);
  - Flash: a blink of 300–400 units and a 300 s cooldown after the 15 s start cooldown;
  - a champion kill: first-blood gold, the death timer from `death_time`, respawn at the fountain;
  - Recall back to the fountain after 8 s.
- **`modern/tests/test_mechanics.py`** covers the attack cadence (launch at 0.3 s, then every 1.0 s at 30 Hz),
  edge-to-edge range, cancel reset, missile homing, CC tenacity (DAMAGE F17) and strongest slow.
- **`modern/tests/test_world_rules.py`** checks waves in all lanes and camps at 1:30, symmetric idle minion
  trades, a ward revealing an enemy in brush, camp kill rewards, attack-move acquisition and the game-end freeze.
- **Scripted 4-minute game** (both champions walk to lane, last-hit, fight with Q/E):
  - passive gold from 65 s;
  - first blood at ~75 s (+400);
  - victims respawn and the kill trade back pays about 390 g;
  - XP from shared minion deaths, both champions level 4 by 3:30;
  - bounties settle to 0;
  - no overflow.
- **Cost (MODERN-021):**
  - Desktop CPU (8 cores, 16 vmapped envs, waves on the map): 108 ms per tick, down from 191 ms before
    MODERN-021. The HEAD 1e15e06 top-lane world (72 units) ran at 41 ms.
  - Compile takes about 2–4 min.
  - Ablations before the fixes: ray fog costs about 30% (`fog="fast"` costs about the same as no fog), and jungle
    plus objectives about 15%.
  - Remaining CPU cost: ray fog about 32% (the CPU reference caster; CUDA uses the fused kernel), routing
    walkability about 19%, lane AI about 10%.
  - GPU RTX 5080: 512 envs 54 ms per tick (9.5k env-ticks/s; the HEAD world reached 11.7k at 1024). 1024 envs run out
    of memory: `route_next` connection checks dominate per-env memory. Fog costs about 5% on GPU. Details in the MODERN-021 ledger row. Measure with `ops/modern/bench.py` and attribute with
    `ops/modern/profile_tick.py`.

## Known gaps and approximations

- **Champions:** only Garen and Jax, one per team. Ally-targeted item, rune and summoner effects are inert because
  there are no allies; jungler ganks need another champion.
- **Champion kits:** Garen and Jax follow the 26.19 client data per [CHAMPIONS.md](CHAMPIONS.md), which lists the
  seven small remaining approximations.
- **Per-system approximations** are listed in each spec: VISION.md (U-VIS-1..3, `fog="fast"` mode), WARDS.md
  (U-W-1..7), JUNGLE.md (pathing distance, patience/reset rates INFERRED-L, Scuttle path), OBJECTIVES.md (Rift
  timing, Baron shred, Rodeo deferred, Herald eye pickup, Voidmite cap, INFERRED-L leash/ability numbers),
  LANES_TERRAIN.md (pads stay after destruction per the wiki; cursor attack-move radius; Rocketbelt arc).
- **Integration simplifications:** empowered-minion splash (Hand of Baron) is not applied (it would need an N×N
  packet set); Gromp's magic bonus lands at launch, not missile arrival; neutral monsters walk the blue team's
  walkable mask. (The Scuttle shrine's 525 sight and the jungle-pet laner reward reductions are wired.)
- **Routing:** a unit keeps steering by its route anchor while that node stays in sight; this replaced the per-tick
  25-candidate nearest-node search (MODERN-022: same or better arrival on 400 real-map routes, 4.2x cheaper
  movement). Units beyond the 16 searches per tick hold position for a tick.
- **Collision:** `collision.resolve` (COLLISION.md): pathing radii, avoidance steering of movers, then soft
  Jacobi separation that never pushes a unit into terrain (movement clearance on its team mask);
  `map.dynamic_terrain.eject` still frees units left inside closed terrain.
- **Observations and actions:** profile `modern-world-v1` (MODERN-005).
  - `lanerl_jax/modern/obs.py` returns `ModernObservation`: 32×20 entity slots (legacy 16 columns, a
    neutral-team column that is now used, then monster, epic monster, ward, Control Ward), the 6-column global
    vector and a 32-column `self` block (stats, cooldowns, mana, shield, summoners, level progress, quest, unspent
    skill points, trinket charges), plus `inventory`, `inventory_stack` and `affordable` (exactly the shop's
    `buy` check per catalog row). The enemy cast memory uses `ChampionLayer.seen_cast` (casts the enemy team saw).
  - `lanerl_jax/modern/actions.py` decodes `(button, sx, sy[, choice])` into `ModernOrders` with 19 buttons:
    the legacy eight, `summoner_d/f`, `level_q..r`, `buy` (catalog row), `sell` / `use_item` (inventory slot),
    `ward` (trinket at the cursor) and `control_ward`. `attack_move` without a hit is a real attack-move order.
    `Loadout(auto_skill=False)` leaves skill points to the policy.
  - Wired into the scan PPO trainer `lanerl_jax/modern/train.py` (TOOL, no experiment yet): reset bank
    from `init_state` + scripted walk to a top-lane hold point until `start_s` (60 s), a decision every 3 ticks
    (orders on the first tick, `no_orders` after), `vec_train`'s relative reward on `econ.gold_total`/`xp` with the
    lane potential on `lane_path` between the top outer turrets. `LanePolicy` has no `choice` head, so `buy`/`sell`/
    `use_item` (and `level_*`, with `auto_skill`) are masked off by default; champions keep their starting items.

## Structural debt (MODERN-021 review)

These are next refactors, in order. None of them changes behaviour.
1. **Phase functions.** *Done.* `step` is now a pipeline of `_input` … `_fog`, `_commit` over a `TickScratch`
   (see the tick-order table). It was verified to be bitwise identical to the previous monolithic `step`.
2. **World-subsystem protocol.** Jungle and objectives are wired into `step` by hand, through about 25
   `if cfg.jungle/objectives is not None` sites across seven phases.
   - Give them a hook protocol like the kits have: stats, ai, attack packets, effects, on-deaths.
   - Each table should carry its own slot range instead of the repeated literal `+ 8` epic width.
3. **One unit table.** `ModernState` columns (`ad`, `mr`, `arange`, …) are renamed into `WorldUnits`
   (`units_view`) and again into item `Units` (`_item_units`). Unit-slot writes are written out four times:
   `_spawn_minions`, `J.write_spawns`, `_apply_objective_writes` and the ward mirror.
   - Keep `s.units: WorldUnits` directly.
   - Add one `write_units(s, mask, UnitWrite)` that also resets the slot.
4. **Champion registry.** Adding a champion touches about 8 places: kit, `ChampionState`, the id dicts, skill
   orders, traits, `data/modern.py`, and the observation one-hots.
   - Use the item/rune registry pattern: a `KITS` tuple with id, name, traits, skill order and hooks.
5. **Last-tick latches.** `damage_matrix`, `death_seen`, `epic_prev`, `large_prev`, `pending_dash`,
   `reset_next`, `dyn` and `visible` carry state from one tick to the next. Group them as `s.prev` and document
   the one-tick-lag rule in one place.
6. **Naming.** Two different systems are both called "modern":
   - `sim/modern.py`: modern Garen/Jax on the legacy map.
   - `world.tick` and friends: this world.
   There are also three modern data directories. Move this world into its own package, or at least give each
   overlapping module a one-line "role / see also" header.
