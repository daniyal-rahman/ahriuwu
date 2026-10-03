# WORLD_IMPLEMENTATION.md — the 26.19 modern world tick

**Status (2026-10-03).** `lanerl_jax/sim/modern_step.py` runs a modern Summoner's Rift game end to end (two
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
| `modern_world.py` | Host config `build_config(loadouts)`. It loads the Map11 terrain (per-team gate masks) and the route graph from the pinned artifacts (`/mnt/nfs/shared/modern-world-map-research/grid-26.19-base`, `/mnt/nfs/shared/WORLD001_map_routes/routes`). It sets the fixed unit layout and the structure chain, and validates loadouts (items, rune pages with client substitutions, summoners, role, skill order). |
| `modern_world_types.py` | Shared contract: `WorldUnits`, `AttackState`, `AttackLaunch`, `CastOrder`, `CCOut` (with cast ids), `Dash`. |
| `modern_mechanics.py` | Basic-attack machine (windup → launch → period; any cancel resets the timer; edge-to-edge range; tick rounding). Missiles, CC timers (tenacity applied at application with the 0.3 s floor, strongest slow, champion-CC clock), capability flags (DAMAGE §10.3), route movement with terrain clamp, Flash landing. |
| `modern_lane_ai.py` | Minion and turret targeting, Call for Help, movement goals, their attack packets, minion spawn stats, structure tick, defense and plates (agent-built, see its docstring). |
| `modern_champions/` | Garen and Jax kits on the packet/CC/stat contract (agent-built). |
| `modern_summoners.py` | Flash, Teleport and Unleashed Teleport (incl. quest rewards), Ignite, Exhaust, Barrier, Heal, Ghost, Cleanse (agent-built). Smite lives in `modern_jungle.smite_step`. |
| `modern_jungle.py` | Camps and Scuttle (spawns, AI, leash/reset, rewards, buffs), Smite, jungle pets. |
| `modern_objectives.py`, `modern_dynamic_terrain_rift.py` | Voidgrubs, Rift Herald (+ Mercenary), Drakes/Soul/Elder, Baron, team buffs; 21 Rift × Baron-pit terrain variants from client navgrid overlays. |
| `modern_wards.py`, `modern_vision.py` | Ward slots and trinkets; fog with sight radii, walls, brush, stealth and true sight. |
| `modern_dynamic_terrain.py`, `modern_map_regions.py`, `modern_item_actives.py` | Structure pads (policy: pads stay, wiki), lane/jungle/river/base regions, item actives (stasis, cleanse, dashes, …). |
| `modern_step.py` | `ModernState`, `ModernOrders`, `init_state(cfg)`, `step(state, orders, cfg) -> (state, TickEvents)`. |

Units (216, `modern_world.layout()`): 2 champions (unit c = holder c = team c), 120 lane-minion slots (40 per lane,
both teams), 48 monster slots (38 camp slots, 8 epic slots), 16 ward slots (8 per team), then 22 turrets,
6 inhibitors and 2 Nexuses. Structures come last so the fogged units are exactly the slots before them.

Positions come from:
- turrets and barracks: the client `base_srx.materials.bin` placements (`geometry.json`);
- inhibitors: the `SRUAP_*_Inhibitor_Idle` placements;
- fountains: `__Spawn_T1/T2`;
- **Nexus**: no placement exists in the decoded bin; it is the centre of its structure pad in the 26.19 navgrid
  (STRUCTURE cells, LANES_TERRAIN), about 214 units from the earlier inferred position.

## Tick order (`step`)

1. **INPUT.** Orders become intents (move, attack, stop). Skill points are spent from `level_up` or the
   champion's default order.
2. **SPAWN.** Wave units spawn per `modern_minions.spawn_event`, 0.8 s apart, both teams, into free slots, with
   `modern_lane_ai.minion_spawn_stats`.
3. **STATS.** `modern_stat_pipeline.compose` combines:
   - static items, shards and kit stats;
   - last tick's dynamic stats from `combat_tick`;
   - slows.

   Static max-HP changes are synced to current HP (STAT.70).
4. **SHOP and CASTS.** Buy and sell happen in the shop area or while dead, with rune purchase blocks applied.
   Then the kit `cast` and `periodic` hooks run, then `modern_summoners.step`.
5. **AI.** The structure tick runs, then minion and turret target selection (last tick's damage matrix feeds
   aggro). Champions attack their ordered target.
6. **MOVE.** Movement steers along the route graph with a swept terrain clamp, at minion or champion speed.
   Champion speed includes summoner MS and Homeguard. Then:
   - kit dashes (Jax Q follows its target);
   - Flash, landing on walkable terrain;
   - Teleport arrival;
   - unit collision (ghosting from Ghost or a dash).
7. **ATTACK.** The attack machine runs for every unit (kit `cannot_attack`, cast lockout, Teleport and dash
   gate it). Champion attacks:
   - crit roll at launch (Bernoulli, X-8), with item forced crits;
   - attacks on structures use the champion structure formula.

   Minion and turret attacks come from `modern_lane_ai.attack_packets`. Ranged attacks become missiles, which
   hit on arrival; champion missile arrivals count as on-hit. Kit `on_attack` and `on_hit` run, and turret
   Crystalline Overgrowth is consumed.
8. **DAMAGE.** World, kit and summoner packets go to `modern_combat.combat_tick`, which runs items and runes
   around `modern_damage`. The tick supplies:
   - defense: champion armor/MR, turret resists with Bulwark, the backdoor ×0.2 on every damage type,
     untargetable structures, Teleport invulnerability, the kit DR/dodge/AoE reduction and Garen E shred;
   - offense: turret 30% armor pen, Exhaust as dealt reduction;
   - `RuneEvents` from world state: windup start/cancel/reset, cast ids, CC with cast ids, impairments,
     summoner casts, blinks, Flash cooldown, deaths, purchases, grants, turret mask.
9. **CC / HEAL.** Kit and summoner heals and shields are applied (HSP, incoming heal, Grievous Wounds). Kit CC
   and the Exhaust slow are applied with tenacity (stat + Garen W + Cleanse); Cleanse removes CC.
10. **DEATH / ECONOMY.**
    - The killer per unit comes from the resolved packets.
    - Minion deaths go out as `MinionDeaths` (last hitter, gold/XP/level fixed at spawn).
    - Plates and destruction come from `structure_damage_events`.
    - `modern_economy.economy_step` handles gold, XP, levels, bounty, kill credit, death timers, respawn,
      Recall, Homeguard and the quest.
11. **TIMERS / OUTPUTS / FOG.**
    - **Fog:** attack-reveal circles open, then `visible`/`sight` are recomputed from final positions for the
      next tick and the observation (`modern_vision`).
    - **Cooldowns:** kit cooldowns start hasted (basic vs ultimate haste), rune refunds apply, and timers
      count down.
    - **Mana:** costs are paid and regen applied.
    - **HP:** regen in 0.5 s pulses, plus the fountain and the Homeguard heal.
    - **Respawn and Recall:** both move the champion to the fountain.
    - **Inventory:** Tear transforms, consumption, and rune item grants into a free slot (acknowledged next
      tick).
    - **Dtypes:** the state is cast back to the incoming dtypes so `lax.scan` carries stay stable.

## Verified

- **`tests/test_modern_step.py`** covers:
  - layout and starting structure vulnerability (only outer turrets);
  - 100 s with no orders: passive gold equals `500 + ambient_payments`, both teams' waves are alive, no packet
    or missile overflow;
  - a shop purchase in the fountain (gold and inventory);
  - Flash: a blink of 300–400 units and a 300 s cooldown after the 15 s start cooldown;
  - a champion kill: first-blood gold, the death timer from `death_time`, respawn at the fountain;
  - Recall back to the fountain after 8 s.
- **`tests/test_modern_mechanics.py`** covers the attack cadence (launch at 0.3 s, then every 1.0 s at 30 Hz),
  edge-to-edge range, cancel reset, missile homing, CC tenacity (DAMAGE F17) and strongest slow.
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
    of memory: `route_next` connection checks dominate per-env memory. Fog costs about 5% on GPU. Details in the MODERN-021 ledger row. Measure with `ops/modern_world_bench.py` and attribute with
    `ops/modern_world_profile.py`.

## Integration added in MODERN-020

- **INPUT:** attack-move orders (`ModernOrders.attack_move`), item stasis gating, the tick's terrain (Rift variant,
  then structure pads), all-lane and camp spawns.
- **STATS:** jungle buffs (Blue, Red, shrine MS) and team objective buffs (drake stacks, soul, Hand of Baron) are
  static stat bonuses; then `objectives_step` (epic spawns, abilities, Rift transformation).
- **CASTS:** Smite (`modern_jungle.smite_step`).
- **AI:** monster and epic-monster AI, Hand of Baron minion empowerment, champion idle acquisition
  (`LA.idle_acquire`, acquisition range 400, chases) and attack-move (`LA.attack_move_step`). Team-owned summons
  (Mercenary, Hunger Voidmites) are minion-like targets for minions and turrets.
- **MOVE:** monsters move with their AI goals; terrain comes from the Rift variant and the structure pads;
  first-wave minions are ghosted; wards don't collide; the Rocketbelt dash starts the tick after its active.
- **ATTACK:** jungle and epic monster attack packets replace the lane-AI packets on their slots; empowered siege
  minions deal their Baron multiplier to structures; champion hits on wards become ward hits (1 HP each).
- **DAMAGE / CC:** Smite, jungle (Red burn and slow, pets) and objective packets; `objectives_packet_mods`; Baron's
  armor/MR shred; stasis invulnerability; monster CC is not champion-sourced; Scuttle slow immunity and −100%
  tenacity; jungle and objective heals, mana and shields; item-active cleanse.
- **DEATH:** camp and objective rewards go to the economy (`EconomyInputs.extra_gold/extra_xp/epic`) and to the
  rune events (epic takedowns, large-monster kills) on the next tick. Map regions supply the quest lane, Homeguard
  endpoint and jungle flags, and the river flag. Hand of Baron gives the 4 s Empowered Recall.
- **TIMERS:** item-active mana-cost and cooldown-rate effects, Armguard transform, pet egg consumption, `ward_step`
  (ward gold, Control Ward use), ward unit writes, ejection from closed terrain, fog with wards and the Rift brush,
  game end (`LA.game_result`; the world freezes once a Nexus falls).
- **Tests:** `tests/test_modern_world_rules.py` checks waves in all lanes and camps at 1:30, symmetric idle minion
  trades, a ward revealing an enemy in brush, camp kill rewards, attack-move acquisition and the game-end freeze.

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
- **Collision:** unit collision uses the legacy `resolve_collisions` without the terrain grid; terrain is enforced by
  the movement clamp and `modern_dynamic_terrain.eject`.
- **Observations and actions:** profile `modern-world-v1` (MODERN-005).
  - `lanerl_jax/obs/modern_builder.py` returns `ModernObservation`: 32×20 entity slots (legacy 16 columns, a
    neutral-team column that is now used, then monster, epic monster, ward, Control Ward), the 6-column global
    vector and a 32-column `self` block (stats, cooldowns, mana, shield, summoners, level progress, quest, unspent
    skill points, trinket charges), plus `inventory`, `inventory_stack` and `affordable` (exactly the shop's
    `buy` check per catalog row). The enemy cast memory uses `ChampionLayer.seen_cast` (casts the enemy team saw).
  - `lanerl_jax/train/modern_actions.py` decodes `(button, sx, sy[, choice])` into `ModernOrders` with 19 buttons:
    the legacy eight, `summoner_d/f`, `level_q..r`, `buy` (catalog row), `sell` / `use_item` (inventory slot),
    `ward` (trinket at the cursor) and `control_ward`. `attack_move` without a hit is a real attack-move order.
    `Loadout(auto_skill=False)` leaves skill points to the policy.
  - Not wired into a trainer or reward yet.

## Structural debt (MODERN-021 review)

These are next refactors, in order. None of them changes behaviour.
1. **Phase functions.** `step` is about 650 lines with about 100 locals carried across phases.
   - Split it into `_input`, `_stats`, …, `_fog` phase functions. Each takes `(s, cfg, scratch)`, where `scratch`
     is a `TickScratch` NamedTuple of the values passed between phases: the stat snapshots, caps, kit/summoner
     outputs, desired targets, packets, CC and rewards.
   - Its fields are then the tick's data-flow map, and each phase can be tested and compiled on its own.
   - Do it one phase at a time, using the step and world-rules tests as the safety net.
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
   - `modern_step` and friends: this world.
   There are also three modern data directories. Move this world into its own package, or at least give each
   overlapping module a one-line "role / see also" header.
