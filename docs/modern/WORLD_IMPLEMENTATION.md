# WORLD_IMPLEMENTATION.md — the 26.19 modern world tick

**Status (2026-10-02).** `lanerl_jax/sim/modern_step.py` runs a modern Summoner's Rift top-lane game end to end,
with every modern system composed in one JAX tick:
- minion waves and minion/turret AI;
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
| `modern_summoners.py` | Flash, Teleport and Unleashed Teleport (incl. quest rewards), Ignite, Exhaust, Barrier, Heal, Ghost, Cleanse (agent-built). Smite is deferred. |
| `modern_step.py` | `ModernState`, `ModernOrders`, `init_state(cfg)`, `step(state, orders, cfg) -> (state, TickEvents)`. |

Units: 2 champions (unit c = holder c = team c), 40 lane-minion slots (top lane, both teams), then 22 turrets,
6 inhibitors and 2 Nexuses.

Positions come from:
- turrets and barracks: the client `base_srx.materials.bin` placements (`geometry.json`);
- inhibitors: the `SRUAP_*_Inhibitor_Idle` placements;
- fountains: `__Spawn_T1/T2`;
- **Nexus**: no placement exists in the decoded bin, so it is set at the Nexus-turret midpoint plus 350 units
  toward the fountain (INFERRED-L, recorded in `cfg.profile`).

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
11. **TIMERS / OUTPUTS.**
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
- **Cost:** about 48 ms per tick single-environment on 2 login-node CPU cores, plus about 2–4 min to compile.
  GPU cost is not measured (MODERN-007 gate; run it through Slurm).

## Known gaps and approximations

- **Champion kits:** the kit agent kept 13 old approximations, listed in `modern_champions/` docstrings and the
  agent report. Garen Q has no 26.x dash; Garen E crit is a fixed ×1.3; Garen W stacks come from any killing
  blow; Jax R resists require a champion hit; Jax E stuns minions too. Kit formulas belong to the champion
  workstream (MODERN-004).
- **Champion behaviour without orders:** champions don't auto-acquire targets; they act only on `ModernOrders`.
- **Vision:** everything is visible (the lane AI and runes get all-visible masks). Brush and fog are deferred
  (MODERN-009).
- **Lanes:** only top-lane waves spawn (both teams); other lanes' structures exist but receive no minions.
- **Collision:** unit collision uses the legacy `resolve_collisions` without the Map1 terrain grid. Terrain is
  enforced by the movement clamp; dynamic terrain is deferred.
- **Quest lane:** "in quest lane" is approximated as "not in the fountain" (ROLE_QUESTS U-RQ-1).
- **Homeguard:** the lane endpoint and jungle masks are not wired (always False). Homeguard ends only through
  combat or Teleport.
- **Ally effects:** ally-targeted item, rune and summoner effects are inert because there are no allies in the
  lane world.
- **Observations and actions:** the RL observation and action schema for the modern state is not implemented
  (MODERN-005). `ModernOrders` is the action interface.
