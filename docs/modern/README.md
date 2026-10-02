# docs/modern — implementer specs for the 26.19 modern world

**Patch pin:** normal PC Summoner's Rift (CLASSIC), patch 26.19, client build **16.19.8230722**.
**Researched:** 2026-10-01. **Status (2026-10-02):** implemented — see the `*_IMPLEMENTATION.md` docs; the modern world tick is `lanerl_jax/sim/modern_step.py`.

These specs replace the old C# server as the reference for the modern JAX port. The C# server stays the
regression oracle for the legacy ruleset only (ledger MODERN-006). `docs/MODERN_PATCH_DELTA.md`
(26.18) is now background. Where these specs correct it, the corrections are listed in each doc.

## Index

| Doc | Owns | Size |
|---|---|---|
| [DAMAGE_AND_STATS.md](DAMAGE_AND_STATS.md) | **Global contract**: stat pipeline, resist/pen, damage packet pipeline, shields, healing/GW/vamp, attack timing and crit, MS, haste, tenacity, regen, max-HP rule, the **hook taxonomy** (`STAT.*`, `DMG.*`, `HEAL.*`, `SHIELD.*`, `TICK.*`) | 662 lines, 32 fixtures |
| [MINIONS.md](MINIONS.md) | Minion stats and upgrade formula, wave schedule and composition, spawn, movement, targeting/aggro/Call for Help (post-26.10), damage ratios, Minion Pushing, gold/XP/CS, per-tick AI pseudocode | 715 |
| [TOWERS.md](TOWERS.md) | All structures: per-tier stats, kill order, targeting AI, Warming Up, shots on minions, plating/Bulwark, outer decay, **exact Crystalline Overgrowth formula**, backdoor DR, inhibitors/Nexus, rewards | 850 |
| [ITEMS.md](ITEMS.md) | Inventory and shop, uniqueness groups, shared named effects (Spellblade, Lifeline, ...), vamp rules, **Tiamat line and Stridebreaker in full**, item hooks, 26.x item changes | 1094 |
| [ITEMS_CATALOG.md](ITEMS_CATALOG.md) | Per-item entries for 214 SR items (210 store + 4 transform), generated from client data, with hand-written hooks and notes | 2195 |
| [ITEMS_IMPLEMENTATION.md](ITEMS_IMPLEMENTATION.md) | **Implemented** item system: layout, coverage, tick order, what the world integrator must still apply, known gaps | — |
| [RUNES_IMPLEMENTATION.md](RUNES_IMPLEMENTATION.md) | **Implemented** runes, shards, STAT/DMG modifier ordering and the combined item+rune tick (`modern_combat`): layout, coverage, tick order, integrator duties, known gaps | — |
| [RUNES.md](RUNES.md) | Page legality, stat shards, every rune in all five trees (incl. Stormraider's Surge, which replaced Phase Rush in 26.9 under id 8230), automatic rune swaps, rune hooks | 1256 |
| [SUMMONER_SPELLS.md](SUMMONER_SPELLS.md) | Flash, Teleport/Unleashed Teleport (role-quest interaction), Ignite, Exhaust, Barrier, Heal, Ghost, Cleanse, Hexflash; Smite listed and deferred | 551 |
| [WORLD_IMPLEMENTATION.md](WORLD_IMPLEMENTATION.md) | **Implemented** modern world tick (`modern_step`): unit layout, tick order, how minions/turrets, kits, summoners, items, runes, economy and quest compose; verified symptoms; gaps | — |
| [ECONOMY_IMPLEMENTATION.md](ECONOMY_IMPLEMENTATION.md) | **Implemented** economy/progression + Top quest, and the **replay oracle**: 145 real 16.9 games checking gold, death timers, kill/assist gold, fountain, level-up HP (spec corrections listed) | — |
| [ECONOMY_PROGRESSION.md](ECONOMY_PROGRESSION.md) | Starting/ambient gold, XP curve to level 20, minion and kill XP sharing, comeback XP, kill credit/assists, 2026 bounty system, structure gold distribution, level-up, death timers, respawn, recall/Homeguard, fountain/shop | 310 |
| [ROLE_QUESTS.md](ROLE_QUESTS.md) | Role binding, Top quest point sources and thresholds, completion event, rewards (level cap 20, XP, Teleport upgrade/shield), other roles briefly | 234 |

Every doc follows the same layout: sources (URLs, wiki `oldid`s, sha256 of cached client files) →
rules tagged `CLIENT` / `RIOT` / `WIKI` / `INFERRED` with confidence → state to carry → events/hooks →
diff vs current code (file:line) → unresolved register with a test scenario for each → numeric test fixtures.

## Evidence locations

- Client data (CommunityDragon `16.19` = build 16.19.8230722):
  `/mnt/nfs/shared/modern-world-map-research/cdragon-16.19/`, checksums in `SHA256SUMS`. This holds the
  items/perks/globals/shared bins, minion/turret/inhibitor/Nexus/fountain/Garen/Jax character bins,
  `map11.bin.json`, `modespecificdata-classic.bin.json`, the en_us string table and the role-quest bins.
- Archived Riot patch notes 26.1–26.19 and wiki pages:
  `/mnt/nfs/shared/modern-world-map-research/{patch-notes-26.x,econ-notes-26.x,econ-wiki,wiki-damage-stats}/` and
  `cdragon-16.19/{riot-patchnotes-26.x,wiki-2026-10-01}/`.
  - The towers researcher's wiki and notes copies are in `towers-research-sources/`, copied from
    `/tmp/towers_research/`. Its oldids are cited inline in TOWERS.md.
- The normal-SR mode block in client data is `GameModeMapData {0b03bf5a}` → `GameModeConstants {6cf687be}`
  (identified by the mode-name hash "Classic"). Many 26.x patch-note sections are **League Classic / Swiftplay /
  ARAM: Mayhem / Arena only**. Every doc filters these out, e.g. 26.16 death timers and Homeguard,
  26.17/26.19 Call-for-Help fixes, 26.15 damage modifiers, Mercury's 35%.

## Hook crosswalk

`DAMAGE_AND_STATS.md` §2 is the canonical taxonomy. ITEMS and RUNES were written in parallel with
their own local names. Map them like this:

| ITEMS.md §10 | RUNES.md §9 step | Canonical slot |
|---|---|---|
| H1 `stat_calc`, H2 `stat_calc_dynamic` | 1 stat composition | `STAT.20`–`STAT.50`, then `STAT.60`/`STAT.70` |
| H3 `on_ability_cast_start`, H4 `on_attack_windup_complete`, H16 `active` | 3 action/cast phase | `TICK.00_INPUT`, `TICK.40_CAST_RESOLVE`, `TICK.50_ATTACK` |
| H5 `on_basic_attack_hit` | 4.1 raw + on-hit procs as separate packets | `DMG.00_DECLARE` (one packet per proc) |
| — | 4.2 outgoing amps (PTA, CdG, Cut Down, Last Stand) | `DMG.40_DEALT_MOD` (additive sum) |
| H7 `on_pre_damage_taken` (% parts: Steelcaps ×0.9, Randuin's) | 4.3 received modifiers | `DMG.60_RECEIVED_MOD` |
| H7 (flat parts: Warden's Mail −15) | 4.4 Bone Plating | `DMG.70_POSTMIT_FLAT` |
| H8 `on_lifeline_check` | — | between `DMG.75_FINAL` and `DMG.85_HEALTH` (Lifeline); `DMG.95_DEATH` (Guardian Angel) |
| — | 4.5 shields then HP | `DMG.80_SHIELD`, `DMG.85_HEALTH` |
| H6 `on_damage_dealt`, H9 `on_damage_taken` | 4.6 / 4.7 | `DMG.90_ON_DAMAGE` (vamp at `DMG.92_VAMP`) |
| H10 `on_kill` | 4.8 death, takedown | `DMG.95_DEATH`, `DMG.98_ON_KILL`; economy distribution after |
| H11 `periodic`, H13 `on_level_up` | 2 timers | `TICK.10_BUFFS`, `TICK.20_REGEN`, `TICK.90_TIMERS` |
| H12 `on_move` | 5 movement | `TICK.30_MOVE` |
| H15 `on_shop_area_enter`, H17 shop actions | — | `TICK.00_INPUT` (shop actions are orders) |

Default order for procs from a single attack (RUNES U-18): main hit → item on-hits → rune on-hits.

## Cross-doc decisions (provisional defaults; each is also an unresolved item)

| ID | Question | Default | Why |
|---|---|---|---|
| X-1 | Level-scaled values at levels 19–20 (top quest cap 20) | `ByCharLevelInterpolation` **extrapolates** `(L−1)/17` past 18, except parts with `mScalePastDefaultMaxLevel=false`. Breakpoint (`level_bp`) formulas continue. Fixed 18-entry arrays (base kill gold, death-timer BRW) **clamp** to their last entry. Champion stat growth continues (`G(20)=19.665`). | The flag is absent on 26/27 perk and 61/62 item calc parts and explicitly `false` only on First Strike's cooldown and mode item 772038, so the engine default must be "scale past". The wiki renders rune formulas "for 20". ITEMS.md was aligned to this on 2026-10-01. |
| X-2 | Where the minion→champion 0.55 / →structure 0.60 ratio applies | Once, at `DMG.45_UNIT_CLASS`, on raw minion AD before resists. It is **not** baked into minion AD. | MINIONS §4 (client `dr_UnitToHero`) resolves DAMAGE U-05. Siege→turret is 0.60 × the 1.4 siege bonus (MINIONS fixture 19.6875). |
| X-3 | Turret shots on minions | % of max HP taken **before armor**: melee 45%, caster 70%, cannon 14/11/8% by tier, super 7% | TOWERS and MINIONS agree; current code uses 5% for supers and applies armor. |
| X-4 | Life steal/omnivamp on damage a shield absorbed | Vamp reads `dmg_final` (`DMG.75`, **before** shields). ITEMS H6 places vamp "after the target's shields"; read that as hook position only, not the amount. | DAMAGE §7 / U-07. Measure: life steal attacking a shielded target. |
| X-5 | Combining damage modifiers | Source-side amps (runes, items, Exhaust) **sum** at `DMG.40`; target-side modifiers **multiply** at `DMG.60` | DAMAGE §2.2 (wiki notes 26.09 change, marked untested); RUNES §1.4 agrees. Current `modern_stats.py:115-128` multiplies everything. |
| X-6 | Stat shards | Adaptive 9 (5.4 AD); AS 10%; AH 8; **MS 2.5%**; HP 65; **scaling HP 10·L** (180 at 18, 200 at 20 per X-1); **tenacity and slow resist 15%** | Client perks; DAMAGE and RUNES agree. Current `modern_items.py:207-219` uses 2% and 10%. |
| X-7 | Server tick | 30 Hz; timers rounded to ticks per DAMAGE §1 | DAMAGE U-01; affects windups (Garen windup modifier 0.5), Overgrowth timing, quest ticks. |
| X-8 | Crit | Base crit damage 2.0 (26.1); IE +0.30; plain random roll until the pseudo-random table is known | DAMAGE §8 / U-09. `autoattack.py:286` crits on every attack whenever crit chance > 0. |

## Highest-impact code discrepancies (from the per-doc diff sections)

1. **Turret and minion systems are not wired in.** Nothing calls the turret targeting, shot, damage or reward
   functions, or the minion upgrade, Minion Slayer and damage-ratio helpers. `modern_world.py` spawns legacy-HP
   minions. Nexus turrets are targetable at game start, and inhibitor/Nexus units are missing (TOWERS D4/D10, MINIONS §9).
2. **The legacy aggro priority is still live in modern mode.** `targeting.py:140` keeps the "champion attacks
   allied minion" entry that 26.10 removed (MINIONS §3/§9).
3. **Overgrowth maximum curve:** `modern_towers.py:132` interpolates linearly. At outer level 9 it gives 957.7;
   the exact value is 882.3 (TOWERS §6, D1).
4. **Negative resists are clamped to 0:** `modern_stats.py:88,100` and `modern_towers.py:174`. The existing
   unit test asserts the wrong answer (DAMAGE D1).
5. **Item stats come from Data Dragon and description text** (`modern_items.py:40-85`). This drops lethality,
   penetration, omnivamp, HSP, crit damage and slow resist, and it sums tenacity instead of multiplying it.
   Only the Hydra uniqueness group is enforced. Hydra and Stridebreaker active geometry is wrong. Switch to
   `items.cdtb.bin.json` (ITEMS §14).
6. **Economy is still legacy Map1:** ambient gold 0.95/0.5 s from 90 s, 4.20 streak bounty, 1600-radius equal
   minion-XP split, 19-row level tables capped at 18, no game-time death factor, legacy fountain
   (ECONOMY §15). Role quests, Homeguard, summoner spells and Teleport state do not exist.
7. **Small value bugs:**
   - Siege/super gold `50+U` should be `49+U` (`modern_minions.py:241`).
   - Melee count is wrongly tied to cannon presence (`:162`).
   - Spawn spacing 0.792 s should be 0.8 s (`:93`).
   - Windup ignores the champion windup modifier (`modern.py:74`).
   - Attack range adds only the target radius (`autoattack.py:148`).

## Top unresolved items, by impact on 1v1 top-lane fidelity

All of these need modern client observation (MODERN-006). Each doc's register gives the exact Practice
Tool or custom-game scenario.

1. Minion target selection: type preference vs pure distance, re-evaluation cadence, how long attacker memory
   lasts, Call-for-Help radii (MINIONS §8).
2. Turret radii: "enemy minion nearby" for backdoor DR (default 1000) and the Overgrowth deferral radius;
   whether the range check is edge-to-edge (TOWERS §12).
3. Damage-packet order within one tick, and the tick rate itself (DAMAGE U-01/U-02).
4. Whether minion stats are fixed at spawn or upgrade while the minion is alive; move-speed step times (MINIONS).
5. Level 19–20 behaviour (X-1) for runes, items, kill gold and death timers.
6. The Top quest "in lane" area and how often passive quest points tick; whether top loses 25% minion gold/XP off-lane before level 3 (ROLE_QUESTS §9).
7. Death-timer scaling start: 10:00 or 15:00 (ECONOMY U-E-*; default 15:00).
8. Teleport travel time and interrupted-channel cooldown (SUMMONER_SPELLS §15).
9. Whether life steal applies to shield-absorbed or overkill damage (X-4).

## Suggested implementation order

It follows MODERN-009 priorities and the dependencies between these specs:

1. Shared math (DAMAGE).
2. Minions (MINIONS).
3. Turrets (TOWERS).
4. Economy (ECONOMY).
5. Items, stats and Tiamat/Stridebreaker (ITEMS).
6. Runes and shards (RUNES).
7. Summoner spells and Teleport (SUMMONER_SPELLS).
8. Role quest (ROLE_QUESTS).

The fixtures in each doc are written to become unit tests directly. Out of scope and still deferred
(MODERN-009): dynamic terrain, full vision/Faelights/wards, jungle and objectives, jungler items, and item
actives other than the Tiamat line and Stridebreaker.
