# ECONOMY_IMPLEMENTATION.md — 26.19 economy, progression and Top quest as implemented

**Status (2026-10-01).** The economy is implemented as pure, fixed-shape JAX. It covers gold, XP, levels, kill
credit, the bounty system, structure gold, death timers, respawn, recall, Homeguard and the fountain. The Top
role quest is implemented too.

Its rules were checked against **real games**. The source is client-memory recordings of 145 games
(patch 16.9, i.e. 26.9; no Summoner's Rift economy change between 26.9 and 26.19). Wherever those recordings
disagreed with the spec, the spec was wrong, and it has been corrected. The world tick (`modern_world.py`) does
**not** call `economy_step` yet.

The rules this implements are in [ECONOMY_PROGRESSION.md](ECONOMY_PROGRESSION.md) and
[ROLE_QUESTS.md](ROLE_QUESTS.md).

## Layout

| File | Role |
|---|---|
| `lanerl_jax/data/build_modern_economy.py` | Host tool. Follows the cached Map11 bin's CLASSIC `GameModeMapData` to its experience curve, experience-mod data, death times and the kill-gold/bounty config `{4fd6b68d}`. Also reads the CLASSIC game-mode constants, and records source sha256s. |
| `lanerl_jax/data/modern/26.19/economy_client.json` | Pinned tables: XP to level 20, kill XP, shared-kill and level-difference multipliers, the minion XP split, death times and scaling points, base kill gold, first blood, the assist window, bounty constants, and gold/fountain constants. |
| `lanerl_jax/sim/modern_economy.py` | See the function list below. |
| `lanerl_jax/sim/modern_role_quest.py` | Event points (minion 2, plate 40, turret 50, takedown 15, epic 30) and the out-of-lane factor `0.25 + 0.75p`. Passive points: 1/3 per s anywhere, 1.5 per s in lane, from 65 s. The roam bank caps at 5 s, or 60 s from level 3, filling at 0.5 s/s. Points stop for 12 s after a recall. Completion at 1200 points gives level cap 20 and +600 XP. After completion: +11% non-takedown XP, +80 XP per takedown, and −25% minion gold/XP outside the lane before level 3. Teleport rewards: the Unleashed Teleport cooldown formula (−30 s with the quest), the free 390 s Teleport, and the 35% max-HP arrival shield. |
| `ops/replay_oracle_extract.{py,sbatch}` | Extracts raw observations from the replay corpus as a Slurm array job. It never reads simulator code. |
| `lanerl_jax/data/modern/oracle/` | `replay_16_9_observations.json.gz` (145 games: gold increments, 7,589 deaths with payout windows, 19,068 fountain stretches, 19,894 level-ups, max-HP changes; sha256 of every source file). `champion_hp_16_9.json` (16.9 HP base and growth of the 169 champions in those games, from CommunityDragon 16.9). |

`modern_economy.py` provides:
- **Gold and levels:** `ambient_payments`, `level_for_xp`, `decimal_level`, `skill_points`, `max_rank`.
- **Minion rewards:** `minion_rewards` (1500 radius plus the last hitter, the split table, the comeback bonus,
  last-hit gold).
- **Kill credit and XP:** `Credit`, `credit_update`, `kill_credit` (15 s window), `kill_xp` and `kill_xp_eligible`.
- **Bounty:** the `Bounty` state machine, `kill_gold`, `assist_gold`, `champion_kill`, accrual, the 5 s deferral,
  shutdown carry-over and the positive buffer.
- **Structures:** `structure_gold` and `structure_eligible`.
- **Lifecycle:** `level_up_sync`, `death_time`, `respawn_due`, `deathguard_ms`, `recall_step`, `homeguard_step`
  and `homeguard_bonus_ms`, `fountain_regen`.
- **Reference tick:** `economy_step`.

## Checked against real games (`tests/test_modern_economy_oracle.py`)

| Rule | Evidence | Result |
|---|---|---|
| Starting gold 500 | all 1,450 champion-games | ✓ |
| Passive gold | 223 clean early-game samples | **Spec corrected.** The first payment is at **65.0 s**, not 65.5 s (U-E-1). Error ±0.04 g. The spec's phase was off by one payment (1.02 g) in every sample. |
| Death timer table, levels 1–18 | 6,025 deaths | ✓ exact. For example, level 1 is 9.985 s observed (sampled) vs 10 s; level 9 is 27.98 s vs 28 s. |
| Death-time scaling | deaths after 15:00 | **Spec corrected.** Scaling accrues **continuously**: 94% of deaths fall within 0.1 s, against 47% for the wiki's 30 s steps. There is no scaling at 10:00–15:00 (U-E-7 resolved). |
| Death timer at levels 19–20 | 13 deaths | Equal to the level-18 value (U-E-6 resolved). |
| First blood | 145 games | ✓ The killer gets base + 100 = 400. |
| Assist gold | 145 first bloods; 4,589 later assisted kills | **Spec corrected:** first-blood assisters share half the +100 too (pool `(min(0.5K, 0.5 base) + 0.5 FB) × early(t)`). Confirmed unchanged: the 50%-of-base cap on later kills, including shutdowns, and the 55–175 s early factor. |
| Fountain after a recall | 2,311 stretches | ✓ +2% max HP every 0.25 s plus Homeguard's 8% of missing HP every 0.5 s. That fits better than every other rate tried, and without the Homeguard term the error is more than 3× worse. |
| Level-up current HP | 7,428 level-ups | ✓ The full max-HP increase, with no missing-health penalty. 98.7% are exact at full HP; a penalty model fits 3× worse. |
| Growth curve G(L) and the scaling-HP shard | 19,887 level-ups across 169 champions | ✓ 90.8% are exact to 0.15 HP with 0, 10 or 20 HP of shard growth. The rest come from kits and items that change max HP (Ornn, Kled, Sion, Gnar…). Linear growth explains under half as many. |
| Level cap 20 | 65 level-ups past 18 | Only top-lane champions pass 18 (Garen, Jax, Kayle, Trundle, Gangplank, Illaoi, Ambessa, Rumble, Yone, Renekton, Tryndamere), and none pass 20. This is consistent with the Top quest. |

Fixture tests in `tests/test_modern_economy.py` cover the spec's own §18 and ROLE_QUESTS §10 fixtures (with the
corrections above) and a jitted `economy_step`.

## Checked against Riot match-v5 timelines (`tests/test_modern_stats_riot_oracle.py`)

The same 147 games' Riot match-v5 match and timeline data (fetched 2026-10-01 with a dev key; the key is
never stored) give per participant-minute Riot's `championStats`, XP, level and gold, plus every kill with its
`bounty`. The anonymised extract `oracle/riot_16_9_frames.json.gz` (`ops/riot_stats_oracle.py`) has no
player names or PUUIDs, and covers 43,470 participant-minutes and 8,162 kills.

| Check | Result |
|---|---|
| XP-to-level table, cap 18/20 | **100%** of 43,470 participant-minutes, including XP banked past the level-18 threshold at cap 18 |
| Base kill gold by victim level, +100 first blood | 98.3% of 466 clean first deaths pay exactly `base[V]` (+100). The misses are minute frames lagging a level-up. |
| Champion stats through `modern_stat_pipeline` (16.9 champion and item records, shards, rune `stats` hooks, item `dynamic_stats` hooks) | Exact integer match on 42,000 participant-minutes: % armor pen .986, magic pen .956, omnivamp .949, life steal .917, tenacity .816, MR .751, armor .740, AP .737, AD .663, AS .632, HP .582, MS .509. Non-support max HP .72, jungle max HP > .9. |
| Do the rune hooks move predictions toward Riot's values? | Yes. With runes, move speed matches +8 points more often (Celerity, etc.), and AD and armor also improve. |
| Do the item passives? | Yes. Rabadon's +30% AP stops being a mismatch cluster once the item `dynamic_stats` hook is included. |
| Sizes of the remaining dynamic max-HP gaps | They match the simulator's implementations. Biscuit Delivery gives +30 per biscuit (residuals of exactly 30/60/90 on 1,700+ frames), Legend: Bloodline +85, Overgrowth +3 per tier, Rod of Ages +100. |

Riot reporting conventions found along the way:
- Stats are truncated to integers.
- `attackSpeed` is reported as 100 × (1 + bonus AS).
- `abilityHaste` and flat `armorPen` are always 0, so they are unusable.

What remains unmatched is dynamic or outside the stat pipeline:
- champion passives (Garen W stacks, Jhin, Vladimir, Veigar …);
- stacking items (Mejai's, Rod of Ages, Riftmaker);
- support-item quest upgrades, which have no purchase event;
- rune stacks that this snapshot cannot see.

**What the recordings cannot check.**
- Current gold is unreadable in this corpus.
- XP isn't recorded, only levels.
- Bounty amounts beyond first blood and assists depend on hidden farm history.
- The quest's internal points aren't visible.

The Riot match-v5 timelines (above) cover XP and the stat pipeline. They cannot cover current gold, quest
points, or bounty history beyond first deaths.

## What the world integrator must still do

**Build the inputs.** Each tick, fill `EconomyInputs`:
- this tick's combat report (for kill credit) and CC;
- champion HP after combat;
- final-blow attribution;
- `MinionDeaths`, with gold and XP from `modern_minions` (`gold_bounty`, XP per type) and the minion level at spawn;
- `StructureEvents` (plate and turret gold from `modern_towers`);
- combat clocks (`modern_combat` `last_champion_combat`);
- fountain, quest-lane, jungle and Homeguard-endpoint masks (geometry, MODERN-011);
- recall and Teleport events.

**Apply the outputs.**
- `respawned`: move to the fountain with full HP and mana, and add Deathguard MS.
- `recalled`: teleport to the fountain.
- Add `homeguard_ms` to bonus MS.
- `levels_gained`: run the stat refresh with `level_up_sync`; skill points are `skill_points(level)`.
- Feed `kills` into the next `combat_tick`.
- Apply `fountain_regen` each tick.

## Known gaps

- **Not reproducible:** team-level shutdown suppression and objective bounties depend on hidden team-advantage
  weights, so they are disabled (ECONOMY §6.2.7).
- **Assists from support effects:** heals, shields and buffs given to the killer don't count toward assists,
  because there are no allies in the lane 1v1.
- **Victim depreciation** uses kill + assist gold excluding first blood. This has not been checked against the
  recordings (the inputs are hidden).
- **Quest geometry:** the "in lane" polygon, roam-bank details and passive-point cadence (U-RQ-1…4) are the
  spec defaults, because quest points aren't visible in the recordings.
- **Homeguard timing:** the decay shape, the endpoint, the 8 s lockout, and the phase of the heal pulse relative
  to arrival (U-E-9).
