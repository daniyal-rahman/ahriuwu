# ECONOMY_PROGRESSION.md — gold, XP, levels, death, recall, fountain, Homeguard (patch 26.19)

**Scope.** Normal PC Summoner's Rift (`CLASSIC`, Map11), patch **26.19**, client build
**16.19.8230722**. Covers: starting/ambient gold, XP curve and sharing, champion-kill gold/XP and
the bounty system, minion/turret/plate reward *distribution* (per-unit values belong to the
minion/tower specs), objective bounties (brief), level-up, level caps, death timers, respawn,
Recall, Homeguard, fountain/obelisk, shop. Role-quest rewards: see `docs/modern/ROLE_QUESTS.md`.
**Retrieval date:** 2026-10-01. Implemented and checked against real games: see [ECONOMY_IMPLEMENTATION.md](ECONOMY_IMPLEMENTATION.md) (corrections marked ORACLE below). Supersedes `docs/MODERN_PATCH_DELTA.md` §9 (pinned
to 26.18), which is re-verified below (§12).

## 0. Sources

| ID | Source | Pin / cache (`/mnt/nfs/shared/modern-world-map-research/…`) |
|---|---|---|
| C0 | `classic-constants.json` = map11.bin `{6cf687be}` (the CLASSIC `mGameModeConstants`; byte-identical, verified) | existing cache |
| C1 | CDragon 16.19 `game/data/maps/shipping/map11/map11.bin.json`: `Maps/Shipping/Map11/Modes/CLASSIC` → `mExperienceCurveData {c6f6a84c}`, `mExperienceModData {0059202c}`, `mDeathTimes {9de5b46c}`, `DefaultRespawnPoints {5ec67f3f}`, `mGameplayConfig {a6506f8a}`, Configs `{4fd6b68d}` (kill gold/bounty), `{faa893bb}` | `cdragon-16.19/map11.bin.json` sha256 `0fe00959…50cd97` |
| C2 | CDragon `shared.cdtb.bin.json` (Recall/Homeguard/Teleport spell objects) | `cdragon-16.19/shared.cdtb.bin.json` sha256 `34f68553…62726` |
| C3 | CDragon `items.cdtb.bin.json` (`Items/Spells/Recall`) | sha256 `6880f35d…fd82da09` |
| C4 | CDragon `game/data/characters/sruap_turret_order5/sruap_turret_order5.bin.json` (Nexus Obelisk) | `cdragon-16.19/sruap_turret_order5.bin.json` sha256 `94f04c3f…71675` |
| C5 | CDragon `en_us/lol.stringtable.json` | `cdragon-16.19/en_us-lol.stringtable.json` sha256 `8c051cb2…8e8` |
| C6 | `lanerl_jax/modern/data/26.19/minions.json` (`BarracksConfig.ExpRadius 1500`, `goldRadius 1250`) | repo |
| R* | Riot notes 26.1–26.19 (URLs in `econ-notes-26.x/26-N.url`; 26.1–26.3 use `/patch-26-N-notes/`, others `/league-of-legends-patch-26-N-notes/`) | `econ-notes-26.x/*.txt` + SHA256SUMS |
| W* | Wiki revisions (all `https://wiki.leagueoflegends.com/en-us/<Title>?oldid=<id>`): Experience (champion) **4053165**; Gold **4039269**; Champion gold bounties **4040646**; Kill **4053216**; Assist **4016680**; Death **4051174**; Recall **4015108**; Spawn (Fountain) **3994097**; Homeguard **4011966**; Teleport **4065272**; Shop **3982704**; Nexus Obelisk **4015143**; Objective bounties **3939821**; Turret **4070072**; Minion **4068797**; Champion statistic **4069636**; Champion ability **4062619** | `econ-wiki/*.wiki` + SHA256SUMS |

Tags: **CDV** client-data verified, **RN** Riot notes, **WIKI**, **INF** inferred; confidence H/M/L.
Hashed bin fields (`{xxxxxxxx}`) could not be resolved from CommunityDragon hash lists (2026-10-01
master); where a hashed value is mapped to a meaning by matching wiki/RN numbers it is tagged
**CDV-value/INF-name**.

---

## 1. Starting gold
| Rule | Tag |
|---|---|
| Each champion starts with **500 gold** (`ai_StartingGold = 500.0`). | CDV (H), WIKI |
| Gold cap 100000 (`Gold_Max`). | CDV (H) |

## 2. Passive ("ambient") gold and XP
| # | Rule | Tag |
|---|---|---|
| 2.1 | Amount `ai_AmbientGoldAmount = 10.2` per `ai_AmbientGoldInterval = 5.0` s ⇒ **2.04 g/s = 20.4 per 10 s**. Paid in **0.5 s ticks of 1.02 g** (wiki "takes place every 0.5 seconds"). | CDV amount/interval (H); cadence WIKI (M) |
| 2.2 | Starts at **65.0 s** (`mission_AmbientGoldStartTime = 65.0`). RN 26.1 "Ambient Gold Start Time: 65 seconds". | CDV (H) |
| 2.3 | Flat for the whole game; no time scaling on SR. | CDV (no scaling constant) + WIKI (H) |
| 2.4 | Paid while dead (no `ai_DisableAmbientGoldWhileDead` override in the CLASSIC constants; legacy default false). | INF (M) |
| 2.5 | Ambient XP = 0 (`ai_AmbientXPAmount` unset). | CDV (H) |
| 2.6 | First payment **at 65.0 s** itself, then every 0.5 s. **Measured** (replay oracle, 145 games, 16.9): lifetime gold before other income equals `500 + 1.02·(⌊(t−65)/0.5⌋+1)` to ±0.04 g; the 65.5 s phase is 1.02 g off in every sample. U-E-1 resolved. | ORACLE (H) |

Legacy (Map1, current code) used 0.95 g / 500 ms after 90 s with a 31-tick float recurrence.

## 3. Experience curve and level caps (CDV `ExperienceCurveData {c6f6a84c}`)
Cumulative XP to *reach* level L (`mExperienceRequiredPerLevel[L-2]`):

| L | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | **19** | **20** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| XP | 280 | 660 | 1140 | 1720 | 2400 | 3180 | 4060 | 5040 | 6120 | 7300 | 8580 | 9960 | 11440 | 13020 | 14700 | 16480 | 18360 | **20340** | **22420** |

(Array continues 24000, 25880, … for other modes; irrelevant.) Per-level deltas 280 + 100·(L−2),
L19 = 1980, L20 = 2080. CDV (H), matches WIKI.

| # | Rule | Tag |
|---|---|---|
| 3.1 | Level cap **18**; **20** for a champion that completed the Top role quest. At cap XP keeps accumulating (tooltip 0/0) but does not level. | RN 26.1, WIKI (H) |
| 3.2 | `level = 1 + #{L in 2..cap : xp ≥ need[L]}`; multi-level jumps allowed in one grant. | CDV table, INF loop (H) |
| 3.3 | "Decimal level" (used by comeback rules) = `L + (xp − need[L]) / (need[L+1] − need[L])`; at cap = L. | WIKI (M) |
| 3.4 | Global clamp on summed XP-gain modifiers: `gcd_PercentEXPBonusMinimum = −1.0`, `gcd_PercentEXPBonusMaximum = 5.0` (i.e. multiplier in [0, 6]). | CDV (H value, M semantics) |
| 3.5 | Regular XP modifiers stack additively with each other; game-mode/global ones multiply. | WIKI (M) |

## 4. Minion XP sharing (values per minion type: minion spec)
| # | Rule | Tag |
|---|---|---|
| 4.1 | Eligible = enemy champions (of the minion) **alive and within 1500 units of the minion's death position** (`BarracksConfig.ExpRadius 1500`, RN 25.S1.1), **plus the last-hitting champion regardless of range/death**. A minion killed by a non-champion still pays XP to eligible champions. | CDV radius (H), WIKI rule (H) |
| 4.2 | Each eligible champion receives `xp_minion × split[n]`, `n` = eligible count, `split = [1.0, 0.65, 0.433, 0.325, 0.26, 0.217]` (`mPlayerMinionSplitXp`). | CDV (H), RN 26.1 |
| 4.3 | **Comeback bonus vs minions.** Minion level `ML` = floor(average integer level of the *owning* team's champions) at the minion's spawn (WIKI Minion) — in a 1v1, ML = owning champion's level at spawn. Active only if `ML > 5` (`aiExp_bonusExpLaneLevelStart = 5`). Let `d = ML − receiver_decimal_level`. If `d > 1` (`…LaneLevelDeltaMin = 1`): bonus `b = 0.2·d` for `d < 2` (`…C1 = 0.2`, `C1UBound = 2`), else `b = 0.4·min(d, 6)` (`…C2 = 0.4`, `LevelDeltaCap = 6`). XP ×(1+b). Max +240%. Negative (over-levelled) variant disabled (`…NegativeLaneLevelStart = 100`). | CDV constants (H), formula shape WIKI pp table (M — exact d=1→2 shape fits table points 20/25/30/35/39.75%, 2→80%, 6→240%) |
| 4.4 | Then multiply by receiver modifiers (Top quest +11%; role −25%/−33% out-of-lane penalties — ROLE_QUESTS §2.3). | RN (M ordering) |
| 4.5 | Minion **gold goes to the last-hitting champion only** (full value, no split, no range). Minion killed by turret/minion/no champion → no gold. `SplitLocalGold`, `ai_GoldRadius2 1000` and Barracks `goldRadius 1250` are not used for lane-minion gold on SR CLASSIC (WIKI "In various modes other than Classic SR, minion deaths may generate gold to nearby allies"). | WIKI (H); purpose of goldRadius unknown (U-E-2) |

## 5. Champion kill: credit, XP

### 5.1 Kill/assist credit
| # | Rule | Tag |
|---|---|---|
| 5.1.1 | A champion is credited with the kill if it affected the victim with damage (any, incl. 0-damage spells), CC (not blind/cripple) or Exhaust within the last **15 s**; credit goes to the **last** such champion before death, even if a minion/turret/monster dealt the final blow. Others with a valid effect in the 15 s window before death get **assists** (also: heals/shields/buffs on the killer or an assister within the window). `AssistDurationOverride = 15.0` in `{4fd6b68d}`. | CDV value (H), WIKI rules (H) |
| 5.1.2 | **Execution**: final blow by a non-champion with no enemy champion credit in 15 s ⇒ no gold bounty dispensed, victim's bounty unchanged, kill streak not interrupted. XP **is** still dispensed to eligible nearby enemies (5.2). | WIKI (H) |
| 5.1.3 | Multi-kill window 10 s (`events_TimeForMultiKill`), 30 s after a quadra (`events_TimeForLastMultiKill`). Cosmetic. | CDV (H) |

### 5.2 Champion-kill XP (CDV `{c6f6a84c}`)
Base solo bounty by **victim level** `V` (`mExperienceGrantedForKillPerLevel`):
`[42, 114, 144, 174, 204, 234, 308, 392, 486, 590, 640, 690, 740, 790, 840, 890, 940, 990, 1040, 1090]` for V = 1..20.

| # | Rule | Tag |
|---|---|---|
| 5.2.1 | Eligible: killer + all assisters (regardless of range/death) + enemy champions alive within **1600** of the death location (`ai_ExpRadius2 = 1600`) + enemy champions that died within the last **10 s** (`aiExp_timeForKillCreditAfterDeath = 10`) and are within range "as if alive". | CDV values (H), WIKI rule (H; "within range" for dead = their corpse position, INF L) |
| 5.2.2 | n = 1: pool = `base[V]`. n ≥ 2: pool = `base[V] × share[V]`, `share = 0.666` for V ≤ 6, `0.82` for V = 7–8, `0.90` for V ≥ 9 (`mExperienceGrantedMultForSharedKillPerLevel`, 9 entries, last repeats). Each eligible gets `pool / n`. | CDV (H); index = victim level per WIKI (M — V14.10 history text says "your level", U-E-3) |
| 5.2.3 | Per-recipient level-difference modifier, `Δ = victim_decimal_level − recipient_decimal_level`: `m = 0.2·(|Δ| − 1)` if `|Δ| > 1` else 0 (`LevelDifferenceExperienceMultiplierPerLevel = [0.0, 0.2]`: 0 at 1 level, +0.2 per level after). Underleveled recipient: ×(1 + m), uncapped except by 3.4. Overleveled recipient: ×(1 − min(m, 0.6)) (floor 40%). | CDV slope (H), WIKI shape/cap (H) |
| 5.2.4 | Top-quest completer: +80 flat per takedown (kill or assist) after 5.2.3; the +11% does not apply to this XP (ROLE_QUESTS §4.1). | RN 26.9 (M) |
| 5.2.5 | XP is paid on execution too (eligible = nearby only, nobody is "credited"). | WIKI (H) |

## 6. Champion kill gold and the bounty system (26.19)

### 6.1 Constants (CDV `{4fd6b68d}` unless noted)
| Field | Value | Meaning (tag) |
|---|---|---|
| `BaseGold[V]` | 300 ×6 (V1–6), then 310, 320, …, 420 (V7–18); V19/20: use 420 (array length 18) | base bounty by **victim** level (CDV H; L19–20 clamp INF M) |
| `FirstBloodBonus` | 100 | +100 g to the killer of the first champion kill of the game (CDV H; RN 26.1 re-added) |
| `AssistDurationOverride` | 15 | assist window (CDV H) |
| `{11f10a40}`, `{937cc95a}`, `{1eacb90a}` | 55, 175, 0.5 | early assist reduction: assist bounty ×50% before 55 s, linear to 100% at 175 s (CDV-value/INF-name; matches WIKI exactly) |
| `{fd43d59f}` | 700 | max payout above base per death (extended bounty) |
| `{907442e7}` = `BountyDisplayThreshold` (name brute-forced by FNV-1a) | 100 | positive buffer / shutdown display threshold |
| `{fe1b406e}` | 50 | minimum kill bounty |
| `{d1f9fb6a}` | 100 | (INF) shutdown start above base; or First-Blood-related |
| `{54ccd262}` | 3.0 | +1 bounty per 3 g earned from kills/assists |
| `{c29d06b9}` | 20.0 | +1 bounty per 20 GV-source gold while bounty ≥ 0 |
| `{fa93507d}` | 7.0 | +1 bounty per 7 GV-source gold while bounty < 0 (RN 26.3 "5 ⇒ 7") |
| `{ec211346}` | 3.5 | −1 bounty per 3.5 g distributed (killer + assisters) on death (RN 26.3 "2.5 ⇒ 3.5") |
| `{a966473c}.{4b733ea3}` | 5.0 | (INF) bounty changes deferred until 5 s out of champion combat |
| `{a966473c}.{00490b09}` | 100 | unknown |
| `{08c1b120}`, `{ce073fe8}`, `{7c29d4b9}` | 0.25, 1.0, 2 | unknown (U-E-4) |
| `ObjectiveBountyConfig` | `{4ec6dd3c}`=1.0, `{3cb3392e}`=2.5 | objective-bounty tuning, unknown semantics |
| `NegativeBountyConfig.{b8ee5cef}` | 100 | unknown (possibly negative-state buffer; WIKI says the 50 g negative buffer was removed in V25.S1.1) |

### 6.2 Bounty state machine (per champion; WIKI *Champion gold bounties* + RN 26.1/26.3)
State: `B` (bounty offset from base, float, may be negative), `buf` (positive-entry buffer
consumed, 0..100), `carry` (extended excess carried to next death), `pending_dB` (deferred
change), `ooc_ms` (time since last champion combat).

1. **Kill payout** for victim at level V: `K = clamp(base[V] + B_pos_paid + B_neg, 50, base[V] + 700)` where `B_pos_paid = min(max(B,0), 700)`; if this is the first champion kill: `K += 100` (first-blood bonus is not part of the bounty). Paid to the credited killer only. (WIKI H; FB CDV H)
2. **Assist payout** (if ≥1 assister): `A = (min(0.5·(K − FB), 0.5·base[V]) + 0.5·FB) × early(t)`, `early(t) = 0.5` for t < 55 s, `0.5 + 0.5·(t−55)/120` for 55–175 s, 1 after; split equally among assisters. Assist gold is *additional* (not taken from K). **Corrected by the replay oracle:** first-blood assists receive half of the +100 too (200 to a lone assister after 175 s, not 150; 145 first bloods); the 50%-of-base cap holds on all later kills including shutdowns (4,589 assisted kills). (ORACLE H, CDV values)
3. **Victim depreciation**: if `B > 0` (positive/shutdown state): remove `min(B, 700)`; excess `B − 700` stays as `carry` and is restored at respawn (wiki: "added onto their next respawn"). If `B ≤ 0`: `B −= (K + A_total)/3.5`; floor: payout never below 50 (excess depreciation below the 50 floor discarded — keep `B ≥ 50 − base[V]`). (WIKI H)
4. **Killer/assister accrual** from champion gold earned `g` (kill + assist + FB gold): `B += g/3`, except that shutdown gold (the part of a kill above `base + 100`) earned while the earner is in a positive state is ignored. When `B` crosses from ≤ 0 into positive, the first **100** of positive accrual fills `buf` and does not raise `B`. (WIKI H; buffer value CDV 100)
5. **GV accrual** (minion/monster gold, support-item generation; *not* ambient gold, plates, turrets, item/rune/champion gold effects): `B += gv/20` if `B ≥ 0`, else `B += gv/7`. (WIKI H, CDV)
6. **Deferral**: on SR, all bounty changes (2–5, except the immediate removal on death) are applied only after the champion has been out of combat with enemy champions for **5 s**. Payout always uses the *applied* B. (WIKI H, CDV 5.0 INF-name)
7. **Team-level suppression** (after 6:00, when neither team is convincingly ahead; shutdown gold reduced 30/60/90/100%) and **objective bounties** (from 14:00; 10% of team gold deficit, cap 1000, minimum per objective outer turret 250, inner/nexus turret 400) depend on hidden team-advantage weights — **not reproducible**. Default for a 1v1 lane sim: **disabled**, flag in profile. (WIKI M)

Executions (5.1.2) change nothing. Death-streak / kill-streak *tiers* no longer exist (removed V14.21);
the legacy `KillSpree`/`DeathSpree`/`GoldFromMinions`/7-over-6 tier logic is obsolete.

## 7. Structure gold distribution (values: tower spec)
| # | Rule | Tag |
|---|---|---|
| 7.1 | **Plate** destroyed (plates at 10/25/45/70/100% missing HP on outer, inner, inhibitor, nexus turrets): local gold (120 g base; −10 g/min from 11:00, floor −40, tower spec) split equally among enemy champions that (a) dealt the destroying hit or damaged the turret (incl. summons) in the last **10 s** — regardless of range/death — or (b) are **alive within 1200** of the turret. | WIKI Turret (H) |
| 7.2 | **Turret destroyed**: same local eligibility for any local gold (outer/inner/inhibitor local = 0 since 26.1; their value lives in plates); **global** gold (outer 50, inner/inhib 25 … per tower spec) to **every** champion of the destroying team regardless of range/death. Destroyed by minions only: global still paid; local to in-range champions (INF M). | WIKI (H) |
| 7.3 | **First turret of the game**: +300 g "share" to nearby/participating champions (split like local gold). | RN 26.1 (H), WIKI |
| 7.4 | Turrets give **no XP** (since V6.9). Legacy `turret_kill_rewards` counted both teams in the split denominator and used 1.5×attack range — both obsolete. | WIKI (H) |
| 7.5 | `events_TimerForBuildingKillCredit = 30.0` in constants conflicts with wiki 10 s (wiki: kill-feed credit 15 s, cosmetic). Default **10 s** for gold. | CDV vs WIKI, U-E-5 |
| 7.6 | Nexus Obelisk (fountain laser) bin carries `globalGoldGivenOnDeath 100`, `globalExpGivenOnDeath 400` (unkillable; irrelevant). | CDV |

## 8. Level-up behaviour
| # | Rule | Tag |
|---|---|---|
| 8.1 | Stat at level n: `base + g·(n−1)·(0.7025 + 0.0175·(n−1))`; per-level gain `g·(0.65 + 0.035·n)`; formula continues unchanged to levels 19–20 (L20 total = 19.665·g). Existing `core.stats.level_growth_sum` implements this; it must accept n up to 20. | WIKI (H) |
| 8.2 | On level-up, **current HP increases by the full max-HP increase** (`ai_levelUp_healthGainNetGain = 1.0`, `ai_levelUp_healthGainPercentMissingPenalty = 0`). Mana: same (INF M). | CDV (H for HP) |
| 8.3 | One skill point per level, levels 1–18 only (max 18 points; levels 19–20 give stats only). Rank limits: basic ability rank ≤ floor(level/2) beyond level 1 (so ranks at 1/3/5/7/9), R at 6/11/16. | WIKI (H) |
| 8.4 | `ai_MaximumHPMaxPenalty = 0.5` — unknown use (likely max-HP reduction floor). Not level-related. | CDV (L meaning) |

## 9. Death timer (CDV `DeathTimes {9de5b46c}`)
`mTimeDeadPerLevel` (BRW, s) for death at level L = 1..18:
`10, 10, 12, 12, 14, 16, 20, 25, 28, 32.5, 35, 37.5, 40, 42.5, 45, 47.5, 50, 52.5`.
Levels 19–20: array has 18 entries → clamp to 52.5 (INF M, U-E-6).

Time scaling fields: `mScalingStartTime 600`, `mScalingIncrementTime 30`, `mScalingPercentIncrease 0.005`,
`mScalingPercentCap 1.5`, `mScalingPoints = [{900, 0.00425}, {1800, 0.003}, {2700, 0.0145}]`.

**Default formula** (WIKI *Death*, matches the scaling points and the cap):
```
TIF accrues CONTINUOUSLY (per 30 s, not in 30 s steps) — measured, see below:
TIF(t) = 0                                                 t < 900
       = 0.00425 * (t-900)/30                              900 <= t < 1800   (max 0.1275)
       = 0.1275 + 0.003  * (t-1800)/30                     1800 <= t < 2700  (max 0.2175)
       = 0.2175 + 0.0145 * (t-2700)/30                     t >= 2700
TIF = min(TIF, 0.5)            # mScalingPercentCap 1.5 = total multiplier cap
death_time = BRW[L] * (1 + TIF)
```
**Measured (replay oracle, 6,025 deaths, 16.9):** the BRW table is exact at every level; no
scaling at 10:00–15:00 (U-E-7 resolved for the wiki); after 15:00 continuous accrual puts 94% of
deaths within 0.1 s, the wiki's ceil steps 47%. **Former disagreement:** the base fields say scaling starts at **600 s at +0.5%/30 s**; the wiki says
nothing before 15:00. If the engine applies the base segment until the first scaling point, deaths
at 10:00–15:00 would be up to +5% longer (e.g. L9 at 14:30: 28 → 29.26 s). Default = wiki (points
override base). U-E-7. Every 26.x death-timer note found (26.16) is **League Classic only** (not SR).
`gcd_PercentRespawnTimeModMinimum = −0.95` clamps summed respawn-time reductions.

Death state: no move/attack/cast/summoner/items (except a few), may shop as if in fountain; cooldowns
keep ticking; channels, windups, dashes interrupted; buffs persist unless flagged otherwise (WIKI H).
Whether death time uses level *at death* (yes, WIKI/legacy) — H.

## 10. Respawn
| # | Rule | Tag |
|---|---|---|
| 10.1 | Respawn at the **centre of the allied fountain** (spawn platform), full HP and mana (mana INF H). | WIKI (H) |
| 10.2 | Game-start spawn uses `RespawnPointData.FirstSpawnPositionOffset` (+160, 0, +120) for team 100 and (−160, 0, −120) for team 200 relative to the team spawn point (per-player slot offsets beyond that unknown). | CDV (H value, M semantics) |
| 10.3 | Before 14:00, respawning grants **Respawn Homeguard ("Deathguard") 75% bonus MS**, no fixed duration since 26.1 (removed by the same rules as Homeguard, §11.2). Also granted after executions. | WIKI (M; not in 26.1 notes) |

## 11. Recall, Homeguard, Teleport-arrival interactions

### 11.1 Recall (`Items/Spells/Recall`, script server-side)
| # | Rule | Tag |
|---|---|---|
| 11.1.1 | Channel **8.0 s** (RN 26.1 mid quest "reducing the channel time from 8 to 4 seconds"; not in client data). Cooldown 0, self-cast, no cost. | RN (H) |
| 11.1.2 | Cast flags: `mCantCancelWhileWindingUp`, `cantCastWhileRooted`, `mNoWinddownIfCancelled`, `mCastingBreaksStealth`. | CDV (H) |
| 11.1.3 | Interrupted by: move/attack/ability/most item or summoner casts by the user; death; silence; ground; root; any **damage > 0 to health** except in the last **0.1 s**. Not interrupted by damage fully absorbed by shields, 0-damage, invulnerability, Death's Dance deferred damage. | WIKI (H) |
| 11.1.4 | On completion: teleport to fountain (same point as respawn); remove Homeguard lockout debuff; start 12 s quest passive lock (ROLE_QUESTS). No HP restored by the recall itself. | WIKI (H), RN 26.9 |
| 11.1.5 | Empowered Recall (Baron/Herald buff, 4 s) out of lane-sim scope; mid-quest empowered recall removed 26.9. | RN |
| 11.1.6 | Current sim windup 500 ms + channel 8000 ms (`step.py:202-206`): the 0.5 s windup is legacy-server behaviour, unverified for modern (U-E-8). |

### 11.2 Homeguard (spell objects `SRHomeguard`, `SRHomeguardSpeed`, `SRHomeguardLockout`, `2026HomeguardSpeed`; values server-side)
| # | Rule | Tag |
|---|---|---|
| 11.2.1 | Available from 0:20. Gained while in the allied fountain; kept (no duration since 26.1) until: reaching the lane **endpoint**, entering combat (dealing/taking champion-relevant damage; damage dealt > 5000 units away exempt), entering the jungle, or completing Teleport. | RN 26.1, WIKI (H) |
| 11.2.2 | Speed **before 14:00: 80% bonus MS decaying to 40% over 4 s** after leaving the fountain; **after 14:00: 150% → 65% over 4 s**. Decay shape linear (INF M). After decay it stays at the floor value until removed. | RN 26.1 (H), shape INF |
| 11.2.3 | Endpoint: always at least to just before the lane's **outermost living allied turret** (wiki estimate ≈500 units behind it at game start) and at least to the inhibitor; after 14:00 or once any turret in that lane is down, tracks ≈2000 units before the furthest allied minion (bounded by the outermost structure rules). | RN 26.1, WIKI (M) |
| 11.2.4 | Lockout: losing Homeguard to combat/jungle applies `SRHomeguardLockout` for **8 s** (WIKI); removed on Recall completion. RN 26.8 "3 ⇒ 6 s" is an ARAM Mayhem augment, not SR. | WIKI (M) |
| 11.2.5 | In fountain with Homeguard: extra heal **8% missing HP and 8% missing mana every 0.5 s**. | WIKI (H; V25.07) |
| 11.2.6 | Homestart (game start, ≤55 s in fountain): 175% bonus MS for 15 s, decays over 0.75 s past outer-turret line; removed by damage/TP. Lane sims starting at the gate can ignore it. | WIKI (M) |

## 12. Fountain (spawn platform), obelisk, shop
| # | Rule | Tag |
|---|---|---|
| 12.1 | Regen tick every **0.25 s** (`sp_RegenTickInterval`) for allied units within **1100** (`sp_RegenRadius`) of the fountain: **+2% max HP** (`sp_HealthRegenPercent`) and **+2.5% max mana** (`sp_ManaRegenPercent`) per tick ⇒ 8% HP/s, 10% mana/s. `sp_HPMaxPenaltyRegenPercent = 0.08` unknown use. | CDV (H), WIKI agrees |
| 12.2 | Nexus Obelisk `SRUAP_Turret_Order5/Chaos5`: range 1250 (acquisition 1250), AS 2.5 (effective 2.0/s per wiki), AD 999 (wiki: 1000 true "internal raw" per hit, ignores shields/undying, does not cause combat), HP 9999, untargetable. Targets nearest enemy unit, no champion priority. | CDV stats (H), WIKI behaviour (M) |
| 12.3 | Shop: purchase/sell only within shop range of the fountain **or while dead**. Sell = **70%** of total cost; 40% for starter items (Doran's, Dark Seal), potions, Control Ward, elixirs, Cull, Guardian Angel; jungle/support quest items unsellable. Undo refunds 100% while not having left shop range / entered combat / item transformed / active used. `mItemSellQueueTime = 0.25 s`. | WIKI (H), CDV queue time |
| 12.4 | Afk warning in fountain — ignore. |

## 13. Events and ordering within a tick (recommended, INF M)
1. Movement/collision final positions. Fountain regen pulse (0.25 s timer) and Homeguard fountain heal (0.5 s timer).
2. Ambient gold tick (0.5 s timer, t ≥ 65 s).
3. Damage resolution; recall interruption check; combat timers (bounty 5 s, Homeguard removal/lockout).
4. **Deaths**: for each dead unit, determine killer/assisters (15 s windows, champion-credit rule).
5. **Bounty/gold**: champion kill payout K + first blood, assist split A (early factor), victim depreciation; minion gold to last hitter; plate/turret local+global+first-turret shares. Queue bounty accruals in `pending_dB`.
6. **XP**: minion XP splits (+comeback, +quest/role modifiers), champion-kill XP pool & per-recipient modifiers (+80 quest flat), turret XP none.
7. Quest event points, passive points, quest completion (+600 XP, cap 20) — ROLE_QUESTS §6.
8. **Level-up** (possibly multiple levels, cap 18/20): max HP/mana + current HP/mana increase; skill points/ranks.
9. **Stat refresh** (growth, items, resist) for this tick's outputs.
10. Death timers use the victim's level *at death* (victims gain no XP from their own death). Respawn countdown, respawn at fountain (+Deathguard before 14:00).
11. Apply deferred bounty changes whose owner has been out of champion combat ≥ 5 s.
Simultaneous deaths: process all deaths of a tick against pre-tick bounty values, in slot order (INF L).

## 14. State the simulator must carry (per champion unless noted)
`gold, xp, level, level_cap, skill_points`; `ambient_timer_ms`; `bounty B, bounty_buf, bounty_carry, pending_dB, champ_combat_ooc_ms`;
damage-credit history `last_hit_by[source] → time` (15 s window per enemy champion, plus helper credit);
`turret_damage_by[turret, champion] → time` (10 s); `first_blood_done` and `first_turret_done` (global);
`respawn_ms`, `death_time_for_xp_ms` (10 s dead-share window); `recall_channel_ms`, recall interrupt flags;
`homeguard_active, homeguard_t_ms (decay), homeguard_lockout_ms, deathguard_active`;
`fountain_regen_timer`, `homeguard_heal_timer` (global or per team); minion `spawn_level` (for 4.3);
quest state (ROLE_QUESTS §6). Tables to 21 rows (levels 0–20).

## 15. Diff vs current implementation
| Location | Current | Modern (this spec) |
|---|---|---|
| `lanerl_jax/modern/world/config.py:107` | `gold.at[:2].set(500.)` | ✓ matches 500 |
| `rewards.py:99-101`, used by modern path at `step.py:1435` | ambient 0.95 g/500 ms after **90 s** (1.84 g/s effective) | 1.02 g/0.5 s after **65 s** (2.04 g/s) |
| `rewards.py:128` `death_rewards` (XP radius `EXP_RADIUS=1600`, `rewards.py:119`; equal split `xp/count`) | 1600 radius, equal division, killer not guaranteed | 1500 radius + killer guaranteed; split table [1, .65, .433, …] per champion; comeback bonus vs minion level |
| `rewards.py:255` `champion_kill_rewards`, constants `rewards.py:69-83` | 4.20 tier system (300·(7/6)^k cap 500, feed 275·0.8^…, min 50, `DeathSpree`/`GoldFromMinions` 1000 g) and double-increment bug | level-scaled base 300→420, continuous bounty B (§6), assist gold (missing entirely today), early-assist factor |
| same, XP | `ExpCurve[V−1]·0.55` (= 154 XP for a level-1 victim) ± min(0.08·Δ, 0.15) | table 42…1590, shared multiplier, ±20%/level beyond 1, penalty cap 60% |
| `rewards.py:393` `turret_kill_rewards` | radius 1.5×attack range, both teams in denominator, global XP | 1200 radius + 10 s damage participants, enemy-team only, no XP, plates, first-turret 300 |
| `profiles.py:352-375` level tables (19 rows from Map1 `ExpCurve.json`/`DeathTimes.json`) | 4.20 Map1 tables, cap 18, `death_times[L] = TimeDeadPerLevel(L+1)` | client 26.19 tables, 21 rows, BRW indexed by death level directly; time multiplier from 15:00 |
| `step.py:1531` respawn | `params["death_times"][lvl]` no game-time factor | ×(1+TIF(t)) |
| `step.py:202-209, 350-373` fountain | +15% max HP every 1.0 s within 1000 of spawn | +2% HP & +2.5% mana every 0.25 s within 1100; +8% missing HP/mana per 0.5 s with Homeguard |
| `step.py:1440-1446` level-up HP | current HP += max-HP growth | ✓ same rule (keep; extend to mana and L19–20) |
| `step.py:1457`, `modern.py:99` | ranks clipped at level 18 | keep 18-point limit; levels 19–20 add no ranks |
| Homeguard/Deathguard, shop rules, assist credit, first turret, kill credit window via `update_hit_flag` (`rewards.py:187`, 15 s, single last attacker only) | absent / single attacker | full per-champion credit sets |
| `docs/MODERN_PATCH_DELTA.md` §9.3/§9.6 | 26.18 research | confirmed except: assist cap is 50% of base (✓), devaluation 1:3.5 (26.3), negative GV 1:7 (26.3), first-turret +300 back (26.1), XP share radius minion 1500 / champion 1600 (✓) |

## 16. Disagreements and chosen defaults
| Topic | A | B | Default | Why |
|---|---|---|---|---|
| Death-timer scaling start | CDV base fields: 600 s, +0.5%/30 s | WIKI: 0 before 15:00, then scaling points | WIKI (points override) | points list exactly reproduces wiki values & 50% cap; base fields look like legacy defaults (U-E-7) |
| Building kill credit window | CDV `events_TimerForBuildingKillCredit 30` | WIKI 10 s for gold, 15 s kill feed | 10 s | wiki explicitly gold-tested; constant may be feed/legacy |
| Shared kill-XP multiplier index | WIKI main text: slain champion level | WIKI V14.10 history: "your level" | victim level | matches sibling array indexed by victim level |
| RN 26.1 devaluation wording "0.2 ⇒ 0.4 per 1 gold … 300 g reduces by 75 g" | RN | WIKI 1:2.5 (26.1) → 1:3.5 (26.3), CDV 3.5 | 1:3.5 | client + latest note |
| Respawn Homeguard (75%) | WIKI | not in RN 26.1 | keep (WIKI) | no removal noted |
| Minion gold radius `goldRadius 1250` / `ai_GoldRadius2 1000` | CDV present | WIKI: last-hit only on Classic SR | last-hit only | wiki explicit; radii likely for other modes/features |
| MODERN_PATCH_DELTA §9.3 "multiplier after 15:00" | agrees with default | — | — | — |

## 17. Unresolved / needs live measurement
| ID | Question | Scenario |
|---|---|---|
| U-E-1 | **Resolved (oracle): 65.0 s.** First ambient payment time (65.0 vs 65.5 s) and exact 0.5 s phase | Custom game, read gold at 64.9/65.1/65.6 s via Live Client Data API (`activePlayer.currentGold`) |
| U-E-2 | Meaning of Barracks `goldRadius 1250`, `ai_GoldRadius2 1000`, `SplitLocalGold` | Let a turret kill a minion with a champion at 900 u; check gold |
| U-E-3 | Shared kill-XP multiplier indexed by victim or recipient level | 2v1 assisted kill on L8 victim by L4 killers; compare 392·0.82/2 vs 0.666 |
| U-E-4 | Hashed `{4fd6b68d}` fields 0.25/1.0/2/`{d1f9fb6a}`100/`NegativeBountyConfig`100 | Bounty experiments: repeated deaths, kill while negative, shutdown sizes |
| U-E-5 | Turret/plate participation window (10 vs 30 s) | Hit turret, walk >1200 away, wait 12 s, plate breaks by minions |
| U-E-6 | BRW and base kill gold at levels 19–20 | **Death timer resolved** (oracle: L19/L20 deaths = L18 value). Kill gold at L19–20 still open. |
| U-E-7 | **Resolved (oracle): none before 15:00; continuous after.** Death-timer scaling 10:00–15:00 | Die at L9 at 14:30: 28.0 s (default) vs 29.26 s |
| U-E-8 | Recall: any pre-channel windup in modern client? damage-interrupt grace 0.1 s | Frame-record recall start; hit at 7.95 s |
| U-E-9 | Homeguard decay curve shape; endpoint distance; lockout 8 s | Record MS each frame leaving fountain at 5:00 and 15:00 |
| U-E-10 | Bounty 5 s combat deferral: which events count as "combat with enemy champions" | Kill, keep trading, read scoreboard bounty |
| U-E-11 | Minion level for comeback XP = owning team's average at spawn (1v1: owner level) | Underleveled champion (L5 vs L8 opponent) farming L8-spawned waves: melee XP 62·(1+0.4·3)=136.4 |
| U-E-12 | Simultaneous same-tick deaths ordering for bounty/first blood | Double kill trade |

## 18. Test fixtures
1. **Ambient**: at t = 120.0 s, no other income: `500 + 111·1.02 = 613.22` g (payments at 65.0, 65.5, … — measured). At 600 s: `500 + 1071·1.02 = 1592.42`.
2. **Levels**: xp 279 → L1; 280 → L2; 18359 → L17; 18360 → L18; 25000 (cap 18) → L18; same with cap 20 → L20; 20339 cap 20 → L18; decimal level at xp 2000 = 5 + (2000−1720)/680 = 5.4118.
3. **Minion XP split**: melee minion 62 XP, 1 champion → 62; 2 → 40.3 each. Comeback: receiver decimal level 4.0, enemy minion level 7 (> 5), d = 3 → ×2.2 → 136.4 (solo). d = 1.5 (ML 6, receiver 4.5) → ×1.3 → 80.6. ML 5 → no bonus.
4. **Champion-kill XP**: V6 solo, recipient 6.0 → 234. Recipient 4.0 → 280.8. Recipient 8.5 → 163.8. Recipient 12.0 → 93.6 (−60% cap). V8, two eligible (killer+assister, both 8.0) → 392·0.82/2 = 160.72 each. V1 solo → 42.
5. **Kill gold**: V8 victim, B = 0, first champion kill of the game, one assister, t = 150 s: killer 320 + 100 = **420**; assister `(min(160,160) + 50)·(0.5+0.5·95/120)` = **188.13**; victim B → −(320 + 188.13)/3.5 = **−145.18** → next payout at V8 = **174.82**. Killer accrual 420/3 = 140 → buffer 100, B = **+40** (applied after 5 s out of combat).
6. **Shutdown**: V10 victim (base 340) with B = 500 → K = 840, B → 0. B = 900 → K = 340 + 700 = 1040, carry 200 restored at respawn (B = 200).
7. **Floor**: base 300, B = −280 → K = 50.
8. **Assist early factor**: t = 40 s → 0.5; t = 115 s → 0.75; t ≥ 175 s → 1.0.
9. **Death timer**: L5 at 9:00 → 14.0 s. L6 at 20:00 → 16 × 1.0425 = 16.68 s. L9 at 31:00 → 28 × 1.1335 = 31.738 s. L18 at 56:00 → 52.5 × 1.5 = 78.75 s (cap). L9 at 15:00.0 → 28.0; at 15:15 → 28 × 1.002125 = 28.0595 (continuous, corrected 2026-10-01).
10. **Fountain**: max HP 2000, HP 500, in fountain without Homeguard: +40 HP per 0.25 s ⇒ full after 38 ticks = 9.5 s. With Homeguard additionally +8% of missing HP every 0.5 s: at t = 0.5 s, 500 + 2·40 = 580, then +0.08·(2000−580) = 693.6 (applying the flat pulses before the missing-HP pulse on coincident ticks — INF).
11. **Plate share**: plate worth G (tower spec; 120 at 8:00). One eligible champion → G; destroyer + an allied champion alive within 1200 → G/2 each; an ally who hit the turret 9 s ago and is now 3000 away (or dead) → still eligible; 11 s ago and out of range → not eligible.
12. **Level-up HP**: champion HP 400/600, growth 100 at level 2 gain 0.72·100 = 72 → 472/672.

## 19. Patch audit 26.1 → 26.19 (economy/progression lines)
| Patch | Change (SR) | In 26.19 |
|---|---|---|
| 26.1 | Ambient gold start 65 s; minion spawn 30 s; minion XP split 100/65/43.3/32.5/26/21.7; minion XP/gold values (minion spec); First Blood +100 back; first turret +300 back; base kill-gold devaluation 1:5 → 1:2.5, GV negative re-accrual 1:10 → 1:5; Homeguard rework (no duration, endpoint, 80→40% / 150→65%); assist early window 55–175 s (wiki); role quests | ✓ (devaluation/negative GV re-tuned 26.3) |
| 26.3 | Devalue 1:2.5 → 1:3.5; farming re-increment 1:5 → 1:7 | ✓ |
| 26.9 | Top quest XP reward rework (ROLE_QUESTS); minion wave HP (minion spec) | ✓ |
| 26.12 | TP shield 35%/10 s | ✓ |
| 26.16 | "Start of Ambient Gold 65 ⇒ 90 s", "Passive Gold 8/10.5/…", "Homeguard activation 0 ⇒ 15 min", "Death Timers 12/14/… ⇒ 10/12/…" — **League Classic mode only**, not SR | excluded |
| 26.19 | Teleport cooldowns (ROLE_QUESTS); Zilean no XP while dead (champion) | ✓ |
| 26.2, 26.4–26.8, 26.10, 26.11, 26.13–26.15, 26.17, 26.18 and mid-patch/hotfix blocks (26.1 Jan 9, 26.3 Feb 5, 26.6 Mar 20, 26.9 Apr 30) | no SR economy/XP/death/recall/fountain/Homeguard changes found (26.8 "Homeguard 3 ⇒ 6 s" is ARAM Mayhem) | — |
