# ROLE_QUESTS.md — 2026 Role Quest system (Top focus), patch 26.19

**Scope.** Normal PC Summoner's Rift (mode `CLASSIC`, Map11), patch **26.19**, client build
**16.19.8230722**. Primary subject: the **Top lane quest** (the simulator's lane). Other roles are
summarised only where they interact with a top-lane 1v1 or define shared rules.
**Retrieval date:** 2026-10-01. Research only; nothing here is implemented.

Companion spec: `docs/modern/ECONOMY_PROGRESSION.md` (XP/gold/level/death/recall/fountain/Homeguard),
which the quest rewards plug into.

## 0. Sources

| ID | Source | Pin |
|---|---|---|
| C1 | CommunityDragon `16.19/game/data/characters/../shared.cdtb.bin.json` (`Shared/Spells/SR_2026_S1_RoleBound_*`, `SummonerTeleport`, `S12_SummonerTeleportUpgrade`) | cache `/mnt/nfs/shared/modern-world-map-research/cdragon-16.19/shared.cdtb.bin.json`, sha256 `34f68553ab38cfe344936473fcb48d99574d5fd51e4feb42994fa0c4fe50769e` |
| C2 | CDragon `16.19/game/en_us/data/menu/en_us/lol.stringtable.json` (quest tooltips) | cache `cdragon-16.19/en_us-lol.stringtable.json`, sha256 `8c051cb2a24b31f3fa9af95d39085b832b0da2b8cf51093f98c4ab0620ecb8e8` |
| C3 | CDragon `16.19/game/data/maps/shipping/map11/map11.bin.json` (CLASSIC mode record, quest-completion VFX, `{4fd6b68d}` kill-gold config) | cache `cdragon-16.19/map11.bin.json`, sha256 `0fe009599998c3f4f602838d3e9a37c583455d76bdd0019571d65ad08250cd97` |
| C4 | CDragon `16.19/game/maps/modespecificdata/classic.bin.json` (NarrativeBarks for RoleBoundQuest completion) | cache `cdragon-16.19/modespecificdata-classic.bin.json`, sha256 `8ace28c1d6b264bc7246919f9f1da3b2e385da9c49560087ab95962738ecf1fa` |
| C5 | CDragon `16.19/game/gameplay.roleboundquestselection.bin.json` (UI only) | cache sha256 `1df650cf…d2ae7c` |
| R1 | Riot patch 26.1 notes, "Role Quests" — https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-1-notes/ | text cache `/mnt/nfs/shared/modern-world-map-research/econ-notes-26.x/26-1.txt` |
| R9 | Riot 26.9 notes, "Role Quests" — https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-9-notes/ | `econ-notes-26.x/26-9.txt` |
| R12 | Riot 26.12 notes, "Systems → Teleport" | `26-12.txt` |
| R16 | Riot 26.16 notes, "Support Role Quest" | `26-16.txt` |
| R19 | Riot 26.19 notes, "Teleport Cooldown" | `26-19.txt` |
| W1 | Wiki *Role Quests* oldid **4064833** (2026-09-22) https://wiki.leagueoflegends.com/en-us/Role_Quests?oldid=4064833 | `econ-wiki/Role_Quests.wiki` |
| W2 | Wiki *Teleport* oldid **4065272** (2026-09-23) | `econ-wiki/Teleport.wiki` |
| W3 | Wiki *Experience (champion)* oldid **4053165** | `econ-wiki/Experience_(champion).wiki` |
| W4 | Wiki *Shop* oldid **3982704** (undo rule) | `econ-wiki/Shop.wiki` |

All 19 patch-note pages 26.1–26.19 (and their embedded mid-patch/hotfix sections) were fetched and
grepped for quest/role/Teleport/XP/level keywords; the full audit trail is in §7. Wiki `.wiki`
files have a `SHA256SUMS` in `econ-wiki/`; notes text in `econ-notes-26.x/SHA256SUMS`.

**Critical client-data finding.** The client ships only *names, icons and tooltips* for the quest
buffs (`SR_2026_S1_RoleBound_Top_Quest`, `..._Top_QuestCompleted`, `..._EnhancedTeleportBonus`,
`SR_2026_S2_RoleBound_RoamingBank`, etc.). Their `mScriptName` scripts are server-side; the
Scripts.wad `.luabin64` files extracted earlier for other buffs are 100–150-byte stubs. **Point
values, thresholds and the in-lane test are therefore NOT client-verifiable**; they come from Riot
notes + wiki. The only quest numbers present in client data are the Teleport cooldown formula and
its quest-completion modifier (§4.3).

Tag legend: **CDV** = client-data verified; **RN** = Riot notes; **WIKI**; **INF** = inferred.
Confidence H/M/L.

---

## 1. Role binding

| # | Rule | Tag |
|---|---|---|
| 1.1 | Each player receives exactly one role quest, chosen from their **assigned position in champion select** (Top/Jungle/Mid/Bot/Support). Swaps only in champ select. It is not inferred from in-game lane presence. | RN 26.1 (H), WIKI W1 |
| 1.2 | Enabled in all SR queues with assigned positions, incl. custom SR lobbies (position must be chosen). Swiftplay/rotating modes: disabled. | RN 26.1 (H) |
| 1.3 | Sim: role is a **scenario parameter per champion** (`role ∈ {TOP, JUNGLE, MID, BOT, SUPPORT, NONE}`), fixed at reset. A 1v1 top scenario sets both champions `TOP`. | INF (H) |
| 1.4 | The "quest lane" for TOP is the top lane. "In the quest lane" (since 26.9) = **anywhere in the lane, outside your base**. Before 26.9 it was "near or past the lane's outermost turret". | RN 26.9 (H for wording; geometry L — see U-RQ-1) |

## 2. Top quest progress (26.19 state)

Completion threshold **1200 points**. Points are a float accumulator `quest_points`; completion
when `quest_points >= 1200` (checked after every grant, same tick). RN 26.1 (1200) + WIKI W1 (H).

### 2.1 Event points

Let `p = min(quest_points / 1200, 1)` (progress fraction, evaluated *before* the grant) and
`out_mult(p) = 0.25 + 0.75·p` (out-of-lane multiplier). WIKI W1 "25% - 100% of that in other
lanes … penalty decreasing linearly by quest progress"; RN 26.9 "-50% ⇒ -75%, decreasing down to
-0% based on Quest Progress". Linearity: WIKI (M). Whether "in lane" for objectives is judged by
the *object's* lane (turret/plate "from top lane turrets") or the *player's* position: for turrets
and plates the wiki says "from top lane turrets" → use the **structure's lane**; for minions use
the **minion's lane assignment** (a top-lane wave minion) — INF (M), U-RQ-2.

| Event (credit rule) | In top lane | Elsewhere | Tag |
|---|---|---|---|
| Minion kill (last hit by this champion, any lane minion; INF that executes by own turret/minions give nothing) | **2** | `2·out_mult(p)` | RN 26.1 (+1/+2 doubled in lane) → 26.9 penalty; WIKI (H values, M credit) |
| Turret takedown (kill or assist credit on a turret, see econ §7.3 eligibility) | **50** | `50·out_mult(p)` | RN/WIKI (H) |
| Turret plate (same eligibility as plate gold share) | **40** | `40·out_mult(p)` | RN/WIKI (H value, M credit) |
| Champion takedown (kill or assist) | **15** | 15 (no lane factor) | RN/WIKI (H) |
| Epic monster takedown | **30** | 30 | RN/WIKI (H) — out of scope for lane sim |

Note 26.1 listed *out-of-lane* base values (minion 1, turret 25, plate 20) = 50% of in-lane; 26.9
changed the out-of-lane factor to 25% at p=0, rising linearly to 100% at p=1.

### 2.2 Passive points

| # | Rule | Tag |
|---|---|---|
| 2.2.1 | Starts at **65 s** game time (1:05). | RN 26.1, WIKI (H) |
| 2.2.2 | Base rate **1 point / 3 s** (0.333/s) anywhere. | RN 26.1/26.9, WIKI (H) |
| 2.2.3 | While "in the quest lane" (or while spending roam bank, 2.2.5): **1.5 points/s** (7.5 per 5 s) — *replaces* (not adds to) the base rate. 26.1–26.8 value: 1.6/s (8 per 5 s). | RN 26.9 "0.333/s, increased to 1.5/s" (H for value; "replaces" M) |
| 2.2.4 | **Recall lockout**: passive points are disabled entirely for **12 s after recalling** (start the 12 s at Recall channel *completion* — INF M). Event points still accrue. | RN 26.9, WIKI (H rule, M trigger) |
| 2.2.5 | **Roam bank** (buff `SR_2026_S2_RoleBound_RoamingBank`, tooltip "Reduced Role Quest Passive Progress"): while in lane, bank roam time; while out of lane with bank > 0, earn the in-lane rate and drain the bank 1 s/s. Bank cap **5 s before champion level 3**, **60 s from level 3**; fill rate **0.5 s per s in lane** ("up to 60 s after spending 120 s in lane"). Whether the 5 s pre-level-3 grace fills at 0.5 s/s is unknown (use 0.5 s/s, INF L). | RN 26.9 (60/120), C2 tooltip text "Staying in lane gives up to 5 seconds of grace period. After level 3, this is increased to up to 60 seconds." (CDV text, H) |
| 2.2.6 | Passive points while dead: unknown; default **no passive gain while dead** (dead champion is in base/not in lane, and base rate likely still ticks) → default: base 1/3 s continues while dead, in-lane rate never applies. INF L (U-RQ-4). |
| 2.2.7 | Tick cadence: notes quote "+1 per 3 s" / "+7.5 per 5 s"; sim default = **continuous accrual each tick** (`rate·dt`). Discrete 3 s / 5 s pulses would shift completion by ≤ 5 s. | INF (M), U-RQ-3 |

Passive-only completion time with perfect lane presence: `65 + 1200/1.5 = 865 s = 14:25`
(the wiki's 13:35 figure uses an inconsistent 96 pts/min). Typical play (CS ≈ 7/min → +14/min,
plates, a takedown) completes ~11–13 min.

### 2.3 Minion economy penalty while questing (top)

Gold **and** XP from minions are reduced by **25% outside of top lane until champion level 3**
(applies to the bounty this champion earns; a top-lane minion is "in lane"). RN 26.1 lists this for
top, mid and bot. Wiki W1 (26.19) lists it only for mid and (33% until level 5, since 26.16) for
support; no note removes it for top. **Default: apply to top** (RN, M). U-RQ-5. In a pure top-lane
1v1 this never triggers.

## 3. Completion event (ordering)

On the tick `quest_points` first reaches ≥ 1200 (after the granting event is processed):
1. Set `quest_complete = True`; remove `..._Top_Quest` buff, add `..._Top_QuestCompleted` (CDV names).
2. Raise this champion's **level cap 18 → 20** (RN/WIKI H).
3. Grant **+600 XP** (flat, RN 26.1/WIKI H). Whether the +11% multiplier applies to this grant:
   default **no** (it is the reward itself) — INF M.
4. Recompute level from total XP with the new cap (XP banked beyond 18 is honoured; INF H —
   wiki: XP still accrues at cap).
5. Teleport reward (§4). 6. Broadcast completion (VFX `SR_2026_S1_RoleBound_QuestCompletion_Top`,
   narrative bark, team chat) — no gameplay effect; expose in observation as public info (visible
   to all if in sight, announced to allies).

## 4. Rewards (26.19)

### 4.1 Experience
| # | Rule | Tag |
|---|---|---|
| 4.1.1 | +600 XP immediately. | RN, WIKI (H) |
| 4.1.2 | **+11% XP from all non-champion-takedown sources** (minions, monsters, turrets, quest flat excluded); since 26.9 (was +12.5% from all). Modifier additive with other regular XP modifiers. | RN 26.9 "+80 flat XP on champion takedown; +11% from all other sources" (H) |
| 4.1.3 | **+80 flat XP on each champion takedown** (kill or assist), in place of the % bonus on that XP. Whether +11% also multiplies the takedown share: RN wording "from all *other* sources" → **no** (H-M). |
| 4.1.4 | Level cap 20; thresholds L19 = 20340, L20 = 22420 cumulative XP (CDV `mExperienceRequiredPerLevel`). See ECONOMY §3. | CDV (H) |

### 4.2 Teleport
| Case | Reward | Tag |
|---|---|---|
| Did **not** take Teleport | Gain **Unleashed Teleport** as a bonus summoner spell in the Role Quest slot, cooldown **390 s** (26.1–26.18: 420 s = "7 minute"). It is Unleashed immediately (not gated by 10:00). | RN 26.1 + RN 26.19 "Free Teleport Cooldown: 420 ⇒ 390"; WIKI W1/W2 (H). The 390 value is **not in client data** (server script). |
| Took Teleport | (a) On TP arrival (after channel + dash) gain a **shield = 35% max HP for 10 s** (26.12; was 30%/30 s). No shield if the channel is interrupted (26.03 undocumented). Buff `SR_2026_S1_RoleBound_EnhancedTeleportBonus` (tooltip "shielded from using teleport recently"). (b) **Unleashed** Teleport cooldown reduced by **30 s** (26.19). | RN 26.12, RN 26.19, WIKI W2 (H); CDV for (b) |

Free-TP initial cooldown on grant: unknown — default **ready immediately** (INF L, U-RQ-6).
Quest-TP behaviour otherwise = Unleashed Teleport (CDV `S12_SummonerTeleportUpgrade`: channel 3 s,
travel speed 4500, max travel 7 s, +50% MS for 4 s on arrival, range 25000, targets allied
minion/structure/ward). Shop "undo" is blocked once the quest TP channel starts (WIKI W4).

### 4.3 Teleport cooldown formula (CDV)
`S12_SummonerTeleportUpgrade.mSpellCalculations.UpgradedCooldown` =
`ByCharLevelBreakpoints(level1=330, perLevel=-10, breakpoint{level 10: additional -10})`
`+ BuffCounter(SR_2026_S1_RoleBound_Top_QuestCompleted [hash 3df2e83f]) × (-30)`.

Interpretation (INF M for intermediate levels; endpoints match RN 26.19 "330 – 240 ⇒ 300 – 210"):
```
unleashed_cd(L) = 330 - 10*(min(L,9) - 1) - (10 if L >= 10 else 0)   # 330,320,...,250 (L9), 240 (L>=10)
unleashed_cd_quest(L) = unleashed_cd(L) - 30                         # top quest complete + own TP
```
Non-top Unleashed TP unchanged (RN 26.19). Base (pre-10:00) Teleport: cooldown 300 s, channel 3 s,
travel 1000 u/s, max travel 8 s, level 7 platform requirement irrelevant in-game (CDV
`SummonerTeleport`; `UpgradeMinute = 10`). The base `SummonerTeleport.UpgradedCooldown` field is a
flat 240 (unused/legacy? U-RQ-7).

## 5. Other roles (brief; only for shared rules)
| Role | Threshold | Notable | Reward (26.19) |
|---|---|---|---|
| Mid | 1350 | takedown 25; +3% (melee) / 1.5% (ranged) of champion damage as points; same minion/plate/turret/passive rules | +8% bonus AD and +8% AP (26.11; 6% in 26.9; Empowered Recall 8→4 s removed 26.9); free tier-3 boots |
| Bot | 1350 | minion 3 in lane; takedown 15 | +300 g, +2 g/minion kill, +40 g per takedown (50→40 in 26.9), boots slot |
| Support | 800 | World Atlas charges (18/21 pts since 26.16) | Bounty of Worlds upgrade, 40 g Control Wards; −33% minion gold/XP outside bot until level 5 (26.16) |
| Jungle | 35 pet treats | — | Primal Smite, +10 g/+10 XP per large monster, jungle MS |
Interactions with a top 1v1: none mechanical, except that a lane opponent from another role (e.g.
a mid laner placed top in an off-role scenario) would earn out-of-lane quest points.

## 6. State and hooks

**State per champion:** `role:int8`, `quest_points:f32`, `quest_complete:bool`,
`roam_bank_s:f32`, `quest_recall_lock_ms:f32`, `in_quest_lane:bool` (derived each tick),
`level_cap:int8` (18/20), `has_quest_tp:bool`, `quest_tp_cd_ms:f32`, `tp_shield_hp:f32`,
`tp_shield_ms:f32`.

**Tick order (insert into ECONOMY §10 ordering):**
1. Positions final → compute `in_quest_lane`.
2. Roam-bank fill/drain; recall lock countdown.
3. Death resolution → kill/assist credit → turret/plate/minion credit (ECONOMY).
4. Quest event points from this tick's credits (using `p` before each grant, events processed in
   the same order as gold distribution: minion kills, plates, turrets, champion takedowns).
5. Passive points (`rate·dt` if `t ≥ 65 s` and not recall-locked).
6. Completion check → §3 (cap raise, +600 XP) **before** this tick's level-up/stat refresh, so the
   level-up pass handles levels 19–20 once.
7. XP modifiers for subsequent grants read `quest_complete` from the *start* of the grant (a
   grant that completes the quest is not itself multiplied — INF).

## 7. Patch audit 26.1 → 26.19 (quest-relevant lines only)
| Patch | Change | Status in 26.19 |
|---|---|---|
| 26.1 | System introduced. Top: 1200 pts; +1/+2 minion; +30 epic; +25/+50 turret; +20/+40 plate; +15 takedown; passive 1/3 s, 8/5 s in lane from 1:05; −25% minion gold/XP outside top until L3. Rewards: cap 20, +600 XP, +12.5% XP all sources, Unleashed TP 7 min CD if no TP, else 30%-max-HP 30 s shield | partially superseded |
| 26.2 | Support quest UX (hotkeys) | n/a |
| 26.3 | Control wards start in quest slot (support); (wiki) TP shield not granted on interrupted channel | in force |
| 26.9 | Out-of-lane objective penalty −50% → −75% decaying to 0% with progress; passive in-lane 1.6 → 1.5/s; "in lane" = anywhere in lane outside base; 12 s passive lock after recall; roam bank (60 s per 120 s, after L3); Top XP +12.5% all → +80 flat per takedown & +11% others; Mid reward → 6% AD/AP; Bot takedown gold 50 → 40 | in force |
| 26.11 | Mid 6% → 8% | in force |
| 26.12 | TP shield 30%/30 s → 35%/10 s | in force |
| 26.16 | Support quest stacks/penalty −33% until L5 | in force |
| 26.19 | Free quest TP 420 → 390 s; Unleashed TP for top quest 330–240 → 300–210; non-top unchanged | in force |
| 26.4–8, 10, 13–15, 17–18 | No SR role-quest system changes found (26.8 "Homeguard" line is an ARAM Mayhem augment; 26.14 only trimmed redundant quest buffs from UI) | — |

## 8. Disagreements and defaults
| Topic | Sources | Default | Why |
|---|---|---|---|
| Passive in-lane rate | RN 26.1 8/5 s; RN 26.9 1.5/s; wiki "7.5 every 5" | **1.5/s** | latest RN, wiki agrees |
| Wiki time-value table (96 pts/min, 13:35 completion) | inconsistent with its own 7.5/5 s | ignore | arithmetic error |
| −25% minion gold/XP until L3 for top | RN 26.1 yes; wiki omits | **apply** | no removing note |
| Out-of-lane multiplier shape | RN "decreasing down to −0% based on progress"; wiki "linearly" | linear `0.25+0.75p` | only explicit shape |
| +11% applying to +600/+80 | unspecified | no | reward semantics |

## 9. Unresolved / needs live measurement
| ID | Question | Test scenario |
|---|---|---|
| U-RQ-1 | Exact "in quest lane" polygon (lane width; where "base" ends; river crossings) | Practice-tool-like custom game with positions: walk perpendicular to top lane at several points, read quest bar rate (1.5 vs 0.333/s) |
| U-RQ-2 | Lane attribution for minion/plate points (object lane vs player position) | Kill a top minion standing in jungle; take a plate while standing at range |
| U-RQ-3 | Passive cadence (continuous vs 3 s/5 s pulses) | Record quest bar at 10 Hz in lane from 1:05 |
| U-RQ-4 | Passive while dead / in fountain | Die at 3:00, record bar during death timer |
| U-RQ-5 | Top −25% minion gold/XP outside lane before L3 still live? | Last-hit a mid caster at level 1 as top: expect 14×0.75 = 10.5 g |
| U-RQ-6 | Free quest TP initial cooldown on grant | Complete quest without TP, inspect slot |
| U-RQ-7 | Meaning of `SummonerTeleport.UpgradedCooldown=240` and Unleashed CD at levels 2–9 | Read Unleashed TP tooltip CD at levels 1, 5, 9, 10, 15 |
| U-RQ-8 | Recall lock start (cast vs completion) | Recall, cancel at 4 s, check passive gain |
| U-RQ-9 | Whether +80 takedown XP is split/multiplied or flat per recipient | Assisted kill with quest complete, compare XP |

## 10. Test fixtures (numeric)
1. Passive only, always in lane from 65 s, no events: `quest_points(t) = 1.5·(t−65)`; completes at **t = 865.0 s**.
2. Out of lane all game, no events: `(t−65)/3`; at 600 s → 178.33 pts.
3. p = 0.5 (600 pts), last hit a mid-lane minion: +`2·(0.25+0.375)` = **+1.25**; take a mid-lane plate: +`40·0.625` = **+25**.
4. Recall completes at 300.0 s in lane: points 300.0→312.0 s gain 0 passive.
5. Roam bank: in lane 200 s at level ≥ 3 → bank 60 s (cap); then out of lane 90 s → first 60 s at 1.5/s (+90), next 30 s at 1/3 s (+10).
6. Completion at level 11 with total XP 7500 → +600 → 8100 → level 12 (8580 needed for 13). Later minion XP 60 → 66.6; an assisted champion kill share S → S + 80.
7. Unleashed TP cooldown with own TP + quest: L5 → 300−30 = **260 s**; L9 → 220; L12 → 210. Without quest: L12 → 240. Quest free TP: **390 s** flat.
8. TP shield: max HP 2000 → 700 HP shield for 10 s after arrival.

## 11. Diff vs current implementation
No role-quest code exists. `lanerl_jax/sim/modern_bridge.py`, `world/config.py`, `rewards.py`,
`step.py` contain no role/quest/level-cap state; level is capped at 18 via 19-row tables
(`profiles.py:333-375`, `rewards.level_for_xp` at `rewards.py:478`, `modern.skill_ranks` clips at
18, `modern.py:99`; `step.py:1457` `_RANK_TABLE[clip(level,0,18)]`). `state.py` has no summoner-
spell/Teleport state. All of §§1–6 is new work; level tables must grow to 21 rows (index 0–20).
