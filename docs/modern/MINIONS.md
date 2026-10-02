# Lane minions — implementable spec, patch 26.19 (normal PC Summoner's Rift, mode CLASSIC)

## 0. Header

| | |
|---|---|
| Scope | Lane minions on Summoner's Rift (melee, caster, siege/cannon, super): stats and per-upgrade growth, wave spawn schedule and composition, movement/sidelane speed, targeting/aggro/Call-for-Help, damage modifiers, gold/XP, death/despawn. Turret-side rules are only referenced (see turret spec). Champion kits out of scope. |
| Patch pin | 26.19, client build **16.19.8230722** (CommunityDragon `raw.communitydragon.org/16.19/`, `content-metadata.json` = `16.19.8230722+branch.releases-16-19.content.release`). Game mode **CLASSIC** (`GameModeMapData {0b03bf5a}`, `mModeName` = fnv1a("Classic") `{48246d53}`). Not Swiftplay, not "League Classic" (the retro mode added in 26.15+; its patch-note sections are excluded). |
| Retrieval date | 2026-10-01 |
| Author role | Research only. No code/data changed. |

### 0.1 Confidence tags used on every rule

- **[CLIENT]** — read directly from 16.19.8230722 client data (bins/stringtable/class defaults). Confidence of the *value* is HIGH; confidence of the *engine semantics* is given separately when inferred.
- **[RIOT]** — Riot patch notes 26.1–26.19.
- **[WIKI]** — League wiki revision cited.
- **[INFERRED]** — derived by reasoning from the above; always states its basis.
- **[LEGACY]** — taken from the LeagueSandbox 4.20 server only because no modern source exists; treat as a placeholder to be measured.
- Confidence: **H**/**M**/**L**.

### 0.2 Sources

Client files (cache dir `/mnt/nfs/shared/modern-world-map-research/cdragon-16.19/`, SHA256 in its `SHA256SUMS`):

| File | SHA256 | Used for |
|---|---|---|
| `sru_orderminionmelee.bin.json` | c82797e44c36fe58e5b076a3717f25ab181e22c72c5671af2eafc6f04a8b94e6 | CharacterRecords/Root + spells |
| `sru_chaosminionmelee.bin.json` | b6a7a8764bd6a9f311cadbabe0b344b6d322b8d54582935ccc0da18365a4edad | red parity check |
| `sru_orderminionranged.bin.json` | b470986d6c8ebdfd993435a88718884239dae22c64624ca3c8826ae22dedb8d6 | caster |
| `sru_chaosminionranged.bin.json` (fetched this pass) | e3344784acf59b81882dfdef6f8921257d75022d233a1860a0e69efb01ca6d38 | red parity |
| `sru_orderminionsiege.bin.json` | dd2a747737e2d62d0eeffe600f428ca1101fd667d125059c1977bf01818e439b | siege |
| `sru_chaosminionsiege.bin.json` (fetched this pass) | 3c3ad09be458115f5d645b45f5ce1656f88b47ce6f723af34bbdb55d258fa8b8 | red parity |
| `sru_orderminionsuper.bin.json` | ca9c7ac79a6ccbf7445d8ef2e1cd5aca22adbfa0538d6380fbbeb12f58264a5f | super |
| `sru_chaosminionsuper.bin.json` (fetched this pass) | 295e310266d6d85002c4479e7bbf912b1910d1b377021bd5a346b9f9cab7c8c7 | red parity |
| `items.cdtb.bin.json` | 6880f35d5a9d82f688192f764e280e6d4bc9c845112b001feb811e3f2ab62726 | minion "rule items" 1508–1512 (`mDataValues`) |
| `shared.cdtb.bin.json` | 34f68553ab38cfe344936473fcb48d99574d5fd51e4feb42994fa0c4fe50769e | shared buffs (SuperMinionAura, MinionPushingPower, SR_2026_S1_MinionFrenzy_Buff) |
| `lol.stringtable.en_us.json` (fetched this pass from `game/en_us/data/menu/en_us/lol.stringtable.json`) | 8c051cb2a24b31f3fa9af95d39085b832b0da2b8cf51093f98c4ab0620ecb8e8 | tooltips of items 1508–1512, buff text |
| `/mnt/nfs/shared/modern-world-map-research/map11-decoded.json` | (map11.bin SHA 45d16148616eb3da612e31f422afb46d756d5fc615f8387b7429343bda5da88a, per ledger MODERN-010) | `BarracksConfig {147211fb}` (Order) / `{e61e55a3}` (Chaos); `GameModeConstants {6cf687be}` (CLASSIC); `GameplayConfig {a6506f8a}`; `ExperienceModData {0059202c}` |
| LeagueToolkit `lol-meta-classes` dump `dumps/16.19.8217343.json` (same patch, build 8217343; not cached — re-fetch from `https://raw.githubusercontent.com/LeagueToolkit/lol-meta-classes/main/dumps/16.19.8217343.json`) | — | class **default values** (e.g. `CharacterRecord.acquisitionRange` default = 750) and full field lists of `MinionUpgradeConfig` |
| CDTB hash lists `hashes.binfields.txt`, `hashes.binhashes.txt` (raw.communitydragon.org/data/hashes/lol/) | — | resolving hashed constant names |
| `lanerl_jax/data/modern/26.19/geometry.json` | source `base_srx.materials.bin` ce17dbee…2f55 | barracks positions, `MinionPath_*` lane splines |

Hash identities resolved this pass (FNV-1a lowercase): `{726ae049}` = `GoldUpgrade`; `{fee040bc}` in `BarracksMinionConfig` = link to the minion object definition (`{4cdad893}` = Order melee, `{2f33d8db}` = Chaos melee, etc.). The two 90-s BarracksConfigs `{147211fb}`/`{e61e55a3}` are the Order and Chaos barracks of the same mode (they differ only in those links and one super-minion field, §1.4).

Riot patch notes (all fetched 2026-10-01; URL form recorded):
26.1 `…/game-updates/patch-26-1-notes/` (Minions section, Game Start Time, Role Quests, Swiftplay "Minion Frenzy"); 26.2, 26.3 (`patch-26-N-notes/`); 26.4–26.19 (`league-of-legends-patch-26-N-notes/`). Minion-relevant: **26.1, 26.9, 26.10, 26.11, 26.16** (Classic-mode only lines excluded), 26.17/26.19 Call-for-Help bug fixes are under the **League Classic** heading (excluded, §3.9). Mid-patch/hotfix sections (26.1 1/9 update, 26.3 Feb-5 hotfix, 26.6, 26.9) contain no minion changes.

Wiki (wiki.leagueoflegends.com/en-us), revisions read:
[Minion oldid 4068797](https://wiki.leagueoflegends.com/en-us/Minion?oldid=4068797) (2026-09-28), [Melee minion oldid 4068807](https://wiki.leagueoflegends.com/en-us/Melee_minion?oldid=4068807), [Caster minion oldid 4015019](https://wiki.leagueoflegends.com/en-us/Caster_minion?oldid=4015019), [Siege minion oldid 4013294](https://wiki.leagueoflegends.com/en-us/Siege_minion?oldid=4013294), [Super minion oldid 4068820](https://wiki.leagueoflegends.com/en-us/Super_minion?oldid=4068820), [Experience (champion) oldid 4053165](https://wiki.leagueoflegends.com/en-us/Experience_(champion)?oldid=4053165), [Kill oldid 4053216](https://wiki.leagueoflegends.com/en-us/Kill?oldid=4053216), [Gold oldid 4039269](https://wiki.leagueoflegends.com/en-us/Gold?oldid=4039269), [Turret oldid 4070072](https://wiki.leagueoflegends.com/en-us/Turret?oldid=4070072), [Inhibitor oldid 4070357](https://wiki.leagueoflegends.com/en-us/Inhibitor?oldid=4070357), [Movement speed oldid 4064468](https://wiki.leagueoflegends.com/en-us/Movement_speed?oldid=4064468), [Sight oldid 4035710](https://wiki.leagueoflegends.com/en-us/Sight?oldid=4035710). Historical diffs used: Minion 4003116→4017998 (26.10 priority removal), 3953340→3953637 (death-grace delay 0.35→0.035 s), 3978992→3978998 (minion-pushing DR "×100" revert).

Project context read: `docs/MODERN_PATCH_DELTA.md` §5 (pinned 26.18; re-verified here — several of its numbers are corrected below), `docs/JAX_FIDELITY_LEDGER.md` MODERN-001..012, `docs/LEAGUE_MECHANICS_CONCEPTS.md` §2–3, `docs/PORT_AUDIT_AI.md`, `lanerl_jax/sim/modern_minions.py`, `modern_world.py` (uncommitted, read only), `targeting.py`, `minion_ai.py`, `profiles.py`, `data/modern/26.19/minions.json`, `geometry.json`.

### 0.3 Headline corrections vs. prior research (`MODERN_PATCH_DELTA.md` §5) and wiki

1. **Minion damage to champions is 55 %, not 60 %** [CLIENT H]: CLASSIC `GameModeConstants {6cf687be}` → `dr_UnitToHero = 0.55`, `dr_UnitToBuilding = 0.60`, `dr_UnitToUnit = 1.0`. Matches Riot 25.S1.1 (both → 55 %) + 26.1 (turrets only → 60 %). The wiki "60 % vs champions" sentence is unsupported.
2. **`*Late` upgrade fields are additive to the early increment** [CLIENT+RIOT H]: per-upgrade gain after `UpgradesBeforeLateGameScaling = 5` upgrades is `Upgrade + UpgradeLate`. Proven by: 26.9 notes "Melee HP 440 (+25 every 90s; then +35 after 5 times)" = `HPUpgrade 25 + HPUpgradeLate 10` (still visible in another mode's barracks `{4b1f14b9}`); caster `1.5 + 2.5 = 4` (wiki +4); siege `1.5 + 2.5 = 4` (26.1 notes "+4 every 90 seconds"). **Siege AD therefore grows +4 from U = 6**, which MODERN_PATCH_DELTA §5.2 omitted.
3. Melee/siege **acquisitionRange = 750** (class default, omitted from the record), caster 700, super 600 [CLIENT H value / M semantics]; wiki "500" disagrees.
4. Death-grace delay is **0.035 s** (`mMinionDeathDelay`), not 0.066 s; cap 190 (`mMinionAAHelperLimit`) [CLIENT H].
5. XP split table values are client floats `[1.0, 0.65, 0.433, 0.325, 0.26, 0.217]` (not 13/30, 13/60) [CLIENT H].
6. Red-team super minions have **no GoldUpgrade** (flat 49 g) while blue supers have +1/upgrade [CLIENT H value; likely a data bug, M].

## 1. Unit stats

### 1.1 Base CharacterRecord values (Root record; Order and Chaos identical except cosmetics) [CLIENT H]

Blue/red diff of all numeric Root fields: only `mFallbackCharacterName` differs, plus `experienceRadius` absent on `SRU_ChaosMinionSiege` (irrelevant — barracks `ExpRadius` governs, §5.2). **No blue/red stat asymmetry.**

| Field (bin name) | Melee | Caster (`…Ranged`) | Siege | Super | Notes |
|---|---|---|---|---|---|
| `baseHPModifiable` | 430 | 275 | 750 | 1500 | U=0 value; first wave already has U=1 (§1.3) |
| `baseDamageModifiable` (AD) | 11 | 19.5 | 36 | 180 | |
| `baseArmorModifiable` | 0 | 0 | 0 | 100 | |
| `baseMR` | 0 (default) | 0 | 0 | −30 | |
| `baseStaticHPRegenModifiable` | 0 | 0 | 0 | 10 | super also has unresolved field `{2290fc9a}` = 0.0015 (likely %-HP regen); see §1.5 |
| `baseMoveSpeedModifiable` | 350 | 350 | 350 | 350 | time increases §2.6 |
| `attackRangeModifiable` | 110 | 550 | 300 | 170 | range measured edge-to-edge (attacker center to target gameplay-radius edge, standard engine rule) [INFERRED H] |
| `attackSpeedModifiable` | 1.25 | 0.667 | 1.0 | 0.85 | |
| `attackSpeedRatioModifiable` | 1.25 | 0.667 | (default 1.0) | 0.85 | only matters for bonus AS (none on SR lane minions except Baron/mid-siege, §4.6) |
| `basicAttack.mAttackTotalTime` (s) | 0.8 | 1.5 | 1.0 | 1.44 | |
| `basicAttack.mAttackCastTime` (s) | 0.393 | 0.47 | 0.30 | 0.50 | equals spell `castFrame/30` (11.8, 14.1, 9.0, 15.0) |
| Windup at base AS = `castTime/totalTime / AS` | **0.393 s** | **0.470 s** | **0.300 s** | **0.408 s** | [INFERRED H] standard windup formula; super's total-time ≠ 1/AS so windup fraction 0.3472 applies |
| Attack period `1/AS` | 0.800 s | 1.4993 s | 1.000 s | 1.1765 s | |
| Missile speed (`mSpell.missileSpeed`) | melee (0) | **650** | **1200** | melee (0) | caster `_Frenzy` variant 1300 is Swiftplay-only |
| `acquisitionRange` | **750** (class default) | **700** | **750** (class default) | **600** | default from meta dump `CharacterRecord.acquisitionRange F32 default 750.0` |
| `firstAcquisitionRange` | 1000 | 900 | — | — | first-wave rule, §3.6 [INFERRED M] |
| `wakeUpRange` | 450 | 635 | — | — | first-wave "ignore champions until much closer", §3.6 [INFERRED M] |
| `perceptionBubbleRadius` (sight) | 1200 | 1200 | 1200 | (none; wiki sight 1350) | [WIKI Sight] agrees |
| `overrideGameplayCollisionRadius` | 48 | 48 | 65 | 65 | unit-vs-unit collision / range edge |
| `pathfindingCollisionRadius` | 35.7437 | 35.7437 | 55.7437 | 55.5208 | navgrid clearance |
| `selectionRadius` | 115 | 115 | 140 | 145 | UI only |
| `towerTargetingPriorityBoost` | 2 | 1 | 5 | 6 | turret preference: super > siege > melee > caster (matches wiki Turret priority) |
| `expGivenOnDeath` | 62 | 31 | 75 | 75 | flat (no `ExpUpgrade`) |
| `goldGivenOnDeath` | 20 | 14 | 49 | 49 | siege/super + `GoldUpgrade`·U |
| `critDamageMultiplier` | 2 | 2 | 2 | 2 | minions never crit (no crit chance) |
| `unitTagsString` | `Minion \| Minion_Lane \| Minion_Lane_Melee` | `…_Ranged` | `…_Siege` | `…_Super` | |
| Rule item (`mClientSideItemInventory`) | 1509 Gusto | 1510 Phreakish Gusto | 1508 Anti-tower Socks | 1511 Super Mech Armor + 1512 Super Mech Power Field | §4 |

The CharScripts `CharScriptSRU_*Minion*_Jade` in each bin (FreeUpgrades = floor(max(0,(t−60)/90)), FreeGold/FreeExp…) are gated on a mode flag and belong to a different (non-CLASSIC) mode. **Do not implement.** [CLIENT; semantics INFERRED M]

### 1.2 CLASSIC barracks config `{147211fb}` (Order) / `{e61e55a3}` (Chaos) [CLIENT H]

```
InitialSpawnTimeSecs              30.0
WaveSpawnIntervalSecs             30.0      (see §2.2 for the 25 s / 20 s phases)
MinionSpawnIntervalSecs           0.800000011920929
UpgradeIntervalSecs               90.0
UpgradesBeforeLateGameScaling     5
MoveSpeedIncreaseInitialDelaySecs 600.0
MoveSpeedIncreaseIntervalSecs     300.0
MoveSpeedIncreaseIncrement        25
MoveSpeedIncreaseMaxTimes         4
ExpRadius                         1500.0
goldRadius                        1250.0
```

`MinionUpgradeStats` (`MinionUpgradeConfig`; absent fields = class default 0):

| MinionType | Unit | HPUpgrade | HPUpgradeLate | HpMaxBonus | DamageUpgrade | DamageUpgradeLate | DamageMax | ArmorUpgrade | ArmorUpgradeGrowth | ArmorMax | GoldUpgrade `{726ae049}` | GoldMax |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4 | melee | 35 | 0 | 1120 | 0 | 3 | 69 | 0 | 0.085 | 20 | 0 | 0 |
| 5 | caster | 9 | 0 | 325 | 1.5 | 2.5 | 105.5 | 0 | 0 | 0 | 0 | 0 |
| 6 | siege | 85 | 0 | 5100 | 1.5 | 2.5 | 90 | 0 | 0 | 0 | 1 | 90 |
| 7 | super | 100 | 0 | 6000 | 5 | 0 | 300 | 0 | 0 | 0 | **1 (Order) / absent (Chaos)** | 90 |

Other `MinionUpgradeConfig` fields exist in the 16.19 class (`HPUpgradeGrowth`, `HPUpgradeGrowthLate`, `HPInhibitor`, `DamageInhibitor`, `MagicResistance(Upgrade)`, `LocalGoldGivenOnLastHit`, `ExpUpgrade`, two unnamed) — all **unset (0)** for CLASSIC SR.

`lanerl_jax/data/modern/26.19/minions.json` copies this config faithfully (Order barracks). Its note "HP/AD growth caps interpreted as bonus caps; armor growth is accumulated per-upgrade growth" is correct for HP/AD (§1.3) — but the file does not record the Chaos variant or the class default acquisitionRange.

### 1.3 Per-upgrade stat formulas — the authoritative definitions

Upgrade index of a minion, `U` (integer ≥ 1). **Latched at spawn** (default, [INFERRED M]; see §8 U-1). The barracks upgrade counter:

```
U_barracks(t) = 0                                   for t < 30
              = 1 + floor((t - 30) / 90)            for t >= 30      [CLIENT InitialSpawnTime/UpgradeInterval + WIKI "starting at 0:30 and every 90 s"] (H)
```
At a tie (`t` exactly an upgrade time and a wave time, e.g. 0:30, 14:00, 21:30), upgrade fires first → the wave gets the new U. Evidence: wave 1 at 0:30 has 465 HP = 430+35 (U=1) [WIKI infobox lower bound + RIOT 26.9 "neutral at U=1"] (H).

Let `E = min(U, 5)` (early upgrades), `L = max(U - 5, 0)` (late upgrades). For each stat with `Up`, `UpLate`, `MaxBonus`:

```
bonus(U)  = Up * U + UpLate * L            # UpLate ADDITIVE on top of Up for upgrades 6+  [CLIENT+RIOT H]
stat(U)   = base + min(bonus(U), MaxBonus) # Max* fields are caps on the BONUS              [CLIENT+WIKI H]
```

Evidence for cap-on-bonus: `19.5 + 105.5 = 125`, `11 + 69 = 80`, `430 + 1120 = 1550`, `275 + 325 = 600` (round totals); wiki infobox formula text "capped at (430 + 1120)" etc.

| Stat | Melee | Caster | Siege | Super |
|---|---|---|---|---|
| HP | `430 + min(35U, 1120)` → cap 1550 at U=32 | `275 + min(9U, 325)` → 600 at U≥37 (599 at 36) | `750 + min(85U, 5100)` → 5850 at U=60 | `1500 + min(100U, 6000)` → 7500 at U=60 |
| AD | `11 + min(3L, 69)` → 80 at U=28 | `19.5 + min(1.5U + 2.5L, 105.5)` → 125 at U≥30 | `36 + min(1.5U + 2.5L, 90)` → 126 at U≥26 | `180 + min(5U, 300)` → 480 at U=60 |
| Armor | `armor_melee(U)` below, cap 20 | 0 | 0 | 100 (+ aura, §4.5) |
| MR | 0 | 0 | 0 | −30 |
| Gold | 20 | 14 | `min(49 + U, 90)` | Order `min(49 + U, 90)`; **Chaos 49** |
| XP | 62 | 31 | 75 | 75 |

Melee armor (`ArmorUpgradeGrowth = 0.085`, no `ArmorUpgrade`): the field name pattern (`HPUpgradeGrowth` historically produced `0.3*(U-1)*U/2`, per the pre-25.S1.1 wiki melee-HP formula) means "the per-upgrade increment grows by 0.085 each upgrade". Whether growth counts from the first upgrade or only from late upgrades is **not determinable from data**. Default = wiki formula [WIKI L]:
```
armor_melee(U) = 0                                   for U <= 5
               = min(0.085 * (U - 6) * (U - 5) / 2, 20)   for U >= 6     # 0 @6, 0.085 @7, 0.255 @8, 0.85 @10
```
Alternatives (keep behind a flag): (B) `0.085*U*(U-1)/2` from U=1; (C) `0.085*L*(L+1)/2`. Distinguishable at ~20 min (§8 U-2). In a 0–10 min episode (U ≤ 7) the three give 0.085 / 1.785 / 0.255 armor.

`GoldMax` semantics: treated as an **absolute total cap** [INFERRED M — pre-26.01 wiki/notes "57 + 3/upgrade, capped at 90" with the same GoldMax 90]. Unreachable before U = 41 (≈ 61:30) so practically irrelevant.

Wiki vs client:

| Item | Client | Wiki | Implement |
|---|---|---|---|
| Siege AD late growth | +4/upgrade from U=6 | "(+4 instead at wave 15+)"; Riot 26.1 "then +4 every 90 seconds at wave 15" | client: from U=6 (8:00). Wave 15 (7:30) is U=5. Riot/wiki phrasing is loose by one wave. M |
| Super AD | `180 + 5U` (185 at U=1) | infobox `180+5x` with stale text "210 + 5 per upgrade" | client |
| Melee/caster/siege/super HP, melee/caster AD | match | match | — |
| Melee armor | growth 0.085 only | (U−6)(U−5)/2·0.085 | wiki shape, flagged L |
| Siege gold | 49 + U | "50 + 1 every upgrade" (same thing with U=1 at 0:30) | client |

### 1.4 Precomputed table (U at spawn) [derived from §1.3]

| U | first wave time with this U | Melee HP/AD/armor | Caster HP/AD | Siege HP/AD/gold | Super HP/AD |
|---|---|---|---|---|---|
| 1 | 0:30 | 465 / 11 / 0 | 284 / 21.0 | 835 / 37.5 / 50 | 1600 / 185 |
| 2 | 2:00 | 500 / 11 / 0 | 293 / 22.5 | 920 / 39.0 / 51 | 1700 / 190 |
| 3 | 3:30 | 535 / 11 / 0 | 302 / 24.0 | 1005 / 40.5 / 52 | 1800 / 195 |
| 4 | 5:00 | 570 / 11 / 0 | 311 / 25.5 | 1090 / 42.0 / 53 | 1900 / 200 |
| 5 | 6:30 | 605 / 11 / 0 | 320 / 27.0 | 1175 / 43.5 / 54 | 2000 / 205 |
| 6 | 8:00 | 640 / 14 / 0 | 329 / 31.0 | 1260 / 47.5 / 55 | 2100 / 210 |
| 7 | 9:30 | 675 / 17 / 0.085 | 338 / 35.0 | 1345 / 51.5 / 56 | 2200 / 215 |
| 10 | 14:00 | 780 / 26 / 0.85 | 365 / 47.0 | 1600 / 63.5 / 59 | 2500 / 230 |
| 12 | 17:00 | 850 / 32 / 1.785 | 383 / 55.0 | 1770 / 71.5 / 61 | 2700 / 240 |
| 17 | 24:30 | 1025 / 47 / 5.61 | 428 / 75.0 | 2195 / 91.5 / 66 | 3200 / 265 |
| 20 | 29:00 | 1130 / 56 / 8.925 | 455 / 87.0 | 2450 / 103.5 / 69 | 3500 / 280 |
| 28 | 41:00 | 1410 / 80 / 20 (cap) | 527 / 119.0 | 3130 / 126 (cap) / 77 | 4300 / 320 |

Wiki "wave gold value" table cross-check: siege gold 59 @14:00 (U=10), 66 @25:00 (U=17), 69 @30:00 (U=20) ✓.

### 1.5 Regeneration [CLIENT H / WIKI M]

Lane melee/caster/siege: 0 HP regen, no mana. Super: `baseStaticHPRegen 10` (per 5 s, i.e. 2 HP/s, engine convention [INFERRED M]) plus unresolved `{2290fc9a} = 0.0015`. Wiki super infobox: "67.5 HP5 for upgrades 1–8, then +0.775/upgrade (estimated)". The two do not reconcile; super regen is LOW priority for a lane sim — implement wiki's estimate if supers are enabled, flag L.

## 2. Waves: schedule, composition, spawn, movement

### 2.1 Wave index convention

Zero-based wave index `i` (wave 0 = 0:30). The client `RotatingWaveBehavior.SpawnCountsByWave` lists are indexed by **global zero-based wave count modulo list length** [INFERRED H — it is the only indexing that reproduces every wiki-listed cannon time: 1:30, 12:00, 13:30, 14:25, 15:15, 24:25, 25:15, 25:40].

### 2.2 Spawn times [RIOT 26.1 H + WIKI H; client carries only the 30 s base]

Riot 26.1: "At 14 minutes, minion waves now spawn every 25 seconds… At 30 minutes, minion waves now spawn every 20 seconds." The client CLASSIC BarracksConfig only has `WaveSpawnIntervalSecs 30`; the cadence change must live in code/script not exported to bins. Wiki gives exact sequences (13:30, 14:00, 14:25 … 29:25, 29:50, 30:10):

```
t_wave(i) = 30 + 30*i                for 0  <= i <= 27     # i=27 -> 14:00 (840 s)
          = 840 + 25*(i - 27)        for 27 <= i <= 65     # i=65 -> 29:50 (1790 s)
          = 1810 + 20*(i - 66)       for i >= 66           # i=66 -> 30:10
```
(The 25→20 s transition: last 25-s interval 29:50; next wave at 30:10 = 20 s after. H per wiki, matches current `wave_spawn_time`.)

Practice Tool uses a different (30 s) cadence per wiki edit 4065342 — irrelevant to CLASSIC.

### 2.3 Composition per wave [CLIENT H for counts; WIKI H for super/siege replacement]

Client behaviours (CLASSIC, both teams):

| Unit | t < 840 | 840 ≤ t < 1500 | 1500 ≤ t < 1800 | t ≥ 1800 |
|---|---|---|---|---|
| Melee | Constant 3 | Rotating `[2,3]` → `2 if i%2==0 else 3` | Constant 2 | Constant 2 |
| Siege | Rotating `[0,0,1]` → `1 if i%3==2` | Rotating `[1,0]` → `1 if i%2==0` | Constant 1 | Constant 1 |
| Caster | Constant 3 | 3 | 3 | **Constant 2** (from 1800) |
| Super | `InhibitorWaveBehavior.SpawnCountPerInhibitorDown = [1,1,2]` | | | |

`t` = the wave's spawn time; a `TimedWaveBehaviorInfo` applies when `t >= StartTimeSecs`.

Resulting wave list: waves 0,1 = 3M+3C; wave 2 (1:30) first cannon; cannons at i ≡ 2 (mod 3) up to i=26 (13:30); i=27 (14:00) no cannon; i=28 (14:25) cannon with 2 melee; cannons on even i through i=52 (24:25); i=53 (24:50, t=1490 < 1500, odd) has no cannon and 3 melee; from i=54 (25:15, t=1515) every wave has a cannon and 2 melee; from i=66 (30:10) 2 casters. (Note the 29:25/29:50 waves still have 3 casters; the caster drop is keyed to t ≥ 1800 and no wave falls in [1790,1810).)

Super minions [WIKI H + CLIENT M]:
- `n_inhib_down` = number of *this team's enemy* inhibitors currently destroyed. Index `SpawnCountPerInhibitorDown[n_inhib_down - 1]` gives supers per wave **in lanes whose enemy inhibitor is down**; with 3 down, every lane gets 2. With 1 or 2 down → 1 super in each lane whose inhibitor is down. [INFERRED H from wiki "two super minions spawn on each lane if all inhibitors destroyed".]
- A wave containing ≥1 super spawns **no siege minion** (super replaces cannon) [WIKI H].
- Melee count is **not** changed by super replacement — client melee rotation is independent of siege [CLIENT H counts / INFERRED M interaction]. E.g. wave 28 with top inhib down: 1 super, **2** melee, 0 siege, 3 casters.
- Supers stop spawning "two waves before the inhibitor respawns" (inhibitor respawn 300 s) [WIKI M]. Implement: no super if `inhib_respawn_time - t_wave < 2 * interval(t_wave)` [INFERRED L on the exact comparison].

Spawn order inside a wave [WIKI H]: supers → melee → siege → casters.

### 2.4 Intra-wave spacing [CLIENT H value]

`MinionSpawnIntervalSecs = 0.8` (float 0.800000011920929). Wiki says 0.79 ("rutngt|0.79", edit 3899497 "Added inter-wave spawn interval"); the 0.792 in current code has no client source (probably a measurement of the wiki "0.79"). **Implement 0.8 s** [CLIENT H]; the wiki 0.79 is a measurement rounding. Unit k (0-based) of wave i spawns at `t_wave(i) + 0.8*k`.

### 2.5 Spawn position and lane path [CLIENT H geometry; INFERRED M behaviour]

From `geometry.json` (client `base_srx.materials.bin`): barracks (XZ)

| Team | Bot (lane 0) | Mid (lane 1) | Top (lane 2) |
|---|---|---|---|
| Order (0) | (2034, 1171) | (2008, 2079) | (1109, 2091) |
| Chaos (1) | (13719, 12845) | (12776, 12784) | (12800, 13745) |

Lane spline `MinionPath_Top` (16 points, Order→Chaos direction) starts (1508, 3597) and ends (11348, 13344); Chaos minions follow it reversed. Minions spawn at the barracks point, walk to the first spline point of their direction, then follow successive points. All units of a wave spawn at the same point; separation comes from the 0.8 s stagger plus collision. Waypoint advance: legacy rule "reached when within 25 units" (`WAYPOINT_MARGIN`) [LEGACY L]. After the last spline point, path to the enemy Nexus (targeting structures, §3.7).

### 2.6 Movement speed [CLIENT H values; timing INFERRED L]

Base 350 [CLIENT; RIOT 26.1]. Time increases: +25 up to 4 times (`MoveSpeedIncreaseIncrement 25`, `MaxTimes 4` → max 450).

Timing ambiguity: `MoveSpeedIncreaseInitialDelaySecs = 600`, `IntervalSecs = 300`. Wiki: "+25 at 11:00, 16:00, 21:00, 26:00" (written when first spawn was 1:05; formula text stale). Candidates:
- (A, default) delay counted from first spawn: increases at **10:30, 15:30, 20:30, 25:30** (30+600+300k). Rationale: matches wiki "11:00" under the 25.S1.1 1:05 start (65+600 = 11:05), and the upgrade timer is also anchored at InitialSpawnTime.
- (B) absolute: 10:00, 15:00, 20:00, 25:00.
- (C) wiki literal 11:00, 16:00, …
Default A, flag L (§8 U-3). Whether it applies to living minions or only new spawns: default **global, applies to all living minions immediately** (it is a base-MS change) [INFERRED L].

### 2.7 Sidelane speed buff (top and bot only) [WIKI M; RIOT 13.10/14.22]

```
for wave number n = i + 1 (1-based), side lanes only, n >= 2, and t_wave < 840:
    B = max(0, 120 - 4.5 * n)                # n=2 -> 111, n=26 -> 3, n>=27 -> 0
    bonus_ms(τ) where τ = time since this minion spawned:
        τ in [0, 7)   : B
        τ in [7, 14)  : max(0, B - 15)
        τ in [14, 21) : max(0, B - 30)
        τ in [21, 25) : max(0, B - 45)
        τ >= 25       : 0 (buff removed)
```
Flat bonus MS, added before multiplicative modifiers, then MS soft caps apply (`raw > 415: 0.8*raw + 83`; `raw > 490: 0.5*raw + 230`) [WIKI Movement speed H for champions; INFERRED M for minions]. Wave 1 never gets it (wiki V14.22 note). The `max(0, …)` clamps are [INFERRED M]. Mid lane: no buff.

Effective speeds wave 2 at 350 base: 461→451.8 (0–7 s), 446→439.8 (7–14 s), 431→427.8 (14–21 s), 416→415.8 (21–25 s), 350 after.

### 2.8 First wave special behaviour [WIKI M]

- Ghosted (no unit collision) **28 s** after spawn for top/bot (18 s mid).
- Ignores enemy champions "until a champion walks much closer" — implement as: during wave 1 (until it first engages the enemy wave), champions are valid targets only within `wakeUpRange` (melee 450, caster 635) [INFERRED M — field name and values fit].
- "When the waves meet, minions spread their attacks on the three enemy melee minions regardless of distance" — implement: first-wave minions use `firstAcquisitionRange` (melee 1000, caster 900) for minion targets and restrict candidates to enemy **melee** minions while any is alive, assigning targets to balance attacker counts (fewest current attackers first, ties by distance). Data hint: CLASSIC `ai_TargetMaxNumAttackers = 5`, `ai_TargetDistanceFactorPerAttacker = 0.8`, `ai_TargetDistanceFactorPerNeightbor = 0.6`, `ai_TargetRangeFactor = 0.7` [CLIENT H values, semantics unknown]. Exact assignment algorithm UNRESOLVED (§8 U-4). Default deterministic rule: sort own first-wave minions by spawn order; minion k targets enemy melee `k mod 3` (both waves symmetric), re-picking the least-attacked living enemy melee when its target dies.
- First wave also ignores the "minions attacking a turret ignore Call for Help" exception (wiki wording "Bar the first wave") — irrelevant in practice.

### 2.9 Collision / steering [INFERRED M; LEGACY]

Minions collide with all units using gameplay radius (48/65) except while ghosted; they respect player-generated terrain. Client CLASSIC constants `ai_PostAvoidanceFilterDuration 0.3`, `ai_PostAvoidanceRotFilterStrength 0.2`, `…Accel 0.125` describe a local-avoidance smoothing filter (semantics unknown). Keep the existing shared collision/steering module; no modern numbers beyond radii are available.

### 2.10 Catch-up / "pushing advantage" / Frenzy (2026)

- **Minion Frenzy / "Inspired Minion"** (`SR_2026_S1_MinionFrenzy_Buff`): Riot 26.1 lists it under **Swiftplay only** ("we're adding a new feature to Swiftplay only: Minion Frenzy"). No later note extends it to CLASSIC. **Do not implement for CLASSIC** [RIOT H].
- No wave catch-up/rubber-band mechanic in CLASSIC 26.1–26.19 notes [RIOT H]. ("Rubberbanding start time" is Swiftplay.)
- Minion Pushing buff (level advantage) — §4.4.

## 3. Targeting, aggro, Call for Help

### 3.1 Priority list (post-26.10) [RIOT 26.10 H; WIKI oldid 4068797 H]

Lower number = higher priority:

| P | Candidate |
|---|---|
| 1 | Enemy champion attacking an allied champion |
| 2 | Enemy minion attacking an allied champion |
| 3 | Enemy minion attacking an allied minion |
| 4 | Enemy turret attacking an allied minion |
| 5 | Closest enemy minion |
| 6 | Closest enemy champion |
| 7 | Structures (turret/inhibitor/nexus) when nothing else — §3.7 [INFERRED H; legacy ClassifyUnit turret 10, inhibitor 12, nexus 13] |

Riot 26.10 verbatim: "Previously, when an enemy champion attacked an allied minion, the enemy champion would enter the minions' aggro priority list. We're removing this action entirely from aggroing minions…". So a champion last-hitting/auto-attacking minions is treated as plain P6 (and is **not** a Call-for-Help source). Wiki diff 4003116→4017998 removed exactly that line.

Within-class ordering: wiki says "closest". Legacy (4.20 ClassifyUnit) ranked siege/super(7) < caster(8) < melee(9) inside "minion"; **no modern source supports a type preference** → default **pure distance** within P5, flag (§8 U-5). Distance metric: center-to-center [INFERRED M]. CLASSIC constant `ai_MinionTargetingHeroBoost = 150` [CLIENT H value, semantics unknown] — most plausible reading: champions' effective distance is increased by 150 when compared against other candidates; irrelevant to a strict-priority implementation because P6 < P5 anyway. Record, do not implement.

Hysteresis [WIKI H, verbatim in 2014 and 2026]: a held target is replaced only by a candidate of **strictly higher** priority (lower P). Equal priority never steals, even if closer.

"Attacking" for P1–P4 means: the candidate's current attack target is an allied unit of that kind *and* it has damaged/attacked it recently. Default memory window: an attack counts while the candidate is mid-attack on that target or landed damage on it within the last **2.0 s** [INFERRED L; wiki: "minions will not stop targeting that champion for a short time afterwards"]. (Legacy uses one-shot per-hit call-for-help events; see §3.3.)

### 3.2 Acquisition ranges [CLIENT H values, M semantics]

| | Melee | Caster | Siege | Super |
|---|---|---|---|---|
| `acquisitionRange` (normal scan, keep-target range) | 750 | 700 | 750 | 600 |
| First wave minion scan (`firstAcquisitionRange`) | 1000 | 900 | — | — |
| First wave champion wake-up (`wakeUpRange`) | 450 | 635 | — | — |

Wiki: "valid targets must be within 500 units… allied champions being attacked by an enemy champion may be within 1000 range of the minion to trigger a Call for Help". Default: **client per-type acquisitionRange for the scan**; **Call-for-Help distances per wiki** (500 general / 1000 for champion-attacks-champion) because the wiki numbers are specifically about CFH and no client field covers them [WIKI M]. Distance test strict `<` (legacy `DistanceSquared < r²`) [LEGACY L].

Target is dropped when: it dies; becomes untargetable/invulnerable-untargetable; is no longer visible to the minion's team (§3.8); or leaves `acquisitionRange` + chase allowance (§3.5).

### 3.3 Call for Help (CFH) [WIKI H triggers; numbers M]

Triggers (an allied *listener* minion receives a CFH naming attacker A):
1. An enemy champion A deals damage to an allied champion V with a basic attack, a unit-targeted ability flagged CFH, or a non-targeted ability explicitly flagged CFH (per-ability flags belong to the champion/item specs; the wiki records them in each ability's Details table). Listener must be within **1000** of V and within its own scan range of A [WIKI M; legacy requires the listener to be in range of both victim and attacker].
2. An enemy minion/turret damaging an allied minion or allied champion (P2–P4 sources) within **500** of the listener [WIKI M / LEGACY].
3. "A champion stands in the minion's path, without any other targets within the minion's attack range and outside of a turret's range" → simply acquire the champion as P6 [WIKI H].

**Removed (26.10):** enemy champion damaging an allied minion is **not** a trigger [RIOT H].

On receipt: compute `P = class(A, V)` from §3.1; if `P < P(current target)`, switch immediately (does not wait for the reevaluation timer) [LEGACY H for the immediate switch; WIKI "re-evaluate between windups"]. If several A, choose lowest P, then closest A ("Minions will prioritize the closest champion in the case there are multiple sources") [WIKI H].

Exceptions:
- A minion whose current target is an enemy **turret** ignores CFH (except first wave) [WIKI H, RIOT 13.10].
- A minion in its attack windup finishes the windup before switching (switch occurs "in between each attack windup") [WIKI M]. Default: switch is applied at the next point the minion is not in windup; an attack already launched (missile in flight) still lands.

### 3.4 Reevaluation cadence [LEGACY M — no modern source]

Keep the legacy controller timing: regular sweep every **250 ms** (timer reset on each sweep), immediate re-evaluation when the target dies or a CFH arrives; a held, valid target is **not** re-prioritised by the regular sweep (only by CFH). Give-up: if the minion has a target but has not attacked for **4.0 s**, ignore that unit for **0.5 s** and re-acquire. Wiki "every few seconds" is compatible. Measure (§8 U-6).

### 3.5 Chase / leash

No modern leash data. Lane minions chase a held target while it remains visible and within `acquisitionRange` of the minion at each evaluation (legacy `IsValidTarget` range test) [LEGACY M]; on loss, return to lane path following from the nearest forward waypoint (legacy "advance waypoint index while reached") [LEGACY M]. They do not have a home-leash distance (unlike monsters).

### 3.6 First wave
See §2.8.

### 3.7 Structures

- With no unit candidate, minions attack the nearest enemy structure on their path: turret → inhibitor (when its turret is dead / it is targetable) → nexus turrets → nexus [WIKI H for existence; LEGACY order].
- Turrets are targetable by minions only when not protected (inner/inhib/nexus turret protection rules belong to the turret spec).
- Minions attacking a turret: ignore CFH (§3.3); can still switch via the regular scan only if their turret target becomes invalid.
- Mid-lane siege minions gain +30 % AS while attacking turrets (`MidLaneAttackSpeedBonus 0.3`, granted on hitting a turret, removed on hitting a non-turret) [CLIENT H + WIKI H]. Top-lane sim: not applicable.

### 3.8 Vision / brush / stealth [WIKI M]

"When the minion loses sight of its target, the minion switches to a new target or keeps advancing." Minions only consider candidates visible to their team (team vision, including brush rules and stealth). A champion entering brush that no ally of the minion sees is dropped as a target at the next evaluation (immediately on vision loss per legacy `UpdateTarget`) [LEGACY M]. In-flight missiles still hit.

### 3.9 Patch audit 26.11–26.19 for targeting [RIOT H]

- 26.11: no minion-system change; Heimerdinger turrets gain +50 range vs minions (champion spec) — the note confirms 26.10's removal applies to **ranged** minions too ("With the ranged minion aggro pull removed last patch…").
- 26.12–26.16, 26.18: none.
- 26.17 "Fixed bugs where many spells did not trigger Call for Help" and 26.19 "Fixed a bug where Runaan's Hurricane Volley could trigger minions' Call for Help" are under the **League Classic** headings — not CLASSIC SR. Treat Runaan's per-item CFH flag as an item-spec question.

## 4. Damage rules

### 4.1 Global damage-ratio table (CLASSIC `GameModeConstants {6cf687be}`, group `{2b322d93}`) [CLIENT H]

"Unit" = minion (and other non-hero AI units), "Hero" = champion, "Building" = turret/inhibitor/nexus.

| Source → Target | Multiplier |
|---|---|
| `dr_UnitToHero` | **0.55** |
| `dr_UnitToBuilding` | **0.60** |
| `dr_UnitToUnit` | 1.0 |
| `dr_HeroToUnit`, `dr_HeroToHero`, `dr_HeroToBuilding`, `dr_Building*` | 1.0 |

Applied multiplicatively to the minion's raw attack damage before resistances [INFERRED H — legacy server and wiki describe it as "minions deal X % damage"]. For comparison, other modes: Swiftplay `{a5dfc7b3}` 0.5/0.6; URF 0.5/0.5; ARSR/Doombots 0.5/0.5; Tutorial 0.5/0.5. **Wiki "60 % against champions" is wrong for CLASSIC; implement 0.55.**

Worked: caster U=1 (21 AD) on a champion with 30 armor: `21 × 0.55 × 100/130 = 8.885`.

### 4.2 Minion vs lane minion bonus ("Minion Slayer") [CLIENT H + RIOT 26.9 H]

On-hit **bonus physical damage = f × target's current HP** (before this hit), only when the target is a lane minion (`Minion_Lane` tag):

| Attacker | f | Source |
|---|---|---|
| Melee | 0.02 | item 1509 `MinionCurrentHealthDamage 0.019999999552965164` |
| Caster | 0.035 | item 1510 `0.03500000014901161` |
| Siege | 0.05 | item 1508 `0.05000000074505806` |
| Super | 0 | no such data value |

Mitigated by the target's armor like the base hit [INFERRED M]. Combined with `dr_UnitToUnit = 1.0`. Then Minion Pushing (§4.4) multiplies.

Worked: caster (21 AD) hits a melee minion at 465 HP, 0 armor: `21 + 0.035 × 465 = 37.275`.

### 4.3 Turret ↔ minion [CLIENT H]

Turret shot damage to minions is a fixed fraction of the minion's **maximum** HP (stringtable tooltips; values hard-coded in text, not data values):

| Minion | Turret shot | Shots to kill from full |
|---|---|---|
| Melee (Gusto) | 45 % max HP | 3 |
| Caster (Phreakish Gusto) | 70 % max HP | 2 |
| Siege (Anti-tower Socks) | 14 % outer / 11 % inner / 8 % inhibitor & nexus | 8 / 10 / 13 |
| Super (Super Mech Armor) | 7 % | 15 |

Minion damage to turrets: `dr_UnitToBuilding 0.6`; siege additionally `TurretDamageBonus 0.4` → `0.6 × 1.4 = 0.84` (RIOT 26.1 "82.5 % ⇒ 84 %", wiki 84 %) [CLIENT H]. Super vs non-turret structures 12.5 % per wiki [WIKI L]. Turret "Reinforced" 80 % DR when no enemy lane minions nearby, plating, etc.: turret spec.

### 4.4 Minion Pushing buff (`MinionPushingPower`) [CLIENT H constants, WIKI M formula]

CLASSIC group `{6ec75fd4}`: `mvm_PercentDamageToOtherMinionsBase 0.05`, `mvm_PercentDamageToOtherMinionsByTowerAdvantage 0.05`, `mvm_FlatDRFromOtherMinionsBase 1.0`, `mvm_LevelDiffCap 3.0`, `mvm_StartTime 210`, `mvm_UpdateInterval 1.0`, `mvm_MaxLifetime 600`, movement-speed terms unset, `mvm_MinionDisableTurretAdvantage` unset.

```
every 1.0 s from t >= 210:
  for each team T and lane l:
      lvl_adv = clamp(avg_level(T) - avg_level(enemy), 0, 3)    # avg of integer champion levels; decimal result
      tur_adv = max(0, enemy_lane_turrets_destroyed(l) - own_lane_turrets_destroyed(l))   # lane turrets only (outer/inner/inhib)
      bonus_dmg[T,l] = (0.05 + 0.05 * tur_adv) * lvl_adv          # extra fraction of damage vs enemy minions
      dr_div[T,l]    = 1 + 1.0 * tur_adv * lvl_adv                # damage taken from enemy minions is divided by this
minion-vs-minion damage = base_dmg * (1 + bonus_dmg[attacker]) / dr_div[target]
```
Applies to already-spawned minions and updates each interval. The wiki revision 3978998 confirms the divisor form (the "×100" variant was reverted). Interpretation of `mvm_FlatDRFromOtherMinionsBase` as the coefficient of `tur_adv·lvl_adv` is [INFERRED M]. 1v1 case with no turrets down: `dr_div = 1`, `bonus ≤ 0.15`.

`mvm_MaxLifetime 600` semantics unknown (perhaps the buff instance lifetime) — no behaviour assigned [CLIENT, L].

Simulator choice for a 1v1: `avg_level` of a one-champion team = that champion's level (scenario assumption; record in profile).

### 4.5 Super minion aura (`SuperMinionAura` / item 1512) [CLIENT H text, radius unknown]

"Grants nearby minions 35 Magic Resistance, and 35 armor." Radius not in data; default **800** [INFERRED L — historic super aura radius]. (`SuperMinionBuff` tooltip "70 % increased damage and 70 Armor/MR" is the pre-V8.23 aura text kept for the buff object; wiki V8.23 removed the damage aura — do not implement the 70 %.) Super minions are always visible to enemies [WIKI H].

### 4.6 Champion damage to minions; other modifiers

- `dr_HeroToUnit = 1.0`: no generic champion-vs-minion reduction in CLASSIC [CLIENT H]. Per-ability/item minion modifiers (wiki "Modified damage" list) belong to champion/item specs.
- No "minion damage reduction vs champions after X min" exists in CLASSIC data or notes [CLIENT+RIOT H].
- Omnivamp/lifesteal vs minions: `ov_OmnivampModifiedRatio 0.333` for AOE/Pet/DoT damage vs `Minion`/`Monster` [CLIENT H] — shared stat system.
- Baron: `BaronMinionDR 0.85` constant, Hand of Baron effects — objective spec.
- Lane-swap detector (item 1501 on turrets): triggers only with two non-jungler enemies / support item in lane; not reachable in 1v1 top. Out of scope.

### 4.7 Death grace (minion-killed-by-minion leeway) [CLIENT H values; WIKI M rule; threshold UNRESOLVED]

`GameplayConfig {a6506f8a}`: `mMinionDeathDelay 0.035`, `mMinionAutoLeeway 0.035`, `mMinionAAHelperLimit 190`.

Wiki: "When a minion is below 0.35 % of its maximum health and receives lethal damage from another minion, its health is set to 1. If it has not received damage from another source within the next 0.035 s, it dies at the end of the delay. Does not take place if the damage would exceed 190."

```
on minion-source damage d to lane minion M (after mitigation):
    if M.hp - d <= 0 and d <= 190 and M.hp_before_hit < LEEWAY * M.max_hp:   # LEEWAY default 0.0035 (wiki)
        M.hp = 1; M.grace_until = now + 0.035; M.grace_killer = attacker
    if now >= M.grace_until and grace active and no other-source damage: M dies, killer = grace_killer
    any non-minion damage during grace: normal resolution (≥1 damage kills; killer = that source)
```
The 0.035 delay matches the client. The threshold is ambiguous: wiki 0.35 % vs reading `mMinionAutoLeeway = 0.035` as a fraction (3.5 %). Default **wiki 0.35 %** with a flag [L] (§8 U-7). Note at a 30 Hz tick the window is ~1 tick.

## 5. Gold, XP, CS

### 5.1 Gold [CLIENT H + WIKI H]

- Gold goes **only to the champion who lands the killing blow**, in full, regardless of range or death status [WIKI Kill/Gold H]. A minion killed by a minion, turret, or neutral gives no gold to anyone. (`goldRadius 1250` and `ai_GoldRadius2 1000` exist in data; CLASSIC uses last-hit-only for minions. `goldRadius` is used by other modes/lane-swap redirect.) [INFERRED H]
- Amounts: melee 20, caster 14, siege `min(49+U, 90)`, super Order `min(49+U,90)` / Chaos 49 — §1.3.
- **U for gold is latched at spawn** (default) [INFERRED L; alternative read-at-death adds ≤1 g per siege]; wiki gold tables are per spawn-time-equivalent.
- No "denies" in modern League (no gold/XP penalty) [WIKI H, "Alpha Week 2: minions can no longer be denied"].
- Role-quest effects: top-lane quest gives "+1 quest point per minion kill (+2 in Top Lane)" and laners get "-25 % minion gold/XP outside own lane until level 3" (26.1) — role-quest spec; the minion-death event must expose `{killer, minion_type, lane_region}`.

### 5.2 XP [CLIENT H]

- Radius: **1500** (barracks `ExpRadius`; RIOT 25.S1.1 "1400→1500"; char-bin `experienceRadius 1400` is overridden) [CLIENT H value, M precedence].
- Recipients: all **enemy champions alive within 1500 of the death position**, plus the killer if it is an enemy champion regardless of range/death [WIKI H]. Granted whether the minion died to a champion, minion or turret [WIKI H].
- Split (`ExperienceModData.mPlayerMinionSplitXp`): each of n recipients gets `base × s[n]`, `s = [1.0, 0.65, 0.433, 0.325, 0.26, 0.217]` (n = 1..6).
- Base: melee 62, caster 31, siege 75, super 75; no time scaling.
- Comeback bonus (CLASSIC group `{0ffa6be3}`, `aiExp_*`): applies only once enemy minion level ≥ 5 (`bonusExpLaneLevelStart 5`; wiki says "minions have reached level 6" → [CONFLICT, default client: minion_level ≥ 5 is "start"; ambiguity whether `>=5` or `>5`, §8 U-8]). With `d = minion_level − recipient_level` (recipient decimal level per wiki):
```
if d <= 1: bonus = 0
elif d < 2: bonus = 0.20 + 0.20*(d-1)        # C1 = 0.2 up to UBound 2
elif d < 3: bonus = 0.80 + 0.40*(d-2)        # wiki jump at 2
else:       bonus = min(1.20 + 0.40*(d-3), 2.40)   # C2 = 0.4, LevelDeltaCap 6
xp_received = base * s[n] * (1 + bonus)
```
(wiki table: d=1.01→20 %, 2→80 %, 2.5→100 %, 3→120 %, 4→160 %, 6→240 %) [WIKI M; client constants consistent but discontinuity at d=2 is wiki-only].
- Minion level = floor(average integer level of the minion's team champions), latched at spawn [WIKI M].

### 5.3 CS
Each lane minion = 1 CS for the killer champion (`minionScoreValue` default 1.0) [CLIENT H].

## 6. Death, despawn, lifecycle

- Minions die at HP ≤ 0 (subject to §4.7). Corpse is non-interactive immediately; despawn/corpse timing has no gameplay effect [INFERRED H].
- No minion regen (except super, §1.5).
- No lifetime limit or "leave early" mechanic for lane minions in CLASSIC [RIOT/CLIENT H]. (`mvm_MaxLifetime` belongs to the pushing buff.)
- Inhibitor destroyed → super waves (§2.3); inhibitor respawn 300 s [WIKI H]. Lane-buff for minions after inhibitor kill ("small buff to allied minions in the lane") has no CLASSIC data (`HPInhibitor/DamageInhibitor` unset) → **none** [CLIENT H].
- Nexus destroyed → game over; end-of-game minion fade (`FadeMinionsDelaySecs 3.5`) is cosmetic.
- 25.05 bug fix: minions path to respawned Nexus turrets [WIKI].

## 7. State, events, and per-tick AI

### 7.1 State the simulator must carry

Global / per team-lane:
- `game_time_s`; `wave_index_next` (int); `unit_index_next` (int, within wave); `upgrade_count` (`U_barracks`, int, or derive from time).
- `ms_increase_count` (0..4) or derive from time (§2.6).
- Per team: `inhib_down[lane]` (bool), `inhib_respawn_at[lane]` (s); `lane_turrets_destroyed[lane]` (int, for pushing).
- Per team: `avg_champion_level` (float), refreshed every 1.0 s from 210 s for pushing (`mvm_UpdateInterval`); `pushing_bonus[lane]`, `pushing_div[lane]`.

Per minion (in addition to shared unit fields x, z, hp, max_hp, team, alive, radii):
- `minion_type` (melee/caster/siege/super), `lane`, `wave_index`, `unit_index`, `spawn_time_s`.
- `U_spawn` (int) → latched `max_hp`, `ad`, `armor`, `gold_bounty`; `minion_level` (int, for comeback XP).
- `is_first_wave` (bool), `ghost_until_s` (spawn + 28 top/bot, + 18 mid).
- `first_wave_engaged` (bool; set once it first targets an enemy minion).
- `sidelane_B` (float), derived bonus MS from `now − spawn_time_s`.
- Movement: `path_waypoint_index`, current waypoints.
- Targeting: `target` (int, −1), `target_priority` (int), `ai_timer_ms`, `time_since_attack_ms`, `ignore_until[unit]` (or a small ring of (unit, until)), `cfh_inbox` (best `(P, attacker, dist)` this tick).
- Attack state: `attack_phase` (idle/windup/recover), `windup_end_s`, `next_attack_ready_s`, pending missile(s) (caster/siege: target, launch_time, speed, damage snapshot).
- Death grace: `grace_until_s`, `grace_killer`.
- Super only: `hp_regen`; aura source flag.
- Per unit (all kinds), for priority classification: `last_attack_target`, `last_attack_target_kind`, `last_damage_dealt_time_s` (to evaluate "X attacking allied Y" with the 2 s memory, §3.1).

### 7.2 Events / hooks (emitted for other systems)

| Event | Payload | Consumers |
|---|---|---|
| `WaveSpawned` | team, lane, wave_index, composition | obs, logging |
| `MinionSpawned` | id, type, U, team, lane, t | — |
| `AttackLaunched` | attacker, target, kind | CFH (if attacker champion & target champion), turret aggro |
| `DamageDealt` | src, dst, amount, type, tags (basic/ability/cfh_flag) | CFH broadcast, turret ally-defense, minion-pushing, death grace |
| `CallForHelp` | attacker, victim, P | minion AI (same tick) |
| `MinionKilled` | minion, killer (unit or none), killer_kind, position, type, gold, xp_base, lane_region | gold (last hitter), XP split, CS, role quest, champion passives |
| `UpgradeTick` | U | — |
| `InhibitorStateChanged` | team, lane, down/up | wave composer |

### 7.3 Ordering within a tick (default)

1. Advance clock; apply barracks upgrade if an upgrade time is crossed (before spawning, §1.3).
2. Spawn due units (each wave unit at `t_wave + 0.8k`) at barracks; latch U/stats/gold/level.
3. Update pushing modifiers (if a 1.0-s boundary crossed) and MS increases.
4. Resolve due **damage** from previous-tick windups/missile arrivals (all units), in deterministic unit order; emit `DamageDealt`; apply death grace; collect deaths.
5. Build CFH inbox from this tick's `DamageDealt` (§3.3) and update "attacking X" memory.
6. Minion AI (7.4) for every alive minion: target validity → CFH switch → sweep → move/attack orders.
7. Movement (lane path / chase), collision (skip ghosted), terrain.
8. Start windups / launch missiles for units whose windup completed (missile lands at `launch + dist/speed`).
9. Process deaths: rewards (gold to killer champion only; XP split within 1500), CS, emit `MinionKilled`; free slots.

Deaths are resolved after all same-tick damage so simultaneous last hits are deterministic (killer = the source whose damage brought HP ≤ 0 first in step-4 order; recommend ordering champions before minions/turrets within a tick to break exact ties [INFERRED L]).

### 7.4 Per-minion AI pseudocode

```
def minion_ai(m, now, dt):
    m.ai_timer += dt
    if m.attack_phase == IDLE or m.target < 0: m.tsa = 0 if m.target < 0 or attacking(m) else m.tsa + dt
    else: m.tsa += 0 if attacking(m) else dt

    # 1. target validity
    valid = (m.target >= 0 and alive(t) and enemy(t) and targetable(t)
             and visible_to_team(t, m.team) and dist(m,t) < scan_range(m, t))
    just_lost = (m.target >= 0) and not valid
    if just_lost: m.target, m.target_priority = -1, INF

    # 2. Call for Help (immediate)
    cfh = m.cfh_inbox                       # best (P, attacker) from this tick, §3.3
    blocked = (is_turret(m.target) and not m.is_first_wave)
    if cfh and not blocked and cfh.P < m.target_priority and not in_windup(m):
        set_target(m, cfh.attacker, cfh.P); m.tsa = 0; m.ai_timer = 0

    # 3. regular sweep (only when needed)
    elif just_lost or m.ai_timer >= 0.250:
        m.ai_timer = 0
        if m.target >= 0 and m.tsa >= 4.0:
            m.ignore[m.target] = now + 0.5; m.target, m.target_priority = -1, INF
        if m.target < 0:                       # held valid targets are never re-prioritised here
            cands = enemies within scan_range, visible, targetable, not ignored
            if m.is_first_wave and not m.first_wave_engaged:
                cands = filter_first_wave(cands)    # §2.8: champions only within wakeUpRange; minions within firstAcquisitionRange, melee-spread
            best = argmin over cands of (P(c), dist(m,c))       # P from §3.1
            if best: set_target(m, best); m.tsa = 0
            elif no unit candidate and structure reachable: set_target(m, nearest_enemy_structure_on_path)

    # 4. act
    if m.target >= 0:
        if dist_edge(m, m.target) <= m.attack_range: stop; start_attack_if_ready(m)
        else: move_toward(m.target)            # chase
    else:
        follow_lane_path(m)                    # advance waypoint index when within 25 units

def scan_range(m, t):  # §3.2
    if m.is_first_wave and not m.first_wave_engaged:
        return m.wakeUpRange if is_champion(t) else m.firstAcquisitionRange
    return m.acquisitionRange

def start_attack_if_ready(m):
    if now >= m.next_attack_ready and m.attack_phase == IDLE:
        m.windup_end = now + windup(m); m.next_attack_ready = now + 1/AS(m)
        # at windup_end: melee -> apply damage; caster/siege -> spawn missile (650 / 1200 u/s)

def minion_hit_damage(m, tgt):
    raw = m.ad
    if is_lane_minion(tgt): raw += f_slayer[m.type] * tgt.hp
    mult = {HERO: 0.55, BUILDING: 0.60, UNIT: 1.0}[class(tgt)]
    if is_turret(tgt) and m.type == SIEGE: mult *= 1.4
    dmg = physical_mitigate(raw * mult, tgt.armor)
    if is_lane_minion(tgt) and enemy_minion(m, tgt):
        dmg = dmg * (1 + pushing_bonus[m.team, lane]) / pushing_div[tgt.team, lane]
    return dmg
```
(Turret→minion shots use §4.3 percent-max-HP instead and bypass armor.)

## 8. Unresolved / needs live measurement

All tests: Custom game or Practice Tool is NOT acceptable for wave timing (Practice Tool cadence differs, wiki edit 4065342). Use a CLASSIC custom game (Summoner's Rift, blind pick) on 26.19 with replay recording; read values from the replay/spectator unit stat panel or Live Client Data where available.

| ID | Question | Default implemented | Test scenario to resolve |
|---|---|---|---|
| U-1 | Are minion stats/gold latched at spawn, or do living minions gain upgrades when the barracks upgrades? | Latched at spawn | Let wave 4 (2:00, U=2) and wave 3 (1:30, U=1) coexist; at 2:05 select a wave-3 melee: 465 max HP ⇒ latched; 500 ⇒ live. Siege: kill a 1:30 cannon after 2:00 and read gold (50 vs 51). |
| U-2 | Melee armor growth offset (wiki A vs B vs C, §1.3) | A | Select a fresh melee at 20:30 (U=14): A = 3.06, B = 7.74, C = 3.83 (displayed rounded); at 24:30 (U=17): A 5.6, B 11.6, C 6.6. |
| U-3 | MS-increase times (10:30 vs 10:00 vs 11:00) and whether living minions get it | 10:30, global | Select a minion idle in lane at 9:55, 10:05, 10:35, 11:05; read MS (350/375). Check a minion spawned before and alive after the change. |
| U-4 | First-wave spread algorithm; meaning of `ai_TargetDistanceFactorPerAttacker 0.8`, `…PerNeightbor 0.6`, `ai_TargetMaxNumAttackers 5`, `ai_TargetRangeFactor 0.7` | Round-robin onto enemy melees | Record wave-1 collision with no champions in lane in 10 games; log each minion's target sequence (replay frame-step). Fit: equal 2-attackers-per-melee vs distance-weighted. |
| U-5 | Type preference inside "closest enemy minion" (legacy cannon>caster>melee) | Pure distance | Freeze a wave so an enemy caster is 100 u closer than an enemy cannon to a fresh allied melee with no other candidates; observe pick; then reverse distances. |
| U-6 | Reevaluation cadence (250 ms), give-up (4 s), ignore (0.5 s), and CFH aggro memory (2 s) | Legacy values, 2 s | Champion autos enemy champion once next to an enemy wave, then walks away; measure time until each minion drops the champion (frame-step, 30 fps). Repeat with an unreachable target (behind wall/flash) for the 4 s give-up. |
| U-7 | Death-grace threshold 0.35 % vs 3.5 % of max HP | 0.35 % | Practice-able in custom: let a melee be hit to ~3 % HP by minions only; in a replay, step frames at the killing minion hit — HP shown 1 for ~1 frame ⇒ 3.5 %. |
| U-8 | Comeback-XP start: minion level ≥ 5 (client `LaneLevelStart 5`) vs "reached level 6" (wiki) | client ≥ 5 interpretation flagged | Two-player custom: one player at level 2 when enemy team minions are level 5 and 6; record XP per melee (62 base ⇒ bonus % visible). |
| U-9 | Call-for-Help radii: wiki 500/1000 vs client acquisitionRange 750/700 | wiki 500/1000 for CFH, client ranges for scans | Ally champion stands 600/800/950 u from own minion while being autoed by enemy champion; does the minion switch? Repeat for scan with enemy champion approaching a lone minion; note acquisition distance. |
| U-10 | Super regen (`{2290fc9a}` 0.0015) and aura radius | wiki estimate / 800 | Inhibitor-down custom; read super HP regen and neighbouring minion armor at various distances. |
| U-11 | Do MS soft caps apply to minions with the sidelane buff? | yes | Measure wave-2 top minion travel time over the first 7 s (expected 451.8 u/s vs 461 u/s uncapped ⇒ 64 u difference). |
| U-12 | Melee count when a super replaces the siege post-14:00 (2 vs 3) | 2 (client rotation independent of siege) | Inhibitor-down custom after 14:00: count melee in an even-index wave. |
| U-13 | Wave intra-spacing 0.8 s (client) vs 0.79/0.792 (wiki/code) | 0.8 | Replay frame-step of a spawn: 7 units span 6 × spacing (4.8 s vs 4.75 s). |
| U-14 | Turret %max-HP shots: mitigated by armor or not; super 7 % (tooltip) | unmitigated, 7 % | Turret spec owner; shoot a super (100 + 35 aura armor) and read loss per shot. |

## 9. Diff vs current implementation

`lanerl_jax/sim/modern_minions.py` (HEAD):

| Loc | Issue | Fix |
|---|---|---|
| `:5-13` docstring, `:59-90` `MinionProfile` | Claims the per-upgrade curve is unsourced; it is fully specified by client `MinionUpgradeConfig` (§1.3). Profiles store U=1 endpoints only. | Replace with `stat(type, U)` per §1.3 using `minions.json` barracks data (HPUpgrade, DamageUpgrade(+Late), Max*, armor growth). |
| `:89` `SUPER_PROFILE` | Mixes U=1 HP (1600) with U=0 AD (180); U=1 AD is 185. | Derive from formula. |
| `:83-90` | No acquisition / first-acquisition / wake-up ranges, windup, missile speed. | Add per type: acq 750/700/750/600; first 1000/900; wake 450/635; windup 0.393/0.47/0.30/0.408 s; missile 650 (caster), 1200 (siege). |
| `:93` `WAVE_UNIT_GAP_S = .792` | Client `MinionSpawnIntervalSecs = 0.8`. | 0.8 (U-13). |
| `:162` melee pruning `(t>=840) & has_cannon` | Client melee rotation `[2,3]` (even i ⇒ 2) and constant 2 from 1500 are independent of super replacement; code gives 3 melee in super waves. | `melee = 3 if t<840 else (2 if i%2==0 else 3) if t<1500 else 2`. Test `test_modern_minions.py` composition `(28, 865, True) → [1,3,0,3]` must become `[1,2,0,3]`; `(54,1515,False,True)` → `[2,2,0,3]`. |
| `:147-163` | `enemy_inhibitor_down`/`all_…` booleans approximate `[1,1,2]` by count — OK; missing "no supers within two waves of inhibitor respawn". | Add respawn-time gate (§2.3). |
| `:230` XP fractions `13/30`, `13/60` | Client floats 0.433, 0.217. | Use client list. |
| `:241` `gold_bounty` siege/super `50 + U` | **Off by one**: client `49 + U` (50 at U=1). Test asserts 55 at U=5; correct is 54. Red supers: flat 49. No GoldMax 90 cap. | `min(49 + U, 90)`; team-aware super. |
| `:255-272` minion pushing | Formula matches; no 1.0-s update quantisation; averaging rule for 1v1 undocumented. | Quantise to 1 s; document team-size assumption. |
| `:302-311` `call_for_help_applies` | Uses caller-supplied acquisition range for the generic case; wiki 500. | Use 500 for generic CFH, 1000 for champion-on-champion (U-9). |
| — | No sidelane buff, MS time increases, first-wave rules, ghosting, death grace, damage ratios (0.55/0.6), Minion Slayer applied in combat, comeback XP, XP radius. Helpers exist for slayer (`:244`) but nothing calls them. | Implement per §2.6–2.8, §4, §5. |

`lanerl_jax/sim/modern_world.py` (uncommitted; read only):

| Loc | Issue |
|---|---|
| `:111-121` `spawn` | Spawns with `params['max_hp'][model]` from the legacy profile table (4.20-derived rows in `profiles.py`) — **no U scaling, legacy HP/AD/gold/XP**, legacy acquisition fallback 475 (`profiles.py:149`). Should latch §1.3 stats at spawn. |
| `:113-115` | Uses `spawn_event` with 0.792 spacing and no inhibitor state. |
| `:126-129` | `birth_time` recorded; no `U_spawn`, `minion_level`, first-wave flag, ghost timer. |

Shared path used by modern mode (`step.py`, `targeting.py`, `minion_ai.py`):

| Loc | Issue |
|---|---|
| `targeting.py:65,140` + `step.py:1347-1359` (CFH enabled by default, `step.py:241`) | `CHAMPION_ATTACKING_MINION` (=5) is still produced by `help_priority_for`, so champion hits on minions still pull aggro — **violates 26.10** in modern mode. Gate it off when `state.modern`. |
| `targeting.py:86-106` `base_priority` | Legacy type ordering cannon(7)<caster(8)<melee(9); modern spec is distance-only (U-5). |
| `targeting.py:124-192` `call_for_help_map` | Uses victim acquisitionRange for both distances; modern: 500 / 1000 (§3.3). |
| `step.py` damage pipeline (`:1153` region) | No minion→champion 0.55 / minion→minion slayer / pushing multipliers in the shared minion attack path; modern towers docstring (`modern_towers.py:164`) expects the caller to scale minion→turret by 0.60/0.84. Verify the caller exists. |
| `modern_towers.py:210` `minion_shot_damage` | Super fraction **0.05**; client tooltip (item 1511) says **7 %**. Also applies armor (with 30 % pen) to a percent-max-HP shot — tooltip/wiki describe a fixed % (U-14). |
| `minion_ai.py:96-103` | 250 ms / 4 s / 0.5 s / 25 u margin — legacy values, acceptable defaults (U-6). |

`lanerl_jax/data/modern/26.19/minions.json`: correct copy of the Order barracks and Chaos unit records; lacks Chaos barracks (`GoldUpgrade` missing for Chaos super), lacks class defaults (acquisitionRange 750, experienceRadius 0), lacks CLASSIC `GameModeConstants` (dr_*, mvm_*, aiExp_*), `ExperienceModData`, `GameplayConfig` death-grace fields, and rule-item data values (1508–1510). Its note on armor ("accumulated per-upgrade growth") is consistent with §1.3 but the offset is unresolved.

`docs/MODERN_PATCH_DELTA.md` §5 corrections: §5.2 siege AD lacks +4 late growth; §5.8 "60 % to champions" → 55 %; §5.9 death grace delay 0.066 → 0.035 s; §5.4 "XP radius unverified" → 1500 (barracks); §5.9 acquisition table "melee unset" → class default 750.

## 10. Test fixtures

All inputs → expected outputs (floats to 1e-4 unless noted).

**Upgrade index** `U_barracks(t)`: 29.9→0; 30→1; 119.9→1; 120→2; 450→5; 480→6; 570→7; 840→10; 1500→17.

**Stats at U** (§1.4): melee U=1 (465, 11, 0); U=6 (640, 14, 0); U=7 (675, 17, 0.085); U=10 (780, 26, 0.85); U=32 HP 1550, U=33 HP 1550. Caster U=5 AD 27.0, U=6 31.0, U=30 125.0, U=29 123.0. Siege U=5 AD 43.5, U=6 47.5, U=26 AD 126.0 (36+1.5·26+2.5·21=127.5→cap 126). Super U=1 (1600, 185).

**Gold**: siege U=1 50, U=5 54, U=10 59, U=41 90, U=50 90. Super Order U=5 54, Chaos U=5 49.

**Wave times**: i=0 30; 26 810; 27 840; 28 865; 52 1465; 53 1490; 54 1515; 65 1790; 66 1810; 67 1830.

**Composition** (supers, melee, siege, casters), no inhib down: i=0 (0,3,0,3); i=2 (0,3,1,3); i=26 (0,3,1,3); i=27 (0,3,0,3); i=28 (0,2,1,3); i=29 (0,3,0,3); i=53 (0,3,0,3); i=54 (0,2,1,3); i=65 (0,2,1,3); i=66 (0,2,1,2). Top inhib down: i=28 (1,2,0,3); i=29 (1,3,0,3); all inhibs down, i=54 (2,2,0,3).

**Unit spawn times**: wave i=2 units k=0..6 at 90 + 0.8k = 90.0, 90.8, …, 94.8; types [M,M,M,S,C,C,C].

**Sidelane** bonus for wave n (1-based), τ since spawn: n=2 τ=0 → 111; τ=7 → 96; τ=14 → 81; τ=21 → 66; τ=25 → 0. n=10 τ=0 → 75. n=26 τ=0 → 3, τ=7 → 0. n=27 → 0. n=1 → 0. Mid lane → 0. Capped MS wave 2 τ=0: 451.8.

**Base MS** (default A): t=629 → 350; 630 → 375; 930 → 400; 1530 → 450; 2000 → 450.

**Windup/period**: melee 0.393/0.8; caster 0.470/1.4993; siege 0.300/1.0; super 0.4083/1.1765.

**Damage** (0 armor unless stated): caster U=1 → champion with 30 armor: 21×0.55×100/130 = 8.8846. Melee U=1 → champion 0 armor: 6.05. Siege U=1 → outer turret 60 armor (ignore turret DR/plating): 37.5×0.6×1.4×100/160 = 19.6875. Caster U=1 → melee minion 465 HP: 21 + 16.275 = 37.275; melee U=1 → caster at 284 HP: 11 + 5.68 = 16.68; siege U=1 → melee at 400 HP: 37.5 + 20 = 57.5. With pushing lvl_adv=2, tur_adv=0: caster→melee 37.275×1.10 = 41.0025 (attacker side advantaged). If instead the *target's* side is advantaged with tur_adv=1, lvl_adv=2 (its divisor = 1 + 1·2 = 3) and the attacker side has no advantage: 37.275/3 = 12.425. Each side's modifier only affects its own minions (bonus when dealing, divisor when receiving).

**Turret shots to kill** (full HP): melee 3, caster 2, siege outer 8, inner 10, inhib 13, super 15.

**XP**: melee killed, 1 enemy champion within 1500 → 62; 2 → 40.3 each; caster with 2 → 20.15 each; siege with 3 → 32.475 each (75×0.433). Champion at 1501 u and not the killer → 0. Comeback (minion level 7, recipient level 5.0, d=2.0, n=1): 62×1.8 = 111.6; d=1.5: 62×1.30 = 80.6; d=6.5: 62×3.4 = 210.8.

**Pushing activation**: t=209.9 → bonus 0; t=210 → active (lvl_adv=1, tur_adv=0 → +5 %); lvl_adv=4.2 → clamped 3 → +15 %.

**Death grace** (default 0.35 %): melee max 465 at 1.5 HP hit by an enemy minion for 11 → HP 1, dies 0.035 s later unless a champion hit lands first (champion gets the kill). Same at 2.0 HP (0.43 %) → dies immediately (threshold 1.6275). Hit of 200 at 1.5 HP → dies immediately (> 190).

**Priority / hysteresis**: minion holding an enemy minion (P5); enemy champion autos an ally champion 800 u away (within 1000) and within 750 of the listener → switch to champion (P1). Enemy champion autos an allied **minion** → no switch (26.10). Minion targeting a turret receives the same P1 CFH → no switch (not first wave). Two P5 candidates at 300 u and 200 u while holding the 300-u one → keep.

