# Modern Summoner's Rift structures: turrets, inhibitors, Nexus (patch 26.19)

**Scope.** All Summoner's Rift structures in normal PC 5v5 Classic at patch **26.19**
(client build **16.19.8230722**): outer (T1), inner (T2), inhibitor (T3) and Nexus turrets,
inhibitors, the Nexus, and a short note on the fountain turret (Nexus Obelisk). Swiftplay,
ARAM, ARAM: Mayhem, Classic (the 26.16 retro mode), Arena and URF variants are **out of
scope**. Where the client data carries a mode override, this doc says so and uses only the
default (SR Classic) value. Fountain regen belongs to the economy spec, and minion stats to
the minion spec.

**Retrieved:** 2026-10-01. **Research only.** No code or data was changed.

**Confidence tags** (every rule carries one):
- `[CLIENT]`: read directly from 16.19.8230722 client data (bins, item data values, string table).
- `[RIOT]`: Riot patch notes 26.1–26.19.
- `[WIKI]`: League wiki at the oldid cited.
- `[INFERRED]`: derived by reasoning from the above. The basis is stated.
- Each tag also carries confidence H / M / L.

## 0. Sources

### 0.1 Client data (CommunityDragon raw 16.19 = build 16.19.8230722, and Riot-manifest extraction)

Cache directory: `/mnt/nfs/shared/modern-world-map-research/cdragon-16.19/` (SHA256SUMS updated).

| File | Source URL | sha256 | Used for |
|---|---|---|---|
| `turret.bin.json` | https://raw.communitydragon.org/16.19/game/data/characters/turret/turret.bin.json | `b1338a74d7d49bf838211378406c1634e6e58e4444e512f1b4e373a55c682a35` | Per-tier CharacterRecords (see 0.3), basic-attack spell (missile speed, cast frame) |
| `items.cdtb.bin.json` | (pre-cached) | `6880f35d5a9d82f688192f764e280e6d4bc9c845112b001feb811e3f2ab62726` | **Turret pseudo-items 1500–1524**: all mDataValues (Warming Up, Reinforced Armor, plating/Bulwark/decay/AD growth, **Overgrowth formula**), minion-vs-turret items 1508–1511 |
| `lol.stringtable.json` (en_us) | https://raw.communitydragon.org/16.19/game/en_us/data/menu/en_us/lol.stringtable.json | `8c051cb2a24b31f3fa9af95d39085b832b0da2b8cf51093f98c4ab0620ecb8e8` | Tooltips that bind the data-value names to semantics |
| `inhibitor.bin.json` | …/characters/inhibitor/inhibitor.bin.json | `80b58f56b530ba6a5b979b3616ac40b3762d498f13fd56c8a7daa1d4a31cdd84` | Inhibitor HP/regen/radius |
| `nexus.bin.json` | …/characters/nexus/nexus.bin.json | `9aa797ed13d8108c1b5b5531961dbb7c87b8ecc2ca9861a021042cf33bf1780f` | Nexus HP/regen/radius |
| `sruap_turret_order5.bin.json` | …/characters/sruap_turret_order5/sruap_turret_order5.bin.json | `94f04c3fc88f257445c68b13609f849395ae3785b39c2f7e937134c0b166d971` | Fountain turret |
| `turretovergrowth.bin.json` | …/characters/turretovergrowth/turretovergrowth.bin.json | `77777c5f02d3b65c9823512bcca8ffdef5ed648d463e9a7236cdfaca514267e7` | Confirms it is a TemplateFighter placeholder (no gameplay numbers) |
| `turretrubble.bin.json`, `sruap_building.bin.json` | …/characters/… | `3a9d978a…10ad`, `268cf7c8…2375` | Placeholders; nothing gameplay-relevant |
| `shared.cdtb.bin.json` | (pre-cached) | `34f68553ab38cfe344936473fcb48d99574d5fd51e4feb42994fa0c4fe50769e` | Turret buff SpellObjects (metadata only), missile specs |
| `/mnt/nfs/shared/modern-world-map-research/classic-constants.json` | Map11 classic GameModeConstants | (see that dir) | `dr_UnitToBuilding 0.60`, `events_TimerForBuildingKillCredit 30`, `SplitLocalGold`, `bar_MaxHP 4000` |
| `/mnt/nfs/shared/modern-world-map-research/map11-decoded.json` | Map11 bin | map11.bin sha `45d16148616eb3da612e31f422afb46d756d5fc615f8387b7429343bda5da88a` | BarracksConfig `SpawnCountPerInhibitorDown [1,1,2]`. Which record names SR uses (`Characters/Turret/CharacterRecords/SR_Outer`, …) |
| `/mnt/nfs/shared/modern-world-map-research/overgrowth-scripts-16.19.8230722/*.luabin64` | Scripts.wad.client | see `towers.json` client_script_evidence | Buff names only (`OvergrowthCooldown`, `OvergrowthReady`, `OvergrowthMeleeBuffer`). No formulas |

### 0.2 Riot patch notes (all 19 audited, plus their embedded hotfix sections)

- 26.1: https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-1-notes/ (season rework: plating, Overgrowth, HP/resists/AD, regen thresholds, Nexus-turret respawn, minion→turret 60%, cannon 84%, first-turret 300g, melee +20%)
- 26.2 and 26.3: `patch-26-N-notes` URL form. 26.4–26.19: `league-of-legends-patch-26-N-notes` URL form.
- Structure-relevant hits:
  - **26.3**: Bulwark 20–100 ⇒ 30–50 per plate.
  - **26.6**: Rift Herald Charge, Spin Attack and Auto Attack now proc Overgrowth.
  - **26.7**: Practice-Tool reset bug fixed (turret DR).
  - **26.10**: Herald-through-Bard-R Overgrowth bug fixed. Inhibitor timer visual fix.
  - **26.12**: Herald magic-vs-true damage to turrets fixed.
- 26.5's "Nexus 5500⇒3000, Nexus Tower 1800⇒3000" is **ARAM: Mayhem only**. 26.16's "Classic Turrets" section is the **League Classic retro mode only**. Neither applies to SR.
- Audit result: **no SR turret, plating, Overgrowth, inhibitor or Nexus number changed between 26.4 and 26.19.**

### 0.3 Wiki (permalinks)

- Turret: https://wiki.leagueoflegends.com/en-us/Turret?oldid=4070072 (2026-09-29)
- Inhibitor: https://wiki.leagueoflegends.com/en-us/Inhibitor?oldid=4070357
- Nexus: https://wiki.leagueoflegends.com/en-us/Nexus?oldid=4070358
- Nexus Obelisk: https://wiki.leagueoflegends.com/en-us/Nexus_Obelisk?oldid=4015143
- Module:ItemData/data: https://wiki.leagueoflegends.com/en-us/Module:ItemData/data?oldid=4069878 (turret items, Gusto/Socks/Super Mech Armor "pre-mitigation" wording)
- V26.01: https://wiki.leagueoflegends.com/en-us/V26.01?oldid=4052188
- V26.03: https://wiki.leagueoflegends.com/en-us/V26.03?oldid=4032790
- V26.06: https://wiki.leagueoflegends.com/en-us/V26.06?oldid=4058113
- V26.10: https://wiki.leagueoflegends.com/en-us/V26.10?oldid=4047645
- V26.02–V26.19 other pages were also checked (revids 4052127 … 4066072). None has an SR structure change.
- Minion pages:
  - Minion: https://wiki.leagueoflegends.com/en-us/Minion?oldid=4068797
  - Siege minion: https://wiki.leagueoflegends.com/en-us/Siege_minion?oldid=4013294
  - Super minion: https://wiki.leagueoflegends.com/en-us/Super_minion?oldid=4068820

### 0.4 Which client record is the SR turret? (resolved)

Map11's unit table names the records `Characters/Turret/CharacterRecords/SR_Outer`, `SR_Inner`,
`SR_Inhibitor` and `SR_NexusTurret`. The hashed entries in `turret.bin` were resolved by FNV-1a of the
lower-cased path:

| Hash | Record path | HP | AD | Armor/MR | Notes |
|---|---|---|---|---|---|
| `{9d71ab2e}` | `SR_Outer` | 9000 | 182 | 60/60 | global gold 50 |
| `{e1f854f5}` | `SR_Inner` | 5000 (inherits root value) | 187 | 60/60 | global 25 |
| `{a1f8429b}` | `SR_Inhibitor` | 4750 | 187 | 60/60 | global 25 |
| `{edd2bd52}` | `SR_Nexus` | 3500 | 165 | 60/60 | global 50 |
| `{075d43f4}`,`{3e2fe8a7}`,`{640bb061}`,`{f3b4a818}` | `Swiftplay_Outer/Inner/Inhibitor/Nexus` | 9000/5000/4750/3500 | 182/187/187/165 | 60 or **50** | Swiftplay: **ignore** |
| `{325fd17c}` | `SR_NexusTurret` | 3000 | 170 | 75/75 | AS 1.35, range 550, 15 HP/s. Referenced only by a mode unit list next to `SRUAP_MageCrystal`: **not normal SR** |
| Root | `Characters/Turret/CharacterRecords/Root` | 5000 | 152 | 15/15 | pre-2026 legacy root values (local gold 250). **Not used by SR** |

The SR_* values match the wiki infobox and the 26.1 notes exactly. This also closes
MODERN_PATCH_DELTA §6.1's "which tier is which model" question for the modern table.

---

## 1. Per-tier stats

### 1.1 Turrets

| Field | Outer (T1) | Inner (T2) | Inhibitor (T3) | Nexus | Tag |
|---|---|---|---|---|---|
| Max HP | 9000 | 5000 | 4750 | 3500 | `[CLIENT H]` (SR_* records) = `[RIOT]` = `[WIKI]` |
| Armor / MR (base) | 60 / 60 | 60 / 60 | 60 / 60 | 60 / 60 | `[CLIENT H]` |
| Base AD | 182 | 187 | 187 | 165 | `[CLIENT H]` |
| AD growth | +12 at 0:30 and every 60 s after; 14 steps; max +168 (350) reached at 13:30 | +16 at 3:00 and every 60 s; 15 steps; max +240 (427) at 17:00 | same as inner (427) | same rule, 165→405 | `[CLIENT H]` item 1515: `OuterBonusADPerMinute 12`, `OuterTotalBonusAD 168`, `OtherBonusADPerMinute 16`. Schedule `[WIKI H]`. Riot prose says "168 at 14 minutes" (see §11 C2) |
| Attack speed | 0.833 (period 1/0.833 = **1.2005 s**) | 0.833 | 0.833 | 0.833 | `[CLIENT H]` |
| Attack range | 750 | 750 | 750 | 750 | `[CLIENT H]` `attackRangeModifiable`. Wiki marks it "edge range" (§1.3) |
| Acquisition range | = attack range (no `acquisitionRange` field on SR records) | | | | `[INFERRED M]` |
| Gameplay (collision) radius | 88.4 | 88.4 | 88.4 | 88.4 | `[CLIENT H]` `overrideGameplayCollisionRadius` |
| Pathfinding radius | 125 | 125 | 125 | 125 | `[CLIENT H]`. The turret also sits on impassable terrain that persists after death `[WIKI H]` |
| Selection radius | 130 | | | | `[CLIENT H]` (UI only) |
| Missile speed | 1200 (homing, unblockable, undodgeable) | | | | `[CLIENT H]` spell `TurretBasicAttack` `missileSpeed 1200`. `[WIKI H]` "Dodge Piercing" |
| Windup | ≈ **0.167 s** | | | | `[INFERRED M]`, see §1.2 |
| Sight radius | 1350 | | | | `[WIKI H]` |
| True sight radius | 1100 (blocked by terrain LOS) | | | | `[CLIENT H]` item 1503 `TrueSightRadius 1100` (ARAM 900). `[RIOT]` 26.1 1095⇒1100 |
| Armor penetration (own attacks) | 30 % | 30 % | 30 % | 30 % | `[CLIENT H]` item 1500 `ArmorPenetration 0.3` |
| HP regen | 0 | 0 | **3 HP/s**, segment-capped 30/75/100 % | **6 HP/s**, segment-capped 40/70/100 % | Rates `[WIKI H]`/`[RIOT H]` (26.1 "6 HP/sec"). Thresholds `[CLIENT H]` string table (1505/1506 tooltips). Record `baseStaticHPRegen = 0`, so regen is script-granted |
| Plates | 5 | 5 | 5 | none | `[CLIENT H]` 1515 `Tier1SRPalisadeCount 5`, `Tier2PalisadeCount 5`. Nexus record lacks item 1515 |
| Plate gold (local, split) | 120, decays −10/min from 11:00 to a floor of 80 | 120 (no decay) | 120 (no decay) | — | `[CLIENT H]` 1515 `Tier1PalisadeGold 120`, `Tier2PalisadeValue 120`, `OuterGoldDecay −10` |
| Destroy gold, global (each ally) | 50 | 25 | 25 | 50 | `[CLIENT H]` `globalGoldGivenOnDeath` |
| Destroy gold, local | 0 | 0 | 0 | 0 | `[CLIENT H]` (field absent on SR_* records). `[RIOT]` 26.1 "Local gold 250⇒0" |
| First-turret bonus | +300 local, shared | | | | `[RIOT H]` 26.1 "Nearby players are now rewarded a share of 300g". `[WIKI H]` |
| Destroy XP | **0** | 0 | 0 | 0 | `[CLIENT H]` `expGivenOnDeath 0`, no `globalExpGivenOnDeath` (4.20 Content had 100 global XP) |
| Overgrowth | yes | yes | yes | **no** | `[CLIENT H]` item 1524 present on SR_Outer/Inner/Inhibitor, absent on SR_Nexus |
| Bulwark | yes | yes | yes | no | `[CLIENT H]` (rides on item 1515) |
| Resistance decay 11:00–15:00 | yes (outer only) | no | no | no | `[CLIENT H]` `OuterDecay*` |
| Respawn | never | never | never | **180 s after death, at 40 % max HP** | `[CLIENT H]` string 1505 "reanimate at 40% health 3 minutes after being destroyed" |
| Pseudo-inventory | 1500, 1502, 1503, 1517, 1524, 1515, 1507 | 1500, 1502, 1503, 1516, 1524, 1515, 1507 | 1500, **1506**, 1503, 1518, 1524, 1515, 1507 | 1500, **1505**, 1503, 1519, 1507 | `[CLIENT H]` `mClientSideItemInventory` |

Item key:
- 1500: Penetrating Bullets / Warming Up
- 1502: Reinforced Armor
- 1503: Warden's Eye
- 1505: Nexus-turret Reinforced Armor + regen + respawn
- 1506: inhibitor-turret Reinforced Armor + regen
- 1507: Overcharged (late-game, §7.2)
- 1515: Turret Plating
- 1516–1519: reward tooltip items
- 1524: Crystalline Overgrowth

Notably **absent from SR records**:
- 1501 "Fortification" / Lane Swap Detector. It is still on the legacy Root record. Removed from SR in 26.1 `[WIKI H]`.
- 1521/1522: ARAM-only Fortification / Tower Power-Up.

### 1.2 Attack timing

- `period = 1 / 0.833 = 1.20048 s`. `[CLIENT H]` The current code uses 1.2. The 0.04 % difference is negligible but should be named.
- `windup_fraction = gcd_AttackDelayCastPercent (0.30) + mAttackDelayCastOffsetPercent (−0.161) = 0.139`. So `windup = 0.139 × 1.20048 = 0.1669 s`. `[INFERRED M]`
  - Uses the standard League formula with `gcd_AttackDelayCastPercent` from classic-constants.
  - Cross-check: the spell `castFrame 4.95` at 30 fps = 0.165 s.
  - The wiki estimates "1 to 2 game ticks" for obelisk-style lasers.
  - Implement the windup as **0.167 s, rounded up to the next sim tick**.
- **Missile flight time** = `max(0, d_launch_to_target − ...)/1200`.
  - The missile starts at bone `joint2`, homes, and resolves on movement complete.
  - Implement: `flight = center_distance / 1200`, recomputed as the target moves (homing). `[INFERRED M]`
- **The shot is lost if the turret dies mid-flight.** `[WIKI H]`
- Once the windup completes, the shot cannot be dodged, parried or intercepted. "Pop homing projectile" untargetability still destroys it. `[WIKI H]`

### 1.3 Range semantics

The wiki marks 750 as "edge range" (`{{tip|er}}`). Default implementation `[INFERRED M]`:

`in_attack_range(turret, u) := dist(center_t, center_u) <= 750 + 88.4 + gameplay_radius(u)`

This is the same edge-to-edge convention as champion attacks. The turret target lock also
releases when this predicate becomes false (§3). Exact offset needs measurement (§12 U5).
25.S1.2 shipped "Corrected attack range offset for various turrets" `[WIKI]`, which shows the
offset matters to Riot but does not give it.

### 1.4 Inhibitor and Nexus (non-attacking structures)

| Field | Inhibitor | Nexus | Tag |
|---|---|---|---|
| Max HP | 4000 | 5500 | `[CLIENT H]` (`inhibitor.bin`, `nexus.bin`; classic `bar_MaxHP 4000`) = `[WIKI H]` |
| HP regen | 15 HP/s, **not** segment-capped | 20 HP/s | `[CLIENT H]` `baseStaticHPRegen` = `[WIKI H]` |
| Armor / MR | 20 / 0 | 20 / 0 | `[WIKI M]`. Client root records omit armor (default 0). Wiki: "undocumented, 2025" for Nexus; V7.15 for inhibitor. See §11 C5 |
| Armor penetration vs it | none (penetration ignored vs inhibitor) | (unstated, assume normal) | `[WIKI M]` |
| Pathfinding radius | 213.75 | 304 | `[CLIENT H]` |
| Respawn | 300 s after destruction (announcer "respawn soon" at 285 s) | — (destruction ends the game) | `[WIKI H]` (unchanged since V4.20) |
| Bounty | 50 gold to the last-hitting champion only | — | `[WIKI M]` |
| Effect of death | enemy waves in that lane spawn super minions; super count per wave by number of that team's inhibitors down = `[1, 1, 2]` (1 down → 1, 2 down → 1, 3 down → 2 per lane); super replaces the cannon slot when applicable | game over | `[CLIENT H]` BarracksConfig `SpawnCountPerInhibitorDown [1,1,2]` + `[WIKI H]` (minion spec owns the wave rules) |
| Vulnerability | only after its lane's inhibitor turret is dead | only after ≥1 inhibitor is down **and** both Nexus turrets are dead | `[WIKI H]` |
| Damage multipliers | Reinforced Armor and Bulwark do **not** apply (no turret items). Minions 60 %. Super minions 12.5 % vs non-turret structures | same | `[CLIENT H]` dr_UnitToBuilding + `[WIKI M]` super 12.5 % |

### 1.5 Fountain turret (Nexus Obelisk): brief

`SRUAP_Turret_Order5` / `Chaos5` `[CLIENT H]`:
- 9999 HP, untargetable/indestructible `[WIKI]`.
- Attack range 1250, acquisition 1250, attack speed 2.5.
- The record has AD 999. The wiki says 1000 **true (internal raw) damage** per tick at 2 ticks/s, ignoring shields/invulnerability and not triggering combat. `[WIKI H]`
- Targets the closest enemy unit in range; no champion priority.
- Never leave a sim unit in the enemy fountain. Model as instant-lethal area. Fountain regen: economy spec.

---

## 2. Vulnerability / ordering

| Structure | Targetable and damageable when | Tag |
|---|---|---|
| Outer turret | always, from game start. Plating exists at game start (26.1 removed the spawn-in) | `[WIKI H]` |
| Inner turret | its lane's outer turret is dead | `[WIKI H]` |
| Inhibitor turret | its lane's inner turret is dead | `[WIKI H]` |
| Inhibitor | its lane's inhibitor turret is dead (inhibitor turrets never respawn, so this is permanent) | `[WIKI H]` |
| Nexus turrets (2) | **≥1 of the team's inhibitors is currently dead** | `[WIKI H]` |
| Nexus | ≥1 inhibitor currently dead **and** both Nexus turrets currently dead | `[WIKI H]` |

Edge cases:
- If all dead inhibitors respawn, Nexus turrets and the Nexus become invulnerable/untargetable again. A respawned Nexus turret is also invulnerable until an inhibitor is down. `[WIKI M]` (25.18 bug-fix wording implies live Nexus turrets toggle targetability with inhibitor state)
- A dead Nexus turret respawns on its 180 s timer regardless of inhibitor state. `[INFERRED M]`
- Untargetable structures take no damage. Their Overgrowth startup clock does not run (§6.2). `[RIOT H]` "Targetable lane turrets are subject to…"

## 3. Targeting AI

### 3.1 Priority list (lowest number = attacked first) `[WIKI H]`

| Prio | Units |
|---|---|
| 0 | most champion pets (Tibbers, Voidlings, …) |
| 1 | siege (cannon) minions, super minions, Yorick Dark Procession |
| 2 | Yorick Mist Walkers |
| 3 | melee minions |
| 4 | caster minions, Wukong clone |
| 5 | Maiden of the Mist, Elise spiderlings, Naafiri packmates |
| 6 | champions |

Ties within a priority class go to the **closest** unit (center distance). Ties at equal
distance are unspecified. Use the lowest slot index as a deterministic convention.
`[INFERRED L]` for the tie-break.

Valid candidates for a turret `T`:
- enemy, alive, targetable, visible to T's team;
- not in stasis or untargetable (untargetability can drop aggro);
- `in_attack_range(T, u)` (§1.3).

Neutral monsters are ignored. Wards, traps and the Rift Herald (as unit or mercenary) are
handled by other specs. Lane-1v1 sim: only champions and lane minions.

### 3.2 Lock and re-acquisition `[WIKI H]` (timing details `[INFERRED M]`)

- **Sticky lock:** T keeps attacking its current target until it dies, leaves attack range, or becomes untargetable.
  - There is **no** periodic re-scan to a higher-priority unit while the lock holds.
  - Exception: the champion-protection override (§3.3).
- **On lock loss**, T selects a new target by §3.1 at the **next attack decision**.
  - A shot already past windup still fires and resolves (homing).
  - Leaving range during the windup: the default is to **cancel** the windup and re-acquire. `[INFERRED L]` §12 U3.
- **Attack cadence:** the attack timer continues across target switches. A new target does not reset the 1.2005 s cooldown. `[INFERRED M]`
- **First acquisition:** "the first enemy unit that comes into range". When several enter on the same tick, apply §3.1.

### 3.3 Champion-protection ("call for help") override `[WIKI H]`

Trigger: an **enemy champion** deals damage to an **allied champion** (ally of T), and the
victim is **within 1400 units of T** (center distance). Details:
- **Damage source:** any source owned by the champion, including pets. `[WIKI H]`
- **What counts as damage:** the damage *attempt* counts even if:
  - it is spell-shielded, invulnerable, dodged, blocked or blinded; or
  - it deals 0 damage.
- **What does not count:** non-damaging effects (e.g. Karthus W).
- **Effect:** T **switches** to the aggressor, provided the aggressor is a valid candidate (in T's range, targetable, visible). This preempts the sticky lock.
- If the aggressor is out of range at trigger time, the default is no retarget. Whether a pending "remembered" aggro exists for some window is **unknown** (§12 U1). Default: no memory; evaluate per damage event.
- **Several aggressors on the same tick:** pick the closest aggressor. `[INFERRED L]`
- **After switching**, the normal sticky lock applies to the aggressor. T stays on that champion until it dies, leaves range or becomes untargetable, even if it stops attacking. `[WIKI M]`
- **25.06 hotfix:** minions are never treated as champions for this rule. `[WIKI H]`
- **Champion attacking an allied *minion* or the turret itself** does **not** trigger the switch. `[WIKI H]`, by omission. The prior 2014-era behaviour is irrelevant here.

### 3.4 Pets / clones

- Pets: priority 0 (attacked before minions).
- Clones (Wukong, Shaco, LeBlanc, Neeko): clone handling is a champion-kit concern. Wukong's clone is listed at priority 4. Others: treat as their listed class, else as a champion. `[WIKI M]`

## 4. Damage

### 4.1 Turret shot vs champion `[CLIENT H]` (item 1500) + `[WIKI H]`

```
def turret_shot_vs_champion(T, now, target_armor):
    stacks = T.warm_stacks if now < T.warm_until else 0      # 0..3
    mult   = 1 + 0.5 * stacks                                # HeatingUpMultiplier 0.5
    eff_armor = target_armor * (1 - 0.30) if target_armor > 0 else target_armor   # ArmorPenetration 0.3
    raw    = AD(T.tier, now) * mult                          # physical
    dmg    = raw * armor_mult(eff_armor)                     # 100/(100+a), or 2-100/(100-a) for a<0
    # on impact (not at launch): stacks advance and the 5 s window refreshes
    T.warm_stacks = min(stacks + 1, 3)                       # HeatingUpMaxStacks 3, MaxPerc 1.5
    T.warm_until  = now + 5.0                                # HeatingUpDuration 5
    return dmg
```

- Multipliers by consecutive champion hit: **1.0, 1.5, 2.0, 2.5, 2.5, …** (max +150 %).
  - The tooltip's `MaxHitDamage = 2.5 × AD`. `MaxDamagePercentTooltip 2.5` confirms the cap. `[CLIENT H]`
- **Reset:** 5 s after the last shot *hits a champion*. Switching between champions does **not** reset. `[WIKI H]`
- Shots on minions neither add stacks nor refresh the window. `[WIKI H]` ("each time they strike a champion")
  - Whether a minion shot inside the 5 s window still keeps the stacks alive is not stated. Default: the window is not refreshed. `[INFERRED M]`
- **Stacks advance on impact** ("strike"), not on launch. `[INFERRED M]` The current code also does this.
- Turret damage to champions is physical. Shields, damage reduction and items apply normally. Not true damage. `[WIKI H]`
- No crit (`critDamageMultiplier` exists, but turrets have 0 crit). `[INFERRED H]`
- No tier-specific champion-damage modifier exists in 26.19 besides AD. `[CLIENT H]` (only item 1500 modifies the shot)

### 4.2 Turret shot vs minion `[CLIENT H]` (items 1508–1511 tooltips) + `[WIKI H]` (ItemData: "as **pre-mitigation** damage")

`raw = minion.max_hp × f`, physical, then reduced by the minion's armor after the turret's 30 % armor penetration.

| Minion | f | Tag |
|---|---|---|
| Melee | 0.45 | 1509 Gusto |
| Caster | 0.70 | 1510 Phreakish Gusto |
| Siege (cannon) | 0.14 outer, 0.11 inner, 0.08 inhibitor/Nexus turret | 1508 Anti-tower Socks |
| Super | **0.07** | 1511 Super Mech Armor. **The current code uses 0.05** (§10 D3) |

Notes:
- Melee minions gain armor late (0 for upgrades 1–5, then rising, capped at 20). Super minions have 100 armor. The Super Mech Power Field gives nearby allied minions +35 armor/MR, which increases the mitigation on turret shots against them. `[CLIENT H]`
- Practical (wiki) last-hit patterns with 0 armor:
  - melee: 2 shots + 1 champion hit;
  - caster: champion hit + 1 shot + champion hit.

### 4.3 Damage to turrets / structures (incoming)

Order of application for each incoming damage packet `P` against turret `T`. `[INFERRED M]` for
the ordering; components `[CLIENT H]`/`[WIKI H]`:

```
def damage_to_turret(T, P, now):
    # 1. pre-mitigation amount and type from the source
    #    champion basic attack: raw = baseAD + bonusAD + 0.6*AP ; type = MAGIC if 0.6*AP > bonusAD else PHYSICAL   [WIKI H]
    #    (strict >; on a tie default PHYSICAL [INFERRED L])
    #    lane minion: raw = minion_AD * 0.60  (dr_UnitToBuilding)                                                     [CLIENT H]
    #    siege minion: raw = AD * 0.60 * 1.40 = 0.84*AD  (Anti-tower Socks TurretDamageBonus 0.4)                    [CLIENT H]
    #      (mid-lane siege +30% AS vs turrets; irrelevant to top lane)
    #    super minion vs turret: 0.60 (12.5% only vs non-turret structures)                                         [WIKI M]
    # 2. resistances: armor or MR = base(60) - outer_decay(now) + bulwark(now, nearby_enemy_champs)
    #    (attacker armor/magic penetration applies; minimum effective resist after pen clamps at 0 for % pen order per shared-stats spec)
    # 3. mitigated = raw * resist_mult(eff_resist) for physical/magic; true damage unmitigated
    # 4. melee champion source: *1.20   (MeleeChampionBonusDamage 0.2)                                           [CLIENT H]
    # 5. Reinforced Armor (backdoor): if active: *0.20 (DamageReduction 80), applies to TRUE damage as well        [CLIENT H]
    # 6. subtract from HP, clamp >= 0; then evaluate plate thresholds / bulwark gains / death (§4.4)
```

- **Abilities:** most spells cannot target or affect turrets. Only the listed exceptions (wiki "Abilities that deal bonus damage to turrets") apply. On-hits apply only when the kit says so. This belongs to the champion agents. `[WIKI H]`
- **Health-based damage modifiers and crits** do not apply to turrets unless stated. `[WIKI H]`
- **Melee +20 %** applies to "damage from melee champions". Default: it applies to all champion-sourced damage from a melee champion, including on-hits. It does **not** apply to the Overgrowth packet (§6.4). `[INFERRED M]`
- **Inhibitor and Nexus:** steps 1–3 only, using their own resistances (20 armor, 0 MR). No melee +20 % and no Reinforced Armor (no pseudo-items). `[INFERRED M]`

### 4.4 Reinforced Armor (backdoor protection)

Source: item 1502 (outer and inner), 1506 (inhibitor turret), 1505 (Nexus turret). `[CLIENT H]` values, `[WIKI H]` rules.

- **Active** (80 % damage reduction, including true damage) when **no enemy lane minion and no Rift Herald mercenary** is "nearby".
  - Pets, traps and neutral monsters do **not** deactivate it.
  - Super minions are lane minions and do deactivate it.
- **Deactivation is immediate** when a qualifying unit becomes nearby. **Re-activation** happens 3 s after the last qualifying unit leaves (or dies).
- **Radius: not published in any source.** Default **1000 units (center-to-center)**. `[INFERRED L]` Choose and expose it as a parameter; see §12 U2.
  - The current world uses 1200 (`world/config.py:132`).
- While active, Overgrowth cannot be consumed (§6).
- **Applies to:** all four turret tiers (all four have a Reinforced item). Not to the inhibitor or Nexus.
- Swiftplay/OFA overrides (90 %) are out of scope.

## 5. Turret plating and Bulwark (outer, inner, inhibitor turrets)

### 5.1 Plates `[RIOT H]` + `[CLIENT H]`

- Plate *k* (1..5) is claimed the first time `hp <= max_hp × (1 − m_k)` with `m = (0.10, 0.25, 0.45, 0.70, 1.00)`. So the HP thresholds are 90 %, 75 %, 55 %, 30 %, 0 %.
  - Outer 9000: 8100 / 6750 / 4950 / 2700 / 0
  - Inner 5000: 4500 / 3750 / 2750 / 1500 / 0
  - Inhibitor turret 4750: 4275 / 3562.5 / 2612.5 / 1425 / 0
- The 5th plate is the destruction itself. So turret destruction pays the 5th plate's gold, plus the global destroy gold, plus (if first) the first-turret 300.
- **Plates are permanent.** The 14:00 fall-off was removed in 26.1. Legacy announcer strings remain in the client but are unused. Regeneration (inhibitor turret) never un-claims a plate or re-pays it. `[INFERRED H]`
- One packet can claim several plates. Each claimed plate pays gold, and each of plates 1–4 grants a Bulwark stack.
- **Plate gold, outer:** `120 − 10 × outer_decay_steps(now)` (120, 110, 100, 90, 80). Inner and inhibitor turret: flat 120.
  - Riot: "Plate value reduces by 10g per minute at minute 11, capped at −40". Max local per outer 600, min 400 `[RIOT H]`.
  - The value is evaluated at the moment the plate is claimed.
- **Sharing:** the gold is **split equally** (`SplitLocalGold true`) among eligible allied champions. Eligible means any of:
  - alive and within 1200 of the turret at the moment of the claim;
  - dealt the claiming blow;
  - they or their summons (incl. Herald mercenary) damaged the turret within the last 10 s. These are eligible regardless of range or death.
  - `[WIKI H]` "same paradigm for local rewards also applies to plating stages".
  - 1v1 lane: the attacker usually gets all 120.
- **Nexus turrets:** no plates. The 26.1 notes list "NEW: Turret Plates" under Nexus turrets by mistake. Client (no 1515 on SR_Nexus), the overview text and the wiki all say none. Implement none. `[CLIENT H]`
- **Structure-kill credit:** `events_TimerForBuildingKillCredit 30 s` (classic constants) is the scoreboard/assist credit window. The local-gold window is 10 s `[WIKI H]`. Keep them separate. `[CLIENT M]`

### 5.2 Bulwark `[CLIENT H]` (item 1515: `BulwarkResists 30`, `BulwarkResistsPerAdditionalChampion 5`, `BulwarkDuration 20`) + `[RIOT H]` 26.1/26.3 + `[WIKI H]`

- Each claim of plates 1–4 adds one **independent** stack lasting 20 s ("overlapping durations" since 26.1; each stack expires on its own timer). At most 4 stacks.
- Bonus armor **and** MR per active stack: `30 + 5 × max(0, n − 1)`, where `n` = number of **alive enemy champions within 850** of the turret center, **evaluated continuously** (at each damage packet).
  - n = 0 or 1 → 30. n = 5 → 50.
  - Max bonus at 4 stacks, n = 5 → 200.
- Bonus resistances are subject to the attacker's penetration like any armor/MR.
- A stack gained by a packet applies to **subsequent** packets, not the packet that claimed it. `[INFERRED M]` (current code does this)

### 5.3 Outer decay `[CLIENT H]` (1515: `OuterDecayBegins 660`, `OuterDecayEnds 900`, `OuterDecayPerMinute −15`, `OuterGoldDecay −10`) + `[RIOT H]`

```
outer_decay_steps(now) = clip(floor((now - 660)/60) + 1, 0, 4)     # 11:00 ->1, 12:00 ->2, 13:00 ->3, 14:00 ->4 (cap)
outer_base_resist(now) = 60 - 15*steps    # 60,45,30,15,0
outer_plate_gold(now)  = 120 - 10*steps   # 120,110,100,90,80
```

Default discrete steps, first at 11:00 (Riot: "starting at minute 11", "at minute 11").
`OuterDecayEnds 900` is read as "no further change after 15:00", which is consistent with a
cap of 4 steps. Alternative readings (continuous ramp 660→900, or steps 12:00…15:00) give the
same cap with a different timeline (§11 C1, §12 U4). This applies to the outer turret only.
Bulwark adds on top of the decayed base.

## 6. Crystalline Overgrowth (lane turrets T1–T3; season 2026)

### 6.1 What it is `[RIOT H]` 26.1

Crystals build up automatically on each **targetable** lane turret. The next attack by an
**enemy champion** (or, since 26.6, the Rift Herald mercenary's Charge / Spin Attack / Auto
Attack) bursts them. The burst is a separate packet of **bonus true damage** to the turret.
It scales with the attacking team's average level and with how long the crystals have grown,
**not** with the attacker's stats. Pseudo-item 1524.

### 6.2 Exact formula: recovered from client data `[CLIENT H]`

Item **1524** `mDataValues` (SR default; the SWIFTPLAY override is listed for verification only):

| Data value | SR | Swiftplay |
|---|---|---|
| `BuildupProcCooldown` | **90** s | 30 |
| `BuildupStartTime` | **60** s | 30 |
| `BuildupEndTime` | **300** s | 210 |
| `BuildupStartDamageMult` | **1.0** | — |
| `BuildupEndDamageMultMin` | **1.65** | 1.75 |
| `BuildupEndDamageMultMax` | **2.15** | 3.0 |
| `DamageBase` | **1.6** (% max HP) | 1.5 |
| `DamagePerLevel` | **0.4** (% max HP per level) | 0.65 |
| `BuildupVisualThresholdMed` / `Large` | 0.25 / 0.75 | — (VFX sizes S/M/L; also the likely melee-buffer tiers, §6.5) |

Reconstruction, checked against **all eight** published endpoints (26.1 notes, SR and Swiftplay):

```
L         = average champion level of the ATTACKING team (see §6.6 and §12 U6 for rounding)
base_pct  = DamageBase + DamagePerLevel * L                          # % of turret MAX HP
f_L       = clip((L - 1) / 17, 0, 1)
end_mult  = EndDamageMultMin + (EndDamageMultMax - EndDamageMultMin) * f_L
g         = time the crystal has been ACTIVE (since it appeared, incl. fast-forward credit)
r         = clip((g - BuildupStartTime) / (BuildupEndTime - BuildupStartTime), 0, 1)   # 60 s hold, 240 s linear ramp
mult      = BuildupStartDamageMult + (end_mult - BuildupStartDamageMult) * r
damage    = turret.max_hp * base_pct/100 * mult                      # TRUE damage
```

| Check | Formula | Published |
|---|---|---|
| SR L1 min | 1.6 + 0.4 = **2.0 %** | 2 % ✓ |
| SR L1 max | 2.0 × 1.65 = **3.3 %** | 3.3 % ✓ |
| SR L18 min | 1.6 + 7.2 = **8.8 %** | 8.8 % ✓ |
| SR L18 max | 8.8 × 2.15 = **18.92 %** | 18.9 % ✓ |
| SP L1 min | 1.5 + 0.65 = 2.15 % | 2.15 % ✓ |
| SP L1 max | 2.15 × 1.75 = 3.7625 % | 3.76 % ✓ |
| SP L18 min | 1.5 + 11.7 = 13.2 % | 13.2 % ✓ |
| SP L18 max | 13.2 × 3.0 = 39.6 % | 39.6 % ✓ |

Confidence:
- The **minimum curve is linear in level and exact.** `[CLIENT H]` It is a data value per level.
- The **end-multiplier interpolation between levels 1 and 18** uses the standard Riot lerp `(L−1)/17`. `[INFERRED M]`
  - The data gives Min/Max multipliers; the Min ↔ L1 and Max ↔ L18 binding is confirmed by all four checks.
  - The interpolation variable itself is not in client data.
  - Alternatives: stepwise per integer level, or lerp on `base_pct`. All agree at the endpoints.
- **Above level 18** (the top-lane role quest allows 20): `base_pct` extends naturally (L20 → 9.6 %). Whether `f_L` clamps at 1 is unknown.
  - Default: clamp `f_L` to [0,1] and let `base_pct` grow. `[INFERRED L]` §12 U6.
- The current implementation's linear interpolation of the **max** endpoint is wrong between levels (it overestimates by up to ~0.85 pp of max HP near L9). See §10 D1 and fixtures §13.3.

### 6.3 Lifecycle `[RIOT H]` + `[WIKI H]` + `[CLIENT H]` buff names `OvergrowthCooldown`, `OvergrowthReady`, `OvergrowthMeleeBuffer`

```
state per lane turret: og_cd_start (time the 90 s cooldown began), og_active (bool), og_growth_origin
on becoming targetable at time t0 (outer: game structure activation; inner/inhib: when predecessor dies):
    og_cd_start = t0                                     # outer: wiki "after 90 seconds (1:40 game time)" => t0 = 0:10
each tick:
    if not og_active and now >= og_cd_start + 90:
        if no enemy unit "near" the turret:              # suppression only blocks APPEARANCE
            og_active = True
            og_growth_origin = og_cd_start + 90          # fast-forward: growth time counts from when it SHOULD have appeared
        # else: stay hidden, clock keeps running (deferred; no damage lost)
    g = now - og_growth_origin  (only meaningful when og_active)
on enemy champion basic attack (or Herald charge/spin/auto) HIT on turret while og_active:
    if Reinforced Armor active: do nothing (crystal not consumed)                       [RIOT H]
    else: deal overgrowth damage (6.2) as a separate TRUE packet; og_active = False; og_cd_start = now
on turret death: discard
```

- **Who can consume it:** an enemy **champion** *attack* (basic attack). Abilities or on-hit-only spells do not count, unless the kit says "attack". Minions **cannot** consume it. `[RIOT H]` "the next time an enemy champion attacks it"
  - The prior MODERN_PATCH_DELTA note about a "minions-attacking-under-turret" trigger is **not supported** by Riot's notes, the wiki or the client tooltip ("Attacking the turret deals bonus damage"). Reject it.
  - `SR_2026_S1_TurretOvergrowthMinionManager` is a persistent manager buff of unknown role (98-byte metadata stub). Most likely it handles minion-proximity suppression. `[INFERRED L]`
- **Suppression:** "if an **enemy unit** is near the turret at that 90 s mark". Default: enemy champions **and** enemy lane minions count. `[INFERRED M]`
  - The radius is unpublished. Default: the turret's attack range edge test (750 + radii). `[INFERRED L]` §12 U2.
  - Once active, the crystal is never removed by enemies arriving. `[WIKI H]`
- **Cooldown restart:** it restarts at consumption. The next crystal appears 90 s later with growth reset (60 s at minimum damage again).
- **Wiki tooltip text:** "capped at 300 seconds of growth time" (ItemData) equals `BuildupEndTime 300`. Consistent with the above: the growth clock g is measured from appearance, and the cap is at g = 300.
- **Damage timing:** the crystal packet applies when the consuming attack hits. Default ordering: the attack's own packet first, then the crystal packet; both use the same tick's resist state. `[INFERRED L]` §12 U7.
- **Reinforced Armor never reduces crystal damage**, because the crystal is not consumable while backdoor is active.

### 6.4 Damage packet properties

- Damage type: **true**. The source is the turret's Overgrowth (wiki: "bonus true damage on their attack").
  - Default: it is **not** multiplied by the melee +20 %. `[INFERRED M]` (Riot: "It's not based on the attacker's stats.")
- It counts toward plate thresholds, Bulwark, destruction and the local-gold assist window like any champion damage. `[INFERRED H]`
- It cannot crit. Lifesteal/omnivamp: none. `[INFERRED M]`

### 6.5 Melee buffer ("attack hitbox increased for melee champions based on the size of the overgrowth") `[RIOT H]` existence; **amount unknown**

- The `OvergrowthMeleeBuffer` buff exists. No numeric value was found in any client bin, the string table, the wiki or the notes.
- Likely tiered by the VFX size thresholds (`BuildupVisualThresholdMed 0.25`, `Large 0.75` of the buildup fraction `r`). `[INFERRED L]`
- Default implementation: melee champions (attack range ≤ 250) get **+0** bonus range. This is a conservative no-op, flagged. §12 U8.

### 6.6 Average team level `[RIOT H]` "average level of the attacking team"

Default: arithmetic mean of the alive *and dead* champions on the attacking team, **not rounded**
(fractional L). `[INFERRED L]` §12 U6. In a 1v1 sim, L = the attacker's level.

## 7. Other structure rules

### 7.1 Regeneration and Nexus-turret respawn `[WIKI H]` / `[CLIENT H]` tooltips / `[RIOT H]`

```
inhib_turret: rate 3 HP/s; segments (0.30, 0.75, 1.00)
nexus_turret: rate 6 HP/s; segments (0.40, 0.70, 1.00)
cap = smallest segment s such that hp/max_hp <= s   (hp exactly AT a boundary: stays capped at that boundary  [INFERRED L])
hp  = min(cap*max_hp, hp + rate*dt)   while alive
```

- Regen is continuous, including while being hit. There is no out-of-combat gate in any source. `[INFERRED M]`
- Outer and inner turrets: no regen.
- Inhibitor: 15 HP/s; Nexus: 20 HP/s. Both uncapped and continuous. `[CLIENT H]`
- **Nexus turret respawn:** 180 s after death, at **40 % max HP** (1400). Warming stacks reset. Not targetable unless an inhibitor is down (§2). `[CLIENT H]`/`[RIOT H]`
- **Inhibitor respawn:** 300 s, at full HP. `[WIKI H]`

### 7.2 Overcharged (item 1507): sudden-death rule, LOW priority

- Present in the client inventory of all four SR records.
- Tooltip:
  - At `StartTime 55` minutes: "begins to malfunction, losing armor and MR".
  - At `DamageTime 60` minutes: "begins to break down, losing an escalating % of max HP every 30 s" (`DamageInterval 30`).
  - The sizes `mEffectAmount [75,75]` are probably the resistance loss. `[CLIENT M]`
- The wiki tags it "exclusive: Clash" `[WIKI M]`. Conflict: §11 C6.
- Irrelevant to sims under 55 min. Do not implement for the lane milestone.

### 7.3 Dynamic terrain (note only, DEFERRED per MODERN-009)

- Turrets sit on impassable terrain that **remains after destruction** (rubble). `[WIKI H]`
- Inhibitor and Nexus footprints (pathfinding radius 213.75 / 304) are also static obstacles.
- Respawning structures do not change collision.
- The only dynamic effect is removing turret *units* (targeting/vision). The navgrid already contains the turret pedestals. No overlay is required for structure death. `[INFERRED M]`

### 7.4 Mechanics that do NOT exist in 26.19 SR (do not implement)

| Mechanic | Status | Tag |
|---|---|---|
| "Fortification" (outer top/mid 50 % DR for the first 5 min) | removed 25.05 | `[WIKI H]` |
| Lane Swap Detector (95 % DR, 1000 % dmg, gold redirection) | removed from SR in 26.01. Client SR records lack item 1501 | `[WIKI H]` + `[CLIENT H]` |
| Plating 14:00 fall-off | removed 26.01 | `[RIOT H]` |
| "Plates take 17 % less damage from minions and ranged champions" | removed 26.01 (replaced by melee +20 %) | `[RIOT H]` |
| +50 permanent resist per dead plate | removed 26.01 | `[RIOT H]` |
| Inner-turret resist growth 16–30 min | removed 26.01 | `[RIOT H]` |
| Turret XP on destroy | none | `[CLIENT H]` |
| Old "Heat" laser on base turrets | removed long ago; all tiers fire the same 1200-speed missile | `[CLIENT H]` (NexusTurretBasicAttack is a 1200-speed missile) |
| `TurretFortification` / `TurretInitialArmor` buff names | legacy buff stubs in shared.bin; not on SR item lists | `[CLIENT M]` |

---

## 8. State the simulator must carry

Per **turret** (fixed capacity 22 on SR; the lane sim may restrict this):

| Field | Type | Meaning |
|---|---|---|
| `tier` | int {0 outer, 1 inner, 2 inhib, 3 nexus} | static |
| `team`, `lane`, `pos` | static | |
| `prereq` | int slot or −1 | outer → −1; inner → outer; inhib → inner; nexus → "any inhibitor of team dead" (a team-level predicate, not a slot) |
| `hp`, `max_hp` | f32 | |
| `alive`, `targetable` | bool | targetable is derived each tick from §2 |
| `respawn_at` | f32 (nexus turret only; +inf otherwise) | |
| `plates_claimed` | int 0..5 | monotone |
| `bulwark_expiry[4]` | f32 | independent stack expiries |
| `backdoor_off_until` | f32 | Reinforced Armor is inactive while `now < backdoor_off_until`; set to `now + 3` whenever a qualifying enemy minion/Herald is near |
| `og_cd_start`, `og_active`, `og_origin` | f32, bool, f32 | §6.3 (`og_cd_start = +inf` while untargetable) |
| `warm_stacks` (0..3), `warm_until` | int, f32 | §4.1 |
| `target` | int slot or −1 | sticky lock |
| `attack_ready_at` | f32 | next allowed attack start |
| `windup_end`, `windup_target` | f32, int | pending shot (−1 if none) |
| in-flight missiles | (source turret, target, launch time / position, damage snapshot policy) | resolved on impact; dropped if the source turret is dead |

Per **inhibitor**:
- `hp` (4000 max), `alive`, `respawn_at` (+300 s), `targetable`;
- `armor 20`, `regen 15/s`.

Per **Nexus**:
- `hp` (5500), `alive`, `targetable`, `regen 20/s`, `armor 20`.

Per **champion** (structure-related):
- `last_damaged_turret_at[turret]`: for the 10 s local-gold assist window. Can be compressed to per-(champion, turret) times in a 1v1.
- Pending "damaged allied champion" events, with the victim's position, for §3.3.
- Team average level: for Overgrowth.

Global:
- `first_turret_taken` (bool);
- game clock;
- per team, the count of inhibitors currently dead (super-minion spawn and Nexus-turret vulnerability).

Damage-snapshot policy for turret missiles: compute the damage **on impact** using:
- AD and warming stacks at impact;
- target armor at impact.

`[INFERRED M]` The current code does this.

## 9. Events / hooks and in-tick ordering

The recommended order within one sim tick (`now = t`, `dt`). Steps marked `[INFERRED M]` are a
deterministic convention; the server order is not observable from the sources.

```
1. Structure bookkeeping
   a. respawn: nexus turrets with now >= respawn_at -> hp = 0.4*max, alive; inhibitors with now >= respawn_at -> full hp
   b. recompute targetable for all structures (§2); a structure becoming targetable starts its Overgrowth cooldown (og_cd_start = now)
   c. regen: inhib turret 3/s, nexus turret 6/s (segment caps), inhibitor 15/s, nexus 20/s
   d. proximity scans (positions from end of previous tick):
        minion_near (backdoor radius)  -> backdoor_off_until = now + 3
        enemy_near (suppression radius) -> Overgrowth appearance gate
        n_champs_850                    -> Bulwark value
   e. Overgrowth appearance (§6.3); warming expiry (stacks = 0 if now >= warm_until)
2. Champion/minion actions resolve (other specs): damage packets to structures go through §4.3 in packet order:
      per packet: mitigate -> hp -= dmg -> claim plates (gold events, Bulwark stacks for later packets) -> record assist time
      -> if attack by enemy champion (or Herald) and og_active and backdoor inactive: crystal packet (§6.2), consume
      -> if hp <= 0: DESTROY event
   Damage events "enemy champion damaged allied champion" are collected here (including blocked / 0 dmg).
3. Turret AI (per alive, targetable-or-not turret; untargetable inner turrets still attack)
   a. drop lock if target dead / untargetable / out of range / invisible
   b. champion-protection override from events of step 2 (and of the previous tick's tail) (§3.3)
   c. if no lock: acquire by §3.1
   d. if a windup completes this tick: launch missile at the locked target (if still valid; else cancel)
   e. if idle and now >= attack_ready_at and target valid: begin windup (0.167 s), attack_ready_at = now + 1.2005
4. Missiles: advance; on impact apply §4.1 / §4.2 (turret source must be alive, else drop)
5. DESTROY handling (in packet order):
   turret: plates -> 5; pay remaining plate gold, global destroy gold (all allies, alive or dead),
           first-turret 300 (local, shared) if !first_turret_taken; unlock successor's targetability next tick;
           nexus turret: respawn_at = now + 180; in-flight shots from it are discarded
   inhibitor: respawn_at = now + 300; last hitter +50; team inhibitors_dead += 1 (super minions)
   nexus: game over
```

Gold events emitted:
- `PlateClaimed(turret, k, value, eligible_set)`;
- `TurretDestroyed(turret, global=50/25/25/50, first_turret_bonus)`;
- `InhibitorDestroyed(last_hitter +50)`.

Sharing rule: §5.1. Eligibility is evaluated at the event's tick. `[WIKI H]`

### 9.1 Per-tick turret AI pseudocode (vectorizable)

```
def turret_tick(T, units, events, now, dt):
    if not T.alive: return
    valid = enemy(units, T) & units.alive & units.targetable & visible_to(T.team) \
            & (dist(T, units) <= 750 + 88.4 + units.radius)
    # (a) lock loss
    if T.target >= 0 and not valid[T.target]: T.target = -1; T.windup_target = -1   # cancel pending windup [INFERRED L]
    # (b) champion protection
    agg = valid & is_champion(units) & events.damaged_allied_champ_within_1400_of(T)
    if any(agg): T.target = argmin_where(agg, dist(T, units))
    # (c) acquire
    if T.target < 0 and any(valid):
        p = min(priority(units)[valid]); T.target = argmin_where(valid & (priority==p), dist(T, units))
    # (d) windup completion
    if T.windup_target >= 0 and now >= T.windup_end:
        launch_missile(T, T.windup_target, speed=1200); T.windup_target = -1
    # (e) start attack
    if T.target >= 0 and T.windup_target < 0 and now >= T.attack_ready_at:
        T.windup_target = T.target; T.windup_end = now + 0.1669; T.attack_ready_at = now + 1/0.833
```

### 9.2 Damage computation pseudocode (incoming)

```
def resist_of(T, now, n850):
    base = 60 - (15*outer_decay_steps(now) if T.tier == OUTER else 0)
    stacks = count(T.bulwark_expiry > now)
    return base + stacks * (30 + 5*max(0, clip(n850, 0, 5) - 1))

def hit_turret(T, now, phys, mag, true, src):          # src: champion/minion/herald metadata
    R = resist_of(T, now, n850(T))
    a = max(0, R*(1-src.armor_pen_pct) - src.lethality_flat)   # see shared stats spec for exact pen order
    m = max(0, R*(1-src.magic_pen_pct) - src.magic_pen_flat)
    dmg = phys*100/(100+a) + mag*100/(100+m) + true
    if src.is_champion and src.is_melee: dmg *= 1.2
    backdoor = now >= T.backdoor_off_until
    if backdoor: dmg *= 0.2
    apply(T, dmg, src, now)                            # plates/bulwark/assist/destroy
    if src.can_consume_overgrowth and T.og_active and not backdoor and T.alive:
        g = now - T.og_origin
        og = T.max_hp * overgrowth_pct(L_team(src), g)  # §6.2
        T.og_active = False; T.og_cd_start = now
        apply(T, og, src, now)                         # TRUE, not melee-amplified, not backdoor-reduced (cannot be active)
```

Implementing the crystal as a separate `apply` means Bulwark gained by the attack's own packet
applies (irrelevant for true damage) and plates claimed by it are paid normally.

---

## 10. Diff vs current implementation

Files:
- `lanerl_jax/modern/lane/towers.py` (MT)
- `lanerl_jax/modern/world/config.py` (MW)
- `lanerl_jax/modern/data/26.19/towers.json` (TJ)
- `lanerl_jax/modern/tests/test_towers.py` (TT)

Line numbers are as of the working tree at research time.

| # | Location | Issue | Fix | Severity |
|---|---|---|---|---|
| D1 | MT:132–140 `overgrowth_level_fractions`; TJ `overgrowth.level_interpolation`, `level_curve_verified:false`, `unresolved[0]`; MW:83 `profile['overgrowth_level_curve']` | **Max fraction interpolated linearly** between 3.3 % and 18.9 %. The client data (item 1524) gives `max = (1.6+0.4L)% × (1.65 + 0.5·(L−1)/17)`, which is quadratic in L. Overestimates mid levels: e.g. L9 on a 9000 HP turret gives 957.7 vs 882.3. **Min fraction is already exact** (`.02+.068(L−1)/17 ≡ 1.6+0.4L %`) | Replace with the §6.2 formula. Store `DamageBase 1.6`, `DamagePerLevel 0.4`, `EndMultMin 1.65`, `EndMultMax 2.15`, `StartTime 60`, `EndTime 300`, `ProcCooldown 90` in TJ with item-1524 provenance. Mark the base curve verified and the multiplier lerp INFERRED-M. Update TT `test_explicit_growth_approximation_endpoints_and_finite_runtime` (L9.5 high .111 → (1.6+3.8)%×(1.65+0.5·8.5/17)=0.054×1.9=**0.1026**) | HIGH |
| D2 | MT:132 clamp 1–18 | Level 19–20 (top role quest) clamped. The client formula extends `base_pct` linearly | Default: clamp only `f_L`. Keep it flagged (§12 U6) | LOW |
| D3 | MT:210 `minion_shot_damage` super fraction `.05` | Client 1511 ("takes **7 %** of its health per turret shot") and wiki ItemData ("7 % … pre-mitigation") | `.05` → `.07`. Update TT `test_minion_hits_and_target_lock` (expects 50) and `test_locked_lane_growth_clock_and_minion_tier_damage` (expects 50/1.7 → 70/1.7) | MED |
| D4 | MW:66–69 `build_config` prerequisite | Nexus turrets get `prereq = −1`, so they are **targetable/damageable from game start**. Inhibitors are not modeled at all, so the inhibitor-turret → inhibitor → Nexus-turret → Nexus chain is missing | Add the inhibitor (4000 HP, 15 HP/s, 20 armor, respawn 300 s) and Nexus (5500, 20 HP/s, 20 armor) units. Nexus-turret targetability = "≥1 own inhibitor dead". Nexus = that ∧ both Nexus turrets dead (§2). Geometry already has `barracks` entries for positions | MED (LOW for a top-lane-only sim) |
| D5 | MW:136–137 `prepare` | The backdoor (`minion_near`) and Overgrowth suppression (`unit_near`) radii are both hard-coded **1200**. Neither is sourced | Make both explicit config parameters with provenance "unpublished; default". Recommended defaults: backdoor 1000, suppression = attack-range edge test (§4.4, §6.3). Record them in TJ `suppression_radius` (currently null) | MED |
| D6 | MW:137 `unit_near` | Counts **any** enemy unit (wards and pets too, once they exist) | Default: enemy champions + lane minions only | LOW |
| D7 | MT:180–181 / `champion_attack` flag | The Overgrowth consumer is champion-only. 26.6 added Rift Herald Charge/Spin/Auto | Caller flag `can_consume_overgrowth` covering Herald (when Herald exists) | LOW (no Herald in the lane sim) |
| D8 | MT:13 `ATTACK_PERIOD = 1.2` | Client AS 0.833 → 1.20048 s. No windup constant exists | Define `ATTACK_SPEED = 0.833`, `WINDUP_S = 0.139/0.833`. Integrate the windup | LOW |
| D9 | MT:15/`ATTACK_RANGE = 750 # edge to edge` | The in-range predicate is not implemented in MT. Caller-owned, but the convention is undocumented in code | Add a helper `in_range(dist_center, target_radius) = dist ≤ 750 + 88.4 + r` and flag §12 U5 | LOW |
| D10 | (missing) | **Turret AI is not integrated into the tick.** Nothing in `lanerl_jax/sim/*.py` calls `select_target`, `champion_shot_impact`, `minion_shot_damage`, `apply_turret_damage` or `local_reward_eligible`. `WorldState.plate_gold` and `first_turret` exist but are never written | Wire §9 into the modern step | HIGH (functional gap) |
| D11 | MT:215 `select_target` | Aggressor preemption picks the nearest aggressor ✓. There is no windup cancel on lock loss and no attack-cooldown state (caller) ✓ by design. Priority classes match §3.1 | OK. Add a test for "aggressor out of range does not preempt" | — |
| D12 | MT:62–75 `advance` | Backdoor grace (`now + 3` while a minion is near) ✓. Overgrowth suppression gates appearance only, and the clock is not paused (fast-forward) ✓. Matches §6.3 | OK | — |
| D13 | MT:150 `overgrowth_damage` ramp offset 150 | `150 = 90 (cooldown) + 60 (hold)`, measured from `growth_since` = cooldown start ✓ equivalent to §6.3. The 240 s ramp ✓ | OK. Name the constants from item 1524 | — |
| D14 | MT:77–79 `decay_steps` | Discrete steps at 11:00…14:00, capped at 4 ✓ (default reading, §11 C1) | OK | — |
| D15 | MT:82–85 `resistance` | Bulwark `30+5(n−1)` with n clipped [1,5] ✓. Dynamic n ✓. Independent stacks ✓ | OK | — |
| D16 | MT:181 | Melee ×1.2 applied to the whole non-crystal packet ✓. The crystal is not amplified ✓ (§6.4). Backdoor ×0.2 also multiplies the crystal, but `proc` already requires ~backdoor, so this is harmless | OK | — |
| D17 | MT:102–119 regen | Rates/segments ✓. A dead inhibitor turret does not regen ✓. Nexus respawn 40 % at +180 s ✓. Warming resets on respawn ✓ | OK | — |
| D18 | MT:196–203 `champion_shot_impact` | 1.0/1.5/2.0/2.5 ✓. 5 s from the last champion hit ✓. 30 % armor pen on positive armor ✓. The stack advances even on a 0-damage hit (shield) ✓ (a hit is a hit) | OK | — |
| D19 | TJ `outer.global_gold_on_destroy_per_champion 50`, `other_tiers.global_gold_per_champion [25,25,50]`, `first_turret_local_bonus 300`, `local_gold_radius 1200`, `local_gold_assist_window_seconds 10` | All ✓ client/wiki. **Missing:** destroy XP = 0 (state it explicitly), inhibitor last-hit 50 g, inhibitor respawn 300 s, `SpawnCountPerInhibitorDown [1,1,2]` | Add with provenance | LOW |
| D20 | TJ `sources` | Missing client provenance for the numbers it already has (item 1500/1502/1503/1515/1524 data values) | Add the item ids + `items.cdtb.bin.json` sha256 | LOW |
| D21 | `lanerl_jax/sim/modern_bridge.py:286` (champion-owned) | Jax W/R magic halved vs turrets (`*0.5`). Not a generic structure rule in 26.19 | Flag to the champion agent. Not a tower-module change | — |
| D22 | MT:127 `champion_structure_attack` | Strict `0.6AP > bonusAD` → magic, so a tie → physical ✓ (convention) | OK | — |

## 11. Conflicts between sources (and the default chosen)

| # | Topic | Sources | Default | Why |
|---|---|---|---|---|
| C1 | Outer decay timeline | Client `OuterDecayBegins 660 / Ends 900 / −15 per min`. Riot "15 per minute starting at minute 11, capped at −60". Wiki "11:00 until 15:00" | Discrete −15 at 11:00, 12:00, 13:00, 14:00 (cap); unchanged after | Riot's "at minute 11" wording plus cap = 4 steps. A continuous ramp 660→900 is equally consistent with the data (§12 U4) |
| C2 | Outer AD cap time | Wiki tooltip: steps at 0:30 + every 60 s → cap 13:30. Riot: "168 at 14 minutes" | 13:30 | Detailed schedule. Riot's figure is a rounded summary |
| C3 | Super minion turret-shot % | Client 1511 7 %, wiki ItemData 7 % pre-mitigation. Wiki super-minion history "~5 %" (a historical post-mitigation estimate: 7 % × 100/170 = 4.1 %) | 7 % pre-mitigation | Client data |
| C4 | Nexus plates | 26.1 Nexus section lists "NEW: Turret Plates". 26.1 overview, client (no 1515 on SR_Nexus) and wiki say none | None | Client data |
| C5 | Inhibitor/Nexus armor | Client root records have no armor field (default 0). Wiki 20 armor for both (Nexus: "undocumented, 2025") | 20 armor, 0 MR | Server-set stats are not visible in client data. The wiki is the only positive evidence. Flag §12 U9 |
| C6 | Overcharged (1507) on SR | Client inventories of all SR records include it. Wiki ItemData says "exclusive: Clash" | Not implemented (≥55 min) | Out of the sim horizon either way |
| C7 | Overgrowth "trigger" | MODERN_PATCH_DELTA §6.4 cited a second source saying minions attacking under the turret trigger it | Enemy champion attack (+ Herald) only | Riot notes, wiki and client tooltip agree |
| C8 | Local gold on outer destroy | Legacy Root record 250. SR_Outer omits it (0). Riot "250 ⇒ 0" | 0 | Client SR record + Riot |
| C9 | Plate gold sharing wording | Client tooltip "shares 120 gold with each nearby champion". Wiki "120 local gold" with the turret local-reward paradigm (split). `SplitLocalGold true` | Split 120 equally among eligible champions | Classic constant + wiki. A 1v1 is unaffected |

## 12. Unresolved / needs live measurement

Practice Tool on patch 26.19, normal SR, with a replay/recording at ≥30 fps unless stated.

| # | Question | Default used | Suggested test scenario |
|---|---|---|---|
| U1 | Does the champion-protection aggro "remember" an aggressor who is out of range at trigger time (and for how long)? | No memory | Ally dummy at 1300 from turret, aggressor champion at 900 (outside 750+radii) autos the ally once, then walks into range 0.5 / 1 / 2 / 4 s later while a minion is the turret's target. Record whether and when the turret switches |
| U2 | Backdoor radius; Overgrowth suppression radius | 1000; attack-range edge | Backdoor: walk one enemy minion (Practice Tool spawn) toward the turret from 1500 in 50-unit steps. Hit the turret with a fixed 100-damage champion basic attack and read the damage number. The radius is where 20 → 100 flips. Repeat leaving (expect a 3 s delay). Suppression: stand at distance d from a turret at its 90 s mark; find the max d at which the crystal is deferred |
| U3 | Lock released during windup (target leaves range mid-windup): cancel or fire? | Cancel | Champion steps out of range timed to the turret's attack start (frame-step the recording). Check whether a shot is launched |
| U4 | Decay discrete vs continuous | Discrete at 11:00 | Read turret armor (shift-hover stats) at 10:59, 11:00, 11:30, 12:00, 14:00, 14:30, 15:00 |
| U5 | Exact range predicate (edge vs center + offsets) | `750 + 88.4 + r_target` | Melee minion or champion of known gameplay radius. Find the max center distance at which the turret begins attacking (approach in 5-unit steps with move commands) |
| U6 | Team average level: fractional or rounded? Multiplier interpolation between L1 and L18? Behaviour above 18 | Fractional; lerp `(L−1)/17`; clamp `f_L` | Practice Tool: set the team level to 9 (all) and to 8/9/9/10/10 (avg 9.2). Wait 300 s+ of growth (max state) and hit an outer turret with a single basic attack (no backdoor: minion nearby). Expected max L9: 9000 × 0.052 × 1.88529 = **882.3**; linear alternative 957.7; stepwise variants differ. Repeat at L=20 |
| U7 | Ordering of the crystal packet vs the attack's own packet, and whether one packet claiming a plate gives the crystal Bulwark (irrelevant: true damage) | Attack first, then crystal | Only matters for plate/gold accounting at threshold edges. Hit a turret at 90 % + ε HP with a consuming attack. Check the plate-gold timing in the event feed |
| U8 | Melee-buffer bonus range amounts | +0 | Melee champion with a fully grown crystal (r = 1). Find the max center distance at which a basic attack on the turret starts, vs no crystal |
| U9 | Inhibitor/Nexus armor 20? | 20 | Hit an inhibitor with a known-AD basic attack (no pen). 100 AD → 83.3 if 20 armor, 100 if 0 |
| U10 | Does a minion shot inside the 5 s Warming window keep the stacks? | Window not refreshed by minion shots | Turret shoots the champion twice, then a minion for 4 s, then the champion again at 6 s after the last champion hit. Expect the stack reset (×1.0) |
| U11 | Overgrowth start time for outer turrets (wiki says 1:40, i.e. cooldown from 0:10) | Cooldown starts at 0:10 | Observe the first crystal appearance time with no enemies near |
| U12 | Same-tick tie-break in target acquisition | Lowest slot | Two equidistant melee minions (hard to stage; low value) |

## 13. Test fixtures (inputs → expected outputs)

All times are game seconds. Turret resist = 60 unless stated. `armor_mult(a) = 100/(100+a)`.

### 13.1 Stats / schedules

| Input | Expected |
|---|---|
| `outer_AD(t)` for t = 0, 29.99, 30, 89.99, 90, 600, 809.99, 810, 2000 | 182, 182, 194, 194, 206, **302**, 338, 350, 350 |
| `inner_AD(t)` (= inhib turret) for t = 0, 179.99, 180, 600, 1020, 2000 | 187, 187, 203, **315**, 427, 427 |
| `nexus_turret_AD(600)` | 293 |
| `outer_decay_steps(t)` for 659.9, 660, 720, 840, 900, 2000 | 0, 1, 2, 4, 4, 4 |
| `outer_base_resist(700)`; `outer_plate_gold(700)` | 45; 110 |
| `outer_base_resist(900)`; `outer_plate_gold(900)` | 0; 80 |
| Plate HP thresholds outer / inner / inhib turret | 8100, 6750, 4950, 2700, 0 / 4500, 3750, 2750, 1500, 0 / 4275, 3562.5, 2612.5, 1425, 0 |
| Max/min local plate gold per outer turret | 600 (all before 11:00) / 400 (all after 14:00) |

### 13.2 Damage

| Scenario | Expected |
|---|---|
| Outer at t = 600 (AD 302) shoots a champion with 40 armor, 4 consecutive hits within 5 s | eff armor 28 → per hit 302 × 100/128 = **235.94**, then ×1.5 = 353.91, ×2.0 = 471.88, ×2.5 = 589.84 |
| Same, 5th hit 6 s after the 4th | 235.94 (reset) |
| Switch from champion A (2 stacks) to champion B within 5 s | B's first hit uses ×2.0 |
| Outer shot vs melee minion, max HP 477, armor 0 | 214.65 |
| Super minion max HP 2000 (illustrative), armor 100 | raw 140, eff armor 70 → **82.35** |
| Cannon, max HP 1000, 0 armor: outer / inner / inhib / nexus turret | 140 / 110 / 80 / 80 |
| Melee champion basic attack, base 60 + bonus 40 AD, 0 AP, 0 pen, t = 300, minions near, 0 Bulwark | 100 × 100/160 × 1.2 = **75.0** |
| Same, no minions within backdoor radius (backdoor on) | 15.0 |
| Same, minion left 2.9 s ago / 3.0 s ago | 75.0 / 15.0 |
| Ranged champion, base AD 100, bonus AD 0, AP 200 (0.6 AP = 120 > 0 bonus AD → magic), MR 60, minions near | raw 100 + 120 = 220 → 220 × 0.625 = **137.5** |
| Bulwark: outer at t = 300, 2 active stacks, 3 enemy champions within 850 | resist 60 + 2 × 40 = 140 |
| Bulwark at t = 900 (decayed), 1 stack, 1 champion | 0 + 30 = 30 |
| Bulwark stacks claimed at t = 100 and t = 110; query at 119.99 / 120 / 130 | 2 stacks / 1 stack / 0 stacks |
| Minion (melee, AD 12 illustrative) hits a turret, resist 60 | 12 × 0.6 × 0.625 = 4.5; cannon AD 40: 40 × 0.84 × 0.625 = 21.0 |
| Inhibitor (armor 20), 100 physical, 30 % armor pen attacker | 83.33 (pen ignored vs inhibitor) |

### 13.3 Overgrowth (outer turret, max HP 9000; inner 5000)

`og(L, g)` with g = seconds since the crystal appeared:

| L | g ≤ 60 (min) | g = 180 (half ramp) | g ≥ 300 (max) | current-code max (D1) |
|---|---|---|---|---|
| 1 | 180.0 | 238.5 | 297.0 | 297.0 |
| 3 | 252.0 | 341.3 | 430.6 | 462.2 |
| 6 | 360.0 | 503.5 | 646.9 | 709.9 |
| 9 | 468.0 | **675.2** | **882.3** | 957.7 |
| 12 | 576.0 | 856.4 | 1136.8 | 1205.5 |
| 18 | 792.0 | 1247.4 | 1702.8 | 1701.0 |
| 20 (clamped f_L) | 864.0 | 1360.8 | 1857.6 | 1701.0 |

Inner turret 5000 HP, L 6, g ≤ 60: 200.0. Fully grown L 6: 359.4.

Lifecycle fixtures:
- **Outer:** cooldown start 10.
  - No enemies near: `og_active` false at 99.99, true at 100.0.
  - Consumed at 400: next active at 490. A hit at 490 deals the min value.
- **Suppression:** enemy near from 95 to 250, then leaves.
  - Active at 250 with `og_origin = 100`, so g = 150 at 250. At L1: 180 × (1 + 0.65 × (90/240)) = **223.875**.
- **Backdoor active** (no minions) at a consuming attack: crystal not consumed, `og_active` stays true, attack dmg × 0.2.
- **Inner turret:**
  - Unlocked when the outer dies at 900. Active at 990, not before.
  - A hit at 1050 (g = 60): min value. At 1170 (g = 180): half ramp.
- **Nexus turret:** never has Overgrowth.

### 13.4 Plates / gold

| Scenario | Expected |
|---|---|
| Fresh outer, single packet of 900 at t = 100 (after mitigation) | hp 8100 → 1 plate, 120 g to the eligible set, Bulwark stack expiring at 120 |
| Fresh outer, single packet of 9000 at t = 100 | 5 plates, 600 g, destroyed, +50 global per ally, +300 first-turret local (shared) |
| Outer at hp 5000 (2 plates claimed), packet 2400 at t = 700 | hp 2600 → plates 3 and 4 → 2 × 110 = 220 g, 2 Bulwark stacks (apply from the next packet) |
| Inner turret plate at t = 900 | 120 g (no decay) |
| Inhibitor turret at 50 % HP (2375, 3 plates claimed), regen 10 s | 2405. Cap 3562.5. Plates stay 3 |
| Nexus turret destroyed at 1000 | respawn at 1180 with 1400 HP. Plates 0, local gold 0, global +50 per ally |
| Eligibility: ally dead 4000 units away, damaged the turret 9.99 s ago | eligible. At 10.01 s ago: not eligible |

### 13.5 Targeting

| Setup (all in range unless stated) | Expected target |
|---|---|
| champion at 300, melee minion at 600, cannon at 700 | cannon |
| locked on a melee minion; a cannon walks into range | stay on the melee minion |
| locked on a melee minion; enemy champion (in range) autos an allied champion standing 1300 from the turret | switch to the champion |
| same, ally at 1450 | no switch |
| same, aggressor out of turret range | no switch (default, U1) |
| enemy champion attacks an allied *minion* | no switch |
| enemy champion's attack on the ally is spell-shielded (0 damage) | switch |
| locked champion leaves range | re-acquire by priority next attack decision |
