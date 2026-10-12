# CHAMPIONS.md — Garen, Jax, Nasus and Renekton, 26.19 (client 16.19.8230722)

**Status (2026-10-12).** Implemented in `lanerl_jax/modern/champions/` (`garen.py`, `jax.py`, `nasus.py`, contract
in `core.py`; `renekton.py`). Tests: `lanerl_jax/modern/tests/test_champions.py`, `test_nasus.py`, `test_nasus_world.py`,
`test_renekton.py`, `test_renekton_world.py`.
Pinned records are regenerated from the client bins by `python -m lanerl_jax.modern.data.build_champions`.

## Sources

- **CLIENT**: the pinned BIN snapshots `lanerl_jax/modern/data/26.19/champions/<name>.json` (spell data values,
  cooldowns, mana, record radii and acquisition range), and the full CDragon 16.19 records in
  `/mnt/nfs/shared/modern-world-map-research/cdragon-16.19/<name>.bin.json` (Nasus, Malphite, Renekton, Darius and
  Tryndamere fetched 2026-10-12, checksums in `SHA256SUMS`): spell calculations,
  `mOverrideAttackTime`, `mRollForCriticalHit`, cast times, spell tags, plus the 16.19 string table tooltips.
  Calculation `mStat` numbers were checked against the item calculations: 2 is AD, 4 is attack speed, 6 is MR,
  7 is move speed, 8 is crit chance, 9 is crit damage and 12 is health.
- **WIKI**: `Template:Data Garen/*` and `Template:Data Jax/*` ability data and the patch history pages,
  fetched raw on 2026-10-02; `Template:Data Nasus/*`, `Template:Data Renekton/*` and their patch histories cached raw in
  `cdragon-16.19/wiki-2026-10-12-champions/`.
- **PATCH**: Riot patch notes 26.1–26.19. These Summoner's Rift changes apply: 26.1 (Garen E crit is 30% of
  bonus crit damage; Jax passive gains a level-19 breakpoint), 26.5 (Garen Q MS duration, E AD ratio),
  26.12 (Jax Q costs 50 mana, Jax E deals 4% max HP) and 26.14 (Garen R base damage 125/200/275). The Garen
  and Jax sections in 26.16 and 26.17 are under **Classic** (Q magic damage, "Relentless Assault" R) or
  **Arena** (Perseverance 4 s, W 0.5/60). They do not apply to the Rift. The client values already include
  every Rift change.
- **INFERRED**: a rule chosen where neither source settles it. The suffix gives confidence (H, M or L).

## Garen (86)

| ID | Rule | Value (ranks 1–5 / 1–3) | Evidence |
|---|---|---|---|
| GAR.P1 | Perseverance regenerates `RegenCalc` % of max HP every 5 s: 1.5 at level 1, +0.2 per level to 6, +0.8 per level for 7–13, +0.4 per level from 14 (10.1% at 18, 10.5% at 19). | breakpoints | CLIENT `ByCharLevelBreakpointsCalculationPart` |
| GAR.P2 | Paid as 1/10 of that amount on every 0.5 s grid boundary. | 0.5 s | WIKI footnote |
| GAR.P3 | Disabled for `DamageTimer` seconds after Garen loses health to an enemy champion or turret, refreshed by each such hit. This timer is not reduced by ability haste. Minion and non-epic monster damage, damage taken by shields, and 0 damage do not disable it. | 8 s | CLIENT value; tooltip "Minion and non-epic jungle monster damage does not stop the regeneration"; WIKI notes |
| GAR.Q1 | Removes slows and grants 35% MS for `MovementSpeedDuration`. | 1.4/1.95/2.5/3.05/3.6 s | CLIENT, PATCH 26.5 |
| GAR.Q2 | The next basic attack within 4.5 s deals `BaseDamage + 0.5 × AD` bonus physical damage (`tADRatio` 1.5 total; the world's basic attack supplies 1.0 AD) and silences for 1.5 s. The bonus damage also applies to structures. | 30/60/90/120/150 | CLIENT, WIKI |
| GAR.Q3 | The empowered attack rolls crit like a normal attack. The bonus damage never crits. | — | CLIENT `GarenQAttack.mRollForCriticalHit = true`; WIKI note |
| GAR.Q4 | Q resets the attack timer. The empowered attack's windup cannot be cancelled. The Q attack's total attack time is `T = 1.7 − 0.2 × bonus AS`, and its windup is `0.2 T`. | — | CLIENT `mOverrideAttackTime`, `mCastTimePercent`; WIKI "locks Garen out of basic attacks … shortened with attack speed" |
| GAR.Q5 | Lunge: against champions, the empowered attack reaches 50 units beyond attack range. | +50 | WIKI V3.x note ("50 units closer to his target than his attack range"); INFERRED-M that this still holds |
| GAR.Q6 | The cooldown starts post-effect: when the empowered attack hits or is dodged, when the window expires, or on death. Q cannot be recast while the window is open. | 8 s | CLIENT tag `SpecialCase_DelayedCooldown`; WIKI `cdstart = post-effect` |
| GAR.W1 | Passive: each stack gives 0.2 armor and 0.2 MR, up to 150 stacks (30 each). There is no bonus at the cap (removed in V25.11). | — | CLIENT, WIKI |
| GAR.W2 | Garen gains 1 stack for each enemy champion he kills (killing blow only, not assists), each minion he last hits, and each monster he kills, large and epic included. Wards, turrets and structures give no stacks. He gains none before W is learned. | 1 per kill | CLIENT `BuffCounterPerKill`, `LargeMonsterStacks` and `EpicMonsterStacks` are all 1; WIKI stack sources; V25.23 learn-gate fix |
| GAR.W3 | Active: a `BaseShield + 0.18 × bonus HP` shield and 60% tenacity for 0.75 s, then `DRPercent` damage reduction for 4 s. Tenacity stacks multiplicatively; the reduction does not apply to true damage. The cooldown starts at cast. | 65–145 shield; 25/29/33/37/41% DR | CLIENT |
| GAR.E1 | Spins `7 + floor(bonus AS / 0.25)` times over 3 s, counted at cast. Spin k hits when it completes, at `k × 3 / n` s after the cast. | — | CLIENT `NumTicks`, `ASPerTick`; WIKI "spin rate 3 ÷ spins … deals damage … when the spin is completed" |
| GAR.E2 | Each spin deals `BaseDamagePerTick + ADRatioPerTick × AD` physical damage, using the AD at the moment of the spin, to enemies within 325 (center to edge). The nearest enemy takes +25%. Wards are not hit. | 4–16 + 40–52% AD | CLIENT, PATCH 26.5 |
| GAR.E3 | Each spin rolls crit independently for `1 + 0.3 × (crit damage − 1)`: ×1.3 at the 200% base, ×1.39 with Infinity Edge. | CritMod 0.3 | CLIENT `CriticalDamage` calculation; PATCH 26.1 |
| GAR.E4 | An enemy champion hit 6 times loses 25% armor for 6 s. The duration refreshes on the 7th hit and every 6th hit after that. While the shred is active, the hit count carries over to the next cast. | — | CLIENT, WIKI, V25.12 fix |
| GAR.E5 | Recast after 1 s, or casting R, ends E. The cooldown starts at the end. Garen cannot attack while spinning, is ghosted, and every spin is its own cast instance (one Conqueror stack per spin). | 9/8.25/7.5/6.75/6 s | CLIENT, WIKI |
| GAR.R1 | Targets an enemy champion within 400 (center to edge). After a 0.435 s cast it deals `BaseDamage + ExecuteDamage × missing HP` true damage, but only if the target is still the same living unit. There is no Villain rule (removed). The cooldown starts at cast. | 125/200/275 + 25/30/35%; 120/100/80 s | CLIENT, PATCH 26.14 |

## Jax (24)

| ID | Rule | Value | Evidence |
|---|---|---|---|
| JAX.P1 | Each attack launched gives a stack (max 8) lasting 2.5 s, refreshed on each new stack. After expiry, one stack falls off every `FallOffRate`. | 0.35 s | CLIENT. The wiki's 0.25 s is overridden by the client value. |
| JAX.P2 | Each stack gives 5% bonus AS, +1.5% at each of levels 4/7/10/13/16/19 (14% at level 19 and above). | — | CLIENT breakpoints, PATCH 26.1 |
| JAX.Q1 | Leaps to a unit within 700 (center to edge), ally or enemy, wards included, but not structures or the Faelight pad. It cannot be cast while rooted. Dash speed is 1400 and follows the target. | 50 mana, 8/7.5/7/6.5/6 s | CLIENT `castRange`, `mExcludedUnitTags`, `cantCastWhileRooted`, PATCH 26.12. Speed: INFERRED-M (legacy layer) |
| JAX.Q2 | On landing on the same living enemy that is not a ward: `Damage + 1.0 × bonus AD` physical, plus W's damage if W is active (this consumes W). An enemy champion target becomes Jax's attack order. The cooldown starts at cast. | 65–225 | CLIENT, WIKI. Physical: the 26.17 magic-damage change is Classic only |
| JAX.W1 | Lasts 10 s. Resets the attack timer, adds +50 range and makes the windup uncancellable. The next landed attack or Q deals `Damage + 0.6 × AP` magic damage as a separate instance, ×0.5 against structures. The cooldown starts on use, on expiry or on death. | 50–190; 30 mana | CLIENT, WIKI |
| JAX.E1 | Evasion lasts 2 s. Jax dodges basic attacks from everything except turrets, and takes 25% less damage from AoE packets. Mana is paid only at the start. | 50–90 mana | CLIENT, WIKI |
| JAX.E2 | Recasting after 1 s, or expiry, deals `(BaseDamage + 0.7 × AP + 4% target max HP) × (1 + 0.2 × min(dodges, 5))` magic damage within 375 (center to edge). Against monsters the max-HP part is capped at 9000. Every enemy hit is stunned for 1 s, including minions and monsters. The cooldown starts at release. | 40–160 | CLIENT `MonsterDamageCap`, PATCH 26.12, WIKI ("stuns them") |
| JAX.R1 | Cast time 0.25 s. Jax can move but not attack. The swing deals `SwingDamageBase + 1.0 × AP` magic damage within 375. Mana 100; the cooldown starts at cast. | 100/175/250; 110/100/90 s | CLIENT |
| JAX.R2 | Only if the swing hits a champion: for 8 s, bonus armor equal to `BaseResists + 0.4 × bonus AD + (champions − 1) × (ResistsPerExtraTarget + 0.1 × bonus AD)`, and `MRMult` (0.6) × that as MR. The value is fixed when the swing lands. | 45/60/75, +20/25/30 | CLIENT `BaseArmor`/`BonusArmor`/`BaseMR`; WIKI "If this hits a champion". Snapshot: INFERRED-M |
| JAX.R3 | Passive: each landed attack adds a stack (max 2, 2.5 s, refreshed). The attack that lands at 2 stacks (1 while R2 is active) consumes them and deals `PassiveBaseDamage + 0.6 × AP` magic damage as an on-hit proc, ×0.5 against structures. That attack's windup cannot be cancelled. Against wards the proc triggers but is not consumed and has no effect. Dodged attacks add no stacks. | 75/130/185 | CLIENT, WIKI |

## Nasus (75)

Patches that apply on the Rift (client values already include them): 26.16 (Q stacks 4, 10 on large kills), 26.17
(passive 10/15/20%), 26.18 (Q stacks no longer crit; Wither steps every 1 s, scaled by tenacity, and no longer ramps
twice as fast), 26.19 (Q base 30–110). The 26.16 "Nasus" block that also lists passive 14/17/20 ⇒ 10/15/20 and W mana
80 ⇒ 100 is a non-Rift mode section; the client keeps W at 80 mana.

| ID | Rule | Value (ranks 1–5 / 1–3) | Evidence |
|---|---|---|---|
| NAS.P1 | Soul Eater: life steal 10% at levels 1–6, 15% at 7–12, 20% from 13. | 10/15/20% | CLIENT `LifestealTooltip` breakpoints, PATCH 26.17 |
| NAS.Q1 | The next basic attack within 10 s gains 50 range and an uncancellable windup. Q resets the attack timer, has no cast time and cannot be recast while the window is open. Cooldown starts post-effect: when the attack lands or is dodged, when the window expires, or on death. | 20 mana; 7.5–3.5 s | CLIENT `BuffDuration`, `BonusRange`, `Trait_AttackReset`; WIKI `cdstart = post-effect` |
| NAS.Q2 | The attack deals `BonusDamage + stacks` bonus physical damage as a separate packet with life steal. The flat `BonusDamage` crits with the attack (× crit damage); the stacks never crit. | 30–110 | CLIENT `CritDamage` calculation; PATCH 26.18; WIKI ("basic damage … also spell damage") |
| NAS.Q3 | If the empowered attack's target dies to Nasus on the tick it lands, Nasus gains 4 permanent stacks, or 10 for champions (killing blow only), siege and super minions, and large monsters (camp large monsters, Voidgrubs, Herald, Baron, drakes, Elder). | 4 / 10 | CLIENT `BasicStacks`, `BigStacks`; tooltip; PATCH 26.16 |
| NAS.W1 | Targets an enemy champion within 700 (center to edge). After a 0.25 s cast time it slows by 35% for 5 s and cripples attack speed by 75% of the slow. The cooldown starts at cast. | 80 mana; 15–11 s | CLIENT `SlowBase`, `Duration`, `AttackSpeedSlowMult`; WIKI cast time |
| NAS.W2 | The slow grows by `SlowPerTick` at 1, 2, 3 and 4 s (47–95% at 4 s). With target tenacity T the steps come every `(1 − T)` s and the slow ends at `5 (1 − T)` s; each step re-applies the slow with the remaining base duration so the world's tenacity rule gives that same end. The cripple uses the base slow value (slow resist does not reduce it). | +3/6/9/12/15% per s | CLIENT `SlowPerTick`, `MaxSlowTooltipOnly`; PATCH 26.18; WIKI notes |
| NAS.E1 | Location within 650 (walks into range). After a 0.25 s cast time and a 0.25 s delay, deals `InitialHitDamage + 0.6 AP` magic damage to enemies within 380 of the point (center to edge). Dying during the cast time cancels it; the delay does not lock Nasus. | 60–100 mana; 12 s; 50–170 | CLIENT `castRange`, `castRadius`, `InitialDamage`; WIKI cast time and delay |
| NAS.E2 | The fire then burns for 5 s: one tick per second (5 ticks) of `DamagePerTick + 0.12 AP` magic damage, and enemies inside lose `ArmorShredPercent` of their armor, lingering 1 s after they leave or the fire ends. | 10–34 per s; 30–50% | CLIENT `TotalDotDamage` = per tick × 5 s and tooltip "over 5 seconds"; WIKI V14.18 "per second", linger |
| NAS.R1 | After a 0.2 s cast time, for 15 s: `BonusHealth` max health, `InitialResistGain` armor and MR, 50 attack range (adds to Q's 50), Siphoning Strike's base cooldown halved. Ends on death. The cooldown starts at cast. | 100 mana; 120/100/80 s; 300/450/600, 40/55/70 | CLIENT, PATCH V13.12 |
| NAS.R2 | Every 0.5 s (30 ticks) enemies within `400 + 0.8 × bonus size %` (424/428/432, center to edge) take `(AOEDamagePercent + 1% per 100 AP) × 0.5` of their max health as magic damage, capped at 120 per tick (240 per second) against every unit type. | 3/4/5% per s | CLIENT `DamageCalc`, `TickRate`, `MaxDamageCap`, `SizeIncreasePercent`; WIKI radius and cap |

### Nasus approximations

1. **Q on wards and structures.** Champion attacks on wards are ward hits that never reach the kit, so Q is not
   spent on a ward and gives no stacks. Turret kills give no stacks either, because the world's takedown matrix
   (`Kills.killed_units`) has no structure columns.
2. **Wither cleanse.** Cleanse or Garen Q end the world slow but the kit's cripple keeps running to the original
   end (the client treats both as one debuff). Wither's first-application windup interrupt (a listed bug) is not
   modelled. Holder tenacity from summoner Cleanse is not in `KitCtx.tenacity`.
3. **Spirit Fire geometry.** One fire per Nasus: at extreme ability haste (cooldown under 5.5 s) a new cast replaces
   the burning one. Sight of the area is not modelled. Radius is the client `castRadius` 380; the wiki lists 400.
   The 0.25 s delay after the cast time is the wiki's reading (INFERRED-M).
4. **Fury of the Sands size.** The size increase only widens the storm; Nasus's gameplay and pathing radii do not
   grow.

## Renekton (58)

Fury is kit state: 0–100 (`arIncrements` 50 × `arMaxSegments` 2); the world's mana stays 0 and the observation's
resource bar shows Fury / 100 (`champions.resource_fraction`). Rune legality treats him as resourceless (RUNES.md
§2.2). Rift patches: 26.17 W cooldown 16–8 s (client); 26.19 R +20 Fury after W bug fix. The 26.17 note's "(removed
Arena Mod)" refers to the Arena override; the client value is the Rift one.

| ID | Rule | Value (ranks 1–5 / 1–3) | Evidence |
|---|---|---|---|
| REN.P1 | Each landed basic attack on a non-structure gives 5 Fury. Below 50% max health every Fury gain is ×1.5. Fury is capped at 100 and reset to 0 on death. | 5; ×1.5 | CLIENT `FuryPerAttack`, `LowHealthPercentThreshold`, `FuryIncreasePercent`; WIKI; death reset INFERRED-M |
| REN.P2 | After 12 s without attacking, dealing or taking damage, Renekton loses 1 Fury every 0.25 s. Dominus keeps the clock refreshed. | 12 s | CLIENT `InCombatDuration`; WIKI |
| REN.P3 | With at least 50 Fury, the next Q, W or Dice (not Slice) consumes 50 Fury and is empowered. Empowered casts generate no Fury. | 50 | WIKI |
| REN.Q1 | Instant cleave within 400 (480 during Dominus, center to edge): `BaseDamage + 1.0 bonus AD` physical (empowered `EmpoweredBaseDamage + 1.4 bonus AD`). Heals per enemy hit: champions `ChampHeal + 0.17 bonus AD`, others `BaseMinionHeal + 0.02 bonus AD`, capped at `BasicHealCap`; empowered heals ×3 with `EmpoweredHealCap`. Fury: 2.5 per non-champion, 10 per champion, at most 30 per cast. For 0.25 s Renekton cannot attack, cast E or cast R. | 7 s; 60–180, 90–270; caps 50–150 / 200–600 | CLIENT `Range`, `UltimateRangeMod`, calculations; WIKI lockout |
| REN.W1 | Attack reset; for 6 s the next basic attack has 175 range (+50) and strikes 2 times (3 empowered) for `BaseDamagePerHit + 0.75 AD` each, stunning 0.75 s (1.5 s). The attack's own AD hit stands for the first strike (it crits as the attack does); the kit adds the rest as one packet. The empowered version first destroys the target's shields unless it is a monster. Not consumed against structures. Cooldown starts post-effect (hit, dodge or expiry). After the strikes Renekton is locked 0.525 s. | 16–8 s; 5–65 + 75% AD | CLIENT `BuffDuration`, `TotalADPerHit`, `StunDuration`, `EnragedStunDuration`; WIKI range, lockout, shields, structures, V14.4 monsters |
| REN.W2 | Fury: 5 per strike (10) plus 10 against a champion; the empowered version gives none. | +10 vs champions | CLIENT `BonusFuryVsChamps`; WIKI |
| REN.E1 | Slice: dashes a fixed 450 toward the cursor at `750 + move speed`; enemies within 50 of the path (plus their radius) take `BaseDamage + 0.9 bonus AD` physical once. Fury 2 per non-champion, 10 per champion, at most 30 per dash. Not while rooted or in the Q lockout. The cooldown starts at the Slice cast. | 16–10 s; 40–160 | CLIENT `castRangeDisplayOverride`, `DashSpeed`, `mLineWidth`, `cantCastWhileRooted`; WIKI `cdstart = on-cast` |
| REN.E2 | If Slice hit an enemy, Dice can be cast within 4 s of the dash's end (no cooldown check): the same dash and damage; empowered it adds `EnragedBaseDamage + 1.35 bonus AD` and `EnragedArmorShred` % armor reduction for 4 s. | +70–250; 25–35% | CLIENT `DiceTimer`, `EnragedArmorShred`, `ShredTimer`; WIKI |
| REN.R1 | After 0.25 s, for 15 s: `HealthGain` max health, +25 attack range, +20 Fury, then +5 Fury per second while below 75. Every 0.5 s enemies within 375 (center to edge) take `(AuraDamagePerSecond + 0.1 AP + 0.1 bonus AD) × 0.5` magic damage. Ends on death; the cooldown starts at cast. | 120/100/80 s; 300/500/700; 30–120 per tick | CLIENT `TotalDamagePerSecond`, `TickRate`, `FuryOnCast`, `FuryPerSecond`; WIKI radius, 75 cap, range |

### Renekton approximations

1. **W strikes.** The strikes are one attack hit plus one kit packet, so item and rune on-hit effects fire once,
   not per strike. The 0.2 s empowered-attack cast time (wiki "estimated") is not modelled; the attack keeps its stat
   windup. Casting R does not end the post-strike lockout.
2. **Slice and Dice geometry.** Kit dashes ignore terrain like Jax Q, so a dash into a wall ends inside it and the
   world ejects him. The hit width (client `mLineWidth` 50 measured from the path) is INFERRED-M. The Dice window is
   measured from the end of the Slice dash (wiki: "within the next 4 seconds", INFERRED-M).
3. **Fury at the edges.** Low-health ×1.5 multiplies each source's amount after its per-cast cap. Dominus size is not
   in the hitbox.

## Approximations that remain

1. **Garen passive inputs.** The kit sees only damage packets. Zero-damage enemy abilities and summoner spells
   without damage, such as Exhaust, do not disable Perseverance. Epic monsters cannot disable it either,
   because the world has no epic monster type. Both are irrelevant to the current lane world.
2. **Garen Q lunge.** The lunge is modelled as +50 range against champions. Garen's position does not move.
   If the world does not supply `attack_target_kind`, every target gets the +50.
3. **Garen Q spell shield.** A spell shield should negate only the silence. The bonus packet carries
   `TAG_ACTIVE_SPELL`, so the world's spell shield also blocks the bonus damage. Changing this is a world
   rule.
4. **Garen E spins.** At most one spin fires per sim tick. This matters only above about 90 spins.
5. **Jax Q dash speed** (1400) and **Jax R resists fixed at the swing** are INFERRED (see the table).
6. **Jax E AoE reduction** applies to every AoE packet. The client limits it to champion sources, but only
   champions emit AoE packets in this world.
7. **Garen R reveal** (1 s true sight on the target) is not modelled, because vision is world-owned.

## What the world must supply or consume

These are optional fields with `None` defaults. Each kit falls back to the behaviour noted above.

| Field | Shape | Meaning |
|---|---|---|
| `KitCtx.attack_target_kind` | (C,) int32 | Kind of the unit the champion is attacking or winding up on. Gates the Garen Q lunge to champions. |
| `KitCtx.rooted` | (C,) bool | The champion is rooted. Blocks Jax Q. |
| `KitOut.attack_target` | (C,) int32, -1 none | Issue a basic-attack order on this unit (Jax Q landing on an enemy champion). |
| `KitAttackMods.windup` | (C,) s, 0 = stat windup | Windup of the next attack (Garen Q: `0.2 T`). |
| `KitAttackMods.period` | (C,) s, 0 = 1/AS | Total attack time of the next attack (Garen Q: `T`). The kit already enforces the recovery with `cannot_attack`. |
| `KitAttackMods.uncancellable` | (C,) bool | The next attack's windup cannot be cancelled (Garen Q, Jax W, the Jax R passive proc). |
| `KitAttackMods.extra_range` | (C,) | Already in the contract, but `world.tick` does not read it. Garen Q and Jax W each give +50. |
| `ghosted(state, kctx)` | (C,) bool | Garen E ghosting, to be ORed into the collision `ghost` mask. |
| `KitCtx.tenacity` | (C,) | Holders' tenacity (stats and kit sources, Garen W). Times the Nasus W steps. |
| `POINT_TARGET_RANGE` | per slot | Walk-in range of a location-targeted spell (Nasus E 650); the world queues the cast and walks to the point. |
| `debuffs(...).attack_speed_cripple` | (N,) | Kit attack-speed cripple (Nasus W), applied to the composed champion attack speed. |
| `KitCtx.move_speed` | (C,) | Holders' move speed after slows (Renekton E dash speed). |
| `KitOut.strip_shields` | (C, N) bool | Destroy the unit's shields before this tick's damage (Renekton empowered W). |
| `KitOut.cast_lockout` | (C,) | Now honoured from any hook, not only on a cast start (Renekton W post-strike lockout). |
| `resource(state, champion_id)` | (C,) | A kit's own resource bar (Renekton Fury), NaN where the holder is another kit. |

Per-champion record values the world reads from `ChampionBase`: `gameplay_radius`
(`overrideGameplayCollisionRadius`, default 65; Nasus 80), `pathing_radius` (`pathfindingCollisionRadius`; Nasus 50)
and `acquisition_range` (Nasus 350, Renekton 600).
