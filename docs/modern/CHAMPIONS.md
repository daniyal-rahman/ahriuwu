# CHAMPIONS.md — Garen, Jax, Nasus, Renekton, Darius, Malphite, Aatrox and Fiora, 26.19 (client 16.19.8230722)

**Status (2026-10-12).** Implemented in `lanerl_jax/modern/champions/` (`garen.py`, `jax.py`, `nasus.py`, contract
in `core.py`; `renekton.py`, `darius.py`, `malphite.py`, `aatrox.py`, `fiora.py`). Tests: `lanerl_jax/modern/tests/test_champions.py`, and
`test_<name>.py` / `test_<name>_world.py` for Nasus, Renekton, Darius, Malphite, Aatrox and Fiora.
Pinned records are regenerated from the client bins by `python -m lanerl_jax.modern.data.build_champions`.

## Sources

- **CLIENT**: the pinned BIN snapshots `lanerl_jax/modern/data/26.19/champions/<name>.json` (spell data values,
  cooldowns, mana, record radii and acquisition range), and the full CDragon 16.19 records in
  `/mnt/nfs/shared/modern-world-map-research/cdragon-16.19/<name>.bin.json` (Nasus, Malphite, Renekton, Darius and
  Tryndamere fetched 2026-10-12; Aatrox, Fiora, Yone, Ambessa, Camille, Jayce, K'Sante and Ornn 2026-10-13; checksums in
  `SHA256SUMS`): spell calculations,
  `mOverrideAttackTime`, `mRollForCriticalHit`, cast times, spell tags, plus the 16.19 string table tooltips.
  Calculation `mStat` numbers were checked against the item calculations: 2 is AD, 4 is attack speed, 6 is MR,
  7 is move speed, 8 is crit chance, 9 is crit damage and 12 is health.
- **WIKI**: `Template:Data Garen/*` and `Template:Data Jax/*` ability data and the patch history pages,
  fetched raw on 2026-10-02; `Template:Data {Nasus,Renekton,Darius,Malphite}/*` and their patch histories cached raw in
  `cdragon-16.19/wiki-2026-10-12-champions/`; the Aatrox, Fiora, Yone, Ambessa, Camille, Jayce, K'Sante and Ornn templates,
  patch histories and hitbox diagrams in `cdragon-16.19/wiki-2026-10-13-champions/`.
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
| NAS.R1 | After a 0.2 s cast time, for 15 s: `BonusHealth` max health, `InitialResistGain` armor and MR, 50 attack range (adds to Q's 50), `SizeIncreasePercent` size (gameplay radius 80 → 104/108/112), Siphoning Strike's base cooldown halved. Ends on death. The cooldown starts at cast. | 100 mana; 120/100/80 s; 300/450/600, 40/55/70 | CLIENT, PATCH V13.12 |
| NAS.R2 | Every 0.5 s (30 ticks) enemies within `max(400, 320 + gameplay radius)` (424/428/432 at base size, center to edge) take `(AOEDamagePercent + 1% per 100 AP) × 0.5` of their max health as magic damage, capped at 120 per tick (240 per second) against every unit type. | 3/4/5% per s | CLIENT `DamageCalc`, `TickRate`, `MaxDamageCap`, `SizeIncreasePercent`; WIKI radius and cap |

### Nasus limitations

Every known difference from the live game. Values marked INFERRED have no client, wiki or patch source; the rest are
modelling shortcuts with the exact rule stated.

1. **Cost and cooldown timing.** W, E and R pay mana and start their cooldown at the key press. The rule (WIKI Cast
   time) is after the cast time (0.25 / 0.25 / 0.2 s), and nothing is paid if Nasus dies during it.
2. **Wither cancel conditions.** The cast always completes once started. Live: it cancels without cost if the target
   dies, becomes untargetable, leaves sight or gets "too far away" during the 0.25 s cast (that distance is unknown).
3. **Spell shields.** Wither deals no damage, so a spell shield (Annul) neither blocks nor pops on it; live, it
   blocks Wither. The same gap lets kit CC from a blocked damage packet through (shared, see below).
4. **Wither details.** Its cripple keeps running if the slow is cleansed (Cleanse, Garen Q); live, both are one
   debuff. The target's attack windup is not interrupted on application (live, a listed bug does). Tenacity from
   summoner Cleanse is not in `KitCtx.tenacity`.
5. **Siphoning Strike stacks.** A stack is granted if the Q target dies to any Nasus damage on the tick the Q attack
   lands; live, the Q attack itself must kill. Turret kills give no stacks (the takedown matrix has no structure
   columns) and Q is not spent on wards (ward hits never reach the kit); live, both give stacks.
6. **Siphoning Strike crit.** On a crit the flat bonus is multiplied by the holder's crit damage, not the attack's
   actual multiplier (differs only with Infinity Edge-style crit scaling).
7. **Siphoning Strike re-press and Fury of the Sands cooldown.** Q cannot be pressed again while empowered, and the
   halved cooldown is fixed when the cooldown starts (unverified either way).
8. **Spirit Fire.** One fire per Nasus (a second cast replaces the first; only possible above ~118 ability haste).
   The 200-radius sight (wiki "2.5 s, estimated") is not modelled. Radius 380 is the client `castRadius`; the wiki
   lists 400. The first hit at cast + 0.25 + 0.25 s and the first tick 1 s after it are INFERRED from the wiki text.
9. **Fury of the Sands storm timing.** The first storm tick 0.5 s after the cast time is INFERRED. (Size is modelled:
   the gameplay radius grows 80 → 104 / 108 / 112, WIKI Unit size, and the storm reaches max(400, 320 + gameplay
   radius) center to edge.)

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

### Renekton limitations

1. **W strikes.** The strikes are one attack hit plus one kit packet, so item and rune on-hit effects fire once,
   not per strike. The 0.2 s empowered-attack cast time (wiki "estimated") is not modelled; the attack keeps its stat
   windup. Casting R does not end the post-strike lockout.
2. **Slice and Dice geometry.** Dashes follow the wiki terrain rule (cross a wall only to a walkable end point,
   else stop short). The hit width (client `mLineWidth` 50 measured from the path) is INFERRED-M. The Dice window is
   measured from the end of the Slice dash (wiki: "within the next 4 seconds", INFERRED-M).
3. **Fury at the edges.** Low-health ×1.5 multiplies each source's amount after its per-cast cap (order INFERRED).
   Fury resets to 0 on death (INFERRED). (Dominus's +20% size is modelled: gameplay radius 80 → 96.)
4. **Cost and cooldown timing.** Dominus starts its cooldown at the key press; live, after its 0.25 s cast time
   (WIKI notes it is not refunded on death during the cast).
5. **Spell shields.** The W stun and other kit CC are not blocked by a spell shield (shared, see below); the empowered
   W's shield strip ignores spell shields, which matches the wiki.

## Darius (122)

Rift patches since 26.1: 26.4 Hemorrhage monster damage ×2.5 (client `MonsterMod`). Gameplay radius is the default 65
(V14.9); pathing radius 25.77.

| ID | Rule | Value (ranks 1–5 / 1–3) | Evidence |
|---|---|---|---|
| DAR.P1 | Damaging basic attacks and abilities (Q blade, W, R after its damage) apply a Hemorrhage stack for 5 s (refreshed, max 5) to non-structures. Each stack deals `13 → 30` (linear in level) `+ 0.3 bonus AD` physical over the 5 s, a quarter every 1.25 s from the first stack (4 ticks per application); ×2.5 against monsters. Dead units lose their stacks. | 5 stacks | CLIENT `BleedDamagePerStack`, `MaxStacks`, `BleedDuration`, `MonsterMod`; WIKI 1.25 s ticks (client `SecondsPerTick` 1.26) |
| DAR.P2 | Five stacks on an enemy champion, or an R execution, grant Noxian Might for 5 s: `30 + 5/level` to 10, `+10/level` 11–13, `+25/level` from 14 bonus AD, and every application sets five stacks. | 30–230 AD | CLIENT `NoxianMightBonusAD`; WIKI |
| DAR.Q1 | Ghosted for 1 s; after 0.75 s (Darius may walk but not attack, cast E or R) the axe hits enemies within 460 (center to edge) for `BaseDamage + TotalADRatio % AD` physical; centres within 240 take 35% and get no stack. Heals 17% of missing health per champion or large monster hit by the blade, at most 3. Cooldown starts at the swing. | 25–45 mana; 9–5 s; 50–170 + 100–140% AD | CLIENT `TotalADRatio`, `InnerCircleDamagePercent`, `MissingHealthHeal`; WIKI radii, heft, `cdstart = post-effect` |
| DAR.W1 | Attack reset; the next attack within 4 s gains 25 range and an uncancellable windup, deals `(ADRatio − 1) × AD` bonus physical damage (× crit damage when the attack crits; basic and spell damage) and slows 90% for 1 s. Cooldown post-effect. | 40 mana; 5 s; 40–60% AD | CLIENT `ADRatio`, `SlowPercent`, `SlowDuration`; WIKI |
| DAR.W2 | If the empowered attack kills its target, the mana is refunded and W restarts at half its cooldown. | 50% | CLIENT `PercentCDRefund`; WIKI |
| DAR.E1 | Passive: `PassivePercentArmorPen` % armor penetration once learned. | 20–40% | CLIENT; PATCH V14.2 |
| DAR.E2 | After a 0.25 s cast time, every enemy within 535 (center to edge) and 25° of the cast direction (plus its angular radius) is dragged at 1784 units/s to 45 from Darius's centre, then rebounds over 0.5 s to 230 from it, airborne throughout, then slowed 40% for 1 s. Darius cannot move or cast Q/R for 0.4 s after the cast time, then attacks the nearest pulled champion (else unit). Cooldown at cast. | 70–30 mana; 26–16 s | CLIENT `castConeAngle`, `SlowPercent`; WIKI radius, lockout, follow-up attack; pull speed, stop and rebound measured on 14 pulls of Garen in the 16.9 replay corpus (20 fps positions; median 1784 u/s, arrival 49, rebound end 230 after 0.50 s, range 200-283) |
| DAR.R1 | Enemy champion within 475 (center to edge); not while rooted. Darius leaps to it over the 0.3667 s cast time, then deals `(RBaseDamage + 0.75 bonus AD) × (1 + 0.2 × stacks)` true damage and applies a stack. Cooldown at cast. | 100/100/0 mana; 120/100/80 s; 125/250/375 | CLIENT `Damage`, `RDamagePercentPerHemoStack`, `castRange`, `mCastTime`, `cantCastWhileRooted` |
| DAR.R2 | If the target dies within 0.15 s, Darius gains Noxian Might; at ranks 1–2 he may recast within 20 s at no cost and without a cooldown check (the first cast's cooldown keeps running), at rank 3 the cooldown resets. Q and E are blocked for 0.25 s after the cast time. | 20 s | CLIENT `RRecastDuration`; WIKI |

### Darius limitations

1. **Apprehend motion.** Pull and rebound use the replay medians (speed 1784, stop 45, rebound to 230 in 0.5 s);
   individual pulls varied (rebound ends 200-283 from Darius), and they were measured on champions only (minions
   use the same numbers). A short pause at arrival seen in 2 of 14 pulls is not modelled. Pulls ignore terrain
   and collision. Darius's 0.4 s post-cast lock also blocks attacks.
2. **Noxian Guillotine.** Fear and slow of nearby minions and monsters after an execution are not modelled (the world
   has no fear). The leap is a dash to the target over the cast time; it does not stop at thin walls.
3. **Hemorrhage reveal** (sight of a target the bleed will kill) and Apprehend's sight are not modelled.
4. **Hemorrhage ticks** every 1.25 s (wiki); the client's `SecondsPerTick` is 1.26.
5. **Cost and cooldown timing.** Apprehend and Noxian Guillotine start their cooldown and pay mana at the key press;
   live, after their cast times (0.25 / 0.3667 s).
6. **Spell shields.** The pull, knock-up and slows are not blocked by a spell shield (shared, see below).

## Malphite (54)

| ID | Rule | Value (ranks 1–5 / 1–3) | Evidence |
|---|---|---|---|
| MAL.P0 | Shard of the Monolith: size grows by 8% per 100 total armor, at most 35%; Thunderclap's bonus armor does not count. | 80 → up to 108 | WIKI Granite Shield (script-only in the client) |
| MAL.P1 | Granite Shield: a shield of 10% max health (no expiry) whenever Malphite has lost no health for 8 s (7 from level 7, 6 from 13) and the shield is below full. It refills the same shield instance (`D.SOURCE_KIT`): a new grant replaces the old one. Damage the shield absorbs does not reset the delay. | 10% max HP | CLIENT `TotalShield`, `PassiveCooldown`; WIKI notes |
| MAL.Q1 | Enemy unit (not structures) within 625 (center to edge); 0.25 s cast time, then a 1200-speed shard: `BaseDamage + 0.6 AP` magic and a `SpeedSteal` % slow for 3 s; Malphite gains the target's current move speed × that slow as flat move speed for 3 s. Fizzles if the target dies. | 70–90 mana; 8 s; 70–270; 20–40% | CLIENT `castRange`, `missileSpeed`, `SpeedSteal`, `SlowDuration`; WIKI cast time, speed steal |
| MAL.W1 | Passive: `BonusArmorPassive` % armor (STAT.40 total multiplier), ×3 while Granite Shield holds. | 10–30% | CLIENT `BonusArmorPassive`, `BonusArmorPassiveMultiplier` |
| MAL.W2 | Active: attack reset; the next attack within 6 s gains 50 range and an uncancellable windup and deals `ThunderclapBaseDamage + 0.2 AP + 0.15 armor` physical on-hit. For 5 s every landed attack makes an aftershock: `ThunderclapSplashDamage + 0.3 AP + 0.15 armor` physical to enemies in a cone from Malphite toward the target, 400 long (the target takes it only if inside). ×1.8 against monsters. Cooldown at cast. | 30–50 mana; 10–8 s; 30–70; 15–55 | CLIENT calculations, `ThunderClapSplashRange`, `ThunderclapBuffDuration`, `MonsterDamageMod`; WIKI window, range, cone origin |
| MAL.E1 | After its 0.2419 s cast time: `BaseDamage + 0.4 armor + 0.6 AP` magic to enemies within 400 (center to edge) and an `ASReduction` % attack-speed cripple for 3 s. | 50–70 mana; 7 s; 60–200; 30–50% | CLIENT `mCastTime`, `castRadius`, `EDamageCalc`, `ASReduction` |
| MAL.R1 | Dashes to the cursor, at most 1000 (cast at max range), at `1500 + move speed`; CC does not stop it. On arrival: `BaseDamage + 0.9 AP` magic to enemies within 325 of the end point and a 1.5 s knock-up. Cooldown at cast. | 100 mana; 130/115/100 s; 200/300/400 | CLIENT `castRange`, `KnockupDuration`, `TotalDamage`; WIKI speed, radius (client `castRadius` 270) |

### Malphite limitations

1. **Thunderclap cone angle.** Neither the client data nor the wiki gives it; the kit uses a 45° half-angle
   (INFERRED-L). Aftershocks fire once per landed attack, not as a separate on-hit item proc.
2. **Seismic Shard speed steal** reads the target's current move speed (after its other slows and the soft caps),
   not the raw value the wiki describes.
3. **Ground Slam cripple** takes the strongest cripple on the target instead of stacking multiplicatively with
   others (`combine_debuffs`). **Shard of the Monolith** size ignores armor reduction (Black Cleaver lowers it live).
4. **Unstoppable Force** damage radius 325 is the wiki value; the client `castRadius` is 270. It lands where the
   dash actually stops (wiki terrain rule).
5. **Cost and cooldown timing.** Seismic Shard and Ground Slam pay at the key press; live, after their cast times
   (0.25 / 0.2419 s).
6. **Spell shields.** The slows, cripple and knock-up are not blocked by a spell shield (shared, see below).

## Aatrox (266)

Rift patches since 26.1 (client values include them): 26.2 (passive 4–10% max health; Q +25 vs monsters), 26.12 (Q
sweetspot +75%), 26.19 (W cooldown 18–12 s; E heal +1.3% per 100 bonus health). Gameplay radius 65, pathing 35,
acquisition 475; no mana (the bar shows World Ender's remaining duration).

| ID | Rule | Value (ranks 1–5 / 1–3) | Evidence |
|---|---|---|---|
| AAT.P1 | Deathbringer Stance: while ready, the next basic attack gains 50 range and deals `PDamage` (4% → 10% by level) of the target's max health as bonus magic damage, capped at 100 → 320 (by level) against monsters; not against structures or wards (not consumed). The bonus is a proc packet with life steal; Aatrox heals 100% of its post-mitigation damage, 25% against minions. Dodged attacks deal nothing. Cooldown 22 → 10 s by level, static (no haste). | 4–10% | CLIENT `PDamage`, `PCooldown`, `MonsterDamageCap`, `PBonusAARange`, `PHealingRatio`, `PHealingMinionMod`; WIKI static cooldown, structures/wards, proc + ApplyLifesteal |
| AAT.P2 | Each basic attack landing on, and each ability hit on, an enemy champion or large monster reduces the current passive cooldown by 2 s, once per attack or cast; 4 s if a Darkin Blade sweetspot hits one. Infernal Chains' first hit and its pull each count. | 2 / 4 s | CLIENT `PChargeRate`; tooltip; WIKI |
| AAT.Q1 | Three casts. The first needs the cooldown; each recast is allowed 1 s (static) after the previous cast and within `QExtensionTime` of it; after the third cast or the window the sequence resets. The cooldown starts on the first cast and restarts on each recast. | 14/12/10/8/6 s | CLIENT `cooldownTime`, `QExtensionTime`; WIKI cdstart, 1 s static |
| AAT.Q2 | 0.6 s cast time: Aatrox cannot walk or attack, Umbral Dash stays castable, only death interrupts. The strike lands at the end, from Aatrox's position then, in the direction fixed at cast. | 0.6 s | CLIENT `AatroxQWrapperCast.mCastTime`; WIKI cast-time notes |
| AAT.Q3 | Areas along the cast direction from Aatrox's centre. Cast 1: rectangle 0–625 long, 180 wide (edge range); sweetspot where the unit's centre is 450–755 along and within 186.5 of the line. Cast 2: trapezoid from 100 behind to 475 ahead, 300 → 500 wide (edge range); sweetspot centre 300–675 along, half-width 301 → 342. Cast 3: circle of 300 around the point 200 ahead (centre range); sweetspot within 180 of it. A sweetspot also needs the main area. | see rule | WIKI dimensions; sweetspot boxes measured off `File:Aatrox_hitbox_areas.png` (1 px = 1 unit) |
| AAT.Q4 | Damage `QBaseDamage + QTotalADRatio × AD` physical, × (1 + 0.25 × cast index), × 1.75 in the sweetspot; × 55% against minions (+5% at levels 6, 11 and 16); +25 against monsters before the cast and sweetspot multipliers. The sweetspot knocks up for 0.25 s (0.5 s against monsters). | 10–70 + 60–90% AD | CLIENT `QDamage`, `QRampBonus`, `QSweetSpotBonus`, `QMinionDamage`, `QMonsterBonus`, `QKnockupDuration`; WIKI |
| AAT.W1 | 0.25 s cast time (Aatrox held in place, E castable), then a chain from Aatrox's position at 1800 units/s along the direction fixed at cast. It stops on the first enemy (minion, monster or champion) whose hitbox comes within 80 of its line, up to 765 edge range (≈ 825 centre range). `WBaseDamage + 0.4 AD` physical, ×2 against minions, and a 1.5 s slow. Cooldown at cast. | 18/16.5/15/13.5/12 s; 30–70; 25–35% | CLIENT `mCastTime`, `mMissileSpec` speed/width, `WDamage`, `WSlowPercentage`; WIKI ranges, minion ×2, PATCH 26.19 |
| AAT.W2 | A champion or large monster hit is tethered to the point where it was hit for 1.5 s. If it is still within 650 of that point at the end, it takes the damage again and is pulled to the point over 0.25 s, airborne for 0.5 s in total. | 650; 0.25 s; 0.5 s | WIKI tether, second hit; zone, pull and airborne time measured on 19 clean pulls in the 16.9 replay corpus (`/mnt/nfs/shared/champ-expansion/aatrox-measurements/clean-pulls.txt`; see limitations 1–2) |
| AAT.E1 | Passive: heals `ESpellVamp` % + 0.013% per bonus health of the post-mitigation, non-persistent damage Aatrox deals to enemy champions. | 16% + 1.3% per 100 bonus HP | CLIENT `TotalEVamp`; PATCH 26.19 |
| AAT.E2 | Dashes toward the cursor, 75–300 (shorter clicks dash 75, longer ones 300) at 800 units/s; no cast time; attack reset; not while rooted; castable during the Q, W and R cast times without cancelling them, and they during it. Cooldown at cast. | 9/8/7/6/5 s | CLIENT `EMinRange`, `EMaxRange`, `EDashSpeed`, `cantCastWhileRooted`, `Trait_AttackReset`; WIKI |
| AAT.R1 | 0.25 s cast time (Aatrox may walk, not attack); cooldown at cast. At the end: fears enemy minions and monsters within 600 (center to edge) for 3 s, and for 10 s grants `RTotalADAmp` × total AD bonus AD, `RMovementSpeedBonus` bonus move speed that loses 10% of its current value every 0.25 s, `RHealingAmp` increased self-healing, ghosting and 5% size. | 120/100/80 s; 20/30/40% AD; 60/80/100% MS; 50/75/100% healing | CLIENT `RDuration`, `RFearRadius`, `RMinionFearDuration`, `RTotalADAmp`, `RMovementSpeedBonus`, `RHealingAmp`; WIKI decay, ghost, size |
| AAT.R2 | Each champion takedown during World Ender adds 5 s (at most 10 s remaining) and restarts the move-speed decay. | +5 s | CLIENT `RExtension`; WIKI |

### Aatrox limitations

1. **Infernal Chains zone.** No source gives its size. Modelled as a circle of 650 around the hit point. The 16.9
   replay corpus (5 games with Aatrox, 20 fps positions) has 19 clean pulls; targets were still pulled from up to 573
   units away, so the zone is at least that large. The wiki's escape note ("at least 628 movement speed" to escape
   straight away from Aatrox) puts it at about 612–690, depending on the slow. The wiki describes the area as
   oriented toward Aatrox, so the live shape may not be a circle.
2. **Infernal Chains pull.** In the replays every pull took about 0.25 s (0.22–0.33 at 20 fps) whatever its length,
   and the target stayed in place about 0.25 s after arriving. The model pulls over 0.25 s with a 0.5 s knock-up.
   Live, the pull ends 3–67 units from the hit point, slightly toward Aatrox; the model ends exactly on it.
3. **Darkin Blade shapes.** The sweetspot boxes are measured off the wiki's hitbox diagram. Cast 2's edge-range test
   grows the trapezoid by the unit's radius, which slightly widens the corners. The bug that briefly stops Aatrox
   from starting an attack after a cast is not modelled.
4. **Deathbringer Stance timing.** The empowered attack is decided at launch, not when the windup starts, so an
   attack that was already winding up when the passive came off cooldown is empowered here but not live. The
   empowered attack's own hit gives no 2 s refund (INFERRED-M).
5. **Umbral Dash.** The extended dash across walls (up to 500 units, faster) and the re-issued move order are not
   modelled; dashes follow the world's wall rule (shared, `mechanics.dash_end`). The wiki notes the measured
   speed is slightly above 800.
6. **World Ender.** The world has no fear, so minions and monsters are stunned for 3 s instead. Live, they flee
   while being slowed by up to 99%, so they barely move and don't attack either way. The AD bonus reads AD without
   dynamic stats (Conqueror stacks, for example). The healing amplification uses the world's `incoming_heal`,
   which also amplifies shields Aatrox receives.
7. **Cost and cooldown timing.** W and R start their cooldowns at the key press; live, after their 0.25 s cast
   times (shared).
8. **Spell shields.** The Q knock-up and the W slow, tether and pull are not blocked by a spell shield, and the
   pull's damage is blocked (live, a spell shield blocks only the chain's first hit; shared, see below).
9. **Heals** read the main damage pass. Follow-up item procs are not counted toward Umbral Dash's healing.

## Fiora (114)

Rift patches since 26.1 (client values include them): 26.19 (health growth 105; Victory Zone 600 radius, 80–130
+ 70% bonus AD healing per second). Gameplay radius 65, pathing 35, acquisition 600; windup 0.2 / 1.45 (13.8%).
Where the wiki text and the client data disagree, the client wins: Riposte slows 50% and cripples 25% (V14.4 "cripple
reduced to 25%"; the wiki template still says 25% for both), the vital speed decays over 2 s (`MovementSpeedDuration`;
wiki 1.85 s), Bladework's window is 3 s (`BuffDuration`; wiki 4 s).

| ID | Rule | Value (ranks 1–5 / 1–3) | Evidence |
|---|---|---|---|
| FIO.P1 | Duelist's Dance: on each enemy champion within 1200 (centre range) Fiora identifies a vital: a direction (north, east, south or west of the target, a 90° arc). It becomes targetable 1.75 s later and lasts 13.25 s more while Fiora stays within 1350 (edge range); then, or when struck, a new one is identified. Directions alternate between the south/west and north/east pairs, starting with south/west, each side 50% (`KitCtx.rand`). | 1.75 + 13.25 s | WIKI notes, tooltip ("after 15 seconds or after striking one") |
| FIO.P2 | Any damage Fiora deals (not reactive) to the target while she stands inside an open vital's arc strikes it: `3% + 4% per 100 bonus AD` of the target's max health as true damage (a proc, one tick later), Fiora heals 35 → 100 (by level) and gains Grand Challenge's `PercentMS` move speed, decaying to 0 over 2 s. | 20/30/40/50% MS (R rank 0–3) | CLIENT `PassiveDamageTotal`, `PassiveHealAmount`, `MovementSpeedDuration`, `PercentMS` |
| FIO.Q1 | Lunge: not while rooted; dashes toward the cursor, 50–400, at 1100 + 350% bonus move speed (at most 1600), or + 300% of a negative bonus (at least 850), fixed at cast. All her abilities are castable during the dash. | 20 mana; 13/11.25/9.5/7.75/6 s | CLIENT `cooldownTime`, `mana`, `cantCastWhileRooted`; WIKI range, speed |
| FIO.Q2 | At the end of the dash it stabs one unit within 420 (centre range), in this order: a challenged champion with a vital facing Fiora, any champion with one, a challenged champion, the lowest-health champion under 20%, the nearest unit within 252 that the stab kills, the nearest champion, a minion it kills, any minion or monster, a structure. `Damage + bADRatio × bonus AD` physical, tagged basic attack and spell (life steal, dodgeable, spell-shieldable) and applying on-hit effects (not on-attack). The cooldown starts when the stab resolves, halved if it struck an enemy. | 70–110 + 90–110% bonus AD | CLIENT `TotalDamage`, `CDRefundPercent`, `castRadiusSecondary`; WIKI priority, damage tags |
| FIO.W1 | Riposte: 0.75 s in which Fiora cannot act and takes no damage from non-turret sources, and CC, slows and forced movement on her are dropped. A hostile stun, root or knock-up dropped this way arms the stun. | 50 mana; 24/22/20/18/16 s post-effect | CLIENT `ParryDuration`; WIKI notes, cdstart |
| FIO.W2 | After 0.5 s the stab leaves from Fiora's position along the direction fixed at cast at 3200 units/s, 750 long, 70 half-width: `BaseDamage + 1.0 AP` magic to every enemy it passes until it reaches an enemy champion, which is slowed 50% and crippled 25% for 2 s, or stunned 2 s if the stun is armed. | 110–270 | CLIENT `StabDamage`, `MSSlowPercent`, `AttackSlowPercent`, `CCDuration`, missile speed/width; WIKI poise time |
| FIO.E1 | Bladework: attack reset; the next two attacks within 3 s gain 25 range and `ASPercent` bonus attack speed. The first slows 30% for 1 s (even if dodged) and cannot crit; the second always crits for `AttackTwoPercentTAD` + the holder's bonus critical damage (an item's forced crit wins when higher). Cooldown after the second attack or the window. | 40 mana; 11/10/9/8/7 s; 50–90% AS; 160–200% | CLIENT `ASPercent`, `AttackTwoPercentTAD`, `SlowPercent`, `BuffDuration`; WIKI notes, cdstart |
| FIO.R1 | Grand Challenge: an enemy champion within 500 (centre range), not one already challenged. Cooldown at cast. After 0.5 s all four of its vitals are open for 8 s regardless of range (no passive vital on it meanwhile); each strikes once as FIO.P2. While within 500 of it (edge range) Fiora has the full `PercentMS` move speed. | 100 mana; 110/90/70 s | CLIENT `MarkDuration`, `MSRingRadius`, `castRange`; WIKI |
| FIO.R2 | Striking all four, or the target dying with at least one struck before the 8 s end, makes a Victory Zone at the target's position for 5 s that heals Fiora `HealPerSecond + 0.7 × bonus AD` per second, every 0.25 s, while she is within 600. | 80/105/130 per s | CLIENT `HealPerSecond`, `Ratio`, `HealRingRadius`, `HealDuration`; PATCH V25.14 (0.25 s ticks), 26.19 |

### Fiora limitations

1. **Vital identification** ignores sight: vitals appear whenever the champion is within 1200 (the wiki's
   out-of-vision 250 estimate is not modelled). The vital's true damage lands one tick (33 ms) after the strike.
2. **What strikes a vital.** Damage that is blocked or absorbed to zero (a spell shield, a parry) strikes
   nothing here. Live, a block still strikes the vital (only spell shields and dodges stop it). The direction is
   Fiora's position at the moment of the damage, including for Riposte's missile.
3. **Lunge.** The "nearest champion in a rectangle behind Fiora" priority and stabbing wards are not modelled;
   Flash stays usable during the dash. The cooldown starts at the stab, 0.36 s after the cast at base speed (0.47 s when slowed); live it
   starts at cast and the stab refunds half.
4. **Riposte.** Negated damage does not consume Fiora's shields, and the attacker's life steal on it does not
   heal (live, both still happen). The immunity drops CC, slows and forced movement, but not Exhaust's damage
   reduction, Ignite's Grievous Wounds, or kit cripples and armor shreds. A parry in the last 0.25 s after the
   stab already struck does not convert its slow into a stun. Fiora stays selectable throughout. The stab's
   range is 750 (the cast indicator). The missile record and the wiki give 900.
5. **Bladework** uses the client's 3 s window (wiki: 4 s). The world ignores crits on structure hits for every
   champion, so Bladework's second attack does not crit a turret (live it does).
6. **Grand Challenge.** Recasting on the same target is simply refused (live: a 1 s cooldown). The Victory Zone
   heals only Fiora, since the lane world has one champion per team.
7. **Spell shields.** The Riposte slow or stun is not blocked by a spell shield (shared, see below).

## Approximations that remain

Shared by every kit:

- **Cost and cooldown timing.** Spells with a cast time pay mana and start their cooldown at the key press; the
  live rule is after the cast time (Garen R 0.435 s, Jax R 0.25 s and the pending kits above).
- **Spell shields and CC.** The world's spell shield blocks enemy ActiveSpell *damage packets* only. CC emitted by a
  kit is applied even when the same cast's damage was blocked, and CC-only casts never pop a spell shield.

Per kit (Garen, Jax):

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
| `displacement(state, kctx, n)` | (N,) × 4 | Forced movement `(active, to_x, to_y, speed)` the move phase applies, ghosted (Darius E). |
| `immobile(state, kctx)` | (C,) bool | The kit holds its champion in place without a world cast lock, so its other spells stay castable (Aatrox Q and W cast times, during which Umbral Dash works). |
| `KitCtx.rand` | (C,) | Uniform [0, 1) folded from the world key each tick (Fiora vital directions). |
| `KitDefense.parry` | (C,) bool | Non-turret damage packets to the holder are negated (`Defense.parry`; Fiora Riposte). |
| `cc_immune(state, kctx)` / `on_parry(state, kctx, parried)` | (C,) bool | The holder's CC, slows and newly started forced movement are dropped this tick (`ChampionLayer.disp_ignored` holds a dropped movement off to its end); `parried` reports a dropped hostile stun, root or knock-up. |
| `KitAttackMods.force_crit` / `crit_mult` | (C,) | The next attack crits for this multiplier (Fiora Bladework); an item's forced crit wins when higher. |
| `KitOut.on_hit_target` | (C,) int32 | The holder's ability applies on-hit effects to this unit: items and runes see a landed hit, not a launch (Fiora Lunge). |
| `size(state, kctx)` | (C,) | Bonus size; the stats phase sets the champion's gameplay radius to record × (1 + size) (wiki Unit size). |
| `KitCtx.dashing` | (C,) bool | A dash is moving the holder; point dashes land per `mechanics.dash_end` (wiki Dash: cross a wall only to a walkable end point, else stop short; Rocketbelt always stops). |
| `Shields.source` / `KitCtx.kit_shield` | (N, K) / (C,) | Shield source tag (`ShieldGrant.kind = SHIELD_* + 16 × SOURCE_*`): a sourced grant replaces the unit's earlier one; the remaining kit-sourced absorb is given to the kit (Malphite). |

Per-champion record values the world reads from `ChampionBase`: `gameplay_radius`
(`overrideGameplayCollisionRadius`, default 65; Nasus 80), `pathing_radius` (`pathfindingCollisionRadius`; Nasus 50)
and `acquisition_range` (Nasus 350, Renekton and Darius 600); Malphite 80 / 50 / 400; Aatrox 65 / 35 / 475; Fiora 65 / 35 / 600. Records that give the attack as `mAttackCastTime / mAttackTotalTime` (Aatrox, Fiora) use that as the windup percent (wiki Attack speed).
