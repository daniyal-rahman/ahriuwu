# CHAMPIONS.md — Garen and Jax, 26.19 (client 16.19.8230722)

**Status (2026-10-02).** Implemented in `lanerl_jax/modern/champions/` (`garen.py`, `jax.py`, contract in
`core.py`). Tests: `lanerl_jax/modern/tests/test_champions.py`.

## Sources

- **CLIENT**: the pinned BIN snapshots `lanerl_jax/modern/data/26.19/champions/{garen,jax}.json` (spell data values,
  cooldowns, mana), and the full CDragon 16.19 records in
  `/mnt/nfs/shared/modern-world-map-research/cdragon-16.19/{garen,jax}.bin.json`: spell calculations,
  `mOverrideAttackTime`, `mRollForCriticalHit`, cast times, spell tags, plus the 16.19 string table tooltips.
  Calculation `mStat` numbers were checked against the item calculations: 2 is AD, 4 is attack speed, 6 is MR,
  7 is move speed, 8 is crit chance, 9 is crit damage and 12 is health.
- **WIKI**: `Template:Data Garen/*` and `Template:Data Jax/*` ability data and the patch history pages,
  fetched raw on 2026-10-02.
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
