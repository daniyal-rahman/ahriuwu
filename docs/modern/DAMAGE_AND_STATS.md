# Shared combat math and modifier ordering, pinned to patch 26.19

**Scope.** This spec covers the global formulas and orderings that every combat system plugs into: stat composition and level growth, resistances and penetration, damage types, tags and modifiers, shields, healing, vamp, crits, attack timing, movement speed, ability haste, tenacity and slow resist, CC precedence, regeneration, max-HP changes, death and execute. It also defines the **hook taxonomy** (§2) that the ITEMS.md, RUNES.md, minion, tower and economy specs should reference by ID. It does **not** cover individual item, rune or champion values. When one is quoted, it is only there as a fixture or as evidence for a global rule.

**Patch pin.** League of Legends **26.19**, client build **16.19.8230722**, normal PC Summoner's Rift (CLASSIC game mode, Map11). League Classic, Arena, ARAM: Mayhem and Swiftplay sections of the patch notes were excluded line by line (see §0.3).

**Retrieved.** 2026-10-01 (wiki revisions as of 2026-09-30T15:54Z or earlier; see oldids).

**Status.** RESEARCH ONLY. No code, data or other docs were changed. The legacy C# (4.20) behaviour is explicitly **not** a reference.

## 0. Sources

### 0.1 Client data (priority 1)

Cache root: `/mnt/nfs/shared/modern-world-map-research/`

| File | What it pins | sha256 |
|---|---|---|
| `classic-constants.json` (client GameModeConstants, CLASSIC) | `gcd_AttackMinDelay=0.333`, `gcd_AttackMaxDelay=5.0`, `gcd_AttackDelay=1.6`, `gcd_AttackDelayCastPercent=0.3`, `gcd_PercentCooldownModMinimum=-0.4`, `gcd_AttackSpeedCatchupPercent=0.25`, `dr_*` unit-class damage ratios, `ov_/sv_/pv_` vamp ratios and tags, `ai_levelUp_healthGain*`, `sp_*` fountain regen, `aiExp_timeForKillCreditAfterDeath=10`, `ai_MaximumHPMaxPenalty=0.5` | `c77ca7c1d3c5c38fec3bb12a28e738f8607b01d1d09e198fe27eed34abd7ba8c` |
| `cdragon-16.19/globals.cdtb.bin.json` (https://raw.communitydragon.org/16.19/) | `GlobalPerLevelStatsFactor` table (levels 1..30); `DamageSourceSettings.damageTagDefinition` (14 tags); damage-modifier categories `["Vulnerable","Durable","Empower","Weaken"]`; `LiveFeatureToggles.OmnivampStat=true`; `UnitObjectTagsSettings` | `ee7c25bb…2b366b` |
| `cdragon-16.19/items.cdtb.bin.json` | Item stat field taxonomy (`mFlat*Mod`, `mPercent*Mod`, `PhysicalLethality`, `mFlatCritDamageMod`, `mPercentTenacityItemMod`, `mPercentSlowResistMod`, `mPercentHealingAmountMod`, `mPercentBaseHPRegenMod`, `mPercentMultiplicativeAttackSpeedMod`…); Grievous Wounds amount 0.40 on every source | `6880f35d…62726` |
| `cdragon-16.19/perks.cdtb.bin.json` + `cdragon-perks.json` | Stat shard values (adaptive 9 = 5.4 AD / 9 AP, AS 10%, AH 8, **MS 2.5%**, HP 65, scaling HP 10–180, **tenacity and slow resist 15%**) | `1427c70c…cd401b1` |
| `cdragon-16.19/garen.bin.json`, `jax.bin.json` (fetched this task) | `critDamageMultiplier: 2.0`; `attackSpeedModifiable`/`attackSpeedRatioModifiable`; `mAttackDelayCastOffsetPercent`; `mAttackDelayCastOffsetPercentAttackSpeedRatio` (Garen 0.5 = windup modifier) | `2c846ed3…66cc`, `3887a8c1…cf` |

### 0.2 Wiki (priority 3)

Raw wikitext is saved in `/mnt/nfs/shared/modern-world-map-research/wiki-damage-stats/` as `<Page>__<oldid>.wikitext`, with `revisions.json` and `SHA256SUMS`. Permalink: `https://wiki.leagueoflegends.com/en-us/index.php?oldid=<oldid>`.

| Page | oldid | Page | oldid |
|---|---|---|---|
| Damage | 4068095 | Damage modifier | 4050138 |
| Armor | 4070944 | Magic resistance | 4070945 |
| Armor penetration (Lethality) | 4035725 | Magic penetration | 4060414 |
| Health | 4035693 | Shield | 4021810 |
| Healing | 4041638 | Grievous Wounds | 3969410 |
| Vamp (Omnivamp) | 4058226 | Life steal | 4060425 |
| Template:Healing modifiers | 4021803 | Heal and shield power | 4021804 |
| Attack speed | 4035691 | Basic attack | 4052292 |
| Critical strike | 4059009 | Attack effects (on-hit) | 4050298 |
| Haste (Ability haste) | 4070414 | Haste/Reducing ability cooldowns | 4039444 |
| Cooldown | 4050188 | Movement speed | 4064468 |
| Slow resist | 4058780 | Tenacity | 4021927 |
| Crowd control | 4052259 | Types of Crowd Control | 4061028 |
| Champion statistic | 4069636 | Adaptive force | 4063874 |
| Health regeneration | 4061735 | Mana regeneration | 4050319 |
| True damage | 4052642 | Physical damage | 4052639 |
| Kill (execute) | 4053216 | Death | 4051174 |
| Spell shield | 3981674 | Invulnerability | 3990045 |
| Untargetability | 3965000 | Combat status | 4058480 |
| Range | 4058499 | Unit size | 4046606 |
| Projectile | 3999064 | Tick and updates | 3805043 |
| Minion | 4068797 | Turret | 4070072 |

### 0.3 Riot patch notes 26.1–26.19 (priority 2)

Saved HTML and text are in `/mnt/nfs/shared/modern-world-map-research/patch-notes-26.x/` (`SHA256SUMS` included). URLs: `https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-{1,2,3}-notes/` and `…/league-of-legends-patch-26-N-notes/` for N=4..19.

**Method.** Each note was split at its mode headings (`Classic`, `Arena`, `ARAM: Mayhem`, `Swiftplay`). Only the leading SR block was searched for systemic keywords (durability, healing, shields, AS, pen, tenacity, MS, AH, crit, vamp, regen, damage reduction).

**SR systemic findings:**

- **26.1:** Base critical strike damage is **200%** for every champion (from 175%). Infinity Edge crit damage changed from +40% to **+30%**. Ability crit scalings are rewritten as "X% of *bonus* crit damage". **Omnivamp** "now applies at 100% value, reduced to 33% value from pets, damage over time damage, and area effect spells against Minions". Level cap is 20 via the top role quest.
- **26.4:** Omnivamp **no longer applies to Smite and Ignite** damage.
- **26.9:** Minion wave durability changes (minion spec).
- **26.10:** Minion aggro trigger change (minion spec).
- **26.16, 26.19:** No SR global formula changes. 26.16 is ADC MR, jungle pets and support quest. 26.19 is Teleport cooldowns.
- **26.11:** Moonstone no longer "double dips" heal/shield power and Grievous Wounds. This is evidence that HSP and GW are both multiplicative factors on the heal (§7).
- No SR patch in 26.1–26.19 changed the armor or MR formula, lethality, the AS cap, the MS soft caps, the haste formula, the GW value, or tenacity stacking.

**Traps found (not SR):**

- 26.16/26.17 "Garen AD 55.5→60" and "Exhaust reduced true damage" bugfix are **League Classic**.
- 26.15 "damage dealt modifiers now apply to item and Augment damage" is **ARAM: Mayhem**.
- 26.19 "35% Tenacity shared with Mercury's Treads" is Classic. SR Mercury's Treads is **30%** in client data (`Items/3111.mPercentTenacityItemMod=0.3`).
- 26.2 Combo Breaker is Arena/Mayhem.

### 0.4 Prior project docs read

- `docs/MODERN_PATCH_DELTA.md` §11 (26.18 research). **Correction:** §11.2 says tenacity is "additive within group, multiplicative across". The 26.19 wiki says the opposite: **multiplicative within a group, additive across groups** (§10).
- `docs/JAX_FIDELITY_LEDGER.md` MODERN-001…012. MODERN-012 flags the flat-reduction pool conflict, which is still unresolved here (§4.3, U-03).
- `docs/PORT_AUDIT_COMBAT.md`. Legacy contrast: `DamageRatios` were dead code in LeagueSandbox. Whether Riot's engine enforces them is open (U-05).

### 0.5 Confidence tags

| Tag | Meaning |
|---|---|
| **[CLIENT]** | Read directly from 16.19.8230722 client data |
| **[RIOT]** | Stated in 26.x patch notes |
| **[WIKI]** | Stated on the cited wiki revision |
| **[INFERRED]** | Derived by this researcher; reason given |

Confidence is H / M / L. Where sources disagree, the default choice and the reason are given inline and collected in §17.

---

## 1. Conventions every implementer must follow

1. **Units.**
   - Distances are game units.
   - Times are **seconds** in formulas. Sim state may keep ms, but must convert explicitly.
   - Percentages are decimals (0.30, not 30).
   - Bonus AS is a decimal fraction (0.25 = +25%).
   - Health regen *stats* are quoted **per 5 s** in tooltips. Client champion records store **per second** (`baseStaticHPRegenModifiable` Garen = 1.6/s ≡ 8 HP5). [CLIENT for the stored value; INFERRED for the per-second unit, M. It matches wiki Garen HP5 = 8. Verify with U-11.]
2. **Float precision.**
   - Riot's server is float32 internally [INFERRED from bin float32 values such as 0.30000001192092896, H].
   - Keep float32 in JAX. Never round damage, health or stats internally.
   - HUD rounding (HP shown ceil-ed, stats to 2 dp, AS tooltip to 3 dp) is **display only** [WIKI Champion statistic/Attack speed, H].
3. **Server tick.**
   - The live server targets **30 Hz (1/30 s ≈ 33.3 ms)**. Actions that end mid-tick complete on the **next whole tick** ("tick rounding"; a 0.25 s cast resolves after 8 ticks = 0.264 s) [WIKI Tick and updates, M].
   - The legacy port runs 60 Hz (`autoattack.py:164` `delta_ms = 1000/60`). That was a LeagueSandbox property, not a Riot one.
   - The modern profile must make tick length a config value and default to 1/30 s once U-01 confirms it. Every timer in this doc must be evaluated with "fires on the first tick whose accumulated time ≥ deadline".
4. **One damage instance = one packet.** Every damaging effect produces packets with the fields in §3.1. Never fold several sources into one number before the pipeline: shields, kill credit and vamp all need per-packet attribution.
5. **Pure, vectorised.** Every rule below is written so it can be an elementwise `xp` function, like `core/stats.py`. Data-dependent branches become `where`.

---

## 2. Hook taxonomy (other specs reference these IDs)

Hooks are named `<PIPELINE>.<nn>_<NAME>`. Numbers give the order; gaps are reserved. A system plugs in by declaring "contributes at `DMG.40_DEALT_MOD`" and so on. **No system may introduce an ordering that is not one of these slots without amending this doc.**

### 2.1 Stat pipeline — `STAT.*` (evaluated whenever a stat is read; recompute every tick or cache with dirty flags)

| ID | Meaning | Typical contributors |
|---|---|---|
| `STAT.00_BASE` | Champion/unit record base value | champion bin, minion/turret bins |
| `STAT.10_GROWTH` | `growth × G(level)` (§3.2), part of **base** | level-up |
| `STAT.20_FLAT_BONUS` | Flat bonus, summed | items (`mFlat*`), shards, runes, buffs, adaptive force |
| `STAT.30_PERCENT_BONUS` | Additive % on (base + flat): MS %, % health, base-regen %, etc. | items `mPercent*`, buffs |
| `STAT.40_MULTIPLICATIVE` | Separately multiplied factors (multiplicative MS, cripple, `mPercentMultiplicativeAttackSpeedMod`) | rare buffs |
| `STAT.50_STAT_DERIVED` | Stats derived from other stats (Sterak's AD from base AD, Overlord's HP→AD, adaptive-force type choice) | item passives; evaluate after 00–40 of their inputs, non-recursively |
| `STAT.60_CAP` | Caps, floors, soft caps (AS 0.2–3.003, MS soft caps, AH ≤ 500, crit ≤ 1, tenacity ≤ 1, slow resist ≤ 1) | engine |
| `STAT.70_MAXHP_SYNC` | Apply max-HP delta rule to current HP (§11) | engine |

### 2.2 Damage pipeline — `DMG.*` (per packet; §5 has exact math)

| ID | Meaning |
|---|---|
| `DMG.00_DECLARE` | Source emits packet: raw amount, type, tags, properties, source, target, cast-instance id |
| `DMG.05_TARGET_VALID` | Target alive/targetable at application; projectiles/attacks cancelled earlier are not here |
| `DMG.10_PREVENT` | Parry (dodge/block/blind) for `RespectDodge` packets; spell shield for `ActiveSpell`-tagged spell effects; invulnerability (→ 0 and stop, still emits a 0-damage event for combat status per wiki) |
| `DMG.15_CRIT` | Crit multiplier applied to crit-capable portion (roll happens at attack **launch**, §8.4) |
| `DMG.20_RAW_ADD` | Raw additions that are part of the same packet before any modifier (e.g. %HP on-hit already folded by its own source — normally separate packets) |
| `DMG.25_DAMAGE_CAP` | Caps vs. monsters/minions applied to pre-mitigation amount |
| `DMG.30_PREMIT_FLAT` | Pre-mitigation flat reduction on target (Fizz passive, Amumu Tantrum, Guardian's Horn); not on true damage |
| `DMG.40_DEALT_MOD` | Source-side % modifiers ("Empower"/"Weaken"): **summed additively** since 26.09, then applied once; Exhaust here (not on true) |
| `DMG.45_UNIT_CLASS` | Unit-class ratio `dr_<src>To<dst>` (§5.6) |
| `DMG.50_RESIST` | Effective resist (reduction→pen order §4) and mitigation multiplier; physical/magic only |
| `DMG.60_RECEIVED_MOD` | Target-side % ("Vulnerable"/"Durable"): **multiplicative** with each other and with resist; true damage ignores "Durable" DR unless the effect says otherwise, but is affected by amplifiers ("Vulnerable") |
| `DMG.70_POSTMIT_FLAT` | Post-mitigation flat reduction (Bone Plating, Warden's Mail, Plated Steelcaps flat parts if any); not on true; clamp ≥ 0 |
| `DMG.75_FINAL` | `dmg_final = max(0, …)`; this is "post-mitigation damage", the number vamp and most "damage dealt" triggers read |
| `DMG.80_SHIELD` | Absorb by shields in priority order (§6) |
| `DMG.85_HEALTH` | Subtract remaining from current HP; apply minimum-health thresholds (Undying Rage etc.) |
| `DMG.90_ON_DAMAGE` | Triggers: combat timers, kill-credit timer, call-for-help/aggro, turret aggression, stacks, grey health, "on damage taken" item passives, Garen Perseverance reset |
| `DMG.92_VAMP` | Life steal / omnivamp heal from `dmg_final` (§7.3), routed into `HEAL.*` |
| `DMG.95_DEATH` | hp ≤ 0 → death-prevention (resurrection/zombie/stasis-on-lethal) → death; killer attribution |
| `DMG.98_ON_KILL` | Bounties (economy spec), takedown stacks, kill-credit/assist window (15 s SR) |

### 2.3 Heal and shield pipelines — `HEAL.*`, `SHIELD.*`

| ID | Meaning |
|---|---|
| `HEAL.00_BASE` | Raw heal amount (ability/item/vamp/regen tick) |
| `HEAL.10_SOURCE_POWER` | × (1 + source heal-and-shield power), only for heals/shields that "benefit from HSP" (not regen, not life steal/omnivamp) |
| `HEAL.20_RECEIVED_MOD` | × (1 + Σ target incoming healing increases) — Spirit Visage, Revitalize, Immortal Path below 50% |
| `HEAL.30_GW` | × (1 − 0.40) if Grievous Wounds active (non-stacking) |
| `HEAL.40_APPLY` | `hp = min(max_hp, hp + heal)`; negative heals floor at 0 and are not damage |
| `SHIELD.00_BASE` … `SHIELD.10_SOURCE_POWER`, `SHIELD.20_RECEIVED_MOD` (Spirit Visage ShieldIncrease), `SHIELD.30_SHIELD_REAVER` (Serpent's Fang), `SHIELD.40_INSERT` (into shield list with type, expiry, priority) |

### 2.4 Tick-level ordering — `TICK.*` (proposed canonical in-tick order for the modern profile)

| ID | Phase |
|---|---|
| `TICK.00_INPUT` | Apply orders (casts, attack commands, item actives) |
| `TICK.10_BUFFS` | Buff clocks advance and expire. Stat buffs added or removed here, then `STAT.*` is re-evaluated |
| `TICK.20_REGEN` | On 0.5 s boundaries: HP and mana regen (§12) |
| `TICK.30_MOVE` | Movement and dashes using this tick's MS (§9) |
| `TICK.40_CAST_RESOLVE` | Cast times and windups completing this tick fire their effects. Missile spawns |
| `TICK.50_ATTACK` | Attack clock and swing gate (§8) |
| `TICK.60_PROJECTILES` | Missiles advance; arrivals emit packets |
| `TICK.70_DAMAGE` | All packets emitted this tick are resolved through `DMG.*` in **deterministic packet order** (U-02) |
| `TICK.80_DEATH` | Deaths, kills, rewards, respawns |
| `TICK.90_TIMERS` | Combat and out-of-combat timers, cooldown decrement (§10.1) |

Legacy `step.py` order (buffs → regen → movement → targeting → autoattack/missile → death) is compatible with this. Whether Riot resolves damage at emission time inside each object's update (as LeagueSandbox did) is **unknown** (U-02). The default is resolve-in-emission order. Packets are sorted by (emitter phase, emitter index, emission sequence), which reproduces the existing lowest-index-crossing kill-credit convention in `step.py`.

## 3. Stat calculation pipeline

### 3.1 Damage packet (state shape, referenced by §5)
`Packet{src, dst, raw: f32, dtype ∈ {PHYSICAL, MAGIC, TRUE}, tags: bitset(14), props: bitset, crit_capable: f32 (portion), is_crit: bool, cast_id: i32, kind_src, kind_dst}`.
- Tags are exactly the client list **[CLIENT `DamageSourceSettings`, H]**: `AOE, Periodic, Indirect, BasicAttack, ActiveSpell, Proc, Pet, NonRedirectable, Item, DoesNotAggroJungle, OnHit, Augment, Burn, NonAmpable`.
- Properties **[WIKI Damage, M — names datamined, semantics partly unknown]**: `ApplyDamageModifier, ApplyLifesteal, ApplyOmnivamp, ApplyCritical, EnableCallForHelp, EnableKill, RespectDodge, RespectImmunity, TriggerDamageEvents, TriggerOnHitEvents` (+deprecated ApplyPhysicalVamp/ApplySpellvamp).
- Defaults: champion basic attack = `{BasicAttack}` + `ApplyLifesteal, TriggerOnHitEvents, RespectDodge, EnableCallForHelp, ApplyDamageModifier, EnableKill`. On-hit item damage = `{OnHit, Proc}` (+`Item`), lifesteal per item ([WIKI Life steal], "most on-hit items" apply it). Ignite/Smite: `Item`, **no `ApplyDamageModifier`**, no omnivamp [WIKI Damage modifier "Exceptions"; RIOT 26.4], true damage that "cannot be amplified" [WIKI True damage]. Executes: untagged, ignore and destroy shields (§5.9).

### 3.2 Level growth — [CLIENT `GlobalPerLevelStatsFactor`, H]
Per-level factor table `F[L]` (index = level, `F[1]=0`): `F[L] = 0.65 + 0.035·L` for L = 2..30 (client lists 0.72, 0.755, …, 1.35 at L=20, … 1.70 at L=30).
Cumulative growth multiplier:
```
G(n) = Σ_{L=2..n} F[L] = (n−1)·(0.7025 + 0.0175·(n−1))        (closed form, exact)
grown(stat) = base + growth · G(level)
```
- G(18) = 17.000, G(19) = 18.315, G(20) = 19.665 (levels 19–20 reachable via top role quest, 26.1 [RIOT]).
- **All growth counts as base** for every stat **except attack speed**, whose growth is **bonus AS** [WIKI Champion statistic, H].
- `core.stats.level_growth_sum` and `combat.growth_sum` are both algebraically equal to this; keep one (`core.stats`) and make it read the client table for level>20 safety (they agree to float precision anyway).

### 3.3 Composition order for a generic stat — [WIKI + INFERRED, M]
```
base_total  = base + growth·G(level)                                   STAT.00/10
bonus_flat  = Σ flat (items, shards, runes, buffs, adaptive)          STAT.20
total_pre   = (base_total + bonus_flat) · (1 + Σ additive_percent)    STAT.30
total       = total_pre · Π (1 + multiplicative_i)                     STAT.40
bonus(stat) = total − base_total     # what "bonus AD/armor/HP" ratios read
```
- Most stats have **no** percent stage in SR 26.19 (items give flat AD/HP/AR/MR/AP). Percent stages that exist: MS (§9), AS (special, §8.1), `mPercentBaseHPRegenMod` / base mana regen % (multiplies **base** regen only: `regen = base_regen·(1+Σ%base) + flat_regen`) [WIKI Health regeneration "stacks additively, flat and percentage"; CLIENT field name `PercentBase…`, H], `mPercentHealingAmountMod` (= HSP, a stat on its own).
- The legacy `Stat.Total` form `((base+baseBonus)(1+pBase)+flat)(1+p)` (`core.stats.stat_total`, `core/stats.py:26`) is a superset that can express all of the above (`pBase` = %base, `p` = %total). Keep it, but document per stat which slot each source uses; **do not** let % bonuses that are "% of base" land in the outer slot.
- Percent **reductions** of a stat (e.g., Black Cleaver armor shred) are not stat modifiers: they are **resist reduction** (§4) and must not be fed through `stat_total`.

### 3.4 Item / shard stat aggregation — [CLIENT items/perks bins, H]
- Item stats are flat sums across six slots, **except**: tenacity (multiplicative, §10.2), slow resist (multiplicative, §9.3), % armor/magic pen (multiplicative across sources, §4.2), lethality and flat magic pen (additive), crit chance (additive, cap 1.0), AH (additive, cap 500), crit damage `mFlatCritDamageMod` (additive onto the 2.0 base; IE = +0.30 → 2.30), HSP (additive), omnivamp/life steal (additive), MS flat (additive; **boots flat MS do not stack** with other boots — only one boots item possible by item-group rule anyway).
- Client field → stat mapping (authoritative; prefer over Data Dragon): `mFlatHPPoolMod→health`, `mFlatPhysicalDamageMod→AD`, `mFlatMagicDamageMod→AP`, `mFlatArmorMod`, `mFlatSpellBlockMod→MR`, `mPercentAttackSpeedMod→bonus AS`, `mPercentMultiplicativeAttackSpeedMod→STAT.40 AS`, `mFlatCritChanceMod`, `mFlatCritDamageMod`, `mPercentLifeStealMod`, `mFlatMovementSpeedMod`, `mPercentMovementSpeedMod`, `mAbilityHasteMod`, `mFlatHPRegenMod`, `mPercentBaseHPRegenMod`, `mFlatMPPoolMod`, `mPercentTenacityItemMod`, `mPercentSlowResistMod`, `mPercentHealingAmountMod→HSP`, `PhysicalLethality→lethality`, `mPercentArmorPenetrationMod`, `mFlatMagicPenetrationMod`, `mPercentMagicPenetrationMod`, `mFlatArmorPenetrationMod`(mode items only), `mPercentCooldownMod` (mode items only; legacy CDR, floor `gcd_PercentCooldownModMinimum=-0.4`). Omnivamp has no top-level field: it is granted by item scripts / data values (Doran's Blade 2.5% per 26.1) — ITEMS.md owns these.
- Stat shards 26.19 [CLIENT perks]: adaptive 9 (5.4 AD / 9 AP); AS +10%; AH +8; **MS +2.5%** (additive % MS); HP +65; scaling HP 10→180 by level (INFERRED linear: `10 + 10·(min(L,18)−1)`, M; behaviour at L19–20 U-14); **tenacity +15% and slow resist +15%** (tenacity group A).

### 3.5 Adaptive force — [WIKI Adaptive force, H]
`1 AF = 0.6 bonus AD or 1 AP`. Type chosen **dynamically**: bonus AD (excluding champion-passive AD/AP; including listed item passives) **>** AP → AD; **<** → AP; tie → champion adaptive type. Evaluate at `STAT.50` from the pre-adaptive values (no recursion: the adaptive grant itself is excluded from the comparison). Re-evaluate whenever items/buffs change (it can flip mid-game).

### 3.6 Caps and floors (STAT.60)
| Stat | Rule | Source |
|---|---|---|
| Attack speed | `clamp(AS, 1/gcd_AttackMaxDelay=0.2, 1/gcd_AttackMinDelay=3.003003)` | [CLIENT 0.333/5.0; WIKI, H] |
| Ability haste | ≤ 500 (sum of AH + basic/ultimate haste) | [WIKI Haste, M] |
| Crit chance | clamp [0, 1]; excess does nothing by default | [WIKI, H] |
| Tenacity | ≤ 1.0 (may be negative) | [WIKI, H] |
| Slow resist | ≤ 1.0 (multiplicative ⇒ naturally < 1) | [WIKI, H] |
| Movement speed | soft caps §9.2, ≥ 0 | [WIKI, H] |
| Resists | **no floor on the stat**; may go negative via flat reduction | [WIKI, H] |
| Max HP | ≥ 1 (INFERRED; never 0 for a living unit) | INFERRED, L |

---

## 4. Resistances, reduction and penetration

### 4.1 Mitigation multiplier — [WIKI Armor/MR, H]
```
mult(R) = 100/(100+R)          if R ≥ 0
        = 2 − 100/(100−R)      if R < 0        # ∈ (1, 2); −100 ⇒ 1.5
post = raw · mult(R_eff)
```
`core.stats.mitigation_multiplier` (`core/stats.py:103`) is correct (boundary R=0 → both branches = 1).

### 4.2 Effective resist (attacker-specific), exact order — [WIKI Armor penetration / Magic penetration, H for order]
```
R0 = base_R + bonus_R                                 (target's actual stat)
# --- reductions (debuffs on target; change the actual stat, visible to everyone) ---
R1 = R0 − flat_reduction                              (may go < 0)
R2 = R1 · Π(1 − pct_reduction_i)   if R1 > 0 else R1  (multiplicative across sources)
# --- penetration (attacker property; target stat unchanged) ---
R3 = R2 · Π(1 − pct_pen_i)         if R2 > 0 else R2  (multiplicative across sources)
     # bonus-only %pen variant: base_part + bonus_part·(1−p), see 4.3
R4 = max(0, R3 − flat_pen)          if R3 > 0 else R3 # lethality / flat MPen never push below 0
                                                      # and do nothing if already ≤ 0
```
Key edge cases:
- **Negative resist from reduction survives penetration**: Target 18 AR, 30 flat red, 30% red → −12 → stays −12 (wiki example B).
- Lethality = **flat armor pen, 1:1, not level-scaled** since V14.1 (handled as an innate stat since V14.9). No 26.x patch changed this [WIKI H; RIOT 26.1–26.19 audit H].
- %pen and %reduction stacking: multiplicative (`1−(1−a)(1−b)`).
- Reductions are applied to the **target's** stat (and so change ratios that scale with target armor, e.g., "bonus armor" scalings on the target? — reductions reduce the target's bonus pool first, see 4.3), penetration only in this packet's calculation.
- Turrets have 30% armor penetration (applies to champions *and* minions) [WIKI Armor penetration notes, H]; already used in `lane/towers.py:199,212`.

### 4.3 Base/bonus pool allocation (affects bonus-only %pen and "bonus resist" ratios only)
Two wiki statements conflict on the same revision: "Flat reductions affect the target's bonus amount first, then their base amount" (Order of calculations) vs. "distributed proportionally between base and bonus" (Flat armor reduction section, with worked example 20/40 −15 → 15/30). The total `R1` is identical either way; only (a) bonus-armor %pen (Serylda-type, not in top-lane scope) and (b) ratios on target **bonus** resist differ.
**Default: proportional** (it is the section with a worked example and the more specific statement), confidence L, register U-03. Percentage reduction and %pen scale both pools.

### 4.4 Fixtures
| Case | R_eff | ×500 phys |
|---|---|---|
| 100 AR, 30% pen, 10 lethality | 100·0.7−10 = **60** | 500·100/160 = **312.5000** |
| 100 AR, 20 flat red, 30% red, 25% pen, 10 leth (current unit test) | ((80·.7)·.75)−10 = **32** | 378.7879 |
| 18 AR, −30 flat, 30% red, 45% pen, 10 leth | **−12** | 500·(2−100/112) = **553.5714** |
| 10 AR, −25 flat, 50% red, 40 flat pen | **−15** (current code returns **0** — bug D1, §16) | 500·(2−100/115)= **565.2174** |
| 300 AR (100 b/200 bn), −30 flat, 30% red, 45% *bonus* pen, 10 leth (wiki A) | 63 + 126·0.55 − 10 = **122.3** | 224.9213 (×1000 → 449.8426) |
| 0 AR, 50% pen, 18 leth | **0** | 500 |
| −50 AR | — | 500·(2−100/150)=**666.6667** |

---

## 5. Damage pipeline (one packet), exact math

### 5.1 Types — [WIKI Damage/True damage, H]
Exactly one of physical (armor), magic (MR), true (no resist). Unknown type is a host-side error (`validate_damage_type` keeps).

### 5.2 Prevention (DMG.10) — [WIKI Invulnerability/Spell shield/Basic attack, H]
- **Invulnerable** target: damage → 0 for all types incl. true; life steal/vamp/"pre-mitigation" effects do nothing; still a 0-damage event for some triggers (combat-status "modern" system triggers on invulnerable targets; wiki comment, L).
- **Parry** (dodge e.g. Jax E, block, attacker blinded): packet with `RespectDodge` dropped *and* on-hit effects not applied; no combat/aggro from it (wiki Combat status comment: neither system triggers on dodge/miss, M).
- **Spell shield**: blocks the whole spell-effect application of an `ActiveSpell` cast instance (multiple instances within ~0.1 s from the same cast) — summoner spells and runes are *not* blocked except Comet/Unleashed/Primal Smite. Takes priority over damage shields and CC immunity. Hydra actives (Tiamat/Stridebreaker/Ravenous/Profane) **are** blocked; Titanic is not.
- **Untargetable**: new targeted packets cannot be declared; already-applied DoTs still hit [WIKI Invulnerability, H].

### 5.3 Crit (DMG.15)
`crit_mult = critDamageMultiplier (2.0, [CLIENT champion bins; RIOT 26.1]) + Σ mFlatCritDamageMod (IE +0.30)`. Applies to the crit-capable portion (basic-attack AD portion; abilities with explicit "X% of bonus crit damage": `1 + X·(crit_mult − 1)`). Randuin's Resilience multiplies incoming crit damage (ITEMS.md). Crits never apply vs structures except listed exceptions [WIKI Critical strike, H].

### 5.4 Pre-mitigation flat (DMG.30) and damage caps (DMG.25)
`raw' = max(0, raw − Σ premit_flat)` (not true damage). Caps vs non-champions apply to the **pre-mitigation** amount (negative resist can still exceed the cap post-mitigation) [WIKI Damage "Capped damage", H].

### 5.5 Dealt modifiers (DMG.40) — [WIKI Damage modifier, M]
Since V26.09 (undocumented in notes; wiki marked "pending for test"): **source-side % modifiers add**: `m_dealt = 1 + Σ amp_i − Σ red_i` (e.g. +10% Coup de Grace, +8% LDR ⇒ ×1.18, not ×1.188). Exhaust (`DamageReduction 35` [CLIENT shared SummonerExhaust]) is a dealt-reduction: −0.35 into the same sum, and **does not reduce true damage** [WIKI True damage, H].
Wiki also states amplifiers are now dealt "as a separate damage instance". **Default: compute as a single packet** (affects only effects that count instances or pre-modifier reads, e.g. Illaoi E / Zed R; out of top-lane scope) — U-04.
Packets without `ApplyDamageModifier` (Ignite, Smite, jungle-item true damage) skip 40 and 60-amplifiers.
True damage **is** affected by amplifiers (since V25.S1.3, except Hemoplague bug) [WIKI True damage, H].
Clamp `m_dealt ≥ 0`.

### 5.6 Unit-class ratio (DMG.45) — [CLIENT classic-constants, H for values; enforcement U-05]
`dr_UnitToHero 0.55, dr_UnitToBuilding 0.60, dr_UnitToUnit 1, dr_HeroTo* 1, dr_BuildingTo* 1`. "Unit" = minion. Wiki Minion page says "Against champions and structures, minions deal 60% damage" (current revision; patch history shows 55%→ to champions at some point). **Default: client values (0.55 to champions, 0.60 to structures)**, applied as a multiplier at DMG.45 **unless the minion spec has already baked the ratio into minion AD** (must be applied exactly once; the minions researcher must state which). Register U-05.

### 5.7 Resist and received modifiers (DMG.50, DMG.60)
```
post = raw' · m_dealt · m_class · mult(R_eff)          (physical/magic)
post = raw' · m_dealt_amp_only · m_class               (true; dealt *reductions* ignored, amps kept)
post *= Π (1 + vuln_i) · Π (1 − dr_i)                  (received; multiplicative; dr_i skipped for TRUE unless effect says "reduces true damage")
```
[WIKI Damage modifier: received modifiers multiply each other and with resistance, H.] Received DR examples in lane: Garen W (applies to physical+magic; current `modern.py:246` matches), Plated Steelcaps (−% from basic attacks; ITEMS.md), Jax E (AoE only), Death's Dance (not DR—damage *delay*, ITEMS.md).

### 5.8 Post-mitigation flat (DMG.70) and final (DMG.75)
`final = max(0, post − Σ postmit_flat)` (not true). `final` is "post-mitigation damage". Rounding: none.

### 5.9 Execute (special path) — [WIKI Kill/Damage, H]
Execute = untagged true damage equal to **current health**, applied only when the condition (threshold) holds, **destroys all shields first**, ignores damage modifiers, respects invulnerability (→ sets hp to 1 under minimum-health thresholds if it respects invulnerability). Skip DMG.30–70; go to DMG.85 with `final = hp`. Collector's Gold/execute thresholds: ITEMS.md. Garen R is **not** an execute in 26.19 (true damage with missing-HP scaling; champion spec).

### 5.10 Champion basic attacks against structures — [WIKI Turret/Basic attack, H]
Damage = 100% base AD + 100% bonus AD + 60% AP; type physical if bonus AD > 0.6·AP, magic if <, (tie U-13, default physical); crits do not apply. Tower spec owns plates/Bulwark; this rule is global.

### 5.11 Ordering of multiple packets in one tick
Packets on the same target are resolved sequentially in the canonical order of §2.4; each sees shields/HP left by the previous one; killer = first packet to bring hp ≤ 0 (existing `step.py` cumulative-sum convention). Packets emitted *after* death in the same tick are dropped except those already in flight that the engine still lands (missiles: lost if target dead — keep existing `missiles.py` behaviour).

---

## 6. Shields

- Shield record: `{amount, type ∈ {ALL, PHYSICAL, MAGIC}, expires_at, priority, source}`. Max simultaneous shields per unit: static capacity K (proposed K=4).
- Absorption (DMG.80) — [WIKI Shield, M]:
  1. Only shields whose type matches the packet absorb (ALL absorbs all types incl. true; PHYSICAL only physical; MAGIC only magic).
  2. Order: special-priority shields first (Camille Adaptive Defenses, Morgana Black Shield — not lane-relevant), then **soonest-expiring first**; ties: earliest-applied first (INFERRED, L).
  3. Each absorbs `min(remaining, amount)`; resistances already applied (shields absorb post-mitigation damage).
- Typed vs untyped tie: typed-before-untyped is **not** stated; default is pure expiry order among applicable shields (U-06).
- Shield strength at creation: `amount = base · (1 + HSP_source) · (1 + Σ received_shield_inc) · (1 − shield_reaver_debuff)` (`SHIELD.*`). Grievous Wounds **does not** affect shields [WIKI GW, H].
- Execute and Nexus-obelisk damage bypass/destroy shields; health costs bypass shields.
- Shields expire at `expires_at` (tick rounding); decaying shields (Sterak's) define their own amount curve.
- Current `step.py` (single `champion.shield` scalar, consumed before HP across all packet rows) = one ALL-type shield; adequate for Garen W only. Must become a list for items/runes.

## 7. Healing, heal & shield power, Grievous Wounds, vamp

### 7.1 Modifier matrix — [WIKI Template:Healing modifiers oldid 4021803, H]
| Healing kind | Source HSP | Spirit Visage-type incoming % | Revitalize | Grievous Wounds |
|---|---|---|---|---|
| Health regeneration | no | yes | no | **yes** |
| Life steal | no | yes (multiplies the stat value) | yes | yes |
| Omnivamp | no | yes (multiplies the stat value) | yes | yes |
| Drain effects (ability "heal for X% of damage") | yes | yes | yes | yes |
| Self heals | yes | yes | yes | yes |
| Self shields | yes | yes | yes | **no** |
| Incoming heals (from ally) | no (the *healer's* HSP applies) | yes | yes | yes |
| Outgoing heals/shields | yes | no | yes | no (applies on recipient) |
| Bonus health | no | no | no | no |

### 7.2 Heal formula
```
heal = base · (1 + HSP_src)                 [HEAL.10, only kinds marked yes; HSP sums additively]
            · (1 + Σ inc_heal_i)            [HEAL.20; increases sum additively among themselves (INFERRED from
                                              "Multiplicative health regeneration modifiers stack additively", M)]
            · (1 − GW)                      [HEAL.30; GW = 0.40 if any Grievous Wounds debuff, else 0; does NOT stack]
hp = min(max_hp, hp + heal)
```
- GW = **40% for all sources**, unified [CLIENT every item `GrievousAmount 0.4`, Thornmail `EnhancedGrievousAmount 0.4`; WIKI GW, H]. No 60% "enhanced" tier in 26.19 SR. Duration per source (items 3 s). Multiple applications overlap/refresh; strength never exceeds 0.40. GW is removed on death/resurrection.
- HSP and GW are multiplicative factors (26.11 Moonstone "double dip" fix confirms both are factors) [RIOT 26.11, H].
- Negative heal reduces HP but is not damage, cannot kill (floor at 0, no death trigger) [WIKI Healing, H].
- Healing at full HP is lost (no overheal) except explicit overheal-to-shield effects (ITEMS.md).

### 7.3 Life steal and omnivamp (DMG.92)
```
ls_heal  = LS  · final_dmg            if packet has ApplyLifesteal (all BasicAttack, most item on-hit)
ov_ratio = 0.333 if (dst is Minion or Monster) and packet.tags ∩ {AOE, Pet, Periodic/DoT} ≠ ∅ else 1.0
ov_heal  = OV · final_dmg · ov_ratio  for every packet (any type) except Ignite/Smite, reactive (Thornmail), and
                                       packets lacking ApplyOmnivamp
heal via HEAL.20/30 (no HSP)
```
- Constants [CLIENT: `ov_OmnivampRatio 1.0`, `ov_OmnivampModifiedRatio 0.333`, `ov_OmnivampModifiedDamageTags [AOE, Pet, DoT]`, `ov_OmnivampModifiedUnitTags [Minion, Monster]`; `pv_PhysicalVampModifiedRatio 0.33`, `sv_SpellVamp*` deprecated stats], [RIOT 26.1, 26.4], [WIKI Vamp, H]. Wiki "Champion statistic" still says "omnivamp 20% vs non-champions" — **stale**, client + 26.1 notes win.
- Both are computed on **post-mitigation** damage (`DMG.75 final`). Whether shield-absorbed damage counts is not documented; **default: yes** (vamp reads `final`, before DMG.80) — INFERRED from "post-mitigation" wording and from the wiki note that life steal still triggers on damage ignored by minimum-health thresholds, M (U-07). Overkill: default **uncapped** (heal on full `final` even beyond target's remaining HP), L (U-07).
- Life steal does nothing vs structures; nothing vs invulnerable targets [WIKI Life steal/Invulnerability, H].

### 7.4 Shield power — see §6.

---

## 8. Attack timing, crit and on-hit

### 8.1 Attack speed — [WIKI Champion statistic/Attack speed, H; CLIENT bins]
```
bonus_AS = growth_AS/100 · G(level) + Σ bonus_AS_sources        (decimal; e.g. 0.25)
AS = AS_base + AS_ratio · bonus_AS                              (AS_base = attackSpeedModifiable, ratio = attackSpeedRatioModifiable)
AS *= Π(1 + mult_AS_i) · Π(1 − cripple_i)                       (STAT.40; cripples multiplicative)
AS = clamp(AS, 0.2, 1/0.333 = 3.003003)                         (STAT.60)
attack_time T = 1/AS
```
Garen 26.19: base 0.625, ratio 0.625, growth 3.65%; Jax: 0.638, 0.638, 3.4% [CLIENT]. `gcd_AttackDelay=1.6` / `AttackDelayOffsetPercent` are the **legacy** route to base AS and must not be used when `attackSpeedModifiable` is present.

### 8.2 Windup — [WIKI Attack speed "Windup", H; CLIENT fields]
```
windup_pct    = 0.3 + mAttackDelayCastOffsetPercent        (or mAttackCastTime/mAttackTotalTime if present)
wmod          = mAttackDelayCastOffsetPercentAttackSpeedRatio (default 1.0)
base_windup   = windup_pct / AS_base
windup        = base_windup + wmod · (T · windup_pct − base_windup)
```
Garen: windup_pct = 0.18, wmod = **0.5**; Jax: 0.2081, wmod 1.0. Champion crit attacks use `critAttacks[*]` offsets (same for both here).
- **Grace period:** in the last server tick before launch the windup is uncancellable by player commands; after launch, one tick input lockout [WIKI Basic attack, M].
- Launch at windup end. Melee: hit at launch. Ranged: projectile at `missileSpeed` (per-attack spell data; minion/tower specs).
- Cooldown = T from swing start; next swing allowed when elapsed ≥ T (tick-rounded).
- Interruption during windup (new command, attacker or target death, target untargetable/out of sight, target "too far" — leash distance U-08, disarming CC, channel/cast lockout) cancels the attack and **resets the attack timer to 0** (attack not launched) [WIKI Attack speed "Attack timer", H]. The existing legacy distinction (suppressed swing keeps cooldown, `autoattack.py` `cancel_suppressed`) is LeagueSandbox; modern default: every pre-launch cancel resets timer to 0 (M) (U-08).
- **Attack reset** (spell with `Trait_AttackReset` or scripted reset, e.g. Garen Q, Jax W): sets attack timer to 0; during windup = cancels current attack [WIKI Basic attack, H].
- Uncancellable windups (per-spell flag) ignore CC/range cancels (champion specs flag them).

### 8.3 Attack range
Basic attacks use **edge-to-edge** range: `dist(center_a, center_t) ≤ range + r_attacker + r_target` with gameplay radii (champions 65) [WIKI Range/Unit size, H]. Legacy `autoattack.ideal_attack_range` adds only the target radius (`autoattack.py:140-148`) — LeagueSandbox behaviour (D11, §16). Spells: centred unless the spell says edge.

### 8.4 Crit roll — [WIKI Critical strike, M]
Roll per basic attack at launch with a **pseudo-random** failure-streak algorithm: luck modifier table every 0.5% crit (values unknown, error-function shape, 50% at 63% crit), cap of modified probability `1 − 0.2·(1−crit)`, probability = min(cap, luck·streak), streak max 5, streak per target. Expected multiplier `1 + crit·(crit_mult − 1)`. **Default for sim: plain Bernoulli(crit) with an explicit RNG key** until the luck table is extracted (U-09); with crit = 0 (Garen/Jax lane loadouts without crit items) no draw is needed.

### 8.5 On-hit order (per landed attack)
1. Parry check (DMG.10) — if parried, nothing below happens (no on-hit, no stacks that require hit).
2. Main attack packet (BasicAttack, physical unless modified) through `DMG.*`.
3. On-hit packets (`OnHit|Proc`), each its own packet, same tick, in item-slot order then champion/rune order (INFERRED, L; U-02). On-hit damage is **not** crit-multiplied unless stated (Runaan's) [WIKI, H].
4. On-hit non-damage effects (stacks, slows, GW application).
"On-attack" effects fire at windup completion (launch), "on-hit" at impact [WIKI Attack effects, H].

---

## 9. Movement speed — [WIKI Movement speed, H]

### 9.1 Raw
```
raw = (MS_base + Σ flat) · (1 + Σ additive_pct) · Π(1 + mult_pct_j) · (1 − s_eff)
s_eff = max_i slow_i · (1 − SR)                    # strongest slow only (since V5.13)
SR    = 1 − Π(1 − sr_k)                             # slow resist, multiplicative, applied also to slows already active
```
Flat slows/"reduce MS to X" are not affected by slow resist; negative bonus MS is not a slow.

### 9.2 Soft caps (applied to raw)
```
raw > 490 : MS = 0.5·raw + 230
415<raw≤490: MS = 0.8·raw + 83
220≤raw≤415: MS = raw
0≤raw<220 : MS = 0.5·raw + 110
raw < 0   : MS = 0.01·raw + 110
```
(Floor ≈ 110 for any % slow; true 0 only via negative-stat effects.) Movement in a tick = `MS·dt` along path (pathing spec).

### 9.3 Slow-resist sources 26.19: shard 15%, Boots of Swiftness/Swiftmarch (ITEMS.md), runes.

---

## 10. Ability haste, cooldowns, tenacity, CC

### 10.1 Cooldowns — [WIKI Haste/Cooldown, H]
`cd = base_cd · 100/(100 + AH_total)`, `AH_total = min(500, AH + basic_or_ult_haste)`. Item haste and summoner haste are separate stats and apply only to item/summoner cooldowns. Base-cooldown reductions apply before AH; flat refunds after AH, to *current* cooldown. Static cooldowns ignore AH. Cooldown starts at the spell's defined point (champion spec). **Haste gained/lost during a cooldown:** undocumented; default = the remaining cooldown is **rescaled proportionally** (`remaining·(100+AH_old)/(100+AH_new)`), INFERRED, L (U-10). Cooldowns continue while dead.
Legacy CDR (`mPercentCooldownMod`, floor −0.4) is mode-only; not used on SR 26.19 items.

### 10.2 Tenacity — [WIKI Tenacity, H]
```
group_X = 1 − Π_{i∈X}(1 − t_i)          (multiplicative within group)
T = clamp(group_A + group_B + group_C, −∞, 1.0)   (additive across groups)
duration' = duration · (1 − T)            if T ≤ 0   (negative tenacity lengthens)
          = max(min(duration, 0.3), duration · (1 − T))  otherwise   (reduction never goes below 0.3 s)
```
- Group A: items (Mercury's 30% [CLIENT], Sterak's 20%, Endless Hunger, Wit's End…), the shard (15%); group B: Cleanse and mode adjustments; group C: Brittle, negative tenacity.
- Computed **once at application**; later tenacity changes do not alter it.
- Not affected: airborne (knockups/backs/pulls), drowsy, nearsight, stasis, suppression. Affected: stun, root, silence, slow **duration**, cripple duration, taunt/charm/fear, blind, disarm, ground, polymorph, sleep (INFERRED: all others, M).
- Floor 0.3 s only applies when reducing (a 0.25 s stun stays 0.25 s; INFERRED, M).
- Garen W 26.x tenacity burst: champion spec (MODERN_PATCH_DELTA §11.2).

### 10.3 CC state and precedence (state the sim must carry)
Represent CC as independent per-type remaining timers (or buff slots) and derive capability flags each tick:
```
can_move   = ¬(stun ∨ root ∨ airborne ∨ suppression ∨ sleep ∨ stasis ∨ forced_action ∨ channel_move_lock)
can_attack = ¬(stun ∨ airborne ∨ suppression ∨ sleep ∨ stasis ∨ disarm ∨ charm ∨ flee ∨ cast_lockout ∨ channel)
can_cast   = ¬(stun ∨ airborne ∨ suppression ∨ sleep ∨ stasis ∨ silence ∨ polymorph ∨ forced_action)
can_dash   = can_move ∧ ¬ground
```
- Multiple CCs coexist; effects are the OR of flags (no "strongest wins" except slow strength and MS). Same-type reapplication: separate buff instances, effective remaining = max (INFERRED, M).
- Cast-inhibiting CC interrupts channels (Disrupt list); root/ground interrupt only movement channels.
- Stasis and suppression also block summoner spells; other cast-inhibiting CC blocks only Flash/Teleport among summoners.
- CC immunity blocks new CC, not active CC. Spell shield is checked before CC immunity.
- Blind: attacks miss on hit (parry), does not prevent declaring.
- Kill credit: any CC except blind/cripple counts [WIKI Kill].

---

## 11. Max-health changes and level-up

- **Max-HP increase** (items, buffs, shards, level): current HP += same delta. **Decrease**: current HP unchanged unless > new max, then clamped [WIKI Health, H]. Not healing (no HSP/GW) [WIKI Healing, H]. `core.stats.change_max_health` (`core/stats.py:59`) is correct.
- **Level-up:** stats gain `growth·F[new_level]`; current HP gains `ΔmaxHP · (ai_levelUp_healthGainNetGain − ai_levelUp_healthGainPercentMissingPenalty·missing_frac)` with client values **1.0** and **0 (unset)** ⇒ full delta [CLIENT, H for values; INFERRED formula, M]. Wiki Healing "actual health regained is lower depending on how wounded" contradicts this for SR — default to client (penalty 0) (U-12). Mana likewise gains full delta (INFERRED, M).
- Max mana changes follow the same increase/decrease rule (INFERRED, M).

---

## 12. Regeneration

- HP and mana regen are applied in **0.5 s ticks**: `hp += (HP5_total/5)·0.5` every 0.5 s (game-clock aligned; U-11), only while alive and hp < max; disabled at exactly full [WIKI Health regeneration, H for 0.5 s; phase INFERRED].
- `HP5_total = (base_regen + growth·G(level))·(1 + Σ %base_regen) + Σ flat_regen` (all per 5 s; champion bin values are per second ×5).
- Regen is healing for modifier purposes: GW (×0.6) and incoming-heal % apply, HSP does not (§7.1).
- Percent-max-HP regeneration passives (Garen Perseverance) are champion-spec effects but should use the same 0.5 s tick unless shown otherwise (U-11). `modern.py:190` integrates continuously per sim tick.
- Fountain (spawn platform): `sp_HealthRegenPercent 0.02`, `sp_ManaRegenPercent 0.025`, `sp_RegenTickInterval 0.25`, `sp_RegenRadius 1100`, `sp_HPMaxPenaltyRegenPercent 0.08` [CLIENT, H values; semantics of the last INFERRED L] — economy/lifecycle spec.

---

## 13. Combat status, death, kill credit

- **In combat** on dealing or receiving damage (incl. 0-damage instances) or CC to/from an enemy unit (champion, minion, monster, turret; not wards/plants). Not if blocked by spell shield, not on dodge/miss. Default out-of-combat delay **5 s** unless the effect states its own [WIKI Combat status, M]. Expose per-unit `last_combat_t` and also `last_damaged_by_champion_t`, `last_dealt_to_champion_t` (effects differ).
- **Death (DMG.95):** after DMG.85, if hp ≤ 0: run death-prevention (min-health thresholds are already applied at DMG.85; revive/zombie effects here); else die. On death: clear buffs/CC/shields/GW, keep cooldowns running [WIKI Death, H].
- **Kill credit:** last enemy champion to damage or CC (not blind/cripple) the victim within **15 s** (SR) gets the kill; others in window get assists; if none, minion/turret/monster final blow = "executed" (no bounty) [WIKI Kill, H]. Client `aiExp_timeForKillCreditAfterDeath = 10` and `events_TimerForBuildingKillCredit = 30` are related timers (economy spec).
- Multi-kill windows 10 s / 30 s [CLIENT events_TimeForMultiKill/LastMultiKill, H].

---

## 14. State the simulator must carry (shared layer; static capacities)

| Per unit | Type | Why |
|---|---|---|
| `level`, `xp` | i8, f32 | growth G(level), level-up HP rule |
| base/growth table row (`model`) | idx | STAT.00/10 |
| flat/percent/multiplicative bonus accumulators per stat (AD, AP, HP, AR, MR, AS, MS, AH, crit, crit_dmg, LS, OV, HSP, lethality, armor_pen_pct[k], mpen_flat, mpen_pct[k], hp5_flat, hp5_pct_base, mp5…, tenacity groups A/B/C, slow_resist list) | f32 | §3; %pen and tenacity/slow resist need either a product accumulator `Π(1−x)` (preferred: store the product) |
| `hp`, `max_hp`, `mana`, `max_mana` | f32 | §11 |
| target-side debuffs: `armor_flat_red`, `armor_pct_red_prod`, `mr_flat_red`, `mr_pct_red_prod`, `received_mod_prod`, `vuln_prod`, `premit_flat`, `postmit_flat`, `gw_until` | f32 | §4/§5/§7 |
| source-side: `dealt_mod_sum` (amps − reductions), `exhaust_until` | f32 | §5.5 |
| shields: K × `{amount, type, expires_at, priority}` | f32/i8 | §6 |
| CC timers per type (stun, root, silence, disarm, airborne, suppression, stasis, charm, flee, taunt, polymorph, ground, blind, cripple(+strength), sleep) | f32 | §10.3 |
| slows: K × `{strength, expires_at}` (strongest-only evaluation) | f32 | §9 |
| status flags: invulnerable, untargetable, spell_shield(count), cc_immune, slow_immune | bool/i8 | §5.2 |
| attack clock: `attack_timer`, `windup_left`, `is_attacking`, `launched`, `aa_target(+seq)`, `crit_streak[target?]` | f32/i32 | §8 |
| cooldowns per slot + `cd_base_scale` snapshot (for U-10 rescale) | f32 | §10.1 |
| `regen_phase_t` (next 0.5 s boundary; global clock suffices) | f32 | §12 |
| combat: `last_combat_t`, `last_hit_by_enemy_champion_t[champ]`, kill-credit/assist ledger (15 s) | f32 | §13 |
| RNG key (crit, any random) | key | §8.4 |

## 15. Events / hooks emitted (consumers: items, runes, champions, economy)
`ON_STAT_CHANGED(stat)`, `ON_LEVEL_UP(new_level)`, `ON_ATTACK_DECLARED`, `ON_ATTACK_LAUNCH` (on-attack), `ON_ATTACK_CANCELLED(reset)`, `ON_ATTACK_RESET`, `ON_HIT(packet)`, `ON_PARRIED`, `ON_DAMAGE_DEALT(packet, final, absorbed)`, `ON_DAMAGE_TAKEN(packet, final, absorbed)`, `ON_SHIELD_BROKEN`, `ON_HEAL(amount, kind)`, `ON_CC_APPLIED(type, duration')`, `ON_ENTER_COMBAT`, `ON_LEAVE_COMBAT`, `ON_LETHAL(packet)` (death-prevention hook), `ON_DEATH(killer, assists)`, `ON_KILL / ON_TAKEDOWN`, `ON_CAST(slot, cast_id)`, `ON_COOLDOWN_STARTED(slot, cd)`. Each fires at the `DMG.*`/`TICK.*` slot named in §2 and carries the packet so subscribers can filter by tags.

---

## 16. Diff vs current implementation

| # | Location | Current | Required (this spec) | Severity |
|---|---|---|---|---|
| D1 | `core/stats.py:88`, `:100` (`armor_after_modifiers`, `magic_resist_after_modifiers`) | `max(0, r − flat_pen − lethality)` unconditionally | `where(r > 0, max(0, r − pen), r)` — negative resist from reduction must survive (wiki ex. B −12). Current returns 0 for negative armor **even with zero penetration**, removing the negative-armor amplification branch for every caller. | **HIGH** |
| D2 | `lanerl_jax/modern/tests/test_stats_items.py` (`armor_after_modifiers(10, flat_reduction=25, …)==0`) | test asserts the bug | expected **−15** | HIGH |
| D3 | `core/stats.py:69-100` signature | one scalar each for %red/%pen | callers must pre-combine `1−Π(1−x)`; add bonus-only %pen (`base,bonus` split) and pool allocation per §4.3 | MED |
| D4 | `core/stats.py:115-128` `apply_damage_modifiers` | all four factors multiplied | dealt modifiers **additive** (26.09); received multiplicative; received DR skipped for TRUE; amplifiers apply to TRUE; packets without `ApplyDamageModifier` skip; add unit-class ratio slot; order relative to resist per §5.7 (commutative for multiplicative parts, but flat pre/post reductions are not) | MED |
| D5 | `core/stats.py:45-56` `adaptive_force_total` | adaptive type is a static loadout flag | dynamic bonus-AD vs AP comparison, tie → champion type (§3.5) | MED (matters once AP or AD items are mixed) |
| D6 | `items/loadout.py:41-49`, `:80-85` stat extraction | Data Dragon `stats` + regex on description; ItemStats lacks lethality, %armor pen, flat/% MPen, omnivamp, HSP, crit damage, base-HP-regen % (label "base health regen" is not matched → silently dropped), base mana regen %, slow resist, adaptive | read client `items.cdtb.bin.json` fields (§3.4 map); extend `ItemStats`; fail on unknown stat fields | **HIGH** for any build with Black Cleaver/Profane/IE/Spirit Visage/boots |
| D7 | `items/loadout.py:152` `item_loadout_stats` | tenacity (and slow resist) summed | multiplicative within tenacity group A; slow resist multiplicative | MED |
| D8 | `items/loadout.py:207-219` stat shards | MS +2%, tenacity/SR +10% | **MS +2.5%, tenacity +15%, slow resist +15%** [CLIENT perks 5010, 5013] (coordinate with RUNES.md) | MED |
| D9 | `modern.py:72-74` + `step.py:996-1000` | windup = `T_base·(0.3+offset)`, then both period and windup divided by `(1 + growth + bonus)` | AS = base + ratio·bonus (ratio = base for Garen/Jax, so period OK), clamp [0.2, 3.003]; windup per §8.2 with **Garen windup modifier 0.5** (lvl6 + 25% AS: 0.2473 s vs current 0.2066 s) | **HIGH** (last-hit timing) |
| D10 | `autoattack.py:285-287` | `crit_chance > 0` ⇒ **every** attack crits | roll per attack (Bernoulli default, PRD later), multiplier `crit_mult = 2.0 + Σ crit_dmg` | HIGH once any crit source exists |
| D11 | `autoattack.py:140-148` `ideal_attack_range` | `range + target_radius` | `range + attacker_radius + target_radius` (edge-to-edge) for the modern profile; coordinate with minion/tower specs | **HIGH** (every engage distance shifts 65 u) |
| D12 | `autoattack.py:164` and sim tick | 60 Hz (LeagueSandbox) | Riot server 30 Hz; make tick configurable; tick-rounding of windups/casts (U-01) | MED |
| D13 | `autoattack.py` `cancel_suppressed` path | suppressed windup cancel keeps cooldown | modern default: any pre-launch cancel resets timer to 0 (U-08) | LOW–MED |
| D14 | `step.py:1119-1131` | single ALL-type shield scalar, consumed by cumulative rows | shield list with type/expiry ordering (§6) | MED (needed for Sterak's, Overheal, runes) |
| D15 | `modern.py:190` (Garen passive) and world regen | continuous per-tick integration | 0.5 s regen ticks; GW ×0.6 on regen | LOW |
| D16 | `modern.py:219-220` (Garen E crit) | Bernoulli, ×1.3 fixed | ×(1 + 0.3·(crit_mult−1)) → 1.3 base, 1.39 with IE [RIOT 26.1]; champion owners | LOW (champion spec) |
| D17 | `lane/towers.py:174-175` | own resist math `max(0, R·(1−p) − flat)` | call shared `core.stats` (same D1 bug duplicated; also no reduction stage) | MED |
| D18 | (absent) | no unit-class ratio (`dr_UnitToHero 0.55`, `dr_UnitToBuilding 0.6`) | apply exactly once (minion spec decides where) | MED (U-05) |
| D19 | (absent) | no GW, HSP, vamp, omnivamp minion penalty | §7 | MED |
| D20 | `core/stats.py:59-66` `change_max_health` | matches wiki | keep; also call on level-up with full delta (§11) | OK |
| D21 | `core/stats.py:34-42` | matches client table | OK; prefer client table lookup for L>20 robustness | OK |

---

## 17. Disagreements and default choices

| Topic | Sources | Default | Why |
|---|---|---|---|
| Flat resist reduction pool | wiki "bonus first" vs "proportional" (same revision) | proportional | worked example; only affects bonus-only pen/ratios |
| Minion damage to champions | client `dr_UnitToHero 0.55` vs wiki Minion "60%" | 0.55 (client) | client data priority; measure U-05 |
| Omnivamp vs non-champions | client 0.333 only for AOE/Pet/DoT vs minions/monsters; wiki Champion statistic "20% vs non-champions" | client | client + 26.1 notes agree |
| Tenacity stacking | MODERN_PATCH_DELTA §11.2 vs wiki | wiki (mult within, add across) | 26.19 revision explicit with example |
| Damage-dealt stacking | pre-26.09 multiplicative vs wiki "additive since V26.09 (undocumented)" | additive | most recent; flagged pending test (U-04) |
| Crit damage | 4.20/legacy 2.0, 10.23–25: 1.75, 26.1: 2.0 | 2.0 (+0.30 IE) | client `critDamageMultiplier 2.0` |
| AS cap | brief "2.5" | 3.003 (1/0.333) | client `gcd_AttackMinDelay` + wiki |
| GW strength | brief "40/60" | 40% flat, no enhanced tier | every client item value is 0.40 |
| Level-up HP penalty | wiki Healing "lower when wounded" vs client penalty constant 0 | full delta | client CLASSIC constant |
| Mercury's tenacity | Classic 35% (26.19 Classic section) vs SR client 30% | 30% | mode separation |
| Tick rate | legacy 60 Hz vs wiki 30 Hz | 30 Hz configurable | wiki; needs U-01 |

---

## 18. Unresolved / needs live measurement

| ID | Question | Test scenario (Practice Tool, 26.19, record replay/video at ≥60 fps + damage numbers) |
|---|---|---|
| U-01 | Server tick 30 Hz and tick rounding | Garen auto-attack a dummy at level 1 for 60 s, count attacks (expect T=1.6 s → 37–38) and frame-time stamps of hit numbers; 0.25 s cast-time spell → measure 0.264 s |
| U-02 | Order of multiple packets same tick / on-hit order; killer when two hits land same tick | Two champions hit a low-HP minion on the same tick (synchronised autos); observe gold; Recurve Bow + Kraken on-hit ordering via damage log |
| U-03 | Flat reduction pool allocation | Dummy with known base/bonus armor; apply Black Cleaver stacks; read target bonus armor in tooltip |
| U-04 | Dealt modifiers additive & separate instance | Two dealt amps (e.g. Coup de Grâce + LDR vs low-HP high-bonus-HP dummy); compare 1.18 vs 1.188 on a large hit |
| U-05 | Unit-class ratio 0.55 vs 0.60 and whether already in minion AD | Melee minion hits on 0-armor champion vs on another minion; ratio of numbers |
| U-06 | Typed vs untyped shield order; tie order | Stack a physical-only shield and an all shield with known expiries; take physical hit |
| U-07 | Vamp on shield-absorbed / overkill damage | Life steal attack into shielded target; overkill last-hit on minion; read heal numbers |
| U-08 | Attack-cancel leash distance; suppressed-windup cancel timer reset | Target walks out during windup at varying distance; cast silence/disarm mid-windup and measure next swing start |
| U-09 | Crit PRD luck table | 2,000 attacks at 25/50% crit on dummy; fit streak model (or extract table from client binary) |
| U-10 | Haste change during cooldown | Cast Q, buy/sell AH item mid-cooldown, measure remaining |
| U-11 | Regen phase and units (per second storage) | Garen at fixed HP out of combat; sample HP every frame; check 0.5 s steps of HP5/10 |
| U-12 | Level-up HP gain while wounded | Level up at 50% HP via XP; read HP delta vs growth |
| U-13 | Champion-vs-turret type tie (bonus AD == 0.6 AP) | Construct tie; observe damage colour |
| U-14 | Scaling-HP shard at levels 19–20, MS shard 2.5% confirmation | Practice Tool level 20 via quest; stat panel |
| U-15 | `gcd_AttackSpeedCatchupPercent 0.25`, `ai_MaximumHPMaxPenalty 0.5` semantics | Unknown constants; search client scripts / measure AS change mid-swing |
| U-16 | Received DR vs true damage for Garen W, Plated Steelcaps | Take true damage (Ignite) with W active; compare tick values |

---

## 19. Test fixtures (float32 tolerance 1e-4 relative unless stated)

### 19.1 Resist / damage
| # | Inputs | Expected |
|---|---|---|
| F1 | 500 phys; AR 100; 30% pen; 10 lethality | R_eff 60; **312.5** |
| F2 | 500 phys; AR 18; −30 flat red; 30% red; 45% pen; 10 leth | R_eff −12; **553.5714** |
| F3 | 500 phys; AR 10; −25 flat; 50% red; 40 flat pen | R_eff **−15**; **565.2174** |
| F4 | 1000 phys; AR 100 base/200 bonus; −30 flat (proportional); 30% red; 45% bonus pen; 10 leth | R_eff 122.3; **449.8426** |
| F5 | 500 magic; MR 80; −20 flat; 30% red; 35% pen; 10 flat pen | R_eff 17.3; 500·100/117.3 = **426.2575** |
| F6 | 100 true; target AR 300, Garen W 30% DR, Exhaust on attacker | **100** (DR and Exhaust ignore true) |
| F7 | 100 true; attacker +10% amp | **110** |
| F8 | dealt amps +10%, +8%; received DR 30%, 20%; 500 phys vs AR 50 | 500·1.18·0.56·(100/150) = **220.2667** |
| F9 | crit, AD 100, IE (crit_mult 2.3), vs AR 50 | **153.3333** |
| F10 | AR −100 | mult **1.5** |

### 19.2 Stats / timing
| # | Inputs | Expected |
|---|---|---|
| F11 | G(1), G(2), G(6), G(18), G(20) | 0, 0.72, 3.95, 17.0, 19.665 |
| F12 | Garen HP lvl1/2/18/20 (690 + 98·G) | 690, 760.56, 2356, 2617.17 |
| F13 | Garen lvl6 (+3.65%·3.95), +25% bonus AS | AS **0.871359**, T **1.147632 s**, windup **0.247287 s** (legacy formula 0.206574) |
| F14 | Garen lvl1 | T 1.6 s, windup 0.288 s; Jax lvl1 T 1.567398 s, windup 0.326191 s |
| F15 | any unit with bonus AS 10 | AS clamped **3.003003** |
| F16 | 10 s base CD; 20 AH / 500 AH / 600 AH | 8.3333 / 1.6667 / 1.6667 (cap) |
| F17 | Mercury's 30% + Sterak's 20% + shard 15% (group A); 1.5 s stun | T 0.524; **0.714 s** |
| F18 | 0.4 s stun, T 0.9 | max(0.3, 0.04) = **0.3 s** |

### 19.3 Movement
| # | Inputs | Expected |
|---|---|---|
| F19 | 340 base + 45 boots; +35% additive (Stridebreaker active) | raw 519.75 → **489.875** (corrected 2026-10-01: 529.375 was 385 × 1.375) |
| F20 | 385 raw; 40% slow; 15% slow resist | **254.1** |
| F21 | 340 raw; 99% slow | raw 3.4 → **111.7** |
| F22 | raw 450 | **443.0** |
| F23 | slows 40% and 20% simultaneously, 400 raw | **240** (strongest only) |

### 19.4 Healing / vamp / HP
| # | Inputs | Expected |
|---|---|---|
| F24 | heal 100, healer HSP 20%, target GW | **72** |
| F25 | 10% LS, 200 post-mit attack, attacker has GW | **12** |
| F26 | 10% omnivamp, 300 AoE damage to minion | **9.99** |
| F27 | 10% omnivamp, 300 AoE damage to champion | **30** |
| F28 | regen 1.6 HP/s (8 HP5) one 0.5 s tick; with GW | 0.8; **0.48** |
| F29 | hp 300/1000, +200 max HP buff then buff ends after +200 heal | 500 → 700 → stays 700 (max 1000) |
| F30 | shields A{50, ALL, exp 2 s}, B{100, ALL, exp 1 s}; 120 post-mit phys | B → 0, A → 30, hp −0 |
| F31 | shields A{80, MAGIC}, B{50, ALL}; 100 post-mit phys | A untouched, B → 0, hp −50 |
| F32 | execute vs target with 200 hp + 300 shield | shields destroyed, hp → 0 |
