# ITEMS_CATALOG.md: per-item reference, Summoner's Rift, patch 26.19

Companion to [ITEMS.md](ITEMS.md). That document has the rules (inventory, shop, uniqueness, Spellblade, Cleave, Lifeline, vamp, hooks, state). This file has per-item detail for every item purchasable on Summoner's Rift (map 11, CLASSIC mode) in client build **16.19.8230722** (patch 26.19), plus the transform-only items.

**How this file was produced.** Most of each entry was generated mechanically from the patch-matched client data, so the numbers are not hand-copied:
- Client data is CommunityDragon `items.cdtb.bin.json` (sha256 `6880f35d…2ab62726`), resolved through the CLASSIC `GameModeMapData` item lists in `map11.bin` (decoded: `map11-decoded.json`, sha256 `3947d4db…e145`).
- Tooltip strings come from `en_us/data/menu/en_us/lol.stringtable.json` (sha256 `8c051cb2…b8e8`).
- Wiki effect text comes from `Module:ItemData/data` at revision **4069878** (2026-09-29; content sha256 `3a43266a…920a`), via https://wiki.leagueoflegends.com/en-us/Module:ItemData/data?oldid=4069878.
- The **Hooks** and **Implementation** lines are hand-written.

**Tags.** Fields marked [CLIENT] are CLIENT-DATA-VERIFIED. "Wiki …" lines are WIKI. Implementation lines carry their own tags. Wherever the client and the wiki disagree, implement the client value. Disagreements are collected in ITEMS.md §13.

**Conventions.**
- Costs are in gold. "combine" is the recipe price paid on top of the components (`price`). "total" is the recursive sum.
- Sell value = round_half_up(total × sellBackModifier). The default modifier is 0.70; it is 0.40 where listed.
- Attack-speed stats are the bonus attack-speed ratio, so 0.25 means +25% of the champion's AS ratio.
- "Base HP/mana regen %" multiplies the champion's base regen.
- Calculation stat names come from the client `mStat` enum: 0 AP, 1 Armor, 2 AD, 6 MR, 7 MS, 8 Crit chance, 12 Max HP, 29 Lethality, 31 Attack range. 13 appears only on Blade of the Ruined King, where it is the target's current HP. `base`/`bonus` prefixes come from `mStatFormula` 1/2; no prefix means the total.
- `lerp_level(a -> b)` = a + (b − a)·(L − 1)/17 (INFERRED; levels 19–20 from the top role quest are unmeasured, see ITEMS.md §15).
- `level_bp(L1=v, +x/level at L>=k)` = v + x·max(0, L − k + 1). This was verified against the patch-notes values for Locket (290→360 at 18).
- `[ranged holder: x m]` means the client multiplies the whole formula by m when the holder is ranged.
- Placeholders such as `[ShieldSize]` in tooltips are calculations listed on the same entry. `[MeleeRangeSplitRegen]`-style tokens are client tooltip variables with no data value; use the mDataValues.
- Items marked [DEFERRED] (jungle items and most actives) are listed but not specified, per ledger MODERN-009. Tiamat-line and Stridebreaker actives are in scope, with full detail in ITEMS.md §8.


## Starter (8)

### Cull (1083)
- Tier (wiki): Starter; client epicness: 1; in SR store: True
- Cost: total 450, combine 450, sell 180 (40%)
- Stats [CLIENT]: AD 7
- mDataValues [CLIENT]: MinionKillGold=1, CompleteGold=350, MinionKillThreshold=100, OnHitHeal=3
- Tooltip (client en_US, values substituted): 7 Attack DamageReap | Restore 3 Health . | Killing minions grants 1 gold, up to 100. Reaching the limit grants another 350 gold.
- Wiki pass "Reap": Killing a minion grants an additional 1, up to a maximum of 100. After having killed 100 minions, grants an additional 350 and permanently disables this passive.
- Hooks: ON_HIT heal 3; ON_MINION_KILL +1 gold (cap 100 kills) then +350 once [TOP-LANE PRIORITY]
- Implementation: Reap: each minion last-hit by holder grants +1 g until 100 minion kills counted; on reaching 100, grant +350 g once and disable. Restore 3 HP on-hit (every basic-attack hit, any target; flat, not life steal). Kill counter persists only while item held (selling resets - INFERRED).

### Dark Seal (1082)
- Tier (wiki): Starter; client epicness: 1; in SR store: True
- Cost: total 350, combine 350, sell 140 (40%)
- Item groups (client): Glory(max 1) | wiki limit: Glory
- Stats [CLIENT]: Health 50, AP 15
- mDataValues [CLIENT]: MaxGloryStacks=10, APPerGlory=4, GloryOnKill=2, GloryOnAssist=1, GloryLossOnDeath=5
- Calculations [CLIENT]: `CurrentGloryAP = APPerGlory x stacks`
- Tooltip (client en_US, values substituted): 15 Ability Power |  50 HealthGlory | Takedowns grant Glory, up to 10. 5 Glory is lost on death. | Gain 4 Ability Power per Glory. | Kills grant 2 Glory and Assists grant 1.
- Wiki pass "Glory": Gain 2 stacks for each champion kill and 1 stack for each assist, up to a maximum of 10 stacks. For every stack, gain 4 ability power, up to 40 at maximum stacks. Lose 5 stacks on death. Stacks are preserved when upgrading to Mejai's Soulstealer.
- Hooks: ON_TAKEDOWN(champion) stacks; ON_DEATH lose stacks; STAT_DYN AP
- Implementation: Glory: kill +2, assist +1, max 10, -5 on death; +4 AP/stack. Stacks shared/preserved with Mejai's (Glory group).

### Doran's Blade (1055)
- Tier (wiki): Starter; client epicness: 1; in SR store: True
- Cost: total 450, combine 450, sell 180 (40%)
- Item groups (client): DoransItems(max 1) | wiki limit: Starter
- Stats [CLIENT]: Health 80, AD 10, Omnivamp 2.5%
- Tooltip (client en_US, values substituted): 10 Attack Damage |  80 Health |  2.5% Omnivamp
- Hooks: STAT (2.5% omnivamp) [TOP-LANE PRIORITY]
- Implementation: No passive since 26.1 (Life Draining removed).

### Doran's Bow (1086)
- Tier (wiki): Starter; client epicness: 1; in SR store: True
- Cost: total 400, combine 400, sell 160 (40%)
- Item groups (client): DoransItems(max 1) | wiki limit: Starter
- Stats [CLIENT]: AD 8, Attack speed 15% (bonus AS ratio), Omnivamp 1.5%
- mDataValues [CLIENT]: HealthOnHit=3
- Tooltip (client en_US, values substituted): 8 Attack Damage |  15% Attack Speed |  1.5% Omnivamp
- Hooks: STAT

### Doran's Helm (1120)
- Tier (wiki): Starter; client epicness: 1; in SR store: True
- Cost: total 450, combine 450, sell 180 (40%)
- Item groups (client): DoransItems(max 1), {fe69194d} | wiki limit: Starter
- Stats [CLIENT]: Health 150, Armor 8, MR 8
- mDataValues [CLIENT]: BonusDamageToMinions=5
- Tooltip (client en_US, values substituted): 150 Health |  8 Armor |  8 Magic ResistHelping Hand | Attacks deal 5 bonus physical damage to minions.
- Wiki pass "Helping Hand": Basic attacks deal 5 bonus physical damage on-hit against minions.
- Hooks: STAT; ON_HIT(minion) +5 phys [TOP-LANE PRIORITY]
- Implementation: Helping Hand +5 physical vs minions.

### Doran's Ring (1056)
- Tier (wiki): Starter; client epicness: 1; in SR store: True
- Cost: total 400, combine 400, sell 160 (40%)
- Item groups (client): DoransItems(max 1), {fe69194d} | wiki limit: Starter
- Stats [CLIENT]: Health 90, AP 18
- mDataValues [CLIENT]: BonusDamage=5, ManaRestorePerSecond=1, ManaToHealthConversion=0.45, HealthRestorePerSecond=0.55, ManaRestorePerSecondUpgraded=2, UpgradeDuration=5
- Calculations [CLIENT]: `{592c02e8} = BonusDamage(=5)`
- Tooltip (client en_US, values substituted): 18 Ability Power |  90 HealthDrain | Restore 1 Mana every second, increased to 2 Mana per second for 5 seconds after dealing damage to an enemy champion. If you can't gain Mana, heal for 45% of this value instead. | Helping Hand | Attacks deal 5 bonus physical damage to minions.
- Wiki pass "Drain": Restore 1 mana every second. Dealing damage to an enemy champion increases the restoration to 2 mana for the next 5 seconds. If you cannot gain mana, healing forinstead.
- Wiki pass2 "Helping Hand": Basic attacks deal 5 bonus physical damage on-hit against minions.
- Hooks: PERIODIC(1 s) mana restore; ON_DAMAGE_DEALT(champion) -> 5 s upgraded restore; ON_HIT(minion) +5 phys
- Implementation: Drain: +1 mana/s, 2 mana/s for 5 s after damaging an enemy champion; manaless champions heal 0.45x the value instead (0.45 / 0.9 HP/s). Helping Hand +5 vs minions.

### Doran's Shield (1054)
- Tier (wiki): Starter; client epicness: 1; in SR store: True
- Cost: total 450, combine 450, sell 180 (40%)
- Item groups (client): DoransItems(max 1), {fe69194d} | wiki limit: Starter
- Stats [CLIENT]: Health 110, Flat HP regen 4 per 5 s
- mDataValues [CLIENT]: RegenDuration=8, BonusDamageToMinions=5, MaxRegenAmount=40, RangeRegenMult=0.66, MaxRangeRegenAmount=30
- Calculations [CLIENT]: `{592c02e8} = BonusDamageToMinions(=5)`
- Tooltip (client en_US, values substituted): 110 HealthEnduring Focus | Restore 4 Health every 5 seconds.  | After taking damage from a champion, restore up to [MeleeRangeSplitRegen] Health over 8 seconds. | Helping Hand | Attacks deal 5 bonus physical damage to minions.Healing after taking damage is based on your missing Health. | Healing is 66% effective on taking area of effect or periodic damage.
- Wiki pass "Enduring Focus": After taking damage from a champion, gain bonus health regeneration per second equal to {{as||health}} for 8 seconds, refreshing on subsequent champion damage taken. Area of effect, damage over time, or proc damage taken trigger this effect with the ranged values.
- Wiki pass2 "Helping Hand": Basic attacks deal 5 bonus physical damage on-hit against minions.
- Hooks: STAT; ON_DAMAGE_TAKEN(from champion) -> Enduring Focus regen buff 8 s; ON_HIT(minion target) +5 phys [TOP-LANE PRIORITY]
- Implementation: Enduring Focus: on post-mitigation damage taken whose source is an enemy champion, (re)start an 8.0 s buff. While active, extra HP regen per second r = (MaxRegen/8) * min(missing_frac/0.75, 1), MaxRegen = 40 (holder melee) / 30 (holder ranged); recomputed each regen tick from CURRENT missing HP [WIKI formula, CLIENT caps]. If the triggering damage is AoE or periodic (DoT) the buff runs at RangeRegenMult=0.66 effectiveness [CLIENT tooltip; wiki says 'ranged values' (0.75) - implement 0.66]. Refresh replaces (does not stack). Helping Hand: +5 physical on-hit vs minions (pre-mitigation, physical, does not apply life steal per wiki 'starter items on-hit').

### Tear of the Goddess (3070)
- Tier (wiki): Starter; client epicness: 1; in SR store: True
- Cost: total 400, combine 400, sell 280 (70%)
- Builds into (client build hint): Archangel's Staff, Manamune
- Item groups (client): TearItems(max 1), {1fd09102} | wiki limit: Manaflow
- Flags: active spell TearsDummySpell
- Stats [CLIENT]: Mana 240
- mDataValues [CLIENT]: ManaPerCharge=3, ManaChargeAmmoCD=8, BonusMinionDamage=5, MaxMana=360, TakedownMana=0, ManaChargeMaxAmmo=4, InternalCDPerCastID=6.5
- Calculations [CLIENT]: `{592c02e8} = BonusMinionDamage(=5)`
- Tooltip (client en_US, values substituted): [FlatMPPoolMod] ManaManaflow  (8s, max 4 charges) | Landing Abilities grants 3 max Mana (doubled vs. champions), up to 360. | Helping Hand | Attacks deal an additional 5 physical damage to minions.
- Wiki pass "Manaflow": Grants a charge every 8 seconds, up to 4 charges. Dealing ability damage, or applying a buff or debuff to an enemy or ally, with a non-innate ability cast instance, consumes a charge to grant 3 bonus mana, increased to 6 if they are a champion, up to maximum of 360 bonus mana. Can only be triggered once per cast instance.
- Wiki pass2 "Helping Hand": Basic attacks deal 5 bonus physical damage on-hit against minions.
- Hooks: ON_ABILITY_HIT mana charge; ON_HIT(minion) +5
- Implementation: Manaflow.


## Consumables and trinkets (9)

### Control Ward (2055)
- Tier (wiki): Consumable; client epicness: 1; in SR store: True
- Cost: total 75, combine 75, sell 30 (40%)
- Item groups (client): WardPink(max 1), Consumable, {c8a26ada}
- Flags: maxStack 2; consumed on use; usable in store; active spell JammerDevice
- Stats [CLIENT]: none
- Tooltip (client en_US, values substituted): Consume | Places a Control Ward that grants vision and reveals enemy Stealth Wards, traps and Camouflaged enemies. | You may carry up to [Effect2Amount] Control Wards.  | Revealed Stealth Wards are disabled for the duration.
- Wiki consume: Places a visible Control Ward at the target location (0.5-second cooldown; 600 range), which {{lcfirst:}}
- Hooks: ACTIVE place ward (deferred vision) [DEFERRED]
- Implementation: Max 2 carried (maxStack 2), WardPink group max1 slot; 1 placed per player (wiki).

### Elixir of Iron (2138)
- Tier (wiki): Consumable; client epicness: 7; in SR store: True
- Cost: total 500, combine 500, sell 200 (40%)
- Item groups (client): Elixir, Consumable, {dd9ecf6f}(max 1) | wiki limit: Elixir
- Flags: required level 9; consumed on use; usable in store; active spell ElixirOfIron
- Stats [CLIENT]: none
- mDataValues [CLIENT]: RequiredLevel=9, TOOLTIPMinimumCost=600, TOOLTIPReductionPerPink=75
- Calculations [CLIENT]: `ChampLevelReached = level_bp(L1=0, +1 once at L>=9)`
- Tooltip (client en_US, values substituted): Requires Level 9 Consume | Grants [Effect1Amount] Health, [Effect2Amount*100]% Tenacity and increased size for [Effect3Amount] minutes. While active, you leave a path behind that boosts allied champions' Move Speed by [Effect4Amount*100]%. | Drinking a different Elixir will replace the existing one's effects.
- Wiki consume: Grants 300 bonus health, 25% Tenacity, and 15% increased size for 180 seconds. While active, moving leaves behind a path briefly that grants 15% bonus movement speed to allied champions within. Can be used while dead.
- Hooks: ACTIVE consume -> 180 s buff (+300 bonus HP, +25% tenacity, +size, ally MS path)
- Implementation: Required level 9; Elixir group purchase cooldown 5 s; drinking another elixir replaces the current one.

### Elixir of Sorcery (2139)
- Tier (wiki): Consumable; client epicness: 7; in SR store: True
- Cost: total 500, combine 500, sell 200 (40%)
- Item groups (client): Elixir, Consumable | wiki limit: Elixir
- Flags: required level 9; consumed on use; usable in store; active spell ElixirOfSorcery
- Stats [CLIENT]: none
- mDataValues [CLIENT]: RequiredLevel=9, TOOLTIPMinimumCost=600, TOOLTIPReductionPerPink=75
- Calculations [CLIENT]: `ChampLevelReached = level_bp(L1=0, +1 once at L>=9)`
- Tooltip (client en_US, values substituted): Requires Level 9 Consume | Grants [Effect2Amount] Ability Power and [Effect6Amount*5]% Mana Regen for [Effect4Amount] minutes. While active, damaging a champion or turret deals [Effect3Amount] bonus true damage ( [Effect5Amount]s against champions). | Drinking a different Elixir will replace the existing one's effects.
- Wiki consume: Grants 50 ability power and 15 bonus mana regeneration for 180 seconds. While active, dealing damage to enemy champions or turrets deals 25 bonus true damage (5 second cooldown on each champion, no cooldown against turrets). Can be used while dead.
- Hooks: ACTIVE consume -> 180 s buff (+50 AP, mana regen, 25 true dmg proc 5 s ICD vs champs)
- Implementation: Level 9.

### Elixir of Wrath (2140)
- Tier (wiki): Consumable; client epicness: 7; in SR store: True
- Cost: total 500, combine 500, sell 200 (40%)
- Item groups (client): Elixir, Consumable | wiki limit: Elixir
- Flags: required level 9; consumed on use; usable in store; active spell ElixirOfWrath
- Stats [CLIENT]: none
- mDataValues [CLIENT]: RequiredLevel=9
- Calculations [CLIENT]: `ChampLevelReached = level_bp(L1=0, +1 once at L>=9)`
- Tooltip (client en_US, values substituted): Requires Level 9 Consume | Grants [Effect2Amount] Attack Damage and [Effect3Amount*100]% Physical Vamp against champions for [Effect4Amount] minutes. | Drinking a different Elixir will replace the existing one's effects.
- Wiki consume: Grants 30 bonus attack damage and heals for 12% of physical damage dealt to champions for 180 seconds. The heal is reduced to 33% effectiveness for area damage. Can be used while dead.
- Hooks: ACTIVE consume -> 180 s buff (+30 bonus AD, 12% physical drain vs champions, AoE 33%) [TOP-LANE PRIORITY]
- Implementation: Level 9. Drain is a heal on post-mitigation physical damage dealt to champions; benefits from heal power, reduced by Grievous Wounds; 33% for area damage [WIKI].

### Farsight Alteration (3363)
- Tier (wiki): Trinket; client epicness: 1; in SR store: True
- Cost: total 0, combine 0, sell 0 (70%)
- Item groups (client): Trinket
- Flags: required level 9; active spell TrinketOrbLvl3
- Stats [CLIENT]: none
- mDataValues [CLIENT]: RequiredLevel=9, TOOLTIPMinimumCost=600, TOOLTIPReductionPerPink=75, PersistVisionRange=500, ScryerVisionRange=800
- Calculations [CLIENT]: `ChampLevelReached = level_bp(L1=0, +1 once at L>=9)`
- Tooltip (client en_US, values substituted): Requires Level 9 |  Active  ([Effect10Amount] - [Effect11Amount]s) | Reveals a distant area for [Effect2Amount] seconds and leaves a Ward that expires upon spotting an enemy champion. | Can see into Terrain and Brush. | Allies cannot target this Ward.
- Wiki act "Trinket": Places a visible Farsight Ward at the target location that grants sight of the surrounding area, including over terrain and through brush and lasting indefinitely. Also grants sight of the area in a 800 radius for 2 seconds. Upon detecting an enemy champion, the ward will increase its sight radius to 800 units and destroy itself after 3 seconds. (cd {{pp|198 to 99|type=average champion level}})

### Health Potion (2003)
- Tier (wiki): Potion; client epicness: 1; in SR store: True
- Cost: total 50, combine 50, sell 20 (40%)
- Item groups (client): Potion(max 1), Consumable, {c8a26ada}
- Flags: maxStack 5; consumed on use; active spell Item2003
- Stats [CLIENT]: none
- mDataValues [CLIENT]: HealAmount=120, PotionDuration=15
- Tooltip (client en_US, values substituted): Consume | Restores 120 Health over 15 seconds. | You may carry up to 5 Health Potions.
- Wiki consume: Health regeneration 4 health every 0.5 seconds over 15 seconds, restoring a total of 120 health (1 second cooldown).
- Hooks: ACTIVE consume -> HoT 120 over 15 s [TOP-LANE PRIORITY]
- Implementation: Consume one charge: buff restoring 120 HP over 15 s (4 HP per 0.5 s tick, 30 ticks) [WIKI]. Multiple activations: each starts own buff? wiki: 1 s cooldown between uses; client data does not encode stacking - INFERRED: separate instances stack (needs measurement). Max stack 5 per slot (client maxStack=5). Potion group max 1 => cannot also hold Refillable.

### Oracle Lens (3364)
- Tier (wiki): Trinket; client epicness: 1; in SR store: True
- Cost: total 0, combine 0, sell 0 (70%)
- Item groups (client): Trinket
- Flags: required level 1; active spell TrinketSweeperLvl3
- Stats [CLIENT]: none
- mDataValues [CLIENT]: Duration=8, MaxAmmo=2, StartingSingleChargeTime=160, EndingSingleChargeTime=100, GameTimeThreshold=14, MaxAmmoPre=1, StartingRadius=600, EndingRadius=750
- Tooltip (client en_US, values substituted): Active  (160 - 100s, max 2 charges) | Reveals enemy Stealth Wards and traps around you for 8 seconds. | Alerts you to nearby Invisible units. | Revealed Stealth Wards are disabled while revealed.
- Wiki act: Consume one charge to summon a Sweeper Drone that escorts you for the next 8 seconds, detecting nearby enemies that are not sight. (cd 5)

### Refillable Potion (2031)
- Tier (wiki): Potion; client epicness: 1; in SR store: True
- Cost: total 150, combine 150, sell 60 (40%)
- Item groups (client): Potion(max 1), Consumable, {c8a26ada}
- Flags: active spell ItemCrystalFlask
- Stats [CLIENT]: none
- mDataValues [CLIENT]: HealAmount=100, ManaAmount=0, PotionDuration=12, MaxCharges=2
- Tooltip (client en_US, values substituted): Active (2 charges) | Restores 100 Health over 12 seconds.  | Refills upon visiting the shop.
- Wiki consume: Consumes a charge to Health regeneration {{as|{{fd|4.16}} health}} every 0.5 seconds over 12 seconds, restoring a total of 100 health (1 second cooldown).
- Wiki pass: Holds charges that refill upon visiting the shop.
- Hooks: ACTIVE charge -> HoT 100 over 12 s; ON_SHOP_VISIT refill to 2 [TOP-LANE PRIORITY]
- Implementation: 2 charges, 100 HP over 12 s (8.33/s), refill on entering shop range (fountain).

### Stealth Ward (3340)
- Tier (wiki): Trinket; client epicness: 1; in SR store: True
- Cost: total 0, combine 0, sell 0 (100%)
- Item groups (client): Trinket
- Flags: active spell TrinketTotemLvl1
- Stats [CLIENT]: none
- mDataValues [CLIENT]: MaxWardsPlaced=3, GameTimeThreshold=14, MaxAmmo=2, StartingSingleChargeTime=210, EndingSingleChargeTime=90
- Tooltip (client en_US, values substituted): Active  (210 - 90s, max [Effect5Amount] charges) | Places an Invisible Stealth Ward that grants vision for [Effect1Amount]-[Effect3Amount] seconds.
- Wiki act "Trinket": Consume a charge to place an invisible Totem Ward at the target location, which grants sight of the surrounding area for [levels: 90 to 120|type=average champion level] seconds. (cd 2)


## Basic (15)

### Amplifying Tome (1052)
- Tier (wiki): Basic; client epicness: None; in SR store: True
- Cost: total 400, combine 400, sell 280 (70%)
- Stats [CLIENT]: AP 20
- Tooltip (client en_US, values substituted): 20 Ability Power

### B. F. Sword (1038)
- Tier (wiki): Basic; client epicness: None; in SR store: True
- Cost: total 1300, combine 1300, sell 910 (70%)
- Stats [CLIENT]: AD 40
- Tooltip (client en_US, values substituted): 40 Attack Damage

### Blasting Wand (1026)
- Tier (wiki): Basic; client epicness: None; in SR store: True
- Cost: total 850, combine 850, sell 595 (70%)
- Stats [CLIENT]: AP 45
- Tooltip (client en_US, values substituted): 45 Ability Power

### Cloak of Agility (1018)
- Tier (wiki): Basic; client epicness: None; in SR store: True
- Cost: total 600, combine 600, sell 420 (70%)
- Stats [CLIENT]: Crit chance 15%
- Tooltip (client en_US, values substituted): 15% Critical Strike Chance

### Cloth Armor (1029)
- Tier (wiki): Basic; client epicness: None; in SR store: True
- Cost: total 300, combine 300, sell 210 (70%)
- Stats [CLIENT]: Armor 15
- Tooltip (client en_US, values substituted): 15 Armor

### Dagger (1042)
- Tier (wiki): Basic; client epicness: None; in SR store: True
- Cost: total 250, combine 250, sell 175 (70%)
- Stats [CLIENT]: Attack speed 10% (bonus AS ratio)
- Tooltip (client en_US, values substituted): 10% Attack Speed

### Faerie Charm (1004)
- Tier (wiki): Basic; client epicness: None; in SR store: True
- Cost: total 200, combine 200, sell 140 (70%)
- Stats [CLIENT]: Base mana regen 50% of base
- Tooltip (client en_US, values substituted): [PercentBaseMPRegenMod*100]% Base Mana Regen

### Glowing Mote (2022)
- Tier (wiki): Basic; client epicness: None; in SR store: True
- Cost: total 250, combine 250, sell 175 (70%)
- Flags: RestrictedBuffName=HeroPassive
- Stats [CLIENT]: Ability haste 5
- Tooltip (client en_US, values substituted): 5 Ability Haste

### Long Sword (1036)
- Tier (wiki): Basic; client epicness: None; in SR store: True
- Cost: total 350, combine 350, sell 245 (70%)
- Stats [CLIENT]: AD 10
- Tooltip (client en_US, values substituted): 10 Attack Damage

### Needlessly Large Rod (1058)
- Tier (wiki): Basic; client epicness: None; in SR store: True
- Cost: total 1200, combine 1200, sell 840 (70%)
- Stats [CLIENT]: AP 65
- Tooltip (client en_US, values substituted): 65 Ability Power

### Null-Magic Mantle (1033)
- Tier (wiki): Basic; client epicness: None; in SR store: True
- Cost: total 400, combine 400, sell 280 (70%)
- Stats [CLIENT]: MR 20
- Tooltip (client en_US, values substituted): 20 Magic Resist

### Pickaxe (1037)
- Tier (wiki): Basic; client epicness: None; in SR store: True
- Cost: total 875, combine 875, sell 613 (70%)
- Stats [CLIENT]: AD 25
- Tooltip (client en_US, values substituted): 25 Attack Damage

### Rejuvenation Bead (1006)
- Tier (wiki): Basic; client epicness: None; in SR store: True
- Cost: total 300, combine 300, sell 120 (40%)
- Stats [CLIENT]: Base HP regen 100% of base
- Tooltip (client en_US, values substituted): 100% Base Health Regen

### Ruby Crystal (1028)
- Tier (wiki): Basic; client epicness: None; in SR store: True
- Cost: total 400, combine 400, sell 280 (70%)
- Stats [CLIENT]: Health 150
- Tooltip (client en_US, values substituted): 150 Health

### Sapphire Crystal (1027)
- Tier (wiki): Basic; client epicness: None; in SR store: True
- Cost: total 300, combine 300, sell 210 (70%)
- Stats [CLIENT]: Mana 300
- Tooltip (client en_US, values substituted): [FlatMPPoolMod] Mana


## Epic (43)

### Aether Wisp (3113)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 900, combine 500, sell 630 (70%) | recipe: Amplifying Tome + 500 g
- Stats [CLIENT]: AP 30, MS 4% (additive % MS)
- Tooltip (client en_US, values substituted): 30 Ability Power |  4% Move Speed

### Bami's Cinder (6660)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 900, combine 250, sell 630 (70%) | recipe: Ruby Crystal + Glowing Mote + 250 g
- Item groups (client): ImmolateItems(max 1) | wiki limit: Immolate
- Stats [CLIENT]: Health 150, Ability haste 5
- mDataValues [CLIENT]: Range=325, MinionMod=0.5, Cooldown=12, MonsterMod=1, AuraDuration=3, TicksPerSecond=1
- Calculations [CLIENT]: `DamagePerTick = 15`; `DPS = calc[DamagePerTick] * TicksPerSecond(=1)`
- Tooltip (client en_US, values substituted): 150 Health |  5 Ability HasteImmolate | After taking or dealing damage, deal [DPS] magic damage per second to nearby enemies for 3 seconds. | Immolate deals 50% increased damage to minions and 100% increased damage to monsters.
- Wiki pass "Immolate": Taking or dealing damage activates this passive for 3 seconds. Deal 15 magic damage every second to enemies within cr 325 (+ 100% bonus size) units, with the damage being increased to 150% against minions and 200% against monsters. This executes minions that would be killed by one more tick of damage.

### Bandleglass Mirror (4642)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 900, combine 50, sell 630 (70%) | recipe: Faerie Charm + Amplifying Tome + Glowing Mote + 50 g
- Stats [CLIENT]: AP 20, Ability haste 10, Base mana regen 100% of base
- Tooltip (client en_US, values substituted): 20 Ability Power |  [PercentBaseMPRegenMod*100]% Base Mana Regen |  10 Ability Haste

### Blighting Jewel (4630)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1100, combine 700, sell 770 (70%) | recipe: Amplifying Tome + 700 g
- Item groups (client): VoidPen(max 1) | wiki limit: Blight
- Stats [CLIENT]: AP 25, Magic pen 13%
- Tooltip (client en_US, values substituted): 25 Ability Power |  13% Magic Penetration

### Bramble Vest (3076)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 800, combine 200, sell 560 (70%) | recipe: Cloth Armor + Cloth Armor + 200 g
- Item groups (client): {d52cd27b}(max 1), {c8a69ca7}
- Stats [CLIENT]: Armor 30
- mDataValues [CLIENT]: GrievousAmount=0.4, GrievousDuration=3, BaseDamage=10, BonusArmorDamageRatio=0
- Calculations [CLIENT]: `TotalDamage = BaseDamage(=10)`
- Tooltip (client en_US, values substituted): 30 ArmorThorns | When hit by an Attack, deal [TotalDamage] magic damage to the attacker and apply 40% Wounds for 3 seconds if they are a champion.
- Wiki pass "Thorns": When struck by a basic attack on-hit, deal 10 magic damage to the attacker and, if they are a champion, inflict them with Grievous Wounds for 3 seconds.
- Hooks: ON_BEING_HIT(basic attack) reactive 10 magic + GW [TOP-LANE PRIORITY]

### Catalyst of Aeons (3803)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1300, combine 200, sell 910 (70%) | recipe: Ruby Crystal + Ruby Crystal + Sapphire Crystal + 200 g
- Item groups (client): EternityItems(max 1) | wiki limit: Eternity
- Stats [CLIENT]: Health 300, Mana 375
- mDataValues [CLIENT]: EternityManaRestore=0.1, EternityHealthRestore=0.25, EternityMaxHealPerCast=20, EternityCDPerCast=1
- Tooltip (client en_US, values substituted): 300 Health |  [FlatMPPoolMod] ManaEternity | Restores 10% of the damage taken from champions as Mana.  | Casting an Ability heals for 25% of Mana spent. | Mana from Eternity calculates from premitigation damage. | Heal from Eternity is capped at 20 Health per cast, or per second for toggle spells.
- Wiki pass "Eternity": Restore mana equal to 10% of pre-mitigation damage [Damage calculated before modifiers] taken from champions, and heal for [levels: 0 to 20|0 to 80 by 5|type=mana spent|label=healing|color=heal|formula=25% of mana spent, up to 20 healing] per cast. Toggled abilities can only heal for up to 20 per second.

### Caulfield's Warhammer (3133)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1050, combine 100, sell 735 (70%) | recipe: Long Sword + Glowing Mote + Long Sword + 100 g
- Stats [CLIENT]: AD 20, Ability haste 10
- Tooltip (client en_US, values substituted): 20 Attack Damage |  10 Ability Haste
- Hooks: STAT

### Chain Vest (1031)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 800, combine 500, sell 560 (70%) | recipe: Cloth Armor + 500 g
- Stats [CLIENT]: Armor 40
- Tooltip (client en_US, values substituted): 40 Armor

### Crystalline Bracer (3801)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 800, combine 100, sell 560 (70%) | recipe: Ruby Crystal + Rejuvenation Bead + 100 g
- Stats [CLIENT]: Health 200, Base HP regen 100% of base
- Tooltip (client en_US, values substituted): 200 Health |  100% Base Health Regen

### Executioner's Calling (3123)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 800, combine 450, sell 560 (70%) | recipe: Long Sword + 450 g
- Item groups (client): {c8a69ca7}
- Stats [CLIENT]: AD 15
- mDataValues [CLIENT]: GrievousDuration=3, GrievousAmount=0.4
- Tooltip (client en_US, values substituted): 15 Attack DamageGrievous Wounds | Dealing physical damage to champions applies 40% Wounds for 3 seconds.
- Wiki pass "Grievous Wounds": Dealing physical damage to enemy champions inflicts them with Grievous Wounds for 3 seconds.
- Hooks: ON_DAMAGE_DEALT(physical, champion) GW 40% 3 s [TOP-LANE PRIORITY]

### Fated Ashes (2508)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 900, combine 500, sell 630 (70%) | recipe: Amplifying Tome + 500 g
- Stats [CLIENT]: AP 30
- mDataValues [CLIENT]: BurnFlatDamagePerSecond=5, BurnDuration=3, MonsterDamageBonus=15, TickFrequency=0.5
- Tooltip (client en_US, values substituted): 30 Ability PowerInflame | Damaging Abilities deal 15 bonus magic damage over 3 seconds. | Deals an additional 45 magic damage to monsters.
- Wiki pass "Inflame": Dealing ability damage burns enemies, causing them to take 15/6 magic damage every 0.5 seconds over 3 seconds, for a total of 15. Against monsters, the burn deals 7.5 bonus magic damage per tick, dealing a total of (15/6)+7.5 magic damage per tick for up to 15+(7.5*6).
- Hooks: ON_ABILITY_DAMAGE burn

### Fiendish Codex (3108)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 850, combine 200, sell 595 (70%) | recipe: Amplifying Tome + Glowing Mote + 200 g
- Stats [CLIENT]: AP 25, Ability haste 10
- Tooltip (client en_US, values substituted): 25 Ability Power |  10 Ability Haste

### Forbidden Idol (3114)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 600, combine 400, sell 420 (70%) | recipe: Faerie Charm + 400 g
- Stats [CLIENT]: Base mana regen 50% of base, Heal & shield power 8%
- Tooltip (client en_US, values substituted): [PercentBaseMPRegenMod*100]% Base Mana Regen |  8% Heal and Shield Power

### Giant's Belt (1011)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 900, combine 500, sell 630 (70%) | recipe: Ruby Crystal + 500 g
- Stats [CLIENT]: Health 350
- Tooltip (client en_US, values substituted): 350 Health

### Glacial Buckler (3024)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 900, combine 50, sell 630 (70%) | recipe: Cloth Armor + Sapphire Crystal + Glowing Mote + 50 g
- Stats [CLIENT]: Armor 25, Ability haste 10, Mana 300
- Tooltip (client en_US, values substituted): 25 Armor |  [FlatMPPoolMod] Mana |  10 Ability Haste

### Haunting Guise (3147)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1300, combine 500, sell 910 (70%) | recipe: Amplifying Tome + Ruby Crystal + 500 g
- Item groups (client): {8fb77690}(max 1)
- Stats [CLIENT]: Health 200, AP 30
- mDataValues [CLIENT]: BuffCounterDuration=3, SecondsInCombat=3, DamageIncreasePerSecond=0.02, DamageIncreaseMax=0.06
- Tooltip (client en_US, values substituted): 30 Ability Power |  200 HealthMadness | For each second in combat with enemy champions, deal 2% bonus damage, up to 6%.
- Wiki pass "Madness": For each second in combat with enemy champions, deal 2% increased damage, stacking up to 3 times for a total of 6%.

### Hearthbound Axe (3051)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1200, combine 250, sell 840 (70%) | recipe: Long Sword + Dagger + Long Sword + 250 g
- Stats [CLIENT]: AD 20, Attack speed 20% (bonus AS ratio)
- Tooltip (client en_US, values substituted): 20 Attack Damage |  20% Attack Speed

### Hexdrinker (3155)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1300, combine 200, sell 910 (70%) | recipe: Long Sword + Null-Magic Mantle + Long Sword + 200 g
- Item groups (client): LifelineItems(max 1) | wiki limit: Lifeline
- Stats [CLIENT]: AD 25, MR 25
- mDataValues [CLIENT]: LowHealthThreshold=0.3, ShieldLifetime=2.5, Cooldown=90
- Calculations [CLIENT]: `MeleeItemCalcValue = lerp_level(110 -> 280)`; `RangedItemCalcValue = calc[MeleeItemCalcValue] * 0.75`
- Tooltip (client en_US, values substituted): 25 Attack Damage |  25 Magic ResistLifeline (cd: Cooldown) | Taking magic damage that would reduce your Health below 30% grants a [MeleeRangedSplit] magic damage Shield for 2.5 seconds.
- Wiki pass "Lifeline": If you would take magic damage that would reduce you below 30% of your maximum health, you first gain a shield that absorbs {{as| magic damage}} for 2.5 seconds. (cd 90)
- Hooks: ON_HP_THRESHOLD(30%, magic damage) Lifeline magic shield [TOP-LANE PRIORITY]
- Implementation: Shield 110->280 by level (ranged x0.75) for 2.5 s; Lifeline group cd 90 s.

### Hextech Alternator (3145)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1100, combine 300, sell 770 (70%) | recipe: Amplifying Tome + Amplifying Tome + 300 g
- Stats [CLIENT]: AP 45
- mDataValues [CLIENT]: Cooldown=40
- Calculations [CLIENT]: `DamageAmount = 65`
- Tooltip (client en_US, values substituted): 45 Ability PowerRevved (cd: Cooldown) | Damaging a champion deals [DamageAmount] bonus magic damage.
- Wiki pass "Revved": Damaging an enemy champion deals 65 bonus magic damage. (cd 40)

### Kindlegem (3067)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 800, combine 150, sell 560 (70%) | recipe: Ruby Crystal + Glowing Mote + 150 g
- Stats [CLIENT]: Health 200, Ability haste 10
- Tooltip (client en_US, values substituted): 200 Health |  10 Ability Haste

### Last Whisper (3035)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1450, combine 750, sell 1015 (70%) | recipe: Long Sword + Long Sword + 750 g
- Item groups (client): LastWhisper(max 1) | wiki limit: Fatality
- Stats [CLIENT]: AD 20, Armor pen 18%
- Tooltip (client en_US, values substituted): 20 Attack Damage |  18% Armor Penetration
- Hooks: STAT (18% armor pen) [TOP-LANE PRIORITY]

### Lost Chapter (3802)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1200, combine 250, sell 840 (70%) | recipe: Amplifying Tome + Sapphire Crystal + Glowing Mote + 250 g
- Item groups (client): MythicItems(max 1)
- Stats [CLIENT]: AP 40, Ability haste 10, Mana 300
- mDataValues [CLIENT]: ManaRestorePercent=0.2, RestorationDuration=3
- Tooltip (client en_US, values substituted): 40 Ability Power |  [FlatMPPoolMod] Mana |  10 Ability HasteEnlighten | Levelling up restores 20% max Mana over 3 seconds.
- Wiki pass "Enlighten": Upon leveling up, restores 20% of maximum mana over 3 seconds.

### Negatron Cloak (1057)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 850, combine 450, sell 595 (70%) | recipe: Null-Magic Mantle + 450 g
- Stats [CLIENT]: MR 45
- Tooltip (client en_US, values substituted): 45 Magic Resist

### Noonquiver (6670)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1300, combine 350, sell 910 (70%) | recipe: Long Sword + Cloak of Agility + 350 g
- Flags: active spell BloodthirsterDummySpell
- Stats [CLIENT]: AD 15, Crit chance 20%
- Tooltip (client en_US, values substituted): 15 Attack Damage |  20% Critical Strike Chance

### Oblivion Orb (3916)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 800, combine 400, sell 560 (70%) | recipe: Amplifying Tome + 400 g
- Item groups (client): {c8a69ca7}
- Stats [CLIENT]: AP 25
- mDataValues [CLIENT]: GrievousAmount=0.4, GrievousDuration=3
- Tooltip (client en_US, values substituted): 25 Ability PowerGrievous Wounds | Dealing magic damage to champions applies 40% Wounds for 3 seconds.
- Wiki pass "Grievous Wounds": Dealing magic damage to enemy champions inflicts them with Grievous Wounds for 3 seconds.

### Phage (3044)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1100, combine 350, sell 770 (70%) | recipe: Ruby Crystal + Long Sword + 350 g
- Stats [CLIENT]: Health 200, AD 15
- mDataValues [CLIENT]: MoveSpeedBonus=20, MoveSpeedDuration=2, RangedMod=0.5
- Calculations [CLIENT]: `MSBonusSplit = MoveSpeedBonus(=20)   [ranged holder: x RangedMod(=0.5)]`
- Tooltip (client en_US, values substituted): 15 Attack Damage |  200 HealthRage | Attacking grants [MSBonusSplit] Move Speed for 2 seconds.
- Wiki pass "Rage": Basic attacks on-hit grant 20 (melee) / 10 (ranged) bonus movement speed for 2 seconds.
- Hooks: ON_HIT(any unit) +20 MS (melee)/10 (ranged) 2 s [TOP-LANE PRIORITY]
- Implementation: Rage: refresh, no stack.

### Quicksilver Sash (3140)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1300, combine 900, sell 910 (70%) | recipe: Null-Magic Mantle + 900 g
- Item groups (client): Quicksilver(max 1) | wiki limit: Quicksilver
- Flags: active spell QuicksilverSash
- Stats [CLIENT]: MR 30
- mDataValues [CLIENT]: Cooldown=90
- Tooltip (client en_US, values substituted): 30 Magic Resist Quicksilver (cd: Cooldown) | Remove all crowd control debuffs (excluding Airborne).
- Wiki act "Quicksilver": Removes all crowd control debuffs (except Airborne) from your champion. (cd 90)
- Hooks: ACTIVE cleanse (deferred) [DEFERRED]

### Rectrix (6690)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 775, combine 425, sell 543 (70%) | recipe: Long Sword + 425 g
- Stats [CLIENT]: AD 15, MS 4% (additive % MS)
- Tooltip (client en_US, values substituted): 15 Attack Damage |  4% Move Speed

### Recurve Bow (1043)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 700, combine 450, sell 490 (70%) | recipe: Dagger + 450 g
- Stats [CLIENT]: Attack speed 15% (bonus AS ratio)
- mDataValues [CLIENT]: OnHitDamage=15
- Calculations [CLIENT]: `{e4493c5b} = OnHitDamage(=15)`
- Tooltip (client en_US, values substituted): 15% Attack SpeedSting | Attacks deal 15 bonus physical damage .
- Wiki pass "Sting": Basic attacks deal 15 bonus physical damage on-hit.

### Scout's Slingshot (3144)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 600, combine 100, sell 420 (70%) | recipe: Dagger + Dagger + 100 g
- Stats [CLIENT]: Attack speed 20% (bonus AS ratio)
- mDataValues [CLIENT]: Cooldown=40
- Calculations [CLIENT]: `DamageAmount = 40`
- Tooltip (client en_US, values substituted): 20% Attack SpeedBullseye (cd: Cooldown) | Damaging a champion deals [DamageAmount] bonus magic damage.  | Attacks reduce this cooldown by 1 second.
- Wiki pass "Bullseye": Damaging an enemy champion deals 40 bonus magic damage (40 second cooldown, reduced by 1 second on-attack).

### Seeker's Armguard (2420)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1600, combine 500, sell 640 (40%) | recipe: Amplifying Tome + Cloth Armor + Amplifying Tome + 500 g
- Item groups (client): StopwatchGroup(max 1), {139a0cb0} | wiki limit: Stasis
- Flags: active spell Item2420
- Stats [CLIENT]: AP 40, Armor 25
- mDataValues [CLIENT]: Duration=2.5
- Tooltip (client en_US, values substituted): 40 Ability Power |  25 Armor Time Stop (Single use) | Enter Stasis for 2.5 seconds.
- Wiki act "Time Stop": Put yourself in stasis (buff) for 2.5 seconds, rendering you untargetable and invulnerable for the duration but also unable to move, declare basic attacks, cast abilities, use summoner spells, or activate items.
- Hooks: ACTIVE stasis 2.5 s single use (deferred) [DEFERRED]

### Serrated Dirk (3134)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1000, combine 300, sell 700 (70%) | recipe: Long Sword + Long Sword + 300 g
- Builds into (client build hint): Youmuu's Ghostblade
- Item groups (client): {20b00c0e}(max 1) | wiki limit: Dirk
- Stats [CLIENT]: AD 20, Lethality 10
- mDataValues [CLIENT]: LethalityAmount=10
- Tooltip (client en_US, values substituted): 20 Attack Damage |  10 Lethality
- Hooks: STAT (10 lethality)

### Sheen (3057)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 900, combine 650, sell 630 (70%) | recipe: Glowing Mote + 650 g
- Item groups (client): {57352a0f}(max 1) | wiki limit: Spellblade
- Stats [CLIENT]: Ability haste 10
- mDataValues [CLIENT]: SpellbladeCooldown=1.5, Cooldown=1.5
- Calculations [CLIENT]: `SpellbladeDamage = 1 x base AD`
- Tooltip (client en_US, values substituted): 10 Ability HasteSpellblade (cd: Cooldown) | After using an Ability, your next Attack deals [SpellbladeDamage] bonus physical damage .
- Wiki pass "Spellblade": After using an ability, your next basic attack within 10 seconds deals 100% base AD bonus physical damage on-hit (1.5 second cooldown, starts after using the empowered attack).
- Hooks: ON_ABILITY_CAST arm; ON_HIT +100% base AD phys [TOP-LANE PRIORITY]
- Implementation: see ITEMS.md §6.1.

### Spectre's Cowl (3211)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1250, combine 150, sell 875 (70%) | recipe: Ruby Crystal + Null-Magic Mantle + Rejuvenation Bead + 150 g
- Stats [CLIENT]: Health 200, MR 35, Base HP regen 100% of base
- mDataValues [CLIENT]: HealthRegenPassive=0, HealthRegenDuration=10, DamageToRegenDurationRatio=0.333
- Tooltip (client en_US, values substituted): 200 Health |  35 Magic Resist |  100% Base Health Regen

### Steel Sigil (2019)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1100, combine 150, sell 770 (70%) | recipe: Cloth Armor + Long Sword + Cloth Armor + 150 g
- Flags: RestrictedBuffName=HeroPassive
- Stats [CLIENT]: AD 15, Armor 30
- mDataValues [CLIENT]: FlatDR=3
- Tooltip (client en_US, values substituted): 15 Attack Damage |  30 Armor

### The Brutalizer (2020)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1337, combine 212, sell 936 (70%) | recipe: Glowing Mote + Pickaxe + 212 g
- Flags: RestrictedBuffName=HeroPassive
- Stats [CLIENT]: AD 25, Ability haste 10, Lethality 5
- mDataValues [CLIENT]: LethalityAmount=5
- Tooltip (client en_US, values substituted): 25 Attack Damage |  10 Ability Haste |  5 Lethality

### Tiamat (3077)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1200, combine 500, sell 840 (70%) | recipe: Long Sword + Long Sword + 500 g
- Builds into (client build hint): Ravenous Hydra, Titanic Hydra, Profane Hydra
- Item groups (client): {c6428663}(max 1) | wiki limit: Hydra
- Flags: active spell 3077Active
- Stats [CLIENT]: AD 25
- mDataValues [CLIENT]: CleaveRadius=350, Cooldown=10, Radius=450, ActiveADRatio=0.75, MaxProcPerAuto=10
- Calculations [CLIENT]: `MeleeItemCalcValue = 0.4 x AD`; `RangedItemCalcValue = 0.2 x AD`; `PrimaryDamage = ActiveADRatio(=0.75) x AD`
- Tooltip (client en_US, values substituted): 25 Attack DamageCleave | Attacks deal [melee/ranged split: calc MeleeItemCalcValue / RangedItemCalcValue] physical damage to nearby enemies. Crescent (cd: Cooldown) | Deal [PrimaryDamage] physical damage to enemies around you. | Cleave does not trigger on structures.
- Wiki pass "Cleave": Basic attacks on-hit deal 40% AD (melee) / 20% AD (ranged) physical damage to other enemies in a cr 350 radius centered around the target.
- Wiki act "Crescent": Deal 75% AD physical damage to enemies within a cr 450 radius in front of you [100 unit offset in the caster's facing direction]. (cd 10)
- Hooks: ON_HIT cleave AoE; ACTIVE Crescent [TOP-LANE PRIORITY]
- Implementation: see ITEMS.md §8 (full Cleave and active spec).

### Tunneler (2021)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1150, combine 400, sell 805 (70%) | recipe: Long Sword + Ruby Crystal + 400 g
- Flags: RestrictedBuffName=HeroPassive
- Stats [CLIENT]: Health 250, AD 15
- Tooltip (client en_US, values substituted): 15 Attack Damage |  250 Health

### Vampiric Scepter (1053)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 900, combine 550, sell 630 (70%) | recipe: Long Sword + 550 g
- Stats [CLIENT]: AD 15, Life steal 7%
- Tooltip (client en_US, values substituted): 15 Attack Damage |  7% Life Steal

### Verdant Barrier (4632)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1600, combine 400, sell 1120 (70%) | recipe: Amplifying Tome + Null-Magic Mantle + Amplifying Tome + 400 g
- Item groups (client): {548f93b0}(max 1) | wiki limit: Annul
- Stats [CLIENT]: AP 40, MR 25
- mDataValues [CLIENT]: MagicResistPerStack=0.3, Period=60, MaxMR=9, CooldownReductionAmount=0.05, SpellShieldCooldown=60, Cooldown=60
- Tooltip (client en_US, values substituted): 40 Ability Power |  25 Magic ResistAnnul (cd: Cooldown) | Grants a Spell Shield that blocks the next enemy Ability. | Item cooldown is restarted when damage is taken from champions.
- Wiki pass "Annul": Grants a spell shield that blocks the next hostile ability (60 second cooldown, timer restarts upon taking damage from champions).

### Warden's Mail (3082)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1000, combine 400, sell 700 (70%) | recipe: Cloth Armor + Cloth Armor + 400 g
- Stats [CLIENT]: Armor 40
- mDataValues [CLIENT]: MaxHPRatio=0.005, WardenDamageMax=0.2, BlockBase=15
- Tooltip (client en_US, values substituted): 40 ArmorRock Solid | Reduce incoming damage from champion Attacks by 15. | Cannot block more than 20% of the Attack's damage.
- Wiki pass "Rock Solid": Every first incoming instance of post-mitigation [Damage calculated after modifiers] basic damage per cast instance is reduced by 15, with a maximum of 20% reduction each.
- Hooks: ON_PRE_DAMAGE_TAKEN(champion basic attack) -15 flat, capped at 20% of the attack [TOP-LANE PRIORITY]
- Implementation: Rock Solid.

### Winged Moonplate (3066)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 800, combine 400, sell 560 (70%) | recipe: Ruby Crystal + 400 g
- Stats [CLIENT]: Health 200, MS 4% (additive % MS)
- Tooltip (client en_US, values substituted): 200 Health |  4% Move Speed

### Zeal (3086)
- Tier (wiki): Epic; client epicness: 4; in SR store: True
- Cost: total 1200, combine 350, sell 840 (70%) | recipe: Cloak of Agility + Dagger + 350 g
- Stats [CLIENT]: Attack speed 15% (bonus AS ratio), Crit chance 15%, MS 4% (additive % MS)
- Tooltip (client en_US, values substituted): 15% Attack Speed |  15% Critical Strike Chance |  4% Move Speed
- Hooks: STAT


## Boots (tier 1/2) (8)

### Berserker's Greaves (3006)
- Tier (wiki): Boots; client epicness: 4; in SR store: True
- Cost: total 1100, combine 300, sell 770 (70%) | recipe: Boots + Dagger + Dagger + 300 g
- Item groups (client): Boots(max 1), BootsWithoutActives(max 1)
- Stats [CLIENT]: Attack speed 30% (bonus AS ratio), Flat MS 45
- mDataValues [CLIENT]: FeatsAS=0.05
- Tooltip (client en_US, values substituted): 30% Attack Speed |  45 Move Speed
- Hooks: STAT [TOP-LANE PRIORITY]

### Boots (1001)
- Tier (wiki): Boots; client epicness: None; in SR store: True
- Cost: total 300, combine 300, sell 210 (70%)
- Item groups (client): Boots(max 1), BootsOfSpeed, BootsWithoutActives(max 1)
- Stats [CLIENT]: Flat MS 25
- Tooltip (client en_US, values substituted): 25 Move Speed
- Hooks: STAT

### Boots of Swiftness (3009)
- Tier (wiki): Boots; client epicness: 4; in SR store: True
- Cost: total 1000, combine 700, sell 700 (70%) | recipe: Boots + 700 g
- Item groups (client): Boots(max 1), BootsWithoutActives(max 1)
- Stats [CLIENT]: Flat MS 55, Slow resist 25%
- mDataValues [CLIENT]: SlowResistTooltip=25, FeatsMS=5
- Tooltip (client en_US, values substituted): 55 Move SpeedFleetfooted | Reduce the effectiveness of Slows by 25%.
- Wiki pass "Fleetfooted": Gain 25% slow resist.
- Hooks: STAT (slow resist 25%) [TOP-LANE PRIORITY]

### Gluttonous Greaves (3008)
- Tier (wiki): Boots; client epicness: 4; in SR store: True
- Cost: total 1000, combine 700, sell 700 (70%) | recipe: Boots + 700 g
- Item groups (client): BootsWithoutActives(max 1), Boots(max 1), {26d872be}(max 1)
- Stats [CLIENT]: Omnivamp 4%, Flat MS 45
- mDataValues [CLIENT]: OmnivampOnTakedown=0.006, MaxStacks=10
- Tooltip (client en_US, values substituted): 45 Move Speed |  4% OmnivampSlay | Gain 0.6% Omnivamp on Champion takedown, stacking up to 10 times.
- Wiki pass "Slay": Scoring a takedown against an enemy champion grants you 0.6% omnivamp, stacking up to 10 times for a total of 6%.
- Hooks: STAT; ON_TAKEDOWN(champion) +0.6% omnivamp permanent (max 10) [TOP-LANE PRIORITY]

### Ionian Boots of Lucidity (3158)
- Tier (wiki): Boots; client epicness: 4; in SR store: True
- Cost: total 900, combine 350, sell 630 (70%) | recipe: Boots + Glowing Mote + 350 g
- Item groups (client): Boots(max 1), BootsWithoutActives(max 1)
- Stats [CLIENT]: Flat MS 45, Ability haste 10
- mDataValues [CLIENT]: SummonerHaste=10, FeatsHaste=5
- Tooltip (client en_US, values substituted): 10 Ability Haste |  45 Move SpeedIonian Insight | Gain 10 Summoner Spell Haste.
- Wiki pass "Ionian Insight": Gain 10 summoner spell haste.
- Hooks: STAT (+10 summoner haste)

### Mercury's Treads (3111)
- Tier (wiki): Boots; client epicness: 4; in SR store: True
- Cost: total 1250, combine 550, sell 875 (70%) | recipe: Boots + Null-Magic Mantle + 550 g
- Item groups (client): Boots(max 1), BootsWithoutActives(max 1)
- Stats [CLIENT]: MR 20, Flat MS 45, Tenacity 30%
- mDataValues [CLIENT]: FeatsMR=5
- Tooltip (client en_US, values substituted): 20 Magic Resist |  45 Move Speed |  30% Tenacity
- Hooks: STAT (tenacity 30%) [TOP-LANE PRIORITY]

### Plated Steelcaps (3047)
- Tier (wiki): Boots; client epicness: 4; in SR store: True
- Cost: total 1200, combine 600, sell 840 (70%) | recipe: Boots + Cloth Armor + 600 g
- Item groups (client): Boots(max 1), BootsWithoutActives(max 1)
- Stats [CLIENT]: Armor 25, Flat MS 45
- mDataValues [CLIENT]: FeatsArmor=5
- Tooltip (client en_US, values substituted): 25 Armor |  45 Move SpeedPlating | Reduces incoming damage from Attacks by [Effect1Amount*100]%.
- Wiki pass "Plating": Reduces all incoming basic damage by 10% (excluding from turret attacks).
- Hooks: ON_PRE_DAMAGE_TAKEN(basic attack) x0.90 [TOP-LANE PRIORITY]
- Implementation: Plating: incoming basic-attack damage x(1-0.10); turret attacks excluded [WIKI named effect].

### Sorcerer's Shoes (3020)
- Tier (wiki): Boots; client epicness: 4; in SR store: True
- Cost: total 1100, combine 800, sell 770 (70%) | recipe: Boots + 800 g
- Item groups (client): Boots(max 1), BootsWithoutActives(max 1)
- Stats [CLIENT]: Flat MS 45, Flat magic pen 12
- mDataValues [CLIENT]: FeatsMPen=2
- Tooltip (client en_US, values substituted): 12 Magic Penetration |  45 Move Speed
- Hooks: STAT


## Boots (tier 3, mid role-quest reward) (7)

### Armored Advance (3174)
- Tier (wiki): Boots; client epicness: 7; in SR store: True
- Cost: total 1200, combine 0, sell 840 (70%) | recipe: Plated Steelcaps + 0 g
- Item groups (client): {a0b9cfea}(max 1), Boots(max 1)
- Flags: requires buff/currency Feats_NoxianBootPurchaseBuff
- Stats [CLIENT]: Armor 35, Flat MS 45
- mDataValues [CLIENT]: DamageReduction=0.1, Cooldown=15, ShieldAmount=0.08, ShieldDuration=5
- Calculations [CLIENT]: `ShieldAmountCalc = level_bp(L1=90, +10/level at L>=9) + ShieldAmount(=0.08) x bonus MaxHP`; `{5e4a31e2} = 1 x stacks`
- Tooltip (client en_US, values substituted): (Only Mid Lane) Locked until Quest is Completed 35 Armor |  45 Move SpeedPlating | Reduces incoming damage from Attacks by 10%. | Noxian Endurance (cd: Cooldown) | After taking physical damage from a Champion, gain a [ShieldAmountCalc] physical shield for 5 seconds.
- Wiki pass: =>Plated Steelcaps
- Wiki pass2 "Noxian Endurance": Taking physical damage from champions grants you a shield that absorbs [levels: 100 to 200|color=pd] (+ 8% bonus health) physical damage for 5 seconds. (cd 15)
- Hooks: ON_PRE_DAMAGE_TAKEN(attack) x0.9; ON_DAMAGE_TAKEN(physical, champion) shield 15 s cd [TOP-LANE PRIORITY]

### Chainlaced Crushers (3173)
- Tier (wiki): Boots; client epicness: 7; in SR store: True
- Cost: total 1250, combine 0, sell 875 (70%) | recipe: Mercury's Treads + 0 g
- Item groups (client): {9bb9c80b}(max 1), Boots(max 1)
- Flags: requires buff/currency Feats_NoxianBootPurchaseBuff
- Stats [CLIENT]: MR 25, Flat MS 45, Tenacity 30%
- mDataValues [CLIENT]: Cooldown=15, ShieldAmount=0.08, ShieldDuration=5
- Calculations [CLIENT]: `ShieldAmountCalc = level_bp(L1=90, +10/level at L>=9) + ShieldAmount(=0.08) x bonus MaxHP`; `{5e4a31e2} = 1 x stacks`
- Tooltip (client en_US, values substituted): (Only Mid Lane) Locked until Quest is Completed 25 Magic Resist |  45 Move Speed |  30% TenacityNoxian Persistence (cd: Cooldown) | After taking magic damage from a Champion, gain a [ShieldAmountCalc] magic shield for 5 seconds.
- Wiki pass "Noxian Persistence": Taking magic damage from champions grants you a shield that absorbs [levels: 100 to 200|color=md] (+ 8% bonus health) magic damage for 5 seconds. (cd 15)
- Hooks: ON_DAMAGE_TAKEN(magic, champion) shield 15 s cd [TOP-LANE PRIORITY]

### Crimson Lucidity (3171)
- Tier (wiki): Boots; client epicness: 7; in SR store: True
- Cost: total 900, combine 0, sell 630 (70%) | recipe: Ionian Boots of Lucidity + 0 g
- Item groups (client): {9db9cb31}(max 1), Boots(max 1)
- Flags: requires buff/currency Feats_NoxianBootPurchaseBuff
- Stats [CLIENT]: Flat MS 45, Ability haste 20
- mDataValues [CLIENT]: SummonerHaste=20, MeleeMS=0.1, Duration=4, RangedMSMultiplier=0.8
- Calculations [CLIENT]: `MSAmount = MeleeMS(=0.1)   [ranged holder: x RangedMSMultiplier(=0.8)] (shown as %)`; `{5e4a31e2} = 1 x stacks`
- Tooltip (client en_US, values substituted): (Only Mid Lane) Locked until Quest is Completed 20 Ability Haste |  45 Move SpeedIonian Insight | Gain 20 Summoner Spell Haste. | Noxian Haste | Empowering or protecting allies with abilities, dealing damage to enemy Champions with abilities, or casting a Summoner Spell grants [MSAmount] Move Speed for 4 seconds. | Noxian Haste can only be triggered once per Ability cast.
- Wiki pass "Ionian Lucidity": Gain 20 summoner spell haste.
- Wiki pass2 "Noxian Haste": heal, shield or buffing an ally, damaging abilities against champions, and using summoner spells grants you 10% (melee) / 8% (ranged) bonus movement speed for 4 seconds. This can be triggered from the same cast instance only once every 4 seconds.

### Gunmetal Greaves (3172)
- Tier (wiki): Boots; client epicness: 7; in SR store: True
- Cost: total 1100, combine 0, sell 770 (70%) | recipe: Berserker's Greaves + 0 g
- Item groups (client): 3172(max 1), Boots(max 1)
- Flags: requires buff/currency Feats_NoxianBootPurchaseBuff
- Stats [CLIENT]: Attack speed 45% (bonus AS ratio), Life steal 5%, Flat MS 45
- mDataValues [CLIENT]: MeleeMS=0.15, RangedMSMultiplier=0.667, Duration=2
- Calculations [CLIENT]: `MSAmount = MeleeMS(=0.15)   [ranged holder: x RangedMSMultiplier(=0.667)] (shown as %)`; `{5e4a31e2} = 1 x stacks`
- Tooltip (client en_US, values substituted): (Only Mid Lane) Locked until Quest is Completed 45% Attack Speed |  45 Move Speed |  5% Life Steal
- Hooks: STAT [TOP-LANE PRIORITY]
- Implementation: T3 of Berserker's (mid quest).

### Immortal Path (3168)
- Tier (wiki): Boots; client epicness: 7; in SR store: True
- Cost: total 1000, combine 0, sell 700 (70%) | recipe: Gluttonous Greaves + 0 g
- Item groups (client): BootsWithoutActives(max 1), Boots(max 1), {9abc050f}(max 1)
- Flags: requires buff/currency Feats_NoxianBootPurchaseBuff
- Stats [CLIENT]: Omnivamp 4%, Flat MS 45
- mDataValues [CLIENT]: OmnivampOnTakedown=0.006, MaxStacks=10, DamageMod=0.04, HealingMod=0.12
- Tooltip (client en_US, values substituted): (Only Mid Lane) Locked until Quest is Completed 45 Move Speed |  4% OmnivampSlay | Gain 0.6% Omnivamp on Champion takedown, stacking up to 10 times. | Now and Forever | While above half Health, deal 4% increased damage.  | While below half Health, gain 12% increased healing, shielding, and regeneration.
- Wiki pass: =>Gluttonous Greaves
- Wiki pass2 "Now and Forever": While above 50% of your maximum health, you deal 4% increased damage. While below 50% of your maximum health, you gain 12% increased heal, shield, and health regeneration.

### Spellslinger's Shoes (3175)
- Tier (wiki): Boots; client epicness: 7; in SR store: True
- Cost: total 1100, combine 0, sell 770 (70%) | recipe: Sorcerer's Shoes + 0 g
- Item groups (client): {a1b9d17d}(max 1), Boots(max 1)
- Flags: requires buff/currency Feats_NoxianBootPurchaseBuff
- Stats [CLIENT]: Flat MS 45, Flat magic pen 20, Magic pen 8%
- Calculations [CLIENT]: `{5e4a31e2} = 1 x stacks`
- Tooltip (client en_US, values substituted): (Only Mid Lane) Locked until Quest is Completed 20 Magic Penetration |  8% Magic Penetration |  45 Move Speed

### Swiftmarch (3170)
- Tier (wiki): Boots; client epicness: 7; in SR store: True
- Cost: total 1000, combine 0, sell 700 (70%) | recipe: Boots of Swiftness + 0 g
- Item groups (client): {9cb9c99e}(max 1), Boots(max 1)
- Flags: requires buff/currency Feats_NoxianBootPurchaseBuff
- Stats [CLIENT]: Flat MS 65, Slow resist 25%
- mDataValues [CLIENT]: MoveSpeedMultiplier=0.04, SlowResistTooltip=0.25, MSAdaptiveRatio=0.05
- Calculations [CLIENT]: `{5e4a31e2} = 1 x stacks`; `MSToAdaptiveCalc = MSAdaptiveRatio(=0.05) x MS`
- Tooltip (client en_US, values substituted): (Only Mid Lane) Locked until Quest is Completed 65 Move SpeedFleetfooted | Reduce the effectiveness of Slows by 25%. | Noxian Fervor | Gain 5% of your Move Speed as Adaptive Force.
- Wiki pass "Fleetfooted": Gain 25% slow resist.
- Wiki pass2 "Noxian Fervor": Gain adaptive force equal to 5% of your total movement speed.


## Legendary (108)

### Abyssal Mask (8020)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2650, combine 1000, sell 1855 (70%) | recipe: Kindlegem + Negatron Cloak + 1000 g
- Item groups (client): 8020(max 1)
- Stats [CLIENT]: Health 350, MR 45, Ability haste 15
- mDataValues [CLIENT]: Radius=700, DamageAmp=0.12
- Tooltip (client en_US, values substituted): 350 Health |  45 Magic Resist |  15 Ability HasteUnmake | Nearby enemy champions take 12% more magic damage. | A Champion can only be affected by one Unmake effect at a time.
- Wiki pass "Unmake": Enemy champions within 700 units [center to edge] of you become cursed, causing them to receive 12% increased magic damage post-mitigation from all sources.
- Hooks: AURA +12% magic damage taken (enemy champions 700)

### Actualizer (2522)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2800, combine 750, sell 1960 (70%) | recipe: Lost Chapter + Blasting Wand + 750 g
- Item groups (client): {a2015eec}(max 1)
- Flags: active spell 2522Active
- Stats [CLIENT]: AP 90, Ability haste 10, Mana 300
- mDataValues [CLIENT]: Cooldown=60, Duration=8, ManaCostIncrease=1, CooldownTick=0.3, OverflowAddition=3, OverflowRevert=1
- Calculations [CLIENT]: `ManaCalc = (15 + 0.005 x bonus Mana) * 0.01 (shown as %)`
- Tooltip (client en_US, values substituted): 90 Ability Power |  [FlatMPPoolMod] Mana |  10 Ability Haste Mana Made Real (cd: Cooldown) | For 8 seconds, your mana is Empowered. While Empowered, your spells cost 100% more Mana, you gain [ManaCalc] increased Ability damage, Shielding, and Healing, and your basic ability cooldowns progress 30% faster.
- Wiki act "Mana Made Real": For 8 seconds, your mana is Empowered. While Empowered: your abilities cost 100% more mana; you gain 15% (+ 0.5% per 100 bonus mana) increased ability damage and pet damage, healing, and shielding; and your basic abilities' cooldowns progress 30% faster. (cd 60)

### Archangel's Staff (3003)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2900, combine 450, sell 2030 (70%) | recipe: Tear of the Goddess + Lost Chapter + Fiendish Codex + 450 g
- Item groups (client): TearItems(max 1), LifelineItems(max 1), {a6ceaee2}(max 1), {1fd09102} | wiki limit: Manaflow
- Flags: active spell ArchAngelsDummySpell
- Stats [CLIENT]: AP 70, Ability haste 25, Mana 600
- mDataValues [CLIENT]: ManaPerCharge=5, ManaChargeAmmoCD=8, ManaChargeMaxAmmo=5, MaxMana=360, InternalCDPerCastID=6.5, APFromMana=0.01
- Tooltip (client en_US, values substituted): 70 Ability Power |  [FlatMPPoolMod] Mana |  25 Ability HasteAwe | Gain Ability Power equal to 1% bonus Mana. | Manaflow  (8s, max 5 charges) | Landing Abilities grants 5 max Mana (doubled vs. champions). | Transforms into Seraph's Embrace at 360 max Mana.
- Wiki pass "Awe": Grants ability power equal to 1% bonus mana.
- Wiki pass2 "Manaflow": Grants a charge every 8 seconds, up to 5 charges. Affecting an enemy or ally with an ability consumes a charge to grant 5 bonus mana, increased to 10 if they are a champion, up to a maximum of 360 bonus mana.
- Wiki pass3: Transforms into Seraph's Embrace at 360 bonus mana.

### Ardent Censer (3504)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2200, combine 700, sell 1540 (70%) | recipe: Aether Wisp + Forbidden Idol + 700 g
- Item groups (client): 3504(max 1)
- Stats [CLIENT]: AP 45, MS 4% (additive % MS), Base mana regen 125% of base, Heal & shield power 10%
- mDataValues [CLIENT]: AttackSpeedMin=0.25, Duration=6, OnHitMin=20
- Tooltip (client en_US, values substituted): 45 Ability Power |  10% Heal and Shield Power |  [PercentBaseMPRegenMod*100]% Base Mana Regen |  4% Move SpeedSanctify | Healing or Shielding an ally enhances you both for 6 seconds, granting 25% Attack Speed and 20 magic damage .
- Wiki pass "Sanctify": Heal or shield allied champions (excluding yourself) enhances you and them for 6 seconds, granting 25% bonus attack speed and 20 bonus magic damage on-hit on basic attacks.

### Axiom Arc (6696)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2750, combine 363, sell 1925 (70%) | recipe: The Brutalizer + Caulfield's Warhammer + 363 g
- Item groups (client): 6696(max 1)
- Stats [CLIENT]: AD 55, Ability haste 20, Lethality 18
- mDataValues [CLIENT]: UltimateRefundBase=10, LethalityAmount=18, ResetWindow=3
- Calculations [CLIENT]: `UltimateRefund = UltimateRefundBase(=10) + 0.25 x Lethality`
- Tooltip (client en_US, values substituted): 55 Attack Damage |  18 Lethality |  20 Ability HasteFlux | When a champion that you damaged within 3 seconds dies, refund [UltimateRefund]% of your Ultimate Ability's total cooldown.
- Wiki pass "Flux": Scoring a takedown against an enemy champion within 3 seconds of damaging them refunds 10% (+ 0.25% per 1 Lethality) of your ultimate ability's total cooldown.

### Bandlepipes (2524)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2300, combine 800, sell 1610 (70%) | recipe: Kindlegem + Cloth Armor + Null-Magic Mantle + 800 g
- Item groups (client): {a0015bc6}(max 1)
- Stats [CLIENT]: Health 200, Armor 20, MR 20, Ability haste 15
- mDataValues [CLIENT]: Cooldown=0, EnemyDetectRange=400, Duration=8, MeleeAuraAttackSpeed=0.3, AuraRange=900, ASDuration=1, MoveSpeed=20, RangedAttackSpeedMultiplier=0.667
- Calculations [CLIENT]: `BuffDuration = Duration(=8)   [ranged holder: x 0.5]`; `AuraAttackSpeed = {eb8d750f}(=0)   [ranged holder: x {ca6deae2}(=0)] (shown as %)`
- Tooltip (client en_US, values substituted): 200 Health |  15 Ability Haste |  20 Armor |  20 Magic ResistFanfare | Slowing or Immobilizing an enemy champion grants Fanfare for [BuffDuration] seconds. Fanfare grants you 20 Move Speed. While you have Fanfare, nearby allies, including yourself, gain [AuraAttackSpeed] Attack Speed.
- Wiki pass "Fanfare": Slow or immobilize an enemy champion empowers you with Fanfare for 8 (melee) / 4 (ranged) seconds, granting you 20 bonus movement speed. While empowered, you and nearby allied champions also gain 30 (melee) / 20 (ranged)% bonus attack speed.

### Banshee's Veil (3102)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 200, sell 2100 (70%) | recipe: Needlessly Large Rod + Verdant Barrier + 200 g
- Item groups (client): 3102(max 1), {548f93b0}(max 1), {2dbc7f6b} | wiki limit: Annul
- Stats [CLIENT]: AP 105, MR 40
- mDataValues [CLIENT]: Cooldown=40
- Tooltip (client en_US, values substituted): 105 Ability Power |  40 Magic ResistAnnul (cd: Cooldown) | Grants a Spell Shield that blocks the next enemy Ability. | Item cooldown is restarted when damage is taken from champions.
- Wiki pass "Annul": Grants a spell shield that blocks the next hostile ability (40 second cooldown, timer restarts upon taking damage from champions).
- Hooks: spell shield (Annul) 40 s cd, reset on champion damage

### Bastionbreaker (2520)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 663, sell 2100 (70%) | recipe: The Brutalizer + Serrated Dirk + 663 g
- Item groups (client): {a4016212}(max 1)
- Stats [CLIENT]: AD 55, Ability haste 15, Lethality 22
- mDataValues [CLIENT]: LethalityAmount=22, Cooldown=20, TakedownWindow=3, BuffDuration=90, RangeModifier=0.8, DoTDuration=3, AbilityDamageRangeMod=0.5
- Calculations [CLIENT]: `DamageCalc = 300 + 25 x Lethality   [ranged holder: x {a99340ef}(=0)]`; `AbilityDamageCalc = 50 + 1.5 x Lethality   [ranged holder: x {d62bdfef}(=0)]`
- Tooltip (client en_US, values substituted): 55 Attack Damage |  22 Lethality |  15 Ability HasteShaped Charge (cd: Cooldown) | Dealing Ability damage to a Champion or Epic Monster deals an additional [AbilityDamageCalc] true damage. | Sabotage | Taking down a champion within 3 seconds of damaging them grants Sabotage for 90 seconds. While you have Sabotage, your next Attack against an Epic Monster or Turret deals an additional [DamageCalc] true damage over 3 seconds.
- Wiki pass "Shaped Charge": Your next instance of ability damage to a champion or epic monster with a champion ability deals 50 (melee) / 25 (ranged) (+ 1.5 (melee) / 0.75 (ranged) per 1 lethality) bonus true damage. (cd 20)
- Wiki pass2 "Sabotage": Scoring a takedown against an enemy champion within 3 seconds of damaging them grants you Sabotage for 90 seconds, empowering your next basic attack against a turret or epic monster to consume the effect to deal 300 (melee) / 240 (ranged) (+ 25 (melee) / 20 (ranged) per 1 lethality) bonus true damage over 3 seconds.
- Hooks: ON_ABILITY_DAMAGE(champ) true dmg (20 s cd); ON_TAKEDOWN Sabotage

### Black Cleaver (3071)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 225, sell 2100 (70%) | recipe: Phage + Kindlegem + Pickaxe + 225 g
- Item groups (client): 3071(max 1), LastWhisper(max 1) | wiki limit: Fatality
- Stats [CLIENT]: Health 400, AD 45, Ability haste 20
- mDataValues [CLIENT]: DebuffDuration=6, MoveSpeedBonus=20, MoveSpeedDuration=2, InternalCD=0.01, MaxStacks=5, ShredPerStack=0.06, RangedMod=0.5
- Calculations [CLIENT]: `MSBonusSplit = MoveSpeedBonus(=20)   [ranged holder: x RangedMod(=0.5)]`
- Tooltip (client en_US, values substituted): 45 Attack Damage |  400 Health |  20 Ability HasteCarve | Dealing physical damage to champions reduces the target's Armor by 6% for 6 seconds. (stacks 5 times). | Fervor | Dealing physical damage grants [MSBonusSplit] Move Speed for 2 seconds.
- Wiki pass "Carve": Dealing physical damage to an enemy champion applies a stack of Carve for 6 seconds, stacking up to 5 times. Each stack inflicts 6% armor reduction, up to 30% at 5 stacks. Non-basic damage dealt on the same target may apply stacks only once per frame.
- Wiki pass2 "Fervor": Dealing physical damage grants you 20 (melee) / 10 (ranged) bonus movement speed for 2 seconds.
- Hooks: ON_DAMAGE_DEALT(physical, champion) Carve stack; ON_DAMAGE_DEALT(physical) Fervor MS [TOP-LANE PRIORITY]
- Implementation: Carve: each physical damage instance to an enemy champion adds 1 stack (max 5, 6 s duration refreshed, 0.01 s per-target ICD), each -6% armor (total armor reduction, multiplicative 'percent armor reduction' stage). Fervor: dealing physical damage grants 20 MS (melee)/10 (ranged) for 2 s.

### Blackfire Torch (2503)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2800, combine 700, sell 1960 (70%) | recipe: Lost Chapter + Fated Ashes + 700 g
- Item groups (client): {1afc0d39}(max 1)
- Flags: active spell BloodthirsterDummySpell
- Stats [CLIENT]: AP 80, Ability haste 20, Mana 600
- mDataValues [CLIENT]: BurnFlatDamagePerSecond=20, APRatio=0.02, BurnDuration=3, MonsterDamageBonus=20, TickFrequency=0.5, APPerStack=0.04, MinionDPS=20, MinionAP=0.02, MonsterDPS=40, MonsterAP=0.02
- Calculations [CLIENT]: `BurnDamagePerSecondCalc = BurnFlatDamagePerSecond(=20) + APRatio(=0.02) x AP`; `MinionBurnCalc = {9adc0c64}(=0) + MinionAP(=0.02) x AP`; `MonsterBurnCalc = {afd69c5a}(=0) + {434396e0}(=0) x AP`
- Tooltip (client en_US, values substituted): 80 Ability Power |  [FlatMPPoolMod] Mana |  20 Ability HasteBaleful Blaze | Damaging Abilities deals [BurnDamagePerSecondCalc] bonus magic damage per second for 3 seconds. | Blackfire | For each enemy champion, epic and large monster affected by your Baleful Blaze, gain 4% Ability Power. Baleful Blaze deals [MinionBurnCalc] magic damage per second to minions and [MonsterBurnCalc] magic damage per second to monsters.
- Wiki pass "Baleful Blaze": Dealing ability damage burns enemies, causing them to take 60/6 (+ 6/6% AP) magic damage every 0.5 seconds over 3 seconds, for a total of 60 (+ 6% AP). Against monsters, the burn deals 10 bonus magic damage per tick, dealing a total of (60/6)+10 (+ 6/6% AP) magic damage per tick for up to 60+(10*6) (+ 6% AP).
- Wiki pass2 "Blackfire": For each champion, epic monster, and large monster afflicted with Baleful Blaze's burn, increase your ability power by 4%.

### Blade of The Ruined King (3153)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3200, combine 725, sell 2240 (70%) | recipe: Vampiric Scepter + Recurve Bow + Pickaxe + 725 g
- Item groups (client): 3153(max 1)
- Stats [CLIENT]: AD 40, Attack speed 25% (bonus AS ratio), Life steal 10%
- mDataValues [CLIENT]: MoveSpeedMod=-0.3, MoveSpeedDuration=1, Cooldown=15, RangedValue=0.06, MeleeValue=0.09, MonsterDamageCap=100, AttackCounterDuration=6, RangedMoveSpeedMod=-0.15
- Calculations [CLIENT]: `SiphonDamage = level_bp(L1=40, +7/level at L>=10)`; `MeleeItemCalcValue = MeleeValue(=0.09) (shown as %)`; `RangedItemCalcValue = RangedValue(=0.06) (shown as %)`; `{c0b34d7a} = MoveSpeedMod(=-0.3)`; `{fe7dca43} = RangedMoveSpeedMod(=-0.15)`; `MeleeItemCalcValueB = (MoveSpeedMod(=-0.3)) * -1 (shown as %)`; `RangedItemCalcValueB = (RangedMoveSpeedMod(=-0.15)) * -1 (shown as %)`; `{d96e60e3} = MeleeValue(=0.09)`; `{42565e68} = RangedValue(=0.06)`; `{79b6144b} = ranged ? calc[{42565e68}] : calc[{d96e60e3}]`; `{405deeb1} = (1 x TargetCurHP?) * calc[{79b6144b}]`
- Tooltip (client en_US, values substituted): 40 Attack Damage |  25% Attack Speed |  10% Life StealMist's Edge | Attacks apply an additional [melee/ranged split: calc MeleeItemCalcValue / RangedItemCalcValue] enemy current Health physical damage . | Clawing Shadows (cd: Cooldown) | Attacking a champion 3 times Slows them by 30% for 1 second. | Maximum Mist's Edge damage dealt to minions and jungle monsters is 100.
- Wiki pass "Mist's Edge": Basic attacks deal bonus physical damage on-hit equal to 9% (melee) / 6% (ranged) of the target's current health, with a maximum of 100 against minions and monsters.
- Wiki pass2 "Clawing Shadows": Basic attacks on-hit against enemy champions apply a stack for 6 seconds, stacking up to 3 times. The third stack consumes them all to slow the target by 30% for 1 second. (cd 15)
- Hooks: ON_HIT % current HP phys; 3-hit slow [TOP-LANE PRIORITY]
- Implementation: Mist's Edge: on-hit physical = 9% (melee)/6% (ranged) of target current HP, capped at 100 vs minions/monsters; applies life steal. Clawing Shadows: 3rd attack vs a champion within 6 s slows 30% (melee)/15% (ranged) 1 s, 15 s cd.

### Bloodletter's Curse (8010)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2900, combine 750, sell 2030 (70%) | recipe: Haunting Guise + Fiendish Codex + 750 g
- Item groups (client): {caaa71d4}, VoidPen(max 1) | wiki limit: Blight
- Stats [CLIENT]: Health 400, AP 65, Ability haste 15
- mDataValues [CLIENT]: ShredPerStack=0.075, DebuffDuration=6, InternalCD=0.3, MaxStacks=4
- Tooltip (client en_US, values substituted): 65 Ability Power |  400 Health |  15 Ability HasteVile Decay | Dealing magic damage with abilities or passives to champions reduces their Magic Resist by 7.5% for 6 seconds. (Stacks 4 times).Each ability cast instance can only apply 1 Vile Decay stack to each champion once every 0.3 second(s).
- Wiki pass "Vile Decay": Dealing magic damage to an enemy champion with a champion ability applies a stack of Vile Decay to them for 6 seconds, stacking up to 4 times and up to once per basic attack or ability per cast instance every 0.3 seconds. Each stack inflicts 7.5% magic resistance reduction, up to 30% at 4 stacks.

### Bloodthirster (3072)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3400, combine 325, sell 2380 (70%) | recipe: B. F. Sword + Pickaxe + Vampiric Scepter + 325 g
- Item groups (client): 3072(max 1)
- Flags: active spell BloodthirsterDummySpell
- Stats [CLIENT]: AD 80, Life steal 15%
- mDataValues [CLIENT]: Threshold=0.7, DecayTime=25
- Calculations [CLIENT]: `OvershieldCalc = level_bp(L1=165, +15/level at L>=9)`
- Tooltip (client en_US, values substituted): 80 Attack Damage |  15% Life StealIchorshield | Convert excess healing from your Lifesteal to a Shield, up to [OvershieldCalc].
- Wiki pass "Ichorshield": Convert the healing received from life steal in excess of maximum health into a shield for up to [levels: 165 + (315-165)/10*(x-1)|1;9 to 20 by 1|formula=165 base, then +15 per level starting from level 9.], which lasts until destroyed.
- Hooks: ON_LIFESTEAL_HEAL overflow -> shield
- Implementation: Ichorshield: excess life-steal healing converted to shield up to 165 (+15/lvl from 9).

### Chempunk Chainsword (6609)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 250, sell 2100 (70%) | recipe: Executioner's Calling + Giant's Belt + Caulfield's Warhammer + 250 g
- Item groups (client): 6609(max 1), {c8a69ca7}
- Stats [CLIENT]: Health 450, AD 45, Ability haste 15
- mDataValues [CLIENT]: GrievousAmount=0.4, GrievousDuration=3
- Tooltip (client en_US, values substituted): 45 Attack Damage |  450 Health |  15 Ability HasteHackshorn | Dealing physical damage applies 40% Wounds to enemy champions for 3 seconds.
- Wiki pass "Hackshorn": Dealing physical damage to enemy champions inflicts them with Grievous Wounds for 3 seconds.
- Hooks: ON_DAMAGE_DEALT(physical, champion) GW [TOP-LANE PRIORITY]

### Cosmic Drive (4629)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 450, sell 2100 (70%) | recipe: Kindlegem + Aether Wisp + Fiendish Codex + 450 g
- Item groups (client): 4629(max 1)
- Stats [CLIENT]: Health 350, AP 70, MS 4% (additive % MS), Ability haste 25
- mDataValues [CLIENT]: StackDuration=4, MaxStacks=3, MovespeedToGrant=40, MaxMovespeedTooltip=0.15, ItemCooldown=10
- Calculations [CLIENT]: `MoveSpeedAmount = 20`
- Tooltip (client en_US, values substituted): 70 Ability Power |  350 Health |  25 Ability Haste |  4% Move SpeedSpelldance | Dealing magic or true damage to champions grants [MovespeedAmount] Move Speed for 4 seconds.
- Wiki pass "Spelldance": Dealing magic or true damage to an enemy champion grants you 20 bonus movement speed for 4 seconds.

### Cryptbloom (3137)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 200, sell 2100 (70%) | recipe: Blighting Jewel + Fiendish Codex + Fiendish Codex + 200 g
- Item groups (client): VoidPen(max 1) | wiki limit: Blight
- Stats [CLIENT]: AP 75, Ability haste 20, Magic pen 30%
- mDataValues [CLIENT]: BaseHeal=100, HealAPRatio=0.2, ItemCooldown=60, TakedownWindow=3, Cooldown=60
- Calculations [CLIENT]: `TotalHealAmount = BaseHeal(=100) + HealAPRatio(=0.2) x AP`
- Tooltip (client en_US, values substituted): 75 Ability Power |  30% Magic Penetration |  20 Ability HasteLife from Death (cd: Cooldown) | When a champion that you damaged within 3 seconds dies, a nova spreads from their corpse that heals for [TotalHealAmount]. | Life From Death cannot trigger while you are dead.
- Wiki pass "Life From Death": Scoring a takedown against an enemy champion while alive and within 3 seconds of damaging them summons a nova that radiates from the location of their death over 1.75 seconds, heal you and allied champions hit for 100 (+ 20% AP). (cd 60)

### Dawncore (6621)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2500, combine 450, sell 1750 (70%) | recipe: Blasting Wand + Forbidden Idol + Forbidden Idol + 450 g
- Item groups (client): {6fcc6138}(max 1)
- Flags: RestrictedBuffName=HeroPassive
- Stats [CLIENT]: AP 45, Base mana regen 100% of base, Heal & shield power 16%
- mDataValues [CLIENT]: APPerManaRegen=10, HSPowerPerManaRegen=0.02
- Tooltip (client en_US, values substituted): 45 Ability Power |  16% Heal and Shield Power |  [PercentBaseMPRegenMod*100]% Base Mana RegenFirst Light | Gain 2% Heal and Shield Power and 10 Ability Power per 100% Base Mana Regen.
- Wiki pass "First Light": Gain 2% heal and shield power and 10 ability power for every additional 100% base mana regeneration.

### Dead Man's Plate (3742)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2900, combine 900, sell 2030 (70%) | recipe: Winged Moonplate + Ruby Crystal + Chain Vest + 900 g
- Item groups (client): 3742(max 1), {db01f901}(max 1) | wiki limit: Momentum
- Stats [CLIENT]: Health 350, Armor 55, MS 4% (additive % MS), Slow resist 15%
- mDataValues [CLIENT]: MaxMovementSpeed=20, MaxStacks=100, BonusDamagePerStack=0.4, MaxStackSlowAmount=0, MaxStackSlowDuration=0, MaxStacksADRatio=1, DurationToMaxStack=4, SlowResistTooltip=0.15
- Calculations [CLIENT]: `MaxDamageCalc = MaxStacksADRatio(=1) x base AD + (BonusDamagePerStack(=0.4))*(100)`
- Tooltip (client en_US, values substituted): 350 Health |  55 Armor |  4% Move SpeedShipwrecker | While moving, build up to 20 bonus Move Speed. Your next Attack discharges built up Move Speed to deal up to [MaxDamageCalc] bonus physical damage. | Unsinkable | Reduce the effectiveness of Slows by 15%.
- Wiki pass "Shipwrecker": While moving, generates 7 stacks of Momentum every 0.25 seconds, granting up to 20 bonus movement speed at 100 stacks after 3.75 seconds of moving. Basic attacks consume all stacks to deal [levels: 0 to 40 for 11 (+ [levels: 0 to 100 for 11 bonus physical damage on-hit.
- Wiki pass2 "Unsinkable": Gain 15% slow resist.
- Hooks: ON_MOVE momentum stacks; ON_HIT discharge [TOP-LANE PRIORITY]
- Implementation: Shipwrecker: see catalog.

### Death's Dance (6333)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3300, combine 275, sell 2310 (70%) | recipe: Steel Sigil + Pickaxe + Caulfield's Warhammer + 275 g
- Item groups (client): 6333(max 1)
- Stats [CLIENT]: AD 60, Armor 50, Ability haste 15
- mDataValues [CLIENT]: BleedDurationWorst=3, HealDuration=2, MS=0.4, TakedownWindow=3, BonusADRatio=0.75
- Calculations [CLIENT]: `HealTotal = BonusADRatio(=0.75) x bonus AD`; `MeleeItemCalcValue = 0.3 (shown as %)`; `RangedItemCalcValue = 0.1 (shown as %)`
- Tooltip (client en_US, values substituted): 60 Attack Damage |  50 Armor |  15 Ability HasteIgnore Pain | [melee/ranged split: calc MeleeItemCalcValue / RangedItemCalcValue] of damage taken is dealt to you over 3 seconds instead. | Defy | When a champion that you damaged within 3 seconds dies, cleanse Ignore Pain's remaining damage and restore [HealTotal] Health over 2 seconds.
- Wiki pass "Ignore Pain": Reduces 30% (melee) / 10% (ranged) of all post-mitigation [Damage calculated after modifiers] physical and magic damage received and instead stores the damage to successively take it as true damage over 3 seconds, dealing a third of the stored damage each second.
- Wiki pass2 "Defy": If an enemy champion dies within 3 seconds of you damaging them, removes Ignore Pain's remaining stored damage and heals you for 75% bonus AD over 2 seconds.
- Hooks: ON_PRE_DAMAGE_TAKEN store 30%/10% as bleed 3 s; ON_TAKEDOWN cleanse+heal [TOP-LANE PRIORITY]
- Implementation: Ignore Pain / Defy.

### Dusk and Dawn (2510)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3100, combine 300, sell 2170 (70%) | recipe: Sheen + Blasting Wand + Kindlegem + Dagger + 300 g
- Item groups (client): {9dff1a09}(max 1), {57352a0f}(max 1) | wiki limit: Spellblade
- Stats [CLIENT]: Health 300, AP 60, Attack speed 20% (bonus AS ratio), Ability haste 20
- mDataValues [CLIENT]: SpellbladeCooldown=1.5, Cooldown=1.5
- Calculations [CLIENT]: `SpellbladeDamage = 0.75 x base AD + 0.1 x AP`; `SpellbladeHealing = 0.1 x AP + 0.03 x bonus MaxHP`
- Tooltip (client en_US, values substituted): 300 Health |  60 Ability Power |  20 Ability Haste |  20% Attack SpeedSpellblade (cd: Cooldown) | After using an Ability, your next Attack deals [SpellbladeDamage] bonus magic damage and heals you for [SpellbladeHealing]  and then applies  effects an additional time.
- Wiki pass "Spellblade": After using an ability, your next basic attack within 10 seconds deals 75% base AD (+ 10% AP) bonus magic damage and heals you for 10% AP (+ 3% bonus health) on-hit, and applies on-hit effects to the target again after a 0.2-second delay (1.5 second cooldown, starts after using the empowered attack).
- Hooks: ON_ABILITY_CAST arm Spellblade; ON_HIT proc magic + heal + extra on-hit application
- Implementation: Spellblade group (shared cooldown SheenDelay 1.5 s).

### Echoes of Helia (6620)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2200, combine 500, sell 1540 (70%) | recipe: Kindlegem + Bandleglass Mirror + 500 g
- Item groups (client): {1410b5c7}(max 1)
- Flags: RestrictedBuffName=HeroPassive
- Stats [CLIENT]: Health 200, AP 35, Ability haste 20, Base mana regen 125% of base
- mDataValues [CLIENT]: DamageStorageRate=0.3, ChargeToHealConversion=1, MinimumHealToTrigger=10
- Calculations [CLIENT]: `MaxCharges = level_bp(L1=80, +10/level from L2)`
- Tooltip (client en_US, values substituted): 35 Ability Power |  200 Health |  20 Ability Haste |  [PercentBaseMPRegenMod*100]% Base Mana RegenSoul Siphon | Gain 30% of pre-mitigation damage dealt to champions as Soul Charges, up to [MaxCharges] Charges. Healing or Shielding an ally consumes all Soul Charges to restore 100% of that value as Health.
- Wiki pass "Soul Siphon": Gain 30% of pre-mitigation damage [Damage calculated before modifiers] dealt to champions as Soul Charges, up to [levels: 80 to 250|tooltipSize=20]. Healing or shielding an allied champion (excluding yourself) consumes all charges to heal them equal to the consumed amount.

### Eclipse (6692)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2900, combine 625, sell 2030 (70%) | recipe: Caulfield's Warhammer + Pickaxe + Long Sword + 625 g
- Item groups (client): {13f4df4a}(max 1)
- Stats [CLIENT]: AD 60, Ability haste 15
- mDataValues [CLIENT]: WindowDuration=2, MeleePercMaxHP=0.08, RangedPercMaxHPMult=0.625, Cooldown=6, ShieldDuration=2, MeleeBaseShield=150, MeleeBonusADShieldRatio=0.4, RangedShieldMult=0.5
- Calculations [CLIENT]: `{d02ea590} = MeleeBaseShield(=150) + {e367e801}(=0) x bonus AD   [ranged holder: x {51df2a01}(=0)]`; `MaxHealthDamageCalc = {b1f09313}(=0)   [ranged holder: x {4b5548be}(=0)] (shown as %)`
- Tooltip (client en_US, values substituted): 60 Attack Damage |  15 Ability HasteEver Rising Moon (cd: Cooldown) | Hitting a champion with 2 separate Attacks or Abilities within 2 seconds deals [MaxHealthDamageCalc] max health physical damage and grants you a [ShieldSplit] Shield for 2 seconds.
- Wiki pass "Ever Rising Moon": Damaging basic attacks, abilities, item effects, and summoner spells, as well as the application of crowd control and damage over time effects, apply stacks against enemy champions, up to one per cast instance per champion. Applying 2 stacks to a champion within a 2 second period deals bonus physical damage to them equal to 8% (melee) / 5% (ranged) of target's maximum health and grants you a shield for 150 (melee) / 75 (ranged) (+ 40% (melee) / 20% (ranged) bonus AD) for 2 seconds. (cd 6)
- Hooks: ON_DAMAGE_DEALT(champion) 2 hits in 2 s -> %maxHP + shield (6 s cd) [TOP-LANE PRIORITY]

### Edge of Night (3814)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 850, sell 2100 (70%) | recipe: Serrated Dirk + Tunneler + 850 g
- Item groups (client): 3814(max 1), {548f93b0}(max 1), {2dbc7f6b} | wiki limit: Annul
- Stats [CLIENT]: Health 250, AD 50, Lethality 15
- mDataValues [CLIENT]: LethalityAmount=15, Cooldown=40
- Tooltip (client en_US, values substituted): 50 Attack Damage |  15 Lethality |  250 HealthAnnul (cd: Cooldown) | Gain a Spell Shield that blocks the next enemy Ability. | Item's cooldown is restarted when damage is taken from champions.
- Wiki pass "Annul": Grants a spell shield that blocks the next hostile ability (40 second cooldown, timer restarts upon taking damage from champions).

### Endless Hunger (2517)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3100, combine 825, sell 2170 (70%) | recipe: Caulfield's Warhammer + Pickaxe + Long Sword + 825 g
- Item groups (client): {9eff1b9c}(max 1)
- Stats [CLIENT]: AD 65, Omnivamp 5%, Tenacity 20%
- mDataValues [CLIENT]: OmnivampOnTakedown=0.15, OmnivampDuration=8, TakedownWindow=3
- Calculations [CLIENT]: `{e4d9f16b} = 5 + 0.13 x bonus AD`; `{87892572} = 5 + 0.1 x bonus AD`
- Tooltip (client en_US, values substituted): 65 Attack Damage |  5% Omnivamp |  20% TenacityFamine | Gain [HasteFromAD] Ability Haste. | Feast | When a champion that you damaged within 3 seconds dies, gain 15% Omnivamp for 8 seconds.
- Wiki pass "Famine": Gain 5 (+ 13% (melee) / 10% (ranged) bonus AD) ability haste.
- Wiki pass2 "Feast": Scoring a takedown against an enemy champion within 3 seconds of damaging them grants you 15% omnivamp for 8 seconds.
- Hooks: STAT_DYN AH = 5 + 13% bAD (melee)/10% (ranged); ON_TAKEDOWN +15% omnivamp 8 s [TOP-LANE PRIORITY]

### Essence Reaver (3508)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3050, combine 500, sell 2135 (70%) | recipe: Sheen + Caulfield's Warhammer + Cloak of Agility + 500 g
- Item groups (client): 3508(max 1), {57352a0f}(max 1) | wiki limit: Spellblade
- Flags: active spell BloodthirsterDummySpell
- Stats [CLIENT]: AD 50, Crit chance 25%, Ability haste 20
- mDataValues [CLIENT]: FlatManaRefund=15, ManaRefundRatio=0.15, BaseADRatio=1.25, CritChanceMultiplier=50, SpellbladeCooldown=1.5, Cooldown=1.5
- Calculations [CLIENT]: `TotalManaRefund = calc[SpellbladeDamage] * 0.5`; `SpellbladeDamage = BaseADRatio(=1.25) x base AD + CritChanceMultiplier(=50) x CritChance`
- Tooltip (client en_US, values substituted): 50 Attack Damage |  20 Ability Haste |  25% Critical Strike ChanceSpellblade (cd: Cooldown) | After using an Ability, your next Attack deals [SpellbladeDamage] bonus physical damage and grants [TotalManaRefund] Mana .
- Wiki pass "Spellblade": After using an ability, your next basic attack within 10 seconds deals 125% base AD (+ [levels: 0 to 50 bonus physical damage on-hit and restores mana equal to(1.5 second cooldown, starts after using the empowered attack).

### Experimental Hexplate (3073)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 500, sell 2100 (70%) | recipe: Tunneler + Dagger + Phage + 500 g
- Item groups (client): {9dcc6220}(max 1)
- Stats [CLIENT]: Health 450, AD 40, Attack speed 20% (bonus AS ratio)
- mDataValues [CLIENT]: UltimateHaste=30, MovementSpeedBonus=0.2, HasteDuration=8, Cooldown=30, BonusASRanged=35, BonusASMelee=50, BonusMSRanged=14, BonusMSMelee=20
- Tooltip (client en_US, values substituted): 40 Attack Damage |  20% Attack Speed |  450 HealthHexcharged | Gain 30 Ultimate Ability Haste. | Overdrive (cd: Cooldown) | After casting your Ultimate, gain [BonusAS]% Attack Speed and [BonusMS]% Move Speed for 8 seconds.
- Wiki pass "Hexcharged": Gain 30 ultimate haste.
- Wiki pass2 "Overdrive": Upon casting your ultimate ability, enter Overdrive to gain 50% (melee) / 35% (ranged) bonus attack speed and 20% (melee) / 14% (ranged) bonus movement speed for 8 seconds (30 second cooldown, starts on ultimate cast).
- Hooks: ON_ULT_CAST AS/MS buff

### Fiendhunter Bolts (2512)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2650, combine 850, sell 1855 (70%) | recipe: Zeal + Scout's Slingshot + 850 g
- Item groups (client): {9bff16e3}(max 1)
- Stats [CLIENT]: Attack speed 45% (bonus AS ratio), Crit chance 25%, MS 4% (additive % MS)
- mDataValues [CLIENT]: UltimateHaste=30, Cooldown=45, Duration=8, BonusTrueDamage=0.15, BonusAS=0.5, NumberOfAttacks=3, CritModifier=0.8
- Tooltip (client en_US, values substituted): 45% Attack Speed |  25% Critical Strike Chance |  4% Move SpeedNight Vigil | Gain 30 Ultimate Ability Haste. | Opening Barrage (cd: Cooldown) | After casting your Ultimate, your next 3 basic attacks within 8 seconds gain 50% Attack Speed and Critically Strike for 80% of your normal Critical Strike damage. If an attack would already Critically Strike, it deals normal Critical Strike damage and also deals 15% bonus true damage.
- Wiki pass "Night Vigil": Gain 30 ultimate haste.
- Wiki pass2 "Opening Barrage": After casting your ultimate ability, your next 3 basic attacks on-attack within 8 seconds gain 50% bonus attack speed and are empowered to critically strike forIf an attack would have already critically struck, it instead critically strikes forand deals bonus true damage equal to 15% of the triggering attack's damage pre-mitigation [Damage calculated before modifiers]. (cd 45)

### Force of Nature (4401)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2800, combine 750, sell 1960 (70%) | recipe: Negatron Cloak + Ruby Crystal + Winged Moonplate + 750 g
- Item groups (client): 4401(max 1)
- Stats [CLIENT]: Health 400, MR 55, MS 4% (additive % MS)
- mDataValues [CLIENT]: BuffDuration=7, MoveSpeed=0.06, MaxStacks=8, ImmobilizeStacks=2, DamageReduction=0, StackRefreshTimer=1, BonusMagicResist=70
- Tooltip (client en_US, values substituted): 400 Health |  55 Magic Resist |  4% Move SpeedSteadfast | Gain 70 Magic Resist and 6% bonus Move Speed after taking magic damage from Champions 8 times. | Immobilizing effects count as 2 instances of damage. | Steadfast resets after 7 seconds.
- Wiki pass "Steadfast": Taking magic damage from champions generates a stack of Steadfast for 7 seconds, stacking up to 8 times with the duration refreshing on subsequent magic damage from them and whenever dealing damage to them. Becoming immobilize by an enemy champion generates 2 stacks and also refreshes the duration. Once per cast instance, each incoming basic attack, ability, or item effect can only generate 1 stack of Steadfast from their damage every 1 second. At maximum stacks, gain 70 bonus magic resistance and 6% bonus movement speed.
- Hooks: ON_DAMAGE_TAKEN(magic, champion) stacks

### Frozen Heart (3110)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2500, combine 600, sell 1750 (70%) | recipe: Warden's Mail + Glacial Buckler + 600 g
- Item groups (client): 3110(max 1)
- Stats [CLIENT]: Armor 75, Ability haste 20, Mana 400
- mDataValues [CLIENT]: ASPDSlow=-0.2, AuraRadius=700
- Tooltip (client en_US, values substituted): 75 Armor |  [FlatMPPoolMod] Mana |  20 Ability HasteWinter's Caress | Reduce the Attack Speed of nearby champions by 20%.
- Wiki pass "Winter's Caress": Cripple the attack speed of enemy champions within 700 units [center to edge] by 20%.
- Hooks: AURA -20% AS enemy champions 700 [TOP-LANE PRIORITY]
- Implementation: Winter's Caress (cripple) on nearby enemy champions within 700.

### Guardian Angel (3026)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3200, combine 800, sell 1280 (40%) | recipe: Steel Sigil + B. F. Sword + 800 g
- Item groups (client): 3026(max 1)
- Flags: active spell GuardianAngel
- Stats [CLIENT]: AD 55, Armor 45
- mDataValues [CLIENT]: Cooldown=300
- Tooltip (client en_US, values substituted): 55 Attack Damage |  45 ArmorRebirth (cd: Cooldown) | Upon taking lethal damage, restores [Effect1Amount*100]% base Health and [Effect4Amount*100]% max Mana after [Effect2Amount] seconds of Stasis.
- Wiki pass "Rebirth": Upon taking lethal damage, enter resurrection for 4 seconds, during which you are invulnerable, untargetable, and unable to act, and afterwards heal for 50% of base health and restore 100% of maximum mana (300 second cooldown, starts after resurrection ends).
- Hooks: ON_LETHAL_DAMAGE revive (300 s cd) [TOP-LANE PRIORITY]
- Implementation: Rebirth: on lethal damage, enter 4 s stasis then revive with 50% base HP and 100% max mana [client mEffectAmount].

### Guinsoo's Rageblade (3124)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 1025, sell 2100 (70%) | recipe: Amplifying Tome + Recurve Bow + Pickaxe + 1025 g
- Item groups (client): {0eb22e4f}(max 1)
- Stats [CLIENT]: AD 30, AP 30, Attack speed 25% (bonus AS ratio)
- mDataValues [CLIENT]: OnHitDamage=30, MythicArmorPen=0, MythicMagicPen=0, AttackSpeedPerStack=0.08, MaxStacks=4, BuffDuration=4, CritChancePerStep=0, DamageAmountPerCritStep=0, MaxDamageFromCrit=0
- Calculations [CLIENT]: `MaxAttackSpeedCalc = (AttackSpeedPerStack(=0.08))*(MaxStacks(=4)) (shown as %)`; `TotalMagicPen = (legendary_count) * MythicMagicPen(=0) (shown as %)`; `TotalArmorPen = (legendary_count) * MythicArmorPen(=0) (shown as %)`; `{592c02e8} = OnHitDamage(=30)`
- Tooltip (client en_US, values substituted): 30 Attack Damage |  30 Ability Power |  25% Attack SpeedWrath | Attacks deal 30 bonus magic damage . | Seething Strike | Attacks grant 8% Attack Speed for 4 seconds. (stacks 4 times).  | While fully stacked, every third Attack applies  effects twice.
- Wiki pass "Wrath": Basic attacks deal 30 bonus magic damage on-hit.
- Wiki pass2 "Seething Strike": Basic attacks on-attack grant 8% bonus attack speed for 4 seconds, stacking up to 4 times for a total of 32% bonus attack speed. At maximum stacks, basic attacks on-attack also grant a Phantom stack for 4 seconds, up to 2 stacks. At 2 Phantom stacks, the next basic attack consumes all of those stacks on-attack to trigger a Phantom Hit that applies on-hit effects to the target after a 0.15-second delay.

### Heartsteel (3084)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 400, sell 2100 (70%) | recipe: Giant's Belt + Crystalline Bracer + Giant's Belt + 400 g
- Item groups (client): {b2ecd6da}(max 1)
- Stats [CLIENT]: Health 900, Base HP regen 100% of base
- mDataValues [CLIENT]: PerTargetCooldown=30, BaseDamage=70, HPRatio=0.06, DamageToMaxHealthRatio=0.1, DistanceToChampion=700, NumTicksToTrigger=6, RangeTrackingBuffDuration=5, TrackerTickRate=0.5, MaxHPRatio=0.06, HealthSizeThreshold=1000, SizeAmount=0.03, SizeCap=0.3, Cooldown=30
- Calculations [CLIENT]: `DamageProcCalc = BaseDamage(=70) + HPRatio(=0.06) x MaxHP`; `ProcHealthGain = calc[DamageProcCalc] * DamageToMaxHealthRatio(=0.1)`; `TotalDemolishTime = (TrackerTickRate(=0.5)) * NumTicksToTrigger(=6)`; `DamageCalc = MaxHPRatio(=0.06) x MaxHP + BaseDamage(=70)`
- Tooltip (client en_US, values substituted): 900 Health |  100% Base Health RegenColossal Consumption (cd: Cooldown) per target | If an enemy champion is nearby for [TotalDemolishTime] seconds, your next Attack against them deals 70 plus 6% of your max Health as bonus physical damage and grants 10% of the damage as max Health. | Goliath | For each 1000 max Health, gain 3% increased size, up to 30%.
- Wiki pass "Colossal Consumption": While within 700 units of an enemy champion, generate a stack on them each second, stacking up to 3 times. Your next basic attack against a target with 3 stacks is empowered to consume them all to deal 70 (+ 6% maximum health) bonus physical damage on-hit and grant you permanent bonus health equal to (30 second cooldown per target).
- Wiki pass2 "Goliath": Gain [levels: key=% increased size.
- Hooks: PERIODIC(0.5 s) stack per nearby enemy champ; ON_HIT proc [TOP-LANE PRIORITY]
- Implementation: Colossal Consumption: while an enemy champion is within 700 for 6 ticks of 0.5 s (3 s) the next basic attack against it deals 70 + 6% max HP bonus physical and grants permanent max HP = 10% of the proc damage (pre-mitigation per tooltip calc); 30 s per-target cooldown. Goliath: +3% size per 1000 max HP, cap 30%.

### Hexoptics C44 (2523)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2800, combine 275, sell 1960 (70%) | recipe: Pickaxe + Noonquiver + Long Sword + 275 g
- Item groups (client): {a301607f}(max 1)
- Stats [CLIENT]: AD 55, Crit chance 25%
- mDataValues [CLIENT]: ExtraRange=100, TakedownWindow=3, Duration=8, MaxRange=500, MaxDamageAmp=0.1
- Tooltip (client en_US, values substituted): 55 Attack Damage |  25% Critical Strike ChanceMagnification | Deal up to 10% increased damage with Attacks, based on how far away the enemy is (max damage at 500 range). | Arcane Aim | When a champion that you damaged within 3 seconds dies, gain 100 additional attack range for 8 seconds.
- Wiki pass "Magnification": Deal [levels: 0 to 10 for 11|0 to 500|key=%|type=distance to target|formula=1% per 50 units away, up to a maximum of 10%.] increased basic damage. The distance is calculated from the er edge of your current position to edge of the target's position at the time they are damaged.
- Wiki pass2 "Arcane Aim": Scoring a takedown against an enemy champion within 3 seconds of damaging them grants you 100 bonus attack range for 8 seconds.

### Hextech Gunblade (3146)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 600, sell 2100 (70%) | recipe: Vampiric Scepter + Hextech Alternator + Amplifying Tome + 600 g
- Item groups (client): 3146(max 1)
- Flags: active spell HextechGunbladeSpell
- Stats [CLIENT]: AD 40, AP 80, Omnivamp 10%
- mDataValues [CLIENT]: SlowAmount=0.25, SlowDuration=1.5, Cooldown=60
- Calculations [CLIENT]: `ActiveDamage = lerp_level(175 -> 253) + 0.3 x AP`
- Tooltip (client en_US, values substituted): 80 Ability Power |  40 Attack Damage |  10% Omnivamp (cd: Cooldown) | Lightning Bolt | Shocks the target enemy champion, dealing [ActiveDamage] magic damage and slowing them by 25% for 1.5 seconds.
- Wiki act "Lightning Bolt": Shocks the target enemy champion with a bolt of lightning, dealing [levels: 175 to 253 (+ 30% AP) magic damage and slow them by 25% for 1.5 seconds. (cd 60)

### Hextech Rocketbelt (3152)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2650, combine 350, sell 1855 (70%) | recipe: Hextech Alternator + Kindlegem + Ruby Crystal + 350 g
- Item groups (client): {12b47332}(max 1)
- Flags: active spell 3152Active
- Stats [CLIENT]: Health 350, AP 60, Ability haste 20
- mDataValues [CLIENT]: Cooldown=50, BaseDamage=100, APRatio=0.1
- Calculations [CLIENT]: `FireboltDamage = BaseDamage(=100) + APRatio(=0.1) x AP`
- Tooltip (client en_US, values substituted): 60 Ability Power |  350 Health |  20 Ability Haste Supersonic (cd: Cooldown) | Dash in target direction, unleashing missiles that deal [FireboltDamage] magic damage. | Supersonic's dash cannot pass through terrain.
- Wiki act "Supersonic": Dash 275 units in the target direction, though not through terrain, then unleash an arc of 7 rockets forward which travel up to 1050 units; and upon collision with an enemy or terrain, explode in a cr 185-radius [estimated] area. Enemies within 85 units [center-to-edge, estimated] of your dash and ones hit by any rocket's explosion are dealt 100 (+ 10% AP) magic damage, once per cast. Supersonic basic attack reset the user's basic attack timer. (cd 50)

### Hollow Radiance (6664)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2800, combine 650, sell 1960 (70%) | recipe: Bami's Cinder + Spectre's Cowl + 650 g
- Item groups (client): 6664(max 1), ImmolateItems(max 1) | wiki limit: Immolate
- Stats [CLIENT]: Health 400, MR 40, Ability haste 10, Base HP regen 100% of base
- mDataValues [CLIENT]: MinionMod=0.25, MonsterMod=0.25, Range=325, AuraDuration=3, TicksPerSecond=1, BaseDamagePerTickTOOLTIPONLY=10, HPRatioPerTickTOOLTIPONLY=1.75, Cooldown=12, ProcAoE=350, ProcDPSMultiplier=2, ChampProcDPSMultiplier=4, TakedownWindow=3, ChampProcAoE=500
- Calculations [CLIENT]: `DamagePerTick = 15 + 0.01 x bonus MaxHP`; `DPS = calc[DamagePerTick] * TicksPerSecond(=1)`; `{049cea52} = calc[DPS] * 1`; `ProcDamageTOOLTIPONLY = calc[DamagePerTick] * ProcDPSMultiplier(=2)`; `{f002950e} = calc[DamagePerTick] * ChampProcDPSMultiplier(=4)`
- Tooltip (client en_US, values substituted): 400 Health |  40 Magic Resist |  10 Ability Haste |  100% Base Health RegenImmolate | After taking or dealing damage, deal [DPS] magic damage per second to nearby enemies for 3 seconds.  | Desolate | Killing an enemy deals [ProcDamageTOOLTIPONLY] magic damage around them. | Immolate deals 25% increased damage to minions and 25% increased damage to monsters.
- Wiki pass "Immolate": Taking or dealing damage activates this passive for 3 seconds. Deal {{as| {{as|(+ % bonus health)}} magic damage|magic damage}} every second to enemies within cr 325 (+ 100% bonus size) units, with the damage being increased to 125% against minions and monsters. This executes minions that would be killed by one more tick of damage.
- Wiki pass2 "Desolate": Killing a non-champion unit [Excluding wards and structures.] causes an eruption around their death location that deals{{ft|{{as|{{ap|*2}} {{as|(+ {{ap|*2}}% bonus health)}} magic damage|magic damage}}|200% of Immolate's damage}}to enemies within 350 units. Scoring a takedown against an enemy champion within 3 seconds of damaging them causes a larger eruption that deals{{ft|{{as|{{ap|*4}} {{as|(+ {{ap|*4}}% bonus health)}} magic damage|magic damage}}|400% of Immolate's damage}}to enemies within 500 units.
- Hooks: Immolate; ON_KILL Desolate burst [TOP-LANE PRIORITY]

### Horizon Focus (4628)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2700, combine 600, sell 1890 (70%) | recipe: Fiendish Codex + Fiendish Codex + Amplifying Tome + 600 g
- Item groups (client): 4628(max 1)
- Stats [CLIENT]: AP 75, Ability haste 25
- mDataValues [CLIENT]: BuffDuration=6, SnipeRange=600, ItemCooldown=30, VisionRadius=1400, VisionDuration=2, SecondaryBuffDuration=3, Cooldown=30, DamageAmp=0.1
- Tooltip (client en_US, values substituted): 75 Ability Power |  25 Ability HasteHypershot | Dealing Ability damage to champions at 600 range or greater Reveals them for 6 seconds. Deal 10% increased damage to enemies Revealed by Hypershot. | Focus (cd: Cooldown) | When Hypershot is triggered, Reveal all other enemy champions within 1400 range of them for 3 seconds.Damage from Pets and traps do not trigger Hypershot.  | Only the initial placement of zone Abilities trigger Hypershot.  | Distance is calculated from Ability cast position.
- Wiki pass "Hypershot": Dealing ability damage to a champion with a champion ability at cr 600 or more units away from the cast position marks them for 6 seconds, standard sight them and increasing your damage dealt to them by 10%.
- Wiki pass2 "Focus": Upon triggering Hypershot, grant sight of the area [Grants sight through terrain and brush] cr 1400 units around the target for 2 seconds and apply Hypershot's mark to enemy champions within the area for 3 seconds. (cd 30)

### Hubris (6697)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2800, combine 750, sell 1960 (70%) | recipe: Serrated Dirk + Caulfield's Warhammer + 750 g
- Item groups (client): {10f4da91}(max 1)
- Stats [CLIENT]: AD 55, Ability haste 10, Lethality 18
- mDataValues [CLIENT]: LethalityAmount=18, MSDuration=3, MinHealthThreshold=0.3, SpellMaxAmp=0, TakedownWindow=3, SpeedAmount=0.3, BonusLethality=15, ADPerStatue=3, BuffDuration=90, BaseADBonus=12
- Calculations [CLIENT]: `{5974e68a} = 1 x stacks`
- Tooltip (client en_US, values substituted): 55 Attack Damage |  18 Lethality |  10 Ability HasteEminence | When a champion that you damaged within 3 seconds dies, gain 12 Attack Damage plus 3 per champion killed for 90 seconds.
- Wiki pass "Eminence": Scoring a takedown against an enemy champion within 3 seconds of damaging them generates a permanent stack and grants you 12 (+ 3 per stack) bonus attack damage for 90 seconds.

### Hullbreaker (3181)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 175, sell 2100 (70%) | recipe: Tunneler + Winged Moonplate + Pickaxe + 175 g
- Item groups (client): 3181(max 1), DisabledOnFIRSTBLOODMode
- Stats [CLIENT]: Health 500, AD 40, MS 4% (additive % MS)
- mDataValues [CLIENT]: SkipperADRatio=1.2, SkipperADRatioVSStructures=3, MaxStackDamageHPRatio=0.05, MaxStackDamageVSStructuresHPRatio=0.1, SkipperStackDuration=10, RangedSkipperADRatio=0.7, RangedSkipperADRatioVSStructures=2
- Calculations [CLIENT]: `MaxStackDamage = SkipperADRatio(=1.2) x base AD + MaxStackDamageHPRatio(=0.05) x MaxHP   [ranged holder: x 0.7]`; `MaxStackDamageVSStructures = SkipperADRatioVSStructures(=3) x base AD + MaxStackDamageVSStructuresHPRatio(=0.1) x MaxHP   [ranged holder: x 0.7]`; `BonusMinionResists = level_bp(L1=70, +6/level at L>=9)   [ranged holder: x 0.5]`
- Tooltip (client en_US, values substituted): 40 Attack Damage |  500 Health |  4% Move SpeedSkipper | Every fifth Attack against champions and epic monsters deals [MaxStackDamage] bonus physical damage, increased to [MaxStackDamageVSStructures] against structures. | Boarding Party | Nearby allied siege and super minions gain [BonusMinionResists] Armor and Magic Resist.
- Wiki pass "Skipper": Basic attacks on-hit against any enemy grant a stack for 10 seconds, stacking up to 5 times. At maximum stacks, or 4 stacks, your next basic attack on-hit against a champion, epic monster, or structure consumes all stacks to deal 120% (melee) / 120*0.7% (ranged) base AD (+ 5% (melee) / 5*0.7% (ranged) maximum health) bonus physical damage, increased to 300% (melee) / 300*0.7% (ranged) base AD (+ 10% (melee) / 10*0.7% (ranged) maximum health) against structures.
- Wiki pass2 "Boarding Party": Allied Blue Siege Minion and Blue Super Minion within er 1050 [Estimated] units gain bonus armor and bonus magic resistance, as well as 10% increased size.
- Hooks: ON_HIT counter 5th attack; AURA siege/super minion resists [TOP-LANE PRIORITY]
- Implementation: Skipper: every 5th basic attack vs champions/epic monsters (stack window 10 s) deals 120% base AD + 5% max HP bonus physical, vs structures 300% base AD + 10% max HP; ranged holders x0.7 on the whole formula (client mRangedMultiplier). Boarding Party: nearby allied siege & super minions gain 70 (+6/lvl from 9) armor & MR (ranged holder x0.5).

### Iceborn Gauntlet (6662)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2900, combine 800, sell 2030 (70%) | recipe: Sheen + Ruby Crystal + Chain Vest + 800 g
- Item groups (client): {7e06cf49}(max 1), {57352a0f}(max 1) | wiki limit: Spellblade
- Flags: active spell 6662_DummySpell
- Stats [CLIENT]: Health 300, Armor 50, Ability haste 15
- mDataValues [CLIENT]: SlowAmount=0.25, SlowFieldDuration=2, AoERadius=300, MonsterMod=1.5, AuraDuration=3, RangedSlowAmount=0.125, SpellbladeMultiplier=1.5, SpellbladeCooldown=1.5, Cooldown=1.5
- Calculations [CLIENT]: `MeleeItemCalcValue = SlowAmount(=0.25) (shown as %)`; `RangedItemCalcValue = RangedSlowAmount(=0.125) (shown as %)`; `SpellbladeDamage = SpellbladeMultiplier(=1.5) x base AD`
- Tooltip (client en_US, values substituted): 300 Health |  50 Armor |  15 Ability HasteSpellblade (cd: Cooldown) | After using an Ability, your next Attack deals [SpellbladeDamage] bonus physical damage  and creates a frost field for 2s that Slows by [SlowAmountMeleeRangedSplit].
- Wiki pass "Spellblade": After using an ability, your next basic attack within 10 seconds deals 150% base AD bonus physical damage on-hit and creates a 300 radius frost field for 2 seconds. Enemies within the field are slowed by 25% (melee) / 25*0.5% (ranged) (1.5 second cooldown, starts after using the empowered attack).
- Hooks: ON_ABILITY_CAST arm; ON_HIT spellblade + frost field [TOP-LANE PRIORITY]

### Immortal Shieldbow (6673)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 825, sell 2100 (70%) | recipe: Pickaxe + Noonquiver + 825 g
- Item groups (client): LifelineItems(max 1), 6673(max 1) | wiki limit: Lifeline
- Stats [CLIENT]: AD 55, Crit chance 25%
- mDataValues [CLIENT]: ShieldDuration=3, Cooldown=90, BuffDuration=8, HealthThreshold=0.3
- Calculations [CLIENT]: `ShieldAmount = level_bp(L1=400, +30/level at L>=9)   [ranged holder: x 0.8]`
- Tooltip (client en_US, values substituted): 55 Attack Damage |  25% Critical Strike ChanceLifeline (cd: Cooldown) | Taking damage that would reduce your Health below 30% grants a [ShieldAmount] Shield for 3 seconds.
- Wiki pass "Lifeline": If you would take damage that would reduce you below 30% of your maximum health, you first gain a shield that absorbs damage for 3 seconds. (cd 90)

### Imperial Mandate (4005)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2400, combine 700, sell 1680 (70%) | recipe: Amplifying Tome + Bandleglass Mirror + Amplifying Tome + 700 g
- Item groups (client): 4005(max 1)
- Stats [CLIENT]: AP 60, Ability haste 15, Base mana regen 150% of base
- mDataValues [CLIENT]: DamageAmp=0.07, DamageAmpDuration=4, ImmobilizingAbilityAH=20
- Tooltip (client en_US, values substituted): 60 Ability Power |  15 Ability Haste |  [PercentBaseMPRegenMod*100]% Base Mana RegenControl | Gain  20 Ability Haste for your abilities with Immobilizing effects. | Command | On Immobilizing an enemy champion, mark them as 7% Vulnerable for 4 seconds. | Immobilizing enemies you have already marked Vulnerable by Command will extend the effect rather than stacking the amplification
- Wiki pass "Control": Abilities with immobilize effects have their cooldown reduced equivalent to 20 ability haste.
- Wiki pass2 "Command": Immobilize an enemy champion marks them as Vulnerable for 4 seconds, increasing the damage they take from all sources by 7%. Subsequent immobilizes against a target extend the duration of the effect.

### Infinity Edge (3031)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3500, combine 725, sell 2450 (70%) | recipe: B. F. Sword + Pickaxe + Cloak of Agility + 725 g
- Item groups (client): {37d64eea}(max 1), {5ddfe837}(max 1)
- Stats [CLIENT]: AD 75, Crit chance 25%, Crit damage 30% (additive to 200% base)
- Tooltip (client en_US, values substituted): 75 Attack Damage |  25% Critical Strike Chance |  30% Critical Strike Damage
- Hooks: STAT (+30% crit damage)

### Jak'Sho, The Protean (6665)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3200, combine 650, sell 2240 (70%) | recipe: Giant's Belt + Chain Vest + Negatron Cloak + 650 g
- Item groups (client): {7f06d0dc}(max 1)
- Stats [CLIENT]: Health 350, Armor 45, MR 45
- mDataValues [CLIENT]: BonusResistPercentage=0.3, MaxStacks=5
- Tooltip (client en_US, values substituted): 350 Health |  45 Armor |  45 Magic ResistVoidborn Resilience | After 5 seconds of champion combat, increase your bonus Armor and Magic Resist by 30% until end of combat.
- Wiki pass "Voidborn Resilience": Gain a stack for each second in combat with enemy champions, stacking up to 5 times. At maximum stacks, increase your bonus armor and bonus magic resistance by 30% until the end of combat.
- Hooks: PERIODIC champion-combat seconds -> +30% bonus resists [TOP-LANE PRIORITY]

### Kaenic Rookern (2504)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2900, combine 800, sell 2030 (70%) | recipe: Spectre's Cowl + Negatron Cloak + 800 g
- Item groups (client): {1bfc0ecc}(max 1)
- Stats [CLIENT]: Health 400, MR 80, Base HP regen 100% of base
- mDataValues [CLIENT]: ShieldAmount=0.15, OutOfCombatDuration=15
- Calculations [CLIENT]: `ShieldCalc = ShieldAmount(=0.15) x MaxHP`
- Tooltip (client en_US, values substituted): 400 Health |  80 Magic Resist |  100% Base Health RegenMagebane | After not taking magic damage for 15 seconds, gain a [ShieldCalc] magic shield.
- Wiki pass "Magebane": After not taking magic damage for 15 seconds, gain a shield that absorbs magic damage equal to 15% of maximum health until destroyed.
- Hooks: ON_DAMAGE_TAKEN(magic) resets 15 s timer; magic shield = 15% max HP [TOP-LANE PRIORITY]
- Implementation: Magebane: after 15 s without taking magic damage gain a magic-only shield of 15% max HP (persist until broken - INFERRED).

### Knight's Vow (3109)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2300, combine 400, sell 1610 (70%) | recipe: Kindlegem + Chain Vest + Rejuvenation Bead + 400 g
- Item groups (client): 3109(max 1)
- Flags: active spell ItemKnightsVow
- Stats [CLIENT]: Health 200, Armor 40, Ability haste 10, Base HP regen 100% of base
- mDataValues [CLIENT]: DamageRedirection=0.14, DamageRedirectionThreshold=0.3, TetherRange=1250, AllyHealingConversion=0.12, Cooldown=60
- Tooltip (client en_US, values substituted): 200 Health |  40 Armor |  10 Ability Haste |  100% Base Health RegenSacrifice | While near your Worthy ally, take 14% of the damage they receive and heal for 12% of the damage they deal to champions. Pledge (cd: Cooldown) | Designate an ally as Worthy.
- Wiki act "Pledge": Designate the target allied champion as being Worthy, forming a tether between you and them. Champions can only be designated as Worthy by one Knight's Vow at a time. You cannot be designated as Worthy by an ally's Knight's Vow once you've formed a tether. (cd 60)
- Wiki pass "Sacrifice": While your Worthy ally is tethered to you [1250 units, center to edge] and you are above 30% of your maximum health, redirect 14% of the pre-mitigation [Damage calculated before the Worthy ally's modifiers] physical and magic damage they take to you as the respective damage type. Additionally, you heal for 12% of the post-mitigation damage [Damage calculated after modifiers] dealt by your Worthy ally to champions.

### Kraken Slayer (6672)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 325, sell 2100 (70%) | recipe: Rectrix + Hearthbound Axe + Recurve Bow + 325 g
- Item groups (client): 6672(max 1)
- Stats [CLIENT]: AD 45, Attack speed 40% (bonus AS ratio), MS 4% (additive % MS)
- mDataValues [CLIENT]: AttackCount=3, BuffDuration=4, MaxAmpNumber=1.75, RangedDamageMultiplier=0.8
- Calculations [CLIENT]: `DamageAmount = level_bp(L1=150, +5/level at L>=9)   [ranged holder: x RangedDamageMultiplier(=0.8)]`; `MaximumDamage = calc[DamageAmount] * MaxAmpNumber(=1.75)`
- Tooltip (client en_US, values substituted): 45 Attack Damage |  40% Attack Speed |  4% Move SpeedBring It Down | Every third Attack deals [DamageAmount] bonus physical damage , increased up to [MaximumDamage] based on their missing Health.
- Wiki pass "Bring It Down": Basic attacks on-hit (melee) / on-attack (ranged) grant a stack for 4 seconds, up to 2 stacks. At 2 stacks, the next basic attack consumes all stacks to deal {{as| bonus physical damage|physical damage}} on-hit, increased by [levels: 0 to 75 by 5|0 to 100|key=%|color=health|type=target's missing health|key1=%], for up to {{as| bonus physical damage|physical damage}}. Abilities that do not trigger on-attack effects, will always grant stacks on-hit.

### Liandry's Torment (6653)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 800, sell 2100 (70%) | recipe: Haunting Guise + Fated Ashes + 800 g
- Item groups (client): {0cff61a1}(max 1)
- Stats [CLIENT]: Health 300, AP 60
- mDataValues [CLIENT]: BurnDuration=3, BurnPercentHealthDamage=0.02, DamageIncreasePerSecond=0.02, DamageIncreaseMax=0.06, TickFrequency=0.5, MonsterDamageCap=40, MaxDamageHPThreshold=1250, BuffCounterDuration=3, MaxStackNumber=3
- Tooltip (client en_US, values substituted): 60 Ability Power |  300 HealthTorment | Damaging Abilities burn enemies for 2% max Health magic damage per second for 3 seconds. | Suffering | For each second in combat with enemy champions, deal 2% bonus damage, up to 6%.
- Wiki pass "Torment": Dealing ability damage or pet damage burns enemies, causing them to take
- Wiki pass2 "Suffering": For each second in combat with enemy champions, deal 2% increased damage, stacking up to 3 times for a total of 6%.

### Lich Bane (3100)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2900, combine 250, sell 2030 (70%) | recipe: Sheen + Aether Wisp + Blasting Wand + 250 g
- Item groups (client): 3100(max 1), {57352a0f}(max 1) | wiki limit: Spellblade
- Stats [CLIENT]: AP 100, MS 6% (additive % MS), Ability haste 10
- mDataValues [CLIENT]: SpellbladeCooldown=1.5, LichBaneAPValue=0.45, SpellbladeADRatio=0.75, SheenASBuff=0.5, Cooldown=1.5, SpellBladeDuration=10
- Calculations [CLIENT]: `SpellbladeDamage = SpellbladeADRatio(=0.75) x base AD + LichBaneAPValue(=0.45) x AP`
- Tooltip (client en_US, values substituted): 100 Ability Power |  6% Move Speed |  10 Ability HasteSpellblade (cd: Cooldown) | After using an Ability, your next Attack within 10 seconds gains 50% Attack Speed and deals [SpellbladeDamage] bonus magic damage .
- Wiki pass "Spellblade": After using an ability, your next basic attack within 10 seconds gains 50% bonus attack speed and deals 75% base AD (+ 45% AP) bonus magic damage on-hit (1.5 second cooldown, starts after using the empowered attack).
- Hooks: ON_ABILITY_CAST arm; ON_HIT magic

### Locket of the Iron Solari (3190)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2200, combine 700, sell 1540 (70%) | recipe: Kindlegem + Cloth Armor + Null-Magic Mantle + 700 g
- Item groups (client): {889717e0}(max 1)
- Flags: active spell 3190Active
- Stats [CLIENT]: Health 200, Armor 30, MR 30, Ability haste 10
- mDataValues [CLIENT]: ShieldDuration=2.5, DiminishedEffectMulitplier=0.25, DiminishedTimer=20, Cooldown=90, ShieldRange=850, ShieldMinTOOLTIP=290, ShieldMaxTOOLTIP=360
- Calculations [CLIENT]: `ShieldAmount = level_bp(L1=290, +7/level at L>=9)`
- Tooltip (client en_US, values substituted): 200 Health |  30 Armor |  30 Magic Resist |  10 Ability Haste Devotion (cd: Cooldown) | Grant nearby allies a 290 - 360 (ally ) Shield that decays over 2.5 seconds. Subsequent Devotion shields within 20 seconds have 25% effect.
- Wiki act "Devotion": Grants you and allied champions within cr 850 units a shield for [levels: 290 to 360 for 11|1;9 to 18|type=target's level] that decays over 2.5 seconds. (cd 90)

### Lord Dominik's Regards (3036)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3300, combine 550, sell 2310 (70%) | recipe: Last Whisper + Noonquiver + 550 g
- Item groups (client): LastWhisper(max 1) | wiki limit: Fatality
- Stats [CLIENT]: AD 35, Crit chance 25%, Armor pen 35%
- mDataValues [CLIENT]: MaxBonusDamagePercent=0.15, MaxBonusHealth=1500
- Tooltip (client en_US, values substituted): 35 Attack Damage |  35% Armor Penetration |  25% Critical Strike ChanceGiant Slayer | Deal up to 15% bonus damage against champions based on their bonus Health. Maximum damage bonus reached at 1500 bonus Health.
- Wiki pass "Giant Slayer": Deal [levels: 0 to 15 for 16|0 to 1500|key=%|type=target's bonus health|formula=1% per 100 bonus health, up to a maximum of 15% at 1500 bonus health.|color=health] increased damage against enemy champions.
- Hooks: ON_DAMAGE_DEALT(champion) amp by target bonus HP
- Implementation: Giant Slayer: damage amp = 0.15*clamp(target_bonusHP/1500,0,1).

### Luden's Echo (6655)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2750, combine 450, sell 1925 (70%) | recipe: Lost Chapter + Hextech Alternator + 450 g
- Item groups (client): {0eff64c7}(max 1)
- Flags: active spell BloodthirsterDummySpell
- Stats [CLIENT]: AP 100, Ability haste 10, Mana 600
- mDataValues [CLIENT]: Cooldown=12, BaseDamage=75, APRatio=0.05, MaxCharges=6, RepeatDamageReduction=0.2, MissileRange=650
- Calculations [CLIENT]: `Damage = BaseDamage(=75) + APRatio(=0.05) x AP`; `SingleTargetMax = calc[Damage] * 2`
- Tooltip (client en_US, values substituted): 100 Ability Power |  [FlatMPPoolMod] Mana |  10 Ability HasteEcho (cd: Cooldown) | Damaging Abilities fire 6 Echoes that deal [Damage] bonus magic damage to the target and nearby enemies. Remaining Echoes fire on the primary target, dealing 20% damage. (Maximum: [SingleTargetMax])
- Wiki pass "Echo": Gain 6 Echo stacks. Dealing ability damage to an enemy consumes all Echo stacks to deal 75 (+ 5% AP) bonus magic damage to them and, for each stack consumed beyond the first, an additional enemy within cr 600 units of them, firing an orb at each secondary target that impacts after to deal the damage. If the number of additional targets fired at is less than the number of stacks consumed, (cd 12)

### Malignance (3118)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2700, combine 650, sell 1890 (70%) | recipe: Lost Chapter + Blasting Wand + 650 g
- Item groups (client): {90aaac30}(max 1)
- Stats [CLIENT]: AP 90, Ability haste 15, Mana 600
- mDataValues [CLIENT]: AOESize=250, UltimateHaste=20, GroundDuration=3, BaseDamage=60, APRatio=0.05, MaxRadius=550
- Calculations [CLIENT]: `{8e8f7a34} = (BaseDamage(=60) + APRatio(=0.05) x AP) * 0.25`; `MagicResistanceShred = 10`; `GroundBurnDamagePerTickTooltipOnly = BaseDamage(=60) + APRatio(=0.05) x AP`
- Tooltip (client en_US, values substituted): 90 Ability Power |  [FlatMPPoolMod] Mana |  15 Ability HasteScorn | Gain 20 Ultimate Ability Haste. | Hatefog | Damaging a champion with your Ultimate burns the ground beneath them for 3s, dealing [GroundBurnDamagePerTickTooltipOnly] magic damage per second and reducing their Magic Resist by [MagicResistanceShred].  | Radius increases based on the damage done. | Basic Attacks cannot trigger Hatefog.
- Wiki pass "Scorn": Gain 20 ultimate haste.
- Wiki pass2 "Hatefog": Dealing non-proc damage or pet damage to enemy champions with your ultimate ability creates a [levels: type=ultimate's damage instance|251;251.8;253.1;255.5;259.8;267.3;280.7;304.2;345.9;419.7;550|0 to 823|formula=250 + 2^(damage dealt/100)] radius scorched zone beneath them for 3 seconds, applying a Curse to enemies within that deals and reduces their magic resistance by 10 (3 second cooldown per target, starts on zone creation).

### Manamune (3004)
- Tier (wiki): =>Muramana; client epicness: 5; in SR store: True
- Cost: total 2900, combine 1100, sell 2030 (70%) | recipe: Tear of the Goddess + Caulfield's Warhammer + Long Sword + 1100 g
- Item groups (client): TearItems(max 1), {a4ceabbc}(max 1), {7be37a10}, {1fd09102} | wiki limit: Manaflow
- Flags: active spell ManamuneDummySpell
- Stats [CLIENT]: AD 35, Ability haste 15, Mana 500
- mDataValues [CLIENT]: ManaPerCharge=3, ManaChargeAmmoCD=8, ManaChargeMaxAmmo=4, MaxMana=360, TakedownMana=0, InternalCDPerCastID=6.5
- Calculations [CLIENT]: `BonusADFromMana = 0.02 x Mana`
- Tooltip (client en_US, values substituted): 35 Attack Damage |  [FlatMPPoolMod] Mana |  15 Ability HasteAwe | Gain [BonusADFromMana] bonus Attack Damage. | Manaflow  (8s, max 4 charges) | Landing Attacks and Abilities grants 3 max Mana (doubled vs. champions). | Transforms into Muramana at 360 max Mana.
- Wiki pass "Awe": Grants bonus attack damage equal to 2% maximum mana.
- Wiki pass2 "Manaflow": Grants a charge every 8 seconds, up to 4 charges. Consumes a charge on-hit and whenever affecting an enemy or ally with an ability to grant 3 bonus mana, increased to 6 for champion targets, up to a maximum of 360 bonus mana. Can only be triggered once per cast instance.
- Wiki pass3: Transforms into Muramana at 360 bonus mana.

### Maw of Malmortius (3156)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3100, combine 750, sell 2170 (70%) | recipe: Hexdrinker + Caulfield's Warhammer + 750 g
- Item groups (client): LifelineItems(max 1) | wiki limit: Lifeline
- Stats [CLIENT]: AD 60, MR 40, Ability haste 15
- mDataValues [CLIENT]: ShieldSize=200, LowHealthThreshold=0.3, ShieldDuration=3, ShieldADScaling=1.5, BuffDuration=5, BuffExtension=3, BuffVamp=0.1, RangedShieldMod=0.75, Cooldown=90
- Calculations [CLIENT]: `MeleeItemCalcValue = ShieldSize(=200) + ShieldADScaling(=1.5) x bonus AD`; `RangedItemCalcValue = calc[MeleeItemCalcValue] * RangedShieldMod(=0.75)`
- Tooltip (client en_US, values substituted): 60 Attack Damage |  15 Ability Haste |  40 Magic ResistLifeline (cd: Cooldown) | Taking magic damage that would reduce your Health below 30% grants a [melee/ranged split: calc MeleeItemCalcValue / RangedItemCalcValue] magic damage Shield for 3 seconds and 10% Omnivamp until end of combat.
- Wiki pass "Lifeline": If you would take magic damage that would reduce you below 30% of your maximum health, you first gain a shield that absorbs 200 (melee) / 150 (ranged) (+ 150% (melee) / 112.5% (ranged) bonus AD) magic damage for 3 seconds. Additionally, triggering this effect grants you 10% omnivamp until the end of combat. (cd 90)
- Hooks: ON_HP_THRESHOLD(30%, magic damage) Lifeline [TOP-LANE PRIORITY]
- Implementation: Shield 200 + 150% bonus AD (ranged x0.75) magic shield 3 s; +10% omnivamp until end of combat.

### Mejai's Soulstealer (3041)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 1500, combine 1150, sell 1050 (70%) | recipe: Dark Seal + 1150 g
- Item groups (client): Glory(max 1) | wiki limit: Glory
- Stats [CLIENT]: Health 100, AP 20
- mDataValues [CLIENT]: MaxGloryStacks=25, APPerGlory=5, GloryOnKill=4, GloryOnAssist=2, GloryLossOnDeath=10, GloryThreshold=10, MoveSpeedMod=0.1
- Calculations [CLIENT]: `CurrentGloryAP = APPerGlory x stacks`
- Tooltip (client en_US, values substituted): 20 Ability Power |  100 HealthGlory | Takedowns grant Glory, up to 25. 10 Glory is lost on death. | Gain 5 Ability Power per Glory and 10% Move Speed at 10 or higher Glory. | Kills grant 4 Glory and Assists grant 2.
- Wiki pass "Glory": Gain 4 stacks for each champion kill and 2 stacks for each assist, up to a maximum of 25 stacks. For every stack, gain 5 ability power, up to 125 at maximum stacks. If you have at least 10 stacks, also gain 10% bonus movement speed. Lose 10 stacks on death. Stacks are preserved from Dark Seal.

### Mercurial Scimitar (3139)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3200, combine 125, sell 2240 (70%) | recipe: Quicksilver Sash + Pickaxe + Vampiric Scepter + 125 g
- Item groups (client): Quicksilver(max 1) | wiki limit: Quicksilver
- Flags: active spell ItemMercurial
- Stats [CLIENT]: AD 50, MR 35, Life steal 10%
- mDataValues [CLIENT]: Cooldown=90, MSDuration=2, MoveSpeed=0.5
- Tooltip (client en_US, values substituted): 50 Attack Damage |  35 Magic Resist |  10% Life Steal Quicksilver (cd: Cooldown) | Activate to remove all crowd control debuffs (excluding Airborne) and gain 50% Move Speed for 2 seconds.
- Wiki act "Quicksilver": Removes all crowd control debuffs (except Airborne) from your champion and grants 50% bonus total movement speed and ghosted for 2 seconds. (cd 90)
- Hooks: ACTIVE cleanse (deferred) [DEFERRED]

### Mikael's Blessing (3222)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2300, combine 900, sell 1610 (70%) | recipe: Kindlegem + Forbidden Idol + 900 g
- Item groups (client): 3222(max 1)
- Flags: active spell 3222Active
- Stats [CLIENT]: Health 250, Ability haste 15, Base mana regen 100% of base, Heal & shield power 12%
- mDataValues [CLIENT]: CleanseCooldown=120, HealAmountMin=100, HealAmountMax=250, Cooldown=120
- Calculations [CLIENT]: `AmountToHeal = (HealAmountMin(=100) + (lerp_level(0 -> 1))*((((HealAmountMin(=100))*(-1) + HealAmountMax(=250)))))`
- Tooltip (client en_US, values substituted): 250 Health |  [PercentBaseMPRegenMod*100]% Base Mana Regen |  12% Heal and Shield Power |  15 Ability Haste Purify (cd: Cooldown) | Remove all crowd control debuffs (excluding Airborne and Suppression) from an ally champion and restore 100 - 250 (ally ) Health.
- Wiki act "Purify": Remove all crowd control debuffs (except Airborne, Blind, Disarm, Nearsight, and Suppression) from yourself or the target allied champion and heal the target for [levels: 100 to 250|type=target's level|color=heal]. (cd 120)

### Moonstone Renewer (6617)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2200, combine 500, sell 1540 (70%) | recipe: Kindlegem + Bandleglass Mirror + 500 g
- Item groups (client): {81091299}(max 1)
- Stats [CLIENT]: Health 200, AP 25, Ability haste 20, Base mana regen 125% of base
- mDataValues [CLIENT]: EffectRange=800, ChainHeal=0.3, ChainShield=0.35, SingleHeal=0.3, SingleShield=0.35
- Tooltip (client en_US, values substituted): 25 Ability Power |  200 Health |  20 Ability Haste |  [PercentBaseMPRegenMod*100]% Base Mana RegenStarlit Grace | Healing or shielding an ally chains the effect to another ally (excluding yourself), healing 30% or shielding 35% of the original amount. | If there are no other allies nearby, heal 30% or shield 35% of the original amount to the same target.
- Wiki pass "Starlit Grace": Heal or shield an allied champion chains the effect to the other nearest and most wounded [Lowest health percent] allied champion within cr 800 units of them (excluding yourself), granting them 30% of the heal or 35% of the shield's initial strength. If no other allied champions are in the radius, grant the same target an additional 30% of the heal or 35% of the shield.

### Morellonomicon (3165)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2850, combine 400, sell 1995 (70%) | recipe: Oblivion Orb + Blasting Wand + Kindlegem + 400 g
- Item groups (client): 3165(max 1), {c8a69ca7}
- Stats [CLIENT]: Health 350, AP 75, Ability haste 15
- mDataValues [CLIENT]: GrievousAmount=0.4, GrievousDuration=3
- Tooltip (client en_US, values substituted): 75 Ability Power |  350 Health |  15 Ability HasteGrievous Wounds | Dealing magic damage to champions applies 40% Wounds for 3 seconds.
- Wiki pass "Grievous Wounds": Dealing magic damage to enemy champions inflicts them with Grievous Wounds for 3 seconds.

### Mortal Reminder (3033)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 150, sell 2100 (70%) | recipe: Executioner's Calling + Last Whisper + Cloak of Agility + 150 g
- Item groups (client): 3033(max 1), LastWhisper(max 1), {c8a69ca7} | wiki limit: Fatality
- Stats [CLIENT]: AD 35, Crit chance 25%, Armor pen 30%
- mDataValues [CLIENT]: GrievousDuration=3, GrievousAmount=0.4
- Tooltip (client en_US, values substituted): 35 Attack Damage |  30% Armor Penetration |  25% Critical Strike ChanceGrievous Wounds | Dealing physical damage applies 40% Wounds to enemy champions for 3 seconds.
- Wiki pass "Grievous Wounds": Dealing physical damage to enemy champions inflicts them with Grievous Wounds for 3 seconds.
- Hooks: ON_DAMAGE_DEALT(physical, champion) apply Grievous Wounds 40% 3 s [TOP-LANE PRIORITY]

### Nashor's Tooth (3115)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2900, combine 500, sell 2030 (70%) | recipe: Recurve Bow + Blasting Wand + Fiendish Codex + 500 g
- Item groups (client): 3115(max 1)
- Flags: active spell Malady
- Stats [CLIENT]: AP 80, Attack speed 50% (bonus AS ratio), Ability haste 15
- mDataValues [CLIENT]: NashorsBaseValue=15, NashorsAPValue=0.15
- Calculations [CLIENT]: `TotalOnHitDamage = NashorsBaseValue(=15) + NashorsAPValue(=0.15) x AP`
- Tooltip (client en_US, values substituted): 80 Ability Power |  50% Attack Speed |  15 Ability HasteIcathian Bite | Attacks deal [TotalOnHitDamage] bonus magic damage .
- Wiki pass "Icathian Bite": Basic attacks deal 15 (+ 15% AP) bonus magic damage on-hit.

### Navori Flickerblade (6675)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2650, combine 950, sell 1855 (70%) | recipe: Dagger + Zeal + Dagger + 950 g
- Item groups (client): 6675(max 1)
- Stats [CLIENT]: Attack speed 40% (bonus AS ratio), Crit chance 25%, MS 4% (additive % MS)
- mDataValues [CLIENT]: CDRAmount=0.15
- Tooltip (client en_US, values substituted): 40% Attack Speed |  25% Critical Strike Chance |  4% Move SpeedTranscendence | Attacks reduce Basic Ability cooldowns by 15% of their remaining cooldown.
- Wiki pass "Transcendence": Basic attacks on-attack reduce the remaining cooldowns of your basic abilities by 15%.

### Overlord's Bloodmail (2501)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3300, combine 1000, sell 2310 (70%) | recipe: Tunneler + Tunneler + 1000 g
- Item groups (client): {18fc0a13}(max 1)
- Stats [CLIENT]: Health 550, AD 30
- mDataValues [CLIENT]: HPToADPercentage=0.025, MissingHealthAD=0.12, MissingHealthThreshold=0.7
- Calculations [CLIENT]: `RemainingHealthThreshold = (-1 + MissingHealthThreshold(=0.7)) * -1 (shown as %)`
- Tooltip (client en_US, values substituted): 30 Attack Damage |  550 HealthTyranny | Gain 2.5% of your bonus Health as Attack Damage. | Retribution | Gain up to 12% Attack Damage based on your missing Health.Maximum Retribution bonus while below [RemainingHealthThreshold] Health.
- Wiki pass "Tyranny": Gain bonus attack damage equal to 2.5% bonus health.
- Wiki pass2 "Retribution": Gain bonus attack damage equal to [levels: 0 to 12 by 1|0 to 70|key=%|key1=%|type=missing health|color=health] of your total attack damage from other sources.
- Hooks: STAT_DYN: AD += 2.5% bonus HP; AD *= 1+up to 12% by missing HP [TOP-LANE PRIORITY]
- Implementation: Tyranny: bonus AD += 0.025 * bonusHP. Retribution: increased AD = 0.12 * clamp(missing_frac / 0.7, 0, 1) [tooltip: max while below 30% HP]; whether the % applies to total AD or bonus AD is not stated by data ('increased Attack Damage') - wiki: see catalog; default total AD (INFERRED).

### Phantom Dancer (3046)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2650, combine 950, sell 1855 (70%) | recipe: Dagger + Zeal + Dagger + 950 g
- Item groups (client): 3046(max 1)
- Stats [CLIENT]: Attack speed 65% (bonus AS ratio), Crit chance 25%, MS 10% (additive % MS)
- Tooltip (client en_US, values substituted): 65% Attack Speed |  25% Critical Strike Chance |  10% Move SpeedSpectral Waltz | Become Ghosted.
- Wiki pass "Spectral Waltz": Become permanently ghosted.
- Hooks: STAT; ghosted [TOP-LANE PRIORITY]

### Profane Hydra (6698)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2850, combine 313, sell 1995 (70%) | recipe: Tiamat + The Brutalizer + 313 g
- Item groups (client): {c6428663}(max 1), {8c259571}(max 3), {09f4cf8c}(max 1) | wiki limit: Hydra
- Flags: active spell 6698Active
- Stats [CLIENT]: AD 55, Ability haste 10, Lethality 18
- mDataValues [CLIENT]: LethalityAmount=18, Cooldown=10, CleaveRadius=350, ActiveRadius=450, HealthThreshold=0.5, MaxProcPerAuto=10
- Calculations [CLIENT]: `SlashDamageBase = 0.8 x AD`; `SlashDamageMax = 0.8 x AD`; `MeleeItemCalcValue = 0.4 x AD`; `RangedItemCalcValue = 0.2 x AD`; `CleaveDamage = 0.4 x AD   [ranged holder: x 0.5]`
- Tooltip (client en_US, values substituted): 55 Attack Damage |  18 Lethality |  10 Ability HasteCleave | Attacks deal [CleaveDamage] physical damage to nearby enemies. Heretical Cleave (cd: Cooldown) | Deal [SlashDamageBase] physical damage around you. | Cleave does not trigger on structures.
- Wiki pass "Cleave": Damaging basic attacks on-hit deal 40% AD (melee) / 20% AD (ranged) physical damage to other enemies in a cr 350 radius centered around the target. The same target may be damaged only once per frame.
- Wiki act "Heretical Cleave": Deal 80% AD physical damage to enemies in a cr 450 radius in front of you [100 unit offset in the caster's facing direction]. (cd 10)
- Hooks: ON_HIT cleave; ACTIVE Heretical Cleave [TOP-LANE PRIORITY]
- Implementation: see ITEMS.md §8 (full Cleave and active spec).

### Protoplasm Harness (2525)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2600, combine 900, sell 1820 (70%) | recipe: Kindlegem + Giant's Belt + 900 g
- Item groups (client): {a1015d59}(max 1), LifelineItems(max 1) | wiki limit: Lifeline
- Stats [CLIENT]: Health 600, Ability haste 20
- mDataValues [CLIENT]: LowHealthThreshold=0.3, Duration=5, MSAmount=0.1, TenacityAmount=0.25, Cooldown=90, SizeIncrease=0.15
- Calculations [CLIENT]: `TotalHealthRegen = lerp_level(100 -> 400) + 1.75 x bonus Armor + 1.75 x bonus MR`; `MaxHealthGain = lerp_level(100 -> 300)`
- Tooltip (client en_US, values substituted): 600 Health |  20 Ability HasteLifeline (cd: Cooldown) | Taking damage that would reduce your Health below 30% causes you to gain [MaxHealthGain] maximum Health for 5 seconds, then heal [TotalHealthRegen] Health over the duration. While regenerating Health, you gain 15% increased Size, 10% Move Speed, and 25% Tenacity.
- Wiki pass "Lifeline": If you would take damage that would reduce you below 30% of your maximum health, you first gain [levels: 100 to 300 for 5 seconds and heal yourself for [levels: 100 to 400|tooltipSize=20|color=heal] (+ 175% bonus armor) (+ 175% bonus magic resistance) over the same duration, during which you also gain 15% increased size, 10% bonus movement speed, and 25% tenacity. (cd 90)
- Hooks: ON_HP_THRESHOLD(30%) Lifeline [TOP-LANE PRIORITY]
- Implementation: Lifeline (shared Lifeline group cooldown 90 s): on damage that would reduce HP below 30%: +MaxHealthGain (100->300 by level) max HP for 5 s and heal (100->400 by level + 1.75 bonus armor + 1.75 bonus MR) over 5 s; +15% size, +10% MS, +25% tenacity while regenerating.

### Rabadon's Deathcap (3089)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3500, combine 1100, sell 2450 (70%) | recipe: Needlessly Large Rod + Needlessly Large Rod + 1100 g
- Item groups (client): 3089(max 1), APMultiplier(max 1)
- Stats [CLIENT]: AP 130
- mDataValues [CLIENT]: APAmp=0.3
- Tooltip (client en_US, values substituted): 130 Ability PowerMagical Opus | Increases your total Ability Power by 30%.
- Wiki pass "Magical Opus": Increase your ability power by 30%.

### Randuin's Omen (3143)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2700, combine 800, sell 1890 (70%) | recipe: Warden's Mail + Giant's Belt + 800 g
- Item groups (client): 3143(max 1)
- Flags: active spell RanduinsOmen
- Stats [CLIENT]: Health 350, Armor 75
- mDataValues [CLIENT]: PercentCritDamageReduction=0.3, SlowAmount=0.7, SlowDuration=2, Radius=500, Cooldown=90
- Tooltip (client en_US, values substituted): 350 Health |  75 ArmorResilience | Receive 30% less damage from Critical Strikes. |  Humility (cd: Cooldown) | Slow nearby enemies by 70% for 2 seconds.
- Wiki act "Humility": Unleash a shockwave around you that slow nearby enemies by 70% for 2 seconds. (cd 90)
- Wiki pass "Resilience": Reduces incoming damage from critical strike by 30%.
- Hooks: ON_PRE_DAMAGE_TAKEN(crit) -30% of crit damage; ACTIVE slow (deferred) [TOP-LANE PRIORITY]
- Implementation: Resilience: critical strikes deal 30% less damage to holder.

### Rapid Firecannon (3094)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2650, combine 850, sell 1855 (70%) | recipe: Zeal + Scout's Slingshot + 850 g
- Item groups (client): 3094(max 1), {040e02e8}
- Stats [CLIENT]: Attack speed 35% (bonus AS ratio), Crit chance 25%, MS 4% (additive % MS)
- mDataValues [CLIENT]: RangePercentIncrease=0.35, MaxRangeIncrease=150, BonusDamage=40
- Tooltip (client en_US, values substituted): 35% Attack Speed |  25% Critical Strike Chance |  4% Move SpeedSharpshooter | Your Energized Attack deals 40 bonus magic damage and gains 35% bonus Attack Range. | Attack Range cannot increase more than 150 units.
- Wiki pass "Energized": Moving and basic attacking generates Energize stacks, up to 100.
- Wiki pass2 "Sharpshooter": When fully Energized, your next basic attack deals 40 bonus magic damage on-hit. Energized attacks gain 35% bonus range, capped at 150.

### Ravenous Hydra (3074)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3300, combine 150, sell 2310 (70%) | recipe: Tiamat + Vampiric Scepter + Caulfield's Warhammer + 150 g
- Item groups (client): {c6428663}(max 1), {8c259571}(max 3), {a4cc6d25}(max 1) | wiki limit: Hydra
- Flags: active spell 3074Active
- Stats [CLIENT]: AD 65, Life steal 12%, Ability haste 15
- mDataValues [CLIENT]: CleaveRadius=350, Cooldown=10, Radius=450, ActiveADRatio=0.8, VampAmp=1, MaxProcPerAuto=10
- Calculations [CLIENT]: `MeleeItemCalcValue = 0.4 x AD`; `RangedItemCalcValue = 0.2 x AD`; `PrimaryDamage = ActiveADRatio(=0.8) x AD`
- Tooltip (client en_US, values substituted): 65 Attack Damage |  15 Ability Haste |  12% Life StealCleave | Attacks deal [melee/ranged split: calc MeleeItemCalcValue / RangedItemCalcValue] physical damage to nearby enemies. Ravenous Crescent (cd: Cooldown) | Deal [PrimaryDamage] physical damage to enemies around you.  | Your Life Steal applies to this damage. | Cleave will benefit from Life Steal. | Cleave does not trigger on structures.
- Wiki act "Ravenous Crescent": Deal 80% AD physical damage to enemies within a cr 450 radius in front of you [100 unit offset in the caster's facing direction]. This damage benefits from life steal at 100% effectiveness. (cd 10)
- Wiki pass "Cleave": Basic attacks on-hit deal 40% AD (melee) / 20% AD (ranged) physical damage to other enemies in a cr 350 radius centered around the target. This damage benefits from life steal at 100% effectiveness.
- Hooks: ON_HIT cleave AoE; ACTIVE Ravenous Crescent [TOP-LANE PRIORITY]
- Implementation: see ITEMS.md §8 (full Cleave and active spec).

### Redemption (3107)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2300, combine 850, sell 1610 (70%) | recipe: Fiendish Codex + Forbidden Idol + 850 g
- Item groups (client): 3107(max 1)
- Flags: active spell ItemRedemption
- Stats [CLIENT]: AP 30, Ability haste 15, Base mana regen 100% of base, Heal & shield power 10%
- mDataValues [CLIENT]: HealMin=150, HealMax=350, DamageToChampions=0.1, Cooldown=90, AOESize=550, CastRange=5500, DiminishedEffect=0.5, DiminishedTimer=8, HPRegenIncrease=0.25, BaseManaRegen=0.25
- Calculations [CLIENT]: `HealAmount = (HealMin(=150) + (lerp_level(0 -> 1))*((((HealMin(=150))*(-1) + HealMax(=350)))))`
- Tooltip (client en_US, values substituted): 30 Ability Power |  15 Ability Haste |  [PercentBaseMPRegenMod*100]% Base Mana Regen |  10% Heal and Shield Power Intervention (cd: Cooldown) | Restore 150 - 350 (ally ) Health to allied units and deal 10% max Health true damage to enemy champions after 2.5 seconds.Can be activated while dead.  | Subsequent Intervention effects on targets are reduced by 50%.
- Wiki act "Intervention": Call upon a 550-radius beam of light to strike upon the target location after 2.5 seconds, granting sight of the area for the duration. Allies within the area are heal for [levels: 150 to 350|type=target's level|color=heal], while enemy champions within take 10% of target's maximum health as true damage. Can be used while dead. (cd 90)

### Riftmaker (4633)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3100, combine 950, sell 2170 (70%) | recipe: Haunting Guise + Fiendish Codex + 950 g
- Item groups (client): {24023b85}(max 1)
- Stats [CLIENT]: Health 350, AP 70, Ability haste 15
- mDataValues [CLIENT]: SecondsInCombat=4, EternityDamageIncreasePerSecond=0.02, EternityDamageIncreaseMax=0.08, BuffCounterDuration=4, HealthToAPConversionPercent=0.02, MaxAPMultiplier=0.05, VampAmountMelee=0.1, VampAmountRanged=0.06
- Calculations [CLIENT]: `{1247259a} = HealthToAPConversionPercent(=0.02) x bonus MaxHP`; `MeleeItemCalcValue = VampAmountMelee(=0.1) (shown as %)`; `RangedItemCalcValue = VampAmountRanged(=0.06) (shown as %)`
- Tooltip (client en_US, values substituted): 70 Ability Power |  350 Health |  15 Ability HasteVoid Corruption | For each second in combat with enemy champions, deal 2% bonus damage, up to 8%. At maximum strength, gain [melee/ranged split: calc MeleeItemCalcValue / RangedItemCalcValue] Omnivamp. | Void Infusion | Gain 2% of your bonus Health as Ability Power.
- Wiki pass "Void Corruption": For each second in combat with champions, deal 2% increased damage, stacking up to 4 times for a total of 8% increased damage. At maximum stacks, gain 10% (melee) / 6% (ranged) omnivamp.
- Wiki pass2 "Void Infusion": Gain ability power equal to 2% bonus health.

### Rod of Ages (6657)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2600, combine 450, sell 1820 (70%) | recipe: Blasting Wand + Catalyst of Aeons + 450 g
- Item groups (client): EternityItems(max 1), {10ff67ed}(max 1) | wiki limit: Eternity
- Flags: active spell BloodthirsterDummySpell
- Stats [CLIENT]: Health 350, AP 45, Mana 500
- mDataValues [CLIENT]: HealthPerStack=10, ManaPerStack=30, APPerStack=3, MaxStacks=10, EternityManaRestore=0.1, EternityHealthRestore=0.25, EternityMaxHealPerCast=20, EternityCDPerCast=1, SecondsPerStack=60
- Tooltip (client en_US, values substituted): 45 Ability Power |  350 Health |  [FlatMPPoolMod] ManaTimeless | This item gains 10 Health, 30 Mana and 3 Ability Power every 60 seconds up to 10 times. Upon reaching max stacks, gain a level. | Eternity | Taking damage from champions restores 10% of the damage as Mana.  | Casting an ability heals for 25% of Mana spent. | Mana from Eternity calculates from premitigation damage. | Heal from Eternity is capped at 20 Health per cast, or per second for toggle spells.
- Wiki pass "Timeless": This item gains 10 bonus health, 30 bonus mana, and 3 ability power every minute, up to 10 times, for a maximum of 100 bonus health, 300 bonus mana, and 30 ability power. Upon reaching maximum stacks, gain a level that preserves your current experience (level cap remains the same).
- Wiki pass2 "Eternity": Restore mana equal to 10% of pre-mitigation damage [Damage calculated before modifiers] taken from champions, and heal for [levels: 0 to 20|0 to 80 by 5|type=mana spent|label=healing|color=heal|formula=25% of mana spent, up to 20 healing] per cast. Toggled abilities can only heal for up to 20 per second.

### Runaan's Hurricane (3085)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2650, combine 850, sell 1855 (70%) | recipe: Zeal + Scout's Slingshot + 850 g
- Item groups (client): 3085(max 1), {8d0a6ae2}(max 1)
- Flags: purchase identities ['Ranged']; active spell AtmasImpalerDummySpell
- Stats [CLIENT]: Attack speed 40% (bonus AS ratio), Crit chance 25%, MS 5% (additive % MS)
- mDataValues [CLIENT]: BoltMinPercent=40, BoltMaxPercent=40, ExtraRangeOnBoltCheck=65, OnHitDamage=0
- Calculations [CLIENT]: `BoltDamage = (0.65) x AD`; `{232dac8a} = 0.65`
- Tooltip (client en_US, values substituted): Must be Ranged 40% Attack Speed |  25% Critical Strike Chance |  5% Move SpeedWind's Fury | Attacks fire bolts at [Effect3Amount] additional enemies near the target. | Each bolt deals [BoltDamage] physical damage and applies  effects.Wind's Fury can Critically Strike.
- Wiki pass "Wind's Fury": Basic attacks on-attack fire additional bolts at up to 2 enemies {{tt|in front of you|180}}, each dealing 65% AD physical damage. Bolts apply on-hit effects and are affected by critical strike modifiers. The bolts will target the closest enemies to you that are not the main target.

### Rylai's Crystal Scepter (3116)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2600, combine 450, sell 1820 (70%) | recipe: Blasting Wand + Giant's Belt + Amplifying Tome + 450 g
- Item groups (client): 3116(max 1)
- Stats [CLIENT]: Health 400, AP 65
- mDataValues [CLIENT]: SlowAmount=0.3, SlowDuration=1
- Tooltip (client en_US, values substituted): 65 Ability Power |  400 HealthRimefrost | Damaging Abilities Slow enemies by 30% for 1 second.
- Wiki pass "Rimefrost": Dealing ability damage slow enemies hit by 30% for 1 second.

### Serpent's Fang (6695)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2500, combine 625, sell 1750 (70%) | recipe: Serrated Dirk + Pickaxe + 625 g
- Item groups (client): 6695(max 1)
- Stats [CLIENT]: AD 55, Lethality 15
- mDataValues [CLIENT]: LethalityAmount=15, ShieldShred=50, ShieldWounds=50, DebuffDuration=3, ShieldWoundsRange=35, ShieldShredRange=35
- Tooltip (client en_US, values substituted): 55 Attack Damage |  15 LethalityShield Reaver | Damaging an enemy champion reduces Shields they gain by [ShieldShredMeleeRangedSplit]% for 3 seconds.  | If they were not already affected by Shield Reaver, reduce Shields on them by [ShieldWoundMeleeRangedSplit]%.Magic shields are not reduced.
- Wiki pass "Shield Reaver": Dealing damage to an enemy champion inflicts them with venom for 3 seconds, reducing any shield they gain within the duration by 50% (melee) / 35% (ranged), and if the target was not already afflicted by the venom, reducing all of their active shields by the same amount.

### Serylda's Grudge (6694)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 500, sell 2100 (70%) | recipe: Caulfield's Warhammer + Last Whisper + 500 g
- Item groups (client): LastWhisper(max 1) | wiki limit: Fatality
- Stats [CLIENT]: AD 45, Ability haste 15, Armor pen 35%
- mDataValues [CLIENT]: SlowAmount=0.3, SlowDuration=1, LethalityAmount=15, SlowThreshold=0.5
- Calculations [CLIENT]: `PenCalc = 0 + 0 x Lethality (shown as %)`
- Tooltip (client en_US, values substituted): 45 Attack Damage |  35% Armor Penetration |  15 Ability HasteBitter Cold | Damaging Abilities Slow enemies below 50% Health by 30% for 1 second.
- Wiki pass "Bitter Cold": Dealing ability damage to an enemy that is at or below 50% of their maximum health slow them by 30% for 1 second.
- Hooks: ON_ABILITY_DAMAGE slow below 50%

### Shadowflame (4645)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3200, combine 900, sell 2240 (70%) | recipe: Hextech Alternator + Needlessly Large Rod + 900 g
- Item groups (client): 4645(max 1)
- Stats [CLIENT]: AP 110, Flat magic pen 15
- mDataValues [CLIENT]: HealthThreshold=0.4, SpellItemDamageAmp=0.2, DamageOverTimeAmp=0.2
- Tooltip (client en_US, values substituted): 110 Ability Power |  15 Magic PenetrationCinderbloom | Magic damage and true damage critically strike enemies below 40% Health, dealing 20% increased damage. | Critical damage modifiers only affect Cinderbloom's bonus damage.
- Wiki pass "Cinderbloom": Your magic and true damage will critical strike for 120% damage against enemies below 40% maximum health.

### Shattered Armguard (2421)
- Tier (wiki): =>Seeker's Armguard; client epicness: None; in SR store: True
- Cost: total 1600, combine 500, sell 640 (40%) | recipe: Amplifying Tome + Cloth Armor + Amplifying Tome + 500 g
- Builds into (client build hint): Zhonya's Hourglass
- Item groups (client): StopwatchGroup(max 1) | wiki limit: Stasis
- Flags: requires buff/currency Item2420
- Stats [CLIENT]: AP 40, Armor 25
- Tooltip (client en_US, values substituted): 40 Ability Power |  25 ArmorShattered Time | Armguard is broken, but can still be upgraded. | After breaking one Armguard, the shopkeeper will only sell you Shattered Armguard.

### Shurelya's Battlesong (2065)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2200, combine 400, sell 1540 (70%) | recipe: Aether Wisp + Bandleglass Mirror + 400 g
- Item groups (client): {d11f947a}(max 1)
- Flags: active spell 2065Active
- Stats [CLIENT]: AP 50, MS 4% (additive % MS), Ability haste 15, Base mana regen 125% of base
- mDataValues [CLIENT]: ActiveMoveSpeed=0.3, BuffDuration=4, ActiveCooldown=75, ItemRange=1000, PerChampCD=4, Cooldown=75
- Tooltip (client en_US, values substituted): 50 Ability Power |  15 Ability Haste |  4% Move Speed |  [PercentBaseMPRegenMod*100]% Base Mana Regen Inspiring Speech (cd: Cooldown) | Grant nearby allies 30% Move Speed for 4 seconds.
- Wiki act "Inspiring Speech": Grants you and all allies within 1000 units 30% bonus movement speed for 4 seconds. (cd 75)

### Spear of Shojin (3161)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3100, combine 675, sell 2170 (70%) | recipe: Pickaxe + Tunneler + Ruby Crystal + 675 g
- Item groups (client): 3161(max 1)
- Stats [CLIENT]: Health 450, AD 45
- mDataValues [CLIENT]: SpellDamageIncrease=0.03, AHBase=25, StackDuration=6, StackCount=4, RangedMod=0.5, TooltipValue=3, CastIDLockout=1
- Calculations [CLIENT]: `MeleeItemCalcValue = TooltipValue(=3)`; `RangedItemCalcValue = calc[MeleeItemCalcValue] * RangedMod(=0.5)`
- Tooltip (client en_US, values substituted): 45 Attack Damage |  450 HealthDragonforce  | Gain 25 Basic Ability Haste. | Focused Will  | Dealing damage with Abilities increases your Champion's Ability and Passive damage by 3% for 6 seconds. (stacks 4 times).Focused Will bonus damage dealt to champions: [f1] | Focused Will stacking has a 1 second lock out per spell cast instance.
- Wiki pass "Dragonforce": Gain 25 basic ability haste.
- Wiki pass2 "Focused Will": Dealing ability damage or pet damage from a non-innate ability cast instance, generates a stack of Focused Will for 6 seconds, stacking up to 4 times. For each stack, your ability damage, proc damage and pet damage originating from your ability cast instance deals 3% increased damage, for a total increase of 12% at maximum stacks. Each cast instance can only grant one stack per second.
- Hooks: STAT; ON_ABILITY_DAMAGE stacks

### Spirit Visage (3065)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2700, combine 650, sell 1890 (70%) | recipe: Spectre's Cowl + Kindlegem + 650 g
- Item groups (client): 3065(max 1)
- Stats [CLIENT]: Health 400, MR 50, Ability haste 10, Base HP regen 100% of base
- mDataValues [CLIENT]: HealingIncrease=0.25, ShieldIncrease=0.25
- Tooltip (client en_US, values substituted): 400 Health |  50 Magic Resist |  10 Ability Haste |  100% Base Health RegenBoundless Vitality | Heals and Shields on you are increased by 25%.
- Wiki pass "Boundless Vitality": Increases all heal and shield received as well as health regeneration by 25%.
- Hooks: STAT; heal/shield received x1.25 [TOP-LANE PRIORITY]
- Implementation: Boundless Vitality multiplies healing, shields, regen? (data: HealingIncrease 0.25, ShieldIncrease 0.25).

### Staff of Flowing Water (6616)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2250, combine 800, sell 1575 (70%) | recipe: Fiendish Codex + Forbidden Idol + 800 g
- Item groups (client): 6616(max 1)
- Stats [CLIENT]: AP 35, Ability haste 10, Base mana regen 125% of base, Heal & shield power 10%
- mDataValues [CLIENT]: BuffDuration=6, APMod=40, AHMod=15
- Tooltip (client en_US, values substituted): 35 Ability Power |  10% Heal and Shield Power |  [PercentBaseMPRegenMod*100]% Base Mana Regen |  10 Ability HasteRapids | Healing or Shielding an ally grants you both  40 Ability Power and  15 Ability Haste for 6 seconds.
- Wiki pass "Rapids": Heal or shield allied champions (excluding yourself) grants you and them 40 ability power and 15 ability haste for 6 seconds.

### Statikk Shiv (3087)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 625, sell 2100 (70%) | recipe: Scout's Slingshot + Aether Wisp + Pickaxe + 625 g
- Item groups (client): 3087(max 1), {040e02e8}
- Flags: RestrictedBuffName=HeroPassive
- Stats [CLIENT]: AD 45, AP 45, Attack speed 30% (bonus AS ratio), MS 4% (additive % MS)
- mDataValues [CLIENT]: BounceDelay=0, BounceRange=500, BonusEnergizedStacks=9, ChainDamage=60, NonChampChainDamage=90
- Calculations [CLIENT]: `BounceCount = level_bp(L1=4, +1 once at L>=6, +1 once at L>=10, +1 once at L>=14, +1 once at L>=20)`
- Tooltip (client en_US, values substituted): 45 Attack Damage |  45 Ability Power |  30% Attack Speed |  4% Move SpeedElectrospark | Your Energized Attacks fire chain lightning, dealing 60 magic damage to up to [BounceCount] targets, increased to 90 magic damage against Minions and Monsters. Applies  On-Hit effects to secondary bounce targets.  | Electroshock  | Basic Attacks grant 9 extra Energized stacks.
- Wiki pass "Energized": Moving and basic attacking generates Energize stacks, up to 100.
- Wiki pass2 "Electroshock": Basic attacks generate 9 bonus Energize stacks, for a total of 15 stacks per attack.
- Wiki pass3 "Electrospark": When fully Energized, your next basic attack on-hit is empowered to form chain lightning, dealing 60 bonus magic damage, increased to 90 against non-champions. This bounces to the closest target within 500 units, repeating from the new target to strike up to [levels: 4 to 8 by 1|1;6;10;14;20] targets, dealing the same damage and applying on-hit effects to secondary targets hit.

### Sterak's Gage (3053)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3200, combine 775, sell 2240 (70%) | recipe: Pickaxe + Tunneler + Ruby Crystal + 775 g
- Item groups (client): LifelineItems(max 1) | wiki limit: Lifeline
- Stats [CLIENT]: Health 400, Tenacity 20%
- mDataValues [CLIENT]: LowHealthThreshold=0.3, ShieldDuration=4.5, HealthPercent=0.1, HealPercent=0.02, RangedEffectiveness=0.6, ReductionAmount=0.5, BaseShieldRatio=0.6, HealDuration=5, ADtoAD=0.5, TimeBeforeDecay=0.75, TenacityDuration=8, SizeIncrease=0.1, TenacityAmount=0, Cooldown=90
- Calculations [CLIENT]: `MeleeItemCalcValue = HealPercent(=0.02) x MaxHP`; `BonusShield = calc[MeleeItemCalcValueB] * 1 x stacks`; `MeleeItemCalcValueB = HealthPercent(=0.1) x bonus MaxHP`; `RangedItemCalcValueB = calc[MeleeItemCalcValueB] * RangedEffectiveness(=0.6)`; `RangedItemCalcValue = calc[MeleeItemCalcValue] * RangedEffectiveness(=0.6)`; `ShieldSize = BaseShieldRatio(=0.6) x bonus MaxHP`; `BonusAD = ADtoAD(=0.5) x base AD`
- Tooltip (client en_US, values substituted): 400 Health |  20% TenacityThe Claws that Catch | Gain [BonusAD] bonus Attack Damage. | Lifeline (cd: Cooldown) | Taking damage that would reduce your Health below 30% grants a [ShieldSize] decaying Shield for 4.5 seconds.
- Wiki pass "The Claws that Catch": Gain bonus attack damage equal to 50% base AD.
- Wiki pass2 "Lifeline": If you would take damage that would reduce you below 30% of your maximum health, you first gain a shield that absorbs damage equal to 60% of bonus health which decays over 4.5 seconds. (cd 90)
- Hooks: STAT_DYN bonus AD = 50% base AD; ON_HP_THRESHOLD(30%) Lifeline shield [TOP-LANE PRIORITY]
- Implementation: The Claws that Catch: bonus AD += 0.50 * base AD. Lifeline (group cd 90 s): on damage that would reduce HP below 30% of max, grant a shield = 60% bonus HP decaying over 4.5 s (linear decay to 0 - INFERRED decay shape; data has TimeBeforeDecay 0.75 s which suggests 0.75 s hold then decay).

### Stormrazor (3095)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3200, combine 700, sell 2240 (70%) | recipe: B. F. Sword + Cloak of Agility + Scout's Slingshot + 700 g
- Item groups (client): 3095(max 1), {040e02e8}
- Stats [CLIENT]: AD 50, Attack speed 25% (bonus AS ratio), Crit chance 25%
- mDataValues [CLIENT]: BuffStrength=0.45, BuffDuration=1.5
- Calculations [CLIENT]: `TotalProcDamage = 100`
- Tooltip (client en_US, values substituted): 50 Attack Damage |  25% Attack Speed |  25% Critical Strike Chance | Bolt | Your Energized Attack applies [TotalProcDamage] bonus magic damage and grants 45% Movement Speed for 1.5 seconds.
- Wiki pass "Energized": Moving and basic attacking generates Energize stacks, up to 100.
- Wiki pass2 "Bolt": When fully Energized, your next basic attack deals 100 bonus magic damage on-hit and grants you 45% bonus movement speed for 1.5 seconds.

### Stormsurge (4646)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2800, combine 800, sell 1960 (70%) | recipe: Hextech Alternator + Aether Wisp + 800 g
- Item groups (client): {20fff835}(max 1)
- Stats [CLIENT]: AP 90, MS 6% (additive % MS), Flat magic pen 15
- mDataValues [CLIENT]: WindowDuration=2.5, Cooldown=30, DelayDuration=2, GoldReward=0, APRatio=0.1, DamageThreshold=0.25, ProcMoveSpeedDuration=1.5, ProcMoveSpeedAmount=0, RangedProcDamageMod=1, BaseDamage=125
- Calculations [CLIENT]: `MeleeItemCalcValue = BaseDamage(=125) + APRatio(=0.1) x AP`; `RangedItemCalcValue = calc[MeleeItemCalcValue] * RangedProcDamageMod(=1)`; `SquallDamage = BaseDamage(=125) + APRatio(=0.1) x AP`
- Tooltip (client en_US, values substituted): 90 Ability Power |  15 Magic Penetration |  6% Move SpeedStormraider (cd: Cooldown) | Dealing 25% of a champion's maximum Health within 2.5s applies Squall to them. | Squall | After 2 seconds, deal [SquallDamage] magic damage. If the target dies before Squall triggers, it damages nearby enemies.
- Wiki pass "Stormraider": Dealing damage to an enemy champion equal to 25% of their maximum health within 2.5 seconds inflicts them with Squall (30 second cooldown, starts on Squall's application).
- Wiki pass2 "Squall": After 2 seconds of having applied Squall, strike the target with lightning, dealing 125 (+ 10% AP) magic damage to them. If the target dies before being struck, they emit an electric field instantly that shocks all enemy champions in a cr 600 radius, dealing them the same damage.

### Stridebreaker (6631)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3300, combine 750, sell 2310 (70%) | recipe: Tiamat + Phage + Dagger + 750 g
- Item groups (client): {8f0da5d1}(max 1), {c6428663}(max 1) | wiki limit: Hydra
- Flags: active spell 6631Active
- Stats [CLIENT]: Health 450, AD 40, Attack speed 25% (bonus AS ratio)
- mDataValues [CLIENT]: Cooldown=15, ADRatio=0.8, Haste=0, MSSlow=-0.35, Duration=3, FlatMS=0, CleaveRadius=450, ActiveMS=0.35, PassiveRadius=350, MoveSpeedMod=0, MoveSpeedDuration=2, DecayRate=0.8, MaxProcPerAuto=10
- Calculations [CLIENT]: `SlashDamage = ADRatio(=0.8) x AD`; `MeleeItemCalcValue = 0.4 x AD`; `RangedItemCalcValue = 0.2 x AD`
- Tooltip (client en_US, values substituted): 40 Attack Damage |  25% Attack Speed |  450 HealthCleave | Attacks deal [melee/ranged split: calc MeleeItemCalcValue / RangedItemCalcValue] physical damage to nearby enemies. Breaking Shockwave (cd: Cooldown) | Deal [SlashDamage] physical damage and Slow nearby enemies by 35%. | Gain 35% decaying Move Speed per champion hit for 3 seconds.Cleave does not trigger on structures. | You can move while casting Breaking Shockwave.
- Wiki act "Breaking Shockwave": Deal 80% AD physical damage to enemies within a cr 450 radius in front of you [100 unit offset in the caster's facing direction] and slow them by 35% for 3 seconds. For each champion hit, gain 35% bonus movement speed decaying over 3 seconds. Can move while casting. (cd 15)
- Wiki pass "Cleave": Basic attacks on-hit deal 40% AD (melee) / 20% AD (ranged) physical damage to other enemies in a cr 350 radius centered around the target.
- Hooks: ON_HIT cleave AoE; ACTIVE Breaking Shockwave [TOP-LANE PRIORITY]
- Implementation: see ITEMS.md §8 (full Cleave and active spec).

### Sundered Sky (6610)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3100, combine 900, sell 2170 (70%) | recipe: Tunneler + Caulfield's Warhammer + 900 g
- Item groups (client): {8209142c}(max 1)
- Stats [CLIENT]: Health 400, AD 40, Ability haste 10
- mDataValues [CLIENT]: CritModifier=0.8, Cooldown=10, HealBaseADRatio=0.9, MissingHealthHeal=0.04, RangedHealMod=0.5
- Calculations [CLIENT]: `{01099b16} = HealBaseADRatio(=0.9) x base AD   [ranged holder: x RangedHealMod(=0.5)]`
- Tooltip (client en_US, values substituted): 40 Attack Damage |  400 Health |  10 Ability HasteLightshield Strike (cd: Cooldown) per target | Your first Attack against a champion Critically Strikes for 80% of your normal Critical Strike damage and restores [HealADSplit] plus 4% of missing Health. | Excess healing is granted as temporary bonus Health.
- Wiki pass "Lightshield Strike": Your next basic attack against a champion is empowered to critical strike forand heal you for 90% (melee) / 45% (ranged) base AD (+ 4% of your missing health) (10 second cooldown per target). Excess healing beyond maximum health is converted to bonus health for 8 seconds.
- Hooks: ON_HIT(first vs champion, 10 s per-target cd) crit 80% + heal [TOP-LANE PRIORITY]

### Sunfire Aegis (3068)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2800, combine 700, sell 1960 (70%) | recipe: Bami's Cinder + Chain Vest + Ruby Crystal + 700 g
- Item groups (client): ImmolateItems(max 1), 3068(max 1) | wiki limit: Immolate
- Stats [CLIENT]: Health 350, Armor 50, Ability haste 10
- mDataValues [CLIENT]: Range=325, MinionMod=0.5, MaxStacks=1, DamageAmpPerStack=0, StackDuration=0, MonsterMod=0.8, AuraDuration=3, TicksPerSecond=1, BaseDamagePerTickTOOLTIPONLY=20, HPRatioPerTickTOOLTIPONLY=1.5
- Calculations [CLIENT]: `DamagePerTick = 20 + 0.015 x bonus MaxHP`; `DPS = calc[DamagePerTick] * TicksPerSecond(=1)`; `{049cea52} = calc[DPS] * 1`
- Tooltip (client en_US, values substituted): 350 Health |  50 Armor |  10 Ability HasteImmolate | After taking or dealing damage, deal [DPS] magic damage per second to nearby enemies for 3 seconds.Immolate deals 50% increased damage to minions and 80% increased damage to monsters.
- Wiki pass "Immolate": Taking or dealing damage activates this passive for 3 seconds. Deal 20 (+ 1.5% bonus health) magic damage every second to enemies within cr 325 (+ 100% bonus size) units, with the damage being increased to 150% against minions and to 180% against monsters. This executes minions that would be killed by one more tick of damage.
- Hooks: ON_DAMAGE_DEALT/TAKEN -> Immolate aura 3 s (1 tick/s) [TOP-LANE PRIORITY]
- Implementation: Immolate: after taking or dealing damage, for 3 s deal (20 + 1.5% bonus HP) magic per second to enemies within 325; x1.5 vs minions (MinionMod 0.5 => +50%), x1.8 vs monsters.

### Terminus (3302)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 1100, sell 2100 (70%) | recipe: Hearthbound Axe + Recurve Bow + 1100 g
- Item groups (client): {c2413581}(max 1), LastWhisper(max 1), VoidPen(max 1) | wiki limit: Fatality
- Stats [CLIENT]: AD 30, Attack speed 35% (bonus AS ratio)
- mDataValues [CLIENT]: BuffDuration=5, DebuffDuration=5, PenMax=0.3, PenPerHit=0.1
- Calculations [CLIENT]: `ARMRPerHitScaling = level_bp(L1=6, +1 once at L>=11, +1 once at L>=14)`; `ARMRMaxScaling = calc[ARMRPerHitScaling] * 3`; `{592c02e8} = calc[OnHitDamage]`; `OnHitDamage = 30 + 0.1 x bonus AD + 0.1 x AP`
- Tooltip (client en_US, values substituted): 30 Attack Damage |  35% Attack SpeedShadow | Attacks deal [OnHitDamage] bonus magic damage . | Juxtaposition | Alternate between Light and Dark Attacks against champions: Light Attacks grant [ARMRPerHitScaling] Armor and Magic Resist (up to [ARMRMaxScaling]) for 5s. Dark Attacks grant 10% Armor Penetration and Magic Penetration (up to 30%) for 5s.
- Wiki pass "Shadow": Basic attacks deal 30 (+10% bonus AD) (+ 10% AP) bonus magic damage on-hit.
- Wiki pass2 "Juxtaposition": Basic attacks on-hit against champions alternate between Light and Dark hits, each one granting a bonus for 5 seconds that stacks up to 3 times. Light hits grant [levels: 6 to 8 for 3|1;11;14|type=level] bonus armor and bonus magic resistance while Dark hits grant 10% armor penetration and magic penetration, for a total of [levels: 6*3 to 8*3 for 3|1;11;14] bonus resistances and 30% resistances penetration at maximum stacks of each.

### The Collector (6676)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 525, sell 2100 (70%) | recipe: Pickaxe + Serrated Dirk + Cloak of Agility + 525 g
- Item groups (client): 6676(max 1)
- Stats [CLIENT]: AD 50, Crit chance 25%, Lethality 10
- mDataValues [CLIENT]: LethalityAmount=10, ExecuteThreshold=0.05, GoldAmount=25
- Tooltip (client en_US, values substituted): 50 Attack Damage |  10 Lethality |  25% Critical Strike ChanceDeath | Your damage executes champions that are below 5% Health. | Taxes | Champion kills grant 25 bonus gold.
- Wiki pass "Death": If you deal post-mitigation [Damage calculated after modifiers] damage that would leave a champion below 5% of their maximum health, execute them.
- Wiki pass2 "Taxes": Killing a champion grants you an additional 25.

### Thornmail (3075)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2450, combine 450, sell 1715 (70%) | recipe: Bramble Vest + Chain Vest + Ruby Crystal + 450 g
- Item groups (client): 3075(max 1), {d52cd27b}(max 1), {c8a69ca7}
- Flags: active spell AtmasImpalerDummySpell
- Stats [CLIENT]: Health 150, Armor 75
- mDataValues [CLIENT]: BaseDamage=20, BonusArmorDamageRatio=0.1, GrievousAmount=0.4, GrievousDuration=3, EnhancedGrievousAmount=0.4
- Calculations [CLIENT]: `TotalDamage = BaseDamage(=20) + BonusArmorDamageRatio(=0.1) x bonus Armor`
- Tooltip (client en_US, values substituted): 150 Health |  75 ArmorThorns | When struck by an Attack, deal [TotalDamage] magic damage to the attacker and apply 40% Wounds for 3 seconds if they are a champion.
- Wiki pass "Thorns": When struck by a basic attack on-hit, deal 20 (+ 10% bonus armor) magic damage to the attacker and, if they are a champion, inflict them with Grievous Wounds for 3 seconds.
- Hooks: ON_BEING_HIT(basic attack) reactive magic + GW [TOP-LANE PRIORITY]
- Implementation: Thorns: when struck by a basic attack deal 20 + 10% bonus armor magic (reactive damage; no vamp) to attacker; if champion apply 40% GW 3 s.

### Titanic Hydra (3748)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3300, combine 50, sell 2310 (70%) | recipe: Tiamat + Tunneler + Giant's Belt + 50 g
- Item groups (client): {c6428663}(max 1), {8c259571}(max 3), {6bd2873b}(max 1) | wiki limit: Hydra
- Flags: active spell 3748Active
- Stats [CLIENT]: Health 600, AD 40
- mDataValues [CLIENT]: PrimaryTargetHPRatio=0.01, SplashHPRatio=0.03, RangedEffectiveness=0.5, ActivePrimaryTargetHPRatio=0.04, ActiveSplashHPRatio=0.09, Cooldown=10, MaxProcPerAuto=10
- Calculations [CLIENT]: `CalcValueB = SplashHPRatio(=0.03) x MaxHP   [ranged holder: x RangedEffectiveness(=0.5)]`; `CalcValue = PrimaryTargetHPRatio(=0.01) x MaxHP   [ranged holder: x RangedEffectiveness(=0.5)]`; `CalcValueC = ActivePrimaryTargetHPRatio(=0.04) x MaxHP   [ranged holder: x RangedEffectiveness(=0.5)]`; `CalcValueD = ActiveSplashHPRatio(=0.09) x MaxHP   [ranged holder: x RangedEffectiveness(=0.5)]`; `OnHitDamageCalc = 0.01 x MaxHP   [ranged holder: x 0.5]`; `ConeDamageCalc = 0.03 x MaxHP   [ranged holder: x 0.5]`
- Tooltip (client en_US, values substituted): 40 Attack Damage |  600 HealthCleave | Attacks deal [OnHitDamageCalc] physical damage on hit and [ConeDamageCalc] physical damage to enemies behind the target. Titanic Crescent (cd: Cooldown) | Empower your next Cleave to deal [CalcValueC] bonus physical damage  and deal [CalcValueD] bonus physical damage to enemies behind the target.Cleave's shockwave does not trigger on structures.
- Wiki pass "Cleave": Basic attacks on-hit deal 1% (melee) / 0.5% (ranged) maximum health bonus physical damage to the target and 3% (melee) / 1.5% (ranged) maximum health physical damage to other enemies in a cone in the direction of the primary target.
- Wiki act "Titanic Crescent": Your next basic attack on-hit within 10 seconds empowers Cleave to deal 4% (melee) / 2% (ranged) maximum health bonus physical damage to the primary target and 9% (melee) / 4.5% (ranged) maximum health physical damage to secondary targets (10 second cooldown, starts after using the empowered attack). Titanic Crescent resets the user's basic attack timer.
- Hooks: ON_HIT on-hit %HP + cone; ACTIVE Titanic Crescent (AA reset) [TOP-LANE PRIORITY]
- Implementation: see ITEMS.md §8 (full Cleave and active spec).

### Trinity Force (3078)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3333, combine 133, sell 2333 (70%) | recipe: Sheen + Phage + Hearthbound Axe + 133 g
- Item groups (client): {a8cc7371}(max 1), {57352a0f}(max 1) | wiki limit: Spellblade
- Stats [CLIENT]: Health 333, AD 36, Attack speed 30% (bonus AS ratio), Ability haste 15
- mDataValues [CLIENT]: SpellbladeMultiplier=2, MoveSpeedBonus=20, MSDuration=2, SpellbladeCooldown=1.5, Cooldown=1.5
- Calculations [CLIENT]: `SpellbladeDamage = SpellbladeMultiplier(=2) x base AD`; `{6e4cefdc} = 20`
- Tooltip (client en_US, values substituted): 36 Attack Damage |  30% Attack Speed |  333 Health |  15 Ability HasteSpellblade (cd: Cooldown) | After using an Ability, your next Attack deals [SpellbladeDamage] bonus physical damage . |   | Quicken | Attacking grants 20 Move Speed for 2 seconds.
- Wiki pass "Spellblade": After using an ability, your next basic attack within 10 seconds deals 200% base AD bonus physical damage on-hit (1.5 second cooldown, starts after using the empowered attack).
- Wiki pass2 "Quicken": Basic attacks on-hit grant 20 bonus movement speed for 2 seconds
- Hooks: ON_ABILITY_CAST arm; ON_HIT +200% base AD phys; ON_HIT +20 MS 2 s [TOP-LANE PRIORITY]
- Implementation: see ITEMS.md §6.1.

### Umbral Glaive (3179)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2800, combine 750, sell 1960 (70%) | recipe: Serrated Dirk + Caulfield's Warhammer + 750 g
- Item groups (client): 3179(max 1)
- Stats [CLIENT]: AD 60, Ability haste 15, Lethality 18
- mDataValues [CLIENT]: LethalityAmount=18, MeleeDamage=3, RangedDamage=2, Cooldown=90, OutOfVisionDuration=1, AttackReadyDuration=4
- Calculations [CLIENT]: `MeleeItemCalcValue = MeleeDamage(=3)`; `RangedItemCalcValue = RangedDamage(=2)`; `ProcDamage = 50 + 1.5 x Lethality`; `TotalWardDamage = 3   [ranged holder: x 0.667]`
- Tooltip (client en_US, values substituted): 60 Attack Damage |  18 Lethality |  15 Ability HasteNightstalker | After being unseen by enemies for 1 second(s), your next attack against a champion deals an additional [ProcDamage] true damage. | Blackout (cd: Cooldown) | When you are near enemy Stealth Wards and traps, reveal them for [Effect2Amount] seconds. While revealing wards, your attacks deal [TotalWardDamage] damage to them.Revealed Stealth Wards are disabled while revealed
- Wiki pass "Blackout": When spotted by enemy stealthed wards or stealthed trap, gain Blackout for 8 seconds. (cd 90)
- Wiki pass2 "Blackout": You disabled ward surrounding stealthed wards, as well as expose and true sight nearby stealthed wards and traps while Blackout is active. Your basic attacks deal 2 (melee) / 1 (ranged) bonus true damage to wards.
- Wiki pass3 "Nightstalker": After being not sight to enemies for at least 1 second, your next basic attack against a champion is empowered to deal 50 (+ 1.5 per 1 Lethality) bonus true damage on-hit. The empowered attack lasts for 4 seconds after being seen by an enemy.

### Unending Despair (2502)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2800, combine 800, sell 1960 (70%) | recipe: Chain Vest + Kindlegem + Ruby Crystal + 800 g
- Item groups (client): {19fc0ba6}(max 1)
- Stats [CLIENT]: Health 400, Armor 50, Ability haste 15
- mDataValues [CLIENT]: DrainRange=650, Cooldown=4, BonusHealthDrainPercentage=0.03, HealMultiplier=2.5
- Calculations [CLIENT]: `DrainCalc = BonusHealthDrainPercentage(=0.03) x bonus MaxHP`; `HealCalc = (calc[DrainCalc]) * HealMultiplier(=2.5)`
- Tooltip (client en_US, values substituted): 400 Health |  50 Armor |  15 Ability HasteAnguish | Every 4 seconds while in combat with champions, deal [DrainCalc] magic damage to nearby enemy champions and heal for 250% of the damage dealt.
- Wiki pass "Anguish": Every 4 seconds after entering combat with champions, sap all enemy champions around you within cr 650 units to deal magic damage equal to 3% of your bonus health to them and heal yourself equal to 250% of the post-mitigation [Damage calculated after modifiers] damage dealt.
- Hooks: PERIODIC(4 s while in champion combat) AoE magic drain [TOP-LANE PRIORITY]
- Implementation: Anguish: every 4.0 s while in combat with champions, deal 3% bonus HP magic damage to enemy champions within 650 and heal 250% of damage dealt.

### Void Staff (3135)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 1050, sell 2100 (70%) | recipe: Blighting Jewel + Blasting Wand + 1050 g
- Item groups (client): VoidPen(max 1) | wiki limit: Blight
- Stats [CLIENT]: AP 95, Magic pen 40%
- Tooltip (client en_US, values substituted): 95 Ability Power |  40% Magic Penetration

### Voltaic Cyclosword (6699)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 963, sell 2100 (70%) | recipe: The Brutalizer + Long Sword + Long Sword + 963 g
- Item groups (client): {0af4d11f}(max 1)
- Stats [CLIENT]: AD 55, Ability haste 10, Lethality 10
- mDataValues [CLIENT]: PercentCurrentHPMelee=9, PercentCurrentHPRanged=7, LethalityBonusModMelee=15, LethalityBonusModRanged=12, LethalityBonusDuration=4, LethalityAmount=10, NonChampCap=200
- Tooltip (client en_US, values substituted): 55 Attack Damage |  10 Lethality |  10 Ability HasteGalvanize | Damaging an Enemy Champions with an ability triggers Energized if it is ready. | Firmament | Your Energized Attack deals [PercentHPMeleeRangedSplit]% target's Current Health as bonus physical damage and grants you  [LethalityBonusModMeleeRangedSplit] Lethality for 4 seconds. | Damage is capped to 200 against non-Champions.
- Wiki pass "Energized": Moving and basic attacking generates Energize stacks, up to 100.
- Wiki pass2 "Galvanize": Dealing ability damage to an enemy champion, with an ability cast instance, triggers the effects of Energized attacks against them if they are ready.
- Wiki pass3 "Firmament": When fully Energized, your next basic attack on-hit grants you 15 (melee) / 12 (ranged) lethality for 4 seconds and deals bonus physical damage equal to 9% (melee) / 7% (ranged) of the target's current health, capped at 200 against non-champions.

### Warmog's Armor (3083)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3100, combine 500, sell 2170 (70%) | recipe: Giant's Belt + Giant's Belt + Crystalline Bracer + 500 g
- Item groups (client): 3083(max 1)
- Flags: active spell WarmogsDummySpell
- Stats [CLIENT]: Health 1000, Base HP regen 100% of base
- mDataValues [CLIENT]: MaxHealthRatio=0.015, SecondsPerHeal=0.5, HealthThreshold=2000, OOCTimerChampion=8, OOCTimer=3, HPAmp=0.12
- Calculations [CLIENT]: `TotalHealing = MaxHealthRatio(=0.015) x MaxHP`; `TotalHealingTooltip = calc[TotalHealing] * 2`
- Tooltip (client en_US, values substituted): 1000 Health |  100% Base Health RegenWarmog's Heart  | If you have 2000 bonus Health and have not taken damage within 8 seconds, restore [TotalHealingTooltip] Health per second. | Warmog's Vitality | Gain bonus Health equal to 12% of your Item Health ([f2]).Non-champion damage disables Warmog's Heart for 3 seconds instead.
- Wiki pass: Grants Warmog's Heart if you have at least 2000 bonus health.
- Wiki pass2 "Warmog's Heart": Health regenerationif damage has not been taken in the last 8 seconds (3 seconds for damage from non-champions).
- Wiki pass3 "Warmog's Vitality": Gain bonus health equal to 12% bonus health from items.
- Hooks: STAT_DYN (+12% item HP); PERIODIC regen when out of damage 8 s (3 s for non-champion damage) and >=2000 bonus HP [TOP-LANE PRIORITY]
- Implementation: Warmog's Heart: if bonusHP >= 2000 and no damage taken for 8 s (non-champion damage only resets a 3 s timer), restore 1.5% max HP every 0.5 s (3%/s) [client SecondsPerHeal 0.5, TotalHealingTooltip x2].

### Whispering Circlet (2526)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2250, combine 850, sell 1575 (70%) | recipe: Forbidden Idol + Ruby Crystal + Tear of the Goddess + 850 g
- Builds into (client build hint): Diadem of Songs
- Item groups (client): {9e0158a0}(max 1), TearItems(max 1), {aa03aa1b}(max 1), {6beffccb}, {1fd09102} | wiki limit: Manaflow
- Stats [CLIENT]: Health 200, Mana 300, Base mana regen 75% of base, Heal & shield power 8%
- mDataValues [CLIENT]: ManaChargeAmmoCD=8, ManaChargeMaxAmmo=5, ManaPerCharge=4, MaxMana=360, InternalCDPerCastID=6.5
- Calculations [CLIENT]: `BonusHSPCalc = 0.005 x bonus Mana`
- Tooltip (client en_US, values substituted): 200 Health |  8% Heal and Shield Power |  [PercentBaseMPRegenMod*100]% Base Mana Regen |  [FlatMPPoolMod] ManaHarmony | Gain [BonusHSPCalc]% Heal and Shield Power. | Manaflow  (8s, max 5 charges) | Landing Abilities grants 4 max Mana (doubled vs. champions). | Transforms into Diadem of Songs at 360 max Mana.
- Wiki pass "Harmony": Grants heal and shield power equal to 0.5% bonus mana.
- Wiki pass2 "Manaflow": Grants a charge every 8 seconds, up to 5 charges. Consumes a charge on-hit and whenever affecting an enemy or ally with an ability to grant 4 bonus mana, increased to 8 for champion targets, up to a maximum of 360 bonus mana. Can only be triggered once per cast instance.
- Wiki pass3: Transforms into Diadem of Songs at 360 bonus mana.

### Winter's Approach (3119)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2400, combine 300, sell 1680 (70%) | recipe: Tear of the Goddess + Giant's Belt + Kindlegem + 300 g
- Item groups (client): TearItems(max 1), 3119(max 1), 3121(max 1) | wiki limit: Manaflow
- Stats [CLIENT]: Health 550, Ability haste 15, Mana 500
- mDataValues [CLIENT]: ManaPerCharge=3, ManaChargeAmmoCD=8, ManaChargeMaxAmmo=4, MaxMana=360, InternalCDPerCastID=6.5
- Calculations [CLIENT]: `BonusHPFromMana = 0.15 x bonus Mana`
- Tooltip (client en_US, values substituted): 550 Health |  [FlatMPPoolMod] Mana |  15 Ability HasteAwe | Gain [BonusHPFromMana] Health. | Manaflow  (8s, max 4 charges) | Landing Attacks and Abilities grant 3 max Mana (doubled vs. champions). | Transforms into Fimbulwinter at 360 max Mana.
- Wiki pass "Awe": Grants bonus health equal to 15% bonus mana.
- Wiki pass2 "Manaflow": Grants a charge every 8 seconds, up to 4 charges. Consumes a charge on-hit and whenever affecting an enemy or ally with an ability to grant 3 bonus mana, increased to 6 for champion targets, up to a maximum of 360 bonus mana.
- Wiki pass3: Transforms into Fimbulwinter at 360 bonus mana.

### Wit's End (3091)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2800, combine 550, sell 1960 (70%) | recipe: Recurve Bow + Negatron Cloak + Recurve Bow + 550 g
- Item groups (client): 3091(max 1)
- Stats [CLIENT]: MR 45, Attack speed 50% (bonus AS ratio), Tenacity 20%
- Calculations [CLIENT]: `OnHitDamage = 45`
- Tooltip (client en_US, values substituted): 50% Attack Speed |  45 Magic Resist |  20% TenacityFray | Attacks deal [OnHitDamage] bonus magic damage .
- Wiki pass "Fray": Basic attacks deal 45 bonus magic damage on-hit.
- Hooks: ON_HIT +45 magic

### Youmuu's Ghostblade (3142)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2800, combine 675, sell 1960 (70%) | recipe: Serrated Dirk + Rectrix + Long Sword + 675 g
- Item groups (client): 3142(max 1)
- Flags: active spell YoumusBlade
- Stats [CLIENT]: AD 55, MS 4% (additive % MS), Lethality 18
- mDataValues [CLIENT]: LethalityAmount=18, CombatTimer=3, Cooldown=45, MeleeItemCalcValueB=20, RangedItemCalcValueB=15, DurationNDV=6, OOCMSndv=20, BaseOOCMS=20
- Calculations [CLIENT]: `Duration = 6   [ranged holder: x 0.667]`; `OOCMS = {74ac7847}(=0)   [ranged holder: x 0.5]`
- Tooltip (client en_US, values substituted): 55 Attack Damage |  18 Lethality |  4% Move SpeedHaunt  | Gain [OOCMS] Move Speed while out of combat. Wraith Step (cd: Cooldown) | Gain % Move Speed and Ghosting for [Duration] seconds.
- Wiki act "Wraith Step": Gain 20% (melee) / 15% (ranged) bonus movement speed and ghosted for 6 (melee) / 4 (ranged) seconds. (cd 45)
- Wiki pass "Haunt": Gain 20 (melee) / 10 (ranged) bonus movement speed while out-of-combat with enemy champions for 3 seconds.
- Hooks: STAT; ACTIVE (deferred); out-of-combat MS [DEFERRED]

### Yun Tal Wildarrows (3032)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3000, combine 750, sell 2100 (70%) | recipe: B. F. Sword + Scout's Slingshot + Long Sword + 750 g
- Item groups (client): {36d64d57}(max 1), {040e02e8}
- Stats [CLIENT]: AD 50, Attack speed 45% (bonus AS ratio)
- mDataValues [CLIENT]: ASDuration=6, CritMax=25, Cooldown=30, AACDR=1, CritCDR=2, ASMod=0.3, CritPerStackMelee=0.4, StackRangedMultiplier=0.5
- Calculations [CLIENT]: `CritPerStackCalc = {52d90ad0}(=0)   [ranged holder: x {1eea3407}(=0)]`
- Tooltip (client en_US, values substituted): 50 Attack Damage |  45% Attack Speed |  [FlatCritChanceMod*100]% Critical Strike ChancePractice Makes Lethal | On-Attack, gain + [CritPerStackCalc]% Critical Strike Chance permanently up to 25%. | Flurry (cd: Cooldown) | On-Attacking an enemy champion, gain 30% Attack Speed for 6 seconds.  | Attacks reduce this cooldown by 1 second, increased to 2 seconds for Critical Strikes.
- Wiki pass "Practice Makes Lethal": Basic attacks on-attack grant 0.4% (melee) / 0.2% (ranged) critical strike chance permanently, stacking up to 63 (melee) / 125 (ranged) times (capped at 25% critical strike chance).
- Wiki pass2 "Flurry": Launching a basic attack against an enemy champion grants you 30% bonus attack speed for 6 seconds (30 second cooldown, reduced by 1 second on-hit and 2 seconds if the attack critical strike).

### Zeke's Convergence (3050)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 2200, combine 700, sell 1540 (70%) | recipe: Kindlegem + Cloth Armor + Null-Magic Mantle + 700 g
- Item groups (client): 3050(max 1)
- Stats [CLIENT]: Health 300, Armor 25, MR 25, Ability haste 10
- mDataValues [CLIENT]: Duration=5, Cooldown=45, SlowAmount=0.3, DamagePerSecond=30, StormRadius=350, UltimateHaste=15, ReadyDuration=5
- Tooltip (client en_US, values substituted): 300 Health |  25 Armor |  25 Magic Resist |  10 Ability HasteCryocombustion | Gain 15 Ultimate Ability Haste. | Frostfire Tempest (cd: Cooldown) | Casting your Ultimate readies a storm for 5 seconds.  | When you get near an enemy champion, summon the storm around you for 5 seconds, dealing 30 magic damage per second and Slowing enemies affected by 30%.
- Wiki pass "Cryocombustion": Gain 15 ultimate haste.
- Wiki pass2 "Frostfire Tempest": Upon casting your ultimate ability, and once an enemy champion is within Frostfire Tempest's radius or after 5 seconds otherwise, you summon a storm of flame and ice around you for 5 seconds. The storm dealsto enemy champions and monsters within a cr 350 radius and slow them by 30% (45 second cooldown, starts on ultimate cast).

### Zhonya's Hourglass (3157)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 3250, combine 450, sell 2275 (70%) | recipe: Needlessly Large Rod + Seeker's Armguard + 450 g
- Item groups (client): BuildsFromStopwatchGroup, 3157(max 1), {139a0cb0} | wiki limit: Stasis
- Flags: active spell ZhonyasHourglass
- Stats [CLIENT]: AP 105, Armor 50
- mDataValues [CLIENT]: Cooldown=120, Duration=2.5
- Tooltip (client en_US, values substituted): 105 Ability Power |  50 Armor Time Stop (cd: Cooldown) | Enter Stasis for 2.5 seconds.
- Wiki act "Time Stop": Put yourself in stasis (buff) for 2.5 seconds, rendering you untargetable and invulnerable for the duration but also unable to move, declare basic attacks, cast abilities, use summoner spells, or activate items. (cd 120)


## Transformed (not purchasable directly) (4)

### Diadem of Songs (2530)
- Tier (wiki): =>Whispering Circlet; client epicness: 5; in SR store: False
- Cost: total 2250, combine 2250, sell 1575 (70%)
- Item groups (client): {aa03aa1b}(max 1), {6beffccb}
- Flags: transforms from Whispering Circlet
- Stats [CLIENT]: Health 200, Mana 1000, Base mana regen 100% of base, Heal & shield power 8%
- mDataValues [CLIENT]: PercentManaToHeal=0.01, AllyRangeCheck=900, AllyCombatDuration=3
- Calculations [CLIENT]: `BonusHSPCalc = 0.005 x bonus Mana`; `ManaToHeal = 0.008 x Mana`
- Tooltip (client en_US, values substituted): 200 Health |  8% Heal and Shield Power |  [FlatMPPoolMod] Mana |  [PercentBaseMPRegenMod*100]% Base Mana RegenHarmony | Gain [BonusHSPCalc]% Heal and Shield Power. | Consonance | While you or any ally you've healed or shielded in the last 3 seconds is in combat with champions, each second, heal the lowest health nearby ally champion for [ManaToHeal].
- Wiki pass "Harmony": Grants heal and shield power equal to 0.5% bonus mana.
- Wiki pass2 "Consonance": Heal the nearest and most wounded [Lowest health percent] allied champion within 900 units for 0.8% of your maximum mana if you or any allied champion you have healed or shield in the last 3 seconds is in combat with enemy champions (1 second cooldown for triggering the heal).

### Fimbulwinter (3121)
- Tier (wiki): ; client epicness: 5; in SR store: False
- Cost: total 2400, combine 2400, sell 1680 (70%)
- Item groups (client): 3121(max 1)
- Flags: transforms from Winter's Approach
- Stats [CLIENT]: Health 550, Ability haste 15, Mana 1000
- mDataValues [CLIENT]: Cooldown=8, ShieldDuration=3, CurrentManaShieldRatio=0.045, Multiplier=0.8
- Calculations [CLIENT]: `BonusHPFromMana = 0.15 x bonus Mana`; `ShieldBase = 100`
- Tooltip (client en_US, values substituted): 550 Health |  [FlatMPPoolMod] Mana |  15 Ability HasteAwe | Gain [BonusHPFromMana] Health. | Everlasting (cd: Cooldown) | Immobilizing or Slowing ( Melee only) an enemy champion grants a Shield that absorbs [ShieldBase] plus 4.5% current Mana damage for 3 seconds.  | The Shield is increased by 80% if more than one enemy is nearby.

### Muramana (3042)
- Tier (wiki): ; client epicness: 5; in SR store: False
- Cost: total 2900, combine 2900, sell 2030 (70%)
- Item groups (client): {a4ceabbc}(max 1), {7be37a10}
- Flags: transforms from Manamune
- Stats [CLIENT]: AD 35, Ability haste 15, Mana 1000
- mDataValues [CLIENT]: BonusADManaRatioTOOLTIPONLY=0.02, OnHitManaRatioTOOLTIPONLY=0.012, AbilityManaRatioRangedTOOLTIPONLY=0.03, AbilityTADRatio=0, PerCastIDLockout=6.5, AbilityManaRatioMeleeTOOLTIPONLY=0.04
- Calculations [CLIENT]: `BonusADFromMana = 0.02 x Mana`; `OnHitDamage = 0.012 x Mana`; `MeleeItemCalcValue = 0.04 x Mana`; `RangedItemCalcValue = 0.03 x Mana`
- Tooltip (client en_US, values substituted): 35 Attack Damage |  [FlatMPPoolMod] Mana |  15 Ability HasteAwe | Gain [BonusADFromMana] bonus Attack Damage. | Shock | Attacks against champions deal [OnHitDamage] bonus physical damage .  | Damaging Abilities against champions deal [melee/ranged split: calc MeleeItemCalcValue / RangedItemCalcValue] bonus physical damage.

### Seraph's Embrace (3040)
- Tier (wiki): =>Archangel's Staff; client epicness: 5; in SR store: False
- Cost: total 2900, combine 2900, sell 2030 (70%)
- Item groups (client): {a6ceaee2}(max 1), LifelineItems(max 1), {7be37a10} | wiki limit: Lifeline
- Flags: transforms from Archangel's Staff
- Stats [CLIENT]: AP 70, Ability haste 25, Mana 1000
- mDataValues [CLIENT]: APFromMana=0.02, HealthThreshold=0.3, ShieldDuration=3, LifelineCooldown=90, Cooldown=90
- Calculations [CLIENT]: `ShieldValue = 0.18 x Mana`; `BonusAPCalc = 0.02 x bonus Mana`
- Tooltip (client en_US, values substituted): 70 Ability Power |  [FlatMPPoolMod] Mana |  25 Ability HasteAwe | Gain [BonusAPCalc] Ability Power. | Lifeline (cd: Cooldown) | Taking damage that would reduce your Health below 30% grants a [ShieldValue] Shield for 3 seconds.
- Wiki pass "Awe": Grants ability power equal to 2% bonus mana.
- Wiki pass2 "Lifeline": If you would take damage that would reduce you below 30% of your maximum health, you first gain a shield for 3 seconds that absorbs damage equal to 18% maximum mana for 3 seconds. (cd 90)


## Support quest line (6)

### Bloodsong (3877)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 400, combine 0, sell 160 (40%) | recipe: Bounty of Worlds + 0 g
- Item groups (client): GoldItems(max 1), {57352a0f}(max 1), {2d85f9c5}, DoransItems(max 1) | wiki limit: Starter
- Flags: requires buff/currency S11Support_Quest_Completion_Buff; active spell ItemGhostWard
- Stats [CLIENT]: Health 200, Base HP regen 75% of base, Base mana regen 75% of base
- mDataValues [CLIENT]: StealthWardCap=4, IncreasedWardRange=300, GP10=9, BaseADRatio=0.1, SheenMult=1, SpellbladeCooldown=1.5, DebuffDuration=4, MeleeDamageAmp=0.08, RangedDamageAmp=0.05, Cooldown=1.5
- Calculations [CLIENT]: `{ce8aadac} = BaseADRatio(=0.1) x base AD`; `SpellbladeDamage = SheenMult(=1) x base AD`
- Tooltip (client en_US, values substituted): Requires completing the Support Quest from World Atlas |  200 Health |  75% Base Health Regen |  [PercentBaseMPRegenMod*100]% Base Mana Regen |  [Effect1Amount] Gold Per 10 SecondsSpellblade (cd: Cooldown) | After using an Ability, your next Attack deals [SpellbladeDamage] bonus physical damage . If the target is a champion, they take  8 % /  5% increased damage for 4 seconds. Active (4 charges) | Places an Invisible Stealth Ward that grants vision. | Recharges upon visiting the shop.
- Wiki act "Ward": Consumes a charge to place a Stealth Ward at the target location, which grants sight of the surrounding area. Charges refill upon visiting the shop. (cd {{fd|0.5}})
- Wiki pass "Spellblade": After using an ability, your next basic attack within 10 seconds deals 100% base AD bonus physical damage on-hit. If the target is a champion, inflict them with Expose Weakness for 4 seconds, causing them to take 8% (melee) / 5% (ranged) increased damage post-mitigation from all sources (1.5 second cooldown, starts after using the empowered attack).

### Celestial Opposition (3869)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 400, combine 0, sell 160 (40%) | recipe: Bounty of Worlds + 0 g
- Item groups (client): GoldItems(max 1), {2d85f9c5}, DoransItems(max 1) | wiki limit: Starter
- Flags: requires buff/currency S11Support_Quest_Completion_Buff; active spell ItemGhostWard
- Stats [CLIENT]: Health 200, Base HP regen 75% of base, Base mana regen 75% of base
- mDataValues [CLIENT]: StealthWardCap=4, IncreasedWardRange=300, GP10=9, Cooldown=18, MeleeShieldDRPercentage=0.35, RangedShieldDRPercentage=0.25, ShieldLingerAfterInitiallyPopped=2, Radius=500, SlowDuration=1.5, SlowAmount=0.5
- Tooltip (client en_US, values substituted): Requires completing the Support Quest from World Atlas |  200 Health |  75% Base Health Regen |  [PercentBaseMPRegenMod*100]% Base Mana Regen |  [Effect1Amount] Gold Per 10 SecondsBlessing of the Mountain (cd: Cooldown) | Reduce incoming champion damage by  35% /  25% for 2 seconds after taking damage from a champion. When the effect ends, slow nearby enemies by 50% for 1.5 seconds. Active (4 charges) | Places an Invisible Stealth Ward that grants vision. | Recharges upon visiting the shop. | Item cooldown is restarted when damage is taken from champions.
- Wiki act "Ward": Consumes a charge to place a Stealth Ward at the target location, which grants sight of the surrounding area. Charges refill upon visiting the shop. (cd {{fd|0.5}})
- Wiki pass "Blessing of the Mountain": Become Blessed to reduce incoming champion damage by 35% (melee) / 25% (ranged) pre-mitigation [Damage calculated before modifiers], lingering for 2 seconds after taking damage from a champion. After the linger ends, you lose Blessed to unleash a shockwave around you that slow enemies within 500 units by 50% for 1.5 seconds (18 second cooldown, timer restarts upon taking damage from champions).

### Dream Maker (3870)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 400, combine 0, sell 160 (40%) | recipe: Bounty of Worlds + 0 g
- Item groups (client): GoldItems(max 1), {2d85f9c5}, DoransItems(max 1) | wiki limit: Starter
- Flags: requires buff/currency S11Support_Quest_Completion_Buff; active spell ItemGhostWard
- Stats [CLIENT]: Health 200, Base HP regen 75% of base, Base mana regen 75% of base
- mDataValues [CLIENT]: StealthWardCap=4, IncreasedWardRange=300, GP10=9, RechargeTime=8, MaxBubbleStack=2, BubbleDuration=3, PurpleBubbleAoEMod=0.333, MinHealAmount=20, Cooldown=8
- Calculations [CLIENT]: `FlatDR = level_bp(L1=50, +12/level at L>=7)`; `ProcDmg = level_bp(L1=40, +10/level at L>=7)`
- Tooltip (client en_US, values substituted): Requires completing the Support Quest from World Atlas |  200 Health |  75% Base Health Regen |  [PercentBaseMPRegenMod*100]% Base Mana Regen |  [Effect1Amount] Gold Per 10 SecondsDream Maker (cd: Cooldown) | Healing or Shielding another ally blows Dream Bubbles to them for 3 seconds. Their next damaging attack or ability deals [ProcDmg] bonus magic damage and the next instance of damage they take is reduced by [FlatDR]. Active (4 charges) | Places an Invisible Stealth Ward that grants vision. | Recharges upon visiting the shop. | Deals 33.3% damage to non-champions when empowering area of effect Abilities.
- Wiki act "Ward": Consumes a charge to place a Stealth Ward at the target location, which grants sight of the surrounding area. Charges refill upon visiting the shop. (cd {{fd|0.5}})
- Wiki pass "Dream Maker": Every 8 seconds, you gain a Blue Dream Bubble and a Purple Dream Bubble. Granting a heal or shield to an allied champion (excluding yourself) causes you to grant both of your Dream Bubbles to them, empowering them for 3 seconds. The Blue Bubble reduces the damage of the next attack or spell they receive from non-minions by [levels: 50 to 194 for 13|1;7 to 18|type=your level] and the Purple Bubble grants them [levels: 40 to 160 for 13 on their next damaging attack or ability, with the latter reduced to 33.3% for area of effect damage against non-champions.

### Solstice Sleigh (3876)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 400, combine 0, sell 160 (40%) | recipe: Bounty of Worlds + 0 g
- Item groups (client): GoldItems(max 1), {2d85f9c5}, DoransItems(max 1) | wiki limit: Starter
- Flags: requires buff/currency S11Support_Quest_Completion_Buff; active spell ItemGhostWard
- Stats [CLIENT]: Health 200, Base HP regen 75% of base, Base mana regen 75% of base
- mDataValues [CLIENT]: StealthWardCap=4, IncreasedWardRange=300, GP10=9, Cooldown=30, BuffDuration=2.5, MoveSpeedBuff=0.2, MoveSpeedRange=1500, BonusHealthBuffRatio=0.07
- Calculations [CLIENT]: `BonusHealthBuff = level_bp(L1=50, +15/level at L>=7)`
- Tooltip (client en_US, values substituted): Requires completing the Support Quest from World Atlas |  200 Health |  75% Base Health Regen |  [PercentBaseMPRegenMod*100]% Base Mana Regen |  [Effect1Amount] Gold Per 10 SecondsGoing Sledding (cd: Cooldown) | Slowing or Immobilizing an enemy champion near allies restores [BonusHealthBuff] Health and grants 20% decaying Move Speed for 2.5 seconds to you and a nearby ally.  Active (4 charges) | Places an Invisible Stealth Ward that grants vision. | Recharges upon visiting the shop.
- Wiki act "Ward": Consumes a charge to place a Stealth Ward at the target location, which grants sight of the surrounding area. Charges refill upon visiting the shop. (cd {{fd|0.5}})
- Wiki pass "Going Sledding": Slow or immobilize an enemy champion causes you and the most wounded [Lowest health percent] allied champion within 1500 units to gain 20% bonus movement speed decaying over 2.5 seconds and [levels: 50 to 230 for 13 bonus health for 2.5 seconds. (cd 30)

### World Atlas (3865)
- Tier (wiki): Starter; client epicness: 1; in SR store: True
- Cost: total 400, combine 400, sell cannot be sold (mCanBeSold unset)
- Item groups (client): GoldItems(max 1), DoransItems(max 1), {86e9a5ac} | wiki limit: Starter
- Flags: requires buff/currency SupportItemPurchaseBuff
- Stats [CLIENT]: Base HP regen 50% of base, Base mana regen 25% of base
- mDataValues [CLIENT]: GP10=3, QuestGoldRequirement=400, ChargeCooldown=20, MaxCharges=3, AllyChampNearbyRadius=2000, GoldOnHit=18, GoldOnHitMelee=18, ExecuteMinionGold=18, MeleeExecutePerc=0.5, RangedExecutePerc=0.333, ExecuteCannonGold=20, AllyChampNearbyRadiusMinion=1300, FirstChargeOffset=20
- Calculations [CLIENT]: `MeleeItemCalcValue = GoldOnHitMelee(=18)`; `RangedItemCalcValue = GoldOnHit(=18)`; `{bdd8f3f7} = {d4217cda}(=0)`; `{1755c90e} = {9f631779}(=0)`; `ExecutePercent = ranged ? calc[{1755c90e}] : calc[{bdd8f3f7}]`; `ExecuteDamage = 1 x MaxHP + 1 x AD`; `{79b6144b} = {"DamageType": 2, "mConditionalGameCalculation": "ExecuteDamage", "mConditionalCalculationRequirements": {"mSubRequirements": [{"mInvertResult": true, "{6166b756}": "ExecuteHealthThreshold", "__type": "{43b8e695}"}, {"mUnitsRequirements": [{"__type": "SameTeamCastRequirement"}, {"mUnitTags": {"mObjectTagList": ["{ea595ca6}"], "__type": "ObjectTags"}, "__type": "HasUnitTagsCastRequirement"}], "mRange": 1300.0, "__type": "HasNNearbyVisibleUnitsRequirement"}], "__type": "HasAllSubRequirementsCastRequirement"}, "{353099b0}": true, "__type": "GameCalculationConditional"}`; `ExecuteHealthThreshold = (calc[ExecutePercent]) x MaxHP + 1 x AD`
- Tooltip (client en_US, values substituted): 50% Base Health Regen |  [PercentBaseMPRegenMod*100]% Base Mana Regen |  [Effect1Amount] Gold Per 10 SecondsSupport Quest | Earn 400 gold from this item to transform it into Runic Compass.  | Shared Riches  (20s, max 3 charges) | While near an ally champion, damage enemy champions or kill minions to gain gold.Damaging Abilities and Attacks against champions or structures grant [melee/ranged split: calc MeleeItemCalcValue / RangedItemCalcValue] gold;Killing a minion grants you 18 gold and the nearest allied champion the kill gold.
- Wiki pass "Support Quest": Earn 400 through Shared Riches and this item's gold generation stat to automatically upgrade to Runic Compass, gaining the Ward active with 3 wards in stock.
- Wiki pass2 "Shared Riches": Grants a Shared Riches charge every 20 seconds, up to 3 charges. When an allied champion is within er 1050 units of you, consume a charge in the following ways:<ul><li>Kill a minion by any means, granting you 18 and the nearest allied champion kill gold. Damaging a minion below 50% (melee) / 30% (ranged) of its maximum health with a basic attack will execute it, if any charges are available.<li>Damage an enemy champion or structure with a basic attack or ability, granting you 18. A charge may be consumed this way only once per attack or ability.

### Zaz'Zak's Realmspike (3871)
- Tier (wiki): Legendary; client epicness: 5; in SR store: True
- Cost: total 400, combine 0, sell 160 (40%) | recipe: Bounty of Worlds + 0 g
- Item groups (client): GoldItems(max 1), {2d85f9c5}, DoransItems(max 1) | wiki limit: Starter
- Flags: requires buff/currency S11Support_Quest_Completion_Buff; active spell ItemGhostWard
- Stats [CLIENT]: Health 200, Base HP regen 75% of base, Base mana regen 75% of base
- mDataValues [CLIENT]: StealthWardCap=4, IncreasedWardRange=300, GP10=9, APRatio=0.15, PercentHPDamage=0.03, MonsterDamageCap=300, BaseDamage=10
- Calculations [CLIENT]: `Cooldown = 10`; `{425b6f6e} = APRatio(=0.15) x AP`; `TooltipDamage = BaseDamage(=10) + APRatio(=0.15) x AP`
- Tooltip (client en_US, values substituted): Requires completing the Support Quest from World Atlas |  200 Health |  75% Base Health Regen |  [PercentBaseMPRegenMod*100]% Base Mana Regen |  [Effect1Amount] Gold Per 10 SecondsVoid Explosion (cd: Cooldown) | Dealing Ability damage to a champion causes an explosion that deals [TooltipDamage] + 3% max Health magic damage. Active (4 charges) | Places an Invisible Stealth Ward that grants vision. | Recharges upon visiting the shop. | Damage is capped at 300 against monsters.
- Wiki act "Ward": Consumes a charge to place a Stealth Ward at the target location, which grants sight of the surrounding area. Charges refill upon visiting the shop. (cd {{fd|0.5}})
- Wiki pass "Void Explosion": Dealing ability damage to an enemy champion creates an explosion at their location after a 0.5-second delay, dealing 10 (+ 15% AP) (+ 3% of each target's maximum health) magic damage to enemies within the area, capped at 300 against monsters. (cd 10)


## Champion-specific (3)

### Kalista's Black Spear (3599)
- Tier (wiki): Starter; client epicness: None; in SR store: True
- Cost: total 0, combine 0, sell 0 (70%)
- Item groups (client): TheBlackSpear(max 1)
- Flags: active spell KalistaPSpellCast
- Stats [CLIENT]: none
- Tooltip (client en_US, values substituted): Bind with an ally for the remainder of the game, becoming Oathsworn Allies. Oathsworn empowers you both while near one another.
- Wiki act "Oathsworn Bond": Consumes this item to initiate a 3.5-second cast time and a 3-second channel afterwards from the user and the target allied champion, both becoming bound allies. The target is unable to act for 6 seconds after the channel's duration. Afterwards, the target becomes an Oathsworn.

### Kalista's Black Spear (3600)
- Tier (wiki): ; client epicness: None; in SR store: True
- Cost: total 0, combine 0, sell 0 (70%)
- Item groups (client): TheBlackSpear(max 1)
- Flags: active spell KalistaPSpellCast
- Stats [CLIENT]: none
- Tooltip (client en_US, values substituted): Bind with an ally for the remainder of the game, becoming Oathsworn Allies. Oathsworn empowers you both while near one another. | Required to use Kalista's Ultimate Ability.

### Scarecrow Effigy (3330)
- Tier (wiki): Trinket; client epicness: 1; in SR store: True
- Cost: total 0, combine 0, sell cannot be sold (mCanBeSold unset)
- Item groups (client): Trinket
- Flags: active spell FiddleSticksScarecrowEffigy
- Stats [CLIENT]: none
- Calculations [CLIENT]: `EffigyDuration = lerp_level(130 -> 300)`; `AMMORECHARGETIME = lerp_level(115 -> 30)`
- Tooltip (client en_US, values substituted): Cannot be sold |  Places an effigy that lasts for [EffigyDuration] seconds and appears exactly as Fiddlesticks does to enemies. Stores one charge every [AmmoRechargeTime] seconds, up to maximum 2 charges. | Enemy champions approaching an effigy will activate it, causing the effigy to fake a random action, after which the effigy will fall apart.
- Wiki act "Trinket": Consume a charge to place a visible Effigy at the target location, which grants sight over the surrounding area for [levels: 130 to 300|tooltipSize=20] seconds. For enemies, it visually appears identical to Fiddlesticks (including on the minimap) and has no visible health bar until it activates. Enemy champion that approach it will activate it, causing it to automatically sound a Danger ping to its allies as well as fake a random action for up to 2 seconds. If not destroyed by that time, it will deal 1 damage to itself. (cd 2)


## Jungle (DEFERRED: listed only) (3)

- Gustwalker Hatchling (1102): 450 g, requires Smite, DoransItems/GoldItems/HuntersTalismanGroup max 1; jungle companion + Smite upgrades. DEFERRED per MODERN-009.
- Mosstomper Seedling (1103): 450 g, requires Smite, DoransItems/GoldItems/HuntersTalismanGroup max 1; jungle companion + Smite upgrades. DEFERRED per MODERN-009.
- Scorchclaw Pup (1101): 450 g, requires Smite, DoransItems/GoldItems/HuntersTalismanGroup max 1; jungle companion + Smite upgrades. DEFERRED per MODERN-009.
