# ITEMS.md: modern item system spec, Summoner's Rift, patch 26.19

**Scope.** This spec covers every item purchasable on Summoner's Rift (map 11, CLASSIC 5v5) in patch **26.19** (client build **16.19.8230722**, normal PC):
- inventory and shop rules;
- uniqueness and item groups;
- stat contributions;
- shared item mechanics (Spellblade, Cleave, Lifeline, Grievous Wounds, Thorns, Immolate, Energized, Helping Hand, Manaflow, Glory and others);
- life steal and omnivamp;
- the full Tiamat/Ravenous/Titanic/Profane/Stridebreaker passives and actives;
- the hooks and state the simulator needs;
- every item change from 26.1 to 26.19.

The per-item detail for all 214 items in scope (210 SR store items plus the 4 transforms 3040, 3042, 3121 and 2530) is in **[ITEMS_CATALOG.md](ITEMS_CATALOG.md)**.

**Out of scope, deferred (MODERN-009).**
- Jungle items: Scorchclaw Pup 1101, Gustwalker Hatchling 1102 and Mosstomper Seedling 1103. They are listed, with no detail.
- All item actives except the Tiamat line and Stridebreaker. These are listed in §16 with a one-line description.
- Vision and ward behaviour. Only the item and inventory rules for wards are noted.
- Global damage-modifier ordering belongs to the damage-pipeline agent. This document only states each item's contribution.

**Patch pin.** 26.19 = client 16.19.8230722. CommunityDragon `https://raw.communitydragon.org/16.19/` is exactly this build (content-metadata confirmed in the cache README).

**Retrieval date.** 2026-10-01. The wiki was fetched 2026-10-01 at the revisions listed below, and the patch notes were fetched 2026-10-01.

## Sources

### Client data (priority 1)
These are in the cache `/mnt/nfs/shared/modern-world-map-research/cdragon-16.19/`, retrieved 2026-10-01 from `https://raw.communitydragon.org/16.19/`.

| File | sha256 | Used for |
|---|---|---|
| `items.cdtb.bin.json` | `6880f35d5a9d82f688192f764e280e6d4bc9c845112b001feb811e3f2ab62726` | ItemData (stats, price, recipe, groups, mDataValues, mItemCalculations, sellBackModifier, flags), ItemGroup (max ownable, purchase cooldowns, slot constraints), item SpellObjects (cast times, tags) |
| `cdragon-items.json` | `f126341e102bf33968afece5ea213b81f61086a5488e94eda4f22940fd82da09` | priceTotal cross-check |
| `shared.cdtb.bin.json` | `34f68553ab38cfe344936473fcb48d99574d5fd51e4feb42994fa0c4fe50769e` | Shared/Spells (SheenDelay, LifeLineCooldown, Titanic cleave projectile) |
| `en_us.lol.stringtable.json` (fetched by this research from `…/16.19/game/en_us/data/menu/en_us/lol.stringtable.json`, appended to SHA256SUMS) | `8c051cb2a24b31f3fa9af95d39085b832b0da2b8cf51093f98c4ab0620ecb8e8` | tooltip text, passive names |
| `../map11-decoded.json` (map11.bin, EUW1 16.19.8230722 extraction) | `3947d4dbb5cb10711546f2df9076ccc82d6dbff9ea5caefa906abdfdb8ebe145` | CLASSIC `GameModeMapData` (hash `{0b03bf5a}`, mode `{48246d53}` = fnv1a("CLASSIC")): 11 `itemLists` define the SR item pool. `mItemShopData` `{8f58df52}` |
| `../geometry-decoded.json` | `77c4d3bd62a70600d7ca1a4da4f04f60bbfb24c5f193e7f3aa8c4ccbc0495fb2` | shopkeeper objects, `ShopGeComponentDef`, `Order/ChaosShopArea*` locators |
| repo `lanerl_jax/data/modern/26.19/items.json` (DDragon 16.19.1, `source_sha256 72c996d0…`) | (repo file) | sell-value cross-check: 210/210 SR store items agree with total × sellBackModifier (default 0.7) |

Hash names were resolved with 32-bit FNV-1a over the lowercased path, e.g. `fnv1a("Items/1001") = {9d24457e}`.

### Riot patch notes (priority 2)
I audited all 19 patches from 26.1 to 26.19. 26.1–26.3 use the URL form `https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-N-notes/` and 26.4–26.19 use `…/league-of-legends-patch-26-N-notes/`. The HTML sha256 values are below.

| Patch | URL | sha256 (HTML as fetched 2026-10-01) |
|---|---|---|
| 26.1 | https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-1-notes/ | `f4bf41832279a42f396e10142434cc8919163f5319f6bbe43c56df4a0854ecb0` |
| 26.2 | https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-2-notes/ | `96ec8c8540e91b6c9855bddb714b100d6ba611ae8bffaefc8174565dd02e6d37` |
| 26.3 | https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-3-notes/ | `2b0eced2b359e2a30677316ad9491659a17f8a5b56e1b43da69a042368871510` |
| 26.4 | https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-4-notes/ | `eb3080f5bdefff81151fd12af1f2fa2c28e69d7f91b57cc5f1d899d9c4842c72` |
| 26.5 | https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-5-notes/ | `1585fe178f40205f8944c44c7e2beebf0c09513663be25133db3d209e6c941d1` |
| 26.6 | https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-6-notes/ | `9326f35ac02908eff8df5af08f1911095adce6596a7089fa5092773fea21e7af` |
| 26.7 | https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-7-notes/ | `ff89f913f5f852732dd7d89fd69b8a85f817b1dd5a862fcac3533cd93f25fdb5` |
| 26.8 | https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-8-notes/ | `39b2027edc5a073037dfe8c9d3aa40fce59cf836971feb53ac3cfa226000a7c3` |
| 26.9 | https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-9-notes/ | `f3dd0108f543ac47f7ccf4704f9a8423fdb08d1cfb73d66157b47f17c397567e` |
| 26.10 | https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-10-notes/ | `35ed57bd2cef637ff7928c788584b510e826daa84b9d182faf3bdf4a2922e9a9` |
| 26.11 | https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-11-notes/ | `be082bdee41349945634bf45e7ab95ca69a6f9900c003b9b5c239a495ed00e65` |
| 26.12 | https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-12-notes/ | `4e2ffddb7c57ec43afbc70976a5ebc9ee8a4df8564296ceedff4f91abae96e4c` |
| 26.13 | https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-13-notes/ | `563e5e9a15044baf71d3f60707db960f814f5a3a47260dae5aca6702bbcfff77` |
| 26.14 | https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-14-notes/ | `30e4e444dc1fb4ab869505834b0f02a94097f0576fbd2e3fe9e6025daa360003` |
| 26.15 | https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-15-notes/ | `6bd37da380c4f959f9576bdc6b9848790e5ecb5f0486b7c51a6b830f430eedfa` |
| 26.16 | https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-16-notes/ | `6f56f7b2ff4f6327d49577509ce073e66a48f4da7151f243690138eb5e67f87f` |
| 26.17 | https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-17-notes/ | `a31d43e435517a81189d2b53eb479f41a0bda1b62d60508de1420de14bb6be90` |
| 26.18 | https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-18-notes/ | `32d993727b5af788926c6c65c13a56d025ad7ffdf19fae65b2e2010bf3c62c5c` |
| 26.19 | https://www.leagueoflegends.com/en-us/news/game-updates/league-of-legends-patch-26-19-notes/ | `e201ab510ed8f877e506d77d5e4dd75a749dd3163a835562eb9db218004e21c1` |

Hotfix sections exist inside the 26.1, 26.3, 26.6 and 26.9 notes. Only the 26.1 hotfix (1/9/2026, Essence Reaver) touches an SR item. There are no standalone 26.x item hotfix articles.

### League wiki (priority 3)
Permalinks use `https://wiki.leagueoflegends.com/en-us/index.php?oldid=N`.

| Page | oldid | Page | oldid |
|---|---|---|---|
| Module:ItemData/data (content sha256 `3a43266a706001ea3e6e414189e7bd9dbcdbf9e3871209214bcff98d9ed8920a`) | 4069878 | Tiamat | 4064196 |
| Ravenous Hydra | 4047314 | Titanic Hydra | 4015389 |
| Profane Hydra | 4064205 | Stridebreaker | 4015388 |
| Sheen | 3981954 | Trinity Force | 3982284 |
| Template:Spellblade info | 4019664 | Named item effect | 4045631 |
| Item | 4050179 | Shop | 3982704 |
| Gold | 4039269 | Role Quests | 4064833 |
| Item group | 4030983 | Life steal | 4060425 |
| Vamp | 4058226 | Attack effects | 4050298 |
| Critical strike | 4059009 | Combat status | 4058480 |
| Spawn | 3994097 | Doran's Shield | 4026378 |
| Grievous Wounds | 3969410 | Movement speed | 4064468 |
| Slow resist | 4058780 | Tenacity | 4021927 |
| Haste | 4070414 | Health Potion | 3975594 |
| Refillable Potion | 3971312 | Elixir of Iron | 4013403 |
| Elixir of Wrath | 3993461 | Feats of Strength (removed 26.1) | 3982010 |

### Secondary
None. No third-party aggregator was used.

### Confidence tags
- **CLIENT** = CLIENT-DATA-VERIFIED: the value appears literally in 16.19.8230722 data.
- **NOTES** = RIOT-NOTES.
- **WIKI**.
- **INFERRED** = derived by me, with the reasoning stated.

Each tag carries a confidence of H, M or L.

**Default reconciliation rule.** Client numbers win over the notes and the wiki, because the notes describe deltas and the wiki lags. Behaviour that is *not* encoded in data (trigger semantics, ordering, geometry) comes from the wiki unless the client tooltip or a spell tag contradicts it.

---

## 1. Executive summary

1. **The SR item pool is defined by the client, not by DDragon's `maps["11"]`.** The CLASSIC item lists contain 266 IDs. 210 are in-store (`mInStore`) and 4 more are transforms; the rest are turret/structure "items", Gangplank upgrades, Triple Tonic elixirs and legacy not-in-store entries. The repo's `items.json` (DDragon `maps["11"]`) has 254 entries. It contains all 210 store items, but it also contains 44 non-SR mirrors (`32xxxx`/`66xxxx` swiftplay or Arena copies, Guardian's items, Cappa Juice, jungle variants 1105–1107), and it lacks the 4 transforms. [CLIENT, H]
2. **Uniqueness is data-driven.** It comes from `ItemGroup.mMaxGroupOwnable` (§4).
   - The groups that matter for a top laner: **Hydra** (Tiamat 3077, Ravenous 3074, Titanic 3748, Profane 6698, Stridebreaker 6631; max 1); **Spellblade** (Sheen, Trinity, Iceborn, Lich Bane, Essence Reaver, Dusk and Dawn, Bloodsong; max 1, shared 1.5 s SheenDelay); **Lifeline** (Sterak's, Maw, Hexdrinker, Shieldbow, Protoplasm, Archangel's, Seraph's; max 1, shared LifeLineCooldown); **Fatality/LastWhisper** (Last Whisper, Mortal Reminder, LDR, Black Cleaver, Serylda's, Terminus; max 1); **Boots** max 1; **DoransItems** (*all* starters, including support and jungle starters) max 1; **Potion** max 1; one copy of every legendary.
   - DDragon `items.json` has **no `itemLimit` at all**. The current loader therefore enforces only the hard-coded Hydra set. [CLIENT, H]
3. **The Hydra-line active geometry in the current code is wrong.** Crescent, Ravenous Crescent, Heretical Cleave and Breaking Shockwave are **450-radius circles centred 100 units in front of the caster**, not a forward half-plane and not caster-centred. [WIKI H; client `castConeDistance=100`]
   - Cast time = the caster's attack windup (capped at 0.2 s; Stridebreaker 0.25 s) [CLIENT `mUseAutoattackCastTimeData` + WIKI].
   - None of them is an attack reset except **Titanic Crescent** (spell tag `Trait_AttackReset`) [CLIENT, H].
   - Cleave is limited to **10 splashes per attack** (`MaxProcPerAuto=10`) [CLIENT, H].
4. **Season 2026 rule changes that affect items.** All are confirmed in the 26.19 data or notes:
   - base crit damage is 200% (IE gives +30%);
   - omnivamp heals 100%, but 33% vs minions/monsters for AoE/DoT/pet damage, and never from Smite or Ignite (26.4);
   - Tear items are semi-unique;
   - tier-3 boots come only from the **mid** role quest;
   - the bot quest moves boots into a role-quest slot (a 7th effective slot);
   - support Control Wards live in the quest slot;
   - Feats of Strength were removed;
   - 9 new legendaries in 26.1; Doran's Bow, Doran's Helm and Gluttonous Greaves in 26.9. [NOTES/CLIENT, H]
5. **Shop.**
   - Starting gold 500 [WIKI, H].
   - Sell refund = round(total × modifier), with modifier 0.7 by default and **0.4** for starters, Dark Seal, Cull, potions, elixirs, Control Ward, Rejuvenation Bead, Seeker's/Shattered Armguard, Guardian Angel and the support line [CLIENT, H].
   - Undo is unlimited until you leave shop range, enter combat, use the item, or an item transforms or is distributed [WIKI, M].
   - Purchase range is the fountain shop area: circle radius **1000** around the per-team shop-area centre [CLIENT field unnamed, INFERRED M].
6. **Top client-vs-notes/wiki disagreements.** The full register is §13; implement the client value in every case.
   - Doran's Bow AS: client 15%, 26.9 notes 12%.
   - Redemption cost: client 2300, notes and wiki 2250.
   - T3 boot shields: client 90 + 10·(L−8) + 8% bonus HP (100 at L9, 190 at L18), notes "100–200".
   - Doran's Shield AoE regen effectiveness: client tooltip 66%, wiki 75%.

---

## 2. Conventions for implementers

### 2.1 Units and stat semantics
These are client `ItemData` field semantics [CLIENT, H; units cross-checked against tooltip format strings].

| Client field | Meaning | Unit and stacking |
|---|---|---|
| `mFlatHPPoolMod` | +max health (bonus) | flat, additive |
| `mFlatPhysicalDamageMod` | +AD (bonus) | flat, additive |
| `mFlatMagicDamageMod` | +AP | flat, additive |
| `mFlatArmorMod` / `mFlatSpellBlockMod` | +armor / +MR (bonus) | flat, additive |
| `mPercentAttackSpeedMod` | +bonus attack-speed ratio (0.25 = +25%) | additive into the bonus-AS sum: `AS = baseAS × (1 + Σbonus + growth)` with the attack-speed ratio rules (owner: stats agent) |
| `mPercentMultiplicativeAttackSpeedMod` | multiplicative AS | 1 item in the bin, none on SR |
| `mFlatCritChanceMod` | crit chance (0.25 = 25%) | additive, capped at 100% [WIKI] |
| `mFlatCritDamageMod` | +crit damage (IE 0.30) | additive to the base 2.00 multiplier (26.1) |
| `mPercentLifeStealMod` | life steal | additive [WIKI] |
| `PercentOmnivampMod` | omnivamp | additive [WIKI] |
| `mFlatMovementSpeedMod` | flat MS (boots) | additive flat term, before percent |
| `mPercentMovementSpeedMod` | additive % MS | `(base + flat)·(1 + Σpct)·Π(1 + mult)·(1 − max slow·(1 − slowresist))` then soft caps [WIKI Movement speed 4064468] |
| `mAbilityHasteMod` | ability haste | additive, `CD/(1 + AH/100)` |
| `mFlatHPRegenMod` | flat HP regen **per second** (tooltip shows ×5 per 5 s) | additive |
| `mPercentBaseHPRegenMod` | +X × champion **base** HP regen (1.0 = +100%) | additive |
| `flatMPPoolMod` / `percentBaseMPRegenMod` | mana / +% base mana regen | additive |
| `PhysicalLethality` | lethality = flat armor pen (full value since 14.1) | additive |
| `mPercentArmorPenetrationMod` | % armor pen (Fatality items) | multiplicative stacking between sources, `1 − Π(1 − p)` (owner: damage pipeline); in practice only one Fatality item is ownable |
| `mFlatMagicPenetrationMod` / `mPercentMagicPenetrationMod` | magic pen | flat additive / % multiplicative |
| `mPercentTenacityItemMod` | tenacity | **multiplicative** between sources: `1 − Π(1 − t)` [WIKI] |
| `mPercentSlowResistMod` | slow resist | **multiplicative** between sources [WIKI] |
| `mPercentHealingAmountMod` | heal and shield power | additive |

**"Bonus" stats.** Every item stat line is a **bonus** stat. "Base AD" means champion base plus level growth only; everything else, including Sterak's Claws, Overlord's Tyranny and Mid-quest +8%, is bonus [WIKI Champion statistic 4069636, H].
- Item stat-calcs that read bonus stats must be evaluated **after** flat item stats are summed.
- Effects that *grant* bonus stats from other stats are dependent stats: Sterak's 50% base AD; Overlord's 2.5% bonus HP → AD; Warmog's +12% item HP; Riftmaker 2% bonus HP → AP; Archangel's/Manamune/Winter's from mana; Swiftmarch 5% MS → adaptive.
- Order the dependent stats so no cycle exists. Recommended pass order:
  1. flat item stats;
  2. Warmog's Vitality (+12% of *item* health only);
  3. HP-derived stats (Overlord's, Riftmaker, Winter's);
  4. base-AD derived (Sterak's);
  5. percent multipliers;
  6. MS-derived (Swiftmarch).

  This order is INFERRED (M). None of these are circular in data, because each reads a different source stat.

### 2.2 Level-scaled calculations
These come from client calc parts and are used by many items [CLIENT, H; formulas INFERRED H].

```
lerp_level(a, b, L)  = a + (b - a) * (max(L,1) - 1) / 17               # ByCharLevelInterpolation; extrapolates past 18 unless mScalePastDefaultMaxLevel=false (README X-1)
level_bp(v1, steps, L) = v1 + sum_over_steps( x * max(0, L - k + 1) )  # mBonusPerLevelAtAndAfter x at level k
                         + sum_over_onceSteps( x * [L >= k] )           # mAdditionalBonusAtThisLevel
                         + init * (L - 1)                               # mInitialBonusPerLevel
```
Verification: Locket `level_bp(290, +7 at L>=9)` gives 290 for L1–8 and 360 at L18, matching the 26.5 notes "290–360 scaling after level 8". Statikk bounces `4 +1 at 6,10,14,20` gives 4→7 at L18 and 8 at L20, matching the 26.9 notes "4–8". Top-lane quest champions reach level 20, which makes values at L19–20 relevant. See §15.

### 2.3 Melee/ranged splits
`IsRangedCastRequirement` / `mRangedMultiplier` select on the **holder's** ranged flag (champion `IsMelee`), not on the attack [CLIENT, H]. Garen, Darius, Jax and other melee champions always take the melee value.

---

## 3. Inventory model

| Rule | Value | Tag |
|---|---|---|
| Main slots | 6 (indices 0–5); item groups with `mInventorySlotMin/Max` constrain the slot | CLIENT H |
| Trinket slot | index 6; group `Trinket` has `mInventorySlotMin = mInventorySlotMax = 6`; only trinkets (3340, 3363, 3364; 3330 Fiddlesticks) | CLIENT H |
| Default trinket | Stealth Ward 3340 at game start (0 g); leaving base without a trinket auto-grants 3340 [WIKI Item]. Trinkets swap via `sidegradeItemLinks` (0 g). Farsight requires level 9; Oracle Lens requires level 1 | CLIENT H / WIKI |
| Role-quest slot | **Bot quest completed:** boots move to the quest slot, freeing a main slot (effective 7 slots). **Support:** Control Wards are stored in the quest slot from game start (26.3), 40 g each and up to 2 stored after quest completion. **Top quest (our case):** no extra slot; its reward is level cap 20, XP and Unleashed Teleport (owned by the quest agent) | NOTES 26.1/26.3 H |
| Stacking | `maxStack`: Health Potion 5, Control Ward 2, Total Biscuit 10, everything else 1. A stack occupies one slot | CLIENT H |
| Full inventory | Consumables flagged `consumeOnAcquire` are applied instantly when the inventory is full [WIKI Item]. Purchase of a non-stackable item with no free slot fails, unless it combines components already held (a recipe frees the component slots first) | WIKI M / INFERRED |
| Recipe consumption | Building an item removes **all components present in the inventory**, recursively. Missing components are bought implicitly at full price. Cost paid = `total(item) − Σ total(components consumed)` | WIKI Shop H |
| Group bypass | Combining a recipe or transformation **bypasses** an item-group limit: "excludes acquiring an item within the group by combining its recipe or via transformation" [WIKI Item 4050179]. This is why Tiamat → Ravenous is legal while holding Tiamat | WIKI H |
| Distributed items | Transforms and quest upgrades (Seraph's, Muramana, Fimbulwinter, Diadem, T3 boots, Shattered Armguard) go into the same slot | WIKI H |

### 3.1 Purchase restrictions (client flags)
- `mItemDataAvailability.mInStore` must be true. Otherwise the item is distributed only (transforms, T3 boots, Triple Tonic elixirs, Your Cut).
- `mRequiredLevel`: Elixirs of Iron, Sorcery and Wrath at **9**; Farsight at 9.
- `mRequiredSpellName = "SummonerSmite"`: jungle pets. **Role binding:** with Smite, you must buy a jungle item before any non-consumable item [WIKI].
- `mRequiredPurchaseIdentities = ["Ranged"]`: Runaan's Hurricane (3085) only.
- `mRequiredChampion`: Scarecrow Effigy (Fiddlesticks), Kalista's Black Spear (Kalista; 3600 = Sylas copy).
- `mRequiredBuffCurrencyName`:
  - T3 boots `Feats_NoxianBootPurchaseBuff`, granted by the mid role quest (the name is a leftover from 2025 Feats of Strength);
  - support items `SupportItemPurchaseBuff` / `S11Support_Quest_Completion_Buff`;
  - Shattered Armguard `Item2420`, after a Seeker's Armguard has been broken.
- Starters: the `DoransItems` group (max 1) contains **every** starter: Doran's Shield/Blade/Ring/Bow/Helm, the jungle pets and the World Atlas line. One starter of this kind per inventory. **Dark Seal, Cull and Tear are not in it** [CLIENT H]. So "Doran's Shield + Cull" or "Doran's Blade + Dark Seal" are legal.
- `Potion` group (max 1): Health Potion 2003 and Refillable Potion 2031 are mutually exclusive [CLIENT H]. Corrupting Potion 2033 is not in store at 26.19 [CLIENT H].
- Elixirs: group `Elixir` has **`mPurchaseCooldown = 5.0` s** and no max. Drinking a different elixir replaces the active one ("Drinking a different Elixir will replace the existing one's effects", client tooltip). The Iron-only group `{dd9ecf6f}` has max 1 [CLIENT H].
- Gold: purchase requires `gold >= cost` (no debt) [WIKI].

### 3.2 Starting state (top lane, 26.19)
- 500 gold [WIKI Gold, H].
- Inventory empty except trinket 3340.

---

## 4. Shop rules

### 4.1 Where you can buy
You can purchase and sell while inside your team's shop area, or while dead [WIKI Item/Shop H; dead purchase: client voice-over asset `Play_vo_Tutorial2_Tip_Shop_While_Dead` exists in map11 data, M].

Geometry [CLIENT values, field names unresolved, so INFERRED M]:

| Team | Shopkeeper object (x, z) | Shop-area centre locator (x, z) | Shop-area "limits" locator (x, z) |
|---|---|---|---|
| Order / blue (100) | (40, 1112) `sru_storekeepersouth` | `OrderShopAreaCenter` (412.9, 416.2) | `OrderShopAreaLimits` (1174.6, 1022.8), 973.7 from centre |
| Chaos / red (200) | (13734.3, 14552.6) `sru_storekeepernorth` | `ChaosShopAreaCenter` (14297.2, 14388.3) | `ChaosShopAreaLimits` (13634.5, 13681.8), 968.6 from centre |

Each shopkeeper carries a `ShopGeComponentDef` with `{d1318f26}=1000.0`, an offset `{0f908963}` and `{651de225}=400.0`. The offset rotated by the object transform lands about 60 units from the area-centre locator.
- **Implement:** `can_shop = dead or dist2(pos, shop_area_center[team]) <= 1000^2`.
- The 400 value is probably a click or interaction radius. Do not use it.
- Unresolved: is the area a circle or the box spanned by centre and limits? See §15, U-1.

### 4.2 Costs, combining, selling, undo

```python
def total_cost(item):                       # CLIENT: price + sum(total(component))
    return item.price + sum(total_cost(c) for c in item.recipe)

def buy(inv, gold, item):
    assert in_shop_or_dead and item.in_store and purchase_restrictions_ok(item)
    owned = consume_components(inv, item)   # depth-first, use owned components first (each owned copy once)
    cost = total_cost(item) - sum(total_cost(c) for c in owned)
    assert gold >= cost and group_limits_ok(inv - owned, item) and slot_available(inv - owned, item)
    ...
def sell_value(item):                       # CLIENT sellBackModifier; DDragon rounding = round half up
    return floor(total_cost(item) * item.sellBackModifier + 0.5)   # default modifier 0.70
```
- Sell modifier **0.40** [CLIENT H]: all Doran's, Dark Seal, Cull, Health and Refillable Potion, Control Ward, Elixirs, Rejuvenation Bead, Seeker's and Shattered Armguard, Guardian Angel, and the World Atlas/support line.
- **1.0** (refund in full): Stealth Ward, Scarecrow Effigy, Elixir of Skill, Your Cut, all at 0 g.
- **0** and cannot be sold: jungle pets.

Examples: Trinity 3333 → 2333; Pickaxe 875 → 613; Brutalizer 1337 → 936.

Selling a component-built item refunds based on the **total**, not the combine price [CLIENT/DDragon H].

- **Undo** [WIKI Shop 3982704, M]: actions are undone LIFO at full refund (including gold earned from gold-income items since the purchase). Undo is allowed only while all of these hold:
  - the player has not left shop range;
  - the player has not entered combat (§4.3);
  - no item has been distributed to them;
  - no item has transformed;
  - no active of a handled item has been cast and no consumable has been consumed;
  - they have not started channelling Unleashed Teleport (top quest).
  T3 boots can be undo-refunded (26.15 bug fix) [NOTES].
- **Combat** (for undo and out-of-combat effects): a unit is in combat on dealing or taking damage or CC to/from an enemy champion, minion, monster or turret (not wards). Most out-of-combat timers are **5 s** [WIKI Combat status 4058480]. The simulator's RL agent will normally buy only in fountain, so an "undo" action can be omitted from the action space. Record it as a known simplification.

### 4.3 Item queue and auto-buy
Item queueing and component auto-purchase are UI conveniences. The sim should expose `buy(item_id)`, which already buys missing components implicitly, plus `sell(slot)`. The engine-level purchase order for "Purchase Components" (highest-cost component first, then left to right) only matters for partial buys [WIKI].

---

## 5. Uniqueness and item groups

The rule [CLIENT ItemGroup `mMaxGroupOwnable`, H]: an inventory may not hold more than `max` items whose `mItemGroups` contains group G. A purchase that would exceed this is rejected, **unless** the new item is produced by combining a recipe that consumes the conflicting member, or by a transformation (§3) [WIKI Item, H]. Every legendary also has a self-named group with max 1, so there are no duplicate legendaries [CLIENT H].

The groups that matter on SR (the members are only SR store and transform items) [CLIENT H]:

| Group (client ID / wiki label) | max | Members | Extra |
|---|---|---|---|
| `{c6428663}` "Hydra" (wiki), "Cleave" (14.1 label) | 1 | 3077 Tiamat, 3074 Ravenous, 3748 Titanic, 6698 Profane, 6631 Stridebreaker | `{8c259571}` max 3 also covers Ravenous, Titanic and Profane. It is redundant on SR |
| `{57352a0f}` Spellblade | 1 | 3057 Sheen, 3078 Trinity, 6662 Iceborn, 3100 Lich Bane, 3508 Essence Reaver, 2510 Dusk and Dawn, 3877 Bloodsong | group carries shared `Shared/Spells/SheenDelay` |
| `LifelineItems` Lifeline | 1 | 3053 Sterak's, 3155 Hexdrinker, 3156 Maw, 6673 Shieldbow, 2525 Protoplasm, 3003 Archangel's, 3040 Seraph's | shared `Shared/Spells/LifeLineCooldown` |
| `LastWhisper` Fatality | 1 | 3035 Last Whisper, 3033 Mortal Reminder, 3036 LDR, 3071 Black Cleaver, 6694 Serylda's, 3302 Terminus | |
| `VoidPen` Blight | 1 | 4630 Blighting Jewel, 3135 Void Staff, 3137 Cryptbloom, 8010 Bloodletter's, 3302 Terminus | |
| `Boots` | 1 | 1001, all T2 (3006, 3008, 3009, 3020, 3047, 3111, 3158), all T3 (3168, 3170–3175) | `BootsWithoutActives` max 1 (T2 plus Immortal Path) is also redundant |
| `DoransItems` Starter | 1 | 1054, 1055, 1056, 1086, 1120, jungle pets 1101–1103, World Atlas line 3865–3877 | `GoldItems` max 1 (jungle and support) |
| `Potion` | 1 | 2003, 2031 | |
| `WardPink` | 1 (and maxStack 2) | 2055 Control Ward | carry up to 2; 1 placed per player [WIKI] |
| `TearItems` Manaflow (semi-unique) | 1 | 3070 Tear, 3003, 3004, 3119, 2526 | the transformed items (3040, 3042, 3121, 2530) are **not** in TearItems, so you can start a new Tear after a transform (26.1) |
| `Quicksilver` | 1 | 3140, 3139 | |
| `{548f93b0}` Annul | 1 | 4632 Verdant Barrier, 3102 Banshee's, 3814 Edge of Night | |
| `ImmolateItems` | 1 | 6660 Bami's, 3068 Sunfire, 6664 Hollow Radiance | |
| `Glory` | 1 | 1082 Dark Seal, 3041 Mejai's | stacks carry over Dark Seal → Mejai's [CLIENT tooltip] |
| `EternityItems` | 1 | 3803 Catalyst, 6657 Rod of Ages | |
| `{d52cd27b}` Thorns | 1 | 3076 Bramble, 3075 Thornmail | |
| `StopwatchGroup` Stasis | 1 | 2420, 2421 | Zhonya's builds from these, so the group is bypassed by the recipe |
| `{db01f901}` Momentum | 1 | 3742 Dead Man's Plate | |
| `{20b00c0e}` Dirk | 1 | 3134 Serrated Dirk | |
| `{c8a69ca7}` (no max) | — | the Grievous Wounds item family | tag only. The GW *effect* is a shared named effect; the debuff does not stack (§6.4) |
| `Trinket` | — | 3340, 3363, 3364, 3330 | slot 6 only |

**Shared-cooldown rule** [WIKI Item, H]: if several equipped items share a named effect, they share its cooldown. On SR the group maxes above make this mostly moot. Spellblade and Lifeline still carry explicit shared cooldown spells (`SheenDelay`, `LifeLineCooldown`) [CLIENT]: model each as **one cooldown per champion**, not per item.

---

## 6. Shared named effects (exact rules)

### 6.1 Spellblade
Items: Sheen 3057, Trinity Force 3078, Iceborn Gauntlet 6662, Lich Bane 3100, Essence Reaver 3508, Dusk and Dawn 2510, Bloodsong 3877. Sources: [CLIENT values; WIKI Template:Spellblade info 4019664 for behaviour].

```
state: sb_armed_until (float, -inf), sb_cd_until (float, -inf)          # one per champion
on_ability_cast_start(champ):                         # at the START of the cast time; cancelled casts still arm (WIKI bug note)
    if now >= sb_cd_until: sb_armed_until = now + 10.0   # SpellBladeDuration=10 (Lich Bane dv); recast refreshes
on_basic_attack_hit(champ, target):                   # on-hit, not on-attack; dodge/blind/miss -> not consumed
    if now < sb_armed_until and target is not plant:
        dmg = spellblade_damage(item, champ)          # bonus damage, proc type, NOT crit-scaled
        apply_on_hit_damage(target, dmg, dtype)       # life steal applies (WIKI); triggers on structures (WIKI)
        sb_armed_until = -inf; sb_cd_until = now + 1.5    # SpellbladeCooldown=1.5 (CLIENT); cd starts after consume
    # attacks "blocked" (e.g. a block effect) still consume and start cd (WIKI)
```

| Item | Bonus damage | Type | Extra on proc [CLIENT] |
|---|---|---|---|
| Sheen | 1.00 × base AD | physical | — |
| Trinity Force | 2.00 × base AD | physical | Quicken (separate passive): any basic-attack hit gives +20 flat MS for 2 s |
| Iceborn Gauntlet | 1.50 × base AD | physical | frost field at the target for 2 s, radius 300 (`AoERadius`): slow 25% (melee) / 12.5% (ranged) to enemies moving through it [WIKI semantics]; `MonsterMod 1.5` |
| Lich Bane | 0.75 × base AD + 0.45 × AP | magic | the empowered attack gains +50% AS (`SheenASBuff`) |
| Essence Reaver | 1.25 × base AD + 50 × crit chance (crit as a fraction, so 25% crit → +12.5) | physical | restores mana = 50% of the proc damage |
| Dusk and Dawn | 0.75 × base AD + 0.10 × AP | magic | heals 0.10 × AP + 0.03 × bonus HP, and applies on-hit effects one additional time |
| Bloodsong (support) | 1.00 × base AD | physical | Maim: target champion takes +8% (melee holder) / +5% (ranged) damage for 4 s |

Notes:
- **Not blocked by spell shield** when applied by a basic attack.
- Is proc damage, so does not trigger spell effects.
- Cannot proc twice from one attack.
- The 1.5 s cooldown is **not** reduced by haste (static), INFERRED M.

### 6.2 Cleave (Tiamat line)
See §8 for the full spec.

### 6.3 Lifeline
Items: Sterak's 3053, Hexdrinker 3155, Maw of Malmortius 3156, Immortal Shieldbow 6673, Protoplasm Harness 2525, Archangel's 3003, Seraph's Embrace 3040. Values [CLIENT]; trigger semantics [WIKI].

```
on_pre_damage_taken(holder, dmg_post_mitigation, dtype):     # before HP is reduced; damage after resistances & reductions,
    if lifeline_cd_ready and (dtype == MAGIC or item not in {Hexdrinker, Maw}):     # before shields absorb it (INFERRED M)
        if holder.hp - dmg_after_existing_shields < 0.30 * holder.max_hp:
            grant(item.lifeline_effect)                       # applied FIRST, then this damage is absorbed by it
            lifeline_cd_until = now + 90.0                    # all listed items: Cooldown 90 (CLIENT)
```

| Item | Effect on trigger [CLIENT] |
|---|---|
| Sterak's Gage | shield = 0.60 × bonus HP, **decaying** over 4.5 s (`TimeBeforeDecay` 0.75; INFERRED: hold 0.75 s, then linear decay to 0 at 4.5 s) |
| Hexdrinker | magic-only shield `lerp_level(110 → 280)` (ranged × 0.75) for 2.5 s; magic damage trigger only |
| Maw of Malmortius | magic-only shield 200 + 1.5 × bonus AD (ranged × 0.75) for 3 s, plus 10% omnivamp until end of combat; magic trigger only |
| Immortal Shieldbow | shield `level_bp(400, +30 at L>=9)` (ranged × 0.8) for 3 s |
| Protoplasm Harness | +`lerp_level(100 → 300)` max HP for 5 s, plus heal `lerp_level(100 → 400) + 1.75·bonus armor + 1.75·bonus MR` over 5 s; +15% size, +10% MS, +25% tenacity while healing |
| Seraph's Embrace | shield = 0.18 × current mana (wiki/tooltip; client calc `0.18 × Mana`) for 3 s |
| Archangel's | is in the Lifeline group (transform target Seraph's has the effect); no lifeline effect before the transform [CLIENT tooltip] |

### 6.4 Grievous Wounds, Thorns
- **Grievous Wounds** [WIKI 3969410 + CLIENT `GrievousAmount=0.4`, `GrievousDuration=3`]:
  - Reduces all healing **and HP regeneration** received by **40%**.
  - It is a single non-stacking debuff: re-application refreshes the duration to max(remaining, 3 s).
  - Since 13.7 it applies even when the damage is fully absorbed by a shield.
  - Physical-damage sources: Executioner's Calling 3123, Mortal Reminder 3033, Chempunk Chainsword 6609. These apply on **physical damage to enemy champions** (any physical damage, including abilities).
  - Magic-damage sources: Oblivion Orb 3916, Morellonomicon 3165.
  - Thorns sources: Bramble 3076, Thornmail 3075.
- **Thorns** (Bramble/Thornmail) [CLIENT]:
  - Trigger: being **struck by a basic attack** (on-hit, so dodged attacks don't trigger).
  - Deals reactive magic damage to the attacker: Bramble 10; Thornmail 20 + 0.10 × bonus armor.
  - If the attacker is a champion, applies 40% GW for 3 s.
  - Reactive damage gets no vamp [WIKI Vamp].

### 6.5 Immolate (Bami's 6660, Sunfire 3068, Hollow Radiance 6664)
Values [CLIENT]:
- **Trigger:** taking or dealing damage, which (re)starts a 3 s aura (`AuraDuration=3`).
- **Tick:** once per second (`TicksPerSecond=1`) to all enemies within **325** (`Range`) of the holder.
- **Damage per tick (magic):**
  - Bami's: 15.
  - Sunfire: 20 + 0.015 × bonus HP.
  - Hollow Radiance: 15 + 0.01 × bonus HP.
- **Multipliers** (`MinionMod` and `MonsterMod` are *increases*):
  - Sunfire: ×1.5 vs minions, ×1.8 vs monsters (matches 26.16 "150%/180%").
  - Bami's: ×1.5 vs minions, ×2.0 vs monsters.
  - Hollow Radiance: ×1.25 vs minions, ×1.25 vs monsters.
- **Hollow Radiance Desolate:** killing an enemy deals 2 × DamagePerTick in 350 around it (champions: 4 × in 500).
- **Inactive out of combat** [WIKI Combat status].
- **Aura ownership:** this is a damaging aura, so it stacks across *different* holders [WIKI Item].
- **Open question (M):** whether the first tick is immediate on trigger or 1 s later. See U-4.

### 6.6 Helping Hand
Items: Doran's Shield, Ring and Helm, and Tear (`BonusDamageToMinions` / `BonusMinionDamage` = 5) [CLIENT].
- +5 **physical** on-hit bonus damage against **minions** only.
- Not life-stolen [WIKI].
- Shared named effect, so it counts once even with Tear + Doran's [WIKI Named item effect].

### 6.7 Plating and Rock Solid
- **Plated Steelcaps 3047 / Armored Advance 3174 (Plating):** incoming basic-attack damage × 0.90 [CLIENT `mEffectAmount[0]=0.1` / `DamageReduction=0.1`]. Turret attacks are excluded [WIKI]. Applied as a received-damage modifier on basic-attack-flagged damage, including on-hit? Wiki: "incoming basic attack damage"; INFERRED that the on-hit components are not reduced, M. See U-6.
- **Warden's Mail 3082 (Rock Solid):** reduce incoming damage from **champion** basic attacks by 15 flat (`BlockBase`), capped at 20% of that attack's damage (`WardenDamageMax`) [CLIENT]. Randuin's Omen builds from it but does not inherit Rock Solid; its client data has no BlockBase [CLIENT].
- **Randuin's Resilience:** critical-strike damage taken × 0.70 [CLIENT `PercentCritDamageReduction=0.3`].

### 6.8 Energized
Items: Rapid Firecannon, Statikk Shiv, Stormrazor, Voltaic Cyclosword. These are not top-lane core; the deep spec is deferred. Stacks build from moving and attacking (100 = ready). Item values are in the catalog.

### 6.9 Manaflow (Tear line)
Values [CLIENT]:
- Charges: one per `ManaChargeAmmoCD = 8` s, max 4 (Archangel's 5).
- Landing an ability (Manamune and Winter's: an ability or an attack) on an enemy consumes a charge and grants `ManaPerCharge` max mana (3; Archangel's 5; Circlet 4), doubled vs champions.
- `InternalCDPerCastID` 6.5 s per cast instance.
- Max 360 bonus mana, then transform.
- Garen is manaless, so this is not relevant to him. It matters to opponents.

### 6.10 Glory (Dark Seal / Mejai's)
Dark Seal [CLIENT]: kill +2, assist +1, max 10, lose 5 on death, +4 AP/stack.

Mejai's [CLIENT]: kill +4, assist +2, max 25, lose 10 on death, +5 AP/stack, +10% MS at ≥10 stacks.

Stacks are preserved Dark Seal ↔ Mejai's.

---

## 7. Life steal, omnivamp and other vamp-like effects

| Stat | Heals from | Rules | Tag |
|---|---|---|---|
| **Life steal** | post-mitigation damage of **basic attacks**, including on-hit item damage tagged *OnHitAppliesLifeSteal* | not vs structures; not affected by heal power; reduced by GW; Spirit Visage ×1.25; additive across sources | WIKI 4060425 H |
| **Omnivamp** | post-mitigation physical, magic and true damage from any source | 100% value, but **×0.33 vs minions and monsters when the damage is AoE, DoT/persistent or pet** (26.1); **never from Smite or Ignite** (26.4); not from reactive damage (Thorns); additive; GW reduces; Spirit Visage ×1.25 | NOTES 26.1/26.4 + WIKI Vamp 4058226 H |
| Physical vamp, spell vamp | — | no sources in 26.19 | WIKI H |
| Drain (Elixir of Wrath 12% physical vs champions, ×0.33 AoE; Knight's Vow) | post-mitigation | is a *heal*, so it benefits from heal power | WIKI H |

**Items whose on-hit damage applies life steal** [WIKI tag OnHitAppliesLifeSteal]: Ardent Censer, BotRK, Bloodsong, Dusk and Dawn, Guinsoo's, Heartsteel, Hullbreaker, Iceborn, Kraken Slayer, Lich Bane, Nashor's, Ravenous Hydra (cleave and active), Recurve Bow, Sheen, Terminus, Titanic Hydra (the on-hit part only, not the cone), Trinity Force, Wit's End.

**Not life-stolen:** starter on-hits (Helping Hand), Dead Man's Plate discharge, the Titanic cone, and Tiamat cleave. Tiamat's cleave is unlisted, so assume no (M); Ravenous explicitly says yes.

**Life steal vs Ravenous active:** `VampAmp = 1.0`, so the holder's life steal applies at 100% effectiveness to Ravenous Crescent damage [CLIENT + tooltip "Your Life Steal applies to this damage"].

Hook ordering for one damage event (INFERRED H, structural): compute post-mitigation damage → apply shields → reduce HP → **then** compute LS/omnivamp heal from the post-mitigation damage, **including the shielded portion**. Wiki: healing is based on damage dealt, and shields do not reduce it (M).
- Excess life-steal healing feeds Bloodthirster's Ichorshield: overflow above max HP becomes a shield up to `level_bp(165, +15 at L>=9)` [CLIENT].

---

## 8. Hydra line and Stridebreaker (FULL, in scope)

### 8.1 Shared Cleave passive
Applies to Tiamat 3077, Ravenous Hydra 3074, Stridebreaker 6631 and Profane Hydra 6698. Titanic is different; see §8.5.

| Field | Value | Tag |
|---|---|---|
| Trigger | each basic attack that **hits** (on-hit). Not when attacking structures. Tiamat, Ravenous and Stridebreaker trigger even if the attack deals 0 (invulnerable target); **Profane does not** (wiki-noted bug) | WIKI H |
| Targets | **other** enemy units (champions, minions, monsters) within **350** (`CleaveRadius`/`PassiveRadius`) of the **primary target's position**, excluding the primary | CLIENT 350 H; centre WIKI H |
| Distance measure | centre-to-centre ≤ 350 vs edge-to-edge is not encoded; implement centre ≤ 350 + target gameplay radius | INFERRED M (U-2) |
| Damage | physical **0.40 × total AD** (melee holder) / 0.20 × total AD (ranged holder), flat with no distance falloff (removed V12.22) | CLIENT H |
| Cap | at most **10** splash targets per attack (`MaxProcPerAuto=10`, V25.14); pick the 10 closest to the primary (INFERRED M) | CLIENT H |
| Damage class | proc damage, tagged AoE, applies no spell effects; not blocked by spell shield | WIKI H |
| Vamp | Ravenous: life steal applies to cleave. Others: no life steal. Omnivamp applies at 33% on minions (AoE) | CLIENT tooltip / NOTES |
| Ordering | after the primary attack damage resolves (same tick); cleave hits are simultaneous | INFERRED M |

### 8.2 Tiamat (3077)
- 1200 g = 2× Long Sword + 500. Sells for 840. +25 AD (26.16 buff from 20).
- **Crescent** (active), all [CLIENT/WIKI H] except where marked:
  - Damage: physical **0.75 × total AD** (`ActiveADRatio`) to all enemies in a **circle of radius 450** whose **centre is 100 units in front of the caster** along the facing (`castConeDistance=100`; wiki "center offset 100").
  - Hits champions, minions and monsters. Not structures (INFERRED; the data's `mAffectsTypeFlags=6154` is the same as the other hydras).
  - Cooldown 10 s, starting **at the end of the cast** [WIKI].
  - Cast time = the caster's attack **windup** time, capped at the base 0.2 s (`mCastTime 0.2`, `mUseAutoattackCastTimeData`; V14.2 note "shifts to windup if lower").
  - `mCantCancelWhileWindingUp`. **Not an attack reset**: no `Trait_AttackReset` tag.
  - Area damage, so it triggers spell effects, is blocked by spell shield, and breaks stealth.
  - Damage is applied at the end of the cast (INFERRED H).
  - Auto-targeted: no target input, uses facing.

### 8.3 Ravenous Hydra (3074)
- 3300 g = Tiamat + Vampiric Scepter + Caulfield's + 150. +65 AD, +12% life steal, +15 AH.
- Cleave as §8.1, with life steal applying.
- **Ravenous Crescent:** 0.80 × total AD physical, radius 450 centred 100 ahead, cd 10 s (end of cast), cast time = windup (max 0.2 s). Life steal applies at 100% (`VampAmp 1.0`). Not an attack reset.

### 8.4 Profane Hydra (6698)
- 2850 g = Tiamat + Brutalizer + 313. +55 AD, +18 lethality, +10 AH.
- Cleave 0.40 × AD within 350 (ranged ×0.5). Does not trigger on 0-damage attacks.
- **Heretical Cleave:**
  - 0.80 × total AD physical (`SlashDamageBase`). The `SlashDamageMax` value is also 0.8, because the low-HP bonus was removed in V14.19; ignore `HealthThreshold 0.5`.
  - Radius 450 (`ActiveRadius`), centred 100 ahead.
  - Cooldown 10 s, starting **at the start of cast**. Cast time = windup (max 0.2 s). Not an attack reset.
  - Active damage does not life-steal (V14.5 fix).

### 8.5 Titanic Hydra (3748)
- 3300 g = Tiamat + Tunneler + Giant's Belt + 50. +600 HP, +40 AD.
- **Cleave:** on basic-attack hit:
  - (a) Bonus physical on-hit to the primary = **1% max HP** (ranged 0.5%). This **applies life steal**, and **does apply vs structures**.
  - (b) Physical damage to other enemies in a **cone behind the primary target**, in the attack direction = **3% max HP** (ranged 1.5%). No life steal, not vs structures, max 10 targets.
- **Titanic Crescent (active):** the next basic attack within 10 s (`Cooldown` 10 / wiki) is empowered.
  - Primary on-hit = **4% max HP**, cone = **9% max HP** (ranged half). These replace the 1%/3% values; they are not additive [WIKI wording "empowers Cleave to deal…", M (U-3)].
  - The active **resets the attack timer** (`Trait_AttackReset`, 0 s cast).
  - The 10 s cooldown **starts after the empowered attack is used**, or when the 10 s window expires (INFERRED M).
- **Cone geometry** is not in the item data. Candidates in client data: `Items/Spells/ItemTitanicHydraCleave` has `castRange 300`, `castRadius 210`; `Shared/Spells/ItemTitanicHydraCleaveProjectile` has missile width 150 and `castRadius 299.3`.
  - Recommended default: an isosceles cone from the primary target's position, axis = attacker→target direction, length 300, half-width at the far end 210 (≈ a 70° full angle). INFERRED L. Needs measurement, U-3.
  - Wiki: "strikes all targets in the cone simultaneously, and is not a projectile".

### 8.6 Stridebreaker (6631)
- 3300 g = Tiamat + Phage + Dagger + 750. +40 AD, +25% AS, +450 HP. Cleave as §8.1 (350 radius, 40%/20% AD).
- **Breaking Shockwave:**
  - Damage: physical **0.80 × total AD** (`ADRatio`).
  - Slow: **35%** (`MSSlow −0.35`) for **3 s** (`Duration`) to all enemies in a **radius 450** circle (`CleaveRadius 450`) centred **100 ahead**.
  - Self MS: gain **+35% bonus MS per enemy champion hit** (`ActiveMS 0.35`), **decaying over 3 s** [WIKI 14.4 "decays over the intended 3 seconds"]. Data also has `MoveSpeedDuration 2`, `DecayRate 0.8`; U-5.
  - Cooldown **15 s, starting at the start of cast**.
  - Cast time = windup (base `mCastTime 0.25`). **The caster can move during the cast** (`mCanMoveWhileChanneling`, tooltip). Not an attack reset.
  - **No dash.** The "Halting Slash" dash and Temper passive were removed in 14.1 and 14.19 [WIKI H]. The brief's "dash geometry" does not exist at 26.19.
  - Slow, INFERRED M: a flat 35% for 3 s, *not* decaying; the tooltip has no decay wording, and the V25.S1.3 bug note mentions a "duration" only.

### 8.7 Pseudocode (JAX-friendly)

```python
def hydra_active_hits(caster_pos, facing_unit, target_pos, target_radius, R=450.0, OFF=100.0):
    centre = caster_pos + OFF * facing_unit
    return norm(target_pos - centre) <= R + target_radius        # edge inclusion INFERRED (U-2)

def crescent(item, ad_total, ...):          # 3077: 0.75, 3074: 0.80, 6698: 0.80, 6631: 0.80
    dmg_raw = RATIO[item] * ad_total        # physical, pre-mitigation, per target
    # 6631: also slow 0.35 for 3 s on each hit; self MS += 0.35 * n_champions_hit, decaying linearly to 0 over 3 s
    # cooldown: 3077/3074 start at cast end; 6698/6631 start at cast start; all 10 s except 6631 15 s

def cleave_targets(primary_pos, others_pos, radius=350.0, cap=10):
    d = norm(others_pos - primary_pos); mask = (d <= radius) & alive & enemy & ~is_structure & ~is_primary
    return top_k_by_smallest(d, mask, cap)

cast_time = min(base_cast_time[item], attacker_windup_seconds)   # base 0.2; Stridebreaker 0.25
```

---

## 9. Top-lane priority items: implementable specs

These are the items a melee bruiser or tank (Garen, Darius, Jax, Sett, Ornn and similar) and their lane opponents actually buy. Values are [CLIENT] unless tagged otherwise. Every other item's numbers are in ITEMS_CATALOG.md.

### 9.1 Starters and consumables
- **Doran's Shield 1054**: 450 g. +110 HP, +0.8 HP/s flat regen. Enduring Focus and Helping Hand as in catalog: after damage from a champion, 8 s of extra regen = (40/8) × min(missing/0.75, 1) HP/s (melee; ranged holder 30/8). AoE/DoT trigger ×0.66.
- **Doran's Blade 1055**: 450 g. +10 AD, +80 HP, 2.5% omnivamp. No passive.
- **Doran's Helm 1120**: 450 g. +150 HP, +8 armor, +8 MR, Helping Hand +5 vs minions.
- **Cull 1083**: 450 g. +7 AD, 3 HP on-hit. +1 g per minion kill, up to 100 kills, then +350 g once.
- **Health Potion 2003**: 50 g. Stack 5. 120 HP over 15 s (4 HP per 0.5 s).
- **Refillable 2031**: 150 g. 2 charges × 100 HP over 12 s; refills in the shop area.
- **Elixir of Iron 2138 / Wrath 2140**: 500 g each, level ≥ 9, 180 s.
  - Iron: +300 bonus HP, +25% tenacity, +15% size (Effect values), plus an ally MS path (+15%).
  - Wrath: +30 bonus AD, 12% physical drain vs champions (×0.33 AoE).
  - Can be used while dead [WIKI]. A new elixir replaces the old one. 5 s purchase cooldown in the Elixir group.

### 9.2 Boots

| Item | Cost | Stats | Effect |
|---|---|---|---|
| Boots 1001 | 300 | 25 MS | — |
| Berserker's Greaves 3006 | 1100 | 45 MS, 30% AS | — |
| Plated Steelcaps 3047 | 1200 | 45 MS, 25 armor | basic-attack damage taken ×0.9 |
| Mercury's Treads 3111 | 1250 | 45 MS, 20 MR, 30% tenacity | — |
| Boots of Swiftness 3009 | 1000 | 55 MS, 25% slow resist | — |
| Ionian Boots 3158 | 900 | 45 MS, 10 AH | +10 summoner haste |
| Sorcerer's Shoes 3020 | 1100 | 45 MS, 12 flat magic pen | — |
| Gluttonous Greaves 3008 | 1000 | 45 MS, 4% omnivamp | +0.6% omnivamp per champion takedown, max 10 stacks (permanent) |
| T3 (mid quest only; upgrade free) | same as T2 | Gunmetal 45 MS / 45% AS / 5% LS; Armored Advance 45 MS / 35 armor / Plating 10%, plus a physical shield `level_bp(90, +10 at L>=9) + 8% bonus HP` for 5 s on physical damage from a champion, 15 s cd; Chainlaced 45 MS / 25 MR / 30% tenacity, plus the same shield shape vs magic; Swiftmarch 65 MS / 25% slow resist / 5% MS → adaptive; Crimson Lucidity 45 MS / 20 AH / 20 summoner haste / 10% MS (ranged ×0.8) 4 s; Spellslinger 45 MS / 20 + 8% magic pen; Immortal Path = Gluttonous, plus +4% damage above 50% HP and +12% heal/shield/regen below 50% HP | |

Garen, Darius and Jax in top lane do not do the mid quest, so **T3 boots are unreachable for a top-laner** [WIKI Role Quests 4064833, H]. Swiftmarch slow resist is 25% in the client but 40% in the 26.1 notes; implement 25%.

### 9.3 Legendaries most relevant to top lane

| Item | Cost | Stats | Effect spec |
|---|---|---|---|
| Stridebreaker 6631 | 3300 | 40 AD, 25% AS, 450 HP | §8.6 |
| Titanic Hydra 3748 | 3300 | 40 AD, 600 HP | §8.5 |
| Ravenous / Profane | 3300 / 2850 | §8 | §8 |
| Trinity Force 3078 | 3333 | 36 AD, 30% AS, 333 HP, 15 AH | Spellblade 200% base AD; Quicken +20 MS for 2 s on attack hit |
| Sterak's Gage 3053 | 3200 | 400 HP, 20% tenacity | +50% base AD as bonus AD; Lifeline shield 60% bonus HP decaying over 4.5 s, 90 s cd (§6.3) |
| Black Cleaver 3071 | 3000 | 45 AD, 400 HP, 20 AH | Carve: 6% armor reduction per stack, up to 5 stacks (30%), 6 s, from any physical damage to champions. Non-basic-attack damage applies one stack per target per frame (0.01 s ICD). Fervor: physical damage → +20 MS (ranged 10) for 2 s |
| Overlord's Bloodmail 2501 | 3300 | 30 AD, 550 HP | Tyranny: +2.5% bonus HP as AD. Retribution: bonus AD = up to 12% of total AD from other sources, linear in missing HP from 0% to 70% missing (full below 30% HP) [WIKI wording "of your total attack damage from other sources"] |
| Sundered Sky 6610 | 3100 | 40 AD, 400 HP, 10 AH | Lightshield Strike: per target, 10 s cd. The first basic attack vs a champion crits at 0.8 × normal crit damage and heals 0.9 × base AD (ranged 0.45) + 4% missing HP. Overheal → bonus health for 8 s |
| Spear of Shojin 3161 | 3100 | 45 AD, 450 HP | +25 basic-ability haste; Focused Will +3% ability and passive damage per stack (ranged 1.5%), 4 stacks, 6 s, 1 stack per cast instance per 1 s |
| Experimental Hexplate 3073 | 3000 | 40 AD, 20% AS, 450 HP | +30 ultimate haste; on ult cast +50% AS / +20% MS for 8 s (ranged 35% / 14%), 30 s cd |
| Eclipse 6692 | 2900 | 60 AD, 15 AH | 2 separate hits on a champion within 2 s → 8% (ranged 5%) target max-HP physical, plus a 150 + 0.4 bAD shield (ranged half) for 2 s; 6 s cd |
| Death's Dance 6333 | 3300 | 60 AD, 50 armor, 15 AH | Ignore Pain: 30% (ranged 10%) of post-mitigation physical and magic damage taken is stored and taken as **true** damage over 3 s (1/3 per second). Defy: takedown within 3 s cleanses the stored damage and heals 0.75 × bonus AD over 2 s |
| Hullbreaker 3181 | 3000 | 40 AD, 500 HP, 4% MS | Skipper: stacks on attacks for 10 s; every 5th attack vs a champion or epic monster deals 1.2 × base AD + 5% max HP bonus physical (structures 3.0 × base AD + 10% max HP), ranged ×0.7. Boarding Party resists for nearby allied siege and super minions: `level_bp(70, +6 at L>=9)` (ranged ×0.5) |
| Dead Man's Plate 3742 | 2900 | 55 armor, 350 HP, 4% MS, 15% slow resist | Shipwrecker: while moving +7 stacks every 0.25 s (wiki) → 100 stacks in 3.75 s (client `DurationToMaxStack 4`); up to +20 flat MS at 100. Next basic attack consumes all stacks for (stacks/100) × (1.0 × base AD + 40) bonus physical on-hit (no life steal) |
| Heartsteel 3084 | 3000 | 900 HP, 100% base regen | Colossal Consumption (see catalog 3084): 70 + 6% max HP, 10% → permanent HP, 30 s per target); Goliath size |
| Jak'Sho 6665 | 3200 | 45 armor, 45 MR, 350 HP | after 5 s of champion combat, bonus armor and MR ×1.30 until end of combat |
| Thornmail 3075 | 2450 | 75 armor, 150 HP | Thorns 20 + 10% bonus armor magic, plus GW 40% 3 s on the attacker |
| Randuin's Omen 3143 | 2700 | 75 armor, 350 HP | crit damage taken ×0.7; active Humility (deferred) |
| Frozen Heart 3110 | 2500 | 75 armor, 400 mana, 20 AH | aura: enemy champions within 700 have −20% AS (cripple; does not trigger combat) |
| Sunfire Aegis 3068 | 2800 | 350 HP, 50 armor, 10 AH | Immolate 20 + 1.5% bonus HP per s, 325 range |
| Hollow Radiance 6664 | 2800 | 400 HP, 40 MR, 10 AH, 100% base regen | Immolate 15 + 1% bonus HP; Desolate |
| Unending Despair 2502 | 2800 | 400 HP, 50 armor, 15 AH | every 4 s in champion combat: 3% bonus HP magic to enemy champions within 650, heal 250% of post-mitigation damage |
| Kaenic Rookern 2504 | 2900 | 400 HP, 80 MR, 100% base regen | magic shield 15% max HP after 15 s with no magic damage taken |
| Force of Nature 4401 | 2800 | 400 HP, 55 MR, 4% MS | Steadfast: 8 stacks (7 s), from magic damage from champions (1 per source per s; immobilize = 2) → +70 MR, +6% MS |
| Spirit Visage 3065 | 2700 | 400 HP, 50 MR, 10 AH, 100% base regen | heals, shields, regen and vamp received ×1.25 |
| Warmog's Armor 3083 | 3100 | 1000 HP, 100% base regen | +12% item HP. If ≥ 2000 bonus HP and no damage for 8 s (3 s if non-champion damage only), heal 1.5% max HP every 0.5 s |
| Mortal Reminder 3033 / Chempunk 6609 / Executioner's 3123 | 3000 / 3000 / 800 | — | physical damage to champions → GW 40% 3 s |
| Endless Hunger 2517 | 3100 | 65 AD, 5% omnivamp, 20% tenacity | AH = 5 + 0.13 × bonus AD (ranged 0.10); takedown → +15% omnivamp for 8 s |
| Guardian Angel 3026 | 3200 | 55 AD, 45 armor | revive: 4 s stasis → 50% base HP, 100% max mana; 300 s cd |
| Maw of Malmortius 3156 | 3100 | 60 AD, 40 MR, 15 AH | Lifeline vs magic: 200 + 1.5 × bonus AD shield, 3 s, plus 10% omnivamp until end of combat |
| Phantom Dancer 3046 | 2650 | 65% AS, 25% crit, 10% MS | permanently ghosted (ignores unit collision) |
| Blade of the Ruined King 3153 | 3200 | 40 AD, 25% AS, 10% LS | on-hit 9% (ranged 6%) of target current HP physical (cap 100 vs minions/monsters, LS applies). 3 hits on a champion → 30% slow 1 s (ranged 15%), 15 s cd |

---

## 10. Events and hooks (ordering contract)

All item passives attach to one of the hooks below. The order *within a tick* follows the damage-pipeline agent's global contract. These are the item-relevant points [INFERRED from WIKI semantics, M–H]:

| # | Hook | Fires when | Items |
|---|---|---|---|
| H1 | `stat_calc` (static) | inventory changes, level-up, buff change | all item stats; group/uniqueness checks; Doran's Blade OV; T3 boots |
| H2 | `stat_calc_dynamic` | every tick, or on any input change | Sterak's (base AD), Overlord's (bonus HP, missing HP), Warmog's Vitality, Endless Hunger AH, Riftmaker/Archangel's/Winter's (HP, mana), Swiftmarch (MS), Jak'Sho (combat time), Glory AP, Gluttonous stacks, Immortal Path (HP%) |
| H3 | `on_ability_cast_start` | at the start of the cast time | arm Spellblade; Shojin lockout; Hexplate/Fiendhunter/Zeke's ult casts; Crimson Lucidity |
| H4 | `on_attack_windup_complete` (on-attack) | when the attack is launched (windup done) | Guinsoo's, Navori, Yun Tal, Runaan's, RFC/Voltaic Energized stacking [WIKI Attack effects 4050298] |
| H5 | `on_basic_attack_hit` (on-hit) | the attack lands; not on dodge, blind or miss | Spellblade proc, Cleave (Tiamat line), Titanic on-hit and cone, BotRK, Recurve, Wit's End, Nashor's, Terminus, Kraken, Hullbreaker, DMP discharge, Sundered Sky, Heartsteel proc, Trinity/Phage MS, Cull heal, Helping Hand, Thorns (target side: H9) |
| H6 | `on_damage_dealt(post-mitigation, type, flags{basic, on_hit, area, dot, pet, reactive, proc})` | after mitigation and after the target's shields | Black Cleaver Carve and Fervor, GW application (by damage type), Eclipse stacks, LS/omnivamp heals (§7), Ichorshield overflow, Elixir of Wrath drain, Immolate activation, Doran's Ring upgrade, Muramana Shock, Spear stacks |
| H7 | `on_pre_damage_taken(raw → reduced)` | before HP change, after attacker modifiers | Plated Steelcaps/Armored Advance (attacks ×0.9), Warden's Mail (−15, cap 20%), Randuin's (crits ×0.7), Death's Dance (store 30%), Celestial Opposition |
| H8 | `on_lifeline_check` | inside H7, when HP − remaining damage < 30% max | Lifeline group (§6.3); Guardian Angel on *lethal* only (after shields) |
| H9 | `on_damage_taken(post)` | after HP change | Doran's Shield regen trigger (champion source), Kaenic timer reset (magic), Force of Nature stacks, T3 boot shields, Immolate activation, Thorns (if basic attack), combat timers, Warmog's timers |
| H10 | `on_kill` / `on_takedown(window 3 s)` | unit death | Cull (minion kill by holder), Glory, Gluttonous, Endless Hunger, Death's Dance Defy, Hubris, Axiom Arc, Hollow Radiance Desolate, Collector gold |
| H11 | `periodic(dt)` | each tick | Immolate ticks (1 Hz), Unending Despair (4 s), HoT potions (0.5 s), Doran's Shield regen, Warmog's (0.5 s), Manaflow charge regen (8 s), Heartsteel proximity (0.5 s), Jak'Sho combat seconds, buff timers, cooldowns |
| H12 | `on_move(distance, dt)` | while moving | DMP Momentum (stacks per 0.25 s moving), Energized |
| H13 | `on_level_up` | level change | Lost Chapter (not top), all `level_bp`/`lerp_level` values, elixir purchase gate |
| H14 | `on_cc_applied(kind)` | applying slow or immobilize | Bandlepipes, Fimbulwinter, Imperial Mandate, Solstice Sleigh |
| H15 | `on_shop_area_enter` | entering the shop area | Refillable refill, support ward charges |
| H16 | `active(slot)` | item key pressed | Tiamat line and Stridebreaker (§8); others deferred |
| H17 | `on_purchase` / `on_sell` / `on_undo` | shop actions | gold, groups, Glory stack transfer, Cull counter reset, Seeker's → Shattered rule |

Within one basic-attack hit, the item-relevant order is (INFERRED M; confirm with the damage agent):
1. Base attack damage (crit roll).
2. On-hit bonus damages are summed per type: Spellblade, BotRK, Titanic on-hit, Recurve, Wit's, DMP, Kraken, Hullbreaker, Sundered Sky crit replaces the crit roll. Each is a separate damage instance with flags `on_hit`/`proc`. BotRK uses the target's current HP **before** this attack's damage (INFERRED M).
3. Mitigation; then H7/H8 on the target.
4. HP change; H9 on the target (Thorns reacts here).
5. H6 on the attacker: life steal from (1) + LS-tagged on-hits, Carve stack, GW.
6. Cleave splash instances (separate AoE damage events, which go through 3–5 for each target).
7. Spellblade consumption and cooldown start.

---

## 11. State the simulator must carry

Per champion:
- `inventory[7] (item_id, stack)`, plus the role-quest slot. Top: none.
- `gold`. Gold-at-purchase for undo (optional).
- Cooldowns: `item_cd[slot]` for actives (Tiamat line, Stridebreaker); `spellblade_cd_until`; `spellblade_armed_until`; `lifeline_cd_until`; `titanic_armed_until`.
- Hydra: `stride_ms_buff (start, n_champs)`.
- Stacks and counters:
  - Carve, per target: 5 stacks with an expiry each (or count + expiry), plus a per-target ICD.
  - Glory stacks.
  - Cull kill count.
  - Gluttonous takedown stacks.
  - DMP momentum (0–100).
  - Hullbreaker stacks and expiry.
  - Sundered Sky per-target cooldown.
  - Heartsteel per-target proximity stacks and cooldown.
  - Eclipse per-target hit window and cooldown.
  - Kraken attack counter.
  - BotRK per-target 3-hit counter and slow cooldown.
  - Manaflow charges.
  - Force of Nature stacks.
  - Jak'Sho combat seconds.
- Timers:
  - `last_damage_from_champion_time`, `last_magic_damage_time` (Kaenic), `last_damage_time` (Warmog's, both champion and non-champion).
  - In-combat flag (5 s).
  - Immolate aura expiry.
  - Doran's Shield regen expiry and its effectiveness (1.0 or 0.66).
  - HoT instances (potion: remaining ticks).
  - Elixir id and expiry.
  - Unending Despair next pulse.
  - Death's Dance bleed pool: queue of (amount, remaining time).
  - GW debuff expiry, *on targets*.
  - Lifeline-granted shields with decay parameters.
  - Overlord's/Sterak's derived stats (recomputed each tick).
- Shop: `in_shop_area` (derived from position or dead), purchase cooldown for elixirs, an undo stack (optional).

Static per patch (host-side, outside JIT):
- item table (stats vector, price, total, sell modifier, recipe, groups, flags, dataValues);
- group table (max ownable);
- the SR item-pool mask.

All of this is fixed-shape. Item effects become a bit-mask of supported effects × per-item parameter rows.

---

## 12. Season 2026 item system and every item change from 26.1 to 26.19

Every value below was re-checked against the 26.19 client. **✓** = the client matches the latest notes value. **≠** = see §13. [NOTES, H]

### 12.1 System changes
- **26.1 (season start)**:
  - Base crit damage 175% → **200%** ✓ [WIKI Critical strike]; IE +30% crit damage ✓.
  - Omnivamp is full value, ×0.33 vs minions for pet, DoT and AoE spells.
  - Tear items are semi-unique (one *stacking* Tear at a time) ✓ (TearItems excludes the transforms). Muramana and Fimbulwinter mana 860 → 1000 ✓.
  - Removed: Aegis of the Legion, Symbiotic Soles (moved into the mid quest), Vigilant Wardstone, and Feats of Strength.
  - Mid quest: tier-2 boots become their tier-3 versions free. Bot quest: boots move to the quest slot. Support: Control Ward 40 g and 2 stored after completion. Ambient gold starts at 65 s.
- 26.3: support Control Wards sit in the quest slot from game start. Stealth Ward cooldown 210–90 s.
- **26.4: Smite and Ignite no longer trigger omnivamp.**
- 26.6–26.7: the support farming penalty was softened, then disabled.
- 26.9: mid-quest reward Empowered Recall → +6% bonus AD/AP; 26.11 raised it to 8%. Tier-3 boots are kept.
- 26.12: top-quest TP shield 35% for 10 s.
- 26.14: item buff durations moved onto the inventory icon (UI only).
- 26.16: off-lane minion penalty for the support line −33% until level 5.
- 26.19: top-quest TP cooldowns 390 s and 300–210 s (quest agent).
- No starting-gold, sell-rate, potion or elixir changes anywhere from 26.1 to 26.19.

### 12.2 Per-item changes (latest value wins; client-checked)

| Patch | Item: change | 26.19 client |
|---|---|---|
| 26.1 | New: Dusk and Dawn, Fiendhunter Bolts, Endless Hunger, Bastionbreaker, Actualizer, Hexoptics C44, Bandlepipes, Protoplasm Harness, Whispering Circlet/Diadem of Songs. Returned: Hextech Gunblade, Stormrazor. T3 boots (6) | present ✓ |
| 26.1 | Doran's Blade: Life Draining removed, +2.5% omnivamp | ✓ |
| 26.1 | Unending Despair armor-only 50 armor, 15 AH, new recipe; Horizon Focus 2700 / 75 AP; Echoes of Helia new values; Essence Reaver back on Sheen (hotfix 1/9: 3050 g, 50 AD); Umbral Glaive 2800 / 60 / 15 / 18; Redemption 30 AP, no HP; Zeke's +15 ult haste; Mortal Reminder 3000 / 30%; LDR 3300 / 35% + Giant Slayer; Yun Tal; IE 3500 / 75 AD / +30% crit damage; Sundered Sky crit 80%, cd 10 s | ✓ (Redemption cost ≠) |
| 26.2 | Bandlepipes 2300; Fiendhunter 45% AS, crit ×0.8, +15% true; Hexoptics 55 AD | ✓ |
| 26.3 | Actualizer 2800; Dusk and Dawn 75% base AD; Endless Hunger 3100 / 65 AD; Protoplasm ratios 175%, AH 20; T3 Steelcaps/Mercs shields `100–200 (lvl 9+) + 8% bonus HP` | ≠ shield formula (§13) |
| 26.4 | Hexoptics max-range 500, takedown range buff 8 s | ✓ |
| 26.5 | Hubris 2800; Locket 290–360 | ✓ |
| 26.6 | Chempunk 3000; Sunfire recipe (Bami's + Chain Vest + Ruby + 600) | ✓ (then 26.16) |
| 26.9 | New Doran's Bow (6 AD / 12% AS / 1.5% OV), Doran's Helm, Gluttonous Greaves / Immortal Path. Removed Opportunity, Trailblazer. Reworked Statikk Shiv, Voltaic Cyclosword. Dusk and Dawn heal and nerfs; Endless Hunger AH 13% / 10%; Hubris 55 AD, +3/stack, +12; Staff of Flowing Water; Axiom Arc 10% + 0.25%/lethality | Doran's Bow AS ≠ (client 15%) |
| 26.10 | Doran's Bow 8 AD; Doran's Helm 140 HP; Gluttonous 1000 g, 0.6% × 10; Lich Bane 45% AP, 6% MS; Voltaic 3000 | ✓ |
| 26.11 | Dream Maker, Moonstone, Imperial Mandate rework, Echoes 30%, Locket 30/30, Knight's 14% / 12%, Zeke's ready window; Hexplate ranged 35% / 14%; Heartsteel 10%; Statikk 45 AD | ✓ |
| 26.13 | Doran's Helm 150 HP / 8 / 8; Imperial Mandate recipe, 60 AP, Control 20, Command 7% | ✓ |
| 26.14 | Immortal Path 4% / 12%; Protoplasm 2600, 100–300 / 100–400; Hextech Rocketbelt (Kindlegem recipe, cd 50, 60 AP, 350 HP) | ✓ |
| 26.15 | Bastionbreaker 3000, Shaped Charge 50 + 150% lethality; Terminus on-hit +10% bAD +10% AP; Yun Tal combine 750 (3000), 45% AS | ✓ |
| 26.16 | Berserker's 30% AS, Gunmetal 45%; Black Cleaver 45 AD, Fervor 20/10; Eclipse 8%/5%, shield 150 + 40% / 75 + 20%; Runaan's 5% MS, bolt 65%; **Sterak's 50% base AD**; Sundered Sky recipe and heal 90% / 45% + 4%; Sunfire 1.5% bonus HP, 150% / 180%, combine 700; Chainlaced 25 MR; Spellslinger 20 + 8%; **Tiamat 25 AD** | ✓ |
| 26.17 | Stormrazor 25% AS; Sundered Sky 400 HP / 40 AD | ✓ |
| 26.18 | Guinsoo's stack duration 4 s | ✓ |
| 26.19 | World Atlas / Runic Compass HP 0/60/200 → regen 50/75/75% | ✓ (Atlas: no HP, 50% regen) |

There are no Hydra/Tiamat-line rule changes in 26.x apart from Tiamat AD. The last mechanical Cleave change was V25.14, the 10-splash cap.

---

## 13. Disagreements: client vs notes and wiki (implement the client value)

| # | Item / rule | Client 16.19 | Notes / wiki | Default and why |
|---|---|---|---|---|
| D1 | Doran's Bow 1086 AS | **15%** | notes 26.9: 12% (no later change in the 26.10–26.19 notes); wiki 15% | **15%**: the client and wiki agree, so this was probably an unlisted change or a notes error |
| D2 | Redemption 3107 total | **2300** (Codex 850 + Idol 600 + 850) | notes 26.1 and wiki: 2250 | **2300**: the client recipe sums to it. Forbidden Idol 600 in client |
| D3 | T3 boot shields (3173, 3174) | `90 + 10·max(0, L−8)` + 8% bonus HP → 90 (L1–8), 100 (L9), 190 (L18), 210 (L20) | notes 26.3: "100–200 (scaling begins at level 9)" | **client formula**. The notes range may count to L19. Irrelevant for top laners, who cannot get T3 |
| D4 | Doran's Shield Enduring Focus for AoE/DoT triggers | **66%** effectiveness (`RangeRegenMult 0.66`, tooltip) | wiki: AoE/DoT/proc use the *ranged* values (30/40 = 75%) | **0.66**: the client tooltip is authoritative. Measure (U-7) |
| D5 | Swiftmarch slow resist | **25%** (`mPercentSlowResistMod 0.25`) | notes 26.1: 40% | **25%** |
| D6 | Hexoptics max-range | 500 | notes 26.1: 750, 26.2 old value 700, 26.4 → 500 | 500 ✓ (an internal notes inconsistency only) |
| D7 | Statikk Shiv total | **3000** (Aether Wisp total 900) | notes "≈2900" (assumed Aether 800) | 3000 |
| D8 | Tiamat Cleave life steal | not tagged | wiki tag list excludes Tiamat; Ravenous explicit | no life steal on Tiamat, Profane or Stridebreaker cleave (M) |
| D9 | Stridebreaker MS buff duration | `Duration 3`, `MoveSpeedDuration 2`, `DecayRate 0.8` | wiki 14.4: decays over 3 s | 3 s linear decay (M, U-5) |
| D10 | Sterak's Lifeline decay | `TimeBeforeDecay 0.75`, `ShieldDuration 4.5` | wiki: "decays over 4.5 s" | hold 0.75 s, then linear decay over the remaining 3.75 s (L–M, U-8) |
| D11 | Legacy tooltip strings | Tiamat `tooltipdynamic` mentions falloff ("Minimum of @SecondaryDamageMin@"); Heartsteel/Warmog's mention Mythic | — | ignore: these are stale string-table entries, not referenced by current `GeneratedTip` |
| D12 | Overlord's Retribution base | tooltip: "increased Attack Damage" | wiki: "% of your total AD from other sources" | wiki: multiply AD from all other sources (M) |
| D13 | Feats boots buff name | T3 boots require `Feats_NoxianBootPurchaseBuff` | the wiki "Item" page also says the Midlane Quest | the mid quest grants that buff (H); Feats itself was removed in 26.1 |

---

## 14. Diff vs current implementation

Files: `lanerl_jax/sim/modern_items.py`, `lanerl_jax/data/modern/26.19/items.json`, `lanerl_jax/sim/modern_stats.py`. All line numbers are in `modern_items.py` unless marked.

| Location | Current | Required (this spec) | Severity |
|---|---|---|---|
| `items.json` (whole) | DDragon `maps["11"]` (254 items: all 210 SR store items plus 44 non-SR mirrors such as 322065 and 663039; lacks the transforms 3040/3042/3121/2530) | Rebuild from the client CLASSIC item lists ∩ `mInStore` (210) plus the 4 transforms. Carry `price`, `total`, `sellBackModifier`, recipe, groups with max, flags, `mDataValues`, calcs | HIGH |
| `items.json`, `itemLimit` | absent for all 254 entries | client `mItemGroups` + ItemGroup `mMaxGroupOwnable` (§5) | HIGH |
| `:40–49` `_STAT_MAP` | maps only 13 DDragon keys. Ability haste and tenacity are parsed from description text (`:79–87`, `import re` inside the loader). No lethality, armor/magic pen, omnivamp, crit damage, base HP regen %, base mana regen %, slow resist (from data), HSP, MR fields beyond flat | map **all** client `m*Mod` fields (§2.1 table). Delete the regex parsing | HIGH |
| `:47–48` | `FlatHPRegenMod` and `FlatHPPoolRegenMod` → `health_regen` with no unit note | units are HP **per second** (Doran's Shield 0.8/s = 4 per 5 s) | LOW (document) |
| `:98` `_HYDRA_ITEM_IDS` | {3074, 3077, 3748, 6631, 6698} | ✓ matches client group `{c6428663}` | OK |
| `:104–115` `ACTIVE_ITEM_SPECS` | Tiamat `shape: "forward"`; Stridebreaker `shape: "radial"`, centred on the caster | both: circle R = 450 centred **100 ahead** of the caster. Add cast time = min(base, windup), cooldown start (end of cast for 3077/3074; start for 6698/6631), no AA reset. Add Ravenous (0.80, LS 100%), Profane (0.80), Titanic (empowered next attack, AA reset, 10 s) | HIGH |
| `:116–138` `validate_item_loadout` | ≤ 6 items; group only via `_HYDRA_ITEM_IDS` or the empty `itemLimit`; any behavioural item not in `ACTIVE_ITEM_SPECS` is "unsupported" | enforce all groups (§5) incl. Lifeline, Spellblade, Fatality, Boots, DoransItems (starters), Potion, Tear; allow stacks (potions 5, wards 2); trinket slot separate; role-quest slot (bot boots) | HIGH |
| `:118` | `len(ids) > 6` | the 6 main slots plus trinket handled separately; stackables count once per stack | MED |
| `:131–133` | Tiamat/Stridebreaker passives (Cleave) counted as "supported" only because their actives are listed | Cleave needs its own on-hit kernel (`tiamat_cleave` exists but has no 10-target cap or structure exclusion) | MED |
| `:223–233` `tiamat_crescent` | hit = within 450 of the caster **and** in the forward half-plane | `‖target − (caster + 100·facing)‖ ≤ 450 (+ target radius)`; no half-plane test | HIGH |
| `:236–245` `stridebreaker_active` | `distance ≤ 450` from the caster; MS bonus `0.35 × champions_hit` with no duration or decay; slow returned with no duration | same circle as above; slow 0.35 for 3 s; MS buff decays to 0 over 3 s; cooldown 15 s from cast start; movement allowed during the cast | HIGH |
| `:248–255` `tiamat_cleave` | 350 radius around the primary, 40%/20% AD, excludes the primary ✓ | + cap 10 nearest, exclude structures, skip when the attack target is a structure, Profane 0-damage rule, Ravenous life-steal flag, Titanic separate kernel (on-hit 1%/0.5% max HP + cone 3%/1.5%) | MED |
| `:150–166` `item_loadout_stats` | sums static stats only | add dependent-stat pass (§2.1): Sterak's, Overlord's, Warmog's Vitality and others | MED |
| `modern_stats.py:64–75` `adaptive_force_total` | AD = 0.6 × AF | ✓ matches Swiftmarch/elixir adaptive force usage | OK |
| `modern_stats.py` | no crit-damage constant | base crit multiplier **2.00** (26.1) + IE 0.30 (owner: stats/damage agent) | MED |
| (missing) | no shop/inventory model | §3–4: buy/sell/combine/undo, shop area radius 1000, starting gold 500, sell rates | HIGH for any modern economy |
| (missing) | no Spellblade, Lifeline, GW, Carve, Immolate, Plating, vamp hooks | §6–§10 | HIGH for the top-lane item set |

---

## 15. Unresolved / needs live measurement

| ID | Question | Default implemented | Suggested test (practice tool, patch 26.19) |
|---|---|---|---|
| U-1 | Shop area shape and size (circle radius 1000 around the `ShopAreaCenter` locator vs the box to `ShopAreaLimits`) | circle r = 1000 | walk out of the fountain along 8 directions; record the position where the gold button greys out (Replay API position or a screenshot grid) |
| U-2 | Hydra active / Cleave radius: centre-to-centre or edge-inclusive | edge-inclusive (R + target gameplay radius) | place a target dummy at increasing distance in 10-unit steps; find the hit boundary for Crescent and Cleave |
| U-3 | Titanic cone geometry (length, angle, origin), and whether empowered values replace or add | origin = primary target, length 300, end half-width 210; replace | dummies in a grid behind the target; activate and attack; read the damage numbers |
| U-4 | Immolate first tick (immediate vs after 1 s) and tick alignment | first tick 1 s after activation, then 1 Hz | Sunfire vs a dummy; timestamped combat log |
| U-5 | Stridebreaker MS decay curve and duration (2 vs 3 s); slow decay | linear 0.35·n → 0 over 3 s; flat slow 3 s | MS readout frames after hitting 1 and 2 champions |
| U-6 | Plated Steelcaps: are on-hit components of an attack reduced? | only the base attack instance | Steelcaps holder hit by a BotRK/Recurve attacker; compare the damage numbers |
| U-7 | Doran's Shield regen formula (cap at 75% missing; 0.66 vs 0.75 for AoE) | `(40/8)·min(miss/0.75, 1)`; AoE 0.66 | take a fixed champion hit at 30/50/80% missing HP; measure HP over 8 s |
| U-8 | Sterak's shield decay shape | 0.75 s hold, then linear | shield bar per frame after the trigger |
| U-9 | `lerp_level` / `level_bp` at champion levels 19–20 (top quest) | `lerp` extrapolates past 18 (flag absent on 61/62 item calc parts, explicit `false` only on mode item 772038; same convention as RUNES §1.1, README X-1); `level_bp` continues linearly | Hexdrinker or Shieldbow shield value read at L19/L20 in a custom game |
| U-10 | Crescent cast time vs windup for very fast attackers, and whether movement or attack commands buffer during the cast | `min(0.2, windup)`; cannot cancel | frame-step at varied attack speed |
| U-11 | Cleave 10-cap selection order | nearest 10 to the primary | 12 minions clustered; count the damage numbers |
| U-12 | Health Potion stacking (two potions drunk 1 s apart: two HoTs or a refresh?) | two independent HoTs | drink 2; measure total HP over 15 s |
| U-13 | Shop-constant hashes in the CLASSIC `GameModeConstants` (`{6cf687be}`): starting gold, sell rate, undo | wiki values (500, 0.7, rules) | resolve the hash names via the CDTB hash lists, or confirm empirically |
| U-14 | Overlord's Retribution base (total vs bonus AD) | total AD from other sources | compare the AD readout at 100% and 25% HP |
| U-15 | `RestrictedBuffName = "HeroPassive"` on Steel Sigil, Brutalizer, Tunneler, Glowing Mote, Statikk, Echoes, Dawncore | ignored (likely a mode or champion restriction) | check whether any champion is blocked from buying these on SR |

---

## 16. Deferred actives and items (listed only)

Item actives, deferred per MODERN-009. One line each, with values in the catalog:
- Seeker's Armguard 2420 / Zhonya's 3157: stasis 2.5 s (single use / 120 s).
- Quicksilver Sash 3140 / Mercurial 3139: cleanse (+50% MS 2 s), 90 s.
- Youmuu's 3142: +20% (ranged 15%) MS and ghost for 6 s (ranged 4 s), 45 s.
- Randuin's 3143: 70% slow 2 s, radius 500, 90 s.
- Hextech Gunblade 3146: 175–253 + 30% AP magic, 25% slow 1.5 s, 60 s.
- Hextech Rocketbelt 3152: dash plus 100 + 10% AP missiles, 50 s.
- Shurelya's 2065: +30% ally MS 4 s, 75 s.
- Locket 3190: team shield 290–360, 90 s.
- Redemption 3107: 150–350 heal plus 10% max HP true damage to enemies, 90 s.
- Mikael's 3222: ally cleanse and heal, 120 s.
- Knight's Vow 3109: bind an ally.
- Actualizer 2522: mana empower 8 s, 60 s.
- Control Ward 2055 and trinkets 3340/3363/3364: vision (deferred to the vision agent).
- Elixirs: consumed on use, specified in §9.1 (in scope as buffs).
- Potions: in scope.

Jungle items (deferred): Scorchclaw Pup 1101, Gustwalker Hatchling 1102, Mosstomper Seedling 1103. All 450 g, require Smite, cannot be sold, starter/gold-item groups.

Non-store or special items in the CLASSIC lists, not modelled as items:
- turret/structure info items 1501–1524 and 771500;
- Recall 2001/2007;
- Poro-Snax 2052;
- Gangplank upgrades 3901–3903 and placeholder 7050;
- Minion Dematerializer 2403;
- Triple Tonic elixirs 2150–2152;
- Total Biscuit 2010 (rune);
- Your Cut 3400 (Pyke);
- not-in-store legacy entries: 2033 Corrupting, 2422, 3002, 3010, 3011, 3013, 3117, 3176, 4635–4637, 4641, 6693, 6701, 8001, 3866, 3867.

---

## 17. Test fixtures (concrete numbers)

All are pre-mitigation unless stated. Mitigation = 100/(100 + armor).

| # | Setup | Expected |
|---|---|---|
| F1 | Tiamat, holder melee, total AD 150; 3 minions at 200, 340 and 360 from the primary | cleave hits 2 minions (200, 340) for **60** each; the 360 minion gets 0 (centre rule). Edge rule with minion radius 48 (INFERRED) would also hit 360 |
| F2 | Cleave, 12 enemies within 350 | exactly **10** splash instances |
| F3 | Crescent, AD 150, caster at (0,0) facing +x; targets at (520,0), (−340,0), (100,440) | centre (100,0): distances 420, 440 and 440, so **all 3 hit for 112.5**. The current code (`tiamat_crescent`, caster-centred 450 + forward half-plane) misses all three: 520 > 450, behind, 451.2 > 450 |
| F4 | Ravenous Crescent, AD 200, holder 12% LS, target armor 50 | damage 160 × 100/150 = **106.67**; LS heal 12.8 (×1.0 VampAmp) |
| F5 | Stridebreaker active, AD 180, hits 2 champions and 3 minions | each takes **144** physical; all slowed 35% for 3 s; self +70% bonus MS at t = 0, decaying linearly to 0 at t = 3 s (35% at 1.5 s); cooldown ready at cast_start + 15 s |
| F6 | Titanic Hydra, melee, max HP 2500, normal attack vs a champion | on-hit +**25** bonus physical (LS-eligible); cone **75** to others behind. Empowered: **100** on-hit, **225** cone. Ranged holder: half |
| F7 | Sterak's: base AD 70 (level-scaled), bonus HP 1000, max HP 2600, HP 900, incoming 200 post-mitigation | Claws +**35** AD. 900 − 200 = 700 < 780 (30%), so Lifeline triggers first: shield **600**; the 200 is absorbed by the shield, leaving 400 shield; HP stays 900; cd 90 s |
| F8 | Trinity Force, base AD 80: cast Q (t = 0), attack hits at t = 0.6 | Spellblade **160** bonus physical; next arm possible from t = 2.1. Q recast at t = 1.0 cannot re-arm until 2.1 |
| F9 | Black Cleaver 5 stacks vs 100 armor (all bonus or base, total) | effective armor before pen **70**. One stack per frame from non-attack damage |
| F10 | Plated Steelcaps vs a 100-damage basic attack | **90**. Warden's Mail vs a 50-damage champion attack: reduction min(15, 10) → **40** |
| F11 | Doran's Shield, melee, max HP 1000, HP 400 (60% missing), champion hit at t = 0 | regen bonus (40/8)·(0.6/0.75) = **4.0 HP/s** (recomputed as HP rises); AoE trigger → 2.64 HP/s |
| F12 | Sell values | Trinity 3333 → **2333**; Pickaxe 875 → **613**; Doran's Shield 450 → **180**; Tiamat 1200 → **840**; Stridebreaker 3300 → **2310** |
| F13 | Buy Ravenous (Tiamat + Vampiric Scepter + Caulfield's + 150) holding Tiamat + Long Sword, 1500 g | Vampiric Scepter = Long Sword + 550, so the owned Long Sword is consumed as a sub-component. Cost = 3300 − 1200 − 350 = **1750** > 1500, so rejected; at 1750 g it is accepted and both slots are freed |
| F14 | Group check | holding Tiamat, buying Stridebreaker by recipe: allowed. Holding Titanic, buying Tiamat: **rejected** (Hydra max 1). Holding Doran's Blade, buying Doran's Shield: rejected. Doran's Shield + Cull: allowed. Health Potion + Refillable: rejected |
| F15 | Level-scaled values | Shieldbow shield L13 = 400 + 30·5 = **550**; Kraken L12 = 150 + 5·4 = **170**; Hexdrinker L10 = 110 + 170·9/17 = **200**; Armored Advance shield L13 with 1000 bonus HP = 90 + 50 + 80 = **220** |
| F16 | Overlord's, bonus HP 2000, other-source AD 200, HP 50% | Tyranny +50 AD; Retribution 12%·(0.5/0.7) = 8.571% of 250 = **+21.4 AD** (default reading D12) |
| F17 | Elixir of Iron purchase at level 8 | rejected; at level 9: OK; a second elixir purchase within 5 s: rejected |
| F18 | Health Potion | +120 HP over 15 s, 4 HP every 0.5 s; under GW: 2.4 per tick |

---

## 18. Summary table (all SR store and transform items, client 16.19.8230722)

Groups: only the non-default, non-self groups are shown (H = Hydra, SB = Spellblade, LL = Lifeline, FAT = Fatality, BLT = Blight, ST = Starter/DoransItems, BT = Boots, POT = Potion, TEAR = Manaflow, IMM = Immolate, ANN = Annul, QS = Quicksilver, GL = Glory, ETR = Eternity, THO = Thorns, MOM = Momentum). Passive names come from the client tooltip `<passive>` and `<active>` tags.

| ID | Item | Category | Total | Combine | Sell | Stats | Named effects | Groups |
|---|---|---|---|---|---|---|---|---|
| 1083 | Cull | Starter | 450 | 450 | 180 | 7AD | Reap |  |
| 1082 | Dark Seal | Starter | 350 | 350 | 140 | 50HP 15AP | Glory | GL |
| 1055 | Doran's Blade | Starter | 450 | 450 | 180 | 80HP 10AD 2.5OV% |  | ST |
| 1086 | Doran's Bow | Starter | 400 | 400 | 160 | 8AD 15AS% 1.5OV% |  | ST |
| 1120 | Doran's Helm | Starter | 450 | 450 | 180 | 150HP 8Arm 8MR | Helping Hand | ST |
| 1056 | Doran's Ring | Starter | 400 | 400 | 160 | 90HP 18AP | Drain, Helping Hand | ST |
| 1054 | Doran's Shield | Starter | 450 | 450 | 180 | 110HP 4HP5 | Enduring Focus, Helping Hand | ST |
| 3070 | Tear of the Goddess | Starter | 400 | 400 | 280 | 240Mana | Manaflow, Helping Hand | TEAR |
| 2055 | Control Ward | Consumables and trinkets | 75 | 75 | 30 |  |  | WARD |
| 2138 | Elixir of Iron | Consumables and trinkets | 500 | 500 | 200 |  |  | ELX |
| 2139 | Elixir of Sorcery | Consumables and trinkets | 500 | 500 | 200 |  |  | ELX |
| 2140 | Elixir of Wrath | Consumables and trinkets | 500 | 500 | 200 |  |  | ELX |
| 3363 | Farsight Alteration | Consumables and trinkets | 0 | 0 | 0 |  |  |  |
| 2003 | Health Potion | Consumables and trinkets | 50 | 50 | 20 |  |  | POT |
| 3364 | Oracle Lens | Consumables and trinkets | 0 | 0 | 0 |  |  |  |
| 2031 | Refillable Potion | Consumables and trinkets | 150 | 150 | 60 |  | Active | POT |
| 3340 | Stealth Ward | Consumables and trinkets | 0 | 0 | 0 |  | Active |  |
| 1052 | Amplifying Tome | Basic | 400 | 400 | 280 | 20AP |  |  |
| 1038 | B. F. Sword | Basic | 1300 | 1300 | 910 | 40AD |  |  |
| 1026 | Blasting Wand | Basic | 850 | 850 | 595 | 45AP |  |  |
| 1018 | Cloak of Agility | Basic | 600 | 600 | 420 | 15Crit% |  |  |
| 1029 | Cloth Armor | Basic | 300 | 300 | 210 | 15Arm |  |  |
| 1042 | Dagger | Basic | 250 | 250 | 175 | 10AS% |  |  |
| 1004 | Faerie Charm | Basic | 200 | 200 | 140 | 50BaseMPRegen% |  |  |
| 2022 | Glowing Mote | Basic | 250 | 250 | 175 | 5AH |  |  |
| 1036 | Long Sword | Basic | 350 | 350 | 245 | 10AD |  |  |
| 1058 | Needlessly Large Rod | Basic | 1200 | 1200 | 840 | 65AP |  |  |
| 1033 | Null-Magic Mantle | Basic | 400 | 400 | 280 | 20MR |  |  |
| 1037 | Pickaxe | Basic | 875 | 875 | 613 | 25AD |  |  |
| 1006 | Rejuvenation Bead | Basic | 300 | 300 | 120 | 100BaseHPRegen% |  |  |
| 1028 | Ruby Crystal | Basic | 400 | 400 | 280 | 150HP |  |  |
| 1027 | Sapphire Crystal | Basic | 300 | 300 | 210 | 300Mana |  |  |
| 3113 | Aether Wisp | Epic | 900 | 500 | 630 | 30AP 4MS% |  |  |
| 6660 | Bami's Cinder | Epic | 900 | 250 | 630 | 150HP 5AH | Immolate | IMM |
| 4642 | Bandleglass Mirror | Epic | 900 | 50 | 630 | 20AP 10AH 100BaseMPRegen% |  |  |
| 4630 | Blighting Jewel | Epic | 1100 | 700 | 770 | 25AP 13MPen% |  | BLT |
| 3076 | Bramble Vest | Epic | 800 | 200 | 560 | 30Arm | Thorns | THO |
| 3803 | Catalyst of Aeons | Epic | 1300 | 200 | 910 | 300HP 375Mana | Eternity | ETR |
| 3133 | Caulfield's Warhammer | Epic | 1050 | 100 | 735 | 20AD 10AH |  |  |
| 1031 | Chain Vest | Epic | 800 | 500 | 560 | 40Arm |  |  |
| 3801 | Crystalline Bracer | Epic | 800 | 100 | 560 | 200HP 100BaseHPRegen% |  |  |
| 3123 | Executioner's Calling | Epic | 800 | 450 | 560 | 15AD | Grievous Wounds |  |
| 2508 | Fated Ashes | Epic | 900 | 500 | 630 | 30AP | Inflame |  |
| 3108 | Fiendish Codex | Epic | 850 | 200 | 595 | 25AP 10AH |  |  |
| 3114 | Forbidden Idol | Epic | 600 | 400 | 420 | 50BaseMPRegen% 8HSP% |  |  |
| 1011 | Giant's Belt | Epic | 900 | 500 | 630 | 350HP |  |  |
| 3024 | Glacial Buckler | Epic | 900 | 50 | 630 | 25Arm 10AH 300Mana |  |  |
| 3147 | Haunting Guise | Epic | 1300 | 500 | 910 | 200HP 30AP | Madness |  |
| 3051 | Hearthbound Axe | Epic | 1200 | 250 | 840 | 20AD 20AS% |  |  |
| 3155 | Hexdrinker | Epic | 1300 | 200 | 910 | 25AD 25MR | Lifeline | LL |
| 3145 | Hextech Alternator | Epic | 1100 | 300 | 770 | 45AP | Revved |  |
| 3067 | Kindlegem | Epic | 800 | 150 | 560 | 200HP 10AH |  |  |
| 3035 | Last Whisper | Epic | 1450 | 750 | 1015 | 20AD 18ArPen% |  | FAT |
| 3802 | Lost Chapter | Epic | 1200 | 250 | 840 | 40AP 10AH 300Mana | Enlighten |  |
| 1057 | Negatron Cloak | Epic | 850 | 450 | 595 | 45MR |  |  |
| 6670 | Noonquiver | Epic | 1300 | 350 | 910 | 15AD 20Crit% |  |  |
| 3916 | Oblivion Orb | Epic | 800 | 400 | 560 | 25AP | Grievous Wounds |  |
| 3044 | Phage | Epic | 1100 | 350 | 770 | 200HP 15AD | Rage |  |
| 3140 | Quicksilver Sash | Epic | 1300 | 900 | 910 | 30MR | Quicksilver | QS |
| 6690 | Rectrix | Epic | 775 | 425 | 543 | 15AD 4MS% |  |  |
| 1043 | Recurve Bow | Epic | 700 | 450 | 490 | 15AS% | Sting |  |
| 3144 | Scout's Slingshot | Epic | 600 | 100 | 420 | 20AS% | Bullseye |  |
| 2420 | Seeker's Armguard | Epic | 1600 | 500 | 640 | 40AP 25Arm | Time Stop | STASIS |
| 3134 | Serrated Dirk | Epic | 1000 | 300 | 700 | 20AD 10Leth |  |  |
| 3057 | Sheen | Epic | 900 | 650 | 630 | 10AH | Spellblade | SB |
| 3211 | Spectre's Cowl | Epic | 1250 | 150 | 875 | 200HP 35MR 100BaseHPRegen% |  |  |
| 2019 | Steel Sigil | Epic | 1100 | 150 | 770 | 15AD 30Arm |  |  |
| 2020 | The Brutalizer | Epic | 1337 | 212 | 936 | 25AD 10AH 5Leth |  |  |
| 3077 | Tiamat | Epic | 1200 | 500 | 840 | 25AD | Cleave, Crescent | H |
| 2021 | Tunneler | Epic | 1150 | 400 | 805 | 250HP 15AD |  |  |
| 1053 | Vampiric Scepter | Epic | 900 | 550 | 630 | 15AD 7LS% |  |  |
| 4632 | Verdant Barrier | Epic | 1600 | 400 | 1120 | 40AP 25MR | Annul | ANN |
| 3082 | Warden's Mail | Epic | 1000 | 400 | 700 | 40Arm |  |  |
| 3066 | Winged Moonplate | Epic | 800 | 400 | 560 | 200HP 4MS% |  |  |
| 3086 | Zeal | Epic | 1200 | 350 | 840 | 15AS% 15Crit% 4MS% |  |  |
| 3006 | Berserker's Greaves | Boots | 1100 | 300 | 770 | 30AS% 45MS |  | BT |
| 1001 | Boots | Boots | 300 | 300 | 210 | 25MS |  | BT |
| 3009 | Boots of Swiftness | Boots | 1000 | 700 | 700 | 55MS 25SlowRes% | Fleetfooted | BT |
| 3008 | Gluttonous Greaves | Boots | 1000 | 700 | 700 | 4OV% 45MS | Slay | BT |
| 3158 | Ionian Boots of Lucidity | Boots | 900 | 350 | 630 | 45MS 10AH | Ionian Insight | BT |
| 3111 | Mercury's Treads | Boots | 1250 | 550 | 875 | 20MR 45MS 30Ten% |  | BT |
| 3047 | Plated Steelcaps | Boots | 1200 | 600 | 840 | 25Arm 45MS | Plating | BT |
| 3020 | Sorcerer's Shoes | Boots | 1100 | 800 | 770 | 45MS 12MPen |  | BT |
| 3174 | Armored Advance | Boots | 1200 | 0 | 840 | 35Arm 45MS | Plating, Noxian Endurance | BT |
| 3173 | Chainlaced Crushers | Boots | 1250 | 0 | 875 | 25MR 45MS 30Ten% | Noxian Persistence | BT |
| 3171 | Crimson Lucidity | Boots | 900 | 0 | 630 | 45MS 20AH | Ionian Insight, Noxian Haste | BT |
| 3172 | Gunmetal Greaves | Boots | 1100 | 0 | 770 | 45AS% 5LS% 45MS |  | BT |
| 3168 | Immortal Path | Boots | 1000 | 0 | 700 | 4OV% 45MS | Slay, Now and Forever | BT |
| 3175 | Spellslinger's Shoes | Boots | 1100 | 0 | 770 | 45MS 20MPen 8MPen% |  | BT |
| 3170 | Swiftmarch | Boots | 1000 | 0 | 700 | 65MS 25SlowRes% | Fleetfooted, Noxian Fervor | BT |
| 8020 | Abyssal Mask | Legendary | 2650 | 1000 | 1855 | 350HP 45MR 15AH | Unmake |  |
| 2522 | Actualizer | Legendary | 2800 | 750 | 1960 | 90AP 10AH 300Mana | Mana Made Real |  |
| 3003 | Archangel's Staff | Legendary | 2900 | 450 | 2030 | 70AP 25AH 600Mana | Awe, Manaflow | LL,TEAR |
| 3504 | Ardent Censer | Legendary | 2200 | 700 | 1540 | 45AP 4MS% 125BaseMPRegen% 10HSP% | Sanctify |  |
| 6696 | Axiom Arc | Legendary | 2750 | 363 | 1925 | 55AD 20AH 18Leth | Flux |  |
| 2524 | Bandlepipes | Legendary | 2300 | 800 | 1610 | 200HP 20Arm 20MR 15AH | Fanfare |  |
| 3102 | Banshee's Veil | Legendary | 3000 | 200 | 2100 | 105AP 40MR | Annul | ANN |
| 2520 | Bastionbreaker | Legendary | 3000 | 663 | 2100 | 55AD 15AH 22Leth | Shaped Charge, Sabotage |  |
| 3071 | Black Cleaver | Legendary | 3000 | 225 | 2100 | 400HP 45AD 20AH | Carve, Fervor | FAT |
| 2503 | Blackfire Torch | Legendary | 2800 | 700 | 1960 | 80AP 20AH 600Mana | Baleful Blaze, Blackfire |  |
| 3153 | Blade of The Ruined King | Legendary | 3200 | 725 | 2240 | 40AD 25AS% 10LS% | Mist's Edge, Clawing Shadows |  |
| 8010 | Bloodletter's Curse | Legendary | 2900 | 750 | 2030 | 400HP 65AP 15AH | Vile Decay | BLT |
| 3072 | Bloodthirster | Legendary | 3400 | 325 | 2380 | 80AD 15LS% | Ichorshield |  |
| 6609 | Chempunk Chainsword | Legendary | 3000 | 250 | 2100 | 450HP 45AD 15AH | Hackshorn |  |
| 4629 | Cosmic Drive | Legendary | 3000 | 450 | 2100 | 350HP 70AP 4MS% 25AH | Spelldance |  |
| 3137 | Cryptbloom | Legendary | 3000 | 200 | 2100 | 75AP 20AH 30MPen% | Life from Death, Life From Death | BLT |
| 6621 | Dawncore | Legendary | 2500 | 450 | 1750 | 45AP 100BaseMPRegen% 16HSP% | First Light |  |
| 3742 | Dead Man's Plate | Legendary | 2900 | 900 | 2030 | 350HP 55Arm 4MS% 15SlowRes% | Shipwrecker, Unsinkable | MOM |
| 6333 | Death's Dance | Legendary | 3300 | 275 | 2310 | 60AD 50Arm 15AH | Ignore Pain, Defy, Ignore Pain's |  |
| 2510 | Dusk and Dawn | Legendary | 3100 | 300 | 2170 | 300HP 60AP 20AS% 20AH | Spellblade | SB |
| 6620 | Echoes of Helia | Legendary | 2200 | 500 | 1540 | 200HP 35AP 20AH 125BaseMPRegen% | Soul Siphon, Soul Charges |  |
| 6692 | Eclipse | Legendary | 2900 | 625 | 2030 | 60AD 15AH | Ever Rising Moon |  |
| 3814 | Edge of Night | Legendary | 3000 | 850 | 2100 | 250HP 50AD 15Leth | Annul | ANN |
| 2517 | Endless Hunger | Legendary | 3100 | 825 | 2170 | 65AD 5OV% 20Ten% | Famine, Feast |  |
| 3508 | Essence Reaver | Legendary | 3050 | 500 | 2135 | 50AD 25Crit% 20AH | Spellblade | SB |
| 3073 | Experimental Hexplate | Legendary | 3000 | 500 | 2100 | 450HP 40AD 20AS% | Hexcharged, Overdrive |  |
| 2512 | Fiendhunter Bolts | Legendary | 2650 | 850 | 1855 | 45AS% 25Crit% 4MS% | Night Vigil, Opening Barrage |  |
| 4401 | Force of Nature | Legendary | 2800 | 750 | 1960 | 400HP 55MR 4MS% | Steadfast |  |
| 3110 | Frozen Heart | Legendary | 2500 | 600 | 1750 | 75Arm 20AH 400Mana | Winter's Caress |  |
| 3026 | Guardian Angel | Legendary | 3200 | 800 | 1280 | 55AD 45Arm | Rebirth |  |
| 3124 | Guinsoo's Rageblade | Legendary | 3000 | 1025 | 2100 | 30AD 30AP 25AS% | Wrath, Seething Strike |  |
| 3084 | Heartsteel | Legendary | 3000 | 400 | 2100 | 900HP 100BaseHPRegen% |  |  |
| 2523 | Hexoptics C44 | Legendary | 2800 | 275 | 1960 | 55AD 25Crit% | Magnification, Arcane Aim |  |
| 3146 | Hextech Gunblade | Legendary | 3000 | 600 | 2100 | 40AD 80AP 10OV% | Lightning Bolt |  |
| 3152 | Hextech Rocketbelt | Legendary | 2650 | 350 | 1855 | 350HP 60AP 20AH | Supersonic, Supersonic's |  |
| 6664 | Hollow Radiance | Legendary | 2800 | 650 | 1960 | 400HP 40MR 10AH 100BaseHPRegen% | Immolate, Desolate | IMM |
| 4628 | Horizon Focus | Legendary | 2700 | 600 | 1890 | 75AP 25AH | Hypershot, Focus |  |
| 6697 | Hubris | Legendary | 2800 | 750 | 1960 | 55AD 10AH 18Leth | Eminence |  |
| 3181 | Hullbreaker | Legendary | 3000 | 175 | 2100 | 500HP 40AD 4MS% | Skipper, Boarding Party |  |
| 6662 | Iceborn Gauntlet | Legendary | 2900 | 800 | 2030 | 300HP 50Arm 15AH | Spellblade | SB |
| 6673 | Immortal Shieldbow | Legendary | 3000 | 825 | 2100 | 55AD 25Crit% | Lifeline | LL |
| 4005 | Imperial Mandate | Legendary | 2400 | 700 | 1680 | 60AP 15AH 150BaseMPRegen% | Control, Command |  |
| 3031 | Infinity Edge | Legendary | 3500 | 725 | 2450 | 75AD 25Crit% 30CritDmg% |  |  |
| 6665 | Jak'Sho, The Protean | Legendary | 3200 | 650 | 2240 | 350HP 45Arm 45MR | Voidborn Resilience |  |
| 2504 | Kaenic Rookern | Legendary | 2900 | 800 | 2030 | 400HP 80MR 100BaseHPRegen% | Magebane |  |
| 3109 | Knight's Vow | Legendary | 2300 | 400 | 1610 | 200HP 40Arm 10AH 100BaseHPRegen% | Sacrifice, Pledge |  |
| 6672 | Kraken Slayer | Legendary | 3000 | 325 | 2100 | 45AD 40AS% 4MS% | Bring It Down |  |
| 6653 | Liandry's Torment | Legendary | 3000 | 800 | 2100 | 300HP 60AP | Torment, Suffering |  |
| 3100 | Lich Bane | Legendary | 2900 | 250 | 2030 | 100AP 6MS% 10AH | Spellblade | SB |
| 3190 | Locket of the Iron Solari | Legendary | 2200 | 700 | 1540 | 200HP 30Arm 30MR 10AH | Devotion |  |
| 3036 | Lord Dominik's Regards | Legendary | 3300 | 550 | 2310 | 35AD 25Crit% 35ArPen% | Giant Slayer | FAT |
| 6655 | Luden's Echo | Legendary | 2750 | 450 | 1925 | 100AP 10AH 600Mana | Echo |  |
| 3118 | Malignance | Legendary | 2700 | 650 | 1890 | 90AP 15AH 600Mana | Scorn, Hatefog |  |
| 3004 | Manamune | Legendary | 2900 | 1100 | 2030 | 35AD 15AH 500Mana | Awe, Manaflow | TEAR |
| 3156 | Maw of Malmortius | Legendary | 3100 | 750 | 2170 | 60AD 40MR 15AH | Lifeline | LL |
| 3041 | Mejai's Soulstealer | Legendary | 1500 | 1150 | 1050 | 100HP 20AP | Glory | GL |
| 3139 | Mercurial Scimitar | Legendary | 3200 | 125 | 2240 | 50AD 35MR 10LS% | Quicksilver, Activate | QS |
| 3222 | Mikael's Blessing | Legendary | 2300 | 900 | 1610 | 250HP 15AH 100BaseMPRegen% 12HSP% | Purify |  |
| 6617 | Moonstone Renewer | Legendary | 2200 | 500 | 1540 | 200HP 25AP 20AH 125BaseMPRegen% |  |  |
| 3165 | Morellonomicon | Legendary | 2850 | 400 | 1995 | 350HP 75AP 15AH | Grievous Wounds |  |
| 3033 | Mortal Reminder | Legendary | 3000 | 150 | 2100 | 35AD 25Crit% 30ArPen% | Grievous Wounds | FAT |
| 3115 | Nashor's Tooth | Legendary | 2900 | 500 | 2030 | 80AP 50AS% 15AH | Icathian Bite |  |
| 6675 | Navori Flickerblade | Legendary | 2650 | 950 | 1855 | 40AS% 25Crit% 4MS% | Transcendence |  |
| 2501 | Overlord's Bloodmail | Legendary | 3300 | 1000 | 2310 | 550HP 30AD | Tyranny, Retribution |  |
| 3046 | Phantom Dancer | Legendary | 2650 | 950 | 1855 | 65AS% 25Crit% 10MS% | Spectral Waltz |  |
| 6698 | Profane Hydra | Legendary | 2850 | 313 | 1995 | 55AD 10AH 18Leth | Cleave, Heretical Cleave | H |
| 2525 | Protoplasm Harness | Legendary | 2600 | 900 | 1820 | 600HP 20AH | Lifeline | LL |
| 3089 | Rabadon's Deathcap | Legendary | 3500 | 1100 | 2450 | 130AP | Magical Opus |  |
| 3143 | Randuin's Omen | Legendary | 2700 | 800 | 1890 | 350HP 75Arm | Resilience, Humility |  |
| 3094 | Rapid Firecannon | Legendary | 2650 | 850 | 1855 | 35AS% 25Crit% 4MS% | Sharpshooter |  |
| 3074 | Ravenous Hydra | Legendary | 3300 | 150 | 2310 | 65AD 12LS% 15AH | Cleave, Ravenous Crescent | H |
| 3107 | Redemption | Legendary | 2300 | 850 | 1610 | 30AP 15AH 100BaseMPRegen% 10HSP% | Intervention |  |
| 4633 | Riftmaker | Legendary | 3100 | 950 | 2170 | 350HP 70AP 15AH | Void Corruption, Void Infusion |  |
| 6657 | Rod of Ages | Legendary | 2600 | 450 | 1820 | 350HP 45AP 500Mana | Eternity | ETR |
| 3085 | Runaan's Hurricane | Legendary | 2650 | 850 | 1855 | 40AS% 25Crit% 5MS% | Wind's Fury |  |
| 3116 | Rylai's Crystal Scepter | Legendary | 2600 | 450 | 1820 | 400HP 65AP | Rimefrost |  |
| 6695 | Serpent's Fang | Legendary | 2500 | 625 | 1750 | 55AD 15Leth | Shield Reaver |  |
| 6694 | Serylda's Grudge | Legendary | 3000 | 500 | 2100 | 45AD 15AH 35ArPen% | Bitter Cold | FAT |
| 4645 | Shadowflame | Legendary | 3200 | 900 | 2240 | 110AP 15MPen | Cinderbloom |  |
| 2421 | Shattered Armguard | Legendary | 1600 | 500 | 640 | 40AP 25Arm | Shattered Time | STASIS |
| 2065 | Shurelya's Battlesong | Legendary | 2200 | 400 | 1540 | 50AP 4MS% 15AH 125BaseMPRegen% | Inspiring Speech |  |
| 3161 | Spear of Shojin | Legendary | 3100 | 675 | 2170 | 450HP 45AD | Dragonforce, Focused Will |  |
| 3065 | Spirit Visage | Legendary | 2700 | 650 | 1890 | 400HP 50MR 10AH 100BaseHPRegen% | Boundless Vitality |  |
| 6616 | Staff of Flowing Water | Legendary | 2250 | 800 | 1575 | 35AP 10AH 125BaseMPRegen% 10HSP% | Rapids |  |
| 3087 | Statikk Shiv | Legendary | 3000 | 625 | 2100 | 45AD 45AP 30AS% 4MS% | Electrospark, Electroshock |  |
| 3053 | Sterak's Gage | Legendary | 3200 | 775 | 2240 | 400HP 20Ten% | The Claws that Catch, Lifeline | LL |
| 3095 | Stormrazor | Legendary | 3200 | 700 | 2240 | 50AD 25AS% 25Crit% | Bolt |  |
| 4646 | Stormsurge | Legendary | 2800 | 800 | 1960 | 90AP 6MS% 15MPen | Stormraider, Squall |  |
| 6631 | Stridebreaker | Legendary | 3300 | 750 | 2310 | 450HP 40AD 25AS% | Cleave, Breaking Shockwave | H |
| 6610 | Sundered Sky | Legendary | 3100 | 900 | 2170 | 400HP 40AD 10AH | Lightshield Strike |  |
| 3068 | Sunfire Aegis | Legendary | 2800 | 700 | 1960 | 350HP 50Arm 10AH | Immolate | IMM |
| 3302 | Terminus | Legendary | 3000 | 1100 | 2100 | 30AD 35AS% | Shadow, Juxtaposition | BLT,FAT |
| 6676 | The Collector | Legendary | 3000 | 525 | 2100 | 50AD 25Crit% 10Leth | Death, Taxes |  |
| 3075 | Thornmail | Legendary | 2450 | 450 | 1715 | 150HP 75Arm | Thorns | THO |
| 3748 | Titanic Hydra | Legendary | 3300 | 50 | 2310 | 600HP 40AD | Cleave, Titanic Crescent, Cleave's | H |
| 3078 | Trinity Force | Legendary | 3333 | 133 | 2333 | 333HP 36AD 30AS% 15AH | Spellblade, Quicken | SB |
| 3179 | Umbral Glaive | Legendary | 2800 | 750 | 1960 | 60AD 15AH 18Leth | Nightstalker, Blackout |  |
| 2502 | Unending Despair | Legendary | 2800 | 800 | 1960 | 400HP 50Arm 15AH | Anguish |  |
| 3135 | Void Staff | Legendary | 3000 | 1050 | 2100 | 95AP 40MPen% |  | BLT |
| 6699 | Voltaic Cyclosword | Legendary | 3000 | 963 | 2100 | 55AD 10AH 10Leth | Galvanize, Firmament |  |
| 3083 | Warmog's Armor | Legendary | 3100 | 500 | 2170 | 1000HP 100BaseHPRegen% | Warmog's Heart, Warmog's Vitality |  |
| 2526 | Whispering Circlet | Legendary | 2250 | 850 | 1575 | 200HP 300Mana 75BaseMPRegen% 8HSP% | Harmony, Manaflow | TEAR |
| 3119 | Winter's Approach | Legendary | 2400 | 300 | 1680 | 550HP 15AH 500Mana | Awe, Manaflow | TEAR |
| 3091 | Wit's End | Legendary | 2800 | 550 | 1960 | 45MR 50AS% 20Ten% | Fray |  |
| 3142 | Youmuu's Ghostblade | Legendary | 2800 | 675 | 1960 | 55AD 4MS% 18Leth | Haunt, Wraith Step |  |
| 3032 | Yun Tal Wildarrows | Legendary | 3000 | 750 | 2100 | 50AD 45AS% | Practice Makes Lethal, Flurry |  |
| 3050 | Zeke's Convergence | Legendary | 2200 | 700 | 1540 | 300HP 25Arm 25MR 10AH | Cryocombustion, Frostfire Tempest |  |
| 3157 | Zhonya's Hourglass | Legendary | 3250 | 450 | 2275 | 105AP 50Arm | Time Stop |  |
| 2530 | Diadem of Songs | Transformed | 2250 | 2250 | 1575 | 200HP 1000Mana 100BaseMPRegen% 8HSP% | Harmony, Consonance |  |
| 3121 | Fimbulwinter | Transformed | 2400 | 2400 | 1680 | 550HP 15AH 1000Mana | Awe, Everlasting |  |
| 3042 | Muramana | Transformed | 2900 | 2900 | 2030 | 35AD 15AH 1000Mana | Awe, Shock |  |
| 3040 | Seraph's Embrace | Transformed | 2900 | 2900 | 2030 | 70AP 25AH 1000Mana | Awe, Lifeline | LL |
| 3877 | Bloodsong | Support quest line | 400 | 0 | 160 | 200HP 75BaseHPRegen% 75BaseMPRegen% | Spellblade | SB,ST |
| 3869 | Celestial Opposition | Support quest line | 400 | 0 | 160 | 200HP 75BaseHPRegen% 75BaseMPRegen% | Blessing of the Mountain | ST |
| 3870 | Dream Maker | Support quest line | 400 | 0 | 160 | 200HP 75BaseHPRegen% 75BaseMPRegen% | Dream Maker | ST |
| 3876 | Solstice Sleigh | Support quest line | 400 | 0 | 160 | 200HP 75BaseHPRegen% 75BaseMPRegen% | Going Sledding | ST |
| 3865 | World Atlas | Support quest line | 400 | 400 | — | 50BaseHPRegen% 25BaseMPRegen% |  | ST |
| 3871 | Zaz'Zak's Realmspike | Support quest line | 400 | 0 | 160 | 200HP 75BaseHPRegen% 75BaseMPRegen% | Void Explosion | ST |
| 3599 | Kalista's Black Spear | Champion-specific | 0 | 0 | 0 |  |  |  |
| 3600 | Kalista's Black Spear | Champion-specific | 0 | 0 | 0 |  |  |  |
| 3330 | Scarecrow Effigy | Champion-specific | 0 | 0 | — |  |  |  |
| 1102 | Gustwalker Hatchling | Jungle | 450 | 450 | — |  | Jungle Companions, Gustwalker's Gait, @TotalCounters@ Treats, @Breakpoint1@ Treats:, @TotalCounters@ Treats: | ST |
| 1103 | Mosstomper Seedling | Jungle | 450 | 450 | — |  | Jungle Companions, Mosstomper's Courage, @TotalCounters@ Treats, @Breakpoint1@ Treats:, @TotalCounters@ Treats: | ST |
| 1101 | Scorchclaw Pup | Jungle | 450 | 450 | — |  | Jungle Companions, Scorchclaw's Slash, @TotalCounters@ Treats, @Breakpoint1@ Treats:, @TotalCounters@ Treats: | ST |
