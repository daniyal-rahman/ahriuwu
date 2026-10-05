# RUNES.md — Runes Reforged spec for the modern JAX simulator (patch 26.19)

**Scope.** Every Runes Reforged rune and stat shard available on normal PC Summoner's Rift
(map 11, queue mode `CLASSIC`) at patch **26.19**, client build **16.19.8230722**. Covers the
page legality rules, every rune's numbers and trigger rules, the champion-combat definitions they
depend on, the state and hooks the simulator needs, and how the current code differs. Mode
overrides (ARAM, URF, Nexus Blitz, Arena, Swiftplay, League Classic) are out of scope and are
**not** to be applied. Champion kits are out of scope (owned by other agents); champion-specific
special cases are listed only where a rune's rules require them.

Depth is highest for the runes a top-lane melee champion takes (Precision keystones, Grasp,
Aftershock, Stormraider's Surge, Resolve minors, Precision minors, Domination row 1 and 3,
Inspiration utility). Vision runes are listed with their effects but are flagged
**DEFERRED-VISION** (MODERN-009 defers wards and vision).

- **Patch pin:** 26.19 / client 16.19.8230722 / Data Dragon 16.19.1.
- **Retrieval date:** 2026-10-01.
- **Researcher:** research-only pass. No code, data JSON or other docs were changed.
- **Implemented:** 2026-10-01, see [RUNES_IMPLEMENTATION.md](RUNES_IMPLEMENTATION.md) (§11 D-1…D-11 resolved).

## 0. Sources

### 0.1 Client data (CommunityDragon raw 16.19 = build 16.19.8230722)

The cache is at `/mnt/nfs/shared/modern-world-map-research/cdragon-16.19/`, and each file has a
SHA256 line in that directory's `SHA256SUMS`.

| File | Origin | SHA256 |
|---|---|---|
| `perks.cdtb.bin.json` | `game/data/perks/…` bin (PerkStyle, Perk, PerkSlot, mEffectAmount, mCalculations) | `1427c70c4d1172198a1a9787362224c871a90cc4f66f3f769ef5830cbcd401b1` |
| `cdragon-perks.json` | `plugins/rcp-be-lol-game-data/global/default/v1/perks.json` (rendered tooltips) | `dd0948288ac0293caf71675cabbd138a7ad512b7a7cbb15c084fa9662233c13a` |
| `globals.cdtb.bin.json` | global bins (`GlobalPerLevelStatsFactor`, `DamageSourceSettings`) | `ee7c25bbf08e4000d5a39078f29334da6cc27f9ca38439d480a511f719b2366b` |
| `items.cdtb.bin.json` | item bins (Total Biscuit 2010, potions, elixirs) | `6880f35d5a9d82f688192f764e280e6d4bc9c845112b001feb811e3f2ab62726` |
| `shared.cdtb.bin.json` | shared spells (summoner spells, Hexflash) | `34f68553ab38cfe344936473fcb48d99574d5fd51e4feb42994fa0c4fe50769e` |
| `en_us-lol.stringtable.json` (fetched 2026-10-01 from `https://raw.communitydragon.org/16.19/game/en_us/data/menu/en_us/lol.stringtable.json`) | localized tooltips | `8c051cb2a24b31f3fa9af95d39085b832b0da2b8cf51093f98c4ab0620ecb8e8` |
| Repo `lanerl_jax/modern/data/26.19/runes.json` | DDragon `16.19.1/data/en_US/runesReforged.json` | upstream `74d1e211…0fc44` (in file); local file `e11961b6a3e81c09f31c52550f7d97fbcd6811ec564e62c4114da4eb76945e7c` |

The rune Lua scripts (`ASSETS/Perks/Styles/**.lua`) are compiled, and **were not decompiled**.
So the bin gives the numbers, but trigger logic not stated in data comes from the wiki and the
tooltips.

### 0.2 Riot patch notes 26.1–26.19 (all fetched 2026-10-01)

- URL form `https://www.leagueoflegends.com/en-us/news/game-updates/patch-26-N-notes/` works for N = 1–3.
- URL form `…/league-of-legends-patch-26-N-notes/` works for N = 4–19.
- In-page hotfix sections exist for 26.1, 26.3, 26.6 and 26.9. There are no standalone 26.x hotfix articles.
- Stripped text is archived in `cdragon-16.19/riot-patchnotes-26.x/26.N.txt`, with sha256 lines in `SHA256SUMS`.

### 0.3 League wiki

All pages were fetched on 2026-10-01 through `api.php?action=query&prop=revisions`. Permalinks
have the form `https://wiki.leagueoflegends.com/en-us/index.php?oldid=N`. The raw JSON is
archived in `cdragon-16.19/wiki-2026-10-01/`.

Most numbers live in `Template:Rune data <Rune>`. Its oldid is cited per rune as **T:oldid**, and
the article page as **P:oldid**. Shared pages:

| Page | oldid |
|---|---|
| Rune | 4070732 |
| Adaptive force | 4063874 |
| Combat status | 4058480 |
| Takedown | 4035702 |
| Assist | 4016680 |
| Damage modifier | 4050138 (flagged "Outdated" by the wiki) |
| Haste | 4070414 |
| Template:Energized info | 4064669 |
| Template:Tip data/Adaptive damage | 4015091 |
| Template:Tip data/Immobilize | 4060567 |
| Template:Tip data/Proc damage | 4060539 |
| Total Biscuit of Everlasting Will | 3971313 |
| Module:ItemData/data/Total Biscuit… | 3905887 |
| Role Quests | 4064833 |

### 0.4 Prior research being re-verified

`docs/MODERN_PATCH_DELTA.md` §7 and §11 were pinned to 26.18. Their corrections are collected in
§12 of this document.

### 0.5 Confidence tags

| Tag | Meaning |
|---|---|
| **CLIENT** | The value is read from the 16.19 bin or the client tooltip string (CLIENT-DATA-VERIFIED). |
| **RIOT** | Riot patch notes 26.x. |
| **WIKI** | League wiki at the cited oldid. |
| **INFERRED** | The researcher's reasoning or default, with a confidence of H, M or L. |

When CLIENT and WIKI agree, the tag is written `CLIENT+WIKI`.

---

## 1. Global conventions every rune uses

### 1.1 Level interpolation (the exact formulas)

The bin uses three level-scaling primitives. These are the **only** formulas implementers need.
`L` is the champion level.

1. **`ByCharLevelInterpolationCalculationPart(start, end)`** is linear:
   `v(L) = start + (end - start) * (L - 1) / 17`.
   This is CLIENT for the shape. The wiki writes the same thing as `a + (b-a)/17*(x-1)` on every
   rune. If the part has `mScaleByStatProgressionMultiplier: true` (only Fleet Footwork's heal
   among the runes), the fraction `(L-1)/17` is replaced by the champion stat-growth fraction:
   `g(L) = n*(0.7025 + 0.0175*n) / 17`, where `n = L - 1`.
   This is CLIENT: `GlobalPerLevelStatsFactor` in `globals.cdtb.bin.json` has per-level steps
   0.72, 0.755, … (+0.035/level), whose cumulative sum equals `n(0.7025+0.0175n)`. It matches
   `core.stats.level_growth_sum`.
2. **`ByCharLevelBreakpointsCalculationPart(level1, initialBonusPerLevel, breakpoints[])`** is
   piecewise per-level increments:
   `v(1) = level1`, and for each level k = 2..L, `v += perLevel(k) + additionalAt(k)`.
   - `perLevel` starts at `initialBonusPerLevel`.
   - A breakpoint at level k sets `perLevel = mBonusPerLevelAtAndAfter`, which **defaults to 0
     when omitted**, from level k onward.
   - `mAdditionalBonusAtThisLevel` is a one-off added at level k.
   This is CLIENT, verified against the wiki's tables for Absorb Life and Unleashed Teleport.
3. **`ByCharLevelFormulaCalculationPart(values[])`** is a table indexed by level, where index 0
   is unused or equal to level 1. It is used only by Presence of Mind.

**Levels 19–20.** The top-lane role quest raises the level cap to 20 (RIOT 26.1; WIKI Role Quests
4064833). The bin has an `mScalePastDefaultMaxLevel` flag, and it is explicitly `false` only on
First Strike's cooldown. **Default (INFERRED-M):** linear interpolations **extrapolate** past 18
using the same formula (`(L-1)/17 > 1`), except where `mScalePastDefaultMaxLevel=false`, which
clamps at L = 18.

Evidence for this default:

- Specifying `false` only matters if the default is `true`.
- The wiki renders every rune formula "for 20" levels. For example, the Health Scaling shard is
  shown as `10*x` → 200 at level 20.

Rune values whose scaling comes from Lua with only Min/Max data (for example Conqueror's
`MinAdaptivePerStack`/`MaxAdaptivePerStack`, Aftershock and the HP shard) carry no flag. Apply the
same default and list each in the Unresolved register (U-01).

### 1.2 Adaptive force and adaptive damage

**Adaptive force (AF).** 1 AF gives 0.6 **bonus** AD *or* 1 AP. The choice is made
**dynamically**, not fixed per loadout: if bonus AD > AP, the AF goes to AD; if AP > bonus AD, it
goes to AP; on a tie (including 0/0), the champion's "adaptive type" decides (physical gives AD).
This is WIKI (Adaptive force 4063874). The 0.6/1.0 ratios are CLIENT (GameplayConfig, already
used in `core.stats.adaptive_force_total`).

- Bonus AD/AP from champion passives does **not** count toward the comparison. Item stats and a
  listed set of item passives do.
- Whether rune AF itself feeds back into the comparison is not stated. **Default (INFERRED-M):**
  compute the comparison on bonus AD and AP **excluding** all AF-derived stats, then apply all AF
  to the chosen stat. This avoids flip-flop oscillation.
- **Top-lane AD champion (Garen):** bonus AD ≥ 0 = AP always, and the tie goes to physical. So all
  AF goes to AD in every reachable loadout (INFERRED-H).

**Adaptive damage** is physical if bonus AD > AP, magic if AP > bonus AD, and adaptive type on a
tie (WIKI Tip data/Adaptive damage 4015091).

Electrocute and Arcane Comet use a different rule ("variable damage"): the type follows which
**ratio term contributes more bonus damage** (AD-ratio term vs AP-ratio term). On a tie or zero it
defaults to **magic** (WIKI T:Electrocute 4060699). For a champion with bonus AD > 0 and AP = 0,
Electrocute is physical; at 0 bonus AD and 0 AP, it is **magic**.

### 1.3 Damage tags that rune triggers read

The client damage tags are `AOE, Periodic, Indirect, BasicAttack, ActiveSpell, Proc, Pet,
NonRedirectable, Item, DoesNotAggroJungle, OnHit, Augment, Burn, NonAmpable` (CLIENT
`Globals/DamageSourceSettings`). The simulator's damage event must carry at least:

- `source_unit`, `source_owner_champion`, `target`
- `damage_type` (phys/magic/true), `pre_mitigation`, `post_mitigation`
- tag bits: `basic_attack`, `on_hit`, `active_spell`, `aoe`, `periodic`, `proc`, `pet`, `item`,
  `indirect`, `non_ampable`
- `cast_instance_id`
- `is_crit`

Rune output damage is generally tagged **Proc** (Electrocute, PTA, Lethal Tempo bolt, Grasp, HoB,
Sudden Impact, Cheap Shot, Dark Harvest, Aery, Scorch, Arcane Comet, First Strike). Proc damage
does **not** trigger "spell effects" and does **not** itself stack Electrocute, Stormraider's or
Dark Harvest (WIKI, per-rune notes).

### 1.4 Damage-dealt modifiers (Coup de Grace, Cut Down, Last Stand, PTA, Axiom; Exhaust)

WIKI (Damage modifier 4050138, marked *Outdated*, with "pending test after V26.09"): "All damage
modifiers stack additively."

**Default (INFERRED-M):**

```
outgoing_multiplier = 1 + sum(amps) - sum(dealt_reductions_like_Exhaust)
final = raw * outgoing_multiplier * (target received-damage modifiers, multiplicative) * resist_multiplier
```

- Received-damage modifiers stack **multiplicatively** with each other and with armor/MR (WIKI,
  same page).
- Since V25.S1.3, Coup de Grace, Cut Down, Last Stand and PTA also amplify **true damage**,
  excluding Smite (WIKI per-rune history).
- Ignite is **not** amplifiable (WIKI Ignite 4053158, V25.15).

See the code discrepancy D-6 in §11.

### 1.5 Combat-state definitions

| Term | Definition | Tag |
|---|---|---|
| **in combat (generic, "damage" system)** | The unit dealt or received damage (including 0-damage instances) or CC to/from an enemy **champion, minion, monster or turret** (not wards). It ends **5 s** after the last such event. A spell-shield-blocked hit does not count. | WIKI Combat status 4058480 |
| **in combat ("modern" system)** | Same as above, but also counts non-damage CC and *does* count hits on invulnerable targets, while ignoring plants and wards. Used by Hextech Flashtraption and Relentless Hunter. | WIKI |
| **champion combat** | The same definition restricted to an **enemy champion** counterpart. Pets cause their owner to enter combat; clones do not. | WIKI |
| **out of combat** | 5 s after the last combat event (Relentless Hunter: "immediately gained after 5 seconds"). | WIKI |
| **takedown** | Kill or assist on an **enemy champion**. Assist window on SR: the enemy dies within **15 s** of your last damage/debuff on it, or of a heal/shield/buff you gave to the killer or an assister. The timer refreshes on each contribution. | WIKI Assist 4016680 |
| **"after taking damage from an enemy champion"** (Second Wind, Bone Plating) | A damage event with `source_owner_champion` = enemy champion and `post_mitigation > 0` that reaches **health**. It does not count if fully absorbed by a shield, absorbed by invulnerability, or reduced to 0. | WIKI P:Second Wind 4012933, P:Bone Plating 4032827 |
| **"damaging an enemy champion"** (Conqueror, ToB, Electrocute…) | Damage event with target = enemy champion, including 0-damage instances unless a rune says otherwise. Arcane Comet excludes 0 damage. Dark Harvest excludes damage < 2. Taste of Blood does not fire at full HP. | WIKI per rune |
| **immobilize** | Airborne, forced action (berserk/charm/flee/taunt), root, sleep, stasis, stun/suspension, suppression. "Impaired movement" adds slows. | WIKI Tip data/Immobilize 4060567 |

**Pet damage default (INFERRED-M):** for "taking damage from an enemy champion", damage whose
`source_owner_champion` is the enemy champion counts. Mark this as U-02.

### 1.6 Healing and shielding modifiers applied to rune heals

Rune heals (Conqueror, Grasp, Fleet, ToB, Triumph, Second Wind regen, Biscuits, Absorb Life, Font
of Life) are ordinary healing. They are reduced by **Grievous Wounds 40%** (WIKI Grievous Wounds
3969410; CLIENT `GrievousAmount 0.4` on Ignite) and increased by Heal & Shield Power and
Revitalize.

The healing is clamped at max HP: overheal is lost. No 26.19 rune grants overheal shields, and
Overheal was removed and replaced by Absorb Life.

---

## 2. Page structure and legality

### 2.1 The 26.19 rune set

The rows below come from the bin `PerkStyle.mSlots` (CLIENT) and match DDragon `runes.json`.
Slot 0 is the keystone row. Hash-named entries were resolved through `mPerkId`.

| Tree (id) | Keystone (row 0) | Row 1 | Row 2 | Row 3 |
|---|---|---|---|---|
| **Precision 8000** | Press the Attack 8005 · Lethal Tempo 8008 · Fleet Footwork 8021 · Conqueror 8010 | Absorb Life 9101 · Triumph 9111 · Presence of Mind 8009 | Legend: Alacrity 9104 · Legend: Haste 9105 · Legend: Bloodline 9103 | Coup de Grace 8014 · Cut Down 8017 · Last Stand 8299 |
| **Domination 8100** | Electrocute 8112 · Dark Harvest 8128 · Hail of Blades 9923 | Cheap Shot 8126 · Taste of Blood 8139 · Sudden Impact 8143 | Sixth Sense 8137 · Grisly Mementos 8140 · Deep Ward 8141 | Treasure Hunter 8135 · Relentless Hunter 8105 · Ultimate Hunter 8106 |
| **Sorcery 8200** | Summon Aery 8214 · Arcane Comet 8229 · **Stormraider's Surge 8230** (DDragon key `PhaseRush`) · **Deathfire Touch 8992** | Axiom Arcanist 8224 (bin name `NullifyingOrb`) · Manaflow Band 8226 · Nimbus Cloak 8275 | Transcendence 8210 · Celerity 8234 · Absolute Focus 8233 | Scorch 8237 · Waterwalking 8232 · Gathering Storm 8236 |
| **Resolve 8400** | Grasp of the Undying 8437 · Aftershock 8439 · Guardian 8465 | Demolish 8446 · Font of Life 8463 · Shield Bash 8401 | Conditioning 8429 · Second Wind 8444 · Bone Plating 8473 | Overgrowth 8451 · Revitalize 8453 · Unflinching 8242 |
| **Inspiration 8300** | Glacial Augment 8351 · Unsealed Spellbook 8360 · First Strike 8369 | Hextech Flashtraption 8306 · Magical Footwear 8304 · Cash Back 8321 | Triple Tonic 8313 (DDragon key `PerfectTiming`) · Time Warp Tonic 8352 · Biscuit Delivery 8345 | Cosmic Insight 8347 · Approach Velocity 8410 · Jack of All Trades 8316 |

There are **62 selectable runes** (5 trees: 17 keystones in rows of 4/3/4/3/3, plus 45 minors). Corrected 2026-10-01 from 63 against `runes_client.json`.

The bin also contains these perks, which are **not selectable** on SR at 26.19 and must be
rejected:

- Retired perks still present: Predator 8124, Zombie Ward 8136, Ghost Poro 8120, Eyeball
  Collection 8138, Ingenious Hunter 8134, Kleptomancy 8359, Celestial Body 8339, Chrysalis 8472,
  Iron Skin 8430, Mirror Shell 8435.
- Stat mods outside the SR shard slots: Armor 5002 (6), Magic Resist 5003 (8), Resist Scaling
  5012 (1–8), AttackDamage 5004, AbilityPower 5006 and AdaptiveScaling 5009 (`mEnabled:false`).
- The `*SetBonus` perks: these are all 0-valued.
- The Domination style script has `LethalityAmount: 15`. **Do not apply it** (INFERRED-H): it is
  not in any tooltip and is legacy.

**Rune changes during 26.1–26.19 (RIOT; WIKI histories agree).**

| Patch | Change |
|---|---|
| 26.1 | **Demolish rework** (see §7.4). Bugfixes for Rengar with Conqueror, Lethal Tempo and Axiom. |
| 26.3 | Cash Back 8% → 7.5%. Phase Rush ranged effectiveness 75% → 50%. Triple Tonic Elixir of Force 30 → 25 AF. Bugfix: Azir W gave only 1 Conqueror stack. |
| 26.9 (season 2) | **Phase Rush removed; Stormraider's Surge added** (same perk id 8230). **Deathfire Touch added** (8992). **Arcane Comet rework** (cooldown refund removed; distance amp up to +100% at 750; base 30–130 → 15–100). **Hail of Blades rework** (AS 160/80 → 120/60%, max 2 bonus hits from resets, new true damage on-hit). Hotfix 4/30 changed Deathfire Touch ratios. |
| 26.10 | Stormraider's 40/30% for 3 s → 48/36% for 4 s. Deathfire 4–12 → 3–12 per second. |
| 26.11 | Deathfire Touch damage becomes magic (was adaptive). Aery shield 30–100 → 20–100 (+10% bonus AD, +5% AP). **Aftershock 35 (+80% bonus) → 45 (+75% bonus)** resists. Guardian cooldown 90–40 → 75–40, shield 45–180 (+25% AP, +5% bonus HP) → 40–150 (+20% AP, +6% bonus HP). |
| 26.14 | Azir W: Conqueror stacks 1 → 2; PTA applies to primary target. |
| 26.15 | Jack of All Trades AF 10/25 → 8/20. Deathfire duration-overwrite bugfix. |
| 26.16 | **Fleet Footwork heal 10–130 → 15–160.** **Hail of Blades AS 120% → 90% melee** (ranged 60% unchanged); true damage 4–20 (+8% bonus AD, +6% AP) → 2–20 (+12% bonus AD, +10% AP). |
| 26.19 | Bugfix: "Camille's W would count as two hits when triggering Bone Plating". Shield Bash proc fix on Camille passive. |

No stat-shard change occurred in 26.1–26.19 (RIOT; subagent audit of all 19 pages).

### 2.2 Legality rules (implement as host-side validation; reject, never silently fix)

1. **Primary path.** Choose exactly one `primary_style` from the 5 trees. Then pick exactly 1
   rune from its row 0 and exactly 1 rune from each of rows 1, 2 and 3 (4 runes). CLIENT
   `mSlots`, WIKI Rune.
2. **Secondary path.**
   - `secondary_style` must be in `primary.mAllowedSubStyles`. For every tree, that list is all 4
     other trees, so the secondary must be ≠ primary.
   - Pick exactly 2 runes from the secondary tree's rows **1–3**, never row 0.
   - The 2 runes must come from **two different rows**.
3. **Shards.** Pick exactly 3, one per row. `Adaptive` may be picked in both Offense and Flex.
   `HealthScaling` may be picked in both Flex and Defense (`mStackable:true`).

   | Row | Allowed (perk id) |
   |---|---|
   | Offense (`OffensiveStats`) | Adaptive 5008 · AttackSpeed 5005 · CDRScaling 5007 |
   | Flex (`FlexStats`) | Adaptive 5008 · MovementSpeed 5010 · HealthScaling 5001 |
   | Defense (`DefensiveStats`) | Health 5011 · Tenacity 5013 · HealthScaling 5001 |

4. **Automatic substitutions at game start** apply silently in-client, so the sim must apply
   them identically. All are WIKI Rune 4070732, "last updated V26.06". The CLIENT shows
   Flashtraption → Cash Back in `mSummonerPerkReplacements`.
   - Champion with no immobilizing effect, or Yorick: Aftershock → **Grasp of the Undying**, and
     Glacial Augment → **First Strike**. *Garen has no immobilize, so Aftershock is not legal in
     effect for Garen.*
   - Champion without mana: Manaflow Band → **Axiom Arcanist**. Without mana or energy: Presence
     of Mind → **Triumph**. Garen is manaless, so both apply.
   - Flash not equipped: Hextech Flashtraption → **Cash Back**.
   - Mode-only substitutions (no river, no structures, no wards) do not apply on SR. Waterwalking
     stays Waterwalking, Demolish stays Demolish, Deep Ward and Sixth Sense stay as picked.
   - Champion-specific: Bel'Veth Ultimate Hunter → Relentless Hunter. Samira Ultimate Hunter →
     Treasure Hunter. Elise, Jayce, Nidalee and Zoe Axiom Arcanist → Nimbus Cloak.
   - **Implementation:** `substitute(page, champion_traits) -> page`, with champion traits
     `{has_immobilize, resource ∈ {mana, energy, none}, special_id}` supplied by the champion
     module.
5. A **missing or invalid page** is replaced in-client by the recommended page. The simulator
   must instead **reject** it, per MODERN-002 ("reject missing required values"). The explicit
   empty page is a separate "no runes" ruleset, not a legal SR page; keep it only as a test/legacy
   switch.

### 2.3 Stat shards (exact values)

All shards are permanent, unconditional and granted at spawn; level-scaling shards update on
level-up. All are CLIENT (`Perks/StatMods/*` `StatGain*`, plus the cdragon tooltip strings) and
WIKI Rune 4070732.

| Shard | Id | Value | Stat bucket | Tag |
|---|---|---|---|---|
| Adaptive Force | 5008 | **+9 AF** = +5.4 bonus AD or +9 AP (`StatGain1 5.4`, `StatGain2 9`) | flat bonus AD/AP (adaptive §1.2) | CLIENT+WIKI |
| Attack Speed | 5005 | **+10% bonus AS** | bonus AS ratio | CLIENT+WIKI |
| Ability Haste | 5007 (`CDRScaling`) | **+8 AH** (all champion abilities; not item or summoner haste) | ability haste | CLIENT+WIKI |
| Move Speed | 5010 | **+2.5% MS** (`StatGain1 2.5`) | percent (additive-percent) bonus MS | CLIENT+WIKI |
| Health Scaling | 5001 | **+10 to +180 by level**, linear: `10 + 170*(L-1)/17 = 10*L`. Extrapolated to 190/200 at L19/20 per §1.1 (U-01). | flat bonus HP | CLIENT (Min 10/Max 180) + WIKI (`10*x for 20`) |
| Health | 5011 | **+65 HP** | flat bonus HP | CLIENT+WIKI |
| Tenacity and Slow Resist | 5013 | **+15% tenacity and +15% slow resist** (`StatGain 15`) | tenacity group (multiplicative stacking with other sources), slow resist | CLIENT+WIKI |

Rules for applying shards:

- **HP gain on level-up.** When Health Scaling grows +10 on level-up, current HP increases by the
  same delta, like any max-HP gain. This is the `change_max_health` behaviour. INFERRED-H; it is
  standard for stat gains.
- **Fixtures, level-1 floors.** A/A/65 gives 10.8 bonus AD and 65 HP. A/A/HS gives 10.8 AD and
  10 HP at L1, and 180 HP at L18. A/HS/HS gives 5.4 AD and 20 HP at L1, and 360 HP at L18 (400
  at L20 under the U-01 default).
- **Recommended page.** The prior doc's Garen page was Conqueror / Triumph / Legend: Haste / Last
  Stand + Sorcery Axiom Arcanist / Celerity, with Adaptive/Adaptive/HealthScaling shards. It is
  legal. The client default shard set for Precision primary with a Sorcery secondary is
  AttackSpeed/Adaptive/HealthScaling (`mDefaultStatModsPerSubStyle`).

---

## 3. Precision (8000)

`lin(a,b) = a + (b-a)*(L-1)/17` per §1.1. "Melee/Ranged" uses the champion's current range type.

### 3.1 Conqueror 8010 — keystone (TOP PRIORITY)

Sources: CLIENT `MinAdaptivePerStack 1.8, MaxAdaptivePerStack 4.0, MaxStacks 12, BuffDuration 5,
TimeUntilNextStackFromSameSpell 4, HealingPercent 0.08, RangedHealingPercent 0.05`;
WIKI T:4060685, P:4058162.

**Rules (CLIENT+WIKI unless noted):**

- **Stack gain.** Triggered by a damage event on an **enemy champion** from you.
  - Basic damage with on-hit (`basic_attack & on_hit`) gives **+2 stacks for melee, +1 for
    ranged**. This applies per on-hit application, so multi-on-hit attacks stack multiple times.
  - Any other damage that is neither basic damage nor non-pet proc damage gives **+2 stacks**, at
    most once per `cast_instance_id`. Damage over time can re-stack from the same cast instance
    once every **4 s** (`TimeUntilNextStackFromSameSpell`).
  - These grant **no** stacks: proc damage from non-pets (rune procs, Grasp, item procs), dodged
    or blinded misses, attacks on invulnerable targets, and basic damage that doesn't apply
    on-hit.
  - These **do** grant stacks: blocked attacks, 0-damage instances, raw/default damage (WIKI).
  - **Special cases:** Ignite gives **+2 on activation** (WIKI P:Ignite). Tiamat/Stridebreaker
    actives stack (V9.14, WIKI). **Garen E Judgment stacks on every damage tick**, which matters
    for Garen.
- **Stacks and duration.** Max 12. All stacks share one **5 s** duration, refreshed by any
  subsequent damage to an enemy champion — even damage that grants no new stacks (INFERRED-M from
  "refreshing on subsequent damage"). On expiry, **all stacks are lost at once**; there is no
  per-stack decay (WIKI Stack page lists only Lethal Tempo as interval-decay).
- **Stat.** Each stack gives `lin(1.8, 4.0)` **AF** (§1.2: 0.6 bonus AD each for AD champions).
  - Full value: 21.6 AF (12.96 AD) at L1, 48 AF (28.8 AD) at L18.
  - **Wiki-documented bug:** the per-stack value is **locked to your level when the first stack
    was generated** and not updated by level-ups while the buff persists. **Default:** reproduce
    it (lock `conq_level` at the 0 → >0 transition). It is a fidelity choice, flagged U-03.
- **Healing at max stacks.** While at 12 stacks, heal for **8% (melee) / 5% (ranged)** of
  **post-mitigation damage dealt to enemy champions**. This covers all damage types, including
  true damage and the triggering instance, and is reduced by Grievous Wounds.
  - Whether the instance that *completes* the 12th stack heals is INFERRED-M "yes": the buff is
    applied in the same damage event, before the heal hook.
  - **Ordering:** run the heal hook *after* damage is applied and after shields absorb. "Damage
    dealt" includes damage absorbed by the target's shield (INFERRED-M; U-04).

```
on_damage_to_enemy_champion(ev):
  if ev.tags.proc and not ev.tags.pet: heal_if_full(ev); return   # proc: no stack, no refresh (INFERRED-M), but full-stack heal applies to all damage
  gain = 0
  if ev.tags.basic_attack and ev.tags.on_hit: gain = 1 if ranged else 2
  elif not ev.tags.basic_attack:
      if ev.cast_id not in conq_seen or now - conq_seen[ev.cast_id] >= 4.0: gain = 2; conq_seen[ev.cast_id] = now
  if conq_stacks == 0 and gain > 0: conq_level = L
  conq_stacks = min(12, conq_stacks + gain); if gain>0 or conq_stacks>0: conq_expire = now + 5.0
  heal_if_full(ev)
heal_if_full(ev): if conq_stacks == 12: heal(self, (0.05 if ranged else 0.08) * ev.post_mitigation)
tick: if now >= conq_expire: conq_stacks = 0
stat: bonus_AF += conq_stacks * lin(1.8, 4.0)(conq_level)
```

### 3.2 Press the Attack 8005 — keystone

Sources: CLIENT `MinDamage 40, MaxDamage 160, HitsRequired 3, TimeBetweenHits 4,
BonusPercentDamage 0.08, AmpPotencyMax/StartSelf/Others 0.08, OutOfCombatTimer 5, Cooldown 6`;
WIKI T:4015574, P:4071927.

- Each basic attack **on-hit** against an enemy champion applies a stack on that champion.
  - Stacks last **4 s**, refreshing on each new stack. Max 3.
  - Attacking a **different** champion expires the stacks on the old one. Tracking is per
    attacker–target pair; multiple users don't interact.
  - Multi-on-hit attacks add multiple stacks.
  - AoE on-hit (Tiamat cleave, Runaan's) does not apply stacks.
- **The 3rd stack** consumes all stacks. It deals `lin(40,160)` **bonus adaptive damage** to the
  target. The damage is proc: no spell effects, and it is not affected by on-hit modifiers.
- **The 3rd stack also** grants the user **+8% damage dealt against enemy champions**, a
  damage-dealt amp (§1.4) covering all types including true damage, excluding Smite.
  - The amp lasts **until 5 s after the user leaves champion combat**. **Default (INFERRED-M):**
    it expires when `now - last_champion_combat_time >= 5`, and applies against all enemy
    champions (the `Others` potency equals the self value 0.08).
  - The triggering attack and the 40–160 proc do **not** benefit from this amp. Later procs and
    damage do. Damage on the same frame as buff gain is not amped.
- **Cooldown 6 s**, starting after the consumption. No stacks can be applied during it.

### 3.3 Lethal Tempo 8008 — keystone

Sources: CLIENT `Duration 6, MaxStacks 6, ASPerStack 0.06 (ranged ×0.8),
NoteDamage lin(9,30)×(1 + bonusAS) (ranged ×0.667)`; WIKI T:4065934.

- Each basic attack **on-attack** (at attack launch, not at hit) against an enemy champion gives
  +1 stack.
  - Duration **6 s**, refreshing. Max **6**.
  - Each stack gives +6% bonus AS for melee, **4.8%** for ranged (CLIENT calc ×0.8; the tooltip
    "4%" is a known text error, WIKI). That is 36% / 28.8% at max.
- **At max stacks**, each basic attack on-attack against a champion fires a bolt.
  - Damage: `lin(9,30) * (1 + bonusAS_ratio)` bonus **adaptive** damage, on arrival. `bonusAS`
    is the bonus attack-speed stat, including Lethal Tempo's own 36%, and is not capped by the AS
    cap. Ranged champions use ×0.667.
  - The damage is proc.
- **Decay.** When the 6 s timer ends, 1 stack is removed immediately and the remaining stacks
  are removed one every **0.3 s**. Gaining a stack stops the decay. This is WIKI; it is the only
  interval-decay rune.
- The bin field `PerfectlyTimedForgivance 0.25` is unexplained (U-05).

### 3.4 Fleet Footwork 8021 — keystone

Sources: CLIENT `HealBase 15, HealMax 160 (mScaleByStatProgressionMultiplier), HealBonusADRatio
0.1, HealAPRatio 0.05, RangedHealMod 0.6, MSBuff 0.2, MSDuration 1, RangedMSMod 0.75,
MinionHealMod 0.15`; WIKI T:4051351, P:4052588. RIOT 26.16 changed the heal from 10–130 to 15–160.

- **Energize charges, 0–100** (WIKI Template:Energized info 4064669):
  - +6 per basic attack on-attack, against any target (INFERRED-M: the target type is not
    restricted).
  - +6 per ability hit that applies on-hit.
  - +1 per **24 units** travelled by any movement: walking, dashes, blinks, displacement.
  - Implement as `energy += dist/24` with a float accumulator, clamped to 100.
- At **100**, the next basic attack is *Energized*. On hit, consume all 100:
  - **Heal** `15 + 145*g(L) + 0.10*bonusAD + 0.05*AP`, where `g` is the stat-growth fraction
    (§1.1). Ranged champions use ×0.6 on everything (6% bonus AD / 3% AP ratios).
  - Gain **+20% bonus MS for 1 s** (ranged 15%).
  - If the target is a **minion**, the heal is ×0.15. Monsters are not reduced (INFERRED-M:
    only minions are named).
  - The charge is **not consumed** by attacks on wards or plants.
- Fixtures (melee, 0 bonus AD/AP): L1 15.0, L6 48.69, L9 72.49, L13 108.40, L18 160.0.

### 3.5 Absorb Life 9101 — row 1

Killing a target heals you `1, +0.25/level until L5, +1/level L6–L10, +2/level from L11` (CLIENT
breakpoints; WIKI T:3997095). Values: L1 1, L5 2, L6 3, L10 7, L11 9, L18 23 (L20 27 under U-01).

- "Target" is any enemy unit **you kill**, i.e. last-hit: minions, monsters, champions.
  INFERRED-M; the wiki says "Killing an enemy".
- The heal is instant.

### 3.6 Triumph 9111 — row 1

Sources: CLIENT `MissingHealthRestored 0.05, TriumphMaxHealthRestored 0.025, BonusGold 20`;
WIKI T:3825924, P:3980916.

- On **champion takedown**, after a **1.0 s** delay (an unblockable missile with fixed 1 s
  travel), heal `0.025*maxHP + 0.05*missingHP_at_takedown_time` and grant **+20 gold**.
- If another takedown happens during the delay, all pending heals are recomputed with the newest
  missing HP (WIKI bug). Reproduce it (INFERRED-M).
- If you die during the delay: INFERRED-M, heal and gold are still delivered and the heal is
  wasted (U-06).

### 3.7 Presence of Mind 8009 — row 1 (manaless champions get Triumph instead)

Sources: CLIENT table `RegenAmount[L]` = 6, 6.8, 7.6, 8.4, 9.2, 10, 12, 14, 16, 18, 20, 23.2,
26.4, 29.6, 32.8, 36, 40, 44 (L1–18), then 52, 56 (L19–20). Also `PercentManaRestore 0.15,
CooldownDuration 8, EnergyRestore 6`; WIKI T:3985219.

- Damaging an enemy champion restores `RegenAmount[L]` mana, ×0.8 for ranged, or 6 energy, on an
  8 s cooldown. The tooltip says 6–50; the calc table gives 44 at L18. **Use the table.**
- A champion takedown restores 15% of max mana/energy after 1 s.
- The rune is consumed even at full mana (WIKI bug).

### 3.8 Legend: Alacrity 9104 / Legend: Haste 9105 / Legend: Bloodline 9103 — row 2

Sources: CLIENT `MinionKillValue 4, LargeMonsterKillValue 25`, three 100-valued hashed keys
(champion takedown, epic monster takedown, and one more), and `MaxLegendStacks`; WIKI
T:3958705/3958696/3971359.

- **Legend points:**
  - +100 per champion **takedown**
  - +100 per epic monster takedown
  - +25 per large monster **kill**
  - +4 per minion **kill** (last hit)
- `stacks = min(MaxLegendStacks, floor(points/100))`. In lane, that is 25 minion kills per stack.
  Points past the cap are irrelevant.

| Rune | Per stack | Base | Max stacks | At max |
|---|---|---|---|---|
| Alacrity | +1.5% bonus AS | +3% bonus AS | 10 | 18% AS |
| **Haste** | **+1.5 basic ability haste** (Q/W/E only, not R) | 0 | **10** | **15 basic AH** |
| Bloodline | +0.45% life steal | 0 | 15 | 6.75% LS, **+85 bonus max HP** |

Basic ability haste adds to ability haste for basic abilities only. The total haste cap is 500
(WIKI Haste 4070414).

### 3.9 Coup de Grace 8014 / Cut Down 8017 / Last Stand 8299 — row 3

All three are damage-dealt amps against **champions** (§1.4). They cover all damage types
including true damage, excluding Smite, since V25.S1.3 (WIKI).

| Rune | Condition (evaluated per damage instance, at the moment it is dealt) | Amp |
|---|---|---|
| Coup de Grace | target `HP/maxHP < 0.40` (CLIENT `EnemyHealthPercentageThreshold 0.4`) | +8% |
| Cut Down | target `HP/maxHP > 0.60` (CLIENT `EnemyHealthPercentageThreshold 0.6`, `BonusPercentDamage 0.08`) | +8% |
| Last Stand | **own** `h = HP/maxHP < 0.60` | `0.05 + 0.06*clamp((0.60-h)/0.30, 0, 1)`; 5% just below 60%, 11% at ≤ 30% |

- **Do not use** Cut Down's leftover bin fields `Min/MaxBonusDamagePercent 0.05/0.15`, which
  belong to the old health-difference version. The 26.19 tooltip is a flat 8% above 60% (CLIENT
  tooltip; WIKI T:3859781).
- Cut Down had a double-application bug fixed in the 26.9 hotfix.
- Coup de Grace checks HP *before* the instance is applied (WIKI: "only triggers on damage dealt
  after the champion is brought below 40%").
- Last Stand's linear ramp between 60% and 30% is WIKI (T:3859782, table "5 to 11"; CLIENT
  `MinBonusDamagePercent 0.05, MaxBonusDamagePercent 0.11, HealthThresholdStart 0.6, End 0.3`).

---

## 4. Domination (8100)

### 4.1 Electrocute 8112 — keystone

Sources: CLIENT `DamageBase 70, DamageMax 240, BonusADRatio 0.1, APRatio 0.05, WindowDuration 3,
Cooldown 20`; WIKI T:4060699, P:4015230.

- **Stacks** are applied to an enemy champion by damaging basic attacks, abilities, item
  effects, summoner spells, CC application and DoT application. There is at most **1 stack per
  cast instance per champion**.
  - Proc damage does not stack unless it is also pet damage. Debuff application by proc effects
    (for example Black Cleaver) does stack.
  - These CC types do not count: blind, cripple, drowsy, kinematics, nearsight, stasis.
- **Trigger.** 3 stacks within **3 s of the first stack**. The window does **not** refresh.
- **Effect.** After a **0.25 s** delay, deal `lin(70,240)` (= 60 + 10L) + 10% bonus AD + 5% AP
  *variable* damage (§1.2).
  - Becoming untargetable during the delay does not prevent the hit.
  - It can trigger while the user is dead.
  - Proc.
- **Cooldown** 20 s.
- **Fixtures:** L1 70, L6 120, L9 150, L18 240.

### 4.2 Dark Harvest 8128 — keystone

Sources: CLIENT `HarvestThreshold 0.5, BaseDamage 30, DamagePerSoulEssence 11, ADRatio 0.1
(bonus), APRatio 0.05, Cooldown 35, CooldownResetValue 1`; WIKI T:4046509.

- **Trigger.** Pet or non-proc damage ≥ 2 to an enemy champion below 50% max HP. Checked before
  or after the damage? INFERRED-M: checked on the HP *after* the triggering instance, per the
  tooltip wording "Damaging a Champion below 50%" (U-07).
- **Effect.** Deals `30 + 11*souls + 0.1*bonusAD + 0.05*AP` bonus adaptive damage. After
  **1.75 s**, gain 1 soul. Souls are permanent and unbounded.
- **Cooldown** 35 s. A champion takedown resets the remaining cooldown to **1 s**.
- **Extra soul.** While off cooldown, you also gain 1 soul when credited with a kill on a
  champion that was killed by a minion, monster or turret.

### 4.3 Hail of Blades 9923 — keystone

Sources: CLIENT `ASBoost 0.9, ASBoostRanged 0.6, Duration 3, NumHits 3, MaxBonusHits 2,
Cooldown 10, BonusDamageMin 2, BonusDamageMax 20, BonusADRatio 0.12, APRatio 0.1`;
WIKI T:4060661, P:4052375; RIOT 26.9 and 26.16.

- **Trigger.** Starting an attack windup against an enemy champion.
  - If the windup completes, gain 2 stacks for 3 s. The triggering attack also benefits, so 3
    empowered attacks in total.
  - The duration refreshes on each basic attack on-attack against a champion.
  - Each on-attack consumes 1 stack.
  - An effect tagged `Trait_AttackReset` adds +1 stack, at most **2** per activation (MaxBonusHits).
- **While active:**
  - +90% bonus AS for melee (ranged 60%), and the AS cap is lifted.
  - Each empowered attack deals `lin(2,20)` + 12% bonus AD + 10% AP **bonus true damage**, as
    proc on-hit.
- **End.** The effect ends when stacks run out or 3 s pass without an attack.
- **Cooldown** 10 s, starting **after the effect ends**.
- **Cancelled windup.** If the triggering windup is cancelled, no stacks are granted and the rune
  goes on a brief cooldown, then resets (WIKI).

### 4.4 Cheap Shot 8126 — row 1

Sources: CLIENT `DamageIncMin 10, DamageIncMax 45, Cooldown 4`; WIKI T:3992989, P:4023785.

- **Trigger.** Non-proc damage to an enemy champion that is *already* affected by one of these
  impairments: immobilize, blind, disarm, ground, nearsight, polymorph, silence or slow.
  - Cripple does not count.
  - The impairment must exist before the damage event. CC applied in the same tick by the same
    cast instance doesn't count, except CC applied on-hit by that instance (WIKI).
- **Effect.** `lin(10,45)` bonus true damage, proc. Cooldown 4 s.

### 4.5 Taste of Blood 8139 — row 1

Sources: CLIENT `HealAmount 16, HealAmountMax 40, ADRatio 0.1, APRatio 0.05, Cooldown 20,
RegenDuration 4`; WIKI T:3997097.

- **Trigger.** Damaging an enemy champion with any damage. It does not trigger at full HP.
- **Effect.** Heal `lin(16,40) + 0.1*bonusAD + 0.05*AP`. Cooldown 20 s.
- The client has `RegenDuration 4`, which suggests the heal may be delivered over 4 s. The
  tooltip and wiki describe it as a single heal. **Default:** instant heal (U-08).
- **Fixtures** (0 bonus AD): L1 16, L9 27.29, L18 40.

### 4.6 Sudden Impact 8143 — row 1

Sources: CLIENT `ArmedDuration 4, Cooldown 10, Min/MaxDamageTooltip 20/80`; WIKI T:3825916,
P:4023786.

- **Arming.** A dash, blink, exit from stealth, Flash, Hexflash, Teleport/Unleashed Teleport
  arrival, or Recall arms the rune for **4 s**. Lunges and emerging from fog or brush do not arm
  it.
- **Effect.** The first damage (including 0-damage instances) to an enemy champion while armed
  deals `lin(20,80)` bonus true damage, as proc.
  - It is not affected by on-attack modifiers (since 26.9).
- **Cooldown.** 10 s, starting after the bonus is applied or the arm expires. It cannot re-arm
  during the active or cooldown phases.

### 4.7 Sixth Sense 8137 / Grisly Mementos 8140 / Deep Ward 8141 — row 2 (DEFERRED-VISION)

| Rune | Effect | Tag |
|---|---|---|
| Sixth Sense | Auto-tracks a nearby untracked enemy ward within 900. From level 11 it also reveals the ward for 10 s. Cooldown 250 s. | CLIENT `{d3bd04a2} 900, LevelThreshold 11, RevealDuration 10`; WIKI |
| Grisly Mementos | +1 memento per champion takedown, up to 18. +6 **trinket haste** each (108 at max). The summoner-haste variant applies only in modes without wards, so **not on SR**. | CLIENT `TrinketAH 6, MaxStacks 18` |
| Deep Ward | Wards in the enemy jungle get +1 HP. Stealth wards last +30–45 s and totem wards +45–150 s, both scaled by **average champion level**. From level 9, river wards also count. | CLIENT; WIKI |

These are catalogue entries only, until wards and vision are implemented. The validator must
still accept them as legal page entries, which leaves them as **no-ops with an explicit
"deferred" flag**.

### 4.8 Treasure Hunter 8135 / Relentless Hunter 8105 / Ultimate Hunter 8106 — row 3

**Bounty Hunter stacks.** You earn 1 stack per **unique** enemy champion you score a takedown on,
up to 5 (CLIENT; WIKI).

| Rune | Effect | Tag |
|---|---|---|
| Treasure Hunter | Each new Bounty Hunter stack gives `50 + 20*(stacks_before)` gold: 50, 70, 90, 110, 130 (total 450). | CLIENT `BaseGoldAmount 50, GoldGrowth 20, TOOLTIPMax 130` |
| Relentless Hunter | +8 **flat** bonus MS per stack while **out of combat** (5 s after the last combat event; §1.5). | CLIENT `OOCMS 8` |
| Ultimate Hunter | +6 ultimate haste, plus 5 per stack (31 at max). Applies only to R. | CLIENT `StartingUltAH 6, AdditionalUltAH 5` |

In a 1v1, only 1 unique enemy exists, so the maximum is **1 stack**.

---

## 5. Sorcery (8200)

### 5.1 Stormraider's Surge 8230 — keystone (replaced Phase Rush in 26.9; TOP PRIORITY)

Sources: CLIENT `StormraidersSurge.lua`: `DamageThreshold 0.25, Window 3, Duration 4,
HasteMax 0.48, RangedEffectiveness 0.75, SlowResist 0.5, Min/MaxCooldown 10/20`, plus
CooldownCalc `lin(20,10)`; WIKI T:4042210, P:4065358; RIOT 26.9 and 26.10.

The perk id and the DDragon key `PhaseRush` are unchanged. Treat `8230` as Stormraider's Surge,
and **delete any Phase Rush 3-hit logic**. The Phase Rush entries in the bin and wiki are
`removed`. Phase Rush's last values (V26.03: 25–50% MS, ranged 12.5–25%, 3 hits in 4 s, CD 30–10)
are historical only.

- **Trigger.** Post-mitigation damage that you deal to **one** enemy champion, summed over a
  sliding **3 s** window, reaches ≥ **25% of that champion's max HP**.
  - Every damage type and source owned by you counts, including procs, DoTs and summoner spells.
    INFERRED-M: the wiki gives no exclusions.
  - The threshold uses the target's max HP at the moment of the check.
- **Effect.** Gain **48%** bonus MS (ranged 36% = 48 × 0.75) and **50% slow resist** for
  **4 s**.
- **Cooldown.** `lin(20,10)`: 20 s at L1, 15.29 s at L9, 10 s at L18. It starts on trigger
  (INFERRED-M).
- **State.** Keep a per-enemy-champion ring buffer of `(t, dmg)` covering 3 s. A bucketed sum at
  the tick resolution is fine.

### 5.2 Summon Aery 8214 — keystone

Sources: CLIENT `DamageBase 10, DamageMax 50, DamageADRatio 0.1 (bonus), DamageAPRatio 0.05,
ShieldBase 20, ShieldMax 100, ShieldRatio 0.05 AP, ShieldRatioAD 0.1, ShieldDuration 2.5`;
WIKI T:4062237.

- **Damage.** Damaging basic attacks, abilities or item effects on an enemy champion send Aery.
  She arrives after **0.45 s** and deals `lin(10,50)` + 10% bonus AD + 5% AP adaptive damage.
  - It does not trigger on persistent proc damage.
  - Ignite's first tick is special-cased to trigger it.
- **Shield.** Buffing, healing or shielding an ally sends Aery to that ally (0.35 s travel). She
  shields for `lin(20,100)` + 10% bonus AD + 5% AP for 2.5 s.
- **Return trip.** She lingers about 2 s, then flies back. She cannot be re-sent until she
  returns.
  - Return speed starts at 200/300/600 at levels 1–5/6–10/11+, plus 10 per 51 units travelled.
  - When she comes within 200 units of the owner, her speed rises by +2000 per 51 units.
- **Fidelity.** A full model needs Aery as a projectile entity. A timer approximation is
  acceptable: unavailable for `0.45 + ~2 + dist/speed` (U-09).

### 5.3 Arcane Comet 8229 — keystone

Sources: CLIENT `DamageBase 15, DamageMax 100, ADRatio 0.1 (bonus), APRatio 0.05, MaxDamageAmp
1.0, MaxRange 750`, CooldownCalc `clamp(lin(20,8), 0.3, 20)`; WIKI T:4060639; RIOT 26.9.

- **Trigger.** Ability or pet damage to an enemy champion. 0-damage instances don't count.
- **Effect.** A comet flies to the target's position at trigger time and lands after about
  **0.8 s**. On landing it deals `(lin(15,100) + 0.1*bonusAD + 0.05*AP) * (1 + min(dist,750)/750)`
  of variable damage (§1.2) in a **140** radius.
  - `dist` is the distance travelled by the comet (caster → target).
  - The comet is a projectile, blockable by Yasuo W, Braum E and Samira W, and by spell shields.
- **Cooldown** `lin(20,8)`. Since 26.9 it has **no** cooldown refunds.

### 5.4 Deathfire Touch 8992 — keystone (added 26.9)

Sources: CLIENT `ADRatio 0.07, APRatio 0.025, TimeToAmp 3, DamageMultiplier 0.75, TicksPerSecond
2, Duration 4, AoEDuration 2, DotDuration 1`, plus `lin(3,12)` per second; WIKI T:4023190,
P:4064330; RIOT 26.9 hotfix, 26.10 and 26.11.

- **Trigger.** Ability or pet damage to an enemy champion applies a burn.
  - Each tick, every **0.5 s**, deals `(lin(3,12) + 0.07*bonusAD + 0.025*AP) / 2` **magic**
    damage (magic since 26.11).
  - After the burn has existed continuously on the target for **3 s**, ticks deal ×**1.75**.
- **Duration by source:** spell 4 s, area 2 s, persistent 1 s, persistent-area 1 s, pet 1 s.
  - A new application refreshes or overwrites only if its total duration ≥ the current burn's
    total duration, or if the current remaining duration is lower than the new one.
  - The stats used are snapshotted at application.
  - Bursts from multiple users stack independently.
- **Damage tags.** The burn is proc and periodic.

### 5.5 Row 1: Axiom Arcanist 8224 / Manaflow Band 8226 / Nimbus Cloak 8275

| Rune | Effect | Sources |
|---|---|---|
| **Axiom Arcanist** | The ultimate's damage, healing and shielding get **+12%** (AoE damage +8%). Self-healing is not amplified. A champion takedown reduces R's **current** cooldown by **7%**. Manaless champions get this in place of Manaflow. | CLIENT `DamageAmp 0.12, AOEAmp 0.08, UltimateRefundBase 7`; WIKI T:3948848; amps are V25.05 values |
| Manaflow Band | Ability damage to a champion, or certain debuffs, gives +25 **max** mana without raising current mana (15 s cooldown), up to +250. After that, restore 1% of missing mana every 5 s. **Manaless champions get Axiom instead.** | CLIENT `ManaIncrease 25, MaxManaIncrease 250, Cooldown 15, PercentManaRestore 0.01 / 5 s` |
| Nimbus Cloak | Casting a summoner spell grants **ghosting** and bonus MS that **decays over 2 s**. The MS starts at **15%** if the spell's (hasted) cooldown is < 100 s, **35%** if 100–250 s, and **45%** if > 250 s. **Teleport** always uses the top bracket. For Teleport and Hexflash, Nimbus triggers when the channel completes or is interrupted. The strongest of concurrent activations applies. | CLIENT `LowCDMSBoost 0.15, {1c32110c} 0.35, HighCDMSBoost 0.45, thresholds 100/250, Duration 2`; WIKI T:4023241 |

For Nimbus Cloak, which side of each threshold is inclusive is U-10. Example: Flash at 300 s
cooldown gets 45%, but Flash hasted by Cosmic Insight to 254 s also gets 45%, and Lucidity +
Cosmic (28 haste → 234 s) gets 35%.

### 5.6 Row 2: Transcendence 8210 / Celerity 8234 / Absolute Focus 8233

| Rune | Effect | Sources |
|---|---|---|
| Transcendence | Level 5: +5 AH. Level 8: +5 AH. Level 11: a champion takedown reduces the **current** cooldowns of basic abilities by 20%. | CLIENT `HasteBonus1/2 0.05`, displayed as 5; `LevelToTurnOn 5/8/11`, `KillCooldownRefund 0.2` |
| **Celerity** | +1% bonus MS, and **all other bonus MS is 7% more effective**: flat, additive-percent and multiplicative-percent buckets are each ×1.07. | CLIENT `PercentMS 0.01, PercentHasteMod 0.07`; WIKI T:3825860 |
| Absolute Focus | While `HP/maxHP > 0.70`, gain `lin(3,30)` AF (1.8–18 AD). | CLIENT `HealthPercent 0.7, Min/MaxAdaptive 3/30` |

**Celerity default (INFERRED-M):** multiply every non-Celerity bonus MS term by 1.07, then add
+1%. The wiki documents client quirks on top of this: the additive-percent part only applies when
other additive bonuses exceed 5%, and the multiplicative part oscillates between stat updates.
**Do not reproduce these quirks** (U-11).

### 5.7 Row 3: Scorch 8237 / Waterwalking 8232 / Gathering Storm 8236

| Rune | Effect | Sources |
|---|---|---|
| Scorch | The next ability damage to an enemy champion sets it on fire. **1 s later** it deals `lin(20,40)` bonus magic damage. Cooldown 10 s. Single target only. Proc, periodic, indirect. | CLIENT `Damage 20, DamageMax 40, DotDuration 1, BurnlockoutDuration 10` |
| Waterwalking | While in the **river**, +10 flat bonus MS and `lin(13,30)` AF. The MS decays over 1 s after leaving the river; the AF is lost immediately. Needs a river region mask; map11 region labels are pending (MODERN-011). | CLIENT `MovementSpeed 10, Min/MaxAdaptive 13/30` |
| Gathering Storm | `m = 1 + floor(t_game/600)`. AF granted = `4*m*(m-1)` AP-equivalent, which is 0 before 10:00, 8 at 10:00, 24 at 20:00 and 48 at 30:00. On AD champions it gives `0.6×` that: 4.8, 14.4, 28.8, and so on. | WIKI T:4011483; CLIENT `UpdateAfterMinutes 10, AdaptiveAP 8` |

The Gathering Storm bin field `AdaptiveAD 6` disagrees with the tooltip (AD 5/14/29/48…, which is
0.6× AP). **Use the 0.6× adaptive conversion.**

---

## 6. Resolve (8400)

### 6.1 Grasp of the Undying 8437 — keystone (TOP PRIORITY)

Sources: CLIENT `PercentHealthDamage 0.035, PercentHealthHeal 0.013, MaxHealthPerProc 5,
RangedPenaltyMod 0.4, RangedHealthPerProc 2, TriggerTime 4, Window 5, MeleeFlatHeal 0`;
WIKI T:4058228, P:4052352, Combat status 4058480 ("damage" combat system).

**Combat events for Grasp.** Grasp uses the damage-only combat system, so the rules are:

- **Count:** dealing damage to, or receiving damage from, **any enemy unit**: champion, minion,
  monster or turret. 0-damage instances count.
- **Do not count:** wards and plants, CC-only events, and hits blocked by a spell shield.
- **Lane farming charges Grasp.**

**Stack generation.** Each combat event at time `t` sets `gen_until = t + 3`. While
`now < gen_until`, gain 1 stack per **1.0 s**, up to **4** stacks.

- Implement as a float accumulator that only advances while generating.
- First stack 1 s after entering combat, 4th at 4 s (`TriggerTime 4`; tooltip "every 4s in
  combat").

**Primed state.** At 4 stacks, Grasp is primed.

- It stays primed while combat continues. It expires **5 s after the last combat event**
  (`Window 5`; V7.22 text "primed for 5 seconds, refreshing whenever you deal or receive
  damage").
- If not primed and `now ≥ gen_until`, partial stacks decay. **Default (INFERRED-M):** they all
  drop to 0 once `now - last_combat ≥ 5` (U-12).

**Proc.** The next basic attack **on-hit** against an **enemy champion** while primed consumes
all 4 stacks and:

1. Deals bonus **magic** damage = **3.5%** of your max HP (ranged 1.4%). It is proc damage: no
   spell effects, not modified by on-hit modifiers.
2. Heals you for **1.3%** of your max HP (ranged 0.52%). This heal is applied **before**
   permanent HP is added (INFERRED-L).
3. Permanently grants **+5 max HP** (ranged +2). This is flat **bonus** HP, which also raises
   current HP by 5 (INFERRED-M).

A blocked attack doesn't proc (V8.1).

- After proc, stacks are 0 and regenerate under the same rules. There is no extra cooldown, so in
  continuous combat it re-primes 4 s after the proc.

```
on_combat_event(t): last_combat = t; gen_until = max(gen_until, t + 3)
tick(dt):
  if stacks < 4 and now < gen_until: acc += dt; while acc >= 1 and stacks < 4: stacks += 1; acc -= 1
  if now - last_combat >= 5: stacks = 0; acc = 0          # primed or not
on_basic_on_hit(target is enemy champion) and stacks == 4:
  deal(magic, 0.035*maxHP*(0.4 if ranged else 1), proc=True)
  heal(0.013*maxHP*(0.4 if ranged else 1)); grasp_bonus_hp += 5 if melee else 2; stacks = 0; acc = 0
```

**Fixtures.**

- Melee at 1000 max HP: proc = 35 magic, heal 13, then max HP 1005.
- Melee at 2000 max HP after 30 procs (+150 HP): damage 70, heal 26.
- Ranged at 1000 max HP: 14 magic, 5.2 heal, +2 HP.

### 6.2 Aftershock 8439 — keystone

Sources: CLIENT `FlatResists 45, PercentBonusResist 0.75, BonusResistMin 80, BonusResistMax 150,
DelayBeforeBurst 2.5, StartingBaseDamage 25, MaxBaseDamage 120, HealthRatio 0.08 (bonus HP),
DamageRadius 350, Cooldown 20`; WIKI T:4022990, P:4023199; RIOT 26.11 (35 + 80% → 45 + 75%).

- **Trigger.** You apply an **immobilize** (§1.5) to an enemy champion that is not
  displacement-immune.
  - **Substituted by Grasp on champions without immobilize** (Garen).
  - **Cooldown** 20 s from the trigger.
- **Effect.** For **2.5 s**:
  - Bonus armor = `min(45 + 0.75*bonusArmor_at_trigger, lin(80,150))`.
  - Bonus MR is the same formula with MR.
  - The 75% part is snapshotted at trigger; the flat 45 is always applied (WIKI note).
- **Shockwave.** Then deal `lin(25,120) + 0.08*bonusHP` **magic** damage to enemy **champions
  and monsters** (not minions) within **350**, centered on you at burst time.
  - It is area damage tagged proc, and **applies spell effects** (WIKI P).
- **Fixtures:**
  - L1, bonus armor 0: +45 armor and +45 MR for 2.5 s, then 25 magic damage (+8% bonus HP).
  - L1, 60 bonus armor: `45 + 45 = 90`, capped at **80**.
  - L18, 200 bonus armor: `45 + 150 = 195`, capped at **150**.

### 6.3 Guardian 8465 — keystone

Sources: CLIENT `SnuggleRange 350, GuardDuration 2.5, ShieldBase/Max 40/150, APRatio 0.2,
HPRatio 0.06 (bonus HP), ShieldDuration 1.5, Cooldown lin(75,40), Threshold lin(50,165)`;
WIKI T:4022993; RIOT 26.11.

- **Guard.** You are Guarded while within 350 of an **allied champion**. Allies you target with
  unit-targeted abilities are Guarded for 2.5 s (wiki: 3 s).
- **Trigger.** You or a Guarded ally takes ≥ `lin(50,165)` post-mitigation damage within 2.5 s,
  or lethal damage, from an enemy champion, monster or turret.
- **Effect.** Both of you get a `lin(40,150) + 20% AP + 6% bonus HP` shield for **1.5 s**.
  - The wiki says 2 s; **use CLIENT 1.5** (U-13).
- **Cooldown** `lin(75,40)`, starting only on trigger.
- In a 1v1 without allied champions, Guardian never triggers.

### 6.4 Demolish 8446 — row 1 (reworked 26.1)

Sources: CLIENT `Demolish_SEASONAL.lua`: `BaseDamageMelee 85, HPRatioMelee 0.28, BaseDamageRanged
50, HPRatioRanged 0.2, CooldownSeconds 30, {97664ba4} 3, {8d810998} 8`; WIKI T:4058470,
P:4058471; RIOT 26.1.

- **Stacks.** Each basic attack on-hit against an enemy **turret** applies 1 Demolish stack on
  that turret, tracked per user.
  - Stacks **never expire**.
  - Multiple turrets can hold stacks at once.
- **The 3rd stack** on a turret consumes them and deals bonus **physical** damage:
  - melee: `85 + 0.28*maxHP`
  - ranged: `50 + 0.20*maxHP`
- **Damage tags.** Proc and basic. It is mitigated by turret armor and by turret damage
  modifiers, which are owned by `docs/modern/TOWERS.md`.
- **On consume:** global 30 s cooldown (no stacks are applied during it), all lingering stacks on
  other turrets are cleared, and that turret cannot be Demolished by **any** user for **3 s**
  (`{97664ba4} = 3` per WIKI).
- `{8d810998} = 8` is unexplained (U-14).
- **Wiki display bug.** The client shows the damage as a crit. **Do not apply crit.**
- **Removed in 26.1:** the old 0.5 s-per-stack proximity charge-up within 600 units, the 45 s
  game-time gate and the 100 + 35% HP damage. None of these apply.
- **Fixture.** Melee at 1500 max HP: 85 + 420 = **505** physical, before turret armor and
  modifiers, on the 3rd, 6th, … attack, respecting the 30 s cooldown.

### 6.5 Font of Life 8463 / Shield Bash 8401 — row 1

| Rune | Effect | Sources |
|---|---|---|
| Font of Life | Slowing or immobilizing an enemy champion heals you and the nearest, most-wounded allied champion within 1000 for `lin(10,50)` (ranged ×0.7). Cooldown 20 s. It still triggers (and goes on cooldown) at full HP. | CLIENT `Cooldown 20, RangedMod 0.7`, `BaseHeal lin(10,50)`; WIKI T:3996429 |
| Shield Bash | Whenever you gain a shield, your next basic attack against an enemy champion deals `lin(5,30) + 2.5% bonusHP + 15% × (shield's initial amount)` bonus **adaptive** damage, as proc on-hit. The empowerment lasts until 2 s after the shield expires. The largest current or recent shield replaces a smaller one. It triggers once per shield gained. | CLIENT `ProcBaseMin/Max 5/30, BonusHealthRatio 2.5, ShieldRatio 15, ProcDuration 2`; WIKI T:4044093 |

### 6.6 Conditioning 8429 / Second Wind 8444 / Bone Plating 8473 — row 2 (TOP PRIORITY)

**Conditioning.** Sources: CLIENT `MinutesRequired 12, ArmorBase 8, MRBase 8, ExtraResist 0.03`;
WIKI T:3825864.

- From game time **12:00**, gain +8 armor and +8 MR, and **total** armor and MR are increased by
  3%: `armor_total = (base + bonus + 8) * 1.03`.
- The 3% portion counts as neither purely base nor purely bonus (WIKI). **Default:** implement it
  as the `percent_bonus` multiplier in `stat_total`, applied on top of the flat 8 (INFERRED-M).
- Fixture: 40 base + 20 bonus armor at 12:00 gives `(40+20+8)*1.03 = 70.04`.

**Second Wind.** Sources: CLIENT `RegenSeconds 10, RegenPercentMax 0.04, RegenFlat 0`;
WIKI T:3960974, P:4012933.

- After **taking damage from an enemy champion** (§1.5; health damage > 0, not shield-absorbed,
  not invulnerable), gain health regen of **0.4% of current missing HP per second** for **10 s**.
  That is 4% missing over 10 s if missing HP stayed constant.
- It is recomputed continuously from **current** missing HP: WIKI "bonus health regeneration
  equal to 2% of missing health" per 5 s.
- Subsequent qualifying damage refreshes the 10 s.
- The flat +1.5 regen was removed in V25.21.
- Fixture: at 600/1000 HP, the first second regenerates 1.6 HP. With no further damage, 10 s
  gives `400*(1 - e^{-0.04}) ≈ 15.69` HP. The tick-discrete sum at 1 s steps is ≈ 15.67.

**Bone Plating.** Sources: CLIENT `BlockBase 30, BlockMax 60, BlockCount 3, BlockDuration 1.5,
Cooldown 55`; WIKI T:3985159, P:4032827; RIOT 26.19 Camille fix.

- After taking damage from an enemy champion (same qualifying rule as Second Wind), Bone Plating
  activates against **that champion** for **1.5 s**.
- The next **3** damage instances from that champion, at most 1 per `cast_instance_id`, each have
  their **post-mitigation** damage reduced by `lin(30,60)`, floored at 0. This applies to all
  types including true damage.
- The **triggering** instance is not reduced. If the triggering cast instance deals further
  instances later, those can be reduced.
- **Cooldown 55 s.** **Default (INFERRED-M):** it starts when the 1.5 s window ends or the 3rd
  block is used, whichever is first. It goes on cooldown even if nothing was blocked.
- Fixture: L1, three 50-damage autos within 1.5 s after the trigger: each deals 20. L9 (44.12
  block): 5.88 each.

### 6.7 Overgrowth 8451 / Revitalize 8453 / Unflinching 8242 — row 3

**Overgrowth.** Sources: CLIENT `Range 1400, UnitsPerTier 8, FlatHealthPerTier 3, ThresholdUnits
120, ThresholdMaxHealthRatio 0.035, {1663f8e7} 0.75, {ca029a2b} 2.0`; WIKI T:3825902, P:4002405;
26.6 hotfix.

- Count every **enemy minion or monster** that **dies within 1400** of you while you have direct
  line of sight to it.
  - Who kills it doesn't matter; your own sight is required, not shared vision.
  - Kills by Rift Herald count since 26.6.
  - Units still count while you are dead.
- Every 8 counted deaths give +3 max HP (bonus), with no cap. At **120** deaths (15 stacks), all
  max HP (base and bonus) is permanently ×**1.035**. The base portion counts as base HP.
- **LOS (INFERRED-M):** until vision ships, use a terrain-only line test against the navgrid
  (U-15).
- The meanings of `{1663f8e7}` and `{ca029a2b}` are unknown (U-15).
- Fixture: a lane wave of 6 minions dying within 1400 gives counter +6. After 10 minutes of about
  60 counted deaths: +21 HP (7 tiers).

**Revitalize.** Sources: CLIENT `HealShieldPower 0.05, HealthCutOff 40, ExtraAmp 10`;
WIKI T:3825912, P:3997189.

- +5% heal and shield power.
- Heals and shields **you cast** on targets below 40% HP, and heals and shields **you receive or
  self-cast** while you are below 40% HP, are ×1.10. This is multiplicative with HSP.
- Barrier/Heal fixture: Heal 80 × 1.05 = 84 above 40% HP, and 80 × 1.05 × 1.10 = 92.4 below.

**Unflinching.** Sources: CLIENT `ResistMin = ResistMax = 10, Duration 2`; WIKI T:3958701,
P:4053306; values changed in V25.05/V25.09.

- While afflicted by crowd control (any type except kinematics or disrupt; **slows count**) from
  an enemy champion, gain **+10 armor and +10 MR**.
- The bonus lingers **2 s** after the last such CC ends.
- For combined damage+CC instances, the damage resolves **before** the bonus.
- Fixture: Garen silenced by an enemy Garen Q for 1.5 s gets +10/+10 from t to t+3.5.

---

## 7. Inspiration (8300)

### 7.1 Keystones

**Glacial Augment 8351.** Sources: CLIENT `RayCount 3, SlowZoneLength 700, SlowZoneWidth 80,
SlowZoneDuration 3, SlowZoneSlowBase 20, SlowZoneSlowbADRatio 0.07, SlowZoneSlowAPRatio 0.06,
SlowZoneSlowHealShieldRatio 90, DmgReduction 0.15, CCCarryOverRatio 100, Cooldown 25`;
WIKI T:4033154.

- **Trigger.** Immobilizing an enemy champion.
- **Effect.** 3 rays run from the target toward you and other nearby enemy champions. Each makes
  a 700 × 80 zone lasting `3 + immobilize duration` s (the duration after tenacity).
  - Enemies in a zone are slowed by `20% + 7% per 100 bonus AD + 6% per 100 AP + 90% × HSP`.
  - They also deal 15% less damage to **your allies** (not to you).
- **Cooldown** 25 s.
- **Substitution.** Becomes First Strike on champions without immobilize.

**Unsealed Spellbook 8360 — basic support only.** Sources: CLIENT `ShardFirstMinutes 6,
ShardRechargeMinutes 4.5, ShardRechargeReductionSeconds 25, NumSummonersBeforeRepeat 3`;
WIKI T:3960876.

- The first swap is available at **6:00**.
- The swap cooldown is **270 s − 25 s per unique summoner spell swapped to**, with a minimum of
  120 s after 6 unique swaps.
- **Conditions for a swap:**
  - You have been out of combat for 5 s.
  - You are not channeling Teleport.
  - You cannot pick a spell already equipped.
  - You must use 3 other spells before repeating one.
- **Cooldowns after a swap:** a newly selected spell starts on a 5 s cooldown. Casting a swapped
  spell puts that slot on a 10 s lockout, and the spell is single-use.
- Smite damage increases after 2 swaps (deferred with jungle).
- **Implementation tier:** DEFERRED. Implement only as a legal-page no-op unless a scenario needs
  it (U-16).

**First Strike 8369.** Sources: CLIENT `GraceWindow 0.25, Duration 3, DamageAmp 0.07,
GoldProcBonus 10, GoldPercentBonus 0.5, Ranged 0.35, OOCTimer 10, cooldown lin(25,15) with
mScalePastDefaultMaxLevel=false`; WIKI T:4058238, P:4065516.

- **Trigger.** You must enter **champion combat by striking first**: your attack or ability hits
  an enemy champion within **0.25 s** of champion combat starting.
  - "Entering champion combat" requires no champion combat for **10 s** (`OOCTimer`;
    INFERRED-M).
  - If an enemy champion damages you first, First Strike goes on **full cooldown without
    activating**.
- **Effect.** Gain +10 gold and the First Strike buff for 3 s.
  - While the buff is active, every post-mitigation damage instance you deal to champions,
    including the triggering one, also deals **7% of that amount as separate bonus true damage**.
  - That bonus is proc and indirect. It does not inherit tags, and it arrives via a 0.4 s missile.
  - It is **not** a damage modifier; it does not stack with §1.4.
- **Gold.** You earn **50%** (ranged 35%) of the bonus true damage dealt as gold, awarded at the
  end.
- **Cooldown.** `lin(25,15)`, clamped at level 18, starting on activation.
- **Substitution.** Replaces Glacial Augment on champions without immobilize.

### 7.2 Row 1: Hextech Flashtraption 8306 / Magical Footwear 8304 / Cash Back 8321

| Rune | Effect | Sources |
|---|---|---|
| Hextech Flashtraption | See `SUMMONER_SPELLS.md` §Hexflash. While Flash's remaining cooldown is > 2 s, Flash is replaced by **Hexflash**. Hexflash: channel up to 2 s (MS set to 0), range growing from 200 toward 400. It can be released after ≥ 1 s and auto-releases at 2 s. It then blinks and grants 50% bonus MS for about 0.25 s. Cooldown 20 s after the channel. Releasing before 1 s or **entering champion combat** sets a 10 s cooldown. **If Flash is not equipped, this becomes Cash Back.** | CLIENT `ChannelDuration 2, MinimumChannelDuration 1, CooldownTime 20, ChampionCombatCooldown 10`; spell `SummonerFlashPerksHextechFlashtraptionV2` `mCastRangeGrowthMax 400, mCastRangeGrowthDuration 2`; WIKI T:4011828 |
| Magical Footwear | You cannot buy Boots or tier-2 boots until you receive free **Slightly Magical Boots** at `12:00 − 45 s × champion takedowns`. If the inventory is full, they arrive when a slot frees up. Your boots give **+10 flat MS** in addition to their own stats. | CLIENT `GiveBootsAtMinute 12, SecondsSoonerPerTakedown 45, AdditionalMovementSpeed 10` |
| Cash Back | On purchase of a **Legendary** item, refund **7.5%** of its total cost. Guardian items don't count. Selling or undoing the item removes the refund. | CLIENT `PercentRefund 0.075`; RIOT 26.3 |

### 7.3 Row 2: Triple Tonic 8313 / Time Warp Tonic 8352 / Biscuit Delivery 8345

**Triple Tonic** (DDragon key `PerfectTiming`). Sources: CLIENT `FirstElixirLevel 3,
SecondElixirLevel 6, ThirdElixirLevel 9`; items 2151/2152/2150; RIOT 26.3.

- Level 3: Elixir of Avarice. Consumed, it gives +5 true damage on-hit against minions for 60 s,
  then +60 gold.
- Level 6: Elixir of Force: +25 AF for 60 s.
- Level 9: Elixir of Skill: +1 skill point, auto-consumed if the inventory is full.
- If the inventory is full, items 1–2 wait for a free slot.

**Time Warp Tonic.** Sources: CLIENT `RestorationPercentage 0.4, BonusMS 0`; WIKI T:3825920.

- Consuming a Health Potion or Refillable Potion **instantly** heals 40% of the potion's total
  restoration (**48** and **40** respectively), in addition to the normal heal over time.
- If a potion is already active, the bonus triggers when the queued potion starts.
- Biscuits are not covered by the wiki list. **Default:** they don't trigger it (INFERRED-M,
  U-17).

**Biscuit Delivery.** Sources: CLIENT perk `BiscuitMinuteInterval 2, SwapOverMinute 6, PermanentHP
30`; item 2010 `FlatHeal 20, MaxHPMultiplier 0.015, MaxHealIncrease 100%, MinHPThreshold 0.3,
Duration 5`; WIKI T:4037523, Total Biscuit 3971313, Module:ItemData 3905887 (V25.22 1.5%).

- **Delivery.** Receive a **Total Biscuit of Everlasting Will** at **2:00, 4:00 and 6:00**. If
  the inventory is full, it arrives when a slot frees up.
- **Consume** (allowed at full HP). Over **5 s**, restore `(20 + 0.015*maxHP) * (1 + m)`, ticked
  every 0.5 s.
  - `m = clamp(missingHP_frac / 0.70, 0, 1)`, computed **at consumption**.
  - This is a heal (affected by Grievous Wounds).
- **Permanent HP.** Consuming **or selling** (5 g) a biscuit gives +30 bonus max HP permanently.
  The gain does **not** raise current HP (WIKI P:3980958).
- **Queue.** Biscuits queue: up to 3, each computed when it starts.
- **Fixtures.** At 1000 max HP and full HP: 35 over 5 s. At 300/1000 (70% missing): 70 over 5 s.
  At 650/1000: `35 × 1.5 = 52.5`.

### 7.4 Row 3: Cosmic Insight 8347 / Approach Velocity 8410 / Jack of All Trades 8316

| Rune | Effect | Sources |
|---|---|---|
| **Cosmic Insight** | **+18 summoner spell haste** and **+10 item haste** (item actives such as Tiamat/Stridebreaker, and trinkets). | CLIENT `SummonerHaste 18, ItemHaste 10` |
| **Approach Velocity** | +**7.5%** bonus MS while facing (within a 180° front arc) any *visible* enemy champion within **1000** that is immobilized, grounded or slowed (the source can be anyone). **15%**, with no range or vision requirement, toward enemy champions whose movement **you** impaired. Drowsy doesn't count (bug). The bonus updates about every 0.5 s and can linger about 0.25 s. | CLIENT `MovementSpeedPercentBonus 0.15, ActivationDistance 1000`; WIKI T:3980961, P:3980960 |
| Jack of All Trades | +1 ability haste per **unique stat type** granted by items. At 5 stacks gain **8 AF**; at 10 stacks gain **20 AF** total. The eligible stat list is in the wiki. | CLIENT `HastePerStack 1`; RIOT 26.15 (8/20); WIKI T:4046578 |

---

## 8. State the simulator must carry (per champion unless noted)

All state is fixed-shape `float32`/`int32`/`bool` arrays, so it is jit/vmap friendly. The page
itself is a static loadout: ids are resolved host-side into per-rune enable flags plus
parameters, so no dynamic dispatch happens in the tick.

| Group | Fields | Used by |
|---|---|---|
| Page (static) | `keystone_id`, `minor_ids[5]`, `shard_ids[3]`, `range_type` (melee/ranged; Jayce-like forms are dynamic), `adaptive_type`, substitution flags | all |
| Generic combat clocks | `last_combat_any_t` (damage system), `last_combat_modern_t`, `last_champion_combat_t`, `champion_combat_start_t`, `last_dmg_taken_from_champ_t[enemy]`, `last_dmg_dealt_to_champ_t[enemy]` | Grasp, PTA, Relentless, First Strike, Flashtraption, Second Wind, Bone Plating |
| Assist ledger | `assist_until_t[enemy_champion]` (15 s window) | takedowns: Triumph, Legend, Dark Harvest, Treasure, PoM, Transcendence, Axiom, Mementos, Magical Footwear |
| Stat mods (derived each tick) | `af_conq`, `af_absfocus`, `af_waterwalk`, `af_storm`, `af_jack`, `af_elixir`; `bonus_as_lt`, `bonus_as_hob`, `bonus_as_legend`; `basic_ah_legend`, `ah_transc`, `ult_ah`; `summoner_haste`, `item_haste`, `trinket_haste`; `ms_flat_*`, `ms_pct_*`, `ms_decay_*`; `armor_mr_aftershock`, `armor_mr_unflinching`, `cond_active`; `perm_bonus_hp` (Grasp, Biscuit, Overgrowth, Bloodline 85); `og_pct_active`; `life_steal_bloodline`; `slow_resist_storm`; `hsp` | stat composition |
| Conqueror | `conq_stacks:int`, `conq_expire_t`, `conq_level_lock:int`, `conq_cast_seen[K]` (cast_id, t) ring | §3.1 |
| PTA | `pta_target:int`, `pta_stacks`, `pta_stack_expire_t`, `pta_cd_until`, `pta_amp_active` | §3.2 |
| Lethal Tempo | `lt_stacks`, `lt_expire_t`, `lt_next_decay_t` | §3.3 |
| Fleet | `fleet_energy:float` (0–100) | §3.4 |
| Legend | `legend_points:int` | §3.8 |
| Electrocute / Stormraider | `elec_first_t[enemy]`, `elec_stacks[enemy]`, `elec_cast_seen[enemy][K]`, `elec_cd_until`; `storm_buf[enemy][Nbuckets]`, `storm_cd_until`, `storm_until` | §4.1, §5.1 |
| HoB / DH / ToB / Cheap Shot / Sudden Impact | `hob_stacks`, `hob_expire_t`, `hob_bonus_used`, `hob_cd_until`; `dh_souls`, `dh_cd_until`; `tob_cd_until`; `cs_cd_until`; `si_armed_until`, `si_cd_until` | §4 |
| Grasp | `grasp_stacks:int`, `grasp_acc:float`, `grasp_gen_until`, `grasp_procs:int` | §6.1 |
| Aftershock | `as_active_until`, `as_bonus_armor`, `as_bonus_mr`, `as_cd_until` | §6.2 |
| Demolish | `demo_stacks[turret]`, `demo_cd_until`, `demo_turret_lock_until[turret]` (**global across all users**) | §6.4 |
| Second Wind / Bone Plating / Unflinching | `sw_until`; `bp_source:int`, `bp_until`, `bp_blocks_left`, `bp_cast_seen[3]`, `bp_cd_until`; `unf_until` | §6.6–6.7 |
| Overgrowth | `og_count:int` | §6.7 |
| Bounty Hunter | `bounty_mask[enemy]:bool` | §4.8 |
| Inspiration | `fs_cd_until`, `fs_until`, `fs_bonus_dmg_acc`; `biscuits_delivered`, `biscuit_queue[3]`, `biscuit_regen_until`, `biscuit_rate`; `boots_due_t`; `elixir_flags` | §7 |
| Aery / Comet / Deathfire / Scorch (non-top) | `aery_free_t`; `comet_cd_until`, `comet_inflight[]`; `dft_burn[enemy]` (start, end, total_dur, snap stats); `scorch_cd_until` | §5 |

## 9. Events and hooks (ordering contract)

**Proposed fixed per-tick ordering.** Items and champion kits share these hooks.

1. **Stat composition.** Combine base growth, items, shards, rune stat mods, buffs, then
   adaptive resolution (§1.2), then `stat_total`.
2. **Timers.** Run expiries (Conqueror, LT decay, Grasp generation/decay, Second Wind regen,
   Bone Plating window end → cooldown start, Aftershock burst at `as_active_until`, biscuit
   delivery and regen ticks), then game-time thresholds (Conditioning 12:00, Magical Footwear,
   Gathering Storm, 2/4/6 min biscuits).
3. **Action/cast phase.** Summoner casts (Nimbus, Sudden Impact arm on Flash/TP/dash), attack
   windup start (HoB trigger), on-attack events (LT stacks/bolts, HoB consume, Fleet +6 energy).
4. **Damage pipeline**, for each damage event in emission order:
   1. Compute raw. Add on-hit bonuses (Grasp proc, PTA proc, Shield Bash, HoB true damage) as
      **separate proc events** queued after the main hit.
   2. Outgoing amps (§1.4): PTA, Coup de Grace, Cut Down, Last Stand, Axiom (R only); and the
      target's Exhaust if the source is exhausted.
   3. Received modifiers on the target (multiplicative), then resist mitigation.
   4. **Bone Plating flat reduction** on post-mitigation damage (target side), applied before
      shields.
   5. Shields absorb, then HP loss.
   6. **On-damage-dealt hooks** for the source, in this order:
      1. Conqueror stack/refresh, then heal
      2. Electrocute / Stormraider window update and trigger checks
      3. Dark Harvest
      4. Cheap Shot
      5. Sudden Impact
      6. Taste of Blood
      7. First Strike bonus
      8. Arcane Comet / Aery / Deathfire / Scorch triggers
      9. PoM
      10. combat clocks
   7. **On-damage-taken hooks** for the target: Second Wind, Bone Plating activation (for future
      instances), Guardian threshold, Grasp/combat clocks, Unflinching (if CC).
   8. Death resolution, then takedown credit, then Triumph (1 s delayed), Legend points, DH
      reset/souls, Bounty stacks, Transcendence/Axiom cooldown cuts, Absorb Life (killer), and
      Overgrowth counts (for all champions within 1400 with LOS).
5. **Movement integration.** MS composition including Celerity, then Fleet energy from distance,
   then Waterwalking region test.

**Ordering versus items:**

- Rune proc damage and item on-hit procs are separate events. Their mutual order within a tick
  is **unresolved** (U-18). **Default:** the main attack, then item on-hits, then rune on-hits.
- Damage amps from items (for example Liandry's) are summed with rune amps (§1.4).
- **Rune heal hooks run after lifesteal/omnivamp from the same event.**

## 10. Unresolved / needs live measurement

| ID | Question | Default chosen | Test scenario (Practice Tool, patch 26.19) |
|---|---|---|---|
| U-01 | Do linear rune values and shards extrapolate past level 18 (top quest level 19–20)? | Extrapolate, except First Strike cooldown | Complete the top quest and reach L20. Read the HP shard (190/200?) and Conqueror stat via the tooltip and the stat panel. |
| U-02 | Does pet or summoned-unit damage count as "damage from an enemy champion" (Second Wind, Bone Plating)? | Yes, if owned by the champion | Let a dummy champion's pet hit you; check whether Second Wind triggers. |
| U-03 | Conqueror level lock across a level-up mid-stack. | Reproduce the lock | Stack to 12 at L5 → level up → read AD. |
| U-04 | Does Conqueror's heal count damage absorbed by the enemy's shield? | Yes, it counts post-mitigation damage including shields | Hit a shielded dummy at 12 stacks. |
| U-05 | Lethal Tempo `PerfectlyTimedForgivance 0.25`. | Ignore | — |
| U-06 | Triumph heal or gold if you die within the 1 s delay. | Delivered | Trade a kill, then die immediately. |
| U-07 | Dark Harvest threshold: pre- or post-hit HP. | Post-hit | Hit a dummy from 55% → 45%. |
| U-08 | Taste of Blood: instant or over `RegenDuration 4`. | Instant | Log the HP timeline. |
| U-09 | Aery's flight model. | Timer approximation | — |
| U-10 | Nimbus Cloak threshold inclusivity, and decay curve shape (linear?). | `<100`, `≤250`, `>250`; linear decay | Cast spells at exact hasted cooldowns. |
| U-11 | Celerity's client quirks. | Clean ×1.07 | MS panel with Ghost plus boots. |
| U-12 | Grasp partial-stack decay and the exact primed expiry. | All lost 5 s after the last combat event | Hit a minion once, wait 3–6 s, observe stacks. |
| U-13 | Guardian shield duration: 1.5 s (client) vs 2 s (wiki). | 1.5 s | — (no 1v1 impact) |
| U-14 | Demolish `{8d810998}=8`. | Unused | Leave the turret for 10 s between hits; check the stacks persist. |
| U-15 | Overgrowth LOS model, and the `{1663f8e7}=0.75`, `{ca029a2b}=2` fields (champion-death or monster weights?). | Terrain LOS, weight 1 per unit | Count stacks after known kills, both near walls and away from them. |
| U-16 | Unsealed Spellbook full rules. | Deferred | — |
| U-17 | Time Warp Tonic on biscuits. | No | Consume a biscuit with TWT. |
| U-18 | Same-tick order of item procs vs rune procs (Conqueror stacking from Tiamat, Grasp vs Sheen-type procs). | Main hit → items → runes | Frame-step a replay. |
| U-19 | Damage amps additive vs multiplicative after 26.9 (wiki says additive, page "outdated"). | Additive within outgoing; Exhaust in the same sum | PTA + Coup de Grace on a dummy below 40%: compare 1.16 vs 1.1664. |
| U-20 | Second Wind discrete regen tick period. | Continuous at the sim tick | — |
| U-21 | Adaptive comparison includes rune AF? | Exclude | AP item plus AD item edge case. |
| U-22 | Bone Plating cooldown start (activation vs window end). | Window end | Measure time to the next activation. |

## 11. Diff vs current implementation

| ID | Location | Current | Spec | Severity |
|---|---|---|---|---|
| D-1 | `lanerl_jax/modern/items/loadout.py:211` | Move-speed shard `+0.02` | **+0.025** (CLIENT `StatGain1 2.5`) | HIGH (wrong number) |
| D-2 | `lanerl_jax/modern/items/loadout.py:218-219` | Tenacity/slow-resist shard `0.10` each | **0.15** each (CLIENT `StatGain 15`) | HIGH |
| D-3 | `lanerl_jax/modern/items/loadout.py:215-216` and test `modern/tests/test_stats_items.py:53-54` | HP scaling clamped at level 18 (360 for two shards at L20) | `10*L`, extrapolating to 200/400 at L20 (U-01 default) | MED (only matters at L19–20, top quest) |
| D-4 | `lanerl_jax/modern/items/loadout.py:189-193` docstring | "move speed +2%… tenacity/slow-resist +10%" | Fix to 2.5% and 15% | LOW |
| D-5 | `lanerl_jax/modern/core/stats.py:45-56` `adaptive_force_total(converts_to_ad=)` | Static loadout flag | Dynamic: bonus AD vs AP comparison with an adaptive-type tie-break (§1.2). This is equivalent for Garen; it differs for hybrid builds. | LOW for the top-lane AD milestone |
| D-6 | `lanerl_jax/modern/core/stats.py:115-128` `apply_damage_modifiers` | `(1+amp)*(1-attacker_reduction)*…` | Outgoing amps and Exhaust in **one additive sum** per the wiki (U-19). Callers must sum the amps before passing them. Target vulnerability/reduction stays multiplicative. | MED |
| D-7 | `lanerl_jax/modern/items/loadout.py:164-174` `validate_rune_page` | Rejects every non-empty page | Implement §2.2: legality, substitution and the deferred-vision flag. Keep fail-closed for runes whose kernels are not yet implemented. | Expected (not a bug) |
| D-8 | `lanerl_jax/modern/items/loadout.py:177-182` `STAT_SHARD_OPTIONS` | String keys | OK as a set. Add id mapping 5008/5005/5007/5010/5001/5011/5013 so DDragon pages validate. | LOW |
| D-9 | `lanerl_jax/modern/items/loadout.py:22-38` `ItemStats` | Has `ability_haste` only | Also needs `basic_ability_haste`, `ultimate_haste`, `summoner_haste`, `item_haste`, `trinket_haste`, `heal_shield_power`, `life_steal` (present) and `bonus_as` vs AS ratio, as separate haste buckets (WIKI Haste). Cosmic Insight and Lucidity grant **summoner** haste, not AH. | MED |
| D-10 | `lanerl_jax/modern/data/26.19/runes.json` | Names come from DDragon (8230 is named Stormraider's Surge, key `PhaseRush`) | The kernel registry must key on **id**, not `key`. A Phase Rush implementation keyed on `PhaseRush` would be wrong. | MED |
| D-11 | No kernels exist for any rune | — | Implement first: Conqueror, Grasp, Second Wind, Bone Plating, Conditioning, Overgrowth, Unflinching, Triumph, Legend: Haste/Alacrity, Last Stand/Coup/Cut Down, PTA, Fleet, Lethal Tempo, Stormraider's, Demolish, Biscuits, Approach Velocity, Electrocute, ToB, Sudden Impact. | — |

## 12. Corrections to `docs/MODERN_PATCH_DELTA.md` §7 and §11 (26.18-pinned)

| Item | MODERN_PATCH_DELTA says | 26.19 verified |
|---|---|---|
| §7.2 Move speed shard | 2.5% | 2.5% ✓. The **code** has 2%, see D-1. |
| §7.2 Health scaling | "+10 to +200 (by level)" | 10–180 at L1–18 (CLIENT). 200 only at L20 if extrapolation holds (U-01). |
| §7.2 Tenacity shard | 15% | ✓. The code has 10% (D-2). |
| §7.4 Conqueror "1.08–2.56 AD per stack, ~13–31 AD" | — | **1.08–2.40 AD per stack** (1.8–4 AF), **12.96–28.8 AD** at 12 stacks (CLIENT; V13.20 values). |
| §7.4 Legend: Haste "up to 15" stacks | — | **Max 10 stacks** × 1.5 = 15 **basic** AH (not ultimate). |
| §7.4 Axiom Arcanist "+14% ult damage (9% AoE)" | — | **12% / 8%** since V25.05. |
| §7.4 Grasp | 3.5% / 1.3% / +5 | ✓ (ranged 40% since V25.12). |
| §7.3 "Why not Aftershock" | — | Correct, and stronger: Aftershock is **auto-substituted to Grasp** for champions without immobilize. |
| Phase Rush | (not covered) | **Removed in 26.9**, replaced by Stormraider's Surge (same id 8230). |

## 13. Test fixtures (numeric; melee unless noted; `lin` values exact to 1e-4)

| # | Setup | Expected |
|---|---|---|
| F-1 | Shards A/A/HS, L1 | +10.8 bonus AD, +10 HP |
| F-2 | Shards A/A/HS, L18 | +10.8 AD, +180 HP |
| F-3 | Shards AS/MS/Tenacity | +10% AS, +2.5% MS, +15% tenacity, +15% slow resist |
| F-4 | Conqueror stacks at L1/L6/L9/L13/L18, 12 stacks | AF per stack 1.8/2.4471/2.8353/3.3529/4.0. Bonus AD at 12 stacks 12.96/17.6188/20.4141/24.1412/28.8 |
| F-5 | Conqueror: 6 melee autos on a champion at 1 s intervals | Stacks 2, 4, …, 12 at the 6th. Heal on the 6th = 8% of its post-mitigation damage (INFERRED, U-04). No hits for 5 s → 0 stacks. |
| F-6 | Conqueror: a Garen E-like DoT, 1 tick per 0.5 s, special-cased | +2 per tick until 12 |
| F-7 | Conqueror: a non-special DoT, one cast instance, 6 s | +2 at t=0, +2 at t=4 |
| F-8 | PTA L1/L9/L18 | Proc 40 / 96.4706 / 160 adaptive. Amp ×1.08 on subsequent damage; the 3rd auto itself is unamplified. |
| F-9 | PTA: autos on A, A, B | Stacks on A cleared; B at 1 |
| F-10 | Lethal Tempo, 6 stacks, +10% shard AS, L1 | Bonus AS from LT 36%. Bolt `9*(1+0.46) = 13.14`. |
| F-11 | Fleet heal (0 bonus AD), L1/L6/L9/L13/L18 | 15 / 48.691 / 72.488 / 108.397 / 160. Versus a minion, ×0.15. |
| F-12 | Fleet energy: walk 2400 units | 100 → energized. Attack-only: 17 attacks (6×17 = 102). |
| F-13 | Grasp at 1000 HP, 4 s after entering combat | Next auto on a champion: +35 magic, heal 13, max HP 1005 |
| F-14 | Grasp ranged at 1000 HP | 14 / 5.2 / +2 |
| F-15 | Second Wind at 600/1000 HP | Regen 1.6 HP/s initially. Total over 10 s ≈ 15.69 (continuous). |
| F-16 | Bone Plating L1: trigger hit 100, then 3 autos of 50 within 1.5 s | Take 100, 20, 20, 20. A 4th auto is unreduced. |
| F-17 | Bone Plating L18, block vs 50 | 0 (floored) |
| F-18 | Conditioning at 11:59.9 vs 12:00, base 40 + bonus 20 armor | 60 → 70.04 |
| F-19 | Overgrowth after 119 vs 120 counted deaths, base 1000 + bonus 500 HP (+42 Overgrowth flat) | `1000+500+42=1542` → at 120 (+3 → 1545) × 1.035 = 1599.075 |
| F-20 | Triumph at 400/1000 HP takedown | After 1 s: +25 + 30 = +55 HP, +20 gold |
| F-21 | Legend: Haste after 49 minion kills + 1 takedown | `49*4 + 100 = 296` points → 2 stacks → +3 basic AH |
| F-22 | Last Stand at HP 59%/45%/30%/10% | +5.2% / +8.0% / +11% / +11% |
| F-23 | Coup de Grace: target at 41% vs 39% | ×1.00 vs ×1.08 |
| F-24 | Cut Down: target at 61% vs 60% | ×1.08 vs ×1.00 |
| F-25 | PTA + Coup on a target <40% (U-19 default) | ×1.16 total |
| F-26 | Electrocute L1/L9/L18, 0 bonus AD/AP | 70 / 150 / 240, magic (zero contributions → magic) |
| F-27 | Taste of Blood L1/L9/L18, 0 bonus AD | 16 / 27.2941 / 40; cooldown 20 s; no trigger at full HP |
| F-28 | Sudden Impact L1/L9/L18 | 20 / 48.2353 / 80 true, within 4 s after Flash |
| F-29 | Stormraider's at L9 vs a 1000 max HP target: 260 post-mitigation damage within 3 s | Trigger: +48% MS, 50% slow resist for 4 s, cooldown 15.294 s. 240 damage within 3 s → no trigger. |
| F-30 | Aftershock L1 / bonus armor 60 / L18 with bonus armor 200 | +45 / +80 (cap) / +150 (cap); burst 25 / — / 120 (+8% bonus HP) |
| F-31 | Demolish melee at 1500 HP | 505 physical (pre-armor) on the 3rd turret hit; the next consume ≥ 30 s later |
| F-32 | Biscuit at 1000 max HP: at 1000 / 650 / 300 HP | 35 / 52.5 / 70 over 5 s; +30 max HP each |
| F-33 | Absorb Life L1/L5/L6/L11/L18 | 1 / 2 / 3 / 9 / 23 per kill |
| F-34 | HoB L1 | 3 empowered autos at +90% AS, +2 true each; cooldown 10 s after the end |
| F-35 | Page legality | Garen page `[8010; 9111, 9105, 8299; 8224, 8234]`: shards valid. Aftershock on Garen → becomes Grasp. Two Sorcery minors from the same row → reject. A secondary keystone → reject. |
