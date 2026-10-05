# RUNES_IMPLEMENTATION.md — 26.19 runes, shards and modifier ordering as implemented

**Status (2026-10-01):** every selectable Summoner's Rift rune of client build 16.19.8230722 (62 runes) and
all 7 stat shards are classified. 60 runes are implemented as pure, fixed-shape JAX. 2 vision runes are
deferred. The shared stat-modifier ordering (STAT.00–70) and damage-modifier ordering (DMG.40/60/70) are
implemented in one place each. Items and runes run together in one reference tick, `combat.combat_tick`.
The world tick (`world/config.py`) does **not** call it yet.

The rules this implements are in [RUNES.md](RUNES.md) and [DAMAGE_AND_STATS.md](DAMAGE_AND_STATS.md).
The item side is in [ITEMS_IMPLEMENTATION.md](ITEMS_IMPLEMENTATION.md).

## Layout

| File | Role |
|---|---|
| `lanerl_jax/modern/data/build_runes.py` | Host tool. Rebuilds `runes_client.json` from the cached 16.19 perk bin. It writes the PerkStyles (rows from `mSlots`, allowed secondaries, default shard sets), the 3 shard slots, and each perk's CLASSIC `mEffectAmount` and `mCalculations`. Mode overrides are dropped. Hash-named slots such as `{3ecd47e5}` (Legend: Haste) are resolved through `mPerkId`. It also records retired perks and source sha256s. |
| `lanerl_jax/modern/data/26.19/runes_client.json` | Pinned table: 62 runes, 7 shards, 5 styles. |
| `lanerl_jax/modern/runes/catalog.py` | `rune_catalog()`, `ea()`, and the level primitives: `lin` (extrapolates past 18 unless `scale_past_18=False`), `lin_growth` (stat-progression fraction), `breakpoints`, `level_table`. Also `RunePage`, `validate_page` (§2.2 rules 1–3, rejects illegal pages), `substitute` and `prepare_page` (client game-start swaps), `page_counts` (the (C, R) matrix kernels close over) and `has_rune`. |
| `lanerl_jax/modern/runes/effects/` | `core.py` holds the contract: `RuneEvents`, `RuneOutputs`, `CombatClocks` and helpers. `__init__.py` holds the registry, dispatch and `coverage_report()`. There is one module per tree: `precision`, `domination`, `sorcery`, `resolve`, `inspiration`. |
| `lanerl_jax/modern/core/stat_pipeline.py` | STAT.* ordering: `compose` (growth as base except AS, flat → % → multiplicative, adaptive force at STAT.50, caps), `move_speed` (Celerity amp, strongest slow × slow resist, soft caps), `attack_speed` (cap 3.003, Hail of Blades lift), `windup` (champion modifier), `cooldown`, `rescale_cooldown`, `tenacity_total`, `cc_duration` (0.3 s floor), `sync_max_health`, and `champion_base(names)` from the pinned champion records. |
| `lanerl_jax/modern/core/stats.py` | Negative resist from reduction now survives % reduction and penetration (DAMAGE D1). `apply_damage_modifiers` puts dealt amps and Exhaust in **one additive sum** and multiplies the received modifiers; true damage ignores DR and Exhaust but keeps amps (D4). `resolve_adaptive` / `adaptive_is_ad` make the dynamic adaptive choice (D5). |
| `lanerl_jax/modern/core/damage.py` | Packets gained `cast_id`, where 0 means each packet is its own instance (RUNES §1.3), and `block`, a DMG.70 per-packet flat reduction on every damage type (Bone Plating). New tags: `TAG_INDIRECT`, `TAG_BURN`, etc. New properties: `PROP_ULTIMATE` (Axiom), `PROP_SUMMONER` (Ignite). Rune provenance is `item = -perk_id`. |
| `lanerl_jax/modern/combat.py` | `CombatState`, `combat_tick`: the single reference tick for items and runes (order below). |
| `lanerl_jax/modern/items/loadout.py` | `stat_shard_stats` takes shard names or perk ids and reads its values from client data. With `adaptive_to_ad=None` it leaves the adaptive force unresolved. `validate_rune_page(page, traits)` validates a page and applies the substitutions; `None` or an empty page selects the no-runes ruleset. |

`ItemStats` gained the rune and shard buckets: `adaptive_force` (unresolved), `item_haste`, `trinket_haste`,
`percent_armor`, `percent_magic_resist`, `percent_health`, `bonus_ms_amp`, `silent_health` and
`attack_speed_cap_lift`.

The rune-granted items are now real catalog items, so the catalog has 220 items:
- Total Biscuit 2010 and Elixirs of Skill, Avarice and Force (2150–2152) are implemented in
  `items.effects.consumables`.
- Slightly Magical Footwear 2422 is stats-only.

## Coverage

`runes.effects.coverage_report()` raises unless each of the 69 perks is exactly one of the following:
- implemented by a tree module;
- **DEFERRED**: Sixth Sense 8137 and Deep Ward 8141, both vision (MODERN-009);
- **STATIC**: the 7 shards, applied host-side by `stat_shard_stats`.

| Module | Runes |
|---|---|
| precision | 13 |
| domination | 10 (+2 deferred) |
| sorcery | 13 |
| resolve | 12 |
| inspiration | 12 |

Each module's `COVERAGE` text states what it implements and any limitation. Kernels key on perk **id**, so 8230 is
Stormraider's Surge with no Phase Rush logic (RUNES D-10).

## Tick order (`combat_tick`)

1. **STAT.50.** Item and rune dynamic stats are computed from the pre-dynamic `Ctx`. All adaptive force is split
   once, comparing the pre-adaptive bonus AD and AP (U-21). Rune formulas read the post-STAT.50
   `ev.bonus_ad` / `ev.ap` / `ev.bonus_attack_speed`.
2. **STAT.70.** Dynamic max-HP changes (Grasp, Overgrowth, Bloodline, elixirs, biscuits) sync to current HP. Gains
   heal by the delta, except `silent_health`. Losses only clamp.
3. **Action phase.** Each hook runs for items first, then runes: on_cast, on_attack, on_hit, then the item on-hit
   re-applications, then the rune on-hits, then item actives, periodic, and on_cc. This is RUNES U-18:
   main hit → items → runes.
4. **Defense and offense.** Fold the item holder defense, item and rune debuffs, penetration, and dynamic
   armor/MR with their STAT.40 multipliers (Conditioning, Aftershock, Unflinching).
5. **Main resolution.** Compact the packets carried from the previous tick, the world packets and the emitted
   packets. Add the DMG.40 amps (items + runes summed: PTA, Coup de Grace, Cut Down, Last Stand, Axiom)
   and the DMG.70 `packet_block` (Bone Plating), then resolve.
6. **Combat clocks**, then on_damage (items, then runes). The clocks follow RUNES §1.5: `last_combat`,
   `last_champion_combat`, `last_hit_by_champion`, `champion_combat_start`, and `struck_first` (a new episode
   after 10 s without champion combat).
7. **Follow-up pass** for trigger damage. Its own trigger packets are **carried** into the next tick, up to 64
   (`CARRY_CAPACITY`), instead of being dropped.
8. **End of tick.** on_takedown; then heals and shields (HSP, incoming, Revitalize via `heal_mult`, GW);
   then rune `post_tick` (with `ev.shield_gained`); then item on_shop; then rune `outputs`.

`combat.item_tick` is `combat_tick` with an empty page, no max-HP sync and no carry.

## What the world integrator must still do

**Carry the tick state forward.** Keep `CombatState`, and carry `out.max_hp` and `out.hp`. Don't add
`dynamic_stats.health` to max HP: `combat_tick` already does that sync. Do add `dynamic_stats` (AD/AP with
adaptive force already resolved, AS, MS buckets, haste) to the champion's own stat reads, for example
through `core.stat_pipeline.compose`.

**Supply `RuneEvents`** (`core.rune_events` builds quiet defaults):
- **Combat:** attack windup start, cancel and reset; cast instance ids; CC with durations; the impaired masks;
  `holder_cc_from_champion`.
- **Movement and spells:** summoner casts with their hasted cooldown; blinks; the Flash cooldown and Hexflash
  requests.
- **Kills and vision:** kills; deaths with `sight`; `visible`.
- **Shop and items:** purchases and sales; `granted` acknowledgements.
- **World state:** `game_time`; `in_river` (river mask pending, MODERN-011); `is_turret`; `uses_energy`;
  `adaptive_physical`.

**Apply `RuneOutputs`:**
- inventory: `grant_item` until acknowledged, and `forbid_purchase` passed into the shop;
- leveling: `skill_points`;
- cooldowns: `basic_cd_refund` / `ult_cd_refund` cut the *current* cooldowns;
- movement: `move_locked` / `blink` / `blink_range` (Hexflash) and `ghosted` (Nimbus);
- `spellbook_swap_ready`, combined with `inspiration.spellbook_can_select`.

`Effects.gold` and `Effects.mana` carry rune gold and mana. Manaflow's max-mana gain arrives as
`dynamic_stats.mana` and must not raise current mana.

**Packet conventions:**
- Pet damage must use `src` = the owner champion, plus `TAG_PET` (U-02).
- R packets carry `PROP_ULTIMATE`.
- Ignite carries `PROP_SUMMONER`.
- Ability packets carry `TAG_ACTIVE_SPELL` and a shared `cast_id` per cast.

**Apply the substitutions.** Call `prepare_page(page, traits)` per champion. The traits for Garen and Jax are in
`CHAMPION_TRAITS`. Garen: Aftershock → Grasp, Glacial → First Strike, PoM → Triumph, Manaflow → Axiom.

## Known gaps (each is also in the module `COVERAGE` text)

- **Ally-targeted parts** (there are no allies in a 1v1): Aery's shield, Guardian's ally shields (allies must be
  holders), Font of Life's ally heal (holders only), Glacial's ally damage reduction, and Revitalize on heals
  cast at low-HP allies.
- **Axiom** amplifies ultimate damage only; ultimate heals and shields are champion-owned.
- **Event gaps:**
  - Shield Bash assumes a 2 s shield life, because no shield duration event exists.
  - Hexflash sees only Cosmic Insight's summoner haste.
  - Relentless Hunter uses the damage combat clock, not the "modern" one that counts CC.
  - Cheap Shot's exception for same-instance on-hit CC is not modelled, because CC carries no cast id.
  - Grasp cannot tell a blocked attack from a landed one.
  - Guardian has no lethal-damage trigger.
- **Same-pass limits:** Bone Plating does not reduce packets later in the pass that triggered it. Coup de
  Grace and Cut Down read the target's HP at tick start.
- **Fixed queue sizes:** Conqueror and Electrocute keep 8 cast instances. Triumph and PoM keep 4 delayed restores.
  First Strike keeps 24 delayed packets. If a queue overflows, the oldest entry is dropped or re-stacked.
- **Deferred:** Unsealed Spellbook implements only the swap timing (U-16). Vision runes are deferred.
- **Defaults for measurement questions:** RUNES §10 U-01…U-22 use the documented defaults; each module
  docstring cites the U-id it relies on. Choices not covered by a U-id are listed in the module docstrings.
  Examples: the Hail of Blades cancelled-windup lockout of 1 s; Comet hits champions only; Cash Back's
  Legendary means catalog `epicness == 5`; the Jack of All Trades stat list.

## Doc corrections made while implementing

- RUNES.md §2.1 said "63 selectable runes". The client data has **62** (17 keystones + 45 minors).
- DAMAGE_AND_STATS.md F19 had an arithmetic error: (340 + 45)·1.35 = 519.75, which soft-caps to **489.875**.
- Guardian's client `ThresholdCalc` (40–150) disagrees with its data values `ThresholdMin/Max` (50–165).
  The module uses the data values, as RUNES §6.3 does.

## Tests

Test files:
- `test_rune_framework.py`: catalog, legality and substitution F-35, level primitives F-4/F-11/F-33, shards
  F-1–F-3, resist order, additive amps F-25, DMG.70 block, adaptive choice, and composition fixtures
  DAMAGE F12–F22.
- `test_combat.py`: max-HP sync including silent HP, combat clocks, all perks classified, and every rune at
  once through the jitted `combat_tick`.
- `test_runes_{precision,domination,sorcery,resolve,inspiration}.py`: 41 + 23 + 22 + 23 + 21 tests
  covering the RUNES §13 fixtures. The Resolve, Inspiration and combat files include end-to-end
  `combat_tick` runs (Bone Plating F-16, Grasp max HP, First Strike).

The command and the result of the full modern run are recorded in ledger row MODERN-014.
