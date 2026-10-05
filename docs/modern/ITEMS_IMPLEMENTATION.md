# ITEMS_IMPLEMENTATION.md — 26.19 item system as implemented

**Update (2026-10-01, runes pass):** items now run inside `combat.combat_tick` together with runes
(see [RUNES_IMPLEMENTATION.md](RUNES_IMPLEMENTATION.md)); `combat.item_tick` is that tick with an empty rune
page. Packets now carry `cast_id`; the catalog has 220 items (adds the rune-granted Biscuit 2010, Elixirs
2150–2152 in `consumables`, and stats-only Slightly Magical Footwear 2422); dynamic armor/MR and max HP are
folded/synced by `combat_tick`.

**Status (2026-10-01):** every Summoner's Rift item of client build 16.19.8230722 is classified, and every
in-scope effect is implemented as pure, fixed-shape JAX with focused tests. The world tick
(`world/config.py`) does **not** dispatch items yet. Until it does, `items.loadout.item_loadout_stats` rejects
any loadout containing an item with behaviour beyond its stat line, so nothing runs silently as stats only.

The rules this implements are in [ITEMS.md](ITEMS.md) (per-item detail in [ITEMS_CATALOG.md](ITEMS_CATALOG.md)).
Global formulas are in [DAMAGE_AND_STATS.md](DAMAGE_AND_STATS.md).

## Layout

| File | Role |
|---|---|
| `lanerl_jax/modern/data/build_items.py` | Host tool. Rebuilds `items_client.json` from the cached 16.19 client bins: CLASSIC item lists, in-store items, transforms and quest components; stats from client fields; groups with max-ownable; recipes; data values; calculations; spells; effect amounts. Records source sha256s. |
| `lanerl_jax/modern/data/26.19/items_client.json` | Pinned table: 215 items, 157 groups. |
| `lanerl_jax/modern/items/catalog.py` | `catalog()`, `ItemStats` (31 bonus-stat fields), stacking rules (tenacity, slow resist and %pen multiply), `lerp_level` (extrapolates past 18, README X-1), `level_bp`. |
| `lanerl_jax/modern/items/inventory.py` | 6 slots + trinket. `buy` (recursive recipe consumption, cost = total − owned components, group limits after consumption, stacks, level, ranged-only, purchase-buff gates, Elixir 5 s group cooldown, shop circle r=1000 or dead), `sell` (client sell modifiers), `replace_item`, `consume_one`, `inventory_stats`. |
| `lanerl_jax/modern/core/damage.py` | Shared DMG/HEAL/SHIELD pipeline: packets with client damage tags, source amps added together and target modifiers multiplied, unit-class ratios (minion→champion 0.55, →structure 0.60), resist order that keeps negative resist, Plating/Randuin's/Warden's slots, the Lifeline check before shields, typed decaying shields, Death's Dance storage (physical/magic only), spell shield, executes, life steal/omnivamp split (33% modified ratio), heal modifiers with 40% Grievous Wounds. |
| `lanerl_jax/modern/items/effects/` | `core.py` (contract), `__init__.py` (registry, dispatch, coverage), `runtime.py` (defense/offense folding, packet resolution, effect application; the items-only reference tick is `combat.item_tick`), one module per family: `starters`, `consumables`, `spellblade`, `hydra`, `fighter`, `defense`, `mage`, `marksman`, `support`, `boots`. |
| `lanerl_jax/modern/items/loadout.py` | Loadout stats and gate, stat shards (2.5% move speed, 15% tenacity/slow resist, health-scaling shard 10·level), rune catalog gate. |

## Coverage

`items.effects.coverage_report()` raises unless each of the 215 catalog items is exactly one of the following:

- **Implemented by a module.** Each module's `COVERAGE` entry gives the item's line and any limitation.
- **`STATS_ONLY`** (39). These items have no client data values, calculations or spell.
- **`DEFERRED`** (10). Jungle pets, wards and trinkets (vision), and champion-locked items, per MODERN-009.

Module counts:

| Module | Items |
|---|---|
| defense | 29 |
| mage | 25 |
| marksman | 23 |
| fighter | 22 |
| support | 19 |
| starters | 16 |
| boots | 14 |
| spellblade | 8 |
| consumables | 5 |
| hydra | 5 |

Item actives other than the Tiamat line and Stridebreaker remain deferred under MODERN-009; their `COVERAGE` text says so. Potions and elixirs are implemented as consumable actives.

## Hook contract and tick order

Hooks are listed in `core.py`. `combat.item_tick` is the reference order, matching the README hook crosswalk:

1. `dynamic_stats` (STAT.50, read from pre-dynamic `Ctx`).
2. `on_cast`.
3. `on_attack`.
4. `on_hit`.
5. On-hit re-application: Dusk and Dawn, Guinsoo's Phantom Hit, and up to 2 Runaan's/Statikk extra targets, each with raw 0.
6. `active`. Called every tick so cast times resolve.
7. `periodic`.
8. `on_cc`.
9. Fold `holder_defense` + `target_debuffs` + item penetration into `Defense`/`Offense`. Lifeline shields are scaled once by heal/shield power and incoming-heal bonus here.
10. `packet_amp`.
11. `resolve`.
12. `on_damage`.
13. One follow-up resolve for trigger damage (executes, Shadowflame, Voltaic, Zaz'Zak's), then `on_damage` on that pass. Second-generation emissions are dropped.
14. `on_takedown`.
15. `apply_effects`: heals with heal/shield power, incoming bonus and Grievous Wounds; shields; slows (strongest wins); Grievous Wounds.
16. `on_shop`.

## Tests

`ops/login_capped.sh 10G 4 .venv-jax/bin/python -m pytest -q lanerl_jax/modern/tests/test_item_framework.py lanerl_jax/modern/tests/test_item_runtime.py lanerl_jax/modern/tests/test_items_*.py lanerl_jax/modern/tests/test_stats_items.py lanerl_jax/modern/tests/test_towers.py lanerl_jax/modern/tests/test_minions.py`
gave **195 passed in 10 min** on 2026-10-01.

The files cover:
- catalog pinning;
- shop fixtures F12–F14 and F17;
- pipeline order and edge cases;
- the Hydra line (F1–F3, F5, F6);
- each family module's client values, timers and holder isolation;
- an end-to-end `item_tick` duel under JIT.

## Cost

Packets are compacted to fixed capacities before resolution: 512 per tick, 256 for the follow-up pass.
Overflow is reported in `ItemTickOut.packet_overflow` and must stay 0. Only packets on champions or
stateful units (shields, Lifeline, Death's Dance) go through the sequential scan, capped at 64. All other
packets resolve exactly in parallel, using per-target running sums in emission order.

`ops/modern/items_bench.py` measured these on the login CPU with 2 cores, for two champions with six items
each and 66 units:
- compile: about 18 s;
- single world: 1.7 ms per tick;
- vmapped ×64: 0.56 ms per environment per tick.

Packet compaction reduced the single-world time from 91 ms. GPU cost has not been measured; it belongs to
the world-integration performance gate (MODERN-007).

## What the world integrator must still do

These are item outputs the world has to apply. Each is exposed by a documented function or field.

- **Inventory changes.**
  - Transforms: `starters.pending_transforms` → `items.inventory.replace_item`.
  - Consumption: `state.consumables.consume_row` → `consume_one`.
  - Support line: World Atlas → Runic Compass. 3866 is not in the client catalog, so it needs a decision.
- **Champion state.**
  - `Effects.mana`, `gold`, `attack_reset`, and `revive` (Guardian Angel: cancel the death, 4 s stasis, revive HP).
  - `ActiveOut` cast lockout and movement permission.
  - Carry `Resolved.max_hp` forward (Protoplasm) and raise current HP with any `ItemStats.health` gain (Sundered Sky overheal).
- **Module helpers the tick does not apply automatically.**
  - Navori `basic_cooldown_scale` and Axiom Arc `ult_refund_fraction` (cooldowns are champion-owned).
  - `marksman.attack_range_bonus` (RFC, Hexoptics).
  - `marksman.shield_reaver` (Serpent's Fang).
  - `fighter.boarding_party_resists` (Hullbreaker minion aura).
  - Status flag `ghosted` (Phantom Dancer).
- **Events the world must supply.**
  - `Kills`, including assists.
  - `CC` (holder slowed or immobilized units).
  - Packet tags on champion ability packets: `TAG_ACTIVE_SPELL`, plus `TAG_AOE`/`TAG_PERIODIC`.
  - `Attack.natural_crit` when crits can be forced.
  - `Ctx.base_mana` and `attack_range`.

## Known gaps (not silent; each is in the module `COVERAGE` text)

- **Ally-targeted effects** are helpers only, because the 1v1 lane has no allied champions:
  - Echoes of Helia, Moonstone and Dream Maker;
  - Ardent Censer and Staff of Flowing Water ally buffs;
  - Knight's Vow and Bandlepipes aura;
  - Diadem Consonance;
  - `support.on_ally_support`.
- **No cast instance ids or ability mana costs on packets.** These effects use per-tick or per-cast approximations, and Rod of Ages' level grant is missing:
  - Manaflow/Muramana per cast;
  - Malignance's "ultimate damage";
  - Bloodletter's ICD;
  - Eternity's mana-spent heal.
- **No epic-monster or monster-size flag.** Affects Bastionbreaker, Hullbreaker and Blackfire.
- **Force of Nature:** an immobilize should count as 2 stacks; it is not counted.
- **Vision and Faelights** (Umbral Glaive, Horizon Focus reveal, wards) are deferred.
- **Shadowflame:** its true-damage amp is a follow-up packet. There is no per-type amp on one packet.
- **Open measurement questions** (ITEMS.md §15, U-1…U-15). Defaults are implemented and cited in the code:
  - shop shape;
  - Hydra edge rule;
  - Titanic cone;
  - Immolate first tick;
  - Stridebreaker decay;
  - Steelcaps on on-hit;
  - Doran's Shield;
  - Sterak's decay;
  - level 19–20 scaling.

## Client-vs-wiki choices recorded by the module authors

Client values are implemented throughout. Specific choices:

| Item | Client | Wiki / other | Implemented |
|---|---|---|---|
| Doran's Bow attack speed | 15% | 12% | 15% |
| Redemption cost | 2300 | 2250 | 2300 |
| Tier-3 boot shields | 90 + 10/level from 9 | 100–200 | client formula |
| Luden's range | 650 | 600 | 650 |
| World Atlas champion-damage ally radius | 2000 | 1050 | 2000 |
| World Atlas ranged execute | 33.3% | 30% | 33.3% |
| Dead Man's Plate time to full stacks | 4 s | 3.75 s | 4 s |
| Hullbreaker ranged vs structures | calc ×0.7 | data value 2 | calculation |
| Seraph's shield | — | ITEMS.md §6.3 says current mana | 0.18 × max mana (wiki and calc type) |
| Diadem heal | data value 0.01 | calc and wiki 0.008 | 0.008 |

Calculations that reference unresolved hashes (resolving to 0) fall back to the matching data values that agree
with the wiki: Yun Tal, Youmuu's, Blackfire, Bastionbreaker, Eclipse, Bandlepipes.
