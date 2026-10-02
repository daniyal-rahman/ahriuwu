"""Patch-26.19 SR item catalog from client build 16.19.8230722.

Host-side loading of ``items_client.json`` (built by
``lanerl_jax.data.build_modern_items``) into immutable NumPy tables that JIT
kernels close over. Item *rows* (0..n_items-1) index every table; the
inventory stores rows, never raw ids. Row ``EMPTY = -1`` is an empty slot.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

PATCH = "26.19"
CLIENT_BUILD = "16.19.8230722"
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "modern" / PATCH / "items_client.json"
EMPTY = -1
N_SLOTS = 7           # six main slots + trinket (index 6)
TRINKET_SLOT = 6


class ItemStats(NamedTuple):
    """Bonus stats contributed by items/shards/effects (all ``bonus``).

    Units: attack speed, crit, vamp, %pen, tenacity, slow resist, heal/shield
    power and percent fields are fractions (0.25 = 25%); ``health_regen`` is
    HP per second; ``percent_base_*_regen`` multiplies champion base regen.
    """
    health: Any = 0.0
    attack_damage: Any = 0.0
    ability_power: Any = 0.0
    armor: Any = 0.0
    magic_resist: Any = 0.0
    attack_speed: Any = 0.0
    crit_chance: Any = 0.0
    life_steal: Any = 0.0
    move_speed: Any = 0.0
    percent_move_speed: Any = 0.0
    ability_haste: Any = 0.0
    health_regen: Any = 0.0
    mana: Any = 0.0
    tenacity: Any = 0.0
    slow_resist: Any = 0.0
    omnivamp: Any = 0.0
    crit_damage: Any = 0.0
    multiplicative_attack_speed: Any = 0.0
    percent_base_health_regen: Any = 0.0
    percent_base_mana_regen: Any = 0.0
    lethality: Any = 0.0
    percent_armor_pen: Any = 0.0
    magic_pen: Any = 0.0
    percent_magic_pen: Any = 0.0
    heal_shield_power: Any = 0.0
    incoming_heal: Any = 0.0          # Spirit Visage-type received heal/shield/regen/vamp increase
    basic_ability_haste: Any = 0.0
    ultimate_haste: Any = 0.0
    summoner_haste: Any = 0.0
    mana_regen: Any = 0.0             # flat mana per second
    # Rune/shard buckets (RUNES.md §8, DAMAGE_AND_STATS §3); no client item sets them.
    adaptive_force: Any = 0.0         # unresolved AF; modern_stats.resolve_adaptive splits it (STAT.50)
    item_haste: Any = 0.0             # item actives (Cosmic Insight)
    trinket_haste: Any = 0.0          # trinkets (Grisly Mementos)
    percent_armor: Any = 0.0          # STAT.40 total-armor multiplier (Conditioning 3%)
    percent_magic_resist: Any = 0.0   # STAT.40 total-MR multiplier
    percent_health: Any = 0.0         # STAT.40 total max-HP multiplier (Overgrowth 3.5%)
    bonus_ms_amp: Any = 0.0           # other bonus MS is (1 + amp) more effective (Celerity 7%)
    silent_health: Any = 0.0          # part of ``health`` whose gains do not raise current HP (Biscuits)
    attack_speed_cap_lift: Any = 0.0  # > 0 lifts the attack-speed cap (Hail of Blades)


STAT_FIELDS = ItemStats._fields
STAT_INDEX = {name: i for i, name in enumerate(STAT_FIELDS)}
# Fields that stack as 1 - prod(1 - x) across sources (DAMAGE_AND_STATS §3.4).
MULTIPLICATIVE_FIELDS = ("tenacity", "slow_resist", "percent_armor_pen", "percent_magic_pen")


def zero_stats(shape=(), xp: Any = jnp) -> ItemStats:
    return ItemStats(*(xp.zeros(shape, xp.float32) for _ in STAT_FIELDS))


def combine_stats(*parts: ItemStats) -> ItemStats:
    """Combine contributions with the documented additive/multiplicative rules."""
    out = {}
    for name in STAT_FIELDS:
        values = [jnp.asarray(getattr(p, name), jnp.float32) for p in parts]
        if name in MULTIPLICATIVE_FIELDS:
            keep = jnp.ones_like(values[0])
            for v in values:
                keep = keep * (1.0 - v)
            out[name] = 1.0 - keep
        else:
            total = values[0]
            for v in values[1:]:
                total = total + v
            out[name] = total
    return ItemStats(**out)


@dataclass(frozen=True)
class ItemSpec:
    item_id: int
    row: int
    name: str
    in_store: bool
    price: int
    total: int
    sell_value: int
    can_be_sold: bool
    max_stack: int
    consumed: bool
    consume_on_acquire: bool
    groups: tuple[str, ...]
    recipe: tuple[int, ...]
    required_level: int
    required_champion: str
    required_spell: str
    required_buff: str
    required_identities: tuple[str, ...]
    stats: ItemStats
    data_values: dict
    calculations: dict
    spell: dict | None
    sidegrades: tuple[int, ...]
    epicness: int
    effect_amount: tuple[float, ...] = ()

    def dv(self, name: str, default: float | None = None) -> float:
        if name in self.data_values:
            return self.data_values[name]
        if default is None:
            raise KeyError(f"item {self.item_id} {self.name} has no data value {name!r}")
        return default


MAX_RECIPE_NODES = 16


class CatalogArrays(NamedTuple):
    """Static JIT tables, leading axis = item row."""
    item_id: Any            # (I,) int32
    stats: Any              # (I, F) float32 per-unit stats
    multiplicative: Any     # (F,) bool
    total: Any              # (I,) int32 recursive cost
    sell_value: Any         # (I,) int32
    can_be_sold: Any        # (I,) bool
    in_store: Any           # (I,) bool
    max_stack: Any          # (I,) int32
    groups: Any             # (I, G) bool membership
    group_max: Any          # (G,) int32, -1 = unlimited
    group_purchase_cd: Any  # (G,) float32
    trinket: Any            # (I,) bool
    required_level: Any     # (I,) int32
    ranged_only: Any        # (I,) bool
    blocked: Any            # (I,) bool: champion/Smite-gated items (deferred)
    required_buff: Any      # (I,) int32 index into BUFF_CURRENCIES, -1 none
    consume_on_acquire: Any  # (I,) bool
    node_item: Any          # (I, M) int32 recipe tree pre-order (component rows), -1 pad
    node_parent: Any        # (I, M) int32 node index of parent, -1 = direct component
    node_total: Any         # (I, M) int32 total cost of node item


BUFF_CURRENCIES = ("Feats_NoxianBootPurchaseBuff", "SupportItemPurchaseBuff",
                   "S11Support_Quest_Completion_Buff", "Item2420")


class Catalog:
    """All SR items of the pinned patch; immutable after construction."""

    def __init__(self, payload: dict):
        if payload.get("schema") != "lanerl-client-sr-items-v1" or payload.get("patch") != PATCH \
                or payload.get("client_build") != CLIENT_BUILD:
            raise RuntimeError("modern item catalog has wrong schema, patch or client build")
        self.sources = dict(payload["sources"])
        self.group_info = dict(payload["groups"])
        rows = sorted(payload["items"].items(), key=lambda kv: int(kv[0]))
        self.ids = tuple(int(k) for k, _ in rows)
        self.row_of = {iid: r for r, iid in enumerate(self.ids)}
        specs = []
        for r, (key, rec) in enumerate(rows):
            stats = ItemStats(**{k: float(v) for k, v in rec["stats"].items()})
            total = int(rec["total"])
            specs.append(ItemSpec(
                item_id=int(key), row=r, name=rec["name"], in_store=rec["in_store"],
                price=int(rec["price"]), total=total,
                sell_value=int(np.floor(total * rec["sell_modifier"] + 0.5)) if rec["can_be_sold"] else 0,
                can_be_sold=bool(rec["can_be_sold"]), max_stack=int(rec["max_stack"]),
                consumed=bool(rec["consumed"]), consume_on_acquire=bool(rec["consume_on_acquire"]),
                groups=tuple(rec["groups"]), recipe=tuple(int(c) for c in rec["recipe"]),
                required_level=int(rec["required_level"]), required_champion=rec["required_champion"],
                required_spell=rec["required_spell"], required_buff=rec["required_buff"],
                required_identities=tuple(rec["required_identities"]), stats=stats,
                data_values=dict(rec["data_values"]), calculations=rec["calculations"],
                spell=rec["spell"], sidegrades=tuple(rec["sidegrades"]), epicness=int(rec["epicness"]),
                effect_amount=tuple(float(x) for x in rec["effect_amount"])))
        self.specs = tuple(specs)
        self.by_id = {s.item_id: s for s in specs}
        self.group_names = tuple(sorted({g for s in specs for g in s.groups}))
        self.arrays = self._arrays()

    def __getitem__(self, item_id: int) -> ItemSpec:
        try:
            return self.by_id[int(item_id)]
        except KeyError:
            raise KeyError(f"item {item_id} is not a patch-{PATCH} SR item") from None

    def __contains__(self, item_id: int) -> bool:
        return int(item_id) in self.by_id

    def row(self, item_id: int) -> int:
        return self.row_of[int(item_id)]

    def dv(self, item_id: int, name: str, default: float | None = None) -> float:
        return self[item_id].dv(name, default)

    def recipe_nodes(self, item_id: int) -> list[tuple[int, int]]:
        """Pre-order (item_id, parent_node_index) of the full component tree."""
        out: list[tuple[int, int]] = []

        def walk(iid: int, parent: int) -> None:
            for c in self[iid].recipe:
                out.append((c, parent))
                walk(c, len(out) - 1)
        walk(item_id, -1)
        if len(out) > MAX_RECIPE_NODES:
            raise RuntimeError(f"item {item_id} recipe tree exceeds {MAX_RECIPE_NODES} nodes")
        return out

    def _arrays(self) -> CatalogArrays:
        n, f, g = len(self.specs), len(STAT_FIELDS), len(self.group_names)
        gi = {name: i for i, name in enumerate(self.group_names)}
        stats = np.zeros((n, f), np.float32)
        groups = np.zeros((n, g), bool)
        nodes = np.full((n, MAX_RECIPE_NODES), EMPTY, np.int32)
        parents = np.full((n, MAX_RECIPE_NODES), -1, np.int32)
        node_total = np.zeros((n, MAX_RECIPE_NODES), np.int32)
        for s in self.specs:
            stats[s.row] = np.asarray(s.stats, np.float32)
            for name in s.groups:
                groups[s.row, gi[name]] = True
            for k, (cid, parent) in enumerate(self.recipe_nodes(s.item_id)):
                nodes[s.row, k] = self.row(cid)
                parents[s.row, k] = parent
                node_total[s.row, k] = self[cid].total
        gmax = np.asarray([int(self.group_info[name]["max_ownable"]) for name in self.group_names], np.int32)
        gcd = np.asarray([float(self.group_info[name].get("purchase_cooldown", 0.0))
                          for name in self.group_names], np.float32)
        spec_col = lambda fn, dt: np.asarray([fn(s) for s in self.specs], dt)
        return CatalogArrays(
            item_id=spec_col(lambda s: s.item_id, np.int32), stats=stats,
            multiplicative=np.asarray([k in MULTIPLICATIVE_FIELDS for k in STAT_FIELDS]),
            total=spec_col(lambda s: s.total, np.int32), sell_value=spec_col(lambda s: s.sell_value, np.int32),
            can_be_sold=spec_col(lambda s: s.can_be_sold, bool), in_store=spec_col(lambda s: s.in_store, bool),
            max_stack=spec_col(lambda s: s.max_stack, np.int32), groups=groups, group_max=gmax,
            group_purchase_cd=gcd, trinket=spec_col(lambda s: "Trinket" in s.groups, bool),
            required_level=spec_col(lambda s: s.required_level, np.int32),
            ranged_only=spec_col(lambda s: "Ranged" in s.required_identities, bool),
            blocked=spec_col(lambda s: bool(s.required_champion or s.required_spell), bool),
            required_buff=spec_col(lambda s: BUFF_CURRENCIES.index(s.required_buff)
                                   if s.required_buff in BUFF_CURRENCIES else (-1 if not s.required_buff else -2),
                                   np.int32),
            consume_on_acquire=spec_col(lambda s: s.consume_on_acquire, bool),
            node_item=nodes, node_parent=parents, node_total=node_total)


@lru_cache(maxsize=1)
def catalog() -> Catalog:
    return Catalog(json.loads(DATA_PATH.read_text()))


def lerp_level(start: Any, end: Any, level: Any) -> Any:
    """``ByCharLevelInterpolation``: linear over 1..18, extrapolating to 19–20
    (README X-1; client ``mScalePastDefaultMaxLevel`` absent)."""
    lv = jnp.maximum(jnp.asarray(level, jnp.float32), 1.0)
    return start + (end - start) * (lv - 1.0) / 17.0


def level_bp(start: Any, per_level: Any, from_level: Any, level: Any) -> Any:
    """``mBonusPerLevelAtAndAfter``: +per_level for each level >= from_level."""
    lv = jnp.asarray(level, jnp.float32)
    return start + per_level * jnp.maximum(0.0, lv - from_level + 1.0)


def ranged_mult(is_ranged: Any, ranged_value: Any) -> Any:
    """Melee/ranged split keyed on the holder (ITEMS.md §2.3)."""
    return jnp.where(is_ranged, ranged_value, 1.0)
