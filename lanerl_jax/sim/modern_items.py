"""Patch-pinned static SR item data and the supported 26.19 top-lane subset.

Base stats come from the patch-matched Data Dragon export. Behavioral item
effects are separate kernels: unknown effects are surfaced to the caller and
can be made a hard error at loadout validation time.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp

from .modern_stats import PHYSICAL

PATCH = "26.19"
MAP_ID = 11
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "modern" / PATCH


class ItemStats(NamedTuple):
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


_STAT_MAP = {
    "FlatHPPoolMod": "health", "FlatPhysicalDamageMod": "attack_damage",
    "FlatMagicDamageMod": "ability_power", "FlatArmorMod": "armor",
    "FlatSpellBlockMod": "magic_resist", "PercentAttackSpeedMod": "attack_speed",
    "FlatCritChanceMod": "crit_chance", "PercentLifeStealMod": "life_steal",
    "FlatMovementSpeedMod": "move_speed", "FlatHPPoolRegenMod": "health_regen",
    "PercentMovementSpeedMod": "percent_move_speed",
    "FlatHPRegenMod": "health_regen", "FlatMPPoolMod": "mana",
}


@dataclass(frozen=True)
class ItemSpec:
    item_id: int
    name: str
    gold: int
    item_limit: str
    stats: ItemStats
    has_behavioral_effect: bool


def _load_item_records() -> dict[int, ItemSpec]:
    payload = json.loads((DATA_DIR / "items.json").read_text())
    if payload.get("schema") != "lanerl-ddragon-sr-items-v1" or payload.get("patch") != PATCH:
        raise RuntimeError("modern item data has wrong schema or patch")
    result = {}
    for key, row in payload["items"].items():
        sums = {field: 0.0 for field in ItemStats._fields}
        for source_key, value in row.get("stats", {}).items():
            dest = _STAT_MAP.get(source_key)
            if dest is not None:
                sums[dest] += float(value)
        # Item descriptions are not machine-readable effect implementations.
        # Mark any passive/active as behavior so a caller must opt into a
        # known effect or explicitly carry the unsupported marker.
        description = row.get("description", "")
        # Data Dragon's stats map omits Ability Haste and a few newer fields;
        # extract only the known `<attention>value</attention> Label` format.
        import re
        for value, percent_mark, label in re.findall(r"<attention>([0-9.]+)(%?)</attention>\s*([^<]+)", description):
            label = label.strip().lower()
            field = {"ability haste": "ability_haste", "health regeneration": "health_regen",
                     "tenacity": "tenacity"}.get(label)
            if field:
                sums[field] += float(value) / (100.0 if percent_mark or label == "tenacity" else 1.0)
        behavioral = any(tag in description for tag in
                         ("<passive>", "<active>", "<consumable>", "<unique>", "<speed>", "<healing>", "<shield>", "<status>"))
        iid = int(key)
        result[iid] = ItemSpec(
            item_id=iid, name=row["name"], gold=int(row.get("gold", {}).get("total", 0)),
            item_limit=str(row.get("itemLimit", "")),
            stats=ItemStats(**sums), has_behavioral_effect=behavioral)
    return result


ITEMS: dict[int, ItemSpec] = _load_item_records()

_HYDRA_ITEM_IDS = frozenset((3074, 3077, 3748, 6631, 6698))

# Current SR item IDs, not Arena's 22xxxx mirror. Active values and timers are
# pinned from the 26.19 game-data-derived reference noted in the fidelity row.
TIAMAT_ID = 3077
STRIDEBREAKER_ID = 6631
ACTIVE_ITEM_SPECS = {
    TIAMAT_ID: {"name": "Crescent", "damage_ad_ratio": 0.75,
                "radius": 450.0, "cooldown_seconds": 10.0, "shape": "forward"},
    STRIDEBREAKER_ID: {"name": "Breaking Shockwave", "damage_ad_ratio": 0.80,
                       "radius": 450.0, "cooldown_seconds": 15.0,
                       "slow": 0.35, "slow_seconds": 3.0,
                       "bonus_move_speed_per_champion": 0.35,
                       "shape": "radial"},
}

# Item-limit groups are encoded by Data Dragon, e.g. "Hydra". Reject a
# duplicate group even when the IDs differ (Tiamat/Stridebreaker/Hydra).
def validate_item_loadout(item_ids: tuple[int, ...] | list[int], *, strict_effects: bool = True) -> tuple[int, ...]:
    ids = tuple(int(i) for i in item_ids)
    if len(ids) > 6:
        raise ValueError("Summoner's Rift inventory supports at most six items")
    groups: dict[str, int] = {}
    unsupported = []
    for iid in ids:
        if iid not in ITEMS:
            raise ValueError(f"item {iid} is not purchasable on Map11 in patch {PATCH}")
        spec = ITEMS[iid]
        group = "Hydra" if iid in _HYDRA_ITEM_IDS else spec.item_limit
        if group:
            if group in groups:
                raise ValueError(f"item-limit group {group!r} conflicts: {groups[group]} and {iid}")
            groups[group] = iid
        supported_scope = (iid in ACTIVE_ITEM_SPECS)
        if spec.has_behavioral_effect and not supported_scope:
            unsupported.append(iid)
    if strict_effects and unsupported:
        names = ", ".join(f"{i} {ITEMS[i].name}" for i in unsupported)
        raise NotImplementedError(f"item effects not implemented for: {names}")
    return tuple(unsupported)


def item_loadout_stats(item_ids: tuple[int, ...] | list[int], *,
                       strict_effects: bool = True) -> tuple[ItemStats, tuple[int, ...]]:
    """Aggregate six-slot static stats; optionally fail on unsupported passives.

    The returned ID tuple is empty in strict mode, or identifies each item
    whose effect the caller needs to handle when ``strict_effects=False``.
    """
    unsupported = validate_item_loadout(item_ids, strict_effects=strict_effects)
    totals = {key: 0.0 for key in ItemStats._fields}
    for iid in item_ids:
        stats = ITEMS[int(iid)].stats
        for key in ItemStats._fields:
            totals[key] += getattr(stats, key)
    return ItemStats(**totals), unsupported


def load_rune_data() -> dict:
    """Return the patch-pinned rune catalog; effect execution is intentionally explicit."""
    payload = json.loads((DATA_DIR / "runes.json").read_text())
    if payload.get("schema") != "lanerl-ddragon-runes-v1" or payload.get("patch") != PATCH:
        raise RuntimeError("modern rune data has wrong schema or patch")
    return payload


def validate_rune_page(rune_ids: tuple[int, ...] | list[int]) -> None:
    """Fail closed until a selected rune has an implemented runtime effect.

    DDragon's rune descriptions/catalog are available as data, but do not
    encode live stacking, cooldown, trigger, or healing rules. The all-empty
    page is the only supported rune page until a ruleset page is selected.
    """
    if rune_ids:
        raise NotImplementedError(
            "selected combat runes are catalogued but their 26.19 runtime "
            "triggers are not implemented; use an explicit empty rune page")


STAT_SHARD_OPTIONS = (
    frozenset(("adaptive", "attack_speed", "ability_haste")),
    frozenset(("adaptive", "move_speed", "health_scaling")),
    frozenset(("health_flat", "tenacity", "health_scaling")),
)
DEFAULT_STAT_SHARDS = ("adaptive", "adaptive", "health_flat")


def stat_shard_stats(shards: tuple[str, str, str] = DEFAULT_STAT_SHARDS,
                     *, level: Any = 1, adaptive_to_ad: bool = True,
                     xp: Any = jnp) -> ItemStats:
    """Compose the three 26.19 stat shards; keystone/minor rune IDs stay separate.

    Shard values follow patch-matched rune data: adaptive +9 (5.4 bonus AD
    or 9 AP), attack speed +10%, ability haste +8, move speed +2%, health
    +65, scaling health +10..180 over levels 1..18, tenacity/slow-resist
    +10%. Scaling health remains capped at its level-18 endpoint.
    """
    if len(shards) != 3:
        raise ValueError("a rune page has exactly three stat shard slots")
    for slot, value in enumerate(shards):
        if value not in STAT_SHARD_OPTIONS[slot]:
            raise ValueError(f"invalid shard {value!r} in slot {slot + 1}")
    totals = {key: 0.0 for key in ItemStats._fields}
    for slot, value in enumerate(shards):
        if value == "adaptive":
            ad_selected = xp.asarray(adaptive_to_ad)
            totals["attack_damage"] += xp.where(ad_selected, 5.4, 0.0)
            totals["ability_power"] += xp.where(ad_selected, 0.0, 9.0)
        elif value == "attack_speed":
            totals["attack_speed"] += 0.10
        elif value == "ability_haste":
            totals["ability_haste"] += 8.0
        elif value == "move_speed":
            totals["percent_move_speed"] += 0.02
        elif value == "health_flat":
            totals["health"] += 65.0
        elif value == "health_scaling":
            shard_level = xp.clip(xp.asarray(level), 1.0, 18.0)
            totals["health"] += 10.0 + 10.0 * (shard_level - 1.0)
        elif value == "tenacity":
            totals["tenacity"] += 0.10
            totals["slow_resist"] += 0.10
    return ItemStats(**totals)


def tiamat_crescent(attack_damage: Any, dx: Any, dz: Any,
                     facing_x: Any, facing_z: Any, *, cooldown_ready: Any = True,
                     xp: Any = jnp) -> tuple[Any, Any]:
    """Tiamat active raw physical damage per target and hit mask.

    ``dx,dz`` are target minus caster. The active is a 450-unit forward
    half-plane; callers apply armor and cooldown state separately.
    """
    in_front = dx * facing_x + dz * facing_z >= 0.0
    hit = (dx * dx + dz * dz <= 450.0 ** 2) & in_front & cooldown_ready
    return xp.where(hit, 0.75 * attack_damage, 0.0), hit


def stridebreaker_active(attack_damage: Any, distance: Any, *,
                         champion_target: Any, cooldown_ready: Any = True,
                         xp: Any = jnp) -> tuple[Any, Any, Any, Any]:
    """Shockwave raw damage/hit/slow/self-MS gain, with 26.19 values."""
    hit = (distance <= 450.0) & cooldown_ready
    damage = xp.where(hit, 0.80 * attack_damage, 0.0)
    slow = xp.where(hit, 0.35, 0.0)
    champion_hits = xp.sum((hit & champion_target).astype(xp.float32))
    move_speed_bonus = xp.where(cooldown_ready, 0.35 * champion_hits, 0.0)
    return damage, hit, slow, move_speed_bonus


def tiamat_cleave(attack_damage: Any, distance_from_primary: Any,
                  is_melee: Any, *, primary_target: Any = False,
                  xp: Any = jnp) -> tuple[Any, Any]:
    """Cleave's splash raw damage (40% melee/20% ranged), 350 radius."""
    hit = (distance_from_primary <= 350.0) & (~primary_target)
    ratio = xp.where(is_melee, 0.40, 0.20)
    return xp.where(hit, attack_damage * ratio, 0.0), hit


def active_damage_type() -> int:
    """Both supported Hydra-family actives deal physical damage."""
    return PHYSICAL
