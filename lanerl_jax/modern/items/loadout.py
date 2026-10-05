"""Patch-26.19 item loadout stats, stat shards and rune page preparation.

Item data comes from client build 16.19.8230722 (``items.catalog``);
inventory/shop rules live in ``items.inventory`` and every item passive or
in-scope active in ``items.effects``; runes in ``runes.catalog`` and
``runes.effects``. This module keeps the small host-side loadout API
used by world construction.
"""
from __future__ import annotations

import json
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..data import PATCH, PATCH_DIR
from ..runes import catalog as R
from . import inventory as I
from .catalog import ItemStats, catalog
from .effects import STATS_ONLY

MAP_ID = 11


def item_loadout_stats(item_ids, *, strict_effects: bool = True) -> tuple[ItemStats, tuple[int, ...]]:
    """Static six-slot stats of a fixed loadout.

    Item effects are implemented in ``items.effects`` but the world
    tick does not dispatch them yet, so by default a loadout containing an
    item with behaviour beyond its stat line is rejected rather than silently
    run as stats only. With ``strict_effects=False`` the second value lists
    those items for the caller to handle.
    """
    ids = tuple(int(i) for i in item_ids)
    I.validate_item_loadout(ids)
    behavioural = tuple(i for i in ids if i not in STATS_ONLY)
    if strict_effects and behavioural:
        names = ", ".join(f"{i} {catalog()[i].name}" for i in behavioural)
        raise NotImplementedError(
            f"item effects are not dispatched by the world tick yet for: {names}")
    inv = I.inventory_from_ids([list(ids)], trinket=False)
    stats = I.inventory_stats(inv)
    return ItemStats(*(float(v[0]) for v in stats)), behavioural


def load_rune_data() -> dict:
    """Return the patch-pinned Data Dragon rune catalog (names and tree layout)."""
    payload = json.loads((PATCH_DIR / "runes.json").read_text())
    if payload.get("schema") != "lanerl-ddragon-runes-v1" or payload.get("patch") != PATCH:
        raise RuntimeError("modern rune data has wrong schema, patch or client build")
    return payload


def validate_rune_page(page, traits=None):
    """Validate a page and apply the client's game-start substitutions.

    ``page`` is a ``runes.catalog.RunePage``; ``None`` or an empty
    sequence is the explicit no-runes ruleset (legacy/test switch, not a
    legal SR page). Returns the prepared page (or ``None``).
    """
    if page is None or (not isinstance(page, R.RunePage) and len(page) == 0):
        return None
    if not isinstance(page, R.RunePage):
        raise TypeError("pass a runes.catalog.RunePage (bare perk lists lack the tree choice)")
    return R.prepare_page(page, traits or R.ChampionTraits(has_immobilize=True))


STAT_SHARD_OPTIONS = (
    frozenset(("adaptive", "attack_speed", "ability_haste")),
    frozenset(("adaptive", "move_speed", "health_scaling")),
    frozenset(("health_flat", "tenacity", "health_scaling")),
)
DEFAULT_STAT_SHARDS = ("adaptive", "adaptive", "health_flat")


def stat_shard_stats(shards=DEFAULT_STAT_SHARDS, *, level: Any = 1, adaptive_to_ad: Any = True,
                     xp: Any = jnp) -> ItemStats:
    """Compose the three 26.19 stat shards (client perks, RUNES.md §2.3).

    ``shards`` are names or perk ids (5008 Adaptive, 5005 AS, 5007 AH, 5010
    MS, 5001 scaling HP, 5011 HP, 5013 tenacity), one per slot. Adaptive +9
    (5.4 bonus AD or 9 AP), attack speed +10%, ability haste +8, move speed
    +2.5%, health +65, scaling health ``10·level`` (10–180 over 1–18,
    extrapolated to 200 at level 20 per README X-1), tenacity and slow
    resist +15% (multiplicative with other sources). ``adaptive_to_ad=None``
    leaves the adaptive shards as unresolved ``adaptive_force`` for
    ``core.stats.resolve_adaptive`` (STAT.50 dynamic choice).
    """
    if len(shards) != 3:
        raise ValueError("a rune page has exactly three stat shard slots")
    names = {v: k for k, v in R.SHARD_NAMES.items()}
    shards = tuple(names.get(s, s) if isinstance(s, (int, np.integer)) else s for s in shards)
    for slot, value in enumerate(shards):
        if value not in STAT_SHARD_OPTIONS[slot]:
            raise ValueError(f"invalid shard {value!r} in slot {slot + 1}")
    ea = R.ea
    totals = {key: 0.0 for key in ItemStats._fields}
    for value in shards:
        if value == "adaptive":
            if adaptive_to_ad is None:
                totals["adaptive_force"] += ea(R.SHARD_ADAPTIVE, "StatGain2")
            else:
                ad_selected = xp.asarray(adaptive_to_ad)
                totals["attack_damage"] += xp.where(ad_selected, ea(R.SHARD_ADAPTIVE, "StatGain1"), 0.0)
                totals["ability_power"] += xp.where(ad_selected, 0.0, ea(R.SHARD_ADAPTIVE, "StatGain2"))
        elif value == "attack_speed":
            totals["attack_speed"] += ea(R.SHARD_AS, "StatGain") / 100.0
        elif value == "ability_haste":
            totals["ability_haste"] += ea(R.SHARD_AH, "HasteGain")
        elif value == "move_speed":
            totals["percent_move_speed"] += ea(R.SHARD_MS, "StatGain1") / 100.0
        elif value == "health_flat":
            totals["health"] += ea(R.SHARD_HEALTH, "StatGain")
        elif value == "health_scaling":
            totals["health"] += R.lin(ea(R.SHARD_HEALTH_SCALING, "StatGainMin"),
                                      ea(R.SHARD_HEALTH_SCALING, "StatGainMax"), level)
        elif value == "tenacity":
            keep = 1.0 - ea(R.SHARD_TENACITY, "StatGain") / 100.0
            totals["tenacity"] = 1.0 - (1.0 - totals["tenacity"]) * keep
            totals["slow_resist"] = 1.0 - (1.0 - totals["slow_resist"]) * keep
    return ItemStats(**totals)
