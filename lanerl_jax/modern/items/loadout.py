"""Loadout preparation used by world construction: rune page validation and stat shards (RUNES.md §2.3)."""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from ..runes import catalog as R
from .catalog import ItemStats


def validate_rune_page(page, traits=None):
    """Apply the client's game-start substitutions to a ``RunePage``; ``None``/empty = no runes (test switch)."""
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
    """Stats of the three shards (names or perk ids, one per slot).

    ``adaptive_to_ad=None`` leaves adaptive shards as unresolved ``adaptive_force`` (STAT.50); tenacity also
    grants slow resist, both multiplicative with other sources.
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
