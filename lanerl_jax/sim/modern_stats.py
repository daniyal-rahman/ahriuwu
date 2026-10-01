"""Patch-26.19 shared stat, resistance, and damage arithmetic.

Elementwise NumPy/JAX functions only: every function can be jit/vmap compiled.
Numbers in a resistance breakdown are kept separate until the stat total has
been composed, so temporary buffs do not get folded into base growth.
"""
from __future__ import annotations

from typing import Any

import numpy as np

PHYSICAL = 0
MAGIC = 1
TRUE = 2


def validate_damage_type(damage_type: int) -> int:
    """Host-side guard to run before tracing a damage-type-specialized kernel."""
    value = int(damage_type)
    if value not in (PHYSICAL, MAGIC, TRUE):
        raise ValueError(f"unsupported damage type {value}")
    return value


def stat_total(base_value: Any, base_bonus: Any = 0.0,
               percent_base_bonus: Any = 0.0, flat_bonus: Any = 0.0,
               percent_bonus: Any = 0.0) -> Any:
    """League ``Stat.Total``: ``((base + baseBonus)*(1+pBase)+flat)*(1+p)``."""
    return ((base_value + base_bonus) * (1.0 + percent_base_bonus)
            + flat_bonus) * (1.0 + percent_bonus)


def level_growth_sum(level: Any, xp: Any = np) -> Any:
    """Champion growth multiplier through level L (nonlinear champion curve)."""
    l = xp.asarray(level)
    n = xp.maximum(l - 1.0, 0.0)
    return n * (0.7025 + 0.0175 * n)


def grown_stat(base: Any, growth: Any, level: Any, xp: Any = np) -> Any:
    return base + growth * level_growth_sum(level, xp)


def adaptive_force_total(adaptive_force: Any, *, converts_to_ad: Any,
                         xp: Any = np) -> tuple[Any, Any]:
    """Split adaptive force using current client GameplayConfig ratios.

    Returns ``(bonus_attack_damage, bonus_ability_power)``. `converts_to_ad`
    is selected when the champion's adaptive stat is AD; this is a loadout
    property, not inferred from an in-tick comparison.
    """
    amount = xp.asarray(adaptive_force)
    ad = xp.where(converts_to_ad, amount * 0.6, xp.zeros_like(amount))
    ap = xp.where(converts_to_ad, xp.zeros_like(amount), amount)
    return ad, ap


def change_max_health(current_hp: Any, old_max_hp: Any, new_max_hp: Any,
                      xp: Any = np) -> tuple[Any, Any]:
    """Apply a max-health change like a stat update: gains heal by delta,
    losses do not damage, then current health is clamped to the new maximum.
    """
    delta = new_max_hp - old_max_hp
    new_current = xp.where(delta > 0.0, current_hp + delta, current_hp)
    return xp.maximum(new_current, 0.0), xp.maximum(new_max_hp, 0.0)


def armor_after_modifiers(
    armor: Any, *, flat_reduction: Any = 0.0,
    percent_reduction: Any = 0.0, percent_penetration: Any = 0.0,
    flat_penetration: Any = 0.0, lethality: Any = 0.0,
    xp: Any = np,
) -> Any:
    """Attacker-effective armor, in League's documented operation order.

    Flat armor reduction may create negative armor; percentage reduction is
    multiplicative and ignored at nonpositive armor; penetration cannot take
    armor below zero. Flat armor reduction's distribution between base and
    bonus armor is deliberately not exposed because published wiki prose
    conflicts on which pool is reduced first. For total armor, the order and
    final total here are equivalent while the percentage stages use total.
    Lethality is full flat penetration since patch 14.1.
    """
    r = armor - flat_reduction
    r = xp.where(r > 0.0, r * (1.0 - percent_reduction), r)
    r = xp.where(r > 0.0, r * (1.0 - percent_penetration), r)
    return xp.maximum(0.0, r - flat_penetration - lethality)


def magic_resist_after_modifiers(
    resist: Any, *, flat_reduction: Any = 0.0,
    percent_reduction: Any = 0.0, percent_penetration: Any = 0.0,
    flat_penetration: Any = 0.0, xp: Any = np,
) -> Any:
    """Attacker-effective MR; follows the same reduction/penetration order."""
    r = resist - flat_reduction
    r = xp.where(r > 0.0, r * (1.0 - percent_reduction), r)
    r = xp.where(r > 0.0, r * (1.0 - percent_penetration), r)
    return xp.maximum(0.0, r - flat_penetration)


def mitigation_multiplier(resist: Any, xp: Any = np) -> Any:
    """Physical/magic mitigation, including the negative-resistance branch."""
    ordinary = 100.0 / (100.0 + resist)
    negative = 2.0 - 100.0 / (100.0 - resist)
    return xp.where(resist < 0.0, negative, ordinary)


def post_mitigation_damage(raw: Any, resist: Any, xp: Any = np) -> Any:
    return xp.where(raw > 0.0, raw * mitigation_multiplier(resist, xp),
                    xp.zeros_like(raw))


def apply_damage_modifiers(
    raw: Any, *, attacker_amp: Any = 0.0, attacker_reduction: Any = 0.0,
    target_vulnerability: Any = 0.0, target_reduction: Any = 0.0,
    xp: Any = np,
) -> Any:
    """Compose independent damage modifiers multiplicatively.

    This helper applies the modifier products to a raw source amount. Callers
    choose where mitigation occurs for their event type; League's armor/MR
    multiplier is itself one received-damage modifier.
    """
    mult = ((1.0 + attacker_amp) * (1.0 - attacker_reduction)
            * (1.0 + target_vulnerability) * (1.0 - target_reduction))
    return xp.maximum(raw * mult, 0.0)


def damage_after_resist(raw: Any, resist: Any, damage_type: Any,
                        xp: Any = np) -> Any:
    """Select physical, magical, or true damage. Unknown type is an error host-side."""
    damage_type = xp.asarray(damage_type)
    physical_or_magic = post_mitigation_damage(raw, resist, xp)
    return xp.where(damage_type == TRUE, xp.maximum(raw, 0.0),
                    xp.where((damage_type == PHYSICAL) | (damage_type == MAGIC),
                             physical_or_magic, xp.zeros_like(raw)))
