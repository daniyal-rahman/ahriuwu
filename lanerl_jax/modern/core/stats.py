"""26.19 elementwise stat, resistance and damage arithmetic (DAMAGE_AND_STATS §3-5); ``xp`` is numpy or jnp."""
from __future__ import annotations

from typing import Any

import numpy as np

PHYSICAL = 0
MAGIC = 1
TRUE = 2
ADAPTIVE_AD_RATIO = 0.6   # GameplayConfig: 1 AF = 0.6 bonus AD or 1 AP


def validate_damage_type(damage_type: int) -> int:
    value = int(damage_type)
    if value not in (PHYSICAL, MAGIC, TRUE):
        raise ValueError(f"unsupported damage type {value}")
    return value


def stat_total(base_value: Any, base_bonus: Any = 0.0, percent_base_bonus: Any = 0.0, flat_bonus: Any = 0.0,
               percent_bonus: Any = 0.0) -> Any:
    """League ``Stat.Total``: ``((base + baseBonus)*(1+pBase)+flat)*(1+p)``."""
    return ((base_value + base_bonus) * (1.0 + percent_base_bonus) + flat_bonus) * (1.0 + percent_bonus)


def level_growth_sum(level: Any, xp: Any = np) -> Any:
    """Champion growth multiplier through level L (nonlinear champion curve)."""
    n = xp.maximum(xp.asarray(level) - 1.0, 0.0)
    return n * (0.7025 + 0.0175 * n)


def adaptive_force_total(adaptive_force: Any, *, converts_to_ad: Any, xp: Any = np) -> tuple[Any, Any]:
    """``(bonus AD, AP)`` for an already chosen adaptive stat."""
    amount = xp.asarray(adaptive_force)
    ad = xp.where(converts_to_ad, amount * ADAPTIVE_AD_RATIO, xp.zeros_like(amount))
    ap = xp.where(converts_to_ad, xp.zeros_like(amount), amount)
    return ad, ap


def adaptive_is_ad(bonus_ad: Any, ability_power: Any, adaptive_physical: Any = True, xp: Any = np) -> Any:
    """Larger of bonus AD / AP wins, ties use the champion's adaptive type (§3.5). Inputs exclude adaptive force."""
    bonus_ad, ability_power = xp.asarray(bonus_ad), xp.asarray(ability_power)
    return xp.where(bonus_ad > ability_power, True,
                    xp.where(ability_power > bonus_ad, False, xp.asarray(adaptive_physical, bool)))


def resolve_adaptive(adaptive_force: Any, bonus_ad: Any, ability_power: Any, adaptive_physical: Any = True,
                     xp: Any = np) -> tuple[Any, Any]:
    """STAT.50: ``(bonus AD, AP)`` granted by ``adaptive_force``."""
    return adaptive_force_total(adaptive_force, converts_to_ad=adaptive_is_ad(
        bonus_ad, ability_power, adaptive_physical, xp), xp=xp)


def change_max_health(current_hp: Any, old_max_hp: Any, new_max_hp: Any, xp: Any = np) -> tuple[Any, Any]:
    """Gains heal by the delta, losses do not damage."""
    delta = new_max_hp - old_max_hp
    new_current = xp.where(delta > 0.0, current_hp + delta, current_hp)
    return xp.maximum(new_current, 0.0), xp.maximum(new_max_hp, 0.0)


def armor_after_modifiers(armor: Any, *, flat_reduction: Any = 0.0, percent_reduction: Any = 0.0,
                          percent_penetration: Any = 0.0, flat_penetration: Any = 0.0, lethality: Any = 0.0,
                          xp: Any = np) -> Any:
    """§4.2 order: flat reduction (may go negative), % reduction and % pen (positive only), then flat pen and
    lethality, which cannot take positive armor below 0."""
    r = armor - flat_reduction
    r = xp.where(r > 0.0, r * (1.0 - percent_reduction), r)
    r = xp.where(r > 0.0, r * (1.0 - percent_penetration), r)
    return xp.where(r > 0.0, xp.maximum(0.0, r - flat_penetration - lethality), r)


def magic_resist_after_modifiers(resist: Any, *, flat_reduction: Any = 0.0, percent_reduction: Any = 0.0,
                                 percent_penetration: Any = 0.0, flat_penetration: Any = 0.0, xp: Any = np) -> Any:
    r = resist - flat_reduction
    r = xp.where(r > 0.0, r * (1.0 - percent_reduction), r)
    r = xp.where(r > 0.0, r * (1.0 - percent_penetration), r)
    return xp.where(r > 0.0, xp.maximum(0.0, r - flat_penetration), r)


def mitigation_multiplier(resist: Any, xp: Any = np) -> Any:
    ordinary = 100.0 / (100.0 + resist)
    negative = 2.0 - 100.0 / (100.0 - resist)
    return xp.where(resist < 0.0, negative, ordinary)


def post_mitigation_damage(raw: Any, resist: Any, xp: Any = np) -> Any:
    return xp.where(raw > 0.0, raw * mitigation_multiplier(resist, xp), xp.zeros_like(raw))


def apply_damage_modifiers(raw: Any, *, attacker_amp: Any = 0.0, attacker_reduction: Any = 0.0,
                           target_vulnerability: Any = 0.0, target_reduction: Any = 0.0, is_true: Any = False,
                           xp: Any = np) -> Any:
    """DMG.40/60 (§5.5, §5.7): source amps and reduction (not on true damage) share one sum; target
    vulnerability and ``target_reduction`` (combined ``1 - prod(1 - r)``, not on true damage) multiply."""
    is_true = xp.asarray(is_true, bool)
    dealt = 1.0 + attacker_amp - xp.where(is_true, 0.0, attacker_reduction)
    received = (1.0 + target_vulnerability) * xp.where(is_true, 1.0, 1.0 - target_reduction)
    return xp.maximum(raw * xp.maximum(dealt, 0.0) * received, 0.0)
