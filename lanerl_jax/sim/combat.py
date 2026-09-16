"""Stats, damage and attack timing, ported from ``Stats.cs`` / ``ObjAIBase.cs``.

Pure functions over scalars or arrays -- they work under numpy for the reference
path and under ``jnp`` for the sim, so there is one implementation rather than
two that can drift.

The four formulas that matter, and the one that is not what it looks like
------------------------------------------------------------------------

**Mitigation** (``Stats.GetPostMitigationDamage``)::

    damage <= 0            -> 0
    mitigationPercent = 100 / (100 + stat)
    stat < 0               -> mitigationPercent = 2 - mitigationPercent
    return damage * mitigationPercent

The negative-resistance branch is easy to miss and easy to get backwards: with
``stat = -100`` the first line gives ``100/0``, so the branch exists to keep
armour shred finite as well as to make it *amplify*.

**Per-level growth is NOT linear.** ``Stats.LevelUp`` increments ``Level``
*first*, then adds ``GetLevelUpStatValue(perLevel)`` where::

    GetLevelUpStatValue(value) = value * (0.65f + 0.035f * Level)

So a stat at level N is ``base + perLevel * sum_{L=2..N} (0.65 + 0.035 L)``, not
``base + perLevel * (N - 1)``.  Confirmed independently: those coefficients are
exactly ``Maps/Map1/StatsProgression.json``'s ``PerLevelStatsFactor`` table
(Level2 = 0.72, Level10 = 1.0, Level11 = 1.035, Level18 = 1.28), which the server
computes rather than reads.  A linear reading is ~19% light on Garen's HP at
level 6 and grows worse -- large enough to change every trade and every
last-hit-under-tower calculation.

**Attack speed** (``Stats.cs:130``, ``:228``)::

    AttackSpeedFlat      = 1 / gcd_AttackDelay / (1 + AttackDelayOffsetPercent)
    GetTotalAttackSpeed  = AttackSpeedFlat * AttackSpeedMultiplier.Total
    _autoAttackCurrentCooldown = 1 / GetTotalAttackSpeed()

and level-ups feed ``GrowthAttackSpeed / 100`` into the multiplier's
``PercentBaseBonus``, through the same non-linear growth factor.

**Stat composition** (``Stat.Total``)::

    ((BaseValue + BaseBonus) * (1 + PercentBaseBonus) + FlatBonus) * (1 + PercentBonus)

The order is load-bearing: flat bonuses land *after* the percent-base multiplier
and *before* the percent multiplier.
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = [
    "post_mitigation_damage",
    "level_up_factor",
    "growth_sum",
    "stat_at_level",
    "stat_total",
    "attack_speed_flat",
    "total_attack_speed",
    "attack_period",
    "attack_windup",
    "TURRET_DAMAGE_VS_MINION",
    "outer_turret_ramps",
    "outer_turret_attack_damage",
]


#: Turret damage to minions is **NOT** discounted on this map. Kept as a named
#: 1.0 rather than deleted, because the 0.7 is real on a different map and the
#: next person to read `SRUAP_Turret_Order3/BasicAttack.cs` will find it again.
#:
#: That script does::
#:
#:     var dmg = owner.Stats.AttackDamage.Total;
#:     if (target is Minion) { dmg *= 0.7f; }
#:
#: and I applied it to every turret. It does not apply here. Scripts are
#: resolved by CHARACTER NAME, and `lanerl/cfg/garen1v1.json` pins `"map": 1`,
#: whose turrets are `OrderTurretNormal`/`ChaosTurretWorm` -- names with no
#: `Characters/` folder, so the spell falls back to `SpellScriptEmpty` and the
#: damage goes through the native `ObjAIBase.AutoAttackHit` at full AD.
#: `SRUAP_Turret_*` are Map11 units and are never spawned on this map.
#:
#: The lesson is the one this project keeps relearning: a Content script only
#: applies if the unit it is named for is the unit actually on the field. I
#: verified the script existed and never verified it was *reached*.
#:
#: Settled against a 600 s recording -- see `data/patch.TURRET_MODELS` for the
#: regen and armour measurements that identify the model.
TURRET_DAMAGE_VS_MINION = 1.0


# --- turret stat growth, from the MAP script rather than the stat table -----
#
# `Maps/Map1/LevelScriptObjects.OnUpdate` ramps turret stats on two schedules,
# and neither is in any Content stat JSON -- they are `StatsModifier`s the map
# script adds on a timer::
#
#     OuterTurretStatsModifier.MagicResist.FlatBonus  = 1     # OUTER tier
#     OuterTurretStatsModifier.AttackDamage.FlatBonus = 4     # note: no Armor
#
#     TurretStatsModifier.Armor.FlatBonus       = 1           # every other tier
#     TurretStatsModifier.MagicResist.FlatBonus = 1
#     TurretStatsModifier.AttackDamage.FlatBonus = 4
#
# fired from::
#
#     if (gameTime >= timeCheck && timesApplied < 30)                UpdateTowerStats();
#     if (gameTime >= outerTurretTimeCheck && outerTurretTimesApplied < 7)
#                                                                   UpdateOuterTurretStats();
#
# with `outerTurretTimeCheck` starting at **30 s** and `timeCheck` at **480 s**,
# each advancing 60 s per application.
#
# So an outer turret's attack damage is NOT its Content value for most of a
# game: it is 152 for the first 30 seconds and 180 from 390 s onward.
#
# VERIFIED against a 600 s idle recording, which is also where the absence of an
# Armor bonus on the outer tier shows up: all 165 measured minion-on-turret hits
# sat at a constant armour 60, and the turret's own hits stepped exactly as this
# predicts --
#
#     210-270 s  predicted 168  observed 168.00 (n=7)
#     270-330 s  predicted 172  observed 172.00 (n=1)
#     390+   s   predicted 180  observed 180.00 (n=16)
#
# including the cannon minions, whose 15 armour turns those into 146.09 /
# 149.57 / 153.04 -- every observed value in the trace is accounted for.
OUTER_TURRET_RAMP_START_MS = 30_000.0
OUTER_TURRET_RAMP_PERIOD_MS = 60_000.0
OUTER_TURRET_RAMP_MAX = 7
#: `AttackDamage.FlatBonus` per application.
TURRET_AD_PER_RAMP = 4.0
#: The other tiers start at 480 s and also gain +1 Armor and +1 MagicResist.
INNER_TURRET_RAMP_START_MS = 480_000.0
INNER_TURRET_RAMP_MAX = 30


def outer_turret_ramps(t_ms: Any, xp: Any = np) -> Any:
    """How many times `UpdateOuterTurretStats` has fired by ``t_ms``. 0..7."""
    n = xp.floor((t_ms - OUTER_TURRET_RAMP_START_MS)
                 / OUTER_TURRET_RAMP_PERIOD_MS) + 1.0
    return xp.clip(xp.where(t_ms >= OUTER_TURRET_RAMP_START_MS, n, 0.0),
                   0.0, float(OUTER_TURRET_RAMP_MAX))


def outer_turret_attack_damage(base_ad: Any, t_ms: Any, xp: Any = np) -> Any:
    """An outer turret's attack damage at game time ``t_ms``. 152 -> 180."""
    return base_ad + TURRET_AD_PER_RAMP * outer_turret_ramps(t_ms, xp)


def post_mitigation_damage(damage: Any, resist: Any, xp: Any = np) -> Any:
    """``Stats.GetPostMitigationDamage`` for physical/magical damage.

    ``resist`` is ``Armor.Total`` or ``MagicResist.Total``. True damage does not
    come through here at all -- the server returns early -- so callers must not
    route it here with ``resist = 0``, which would be the same number by luck
    rather than by contract.
    """
    pct = 100.0 / (100.0 + resist)
    pct = xp.where(resist < 0, 2.0 - pct, pct)
    return xp.where(damage <= 0.0, xp.zeros_like(damage * pct), damage * pct)


def level_up_factor(level: Any) -> Any:
    """``GetLevelUpStatValue``'s multiplier at the level being *reached*."""
    return 0.65 + 0.035 * level


def growth_sum(level: Any, xp: Any = np) -> Any:
    """Total growth multiplier accumulated from level 1 up to ``level``.

    Closed form of ``sum_{L=2..N} (0.65 + 0.035 L)``, which is
    ``0.65(N-1) + 0.035 (N(N+1)/2 - 1)``. Written closed-form rather than as a
    loop so it stays a pure elementwise op under ``vmap``.
    """
    n = xp.asarray(level, dtype=xp.float32 if hasattr(xp, "float32") else float)
    return xp.where(n <= 1, xp.zeros_like(n),
                    0.65 * (n - 1.0) + 0.035 * (n * (n + 1.0) / 2.0 - 1.0))


def stat_at_level(base: Any, per_level: Any, level: Any, xp: Any = np) -> Any:
    """A base stat grown to ``level`` through the server's non-linear curve."""
    return base + per_level * growth_sum(level, xp)


def stat_total(base_value: Any, base_bonus: Any = 0.0, percent_base_bonus: Any = 0.0,
               flat_bonus: Any = 0.0, percent_bonus: Any = 0.0) -> Any:
    """``Stat.Total``. Order matters; see the module docstring."""
    return ((base_value + base_bonus) * (1.0 + percent_base_bonus)
            + flat_bonus) * (1.0 + percent_bonus)


def attack_speed_flat(global_attack_delay: float,
                      attack_delay_offset_percent: Any) -> Any:
    """``Stats.cs:130``. ``global_attack_delay`` is ``gcd_AttackDelay`` (1.6 s)."""
    return 1.0 / global_attack_delay / (1.0 + attack_delay_offset_percent)


def total_attack_speed(flat: Any, multiplier: Any = 1.0) -> Any:
    """``Stats.GetTotalAttackSpeed``: attacks per second."""
    return flat * multiplier


def attack_period(flat: Any, multiplier: Any = 1.0) -> Any:
    """Seconds between swings -- what ``_autoAttackCurrentCooldown`` is set to."""
    return 1.0 / total_attack_speed(flat, multiplier)


def attack_windup(period: Any, global_cast_percent: float,
                  attack_delay_cast_offset_percent: Any) -> Any:
    """Seconds from swing start until the damage lands.

    ``gcd_AttackDelayCastPercent`` (0.3) is the global fraction of the attack
    period spent winding up, shifted per character by
    ``AttackDelayCastOffsetPercent``. Garen's is negative (-0.0917), so he
    commits damage slightly earlier in the swing than the global default.

    This is the quantity last-hitting is timed against, and it is one of the
    fields the state dump does **not** expose, so it is currently checked
    against `lanerl_rl/constants.py`'s independently derived values rather than
    against the server directly.
    """
    return period * (global_cast_percent + attack_delay_cast_offset_percent)
