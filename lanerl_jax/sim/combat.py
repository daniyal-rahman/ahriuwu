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
    "other_turret_ramps",
    "other_turret_attack_damage",
    "other_turret_armor",
    "garen_passive_exempt",
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
#: `AttackDamage.FlatBonus` per application, both schedules.
TURRET_AD_PER_RAMP = 4.0
#: `Armor.FlatBonus` / `MagicResist.FlatBonus` per application on the OTHER
#: schedule. The outer schedule grants MagicResist too (`OuterTurretStats
#: Modifier.MagicResist.FlatBonus = 1`, `:164`) but never Armor -- and nothing
#: in this sim currently resolves magic damage against a turret, so that half
#: of the outer modifier has no observable effect and is intentionally not
#: wired into `outer_turret_attack_damage`'s caller.
TURRET_ARMOR_PER_RAMP = 1.0

#: The other four tiers -- INNER, INHIBITOR, NEXUS (FOUNTAIN is excluded by the
#: server itself, see below) -- start at **480 s**, not 30 s, and each
#: application adds Armor and MagicResist as well as AD. 480 s is INSIDE a
#: 600 s episode (two applications land, at 480 s and 540 s), so this schedule
#: is live and was previously unmodelled entirely -- every non-outer turret
#: stayed at its Content armour/AD for the whole game.
OTHER_TURRET_RAMP_START_MS = 480_000.0
OTHER_TURRET_RAMP_PERIOD_MS = 60_000.0
#: `UpdateTowerStats` runs while `timesApplied < 30` (`:172`), so 30 is the
#: cap that matters for INHIBITOR and NEXUS. INNER is special-cased out once
#: `timesApplied >= 20` (`:234`) -- i.e. from its 21st application, at
#: `480_000 + 20 * 60_000 = 1_680_000` ms -- which is **not reachable inside a
#: 600 s episode** (at most 2 applications land here). Modelling one ramp
#: function for all three tiers is therefore exact for every episode length
#: this project runs today; the day episodes exceed ~28 minutes, INNER needs
#: its own capped variant.
OTHER_TURRET_RAMP_MAX = 30
#: FOUNTAIN_TURRET is excluded from `UpdateTowerStats` explicitly (`:234`,
#: `OUTER_TURRET || FOUNTAIN_TURRET || ...`) and was never a candidate for
#: `UpdateOuterTurretStats` either (`:255`, which only ever looks up the
#: OUTER_TURRET of each lane). So a fountain gets neither ramp, ever -- it is
#: not "the other schedule with a 0 rate", it is not on any schedule.


def outer_turret_ramps(t_ms: Any, xp: Any = np) -> Any:
    """How many times `UpdateOuterTurretStats` has fired by ``t_ms``. 0..7."""
    n = xp.floor((t_ms - OUTER_TURRET_RAMP_START_MS)
                 / OUTER_TURRET_RAMP_PERIOD_MS) + 1.0
    return xp.clip(xp.where(t_ms >= OUTER_TURRET_RAMP_START_MS, n, 0.0),
                   0.0, float(OUTER_TURRET_RAMP_MAX))


def outer_turret_attack_damage(base_ad: Any, t_ms: Any, xp: Any = np) -> Any:
    """An outer turret's attack damage at game time ``t_ms``. 152 -> 180."""
    return base_ad + TURRET_AD_PER_RAMP * outer_turret_ramps(t_ms, xp)


def other_turret_ramps(t_ms: Any, xp: Any = np) -> Any:
    """How many times `UpdateTowerStats` has fired by ``t_ms``, for an
    INNER/INHIBITOR/NEXUS turret. 0..30, first at 480 s. See
    `OTHER_TURRET_RAMP_MAX` for why INNER's own 20-application cutoff does not
    need a separate function at this project's episode lengths."""
    n = xp.floor((t_ms - OTHER_TURRET_RAMP_START_MS)
                 / OTHER_TURRET_RAMP_PERIOD_MS) + 1.0
    return xp.clip(xp.where(t_ms >= OTHER_TURRET_RAMP_START_MS, n, 0.0),
                   0.0, float(OTHER_TURRET_RAMP_MAX))


def other_turret_attack_damage(base_ad: Any, t_ms: Any, xp: Any = np) -> Any:
    """An INNER/INHIBITOR/NEXUS turret's attack damage at game time ``t_ms``."""
    return base_ad + TURRET_AD_PER_RAMP * other_turret_ramps(t_ms, xp)


def other_turret_armor(base_armor: Any, t_ms: Any, xp: Any = np) -> Any:
    """An INNER/INHIBITOR/NEXUS turret's armour at game time ``t_ms``.

    The outer tier has no equivalent -- `OuterTurretStatsModifier` never sets
    an Armor bonus (`:164-165`), which is also what the constant-armour-60
    measurement in `data.patch.TURRET_MODELS` depends on.
    """
    return base_armor + TURRET_ARMOR_PER_RAMP * other_turret_ramps(t_ms, xp)


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


def garen_passive_exempt(attacker_is_lane_minion: Any, attacker_is_cannon_or_super: Any,
                         victim_level: Any, xp: Any = np) -> Any:
    """``CharScriptGaren.ShouldPassiveTurnOff``'s UnitTag/level gate.

    ``(attacker, victim)`` pair: ``True`` where a hit from that attacker does
    NOT count as combat against that victim's Garen passive (the
    ``PostMitigationDamage<=0`` half of the real check is not reproduced here
    -- callers already gate on ``damage>0`` before this matters, e.g.
    ``step.py``'s ``hit_by_combat`` sums only positive `dmg_ij` entries).

    ``UnitTag`` is ``[Flags]` with NO explicit values
    (`GameServerCore/Enums/UnitTag.cs`), so C# numbers it sequentially:
    Champion=0, Champion_Clone=1, Minion=2, Minion_Lane=3,
    Minion_Lane_Siege=4, Minion_Lane_Super=5, Minion_Summon=6, Monster=7.
    ``MINION_UNIT_TAG_PASSIVE_EXCEPTIONS`` (`CharScriptGaren.cs:21-28`) lists
    ``{Minion, Minion_Lane, Minion_Lane_Siege, Minion_Lane_Super,
    Minion_Summon}`` BY NAME -- i.e. the raw values ``{2,3,4,5,6}`` -- but a
    real minion's ``UnitTags`` field is the bitwise OR of ALL its tags:
    melee/caster (``Blue_Minion_Basic.json``, ``"Minion | Minion_Lane"``) is
    ``2|3`` = **3** (in the set, exempt unconditionally); cannon
    (``"Minion | Minion_Lane | Minion_Lane_Siege"``) is ``2|3|4`` = **7**, and
    super (``"Minion | Minion_Lane | Minion_Lane_Super"``) is ``2|3|5`` =
    **7** too -- BOTH collide with ``Monster``'s raw value and neither is in
    ``{2,3,4,5,6}``, so despite ``Minion_Lane_Siege``/``Minion_Lane_Super``
    being named right there in the exceptions list, NEITHER is ever exempted
    by that check. The only other exemption (`CharScriptGaren.cs:108-111`) is
    ``unit.Stats.Level>=11 && UnitTag.Monster.Equals(Attacker.UnitTags)`` --
    the SAME Monster-value collision, gated on the DEFENDER's (victim's) own
    level, not the attacker's.

    Extracted as a small pure function (rather than left inline in
    ``step.py``) specifically so it is directly unit-testable without
    needing a live auto-attack to land -- a super minion's basic attack has
    Content ``MissileSpeed: 0`` (`Blue_Minion_MechMeleeBasicAttack.json`),
    which stalls `sim/missiles.py`'s travel-time model indefinitely and is a
    separate, pre-existing gap this function's tests must not depend on.
    """
    minion = xp.asarray(attacker_is_lane_minion)[:, None]
    cannon_or_super = xp.asarray(attacker_is_cannon_or_super)[:, None]
    level_ok = (xp.asarray(victim_level) >= 11)[None, :]
    return minion & (~cannon_or_super | level_ok)


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
                      attack_delay_offset_percent: Any, xp: Any = np) -> Any:
    """``Stats.cs:130``. ``global_attack_delay`` is ``gcd_AttackDelay`` (1.6 s).

    ``1 + attack_delay_offset_percent`` is exactly 0 for one real unit: the
    FOUNTAIN turret's Content ``AttackDelayOffsetPercent`` is **-1**
    (``Stats/OrderTurretShrine/OrderTurretShrine.json``, matched by
    ``ChaosTurretShrine``). C# float division by zero is `+Infinity`, not an
    exception -- the fountain's `AttackSpeedFlat` is meant to come out
    unboundedly fast. Divided through ``xp`` rather than Python's bare ``/``
    so that is what this returns too, instead of a `ZeroDivisionError` the
    moment the fountain's profile row is built. (Downstream, `attack_period`
    then reads `1/inf = 0` and `attack_windup` reads `0 * k = 0`, both of
    which are ordinary float arithmetic with no further zero denominators.)
    The fountain never actually fires in this slice regardless -- it is
    outside every unit's reach in a top-lane 1v1, see `sim.init.ALL_TURRETS`
    -- but building its stat row must not crash on the way there.
    """
    with np.errstate(divide="ignore"):
        return 1.0 / global_attack_delay / xp.asarray(1.0 + attack_delay_offset_percent)


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
