"""Unit stat profiles: one row per (kind, minion type, team).

Why profiles rather than per-slot constants
-------------------------------------------
A minion slot is reused by whatever spawns into it, so its stats cannot be baked
once at init -- the same array index is a melee minion in one wave and a cannon
in the next.  And the two sides are **not** interchangeable: measured from
Content, blue and red cannons differ in ``AttackRange`` (300 vs 280) and
``GoldGivenOnDeath`` (35 vs 30), and three of the four minion types differ in
attack wind-up.  A per-kind table would quietly average that away; a per-(kind,
type, **team**) table carries it.

So the state holds a small ``profile`` id per unit and the tick gathers stats
through it. One extra gather per stat per tick, and the asymmetry survives.
"""
from __future__ import annotations

from typing import Dict, Tuple

import jax.numpy as jnp
import numpy as np

from ..data.patch import PatchTable, UnitStats, load_patch
from .combat import attack_period, attack_speed_flat, attack_windup
from .state import Kind, Team
from .targeting import MinionType

__all__ = ["PROFILES", "N_PROFILES", "profile_id", "build_profile_tables"]

#: ``(kind, minion_type, team)`` -> row. ``minion_type`` is -1 for non-minions.
PROFILES: Tuple[Tuple[int, int, int], ...] = (
    (Kind.CHAMPION, -1, Team.BLUE),
    (Kind.CHAMPION, -1, Team.RED),
    (Kind.LANE_MINION, MinionType.MELEE, Team.BLUE),
    (Kind.LANE_MINION, MinionType.MELEE, Team.RED),
    (Kind.LANE_MINION, MinionType.CASTER, Team.BLUE),
    (Kind.LANE_MINION, MinionType.CASTER, Team.RED),
    (Kind.LANE_MINION, MinionType.CANNON, Team.BLUE),
    (Kind.LANE_MINION, MinionType.CANNON, Team.RED),
    (Kind.LANE_MINION, MinionType.SUPER, Team.BLUE),
    (Kind.LANE_MINION, MinionType.SUPER, Team.RED),
    (Kind.TURRET, -1, Team.BLUE),
    (Kind.TURRET, -1, Team.RED),
)
N_PROFILES = len(PROFILES)

_KEY_TO_ROW: Dict[Tuple[int, int, int], int] = {k: i for i, k in enumerate(PROFILES)}
_MINION_KEY = {
    MinionType.MELEE: "melee", MinionType.CASTER: "caster",
    MinionType.CANNON: "cannon", MinionType.SUPER: "super",
}


def profile_id(kind: int, minion_type: int, team: int) -> int:
    """Host-side lookup, for spawning. Raises on an unknown combination rather
    than defaulting -- a unit with the wrong stat row is a silent mechanic bug."""
    key = (kind, minion_type if kind == Kind.LANE_MINION else -1, team)
    if key not in _KEY_TO_ROW:
        raise KeyError(f"no stat profile for kind={kind} type={minion_type} team={team}")
    return _KEY_TO_ROW[key]


def _stats_for(patch: PatchTable, kind: int, mtype: int, team: int) -> UnitStats:
    if kind == Kind.CHAMPION:
        return patch.champion
    if kind == Kind.TURRET:
        return next(iter(patch.turrets.values()))
    side = "blue" if team == Team.BLUE else "red"
    return patch.minions[f"{_MINION_KEY[mtype]}_{side}"]


def build_profile_tables(patch: PatchTable | None = None, dtype=jnp.float32) -> dict:
    """``(N_PROFILES,)`` arrays of every stat the tick needs."""
    patch = patch or load_patch()
    cols = {k: np.zeros(N_PROFILES, np.float32) for k in (
        "move_speed", "acquisition_range", "attack_range", "collision_radius",
        "attack_period", "attack_windup", "attack_damage", "armor",
        "magic_resist", "max_hp", "gold_on_death", "xp_on_death",
        "pathfinding_radius")}

    for row, (kind, mtype, team) in enumerate(PROFILES):
        u = _stats_for(patch, kind, mtype, team)
        flat = attack_speed_flat(patch.global_attack_delay,
                                 u.attack_delay_offset_percent)
        period = attack_period(flat)
        cols["move_speed"][row] = 0.0 if kind == Kind.TURRET else u.move_speed
        # Minions carry no AcquisitionRange in Content; the server falls back to
        # its own default rather than treating them as blind.
        cols["acquisition_range"][row] = u.acquisition_range or 600.0
        cols["attack_range"][row] = u.attack_range
        cols["collision_radius"][row] = u.collision_radius
        cols["pathfinding_radius"][row] = u.pathfinding_radius
        cols["attack_period"][row] = period
        cols["attack_windup"][row] = attack_windup(
            period, patch.global_attack_delay_cast_percent,
            u.attack_delay_cast_offset_percent)
        cols["attack_damage"][row] = u.base_ad
        cols["armor"][row] = u.armor
        cols["magic_resist"][row] = u.magic_resist
        cols["max_hp"][row] = u.base_hp
        cols["gold_on_death"][row] = u.gold_given_on_death
        cols["xp_on_death"][row] = u.exp_given_on_death

    # Measured deltas that Content does not carry; see sim/init.py.
    from .init import (
        RUNE_AD_BONUS, RUNE_ARMOR_BONUS, RUNE_HP_BONUS, TURRET_HP_BONUS)
    for row, (kind, _, _) in enumerate(PROFILES):
        if kind == Kind.CHAMPION:
            cols["max_hp"][row] += RUNE_HP_BONUS
            cols["attack_damage"][row] += RUNE_AD_BONUS
            cols["armor"][row] += RUNE_ARMOR_BONUS
        elif kind == Kind.TURRET:
            cols["max_hp"][row] += TURRET_HP_BONUS

    # `LevelScript.Init`'s MinionModifiers are **NOT APPLIED** by the server.
    #
    # The dictionary exists and is populated (+20 HP / +1 AD / +3 armour on
    # melee, and so on), and its own declaration says why it does nothing:
    #
    #     //These minion modifiers will remain unused for the moment, untill i
    #     //pull the spawning systems to MapScripts
    #
    # I read that comment and applied them anyway. Caught by measurement: the
    # state dump's live-minion max-HP histogram is **290 / 455 / 700**, the raw
    # Content values, not the 297.5 / 475 / 727 the modifiers would produce.
    #
    # The cost of the mistake was not the HP alone. The extra armour (+3 melee,
    # +0.625 caster, +3 cannon) compounds with it, so minions were both tankier
    # and harder to damage, and the sim's steady-state minion population ran
    # **+19% above the server's** -- a discrepancy I had already written down as
    # "real and unexplained".
    #
    # Left here as a comment rather than deleted: when the server does pull the
    # spawning systems into MapScripts these become live, and the next person
    # should find the table rather than rediscover it.

    out = {k: jnp.asarray(v, dtype) for k, v in cols.items()}
    # `ExpCurve` thresholds, index i = XP needed to reach level i+1, so [0] is 0.
    out["xp_curve"] = jnp.asarray(
        [0.0] + [patch.xp_for_level(i) for i in range(2, 19)], dtype)
    # `DeathTimes.TimeDeadPerLevel`, seconds, indexed by level-1.
    from ..data.patch import load_map_table
    dt_tbl = load_map_table("DeathTimes")["TimeDeadPerLevel"]
    out["death_times"] = jnp.asarray(
        [float(dt_tbl[f"Level{i:02d}"]) for i in range(1, 19)], dtype)
    return out
