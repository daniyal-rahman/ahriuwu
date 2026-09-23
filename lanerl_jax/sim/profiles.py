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
from .combat import (attack_period, attack_speed_flat,
                      attack_windup, stat_at_level)
from .state import Kind, Team, TurretTier
from .targeting import MinionType

__all__ = ["PROFILES", "N_PROFILES", "TURRET_MODEL_NAME", "profile_id",
          "build_profile_tables", "LEVEL_ROWS"]

#: Rows in every level-indexed table: row ``L`` is champion level ``L``
#: (1..18), row 0 unused (`STRUCT-005`).
LEVEL_ROWS = 19

#: ``(kind, subtype, team)`` -> row. ``subtype`` is `MinionType` for a lane
#: minion, `TurretTier` for a turret, and -1 for a champion (there is only one
#: model). A turret's subtype used to be forced to -1 too -- one row per team,
#: shared by outer/inner/inhibitor/nexus/fountain alike -- which is the bug
#: `TurretTier` exists to fix; see its docstring in `sim.state`.
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
    (Kind.TURRET, TurretTier.OUTER, Team.BLUE),
    (Kind.TURRET, TurretTier.OUTER, Team.RED),
    (Kind.TURRET, TurretTier.INNER, Team.BLUE),
    (Kind.TURRET, TurretTier.INNER, Team.RED),
    (Kind.TURRET, TurretTier.INHIBITOR, Team.BLUE),
    (Kind.TURRET, TurretTier.INHIBITOR, Team.RED),
    (Kind.TURRET, TurretTier.NEXUS, Team.BLUE),
    (Kind.TURRET, TurretTier.NEXUS, Team.RED),
    (Kind.TURRET, TurretTier.FOUNTAIN, Team.BLUE),
    (Kind.TURRET, TurretTier.FOUNTAIN, Team.RED),
)
N_PROFILES = len(PROFILES)

_KEY_TO_ROW: Dict[Tuple[int, int, int], int] = {k: i for i, k in enumerate(PROFILES)}
_MINION_KEY = {
    MinionType.MELEE: "melee", MinionType.CASTER: "caster",
    MinionType.CANNON: "cannon", MinionType.SUPER: "super",
}
#: ``(team, tier)`` -> the Content model name that tier actually spawns on
#: Map1, per `TowerModels` (`Maps/Map1/LevelScriptObjects.cs:77-89`). Blue is
#: "Order", red is "Chaos", and red's names do NOT mirror blue's tier-for-tier
#: -- `ChaosTurretNormal` is red's NEXUS, not its outer, see
#: `data.patch.TURRET_MODELS`.
TURRET_MODEL_NAME: Dict[Tuple[int, int], str] = {
    (Team.BLUE, TurretTier.OUTER): "OrderTurretNormal",
    (Team.RED, TurretTier.OUTER): "ChaosTurretWorm",
    (Team.BLUE, TurretTier.INNER): "OrderTurretNormal2",
    (Team.RED, TurretTier.INNER): "ChaosTurretWorm2",
    (Team.BLUE, TurretTier.INHIBITOR): "OrderTurretDragon",
    (Team.RED, TurretTier.INHIBITOR): "ChaosTurretGiant",
    (Team.BLUE, TurretTier.NEXUS): "OrderTurretAngel",
    (Team.RED, TurretTier.NEXUS): "ChaosTurretNormal",
    (Team.BLUE, TurretTier.FOUNTAIN): "OrderTurretShrine",
    (Team.RED, TurretTier.FOUNTAIN): "ChaosTurretShrine",
}


def profile_id(kind: int, subtype: int, team: int) -> int:
    """Host-side lookup, for spawning. Raises on an unknown combination rather
    than defaulting -- a unit with the wrong stat row is a silent mechanic bug.

    ``subtype`` is a `MinionType` for a lane minion, a `TurretTier` for a
    turret, and ignored (forced to -1) for anything else.
    """
    key = (kind, subtype if kind in (Kind.LANE_MINION, Kind.TURRET) else -1, team)
    if key not in _KEY_TO_ROW:
        raise KeyError(f"no stat profile for kind={kind} type={subtype} team={team}")
    return _KEY_TO_ROW[key]


def _stats_for(patch: PatchTable, kind: int, mtype: int, team: int) -> UnitStats:
    if kind == Kind.CHAMPION:
        return patch.champion
    if kind == Kind.TURRET:
        return patch.turrets[TURRET_MODEL_NAME[(team, mtype)]]
    side = "blue" if team == Team.BLUE else "red"
    return patch.minions[f"{_MINION_KEY[mtype]}_{side}"]


def build_profile_tables(patch: PatchTable | None = None, dtype=jnp.float32) -> dict:
    """``(N_PROFILES,)`` arrays of every stat the tick needs."""
    patch = patch or load_patch()
    cols = {k: np.zeros(N_PROFILES, np.float32) for k in (
        "move_speed", "acquisition_range", "attack_range", "collision_radius",
        "attack_period", "attack_windup", "attack_damage", "armor",
        "magic_resist", "max_hp", "gold_on_death", "xp_on_death",
        "pathfinding_radius", "fires_missile", "missile_speed",
        "hp_regen", "hp_regen_per_level", "hp_per_level", "ad_per_level",
        "armor_per_level", "mr_per_level", "attack_speed_per_level",
        # `LaneTurret.Die` (`GameServerLib/GameObjects/AttackableUnits/AI/
        # LaneTurret.cs:37-88`) reads THESE fields, not the plain
        # `GoldGivenOnDeath`/`ExpGivenOnDeath` pair above -- zero for every
        # non-turret model. See `sim/rewards.turret_kill_rewards`.
        "local_gold_on_death", "global_gold_on_death", "global_xp_on_death",
        # `Stat.Armor.FlatBonus`'s share of the "armor" column above, kept
        # separate because Garen's W passive (`GarenWPassive.cs:34-37`)
        # composes `PercentBaseBonus`/`PercentBonus` around
        # `(BaseValue+BaseBonus)` and `FlatBonus` DIFFERENTLY -- see
        # `step.py`'s W-passive block. Confirmed this is really where a rune
        # lands: `Champion.OnAdded` applies each rune page entry as an ITEM
        # (`ItemData : StatsModifier`, `ItemData.cs:70`:
        # `Armor.FlatBonus = file.GetFloat("Data", "FlatArmorMod")`), not as
        # `BaseValue`/`BaseBonus`. Nonzero only for the champion rows
        # (`RUNE_ARMOR_BONUS`); zero for minions/turrets, whose armour has no
        # rune/item source in this project and for whom this column is never
        # read for anything (W passive is Garen-only).
        "armor_flat_bonus")}

    for row, (kind, mtype, team) in enumerate(PROFILES):
        u = _stats_for(patch, kind, mtype, team)
        flat = attack_speed_flat(patch.global_attack_delay,
                                 u.attack_delay_offset_percent)
        period = attack_period(flat)
        cols["move_speed"][row] = 0.0 if kind == Kind.TURRET else u.move_speed
        # Minions carry no AcquisitionRange in Content; the server falls back to
        # its own default rather than treating them as blind.
        # CharData.cs:98 -- server default is 475, not 600
        cols["acquisition_range"][row] = u.acquisition_range or 475.0
        cols["attack_range"][row] = u.attack_range
        # `GameObject.CollisionRadius` -- the TRIGGER radius `IsCollidingWith`
        # sums (`GameObject.cs:226-229`), NOT the same field as
        # `PathfindingRadius` below (that one is the RESOLUTION radius
        # `GetCircleEscapePoint` uses). `ObjAIBase`'s ctor
        # (`ObjAIBase.cs:105-116`) only falls back to
        # `CharData.GameplayCollisionRadius` (`u.collision_radius`, as loaded
        # by `data.patch`) when the caller passes a non-positive
        # `collisionRadius` argument -- true of `BaseTurret`, which passes
        # none at all, so turrets genuinely get their tier's CharData value.
        # It is NOT true of `Minion`/`Champion`: `Minion.cs:57` and
        # `Champion.cs:52` hard-code `40` and `30` into that same argument,
        # unconditionally, so Content's `GameplayCollisionRadius` is never
        # even read for them. Reading it anyway is how a cannon/super minion
        # (`GameplayCollisionRadius` 65 in Content) got a 65-unit trigger
        # radius instead of 40, and how Garen (`GameplayCollisionRadius` -1,
        # so `u.collision_radius` falls back to CharData's OWN 40 default)
        # got 40 instead of his real 30.
        cols["collision_radius"][row] = (
            40.0 if kind == Kind.LANE_MINION else
            30.0 if kind == Kind.CHAMPION else
            u.collision_radius)
        cols["pathfinding_radius"][row] = u.pathfinding_radius
        cols["attack_period"][row] = period
        cols["attack_windup"][row] = attack_windup(
            period, patch.global_attack_delay_cast_percent,
            u.attack_delay_cast_offset_percent)
        cols["attack_damage"][row] = u.base_ad
        # `Stats.LevelUp` (`GameServerLib/GameObjects/Stats/Stats.cs:270-271`)
        # grows `AttackDamage` on every level-up through the SAME non-linear
        # curve as every other per-level stat -- `combat.stat_at_level`, via
        # `AttackDamagePerLevel.BaseValue`/`.FlatBonus` both fed through
        # `GetLevelUpStatValue`. Only a champion has a level that moves inside
        # an episode (`rewards.level_for_xp`, applied in `step.py` only where
        # `kind == Kind.CHAMPION`); left 0 for minions/turrets/anything else so
        # `state.level` there -- always 1, see `state.py`'s init -- is inert
        # even if it were ever misread. See `step.py`'s `tick()` for where
        # this is added back on top of the level-1(+rune) baseline above.
        cols["ad_per_level"][row] = (
            u.ad_per_level if kind == Kind.CHAMPION else 0.0)
        # The observation exposes the server's live self stats.  Preserve the
        # per-level source values next to AD so it does not reconstruct armor
        # or MR through a second, drifting path.
        cols["armor_per_level"][row] = (
            u.armor_per_level if kind == Kind.CHAMPION else 0.0)
        cols["mr_per_level"][row] = (
            u.mr_per_level if kind == Kind.CHAMPION else 0.0)
        # `Stats.LevelUp` applies `GrowthAttackSpeed / 100` as an
        # `AttackSpeedMultiplier.PercentBaseBonus` on every champion level-up.
        # Keep the raw percentage here; `step.tick` applies the same non-linear
        # growth sum that it uses for AD, rather than freezing the level-one
        # period/windup baked above for an entire episode.
        cols["attack_speed_per_level"][row] = (
            u.attack_speed_per_level if kind == Kind.CHAMPION else 0.0)
        cols["hp_per_level"][row] = (
            u.hp_per_level if kind == Kind.CHAMPION else 0.0)
        cols["hp_regen_per_level"][row] = (
            u.hp_regen_per_level if kind == Kind.CHAMPION else 0.0)
        cols["armor"][row] = u.armor
        cols["magic_resist"][row] = u.magic_resist
        cols["max_hp"][row] = u.base_hp
        cols["gold_on_death"][row] = u.gold_given_on_death
        cols["xp_on_death"][row] = u.exp_given_on_death
        cols["local_gold_on_death"][row] = u.local_gold_given_on_death
        cols["global_gold_on_death"][row] = u.global_gold_given_on_death
        cols["global_xp_on_death"][row] = u.global_exp_given_on_death
        # `Spell.FinishCasting`: a basic attack becomes a missile when the
        # attacker is ranged AND its BasicAttack script is empty. In this
        # slice the second half is never false -- see `sim/missiles.py` for
        # why a lane turret does NOT get the exemption its own AD/armour once
        # (wrongly) implied -- so the rule collapses to `not IsMelee`.
        cols["fires_missile"][row] = float(not u.is_melee)
        # `SpellData.MissileSpeed` for THIS unit's own basic attack -- not a
        # shared constant. Meaningless (and unread, since `fires_missile` is 0)
        # for melee units.
        cols["missile_speed"][row] = u.missile_speed
        # `Stats.Update` adds `HealthRegeneration.Total * diff * 0.001f` on a
        # 500 ms accumulator, so this is HP per SECOND -- not per five seconds,
        # whatever League's display convention says. Champions scale it per
        # level through the same growth curve as every other per-level stat.
        # Minions are 0.0 on this map; Garen is 1.568 + 0.1 per level.
        # Turrets are NOT uniformly 0 -- that was only ever true of the outer
        # tier (`data.patch.TURRET_MODELS`'s regen measurement). INHIBITOR is
        # 3.0 and NEXUS is 6.0 in Content; neither could show up while every
        # turret shared the outer profile. Omitting champion regen entirely is
        # why our champion died 7 times in an oracle-driven 600 s episode where
        # the server's died 0 times.
        cols["hp_regen"][row] = (
            float(stat_at_level(u.base_hp_regen, u.hp_regen_per_level, 1))
            if kind == Kind.CHAMPION else float(u.base_hp_regen))

    # Measured deltas that Content does not carry; see sim/init.py.
    from .init import (
        MASTERY_AD_PER_LEVEL_BONUS, MASTERY_HP_FLAT_BONUS,
        MASTERY_HP_PERCENT_BONUS, RUNE_AD_BONUS, RUNE_ARMOR_BONUS,
        TURRET_HP_BONUS, TURRET_HP_BONUS_NEXUS)
    for row, (kind, tier, _) in enumerate(PROFILES):
        if kind == Kind.CHAMPION:
            # `Stat.Total = ((BaseValue + BaseBonus) * (1 + PercentBaseBonus)
            #                + FlatBonus) * (1 + PercentBonus)` (`Stat.cs:68`).
            # `Veteran's Scars` is the FlatBonus term and `Juggernaut` is the
            # outer PercentBonus, so the order here is the server's order and
            # not interchangeable with adding one measured number.
            cols["max_hp"][row] = (
                (cols["max_hp"][row] + MASTERY_HP_FLAT_BONUS)
                * (1.0 + MASTERY_HP_PERCENT_BONUS))
            # ...and the SAME multiplier applies to every level-up increment,
            # because `Stats.LevelUp` (`Stats.cs:267-268`) adds
            # `GetLevelUpStatValue(HealthPerLevel)` to `HealthPoints.BaseValue`
            # -- inside the percentage, not outside it. `step.py` grows max HP
            # as `hp_per_level * d(growth_sum)`, so the only place the factor
            # can live is folded into this column.
            #
            # This is the half of `STAT-001` that no level-1 check could ever
            # have caught, and it ran the other way from the level-1 error: the
            # sim gained 465.12 HP from level 1 to 7 where the server gains
            # 479.07. With the fold, the sim reproduces the server's dumped
            # ladder exactly at every level -- 671/743/817/895/977/1062/1150/
            # 1242 at levels 1..8, measured in `runs/g3_server.npz` and
            # `runs/tier15_noshop/drive_obs.jsonl`.
            #
            # One residual this cannot reach from here, stated rather than
            # hidden: `Stats.LevelUp` raises CURRENT health by the *unscaled*
            # `statsLevelUp.HealthPoints.BaseValue` (`Stats.cs:279`) while max
            # health rises by that times 1.03, so the server's champion drops
            # ~2.07 HP short of its new maximum on each level-up. `step.py`
            # adds one `hp_growth` to both, so the sim now over-credits current
            # HP by that much per level (12.4 HP by level 7, ~1% of the pool,
            # and refilled by regen within a couple of seconds). Fixing it
            # properly needs a second column read only by the current-HP term.
            cols["hp_per_level"][row] *= (1.0 + MASTERY_HP_PERCENT_BONUS)
            cols["attack_damage"][row] += RUNE_AD_BONUS
            # `Brute Force` raises the per-level SLOPE, not the level-1 value
            # -- `AttackDamagePerLevel` is a `Stat` and `Stats.LevelUp` runs
            # its `.BaseValue` and `.FlatBonus` through `GetLevelUpStatValue`
            # separately (`Stats.cs:270-271`). Both land in `AttackDamage`,
            # and `step.py` grows AD as `ad_per_level * d(growth_sum)`, so a
            # single summed slope column reproduces the server exactly. It
            # must NOT go in `attack_damage`: that would add 0.55 at level 1,
            # where the server adds nothing. See `STAT-002`.
            cols["ad_per_level"][row] += MASTERY_AD_PER_LEVEL_BONUS
            cols["armor"][row] += RUNE_ARMOR_BONUS
            # Doran's Shield's +80 max HP and +1.2 HP/s regen are deliberately
            # NOT here. See `init.DORANS_SHIELD_HP` for why, and for the one
            # line each that puts them back if the shop is ever turned on.
            # W's passive reads `Stat.Total`, which needs FlatBonus separated
            # from the base term -- the rune's armour lands in FlatBonus.
            cols["armor_flat_bonus"][row] = RUNE_ARMOR_BONUS
        elif kind == Kind.TURRET:
            # `OnMatchStart` (`:121-153`): every turret except the fountain
            # gets `HealthPoints.BaseBonus = 250 * enemyCount` (1v1: 250);
            # the NEXUS pair gets `125 * enemyCount` (1v1: 125) instead; the
            # FOUNTAIN is skipped by an explicit `continue` and gets neither.
            if tier == TurretTier.FOUNTAIN:
                pass
            elif tier == TurretTier.NEXUS:
                cols["max_hp"][row] += TURRET_HP_BONUS_NEXUS
            else:
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
    # ---- level-indexed tables (`STRUCT-005`) --------------------------------
    # ONE convention for every table keyed by champion level: `LEVEL_ROWS`
    # (19) rows, row `L` is the value FOR a champion at level `L` (1..18), row
    # 0 is unused. Readers index with the level itself, clipped to [1, 18] --
    # never `level - 1`, never `level + 1`. These tables used to have three
    # bases (level-1 for `xp_curve` and `death_times`, the latter with a `+1`
    # at the reader; level for `champion_kill_exp` and `RANKS_BY_LEVEL`), and
    # the respawn off-by-one (`eaa2e77`) was that mismatch. Each row is pinned
    # to its C# expression, for every level, in `tests/test_level_tables.py`.
    from ..data.patch import load_map_table
    exp_json = load_map_table("ExpCurve")
    # `mapData.ExpCurve` (`Package.cs:124-130`): `Level2`, `Level3`, ... in
    # order, so `ExpCurve[k]` is JSON `Level(k+2)`.
    exp_curve_cs = [float(exp_json["EXP"][f"Level{i}"])
                    for i in range(2, len(exp_json["EXP"]) + 2)]
    # `xp_to_reach_level[L]`: cumulative XP at which a champion BECOMES level
    # L. `Champion.AddExperience` levels L-1 -> L once
    # `Experience >= ExpCurve[(L-1) - 1]` (`Champion.cs:330,343`), so row L is
    # `ExpCurve[L-2]`; level 1 is where every champion starts, row 1 = 0.
    out["xp_to_reach_level"] = jnp.asarray(
        [0.0, 0.0] + [exp_curve_cs[L - 2] for L in range(2, LEVEL_ROWS)], dtype)
    # `death_times[L]`: seconds a champion that dies at level L stays dead.
    # `Champion.Die` reads `MapData.DeathTimes[Stats.Level]` (`Champion.cs:400`)
    # and `Package.cs:141-153` fills that list from `i = 1`, so
    # `DeathTimes[k]` is JSON `TimeDeadPerLevel.Level(k+1)`: row L is
    # `Level(L+1)` -- the `+1` lives in the table, not at the reader.
    dt_tbl = load_map_table("DeathTimes")["TimeDeadPerLevel"]
    death_times_cs = [float(dt_tbl[f"Level{i:02d}"])
                      for i in range(1, len(dt_tbl))]
    out["death_times"] = jnp.asarray(
        [death_times_cs[L] for L in range(LEVEL_ROWS)], dtype)
    # `champion_kill_exp[L]`: XP for killing a level-L champion, before the
    # level-difference adjustment. `Champion.Die` (`Champion.cs:444`):
    # `EXP = mapData.ExpCurve[Stats.Level - 1] * mapData.BaseExpMultiple`,
    # i.e. JSON `Level(L+1)` -- a level-1 victim is worth `Level2` = 280, a
    # level-18 victim `Level19` = 19060 -- times Map1's `BaseExpMultiple`
    # (0.55, `ExpCurve.json`'s `ExpGrantedOnDeath`). See
    # `sim/rewards.champion_kill_rewards`.
    base_exp_multiple = float(exp_json["ExpGrantedOnDeath"]["BaseExpMultiple"])
    out["champion_kill_exp"] = jnp.asarray(
        [0.0] + [exp_curve_cs[L - 1] * base_exp_multiple
                 for L in range(1, LEVEL_ROWS)], dtype)
    return out
