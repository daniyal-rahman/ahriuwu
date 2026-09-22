"""Build a starting lane state, and spawn waves into it.

Where the numbers come from, and why two sources
------------------------------------------------
**Rules** come from the C# source.  **Stats** come from the patch table
(``Content``).  **Geometry** comes from a recorded state dump, because the map
package format is a separate parser this project does not need: every turret,
nexus, inhibitor and spawn point is already in a ``LANERL_STATEROW`` snapshot at
``t=0``, at the server's own 1/16-unit resolution.  Reading the reference is
both cheaper and more trustworthy than re-deriving it.

The measured spawn deltas are the interesting part
--------------------------------------------------
Content is **not** the whole spawn state, and pretending otherwise starts every
episode with the wrong numbers:

=====================  ==========  ============  =============================
quantity               Content     observed      gap
=====================  ==========  ============  =============================
Garen max HP @ L1         616.28    671.848       +55.568  masteries (see below)
Garen attack damage        57.88     78.134766    +20.2548  runes + a mastery
Garen armor                27.536     36.536133    +9.0     runes
Garen magic resist         32.1      44.16        +12.06   runes -- NOT MODELLED
turret max HP            1300.0    1550.0        +250.0    outer-turret bonus
=====================  ==========  ============  =============================

Every one of those was checked back to its source rather than left as a
measured delta, and the exercise moved two of them (see `STAT-001`):

* **Max HP** is not a rune bonus and not a flat one. The page's runes carry no
  health at all; the +55.568 is ``Veteran's Scars`` (+36 flat) composed with
  ``Juggernaut`` (+3% of the whole stat), and the percentage applies to every
  per-level increment too. The old figure here was 754.248, which also carried
  an auto-bought Doran's Shield -- see :data:`DORANS_SHIELD_HP`.
* **Attack damage** decomposes as 9x0.945 + 3x2.25 = 15.2548 from the marks and
  quintessences, plus a flat 5.0 from ``Martial Mastery`` (talent 4132).
  ``Brute Force`` (talent 4122 rank 3) additionally adds 0.55 to
  ``AttackDamagePerLevel.FlatBonus``, which is a *slope*, not a constant, and
  lives in ``profiles.py``'s ``ad_per_level`` column.
* **Magic resist** is 9x1.34 from the glyphs and is **not modelled**: it lands
  in ``MagicResist.FlatBonus``, which ``step.py`` has no column for (its
  ``armor_flat_bonus`` twin exists only for armour), and Garen's W passive
  composes flat and base terms differently. Low impact in an all-physical
  mirror -- only Garen's R deals magic damage -- so it is recorded in the
  ledger rather than half-fixed here.

There is also a *staging* effect worth knowing: the champion's max HP reads
its pre-page value in the ``t=0`` snapshot and the full value once play starts,
because ``LanerlEpisode`` applies the page over the first ticks rather than at
construction. So "the value at t=0" and "the value during play" are different
questions, and this module targets the second.

These deltas are recorded as named constants with their provenance rather than
folded into the base stats, so that a modern-patch swap changes the base and
leaves the delta visible and separately checkable.
"""
from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..data.patch import PatchTable, load_patch
from .state import (
    CH_SLICE,
    MAX_WAYPOINTS,
    MI_SLICE,
    N_MINIONS,
    N_UNITS,
    TU_SLICE,
    Kind,
    LaneState,
    MoveOrder,
    Team,
    TurretTier,
    empty_state,
)
from .targeting import MinionType
from .waves import FIRST_WAVE_MS

__all__ = [
    "CHAMPION_SPAWN", "TOP_OUTER_TURRET", "ALL_TURRETS", "MINION_SPAWN",
    "TOP_LANE_PATH", "MASTERY_HP_FLAT_BONUS", "MASTERY_HP_PERCENT_BONUS",
    "MASTERY_AD_PER_LEVEL_BONUS",
    "TURRET_HP_BONUS", "TURRET_HP_BONUS_NEXUS",
    "lane_params", "init_lane", "spawn_minion",
]

# --- measured from a LANERL_STATEROW snapshot, 2026-09-16 -------------------
#: ``__Spawn_T1`` / ``__Spawn_T2``.
CHAMPION_SPAWN: Dict[int, Tuple[float, float]] = {
    Team.BLUE: (26.0, 280.0),
    Team.RED: (13927.0, 14175.0),
}
#: The two turrets that actually act in a TOPONLY 1v1. Blue's matches
#: ``lanerl_rl.constants.TOP_OUTER_TURRET`` exactly, which is a free cross-check
#: on the extraction.
TOP_OUTER_TURRET: Dict[int, Tuple[float, float]] = {
    Team.BLUE: (574.6, 10220.5),
    Team.RED: (3911.7, 13654.8),
}
#: **Every turret the server places**, measured from the same ``t = 0``
#: ``LANERL_STATEROW`` snapshot as the champion spawns, at its own 1/16-unit
#: resolution. 12 per team: 9 lane turrets (3 lanes x outer/inner/inhibitor),
#: 2 nexus turrets and 1 fountain turret.
#:
#: WHY ALL OF THEM, AND NOT JUST THE TOP OUTER PAIR
#: ------------------------------------------------
#: This was booked as "only the top outer pair can ever act in a TOPONLY 1v1",
#: and that is **false the moment a wave pushes**. Ten of these sit within 900
#: units of the top-lane polyline, five per side, and they are what bounds the
#: lane::
#:
#:     blue   0.000  0.016  0.109  0.217  0.388      (nexus x2, inhib, inner, outer)
#:     red    0.605  0.770  0.894  0.981  1.000
#:
#: as a fraction of the lane path. With only the outer pair modelled, a wave
#: that wins the midlane fight walks past the enemy outer turret at 0.388 and
#: meets **nothing at all** for the rest of the map. The sim ran away to 2 blue
#: minions against 28 red by ten minutes; the server, which has a turret at
#: 0.217 waiting, stays near 21 live minions with a p95 of 27.
#:
#: Positions and max HP are exact. Per-turret *combat* stats used to not be
#: distinguished -- every turret used the outer-lane profile, so the nexus
#: (1425 HP) and fountain (9999 HP) turrets shot like an outer turret. That
#: was a booked approximation for HP alone (max HP here is always the measured
#: value, never derived from a profile); it was NOT harmless for AD/armour,
#: because `LevelScriptObjects.OnUpdate` ramps non-outer tiers starting at
#: 480 s -- inside a 600 s episode -- and that schedule was simply not run.
#: See `sim.state.TurretTier`.
#:
#: RESOLVED 2026-09-16: a 5th field carries each placed turret's tier, so
#: `profile_id(Kind.TURRET, tier, team)` gives it that model's own AD/armour/
#: regen and its own ramp (`sim.combat.other_turret_ramps`). Tiers below are
#: NOT inferred from HP or position -- outer/inner/inhibitor/nexus/fountain
#: all disagree on HP (outer=inner=inhibitor=1550, nexus=1425, fountain=9999)
#: but nothing here distinguishes outer from inner from inhibitor by HP alone.
#: They come from cross-referencing this measured geometry against
#: `LevelScriptObjects.CreateBuildings`/`GetTurretType`
#: (`Maps/Map1/LevelScriptObjects.cs:294-393`) and the vendored map scene
#: files (`Maps/Map1/Scene/Turret_T{1,2}_{C,L,R}_NN.sco.json`,
#: ``CentralPoint.X``/``CentralPoint.Z``, which is exactly ``(x, y)`` here --
#: see `CreateBuildings`'s `new Vector2(turretObj.CentralPoint.X,
#: turretObj.CentralPoint.Z)`), matched to this table's positions at better
#: than 1 unit. Two things fall out of that cross-reference that are easy to
#: get wrong by inspection alone:
#:
#: * Team 1 (order/blue) has NO ``Turret_T1_L_01``/``Turret_T1_R_01`` --
#:   its top and bottom lane inhibitor turrets are ``Turret_T1_C_06`` and
#:   ``Turret_T1_C_07``, whose *type* is computed from `lane == LANE_C` (so
#:   `GetTurretType` resolves them as it would a THIRD "mid" inhibitor) and
#:   only afterwards re-labelled `LANE_L`/`LANE_R` by a `switch` on the raw
#:   object name (`:348-357`) -- type and lane are decided from two different
#:   pieces of state, computed in that order. Team 2 (chaos/red) has no such
#:   split: it carries its own ``Turret_T2_L_01``/``Turret_T2_R_01`` directly.
#: * Only 10 of these 24 (5 per side) sit within 900 units of
#:   `TOP_LANE_PATH` -- the top lane's own outer/inner/inhibitor plus both
#:   nexus turrets, at path fractions blue ``0.000 0.016 0.109 0.217 0.388``
#:   / red ``0.605 0.770 0.894 0.981 1.000`` (nexus x2, inhib, inner, outer,
#:   each team read towards its own base). The remaining 14 are mid- and
#:   bot-lane turrets that this top-lane-only sim can never bring a unit
#:   near; their tiers are exact (same cross-reference), not guessed, but
#:   are also provably inert here.
ALL_TURRETS: Tuple[Tuple[int, float, float, float, int], ...] = (
    (Team.BLUE, -236.0625, -53.3125, 9999.0, TurretTier.FOUNTAIN),
    (Team.BLUE, 574.625, 10220.5, 1550.0, TurretTier.OUTER),        # top
    (Team.BLUE, 802.8125, 4052.375, 1550.0, TurretTier.INHIBITOR),  # top (T1_C_06)
    (Team.BLUE, 1106.25, 6465.25, 1550.0, TurretTier.INNER),        # top
    (Team.BLUE, 1341.625, 2030.0, 1425.0, TurretTier.NEXUS),
    (Team.BLUE, 1768.1875, 1589.4375, 1425.0, TurretTier.NEXUS),
    (Team.BLUE, 3234.0, 3447.25, 1550.0, TurretTier.INHIBITOR),     # mid
    (Team.BLUE, 3747.25, 1041.0625, 1550.0, TurretTier.INHIBITOR),  # bot (T1_C_07)
    (Team.BLUE, 4657.0, 4591.9375, 1550.0, TurretTier.INNER),       # mid
    (Team.BLUE, 5448.375, 6169.125, 1550.0, TurretTier.OUTER),      # mid
    (Team.BLUE, 6512.5, 1262.625, 1550.0, TurretTier.INNER),        # bot
    (Team.BLUE, 10097.625, 808.75, 1550.0, TurretTier.OUTER),       # bot
    (Team.RED, 3911.6875, 13654.8125, 1550.0, TurretTier.OUTER),    # top
    (Team.RED, 7536.5, 13190.8125, 1550.0, TurretTier.INNER),       # top
    (Team.RED, 8548.8125, 8289.5, 1550.0, TurretTier.OUTER),        # mid
    (Team.RED, 9361.0625, 9892.625, 1550.0, TurretTier.INNER),      # mid
    (Team.RED, 10261.875, 13465.9375, 1550.0, TurretTier.INHIBITOR),  # top
    (Team.RED, 10743.5625, 11010.0625, 1550.0, TurretTier.INHIBITOR),  # mid
    (Team.RED, 12118.125, 12876.625, 1425.0, TurretTier.NEXUS),
    (Team.RED, 12662.5, 12442.6875, 1425.0, TurretTier.NEXUS),
    (Team.RED, 12920.8125, 8005.3125, 1550.0, TurretTier.INNER),    # bot
    (Team.RED, 13205.8125, 10474.625, 1550.0, TurretTier.INHIBITOR),  # bot
    (Team.RED, 13459.625, 4284.25, 1550.0, TurretTier.OUTER),       # bot
    (Team.RED, 14157.0, 14456.375, 9999.0, TurretTier.FOUNTAIN),
)

#: Best-effort creation-order PRIORITY per turret tier, used only to seed
#: `LaneState.spawn_seq` (see its docstring in `sim.state`). Reconstructed
#: from `LevelScriptObjects.CreateBuildings`
#: (`Maps/Map1/LevelScriptObjects.cs:294-361`), which runs three separate
#: loops, in this order: every nexus, then every inhibitor, then a single
#: loop over `_mapObjects[GameObjectTypes.ObjAIBase_Turret]` that creates the
#: outer/inner/fountain turrets together, in whatever order the map's own
#: scene file lists them -- a file this project has not parsed, so the
#: relative order WITHIN that third loop (outer vs. inner vs. fountain, and
#: between two turrets of the same tier) is an unverified guess, flagged
#: rather than hidden, exactly like the wave-spawn blue/red tie-break
#: `parity.tier1_collision_sequential.estimate_creation_order` already
#: names. It is also low-stakes here: `ALL_TURRETS`' own docstring measures
#: that only the two top-lane OUTER turrets ever sit within collision range
#: of a live unit in this slice, so two turrets contesting the same third
#: object at once essentially never happens.
_TURRET_CREATION_PRIORITY: Dict[int, int] = {
    TurretTier.NEXUS: 0,
    TurretTier.INHIBITOR: 1,
    TurretTier.OUTER: 2,
    TurretTier.INNER: 2,
    TurretTier.FOUNTAIN: 2,
}

#: Lane-minion barracks: ``CentralPoint.X/Z`` from Map1's
#: ``__P_{Order,Chaos}_Spawn_Barracks__L01.sco.json``.  Do not round these to
#: the canonical dump's 1/16-unit wire grid: `CreateLaneMinion` receives the
#: source ``float`` coordinates directly, and the sub-unit component is
#: visible on every freshly spawned minion before any collision can occur.
MINION_SPAWN: Dict[int, Tuple[float, float]] = {
    Team.BLUE: (917.7302, 1720.3623),
    Team.RED: (12451.0508, 13217.5420),
}
#: ``LanerlLane.TopLaneDefault`` -- taken verbatim from the map script's
#: ``MinionPaths`` so minions walk the same line the server walks them along.
TOP_LANE_PATH: Tuple[Tuple[float, float], ...] = (
    (917.0, 1725.0), (1170.0, 4041.0), (861.0, 6459.0), (880.0, 10180.0),
    (1268.0, 11675.0), (2806.0, 13075.0), (3907.0, 13243.0), (7550.0, 13407.0),
    (10244.0, 13238.0), (10947.0, 13135.0), (12511.0, 12776.0),
)

#: Champion max HP above the Content base curve. **Not runes** -- the rune page
#: in ``lanerl/cfg/garen1v1.json`` grants exactly zero health. Read the four
#: rune items it actually lists (``Content/LeagueSandbox-Default/Items/<id>``):
#: 9x ``5245`` Greater Mark of Attack Damage (``FlatPhysicalDamageMod`` 0.945),
#: 9x ``5317`` Greater Seal of Armor (``FlatArmorMod`` 1), 9x ``5289`` Greater
#: Glyph of Magic Resist (``FlatSpellBlockMod`` 1.34), 3x ``5335`` Greater
#: Quintessence of Attack Damage (``FlatPhysicalDamageMod`` 2.25). No HP
#: anywhere. Every point of the champion's extra health is a **mastery**, and
#: there are two of them -- which matters, because they enter ``Stat.Total``
#: (``Stat.cs:68``) at different places::
#:
#:     Total = ((BaseValue + BaseBonus) * (1 + PercentBaseBonus)
#:              + FlatBonus) * (1 + PercentBonus)
#:
#: ``Veteran's Scars`` (talent ``4222``, rank 3 in the config) is
#: ``HealthPoints.FlatBonus = 12.0f * rank`` and ``Juggernaut`` (talent
#: ``4232``) is ``HealthPoints.PercentBonus = 0.03f``
#: (``Content/LeagueSandbox-Scripts/Talents/Defense/``). Of the sixteen talents
#: the config lists only four have scripts at all; the rest resolve to
#: ``EmptyTalentScript`` and do nothing.
#:
#: The percentage is the whole reason this is two constants rather than one
#: measured delta. ``Stats.LevelUp`` adds its increment to
#: ``HealthPoints.BaseValue``, so ``PercentBonus`` scales **every per-level
#: increment too** -- which is where the old single ``RUNE_HP_BONUS`` went
#: wrong in a way no level-1 check could catch. See ``profiles.py``, which
#: folds it into the ``hp_per_level`` column for exactly this reason, and
#: `STAT-001` in the fidelity ledger for the measurement.
MASTERY_HP_FLAT_BONUS = 36.0        # Veteran's Scars, talent 4222 rank 3
MASTERY_HP_PERCENT_BONUS = 0.03     # Juggernaut, talent 4232
#: Doran's Shield (item ``1054``, ``BuildPath[0]``), for reference only:
#: **neither of these is applied.** The sim has no item model at all
#: (`ITEM-001`, `SCOPE-001`), and every server-side parity instrument now runs
#: the shop off -- the gate-3 driver (``parity/tests/test_last_hit_gate.py``
#: passes ``autobuy=False``), the Tier-1.5 fixture (``--no-autobuy``),
#: ``hp_band.run_server_band`` and ``isolation`` all default it off. So the
#: reference the sim must match is the **item-free** champion, and carrying
#: one item's two stats while modelling none of the other seven build-path
#: items is worse than carrying none.
#:
#: This is a correction, not a preference. ``RUNE_HP_BONUS`` used to be
#: ``754.248046875 - 616.28``, read from a dump taken with the shop ON, and
#: 754.248046875 is exactly ``(616.28 + 36) * 1.03 + 80 * 1.03`` -- i.e. it
#: silently carried this item's ``FlatHPPoolMod``. Measured both ways in the
#: same fixture pair: ``runs/tier15/drive_obs.jsonl`` (shop on) reports the
#: champion at 754 max HP in its very first frame, and
#: ``runs/tier15_noshop/`` reports 671, as does the gate-3 server trace
#: ``runs/g3_server.npz`` for all 600 s.
#:
#: If the shop is ever turned back on for a parity or training run, both go
#: back: ``+ DORANS_SHIELD_HP * (1 + MASTERY_HP_PERCENT_BONUS)`` on the
#: ``max_hp`` column (``FlatHPPoolMod`` lands in ``HealthPoints.FlatBonus``,
#: ``ItemData.cs:81``, *inside* Juggernaut's multiplier) and
#: ``+ DORANS_SHIELD_HP_REGEN`` on ``hp_regen``
#: (``ItemPassives/DoransShield.cs``: ``HealthRegeneration.BaseBonus +=
#: 1.2f``, and nothing scales regen by percentage here).
DORANS_SHIELD_HP = 80.0
DORANS_SHIELD_HP_REGEN = 1.2
#: Non-nexus, non-fountain turret max HP above Content. 1550 observed vs 1300
#: BaseHP -- true of the outer, inner AND inhibitor tiers alike, since all
#: three share BaseHP 1300 (see `data/patch.TURRET_MODELS`).
#:
#: `LevelScriptObjects.OnMatchStart` (`:121-153`... `:145`) sets
#: `HealthPoints.BaseBonus = 250.0f * Players[enemyTeam].Count` for every
#: turret except the nexus pair (`TURRET_HP_BONUS_NEXUS`) and the fountain,
#: which is skipped by an explicit `continue` and gets no HP bonus at all
#: (`ALL_TURRETS`'s fountain entries are the bare Content 9999). In this
#: project's 1v1, `enemyTeam.Count == 1`, so the multiplier drops out and 250
#: is the bonus outright -- it would need to change if the slice ever grows
#: past a 1v1.
#:
#: The bonus is unchanged by the Map11-vs-Map1 turret correction only because
#: both candidates happen to carry BaseHP 1300 -- which is exactly why that
#: mix-up survived every cross-check the turret had. See
#: `data/patch.TURRET_MODELS`.
TURRET_HP_BONUS = 250.0
#: The nexus pair's own bonus (`OnMatchStart:149`):
#: `HealthPoints.BaseBonus = 125.0f * Players[enemyTeam].Count`, again with
#: the multiplier at 1 in a 1v1. 1300 + 125 = 1425, `ALL_TURRETS`'s measured
#: nexus HP.
TURRET_HP_BONUS_NEXUS = 125.0

#: The rest of the rune page, measured the same way -- from the dump's own
#: quantised values in a 600 s idle run, against the Content base.
#:
#: `lanerl_rl/constants.py` states the lesson these exist to avoid, having paid
#: for it: *"constants.garen_attack_damage() read 57.88 at level 1, then 73.14
#: once the rune page was modelled, against a true 78.14 -- a re-derived server
#: quantity is wrong by however much of the server you forgot."* So these are
#: **measured deltas**, not a reconstruction of which runes the page contains.
#:
#: Both have since been traced back to source and agree exactly, which is why
#: they are left as measured deltas: AD is 9 x 0.945 (marks ``5245``) + 3 x 2.25
#: (quints ``5335``) = 15.2548 of rune, plus ``Martial Mastery``'s flat 5.0
#: (talent ``4132``), = 20.2548; armour is 9 x 1.0 from the seals (``5317``),
#: = 9.0, with no armour talent scripted at all.
RUNE_AD_BONUS = 78.134765625 - 57.88            # +20.2548
RUNE_ARMOR_BONUS = 36.5361328125 - 27.5361328125  # +9.0

#: `Brute Force` (talent `4122`, rank 3 in `lanerl/cfg/garen1v1.json`) does
#: NOT add flat AD -- it adds to the champion's AD *slope*.
#: `Talents/Offense/Brute Force.cs` writes
#: `StatsModifier.AttackDamagePerLevel.FlatBonus = statPerRank[rank-1]`
#: (`{0.22, 0.39, 0.55}`), and `AttackDamagePerLevel` is itself a full `Stat`,
#: so `Stats.LevelUp` grows AD through *both* of its terms
#: (`Stats.cs:270-271`): `AttackDamage.BaseValue` from
#: `GetLevelUpStatValue(AttackDamagePerLevel.BaseValue)` and
#: `AttackDamage.FlatBonus` from `GetLevelUpStatValue(...FlatBonus)`. The
#: server's Garen therefore gains `(3.5 + 0.55) * growth(L)` per level.
#:
#: This is why level 1 looked exact while every later level did not: the
#: talent contributes nothing at level 1 and `+0.55 * growth_sum(L)`
#: thereafter -- +0.40 AD at level 2 rising to +2.66 (3.4%) at level 7.
#: `PORT_AUDIT_COMBAT` checked `AttackDamagePerLevel` for item modifiers,
#: found none in scope and concluded only `.BaseValue` was nonzero; true of
#: items, false of masteries. See `STAT-002`.
MASTERY_AD_PER_LEVEL_BONUS = 0.55   # Brute Force, talent 4122 rank 3


def lane_params(patch: PatchTable | None = None, dtype=jnp.float32) -> dict:
    """The stat tables the tick gathers through ``state.model``.

    Just :func:`lanerl_jax.sim.profiles.build_profile_tables`; kept as a named
    entry point because the tick's contract is "params + state", and callers
    should not have to know which module the rows come from.
    """
    from .profiles import build_profile_tables

    return build_profile_tables(patch, dtype)


def _legacy_lane_params(patch: PatchTable | None = None, dtype=jnp.float32) -> dict:
    from .combat import attack_period, attack_speed_flat, attack_windup

    patch = patch or load_patch()
    g = patch.champion
    melee = patch.minions["melee_blue"]
    turret = next(iter(patch.turrets.values()))

    n = N_UNITS
    z = lambda v: np.full(n, v, dtype=np.float32)   # noqa: E731

    def period_windup(u):
        flat = attack_speed_flat(patch.global_attack_delay,
                                 u.attack_delay_offset_percent)
        p = attack_period(flat)
        return p, attack_windup(p, patch.global_attack_delay_cast_percent,
                                u.attack_delay_cast_offset_percent)

    gp, gw = period_windup(g)
    mp, mw = period_windup(melee)
    tp, tw = period_windup(turret)

    move = z(melee.move_speed); acq = z(melee.acquisition_range or 475.0)
    rng = z(melee.attack_range); col = z(melee.collision_radius or 48.0)
    per = z(mp); win = z(mw); ad = z(melee.base_ad); ar = z(melee.armor)

    for sl, u, (p_, w_) in ((CH_SLICE, g, (gp, gw)), (TU_SLICE, turret, (tp, tw))):
        move[sl] = u.move_speed
        acq[sl] = u.acquisition_range or 475.0
        rng[sl] = u.attack_range
        col[sl] = u.collision_radius or 48.0
        per[sl] = p_
        win[sl] = w_
        ad[sl] = u.base_ad
        ar[sl] = u.armor
    move[TU_SLICE] = 0.0        # turrets never move

    mt = np.full(n, MinionType.MELEE, np.int8)
    return {
        "move_speed": jnp.asarray(move, dtype),
        "acquisition_range": jnp.asarray(acq, dtype),
        "attack_range": jnp.asarray(rng, dtype),
        "collision_radius": jnp.asarray(col, dtype),
        "attack_period": jnp.asarray(per, dtype),
        "attack_windup": jnp.asarray(win, dtype),
        "attack_damage": jnp.asarray(ad, dtype),
        "armor": jnp.asarray(ar, dtype),
        "minion_type": jnp.asarray(mt),
    }


def init_lane(patch: PatchTable | None = None, dtype=jnp.float32,
              seed: int = 0, include_all_turrets: bool = True) -> LaneState:
    """A fresh top-lane 1v1 at ``t = 0``: two champions, the turrets, no minions.

    ``include_all_turrets`` defaults **on**, and used to default off with the
    reasoning that "only the top outer pair can ever act in this scenario".
    That reasoning was wrong, and wrong in a way that cost 21% of the minion
    population -- see :data:`ALL_TURRETS`. Five turrets per side sit on the top
    lane, and the ones behind the outer turret are what stops a winning wave
    from marching into the enemy base unopposed.

    Pass ``False`` for an isolated arena when a test wants to place units
    without a turret shooting at them. It is a test convenience, not a model of
    the server: the server always has all 24.
    """
    patch = patch or load_patch()
    s = empty_state(dtype=dtype, seed=seed)
    n = N_UNITS

    kind = np.zeros(n, np.int8)
    team = np.full(n, Team.NEUTRAL, np.int8)
    alive = np.zeros(n, bool)
    x = np.zeros(n, np.float32)
    y = np.zeros(n, np.float32)
    hp = np.zeros(n, np.float32)

    from .profiles import profile_id
    model = np.zeros(n, np.int8)

    champ_hp = ((patch.champion.hp_at_level(1) + MASTERY_HP_FLAT_BONUS)
                * (1.0 + MASTERY_HP_PERCENT_BONUS))
    for i, t in enumerate((Team.BLUE, Team.RED)):
        kind[i] = Kind.CHAMPION
        team[i] = t
        model[i] = profile_id(Kind.CHAMPION, -1, t)
        x[i], y[i] = CHAMPION_SPAWN[t]
        hp[i] = champ_hp
        alive[i] = True

    # `spawn_seq`: the map's own objects (nexuses, inhibitors, turrets) are
    # created by `Map.Init()`, which `Game.Initialize` runs BEFORE its
    # `PlayerManager.AddPlayer` loop constructs either champion
    # (`GameServerLib/Game.cs:142-178`; `AddPlayer` itself calls
    # `ObjectManager.AddObject` on the new `Champion` immediately,
    # `Players/PlayerManager.cs:27-66`) -- so turrets rank before BOTH
    # champions, unconditionally, not just "both before any minion". Filled
    # in below once each unit's slot is known; every minion born later gets a
    # higher rank still, from `next_spawn_seq` (see `spawn_minion`).
    spawn_seq = np.zeros(n, np.int32)

    t0 = TU_SLICE.start
    if include_all_turrets:
        placed = list(ALL_TURRETS)
    else:
        # The isolated arena's one turret per side is an OUTER turret -- it is
        # the one `TOP_OUTER_TURRET` names, so it gets `TurretTier.OUTER`
        # rather than the placeholder -1 the pre-tier code used. -1 is no
        # longer a turret subtype `profile_id` accepts at all (every turret
        # row now needs a real tier), so this is not optional.
        from .profiles import TURRET_MODEL_NAME
        base = ({t: patch.turrets[TURRET_MODEL_NAME[(t, TurretTier.OUTER)]].base_hp
                + TURRET_HP_BONUS for t in (Team.BLUE, Team.RED)})
        placed = [(t, *TOP_OUTER_TURRET[t], base[t], TurretTier.OUTER)
                 for t in (Team.BLUE, Team.RED)]
    assert len(placed) <= TU_SLICE.stop - t0, (
        f"{len(placed)} turrets into {TU_SLICE.stop - t0} slots")
    for j, (t, tx, ty, thp, tier) in enumerate(placed):
        i = t0 + j
        kind[i] = Kind.TURRET
        team[i] = t
        model[i] = profile_id(Kind.TURRET, tier, t)
        x[i], y[i] = tx, ty
        hp[i] = thp
        alive[i] = True

    # Turrets first (see `_TURRET_CREATION_PRIORITY`), by (tier priority,
    # team, list position) -- the last two are a stable, deterministic
    # tie-break where the true sub-order is unverified, not a claim that
    # blue-before-red or `ALL_TURRETS`' own ordering is the server's.
    turret_rank = sorted(
        range(len(placed)),
        key=lambda j: (_TURRET_CREATION_PRIORITY[placed[j][4]], placed[j][0], j))
    for rank, j in enumerate(turret_rank):
        spawn_seq[t0 + j] = rank
    n_turrets_placed = len(placed)
    # Then the two champions, in `Config.Players` order -- (blue, red)
    # throughout this project, see `CHAMPION_SPAWN`.
    spawn_seq[0] = n_turrets_placed
    spawn_seq[1] = n_turrets_placed + 1

    return s.replace(
        model=jnp.asarray(model),
        spawn_x=jnp.asarray(x, dtype), spawn_y=jnp.asarray(y, dtype),
        kind=jnp.asarray(kind), team=jnp.asarray(team), alive=jnp.asarray(alive),
        x=jnp.asarray(x, dtype), y=jnp.asarray(y, dtype),
        collision_x=jnp.asarray(x, dtype), collision_y=jnp.asarray(y, dtype),
        collision_present=jnp.asarray(alive & (kind != Kind.NONE)),
        hp=jnp.asarray(hp, dtype), max_hp=jnp.asarray(hp, dtype),
        next_spawn_ms=jnp.asarray(FIRST_WAVE_MS, dtype),
        move_order=jnp.full((n,), MoveOrder.NONE, jnp.int8),
        spawn_seq=jnp.asarray(spawn_seq),
        next_spawn_seq=jnp.asarray(n_turrets_placed + 2, dtype=jnp.int32),
    )


def spawn_minion(state: LaneState, team, profile, hp,
                 path: jax.Array, enabled=True, spawn_xy=None) -> LaneState:
    """Write one minion into the lowest free minion slot.

    Lowest free slot, not a random or round-robin one: the server's target
    acquisition breaks ties by object-collection order and ``argmin`` takes the
    lowest index, so slot assignment is part of the parity surface (see
    ``LaneState``'s docstring). Deterministic assignment keeps the two orders
    comparable.

    Silently does nothing when every slot is taken. That is the one place a
    fixed-shape sim can lose a unit, so the caller must watch it -- the measured
    p99 is 27 live minions against 40 slots, but a cap is a promise about a
    distribution, not a guarantee.
    """
    free = (~state.alive) & (jnp.arange(N_UNITS) >= MI_SLICE.start) \
        & (jnp.arange(N_UNITS) < MI_SLICE.stop)
    i = jnp.argmax(free)
    # `enabled` lets the caller invoke this every tick unconditionally: under
    # `vmap` a `cond` executes both branches anyway, so a masked write is the
    # same cost and keeps the cost visible.
    ok = jnp.any(free) & jnp.asarray(enabled)

    # WHERE a minion appears is NOT the first vertex of its path.
    #
    # It was, and for blue that is nearly true -- `TOP_LANE_PATH[0]` is
    # (917, 1725) against a measured barracks of (918, 1720), 5.1 units out.
    # For RED it is not true at all. Red walks the path reversed, so `path[0]`
    # is `TOP_LANE_PATH[-1]` = (12511, 12776), and the measured red barracks is
    # (12451, 13218) -- **446 units away**.
    #
    # That asymmetry is a persistent head start. Red minions began their march
    # ~446 units further down the lane than the server puts them (growing to
    # ~700 as they converge onto the polyline), arriving about 2.15 s early,
    # every wave, forever. Waves therefore met on blue's side of where the
    # server has them meet, blue fought at a standing disadvantage, and the
    # lane ran away to red: by ten minutes the sim held ~2 blue minions against
    # ~28 red, where the server oscillates around the middle all game with a
    # mean imbalance of 2.6.
    #
    # It hid because the sim was self-consistent. Both sides spawned at their
    # own path end, so the sim's own first-wave clash landed at lane fraction
    # 0.500 -- perfectly symmetric, and measured as such -- while being the
    # wrong geometry. Checking a simulation against itself cannot find this;
    # only the dump can.
    sx = path[0, 0] if spawn_xy is None else jnp.asarray(spawn_xy[0], state.x.dtype)
    sy = path[0, 1] if spawn_xy is None else jnp.asarray(spawn_xy[1], state.y.dtype)
    # `StopMovement()` leaves a one-point route at the actual spawn position.
    # The immutable `path` belongs to LaneMinionAI, not this transient route;
    # its first reevaluation notices the destination differs from
    # PathingWaypoints[0] and installs the two-point movement path.
    wp = jnp.zeros((MAX_WAYPOINTS, 2), state.x.dtype).at[0].set(
        jnp.stack([sx, sy]))

    def setv(arr, v):
        return jnp.where(ok, arr.at[i].set(v), arr)

    # `spawn_seq`: the server creates a genuinely new `Minion` GameObject per
    # wave spawn (the old occupant of a recycled slot was a DIFFERENT object,
    # already `RemoveObject`d on death) -- so this slot's creation rank is
    # "whatever the running counter is now", not anything derived from the
    # slot index. `next_spawn_seq` only advances when a minion is actually
    # written (`ok`), exactly like every other masked write here.
    return state.replace(
        kind=setv(state.kind, jnp.int8(Kind.LANE_MINION)),
        team=setv(state.team, jnp.asarray(team, jnp.int8)),
        alive=setv(state.alive, True),
        model=setv(state.model, jnp.asarray(profile, jnp.int8)),
        spawn_seq=setv(state.spawn_seq, state.next_spawn_seq.astype(jnp.int32)),
        x=setv(state.x, sx), y=setv(state.y, sy),
        collision_x=setv(state.collision_x, sx),
        collision_y=setv(state.collision_y, sy),
        collision_present=setv(state.collision_present, True),
        hp=setv(state.hp, jnp.asarray(hp, state.hp.dtype)),
        max_hp=setv(state.max_hp, jnp.asarray(hp, state.hp.dtype)),
        # LaneMinion's constructor calls StopMovement()/sets Hold. Its AI's
        # first 250-ms-immediate reevaluation installs the route to
        # PathingWaypoints[0]; it does NOT begin by following the complete
        # immutable lane list as one movement path.
        waypoints=jnp.where(ok, state.waypoints.at[i].set(wp), state.waypoints),
        n_waypoints=setv(state.n_waypoints, jnp.int8(1)),
        waypoint_key=setv(state.waypoint_key, jnp.int8(1)),
        lane_waypoint_key=setv(state.lane_waypoint_key, jnp.int8(0)),
        move_order=setv(state.move_order, jnp.int8(MoveOrder.HOLD)),
        target=setv(state.target, jnp.int8(-1)),
        ai_timer=setv(state.ai_timer, jnp.asarray(250.0, state.x.dtype)),
        next_spawn_seq=jnp.where(ok, state.next_spawn_seq + 1,
                                 state.next_spawn_seq),
    )
