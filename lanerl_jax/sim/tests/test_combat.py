"""Damage modifiers that are properties of the attacker/target PAIR.

A stat table cannot hold these: the same turret, on the same tick, deals
different damage to a minion and to a champion. They live in the basic-attack
scripts on the server, which is why they are easy to miss when porting from the
stat JSON -- and missing one is worth 43% of a turret's output against a wave.
"""
from __future__ import annotations

import numpy as np
import pytest



def test_this_maps_turrets_do_NOT_discount_damage_to_minions():
    """The 0.7 is real, on a map we do not run. This pins that it stays off.

    `SRUAP_Turret_Order3/BasicAttack.cs` does `if (target is Minion) dmg *=
    0.7f`, and I applied it to every turret on the strength of having read it.
    Scripts resolve by CHARACTER NAME: this config pins map 1, whose turrets
    are `OrderTurretNormal`/`ChaosTurretWorm`, which have no `Characters/`
    folder and so fall back to `SpellScriptEmpty` and full-AD native damage.
    `SRUAP_Turret_*` are Map11 units.

    Verifying a script exists is not verifying it runs. Three measurements on
    a 600 s recording identify the model: zero HP regen in 36,001 snapshots,
    turret HP drops landing on exactly 7.500/14.375/25.000 (armour 60, not
    67), and the config's own map id.
    """
    import jax.numpy as jnp
    from lanerl_jax.sim.combat import TURRET_DAMAGE_VS_MINION
    from lanerl_jax.sim.state import Kind
    from lanerl_jax.sim.step import _attack_damage_against

    assert TURRET_DAMAGE_VS_MINION == 1.0
    ad = jnp.asarray([190.0, 190.0, 190.0, 78.0])
    attacker = jnp.asarray([Kind.TURRET, Kind.TURRET, Kind.CHAMPION,
                            Kind.LANE_MINION], jnp.int8)
    target = jnp.asarray([Kind.LANE_MINION, Kind.CHAMPION, Kind.LANE_MINION,
                          Kind.LANE_MINION], jnp.int8)
    got = _attack_damage_against(ad, attacker, target)
    assert float(got[0]) == pytest.approx(190.0)   # turret -> minion: NOT discounted
    assert float(got[1]) == pytest.approx(190.0)   # turret -> champion: full
    assert float(got[2]) == pytest.approx(190.0)   # champion -> minion: full
    assert float(got[3]) == pytest.approx(78.0)    # minion  -> minion: full


def test_every_unit_model_is_one_this_map_actually_spawns():
    """The check that would have caught the wrong turret before it shipped.

    A Content stat or script is only real if the CHARACTER NAME it belongs to
    is the unit the configured map spawns. `lanerl/cfg/garen1v1.json` pins
    `"map": 1`, and Map1's scripts name every unit explicitly:

        LevelScript.cs:27-36          minions, per team
        LevelScriptObjects.cs:77-89   turrets, per tier and team

    The minion names were right. The turret names were Map11's
    (`SRUAP_Turret_Order3`/`Chaos3`), which this map never spawns -- and the
    error hid because the two candidates agree on BaseHP 1300 and range 750,
    the two turret numbers the sim had already matched against a dump.

    Pinned here as names rather than as stats so that a patch-table swap moves
    the values and still has to name a unit that exists on the map.
    """
    from lanerl_jax.data.patch import MINION_MODELS, TURRET_MODELS

    assert MINION_MODELS["melee"] == ("Blue_Minion_Basic", "Red_Minion_Basic")
    assert MINION_MODELS["caster"] == ("Blue_Minion_Wizard", "Red_Minion_Wizard")
    assert MINION_MODELS["cannon"] == ("Blue_Minion_MechCannon",
                                       "Red_Minion_MechCannon")
    # All five tiers, both teams. Red's OUTER is ChaosTurretWorm -- NOT
    # ChaosTurretNormal, which is chaos's *nexus* turret (AD 180, armour 65,
    # regen 6). Pairing the teams by matching names picks the wrong unit.
    assert TURRET_MODELS == (
        "OrderTurretNormal", "ChaosTurretWorm",
        "OrderTurretNormal2", "ChaosTurretWorm2",
        "OrderTurretDragon", "ChaosTurretGiant",
        "OrderTurretAngel", "ChaosTurretNormal",
        "OrderTurretShrine", "ChaosTurretShrine",
    )


def test_the_turret_stats_are_the_ones_measured_off_the_server():
    """The OUTER tier is AD 152 / armour 60 / regen 0, confirmed against a
    600 s recording -- and every OTHER tier is confirmed to be genuinely
    DIFFERENT, per `data.patch.TURRET_MODELS`'s Content table.

    * armour 60: single-step turret HP drops land on exactly 7.500 (melee,
      93x), 14.375 (caster, 68x) and 25.000 (cannon, 4x) -- minion AD times
      100/160. Armour 67 predicts 7.186 / 13.772 / 23.952 and not one drop
      matched.
    * regen 0: turret HP was monotone over all 36,001 snapshots. `Stats.Update`
      has no combat gate, so a damaged turret with regen 3 would have healed
      continuously.

    Both measurements were made back when every placed turret ran off the
    OUTER model, so they say nothing about the other four tiers -- pinning
    those to Content is what this test adds. All five tiers still agree on
    BaseHP 1300 (except the fountain's 9999), which is why the naive
    `250 * enemy_count` HP bonus alone could never have told them apart; see
    `sim.init.TURRET_HP_BONUS_NEXUS`.
    """
    from lanerl_jax.data.patch import load_patch

    turrets = load_patch().turrets
    # OUTER: the tier every turret used to be built from.
    for name in ("OrderTurretNormal", "ChaosTurretWorm"):
        u = turrets[name]
        assert u.base_ad == pytest.approx(152.0), name
        assert u.armor == pytest.approx(60.0), name
        assert u.base_hp_regen == pytest.approx(0.0), name
    # INNER: same armour as outer, but NOT the same AD -- the one field the
    # outer-profile approximation got wrong even before any ramp is applied.
    for name in ("OrderTurretNormal2", "ChaosTurretWorm2"):
        u = turrets[name]
        assert u.base_ad == pytest.approx(170.0), name
        assert u.armor == pytest.approx(60.0), name
        assert u.base_hp_regen == pytest.approx(0.0), name
    # INHIBITOR: matches the Map11-turret numbers this project once mistook
    # for Map1's outer tier -- genuinely correct here, for a different tier.
    for name in ("OrderTurretDragon", "ChaosTurretGiant"):
        u = turrets[name]
        assert u.base_ad == pytest.approx(190.0), name
        assert u.armor == pytest.approx(67.0), name
        assert u.base_hp_regen == pytest.approx(3.0), name
    # NEXUS.
    for name in ("OrderTurretAngel", "ChaosTurretNormal"):
        u = turrets[name]
        assert u.base_ad == pytest.approx(180.0), name
        assert u.armor == pytest.approx(65.0), name
        assert u.base_hp_regen == pytest.approx(6.0), name
    # FOUNTAIN: enormous HP, harmless AD (nothing survives to be hit by it in
    # this slice), zero armour and zero regen.
    for name in ("OrderTurretShrine", "ChaosTurretShrine"):
        u = turrets[name]
        assert u.base_hp == pytest.approx(9999.0), name
        assert u.armor == pytest.approx(0.0), name
        assert u.base_hp_regen == pytest.approx(0.0), name
    # Every non-fountain tier shares this BaseHP -- see the docstring above.
    for name, u in turrets.items():
        if name not in ("OrderTurretShrine", "ChaosTurretShrine"):
            assert u.base_hp == pytest.approx(1300.0), name


@pytest.mark.parametrize("t_s,expect_ad", [
    (0, 152), (29, 152), (30, 156), (89, 156), (90, 160), (150, 164),
    (210, 168), (270, 172), (330, 176), (390, 180), (600, 180), (1800, 180),
])
def test_the_outer_turret_ramps_from_152_to_180(t_s, expect_ad):
    """Turret AD is not its Content value for most of a game.

    `Maps/Map1/LevelScriptObjects` adds `AttackDamage.FlatBonus = 4` to outer
    turrets starting at 30 s, every 60 s, at most 7 times. None of it is in any
    stat JSON -- it is a `StatsModifier` the MAP script adds on a timer, so a
    port built from the stat tables misses it entirely and the turret stays
    weak for the whole game.

    The schedule is confirmed against a 600 s idle recording: turret hits on
    zero-armour minions came in at exactly 168 in 210-270 s (n=7), 172 in
    270-330 s (n=1) and 180 after 390 s (n=16), and the cannon minions' 15
    armour turns those same numbers into 146.09 / 149.57 / 153.04, which
    accounts for every remaining value in the trace.
    """
    from lanerl_jax.sim.combat import outer_turret_attack_damage

    got = float(outer_turret_attack_damage(np.float64(152.0),
                                           np.float64(t_s * 1000.0)))
    assert got == pytest.approx(expect_ad), f"at t={t_s}s"


def test_the_ramp_stops_after_seven_applications():
    """`outerTurretTimesApplied < 7` -- it is capped, not unbounded."""
    from lanerl_jax.sim.combat import outer_turret_ramps

    assert float(outer_turret_ramps(np.float64(390_000.0))) == 7.0
    assert float(outer_turret_ramps(np.float64(10_000_000.0))) == 7.0
    assert float(outer_turret_ramps(np.float64(0.0))) == 0.0


@pytest.mark.parametrize("t_s,expect_ad,expect_armor", [
    (0, 190, 67), (479, 190, 67), (480, 194, 68), (539, 194, 68),
    (540, 198, 69), (599, 198, 69), (600, 202, 70),
])
def test_the_other_tiers_ramp_ad_and_armor_from_480s(t_s, expect_ad, expect_armor):
    """The schedule `data.patch.TURRET_MODELS`'s "STILL APPROXIMATE" note used
    to warn about: `LevelScriptObjects.OnUpdate` (`:159-266`) ramps every
    non-outer, non-fountain turret on a SEPARATE timer from the outer one --
    starting at 480 s, not 30 s, and adding Armor as well as AD.

    480 s is INSIDE a 600 s episode, so unlike the outer ramp (done by 390 s)
    this schedule is still actively changing an inhibitor or nexus turret's
    stats for the entire second half of the episode. Before per-tier profiles
    existed this could not be modelled at all, because every turret ran the
    OUTER schedule (or none), and the outer schedule caps out at t=390s with
    no further change -- so a turret that should be getting stronger for the
    last two minutes of the episode was frozen instead.

    Values here use an INHIBITOR turret's Content base (AD 190, armour 67);
    NEXUS (180/65) and INNER (170/60) shift by the same per-application deltas
    from their own base -- unlike OUTER, all three of these DO get the armour
    bonus (`combat.py`'s module docstring: "note: no Armor" is called out
    specifically for the outer schedule because every other tier has one).
    """
    from lanerl_jax.sim.combat import other_turret_armor, other_turret_attack_damage

    ad = float(other_turret_attack_damage(np.float64(190.0), np.float64(t_s * 1000.0)))
    armor = float(other_turret_armor(np.float64(67.0), np.float64(t_s * 1000.0)))
    assert ad == pytest.approx(expect_ad), f"AD at t={t_s}s"
    assert armor == pytest.approx(expect_armor), f"armour at t={t_s}s"


def test_the_other_ramp_stops_after_thirty_applications():
    """`timesApplied < 30` (`:172`) -- INHIBITOR and NEXUS stop climbing after
    30 applications (at `480 + 29*60 = 2220` s), which no episode this project
    runs reaches, so this pins the cap exists rather than that it matters yet.
    """
    from lanerl_jax.sim.combat import other_turret_ramps

    assert float(other_turret_ramps(np.float64(480_000.0))) == 1.0
    assert float(other_turret_ramps(np.float64(2_220_000.0))) == 30.0
    assert float(other_turret_ramps(np.float64(100_000_000.0))) == 30.0
    assert float(other_turret_ramps(np.float64(0.0))) == 0.0


def test_the_fountain_turret_is_on_neither_ramp_schedule():
    """`UpdateTowerStats` excludes `FOUNTAIN_TURRET` by name (`:234`) and
    `UpdateOuterTurretStats` only ever looks up each lane's OUTER_TURRET
    (`:255`) -- so unlike the other four tiers, a fountain's AD and armour are
    flat for the whole game. Checked through `step._attack_damage_against`
    directly (dispatch by `model`, i.e. by tier), not just through the two
    ramp functions in isolation, because the dispatch is the part a per-tier
    rewrite could get wrong even with both ramp functions individually
    correct -- e.g. by defaulting an unrecognised tier to the OUTER schedule
    instead of to "no ramp".
    """
    import jax.numpy as jnp

    from lanerl_jax.sim.profiles import profile_id
    from lanerl_jax.sim.state import Kind, Team, TurretTier
    from lanerl_jax.sim.step import _attack_damage_against

    model = jnp.asarray([profile_id(Kind.TURRET, TurretTier.FOUNTAIN, Team.BLUE)])
    ad = jnp.asarray([999.0])
    attacker = jnp.asarray([Kind.TURRET], jnp.int8)
    target = jnp.asarray([Kind.CHAMPION], jnp.int8)
    for t_s in (0, 30, 480, 540, 10_000):
        got = _attack_damage_against(ad, attacker, target, model,
                                     np.float64(t_s * 1000.0))
        assert float(got[0]) == pytest.approx(999.0), f"at t={t_s}s"


def test_the_ad_ramp_dispatches_by_tier_not_by_kind():
    """The bug this whole change fixes, pinned at the dispatch site: before
    `TurretTier` existed, `_attack_damage_against` applied the OUTER schedule
    to `attacker_kind == Kind.TURRET` -- true of all five tiers -- which is
    why an inhibitor turret used to ramp on the wrong clock (done by 390 s
    instead of still climbing at 600 s) using the wrong per-application step
    (armour never moved at all). This drives OUTER, INNER, INHIBITOR, NEXUS
    and FOUNTAIN through the same call at the same ``t_ms`` and checks each
    one lands on its own schedule.
    """
    import jax.numpy as jnp

    from lanerl_jax.sim.profiles import profile_id
    from lanerl_jax.sim.state import Kind, Team, TurretTier
    from lanerl_jax.sim.step import _attack_damage_against

    tiers = [TurretTier.OUTER, TurretTier.INNER, TurretTier.INHIBITOR,
            TurretTier.NEXUS, TurretTier.FOUNTAIN]
    model = jnp.asarray([profile_id(Kind.TURRET, t, Team.BLUE) for t in tiers])
    base_ad = jnp.asarray([152.0, 170.0, 190.0, 180.0, 999.0])
    attacker = jnp.full((5,), Kind.TURRET, jnp.int8)
    target = jnp.full((5,), Kind.CHAMPION, jnp.int8)

    # t = 500 s: the outer ramp is long done (capped at 390 s); the other
    # schedule has fired once (at 480 s) and not twice (next is 540 s).
    got = _attack_damage_against(base_ad, attacker, target, model,
                                 np.float64(500_000.0))
    assert float(got[0]) == pytest.approx(180.0)   # OUTER: capped, +28 total
    assert float(got[1]) == pytest.approx(174.0)   # INNER: one application, +4
    assert float(got[2]) == pytest.approx(194.0)   # INHIBITOR: one application
    assert float(got[3]) == pytest.approx(184.0)   # NEXUS: one application
    assert float(got[4]) == pytest.approx(999.0)   # FOUNTAIN: never ramps


def test_hp_regen_rates_come_from_content_and_are_per_second():
    """`Stats.Update` adds `rate * diff * 0.001` with diff in ms, on a 500 ms
    accumulator -- so the stored number is HP per SECOND, not per five seconds.

    Minions are 0 on this map. Turrets are NOT uniformly 0 -- that was only
    ever true of the OUTER tier every turret used to be built from. INHIBITOR
    is 3.0 and NEXUS is 6.0 HP/s in Content (`data.patch.TURRET_MODELS`), which
    is 1,800 / 3,600 HP over a ten-minute game against a 1,550 HP pool and
    could never show up while every placed turret shared the outer row.

    The CHAMPION row is pure Content **again**, and that is the assertion.
    Garen's `BaseStaticHPRegen` is 1.568 and nothing in the config's rune or
    mastery page touches regen (only four of its sixteen talents have scripts
    at all, and none is a regen talent). The +1.2 this row used to carry was
    Doran's Shield, auto-bought by `LanerlHooks` -- an ITEM, and the sim models
    no items. Every server-side parity instrument now runs with the shop off
    (`LANERL_AUTOBUY=0`), so an item's regen here is a stat the reference does
    not have. See `sim.init.DORANS_SHIELD_HP_REGEN`, which keeps the value and
    its provenance for the day the shop is turned back on, and `STAT-001`.
    """
    from lanerl_jax.data.patch import load_patch
    from lanerl_jax.sim.init import lane_params
    from lanerl_jax.sim.profiles import profile_id
    from lanerl_jax.sim.state import Kind, Team, TurretTier
    from lanerl_jax.sim.targeting import MinionType

    p = lane_params(load_patch())
    reg = np.asarray(p["hp_regen"])
    assert reg[profile_id(Kind.CHAMPION, -1, Team.BLUE)] == pytest.approx(1.568)
    assert reg[profile_id(Kind.LANE_MINION, MinionType.MELEE, Team.BLUE)] == 0.0
    for team in (Team.BLUE, Team.RED):
        assert reg[profile_id(Kind.TURRET, TurretTier.OUTER, team)] == 0.0
        assert reg[profile_id(Kind.TURRET, TurretTier.INNER, team)] == 0.0
        assert reg[profile_id(Kind.TURRET, TurretTier.INHIBITOR, team)] == \
            pytest.approx(3.0)
        assert reg[profile_id(Kind.TURRET, TurretTier.NEXUS, team)] == \
            pytest.approx(6.0)
        assert reg[profile_id(Kind.TURRET, TurretTier.FOUNTAIN, team)] == 0.0


def test_garens_passive_heals_more_at_higher_level_brackets():
    """`HEALTH_PERCENTAGES = {0.004, 0.008, 0.02}`, brackets at 11 and 16."""
    import jax.numpy as jnp
    from lanerl_jax.sim.regen import GAREN_HEAL_PCT, garen_heal_bracket

    assert GAREN_HEAL_PCT == (0.004, 0.008, 0.02)
    lv = jnp.asarray([1, 10, 11, 15, 16, 18], jnp.int32)
    assert list(np.asarray(garen_heal_bracket(lv))) == [0, 0, 1, 1, 2, 2]


def test_regen_never_touches_a_unit_that_regenerates_nothing():
    """The clamp lives INSIDE the server's guard, and getting that wrong is
    not a rounding error -- it is a kill.

    `Stats.Update` only assigns `CurrentHealth` inside
    `if (regen > 0 && CurrentHealth < HealthPoints.Total && CurrentHealth > 0)`.
    Clamping every living unit to max_hp instead zeroed a fixture minion that
    carried hp 1.0 with max_hp 0 -- `min(1.0, 0.0)` -- so it died with no
    attacker and no champion was credited the kill.
    """
    import jax.numpy as jnp
    from lanerl_jax.sim.regen import step_regen
    from lanerl_jax.sim.state import Kind

    n = 3
    out = step_regen(
        hp=jnp.asarray([1.0, 100.0, 50.0]),
        max_hp=jnp.asarray([0.0, 200.0, 50.0]),        # slot 0: hp > max_hp
        alive=jnp.asarray([True, True, True]),
        kind=jnp.asarray([Kind.LANE_MINION, Kind.LANE_MINION, Kind.CHAMPION],
                         jnp.int8),
        level=jnp.ones((n,), jnp.int32),
        hp_regen=jnp.zeros((n,)),                       # minions regen nothing
        stat_timer=jnp.full((n,), 499.0),
        heal_timer=jnp.zeros((n,)),
        ms_since_damaged=jnp.zeros((n,)),               # in combat: no passive
        delta_ms=1000.0 / 60.0)
    assert float(out.hp[0]) == pytest.approx(1.0), "untouched, not clamped to 0"
    assert float(out.hp[1]) == pytest.approx(100.0)


def test_a_swing_that_leaves_range_mid_windup_is_cancelled_and_refunds_the_cooldown():
    """`ObjAIBase.cs:1193-1199`: `CancelAutoAttack(!HasAutoAttacked, true)`
    fires the instant a still-casting swing's target leaves `idealRange` --
    `HasAutoAttacked` is false throughout the windup, so this is always a
    `reset=true` cancel: cooldown AND windup both zero, not merely a whiffed
    swing that still pays its cooldown. Before this fix `step_autoattack` ran
    the windup to completion regardless of `in_range` (`docs/PORT_AUDIT_AI.md`
    row 10.4).
    """
    import jax.numpy as jnp
    from lanerl_jax.sim.autoattack import step_autoattack

    out = step_autoattack(
        aa_cooldown=jnp.asarray([0.9]),
        aa_windup=jnp.asarray([0.2]),          # still winding, not due to land
        is_attacking=jnp.asarray([True]),
        has_auto_attacked=jnp.asarray([False]),
        in_range=jnp.asarray([False]),          # target just left range
        can_attack=jnp.asarray([True]),
        has_target=jnp.asarray([True]),
        attack_period=jnp.asarray([1.6]),
        windup_time=jnp.asarray([0.5]),
        attack_damage=jnp.asarray([70.0]),
        target_resist=jnp.asarray([30.0]),
        delta_ms=1000.0 / 60.0)
    assert bool(out.is_attacking[0]) is False, "swing aborted"
    assert float(out.aa_windup[0]) == 0.0, "windup reset, not merely paused"
    assert float(out.aa_cooldown[0]) == 0.0, "reset=true -- immediate re-engage"
    assert bool(out.hit[0]) is False
    assert float(out.damage[0]) == 0.0


def test_a_swing_that_completes_this_tick_is_not_retroactively_cancelled():
    """Server tick order: `Spell.Update` (which resolves a completing swing)
    runs BEFORE `UpdateTarget` (which would cancel it), so a swing whose
    windup reaches zero this exact tick already lands even if the caller's
    `in_range` for THIS tick reads false (e.g. the target stepped out of
    range at the same instant the hit connects). Only a swing that is still
    winding up AFTER this tick's decrement is a cancellation candidate.
    """
    import jax.numpy as jnp
    from lanerl_jax.sim.autoattack import step_autoattack

    dt = 1000.0 / 60.0
    out = step_autoattack(
        aa_cooldown=jnp.asarray([0.9]),
        aa_windup=jnp.asarray([dt / 1000.0]),   # completes THIS tick
        is_attacking=jnp.asarray([True]),
        has_auto_attacked=jnp.asarray([False]),
        in_range=jnp.asarray([False]),
        can_attack=jnp.asarray([True]),
        has_target=jnp.asarray([True]),
        attack_period=jnp.asarray([1.6]),
        windup_time=jnp.asarray([0.5]),
        attack_damage=jnp.asarray([70.0]),
        target_resist=jnp.asarray([30.0]),
        delta_ms=dt)
    assert bool(out.hit[0]) is True, "the hit still lands"
    assert float(out.damage[0]) > 0.0


def _minion_breaks_garens_combat(minion_type, level):
    """Run champion 0 (Garen, at ``level``) against one enemy minion of
    ``minion_type`` until the minion's autoattack lands, and report whether
    that hit reset ``ms_since_damaged`` (i.e. counted as combat).
    """
    import jax.numpy as jnp

    from lanerl_jax.data.patch import load_patch
    from lanerl_jax.sim.init import init_lane, lane_params
    from lanerl_jax.sim.profiles import profile_id
    from lanerl_jax.sim.state import Kind, Team
    from lanerl_jax.sim.step import tick
    from lanerl_jax.sim.targeting import MinionType  # noqa: F401 (re-exported name below)

    patch = load_patch()
    params = lane_params(patch)
    s = init_lane(patch, include_all_turrets=False)
    kind = np.asarray(s.kind).copy()
    team = np.asarray(s.team).copy()
    alive = np.asarray(s.alive).copy()
    x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy()
    model = np.asarray(s.model).copy()
    kind[1] = Kind.LANE_MINION
    team[1] = Team.RED
    alive[1] = True
    model[1] = profile_id(Kind.LANE_MINION, minion_type, Team.RED)
    x[0], y[0] = 6000.0, 6000.0
    x[1], y[1] = 6000.0 + 50.0, 6000.0     # well within any minion's range
    s = s.replace(kind=jnp.asarray(kind), team=jnp.asarray(team),
                  alive=jnp.asarray(alive), x=jnp.asarray(x), y=jnp.asarray(y),
                  model=jnp.asarray(model),
                  xp=s.xp.at[0].set(float(patch.xp_for_level(level)) + 1.0)
                  if level > 1 else s.xp)
    for _ in range(200):
        hp_before = float(s.hp[0])
        s = tick(s, params)
        if float(s.hp[0]) < hp_before:
            return float(s.ms_since_damaged[0]) < 1.0
    raise AssertionError("the minion never landed a hit in 200 ticks")


def test_cannon_minions_break_garens_passive_below_level_11_through_a_real_tick():
    """`CharScriptGaren.cs:21-28,101-111`: cannon's OR-ed `UnitTags` collides
    with `UnitTag.Monster` (raw value 7) and is NOT in the exceptions list
    despite `Minion_Lane_Siege` being named in it -- see
    `combat.garen_passive_exempt`'s docstring for the full derivation.
    Exercised through a real auto-attack landing (not just the pure
    function below), since cannon minions are melee-basic-attack-compatible
    enough for that to actually resolve in under 200 ticks -- unlike super
    minions, see the note on the direct-function tests below.
    """
    from lanerl_jax.sim.targeting import MinionType

    assert _minion_breaks_garens_combat(MinionType.CANNON, level=1) is True
    assert _minion_breaks_garens_combat(MinionType.CANNON, level=11) is False, \
        "the Monster-value collision is double-edged: it also re-exempts " \
        "from level 11 onward, gated on GAREN'S OWN level -- previously " \
        "missing entirely from this port"


def test_melee_and_caster_minions_never_break_the_passive_through_a_real_tick():
    from lanerl_jax.sim.targeting import MinionType

    assert _minion_breaks_garens_combat(MinionType.MELEE, level=1) is False
    assert _minion_breaks_garens_combat(MinionType.MELEE, level=11) is False
    assert _minion_breaks_garens_combat(MinionType.CASTER, level=1) is False


# `garen_passive_exempt` directly, covering the full attacker/level matrix
# (including SUPER) without needing a live auto-attack to land. Super
# minions turned out to be untestable through a real tick with the harness
# above: `Blue_Minion_MechMeleeBasicAttack.json` has Content `MissileSpeed:
# 0`, and `sim/missiles.py`'s travel-time model (`step = m_speed * dt_s`)
# has no floor on that -- the missile launches (confirmed: `missile_alive`
# goes to 1 right on schedule) and then never arrives, in 200 ticks or ever.
# This is a real, separate, pre-existing gap (missiles.py is not owned by
# this task and the fix -- some minimum/instant-arrival speed -- was not
# chased further here), NOT a reason to leave the super-minion case
# unverified: hence testing the pure exemption rule directly instead.
def test_garen_passive_exempt_matrix():
    import jax.numpy as jnp

    from lanerl_jax.sim.combat import garen_passive_exempt
    from lanerl_jax.sim.state import Kind
    from lanerl_jax.sim.targeting import MinionType

    kinds = [Kind.LANE_MINION] * 4 + [Kind.CHAMPION, Kind.TURRET]
    mtypes = [MinionType.MELEE, MinionType.CASTER, MinionType.CANNON,
             MinionType.SUPER, -1, -1]
    is_lane_minion = jnp.asarray([k == Kind.LANE_MINION for k in kinds])
    is_cannon_or_super = jnp.asarray(
        [m in (MinionType.CANNON, MinionType.SUPER) for m in mtypes])

    for level, want in ((1, [True, True, False, False, False, False]),
                       (10, [True, True, False, False, False, False]),
                       (11, [True, True, True, True, False, False]),
                       (18, [True, True, True, True, False, False])):
        got = garen_passive_exempt(is_lane_minion, is_cannon_or_super,
                                   jnp.asarray([level]), jnp)[:, 0]
        assert list(np.asarray(got)) == want, f"level {level}"


# ---------------------------------------------------------------------------
# `ENT-01` / `ENT-02` (entity audit, 2026-09-23): two ways a policy earned CS
# in the sim that the server does not pay. Both were found by an audit that
# read `ObjAIBase.UpdateTarget` against `step.py`, and both were then
# reproduced through a real tick before being fixed.
# ---------------------------------------------------------------------------
def _arena_with_minions(patch, minions):
    """Blue Garen (slot 0) at (6000, 6000) and a list of ``(slot, team, hp,
    dx)`` melee minions placed ``dx`` units along +x from him. Champion 1 is
    parked far away so it plays no part. All 24 turrets are omitted (an
    in-range turret steals kills; see the lowest-index test in test_rewards).
    """
    import jax.numpy as jnp

    from lanerl_jax.sim.init import init_lane
    from lanerl_jax.sim.profiles import profile_id
    from lanerl_jax.sim.state import Kind, Team
    from lanerl_jax.sim.targeting import MinionType

    s = init_lane(patch, include_all_turrets=False)
    kind = np.asarray(s.kind).copy(); team = np.asarray(s.team).copy()
    alive = np.asarray(s.alive).copy(); x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy(); hp = np.asarray(s.hp).copy()
    model = np.asarray(s.model).copy()
    x[0], y[0] = 6000.0, 6000.0
    x[1], y[1] = 1000.0, 1000.0
    team[0], team[1] = Team.BLUE, Team.RED
    for slot, t, h, dx in minions:
        kind[slot] = Kind.LANE_MINION
        team[slot] = t
        alive[slot] = True
        model[slot] = profile_id(Kind.LANE_MINION, MinionType.MELEE, t)
        x[slot], y[slot] = 6000.0 + dx, 6000.0
        hp[slot] = h
    present = alive & (kind != Kind.NONE)
    return s.replace(
        kind=jnp.asarray(kind), team=jnp.asarray(team), alive=jnp.asarray(alive),
        x=jnp.asarray(x), y=jnp.asarray(y), hp=jnp.asarray(hp),
        collision_x=jnp.asarray(x), collision_y=jnp.asarray(y),
        collision_present=jnp.asarray(present), model=jnp.asarray(model),
        target=jnp.asarray(np.full(kind.shape[0], -1, np.int8)))


def _attack(slot):
    import jax.numpy as jnp

    from lanerl_jax.sim.orders import OrderKind, Orders
    return Orders(kind=jnp.asarray([OrderKind.ATTACK, OrderKind.NOOP], jnp.int8),
                  x=jnp.zeros(2), y=jnp.zeros(2),
                  target=jnp.asarray([slot, -1], jnp.int8))


def test_an_attack_order_on_an_allied_minion_holds_it_but_never_swings_or_pays():
    """`ENT-01`. `LanerlControl` sets any unit as `TargetUnit` with no team
    check, and `ObjAIBase.UpdateTarget` then does nothing with it: the swing,
    chase and hold branch is inside `if (TargetUnit.Team != Team ...)`
    (`ObjAIBase.cs:1285`). So the order is ACCEPTED and HELD (it releases a
    sticky enemy target, the server's only disengage) and never swings.

    Before the fix the sim chased, swung, killed a 50 HP allied minion in 22
    ticks and paid +20 gold and +1 CS for it -- a policy that learned to deny
    its own wave would score CS here and zero in the server.
    """
    import jax

    from lanerl_jax.data.patch import load_patch
    from lanerl_jax.sim.init import lane_params
    from lanerl_jax.sim.orders import apply_orders
    from lanerl_jax.sim.state import Team
    from lanerl_jax.sim.step import tick

    patch = load_patch()
    params = lane_params(patch)
    ally = 2
    s = _arena_with_minions(patch, [(ally, Team.BLUE, 50.0, 60.0)])
    s = apply_orders(s, _attack(ally), params)
    assert int(s.target[0]) == ally, "the order is accepted, as on the server"
    gold0 = float(s.gold[0])
    # Jitted: an eager `tick` is ~10 s per call on the login node.
    jtick = jax.jit(lambda st: tick(st, params))
    for _ in range(90):
        s = jtick(s)
    assert bool(s.alive[ally]) and float(s.hp[ally]) == 50.0, "never swung at"
    assert int(s.cs[0]) == 0 and float(s.gold[0]) == gold0
    assert not bool(s.is_attacking[0])
    assert int(s.target[0]) == ally, "held, not dropped: the disengage works"


def test_a_retarget_during_the_windup_lands_on_the_unit_the_swing_started_on():
    """`ENT-02`. `Spell.FinishCasting` applies the melee hit to
    `CastInfo.Targets[0].Unit` (`Spell.cs:1030`), the unit the swing was
    declared on; `SetTargetUnit` never rewrites it, and the
    `SetCurrentTarget` branch (`ObjAIBase.cs:1326`) is dead code behind the
    `IsAttacking` early return at `:1245`.

    Before the fix the hit resolved against the CURRENT target: start a swing
    on the 455 HP minion A, re-order onto the 30 HP minion B on the last
    frame, and B died while A was untouched -- zero-wind-up last-hitting.
    """
    import jax

    from lanerl_jax.data.patch import load_patch
    from lanerl_jax.sim.init import lane_params
    from lanerl_jax.sim.orders import apply_orders
    from lanerl_jax.sim.state import Team
    from lanerl_jax.sim.step import tick

    patch = load_patch()
    params = lane_params(patch)
    a, b = 2, 3
    s = _arena_with_minions(patch, [(a, Team.RED, 455.0, 60.0),
                                    (b, Team.RED, 30.0, 110.0)])
    s = apply_orders(s, _attack(a), params)
    jtick = jax.jit(lambda st: tick(st, params))
    for _ in range(5):
        s = jtick(s)
        if bool(s.is_attacking[0]):
            break
    assert bool(s.is_attacking[0]) and int(s.aa_target[0]) == a
    # Re-aim mid-wind-up. The current target changes; the swing's does not.
    s = apply_orders(s, _attack(b), params)
    assert int(s.target[0]) == b and int(s.aa_target[0]) == a
    hp_a, hp_b = float(s.hp[a]), float(s.hp[b])
    for _ in range(60):
        s = jtick(s)
        if float(s.hp[a]) < hp_a or float(s.hp[b]) < hp_b:
            break
    assert float(s.hp[a]) < hp_a, "the hit landed on A, the swing's target"
    assert float(s.hp[b]) == hp_b and bool(s.alive[b]), "B untouched"
    assert int(s.cs[0]) == 0
    assert int(s.aa_target[0]) == -1, "cleared once the swing has landed"


def test_death_rewards_never_pay_a_same_team_killer():
    """Belt and braces under `ENT-01`: even if a same-team killer index
    reached attribution, `death_rewards` pays no gold and no CS for it."""
    import jax.numpy as jnp

    from lanerl_jax.sim.rewards import death_rewards
    from lanerl_jax.sim.state import Kind, Team

    kind = jnp.asarray([Kind.CHAMPION, Kind.CHAMPION, Kind.LANE_MINION], jnp.int8)
    team = jnp.asarray([Team.BLUE, Team.RED, Team.BLUE], jnp.int8)
    out = death_rewards(
        died=jnp.asarray([False, False, True]),
        killer=jnp.asarray([-1, -1, 0], jnp.int8),
        x=jnp.zeros(3), y=jnp.zeros(3), team=team, kind=kind,
        alive=jnp.asarray([True, True, False]),
        gold_on_death=jnp.asarray([0.0, 0.0, 20.0]),
        xp_on_death=jnp.asarray([0.0, 0.0, 60.0]))
    assert float(out.gold[0]) == 0.0 and int(out.cs[0]) == 0
