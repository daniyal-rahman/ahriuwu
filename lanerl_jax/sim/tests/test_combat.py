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

    The CHAMPION is the one row that is no longer pure Content. The server
    auto-buys Doran's Shield at boot and `ItemPassives/DoransShield.cs` does
    `HealthRegeneration.BaseBonus += 1.2f`, so its champion regenerates at
    Garen's Content 1.568 PLUS 1.2. The dump does not expose regen, which is
    why this stat was taken from Content in the first place and why the item
    was missed; max HP, AD and armour are all measured against the dump
    instead. See `sim.init.DORANS_SHIELD_HP_REGEN`.
    """
    from lanerl_jax.data.patch import load_patch
    from lanerl_jax.sim.init import lane_params
    from lanerl_jax.sim.profiles import profile_id
    from lanerl_jax.sim.state import Kind, Team, TurretTier
    from lanerl_jax.sim.targeting import MinionType

    p = lane_params(load_patch())
    reg = np.asarray(p["hp_regen"])
    from lanerl_jax.sim.init import DORANS_SHIELD_HP_REGEN
    assert reg[profile_id(Kind.CHAMPION, -1, Team.BLUE)] == \
        pytest.approx(1.568 + DORANS_SHIELD_HP_REGEN)
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
