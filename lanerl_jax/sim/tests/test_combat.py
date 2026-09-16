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
    # Map1 OUTER_TURRET, per team. Red's is ChaosTurretWorm -- NOT
    # ChaosTurretNormal, which is chaos's *nexus* turret (AD 180, armour 65,
    # regen 6). Pairing the teams by matching names picks the wrong unit.
    assert TURRET_MODELS == ("OrderTurretNormal", "ChaosTurretWorm")


def test_the_turret_stats_are_the_ones_measured_off_the_server():
    """AD 152 / armour 60 / regen 0, each confirmed against a 600 s recording.

    * armour 60: single-step turret HP drops land on exactly 7.500 (melee,
      93x), 14.375 (caster, 68x) and 25.000 (cannon, 4x) -- minion AD times
      100/160. Armour 67 predicts 7.186 / 13.772 / 23.952 and not one drop
      matched.
    * regen 0: turret HP was monotone over all 36,001 snapshots. `Stats.Update`
      has no combat gate, so a damaged turret with regen 3 would have healed
      continuously.
    """
    from lanerl_jax.data.patch import load_patch

    for name, u in load_patch().turrets.items():
        assert u.base_ad == pytest.approx(152.0), name
        assert u.armor == pytest.approx(60.0), name
        assert u.base_hp_regen == pytest.approx(0.0), name
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
