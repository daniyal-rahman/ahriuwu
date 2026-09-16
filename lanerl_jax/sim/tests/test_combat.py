"""Damage modifiers that are properties of the attacker/target PAIR.

A stat table cannot hold these: the same turret, on the same tick, deals
different damage to a minion and to a champion. They live in the basic-attack
scripts on the server, which is why they are easy to miss when porting from the
stat JSON -- and missing one is worth 43% of a turret's output against a wave.
"""
from __future__ import annotations

import pytest



def test_a_turret_deals_seventy_percent_to_minions_and_full_to_champions():
    """``if (target is Minion) dmg *= 0.7f`` -- in every lane turret's script.

    It is a property of the attacker/target PAIR, so it cannot be folded into
    the turret's attack-damage stat: the same turret does full damage to a
    champion on the same tick.

    Missing it made the sim's turrets 43% stronger against a wave than the
    server's, which is a caster dead in 2 shots rather than 3.
    """
    import jax.numpy as jnp
    from lanerl_jax.sim.combat import TURRET_DAMAGE_VS_MINION
    from lanerl_jax.sim.state import Kind
    from lanerl_jax.sim.step import _attack_damage_against

    assert TURRET_DAMAGE_VS_MINION == 0.7
    ad = jnp.asarray([190.0, 190.0, 190.0, 78.0])
    attacker = jnp.asarray([Kind.TURRET, Kind.TURRET, Kind.CHAMPION,
                            Kind.LANE_MINION], jnp.int8)
    target = jnp.asarray([Kind.LANE_MINION, Kind.CHAMPION, Kind.LANE_MINION,
                          Kind.LANE_MINION], jnp.int8)
    got = _attack_damage_against(ad, attacker, target)
    assert float(got[0]) == pytest.approx(133.0)   # turret -> minion
    assert float(got[1]) == pytest.approx(190.0)   # turret -> champion: full
    assert float(got[2]) == pytest.approx(190.0)   # champion -> minion: full
    assert float(got[3]) == pytest.approx(78.0)    # minion  -> minion: full
