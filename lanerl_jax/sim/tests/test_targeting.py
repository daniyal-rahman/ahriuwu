"""Target-selection ties follow server object creation order, not JAX slots."""
from __future__ import annotations

import jax.numpy as jnp

from lanerl_jax.sim.state import Kind, Team
from lanerl_jax.sim.targeting import (ClassifyUnit, MinionType,
                                      minion_acquire, nearest_enemy,
                                      turret_acquire)


def _scene():
    # Slot 0 is the chooser; slots 1 and 2 are exactly tied candidates.  Slot
    # 2 was created first, deliberately opposing jnp.argmin's slot tie-break.
    x = jnp.asarray([0.0, 100.0, -100.0])
    y = jnp.zeros(3)
    team = jnp.asarray([Team.BLUE, Team.RED, Team.RED], dtype=jnp.int8)
    alive = jnp.ones(3, dtype=bool)
    spawn_seq = jnp.asarray([0, 20, 10], dtype=jnp.int32)
    return x, y, team, alive, spawn_seq


def test_champion_nearest_tie_uses_creation_order_not_slot():
    x, y, team, alive, seq = _scene()
    got = nearest_enemy(x, y, team, alive, alive,
                        jnp.full(3, 500.0), seq)
    assert int(got[0]) == 2


def test_turret_priority_tie_uses_creation_order_not_slot():
    x, y, team, alive, seq = _scene()
    kind = jnp.asarray([Kind.TURRET, Kind.LANE_MINION, Kind.LANE_MINION],
                       dtype=jnp.int8)
    got = turret_acquire(
        x, y, team, alive, alive, kind,
        jnp.asarray([MinionType.MELEE] * 3, dtype=jnp.int8),
        jnp.full(3, 500.0), jnp.full(3, -1, dtype=jnp.int8),
        jnp.full(3, -1, dtype=jnp.int8), jnp.full(3, 500.0), seq,
        collision_radius=jnp.zeros(3))
    assert int(got[0]) == 2


def test_minion_equal_priority_distance_tie_uses_creation_order_not_slot():
    x, y, team, alive, seq = _scene()
    priority = jnp.full((3, 3), ClassifyUnit.MELEE_MINION, dtype=jnp.int8)
    got = minion_acquire(
        x, y, team, alive, alive, jnp.ones(3, dtype=bool), priority,
        jnp.full(3, 500.0), jnp.full(3, -1, dtype=jnp.int8),
        jnp.full(3, ClassifyUnit.DEFAULT, dtype=jnp.int8),
        jnp.zeros((3, 3), dtype=bool), spawn_seq=seq)
    assert int(got[0]) == 2


# ``ObjAIBase.ClassifyTarget(attacker, victim)`` (`ObjAIBase.cs:388-448`),
# written out pair by pair. The victim switch covers champion/minion victims
# only; every other pair falls through to the ATTACKER's base class.
_ATTACKERS = [  # (label, kind, minion type)
    ("champion", Kind.CHAMPION, MinionType.MELEE),
    ("melee", Kind.LANE_MINION, MinionType.MELEE),
    ("caster", Kind.LANE_MINION, MinionType.CASTER),
    ("cannon", Kind.LANE_MINION, MinionType.CANNON),
    ("super", Kind.LANE_MINION, MinionType.SUPER),
    ("turret", Kind.TURRET, MinionType.MELEE),
]
_VICTIMS = [("champion", Kind.CHAMPION), ("minion", Kind.LANE_MINION),
            ("turret", Kind.TURRET)]
_MINION_BASE = {"melee": ClassifyUnit.MELEE_MINION,
                "caster": ClassifyUnit.CASTER_MINION,
                "cannon": ClassifyUnit.SUPER_OR_CANNON_MINION,
                "super": ClassifyUnit.SUPER_OR_CANNON_MINION}


def _classify_target(attacker: str, victim: str) -> int:
    if attacker == "champion":
        return {"champion": ClassifyUnit.CHAMPION_ATTACKING_CHAMPION,
                "minion": ClassifyUnit.CHAMPION_ATTACKING_MINION,
                "turret": ClassifyUnit.CHAMPION}[victim]     # fall-through
    if attacker == "turret":
        return {"champion": ClassifyUnit.TURRET,             # fall-through
                "minion": ClassifyUnit.TURRET_ATTACKING_MINION,
                "turret": ClassifyUnit.TURRET}[victim]       # fall-through
    return {"champion": ClassifyUnit.MINION_ATTACKING_CHAMPION,
            "minion": ClassifyUnit.MINION_ATTACKING_MINION,
            "turret": _MINION_BASE[attacker]}[victim]        # fall-through


def test_call_for_help_priority_matches_classify_target_for_every_pair():
    """`ENT-06`. The port returned ``DEFAULT`` (no call) for the three
    fall-through pairs involving a turret -- turret->champion,
    champion->turret, minion->turret -- where the server registers the
    attacker at its base class."""
    from lanerl_jax.sim.targeting import help_priority_for

    ak = jnp.asarray([a[1] for a in _ATTACKERS], jnp.int8)[:, None]
    am = jnp.asarray([a[2] for a in _ATTACKERS], jnp.int8)[:, None]
    vk = jnp.asarray([v[1] for v in _VICTIMS], jnp.int8)[None, :]
    got = help_priority_for(ak, vk, am)
    wrong = {}
    for i, a in enumerate(_ATTACKERS):
        for j, v in enumerate(_VICTIMS):
            want = int(_classify_target(a[0], v[0]))
            if int(got[i, j]) != want:
                wrong[(a[0], v[0])] = (int(got[i, j]), want)
    assert not wrong, f"(attacker, victim): (got, ClassifyTarget) {wrong}"
