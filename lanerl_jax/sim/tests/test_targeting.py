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
        jnp.full(3, -1, dtype=jnp.int8), jnp.full(3, 500.0), seq)
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
