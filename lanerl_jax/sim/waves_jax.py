"""``LevelScript.Update``'s wave loop, as a traceable JAX step.

The Python version in :mod:`lanerl_jax.sim.waves` is the readable reference and
the thing the recording was validated against (six wave starts to within 40 ms).
This is the same loop with the table lookups turned into fixed-shape gathers so
it can live inside the compiled tick.

The wave tables are padded to nine entries with ``-1``
-----------------------------------------------------
``_minionNumber`` runs 0..8 regardless of how long the wave actually is, because
``SetUpLaneMinion`` only returns true at 8 and ``CreateLaneMinion`` returns
early when ``list.Count <= minionNo``. Padding with "spawn nothing" reproduces
that exactly and keeps the gather in range, where a Python ``if`` would have to
become a branch.
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from .waves import (
    MINION_SPACING_MS,
    SPAWN_INTERVAL_MS,
    WAVE_COUNTER_MAX,
    WAVES,
    MinionType,
)

__all__ = ["REGULAR", "CANNON", "step_waves_jax"]

_PAD = 9


def _pad(wave) -> Tuple[int, ...]:
    return tuple(wave) + (-1,) * (_PAD - len(wave))


REGULAR = jnp.asarray(_pad(WAVES["RegularMinionWave"]), jnp.int8)
CANNON = jnp.asarray(_pad(WAVES["CannonMinionWave"]), jnp.int8)


def step_waves_jax(t_ms: jax.Array, next_spawn_ms: jax.Array,
                   minion_number: jax.Array, cannon_count: jax.Array,
                   cannon_cap: int = 2):
    """One tick of the spawner.

    Returns ``(minion_type, next_spawn_ms, minion_number, cannon_count)`` where
    ``minion_type`` is -1 for "nothing spawns this tick". One value serves both
    barracks: the server spawns the same wave index for each, which is why a
    1v1 top lane gets one minion per side per 800 ms rather than two per side.
    """
    wave = jnp.where(cannon_count >= cannon_cap, CANNON, REGULAR)

    started = minion_number > 0
    due_mid = t_ms >= next_spawn_ms + minion_number * MINION_SPACING_MS
    due_first = t_ms >= next_spawn_ms
    fires = jnp.where(started, due_mid, due_first)

    idx = jnp.clip(minion_number, 0, _PAD - 1)
    mtype = jnp.where(fires, wave[idx], jnp.int8(-1))

    # `if (_minionNumber < 8) return false;` -- the wave closes at 8, and only
    # then is NextSpawnTime pushed forward, which is where the 36.4 s period
    # comes from.
    closes = fires & started & (minion_number >= WAVE_COUNTER_MAX)
    new_number = jnp.where(closes, 0,
                           jnp.where(fires, minion_number + 1, minion_number))
    # `NextSpawnTime = (long)gameTime + SpawnInterval` -- the cast truncates, and
    # that truncation is what makes the observed period drift by a millisecond
    # per wave against a naive float model.
    new_next = jnp.where(
        closes, jnp.floor(t_ms) + SPAWN_INTERVAL_MS, next_spawn_ms)
    new_cannon = jnp.where(
        closes, jnp.where(cannon_count >= cannon_cap, 0, cannon_count + 1),
        cannon_count)
    return mtype, new_next.astype(next_spawn_ms.dtype), \
        new_number.astype(minion_number.dtype), new_cannon.astype(cannon_count.dtype)
