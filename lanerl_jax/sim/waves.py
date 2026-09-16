"""Minion wave spawning, ported from ``Maps/Map1/LevelScript.cs``.

Fully deterministic and table-driven -- no RNG anywhere -- which makes it one of
the few mechanics that can be checked against a recording exactly rather than
statistically.

The server's loop
----------------
``LevelScript.Update``::

    if (_minionNumber > 0) {
        if (gameTime >= NextSpawnTime + _minionNumber * 8 * 100) {
            if (SetUpLaneMinion()) { _minionNumber = 0;
                                     NextSpawnTime = (long)gameTime + SpawnInterval; }
            else _minionNumber++;
        }
    } else if (gameTime >= NextSpawnTime) {
        SetUpLaneMinion();
        _minionNumber++;
    }

and ``SetUpLaneMinion`` spawns ``wave[_minionNumber]`` for each barrack, then::

    if (_minionNumber < 8) return false;

Three consequences that a "six minions every thirty seconds" model gets wrong:

1. **Minions arrive 800 ms apart**, not together -- ``_minionNumber * 8 * 100``.
2. **The counter runs to 8 regardless of wave size.** ``CreateLaneMinion``
   returns early when ``list.Count <= minionNo``, so a 6-entry regular wave
   spawns nothing on iterations 6, 7 and 8 while the counter keeps advancing.
3. **Waves are therefore ~36.4 s apart, not 30.** The reset happens on the
   iteration where ``_minionNumber == 8``, i.e. at ``NextSpawnTime + 6400`` ms,
   and only *then* is ``NextSpawnTime`` set to ``gameTime + SpawnInterval``.
   The period is ``SpawnInterval + 6400`` ms, plus up to one tick of overshoot.

The first wave is at 90 s (``NextSpawnTime`` initialiser, and
``LanerlEpisode`` re-seeds it from ``LANERL_FIRST_WAVE_MS`` on reset).

Cannon cadence
--------------
``MinionWaveToSpawn`` compares a running ``_cannonMinionCount`` against a cap
that steps down with game time: 2 from 0:00, 1 from 20:00, 0 from 35:00. When
the count reaches the cap the wave becomes a cannon wave and the count resets.
With cap 2 that is every third wave. Laning never reaches the 20-minute step, so
the cap is effectively constant 2 for this project -- but it is read from the
table rather than hardcoded, because a modern-patch swap changes exactly this
kind of number.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

__all__ = [
    "MinionType", "WAVES", "SPAWN_INTERVAL_MS", "MINION_SPACING_MS",
    "WAVE_COUNTER_MAX", "FIRST_WAVE_MS", "CANNON_TIMESTAMPS",
    "cannon_cap", "wave_for", "WaveState", "step_waves", "spawn_schedule",
]


class MinionType:
    MELEE = 0
    CASTER = 1
    CANNON = 2
    SUPER = 3


#: ``MinionWaveTypes``, in order. Index ``_minionNumber`` into these.
WAVES = {
    "RegularMinionWave": (MinionType.MELEE, MinionType.MELEE, MinionType.MELEE,
                          MinionType.CASTER, MinionType.CASTER, MinionType.CASTER),
    "CannonMinionWave": (MinionType.MELEE, MinionType.MELEE, MinionType.MELEE,
                         MinionType.CANNON, MinionType.CASTER, MinionType.CASTER,
                         MinionType.CASTER),
    "SuperMinionWave": (MinionType.SUPER, MinionType.MELEE, MinionType.MELEE,
                        MinionType.MELEE, MinionType.CASTER, MinionType.CASTER,
                        MinionType.CASTER),
}

SPAWN_INTERVAL_MS = 30_000        # MapScriptMetadata.SpawnInterval
MINION_SPACING_MS = 800           # _minionNumber * 8 * 100
WAVE_COUNTER_MAX = 8              # `if (_minionNumber < 8) return false;`
FIRST_WAVE_MS = 90_000            # NextSpawnTime initialiser / LANERL_FIRST_WAVE_MS

#: (game time ms, cap) from ``MinionWaveToSpawn``; the LAST satisfied wins.
CANNON_TIMESTAMPS: Tuple[Tuple[int, int], ...] = (
    (0, 2), (20 * 60 * 1000, 1), (35 * 60 * 1000, 0),
)


def cannon_cap(game_time_ms: float) -> int:
    cap = 2
    for t, c in CANNON_TIMESTAMPS:
        if game_time_ms >= t:
            cap = c
    return cap


def wave_for(game_time_ms: float, cannon_count: int,
             inhibitor_dead: bool = False) -> Tuple[int, Tuple[int, ...]]:
    """``MinionWaveToSpawn`` -> ``(cap, wave)``.

    ``DoubleSuperMinionWave`` exists in the server but needs every inhibitor
    down, which cannot happen in a laning-phase episode; it is omitted rather
    than carried as dead data.
    """
    cap = cannon_cap(game_time_ms)
    name = "RegularMinionWave"
    if cannon_count >= cap:
        name = "CannonMinionWave"
    if inhibitor_dead:
        name = "SuperMinionWave"
    return cap, WAVES[name]


@dataclass(slots=True)
class WaveState:
    """``NextSpawnTime`` / ``_minionNumber`` / ``_cannonMinionCount``."""

    next_spawn_ms: float = FIRST_WAVE_MS
    minion_number: int = 0
    cannon_count: int = 0


def step_waves(st: WaveState, game_time_ms: float,
               inhibitor_dead: bool = False) -> List[int]:
    """One tick of ``LevelScript.Update``. Returns the minion types to spawn
    **per barrack** (so a 1v1 top lane spawns one of each per side)."""
    spawned: List[int] = []
    if st.minion_number > 0:
        if game_time_ms >= st.next_spawn_ms + st.minion_number * MINION_SPACING_MS:
            cap, wave = wave_for(game_time_ms, st.cannon_count, inhibitor_dead)
            if st.minion_number < len(wave):
                spawned.append(wave[st.minion_number])
            if st.minion_number >= WAVE_COUNTER_MAX:          # SetUpLaneMinion -> true
                st.minion_number = 0
                # `NextSpawnTime = (long)gameTime + SpawnInterval` -- the cast
                # TRUNCATES, and that sub-millisecond loss is what makes the
                # observed wave starts drift a hair later each wave. Dropping it
                # costs ~40 ms by the sixth wave, which is small but is exactly
                # the size of the residual it was being blamed for.
                st.next_spawn_ms = float(int(game_time_ms)) + SPAWN_INTERVAL_MS
                st.cannon_count = 0 if st.cannon_count >= cap else st.cannon_count + 1
            else:
                st.minion_number += 1
    elif game_time_ms >= st.next_spawn_ms:
        cap, wave = wave_for(game_time_ms, st.cannon_count, inhibitor_dead)
        if 0 < len(wave):
            spawned.append(wave[0])
        st.minion_number += 1
    return spawned


def spawn_schedule(until_ms: float, tick_ms: float = 1000.0 / 60.0,
                   first_wave_ms: float = FIRST_WAVE_MS
                   ) -> List[Tuple[float, int]]:
    """Every ``(game_time_ms, minion_type)`` one barrack spawns, up to ``until_ms``.

    Runs the real tick loop rather than a closed form, because the 800 ms
    spacing interacts with the tick grid and with the counter-to-8 reset in
    ways a closed form would have to re-derive (and get wrong).

    **The clock accumulates in float32**, because ``Game.GameTime`` is a C#
    ``float`` and ``GameTime += diff`` accumulates its rounding. Running this
    loop in float64 -- the obvious Python default -- puts the sixth wave 40 ms
    early against the recording, and that 40 ms was briefly blamed on the
    ``(long)gameTime`` truncation instead. Third time in this project that using
    double precision where the server uses single produced a small systematic
    divergence; the pathfinder and the auto-attack clock were the other two.
    """
    st = WaveState(next_spawn_ms=first_wave_ms)
    out: List[Tuple[float, int]] = []
    t = np.float32(0.0)
    step = np.float32(tick_ms)
    limit = np.float32(until_ms)
    while t <= limit:
        for m in step_waves(st, float(t)):
            out.append((float(t), m))
        t = np.float32(t + step)
    return out
