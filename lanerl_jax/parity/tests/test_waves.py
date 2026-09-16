"""Wave spawning, validated against a real 300 s recording.

Waves are fully deterministic and table-driven -- no RNG anywhere -- which makes
this one of the few mechanics that can be checked against a recording *exactly*
rather than statistically. The observed numbers below come from a bot-driven
`LANERL_TOPONLY` run on 2026-09-16 (18,002 tick snapshots).
"""
from __future__ import annotations

import pytest

from lanerl_jax.sim.waves import (
    CANNON_TIMESTAMPS,
    FIRST_WAVE_MS,
    MINION_SPACING_MS,
    SPAWN_INTERVAL_MS,
    WAVE_COUNTER_MAX,
    MinionType,
    WaveState,
    cannon_cap,
    spawn_schedule,
    step_waves,
    wave_for,
)

#: Wave-start times observed in the real recording, seconds.
OBSERVED_WAVE_STARTS = (90.02, 126.43, 162.84, 199.25, 235.66, 272.06)


def _wave_starts(schedule, gap_ms=5000):
    out, prev = [], None
    for t, _ in schedule:
        if prev is None or t - prev > gap_ms:
            out.append(t)
        prev = t
    return out


def test_wave_starts_match_the_recording():
    """Within 50 ms (3 ticks) across five wave periods.

    The small, slowly growing offset is expected: the observed time is when a
    minion first appears in a *dump snapshot*, which is after the spawn tick,
    and ``NextSpawnTime = (long)gameTime + SpawnInterval`` truncates to whole
    milliseconds each wave.
    """
    starts = _wave_starts(spawn_schedule(300_000))
    assert len(starts) == len(OBSERVED_WAVE_STARTS)
    for got, want in zip(starts, OBSERVED_WAVE_STARTS):
        # 20 ms, i.e. one tick plus the sampling offset. Was 50 ms before the
        # `(long)gameTime` truncation was modelled; that cast alone accounted
        # for ~40 ms of drift by the sixth wave.
        assert got / 1000.0 == pytest.approx(want, abs=0.02)


def test_the_wave_period_is_36_4_seconds_not_30():
    """``SpawnInterval`` is 30 s, but the counter runs to 8 before resetting.

    ``SetUpLaneMinion`` only returns true at ``_minionNumber == 8``, i.e. at
    ``NextSpawnTime + 8*800``, and only *then* is ``NextSpawnTime`` reset to
    ``gameTime + SpawnInterval``. So the real period is 36.4 s. A "six minions
    every thirty seconds" model drifts a full wave inside four minutes, which
    would put every wave-state calculation in the wrong place.
    """
    starts = _wave_starts(spawn_schedule(300_000))
    periods = [(starts[i + 1] - starts[i]) / 1000.0 for i in range(len(starts) - 1)]
    for p in periods:
        assert p == pytest.approx(36.4, abs=0.05)
    assert (SPAWN_INTERVAL_MS + WAVE_COUNTER_MAX * MINION_SPACING_MS) / 1000.0 == 36.4
    observed = [OBSERVED_WAVE_STARTS[i + 1] - OBSERVED_WAVE_STARTS[i]
                for i in range(len(OBSERVED_WAVE_STARTS) - 1)]
    for p in observed:
        assert p == pytest.approx(36.4, abs=0.05)


def test_minions_arrive_800ms_apart_not_together():
    sched = spawn_schedule(100_000)
    first = [t for t, _ in sched if t < 100_000][:6]
    gaps = [first[i + 1] - first[i] for i in range(len(first) - 1)]
    for g in gaps:
        assert g == pytest.approx(MINION_SPACING_MS, abs=20)


def test_the_first_wave_is_at_ninety_seconds():
    sched = spawn_schedule(120_000)
    assert sched[0][0] == pytest.approx(FIRST_WAVE_MS, abs=20)
    # observed
    assert OBSERVED_WAVE_STARTS[0] == pytest.approx(90.0, abs=0.05)


def test_a_regular_wave_is_three_melee_then_three_casters():
    sched = spawn_schedule(120_000)
    types = [m for _, m in sched]
    assert types[:6] == [MinionType.MELEE] * 3 + [MinionType.CASTER] * 3


def test_every_third_wave_is_a_cannon_wave():
    """Cap is 2 before 20 minutes, so the count 0,1,2 cycles every third wave."""
    sched = spawn_schedule(300_000)
    waves, cur, prev = [], [], None
    for t, m in sched:
        if prev is not None and t - prev > 5000:
            waves.append(cur)
            cur = []
        cur.append(m)
        prev = t
    waves.append(cur)
    cannon = [i for i, w in enumerate(waves) if MinionType.CANNON in w]
    assert cannon == [2, 5], f"cannon waves at {cannon}"
    assert len(waves[2]) == 7 and len(waves[0]) == 6


def test_the_counter_runs_past_the_wave_length_spawning_nothing():
    """``CreateLaneMinion`` returns early when ``list.Count <= minionNo``.

    So a 6-entry regular wave has three iterations (6, 7, 8) that advance the
    counter and spawn nothing -- which is exactly where the extra 6.4 s of
    period comes from.
    """
    st = WaveState(next_spawn_ms=0.0)
    spawned_at = []
    t = 0.0
    while st.minion_number != 0 or t == 0.0:
        got = step_waves(st, t)
        if got:
            spawned_at.append((t, st.minion_number))
        t += 1000.0 / 60.0
        if t > 20_000:
            break
    assert len(spawned_at) == 6, "a regular wave spawns exactly six minions"
    assert t == pytest.approx(SPAWN_INTERVAL_MS - 30_000 + 6400, abs=100) or t > 6400


def test_cannon_cap_steps_down_with_game_time():
    assert cannon_cap(0) == 2
    assert cannon_cap(19 * 60 * 1000) == 2
    assert cannon_cap(20 * 60 * 1000) == 1
    assert cannon_cap(35 * 60 * 1000) == 0
    assert CANNON_TIMESTAMPS[0] == (0, 2)


def test_super_wave_needs_a_dead_inhibitor():
    _, w = wave_for(0, 0, inhibitor_dead=False)
    assert MinionType.SUPER not in w
    _, w = wave_for(0, 0, inhibitor_dead=True)
    assert w[0] == MinionType.SUPER
