"""The sim's fixed timestep is only valid against a free-run server.

`lanerl_jax.sim.movement_jax.TICK_MS` is a hard 1000/60, and every parity
comparison this project makes -- movement traces, state-hash pairs, the minion
population -- assumes the server advanced by exactly that much per tick.

The server does NOT do that by default. `Game.GameLoop` measures real
wall-clock elapsed time with a Stopwatch and passes it as `deltaTime`, so
`diff` jitters with OS scheduling, GC and the previous tick's own work. It is
pinned to exactly `REFRESH_RATE = 1000.0/60.0` in two cases only: the first
tick (to avoid `Update(0)`), and when `LANERL_FREERUN=1`.

So if this default ever flips, the fixed-timestep sim would be diffed against a
jittered ground truth and every parity number would quietly become noise
measured in milliseconds of scheduler luck -- while still looking like a
simulation disagreement. That is a bad failure to debug and a cheap one to
prevent.
"""
from __future__ import annotations

from lanerl_train.vec import ServerLaunchSpec


def test_recordings_are_fixed_timestep():
    assert ServerLaunchSpec().freerun is True, (
        "ServerLaunchSpec.freerun defaults to False -- the server would run on "
        "measured wall-clock time and no parity trace taken with it is "
        "comparable to the fixed-TICK_MS sim")


def test_the_sim_tick_matches_the_servers_free_run_constant():
    """1000/60 on both sides, from the server's REFRESH_RATE."""
    from lanerl_jax.sim.movement_jax import TICK_MS

    assert TICK_MS == 1000.0 / 60.0
