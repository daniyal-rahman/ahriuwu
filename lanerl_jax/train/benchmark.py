"""Throughput, measured the way the J1 gate is written.

The gate says **with the real policy in the loop**, not the sim alone. That
wording is deliberate: the entity transformer over 32 slots may well dominate,
and a sim-only number is both the flattering one and the misleading one. So this
measures the whole decision -- observation, policy forward, action decode, two
simulator ticks -- and reports the sim-only figure separately so the split is
visible rather than assumed.

The baseline it is measured against is the production stack's own logged number:
**1,129 decisions/s** (``runs/rl-league-0915c``, 48,203 s wall, 16,800 updates).

What this deliberately does NOT measure
---------------------------------------
The learner. There is no gradient step here, so this is the *acting* half of
the loop. Under an Anakin design the update is part of the same XLA program and
its cost adds to this, which is why the J3 gate re-measures end to end rather
than trusting this number.

Sampling uses one PRNG key split per step, which is the real cost; the action
decode maps screen bins to a world offset with a straight-line approximation
rather than `projection.screen_to_world_centred`. The decode is a handful of
elementwise ops either way, so it does not move the number -- but it is named
here rather than quietly assumed.
"""
from __future__ import annotations

import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..obs.builder import build_observation
from ..obs.frame import make_lane_frame
from ..sim.init import TOP_LANE_PATH, TOP_OUTER_TURRET, init_lane, lane_params
from ..sim.orders import OrderKind, Orders, apply_orders
from ..sim.state import Team
from ..sim.step import step_decision
from .policy import LanePolicy, PolicyConfig

__all__ = ["BenchResult", "run_benchmark"]

BLUE_NEXUS = (1131.8, 1426.3)
RED_NEXUS = (12760.9, 13026.1)
#: `constants.SCREEN_RADIUS` -- how far a screen click can name a point.
SCREEN_RADIUS = 1800.0


class BenchResult(NamedTuple):
    n_envs: int
    compile_s: float
    decisions_per_s: float
    sim_only_decisions_per_s: float
    device: str

    def report(self, baseline: float = 1129.0) -> str:
        return (
            f"{self.n_envs:>6} envs | compile {self.compile_s:6.1f}s | "
            f"full loop {self.decisions_per_s:>12,.0f} dec/s "
            f"({self.decisions_per_s / baseline:>6.0f}x) | "
            f"sim only {self.sim_only_decisions_per_s:>12,.0f} dec/s"
        )


def _decode(logits, state, key):
    """Sample an action and turn it into champion orders.

    One key split per step, which is the real per-decision cost. The screen
    heads name a point relative to the champion; the mapping used here is a
    plain polar offset rather than `projection.screen_to_world_centred`, which
    is a few elementwise ops either way.
    """
    kb, kx, ky, kt = jax.random.split(key, 4)
    button = jax.random.categorical(kb, logits.button)
    sx = jax.random.categorical(kx, logits.screen_x)
    sy = jax.random.categorical(ky, logits.screen_y)
    tgt = jax.random.categorical(kt, logits.target)

    nx = (sx + 0.5) / logits.screen_x.shape[-1] * 2.0 - 1.0
    ny = (sy + 0.5) / logits.screen_y.shape[-1] * 2.0 - 1.0
    ox = state.x[:2] + nx * SCREEN_RADIUS
    oy = state.y[:2] + ny * SCREEN_RADIUS

    # BUTTONS = (noop, move, attack_move, q, w, e, r, recall)
    kind = jnp.where(button == 1, OrderKind.MOVE,
                     jnp.where(button == 2, OrderKind.ATTACK,
                               jnp.where(button == 5, OrderKind.CAST_E,
                                         OrderKind.NOOP)))
    return Orders(kind=kind.astype(jnp.int8), x=ox, y=oy,
                  target=tgt.astype(jnp.int8))


def run_benchmark(n_envs: int, steps: int = 60, warmup: int = 3,
                  seed: int = 0) -> BenchResult:
    patch_params = lane_params()
    path = jnp.asarray(np.array(TOP_LANE_PATH, np.float32))
    frame = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                            TOP_OUTER_TURRET[Team.RED], BLUE_NEXUS)
    policy = LanePolicy(PolicyConfig())

    base = init_lane()
    states = jax.tree.map(lambda a: jnp.broadcast_to(a, (n_envs,) + a.shape), base)

    obs0 = build_observation(base, 0, frame)
    variables = policy.init(jax.random.key(seed), obs0.entities[None],
                            obs0.entity_pad_mask[None], obs0.self_vec[None],
                            obs0.global_vec[None])

    def one_env(state, key):
        # both champions act; the policy is shared, which is the mirror setup
        obs = jax.vmap(lambda i: build_observation(state, i, frame))(jnp.arange(2))
        logits = policy.apply(variables, obs.entities, obs.entity_pad_mask,
                              obs.self_vec, obs.global_vec)
        orders = _decode(logits, state, key)
        state = apply_orders(state, orders)
        return step_decision(state, patch_params, lane_path=path)

    @jax.jit
    def full_step(states, keys):
        return jax.vmap(one_env)(states, keys)

    @jax.jit
    def sim_step(states):
        return jax.vmap(lambda s: step_decision(s, patch_params, lane_path=path))(states)

    keys = jax.random.split(jax.random.key(seed), n_envs)
    t0 = time.perf_counter()
    out = full_step(states, keys)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - t0

    def timeit(fn, st, with_keys):
        for _ in range(warmup):
            st = fn(st, keys) if with_keys else fn(st)
        jax.block_until_ready(st)
        t = time.perf_counter()
        for _ in range(steps):
            st = fn(st, keys) if with_keys else fn(st)
        jax.block_until_ready(st)
        return (time.perf_counter() - t) / steps

    per_full = timeit(full_step, states, True)
    jax.block_until_ready(sim_step(states))
    per_sim = timeit(sim_step, states, False)

    return BenchResult(
        n_envs=n_envs, compile_s=compile_s,
        decisions_per_s=n_envs / per_full,
        sim_only_decisions_per_s=n_envs / per_sim,
        device=str(jax.devices()[0]),
    )


if __name__ == "__main__":
    import sys

    sizes = [int(a) for a in sys.argv[1:]] or [64, 256, 1024, 4096]
    print(f"device: {jax.devices()[0]}")
    print(f"baseline: the production stack's own logged 1,129 decisions/s\n")
    for n in sizes:
        try:
            print(run_benchmark(n).report())
        except Exception as exc:                     # OOM is a real answer
            print(f"{n:>6} envs | FAILED: {type(exc).__name__}: {str(exc)[:120]}")
