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


class ResetBenchResult(NamedTuple):
    """J1 gate 6 (D11): reset cost, measured against one `step_decision`.

    Both timings are taken at the SAME `n_envs`, on the same array shapes, so
    the ratio is not an apples-to-oranges comparison between two different
    benchmark runs.
    """
    n_envs: int
    reset_s: float
    step_s: float
    device: str

    @property
    def ratio(self) -> float:
        return self.reset_s / self.step_s

    def report(self) -> str:
        verdict = "PASSES" if self.ratio < 0.1 else "FAILS"
        return (
            f"{self.n_envs:>6} envs | reset {self.reset_s * 1e6:8.1f} us | "
            f"step {self.step_s * 1e6:8.1f} us | "
            f"reset/step {self.ratio:6.2%} | gate 6 {verdict}"
        )


def run_reset_benchmark(n_envs: int, steps: int = 200, warmup: int = 5,
                        seed: int = 0) -> ResetBenchResult:
    """Time the trainer's actual reset op against one `step_decision`.

    D11 says reset must stay "pure array initialisation from static
    constants" -- a `jax.tree.map` `jnp.where` against a constant pytree,
    exactly `trainer.py`'s ``nxt = jax.tree.map(lambda a, b: jnp.where(done, b,
    a), nxt, fresh)`` -- and gate 6 asks that this be measured separately and
    confirmed small relative to a step. It never was; this is that
    measurement.

    Reproduces the trainer's shapes exactly rather than approximating them:
    `fresh` is the SAME unbatched constant pytree `init_lane()` produces
    (closed over, not broadcast -- broadcasting it would time allocating 512
    copies of a constant, not the where-select itself), `done` is one bool
    per env exactly as `nxt.t_ms >= episode_s * 1000.0` produces one bool per
    env, and the whole thing is `vmap`ped over `n_envs` the same way
    `_env_step`'s `one` is.
    """
    patch_params = lane_params()
    path = jnp.asarray(np.array(TOP_LANE_PATH, np.float32))
    # `trainer.py` broadcasts this SAME constant for both the initial batch
    # and the reset target (`env_state = jax.tree.map(..., fresh)`); mirrored
    # here rather than building the starting batch from a second call.
    fresh = init_lane()
    states = jax.tree.map(lambda a: jnp.broadcast_to(a, (n_envs,) + a.shape), fresh)

    def reset_one(state, done):
        return jax.tree.map(lambda a, b: jnp.where(done, b, a), state, fresh)

    @jax.jit
    def reset_batch(states, dones):
        return jax.vmap(reset_one)(states, dones)

    @jax.jit
    def sim_step(states):
        return jax.vmap(lambda s: step_decision(s, patch_params, lane_path=path))(states)

    # Alternating true/false rather than all-true: `jnp.where`'s cost does not
    # depend on the predicate's VALUE (XLA does not branch per-element), so
    # this is only to avoid a benchmark that happens to look identical to
    # "always reset" or "never reset" and to keep the compiled program honest
    # about handling both.
    dones = (jnp.arange(n_envs) % 2 == 0)

    def timeit(fn, *args):
        st = fn(*args)
        for _ in range(warmup - 1):
            st = fn(*args)
        jax.block_until_ready(st)
        t0 = time.perf_counter()
        for _ in range(steps):
            st = fn(*args)
        jax.block_until_ready(st)
        return (time.perf_counter() - t0) / steps

    reset_s = timeit(reset_batch, states, dones)
    step_s = timeit(sim_step, states)

    return ResetBenchResult(n_envs=n_envs, reset_s=reset_s, step_s=step_s,
                            device=str(jax.devices()[0]))


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

    print("\nJ1 gate 6 (D11): reset cost vs. one step_decision\n")
    for n in sizes:
        try:
            print(run_reset_benchmark(n).report())
        except Exception as exc:
            print(f"{n:>6} envs | FAILED: {type(exc).__name__}: {str(exc)[:120]}")
