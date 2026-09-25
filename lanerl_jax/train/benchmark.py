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

Sampling uses one PRNG key split per step, which is the real cost. The shared
action decoder uses the calibrated locked-camera perspective projection and
the observer-side lane reflection, the same mapping as the real environment.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_rl.constants import DECISION_HZ

from ..obs.builder import build_observation
from ..obs.frame import make_lane_frame
from ..sim.config import DEFAULT_ROUTE_ARTIFACT, SimConfig
from ..sim.init import TOP_OUTER_TURRET, init_lane
from ..sim.state import Team
from ..sim.step import env_advance, env_step
from .policy import LanePolicy, PolicyConfig, apply_flattened_batch
from .actions import orders_from

__all__ = ["BenchResult", "run_benchmark", "warm_state", "gate4_route_inputs",
           "GATE4_ENVS", "GATE4_WARM_S", "GATE4_STEPS", "GATE4_WARMUP"]

#: The canonical J1 gate-4 workload, frozen so the number is reproducible by
#: one command rather than by an ad-hoc script.  Every field here was chosen
#: by a measurement recorded in `docs/JAX_REWRITE_PLAN.md` §1.12, not by
#: taste: 4096 envs is the stable realistic batch, 150 s of warm-up is what
#: produces the live ~44-entity minion-bearing state (a cold `init_lane` has
#: no minions and flatters the number), and 60 timed steps after 5 warmups is
#: the shortest run that stopped moving between repeats -- 20/30-step samples
#: above 56k were discarded as unstable.  Routing is ON because the gate is
#: the routed number; `--no-route-table` is a labelled control, not a result.
GATE4_ENVS = 4096
GATE4_WARM_S = 150.0
GATE4_STEPS = 60
GATE4_WARMUP = 5

BLUE_NEXUS = (1131.8, 1426.3)
RED_NEXUS = (12760.9, 13026.1)


# `DEFAULT_ROUTE_ARTIFACT` is `sim/config.py`'s (re-exported: `movement_parity`
# imports it from here).


def gate4_route_inputs(artifact_path: Path | None = None, *,
                       table_disabled: bool = False):
    """Return the routed gate-4 inputs, or the explicitly named raw control.

    Mirrors :func:`lanerl_jax.parity.last_hit_drive.gate3_route_inputs` on
    purpose: gate 3 and gate 4 must not silently measure two different
    movement semantics.  Routing is the default because the gate is the routed
    number; the two-point control is reachable only by asking for it by name.
    """
    if table_disabled:
        return None, None
    path = DEFAULT_ROUTE_ARTIFACT if artifact_path is None else Path(artifact_path)
    if not path.exists():
        raise FileNotFoundError(
            f"route artifact not found: {path}. Build it with: "
            "python -m lanerl_jax.data.local_route_artifact "
            f"--out {path} --radius 35 --offset-radius 50")
    from ..data.local_route_artifact import load_local_route_artifact
    from ..sim.terrain_jax import map1_terrain

    artifact = load_local_route_artifact(path, pathfinding_radius=35.0)
    return artifact.as_jax(), map1_terrain()


def warm_state(warm_s: float = GATE4_WARM_S, *, seed: int = 0):
    """Step one env forward `warm_s` seconds so the benchmark has real minions.

    A cold `init_lane()` has no minions on the map.  Benchmarking it measures
    a mostly-empty entity table and reports a number the training loop will
    never see -- §1.12 records a 59,153 dec/s result that was discarded for
    exactly this reason.  So the canonical workload warms one environment and
    broadcasts it, which is also why the run prints its live entity count:
    a silently-cold state should be visible in the output, not inferred.

    Unrouted on purpose.  This only has to produce a realistic *population*;
    routing it would make the benchmark's setup depend on the artifact it is
    trying to time, and the champion is not under orders here anyway.
    """
    sim = SimConfig.scripted()

    @jax.jit
    def one(state):
        return env_advance(state, sim)

    state = init_lane(seed=seed)
    for _ in range(int(round(warm_s * DECISION_HZ))):
        state = one(state)
    return jax.block_until_ready(state)


def live_entities(state) -> int:
    """How many units are actually alive in `state` -- printed, not assumed."""
    return int(np.asarray(state.alive).sum())


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


def _decode(logits, state, slot_unit, key, frame=None, *, vision=None):
    """Sample an action and turn it into champion orders.

    One key split per step, which is the real per-decision cost. The screen
    heads name a point in the calibrated champion-centred viewport.
    """
    kb, kx, ky = jax.random.split(key, 3)
    button = jax.random.categorical(kb, logits.button)
    sx = jax.random.categorical(kx, logits.screen_x)
    sy = jax.random.categorical(ky, logits.screen_y)

    return orders_from((button, sx, sy), state, slot_unit, frame,
                       cfg_x=logits.screen_x.shape[-1],
                       cfg_y=logits.screen_y.shape[-1], vision=vision)


def run_benchmark(n_envs: int, steps: int = 60, warmup: int = 3,
                  seed: int = 0, *, route_table=None, terrain=None,
                  enable_collision: bool = True,
                  collision_terrain: bool = False,
                  defer_collision_terrain: bool = True,
                  initial_state=None, sim_config=None) -> BenchResult:
    """Time the trainer's env step. ``sim_config`` (a `SimConfig`) overrides
    the route/terrain/collision keywords; by default they build
    ``SimConfig.training`` with those flags, which ARE the training flags
    unless a caller changes them (`STRUCT-003`)."""
    if sim_config is None:
        sim_config = SimConfig.training(
            route_artifact=None, route_table=route_table, terrain=terrain,
        ).replace(enable_collision=enable_collision,
                  collision_terrain=collision_terrain,
                  defer_collision_terrain=defer_collision_terrain,
                  name="benchmark")
    sim = sim_config
    patch_params = sim.params
    frame = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE],
                            TOP_OUTER_TURRET[Team.RED], BLUE_NEXUS)
    red_frame = make_lane_frame(TOP_OUTER_TURRET[Team.RED],
                                TOP_OUTER_TURRET[Team.BLUE], RED_NEXUS)
    policy = LanePolicy(PolicyConfig())

    base = init_lane() if initial_state is None else initial_state
    states = jax.tree.map(lambda a: jnp.broadcast_to(a, (n_envs,) + a.shape), base)

    obs0 = build_observation(base, 0, frame, params=patch_params, vision=sim.vision)
    variables = policy.init(jax.random.key(seed), obs0.entities[None],
                            obs0.entity_pad_mask[None], obs0.self_vec[None],
                            obs0.global_vec[None])

    def observe_one(state):
        # both champions act; the policy is shared, which is the mirror setup
        blue = build_observation(state, 0, frame, params=patch_params, vision=sim.vision)
        red = build_observation(state, 1, red_frame, params=patch_params, vision=sim.vision)
        return jax.tree.map(lambda a, b: jnp.stack([a, b]), blue, red)

    def finish_one(state, obs, logits, key):
        orders = _decode(logits, state, obs.slot_unit, key, frame, vision=sim.vision)
        return env_step(state, orders, sim)

    @jax.jit
    def full_step(states, keys):
        obs = jax.vmap(observe_one)(states)
        logits = apply_flattened_batch(
            policy, variables, obs.entities, obs.entity_pad_mask,
            obs.self_vec, obs.global_vec)
        return jax.vmap(finish_one)(states, obs, logits, keys)

    @jax.jit
    def sim_step(states):
        return jax.vmap(lambda s: env_advance(s, sim))(states)

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
    sim = SimConfig.scripted()
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
        return jax.vmap(lambda s: env_advance(s, sim))(states)

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


def _main() -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="J1 gate 4: full-loop throughput, the way the gate is worded.")
    ap.add_argument("--envs", type=int, nargs="+", default=[GATE4_ENVS],
                    help="batch size(s) (default: the canonical %(default)s)")
    ap.add_argument("--steps", type=int, default=GATE4_STEPS,
                    help="timed steps (default: %(default)s; shorter runs were "
                         "measured to be unstable and must not be quoted)")
    ap.add_argument("--warmup", type=int, default=GATE4_WARMUP)
    ap.add_argument("--warm-s", type=float, default=GATE4_WARM_S,
                    help="seconds of simulation used to build the live "
                         "minion-bearing initial state (default: %(default)s). "
                         "0 uses a cold init_lane, which has no minions and is "
                         "NOT a gate result.")
    ap.add_argument("--route-artifact", type=Path, default=None)
    ap.add_argument("--no-route-table", action="store_true",
                    help="the PATH-001 two-point control. Faster, and NOT gate "
                         "evidence -- the gate is the routed number.")
    ap.add_argument("--no-smooth", action="store_true",
                    help="labelled control: skip the server's SmoothPath pass. "
                         "NOT a gate result -- it emits 6-9 waypoints where the "
                         "server emits 3 (see PATH-001).")
    ap.add_argument("--smooth-line-steps", type=int, default=None,
                    help="sweep SMOOTH_CAST_LINE_STEPS. Lowering it below the "
                         "measured population fails CLOSED (less smoothing, "
                         "reported via smooth_exhausted), but it is an "
                         "approximation and not the canonical setting.")
    ap.add_argument("--route-unroll", type=int, default=None,
                    help="override sim.local_pathing.ROUTE_LOOP_UNROLL, the "
                         "number of identical masked hop bodies chained per "
                         "while-loop iteration. Semantics-free: the route, the "
                         "status and the raw-hop boundary are unchanged, only "
                         "how much XLA loop control each hop pays. Printed in "
                         "the header so a swept number is never quoted as the "
                         "canonical one by accident.")
    ap.add_argument("--reset-bench", action="store_true",
                    help="also run gate 6 (reset cost vs one step)")
    ap.add_argument("--baseline", type=float, default=1129.0,
                    help="production stack's own logged decisions/s")
    a = ap.parse_args()

    if a.no_smooth or a.smooth_line_steps is not None:
        from lanerl_jax.sim import local_pathing, terrain_jax
        if a.smooth_line_steps is not None:
            terrain_jax.SMOOTH_CAST_LINE_STEPS = a.smooth_line_steps
            local_pathing.SMOOTH_CAST_LINE_STEPS = a.smooth_line_steps
        if a.no_smooth:
            _orig = local_pathing.build_local_waypoints
            local_pathing.build_local_waypoints = (
                lambda *args, **kw: _orig(*args, **{**kw, "smooth": False}))
            import lanerl_jax.sim.orders  # imports the symbol lazily; nothing to patch
    if a.route_unroll is not None:
        from ..sim import local_pathing
        local_pathing.ROUTE_LOOP_UNROLL = a.route_unroll

    target = a.baseline * 50.0
    routed = not a.no_route_table
    print(f"device: {jax.devices()[0]}")
    print(f"baseline: {a.baseline:,.0f} decisions/s (production stack, logged) "
          f"-> gate 4 target >= {target:,.0f}")
    print(f"mode: {'ROUTED (gate evidence)' if routed else 'NO-ROUTE CONTROL (not gate evidence)'}")
    if routed:
        from ..sim.local_pathing import ROUTE_LOOP_UNROLL
        print(f"route loop unroll: {ROUTE_LOOP_UNROLL}"
              + ("  <-- SWEPT, not the canonical setting" if a.route_unroll is not None else ""))

    route_table, terrain = gate4_route_inputs(a.route_artifact,
                                              table_disabled=a.no_route_table)
    if a.warm_s > 0:
        t0 = time.perf_counter()
        base = warm_state(a.warm_s)
        print(f"initial state: {a.warm_s:.0f}s warmed, {live_entities(base)} live "
              f"entities ({time.perf_counter() - t0:.1f}s to build)")
    else:
        base = None
        print("initial state: COLD init_lane -- no minions, NOT a gate result")
    print(f"protocol: {a.steps} timed steps after {a.warmup} warmups\n")

    for n in a.envs:
        try:
            r = run_benchmark(n, steps=a.steps, warmup=a.warmup,
                              route_table=route_table, terrain=terrain,
                              initial_state=base)
        except Exception as exc:                     # OOM is a real answer
            print(f"{n:>6} envs | FAILED: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        shortfall = (target - r.decisions_per_s) / target
        verdict = ("PASS" if r.decisions_per_s >= target
                   else f"fails by {shortfall:.2%}")
        if not routed:
            verdict += " (control)"
        print(f"{r.report(a.baseline)} | gate 4 {verdict}")

    if a.reset_bench:
        print("\nJ1 gate 6 (D11): reset cost vs. one step_decision\n")
        for n in a.envs:
            try:
                print(run_reset_benchmark(n).report())
            except Exception as exc:
                print(f"{n:>6} envs | FAILED: {type(exc).__name__}: {str(exc)[:120]}")


if __name__ == "__main__":
    _main()
