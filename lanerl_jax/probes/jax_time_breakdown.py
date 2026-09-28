"""PERF-002: small CPU component microbenchmark, no training or checkpoints.

Run: ops/login_capped.sh 8G 2 .venv-jax/bin/python -u -m
     lanerl_jax.probes.jax_time_breakdown
Times repeated inputs after compilation, with all output leaves synchronized.
Standalone component times do not sum exactly to a fused rollout's runtime.
"""
import json
import time
import statistics

import jax
import jax.numpy as jnp

from lanerl_jax.sim.config import SimConfig
from lanerl_jax.sim.init import init_lane
from lanerl_jax.sim.step import env_advance, env_apply
from lanerl_jax.obs.builder import build_observation
from lanerl_jax.parity.policy_driver import _lane_frames
from lanerl_jax.train.policy import LanePolicy, PolicyConfig
from lanerl_jax.train.actions import orders_from
from lanerl_jax.train.trainer import _sample


def measure(name, fn, *args):
    fn = jax.jit(fn)
    start = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    compile_s = time.perf_counter() - start
    for _ in range(3):
        jax.block_until_ready(fn(*args))
    samples = []
    for _ in range(3):
        start = time.perf_counter()
        for _ in range(30):
            jax.block_until_ready(fn(*args))
        samples.append((time.perf_counter() - start) * 1000 / 30)
    print(json.dumps(dict(component=name, compile_s=round(compile_s, 3),
                         median_ms=round(statistics.median(samples), 4),
                         repeat_ms=[round(x, 4) for x in samples])), flush=True)
    return out


def main():
    sim = SimConfig.training().replace(step_ticks=6)
    frames = _lane_frames()
    print(json.dumps(dict(device=str(jax.devices()[0]), envs=1, ticks=6,
                         fixture='150s idle champions, live waves; fixed input')), flush=True)
    warm = jax.jit(lambda s: jax.lax.fori_loop(0, 1500, lambda _, s: env_advance(s, sim), s))
    start = time.perf_counter()
    state = jax.block_until_ready(warm(init_lane()))
    print(json.dumps(dict(warm_s=round(time.perf_counter()-start, 2),
                         alive=int(state.alive.sum()), slots=state.alive.size)), flush=True)

    def observe(s):
        obs = [build_observation(s, t, frames[t], params=sim.params,
                                horizon_s=600., vision=sim.vision) for t in (0, 1)]
        return jax.tree.map(lambda a, b: jnp.stack([a, b]), *obs)

    obs = measure('observe_both', observe, state)
    policy = LanePolicy(PolicyConfig(core='gru', core_norm=True, core_residual=True))
    carry = policy.initial_carry((2,))
    params = policy.init(jax.random.key(0), obs.entities, obs.entity_pad_mask,
                         obs.self_vec, obs.global_vec, carry)

    def act(p, o, c, key):
        logits, c = policy.apply(p, o.entities, o.entity_pad_mask, o.self_vec, o.global_vec, c)
        return _sample(logits, key, ~o.entity_pad_mask, None), c

    sampled, _ = measure('gru_and_sample_both', act, params, obs, carry, jax.random.key(1))
    def decode(s, a):
        return orders_from(a, s, None, frames[0], snap_moves=False,
                           params=sim.params, vision=sim.vision, drop_unwalkable_moves=True)
    orders = measure('action_decode', decode, state, sampled[0])
    applied = measure('apply_orders_routing', lambda s, o: env_apply(s, o, sim), state, orders)
    measure('simulation_6_ticks', lambda s: env_advance(s, sim), applied)
    no_collision = sim.replace(enable_collision=False)
    measure('simulation_no_collision_CONTROL', lambda s: env_advance(s, no_collision), applied)
    no_help = sim.replace(enable_call_for_help=False)
    measure('simulation_no_call_for_help_CONTROL', lambda s: env_advance(s, no_help), applied)


if __name__ == '__main__':
    main()
