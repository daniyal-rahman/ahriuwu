"""PERF-005 bounded ray-kernel equivalence and N128 collection/update A/B."""
import argparse
import json
from pathlib import Path
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.obs import vision
from lanerl_jax.probes.perf005_ray_kernel import clear_ray_fused


def canary(interpret=False):
    rng = np.random.default_rng(51)
    grid = vision.map1_vision()
    count = 257 if interpret else 131073
    x0 = rng.uniform(-100, 15500, count).astype('float32')
    y0 = rng.uniform(-100, 15500, count).astype('float32')
    x1 = x0 + rng.uniform(-1500, 1500, count).astype('float32')
    y1 = y0 + rng.uniform(-1500, 1500, count).astype('float32')
    x1[::7] = x0[::7]  # vertical / horizontal / point rays
    y1[::11] = y0[::11]
    x0[::13] = np.round(x0[::13]/50)*50  # grid edges and corners
    y0[::13] = np.round(y0[::13]/50)*50
    enabled = jnp.asarray(rng.random(count) > .2)
    args = tuple(jnp.asarray(x) for x in (x0, y0, x1, y1))
    ref = jax.jit(lambda *x: vision.clear_ray(grid, *x, enabled=enabled))(*args)
    got = jax.jit(lambda *x: clear_ray_fused(grid, *x, enabled=enabled, interpret=interpret))(*args)
    np.testing.assert_array_equal(ref, got)
    # Nested batching matches collector env/team layouts; masked padding too.
    small = tuple(x[:256].reshape(4, 2, 32) for x in args)
    def batched(fn):
        return jax.jit(jax.vmap(jax.vmap(lambda *x: fn(grid, *x))))(*small)
    np.testing.assert_array_equal(batched(vision.clear_ray),
        batched(lambda *x: clear_ray_fused(*x, interpret=interpret)))
    print(f'RAY CHECK PASSED: {count} map rays and nested vmap outputs exactly match', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path)
    p.add_argument('--canary', action='store_true')
    p.add_argument('--interpret', action='store_true')
    a = p.parse_args()
    jax.config.update('jax_default_matmul_precision', 'highest')
    if a.canary:
        canary(a.interpret)
        return
    from flax.serialization import from_state_dict, msgpack_restore
    from lanerl_jax.sim.config import SimConfig
    from lanerl_jax.train.policy import PolicyConfig
    from lanerl_jax.train.ppo import PPOConfig
    from lanerl_jax.train.vec_train import VecConfig, make_vec_train, prepare_bank
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = VecConfig(n_envs=128, rollout_steps=128, n_updates=1, n_minibatches=4,
        bank_size=4, start_jitter_s=20., ppo=PPOConfig.standard(),
        policy=PolicyConfig(core='gru', core_norm=True, core_residual=True))
    sim = SimConfig.training().replace(step_ticks=6)
    bank = prepare_bank(cfg, sim, a.out/'bank', seed=0)
    base = make_vec_train(cfg, sim, bank)
    runner = base['initial_runner'](jax.random.key(0))
    checkpoint = Path('lanerl_jax/runs/E31_jax_noprior_gru/seed0/jax-farm-s0-20260928-184158-ec566530/ckpt_latest.msgpack')
    params = from_state_dict(runner.params, msgpack_restore(checkpoint.read_bytes())['params'])
    runner = base['initial_runner'](jax.random.key(0), params)
    runner = runner._replace(deadline_ms=jnp.full_like(runner.deadline_ms, 600000))
    result = dict(config=repr(cfg), checkpoint=str(checkpoint), stages={}, status='running')
    def save():
        (a.out/'result.json').write_text(json.dumps(result, indent=2))
    def compile_(name, fn):
        print('compile', name, flush=True)
        start = time.perf_counter()
        executable = jax.jit(fn).lower(runner).compile()
        result['stages'][name] = dict(compile_s=time.perf_counter()-start)
        return executable
    collect_base = compile_('reference_collect', base['collect'])
    update_base = compile_('reference_update', lambda r: base['run_chunk'](r, 1))
    for i in range(14):
        runner = jax.block_until_ready(collect_base(runner))[0]
    result['game_age_s'] = float(runner.env_state.t_ms.mean()/1000)
    vision.clear_ray = clear_ray_fused
    candidate = make_vec_train(cfg, sim, bank)
    collect_new = compile_('fused_ray_collect', candidate['collect'])
    update_new = compile_('fused_ray_update', lambda r: candidate['run_chunk'](r, 1))

    def compare(x, y):
        for aa, bb in zip(jax.tree.leaves(x), jax.tree.leaves(y)):
            if jnp.issubdtype(aa.dtype, jax.dtypes.prng_key):
                aa, bb = jax.random.key_data(aa), jax.random.key_data(bb)
            if jnp.issubdtype(aa.dtype, jnp.inexact):
                np.testing.assert_allclose(aa, bb, rtol=1e-5, atol=1e-5, equal_nan=True)
            else:
                np.testing.assert_array_equal(aa, bb)
    outputs = {}
    for name, fn in [('reference_collect', collect_base), ('fused_ray_collect', collect_new),
                     ('reference_update', update_base), ('fused_ray_update', update_new)]:
        outputs[name] = jax.block_until_ready(fn(runner))
    compare(outputs['reference_collect'], outputs['fused_ray_collect'])
    compare(outputs['reference_update'], outputs['fused_ray_update'])
    result['trajectory_and_update_equivalence'] = 'passed: discrete exact, floats rtol/atol 1e-5'
    print('COLLECT/UPDATE EQUIVALENCE PASSED', flush=True)
    # Alternate reference/candidate runs to reduce clock/order bias.
    for kind, left, right in [('collect', collect_base, collect_new), ('update', update_base, update_new)]:
        samples = {'reference': [], 'fused_ray': []}
        for repeat in range(4):
            pair = [('reference', left), ('fused_ray', right)]
            for name, fn in pair if repeat % 2 == 0 else pair[::-1]:
                start = time.perf_counter()
                jax.block_until_ready(fn(runner))
                samples[name].append(time.perf_counter()-start)
        for name, values in samples.items():
            result['stages'][name+'_'+kind].update(samples_s=values, median_s=statistics.median(values))
        result[kind+'_speedup'] = statistics.median(samples['reference'])/statistics.median(samples['fused_ray'])
        save()
        print(kind, samples, 'speedup', result[kind+'_speedup'], flush=True)
    result['status'] = 'complete'
    save()
    print('PROFILE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
