"""PERF-003: actual vec GRU collection/learning timings and memory, no saved model.

Launched by ops/launch.py PERF003_gru_profile. Repeated measurements reuse the
same input, so benchmark optimizer outputs are discarded, not trained onward.
Separate compiled stages expose memory/time; their sum is not a fused timing.
"""
import argparse
import json
from pathlib import Path
import statistics
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--envs', type=int, required=True)
    p.add_argument('--rollout', type=int, default=128)
    p.add_argument('--reps', type=int, default=3)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--canary', action='store_true')
    a = p.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from lanerl_jax.sim.config import SimConfig
    from lanerl_jax.sim.init import init_lane
    from lanerl_jax.train.policy import PolicyConfig
    from lanerl_jax.train.ppo import PPOConfig
    from lanerl_jax.train.vec_train import VecConfig, make_vec_train, prepare_bank
    jax.config.update('jax_default_matmul_precision', 'highest')
    a.out.mkdir(parents=True, exist_ok=True)
    result = dict(envs=a.envs, rollout=a.rollout, device=str(jax.devices()[0]),
                  backend=jax.default_backend(), canary=a.canary, stages={})
    print(json.dumps(result), flush=True)
    cfg = VecConfig(n_envs=a.envs, rollout_steps=a.rollout, n_updates=1,
                    n_minibatches=4, bank_size=4, start_jitter_s=20.,
                    ppo=PPOConfig.standard(),
                    policy=PolicyConfig(core='gru', core_norm=True, core_residual=True))
    sim = SimConfig.training().replace(step_ticks=6)
    t = time.perf_counter()
    if a.canary:
        bank = jax.tree.map(lambda x: x[None], init_lane())
    else:
        bank = prepare_bank(cfg, sim, a.out / 'bank', seed=0)
    result['bank_s'] = time.perf_counter() - t
    print('bank ready', result['bank_s'], flush=True)
    built = make_vec_train(cfg, sim, bank)
    runner = jax.block_until_ready(built['initial_runner'](jax.random.key(0)))

    def nbytes(tree):
        return sum(x.size * x.dtype.itemsize for x in jax.tree.leaves(tree))

    def bench(name, fn, *args):
        print('compile', name, flush=True)
        t = time.perf_counter()
        executable = jax.jit(fn).lower(*args).compile()
        comp = time.perf_counter() - t
        mem = executable.memory_analysis()
        memory = {k: getattr(mem, k) for k in (
            'argument_size_in_bytes', 'output_size_in_bytes',
            'temp_size_in_bytes', 'alias_size_in_bytes')}
        out = jax.block_until_ready(executable(*args))
        samples = []
        for _ in range(a.reps):
            t = time.perf_counter()
            out = jax.block_until_ready(executable(*args))
            samples.append(time.perf_counter() - t)
        info = dict(compile_s=comp, median_s=statistics.median(samples),
                    samples_s=samples, compiled_memory=memory,
                    device_memory=jax.devices()[0].memory_stats())
        result['stages'][name] = info
        print(json.dumps(dict(stage=name, **info)), flush=True)
        return out

    collected = bench('collect', built['collect'], runner)
    learned = bench('learn', built['learn'], *collected)
    metrics = learned[1]
    assert float(metrics['loss_nonfinite']) == 0., metrics
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(learned[0].params))
    result['logical_bytes'] = dict(params=nbytes(runner.params),
                                   optimizer=nbytes(runner.opt_state),
                                   env_state=nbytes(runner.env_state),
                                   bank=nbytes(bank), transition=nbytes(collected[1]))
    result['transition_fields_bytes'] = {k: nbytes(v) for k, v in collected[1]._asdict().items()}
    if a.canary:
        fused_runner, fused_metrics = jax.block_until_ready(jax.jit(built['run_chunk'], static_argnums=1)(runner, 1))
        for x, y in zip(jax.tree.leaves(learned[0]), jax.tree.leaves(fused_runner)):
            if jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
                x, y = jax.random.key_data(x), jax.random.key_data(y)
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-4, rtol=1e-4)
        for key in metrics:
            np.testing.assert_allclose(np.asarray(metrics[key]), np.asarray(fused_metrics[key][0]),
                                       atol=1e-4, rtol=1e-4, equal_nan=True)
        print('CANARY PASSED: split/fused agreement; finite parameters and loss', flush=True)
    seconds = sum(result['stages'][k]['median_s'] for k in ('collect', 'learn'))
    result['split_champion_decisions_per_s'] = a.envs * 2 * a.rollout / seconds
    result['status'] = 'complete'
    (a.out / 'result.json').write_text(json.dumps(result, indent=2))
    print('RESULT', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
