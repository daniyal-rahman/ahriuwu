"""Fixed-workload full simulator/GRU profiling. No optimizer outputs are saved.

Only invoked by the PERF004 experiment launcher. Scope instrumentation is
probe-local; the canary compares original and labelled lowered computations.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import statistics
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--canary', action='store_true')
    p.add_argument('--single-session', action='store_true',
                   help='Defer selected traces to one final profiler session (PERF004c recovery).')
    a = p.parse_args()
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax.serialization import from_state_dict, msgpack_restore
    from lanerl_jax.sim.config import SimConfig
    from lanerl_jax.sim.init import init_lane
    from lanerl_jax.train.policy import PolicyConfig
    from lanerl_jax.train.ppo import PPOConfig
    from lanerl_jax.train import vec_train
    from lanerl_jax.probes.perf004_instrument import install, PHASES
    jax.config.update('jax_default_matmul_precision', 'highest')
    a.out.mkdir(parents=True, exist_ok=True)
    cfg = vec_train.VecConfig(n_envs=2 if a.canary else 128,
        rollout_steps=4 if a.canary else 128, n_updates=1, n_minibatches=4,
        bank_size=4, start_jitter_s=20., ppo=PPOConfig.standard(),
        policy=PolicyConfig(core='gru', core_norm=True, core_residual=True))
    sim = SimConfig.training().replace(step_ticks=6)
    bank = (jax.tree.map(lambda x: x[None], init_lane()) if a.canary else
            vec_train.prepare_bank(cfg, sim, a.out / 'bank', seed=0))
    baseline = vec_train.make_vec_train(cfg, sim, bank)
    runner = jax.block_until_ready(baseline['initial_runner'](jax.random.key(0)))
    result = dict(device=str(jax.devices()[0]), jax=jax.__version__,
        source=json.loads((a.out.parent / 'launch.json').read_text())['source'],
        config=repr(cfg), sim=repr(sim), stages={}, cohorts={}, traces=[])
    pending_traces = []

    def save():
        (a.out / 'profile_manifest.json').write_text(json.dumps(result, indent=2))

    def compile_stage(tag, fn, args):
        print('compile', tag, flush=True)
        start = time.perf_counter()
        executable = jax.jit(fn).lower(*args).compile()
        memory = executable.memory_analysis()
        result['stages'][tag] = dict(compile_s=time.perf_counter()-start,
            memory={k: getattr(memory, k) for k in ('argument_size_in_bytes',
                'output_size_in_bytes', 'temp_size_in_bytes', 'alias_size_in_bytes')})
        with gzip.open(a.out / f'{tag}.hlo.txt.gz', 'wt') as f:
            f.write(executable.as_text())
        save()
        return executable

    def measure(tag, executable, args, trace=False, hlo=None):
        output = jax.block_until_ready(executable(*args))
        samples = []
        for _ in range(3):
            start = time.perf_counter()
            output = jax.block_until_ready(executable(*args))
            samples.append(time.perf_counter()-start)
        info = dict(samples_s=samples, median_s=statistics.median(samples),
                    device_memory=jax.devices()[0].memory_stats())
        if a.single_session:
            if tag in ('random_35_collect', 'E31_35_collect', 'E31_14_learn'):
                pending_traces.append((tag, executable, args, hlo))
        elif trace:
            directory = a.out / 'traces' / tag
            jax.profiler.start_trace(str(directory))
            start = time.perf_counter()
            with jax.profiler.TraceAnnotation(tag):
                output = jax.block_until_ready(executable(*args))
            wall = time.perf_counter()-start
            jax.profiler.stop_trace()
            info['profiled_s'] = wall
            result['traces'].append(dict(tag=tag, trace=str(directory.relative_to(a.out)),
                                         hlo=f'{hlo}.hlo.txt.gz', wall_s=wall))
        result['stages'][tag] = {**result['stages'].get(tag, {}), **info}
        print(tag, json.dumps(info), flush=True)
        save()
        return output

    if a.canary:
        # Lower before replacing global functions: captures the original graph.
        original_lowered = jax.jit(baseline['collect']).lower(runner)
        original_ir = original_lowered.compiler_ir('stablehlo').operation.get_asm(
            large_elements_limit=5, enable_debug_info=False)
        original = original_lowered.compile()
        original_collected = jax.block_until_ready(original(runner))
        original_learn_lowered = jax.jit(baseline['learn']).lower(*original_collected)
        original_learn_ir = original_learn_lowered.compiler_ir('stablehlo').operation.get_asm(
            large_elements_limit=5, enable_debug_info=False)
        original_learn = original_learn_lowered.compile()
        expected = jax.block_until_ready(original_learn(*original_collected))
    install()
    built = vec_train.make_vec_train(cfg, sim, bank)
    result['tick_phases'] = PHASES
    if a.canary:
        labelled_lowered = jax.jit(built['collect']).lower(runner)
        labelled_ir = labelled_lowered.compiler_ir('stablehlo').operation.get_asm(
            large_elements_limit=5, enable_debug_info=False)
        assert original_ir == labelled_ir, 'collect computation changed'
        learn_lowered = jax.jit(built['learn']).lower(*original_collected)
        assert original_learn_ir == learn_lowered.compiler_ir('stablehlo').operation.get_asm(
            large_elements_limit=5, enable_debug_info=False), 'learn computation changed'
    collect = compile_stage('collect', built['collect'], (runner,))
    collected = measure('initial_collect', collect, (runner,), a.canary, 'collect')
    learn = compile_stage('learn', built['learn'], collected)
    learned = measure('initial_learn', learn, collected, a.canary, 'learn')
    assert float(learned[1]['loss_nonfinite']) == 0.
    if a.canary:
        for actual_tree, expected_tree in ((collected, original_collected), (learned, expected)):
            for x, y in zip(jax.tree.leaves(actual_tree), jax.tree.leaves(expected_tree)):
                if jnp.issubdtype(x.dtype, jax.dtypes.prng_key):
                    x, y = jax.random.key_data(x), jax.random.key_data(y)
                np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-4, rtol=1e-4)
        from lanerl_jax.probes.perf004_analyze import summarize
        for entry in result['traces']:
            summary = summarize(a.out / entry['trace'], a.out / entry['hlo'], entry['wall_s'])
            assert summary['hlo_mapped_events'] > 0
            assert any('tick_' in row['name'] for row in summary['tables']['phase_inclusive']) if entry['tag'].endswith('collect') else True
        result['status'] = 'canary_passed'
        save()
        print('CANARY PASSED: identical lowered math; matching outputs; GPU source attribution', flush=True)
        return

    fused = compile_stage('fused', lambda r: built['run_chunk'](r, 1), (runner,))
    checkpoint = Path('lanerl_jax/runs/E31_jax_noprior_gru/seed0/jax-farm-s0-20260928-184158-ec566530/ckpt_latest.msgpack')
    checkpoint_bytes = checkpoint.read_bytes()
    trained = from_state_dict(runner.params, msgpack_restore(checkpoint_bytes)['params'])
    result['checkpoint'] = dict(path=str(checkpoint), sha256=hashlib.sha256(checkpoint_bytes).hexdigest())
    for policy_name, params in [('random', runner.params), ('E31', trained)]:
        current = built['initial_runner'](jax.random.key(0), params)
        # Cohort construction only: suppress initial deadline jitter so late
        # snapshots retain their game age. All benchmark calls use this state.
        current = current._replace(deadline_ms=jnp.full_like(current.deadline_ms, 600000))
        for index in range(36):
            if index in (0, 14, 35):
                tag = f'{policy_name}_{index:02d}'
                state = current.env_state
                def stats(x):
                    x = np.asarray(x)
                    return dict(min=float(x.min()), mean=float(x.mean()), max=float(x.max()))
                result['cohorts'][tag] = dict(t_ms=stats(state.t_ms),
                    alive=stats(state.alive.sum(-1)), missiles=stats(state.missile_alive.sum(-1)),
                    champion_hp=stats(state.hp[:, :2]), champion_cs=stats(state.cs[:, :2]),
                    deadline_ms=stats(current.deadline_ms))
                sampled = measure(tag+'_collect', collect, (current,), True, 'collect')
                updated = measure(tag+'_learn', learn, sampled, index == 14, 'learn')
                assert float(updated[1]['loss_nonfinite']) == 0.
                measure(tag+'_fused', fused, (current,), index == 14, 'fused')
                current = sampled[0]
            else:
                current = jax.block_until_ready(collect(current))[0]
            print('cohort advance', policy_name, index, flush=True)
    if a.single_session:
        directory = a.out / 'traces' / 'recovery'
        print('start single trace session', [x[0] for x in pending_traces], flush=True)
        jax.profiler.start_trace(str(directory))
        for tag, executable, args, hlo in pending_traces:
            start = time.perf_counter()
            with jax.profiler.TraceAnnotation(tag):
                jax.block_until_ready(executable(*args))
            wall = time.perf_counter()-start
            result['traces'].append(dict(tag=tag, trace=str(directory.relative_to(a.out)),
                hlo=f'{hlo}.hlo.txt.gz', wall_s=wall, annotation=tag))
        jax.profiler.stop_trace()
    result['status'] = 'complete'
    save()
    print('PROFILE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
