"""E33/E34: bounded, interleaved BC/random comparison on the vectorized trainer.

No changes to PPO or simulation. Separate runners/optimizers, shared compiled
functions. Frozen evaluations use held-out starts, mirror and a fixed heuristic.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax.serialization import from_state_dict, msgpack_restore, to_state_dict
from lanerl_rl.constants import BUTTON_INDEX

from .policy import PolicyConfig
from .ppo import PPOConfig
from .run_manifest import RunDir, file_sha256
from .vec_train import VecConfig, make_vec_train, prepare_bank
from ..sim.config import SimConfig, DEFAULT_ROUTE_ARTIFACT


def adjust_button_bias(params, shifts):
    """Experimental initialization only; retain all other checkpoint weights."""
    state = to_state_dict(params)
    bias = jnp.asarray(state['params']['button']['bias'])
    for name, amount in shifts.items():
        if name not in BUTTON_INDEX or not np.isfinite(amount):
            raise ValueError(f'invalid button bias shift: {name}={amount}')
        bias = bias.at[BUTTON_INDEX[name]].add(float(amount))
    state['params']['button']['bias'] = bias
    return from_state_dict(params, state)


def first_episode_rows(tr, seen, mode):
    """Keep exactly the first completed episode per environment, not reset tails."""
    done = np.asarray(tr.done_full[:, :, 0])
    cs, gold, xp = map(np.asarray, (tr.cs, tr.gold, tr.xp))
    rows = []
    for env in np.flatnonzero(~seen & done.any(axis=0)):
        t = np.flatnonzero(done[:, env])[0]
        rows.append(dict(env=int(env), opponent=mode,
                         cs=cs[t, env].tolist(), gold=gold[t, env].tolist(),
                         xp=xp[t, env].tolist()))
        seen[env] = True
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('experiment')
    a = p.parse_args()
    spec = json.loads(Path('experiments', a.experiment+'.json').read_text())
    started = time.monotonic()
    stopping = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda signum, frame: stopping.append(signum))
    def stop():
        return bool(stopping) or time.monotonic()-started >= spec['max_seconds']

    jax.config.update('jax_default_matmul_precision', 'highest')
    scratch = Path('/scratch') / (a.experiment+'-'+os.environ['SLURM_JOB_ID'])
    scratch.mkdir(parents=True, exist_ok=True)
    shutil.copytree(DEFAULT_ROUTE_ARTIFACT, scratch/'routes')
    checkpoint = Path(spec['bc_checkpoint'])
    shutil.copyfile(checkpoint, scratch/'bc.msgpack')
    shared = Path('/mnt/nfs/checkpoints/lanerl-jax') / a.experiment
    shared.mkdir(parents=True, exist_ok=True)
    pcfg = PolicyConfig(core='gru', core_norm=True, core_residual=True,
                        detach_critic=True)
    cfg = VecConfig(n_envs=128, rollout_steps=128, n_updates=spec['updates'],
                    n_minibatches=4, bank_size=16, start_jitter_s=20.,
                    lr_anneal=True, policy=pcfg,
                    ppo=PPOConfig.standard(lr=1e-5, entropy_coef=0.))
    sim = SimConfig.training(route_artifact=scratch/'routes').replace(step_ticks=6)
    bank = prepare_bank(cfg, sim, shared/'train_bank', seed=0)
    eval_bank = prepare_bank(cfg, sim, shared/'eval_bank', seed=1007)
    built = make_vec_train(cfg, sim, bank)
    initial = built['init_params'](jax.random.key(0))
    bc = from_state_dict(initial, msgpack_restore((scratch/'bc.msgpack').read_bytes())['params'])
    shifts = spec.get('bc_button_bias_shift')
    second = adjust_button_bias(bc, shifts) if shifts is not None else initial
    arms = []
    for arm_id, params in zip(spec['arms'], (bc, second)):
        run = RunDir(shared/arm_id, 'vec-s0', dict(
            train={'policy': pcfg._asdict()}, ppo=cfg.ppo._asdict(),
            vec={k:v for k,v in cfg._asdict().items() if k not in ('policy','ppo')},
            collector=dict(episode_s=600., step_ticks=6, start_near_wave=True,
                           start_jitter_s=20., unwalkable_click='noop'),
            environment='jax-vectorised', opponent='mirror-self-play',
            initialization=('BC' if arm_id == spec['arms'][0] else
                {'BC_button_bias_shift': shifts} if shifts is not None else 'random'),
            bc_source_sha256=file_sha256(checkpoint), paired_spec=spec,
            sim=sim.describe(), sim_fingerprint=sim.fingerprint()),
            notes='Matched initialization experiment; one training seed. Frozen evals in evaluations.jsonl.')
        run.keep_checkpoints = 0  # preserve every checkpoint; no destructive rotation
        arms.append(dict(id=arm_id, run=run, runner=built['initial_runner'](jax.random.key(0), params),
                         update=0, failed=False))
        print('ARM', arm_id, 'run dir', run.path, flush=True)
    status_path = shared/'study.json'
    def status(state):
        status_path.write_text(json.dumps(dict(status=state, elapsed_s=time.monotonic()-started,
            job=os.environ['SLURM_JOB_ID'], arms=[dict(id=x['id'], path=str(x['run'].path),
            update=x['update'], failed=x['failed']) for x in arms]), indent=2))
    def save(arm):
        r = arm['runner']
        arm['run'].save(int(r.step), arm['update'], {'params':r.params,
            'opt_state':r.opt_state, 'step':r.step}, latest=not arm['failed'])
    for arm in arms:
        save(arm)
    status('compiling')
    print('COMPILE training update', flush=True)
    update_fn = jax.jit(lambda r: built['run_chunk'](r, 1)).lower(arms[0]['runner']).compile()
    evals = {}
    for mode in ('mirror', 'lasthit'):
        eb = make_vec_train(cfg._replace(opponent=mode), sim, eval_bank)
        er = eb['initial_runner'](jax.random.key(2007), arms[0]['runner'].params)
        er = er._replace(deadline_ms=jnp.full_like(er.deadline_ms, 600000))
        print('COMPILE frozen evaluation', mode, flush=True)
        evals[mode] = (er, jax.jit(eb['collect']).lower(er).compile())
    print('TRAINING READY: compiled shared update and frozen evaluators', flush=True)

    def evaluate(arm):
        summaries = {}
        for mode, (template, fn) in evals.items():
            r = template._replace(params=arm['runner'].params)
            seen = np.zeros(cfg.n_envs, bool)
            rows = []
            # Full horizon bound even if all starting states were at t=0.
            for _ in range(int(cfg.episode_s*10/cfg.rollout_steps)+2):
                if stop():
                    return
                r, tr, _ = jax.block_until_ready(fn(r))
                rows.extend(first_episode_rows(tr, seen, mode))
                if seen.all():
                    break
            if not seen.all():
                raise RuntimeError('Frozen evaluation failed to finish all environments')
            cs = np.array([x['cs'] for x in rows])
            gold = np.array([x['gold'] for x in rows])
            summary = dict(update=arm['update'], opponent=mode, frozen=True,
                games=len(rows), unique_start_bank_states=16, start_seed=1007, action_seed=2007,
                cs_mean=cs.mean(axis=0).tolist(), gold_diff_mean=float((gold[:,0]-gold[:,1]).mean()),
                episodes=rows)
            with (arm['run'].path/'evaluations.jsonl').open('a') as f:
                f.write(json.dumps(summary)+'\n')
            print('FROZEN', arm['id'], 'u', arm['update'], mode,
                  'CS', summary['cs_mean'], 'gold_diff', summary['gold_diff_mean'], flush=True)
            summaries[mode] = summary
        return summaries
    status('running')
    gate_failed = False
    try:
        for arm in arms:
            arm['initial_eval'] = evaluate(arm)
        if spec.get('initial_cs_retention') is not None and not stop():
            baseline = arms[0]['initial_eval']['lasthit']['cs_mean'][0]
            candidate = arms[1]['initial_eval']['lasthit']['cs_mean'][0]
            gate_failed = candidate < spec['initial_cs_retention'] * baseline
            gate = dict(baseline_cs=baseline, candidate_cs=candidate,
                minimum_ratio=spec['initial_cs_retention'], passed=not gate_failed)
            (shared/'initialization_gate.json').write_text(json.dumps(gate,indent=2))
            print('INITIALIZATION GATE',json.dumps(gate),flush=True)
        while not gate_failed and not stop() and any(x['update'] < cfg.n_updates and not x['failed'] for x in arms):
            for arm in arms:
                if stop() or arm['failed'] or arm['update'] >= cfg.n_updates:
                    continue
                end = min(arm['update']+spec['block_updates'], cfg.n_updates)
                while arm['update'] < end and not stop():
                    t = time.monotonic()
                    r, metrics = jax.block_until_ready(update_fn(arm['runner']))
                    arm['runner'] = r
                    arm['update'] += 1
                    row = {k:float(np.asarray(v).reshape(-1)[0]) for k,v in metrics.items()}
                    row.update(update=arm['update'], step=int(r.step), update_s=time.monotonic()-t)
                    arm['run'].log(row)
                    if row['loss_nonfinite'] or not np.isfinite(row['policy_loss']):
                        arm['failed'] = True
                        print('DIVERGED', arm['id'], arm['update'], flush=True)
                        break
                    if arm['update'] % 25 == 0:
                        print('TRAIN', arm['id'], json.dumps(row), flush=True)
                    if arm['update'] % 100 == 0:
                        save(arm)
                        status('running')
                save(arm)
                status('running')
                if not arm['failed'] and (arm['update'] % spec['eval_every'] == 0 or arm['update'] == cfg.n_updates):
                    evaluate(arm)
    finally:
        for arm in arms:
            save(arm)
            arm['run'].set_results(status='initial_gate_failed' if gate_failed else 'diverged' if arm['failed'] else
                'finished' if arm['update'] == cfg.n_updates else 'interrupted', updates=arm['update'])
            arm['run'].close()
        status('initial_gate_failed' if gate_failed else 'complete' if all(x['update'] == cfg.n_updates for x in arms) else 'stopped')
    print('PROFILE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
