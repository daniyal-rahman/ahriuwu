"""Bounded AFK GRU imitation and DAgger study through the production click path.

Teacher labels use observations plus the existing script's public static data.
Full observation prefixes are replayed at current weights before each training
window. No artificial action delay, privileged actor inputs or physics changes.
"""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import time
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.serialization import from_state_dict, msgpack_restore

from ..obs.builder import Observation
from ..sim.config import DEFAULT_ROUTE_ARTIFACT, SimConfig
from .policy import PolicyConfig, VALUE_HEAD_NAME
from .ppo import PPOConfig, factored_log_prob, screen_head_usage
from .run_manifest import RunDir, file_sha256
from .scripted_policy import scripted_act
from .vec_train import VecConfig, make_vec_train
from .wave_evaluation import evaluate_frozen
from .wave_scenario import START_MS, park_afk_opponent, prepare_scenario_bank

OBS_KEYS = ('entities', 'mask', 'self', 'global')


def label_observations(e, m, s, g):
    """No unit IDs/state available to the scripted labeller."""
    obs = Observation(e, m, s, g, jnp.full(e.shape[:-1], -1, jnp.int32))
    return jnp.stack(scripted_act(obs, None), -1)


label_batch = jax.jit(jax.vmap(jax.vmap(label_observations)))


def episode_valid(done):
    """Time-major: include the first terminal action, exclude all later episodes."""
    prior = np.concatenate([np.zeros_like(done[:1]), np.cumsum(done, axis=0)[:-1]], axis=0)
    return prior == 0


def discounted_returns(reward, done, valid, gamma):
    """Agent-major true finite-episode returns, for frozen-actor critic fitting."""
    def step(carry, xs):
        r, d, v = xs
        carry = jnp.where(v, r + gamma * carry * (1-d), 0.)
        return carry, carry
    _, result = jax.lax.scan(step, jnp.zeros(reward.shape[0]),
        tuple(jnp.swapaxes(x, 0, 1) for x in (reward, done, valid)), reverse=True)
    return result.T


def apply_frame(policy, params, carry, e, m, s, g, reset):
    carry = jnp.where(reset[:, None], 0., carry)
    logits, carry = policy.apply(params, e, m, s, g, carry)
    return carry.astype(jnp.float32), logits


def replay_prefix(policy, params, batch, start):
    """Recompute all preceding observations; never reset at a window boundary."""
    def step(i, carry):
        return apply_frame(policy, params, carry,
            *(batch[k][:, i] for k in OBS_KEYS), batch['reset'][:, i])[0]
    return jax.lax.fori_loop(0, start, step,
        policy.initial_carry((batch['self'].shape[0],)))


def sequence_forward(policy, params, carry, batch):
    def step(c, xs):
        return apply_frame(policy, params, c, *xs)
    xs = tuple(jnp.swapaxes(batch[k], 0, 1) for k in (*OBS_KEYS, 'reset'))
    carry, logits = jax.lax.scan(step, carry, xs)
    return carry, jax.tree.map(lambda a: jnp.swapaxes(a, 0, 1), logits)


def imitation_loss(logits, labels, valid, attack_weight):
    use = screen_head_usage(labels[..., 0])[0]
    lp = factored_log_prob((logits.button, logits.screen_x, logits.screen_y), labels, use)
    weight = valid * jnp.where(labels[..., 0] == 2, attack_weight, 1.)
    loss = -(lp * weight).sum() / jnp.maximum(weight.sum(), 1.)
    predicted = jnp.stack([jnp.argmax(x, -1) for x in
        (logits.button, logits.screen_x, logits.screen_y)], -1)
    attack = valid * (labels[..., 0] == 2)
    attack_exact = ((predicted == labels).all(-1) * attack).sum() / jnp.maximum(attack.sum(), 1.)
    button_acc = ((predicted[..., 0] == labels[..., 0]) * valid).sum() / jnp.maximum(valid.sum(), 1.)
    return loss, dict(button_accuracy=button_acc, attack_exact=attack_exact)


def make_update(policy, tx, seq_len, attack_weight, *, critic_only=False):
    @jax.jit
    def update(params, opt_state, episodes, start):
        # Replayed using the current parameters, outside differentiation.
        carry = jax.lax.stop_gradient(replay_prefix(policy, params, episodes, start))
        batch = {k: jax.lax.dynamic_slice_in_dim(v, start, seq_len, axis=1)
                 for k,v in episodes.items()}
        def loss_fn(p):
            _, logits = sequence_forward(policy, p, carry, batch)
            if critic_only:
                valid = batch['valid']
                error = logits.value - batch['returns']
                return (error**2 * valid).sum()/jnp.maximum(valid.sum(), 1.), {}
            return imitation_loss(logits, batch['action'], batch['valid'], attack_weight)
        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, dict(loss=loss, grad_norm=optax.global_norm(grads), **metrics)
    return update


def setup_collector(cfg, sim, bank, params, seed, teacher=False):
    built = make_vec_train(cfg, sim, bank, blue_actor=scripted_act if teacher else None)
    runner = built['initial_runner'](jax.random.key(seed), params)
    indices = jnp.arange(cfg.n_envs) % len(bank.t_ms)
    runner = runner._replace(env_state=jax.tree.map(lambda b: b[indices], bank))
    return built, runner, jax.jit(built['collect'])


def capture(runner, collect, duration_s, stop):
    pieces = {k: [] for k in (*OBS_KEYS, 'action', 'behavior', 'done', 'reward')}
    seen = np.zeros(runner.env_state.t_ms.shape, bool)
    for _ in range(int(np.ceil(duration_s*10/128))+2):
        if stop():
            raise InterruptedError('stopped during demonstration collection')
        runner, tr, _ = jax.block_until_ready(collect(runner))
        obs = [np.asarray(x[:, :, 0]) for x in
               (tr.obs_entities, tr.obs_mask, tr.obs_self, tr.obs_global)]
        labels = np.asarray(label_batch(*(jnp.asarray(x) for x in obs)), np.int32)
        for k,v in zip(OBS_KEYS, obs):
            pieces[k].append(v)
        pieces['action'].append(labels)
        pieces['behavior'].append(np.stack([np.asarray(a[:, :, 0]) for a in tr.action], -1))
        done = np.asarray(tr.done_full[:, :, 0])
        pieces['done'].append(done)
        pieces['reward'].append(np.asarray(tr.cs_delta[:, :, 0]))
        seen |= done.any(0)
        if seen.all():
            break
    if not seen.all():
        raise RuntimeError('demonstration collection did not finish every first episode')
    data = {k: np.concatenate(v, axis=0) for k,v in pieces.items()}
    data['valid'] = episode_valid(data['done']).astype(np.float32)
    data['reset'] = np.concatenate([np.zeros_like(data['done'][:1]), data['done'][:-1]])
    data = {k: np.swapaxes(v, 0, 1) for k,v in data.items()}
    return data


def actor_unchanged(before, after):
    for key in before['params']:
        if key == VALUE_HEAD_NAME:
            continue
        for a,b in zip(jax.tree.leaves(before['params'][key]), jax.tree.leaves(after['params'][key])):
            if not bool(jnp.array_equal(a, b)):
                return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('experiment')
    args = parser.parse_args()
    spec = json.loads(Path('experiments', args.experiment+'.json').read_text())
    start_time = time.monotonic()
    stopping = []
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGUSR1):
        signal.signal(sig, lambda signum, frame: stopping.append(signum))
    def stop():
        return bool(stopping) or time.monotonic()-start_time >= spec['max_seconds']
    jax.config.update('jax_default_matmul_precision', 'highest')
    jax.config.update('jax_compilation_cache_dir', '/scratch/lanerl-jax-compilation-cache')
    scratch = Path('/scratch') / (spec['id']+'-'+os.environ['SLURM_JOB_ID'])
    scratch.mkdir(parents=True)
    shutil.copytree(DEFAULT_ROUTE_ARTIFACT, scratch/'routes')
    shutil.copyfile(spec['init_from'], scratch/'initial.msgpack')
    if file_sha256(scratch/'initial.msgpack') != spec['init_sha256']:
        raise RuntimeError('initial checkpoint SHA mismatch')
    out = Path('/mnt/nfs/checkpoints/lanerl-jax')/spec['id']
    out.mkdir(parents=True, exist_ok=True)
    sim = SimConfig.training(route_artifact=scratch/'routes').replace(step_ticks=6)
    train_bank = park_afk_opponent(prepare_scenario_bank(sim, out/'train_bank', spec['train_offsets'], 0))
    eval_bank = park_afk_opponent(prepare_scenario_bank(sim, out/'eval_bank', spec['eval_offsets'], 1007))
    pcfg = PolicyConfig(core='gru', core_norm=True, core_residual=True, detach_critic=True)
    cfg = VecConfig(n_envs=spec['teacher_episodes'], rollout_steps=128, n_updates=1, n_minibatches=4,
        episode_s=START_MS/1000+spec['duration_s'], observation_horizon_s=600.,
        stagger_initial=False, bank_size=len(train_bank.t_ms), policy=pcfg,
        opponent='afk', cs_only=True, xp_scale=0., tower_damage_personal=True,
        ppo=PPOConfig.standard(lr=1e-4, entropy_coef=.001))
    base = make_vec_train(cfg, sim, train_bank)
    initialized = base['init_params'](jax.random.key(0))
    params = from_state_dict(initialized, msgpack_restore((scratch/'initial.msgpack').read_bytes())['params'])
    policy = base['policy']
    tx = optax.chain(optax.clip_by_global_norm(.5), optax.adam(spec['bc_lr']))
    opt_state = tx.init(params)
    run = RunDir(out, 'gru-dagger-s0', dict(train={'policy': pcfg._asdict()}, scenario=spec,
        environment='jax-vectorised', initialization='E78 final actor, fresh supervised Adam',
        init_source_sha256=spec['init_sha256'], command='python3 ops/launch.py '+spec['id'],
        sim=sim.describe(), sim_fingerprint=sim.fingerprint()),
        notes='Scripted prior diagnostic, not a from-scratch PPO result. No action delays.')
    run.keep_checkpoints = 0
    step = 0
    stage = 0
    status = 'running'
    def save():
        ck = run.save(step, stage, dict(params=params, opt_state=opt_state, step=step))
        (out/'study.json').write_text(json.dumps(dict(job=os.environ['SLURM_JOB_ID'],
            path=str(run.path), checkpoint=str(ck), update=stage, status=status,
            elapsed_s=time.monotonic()-start_time), indent=2))
        return ck
    ecfg = cfg._replace(n_envs=64, bank_size=len(eval_bank.t_ms))
    _, er, eval_fn = setup_collector(ecfg, sim, eval_bank, params, 2007)
    evaluations = {'afk': (er, eval_fn)}
    def evaluate(stage_num):
        result = evaluate_frozen(SimpleNamespace(params=params), evaluations, cfg,
            spec, stage_num, run, stop)
        if result is None:
            raise InterruptedError('stopped during frozen evaluation')
        return result[0]
    update = make_update(policy, tx, spec['seq_len'], spec['attack_weight'])
    rng = np.random.default_rng(spec['train_seed'])
    datasets = []
    final_eval = None
    try:
        save()
        evaluate(0)  # exact retention against E78 final on the existing64 suite
        teacher_dir = out/'teacher'
        teacher_dir.mkdir()
        _, teacher_er, teacher_efn = setup_collector(ecfg, sim, eval_bank, params, 2007, teacher=True)
        teacher_spec = {k:v for k,v in spec.items() if k != 'initial_eval_reference'}
        teacher_result = evaluate_frozen(SimpleNamespace(params=params),
            {'afk': (teacher_er, teacher_efn)}, cfg, teacher_spec, 0,
            SimpleNamespace(path=teacher_dir), stop)
        if teacher_result is None:
            raise InterruptedError('stopped during teacher evaluation')
        for round_idx, epochs in enumerate(spec['epochs_per_round']):
            n = spec['teacher_episodes'] if round_idx == 0 else spec['dagger_episodes']
            dc = cfg._replace(n_envs=n)
            _, dr, collect = setup_collector(dc, sim, train_bank, params,
                3107+round_idx, teacher=round_idx == 0)
            data = capture(dr, collect, spec['duration_s'], stop)
            data_path = scratch/f'dataset_round{round_idx}.npz'
            np.savez_compressed(data_path, **data)
            shutil.copyfile(data_path, run.path/data_path.name)
            datasets.append(data)
            merged = {k: np.concatenate([d[k] for d in datasets]) for k in data if k != 'behavior'}
            device = jax.tree.map(jnp.asarray, merged)
            n_episodes, n_steps = merged['valid'].shape
            starts = np.arange(0, n_steps, spec['seq_len'])
            if n_steps % spec['seq_len'] or n_episodes % spec['batch_episodes']:
                raise ValueError('dataset must divide fixed recurrent batch/window sizes')
            agreement = ((data['behavior'] == data['action']).all(-1)*data['valid']).sum()/data['valid'].sum()
            print('DATASET', json.dumps(dict(round=round_idx, episodes=n_episodes,
                valid_decisions=float(merged['valid'].sum()), teacher_agreement=float(agreement))), flush=True)
            for epoch in range(epochs):
                order = rng.permutation(n_episodes).reshape(-1, spec['batch_episodes'])
                work = [(ids, int(t)) for ids in order for t in starts]
                rng.shuffle(work)
                totals = []
                for ids,t in work:
                    if stop():
                        raise InterruptedError('stopped during supervised training')
                    params, opt_state, metrics = jax.block_until_ready(update(params, opt_state,
                        jax.tree.map(lambda a:a[ids], device), jnp.int32(t)))
                    metrics = {k:float(v) for k,v in metrics.items()}
                    if not all(np.isfinite(v) for v in metrics.values()):
                        raise RuntimeError('nonfinite imitation update')
                    totals.append(metrics)
                    step += spec['batch_episodes']*spec['seq_len']
                row = dict(round=round_idx, epoch=epoch+1, step=step,
                    **{k:float(np.mean([m[k] for m in totals])) for k in totals[0]})
                run.log(row)
                if (epoch+1)%5 == 0:
                    print('IMITATION',json.dumps(row),flush=True)
                    save()
            stage = round_idx+1
            save()
            final_eval = evaluate(stage)
        # Fit only the detached value head on actual frozen-clone returns before PPO.
        # Fresh Adam is essential: actor momentum from imitation must not move it.
        _, dr, collect = setup_collector(cfg._replace(n_envs=spec['dagger_episodes']),
            sim, train_bank, params, 4107)
        values = capture(dr, collect, spec['duration_s'], stop)
        values.pop('behavior')
        values = jax.tree.map(jnp.asarray, values)
        values['returns'] = discounted_returns(values['reward'], values['done'],
            values['valid'], cfg.ppo.gamma)
        before = params
        opt_state = tx.init(params)
        value_update = make_update(policy, tx, spec['seq_len'], spec['attack_weight'], critic_only=True)
        n, length = values['valid'].shape
        for i in range(spec['critic_updates']):
            if stop():
                raise InterruptedError('stopped during frozen-actor value fitting')
            ids = rng.choice(n, spec['batch_episodes'], replace=False)
            t = int(rng.choice(np.arange(0,length,spec['seq_len'])))
            params, opt_state, metrics = jax.block_until_ready(value_update(params, opt_state,
                jax.tree.map(lambda a:a[ids], values), jnp.int32(t)))
            if not np.isfinite(float(metrics['loss'])):
                raise RuntimeError('nonfinite critic fit')
            step += spec['batch_episodes']*spec['seq_len']
            if (i+1)%64 == 0:
                run.log(dict(phase='critic_fit', update=i+1, **{k:float(v) for k,v in metrics.items()}))
        if not actor_unchanged(before, params):
            raise RuntimeError('critic fitting changed actor parameters')
        print('FROZEN ACTOR CANARY PASSED: only value head changed',flush=True)
        status = 'complete'
        ck = save()
        shutil.copyfile(ck, out/'final.msgpack')
        shutil.copyfile(run.path/'evaluations.jsonl',out/'evaluations.jsonl')
        blue = [r for r in final_eval['episodes'] if r['team']==0]
        teacher_blue = [r for r in teacher_result[0]['episodes'] if r['team']==0]
        score = float(np.mean([r['cs'] for r in blue]))
        teacher_score = float(np.mean([r['cs'] for r in teacher_blue]))
        handoff = dict(status='complete', checkpoint=str(out/'final.msgpack'),
            checkpoint_sha256=file_sha256(out/'final.msgpack'), evaluation_update=stage,
            frozen_cs=score, teacher_cs=teacher_score,
            competent=score>=max(11., .85*teacher_score), actor_unchanged_after_value_fit=True)
        (out/'handoff.json').write_text(json.dumps(handoff,indent=2))
        run.set_results(**handoff)
    except InterruptedError as exc:
        status = 'interrupted'
        print(str(exc),flush=True)
        raise
    except Exception:
        status = 'failed'
        raise
    finally:
        save()
        run.set_results(status=status, stage=stage)
        run.close()
    print('PROFILE COMPLETE',flush=True)


if __name__ == '__main__':
    main()
