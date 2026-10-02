"""E86 DIAG: measure detached-critic clipping on real E82 PPO batches.

No policy is exported. The production update is reproduced on each identical
batch before the isolated clipping variant is measured.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.serialization import from_state_dict, msgpack_restore

from lanerl_jax.sim.config import DEFAULT_ROUTE_ARTIFACT, SimConfig
from lanerl_jax.train.learner import make_learner
from lanerl_jax.train.policy import PolicyConfig, VALUE_HEAD_NAME
from lanerl_jax.train.ppo import PPOConfig, factored_log_prob, gae, policy_loss, update_epochs, screen_head_usage
from lanerl_jax.train.run_manifest import file_sha256, git_provenance
from lanerl_jax.train.vec_train import VecConfig, make_vec_train
from lanerl_jax.train.wave_scenario import START_MS, park_afk_opponent, prepare_scenario_bank
from lanerl_jax.train.wave_scenario_train import warmup_staggered_runner


def _is_value(path):
    return any(getattr(part, "key", None) == VALUE_HEAD_NAME for part in path)


def _without_value(tree):
    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: jnp.zeros_like(leaf) if _is_value(path) else leaf, tree)


def _without_actor(tree):
    return jax.tree_util.tree_map_with_path(
        lambda path, leaf: leaf if _is_value(path) else jnp.zeros_like(leaf), tree)


def _clip_without_value_for_actor(max_norm):
    """Actor norm sets actor clip; value head retains production total clip."""
    def transform(grads, params):
        del params
        total = optax.global_norm(grads)
        actor = optax.global_norm(_without_value(grads))
        standard_scale = jnp.where(total < max_norm, 1., max_norm / total)
        actor_scale = jnp.where(actor < max_norm, 1., max_norm / actor)
        return jax.tree_util.tree_map_with_path(
            lambda path, leaf: leaf * (standard_scale if _is_value(path) else actor_scale), grads)
    return optax.stateless(transform)


def _first_minibatch(batch, rng, cfg):
    _, key = jax.random.split(rng)
    ids = jax.random.permutation(key, batch['value'].shape[0])[:batch['value'].shape[0] // cfg.n_minibatches]
    return jax.tree.map(lambda x: x[ids], batch)


def _module_deltas(before, after):
    groups = {}
    for (path, old), (_, new) in zip(jax.tree_util.tree_flatten_with_path(before)[0],
                                     jax.tree_util.tree_flatten_with_path(after)[0]):
        group = getattr(path[1], 'key', str(path[1])) if len(path) > 1 else str(path[0])
        groups.setdefault(group, []).append(jnp.sum(jnp.square(new - old)))
    return {name: float(jnp.sqrt(jnp.sum(jnp.stack(squares)))) for name, squares in groups.items()}


def _max_tree_error(a, b):
    return max(float(jnp.max(jnp.abs(x-y))) for x, y in zip(jax.tree.leaves(a), jax.tree.leaves(b)))


def _probability_report(loss, old, new, batch, cs):
    before = loss.forward(old, batch)
    after = loss.forward(new, batch)
    old_lp = batch['log_prob']
    new_lp = factored_log_prob((after.button, after.screen_x, after.screen_y),
                               batch['action'], batch['uses_screen'])
    delta = np.asarray(new_lp-old_lp)
    positive = np.asarray(batch['adv'] > 0)
    cs = np.asarray(cs)
    def selected(mask):
        return {'n': int(mask.sum()), 'mean_delta_logprob': float(delta[mask].mean()) if mask.any() else None,
                'fraction_increased': float((delta[mask] > 0).mean()) if mask.any() else None}
    def head_kl(a, b):
        la, lb = jax.nn.log_softmax(a), jax.nn.log_softmax(b)
        return jnp.sum(jnp.exp(la) * (la-lb), axis=-1)
    # Coordinates are used only by move, attack_move and R. This is the exact
    # KL of the action distribution, not the unconditional head-entropy sum.
    from lanerl_rl.constants import BUTTON_INDEX
    bp = jax.nn.softmax(before.button)
    p_use = sum(bp[..., BUTTON_INDEX[name]] for name in ('move', 'attack_move', 'r'))
    exact_kl = head_kl(before.button, after.button) + p_use * (
        head_kl(before.screen_x, after.screen_x) + head_kl(before.screen_y, after.screen_y))
    return {'exact_joint_kl_mean': float(jnp.mean(exact_kl)),
            'sampled_old_minus_new_logprob': float(jnp.mean(old_lp-new_lp)),
            'all': selected(np.ones(delta.shape, bool)),
            'positive_raw_advantage': selected(positive),
            'cs_event': selected(cs),
            'cs_event_positive_advantage': selected(cs & positive)}


def _single_positive_advantage_sanity(cfg):
    logits = jnp.zeros((3,), jnp.float32)
    old_lp = jax.nn.log_softmax(logits)[0]
    def objective(x):
        lp = jax.nn.log_softmax(x)[0]
        return policy_loss(lp[None], old_lp[None], jnp.ones((1,)),
                           cfg._replace(normalize_advantage=False))[0]
    gradient = jax.grad(objective)(logits)
    new = logits - .01 * gradient
    movement = float(jax.nn.log_softmax(new)[0] - old_lp)
    assert movement > 0 and np.isfinite(movement)
    return {'normalization': False, 'chosen_delta_logprob': movement,
            'gradient': np.asarray(gradient).tolist()}


def _e85_anchors(policy, spec, source_params):
    """Fixed E85 observations, with the WHOLE prefix replayed at each weight set."""
    path = Path(spec['e85_cases'])
    result = json.loads((path/'result.json').read_text())
    assert result['status'] == 'complete'
    assert result['spec']['checkpoint_sha256'] == spec['checkpoint_sha256']
    rows = result['rows']
    ids = np.asarray([row['case'] for row in rows])
    data = np.load(path/'histories.npz')
    stops = jnp.asarray(data['found_at'][ids])
    length = int(np.max(stops)) + 1
    obs = tuple(jnp.asarray(data[k][:length, ids]) for k in ('entities', 'mask', 'self', 'global_'))
    actions = jnp.asarray([[row['chosen_actions'][i] for i in (1, 2)] for row in rows])
    expected = jnp.asarray([row['original_value'] for row in rows])

    @jax.jit
    def forward(params, obs, stops, actions):
        def step(state, xs):
            c, b, x, y, v = state
            t, e, m, sv, gv = xs
            lg, nc = policy.apply(params, e, m, sv, gv, c)
            take = t == stops
            return (jnp.where((t < stops)[:, None], nc, c),
                    jnp.where(take[:, None], lg.button, b),
                    jnp.where(take[:, None], lg.screen_x, x),
                    jnp.where(take[:, None], lg.screen_y, y),
                    jnp.where(take, lg.value, v)), None
        n = actions.shape[0]
        initial = (policy.initial_carry((n,)), jnp.zeros((n, 8)),
                   jnp.zeros((n, 96)), jnp.zeros((n, 54)), jnp.zeros(n))
        (_, b, x, y, v), _ = jax.lax.scan(step, initial, (jnp.arange(obs[0].shape[0]), *obs))
        action = tuple(actions[..., i] for i in range(3))
        logits = tuple(jnp.broadcast_to(z[:, None], (n, 2, z.shape[-1])) for z in (b, x, y))
        lp = factored_log_prob(logits, action, screen_head_usage(action[0])[0])
        return lp, v

    def probabilities(params):
        lp, v = forward(params, obs, stops, actions)
        return np.asarray(lp), np.asarray(v)
    _, actual = probabilities(source_params)
    error = float(np.max(np.abs(actual - np.asarray(expected))))
    assert error < 1e-4, ('E85 full-prefix value reproduction', error)
    return probabilities, dict(cases=ids.tolist(), source_value_max_error=error,
        menu_recovery_gate=[row['validation_8s_delta_mean'][1] >= .5 and
                           row['validation_cs_delta_mean'][1] >= 0 for row in rows],
        columns=['selected_menu_one_action', 'selected_original_sample'],
        limitation='Representative click probability, not aggregate mass of equivalent actions. Only two E85 cases pass its recovery gate. Observations held fixed; prefixes recomputed at new weights. Mixed-batch regression on an individual case is not itself a bug.')


def main():
    spec = json.loads((Path('experiments') / (sys.argv[1] + '.json')).read_text())
    assert spec['engine'] == 'afk-optimizer-audit' and 1 <= spec['batches'] <= 2
    started = time.monotonic()
    stopped = []
    for sig in (signal.SIGTERM, signal.SIGUSR1, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.append(True))
    stop = lambda: bool(stopped) or time.monotonic()-started >= spec['max_seconds']
    out = Path('/mnt/nfs/shared') / spec['id']
    out.mkdir(exist_ok=False)
    scratch = Path('/scratch') / (spec['id'] + '-' + os.environ['SLURM_JOB_ID'])
    scratch.mkdir()
    shutil.copytree(DEFAULT_ROUTE_ARTIFACT, scratch/'routes')
    source = Path(spec['checkpoint'])
    staged = scratch / source.name
    shutil.copyfile(source, staged)
    actual_sha = file_sha256(staged)
    assert actual_sha == spec['checkpoint_sha256'], (actual_sha, spec['checkpoint_sha256'])
    payload = msgpack_restore(staged.read_bytes())
    jax.config.update('jax_default_matmul_precision', 'highest')
    jax.config.update('jax_compilation_cache_dir', '/scratch/lanerl-jax-compilation-cache')
    sim = SimConfig.training(route_artifact=scratch/'routes').replace(step_ticks=6)
    bank = park_afk_opponent(prepare_scenario_bank(sim, out/'bank', [-60, -20, 20, 60], 0))
    pcfg = PolicyConfig(core='gru', core_norm=True, core_residual=True, detach_critic=True)
    ppo = PPOConfig.standard(lr=1e-4, entropy_coef=.001, discount=.99, epochs=4)
    cfg = VecConfig(n_envs=128, rollout_steps=128, n_minibatches=4,
                    episode_s=START_MS/1000+120., observation_horizon_s=600.,
                    stagger_initial=True, bank_size=len(bank.t_ms), opponent='afk',
                    cs_only=True, policy=pcfg, ppo=ppo, lr_anneal=False)
    built = make_vec_train(cfg, sim, bank)
    template = built['init_params'](jax.random.key(0))
    params = from_state_dict(template, payload['params'])
    runner = built['initial_runner'](jax.random.key(0), params)
    fresh_state = runner.opt_state
    saved_state = from_state_dict(fresh_state, payload['opt_state'])
    # Fresh near-wave trajectories, but compare both fresh and mature E82 Adam
    # states on each identical batch. No E82 rollout/RNG state is restored.
    runner = runner._replace(opt_state=saved_state)
    runner, warmup = warmup_staggered_runner(built, runner, cfg,
        stop=stop)
    tx, loss = make_learner(built['policy'], ppo._replace(n_minibatches=cfg.n_minibatches))
    variant_tx = optax.chain(_clip_without_value_for_actor(ppo.max_grad_norm),
                             optax.adam(ppo.lr, eps=1e-5))
    assert jax.tree.structure(tx.init(params)) == jax.tree.structure(variant_tx.init(params))
    report = {'experiment': spec['id'], 'checkpoint_sha256': actual_sha,
              'source': git_provenance(),
              'optimizer': 'fresh and E82 saved Adam compared on each identical batch; E82 rollout/RNG not restored',
              'warmup': warmup, 'single_action_sanity': _single_positive_advantage_sanity(ppo),
              'batches': [], 'complete': False,
              'interpretation_limit': 'Identical-batch updates are diagnostic only. Adam can cancel a common gradient scale, especially with fresh moments; a clip-scale difference need not produce an equal parameter or policy difference. No frozen CS outcome or causal action attribution.'}
    anchor_fn, anchor_info = _e85_anchors(built['policy'], spec, params)
    report['e85_full_prefix'] = anchor_info
    (out/'result.json').write_text(json.dumps(report, indent=2))
    rollout = jax.jit(built['rollout'])
    production_learn = jax.jit(built['learn'])
    def bootstrap(r):
        last_obs, _ = jax.vmap(built['observe'])(r.env_state, r.visible_history)
        logits, _ = jax.vmap(lambda o, c: built['policy'].apply(r.params,
            o.entities, o.entity_pad_mask, o.self_vec, o.global_vec, c))(last_obs, r.carry)
        return logits.value
    boot = jax.jit(bootstrap)
    standard = jax.jit(lambda p, s, b, k: update_epochs(lambda q, x: loss(q, x, ppo),
        tx, p, s, b, k, epochs=ppo.epochs, n_minibatches=cfg.n_minibatches,
        max_grad_norm=ppo.max_grad_norm))
    variant = jax.jit(lambda p, s, b, k: update_epochs(lambda q, x: loss(q, x, ppo),
        variant_tx, p, s, b, k, epochs=ppo.epochs, n_minibatches=cfg.n_minibatches,
        max_grad_norm=ppo.max_grad_norm))
    for index in range(spec['batches']):
        if stop():
            raise TimeoutError('E86 max worker duration before next batch')
        before = runner
        after, tr, batch = jax.block_until_ready(rollout(before))
        adv, returns = gae(tr.reward, tr.value, tr.done, boot(after), ppo.gamma, ppo.gae_lambda)
        def rows(x):
            x = jnp.moveaxis(x[:, :, :1], 0, 2)
            return x.reshape((cfg.n_envs, cfg.rollout_steps) + x.shape[3:])
        batch = dict(batch, adv=rows(adv), returns=rows(returns))
        old_logits = loss.forward(before.params, batch)
        replay_lp = factored_log_prob((old_logits.button, old_logits.screen_x,
                                      old_logits.screen_y), batch['action'], batch['uses_screen'])
        likelihood_error = float(jnp.max(jnp.abs(replay_lp-batch['log_prob'])))
        assert likelihood_error < 1e-4, likelihood_error
        first = _first_minibatch(batch, after.rng, cfg)
        (_, _), grads = jax.value_and_grad(lambda p: loss(p, first, ppo), has_aux=True)(before.params)
        total_norm = float(optax.global_norm(grads))
        actor_norm = float(optax.global_norm(_without_value(grads)))
        value_norm = float(optax.global_norm(_without_actor(grads)))
        standard_scale = 1. if total_norm < ppo.max_grad_norm else ppo.max_grad_norm/total_norm
        actor_scale = 1. if actor_norm < ppo.max_grad_norm else ppo.max_grad_norm/actor_norm
        cs = rows(tr.cs_delta) > 0
        record = {'index': index, 'cs_events': int(jnp.sum(cs)),
                  'module_parameter_norm': _module_deltas(jax.tree.map(jnp.zeros_like, before.params), before.params),
                  'actor_learner_old_logprob_max_error': likelihood_error,
                  'first_minibatch': {'total_raw_gradient_norm': total_norm,
                      'actor_raw_gradient_norm': actor_norm, 'value_head_raw_gradient_norm': value_norm,
                      'standard_clip_scale': standard_scale, 'actor_only_clip_scale': actor_scale},
                  'optimizers': {}}
        anchor_before, _ = anchor_fn(before.params)
        saved_production = None
        for mode, state in (('fresh', fresh_state), ('restored_e82', after.opt_state)):
            if stop():
                raise TimeoutError('E86 signal/time bound before optimizer comparison')
            isolated = after._replace(opt_state=state)
            prod, metrics = jax.block_until_ready(production_learn(isolated, tr, before.carry))
            std_params, std_state, std_key, _ = jax.block_until_ready(
                standard(before.params, state, batch, after.rng))
            err = _max_tree_error(prod.params, std_params)
            state_err = _max_tree_error(prod.opt_state, std_state)
            assert (err <= 1e-6 and state_err <= 1e-6 and
                    bool(jnp.array_equal(prod.rng, std_key))), (mode, err, state_err)
            alt_params, _, alt_key, alt_metrics = jax.block_until_ready(
                variant(before.params, state, batch, after.rng))
            assert bool(jnp.array_equal(std_key, alt_key))
            assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(alt_params))
            assert not bool(metrics['loss_nonfinite']) and not bool(alt_metrics['loss_nonfinite'])
            record['optimizers'][mode] = {
                'production_reproduction': {'param_max_error': err, 'opt_state_max_error': state_err,
                    'rng_exact': True},
                'standard': {'module_parameter_delta_norm': _module_deltas(before.params, std_params),
                    'probability_stored_carry_replay': _probability_report(loss, before.params, std_params, batch, cs)},
                'exclude_value_from_actor_clip': {
                    'module_parameter_delta_norm': _module_deltas(before.params, alt_params),
                    'probability_stored_carry_replay': _probability_report(loss, before.params, alt_params, batch, cs)},
                'standard_vs_variant_parameter_max_error': _max_tree_error(std_params, alt_params),
                'standard_metrics': {k: float(v) for k, v in metrics.items()}}
            anchor_std, _ = anchor_fn(std_params)
            anchor_alt, _ = anchor_fn(alt_params)
            record['optimizers'][mode]['e85_full_prefix'] = dict(
                before_logprob=anchor_before.tolist(),
                standard_delta_logprob=(anchor_std-anchor_before).tolist(),
                variant_delta_logprob=(anchor_alt-anchor_before).tolist())
            if mode == 'restored_e82':
                saved_production = prod
        report['batches'].append(record)
        (out/'result.json').write_text(json.dumps(report, indent=2))
        print('E86 BATCH', index, 'clip', standard_scale, actor_scale,
              'restored_adam_param_difference',
              record['optimizers']['restored_e82']['standard_vs_variant_parameter_max_error'], flush=True)
        runner = saved_production
    report['complete'] = True
    report['elapsed_s'] = time.monotonic()-started
    (out/'result.json').write_text(json.dumps(report, indent=2))
    print('PROFILE COMPLETE', flush=True)


if __name__ == '__main__':
    main()
