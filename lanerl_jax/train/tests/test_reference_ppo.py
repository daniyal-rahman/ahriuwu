"""PPO-17 reference equivalence and rollout/update boundary regressions."""
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from lanerl_jax.train.learner import make_learner, make_update
from lanerl_jax.train.policy import ActionLogits, LanePolicy, PolicyConfig
from lanerl_jax.train.ppo import PPOConfig, factored_log_prob, gae, update_epochs
from .purejaxrl_reference import reference_functions


class Categorical:
    """Minimal distribution adapter; independent of production helpers."""
    def __init__(self, logits):
        self.logits = logits

    def log_prob(self, action):
        return jnp.take_along_axis(jax.nn.log_softmax(self.logits), action[..., None], -1)[..., 0]

    def entropy(self):
        lp = jax.nn.log_softmax(self.logits)
        return -(jnp.exp(lp) * lp).sum(-1)


class SingleHeadPolicy:
    cfg = SimpleNamespace(core="mlp")

    def apply(self, params, entities, mask, sv, gv):
        # Degenerate size-one click heads have zero entropy/log-prob: this
        # exercises the full production loss with precisely one categorical.
        return ActionLogits(sv @ params['actor'], jnp.zeros((*sv.shape[:-1], 1)),
                            jnp.zeros((*sv.shape[:-1], 1)), sv @ params['value'])


class ReferenceNetwork:
    def apply(self, params, carry, obs_done):
        sv, _ = obs_done
        return carry, Categorical(sv @ params['actor']), sv @ params['value']


def synthetic_batch(n=12):
    rng = np.random.default_rng(71)
    f = lambda shape: jnp.asarray(rng.normal(size=shape), jnp.float32)
    params = {'actor': f((4, 5)), 'value': f((4,))}
    batch = {'self': f((n, 4)), 'entities': f((n, 1)), 'mask': jnp.zeros((n, 1), bool),
             'global': f((n, 1)), 'value': f((n,)), 'log_prob': f((n,)),
             'adv': f((n,)) * 3 + 2, 'returns': f((n,))}
    batch['action'] = (jnp.arange(n) % 5, jnp.zeros(n, jnp.int32), jnp.zeros(n, jnp.int32))
    batch['uses_screen'] = jnp.zeros(n)
    return params, batch


def test_loss_and_gradients_match_vendored_reference():
    cfg = PPOConfig.standard()
    params, batch = synthetic_batch()
    ref_loss, _ = reference_functions(dict(CLIP_EPS=cfg.clip_eps, VF_COEF=cfg.value_coef,
                                          ENT_COEF=cfg.entropy_coef), ReferenceNetwork())
    trajectory = SimpleNamespace(obs=batch['self'], done=jnp.zeros(12, bool),
                                 action=batch['action'][0], value=batch['value'], log_prob=batch['log_prob'])
    _, loss = make_learner(SingleHeadPolicy(), cfg)
    expected, grads = jax.value_and_grad(ref_loss, has_aux=True)(
        params, jnp.zeros((1, 12, 1)), trajectory, batch['adv'], batch['returns'])
    actual, got_grads = jax.value_and_grad(lambda p: loss(p, batch, cfg), has_aux=True)(params)
    np.testing.assert_allclose(actual[0], expected[0], atol=1e-6, rtol=0)
    np.testing.assert_allclose([actual[1][k] for k in ('value_loss', 'policy_loss', 'entropy')],
                               expected[1], atol=1e-6, rtol=0)
    for got, want in zip(jax.tree.leaves(got_grads), jax.tree.leaves(grads)):
        np.testing.assert_allclose(got, want, atol=1e-6, rtol=0)


def test_gae_matches_vendored_reference_with_internal_and_final_dones():
    rng = np.random.default_rng(52)
    rewards, values = [jnp.asarray(rng.normal(size=(19, 4)), jnp.float32) for _ in range(2)]
    dones = jnp.asarray(rng.random((19, 4)) < .3).at[-1].set(jnp.array([True, False, True, False]))
    last_value = jnp.asarray([12., -4., 7., 9.], jnp.float32)
    resets = jnp.concatenate((jnp.ones_like(dones[:1]), dones[:-1]))
    # NamedTuple is required by lax.scan; use the upstream transition fields.
    from typing import NamedTuple
    class Trajectory(NamedTuple):
        done: object
        value: object
        reward: object
    _, ref_gae = reference_functions(dict(GAMMA=.99, GAE_LAMBDA=.95), None)
    want = ref_gae(Trajectory(resets, values, rewards), last_value, dones[-1])
    got = gae(rewards, values, dones, last_value, .99, .95)
    np.testing.assert_allclose(got, want, atol=1e-6, rtol=0)


def test_update_matches_explicit_reference_steps_and_schedule():
    cfg = PPOConfig.standard(epochs=3, n_minibatches=3)
    params, batch = synthetic_batch()
    steps_per_update = cfg.epochs * cfg.n_minibatches
    tx, loss = make_learner(SingleHeadPolicy(), cfg, anneal_steps=2 * steps_per_update)
    update = make_update(tx, loss, cfg)
    # Independent optimizer construction and Python epoch/minibatch loop.
    reference_tx = optax.chain(optax.clip_by_global_norm(.5), optax.adam(
        lambda count: cfg.lr * (1 - (count // steps_per_update) / 2), eps=1e-5))
    state, reference_state = tx.init(params), reference_tx.init(params)
    wanted, key = params, jax.random.key(33)
    got, got_state, got_key, info = update(params, state, batch, key)
    infos = []
    for _ in range(cfg.epochs):
        key, shuffle_key = jax.random.split(key)
        permutation = jax.random.permutation(shuffle_key, 12)
        for ids in permutation.reshape(3, 4):
            minibatch = jax.tree.map(lambda x: x[ids], batch)
            (_, diagnostic), grads = jax.value_and_grad(lambda p: loss(p, minibatch, cfg), has_aux=True)(wanted)
            delta, reference_state = reference_tx.update(grads, reference_state, wanted)
            wanted = optax.apply_updates(wanted, delta)
            infos.append(diagnostic)
    for actual, expected in zip(jax.tree.leaves((got, got_state)), jax.tree.leaves((wanted, reference_state))):
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
    np.testing.assert_array_equal(jax.random.key_data(got_key), jax.random.key_data(key))
    for k in infos[0]:
        np.testing.assert_allclose(info[k], jnp.mean(jnp.stack([i[k] for i in infos])), atol=1e-6)
    # Check decay boundary directly: same gradient, count 0..8 has constant
    # LR, count 9..17 has half LR, and count 18 has zero LR.
    scalar = jnp.asarray(0., jnp.float32)
    sched_tx, _ = make_learner(SingleHeadPolicy(), cfg, anneal_steps=18)
    opt_state = sched_tx.init(scalar)
    deltas = []
    for _ in range(19):
        delta, opt_state = sched_tx.update(jnp.ones_like(scalar), opt_state, scalar)
        deltas.append(float(delta))
    np.testing.assert_allclose(deltas[:9], deltas[0], atol=1e-8)
    np.testing.assert_allclose(deltas[9:18], np.asarray(deltas[:9]) * .5, atol=1e-8)
    assert deltas[18] == 0


def test_shuffle_keeps_whole_trajectories_and_carry_together():
    n, t = 8, 5
    sequence = jnp.arange(n)[:, None] * 10 + jnp.arange(t)[None, :]
    batch = {'value': sequence.astype(jnp.float32), 'carry0': jnp.arange(n)[:, None] * 10}
    def loss(p, b):
        valid = jnp.all(b['value'] == b['carry0'] + jnp.arange(t)[None, :])
        return (p - 1) ** 2, {'sequence_intact': valid.astype(jnp.float32), 'approx_kl': p ** 2}
    tx = optax.sgd(.1)
    p = jnp.asarray(0., jnp.float32)
    got, _, _, info = update_epochs(loss, tx, p, tx.init(p), batch, jax.random.key(1),
                                    epochs=3, n_minibatches=4, max_grad_norm=.5)
    assert float(info['sequence_intact']) == 1
    # Every step applies, even after the diagnostic KL grows.
    assert float(got) == pytest.approx(1 - .8 ** 12, abs=1e-6)


@pytest.mark.parametrize('click_mask', [False, True])
def test_gru_scan_replays_rollout_and_resets_after_done(click_mask):
    cfg = PolicyConfig(core='gru', d_model=8, n_layers=1, n_heads=1, ffn_dim=8,
                       ctx_dim=8, core_dim=8, mlp_hidden=8, mlp_layers=1)
    policy = LanePolicy(cfg)
    rng = np.random.default_rng(14)
    n, t = 2, 5
    f = lambda shape: jnp.asarray(rng.normal(size=shape), jnp.float32)
    batch = {'entities': f((n, t, cfg.n_slots, cfg.entity_dim)),
             'mask': jnp.zeros((n, t, cfg.n_slots), bool),
             'self': f((n, t, cfg.self_dim)), 'global': f((n, t, cfg.global_dim)),
             'carry0': f((n, cfg.core_dim)),
             'done': jnp.array([[True, False, True, False, False], [False, False, False, False, True]])}
    carry = batch['carry0']
    params = policy.init(jax.random.key(0), *(batch[k][:, 0] for k in ('entities', 'mask', 'self', 'global')), carry)
    from lanerl_jax.train.trainer import _sample
    rollout, actions, logprobs = [], [], []
    if click_mask:
        batch['click_mask'] = jnp.asarray(rng.random((n, t, 96, 54)) > .4)
    for i in range(t):
        logits, carry = policy.apply(params, *(batch[k][:, i] for k in ('entities', 'mask', 'self', 'global')), carry)
        action, lp, _ = _sample(logits, jax.random.key(i + 5), None,
                                click_mask=batch['click_mask'][:, i] if click_mask else None)
        rollout.append(logits); actions.append(action); logprobs.append(lp)
        carry = jnp.where(batch['done'][:, i, None], policy.initial_carry((n,)), carry)
    batch['action'] = jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *actions)
    _, loss = make_learner(policy, PPOConfig.standard())
    recomputed = jax.jit(loss.forward)(params, batch)
    expected = jax.tree.map(lambda *xs: jnp.stack(xs, axis=1), *rollout)
    for got, want in zip(recomputed, expected):
        np.testing.assert_allclose(got, want, atol=1e-6, rtol=1e-6)
    lp = factored_log_prob(recomputed[:3], batch['action'], click_mask=batch.get('click_mask'))
    np.testing.assert_allclose(lp, jnp.stack(logprobs, axis=1), atol=1e-6, rtol=1e-6)
    # Reset removes dependence on nonzero carry0 only for the terminated row.
    changed = loss.forward(params, {**batch, 'carry0': batch['carry0'] + 10})
    np.testing.assert_allclose(changed.button[0, 1:], recomputed.button[0, 1:], atol=1e-6)
    assert not np.allclose(changed.button[1, 1], recomputed.button[1, 1], atol=1e-6)


@pytest.mark.parametrize('click_mask', [False, True])
def test_optional_prior_penalty_is_added_to_reference_loss(click_mask):
    params, batch = synthetic_batch()
    cfg = PPOConfig.standard(kl_prior_coef=.7)
    prior = jax.tree.map(lambda p: p * .6, params)
    if click_mask:
        batch['click_mask'] = jnp.ones((12, 1, 1), bool)
    _, loss = make_learner(SingleHeadPolicy(), cfg, prior_params=prior)
    got, info = loss(params, batch, cfg)
    base, base_info = loss(params, batch, cfg._replace(kl_prior_coef=0))
    p = jax.nn.softmax(batch['self'] @ prior['actor'])
    want_kl = (p * (jax.nn.log_softmax(batch['self'] @ prior['actor']) -
                    jax.nn.log_softmax(batch['self'] @ params['actor']))).sum(-1).mean()
    np.testing.assert_allclose(info['kl_prior'], want_kl, atol=1e-6)
    np.testing.assert_allclose(got, base + .7 * want_kl, atol=1e-6)
    for name, val in base_info.items():
        np.testing.assert_allclose(info[name], val, atol=1e-6)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(jax.grad(lambda p: loss(p, batch, cfg)[0])(params)))


def test_detached_critic_preserves_outputs_and_zeroes_value_trunk_gradient():
    cfg = PolicyConfig(d_model=8, n_layers=1, n_heads=1, ffn_dim=8, ctx_dim=8,
                       core_dim=8, mlp_hidden=8, mlp_layers=1)
    policy = LanePolicy(cfg)
    rng = np.random.default_rng(23)
    f = lambda shape: jnp.asarray(rng.normal(size=shape), jnp.float32)
    n = 4
    batch = dict(entities=f((n, cfg.n_slots, cfg.entity_dim)), mask=jnp.zeros((n, cfg.n_slots), bool),
                 self=f((n, cfg.self_dim)), global_=f((n, cfg.global_dim)))
    batch['global'] = batch.pop('global_')
    params = policy.init(jax.random.key(1), *(batch[k] for k in ('entities', 'mask', 'self', 'global')))
    ppo = PPOConfig.standard()
    _, shared = make_learner(policy, ppo)
    _, detached = make_learner(LanePolicy(cfg._replace(detach_critic=True)), ppo)
    logits = shared.forward(params, batch)
    batch.update(action=tuple(jnp.zeros(n, jnp.int32) for _ in range(3)),
                 value=logits.value, returns=logits.value + 1, adv=f((n,)))
    batch['log_prob'] = factored_log_prob(logits[:3], batch['action'])
    for a, b in zip(logits, detached.forward(params, batch)):
        np.testing.assert_array_equal(a, b)
    shared_norms = jax.jit(lambda p: shared.trunk_grad_norms(p, batch, ppo))(params)
    detached_norms = jax.jit(lambda p: detached.trunk_grad_norms(p, batch, ppo))(params)
    assert float(shared_norms['g_trunk_value']) > 0
    assert float(detached_norms['g_trunk_value']) == 0
    for k in ('g_trunk_pg', 'g_trunk_entropy'):
        np.testing.assert_allclose(shared_norms[k], detached_norms[k], atol=1e-6)
