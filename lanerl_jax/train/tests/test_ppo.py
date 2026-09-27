"""PPO-17: reference surrogate and current three-head action contract.

Torch GAE/value cross-checks remain; dual clipping and four-head entropy
tests were replaced because neither describes the reference learner.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from lanerl_jax.train.ppo import (
    PPOConfig,
    factored_entropy,
    factored_log_prob,
    gae,
    gamma_for_horizon,
    policy_loss,
    value_loss,
)

torch = pytest.importorskip("torch", reason="the reference implementation is torch")
from lanerl_rl import constants as C  # noqa: E402
from lanerl_rl.ppo import PPOConfig as TorchCfg  # noqa: E402
from lanerl_rl.ppo import compute_gae as torch_gae  # noqa: E402


def test_gamma_matches_the_production_config():
    """`resolved_config.json` from the last league run, and ppo.py's own
    worked examples -- including the 15 Hz ones, which is the case its
    docstring got wrong for as long as the field existed."""
    assert PPOConfig().gamma == pytest.approx(0.9997222222222222)
    assert gamma_for_horizon(30, 30) == pytest.approx(0.998889, abs=1e-6)
    assert gamma_for_horizon(45, 30) == pytest.approx(0.999259, abs=1e-6)
    assert gamma_for_horizon(30, 15) == pytest.approx(0.997778, abs=1e-6)
    assert TorchCfg().gamma == pytest.approx(PPOConfig().gamma)


def test_gae_matches_the_torch_implementation():
    rng = np.random.default_rng(0)
    T, B = 64, 8
    rewards = rng.normal(size=(T, B)).astype(np.float32)
    values = rng.normal(size=(T, B)).astype(np.float32)
    dones = (rng.random((T, B)) < 0.05).astype(np.float32)
    last = rng.normal(size=(B,)).astype(np.float32)
    cfg = PPOConfig()

    t_adv, t_ret = torch_gae(torch.tensor(rewards), torch.tensor(values),
                             torch.tensor(dones), torch.tensor(last),
                             cfg.gamma, cfg.gae_lambda)
    j_adv, j_ret = gae(jnp.asarray(rewards), jnp.asarray(values),
                       jnp.asarray(dones), jnp.asarray(last),
                       cfg.gamma, cfg.gae_lambda)
    np.testing.assert_allclose(np.asarray(j_adv), t_adv.numpy(), atol=1e-4)
    np.testing.assert_allclose(np.asarray(j_ret), t_ret.numpy(), atol=1e-4)


def test_gae_does_not_bootstrap_through_a_terminal_step():
    """`dones[t] == 1` must cut the chain, not merely discount it."""
    cfg = PPOConfig()
    T = 5
    rewards = jnp.zeros((T, 1))
    values = jnp.ones((T, 1))
    dones = jnp.zeros((T, 1)).at[2].set(1.0)
    adv, _ = gae(rewards, values, dones, jnp.ones((1,)), cfg.gamma, cfg.gae_lambda)
    # at t=2 the only term left is -value
    assert float(adv[2, 0]) == pytest.approx(-1.0, abs=1e-5)


def test_reference_surrogate_has_no_dual_clip():
    # PPO-17 replaces the old negative-advantage dual-clip assertion.
    cfg = PPOConfig(normalize_advantage=False)
    got, _ = policy_loss(jnp.asarray([4.0]), jnp.asarray([0.0]), jnp.asarray([-1.0]), cfg)
    assert float(got) == pytest.approx(np.exp(4.0), rel=1e-6)


def test_policy_loss_matches_normalized_clipped_torch_surrogate():
    rng = np.random.default_rng(1)
    lp, old, adv = (rng.normal(size=512).astype(np.float32) for _ in range(3))
    cfg = PPOConfig.standard()
    ratio = torch.exp(torch.tensor(lp) - torch.tensor(old))
    a = torch.tensor(adv)
    a = (a - a.mean()) / (a.std(unbiased=False) + 1e-8)
    want = -torch.minimum(ratio * a, torch.clamp(ratio, .8, 1.2) * a).mean()
    got, stats = policy_loss(jnp.asarray(lp), jnp.asarray(old), jnp.asarray(adv), cfg)
    assert float(got) == pytest.approx(float(want), abs=1e-6)
    assert 0 <= float(stats["clip_frac"]) <= 1


def test_value_loss_matches_the_torch_version():
    rng = np.random.default_rng(2)
    n = 256
    v = rng.normal(size=n).astype(np.float32)
    ov = rng.normal(size=n).astype(np.float32)
    ret = rng.normal(size=n).astype(np.float32)
    cfg = PPOConfig()

    tv, tov, tr = torch.tensor(v), torch.tensor(ov), torch.tensor(ret)
    unclipped = (tv - tr) ** 2
    clipped_v = tov + torch.clamp(tv - tov, -cfg.clip_eps, cfg.clip_eps)
    want = float(0.5 * torch.max(unclipped, (clipped_v - tr) ** 2).mean())

    got = value_loss(jnp.asarray(v), jnp.asarray(ov), jnp.asarray(ret), cfg)
    assert float(got) == pytest.approx(want, abs=1e-5)


def test_three_head_logprob_and_unconditional_entropy():
    # PPO-17 retires four-head wire-action ceilings: the policy has no target.
    from lanerl_jax.train.ppo import _chosen, _entropy, screen_head_usage, MAX_SCREEN_CLICK_ENTROPY
    import jax
    heads = tuple(jax.random.normal(jax.random.key(i), (8, k)) for i, k in enumerate((8, 96, 54)))
    actions = (jnp.arange(8), jnp.arange(8), jnp.arange(8))
    used, _ = screen_head_usage(actions[0])
    expected = _chosen(heads[0], actions[0]) + used * (_chosen(heads[1], actions[1]) + _chosen(heads[2], actions[2]))
    np.testing.assert_allclose(factored_log_prob(heads, actions), expected, atol=1e-6)
    np.testing.assert_allclose(factored_log_prob(heads, jnp.stack(actions, -1)), expected, atol=1e-6)
    np.testing.assert_allclose(factored_entropy(heads), sum(_entropy(h) for h in heads), atol=1e-6)
    assert MAX_SCREEN_CLICK_ENTROPY == pytest.approx(np.log(8 * 96 * 54))
    with pytest.raises(ValueError):
        factored_log_prob(heads + (heads[0],), actions + (actions[0],))


def test_standard_defaults():
    cfg = PPOConfig.standard()
    assert cfg.gamma == .99
    assert cfg.lr == 2.5e-4 and cfg.gae_lambda == .95
    assert cfg.clip_eps == .2 and cfg.value_coef == .5
    assert cfg.entropy_coef == .01 / 3 and cfg.max_grad_norm == .5
    assert cfg.epochs == cfg.n_minibatches == 4
    assert cfg.normalize_advantage
