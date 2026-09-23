"""The PPO losses, checked against the PyTorch implementation they were ported
from -- which is J3's gate 2: "loss components, advantage statistics and entropy
match on an identical fixed batch".

Checking against the real `lanerl_rl.ppo` rather than against a reimplementation
of the formulas is the point. A second transcription of the same equations
agrees with the first whether or not either matches the paper.
"""
from __future__ import annotations

import numpy as np
import pytest

import jax.numpy as jnp

from lanerl_jax.train.ppo import (
    MAX_FACTORED_ENTROPY,
    USES_SCREEN_HEADS,
    USES_TARGET_HEAD,
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


def test_dual_clip_bounds_a_negative_advantage():
    """The whole point: `min(rA, clip(r)A)` is unbounded below when A < 0 and
    r explodes, so one bad minibatch produces an enormous gradient."""
    cfg = PPOConfig()
    adv = jnp.asarray([-1.0])
    old = jnp.asarray([0.0])
    huge = jnp.asarray([20.0])            # ratio = e^20
    loss, _ = policy_loss(huge, old, adv, cfg)
    assert float(loss) == pytest.approx(cfg.dual_clip, abs=1e-5)


def test_dual_clip_does_not_touch_a_positive_advantage():
    cfg = PPOConfig()
    adv = jnp.asarray([1.0])
    lp = jnp.asarray([0.5])
    old = jnp.asarray([0.0])
    loss, _ = policy_loss(lp, old, adv, cfg)
    assert float(loss) == pytest.approx(-(1.0 + cfg.clip_eps), abs=1e-5)


def test_policy_loss_matches_the_torch_surrogate():
    """Same equations, transcribed independently; a disagreement means one of
    the two transcriptions is wrong."""
    rng = np.random.default_rng(1)
    n = 512
    lp = rng.normal(scale=0.3, size=n).astype(np.float32)
    old = rng.normal(scale=0.3, size=n).astype(np.float32)
    adv = rng.normal(size=n).astype(np.float32)
    cfg = PPOConfig()

    ratio = torch.exp(torch.tensor(lp) - torch.tensor(old))
    a = torch.tensor(adv)
    surr1 = ratio * a
    surr2 = torch.clamp(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * a
    inner = torch.min(surr1, surr2)
    obj = torch.where(a < 0.0, torch.max(inner, cfg.dual_clip * a), inner)
    want = float(-obj.mean())

    got, stats = policy_loss(jnp.asarray(lp), jnp.asarray(old), jnp.asarray(adv), cfg)
    assert float(got) == pytest.approx(want, abs=1e-5)
    assert 0.0 <= float(stats["clip_frac"]) <= 1.0
    assert 0.0 <= float(stats["dual_clip_frac"]) <= 1.0


def test_value_loss_matches_the_torch_version():
    rng = np.random.default_rng(2)
    n = 256
    v = rng.normal(size=n).astype(np.float32)
    ov = rng.normal(size=n).astype(np.float32)
    ret = rng.normal(size=n).astype(np.float32)
    cfg = PPOConfig()

    tv, tov, tr = torch.tensor(v), torch.tensor(ov), torch.tensor(ret)
    unclipped = (tv - tr) ** 2
    clipped_v = tov + torch.clamp(tv - tov, -cfg.value_clip_eps, cfg.value_clip_eps)
    want = float(0.5 * torch.max(unclipped, (clipped_v - tr) ** 2).mean())

    got = value_loss(jnp.asarray(v), jnp.asarray(ov), jnp.asarray(ret), cfg)
    assert float(got) == pytest.approx(want, abs=1e-5)


def test_factored_entropy_ceiling_is_the_MASKED_action_space():
    """12.050 nats: `ln(sum_b exp(c_b))`, the supremum of the entropy that
    counts each auxiliary head only in proportion to the buttons that put it
    on the wire. It is attained by `softmax(c)` over the buttons with uniform
    auxiliary heads -- NOT by a uniform button, and NOT the 14.099 of the
    unmasked four-head sum, which counted screen and target heads the
    behaviour never used (`PPO-01`). The 9.940 in `lanerl_rl/ppo.py`'s notes
    is the unmasked sum for the 9x9 grid of 2026-09-11; both are history.
    """
    c = (USES_SCREEN_HEADS * (np.log(C.N_SCREEN_X) + np.log(C.N_SCREEN_Y))
         + USES_TARGET_HEAD * np.log(C.N_SLOTS))
    best = [c[None, :], jnp.zeros((1, C.N_SCREEN_X)),
            jnp.zeros((1, C.N_SCREEN_Y)), jnp.zeros((1, C.N_SLOTS))]
    assert float(factored_entropy(best)[0]) == pytest.approx(
        MAX_FACTORED_ENTROPY, abs=1e-4)
    assert MAX_FACTORED_ENTROPY == pytest.approx(12.050, abs=1e-3)
    uniform = [jnp.zeros((1, k)) for k in
               (len(C.BUTTONS), C.N_SCREEN_X, C.N_SCREEN_Y, C.N_SLOTS)]
    assert float(factored_entropy(uniform)[0]) == pytest.approx(5.084, abs=1e-3)
    unmasked = float(np.log(8) + np.log(96) + np.log(54) + np.log(32))
    assert unmasked == pytest.approx(14.099, abs=1e-3)
    assert MAX_FACTORED_ENTROPY < unmasked


def test_factored_log_prob_counts_only_the_heads_the_button_uses():
    """`train/actions.orders_from` is the only thing that decides which heads
    reach the wire: noop/recall/q/w/e read none, move reads screen_x/y,
    attack_move reads target (and screen when the slot is empty), r reads
    target. The port summed all four heads for every button, so an E cast --
    80% of the trained policy's decisions -- carried three heads of pure noise
    in its ratio (`PPO-01`).

    The torch reference's `_head_usage` marks q/w/e/r as using BOTH because
    its wire cast carries a point and a target; the JAX decoder's casts are
    self-casts (Q/W/E) or target-only (R), so the table differs on purpose.
    """
    import jax.nn as jnn

    rng = np.random.default_rng(3)
    widths = (len(C.BUTTONS), C.N_SCREEN_X, C.N_SCREEN_Y, C.N_SLOTS)
    n = len(C.BUTTONS)
    logits = [jnp.asarray(rng.normal(size=(n, k)).astype(np.float32))
              for k in widths]
    acts = [jnp.arange(n)] + [jnp.asarray(rng.integers(0, k, size=n))
                              for k in widths[1:]]
    got = factored_log_prob(logits, acts)
    uses = {"noop": (0, 0), "recall": (0, 0), "q": (0, 0), "w": (0, 0),
            "e": (0, 0), "move": (1, 0), "attack_move": (1, 1), "r": (0, 1)}
    for name, b in C.BUTTON_INDEX.items():
        lp = [float(jnn.log_softmax(lg[b])[int(a[b])])
              for lg, a in zip(logits, acts)]
        scr, tgt = uses[name]
        want = lp[0] + scr * (lp[1] + lp[2]) + tgt * lp[3]
        assert float(got[b]) == pytest.approx(want, abs=1e-5), name
    # And the negative form: re-rolling the screen/target logits under a
    # button that does not read them leaves its log-prob untouched.
    other = [logits[0]] + [jnp.asarray(rng.normal(size=(n, k)).astype(np.float32))
                           for k in widths[1:]]
    again = factored_log_prob(other, acts)
    for name in ("noop", "recall", "q", "w", "e"):
        b = C.BUTTON_INDEX[name]
        assert float(again[b]) == pytest.approx(float(got[b]), abs=1e-6), name
    assert float(again[C.BUTTON_INDEX["move"]]) != pytest.approx(
        float(got[C.BUTTON_INDEX["move"]]), abs=1e-3)
