"""The learner must recompute exactly the log-probs the actor stored.

WHY THIS IS THE SHARPEST CHEAP TEST IN THE SUITE
------------------------------------------------
On data collected by the CURRENT parameters, before any gradient step, the
PPO importance ratio ``exp(log_prob_new - log_prob_old)`` must be exactly 1.
There is no approximation involved: it is the same network, the same inputs,
the same actions. Any deviation means the actor and the learner disagree
about what happened, and every gradient computed from that data is being
weighted by a number that should have been 1.

It is sharp because it fails on things nothing else here would notice:

* action masks applied on one side and not the other, or rebuilt differently
* the recurrent state at a chunk boundary not being the state the actor had
* burn-in not actually re-warming the hidden state
* actions stored under one head layout and read under another -- which this
  project has done before, when a rename changed ``move_x`` to ``screen_x``
  in some places and not others
* float dtype or device differences between collection and training

And it is nearly free: no server, no game, no training run.

WHAT A FAILURE IS NOT
---------------------
A ratio away from 1 on STALE data is expected and fine -- that is what PPO's
importance weighting is for, and this project deliberately trains on rollouts
up to ``max_staleness`` versions old. This test uses fresh data from the live
parameters precisely so that 1.0 is the known answer.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from lanerl_rl import constants as C
from lanerl_rl.model import LanePolicy, ModelConfig
from lanerl_rl.ppo import ACTION_KEYS, MASK_KEYS, DualClipPPO, PPOConfig, RecurrentRolloutBuffer


def _cfg() -> PPOConfig:
    # The production recurrent shape, so the chunking this test exercises is
    # the chunking training actually runs.
    return PPOConfig(chunk_len=16, burn_in=8, minibatch_chunks=4, epochs=1)


def _fill_buffer_from_policy(ppo: DualClipPPO, T: int, B: int, seed: int = 0):
    """Roll the LIVE policy forward and store what it actually chose.

    This mirrors what an actor does: act, record the log-prob the acting
    distribution assigned, move on. The learner then has to reproduce it.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mc = ppo.policy.cfg
    buf = RecurrentRolloutBuffer(T, B, mc, device=torch.device("cpu"))
    state = ppo.policy.initial_state(B)

    for t in range(T):
        obs = {
            "entities": torch.as_tensor(
                rng.normal(size=(B, 1, mc.n_slots, mc.entity_dim)).astype(np.float32)),
            "entity_pad_mask": torch.zeros(B, 1, mc.n_slots, dtype=torch.bool),
            "self_vec": torch.as_tensor(rng.normal(size=(B, 1, mc.self_dim)).astype(np.float32)),
            "global_vec": torch.as_tensor(rng.normal(size=(B, 1, mc.global_dim)).astype(np.float32)),
            "priv_entities": torch.as_tensor(
                rng.normal(size=(B, 1, mc.n_slots, mc.entity_dim)).astype(np.float32)),
            "priv_pad_mask": torch.zeros(B, 1, mc.n_slots, dtype=torch.bool),
            "priv_vec": torch.as_tensor(rng.normal(size=(B, 1, mc.priv_dim)).astype(np.float32)),
        }
        # Non-trivial masks: an all-true mask would not catch a side applying
        # masking the other does not.
        masks = {}
        for k, n in (("button", mc.n_buttons), ("screen_x", mc.n_screen_x),
                     ("screen_y", mc.n_screen_y), ("target", mc.n_slots)):
            m = torch.as_tensor(rng.random((B, 1, n)) > 0.3)
            m[:, :, 0] = True          # never hand the policy an all-masked head
            masks[k] = m
        # FLOAT: the learner feeds the model `buffer.resets`, which is float,
        # and gru_with_resets does `1 - resets`. Passing bool here would make
        # the actor take a different code path from the learner -- which is
        # precisely the class of mismatch this test exists to detect, so it
        # must not be introduced by the test itself.
        resets = torch.zeros(B, 1, dtype=torch.float32)
        if t == 0:
            resets[:] = 1.0

        with torch.no_grad():
            dist, value, nxt = ppo.policy(
                state=state, resets=resets, action_masks=masks, **obs)
            action = dist.sample()
            logp = dist.log_prob(action)

        buf.add(
            obs={k: v[:, 0] for k, v in obs.items()},
            masks={k: v[:, 0] for k, v in masks.items()},
            action={k: v[:, 0] for k, v in action.items()},
            log_prob=logp[:, 0],
            value=value[:, 0],
            reward=torch.as_tensor(rng.normal(size=B).astype(np.float32)),
            # FLOAT, not bool: the buffer stores these as floats and
            # compute_gae does `1 - dones`, which raises on a bool tensor.
            done=torch.zeros(B, dtype=torch.float32),
            reset=resets[:, 0],
            state=state,
        )
        state = nxt
    return buf


def test_the_ppo_ratio_is_exactly_one_on_fresh_data():
    torch.manual_seed(0)
    ppo = DualClipPPO(LanePolicy(ModelConfig()), _cfg())
    T, B = 64, 4
    buf = _fill_buffer_from_policy(ppo, T, B)
    buf.finish(last_value=torch.zeros(B), gamma=ppo.cfg.gamma,
               lam=ppo.cfg.gae_lambda)

    worst = 0.0
    n_batches = 0
    for batch in buf.iter_minibatches(ppo.cfg.chunk_len, ppo.cfg.burn_in,
                                 ppo.cfg.minibatch_chunks):
        bi = int(batch["burn_in"])
        with torch.no_grad():
            dist, _ = ppo._forward_chunk(batch)
            actions = {k: batch["actions"][k][:, bi:] for k in ACTION_KEYS}
            new_lp = dist.log_prob(actions)
        old_lp = batch["log_probs"][:, bi:]
        ratio = torch.exp(new_lp - old_lp)
        worst = max(worst, float((ratio - 1.0).abs().max()))
        n_batches += 1

    assert n_batches > 0, "no minibatches were produced; the probe is vacuous"
    # float32 through a transformer + GRU: 1e-4 is generous for arithmetic
    # reordering and tight enough that any real disagreement shows.
    assert worst < 1e-4, (
        f"the learner recomputed a log-prob the actor never assigned: worst "
        f"|ratio - 1| = {worst:.3e} on FRESH data from the live parameters, "
        f"where the answer is exactly 1. Every gradient from this data is "
        f"weighted by a number that should have been 1. Suspects, in order: "
        f"action masks rebuilt differently between collection and training; "
        f"the recurrent state at a chunk start not being the state the actor "
        f"had; burn-in not re-warming; actions stored under one head layout "
        f"and read under another."
    )


def test_the_probe_would_notice_a_disagreement():
    """The test above is worthless if a wrong log-prob still gives ratio 1.

    Perturb the stored log-probs and confirm the check fails -- otherwise a
    passing run proves nothing.
    """
    torch.manual_seed(0)
    ppo = DualClipPPO(LanePolicy(ModelConfig()), _cfg())
    T, B = 32, 4
    buf = _fill_buffer_from_policy(ppo, T, B)
    buf.log_probs += 0.01          # a 1% error in the ratio
    buf.finish(last_value=torch.zeros(B), gamma=ppo.cfg.gamma,
               lam=ppo.cfg.gae_lambda)

    worst = 0.0
    for batch in buf.iter_minibatches(ppo.cfg.chunk_len, ppo.cfg.burn_in,
                                 ppo.cfg.minibatch_chunks):
        bi = int(batch["burn_in"])
        with torch.no_grad():
            dist, _ = ppo._forward_chunk(batch)
            new_lp = dist.log_prob({k: batch["actions"][k][:, bi:] for k in ACTION_KEYS})
        worst = max(worst, float((torch.exp(new_lp - batch["log_probs"][:, bi:]) - 1.0).abs().max()))
    assert worst > 1e-3, (
        f"a deliberate 0.01 error in the stored log-probs produced |ratio-1| "
        f"= {worst:.3e}, so the real test cannot detect a disagreement either"
    )
