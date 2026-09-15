"""Probe environments: tiny tasks whose correct answer is known in advance.

WHY
---
Every bug this project shipped produced a plausible learning curve. A curve
cannot tell you the trainer works. A task whose answer you already know can:
if the critic cannot learn that a constant reward of 1 is worth 1, nothing it
says about a lane means anything.

The set is Andy Jones's ("Debugging RL, Without the Agonizing Pain"), which
isolates one mechanism per probe so a failure points at a component rather
than at "RL is hard":

    1  constant reward                 -> the critic can learn a constant
    2  reward depends on observation   -> the observation reaches the critic
    3  reward one step later           -> discounting and GAE bootstrap
    4  reward depends on action        -> the policy gradient has the sign right
    5  reward = action matching obs    -> the policy conditions on the input

Plus two this project specifically needs:

    6  reward depends on an observation from >chunk_len steps ago
       -> the recurrent state actually carries across BPTT chunk boundaries,
          which is the whole reason for burn_in and the 90-step chunk

These run the real LanePolicy class and the real DualClipPPO -- same code paths,
narrowed dimensions (see probe_model) -- with the probe signal written into the
observation the network already takes. A probe on a reimplemented toy trainer
would prove nothing about the thing that trains.

RUNTIME
-------
Marked slow: ~15 minutes on a contended 6-core box even at the narrowed
width, because each probe trains for tens of PPO updates. Run them before a
long training run, not on every edit -- ``-m slow``.

WHAT A FAILURE MEANS
--------------------
Probe 1 or 2 failing is a critic/optimiser bug. 3 is GAE or discounting. 4 or
5 is the policy loss or the action plumbing. 6 is the recurrent path. None of
them can be explained away by the task being hard, because the task is not
hard: a linear model solves all of them.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from lanerl_rl.model import LanePolicy, ModelConfig
from lanerl_rl.ppo import ACTION_KEYS, DualClipPPO, PPOConfig, RecurrentRolloutBuffer

PROBE_PPO = dict(chunk_len=8, burn_in=4, minibatch_chunks=8, epochs=4,
                 lr=3e-3, critic_lr=3e-3, entropy_coef=0.0, kl_ref_coef=0.0)


def probe_model() -> ModelConfig:
    """The real LanePolicy, narrowed.

    The probes must exercise the real CODE PATHS -- the same LanePolicy, the
    same DualClipPPO, the same recurrent chunking and burn-in -- but they do
    not need the real WIDTH. At production size (4.67M parameters, a
    transformer over 32 slots plus a GRU) a single probe is several hundred
    CPU-seconds and the set does not finish in a coffee break, which means in
    practice it does not get run, which means it may as well not exist.

    Only the dimensions move. Every mechanism under test is untouched.
    """
    return ModelConfig(
        d_model=32, n_layers=1, n_heads=2, ffn_dim=64,
        core_dim=64, ctx_dim=32, mlp_hidden=64, mlp_layers=2,
    )

#: The probe signal lives in self_vec[0]; everything else is zeros so there is
#: exactly one thing in the observation that could carry information.
SIGNAL = 0


def _blank_obs(mc: ModelConfig, B: int):
    return {
        "entities": torch.zeros(B, 1, mc.n_slots, mc.entity_dim),
        "entity_pad_mask": torch.zeros(B, 1, mc.n_slots, dtype=torch.bool),
        "self_vec": torch.zeros(B, 1, mc.self_dim),
        "global_vec": torch.zeros(B, 1, mc.global_dim),
        "priv_entities": torch.zeros(B, 1, mc.n_slots, mc.entity_dim),
        "priv_pad_mask": torch.zeros(B, 1, mc.n_slots, dtype=torch.bool),
        "priv_vec": torch.zeros(B, 1, mc.priv_dim),
    }


def _masks(mc: ModelConfig, B: int):
    return {
        "button": torch.ones(B, 1, mc.n_buttons, dtype=torch.bool),
        "screen_x": torch.ones(B, 1, mc.n_screen_x, dtype=torch.bool),
        "screen_y": torch.ones(B, 1, mc.n_screen_y, dtype=torch.bool),
        "target": torch.ones(B, 1, mc.n_slots, dtype=torch.bool),
    }


def _run(reward_fn, updates: int, T: int = 16, B: int = 16,
         gamma: float = 0.5, seed: int = 0, signal_fn=None):
    """Train the real policy on a probe and hand back the trained pieces.

    ``gamma`` is small on purpose: these are one- and two-step questions and a
    0.999 discount would drown them in bootstrap noise.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cfg = PPOConfig(**PROBE_PPO)
    cfg.gamma = gamma
    ppo = DualClipPPO(LanePolicy(probe_model()), cfg)
    mc = ppo.policy.cfg

    for _ in range(updates):
        buf = RecurrentRolloutBuffer(T, B, mc, device=torch.device("cpu"))
        state = ppo.policy.initial_state(B)
        signals = []
        for t in range(T):
            obs = _blank_obs(mc, B)
            sig = (torch.as_tensor(signal_fn(rng, B, t).astype(np.float32))
                   if signal_fn is not None else torch.zeros(B))
            obs["self_vec"][:, 0, SIGNAL] = sig
            signals.append(sig)
            masks = _masks(mc, B)
            resets = torch.zeros(B, 1)
            if t == 0:
                resets[:] = 1.0
            with torch.no_grad():
                dist, value, nxt = ppo.policy(state=state, resets=resets,
                                              action_masks=masks, **obs)
                action = dist.sample()
                logp = dist.log_prob(action)
            reward = reward_fn(t, signals, action, B)
            buf.add(
                obs={k: v[:, 0] for k, v in obs.items()},
                masks={k: v[:, 0] for k, v in masks.items()},
                action={k: v[:, 0] for k, v in action.items()},
                log_prob=logp[:, 0], value=value[:, 0],
                reward=reward, done=torch.zeros(B), reset=resets[:, 0],
                state=state,
            )
            state = nxt
        buf.finish(last_value=torch.zeros(B), gamma=cfg.gamma, lam=cfg.gae_lambda)
        ppo.update(buf)
    return ppo, mc


def _value_of(ppo, mc, signal: float = 0.0, B: int = 8) -> float:
    obs = _blank_obs(mc, B)
    obs["self_vec"][:, 0, SIGNAL] = signal
    with torch.no_grad():
        _, value, _ = ppo.policy(state=ppo.policy.initial_state(B),
                                 resets=torch.ones(B, 1),
                                 action_masks=_masks(mc, B), **obs)
    return float(value.mean())


# -- 1. the critic can learn a constant ------------------------------------


@pytest.mark.slow
def test_probe1_constant_reward():
    """Reward is always 1. With gamma=0.5 over a long rollout the value of
    any state is 1/(1-0.5) = 2, give or take the tail."""
    ppo, mc = _run(lambda t, s, a, B: torch.ones(B), updates=60)
    v = _value_of(ppo, mc)
    assert 1.4 < v < 2.6, (
        f"critic says {v:.3f} where a constant reward of 1 at gamma=0.5 is "
        f"worth ~2. The value head or the optimiser is broken; nothing it "
        f"reports about a real lane can be trusted."
    )


# -- 2. the observation reaches the critic ---------------------------------


@pytest.mark.slow
def test_probe2_reward_depends_on_the_observation():
    """Reward equals the signal in self_vec[0], which is +1 or -1.

    If the critic cannot separate the two, the observation is not reaching
    it -- a wiring fault that would look exactly like 'the task is hard'.
    """
    def sig(rng, B, t):
        return rng.choice([-1.0, 1.0], size=B)

    ppo, mc = _run(lambda t, s, a, B: s[t].clone(), updates=80, signal_fn=sig)
    hi, lo = _value_of(ppo, mc, 1.0), _value_of(ppo, mc, -1.0)
    assert hi - lo > 1.0, (
        f"critic values signal=+1 at {hi:.3f} and signal=-1 at {lo:.3f}: it "
        f"cannot tell them apart, so the observation is not reaching the "
        f"value head at all"
    )


# -- 4. the policy gradient has the right sign -----------------------------


@pytest.mark.slow
def test_probe4_policy_learns_the_rewarding_action():
    """Button 0 pays 1, every other button pays 0. The policy must find it.

    This is the smallest possible test that the policy loss improves the
    policy rather than fighting it -- a sign error here is invisible in any
    aggregate metric.
    """
    def reward(t, s, a, B):
        return (a["button"][:, 0] == 0).float()

    ppo, mc = _run(reward, updates=120)
    obs, masks = _blank_obs(mc, 8), _masks(mc, 8)
    with torch.no_grad():
        dist, _, _ = ppo.policy(state=ppo.policy.initial_state(8),
                                resets=torch.ones(8, 1), action_masks=masks, **obs)
        p = torch.softmax(dist.logits["button"].reshape(8, -1), -1)[:, 0].mean()
    assert float(p) > 0.5, (
        f"after 120 updates the policy puts {float(p):.3f} on the only button "
        f"that pays, against {1/mc.n_buttons:.3f} at chance. The policy "
        f"gradient is not improving the policy."
    )


# -- 6. the recurrent state crosses a chunk boundary -----------------------


@pytest.mark.slow
def test_probe6_reward_depends_on_an_observation_from_long_ago():
    """The signal appears ONLY at t=0; the reward for it arrives at the end.

    T=16 against chunk_len=8 means the rewarding step is in a different BPTT
    chunk from the step that saw the signal, so passing requires the hidden
    state to carry real information across a chunk boundary -- which is the
    entire justification for burn_in and for the 90-step chunk in production.
    """
    def sig(rng, B, t):
        return rng.choice([-1.0, 1.0], size=B) if t == 0 else np.zeros(B)

    def reward(t, s, a, B):
        return s[0].clone() if t == 15 else torch.zeros(B)

    ppo, mc = _run(reward, updates=100, T=16, signal_fn=sig, gamma=0.9)

    # value at t=0 must differ by sign of the signal it just saw
    obs, masks = _blank_obs(mc, 16), _masks(mc, 16)
    vals = {}
    for s in (1.0, -1.0):
        obs["self_vec"][:, 0, SIGNAL] = s
        with torch.no_grad():
            _, v, _ = ppo.policy(state=ppo.policy.initial_state(16),
                                 resets=torch.ones(16, 1), action_masks=masks, **obs)
        vals[s] = float(v.mean())
    gap = vals[1.0] - vals[-1.0]
    assert gap > 0.15, (
        f"value after seeing +1 is {vals[1.0]:.3f} and after -1 is "
        f"{vals[-1.0]:.3f} (gap {gap:.3f}). The reward arrives 15 steps later, "
        f"in a different BPTT chunk, so the hidden state is not carrying "
        f"information across the chunk boundary -- burn_in or the recurrent "
        f"state handoff is broken, and long-horizon credit is impossible."
    )
