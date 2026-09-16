"""The lane reward, ported from ``lanerl_rl/reward.py`` -- core terms only.

Zero-sum, with the shape the source uses::

    r_team = raw_team - alpha * raw_other

At ``alpha = 1`` and with shaping off, ``r_blue == -r_red`` exactly. Alpha
anneals from 0.5 to 1 over training: starting below 1 lets the agent learn to
farm at all before the game becomes exactly adversarial.

Two terms are not what a naive reading gives, and both are documented failures
in the source
---------------------------------------------------------------------------

**``money`` is on EARNED gold, not the wallet.** Buying an item makes the wallet
*fall*, so ``money * delta(wallet)`` paid **-8.0** for a purchase. The fix is
load-bearing and survives here even though this sim has no shop: earned and
wallet coincide only while nothing is ever spent, and the moment items land the
distinction returns. `LaneState.gold` is cumulative earnings, so it is already
the right quantity -- but the reward asks for earnings explicitly rather than
relying on that coincidence.

**``hp_point`` is a POTENTIAL DIFFERENCE on hp fraction, not a raw delta.** So a
death costs exactly the potential drop and the respawn refunds it. Weight 4.0
rather than the published 2.0, raised so that trading can pay for its own
opportunity cost.

**The ambient trickle is removed from the money term.** 1.9 gold/s arrives
whether the agent plays or not, and paying for it rewards standing still. The
sim knows the rate exactly (`sim/rewards.AMBIENT_GOLD_*`), so this subtracts the
known quantity rather than estimating it.

Booked as NOT ported yet
------------------------
* potential-based last-hit shaping and ``lane_approach`` (policy-invariant
  terms; `F = gamma*Phi(s') - Phi(s)` with a gamma that MUST match the
  trainer's or the invariance is lost)
* ``kill`` credit, which needs killer attribution routed out of the sim
* ``tower_hp``

Their weights are carried in the config and multiplied by zero, so turning one
on is a one-line change and nothing silently reads as "included".
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from ..sim.rewards import AMBIENT_GOLD_AMOUNT, AMBIENT_GOLD_INTERVAL_MS
from ..sim.state import LaneState

__all__ = ["RewardWeights", "RewardConfig", "RewardState", "reward_init",
           "lane_reward", "AMBIENT_GOLD_PER_S"]

#: 0.95 per 500 ms, measured from the server -- see `sim/rewards.py`.
AMBIENT_GOLD_PER_S = AMBIENT_GOLD_AMOUNT / (AMBIENT_GOLD_INTERVAL_MS / 1000.0)


class RewardWeights(NamedTuple):
    """`runs/rl-league-0915e/resolved_config.json`. Unported terms are kept at
    their real weights but gated off, so nothing reads as included."""

    money: float = 0.008
    hp_point: float = 4.0
    death: float = -1.0
    exp: float = 0.001
    last_hit: float = 1.0
    # --- carried, not yet ported (see the module docstring) ---
    kill: float = -0.5
    tower_hp: float = 10.0
    lane_approach: float = 0.07


class RewardConfig(NamedTuple):
    weights: RewardWeights = RewardWeights()
    zero_sum_alpha_start: float = 0.5
    zero_sum_alpha_end: float = 1.0
    zero_sum_anneal_steps: int = 2_000_000
    subtract_ambient_gold: bool = True
    #: which terms are actually live; the rest multiply by zero
    enable_kill: bool = False
    enable_tower: bool = False
    enable_shaping: bool = False

    def alpha(self, train_step) -> jax.Array:
        if self.zero_sum_anneal_steps <= 0:
            return jnp.asarray(self.zero_sum_alpha_end)
        frac = jnp.clip(train_step / self.zero_sum_anneal_steps, 0.0, 1.0)
        return (self.zero_sum_alpha_start
                + frac * (self.zero_sum_alpha_end - self.zero_sum_alpha_start))


class RewardState(NamedTuple):
    """Per-champion previous values. ``(2,)`` arrays, blue then red."""

    gold: jax.Array
    xp: jax.Array
    hp_frac: jax.Array
    deaths: jax.Array
    cs: jax.Array
    primed: jax.Array          # the first step has no previous to difference


def reward_init(state: LaneState) -> RewardState:
    hp = jnp.where(state.max_hp[:2] > 0, state.hp[:2] / state.max_hp[:2], 0.0)
    return RewardState(
        gold=state.gold[:2], xp=state.xp[:2], hp_frac=hp,
        deaths=state.deaths[:2].astype(jnp.float32),
        cs=state.cs[:2].astype(jnp.float32),
        primed=jnp.zeros((), bool))


def lane_reward(state: LaneState, prev: RewardState, dt_s: float,
                cfg: RewardConfig = RewardConfig(), train_step=0):
    """One step of reward for both champions. Returns ``(reward(2,), new_prev)``.

    ``primed`` exists because the first call after a reset has no previous state
    to difference against. The source is explicit about the equivalent: a
    potential-based term would otherwise score the jump from the previous
    episode's final state to this one's initial state as a real transition.
    """
    w = cfg.weights
    hp = jnp.where(state.max_hp[:2] > 0, state.hp[:2] / state.max_hp[:2], 0.0)

    d_gold = state.gold[:2] - prev.gold
    if cfg.subtract_ambient_gold:
        d_gold = d_gold - AMBIENT_GOLD_PER_S * dt_s
    d_xp = state.xp[:2] - prev.xp
    d_hp = hp - prev.hp_frac                       # potential difference
    d_deaths = state.deaths[:2].astype(jnp.float32) - prev.deaths
    d_cs = state.cs[:2].astype(jnp.float32) - prev.cs

    raw = (w.money * d_gold
           + w.exp * d_xp
           + w.hp_point * d_hp
           + w.death * d_deaths
           + w.last_hit * d_cs)
    raw = jnp.where(prev.primed, raw, jnp.zeros_like(raw))

    # r_self - alpha * r_other
    alpha = cfg.alpha(train_step)
    reward = raw - alpha * raw[::-1]

    return reward, RewardState(
        gold=state.gold[:2], xp=state.xp[:2], hp_frac=hp,
        deaths=state.deaths[:2].astype(jnp.float32),
        cs=state.cs[:2].astype(jnp.float32),
        primed=jnp.ones((), bool))
