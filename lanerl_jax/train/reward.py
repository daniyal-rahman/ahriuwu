"""The lane reward, ported from ``lanerl_rl/reward.py`` -- core terms only.

Zero-sum, exactly::

    r_team = raw_team - raw_other          (+ the per-agent lane potential)

so with shaping ignored ``r_blue == -r_red``. The source mixed with an
``alpha`` annealed 0.5 -> 1; that anneal is gone (below), and alpha is the
constant 1.

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

**The ambient trickle is removed from the money term.** ~1.84 gold/s arrives
whether the agent plays or not, and paying for it rewards standing still. The
sim knows the rate exactly (`sim/rewards.AMBIENT_GOLD_*`), so this subtracts the
known quantity rather than estimating it -- from 90 s, when the sim starts
paying it (`REW-07`).

``lane_approach``: why the walk has to be paid for
--------------------------------------------------
Blue spawns at (26, 280) and the wave meets at (3907, 13243) -- **13,532 units,
39 s of walking**. With no term that notices the walk, the reward is *exactly
zero* everywhere along it: `money` has the ambient trickle subtracted out, and
gold, xp, hp and cs cannot move until the champion reaches minions. A uniform
random policy covers `sqrt(18000) * 11.5 ~ 1,500` units of displacement in a
whole 600 s episode. It never arrives, so it never sees a reward, so there is
no gradient -- which is why CS was 0.00000 for 400 updates.

The fix is a **potential**, not a per-step bonus, and the distinction is the
whole point: `F = gamma*Phi(s') - Phi(s)` is policy-invariant (Ng, Harada &
Russell 1999), so it changes which policy is FOUND and never which policy is
optimal. A naive "closer to lane than last tick" bonus is not a potential and
pays an agent to oscillate toward and away from lane forever.

**The gamma MUST be the trainer's gamma or the invariance is lost**, so it is a
required keyword argument rather than a config default -- see
:func:`lane_reward`.

Removed for the baseline (2026-09-23)
-------------------------------------
Recoverable from commit ``490bb38`` (`reward.py` and the trainer's
``zero_sum_alpha`` metric):

* ``RewardConfig.zero_sum_alpha_start/_end/zero_sum_anneal_steps`` and
  ``RewardConfig.alpha()``: the 0.5 -> 1.0 anneal. Its clock counted
  champion-decisions (65,536 per update), so it ended at update ~31, before
  any full episode, and restarted on every resume (`PPO-06`/`REW-02`).
  Alpha is now the constant 1 and ``lane_reward`` has no ``train_step``.
* ``RewardConfig.enable_kill/enable_tower/enable_shaping`` and
  ``RewardWeights.kill/tower_hp``: unported terms, unread, then guarded by a
  ``NotImplementedError`` (`REW-06`). Kills are still paid through gold/XP
  (`REW-01`).
* ``RewardWeights.last_hit`` (0.0) and ``RewardState.cs``: a term multiplied
  by zero, logged as an always-zero ``reward_last_hit``.
* ``RewardConfig.enable_lane_approach`` and ``subtract_ambient_gold``
  (both True, no CLI flag): toggles whose off paths no run could reach.
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

import numpy as np

from ..sim.init import TOP_OUTER_TURRET
from ..sim.rewards import (AMBIENT_GOLD_AMOUNT, AMBIENT_GOLD_DELAY_MS,
                           AMBIENT_GOLD_PERIOD_TICKS)
from ..sim.state import LaneState, Team

__all__ = ["RewardWeights", "RewardConfig", "RewardState", "reward_init",
           "lane_reward", "AMBIENT_GOLD_PER_S", "LANE_HALF_WIDTH",
           "NEXUS_POSITION", "LANE_AXIS", "lane_approach_potential",
           "lane_corridor_distance"]

#: ``lanerl_rl.constants.LANE_HALF_WIDTH``. The corridor the potential
#: saturates inside, so it stops paying exactly where being in lane begins.
LANE_HALF_WIDTH = 1400.0

#: ``lanerl_rl.constants.NEXUS_POSITION`` -- the handedness reference only.
NEXUS_POSITION = {Team.BLUE: (1131.0, 1426.0), Team.RED: (12760.0, 13026.0)}


def _lane_axis():
    """Per-team lane frames, as ``(2,)`` arrays indexed blue-then-red.

    Ported from ``lanerl_rl.frame.LaneFrame``. ``s`` is progress from the
    agent's own outer turret towards the enemy's, ``n`` the perpendicular
    offset, and the normal is flipped so the agent's own nexus lies at
    ``n < 0``.

    The handedness rule is load-bearing rather than cosmetic. In a same-lane
    1v1 both champions occupy the **same physical corridor** and merely enter
    it from opposite ends, so red's axis is exactly antiparallel to blue's and
    the plain left-hand normal comes out with opposite signs for the two
    agents. Forcing "own nexus at n < 0" makes both adopt the same world
    normal, and the map between the frames becomes ``(s, n) -> (L - s, n)``:
    a reflection across the lane's perpendicular bisector. That is the correct
    symmetry for a duel, and it is why the two champions' potentials are
    mirror images instead of unrelated numbers.

    Computed once in float64 numpy at import: it is a fixed property of the
    map, so there is nothing to trace and nothing to recompute per step.
    """
    ox, oy, ax, ay, nx, ny, ln = [], [], [], [], [], [], []
    for team in (Team.BLUE, Team.RED):
        enemy = Team.RED if team == Team.BLUE else Team.BLUE
        o = np.asarray(TOP_OUTER_TURRET[team], np.float64)
        e = np.asarray(TOP_OUTER_TURRET[enemy], np.float64)
        d = e - o
        length = float(np.hypot(*d))
        axis = d / length
        normal = np.asarray([-axis[1], axis[0]])
        ref = np.asarray(NEXUS_POSITION[team], np.float64) - o
        if float(ref @ normal) > 0.0:          # own nexus must sit at n < 0
            normal = -normal
        ox.append(o[0]); oy.append(o[1])
        ax.append(axis[0]); ay.append(axis[1])
        nx.append(normal[0]); ny.append(normal[1])
        ln.append(length)
    f = lambda v: jnp.asarray(v, jnp.float32)
    return dict(origin_x=f(ox), origin_y=f(oy), axis_x=f(ax), axis_y=f(ay),
                normal_x=f(nx), normal_y=f(ny), length=f(ln))


LANE_AXIS = _lane_axis()


def lane_corridor_distance(x, y, corridor: float = LANE_HALF_WIDTH, axis=None):
    """Distance in game units from each champion to its lane corridor
    RECTANGLE; 0 anywhere inside. ``x``/``y`` are ``(2,)``, blue then red.

    Distance is to the **rectangle**, not to the axis: a champion at the right
    ``s`` but 3k units off-axis and one at ``n = 0`` but sitting in its own
    base are both far, and both get a gradient pointing at the nearest piece
    of lane. Also the trainer's ``lane_dist`` diagnostic, read directly
    rather than recovered from the potential by dividing out its weight
    (`REW-09`: that read 0 with shaping off and divided by zero at weight 0).
    """
    a = axis if axis is not None else LANE_AXIS
    px = x - a["origin_x"]
    py = y - a["origin_y"]
    s = px * a["axis_x"] + py * a["axis_y"]
    n = px * a["normal_x"] + py * a["normal_y"]
    off_n = jnp.maximum(0.0, jnp.abs(n) - corridor)
    off_s = jnp.maximum(
        0.0, jnp.maximum(-corridor - s, s - (a["length"] + corridor)))
    return jnp.hypot(off_n, off_s)


def lane_approach_potential(x, y, per_1000: float,
                            corridor: float = LANE_HALF_WIDTH, axis=None):
    r"""``-per_1000/1000 * lane_corridor_distance``; 0 once inside.

    From blue's spawn this is about **-0.56**, so the entire walk is worth
    roughly half of one last hit -- enough to point the way, far too little
    to be worth farming instead of minions.
    """
    return -(per_1000 / 1000.0) * lane_corridor_distance(x, y, corridor, axis)

#: 0.95 per 31 ticks (~517 ms) -- the server's float32 500 ms timer overshoots
#: by one tick, see `sim/rewards.py`. ~1.839 gold/s, not the nominal 1.9: the
#: nominal rate over-subtracted ~0.06 gold/s from every post-90 s decision.
AMBIENT_GOLD_PER_S = AMBIENT_GOLD_AMOUNT / (
    AMBIENT_GOLD_PERIOD_TICKS * (1000.0 / 60.0) / 1000.0)


class RewardWeights(NamedTuple):
    """`runs/rl-league-0915e/resolved_config.json`, rebalanced 2026-09-23.

    GOLD AND XP ARE THE PRIMARY TERMS; there is no flat per-last-hit term.

    The inherited weights were `money` 0.008, `exp` 0.001, `last_hit` 1.0.
    Priced against real minion values (melee 20g/77xp, caster 10g/51xp,
    cannon 30g/94xp) that made a last hit worth:

        term        melee   caster   cannon
        CS  x1.0    1.000    1.000    1.000
        gold x.008  0.160    0.080    0.240
        xp  x.001   0.077    0.051    0.094

    Two problems. The flat CS term was 6x the gold term AND identical across
    minion types, so the agent was explicitly taught that a 10-gold caster
    and a 30-gold cannon are worth the same. And it DOUBLE-COUNTED: a last
    hit yields +1 cs and +gold, so the proxy drowned the quantity it proxies.

    Rescaled so a melee last hit still totals ~1.0, split roughly 2:1
    gold:xp:

        gold 20x0.0335 + xp 77x0.0043 = 0.67 + 0.33 = 1.00  melee
                                        0.34 + 0.22 = 0.55  caster
                                        1.01 + 0.40 = 1.40  cannon

    a 2.5x spread that matches lane value. XP is the denser half: it accrues
    from PROXIMITY to a dying minion, not only from the killing blow, so it
    supplies the "be in lane while minions die" gradient without inventing a
    shaping term.

    REPRODUCIBILITY: runs before 2026-09-23 used the old weights; a reward
    curve is not comparable across this change.
    """

    money: float = 0.0335
    hp_point: float = 4.0
    death: float = -1.0
    exp: float = 0.0043
    #: the potential-based walk-to-lane shaping, per 1000 units of distance
    lane_approach: float = 0.07


class RewardConfig(NamedTuple):
    weights: RewardWeights = RewardWeights()


class RewardState(NamedTuple):
    """Per-champion previous values. ``(2,)`` arrays, blue then red."""

    gold: jax.Array
    xp: jax.Array
    hp_frac: jax.Array
    deaths: jax.Array
    primed: jax.Array          # the first step has no previous to difference
    phi: jax.Array             # Phi(s) of the shaping potential


def _phi(state: LaneState, cfg: RewardConfig):
    """``Phi(s)``, the lane-approach potential, ``(2,)``."""
    return lane_approach_potential(
        state.x[:2], state.y[:2], per_1000=cfg.weights.lane_approach)


def _snapshot(state: LaneState, hp, primed, phi) -> RewardState:
    return RewardState(
        gold=state.gold[:2], xp=state.xp[:2], hp_frac=hp,
        deaths=state.deaths[:2].astype(jnp.float32),
        primed=primed, phi=phi)


def _hp_frac(state: LaneState):
    return jnp.where(state.max_hp[:2] > 0, state.hp[:2] / state.max_hp[:2], 0.0)


def reward_init(state: LaneState,
                cfg: RewardConfig = RewardConfig()) -> RewardState:
    return _snapshot(state, _hp_frac(state), jnp.zeros((), bool),
                     _phi(state, cfg))


def lane_reward(state: LaneState, prev: RewardState, dt_s: float,
                cfg: RewardConfig = RewardConfig(), *, gamma: float = None,
                return_terms: bool = False):
    """One step of reward for both champions. Returns ``(reward(2,), new_prev)``.

    With ``return_terms`` it returns ``(reward, new_prev, terms)`` instead, where
    ``terms`` is the per-weight breakdown and sums to ``reward`` exactly.

    ``primed`` exists because the first call after a reset has no previous state
    to difference against. The source is explicit about the equivalent: a
    potential-based term would otherwise score the jump from the previous
    episode's final state to this one's initial state as a real transition.

    The shaping is the UNDISCOUNTED potential difference ``Phi(s') - Phi(s)``
    (`REW-11`, 2026-09-25). The textbook ``gamma*Phi(s') - Phi(s)`` is
    policy-invariant for the discounted objective, but it pays
    ``(1-gamma)*|Phi|`` on every step a champion spends standing still away
    from the lane: at 30 Hz with the 120 s horizon that is ~2.8 reward per
    600 s episode for sitting in the fountain, against ~1.0 for a melee last
    hit -- a denser reward than farming. The difference form pays exactly
    zero while stationary and only the endpoints of a walk; a there-and-back
    trip sums to zero exactly. ``gamma`` is accepted and unused so callers
    that thread the trainer's gamma keep working.
    """
    w = cfg.weights
    hp = _hp_frac(state)

    d_gold = state.gold[:2] - prev.gold
    # Only once the sim actually pays it: ambient gold starts at
    # `AMBIENT_GOLD_DELAY_MS` (90 s), and subtracting from t=0 put a
    # policy-independent -5.7 raw into every episode's first 90 s (`REW-07`).
    paying = state.t_ms >= AMBIENT_GOLD_DELAY_MS
    d_gold = d_gold - jnp.where(paying, AMBIENT_GOLD_PER_S * dt_s, 0.0)
    d_xp = state.xp[:2] - prev.xp
    d_hp = hp - prev.hp_frac                       # potential difference
    d_deaths = state.deaths[:2].astype(jnp.float32) - prev.deaths

    def zs(t):
        # unprimed -> nothing; then r_self - r_other (alpha = 1)
        t = jnp.where(prev.primed, t, jnp.zeros_like(t))
        return t - t[::-1]

    # Shaping is added AFTER the zero-sum combination, not inside it. It is
    # policy-invariant per agent; zero-summing it would make each agent's
    # shaping depend on the other's position, which is neither invariant nor
    # what the source does (``rewards[t] += shaping[t]``).
    phi = _phi(state, cfg)
    terms = {
        "money": zs(w.money * d_gold),
        "exp": zs(w.exp * d_xp),
        "hp_point": zs(w.hp_point * d_hp),
        "death": zs(w.death * d_deaths),
        "shaping": jnp.where(prev.primed, phi - prev.phi, 0.0),
    }
    # The sum of the per-term contributions IS the reward, so the breakdown
    # below adds up by construction (`test_reward_terms_sum_to_reward`).
    reward = (terms["money"] + terms["exp"] + terms["hp_point"]
              + terms["death"] + terms["shaping"])
    new_prev = _snapshot(state, hp, jnp.ones((), bool), phi)
    if return_terms:
        # PER-TERM CONTRIBUTIONS, reported after the zero-sum combination so
        # the sum is checkable rather than indicative. Only the TOTAL used to
        # be logged, so when cs@10min moved there was no way to say which term
        # moved it -- and the last reward change (`RL-002`) was a reweighting.
        return reward, new_prev, terms
    return reward, new_prev
