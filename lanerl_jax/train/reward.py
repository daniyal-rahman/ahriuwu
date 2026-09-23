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
required argument rather than a config default -- see :func:`lane_reward`.

Booked as NOT ported yet
------------------------
* potential-based last-hit shaping (``enable_shaping``)
* ``kill`` credit, which needs killer attribution routed out of the sim
* ``tower_hp``

Their weights are carried in the config and multiplied by zero, so turning one
on is a one-line change and nothing silently reads as "included".
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

import numpy as np

from ..sim.init import TOP_OUTER_TURRET
from ..sim.rewards import AMBIENT_GOLD_AMOUNT, AMBIENT_GOLD_INTERVAL_MS
from ..sim.state import LaneState, Team

__all__ = ["RewardWeights", "RewardConfig", "RewardState", "reward_init",
           "lane_reward", "AMBIENT_GOLD_PER_S", "LANE_HALF_WIDTH",
           "NEXUS_POSITION", "LANE_AXIS", "lane_approach_potential"]

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


def lane_approach_potential(x, y, per_1000: float,
                            corridor: float = LANE_HALF_WIDTH, axis=None):
    r"""``-per_1000/1000 * distance(champ, lane corridor)``; 0 once inside.

    ``x`` and ``y`` are ``(2,)``, blue then red, and the result is ``(2,)``.

    Distance is to the **rectangle**, not to the axis: a champion at the right
    ``s`` but 3k units off-axis and one at ``n = 0`` but sitting in its own
    base are both far, and both get a gradient pointing at the nearest piece of
    lane. From blue's spawn this is about **-0.56**, so the entire walk is
    worth roughly half of one last hit -- enough to point the way, far too
    little to be worth farming instead of minions.
    """
    a = axis if axis is not None else LANE_AXIS
    px = x - a["origin_x"]
    py = y - a["origin_y"]
    s = px * a["axis_x"] + py * a["axis_y"]
    n = px * a["normal_x"] + py * a["normal_y"]
    off_n = jnp.maximum(0.0, jnp.abs(n) - corridor)
    off_s = jnp.maximum(
        0.0, jnp.maximum(-corridor - s, s - (a["length"] + corridor)))
    return -(per_1000 / 1000.0) * jnp.hypot(off_n, off_s)

#: 0.95 per 500 ms, measured from the server -- see `sim/rewards.py`.
AMBIENT_GOLD_PER_S = AMBIENT_GOLD_AMOUNT / (AMBIENT_GOLD_INTERVAL_MS / 1000.0)


class RewardWeights(NamedTuple):
    """`runs/rl-league-0915e/resolved_config.json`. Unported terms are kept at
    their real weights but gated off, so nothing reads as included."""

    #: GOLD AND XP ARE THE PRIMARY TERMS, and `last_hit` is deliberately 0.
    #:
    #: The inherited weights were `money` 0.008, `exp` 0.001, `last_hit` 1.0.
    #: Priced against real minion values (melee 20g/77xp, caster 10g/51xp,
    #: cannon 30g/94xp) that made a last hit worth:
    #:
    #:     term        melee   caster   cannon
    #:     CS  x1.0    1.000    1.000    1.000
    #:     gold x.008  0.160    0.080    0.240
    #:     xp  x.001   0.077    0.051    0.094
    #:
    #: Two problems. The flat CS term is 6x the gold term AND identical across
    #: minion types, so the agent was explicitly taught that a 10-gold caster
    #: and a 30-gold cannon are worth the same -- erasing the distinction that
    #: makes the cannon the biggest CS on the board. And it DOUBLE-COUNTS: a
    #: last hit yields +1 cs and +gold, so the proxy drowned the quantity it
    #: proxies for.
    #:
    #: Rescaled so a melee last hit still totals ~1.0 -- the old scale, so no
    #: other weight needs re-tuning -- split roughly 2:1 gold:xp:
    #:
    #:     gold 20x0.0335 + xp 77x0.0043 = 0.67 + 0.33 = 1.00  melee
    #:                                     0.34 + 0.22 = 0.55  caster
    #:                                     1.01 + 0.40 = 1.40  cannon
    #:
    #: a 2.5x spread that matches lane value. XP is the denser half: it accrues
    #: from PROXIMITY to a dying minion, not only from the killing blow, so
    #: raising it supplies the "be in lane while minions die" gradient without
    #: inventing a shaping term.
    #:
    #: REPRODUCIBILITY: runs before 2026-09-23 used the old weights; a reward
    #: curve is not comparable across this change.
    money: float = 0.0335
    hp_point: float = 4.0
    death: float = -1.0
    exp: float = 0.0043
    #: 0.0 -- see `money`. Kept as a knob rather than deleted so the old
    #: behaviour is one assignment away and the change stays measurable.
    last_hit: float = 0.0
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
    #: potential-based LAST-HIT shaping; still unported
    enable_shaping: bool = False
    #: potential-based lane-approach shaping. ON, because with it off the
    #: reward is identically zero along the 13,532-unit walk to lane and the
    #: agent has no gradient to follow -- see the module docstring.
    enable_lane_approach: bool = True

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
    phi: jax.Array             # Phi(s) of the shaping potential


def _phi(state: LaneState, cfg: RewardConfig):
    """``Phi(s)`` -- the sum of the enabled potentials. Two potentials sum to
    one potential, so the invariance survives adding more of them later."""
    phi = jnp.zeros((2,), jnp.float32)
    if cfg.enable_lane_approach:
        phi = phi + lane_approach_potential(
            state.x[:2], state.y[:2], per_1000=cfg.weights.lane_approach)
    return phi


def reward_init(state: LaneState,
                cfg: RewardConfig = RewardConfig()) -> RewardState:
    hp = jnp.where(state.max_hp[:2] > 0, state.hp[:2] / state.max_hp[:2], 0.0)
    return RewardState(
        gold=state.gold[:2], xp=state.xp[:2], hp_frac=hp,
        deaths=state.deaths[:2].astype(jnp.float32),
        cs=state.cs[:2].astype(jnp.float32),
        primed=jnp.zeros((), bool), phi=_phi(state, cfg))


def lane_reward(state: LaneState, prev: RewardState, dt_s: float,
                cfg: RewardConfig = RewardConfig(), train_step=0,
                gamma: float | None = None):
    """One step of reward for both champions. Returns ``(reward(2,), new_prev)``.

    ``primed`` exists because the first call after a reset has no previous state
    to difference against. The source is explicit about the equivalent: a
    potential-based term would otherwise score the jump from the previous
    episode's final state to this one's initial state as a real transition.

    ``gamma`` is **required** whenever a potential is enabled, and it must be
    the same gamma the advantage estimator uses. ``F = gamma*Phi(s') - Phi(s)``
    is policy-invariant only under the discount it was built for; pass the
    wrong one and the shaping silently stops being invariant while still
    looking like it works. There is no default for exactly that reason -- a
    default would be a number that is right by luck.
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

    # Shaping is added AFTER the zero-sum combination, not inside it. It is
    # policy-invariant per agent; zero-summing it would make each agent's
    # shaping depend on the other's position, which is neither invariant nor
    # what the source does (``rewards[t] += shaping[t]``).
    phi = _phi(state, cfg)
    if cfg.enable_lane_approach or cfg.enable_shaping:
        if gamma is None:
            raise ValueError(
                "lane_reward needs the trainer's gamma when a potential is "
                "enabled: F = gamma*Phi(s') - Phi(s) is policy-invariant only "
                "under the discount it was built for. Pass cfg.ppo.gamma.")
        shaping = gamma * phi - prev.phi
        reward = reward + jnp.where(prev.primed, shaping, jnp.zeros_like(shaping))

    return reward, RewardState(
        gold=state.gold[:2], xp=state.xp[:2], hp_frac=hp,
        deaths=state.deaths[:2].astype(jnp.float32),
        cs=state.cs[:2].astype(jnp.float32),
        primed=jnp.ones((), bool), phi=phi)
