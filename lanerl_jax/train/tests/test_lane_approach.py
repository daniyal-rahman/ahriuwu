"""The potential that makes the walk to lane learnable.

WHY THIS TERM EXISTS
--------------------
Blue spawns at (26, 280); the wave meets at (3907, 13243). That is 13,532
units, 39 seconds of walking, across which the reward is *identically zero*:
`money` has the ambient trickle subtracted out, and gold, xp, hp and cs cannot
move until the champion reaches minions. A uniform random policy covers about
sqrt(18000) * 11.5 ~ 1,500 units of net displacement in a whole 600 s episode.

So the agent never arrived, never saw a reward, and had no gradient. CS was
exactly 0.00000 across 400 updates and 26 million champion-decisions -- and it
read as "the policy has not learned yet" rather than as "the task is
impossible", which is why it cost a full training run to find.

WHY A POTENTIAL AND NOT A BONUS
-------------------------------
`F = gamma*Phi(s') - Phi(s)` is policy-invariant (Ng, Harada & Russell 1999):
it changes which policy is FOUND, never which policy is optimal. A per-step
"closer than last tick" bonus is NOT a potential, and it pays an agent to
oscillate toward and away from lane forever. These tests pin the difference.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.sim.init import CHAMPION_SPAWN, TOP_OUTER_TURRET, init_lane
from lanerl_jax.sim.state import Team
from lanerl_jax.train.reward import (
    LANE_AXIS,
    LANE_HALF_WIDTH,
    RewardConfig,
    RewardWeights,
    lane_approach_potential,
    lane_reward,
    reward_init,
)

W = RewardWeights().lane_approach


def phi(x, y):
    return lane_approach_potential(
        jnp.asarray(x, jnp.float32), jnp.asarray(y, jnp.float32), per_1000=W)


def test_the_potential_is_zero_in_lane_and_negative_in_the_fountain():
    """The gradient has to point somewhere, and it has to stop at the corridor."""
    spawn = phi([CHAMPION_SPAWN[Team.BLUE][0], CHAMPION_SPAWN[Team.RED][0]],
                [CHAMPION_SPAWN[Team.BLUE][1], CHAMPION_SPAWN[Team.RED][1]])
    meet = phi([3907.0, 3907.0], [13243.0, 13243.0])

    assert float(meet[0]) == 0.0 and float(meet[1]) == 0.0, (
        "the wave meeting point is inside the corridor, so the potential must "
        "saturate there -- otherwise it keeps paying to walk past the wave")
    assert float(spawn[0]) < 0.0 and float(spawn[1]) < 0.0
    # near-mirror: the spawns are not exactly reflections, so not identical
    assert abs(float(spawn[0]) - float(spawn[1])) < 0.02


#: What one melee minion is worth under the CURRENT weights. Priced from the
#: patch (melee 20 gold, 77 xp) rather than from `RewardWeights.last_hit`,
#: which is now 0.0: gold and xp are the primary terms and the flat per-kill
#: term was removed because it double-counted gold and paid the same for a
#: 10-gold caster as a 30-gold cannon. The INVARIANT this test defends is
#: unchanged -- shaping must guide, never compete with farming -- only the
#: reference moved, so it is computed here instead of read off a weight that
#: no longer carries it.
MELEE_GOLD, MELEE_XP = 20.0, 77.0


def _one_melee_last_hit(w: RewardWeights) -> float:
    return w.money * MELEE_GOLD + w.exp * MELEE_XP + w.last_hit


def test_the_whole_walk_is_worth_less_than_one_last_hit():
    """Scale is the safety property. Shaping must guide, never compete.

    A walk worth more than one minion would make standing in lane a better
    living than killing things in it.
    """
    w = RewardWeights()
    one_hit = _one_melee_last_hit(w)
    assert one_hit > 0.0, (
        "no term pays for a last hit at all -- gold, xp and last_hit are all "
        "zero, and the walk-versus-farm comparison below is then vacuous")
    spawn = phi([CHAMPION_SPAWN[Team.BLUE][0]], [CHAMPION_SPAWN[Team.BLUE][1]])
    walk = 0.0 - float(spawn[0])
    assert 0.3 < walk < 1.0, f"the walk pays {walk:.3f}"
    assert walk < one_hit, (
        f"the walk pays {walk:.3f} against {one_hit:.3f} for one melee minion")


def test_the_discounted_shaping_telescopes_to_the_endpoints():
    """THE invariance property, stated as an identity.

    sum_t gamma^t * (gamma*Phi(s_{t+1}) - Phi(s_t))  ==  gamma^T*Phi(s_T) - Phi(s_0)

    Every intermediate term cancels, so the shaping contributes a quantity that
    depends ONLY on where the trajectory starts and ends -- never on the route
    taken or how long it took. That is exactly why it cannot create a cycle the
    agent can farm, and it is the thing a "closer than last tick" bonus fails.
    """
    gamma = 0.999
    rng = np.random.default_rng(0)
    # a deliberately silly route: it wanders, doubles back, and loops
    xs = np.cumsum(rng.normal(0, 400, 200)) + 3000.0
    ys = np.cumsum(rng.normal(0, 400, 200)) + 8000.0

    phis = [float(phi([x], [y])[0]) for x, y in zip(xs, ys)]
    total = sum((gamma ** t) * (gamma * phis[t + 1] - phis[t])
                for t in range(len(phis) - 1))
    expect = (gamma ** (len(phis) - 1)) * phis[-1] - phis[0]
    assert total == pytest.approx(expect, abs=1e-6), (
        "the discounted shaping did not telescope, so it is not a potential "
        "and the policy-invariance guarantee is gone")


def test_a_round_trip_pays_almost_nothing():
    """The concrete form of invariance: oscillation is not profitable.

    Walk halfway to lane and come back. A naive 'closer than last tick' bonus
    pays the full approach and charges nothing for the retreat, so repeating
    the round trip farms unbounded reward. A potential charges it back.
    """
    gamma = 0.999
    out = [(26.0 + t * 200.0, 280.0 + t * 600.0) for t in range(12)]
    route = out + out[::-1]
    phis = [float(phi([x], [y])[0]) for x, y in route]
    total = sum((gamma ** t) * (gamma * phis[t + 1] - phis[t])
                for t in range(len(phis) - 1))
    assert abs(total) < 0.03, (
        f"a there-and-back round trip paid {total:.4f}; a potential must "
        "return to where it started")


def test_distance_is_to_the_rectangle_not_to_the_axis():
    """A champion on the lane axis but deep in its own base is NOT in lane.

    Projecting onto the axis alone would score it as perfectly positioned,
    which is how an agent learns to stand in the fountain facing the right way.
    """
    a = {k: np.asarray(v) for k, v in LANE_AXIS.items()}
    # a point at n = 0 but far behind the own turret (s very negative)
    s_behind = -6000.0
    x = float(a["origin_x"][0] + s_behind * a["axis_x"][0])
    y = float(a["origin_y"][0] + s_behind * a["axis_y"][0])
    behind = float(phi([x], [y])[0])
    assert behind < -0.3, (
        f"a point on the lane AXIS but {abs(s_behind):.0f} units behind the "
        f"turret scored {behind:.3f}; distance must be to the corridor "
        "rectangle, not to the infinite line")

    # and a point at the right s but far off-axis is equally not in lane
    s_mid = 2400.0
    off = 4000.0
    x = float(a["origin_x"][0] + s_mid * a["axis_x"][0] + off * a["normal_x"][0])
    y = float(a["origin_y"][0] + s_mid * a["axis_y"][0] + off * a["normal_y"][0])
    assert float(phi([x], [y])[0]) == pytest.approx(
        -(W / 1000.0) * (off - LANE_HALF_WIDTH), abs=1e-3)


def test_both_teams_share_one_world_normal():
    """The handedness rule, which is load-bearing and easy to get backwards.

    Red's lane axis is exactly antiparallel to blue's, so the plain left-hand
    normal comes out with opposite signs for the two agents. Forcing 'own
    nexus at n < 0' makes both adopt the SAME world normal, so `n > 0` means
    one physical direction for both -- and the frames map to each other by
    (s, n) -> (L - s, n), a reflection, which is the correct symmetry for two
    champions sharing one corridor.
    """
    a = {k: np.asarray(v) for k, v in LANE_AXIS.items()}
    assert a["axis_x"][0] == pytest.approx(-a["axis_x"][1], abs=1e-5)
    assert a["axis_y"][0] == pytest.approx(-a["axis_y"][1], abs=1e-5)
    # the normals agree in the world frame -- this is the whole point
    assert a["normal_x"][0] == pytest.approx(a["normal_x"][1], abs=1e-5)
    assert a["normal_y"][0] == pytest.approx(a["normal_y"][1], abs=1e-5)

    # ... and therefore a single world point has s -> L - s between frames
    px, py = 3000.0, 11000.0
    L = float(a["length"][0])
    s_blue = ((px - a["origin_x"][0]) * a["axis_x"][0]
              + (py - a["origin_y"][0]) * a["axis_y"][0])
    s_red = ((px - a["origin_x"][1]) * a["axis_x"][1]
             + (py - a["origin_y"][1]) * a["axis_y"][1])
    assert s_blue == pytest.approx(L - s_red, abs=1e-2)


def test_shaping_without_the_trainers_gamma_is_refused():
    """No default gamma, because a default would be right only by luck.

    F is policy-invariant under the discount it was built for and no other.
    A mismatched gamma leaves the term looking like it works while quietly
    no longer being invariant -- so this raises rather than guessing.
    """
    state = init_lane()
    cfg = RewardConfig()
    assert cfg.enable_lane_approach
    rs = reward_init(state, cfg)
    with pytest.raises(ValueError, match="gamma"):
        lane_reward(state, rs, dt_s=1 / 30.0, cfg=cfg)


def test_the_first_step_after_a_reset_pays_no_shaping():
    """Priming. Without it the reset itself is scored as a transition.

    On the tick after a reset the champion teleports from wherever the last
    episode ended back to the fountain. That is not a move the agent made, and
    paying for it would hand a large negative to whichever action happened to
    be sampled last.
    """
    state = init_lane()
    cfg = RewardConfig()
    rs = reward_init(state, cfg)
    assert not bool(rs.primed)
    r, rs2 = lane_reward(state, rs, dt_s=1 / 30.0, cfg=cfg, gamma=0.999)
    assert np.allclose(np.asarray(r), 0.0), (
        f"the unprimed first step paid {np.asarray(r)}")
    assert bool(rs2.primed)


def test_walking_towards_lane_pays_and_walking_away_charges():
    """The sign, end to end through lane_reward rather than the bare potential."""
    cfg = RewardConfig()
    state = init_lane()
    rs = reward_init(state, cfg)
    _, rs = lane_reward(state, rs, 1 / 30.0, cfg, gamma=0.999)   # prime

    ax = np.asarray(LANE_AXIS["axis_x"])[0], np.asarray(LANE_AXIS["axis_y"])[0]
    towards = state.replace(
        x=state.x.at[0].add(300.0 * ax[0]), y=state.y.at[0].add(300.0 * ax[1]))
    away = state.replace(
        x=state.x.at[0].add(-300.0 * ax[0]), y=state.y.at[0].add(-300.0 * ax[1]))

    r_to, _ = lane_reward(towards, rs, 1 / 30.0, cfg, gamma=0.999)
    r_away, _ = lane_reward(away, rs, 1 / 30.0, cfg, gamma=0.999)
    assert float(r_to[0]) > float(r_away[0])
    assert float(r_to[0]) > 0.0 > float(r_away[0])
