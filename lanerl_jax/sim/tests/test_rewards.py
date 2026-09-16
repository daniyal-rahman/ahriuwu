"""Gold, CS, XP and kill attribution -- the reward surface.

Gold is last-hit only and XP is proximity-shared, and that asymmetry is the
whole of laning. These are synthetic rather than full-game tests on purpose: a
full game confounds "the mechanic is wrong" with "the policy did not last-hit",
and the first time this was checked end-to-end the champion killed a minion,
correctly received XP, correctly received **no gold** (an allied minion landed
the final 0.6 damage), and that looked like a bug for several minutes.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

import jax.numpy as jnp

from lanerl_jax.data.patch import CONTENT_ROOT, load_patch  # noqa: E402
from lanerl_jax.sim.init import init_lane, lane_params  # noqa: E402
from lanerl_jax.sim.orders import OrderKind, Orders, apply_orders  # noqa: E402
from lanerl_jax.sim.rewards import (  # noqa: E402
    EXP_RADIUS,
    death_rewards,
    level_for_xp,
)
from lanerl_jax.sim.state import Kind, Team  # noqa: E402
from lanerl_jax.sim.step import step_decision  # noqa: E402

pytestmark = pytest.mark.skipif(
    not CONTENT_ROOT.exists(), reason="vendored Content tree not available"
)

C, M = Kind.CHAMPION, Kind.LANE_MINION
B, R = Team.BLUE, Team.RED


def _rw(died, killer, x, kinds, teams, alive=None, gold=20.5, xp=77.0):
    n = len(x)
    return death_rewards(
        died=jnp.asarray(died), killer=jnp.asarray(killer, jnp.int8),
        x=jnp.asarray(x, jnp.float32), y=jnp.zeros(n, jnp.float32),
        team=jnp.asarray(teams, jnp.int8), kind=jnp.asarray(kinds, jnp.int8),
        alive=jnp.asarray(alive if alive is not None else [True] * n),
        gold_on_death=jnp.full(n, gold, jnp.float32),
        xp_on_death=jnp.full(n, xp, jnp.float32))


def test_gold_goes_only_to_the_killer():
    r = _rw([False, False, True], [-1, -1, 0], [0.0, 100.0, 50.0],
            [C, C, M], [B, B, R])
    assert float(r.gold[0]) == pytest.approx(20.5)
    assert float(r.gold[1]) == 0.0
    assert int(r.cs[0]) == 1 and int(r.cs[1]) == 0


def test_xp_is_shared_by_proximity_to_the_corpse():
    """Both champions in range split it; the radius is measured from the corpse,
    not from the killer."""
    r = _rw([False, False, True], [-1, -1, 0], [0.0, 100.0, 50.0],
            [C, C, M], [B, B, R])
    assert float(r.xp[0]) == pytest.approx(38.5)
    assert float(r.xp[1]) == pytest.approx(38.5)


def test_a_champion_outside_exp_radius_gets_nothing():
    far = EXP_RADIUS + 100.0
    r = _rw([False, False, True], [-1, -1, 0], [0.0, far, 0.0],
            [C, C, M], [B, B, R])
    assert float(r.xp[0]) == pytest.approx(77.0)     # sole recipient
    assert float(r.xp[1]) == 0.0


def test_a_minion_killer_pays_no_gold():
    """`Champion.OnKill` is the only gold path, so a minion last-hit by another
    minion is worth nothing to anyone -- which is exactly what makes last-hitting
    a skill rather than a formality."""
    r = _rw([False, False, True], [-1, -1, 1], [0.0, 60.0, 50.0],
            [C, M, M], [B, B, R])
    assert float(r.gold.sum()) == 0.0
    assert int(r.cs.sum()) == 0
    assert float(r.xp[0]) > 0.0, "XP is still shared even when nobody earns gold"


def test_a_dead_champion_does_not_share_xp():
    r = _rw([False, False, True], [-1, -1, 0], [0.0, 100.0, 50.0],
            [C, C, M], [B, B, R], alive=[True, False, False])
    assert float(r.xp[0]) == pytest.approx(77.0)
    assert float(r.xp[1]) == 0.0


def test_level_thresholds_are_the_servers(): 
    patch = load_patch()
    curve = jnp.asarray([0.0] + [patch.xp_for_level(i) for i in range(2, 19)],
                        jnp.float32)
    got = [int(level_for_xp(jnp.asarray([v], jnp.float32), curve)[0])
           for v in (0.0, 279.0, 280.0, 660.0, 2400.0, 18360.0)]
    assert got == [1, 1, 2, 3, 6, 18]


def test_the_lowest_index_attacker_to_cross_zero_takes_the_kill():
    """`TakeDamage` records the killer only on the `!IsDead && hp <= 0`
    transition, and damage is applied in object order, so simultaneous hits do
    not race -- the first one to cross zero wins and later ones cannot reclaim it.

    Built as a real two-attacker tick rather than by calling `death_rewards`
    directly, because the attribution lives in `step.tick`'s cumulative sum.
    """
    patch = load_patch()
    params = lane_params(patch)
    # An empty arena. This test already learned once that a turret will happily
    # steal the kill it is trying to attribute (see the note below); with all 24
    # placed there is no open ground left to stand on.
    s = init_lane(patch, include_all_turrets=False)
    n = s.kind.shape[0]
    # put both champions on top of a nearly-dead red minion
    kind = np.asarray(s.kind).copy()
    team = np.asarray(s.team).copy()
    alive = np.asarray(s.alive).copy()
    x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy()
    hp = np.asarray(s.hp).copy()
    model = np.asarray(s.model).copy()
    from lanerl_jax.sim.profiles import profile_id
    from lanerl_jax.sim.targeting import MinionType
    mi = 2
    kind[mi] = M
    team[mi] = R
    alive[mi] = True
    model[mi] = profile_id(M, MinionType.MELEE, R)
    # Well away from either turret. The first version of this test stood them
    # 470 units from the blue top turret -- inside its 750 range -- and the
    # turret took the kill, so neither champion earned anything. The sim was
    # right and the fixture was wrong.
    x[0] = x[1] = x[mi] = 6000.0
    y[0] = y[1] = y[mi] = 6000.0
    team[0] = team[1] = B
    hp[mi] = 1.0                       # one hit from either champion kills it
    s = s.replace(kind=jnp.asarray(kind), team=jnp.asarray(team),
                  alive=jnp.asarray(alive), x=jnp.asarray(x), y=jnp.asarray(y),
                  hp=jnp.asarray(hp), model=jnp.asarray(model),
                  target=jnp.asarray(np.full(n, -1, np.int8)))
    orders = Orders(kind=jnp.asarray([OrderKind.ATTACK, OrderKind.ATTACK], jnp.int8),
                    x=jnp.zeros(2), y=jnp.zeros(2),
                    target=jnp.asarray([mi, mi], jnp.int8))
    s = apply_orders(s, orders)
    for _ in range(40):                # long enough for the wind-up to resolve
        s = step_decision(s, params)
        if not bool(s.alive[mi]):
            break
    assert not bool(s.alive[mi]), "the minion should have died"
    assert int(s.cs[0]) + int(s.cs[1]) == 1, "exactly one champion gets the CS"
    assert int(s.cs[0]) == 1, "ties go to the lower slot index, as in object order"
    assert float(s.gold[0]) > 0.0 and float(s.gold[1]) == 0.0


#: Measured from a 600 s idle server run (2026-09-16): the champion's gold is 0
#: until t=90 s, then rises 0.9502 per ~517 ms.
OBSERVED_AMBIENT = {91.03: 1.900, 95.03: 9.500}


def test_ambient_gold_matches_the_server():
    """Income only -- **not** a gold total.

    The server's gold *drops* over a long idle run, because
    `LanerlHooks.AutoBuyUndriven` buys items for an undriven champion. So a
    total-gold comparison is clean only until the first purchase; after that the
    rates still match and the levels differ by whatever was spent. The sim
    models income, not the shop, and that is a deliberate scope line: purchases
    are an action the policy should own, not something the environment does
    behind it.
    """
    from lanerl_jax.sim.rewards import (
        AMBIENT_GOLD_AMOUNT,
        AMBIENT_GOLD_DELAY_MS,
        AMBIENT_GOLD_INTERVAL_MS,
        ambient_gold,
    )

    tick = 1000.0 / 60.0
    t = 0.0
    timer = jnp.zeros(1)
    total = 0.0
    curve = {}
    while t < 100_000.0:
        g, timer = ambient_gold(jnp.float32(t), timer, jnp.ones(1, bool))
        total += float(g[0])
        t += tick
        curve[round(t / 1000.0, 2)] = total
    for t_s, want in OBSERVED_AMBIENT.items():
        got = curve[min(curve, key=lambda k: abs(k - t_s))]
        assert got == pytest.approx(want, abs=AMBIENT_GOLD_AMOUNT + 1e-6), \
            f"t={t_s}s sim {got} server {want}"
    # nothing before the delay
    assert curve[min(curve, key=lambda k: abs(k - 89.0))] == 0.0
    assert AMBIENT_GOLD_DELAY_MS == 90_000.0
    # granularity, not just rate: the reward signal sees the steps
    assert AMBIENT_GOLD_AMOUNT == 0.95 and AMBIENT_GOLD_INTERVAL_MS == 500.0


def test_ambient_gold_rate_is_the_documented_one():
    """0.95 per 500 ms is 1.9 gold/s -- `constants.AMBIENT_GOLD_PER_S`."""
    from lanerl_jax.sim.rewards import (
        AMBIENT_GOLD_AMOUNT,
        AMBIENT_GOLD_INTERVAL_MS,
    )

    rate = AMBIENT_GOLD_AMOUNT / (AMBIENT_GOLD_INTERVAL_MS / 1000.0)
    assert rate == pytest.approx(9.5 / 5.0)
