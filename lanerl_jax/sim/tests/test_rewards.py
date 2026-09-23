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

# `apply_orders` requires params (`STRUCT-003`); the level-one
# placeholder AD it used to fall back to is gone.
_PARAMS = lane_params() if CONTENT_ROOT.exists() else None

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
    curve = jnp.asarray([0.0, 0.0] + [patch.xp_for_level(i) for i in range(2, 19)],
                        jnp.float32)            # level-indexed, STRUCT-005
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
    # This fixture injects a fresh pre-collision snapshot.  The server's
    # CollisionHandler uses a separately stored quadtree position from the
    # preceding rebuild; for a constructed snapshot that cache must describe
    # the same injected world (and the newly activated minion must have a
    # node), rather than the unrelated lane-start layout from `init_lane`.
    collision_present = alive & (kind != Kind.NONE)
    s = s.replace(kind=jnp.asarray(kind), team=jnp.asarray(team),
                  alive=jnp.asarray(alive), x=jnp.asarray(x), y=jnp.asarray(y),
                  collision_x=jnp.asarray(x), collision_y=jnp.asarray(y),
                  collision_present=jnp.asarray(collision_present),
                  hp=jnp.asarray(hp), model=jnp.asarray(model),
                  target=jnp.asarray(np.full(n, -1, np.int8)))
    orders = Orders(kind=jnp.asarray([OrderKind.ATTACK, OrderKind.ATTACK], jnp.int8),
                    x=jnp.zeros(2), y=jnp.zeros(2),
                    target=jnp.asarray([mi, mi], jnp.int8))
    s = apply_orders(s, orders, _PARAMS)
    # Enter the tick just before two already-started ordinary swings finish.
    # The test is about `tick`'s ordered cumulative attribution, not the
    # separate wall-clock time to begin an autoattack; priming this legal
    # in-windup state keeps the fixture fast despite collision's full update.
    s = s.replace(
        is_attacking=s.is_attacking.at[:2].set(True),
        aa_windup=s.aa_windup.at[:2].set(0.001),
    )
    s = step_decision(s, params, step_ticks=1)
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


# =====================================================================
# Champion-kill gold/XP -- `Champion.Die`, `Champion.cs:392-461`
# =====================================================================

T = Kind.TURRET


def _kill_exp_table():
    from lanerl_jax.sim.profiles import build_profile_tables
    return build_profile_tables(load_patch())["champion_kill_exp"]


def _ckr(n, died, kind, level, killer, hit_flag_ms=None, hit_flag_by=None,
        kill_spree=None, death_spree=None, gold_from_minions=None,
        first_blood_done=True):
    from lanerl_jax.sim.rewards import champion_kill_rewards
    z = lambda: jnp.zeros(n)  # noqa: E731
    return champion_kill_rewards(
        died=jnp.asarray(died), kind=jnp.asarray(kind, jnp.int8),
        level=jnp.asarray(level, jnp.int8), killer=jnp.asarray(killer, jnp.int8),
        hit_flag_ms=jnp.asarray(hit_flag_ms) if hit_flag_ms is not None else z(),
        hit_flag_by=(jnp.asarray(hit_flag_by, jnp.int8) if hit_flag_by is not None
                    else jnp.full(n, -1, jnp.int8)),
        kill_spree=(jnp.asarray(kill_spree, jnp.int32) if kill_spree is not None
                   else jnp.zeros(n, jnp.int32)),
        death_spree=(jnp.asarray(death_spree, jnp.int32) if death_spree is not None
                    else jnp.zeros(n, jnp.int32)),
        gold_from_minions=(jnp.asarray(gold_from_minions) if gold_from_minions
                          is not None else z()),
        first_blood_done=jnp.asarray(first_blood_done),
        kill_exp_table=_kill_exp_table())


def test_a_plain_champion_kill_pays_base_gold_and_updates_both_sprees():
    r = _ckr(2, [True, False], [C, C], [3, 3], [1, -1])
    from lanerl_jax.sim.rewards import CHAMPION_BASE_GOLD
    assert float(r.gold[1]) == pytest.approx(CHAMPION_BASE_GOLD)
    assert float(r.gold[0]) == 0.0
    assert int(r.kills[1]) == 1
    assert int(r.kill_spree[1]) == 1, "killer's own streak, +1"
    assert int(r.death_spree[0]) == 1, "victim's own death spree, +1"
    assert int(r.kill_spree[0]) == 0


def test_shutdown_bonus_is_scaled_by_the_VICTIMS_own_kill_spree_not_the_killers():
    """`Champion.cs:421-425`: bare `KillSpree`/`DeathSpree` inside `Die()` mean
    `this.KillSpree`/`this.DeathSpree` -- Die() runs on the VICTIM. Killing a
    fed enemy (their own streak) is a shutdown bonus, not a self-referential
    'my own streak pays more' loop."""
    r = _ckr(2, [True, False], [C, C], [5, 5], [1, -1], kill_spree=[3, 0])
    expect = min(300.0 * (7.0 / 6.0) ** (3 - 1), 500.0)
    assert float(r.gold[1]) == pytest.approx(expect)


def test_kill_spree_of_exactly_one_is_flat_base_gold_regardless_of_death_spree():
    """`KillSpree>1` is false and `KillSpree==0` is false: `KillSpree==1`
    hits neither branch, literally, even with a large DeathSpree."""
    r = _ckr(2, [True, False], [C, C], [5, 5], [1, -1],
            kill_spree=[1, 0], death_spree=[7, 0])
    assert float(r.gold[1]) == pytest.approx(300.0)


def test_feeding_discount_and_the_double_deathspree_increment_bug():
    """`Champion.cs:426-436` then `:496`: the feeding branch bumps the
    VICTIM's own DeathSpree once inside it (using the OLD value for the
    `Math.Pow` above it) and once more, unconditionally, at the very end --
    a real double-increment in the server, reproduced bug-for-bug."""
    r = _ckr(2, [True, False], [C, C], [5, 5], [1, -1], death_spree=[3, 0])
    expect = max(300.0 * (11.0 / 12.0) * (0.8 ** (3 // 2)), 50.0)
    assert float(r.gold[1]) == pytest.approx(expect)
    assert int(r.death_spree[0]) == 5, "3 -> +1 (branch) -> +1 (unconditional)"


def test_first_blood_is_a_flat_bonus_awarded_exactly_once():
    r = _ckr(2, [True, False], [C, C], [1, 1], [1, -1], first_blood_done=False)
    assert float(r.gold[1]) == pytest.approx(300.0 + 100.0)
    assert bool(r.first_blood_done) is True


def test_exp_penalty_for_killing_an_underlevelled_enemy_capped_at_15_percent():
    """`Champion.cs:444-454`: `EXPDiff` is negated when the KILLER is the
    higher-level side -- killing down is worth less."""
    table = _kill_exp_table()
    r = _ckr(2, [True, False], [C, C], [5, 10], [1, -1])
    base = float(table[5])
    mult = min(0.08 * 5, 0.15)
    assert float(r.xp[1]) == pytest.approx(base - base * mult)


def test_exp_bonus_for_killing_an_overlevelled_enemy():
    table = _kill_exp_table()
    r = _ckr(2, [True, False], [C, C], [10, 5], [1, -1])
    base = float(table[10])
    mult = min(0.08 * 5, 0.15)
    assert float(r.xp[1]) == pytest.approx(base + base * mult)


def test_hit_flag_fallback_credits_the_champion_who_recently_hit_the_victim():
    """`Champion.cs:404-408`: killed by a minion, but still credited to the
    enemy champion who hit the victim within the last 15 s -- the server's
    own comment: "Killed by turret, minion or monster, but still give gold
    to the enemy." """
    r = _ckr(3, [True, False, False], [C, C, M], [3, 3, 1], [2, -1, -1],
            hit_flag_ms=[5000.0, 0.0, 0.0], hit_flag_by=[1, -1, -1])
    assert float(r.gold[1]) == pytest.approx(300.0), "champion 1 gets the gold"
    assert float(r.gold[2]) == 0.0, "the minion that landed the kill gets nothing"
    assert int(r.kills[1]) == 1


def test_hit_flag_fallback_requires_the_last_hit_to_have_been_from_a_champion():
    """`_playerHitId` is overwritten by EVERY hit, champion or not -- if the
    last recorded hit was itself from the minion, the cast to `Champion`
    fails and nobody is credited."""
    r = _ckr(3, [True, False, False], [C, C, M], [3, 3, 1], [2, -1, -1],
            hit_flag_ms=[5000.0, 0.0, 0.0], hit_flag_by=[2, -1, -1])
    assert float(r.gold.sum()) == 0.0


def test_no_cKiller_pays_nobody_and_freezes_the_victims_own_sprees():
    """`Champion.cs:411-416`: `cKiller == null` returns before touching
    `KillSpree`/`DeathSpree` at all."""
    r = _ckr(3, [True, False, False], [C, C, M], [3, 3, 1], [2, -1, -1],
            hit_flag_ms=[0.0, 0.0, 0.0], hit_flag_by=[1, -1, -1],
            death_spree=[2, 0, 0])
    assert float(r.gold.sum()) == 0.0
    assert float(r.xp.sum()) == 0.0
    assert int(r.death_spree[0]) == 2, "frozen, not incremented"
    assert int(r.kill_spree[0]) == 0


# =====================================================================
# Turret-destruction gold/XP -- `LaneTurret.Die`, `LaneTurret.cs:37-88`
# =====================================================================


def test_turret_local_gold_splits_among_every_champion_in_range_not_just_enemies():
    """`LaneTurret.cs:44`: `GetChampionsInRange` has no team filter -- an
    ally standing in range dilutes the denominator even though only the
    enemy receives any gold at all."""
    from lanerl_jax.sim.rewards import turret_kill_rewards
    gold, xp = turret_kill_rewards(
        died=jnp.asarray([True, False, False]),
        kind=jnp.asarray([T, C, C], jnp.int8),
        team=jnp.asarray([B, R, B], jnp.int8),
        alive=jnp.asarray([True, True, True]),
        x=jnp.asarray([0.0, 100.0, 100.0]), y=jnp.zeros(3),
        local_gold=jnp.asarray([150.0, 0.0, 0.0]),
        global_gold=jnp.asarray([100.0, 0.0, 0.0]),
        global_xp=jnp.asarray([0.0, 0.0, 0.0]),
        attack_range=jnp.asarray([750.0, 0.0, 0.0]))
    assert float(gold[1]) == pytest.approx(150.0 / 2 + 100.0), \
        "diluted by the ally standing in range"
    assert float(gold[2]) == 0.0, "the ally itself gets nothing"


def test_turret_out_of_range_enemy_gets_only_global_gold_but_full_xp():
    from lanerl_jax.sim.rewards import turret_kill_rewards
    gold, xp = turret_kill_rewards(
        died=jnp.asarray([True, False]), kind=jnp.asarray([T, C], jnp.int8),
        team=jnp.asarray([B, R], jnp.int8), alive=jnp.asarray([True, True]),
        x=jnp.asarray([0.0, 5000.0]), y=jnp.zeros(2),
        local_gold=jnp.asarray([150.0, 0.0]),
        global_gold=jnp.asarray([100.0, 0.0]),
        global_xp=jnp.asarray([50.0, 0.0]),
        attack_range=jnp.asarray([750.0, 0.0]))
    assert float(gold[1]) == pytest.approx(100.0)
    assert float(xp[1]) == pytest.approx(50.0), "XP is unconditional on range"


def test_inhibitor_and_nexus_tier_turrets_never_have_a_local_share():
    """`LocalGoldGivenOnDeath` is 0 in Content for these tiers -- the
    `localGold>0` gate is false regardless of who's in range, so every enemy
    gets flat global gold even standing on top of the turret."""
    from lanerl_jax.sim.rewards import turret_kill_rewards
    gold, xp = turret_kill_rewards(
        died=jnp.asarray([True, False]), kind=jnp.asarray([T, C], jnp.int8),
        team=jnp.asarray([B, R], jnp.int8), alive=jnp.asarray([True, True]),
        x=jnp.asarray([0.0, 10.0]), y=jnp.zeros(2),
        local_gold=jnp.asarray([0.0, 0.0]),
        global_gold=jnp.asarray([175.0, 0.0]),
        global_xp=jnp.asarray([100.0, 0.0]),
        attack_range=jnp.asarray([750.0, 0.0]))
    assert float(gold[1]) == pytest.approx(175.0)
    assert float(xp[1]) == pytest.approx(100.0)


def test_an_allied_turret_dying_pays_the_ally_nothing():
    from lanerl_jax.sim.rewards import turret_kill_rewards
    gold, xp = turret_kill_rewards(
        died=jnp.asarray([True, False]), kind=jnp.asarray([T, C], jnp.int8),
        team=jnp.asarray([B, B], jnp.int8), alive=jnp.asarray([True, True]),
        x=jnp.asarray([0.0, 10.0]), y=jnp.zeros(2),
        local_gold=jnp.asarray([150.0, 0.0]),
        global_gold=jnp.asarray([100.0, 0.0]),
        global_xp=jnp.asarray([100.0, 0.0]),
        attack_range=jnp.asarray([750.0, 0.0]))
    assert float(gold[1]) == 0.0 and float(xp[1]) == 0.0


# =====================================================================
# `_championHitFlagTimer`/`_playerHitId` and the minion-gold death-spree decay
# =====================================================================


def test_hit_flag_resets_to_the_full_timer_on_any_hit():
    from lanerl_jax.sim.rewards import HIT_FLAG_MS, update_hit_flag
    n = 2
    damage_ij = jnp.zeros((n, n)).at[1, 0].set(30.0)   # unit 1 hits unit 0
    ms, by = update_hit_flag(
        kind=jnp.asarray([C, M], jnp.int8), damage_ij=damage_ij,
        buff_damage=jnp.zeros(n), buff_dealt_by=jnp.full(n, -1, jnp.int8),
        hit_flag_ms=jnp.asarray([1000.0, 0.0]),
        hit_flag_by=jnp.asarray([-1, -1], jnp.int8), delta_ms=1000.0 / 60.0)
    assert float(ms[0]) == HIT_FLAG_MS
    assert int(by[0]) == 1
    assert float(ms[1]) == pytest.approx(0.0), "never hit; already at the floor"


def test_hit_flag_decays_and_keeps_its_last_attacker_when_not_hit():
    from lanerl_jax.sim.rewards import update_hit_flag
    ms, by = update_hit_flag(
        kind=jnp.asarray([C], jnp.int8), damage_ij=jnp.zeros((1, 1)),
        buff_damage=jnp.zeros(1), buff_dealt_by=jnp.full(1, -1, jnp.int8),
        hit_flag_ms=jnp.asarray([100.0]), hit_flag_by=jnp.asarray([3], jnp.int8),
        delta_ms=1000.0 / 60.0)
    assert float(ms[0]) == pytest.approx(100.0 - 1000.0 / 60.0)
    assert int(by[0]) == 3


def test_minion_gold_only_accumulates_into_deathspree_decay_while_on_a_spree():
    from lanerl_jax.sim.rewards import minion_gold_deathspree_decay
    gfm, ds = minion_gold_deathspree_decay(
        minion_gold=jnp.asarray([20.0, 20.0]),
        death_spree=jnp.asarray([0, 2], jnp.int32),
        gold_from_minions=jnp.asarray([500.0, 500.0]))
    assert float(gfm[0]) == pytest.approx(500.0), "DeathSpree==0 -- not accumulated"
    assert float(gfm[1]) == pytest.approx(520.0)
    assert int(ds[0]) == 0 and int(ds[1]) == 2


def test_crossing_1000_gold_from_minions_knocks_off_exactly_one_deathspree_stack():
    """`Champion.cs:384-388`: an `if`, not a `while` -- gold that would cross
    1000 twice in one tick still only pays off one stack."""
    from lanerl_jax.sim.rewards import minion_gold_deathspree_decay
    gfm, ds = minion_gold_deathspree_decay(
        minion_gold=jnp.asarray([2500.0]),
        death_spree=jnp.asarray([3], jnp.int32),
        gold_from_minions=jnp.asarray([0.0]))
    assert float(gfm[0]) == pytest.approx(1500.0)
    assert int(ds[0]) == 2


# =====================================================================
# End-to-end: is any of this actually wired into `step.tick`?
# =====================================================================


def test_a_champion_kill_pays_gold_and_xp_through_a_real_tick():
    """Confirms `champion_kill_rewards` is reachable from `step.tick`, not
    merely correct in isolation -- the same style as
    `test_the_lowest_index_attacker_to_cross_zero_takes_the_kill` above."""
    patch = load_patch()
    params = lane_params(patch)
    s = init_lane(patch, include_all_turrets=False)
    n = s.kind.shape[0]
    x = np.asarray(s.x).copy()
    y = np.asarray(s.y).copy()
    team = np.asarray(s.team).copy()
    hp = np.asarray(s.hp).copy()
    x[0] = x[1] = 6000.0
    y[0] = y[1] = 6000.0
    team[0], team[1] = B, R
    hp[1] = 1.0                        # one hit from champion 0 kills champion 1
    s = s.replace(x=jnp.asarray(x), y=jnp.asarray(y), team=jnp.asarray(team),
                  hp=jnp.asarray(hp), target=jnp.asarray(np.full(n, -1, np.int8)))
    orders = Orders(kind=jnp.asarray([OrderKind.ATTACK, OrderKind.NOOP], jnp.int8),
                    x=jnp.zeros(2), y=jnp.zeros(2),
                    target=jnp.asarray([1, -1], jnp.int8))
    s = apply_orders(s, orders, _PARAMS)
    gold_before = float(s.gold[0])
    for _ in range(40):
        s = step_decision(s, params)
        if not bool(s.alive[1]):
            break
    assert not bool(s.alive[1]), "champion 1 should have died"
    assert float(s.gold[0]) > gold_before
    assert int(s.kills[0]) == 1
    assert int(s.kill_spree[0]) == 1
    assert int(s.death_spree[1]) == 1
