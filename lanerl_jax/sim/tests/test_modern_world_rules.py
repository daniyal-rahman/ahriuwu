"""Whole-world rule symptoms after the MODERN-020 integration (``modern_step`` with every system).

Each test drives the real compiled tick and checks something a player would
notice: waves walk down all three lanes, camps appear on the clock, a ward in
brush shows the enemy hiding there, killing a camp pays its owner, attack-move
picks up an enemy on the way, minion trades are symmetric between the teams,
and a destroyed Nexus ends (freezes) the game. The world (Jax runs Overgrowth)
and its one compiled tick program come from ``modern_world_harness``, shared
with the other full-tick modules (several minutes to compile on CPU).
"""
from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.sim import modern_world_types as W
from lanerl_jax.sim.tests import modern_world_harness as H

if not H.artifacts_present():
    pytest.skip("modern map/route artifacts not present", allow_module_level=True)

from lanerl_jax.sim import modern_jungle as J  # noqa: E402
from lanerl_jax.sim import modern_step as MS  # noqa: E402

assert 8451 in H.JAX_PAGE.secondary          # Overgrowth (test_overgrowth_counts_...)
BRUSH = (2274.0, 13558.0)             # lane brush beside the top-lane midpoint (test_modern_vision)
BRUSH_EDGE = (2469.0, 13357.0)        # just outside it
CHUNK = 30                            # ticks per second
orders = H.orders


def world():
    return H.world(), H.step, H.run, H.refresh


@lru_cache(maxsize=1)
def state_at_95s():
    cfg, _, run, _ = world()
    s, (po, mo) = run(MS.init_state(cfg), MS.no_orders(), 95 * CHUNK)
    return s, max(int(po), int(mo))


def test_waves_in_every_lane_and_camps_on_the_clock():
    s, over = state_at_95s()
    assert over == 0
    k, a, lane, team = (np.asarray(v) for v in (s.kind, s.alive, s.lane_ai.lane, s.team))
    minion = (k == W.KIND_MINION) & a
    for l in range(3):
        for t in (0, 1):
            assert (minion & (lane == l) & (team == t)).sum() > 0, (l, t)
    mon = (k == W.KIND_MONSTER) & a
    subs = np.asarray(s.sub)[mon]
    assert mon.sum() == 28                                    # 14 camp monsters per side at 1:30
    assert not (subs == J.Monster.SCUTTLE).any()              # Scuttle Crabs come later (3:30)
    assert (np.asarray(s.team)[mon] == W.NEUTRAL).all()


def test_minion_trades_are_symmetric_between_teams():
    cfg, _, run, _ = world()
    s, _ = state_at_95s()
    s, _ = run(s, MS.no_orders(), 120 * CHUNK)               # to 3:35, idle champions in the fountains
    k, a, team = np.asarray(s.kind), np.asarray(s.alive), np.asarray(s.team)
    blue, red = ((k == W.KIND_MINION) & a & (team == t) for t in (0, 1))
    assert abs(int(blue.sum()) - int(red.sum())) <= 6
    gold = np.asarray(s.econ.gold)
    assert abs(gold[0] - gold[1]) < 1e-3                      # nobody farmed: identical passive gold


def test_ward_in_brush_reveals_the_enemy_hiding_there():
    cfg, step, _, refresh = world()
    s = MS.init_state(cfg)
    s = refresh(s._replace(x=s.x.at[0].set(BRUSH_EDGE[0]).at[1].set(BRUSH[0]),
                           y=s.y.at[0].set(BRUSH_EDGE[1]).at[1].set(BRUSH[1]), t=jnp.float32(60.0)))
    assert not bool(s.visible[0, 1])
    s, _ = step(s, orders(ward_kind=[0, -1], ward_x=[BRUSH[0] + 40.0, 0.0], ward_y=[BRUSH[1], 0.0]))
    wards = (np.asarray(s.kind) == W.KIND_WARD) & np.asarray(s.alive)
    assert wards.sum() == 1 and np.asarray(s.team)[wards][0] == 0
    assert bool(s.visible[0, 1])                              # the ward sees into its brush
    ward = int(np.flatnonzero(wards)[0])
    assert bool(s.visible[1, ward])                           # a fresh Totem Ward is visible for 2 s
    for _ in range(70):
        s, _ = step(s, MS.no_orders())
    assert bool(s.visible[0, 1]) and not bool(s.visible[1, ward])   # then stealthed; still spotting Jax


def test_killing_a_camp_pays_the_killer():
    cfg, step, _, refresh = world()
    s, _ = state_at_95s()
    k, sub, a = np.asarray(s.kind), np.asarray(s.sub), np.asarray(s.alive)
    gromp = int(np.flatnonzero((k == W.KIND_MONSTER) & a & (sub == J.Monster.GROMP))[0])
    gx, gy = float(s.x[gromp]), float(s.y[gromp])
    s = refresh(s._replace(x=s.x.at[0].set(gx + 150.0), y=s.y.at[0].set(gy), hp=s.hp.at[gromp].set(5.0)))
    g0, xp0 = float(s.econ.gold[0]), float(s.econ.xp[0])
    for _ in range(45):
        s, _ = step(s, orders(attack=[gromp, -1]))
        if not bool(s.alive[gromp]):
            break
    assert not bool(s.alive[gromp])
    assert float(s.econ.xp[0]) > xp0 + 50.0
    assert float(s.econ.gold[0]) > g0 + 50.0


def test_attack_move_acquires_an_enemy_on_the_way():
    cfg, step, _, refresh = world()
    s = MS.init_state(cfg)
    lane = np.asarray(cfg.lane_path)
    mx, my = lane[len(lane) // 2]
    s = refresh(s._replace(x=s.x.at[0].set(mx).at[1].set(mx + 300.0), y=s.y.at[0].set(my).at[1].set(my),
                           t=jnp.float32(30.0)))
    hp0 = float(s.hp[1])
    s, _ = step(s, orders(attack_move=[True, False], move_x=[mx + 2000.0, 0.0], move_y=[my, 0.0]))
    for _ in range(60):
        s, _ = step(s, MS.no_orders())
    assert int(s.amove.held[0]) == 1
    assert float(s.hp[1]) < hp0


def test_destroyed_nexus_ends_and_freezes_the_game():
    cfg, step, _, _ = world()
    s = MS.init_state(cfg)
    nexus = int(np.flatnonzero((np.asarray(cfg.unit_kind) == W.KIND_NEXUS) & (np.asarray(cfg.unit_team) == 1))[0])
    tur = s.towers.turret
    s = s._replace(towers=s.towers._replace(turret=tur._replace(hp=tur.hp.at[nexus].set(0.0))))
    s, _ = step(s, MS.no_orders())
    assert bool(s.game_over) and int(s.winner) == 0
    t = float(s.t)
    s, _ = step(s, MS.no_orders())
    assert float(s.t) == t


def test_overgrowth_counts_a_minion_death_the_holder_saw():
    cfg, step, _, refresh = world()
    s, _ = state_at_95s()
    k, a, team = np.asarray(s.kind), np.asarray(s.alive), np.asarray(s.team)
    m = int(np.flatnonzero((k == W.KIND_MINION) & a & (team == 0))[0])
    mx, my = float(s.x[m]), float(s.y[m])
    s = refresh(s._replace(x=s.x.at[1].set(mx + 120.0), y=s.y.at[1].set(my), hp=s.hp.at[m].set(1.0)))
    og0 = int(s.combat.runes.resolve.og_count[1])
    for _ in range(45):
        s, _ = step(s, orders(attack=[-1, m]))
        if not bool(s.alive[m]):
            break
    assert not bool(s.alive[m])
    s, _ = step(s, MS.no_orders())                                    # counted the tick after (latched sight)
    assert int(s.combat.runes.resolve.og_count[1]) >= og0 + 1


def test_stridebreaker_cast_walks_but_cannot_attack_and_its_slow_lands_in_cc():
    from lanerl_jax.sim import modern_inventory as I
    cfg, step, _, refresh = world()
    s = MS.init_state(cfg)
    lane = np.asarray(cfg.lane_path)
    mx, my = lane[len(lane) // 2]
    inv = I.inventory_from_ids([[1055, 2003, 6631], [1055, 2003]])
    s = refresh(s._replace(x=s.x.at[0].set(mx).at[1].set(mx + 150.0), y=s.y.at[0].set(my).at[1].set(my + 150.0),
                           t=jnp.float32(30.0), champ=s.champ._replace(inventory=inv)))
    s, _ = step(s, orders(item_active=[6631, 0]))
    t1 = float(s.t)
    assert float(s.champ.item_cast_until[0]) > t1 >= float(s.champ.cast_lock_until[0])   # attack/cast lock only
    for _ in range(12):
        s, _ = step(s, MS.no_orders())
    assert float(s.cc.slow[1]) > 0.3 and float(s.cc.slow_until[1]) > float(s.t)          # 35% item slow


def test_slowed_minions_walk_slower():
    cfg, step, _, _ = world()
    s, _ = state_at_95s()
    s1, _ = step(s, MS.no_orders())
    k, a = np.asarray(s1.kind), np.asarray(s1.alive)
    moved = np.hypot(np.asarray(s1.x) - np.asarray(s.x), np.asarray(s1.y) - np.asarray(s.y))
    m = int(np.flatnonzero((k == W.KIND_MINION) & a & (moved > 5.0))[0])
    slowed = s1._replace(cc=s1.cc._replace(slow=s1.cc.slow.at[m].set(0.5), slow_until=s1.cc.slow_until.at[m].set(1e4)))
    free, slow = s1, slowed
    for _ in range(3):
        free, _ = step(free, MS.no_orders())
        slow, _ = step(slow, MS.no_orders())
    d = lambda t: float(np.hypot(float(t.x[m]) - float(s1.x[m]), float(t.y[m]) - float(s1.y[m])))  # noqa: E731
    assert d(slow) < 0.7 * d(free)


def test_both_champions_walk_from_base_to_the_top_lane():
    """Routes from both fountains reach the lane (red used to stay pinned at its top inhibitor, whose
    collision circle reached past the navgrid pad the routes are baked around)."""
    cfg, step, run, _ = world()
    lane = np.asarray(cfg.lane_path)
    goal = lane[len(lane) // 2]
    o = orders(move=[True, True], move_x=[goal[0]] * 2, move_y=[goal[1]] * 2)
    s, _ = step(MS.init_state(cfg), o)
    s, _ = run(s, MS.no_orders(), 60 * CHUNK)
    d = np.hypot(np.asarray(s.x[:2]) - goal[0], np.asarray(s.y[:2]) - goal[1])
    assert (d < 150.0).all(), d


def test_a_dash_keeps_moving_after_its_start_tick():
    """Dash state is kept across ticks (it used to be dropped after the start tick)."""
    cfg, step, run, _ = world()
    s = MS.init_state(cfg)
    x0, y0 = float(s.x[0]), float(s.y[0])
    dash = W.Dash(jnp.asarray([True, False]), jnp.asarray([x0 + 400.0, 0.0]), jnp.asarray([y0 + 400.0, 0.0]),
                  jnp.asarray([800.0, 0.0]), jnp.asarray([-1, -1], jnp.int32), jnp.asarray([False, False]))
    s, _ = step(s._replace(pending_dash=dash), MS.no_orders())
    s1 = np.hypot(float(s.x[0]) - x0, float(s.y[0]) - y0)
    s, _ = run(s, MS.no_orders(), 10)
    s2 = np.hypot(float(s.x[0]) - x0, float(s.y[0]) - y0)
    assert s1 < 40.0 and s2 > s1 + 150.0, (s1, s2)          # 800 u/s for 10 more ticks: ~260 u
