"""Whole-world rule symptoms through the real compiled tick (``world_harness``): waves in every lane, camps on the
clock, wards, camp rewards, attack-move, symmetric minion trades, Overgrowth, Stridebreaker, collision and routes,
walk-in casting, fog chasing, Minion Pushing and the Nexus game over."""
from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.core import types as W
from lanerl_jax.modern.tests import world_harness as H

if not H.artifacts_present():
    pytest.skip("modern map/route artifacts not present", allow_module_level=True)

from lanerl_jax.modern import world as MS  # noqa: E402
from lanerl_jax.modern.jungle import camps as J  # noqa: E402
from lanerl_jax.modern.world.phases.attack import minion_pushing  # noqa: E402

assert 8451 in H.JAX_PAGE.secondary          # Overgrowth (test_overgrowth_counts_...)
BRUSH = (2274.0, 13558.0)             # lane brush beside the top-lane midpoint (test_vision)
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
    mx, my = H.lane_mid(cfg)
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
    from lanerl_jax.modern.items import inventory as I
    cfg, step, _, refresh = world()
    s = MS.init_state(cfg)
    mx, my = H.lane_mid(cfg)
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
    cfg, step, run, _ = world()
    goal = H.lane_mid(cfg)
    o = orders(move=[True, True], move_x=[goal[0]] * 2, move_y=[goal[1]] * 2)
    s, _ = step(MS.init_state(cfg), o)
    s, _ = run(s, MS.no_orders(), 60 * CHUNK)
    d = np.hypot(np.asarray(s.x[:2]) - goal[0], np.asarray(s.y[:2]) - goal[1])
    assert (d < 150.0).all(), d


def test_a_dash_keeps_moving_after_its_start_tick():
    cfg, step, run, _ = world()
    s = MS.init_state(cfg)
    x0, y0 = float(s.x[0]), float(s.y[0])
    dash = W.Dash(jnp.asarray([True, False]), jnp.asarray([x0 + 400.0, 0.0]), jnp.asarray([y0 + 400.0, 0.0]),
                  jnp.asarray([800.0, 0.0]), jnp.asarray([-1, -1], jnp.int32), jnp.asarray([False, False]))
    s, _ = step(s._replace(prev=s.prev._replace(pending_dash=dash)), MS.no_orders())
    s1 = np.hypot(float(s.x[0]) - x0, float(s.y[0]) - y0)
    s, _ = run(s, MS.no_orders(), 10)
    s2 = np.hypot(float(s.x[0]) - x0, float(s.y[0]) - y0)
    assert s1 < 40.0 and s2 > s1 + 150.0, (s1, s2)          # 800 u/s for 10 more ticks: ~260 u


def _arc(lane):
    """Cumulative arc length of the (W, 2) lane polyline and a point->arc projector."""
    seg = np.diff(lane, axis=0)
    cum = np.concatenate([[0.0], np.cumsum(np.hypot(seg[:, 0], seg[:, 1]))])

    def at(a):
        i = int(np.clip(np.searchsorted(cum, a) - 1, 0, len(seg) - 1))
        f = (a - cum[i]) / max(cum[i + 1] - cum[i], 1e-6)
        return lane[i] + f * seg[i]

    def project(p):
        a = lane[:-1]
        t = np.clip(np.einsum("ij,ij->i", p - a, seg) / np.maximum(np.einsum("ij,ij->i", seg, seg), 1e-6), 0, 1)
        q = a + t[:, None] * seg
        i = int(np.argmin(np.hypot(*(q - p).T)))
        return cum[i] + t[i] * np.hypot(*seg[i])
    return cum, at, project


def test_champion_walks_back_through_its_own_wave_in_bounded_time():
    """Garen walking to base straight through his oncoming wave steers around it (COLLISION.md)."""
    from lanerl_jax.modern.lane import ai as LA
    cfg, step, run, refresh = world()
    s, _ = state_at_95s()
    lane = np.asarray(cfg.lane_path)
    _, at, project = _arc(lane)
    k, a, team, ln = (np.asarray(v) for v in (s.kind, s.alive, s.team, s.lane_ai.lane))
    wave = np.flatnonzero((k == W.KIND_MINION) & a & (team == 0) & (ln == LA.LANE_TOP))
    arcs = np.array([project(np.array([float(s.x[m]), float(s.y[m])])) for m in wave])
    front = arcs.max()
    rear = arcs[arcs > front - 1200.0].min()
    start, goal = at(front + 150.0), at(rear - 500.0)
    length = front + 150.0 - (rear - 500.0)
    s = refresh(s._replace(x=s.x.at[0].set(float(start[0])), y=s.y.at[0].set(float(start[1])),
                           route_anchor=s.route_anchor.at[0].set(-1)))
    pts = np.stack([np.asarray(s.x)[wave], np.asarray(s.y)[wave]], -1)
    u = (goal - start) / np.hypot(*(goal - start))
    d = np.abs(u[0] * (pts[:, 1] - start[1]) - u[1] * (pts[:, 0] - start[0]))
    assert (d < 150.0).sum() >= 3                                       # the wave really is in the way
    o = orders(move=[True, False], move_x=[float(goal[0]), 0.0], move_y=[float(goal[1]), 0.0])
    bound = length / 340.0 * 1.35 + 1.0                                 # Garen 340 u/s base
    s, _ = step(s, o)
    s, _ = run(s, MS.no_orders(), int(bound * CHUNK))
    assert np.hypot(float(s.x[0]) - goal[0], float(s.y[0]) - goal[1]) < 150.0, (length, bound)


def test_top_waves_meet_and_fight_near_the_middle():
    from lanerl_jax.modern.lane import ai as LA
    cfg, step, _, _ = world()
    s, _ = state_at_95s()
    xy = np.stack([np.asarray(s.x), np.asarray(s.y)], -1)

    def top(s, t):
        k, a, team, ln = (np.asarray(v) for v in (s.kind, s.alive, s.team, s.lane_ai.lane))
        return (k == W.KIND_MINION) & a & (team == t) & (ln == LA.LANE_TOP)
    b, r = top(s, 0), top(s, 1)
    gap = np.hypot(*(xy[b][:, None] - xy[r][None, :]).transpose(2, 0, 1)).min()
    assert gap < 700.0                                                  # in each other's attack range
    hurt = np.zeros(2, bool)                                            # both sides take hits within 1 s
    for _ in range(CHUNK):
        s, _ = step(s, MS.no_orders())
        dmg = np.asarray(s.hp) < np.asarray(s.max_hp) - 1.0
        hurt |= [bool((dmg & top(s, 0)).any()), bool((dmg & top(s, 1)).any())]
    assert hurt.all()


def test_a_move_order_ends_on_arrival():
    """MECHANICS_AUDIT #2."""
    cfg, step, run, _ = world()
    s = MS.init_state(cfg)
    gx, gy = float(s.x[0]) + 300.0, float(s.y[0]) + 300.0
    s, _ = step(s, orders(move=[True, False], move_x=[gx, 0.0], move_y=[gy, 0.0]))
    assert bool(s.champ.moving[0])
    s, _ = run(s, MS.no_orders(), 60)
    assert np.hypot(float(s.x[0]) - gx, float(s.y[0]) - gy) < 10.0 and not bool(s.champ.moving[0])


def test_an_out_of_range_jax_q_walks_into_range_and_casts():
    """MECHANICS_AUDIT #4."""
    cfg, step, run, refresh = world()
    s = MS.init_state(cfg)
    s = refresh(s._replace(x=s.x.at[0].set(BRUSH_EDGE[0]).at[1].set(BRUSH_EDGE[0] + 1100.0),
                           y=s.y.at[0].set(BRUSH_EDGE[1]).at[1].set(BRUSH_EDGE[1]), t=jnp.float32(60.0),
                           champ=s.champ._replace(ranks=s.champ.ranks.at[1, 0].set(1))))
    assert bool(s.visible[1, 0])
    s, _ = step(s, orders(cast_slot=[-1, 0], cast_target=[-1, 0]))
    assert int(s.champ.queued_cast.slot[1]) == 0 and float(s.champ.cooldowns[1, 0]) == 0.0
    s, _ = run(s, MS.no_orders(), 90)
    assert float(s.champ.cooldowns[1, 0]) > 0.0 and int(s.champ.queued_cast.slot[1]) == -1


def test_an_attack_target_lost_to_fog_is_chased_to_where_it_was_seen():
    """MECHANICS_AUDIT #10."""
    cfg, step, run, refresh = world()
    s = MS.init_state(cfg)
    seen_at = BRUSH_EDGE                                       # just outside the lane brush
    s = refresh(s._replace(x=s.x.at[0].set(BRUSH_EDGE[0] + 500.0).at[1].set(seen_at[0]),
                           y=s.y.at[0].set(BRUSH_EDGE[1] - 200.0).at[1].set(seen_at[1]), t=jnp.float32(60.0)))
    assert bool(s.visible[0, 1])
    s, _ = step(s, orders(attack=[1, -1]))
    s = refresh(s._replace(x=s.x.at[1].set(BRUSH[0]), y=s.y.at[1].set(BRUSH[1])))   # Jax ducks into brush
    assert not bool(s.visible[0, 1])
    s, _ = step(s, MS.no_orders())
    assert int(s.champ.attack_order[0]) == -1 and bool(s.champ.moving[0])
    np.testing.assert_allclose(np.asarray(s.champ.move_goal[0]), seen_at, atol=20.0)


def test_minion_pushing_favours_the_higher_level_team():
    """MECHANICS_AUDIT #6: from 3:30 a level lead buffs the leading team's lane minions."""
    cfg, _, _, _ = world()
    s, _ = state_at_95s()
    s = s._replace(econ=s.econ._replace(level=jnp.asarray([3, 1], s.econ.level.dtype)))
    bonus, div = minion_pushing(s, cfg, s.lane_ai, 300.0)
    minion = (np.asarray(s.kind) == W.KIND_MINION) & np.asarray(s.alive)
    team = np.asarray(s.team)
    assert np.allclose(np.asarray(bonus)[minion & (team == 0)], 0.10)   # (5% + 0 turret lead) x 2 levels
    assert np.allclose(np.asarray(bonus)[minion & (team == 1)], 0.0)
    assert np.allclose(np.asarray(minion_pushing(s, cfg, s.lane_ai, 200.0)[0]), 0.0)   # before 3:30
    assert np.allclose(np.asarray(div)[minion], 1.0)                   # equal turrets: no divisor
