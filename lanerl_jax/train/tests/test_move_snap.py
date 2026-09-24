"""The MOVE-point snap in `train/actions.orders_from` (`PATH-010`, R2).

(a) every 96x54 screen bin, decoded from the blue spawn, the red spawn and
    three lane positions (red at the canonical mirror of blue's), yields a
    MOVE point whose own cell is standable for a 35-u champion by the sim's own
    query (`terrain_jax.is_walkable(cell centre, 35)`).  The raw decode is
    measured too and must NOT pass, or the check checks nothing.
(b) a click whose cell is already standable is passed through bit-identically,
    and non-MOVE orders are untouched.
(c) the host `GetPath` (`NavGrid.get_cell_path`, r=35) from each spawn to every
    snapped goal it decodes to is non-null.
(d) the table is the route artifact's traversable set and orders_from stays
    jit/vmap-compatible.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.data.navgrid import GAREN_PATHFINDING_RADIUS, NavGrid
from lanerl_jax.obs.frame import make_lane_frame, to_lane
from lanerl_jax.sim.init import CHAMPION_SPAWN, TOP_OUTER_TURRET, init_lane
from lanerl_jax.sim.orders import OrderKind
from lanerl_jax.sim.state import Team
from lanerl_jax.sim.terrain_jax import is_walkable, map1_terrain
from lanerl_jax.train.actions import (MOVE_SNAP_RADIUS, move_snap_table,
                                      orders_from)
from lanerl_jax.train.trainer import BLUE_NEXUS, RED_NEXUS
from lanerl_rl.constants import BUTTON_INDEX, N_SCREEN_X, N_SCREEN_Y, N_SLOTS

FB = make_lane_frame(TOP_OUTER_TURRET[Team.BLUE], TOP_OUTER_TURRET[Team.RED],
                     BLUE_NEXUS)
FR = make_lane_frame(TOP_OUTER_TURRET[Team.RED], TOP_OUTER_TURRET[Team.BLUE],
                     RED_NEXUS)


def _mirror(p):
    """Red's position at blue's canonical (s, n) at world point ``p``."""
    s, n = (float(v) for v in to_lane(FB, jnp.float32(p[0]), jnp.float32(p[1])))
    return (float(FR.origin[0] + s * FR.axis[0] + n * FR.normal[0]),
            float(FR.origin[1] + s * FR.axis[1] + n * FR.normal[1]))


LANE = {"turret_front": (880.0, 10180.0), "lane_corner": (1268.0, 11675.0),
        "lane_2806_13075": (2806.0, 13075.0)}
CASES = {"spawn": (CHAMPION_SPAWN[Team.BLUE], CHAMPION_SPAWN[Team.RED])}
CASES.update({k: (p, _mirror(p)) for k, p in LANE.items()})


def _decode(blue, red, button, snap):
    st = init_lane()
    st = st.replace(x=st.x.at[0].set(blue[0]).at[1].set(red[0]),
                    y=st.y.at[0].set(blue[1]).at[1].set(red[1]))
    sx, sy = np.meshgrid(np.arange(N_SCREEN_X), np.arange(N_SCREEN_Y), indexing="ij")
    sx, sy = jnp.asarray(sx.ravel()), jnp.asarray(sy.ravel())
    slots = jnp.full((2, N_SLOTS), -1, jnp.int32)

    def one(a, b):
        act = (jnp.full((2,), button), jnp.stack([a, a]), jnp.stack([b, b]),
               jnp.zeros((2,), jnp.int32))
        o = orders_from(act, st, slots, FB, snap_moves=snap)
        return o.kind, o.x, o.y
    k, x, y = jax.jit(jax.vmap(one))(sx, sy)
    return np.asarray(k), np.asarray(x), np.asarray(y)


_T = map1_terrain()
_H, _W = _T.walkable.shape


@jax.jit
@jax.vmap
def _cell_standable(x, y):
    """Is the point in the grid and its cell's centre walkable at r=35?"""
    nx = (x - _T.min_x) / _T.cell_size
    ny = (y - _T.min_y) / _T.cell_size
    inb = (nx >= 0) & (nx < _W) & (ny >= 0) & (ny < _H)
    cx = _T.min_x + (jnp.floor(nx) + 0.5) * _T.cell_size
    cy = _T.min_y + (jnp.floor(ny) + 0.5) * _T.cell_size
    return inb & is_walkable(cx, cy, jnp.float32(MOVE_SNAP_RADIUS), _T)


def _unstandable_fraction(k, x, y):
    m = k == OrderKind.MOVE
    ok = np.asarray(_cell_standable(jnp.asarray(x[m]), jnp.asarray(y[m])))
    return 1.0 - ok.mean(), int(m.sum())


def test_radius_is_the_sims_pathfinding_radius():
    assert MOVE_SNAP_RADIUS == GAREN_PATHFINDING_RADIUS == 35.0


def test_table_is_the_route_artifacts_traversable_set():
    t = move_snap_table()
    st = np.asarray(t.standable)
    near = np.asarray(t.nearest)
    assert st.sum() == 47477                      # route artifact source_count
    assert np.all(st[near])                       # every target standable
    idx = np.arange(st.size)
    assert np.array_equal(near[st], idx[st])      # standable cells map to self


@pytest.mark.parametrize("button", ["move", "attack_move"])
@pytest.mark.parametrize("case", list(CASES))
def test_a_every_bin_decodes_to_a_standable_cell(case, button):
    blue, red = CASES[case]
    b = BUTTON_INDEX[button]
    k0, x0, y0 = _decode(blue, red, b, snap=False)
    k1, x1, y1 = _decode(blue, red, b, snap=True)
    assert np.array_equal(k0, k1)
    report = []
    for side in (0, 1):
        before, n = _unstandable_fraction(k0[:, side], x0[:, side], y0[:, side])
        after, _ = _unstandable_fraction(k1[:, side], x1[:, side], y1[:, side])
        report.append((side, n, before, after))
        assert n > 4000
        assert after == 0.0, (case, side, after)
        if case == "spawn":
            # the check is live: the raw decode fails it from either spawn
            assert before > 0.3, (case, side, before)
    print(f"\n{case} {button}: " + "  ".join(
        f"{'blue' if s == 0 else 'red'} n={n} unstandable before={b_:.3f} after={a_:.3f}"
        for s, n, b_, a_ in report))


@pytest.mark.parametrize("case", list(CASES))
def test_b_standable_clicks_bit_identical(case):
    blue, red = CASES[case]
    for button in ("move", "attack_move", "q", "w", "e", "r", "recall"):
        k0, x0, y0 = _decode(blue, red, BUTTON_INDEX[button], snap=False)
        k1, x1, y1 = _decode(blue, red, BUTTON_INDEX[button], snap=True)
        assert np.array_equal(k0, k1)
        ok = np.zeros(k0.shape, bool)
        for side in (0, 1):
            ok[:, side] = np.asarray(_cell_standable(jnp.asarray(x0[:, side]),
                                                     jnp.asarray(y0[:, side])))
        same = (k0 != OrderKind.MOVE) | ok
        assert same.sum() > 0
        assert np.array_equal(x0[same].view(np.uint32), x1[same].view(np.uint32))
        assert np.array_equal(y0[same].view(np.uint32), y1[same].view(np.uint32))
        if button == "move":
            changed = int((~same).sum())
            assert changed > 0


def test_c_snapped_goal_has_a_host_path_from_each_spawn():
    grid = NavGrid.load()
    blue, red = CASES["spawn"]
    k, x, y = _decode(blue, red, BUTTON_INDEX["move"], snap=True)
    for side, src in ((0, blue), (1, red)):
        m = k[:, side] == OrderKind.MOVE
        goals = sorted({(float(a), float(b)) for a, b in zip(x[m, side], y[m, side])})
        nulls = [g for g in goals
                 if grid.get_cell_path(tuple(map(float, src)), g,
                                       GAREN_PATHFINDING_RADIUS) is None
                 and g != tuple(map(float, src))]
        assert not nulls, (side, len(nulls), len(goals), nulls[:5])


def test_d_orders_from_jits_under_vmap_over_envs():
    st = init_lane()
    n_env = 4
    states = jax.tree_util.tree_map(lambda a: jnp.stack([a] * n_env), st)
    rng = np.random.default_rng(0)
    act = (jnp.full((n_env, 2), BUTTON_INDEX["move"]),
           jnp.asarray(rng.integers(0, N_SCREEN_X, (n_env, 2))),
           jnp.asarray(rng.integers(0, N_SCREEN_Y, (n_env, 2))),
           jnp.zeros((n_env, 2), jnp.int32))
    slots = jnp.full((n_env, 2, N_SLOTS), -1, jnp.int32)
    f = jax.jit(jax.vmap(lambda a, s, su: orders_from(a, s, su, FB)))
    o = f(act, states, slots)
    ok = np.asarray(_cell_standable(o.x.reshape(-1), o.y.reshape(-1)))
    assert ok[np.asarray(o.kind).reshape(-1) == OrderKind.MOVE].all()
