"""``collision``: avoidance steering + soft separation (docs/modern/COLLISION.md), no tick compile."""
import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.modern import collision as UC
from lanerl_jax.modern.core import types as W
from lanerl_jax.modern.map.terrain import StaticTerrain, is_walkable

DT = 1.0 / 30.0
R = 35.0                                         # champion pathing radius
CD = float(UC.contact_distance(R, R))            # pair contact distance


def _terrain(wall_x=None):
    """2000 x 2000 open field (50 u cells); optional solid wall for x >= ``wall_x``."""
    g = np.ones((40, 40), bool)
    if wall_x is not None:
        g[:, int(wall_x // 50):] = False
    t = StaticTerrain(jnp.asarray(g), 50.0, 0.0, 0.0, 2000.0, 2000.0)
    return (t, t)


def _resolve(x0, y0, x1, y1, *, ghosted=None, moving=None, goal=None, terrain=None, radius=R):
    n = len(x0)
    f = lambda v: jnp.asarray(v, jnp.float32)                                  # noqa: E731
    gx, gy = (np.asarray(x1, float), np.asarray(y1, float)) if goal is None else goal
    return _jit_resolve(f(x0), f(y0), f(x1), f(y1), jnp.full((n,), radius, jnp.float32),
                        jnp.zeros((n,), bool) if ghosted is None else jnp.asarray(ghosted),
                        jnp.zeros((n,), bool) if moving is None else jnp.asarray(moving),
                        f(gx), f(gy), terrain or _terrain())


@jax.jit
def _jit_resolve(x0, y0, x1, y1, r, ghosted, moving, gx, gy, terrain):
    n = x0.shape[0]
    return UC.resolve(x0, y0, x1, y1, radius=r, collide=jnp.ones((n,), bool), ghosted=ghosted, moving=moving,
                      goal_x=gx, goal_y=gy, team=jnp.zeros((n,), jnp.int32), clearance=jnp.full((n,), 35.0),
                      terrain=terrain, dt=DT)


def _walk(x, y, goal, speed, moving, ticks, terrain=None, ghosted=None):
    """Straight-line 'route' steps toward ``goal`` then collision, like ``_move``; returns the track."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    gx, gy = np.asarray(goal[0], float), np.asarray(goal[1], float)
    track = [(x.copy(), y.copy())]
    for _ in range(ticks):
        dx, dy = gx - x, gy - y
        d = np.hypot(dx, dy)
        st = np.minimum(np.asarray(speed) * DT, d)
        mv = np.asarray(moving) & (d > 1e-3)
        x1 = np.where(mv, x + dx / np.maximum(d, 1e-6) * st, x)
        y1 = np.where(mv, y + dy / np.maximum(d, 1e-6) * st, y)
        nx, ny = _resolve(x, y, x1, y1, moving=mv, goal=(gx, gy), terrain=terrain, ghosted=ghosted)
        x, y = np.asarray(nx, float), np.asarray(ny, float)
        track.append((x.copy(), y.copy()))
    return track


def test_pathing_radii_from_client_records():
    kind = jnp.asarray([W.KIND_CHAMPION, W.KIND_MINION, W.KIND_MINION, W.KIND_MINION, W.KIND_MONSTER])
    sub = jnp.asarray([0, 0, 2, 3, 7])                                          # Krug = 7
    r = np.asarray(UC.pathing_radius(kind, sub, jnp.asarray([65.0, 48.0, 65.0, 65.0, 100.0])))
    np.testing.assert_allclose(r, [35.0, 35.7437, 55.7437, 55.5208, 85.0], atol=1e-3)


def test_overlap_is_resolved_softly_over_a_few_ticks():
    r = 200.0                                                                   # deep overlap: capped per round
    x, y = [1000.0, 1010.0], [1000.0, 1000.0]
    nx, _ = _jit_resolve(*(jnp.asarray(v, jnp.float32) for v in (x, y, x, y)), jnp.full((2,), r),
                         jnp.zeros((2,), bool), jnp.zeros((2,), bool), jnp.asarray(x), jnp.asarray(y), _terrain())
    cd = float(UC.contact_distance(r, r))
    assert 10.0 < float(nx[1] - nx[0]) < cd - 1.0
    x, y = [1000.0, 1000.0 + CD / 3], [1000.0, 1000.0]                          # shallow overlap: one tick
    nx, _ = _resolve(x, y, x, y)
    assert abs(float(nx[1] - nx[0]) - CD) < 0.5
    assert abs(float(nx[0] + nx[1]) / 2 - (1000.0 + CD / 6)) < 1e-3              # equal mobility: symmetric


def test_stationary_unit_yields_less_than_a_mover():
    x, y = [1000.0, 1000.0 + CD / 2], [1000.0, 1000.0]
    nx, _ = _resolve(x, y, x, y, moving=[True, False])
    assert (1000.0 - float(nx[0])) > 3.0 * (float(nx[1]) - 1000.0 - CD / 2) > 0.0


def test_ghosted_units_neither_block_nor_are_pushed():
    x, y = [1000.0, 1010.0, 1500.0], [1000.0, 1000.0, 1000.0]
    nx, ny = _resolve(x, y, x, y, ghosted=[True, False, False])
    np.testing.assert_allclose(np.asarray(nx), x)
    np.testing.assert_allclose(np.asarray(ny), y)


def test_no_push_into_a_wall_partner_takes_the_push():
    ter = _terrain(wall_x=1100.0)
    x, y = [1060.0, 1060.0 - CD / 3], [1000.0, 1000.0]                          # unit 0 is flush with the wall
    for _ in range(4):
        x, y = (np.asarray(v, float) for v in _resolve(x, y, x, y, terrain=ter))
    assert bool(is_walkable(jnp.float32(x[0]), jnp.float32(y[0]), 35.0, ter[0]))
    assert x[0] <= 1065.0 and x[0] - x[1] >= CD - 1.0


def test_deterministic_and_order_independent():
    rng = np.random.default_rng(0)
    x, y = rng.uniform(950, 1050, 12), rng.uniform(950, 1050, 12)
    a = np.asarray(_resolve(x, y, x, y))
    b = np.asarray(_resolve(x, y, x, y))
    np.testing.assert_array_equal(a, b)
    p = rng.permutation(12)
    c = np.asarray(_resolve(x[p], y[p], x[p], y[p]))
    np.testing.assert_allclose(c, a[:, p], atol=1e-3)


def test_coincident_units_separate():
    nx, ny = _resolve([1000.0, 1000.0], [1000.0, 1000.0], [1000.0, 1000.0], [1000.0, 1000.0])
    assert np.hypot(float(nx[1] - nx[0]), float(ny[1] - ny[0])) > 10.0


def test_mover_steers_around_a_standing_unit_without_losing_speed():
    # Unit 0 walks +x at 345 u/s straight through unit 1 standing on its line.
    tr = _walk([800.0, 1000.0], [1000.0, 1000.0], ([1300.0, 1000.0], [1000.0, 1000.0]), [345.0, 0.0],
               [True, False], 45)
    xs = np.array([t[0] for t in tr]); ys = np.array([t[1] for t in tr])
    gap = np.hypot(xs[:, 0] - xs[:, 1], ys[:, 0] - ys[:, 1])
    assert gap.min() > CD - 3.0                                             # went around, no deep overlap
    assert xs[-1, 0] > 1200.0                                                  # passed it (1.5 s for 500 u)
    assert np.hypot(xs[-1, 1] - 1000.0, ys[-1, 1] - 1000.0) < 5.0              # the stander barely moved


def test_head_on_movers_pass_each_other():
    tr = _walk([800.0, 1200.0], [1000.0, 1000.0], ([1400.0, 600.0], [1000.0, 1000.0]), [345.0, 345.0],
               [True, True], 60)
    x, y = tr[-1]
    assert x[0] > 1300.0 and x[1] < 700.0
    gaps = [np.hypot(t[0][0] - t[0][1], t[1][0] - t[1][1]) for t in tr]
    assert min(gaps) > CD - 3.0


def test_chase_target_on_the_goal_is_not_avoided():
    # Goal = the standing unit's position (attack chase): approach straight, stop at contact.
    tr = _walk([800.0, 1000.0], [1000.0, 1000.0], ([1000.0, 1000.0], [1000.0, 1000.0]), [345.0, 0.0],
               [True, False], 30)
    ys = np.array([t[1][0] for t in tr])
    assert np.abs(ys - 1000.0).max() < 1e-3                                    # no sidestep


def test_avoidance_does_not_steer_into_a_wall():
    ter = _terrain(wall_x=1100.0)
    # Walking +y along the wall with the blocker on the open side: the free side is the wall.
    tr = _walk([1060.0, 1030.0], [800.0, 1000.0], ([1060.0, 1030.0], [1300.0, 1000.0]), [345.0, 0.0],
               [True, False], 40, terrain=ter)
    for x, y in tr:
        assert bool(is_walkable(jnp.float32(x[0]), jnp.float32(y[0]), 35.0, ter[0]))


def test_mover_gets_around_a_standing_clump():
    # A 2 x 3 block of standing units (contact-spaced) centred on the path, like a wave fighting.
    bx = [1000.0, 1000.0, 1000.0, 1000.0 + CD, 1000.0 + CD, 1000.0 + CD]
    by = [1000.0 - CD, 1000.0, 1000.0 + CD] * 2
    x = [700.0] + bx
    y = [1000.0] + by
    speed = [345.0] + [0.0] * 6
    moving = [True] + [False] * 6
    tr = _walk(x, y, ([1500.0] + bx, [1000.0] + by), speed, moving, 75)        # 800 u: 2.3 s straight
    assert tr[-1][0][0] > 1350.0
    for t in tr[1:]:
        assert np.hypot(t[0][1:] - np.asarray(bx), t[1][1:] - np.asarray(by)).max() < 15.0   # clump holds
