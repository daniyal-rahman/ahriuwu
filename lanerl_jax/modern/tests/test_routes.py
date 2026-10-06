from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from lanerl_jax.modern.data.navgrid import ModernMapGrid
from lanerl_jax.modern.data.routes import build_routes, clear_segment, load_routes
from lanerl_jax.modern.map.pathing import route_next, segment_clear


def grid(flags):
    n = flags.shape[0]
    return ModernMapGrid(flags, np.zeros((n, n, 4), np.uint8), np.zeros((n + 1, n + 1), np.float32),
                         (25, 25), 50, (0, -1, 0), (50 * n, 1, 50 * n))


def test_routes_go_around_wall_and_are_jittable(tmp_path):
    flags = np.zeros((12, 12), np.uint16)
    flags[2:10, 6] = 2
    g = grid(flags)
    build_routes(g, tmp_path / "routes", spacing=50, radius=10)
    routes, _ = load_routes(tmp_path / "routes", g)
    src, goal = jnp.array([225., 225.]), jnp.array([425., 225.])
    assert not clear_segment(g, np.asarray(src), np.asarray(goal), 10)
    f = jax.jit(lambda p: route_next(p, goal, 10., routes, g.as_jax()))
    point = src
    for _ in range(30):
        nxt, ok = f(point)
        assert bool(ok)
        assert clear_segment(g, np.asarray(point), np.asarray(nxt), 10)
        if np.linalg.norm(np.asarray(nxt - goal)) < .01:
            break
        assert np.linalg.norm(np.asarray(nxt - point)) > .01
        point = nxt
    else:
        raise AssertionError("route failed to reach goal")
    assert not bool(segment_clear(src, goal, 10, g.as_jax()))


def test_route_artifact_rejects_different_terrain(tmp_path):
    flags = np.zeros((4, 4), np.uint16)
    g = grid(flags)
    build_routes(g, tmp_path / "routes", spacing=50, radius=10)
    f = flags.copy()
    f[1, 1] = 2
    with pytest.raises(ValueError, match="terrain mismatch"):
        load_routes(tmp_path / "routes", replace(g, flags=f))
