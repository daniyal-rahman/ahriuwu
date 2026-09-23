"""The JAX mover must equal the numpy reference, which was measured vs the server.

``server <--measured-- movement.py <--asserted here-- movement_jax.py``

The point of the middle link is economy: the server measurement costs a boot and
several minutes, so it is done once against a plain Python reference, and every
later implementation is held to *that* by cheap array equality. If this file
goes red, the JAX port drifted -- the server measurement is still good.
"""
from __future__ import annotations

import jax
import numpy as np
import pytest

import jax.numpy as jnp

from lanerl_jax.sim.movement import MoveState, step_move  # noqa: E402
from lanerl_jax.sim.movement_jax import (  # noqa: E402
    MAX_STEPS_PER_TICK,
    TICK_MS,
    max_steps_used,
    step_move_units,
)

W = 24


def _random_paths(n: int, seed: int, speed_lo=200.0, speed_hi=500.0):
    rng = np.random.default_rng(seed)
    wp = np.zeros((n, W, 2))
    nwp = np.zeros(n, np.int32)
    key = np.ones(n, np.int32)
    speeds = rng.uniform(speed_lo, speed_hi, n)
    xs = np.zeros(n)
    ys = np.zeros(n)
    for i in range(n):
        m = int(rng.integers(2, W))
        pts = np.cumsum(rng.uniform(-80, 80, (m, 2)), axis=0) + rng.uniform(0, 5000, 2)
        wp[i, :m] = pts
        nwp[i] = m
        key[i] = int(rng.integers(1, m))
        xs[i], ys[i] = pts[key[i] - 1]
    return xs, ys, wp, nwp, key, speeds


def _reference(xs, ys, wp, nwp, key, speeds, can_move=None):
    rx, ry, rk, moved = xs.copy(), ys.copy(), key.copy(), np.zeros(len(xs), bool)
    for i in range(len(xs)):
        if can_move is not None and not can_move[i]:
            continue
        st = MoveState(x=float(xs[i]), y=float(ys[i]))
        st.waypoints = [tuple(p) for p in wp[i, : nwp[i]]]
        st.key = int(key[i])
        moved[i] = step_move(st, float(speeds[i]), TICK_MS, max_waypoints=32)
        rx[i], ry[i], rk[i] = st.x, st.y, st.key
    return rx, ry, rk, moved


def _run(xs, ys, wp, nwp, key, speeds, can_move=None, max_steps=MAX_STEPS_PER_TICK):
    if can_move is None:
        can_move = np.ones(len(xs), bool)
    return step_move_units(
        jnp.asarray(xs), jnp.asarray(ys), jnp.asarray(wp),
        jnp.asarray(key, jnp.int8), jnp.asarray(nwp, jnp.int8),
        jnp.asarray(speeds), jnp.asarray(can_move), TICK_MS, max_steps,
    )


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_jax_mover_equals_the_numpy_reference(seed):
    xs, ys, wp, nwp, key, speeds = _random_paths(96, seed)
    rx, ry, rk, rmoved = _reference(xs, ys, wp, nwp, key, speeds)
    jx, jy, jk, jmoved = _run(xs, ys, wp, nwp, key, speeds)
    np.testing.assert_allclose(np.asarray(jx), rx, rtol=0, atol=1e-9)
    np.testing.assert_allclose(np.asarray(jy), ry, rtol=0, atol=1e-9)
    assert (np.asarray(jk).astype(int) == rk).all()
    assert (np.asarray(jmoved) == rmoved).all()


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_the_float32_mover_tracks_the_reference_to_float32_rounding(seed):
    """`STRUCT-007`: the pin above runs in float64 (the suite's conftest
    enables x64), while production state is float32. This is the same
    comparison in the production dtype: every waypoint decision identical, and
    positions within 4 float32 ulps of the double-precision reference (one
    tick's arithmetic is a handful of rounded ops: a subtract, a norm, a
    divide, a multiply-add)."""
    xs, ys, wp, nwp, key, speeds = _random_paths(96, seed)
    f32 = np.float32
    # the reference runs on the SAME (float32-representable) inputs, so what
    # is measured is the tick's arithmetic, not the input rounding
    xs, ys, wp, speeds = (a.astype(f32).astype(np.float64)
                          for a in (xs, ys, wp, speeds))
    rx, ry, rk, rmoved = _reference(xs, ys, wp, nwp, key, speeds)
    jx, jy, jk, jmoved = step_move_units(
        jnp.asarray(xs, jnp.float32), jnp.asarray(ys, jnp.float32),
        jnp.asarray(wp, jnp.float32),
        jnp.asarray(key, jnp.int8), jnp.asarray(nwp, jnp.int8),
        jnp.asarray(speeds, jnp.float32), jnp.ones(len(xs), bool), TICK_MS,
        MAX_STEPS_PER_TICK)
    assert np.asarray(jx).dtype == np.float32
    assert (np.asarray(jk).astype(int) == rk).all()
    assert (np.asarray(jmoved) == rmoved).all()
    for got, ref in ((np.asarray(jx), rx), (np.asarray(jy), ry)):
        ulp = np.spacing(np.abs(ref).astype(f32)).astype(np.float64)
        assert (np.abs(got.astype(np.float64) - ref) <= 4 * ulp).all(), (
            np.max(np.abs(got.astype(np.float64) - ref) / ulp))


def test_can_move_false_is_a_no_op():
    """``ObjAIBase.Move`` returns false under CastSpell/OrderNone/Stop/Taunt."""
    xs, ys, wp, nwp, key, speeds = _random_paths(32, 7)
    can = np.zeros(32, bool)
    jx, jy, jk, jmoved = _run(xs, ys, wp, nwp, key, speeds, can_move=can)
    np.testing.assert_array_equal(np.asarray(jx), xs)
    np.testing.assert_array_equal(np.asarray(jy), ys)
    assert (np.asarray(jk).astype(int) == key).all()
    assert not np.asarray(jmoved).any()


def test_a_finished_path_does_not_move():
    """``CurrentWaypointKey >= Waypoints.Count`` -> the method returns false."""
    wp = np.zeros((1, W, 2))
    wp[0, :2] = [[0.0, 0.0], [100.0, 0.0]]
    jx, jy, jk, moved = _run(np.array([100.0]), np.array([0.0]), wp,
                             np.array([2]), np.array([2]), np.array([345.0]))
    assert float(jx[0]) == 100.0 and not bool(moved[0])


def test_leftover_distance_carries_across_waypoints():
    """The tick does not end at a waypoint; the remainder is spent on the next.

    A mover that stopped at each waypoint would lag by up to one waypoint per
    tick around a tight corner, which is exactly where last-hit positioning
    happens.
    """
    wp = np.zeros((1, W, 2))
    wp[0, :4] = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [1000.0, 0.0]]
    jx, _, jk, _ = _run(np.array([0.0]), np.array([0.0]), wp,
                        np.array([4]), np.array([1]), np.array([345.0]))
    assert float(jx[0]) == pytest.approx(5.75)     # 345 * 16.667 / 1000
    assert int(jk[0]) == 3


def test_the_step_bound_is_large_enough_for_random_paths():
    """If a tick needs more steps than the bound, units stop short silently.

    ``max_steps_used`` measures the requirement so the bound is set from data.
    """
    xs, ys, wp, nwp, key, speeds = _random_paths(256, 11, speed_lo=300, speed_hi=700)
    used = np.asarray(max_steps_used(
        jnp.asarray(wp), jnp.asarray(key, jnp.int8),
        jnp.asarray(nwp, jnp.int8), jnp.asarray(speeds), TICK_MS, probe=32))
    assert used.max() < 32, "probe itself was saturated; the answer is a lower bound"
    assert used.max() <= MAX_STEPS_PER_TICK, (
        f"a tick consumed {used.max()} waypoints but MAX_STEPS_PER_TICK is "
        f"{MAX_STEPS_PER_TICK}; raise it rather than letting units stop short"
    )


def test_it_vmaps_over_environments():
    xs, ys, wp, nwp, key, speeds = _random_paths(66, 3)
    f = lambda a, b: step_move_units(  # noqa: E731
        a, b, jnp.asarray(wp), jnp.asarray(key, jnp.int8),
        jnp.asarray(nwp, jnp.int8), jnp.asarray(speeds),
        jnp.ones(66, bool), TICK_MS, MAX_STEPS_PER_TICK)
    out = jax.vmap(f)(jnp.tile(jnp.asarray(xs), (32, 1)),
                      jnp.tile(jnp.asarray(ys), (32, 1)))
    assert out[0].shape == (32, 66)
    # every env got the same inputs, so every env must agree
    assert np.allclose(np.asarray(out[0])[0], np.asarray(out[0])[-1])


def test_jit_and_scan_compose():
    """The step must survive being scanned over ticks inside one jit."""
    xs, ys, wp, nwp, key, speeds = _random_paths(66, 5)
    wpj, keyj = jnp.asarray(wp), jnp.asarray(key, jnp.int8)
    nwpj, spj = jnp.asarray(nwp, jnp.int8), jnp.asarray(speeds)

    @jax.jit
    def roll(x, y, k):
        def one(c, _):
            x, y, k = c
            x, y, k, _ = step_move_units(x, y, wpj, k, nwpj, spj,
                                         jnp.ones(66, bool), TICK_MS,
                                         MAX_STEPS_PER_TICK)
            return (x, y, k), None
        (x, y, k), _ = jax.lax.scan(one, (x, y, k), None, length=60)
        return x, y, k

    x1, y1, _ = roll(jnp.asarray(xs), jnp.asarray(ys), keyj)
    assert np.isfinite(np.asarray(x1)).all() and np.isfinite(np.asarray(y1)).all()
