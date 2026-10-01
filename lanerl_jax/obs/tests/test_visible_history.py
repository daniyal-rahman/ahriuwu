"""Visible-history identity, masking, reset and permutation contracts."""
import jax
import jax.numpy as jnp
import numpy as np

from lanerl_jax.obs.builder import NORM_DIST, NORM_XY
from lanerl_jax.obs.visible_history import (
    append_visible_history, empty_visible_history, HISTORY_ENTITY_DIM, PAST_SAMPLES)


def frame(hp=(.5, .8), xy=((100., 100.), (260., 100.)), own=(0., 0.)):
    e = np.zeros((32, 16), np.float32)
    e[:2, 0] = 1
    e[:2, 1:3] = (np.asarray(xy)-np.asarray(own))/NORM_DIST
    e[:2, 3] = hp
    e[:2, 5] = 1  # minion
    e[:2, 11] = 1  # enemy
    e[:2, 14] = 1  # caster
    sv = np.zeros(16, np.float32)
    sv[:2] = np.asarray(own)/NORM_XY
    return jnp.asarray(e), jnp.asarray(e[:, 0] == 0), jnp.asarray(sv)


step = jax.jit(append_visible_history)


def temporal(e):
    assert e.shape == (32, HISTORY_ENTITY_DIM)
    return np.asarray(e[:, 16:]).reshape(32, PAST_SAMPLES, 4)


def test_visible_history_tracks_health_and_camera_compensated_positions():
    e, m, sv = frame()
    first, h = step(e, m, sv, empty_visible_history())
    np.testing.assert_array_equal(first[:, :16], e)
    assert not temporal(first).any()
    e2, m2, sv2 = frame(hp=(.4, .7), xy=((110., 100.), (250., 100.)), own=(70., -40.))
    second, h = step(e2, m2, sv2, h)
    t = temporal(second)
    np.testing.assert_allclose(t[:2, 0], [[.5, -10/NORM_DIST, 0, 1], [.8, 10/NORM_DIST, 0, 1]], atol=1e-7)
    assert not t[:, 1:].any()
    assert not t[2:].any()


def test_visible_history_is_equivariant_to_arbitrary_row_permutation():
    e, m, sv = frame()
    _, h = step(e, m, sv, empty_visible_history())
    e2, m2, sv2 = frame(hp=(.2, .6), xy=((115., 100.), (245., 100.)))
    ref, _ = step(e2, m2, sv2, h)
    perm = np.random.default_rng(12).permutation(32)
    past_perm = np.random.default_rng(14).permutation(32)
    hp = jax.tree.map(lambda x: x[past_perm], h)
    got, _ = step(e2[perm], m2[perm], sv2, hp)
    np.testing.assert_array_equal(got, ref[perm])


def test_visible_history_drops_fog_disappearance_and_episode_memory():
    e, m, sv = frame()
    _, h = step(e, m, sv, empty_visible_history())
    hidden = e.at[0].set(0)
    _, h = step(hidden, m.at[0].set(True), sv, h)
    seen_again, _ = step(e, m, sv, h)
    assert not temporal(seen_again)[0].any()
    assert temporal(seen_again)[1, 0, 3] == 1
    reset, _ = step(e, m, sv, empty_visible_history())
    assert not temporal(reset).any()


def test_visible_history_rejects_ambiguous_far_and_wrong_type_matches():
    e, m, sv = frame(xy=((100., 100.), (100., 100.)))
    _, h = step(e, m, sv, empty_visible_history())
    ambiguous, _ = step(e, m, sv, h)
    assert not temporal(ambiguous).any()
    e, m, sv = frame()
    _, h = step(e, m, sv, empty_visible_history())
    far, _ = step(e.at[0, 1].add(1000/NORM_DIST), m, sv, h)
    assert not temporal(far)[0].any()
    changed = e.at[0, 14].set(0).at[0, 13].set(1)
    wrong_type, _ = step(changed, m, sv, h)
    assert not temporal(wrong_type)[0].any()


def test_visible_history_keeps_zero_hp_visible_and_all_fifteen_lags():
    h = empty_visible_history()
    for i in range(17):
        e, m, sv = frame(hp=(max(15-i, 0)/60., .8))
        out, h = step(e, m, sv, h)
    t = temporal(out)
    np.testing.assert_array_equal(t[0, :, 3], np.ones(15))
    assert t[0, 0, 0] == 0 and t[0, 0, 3] == 1
    np.testing.assert_allclose(t[0, :, 0], np.arange(15)/60., atol=1e-7)
    assert np.isfinite(out).all()
