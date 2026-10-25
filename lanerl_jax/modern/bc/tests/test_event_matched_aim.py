import numpy as np

from lanerl_jax.modern.bc.event_matched_aim import direction_error, matched_indices


def test_matched_aim_keeps_predicted_and_received_frame_indices_separate():
    t = np.arange(6)*.1
    pairs = matched_indices(t, np.array([0, 0, 3, 0, 0, 0]),
                            np.array([0, 3, 0, 0, 0, 0]), np.ones(6, bool), 3)
    np.testing.assert_array_equal(pairs, [[1, 2]])
    predicted = np.array([0., np.deg2rad(179), 0., 0., 0., 0.])
    target = np.array([0., 0., np.deg2rad(-179), 0., 0., 0.])
    np.testing.assert_allclose(direction_error(predicted[pairs[:, 0]], target[pairs[:, 1]]), [2.])


def test_matched_aim_does_not_pair_across_invalid_scope():
    pairs = matched_indices(np.arange(4)*.1, np.array([3, 0, 0, 0]), np.array([0, 0, 0, 3]),
                            np.array([True, False, True, True]), 3)
    assert pairs.shape == (0, 2)
