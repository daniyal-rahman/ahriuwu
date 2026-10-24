import numpy as np
import pytest

from lanerl_jax.modern.bc.history import received_cast_ages


def test_received_age_is_causal_and_unknown_before_first_cast():
    t = np.array([0., .1, .2, .3, .4])
    before = received_cast_ages(t, [(.15, 0)])
    after = received_cast_ages(t, [(.15, 0), (.3, 0), (10., 1)])
    assert np.isnan(before[:2]).all()
    np.testing.assert_allclose(before[2:, 0], [.05, .15, .25], atol=1e-7)
    np.testing.assert_array_equal(before[:3], after[:3])
    np.testing.assert_allclose(after[3:, 0], [0., .1], atol=1e-7)
    assert np.isnan(after[:, 1:]).all()
    # Stored observation lag must delay the feature together with the original inputs.
    np.testing.assert_array_equal(before[np.maximum(np.arange(5)-1, 0)][3], before[2])


def test_received_age_canonical_summoners_and_out_of_window_history():
    result = received_cast_ages([100., 150.], [(20., 4), (40., 5), (75., 12)], {4: 1, 5: 0})
    np.testing.assert_array_equal(result[:, 4:], [[60., 80.], [110., 130.]])
    assert np.isnan(result[:, :4]).all()


def test_received_age_refuses_invalid_time_or_summoner_identity():
    with pytest.raises(ValueError):
        received_cast_ages([1., 0.], [])
    with pytest.raises(ValueError):
        received_cast_ages([1.], [(float('nan'), 0)])
    with pytest.raises(ValueError):
        received_cast_ages([1.], [], {4: 0, 5: 0})
