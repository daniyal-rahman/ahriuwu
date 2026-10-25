import numpy as np

from lanerl_jax.modern.bc.event_calibration import calibrated_probabilities, fit_rate_bias


def test_temperature_bias_restores_calibration_counts_and_is_causal():
    probability = np.tile([.9, .1, 0.], (100, 1))
    labels = np.array([0]*60+[1]*40)
    bias = fit_rate_bias(probability, labels, .5)
    calibrated = calibrated_probabilities(probability, .5, bias)
    np.testing.assert_allclose(calibrated.mean(0), [.6, .4, 0.], atol=1e-5)
    np.testing.assert_array_equal(calibrated_probabilities(probability[:2], .5, bias), calibrated[:2])
    assert np.isfinite(bias).all()
