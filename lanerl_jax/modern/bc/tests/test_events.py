import numpy as np
import pytest

from lanerl_jax.modern.bc.events import BUTTON_INDEX as B, event_matches, smooth_cast_targets


def test_cast_smoothing_preserves_mass_with_collisions_gaps_and_boundaries():
    buttons = np.array([B['q'], 0, B['move'], B['e'], B['q'], 0, 0, B['r']])
    valid = np.array([True, True, True, True, True, False, True, True])
    target = smooth_cast_targets(buttons, valid)
    np.testing.assert_allclose(target.sum(1), valid, atol=1e-7)
    assert (target >= 0).all()
    for name in ('q', 'w', 'e', 'r'):
        np.testing.assert_allclose(target[:, B[name]].sum(), ((buttons == B[name]) & valid).sum())
    np.testing.assert_array_equal(target[2], np.eye(len(B))[B['move']])
    assert not target[5].any()


def test_single_frame_kernel_is_original_one_hot_and_invalid_rows_have_no_loss():
    buttons = np.array([0, B['q'], B['attack'], B['q']])
    valid = np.array([True, True, True, False])
    np.testing.assert_array_equal(smooth_cast_targets(buttons, valid, (1,)),
                                  np.eye(len(B))[buttons] * valid[:, None])


def test_event_matching_prevents_burst_predictions_from_getting_repeated_credit():
    result = event_matches([.8, .9, 1., 1.1, 1.2], [1.])
    assert result['true_positive'] == 1 and result['false_positive'] == 4
    assert result['recall'] == 1 and result['precision'] == .2


def test_ordered_matching_avoids_nearest_pair_cardinality_failure():
    # Nearest-first would match4→5 and lose9→5, instead of4→0 and9→5.
    result = event_matches([4., 9.], [0., 5.], tolerance=4.)
    assert result['pairs'] == [(0, 0), (1, 1)] and result['f1'] == 1


def test_event_contract_refuses_nonfinite_unordered_and_invalid_targets():
    with pytest.raises(ValueError):
        event_matches([1., 0.], [0.])
    with pytest.raises(ValueError):
        event_matches([np.nan], [0.])
    with pytest.raises(ValueError):
        smooth_cast_targets(np.array([99]), [True])
    with pytest.raises(ValueError):
        smooth_cast_targets(np.array([0]), [True], (1, 1))
