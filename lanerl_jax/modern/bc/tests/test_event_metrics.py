import numpy as np

from lanerl_jax.modern.bc.event_metrics import average_precision, valid_segments, sequence_metrics


def test_ap_ties_are_not_broken_using_label_order():
    assert average_precision([True, False, False, True], [1., 1., 1., 1.]) == .5
    assert average_precision([True, False], [1., 0.]) == 1.
    assert average_precision([False], [1.]) is None


def test_event_matching_cannot_cross_invalid_observation_gap():
    assert valid_segments([True, True, False, True]) == [(0, 2), (3, 4)]
    arrays = dict(t=np.arange(4)*.1, button=np.array([3, 0, 0, 0]), valid=np.array([True, True, False, True]),
                  aim_valid=np.zeros(4, bool), aim_mode=np.zeros(4, np.int8))
    pred = np.array([0, 0, 0, 3])
    predictions = dict(button=pred, probabilities=np.eye(9)[pred])
    metrics = sequence_metrics(arrays, predictions, burn=0)
    assert metrics['classes']['q']['received'] == metrics['classes']['q']['predicted'] == 1
    assert metrics['classes']['q']['true_positive'] == 0
