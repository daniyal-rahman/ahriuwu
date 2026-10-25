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


def test_joint_target_quality_counts_missed_buttons_as_failures():
    arrays = dict(t=np.arange(2)*.1, button=np.array([3, 3]), valid=np.ones(2, bool),
                  aim_valid=np.ones(2, bool), aim_mode=np.full(2, 2),
                  screen_x=np.array([5, 5]), screen_y=np.array([3, 3]))
    predictions = dict(button=np.array([3, 0]), probabilities=np.eye(9)[[3, 0]],
                       actual_grid=np.full(2, 3*96+5), teacher_grid=np.full(2, 3*96+5))
    result = sequence_metrics(arrays, predictions, burn=0)
    actual = result['targets_by_button']['actual_point']['q']
    assert actual['count'] == 1 and actual['eligible'] == 2
    assert actual['within2'] == 1. and actual['joint_success'] == .5
