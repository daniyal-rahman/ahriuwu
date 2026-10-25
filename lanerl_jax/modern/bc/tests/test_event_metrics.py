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


def test_tolerant_joint_target_uses_identity_not_slot_or_exact_frame():
    arrays = dict(t=np.arange(3)*.1, button=np.array([3, 0, 0]), valid=np.ones(3, bool),
        aim_valid=np.array([True, False, False]), aim_mode=np.array([1, 0, 0]), target_row=np.array([1, -1, -1]),
        unit_ids=np.array([[-1, 99], [99, -1], [-1, -1]]))
    predictions = dict(button=np.array([0, 3, 0]), probabilities=np.eye(9)[[0, 3, 0]],
        actual_pointer=np.array([-1, 0, -1]), teacher_pointer=np.array([1, 0, 0]))
    result = sequence_metrics(arrays, predictions, burn=0)
    assert result['targets_by_button']['actual_unit']['q']['joint_success'] == 0
    assert result['matched_event_targets']['q']['joint_success'] == 1


def test_tolerant_point_score_reprojects_between_moving_cameras():
    from lanerl_jax.modern.bc.event_metrics import cell_world
    from lanerl_jax.modern.bc.labels import lane_to_screen
    axis, normal = np.array([1., 0.]), np.array([0., 1.])
    world = cell_world(27*96 + 48, [0., 0.], axis, normal)
    next_anchor = np.array([100., 0.])
    sx, sy = lane_to_screen(*(world - next_anchor))
    next_cell = int(sy*54)*96 + int(sx*96)
    arrays = dict(t=np.arange(2)*.1, button=np.array([3, 0]), valid=np.ones(2, bool),
        aim_valid=np.array([True, False]), aim_mode=np.array([2, 0]), screen_x=np.array([48, 0]),
        screen_y=np.array([27, 0]), anchor_xy=np.array([[0., 0.], next_anchor]), frame_axis=axis, frame_normal=normal)
    predictions = dict(button=np.array([0, 3]), probabilities=np.eye(9)[[0, 3]],
        actual_grid=np.array([0, next_cell]), teacher_grid=np.array([27*96+48, 0]))
    result = sequence_metrics(arrays, predictions, burn=0)
    assert result['matched_event_targets']['q']['joint_success'] == 1


def test_target_coverage_exposes_clipped_tail():
    arrays = dict(t=np.arange(2)*.1, button=np.array([3, 3]), valid=np.ones(2, bool),
        aim_valid=np.array([True, False]), aim_mode=np.array([2, 2]), clipped=np.array([False, True]),
        screen_x=np.array([5, 5]), screen_y=np.array([3, 3]))
    predictions = dict(button=np.array([3, 3]), probabilities=np.eye(9)[[3, 3]],
        actual_grid=np.full(2, 3*96+5), teacher_grid=np.full(2, 3*96+5))
    result = sequence_metrics(arrays, predictions, burn=0)
    coverage = result['target_coverage_by_button']['q']
    assert coverage == dict(events=2, required=2, eligible=1, clipped=1, coverage=.5)
