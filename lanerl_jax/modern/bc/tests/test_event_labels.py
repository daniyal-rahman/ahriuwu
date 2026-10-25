import numpy as np

from lanerl_jax.modern.bc.event_labels import label_events, UNIT_TARGET
from lanerl_jax.modern.bc.events import BUTTON_INDEX as B
from lanerl_jax.modern.bc.replay import Action


def labels(actions, ids=None):
    n = 8
    return label_events(np.arange(n) * .1, actions, np.zeros((n, 2)),
                        np.full((n, 32), -1) if ids is None else ids,
                        np.array([1., 0.]), np.array([0., 1.]), np.ones(n, bool), me=1)


def test_missing_cast_target_keeps_button_and_only_masks_aim():
    result = labels([Action(.2, 'cast', target=99, slot=0, mode='unit')])
    assert result.button[2] == B['q'] and result.valid[2]
    assert result.aim_mode[2] == UNIT_TARGET and not result.aim_valid[2]


def test_target_pointer_uses_delayed_identity_instead_of_slot_stability():
    ids = np.full((8, 32), -1)
    ids[2, 4], ids[3, 7] = 99, 99
    result = labels([Action(.2, 'cast', target=99, slot=0), Action(.3, 'cast', target=99, slot=0)], ids)
    np.testing.assert_array_equal(result.target_row[2:4], [4, 7])
    assert result.aim_valid[2:4].all()


def test_repeated_attack_execution_is_not_repeated_input_and_move_breaks_it():
    result = labels([Action(.1, 'attack', target=99), Action(.2, 'attack', target=99),
                     Action(.3, 'path', point=(300., 0.), n_waypoints=2),
                     Action(.4, 'attack', target=99)])
    np.testing.assert_array_equal(result.button[1:5], [B['attack'], B['continue'], B['move'], B['attack']])
    assert not result.button[5:].any()


def test_cast_priority_self_no_cursor_and_unsupported_not_false_continue():
    result = labels([Action(.2, 'path', point=(300., 0.), n_waypoints=2),
                     Action(.21, 'cast', slot=1, mode='self'), Action(.4, 'cast', slot=6)])
    assert result.button[2] == B['w'] and not result.aim_valid[2]
    assert result.collisions['cast_move_bins'] == 1
    assert not result.valid[4]
