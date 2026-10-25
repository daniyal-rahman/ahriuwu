from types import SimpleNamespace

import numpy as np

from lanerl_jax.modern.bc.event_inputs import previous_received_events, visible_cast_ages, encode_ages
from lanerl_jax.modern.bc.events import BUTTON_INDEX as B
from lanerl_jax.modern.bc.replay import Action


def test_history_uses_observation_time_and_stable_target_identity():
    t = np.array([.0, .1, .2, .3])
    ids = np.full((4, 32), -1); ids[1, 2], ids[2, 5] = 99, 99
    actions = [Action(.05, 'attack', target=99), Action(.25, 'cast', slot=0)]
    f, target = previous_received_events(t, actions, np.zeros((4, 2)), ids,
                                          np.array([1., 0.]), np.array([0., 1.]))
    assert np.isnan(f[0]).all()
    assert f[1, B['attack']] == f[2, B['attack']] == 1
    assert target[1, 2] == target[2, 5] == 1
    assert f[2, B['q']] == 0 and f[3, B['q']] == 1
    assert encode_ages(140.) > encode_ages(30.)


def test_hidden_enemy_cast_is_not_revealed_when_caster_later_appears():
    units = [SimpleNamespace(id=1, team=0), SimpleNamespace(id=2, team=1)]
    g = SimpleNamespace(t=np.array([0., .1, .2]), me=1, units=units,
                        index=lambda uid: uid-1, champion_casts=[(.05, 2, 0), (.15, 2, 1)])
    selected = np.full((3, 32), -1); selected[:, 0] = 1
    visible = np.array([[True, False], [True, True], [True, True]])
    result = visible_cast_ages(g, selected, visible, np.arange(3))
    assert np.isnan(result[:, 0, 0]).all()
    assert np.isnan(result[:2, 0, 1]).all() and np.isfinite(result[2, 0, 1])
