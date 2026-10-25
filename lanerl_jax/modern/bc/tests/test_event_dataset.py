import numpy as np

from lanerl_jax.modern.bc.event_dataset import save_events


def test_event_files_keep_champion_suffix_and_refuse_overwrites(tmp_path):
    import pytest
    for champion in ('Ahri', 'Garen'):
        out = tmp_path / ('NA1-123.' + champion)
        save_events(out, dict(button=np.array([0])), dict(game=out.name), [])
        assert (tmp_path / (out.name + '.npz')).exists()
        assert (tmp_path / (out.name + '.events.json.gz')).exists()
        with pytest.raises(FileExistsError):
            save_events(out, {}, {}, [])


def test_co_movement_missing_delayed_anchor_masks_aim_but_preserves_evidence():
    from types import SimpleNamespace as NS
    from lanerl_jax.modern.bc.event_dataset import co_movement_targets
    actions = []
    for t in (0., .1, .2):
        actions.extend([NS(t=t, kind='cast'), NS(t=t + .01, kind='path', n_waypoints=2, point=(100., 0.))])
    labels = NS(button=np.array([3, 3, 3]), valid=np.array([True, False, True]), source=np.array([0, 2, 4]))
    source, valid, x, y, dt = co_movement_targets(
        actions, np.array([0., .1, .2]), labels,
        np.array([[np.nan, 0.], [0., 0.], [0., 0.]]), np.array([1., 0.]), np.array([0., 1.]))
    np.testing.assert_array_equal(source, [1, 3, 5])
    np.testing.assert_array_equal(valid, [False, False, True])
    np.testing.assert_allclose(dt, .01)
    assert x[0] == y[0] == 0
