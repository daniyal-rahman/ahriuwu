import numpy as np
import pytest

from lanerl_jax.modern.bc.event_batches import event_batch, load_event_games, LABELS
from lanerl_jax.modern.bc.event_loss import INPUTS


def test_window_masks_burn_and_tail_without_broadcasting_future_frames():
    a = {k: np.arange(5, dtype=np.int32) for k in INPUTS + LABELS}
    a.update(own_id=np.asarray(8), enemy_id=np.asarray(9), patch=np.asarray(0),
             pad=np.zeros((5, 32), bool), valid=np.ones(5, bool), button_target=np.eye(9)[[0]*5])
    a['valid'][4] = False
    batch = event_batch([dict(arrays=a)], [(0, 2)], window=4, burn=1)
    np.testing.assert_array_equal(batch['weight'], [[0., 1., 0., 0.]])
    np.testing.assert_array_equal(batch['own_id'], [[8, 8, 8, 0]])
    assert batch['pad'][0, 3].all()
    assert not batch['pad'][0, :3].any()
    assert batch['button_target'][0, 3].sum() == 0


def test_incomplete_fixture_manifest_is_not_loadable(tmp_path):
    (tmp_path/'manifest.json').write_text('{"passed": false}')
    with pytest.raises(ValueError, match='completed'):
        load_event_games(tmp_path)
