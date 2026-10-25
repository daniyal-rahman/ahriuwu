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


def test_overlapping_windows_cover_each_post_burn_label_once():
    from lanerl_jax.modern.bc.event_batches import event_windows
    game = dict(arrays=dict(button=np.zeros(811, np.int32)))
    counts = np.zeros(811, np.int32)
    for _, start in event_windows([game], 256, burn=32):
        counts[start+32:min(start+256, len(counts))] += 1
    assert not counts[:32].any()
    np.testing.assert_array_equal(counts[32:], 1)


def test_loader_rejects_cross_split_matches_before_selecting_train(tmp_path):
    import json
    from lanerl_jax.modern.bc.event_dataset import CONTRACT
    rows = [dict(game='one', base='match', split='train'), dict(game='two', base='match', split='val')]
    (tmp_path/'manifest.json').write_text(json.dumps(dict(passed=True, contract=CONTRACT, games=rows)))
    with pytest.raises(ValueError, match='crossing'):
        load_event_games(tmp_path, split='train')


def test_loader_checks_finalized_artifact_digests(tmp_path):
    import json
    from lanerl_jax.modern.bc.event_dataset import CONTRACT, save_events, artifact_hashes
    from lanerl_jax.modern.bc.sources import SOURCE_CONTRACT
    row = dict(game='one', base='match', split='train', champion='Ahri', role='mid')
    prefix = tmp_path/'games/one'
    save_events(prefix, dict(button=np.array([0]), valid=np.array([True])),
                dict(row, contract=CONTRACT, profile='modern-world-v4', source_contract=SOURCE_CONTRACT), [])
    manifest = dict(passed=True, contract=CONTRACT, games=[row], artifacts={'one': artifact_hashes(prefix)})
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    assert len(load_event_games(tmp_path)) == 1
    with open(str(prefix)+'.npz', 'ab') as f:
        f.write(b'changed')
    with pytest.raises(ValueError, match='integrity'):
        load_event_games(tmp_path)
