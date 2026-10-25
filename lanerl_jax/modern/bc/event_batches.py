"""Load v2 fixtures and form recurrent windows without crossing perspectives."""
from pathlib import Path
import json

import numpy as np

from .event_dataset import CONTRACT, artifact_hashes
from .sources import require_replay_only
from .event_loss import INPUTS
from .events import smooth_cast_targets


SCALARS = ('own_id', 'enemy_id', 'patch')
LABELS = ('button', 'aim_mode', 'aim_valid', 'target_row', 'screen_x', 'screen_y', 'angle',
          'co_move_source', 'co_move_valid', 'co_move_x', 'co_move_y', 'co_move_delta', 'co_move_after', 'co_move_order_valid')
AUXILIARIES = ('own_future', 'own_future_valid', 'enemy_future', 'enemy_future_valid',
               'own_hp_future', 'own_hp_future_valid',
               'next_button', 'next_delay', 'next_valid', 'next_observed')


def load_event_games(root, split=None, smoothing=(1.,), champion=None):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if not manifest.get('passed') or manifest.get('contract') != CONTRACT:
        raise ValueError('v2 examples require a completed versioned manifest')
    games, splits, names = [], {}, set()
    for row in manifest['games']:
        if row['game'] in names or splits.setdefault(row['base'], row['split']) != row['split']:
            raise ValueError('duplicate perspective or match crossing dataset splits')
        names.add(row['game'])
    for row in manifest['games']:
        if split is not None and row['split'] != split:
            continue
        if champion is not None and row['champion'] != champion:
            continue
        prefix = root / 'games' / row['game']
        expected = manifest.get('artifacts', {}).get(row['game'])
        if expected is None or artifact_hashes(prefix) != expected:
            raise ValueError('v2 artifact integrity differs from finalized manifest')
        metadata = json.loads(Path(str(prefix) + '.json').read_text())
        require_replay_only('v2 event examples', metadata)
        if metadata['contract'] != CONTRACT or any(metadata.get(k) != row[k] for k in ('game', 'base', 'split', 'champion', 'role')):
            raise ValueError('v2 perspective contract/identity mismatch')
        with np.load(str(prefix) + '.npz') as data:
            arrays = {k: data[k].copy() for k in data.files}
        arrays['button_target'] = smooth_cast_targets(arrays['button'], arrays['valid'], smoothing)
        games.append(dict(row=row, metadata=metadata, arrays=arrays))
    return games


def event_windows(games, window, rng=None, burn=0):
    if not 0 <= burn < window:
        raise ValueError('burn must be shorter than window')
    stride = window - burn
    result = []
    for gi, g in enumerate(games):
        n = len(g['arrays']['button'])
        offset = 0 if rng is None else int(rng.integers(stride))
        result.extend((gi, s) for s in range(offset, n, stride))
        if offset:
            result.append((gi, 0))
    return result


def event_batch(games, items, window=256, burn=32):
    if not items or not 0 <= burn < window:
        raise ValueError('batch needs nonempty windows and burn shorter than window')
    first = games[items[0][0]]['arrays']
    keys = INPUTS + LABELS + ('button_target',) + tuple(k for k in AUXILIARIES if k in first)
    batch = {}
    for key in keys:
        shape = first[key].shape if key in SCALARS else first[key].shape[1:]
        dtype = np.float32 if first[key].dtype.kind == 'f' else first[key].dtype
        batch[key] = np.zeros((len(items), window) + shape, dtype=dtype)
    batch['pad'].fill(True)
    batch['weight'] = np.zeros((len(items), window), np.float32)
    for i, (gi, start) in enumerate(items):
        arrays = games[gi]['arrays']
        end = min(start + window, len(arrays['button']))
        length = end - start
        if start < 0 or length <= 0:
            raise ValueError('window outside perspective')
        for key in keys:
            batch[key][i, :length] = arrays[key] if key in SCALARS else arrays[key][start:end]
        batch['weight'][i, :length] = arrays['valid'][start:end]
        batch['weight'][i, :burn] = 0.
    return batch


def ahri_angle_targets(games):
    """Q/E direction ablation on exactly the existing valid point labels."""
    from .event_labels import POINT_TARGET, ANGLE_TARGET
    from .events import BUTTON_INDEX as B
    for game in games:
        if game['row']['champion'] != 'Ahri':
            continue
        arrays = game['arrays']
        selected = (arrays['valid'] & arrays['aim_valid'] &
                    (arrays['aim_mode'] == POINT_TARGET) &
                    np.isin(arrays['button'], [B['q'], B['e']]) & np.isfinite(arrays['angle']))
        arrays['aim_mode'][selected] = ANGLE_TARGET
    return games


def drop_previous_event_history(batch, rng, probability):
    """Hide one whole command-history group per window, including burn-in."""
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError('previous-event dropout must be in [0,1]')
    if probability == 0:
        return batch
    selected = rng.random(batch['self'].shape[0]) < probability
    result = dict(batch)
    result['self'] = batch['self'].copy()
    result['entities'] = batch['entities'].copy()
    result['self'][selected, ..., 28:] = np.nan
    result['entities'][selected, ..., 24:] = np.nan
    return result
