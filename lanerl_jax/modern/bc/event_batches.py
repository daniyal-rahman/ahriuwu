"""Load v2 fixtures and form recurrent windows without crossing perspectives."""
from pathlib import Path
import json

import numpy as np

from .event_dataset import CONTRACT
from .event_loss import INPUTS
from .events import smooth_cast_targets


SCALARS = ('own_id', 'enemy_id', 'patch')
LABELS = ('button', 'aim_mode', 'aim_valid', 'target_row', 'screen_x', 'screen_y', 'angle',
          'co_move_source', 'co_move_valid', 'co_move_x', 'co_move_y', 'co_move_delta')
AUXILIARIES = ('own_future', 'own_future_valid', 'enemy_future', 'enemy_future_valid',
               'own_hp_future', 'own_hp_future_valid')


def load_event_games(root, split=None, smoothing=(1.,), champion=None):
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    if not manifest.get('passed') or manifest.get('contract') != CONTRACT:
        raise ValueError('v2 examples require a completed versioned manifest')
    games = []
    for row in manifest['games']:
        if split is not None and row['split'] != split:
            continue
        if champion is not None and row['champion'] != champion:
            continue
        prefix = root / 'games' / row['game']
        metadata = json.loads(Path(str(prefix) + '.json').read_text())
        if metadata['contract'] != CONTRACT or metadata['base'] != row['base']:
            raise ValueError('v2 perspective contract/identity mismatch')
        with np.load(str(prefix) + '.npz') as data:
            arrays = {k: data[k].copy() for k in data.files}
        arrays['button_target'] = smooth_cast_targets(arrays['button'], arrays['valid'], smoothing)
        games.append(dict(row=row, metadata=metadata, arrays=arrays))
    return games


def event_windows(games, window, rng=None):
    result = []
    for gi, g in enumerate(games):
        n = len(g['arrays']['button'])
        offset = 0 if rng is None else int(rng.integers(window))
        result.extend((gi, s) for s in range(offset, n, window))
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
