"""Optional replay future targets; observation-relative horizons, BC.md section15."""
import numpy as np

from ..obs import NORM_XY, NORM_DIST


def future_targets(arrays, max_step=1500.):
    """Mask entire horizons at gaps, deaths, scope loss or large position jumps."""
    t = np.asarray(arrays['t'])
    own = np.asarray(arrays['self'])
    xy = own[:, :2].astype(np.float64) * NORM_XY
    enemy = xy + np.asarray(arrays['entities'][:, 0, 1:3], np.float64) * NORM_DIST
    ids = arrays['unit_ids'][:, 0]
    cadence = np.r_[True, np.isclose(np.diff(t), .1, atol=1e-5, rtol=0)]
    alive = np.isfinite(own[:, 10]) & (own[:, 10] == 0)
    good = arrays['valid'] & alive & np.isfinite(xy).all(-1) & cadence
    good &= np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=-1) <= max_step]
    visible = (~arrays['pad'][:, 0] & (ids >= 0) & np.isfinite(enemy).all(-1)
               & np.isfinite(arrays['entities'][:, 0, 3]) & (arrays['entities'][:, 0, 3] > 0))
    visible &= np.r_[True, ids[1:] == ids[:-1]]
    visible &= np.r_[True, np.linalg.norm(np.diff(enemy, axis=0), axis=-1) <= max_step]

    def horizon(mask, steps):
        result = np.zeros(len(t), bool)
        bad = np.r_[0, np.cumsum(~mask)]
        n = len(t)-steps
        if n > 0:
            result[:n] = (bad[steps+1:]-bad[:n]) == 0
        return result

    def position(points, steps, mask):
        valid = horizon(mask, steps)
        target = np.zeros(len(t), np.int32)
        n = max(len(t)-steps, 0)
        delta = (points[steps:]-xy[:n])/3000.
        in_grid = np.isfinite(delta).all(-1) & (np.abs(delta) < 1).all(-1)
        valid[:n] &= in_grid
        # Coarse lane-frame square, 24x14, centered on CURRENT own position.
        cells = np.floor((np.nan_to_num(delta, nan=0., posinf=0., neginf=0.)+1)*np.array([12, 7])).astype(np.int32)
        target[:n] = np.clip(cells[:, 1], 0, 13)*24 + np.clip(cells[:, 0], 0, 23)
        return target, valid

    own_position, own_valid = position(xy, 20, good)
    enemy_position, enemy_valid = position(enemy, 10, good & visible)
    hp_valid = horizon(good & np.isfinite(own[:, 2]) & (own[:, 2] >= 0) & (own[:, 2] <= 1), 30)
    hp = np.zeros(len(t), np.float32)
    if len(t) > 30:
        hp[:-30] = np.nan_to_num(own[30:, 2])
    return dict(own_future=own_position, own_future_valid=own_valid,
                enemy_future=enemy_position, enemy_future_valid=enemy_valid,
                own_hp_future=hp, own_hp_future_valid=hp_valid)
