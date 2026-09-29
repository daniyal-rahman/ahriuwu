"""Frozen evaluation must not mix first games with automatic reset tails."""
from types import SimpleNamespace
import numpy as np
from lanerl_jax.train.paired_vec_train import first_episode_rows


def test_first_episode_rows_ignore_resets_and_previously_finished_envs():
    done = np.zeros((4, 2, 2), bool)
    done[1, 0] = True
    done[3, 0] = True
    done[2, 1] = True
    values = np.arange(16).reshape(4, 2, 2)
    tr = SimpleNamespace(done_full=done, cs=values, gold=values, xp=values)
    seen = np.array([False, True])
    rows = first_episode_rows(tr, seen, 'mirror')
    assert len(rows) == 1
    assert rows[0]['env'] == 0
    assert rows[0]['cs'] == values[1, 0].tolist()
    assert seen.all()
    assert first_episode_rows(tr, seen, 'mirror') == []
