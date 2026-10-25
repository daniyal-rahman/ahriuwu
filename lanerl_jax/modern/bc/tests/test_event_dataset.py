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
