"""Replay-only hold, received countdowns, observation lag and the self-column map."""

import numpy as np
import pytest

from lanerl_jax.modern import champions as K
from lanerl_jax.modern import obs as OB
from lanerl_jax.modern.bc import convert as C


def test_self_columns_match_the_builder_layout():
    col = C._self_columns("modern-world-v1")
    assert col["ad"] == 10 and col["dead"] == 14 and col["own_Garen"] == 16
    assert col["enemy_Garen"] == 16 + len(K.KITS) and col["trinket"] == OB.MODERN_WORLD_SELF_DIM - 1


def test_lagged_pairs_actions_with_older_observations():
    T = 6
    base = C.BCGame(entities=np.arange(T, dtype=np.float32)[:, None, None] * np.ones((T, 32, 20), np.float32),
                    entity_pad_mask=np.zeros((T, 32), bool), self_vec=np.zeros((T, 56), np.float32),
                    global_vec=np.zeros((T, 6), np.float32), champion_stats=np.zeros((T, 2, 11), np.float32),
                    button=np.zeros(T, np.int32), screen_x=np.zeros(T, np.int32), screen_y=np.zeros(T, np.int32),
                    uses_screen=np.zeros(T, np.float32), click_valid=np.zeros(T, bool),
                    click_on_unit=np.zeros(T, bool), label_valid=np.ones(T, bool), held=np.zeros(T, bool),
                        in_scope=np.ones(T, bool),
                    obs_index=np.arange(T, dtype=np.int32), t=np.arange(T) * 0.1, meta={})
    lag = base.lagged(2)
    assert np.array_equal(lag.obs_index, [0, 0, 0, 1, 2, 3]) and lag.entities[4, 0, 0] == 2.0
    assert np.array_equal(lag.label_valid, [False, False, True, True, True, True])


def test_conversion_rejects_mixed_sources_before_reading_a_game(tmp_path):
    from lanerl_jax.modern.bc.sources import ReplayOnlyUnavailable
    with pytest.raises(ReplayOnlyUnavailable, match="not replay-only"):
        C.convert_game(tmp_path / "not-a-game", profile="modern-world-v3")
    assert not list(tmp_path.iterdir())


def test_received_timers_never_use_a_kit_or_future_message():
    t = np.array([9., 10., 10.5, 11., 12., 13., 14., 15., 16.])
    events = [(11., 0, 2., 2., 0), (15., 0, 1., 1., 0), (10., 8, 100., 100., 0)]
    cd, unknown = C.received_spell_state(t, events, [(10., 0), (14., 0)])
    assert np.isnan(cd[:3, 0]).all()  # no backfill from 11s to the 10s cast
    np.testing.assert_allclose(cd[3:6, 0], [2., 1., 0.])
    assert np.isnan(cd[6, 0])       # another cast invalidates the expired timer
    np.testing.assert_allclose(cd[7:, 0], [1., 0.])
    assert np.isnan(cd[:, 1:]).all()  # silent slots and item countdowns supply no ability values
    assert unknown == int(np.isnan(cd).sum())


def test_countdown_at_cast_time_wins_and_corrections_replace_it():
    t = np.array([10., 11., 12., 13.])
    cd, _ = C.received_spell_state(t, [(10., 2, 9., 9., 0), (12., 2, 0., 9., 1)], [(10., 2)])
    np.testing.assert_allclose(cd[:, 2], [9., 8., 0., 0.])


def test_missing_or_invalid_countdown_is_unknown():
    cd, _ = C.received_spell_state([0., 1.], [(0., 0, -1., 1., 0)], [])
    assert np.isnan(cd).all()


def test_dataset_and_training_entrypoints_fail_before_io(tmp_path):
    from lanerl_jax.modern.bc import dataset, train
    from lanerl_jax.modern.bc.sources import ReplayOnlyUnavailable
    for fn in (lambda: dataset.convert(tmp_path, 0, 1),
               lambda: dataset.relabel(tmp_path, 0, 1),
               lambda: dataset.finalize(tmp_path),
               lambda: C.relabel_game(tmp_path, "Garen"),
               lambda: train.load_games(tmp_path, "train", cache=tmp_path / "cache")):
        with pytest.raises(ReplayOnlyUnavailable, match="not replay-only"):
            fn()
    assert not list(tmp_path.iterdir())


def test_finalization_rejects_missing_planned_games(tmp_path):
    import json
    from lanerl_jax.modern.bc.dataset import finalize
    from lanerl_jax.modern.bc.sources import SOURCE_CONTRACT
    (tmp_path / "plan.json").write_text(json.dumps(dict(source_contract=SOURCE_CONTRACT,
        profile=OB.REPLAY_PROFILE, games=[dict(game="missing")])))
    with pytest.raises(ValueError, match="planned games are missing"):
        finalize(tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_cache_rebuilds_when_observation_file_changes(tmp_path):
    import json
    from lanerl_jax.modern.bc.train import _arrays
    from lanerl_jax.modern.bc.sources import SOURCE_CONTRACT
    f = tmp_path / "game.npz"
    f.with_suffix(".json").write_text(json.dumps(dict(source_contract=SOURCE_CONTRACT,
        provenance=dict(obs_profile=OB.REPLAY_PROFILE))))
    cache = tmp_path / "cache"
    cache.mkdir()
    d = dict(entities=np.zeros((1, 32, 20)), entity_pad_mask=np.ones((1, 32), bool),
             self_vec=np.ones((1, OB.self_dim(OB.REPLAY_PROFILE))))
    np.savez(f, **d)
    assert _arrays(f, cache)["self_vec"][0, 0] == 1.
    d["self_vec"] *= 2.
    np.savez(f, **d)
    assert _arrays(f, cache)["self_vec"][0, 0] == 2.
