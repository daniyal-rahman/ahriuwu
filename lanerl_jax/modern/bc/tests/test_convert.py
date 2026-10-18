"""The converter: cooldown rebuild rules, the lag, the self-column map, and one real game end to end."""
import json
from pathlib import Path

import numpy as np
import pytest

from lanerl_jax.modern import champions as K
from lanerl_jax.modern import obs as OB
from lanerl_jax.modern.actions import MODERN_BUTTONS
from lanerl_jax.modern.bc import convert as C
from lanerl_jax.modern.data.champions import cooldowns

GAME = Path("/mnt/nfs/datasets/rofl_bc_2619_garen_top_native_20261015/games/NA1-5650865612")


def test_spell_state_follows_garen_rules():
    t = np.arange(0.0, 60.0, 0.1)
    q_cd, e_cd, r_cd = cooldowns("Garen", "Q")[0], cooldowns("Garen", "E")[0], cooldowns("Garen", "R")[0]
    casts = [(10.0, 0), (20.0, 2), (21.5, 2), (40.0, 3)]
    cd, conflicts = C.spell_state(t, casts, attack_times=[11.0], rank_at=lambda tc, s: 1, haste_at=lambda tc: 0.0)
    # Q: the window runs to the attack at 11.0, then the cooldown.
    assert cd[105, 0] == 0.0 and cd[110, 0] == pytest.approx(q_cd, abs=0.11)
    # E: a press 1.5 s into the spin ends it; the cooldown starts then.
    assert cd[205, 2] == 0.0 and cd[215, 2] == pytest.approx(e_cd, abs=0.11)
    # R: the cooldown starts after the cast time.
    assert cd[400, 3] == 0.0 and cd[405, 3] == pytest.approx(r_cd - (0.5 - C.G.R_CAST_TIME), abs=0.11)
    assert conflicts == 0
    # Haste shortens the cooldown; a premature press counts as a conflict and still starts the spell.
    cd2, conflicts2 = C.spell_state(t, [(10.0, 1), (12.0, 1)], [], lambda tc, s: 1, lambda tc: 100.0)
    w_cd = cooldowns("Garen", "W")[0]
    assert cd2[101, 1] == pytest.approx(w_cd / 2.0 - 0.1, abs=0.11) and conflicts2 == 1


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


@pytest.mark.skipif(not GAME.exists(), reason="the Garen TOP native export is not mounted")
def test_real_game_round_trip(tmp_path):
    bc = C.convert_game(GAME, window=(300.0, 330.0))
    T = len(bc.t)
    assert T == 300 and bc.entities.shape == (T, OB.N_SLOTS, OB.MODERN_ENTITY_DIM)
    assert bc.self_vec.shape == (T, OB.self_dim(OB.PROFILE)) and bc.global_vec.shape == (T, OB.GLOBAL_DIM)
    assert np.isfinite(bc.entities).all() and np.isfinite(bc.self_vec).all()
    assert bc.button.max() < len(MODERN_BUTTONS) and bc.meta["counts"]["label_valid"] > 0
    lc = bc.meta["label_check"]
    assert lc["attacks_resolved"] >= 0.9 * lc["attacks"] and lc["stops_resolved"] == lc["stops"]
    bc.lagged(1).save(tmp_path)
    saved = np.load(tmp_path / f"{GAME.name}.npz")
    meta = json.loads((tmp_path / f"{GAME.name}.json").read_text())
    assert saved["obs_index"][1] == 0 and meta["observation_lag"]["decisions"] == [1, 1]
    assert meta["provenance"]["cooldown_fields_read"] is False


def test_received_spell_state_uses_the_timers_then_the_kit_table():
    t = np.arange(0.0, 60.0, 0.1)
    # Jax: E sends a 1 s window then its cooldown at the recast; W sends its cooldown; Q sends nothing.
    events = [(20.0, 2, 1.0, 1.0, 0), (21.6, 2, 16.5, 16.5, 0), (30.0, 1, 6.8, 6.8, 0), (30.0, 8, 5.0, 5.0, 0)]
    casts = [(10.0, 0), (20.0, 2), (21.6, 2), (30.0, 1), (45.0, 0)]
    cd, fallback = C.received_spell_state(t, events, casts, "Jax", rank_at=lambda tc, s: 1, haste_at=lambda tc: 0.0)
    assert cd[205, 2] == pytest.approx(0.5, abs=0.11) and cd[220, 2] == pytest.approx(16.1, abs=0.11)
    assert cd[300, 1] == pytest.approx(6.8, abs=0.11) and cd[370, 1] == 0.0 and cd[:300, 1].max() == 0.0
    q_cd = cooldowns("Jax", "Q")[0]
    assert fallback == 2 and cd[101, 0] == pytest.approx(q_cd - 0.1, abs=0.11) and cd[180, 0] == 0.0
    assert cd[451, 0] == pytest.approx(q_cd - 0.1, abs=0.11)
    # An item countdown (slot 8) never touches the ability columns.
    assert cd[:, 3].max() == 0.0
