import math

import pytest

from lanerl_jax.parity.record import (
    LOCAL_MOVE_RADIUS,
    _legal_click_offsets,
    scripted_action,
)


def test_scripted_drive_emits_only_policy_reachable_non_minimap_clicks():
    obs = {
        "u": [
            {"k": "Champion", "tm": 100, "x": 100.0, "y": 200.0},
            {"k": "Champion", "tm": 200, "x": 13900.0, "y": 14000.0},
        ]
    }
    action = scripted_action(obs, 1)
    for side, champion in (("blue", obs["u"][0]), ("red", obs["u"][1])):
        move = action[side]
        assert move["t"] == "move"
        distance = math.hypot(
            move["x"] - champion["x"], move["y"] - champion["y"])
        assert distance <= LOCAL_MOVE_RADIUS + 1e-6
        offset = (move["x"] - champion["x"], move["y"] - champion["y"])
        assert any(
            offset[0] == pytest.approx(candidate[0], abs=1e-9)
            and offset[1] == pytest.approx(candidate[1], abs=1e-9)
            for candidate in _legal_click_offsets(int(champion["tm"])))
