"""Replay fidelity oracle: 16.9 (= 26.9) client-memory replays vs the 26.19 sim.

Runs ``ops/modern/replay_fidelity.py`` extraction on one smoke-test game (~5 s) and checks respawn, death time,
ambient gold while dead and Homeguard speed against the simulator. Skipped when the dataset is not mounted.
Full-corpus numbers: docs/modern/REPLAY_FIDELITY.md.
"""
import statistics
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from lanerl_jax.modern import economy as E
from lanerl_jax.modern.world import config as W
from ops.modern import replay_fidelity as T

GAME = Path("/mnt/nfs/datasets/lol_replays_16_9_772_smoketest/NA1_5552036294")


@lru_cache(maxsize=1)
def game():
    if not (GAME / "raw_mem.json").exists():
        pytest.skip("replay smoke-test dataset not mounted")
    return T.extract_game(str(GAME))


def heroes():
    return game()["heroes"].values()


def test_fountain_matches_sim():
    assert T.FOUNTAIN["blue"] == W.FOUNTAINS[0] and T.FOUNTAIN["red"] == W.FOUNTAINS[1]


def test_respawn_full_hp_at_fountain_centre():
    rs = [r for h in heroes() for r in h["respawns"]]
    assert len(rs) > 30
    assert np.mean([abs(r["hp"] - r["hp_max"]) <= 0.15 for r in rs]) > 0.9
    assert np.mean([r["dist_fountain"] <= 1.0 for r in rs]) > 0.9


def test_death_time_matches_sim():
    err = []
    for h in heroes():
        lv = {round(d["gt_death"], 3): d["level"] for d in h["dead_gold"]}
        for r in h["respawns"]:
            t_d = r["gt"] - r["dead_s"]
            if round(t_d, 3) in lv:
                err.append(r["dead_s"] - float(E.death_time(lv[round(t_d, 3)], t_d)))
    assert len(err) > 25
    assert np.mean(np.abs(err) <= 0.15) > 0.9


def test_ambient_gold_while_dead():
    rate = E._const("ai_AmbientGoldAmount") / E._const("ai_AmbientGoldInterval")
    obs = [d["delta"] / (d["t1"] - d["t0"]) for h in heroes() for d in h["dead_gold"] if d["big"] == 0]
    assert len(obs) > 20
    assert abs(statistics.median(obs) - rate) < 0.03


def test_homeguard_floor_is_bonus_percent_ms_before_soft_cap():
    """Before 14:00 the post-decay Homeguard speed is soft_cap(raw * (1 + 0.40))."""
    rows = []
    for h in heroes():
        for e in h["exits"]:
            if e["kind"] not in ("recall", "respawn") or e["gt"] >= E.HOMEGUARD_SWITCH:
                continue
            v_hg = T.plateau(e["series"], 4.5, 8.0)
            v0 = v_hg and T.post_plateau(e["series"], v_hg)
            if v0 is None:
                continue
            hf = float(E.homeguard_bonus_ms(e["gt"], 10.0))
            rows.append((v_hg, T.soft_cap(T.inv_soft_cap(v0) * (1 + hf))))
    assert len(rows) >= 10
    assert np.median([abs(o - p) for o, p in rows]) < 4.0
