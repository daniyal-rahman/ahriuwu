"""Stat pipeline, items, runes and economy against Riot match-v5 timelines.

``lanerl_jax/data/modern/oracle/riot_16_9_frames.json.gz`` is an anonymised
extract (``ops/riot_stats_oracle.py``) of the match-v5 timelines of the 147
recorded 16.9 games: per participant-minute Riot's ``championStats``, level,
XP and gold, the inventory rebuilt from item events, the rune page and shards,
and every champion kill with its ``bounty``. Predictions use 16.9 champion
records and item stats (``client_16_9_stats.json``) with the simulator's own
``modern_stat_pipeline``, ``stat_shard_stats``, rune ``stats`` hooks and item
``dynamic_stats`` hooks (dynamic stacks at their initial state).

Riot truncates stats to integers and reports attack speed as 100 × (1 + bonus
AS); ability haste and flat armor penetration are always 0 in the timeline,
so they are not checked. Thresholds sit below the measured exact-match rates
(2026-10-01); the misses are dynamic effects the snapshot cannot see (champion
passives, stacking items and runes, support-item quest upgrades), and the
tests below assert that their sizes match the simulator's implementations.
"""
import gzip
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

ORACLE = Path(__file__).resolve().parents[2] / "data" / "modern" / "oracle"


@lru_cache(maxsize=1)
def table():
    path = ORACLE / "riot_16_9_frames.json.gz"
    if not path.exists():
        pytest.skip("Riot oracle not present")
    return json.load(gzip.open(path))


@lru_cache(maxsize=4)
def prediction(runes=True, n_matches=None):
    from ops.riot_stats_oracle import predict
    t = table()
    if n_matches is not None:
        keep = {m["match"] for m in t["matches"][:n_matches]}
        t = dict(t, frames=[r for r in t["frames"] if r[0] in keep])
    client = json.loads((ORACLE / "client_16_9_stats.json").read_text())
    return predict(t, client, runes=runes)


def exact(out, stat, mask=None):
    o = out["obs"][:, out["stat_index"][stat]]
    hit = o == np.floor(out["pred"][stat] + 1e-3)
    return hit.mean() if mask is None else hit[mask].mean()


def test_xp_table_matches_every_participant_minute():
    from lanerl_jax.sim import modern_economy as E
    t = table()
    f = {k: i for i, k in enumerate(t["fields"])}
    xp = np.asarray([r[f["xp"]] for r in t["frames"]], float)
    lv = np.asarray([r[f["level"]] for r in t["frames"]])
    ok = (lv == np.asarray(E.level_for_xp(xp, 18))) | ((lv > 18) & (lv == np.asarray(E.level_for_xp(xp, 20))))
    assert len(lv) > 40000 and ok.mean() == 1.0


def test_first_death_bounty_is_base_gold_by_level():
    """A champion with no takedowns and < 2000 farmed gold has bounty 0 (the
    100-point positive buffer), so its first death pays base[V] (+100 first blood)."""
    from collections import Counter, defaultdict
    from lanerl_jax.sim import modern_economy as E
    t = table()
    f = {k: i for i, k in enumerate(t["fields"])}
    frames = defaultdict(list)
    for r in t["frames"]:
        frames[(r[f["match"]], r[f["pid"]])].append(r)
    by_match = defaultdict(list)
    for k in t["kills"]:
        by_match[k[0]].append(k)
    ok = n = 0
    for m, ks in by_match.items():
        took, died, first = Counter(), Counter(), True
        for _, ts, killer, victim, assists, bounty, _, _ in sorted(ks, key=lambda k: k[1]):
            if killer == 0:
                died[victim] += 1
                continue
            before = [r for r in frames[(m, victim)] if r[f["t_ms"]] <= ts]
            if before and took[victim] == 0 and died[victim] == 0 and before[-1][f["total_gold"]] < 2500:
                want = float(E.base_kill_gold(before[-1][f["level"]])) + (100.0 if first else 0.0)
                ok, n = ok + (bounty == want), n + 1
            first = False
            took[killer] += 1
            died[victim] += 1
            for a in assists:
                took[a] += 1
    assert n > 400 and ok / n > 0.97


# Measured 2026-10-01 over 42,000 participant-minutes: HP .582, AD .663, AP .737,
# armor .740, MR .751, MS .509, AS .632, magic pen .956, %armor pen .986,
# life steal .917, omnivamp .949, tenacity .816.
THRESHOLDS = {"healthMax": 0.55, "attackDamage": 0.63, "abilityPower": 0.70, "armor": 0.70,
              "magicResist": 0.72, "movementSpeed": 0.48, "attackSpeed": 0.60, "magicPen": 0.93,
              "armorPenPercent": 0.97, "lifesteal": 0.89, "omnivamp": 0.92, "ccReduction": 0.78}


@pytest.mark.parametrize("stat", sorted(THRESHOLDS))
def test_champion_stats_exact_match_rate(stat):
    assert exact(prediction(), stat) > THRESHOLDS[stat]


def test_non_support_max_hp_and_jungle():
    """Supports' quest-upgraded items never appear as purchases; without them max
    HP is exact far more often, and junglers (no HP-stacking lane runes) almost always."""
    out = prediction()
    f = out["fields"]
    pos = np.asarray([r[f["position"]] for r in out["rows"]])
    assert exact(out, "healthMax", pos != "UTILITY") > 0.68
    assert exact(out, "healthMax", pos == "JUNGLE") > 0.9


def test_runes_improve_the_match():
    """Rune stat hooks (Celerity, Alacrity base, Absolute Focus, Gathering Storm,
    Conditioning, Transcendence ...) move predictions toward Riot's values."""
    on, off = prediction(True, 40), prediction(False, 40)
    for stat, gain in (("movementSpeed", 0.05), ("attackDamage", 0.02), ("armor", 0.005)):
        assert exact(on, stat) > exact(off, stat) + gain


def test_dynamic_max_hp_residuals_have_the_implemented_sizes():
    """Remaining max-HP gaps on rune holders come in the sizes the simulator
    implements: Biscuit Delivery +30 per biscuit eaten (30/60/90), Legend:
    Bloodline +85, Overgrowth +3 per 8 deaths."""
    out = prediction()
    f = out["fields"]
    o = out["obs"][:, out["stat_index"]["healthMax"]]
    d = o - np.floor(out["pred"]["healthMax"] + 1e-3)
    runes = [set(r[f["runes"]]) for r in out["rows"]]
    pos = np.asarray([r[f["position"]] for r in out["rows"]])
    clean = lambda r: not (r & {8437, 8451, 9103, 8473})          # no other HP runes
    biscuit = np.asarray([8345 in r and clean(r - {8345}) for r in runes]) & (pos != "UTILITY")
    assert biscuit.sum() > 500
    assert np.mean(np.isin(np.round(d[biscuit]), [0, 30, 60, 90])) > 0.75
    blood = np.asarray([9103 in r and not (r & {8345, 8437, 8451}) for r in runes]) & (pos != "UTILITY")
    assert blood.sum() > 300
    assert np.mean(np.isin(np.round(d[blood]), [0, 85])) > 0.7
    grow = np.asarray([8451 in r and not (r & {8345, 8437, 9103}) for r in runes]) & (pos != "UTILITY")
    early = grow & (np.asarray([r[f["t_ms"]] for r in out["rows"]]) <= 600_000)
    assert early.sum() > 300
    assert np.mean(np.round(d[early]) % 3 == 0) > 0.7
