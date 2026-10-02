"""Economy rules against real games: the 16.9 replay-memory oracle.

``lanerl_jax/data/modern/oracle/replay_16_9_observations.json.gz`` holds raw
client-memory observations of 145 recorded games (all 10 champions, ~40 Hz),
extracted by ``ops/replay_oracle_extract.py`` without reference to the
simulator. These tests compare ``modern_economy`` against what the client
actually did, so they catch errors in the *spec*, not just in the code.
Patch 16.9 (26.9): no SR economy change to 26.19 (ECONOMY §19 audit).

Each check asserts a population statistic (sampling is 40 Hz, values are
rounded to 0.1, and windows can overlap minion kills), with thresholds set
well inside the gap to the nearest wrong model; the wrong models found while
building them are asserted to fit worse.
"""
import gzip
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from lanerl_jax.sim import modern_economy as E

ORACLE = Path(__file__).resolve().parents[2] / "data" / "modern" / "oracle" / "replay_16_9_observations.json.gz"


@lru_cache(maxsize=1)
def games():
    if not ORACLE.exists():
        pytest.skip("replay oracle not present")
    d = json.load(gzip.open(ORACLE))
    assert d["header"]["patch"] == "16.9"
    return [g for g in d["games"] if g["status"] == "ok"]


def real_deaths(g):
    return sorted((x for x in g["deaths"] if (x.get("dead_duration") or 0) >= 5), key=lambda x: x["gt"])


def test_oracle_coverage():
    assert len(games()) == 145
    assert sum(len(real_deaths(g)) for g in games()) > 5000
    for g in games():
        assert all(v["gold_total"] == E.starting_gold() for v in g["initial_gold"].values())


def test_ambient_gold_phase_and_rate():
    """Lifetime gold before any non-ambient income equals 500 + payments at
    65.0, 65.5, ... (U-E-1); the spec's 65.5 start is one payment short."""
    errs, errs_late = [], []
    for g in games():
        for inc in g["early_gold_total_increments"].values():
            big = [t for t, v in inc if v > 1.15]
            stop = min(big) if big else 150.0
            if stop < 75:
                continue
            seen = [(t, v) for t, v in inc if t < stop - 0.3]
            if not seen:
                continue
            t_last, total = max(t for t, _ in seen), 500 + sum(v for _, v in seen)
            errs.append(total - 500 - float(E.ambient_payments(0.0, t_last)))
            errs_late.append(total - 500 - 1.02 * np.floor((t_last - 65.5) / 0.5 + 1))
    errs, errs_late = np.abs(errs), np.abs(errs_late)
    assert len(errs) > 150
    assert np.mean(errs <= 0.06) > 0.95
    assert np.mean(errs_late <= 0.06) < 0.05


def test_death_timers_by_level_and_game_time():
    """Dead duration = BRW[level] x (1 + TIF(t)); TIF accrues continuously
    from 15:00 and is 0 before (U-E-7); levels 19–20 use the level-18 value
    (U-E-6). The sampled duration runs ~1 sample
    short, so allow [-0.15, +0.1] s."""
    obs, pred, step_pred, t_all = [], [], [], []
    for g in games():
        for x in real_deaths(g):
            if x.get("dead_duration") is None:     # levels 19–20 included: table clamps (U-E-6)
                continue
            obs.append(x["dead_duration"])
            pred.append(float(E.death_time(x["level"], x["gt"])))
            t_all.append(x["gt"])
            # The wiki's per-segment ceil steps (the replaced model).
            f, t = 0.0, x["gt"]
            pts = E.econ()["death_scaling_points"]
            for i, (s, p) in enumerate(pts):
                end = pts[i + 1][0] if i + 1 < len(pts) else 1e9
                f += p * np.ceil(max(min(t, end) - s, 0.0) / 30.0 - 1e-9)
            step_pred.append(E.econ()["death_time_per_level"][min(x["level"], 18) - 1] * (1 + min(f, 0.5)))
    obs, pred, step_pred, t_all = map(np.asarray, (obs, pred, step_pred, t_all))
    err = obs - pred
    ok = (err >= -0.15) & (err <= 0.10)
    assert len(obs) > 5000
    assert np.mean(ok) > 0.93
    early = t_all < 900
    assert np.mean(ok[early]) > 0.97                       # includes 10:00–15:00: no scaling there
    late = (t_all >= 900) & ok                             # revives/passives excluded via ok
    step_err = obs[late] - step_pred[late]
    assert np.median(np.abs(err[late] + 0.025)) < 0.5 * np.median(np.abs(step_err + 0.025))


def _payouts(g, death):
    team = g["meta"]["team"]
    got = [(h, v) for h, t, v in death["window_gold_chunks"]
           if abs(t - death["gt"]) <= 0.06 and team[h] != death["team"]]
    return sorted(got, key=lambda z: -z[1])


def test_first_blood_kill_and_assist_gold():
    """First blood pays base + 100 to the killer; assisters share
    (min(0.5K, 0.5 base) + 0.5 FB) x early(t)."""
    kills, assist_ok, assist_n = 0, 0, 0
    for g in games():
        deaths = real_deaths(g)
        if not deaths:
            continue
        fb = deaths[0]
        pay = _payouts(g, fb)
        if not pay:
            continue
        k = pay[0][1]
        expect = float(E.kill_gold(0.0, fb["level"], True))
        kills += abs(k - expect) <= 1.5
        ast = [v for _, v in pay[1:] if v > 3]
        if ast and abs(k - expect) <= 1.5:
            each, _ = E.assist_gold(k - 100.0, fb["level"], fb["gt"], len(ast), 100.0)
            assist_n += len(ast)
            assist_ok += sum(abs(v - float(each)) <= 2.0 for v in ast)
    assert kills >= 0.8 * len(games())
    assert assist_n > 50 and assist_ok / assist_n > 0.9


def _assist_matches(ast, each):
    """An assister's sample may also carry one ambient payment (+1.0/1.1)."""
    return [-0.6 <= v - each <= 1.7 for v in ast]


def test_assist_gold_cap_on_later_kills():
    """Assist pool = min(0.5 K, 0.5 base) x early(t) for non-first-blood kills,
    including shutdowns (K far above base): the pool stays at half the base."""
    ok, n, shut_ok, shut_n, uncapped_ok = 0, 0, 0, 0, 0
    for g in games():
        for x in real_deaths(g)[1:]:
            pay = _payouts(g, x)
            ast = [v for _, v in pay[1:] if v > 3]
            if not ast:
                continue
            k, lv = pay[0][1], min(x["level"], 18)
            each, _ = E.assist_gold(k, lv, x["gt"], len(ast))
            hits = _assist_matches(ast, float(each))
            ok, n = ok + sum(hits), n + len(hits)
            if k > 1.5 * float(E.base_kill_gold(lv)):
                shut_ok, shut_n = shut_ok + sum(hits), shut_n + len(hits)
                uncapped = 0.5 * k * float(E.early_assist_factor(x["gt"])) / len(ast)
                uncapped_ok += sum(_assist_matches(ast, uncapped))
    assert n > 5000 and ok / n > 0.9
    assert shut_n > 200 and shut_ok / shut_n > 0.85 and uncapped_ok / shut_n < 0.1


def _sim_fountain(hp0, t, homeguard, phase_flat, phase_hg):
    hp, out, events = hp0, [], sorted([(phase_flat + 0.25 * k, "f") for k in range(80)]
                                      + [(phase_hg + 0.5 * k, "h") for k in range(40)])
    j = 0
    for ti in t:
        while j < len(events) and events[j][0] <= ti + 1e-9:
            if events[j][1] == "f":
                hp = min(1.0, hp + float(E.econ()["constants"]["sp_HealthRegenPercent"]))
            elif homeguard:
                hp = 1.0 - (1.0 - hp) * (1.0 - E.HOMEGUARD_FOUNTAIN_HEAL)
            j += 1
        out.append(hp)
    return np.asarray(out)


def test_fountain_regen_after_recall():
    """After a recall (Homeguard active) HP follows +2% max HP / 0.25 s plus
    8% of missing HP / 0.5 s; without the Homeguard heal the fit is far worse."""
    segs = []
    for g in games():
        for s in g["fountain_segments"]:
            ser = s["series"]
            if s["kind"] != "recall" or isinstance(ser["hp_max"], list) or len(ser["hp"]) < 30:
                continue
            m = float(ser["hp_max"])
            if ser["hp"][0] > 0.6 * m:
                continue
            segs.append((np.asarray(ser["dt"][:40]), np.asarray(ser["hp"][:40]) / m))
        if len(segs) >= 120:
            break
    phases = [(pf, ph) for pf in np.linspace(0, 0.24, 7) for ph in np.linspace(0, 0.49, 9)]
    fit = lambda hg: np.median([min(np.mean(np.abs(_sim_fountain(f[0], t, hg, *p) - f)) for p in phases)
                                for t, f in segs])
    with_hg, without = fit(True), fit(False)
    assert len(segs) >= 100
    assert with_hg < 0.015 and without > 3 * with_hg


def test_level_up_raises_current_hp_by_max_gain():
    """On the sample where max HP rises at a level-up, current HP rises by the
    same amount (ai_levelUp_healthGainNetGain 1.0, no missing-HP penalty).

    Wounded champions sometimes gain a little more on that sample (life steal,
    regen, junglers' +6 camp heal on the killing blow); a missing-HP penalty
    would show as a shortfall instead, so assert there is none."""
    diffs, full, penalised = [], [], []
    for g in games():
        f = {k: i for i, k in enumerate(g["levelups"]["fields"])}
        for r in g["levelups"]["rows"]:
            dmax = r[f["hp_max_after"]] - r[f["hp_max_before"]]
            if not r[f["alive"]] or dmax <= 1.0 or r[f["hp_before"]] <= 0:
                continue
            hp, _ = E.level_up_sync(r[f["hp_before"]], r[f["hp_max_before"]], r[f["hp_max_after"]])
            diffs.append(r[f["hp_after"]] - float(hp))
            full.append(r[f["hp_before"]] >= r[f["hp_max_before"]] - 0.05)
            missing = 1.0 - r[f["hp_before"]] / r[f["hp_max_before"]]
            # Wiki "Healing" alternative: the gain shrinks with missing health.
            penalised.append(r[f["hp_after"]] - (r[f["hp_before"]] + dmax * (1.0 - missing))
                             if missing > 0.3 else np.nan)
    diffs, full, penalised = np.asarray(diffs), np.asarray(full), np.asarray(penalised)
    assert len(diffs) > 5000
    assert np.mean(np.abs(diffs[full]) <= 0.15) > 0.95
    assert np.median(np.abs(diffs)) <= 0.1 and np.mean(diffs >= -0.15) > 0.93
    heavy = ~np.isnan(penalised)
    assert heavy.sum() > 1000
    assert np.mean(np.abs(diffs[heavy]) <= 0.15) > 3 * np.mean(np.abs(penalised[heavy]) <= 0.15)


def test_level_up_max_hp_growth_all_champions():
    """Max-HP rise at each level-up = hpPerLevel x (G(L) − G(L−1)) + 10 per
    scaling-HP shard (0, 1 or 2), using the 16.9 champion records of all 169
    champions in the games. Checks the stat-growth curve (DAMAGE §3.2) and the
    shard (10·level, RUNES §2.3) against the client; the linear growth curve
    the legacy port used explains far less."""
    from lanerl_jax.sim.modern_stats import level_growth_sum
    path = ORACLE.parent / "champion_hp_16_9.json"
    champs = json.loads(path.read_text())["champions"]
    explained = linear = n = 0
    for g in games():
        f = {k: i for i, k in enumerate(g["levelups"]["fields"])}
        for r in g["levelups"]["rows"]:
            if r[f["hp_max_probe"]] is None:
                continue
            c = champs[r[f["hero"]].lower()]
            lo, hi = r[f["old_level"]], r[f["new_level"]]
            dmax = r[f["hp_max_probe"]] - r[f["hp_max_before"]]
            rem = dmax - c["hp_per_level"] * (float(level_growth_sum(hi)) - float(level_growth_sum(lo)))
            rem_lin = dmax - c["hp_per_level"] * (hi - lo)
            explained += any(abs(rem - k) <= 0.15 for k in (0.0, 10.0, 20.0))
            linear += any(abs(rem_lin - k) <= 0.15 for k in (0.0, 10.0, 20.0))
            n += 1
    assert n > 15000
    assert explained / n > 0.88 and linear / n < 0.5 * explained / n
