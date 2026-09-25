"""In-process episode reset: does it actually hand back a clean t=0 game?

These assertions are made against the server's own 10Hz state record, not against
the reset function's return value, so they check what the next episode will really
observe rather than what the reset believes it did.
"""
from __future__ import annotations

import pytest

from lanerl_bot import content, telemetry
from lanerl_bot.tests.conftest import requires_server

pytestmark = [pytest.mark.slow, requires_server]

# Fountain spawns for the two teams on Map1, as recorded by the server itself.
BLUE_FOUNTAIN = (26, 280)
PURPLE_FOUNTAIN = (13927, 14175)
FOUNTAIN_TOLERANCE = 900  # a champion is teleported to the fountain, not pixel-pinned


def _post_reset_ticks(rec_rows: list[dict]) -> list[dict]:
    """Ticks whose clock has just been rewound.

    The record is written at the top of Game.Update and the reset runs just after,
    so the first tick that reads back near zero is the first observation of the new
    episode.
    """
    out = []
    prev_t = None
    for row in rec_rows:
        t = row["t"]
        if prev_t is not None and t < prev_t:
            out.append(row)
        prev_t = t
    return out


def test_reset_reports_are_emitted(reset_run):
    text, _ = reset_run
    reports = telemetry.parse_reset_reports(text)
    assert len(reports) >= 3, f"expected 3 episodes, got {len(reports)}"
    for r in reports:
        assert r.champions_reset == 2
        assert r.wave_timers_reset
        assert r.t_after == 0
        assert r.t_before > 0
        assert r.buildings_restored > 0
        assert r.buildings_unrevivable == 0
        # every structure on the map, both teams
        assert r.buildings_restored >= 24


def test_reset_despawns_the_wave(reset_run):
    """Every minion on the map is gone, and the first reset had real work to do."""
    text, rec = reset_run
    reports = telemetry.parse_reset_reports(text)
    assert any(r.minions_removed > 0 for r in reports), \
        "no reset removed a minion -- the benchmark would be measuring nothing"

    rows = telemetry.load_state_jsonl(rec)
    post = _post_reset_ticks(rows)
    assert len(post) >= 2, "fewer than two resets visible in the recorded state"
    for tick in post:
        assert telemetry.units_of_kind(tick, "LaneMinion") == [], \
            f"minions survived the reset at t={tick['t']}"


def test_reset_rewinds_the_clock(reset_run):
    _, rec = reset_run
    rows = telemetry.load_state_jsonl(rec)
    post = _post_reset_ticks(rows)
    assert post
    for tick in post:
        # one tick of game time may have elapsed before the record was written
        assert tick["t"] <= 200, f"clock not rewound: t={tick['t']}"


def test_reset_restores_champions(reset_run):
    """Full HP, level 1, starting gold, back at the fountain."""
    _, rec = reset_run
    rows = telemetry.load_state_jsonl(rec)
    post = _post_reset_ticks(rows)
    assert post

    for tick in post:
        champs = telemetry.units_of_kind(tick, "Champion")
        assert len(champs) == 2, f"expected 2 champions, saw {len(champs)}"
        for c in champs:
            assert c["hp"] == c["mhp"], f"champion not full HP: {c}"
            assert c["lvl"] == 1, f"champion not level 1: {c}"
            assert c["xp"] == 0, f"champion kept experience: {c}"
            assert c["gold"] == pytest.approx(content.STARTING_GOLD, abs=1), \
                f"champion gold not reset: {c}"
            # passive gold accrues all episode, so this is a real assertion: at the
            # 150s reset point an idle champion is well past 475
            assert c["gold"] < 700

            fountain = BLUE_FOUNTAIN if c["tm"] == 100 else PURPLE_FOUNTAIN
            dx = abs(c["x"] - fountain[0])
            dy = abs(c["y"] - fountain[1])
            assert max(dx, dy) <= FOUNTAIN_TOLERANCE, \
                f"champion not at its fountain: {c} vs {fountain}"


def test_reset_restores_structures(reset_run):
    """Turrets, inhibitors and the nexus are all back at full HP."""
    _, rec = reset_run
    rows = telemetry.load_state_jsonl(rec)
    post = _post_reset_ticks(rows)
    assert post
    for tick in post:
        for kind in ("LaneTurret", "Inhibitor", "Nexus"):
            units = telemetry.units_of_kind(tick, kind)
            assert units, f"no {kind} in the recorded state"
            for u in units:
                assert u["hp"] == u["mhp"], f"{kind} not restored: {u}"


def test_second_episode_replays_the_same_wave_schedule(reset_run):
    """A reset is only useful if the next episode is the same experiment.

    The first wave must arrive at the same game time in every episode, which is what
    rewinding NextSpawnTime alongside the clock buys.
    """
    _, rec = reset_run
    rows = telemetry.load_state_jsonl(rec)

    episodes: list[list[dict]] = [[]]
    prev_t = None
    for row in rows:
        if prev_t is not None and row["t"] < prev_t:
            episodes.append([])
        episodes[-1].append(row)
        prev_t = row["t"]

    first_minion_times = []
    for ep in episodes:
        t = next((r["t"] for r in ep if telemetry.units_of_kind(r, "LaneMinion")), None)
        if t is not None:
            first_minion_times.append(t)

    assert len(first_minion_times) >= 2, "fewer than two episodes saw a wave"
    spread = max(first_minion_times) - min(first_minion_times)
    assert spread <= 1500, f"wave schedule drifted across episodes: {first_minion_times}"
    # and it is the map's real first-wave time, not an artefact
    assert min(first_minion_times) >= content.FIRST_WAVE_MS - 500


def test_reset_is_orders_of_magnitude_cheaper_than_a_process_restart(reset_run):
    """The whole point. A process restart measured ~12s on this machine."""
    text, _ = reset_run
    reports = telemetry.parse_reset_reports(text)
    assert reports
    worst = max(r.duration_ms for r in reports)
    assert worst < 500.0, f"in-process reset took {worst:.1f}ms"
