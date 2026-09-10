"""The bot must farm meaningfully better than doing nothing.

This is the test that says the scripted layer is worth having at all: if the bot
does not beat an idle champion by a wide margin over a real ten-minute lane, it is
not a warm start, not an anchor, and not a baseline.
"""
from __future__ import annotations

import pytest

from lanerl_bot import telemetry
from lanerl_bot.tests.conftest import requires_server, run_server

pytestmark = [pytest.mark.slow, requires_server]

WINDOW_MS = 601_000  # ten minutes of game time

BASE_ENV = {
    "LANERL_TOPONLY": "1",       # 1v1 lane: only the top lane spawns
    "LANERL_EXIT_AT": str(WINDOW_MS),
}


def _cs(text: str, name: str) -> int:
    rows = telemetry.parse_cs_lines(text)
    row = telemetry.cs_at(rows, WINDOW_MS, name)
    assert row is not None, f"no CS reading for {name}; server likely died"
    return row.cs


@pytest.fixture(scope="module")
def donothing_cs(artifacts) -> int:
    text = run_server(BASE_ENV | {"LANERL_BOT": "none"},
                      artifacts / "pytest_donothing.log", timeout_s=900)
    return _cs(text, "bluebot")


@pytest.fixture(scope="module")
def bot_cs(artifacts) -> int:
    text = run_server(BASE_ENV | {"LANERL_BOT": "blue", "LANERL_BOT_SEED": "1234"},
                      artifacts / "pytest_bot.log", timeout_s=900)
    return _cs(text, "bluebot")


def test_do_nothing_baseline_farms_nothing(donothing_cs):
    """An undriven champion stands in its fountain for the whole game."""
    assert donothing_cs == 0


def test_bot_beats_the_do_nothing_baseline(bot_cs, donothing_cs):
    assert bot_cs > donothing_cs
    # "meaningfully more" -- a wave is 6-7 minions, so several waves' worth
    assert bot_cs >= 20, f"bot only reached {bot_cs} CS@10"


def test_bot_cs_is_in_the_measured_band(bot_cs):
    """Guard against a silent regression in the bot.

    The measured distribution over 8 seeds is a mean in the mid-30s with a spread
    of roughly 25-45, so the band is wide on purpose: it catches "the bot broke",
    not "the bot got unlucky".
    """
    assert 20 <= bot_cs <= 90, f"CS@10 = {bot_cs}, outside the measured band"


def test_bot_levels_up_and_spends_its_gold(artifacts):
    """Auto-level and auto-buy are wired to the real server paths, not stubs."""
    text = run_server(BASE_ENV | {"LANERL_BOT": "blue"},
                      artifacts / "pytest_bot_progress.log", timeout_s=900)
    rows = telemetry.parse_cs_lines(text)
    blue = [r for r in rows if r.name == "bluebot"]
    red = [r for r in rows if r.name == "redbot"]
    assert blue and red

    # levelling: XP from farmed minions must have carried it past level 1
    assert max(r.lvl for r in blue) >= 5

    # shopping: Doran's Shield is +80 max HP, so max HP must exceed Garen's base
    # 616 before any level-ups could account for it
    assert blue[0].mhp >= 690, f"no starting item bought: mhp={blue[0].mhp}"
    # the idle champion bought nothing and stayed at base HP
    assert red[0].mhp == 616

    # and it spent gold rather than hoarding it. Comparing the two champions'
    # balances directly does not work -- the bot out-earns the idle one by more than
    # it spends -- so compare its balance against what it *would* hold had it bought
    # nothing: the idle champion's passive income plus a conservative floor on its
    # own CS income (the cheapest minion on this server, the caster, gives 10).
    final = blue[-1]
    never_spent = red[-1].gold + final.cs * 10
    assert final.gold < never_spent, (
        f"bot holds {final.gold} but would hold at least {never_spent} unspent"
    )


def test_bot_attaches_to_the_requested_side_only(artifacts):
    text = run_server({"LANERL_TOPONLY": "1", "LANERL_EXIT_AT": "150000",
                       "LANERL_BOT": "blue"},
                      artifacts / "pytest_attach.log", timeout_s=600)
    attaches = [l for l in text.splitlines() if l.startswith("LANERL_BOT_ATTACH")]
    assert len(attaches) == 1
    assert "team=100" in attaches[0]
