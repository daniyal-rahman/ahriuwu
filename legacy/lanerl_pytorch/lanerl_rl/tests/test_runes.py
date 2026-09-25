"""Attack damage comes from the SERVER, and nothing re-derives it.

There used to be a `constants.garen_attack_damage(level)` that recomputed the
champion's AD in Python. It was wrong three times in a row, each time by
whatever part of the server had been forgotten:

    57.88   base + level curve            (no rune page)
    73.14   + the 30-rune page            (no mastery page)
    78.14   what the server actually says

Every wrong value fed `E_AA_KILLABLE = sigmoid((ad - minion_hp)/10)`, the
last-hit signal, so the agent was told that killable minions were not
killable -- while the scripted bot read `Stats.AttackDamage.Total` off the C#
side and last-hit correctly.

The rune and mastery pages are fixed in the config; the control channel emits
the resulting total. These tests pin that arrangement: the function is gone,
the wire carries the value, and the value matches what the config implies.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from lanerl_rl import constants as C

_REPO = Path(__file__).resolve().parents[2]
CFG = _REPO / "lanerl/cfg/garen1v1.json"
FRAMES = Path(__file__).resolve().parent / "data" / "frames_v2.jsonl"


def test_there_is_no_python_attack_damage_derivation():
    """The regression guard. Re-adding one re-opens a bug we hit three times."""
    assert not hasattr(C, "garen_attack_damage"), (
        "attack damage must be read from Unit.ad, not recomputed: the server "
        "applies runes and masteries that any Python copy will drift from"
    )
    assert not hasattr(C, "RUNE_FLAT_AD")


@pytest.mark.skipif(not FRAMES.exists(), reason="frame fixture not present")
def test_the_wire_carries_attack_damage_on_every_champion_sample():
    n = miss = 0
    for line in FRAMES.read_text().splitlines():
        if not line.strip():
            continue
        for u in json.loads(line)["u"]:
            if u.get("k") == "Champion":
                n += 1
                miss += u.get("ad") is None
    assert n > 0
    assert miss == 0, f"{miss}/{n} champion samples carry no 'ad'"


@pytest.mark.skipif(not FRAMES.exists(), reason="frame fixture not present")
def test_the_observation_reports_the_servers_attack_damage_not_a_guess():
    from lanerl_rl.frame import decode_frame
    from lanerl_rl.obs import ObservationBuilder

    rows = [json.loads(l) for l in FRAMES.read_text().splitlines() if l.strip()]
    b = ObservationBuilder(C.TEAM_BLUE)
    for r in rows[:40]:
        b.build(decode_frame(r))
    wire = next(
        u["ad"] for u in rows[39]["u"]
        if u.get("k") == "Champion" and u.get("tm") == C.TEAM_BLUE
    )
    assert b._aa_damage == pytest.approx(float(wire))
    # and it is materially above the old base-plus-curve figure, which is the
    # whole reason this file exists
    assert b._aa_damage > C.GAREN_BASE_AD + 10.0


@pytest.mark.skipif(not FRAMES.exists(), reason="frame fixture not present")
def test_a_missing_ad_yields_no_potential_rather_than_a_guess(caplog):
    """Absence must be loud and inert, never silently substituted."""
    from lanerl_rl.frame import decode_frame
    from lanerl_rl.obs import ObservationBuilder

    raw = json.loads(FRAMES.read_text().splitlines()[0])
    for u in raw["u"]:
        u.pop("ad", None)
    b = ObservationBuilder(C.TEAM_BLUE)
    with caplog.at_level("ERROR"):
        b.build(decode_frame(raw))
    assert b._aa_damage == 0.0
    assert any("no 'ad'" in r.message for r in caplog.records)


def test_the_config_still_grants_the_rune_page_the_numbers_assume():
    """If the page is removed, AD changes and every anchor number moves."""
    cfg = json.loads(CFG.read_text())
    for p in cfg["players"]:
        assert len(p.get("runes") or {}) == 30, p.get("name")
