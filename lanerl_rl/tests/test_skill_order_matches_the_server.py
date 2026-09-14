"""The Python skill order must equal the SERVER's, which is what runs.

This fact lived in four places that disagreed:

    LanerlConfig.SkillOrder          C#, and the only one that actually runs
    lanerl_bot.build.GAREN_SKILL_ORDER   Python, its own copy, Q first
    obs.AbilityBook.MIN_LEVEL        Python, a fourth order again (Q@1 W@2 E@3)
    a comment in LanerlControl.cs    prose, stale the moment the array changed

Three are now derived from ``constants.GAREN_SKILL_ORDER``. The fourth is in
another language and cannot be imported, so it is PARSED and compared -- the
same trick ``lanerl_rl.audit`` uses to keep the wire schema honest against the
C# emitter.

Why this matters more than tidiness: the action mask is built from the ranks
the observation believes in. If those disagree with the server, the mask
forbids a spell the champion HAS and offers one it does not -- and casting an
unlearned spell is not a no-op, because nothing in ``Spell.Cast`` checks the
level. It grants the effect anyway. That exact bug has already happened here.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from lanerl_rl import constants as C
from lanerl_rl.obs import AbilityBook

VENDOR_ENV = "LANERL_VENDOR_DIR"
DEFAULT_VENDOR = Path("/srv/nfs/projects/lanerl-vendor")


def _config_cs() -> Path | None:
    root = Path(os.environ.get(VENDOR_ENV, DEFAULT_VENDOR))
    p = root / "LoLServer" / "GameServerLib" / "Lanerl" / "LanerlConfig.cs"
    return p if p.exists() else None


def test_the_python_order_is_internally_consistent():
    """build.py and AbilityBook must both come from the one constant."""
    from lanerl_bot.build import GAREN_SKILL_ORDER as bot_order

    assert tuple(bot_order) == tuple(C.GAREN_SKILL_ORDER), (
        "lanerl_bot.build holds its own copy of the skill order again"
    )
    # MIN_LEVEL[slot] is the first champion level at which `slot` has a point.
    for slot in range(4):
        levels = [i + 1 for i, s in enumerate(C.GAREN_SKILL_ORDER) if s == slot]
        assert AbilityBook.MIN_LEVEL[slot] == levels[0], (
            f"slot {slot}: MIN_LEVEL says {AbilityBook.MIN_LEVEL[slot]}, "
            f"the order says {levels[0]}"
        )


def test_the_order_is_well_formed():
    o = C.GAREN_SKILL_ORDER
    assert len(o) == C.MAX_LEVEL, f"{len(o)} entries for {C.MAX_LEVEL} levels"
    counts = {s: o.count(s) for s in range(4)}
    assert counts[0] == counts[1] == counts[2] == 5, f"QWE need 5 ranks each: {counts}"
    assert counts[3] == 3, f"R has 3 ranks: {counts}"
    assert [i + 1 for i, s in enumerate(o) if s == 3] == [6, 11, 16], "R is 6/11/16"
    assert o[0] == 2, "E first -- it is the farming and trading spell"


def test_the_server_agrees():
    """Parsed from LanerlConfig.cs, because it is the copy that runs."""
    cs = _config_cs()
    if cs is None:
        pytest.skip(f"vendored server not present (set {VENDOR_ENV} to check)")
    m = re.search(r"SkillOrder\s*=\s*\{([^}]*)\}", cs.read_text())
    assert m, "could not find SkillOrder in LanerlConfig.cs -- the parser needs updating"
    server = tuple(int(x) for x in re.findall(r"-?\d+", m.group(1)))
    assert server == tuple(C.GAREN_SKILL_ORDER), (
        f"the server levels {server} and Python believes {tuple(C.GAREN_SKILL_ORDER)}. "
        f"The server wins at runtime, so the observation's spell ranks -- and the "
        f"action mask built from them -- are wrong until these agree."
    )
