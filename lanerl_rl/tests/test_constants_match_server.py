"""Pin the Python constants to the C# they claim to be read from.

Four constants in :mod:`lanerl_rl.constants` were wrong at once on 2026-09-12 --
``TARGET_RADIUS`` (champion and minion), ``VISION_RADIUS_CHAMPION`` /
``VISION_RADIUS_MINION`` (swapped), and ``GAREN_COOLDOWNS["E"]`` (rank-5 value
at every rank).  Every one of them was a number copied by hand from the server
and then drifted, and every one read as authoritative because the module's
docstring promises the values are server-derived.

Three of the four shared a single cause: reasoning from content ``CharData``
when the CONSTRUCTOR ARGUMENT is what the server actually uses.
``ObjAIBase.cs:106-116`` and ``:128-140`` both take the ctor argument first and
only fall through to CharData when it is <= 0, which never happens for
champions or minions.  So the content files are a decoy: they contain plausible
values that are never read.

This test parses the C# and the ability JSON directly, so the next drift fails
here instead of silently moving the observation.  It is deliberately a pinning
test, not a "roughly equal" one -- these are exact integers on the wire.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from lanerl_rl import constants as C

# Resolve from THIS file. /mnt/nfs exists on both nodes but /srv/nfs only on
# danilogin, so the hardcoded path this used to carry made every drift-guard
# below SKIP silently on desktop -- the node the suite actually runs on. A
# guard that skips where it is needed is not a guard.
VENDOR = Path(__file__).resolve().parents[2].parent / "lanerl-vendor/LoLServer"
GSL = VENDOR / "GameServerLib/GameObjects/AttackableUnits/AI"
CONTENT = VENDOR / "Content/LeagueSandbox-Default/Data/Characters"

pytestmark = pytest.mark.skipif(
    not GSL.exists(), reason="vendored server tree not present"
)


def _base_args(path: Path) -> list:
    """Split the ``: base(...)`` initialiser into top-level positional args.

    The arguments are POSITIONAL, not named, so this cannot key off a label --
    it has to respect nesting, because ``new Vector2()`` sits in the middle of
    the list and its parentheses would otherwise swallow the split.

    ``ObjAIBase``'s signature is
    ``(game, model, name, collisionRadius, position, visionRadius, skinId, ...)``
    so index 3 is the collision radius and index 5 the vision radius.
    """
    src = path.read_text(errors="ignore")
    i = src.index(": base(")
    start = src.index("(", i)
    depth, out, cur = 0, [], ""
    for ch in src[start:]:
        if ch in "([":
            depth += 1
            if depth == 1:
                continue
        elif ch in ")]":
            depth -= 1
            if depth == 0:
                out.append(cur.strip())
                return out
        if depth == 1 and ch == ",":
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    raise AssertionError(f"unbalanced base(...) in {path}")


I_COLLISION, I_VISION = 3, 5


def test_champion_collision_and_vision_radius_match_the_ctor():
    args = _base_args(GSL / "Champion.cs")
    assert float(args[I_COLLISION]) == C.TARGET_RADIUS["champion"], (
        "Champion.cs passes collisionRadius to base(); the server's in-range "
        "test is centre-to-centre plus the TARGET's collision radius "
        "(ObjAIBase.cs:1195). Garen.json PathfindingCollisionRadius is NOT it."
    )
    assert float(args[I_VISION]) == C.VISION_RADIUS_CHAMPION


def test_minion_collision_and_vision_radius_match_the_ctor():
    args = _base_args(GSL / "Minion.cs")
    assert float(args[I_COLLISION]) == C.TARGET_RADIUS["minion"]
    assert float(args[I_VISION]) == C.VISION_RADIUS_MINION


def test_champion_sees_further_than_a_minion():
    """The specific inversion that shipped.

    Both radii being individually plausible is exactly why the swap survived:
    1100 and 1200 are both defensible numbers. The ORDER is the invariant.
    """
    assert C.VISION_RADIUS_CHAMPION > C.VISION_RADIUS_MINION


def test_a_minion_is_a_bigger_target_than_a_champion():
    """Reach for a minion was 150 against the server's 165 -- 15 units short.

    On a last-hit task that gap is the entire margin: the agent was told a
    minion it could legally hit was out of range.
    """
    assert C.TARGET_RADIUS["minion"] > C.TARGET_RADIUS["champion"]
    assert C.AA_RANGE_GAREN + C.TARGET_RADIUS["minion"] == 165.0


@pytest.mark.parametrize("slot", ["Q", "W", "E", "R"])
def test_garen_cooldowns_match_the_ability_json(slot):
    path = VENDOR / f"Content/LeagueSandbox-Default/Spells/Garen{slot}/Garen{slot}.json"
    if not path.exists():
        pytest.skip(f"{path.name} not in this content tree")
    blob = json.loads(path.read_text(errors="ignore"))
    # Values.SpellData, and the numbers are STRINGS ("13.0000"). Reading them
    # with .get("SpellData") off the top level silently returns nothing, which
    # is how this test first passed by skipping itself.
    spell = blob.get("Values", blob).get("SpellData", {})
    ranks = [spell.get(f"Cooldown{i}") for i in range(1, 6)]
    assert all(r is not None for r in ranks), (
        f"Garen{slot}.json has no Cooldown1..5 under Values.SpellData -- the "
        f"content layout changed and this test would silently stop checking"
    )
    ranks = [float(r) for r in ranks]
    got = C.GAREN_COOLDOWNS[slot]
    if slot == "R":
        # R has three real ranks; 4-5 are padding that deliberately does not
        # match the JSON's own 120/120 filler. Champion.LevelUpSpell caps R
        # at rank 3, so the padding is unreachable.
        assert list(got[:3]) == ranks[:3]
    else:
        assert list(got) == ranks, (
            f"{slot} disagrees with Garen{slot}.json. lanerl_bot/build.py reads "
            f"this same file, so a mismatch means the two halves of the stack "
            f"model a different ability."
        )


def test_enemy_cooldown_assumption_is_the_minimum_over_ranks():
    """The safety direction, asserted rather than described.

    We never read the enemy's real spell ranks (that would be a leak), so the
    denominator is an assumption. The one we want is "the enemy is readier than
    it is" -- the SHORTEST cooldown, so the estimate decays fastest and reports
    the ability back up early. Rank 1 is the LONGEST for W (24 vs 20) and R
    (160 vs 80), so the old rank-1 choice did precisely the opposite: it told
    the agent the enemy was still on cooldown, and the agent walked into it.
    """
    for i, slot in enumerate(C.SPELL_SLOTS):
        assert C.ENEMY_COOLDOWN_ASSUMED[i] == min(C.GAREN_COOLDOWNS[slot]), slot
        assert C.ENEMY_COOLDOWN_ASSUMED[i] <= C.GAREN_COOLDOWNS[slot][0]


def test_the_decision_rate_has_exactly_one_definition():
    """STEP_TICKS was duplicated into three call sites and drifted in two.

    ``VecLaneEnv`` had a hardcoded 4 and ``ControlBackend`` defaulted to 4,
    both against ``C.STEP_TICKS = 2``, so a caller who omitted the argument
    silently halved the decision rate -- and every horizon derived from it,
    including the shaping gamma, doubled.
    """
    from lanerl_train.vec import ServerLaunchSpec

    # Pins the LIVE launcher. This used to pin ControlBackend, which was the
    # single-env backend in lanerl_rl/env.py -- deleted 2026-09-12 as a second,
    # untouched-by-production implementation of the same wiring. Pinning a dead
    # twin is how the two drifted to different step_ticks in the first place.
    assert ServerLaunchSpec().step_ticks == C.STEP_TICKS
    assert C.DECISION_HZ == pytest.approx(60.0 / C.STEP_TICKS)
