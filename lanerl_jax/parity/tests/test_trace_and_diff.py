"""The J0 gate: the instrument is tested before anything is measured with it.

A parity harness that mis-parses or under-reports is worse than none, because it
produces green diffs over garbage and the rewrite proceeds on a lie.  So every
test here is of the form "corrupt exactly one thing and prove the differ names
it", plus the negative control that an unmodified trace diffs clean.

The synthetic rows below are written by hand to the grammar in
``LanerlStateDump.Describe``.  ``test_real_server_dump_parses`` is the check
that the hand-written grammar still matches the real thing.
"""
from __future__ import annotations

import pytest

from lanerl_jax.parity.diff import DEFAULT_TOLERANCE, diff_snapshots, diff_traces
from lanerl_jax.parity.trace import (
    PosQ,
    StatQ,
    TraceFormatError,
    load_trace,
    parse_row,
    parse_stream,
)

# One champion, one lane minion, one turret, one non-attackable prop.
CHAMP = (
    "Champion|100|14672,163520|759808/772096|A"          # kind team x,y hp/max alive
    "|2|3|-|-|1|GarenPassive"                             # ObjAIBase block
    "|59269,28197,45219,353280,1024|1|486400|0|0|1"       # Champion stats..skillpoints
    "|1:0|0:-1|1:5120|-1:-1"                              # Q/W/E/R level:cooldown
)
MINION = "LaneMinion|200|18000,170000|477184/477184|A|4|2|-|-|1|"
TURRET = "LaneTurret|100|9184,163536|3379200/3379200|A|0|-1|-|-|1|"
PROP = "LevelProp|12000,90000"


def _log(t_ms: int, rows, h: str = "a1b2c3d4e5f60718") -> str:
    lines = [f"[INFO] LANERL_STATEHASH t={t_ms} n={len(rows)} h={h}"]
    lines += [f"LANERL_STATEROW t={t_ms} {r}" for r in rows]
    lines.append("LANERL_TPS 2580.0 ticks/s  speedup 43.00x  gametime 61.2s")
    return "\n".join(lines) + "\n"


ROWS = [CHAMP, MINION, TURRET, PROP]


def test_every_row_shape_in_Describe_parses():
    champ = parse_row(CHAMP)
    assert champ.kind == "Champion" and champ.team == 100
    assert champ.q_x == 14672 and champ.x == pytest.approx(14672 / PosQ)
    assert champ.hp == pytest.approx(759808 / StatQ)
    assert champ.dead is False
    assert champ.ai is not None and champ.ai.buffs == ("GarenPassive",)
    assert champ.champ is not None
    assert champ.champ.level == 1 and champ.champ.minions_killed == 0
    # E is on a 5-second cooldown; W holds no spell at all.
    assert champ.champ.spells[2] == (1, 5120)
    assert champ.champ.spells[1] == (0, -1)

    minion = parse_row(MINION)
    assert minion.kind == "LaneMinion" and minion.ai is not None
    assert minion.ai.buffs == ()          # empty buff list, not a buff named ""
    assert minion.champ is None

    prop = parse_row(PROP)
    assert prop.kind == "LevelProp" and prop.team is None and prop.ai is None


def test_an_unknown_field_count_raises_rather_than_guessing():
    with pytest.raises(TraceFormatError, match="dump format changed"):
        parse_row("Champion|100|1,2|3/4|A|extra")


def test_a_truncated_snapshot_raises_instead_of_scoring_agreement():
    """The most dangerous failure available: rows dropped, diff comes back clean."""
    log = _log(1000, ROWS)
    kept = [ln for ln in log.splitlines() if "LaneMinion" not in ln]
    with pytest.raises(TraceFormatError, match="STATEHASH said n=4"):
        parse_stream(kept)


def test_interleaved_rows_raise():
    lines = _log(1000, ROWS).splitlines() + [f"LANERL_STATEROW t=1000 {MINION}"]
    lines.insert(-1, "[INFO] LANERL_STATEHASH t=1033 n=1 h=0000000000000001")
    with pytest.raises(TraceFormatError, match="after that snapshot was closed"):
        parse_stream(lines)


def test_identical_traces_diff_clean():
    a = parse_stream(_log(1000, ROWS).splitlines())
    b = parse_stream(_log(1000, ROWS).splitlines())
    d = diff_snapshots(a[0], b[0])
    assert d.clean, d.report()
    assert d.hash_equal is True


def test_equal_content_but_reordered_rows_still_diffs_clean():
    """Rows are content-sorted by the server, so order carries no information.

    A differ that zipped the two row lists would report four bogus diffs here.
    """
    a = parse_stream(_log(1000, ROWS).splitlines())
    b = parse_stream(_log(1000, [TURRET, PROP, CHAMP, MINION], h="ffffffffffffffff").splitlines())
    d = diff_snapshots(a[0], b[0])
    assert d.hash_equal is False          # forced, so the fast path is not taken
    assert d.clean, d.report()


@pytest.mark.parametrize(
    "corrupt,expect",
    [
        (lambda r: r.replace("759808/772096", "700000/772096"), "hp"),
        (lambda r: r.replace("|1|486400|", "|2|486400|"), "level"),
        (lambda r: r.replace("|1:0|", "|1:2048|"), "spell.Q.cooldown"),
        (lambda r: r.replace("|-|-|1|GarenPassive", "|GarenE|-|1|GarenPassive"), "cast_spell"),
        (lambda r: r.replace("|GarenPassive", "|GarenE+GarenPassive"), "buffs"),
        (lambda r: r.replace("|2|3|-|-|", "|2|4|-|-|"), "waypoints"),
        (lambda r: r.replace("772096|A", "772096|D"), "dead"),
    ],
)
def test_one_corrupted_champion_field_is_named(corrupt, expect):
    a = parse_stream(_log(1000, ROWS).splitlines())
    b = parse_stream(_log(1000, [corrupt(CHAMP)] + ROWS[1:], h="ffffffffffffffff").splitlines())
    d = diff_snapshots(a[0], b[0])
    assert not d.clean
    names = [f.name for ed in d.entity_diffs for f in ed.fields]
    assert expect in names, f"expected {expect} in {names}\n{d.report()}"


def test_a_position_shift_inside_quantisation_is_tolerated_and_outside_is_not():
    a = parse_stream(_log(1000, ROWS).splitlines())
    near = parse_stream(_log(1000, [CHAMP.replace("14672,163520", "14673,163520")] + ROWS[1:],
                             h="f" * 16).splitlines())
    assert diff_snapshots(a[0], near[0]).clean
    far = parse_stream(_log(1000, [CHAMP.replace("14672,163520", "14700,163520")] + ROWS[1:],
                            h="f" * 16).splitlines())
    d = diff_snapshots(a[0], far[0])
    assert "pos.x" in [f.name for ed in d.entity_diffs for f in ed.fields], d.report()


def test_a_missing_entity_is_reported_not_dropped():
    """A minion that should have died and did not is the headline failure."""
    a = parse_stream(_log(1000, ROWS).splitlines())
    b = parse_stream(_log(1000, [CHAMP, TURRET, PROP], h="f" * 16).splitlines())
    d = diff_snapshots(a[0], b[0])
    assert len(d.unmatched) == 1
    assert d.unmatched[0].kind == "LaneMinion"
    assert d.unmatched[0].only_in == "left"
    assert "present only in left" in d.report()


def test_an_entity_too_far_to_match_is_unmatched_on_both_sides():
    a = parse_stream(_log(1000, ROWS).splitlines())
    moved = MINION.replace("18000,170000", "60000,170000")
    b = parse_stream(_log(1000, [CHAMP, moved, TURRET, PROP], h="f" * 16).splitlines())
    d = diff_snapshots(a[0], b[0])
    sides = sorted(x.only_in for x in d.unmatched)
    assert sides == ["left", "right"], d.report()


def test_traces_align_on_game_time_not_index():
    """A dropped snapshot must cost one comparison, not shift every later one."""
    left = parse_stream((_log(1000, ROWS) + _log(1033, ROWS) + _log(1066, ROWS)).splitlines())
    right = parse_stream((_log(1000, ROWS) + _log(1066, ROWS)).splitlines())
    diffs = diff_traces(left, right)
    assert [d.t_ms_left for d in diffs] == [1000, 1033, 1066]
    assert diffs[0].clean and diffs[2].clean
    assert diffs[1].time_mismatch


@pytest.mark.skipif(
    not __import__("lanerl_train.paths", fromlist=["paths"]).server_available(),
    reason="vendored server build not available",
)
def test_real_server_dump_parses(tmp_path):
    """The hand-written grammar above must still match what the server emits.

    Marked slow: boots a real server. This is the test that catches
    ``Describe`` growing a field, which silently invalidates every parity
    number until the parser is updated.
    """
    pytest.importorskip("lanerl_train.vec")
    from lanerl_jax.parity.record import record_trace

    log = record_trace(tmp_path, decisions=40, port_base=41500)
    trace = load_trace(log)
    assert len(trace) >= 20, f"only {len(trace)} snapshots in {log}"
    champs = [e for e in trace[-1].entities if e.is_champion]
    assert len(champs) == 2, [e.kind for e in trace[-1].entities]
    assert all(c.champ is not None and c.champ.level >= 1 for c in champs)


def test_scoping_excludes_from_the_diff_but_never_from_the_count():
    """A scoped diff must still say how much it is not looking at.

    The failure this guards: a run lost four turrets and nobody noticed. If
    scoping could silently drop a kind, the differ would reproduce that bug.
    """
    from dataclasses import replace

    from lanerl_jax.parity.diff import LANE_KINDS

    tol = replace(DEFAULT_TOLERANCE, kinds=LANE_KINDS)
    a = parse_stream(_log(1000, ROWS).splitlines())
    # the prop moves, and a turret goes missing
    moved_prop = PROP.replace("12000,90000", "99000,90000")
    b = parse_stream(_log(1000, [CHAMP, MINION, moved_prop], h="f" * 16).splitlines())

    d = diff_snapshots(a[0], b[0], tol)
    kinds_reported = {x.kind for x in d.entity_diffs}
    assert "LevelProp" not in kinds_reported          # out of scope, not diffed
    assert d.skipped.get("LevelProp") == 2            # but counted
    assert "LaneTurret" in kinds_reported             # in scope, so its loss shows
    assert "out of scope" in d.report()
