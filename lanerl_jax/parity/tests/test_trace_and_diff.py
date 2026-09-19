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


def test_optional_internal_state_parses_without_changing_canonical_count():
    lines = _log(1000, [MINION]).splitlines()
    lines += [
        "LANERL_INTERNAL t=1000 ai id=77 kind=LaneMinion team=200 "
        "x=18000 y=170000 xbits=1150042112 ybits=1170513920 "
        "target=88,LaneMinion,100,17900,169900 wpkey=1 "
        "wps=18000,170000;17900,169900 coll=17998,169998 "
        "collbits=1150041088,1170513792 "
        "aacd=512 aastate=1 aacast=0 aadelay=100 aawindup=245 "
        "attacking=1 hasaa=0 aitimer=128000 ailocal=92160000 aitsa=0 "
        "aiprio=4 aiwp=3 aihad=1 aiignore=88:92672000 aihelp=88:2",
        "LANERL_INTERNAL t=1000 missile id=99 kind=SpellMissile x=18001 y=170001 "
        "owner=77 target=88 speed=665600 damage=24576",
    ]
    trace = parse_stream(lines)
    assert len(trace[0].entities) == 1
    ai = trace[0].ai_internals[0]
    assert ai.net_id == 77 and ai.target_net_id == 88
    assert ai.ignored == ((88, 92672000),) and ai.help == ((88, 2),)
    assert ai.q_aa_windup == 245 and ai.is_attacking
    assert ai.collision_observed
    assert (ai.collision_q_x, ai.collision_q_y) == (17998, 169998)
    assert ai.x_bits == 1150042112 and ai.collision_x_bits == 1150041088
    missile = trace[0].missile_internals[0]
    assert missile.owner_net_id == 77 and missile.q_damage == 24576


def test_legacy_internal_distinguishes_unobserved_collision_from_explicit_none():
    base = (
        "id=77 kind=LaneMinion team=200 x=18000 y=170000 "
        "target=0,-,0,0,0 aacd=0 aastate=0 aacast=0 aadelay=0 "
        "aawindup=0 attacking=0 hasaa=0 aitimer=- ailocal=- aitsa=- "
        "aiprio=- aiwp=- aihad=- aiignore=- aihelp=-"
    )
    legacy = parse_stream(
        _log(1000, [MINION]).splitlines()
        + [f"LANERL_INTERNAL t=1000 ai {base}"])[0].ai_internals[0]
    explicit = parse_stream(
        _log(1000, [MINION]).splitlines()
        + [f"LANERL_INTERNAL t=1000 ai {base} wpkey=0 wps=none coll=none"]
    )[0].ai_internals[0]

    assert not legacy.collision_observed
    assert legacy.collision_q_x is None and legacy.collision_q_y is None
    assert legacy.waypoints == () and legacy.waypoint_key == 0
    assert explicit.collision_observed
    assert explicit.collision_q_x is None and explicit.collision_q_y is None


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


# --------------------------------------------------------------------------
# load_trace_window: the full loader is the ORACLE, never a re-derivation
# --------------------------------------------------------------------------
# These deliberately do not restate what a windowed parse "should" produce.
# Restating it would reproduce any mistake in the windowing logic inside the
# expectation and pass -- the `STAT-002` trap, where two tests agreed with a
# wrong constant because both read it from the code under test. Instead the
# unwindowed `load_trace` parses the same bytes and IS the expectation.


def _multi_tick_log(times):
    return "".join(_log(t, ROWS, h=f"{i:016x}") for i, t in enumerate(times))


def test_a_windowed_parse_agrees_with_the_full_parse_inside_the_window(tmp_path):
    from lanerl_jax.parity.trace import load_trace_window

    times = [1000 + 33 * i for i in range(10)]
    log = tmp_path / "s.log"
    log.write_text(_multi_tick_log(times))

    full = load_trace(log)
    win = load_trace_window(log, from_ms=1099, to_ms=1198)

    # Same snapshots, same order, same count -- windowing must not renumber.
    assert [s.t_ms for s in win.snapshots] == [s.t_ms for s in full.snapshots]
    for a, b in zip(full.snapshots, win.snapshots):
        if b.placeholder:
            continue
        assert b.state_hash == a.state_hash and b.expected_n == a.expected_n
        assert [e.describe() if hasattr(e, "describe") else repr(e)
                for e in b.entities] == [
               e.describe() if hasattr(e, "describe") else repr(e)
               for e in a.entities], f"t={a.t_ms} rows differ from the full parse"
    real = [s.t_ms for s in win.snapshots if not s.placeholder]
    # 1099..1198 covers 1099/1132/1165/1198, plus the one successor the
    # docstring promises so the window's last pair is not silently dropped.
    assert real == [1099, 1132, 1165, 1198, 1231]


def test_a_window_placeholder_refuses_to_be_scored_instead_of_agreeing(tmp_path):
    """The failure this exists to prevent: a placeholder has no entities
    because none were PARSED. Diffed against a real snapshot it would score
    every entity as missing-on-both-sides, i.e. as agreement -- the same
    silent corruption `parse_stream` raises over for a truncated snapshot.
    """
    from lanerl_jax.parity.trace import load_trace_window

    log = tmp_path / "s.log"
    log.write_text(_multi_tick_log([1000, 1033, 1066]))
    win = load_trace_window(log, from_ms=1066, to_ms=1066)
    ghost = win.snapshots[0]
    assert ghost.placeholder and ghost.t_ms == 1000
    with pytest.raises(TraceFormatError, match="window placeholder"):
        ghost.by_group()
    with pytest.raises(TraceFormatError, match="window placeholder"):
        ghost.champion(100)
    with pytest.raises(TraceFormatError, match="window placeholder"):
        diff_snapshots(ghost, win.snapshots[-1])


def test_max_snapshots_bounds_the_parse_not_just_the_caller_s_loop(tmp_path):
    """``--max-pairs`` used to bound the WORK while the parse had already
    built every entity in the file -- which is why a 2,000-pair slice still
    OOMed. A cap the parse itself honours is the difference.
    """
    from lanerl_jax.parity.trace import load_trace_window

    times = [1000 + 33 * i for i in range(20)]
    log = tmp_path / "s.log"
    log.write_text(_multi_tick_log(times))
    win = load_trace_window(log, max_snapshots=3)
    assert len(win.snapshots) == 20, "snapshot indices must still line up"
    assert sum(not s.placeholder for s in win.snapshots) == 3
    assert sum(len(s.entities) for s in win.snapshots) == 3 * len(ROWS)


def test_the_wave_replay_is_unchanged_by_windowing(tmp_path):
    """`replay_wave_states` is the one consumer that genuinely needs every
    tick from `FIRST_WAVE_MS` forward, and it is the reason placeholders
    carry `t_ms` rather than the window simply starting late.
    """
    from lanerl_jax.parity.inject import replay_wave_states
    from lanerl_jax.parity.trace import load_trace_window

    times = [90000 + 33 * i for i in range(40)]
    log = tmp_path / "s.log"
    log.write_text(_multi_tick_log(times))
    full = replay_wave_states(load_trace(log).snapshots)
    win = replay_wave_states(load_trace_window(log, from_ms=91000).snapshots)
    assert [(w.next_spawn_ms, w.minion_number, w.cannon_count) for w in win] == \
           [(w.next_spawn_ms, w.minion_number, w.cannon_count) for w in full]
