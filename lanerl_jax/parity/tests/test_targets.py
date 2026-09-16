"""Targeting recovered from outputs the server already emits.

These are the lines this project would otherwise have had to add a C# field to
get. The vendored server is shared with every other worktree, so the grammar
below is a contract with code we do not own -- which is exactly why
``test_target_traces_on_a_real_server`` exists: a format change upstream must
fail here loudly rather than silently emptying the trace.
"""
from __future__ import annotations

import pytest

from lanerl_jax.parity.targets import (
    TRACE_ENV,
    champion_targets_from_obs,
    parse_target_traces,
)

TURRET_NONE = "LANERL_TURRET t=233 turret=Turret_T1_C_02_A team=100 target=none"
TURRET_HIT = (
    "LANERL_TURRET t=61500 turret=Turret_T1_C_05_A team=100 target=Red_Minion_Basic "
    "ttype=LaneMinion tteam=200 d=612"
)
TURRET_SAME_TEAM = (
    "LANERL_TURRET t=61500 turret=Turret_T1_C_05_A team=100 target=Blue_Minion_Basic "
    "ttype=LaneMinion tteam=100 d=400 SAME_TEAM"
)
MRT = ("MRT id=1073743485 lt=34766 from=none to=LaneMinion cfh=0 held=34766 "
       "fromprio=14 toprio=9")
MRT_CFH = ("MRT id=1073743485 lt=41000 from=LaneMinion to=Champion cfh=1 held=6234 "
           "fromprio=9 toprio=5")


def test_the_two_trace_switches_are_the_ones_the_server_reads():
    assert TRACE_ENV == {"LANERL_AGGRO_TRACE": "1", "LANERL_TURRET_TRACE": "1"}


def test_a_turret_with_no_target_parses_as_none_not_as_a_unit_named_none():
    t = parse_target_traces([TURRET_NONE]).turret[0]
    assert t.target is None and t.target_type is None and t.distance is None
    assert t.turret == "Turret_T1_C_02_A" and t.team == 100


def test_a_turret_target_carries_enough_to_identify_the_unit():
    t = parse_target_traces([TURRET_HIT]).turret[0]
    assert (t.target_type, t.target_team, t.distance) == ("LaneMinion", 200, 612)


def test_the_same_team_suffix_does_not_break_the_grammar():
    """``TurretTrace`` appends SAME_TEAM when it sees a friendly target.

    That suffix marks a server bug worth noticing, so the parser must survive
    it rather than dropping the line.
    """
    t = parse_target_traces([TURRET_SAME_TEAM]).turret[0]
    assert t.target_team == 100 and t.team == 100


def test_held_target_is_carried_forward_between_changes():
    """The trace prints on CHANGE. "No line" means "still that target"."""
    tt = parse_target_traces([TURRET_NONE, TURRET_HIT])
    assert tt.turret_target_at(70000, "Turret_T1_C_05_A").target == "Red_Minion_Basic"
    assert tt.turret_target_at(1000, "Turret_T1_C_02_A").target is None
    assert tt.turret_target_at(70000, "Turret_T1_C_02_A").target is None


def test_minion_retarget_carries_priority_on_both_sides():
    m = parse_target_traces([MRT]).minion[0]
    # 14 = ClassifyUnit.DEFAULT, 9 = MELEE_MINION. Lower is higher priority.
    assert (m.from_priority, m.to_priority) == (14, 9)
    assert m.from_kind == "none" and m.to_kind == "LaneMinion"
    assert m.held_ms == 34766 and m.from_call_for_help is False


def test_a_call_for_help_retarget_is_flagged():
    m = parse_target_traces([MRT_CFH]).minion[0]
    assert m.from_call_for_help is True
    assert m.to_kind == "Champion" and m.to_priority == 5   # CHAMPION_ATTACKING_MINION


def test_traces_survive_interleaved_server_chatter():
    noise = ["LANERL_TPS 2580.0 ticks/s", "[INFO] something", "LANERL_CS blue=3"]
    tt = parse_target_traces(noise[:1] + [MRT] + noise[1:] + [TURRET_HIT])
    assert len(tt.minion) == 1 and len(tt.turret) == 1


def test_champion_target_resolves_the_netid_within_the_same_observation():
    obs = {"t": 5000, "u": [
        {"id": 7, "k": "Champion", "tm": 100, "x": 1, "y": 2, "tgt": 42, "atk": 1, "mo": 3},
        {"id": 8, "k": "Champion", "tm": 200, "x": 3, "y": 4, "tgt": 0, "atk": 0, "mo": 2},
        {"id": 42, "k": "LaneMinion", "tm": 200, "x": 5, "y": 6},
    ]}
    blue, red = champion_targets_from_obs(obs)
    assert blue.target_net_id == 42 and blue.target_kind == "LaneMinion"
    assert blue.target_team == 200 and blue.is_attacking and blue.move_order == 3
    assert red.target_net_id is None and red.target_kind is None and not red.is_attacking


def test_an_unresolvable_target_is_kept_not_dropped():
    """A target absent from the observation is a fact (fog, or it just died)."""
    obs = {"t": 1, "u": [{"id": 7, "k": "Champion", "tm": 100, "tgt": 999, "atk": 0, "mo": 0}]}
    c = champion_targets_from_obs(obs)[0]
    assert c.target_net_id == 999 and c.target_kind is None


@pytest.mark.slow
@pytest.mark.skipif(
    not __import__("lanerl_train.paths", fromlist=["paths"]).server_available(),
    reason="vendored server build not available",
)
def test_target_traces_on_a_real_server(tmp_path):
    """The grammars above are a contract with a server we do not own.

    Validated on a 200 s bot-driven run (2026-09-16): 153 minion retargets with
    priorities matching the ``ClassifyUnit`` enum, 689 resolved champion targets,
    0 unresolvable NetIds. This test is the smaller version that keeps it honest.
    """
    from lanerl_jax.parity.record import record_fixture

    # 3,900 decisions = 130 s of game time. The length is load-bearing: the
    # first minion wave is at 90 s, so a shorter fixture contains no minions at
    # all and `targets.minion` is legitimately empty -- which reads as "the
    # AGGRO_TRACE switch was renamed" rather than "the run ended too early".
    fx = record_fixture(tmp_path, decisions=3900, port_base=43600, tag="targets")
    trace, targets, actions, obs = fx.load()

    assert len(trace) > 100, f"only {len(trace)} snapshots"
    assert len(obs) > 100 and len(actions.t_ms) == len(actions.blue) == len(actions.red)

    # The scripted drive issues attack-moves, so champions must acquire targets.
    ct = [c for o in obs for c in champion_targets_from_obs(o)]
    assert ct, "no champion rows carried tgt/atk -- did the wire format change?"
    assert any(c.target_net_id is not None for c in ct), (
        "no champion ever had a target across the whole fixture; either the "
        "scripted drive stopped issuing attack-moves or `tgt` left the wire"
    )
    assert all(c.target_kind is not None
               for c in ct if c.target_net_id is not None), "unresolvable target NetId"

    # Minions must retarget; the priorities must be in the ClassifyUnit range.
    assert float(trace[-1].t_ms) > 95_000, (
        f"fixture ended at {trace[-1].t_ms/1000:.0f}s, before the 90 s first "
        "wave -- lengthen it rather than dropping the minion assertion")
    assert targets.minion, "LANERL_AGGRO_TRACE produced nothing -- switch renamed?"
    assert all(1 <= m.to_priority <= 14 for m in targets.minion)
