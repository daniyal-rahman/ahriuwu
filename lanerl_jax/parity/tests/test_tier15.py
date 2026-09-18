"""The instrument before the measurement, for Tier 1.5.

`tier15` answers two ledger questions, so the ways it could quietly lie are
worth more than the ways it could crash:

* it could pick a start state that does not satisfy the predicate it claims,
  which would make an "identical situation" comparison a comparison of two
  different situations under a reassuring label;
* it could keep running past the point where the two sides stop executing the
  same order stream, which is the mistake Tier 2 cannot avoid and Tier 1.5
  exists to avoid;
* it could report a max error and hide whether the split was immediate or
  gradual, which is what the "curve, not a scalar" requirement is about.

Every test below is one of those three. None of them boot a server.
"""
from __future__ import annotations

import math
from dataclasses import replace as dc_replace

import pytest

from lanerl_jax.parity.action_replay import RecordedDecision
from lanerl_jax.parity.tier15 import (STEP_TICKS, SideTick, StartSelection,
                                      Tick15Row, Tier15Result,
                                      _seed_champion_orders, select_start)
from lanerl_jax.parity.trace import parse_stream

# ---------------------------------------------------------------------------
# synthetic trace, written to LanerlStateDump.Describe's grammar
# ---------------------------------------------------------------------------

def _champ(x_q: int, y_q: int, move_order: int = 2) -> str:
    return (f"Champion|100|{x_q},{y_q}|759808/772096|A"
            f"|{move_order}|3|-|-|1|-"
            "|59269,28197,45219,353280,1024|1|486400|0|0|1"
            "|1:0|0:-1|1:5120|-1:-1")


def _minion(team: int, x_q: int, y_q: int, max_hp_q: int) -> str:
    return f"LaneMinion|{team}|{x_q},{y_q}|{max_hp_q}/{max_hp_q}|A|4|2|-|-|1|"


def _internal(net_id: int, kind: str, team: int, x_q: int, y_q: int,
              target_net_id: int = 0) -> str:
    return (f"LANERL_INTERNAL t=%d ai id={net_id} kind={kind} team={team} "
            f"x={x_q} y={y_q} target={target_net_id},-,0,0,0 aacd=0 aastate=0 "
            "aacast=0 aadelay=0 aawindup=0 attacking=0 hasaa=0 aitimer=- "
            "ailocal=- aitsa=- aiprio=- aiwp=- aihad=- aiignore=- aihelp=-")


def _snapshot(t_ms: int, rows, internals) -> str:
    out = [f"LANERL_STATEHASH t={t_ms} n={len(rows)} h={t_ms:016x}"]
    out += [f"LANERL_STATEROW t={t_ms} {r}" for r in rows]
    out += [i % t_ms if "%d" in i else i for i in internals]
    return "\n".join(out)


#: melee/caster/cannon max HP in StatQ units, per `inject.infer_minion_model`.
MELEE_Q, CASTER_Q, CANNON_Q = 455 * 1024, 290 * 1024, 700 * 1024


@pytest.fixture(scope="module")
def params():
    from lanerl_jax.sim.init import lane_params
    return lane_params()


def _trace(specs):
    """specs: list of (t_ms, champ_xq, minion_xq, minion_hp_q, target_of_minion)."""
    text = []
    for t_ms, cxq, mxq, mhp, tgt in specs:
        rows = [_champ(cxq, 100_000), _minion(200, mxq, 100_000, mhp)]
        internals = [
            _internal(7001, "Champion", 100, cxq, 100_000),
            _internal(8001, "LaneMinion", 200, mxq, 100_000, tgt),
        ]
        text.append(_snapshot(t_ms, rows, internals))
    return parse_stream("\n".join(text).splitlines())


# ---------------------------------------------------------------------------
# 1. the start state actually satisfies the stated predicate
# ---------------------------------------------------------------------------

def test_champ_near_minion_picks_the_first_tick_inside_the_radius(params):
    # champion fixed at x=6000 world units; the minion closes 100 u per tick.
    specs = [(t * 17, 6000 * 16, (6400 - 100 * t) * 16, MELEE_Q, 0)
             for t in range(6)]
    sel = select_start(_trace(specs), "champ_near_minion", params, radius=90.0)
    assert sel.predicate == "champ_near_minion"
    # t=0..2 are 400/300/200 u apart; t=3 is 100 (still out); t=4 is exactly 0.
    assert sel.detail["nearest_enemy_minion_u"] <= 90.0
    assert sel.index == 4
    # and the tick BEFORE it must not satisfy the predicate, or the "first"
    # in the docstring is not first.
    with pytest.raises(ValueError, match="never fired"):
        select_start(_trace(specs[:4]), "champ_near_minion", params, radius=90.0)


def test_champ_near_minion_refuses_a_stationary_champion(params):
    """The COLL-003 ratchet needs a champion that is trying to walk."""
    specs = [(t * 17, 6000 * 16, 6000 * 16, MELEE_Q, 0) for t in range(3)]
    tr = _trace(specs)
    assert select_start(tr, "champ_near_minion", params).index == 0
    for snap in tr:
        ch = snap.entities[0]
        snap.entities[0] = dc_replace(
            ch, ai=dc_replace(ch.ai, move_order=1))          # HOLD
    with pytest.raises(ValueError, match="never fired"):
        select_start(tr, "champ_near_minion", params)


def test_minion_type_filter_is_the_cannon_asymmetry_COLL_003_is_about(params):
    """A melee in contact is not the case whose escape exceeds its trigger."""
    specs = [(0, 6000 * 16, 6000 * 16, MELEE_Q, 0),
             (17, 6000 * 16, 6000 * 16, CANNON_Q, 0)]
    tr = _trace(specs)
    assert select_start(tr, "champ_near_minion", params,
                        minion_type="any").index == 0
    assert select_start(tr, "champ_near_minion", params,
                        minion_type="cannon").index == 1


def test_engaged_reads_the_target_from_the_diagnostic_stream(params):
    """Without internals there is no minion target at all, so no `engaged`."""
    specs = [(0, 6000 * 16, 6100 * 16, MELEE_Q, 0),
             (17, 6000 * 16, 6100 * 16, MELEE_Q, 7001)]
    tr = _trace(specs)
    sel = select_start(tr, "engaged", params, min_attackers=1)
    assert sel.index == 1 and sel.detail["n_attackers"] == 1
    with pytest.raises(ValueError, match="never fired"):
        select_start(tr, "engaged", params, min_attackers=2)


def test_a_predicate_that_never_fires_refuses_rather_than_approximating(params):
    specs = [(t * 17, 6000 * 16, 60_000 * 16, MELEE_Q, 0) for t in range(4)]
    with pytest.raises(ValueError) as e:
        select_start(_trace(specs), "champ_near_minion", params, radius=90.0)
    assert "do NOT hand-pick" in str(e.value)


def test_tail_ticks_keeps_the_whole_window_inside_the_trace(params):
    """A start state with no room for the window is a truncated comparison."""
    specs = [(t * 17, 6000 * 16, 6000 * 16, MELEE_Q, 0) for t in range(4)]
    tr = _trace(specs)
    assert select_start(tr, "champ_near_minion", params, tail_ticks=0).index == 0
    with pytest.raises(ValueError, match="never fired|fewer than"):
        select_start(tr, "champ_near_minion", params, tail_ticks=4)


# ---------------------------------------------------------------------------
# 2. the order stream stays identical, or the window ends
# ---------------------------------------------------------------------------

def test_seed_replays_a_move_and_refuses_to_replay_a_cast_or_attack():
    """Re-issuing a move restores an unobservable waypoint list; re-issuing a
    cast would fire a second spell the server never cast."""
    ids = {5: 3}
    move = RecordedDecision(10, {"t": "move", "x": 1.0, "y": 2.0}, {"t": "noop"})
    orders = _seed_champion_orders(move, ids)
    assert orders is not None
    assert float(orders.x[0]) == 1.0 and float(orders.y[0]) == 2.0

    cast = RecordedDecision(10, {"t": "cast", "slot": 2, "id": 0}, {"t": "noop"})
    assert _seed_champion_orders(cast, ids) is None
    attack = RecordedDecision(10, {"t": "attack", "id": 5}, {"t": "noop"})
    assert _seed_champion_orders(attack, ids) is None
    assert _seed_champion_orders(None, ids) is None


# ---------------------------------------------------------------------------
# 3. divergence is reported as a curve, not a scalar
# ---------------------------------------------------------------------------

def _side(x: float, hp: float = 100.0, attackers=()) -> SideTick:
    return SideTick(t_ms=0, champ_x=x, champ_y=0.0, champ_hp=hp,
                    champ_alive=True, champ_move_order=2, champ_cs=0,
                    champ_deaths=0, displacement=0.0, n_own_minions=3,
                    n_foe_minions=3, nearest_enemy_minion_u=50.0,
                    n_attackers=len(attackers), attacker_ids=tuple(attackers),
                    attacker_dist_to_own_wave_u=(600.0,) * len(attackers),
                    attacker_dist_to_champ_u=(120.0,) * len(attackers))


def _result(errs, attackers_sim=(), attackers_srv=()) -> Tier15Result:
    rows = []
    for k, e in enumerate(errs, 1):
        a = attackers_sim[k - 1] if attackers_sim else ()
        b = attackers_srv[k - 1] if attackers_srv else ()
        rows.append(Tick15Row(
            k=k, decision=k / STEP_TICKS, t_ms=k * 17,
            sim=_side(e, 100.0 - e, a), srv=_side(0.0, 100.0, b),
            champ_pos_err=e, champ_hp_err=-e, matched_minions=3,
            minion_pos_err_max=e / 2, minion_pos_err_mean=e / 4,
            unmatched_sim=0, unmatched_srv=0,
            field_diffs=("pos.x",) if e > 1 else ()))
    return Tier15Result(
        fixture="synthetic", start=StartSelection(0, 0, "index"),
        decisions_requested=len(errs), ticks_run=len(errs),
        truncated_reason=None, injection={}, champion_waypoint_mode="freeze",
        routed=True, rows=rows)


def test_gradual_and_immediate_splits_are_distinguishable_by_first_crossing():
    """Same max error, opposite shapes -- the number a scalar would hide."""
    gradual = _result([0.1 * k for k in range(1, 101)])
    immediate = _result([10.0] * 100)
    assert max(gradual.champ_pos_curve()) == pytest.approx(10.0)
    assert max(immediate.champ_pos_curve()) == pytest.approx(10.0)
    assert gradual.first_crossing(gradual.champ_pos_curve(), 1.0) == 11
    assert immediate.first_crossing(immediate.champ_pos_curve(), 1.0) == 1
    assert gradual.first_crossing(gradual.champ_pos_curve(), 100.0) is None


def test_field_onset_is_the_first_tick_a_field_moved_not_the_worst():
    res = _result([0.5, 0.5, 4.0, 0.5, 9.0])
    assert res.field_onsets() == {"pos.x": 3}


def test_holds_are_split_by_attacker_and_a_gap_closes_one():
    """A minion that drops the champion and re-acquires has TWO holds. Scoring
    it as one long hold is exactly how a 1.26x hold-length gap could be
    manufactured out of nothing."""
    sim = [(1,), (1,), (), (1,), (1,)]
    srv = [(1,), (1,), (1,), (1,), (1,)]
    res = _result([0.0] * 5, attackers_sim=sim, attackers_srv=srv)
    h = res.hold_stats()
    assert h["sim"]["holds_closed"] == 1 and h["sim"]["mean_hold_ticks"] == 2
    assert h["sim"]["holds_open_at_window_end"] == 1
    assert h["server"]["holds_closed"] == 0
    assert h["server"]["longest_open_hold_ticks"] == 5


def test_attacker_distance_is_per_attacker_not_a_median_of_medians():
    res = _result([0.0] * 4, attackers_sim=[(1, 2)] * 4,
                  attackers_srv=[(1,)] * 4)
    h = res.hold_stats()
    assert h["sim"]["attacker_ticks"] == 8      # 2 attackers x 4 ticks
    assert h["server"]["attacker_ticks"] == 4
    assert h["sim"]["frac_attacker_ticks_beyond_475u"] == 1.0


def test_report_renders_and_names_the_asymmetry_it_cannot_remove():
    text = _result([0.1 * k for k in range(1, 21)]).report(curve_points=5)
    assert "ONLY THE SIM IS INJECTED" in text
    assert "first crossing" in text
    assert "Q1" in text and "Q2" in text
