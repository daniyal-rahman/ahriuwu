"""The post-step controller comparison: target identity and the AA fire tick.

These are the two fields the injector restores (RESET-001/RESET-002) and the
aggregate report did not score until this pass.  An unscored field is not a
passing field, so the thing worth pinning is not "does it report 100%" -- it
is that a *wrong* sim would be caught: a retarget the server did not make, and
a swing on the wrong tick.  Both are asserted below against a hand-built
tick where the sim is deliberately wrong in one place and right in another.
"""
from types import SimpleNamespace

import numpy as np

from lanerl_jax.parity.one_step import _compare_controller, _q_away_from_zero
from lanerl_jax.parity.trace import AIInternal, Entity, Snapshot, StatQ
from lanerl_jax.parity.inject import UnitInjectionNote


def _internal(net_id, *, target_net_id, target_kind, q_aa_cooldown,
              has_auto_attacked=False, is_attacking=False, kind="LaneMinion",
              team=100):
    return AIInternal(
        net_id=net_id, kind=kind, team=team, q_x=0, q_y=0,
        x_bits=None, y_bits=None,
        target_net_id=target_net_id, target_kind=target_kind, target_team=200,
        target_q_x=0, target_q_y=0, waypoint_key=0, waypoints=(),
        collision_observed=False, collision_q_x=None, collision_q_y=None,
        collision_x_bits=None, collision_y_bits=None,
        q_aa_cooldown=q_aa_cooldown, aa_state=0, q_aa_cast=0, q_aa_delay=0,
        q_aa_windup=0, is_attacking=is_attacking,
        has_auto_attacked=has_auto_attacked,
        q_ai_timer=None, q_ai_local=None, q_time_since_attack=None,
        target_priority=None, lane_waypoint_key=None, had_target=None,
        ignored=(), help=(),
    )


def _note(slot, kind="LaneMinion", team=100):
    return UnitInjectionNote(
        kind=kind, team=team, x=0.0, y=0.0, model_row=0, model_reason="test",
        movement_trustworthy=True, movement_reason="test", slot=slot,
        entity=Entity(kind=kind, q_x=0, q_y=0, team=team),
    )


def _entity(kind="LaneMinion", team=100, dead=False):
    return Entity(kind=kind, q_x=0, q_y=0, team=team, dead=dead)


def test_fire_tick_and_target_identity_catch_a_wrong_sim():
    # Three units.  slot 0 keeps its target and swings on the same tick as
    # the server; slot 1 swings a tick the server did not; slot 2 retargets
    # to a unit the server is not attacking.
    period = 1.6
    pre_cd = np.array([0.0, 0.0, 0.5], np.float32)
    post_cd = np.array([period, period, 0.4833], np.float32)
    state_n = SimpleNamespace(
        aa_cooldown=pre_cd,
        has_auto_attacked=np.array([False, False, True]),
    )
    pred_state = SimpleNamespace(
        target=np.array([1, -1, 0], np.int8),
        is_attacking=np.array([True, True, False]),
        has_auto_attacked=np.array([False, False, True]),
        aa_cooldown=post_cd,
    )
    pre_net_id_to_slot = {10: 0, 11: 1, 12: 2}
    note_by_slot = {0: _note(0), 1: _note(1), 2: _note(2, kind="Champion")}
    real_by_net_id = {10: _entity(), 11: _entity(), 12: _entity("Champion")}
    snap = Snapshot(t_ms=1, entities=[], ai_internals=[
        # server: slot 0 fired (cooldown re-armed) at the same target
        _internal(10, target_net_id=11, target_kind="LaneMinion",
                  q_aa_cooldown=_q_away_from_zero(period, StatQ)),
        # server: slot 1 did NOT fire -- its cooldown only ran down
        _internal(11, target_net_id=0, target_kind="-", q_aa_cooldown=0),
        # server: slot 2's champion is attacking net 11, not net 10
        _internal(12, target_net_id=11, target_kind="LaneMinion",
                  q_aa_cooldown=_q_away_from_zero(0.4833, StatQ),
                  kind="Champion"),
    ])

    pairs = _compare_controller(
        state_n, pred_state, snap, real_by_net_id, pre_net_id_to_slot,
        note_by_slot, np.array([True, True, True]))
    by_slot = {p.slot: p for p in pairs}
    assert set(by_slot) == {0, 1, 2}

    # The agreeing case: same target, same fire tick.
    assert by_slot[0].sim_target_net_id == by_slot[0].server_target_net_id == 11
    assert by_slot[0].sim_fire and by_slot[0].server_fire

    # The sim swung and the server did not.  This is the comparison that was
    # missing: nothing else in the report can see it.
    assert by_slot[1].sim_fire and not by_slot[1].server_fire

    # The sim targets net 10; the server targets net 11.
    assert by_slot[2].sim_target_net_id == 10
    assert by_slot[2].server_target_net_id == 11


def test_no_target_is_scored_as_net_id_zero_not_skipped():
    """``TargetDescriptor`` writes NetId 0 for "no target", and the sim's own
    sentinel is -1.  They have to compare equal, or every idle unit in the
    corpus would read as a target disagreement."""
    state_n = SimpleNamespace(aa_cooldown=np.array([0.0], np.float32),
                             has_auto_attacked=np.array([False]))
    pred_state = SimpleNamespace(
        target=np.array([-1], np.int8), is_attacking=np.array([False]),
        has_auto_attacked=np.array([False]),
        aa_cooldown=np.array([0.0], np.float32))
    snap = Snapshot(t_ms=1, entities=[], ai_internals=[
        _internal(10, target_net_id=0, target_kind="-", q_aa_cooldown=0)])
    pairs = _compare_controller(
        state_n, pred_state, snap, {10: _entity()}, {10: 0},
        {0: _note(0)}, np.array([True]))
    assert len(pairs) == 1
    assert pairs[0].sim_target_net_id == 0 == pairs[0].server_target_net_id


def test_a_unit_either_side_killed_is_left_to_the_death_pass():
    """Scoring a dead unit's controller state would double-count a death
    disagreement as a target one."""
    state_n = SimpleNamespace(aa_cooldown=np.zeros(2, np.float32),
                              has_auto_attacked=np.zeros(2, bool))
    pred_state = SimpleNamespace(
        target=np.array([-1, -1], np.int8), is_attacking=np.zeros(2, bool),
        has_auto_attacked=np.zeros(2, bool), aa_cooldown=np.zeros(2, np.float32))
    snap = Snapshot(t_ms=1, entities=[], ai_internals=[
        _internal(10, target_net_id=0, target_kind="-", q_aa_cooldown=0),
        _internal(11, target_net_id=0, target_kind="-", q_aa_cooldown=0)])

    # net 10: the server killed it.  net 11: the sim killed it.
    pairs = _compare_controller(
        state_n, pred_state, snap,
        {10: _entity(dead=True), 11: _entity()}, {10: 0, 11: 1},
        {0: _note(0), 1: _note(1)}, np.array([True, False]))
    assert pairs == []
