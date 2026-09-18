import numpy as np
import pytest

from lanerl_jax.parity.action_replay import (
    ActionReplayError,
    RecordedDecision,
    align_action_log,
    decision_to_orders,
    net_id_to_injected_slot,
)
from lanerl_jax.parity.inject import UnitInjectionNote
from lanerl_jax.parity.record import ActionLog
from lanerl_jax.parity.trace import Snapshot, parse_stream
from lanerl_jax.parity.trace import parse_row
from lanerl_jax.sim.orders import OrderKind
from lanerl_jax.sim.init import empty_state
from lanerl_jax.sim.state import Kind


def test_action_times_align_to_rounded_endpoint_dump_not_preceding_tick():
    trace = [Snapshot(t_ms=t) for t in (0, 17, 33, 50, 67, 83)]
    actions = ActionLog(
        t_ms=[16, 50, 83],
        blue=[{"t": "move", "x": 1, "y": 2}, {"t": "noop"}, {"t": "recall"}],
        red=[{"t": "noop"}] * 3,
    )
    aligned = align_action_log(trace, actions)
    assert list(aligned) == [1, 3, 5]
    assert aligned[1].source_t_ms == 16
    # Pair 0 -> 1 receives the first action at its endpoint. Applying it to
    # pair 1 -> 2 would move one whole server tick late.
    assert 0 not in aligned and 2 not in aligned


def test_alignment_ignores_actions_outside_a_sliced_trace_but_rejects_holes():
    trace = [Snapshot(t_ms=t) for t in (1000, 1017, 1033)]
    actions = ActionLog([16, 1016], [{"t": "noop"}] * 2, [{"t": "noop"}] * 2)
    assert list(align_action_log(trace, actions)) == [1]

    corrupt = ActionLog([1008], [{"t": "noop"}], [{"t": "noop"}])
    with pytest.raises(ActionReplayError, match="no snapshot"):
        align_action_log(trace, corrupt)


def test_all_phase_one_wire_orders_convert_to_two_champion_slots():
    ids = {111: 7, 222: 1}
    cases = [
        ({"t": "noop"}, OrderKind.NOOP, -1),
        ({"t": "move", "x": 12.5, "y": 30}, OrderKind.MOVE, -1),
        ({"t": "attack", "id": 111}, OrderKind.ATTACK, 7),
        ({"t": "cast", "slot": 0}, OrderKind.CAST_Q, -1),
        ({"t": "cast", "slot": 1}, OrderKind.CAST_W, -1),
        ({"t": "cast", "slot": 2}, OrderKind.CAST_E, -1),
        ({"t": "cast", "slot": 3, "id": 222}, OrderKind.CAST_R, 1),
        ({"t": "recall"}, OrderKind.RECALL, -1),
    ]
    for wire, expected_kind, expected_target in cases:
        orders = decision_to_orders(
            RecordedDecision(100, wire, {"t": "noop"}), ids)
        assert int(np.asarray(orders.kind)[0]) == expected_kind
        assert int(np.asarray(orders.target)[0]) == expected_target


def test_targeted_order_without_diagnostic_netid_fails_closed():
    decision = RecordedDecision(
        100, {"t": "attack", "id": 999}, {"t": "noop"})
    with pytest.raises(ActionReplayError, match="absent from diagnostic"):
        decision_to_orders(decision, {})


def test_diagnostic_netid_maps_to_the_slot_chosen_by_injection():
    lines = [
        "LANERL_STATEHASH t=100 n=1 h=0000000000000001",
        "LANERL_STATEROW t=100 LaneMinion|200|18000,170000|477184/477184|A|4|2|-|-|1|",
        "LANERL_INTERNAL t=100 ai id=77 kind=LaneMinion team=200 "
        "x=18000 y=170000 target=0,-,0,0,0 wpkey=1 wps=none coll=none "
        "aacd=0 aastate=0 aacast=0 aadelay=0 aawindup=0 "
        "attacking=0 hasaa=0 aitimer=0 ailocal=0 aitsa=0 "
        "aiprio=4 aiwp=3 aihad=0 aiignore=- aihelp=-",
    ]
    snapshot = parse_stream(lines)[0]
    ent = snapshot.entities[0]
    note = UnitInjectionNote(
        kind=ent.kind, team=ent.team, x=ent.x, y=ent.y,
        model_row=3, model_reason="test", movement_trustworthy=True,
        movement_reason="test", slot=9, entity=ent)
    assert net_id_to_injected_slot(snapshot, [note]) == {77: 9}


def test_one_step_netid_survives_collision_displacement_beyond_legacy_radius(monkeypatch):
    """A 48u push is still the same unit, not one death plus one spawn."""
    import jax.numpy as jnp
    import lanerl_jax.parity.one_step as one_step

    before = parse_row(
        "LaneMinion|200|18000,170000|477184/477184|A|4|2|-|-|1|")
    # 48 world units = 768 position quanta, far beyond the old 8u matcher.
    endpoint_lines = [
        "LANERL_STATEHASH t=1017 n=1 h=0000000000000001",
        "LANERL_STATEROW t=1017 LaneMinion|200|18768,170000|477184/477184|A|4|2|-|-|1|",
        "LANERL_INTERNAL t=1017 ai id=77 kind=LaneMinion team=200 "
        "x=18768 y=170000 target=0,-,0,0,0 wpkey=0 wps=none coll=none "
        "aacd=0 aastate=0 aacast=0 aadelay=0 aawindup=0 attacking=0 hasaa=0 "
        "aitimer=0 ailocal=0 aitsa=0 aiprio=4 aiwp=0 aihad=0 aiignore=- aihelp=-",
    ]
    endpoint = parse_stream(endpoint_lines)[0]
    note = UnitInjectionNote(
        kind=before.kind, team=before.team, x=before.x, y=before.y,
        model_row=3, model_reason="test", movement_trustworthy=True,
        movement_reason="exact diagnostic", slot=2, entity=before)
    state = empty_state().replace(
        t_ms=jnp.asarray(1000.0),
        kind=empty_state().kind.at[2].set(Kind.LANE_MINION),
        alive=empty_state().alive.at[2].set(True))
    predicted = Snapshot(t_ms=1017, entities=[before])
    monkeypatch.setattr(one_step, "_tick_jit", lambda s, params, lane_path=None: s)
    monkeypatch.setattr(
        one_step, "state_to_snapshot",
        lambda state, t_ms, params: predicted)

    result = one_step.compare_one_tick(
        state, [note], endpoint, {}, None, pre_net_id_to_slot={77: 2})
    assert result.identity_mode == "diagnostic NetId"
    assert len(result.deaths) == 1
    assert result.deaths[0].server_alive_after
    assert result.spawns[0].n_real_new == 0
    assert len(result.matched) == 1
    assert result.matched[0].match_distance_q == pytest.approx(768.0)


def test_endpoint_move_replay_uses_the_production_route_inputs(monkeypatch):
    """Driven parity must compare the server route, not a raw two-point move."""
    import jax.numpy as jnp
    import lanerl_jax.parity.one_step as one_step

    state = empty_state()
    endpoint = Snapshot(t_ms=17)
    orders = decision_to_orders(
        RecordedDecision(17, {"t": "move", "x": 10, "y": 20},
                         {"t": "noop"}), {})
    route_table = object()
    terrain = object()
    seen = {}

    monkeypatch.setattr(one_step, "_tick_jit", lambda s, params, lane_path=None: s)
    monkeypatch.setattr(
        one_step, "apply_orders",
        lambda s, o, p, **kwargs: seen.update(kwargs) or s)
    monkeypatch.setattr(
        one_step, "state_to_snapshot",
        lambda state, t_ms, params: Snapshot(t_ms=int(t_ms)))

    one_step.compare_one_tick(
        state, [], endpoint, {}, jnp.zeros((1, 2)),
        endpoint_orders=orders, route_table=route_table, terrain=terrain)
    assert seen == {"route_table": route_table, "terrain": terrain}
