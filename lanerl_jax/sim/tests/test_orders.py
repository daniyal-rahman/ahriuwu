"""Champion order ingress: sticky targets, and the no-op Stop order.

``docs/PORT_AUDIT_AI.md`` rows 3.2/3.4: our training/eval/parity harness never
goes through the client packet path -- it drives ``GameServerLib/Lanerl/
LanerlControl.cs``, whose ``Move`` case never touches ``TargetUnit`` at all,
and whose wire (``LanerlWire.LanerlOrderKind``) has no ``Stop`` kind. Both are
pinned here directly against ``apply_orders``, independent of the full tick
(``sim/tests/test_lane.py`` and the parity gates exercise the end-to-end
consequence via ``step.py``'s "3b. RefreshWaypoints" block, which is unowned
by this file and untouched by this fix).
"""
from __future__ import annotations

import jax.numpy as jnp
import pytest

from lanerl_jax.sim.orders import OrderKind, Orders, apply_orders
from lanerl_jax.sim.state import Kind, MoveOrder, Team, empty_state

pytestmark = pytest.mark.filterwarnings("ignore")


def _state_with_target(target: int = 5):
    s = empty_state()
    n = s.kind.shape[0]
    kind = jnp.asarray(s.kind).at[0].set(Kind.CHAMPION).at[1].set(Kind.CHAMPION) \
        .at[target].set(Kind.LANE_MINION)
    team = jnp.asarray(s.team).at[0].set(Team.BLUE).at[1].set(Team.RED) \
        .at[target].set(Team.RED)
    alive = jnp.zeros((n,), bool).at[0].set(True).at[1].set(True).at[target].set(True)
    s = s.replace(kind=kind, team=team, alive=alive,
                  target=jnp.asarray(s.target).at[0].set(target))
    return s


def _order(kind, x=0.0, y=0.0, target=-1):
    return Orders(kind=jnp.asarray([kind, OrderKind.NOOP], jnp.int8),
                 x=jnp.asarray([x, 0.0]), y=jnp.asarray([y, 0.0]),
                 target=jnp.asarray([target, -1], jnp.int8))


def test_a_move_order_does_not_clear_a_held_target():
    """`LanerlControl.cs:349-371` -- `case Move` never calls `SetTargetUnit`."""
    s = _state_with_target(target=5)
    s2 = apply_orders(s, _order(OrderKind.MOVE, x=1000.0, y=1000.0))
    assert int(s2.target[0]) == 5, "a Move order must not drop the held target"
    assert int(s2.move_order[0]) == MoveOrder.MOVE_TO


def test_an_attack_order_still_replaces_the_target():
    """`Attack` is `SetTargetUnit` alone -- unaffected by this fix."""
    s = _state_with_target(target=5)
    s2 = apply_orders(s, _order(OrderKind.ATTACK, target=6))
    assert int(s2.target[0]) == 6


def test_stop_is_a_true_no_op():
    """`LanerlWire.LanerlOrderKind` has no `Stop` member at all -- our own
    `OrderKind.STOP` (with no wire receiver) must not clear the target or
    change `move_order` either, matching what happens when
    `LanerlControl.Execute`'s switch sees an order kind it does not handle."""
    s = _state_with_target(target=5)
    s = s.replace(move_order=jnp.asarray(s.move_order).at[0].set(MoveOrder.ATTACK_TO))
    s2 = apply_orders(s, _order(OrderKind.STOP))
    assert int(s2.target[0]) == 5
    assert int(s2.move_order[0]) == MoveOrder.ATTACK_TO


def test_noop_leaves_target_and_move_order_untouched():
    s = _state_with_target(target=5)
    s = s.replace(move_order=jnp.asarray(s.move_order).at[0].set(MoveOrder.HOLD))
    s2 = apply_orders(s, _order(OrderKind.NOOP))
    assert int(s2.target[0]) == 5
    assert int(s2.move_order[0]) == MoveOrder.HOLD
