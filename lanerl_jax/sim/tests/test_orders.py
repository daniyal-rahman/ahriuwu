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
import numpy as np
import pytest

from lanerl_jax.sim.init import init_lane, lane_params
from lanerl_jax.sim.local_pathing import LocalRouteStatus
from lanerl_jax.sim.orders import OrderKind, Orders, apply_orders
from lanerl_jax.sim.spells import Q_BUFF_SLOT, BuffId, Slot
from lanerl_jax.sim.state import Kind, MoveOrder, Team, empty_state
from lanerl_jax.sim.step import tick

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
    assert int(s2.route_status[0]) == LocalRouteStatus.TABLE_DISABLED


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


def test_silence_blocks_spell_orders_but_not_movement():
    s = _state_with_target()
    s = s.replace(
        silenced_ms=s.silenced_ms.at[0].set(500.0),
        spell_level=s.spell_level.at[0, Slot.Q].set(1))
    blocked = apply_orders(s, _order(OrderKind.CAST_Q))
    assert int(blocked.buff_id[0, Q_BUFF_SLOT]) == BuffId.NONE

    moved = apply_orders(s, _order(OrderKind.MOVE, x=50.0, y=75.0))
    assert int(moved.move_order[0]) == MoveOrder.MOVE_TO


def test_recall_stops_an_unfinished_path_and_starts_its_windup():
    """LanerlControl stops before casting the BluePill, not after it."""
    s = _state_with_target()
    s = s.replace(
        move_order=s.move_order.at[0].set(MoveOrder.MOVE_TO),
        n_waypoints=s.n_waypoints.at[0].set(2),
        waypoint_key=s.waypoint_key.at[0].set(1),
    )
    s2 = apply_orders(s, _order(OrderKind.RECALL))
    assert float(s2.recall_windup_ms[0]) == pytest.approx(500.0)
    assert float(s2.recall_channel_ms[0]) == 0.0
    assert int(s2.move_order[0]) == MoveOrder.STOP
    assert int(s2.target[0]) == -1
    assert int(s2.n_waypoints[0]) == 1


def test_recall_keeps_a_target_when_its_old_path_was_already_finished():
    """`UpdateMoveOrder(Stop)` only clears a target inside `!IsPathEnded`."""
    s = _state_with_target(target=5)
    s = s.replace(n_waypoints=s.n_waypoints.at[0].set(1),
                  waypoint_key=s.waypoint_key.at[0].set(1))
    s2 = apply_orders(s, _order(OrderKind.RECALL))
    assert int(s2.target[0]) == 5


@pytest.mark.parametrize("kind, slot, target", [
    (OrderKind.CAST_Q, Slot.Q, -1),
    (OrderKind.CAST_W, Slot.W, -1),
    (OrderKind.CAST_E, Slot.E, -1),
    (OrderKind.CAST_R, Slot.R, 1),
])
def test_only_successful_visible_enemy_casts_enter_observer_memory(kind, slot, target):
    """This is witnessed-event memory, not an opponent cooldown side channel."""
    s = _state_with_target()
    s = s.replace(
        # The champions are close, visible, and inside the 1800-unit UI radius.
        x=s.x.at[0].set(0.0).at[1].set(100.0),
        spell_level=s.spell_level.at[0, slot].set(1),
    )
    out = apply_orders(s, _order(kind, target=target))
    assert float(out.observed_enemy_cast_ms[1, slot]) == 0.0
    assert np.all(np.asarray(out.observed_enemy_cast_ms[0]) == -1.0), (
        "a champion must never observe its own cast")
    other = [i for i in range(4) if i != slot]
    assert np.all(np.asarray(out.observed_enemy_cast_ms[1, other]) == -1.0)


def test_fogged_or_off_screen_casts_do_not_enter_observer_memory():
    # Fogged: red's champion is outside blue's 1200 vision and no red ally
    # grants vision on blue's caster.
    s = _state_with_target()
    s = s.replace(
        x=s.x.at[0].set(0.0).at[1].set(1300.0),
        # `_state_with_target` includes a red minion for target tests; remove
        # it here so it cannot supply red vision on the blue caster.
        alive=s.alive.at[5].set(False),
        spell_level=s.spell_level.at[0, Slot.Q].set(1),
    )
    fogged = apply_orders(s, _order(OrderKind.CAST_Q))
    assert np.all(np.asarray(fogged.observed_enemy_cast_ms) == -1.0)

    # A red minion can put blue on the team's minimap, but the red champion is
    # 1801 away.  A cast animation outside its own screen must not leak in.
    s = _state_with_target()
    kind = s.kind.at[2].set(Kind.LANE_MINION)
    team = s.team.at[2].set(Team.RED)
    alive = s.alive.at[2].set(True)
    s = s.replace(
        kind=kind, team=team, alive=alive,
        x=s.x.at[0].set(0.0).at[1].set(1801.0).at[2].set(0.0),
        spell_level=s.spell_level.at[0, Slot.Q].set(1),
    )
    off_screen = apply_orders(s, _order(OrderKind.CAST_Q))
    assert np.all(np.asarray(off_screen.observed_enemy_cast_ms) == -1.0)


def test_failed_spell_orders_do_not_reset_cast_memory():
    s = _state_with_target()
    s = s.replace(x=s.x.at[0].set(0.0).at[1].set(100.0))
    # Q at rank zero is an unavailable order, not a witnessed cast.
    unlearned = apply_orders(s, _order(OrderKind.CAST_Q))
    assert np.all(np.asarray(unlearned.observed_enemy_cast_ms) == -1.0)

    # A learned spell on cooldown is equally a no-op at the ingress boundary.
    s = s.replace(spell_level=s.spell_level.at[0, Slot.Q].set(1),
                  spell_cooldown=s.spell_cooldown.at[0, Slot.Q].set(1.0))
    cooling_down = apply_orders(s, _order(OrderKind.CAST_Q))
    assert np.all(np.asarray(cooling_down.observed_enemy_cast_ms) == -1.0)


def test_observed_cast_clocks_advance_without_destroying_never_seen_sentinel():
    s = _state_with_target().replace(observed_enemy_cast_ms=jnp.asarray([
        [0.0, -1.0, 7.0, -1.0],
        [-1.0, 3.0, -1.0, -1.0],
    ]))
    out = tick(s, lane_params(), delta_ms=25.0, lane_path=None)
    np.testing.assert_allclose(np.asarray(out.observed_enemy_cast_ms), [
        [25.0, -1.0, 32.0, -1.0],
        [-1.0, 28.0, -1.0, -1.0],
    ])
    assert np.all(np.asarray(empty_state().observed_enemy_cast_ms) == -1.0), (
        "a fresh episode/reset must erase all cast-memory history")
    assert np.all(np.asarray(init_lane().observed_enemy_cast_ms) == -1.0), (
        "the trainer's constant episode-reset state must also start unseen")
