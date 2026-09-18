"""Replay a recorded control-channel action at its exact server boundary.

``LanerlControl.OnTick`` builds an observation, reads/executes its action, and
then ``LanerlStateDump.Emit`` records that same boundary.  Object movement for
the interval has already happened before the *next* control boundary.  Thus,
for a one-tick pair ``snapshot[i] -> snapshot[i+1]``, an action whose timestamp
matches ``snapshot[i+1]`` is applied **after** the simulated tick and before
the endpoint comparison.  Applying it before the tick moves a champion one
tick too early -- exactly the dynamic-pathing error this replay exists to
measure.

ActionLog timestamps come from ``(int)GameTime`` (truncate), while the state
dump uses rounded milliseconds.  They may differ by one millisecond, so the
alignment is nearest-within-one, monotonic, unique, and fail-closed.
"""
from __future__ import annotations

from bisect import bisect_left
from dataclasses import dataclass
from typing import Dict, Mapping, Sequence, Tuple

import jax.numpy as jnp

from ..sim.orders import OrderKind, Orders
from .diagnostic_identity import net_id_to_injected_slot
from .record import ActionLog
from .trace import Snapshot

__all__ = [
    "ActionReplayError", "RecordedDecision", "align_action_log",
    "net_id_to_injected_slot", "decision_to_orders",
]


class ActionReplayError(ValueError):
    """A recorded order cannot be aligned or represented without guessing."""


@dataclass(frozen=True, slots=True)
class RecordedDecision:
    source_t_ms: int
    blue: Mapping
    red: Mapping


def align_action_log(trace: Sequence[Snapshot], actions: ActionLog,
                     tolerance_ms: int = 1) -> Dict[int, RecordedDecision]:
    """Map snapshot index -> decision executed immediately before that dump."""
    if tolerance_ms < 0:
        raise ValueError("tolerance_ms must be non-negative")
    times = [s.t_ms for s in trace]
    if any(b <= a for a, b in zip(times, times[1:])):
        raise ActionReplayError("trace snapshot times must be strictly increasing")

    out: Dict[int, RecordedDecision] = {}
    previous_idx = -1
    for t_ms, blue, red in zip(actions.t_ms, actions.blue, actions.red):
        at = bisect_left(times, t_ms)
        candidates = [i for i in (at - 1, at) if 0 <= i < len(times)]
        if not candidates:
            continue
        idx = min(candidates, key=lambda i: (abs(times[i] - t_ms), i))
        distance = abs(times[idx] - t_ms)
        if distance > tolerance_ms:
            # An action log often covers more episode than a sliced trace.  An
            # out-of-range decision is irrelevant; an in-range miss is corrupt.
            if not times or t_ms < times[0] - tolerance_ms or t_ms > times[-1] + tolerance_ms:
                continue
            raise ActionReplayError(
                f"action at {t_ms}ms has no snapshot within {tolerance_ms}ms "
                f"(nearest is {times[idx]}ms)")
        if idx <= previous_idx:
            raise ActionReplayError(
                f"actions are not uniquely monotonic: {t_ms}ms mapped to "
                f"snapshot[{idx}]={times[idx]}ms after index {previous_idx}")
        previous_idx = idx
        out[idx] = RecordedDecision(int(t_ms), blue, red)
    return out


_CAST_KIND = {
    0: OrderKind.CAST_Q,
    1: OrderKind.CAST_W,
    2: OrderKind.CAST_E,
    3: OrderKind.CAST_R,
}


def _decode_wire_order(order: Mapping, ids: Mapping[int, int]) -> Tuple[int, float, float, int]:
    kind = order.get("t", "noop")
    if kind == "noop":
        return OrderKind.NOOP, 0.0, 0.0, -1
    if kind == "move":
        try:
            return OrderKind.MOVE, float(order["x"]), float(order["y"]), -1
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionReplayError(f"malformed recorded move: {order!r}") from exc
    if kind == "attack":
        try:
            net_id = int(order["id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionReplayError(f"malformed recorded attack: {order!r}") from exc
        if net_id not in ids:
            raise ActionReplayError(
                f"attack target NetId {net_id} is absent from diagnostic injection")
        return OrderKind.ATTACK, 0.0, 0.0, ids[net_id]
    if kind == "cast":
        try:
            slot = int(order["slot"])
            sim_kind = _CAST_KIND[slot]
        except (KeyError, TypeError, ValueError) as exc:
            raise ActionReplayError(f"malformed/unsupported recorded cast: {order!r}") from exc
        target = -1
        if slot == 3 and int(order.get("id", 0)):
            net_id = int(order["id"])
            if net_id not in ids:
                raise ActionReplayError(
                    f"R target NetId {net_id} is absent from diagnostic injection")
            target = ids[net_id]
        return sim_kind, float(order.get("x", 0.0)), float(order.get("y", 0.0)), target
    if kind == "recall":
        return OrderKind.RECALL, 0.0, 0.0, -1
    raise ActionReplayError(f"unsupported recorded order kind {kind!r}: {order!r}")


def decision_to_orders(decision: RecordedDecision,
                       net_ids: Mapping[int, int]) -> Orders:
    """Convert blue/red semantic wire orders to the simulator's two slots."""
    decoded = [
        _decode_wire_order(decision.blue, net_ids),
        _decode_wire_order(decision.red, net_ids),
    ]
    return Orders(
        kind=jnp.asarray([v[0] for v in decoded], jnp.int8),
        x=jnp.asarray([v[1] for v in decoded], jnp.float32),
        y=jnp.asarray([v[2] for v in decoded], jnp.float32),
        target=jnp.asarray([v[3] for v in decoded], jnp.int8),
    )
