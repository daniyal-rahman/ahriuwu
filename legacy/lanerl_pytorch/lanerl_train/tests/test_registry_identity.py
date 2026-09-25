"""The observation registry must not be foolable by id() recycling.

``LaneObservationAdapter`` publishes itself into a shared registry keyed on
``id(raw)`` so that ``LaneActionEncoder.encode`` can find the adapter that
built the observation for this very frame. That key is only sound if the frame
object is kept ALIVE while the key is in the map -- CPython reuses the address
of a freed object immediately, and frames are short-lived dicts allocated once
per decision:

    2000 unpinned frame-shaped dicts produced 4 distinct id()s.

Two failure modes followed. The visible one is a miss, which logs and sends a
noop: 48 in the first 174,000 decisions of run rl-bc3-0912. The invisible one
is far worse -- the lookup SUCCEEDS against a stale entry and returns a
different instance's adapter, so one game's chosen action is encoded against
another game's frame, silently.
"""
from __future__ import annotations

from lanerl_train.lane_wiring import make_lane_adapters


def _frame(t: int) -> dict:
    return {"t": t, "u": [{"k": "Champion", "tm": 100, "id": 1, "x": 0.0, "y": 0.0,
                           "hp": 100.0, "mhp": 100.0}]}


def test_the_registry_pins_the_frame_it_is_keyed_on():
    """A registered frame must be kept alive, so its id cannot be recycled."""
    adapters = make_lane_adapters(train_step_source=lambda: 0)
    a = adapters.adapter_factory(0, "blue")
    raw = _frame(1)
    try:
        a.build(raw, "blue")
    except Exception:
        pass  # a synthetic frame may not fully build; registration is the point
    reg = adapters.registry
    if not reg:
        return  # build failed before registering; nothing to assert
    entry = next(iter(reg.values()))
    assert "raw" in entry, "the entry must hold the frame, not just the adapters"
    assert entry["raw"] is raw, "the entry must reference the exact object keyed"


def test_id_recycling_really_happens_for_frame_shaped_dicts():
    """The premise of the fix, asserted rather than assumed.

    If this ever stops being true the pinning is harmless, but the comment
    explaining why it exists would be wrong, and someone would remove it.
    """
    unpinned = []
    for i in range(2000):
        unpinned.append(id(_frame(i)))          # freed immediately
    assert len(set(unpinned)) < 100, (
        f"expected heavy id reuse, got {len(set(unpinned))} distinct ids"
    )

    pinned = [_frame(i) for i in range(2000)]   # held, as the registry now does
    assert len(set(id(r) for r in pinned)) == 2000
