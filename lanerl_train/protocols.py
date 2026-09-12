"""The interfaces this package codes against.

``lanerl_rl`` (observation builder, policy, PPO) and the C# server are owned
elsewhere and are moving.  Rather than import their concrete classes and become
a merge conflict, everything here is expressed as a :class:`typing.Protocol`.
Wiring the real implementations in is then a matter of passing an object that
already satisfies the protocol -- for the ones that do not yet, the mismatch
shows up as a type error at the seam rather than as behaviour drift.

Correspondence to what exists today (read-only, for reference):

============================  =====================================================
protocol                      the thing that will implement it
============================  =====================================================
:class:`ObservationAdapter`   ``lanerl_rl.obs.ObservationBuilder`` + ``frame.decode_frame``
:class:`BatchPolicy`          ``lanerl_rl.model.LanePolicy`` (``act`` is already batched)
:class:`ActionEncoder`        ``lanerl_rl.env.decode_action`` -> control-channel JSON
:class:`Learner`              ``lanerl_rl.ppo.DualClipPPO``
============================  =====================================================
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

__all__ = [
    "RawObs",
    "ControlAction",
    "Side",
    "BLUE",
    "RED",
    "SIDES",
    "ObservationAdapter",
    "BatchPolicy",
    "ActionEncoder",
    "Learner",
    "CheckpointStore",
]

#: One decoded observation line from ``LanerlControl.BuildObservation``.
#:
#: ``{"t": <game_ms>, "u": [<unit>, ...]}``, where every unit carries
#: ``id k tm x y hp mhp vb vr`` and a champion adds
#: ``gold xp lvl rc tgt atk mo cd0 cd1 cd2 cd3``.
#:
#: What each field is allowed to reach is **not** documented here -- prose
#: drifts, and this comment already had -- it is enforced by
#: :data:`lanerl_rl.frame.WIRE_FIELDS`, which classifies every key and is
#: cross-checked against the C# source, against ``decode_frame`` and against a
#: generated leak probe by ``python -m lanerl_rl.audit``.  The summary, as of
#: that table:
#:
#: ==================  ==============  =========================================
#: field               disposition     note
#: ==================  ==============  =========================================
#: ``t``               actor-visible   the game clock is on the HUD
#: ``u``               internal        the unit table; the fog gate filters it
#: ``id``              internal        keys memory and the target head's
#:                                     slot->netid map, never a feature
#: ``k`` ``tm``        actor-visible   entity type and ally/enemy relation
#: ``x`` ``y``         actor-visible   fog-gated
#: ``hp`` ``mhp``      actor-visible   quantised to a health bar for the actor,
#:                                     exact for the critic
#: ``vb`` ``vr``       internal        IS the fog gate; not a feature
#: ``gold`` ``xp``     own: actor      own HUD; the enemy's goes to the critic
#: ``lvl`` ``cs``      own: actor      own HUD; the enemy's goes to the critic
#: ``rc``              own: actor      own recall channel is HUD state; the
#:                                     enemy's is privileged
#: ``cd0``..``cd3``    own: actor      own cooldown sweep is on the ability bar;
#:                                     the enemy's VALUE is privileged and only
#:                                     a witnessed cast may move the actor obs
#: ``sl``              own: actor      own ranks are HUD; not emitted today
#: ``tgt``             UNCONSUMED      enemy TargetUnit netid: server truth about
#:                                     intent, invisible on a screenshot, and it
#:                                     survives fog. Nothing decodes it
#: ``atk``             UNCONSUMED      Champion.IsAttacking, order debugging
#: ``mo``              UNCONSUMED      Champion.MoveOrder, order debugging
#: ==================  ==============  =========================================
#:
#: ``vis`` (legacy team list) and ``cd`` (legacy seconds list) are still
#: decoded for old ``LANERL_RECORD`` dumps.  Anything **not** in that table
#: makes ``decode_frame`` raise ``UnknownWireField``.
RawObs = Dict[str, Any]

#: One side's order, exactly as ``LanerlControl.Execute`` reads it:
#: ``{"t": "move"|"attack"|"cast"|"noop", "x","y","id","slot"}``.
ControlAction = Dict[str, Any]

Side = str
BLUE: Side = "blue"
#: ``LanerlControl.ApplyActions`` keys TEAM_PURPLE as ``"red"``.
RED: Side = "red"
SIDES: Tuple[Side, Side] = (BLUE, RED)


@runtime_checkable
class ObservationAdapter(Protocol):
    """Turns a raw control-channel observation into policy input.

    One adapter instance is stateful **per (env, side)** -- the real one carries
    fog memory and an ability book -- so the vec runner owns a grid of them and
    resets the right cells on episode boundaries.
    """

    def reset(self) -> None:
        """Clear per-episode memory.  Called on every episode boundary."""
        ...

    def build(self, raw: RawObs, side: Side) -> Any:
        """Return whatever :meth:`BatchPolicy.act_batch` consumes."""
        ...


@runtime_checkable
class BatchPolicy(Protocol):
    """A policy that is called **once per batch**, never once per env.

    Measured on this stack: 1.70 ms/decision unbatched against 0.058 ms at batch
    24.  Over 16 instances (0.78 ms of simulator wall per decision at 30 Hz and
    43x real time) that is the difference between retaining 31% and 93% of
    simulator throughput, so ``act_batch`` is the only entry point the vec runner
    uses.  See ``vec.py``'s module docstring for the derivation; the "41% and
    96%" this used to quote was a 15 Hz figure computed two different ways.
    """

    @property
    def version(self) -> int:
        """Monotonic parameter version, for staleness accounting."""
        ...

    def initial_state(self, batch: int) -> Any:
        """Recurrent state for ``batch`` independent streams."""
        ...

    def act_batch(
        self,
        observations: Sequence[Any],
        state: Any,
        resets: Optional[Sequence[bool]] = None,
        deterministic: bool = False,
    ) -> Tuple[Sequence[Any], Any]:
        """``(actions, next_state)``, both in the order of ``observations``.

        ``resets[k]`` marks slot ``k`` as the first step of a new episode, so
        the recurrent state for that column must be zeroed *before* the step.
        This is the same signal ``lanerl_rl.model.gru_with_resets`` already
        takes, which is why it rides in the call rather than in a separate
        state-surgery method: slot columns never move, so nothing has to be
        reindexed when one env resets and the others do not.
        """
        ...


@runtime_checkable
class ActionEncoder(Protocol):
    """Policy action -> one control-channel order dict."""

    def encode(self, action: Any, raw: RawObs, side: Side) -> ControlAction:
        ...


@runtime_checkable
class Learner(Protocol):
    """The optimisation half.  ``lanerl_rl.ppo.DualClipPPO`` shaped."""

    def update(self, batch: Any) -> Mapping[str, float]:
        """Consume one rollout batch, return scalar metrics."""
        ...

    def state_payload(self) -> Mapping[str, Any]:
        """Everything needed to resume: weights, optimiser, counters."""
        ...

    def load_payload(self, payload: Mapping[str, Any]) -> None:
        ...

    def policy_payload(self) -> Mapping[str, Any]:
        """Just the weights an actor needs to act (no optimiser state)."""
        ...


@runtime_checkable
class CheckpointStore(Protocol):
    def save(self, name: str, payload: Mapping[str, Any]) -> str:
        ...

    def load(self, name: str) -> Mapping[str, Any]:
        ...

    def latest(self) -> Optional[str]:
        ...
