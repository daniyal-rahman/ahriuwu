"""Wires the real vectorized environment (``lanerl_train.vec``), the real
policy (``lanerl_rl.model.LanePolicy``) and the real learner
(``lanerl_rl.ppo.DualClipPPO``) into the Protocol seams ``lanerl_train.protocols``
declares and ``lanerl_train.vec``/``lanerl_train.run`` actually call.

Nobody had done this before: ``TrainingLoop`` was exercised only against
``FakeInstance`` (``lanerl_train/tests/fakes.py``), and
``lanerl_train.protocols.BatchPolicy``'s own docstring claims ``LanePolicy``
already satisfies it ("act is already batched") when in fact ``LanePolicy.act``
takes one pre-batched tensor dict and owns no external state -- see
:class:`LanePolicyActor` for the actual adapter. Reward computation
(``lanerl_rl.reward.ZeroSumLaneReward``) was never wired into the vectorized
path at all; ``lanerl_train.vec`` computes none.

Nothing in ``lanerl_rl`` or ``lanerl_train.vec`` is modified here; this is glue
only, and it mirrors ``lanerl_rl.env.LaneEnv``'s reference wiring (one real
instance) at the vectorized, many-instance layer instead.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from lanerl_rl import constants as C
from lanerl_rl.env import decode_action, order_for_command
from lanerl_rl.frame import ApproxFogModel, Frame, decode_frame, visible_ids_for
from lanerl_rl.infer import collate_observations
from lanerl_rl.model import LanePolicy, RecurrentState
from lanerl_rl.obs import AgentObservation, ObservationBuilder
from lanerl_rl.ppo import MASK_KEYS, OBS_KEYS, RecurrentRolloutBuffer
from lanerl_rl.reward import LaneRewardConfig, ZeroSumLaneReward

from .protocols import BLUE, RED, RawObs, Side
from .run import EpisodeResult, Rollout
from .vec import VecDriver

log = logging.getLogger("lanerl_train.lane_wiring")

__all__ = [
    "TEAM_OF_SIDE",
    "InstanceRewardContext",
    "LaneObservationAdapter",
    "LaneActionEncoder",
    "LanePolicyActor",
    "LaneAdapters",
    "make_lane_adapters",
    "collect_rollout",
    "make_collect_fn",
]

_ACTION_KEYS = ("button", "move_x", "move_z", "target")
TEAM_OF_SIDE: Dict[Side, int] = {BLUE: C.TEAM_BLUE, RED: C.TEAM_RED}


def _slot_netids_for(builder: ObservationBuilder, frame: Frame, team: int) -> List[Optional[int]]:
    """Mirrors ``lanerl_rl.env.LaneEnv._slot_netids_for`` exactly.

    That method's own comment is the reason this is a free function rather
    than a cached attribute: the slot->netid map must be recomputed from the
    current frame every time, never carried across a fog change, and it must
    live outside the observation so the netid itself never becomes a feature.
    """
    me = frame.champion_of_team(team)
    if me is None:
        return [None] * C.N_SLOTS
    visible, _ = visible_ids_for(frame, team, builder.fog_model)
    ax, ay = builder.transform.point(me.x, me.y)
    slots = builder._slot_entities(frame.t_ms, ax, ay, me.id, visible)
    return [None if e is None else e.uid for e in slots]


class InstanceRewardContext:
    """One ``ZeroSumLaneReward`` per server instance, stepped exactly once per
    tick no matter how many side-adapters ask.

    Reward is scoped to the *instance*, not the side: zero-sum couples the two
    teams' rewards together (``lanerl_rl.reward.ZeroSumLaneReward``), so it
    cannot live on a per-(instance, side) adapter without either stepping it
    twice a tick or silently picking one side to own it.

    ``lanerl_rl.env.LaneEnv`` only ever calls ``reward.step()`` from
    ``LaneEnv.step()`` -- never from ``reset()`` -- because the very first
    frame of an episode is not the *result* of an action; there is nothing to
    attribute a reward to yet, and a potential-based term would otherwise
    score the jump from the previous episode's final state to this episode's
    initial one as if it were a real transition. ``VecDriver`` gives no
    "this frame is a fresh reset" flag to ``ObservationAdapter.build()``
    directly, but it does call ``adapter.reset()`` at exactly the moments that
    matter (construction, and every episode boundary via
    ``_on_new_observations``), so :meth:`mark_reset` -- called from there --
    is what suppresses the one bogus ``reward.step()`` call that would
    otherwise happen on the next ``build()``.
    """

    def __init__(self, reward_cfg: Optional[LaneRewardConfig] = None):
        self.reward = ZeroSumLaneReward(cfg=reward_cfg)
        self._last_raw_id: Optional[int] = None
        self._skip_next = True  # nothing to reward before the first action
        self.last_values: Dict[int, float] = {}
        self.last_info: Dict[str, object] = {}
        self.valid = False  # False on the skipped (reset) frame
        #: The reward of the transition that ENDED the last episode, kept
        #: across the reset that immediately follows it.  The collector cannot
        #: read it any other way: ``VecDriver.step`` steps the reward model on
        #: the terminal frame and then resets the adapters inside the same
        #: call, so by the time ``collect_rollout`` gets control the live
        #: fields are already cleared -- and that row has not been written yet.
        self.terminal_values: Dict[int, float] = {}
        self.terminal_valid = False

    def mark_reset(self) -> None:
        # Guarded because ONE context is shared by both side-adapters of an
        # instance, so a boundary calls this twice. Unguarded, the second call
        # overwrote the snapshot with the cleared state and the terminal reward
        # was lost again -- the exact bug this field exists to fix, reappearing
        # one layer down. `_last_raw_id is None` is precisely "nothing has been
        # seen since the last reset", so the second call is a no-op.
        if self._last_raw_id is not None:
            self.terminal_values = dict(self.last_values)
            self.terminal_valid = self.valid
        self.reward.reset()
        self._last_raw_id = None
        self._skip_next = True
        self.last_values = {}
        self.last_info = {}
        self.valid = False

    def step_once(self, raw_id: int, frame: Frame, train_step: int) -> None:
        if raw_id == self._last_raw_id:
            return
        self._last_raw_id = raw_id
        if self._skip_next:
            self._skip_next = False
            self.last_values = {}
            self.last_info = {}
            self.valid = False
            return
        self.last_values, self.last_info = self.reward.step(frame, train_step)
        self.valid = True


class LaneObservationAdapter:
    """``lanerl_train.protocols.ObservationAdapter`` for one (instance, side).

    Registers itself keyed by ``id(raw)`` so the single shared
    :class:`LaneActionEncoder` can find the *same* stateful
    ``ObservationBuilder`` that built the observation: action decoding needs
    the same slot->netid map the observation was built against, and
    ``note_cast``/``note_attack`` must land on that same builder or its
    ability-readiness and attack-clock features drift from what the agent
    actually did (see ``lanerl_rl.env.LaneEnv.decode``, which this mirrors).

    Safe against ``id()`` reuse: within one ``VecDriver.step()``, ``_forward``
    (which calls :meth:`build`) always completes before ``_scatter`` (which
    calls the encoder) touches the same instance's ``raw``, and
    ``VecLaneEnv.last_obs`` keeps every raw object alive for that whole
    window -- so the id cannot have been recycled in between.

    Each adapter keeps exactly ONE registration and drops its previous one on
    every build, which is the only thing that bounds the registry: nothing
    ever removed an entry, so it grew by ~0.6 entries per decision (measured:
    1,215 entries after 2,048 decisions, the shortfall being ``id`` reuse).
    At the first run's 33M decisions that is on the order of 20M live dict
    entries in the actor thread, for a map whose entries are needed for the
    few microseconds between ``build`` and ``encode``.
    """

    def __init__(
        self,
        side: Side,
        registry: Dict[int, dict],
        reward_ctx: InstanceRewardContext,
        train_step_source,
        fog_model: Optional[ApproxFogModel] = None,
    ):
        self.side = side
        self.team = TEAM_OF_SIDE[side]
        self.builder = ObservationBuilder(self.team, fog_model=fog_model or ApproxFogModel())
        self._registry = registry
        self.reward_ctx = reward_ctx
        self._train_step_source = train_step_source
        self.last_frame: Optional[Frame] = None
        self.last_slot_netids: List[Optional[int]] = [None] * C.N_SLOTS
        self.last_slot_valid = np.zeros(C.N_SLOTS, dtype=bool)
        self._registered_id: Optional[int] = None
        self._awaiting_first_build = True

    def _unregister(self) -> None:
        """Drop this adapter's registry slot.

        reset() used to leave it behind. The last thing built before a reset is
        the TERMINAL observation, and VecDriver then overwrites result.obs[i]
        with the reset frame, so that dict is freed while the registry still
        keys on its address -- a dangling entry that a later frame can be
        allocated on top of.
        """
        if self._registered_id is None:
            return
        entry = self._registry.get(self._registered_id)
        if entry is not None:
            entry["sides"].pop(self.side, None)
            if not entry["sides"]:
                self._registry.pop(self._registered_id, None)
        self._registered_id = None

    def reset(self) -> None:
        self._unregister()
        self.builder.reset()
        #: True from reset() until the next build(). VecDriver sends one action
        #: line on the reset step itself, so encode() legitimately runs with no
        #: frame exactly once per boundary -- that is a noop by design, not the
        #: "impossible" case the error below is for.
        self._awaiting_first_build = True
        self.last_frame = None
        self.last_slot_netids = [None] * C.N_SLOTS
        self.last_slot_valid = np.zeros(C.N_SLOTS, dtype=bool)
        self.reward_ctx.mark_reset()

    def _register(self, raw: RawObs) -> None:
        """Take this adapter's single slot in the shared registry.

        Keyed on ``id(raw)``, which is only safe if the object is kept ALIVE
        for as long as the key is in the map. CPython recycles the id of a
        freed object immediately, and these frames are short-lived dicts
        allocated once per decision, so without a strong reference two
        different instances' frames collide on the same key routinely. The
        visible half of that is a miss -- 48 in the first 174,000 decisions of
        run rl-bc3-0912. The invisible half is worse: the lookup SUCCEEDS and
        hands back another instance's adapter, so one game's action is encoded
        against another game's frame, with no error anywhere.

        Storing the frame alongside the adapters pins the id for the lifetime
        of the entry, so a collision cannot happen, and ``encode`` additionally
        verifies identity rather than trusting the key.
        """
        raw_id = id(raw)
        if self._registered_id is not None and self._registered_id != raw_id:
            entry = self._registry.get(self._registered_id)
            if entry is not None:
                entry["sides"].pop(self.side, None)
                if not entry["sides"]:
                    self._registry.pop(self._registered_id, None)
        self._registered_id = raw_id
        self._registry.setdefault(raw_id, {"raw": raw, "sides": {}})
        # Re-pin: the surviving entry may have been created by the other side
        # for this same frame, which is fine, but it must reference THIS object.
        self._registry[raw_id]["raw"] = raw
        self._registry[raw_id]["sides"][self.side] = self

    @staticmethod
    def builder_entities_valid(obs: AgentObservation):
        """The E_VALID column of the observation, as booleans."""
        return obs.entities[:, C.E_VALID] > 0.5

    def build(self, raw: RawObs, side: Side) -> AgentObservation:
        assert side == self.side, (side, self.side)
        frame = decode_frame(raw)
        me = frame.champion_of_team(self.team)
        if me is not None and me.recalling is not None:
            self.builder.set_recalling(bool(me.recalling))
        obs = self.builder.build(frame)
        self._awaiting_first_build = False
        self.last_frame = frame
        self.last_slot_netids = _slot_netids_for(self.builder, frame, self.team)
        # The validity column decode_action actually reads, taken from the real
        # observation rather than re-derived. A slot can hold a netid and still
        # be INVALID: _slot_entities also slots remembered-but-fogged entities,
        # which carry a uid but E_VALID = 0.
        self.last_slot_valid = self.builder_entities_valid(obs)
        self.reward_ctx.step_once(id(raw), frame, self._train_step_source())
        self._register(raw)
        return obs


class LaneActionEncoder:
    """The single ``ActionEncoder`` shared across every instance and side.

    Looks the calling adapter up by ``id(raw)`` (populated by that adapter's
    own :meth:`LaneObservationAdapter.build` earlier in the same step) so it
    can decode against the exact builder state -- and the exact slot->netid
    map -- the observation was built from, and feed ``note_cast``/
    ``note_attack`` back to it.
    """

    def __init__(
        self,
        registry: Dict[int, dict],
        move_distance: float = 500.0,
    ):
        self._registry = registry
        self.move_distance = float(move_distance)

    def encode(self, action: Any, raw: RawObs, side: Side) -> Dict[str, object]:
        entry = self._registry.get(id(raw))
        if entry is not None and entry["raw"] is not raw:
            # An id collision that the strong reference should have made
            # impossible. Treat it as a miss rather than encoding this action
            # against a different instance's frame.
            log.error(
                "registry id collision on %s: the entry holds a DIFFERENT frame "
                "object. Discarding the action rather than applying it to the "
                "wrong game.", id(raw),
            )
            entry = None
        adapter = (entry or {}).get("sides", {}).get(side)
        if adapter is None or adapter.last_frame is None:
            # Two different situations, and conflating them made the log useless.
            #
            # Expected: VecDriver sends one action line on the reset step, before
            # any build() for the new episode, so there is nothing to encode
            # against. One noop per boundary out of ~18,000 decisions.
            #
            # Impossible: a missing adapter at any OTHER time means encode() and
            # build() disagree about which frame this step is, and the action the
            # policy chose is being thrown away silently.
            #
            # Run 694 logged this 192 times in a 429-line log -- every one of them
            # the expected kind, because end_on_death made an episode end every
            # ~103 s. At ERROR level that buried anything real, and it read like
            # the policy's actions were being dropped wholesale. They were not.
            if adapter is not None and adapter._awaiting_first_build:
                log.debug("encode() on the reset step for side=%s: noop by design", side)
            else:
                # Carry enough to tell the cases apart without another run:
                #   registered_id set and != id(raw) -> the adapter moved on
                #     (build() for a later frame evicted this one): ordering.
                #   registered_id None                -> build() never completed
                #     for this frame (it threw after the builder call).
                #   entry present but side missing    -> another instance owns
                #     this key: an id collision.
                others = self._registry.get(id(raw), {}).get("sides", {})
                log.error(
                    "no registered observation adapter for id(raw)=%s side=%s, and it is "
                    "NOT the reset step -- the chosen action is being DISCARDED. "
                    "entry_present=%s sides_in_entry=%s t_of_raw=%s",
                    id(raw), side, self._registry.get(id(raw)) is not None,
                    sorted(others), raw.get("t"),
                )
            return {"t": "noop"}
        frame = adapter.last_frame
        me = frame.champion_of_team(adapter.team)
        if me is None:
            return {"t": "noop"}
        slot_netids = adapter.last_slot_netids
        entities = np.zeros((C.N_SLOTS, C.ENTITY_DIM), dtype=np.float32)
        entities[:, C.E_VALID] = adapter.last_slot_valid.astype(np.float32)
        # decode_action only ever reads .entities[:, E_VALID] off the
        # observation it is given, so a bare namespace with that one column
        # stands in for a real AgentObservation without rebuilding one -- but
        # the column has to be the REAL one.
        #
        # This used to set E_VALID = 1 for every slot holding a netid, on the
        # stated grounds that the gate is "redundant with slot_netids already
        # being None for an empty slot". That is false: _slot_entities also
        # slots remembered-but-fogged entities, which have a uid and E_VALID = 0.
        # Measured, 25 of 30 frames disagreed on 7 slots. It was inert only
        # because the target action mask enforces the same thing -- except on
        # _build_action_mask's target[0] = True fallback, where this path would
        # have put an attack on a FOGGED enemy champion onto the wire while
        # LaneEnv sent a plain move for the same action. lanerl_rl.audit covers
        # the observation, not the action, so nothing would have caught it.
        fake_obs = SimpleNamespace(entities=entities)
        cmd = decode_action(
            action, adapter.builder, fake_obs, me, slot_netids, move_distance=self.move_distance
        )
        if cmd.kind == "cast" and cmd.spell_slot is not None:
            adapter.builder.note_cast(cmd.spell_slot, frame.t_ms)
        if cmd.kind == "attack_move" and cmd.target_netid is not None:
            adapter.builder.note_attack(frame.t_ms)
        if me.recalling is None:
            adapter.builder.set_recalling(cmd.kind == "recall")
        return order_for_command(cmd)


class LanePolicyActor:
    """``lanerl_train.protocols.BatchPolicy`` for ``lanerl_rl.model.LanePolicy``.

    ``LanePolicy.act`` takes one already-batched tensor dict and owns no
    external state; ``lanerl_rl.infer.BatchedActor`` owns its state
    internally, which is right for a fixed-membership benchmark loop but wrong
    for ``VecDriver``, which keeps one state array *per named policy key* and
    must be able to zero exactly the slots that reset without touching the
    others. This class is the missing middle: state passed in and returned,
    never held by the policy itself.
    """

    def __init__(self, policy: LanePolicy, device: str = "cpu", deterministic: bool = False):
        self.policy = policy
        self.device = torch.device(device)
        self.deterministic = bool(deterministic)
        self._version = 0
        # Stashed for the rollout collector to read right after each
        # act_batch() call: BatchPolicy's return type is fixed by the protocol
        # to (actions, next_state) and cannot carry them directly.
        self.last_log_probs: Optional[torch.Tensor] = None
        self.last_values: Optional[torch.Tensor] = None
        self.last_batch: Optional[Dict[str, Any]] = None
        self.last_actions: Optional[Dict[str, torch.Tensor]] = None  # (N,) long, per key
        self.last_state_in: Optional[RecurrentState] = None  # state ENTERING this call

    @property
    def version(self) -> int:
        return self._version

    def set_version(self, v: int) -> None:
        self._version = int(v)

    def initial_state(self, batch: int) -> RecurrentState:
        return self.policy.initial_state(batch, device=self.device)

    @torch.no_grad()
    def act_batch(
        self,
        observations: Sequence[AgentObservation],
        state: RecurrentState,
        resets: Optional[Sequence[bool]] = None,
        deterministic: bool = False,
    ) -> Tuple[Sequence[Dict[str, int]], RecurrentState]:
        n = len(observations)
        batch = collate_observations(observations, device=self.device)
        reset_t = None
        if resets is not None:
            reset_t = torch.as_tensor(
                np.asarray(resets, dtype=np.float32), device=self.device
            ).reshape(n, 1)
        dist, value, new_state = self.policy.forward(
            entities=batch["entities"],
            entity_pad_mask=batch["entity_pad_mask"],
            self_vec=batch["self_vec"],
            global_vec=batch["global_vec"],
            priv_entities=batch["priv_entities"],
            priv_pad_mask=batch["priv_pad_mask"],
            priv_vec=batch["priv_vec"],
            state=state,
            resets=reset_t,
            action_masks=batch["action_masks"],
        )
        want_det = deterministic or self.deterministic
        action = dist.mode() if want_det else dist.sample()
        self.last_log_probs = dist.log_prob(action)[:, 0].detach()
        self.last_values = value[:, 0].detach()
        self.last_batch = batch
        self.last_actions = {k: action[k][:, 0].detach() for k in _ACTION_KEYS}
        self.last_state_in = state
        actions = [{k: int(action[k][b, 0]) for k in _ACTION_KEYS} for b in range(n)]
        return actions, new_state


@dataclass
class LaneAdapters:
    """Everything one ``VecDriver`` needs beyond the policy itself: the
    per-slot observation adapter factory, the single shared encoder, and the
    per-instance reward contexts a rollout collector reads rewards from."""

    adapter_factory: Any
    encoder: LaneActionEncoder
    reward_contexts: Dict[int, InstanceRewardContext] = field(default_factory=dict)
    registry: Dict[int, dict] = field(default_factory=dict)


def make_lane_adapters(
    train_step_source,
    reward_cfg: Optional[LaneRewardConfig] = None,
    move_distance: float = 500.0,
) -> LaneAdapters:
    """Build the ``adapter_factory``/``encoder`` pair ``VecLaneEnv``/``VecDriver``
    want, plus a lookup of the per-instance reward context so a rollout
    collector can read ``reward_contexts[i].last_values`` after each step.
    """
    registry: Dict[int, dict] = {}
    fog_model = ApproxFogModel()
    reward_contexts: Dict[int, InstanceRewardContext] = {}

    def adapter_factory(i: int, side: Side) -> LaneObservationAdapter:
        ctx = reward_contexts.setdefault(i, InstanceRewardContext(reward_cfg))
        return LaneObservationAdapter(side, registry, ctx, train_step_source, fog_model=fog_model)

    encoder = LaneActionEncoder(registry, move_distance=move_distance)
    return LaneAdapters(
        adapter_factory=adapter_factory,
        encoder=encoder,
        reward_contexts=reward_contexts,
        registry=registry,
    )


# --------------------------------------------------------------------------
# Rollout collection: the piece that turns a real VecDriver + a real policy
# into the Rollout ActorLoop/TrainingLoop actually consume.
# --------------------------------------------------------------------------


def collect_rollout(
    driver: VecDriver,
    actor: LanePolicyActor,
    reward_contexts: Dict[int, InstanceRewardContext],
    policy_key: str,
    num_steps: int,
    gamma: float,
    gae_lambda: float,
    actor_id: int = 0,
    param_version: int = 0,
) -> Rollout:
    """Drive ``driver`` for ``num_steps`` decisions and return one ``Rollout``.

    ``driver.slots[policy_key]`` fixes the (instance, side) -> buffer-row
    order for the whole rollout (``VecDriver.set_assignments`` builds it once
    and never reorders it), so everything :class:`LanePolicyActor` stashes on
    each ``act_batch`` call lines up with it directly.

    Reward timing, and why this holds one extra iteration of state: at the
    top of iteration ``t``, ``driver.step()`` first calls every adapter's
    ``build()`` against the frame already sitting in ``env.last_obs`` -- the
    *result* of iteration ``t-1``'s action, not of the action iteration ``t``
    is about to choose. :class:`InstanceRewardContext` computes reward
    exactly there (see its docstring), so the value available right after
    ``driver.step()`` returns at iteration ``t`` is ``r_{t-1}``: the reward
    for the *previous* iteration's transition. So the (obs, action, log_prob,
    value, state) captured at iteration ``t-1`` is written to the buffer only
    once iteration ``t`` supplies the reward that belongs with it -- never
    at the iteration that produced it. ``num_steps`` decisions therefore
    yield ``num_steps - 1`` buffer rows.

    The ``done`` flag obeys the same lag, and used not to. ``driver.step()``
    at iteration ``t`` returns the boundaries it found in the observation
    *reached by iteration ``t``'s action*, so that flag belongs to the row
    holding iteration ``t``'s (obs, action) -- the row written at iteration
    ``t+1``, not the one written at ``t``. Writing it a row early made GAE
    bootstrap straight through the terminal transition and then cut a
    perfectly ordinary one instead (``compute_gae``: ``dones[t]`` gates
    ``values[t+1]``). Hence ``prev_dones``.
    """
    slots = list(driver.slots[policy_key])
    n = len(slots)
    if n == 0:
        raise ValueError(f"policy key {policy_key!r} drives no slots")
    if num_steps < 2:
        raise ValueError("num_steps must be >= 2 (one iteration is reward-alignment lag)")
    buffer = RecurrentRolloutBuffer(num_steps - 1, n, cfg=actor.policy.cfg, device=actor.device)
    episodes: List[EpisodeResult] = []
    # Per-instance episode accounting, keyed by INSTANCE and carried on the
    # driver so it survives a rollout boundary. Both fields used to be
    # placeholders: length_steps was the ROLLOUT index t (0..num_steps, so it
    # averaged ~half the rollout regardless of the real episode -- 1,372 of
    # 1,670 completed episodes in runs/rl-overnight-0911-0608 look like
    # four-second games for this reason alone, and none of them were), and
    # cs_at_10 was hardcoded None.
    ep_state: Dict[int, Dict[str, Any]] = getattr(driver, "_lanerl_ep_state", None)
    if ep_state is None:
        ep_state = {}
        setattr(driver, "_lanerl_ep_state", ep_state)
    instances = sorted({i for i, _ in slots})

    def _state_for(i: int) -> Dict[str, Any]:
        st = ep_state.get(i)
        if st is None:
            st = {"steps": 0, "ret": {}}
            ep_state[i] = st
        return st

    pending: Optional[Dict[str, Any]] = None
    last_values_now: Optional[torch.Tensor] = None
    # Boundaries found by the PREVIOUS iteration: they belong to the row this
    # iteration is about to write, not to the one it just found.
    prev_dones: Dict[int, str] = {}

    for t in range(num_steps):
        # Snapshot BEFORE step(): _forward() reads _pending_resets to build the
        # resets tensor act_batch() actually saw, then clears it to False for
        # every slot at the end of the same call. Reading it after step() would
        # describe the *next* iteration's resets, not this one's.
        resets_before = list(driver._pending_resets)
        for _i in instances:
            _state_for(_i)["steps"] += 1
        result, dones = driver.step()
        obs_now = actor.last_batch
        log_probs_now = actor.last_log_probs
        values_now = actor.last_values
        actions_now = actor.last_actions
        state_in_now = actor.last_state_in
        last_values_now = values_now

        if pending is not None:
            rewards = torch.zeros(n, dtype=torch.float32, device=actor.device)
            done_t = torch.zeros(n, dtype=torch.float32, device=actor.device)
            for row, (i, side) in enumerate(slots):
                ctx = reward_contexts[i]
                team = TEAM_OF_SIDE[side]
                if i in prev_dones:
                    # The row being written IS the terminal transition, so its
                    # reward is the one computed on the frame the episode ended
                    # on -- not the live one, which belongs to the fresh episode
                    # and is invalid by design on its first frame.
                    done_t[row] = 1.0
                    if ctx.terminal_valid:
                        rewards[row] = float(ctx.terminal_values.get(team, 0.0))
                elif ctx.valid:
                    rewards[row] = float(ctx.last_values.get(team, 0.0))
                    # accumulate the undiscounted episode return, so a run can
                    # be asked "is the agent getting any reward at all?"
                    _state_for(i)["ret"][side] = (
                        _state_for(i)["ret"].get(side, 0.0) + float(rewards[row])
                    )
            buffer.add(
                obs=pending["obs"],
                masks=pending["masks"],
                action=pending["action"],
                log_prob=pending["log_prob"],
                value=pending["value"],
                reward=rewards,
                done=done_t,
                reset=pending["reset"],
                state=pending["state"],
            )

        # Close out the episodes that ended on THIS iteration. Done here rather
        # than inside the `pending` block above: an episode that ends on the
        # first iteration of a rollout has no row to write, and dropping its
        # record also left its step counter running, so the next episode
        # reported the sum of the two (17994 = 2 x 8997, in that run).
        for i, reason in dones.items():
            ctx = reward_contexts[i]
            st = ep_state.pop(i, None)
            ret = dict(st["ret"]) if st else {}
            own_sides = [s for j, s in slots if j == i]
            # The terminal transition's reward is already known -- VecDriver
            # stepped the reward model on the terminal frame before resetting --
            # and it is added here so ep_return covers the whole episode even
            # though its buffer row is written on the next iteration.
            if ctx.terminal_valid:
                for side in own_sides:
                    ret[side] = ret.get(side, 0.0) + float(
                        ctx.terminal_values.get(TEAM_OF_SIDE[side], 0.0)
                    )
            _, cs_by_team = _episode_readout(result.terminal_obs.get(i))
            # CS@10 only means something for an episode that REACHED 10
            # minutes; a death-ended game has not had the chance to farm one.
            cs10 = None
            if reason == "time" and cs_by_team:
                # MEAN of the two sides, not max. In self-play both champions
                # are driven by the SAME policy, so max() reports the better of
                # two draws from one distribution -- an upward-biased estimator
                # of that policy's CS, by about 0.56 sigma for a normal pair
                # (~+3.6 CS at the observed sigma of 6.4). It also shrinks as
                # the spread changes, so it distorts trends as well as levels:
                # a run whose variance grew would look like it was improving.
                # Averaging is unbiased and uses both samples.
                cs10 = float(sum(cs_by_team.values()) / len(cs_by_team))
            # One side's return, not the sum: with both sides driven by the
            # same policy and a zero-sum reward, blue + red is ~0 by
            # construction, which is a number that can never say anything.
            own = own_sides[0] if own_sides else None
            episodes.append(
                EpisodeResult(
                    agent=policy_key,
                    opponent_id=policy_key,
                    opponent_category="self",
                    score=0.5,
                    cs_at_10=cs10,
                    length_steps=int(st["steps"]) if st else 0,
                    ep_return=float(ret.get(own, 0.0)) if own is not None else None,
                    reason=reason,
                    instance=i,
                )
            )
        prev_dones = dict(dones)

        reset_t = torch.tensor(
            [1.0 if resets_before[i] else 0.0 for i, _ in slots],
            dtype=torch.float32,
            device=actor.device,
        )
        # collate_observations stacks with an explicit T=1 axis (the model is
        # written for (B, T, ...)); the buffer stores (T, B, ...) with T being
        # the rollout step, so that per-call axis must be squeezed out here.
        pending = {
            "obs": {k: obs_now[k][:, 0] for k in OBS_KEYS},
            "masks": {k: obs_now["action_masks"][k][:, 0] for k in MASK_KEYS},
            "action": actions_now,
            "log_prob": log_probs_now,
            "value": values_now,
            "reset": reset_t,
            "state": state_in_now,
        }

    if buffer.step == 0:
        raise RuntimeError(
            f"collect_rollout produced zero buffer rows in {num_steps} steps; every "
            f"episode must have ended on the very first iteration for this to happen"
        )
    buffer.finish(last_values_now.to(actor.device), gamma, gae_lambda)

    return Rollout(
        actor_id=actor_id,
        param_version=param_version,
        steps=buffer.step,
        data=buffer,
        episodes=episodes,
        mixture={policy_key: 1.0},
        # One buffer row is one decision in EACH of the n slots. Without this,
        # throughput is reported 2 x envs_per_actor too low, which is how the
        # first run's 3.85M logged "env steps" were really 30.8M decisions.
        parallel_envs=n,
    )


def make_collect_fn(
    build_driver_for_actor: Any,
    policy_key: str,
    rollout_steps: int,
    gamma: float,
    gae_lambda: float,
):
    """Build the single ``collect(actor_id, payload, version) -> Rollout``
    callable :class:`lanerl_train.run.TrainingLoop` shares across every
    :class:`lanerl_train.run.ActorLoop`, each running on its own thread.

    ``build_driver_for_actor(actor_id)`` must return ``(driver, actor,
    reward_contexts)`` for that actor's own real server instances -- called
    once per ``actor_id``, lazily, on that actor's own thread the first time
    it calls ``collect``. Each actor must own an independent set of server
    processes and ports (see ``lanerl_train.ports.PortAllocator``: two actors
    sharing a base collide); dispatching by ``actor_id`` here, rather than
    building one driver up front, is what lets ``build_driver_for_actor``
    give each one a disjoint port block without this function needing to
    know ``num_actors`` itself. Safe across threads because each actor
    thread only ever touches its own ``actor_id``'s entry.
    """
    state: Dict[int, Dict[str, Any]] = {}

    def collect(actor_id: int, payload: Mapping[str, Any], version: int) -> Rollout:
        if actor_id not in state:
            driver, actor, reward_contexts = build_driver_for_actor(actor_id)
            state[actor_id] = {"driver": driver, "actor": actor, "reward_contexts": reward_contexts}
        s = state[actor_id]
        if payload:
            s["actor"].policy.load_state_dict(payload["policy"])
        s["actor"].set_version(version)
        return collect_rollout(
            s["driver"],
            s["actor"],
            s["reward_contexts"],
            policy_key,
            rollout_steps,
            gamma,
            gae_lambda,
            actor_id=actor_id,
            param_version=version,
        )

    return collect


def _episode_readout(raw) -> Tuple[int, Dict[int, int]]:
    """Game time (ms) and per-team champion CS from a raw control-channel obs.

    Returns ``(0, {})`` for anything unreadable rather than raising: a metrics
    readout must never be able to kill a training run.
    """
    if not isinstance(raw, dict):
        return 0, {}
    try:
        t = int(raw.get("t", 0))
        cs: Dict[int, int] = {}
        for u in raw.get("u", ()):
            if u.get("k") == "Champion" and u.get("cs") is not None:
                cs[int(u.get("tm", 0))] = int(u["cs"])
        return t, cs
    except Exception:
        return 0, {}
