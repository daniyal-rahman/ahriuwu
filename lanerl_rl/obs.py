"""Observation builder for the 1v1 Garen lane.

Design rules this module is built to satisfy
============================================

**1. The canonical frame is the LANE frame, not a 180 degree rotation.**
Every position, offset, velocity and heading is expressed as ``(s, n)``:
signed progress from the agent's own outer turret towards the enemy's, and the
perpendicular offset, with the normal pinned so that ``n > 0`` means "away from
both bases, towards the outer wall" for *both* agents.  In a same-lane 1v1 the
two champions share one corridor and enter it from opposite ends, so the map
between the two agents' frames is ``(s, n) -> (L - s, n)``: a **reflection**.
A 180 degree world rotation -- the previous canonicalisation -- maps blue's TOP
lane onto red's BOT lane (``rot180(574, 10220) = (13409, 4227)``, red's bot
outer turret), i.e. onto a different corridor with an oppositely signed normal,
so the two agents were not in fact sampling one distribution.  The chirality
assumption that makes a reflection legal is stated at
:class:`~lanerl_rl.frame.LaneFrame` and re-stated in :meth:`ObservationBuilder.__init__`.

**2. The actor never reads the raw frame.**  The actor path reads exclusively
from :class:`~lanerl_rl.frame.AgentMemory`, and memory is only ever fed with
units that passed the fog gate (``GameObject.IsVisibleByTeam`` server-side via
``vb``/``vr``, or :class:`~lanerl_rl.frame.ApproxFogModel` as a documented
fallback).  A fogged unit therefore cannot leak into the actor observation even
by accident -- the code path that would have to read it does not exist.
``audit.py`` proves this both statically and by differential testing.

**3. Fogged entities get ``valid = 0``.**  A remembered-but-currently-fogged
entity keeps its slot with ``valid = 0``, ``visible_now = 0``, a non-zero
``staleness``, and its *last known* relative position.  It is masked out of the
transformer via ``entity_pad_mask``.  It is never written into a slot with
``valid = 1``: writing ``ds = dn = 0, valid = 1`` for a fogged enemy would tell
the policy "the enemy is standing on top of me", which is the single worst
hallucination available.

**4. Memory lives in the observation builder, not in the recurrent core.**
At 15 Hz a 20 s cooldown is 300 decisions and a 5 s vision gap is 75 -- far
beyond any BPTT window this stack can afford.  Anything the agent must remember
for longer than a burn-in is therefore *computed exactly* here:

* per enemy ability: ``time_since_observed_cast``, ``cd_estimate`` (clamped)
  and a ``never_observed`` flag (see
  :class:`~lanerl_rl.frame.EnemyAbilityIntel`);
* per fogged unit: ``last_seen`` position, ``age_of_estimate``,
  ``last_seen_heading``, and ``reachability_radius = age * move_speed`` -- a
  remembered dot is a disc, and the policy is told the radius;
* self: ``attack_cycle_phase``, ``time_until_next_attack_ready``,
  ``windup_remaining`` (see :class:`AttackClock`).

**5. Every field is plausibly recoverable from a screenshot.**  Health
fractions are quantised to health-bar resolution (1/60).  Health is only
reported at all when the unit is both visible and on screen
(``dist <= SCREEN_RADIUS``); off-screen units are known from the minimap by
position only.  No enemy gold, enemy experience, enemy exact HP, enemy spell
ranks, enemy *readable* cooldowns, server target ids, pathing waypoints or
aggro tables appear anywhere in the actor observation.  Those live in
``priv_vec`` / ``priv_entities``, which are **critic only**.

**6. No slot-index feature.**  Slots are grouped by type and canonically
ordered, but nothing in the entity fields encodes *which* slot a row landed in,
so the attention encoder stays permutation-equivariant within a block.


Exact tensor layout
===================

``entities``: ``(32, 40)`` float32.  Slot blocks::

    [ 0    ]  enemy champion
    [ 1:13]  ally minions,   nearest 12
    [13:25]  enemy minions,  nearest 12, head-sorted by ASCENDING HP
    [25:27]  turrets,        nearest 2 (either team)
    [27:32]  spare: nearest entities not slotted above

32 slots, not 20: measured on ``lanerl/logs/state.jsonl``, the count of live
non-building units within 1500 units of the top-lane meeting point has median
8, p95 15, max 18 -- and that recording had two idle champions, so it
undercounts a contested lane.  See ``constants.N_SLOTS``.

The first ``LAST_HIT_SORT_K = 4`` enemy-minion slots are re-sorted by
**ascending current HP**, so slot 13 is always the last-hit candidate.  The
exploration arithmetic this buys, on the measured 854 ms median last-hit
window: a target head over canonical entity slots lands a last hit with
probability 0.126 under a uniform random policy, against 0.0042 for a raw
position grid -- 30x.  Sorting by HP additionally pins the answer to a *fixed*
slot index, so the reinforcement lands on one action rather than being smeared
over whichever slot the minion happened to occupy.

Honest caveat on that number: 0.126 was computed for an 8-slot target head, and
this table has 32 slots, so uniform-random targeting is 4x more dilute.  The
comparison against a position grid still holds by a wide margin, and the HP sort
is what recovers most of the loss (the answer is one fixed index rather than a
wandering one) -- but if exploration turns out to be the binding constraint, the
lever to pull is a smaller *target* head over a canonical subset, not fewer
entity slots: the slot count is set by how much of the lane the agent can see,
which is a separate measurement (median 8, p95 15, max 18 units within 1500 of
the lane meeting point).

Per-slot fields: see ``constants.ENTITY_FIELD_NAMES``.

``entity_pad_mask``: ``(32,)`` bool, ``True`` where the slot must be masked out
of attention (i.e. ``valid == 0``).  This is the ``key_padding_mask`` for
``nn.TransformerEncoder``.

``self_vec``: ``(64,)`` float32 -- see ``constants.SELF_FIELD_NAMES``.
``global_vec``: ``(48,)`` float32 -- see ``constants.GLOBAL_FIELD_NAMES``.
``priv_vec``: ``(96,)`` float32 -- **critic only**.
``priv_entities``: ``(32, 40)`` float32 -- **critic only**, same layout as
``entities`` but built with fog disabled.


Environment call sites
======================
The builder is self-sufficient except for events only the environment knows it
issued.  ``env.LaneEnv.decode`` calls all three:

    builder.note_cast(slot, t_ms)      # on a `cast` command
    builder.set_recalling(flag)        # on a `recall` command
    builder.note_attack(t_ms)          # on a *targeted* `attack_move` command

``note_attack`` fires only when the order actually reaches the wire as
``{"t":"attack"}``.  An ``attack_move`` with no selected target is sent as a
plain move, no swing is issued, and noting one would invent an attack phase --
so those keep ``attack_timing_known`` at 0, which is the honest answer.

The training reward has moved: ``env.RewardConfig`` / ``env.LaneReward`` are the
old delta-hp/delta-gold sum, and ``reward.ZeroSumLaneReward`` -- which is what
``LaneEnv`` now uses -- replaces them.  See ``reward.py``.
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np

from . import constants as C
from .frame import (
    AgentMemory,
    ApproxFogModel,
    CreepScoreEstimator,
    Frame,
    LaneFrame,
    LaneTransform,
    Unit,
    visible_ids_for,
)

__all__ = [
    "AgentObservation",
    "ActionMask",
    "AbilityBook",
    "AttackClock",
    "ObservationBuilder",
    "ObservationError",
    "OBS_VALUE_LIMIT",
    "obs_checks_enabled",
    "check_observation",
    "relaxed_obs_range",
    "ACTOR_PATH_FUNCTIONS",
    "PRIVILEGED_PATH_FUNCTIONS",
]

log = logging.getLogger("lanerl_rl.obs")

#: Functions that produce the *actor* observation.  ``audit.py`` static-checks
#: exactly these for privileged references.
ACTOR_PATH_FUNCTIONS = (
    "_slot_entities",
    "_fill_entity_row",
    "_build_self_vec",
    "_build_global_vec",
    "_build_action_mask",
    "sync",  # AbilityBook.sync -- the agent's OWN ability bar
)

#: Functions that are allowed to see the unfogged server state.
PRIVILEGED_PATH_FUNCTIONS = ("_build_priv_vec", "_build_priv_entities")


def _quantize_hp(frac: float) -> float:
    """Snap an hp fraction to health-bar resolution (1/60)."""
    frac = 0.0 if frac < 0.0 else (1.0 if frac > 1.0 else frac)
    return round(frac * C.HP_BAR_STEPS) / C.HP_BAR_STEPS


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


# --------------------------------------------------------------------------
# Ability bookkeeping (own kit)
# --------------------------------------------------------------------------


class AbilityBook:
    """The agent's OWN Q/W/E/R ranks and cooldowns.

    Own cooldowns and own ranks are on the agent's own HUD, so reading them off
    the control channel's ``cd0..cd3`` is deployable.  When the frame carries no
    cooldowns, they are tracked from the casts the environment itself issued
    (:meth:`note_cast`) using Garen's base cooldown table.  Ranks follow the
    standard availability rule (Q at 1, W at 2, E at 3, R at 6) unless the
    server reports them.

    The ENEMY's ability bar goes through :class:`~lanerl_rl.frame.EnemyAbilityIntel`
    instead, which only ever records *witnessed casts*.
    """

    SLOTS = C.SPELL_SLOTS
    #: First champion level at which each slot has a point, DERIVED from
    #: C.GAREN_SKILL_ORDER rather than restated. This was a hardcoded
    #: (1, 2, 3, 6) -- Q@1 W@2 E@3 -- which matched neither the server's order
    #: then nor now. Only reached when the wire carries no 'sl'.
    MIN_LEVEL = tuple(
        next((lv + 1 for lv, sl in enumerate(C.GAREN_SKILL_ORDER) if sl == slot),
             C.MAX_LEVEL + 1)
        for slot in range(4)
    )

    def __init__(self) -> None:
        self.ranks = [0, 0, 0, 0]
        self.last_cast_ms: List[Optional[float]] = [None, None, None, None]
        self.remaining_s: Optional[List[Optional[float]]] = None

    def reset(self) -> None:
        self.__init__()

    def note_cast(self, slot: int, t_ms: float) -> None:
        if not 0 <= slot < 4:
            raise IndexError(slot)
        self.last_cast_ms[slot] = float(t_ms)

    def sync(self, unit: Optional[Unit], champ_level: int) -> None:
        """Refresh from the agent's own champion row.

        Audited: the only server-only attributes read here are the agent's
        *own* ``cooldowns`` and ``spell_levels``, both of which are drawn on the
        agent's own HUD.  ``audit.ATTR_EXEMPTIONS`` records that reason.
        """
        if unit is not None and unit.spell_levels is not None:
            self.ranks = [int(r) for r in unit.spell_levels]
        else:
            self.ranks = [1 if champ_level >= m else 0 for m in self.MIN_LEVEL]
        if unit is not None and unit.cooldowns is not None:
            self.remaining_s = [None if c is None else float(c) for c in unit.cooldowns]
        else:
            self.remaining_s = None

    def readiness(self, t_ms: float) -> List[float]:
        """Per-slot readiness in [0, 1]; 1.0 == castable now."""
        out = []
        for i in range(4):
            if self.ranks[i] <= 0:
                out.append(0.0)
                continue
            total = C.GAREN_COOLDOWNS[self.SLOTS[i]][max(0, self.ranks[i] - 1)]
            if self.remaining_s is not None and self.remaining_s[i] is not None:
                rem = max(0.0, self.remaining_s[i])
                out.append(1.0 if rem <= 0.0 else max(0.0, 1.0 - rem / total))
                continue
            last = self.last_cast_ms[i]
            if last is None:
                out.append(1.0)
                continue
            elapsed = (t_ms - last) / 1000.0
            out.append(min(1.0, max(0.0, elapsed / total)))
        return out

    def learned(self) -> List[float]:
        return [1.0 if r > 0 else 0.0 for r in self.ranks]


class AttackClock:
    """Where the agent is in its own auto-attack cycle.

    Three exact quantities, all derived from the server's own content constants
    (``gcd_AttackDelay = 1.6``, ``gcd_AttackDelayCastPercent = 0.3``, Garen's
    ``AttackDelayCastOffsetPercent = -0.091666667`` and
    ``AttackSpeedPerLevel = 2.9``) via :func:`constants.garen_attack_period` and
    :func:`constants.garen_attack_windup`:

    ``attack_cycle_phase``            elapsed / period, clamped to [0, 1]
    ``time_until_next_attack_ready``  (period - elapsed) / period, clamped
    ``windup_remaining``              (windup - elapsed) / windup, clamped

    At level 1 the period is 1.600 s and the windup 0.333 s; at 15 Hz that is 24
    and 5 decisions, so the phase is genuinely actionable -- it is the
    difference between "start the swing now and the minion dies" and "start it
    now and an allied minion steals it".

    Provenance of the swing event
    -----------------------------
    The control channel reports spell cooldowns but not the auto-attack timer,
    so a swing must be *told* to the clock via :meth:`note_attack` -- the
    environment knows, because it issued the order.  Until an attack has been
    noted, ``attack_timing_known`` is 0 and the three timing fields are 0, which
    is honest: no swing has been observed, so there is no phase.  This is the
    one derived feature that needs a call site outside this module; it is wired
    in ``env.LaneEnv.decode``, next to ``builder.note_cast(...)``, on every
    targeted ``attack_move``.
    """

    __slots__ = ("last_attack_ms", "level")

    def __init__(self) -> None:
        self.last_attack_ms: Optional[float] = None
        self.level: int = 1

    def reset(self) -> None:
        self.last_attack_ms = None
        self.level = 1

    def note_attack(self, t_ms: float) -> None:
        self.last_attack_ms = float(t_ms)

    def features(self, t_ms: float, level: int) -> Tuple[float, float, float, float]:
        """``(phase, time_until_ready, windup_remaining, known)``."""
        if self.last_attack_ms is None:
            return 0.0, 0.0, 0.0, 0.0
        period = C.garen_attack_period(level)
        windup = C.garen_attack_windup(level)
        elapsed = max(0.0, (float(t_ms) - self.last_attack_ms) / 1000.0)
        phase = min(elapsed / period, 1.0)
        until = max(0.0, (period - elapsed) / period)
        wind = max(0.0, (windup - elapsed) / windup) if windup > 0.0 else 0.0
        return phase, until, wind, 1.0


# --------------------------------------------------------------------------
# Observation containers
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ActionMask:
    button: np.ndarray  # (N_BUTTONS,) bool, True = legal
    move_x: np.ndarray  # (N_MOVE_BINS,) bool -- lane-parallel (s) component
    move_z: np.ndarray  # (N_MOVE_BINS,) bool -- lane-perpendicular (n) component
    target: np.ndarray  # (N_SLOTS,) bool


@dataclass(slots=True)
class AgentObservation:
    """One agent's view of one tick.

    ``entities`` / ``entity_pad_mask`` / ``self_vec`` / ``global_vec`` are the
    **actor** inputs.  ``priv_entities`` / ``priv_pad_mask`` / ``priv_vec`` are
    **critic only** and must never be routed into the policy trunk; the model
    API keeps them in separate arguments for exactly that reason.
    """

    entities: np.ndarray
    entity_pad_mask: np.ndarray
    self_vec: np.ndarray
    global_vec: np.ndarray
    priv_entities: np.ndarray
    priv_pad_mask: np.ndarray
    priv_vec: np.ndarray
    action_mask: ActionMask
    t_ms: int
    fog_source: str

    def actor_inputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.entities, self.entity_pad_mask, self.self_vec, self.global_vec

    def critic_inputs(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return self.priv_entities, self.priv_pad_mask, self.priv_vec, self.self_vec, self.global_vec


# --------------------------------------------------------------------------
# Runtime guards on the tensors that reach the policy
# --------------------------------------------------------------------------


class ObservationError(ValueError):
    """An observation that must not be allowed to reach the policy.

    Fatal on purpose.  A single NaN in ``entities`` propagates through the
    attention encoder into every head, ``dist.sample()`` then raises somewhere
    with no reference to the field that caused it, and -- worse -- a NaN that
    reaches the optimiser turns the whole network to NaN in one step, at which
    point the run keeps producing updates forever and every metric is quietly
    meaningless.  That is the shape of 13,475 updates of nothing.
    """


#: Nothing in a normalised observation should exceed this in absolute value.
#:
#: The bound is not arbitrary.  Every field is one of: a flag or one-hot (0..1),
#: a cosine/sine (-1..1), a fraction explicitly clamped at 4.0 (gold, CS, hp_abs,
#: minion counts), or a geometric quantity divided by ``NORM_XY``/``NORM_DIST``
#: = 3000 units.  Summoner's Rift is ~15000 units across, so the largest value
#: geometry alone can produce is ~5.  20 leaves 4x headroom over that and still
#: catches the failure this is really for: a RAW world coordinate reaching a
#: field that should have been normalised (12000 / 1 = 12000), which is silent
#: today -- the policy simply saturates and nobody can tell from a loss curve.
#:
#: Measured over 400 decisions of a two-champion, 14-minion lane frame, the
#: largest absolute value produced by any field is 3.0 (in ``entities``), so the
#: margin against a false positive is 6.7x.
OBS_VALUE_LIMIT = 20.0

#: ``LANERL_OBS_STRICT=0`` turns the guards off for a hot rollout loop.
#:
#: Measured on danilogin, same 400-decision frame sequence: ``build()`` takes
#: 535 us with the guards off and 601 us with them on -- 66 us, 12.3%.  The
#: check itself is 40 us; disabled it is 0.4 us, i.e. one ``os.environ`` lookup.
#: Default ON: 12% of an observation build is ~0.05% of a decision at this
#: stack's measured throughput (664 decisions/s against 12 servers, where the
#: learner is 5% of wall time and the simulator is nearly all of it), and the
#: failure it prevents costs a whole run.  ``test_obs_guards.py`` keeps the
#: overhead inside a budget rather than trusting this note.
_STRICT_ENV = "LANERL_OBS_STRICT"

_FIELD_NAMES = {
    "entities": C.ENTITY_FIELD_NAMES,
    "priv_entities": C.ENTITY_FIELD_NAMES,
    "self_vec": C.SELF_FIELD_NAMES,
    "global_vec": C.GLOBAL_FIELD_NAMES,
    "priv_vec": C.PRIV_FIELD_NAMES,
}

_EXPECTED_SHAPES = {
    "entities": (C.N_SLOTS, C.ENTITY_DIM),
    "priv_entities": (C.N_SLOTS, C.ENTITY_DIM),
    "entity_pad_mask": (C.N_SLOTS,),
    "priv_pad_mask": (C.N_SLOTS,),
    "self_vec": (C.SELF_DIM,),
    "global_vec": (C.GLOBAL_DIM,),
    "priv_vec": (C.PRIV_DIM,),
}


#: Set by :func:`relaxed_obs_range` only.  Not an env var: relaxing the range
#: is a property of one call site, not of a process.
_range_limit_override: Optional[float] = None


@contextmanager
def relaxed_obs_range(limit: float = math.inf) -> Iterator[None]:
    """Relax the RANGE check (only) for code that builds impossible frames.

    ``lanerl_rl.audit`` poisons privileged fields with 999,999 gold precisely so
    that a leak into the actor path is unmistakable; that is a value the range
    guard is right to reject everywhere else.  Finiteness, shape and mask
    consistency stay on -- those the audit's frames should still satisfy.
    """
    global _range_limit_override
    prev = _range_limit_override
    _range_limit_override = float(limit)
    try:
        yield
    finally:
        _range_limit_override = prev


def obs_checks_enabled() -> bool:
    """Whether :func:`check_observation` does anything.  Read per call.

    Read per call rather than cached at import so a test can flip it with
    ``monkeypatch.setenv`` -- a module-level constant would make the disabled
    path untestable in the same process, which is how a guard ends up shipped
    with its off switch never exercised.
    """
    return os.environ.get(_STRICT_ENV, "1") != "0"


def _describe(array_name: str, flat_index: int) -> str:
    """``entities[slot 13].hp_frac`` from an array name and a flat index."""
    names = _FIELD_NAMES.get(array_name)
    if names is None:
        return f"{array_name}[{flat_index}]"
    if array_name in ("entities", "priv_entities"):
        slot, field = divmod(int(flat_index), C.ENTITY_DIM)
        return f"{array_name}[slot {slot}].{names[field]} (field {field})"
    return f"{array_name}.{names[int(flat_index)]} (index {flat_index})"


def _check_array(name: str, a: np.ndarray, limit: float) -> None:
    expected = _EXPECTED_SHAPES[name]
    if a.shape != expected:
        raise ObservationError(
            f"{name} has shape {a.shape}, expected {expected}. A shape change here is a "
            f"layout change: the policy would read a different field under every index."
        )
    if a.dtype == np.bool_:
        return
    flat = np.asarray(a).reshape(-1)
    bad = ~np.isfinite(flat)
    if bad.any():
        idx = int(np.flatnonzero(bad)[0])
        n = int(bad.sum())
        raise ObservationError(
            f"{_describe(name, idx)} is {flat[idx]!r} ({n} non-finite value(s) in {name}). "
            f"A NaN here reaches every head through the attention encoder and, once it "
            f"reaches the optimiser, makes the whole network NaN in one step while the "
            f"run carries on producing updates."
        )
    over = np.abs(flat) > limit
    if over.any():
        idx = int(np.flatnonzero(over)[0])
        n = int(over.sum())
        raise ObservationError(
            f"{_describe(name, idx)} is {float(flat[idx]):.6g}, beyond the normalised "
            f"range +-{limit:g} ({n} value(s) over). Geometry alone cannot exceed ~5 "
            f"here, so this is an unnormalised quantity reaching a normalised field."
        )


def check_observation(
    obs: "AgentObservation", where: str = "", limit: Optional[float] = None
) -> None:
    """Validate one observation, loudly and with the offending field named.

    No-op when :func:`obs_checks_enabled` is False.  Raises
    :class:`ObservationError` naming the array, the slot, the field and the
    value -- the point is that the failure message is the diagnosis, not the
    start of one.
    """
    if not obs_checks_enabled():
        return
    if limit is None:
        limit = OBS_VALUE_LIMIT if _range_limit_override is None else _range_limit_override
    try:
        for name in (
            "entities",
            "self_vec",
            "global_vec",
            "priv_entities",
            "priv_vec",
            "entity_pad_mask",
            "priv_pad_mask",
        ):
            _check_array(name, getattr(obs, name), limit)

        # The pad mask IS the attention key_padding_mask. If it disagrees with
        # the valid column, either fogged slots get attended to (hallucinated
        # enemies at ds=dn=0 -- see rule 3 in the module docstring) or real
        # ones get masked out. Both are silent.
        for ent_name, mask_name in (
            ("entities", "entity_pad_mask"),
            ("priv_entities", "priv_pad_mask"),
        ):
            ent = getattr(obs, ent_name)
            mask = getattr(obs, mask_name)
            expected = ent[:, C.E_VALID] < 0.5
            if not np.array_equal(np.asarray(mask, dtype=bool), expected):
                wrong = int(np.flatnonzero(np.asarray(mask, dtype=bool) != expected)[0])
                raise ObservationError(
                    f"{mask_name}[{wrong}] is {bool(mask[wrong])} but "
                    f"{ent_name}[{wrong}].valid is {float(ent[wrong, C.E_VALID])}. The pad "
                    f"mask is the attention key_padding_mask; disagreeing with 'valid' "
                    f"either hides a real entity or attends to a hallucinated one."
                )

        m = obs.action_mask
        for field, width in (
            ("button", C.N_BUTTONS),
            ("move_x", C.N_MOVE_BINS),
            ("move_z", C.N_MOVE_BINS),
            ("target", C.N_SLOTS),
        ):
            arr = getattr(m, field)
            if arr.shape != (width,):
                raise ObservationError(
                    f"action_mask.{field} has shape {arr.shape}, expected {(width,)}"
                )
            if not bool(np.asarray(arr).any()):
                raise ObservationError(
                    f"action_mask.{field} is all False. A fully masked categorical gives "
                    f"log-softmax over nothing, i.e. NaN log-probs for every action -- "
                    f"which is why _build_action_mask force-enables target slot 0."
                )
    except ObservationError as exc:
        log.error("OBSERVATION GUARD FAILED%s: %s", f" ({where})" if where else "", exc)
        raise


# --------------------------------------------------------------------------
# The builder
# --------------------------------------------------------------------------


@dataclass(slots=True)
class _SlotEntity:
    """A candidate for one entity slot, already in lane-local coordinates.

    **Exactly the fields something downstream reads.** This carried ten more --
    ``visible``, ``on_screen``, ``staleness``, ``age_s``, ``hp_d_short``,
    ``hp_d_long``, ``vs``, ``vn``, ``heading``, ``reach_radius`` -- left behind
    when the observation was cut from 1,392 floats to 534. The fields left the
    tensor layout; the code computing them did not, so every one was still
    being computed per candidate per decision and then discarded.

    That was not free. ``velocity`` and ``hp_delta`` each walk a per-unit
    history ring (``UnitMemory._sample_at_or_before``), and filling these ten
    fields cost SIX ring scans per candidate per decision per side. Profiled
    over 27,648 builds: ``_sample_at_or_before`` alone took 8.87M calls and
    10.6 s of a 42.7 s observation build -- 321 scans per single ``build()``,
    the hottest function in the stack by a factor of two, entirely for numbers
    no tensor ever received.

    If a field is added back here, it must be written into ``ENTITY_FIELD_NAMES``
    in the same change, or it is dead weight on the hot path again.
    """

    uid: int
    etype: str
    team_rel: str  # "ally" | "enemy" | "neutral"
    s: float
    n: float
    dist: float
    hp_known: bool
    minion_type: Optional[int]
    hp_frac: float
    mhp: float


class ObservationBuilder:
    """Turns a stream of :class:`~lanerl_rl.frame.Frame` into agent observations.

    One builder per agent.  ``team`` is the server team id (100 BLUE / 200 RED).
    ``lane_own_world`` / ``lane_enemy_world`` are the lane anchors in **raw
    world** coordinates, defaulting to the top-lane outer turrets read out of a
    recorded frame.
    """

    def __init__(
        self,
        team: int,
        lane_own_world: Optional[Sequence[float]] = None,
        lane_enemy_world: Optional[Sequence[float]] = None,
        aa_range: float = C.AA_RANGE_GAREN,
        screen_radius: float = C.SCREEN_RADIUS,
        fog_model: Optional[ApproxFogModel] = None,
        lane_handedness_ref_world: Optional[Sequence[float]] = "default",
        move_speed: float = C.GAREN_MOVE_SPEED,
    ):
        # ------------------------------------------------------------------
        # CHIRALITY ASSUMPTION -- READ THIS BEFORE CHANGING CHAMPION.
        #
        # The canonical frame is the LANE frame, and the map between the two
        # agents' lane frames is (s, n) -> (L - s, n): a REFLECTION, not a
        # rotation.  A reflection swaps left and right.  That is safe here only
        # because Garen's kit is chirally symmetric:
        #
        #     Q  GarenQ   self-buff (empowered next attack)   no direction
        #     W  GarenW   self-shield                          no direction
        #     E  GarenE   spin, radially symmetric AoE         no direction
        #     R  GarenR   point-target execute                 no direction
        #
        # No skillshots, no cones, no directional dashes, no walls.  A reflected
        # policy therefore produces orders that are correct after reflection.
        #
        # For ANY other champion this is WRONG and will silently mirror every
        # aimed ability.  If you change champion, either (a) prove the kit is
        # chirally symmetric and update this comment, or (b) replace LaneFrame
        # with a rotation-based canonicalisation and accept that blue-top maps
        # onto red-bot.  Do not just swap the champion id.
        # ------------------------------------------------------------------
        self.team = int(team)
        self.enemy_team = C.TEAM_RED if self.team == C.TEAM_BLUE else C.TEAM_BLUE
        own = lane_own_world if lane_own_world is not None else C.TOP_OUTER_TURRET[self.team]
        foe = lane_enemy_world if lane_enemy_world is not None else C.TOP_OUTER_TURRET[self.enemy_team]
        if isinstance(lane_handedness_ref_world, str) and lane_handedness_ref_world == "default":
            lane_handedness_ref_world = C.NEXUS_POSITION[self.team]
        self.lane = LaneFrame(own, foe, lane_handedness_ref_world)
        self.transform = LaneTransform(self.lane, self.team)
        self.aa_range = float(aa_range)
        self.screen_radius = float(screen_radius)
        self.move_speed = float(move_speed)
        self.fog_model = fog_model
        self.abilities = AbilityBook()
        self.attack_clock = AttackClock()
        self.memory = AgentMemory()
        self.cs_estimator = CreepScoreEstimator(self.team, aa_range=self.aa_range)
        self.fog_source = "unknown"
        self._aa_damage = 0.0
        self._prev_t_ms: Optional[int] = None
        self._warned_no_ad = False
        self._recalling = False

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        self.abilities.reset()
        self.attack_clock.reset()
        self.memory.reset()
        self.cs_estimator.reset()
        self.fog_source = "unknown"
        self._aa_damage = 0.0
        self._prev_t_ms = None
        self._recalling = False

    def note_cast(self, slot: int, t_ms: float) -> None:
        """Tell the builder the environment just issued an ability cast."""
        self.abilities.note_cast(slot, t_ms)

    def note_attack(self, t_ms: float) -> None:
        """Tell the builder the environment just issued an auto-attack order.

        Wire this next to :meth:`note_cast`; without it the three attack-cycle
        features stay at 0 with ``attack_timing_known = 0``.
        """
        self.attack_clock.note_attack(t_ms)

    def set_recalling(self, flag: bool) -> None:
        self._recalling = bool(flag)

    # -- main entry point --------------------------------------------------

    def build(self, frame: Frame) -> AgentObservation:
        self_u = frame.champion_of_team(self.team)
        enemy_u = frame.champion_of_team(self.enemy_team)
        if self_u is None:
            raise ValueError(f"no champion for team {self.team} in frame t={frame.t_ms}")

        visible, source = visible_ids_for(frame, self.team, self.fog_model)
        self.fog_source = source

        # The enemy's ability bar is only readable when they are visible AND on
        # screen -- a cast animation is not legible from the minimap.
        enemy_on_screen = False
        if enemy_u is not None and enemy_u.id in visible:
            enemy_on_screen = (
                math.hypot(enemy_u.x - self_u.x, enemy_u.y - self_u.y) <= self.screen_radius
            )
        self.memory.update(
            frame,
            visible,
            self_u.id,
            None if enemy_u is None else enemy_u.id,
            enemy_on_screen=enemy_on_screen,
        )
        self.cs_estimator.update(frame, self_u)

        level = int(self_u.lvl or 1)
        self.abilities.sync(self_u, level)
        # The SERVER's attack damage, not a re-derivation of it. Every attempt
        # to recompute this has been wrong by whatever part of the server was
        # forgotten: 57.88 with no runes, 73.14 once the rune page was
        # modelled, against a true 78.14 -- the remainder being Martial Mastery
        # (+5 flat) and Brute Force (+0.55/level) from the mastery page. The
        # wire has carried it since 2026-09-12; use it, and fall back to the
        # derivation only for recordings that predate the field.
        if self_u.ad is None:
            # No guess. The derivation that used to live here was wrong three
            # times running; an unknown AD makes the last-hit potential
            # unavailable, which is honest, and only affects recordings made
            # before the wire carried `ad`.
            if not self._warned_no_ad:
                self._warned_no_ad = True
                log.error(
                    "frame carries no 'ad' for the agent's champion; the "
                    "last-hit shaping potential is unavailable for this "
                    "stream (pre-2026-09-12 recording?)"
                )
            self._aa_damage = 0.0
        else:
            self._aa_damage = float(self_u.ad)

        ax, ay = self.lane.point(self_u.x, self_u.y)

        # Turret positions in lane-local space, for the in-range flags.
        enemy_turrets = self._enemy_turret_positions()
        own_turrets = self._own_turret_positions()

        slots = self._slot_entities(frame.t_ms, ax, ay, self_u.id, visible)
        entities, pad = self._render_slots(slots, ax, ay, enemy_turrets)

        self_vec = self._build_self_vec(frame, self_u, ax, ay, own_turrets, enemy_turrets)
        global_vec = self._build_global_vec(frame, self_u, enemy_u, visible, ax, ay)
        action_mask = self._build_action_mask(frame, self_u, entities)

        priv_slots = self._build_priv_entities(frame, ax, ay, self_u.id)
        priv_entities, priv_pad = self._render_slots(priv_slots, ax, ay, enemy_turrets)
        priv_vec = self._build_priv_vec(frame, self_u, enemy_u, visible, ax, ay, priv_slots)

        # The clock-rewind canary. dt is no longer a FEATURE -- at a lockstep
        # 30 Hz it is ~1.0 by construction, and the network has no use for
        # "how long since my last decision" -- but the DETECTION is worth
        # keeping on its own: this is the check that caught an episode reset
        # failing to reach the builder (t 599,979 -> 16 ms), which had been
        # silently contaminating 2 of every 3 evaluation games with the
        # previous game's unit memory.
        if self._prev_t_ms is not None and frame.t_ms < self._prev_t_ms:
            log.error(
                "game clock went backwards (%d -> %d ms): an episode reset did "
                "not reach the observation builder, so this observation still "
                "carries the previous episode's memory",
                self._prev_t_ms, frame.t_ms,
            )
            self.reset()
        self._prev_t_ms = frame.t_ms

        obs = AgentObservation(
            entities=entities,
            entity_pad_mask=pad,
            self_vec=self_vec,
            global_vec=global_vec,
            priv_entities=priv_entities,
            priv_pad_mask=priv_pad,
            priv_vec=priv_vec,
            action_mask=action_mask,
            t_ms=frame.t_ms,
            fog_source=source,
        )
        check_observation(obs, where=f"team={self.team} t_ms={frame.t_ms}")
        return obs

    # -- helpers shared by both paths --------------------------------------

    def _team_rel(self, team: int) -> str:
        if team == self.team:
            return "ally"
        if team == self.enemy_team:
            return "enemy"
        return "neutral"

    def _enemy_turret_positions(self) -> np.ndarray:
        pts = [
            self.lane.point(m.last_x, m.last_y)
            for m in self.memory.units.values()
            if m.etype == "turret" and m.team == self.enemy_team and m.last_hp_frac > 0.0
        ]
        return np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 2))

    def _own_turret_positions(self) -> np.ndarray:
        pts = [
            self.lane.point(m.last_x, m.last_y)
            for m in self.memory.units.values()
            if m.etype == "turret" and m.team == self.team and m.last_hp_frac > 0.0
        ]
        return np.asarray(pts, dtype=np.float64) if pts else np.zeros((0, 2))

    @staticmethod
    def _min_dist(px: float, py: float, pts: np.ndarray) -> float:
        if pts.shape[0] == 0:
            return float("inf")
        return float(np.sqrt(np.min((pts[:, 0] - px) ** 2 + (pts[:, 1] - py) ** 2)))

    # -- ACTOR PATH --------------------------------------------------------
    # Everything below this line up to the privileged section reads only from
    # ``self.memory``, never from ``frame.units``.

    def _slot_entities(
        self,
        t_ms: int,
        ax: float,
        ay: float,
        self_id: int,
        visible: set,
    ) -> List[Optional[_SlotEntity]]:
        """Fill the 32 slots from the agent's memory.

        Candidates are exclusively memory entries -- i.e. things the agent has
        VISIBLE UNITS ONLY, as of 2026-09-12.

        This used to also slot currently-fogged units, carrying their
        last-known position plus age/heading/reachability so the network could
        reason about where they might be now. Those memory fields are gone --
        the GRU is the memory -- and with them the only thing that made a
        fogged slot safe. A remembered unit emitted with ``valid = 1`` and a
        stale position reads to the network as a live sighting, which is worse
        than not emitting it: it is a confident wrong answer rather than a
        missing one.
        """
        candidates: List[_SlotEntity] = []
        for uid, mem in self.memory.units.items():
            if uid == self_id:
                continue
            if mem.last_hp_frac <= 0.0:
                # We watched it die; a player knows it is gone.
                continue
            # Fog gate FIRST. This used to run last, after the position,
            # velocity, heading and both hp deltas had already been computed --
            # so every fogged unit paid the full cost of a slot it was then
            # dropped from.
            if uid not in visible:
                continue          # fogged: the GRU remembers, the slot does not
            if mem.age_s(float(t_ms)) > C.FORGET_S:
                continue
            cs_, cn_ = self.lane.point(mem.last_x, mem.last_y)
            dist = math.hypot(cs_ - ax, cn_ - ay)
            candidates.append(
                _SlotEntity(
                    uid=uid,
                    etype=mem.etype,
                    team_rel=self._team_rel(mem.team),
                    s=cs_,
                    n=cn_,
                    dist=dist,
                    # Visible by construction (fogged units returned above), so
                    # readable hp is exactly "on screen".
                    hp_known=dist <= self.screen_radius,
                    minion_type=mem.minion_type,
                    hp_frac=mem.last_hp_frac,
                    mhp=mem.last_mhp,
                )
            )
        return self._assign_slots(candidates)

    @staticmethod
    def _assign_slots(candidates: List[_SlotEntity]) -> List[Optional[_SlotEntity]]:
        """Group by (type-block, canonical order) into the fixed 32-slot layout.

        Ties are broken by unit id so the assignment is a pure function of the
        state -- which is what makes the mirroring test able to assert *exact*
        equality.

        The enemy-minion block gets one extra canonicalisation: the nearest
        ``LAST_HIT_SORT_K`` minions are re-sorted by **ascending current HP**,
        so the head of the block is always the minion closest to dying.  That
        makes "last hit" a fixed target-head index instead of a moving one; see
        the module docstring for the exploration arithmetic.  Minions whose HP
        we cannot currently read (fogged, or off screen) sort last within the
        head, because guessing is worse than deferring.
        """

        def key(e: _SlotEntity):
            return (e.dist, e.uid)

        def hp_key(e: _SlotEntity):
            if not e.hp_known:
                return (1, 0.0, e.dist, e.uid)
            return (0, e.hp_frac * e.mhp, e.dist, e.uid)

        enemy_champs = sorted(
            [e for e in candidates if e.etype == "champion" and e.team_rel == "enemy"], key=key
        )
        ally_minions = sorted([e for e in candidates if e.etype == "minion" and e.team_rel == "ally"], key=key)
        enemy_minions = sorted(
            [e for e in candidates if e.etype == "minion" and e.team_rel in ("enemy", "neutral")], key=key
        )
        k = C.LAST_HIT_SORT_K
        enemy_minions = sorted(enemy_minions[:k], key=hp_key) + enemy_minions[k:]
        turrets = sorted([e for e in candidates if e.etype == "turret"], key=key)

        slots: List[Optional[_SlotEntity]] = [None] * C.N_SLOTS
        used = set()

        def place(block: Tuple[int, int], items: List[_SlotEntity]) -> None:
            lo, hi = block
            for i, e in zip(range(lo, hi), items):
                slots[i] = e
                used.add(e.uid)

        place(C.SLOT_ENEMY_CHAMP, enemy_champs)
        place(C.SLOT_ALLY_MINION, ally_minions)
        place(C.SLOT_ENEMY_MINION, enemy_minions)
        place(C.SLOT_TURRET, turrets)

        spare = sorted([e for e in candidates if e.uid not in used], key=key)
        n_spare = C.SLOT_SPARE[1] - C.SLOT_SPARE[0]
        place(C.SLOT_SPARE, spare[:n_spare])
        return slots

    def _fill_entity_row(
        self,
        row: np.ndarray,
        e: _SlotEntity,
        ax: float,
        ay: float,
        enemy_turrets: np.ndarray,
    ) -> None:
        """Write one entity row: where it is, how hurt it is, what it is.

        Deliberately minimal. Everything this used to also write was either a
        function of ds/dn (dist, bearing, in_my_aa_range), a time derivative
        the GRU exists to compute (hp deltas, velocity), fog bookkeeping that
        is now moot because only VISIBLE units are slotted, or an ANSWER
        computed on the network's behalf -- and the answers were where the bugs
        were: ``one_shot_kill_score`` and ``shots_to_kill`` both used a Python
        re-derivation of attack damage that read 57.88 against the server's
        true 78.14, so the agent was told that killable minions were not
        killable.

        ``enemy_turrets`` is retained in the signature because the caller still
        computes it for the reward path; it is intentionally unused here.
        """
        row[C.E_VALID] = 1.0
        row[C.E_DS] = (e.s - ax) / C.NORM_XY
        row[C.E_DN] = (e.n - ay) / C.NORM_XY
        if e.hp_known:
            row[C.E_HP_FRAC] = _quantize_hp(e.hp_frac)
        row[C.E_TYPE_ONEHOT][
            C.ENTITY_TYPE_INDEX.get(e.etype, C.ENTITY_TYPE_INDEX["other"])
        ] = 1.0
        row[C.E_TEAM_ONEHOT][C.ENTITY_TEAMS.index(e.team_rel)] = 1.0
        # WHICH lane minion. All-zero for anything that is not one, and for a
        # SUPER minion, which cannot appear inside ten minutes.
        idx = C.MINION_TYPE_INDEX.get(e.minion_type)
        if idx is not None:
            row[C.E_MINION_SUBTYPE][idx] = 1.0

    def _render_slots(
        self,
        slots: List[Optional[_SlotEntity]],
        ax: float,
        ay: float,
        enemy_turrets: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        out = np.zeros((C.N_SLOTS, C.ENTITY_DIM), dtype=np.float32)
        for i, e in enumerate(slots):
            if e is None:
                continue
            self._fill_entity_row(out[i], e, ax, ay, enemy_turrets)
        pad = out[:, C.E_VALID] < 0.5
        return out, pad

    def _build_self_vec(
        self,
        frame: Frame,
        self_u: Unit,
        ax: float,
        ay: float,
        own_turrets: np.ndarray,
        enemy_turrets: np.ndarray,
    ) -> np.ndarray:
        """Own state.  Everything here is on the agent's own HUD.

        Minimal by design. The level one-hot (18 floats for one integer), the
        second gold/cs encodings, the six region one-hots and the turret-range
        booleans were all either redundant or functions of (lane_s, lane_n);
        the velocity and hp-delta fields are time derivatives the GRU exists to
        compute; the four attack-cycle fields were driven by a Python clock fed
        by orders ISSUED rather than swings landed, and were identically zero
        across all 99,654 behaviour-cloning rows while being live at RL time.

        ``ad/ap/armor/mr`` come off the WIRE. They used to be recomputed here
        via ``constants.garen_attack_damage``, which read 57.88 at level 1
        against the server's true 78.14 -- once for a missing rune page, and
        again for a mastery page nobody had modelled either.
        """
        v = np.zeros(C.SELF_DIM, dtype=np.float32)
        hp_frac = 0.0 if self_u.mhp <= 0 else max(0.0, min(1.0, self_u.hp / self_u.mhp))

        v[C.S_LANE_S] = ax / self.lane.length
        v[C.S_LANE_N] = ay / C.LANE_HALF_WIDTH
        v[C.S_HP_FRAC] = hp_frac

        level = max(1, min(C.MAX_LEVEL, int(self_u.lvl or 1)))
        v[C.S_LEVEL_NORM] = level / C.MAX_LEVEL
        v[C.S_GOLD_NORM] = min(float(self_u.gold or 0.0) / C.NORM_GOLD, 4.0)
        cs = self_u.cs if self_u.cs is not None else self.cs_estimator.cs
        v[C.S_CS_NORM] = min(cs / C.NORM_CS, 4.0)

        # Remaining cooldown as a fraction of the full one: 0 = ready.
        # `readiness` returns 1 when ready, so invert to keep "0 means nothing
        # to think about" consistent across the vector.
        #
        # DELIBERATE CONFLATION, flagged because this project keeps getting
        # bitten by merged states: an UNLEARNED spell also lands on 1.0 here,
        # identical to "just cast". They differ in what resolves them -- a
        # cooldown ticks down, an unlearned spell needs a level -- but they are
        # the same to the only decision that consumes this (can I cast it now),
        # the action mask already forbids casting an unlearned spell, and
        # S_LEVEL_NORM carries "when does R come online". The four *_learned
        # flags the old layout spent on this are not worth four inputs.
        q, w, e, r = self.abilities.readiness(float(frame.t_ms))
        v[C.S_CD_Q], v[C.S_CD_W] = 1.0 - q, 1.0 - w
        v[C.S_CD_E], v[C.S_CD_R] = 1.0 - e, 1.0 - r

        v[C.S_AD] = float(self_u.ad or 0.0) / C.NORM_AD
        v[C.S_AP] = float(self_u.ap or 0.0) / C.NORM_AD
        v[C.S_ARMOR] = float(self_u.armor or 0.0) / C.NORM_AD
        v[C.S_MR] = float(self_u.mr or 0.0) / C.NORM_AD

        v[C.S_IS_DEAD] = 0.0 if self_u.alive else 1.0
        v[C.S_RECALLING] = 1.0 if self._recalling else 0.0
        return v

    def _build_global_vec(
        self,
        frame: Frame,
        self_u: Unit,
        enemy_u: Optional[Unit],
        visible: set,
        ax: float,
        ay: float,
    ) -> np.ndarray:
        """Match-level context.  Six numbers.

        What used to be here and is not any more: six clock-phase one-hots and
        a wave sin/cos (all functions of the clock), kills/deaths/assists
        (~always zero in a 1v1 lane, and death is already a reward term), a
        ten-field model of where the fogged enemy might be, and eight fields of
        enemy-cooldown ESTIMATE.

        The cooldown estimates were built on ``ENEMY_COOLDOWN_ASSUMED``, whose
        own docstring described a safety direction opposite to what the code
        did, on top of a Garen-E cooldown that was wrong at every rank. What is
        genuinely observable is how long ago we watched him cast something, so
        that is what is kept; the network can learn what it implies.

        The fogged enemy's position is not here and cannot be -- nothing in
        this function can reach an invisible unit's live state.
        """
        g = np.zeros(C.GLOBAL_DIM, dtype=np.float32)
        g[C.G_CLOCK_NORM] = min(frame.t_s / C.GAME_LENGTH_S, 2.0)
        g[C.G_ENEMY_VISIBLE] = (
            1.0 if (enemy_u is not None and enemy_u.id in visible) else 0.0
        )
        since, _est, _unknown = self.memory.enemy_intel.features(float(frame.t_ms))
        g[C.G_ENEMY_ABILITY_SINCE_CAST] = since
        return g

    def _build_action_mask(self, frame: Frame, self_u: Unit, entities: np.ndarray) -> ActionMask:
        button = np.zeros(C.N_BUTTONS, dtype=bool)
        button[C.BUTTON_INDEX["noop"]] = True
        alive = self_u.alive
        if alive:
            button[C.BUTTON_INDEX["move"]] = True
            button[C.BUTTON_INDEX["attack_move"]] = True
            button[C.BUTTON_INDEX["recall"]] = True
            ready = self.abilities.readiness(float(frame.t_ms))
            learned = self.abilities.learned()
            for i, name in enumerate(("q", "w", "e", "r")):
                button[C.BUTTON_INDEX[name]] = bool(learned[i] > 0.0 and ready[i] >= 1.0)

        move_x = np.ones(C.N_MOVE_BINS, dtype=bool)
        move_z = np.ones(C.N_MOVE_BINS, dtype=bool)
        target = entities[:, C.E_VALID] > 0.5
        if not target.any():
            # Never hand the policy an all-masked categorical.
            target = target.copy()
            target[0] = True
        return ActionMask(button=button, move_x=move_x, move_z=move_z, target=target)

    # -- PRIVILEGED PATH (critic only) -------------------------------------
    # These may read the raw frame.  Their output must never be routed into
    # the policy trunk.

    def _build_priv_entities(
        self, frame: Frame, ax: float, ay: float, self_id: int
    ) -> List[Optional[_SlotEntity]]:
        """Same slot layout as the actor table, but with fog disabled."""
        candidates: List[_SlotEntity] = []
        for uid, u in frame.units.items():
            if uid == self_id or not u.alive:
                continue
            cs_, cn_ = self.lane.point(u.x, u.y)
            candidates.append(
                _SlotEntity(
                    uid=uid,
                    etype=u.etype,
                    team_rel=self._team_rel(u.team),
                    s=cs_,
                    n=cn_,
                    dist=math.hypot(cs_ - ax, cn_ - ay),
                    hp_known=True,  # fog is off on this path
                    minion_type=u.minion_type,
                    hp_frac=0.0 if u.mhp <= 0 else max(0.0, min(1.0, u.hp / u.mhp)),
                    mhp=u.mhp,
                )
            )
        return self._assign_slots(candidates)

    def _build_priv_vec(
        self,
        frame: Frame,
        self_u: Unit,
        enemy_u: Optional[Unit],
        visible: set,
        ax: float,
        ay: float,
        priv_slots: Optional[List[Optional[_SlotEntity]]] = None,
    ) -> np.ndarray:
        """Unfogged summary for the asymmetric critic."""
        p = np.zeros(C.PRIV_DIM, dtype=np.float32)
        if enemy_u is not None:
            ehp = 0.0 if enemy_u.mhp <= 0 else max(0.0, min(1.0, enemy_u.hp / enemy_u.mhp))
            p[C.P_E_HP_FRAC] = ehp
            p[C.P_E_HP_ABS] = enemy_u.hp / 2000.0
            elvl = max(1, min(C.MAX_LEVEL, int(enemy_u.lvl or 1)))
            p[C.P_E_LEVEL_NORM] = elvl / C.MAX_LEVEL
            p[C.P_E_LEVEL_ONEHOT][elvl - 1] = 1.0
            egold = float(enemy_u.gold or 0.0)
            p[C.P_E_GOLD_NORM] = min(egold / C.NORM_GOLD, 4.0)
            p[C.P_E_GOLD_LOG] = math.log1p(max(egold, 0.0)) / math.log1p(20000.0)
            ecs = enemy_u.cs if enemy_u.cs is not None else 0
            p[C.P_E_CS_NORM] = min(ecs / C.NORM_CS, 4.0)
            es_, en_ = self.lane.point(enemy_u.x, enemy_u.y)
            p[C.P_E_LANE_S] = es_ / self.lane.length
            p[C.P_E_LANE_N] = en_ / C.LANE_HALF_WIDTH
            p[C.P_E_DS] = (es_ - ax) / C.NORM_XY
            p[C.P_E_DN] = (en_ - ay) / C.NORM_XY
            p[C.P_E_DIST] = math.hypot(es_ - ax, en_ - ay) / C.NORM_DIST
            emem = self.memory.units.get(enemy_u.id)
            if emem is not None:
                rvx, rvy = emem.velocity(float(frame.t_ms))
                cvs, cvn = self.lane.vector(rvx, rvy)
                p[C.P_E_VEL_DS] = cvs / C.NORM_VEL
                p[C.P_E_VEL_DN] = cvn / C.NORM_VEL
            p[C.P_E_ALIVE] = 1.0 if enemy_u.alive else 0.0
            p[C.P_E_VISIBLE_TO_ME] = 1.0 if enemy_u.id in visible else 0.0
            if enemy_u.cooldowns is not None:
                for i, cd in enumerate(enemy_u.cooldowns[:4]):
                    base = C.ENEMY_COOLDOWN_ASSUMED_RANK1[i]
                    p[C.P_E_COOLDOWN_TRUE.start + i] = (
                        0.0 if cd is None else max(0.0, min(1.0, float(cd) / base))
                    )

            my_gold = float(self_u.gold or 0.0)
            my_xp = float(self_u.xp or 0.0)
            p[C.P_GOLD_DIFF] = (my_gold - egold) / C.NORM_GOLD
            p[C.P_XP_DIFF] = (my_xp - float(enemy_u.xp or 0.0)) / C.NORM_XP
            p[C.P_LEVEL_DIFF] = (int(self_u.lvl or 1) - elvl) / C.MAX_LEVEL
            my_hp = 0.0 if self_u.mhp <= 0 else max(0.0, min(1.0, self_u.hp / self_u.mhp))
            p[C.P_HP_FRAC_DIFF] = my_hp - ehp
            my_cs = self_u.cs if self_u.cs is not None else self.cs_estimator.cs
            p[C.P_CS_DIFF] = (my_cs - ecs) / C.NORM_CS

        n_ally = n_enemy = 0
        hp_ally = hp_enemy = 0.0
        own_turret_hp = enemy_turret_hp = 0.0
        own_turret_d = enemy_turret_d = float("inf")
        for u in frame.units.values():
            if u.etype == "minion" and u.alive:
                frac = 0.0 if u.mhp <= 0 else u.hp / u.mhp
                if u.team == self.team:
                    n_ally += 1
                    hp_ally += frac
                else:
                    n_enemy += 1
                    hp_enemy += frac
            elif u.etype == "turret":
                cs_, cn_ = self.lane.point(u.x, u.y)
                d = math.hypot(cs_ - ax, cn_ - ay)
                frac = 0.0 if u.mhp <= 0 else max(0.0, min(1.0, u.hp / u.mhp))
                if u.team == self.team and d < own_turret_d:
                    own_turret_d, own_turret_hp = d, frac
                elif u.team != self.team and d < enemy_turret_d:
                    enemy_turret_d, enemy_turret_hp = d, frac
        p[C.P_N_ALLY_MINIONS] = min(n_ally / 8.0, 4.0)
        p[C.P_N_ENEMY_MINIONS] = min(n_enemy / 8.0, 4.0)
        p[C.P_ALLY_MINION_HP] = min(hp_ally / 8.0, 4.0)
        p[C.P_ENEMY_MINION_HP] = min(hp_enemy / 8.0, 4.0)
        p[C.P_OWN_TURRET_HP] = own_turret_hp
        p[C.P_ENEMY_TURRET_HP] = enemy_turret_hp

        p[C.P_MY_LANE_S] = ax / self.lane.length
        p[C.P_MY_LANE_N] = ay / C.LANE_HALF_WIDTH
        p[C.P_MY_HP_FRAC] = 0.0 if self_u.mhp <= 0 else max(0.0, min(1.0, self_u.hp / self_u.mhp))
        p[C.P_MY_GOLD_NORM] = min(float(self_u.gold or 0.0) / C.NORM_GOLD, 4.0)
        p[C.P_MY_XP_NORM] = min(float(self_u.xp or 0.0) / C.NORM_XP, 4.0)
        p[C.P_CLOCK_NORM] = min(frame.t_s / C.GAME_LENGTH_S, 2.0)

        if priv_slots is not None:
            lo, hi = C.SLOT_ENEMY_MINION
            pot = 0.0
            reach = self.aa_range + C.TARGET_RADIUS["minion"] + C.AA_RANGE_EPS
            for e in priv_slots[lo:hi]:
                if e is None or e.team_rel == "ally" or e.dist > reach:
                    continue
                pot += _sigmoid((self._aa_damage - e.hp_frac * e.mhp) / C.AA_KILL_KAPPA_HP)
            # Scale by SHAPING_C, so this really is Phi and not 20x it.
            # reward.last_hit_potential multiplies its sigmoid sum by c = 0.05;
            # this omitted it, so the critic's "value of the shaping potential"
            # was twenty times the quantity the reward actually adds.
            p[C.P_LAST_HIT_POTENTIAL] = C.SHAPING_C * pot
        return p
