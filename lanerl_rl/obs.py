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

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

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
    "ACTOR_PATH_FUNCTIONS",
    "PRIVILEGED_PATH_FUNCTIONS",
]

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
    MIN_LEVEL = (1, 2, 3, 6)

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
# The builder
# --------------------------------------------------------------------------


@dataclass(slots=True)
class _SlotEntity:
    """A candidate for one entity slot, already in lane-local coordinates."""

    uid: int
    etype: str
    team_rel: str  # "ally" | "enemy" | "neutral"
    s: float
    n: float
    dist: float
    visible: bool
    on_screen: bool
    hp_known: bool
    staleness: float
    age_s: float
    hp_frac: float
    mhp: float
    hp_d_short: float
    hp_d_long: float
    vs: float
    vn: float
    heading: Optional[Tuple[float, float]]
    reach_radius: float


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
        self._aa_damage = C.garen_attack_damage(1)
        self._prev_t_ms: Optional[int] = None
        self._recalling = False

    # -- lifecycle ---------------------------------------------------------

    def reset(self) -> None:
        self.abilities.reset()
        self.attack_clock.reset()
        self.memory.reset()
        self.cs_estimator.reset()
        self.fog_source = "unknown"
        self._aa_damage = C.garen_attack_damage(1)
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
        self._aa_damage = C.garen_attack_damage(level)

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

        self._prev_t_ms = frame.t_ms

        return AgentObservation(
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
        actually seen.  Currently-fogged entries are kept (with ``visible``
        False) so that their last-known position, age, heading and reachability
        radius are available, but they are emitted with ``valid = 0`` and masked
        out of attention.
        """
        candidates: List[_SlotEntity] = []
        for uid, mem in self.memory.units.items():
            if uid == self_id:
                continue
            if mem.last_hp_frac <= 0.0:
                # We watched it die; a player knows it is gone.
                continue
            age = mem.age_s(float(t_ms))
            if age > C.FORGET_S:
                continue
            cs_, cn_ = self.lane.point(mem.last_x, mem.last_y)
            vx, vy = mem.velocity(float(t_ms))
            cvs, cvn = self.lane.vector(vx, vy)
            heading = None
            if mem.last_heading is not None:
                hs, hn = self.lane.vector(*mem.last_heading)
                mag = math.hypot(hs, hn)
                if mag > 1e-9:
                    heading = (hs / mag, hn / mag)
            dist = math.hypot(cs_ - ax, cn_ - ay)
            vis = uid in visible
            on_screen = dist <= self.screen_radius
            candidates.append(
                _SlotEntity(
                    uid=uid,
                    etype=mem.etype,
                    team_rel=self._team_rel(mem.team),
                    s=cs_,
                    n=cn_,
                    dist=dist,
                    visible=vis,
                    on_screen=on_screen,
                    hp_known=vis and on_screen,
                    staleness=min(age / C.STALE_HORIZON_S, 1.0),
                    age_s=age,
                    hp_frac=mem.last_hp_frac,
                    mhp=mem.last_mhp,
                    hp_d_short=mem.hp_delta(C.HP_DELTA_SHORT_MS),
                    hp_d_long=mem.hp_delta(C.HP_DELTA_LONG_MS),
                    vs=cvs,
                    vn=cvn,
                    heading=heading,
                    reach_radius=mem.reachability_radius(float(t_ms), self.move_speed),
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
        """Write one 40-field entity row.  Actor path: no server-only fields."""
        ds = e.s - ax
        dn = e.n - ay
        dist = e.dist

        row[C.E_VALID] = 1.0 if e.visible else 0.0
        row[C.E_VISIBLE_NOW] = 1.0 if e.visible else 0.0
        row[C.E_STALENESS] = e.staleness
        row[C.E_TYPE_ONEHOT][C.ENTITY_TYPE_INDEX.get(e.etype, C.ENTITY_TYPE_INDEX["other"])] = 1.0
        row[C.E_TEAM_ONEHOT][C.ENTITY_TEAMS.index(e.team_rel)] = 1.0
        row[C.E_DS] = ds / C.NORM_XY
        row[C.E_DN] = dn / C.NORM_XY
        row[C.E_DIST] = dist / C.NORM_DIST
        if dist > 1e-9:
            row[C.E_COS_BEARING] = ds / dist
            row[C.E_SIN_BEARING] = dn / dist
        if e.hp_known:
            q = _quantize_hp(e.hp_frac)
            row[C.E_HP_FRAC] = q
            row[C.E_HP_DELTA_SHORT] = q - _quantize_hp(e.hp_frac - e.hp_d_short)
            row[C.E_HP_DELTA_LONG] = q - _quantize_hp(e.hp_frac - e.hp_d_long)
            hp_abs = q * e.mhp
            row[C.E_HP_ABS] = min(hp_abs / C.NORM_HP_ABS, 4.0)
            # "Does one auto-attack kill it?"  Both terms are on screen: the
            # health bar gives hp_abs, and the agent's own AD is on its HUD.
            row[C.E_AA_KILLABLE] = _sigmoid((self._aa_damage - hp_abs) / C.AA_KILL_KAPPA_HP)
            shots = math.ceil(hp_abs / max(self._aa_damage, 1e-6))
            row[C.E_AA_SHOTS_TO_KILL] = min(shots / 8.0, 1.0)
        reach = self.aa_range + C.TARGET_RADIUS.get(e.etype, C.TARGET_RADIUS["other"])
        row[C.E_IN_MY_AA_RANGE] = 1.0 if dist <= reach else 0.0
        row[C.E_IN_ENEMY_TURRET_RANGE] = (
            1.0 if self._min_dist(e.s, e.n, enemy_turrets) <= C.TURRET_ATTACK_RANGE else 0.0
        )
        row[C.E_VEL_DS] = e.vs / C.NORM_VEL
        row[C.E_VEL_DN] = e.vn / C.NORM_VEL
        row[C.E_ON_SCREEN] = 1.0 if e.on_screen else 0.0
        row[C.E_HP_KNOWN] = 1.0 if e.hp_known else 0.0
        # Derived memory: how old the estimate is, which way they were going,
        # and how big the disc of possible current positions has grown.
        row[C.E_AGE_S] = min(e.age_s / C.FORGET_S, 1.0)
        if e.heading is not None:
            row[C.E_LAST_HEADING_COS] = e.heading[0]
            row[C.E_LAST_HEADING_SIN] = e.heading[1]
            row[C.E_HEADING_KNOWN] = 1.0
        row[C.E_REACH_RADIUS] = min(e.reach_radius / C.NORM_DIST, 2.0)

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
        """Own state.  Everything here is on the agent's own HUD."""
        v = np.zeros(C.SELF_DIM, dtype=np.float32)
        mem = self.memory.units.get(self_u.id)
        hp_frac = 0.0 if self_u.mhp <= 0 else max(0.0, min(1.0, self_u.hp / self_u.mhp))
        v[C.S_HP_FRAC] = hp_frac
        if mem is not None:
            v[C.S_HP_DELTA_SHORT] = mem.hp_delta(C.HP_DELTA_SHORT_MS)
            v[C.S_HP_DELTA_LONG] = mem.hp_delta(C.HP_DELTA_LONG_MS)

        level = int(self_u.lvl or 1)
        level = max(1, min(C.MAX_LEVEL, level))
        v[C.S_LEVEL_NORM] = level / C.MAX_LEVEL
        v[C.S_LEVEL_ONEHOT][level - 1] = 1.0

        gold = float(self_u.gold or 0.0)
        v[C.S_GOLD_NORM] = min(gold / C.NORM_GOLD, 4.0)
        v[C.S_GOLD_LOG] = math.log1p(max(gold, 0.0)) / math.log1p(20000.0)

        cs = self_u.cs if self_u.cs is not None else self.cs_estimator.cs
        v[C.S_CS_NORM] = min(cs / C.NORM_CS, 4.0)
        v[C.S_CS_LOG] = math.log1p(max(cs, 0)) / math.log1p(500.0)

        ready = self.abilities.readiness(float(frame.t_ms))
        learned = self.abilities.learned()
        v[C.S_Q_READY], v[C.S_W_READY], v[C.S_E_READY], v[C.S_R_READY] = ready
        v[C.S_Q_LEARNED], v[C.S_W_LEARNED], v[C.S_E_LEARNED], v[C.S_R_LEARNED] = learned

        v[C.S_LANE_S] = ax / self.lane.length
        v[C.S_LANE_N] = ay / C.LANE_HALF_WIDTH

        vs = vn = 0.0
        if mem is not None:
            rvx, rvy = mem.velocity(float(frame.t_ms))
            vs, vn = self.lane.vector(rvx, rvy)
        v[C.S_LANE_S_VEL] = vs / C.NORM_VEL
        v[C.S_LANE_N_VEL] = vn / C.NORM_VEL
        speed = math.hypot(vs, vn)
        if speed > 1e-6:
            v[C.S_MOVE_COS] = vs / speed
            v[C.S_MOVE_SIN] = vn / speed
        v[C.S_SPEED_NORM] = min(speed / C.NORM_VEL, 2.0)
        v[C.S_IS_DEAD] = 0.0 if self_u.alive else 1.0

        v[C.S_REGION_ONEHOT][self._region_index(ax / self.lane.length)] = 1.0
        v[C.S_IN_OWN_TURRET_RANGE] = (
            1.0 if self._min_dist(ax, ay, own_turrets) <= C.TURRET_ATTACK_RANGE else 0.0
        )
        v[C.S_IN_ENEMY_TURRET_RANGE] = (
            1.0 if self._min_dist(ax, ay, enemy_turrets) <= C.TURRET_ATTACK_RANGE else 0.0
        )
        v[C.S_RECALLING] = 1.0 if self._recalling else 0.0
        v[C.S_OFF_LANE] = 1.0 if abs(ay) > C.LANE_HALF_WIDTH else 0.0

        phase, until, wind, known = self.attack_clock.features(float(frame.t_ms), level)
        v[C.S_ATTACK_CYCLE_PHASE] = phase
        v[C.S_TIME_UNTIL_NEXT_ATTACK] = until
        v[C.S_WINDUP_REMAINING] = wind
        v[C.S_ATTACK_TIMING_KNOWN] = known
        v[C.S_AA_DAMAGE_NORM] = C.garen_attack_damage(level) / C.NORM_AD
        v[C.S_AA_RANGE_NORM] = self.aa_range / C.NORM_DIST
        return v

    @staticmethod
    def _region_index(s: float) -> int:
        if s < -0.15:
            return 0  # own_fountain (behind own turret, towards own base)
        if s < 0.15:
            return 1  # own_turret_zone
        if s < 0.42:
            return 2  # own_side_lane
        if s < 0.58:
            return 3  # lane_mid
        if s < 0.85:
            return 4  # enemy_side_lane
        return 5  # enemy_turret_zone

    def _build_global_vec(
        self,
        frame: Frame,
        self_u: Unit,
        enemy_u: Optional[Unit],
        visible: set,
        ax: float,
        ay: float,
    ) -> np.ndarray:
        """Match-level context, plus the two pieces of long-horizon memory.

        Note what is here and what is not: the enemy's *last known* position,
        how long ago we saw them, which way they were heading, and how far they
        could have got since -- a player carries all four.  The enemy's current
        position while fogged is not here and cannot be; nothing in this
        function can reach an invisible unit's live state.  Likewise the enemy
        ability block is built from witnessed casts only.
        """
        g = np.zeros(C.GLOBAL_DIM, dtype=np.float32)
        t_s = frame.t_s
        g[C.G_CLOCK_NORM] = min(t_s / C.GAME_LENGTH_S, 2.0)
        phase = 0
        for b in C.CLOCK_PHASE_BOUNDS_S:
            if t_s >= b:
                phase += 1
        g[C.G_CLOCK_PHASE][phase] = 1.0

        wave_theta = 2.0 * math.pi * ((t_s % C.MINION_WAVE_PERIOD_S) / C.MINION_WAVE_PERIOD_S)
        g[C.G_WAVE_SIN] = math.sin(wave_theta)
        g[C.G_WAVE_COS] = math.cos(wave_theta)

        g[C.G_MY_KILLS] = self.memory.kills / 5.0
        g[C.G_MY_DEATHS] = self.memory.deaths / 5.0
        g[C.G_MY_ASSISTS] = self.memory.assists / 5.0

        enemy_visible = enemy_u is not None and enemy_u.id in visible
        g[C.G_ENEMY_VISIBLE] = 1.0 if enemy_visible else 0.0
        unseen_s = max(0.0, (frame.t_ms - self.memory.enemy_last_seen_ms) / 1000.0)
        g[C.G_ENEMY_UNSEEN_TIME] = min(unseen_s / 30.0, 1.0)

        n_ally = sum(
            1
            for uid, m in self.memory.units.items()
            if m.etype == "minion" and m.team == self.team and uid in visible
        )
        n_enemy = sum(
            1
            for uid, m in self.memory.units.items()
            if m.etype == "minion" and m.team != self.team and uid in visible
        )
        g[C.G_N_ALLY_MINIONS] = min(n_ally / 8.0, 2.0)
        g[C.G_N_ENEMY_MINIONS] = min(n_enemy / 8.0, 2.0)
        g[C.G_IS_DEAD] = 0.0 if self_u.alive else 1.0
        dt = C.DECISION_DT_MS if self._prev_t_ms is None else float(frame.t_ms - self._prev_t_ms)
        g[C.G_DT_NORM] = dt / C.DECISION_DT_MS

        # Remembered enemy champion whereabouts.
        if enemy_u is not None:
            emem = self.memory.units.get(enemy_u.id)
            if emem is not None and emem.ever_seen:
                es_, en_ = self.lane.point(emem.last_x, emem.last_y)
                g[C.G_ENEMY_MEM_LANE_S] = es_ / self.lane.length
                g[C.G_ENEMY_MEM_LANE_N] = en_ / C.LANE_HALF_WIDTH
                g[C.G_ENEMY_MEM_DS] = (es_ - ax) / C.NORM_XY
                g[C.G_ENEMY_MEM_DN] = (en_ - ay) / C.NORM_XY
                g[C.G_ENEMY_MEM_VALID] = 1.0
                age = emem.age_s(float(frame.t_ms))
                g[C.G_ENEMY_MEM_AGE] = min(age / C.ENEMY_MEMORY_HORIZON_S, 1.0)
                if emem.last_heading is not None:
                    hs, hn = self.lane.vector(*emem.last_heading)
                    mag = math.hypot(hs, hn)
                    if mag > 1e-9:
                        g[C.G_ENEMY_MEM_HEADING_COS] = hs / mag
                        g[C.G_ENEMY_MEM_HEADING_SIN] = hn / mag
                        g[C.G_ENEMY_MEM_HEADING_KNOWN] = 1.0
                g[C.G_ENEMY_REACH_RADIUS] = min(
                    emem.reachability_radius(float(frame.t_ms), self.move_speed) / C.NORM_DIST, 4.0
                )

        since, est, unknown = self.memory.enemy_intel.features(float(frame.t_ms))
        g[C.G_ENEMY_ABILITY_SINCE_CAST] = since
        g[C.G_ENEMY_ABILITY_CD_EST] = est
        g[C.G_ENEMY_ABILITY_UNKNOWN] = unknown
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
            mem = self.memory.units.get(uid)
            if mem is not None:
                rvx, rvy = mem.velocity(float(frame.t_ms))
            else:
                rvx = rvy = 0.0
            cvs, cvn = self.lane.vector(rvx, rvy)
            hp_frac = 0.0 if u.mhp <= 0 else max(0.0, min(1.0, u.hp / u.mhp))
            heading = None
            if mem is not None and mem.last_heading is not None:
                hs, hn = self.lane.vector(*mem.last_heading)
                mag = math.hypot(hs, hn)
                if mag > 1e-9:
                    heading = (hs / mag, hn / mag)
            candidates.append(
                _SlotEntity(
                    uid=uid,
                    etype=u.etype,
                    team_rel=self._team_rel(u.team),
                    s=cs_,
                    n=cn_,
                    dist=math.hypot(cs_ - ax, cn_ - ay),
                    visible=True,
                    on_screen=True,
                    hp_known=True,
                    staleness=0.0,
                    age_s=0.0,
                    hp_frac=hp_frac,
                    mhp=u.mhp,
                    hp_d_short=0.0 if mem is None else mem.hp_delta(C.HP_DELTA_SHORT_MS),
                    hp_d_long=0.0 if mem is None else mem.hp_delta(C.HP_DELTA_LONG_MS),
                    vs=cvs,
                    vn=cvn,
                    heading=heading,
                    reach_radius=0.0,
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
            p[C.P_LAST_HIT_POTENTIAL] = pot
        return p
