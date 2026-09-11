"""Static constants for the 1v1 Garen lane RL stack.

Everything here is either (a) read out of the vendored server content, or
(b) a deliberate design choice that is documented inline.  Nothing here is
guessed silently -- where a number is an approximation it says so.

Provenance of the map numbers
-----------------------------
``Content/LeagueSandbox-Default/AIMesh/Map1/AIPath.aimesh_ngrid`` header
(version 3.1) gives::

    MinGridPosition = (-328.903, -67.287, -110.192)
    MaxGridPosition = (14311.771, 184.973, 14556.880)

and ``NavigationGrid`` computes::

    MapWidth  = Max.X + Min.X = 13982.868...
    MapHeight = Max.Z + Min.Z = 14446.688...
    MiddleOfMap = (MapWidth / 2, MapHeight / 2)

``MIRROR_X`` / ``MIRROR_Y`` below are those two values **rounded to
integers**.  That rounding is deliberate: the server recorder emits integer
positions (``(int)au.Position.X``), and rounding the mirror constants to
integers makes the 180 degree rotation an *exact* involution in IEEE-754
(``M - (M - x) == x`` bit-for-bit for integral ``x`` well under 2**53).
The physical error introduced is 0.13 game units, which is ~0.4% of a
champion's pathfinding radius, i.e. irrelevant.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# Decision rate
# --------------------------------------------------------------------------
#
# The server steps its simulation at a fixed 60 Hz (``Game.cs`` drives
# ``REFRESH_RATE``), and ``LanerlControl`` emits one observation every
# ``LANERL_STEP_TICKS`` ticks.  Therefore the ONLY legal decision rates are
# ``60 / k`` for integer ``k >= 1``.  50 Hz -- which an earlier draft of this
# stack used -- is 1.2 ticks per decision and is not expressible at all.
#
# We run at 30 Hz (k=2).  The requested 20 ms / 50 Hz is not expressible; 30 Hz
# is the nearest legal rate that the eventual PIXEL pipeline can actually feed,
# since screen capture realistically runs 30-60 FPS and a decision faster than
# the frame rate has no new observation behind it.
#
# KNOWN TENSION, recorded deliberately: the evidence below argues for 15 Hz, and
# 30 Hz is a doubling of exactly the quantity AlphaStar measured as costing
# -129 Elo.  Measured cost here is compute, not sim speed -- the server+transport
# round trip is only ~0.1-0.5 ms/decision (lanerl_bot/bench/decision_rate.py), so
# rate is paid almost entirely in policy forward passes: ~45 ms of inference per
# game-second at 30 Hz vs ~22 ms at 15 Hz.  If self-play plateaus or the credit
# assignment looks too long, DROPPING TO 15 Hz (STEP_TICKS = 4) IS THE FIRST KNOB
# TO TURN -- it is a one-line change here and costs nothing else.
#
# Why not "as fast as possible":
#
# * AlphaStar (Vinyals et al. 2019, Nature 575, Extended Data Fig. 3) measured
#   the cost of a faster action rate directly: doubling the agent's APS moved
#   it from 1540 Elo to 1411 Elo, i.e. **-129 Elo**.  A higher decision rate
#   lengthens the credit-assignment horizon in steps without adding any
#   information, and the optimiser pays for it.
# * Our own recording says the task does not need reaction speed.  The last-hit
#   window -- the interval during which a minion's health is low enough that one
#   Garen auto-attack kills it, and it is still alive -- has a median of 854 ms
#   and a p90 of 2285 ms.  Garen's auto-attack windup is 333 ms (derived below
#   from the server's own content), so the decision that matters is taken
#   ~0.3 s *before* the kill: last-hitting is a PREDICTION problem at a 0.3 s
#   lead, not a reaction problem.  At 30 Hz a 854 ms window is 25.6 decisions
#   wide (12.8 at 15 Hz) -- either rate has ample room, which is why this
#   argument bounds the rate from BELOW but does not by itself pick 30 over 15.

#: Server simulation rate.  Not configurable; it is the vendored server's.
SERVER_TICK_HZ = 60.0

#: ``LANERL_STEP_TICKS``: server ticks per agent decision.
STEP_TICKS = 2

#: Agent decision rate.  60 / 2 = 30 Hz (33.3 ms).
DECISION_HZ = SERVER_TICK_HZ / STEP_TICKS
DECISION_DT_S = 1.0 / DECISION_HZ
DECISION_DT_MS = 1000.0 * DECISION_DT_S

#: Ceiling on ``global_vec.dt_norm``.  Past a few missed decisions the exact
#: gap is not actionable, and an uncapped ratio is the only field in the actor
#: observation whose healthy range is unbounded -- which makes it the one field
#: a range guard cannot bound.  8 decisions is a quarter of a second at 30 Hz.
DT_NORM_CAP = 8.0

#: Measured on ``lanerl/logs/state.jsonl``: the last-hit window in milliseconds.
LAST_HIT_WINDOW_MEDIAN_MS = 854.0
LAST_HIT_WINDOW_P90_MS = 2285.0

#: Default discount horizon, in SECONDS.  gamma is derived from this and the
#: decision rate, never configured raw -- a raw gamma silently means a
#: different amount of time whenever the decision rate changes.
DEFAULT_HORIZON_S = 30.0


def legal_step_ticks(decision_hz: float, tol: float = 1e-6) -> int:
    """Return ``LANERL_STEP_TICKS`` for ``decision_hz``, or raise.

    Only ``60 / k`` rates exist.  Anything else is a request the server cannot
    honour, and silently rounding it would desynchronise the agent's notion of
    a timestep from the simulator's.
    """
    k = SERVER_TICK_HZ / float(decision_hz)
    if k < 1.0 - tol or abs(k - round(k)) > tol:
        legal = ", ".join(f"{SERVER_TICK_HZ / n:g}" for n in range(1, 13))
        raise ValueError(
            f"decision rate {decision_hz:g} Hz is not expressible on a "
            f"{SERVER_TICK_HZ:g} Hz server: it needs {k:.4g} ticks per decision. "
            f"Legal rates are 60/k Hz: {legal}, ..."
        )
    return int(round(k))


def gamma_for_horizon(horizon_s: float, decision_hz: float = DECISION_HZ) -> float:
    """Discount factor whose 1/(1-gamma) horizon is ``horizon_s`` seconds.

    ``gamma = 1 - 1 / (horizon_s * decision_hz)``.  At 15 Hz::

        30 s -> 0.997778
        45 s -> 0.998519

    Configure the horizon; let gamma follow.  Copying a gamma across a change
    of decision rate is the classic way to accidentally change the objective.
    """
    if horizon_s <= 0.0:
        raise ValueError("horizon_s must be > 0")
    steps = horizon_s * float(decision_hz)
    if steps <= 1.0:
        raise ValueError(f"horizon {horizon_s}s at {decision_hz}Hz is {steps} steps; too short")
    return 1.0 - 1.0 / steps


def horizon_for_gamma(gamma: float, decision_hz: float = DECISION_HZ) -> float:
    """Inverse of :func:`gamma_for_horizon`, in seconds."""
    if not 0.0 < gamma < 1.0:
        raise ValueError("gamma must be in (0, 1)")
    return 1.0 / ((1.0 - gamma) * float(decision_hz))


# --------------------------------------------------------------------------
# Map geometry
# --------------------------------------------------------------------------

#: Navgrid-derived map extent (exact float values, for reference / diagnostics).
MAP_WIDTH_EXACT = 13982.868469238281
MAP_HEIGHT_EXACT = 14446.68758392334

#: Integer mirror constants.  ``rot180(x, y) = (MIRROR_X - x, MIRROR_Y - y)``.
MIRROR_X = 13983
MIRROR_Y = 14447

#: Team ids as the server writes them into the JSONL (``(int)au.Team``).
TEAM_BLUE = 100
TEAM_RED = 200  # server calls it PURPLE internally
TEAM_NEUTRAL = 300

TEAM_NAMES = {TEAM_BLUE: "BLUE", TEAM_RED: "RED", TEAM_NEUTRAL: "NEUTRAL"}

#: Top-lane outer turret positions, in raw world coordinates, read out of a
#: recorded frame (``lanerl/logs/state.jsonl``, t=0) by picking, per team, the
#: top-lane ``LaneTurret`` furthest from that team's Nexus.
TOP_OUTER_TURRET = {
    TEAM_BLUE: (574.0, 10220.0),
    TEAM_RED: (3911.0, 13654.0),
}

#: Nexus positions, raw world coordinates, read out of a recorded frame.
#: Used as the handedness reference for :class:`~lanerl_rl.frame.LaneFrame`.
NEXUS_POSITION = {
    TEAM_BLUE: (1131.0, 1426.0),
    TEAM_RED: (12760.0, 13026.0),
}

#: Half-width used to normalise the perpendicular lane coordinate ``n``.
LANE_HALF_WIDTH = 1400.0

# --------------------------------------------------------------------------
# Unit properties (from Content/LeagueSandbox-Default/Stats)
# --------------------------------------------------------------------------

#: ``ObjAIBase`` falls back to 1100 when CharData.PerceptionBubbleRadius == 0,
#: which is the case for Garen (its stats file has no PerceptionBubbleRadius).
VISION_RADIUS_CHAMPION = 1100.0
#: SRU_*MinionMelee/Ranged all declare PerceptionBubbleRadius = 1200.
VISION_RADIUS_MINION = 1200.0
#: ``BaseTurret`` ctor passes ``visionRadius: 800``.
VISION_RADIUS_TURRET = 800.0
#: Buildings (inhib/nexus).  Not read from content; a documented approximation
#: only used by the *fallback* fog model.
VISION_RADIUS_BUILDING = 1350.0

#: Garen.json Data.AttackRange.
AA_RANGE_GAREN = 125.0
#: Garen.json Data.MoveSpeed.  Used for the fogged-enemy reachability radius.
GAREN_MOVE_SPEED = 345.0
#: OrderTurretNormal / SRUAP_Turret_*.json Data.AttackRange.
TURRET_ATTACK_RANGE = 750.0

# -- auto-attack timing, derived exactly from the server's own content ------
#
#   Maps/Map1/Constants.json : gcd_AttackDelay = 1.6, gcd_AttackDelayCastPercent = 0.3
#   Stats/Garen/Garen.json   : AttackDelayOffsetPercent = 0,
#                              AttackDelayCastOffsetPercent = -0.091666667,
#                              AttackSpeedPerLevel = 2.9 (percent)
#
#   Stats.cs:130   AttackSpeedFlat = 1 / AttackDelay / (1 + AttackDelayOffsetPercent)
#   Spell.cs:508   totalTime = AttackDelay * (1 + AttackDelayOffsetPercent)
#   Spell.cs:509   castTime  = totalTime * (AttackDelayCastPercent + AttackDelayCastOffsetPercent)
#
GLOBAL_ATTACK_DELAY_S = 1.6
GLOBAL_ATTACK_DELAY_CAST_PERCENT = 0.3
GAREN_ATTACK_DELAY_OFFSET_PERCENT = 0.0
GAREN_ATTACK_DELAY_CAST_OFFSET_PERCENT = -0.091666667
GAREN_ATTACK_SPEED_PER_LEVEL = 2.9  # percent, Stats.cs:276 GrowthAttackSpeed / 100

#: Attack period and windup at level 1.  1.6 * (0.3 - 0.0916667) = 0.3333 s.
GAREN_BASE_ATTACK_PERIOD_S = GLOBAL_ATTACK_DELAY_S * (1.0 + GAREN_ATTACK_DELAY_OFFSET_PERCENT)
GAREN_BASE_WINDUP_S = GAREN_BASE_ATTACK_PERIOD_S * (
    GLOBAL_ATTACK_DELAY_CAST_PERCENT + GAREN_ATTACK_DELAY_CAST_OFFSET_PERCENT
)

#: Garen.json Data.BaseDamage / DamagePerLevel.
GAREN_BASE_AD = 57.88
GAREN_AD_PER_LEVEL = 3.5


def _level_growth_multiplier(level: int) -> float:
    """``Stats.GetLevelUpStatValue``: ``value * (0.65 + 0.035 * Level)``.

    ``Level`` there is the level *after* the increment, so the cumulative bonus
    at level L is ``per_level * sum_{k=2..L} (0.65 + 0.035 k)``.
    """
    return sum(0.65 + 0.035 * k for k in range(2, int(level) + 1))


def garen_attack_damage(level: int) -> float:
    """Garen's total AD at ``level``, exactly as ``Stats.LevelUp`` computes it."""
    level = max(1, min(MAX_LEVEL, int(level)))
    return GAREN_BASE_AD + GAREN_AD_PER_LEVEL * _level_growth_multiplier(level)


def garen_attack_period(level: int) -> float:
    """Seconds between Garen's auto-attacks at ``level`` (no items/buffs)."""
    level = max(1, min(MAX_LEVEL, int(level)))
    bonus = (GAREN_ATTACK_SPEED_PER_LEVEL / 100.0) * _level_growth_multiplier(level)
    return GAREN_BASE_ATTACK_PERIOD_S / (1.0 + bonus)


def garen_attack_windup(level: int) -> float:
    """Seconds of windup before Garen's auto-attack damage lands, at ``level``."""
    return garen_attack_period(level) * (
        GLOBAL_ATTACK_DELAY_CAST_PERCENT + GAREN_ATTACK_DELAY_CAST_OFFSET_PERCENT
    )


#: Ambient ("passive") gold, ``Maps/Map1/Constants.json``::
#:
#:     ai_AmbientGoldAmount   = 9.5
#:     ai_AmbientGoldInterval = 5.0     -> 1.9 gold/s = 19 gold / 10 s
#:     ai_AmbientGoldDelay    = 90.0
#:
#: Over the first ten minutes that is (600 - 90) * 1.9 = 969 gold that arrives
#: whatever the policy does.  Against ~1200 gold from a competent 60-CS ten
#: minutes it is ~45% of the total gold delta -- so ``delta(total gold)`` is
#: nearly half noise, and must not be used as a reward.  ``ai_AmbientXPAmount``
#: is 0.0 on Map1, so experience needs no such correction.
AMBIENT_GOLD_AMOUNT = 9.5
AMBIENT_GOLD_INTERVAL_S = 5.0
AMBIENT_GOLD_PER_S = AMBIENT_GOLD_AMOUNT / AMBIENT_GOLD_INTERVAL_S
AMBIENT_GOLD_DELAY_S = 90.0

#: Effective "target radius" added to attack-range checks, by entity type.
#: Garen.json PathfindingCollisionRadius = 35, SelectionRadius = 75.  The
#: server's in-range test is centre-to-centre plus the target's collision
#: radius; these are the collision radii we assume.
TARGET_RADIUS = {
    "champion": 35.0,
    "minion": 25.0,
    "turret": 88.0,
    "inhibitor": 100.0,
    "nexus": 100.0,
    "other": 40.0,
}

#: Garen ability base cooldowns, per rank, from Content/.../Spells/Garen{Q,W,E,R}.
GAREN_COOLDOWNS = {
    "Q": (8.0, 8.0, 8.0, 8.0, 8.0),
    "W": (24.0, 23.0, 22.0, 21.0, 20.0),
    "E": (9.0, 9.0, 9.0, 9.0, 9.0),
    "R": (160.0, 120.0, 80.0, 80.0, 80.0),
}
SPELL_SLOTS = ("Q", "W", "E", "R")

#: Rank-1 cooldowns, used as the *estimate* denominator for an ENEMY ability.
#: We deliberately never read the enemy's spell ranks (a rank read would be a
#: leak): Q and E are rank-invariant so the estimate is exact for them, and for
#: W and R rank 1 over-estimates the remaining cooldown, which is the safe
#: direction (the agent assumes the enemy is readier than it is).
ENEMY_COOLDOWN_ASSUMED_RANK1 = tuple(GAREN_COOLDOWNS[s][0] for s in SPELL_SLOTS)

# --------------------------------------------------------------------------
# Perception / degradation model
# --------------------------------------------------------------------------

#: Health bars are ~60 px wide at the default client resolution.  Every entity
#: hp fraction is snapped to this grid so that no sub-bar precision -- which a
#: screenshot could never give us -- can be learned from.
HP_BAR_STEPS = 60

#: Radius (game units) inside which a unit is assumed to be on the player's
#: screen at default camera zoom.  Position of a *visible* unit is recoverable
#: off-screen from the minimap; its health bar is not.  Fields that need a
#: health bar are gated on this.
SCREEN_RADIUS = 1800.0

#: Seconds after which a remembered (fogged) unit is fully stale.
STALE_HORIZON_S = 5.0
#: Seconds after which a remembered unit is dropped from memory entirely.
FORGET_S = 8.0
#: Seconds over which the remembered *champion* estimate is normalised.  A
#: player keeps thinking about where the enemy went for far longer than they
#: think about a minion.
ENEMY_MEMORY_HORIZON_S = 60.0

#: Window used for finite-difference velocity estimates.
VEL_WINDOW_MS = 200.0
#: HP-delta lookback windows required by the spec.
HP_DELTA_SHORT_MS = 500.0
HP_DELTA_LONG_MS = 2000.0

#: A rise in an observed enemy cooldown larger than this means "they just cast".
#: Garen's shortest cooldown is 8 s, so 1 s is far below any real cast and far
#: above float noise / the 1 ms quantisation of the control channel.
CAST_DETECT_RISE_S = 1.0
#: Two consecutive sightings must be at most this far apart for a cooldown rise
#: between them to count as a *witnessed* cast.  One decision step is 66.7 ms;
#: 300 ms allows a couple of dropped frames and nothing more.  Without this the
#: agent would be inferring casts it never saw, from a number no screenshot
#: shows.
CAST_OBSERVE_MAX_GAP_MS = 300.0

# --------------------------------------------------------------------------
# Observation tensor shapes
# --------------------------------------------------------------------------
#
# 32 entity slots, not 20.  Measured on ``lanerl/logs/state.jsonl`` (4055 frames
# after the first wave spawns), counting live non-building units within a radius
# of the top-lane meeting point::
#
#     r = 1500:  median  8, p90 14, p95 15, p99 17, max 18
#     r = 2000:  median 10, p90 15, p95 16, p99 18, max 18
#
# and that recording has two IDLE champions, so it undercounts a real game
# (no wave crashes, no freezes, no double waves).  20 slots -- 8 per minion side
# -- is already marginal at a wave collision; a stacked double wave is 12 a side.
N_SLOTS = 32
ENTITY_DIM = 40
SELF_DIM = 64
GLOBAL_DIM = 48
PRIV_DIM = 96

#: Slot blocks.  Ordering is (type-block, then a per-block sort key).  There is
#: deliberately **no** slot-index feature: the encoder must stay permutation
#: equivariant within a block.
SLOT_ENEMY_CHAMP = (0, 1)
SLOT_ALLY_MINION = (1, 13)
SLOT_ENEMY_MINION = (13, 25)
SLOT_TURRET = (25, 27)
SLOT_SPARE = (27, 32)

#: How many of the nearest enemy minions get re-sorted by ascending HP so that
#: the head of the enemy-minion block is always the last-hit candidate.  See
#: ``obs.ObservationBuilder._assign_slots``.
LAST_HIT_SORT_K = 4

#: Entity type one-hot ordering.
ENTITY_TYPES = ("champion", "minion", "turret", "inhibitor", "nexus", "other")
N_ENTITY_TYPES = len(ENTITY_TYPES)
ENTITY_TYPE_INDEX = {name: i for i, name in enumerate(ENTITY_TYPES)}

#: Team one-hot ordering, *relative to the observing agent*.
ENTITY_TEAMS = ("ally", "enemy", "neutral")
N_ENTITY_TEAMS = len(ENTITY_TEAMS)

#: Server ``k`` (GetType().Name) -> our coarse entity type.
KIND_TO_TYPE = {
    "Champion": "champion",
    "LaneMinion": "minion",
    "Minion": "minion",
    "LaneTurret": "turret",
    "BaseTurret": "turret",
    "Inhibitor": "inhibitor",
    "Nexus": "nexus",
    "Monster": "other",
    "Placeable": "other",
}

# Normalisation scales.
NORM_XY = 3000.0
NORM_DIST = 3000.0
NORM_VEL = 600.0  # units/s; Garen base MoveSpeed is 345

#: Teleport detection for the finite-difference velocity, as a DISPLACEMENT
#: budget rather than a speed cap.
#:
#: A speed cap was the first attempt and it was wrong: Garen carries Flash
#: (garen1v1.json summoner1), a legitimate ~400-unit blink, which over the
#: 200 ms velocity window implies 2000 u/s and would have been silently
#: deleted -- destroying a real movement the agent must see. A speed cap also
#: depends on the sample window, so it misfires whenever a unit has been
#: unseen for a while.
#:
#: Budget = what walking could cover in dt, with slack, PLUS one blink:
#:     max_step = MAX_WALK_SPEED * dt * WALK_SLACK + BLINK_ALLOWANCE
#: Walking 200 ms -> 69 units (budget 704, kept). Flash -> 400 (kept).
#: Recall/respawn to fountain -> ~12,000 (rejected). Unseen 3 s then seen
#: walking -> 1035 against a budget of 2152 (kept).
MAX_WALK_SPEED = 345.0        # Garen base MoveSpeed
WALK_SLACK = 1.5              # haste, boots, terrain shortcuts
BLINK_ALLOWANCE = 600.0       # Flash is ~400; longest blink in game ~500
NORM_GOLD = 3000.0
NORM_CS = 200.0
NORM_XP = 10000.0
NORM_HP_ABS = 1000.0
NORM_AD = 200.0
MAX_LEVEL = 18
GAME_LENGTH_S = 1800.0
MINION_WAVE_PERIOD_S = 30.0

#: Softness of the "one auto-attack kills it" sigmoid, in HP.  Also the kappa
#: of the potential-based last-hit shaping term in ``reward.py``; they describe
#: the same thing and must not drift apart.
AA_KILL_KAPPA_HP = 10.0
#: Slack added to the attack-range test used by that feature (game units).
AA_RANGE_EPS = 25.0

# --------------------------------------------------------------------------
# Action space
# --------------------------------------------------------------------------

BUTTONS = ("noop", "move", "attack_move", "q", "w", "e", "r", "recall")
N_BUTTONS = len(BUTTONS)
BUTTON_INDEX = {name: i for i, name in enumerate(BUTTONS)}
#: Buttons that consume the target head.
TARGETED_BUTTONS = frozenset({"attack_move", "e", "q"})
#: Buttons that consume the move head.
MOVE_BUTTONS = frozenset({"move", "attack_move"})

#: The two move heads are named ``move_x`` / ``move_z`` for wire compatibility
#: with the environment, but since the lane-frame change they address the
#: LANE-LOCAL axes: ``move_x`` is the s (down-lane) component and ``move_z`` is
#: the n (across-lane) component.  ``frame.LaneTransform.vector`` maps that pair
#: into a world direction.
N_MOVE_BINS = 9
MOVE_BIN_VALUES = np.linspace(-1.0, 1.0, N_MOVE_BINS).astype(np.float32)
MOVE_AXIS_NAMES = {"move_x": "lane_s", "move_z": "lane_n"}

# --------------------------------------------------------------------------
# Entity feature layout (ENTITY_DIM = 40)
# --------------------------------------------------------------------------

E_VALID = 0
E_VISIBLE_NOW = 1
E_STALENESS = 2
E_TYPE_ONEHOT = slice(3, 3 + N_ENTITY_TYPES)          # 3..8
E_TEAM_ONEHOT = slice(9, 9 + N_ENTITY_TEAMS)          # 9..11
E_DS = 12
E_DN = 13
E_DIST = 14
E_COS_BEARING = 15
E_SIN_BEARING = 16
E_HP_FRAC = 17
E_HP_DELTA_SHORT = 18
E_HP_DELTA_LONG = 19
E_IN_MY_AA_RANGE = 20
E_IN_ENEMY_TURRET_RANGE = 21
E_VEL_DS = 22
E_VEL_DN = 23
E_ON_SCREEN = 24
E_HP_KNOWN = 25
# -- derived memory (computed exactly, not learned; see obs.py docstring) ---
E_AGE_S = 26            # seconds since last sighting / FORGET_S
E_LAST_HEADING_COS = 27  # heading at the last sighting, lane-local
E_LAST_HEADING_SIN = 28
E_HEADING_KNOWN = 29
E_REACH_RADIUS = 30      # age * move speed / NORM_DIST -- where it *could* be
# -- derived last-hit arithmetic -------------------------------------------
E_HP_ABS = 31            # hp_frac * max_hp / NORM_HP_ABS, 0 unless hp_known
E_AA_KILLABLE = 32       # sigmoid((my AD - hp) / kappa), 0 unless hp_known
E_AA_SHOTS_TO_KILL = 33  # ceil(hp / my AD) / 8, clamped, 0 unless hp_known
E_RESERVED = slice(34, ENTITY_DIM)

#: Human-readable names, index-aligned, used by ``audit.py``.
ENTITY_FIELD_NAMES = (
    ["valid", "visible_now", "staleness"]
    + [f"type_{t}" for t in ENTITY_TYPES]
    + [f"team_{t}" for t in ENTITY_TEAMS]
    + [
        "lane_ds",
        "lane_dn",
        "dist",
        "cos_bearing",
        "sin_bearing",
        "hp_frac",
        "hp_frac_delta_0p5s",
        "hp_frac_delta_2s",
        "in_my_aa_range",
        "in_enemy_turret_range",
        "vel_ds",
        "vel_dn",
        "on_screen",
        "hp_known",
        "age_of_estimate",
        "last_seen_heading_cos",
        "last_seen_heading_sin",
        "last_seen_heading_known",
        "reachability_radius",
        "hp_absolute",
        "one_shot_kill_score",
        "shots_to_kill",
    ]
    + [f"reserved_{i}" for i in range(34, ENTITY_DIM)]
)
assert len(ENTITY_FIELD_NAMES) == ENTITY_DIM, len(ENTITY_FIELD_NAMES)

# --------------------------------------------------------------------------
# Self feature layout (SELF_DIM = 64)
# --------------------------------------------------------------------------

S_HP_FRAC = 0
S_HP_DELTA_SHORT = 1
S_HP_DELTA_LONG = 2
S_LEVEL_NORM = 3
S_LEVEL_ONEHOT = slice(4, 4 + MAX_LEVEL)   # 4..21
S_GOLD_NORM = 22
S_GOLD_LOG = 23
S_CS_NORM = 24
S_CS_LOG = 25
S_Q_READY = 26
S_W_READY = 27
S_E_READY = 28
S_R_READY = 29
S_Q_LEARNED = 30
S_W_LEARNED = 31
S_E_LEARNED = 32
S_R_LEARNED = 33
S_LANE_S = 34
S_LANE_N = 35
S_LANE_S_VEL = 36
S_LANE_N_VEL = 37
S_MOVE_COS = 38
S_MOVE_SIN = 39
S_SPEED_NORM = 40
S_IS_DEAD = 41
S_REGION_ONEHOT = slice(42, 48)  # 42..47
S_IN_OWN_TURRET_RANGE = 48
S_IN_ENEMY_TURRET_RANGE = 49
S_RECALLING = 50
S_OFF_LANE = 51
# -- auto-attack cycle (exact arithmetic, see obs.AttackClock) --------------
S_ATTACK_CYCLE_PHASE = 52
S_TIME_UNTIL_NEXT_ATTACK = 53
S_WINDUP_REMAINING = 54
S_ATTACK_TIMING_KNOWN = 55
S_AA_DAMAGE_NORM = 56
S_AA_RANGE_NORM = 57
S_RESERVED = slice(58, SELF_DIM)

REGIONS = (
    "own_fountain",
    "own_turret_zone",
    "own_side_lane",
    "lane_mid",
    "enemy_side_lane",
    "enemy_turret_zone",
)
N_REGIONS = len(REGIONS)
assert N_REGIONS == 6

SELF_FIELD_NAMES = (
    ["hp_frac", "hp_frac_delta_0p5s", "hp_frac_delta_2s", "level_norm"]
    + [f"level_{i + 1}" for i in range(MAX_LEVEL)]
    + [
        "gold_norm",
        "gold_log",
        "cs_norm",
        "cs_log",
        "q_ready",
        "w_ready",
        "e_ready",
        "r_ready",
        "q_learned",
        "w_learned",
        "e_learned",
        "r_learned",
        "lane_s",
        "lane_n",
        "lane_s_vel",
        "lane_n_vel",
        "move_cos",
        "move_sin",
        "speed_norm",
        "is_dead",
    ]
    + [f"region_{r}" for r in REGIONS]
    + ["in_own_turret_range", "in_enemy_turret_range", "recalling", "off_lane"]
    + [
        "attack_cycle_phase",
        "time_until_next_attack_ready",
        "windup_remaining",
        "attack_timing_known",
        "my_attack_damage",
        "my_attack_range",
    ]
    + [f"reserved_{i}" for i in range(58, SELF_DIM)]
)
assert len(SELF_FIELD_NAMES) == SELF_DIM, len(SELF_FIELD_NAMES)

# --------------------------------------------------------------------------
# Global feature layout (GLOBAL_DIM = 48)
# --------------------------------------------------------------------------

G_CLOCK_NORM = 0
G_CLOCK_PHASE = slice(1, 7)  # 1..6
G_WAVE_SIN = 7
G_WAVE_COS = 8
G_MY_KILLS = 9
G_MY_DEATHS = 10
G_MY_ASSISTS = 11
G_ENEMY_VISIBLE = 12
G_ENEMY_UNSEEN_TIME = 13
G_N_ALLY_MINIONS = 14
G_N_ENEMY_MINIONS = 15
G_IS_DEAD = 16
G_DT_NORM = 17
#: Remembered whereabouts of the enemy champion.  A human remembers where they
#: last saw the enemy, roughly how long ago, which way they were facing, and --
#: crucially -- how far they *could* have got since.  All of that is legal to
#: encode.  The enemy's *current* position while fogged is not here and cannot be.
G_ENEMY_MEM_LANE_S = 18
G_ENEMY_MEM_LANE_N = 19
G_ENEMY_MEM_DS = 20
G_ENEMY_MEM_DN = 21
G_ENEMY_MEM_VALID = 22
G_ENEMY_MEM_AGE = 23
G_ENEMY_MEM_HEADING_COS = 24
G_ENEMY_MEM_HEADING_SIN = 25
G_ENEMY_MEM_HEADING_KNOWN = 26
G_ENEMY_REACH_RADIUS = 27
#: Enemy ability book, built from *witnessed casts only* (see
#: ``frame.EnemyAbilityIntel``).  At 15 Hz a 20 s cooldown is 300 decisions --
#: no BPTT window reaches that, so this is computed exactly instead of hoped for.
G_ENEMY_ABILITY_SINCE_CAST = slice(28, 32)   # 28..31
G_ENEMY_ABILITY_CD_EST = slice(32, 36)       # 32..35
G_ENEMY_ABILITY_UNKNOWN = slice(36, 40)      # 36..39
G_RESERVED = slice(40, GLOBAL_DIM)

CLOCK_PHASE_BOUNDS_S = (120.0, 300.0, 600.0, 900.0, 1200.0)  # -> 6 buckets

GLOBAL_FIELD_NAMES = (
    ["clock_norm"]
    + [f"clock_phase_{i}" for i in range(6)]
    + [
        "wave_phase_sin",
        "wave_phase_cos",
        "my_kills",
        "my_deaths",
        "my_assists",
        "enemy_visible",
        "enemy_unseen_time",
        "n_visible_ally_minions",
        "n_visible_enemy_minions",
        "is_dead",
        "dt_norm",
        "enemy_mem_lane_s",
        "enemy_mem_lane_n",
        "enemy_mem_ds",
        "enemy_mem_dn",
        "enemy_mem_valid",
        "enemy_mem_age_of_estimate",
        "enemy_mem_last_seen_heading_cos",
        "enemy_mem_last_seen_heading_sin",
        "enemy_mem_last_seen_heading_known",
        "enemy_mem_reachability_radius",
    ]
    + [f"enemy_ability_{s.lower()}_time_since_observed_cast" for s in SPELL_SLOTS]
    + [f"enemy_ability_{s.lower()}_cd_estimate" for s in SPELL_SLOTS]
    + [f"enemy_ability_{s.lower()}_never_observed" for s in SPELL_SLOTS]
    + [f"reserved_{i}" for i in range(40, GLOBAL_DIM)]
)
assert len(GLOBAL_FIELD_NAMES) == GLOBAL_DIM, len(GLOBAL_FIELD_NAMES)

# --------------------------------------------------------------------------
# Privileged (critic-only) layout (PRIV_DIM = 96)
# --------------------------------------------------------------------------

P_E_HP_FRAC = 0
P_E_HP_ABS = 1
P_E_LEVEL_NORM = 2
P_E_LEVEL_ONEHOT = slice(3, 3 + MAX_LEVEL)  # 3..20
P_E_GOLD_NORM = 21
P_E_GOLD_LOG = 22
P_E_CS_NORM = 23
P_E_LANE_S = 24
P_E_LANE_N = 25
P_E_DS = 26
P_E_DN = 27
P_E_DIST = 28
P_E_VEL_DS = 29
P_E_VEL_DN = 30
P_E_ALIVE = 31
P_E_VISIBLE_TO_ME = 32
P_GOLD_DIFF = 33
P_XP_DIFF = 34
P_LEVEL_DIFF = 35
P_HP_FRAC_DIFF = 36
P_CS_DIFF = 37
P_N_ALLY_MINIONS = 38
P_N_ENEMY_MINIONS = 39
P_ALLY_MINION_HP = 40
P_ENEMY_MINION_HP = 41
P_OWN_TURRET_HP = 42
P_ENEMY_TURRET_HP = 43
P_MY_LANE_S = 44
P_MY_LANE_N = 45
P_MY_HP_FRAC = 46
P_MY_GOLD_NORM = 47
P_MY_XP_NORM = 48
P_CLOCK_NORM = 49
#: The enemy's TRUE remaining cooldowns, straight off the control channel.
#: Critic only -- the actor gets the witnessed-cast estimate instead.
P_E_COOLDOWN_TRUE = slice(50, 54)  # 50..53
#: The value of the last-hit shaping potential, for critic bookkeeping.
P_LAST_HIT_POTENTIAL = 54
P_RESERVED = slice(55, PRIV_DIM)

PRIV_FIELD_NAMES = (
    ["enemy_hp_frac", "enemy_hp_abs", "enemy_level_norm"]
    + [f"enemy_level_{i + 1}" for i in range(MAX_LEVEL)]
    + [
        "enemy_gold_norm",
        "enemy_gold_log",
        "enemy_cs_norm",
        "enemy_lane_s",
        "enemy_lane_n",
        "enemy_ds",
        "enemy_dn",
        "enemy_dist",
        "enemy_vel_ds",
        "enemy_vel_dn",
        "enemy_alive",
        "enemy_visible_to_me",
        "gold_diff",
        "xp_diff",
        "level_diff",
        "hp_frac_diff",
        "cs_diff",
        "n_ally_minions_true",
        "n_enemy_minions_true",
        "ally_minion_hp_true",
        "enemy_minion_hp_true",
        "own_turret_hp_true",
        "enemy_turret_hp_true",
        "my_lane_s",
        "my_lane_n",
        "my_hp_frac_true",
        "my_gold_norm",
        "my_xp_norm",
        "clock_norm",
    ]
    + [f"enemy_cooldown_{s.lower()}_true" for s in SPELL_SLOTS]
    + ["last_hit_potential_true"]
    + [f"reserved_{i}" for i in range(55, PRIV_DIM)]
)
assert len(PRIV_FIELD_NAMES) == PRIV_DIM, len(PRIV_FIELD_NAMES)
