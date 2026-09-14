"""Static constants for the 1v1 Garen lane RL stack.

Everything here is either (a) read out of the vendored server content, or
(b) a deliberate design choice that is documented inline.  Nothing here is
guessed silently -- where a number is an approximation it says so.

That promise was broken four times, all found and corrected on 2026-09-12:
:data:`TARGET_RADIUS` (champion and minion both wrong, from a field the server
never uses for range), :data:`VISION_RADIUS_CHAMPION` / :data:`VISION_RADIUS_MINION`
(swapped), :data:`GAREN_COOLDOWNS` (``E`` held its rank-5 value at every rank),
and :data:`ENEMY_COOLDOWN_ASSUMED` (its stated safety direction was the opposite
of what the code did).  The common cause in three of the four: reasoning from
``CharData`` when the constructor argument is what wins.  Each is now the C#
ctor value with the file and line cited inline.

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
#:
#: 60, not 30, because the lane's actual dynamics are slower than 30 s and at
#: 30 they were invisible to the objective.  A shove-and-bounce wave cycle is
#: 1-3 minutes, and gold and experience are lagging indicators of lane control
#: on the order of minutes.  What the discount does to that, at 30 Hz::
#:
#:     horizon_s   reward 120 s away   reward 180 s away
#:            30              1.83%               0.25%
#:            60             13.53%               4.97%
#:           120             36.78%              22.31%
#:
#: At 30 s a payoff two minutes out is worth under 2% of face value, so no
#: amount of training can teach wave management: the credit never arrives.
#: 60 s is the conservative step -- raising the horizon also raises value
#: variance and makes credit assignment harder, so this is not free, and 120
#: is available if 60 proves too short.
#:
#: NOT the only horizon in the stack, and the smaller one is now the binding
#: constraint: ``PPOConfig.chunk_len = 16`` means BPTT reaches back 0.53 s, so
#: the GRU is only ever TRAINED to use half a second of memory however far the
#: discount sees.
DEFAULT_HORIZON_S = 60.0


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


#: Scale on the last-hit shaping potential ``Phi``.  ONE definition: it was
#: duplicated as ``last_hit_potential(c=0.05)``, ``LaneRewardConfig.shaping_c``
#: and -- by omission -- the critic feature ``P_LAST_HIT_POTENTIAL``, which left
#: it off entirely and so reported 20x the potential it is named after.
SHAPING_C = 0.05


def gamma_for_horizon(horizon_s: float, decision_hz: float = DECISION_HZ) -> float:
    """Discount factor whose 1/(1-gamma) horizon is ``horizon_s`` seconds.

    ``gamma = 1 - 1 / (horizon_s * decision_hz)``.  At the default 30 Hz::

        30 s -> 0.998889
        45 s -> 0.999259

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
#:
#: This is a NORMALISER, not a measurement: the top lane's actual mean walkable
#: half-width is ~674 units (measured 2026-09-12 off AIPath.aimesh_ngrid, 293x294
#: cells at 50u). 1400 therefore puts a champion hugging the lane wall at
#: n ~ 0.48 rather than ~1.0, i.e. n uses about half its range. Left as-is
#: deliberately: |n| up to 1400 covers the jungle entrances and the tri-brush,
#: which a champion does legitimately stand in, and a normaliser that clips the
#: moment you step off the path would be worse than one that is loose.
LANE_HALF_WIDTH = 1400.0

# --------------------------------------------------------------------------
# Unit properties (from Content/LeagueSandbox-Default/Stats)
# --------------------------------------------------------------------------

#: Sight radii, taken from the C# constructor arguments -- NOT from content
#: ``CharData.PerceptionBubbleRadius``.  ``ObjAIBase.cs:128-140`` tests
#: ``if (visionRadius > 0)`` FIRST and the ctor argument is positive for both
#: types, so the CharData branch is unreachable here and the content's
#: ``PerceptionBubbleRadius = 1200`` on minions is never read.
#:
#: These two were swapped until 2026-09-12.  A champion radius of 1100 against
#: a real 1200 let the fallback fog model HIDE a unit in the 1100-1200 annulus,
#: breaking the safety property asserted at ``frame.py:29-32`` and
#: ``frame.py:831-834``: that ``ApproxFogModel`` can only ever *add* visibility,
#: never remove one the server showed.  Corrected while BC was being retrained
#: from scratch, so no trained artifact encodes the old radii.
VISION_RADIUS_CHAMPION = 1200.0      # Champion.cs:52
VISION_RADIUS_MINION = 1100.0        # Minion.cs:57
#: ``BaseTurret`` ctor passes ``visionRadius: 800``.
VISION_RADIUS_TURRET = 800.0
#: Buildings (inhib/nexus).  A documented approximation used only by the
#: *fallback* fog model.  Map1 actually passes nexus sight 1700 and inhibitor
#: sight 0 (``Maps/Map1/LevelScriptObjects.cs:304,321``), so this single number
#: is wrong in both directions at once; it is an approximation on purpose.
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


#: There is deliberately NO garen_attack_damage() any more.
#:
#: It was a Python re-derivation of a number the server already computes, and
#: it was wrong three times in a row, each time by whatever part of the server
#: had been forgotten: 57.88 with no runes, 73.14 once the rune page was
#: modelled, against a true 78.14 -- the remainder being Martial Mastery (+5
#: flat) and Brute Force (+0.55/level) from the mastery page. Each wrong value
#: fed the last-hit features, so the agent was told that killable minions were
#: not killable.
#:
#: The rune and mastery pages are FIXED in lanerl/cfg/garen1v1.json, and the
#: control channel emits the resulting total as ``ad`` (LanerlControl.cs). Read
#: ``Unit.ad``. Where it is absent -- only recordings made before 2026-09-12 --
#: the honest answer is that attack damage is unknown, not a guess.

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
#: RAW content values.  The server does NOT pay in these units: ``GlobalData.cs``
#: :96-97 rescales them to ``0.95`` gold every ``500`` ms before ``Champion``
#: ever reads them (``9.5 / (10 / 5) / 5`` and ``5.0 * 100``).  Only the *rate*
#: below survives the rescaling, and the rate is the only thing the reward uses.
#: Do not reason about lumpiness from these two numbers.
AMBIENT_GOLD_AMOUNT = 9.5
AMBIENT_GOLD_INTERVAL_S = 5.0
#: 1.9 gold/s.  This is the number the server actually realises, via 0.95 / 500 ms.
AMBIENT_GOLD_PER_S = AMBIENT_GOLD_AMOUNT / AMBIENT_GOLD_INTERVAL_S
#: ``ai_AmbientGoldDelay``, x1000 into ms by GlobalData.  ``Champion.Update``
#: tests only this, never ``ai_AmbientGoldDelayFirstBlood`` (declared, loaded,
#: read nowhere), so the delay really is always 90 s.
AMBIENT_GOLD_DELAY_S = 90.0

#: Effective "target radius" added to attack-range checks, by entity type.
#: The server's in-range test is centre-to-centre plus the target's
#: ``CollisionRadius`` (``ObjAIBase.cs:1195``, ``:1235``, ``:1274``).
#:
#: These are the CONSTRUCTOR arguments, not content ``CharData``.
#: ``ObjAIBase.cs:106-116`` takes the ctor argument first and never reaches
#: CharData for champions or minions, so ``Garen.json
#: PathfindingCollisionRadius = 35`` -- which this used to cite -- is not what
#: the server uses for range, and is not even consulted.
#:
#: Champion was 35 and minion 25 until 2026-09-12.  Reach for a minion is
#: ``AA_RANGE_GAREN + 40 = 165`` on the server but was computed as 150 here, so
#: ``reward.last_hit_potential`` and the ``obs``/``frame`` in-range flags were
#: 15 units short: they missed a minion the server would have let us hit.  On a
#: last-hit task that is the entire margin.
TARGET_RADIUS = {
    "champion": 30.0,     # Champion.cs:52
    "minion": 40.0,       # Minion.cs:57
    "turret": 88.0,
    "inhibitor": 214.0,   # LevelScriptObjects.cs:304
    "nexus": 353.0,       # LevelScriptObjects.cs:321
    "other": 40.0,        # server's own fallback is also 40 (ObjAIBase.cs:116)
}

#: Garen ability base cooldowns, per rank, from Content/.../Spells/Garen{Q,W,E,R}.
#:
#: ``E`` carried its rank-5 value (9 s) at every rank until 2026-09-12, while
#: ``lanerl_bot/build.py`` read 13/12/11/10/9 from the same JSON -- the two
#: halves of the stack disagreed about the enemy's E.
GAREN_COOLDOWNS = {
    "Q": (8.0, 8.0, 8.0, 8.0, 8.0),
    "W": (24.0, 23.0, 22.0, 21.0, 20.0),
    "E": (13.0, 12.0, 11.0, 10.0, 9.0),  # Spells/GarenE/GarenE.json
    # R has three ranks; entries 4-5 are padding and do not match the JSON's
    # own 120/120 filler. Harmless -- Champion.LevelUpSpell caps R at rank 3.
    "R": (160.0, 120.0, 80.0, 80.0, 80.0),
}
SPELL_SLOTS = ("Q", "W", "E", "R")

#: Rank-1 cooldowns, used as the *estimate* denominator for an ENEMY ability.
#: We deliberately never read the enemy's spell ranks (a rank read would be a
#: leak).  Q is genuinely rank-invariant, so for Q the estimate is exact.
#:
#: The denominator is the MINIMUM over ranks, not rank 1.  The safety direction
#: we want is "assume the enemy is readier than it is", which needs the
#: SHORTEST cooldown, so the estimate decays fastest and reports the ability
#: back up early.  Rank 1 is the LONGEST for W (24 vs 20) and R (160 vs 80), so
#: using it did the exact opposite of the property the comment claimed: the
#: agent was told the enemy was LESS ready than it was, and walked into it.
#: Measured before the fix -- enemy W at rank 5 was still reported 0.125 down at
#: dt=21 s, enemy R at rank 3 still 0.188 down at dt=130 s, both long back up.
#:
#: Q is genuinely rank-invariant, so for Q this is exact.  We never read the
#: enemy's real spell ranks: that would be a leak.
ENEMY_COOLDOWN_ASSUMED = tuple(min(GAREN_COOLDOWNS[s]) for s in SPELL_SLOTS)
#: Deprecated alias; the name said rank 1, the value is the min over ranks.
ENEMY_COOLDOWN_ASSUMED_RANK1 = ENEMY_COOLDOWN_ASSUMED

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
#: between them to count as a *witnessed* cast.  One decision step is 33.3 ms;
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
# Entity feature layout
# --------------------------------------------------------------------------
#
# MINIMAL BY DESIGN, rebuilt 2026-09-12 from 40 fields to 10.
#
# What a player can see about a unit is where it is, how hurt it is, and what
# it is. Everything else the old layout carried was either a function of those
# (dist, bearing, in_my_aa_range -- all computable from ds/dn), a temporal
# derivative the GRU exists to compute (hp deltas), or an ANSWER precomputed on
# the network's behalf. The precomputed answers are where the bugs lived:
#
#   one_shot_kill_score  sigmoid((my_AD - hp)/kappa) -- the last-hit decision
#                        itself, computed from a Python re-derivation of AD
#                        that read 57.88 against the server's true 78.14.
#   shots_to_kill        same wrong AD, same class of mistake.
#   hp_absolute          carried at /1000 here and /2000 in priv_vec: the same
#                        quantity on two scales in one observation.
#   reach_radius,        a fog-memory model whose horizon constant was already
#   last_heading_*,      dead (ENEMY_MEMORY_HORIZON_S=60 unreachable because
#   hp_known, age_s      FORGET_S=8 deletes the memory first).
#   visible_now,         degenerate now that only VISIBLE units are slotted --
#   staleness            the GRU is the memory.
#   6 reserved floats    x 32 slots = 192 permanently-zero inputs.
#
# Positions are RELATIVE to the agent in lane coordinates and in raw game
# units (normalised by NORM_DIST), because that is what "how far away is it"
# means to a player. Absolute position lives on the self vector only.
E_VALID = 0
E_DS = 1                 # (target - me) along the lane axis / NORM_DIST
E_DN = 2                 # (target - me) perpendicular / NORM_DIST
E_HP_FRAC = 3            # quantised to health-bar resolution
E_TYPE_ONEHOT = slice(4, 4 + N_ENTITY_TYPES)
E_TEAM_ONEHOT = slice(4 + N_ENTITY_TYPES, 4 + N_ENTITY_TYPES + N_ENTITY_TEAMS)
#: WHICH KIND of lane minion, one-hot, all-zero for anything that is not one.
#:
#: Every lane minion reaches the wire as kind "LaneMinion", so melee, caster
#: and cannon were a single category to the policy -- while their max health is
#: 455 / 290 / 700 and the hp_frac at which one auto-attack kills them is
#: 0.172 / 0.263 / 0.112, a 2.3x spread. The agent was being asked to learn one
#: last-hit threshold correct for none of the three, from an hp_frac it could
#: not scale.
#:
#: Categorical on purpose: this says WHAT the thing is, and leaves working out
#: what that implies about damage to the network. Max health is deliberately
#: NOT fed -- a bar's length is what a player reads as "how hurt", and the
#: identity of the unit is what tells them how much that is worth.
_MT = 4 + N_ENTITY_TYPES + N_ENTITY_TEAMS
E_MINION_MELEE = _MT
E_MINION_CASTER = _MT + 1
E_MINION_CANNON = _MT + 2
E_MINION_SUBTYPE = slice(_MT, _MT + 3)
#: MinionSpawnType -> index above. SUPER (1) is absent on purpose: it only
#: spawns after an inhibitor falls, which cannot happen in a 10-minute lane,
#: and a one-hot bit that is never set is the padding this layout removed.
MINION_TYPE_INDEX = {0: 0, 3: 1, 2: 2}
ENTITY_DIM = _MT + 3

#: Human-readable names, index-aligned, used by ``audit.py``.
ENTITY_FIELD_NAMES = tuple(
    ["valid", "lane_ds", "lane_dn", "hp_frac"]
    + [f"type_{t}" for t in ENTITY_TYPES]
    + [f"team_{t}" for t in ENTITY_TEAMS]
    + ["minion_melee", "minion_caster", "minion_cannon"]
)
assert len(ENTITY_FIELD_NAMES) == ENTITY_DIM, len(ENTITY_FIELD_NAMES)

# --------------------------------------------------------------------------
# Self feature layout
# --------------------------------------------------------------------------
#
# MINIMAL BY DESIGN, rebuilt 2026-09-12 from 64 fields to 16.
#
# Removed, and why:
#   18 level one-hots     redundant with S_LEVEL_NORM -- 19 inputs for one int
#   gold_log, cs_log      second encoding of gold_norm / cs_norm
#   6 region one-hots,    all functions of (lane_s, lane_n) and known turret
#   off_lane, in_*_range  positions; the network can threshold a scalar
#   hp deltas, velocities the GRU exists to compute time derivatives
#   move_cos/sin, speed   derivable from successive positions
#   4 attack_cycle_*      driven by a Python-maintained clock fed by orders
#                         ISSUED, not swings landed; identically 0 across all
#                         99,654 BC rows while being live at RL time
#   6 reserved            permanently zero
#
# ``ad/ap/armor/mr`` are the server's OWN numbers off the wire, not
# recomputed. constants.garen_attack_damage() read 57.88 at level 1, then
# 73.14 once the rune page was modelled, against a true 78.14 -- a re-derived
# server quantity is wrong by however much of the server you forgot.
S_LANE_S = 0             # absolute position along the lane
S_LANE_N = 1             # absolute perpendicular offset
S_HP_FRAC = 2
S_LEVEL_NORM = 3
S_GOLD_NORM = 4
S_CS_NORM = 5
S_CD_Q = 6               # remaining cooldown / base cooldown, 0 = ready
S_CD_W = 7
S_CD_E = 8
S_CD_R = 9
S_AD = 10                # off the wire
S_AP = 11
S_ARMOR = 12
S_MR = 13
S_IS_DEAD = 14
S_RECALLING = 15
SELF_DIM = 16

SELF_FIELD_NAMES = (
    "lane_s", "lane_n", "hp_frac", "level_norm", "gold_norm", "cs_norm",
    "cd_q", "cd_w", "cd_e", "cd_r",
    "ad", "ap", "armor", "mr", "is_dead", "recalling",
)
assert len(SELF_FIELD_NAMES) == SELF_DIM, len(SELF_FIELD_NAMES)

# --------------------------------------------------------------------------
# Global feature layout
# --------------------------------------------------------------------------
#
# MINIMAL BY DESIGN, rebuilt 2026-09-12 from 48 fields to 6.
#
# Removed:
#   6 clock phase one-hots   redundant with G_CLOCK_NORM
#   wave sin/cos             a function of the clock
#   kills/deaths/assists     in a 1v1 lane these are ~always 0; the death
#                            event is already a reward term
#   10 enemy-memory fields   a hand-built model of "where did he go" whose own
#                            horizon constant was unreachable
#                            (ENEMY_MEMORY_HORIZON_S=60 vs FORGET_S=8). The GRU
#                            is the memory.
#   8 enemy ability cd_est   built on ENEMY_COOLDOWN_ASSUMED, whose documented
#     + never_observed       safety direction was BACKWARDS, on top of a
#                            Garen-E cooldown wrong at every rank. What is
#                            actually observable is "how long since I saw him
#                            cast it", which is kept.
#   8 reserved               permanently zero
G_CLOCK_NORM = 0
G_ENEMY_VISIBLE = 1
G_ENEMY_ABILITY_SINCE_CAST = slice(2, 6)     # 2..5, one per spell slot
GLOBAL_DIM = 6

GLOBAL_FIELD_NAMES = (
    "clock_norm", "enemy_visible",
    "enemy_q_time_since_observed_cast", "enemy_w_time_since_observed_cast",
    "enemy_e_time_since_observed_cast", "enemy_r_time_since_observed_cast",
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
