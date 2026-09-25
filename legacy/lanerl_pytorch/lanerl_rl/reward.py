"""The lane reward: JueWu-shaped, symmetrised zero-sum, potential-based.

Why this module exists separately
---------------------------------
``env.py`` still ships the old ``RewardConfig`` / ``LaneReward`` pair (a raw
delta-hp / delta-gold / delta-xp sum).  This module is the replacement.  It is
a drop-in at the call site: build one :class:`ZeroSumLaneReward` for the whole
env and call :meth:`ZeroSumLaneReward.step` once per tick instead of calling
``LaneReward.raw`` per team and subtracting.

The shape
---------
The weights START from the 1v1 solo-lane configuration used by the JueWu line
of work (Ye et al. 2020, "Mastering Complex Control in MOBA Games with Deep
Reinforcement Learning"; the same table ships as the default reward config of
the Honor of Kings 1v1 gym environment)::

    hp_point    2.0     as a POTENTIAL DIFFERENCE, see below
    tower_hp   10.0
    money       0.008   on EARNED gold only, see below
    mana        0.8     -> 0.0 here: Garen is manaless (Garen.json BaseMP = 0)
    exp         0.008
    death      -1.0
    kill       -0.5
    last_hit    0.5

``tower_hp``, ``money``, ``death`` and ``kill`` are used as published.  Four
are not, and every deviation is argued with arithmetic in the next section.
Nothing here is a taste judgement: each number was picked against a measured
600 s episode and each has a test that fails if it is silently moved.

The negative ``kill`` weight is not a typo and is not a bug: it is what makes
the kill/death pair antisymmetric once the zero-sum subtraction is applied.
Killing the enemy fires *their* ``death`` (-1.0) and *my* ``kill`` (-0.5), so
``r = r_self - r_opp = -0.5 - (-1.0) = +0.5``; dying gives exactly ``-0.5``.
Take the weights apart and the pair stops balancing -- so when dying needed to
hurt more (below), ``hp_point`` was raised and this pair was left alone.

Deviations from the published table
-----------------------------------
::

    weight      published   here    why, in one line
    mana          0.8       0.0     Garen.json BaseMP = 0: nothing to measure
    exp           0.008     0.001   it pays for proximity, not for skill
    last_hit      0.5       1.0     the one term the agent fully controls
    hp_point      2.0       4.0     trading must beat its opportunity cost
    spend        (none)     0.0     the purchase is scripted, and paying for
                                    it would pay the agent for dying

**exp pays for proximity, not for skill.**  ``AttackableUnit.Die`` hands
``ExpGivenOnDeath`` to every enemy champion within ``ai_ExpRadius2 = 1600``
units of the corpse, split among them, *regardless of who landed the kill*;
gold goes through ``Champion.OnKill`` and reaches the killer only.  Garen's
last-hit reach is ``AA_RANGE_GAREN + TARGET_RADIUS["minion"] + AA_RANGE_EPS``
= 190 units.  The XP radius is therefore 8.4x the reach and 71x the area: XP is
what the lane pays for standing in it, gold is what it pays for hitting the
minion.  Per melee minion (``Blue_Minion_Basic``: 77 XP, 20 gold), at the
published weights::

    just being there     0.008 * 77          = 0.616
    landing the hit      0.5 + 0.008 * 20    = 0.660

-- 48% of a dying minion's payout required no skill at all.  Measured over a
real 600 s episode (46 CS), ``exp`` totalled **+34.82** against ``last_hit``
+23.00 and ``money`` +5.27, and carried 14.27 of the ~29 total ``|r|`` the
episode moved: the largest term in the table, and the one the policy least
controls.  At ``exp = 0.001`` with ``last_hit = 1.0``::

    just being there     0.001 * 77          = 0.077
    landing the hit      1.0 + 0.008 * 20    = 1.160

-- 6.2%.  As a rate: a 3-melee/3-caster wave is ``3*77 + 3*51 = 384`` XP every
30 s, i.e. 12.8 XP/s of standing next to a wave that dies on its own, so one
melee last hit is worth::

    before   0.660 / (0.008 * 12.8)  =   6.4 seconds of standing there
    after    1.160 / (0.001 * 12.8)  =  90.6 seconds of standing there

``exp`` is kept rather than deleted because it is the only DENSE term that
notices the agent leaving lane, being zoned off the wave, or lying dead, and --
under the zero-sum subtraction -- the only one that scores denying the enemy
their half of it.  It simply must not be the biggest number in the table.

**last_hit 0.5 -> 1.0.**  This is the objective.  It is now computed from the
server's own ``ChampStats.MinionsKilled`` (``CreepScoreEstimator`` prefers the
wire's ``cs``), so unlike the old proximity estimate it cannot be collected by
standing in a wave somebody else is killing.  Doubling it, together with the
8x cut to ``exp``, moves farming from 36% to ~81% of the episode's positive
return while leaving the return's *scale* alone (the measured episode goes from
+64.1 to +56.7), so no PPO coefficient has to move with it.  ``money`` stays at
0.008 on top, and is what makes a cannon (35g) worth more than a caster (10g):
``last_hit`` alone is flat per minion.

**hp_point 2.0 -> 4.0, so that trading can pay for itself.**  The term is
``w * (phi(h_t) - phi(h_{t-1}))`` on both champions, and the zero-sum
subtraction turns a trade into ``w * (dphi_them - dphi_me)``.  A clean level-6
trade -- 25% off them, 15% off me, both from full::

    phi(1.00) - phi(0.75) = 0.12695     (them)
    phi(1.00) - phi(0.85) = 0.07525     (me)
    r = w * (0.12695 - 0.07525)         = 0.0517 * w

At the published ``w = 2.0`` that is +0.103, **9%** of one melee last hit
(1.16): any trade costing even a single CS was a loss, which is what the
measured episode shows -- ``hp_point`` moved 1.14 of total magnitude in 300 s,
the signature of an agent that never trades.  At ``w = 4.0`` it is +0.207, 18%
of a CS: poke that is free (they step into range, Q is up) pays for itself,
poke that costs a wave still does not, which is the correct ordering.

The quartic is what makes this more than a linear rescale, and it is why 18%
is the right answer for *that* trade: chipping a full-health enemy to 75% is
genuinely worth very little (``phi'(1) = 0.5``), while the same 25% taken off a
champion already at 40% -- the chunk that sets up a kill -- is worth five times
as much (``phi'(0) = 2.5``)::

    phi(0.40) - phi(0.15) = 0.3212  ->  4.0 * 0.3212 = +1.285   (1.1 CS)

and a whole health bar, which is what killing someone from full is worth on
this term, is ``w = 4.0`` (3.4 CS) before the kill/death pair adds its +0.5.
Not raised further: at ``w = 8`` the kill-setup chunk is 2.2 CS and a health
bar is 6.9, so trading outbids farming outright and the agent learns to poke
instead of last-hit -- the opposite failure.

Dying gets more expensive from this too, and that is the point -- but only
after the respawn refund below was closed.  Raising ``death`` would have been
the obvious way to make dying hurt, and is precisely what must not happen: see
the kill/death pairing above.

A respawn is not a heal
-----------------------
Raising ``hp_point`` surfaced a bug that was already there at the published
2.0.  ``phi`` is a potential on the hp LEVEL, so the jump from 0 back to full
on respawn pays ``w * phi(1) = w`` -- while the death itself only charged
``w * phi(h_at_death)``.  The difference is a refund, and for any death below
full hp it is a *profit*::

    w = 2.0, die at 30% hp
        hp_point  -1.060 (the death)  +2.000 (the respawn)
        death     -1.000
        raw       -0.060  ->  zero-sum vs the killer's -0.5  =  +0.44

    w = 4.0, die at 30% hp                                   =  +1.38
    w = 2.0, die at  5% hp                                   =  +1.27

i.e. under the published table the agent was **paid to feed**, and paid more
the lower it was when it died.  ``_AgentReward.raw`` now suppresses the
``hp_point`` delta across the dead->alive transition, so a death costs exactly
``w * phi(h_at_death)`` and nothing comes back.  That is the right number: the
potential has already charged for every point of hp lost on the way down, so
the total cost of going from full to dead is ``w`` however the path ran, and
dying at 1 hp is cheap only because it was already paid for.  At ``w = 4.0``:

    die at full hp   -4.0 - 1.0 + 0.5  =  -4.5
    die at 30% hp    -2.12 - 1.0 + 0.5 =  -2.62
    die at  5% hp    -0.47 - 1.0 + 0.5 =  -0.97

against +1.16 for a melee last hit: three deaths cost roughly six CS.  This is
not potential-based shaping and carries no gamma, so there is no Ng-et-al.
invariance to break, and the suppression applies identically to both agents, so
the antisymmetry at ``alpha = 1`` is untouched.
``test_dying_is_never_profitable`` and ``test_a_respawn_is_not_a_heal`` pin it.

The HP term is a potential difference -- the bug this file exists to avoid
-------------------------------------------------------------------------
With ``phi(x) = (x + 1 - (1 - x)**4) / 2`` on the hp fraction ``x``, the term is

    w * (phi(h_t) - phi(h_{t-1}))          CORRECT

and **not**

    w * phi(h_t - h_{t-1})                 WRONG

``phi`` maps [0, 1] onto [0, 1] with ``phi'(0) = 2.5`` and ``phi'(1) = 0.5``:
health near death is worth 5x health near full, which is the entire point of
the quartic.  Applying ``phi`` to the *delta* throws that away -- a delta is not
a health level, so the curvature no longer means anything -- and it is not even
telescoping.  A known LeagueSandbox-based RL environment (pylol) has exactly
this bug.  The damage is measurable: on a 2%-amplitude HP oscillation
(1.00 -> 0.98 -> 1.00, which lane trading produces constantly), the correct
form sums to exactly 0 while the wrong form sums to

    w * (phi(-0.02) + phi(+0.02)) = w * (-0.05122 + 0.04882) = -0.0024 * w

per cycle -- ``-0.0096`` at the ``hp_point = 4.0`` used here -- a persistent
negative drift that teaches the agent to avoid trading at all.  Note that the
drift scales with the weight, so raising ``hp_point`` to make trading worth
doing would have made this bug *worse* had it still been present.
:func:`hp_potential_delta` is the correct form and
:func:`_wrong_hp_potential_of_delta` exists only so the test suite can assert
the difference.

Gold: earned, not total
-----------------------
``ai_AmbientGoldAmount = 9.5`` every ``ai_AmbientGoldInterval = 5.0`` s from
``ai_AmbientGoldDelay = 90.0`` s (``Maps/Map1/Constants.json``) is 1.9 gold/s
that arrives whatever the policy does -- 969 gold over the first ten minutes,
against roughly 1200 from a competent 60-CS ten minutes.  Rewarding
``delta(total gold)`` therefore pays out ~45% of its budget for the passage of
time.  ``subtract_ambient_gold`` removes the trickle at its known rate.
Experience needs no such correction: ``ai_AmbientXPAmount`` is 0.0 on Map1.

Two details on that subtraction.  First, the lump is NOT 9.5 every 5 s, and this
paragraph claimed it was.  ``GlobalData.cs:96-97`` rescales the content values
before anything uses them -- ``AmbientGoldAmount = 9.5 / (10 / 5) / 5 = 0.95``
and ``AmbientGoldInterval = 5.0 * 100 = 500`` ms -- and ``Champion.Update``
(``Champion.cs:236-237``) pays 0.95 on a 500 ms timer.  The net 1.9 gold/s is
right; the granularity is ten times finer.  So the residual on a paying decision
is ``0.008 * (0.95 - 1.9 * dt)`` = **+0.0071 every 15 decisions** at 30 Hz, not
+0.076 every 150.  That is deliberate: the sum over any window is unbiased, and
the residual is a pure function of the game clock, which is in the observation
(``clock_norm``, ``wave_phase_*`` -- the latter is a 30 s sawtooth, so it pins
the 500 ms phase only through its own 6-cycle structure) -- so the value function
absorbs it and it cancels out of the advantage rather than biasing the gradient.
Subtracting on a guessed lump schedule instead would replace a predictable
+0.0071 with an unpredictable +/-0.0071 pair whenever the phase was off by a
tick, which is strictly worse.  The argument only got stronger when the number
was corrected: 0.0071 against 0.5 for a last hit.  Second, this server has no
first-blood shortcut:
``Champion.Update`` only ever tests ``AmbientGoldDelay``, so the
``ai_AmbientGoldDelayFirstBlood = 30`` in the content file is dead and the
delay is always 90 s.

``spend``, and why it is a named zero
-------------------------------------
``gold`` on the wire is ``ch.Stats.Gold``, the WALLET, so a purchase makes it
*fall*.  Scoring ``money * delta(wallet)`` therefore paid -8.0 for buying a
1000g item: the agent was penalised for shopping and rewarded for hoarding.
That half of the fix stands and is load-bearing -- ``money`` is computed on
gold EARNED (``max(0, delta wallet)``, i.e. the wallet delta with any
purchase added back), so a purchase is reward-NEUTRAL.
``test_buying_an_item_is_not_punished`` guards it.

The other half -- a ``spend`` bonus of ``0.25 * money`` "so that gold in items
beats gold in the bank" -- measured exactly +0.00 over a 600 s episode.  It is
now 0.0, for two independent reasons.

1.  **It cannot fire.**  It is a wallet-drop detector and the drop is not on
    the wire.  ``LanerlHooks.OnTick`` runs ``_control.OnTick`` (which emits the
    observation and consumes the ``{"cmd":"reset"}`` line) BEFORE
    ``AutoBuyUndriven``, and ``AutoBuyUndriven`` is fountain-gated -- so the
    single shopping trip of a deathless episode happens in the same server
    tick as the reset, between the last observation of the old episode and the
    first observation of the new one.  Verified on the recordings: the first
    frame of ``lanerl_rl/tests/data/frames_v2.jsonl`` (t = 16 ms) already reads
    ``gold = 0``, the 475 starting gold having gone on ``BuildPath[0]`` (item
    1054, Doran's Shield, 475g) before Python saw a frame, and neither that
    recording nor ``lanerl/logs/state.jsonl`` (4846 frames, 525 s) contains a
    single *decreasing* wallet sample.  Nothing in this module can recover a
    transition that was never emitted.  Putting it on the wire means a
    cumulative-spend counter emitted by ``LanerlControl.BuildObservation``, a
    ``WIRE_FIELDS`` entry and a ``Unit`` field -- all outside this file.

2.  **It should not fire.**  Buying is not an action the policy has.
    ``LanerlHooks.AutoBuyUndriven`` walks a fixed ``BuildPath`` greedily
    whenever the champion is inside ``FountainRadius``; the agent chooses
    neither the items nor the moment, so ``spend`` is one more term paid for
    the passage of time -- the exact thing ``subtract_ambient_gold`` exists to
    delete.  Worse, its largest payouts would land on the WRONG event.
    ``ShopState.BuyOutOnRespawn`` fires on the walk back from a death and buys
    out everything affordable, so at the old 0.002 a respawn that cleared a
    1500g bank paid ``+3.00`` against the ``-0.50`` the kill/death pair charges
    for dying: the reward would have been strictly positive for getting
    killed.  ``test_dying_is_never_profitable`` pins that shut.

    If purchasing ever becomes a real action, restore the weight -- but put the
    counter on the wire first, or it will keep reading zero.

Last-hit shaping
----------------
One potential-based shaping term, in the sense of Ng, Harada & Russell (1999):
``F(s, a, s') = gamma * Phi(s') - Phi(s)`` leaves the optimal policy unchanged
for any bounded real ``Phi``.  Here

    Phi(s) = c * sum over enemy minions of
                 1[dist <= AA_range + eps] * sigmoid((AA_damage - hp_m) / kappa)

with ``c = 0.05`` and ``kappa = 10`` HP: "a minion I could kill right now is
standing in my range".  Being policy-invariant, this can be tuned freely
without changing what the agent is ultimately optimising -- which is exactly
why it is the only shaping term here.

Evaluation is NOT the training reward
-------------------------------------
:class:`AbsoluteLaneMetrics` tracks CS@10min, gold, deaths and the outcome
against a frozen opponent.  Keep it.  In a mirror self-play match under a purely
relative reward, if both agents improve equally the reward stays ~0 and the
win rate stays 50% -- whether both are improving or both are rotting.  A
non-zero-sum absolute yardstick is the only thing that tells those apart.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

from . import constants as C
from .frame import CreepScoreEstimator, Frame, LaneFrame, Unit

__all__ = [
    "RewardWeights",
    "LaneRewardConfig",
    "hp_potential",
    "hp_potential_delta",
    "last_hit_potential",
    "AgentRewardState",
    "ZeroSumLaneReward",
    "AbsoluteLaneMetrics",
    "WinRateTracker",
]


# --------------------------------------------------------------------------
# Potentials
# --------------------------------------------------------------------------


def hp_potential(x: float) -> float:
    """``phi(x) = (x + 1 - (1 - x)**4) / 2`` on an hp fraction.

    ``phi(0) = 0``, ``phi(1) = 1``, ``phi'(0) / phi'(1) = 5``.
    """
    x = 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)
    return (x + 1.0 - (1.0 - x) ** 4) / 2.0


def hp_potential_delta(h_now: float, h_prev: float) -> float:
    """``phi(h_t) - phi(h_{t-1})``.  The correct HP term.  Telescoping."""
    return hp_potential(h_now) - hp_potential(h_prev)


def _wrong_hp_potential_of_delta(h_now: float, h_prev: float) -> float:
    """``phi(h_t - h_{t-1})``.  The BUG.  Present only for the test to reject.

    Note ``hp_potential`` clamps to [0, 1], so this variant is evaluated on the
    unclamped quartic -- otherwise every hp loss would map to exactly 0 and the
    bug would look harmless rather than merely wrong.
    """
    d = h_now - h_prev
    return (d + 1.0 - (1.0 - d) ** 4) / 2.0


def _sigmoid(x: float) -> float:
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def last_hit_potential(
    frame: Frame,
    champ: Optional[Unit],
    enemy_team: int,
    attack_damage: float,
    c: float = C.SHAPING_C,
    kappa: float = C.AA_KILL_KAPPA_HP,
    aa_range: float = C.AA_RANGE_GAREN,
    eps: float = C.AA_RANGE_EPS,
) -> float:
    """``Phi(s)``: killable enemy minions currently inside auto-attack range."""
    if champ is None or not champ.alive:
        return 0.0
    reach = aa_range + C.TARGET_RADIUS["minion"] + eps
    total = 0.0
    for u in frame.units.values():
        if u.etype != "minion" or u.team == champ.team or not u.alive:
            continue
        if math.hypot(u.x - champ.x, u.y - champ.y) > reach:
            continue
        total += _sigmoid((attack_damage - u.hp) / kappa)
    return c * total


def lane_approach_potential(
    lane: "LaneFrame",
    champ: Optional[Unit],
    per_1000: float,
    corridor: float,
) -> float:
    r"""``-per_1000/1000 * distance(champ, lane corridor)``; 0 once inside.

    A potential, in the sense of Ng, Harada & Russell (1999), so the shaping
    ``gamma*Phi(s') - Phi(s)`` it generates is policy-invariant: it changes
    which policies are FOUND, never which policy is optimal.  That is the
    whole reason the walk to lane is paid this way rather than as a per-step
    "closer than last tick" bonus, which is not a potential and would pay an
    agent to oscillate toward and away from the lane forever.

    The corridor is the same rectangle ``lane_presence`` scores -- ``|n| <=
    corridor`` and ``-corridor <= s <= length + corridor`` -- so the potential
    saturates exactly where the indicator starts paying, and the two terms
    hand off instead of double-counting.  Distance is to the RECTANGLE, not to
    the axis: a champion at the correct ``s`` but 3k units off-axis and one at
    ``n = 0`` but sitting in the base are both far, and both get a gradient
    pointing at the nearest piece of lane.
    """
    if champ is None:
        return 0.0
    s, n = lane.point(champ.x, champ.y)
    off_n = max(0.0, abs(n) - corridor)
    off_s = max(0.0, -corridor - s, s - (lane.length + corridor))
    return -(per_1000 / 1000.0) * math.hypot(off_n, off_s)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class RewardWeights:
    """JueWu 1v1 solo-lane weights, retuned for this lane.

    See the module docstring for the provenance of the published table and for
    the arithmetic behind every deviation from it.  Four weights differ, and
    each one has a test that fails if it moves: ``test_a_minion_pays_mostly_for
    _the_last_hit_not_for_standing_there``, ``test_last_hit_dominates_the_
    measured_episode``, ``test_a_winning_trade_is_worth_a_fraction_of_a_cs``
    and ``test_dying_is_never_profitable``.
    """

    #: 2.0 published.  4.0 here: at 2.0 a clean trade was worth 14% of a last
    #: hit and the agent never traded (1.14 of total |r| over 300 s).  Doubling
    #: it is also what makes dying expensive without touching kill/death.
    hp_point: float = 4.0
    tower_hp: float = 10.0
    money: float = 0.008
    #: Garen.json ``BaseMP = 0`` and ``PARType = None``: there is no mana bar to
    #: reward.  Kept as a named zero so the deviation from the published table
    #: is visible rather than silently missing.
    mana: float = 0.0
    #: 0.008 published.  0.001 here: XP is granted to every champion within
    #: ``ai_ExpRadius2 = 1600`` of a dying minion whoever killed it, against a
    #: 190-unit last-hit reach, so at 0.008 fully 48% of a minion's payout was
    #: for proximity.  Kept non-zero because it is the only dense term that
    #: notices the agent leaving lane, being zoned, or lying dead.
    exp: float = 0.001
    death: float = -1.0
    kill: float = -0.5
    #: 0.5 published.  1.0 here: this is the objective, and it is the only term
    #: the policy fully controls.  Read off the server's own
    #: ``ChampStats.MinionsKilled``, so it cannot be farmed by proximity.
    last_hit: float = 1.0
    #: Dense reward for BEING IN THE TOP LANE, per decision. Deliberately
    #: TINY: a tie-breaker, not an incentive.
    #:
    #: 0.0001 is ~1.8 over an 18,000-decision episode, against ~40 for a
    #: 40-CS game -- about 4%, or two CS worth of pull across ten minutes.
    #: Enough to prefer the lane when nothing else distinguishes two states,
    #: far too little to pay for standing in it.
    #:
    #: It was 0.0005 (~9.0, ~18-20%), which is not a tie-breaker, it is a
    #: second objective. The comment then claimed it was "not enough to pay
    #: the agent to stand still" -- an assertion, never measured.
    #:
    #: What the A/B actually showed (runs rl-curric-0914 vs -0914b, no-enemy,
    #: identical but for this weight):
    #:
    #:     upd    with 0.0005     with 0.0
    #:     300    29.4 CS         33.3 CS
    #:     600    29.5            30.1
    #:     900    23.9            21.6
    #:
    #: So at 0.0005 it cost ~4 CS early and did NOT cause the late collapse --
    #: both runs collapse. It is a real cost at that size and an exonerated
    #: suspect for the decline, which is why this is now small rather than
    #: removed.
    #: DEFAULT 0.0. Removed: this repo's own A/B already argued for it and
    #: the weight was merely shrunk instead of dropped. From the table above,
    #: 0.0 was equal or better at every checkpoint measured.
    #:
    #: It is also the only positional term here that is NOT potential-based --
    #: `w * 1[in corridor]` per tick, not `gamma*phi(s') - phi(s)` -- so
    #: unlike lane_approach it carries no policy-invariance guarantee. It is a
    #: standing payment for occupying a rectangle, which biases toward
    #: loitering in lane over doing anything worth more elsewhere, and it pays
    #: whether or not the agent is doing anything while it stands there. The
    #: 0-CS traces collected 1.05/1.07/0.97 of it per episode while converting
    #: 0 of ~82 minions.
    #:
    #: "Be in lane" is implied by the objective: minions die in lane, and
    #: last_hit and money only pay there. It does not need its own line item.
    lane_presence: float = 0.0
    #: How far off the lane axis still counts as "in lane", in game units.
    #: LANE_HALF_WIDTH (1400) is the corridor the observation already uses, so
    #: the reward and the observation agree on where the lane is.
    lane_corridor: float = C.LANE_HALF_WIDTH
    #: Reward per 1000 game units of distance CLOSED toward the lane corridor,
    #: paid as a potential (see ``lane_approach_potential``), not as a rate.
    #:
    #: This addresses a gap ``lane_presence`` structurally cannot: that weight
    #: is an indicator paying only once the agent is ALREADY in the corridor,
    #: so from the fountain its gradient is exactly zero in every direction.
    #: The fountain is 6,835 units out, a decision lasts 33 ms (~11 units of
    #: travel), and re-drawing a direction at 30 Hz is a random walk covering
    #: ~11*sqrt(9000) ~ 1.1k units over a 300 s episode, so a policy with no
    #: prior has no route to its first minion.
    #:
    #: HOW MUCH THAT MATTERS IN PRACTICE IS UNMEASURED, and the honest record
    #: of the run that prompted this weight says it was NOT the binding
    #: constraint there. ``rl-screen-0914`` (first from-scratch run in this
    #: project; every earlier run that farmed was BC-initialised) looked from
    #: the instance logs like a champion that never left base -- 0.00 CS at
    #: level 1.00 flat through 210 s. Its own episode records say otherwise:
    #:
    #:     median 29% of each episode INSIDE the corridor (max 72%)
    #:     50 of 62 episodes ended in death
    #:     5.00 total last-hit reward across all 62 episodes
    #:     corridor share over training: 18/17/26/20/23% -> 0/0/0/0/0%
    #:
    #: It found lane, could not farm once there, died four times in five, and
    #: training correctly taught it to leave: episode return ~-4.5, with lane
    #: presence paying +0.25 against ~-4.8 for dying. That is rational
    #: avoidance, not failed exploration, and 0.478 for the walk cannot
    #: outweigh -4.8 -- this weight would not have rescued that run. The fix
    #: there was a BC prior, which supplies last-hitting and makes the lane
    #: worth standing in.
    #:
    #: It is kept, defaulted on and ablatable via ``--lane-approach 0``,
    #: because it is a genuine potential and so CANNOT change which policy is
    #: optimal -- only which ones are reachable -- and the exploration gap it
    #: closes is real even though it was not what broke that run.
    #:
    #: 0.07/1000 pays 0.478 over that walk -- about half a last hit, and
    #: deliberately under the 1.0 death weight, because the potential is also
    #: what makes a death sting twice: dying teleports the champion to the
    #: fountain, a real -0.478 step down in Phi. That is recovered by walking
    #: back, so the SUM is unchanged and dying stays strictly unprofitable
    #: (``test_dying_is_never_profitable`` and
    #: ``test_a_death_costs_the_walk_back_and_no_more`` cover it) -- but it is
    #: the reason this is 0.07 and not the 0.29 that would pay a full 2.0.
    lane_approach: float = 0.07
    #: Gold converted into items.  ZERO, deliberately -- see the module
    #: docstring's "``spend``, and why it is a named zero".  Two reasons: the
    #: wallet drop is never on the wire (the scripted buy happens between the
    #: reset and the first observation of the episode), and buying is not an
    #: action the policy has, so paying for it would pay for the passage of
    #: time -- and, because ``AutoBuyUndriven`` buys out the bank on the trip
    #: back from a death, would have paid +3.00 for getting killed against the
    #: -0.50 the kill/death pair charges.
    #:
    #: The half of that fix which DOES stand is in ``_AgentReward.raw``:
    #: ``money`` is scored on gold EARNED, so a purchase is reward-neutral
    #: rather than an 8.0 penalty for shopping.  Do not remove that with this.
    spend: float = 0.0


@dataclass
class LaneRewardConfig:
    weights: RewardWeights = field(default_factory=RewardWeights)
    #: Zero-sum coefficient ``alpha`` in ``r = r_self - alpha * r_opponent``,
    #: annealed from ``start`` to ``end`` over ``anneal_steps`` training steps.
    #: Starting below 1 lets the agent first learn to farm at all; ending at 1
    #: makes the game exactly zero-sum, which is what it is.
    zero_sum_alpha_start: float = 0.5
    zero_sum_alpha_end: float = 1.0
    zero_sum_anneal_steps: int = 2_000_000
    #: Remove the 1.9 gold/s ambient trickle from the money term.
    subtract_ambient_gold: bool = True
    #: Potential-based last-hit shaping (Ng et al. 1999).  Policy-invariant.
    last_hit_shaping: bool = True
    shaping_c: float = C.SHAPING_C
    shaping_kappa: float = C.AA_KILL_KAPPA_HP
    shaping_eps: float = C.AA_RANGE_EPS
    #: The gamma used in ``F = gamma * Phi(s') - Phi(s)``.  Must match the
    #: trainer's gamma or the shaping stops being policy-invariant.
    gamma: float = field(default_factory=lambda: C.gamma_for_horizon(C.DEFAULT_HORIZON_S))
    aa_range: float = C.AA_RANGE_GAREN

    def alpha(self, train_step: int) -> float:
        if self.zero_sum_anneal_steps <= 0:
            return self.zero_sum_alpha_end
        t = min(max(float(train_step) / self.zero_sum_anneal_steps, 0.0), 1.0)
        return self.zero_sum_alpha_start + t * (self.zero_sum_alpha_end - self.zero_sum_alpha_start)


# --------------------------------------------------------------------------
# Per-agent raw reward
# --------------------------------------------------------------------------


@dataclass
class AgentRewardState:
    hp_frac: float = 0.0
    own_tower_hp: float = 0.0
    enemy_tower_hp: float = 0.0
    gold: float = 0.0
    xp: float = 0.0
    alive: float = 1.0
    t_s: float = 0.0


class _AgentReward:
    """One side's raw (non-zero-sum) reward, before the opponent subtraction."""

    def __init__(self, team: int, cfg: LaneRewardConfig):
        self.team = int(team)
        self.enemy_team = C.TEAM_RED if self.team == C.TEAM_BLUE else C.TEAM_BLUE
        self.cfg = cfg
        self.cs = CreepScoreEstimator(self.team, aa_range=cfg.aa_range)
        # Same anchors and the same handedness rule the observation builder
        # uses, so "in lane" means one thing in this codebase rather than two.
        self._lane = LaneFrame(
            C.TOP_OUTER_TURRET[self.team],
            C.TOP_OUTER_TURRET[self.enemy_team],
            C.NEXUS_POSITION[self.team],
        )
        self.prev: Optional[AgentRewardState] = None
        self.prev_potential: float = 0.0
        #: Has ``prev_potential`` been set from a real frame yet?
        #:
        #: It must not start at 0.0 and be used. ``_AgentReward.raw`` already
        #: guards its own first tick (``if self.prev is None: return 0.0``);
        #: the shaping block had no equivalent, so the first transition of
        #: every episode was charged ``gamma*Phi(s1) - 0`` instead of
        #: ``gamma*Phi(s1) - Phi(s0)``.
        #:
        #: That was harmless until lane_approach existed, because the only
        #: potential was last-hit shaping and Phi(fountain) genuinely IS 0
        #: there -- no minion is within attack range of a fountain. The
        #: approach potential made Phi(fountain) = -0.478, so the unprimed
        #: zero started injecting a phantom -0.478 into the first shaped step
        #: of every episode: measured -0.493506 against a true -0.015072.
        #:
        #: Policy-invariant (a per-episode additive constant cannot change the
        #: argmax) but it is a lie in the reward budget this module accounts
        #: for term by term, it repeats once per episode, and it lands on the
        #: value estimate at episode start -- the state exploration depends on
        #: most.
        self._potential_primed: bool = False
        self.terms: Dict[str, float] = {}
        #: Gold observed leaving the wallet this episode, UNWEIGHTED and not
        #: part of the reward.  Diagnostic only; see the `spend` weight.
        self.spent_gold_total: float = 0.0

    def reset(self) -> None:
        self.cs.reset()
        self.prev = None
        self.prev_potential = 0.0
        self._potential_primed = False
        self.terms = {}
        self.spent_gold_total = 0.0

    # -- snapshot ----------------------------------------------------------

    def snapshot(self, frame: Frame) -> AgentRewardState:
        me = frame.champion_of_team(self.team)
        own_hp = enemy_hp = 0.0
        own_n = enemy_n = 0
        for u in frame.units.values():
            if u.etype != "turret":
                continue
            frac = 0.0 if u.mhp <= 0 else max(0.0, min(1.0, u.hp / u.mhp))
            if u.team == self.team:
                own_hp += frac
                own_n += 1
            elif u.team == self.enemy_team:
                enemy_hp += frac
                enemy_n += 1
        return AgentRewardState(
            hp_frac=0.0 if me is None or me.mhp <= 0 else max(0.0, min(1.0, me.hp / me.mhp)),
            own_tower_hp=own_hp / max(own_n, 1),
            enemy_tower_hp=enemy_hp / max(enemy_n, 1),
            gold=0.0 if me is None else float(me.gold or 0.0),
            xp=0.0 if me is None else float(me.xp or 0.0),
            alive=0.0 if me is None else float(me.hp > 0),
            t_s=frame.t_s,
        )

    # -- one tick ----------------------------------------------------------

    def raw(self, frame: Frame) -> float:
        """The agent's own reward for this tick, terms recorded in ``self.terms``."""
        me = frame.champion_of_team(self.team)
        cur = self.snapshot(frame)
        self.cs.update(frame, me)
        if self.prev is None:
            self.prev = cur
            self.terms = {}
            return 0.0

        w = self.cfg.weights
        p, c = self.prev, cur
        dt = max(0.0, c.t_s - p.t_s)
        terms: Dict[str, float] = {}

        # HP as a POTENTIAL DIFFERENCE.  See the module docstring.
        #
        # A RESPAWN IS NOT A HEAL. `phi` is a potential on the hp LEVEL, so the
        # jump from 0 back to full pays `w * phi(1) = w` -- refunding the whole
        # cost of the death, and then some, because the death only charged
        # `w * phi(h_at_death)`. Measured at the published w = 2.0, dying below
        # ~55% hp was already NET POSITIVE once the zero-sum subtraction added
        # the killer's -0.5 back:
        #
        #     die at 30% hp:  -1.060 (hp) + 2.000 (respawn) - 1.0 (death)
        #                      = -0.060 raw  ->  +0.44 after -(-0.5)
        #
        # and at w = 4.0 it would have been +1.38. The agent was being paid to
        # feed. Suppressing the delta across the dead->alive transition makes
        # the cost of a death exactly `w * phi(h_at_death)` with no rebate,
        # which is the right number: the potential already charged for every
        # point of hp lost on the way down, so the path no longer matters and
        # dying from full always costs the full `w`.
        #
        # This is not a shaping term and carries no gamma, so there is no
        # Ng-et-al. invariance to break here; and the suppression is applied
        # identically to both agents, so the zero-sum antisymmetry is untouched.
        respawned = p.alive < 0.5 <= c.alive
        terms["hp_point"] = (
            0.0 if respawned else w.hp_point * hp_potential_delta(c.hp_frac, p.hp_frac)
        )

        # Towers: mine lost is negative, theirs lost is positive.
        terms["tower_hp"] = w.tower_hp * (
            (p.enemy_tower_hp - c.enemy_tower_hp) - (p.own_tower_hp - c.own_tower_hp)
        )

        # Gold. `gold` is the WALLET, so a purchase makes it drop; scoring the
        # raw wallet delta paid -8.0 for buying a 1000g item, i.e. it punished
        # shopping and rewarded hoarding. Split it:
        #   spent  = the wallet falling with no other explanation (a purchase)
        #   earned = wallet delta + spent, which is income and never negative
        #            for buying
        # so a purchase is reward-NEUTRAL on `money`. That split is the part
        # that matters and it stays.
        #
        # `spend` itself is weighted 0.0 -- see the module docstring. The
        # UNWEIGHTED gold is still accumulated into `spent_gold_total` and
        # reported in the step info, so a future run can tell "the wallet never
        # visibly dropped" (the detector's problem, which is what is happening
        # today) from "it dropped and we chose not to pay for it" (the
        # weight's). A term that reads 0.00 for two different reasons is how
        # this went unnoticed in the first place.
        d_wallet = c.gold - p.gold
        spent = max(0.0, -d_wallet)
        d_gold = d_wallet + spent          # == max(0, d_wallet): income only
        if self.cfg.subtract_ambient_gold and c.t_s >= C.AMBIENT_GOLD_DELAY_S:
            d_gold -= C.AMBIENT_GOLD_PER_S * dt
        terms["money"] = w.money * d_gold
        terms["spend"] = w.spend * spent
        self.spent_gold_total += spent

        terms["exp"] = w.exp * max(0.0, c.xp - p.xp)
        # Garen has no mana bar, so the published table's `mana` term has
        # nothing to measure.  It stays here, explicitly zero, so the deviation
        # is visible in the term breakdown rather than silently absent.
        terms["mana"] = 0.0

        # KDA.  The weights are signed so that the zero-sum subtraction turns
        # them into +/-0.5; see the module docstring.  `kill` is credited by
        # ZeroSumLaneReward, which resolves both champions' deaths at once.
        terms["death"] = w.death if (p.alive > 0.5 >= c.alive) else 0.0
        terms["kill"] = 0.0

        terms["last_hit"] = w.last_hit * self.cs.last_hits_this_step

        # In lane, or not. `n` is the perpendicular offset from the lane axis;
        # `s` is progress along it, so a champion in its own base or past the
        # enemy turret is out of lane even when n is small.
        terms["lane_presence"] = 0.0
        me = frame.champion_of_team(self.team)
        if me is not None and c.alive > 0.5:
            ls, ln = self._lane.point(me.x, me.y)
            in_corridor = abs(ln) <= w.lane_corridor
            on_lane_span = -w.lane_corridor <= ls <= self._lane.length + w.lane_corridor
            terms["lane_presence"] = w.lane_presence if (in_corridor and on_lane_span) else 0.0

        self.terms = terms
        self.prev = cur
        return float(sum(terms.values()))

    # -- shaping -----------------------------------------------------------

    def potential(self, frame: Frame, attack_damage: float) -> float:
        me = frame.champion_of_team(self.team)
        phi = 0.0
        if self.cfg.last_hit_shaping:
            phi += last_hit_potential(
                frame,
                me,
                self.enemy_team,
                attack_damage,
                c=self.cfg.shaping_c,
                kappa=self.cfg.shaping_kappa,
                aa_range=self.cfg.aa_range,
                eps=self.cfg.shaping_eps,
            )
        # Two potentials sum to one potential, so the invariance survives.
        phi += lane_approach_potential(
            self._lane, me,
            per_1000=self.cfg.weights.lane_approach,
            corridor=self.cfg.weights.lane_corridor,
        )
        return phi


# --------------------------------------------------------------------------
# The two-agent reward
# --------------------------------------------------------------------------


class ZeroSumLaneReward:
    """The training reward for both agents, computed together.

    ``step`` must be called once per environment tick, with the *new* frame.

    ``r_team = raw_team - alpha * raw_other + F_team`` where ``F`` is the
    potential-based last-hit shaping (policy-invariant, so it does not break
    the zero-sum property of the optimal policy even though it is not itself
    antisymmetric).  At ``alpha = 1`` and with shaping off, ``r_blue == -r_red``
    exactly.
    """

    def __init__(self, teams: Sequence[int] = (C.TEAM_BLUE, C.TEAM_RED), cfg: Optional[LaneRewardConfig] = None):
        self.cfg = cfg or LaneRewardConfig()
        self.teams = tuple(int(t) for t in teams)
        if len(self.teams) != 2:
            raise ValueError("the lane reward is defined for exactly two agents")
        self.agents = {t: _AgentReward(t, self.cfg) for t in self.teams}
        self._prev_alive: Dict[int, Optional[bool]] = {t: None for t in self.teams}

    def reset(self) -> None:
        for a in self.agents.values():
            a.reset()
        self._prev_alive = {t: None for t in self.teams}

    def step(self, frame: Frame, train_step: int = 0) -> Tuple[Dict[int, float], Dict[str, object]]:
        a_team, b_team = self.teams

        # Deaths this tick, resolved once for both sides so the kill term can be
        # credited to the killer without double counting.
        died: Dict[int, bool] = {}
        #: WHERE each death happened, on the tick the corpse first appears.
        #: Three floats per death is nothing next to a position trace, and it
        #: answers the question a trace was being run to answer: a champion
        #: dying repeatedly under the enemy turret and one dying to the wave
        #: in its own half produce identical `deaths=` counts and completely
        #: different maps.
        death_pos: Dict[int, Tuple[float, float, float]] = {}
        for t in self.teams:
            ch = frame.champion_of_team(t)
            alive = None if ch is None else ch.alive
            was = self._prev_alive[t]
            died[t] = bool(was is True and alive is False)
            if died[t] and ch is not None:
                death_pos[t] = (float(frame.t_ms), float(ch.x), float(ch.y))
            self._prev_alive[t] = alive

        raw: Dict[int, float] = {}
        for t in self.teams:
            other = b_team if t == a_team else a_team
            r = self.agents[t].raw(frame)
            # `kill` fires on the agent that did the killing.
            if died[other]:
                kill_term = self.cfg.weights.kill
                self.agents[t].terms["kill"] = kill_term
                r += kill_term
            raw[t] = r

        alpha = self.cfg.alpha(train_step)
        rewards = {
            a_team: raw[a_team] - alpha * raw[b_team],
            b_team: raw[b_team] - alpha * raw[a_team],
        }

        shaping: Dict[int, float] = {t: 0.0 for t in self.teams}
        # Either potential switched on runs the block. `last_hit_shaping` used
        # to be the only one and so gated the loop; leaving it that way would
        # have made `lane_approach` silently dead whenever last-hit shaping was
        # turned off, which is exactly the ablation someone would run first.
        if self.cfg.last_hit_shaping or self.cfg.weights.lane_approach:
            for t in self.teams:
                ch = frame.champion_of_team(t)
                # Wire AD, not a re-derivation -- see obs.ObservationBuilder.
                # The derivation is 5.0 low at level 1 even after the rune page
                # was modelled, because the mastery page adds Martial Mastery
                # (+5) and Brute Force (+0.55/level) on top.
                # The server's number or nothing -- see constants.py on why
                # there is no longer a Python attack-damage derivation.
                ad = float(ch.ad) if (ch is not None and ch.ad is not None) else 0.0
                agent = self.agents[t]
                phi_next = agent.potential(frame, ad)
                if not agent._potential_primed:
                    # First frame this episode: there is no previous state, so
                    # there is no transition to shape. Prime and pay nothing,
                    # exactly as `raw` does on its own first tick.
                    agent._potential_primed = True
                    agent.prev_potential = phi_next
                    shaping[t] = 0.0
                else:
                    shaping[t] = self.cfg.gamma * phi_next - agent.prev_potential
                    agent.prev_potential = phi_next
                rewards[t] += shaping[t]

        info = {
            "raw": raw,
            "alpha": alpha,
            "shaping": shaping,
            "terms": {t: dict(self.agents[t].terms) for t in self.teams},
            "died": died,
            "death_pos": death_pos,
            # Diagnostic, not reward: gold seen leaving the wallet this
            # episode. `spend` is weighted 0.0 and this is how a run tells a
            # zero weight apart from a detector that never fired.
            "spent_gold": {t: self.agents[t].spent_gold_total for t in self.teams},
        }
        return rewards, info


# --------------------------------------------------------------------------
# Absolute (non-zero-sum) evaluation
# --------------------------------------------------------------------------


class AbsoluteLaneMetrics:
    """The yardstick that self-play cannot fake.

    Zero-sum reward and mirror-match win rate are both blind to a uniform
    change in skill: two agents that improve together, or rot together, produce
    reward ~0 and win rate ~50% either way.  These are absolute, and they are
    the numbers to plot.  Never feed them back as a reward.
    """

    CS_CHECKPOINTS_S = (300.0, 600.0, 900.0)

    def __init__(self, team: int, aa_range: float = C.AA_RANGE_GAREN):
        self.team = int(team)
        self.cs = CreepScoreEstimator(self.team, aa_range=aa_range)
        self.reset()

    def reset(self) -> None:
        self.cs.reset()
        self.deaths = 0
        self.kills = 0
        self.gold = 0.0
        self.level = 1
        self.t_s = 0.0
        self.cs_at: Dict[int, int] = {}
        self.gold_at: Dict[int, float] = {}
        self._prev_alive: Optional[bool] = None
        self._enemy_prev_alive: Optional[bool] = None

    def update(self, frame: Frame) -> None:
        enemy_team = C.TEAM_RED if self.team == C.TEAM_BLUE else C.TEAM_BLUE
        me = frame.champion_of_team(self.team)
        foe = frame.champion_of_team(enemy_team)
        self.cs.update(frame, me)
        self.t_s = frame.t_s
        if me is not None:
            self.gold = float(me.gold or 0.0)
            self.level = int(me.lvl or 1)
            if self._prev_alive is True and not me.alive:
                self.deaths += 1
            self._prev_alive = me.alive
        if foe is not None:
            if self._enemy_prev_alive is True and not foe.alive:
                self.kills += 1
            self._enemy_prev_alive = foe.alive
        for mark in self.CS_CHECKPOINTS_S:
            key = int(mark)
            if key not in self.cs_at and frame.t_s >= mark:
                self.cs_at[key] = self.cs.cs
                self.gold_at[key] = self.gold

    def report(self) -> Dict[str, float]:
        out: Dict[str, float] = {
            "cs": float(self.cs.cs),
            "cs_per_min": float(self.cs.cs) / max(self.t_s / 60.0, 1e-6),
            "gold": self.gold,
            "level": float(self.level),
            "deaths": float(self.deaths),
            "kills": float(self.kills),
            "duration_s": self.t_s,
        }
        for mark in self.CS_CHECKPOINTS_S:
            key = int(mark)
            out[f"cs_at_{key // 60}min"] = float(self.cs_at.get(key, float("nan")))
            out[f"gold_at_{key // 60}min"] = float(self.gold_at.get(key, float("nan")))
        return out


class WinRateTracker:
    """Win rate against a **frozen** opponent (the scripted bot, or an old self).

    Against a co-evolving mirror this number is 50% by construction and carries
    no information; freeze the opponent and it becomes the only honest scalar in
    the run.
    """

    def __init__(self, name: str = "frozen_scripted_bot"):
        self.name = name
        self.wins = 0
        self.losses = 0
        self.draws = 0

    def reset(self) -> None:
        self.wins = self.losses = self.draws = 0

    def record(self, outcome: str) -> None:
        if outcome == "win":
            self.wins += 1
        elif outcome == "loss":
            self.losses += 1
        elif outcome == "draw":
            self.draws += 1
        else:
            raise ValueError(f"outcome must be win/loss/draw, got {outcome!r}")

    @property
    def n(self) -> int:
        return self.wins + self.losses + self.draws

    def rate(self) -> float:
        """Wins + half draws, over games played.  NaN before the first game."""
        if self.n == 0:
            return float("nan")
        return (self.wins + 0.5 * self.draws) / self.n


def lane_outcome(
    frame: Frame,
    team: int,
    cs_self: int,
    cs_enemy: int,
    gold_margin_for_win: float = 300.0,
) -> str:
    """Decide win / loss / draw for one finished lane.

    Priority: a destroyed turret settles it; otherwise the gold lead at the end,
    with CS as the tie-break.  Deliberately simple and deliberately absolute --
    it is compared against a frozen opponent, not against the agent's twin.
    """
    enemy_team = C.TEAM_RED if team == C.TEAM_BLUE else C.TEAM_BLUE
    for u in frame.units.values():
        if u.etype == "turret" and u.hp <= 0:
            return "win" if u.team == enemy_team else "loss"
    me = frame.champion_of_team(team)
    foe = frame.champion_of_team(enemy_team)
    g_me = 0.0 if me is None else float(me.gold or 0.0)
    g_foe = 0.0 if foe is None else float(foe.gold or 0.0)
    if g_me - g_foe > gold_margin_for_win:
        return "win"
    if g_foe - g_me > gold_margin_for_win:
        return "loss"
    if cs_self > cs_enemy:
        return "win"
    if cs_enemy > cs_self:
        return "loss"
    return "draw"
