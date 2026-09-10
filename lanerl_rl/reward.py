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
Weights follow the 1v1 solo-lane configuration used by the JueWu line of work
(Ye et al. 2020, "Mastering Complex Control in MOBA Games with Deep
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

The negative ``kill`` weight is not a typo and is not a bug: it is what makes
the kill/death pair antisymmetric once the zero-sum subtraction is applied.
Killing the enemy fires *their* ``death`` (-1.0) and *my* ``kill`` (-0.5), so
``r = r_self - r_opp = -0.5 - (-1.0) = +0.5``; dying gives exactly ``-0.5``.
Take the weights apart and the pair stops balancing.

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

    2.0 * (phi(-0.02) + phi(+0.02)) = 2.0 * (-0.05122 + 0.04882) = -0.0048

per cycle -- a persistent negative drift that teaches the agent to avoid
trading at all.  :func:`hp_potential_delta` is the correct form and
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

Two details on that subtraction.  First, the server pays in 9.5 lumps every 5 s
while we subtract 1.9/s continuously, so a single tick every 5 s carries a
+0.076 residual.  That is deliberate: the sum over any window is unbiased, and
the residual is a pure function of the game clock, which is in the observation
(``clock_norm``, ``wave_phase_*``) -- so the value function absorbs it and it
cancels out of the advantage rather than biasing the gradient.  Subtracting on a
guessed lump schedule instead would replace a predictable +0.076 with an
unpredictable +/-0.076 pair whenever the phase was off by a tick, which is
strictly worse.  Second, this server has no first-blood shortcut:
``Champion.Update`` only ever tests ``AmbientGoldDelay``, so the
``ai_AmbientGoldDelayFirstBlood = 30`` in the content file is dead and the
delay is always 90 s.

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
from .frame import CreepScoreEstimator, Frame, Unit

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
    c: float = 0.05,
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


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass
class RewardWeights:
    """JueWu 1v1 solo-lane weights.  See the module docstring for provenance."""

    hp_point: float = 2.0
    tower_hp: float = 10.0
    money: float = 0.008
    #: Garen.json ``BaseMP = 0`` and ``PARType = None``: there is no mana bar to
    #: reward.  Kept as a named zero so the deviation from the published table
    #: is visible rather than silently missing.
    mana: float = 0.0
    exp: float = 0.008
    death: float = -1.0
    kill: float = -0.5
    last_hit: float = 0.5
    #: Gold converted into items, as a fraction of the `money` weight.
    #:
    #: `gold` in the observation is the WALLET, not lifetime earnings, so a
    #: purchase makes it fall. Scoring `money * delta(wallet)` therefore paid
    #: -8.0 for buying a 1000g item: the agent was penalised for shopping and
    #: rewarded for hoarding. The money term is now computed on gold EARNED
    #: (wallet delta plus whatever was spent this step), which makes a purchase
    #: reward-neutral, and this weight adds a small bonus on top so that gold in
    #: items beats gold in the bank.
    spend: float = 0.002        # 0.25 x money


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
    shaping_c: float = 0.05
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
        self.prev: Optional[AgentRewardState] = None
        self.prev_potential: float = 0.0
        self.terms: Dict[str, float] = {}

    def reset(self) -> None:
        self.cs.reset()
        self.prev = None
        self.prev_potential = 0.0
        self.terms = {}

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
        terms["hp_point"] = w.hp_point * hp_potential_delta(c.hp_frac, p.hp_frac)

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
        # so a purchase is reward-NEUTRAL on `money`, and `spend` then adds a
        # small bonus for turning gold into stats.
        d_wallet = c.gold - p.gold
        spent = max(0.0, -d_wallet)
        d_gold = d_wallet + spent          # == max(0, d_wallet): income only
        if self.cfg.subtract_ambient_gold and c.t_s >= C.AMBIENT_GOLD_DELAY_S:
            d_gold -= C.AMBIENT_GOLD_PER_S * dt
        terms["money"] = w.money * d_gold
        terms["spend"] = w.spend * spent

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

        self.terms = terms
        self.prev = cur
        return float(sum(terms.values()))

    # -- shaping -----------------------------------------------------------

    def potential(self, frame: Frame, attack_damage: float) -> float:
        return last_hit_potential(
            frame,
            frame.champion_of_team(self.team),
            self.enemy_team,
            attack_damage,
            c=self.cfg.shaping_c,
            kappa=self.cfg.shaping_kappa,
            aa_range=self.cfg.aa_range,
            eps=self.cfg.shaping_eps,
        )


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
        for t in self.teams:
            ch = frame.champion_of_team(t)
            alive = None if ch is None else ch.alive
            was = self._prev_alive[t]
            died[t] = bool(was is True and alive is False)
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
        if self.cfg.last_hit_shaping:
            for t in self.teams:
                ch = frame.champion_of_team(t)
                ad = C.garen_attack_damage(int((ch.lvl if ch is not None else 1) or 1))
                agent = self.agents[t]
                phi_next = agent.potential(frame, ad)
                shaping[t] = self.cfg.gamma * phi_next - agent.prev_potential
                agent.prev_potential = phi_next
                rewards[t] += shaping[t]

        info = {
            "raw": raw,
            "alpha": alpha,
            "shaping": shaping,
            "terms": {t: dict(self.agents[t].terms) for t in self.teams},
            "died": died,
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
