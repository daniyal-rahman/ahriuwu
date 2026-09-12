"""Play the live policy against a FROZEN opponent, on the training cadence.

Why this module exists
----------------------
``lanerl_train.eval`` has had the anchor machinery -- ``AnchorSpec``,
``default_anchors()``, ``win_rate_vs_anchor``, a Bradley-Terry fit pinned to
``scripted_gold`` -- since before the first real run.  Nothing ever gave it a
game to score.  So every one of the first run's 40 eval reports read::

    win_rate_vs_anchor {bronze: (None, 0), gold: (None, 0), diamond: (None, 0)}

and the only other number in the report, ``score``, is **0.5 by construction**
in a symmetric mirror: both sides are the same weights, so the win rate is 50%
at initialisation, at convergence, and while the policy rots.  That is why
16,200 updates of a policy that farmed 0 CS in ten minutes looked, from the
metrics, exactly like a run that was working.  (The 0 CS is not an artefact of
the broken readout: all 13,520 ``LANERL_CS`` lines in that run's own instance
logs read ``cs=0``, and ``lvl=1`` as late as ``t=574214``.)

The fix is not a better metric.  It is playing a game against something that
does not move: the in-server scripted Garen, frozen at three difficulties with
published CS@10 values (bronze 16.7, gold 29.5, diamond 35.2).

How the game is set up
----------------------
No second network is needed.  ``SideAssignment(blue=<policy>, red=None)`` makes
the action line omit the ``"red"`` key, ``LanerlControl.ApplyActions`` finds no
object for it, and the bot's own orders stand -- with ``LANERL_BOT=purple`` and
``LANERL_BOT_CONFIG`` pointing at the anchor's difficulty JSON.  Both of those
already exist on ``ServerLaunchSpec``; this module is the piece that puts them
together and turns the outcome into an :class:`~lanerl_train.run.EpisodeResult`
the evaluator can score.

Cost, stated up front
---------------------
An anchor game is a real ten-minute game.  The 3.7x real time this used to
quote is ``bench/out/process_restart.json`` (``sim10min_s`` median 163.5 s),
which is a FREERUN server with no control channel and no policy attached, and
it predates the 15 -> 30 Hz change that doubled the decisions an anchor game
costs -- so treat ~160 s of idle learner as a floor, not a measurement.  The
downstream 13% checks out against the recorded run's real 0.302 updates/s
(160 / (400 / 0.302) = 12%), so at ``--eval-every 400`` ONE anchor game per
cycle costs on the order of 13% of throughput and three would cost 40%.
UNSETTLED: nothing has yet timed a real ``play_anchor_episodes`` cycle; the
``anchor_eval`` metrics row records ``elapsed_s`` and no run dir has one.  Hence :attr:`AnchorEvalConfig.rotate`: one anchor per cycle,
round-robin, so the ladder fills in over three cycles instead of paying for all
of it every time.

That arithmetic was written for ONE game per anchor per cycle, and one game
cannot resolve anything: see :attr:`AnchorEvalConfig.episodes_per_anchor` for
the measured power table.  A cycle now costs ``episodes_per_anchor`` games,
divided by ``--anchor-envs`` servers running them in lockstep, so the knob that
pays for statistical power without stalling the learner is ``--anchor-envs``,
and the knob that pays for it less often is ``--eval-every``.
"""

from __future__ import annotations

import logging
import math

import torch
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from lanerl_rl import constants as C

from .eval import AnchorSpec, validate_anchors
from .protocols import BLUE, RED, Side
from .run import EpisodeResult
from .vec import EpisodeSpec, ServerLaunchSpec, SideAssignment, VecDriver, VecLaneEnv

__all__ = [
    "AnchorEvalConfig",
    "AnchorEvalError",
    "GAME_DECISIONS",
    "score_for_reason",
    "score_for_deaths",
    "anchor_launch_spec",
    "play_anchor_episodes",
    "AnchorEvaluator",
]

log = logging.getLogger("lanerl_train.anchor_eval")

_TEAM_OF_SIDE: Dict[Side, int] = {BLUE: C.TEAM_BLUE, RED: C.TEAM_RED}

#: The policy map key a FROZEN network anchor is filed under.
#:
#: This constant was referenced by ``make_anchor_driver_factory`` and never
#: defined, so the ``kind == "policy"`` branch was a ``NameError`` waiting for
#: its first caller.  The factory is invoked lazily -- once, on the first eval
#: cycle that rotates to that anchor -- so ``--anchors ...,bc_policy`` did not
#: fail at startup or at update 1; it failed at update 800, an hour into a GPU
#: run, from inside ``TrainingLoop.step_once``, which ends the run.  Nothing
#: caught it because the tests only ever exercised ``anchor_launch_spec`` and
#: ``anchors_for_run``, never the factory that builds the driver.
_ANCHOR_KEY = "anchor"

#: Decision rounds in one full ten-minute game at the current decision rate.
#:
#: DERIVED, never a literal.  The bound below it feeds was once 12_000, which
#: was a 15 Hz number; the 15 -> 30 Hz change silently halved the game that
#: bound allowed and made ``reason == "time"`` unreachable.
GAME_DECISIONS: int = int(round(600_000.0 / C.DECISION_DT_MS))

#: Slack on the derived step bound: the reset step between in-process episodes,
#: plus instances that finish at slightly different ticks.
_STEP_BUDGET_SLACK = 1.2


class AnchorEvalError(RuntimeError):
    """Anchor evaluation could not produce a result it is willing to report."""


@dataclass
class AnchorEvalConfig:
    """How much game time one eval cycle is allowed to spend."""

    #: Games per anchor evaluated in one cycle.
    #:
    #: WHY 35 AND NOT 1
    #: ----------------
    #: CS@10 is the headline metric and it is NOISY.  Measured on the 144
    #: self-play episodes of ``runs/rl-bc4-0912/metrics.jsonl`` (every one
    #: ``reason == "time"``, so every one carries a real CS@10):
    #:
    #:     n = 144   mean 36.44   sd 7.31 (95% CI 6.55 - 8.26)
    #:     median 36   min 11   max 53
    #:
    #: That sd is the whole problem.  Power to detect a difference of ``d`` CS
    #: between two groups of ``n`` games each (two-sided t, alpha = 0.05,
    #: power = 0.80, sd = 7.31, exact noncentral t):
    #:
    #:     d (CS)   |  2    3    5    7   10   12   15   20
    #:     n/group  | 211   95   35   19   10    7    5    4
    #:
    #: Read the other way -- the smallest difference ``n`` games per group can
    #: detect at 80% power, and the 95% CI on a single cycle's mean:
    #:
    #:     n/group  |   1     2     4     8    16    24    35    48
    #:     MDE (CS) |  29*  41.3  17.4  11.1   7.5   6.0   5.0   4.2
    #:     CI +-    |  n/a  65.7  11.6   6.1   3.9   3.1   2.5   2.1
    #:
    #:     (*) n=1 has no within-group df, so the exact-t MDE is undefined;
    #:         29.0 is the normal approximation, 2.80 * 7.31 * sqrt(2/n).
    #:
    #: At the old default of ONE game per anchor per 400 updates the evaluator
    #: could not distinguish the 36 CS this run actually farmed from 7 CS.  It
    #: is not a weak measurement, it is not a measurement: every anchor number
    #: this project has ever produced was a single draw from a distribution
    #: three times wider than the effects being argued about.
    #:
    #: 35 is the cell that buys ~5 CS.  It is EXPENSIVE -- 35 real ten-minute
    #: games, and the module docstring's floor for one is ~160 s of wall clock,
    #: so ~93 minutes per anchor per cycle at ``--anchor-envs 1`` and ~13
    #: minutes at ``--anchor-envs 8``, which run in lockstep.  Lower it
    #: deliberately if you must, and then quote the MDE row you actually
    #: bought, not the number you wanted.
    #:
    #: Caveats, stated so they are not quoted away:  the sd is measured in
    #: SELF-PLAY, because every anchor game in that run reported ``cs_at_10 =
    #: 0.0`` (the argmax-is-noop bug); anchor CS may well be more variable, and
    #: 7.31 is then an under-estimate.  The table is also for the UNPAIRED
    #: comparison -- two eval cycles, or two checkpoints.  Comparing the agent
    #: with the anchor IN THE SAME GAME (``EpisodeResult.opponent_cs_at_10``)
    #: is paired and therefore cheaper, but the sd of that difference has never
    #: been measured, so there is no honest table for it yet.
    episodes_per_anchor: int = 35
    #: Evaluate ONE anchor per cycle, round-robin, rather than all of them.
    #: See the module docstring for the arithmetic this is protecting.
    rotate: bool = True
    #: Hard bound on decision ROUNDS per cycle, or ``None`` to derive one.  A
    #: server whose clock stops would otherwise hold the learner forever, and
    #: an eval that hangs a run is worse than an eval that does not happen.
    #:
    #: DERIVED, never a literal -- it was 12_000, and that was a 15 Hz number.
    #: A ten-minute game (``AnchorEvaluator.max_game_ms``, 600 s) is 18,000
    #: decisions at the current 30 Hz, so the loop below gave up at 400 game
    #: seconds: ``reason == "time"`` was unreachable, every anchor game scored
    #: ``cs_at_10 = None``, and a game in which nobody died produced no episode
    #: at all and raised :class:`AnchorEvalError` -- which propagates out of
    #: ``TrainingLoop.step_once`` and ends the run.  ``EpisodeSpec.max_steps``
    #: (vec.py, 20_000) was sized for 30 Hz; this one was not, and two bounds on
    #: the same quantity is the bug.
    #:
    #: It was ALSO a one-game bound while ``episodes_per_anchor`` was a free
    #: parameter, so asking for n>1 games would have quietly returned one game
    #: and no error (``play_anchor_episodes`` only raises when it collects
    #: NOTHING) -- an underpowered result wearing the label of a powered one.
    #: ``None`` now means :meth:`steps_budget`: one full game per WAVE of
    #: instances, times the number of waves, plus 20% slack.  Pass an int to
    #: override, which is the only thing a test or a hang-hunt should do.
    max_steps: Optional[int] = None
    #: Sample actions; do NOT take the argmax.
    #:
    #: This was True, and it made every anchor number this project has ever
    #: produced meaningless. Measured on the BC checkpoint, same weights, same
    #: server, 240 s:
    #:
    #:     deterministic=False   dist_from_spawn 10,975   cs 6   lvl 3
    #:     deterministic=True    dist_from_spawn      0   cs 0   lvl 1
    #:
    #: 7,199 of 7,199 argmax actions were noop. That is not a degenerate
    #: policy -- it is the correct mode of an honest distribution. Because the
    #: server pathfinds, the bot issues ONE move order and then idles for many
    #: frames while it walks: a real 90 s demo game is {'noop': 836, 'move':
    #: 11}. BC clones that faithfully, so ~99% of the early-game button mass is
    #: noop and the argmax is noop in every state. Sampling issues the rare
    #: move orders that actually carry the champion, and each one paths
    #: thousands of units.
    #:
    #: So for THIS action space the mode is not a summary of the policy, it is
    #: a different and much worse policy. Anchor games reported the agent at
    #: level 1, full HP, 0 CS, 0 deaths -- standing in the fountain for ten
    #: minutes -- while the same weights farm 36 CS in self-play and 37.3
    #: against the same bot under sampling.
    deterministic: bool = False
    #: The side the agent plays.  Fixed, not alternated: ``LANERL_BOT=purple``
    #: hands RED to the bot, and the observation builder's lane frame is already
    #: side-symmetric (see ``lanerl_rl.obs``), so there is nothing to balance.
    agent_side: Side = BLUE

    def steps_budget(self, n_episodes: int, n_envs: int = 1) -> int:
        """Decision rounds one cycle may spend collecting ``n_episodes`` games.

        A "round" is one :meth:`VecDriver.step`, which advances EVERY instance,
        so ``n_envs`` games are played per ``GAME_DECISIONS`` rounds and the
        budget scales with the number of WAVES, not with the number of games.
        An explicit :attr:`max_steps` wins: a test that wants the clock-stopped
        failure needs to be able to ask for a tiny bound.
        """
        if self.max_steps is not None:
            return int(self.max_steps)
        waves = math.ceil(max(1, int(n_episodes)) / max(1, int(n_envs)))
        return int(round(_STEP_BUDGET_SLACK * waves * GAME_DECISIONS))


def score_for_reason(reason: str, agent_team: int) -> float:
    """The agent's result, in the 1.0 / 0.5 / 0.0 convention ``MatchRecord`` wants.

    A lane that reaches ten minutes with neither champion dead is the COMMON
    case here, not an edge case, and it is a genuine draw: folding it into a
    loss would put every anchor win rate near zero and make the ladder useless.
    CS@10 is what separates two draws.
    """
    if reason.startswith("death_team_"):
        died = reason[len("death_team_") :]
        if died == str(agent_team):
            return 0.0
        return 1.0
    # "time" (reached max_game_ms) and "max_steps" are both draws.
    return 0.5


def _champion_of_team(raw: Mapping[str, Any], team: int) -> Optional[Mapping[str, Any]]:
    for u in raw.get("u", ()):
        if u.get("k") == "Champion" and int(u.get("tm", -1)) == int(team):
            return u
    return None


def _page_canary(raw: Optional[Mapping[str, Any]], team: int) -> Tuple[
    Optional[float], Optional[float]
]:
    """``(ad, mhp)`` for one team's champion, or ``(None, None)``.

    Feeds ``EpisodeResult.first_frame_ad`` / ``first_frame_mhp``: the canary for
    a reset that strips the rune page (mhp 672 -> 616, ad 78.14 -> 57.88).  An
    absent field stays None -- three wrong attack-damage constants in a row came
    out of defaulting one.
    """
    if raw is None:
        return None, None
    u = _champion_of_team(raw, team)
    if u is None:
        return None, None
    ad = u.get("ad")
    mhp = u.get("mhp")
    return (None if ad is None else float(ad), None if mhp is None else float(mhp))


def score_for_deaths(agent_deaths: int, opponent_deaths: int) -> float:
    """The agent's result in a game that was PLAYED OUT rather than stopped.

    ``score_for_reason`` can only read the reason an episode ended, so once
    anchor games stopped ending at the first death (see
    ``make_anchor_driver_factory(end_on_death=...)``) every one of them ended
    with ``reason == "time"`` and scored exactly 0.5.  That is the disease this
    whole module exists to cure -- ``score`` was already 0.5 by construction in
    a symmetric mirror, and a ladder whose every rung reads 0.50 tells you
    nothing -- so it must not be re-introduced through the back door by a fix
    to a different bug.

    The death DIFFERENTIAL over the full ten minutes is the replacement: it
    agrees with ``score_for_reason`` on a game that ended at the first death
    (1-0 one way or the other), and it keeps a deathless lane a genuine draw,
    which is the common case and what CS@10 is there to separate.
    """
    if agent_deaths < opponent_deaths:
        return 1.0
    if agent_deaths > opponent_deaths:
        return 0.0
    return 0.5


def anchor_launch_spec(anchor: AnchorSpec, base: Optional[ServerLaunchSpec] = None,
                       seed: Optional[int] = None) -> ServerLaunchSpec:
    """A launch spec for this anchor.

    ``scripted``: RED is the in-server bot, configured from the anchor's JSON.
    ``policy``: NO in-server bot at all -- RED is driven over the control
    channel by a second, frozen network, so the server must be told
    ``LANERL_BOT=none`` or the bot would fight the network for the same
    champion (the ``LANERL_BOT`` default-to-"blue" bug, in a new costume).
    """
    if anchor.kind not in ("scripted", "policy"):
        raise AnchorEvalError(
            f"anchor {anchor.id!r} is kind={anchor.kind!r}; expected 'scripted' or 'policy'"
        )
    if anchor.kind == "policy":
        src = base or ServerLaunchSpec()
        if anchor.resource is None or not Path(anchor.resource).exists():
            raise AnchorEvalError(
                f"policy anchor {anchor.id!r} has no usable checkpoint ({anchor.resource})"
            )
        return ServerLaunchSpec(
            config_path=src.config_path,
            server_dir=src.server_dir,
            dotnet_root=src.dotnet_root,
            step_ticks=src.step_ticks,
            toponly=src.toponly,
            freerun=src.freerun,
            bot_teams="none",
            bot_seed=seed,
            extra_env=dict(src.extra_env),
            connect_timeout_s=src.connect_timeout_s,
            shutdown_timeout_s=src.shutdown_timeout_s,
        )
    if anchor.resource is None or not Path(anchor.resource).exists():
        raise AnchorEvalError(
            f"anchor {anchor.id!r} has no usable resource ({anchor.resource}); "
            f"validate_anchors() should have refused this run at startup"
        )
    src = base or ServerLaunchSpec()
    return ServerLaunchSpec(
        config_path=src.config_path,
        server_dir=src.server_dir,
        dotnet_root=src.dotnet_root,
        step_ticks=src.step_ticks,
        toponly=src.toponly,
        freerun=src.freerun,
        bot_teams=anchor.bot_teams,
        bot_config=Path(anchor.resource),
        bot_seed=seed,
        extra_env=dict(src.extra_env),
        connect_timeout_s=src.connect_timeout_s,
        shutdown_timeout_s=src.shutdown_timeout_s,
    )


def play_anchor_episodes(
    driver: VecDriver,
    anchor_id: str,
    agent_id: str,
    n_episodes: int,
    config: Optional[AnchorEvalConfig] = None,
) -> List[EpisodeResult]:
    """Drive ``driver`` until ``n_episodes`` have finished, and score them.

    ``driver`` must already be started and assigned with the agent on
    ``config.agent_side`` and the other side left to the in-server bot.

    Deaths are counted here rather than read off the episode's end reason,
    because an anchor game now plays through them
    (``make_anchor_driver_factory(end_on_death=False)``) and would otherwise
    score 0.5 every time.  See :func:`score_for_deaths`.
    """
    cfg = config or AnchorEvalConfig()
    agent_team = _TEAM_OF_SIDE[cfg.agent_side]
    budget = cfg.steps_budget(n_episodes, driver.env.n)
    out: List[EpisodeResult] = []
    steps = 0
    #: Decisions since each instance's CURRENT episode began.  ``steps`` is the
    #: loop counter, so using it as ``length_steps`` reported the second game as
    #: the length of both and the Nth as the length of all N -- lengths came out
    #: 7, 14, 21, 28 for four identical games.  Same placeholder that made the
    #: training run's episodes look four seconds long; see
    #: ``lane_wiring.collect_rollout``.
    since_reset: Dict[int, int] = {}
    #: ``instance -> team -> deaths in the CURRENT episode``, edge-triggered on
    #: hp crossing zero.  A corpse reports hp <= 0 for every frame until it
    #: respawns, so a level-triggered count would score one death as sixty.
    deaths: Dict[int, Dict[int, int]] = {}
    alive: Dict[Tuple[int, int], bool] = {}
    #: ``instance -> (ad, mhp)`` for the agent, from the earliest frame of the
    #: current episode this call has seen.  The drivers are cached for the life
    #: of the run, so for the FIRST episode of a cycle that frame is mid-game --
    #: which is fine for a canary whose whole point is that these values must
    #: not change within an episode.
    page: Dict[int, Tuple[Optional[float], Optional[float]]] = {}
    while len(out) < n_episodes and steps < budget:
        result, dones = driver.step(deterministic=cfg.deterministic)
        steps += 1
        for i in range(driver.env.n):
            since_reset[i] = since_reset.get(i, 0) + 1
            # On a boundary step ``obs[i]`` is already the POST-RESET frame --
            # full hp, new episode -- and ``terminal_obs[i]`` is the frame that
            # ended the game. Reading obs[i] there would miss a death on the
            # last frame and attribute a fresh champion to the finished game.
            raw = result.terminal_obs.get(i) or result.obs[i]
            if raw is None:
                continue
            if i not in page:
                page[i] = _page_canary(raw, agent_team)
            for u in raw.get("u", ()):
                if u.get("k") != "Champion":
                    continue
                team = int(u.get("tm", -1))
                standing = int(u.get("hp", 1)) > 0
                if alive.get((i, team), True) and not standing:
                    per_team = deaths.setdefault(i, {})
                    per_team[team] = per_team.get(team, 0) + 1
                alive[(i, team)] = standing
        for i, reason in sorted(dones.items()):
            # CS@10 only means anything for an episode that REACHED ten
            # minutes. A game that ended on a death has not had the chance to
            # farm one, and recording 0 there would drag the headline metric
            # down with no warning -- the same mistake vec.cs_at_10 documents.
            cs10: Optional[float] = None
            opp_cs10: Optional[float] = None
            if reason == "time":
                by_team = driver.cs_at_10(i)
                if agent_team in by_team:
                    cs10 = float(by_team[agent_team])
                # The anchor's OWN CS from this same game -- the honest
                # yardstick, replacing a hardcoded reference measured on a
                # different champion (see EpisodeResult.opponent_cs_at_10).
                for t, v in by_team.items():
                    if t != agent_team:
                        opp_cs10 = float(v)
            # Clear the per-episode death bookkeeping BEFORE scoring the next
            # game on this instance; a champion is alive again after a reset.
            by_deaths = deaths.pop(i, {})
            for key in [k for k in alive if k[0] == i]:
                alive[key] = True
            agent_deaths = int(by_deaths.get(agent_team, 0))
            opp_deaths = sum(int(v) for t, v in by_deaths.items() if t != agent_team)
            score = (
                score_for_reason(reason, agent_team)
                if reason.startswith("death_team_")
                else score_for_deaths(agent_deaths, opp_deaths)
            )
            length = since_reset.pop(i, 0)
            ad, mhp = page.pop(i, (None, None))
            out.append(
                EpisodeResult(
                    agent=agent_id,
                    opponent_id=anchor_id,
                    opponent_category="anchor",
                    score=score,
                    cs_at_10=cs10,
                    length_steps=length,
                    reason=reason,
                    instance=i,
                    opponent_cs_at_10=opp_cs10,
                    # Same convention as lane_wiring._accumulate_episode: a
                    # champion death on the other team is a kill for this one.
                    # In a 1v1 lane the wave can also do it, so treat kills as
                    # "the opponent died", not "the agent killed it".
                    deaths=agent_deaths,
                    kills=opp_deaths,
                    first_frame_ad=ad,
                    first_frame_mhp=mhp,
                )
            )
            # Per GAME, not just per cycle: "41 CS with 0 deaths" and "41 CS
            # with 5 deaths" are different games, and the cycle-level line
            # averages them away.
            log.info(
                "anchor %s game %d/%d on instance %d: %s in %d decisions, score %.1f, "
                "deaths %d-%d, CS@10 %s (anchor %s)",
                anchor_id, len(out), n_episodes, i, reason, length, score,
                agent_deaths, opp_deaths, cs10, opp_cs10,
            )
            if len(out) >= n_episodes:
                break
    if not out:
        raise AnchorEvalError(
            f"no episode against {anchor_id} finished in {budget} decisions "
            f"({budget / C.DECISION_HZ / 60.0:.0f} minutes of game time). The "
            f"server's clock is not advancing; an eval that silently returns nothing "
            f"is how the last run's ladder stayed empty."
        )
    if len(out) < n_episodes:
        # A SHORT cycle is the underpowered-result failure wearing the label of
        # a powered one: the caller asked for enough games to resolve ~5 CS and
        # got fewer, with a mean whose CI is wider than it thinks. Loud, and
        # with the arithmetic, because the usual cause is a max_steps override
        # or an n_episodes that does not divide into the instances available.
        log.error(
            "anchor %s: asked for %d game(s), collected %d in %d decision rounds "
            "(budget %d, %d instance(s)). The reported mean is more uncertain than "
            "episodes_per_anchor promises -- see AnchorEvalConfig.episodes_per_anchor "
            "for what %d games can actually resolve.",
            anchor_id, n_episodes, len(out), steps, budget, driver.env.n, len(out),
        )
    return out


class AnchorEvaluator:
    """The callable :class:`lanerl_train.run.TrainingLoop` invokes on its eval cadence.

    ``driver_factory(anchor) -> (driver, actor)`` owns the servers.  It is called
    lazily, once per anchor, and the result is cached for the life of the run:
    a process restart costs ~12 s against 0.23 ms for an in-process episode
    reset, so tearing the anchor's servers down between evals would cost more
    than the evaluation.
    """

    def __init__(
        self,
        anchors: Sequence[AnchorSpec],
        driver_factory: Callable[[AnchorSpec], Tuple[VecDriver, Any]],
        agent_id_fn: Callable[[], str],
        config: Optional[AnchorEvalConfig] = None,
    ):
        validate_anchors(anchors)
        self.anchors = list(anchors)
        self.driver_factory = driver_factory
        self.agent_id_fn = agent_id_fn
        self.cfg = config or AnchorEvalConfig()
        self._drivers: Dict[str, Tuple[VecDriver, Any]] = {}
        self._next = 0
        self.cycles = 0

    def anchors_for_cycle(self) -> List[AnchorSpec]:
        if not self.cfg.rotate:
            return list(self.anchors)
        a = self.anchors[self._next % len(self.anchors)]
        self._next += 1
        return [a]

    def __call__(self, update: int, payload: Mapping[str, Any]) -> List[EpisodeResult]:
        self.cycles += 1
        results: List[EpisodeResult] = []
        agent_id = self.agent_id_fn()
        for anchor in self.anchors_for_cycle():
            t0 = time.monotonic()
            driver, actor = self._driver_for(anchor)
            self._load(actor, payload)
            episodes = play_anchor_episodes(
                driver, anchor.id, agent_id, self.cfg.episodes_per_anchor, self.cfg
            )
            results.extend(episodes)
            log.info(
                "ANCHOR EVAL update=%d %s vs %s: %d game(s) in %.0fs, score %.2f, CS@10 %s "
                "(anchor scored %s in the same games)",
                update,
                agent_id,
                anchor.id,
                len(episodes),
                time.monotonic() - t0,
                sum(e.score for e in episodes) / len(episodes),
                [e.cs_at_10 for e in episodes],
                # measured in the SAME games, not a constant from a config
                # that no longer exists
                [e.opponent_cs_at_10 for e in episodes],
            )
        return results

    # -- internals ---------------------------------------------------------

    def _driver_for(self, anchor: AnchorSpec) -> Tuple[VecDriver, Any]:
        if anchor.id not in self._drivers:
            self._drivers[anchor.id] = self.driver_factory(anchor)
        return self._drivers[anchor.id]

    @staticmethod
    def _load(actor: Any, payload: Mapping[str, Any]) -> None:
        """Point the eval actor at the CURRENT weights.

        Without this the anchor ladder measures whatever the eval actor was
        constructed with -- a frozen random policy -- forever, while looking
        exactly like a working evaluation. The same class of mistake as the
        anchors never being given a resource.
        """
        if not payload:
            raise AnchorEvalError(
                "the parameter store handed the anchor evaluator an empty payload, so "
                "the eval would run on the actor's construction-time weights and report "
                "a win rate for a policy that is not being trained"
            )
        policy = getattr(actor, "policy", None)
        loader = getattr(policy, "load_state_dict", None)
        if not callable(loader):
            raise AnchorEvalError(
                f"anchor eval actor {type(actor).__name__} exposes no .policy."
                f"load_state_dict(); it cannot be given the current weights"
            )
        loader(payload["policy"])

    def close(self) -> None:
        for anchor_id, (driver, _actor) in self._drivers.items():
            try:
                driver.env.close()
            except Exception:
                log.error("error closing anchor %s's servers", anchor_id, exc_info=True)
        self._drivers.clear()


def make_anchor_driver_factory(
    build_policy_actor: Callable[[], Any],
    policy_key: str,
    port_base: int,
    log_dir: Path,
    adapter_factory_for: Callable[[], Any],
    envs: int = 1,
    max_game_ms: int = 600_000,
    base_spec: Optional[ServerLaunchSpec] = None,
    seed: int = 0,
    port_stride: int = 64,
    end_on_death: bool = False,
) -> Callable[[AnchorSpec], Tuple[VecDriver, Any]]:
    """A ``driver_factory`` that launches real servers with the anchor's bot config.

    Each anchor gets its own port block, because the drivers are kept alive for
    the life of the run (see :class:`AnchorEvaluator`) and two of them sharing a
    base collide on the first instance -- the second server then dies during
    start-up, which reads as "the anchor is unbeatable".

    ``end_on_death`` DEFAULTS TO FALSE, which is the opposite of
    ``EpisodeSpec``'s own default and deliberate.  This line used to build
    ``EpisodeSpec(max_game_ms=max_game_ms)`` and inherit ``end_on_death=True``,
    while the training drivers were launched with ``--no-end-on-death``: the
    eval and the thing it was evaluating did not agree on what an episode IS.
    Two consequences, both invisible:

    * ``play_anchor_episodes`` only records CS@10 for ``reason == "time"``, so
      anchor CS@10 was measured on DEATHLESS GAMES ONLY -- pure survivorship
      bias.  A policy that learns to trade, and therefore sometimes dies, is
      scored as though it stopped farming.  ``runs/rl-bc4-0912`` has the shape
      of it: 5 of its 6 anchor games ended on the clock, the 6th ended
      ``death_team_100`` at 12,721 steps and contributed ``cs_at_10 = None``,
      while all 144 self-play episodes of the same run ran the full 18,001.
    * a ten-minute game is the unit CS@10 is DEFINED on.  An episode cut short
      at 103 s (the first BC-init run's mean) cannot produce one at all.

    Pass ``True`` only to reproduce the old behaviour deliberately.
    """
    from .ports import PortAllocator

    if policy_key == _ANCHOR_KEY:
        raise AnchorEvalError(
            f"the live policy and the frozen policy anchor would both be filed under "
            f"{_ANCHOR_KEY!r} in the driver's policy map, so the anchor would be handed "
            f"the current weights on every cycle and the rung would move with the agent. "
            f"Choose a different policy_key."
        )

    assigned: Dict[str, int] = {}

    def factory(anchor: AnchorSpec) -> Tuple[VecDriver, Any]:
        idx = assigned.setdefault(anchor.id, len(assigned))
        allocator = PortAllocator(base=port_base + idx * envs * port_stride)
        ports = allocator.allocate(envs)
        adapters = adapter_factory_for()
        actor = build_policy_actor()
        env = VecLaneEnv(
            n=envs,
            spec=anchor_launch_spec(anchor, base_spec, seed=seed),
            ports=ports,
            log_dir=Path(log_dir) / f"anchor_{anchor.id}",
        )
        policies = {policy_key: actor}
        if anchor.kind == "policy":
            # A frozen NETWORK on red. This is the rung that answers "is the
            # agent better than the prior it started from" -- the scripted
            # anchors cannot, because they are a different kind of opponent
            # entirely. It stays frozen: built once here, never handed the
            # current weights by AnchorEvaluator._load (which only touches
            # `actor`).
            frozen = build_policy_actor()
            blob = torch.load(anchor.resource, map_location="cpu", weights_only=False)
            state = blob.get("policy", blob)
            frozen.policy.load_state_dict(state)
            frozen.policy.eval()
            policies[_ANCHOR_KEY] = frozen
            assignments = [SideAssignment(blue=policy_key, red=_ANCHOR_KEY)
                           for _ in range(envs)]
        else:
            # RED is left to the in-server bot: omitting the key is how a frozen
            # scripted anchor is played without a second network.
            assignments = [SideAssignment(blue=policy_key, red=None)
                           for _ in range(envs)]
        driver = VecDriver(
            env=env,
            policies=policies,
            adapter_factory=adapters.adapter_factory,
            encoder=adapters.encoder,
            assignments=assignments,
            episode=EpisodeSpec(max_game_ms=max_game_ms, end_on_death=end_on_death),
        )
        log.info(
            "anchor %s: starting %d server(s) at port base %d with bot config %s",
            anchor.id,
            envs,
            port_base + idx * envs * port_stride,
            anchor.resource,
        )
        driver.start()
        return driver, actor

    return factory
