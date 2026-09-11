"""Play the live policy against a FROZEN opponent, on the training cadence.

Why this module exists
----------------------
``lanerl_train.eval`` has had the anchor machinery -- ``AnchorSpec``,
``default_anchors()``, ``win_rate_vs_anchor``, a Bradley-Terry fit pinned to
``scripted_gold`` -- since before the first real run.  Nothing ever gave it a
game to score.  So every one of the first run's 37 eval reports read::

    win_rate_vs_anchor {bronze: (None, 0), gold: (None, 0), diamond: (None, 0)}

and the only other number in the report, ``score``, is **0.5 by construction**
in a symmetric mirror: both sides are the same weights, so the win rate is 50%
at initialisation, at convergence, and while the policy rots.  That is why
13,475 updates of a policy that farmed 0 CS in ten minutes looked, from the
metrics, exactly like a run that was working.

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
An anchor game is a real ten-minute game.  At this stack's measured ~3.7x
real time that is around 160 s of wall clock, during which the learner is idle.
At ``--eval-every 400`` and 0.33 updates/s an eval cycle comes round about every
20 minutes, so ONE anchor game per cycle costs ~13% of throughput and three
would cost 40%.  Hence :attr:`AnchorEvalConfig.rotate`: one anchor per cycle,
round-robin, so the ladder fills in over three cycles instead of paying for all
of it every time.
"""

from __future__ import annotations

import logging
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
    "score_for_reason",
    "anchor_launch_spec",
    "play_anchor_episodes",
    "AnchorEvaluator",
]

log = logging.getLogger("lanerl_train.anchor_eval")

_TEAM_OF_SIDE: Dict[Side, int] = {BLUE: C.TEAM_BLUE, RED: C.TEAM_RED}


class AnchorEvalError(RuntimeError):
    """Anchor evaluation could not produce a result it is willing to report."""


@dataclass
class AnchorEvalConfig:
    """How much game time one eval cycle is allowed to spend."""

    #: Games per anchor evaluated in one cycle.
    episodes_per_anchor: int = 1
    #: Evaluate ONE anchor per cycle, round-robin, rather than all of them.
    #: See the module docstring for the arithmetic this is protecting.
    rotate: bool = True
    #: Hard bound on decisions per cycle.  A server whose clock stops would
    #: otherwise hold the learner forever, and an eval that hangs a run is worse
    #: than an eval that does not happen.
    max_steps: int = 12_000
    #: Act greedily during evaluation.  Sampling measures the exploration
    #: distribution; the question an anchor answers is "how good is the policy",
    #: which is the mode.
    deterministic: bool = True
    #: The side the agent plays.  Fixed, not alternated: ``LANERL_BOT=purple``
    #: hands RED to the bot, and the observation builder's lane frame is already
    #: side-symmetric (see ``lanerl_rl.obs``), so there is nothing to balance.
    agent_side: Side = BLUE


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


def anchor_launch_spec(anchor: AnchorSpec, base: Optional[ServerLaunchSpec] = None,
                       seed: Optional[int] = None) -> ServerLaunchSpec:
    """A launch spec whose RED champion is this anchor's frozen scripted bot."""
    if anchor.kind != "scripted":
        raise AnchorEvalError(
            f"anchor {anchor.id!r} is kind={anchor.kind!r}; only 'scripted' anchors can be "
            f"played by the in-server bot. A 'policy' anchor needs a second network on "
            f"RED, which this runner does not build."
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
    """
    cfg = config or AnchorEvalConfig()
    agent_team = _TEAM_OF_SIDE[cfg.agent_side]
    out: List[EpisodeResult] = []
    steps = 0
    #: Decisions since each instance's CURRENT episode began.  ``steps`` is the
    #: loop counter, so using it as ``length_steps`` reported the second game as
    #: the length of both and the Nth as the length of all N -- lengths came out
    #: 7, 14, 21, 28 for four identical games.  Same placeholder that made the
    #: training run's episodes look four seconds long; see
    #: ``lane_wiring.collect_rollout``.
    since_reset: Dict[int, int] = {}
    while len(out) < n_episodes and steps < cfg.max_steps:
        _result, dones = driver.step(deterministic=cfg.deterministic)
        steps += 1
        for i in range(driver.env.n):
            since_reset[i] = since_reset.get(i, 0) + 1
        for i, reason in sorted(dones.items()):
            # CS@10 only means anything for an episode that REACHED ten
            # minutes. A game that ended on a death has not had the chance to
            # farm one, and recording 0 there would drag the headline metric
            # down with no warning -- the same mistake vec.cs_at_10 documents.
            cs10: Optional[float] = None
            if reason == "time":
                by_team = driver.cs_at_10(i)
                if agent_team in by_team:
                    cs10 = float(by_team[agent_team])
            out.append(
                EpisodeResult(
                    agent=agent_id,
                    opponent_id=anchor_id,
                    opponent_category="anchor",
                    score=score_for_reason(reason, agent_team),
                    cs_at_10=cs10,
                    length_steps=since_reset.pop(i, 0),
                    reason=reason,
                    instance=i,
                )
            )
            if len(out) >= n_episodes:
                break
    if not out:
        raise AnchorEvalError(
            f"no episode against {anchor_id} finished in {cfg.max_steps} decisions "
            f"({cfg.max_steps / C.DECISION_HZ / 60.0:.0f} minutes of game time). The "
            f"server's clock is not advancing; an eval that silently returns nothing "
            f"is how the last run's ladder stayed empty."
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
                "(anchor reference %s)",
                update,
                agent_id,
                anchor.id,
                len(episodes),
                time.monotonic() - t0,
                sum(e.score for e in episodes) / len(episodes),
                [e.cs_at_10 for e in episodes],
                anchor.reference_cs_at_10,
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
) -> Callable[[AnchorSpec], Tuple[VecDriver, Any]]:
    """A ``driver_factory`` that launches real servers with the anchor's bot config.

    Each anchor gets its own port block, because the drivers are kept alive for
    the life of the run (see :class:`AnchorEvaluator`) and two of them sharing a
    base collide on the first instance -- the second server then dies during
    start-up, which reads as "the anchor is unbeatable".
    """
    from .ports import PortAllocator

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
        driver = VecDriver(
            env=env,
            policies={policy_key: actor},
            adapter_factory=adapters.adapter_factory,
            encoder=adapters.encoder,
            # RED is left to the in-server bot: omitting the key is how a frozen
            # scripted anchor is played without a second network.
            assignments=[SideAssignment(blue=policy_key, red=None) for _ in range(envs)],
            episode=EpisodeSpec(max_game_ms=max_game_ms),
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
