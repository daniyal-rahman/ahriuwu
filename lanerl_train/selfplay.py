"""The seam: opponent sampling -> live instances -> episodes -> the learner.

Two constraints from the actual server shape this file, and neither is obvious
from the design sketch.

**A scripted anchor's difficulty is process-level, not episode-level.**
``LanerlBotConfig.Load()`` reads ``LANERL_BOT_CONFIG`` once, at construction.
So "5% of episodes against the frozen bronze bot" cannot be a per-episode draw
on a shared instance: it has to be *instances dedicated* to the scripted
anchors, and changing one anchor's difficulty costs a 12.08 s process restart.
:func:`plan_instances` does that allocation and says loudly when the instance
count is too small to cover the whole ladder at once.  The BC-policy anchor is a
network, so it needs no dedicated instance -- it rides the normal opponent slot.

**Fixed batch slots beat per-episode policy identity.**  A recurrent policy's
hidden state is a column per slot; if the opponent's *identity* changed per
episode, the slots would have to be reindexed every boundary.  Instead each
opponent group owns one permanent slot backed by a :class:`SwappableOpponent`
whose *weights* are swapped.  The number of groups is the batching knob:
measured 1.70 ms/decision at batch 1 against 0.058 ms at batch 24, so with 16
instances, 4 groups costs 5 forwards of batch 16/4/4/4/4 per step instead of 32
forwards of batch 1.  Sampling stays correct in distribution over episodes; it
is merely correlated within a group, which is the price of the batch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from .eval import AnchorSpec, anchor_episode_budget
from .league import LATEST, OpponentSampler, OpponentSpec
from .run import EpisodeResult, Rollout
from .vec import ServerLaunchSpec, SideAssignment, VecDriver

__all__ = [
    "BLUE_TEAM",
    "RED_TEAM",
    "lane_outcome",
    "InstancePlan",
    "plan_instances",
    "SwappableOpponent",
    "SelfPlayCollector",
]

log = logging.getLogger("lanerl_train.selfplay")

BLUE_TEAM = 100
RED_TEAM = 200  # TEAM_PURPLE, keyed as "red" by LanerlControl.ApplyActions


def lane_outcome(
    final_obs: Mapping[str, Any],
    cs: Optional[Mapping[int, int]] = None,
    gold_draw_margin: float = 150.0,
) -> float:
    """Blue's result for a finished lane episode: 1.0 / 0.5 / 0.0.

    Death dominates, then gold.  Gold rather than CS because it already folds in
    CS, kills and turret plates, and unlike CS it is in the observation itself
    rather than only in the log.  A margin below ``gold_draw_margin`` is a draw:
    a ten-minute mirror lane decided by twelve gold is noise, and scoring noise
    as a win is how a rating system fills up with signal that is not there.
    """
    champs = {u["tm"]: u for u in final_obs.get("u", ()) if u.get("k") == "Champion"}
    blue, red = champs.get(BLUE_TEAM), champs.get(RED_TEAM)
    if blue is None or red is None:
        log.error(
            "episode ended with a champion missing from the observation (blue=%s red=%s); "
            "scoring it a draw, but this means the episode is not usable evidence",
            blue is not None,
            red is not None,
        )
        return 0.5
    blue_dead = int(blue.get("hp", 1)) <= 0
    red_dead = int(red.get("hp", 1)) <= 0
    if blue_dead != red_dead:
        return 0.0 if blue_dead else 1.0
    bg, rg = float(blue.get("gold", 0.0)), float(red.get("gold", 0.0))
    if abs(bg - rg) < gold_draw_margin:
        if cs:
            bc, rc = cs.get(BLUE_TEAM), cs.get(RED_TEAM)
            if bc is not None and rc is not None and bc != rc:
                return 1.0 if bc > rc else 0.0
        return 0.5
    return 1.0 if bg > rg else 0.0


# --------------------------------------------------------------------------
# Instance allocation
# --------------------------------------------------------------------------


@dataclass
class InstancePlan:
    """One instance's role for the life of its process."""

    index: int
    role: str  # "policy" | "anchor"
    spec: ServerLaunchSpec
    anchor: Optional[AnchorSpec] = None

    @property
    def opponent_is_scripted(self) -> bool:
        return self.role == "anchor"


def plan_instances(
    n: int,
    base_spec: ServerLaunchSpec,
    anchors: Sequence[AnchorSpec] = (),
    share: float = 0.05,
    bot_seed: int = 1234,
) -> List[InstancePlan]:
    """Split ``n`` instances between self-play and the frozen scripted anchors.

    Only *scripted* anchors need a dedicated instance; a policy anchor (the BC
    net) plays through the ordinary opponent slot.  When there are fewer anchor
    instances than scripted difficulties this logs an error naming exactly which
    rungs of the ladder are unmeasured this segment -- a missing rung looks
    identical to a working setup from the metrics side, and that is precisely
    the class of silence this project keeps paying for.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    scripted = [a for a in anchors if a.kind == "scripted"]
    missing = [a.id for a in scripted if not a.exists()]
    if missing:
        log.error(
            "scripted anchors %s have no config file and cannot be played; the frozen "
            "rung(s) they represent will be absent from every eval report",
            missing,
        )
        scripted = [a for a in scripted if a.exists()]

    n_anchor = min(n - 1, anchor_episode_budget(n, share)) if scripted else 0
    n_anchor = max(n_anchor, 0)
    if scripted and n_anchor == 0:
        log.error(
            "n=%d instances leaves no room for a dedicated scripted-anchor instance at a "
            "%.0f%% share; win rate against a frozen opponent will not be measured",
            n,
            100 * share,
        )
    if 0 < n_anchor < len(scripted):
        covered = [a.id for a in scripted[:n_anchor]]
        log.error(
            "only %d anchor instance(s) for %d scripted difficulties: this segment measures "
            "%s and leaves %s unplayed. Raise n, raise the anchor share, or rotate the "
            "anchor instances deliberately (a difficulty change is a 12s process restart, "
            "because LanerlBotConfig reads LANERL_BOT_CONFIG once).",
            n_anchor,
            len(scripted),
            covered,
            [a.id for a in scripted[n_anchor:]],
        )

    plans: List[InstancePlan] = []
    for i in range(n):
        if i < n_anchor:
            anchor = scripted[i % len(scripted)]
            spec = _clone_spec(
                base_spec,
                bot_teams=anchor.bot_teams,
                bot_config=anchor.resource,
                bot_seed=bot_seed + i,
            )
            plans.append(InstancePlan(i, "anchor", spec, anchor))
        else:
            plans.append(InstancePlan(i, "policy", _clone_spec(base_spec, bot_teams="none")))
    return plans


def _clone_spec(spec: ServerLaunchSpec, **overrides: Any) -> ServerLaunchSpec:
    fields = dict(
        config_path=spec.config_path,
        server_dir=spec.server_dir,
        dotnet_root=spec.dotnet_root,
        step_ticks=spec.step_ticks,
        toponly=spec.toponly,
        freerun=spec.freerun,
        bot_teams=spec.bot_teams,
        bot_config=spec.bot_config,
        bot_seed=spec.bot_seed,
        extra_env=dict(spec.extra_env),
        connect_timeout_s=spec.connect_timeout_s,
        shutdown_timeout_s=spec.shutdown_timeout_s,
    )
    fields.update(overrides)
    return ServerLaunchSpec(**fields)


# --------------------------------------------------------------------------
# The opponent slot
# --------------------------------------------------------------------------


class SwappableOpponent:
    """A permanent batch slot whose *weights* change, not its identity.

    Wraps a policy of the same class as the learner's.  ``set_opponent`` points
    it at the live parameters (for the 40% mirror slice) or loads a snapshot.
    Keeping the slot fixed is what lets the recurrent state stay column-aligned
    across episode boundaries.
    """

    def __init__(self, inner: Any, load_payload: Callable[[Any, Mapping[str, Any]], None]):
        self.inner = inner
        self._load = load_payload
        self.current: Optional[str] = None
        self.swaps = 0

    # -- BatchPolicy passthrough ------------------------------------------

    @property
    def version(self) -> int:
        return getattr(self.inner, "version", 0)

    def initial_state(self, batch: int) -> Any:
        return self.inner.initial_state(batch)

    def act_batch(self, observations, state, resets=None, deterministic=False):
        return self.inner.act_batch(observations, state, resets=resets, deterministic=deterministic)

    # -- swapping ----------------------------------------------------------

    def set_opponent(self, spec: OpponentSpec, live_payload: Mapping[str, Any],
                     load_snapshot: Callable[[OpponentSpec], Mapping[str, Any]]) -> None:
        if spec.id == self.current and spec.id != LATEST:
            return  # already loaded; a reload would be pure cost
        payload = live_payload if spec.is_latest else load_snapshot(spec)
        self._load(self.inner, payload)
        self.current = spec.id
        self.swaps += 1


# --------------------------------------------------------------------------
# The collector
# --------------------------------------------------------------------------


@dataclass
class _EnvEpisode:
    opponent: OpponentSpec
    group: int
    steps: int = 0
    first_t: int = 0


class SelfPlayCollector:
    """Produces one :class:`~lanerl_train.run.Rollout` per call.

    Designed to be handed straight to :class:`~lanerl_train.run.ActorLoop` as
    its ``collect``.  It owns episode accounting -- who played whom, who won,
    what CS@10 was -- and delegates trajectory storage to ``on_step`` so the
    actual PPO buffer stays in ``lanerl_rl``.
    """

    def __init__(
        self,
        driver: VecDriver,
        sampler: OpponentSampler,
        opponents: Mapping[int, SwappableOpponent],
        group_of: Sequence[int],
        plans: Sequence[InstancePlan],
        rollout_steps: int,
        load_snapshot: Callable[[OpponentSpec], Mapping[str, Any]],
        agent_id: Callable[[], str] = lambda: "agent",
        on_step: Optional[Callable[[Any, Any], None]] = None,
    ):
        self.driver = driver
        self.sampler = sampler
        self.opponents = dict(opponents)
        self.group_of = list(group_of)
        self.plans = list(plans)
        self.rollout_steps = int(rollout_steps)
        self.load_snapshot = load_snapshot
        self.agent_id = agent_id
        self.on_step = on_step
        self.episodes: Dict[int, _EnvEpisode] = {}
        self.started = False
        self.mixture_counts: Dict[str, int] = {}

    # -- opponent assignment ----------------------------------------------

    def _sample_for(self, index: int, live_payload: Mapping[str, Any]) -> OpponentSpec:
        plan = self.plans[index]
        if plan.opponent_is_scripted:
            # Fixed for the life of the process; see the module docstring.
            spec = OpponentSpec(category="anchor", id=plan.anchor.id, anchor=plan.anchor)
        else:
            spec = self.sampler.sample()
            group = self.group_of[index]
            opp = self.opponents.get(group)
            if opp is None:
                raise KeyError(
                    f"instance {index} maps to opponent group {group}, which has no slot"
                )
            opp.set_opponent(spec, live_payload, self.load_snapshot)
        self.mixture_counts[spec.category] = self.mixture_counts.get(spec.category, 0) + 1
        return spec

    def _begin_episode(self, index: int, live_payload: Mapping[str, Any]) -> None:
        obs = self.driver.env.last_obs[index]
        self.episodes[index] = _EnvEpisode(
            opponent=self._sample_for(index, live_payload),
            group=self.group_of[index],
            first_t=int(obs["t"]) if obs else 0,
        )

    def _finish_episode(self, index: int, reason: str) -> Optional[EpisodeResult]:
        ep = self.episodes.get(index)
        obs = self.driver.env.last_obs[index]
        if ep is None or obs is None:
            return None
        cs = self.driver.cs_at_10(index)
        score = lane_outcome(obs, cs)
        return EpisodeResult(
            agent=self.agent_id(),
            opponent_id=ep.opponent.id,
            opponent_category=ep.opponent.category,
            score=score,
            # Absent, not zero: an episode that ended before ten minutes has no
            # CS@10 and must not be averaged in as a bad one.
            cs_at_10=float(cs[BLUE_TEAM]) if BLUE_TEAM in cs else None,
            length_steps=ep.steps,
            reason=reason,
            instance=index,
        )

    # -- the rollout -------------------------------------------------------

    def __call__(self, actor_id: int, live_payload: Mapping[str, Any], version: int) -> Rollout:
        if not self.started:
            self.driver.start()
            self.started = True
            for i in range(self.driver.env.n):
                self._begin_episode(i, live_payload)

        finished: List[EpisodeResult] = []
        self.mixture_counts = {}
        for _ in range(self.rollout_steps):
            result, dones = self.driver.step()
            for i in range(self.driver.env.n):
                if i in self.episodes and result.obs[i] is not None:
                    self.episodes[i].steps += 1
            if self.on_step is not None:
                self.on_step(result, dones)
            for i in result.restarted:
                # A restarted instance lost its episode; the trajectory is
                # already discarded upstream, and scoring a crashed lane would
                # feed noise straight into the rating system.
                log.error(
                    "instance %d restarted mid-episode against %s; its episode is dropped",
                    i,
                    self.episodes[i].opponent.id if i in self.episodes else "?",
                )
                self._begin_episode(i, live_payload)
            for i, reason in dones.items():
                if i in result.restarted:
                    continue
                done = self._finish_episode(i, reason)
                if done is not None:
                    finished.append(done)
                self._begin_episode(i, live_payload)

        total = sum(self.mixture_counts.values())
        mixture = (
            {k: v / total for k, v in self.mixture_counts.items()} if total else {}
        )
        return Rollout(
            actor_id=actor_id,
            param_version=version,
            steps=self.rollout_steps * self.driver.env.n,
            data=None,  # the trajectory buffer is the on_step callback's business
            episodes=finished,
            mixture=mixture,
        )


def build_assignments(
    plans: Sequence[InstancePlan], group_of: Sequence[int], agent_key: str = "agent"
) -> List[SideAssignment]:
    """Blue is always the learner; red is a group slot or the in-server bot.

    Blue-always is legitimate because ``lanerl_rl.obs`` mirrors red side into the
    same canonical frame, so a single policy covers both -- and it keeps CS@10
    attribution to one team id instead of two.
    """
    out: List[SideAssignment] = []
    for p in plans:
        if p.opponent_is_scripted:
            out.append(SideAssignment(blue=agent_key, red=None))
        else:
            out.append(SideAssignment(blue=agent_key, red=f"opp@{group_of[p.index]}"))
    return out


def group_assignment(n: int, groups: int) -> List[int]:
    """Round-robin instances over opponent groups.

    ``groups`` is the batching knob: fewer groups means larger opponent batches
    (0.058 ms/decision at batch 24 against 1.70 ms at batch 1) and more
    correlation between the opponents an actor faces within one episode block.
    """
    if groups <= 0:
        raise ValueError("groups must be positive")
    if groups > n:
        log.warning(
            "%d opponent groups for %d instances means batch-1 opponent forwards, which "
            "costs ~30x per decision; capping at %d",
            groups,
            n,
            n,
        )
        groups = n
    return [i % groups for i in range(n)]
