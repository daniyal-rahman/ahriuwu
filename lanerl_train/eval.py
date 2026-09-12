"""Evaluation for a symmetric mirror match, where the obvious metric is useless.

In a 1v1 Garen mirror, win rate against your own latest parameters is 50% **by
construction**, at initialisation and at convergence alike, whether the policy is
improving or rotting.  It is not a weak signal, it is a constant.  Everything
here exists because of that:

``CS@10``
    Absolute and non-zero-sum: it cannot be held flat by both sides getting
    worse together.  The scripted baseline sits at 31.9 with zero deaths
    (curriculum: diamond 35.2 / gold 29.5 / bronze 16.7), so the number has a
    known scale from day one.

win rate vs the frozen scripted bot
    Frozen is the operative word.  Three difficulties give a coarse ladder that
    does not move under the agent's feet.

Bradley-Terry / BayesElo
    A single ordering over every agent that has ever played, fitted from all
    recorded pairwise outcomes.

minimum win rate against your own past checkpoints
    AlphaStar's rot signature.  Naive self-play there reached Elo 1519 while its
    minimum win rate against its own past fell to 46% -- it had learned to beat
    the current opponent by forgetting how to beat the old one.  A rising Elo
    with a falling minimum is the exact shape of that failure, and only this
    pair of numbers together shows it.

Four anchors are frozen permanently (scripted bronze/gold/diamond and the BC
policy) and get ~5% of episodes, so the ladder always has a fixed rung.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import paths

__all__ = [
    "ELO_SCALE",
    "MatchRecord",
    "PairTable",
    "bradley_terry",
    "elo_from_ratings",
    "expected_score",
    "AnchorSpec",
    "AnchorConfigError",
    "default_anchors",
    "DEFAULT_RUN_ANCHORS",
    "anchors_for_run",
    "validate_anchors",
    "ANCHOR_EPISODE_SHARE",
    "anchor_episode_budget",
    "CsTracker",
    "EvalReport",
    "Evaluator",
]

log = logging.getLogger("lanerl_train.eval")

#: Elo points per natural-log unit of Bradley-Terry strength (400 / ln 10).
ELO_SCALE = 400.0 / math.log(10.0)

#: Fraction of episodes reserved for the permanently frozen anchors.
ANCHOR_EPISODE_SHARE = 0.05


# --------------------------------------------------------------------------
# Match records
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MatchRecord:
    """One completed episode, from ``agent_a``'s point of view.

    ``score_a`` is 1.0 / 0.5 / 0.0.  Draws are first-class: a ten-minute lane
    that ends with neither champion dead is the *common* case here, and folding
    it into a loss would bias every rating.
    """

    agent_a: str
    agent_b: str
    score_a: float
    step: int = 0
    meta: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.score_a <= 1.0):
            raise ValueError(f"score_a must be in [0,1], got {self.score_a}")
        if self.agent_a == self.agent_b:
            raise ValueError(
                f"self-match against {self.agent_a!r} carries no information and would "
                f"pull every rating toward the mean; record it as an episode, not a match"
            )

    def to_json(self) -> str:
        return json.dumps({"kind": "match", **asdict(self)}, separators=(",", ":"))


class PairTable:
    """Accumulated pairwise results, symmetric by construction."""

    def __init__(self) -> None:
        self._score: Dict[Tuple[str, str], float] = defaultdict(float)
        self._games: Dict[Tuple[str, str], int] = defaultdict(int)
        self.players: List[str] = []
        self._seen: set = set()

    def _touch(self, name: str) -> None:
        if name not in self._seen:
            self._seen.add(name)
            self.players.append(name)

    def add(self, rec: MatchRecord) -> None:
        self._touch(rec.agent_a)
        self._touch(rec.agent_b)
        self._score[(rec.agent_a, rec.agent_b)] += rec.score_a
        self._score[(rec.agent_b, rec.agent_a)] += 1.0 - rec.score_a
        self._games[(rec.agent_a, rec.agent_b)] += 1
        self._games[(rec.agent_b, rec.agent_a)] += 1

    def extend(self, records: Iterable[MatchRecord]) -> None:
        for r in records:
            self.add(r)

    def games(self, a: str, b: str) -> int:
        return self._games.get((a, b), 0)

    def score(self, a: str, b: str) -> float:
        return self._score.get((a, b), 0.0)

    def win_rate(self, a: str, b: str) -> Optional[float]:
        n = self.games(a, b)
        return None if n == 0 else self.score(a, b) / n

    def total_games(self, a: str) -> int:
        return sum(n for (x, _), n in self._games.items() if x == a)

    def opponents(self, a: str) -> List[str]:
        return sorted({y for (x, y), n in self._games.items() if x == a and n > 0})


# --------------------------------------------------------------------------
# Bradley-Terry / Elo
# --------------------------------------------------------------------------


def bradley_terry(
    table: PairTable,
    prior_games: float = 2.0,
    max_iter: int = 1000,
    tol: float = 1e-10,
    anchor: Optional[str] = None,
    anchor_elo: float = 0.0,
) -> Dict[str, float]:
    """Fit Bradley-Terry strengths by MM (Zermelo) iteration, in Elo points.

    ``prior_games`` adds that many drawn games against a virtual opponent of
    fixed strength.  Without it an undefeated player has infinite rating and the
    iteration never converges -- and an undefeated player is guaranteed here,
    because a fresh checkpoint's first few games against a bronze anchor are
    routinely a clean sweep.  This is the "Bayes" in BayesElo: a weak prior that
    keeps early ratings finite and honest.

    By default ratings are centred on zero.  Pass ``anchor`` to pin one player
    (the frozen gold bot, say) so the scale does not drift as the pool grows --
    a moving zero makes an Elo time series unreadable.
    """
    players = list(table.players)
    if not players:
        return {}
    if prior_games <= 0:
        raise ValueError(
            "prior_games must be > 0; without it an undefeated player has unbounded "
            "rating and the fit does not converge"
        )
    gamma = {p: 1.0 for p in players}
    wins = {
        p: sum(table.score(p, q) for q in players if q != p) + 0.5 * prior_games
        for p in players
    }
    for it in range(max_iter):
        max_rel = 0.0
        for p in players:
            denom = prior_games / (gamma[p] + 1.0)
            for q in players:
                if q == p:
                    continue
                n = table.games(p, q)
                if n:
                    denom += n / (gamma[p] + gamma[q])
            if denom <= 0.0:  # pragma: no cover - prior_games > 0 prevents this
                continue
            new = wins[p] / denom
            max_rel = max(max_rel, abs(new - gamma[p]) / max(gamma[p], 1e-12))
            gamma[p] = new
        if max_rel < tol:
            break
    else:
        log.warning(
            "bradley_terry did not converge in %d iterations (last relative change %.2e); "
            "ratings are usable but treat small differences with suspicion",
            max_iter,
            max_rel,
        )
    ratings = {p: ELO_SCALE * math.log(max(g, 1e-300)) for p, g in gamma.items()}
    if anchor is not None:
        if anchor not in ratings:
            raise KeyError(f"anchor {anchor!r} has no recorded matches; cannot pin the scale")
        shift = anchor_elo - ratings[anchor]
    else:
        shift = -sum(ratings.values()) / len(ratings)
    return {p: r + shift for p, r in ratings.items()}


def elo_from_ratings(ratings: Mapping[str, float]) -> List[Tuple[str, float]]:
    """Ratings sorted strongest first."""
    return sorted(ratings.items(), key=lambda kv: -kv[1])


def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


# --------------------------------------------------------------------------
# Frozen anchors
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorSpec:
    """A permanently frozen opponent.

    Frozen means never retrained, never evicted from the pool, and never
    resampled from a moving distribution: it is the only fixed rung on the
    ladder, so the day it starts moving every historical Elo becomes a lie.
    """

    id: str
    kind: str  # "scripted" | "policy"
    #: For "scripted": the ``LANERL_BOT_CONFIG`` JSON.  For "policy": a checkpoint.
    resource: Optional[Path] = None
    #: Published CS@10 for the scripted curriculum; None where not measured.
    reference_cs_at_10: Optional[float] = None
    #: Which side the anchor drives, as ``LANERL_BOT`` spells it.
    bot_teams: str = "purple"

    def exists(self) -> bool:
        return self.resource is not None and Path(self.resource).exists()


def default_anchors(bc_checkpoint: Optional[Path] = None) -> List[AnchorSpec]:
    """The four permanent anchors: three scripted difficulties plus BC.

    Reference CS@10 values are the measured ones for this bot, so a regression
    in the *anchor* (a content change, a server patch) is visible as the anchor
    drifting off its own published number rather than as unexplained agent
    weirdness.
    """
    cfg = paths.bot_config_dir()
    anchors = [
        # The three reference CS numbers below are STALE and no longer used by the
        # eval line, which now reports the anchor's CS measured in the same
        # game (EpisodeResult.opponent_cs_at_10). They were measured on a bot
        # config with no rune or mastery page -- 57.88 AD against the 78.14 the
        # agent actually faces -- so "diamond" 35.2 sits BELOW what the bronze
        # bot really farms (~48 seeded). Kept only as a historical marker;
        # do not cite them.
        AnchorSpec("scripted_bronze", "scripted", cfg / "anchor_bronze.json", 16.7),
        AnchorSpec("scripted_gold", "scripted", cfg / "anchor_gold.json", 29.5),
        AnchorSpec("scripted_diamond", "scripted", cfg / "anchor_diamond.json", 35.2),
    ]
    anchors.append(AnchorSpec("bc_policy", "policy", bc_checkpoint, None))
    for a in anchors:
        if a.resource is None:
            log.warning(
                "anchor %s has no resource configured; it will be skipped when sampling. "
                "Pass bc_checkpoint= to default_anchors() once the BC policy is exported.",
                a.id,
            )
        elif not a.exists():
            log.error(
                "anchor %s points at %s, which does not exist. A missing anchor silently "
                "removes a rung from the ladder -- fix the path or drop the anchor "
                "deliberately.",
                a.id,
                a.resource,
            )
    return anchors


class AnchorConfigError(RuntimeError):
    """An anchor set that cannot actually be played.

    Fatal at startup, deliberately.  ``default_anchors()`` logs and continues
    because it is a *description* of the four permanent rungs, and a data
    structure should not raise -- but a RUN that carries an unplayable anchor
    produces an eval section reading ``(None, 0)`` forever, and 37 of those went
    by unread in the first real run while the policy learned nothing.  A warning
    was not enough; it has already been proved not to be.
    """


#: The anchors a run uses unless told otherwise.  ``bc_policy`` is deliberately
#: NOT here: it has no resource until a BC checkpoint exists, and including an
#: anchor that cannot be played is the exact failure this module now refuses.
DEFAULT_RUN_ANCHORS: Tuple[str, ...] = (
    "scripted_bronze",
    "scripted_gold",
    "scripted_diamond",
)


def validate_anchors(anchors: Sequence[AnchorSpec]) -> None:
    """Raise unless every anchor can actually be played."""
    if not anchors:
        raise AnchorConfigError(
            "the anchor list is empty. In a symmetric mirror every score is 0.5 by "
            "construction, so with no frozen opponent there is nothing for evaluation "
            "to measure. Configure at least one anchor, or turn evaluation off "
            "deliberately."
        )
    unconfigured = [a.id for a in anchors if a.resource is None]
    absent = [(a.id, str(a.resource)) for a in anchors if a.resource is not None and not a.exists()]
    if unconfigured or absent:
        parts = []
        if unconfigured:
            parts.append(f"no resource configured: {unconfigured}")
        if absent:
            parts.append("resource does not exist: " + ", ".join(f"{i} -> {p}" for i, p in absent))
        raise AnchorConfigError(
            "; ".join(parts)
            + ". A missing anchor removes a rung from the ladder silently -- fix the "
            "path, export the checkpoint, or drop the anchor from --anchors."
        )

    # Reject here what anchor_launch_spec would reject LATER. Only a 'scripted'
    # anchor can be played by the in-server bot, and anchor_launch_spec raises
    # AnchorEvalError for anything else -- but that raise happens inside
    # _run_anchor_eval, which propagates through _periodic -> step_once and
    # KILLS THE RUN, hundreds of updates and an hour of GPU time after start.
    # `--bc-checkpoint` was the only way to reach it: it attaches a resource to
    # the bc_policy anchor, whose kind is "policy", so the checks above pass and
    # the run dies at the first eval. Fail at argument-parse time instead.
    unplayable = [(a.id, a.kind) for a in anchors
                  if a.kind not in ("scripted", "policy")]
    if unplayable:
        raise AnchorConfigError(
            "these anchors cannot be played by the in-server bot: "
            + ", ".join(f"{i} (kind={k!r})" for i, k in unplayable)
            + ". Only 'scripted' (in-server bot) and 'policy' (a frozen network "
            "on red) can be played; anchor_launch_spec would raise on anything "
            "else at the first evaluation and end the run. Drop them from --anchors."
        )


def anchors_for_run(
    names: Sequence[str] = DEFAULT_RUN_ANCHORS,
    bc_checkpoint: Optional[Path] = None,
) -> List[AnchorSpec]:
    """The validated anchor list for one run.  Raises rather than warning."""
    catalogue = {a.id: a for a in default_anchors(bc_checkpoint)}
    unknown = [n for n in names if n not in catalogue]
    if unknown:
        raise AnchorConfigError(
            f"unknown anchor(s) {unknown}; known anchors are {sorted(catalogue)}"
        )
    chosen = [catalogue[n] for n in names]
    validate_anchors(chosen)
    return chosen


def anchor_episode_budget(total_episodes: int, share: float = ANCHOR_EPISODE_SHARE) -> int:
    """How many of ``total_episodes`` to spend on frozen anchors (>=1 if any)."""
    if total_episodes <= 0:
        return 0
    return max(1, int(round(share * total_episodes)))


# --------------------------------------------------------------------------
# CS@10
# --------------------------------------------------------------------------


class CsTracker:
    """Rolling CS@10 per agent.

    A window, not a lifetime mean: the point of this number is to answer "is the
    policy better than it was an hour ago", and a lifetime mean takes longer to
    move than the training run takes to rot.
    """

    def __init__(self, window: int = 200):
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = int(window)
        self._samples: Dict[str, List[float]] = defaultdict(list)

    def add(self, agent: str, cs: float) -> None:
        if cs < 0:
            raise ValueError(f"CS cannot be negative, got {cs} for {agent!r}")
        buf = self._samples[agent]
        buf.append(float(cs))
        if len(buf) > self.window:
            del buf[: len(buf) - self.window]

    def stats(self, agent: str) -> Optional[Tuple[float, float, int]]:
        """``(mean, stdev, n)`` or ``None`` when nothing has been recorded."""
        buf = self._samples.get(agent)
        if not buf:
            return None
        n = len(buf)
        mean = sum(buf) / n
        var = sum((x - mean) ** 2 for x in buf) / max(n - 1, 1)
        return mean, math.sqrt(var), n

    def agents(self) -> List[str]:
        return sorted(self._samples)


# --------------------------------------------------------------------------
# The report
# --------------------------------------------------------------------------


@dataclass
class EvalReport:
    step: int
    latest: str
    cs_at_10: Optional[Tuple[float, float, int]]
    win_rate_vs_anchor: Dict[str, Tuple[Optional[float], int]]
    elo: Dict[str, float]
    min_win_rate_vs_past: Optional[Tuple[str, float, int]]
    past_considered: int
    past_skipped_low_n: int
    rot_warning: bool
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps({"kind": "eval", **asdict(self)}, separators=(",", ":"), default=str)

    def summary(self) -> str:
        cs = "n/a" if self.cs_at_10 is None else f"{self.cs_at_10[0]:.1f}+-{self.cs_at_10[1]:.1f}"
        mw = (
            "n/a"
            if self.min_win_rate_vs_past is None
            else f"{self.min_win_rate_vs_past[1]:.3f} vs {self.min_win_rate_vs_past[0]}"
        )
        anchors = " ".join(
            f"{k}={'n/a' if v[0] is None else format(v[0], '.2f')}({v[1]})"
            for k, v in sorted(self.win_rate_vs_anchor.items())
        )
        return (
            f"step={self.step} elo={self.elo.get(self.latest, float('nan')):.0f} "
            f"cs@10={cs} min_wr_past={mw} {anchors}"
            + ("  ROT WARNING" if self.rot_warning else "")
        )


class Evaluator:
    """Accumulates matches and CS, and produces the report on demand."""

    def __init__(
        self,
        anchors: Optional[Sequence[AnchorSpec]] = None,
        cs_window: int = 200,
        min_games_for_min_winrate: int = 10,
        rot_threshold: float = 0.50,
        elo_anchor: Optional[str] = "scripted_gold",
    ):
        self.table = PairTable()
        self.cs = CsTracker(cs_window)
        self.anchors = list(anchors) if anchors is not None else default_anchors()
        self.anchor_ids = [a.id for a in self.anchors]
        self.min_games_for_min_winrate = int(min_games_for_min_winrate)
        self.rot_threshold = float(rot_threshold)
        self.elo_anchor = elo_anchor
        self.records: List[MatchRecord] = []

    # -- ingestion ---------------------------------------------------------

    def record_match(self, rec: MatchRecord) -> None:
        self.table.add(rec)
        self.records.append(rec)

    def record_cs(self, agent: str, cs: float) -> None:
        self.cs.add(agent, cs)

    def load_jsonl(self, path: Path) -> int:
        """Replay a metrics log.  Returns how many matches were ingested.

        Unparseable lines raise: a metrics file that is quietly half-read gives
        a report that is quietly half-true.
        """
        n = 0
        with Path(path).open() as fh:
            for lineno, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except ValueError as exc:
                    raise ValueError(f"{path}:{lineno} is not JSON: {exc}") from exc
                if obj.get("kind") == "match":
                    self.record_match(
                        MatchRecord(
                            agent_a=obj["agent_a"],
                            agent_b=obj["agent_b"],
                            score_a=float(obj["score_a"]),
                            step=int(obj.get("step", 0)),
                            meta=obj.get("meta", {}),
                        )
                    )
                    n += 1
                elif obj.get("kind") == "episode" and obj.get("cs_at_10") is not None:
                    self.record_cs(obj["agent"], float(obj["cs_at_10"]))
        return n

    # -- the four numbers --------------------------------------------------

    def min_win_rate_vs_past(
        self, latest: str, past: Sequence[str]
    ) -> Tuple[Optional[Tuple[str, float, int]], int, int]:
        """The AlphaStar rot signature.

        Returns ``((opponent, win_rate, games) | None, considered, skipped)``.
        Opponents with too few games are *skipped and counted*, never treated as
        0% -- one unlucky first game against a fresh snapshot would otherwise
        raise a false alarm every time the pool grows.
        """
        worst: Optional[Tuple[str, float, int]] = None
        considered = 0
        skipped = 0
        for p in past:
            if p == latest:
                continue
            n = self.table.games(latest, p)
            if n < self.min_games_for_min_winrate:
                skipped += 1
                continue
            wr = self.table.score(latest, p) / n
            considered += 1
            if worst is None or wr < worst[1]:
                worst = (p, wr, n)
        return worst, considered, skipped

    def report(self, step: int, latest: str, past: Sequence[str]) -> EvalReport:
        notes: List[str] = []
        anchor_wr: Dict[str, Tuple[Optional[float], int]] = {}
        for aid in self.anchor_ids:
            anchor_wr[aid] = (self.table.win_rate(latest, aid), self.table.games(latest, aid))
            if anchor_wr[aid][1] == 0:
                notes.append(f"no games against anchor {aid} yet")

        elo: Dict[str, float] = {}
        if self.table.players:
            use_anchor = (
                self.elo_anchor if self.elo_anchor in self.table.players else None
            )
            if self.elo_anchor is not None and use_anchor is None:
                notes.append(
                    f"elo anchor {self.elo_anchor!r} has no matches yet; ratings are "
                    f"mean-centred and will shift as the pool grows"
                )
            elo = bradley_terry(self.table, anchor=use_anchor)

        worst, considered, skipped = self.min_win_rate_vs_past(latest, past)
        rot = worst is not None and worst[1] < self.rot_threshold
        if rot:
            log.error(
                "ROT WARNING at step %d: min win rate vs own past is %.3f against %s over "
                "%d games. Rising Elo with a falling minimum is exactly the naive-self-play "
                "failure (AlphaStar: Elo 1519, min win rate 46%%).",
                step,
                worst[1],
                worst[0],
                worst[2],
            )
        if considered == 0 and past:
            notes.append(
                f"min win rate vs past is undefined: all {skipped} past checkpoints have "
                f"fewer than {self.min_games_for_min_winrate} games"
            )
        return EvalReport(
            step=step,
            latest=latest,
            cs_at_10=self.cs.stats(latest),
            win_rate_vs_anchor=anchor_wr,
            elo=elo,
            min_win_rate_vs_past=worst,
            past_considered=considered,
            past_skipped_low_n=skipped,
            rot_warning=rot,
            notes=notes,
        )
