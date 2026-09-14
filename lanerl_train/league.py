"""Who the agent plays, and how the pool of past selves is kept.

The mixture, and why each slice exists
--------------------------------------
40% **latest self** -- true mirror self-play, the *same live weights* on both
sides.  This is the on-policy signal; it is also the slice whose win rate is
pinned at 50% forever, which is why :mod:`lanerl_train.eval` exists.

40% **PFSP** over the pool, weight proportional to ``(1 - p_i)^2`` where ``p_i``
is the running win rate against snapshot ``i``.  Squaring concentrates play on
the opponents currently being lost to, which is the whole mechanism by which
league play beats naive self-play.

15% **uniform** over the whole pool -- anti-forgetting.  PFSP alone abandons any
opponent already beaten, and that abandonment *is* the rot: AlphaStar's naive
self-play reached Elo 1519 while its minimum win rate against its own past fell
to 46%.  This slice is what keeps old opponents in the training distribution.

5% **frozen scripted anchors** -- the only opponents whose strength is fixed, so
the only ones a win rate can be read against as an absolute.

Deliberately *not* built: an AlphaStar-style league with separate main-exploiter
and league-exploiter agents.  Those need their own training processes, and the
CPU budget here is binding -- 16 instances on 16 cores, per-instance throughput
already degrading with instance count.  Splitting it three ways to chase a
mechanism designed for a full RTS with a much larger strategy space would cost
more than it could return in a 1v1 lane.
"""

from __future__ import annotations

import json
import logging
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from .eval import AnchorSpec

__all__ = [
    "LeagueConfigError",
    "Snapshot",
    "LeagueConfig",
    "CheckpointPool",
    "WinRateTracker",
    "OpponentSpec",
    "OpponentSampler",
    "LATEST",
]

log = logging.getLogger("lanerl_train.league")

#: The id used for "the live parameters", on both sides of a mirror game.
LATEST = "latest"


class LeagueConfigError(ValueError):
    """A league configuration that would silently misbehave."""


@dataclass(frozen=True)
class Snapshot:
    """A frozen copy of past parameters."""

    id: str
    step: int
    path: Optional[str] = None
    created_s: float = 0.0

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


@dataclass
class LeagueConfig:
    # OpenAI Five's mixture (Berner et al. 2019): 80% against the latest
    # policy, 20% against past versions sampled by a quality score (PFSP here).
    #
    # This was AlphaStar's main-agent mixture (35/50/15), which is the right
    # shape for StarCraft and the wrong one here. AlphaStar needs a heavy
    # league because StarCraft is strongly NON-TRANSITIVE -- rush beats eco,
    # eco beats tech, tech beats rush -- so a main agent must stay robust to
    # strategies its own history never produced. A 1v1 mirror lane is mostly
    # last-hitting, trading and wave state, which is largely transitive: being
    # better is just being better.
    #
    # It is also much cheaper. A "latest" draw is a true mirror, so BOTH sides'
    # transitions are on-policy and the rollout carries 24 slots instead of 12
    # (see procactor._apply_pending). At 40% latest that is 1.40x the data rate
    # of a pure league; at 80% it is 1.80x, against 2.00x for a pure mirror.
    p_latest: float = 0.80
    p_pfsp: float = 0.15
    #: Zero deliberately: OpenAI Five sampled past opponents by quality alone.
    #: PFSP already covers the pool, and the uniform slice mostly bought
    #: coverage that matters when the pool is large and diverse.
    p_uniform: float = 0.00
    p_anchor: float = 0.05
    #: Pool bounds.  Below ``pool_min`` the pool is not thinned at all; above
    #: ``pool_max`` the oldest middle entries go first (see :meth:`CheckpointPool.add`).
    pool_min: int = 10
    pool_max: int = 30
    #: How many of the most recent snapshots are never thinned.
    keep_recent: int = 5
    pfsp_exponent: float = 2.0
    #: Beta prior on each pairwise win rate: (1,1) starts everyone at 0.5, so a
    #: brand-new snapshot gets a middling PFSP weight rather than a wild one.
    prior_wins: float = 1.0
    prior_losses: float = 1.0
    #: Exponential forgetting on the pairwise win rate, in games.  The agent
    #: changes under the record, so a lifetime mean describes a policy that no
    #: longer exists.  ``None`` disables decay.
    halflife_games: Optional[float] = 200.0

    def __post_init__(self) -> None:
        ps = {
            "p_latest": self.p_latest,
            "p_pfsp": self.p_pfsp,
            "p_uniform": self.p_uniform,
            "p_anchor": self.p_anchor,
        }
        for name, v in ps.items():
            if v < 0.0:
                raise LeagueConfigError(f"{name} must be non-negative, got {v}")
        total = sum(ps.values())
        if abs(total - 1.0) > 1e-9:
            raise LeagueConfigError(
                f"opponent mixture must sum to 1.0, got {total:.6f} from {ps}. "
                f"A mixture that does not sum to one silently reweights every slice."
            )
        if self.pool_min < 1 or self.pool_max < self.pool_min:
            raise LeagueConfigError(
                f"need 1 <= pool_min <= pool_max, got {self.pool_min}..{self.pool_max}"
            )
        if self.keep_recent < 1 or self.keep_recent > self.pool_max:
            raise LeagueConfigError(
                f"keep_recent must be in 1..pool_max, got {self.keep_recent}"
            )
        if self.pfsp_exponent <= 0:
            raise LeagueConfigError("pfsp_exponent must be positive")
        if self.prior_wins <= 0 or self.prior_losses <= 0:
            raise LeagueConfigError(
                "the Beta prior must be positive on both sides; a zero prior makes the "
                "first game against a new snapshot decide its PFSP weight outright"
            )
        if self.halflife_games is not None and self.halflife_games <= 0:
            raise LeagueConfigError("halflife_games must be positive or None")

    def weights(self) -> Dict[str, float]:
        return {
            "latest": self.p_latest,
            "pfsp": self.p_pfsp,
            "uniform": self.p_uniform,
            "anchor": self.p_anchor,
        }


class CheckpointPool:
    """A bounded pool of past snapshots.

    Eviction keeps three things, in priority order: the ``keep_recent`` newest
    (PFSP needs current opposition), the *oldest* entry (the origin is the
    longest lever on the rot signal -- if the agent can no longer beat where it
    started, nothing else matters), and an even spread of steps in between.
    Dropping the oldest first would make min-win-rate-vs-past a measurement over
    a window that slides along with the agent, which is no measurement at all.
    """

    def __init__(self, config: Optional[LeagueConfig] = None):
        self.config = config or LeagueConfig()
        self._snapshots: List[Snapshot] = []
        self.evicted: List[Snapshot] = []

    def __len__(self) -> int:
        return len(self._snapshots)

    @property
    def snapshots(self) -> List[Snapshot]:
        return list(self._snapshots)

    def ids(self) -> List[str]:
        return [s.id for s in self._snapshots]

    def get(self, snapshot_id: str) -> Optional[Snapshot]:
        for s in self._snapshots:
            if s.id == snapshot_id:
                return s
        return None

    def add(self, snapshot: Snapshot) -> List[Snapshot]:
        """Add a snapshot, returning whatever had to be evicted."""
        if any(s.id == snapshot.id for s in self._snapshots):
            raise LeagueConfigError(
                f"snapshot id {snapshot.id!r} is already in the pool; ids must be unique "
                f"or the win-rate table conflates two different sets of parameters"
            )
        self._snapshots.append(snapshot)
        self._snapshots.sort(key=lambda s: (s.step, s.id))
        return self._thin()

    def _thin(self) -> List[Snapshot]:
        cfg = self.config
        if len(self._snapshots) <= cfg.pool_max:
            return []
        keep_idx = set()
        n = len(self._snapshots)
        keep_idx.add(0)  # the origin
        for k in range(1, cfg.keep_recent + 1):  # the newest
            keep_idx.add(n - k)
        budget = cfg.pool_max - len(keep_idx)
        middle = [i for i in range(n) if i not in keep_idx]
        if budget > 0 and middle:
            # Even spread over the surviving middle, by position in step order.
            take = min(budget, len(middle))
            for j in range(take):
                keep_idx.add(middle[round(j * (len(middle) - 1) / max(take - 1, 1))])
        kept = [s for i, s in enumerate(self._snapshots) if i in keep_idx]
        # Rounding collisions can leave us under budget; that is fine (we never
        # exceed pool_max), but never let it drop below what we promised to keep.
        dropped = [s for i, s in enumerate(self._snapshots) if i not in keep_idx]
        self._snapshots = kept
        self.evicted.extend(dropped)
        if dropped:
            log.info(
                "pool thinned to %d (max %d); evicted %s",
                len(kept),
                cfg.pool_max,
                [s.id for s in dropped],
            )
        return dropped

    def to_dict(self) -> Dict[str, object]:
        return {
            "snapshots": [s.to_dict() for s in self._snapshots],
            "evicted": [s.to_dict() for s in self.evicted],
        }

    @classmethod
    def from_dict(
        cls, data: Mapping[str, object], config: Optional[LeagueConfig] = None
    ) -> "CheckpointPool":
        pool = cls(config)
        raw = data.get("snapshots", [])
        pool._snapshots = [Snapshot(**s) for s in raw]  # type: ignore[arg-type]
        pool.evicted = [Snapshot(**s) for s in data.get("evicted", [])]  # type: ignore[arg-type]
        pool._snapshots.sort(key=lambda s: (s.step, s.id))
        return pool


class WinRateTracker:
    """Running win rate of the *current* agent against each opponent id.

    Beta posterior mean with exponential forgetting, so a snapshot beaten
    decisively 300 games ago does not keep a PFSP weight of zero after the agent
    has drifted away from whatever beat it.
    """

    def __init__(self, config: Optional[LeagueConfig] = None):
        self.config = config or LeagueConfig()
        self._w: Dict[str, float] = {}
        self._l: Dict[str, float] = {}
        self._n: Dict[str, int] = {}

    def _decay(self) -> float:
        h = self.config.halflife_games
        return 1.0 if h is None else 0.5 ** (1.0 / h)

    def record(self, opponent_id: str, score: float) -> None:
        """``score`` is the current agent's result: 1.0 win, 0.5 draw, 0.0 loss."""
        if not (0.0 <= score <= 1.0):
            raise ValueError(f"score must be in [0,1], got {score}")
        d = self._decay()
        self._w[opponent_id] = self._w.get(opponent_id, 0.0) * d + score
        self._l[opponent_id] = self._l.get(opponent_id, 0.0) * d + (1.0 - score)
        self._n[opponent_id] = self._n.get(opponent_id, 0) + 1

    def p(self, opponent_id: str) -> float:
        """Posterior mean win rate.  0.5 for an opponent never played."""
        a = self._w.get(opponent_id, 0.0) + self.config.prior_wins
        b = self._l.get(opponent_id, 0.0) + self.config.prior_losses
        return a / (a + b)

    def n(self, opponent_id: str) -> int:
        return self._n.get(opponent_id, 0)

    def to_dict(self) -> Dict[str, object]:
        return {"w": dict(self._w), "l": dict(self._l), "n": dict(self._n)}

    @classmethod
    def from_dict(
        cls, data: Mapping[str, object], config: Optional[LeagueConfig] = None
    ) -> "WinRateTracker":
        t = cls(config)
        t._w = dict(data.get("w", {}))  # type: ignore[arg-type]
        t._l = dict(data.get("l", {}))  # type: ignore[arg-type]
        t._n = {k: int(v) for k, v in dict(data.get("n", {})).items()}  # type: ignore[arg-type]
        return t


@dataclass(frozen=True)
class OpponentSpec:
    """One sampled opponent.

    ``category`` is the mixture slice it came from, kept so the realised mixture
    can be checked against the target in the metrics log rather than assumed.
    """

    category: str  # "latest" | "pfsp" | "uniform" | "anchor"
    id: str
    snapshot: Optional[Snapshot] = None
    anchor: Optional[AnchorSpec] = None

    @property
    def is_latest(self) -> bool:
        """True when both sides share the live weights (true mirror self-play)."""
        return self.category == "latest"


class OpponentSampler:
    """Draws opponents according to the league mixture.

    Empty slices are dropped and the remainder renormalised, which is the honest
    behaviour at step 0 when the pool is empty -- but it is logged once per
    category, because a permanently empty anchor slice (a missing config file,
    say) looks exactly like everything working.
    """

    def __init__(
        self,
        pool: CheckpointPool,
        win_rates: WinRateTracker,
        anchors: Sequence[AnchorSpec] = (),
        config: Optional[LeagueConfig] = None,
        rng: Optional[random.Random] = None,
    ):
        self.pool = pool
        self.win_rates = win_rates
        self.anchors = [a for a in anchors]
        self.config = config or pool.config
        self.rng = rng or random.Random()
        self.counts: Dict[str, int] = {k: 0 for k in self.config.weights()}
        self._warned: set = set()

    # -- PFSP --------------------------------------------------------------

    def pfsp_weights(self, snapshots: Optional[Sequence[Snapshot]] = None) -> Dict[str, float]:
        """``(1 - p_i)^exponent``, normalised.

        If the agent beats every snapshot outright the weights all collapse to
        zero; that falls back to uniform rather than dividing by zero, and says
        so, because "I win everything" is also the shape of "the win rates are
        not being recorded".
        """
        snaps = list(snapshots if snapshots is not None else self.pool.snapshots)
        if not snaps:
            return {}
        raw = {
            s.id: max(0.0, 1.0 - self.win_rates.p(s.id)) ** self.config.pfsp_exponent
            for s in snaps
        }
        total = sum(raw.values())
        if total <= 0.0:
            self._warn_once(
                "pfsp_degenerate",
                "every PFSP weight is zero (win rate 1.0 against every snapshot); "
                "falling back to uniform. Check that match results are being recorded.",
            )
            return {s.id: 1.0 / len(snaps) for s in snaps}
        return {k: v / total for k, v in raw.items()}

    # -- sampling ----------------------------------------------------------

    def _available(self) -> Dict[str, float]:
        cfg = self.config
        w = cfg.weights()
        avail: Dict[str, float] = {"latest": w["latest"]}
        if self.pool.snapshots:
            avail["pfsp"] = w["pfsp"]
            avail["uniform"] = w["uniform"]
        else:
            self._warn_once(
                "empty_pool",
                "the checkpoint pool is empty; PFSP and uniform slices are unavailable "
                "and their probability mass goes to latest/anchors. Expected only at the "
                "start of a run.",
            )
        usable = [a for a in self.anchors if a.kind != "scripted" or a.exists()]
        if usable:
            avail["anchor"] = w["anchor"]
        elif self.anchors:
            self._warn_once(
                "no_usable_anchors",
                "every configured anchor is unusable (missing resource); the frozen rung "
                "of the ladder is gone and win rate vs a fixed opponent cannot be measured.",
            )
        avail = {k: v for k, v in avail.items() if v > 0.0}
        if not avail:
            raise LeagueConfigError(
                "no opponent category is available: the pool is empty, there are no usable "
                "anchors, and p_latest is zero. There is nothing to play against."
            )
        return avail

    def sample(self) -> OpponentSpec:
        avail = self._available()
        category = self._choose(avail)
        self.counts[category] = self.counts.get(category, 0) + 1
        if category == "latest":
            return OpponentSpec(category="latest", id=LATEST)
        if category == "anchor":
            usable = [a for a in self.anchors if a.kind != "scripted" or a.exists()]
            a = usable[self.rng.randrange(len(usable))]
            return OpponentSpec(category="anchor", id=a.id, anchor=a)
        snaps = self.pool.snapshots
        if category == "uniform":
            s = snaps[self.rng.randrange(len(snaps))]
            return OpponentSpec(category="uniform", id=s.id, snapshot=s)
        weights = self.pfsp_weights(snaps)
        s = self._choose_snapshot(snaps, weights)
        return OpponentSpec(category="pfsp", id=s.id, snapshot=s)

    def sample_many(self, n: int) -> List[OpponentSpec]:
        return [self.sample() for _ in range(n)]

    def _choose(self, weights: Mapping[str, float]) -> str:
        total = sum(weights.values())
        r = self.rng.random() * total
        acc = 0.0
        for k, v in weights.items():
            acc += v
            if r < acc:
                return k
        return next(reversed(list(weights)))  # float dust at the top of the range

    def _choose_snapshot(
        self, snaps: Sequence[Snapshot], weights: Mapping[str, float]
    ) -> Snapshot:
        total = sum(weights.get(s.id, 0.0) for s in snaps)
        if total <= 0.0:  # pragma: no cover - pfsp_weights already guards this
            return snaps[self.rng.randrange(len(snaps))]
        r = self.rng.random() * total
        acc = 0.0
        for s in snaps:
            acc += weights.get(s.id, 0.0)
            if r < acc:
                return s
        return snaps[-1]

    # -- bookkeeping -------------------------------------------------------

    def realised_mixture(self) -> Dict[str, float]:
        total = sum(self.counts.values())
        if total == 0:
            return {k: 0.0 for k in self.counts}
        return {k: v / total for k, v in self.counts.items()}

    def mixture_drift(self) -> Dict[str, float]:
        """Realised minus target, per category.  Log it; do not assume it is zero."""
        target = self.config.weights()
        real = self.realised_mixture()
        return {k: real.get(k, 0.0) - target.get(k, 0.0) for k in target}

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        log.warning("%s", message)

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> Dict[str, object]:
        return {
            "pool": self.pool.to_dict(),
            "win_rates": self.win_rates.to_dict(),
            "counts": dict(self.counts),
        }

    def load_dict(self, data: Mapping[str, object]) -> None:
        self.pool = CheckpointPool.from_dict(data["pool"], self.config)  # type: ignore[arg-type]
        rates = data["win_rates"]
        self.win_rates = WinRateTracker.from_dict(rates, self.config)  # type: ignore[arg-type]
        counts = dict(data.get("counts", {}))  # type: ignore[arg-type]
        self.counts = {k: int(v) for k, v in counts.items()}

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))


def expected_pfsp_share(
    win_rates: Mapping[str, float], exponent: float = 2.0
) -> Dict[str, float]:
    """Reference implementation of the PFSP distribution, for tests and audits.

    Kept deliberately separate from :meth:`OpponentSampler.pfsp_weights` so a
    test compares two independent expressions of ``(1-p)^k`` rather than the
    sampler against itself.
    """
    raw = {k: max(0.0, 1.0 - p) ** exponent for k, p in win_rates.items()}
    total = math.fsum(raw.values())
    if total <= 0.0:
        n = len(raw)
        return {k: 1.0 / n for k in raw} if n else {}
    return {k: v / total for k, v in raw.items()}
