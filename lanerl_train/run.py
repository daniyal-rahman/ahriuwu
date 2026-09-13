"""Async actors, one central learner, bounded staleness, and a resumable run dir.

Staleness is the load-bearing constraint
----------------------------------------
Actors collect with the parameters they last pulled; the learner has moved on by
the time a rollout lands.  OpenAI Five measured significant slowdown beyond ~8
parameter versions of lag and ran near zero.  The target here is 0-1, and that
is enforced in three places rather than hoped for:

1. an actor pulls the newest parameters at the *start* of every rollout, so lag
   accumulates only during one rollout;
2. the rollout queue is small, so actors block on backpressure instead of racing
   ahead producing data the learner will refuse;
3. the learner *rejects* a rollout older than the bound, counts the rejection,
   and shouts when the rejection rate is material.

Point 3 is the only one that is a guarantee.  Points 1 and 2 make it rarely fire.

The relation that actually sets the lag is
``max observed staleness ~= queue_capacity + num_actors - 1``: the learner can
consume that many rollouts, publishing once each, between an actor's pull and
its push.  :class:`TrainingLoop` checks this against ``max_staleness`` at
construction and refuses a configuration that is guaranteed to throw its own
data away.

Everything the run needs to be resumed lives in the run directory:
``state.json`` (counters, RNG, league), ``checkpoints/`` (weights + optimiser),
``metrics.jsonl`` (one JSON object per line, flushed on write, carrying exactly
what :mod:`lanerl_train.eval` replays).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import random
import threading
import time
import traceback
from collections import deque
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterator, List, Mapping, Optional, Tuple

from . import paths
from .eval import Evaluator, MatchRecord
from .league import (
    LATEST,
    CheckpointPool,
    LeagueConfig,
    OpponentSampler,
    Snapshot,
    WinRateTracker,
)

__all__ = [
    "TrainingError",
    "ActorFailure",
    "StalledRun",
    "EpisodeResult",
    "Rollout",
    "MetricsLog",
    "ParameterStore",
    "StalenessTracker",
    "ThroughputMeter",
    "GpuProbe",
    "TrainStepCounter",
    "CheckpointManager",
    "RunConfig",
    "RunState",
    "ActorLoop",
    "TrainingLoop",
]

log = logging.getLogger("lanerl_train.run")


class TrainingError(RuntimeError):
    """Anything that must stop the run."""


class ActorFailure(TrainingError):
    """An actor thread raised.  Re-raised on the learner thread with its traceback."""


class StalledRun(TrainingError):
    """No update completed within the watchdog window.

    Deliberately about *progress*, not liveness.  "Is the process up" was true
    throughout a two-and-a-half-day crash loop on this project; a step counter
    that stops moving is the only honest liveness signal.
    """


# --------------------------------------------------------------------------
# Data carried between actors and learner
# --------------------------------------------------------------------------


@dataclass
class EpisodeResult:
    """One finished episode, in the form both the league and eval need."""

    agent: str
    opponent_id: str
    opponent_category: str
    #: The agent's result: 1.0 win, 0.5 draw, 0.0 loss.
    score: float
    #: Absolute CS at ten minutes, or ``None`` when the episode ended early.
    #: Never substitute 0 for unknown -- see ``serverlog.cs_at``.
    cs_at_10: Optional[float] = None
    length_steps: int = 0
    reason: str = ""
    instance: int = -1
    #: Undiscounted sum of shaped reward over the episode. Absent for the whole
    #: first run, which is why 16,200 updates produced no way to tell whether
    #: the agent was receiving any signal at all.
    ep_return: Optional[float] = None
    #: The OPPONENT's CS at ten minutes, from the same game.
    #:
    #: Replaces the hardcoded ``AnchorSpec.reference_cs_at_10`` (16.7 / 29.5 /
    #: 35.2) that the eval line used to print beside the agent's number. Those
    #: three constants were measured on a bot config with no rune or mastery
    #: page -- 57.88 attack damage against the 78.14 the agent actually faces
    #: -- so by the time they were being quoted the "diamond" reference of
    #: 35.2 sat BELOW what the bronze bot really farms. Measuring the opponent
    #: in the same episode cannot go stale, and it is free: the server already
    #: reports CS for both teams.
    opponent_cs_at_10: Optional[float] = None
    #: Undiscounted per-episode sum of every RAW reward term, by name
    #: (``lanerl_rl.reward._AgentReward.terms`` plus ``shaping``).
    #:
    #: Computed on every tick since the reward was written, and thrown away on
    #: every tick since the reward was written. Nobody could ask "which term is
    #: the policy chasing?", which is how the respawn refund -- dying was
    #: NET POSITIVE, by +0.44 at the published weights -- survived weeks of
    #: runs whose every other metric looked ordinary. These are the raw,
    #: pre-zero-sum terms, so they do not add up to :attr:`ep_return`: the gap
    #: between the two IS the opponent's contribution, i.e. what ``alpha``
    #: scales.
    reward_terms: Optional[Mapping[str, float]] = None
    #: Kills and deaths in this episode, counted off the same champion-death
    #: transitions the reward's kill/death pair is charged on. The server
    #: prints ``deaths=`` on its ``LANERL_CS`` rows and nothing ever read it;
    #: with ``--no-end-on-death`` an episode can now contain several, and
    #: "is the agent feeding?" was unanswerable from the metrics.
    kills: int = 0
    deaths: int = 0
    #: The agent champion's attack damage and max HP on the FIRST frame of the
    #: episode.
    #:
    #: A canary for one specific regression: an in-process reset that strips
    #: the rune page takes ``mhp`` 672 -> 616 and the AD with it, so every
    #: episode after the first is a quietly different game from the one being
    #: measured -- and nothing in the observation, the reward or the CS readout
    #: looks wrong while it happens. ``None`` when the wire did not carry the
    #: field (older recordings); never substitute 0.
    first_frame_ad: Optional[float] = None
    first_frame_mhp: Optional[float] = None

    def __post_init__(self) -> None:
        if not (0.0 <= self.score <= 1.0):
            raise ValueError(f"score must be in [0,1], got {self.score}")


@dataclass
class Rollout:
    """One actor's contribution to one update."""

    actor_id: int
    param_version: int
    #: Rollout ROWS, not environment decisions.  One row is one timestep across
    #: every parallel env in this actor's batch, so the decision count is
    #: ``steps * parallel_envs``.  ``RunState.total_env_steps`` sums this field
    #: and is therefore also in rows; the first run's 4,131,000 "env steps"
    #: (``state.json``, final) were 33,048,000 actual decisions at 8 envs.  The
    #: 3,852,540 / 30.8M that stood here was a mid-run snapshot, not the total.
    #: Left as-is because the reward's zero-sum anneal
    #: is already denominated in it, and moving that clock mid-run would restart
    #: the anneal.
    steps: int
    data: Any = None
    episodes: List[EpisodeResult] = field(default_factory=list)
    #: Realised opponent mixture within this rollout, for drift checking.
    mixture: Mapping[str, float] = field(default_factory=dict)
    collected_at: float = field(default_factory=time.time)
    #: How many (instance, side) slots one row covers.  Set by the collector;
    #: 1 means "this rollout counts rows and decisions the same way".
    parallel_envs: int = 1
    #: Fraction of this rollout's decisions spent on each button
    #: (``lane_wiring.button_marginals``).  Logged per update, because it is
    #: the one curve that shows a behaviour-cloning prior eroding: entropy
    #: rose 1.64 -> 3.96 over 140 updates of the first BC-initialised run
    #: while CS went nowhere, and nothing recorded WHAT the policy had started
    #: doing instead.
    action_marginals: Mapping[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------


class MetricsLog:
    """Append-only JSONL, flushed per record.

    Flushed rather than buffered because the interesting runs are the ones that
    die: a metrics file that loses its last 8 KB loses precisely the records
    that explain the death.
    """

    def __init__(self, path: Path, fsync: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", buffering=1)
        self._lock = threading.Lock()
        self.fsync = bool(fsync)
        self.written = 0

    def write(self, kind: str, **fields: Any) -> None:
        rec = {"kind": kind, "wall": time.time(), **fields}
        line = json.dumps(rec, separators=(",", ":"), default=str)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            if self.fsync:
                os.fsync(self._fh.fileno())
            self.written += 1

    def close(self) -> None:
        with self._lock:
            if not self._fh.closed:
                self._fh.close()


# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------


class ParameterStore:
    """The learner's published parameters, versioned and thread-safe."""

    def __init__(self, payload: Optional[Mapping[str, Any]] = None):
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)
        self._version = 0
        self._payload: Mapping[str, Any] = payload if payload is not None else {}

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    def publish(self, payload: Mapping[str, Any]) -> int:
        with self._cv:
            self._version += 1
            self._payload = payload
            self._cv.notify_all()
            return self._version

    def pull(self) -> tuple:
        """``(version, payload)`` as of now."""
        with self._lock:
            return self._version, self._payload

    def wait_for_version_above(self, version: int, timeout: float) -> bool:
        with self._cv:
            return self._cv.wait_for(lambda: self._version > version, timeout=timeout)


class StalenessTracker:
    """Enforces the parameter-lag bound and reports on it.

    A rejected rollout is wasted actor time, so the rejection *rate* is the
    metric that matters: a bound that is silently rejecting a third of the data
    is a misconfiguration, not a safety net doing its job.
    """

    def __init__(self, max_staleness: int = 1, warn_reject_rate: float = 0.05):
        if max_staleness < 0:
            raise ValueError("max_staleness must be >= 0")
        self.max_staleness = int(max_staleness)
        self.warn_reject_rate = float(warn_reject_rate)
        self.accepted = 0
        self.rejected = 0
        #: Worst lag seen at all, rejections included.
        self.max_observed = 0
        #: Worst lag of data actually trained on.  This is the number the bound
        #: guarantees, and the one worth plotting: ``max_observed`` can exceed
        #: the bound precisely because the guard did its job.
        self.max_accepted = 0
        self._sum = 0
        self._warned_at = 0

    def staleness(self, rollout_version: int, learner_version: int) -> int:
        return max(0, learner_version - rollout_version)

    def admit(self, rollout_version: int, learner_version: int) -> bool:
        s = self.staleness(rollout_version, learner_version)
        self.max_observed = max(self.max_observed, s)
        if s > self.max_staleness:
            self.rejected += 1
            log.error(
                "REJECTED a rollout at staleness %d > max %d (rollout v%d, learner v%d). "
                "That is discarded actor work: shrink the rollout queue, add actors, or "
                "raise max_staleness deliberately.",
                s,
                self.max_staleness,
                rollout_version,
                learner_version,
            )
            self._maybe_warn()
            return False
        self.accepted += 1
        self.max_accepted = max(self.max_accepted, s)
        self._sum += s
        return True

    @property
    def total(self) -> int:
        return self.accepted + self.rejected

    @property
    def reject_rate(self) -> float:
        return self.rejected / self.total if self.total else 0.0

    @property
    def mean_staleness(self) -> float:
        return self._sum / self.accepted if self.accepted else 0.0

    def _maybe_warn(self) -> None:
        if self.total < 20 or self.rejected == self._warned_at:
            return
        if self.reject_rate >= self.warn_reject_rate:
            self._warned_at = self.rejected
            log.error(
                "staleness rejection rate is %.1f%% (%d of %d). The actors are producing "
                "data the learner will not use.",
                100.0 * self.reject_rate,
                self.rejected,
                self.total,
            )

    def stats(self) -> Dict[str, float]:
        return {
            "accepted": self.accepted,
            "rejected": self.rejected,
            "reject_rate": self.reject_rate,
            "mean_staleness": self.mean_staleness,
            "max_staleness_observed": self.max_observed,
            "max_staleness_trained_on": self.max_accepted,
            "max_staleness_allowed": self.max_staleness,
        }


# --------------------------------------------------------------------------
# Throughput and resource telemetry
# --------------------------------------------------------------------------


class ThroughputMeter:
    """How fast the run is going, and where each update's wall time went.

    The first real run produced 16,200 updates and ``metrics.jsonl`` could not
    answer "is this compute-bound, env-bound or stalled?".  The only throughput
    figure anyone had was anecdotal ("nvidia-smi says about 5%"), which is not a
    number you can put in a post-mortem or compare against the next run.

    Three quantities settle it from the log alone:

    ``learner_s``     seconds inside ``learner.update()`` for this update;
    ``wait_s``        seconds blocked on ``queue.get()`` waiting for a rollout;
    ``learner_frac``  ``learner_s / update_s``.

    ``learner_frac`` near 1 is compute-bound, near 0 is env-bound, and an
    ``update_s`` that grows while neither of the other two does is a stall
    somewhere else entirely (metrics I/O, checkpointing, the eval hook).

    The rolling window is counted in **updates, not seconds**: a run whose rate
    collapses must still report a rate, and a time-based window reports nothing
    at exactly the moment the number is most wanted.

    ``clock`` is injected so a test can assert exact rates instead of sleeping.
    """

    def __init__(self, window: int = 50, clock: Callable[[], float] = time.monotonic):
        if window < 1:
            raise ValueError(f"window must be >= 1 update, got {window}")
        self.window = int(window)
        self._clock = clock
        self.started_at = float(clock())
        #: ``(wall, cumulative rows, cumulative decisions)`` marks, oldest first.
        self._marks: Deque[Tuple[float, int, int]] = deque(maxlen=self.window + 1)
        self._marks.append((self.started_at, 0, 0))
        self.updates = 0
        self.total_steps = 0
        self.total_decisions = 0
        #: What :meth:`seed` carried in from a previous process, so the
        #: lifetime rates in :meth:`totals` stay rates *of this process* while
        #: the cumulative counters stay cumulative over the whole run.
        self._seed_steps = 0
        self._seed_decisions = 0
        self.total_wait_s = 0.0
        self.total_learner_s = 0.0
        self.total_rejected_s = 0.0
        self._wait_s = 0.0
        self._learner_s = 0.0
        self._rejected_s = 0.0

    # -- accounting for one update ----------------------------------------

    def note_wait(self, seconds: float) -> None:
        self._wait_s += float(seconds)
        self.total_wait_s += float(seconds)

    def note_learner(self, seconds: float) -> None:
        self._learner_s += float(seconds)
        self.total_learner_s += float(seconds)

    @contextmanager
    def waiting(self) -> Iterator[None]:
        t0 = self._clock()
        try:
            yield
        finally:
            self.note_wait(self._clock() - t0)

    @contextmanager
    def learning(self) -> Iterator[None]:
        t0 = self._clock()
        try:
            yield
        finally:
            self.note_learner(self._clock() - t0)

    def discard_pending(self) -> None:
        """Reclassify the pending wait as time spent on a rollout that was thrown away.

        A rollout rejected for staleness never becomes an update, so its wait
        must not be billed as productive ``wait_s`` on the next one -- but it
        must not vanish either, or the buckets stop adding up to ``update_s``
        and the log quietly loses the most expensive thing in the run.
        ``wait_s + learner_s + rejected_s`` should account for ``update_s``;
        whatever is left over is overhead in this loop itself.
        """
        self._rejected_s += self._wait_s + self._learner_s
        self.total_rejected_s += self._wait_s + self._learner_s
        self._wait_s = 0.0
        self._learner_s = 0.0

    def seed(self, total_steps: int, total_decisions: int) -> None:
        """Adopt a resumed run's lifetime counters.

        Without this, ``total_decisions`` in the metrics restarts at zero on
        every resume -- the first run resumed 7 times -- so the log offers no
        cumulative sample count at all, which is the very confusion the
        rows-vs-decisions split exists to end.  The window baseline moves with
        it, or the first update after a resume reports a rate computed against
        millions of steps in no time at all.
        """
        self.total_steps = int(total_steps)
        self.total_decisions = int(total_decisions)
        self._seed_steps = self.total_steps
        self._seed_decisions = self.total_decisions
        self._marks.clear()
        self._marks.append((float(self._clock()), self.total_steps, self.total_decisions))

    def record(self, steps: int, parallel_envs: int = 1) -> Dict[str, Any]:
        """Close out one update and return the row fields to log."""
        now = float(self._clock())
        prev_wall = self._marks[-1][0]
        self.updates += 1
        self.total_steps += int(steps)
        self.total_decisions += int(steps) * max(1, int(parallel_envs))
        self._marks.append((now, self.total_steps, self.total_decisions))
        t0, s0, d0 = self._marks[0]
        span = now - t0
        n = len(self._marks) - 1
        update_s = now - prev_wall
        out: Dict[str, Any] = {
            "wall_s": now - self.started_at,
            "update_s": update_s,
            "wait_s": self._wait_s,
            "learner_s": self._learner_s,
            "rejected_s": self._rejected_s,
            "learner_frac": (self._learner_s / update_s) if update_s > 0 else None,
            "window_updates": n,
            "updates_per_s": (n / span) if span > 0 else None,
            "env_steps_per_s": ((self.total_steps - s0) / span) if span > 0 else None,
            "decisions_per_s": ((self.total_decisions - d0) / span) if span > 0 else None,
            "total_decisions": self.total_decisions,
        }
        self._wait_s = 0.0
        self._learner_s = 0.0
        self._rejected_s = 0.0
        return out

    def totals(self) -> Dict[str, Any]:
        """Lifetime figures, for the shutdown/failure record."""
        span = float(self._clock()) - self.started_at
        return {
            "total_wall_s": span,
            "total_wait_s": self.total_wait_s,
            "total_learner_s": self.total_learner_s,
            "total_rejected_s": self.total_rejected_s,
            "mean_updates_per_s": (self.updates / span) if span > 0 else None,
            # Net of anything `seed` carried in: these are rates of THIS
            # process, while the cumulative counters span the whole run.
            "mean_env_steps_per_s": (
                (self.total_steps - self._seed_steps) / span if span > 0 else None
            ),
            "mean_decisions_per_s": (
                (self.total_decisions - self._seed_decisions) / span if span > 0 else None
            ),
        }


class GpuProbe:
    """GPU utilisation and memory, or ``None`` everywhere it cannot be had.

    Every field is optional on purpose.  This stack is unit-tested on a CPU-only
    path, ``torch.cuda.utilization()`` needs ``pynvml`` which is not installed on
    either node here, and a telemetry call that can raise is a telemetry call
    that will eventually kill a run.  So each probe is tried once, and a failure
    disables that probe for the life of the process rather than paying an
    exception per update.

    ``utilization()`` is an NVML round trip (~1 ms measured elsewhere) against
    ~2 us for ``memory_allocated()``, so it is sampled every
    ``utilization_every`` updates and the cached reading is reused in between.
    """

    def __init__(self, device: Optional[str] = None, utilization_every: int = 25):
        self.utilization_every = max(1, int(utilization_every))
        self._calls = 0
        self._torch = None
        self._enabled = False
        self._util_ok = True
        self._util_cached: Optional[float] = None
        self.device = device
        try:  # pragma: no cover - depends on the environment
            import torch

            self._torch = torch
            self._enabled = bool(device is None or "cuda" in str(device)) and torch.cuda.is_available()
        except Exception:
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def sample(self) -> Dict[str, Any]:
        if not self._enabled:
            return {"util_pct": None, "mem_allocated_mb": None, "mem_reserved_mb": None}
        torch = self._torch
        out: Dict[str, Any] = {"util_pct": None, "mem_allocated_mb": None, "mem_reserved_mb": None}
        try:
            out["mem_allocated_mb"] = torch.cuda.memory_allocated() / (1024.0 * 1024.0)
            out["mem_reserved_mb"] = torch.cuda.memory_reserved() / (1024.0 * 1024.0)
        except Exception as exc:  # pragma: no cover - driver-dependent
            log.error("GPU memory probe failed and is now disabled: %s", exc)
            self._enabled = False
            return out
        if self._util_ok and self._calls % self.utilization_every == 0:
            try:
                self._util_cached = float(torch.cuda.utilization())
            except Exception as exc:
                self._util_ok = False
                self._util_cached = None
                log.warning(
                    "torch.cuda.utilization() is unavailable (%s); GPU memory is still "
                    "logged but gpu/util_pct will be null for this run. `pip install "
                    "pynvml` if the utilisation number is wanted.",
                    exc,
                )
        out["util_pct"] = self._util_cached
        self._calls += 1
        return out


# --------------------------------------------------------------------------
# The training clock the environments read
# --------------------------------------------------------------------------


class TrainStepCounter:
    """How far along the run is, published to the environment side.

    ``lanerl_rl.reward.LaneRewardConfig`` anneals the zero-sum coefficient
    ``alpha`` from 0.5 to 1.0 over ``zero_sum_anneal_steps``, driven by a number
    the trainer is supposed to hand it.  Nothing handed it one, so alpha was
    pinned at 0.5 for the whole of every run and the anneal was dead code.  This
    is the missing hand-off.

    Read from actor threads and written from the learner thread, hence the lock.
    Monotonic on purpose: a schedule that can go backwards is a schedule that
    can be replayed, and a resume that forgot to restore it would silently
    restart the anneal.
    """

    def __init__(self, value: int = 0):
        self._lock = threading.Lock()
        self._value = int(value)

    @property
    def value(self) -> int:
        with self._lock:
            return self._value

    def __call__(self) -> int:
        """So it can be passed straight in as a ``train_step_source`` callable."""
        return self.value

    def set(self, value: int) -> int:
        v = int(value)
        with self._lock:
            if v < self._value:
                log.error(
                    "train step went backwards (%d -> %d); ignoring. Every env-side "
                    "schedule reads this, so a rewind would silently re-run an anneal.",
                    self._value,
                    v,
                )
                return self._value
            self._value = v
            return v


# --------------------------------------------------------------------------
# Checkpoints
# --------------------------------------------------------------------------


class CheckpointManager:
    """Writes checkpoints atomically and knows which one is newest.

    ``torch.save`` when torch is importable, pickle otherwise -- the training
    loop must be testable without importing a 2 GB CUDA stack.  Writes go to a
    temp file and are renamed, so a checkpoint interrupted mid-write never
    becomes the file a resume picks up.
    """

    def __init__(self, directory: Path, keep_last: Optional[int] = None):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self._torch = None
        try:  # pragma: no cover - depends on the environment
            import torch  # noqa: F401

            self._torch = torch
            self.suffix = ".pt"
        except Exception:
            self.suffix = ".pkl"

    def path_for(self, update: int) -> Path:
        return self.dir / f"update_{update:08d}{self.suffix}"

    def save(self, update: int, payload: Mapping[str, Any]) -> Path:
        target = self.path_for(update)
        tmp = target.with_suffix(target.suffix + ".tmp")
        if self._torch is not None:
            self._torch.save(dict(payload), tmp)
        else:
            import pickle

            with tmp.open("wb") as fh:
                pickle.dump(dict(payload), fh)
        os.replace(tmp, target)
        self._prune()
        log.info("checkpoint written: %s", target)
        return target

    def load(self, path: Path) -> Mapping[str, Any]:
        p = Path(path)
        if not p.exists():
            raise TrainingError(f"checkpoint {p} does not exist")
        if self._torch is not None and p.suffix == ".pt":
            return self._torch.load(p, map_location="cpu", weights_only=False)
        import pickle

        with p.open("rb") as fh:
            return pickle.load(fh)

    def all_checkpoints(self) -> List[Path]:
        return sorted(self.dir.glob(f"update_*{self.suffix}"))

    def latest(self) -> Optional[Path]:
        found = self.all_checkpoints()
        return found[-1] if found else None

    #: Never prune a checkpoint whose update number is a multiple of this.
    #: Rotation deleted EVERY early checkpoint of the first real run, so when
    #: the policy turned out not to be learning there was nothing left to
    #: compare against -- no before/after, no regression bisect, nothing. A
    #: milestone ladder costs a few hundred MB and buys the whole history.
    MILESTONE_EVERY = 1000

    def _is_milestone(self, p: Path) -> bool:
        try:
            return int(p.stem.split("_")[1]) % self.MILESTONE_EVERY == 0
        except (IndexError, ValueError):
            return False

    def _prune(self) -> None:
        if not self.keep_last:
            return
        found = [p for p in self.all_checkpoints() if not self._is_milestone(p)]
        for p in found[: max(0, len(found) - self.keep_last)]:
            try:
                p.unlink()
            except OSError as exc:
                log.error("could not prune checkpoint %s: %s", p, exc)


# --------------------------------------------------------------------------
# Configuration and persistent state
# --------------------------------------------------------------------------


@dataclass
class RunConfig:
    run_dir: Path
    num_actors: int = 2
    envs_per_actor: int = 8
    rollout_steps: int = 128
    #: How many learner versions old a rollout may be and still be trained on.
    #:
    #: ``None`` (the default) derives it from the architecture:
    #: ``queue_capacity + num_actors - 1``, which is exactly the worst case
    #: ``__post_init__`` computes below. Anything stricter rejects work the
    #: pipeline is GUARANTEED to produce.
    #:
    #: This was a hard ``1``, with the note "OpenAI Five: >8 versions of lag
    #: costs real throughput. 0-1 is the target." That misreads the result.
    #: Five ENGINEERED staleness down to 1-2 and measured that >8 hurt sample
    #: efficiency; they did not set a reject threshold at 1. Rejecting is not a
    #: way to achieve low staleness -- the rollout has already been collected,
    #: so the only thing a rejection saves is the learner step, and the only
    #: thing it costs is every CPU-second the servers and actors spent.
    #:
    #: Measured, on the 2026-09-13 scaling probe: EVERY rejection across all
    #: five configs was staleness exactly 2, against this bound of 1. That
    #: discarded 45% of collected rollouts at 8 instances, rising to 80% at 96
    #: -- four of every five games simulated, built and then binned. Correcting
    #: for it, collection throughput was FLAT at ~900 decisions/s across a 12x
    #: range of instances; the "throughput falls as you add instances" curve
    #: that sent three separate investigations after CPU and stragglers was
    #: mostly this, because `decisions_per_s` only counts rollouts the learner
    #: accepts.
    #:
    #: ``__post_init__`` still warns when an EXPLICIT value is below the
    #: architectural worst case -- that warning existed and fired all along,
    #: which is the other lesson here.
    max_staleness: Optional[int] = None
    queue_capacity: int = 2
    total_updates: int = 1_000_000
    checkpoint_every: int = 200
    snapshot_every: int = 400  # how often a checkpoint joins the opponent pool
    eval_every: int = 100
    keep_last_checkpoints: Optional[int] = None
    #: No completed update within this window means the run is stalled.
    stall_timeout_s: float = 900.0
    seed: int = 0
    #: Which clock drives the env-side schedules (currently only the reward's
    #: zero-sum anneal).  ``"env_steps"`` counts decisions collected across all
    #: instances; ``"updates"`` counts learner steps.  They differ by
    #: ``rollout_steps * envs_per_actor``, i.e. three orders of magnitude here,
    #: so which one ``LaneRewardConfig.zero_sum_anneal_steps`` is denominated in
    #: is not a detail.  Default ``"env_steps"``: the published default of
    #: 2,000,000 is a sample count in every reference this reward is copied
    #: from, and 2M *updates* at this batch size is a run nobody will ever
    #: finish.  Set it deliberately rather than inheriting the guess.
    anneal_clock: str = "env_steps"
    league: LeagueConfig = field(default_factory=LeagueConfig)

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        if self.anneal_clock not in ("env_steps", "updates"):
            raise ValueError(
                f"anneal_clock must be 'env_steps' or 'updates', got {self.anneal_clock!r}"
            )
        if self.num_actors < 0:
            raise ValueError("num_actors must be >= 0")
        if self.rollout_steps <= 0 or self.queue_capacity <= 0:
            raise ValueError("rollout_steps and queue_capacity must be positive")
        worst = self.queue_capacity + max(self.num_actors, 1) - 1
        if self.max_staleness is None:
            self.max_staleness = worst
            log.info(
                "max_staleness not set; deriving %d from queue_capacity=%d and "
                "num_actors=%d so the learner does not reject rollouts this "
                "configuration is guaranteed to produce",
                worst, self.queue_capacity, self.num_actors,
            )
        elif worst > self.max_staleness:
            log.warning(
                "with queue_capacity=%d and num_actors=%d the learner can publish up to %d "
                "versions between an actor's pull and its push, above max_staleness=%d. "
                "Expect rejected rollouts; shrink the queue or raise the bound knowingly.",
                self.queue_capacity,
                self.num_actors,
                worst,
                self.max_staleness,
            )


@dataclass
class RunState:
    """Everything a resume needs that is not weights."""

    update: int = 0
    param_version: int = 0
    #: Rollout ROWS, summed over every accepted rollout -- **not** decisions.
    #: One row covers ``Rollout.parallel_envs`` simultaneous decisions, so the
    #: first run's 4,131,000 "env steps" were 33,048,000 decisions (8 slots: 2
    #: sides x 4 instances).  Kept in rows deliberately: the reward's zero-sum
    #: anneal is denominated in this number (``_anneal_clock_value``), and
    #: redefining it mid-flight would restart the anneal on every existing run.
    #: :attr:`total_decisions` is the honest sample count; use that one for
    #: anything comparing this stack to a published sample budget.
    total_env_steps: int = 0
    #: Environment decisions, ``sum(rows * parallel_envs)``.  Persisted rather
    #: than left to :class:`ThroughputMeter`, whose copy starts from zero on
    #: every resume -- the first run resumed 7 times, so its logged decision
    #: count restarted 7 times too.
    total_decisions: int = 0
    total_episodes: int = 0
    rng_state: Optional[list] = None
    league: Dict[str, Any] = field(default_factory=dict)
    checkpoint: Optional[str] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "RunState":
        data = json.loads(Path(path).read_text())
        return cls(**data)


# --------------------------------------------------------------------------
# Actors
# --------------------------------------------------------------------------


class ActorLoop:
    """One async actor thread.

    Owns nothing about the environment: ``collect`` is injected, so the same
    loop drives a real :class:`lanerl_train.vec.VecDriver` in production and a
    deterministic fake in a test.  Its only responsibilities are pulling fresh
    parameters before each rollout, respecting backpressure, and *never
    swallowing an exception* -- a dead actor that leaves the run apparently
    healthy is the most expensive failure mode this project has.
    """

    def __init__(
        self,
        actor_id: int,
        store: ParameterStore,
        out_queue: "queue.Queue[Rollout]",
        collect: Callable[[int, Mapping[str, Any], int], Rollout],
        error_queue: "queue.Queue[tuple]",
        stop_event: threading.Event,
        put_timeout_s: float = 300.0,
    ):
        self.actor_id = int(actor_id)
        self.store = store
        self.out_queue = out_queue
        self.collect = collect
        self.error_queue = error_queue
        self.stop_event = stop_event
        self.put_timeout_s = float(put_timeout_s)
        self.rollouts = 0
        self.pulls = 0
        self.param_version = -1
        self.thread: Optional[threading.Thread] = None

    def run_once(self) -> Optional[Rollout]:
        """Pull parameters, collect one rollout, enqueue it.

        The pull happens *before* collection, which is what keeps staleness at
        0 for the first step of the rollout and at most the number of learner
        updates that land during it.
        """
        version, payload = self.store.pull()
        if version != self.param_version:
            self.pulls += 1
        self.param_version = version
        rollout = self.collect(self.actor_id, payload, version)
        if rollout is None:
            return None
        rollout.actor_id = self.actor_id
        rollout.param_version = version
        try:
            self.out_queue.put(rollout, timeout=self.put_timeout_s)
        except queue.Full:
            raise TrainingError(
                f"actor {self.actor_id}: the learner has not consumed a rollout in "
                f"{self.put_timeout_s:.0f}s. The learner is stuck or dead; refusing to "
                f"buffer more and hide it."
            ) from None
        self.rollouts += 1
        return rollout

    def _run(self) -> None:
        try:
            while not self.stop_event.is_set():
                self.run_once()
        except BaseException as exc:  # noqa: BLE001 - deliberately everything
            self.error_queue.put((self.actor_id, exc, traceback.format_exc()))
            log.error("ACTOR %d DIED: %s", self.actor_id, exc, exc_info=True)
            self.stop_event.set()
            # Wake a learner blocked on get() instead of making it wait out the
            # full stall timeout to discover a failure that already happened.
            try:
                self.out_queue.put_nowait(None)  # type: ignore[arg-type]
            except queue.Full:
                pass

    def start(self) -> threading.Thread:
        self.thread = threading.Thread(
            target=self._run, name=f"lanerl-actor-{self.actor_id}", daemon=True
        )
        self.thread.start()
        return self.thread


# --------------------------------------------------------------------------
# The learner loop
# --------------------------------------------------------------------------


class TrainingLoop:
    """Central learner: consume rollouts, update, publish, record, checkpoint."""

    def __init__(
        self,
        config: RunConfig,
        learner: Any,
        collect: Optional[Callable[[int, Mapping[str, Any], int], Rollout]] = None,
        evaluator: Optional[Evaluator] = None,
        sampler: Optional[OpponentSampler] = None,
        metrics: Optional[MetricsLog] = None,
        anchor_eval: Optional[Callable[[int, Mapping[str, Any]], List[EpisodeResult]]] = None,
        throughput: Optional["ThroughputMeter"] = None,
        gpu: Optional["GpuProbe"] = None,
    ):
        self.cfg = config
        self.learner = learner
        self.collect = collect
        self.run_dir = Path(config.run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics or MetricsLog(self.run_dir / "metrics.jsonl")
        self.checkpoints = CheckpointManager(
            self.run_dir / "checkpoints", config.keep_last_checkpoints
        )
        self.store = ParameterStore(self._policy_payload())
        self.queue: "queue.Queue[Rollout]" = queue.Queue(maxsize=config.queue_capacity)
        self.errors: "queue.Queue[tuple]" = queue.Queue()
        self.stop_event = threading.Event()
        self.staleness = StalenessTracker(config.max_staleness)
        self.evaluator = evaluator or Evaluator()
        pool = CheckpointPool(config.league)
        self.sampler = sampler or OpponentSampler(
            pool,
            WinRateTracker(config.league),
            anchors=self.evaluator.anchors,
            config=config.league,
            rng=random.Random(config.seed),
        )
        self.rng = random.Random(config.seed)
        self.state = RunState(param_version=self.store.version)
        #: Published to the environment side; see :class:`TrainStepCounter`.
        #: Hand ``loop.train_steps`` to anything that needs to know how far the
        #: run has got -- it is callable, so it drops straight into
        #: ``lanerl_rl.env.LaneEnv(train_step_source=...)``.
        self.train_steps = TrainStepCounter()
        self.actors: List[ActorLoop] = []
        self._last_update_at = time.monotonic()
        self.throughput = throughput or ThroughputMeter()
        self.gpu = gpu or GpuProbe(getattr(config, "device", None))
        #: Periodic evaluation against a frozen opponent; see
        #: :mod:`lanerl_train.anchor_eval`.  ``None`` means the run has none, and
        #: :meth:`require_anchor_eval` is what turns that into a startup failure
        #: rather than an eval section that quietly reports ``(None, 0)`` for
        #: every anchor forever, which is what the first run did.
        self.anchor_eval = anchor_eval

    # -- helpers -----------------------------------------------------------

    def _anneal_clock_value(self) -> int:
        if self.cfg.anneal_clock == "updates":
            return self.state.update
        return self.state.total_env_steps

    def _publish_train_step(self) -> int:
        return self.train_steps.set(self._anneal_clock_value())

    def _policy_payload(self) -> Mapping[str, Any]:
        fn = getattr(self.learner, "policy_payload", None)
        return dict(fn()) if callable(fn) else {}

    @property
    def state_path(self) -> Path:
        return self.run_dir / "state.json"

    def agent_id(self, update: Optional[int] = None) -> str:
        """The run's own identity for rating and CS purposes.

        Bucketed to ``snapshot_every``, not to the update number, and this is
        load-bearing.  It used to be ``agent@<update>``, i.e. a NEW player on
        every single update, so:

        * ``Evaluator.report`` asked for ``cs_at_10`` of ``agent@9000`` while
          episodes had been recorded under whatever the collector called itself
          (``"self"``).  The first run logged CS@10 for 179 episodes and every
          one of its 40 eval reports still said ``cs_at_10: null``;
        * no ``(latest, past)`` pair could ever reach the 10 games
          ``min_win_rate_vs_past`` needs, so the AlphaStar rot signature was
          structurally unreachable;
        * win rate against an anchor would reset after every update.

        A snapshot interval is already this project's unit of "a distinguishable
        version of the agent" -- it is what enters the opponent pool -- so it is
        the right granularity for a rating too.
        """
        era = int(self.cfg.snapshot_every or 0)
        # `update` lets a caller ask for the id of an EARLIER bucket. The eval
        # report needs it: it fires on the first update of a fresh bucket, so
        # the bucket holding the episodes is the previous one.
        at = self.state.update if update is None else int(update)
        bucket = (at // era) * era if era > 0 else 0
        return f"agent@{bucket}"

    # -- persistence -------------------------------------------------------

    def save_state(self, checkpoint: Optional[Path] = None) -> Path:
        self.state.rng_state = list(self.rng.getstate())  # type: ignore[arg-type]
        self.state.league = self.sampler.to_dict()
        if checkpoint is not None:
            self.state.checkpoint = str(checkpoint)
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(self.state.to_json())
        os.replace(tmp, self.state_path)
        return self.state_path

    def resume(self) -> bool:
        """Restore from ``run_dir``.  Returns False when there is nothing to resume."""
        if not self.state_path.exists():
            log.info("no state.json in %s; starting fresh", self.run_dir)
            return False
        self.state = RunState.load(self.state_path)
        if self.state.rng_state:
            s = self.state.rng_state
            self.rng.setstate((s[0], tuple(s[1]), s[2]))
        if self.state.league:
            self.sampler.load_dict(self.state.league)
        ckpt = self.state.checkpoint or self.checkpoints.latest()
        if ckpt is not None:
            payload = self.checkpoints.load(Path(ckpt))
            loader = getattr(self.learner, "load_payload", None)
            if callable(loader):
                loader(payload)
            else:
                raise TrainingError(
                    f"{type(self.learner).__name__} has no load_payload(); a resume that "
                    f"silently keeps random weights is worse than no resume"
                )
            log.info("resumed from %s at update %d", ckpt, self.state.update)
        else:
            log.error(
                "state.json says update=%d but there is no checkpoint in %s. Resuming the "
                "counters onto untrained weights would corrupt every metric downstream.",
                self.state.update,
                self.checkpoints.dir,
            )
            raise TrainingError("resume requested but no checkpoint found")
        # The store must carry the resumed weights, not the constructor's.
        self.store = ParameterStore(self._policy_payload())
        for _ in range(max(0, self.state.param_version - self.store.version)):
            self.store.publish(self._policy_payload())
        # And the training clock must carry the resumed position, or a resume
        # silently rewinds every env-side anneal to its start.
        self._publish_train_step()
        if self.state.total_decisions == 0 and self.state.total_env_steps > 0:
            log.warning(
                "state.json has total_env_steps=%d but no total_decisions: it was written "
                "before the two clocks were separated. The decision count starts from this "
                "resume and UNDERSTATES the run; total_env_steps (rows) is unaffected and "
                "so is the anneal. Multiply by the rollout's parallel_envs to recover it.",
                self.state.total_env_steps,
            )
        self.throughput.seed(self.state.total_env_steps, self.state.total_decisions)
        self.metrics.write(
            "resume",
            update=self.state.update,
            checkpoint=str(ckpt),
            train_step=self.train_steps.value,
        )
        return True

    # -- ingestion ---------------------------------------------------------

    def record_episode(self, ep: EpisodeResult) -> None:
        """Feed one episode into the league, the evaluator and the metrics log."""
        self.state.total_episodes += 1
        # Everything downstream is keyed by the RUN's agent id, not by whatever
        # the collector called itself. ``collect_rollout`` labels its episodes
        # with the policy key ("self"), so CS recorded under that label was
        # invisible to a report asking about "agent@N" -- 298 CS@10 readings in
        # the first run, every eval row still saying null. See agent_id().
        agent = self.agent_id()
        # Skip rating a self-match as well as the LATEST sentinel. Under a pure
        # self-play mixture BOTH sides carry the same id (mixture {"self": 1.0}),
        # which is not the LATEST sentinel, so it slipped through to MatchRecord
        # and tripped its guard -- killing the run at the first completed
        # episode. A self-match carries no rating information either way; it is
        # recorded as an episode below, which is what MatchRecord's own error
        # message tells you to do.
        is_self_match = ep.opponent_id in (LATEST, ep.agent, agent)
        if not is_self_match:
            self.sampler.win_rates.record(ep.opponent_id, ep.score)
            self.evaluator.record_match(
                MatchRecord(
                    agent_a=agent,
                    agent_b=ep.opponent_id,
                    score_a=ep.score,
                    step=self.state.update,
                    meta={"category": ep.opponent_category, "reason": ep.reason},
                )
            )
        if ep.cs_at_10 is not None:
            # WITH the category. Pooling a mirror match, a scripted-bot
            # training game and an anchor game into one mean produces a
            # headline CS whose value depends on the eval cadence -- and
            # ``load_jsonl`` passes the category, so a replay of this run's own
            # metrics reached a conclusion the live report could not.
            self.evaluator.record_cs(agent, ep.cs_at_10, category=ep.opponent_category)
        # The same call ``Evaluator.load_jsonl`` makes when it replays the
        # episode row written below. Only the replay path had it, so a live
        # run's report and a report rebuilt from that run's own metrics.jsonl
        # disagreed about the opponent's CS -- and the live one is the one
        # anybody watches.
        if ep.opponent_cs_at_10 is not None and ep.opponent_id:
            self.evaluator.record_opponent_cs(ep.opponent_id, ep.opponent_cs_at_10)
        self.metrics.write(
            "episode",
            update=self.state.update,
            ep_return=ep.ep_return,
            # The id the evaluator used, so Evaluator.load_jsonl replays to the
            # same table the live run built. The collector's own label is kept
            # beside it rather than instead of it.
            agent=agent,
            collector_agent=ep.agent,
            opponent=ep.opponent_id,
            opponent_category=ep.opponent_category,
            score=ep.score,
            cs_at_10=ep.cs_at_10,
            # The opponent's CS from the SAME game. It was computed, logged to
            # INFO by AnchorEvaluator, and then dropped on the floor -- so the
            # only durable record of how the agent did RELATIVE to the thing it
            # played was a hardcoded constant measured on a different bot.
            opponent_cs_at_10=ep.opponent_cs_at_10,
            length_steps=ep.length_steps,
            reason=ep.reason,
            instance=ep.instance,
            kills=ep.kills,
            deaths=ep.deaths,
            # The rune-page canary; see EpisodeResult.first_frame_ad.
            first_frame_ad=ep.first_frame_ad,
            first_frame_mhp=ep.first_frame_mhp,
            # One flat object of ~12 floats, per EPISODE (~1 per 18,000
            # decisions), not per step: this is what makes the log able to
            # answer "which term is the policy chasing?" without becoming the
            # dominant cost of the run.
            reward_terms=dict(ep.reward_terms) if ep.reward_terms else None,
        )
        if not is_self_match:
            self.metrics.write(
                "match",
                agent_a=agent,
                agent_b=ep.opponent_id,
                score_a=ep.score,
                step=self.state.update,
                meta={"category": ep.opponent_category},
            )

    def submit(self, rollout: Rollout) -> None:
        """Enqueue a rollout directly (synchronous / test use)."""
        self.queue.put(rollout, timeout=self.cfg.stall_timeout_s)

    # -- the update --------------------------------------------------------

    def _check_actors(self) -> None:
        try:
            actor_id, exc, tb = self.errors.get_nowait()
        except queue.Empty:
            return
        raise ActorFailure(f"actor {actor_id} died: {exc}\n{tb}") from exc

    def step_once(self, timeout: Optional[float] = None) -> bool:
        """Consume one rollout and perform at most one update.

        Returns True when an update happened, False when the rollout was
        rejected for staleness (already counted and logged).
        """
        self._check_actors()
        wait = self.cfg.stall_timeout_s if timeout is None else timeout
        try:
            with self.throughput.waiting():
                rollout = self.queue.get(timeout=wait)
        except queue.Empty:
            self.throughput.discard_pending()
            self._check_actors()
            raise StalledRun(
                f"no rollout in {wait:.0f}s at update {self.state.update}. Actors are alive "
                f"but not producing -- check the per-instance server logs."
            ) from None
        self._check_actors()
        if rollout is None:  # the wake-up sentinel from a dying actor
            raise ActorFailure(
                "an actor put its failure sentinel on the queue but reported no exception; "
                "this is a bug in ActorLoop, not a recoverable condition"
            )

        if not self.staleness.admit(rollout.param_version, self.state.param_version):
            self.throughput.discard_pending()
            self.metrics.write(
                "rollout_rejected",
                update=self.state.update,
                actor=rollout.actor_id,
                rollout_version=rollout.param_version,
                learner_version=self.state.param_version,
                **self.staleness.stats(),
            )
            return False

        with self.throughput.learning():
            metrics = dict(self.learner.update(rollout.data))
        self.state.update += 1
        self.state.total_env_steps += int(rollout.steps)
        self.state.total_decisions += int(rollout.steps) * max(1, int(rollout.parallel_envs))
        self.state.param_version = self.store.publish(self._policy_payload())
        # Publish the training clock with the parameters, not on some other
        # cadence: an actor that has just pulled version N should be collecting
        # under the schedule that belongs to version N.
        train_step = self._publish_train_step()
        self._last_update_at = time.monotonic()

        for ep in rollout.episodes:
            self.record_episode(ep)

        thr = self.throughput.record(rollout.steps, rollout.parallel_envs)
        gpu = self.gpu.sample()
        self.metrics.write(
            "update",
            update=self.state.update,
            actor=rollout.actor_id,
            steps=rollout.steps,
            parallel_envs=rollout.parallel_envs,
            # Two clocks, both in every row, because one of them is a lie by
            # any ordinary reading of its name: total_env_steps counts rollout
            # ROWS (it is what the zero-sum anneal is denominated in and so
            # cannot be redefined), total_decisions counts environment
            # decisions -- rows x parallel_envs, 8x larger in the standard
            # 4-instance mirror configuration.
            total_env_steps=self.state.total_env_steps,
            total_env_rows=self.state.total_env_steps,
            total_decisions=self.state.total_decisions,
            total_episodes=self.state.total_episodes,
            param_version=self.state.param_version,
            train_step=train_step,
            anneal_clock=self.cfg.anneal_clock,
            staleness=self.staleness.staleness(
                rollout.param_version, self.state.param_version - 1
            ),
            mixture=dict(rollout.mixture),
            mixture_drift=self.sampler.mixture_drift(),
            # Per-button action marginals, per update. See
            # Rollout.action_marginals: this is the curve that shows a BC prior
            # eroding, and it is only meaningful as a time series, so it goes
            # on every update row rather than into the eval section.
            **{f"actions/{k}": v for k, v in dict(rollout.action_marginals).items()},
            **{f"loss/{k}": v for k, v in metrics.items()},
            **{f"throughput/{k}": v for k, v in thr.items()},
            **{f"gpu/{k}": v for k, v in gpu.items()},
        )
        self._periodic()
        return True

    def require_anchor_eval(self) -> None:
        """Refuse to start a run whose eval cadence would measure nothing.

        ``eval_every`` fires the report either way, and in a symmetric mirror
        every number in that report except CS@10 is 0.5 by construction.  The
        first run logged 37 of those, each one saying ``(None, 0)`` against all
        four anchors, and nobody read them as "no evaluation is happening".
        Called from the entrypoint, not the constructor, so a unit test can
        still build a loop with no anchor machinery at all.
        """
        if not self.cfg.eval_every:
            return
        if self.anchor_eval is None:
            raise TrainingError(
                f"eval_every={self.cfg.eval_every} but this run has no anchor evaluator, so "
                f"every eval report would be win_rate_vs_anchor=(None, 0) forever -- which "
                f"is exactly how 16,200 updates of a non-learning policy went unnoticed. "
                f"Pass anchor_eval=..., or set --eval-every 0 to say deliberately that this "
                f"run is not evaluated."
            )

    def _run_anchor_eval(self, update: int) -> None:
        """Play the frozen anchors and feed the results to the evaluator."""
        if self.anchor_eval is None:
            return
        t0 = time.monotonic()
        episodes = self.anchor_eval(update, self.store.pull()[1])
        for ep in episodes:
            self.record_episode(ep)
        self.metrics.write(
            "anchor_eval",
            update=update,
            episodes=len(episodes),
            elapsed_s=time.monotonic() - t0,
            by_anchor={
                aid: sum(1 for e in episodes if e.opponent_id == aid)
                for aid in sorted({e.opponent_id for e in episodes})
            },
        )

    def _periodic(self) -> None:
        u = self.state.update
        ckpt: Optional[Path] = None
        if self.cfg.checkpoint_every and u % self.cfg.checkpoint_every == 0:
            payload = self.learner.state_payload()
            ckpt = self.checkpoints.save(u, payload)
        if self.cfg.snapshot_every and u % self.cfg.snapshot_every == 0:
            path = ckpt or self.checkpoints.save(u, self.learner.state_payload())
            snap = Snapshot(id=f"snap@{u}", step=u, path=str(path), created_s=time.time())
            evicted = self.sampler.pool.add(snap)
            self.metrics.write(
                "snapshot",
                update=u,
                id=snap.id,
                path=snap.path,
                pool_size=len(self.sampler.pool),
                evicted=[s.id for s in evicted],
            )
            ckpt = path
        if self.cfg.eval_every and u % self.cfg.eval_every == 0:
            self._run_anchor_eval(u)
            # Report against the agent id that HAS the episodes.
            #
            # agent_id() buckets by snapshot_every, and eval_every is a
            # multiple of it in both real launchers (--eval-every 400
            # --snapshot-every 200), so this line always fired on the FIRST
            # update of a brand-new bucket. The only episode in that bucket was
            # the anchor game _run_anchor_eval had recorded seconds earlier, so
            # every eval row read `cs_at_10: [0.0, 0.0, 1]` -- n=1 -- while the
            # 12-13 self-play episodes of the bucket that had just closed
            # averaged 34-38 and were never aggregated. The run's headline
            # absolute metric read 0.0 for its whole life.
            eval_id = self.agent_id()
            def _n(aid):
                st = self.evaluator.cs.stats(aid)
                return 0 if st is None else st[2]
            if _n(eval_id) <= 1 and self.cfg.snapshot_every:
                prev = self.agent_id(max(0, u - self.cfg.snapshot_every))
                if _n(prev) > _n(eval_id):
                    eval_id = prev
            report = self.evaluator.report(u, eval_id, self.sampler.pool.ids())
            fields = json.loads(report.to_json())
            fields.pop("kind", None)  # MetricsLog supplies it
            self.metrics.write("eval", **fields)
            log.info("EVAL %s", report.summary())
        if ckpt is not None:
            self.save_state(ckpt)

    # -- driving -----------------------------------------------------------

    def start_actors(self) -> None:
        if self.collect is None:
            raise TrainingError(
                "start_actors() needs a collect callable; construct TrainingLoop with "
                "collect=... or drive it synchronously with submit()/step_once()"
            )
        for i in range(self.cfg.num_actors):
            actor = ActorLoop(
                actor_id=i,
                store=self.store,
                out_queue=self.queue,
                collect=self.collect,
                error_queue=self.errors,
                stop_event=self.stop_event,
            )
            actor.start()
            self.actors.append(actor)
        log.info("started %d actors", len(self.actors))

    def run(self, max_updates: Optional[int] = None) -> RunState:
        """Run until ``max_updates`` more updates have completed."""
        target = self.state.update + (
            max_updates if max_updates is not None else self.cfg.total_updates
        )
        try:
            while self.state.update < target and not self.stop_event.is_set():
                self.step_once()
            # An actor sets stop_event as it dies, which would otherwise let the
            # loop exit *normally* and hide the death behind a short run.
            self._check_actors()
        except BaseException:
            self.stop_event.set()
            self.metrics.write(
                "run_failed",
                update=self.state.update,
                **self.staleness.stats(),
                **self.throughput.totals(),
            )
            raise
        finally:
            self.save_state()
        return self.state

    def shutdown(self, join_timeout_s: float = 30.0) -> None:
        self.stop_event.set()
        # Drain so a blocked put() can return and the thread can see the stop flag.
        for _ in range(self.queue.qsize()):
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        for actor in self.actors:
            if actor.thread is not None:
                actor.thread.join(timeout=join_timeout_s)
                if actor.thread.is_alive():
                    log.error(
                        "actor %d did not stop within %.0fs; it is probably blocked on a "
                        "server socket. The process will not exit cleanly.",
                        actor.actor_id,
                        join_timeout_s,
                    )
        self.metrics.write(
            "shutdown",
            update=self.state.update,
            **self.staleness.stats(),
            **self.throughput.totals(),
        )
        self.metrics.close()


def default_run_dir(name: str) -> Path:
    """``runs/<name>`` under the repo (or ``LANERL_RUNS_DIR``), created."""
    d = paths.runs_root() / name
    d.mkdir(parents=True, exist_ok=True)
    return d
