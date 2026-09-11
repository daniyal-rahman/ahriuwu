"""Read one run directory down to the handful of numbers worth comparing.

Shared by :mod:`lanerl_train.compare` (run A vs run B) and
:mod:`lanerl_train.seeds` (the same config across N seeds), so both tools can
never disagree about what "final ep_return" means -- two definitions of the
headline metric is how you get two teams arguing about which run was better.

Every derived number here is either read straight out of ``metrics.jsonl`` or
stated with the window it was computed over.  Two rules follow from the first
run's post-mortem:

**Absent is not zero.**  ``ep_return`` did not exist for the whole of the first
run, and ``cs_at_10`` is ``None`` for every episode that ended before ten
minutes.  Both come back as ``None`` with an ``n`` of 0, never as 0.0.

**A resume is a seam, not a gap.**  Wall-clock span is summed over contiguous
segments: the first run has eight of them, and the naive ``last.wall -
first.wall`` counts an hour of queued Slurm time as training time, overstating
the span by 7% and understating throughput by the same.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

__all__ = [
    "HEADLINE_FIELDS",
    "TAIL_WINDOW",
    "RESUME_GAP_S",
    "RunSummary",
    "flatten_config",
    "config_differences",
    "load_run",
    "load_runs",
]

log = logging.getLogger("lanerl_train.runstats")

#: Episodes averaged for the "final" figure of a per-episode metric.  A single
#: last episode is noise: the first run's per-episode return varies by more
#: between neighbouring episodes than across the whole run.
TAIL_WINDOW = 50

#: A gap longer than this between consecutive ``update`` rows is a resume (a
#: requeue, a crash loop, a node change), not training time.
RESUME_GAP_S = 300.0

#: The numbers a comparison prints, in the order it prints them.  ``(attribute,
#: label, format)``; ``None`` formats as "n/a" whatever the format says.
HEADLINE_FIELDS: Tuple[Tuple[str, str, str], ...] = (
    ("updates", "updates", "d"),
    ("episodes", "episodes", "d"),
    ("wall_h", "wall clock (h)", ".2f"),
    ("updates_per_s", "updates/s", ".3f"),
    ("decisions_per_s", "decisions/s", ".1f"),
    ("learner_frac", "learner frac", ".3f"),
    ("gpu_util_pct", "gpu util %", ".1f"),
    ("ep_return_final", f"ep_return (last {TAIL_WINDOW})", ".4f"),
    ("ep_return_best", f"ep_return best {TAIL_WINDOW}-mean", ".4f"),
    ("cs_at_10_mean", "CS@10 mean", ".1f"),
    ("entropy_first", "entropy (first 500)", ".3f"),
    ("entropy_last", "entropy (last 500)", ".3f"),
    ("entropy_slope_per_1k", "entropy slope /1k upd", "+.4f"),
    ("value_loss_last", "value_loss (last 500)", ".5f"),
    ("approx_kl_last", "approx_kl (last 500)", ".4f"),
    ("clip_frac_last", "clip_frac (last 500)", ".3f"),
    ("epochs_run_last", "epochs_run (last 500)", ".2f"),
    ("anchor_win_rate", "win rate vs anchors", ".3f"),
    ("staleness_reject_rate", "rollouts rejected", ".3f"),
)


@dataclass
class RunSummary:
    """One run, reduced to what a comparison needs."""

    name: str
    path: Path
    config: Dict[str, Any] = field(default_factory=dict)

    updates: Optional[int] = None
    episodes: Optional[int] = None
    wall_h: Optional[float] = None
    updates_per_s: Optional[float] = None
    env_steps_per_s: Optional[float] = None
    decisions_per_s: Optional[float] = None
    learner_frac: Optional[float] = None
    gpu_util_pct: Optional[float] = None
    gpu_mem_mb: Optional[float] = None

    ep_return_final: Optional[float] = None
    ep_return_best: Optional[float] = None
    ep_return_n: int = 0
    cs_at_10_mean: Optional[float] = None
    cs_at_10_n: int = 0

    entropy_first: Optional[float] = None
    entropy_last: Optional[float] = None
    entropy_slope_per_1k: Optional[float] = None
    value_loss_last: Optional[float] = None
    approx_kl_last: Optional[float] = None
    clip_frac_last: Optional[float] = None
    epochs_run_last: Optional[float] = None

    anchor_win_rate: Optional[float] = None
    anchor_games: int = 0
    staleness_reject_rate: Optional[float] = None

    #: Anything the reader had to infer, guess around, or could not find.  These
    #: are printed: a comparison with an unstated fallback in it is a trap.
    notes: List[str] = field(default_factory=list)

    def get(self, attr: str) -> Any:
        return getattr(self, attr, None)


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def flatten_config(obj: Any, prefix: str = "") -> Dict[str, Any]:
    """``{"ppo_config.lr": 0.0003, ...}`` from nested ``resolved_config.json``.

    Flattened rather than compared tree-wise so a diff names the exact leaf that
    changed -- "ppo_config differs" is not an answer to "what was different".
    """
    out: Dict[str, Any] = {}
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            out.update(flatten_config(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, (list, tuple)):
        out[prefix] = json.dumps(list(obj), default=str)
    else:
        out[prefix] = obj
    return out


def config_differences(summaries: Sequence[RunSummary]) -> Tuple[List[str], List[str]]:
    """``(keys that differ, keys present in some runs but not others)``.

    A key missing from one run is called out separately from a key with a
    different value: "run B has no ``ppo_config.target_kl``" usually means the
    two runs were produced by different versions of the code, which is a much
    bigger finding than a knob being turned.
    """
    if not summaries:
        return [], []
    all_keys = sorted({k for s in summaries for k in s.config})
    differing: List[str] = []
    missing: List[str] = []
    for k in all_keys:
        present = [s for s in summaries if k in s.config]
        if len(present) != len(summaries):
            missing.append(k)
            continue
        first = present[0].config[k]
        if any(s.config[k] != first for s in present[1:]):
            differing.append(k)
    return differing, missing


# --------------------------------------------------------------------------
# Small statistics, kept local so this module has no scipy dependency
# --------------------------------------------------------------------------


def _mean(xs: Sequence[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def _slope_per_1k(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    """Least-squares slope of ``y`` against ``x``, scaled to 1000 x-units."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 0.0:
        return None
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return 1000.0 * sxy / sxx


def _contiguous_wall_s(walls: Sequence[float], gap_s: float = RESUME_GAP_S) -> float:
    """Total wall time, with resume gaps removed."""
    total = 0.0
    for a, b in zip(walls, walls[1:]):
        d = b - a
        if 0.0 <= d <= gap_s:
            total += d
    return total


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open() as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except ValueError as exc:
                # Deliberately not fatal, unlike Evaluator.load_jsonl: a run
                # killed mid-write leaves a torn last line, and refusing to
                # summarise the whole 10 MB because of it is the wrong trade
                # for a comparison tool. Counted and reported, never ignored.
                log.warning("%s:%d is not JSON (%s); skipping", path, lineno, exc)
                yield {"kind": "_unparseable"}


def load_run(path: Path, tail_window: int = TAIL_WINDOW) -> RunSummary:
    """Summarise one ``runs/<name>`` directory.

    Tolerates a missing ``resolved_config.json`` (runs predating it) and a
    missing ``metrics.jsonl`` (a run that died before its first update), because
    those are exactly the runs worth looking at.
    """
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"{path} is not a run directory")
    s = RunSummary(name=path.name, path=path)

    cfg_path = path / "resolved_config.json"
    if cfg_path.exists():
        s.config = flatten_config(json.loads(cfg_path.read_text()))
    else:
        s.notes.append("no resolved_config.json; config columns are empty")

    metrics_path = path / "metrics.jsonl"
    if not metrics_path.exists():
        s.notes.append("no metrics.jsonl; the run produced no updates")
        return s

    updates: List[Dict[str, Any]] = []
    returns: List[float] = []
    cs: List[float] = []
    n_episodes = 0
    torn = 0
    anchor_score = 0.0
    anchor_games = 0
    last_reject_rate: Optional[float] = None
    for rec in _iter_records(metrics_path):
        kind = rec.get("kind")
        if kind == "update":
            updates.append(rec)
        elif kind == "episode":
            n_episodes += 1
            if rec.get("ep_return") is not None:
                returns.append(float(rec["ep_return"]))
            if rec.get("cs_at_10") is not None:
                cs.append(float(rec["cs_at_10"]))
            cat = rec.get("opponent_category")
            if cat == "anchor" and rec.get("score") is not None:
                anchor_score += float(rec["score"])
                anchor_games += 1
        elif kind in ("rollout_rejected", "shutdown", "run_failed"):
            if rec.get("reject_rate") is not None:
                last_reject_rate = float(rec["reject_rate"])
        elif kind == "_unparseable":
            torn += 1
    if torn:
        s.notes.append(f"{torn} unparseable line(s) skipped")

    s.episodes = n_episodes
    s.staleness_reject_rate = last_reject_rate
    if anchor_games:
        s.anchor_win_rate = anchor_score / anchor_games
        s.anchor_games = anchor_games
    else:
        s.notes.append(
            "no episodes against a frozen anchor: every score in this run is a "
            "symmetric self-play 0.5 and measures nothing"
        )

    if not updates:
        s.notes.append("metrics.jsonl has no 'update' rows")
        return s

    s.updates = len(updates)

    # -- wall clock and throughput ---------------------------------------
    walls = [float(r["wall"]) for r in updates if "wall" in r]
    wall_s = _contiguous_wall_s(walls)
    n_seams = sum(1 for a, b in zip(walls, walls[1:]) if (b - a) > RESUME_GAP_S)
    if n_seams:
        s.notes.append(
            f"{n_seams} resume seam(s) excluded from wall clock "
            f"(raw span {(walls[-1] - walls[0]) / 3600.0:.2f} h)"
        )
    s.wall_h = wall_s / 3600.0 if wall_s > 0 else None

    tail = updates[-500:]
    have_throughput = any("throughput/updates_per_s" in r for r in tail)
    if have_throughput:
        s.updates_per_s = _mean([r["throughput/updates_per_s"] for r in tail
                                 if r.get("throughput/updates_per_s") is not None])
        s.env_steps_per_s = _mean([r["throughput/env_steps_per_s"] for r in tail
                                   if r.get("throughput/env_steps_per_s") is not None])
        s.decisions_per_s = _mean([r["throughput/decisions_per_s"] for r in tail
                                   if r.get("throughput/decisions_per_s") is not None])
        s.learner_frac = _mean([r["throughput/learner_frac"] for r in tail
                                if r.get("throughput/learner_frac") is not None])
    elif wall_s > 0:
        # Runs predating the telemetry: derive what can be derived, and say so,
        # rather than printing a blank where a comparison needs a number.
        steps = sum(int(r.get("steps", 0)) for r in updates)
        s.updates_per_s = len(updates) / wall_s
        s.env_steps_per_s = steps / wall_s
        par = updates[-1].get("parallel_envs")
        if par:
            s.decisions_per_s = steps * int(par) / wall_s
        s.notes.append(
            "throughput derived from 'wall' timestamps (this run predates "
            "throughput/* telemetry); learner_frac is unavailable"
        )

    gpu_util = [r["gpu/util_pct"] for r in tail if r.get("gpu/util_pct") is not None]
    s.gpu_util_pct = _mean(gpu_util)
    gpu_mem = [r["gpu/mem_allocated_mb"] for r in tail
               if r.get("gpu/mem_allocated_mb") is not None]
    s.gpu_mem_mb = _mean(gpu_mem)

    # -- learning signal ---------------------------------------------------
    s.ep_return_n = len(returns)
    if returns:
        s.ep_return_final = _mean(returns[-tail_window:])
        best = None
        for i in range(len(returns)):
            w = returns[max(0, i - tail_window + 1) : i + 1]
            m = _mean(w)
            if m is not None and (best is None or m > best):
                best = m
        s.ep_return_best = best
    else:
        s.notes.append("no episode carried ep_return (the field postdates this run)")

    s.cs_at_10_n = len(cs)
    s.cs_at_10_mean = _mean(cs)

    def loss_tail(key: str, n: int = 500) -> Optional[float]:
        vals = [r[key] for r in updates[-n:] if r.get(key) is not None]
        return _mean(vals)

    head_ent = [r["loss/entropy"] for r in updates[:500] if r.get("loss/entropy") is not None]
    s.entropy_first = _mean(head_ent)
    s.entropy_last = loss_tail("loss/entropy")
    ent_x = [float(r["update"]) for r in updates if r.get("loss/entropy") is not None]
    ent_y = [float(r["loss/entropy"]) for r in updates if r.get("loss/entropy") is not None]
    s.entropy_slope_per_1k = _slope_per_1k(ent_x, ent_y)
    s.value_loss_last = loss_tail("loss/value_loss")
    s.approx_kl_last = loss_tail("loss/approx_kl")
    s.clip_frac_last = loss_tail("loss/clip_frac")
    s.epochs_run_last = loss_tail("loss/epochs_run")
    return s


def load_runs(paths_: Sequence[Path], tail_window: int = TAIL_WINDOW) -> List[RunSummary]:
    return [load_run(Path(p), tail_window) for p in paths_]


def format_value(value: Any, spec: str) -> str:
    """Format for the comparison table.  ``None`` is always "n/a"."""
    if value is None:
        return "n/a"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return "n/a"
    # A count field averaged across seeds is no longer an integer; falling back
    # to repr() there prints 7687.499999999999 into a comparison table.
    if spec.endswith("d") and isinstance(value, float):
        spec = ".1f"
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)
