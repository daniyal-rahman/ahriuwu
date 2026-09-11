"""The same config across N seeds, and an honest interval over the results.

Every number this project has produced is n=1.  A single PPO run tells you
almost nothing: RL seed variance routinely exceeds the effect size of the
hyper-parameter change being tested, so "run B beat run A" from one seed each
is a coin flip dressed up as a result.

Two halves, deliberately separate:

``plan`` / ``launch``
    Expand one config into N runs named ``<base>-s<seed>``, each with its own
    ``--seed`` and its own run directory.  **Nothing is executed unless
    ``--execute`` is passed.**  Ports are strided per seed as well as per actor,
    because two concurrent runs sharing ``--port-base`` collide on the first
    instance and the second one dies during server start-up.

``aggregate``
    Mean and a 95% interval across seeds for the headline metrics, using the
    same :mod:`lanerl_train.runstats` definitions the pairwise comparison uses.

The interval is a **Student-t interval on the across-seed mean** by default
(``--ci t``), which assumes the per-seed metric is roughly normal -- reasonable
for a mean of many episodes, and it is the only thing worth doing at n=3.
``--ci bootstrap`` switches to a percentile bootstrap of the mean, which drops
the normality assumption but needs more seeds to say anything (at n=3 there are
only 10 distinct resamples, so its interval is quantised and usually narrower
than the truth -- the output says so).

Whichever is chosen, the output states n, and shouts when n is too small for
the interval to mean anything.  An interval that is quietly meaningless is
worse than no interval.
"""

from __future__ import annotations

import argparse
import logging
import math
import random
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import paths
from .runstats import HEADLINE_FIELDS, RunSummary, format_value, load_runs
from .slurm import SlurmJob, submit

__all__ = [
    "T95",
    "MIN_SEEDS_FOR_A_USABLE_CI",
    "SeedPlan",
    "Interval",
    "t_interval",
    "bootstrap_interval",
    "plan_seeds",
    "aggregate",
    "render_plan",
    "render_aggregate",
    "main",
]

log = logging.getLogger("lanerl_train.seeds")

#: Two-sided 95% Student-t critical values by degrees of freedom.  Tabulated
#: rather than pulled from scipy: this package has no scipy dependency and one
#: table of 30 constants is cheaper than acquiring one.
T95: Dict[int, float] = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
    8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145,
    15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060, 26: 2.056,
    27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}

#: Beyond the table, the normal quantile.  At df=31 the true value is 2.040, so
#: this understates the half-width by 4%; past df=60 the error is under 1%.
T95_LARGE = 1.960

#: Below this many seeds the interval is reported but flagged as unusable.  At
#: n=2 the t multiplier is 12.7, so the interval is wide enough to contain
#: essentially any hypothesis -- printing it without the flag invites someone
#: to read a 3-seed result as a measurement.
MIN_SEEDS_FOR_A_USABLE_CI = 3

#: Stride between consecutive seeds' port bases.  ``__main__`` already strides
#: 64 x envs_per_actor per actor; this must clear the widest plausible run.
PORT_STRIDE_PER_SEED = 2048


def t_critical(df: int) -> float:
    if df <= 0:
        raise ValueError(f"need df >= 1 for a t interval, got {df}")
    return T95.get(df, T95_LARGE)


@dataclass
class Interval:
    """A mean with an interval, and how believable it is."""

    n: int
    mean: Optional[float]
    lo: Optional[float]
    hi: Optional[float]
    method: str
    #: Set when ``n`` is too small for the interval to be evidence of anything.
    unusable: bool = False
    note: str = ""

    @property
    def half_width(self) -> Optional[float]:
        if self.lo is None or self.hi is None:
            return None
        return (self.hi - self.lo) / 2.0


def t_interval(values: Sequence[float]) -> Interval:
    """Mean +- t(.975, n-1) * s / sqrt(n)."""
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        return Interval(0, None, None, None, "t", True, "no seed reported this metric")
    mean = sum(xs) / n
    if n == 1:
        return Interval(
            1, mean, None, None, "t", True,
            "n=1: there is no interval, only a number",
        )
    var = sum((x - mean) ** 2 for x in xs) / (n - 1)
    half = t_critical(n - 1) * math.sqrt(var / n)
    return Interval(
        n, mean, mean - half, mean + half, f"t(df={n - 1})",
        unusable=n < MIN_SEEDS_FOR_A_USABLE_CI,
        note="" if n >= MIN_SEEDS_FOR_A_USABLE_CI
        else f"n={n}: t multiplier is {t_critical(n - 1):.1f}",
    )


def bootstrap_interval(
    values: Sequence[float], resamples: int = 10000, seed: int = 0
) -> Interval:
    """Percentile bootstrap of the mean.

    Seeded so the same inputs give the same interval; an interval that moves
    between two invocations of the same command is not reportable.
    """
    xs = [float(v) for v in values]
    n = len(xs)
    if n == 0:
        return Interval(0, None, None, None, "bootstrap", True, "no seed reported this metric")
    mean = sum(xs) / n
    if n == 1:
        return Interval(
            1, mean, None, None, "bootstrap", True,
            "n=1: every resample is the same point",
        )
    rng = random.Random(seed)
    means = sorted(
        sum(xs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples)
    )
    lo = means[int(0.025 * (resamples - 1))]
    hi = means[int(math.ceil(0.975 * (resamples - 1)))]
    distinct = math.comb(2 * n - 1, n)  # multisets of size n from n values
    note = ""
    unusable = n < MIN_SEEDS_FOR_A_USABLE_CI
    if distinct < 100:
        unusable = True
        note = (
            f"only {distinct} distinct resamples exist at n={n}; the percentile "
            f"bootstrap is quantised and biased narrow here -- prefer --ci t"
        )
    return Interval(n, mean, lo, hi, f"bootstrap({resamples})", unusable, note)


# --------------------------------------------------------------------------
# Launching
# --------------------------------------------------------------------------


@dataclass
class SeedPlan:
    """One seed's run: its name, its directory, and the exact argv."""

    seed: int
    run_name: str
    run_dir: Path
    argv: List[str]

    def shell(self, python: str = sys.executable) -> str:
        return " ".join(shlex.quote(a) for a in [python, *self.argv])


def plan_seeds(
    base_name: str,
    seeds: Sequence[int],
    extra_args: Sequence[str] = (),
    port_base: int = 21000,
    runs_root: Optional[Path] = None,
) -> List[SeedPlan]:
    """Expand one config into one :class:`SeedPlan` per seed.

    ``extra_args`` is passed through to ``python -m lanerl_train`` verbatim, so
    a sweep is described in exactly the flags a single run is described in.
    ``--run-name``, ``--seed`` and ``--port-base`` are supplied here and are
    rejected in ``extra_args``: silently overriding the caller's ``--seed`` with
    a generated one, or letting theirs win and running the same seed N times, is
    precisely the kind of "green but vacuous" result this tooling exists to stop.
    """
    if not seeds:
        raise ValueError("need at least one seed")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"duplicate seeds in {list(seeds)}; N runs of one seed is still n=1")
    owned = {"--run-name", "--seed", "--port-base"}
    clash = sorted(owned & {a.split("=")[0] for a in extra_args})
    if clash:
        raise ValueError(
            f"{clash} is set per seed by this launcher; passing it in extra args would "
            f"either be ignored or collapse the sweep onto one seed/port/dir"
        )
    root = Path(runs_root) if runs_root is not None else paths.runs_root()
    plans: List[SeedPlan] = []
    for i, seed in enumerate(seeds):
        name = f"{base_name}-s{seed}"
        argv = [
            "-m", "lanerl_train",
            "--run-name", name,
            "--seed", str(seed),
            "--port-base", str(port_base + i * PORT_STRIDE_PER_SEED),
            *extra_args,
        ]
        plans.append(SeedPlan(seed=seed, run_name=name, run_dir=root / name, argv=argv))
    return plans


def render_plan(plans: Sequence[SeedPlan], python: str = sys.executable) -> str:
    lines = [f"{len(plans)} seed run(s) planned:", ""]
    for p in plans:
        lines.append(f"  seed {p.seed}: {p.run_dir}")
        lines.append(f"    {p.shell(python)}")
    lines.append("")
    lines.append("Nothing has been launched. Add --execute to run them.")
    return "\n".join(lines) + "\n"


def launch(
    plans: Sequence[SeedPlan],
    execute: bool = False,
    node: Optional[str] = None,
    partition: str = "cpu",
    cpus_per_task: int = 16,
    time_limit: Optional[str] = None,
    python: Optional[str] = None,
    log_dir: Optional[Path] = None,
) -> List[str]:
    """Start the planned runs.  With ``execute=False`` nothing is spawned.

    ``node`` selects Slurm submission (via :mod:`lanerl_train.slurm`, so log
    paths are translated for the target node); without it the runs are started
    as local subprocesses, which is only sane for one or two seeds on a machine
    with cores to spare.
    """
    py = python or sys.executable
    handles: List[str] = []
    for p in plans:
        if node:
            job = SlurmJob(
                name=p.run_name,
                node=node,
                command=p.argv,
                log_dir=Path(log_dir) if log_dir else paths.repo_root() / "lanerl/logs",
                partition=partition,
                cpus_per_task=cpus_per_task,
                time_limit=time_limit,
                python=py,
            )
            handles.append(submit(job, dry_run=not execute))
        elif execute:
            proc = subprocess.Popen([py, *p.argv], cwd=str(paths.repo_root()))
            handles.append(f"pid {proc.pid}")
        else:
            handles.append(p.shell(py))
    return handles


# --------------------------------------------------------------------------
# Aggregating
# --------------------------------------------------------------------------


def aggregate(
    summaries: Sequence[RunSummary], method: str = "t", bootstrap_seed: int = 0
) -> Dict[str, Interval]:
    """One :class:`Interval` per headline metric, across seeds."""
    if method not in ("t", "bootstrap"):
        raise ValueError(f"method must be 't' or 'bootstrap', got {method!r}")
    out: Dict[str, Interval] = {}
    for attr, _label, _spec in HEADLINE_FIELDS:
        values = [
            float(s.get(attr)) for s in summaries
            if isinstance(s.get(attr), (int, float)) and not isinstance(s.get(attr), bool)
        ]
        out[attr] = (
            t_interval(values) if method == "t"
            else bootstrap_interval(values, seed=bootstrap_seed)
        )
    return out


def _config_disagreements(summaries: Sequence[RunSummary]) -> List[str]:
    """Config keys that differ across seeds, other than the ones that must.

    A "seed sweep" whose members differ in ``lr`` is not a seed sweep, and the
    aggregate mean of it is meaningless.  Caught here rather than trusted.
    """
    expected = {"args.seed", "args.run_name", "args.port_base", "run_config.seed",
                "run_config.run_dir"}
    keys = sorted({k for s in summaries for k in s.config})
    bad = []
    for k in keys:
        if k in expected:
            continue
        vals = {json_safe(s.config.get(k, "<absent>")) for s in summaries}
        if len(vals) > 1:
            bad.append(k)
    return bad


def json_safe(v: Any) -> Any:
    return v if isinstance(v, (str, int, float, bool, type(None))) else str(v)


def render_aggregate(
    summaries: Sequence[RunSummary], intervals: Dict[str, Interval], method: str
) -> str:
    n = len(summaries)
    lines = ["=" * 78, f"SEED AGGREGATE over {n} run(s)", "=" * 78]
    for s in summaries:
        lines.append(f"  {s.name}  ({s.path})")
    lines.append("")

    if n < MIN_SEEDS_FOR_A_USABLE_CI:
        lines.append("!" * 78)
        lines.append(
            f"!! n={n}. This is NOT a measurement. A 95% t interval at n={n} has a "
            f"multiplier of\n!! {t_critical(max(n - 1, 1)):.1f}; it will contain almost "
            f"any hypothesis you care to test. Run at least\n!! {MIN_SEEDS_FOR_A_USABLE_CI} "
            f"seeds before reporting a difference."
        )
        lines.append("!" * 78)
        lines.append("")

    bad = _config_disagreements(summaries)
    if bad:
        lines.append("!" * 78)
        lines.append(
            "!! These runs do NOT share a config, so this is not a seed sweep and the\n"
            "!! aggregate below is meaningless. Differing keys:"
        )
        for k in bad:
            lines.append(f"!!   {k}: " + ", ".join(
                f"{s.name}={json_safe(s.config.get(k, '<absent>'))}" for s in summaries
            ))
        lines.append("!" * 78)
        lines.append("")

    header = ["metric", "mean", "95% CI", "n", "per-seed"]
    rows = []
    for attr, label, spec in HEADLINE_FIELDS:
        iv = intervals[attr]
        mean = format_value(iv.mean, spec)
        if iv.lo is None or iv.hi is None:
            ci = "-- " + (iv.note or "no interval")
        else:
            ci = f"[{format_value(iv.lo, spec)}, {format_value(iv.hi, spec)}]"
            if iv.unusable:
                ci += "  (UNUSABLE"+ (f": {iv.note}" if iv.note else "") + ")"
        per = " ".join(format_value(s.get(attr), spec) for s in summaries)
        rows.append([label, mean, ci, str(iv.n), per])

    widths = [max(len(h), *(len(r[i]) for r in rows)) for i, h in enumerate(header)]
    lines.append("  " + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header)))
    lines.append("  " + "  ".join("-" * widths[i] for i in range(len(header))))
    for r in rows:
        lines.append("  " + "  ".join(r[i].ljust(widths[i]) for i in range(len(header))))
    lines.append("")
    lines.append(f"  interval method: {method}")
    lines.append(
        "  'n' counts the seeds that actually reported the metric, which can be\n"
        "  fewer than the runs given -- a metric no seed recorded shows n=0."
    )
    notes = [(s.name, note) for s in summaries for note in s.notes]
    if notes:
        lines.append("")
        lines.append("NOTES")
        for name, note in notes:
            lines.append(f"  [{name}] {note}")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m lanerl_train.seeds",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pl = sub.add_parser("plan", help="print the per-seed commands; launch nothing")
    la = sub.add_parser("launch", help="same, and start them with --execute")
    for q in (pl, la):
        q.add_argument("--run-name", required=True, help="base name; runs become <base>-s<seed>")
        q.add_argument(
            "--seeds", type=int, nargs="+", required=True,
            help=f"seed values; fewer than {MIN_SEEDS_FOR_A_USABLE_CI} gives no usable CI",
        )
        q.add_argument("--port-base", type=int, default=21000)
        q.add_argument("--python", default=sys.executable)
        q.add_argument(
            "extra", nargs=argparse.REMAINDER,
            help="args after -- are passed to `python -m lanerl_train` verbatim",
        )
    la.add_argument("--execute", action="store_true", help="actually start the runs")
    la.add_argument("--node", default=None, help="submit to this Slurm node instead of running locally")
    la.add_argument("--partition", default="cpu")
    la.add_argument("--cpus-per-task", type=int, default=16)
    la.add_argument("--time-limit", default=None)

    ag = sub.add_parser("aggregate", help="mean +- 95% CI across seed runs")
    ag.add_argument("runs", nargs="+", type=Path)
    ag.add_argument("--ci", choices=("t", "bootstrap"), default="t")
    ag.add_argument("--bootstrap-seed", type=int, default=0)
    return p


def _strip_sep(extra: Sequence[str]) -> List[str]:
    args = list(extra)
    return args[1:] if args and args[0] == "--" else args


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = build_argparser().parse_args(argv)

    if args.cmd in ("plan", "launch"):
        plans = plan_seeds(
            args.run_name, args.seeds, _strip_sep(args.extra), port_base=args.port_base
        )
        if len(args.seeds) < MIN_SEEDS_FOR_A_USABLE_CI:
            log.warning(
                "%d seed(s) requested; %d is the minimum that yields a usable 95%% "
                "interval, and every number this project has reported so far was n=1.",
                len(args.seeds),
                MIN_SEEDS_FOR_A_USABLE_CI,
            )
        if args.cmd == "plan":
            sys.stdout.write(render_plan(plans, args.python))
            return 0
        handles = launch(
            plans,
            execute=args.execute,
            node=args.node,
            partition=args.partition,
            cpus_per_task=args.cpus_per_task,
            time_limit=args.time_limit,
            python=args.python,
        )
        for p, h in zip(plans, handles):
            print(f"{p.run_name}: {h}")
        if not args.execute:
            print("\nNothing was launched (no --execute).")
        return 0

    summaries = load_runs(args.runs)
    intervals = aggregate(summaries, method=args.ci, bootstrap_seed=args.bootstrap_seed)
    sys.stdout.write(render_aggregate(summaries, intervals, args.ci))
    return 0


if __name__ == "__main__":
    sys.exit(main())
