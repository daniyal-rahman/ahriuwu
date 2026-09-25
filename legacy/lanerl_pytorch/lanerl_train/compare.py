"""``python -m lanerl_train.compare runs/a runs/b`` -- what was different?

Two runs of this stack differ in a resolved config of ~60 leaves and in a
metrics file of ~17,000 rows, and until now the only way to ask "what changed"
was to diff two JSON files by eye and then squint at a plot.  That is why the
first run's ``max_staleness=4`` (against the documented target of 0-1) and its
``eval_every`` that evaluated nothing both survived thirteen thousand updates.

Output is plain text, fixed-width, and deliberately narrow enough to read in a
terminal.  Two sections:

1. **config** -- only the keys that DIFFER, plus any key one run has and
   another does not (which usually means the runs came from different code, a
   bigger finding than a turned knob).
2. **headline metrics** -- one row per metric, one column per run, from
   :mod:`lanerl_train.runstats` so this and the multi-seed aggregator can never
   disagree about what "final ep_return" means.

``--all-config`` prints the identical keys too, for when the question is "what
was this run actually configured with".
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Sequence

from .runstats import (
    HEADLINE_FIELDS,
    RunSummary,
    config_differences,
    format_value,
    load_runs,
)

__all__ = ["render", "main"]

log = logging.getLogger("lanerl_train.compare")

#: Long values (a run_dir, a serialised league config) are truncated to this so
#: the table stays readable; the full value is always in resolved_config.json.
MAX_CELL = 28


def _cell(value: object, width: int) -> str:
    text = "-" if value is None else str(value)
    if len(text) > MAX_CELL:
        text = text[: MAX_CELL - 1] + "…"
    return text.ljust(width)


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], indent: str = "  ") -> List[str]:
    cols = len(header)
    widths = [len(h) for h in header]
    for r in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(r[i]))
    out = [indent + "  ".join(h.ljust(widths[i]) for i, h in enumerate(header))]
    out.append(indent + "  ".join("-" * widths[i] for i in range(cols)))
    for r in rows:
        out.append(indent + "  ".join(r[i].ljust(widths[i]) for i in range(cols)))
    return out


def render(summaries: Sequence[RunSummary], all_config: bool = False) -> str:
    """The whole report, as one string."""
    if not summaries:
        return "no runs given\n"
    names = [s.name for s in summaries]
    lines: List[str] = []

    lines.append("=" * 78)
    lines.append("RUNS")
    lines.append("=" * 78)
    for s in summaries:
        lines.append(f"  {s.name}: {s.path}")
    lines.append("")

    # -- config ---------------------------------------------------------
    differing, missing = config_differences(summaries)
    keys = sorted(set(differing) | set(missing))
    if all_config:
        keys = sorted({k for s in summaries for k in s.config})
    lines.append("=" * 78)
    if all_config:
        lines.append(f"CONFIG (all {len(keys)} keys; * marks a difference)")
    else:
        lines.append(
            f"CONFIG DIFFERENCES ({len(differing)} differing, "
            f"{len(missing)} present in some runs only)"
        )
    lines.append("=" * 78)
    if not keys:
        lines.append("  the resolved configs are identical")
    else:
        rows = []
        for k in keys:
            marker = "*" if k in differing or k in missing else " "
            row = [marker + k]
            for s in summaries:
                row.append(_cell(s.config.get(k, "<absent>"), 0))
            rows.append(row)
        lines.extend(_table(["key"] + names, rows))
    lines.append("")

    # -- headline metrics ------------------------------------------------
    lines.append("=" * 78)
    lines.append("HEADLINE METRICS")
    lines.append("=" * 78)
    rows = []
    for attr, label, spec in HEADLINE_FIELDS:
        row = [label]
        for s in summaries:
            row.append(format_value(s.get(attr), spec))
        rows.append(row)
    lines.extend(_table(["metric"] + names, rows))
    lines.append("")
    lines.append(
        "  ep_return/CS@10 counts (n) below; 'n/a' means the field was never "
        "recorded,\n  which is NOT the same as zero."
    )
    counts = [
        ["ep_return n"] + [str(s.ep_return_n) for s in summaries],
        ["CS@10 n"] + [str(s.cs_at_10_n) for s in summaries],
        ["anchor games"] + [str(s.anchor_games) for s in summaries],
    ]
    lines.extend(_table(["count"] + names, counts))
    lines.append("")

    # -- notes -------------------------------------------------------------
    any_notes = any(s.notes for s in summaries)
    lines.append("=" * 78)
    lines.append("NOTES" if any_notes else "NOTES (none)")
    lines.append("=" * 78)
    for s in summaries:
        for note in s.notes:
            lines.append(f"  [{s.name}] {note}")
    return "\n".join(lines) + "\n"


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m lanerl_train.compare",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("runs", nargs="+", type=Path, help="run directories (runs/<name>)")
    p.add_argument(
        "--all-config",
        action="store_true",
        help="print every resolved-config key, not only the differing ones",
    )
    return p


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    args = build_argparser().parse_args(argv)
    summaries = load_runs(args.runs)
    sys.stdout.write(render(summaries, all_config=args.all_config))
    return 0


if __name__ == "__main__":
    sys.exit(main())
