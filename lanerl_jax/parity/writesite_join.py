"""
Join tier-1 residual rows against the server's own write-site stream.

WHY
---
A tier-1 residual says a unit's `target` / `move_order` / `waypoints`
disagreed. It cannot say which branch of the server produced the server's
value, so every attribution made from a residual alone has been an argument
from signature -- and four such arguments have been REFUTED in this project
(`CFH-002`, `GIVE-001`, `TGT-STALE`, `AA-005`), each by the first direct
measurement.

Each of the three fields has exactly one write site, and those sites now
carry `caller=<member>@<line>` (see `lanerl/patch_writesites.py`). So for a
disagreeing unit-tick the server can be asked directly: which C# line set
your value? This joins the two streams and answers that as a count.

THE DENOMINATOR IS THE POINT
----------------------------
A caller histogram over disagreeing rows alone is unreadable: the most
common caller in the residual is usually just the most common caller
overall. This project has already published one finding that dissolved on
contact with its base rate (`<no-event>` at "7 of 9", against a 93.3% base
rate). So every caller is reported as share-of-residual, share-of-corpus,
and the ratio; only the ratio is evidence.

JOIN WINDOW
-----------
Rows are keyed at the INJECTION tick N. The server's write lands during
tick N+1, so the window is [t, t + window_ms] with window_ms one decision
step (33.3 ms at 30 Hz off a 60 Hz tick) by default. Joining on tick N
alone was tried and missed essentially every event -- a 16.67 ms stride
against a 33.3 ms key.

USAGE
    python -m lanerl_jax.parity.writesite_join --rows <rows.jsonl> \
        --log <server.log> [--window-ms 34]
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path

# The write-site stream. `rest` is parsed lazily: most lines are not joined.
_LINE = re.compile(
    r"LANERL_DECISION t=(?P<t>-?\d+) k=(?P<k>\w+) id=(?P<id>\d+)(?P<rest>.*)$")

# Which decision kind carries the answer for which residual family.
_FAMILY_KIND = {
    "target": ("SetTargetUnit",),
    "move_order": ("UpdateMoveOrder", "UpdateMoveOrderVetoed"),
    "waypoints": ("SetWaypoints", "SetWaypointsRejected"),
}

_NO_EVENT = "<no-write>"


def _fields(rest: str) -> dict:
    out = {}
    for tok in rest.split():
        k, _, v = tok.partition("=")
        if v:
            out[k] = v
    return out


def index_writes(log: Path, kinds: set[str]):
    """net_id -> sorted list of (t_ms, kind, fields). Streaming: the corpus log
    is ~700 MB and materialising every event costs more than the join saves."""
    by_unit: dict[int, list] = collections.defaultdict(list)
    totals = collections.Counter()
    with log.open(errors="replace") as fh:
        for line in fh:
            if "LANERL_DECISION" not in line:
                continue
            m = _LINE.search(line)
            if m is None or m.group("k") not in kinds:
                continue
            f = _fields(m.group("rest"))
            by_unit[int(m.group("id"))].append(
                (int(m.group("t")), m.group("k"), f))
            totals[(m.group("k"), f.get("caller", "?"))] += 1
    for v in by_unit.values():
        v.sort(key=lambda r: r[0])
    return by_unit, totals


def _lookup(by_unit, net, t0, t1, kinds):
    hits = []
    for t, kind, f in by_unit.get(net, ()):
        if t < t0:
            continue
        if t > t1:
            break
        if kind in kinds:
            hits.append((t, kind, f))
    return hits


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--log", type=Path, required=True)
    ap.add_argument("--window-ms", type=int, default=34)
    ap.add_argument("--examples", type=int, default=8)
    a = ap.parse_args(argv)

    rows = [json.loads(ln) for ln in a.rows.read_text().splitlines() if ln.strip()]
    fams = collections.Counter(r["family"] for r in rows)
    print(f"{len(rows)} residual rows: "
          + ", ".join(f"{k}={v}" for k, v in fams.most_common()))

    kinds = {k for ks in _FAMILY_KIND.values() for k in ks}
    by_unit, totals = index_writes(a.log, kinds)
    print(f"indexed {sum(len(v) for v in by_unit.values())} write events "
          f"over {len(by_unit)} units\n")

    for fam, fam_kinds in _FAMILY_KIND.items():
        sub = [r for r in rows if r["family"] == fam]
        if not sub:
            continue
        kset = set(fam_kinds)
        corpus_total = sum(n for (k, _c), n in totals.items() if k in kset)
        corpus = collections.Counter()
        for (k, c), n in totals.items():
            if k in kset:
                corpus[f"{k}:{c}"] += n

        hist = collections.Counter()
        examples = collections.defaultdict(list)
        for r in sub:
            net = r.get("net") or r.get("net_id")
            t = r.get("t_ms")
            if net is None or t is None:
                hist["<unkeyed>"] += 1
                continue
            hits = _lookup(by_unit, int(net), int(t), int(t) + a.window_ms, kset)
            if not hits:
                hist[_NO_EVENT] += 1
                continue
            # The LAST write in the window is the one the post-tick dump shows.
            t_h, k_h, f_h = hits[-1]
            key = f"{k_h}:{f_h.get('caller', '?')}"
            hist[key] += 1
            if len(examples[key]) < a.examples:
                examples[key].append((t, net, r, f_h))

        print(f"== {fam}: {len(sub)} disagreeing rows ==")
        print(f"   {'write site':<44} {'resid':>6} {'%res':>6} "
              f"{'%corpus':>8} {'ratio':>7}")
        for key, n in hist.most_common():
            pres = 100 * n / len(sub)
            if key in (_NO_EVENT, "<unkeyed>"):
                print(f"   {key:<44} {n:>6} {pres:>5.1f}% "
                      f"{'-':>8} {'-':>7}")
                continue
            pcor = 100 * corpus.get(key, 0) / max(1, corpus_total)
            ratio = (pres / pcor) if pcor else float("inf")
            print(f"   {key:<44} {n:>6} {pres:>5.1f}% "
                  f"{pcor:>7.2f}% {ratio:>6.1f}x")
        print()

    print("ratio = share of the residual / share of the corpus. 1.0x means the "
          "site is no more common\namong disagreements than among all writes, "
          "i.e. it explains nothing. Only a large\nratio is evidence, and "
          f"`{_NO_EVENT}` means the server did not write at all in the "
          "window\n-- for those the port invented a change the server never made.")


if __name__ == "__main__":
    main()
