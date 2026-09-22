"""Pool the per-job outcome JSONs into one 30-seed verdict."""
import json, math, statistics, sys
from pathlib import Path

# RUN GUARD. Per-job JSONs are written at job END, so a re-run leaves the
# previous run's file in place until its replacement lands; pooling once mixed
# 24 valid seeds with 6 from a superseded run whose reference gap was a broken
# 0.000, which flipped a verdict.
#
# mtime was the first attempt and it is the WRONG SIGNAL: jobs in one batch
# routinely finish minutes apart when one is queued behind the others, and the
# skew rule then threw away the 24 seeds that finished first and kept the 6
# that finished last. Rows now carry an explicit `run_tag`; untagged files fall
# back to mtime with a generous window, and the fallback says so.
paths = sorted(Path("lanerl_jax/runs").glob(
    sys.argv[1] if len(sys.argv) > 1 else "noisy_g*.json"))
if not paths:
    raise SystemExit("no outcome JSONs found")
loaded = [(p, json.loads(p.read_text())) for p in paths]
tags = {r.get("run_tag") for _p, rs in loaded for r in rs}
rows, stale = [], []
if tags - {None}:
    from collections import Counter
    best = Counter(r.get("run_tag") for _p, rs in loaded
                   for r in rs if r.get("run_tag")).most_common(1)[0][0]
    for p, rs in loaded:
        keep = [r for r in rs if r.get("run_tag") == best]
        (rows.extend(keep) if keep
         else stale.append(f"{p.name} (tag != {best})"))
    print(f"run_tag = {best}")
else:
    SKEW = 3600
    newest = max(p.stat().st_mtime for p, _ in loaded)
    print("!! no run_tag in these files -- falling back to mtime with a "
          f"{SKEW}s window, which cannot distinguish a slow job from a stale one")
    for p, rs in loaded:
        if newest - p.stat().st_mtime > SKEW:
            stale.append(f"{p.name} ({int(newest - p.stat().st_mtime)}s older)")
        else:
            rows.extend(rs)
if stale:
    print(f"!! SKIPPED: {', '.join(stale)}")
print(f"pooled {len(rows)} seeds from {len(paths) - len(stale)} jobs"
      f" ({len(paths)} found, {len(stale)} skipped)")

# the inertness guard, applied to the POOL as well as each job
sig = {json.dumps({k: v for k, v in r.items() if k != "seed"},
                  sort_keys=True, default=str) for r in rows}
print(f"distinct outcomes: {len(sig)} of {len(rows)}"
      + ("   !! SEEDS INERT -- this is one run, not a sample" if len(sig) == 1
         else "   (knob is live)"))

def gap(a, b):
    la, lb = a["levels"], b["levels"]
    la = {int(k): v for k, v in la.items()}; lb = {int(k): v for k, v in lb.items()}
    sh = sorted(set(la) & set(lb))
    return {
        "cs": abs(a["cs"] - b["cs"]),
        "deaths": abs(a["deaths"] - b["deaths"]),
        "max_level": abs(max(la, default=0) - max(lb, default=0)),
        "levelup_lag": statistics.mean(abs(la[l] - lb[l]) for l in sh) if sh else float("nan"),
        "xp_share": abs(a["xp_share"] - b["xp_share"]),
    }

have_shuf = all("shuffled" in r for r in rows)
ss = [gap(r["sim"], r["server"]) for r in rows]
sh = [gap(r["server"], r["shuffled"]) for r in rows] if have_shuf else None

# PAIRED, because the three arms share a seed. The first version of this
# compared two MEANS with no test, which `gate3_outcomes.py` itself documents
# as a silent defect: the spreads here are as large as the means (cs
# 2.000+/-2.066 against 1.533+/-2.262), so mean-vs-mean called EXCESS on all
# five metrics where the paired test separates one. Worse, this file is the
# tool that pools ACROSS jobs, so it produced the headline verdict while its
# sibling had already been corrected -- nothing at HEAD could reproduce the
# published t and CI values.
print(f"\n{'metric':<13} {'mean diff':>10} {'95% CI':>22} {'t':>7}  verdict")
for m in ("cs", "deaths", "max_level", "levelup_lag", "xp_share"):
    if not sh:
        print(f"{m:<13} {'(no reference -- rerun with --shuffled)':>44}")
        continue
    d = [a[m] - b[m] for a, b in zip(ss, sh)
         if not (isinstance(a[m], float) and math.isnan(a[m]))
         and not (isinstance(b[m], float) and math.isnan(b[m]))]
    n = len(d)
    if n < 2:
        print(f"{m:<13} {'(too few paired samples)':>44}")
        continue
    mu = statistics.mean(d)
    se = statistics.stdev(d) / math.sqrt(n)
    t = mu / se if se else float("inf")
    lo, hi = mu - 1.96 * se, mu + 1.96 * se
    v = "EXCESS" if lo > 0 else ("PASS" if hi < 0 else "INDISTINGUISHABLE")
    print(f"{m:<13} {mu:>10.3f} {lo:>10.2f}..{hi:<10.2f} {t:>7.2f}  {v}")
print("\ndiff > 0 = the SIM deviates from the server MORE than the server")
print("deviates from ITSELF under a permuted update order.")
print("INDISTINGUISHABLE is neither pass nor fail: at this sample size the")
print("experiment cannot separate them.")
