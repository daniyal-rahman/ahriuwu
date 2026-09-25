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
# SIGNED bias and POWER, alongside the agreement test. Both were missing and
# both produced published errors.
#
# `gap()` takes abs(), so a large value means the two arms DISAGREE per seed --
# it says nothing about direction. `deaths 0.542 EXCESS` was reported as "the
# sim dies 0.54 times more per episode"; the signed bias is -0.125 (t=-0.37),
# i.e. the sim dies slightly LESS and indistinguishably so. The marginals were
# sim 1.625 / server 1.750. An agreement metric read as a directional one.
#
# And a metric can be INDISTINGUISHABLE because the test has no power. If the
# order-shuffle already destroys all per-seed correlation, the reference gap
# equals the fully-decoupled bound E|X-Y| and NOTHING can exceed it. Measured:
# cs 1.06 and attacks 0.97 of that bound -- those two can never fail, so
# reporting them as "indistinguishable from the floor" was not evidence of
# agreement. deaths 0.47, xp_share 0.46, max_level 0.59 do have headroom.
def _decouple_bound(xs, ys):
    """E|X-Y| if the two arms were independent: all cross-seed pairs."""
    return (statistics.mean(abs(x - y) for x in xs for y in ys)
            if xs and ys else float("nan"))

print(f"\n{'metric':<13} {'|diff|':>9} {'95% CI':>20} {'t':>6} "
      f"{'signed':>9} {'power':>7}  verdict")
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

    # signed sim-vs-server bias on the raw metric, where one exists
    signed = float("nan")
    if m in ("cs", "deaths"):
        sv = [r["sim"][m] - r["server"][m] for r in rows]
        signed = statistics.mean(sv)
    # power: reference gap over the fully-decoupled bound. ~1.0 = no power.
    ref = statistics.mean(
        [b[m] for b in sh
         if not (isinstance(b[m], float) and math.isnan(b[m]))] or [float("nan")])
    if m in ("cs", "deaths", "max_level"):
        bound = _decouple_bound([r["server"][m] for r in rows],
                                [r["shuffled"][m] for r in rows]) \
            if m in ("cs", "deaths") else float("nan")
    else:
        bound = float("nan")
    power = ref / bound if bound and not math.isnan(bound) else float("nan")
    pw = "NONE" if (not math.isnan(power) and power >= 0.9) else (
        f"{power:.2f}" if not math.isnan(power) else "-")
    sg = f"{signed:+.3f}" if not math.isnan(signed) else "-"
    print(f"{m:<13} {mu:>9.3f} {lo:>9.2f}..{hi:<9.2f} {t:>6.2f} "
          f"{sg:>9} {pw:>7}  {v}")

print("\n|diff| is an AGREEMENT metric -- it is abs(), so it says nothing about")
print("direction. `signed` is the raw sim-minus-server bias; read it before")
print("claiming the sim does more or less of anything.")
print("power = reference gap / fully-decoupled bound. NONE (>=0.9) means the")
print("shuffle already destroys all per-seed pairing, so the metric CANNOT fail")
print("and an INDISTINGUISHABLE verdict on it is not evidence of agreement.")
