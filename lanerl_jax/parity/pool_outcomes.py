"""Pool the per-job outcome JSONs into one 30-seed verdict."""
import json, math, statistics, sys
from pathlib import Path

# STALENESS GUARD. Per-job JSONs are written at job END, so a re-run leaves
# the previous run's file in place until its replacement lands. Pooling once
# mixed 24 valid seeds with 6 from a superseded run whose reference gap was a
# broken 0.000 -- which biased the reference DOWN and flipped `levelup_lag`
# from indistinguishable to EXCESS. Newest-file mtime is the reference; a file
# more than `SKEW` older than it is from a different run.
SKEW = 300
paths = sorted(Path("lanerl_jax/runs").glob(
    sys.argv[1] if len(sys.argv) > 1 else "noisy_g*.json"))
if not paths:
    raise SystemExit("no outcome JSONs found")
newest = max(q.stat().st_mtime for q in paths)
rows, stale = [], []
for p in paths:
    if newest - p.stat().st_mtime > SKEW:
        stale.append(f"{p.name} ({int(newest - p.stat().st_mtime)}s older)")
        continue
    rows.extend(json.loads(p.read_text()))
if stale:
    print(f"!! SKIPPED STALE (from a superseded run): {', '.join(stale)}")
print(f"pooled {len(rows)} seeds from {len(list(Path('lanerl_jax/runs').glob('noisy_g*.json')))} jobs")

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

print(f"\n{'metric':<14} {'sim-server mean':>16} {'sd':>8} "
      + (f"{'srv-shuffled mean':>18} {'sd':>8}   verdict" if sh else ""))
for m in ("cs", "deaths", "max_level", "levelup_lag", "xp_share"):
    va = [g[m] for g in ss if not (isinstance(g[m], float) and math.isnan(g[m]))]
    ma = statistics.mean(va) if va else float("nan")
    sa = statistics.pstdev(va) if len(va) > 1 else 0.0
    line = f"{m:<14} {ma:>16.3f} {sa:>8.3f}"
    if sh:
        vb = [g[m] for g in sh if not (isinstance(g[m], float) and math.isnan(g[m]))]
        mb = statistics.mean(vb) if vb else float("nan")
        sb = statistics.pstdev(vb) if len(vb) > 1 else 0.0
        line += f" {mb:>18.3f} {sb:>8.3f}   {'PASS' if ma <= mb else 'EXCESS'}"
    print(line)
if sh:
    print("\nPASS = the simulator differs from the server by no more than the server\n"
          "differs from ITSELF under an equally arbitrary update order.")
