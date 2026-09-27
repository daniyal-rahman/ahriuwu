"""Probe: per-WAVE timeline on a trace, to locate the FIRST divergence between
two engines and whether it is before contact (spawn + movement) or after it
(aggro, micro-pathing, damage). For every minion instance (contiguous alive run
of a slot with a minion profile code): team, model, spawn frame, the frame it
crosses lane marks (fractions of the lane), death frame. Waves = spawn-frame
clusters per team. Prints per wave: spawn s, arrival at the 0.40/0.50/0.60
marks (blue crosses upward, red downward), first death s, survivors 20 s after
the first death, and the wave's total deaths.
    python -m lanerl_jax.probes.wave_timeline jax.npz server.npz
"""
import sys, json, numpy as np
from lanerl_jax.train.reward import LANE_AXIS
a = {k: float(np.asarray(v)[0]) for k, v in LANE_AXIS.items()}; L = a["length"]
def s_of(x, y): return (x - a["origin_x"]) * a["axis_x"] + (y - a["origin_y"]) * a["axis_y"]
def instances(p):
    z = np.load(p, allow_pickle=True); meta = json.loads(str(z["metadata"])); hz = meta.get("hz", 10.)
    alive, hp, team, model, x, y, cs, kind = (z[k] for k in ("alive", "hp", "team", "model", "x", "y", "cs", "kind"))
    ch = [i for i in range(len(kind[0])) if kind[0][i] == 1]; blue = min(ch, key=lambda i: team[0, i])
    t0 = int(np.argmax(cs[:, blue] > 0)) - int(12 * hz)     # common clock: first last-hit - 12 s
    out = []
    for i in range(alive.shape[1]):
        al = (alive[:, i] > 0) & (hp[:, i] > 0) & (model[:, i] >= 2) & (model[:, i] <= 7)
        starts = np.where(al[1:] & ~al[:-1])[0] + 1
        if al[0]: starts = np.concatenate([[0], starts])
        ends = np.where(al[:-1] & ~al[1:])[0]
        for s0 in starts:
            e = ends[ends >= s0]; e = int(e[0]) if len(e) else alive.shape[0] - 1
            s = s_of(x[s0:e + 1, i], y[s0:e + 1, i]); tm = int(team[s0, i])
            marks = {}
            for frac in (0.40, 0.50, 0.60):
                if tm == 0: k = np.where(s >= frac * L)[0]
                else: k = np.where(s <= frac * L)[0]
                marks[frac] = (s0 + int(k[0]) - t0) / hz if len(k) else np.nan
            out.append(dict(team=tm, model=int(model[s0, i]), spawn=(s0 - t0) / hz, death=(e - t0) / hz if e < alive.shape[0] - 1 else np.nan, marks=marks))
    return out
def waves(inst, team):
    rows = sorted([r for r in inst if r["team"] == team], key=lambda r: r["spawn"]); ws = []
    for r in rows:
        if ws and r["spawn"] - ws[-1][-1]["spawn"] < 5: ws[-1].append(r)
        else: ws.append([r])
    return ws
def summarize(ws):
    out = []
    for w in ws:
        deaths = sorted([r["death"] for r in w if np.isfinite(r["death"])]); first = deaths[0] if deaths else np.nan
        surv20 = sum(1 for r in w if not np.isfinite(r["death"]) or r["death"] > first + 20) if deaths else len(w)
        m = lambda f: np.nanmedian([r["marks"][f] for r in w])
        out.append(dict(n=len(w), spawn=w[0]["spawn"], m40=m(0.40), m50=m(0.50), m60=m(0.60), first_death=first, surv20=surv20, deaths=len(deaths)))
    return out
A, B = instances(sys.argv[1]), instances(sys.argv[2])
for team, name, marks in ((0, "BLUE", ("m40", "m50", "m60")), (1, "RED", ("m60", "m50", "m40"))):
    sa, sb = summarize(waves(A, team)), summarize(waves(B, team))
    print(f"\n{name} waves (times in s on the common clock; J = JAX, S = server)")
    print(f"{'wave':>4s} | {'n J/S':>7s} | {'spawn J/S':>13s} | {'reach '+marks[0]+' J/S':>17s} | {'reach '+marks[1]+' J/S':>17s} | {'first death J/S':>15s} | {'alive +20s J/S':>14s} | {'deaths J/S':>10s}")
    for k, (x, y) in enumerate(zip(sa, sb)):
        print(f"{k:4d} | {x['n']:3d} / {y['n']:3d} | {x['spawn']:6.1f} / {y['spawn']:6.1f} | {x[marks[0]]:8.1f} / {y[marks[0]]:6.1f} | {x[marks[1]]:8.1f} / {y[marks[1]]:6.1f} | {x['first_death']:7.1f} / {y['first_death']:5.1f} | {x['surv20']:6d} / {y['surv20']:5d} | {x['deaths']:4d} / {y['deaths']:3d}")
