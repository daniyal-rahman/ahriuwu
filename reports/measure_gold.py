"""Re-measure the enemy-gold liveness claims from labels.json directly.

Independent of docs/ENEMY_GOLD_*: reads a stratified sample of games, extracts
every hero's gold_total per frame, and recomputes (i) the own-vs-opponent lump
spectrum and its total variation distance, (ii) the update cadence split by
whether the opponent is on screen, (iii) the longest constant-gold hold.
"""
import json, os, sys, glob
import numpy as np

ROOT = "/srv/nfs/datasets/lol_replays_16_9_772"
sys.path.insert(0, "/srv/nfs/projects/ahriuwu/src")
from ahriuwu.data.lane_opponent import resolve_lane_opponent   # noqa

games = sorted(d for d in os.listdir(ROOT) if d.startswith("NA1_"))
sel = games[::6][:24]

own_l, opp_l = [], []
rate_on, rate_off = [], []
hold_own, hold_opp = [], []
n_games = 0
for g in sel:
    p = f"{ROOT}/{g}/labels.json"
    if not os.path.exists(p):
        continue
    try:
        lab = json.load(open(p))
    except Exception:
        continue
    opp = resolve_lane_opponent(lab)
    if not opp:
        continue
    frames = lab["frames"]
    own, oth, onscr, gt = [], [], [], []
    for f in frames:
        L = f.get("label") or {}
        vh = L.get("visible_heroes") or []
        cs = L.get("champion_stats") or {}
        o = cs.get("gold_total")
        e = None; sc = None
        for h in vh:
            if h.get("name") == opp:
                e = h.get("gold_total"); sc = h.get("screen")
        if o is None or e is None:
            continue
        own.append(o); oth.append(e); onscr.append(sc is not None)
        gt.append(f.get("gt", np.nan))
    if len(own) < 5000:
        continue
    own = np.array(own, float); oth = np.array(oth, float)
    onscr = np.array(onscr, bool); gt = np.array(gt, float)
    d_own, d_opp = np.diff(own), np.diff(oth)
    own_l.append(d_own[d_own >= 3]); opp_l.append(d_opp[d_opp >= 3])
    late = gt[1:] > 150
    m_on, m_off = late & onscr[1:], late & ~onscr[1:]
    if m_on.sum() > 500 and m_off.sum() > 500:
        rate_on.append(float((d_opp[m_on] > 0).mean()))
        rate_off.append(float((d_opp[m_off] > 0).mean()))
    for arr, sink in ((own, hold_own), (oth, hold_opp)):
        ch = np.flatnonzero(np.diff(arr) != 0) + 1
        b = np.concatenate(([0], ch, [len(arr)]))
        runs = np.diff(b)
        sink.append(int(runs[gt[b[:-1]] > 150].max()) if (gt[b[:-1]] > 150).any() else 0)
    n_games += 1
    print("ok", g, opp, len(own), flush=True)

own_l = np.concatenate(own_l); opp_l = np.concatenate(opp_l)
edges = np.arange(3, 121, 1)
ho, _ = np.histogram(own_l, bins=edges); he, _ = np.histogram(opp_l, bins=edges)
po, pe = ho / ho.sum(), he / he.sum()
tvd = 0.5 * np.abs(po - pe).sum()
out = dict(n_games=n_games, n_lumps_own=int(own_l.size), n_lumps_opp=int(opp_l.size),
           tvd=float(tvd), edges=edges[:-1].tolist(), p_own=po.tolist(), p_opp=pe.tolist(),
           rate_on=float(np.mean(rate_on)), rate_off=float(np.mean(rate_off)),
           rate_ratio=float(np.mean(rate_off) / np.mean(rate_on)),
           hold_own_max_frames=int(max(hold_own)), hold_opp_max_frames=int(max(hold_opp)))
json.dump(out, open("/srv/nfs/projects/ahriuwu/reports/figdata/gold_stats.json", "w"), indent=1)
print({k: v for k, v in out.items() if not isinstance(v, list)})
