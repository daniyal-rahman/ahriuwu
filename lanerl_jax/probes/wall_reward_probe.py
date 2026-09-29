"""Probe (one-off): is the wall REWARDED? From a replay trace, per champion,
bucket frames by distance to the nearest wall and report what happened there:
CS gained per minute, deaths per minute, level-ups (XP proxy) per minute,
enemy-champion proximity, and the fraction of time. If the wall bucket earns
no more than the lane bucket, standing there is not a learned reward preference."""
import sys, json, numpy as np
from pathlib import Path
from scipy import ndimage
for rd in sys.argv[1:]:
    rd = Path(rd); z = np.load(rd / "trace.npz", allow_pickle=True); meta = json.loads(str(z["metadata"]))
    d = {k: z[k] for k in ("x", "y", "kind", "team", "alive", "walkable", "cs", "deaths", "level", "hp")}
    terr = meta["terrain"]; W = d["walkable"].astype(bool); cell = terr["cell_size"]; mx, my = terr["min_x"], terr["min_y"]
    dist = ndimage.distance_transform_edt(W) * cell
    kind, team = d["kind"][0], d["team"][0]; champs = list(np.where(kind == 1)[0]); hz = meta["hz"]
    print(f"== {rd.name}: {meta['label']}")
    def wd(x, y):
        i, j = int((y - my) / cell), int((x - mx) / cell)
        return float(dist[i, j]) if 0 <= i < W.shape[0] and 0 <= j < W.shape[1] and W[i, j] else 0.0
    for c in champs:
        other = [o for o in champs if o != c][0]
        T = d["x"].shape[0]
        w = np.array([wd(d["x"][t, c], d["y"][t, c]) for t in range(T)])
        enemy_d = np.hypot(d["x"][:, c] - d["x"][:, other], d["y"][:, c] - d["y"][:, other])
        cs_gain = np.diff(d["cs"][:, c], prepend=d["cs"][0, c]) > 0
        al = d["alive"][:, c] > 0; death = np.concatenate([[False], al[:-1] & ~al[1:]])
        lvl = np.diff(d["level"][:, c], prepend=d["level"][0, c]) > 0
        alive = d["alive"][:, c] > 0
        print(f"  champion slot {c} team {int(team[c])}")
        for name, sel in (("at wall  (<150 u)", w < 150), ("near     (150-400)", (w >= 150) & (w < 400)), ("open     (>=400)", w >= 400)):
            sel = sel & alive; mins = sel.sum() / hz / 60
            if mins < 0.1: print(f"    {name}: {mins:.1f} min"); continue
            print(f"    {name}: {mins:4.1f} min ({sel.mean():.0%}) | CS/min {cs_gain[sel].sum()/mins:4.1f} | deaths/min {death[sel].sum()/mins:.2f} | level-ups/min {lvl[sel].sum()/mins:.2f} | enemy champ median dist {np.median(enemy_d[sel]):5.0f} u, <800 u {(enemy_d[sel]<800).mean():.0%}")
