"""Probe (one-off): is the wall sticky through the click interface? For each
replay: fraction of move/attack-move clicks whose WORLD target is unwalkable
(off-map or wall), how that fraction depends on the champion's distance to the
nearest wall, where those clicks resolve, and the champion-champion distance
over time with deaths marked."""
import sys, json, numpy as np
from pathlib import Path
from scipy import ndimage

for rd in sys.argv[1:]:
    rd = Path(rd); z = np.load(rd / "trace.npz", allow_pickle=True); meta = json.loads(str(z["metadata"]))
    d = {k: z[k] for k in ("x", "y", "kind", "team", "alive", "walkable")}   # decompress ONCE (npz re-reads per access)
    terr = meta["terrain"]; W = d["walkable"].astype(bool); cell = terr["cell_size"]; mx, my = terr["min_x"], terr["min_y"]
    dist_to_wall = ndimage.distance_transform_edt(W) * cell          # distance from a walkable cell to the nearest unwalkable
    acts = json.load(open(rd / "policy_actions.json"))
    t_ms = np.asarray(acts["t_ms"]); hz = meta["hz"]
    kind, team = d["kind"][0], d["team"][0]; champs = np.where(kind == 1)[0]
    tv = sorted(int(team[c]) for c in champs)
    c_by_team = {(100 if i == 0 else 200): c for i, c in enumerate(sorted(champs, key=lambda c: int(team[c])))}  # lower team id = blue
    print(f"== {rd.name}: {meta['label']}")
    def cellidx(x, y):
        i = int((y - my) / cell); j = int((x - mx) / cell)
        return (i, j) if 0 <= i < W.shape[0] and 0 <= j < W.shape[1] else None
    for side, tm in (("blue", 100), ("red", 200)):
        c = c_by_team.get(tm); 
        if c is None: continue
        n_click = n_unwalk = n_offmap = 0; unwalk_by_walldist = {"<300": [0, 0], "300-800": [0, 0], ">800": [0, 0]}
        toward_wall = 0
        for k, a in enumerate(acts[side]):
            if not a or a.get("t") not in ("move", "attack_move", "click"): continue
            fi = min(int(t_ms[k] / 1000 * hz), d["x"].shape[0] - 1)
            cx, cy = float(d["x"][fi, c]), float(d["y"][fi, c])
            ci = cellidx(cx, cy)
            wd = float(dist_to_wall[ci]) if ci and W[ci] else 0.0
            key = "<300" if wd < 300 else "300-800" if wd < 800 else ">800"
            ti = cellidx(a["x"], a["y"]); unwalk = (ti is None) or (not W[ti])
            n_click += 1; unwalk_by_walldist[key][1] += 1
            if unwalk:
                n_unwalk += 1; unwalk_by_walldist[key][0] += 1
                if ti is None: n_offmap += 1
        print(f"  {side}: clicks {n_click}, target unwalkable {n_unwalk/max(n_click,1):.2f} (off-grid {n_offmap/max(n_click,1):.2f}); "
              "unwalkable fraction by champion distance to nearest wall: " +
              ", ".join(f"{k} {v[0]/max(v[1],1):.2f} (n={v[1]})" for k, v in unwalk_by_walldist.items()))
        ok = d["alive"][:, c] > 0
        wd_t = np.array([dist_to_wall[cellidx(x, y)] if cellidx(x, y) and W[cellidx(x, y)] else 0.0 for x, y in zip(d["x"][ok, c], d["y"][ok, c])])
        print(f"  {side}: champion distance to nearest wall: median {np.median(wd_t):.0f} u, frac <150 u {(wd_t<150).mean():.2f}, frac <300 u {(wd_t<300).mean():.2f}")
    if len(champs) == 2:
        b, r = champs
        dd = np.hypot(d["x"][:, b] - d["x"][:, r], d["y"][:, b] - d["y"][:, r])
        both = (d["alive"][:, b] > 0) & (d["alive"][:, r] > 0)
        dead = []
        for c in champs:
            al = d["alive"][:, c] > 0; dead += [(int(i), int(c), float(dd[i])) for i in np.where(al[:-1] & ~al[1:])[0]]
        q = [int(x) for x in np.percentile(dd[both], [10, 50, 90])]
        print(f"  champion-champion distance: p10/50/90 {q} u; frac <1500 u {(dd[both]<1500).mean():.2f}; deaths (frame, slot, dist at death): {dead}")
