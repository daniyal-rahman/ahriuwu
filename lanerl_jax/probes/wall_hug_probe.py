"""Probe (one-off): WHERE do trained champions stand, in the lane frame?
For each replay trace: per-frame (s, n) of both champions in their own team's
lane frame (n > 0 = map-edge side, away from own nexus), the fraction of time
outside the shaping corridor on each side, where minion deaths happen, and a
heatmap PNG over the walkable grid with the corridor rectangle drawn."""
import sys, json, numpy as np
from pathlib import Path
from lanerl_jax.train.reward import LANE_AXIS, LANE_HALF_WIDTH
from PIL import Image, ImageDraw

def frame(x, y, team):
    a = {k: float(np.asarray(v)[team]) for k, v in LANE_AXIS.items()}
    px, py = x - a["origin_x"], y - a["origin_y"]
    return px * a["axis_x"] + py * a["axis_y"], px * a["normal_x"] + py * a["normal_y"], a

for p in sys.argv[1:]:
    d = np.load(p, allow_pickle=True); meta = json.loads(str(d["metadata"]))
    kind, team = d["kind"][0], d["team"][0]; alive = d["alive"]
    champs = np.where(kind == 1)[0]
    print(f"== {Path(p).parent.name}: {meta['label']}")
    W = d["walkable"]; terr = meta.get("terrain", {})
    for c in champs:
        t = int(team[c]); ti = 0 if t in (0, 100) else 1
        ok = alive[:, c] > 0
        s, n, a = frame(d["x"][ok, c], d["y"][ok, c], ti)
        L = a["length"]
        out_edge = (n > LANE_HALF_WIDTH).mean(); out_in = (n < -LANE_HALF_WIDTH).mean()
        beyond = ((s < -LANE_HALF_WIDTH) | (s > L + LANE_HALF_WIDTH)).mean()
        print(f"  champ slot {c} team {t}: frames alive {ok.sum()}; n mean {n.mean():7.0f} p10 {np.percentile(n,10):7.0f} p90 {np.percentile(n,90):7.0f}; "
              f"outside corridor: edge side {out_edge:.2f}, nexus side {out_in:.2f}, beyond ends {beyond:.2f}; s mean {s.mean():.0f}/{L:.0f}")
    # minion deaths: alive 1 -> 0 transitions for kind 0
    mins = np.where(kind == 0)[0]
    deaths = []
    for m in mins:
        al = alive[:, m] > 0
        idx = np.where(al[:-1] & ~al[1:])[0]
        for i in idx: deaths.append((d["x"][i, m], d["y"][i, m]))
    if deaths:
        dx, dy = np.array(deaths).T
        s, n, a = frame(dx, dy, 0)
        print(f"  minion deaths {len(deaths)}: n mean {n.mean():.0f} p10 {np.percentile(n,10):.0f} p90 {np.percentile(n,90):.0f}, outside corridor edge side {(n>LANE_HALF_WIDTH).mean():.2f}")
    # heatmap over the walkable grid
    H, Wd = W.shape
    ox, oy = terr.get("origin", [0, 0])[:2] if "origin" in terr else (0.0, 0.0)
    cell = terr.get("cell", terr.get("cell_size", 50.0))
    img = Image.fromarray(np.where(W, 200, 60).astype(np.uint8)).convert("RGB")
    draw = ImageDraw.Draw(img)
    def to_px(x, y): return ((x - ox) / cell, (y - oy) / cell)
    for c, col in zip(champs, ((255, 60, 60), (60, 120, 255))):
        ok = alive[:, c] > 0
        for x, y in zip(d["x"][ok, c][::3], d["y"][ok, c][::3]):
            px, py = to_px(x, y); draw.point((px, py), fill=col)
    for x, y in deaths: px, py = to_px(x, y); draw.point((px, py), fill=(255, 255, 0))
    # corridor rectangle for blue's frame
    a = {k: float(np.asarray(v)[0]) for k, v in LANE_AXIS.items()}
    corners = []
    for ss, nn in ((-LANE_HALF_WIDTH, -LANE_HALF_WIDTH), (a["length"] + LANE_HALF_WIDTH, -LANE_HALF_WIDTH),
                   (a["length"] + LANE_HALF_WIDTH, LANE_HALF_WIDTH), (-LANE_HALF_WIDTH, LANE_HALF_WIDTH)):
        x = a["origin_x"] + ss * a["axis_x"] + nn * a["normal_x"]; y = a["origin_y"] + ss * a["axis_y"] + nn * a["normal_y"]
        corners.append(to_px(x, y))
    draw.polygon(corners, outline=(0, 255, 0))
    img = img.transpose(Image.FLIP_TOP_BOTTOM).resize((Wd * 3, H * 3), Image.NEAREST)
    out = Path(p).parent / "wall_hug_heatmap.png"; img.save(out); print("  heatmap", out, "terrain meta", {k: terr[k] for k in terr if k != 'walkable'} if isinstance(terr, dict) else terr)
