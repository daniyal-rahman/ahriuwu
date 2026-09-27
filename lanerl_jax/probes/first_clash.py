"""Probe: the first clash, second by second in GAME TIME, on two traces.
Per second from control start: blue champion position (lane s), whether it is
attacking and what (model, hp of its aa_target), the lowest-HP enemy minion in
600 u (hp), the number of enemy/own minions within 800 u of the champion, and
the champion's CS. Where the first last-hit timing diverges is where combat
dynamics differ.
    python -m lanerl_jax.probes.first_clash jax.npz server.npz [start_s=120] [seconds=40]
"""
import sys, json, numpy as np
from lanerl_jax.train.reward import LANE_AXIS
a = {k: float(np.asarray(v)[0]) for k, v in LANE_AXIS.items()}
def s_of(x, y): return (x - a["origin_x"]) * a["axis_x"] + (y - a["origin_y"]) * a["axis_y"]
def load(p):
    z = np.load(p, allow_pickle=True); meta = json.loads(str(z["metadata"])); hz = meta.get("hz", 10.)
    d = {k: z[k] for k in ("x", "y", "kind", "team", "alive", "hp", "model", "cs", "is_attacking", "aa_target", "aa_windup", "t_ms")}
    d["hz"] = hz; d["t0"] = float(np.asarray(d["t_ms"]).reshape(-1)[0]) / 1000.  # game time of frame 0
    kind = d["kind"][0]; ch = [i for i in range(len(kind)) if kind[i] == 1]; d["blue"] = min(ch, key=lambda i: d["team"][0, i])
    return d
def row(d, gt):
    f = int(round((gt - d["t0"]) * d["hz"]))
    if f < 0 or f >= d["x"].shape[0]: return None
    b = d["blue"]; bx, by = d["x"][f, b], d["y"][f, b]
    mins = (d["model"][f] >= 2) & (d["model"][f] <= 7) & (d["alive"][f] > 0) & (d["hp"][f] > 0)
    dist = np.hypot(d["x"][f] - bx, d["y"][f] - by)
    enemy = mins & (d["team"][f] == 1); own = mins & (d["team"][f] == 0)
    near_e = enemy & (dist < 800); near_o = own & (dist < 800)
    lowest = float(d["hp"][f][enemy & (dist < 600)].min()) if (enemy & (dist < 600)).any() else np.nan
    tgt = int(d["aa_target"][f, b]) if d["aa_target"].ndim == 2 else -1
    tgt_desc = f"m{int(d['model'][f, tgt])}:{int(d['hp'][f, tgt])}" if 0 <= tgt < d["x"].shape[1] and d["alive"][f, tgt] > 0 else "-"
    return dict(s=float(s_of(bx, by)), atk=int(d["is_attacking"][f, b]) if d["is_attacking"].ndim == 2 else -1, tgt=tgt_desc,
                low=lowest, ne=int(near_e.sum()), no=int(near_o.sum()), cs=int(d["cs"][f, b]))
A, B = load(sys.argv[1]), load(sys.argv[2]); start = float(sys.argv[3]) if len(sys.argv) > 3 else 120.; secs = int(sys.argv[4]) if len(sys.argv) > 4 else 40
print(f"frame-0 game time: JAX {A['t0']:.1f} s, server {B['t0']:.1f} s")
print(f"{'t(s)':>5s} | {'champ s J/S':>13s} | {'attacking J/S':>13s} | {'aa target J/S':>21s} | {'lowest enemy hp J/S':>19s} | {'enemy/own in 800u J/S':>22s} | {'CS J/S':>7s}")
for k in range(secs + 1):
    gt = start + k; x, y = row(A, gt), row(B, gt)
    if x is None or y is None: continue
    print(f"{gt:5.0f} | {x['s']:5.0f} / {y['s']:5.0f} | {x['atk']:5d} / {y['atk']:5d} | {x['tgt']:>9s} / {y['tgt']:>9s} | {x['low']:8.0f} / {y['low']:8.0f} | {x['ne']}/{x['no']:>2d} / {y['ne']}/{y['no']:>2d} | {x['cs']:3d} / {y['cs']:2d}")
