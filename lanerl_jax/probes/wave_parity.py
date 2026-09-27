"""Probe: WAVE dynamics on two traces of the same deterministic scripted game.
Per-frame minion team codes (0 blue, 1 red; anything else = empty slot) give,
every 30 s from the first last-hit: alive blue/red minions, each side's front
(max/min lane-frame s of its alive minions), the meeting point, and the blue
champion's s. This is "do the minions differ?" as numbers.
    python -m lanerl_jax.probes.wave_parity jax_trace.npz server_trace.npz
"""
import sys, json, numpy as np
from lanerl_jax.train.reward import LANE_AXIS
a = {k: float(np.asarray(v)[0]) for k, v in LANE_AXIS.items()}
def s_of(x, y): return (x - a["origin_x"]) * a["axis_x"] + (y - a["origin_y"]) * a["axis_y"]
def load(p):
    z = np.load(p, allow_pickle=True); meta = json.loads(str(z["metadata"]))
    d = {k: z[k] for k in ("x", "y", "kind", "team", "alive", "cs", "hp", "model")}; d["hz"] = meta.get("hz", 10.)
    kind = d["kind"][0]; mins = np.arange(len(kind))
    ch = [i for i in range(len(kind)) if kind[i] == 1]; blue = min(ch, key=lambda i: d["team"][0, i])
    d["mins"], d["blue"] = mins, blue
    d["base"] = max(0, int(np.argmax(d["cs"][:, blue] > 0)) - int(12 * d["hz"]))
    return d
def row(d, t_s):
    f = d["base"] + int(t_s * d["hz"])
    if f >= d["x"].shape[0]: return None
    m = d["mins"]; md = d["model"][f, m]; al = (d["alive"][f, m] > 0) & (d["hp"][f, m] > 0) & (md >= 2) & (md <= 7); tm = d["team"][f, m]
    s = s_of(d["x"][f, m], d["y"][f, m])
    b = al & (tm == 0); r = al & (tm == 1)
    return dict(nb=int(b.sum()), nr=int(r.sum()), b_front=float(s[b].max()) if b.any() else np.nan, r_front=float(s[r].min()) if r.any() else np.nan,
                champ=float(s_of(d["x"][f, d["blue"]], d["y"][f, d["blue"]])), cs=int(d["cs"][f, d["blue"]]))
A, B = load(sys.argv[1]), load(sys.argv[2])
print(f"{'t(s)':>5s} | {'blue alive J/S':>14s} | {'red alive J/S':>13s} | {'blue front s J/S':>16s} | {'red front s J/S':>15s} | {'champ s J/S':>13s} | {'CS J/S':>7s}")
for t in range(0, 481, 30):
    x, y = row(A, t), row(B, t)
    if x is None or y is None: break
    print(f"{t:5d} | {x['nb']:6d} / {y['nb']:5d} | {x['nr']:6d} / {y['nr']:4d} | {x['b_front']:7.0f} / {y['b_front']:6.0f} | {x['r_front']:7.0f} / {y['r_front']:5.0f} | {x['champ']:6.0f} / {y['champ']:4.0f} | {x['cs']:3d} / {y['cs']:2d}")
