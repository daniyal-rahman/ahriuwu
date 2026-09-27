"""Probe: per second in game time, the blue champion's CLICK (button, screen
cell -> lane-frame offset from the champion) and its DISPLACEMENT over the
next second, on two traces. Same clicks but different displacement = movement
execution differs; different clicks = the policy saw different observations."""
import sys, json, numpy as np
from lanerl_jax.train.reward import LANE_AXIS
from lanerl_rl.constants import SCREEN_X_VALUES, SCREEN_Y_VALUES, BUTTONS
from lanerl_rl.projection import screen_to_world_centred
a = {k: float(np.asarray(v)[0]) for k, v in LANE_AXIS.items()}
def s_of(x, y): return (x - a["origin_x"]) * a["axis_x"] + (y - a["origin_y"]) * a["axis_y"]
def n_of(x, y): return (x - a["origin_x"]) * a["normal_x"] + (y - a["origin_y"]) * a["normal_y"]
def load(p):
    z = np.load(p, allow_pickle=True); meta = json.loads(str(z["metadata"])); hz = meta.get("hz", 10.)
    d = {k: (z[k] if k in z.files else None) for k in ("x", "y", "kind", "team", "t_ms", "button", "screen_x", "screen_y", "click_x", "click_y", "cs")}
    d["hz"] = hz; d["t0"] = float(np.asarray(d["t_ms"]).reshape(-1)[0]) / 1000.
    kind = d["kind"][0]; ch = [i for i in range(len(kind)) if kind[i] == 1]; d["blue"] = min(ch, key=lambda i: d["team"][0, i])
    d["files"] = z.files; return d
def row(d, gt):
    f = int(round((gt - d["t0"]) * d["hz"])); f1 = min(f + int(d["hz"]), d["x"].shape[0] - 1)
    if f < 0 or f >= d["x"].shape[0]: return None
    b = d["blue"]; bx, by = d["x"][f, b], d["y"][f, b]
    btn = d["button"][f]; btn = int(np.asarray(btn).reshape(-1)[0])
    cx = np.asarray(d["click_x"][f]).reshape(-1)[0]; cy = np.asarray(d["click_y"][f]).reshape(-1)[0]
    ds = float(s_of(cx, cy) - s_of(bx, by)); dn = float(n_of(cx, cy) - n_of(bx, by))
    move_s = float(s_of(d["x"][f1, b], d["y"][f1, b]) - s_of(bx, by)); move_n = float(n_of(d["x"][f1, b], d["y"][f1, b]) - n_of(bx, by))
    return dict(btn=BUTTONS[btn] if 0 <= btn < len(BUTTONS) else str(btn), ds=ds, dn=dn, ms=move_s, mn=move_n, cs=int(d["cs"][f, b]))
A, B = load(sys.argv[1]), load(sys.argv[2]); start = float(sys.argv[3]) if len(sys.argv) > 3 else 120.; secs = int(sys.argv[4]) if len(sys.argv) > 4 else 25
print("fields JAX:", [k for k in ("button", "screen_x", "screen_y", "click_x") if k in A["files"]], "| server:", [k for k in ("button", "screen_x", "screen_y", "click_x") if k in B["files"]])
print(f"{'t(s)':>5s} | {'button J / S':>25s} | {'click ds,dn J':>14s} | {'click ds,dn S':>14s} | {'moved ds,dn J':>14s} | {'moved ds,dn S':>14s} | {'CS J/S':>7s}")
for k in range(secs + 1):
    gt = start + k; x, y = row(A, gt), row(B, gt)
    if x is None or y is None: continue
    print(f"{gt:5.0f} | {x['btn']:>11s} / {y['btn']:>11s} | {x['ds']:6.0f},{x['dn']:6.0f} | {y['ds']:6.0f},{y['dn']:6.0f} | {x['ms']:6.0f},{x['mn']:6.0f} | {y['ms']:6.0f},{y['mn']:6.0f} | {x['cs']:3d} / {y['cs']:2d}")
