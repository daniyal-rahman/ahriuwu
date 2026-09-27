"""Probe: event-level parity of two replay traces (JAX vs server) of the SAME
deterministic scripted game. Per game minute: enemy/own minion deaths, the
blue champion's CS, minion deaths near blue's outer turret (turret steals),
the enemy wave's centroid along the lane, and the champion's position.

    python -m lanerl_jax.probes.trace_parity_diff jax_trace.npz server_trace.npz
"""
import sys, json, numpy as np
from lanerl_jax.train.reward import LANE_AXIS
BLUE_TURRET = (981., 10441.)   # TOP_OUTER_TURRET[BLUE] (lane frame origin, blue)

def load(p):
    z = np.load(p, allow_pickle=True); meta = json.loads(str(z["metadata"]))
    d = {k: z[k] for k in ("x", "y", "kind", "team", "alive", "cs", "hp")}
    d["spawn_seq"] = z["spawn_seq"] if "spawn_seq" in z.files else np.zeros_like(z["alive"])
    d["hz"] = meta.get("hz", 10.); d["start"] = float(meta.get("start_seconds", 0.0)); d["label"] = meta.get("label", p)
    return d

def per_minute(d, minutes=10):
    """Minutes are counted from the blue champion's FIRST LAST-HIT minus 12 s
    (both traces start at different points of the setup walk). Minion side is
    the lane-frame position where the slot was first seen alive (minion team
    codes differ between the two renderers); a death is hp > 0 -> hp <= 0 or
    alive 1 -> 0 (the JAX renderer recycles slots)."""
    kind, team = d["kind"][0], d["team"][0]; hz = d["hz"]
    T = d["x"].shape[0]
    blue_t = min(team[kind == 1]); champ = [i for i in range(len(kind)) if kind[i] == 1 and team[i] == blue_t][0]
    minions = [i for i in range(len(kind)) if kind[i] == 0]
    a = {k: float(np.asarray(v)[0]) for k, v in LANE_AXIS.items()}
    def s_of(x, y): return (x - a["origin_x"]) * a["axis_x"] + (y - a["origin_y"]) * a["axis_y"]
    # A minion INSTANCE is (slot, spawn_seq): the JAX renderer recycles slots.
    instances = []
    for i in minions:
        seqs = d["spawn_seq"][:, i]
        for sq in np.unique(seqs):
            fr = np.where((seqs == sq) & (d["alive"][:, i] > 0))[0]
            if len(fr) == 0: continue
            sd = "own" if s_of(d["x"][fr[0], i], d["y"][fr[0], i]) < a["length"] / 2 else "enemy"
            instances.append((i, sq, sd, fr[0], fr[-1]))
    enemy = [(i, sq, f0_, f1_) for i, sq, sd, f0_, f1_ in instances if sd == "enemy"]
    own = [(i, sq, f0_, f1_) for i, sq, sd, f0_, f1_ in instances if sd == "own"]
    first_cs = int(np.argmax(d["cs"][:, champ] > 0)); base = max(0, first_cs - int(12 * hz))
    rows = []
    for m in range(1, minutes + 1):
        f0, f1 = base + int((m - 1) * 60 * hz), min(base + int(m * 60 * hz), T - 1)
        if f0 >= T: break
        def deaths(inst):
            n = 0; near = 0
            for i, sq, fa, fb in inst:
                # the instance's last alive frame is its death (if inside the window and before the trace end)
                if f0 <= fb < f1 and fb < T - 1:
                    n += 1
                    if np.hypot(d["x"][fb, i] - BLUE_TURRET[0], d["y"][fb, i] - BLUE_TURRET[1]) < 900: near += 1
            return n, near
        de, de_near = deaths(enemy); do, _ = deaths(own)
        cs = int(d["cs"][f1, champ]) - int(d["cs"][f0, champ])
        alive_e = [i for i, sq, fa, fb in enemy if fa <= f1 <= fb]
        cen = float(np.mean([s_of(d["x"][f1, i], d["y"][f1, i]) for i in alive_e])) if alive_e else float("nan")
        rows.append(dict(minute=m, enemy_deaths=de, enemy_deaths_near_blue_turret=de_near, own_deaths=do, cs=cs,
                         enemy_wave_s=cen, champ_s=float(s_of(d["x"][f1, champ], d["y"][f1, champ])), cs_total=int(d["cs"][f1, champ])))
    return rows

A, B = load(sys.argv[1]), load(sys.argv[2])
ra, rb = per_minute(A), per_minute(B)
print(f"{'min':>3s} | {'enemy deaths J/S':>16s} | {'near blue turret J/S':>20s} | {'own deaths J/S':>14s} | {'CS J/S':>7s} | {'CS total J/S':>12s} | {'enemy wave s J/S':>17s} | {'champ s J/S':>13s}")
for x, y in zip(ra, rb):
    print(f"{x['minute']:3d} | {x['enemy_deaths']:7d} / {y['enemy_deaths']:6d} | {x['enemy_deaths_near_blue_turret']:9d} / {y['enemy_deaths_near_blue_turret']:8d} | {x['own_deaths']:6d} / {y['own_deaths']:5d} | {x['cs']:3d} / {y['cs']:2d} | {x['cs_total']:5d} / {y['cs_total']:5d} | {x['enemy_wave_s']:7.0f} / {y['enemy_wave_s']:7.0f} | {x['champ_s']:5.0f} / {y['champ_s']:5.0f}")
