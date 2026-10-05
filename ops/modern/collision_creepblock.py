"""Creep-block proxy of the modern world tick (docs/modern/COLLISION.md §3), sim side.

Garen walks up and down the top lane between points 1100 u outside the two outer turrets, from
90 s for ``--seconds`` of game time, through both waves; Jax idles in his fountain. Per tick we
record Garen's position and the distance to the nearest live minion, then report the same
statistics as the replay measurement (``ops/modern/collision_replay.py``):

  ratio      displacement speed over a 0.2 s / 0.5 s window / nominal speed (mode of free walking)
  path       path-length speed over 0.25 s windows / nominal (the replay tool's speed)
  stalls     runs of path ratio < 0.6, per minute of qualifying time
  detour     fraction of 0.45 s windows with chord / path length < 0.9

split by "near minions" (a minion within 300 u) vs "free". Qualifying ticks: alive, >= 300 u from
the turn-around point. Run on the desktop through Slurm (one full-tick compile, ~8 GB):

    PYTHONPATH=<snapshot> python ops/modern/collision_creepblock.py --seconds 240 --json out.json
"""
from __future__ import annotations

import argparse
import json

import jax
import jax.numpy as jnp
import numpy as np


def build():
    from lanerl_jax.modern.tests import world_harness as H
    return H.world()


def endpoints(cfg, margin=1000.0):
    """Top-lane points ``margin`` u (along the lane) inside the two outer top turrets."""
    from lanerl_jax.modern.core import types as W
    lane = np.asarray(cfg.lane_path, float)
    seg = np.diff(lane, axis=0)
    seglen = np.hypot(seg[:, 0], seg[:, 1])
    cum = np.concatenate([[0.0], np.cumsum(seglen)])

    def project(p):
        t = np.clip(np.einsum("ij,ij->i", p - lane[:-1], seg) / np.maximum(seglen ** 2, 1e-6), 0, 1)
        q = lane[:-1] + t[:, None] * seg
        i = int(np.argmin(np.hypot(*(q - p).T)))
        return cum[i] + t[i] * seglen[i]

    def at(a):
        i = int(np.clip(np.searchsorted(cum, a) - 1, 0, len(seg) - 1))
        return lane[i] + (a - cum[i]) / max(seglen[i], 1e-6) * seg[i]
    kind, team = np.asarray(cfg.unit_kind), np.asarray(cfg.unit_team)
    ux, uy = np.asarray(cfg.unit_x, float), np.asarray(cfg.unit_y, float)
    mid = at(cum[-1] / 2)
    arcs = []
    for t in (0, 1):
        tur = np.flatnonzero((kind == W.KIND_TURRET) & (team == t))
        j = tur[np.argmin(np.hypot(ux[tur] - mid[0], uy[tur] - mid[1]))]
        arcs.append(project(np.array([ux[j], uy[j]])))
    return np.stack([at(arcs[0] + margin), at(arcs[1] - margin)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=240.0)
    ap.add_argument("--json", default=None)
    ap.add_argument("--set", nargs="*", default=[], help="collision constant overrides NAME=VALUE")
    args = ap.parse_args()
    if args.set:
        from lanerl_jax.modern import collision as UC
        for kv in args.set:
            k, v = kv.split("=")
            setattr(UC, k, type(getattr(UC, k))(eval(v)))
    from lanerl_jax.modern import world as MS
    from lanerl_jax.modern.core import types as W
    cfg = build()
    ends = jnp.asarray(endpoints(cfg), jnp.float32)
    chunk = 300

    @jax.jit
    def warm(s):
        return jax.lax.fori_loop(0, 90 * 30, lambda _, s: MS.step(s, MS.no_orders(), cfg)[0], s)

    @jax.jit
    def run(carry):
        def body(c, _):
            s, tgt = c
            g = ends[tgt]
            o = MS.no_orders()._replace(move=jnp.asarray([True, False]), move_x=jnp.asarray([g[0], 0.0]),
                                        move_y=jnp.asarray([g[1], 0.0]))
            s, _ = MS.step(s, o, cfg)
            dg = jnp.hypot(s.x[0] - g[0], s.y[0] - g[1])
            tgt = jnp.where(dg < 60.0, 1 - tgt, tgt)
            mn = (s.kind == W.KIND_MINION) & s.alive
            dm = jnp.min(jnp.where(mn, jnp.hypot(s.x - s.x[0], s.y - s.y[0]), 1e9))
            return (s, tgt), jnp.stack([s.t, s.x[0], s.y[0], s.alive[0].astype(jnp.float32), dm,
                                        jnp.minimum(dg, jnp.hypot(s.x[0] - ends[1 - tgt][0],
                                                                  s.y[0] - ends[1 - tgt][1]))])
        return jax.lax.scan(body, carry, None, length=chunk)

    s = warm(MS.init_state(cfg))
    carry, rows = (s, jnp.int32(1)), []
    for _ in range(int(args.seconds * 30 // chunk)):
        carry, r = run(carry)
        rows.append(np.asarray(r))
    rec = np.concatenate(rows)
    out = analyze(rec, hz=30.0)
    out["ends"] = np.asarray(ends).tolist()
    print(json.dumps(out, indent=1))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f)
        np.save(args.json.replace(".json", "_trace.npy"), rec)


def analyze(rec, hz):
    t, x, y, alive, dmin, dend = rec.T
    n = len(t)
    w2, w5, w45 = round(0.2 * hz), round(0.5 * hz), round(0.45 * hz)
    step = np.hypot(np.diff(x), np.diff(y)) * hz
    ok1 = (alive[1:] > 0) & (alive[:-1] > 0)
    nominal = float(np.median(step[ok1 & (dmin[1:] > 400) & (dend[1:] > 300) & (step > 100)]))

    def win(w):
        i = np.arange(n - w)
        sp = np.hypot(x[i + w] - x[i], y[i + w] - y[i]) * hz / w
        path = np.array([np.sum(np.hypot(np.diff(x[j:j + w + 1]), np.diff(y[j:j + w + 1]))) for j in i])
        ok = (alive[i] > 0) & (alive[i + w] > 0) & (np.minimum(dend[i], dend[i + w]) > 300)
        return sp / nominal, path, np.hypot(x[i + w] - x[i], y[i + w] - y[i]), ok, dmin[i]
    res = {"nominal_speed": nominal, "seconds": float(t[-1] - t[0])}
    for name, w in (("0.2s", w2), ("0.5s", w5)):
        r, _, _, ok, dm = win(w)
        for cond, m in (("near", ok & (dm < 300)), ("free", ok & (dm >= 300)), ("all", ok)):
            v = r[m]
            res[f"ratio_{name}_{cond}"] = {"n": int(m.sum()), **{f"p{q}": float(np.percentile(v, q)) if len(v) else None
                                                                for q in (1, 5, 10, 25, 50)}}
    # Replay-comparable: path-length speed over ~0.25 s windows (ops/modern/collision_replay.py).
    w25 = round(0.25 * hz)
    _, path, _, ok, dm = win(w25)
    rp = path * hz / w25 / nominal
    res["path_ratio_0.25s_all"] = {"n": int(ok.sum()), **{f"p{q}": float(np.percentile(rp[ok], q))
                                                         for q in (1, 5, 10, 25, 50)},
                                   "frac_lt_0.6": float((rp[ok] < 0.6).mean())}
    r = rp                                                       # stalls on the replay's path-speed windows
    low = (r < 0.6) & ok
    for cond, m in (("near", dm < 300), ("all", np.ones_like(ok))):
        q = ok & m
        lm = low & m
        runs, cur = [], 0
        for v in lm:
            if v:
                cur += 1
            elif cur:
                runs.append(cur); cur = 0
        runs = [c for c in runs if c >= 1]                       # each start is a 0.2 s window below 60 %
        minutes = q.sum() / hz / 60.0
        res[f"stalls_{cond}"] = {"per_min": len(runs) / max(minutes, 1e-9), "minutes": float(minutes),
                                 "median_s": float(np.median(runs) / hz + w25 / hz) if runs else None,
                                 "p90_s": float(np.percentile(runs, 90) / hz + w25 / hz) if runs else None}
    _, path, chord, ok, dm = win(w45)
    st = chord / np.maximum(path, 1e-6)
    moving = path > 0.3 * nominal * 0.45
    for cond, m in (("near", ok & moving & (dm < 300)), ("free", ok & moving & (dm >= 300))):
        res[f"detour_{cond}"] = {"n": int(m.sum()), "frac_straightness_lt_0.9": float((st[m] < 0.9).mean())
                                 if m.any() else None}
    return res


if __name__ == "__main__":
    main()
