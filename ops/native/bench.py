"""Throughput of the native lane-slice tick: env-ticks/s over a batch of top-lane envs (champions idle).

Starts every env from the JAX world at 0:00 (or after ``--warm`` native ticks, so waves are on the map), then
times ``--ticks`` ticks of all envs per thread count.

    python -m ops.native.bench --envs 256 --ticks 1800 --threads 1 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "native" / "python"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--ticks", type=int, default=1800)
    ap.add_argument("--warm", type=int, default=1800, help="native ticks before timing (0:00 -> 1:00)")
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    args = ap.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np

    from lanerl_jax.modern import world as MS
    from ops.modern.golden import build
    import lanesim as LS
    cfg = build("top")
    world = LS.NativeWorld(cfg)
    env = LS.env_from_state(world, MS.init_state(cfg))
    for _ in range(args.warm):
        world.step(env)
    print(json.dumps({"start_game_s": float(env["t"][0]), "minions": int(np.sum((env["kind"] == 2) & (env["alive"] > 0)))}))
    one = LS.Batch(world, env, 1)                           # runs on this thread: per-phase profile
    LS.profile()
    t0 = time.perf_counter()
    one.run(args.ticks, 1)
    sec = time.perf_counter() - t0
    prof = LS.profile()
    tot = sum(v for k, v in prof.items() if "." not in k)
    print(json.dumps({"profile_us_per_tick": {k: round(v / args.ticks * 1e6, 2) for k, v in prof.items()},
                      "phases_us_per_tick": round(tot / args.ticks * 1e6, 2),
                      "wall_us_per_tick": round(sec / args.ticks * 1e6, 2)}), flush=True)
    for th in args.threads:
        batch = LS.Batch(world, env, args.envs)
        batch.run(30, th)                                   # first touch / thread start
        t0 = time.perf_counter()
        over = batch.run(args.ticks, th)
        dt = time.perf_counter() - t0
        end = batch.get(0)
        print(json.dumps({"threads": th, "envs": args.envs, "ticks": args.ticks, "seconds": round(dt, 3),
                          "env_ticks_per_s": round(args.envs * args.ticks / dt, 1),
                          "us_per_env_tick_per_thread": round(dt * th / (args.envs * args.ticks) * 1e6, 2),
                          "overflow": over.tolist(), "end_game_s": float(end["t"][0]),
                          "minions_end": int(np.sum((end["kind"] == 2) & (end["alive"] > 0))),
                          "env_kb": round(LS.lib().ls_batch_env_bytes(batch.ptr) / 1024, 1)}), flush=True)


if __name__ == "__main__":
    main()
