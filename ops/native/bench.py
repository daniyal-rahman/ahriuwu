"""Throughput of the native tick: env-ticks/s over a batch of top-lane envs.

Same protocol as ops/modern/bench: every env starts from the JAX world at 0:00 with its own key
(``split(PRNGKey(0), envs)``), runs ``--warm`` ticks (waves on the map), then ``--ticks`` ticks of all envs are
timed per thread count. A single env is profiled per phase first.

    python -m ops.native.bench --world bench --orders 3 --envs 256 --ticks 1800 --threads 1 8
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
    ap.add_argument("--warm", type=int, default=1200, help="ticks before timing (as ops/modern/bench --warm-ticks)")
    ap.add_argument("--threads", type=int, nargs="+", default=[1])
    ap.add_argument("--orders", type=int, default=0,
                    help="bit 0: the JAX bench's scripted orders (else none); bit 1: the full tick with champions")
    ap.add_argument("--world", choices=("golden", "bench"), default="golden",
                    help="golden top lane, or ops/modern/bench --allowlist top lane (the JAX benchmark's world)")
    ap.add_argument("--sections", type=int, default=0, help="print the N slowest named sections of the profiled env")
    args = ap.parse_args()
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import numpy as np

    from lanerl_jax.modern import world as MS
    from ops.modern.golden import build
    import lanesim as LS
    if args.world == "bench":
        from ops.modern.bench import build_world
        cfg, _ = build_world(argparse.Namespace(fog="rays", no_jungle=True, no_objectives=True, lanes=[2],
                                                packet_capacity=0, allowlist=True))
    else:
        cfg = build("top")
    world = LS.NativeWorld(cfg)
    import jax
    s0 = MS.init_state(cfg)
    keys = np.asarray(jax.random.split(jax.random.PRNGKey(0), args.envs))

    def batch(n: int) -> "LS.Batch":
        """``n`` envs at 0:00 with the JAX bench's keys, warmed for ``--warm`` ticks on every core."""
        env = LS.env_from_state(world, s0)
        b = LS.Batch(world, env, n, args.orders)
        for k in range(n):
            env["key"][:] = keys[k]
            b.set(k, env)
        b.run(args.warm, 0)
        return b

    one = batch(1)                                                       # runs on this thread: per-phase profile
    env = one.get(0)
    print(json.dumps({"start_game_s": float(env["t"][0]), "minions": int(np.sum((env["kind"] == 2) & (env["alive"] > 0)))}))
    LS.profile()
    t0 = time.perf_counter()
    one.run(args.ticks, 1)
    sec = time.perf_counter() - t0
    prof = LS.profile()
    if args.sections:
        LS.lib().ls_prof_enable(1)
        LS.sections()
        one.run(args.ticks, 1)
        LS.lib().ls_prof_enable(0)
        top = sorted(LS.sections().items(), key=lambda kv: -kv[1][0])[:args.sections]
        for name, (s, calls) in top:
            print(json.dumps({"section": name, "us_per_tick": round(s / args.ticks * 1e6, 2),
                              "calls_per_tick": round(calls / args.ticks, 2)}), flush=True)
    tot = sum(v for k, v in prof.items() if "." not in k)
    print(json.dumps({"profile_us_per_tick": {k: round(v / args.ticks * 1e6, 2) for k, v in prof.items()},
                      "phases_us_per_tick": round(tot / args.ticks * 1e6, 2),
                      "wall_us_per_tick": round(sec / args.ticks * 1e6, 2)}), flush=True)
    for th in args.threads:
        b = batch(args.envs)
        t0 = time.perf_counter()
        over = b.run(args.ticks, th)
        dt = time.perf_counter() - t0
        end = b.get(0)
        print(json.dumps({"threads": th, "envs": args.envs, "ticks": args.ticks, "seconds": round(dt, 3),
                          "env_ticks_per_s": round(args.envs * args.ticks / dt, 1),
                          "us_per_env_tick_per_thread": round(dt * th / (args.envs * args.ticks) * 1e6, 2),
                          "overflow": over.tolist(), "end_game_s": float(end["t"][0]),
                          "minions_end": int(np.sum((end["kind"] == 2) & (end["alive"] > 0))),
                          "env_kb": round(LS.lib().ls_batch_env_bytes(b.ptr) / 1024, 1)}), flush=True)

if __name__ == "__main__":
    main()
