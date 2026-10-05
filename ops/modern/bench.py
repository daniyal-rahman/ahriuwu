"""Benchmark the modern world tick (``world.tick.step``), single and vmapped.

TOOL (MODERN-017). Builds the Garen-vs-Jax lane world, replicates the initial
state over B environments (independent PRNG keys), and times ``lax.scan`` of
T ticks of ``step`` under ``jit(vmap(...))`` with scripted "walk to lane and
attack the nearest enemy" orders computed inside the scan (no host loop).

    python -m ops.modern.bench --envs 1 64 512 --ticks 300 [--fog rays|fast|off]
        [--no-jungle] [--no-objectives] [--lanes 0 1 2]

Prints one JSON line per batch size: compile seconds, steady seconds per
tick, env-ticks per second, packet/missile overflow and the peak valid
packets per tick (main and follow-up pass; capacities 512 / 256).
"""
from __future__ import annotations

import argparse
import json
import time

import jax
import jax.numpy as jnp


def scripted_orders(s, lane_mid):
    from lanerl_jax.modern import world as MS
    from lanerl_jax.modern.core import types as W
    c = 2
    o = MS.no_orders()
    d = jnp.sqrt((s.x[None, :] - s.x[:c, None]) ** 2 + (s.y[None, :] - s.y[:c, None]) ** 2)
    enemy = (s.team[None, :] != s.team[:c, None]) & s.alive[None, :] & (s.kind[None, :] != W.KIND_NONE) \
        & (s.kind[None, :] != W.KIND_NEXUS) & (s.kind[None, :] != W.KIND_INHIBITOR)
    near = jnp.where(enemy & (d < 700.0), d, jnp.inf)
    tgt = jnp.argmin(near, axis=1).astype(jnp.int32)
    has = jnp.isfinite(jnp.min(near, axis=1))
    return o._replace(attack=jnp.where(has, tgt, -1), move=~has,
                      move_x=jnp.broadcast_to(lane_mid[0], (c,)), move_y=jnp.broadcast_to(lane_mid[1], (c,)))


def add_world_args(ap: argparse.ArgumentParser) -> None:
    """World-variant flags shared with ``ops.modern.profile_tick`` (ablations)."""
    ap.add_argument("--fog", choices=("rays", "fast", "off"), default="rays")
    ap.add_argument("--no-fog", action="store_true", help="alias for --fog off")
    ap.add_argument("--no-jungle", action="store_true")
    ap.add_argument("--no-objectives", action="store_true")
    ap.add_argument("--lanes", type=int, nargs="+", default=[0, 1, 2])


def build_world(args):
    """``(cfg, run)``: the Garen-vs-Jax world for ``args`` and ``run(state, ticks) -> (state, overflow)``."""
    from lanerl_jax.modern import world as MS
    from lanerl_jax.modern.runes import catalog as RD
    from lanerl_jax.modern.world import config as MW
    lo = (MW.Loadout("Garen", items=(1055, 2003), rune_page=RD.GAREN_DEFAULT_PAGE),
          MW.Loadout("Jax", items=(1055, 2003), rune_page=RD.RunePage(
              RD.PRECISION, 8010, (9111, 9104, 8299), RD.RESOLVE, (8444, 8242), (5005, 5008, 5001))))
    fog = False if (args.no_fog or args.fog == "off") else args.fog
    cfg = MW.build_config(lo, fog=fog, lanes=tuple(args.lanes), jungle=not args.no_jungle,
                          objectives=not args.no_objectives)
    lane = cfg.lane_path
    lane_mid = lane[lane.shape[0] // 2]

    def run(s, ticks):
        def body(s, _):
            s, e = MS.step(s, scripted_orders(s, lane_mid), cfg)
            used = (jnp.sum(e.report.packets.valid), jnp.sum(e.follow_up.packets.valid))
            return s, (e.packet_overflow, e.missile_overflow, used)
        return jax.lax.scan(body, s, None, length=ticks)
    return cfg, run


def init_batch(cfg, b: int):
    from lanerl_jax.modern import world as MS
    s0 = MS.init_state(cfg)
    batch = jax.tree.map(lambda a: jnp.broadcast_to(a, (b,) + jnp.shape(a)), s0)
    return batch._replace(key=jax.random.split(jax.random.PRNGKey(0), b))


def warm_up(timed, batch, warm_ticks: int, ticks: int):
    """Run ``timed`` (``ticks`` per call) until ``warm_ticks`` have passed: one compile, waves on the map."""
    out = None
    for _ in range(max(1, -(-warm_ticks // ticks))):
        batch, out = timed(batch)
    jax.block_until_ready(batch.t)
    return batch, out


def world_label(args) -> dict:
    return {"fog": "off" if args.no_fog else args.fog, "jungle": not args.no_jungle,
            "objectives": not args.no_objectives, "lanes": list(args.lanes)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--envs", type=int, nargs="+", default=[1, 64])
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--warm-ticks", type=int, default=1200, help="ticks before timing (waves on the map)")
    add_world_args(ap)
    args = ap.parse_args()
    from lanerl_jax.modern.jax_cache import enable_compile_cache
    enable_compile_cache()
    cfg, run = build_world(args)
    print(json.dumps({"backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
                      **world_label(args)}), flush=True)
    for b in args.envs:
        timed = jax.jit(jax.vmap(lambda s: run(s, args.ticks)))
        t0 = time.time()
        batch, _ = warm_up(timed, init_batch(cfg, b), args.warm_ticks, args.ticks)
        t_warm = time.time() - t0
        t0 = time.time()
        out, (po, mo, _) = timed(batch)
        jax.block_until_ready(out.t)
        t_compile_run = time.time() - t0
        t0 = time.time()
        out, (po, mo, (pm, pf)) = timed(out)
        jax.block_until_ready(out.t)
        steady = time.time() - t0
        print(json.dumps({"envs": b, "ticks": args.ticks, "warm_compile_and_run_s": round(t_warm, 2),
                          "timed_second_call_s": round(t_compile_run, 2), "steady_s": round(steady, 3),
                          "s_per_tick": steady / args.ticks, "env_ticks_per_s": b * args.ticks / steady,
                          "game_time_s": float(out.t[0]), "packet_overflow": int(po.max()),
                          "missile_overflow": int(mo.max()), "packets_max": int(pm.max()),
                          "follow_up_packets_max": int(pf.max())}), flush=True)


if __name__ == "__main__":
    main()
