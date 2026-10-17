"""Benchmark the modern world tick (``world.tick.step``) under ``jit(vmap(lax.scan))`` (MODERN-017).

Garen-vs-Jax (``--champions``) lane world replicated over B environments (independent PRNG keys); scripted "walk to lane and
attack the nearest enemy" orders are computed inside the scan.

    python -m ops.modern.bench --envs 1 64 512 --ticks 300 [--fog rays|fast|off] [--no-jungle]
        [--no-objectives] [--lanes 0 1 2] [--champions Garen Jax] [--roster all | NAME ...]

Use --orders abilities --warm-ticks 5400 for later lane combat, including ability attempts.
Prints synchronized sample times and their median, tick/game-time throughput, timed game window,
final minion count, process peak device memory, overflow and peak packet counts. A frozen game or
any overflow invalidates the measurement. Policy inference and PPO are outside this tool.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time

import jax
import jax.numpy as jnp


def scripted_orders(s, lane_mid, abilities=False):
    from lanerl_jax.modern import world as MS
    from lanerl_jax.modern.core import types as W
    c = 2
    d = jnp.sqrt((s.x[None, :] - s.x[:c, None]) ** 2 + (s.y[None, :] - s.y[:c, None]) ** 2)
    enemy = (s.team[None, :] != s.team[:c, None]) & s.alive[None, :] & (s.kind[None, :] != W.KIND_NONE) \
        & (s.kind[None, :] != W.KIND_NEXUS) & (s.kind[None, :] != W.KIND_INHIBITOR)
    near = jnp.where(enemy & (d < 700.0), d, jnp.inf)
    has = jnp.isfinite(jnp.min(near, axis=1))
    orders = MS.no_orders()._replace(attack=jnp.where(has, jnp.argmin(near, axis=1).astype(jnp.int32), -1), move=~has,
                                   move_x=jnp.broadcast_to(lane_mid[0], (c,)),
                                   move_y=jnp.broadcast_to(lane_mid[1], (c,)))
    if abilities:
        holder = jnp.arange(c, dtype=jnp.int32)
        target = jnp.argmin(jnp.where(enemy, d, jnp.inf), axis=1).astype(jnp.int32)
        attempt = (s.tick % 30 == holder * 15) & jnp.any(enemy, axis=1)
        orders = orders._replace(cast_slot=jnp.where(attempt, (s.tick // 30 + holder) % 4, -1),
                                 cast_target=target, cast_x=s.x[target], cast_y=s.y[target])
    return orders


def add_world_args(ap: argparse.ArgumentParser) -> None:
    """World-variant flags shared by bench, profile_tick and perf_guard."""
    ap.add_argument("--orders", choices=("lane", "abilities"), default="lane",
                    help="scripted movement/attacks, optionally attempting each ability in rotation")
    ap.add_argument("--fog", choices=("rays", "fast", "off"), default="rays")
    ap.add_argument("--no-jungle", action="store_true")
    ap.add_argument("--no-objectives", action="store_true")
    ap.add_argument("--lanes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--packet-capacity", type=int, default=0, help="damage packets per tick (0: two per unit)")
    ap.add_argument("--allowlist", action="store_true", help="restrict each champion to its top-lane allow-list")
    ap.add_argument("--lane-structures", action="store_true", help="only the spawning lanes' structures")
    ap.add_argument("--champions", nargs=2, default=["Garen", "Jax"], help="blue and red champion")
    ap.add_argument("--roster", nargs="+", default=[], help="further kits to compile (a champion pool), or 'all'")


def build_world(args):
    """``(cfg, run)``: the ``--champions`` world (Garen vs Jax) for ``args`` and ``run(state, ticks) -> (state,
    per-tick stats)``. Champions other than Garen and Jax take Riot's recommended top page (``data.loadouts``)."""
    from lanerl_jax.modern import world as MS
    from lanerl_jax.modern.runes import catalog as RD
    from lanerl_jax.modern.data.loadouts import allowed_items, rune_pages
    from lanerl_jax.modern.world import config as MW
    shop = (lambda name: allowed_items(name)) if args.allowlist else (lambda name: ())      # noqa: E731
    pages = {"Garen": RD.GAREN_DEFAULT_PAGE, "Jax": RD.RunePage(
        RD.PRECISION, 8010, (9111, 9104, 8299), RD.RESOLVE, (8444, 8242), (5005, 5008, 5001))}
    lo = tuple(MW.Loadout(nm, items=(1055, 2003), rune_page=pages[nm] if nm in pages else rune_pages(nm)[0],
                          allowed_items=shop(nm)) for nm in args.champions)
    cfg = MW.build_config(lo, fog=False if args.fog == "off" else args.fog, lanes=tuple(args.lanes),
                          jungle=not args.no_jungle, objectives=not args.no_objectives,
                          packet_capacity=args.packet_capacity, lane_structures=args.lane_structures,
                          roster="all" if args.roster == ["all"] else tuple(args.roster))
    lane_mid = cfg.lane_path[cfg.lane_path.shape[0] // 2]

    def run(s, ticks):
        def body(s, _):
            s, e = MS.step(s, scripted_orders(s, lane_mid, getattr(args, "orders", "lane") == "abilities"), cfg)
            used = (jnp.sum(e.report.packets.valid), jnp.sum(e.follow_up.packets.valid))
            return s, (e.packet_overflow, e.missile_overflow, e.ray_overflow, e.item_overflow, used)
        return jax.lax.scan(body, s, None, length=ticks)
    return cfg, run


def init_batch(cfg, b: int):
    from lanerl_jax.modern import world as MS
    batch = jax.tree.map(lambda a: jnp.broadcast_to(a, (b,) + jnp.shape(a)), MS.init_state(cfg))
    return batch._replace(key=jax.random.split(jax.random.PRNGKey(0), b))


def warm_up(timed, batch, warm_ticks: int, ticks: int):
    """Call ``timed`` (``ticks`` per call) until ``warm_ticks`` have passed: compiles, and waves reach the map."""
    out = None
    for _ in range(max(1, -(-warm_ticks // ticks))):
        batch, out = timed(batch)
    jax.block_until_ready(batch.t)
    return batch, out


def world_label(args) -> dict:
    return {"orders": getattr(args, "orders", "lane"), "fog": args.fog, "jungle": not args.no_jungle, "objectives": not args.no_objectives,
            "lanes": list(args.lanes), **({"packet_capacity": args.packet_capacity} if args.packet_capacity else {}),
            **({"allowlist": True} if args.allowlist else {}),
            **({"lane_structures": True} if args.lane_structures else {}),
            **({"champions": args.champions} if args.champions != ["Garen", "Jax"] else {}),
            **({"roster": args.roster} if args.roster else {})}


def sample_summary(seconds, envs, ticks, dt):
    """Normalize measured world ticks; no policy or learner cost is implied."""
    import math
    if not seconds or any(not math.isfinite(x) or x <= 0 for x in seconds):
        raise ValueError("sample durations must be positive and finite")
    if envs < 1 or ticks < 1 or not math.isfinite(dt) or dt <= 0:
        raise ValueError("environment/tick counts and tick duration must be positive")
    median = statistics.median(seconds)
    rate = envs * ticks / median
    return dict(samples_s=seconds, samples=len(seconds), steady_s=median, min_s=min(seconds), max_s=max(seconds),
                s_per_tick=median / ticks, env_ticks_per_s=rate, simulated_game_seconds_per_wall_s=rate * dt)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--envs", type=int, nargs="+", default=[1, 64])
    ap.add_argument("--ticks", type=int, default=300)
    ap.add_argument("--warm-ticks", type=int, default=1200, help="world ticks before timing")
    ap.add_argument("--samples", type=int, default=5, help="consecutive synchronized calls; report median")
    add_world_args(ap)
    args = ap.parse_args()
    if min(args.envs) < 1 or args.ticks < 1 or args.samples < 1 or args.warm_ticks < 0:
        ap.error("envs/ticks/samples must be positive and warm-ticks nonnegative")
    from lanerl_jax.modern.jax_cache import enable_compile_cache
    from lanerl_jax.modern.core import types as W
    enable_compile_cache()
    cfg, run = build_world(args)
    print(json.dumps({"backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
                      **world_label(args)}), flush=True)
    for b in args.envs:
        timed = jax.jit(jax.vmap(lambda s: run(s, args.ticks)))
        t0 = time.perf_counter()
        batch, _ = warm_up(timed, init_batch(cfg, b), args.warm_ticks, args.ticks)
        t_warm = time.perf_counter() - t0
        start_s = float(batch.t[0])
        samples, maxima = [], [0] * 6
        for _ in range(args.samples):
            t0 = time.perf_counter()
            batch, stats = jax.block_until_ready(timed(batch))
            samples.append(time.perf_counter() - t0)
            po, mo, ro, io, (pm, pf) = stats
            maxima = [max(a, int(v.max())) for a, v in zip(maxima, (po, mo, ro, io, pm, pf))]
        frozen = int(jnp.sum(batch.game_over))
        mem = jax.devices()[0].memory_stats() or {}
        row = dict(envs=b, ticks=args.ticks, warm_ticks=args.warm_ticks,
                   warm_compile_and_run_s=t_warm, timed_start_game_s=start_s, game_time_s=float(batch.t[0]),
                   packet_overflow=maxima[0], missile_overflow=maxima[1], ray_overflow=maxima[2],
                   item_overflow=maxima[3], packets_max=maxima[4], follow_up_packets_max=maxima[5],
                   frozen_envs=frozen, alive_minions_mean=float(jnp.sum(batch.alive & (batch.kind == W.KIND_MINION)) / b),
                   process_peak_device_bytes=mem.get("peak_bytes_in_use"),
                   **sample_summary(samples, b, args.ticks, cfg.dt))
        print(json.dumps(row), flush=True)
        if any(maxima[:4]) or frozen:
            raise RuntimeError("invalid sim timing: overflow or a frozen completed game")


if __name__ == "__main__":
    main()
