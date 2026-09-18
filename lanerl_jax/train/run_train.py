"""Run a training job and print the curve. `python -m lanerl_jax.train.run_train`."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import numpy as np

from .trainer import TrainConfig, make_train


DEFAULT_ROUTE_ARTIFACT = (Path(__file__).resolve().parents[2] / "data" /
                          "jax_routes" / "map1_garen_r35_o50_v2")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--rollout", type=int, default=128)
    ap.add_argument("--updates", type=int, default=30)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episode-s", type=float, default=600.0,
                    help="episode length in seconds. NOT the discount horizon "
                         "(PPOConfig.horizon_s); see TrainConfig.")
    ap.add_argument("--every", type=int, default=3)
    ap.add_argument("--route-artifact", type=Path, default=DEFAULT_ROUTE_ARTIFACT,
                    help="Map1 local-route artifact (default: %(default)s)")
    ap.add_argument(
        "--no-route-table", action="store_true",
        help="explicitly run the PATH-001 two-point approximation; intended "
             "only for comparison/debugging")
    ap.add_argument(
        "--time-steady", action="store_true",
        help="run the whole job TWICE to separate compile from steady state. "
             "Exact, and it doubles the cost -- `n_updates` is the scan length "
             "and therefore baked into the graph, so there is no cheaper honest "
             "way. Use it for a benchmark, not for a real training run.")
    a = ap.parse_args()

    cfg = TrainConfig(n_envs=a.envs, rollout_steps=a.rollout,
                      n_updates=a.updates, n_minibatches=a.minibatches,
                      episode_s=a.episode_s)
    n_dec = cfg.n_envs * cfg.rollout_steps * cfg.n_updates
    print(f"device {jax.devices()[0]}")
    print(f"{cfg.n_envs} envs x {cfg.rollout_steps} steps x {cfg.n_updates} "
          f"updates = {n_dec:,} env-decisions "
          f"({n_dec * 2:,} champion-decisions)")
    print(f"episode {cfg.episode_s:.0f}s ({cfg.episode_steps:,} decisions) | "
          f"discount horizon {cfg.ppo.horizon_s:.0f}s "
          f"(gamma {cfg.ppo.gamma:.6f}) | "
          f"{cfg.n_updates * cfg.rollout_steps / cfg.episode_steps:.1f} "
          f"episodes per env")

    route_table = terrain = None
    if not a.no_route_table:
        if not a.route_artifact.exists():
            ap.error(
                f"route artifact not found: {a.route_artifact}. Build it with: "
                "python -m lanerl_jax.data.local_route_artifact "
                f"--out {a.route_artifact} --radius 35 --offset-radius 50")
        from ..data.local_route_artifact import load_local_route_artifact
        from ..sim.terrain_jax import map1_terrain

        artifact = load_local_route_artifact(
            a.route_artifact, pathfinding_radius=35.0)
        route_table = artifact.as_jax()
        terrain = map1_terrain()
        print(f"local routes {artifact.manifest.source_count:,} source cells x "
              f"{artifact.manifest.table_shape[1]}² offsets "
              f"({artifact.next_hop.nbytes / 2**20:.1f} MiB)")
    else:
        print("WARNING: local routing disabled; Move uses the PATH-001 raw segment")

    train = jax.jit(make_train(cfg, route_table=route_table, terrain=terrain))
    t0 = time.perf_counter()
    out = train(jax.random.key(a.seed))
    jax.block_until_ready(out)
    first = time.perf_counter() - t0

    if a.time_steady:
        # Same shapes, so no recompile; the difference is the compile.
        t0 = time.perf_counter()
        jax.block_until_ready(train(jax.random.key(a.seed + 1)))
        steady = time.perf_counter() - t0
        print(f"first call {first:.1f}s (compile ~{first - steady:.1f}s) | "
              f"steady {steady:.1f}s -> {n_dec / steady:,.0f} env-decisions/s "
              f"END TO END, gradient step included")
    else:
        print(f"wall {first:.1f}s -> {n_dec / first:,.0f} env-decisions/s "
              f"including compile (pass --time-steady to separate them)")

    _, m = out
    print()
    cols = ("reward", "entropy", "approx_kl", "clip_frac", "value_loss",
            "lane_dist", "route_nonready", "cs_at_10min")
    print(f"{'upd':>5}" + "".join(f"{c:>12}" for c in cols))
    for i in range(0, cfg.n_updates, a.every):
        row = "".join(f"{float(np.asarray(m[c])[i]):>12.4f}" for c in cols)
        print(f"{i:>5}{row}")
    print()
    e0 = float(np.asarray(m["entropy"])[0])
    e1 = float(np.asarray(m["entropy"])[-1])
    print(f"entropy {e0:.3f} -> {e1:.3f} of a 14.099 uniform maximum "
          f"({100 * e1 / 14.099:.0f}%)")
    print(f"mean reward first/last update: "
          f"{float(np.asarray(m['reward'])[0]):+.5f} -> "
          f"{float(np.asarray(m['reward'])[-1]):+.5f}")

    # --- the two numbers the sampled table above cannot show ---------------
    #
    # cs_at_10min is NaN except in the one update whose rollout happens to
    # contain an episode boundary -- 18,000 decisions per episode against 128
    # per rollout, so roughly 1 update in 141. Printing every `--every`th row
    # samples past all of them: an 800-update run reports NaN in every printed
    # row while the number exists. Print the episodes themselves instead.
    cs = np.asarray(m["cs_at_10min"])
    done_at = np.flatnonzero(~np.isnan(cs))
    if len(done_at):
        print(f"\nCS@10min, per episode that ENDED ({len(done_at)} of them):")
        for i in done_at:
            print(f"  update {int(i):>5}   {float(cs[i]):.3f} CS per champion")
    else:
        print(f"\nNo episode finished: {cfg.n_updates} updates x "
              f"{cfg.rollout_steps} steps = "
              f"{cfg.n_updates * cfg.rollout_steps:,} decisions against an "
              f"{cfg.episode_steps:,}-decision episode. Run at least "
              f"{cfg.episode_steps // cfg.rollout_steps + 1} updates.")

    # lane_dist is dominated by WHERE in the episode each rollout falls: a
    # rollout just after a reset has both champions back at the fountain at
    # ~8,000. The minimum over the run is what says whether they ever arrive.
    ld = np.asarray(m["lane_dist"])
    print(f"\nlane distance: start {ld[0]:,.0f} -> min {ld.min():,.0f} "
          f"(0 = inside the lane corridor, ~8,000 = the fountain)")


if __name__ == "__main__":
    main()
