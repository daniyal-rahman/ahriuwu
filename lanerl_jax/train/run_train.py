"""Run a training job and print the curve. `python -m lanerl_jax.train.run_train`."""
from __future__ import annotations

import argparse
import time

import jax
import numpy as np

from .trainer import TrainConfig, make_train


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

    train = jax.jit(make_train(cfg))
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
    cols = ("reward", "entropy", "approx_kl", "clip_frac", "value_loss", "cs")
    print(f"{'upd':>5}" + "".join(f"{c:>11}" for c in cols))
    for i in range(0, cfg.n_updates, a.every):
        row = "".join(f"{float(np.asarray(m[c])[i]):>11.5f}" for c in cols)
        print(f"{i:>5}{row}")
    print()
    e0 = float(np.asarray(m["entropy"])[0])
    e1 = float(np.asarray(m["entropy"])[-1])
    print(f"entropy {e0:.3f} -> {e1:.3f} of a 14.099 uniform maximum "
          f"({100 * e1 / 14.099:.0f}%)")
    print(f"mean reward first/last update: "
          f"{float(np.asarray(m['reward'])[0]):+.5f} -> "
          f"{float(np.asarray(m['reward'])[-1]):+.5f}")


if __name__ == "__main__":
    main()
