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
    ap.add_argument("--every", type=int, default=3)
    a = ap.parse_args()

    cfg = TrainConfig(n_envs=a.envs, rollout_steps=a.rollout,
                      n_updates=a.updates, n_minibatches=a.minibatches)
    n_dec = cfg.n_envs * cfg.rollout_steps * cfg.n_updates
    print(f"device {jax.devices()[0]}")
    print(f"{cfg.n_envs} envs x {cfg.rollout_steps} steps x {cfg.n_updates} "
          f"updates = {n_dec:,} env-decisions "
          f"({n_dec * 2:,} champion-decisions)")

    train = jax.jit(make_train(cfg))
    t0 = time.perf_counter()
    out = train(jax.random.key(a.seed))
    jax.block_until_ready(out)
    first = time.perf_counter() - t0

    # Second call: same shapes, so no recompile. The difference is the compile.
    # `n_updates` is the scan length and therefore baked into the graph, so the
    # only honest way to separate the two is to run it twice.
    t0 = time.perf_counter()
    again = train(jax.random.key(a.seed + 1))
    jax.block_until_ready(again)
    steady = time.perf_counter() - t0

    _, m = out
    print(f"first call {first:.1f}s (compile ~{first - steady:.1f}s) | "
          f"steady {steady:.1f}s -> {n_dec / steady:,.0f} env-decisions/s "
          f"END TO END, gradient step included")
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
