"""Benchmark the on-device trainer (trainer.make_train): decisions/s and peak
device memory for one env count. `python -m ...vec_bench <n_envs> [rollout] [updates]`."""
import sys, time, json, jax, jax.numpy as jnp
from lanerl_jax.train.trainer import TrainConfig, make_train
from lanerl_jax.train.ppo import PPOConfig
from lanerl_jax.sim.config import SimConfig, DEFAULT_ROUTE_ARTIFACT
n = int(sys.argv[1]); T = int(sys.argv[2]) if len(sys.argv) > 2 else 128; U = int(sys.argv[3]) if len(sys.argv) > 3 else 3
sim = SimConfig.training(route_artifact=DEFAULT_ROUTE_ARTIFACT).replace(step_ticks=6)
cfg = TrainConfig(n_envs=n, rollout_steps=T, n_updates=U, n_minibatches=4, episode_s=600.0, ppo=PPOConfig.standard(decision_hz=10.0))
train = make_train(cfg, sim_config=sim)
dev = jax.local_devices()[0]
t0 = time.time(); runner = train.initial_runner(jax.random.key(0)); jax.block_until_ready(runner.params); t_init = time.time() - t0
t0 = time.time(); runner, m = train.run_chunk(runner, 1); jax.block_until_ready(runner.params); t_compile = time.time() - t0
t0 = time.time(); runner, m = train.run_chunk(runner, U); jax.block_until_ready(runner.params); t_run = time.time() - t0
dec = U * T * n * 2
stats = dev.memory_stats() or {}
print(json.dumps(dict(n_envs=n, rollout=T, updates=U, backend=jax.default_backend(), init_s=round(t_init, 1), compile_s=round(t_compile, 1), run_s=round(t_run, 2),
                      decisions_per_s=round(dec / t_run), steps_per_s_per_env=round(U * T / t_run, 1),
                      peak_gb=round(stats.get("peak_bytes_in_use", 0) / 1e9, 2), in_use_gb=round(stats.get("bytes_in_use", 0) / 1e9, 2))), flush=True)
