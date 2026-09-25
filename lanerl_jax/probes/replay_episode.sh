#!/bin/bash
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu PYTHONPATH=/mnt/nfs/projects/ahriuwu-lanerl-jax LANERL_VENDOR_ROOT=/mnt/nfs/projects/lanerl-vendor
.venv-jax/bin/python -m lanerl_jax.train.server_eval /mnt/nfs/projects/ahriuwu-lanerl-jax/lanerl_jax/runs/server_train/mirror-wave-s0/server-farm-s0-20260925-195150-1716b293/eval_u1080.msgpack --red policy --start-near-wave --step-ticks 6 --seed 0 --port-base 22700 --server-dir /mnt/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/DeadProbe/net6.0 --out lanerl_jax/runs/EVAL/replay_u1080 && echo EVAL-OK
.venv-jax/bin/python -m lanerl_jax.parity.render_recording lanerl_jax/runs/EVAL/replay_u1080 --out lanerl_jax/runs/EVAL/replay_u1080/trace.npz --label "E01 u1080 mirror, C# server, 10 Hz" && echo CONVERT-OK
.venv-jax/bin/python -m lanerl_jax.replay_render lanerl_jax/runs/EVAL/replay_u1080/trace.npz --out-dir lanerl_jax/runs/EVAL/replay_u1080/map --view map --video && echo MAP-OK
.venv-jax/bin/python -m lanerl_jax.replay_render lanerl_jax/runs/EVAL/replay_u1080/trace.npz --out-dir lanerl_jax/runs/EVAL/replay_u1080/combat --view combat --video && echo COMBAT-OK
