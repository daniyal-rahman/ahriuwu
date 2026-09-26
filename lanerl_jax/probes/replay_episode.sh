#!/bin/bash
# Record one mirror episode of a checkpoint on the C# server and render both views.
#   CKPT=<...msgpack> OUT=lanerl_jax/runs/EVAL/replay_<tag> LABEL="..." [PORT=23500] replay_episode.sh
set -x
cd /mnt/nfs/projects/ahriuwu-lanerl-jax
export PYTHONUNBUFFERED=1 JAX_PLATFORMS=cpu PYTHONPATH=/mnt/nfs/projects/ahriuwu-lanerl-jax LANERL_VENDOR_ROOT=/mnt/nfs/projects/lanerl-vendor
: "${CKPT:?}" "${OUT:?}"; LABEL=${LABEL:-replay}; PORT=${PORT:-23500}
rm -rf "$OUT"
.venv-jax/bin/python -m lanerl_jax.train.server_eval "$CKPT" --red policy --start-near-wave --step-ticks 6 --seed 0 --port-base $PORT --server-dir /mnt/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/ClickV3/net6.0 --out "$OUT" && echo EVAL-OK
.venv-jax/bin/python -m lanerl_jax.parity.render_recording "$OUT" --out "$OUT/trace.npz" --label "$LABEL" && echo CONVERT-OK
.venv-jax/bin/python -m lanerl_jax.replay_render "$OUT/trace.npz" --out-dir "$OUT/map" --view map --video && echo MAP-OK
.venv-jax/bin/python -m lanerl_jax.replay_render "$OUT/trace.npz" --out-dir "$OUT/combat" --view combat --video && echo COMBAT-OK
