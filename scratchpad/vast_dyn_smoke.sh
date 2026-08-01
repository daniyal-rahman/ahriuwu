#!/bin/bash
# One-shot Vast smoke for the dynamics DDP port, mirroring the tokenizer workflow
# (docs/VAST.md §3): provision cheapest 2x4090 -> ship code -> stage latents from
# R2 -> run real 2-GPU NCCL DDP training -> validate -> ALWAYS destroy.
set -uo pipefail
export PATH=/home/dani/.vastcli/bin:/home/dani/bin:$PATH
VAST=/home/dani/.vastcli/bin/vastai
KEY=/home/dani/.ssh/id_ed25519
LOG=/srv/nfs/projects/ahriuwu/scratchpad/vast_smoke.log
: > "$LOG"
say(){ echo "[$(date '+%T')] $*" | tee -a "$LOG"; }

IID=""
destroy(){ if [ -n "$IID" ]; then say "DESTROY instance $IID"; printf 'y\n' | timeout 60 "$VAST" destroy instance "$IID" >>"$LOG" 2>&1; fi; }
trap destroy EXIT

# 1) cheapest rentable 2x4090
say "searching 2x4090 offers..."
OIDS=$($VAST search offers 'gpu_name=RTX_4090 num_gpus=2 inet_down>400 rentable=true disk_space>70' -o dph --raw 2>>"$LOG" \
  | python3 -c "import sys,json; o=json.load(sys.stdin); print('\n'.join(str(x['id']) for x in o[:5]))")
[ -n "$OIDS" ] || { say "no offers"; exit 1; }

# 2) create (retry across the cheapest few if one was just taken)
for OID in $OIDS; do
  say "trying offer $OID"
  R=$($VAST create instance "$OID" --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime --disk 70 --ssh --direct --raw 2>>"$LOG")
  echo "$R" >>"$LOG"
  IID=$(echo "$R" | python3 -c "import sys,json;
try:
 d=json.load(sys.stdin); print(d.get('new_contract','') if d.get('success',True) else '')
except: print('')" 2>/dev/null)
  [ -n "$IID" ] && { say "created instance $IID"; break; }
done
[ -n "$IID" ] || { say "create failed on all offers"; exit 1; }
$VAST start instance "$IID" >>"$LOG" 2>&1 || true

# 3) poll for running + a REAL direct endpoint
HOST=""; PORT=""; ST=""
for i in $(seq 1 60); do
  read -r ST HOST PORT < <($VAST show instance "$IID" --raw 2>/dev/null | python3 -c "import sys,json;
d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d;
print(d.get('actual_status'), d.get('public_ipaddr'), d.get('direct_port_start'))" 2>/dev/null)
  say "poll $i: status=$ST host=$HOST port=$PORT"
  [ "$ST" = "running" ] && [ -n "$PORT" ] && [ "$PORT" != "None" ] && [ "$PORT" != "-1" ] && break
  sleep 15
done
[ "$ST" = "running" ] && [ -n "$PORT" ] && [ "$PORT" != "None" ] || { say "never reached running+endpoint"; exit 1; }

SSH(){ ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o BatchMode=yes -o ServerAliveInterval=8 -p "$PORT" root@"$HOST" "$@"; }

# 4) stable-sshd gate: 3 consecutive OK after a settle (image still pulling)
sleep 40
ok=0; for i in $(seq 1 60); do
  if SSH 'echo ok' >/dev/null 2>&1; then ok=$((ok+1)); [ "$ok" -ge 3 ] && break; else ok=0; fi
  sleep 10
done
[ "$ok" -ge 3 ] || { say "sshd never stable"; exit 1; }
say "sshd stable @ $HOST:$PORT"

# 5) ship code + rclone remote config
say "shipping code + rclone config..."
tar czf - -C /srv/nfs/projects/ahriuwu src scripts | SSH 'mkdir -p /root/ahriuwu && tar xzf - -C /root/ahriuwu' || { say "code ship failed"; exit 1; }
SSH 'mkdir -p /root/.config/rclone && cat > /root/.config/rclone/rclone.conf' < /home/dani/.config/rclone/rclone.conf

# 6) on-box: deps + rclone + stage 3 matches + run the DDP smoke (real NCCL)
say "on box: deps, stage from R2, run DDP smoke (timeout 300s)..."
SSH 'bash -s' >>"$LOG" 2>&1 <<'REMOTE'
set -e
export PATH=/opt/conda/bin:/usr/local/bin:$PATH
echo "=== GPUs ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
which rclone >/dev/null 2>&1 || { echo "installing rclone"; curl -s https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/r.zip && cd /tmp && (unzip -q r.zip || python3 -m zipfile -e r.zip .) && cp rclone-*-linux-amd64/rclone /usr/local/bin/ && chmod +x /usr/local/bin/rclone; }
echo "=== staging 3 latent matches from R2 ==="
mkdir -p /root/latents
for m in $(rclone lsf r2:ahriuwu-yt-pretrain/dynamics_replay_latents_v7_tok6000_clean/ | grep '\.pt$' | head -3); do
  rclone copy "r2:ahriuwu-yt-pretrain/dynamics_replay_latents_v7_tok6000_clean/$m" /root/latents --s3-no-check-bucket --transfers 8
done
echo "staged $(ls /root/latents/*.pt 2>/dev/null | wc -l) matches ($(du -sh /root/latents | cut -f1))"
pip -q install opencv-python-headless wandb 2>&1 | tail -2 || true
cd /root/ahriuwu
export PYTHONPATH=/root/ahriuwu/src PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SHM_DISABLE=0 NCCL_DEBUG=WARN TORCH_NCCL_ASYNC_ERROR_HANDLING=1
echo "=== torchrun 2-GPU DDP dynamics (latents-only, real config) ==="
timeout 300 torchrun --standalone --nproc_per_node=2 scripts/train_dynamics.py \
  --latents-dir /root/latents --packed --checkpoint-dir /root/ckpt \
  --model-size medium --latent-dim 32 --num-kv-heads 4 --num-register-tokens 8 --soft-cap 50.0 \
  --alternating-lengths --seq-len-short 128 --seq-len-long 256 \
  --batch-size-short 2 --batch-size-long 1 --long-ratio 0.1 \
  --gradient-accumulation-short 2 --gradient-accumulation-long 2 \
  --gradient-checkpointing --no-compile --stride 16 --save-steps 5 --eval-interval 5 \
  --holdout-videos 0 --num-workers 4 --epochs 1 --no-wandb ; echo "TRAIN_EXIT=$?"
echo "=== checkpoints written ==="; ls -la /root/ckpt/*.pt 2>/dev/null || echo "NO CHECKPOINT"
echo "=== gpu mem after ==="; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
REMOTE

say "smoke finished (see log). destroying box."
