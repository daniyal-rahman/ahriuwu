#!/bin/bash
# Accelerate the hud-fix dynamics run (job 168) on Vast: provision 4x4090, resume
# from 168's checkpoint (step 2312, on R2), train UNLABELED + pixel-HUD-masked DDP,
# stream ckpt->R2 every cycle, auto-stop on PLATEAU (no new best tau0.9 in ~12
# evals) or when the $BUDGET_USD time-cap hits, then destroy. Streams keep R2
# current so worst-case nothing is lost/overspent. Runs on danilogin.
set -uo pipefail
export PATH=/home/dani/.vastcli/bin:/home/dani/bin:$PATH
VAST=/home/dani/.vastcli/bin/vastai
KEY=/home/dani/.ssh/id_ed25519
LOG=/srv/nfs/projects/ahriuwu/scratchpad/vast_hudfix.log
: > "$LOG"
say(){ echo "[$(date '+%T')] $*" | tee -a "$LOG"; }
BUDGET_USD="${BUDGET_USD:-20}"
R2=r2:ahriuwu-yt-pretrain
SEED="$R2/dynamics_hudfix_seed/dynamics_latest.pt"
CKR2="$R2/dynamics_hudfix_accel"

IID=""; PRICE=""
destroy(){ if [ -n "$IID" ]; then say "DESTROY $IID"; printf 'y\n' | timeout 60 "$VAST" destroy instance "$IID" >>"$LOG" 2>&1; fi; }
trap destroy EXIT

pick(){ $VAST search offers "gpu_name=RTX_4090 num_gpus=$1 inet_down>500 rentable=true disk_space>140" -o dph --raw 2>>"$LOG" \
  | python3 -c "import sys,json; o=json.load(sys.stdin); print('\n'.join(f\"{x['id']} {x['dph_total']}\" for x in o[:5]))"; }
NGPU=4; OFFERS=$(pick 4); [ -z "$OFFERS" ] && { NGPU=2; OFFERS=$(pick 2); }
[ -n "$OFFERS" ] || { say "no 4090 offers"; exit 1; }
say "NGPU=$NGPU candidate offers: $(echo "$OFFERS" | tr '\n' ';')"

while read -r OID PR; do
  [ -z "$OID" ] && continue
  say "try offer $OID @ \$$PR/hr"
  R=$($VAST create instance "$OID" --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime --disk 150 --ssh --direct --raw 2>>"$LOG")
  echo "$R" >>"$LOG"
  IID=$(echo "$R" | python3 -c "import sys,json
try: d=json.load(sys.stdin); print(d.get('new_contract','') if d.get('success',True) else '')
except: print('')" 2>/dev/null)
  [ -n "$IID" ] && { PRICE="$PR"; break; }
done <<< "$OFFERS"
[ -n "$IID" ] || { say "create failed on all offers"; exit 1; }
say "instance $IID @ \$$PRICE/hr"
$VAST start instance "$IID" >>"$LOG" 2>&1 || true

TRAIN_SECONDS=$(python3 -c "b=$BUDGET_USD; p=float('$PRICE'); print(int(max(1200, b/p*3600 - 1800)))")
say "budget \$$BUDGET_USD @ \$$PRICE/hr -> train window ${TRAIN_SECONDS}s (~$((TRAIN_SECONDS/3600))h)"

HOST=""; PORT=""; ST=""
for i in $(seq 1 60); do
  read -r ST HOST PORT < <($VAST show instance "$IID" --raw 2>/dev/null | python3 -c "import sys,json
d=json.load(sys.stdin); d=d[0] if isinstance(d,list) else d
print(d.get('actual_status'), d.get('public_ipaddr'), d.get('direct_port_start'))" 2>/dev/null)
  say "poll $i: status=$ST host=$HOST port=$PORT"
  [ "$ST" = "running" ] && [ -n "$PORT" ] && [ "$PORT" != "None" ] && [ "$PORT" != "-1" ] && break
  sleep 15
done
[ "$ST" = "running" ] && [ -n "$PORT" ] && [ "$PORT" != "None" ] || { say "never running"; exit 1; }

SSH(){ ssh -i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=15 -o BatchMode=yes -o ServerAliveInterval=8 -p "$PORT" root@"$HOST" "$@"; }

sleep 40
ok=0; for i in $(seq 1 60); do
  if SSH 'echo ok' >/dev/null 2>&1; then ok=$((ok+1)); [ "$ok" -ge 3 ] && break; else ok=0; fi
  sleep 10
done
[ "$ok" -ge 3 ] || { say "sshd never stable"; exit 1; }
say "sshd stable @ $HOST:$PORT"

say "shipping working tree (src+scripts) + rclone config..."
tar czf - -C /srv/nfs/projects/ahriuwu src scripts | SSH 'mkdir -p /root/ahriuwu && tar xzf - -C /root/ahriuwu' || { say "ship failed"; exit 1; }
SSH 'mkdir -p /root/.config/rclone && cat > /root/.config/rclone/rclone.conf' < /home/dani/.config/rclone/rclone.conf

say "on box: deps + stage (combined latents + tokenizer + mask + seed) + launch pixel-HUD DDP + R2 streamer..."
SSH 'bash -s' >>"$LOG" 2>&1 <<REMOTE
set -e
export PATH=/opt/conda/bin:/usr/local/bin:\$PATH
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
which rclone >/dev/null 2>&1 || (curl -s https://downloads.rclone.org/rclone-current-linux-amd64.zip -o /tmp/r.zip && cd /tmp && (unzip -q r.zip || python3 -m zipfile -e r.zip .) && cp rclone-*-linux-amd64/rclone /usr/local/bin/ && chmod +x /usr/local/bin/rclone)
pip -q install opencv-python-headless wandb 2>&1 | tail -1 || true
mkdir -p /workspace/seed /workspace/ckpt /workspace/nolabels
cd /root/ahriuwu
echo "staging combined latents + tokenizer + mask..."; DEST=/workspace bash scripts/stage_dyn_hudfix.sh
echo "staging seed..."; rclone copyto $SEED /workspace/seed/seed.pt --s3-no-check-bucket
echo "seed \$(stat -c%s /workspace/seed/seed.pt 2>/dev/null) bytes"
export NGPU=$NGPU NCCL_SHM_DISABLE=0
export DYN_ARGS_FILE=scripts/dyn_train_args_hudfix.sh
export LATENTS_DIR=/workspace/latents_hudfix CHECKPOINT_DIR=/workspace/ckpt LABELS_ROOT=/workspace/nolabels
export TOK_CKPT=/workspace/tok/transformer_tokenizer_latest.pt HUD_MASK=/workspace/mask/hud_valid_mask_352.pt
export INIT_RESUME=/workspace/seed/seed.pt R2_CKPT=$CKR2 WANDB_MODE=offline PIXEL_FRAMES=4
export RUN_LOG=/workspace/train.log STOP_FILE=/workspace/.stop
setsid bash scripts/vast_supervised_dyn.sh </dev/null >/workspace/supervisor.log 2>&1 &
sleep 8; echo "supervisor launched; train.log tail:"; tail -n 6 /workspace/train.log 2>/dev/null
REMOTE

say "training launched; monitoring up to ${TRAIN_SECONDS}s (auto-stop on plateau)..."
ITERS=$((TRAIN_SECONDS/150)); [ "$ITERS" -lt 1 ] && ITERS=1
BEST=0; NOIMP=0
for i in $(seq 1 "$ITERS"); do
  sleep 150
  P=$(SSH 'grep -aE "Epoch 0 \[[0-9]|EVAL step|Traceback|CUDA out of memory|RuntimeError|Watchdog" /workspace/train.log 2>/dev/null | tail -2' 2>/dev/null)
  say "[$i/$ITERS] ${P//$'\n'/ | }"
  # no-progress abort: DDP+pixel-HUD is untested; a hang shows NO Epoch line and
  # the supervisor won't restart a hung (not-crashed) process. NCCL init + first
  # step should appear within ~25 min; if not, kill it rather than burn credits.
  HASPROG=$(SSH 'grep -ac "Epoch 0 \[" /workspace/train.log 2>/dev/null' 2>/dev/null || echo 0)
  if [ "${HASPROG:-0}" -eq 0 ] && [ "$i" -ge 10 ]; then
    say "NO TRAINING PROGRESS after ~$((i*150/60))min (no 'Epoch 0 [' line) -> likely DDP hang/crash -> ABORT"; break
  fi
  # plateau check on best teacher-forced tau0.9
  TAU=$(SSH 'grep -aoE "psnr_tau0.9=[0-9.]+" /workspace/train.log 2>/dev/null | tail -1' 2>/dev/null | grep -oE "[0-9.]+" | tail -1)
  if [ -n "$TAU" ]; then
    IMP=$(python3 -c "print(1 if $TAU > $BEST + 0.2 else 0)" 2>/dev/null || echo 0)
    if [ "$IMP" = "1" ]; then BEST=$TAU; NOIMP=0; else NOIMP=$((NOIMP+1)); fi
    say "    tau0.9=$TAU best=$BEST noimp=$NOIMP/16"
    if [ "$NOIMP" -ge 16 ]; then say "PLATEAU (no new best tau0.9 in 16 evals) -> stop early"; break; fi
  fi
done

say "stopping: stop supervisor + final stream + destroy"
SSH 'touch /workspace/.stop; pkill -f "[t]orchrun" 2>/dev/null; pkill -f "[v]ast_supervised_dyn" 2>/dev/null; sleep 15; [ -f /workspace/ckpt/dynamics_latest.pt ] && rclone copyto /workspace/ckpt/dynamics_latest.pt '"$CKR2"'/dynamics_latest.pt --s3-no-check-bucket --s3-disable-checksum && echo FINAL_STREAM_OK' >>"$LOG" 2>&1 || true
say "DONE. latest -> $CKR2/dynamics_latest.pt"
