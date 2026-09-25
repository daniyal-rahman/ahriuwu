#!/usr/bin/env bash
# Where does throughput stop scaling with server instances?
#
# SUPERSEDED by lanerl/gil_probe.py + proc_scale.sh. Kept because its
# decisions/s column is what exposed the staleness rejection, but read it
# with two caveats:
#
#   * decisions/s counts only rollouts the learner ACCEPTS, so discarded
#     work shows up as a throughput collapse. Divide by (1 - reject_rate)
#     -- from the run's own rollout_rejected rows -- for the real rate.
#   * the GPU and load columns were sampled after the run exited and are
#     now disabled rather than left to mislead again.
cd /mnt/nfs/projects/ahriuwu-lanerl
PY=/home/dani/miniconda3/envs/ml/bin/python
echo "actors x envs = instances | decisions/s | learner_frac | GPU% | load"
for cfg in "2 4" "3 8" "4 12" "6 12" "8 12"; do
  set -- $cfg; A=$1; E=$2; N=$((A*E))
  R=scale-$N
  rm -rf runs/$R
  timeout 300 $PY -m lanerl_train --run-name $R \
    --num-actors $A --envs-per-actor $E --rollout-steps 128 \
    --total-updates 60 --checkpoint-every 10000 --eval-every 0 --anchors '' \
    --device cuda --port-base $((9000 + N*80)) --anchor-port-base $((40000 + N*80)) \
    --init-from demos/bc_policy.pt --kl-ref-coef 0.05 \
    --lr 1e-5 --critic-lr 3e-4 --alpha 0.5 --no-end-on-death \
    >/dev/null 2>&1
  # NOT SAMPLED. These used to read nvidia-smi and `uptime` HERE -- after
  # `timeout` returned, i.e. with the run already dead -- so they measured an
  # idle machine. The "load 0.55 on a 16-core box" that column produced was
  # read as "nothing is saturated" and sent a whole investigation the wrong
  # way. Use lanerl/gil_probe.py, which samples for the DURATION of the run.
  G=n/a
  L=n/a
  $PY - "$R" "$A" "$E" "$N" "$G" "$L" <<'PYEOF'
import json,sys,statistics,pathlib
r,a,e,n,g,l = sys.argv[1:7]
p=pathlib.Path(f"runs/{r}/metrics.jsonl")
if not p.exists(): print(f"  {a}x{e} = {n:>3} | FAILED"); raise SystemExit
rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
d=[x.get("throughput/decisions_per_s") for x in rows if x.get("throughput/decisions_per_s")]
f=[x.get("throughput/learner_frac") for x in rows if x.get("throughput/learner_frac")]
d=d[len(d)//3:]; f=f[len(f)//3:]   # drop warm-up
print(f"  {a}x{e} = {n:>3} | {statistics.mean(d):9,.0f} | {statistics.mean(f):11.2f} | {g:>4} | {l}")
PYEOF
done
