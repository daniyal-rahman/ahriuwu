#!/bin/bash
# Wait for a training run to finish, reap a job that hangs after its manifest
# says complete (E19 held 8 cores for 20 min that way), stop its periodic
# evaluator, and run the 12-episode final frozen mirror evaluation.
#   ops/finish_run.sh <run root under lanerl_jax/runs> <slurm job name> <eval tag>
cd /srv/nfs/projects/ahriuwu-lanerl-jax
ROOT=$1; JOB=$2; TAG=$3
while true; do
  st=$(squeue -h -o '%j %t' | awk -v j=$JOB '$1==j{print $2}')
  [ -z "$st" ] && break
  RUN=$(ls -d $ROOT/*-s[0-9]*/ 2>/dev/null | tail -1)
  status=$(python3 -c "import json;print(json.load(open('$RUN/manifest.json')).get('results',{}).get('status',''))" 2>/dev/null)
  if [ "$status" = "complete" ]; then
    sleep 300
    jid=$(squeue -h -o '%i %j' | awk -v j=$JOB '$2==j{print $1}')
    [ -n "$jid" ] && { scancel $jid; echo "reaped $JOB ($jid): manifest complete, process did not exit"; }
    break
  fi
  sleep 120
done
echo "$JOB finished at $(date -u +%FT%TZ)"
python3 - "$ROOT" <<'PY'
import os,signal,sys
root=sys.argv[1]
for pid in os.listdir('/proc'):
    if not pid.isdigit(): continue
    try: cmd=open(f'/proc/{pid}/cmdline','rb').read().replace(b'\0',b' ').decode()
    except Exception: continue
    if cmd.startswith('bash ops/periodic_eval.sh') and root in cmd: os.kill(int(pid), signal.SIGTERM); print('evaluator stopped', pid)
PY
CK=$(python3 -c "
from pathlib import Path; import sys
c=sorted(Path('$ROOT').glob('*-s[0-9]*/ckpt_latest.msgpack'), key=lambda p:p.stat().st_mtime)[-1]
print('/mnt/nfs/projects/ahriuwu-lanerl-jax/'+str(c))")
python3 ops/launch.py eval --ckpt $CK --opponent mirror --envs 4 --episodes 3 --tag $TAG > lanerl_jax/runs/EVAL/$TAG.launch.out 2>&1
sleep 30; while squeue -h -o %j | grep -q "^$TAG$"; do sleep 30; done
grep '"agent"' lanerl_jax/runs/EVAL/$TAG.out | python3 -c "
import sys,json,statistics as st
rows=[json.loads(l) for l in sys.stdin if l.strip().startswith('{')]
for t in (0,1):
    cs=[r['cs'] for r in rows if r['team']==t]; cs and print('$TAG team',t,'n',len(cs),'mean %.1f min %.0f max %.0f'%(st.mean(cs),min(cs),max(cs)),'deaths %.1f'%st.mean([r['deaths'] for r in rows if r['team']==t]))"
