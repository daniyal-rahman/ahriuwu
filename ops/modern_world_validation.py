"""TOOL: snapshot this worktree and validate modern world assets and runtime via Slurm."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time


def launch(spec,dry):
    root=Path(__file__).resolve().parents[1]
    out=Path('/mnt/nfs/shared')/spec['id']
    snapshot=out/'source'
    python='/mnt/nfs/projects/ahriuwu-lanerl-jax/.venv-gpu/bin/python'
    command=['sbatch','--parsable','-p','cpu','--nodelist=desktop','--cpus-per-task=2','--mem=6G',
             '--time='+spec['time'],'--job-name='+spec['id'],'--output=/mnt/nfs/shared/'+spec['id']+'-%j.out',
             '--chdir='+str(snapshot),str(snapshot/'slurm/modern_world_validation.sbatch'),spec['mode'],str(out)]
    print(shlex.join(command),flush=True)
    if dry:return
    out.mkdir(exist_ok=False);snapshot.mkdir()
    import shutil
    files=subprocess.check_output(['git','ls-files','--cached','--others','--exclude-standard','-z'],cwd=root).decode().split('\0')
    for name in set(files):
        src=root/name
        if name and src.is_file() and not src.is_symlink():
            dst=snapshot/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
    jid=subprocess.check_output(command,text=True).strip().split(';')[0]
    (out/'launch.json').write_text(json.dumps({'job':jid,'command':command,'spec':spec,'worktree':str(root)},indent=2))
    print('job',jid,flush=True)
    started=None
    while True:
        state=subprocess.check_output(['squeue','-h','-j',jid,'-o','%T'],text=True).strip()
        log=Path(f'/mnt/nfs/shared/{spec["id"]}-{jid}.out');text=log.read_text(errors='replace') if log.exists() else ''
        if not state:
            accounting=subprocess.check_output(['sacct','-n','-X','-j',jid,'-o','State,ExitCode'],text=True).strip()
            if 'COMPLETED' not in accounting or '0:0' not in accounting or 'MODERN WORLD VALIDATION COMPLETE' not in text:raise SystemExit('VALIDATION FAILED: '+accounting+'\n'+text[-2500:])
            print('validation completed; canary passed',flush=True);return
        if 'Traceback' in text:raise SystemExit(text[-2500:])
        if state=='RUNNING':
            if started is None:started=time.monotonic()
            if time.monotonic()-started>=180 and 'CANARY PASSED' in text:
                print(jid+' healthy after 180s; canary passed',flush=True);return
        time.sleep(10)
