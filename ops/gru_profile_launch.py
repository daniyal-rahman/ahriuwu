"""Narrow PERF-003 diagnostic launcher; invoked only through ops/launch.py."""
import json
from pathlib import Path
import shlex
import subprocess
import time


def launch(spec, dry):
    root = Path('/srv/nfs/projects/ahriuwu-lanerl-jax')
    out = root / 'lanerl_jax/runs' / spec['id']
    if out.exists():
        raise SystemExit(f'REFUSED: diagnostic output already exists: {out}')
    full = spec['engine'] == 'full-profile'
    cmd = ['sbatch', '--parsable', '--partition=gpup', '--gres=gpu:1',
           '--nodelist=desktop', '--cpus-per-task=4', '--mem=24G' if full else '--mem=20G',
           '--time=01:30:00' if full else '--time=00:35:00',
           '--chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax',
           '--no-requeue', f"--job-name={spec['id']}",
           f"--output=/mnt/nfs/shared/{spec['id']}-%j.out",
           'slurm/full_profile.sbatch' if full else 'slurm/gru_profile.sbatch', spec['id']]
    print('command:', shlex.join(cmd), flush=True)
    if dry:
        return
    out.mkdir(parents=True)
    jid = subprocess.check_output(cmd, text=True, cwd=root).strip().split(';')[0]
    (out / 'launch.json').write_text(json.dumps(dict(spec=spec, command=cmd, job=jid), indent=2))
    print('job', jid, flush=True)
    log = Path(f"/mnt/nfs/shared/{spec['id']}-{jid}.out")
    # Bounded startup watch; the worker canary must pass before full profiles.
    running_since = None
    while True:
        state = subprocess.check_output(['squeue', '-h', '-j', jid, '-o', '%T'], text=True).strip()
        text = log.read_text(errors='replace') if log.exists() else ''
        complete = 'PROFILE COMPLETE' in text
        if 'Traceback' in text or (not state and not complete):
            raise SystemExit('PROFILE FAILED: ' + text[-2000:])
        if complete:
            print('profile complete; canary passed', flush=True)
            return
        if state == 'RUNNING':
            if running_since is None:
                running_since = time.monotonic()
            if time.monotonic() - running_since >= 180 and 'CANARY PASSED' in text:
                print(f'{jid} healthy after 180s; canary passed', flush=True)
                return
        time.sleep(15)
