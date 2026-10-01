"""Narrow PERF-003 diagnostic launcher; invoked only through ops/launch.py."""
import json
from pathlib import Path
import shlex
import subprocess
import time


def launch(spec, dry, resume=None):
    root = Path('/srv/nfs/projects/ahriuwu-lanerl-jax')
    out = root / 'lanerl_jax/runs' / spec['id']
    if resume:
        if spec['engine'] != 'wave-scenario':
            raise SystemExit('Resume is supported here only for the same wave-scenario experiment')
        checkpoint = Path(resume)
        manifest = json.loads((checkpoint.parent/'manifest.json').read_text())
        if not checkpoint.is_file() or manifest['config']['scenario'] != spec:
            raise SystemExit('Resume checkpoint must belong to this exact experiment spec')
        if subprocess.check_output(['squeue','-h','-n',spec['id'],'-o','%i'],text=True).strip():
            raise SystemExit('Experiment already queued or running')
        out = out / 'continuations' / time.strftime('%Y%m%d-%H%M%S')
    if out.exists():
        raise SystemExit(f'REFUSED: diagnostic output already exists: {out}')
    full = spec['engine'] == 'full-profile'
    script = {'credit-audit': 'slurm/credit_audit.sbatch',
              'full-profile': 'slurm/full_profile.sbatch',
              'gru-profile': 'slurm/gru_profile.sbatch',
              'vision-ab': 'slurm/vision_ab.sbatch',
              'bush-ab': 'slurm/bush_ab.sbatch',
              'vision-smoke': 'slurm/vision_smoke.sbatch',
              'paired-vec': 'slurm/paired_vec.sbatch',
              'wave-scenario': 'slurm/wave_scenario.sbatch',
              'wave-replay': 'slurm/wave_replay.sbatch',
              'replay-pair': 'slurm/replay_pair.sbatch',
              'escape-counterfactual': 'slurm/escape_counterfactual.sbatch'}[spec['engine']]
    cmd = ['sbatch', '--parsable', '--partition=gpup', '--gres=gpu:1',
           '--nodelist=desktop', '--cpus-per-task=4', '--mem=24G' if full else '--mem=20G',
           '--time='+spec.get('slurm_time', '01:30:00' if full else '00:35:00'),
           '--chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax',
           '--signal=USR1@120', '--no-requeue', f"--job-name={spec['id']}",
           f"--output=/mnt/nfs/shared/{spec['id']}-%j.out",
           script, spec['id']]
    if resume:
        cmd.extend(['--resume', str(checkpoint)])
    if spec.get('backend') == 'cpu':
        if spec['engine'] not in ('replay-pair', 'escape-counterfactual', 'wave-replay'):
            raise SystemExit('CPU fallback is limited to frozen replay diagnostics')
        cmd = [x for x in cmd if x != '--gres=gpu:1']
        changes = {'--partition=gpup':'--partition=cpu', '--nodelist=desktop':'--nodelist=danilogin',
                   '--cpus-per-task=4':'--cpus-per-task=2', '--mem=20G':'--mem=10G'}
        cmd = [changes.get(x,x) for x in cmd]
    print('command:', shlex.join(cmd), flush=True)
    if dry:
        return
    out.mkdir(parents=True)
    jid = subprocess.check_output(cmd, text=True, cwd=root).strip().split(';')[0]
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True, cwd=root).strip()
    (out / 'launch.json').write_text(json.dumps(dict(spec=spec, command=cmd, job=jid, source=source), indent=2))
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
