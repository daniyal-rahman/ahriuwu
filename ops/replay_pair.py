"""Versioned paired replay/render worker, invoked through ops/launch.py."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from lanerl_jax.sim.config import DEFAULT_ROUTE_ARTIFACT

spec = json.loads(Path('experiments', sys.argv[1]+'.json').read_text())
scratch = Path('/scratch') / (spec['id']+'-'+os.environ['SLURM_JOB_ID'])
scratch.mkdir(parents=True, exist_ok=True)
shutil.copytree(DEFAULT_ROUTE_ARTIFACT, scratch/'routes')
out = Path('/mnt/nfs/shared') / spec['id']
out.mkdir(parents=True, exist_ok=True)
for arm in spec['arms']:
    source = Path(arm['checkpoint'])
    staged = scratch/arm['id']
    staged.mkdir()
    shutil.copyfile(source, staged/'checkpoint.msgpack')
    shutil.copyfile(source.parent/'manifest.json', staged/'manifest.json')
    dest = out/arm['id']
    subprocess.run([sys.executable, '-m', 'lanerl_jax.train.jax_eval',
        str(staged/'checkpoint.msgpack'), '--out', str(dest), '--seed', str(spec['seed']),
        '--seconds', str(spec['seconds']), '--step-ticks', '6', '--start-near-wave',
        '--route-artifact', str(scratch/'routes'), '--red', 'policy', '--replay',
        '--replay-hz', '10', '--label', arm['id']+' final u2500 | mirror | seed '+str(spec['seed'])], check=True)
    subprocess.run([sys.executable, '-m', 'lanerl_jax.replay_render', str(dest/'trace.npz'),
        '--out-dir', str(dest/'minimap'), '--video', '--view', 'map', '--speed', str(spec['speed'])], check=True)
    print('VIDEO READY', dest/'minimap/replay.mp4', flush=True)
print('PROFILE COMPLETE', flush=True)
