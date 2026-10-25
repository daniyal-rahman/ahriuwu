"""Sharded target-only event conversion from immutable replay/reference snapshots."""
import argparse
import ctypes
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from . import replay as R, convert as C
from .event_dataset import CONTRACT, convert_events, save_events, artifact_hashes
from .event_inputs import ENTITY_DIM, SELF_DIM, GLOBAL_DIM


def write_json(path, value):
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2))
    temp.replace(path)


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    p.add_argument('--reference', type=Path)
    p.add_argument('--catalogs', type=Path)
    p.add_argument('--decoder', type=Path)
    p.add_argument('--decoder-hashes', type=Path)
    p.add_argument('--scratch', type=Path)
    p.add_argument('--finish', action='store_true')
    a = p.parse_args()
    plan_bytes = a.plan.read_bytes()
    plan = json.loads(plan_bytes)
    digest = hashlib.sha256(plan_bytes).hexdigest()
    rows = plan['games']
    names = [row['game'] for row in rows]
    assert len(names) == len(set(names)) and 0 <= a.shard < a.shards
    base_splits = {}
    for row in rows:
        previous = base_splits.setdefault(row['base'], row['split'])
        assert previous == row['split'], 'perspectives from one match cross splits'
    if a.finish:
        parts = [json.loads((a.output / ('shard-'+str(i)+'.json')).read_text()) for i in range(a.shards)]
        assert all(x['passed'] and x['plan_sha256'] == digest for x in parts)
        sources = {part.get('source_commit') for part in parts}
        if (any(part.get('contract') != CONTRACT for part in parts)
                or len(sources) != 1 or not next(iter(sources))):
            raise ValueError('conversion shards mix source commits or dataset contracts')
        reports = [r for part in parts for r in part['reports']]
        assert sorted(r['game'] for r in reports) == sorted(names)
        for name in names:
            for suffix in ('.npz', '.json', '.events.json.gz'):
                assert (a.output/'games'/(name+suffix)).is_file()
        write_json(a.output/'manifest.json', dict(passed=True, contract=CONTRACT, source_commit=next(iter(sources)), plan_sha256=digest,
            games=rows, reports=reports, shards=parts,
            artifacts={name: artifact_hashes(a.output/'games'/name) for name in names}, limits='target-only replay events; no policy-quality gate'))
        print('FINALIZED', a.output, len(rows), flush=True)
        return
    if any(x is None for x in (a.reference, a.catalogs, a.decoder, a.decoder_hashes, a.scratch)):
        p.error('conversion requires reference, catalogs, decoder, decoder-hashes and scratch')
    R.DECODER_SCRIPTS = a.decoder
    for name, expected in json.loads(a.decoder_hashes.read_text()).items():
        assert hashlib.sha256((a.decoder/name).read_bytes()).hexdigest() == expected
    selected = rows[a.shard::a.shards]
    (a.output/'games').mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    reports = []
    libc = ctypes.CDLL('libc.so.6')
    for row in selected:
        source = Path(row['path'])
        dataset = a.scratch/'inputs'/source.parent.parent.name
        (dataset/'decoder-snapshot').mkdir(parents=True, exist_ok=True)
        (dataset/'decoder-snapshot/COMMIT').write_text(R.decoder_commit(source.parent.parent)+'\n')
        local = dataset/'games'/source.name
        size = sum(f.stat().st_size for f in source.rglob('*') if f.is_file())
        assert shutil.disk_usage(a.scratch).free > size + 20*1024**3
        shutil.copytree(source, local, symlinks=False)
        reference = a.scratch/'reference.npz'
        shutil.copyfile(a.reference/'games'/(row['game']+'.npz'), reference)
        game = R.read_game(local, R.decision_times(30., 840.), row['champion'], C.ROLE_LANE[row['role']])
        arrays, meta, raw = convert_events(game, row, reference, a.catalogs)
        assert arrays['entities'].shape == (8100, 32, ENTITY_DIM)
        assert arrays['self'].shape == (8100, SELF_DIM) and arrays['global_'].shape == (8100, GLOBAL_DIM)
        assert not np.any(arrays['aim_valid'] & ~arrays['valid'])
        assert not np.any(arrays['aim_valid'] & (arrays['aim_mode'] == 1) & (arrays['target_row'] < 0))
        aim_counts = {}
        for b in range(9):
            mask = arrays['valid'] & (arrays['button'] == b)
            aim_counts[str(b)] = dict(count=int(mask.sum()), aim=int((mask & arrays['aim_valid']).sum()),
                                      clipped=int((mask & arrays['clipped']).sum()))
        meta['aim_counts'] = aim_counts
        save_events(a.output/'games'/row['game'], arrays, meta, raw)
        reports.append(dict(game=row['game'], counts=meta['counts'], aim_counts=aim_counts, collisions=meta['collisions']))
        print(len(reports), '/', len(selected), row['game'], meta['counts'], flush=True)
        del game, arrays, meta, raw
        R._LAST_FIELDS[:] = [None, None]
        R._LANE_CACHE.clear()
        gc.collect()
        libc.malloc_trim(0)
        shutil.rmtree(local)
        reference.unlink()
    write_json(a.output/('shard-'+str(a.shard)+'.json'), dict(passed=True, contract=CONTRACT,
        plan_sha256=digest, source_commit=Path('.bc-source-commit').read_text().strip(),
        job=os.environ.get('SLURM_JOB_ID'), reports=reports, seconds=time.monotonic()-started))


if __name__ == '__main__':
    main()
