"""Independent v2 label audit on a completed immutable fixture, run via Slurm."""
import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path

import numpy as np

from lanerl_jax.modern.bc.event_labels import label_events
from lanerl_jax.modern.bc.event_inputs import previous_received_events, SELF_FIELDS
from lanerl_jax.modern.bc.events import BUTTON_INDEX as B
from lanerl_jax.modern.bc.replay import Action, _actions


def quantiles(values):
    return dict(zip(('p10', 'p50', 'p90', 'p99'), map(float, np.quantile(values, [.1, .5, .9, .99])))) if len(values) else None


def regressions():
    t = np.arange(12) * .1
    ids = np.full((len(t), 32), -1)
    xy = np.zeros((len(t), 2))
    axis, normal = np.array([1., 0.]), np.array([0., 1.])
    actions = [Action(.1, 'attack', target=99), Action(.2, 'path', point=(0., 0.), n_waypoints=1),
               Action(.6, 'attack', target=99)]
    labels = label_events(t, actions, xy, ids, axis, normal, np.ones(len(t), bool), 1)
    history, _ = previous_received_events(t, actions, xy, ids, axis, normal)
    # Reproduce current behavior; this is evidence, not an assertion of correctness.
    class FakeFields:
        def rows(self, name, family=None):
            return [0], [1.], [1]
        def value(self, name, row):
            if name == 'movement.received_path':
                return dict(waypoints_xz=[[0., 0.], [100., 0.]])
            if name == 'event.basic_attack_execution_proxy':
                return dict(target_id=99)
            return dict(slot=0, targeting_mode='self')
    ordered, _ = _actions(FakeFields(), 1, 2)
    return dict(equal_clock_read_order=[a.kind for a in ordered],
                attack_after_single_point_path_button=int(labels.button[6]),
                attack_after_single_point_path_history_age=float(history[7, -1]),
                attack_button_index=B['attack'])


def profile(root):
    manifest = json.loads((root / 'manifest.json').read_text())
    assert manifest['passed']
    groups = defaultdict(lambda: dict(counts=Counter(), values=defaultdict(list), games=[]))
    for row in manifest['games']:
        prefix = root / 'games' / row['game']
        with np.load(str(prefix) + '.npz') as f:
            a = {k: f[k] for k in f.files}
        with gzip.open(str(prefix) + '.events.json.gz', 'rt') as f:
            raw = json.load(f)
        c, values = Counter(), defaultdict(list)
        metadata = json.loads(Path(str(prefix) + '.json').read_text())
        c['perspectives'] = 1
        c['matched_observation_parity'] = int(metadata['observation_parity'])
        c['finite_metadata_values'] = int(np.isfinite(a['own_metadata']).sum())
        c['total_metadata_values'] = int(a['own_metadata'].size)
        valid, button = a['valid'], a['button']
        c['frames'], c['valid'] = len(valid), int(valid.sum())
        blind = (np.arange(len(valid)) % 256) < 32
        c['valid_never_trained_default_windows'] = int((valid & blind).sum())
        for name, b in B.items():
            mask = valid & (button == b)
            c[name] = int(mask.sum())
            c[name + '_aim'] = int((mask & a['aim_valid']).sum())
            c[name + '_clipped'] = int((mask & a['clipped']).sum())
            c[name + '_burn_dropped'] = int((mask & blind).sum())
            if mask.any():
                event_times = a['t'][mask]
                values[name + '_interval_seconds'].extend(np.diff(event_times).tolist())
        # All nonpadded rows are candidates in the current policy.
        attacks = valid & (button == B['attack']) & a['aim_valid']
        allies = (a['entities'][..., 10] > .5) & ~a['pad']
        c['attack_frames_with_allied_pointer_candidates'] = int((attacks & allies.any(1)).sum())
        c['attack_allied_pointer_candidate_slots'] = int(allies[attacks].sum())
        c['attack_pointer_candidate_slots'] = int((~a['pad'][attacks]).sum())
        c['unit_targets'] = int((valid & a['aim_valid'] & (a['aim_mode'] == 1)).sum())
        c['co_move'] = int((valid & (a['co_move_source'] >= 0)).sum())
        c['co_move_aim'] = int((valid & a['co_move_valid']).sum())
        cast_frames = valid & (button >= B['q']) & (button <= B['r'])
        c['cast_frames'] = int(cast_frames.sum())
        # A trivial last-destination baseline measures redundant cursor supervision.
        point_start = len(SELF_FIELDS) + 4 + len(B)
        history_xy = a['self'][:, point_start:point_start + 2] * 3000.
        ds, dn = history_xy.T
        from lanerl_jax.modern.screen import FLOOR_Y, CAM_Y, TILT_DEG, FOV_V_DEG, RESOLUTION
        ct, st = math.cos(math.radians(TILT_DEG)), math.sin(math.radians(TILT_DEG))
        tv = math.tan(math.radians(FOV_V_DEG)/2.)
        th = tv * RESOLUTION[0]/RESOLUTION[1]
        dz = dn - (FLOOR_Y-CAM_Y)*ct/st
        vy, vz = (FLOOR_Y-CAM_Y)*ct + dz*st, -(FLOOR_Y-CAM_Y)*st + dz*ct
        with np.errstate(invalid='ignore', divide='ignore'):
            hx, hy = (0.5 + ds/(2.*th*vz))*96, (0.5 - vy/(2.*tv*vz))*54
        move_aim = valid & (button == B['move']) & a['aim_valid']
        known = move_aim & np.isfinite(hx) & np.isfinite(hy)
        dx, dy = np.floor(hx[known]) - a['screen_x'][known], np.floor(hy[known]) - a['screen_y'][known]
        c['move_last_destination_baseline_eligible'] = int(known.sum())
        c['move_last_destination_baseline_within2'] = int(((np.abs(dx) <= 2) & (np.abs(dy) <= 2)).sum())
        values['move_last_destination_baseline_cells'].extend(np.hypot(dx,dy).tolist())
        selected_sources = set(map(int, a['event_source'][valid]))
        previous_move = previous_path = previous_attack = None
        interrupted = False
        bins = defaultdict(list)
        for i, event in enumerate(raw):
            k = int(np.floor((event['t'] - a['t'][0]) / .1 + 1e-9))
            if not 0 <= k < len(valid) or not valid[k]:
                continue
            kind = event['kind']
            bins[k].append(event)
            c['raw_' + kind] += 1
            if kind == 'path':
                c['raw_path_onepoint' if event['n_waypoints'] == 1 else 'raw_path_multipoint'] += 1
                if event['n_waypoints'] == 1:
                    interrupted = True
                else:
                    if previous_move is not None:
                        dt = event['t'] - previous_move['t']
                        distance = np.linalg.norm(np.array(event['point']) - previous_move['point'])
                        values['consecutive_move_endpoint_distance'].append(float(distance))
                        values['consecutive_move_interval_seconds'].append(float(dt))
                        c['move_near_previous_150u_500ms'] += int(dt <= .5 and distance <= 150)
                        c['move_identical_endpoint_500ms'] += int(dt <= .5 and distance == 0)
                    previous_move = event
                    if previous_attack is not None:
                        values['path_after_attack_seconds'].append(event['t'] - previous_attack['t'])
                        c['multipoint_path_within_200ms_after_attack'] += int(event['t'] - previous_attack['t'] <= .2)
                    interrupted = True
                previous_path = event
            elif kind == 'attack':
                if previous_attack is not None and previous_attack['target'] == event['target'] and interrupted:
                    c['same_target_attack_after_path'] += 1
                    if previous_path and previous_path['n_waypoints'] == 1:
                        c['same_target_attack_after_onepoint_path'] += 1
                        c['same_target_attack_after_onepoint_path_selected'] += int(i in selected_sources)
                previous_attack, interrupted = event, False
            elif kind == 'cast' and event['slot'] in range(4):
                c['raw_qwer'] += 1
                c['raw_qwer_not_selected'] += int(i not in selected_sources)
        ties = defaultdict(list)
        for k, events in bins.items():
            for event in events:
                ties[event['t']].append(event)
        for events in ties.values():
            kinds = {e['kind'] for e in events}
            c['equal_clock_cross_kind_groups'] += int(len(kinds) > 1)
            c['equal_clock_cast_multipoint_path'] += int('cast' in kinds and any(e['kind']=='path' and e['n_waypoints']>1 for e in events))
        for k, events in bins.items():
            qwer = [e for e in events if e['kind'] == 'cast' and e['slot'] in range(4)]
            c['multiple_qwer_bins'] += int(len(qwer) > 1)
            c['cast_path_before'] += int(bool(qwer) and any(e['kind'] == 'path' and e['n_waypoints'] > 1 and e['t'] < qwer[-1]['t'] for e in events))
            c['cast_path_after'] += int(bool(qwer) and any(e['kind'] == 'path' and e['n_waypoints'] > 1 and e['t'] >= qwer[-1]['t'] for e in events))
        for champion in (row['champion'], 'all'):
            g = groups[champion]
            g['counts'].update(c)
            for name, v in values.items():
                g['values'][name].extend(v)
            g['games'].append(dict(game=row['game'], split=row['split'], counts=dict(c)))
    return dict(contract=manifest['contract'], source=manifest.get('source_commit'),
                limits='Completed manifest; subgroup/sample counts in output; path reissues and attack timing are proxies, not verified clicks; counts are descriptive',
                synthetic=regressions(), groups={name: dict(counts=dict(g['counts']),
                    quantiles={k: quantiles(v) for k, v in g['values'].items()}, games=g['games']) for name, g in groups.items()})


def profile_sources(root, decoder, scratch):
    import ctypes
    import gc
    import shutil
    import sys
    sys.path.insert(0, str(decoder))
    import event_training_loader as loader
    from lanerl_jax.modern.bc.replay import _Fields
    manifest = json.loads((root / 'manifest.json').read_text())
    results = []
    for game in manifest['games']:
        source = Path(game['path']) / 'export'
        local = scratch / 'raw-game'
        needed = sum(p.stat().st_size for p in (source/'event-training').rglob('*') if p.is_file())
        assert shutil.disk_usage(scratch).free > needed + 20*1024**3
        shutil.copytree(source/'event-training', local, symlinks=False)
        roster = json.loads((source/'manifest.json').read_text())['initial_roster']
        me = next(r['object_id'] for r in roster if r['name'] == game['champion'])
        native = loader.load_event_game(local)
        fields = _Fields(native, loader)
        events = []
        counts = Counter()
        for name, kind in [('movement.received_path', 'path'), ('event.basic_attack_execution_proxy', 'attack'),
                           ('event.ability_cast_received_proxy', 'cast')]:
            rows, times, ids = fields.rows(name, 'champion')
            for row, tm, uid in zip(rows, times, ids):
                if uid != me or not 30. <= tm < 840.:
                    continue
                v = fields.value(name, row)
                order = int(native['fields'][name]['update_index'][row])
                event = dict(t=float(tm), order=order, kind=kind)
                if kind == 'path':
                    event.update(multipoint=len(v['waypoints_xz']) > 1,
                                 special=v.get('special_motion_parameters') is not None)
                    counts['paths'] += 1
                    counts['special_paths'] += int(event['special'])
                    counts['special_multipoint_paths'] += int(event['special'] and event['multipoint'])
                elif kind == 'cast':
                    event['slot'] = v['slot']
                events.append(event)
        same_clock = defaultdict(list)
        for event in events:
            same_clock[event['t']].append(event)
        examples = []
        for tm, rows in same_clock.items():
            casts = [r for r in rows if r['kind']=='cast' and r['slot'] in range(4)]
            paths = [r for r in rows if r['kind']=='path' and r['multipoint']]
            for cast in casts:
                for path in paths:
                    counts['same_clock_cast_path_pairs'] += 1
                    counts['cast_before_path_source_order'] += int(cast['order'] < path['order'])
                    counts['cast_path_same_source_order'] += int(cast['order'] == path['order'])
                    if cast['order'] < path['order'] and len(examples) < 3:
                        examples.append(dict(cast=cast, path=path))
        cast_times = np.array([e['t'] for e in events if e['kind']=='cast' and e['slot'] in range(4)])
        for event in events:
            if event['kind']=='path' and event['special'] and len(cast_times):
                counts['special_path_within_200ms_qwer'] += int(np.min(np.abs(cast_times-event['t'])) <= .2)
        results.append(dict(game=game['game'], champion=game['champion'], counts=dict(counts), examples=examples))
        del native, fields, events
        shutil.rmtree(local)
        gc.collect()
        ctypes.CDLL('libc.so.6').malloc_trim(0)
    total = Counter()
    for result in results:
        total.update(result['counts'])
    return dict(counts=dict(total), games=results,
                limits='raw 30-840s records before alive/corridor filtering; source order is update_index; special motion is not proof of an issued move')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('dataset', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--decoder', type=Path)
    p.add_argument('--scratch', type=Path)
    args = p.parse_args()
    if args.decoder and args.scratch is None:
        p.error('--decoder requires --scratch for private source staging')
    result = profile_sources(args.dataset, args.decoder, args.scratch) if args.decoder else profile(args.dataset)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: dict(counts=v['counts'], quantiles=v['quantiles']) for k, v in result['groups'].items()} if 'groups' in result else result['counts'], indent=2))
