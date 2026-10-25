"""Information-loss and replay-qualification diagnostics, not policy quality gates."""
import argparse
from collections import Counter
import copy
import gc
import json
from pathlib import Path
import shutil
import sys

import numpy as np

from lanerl_jax.modern.bc import replay as R
from lanerl_jax.modern.bc.event_inputs import previous_received_events, SELF_FIELDS
from lanerl_jax.modern.bc.events import BUTTONS


def synthetic_findings():
    from lanerl_jax.modern.bc.tests.test_replay_fields import fixture, build
    original = fixture()
    baseline = build(original)
    aliases = {}
    for name in ('haste_and_attack_speed', 'absolute_hp', 'absolute_mana'):
        changed = copy.deepcopy(original)
        if name == 'haste_and_attack_speed':
            changed.champion_stats[:, 0, 4] *= 2
            changed.champion_stats[:, 0, 6] = 100.
        elif name == 'absolute_hp':
            changed.hp[:, 0] *= 2
            changed.max_hp[:, 0] *= 2
            changed.champion_stats[:, 0, 7] *= 2
        else:
            original.max_mana[:] = 100.
            original.mana[:] = 50.
            baseline = build(original)
            changed = copy.deepcopy(original)
            changed.max_mana[:] = 1000.
            changed.mana[:] = 500.
            changed.champion_stats[:, 0, 8] *= 10
        aliases[name] = all(np.array_equal(a, b, equal_nan=True) for a, b in zip(baseline, build(changed)))
    times = np.arange(4)*.1
    first = R.Action(.05, 'attack', target=99)
    extra = R.Action(.15, 'attack', target=99)
    args = (np.zeros((4, 2)), np.full((4, 32), -1), np.array([1., 0.]), np.array([0., 1.]))
    left = previous_received_events(times, [first], *args)
    right = previous_received_events(times, [first, extra], *args)
    aliases['latest_repeated_attack_execution'] = all(np.array_equal(a, b, equal_nan=True) for a, b in zip(left, right))

    class DamageFields:
        def rows(self, name, family=None):
            return np.array([0]), np.array([10.02 if 'damage' in name else 10.]), np.array([50])
        def value(self, name, row):
            return dict(receiver_id=50, participant_ids=[50, 1], victim_id=None, source_id=None,
                        victim_source_roles_available=False, consumer_application_executed=False)
    fields = DamageFields()
    notices = R._damage_notices(fields)
    credits = R._last_hits(fields, me=1, death_times={50: [10.]})
    return dict(input_aliases=aliases, omitted_self_fields=[n for n in ('attack_speed', 'ability_haste', 'max_hp', 'max_mana') if n not in SELF_FIELDS],
                unqualified_damage_notice_promoted=notices, death_credited_before_notice=credits,
                limits='synthetic equality proves information loss/contract violations; not a measured effect on learned policy quality')


def position_findings(decoder):
    sys.path.insert(0, str(decoder))
    import position_model
    times = np.array([.5, 1.5, 2.5])
    path = dict(object_id=1, waypoints_xz=[[0., 0.], [1000., 0.]], speed_f32=300., flags=4)
    class Fields:
        def __init__(self, special):
            self.path = dict(path)
            if special:
                self.path['special_motion_parameters'] = dict(scalar_operands_0_to_0c=[1000., 0., 0., 0.])
        def rows(self, name, family=None):
            if name == 'movement.received_path' and family == 'champion':
                return np.array([0]), np.array([0.]), np.array([1])
            if name == 'event.unit_death_received_proxy':
                return np.array([0]), np.array([1.]), np.array([1])
            return np.array([], int), np.array([]), np.array([], int)
        def value(self, name, row):
            return self.path if name == 'movement.received_path' else dict(victim_id=1)
        def asof(self, name, uid, query):
            return np.array([100., 0., 100.]) if name == 'stat.mHP' and uid == 1 else np.full(3, 100.)
        def series(self, name, uid):
            if name == 'stat.mHP':
                return dict(time_seconds=np.array([0., 2.]), values=np.array([100., 100.]), available=np.ones(2, bool))
            return dict(time_seconds=np.array([0.]), values=np.array([300.]), available=np.ones(1, bool))
    champions = [R.Unit(1, 'champion', 0, 'Garen'), R.Unit(2, 'champion', 1, 'Jax')]
    ordinary = R._read_lane_units(Fields(False), position_model, champions, times, 2)
    special = R._read_lane_units(Fields(True), position_model, champions, times, 2)
    model = position_model.MinionPositions([], phase_entity_ids=[1], champion_ids=[1])
    model.movement(0., [Fields(True).path])
    evidence = model.estimate(1, .5)
    return dict(postdeath_position_from_predeath_path=float(ordinary[5][2, 0]),
                source_path_time=0., death_time=1., respawn_hp_receipt=2., query_time=2.5,
                special_position_consumed=bool(np.isfinite(special[5][0, 0])),
                special_policy_position_accepted=evidence['policy_position_accepted'],
                special_approximation_accepted=evidence['approximation_accepted_for_bc'],
                limits='synthetic source-contract reproductions, not measured corpus frequencies')


def memorization_support(root):
    rows = [r for r in json.loads((root/'manifest.json').read_text())['games'] if r['split'] == 'train']
    arrays = []
    for row in rows:
        with np.load(root/'games'/(row['game']+'.npz')) as f:
            arrays.append({k: f[k] for k in ('button', 'valid', 'aim_valid', 'aim_mode')})
    reports = []
    total = 0
    for champion, mode in [('Ahri', 2), ('Garen', 1)]:
        options = []
        for gi, (row, x) in enumerate(zip(rows, arrays)):
            if row['champion'] != champion:
                continue
            for t in np.flatnonzero(x['valid'] & x['aim_valid'] & (x['aim_mode'] == mode)):
                options.append((int(3 <= x['button'][t] <= 6), gi, max(0, int(t)-32)))
        _, gi, start = max(options)
        x = arrays[gi]
        valid = x['valid'][start:start+64].copy()
        valid[:8] = False
        button = x['button'][start:start+64]
        counts = {name: int((valid & (button == b)).sum()) for b, name in enumerate(BUTTONS)}
        aims = {name: int((valid & (button == b) & x['aim_valid'][start:start+64]).sum()) for b, name in enumerate(BUTTONS)}
        total += int(valid.sum())
        reports.append(dict(game=rows[gi]['game'], start=start, button_counts=counts, aim_counts=aims))
    return dict(windows=reports, total_valid=total, two_missed_casts_accuracy=(total-2)/total,
                two_missed_casts_nll_at_probability_point2=float(-2*np.log(.2)/total),
                limits='replicates current two-window selector; aggregate strict gates do not require per-button/target success')


def raw_findings(plan, decoder, scratch):
    sys.path.insert(0, str(decoder))
    import event_training_loader as loader
    import position_model
    rows = json.loads(plan.read_text())['games']
    reports = []
    for row in rows:
        source = Path(row['path'])/'export'
        local = scratch/'native'
        size = sum(p.stat().st_size for p in (source/'event-training').rglob('*') if p.is_file())
        assert shutil.disk_usage(scratch).free > size+20*1024**3
        shutil.copytree(source/'event-training', local, symlinks=False)
        meta = json.loads((source/'manifest.json').read_text())
        native = loader.load_event_game(local)
        fields = R._Fields(native, loader)
        roster = meta['initial_roster']
        me = next(r['object_id'] for r in roster if r['name'] == row['champion'])
        own_row = next(r for r in roster if r['object_id'] == me)
        opponent = next(r['object_id'] for r in roster if r['team'] != own_row['team'] and r['assigned_role'] == own_row['assigned_role'])
        t = R.decision_times(30., 840.)
        counts, examples = Counter(), []
        damage_rows, damage_times, _ = fields.rows('event.received_damage_notice')
        damage = {}
        for index, clock in zip(damage_rows, damage_times):
            if not 30. <= clock < 840.:
                continue
            d = fields.value('event.received_damage_notice', index)
            counts['damage_records'] += 1
            counts['damage_unqualified_roles'] += not d.get('victim_source_roles_available', False)
            p = d['participant_ids']
            if len(p) > 1:
                damage.setdefault(int(d['receiver_id']), []).append((float(clock), int(p[1])))
        death_times = {}
        for index, clock, _ in zip(*fields.rows('event.unit_death_received_proxy')):
            uid = fields.value('event.unit_death_received_proxy', index)['victim_id']
            death_times.setdefault(uid, []).append(float(clock))
        minion_ids = set(map(int, fields.rows('event.unit_death_received_proxy', 'minion')[2]))
        for uid, deaths in death_times.items():
            if uid not in minion_ids:
                continue
            clock = deaths[0]
            if not 30. <= clock < 840.:
                continue
            eligible = [r for r in damage.get(uid, []) if r[0] <= clock+R._LAST_HIT_TOLERANCE_S]
            if eligible and eligible[-1][1] == me:
                counts['current_cs_rule_credits'] += 1
                counts['cs_rule_credits_future_notice'] += eligible[-1][0] > clock
        ids = [r['object_id'] for r in roster]
        model = position_model.MinionPositions([], phase_entity_ids=ids, champion_ids=ids)
        for index, clock, _ in zip(*fields.rows('movement.received_path', 'champion')):
            model.movement(float(clock), [fields.value('movement.received_path', index)])
        for uid in (me, opponent):
            speed = fields.series('stat.mMoveSpeed', uid)
            for clock, value, known in zip(speed['time_seconds'], speed['values'], speed['available']):
                if known:
                    model.speed(float(clock), uid, float(value))
            positive_hp = fields.series('stat.mHP', uid)
            positives = positive_hp['time_seconds'][positive_hp['available'] & (positive_hp['values'] > 0)]
            deaths = death_times.get(uid, [])
            for clock in t:
                alive = not any(d <= clock < R._first_after(positives, d) for d in deaths)
                if not alive:
                    continue
                e = model.estimate(uid, float(clock))
                if not e['position_available']:
                    continue
                counts['alive_champion_position_queries'] += 1
                if not e.get('policy_position_accepted') and not e.get('approximation_accepted_for_bc'):
                    counts['available_but_not_accepted_positions'] += 1
                earlier_deaths = [d for d in deaths if d <= clock]
                stale = earlier_deaths and e['source']['movement_clock'] < max(earlier_deaths)
                if stale:
                    counts['alive_position_uses_predeath_path'] += 1
                    if len(examples) < 3:
                        examples.append(dict(unit=uid, time=float(clock), last_death=max(earlier_deaths),
                                             path_time=e['source']['movement_clock']))
        for name in ('mAbilityHasteMod', 'mAttackSpeedMod', 'mPercentAttackSpeedMod', 'mMaxHP', 'mMaxPAR'):
            values = fields.asof('stat.'+name, me, t)
            counts[name+'_finite'] = int(np.isfinite(values).sum())
            counts[name+'_positive'] = int((values > 0).sum())
        reports.append(dict(game=row['game'], counts=dict(counts), examples=examples))
        del native, fields, model
        gc.collect()
        shutil.rmtree(local)
    total = Counter()
    for report in reports:
        total.update(report['counts'])
    return dict(counts=dict(total), reports=reports,
                limits='24-perspective convenience sample; own/paired positions queried before fog/corridor filtering; receipt-time and qualification diagnostics only')


if __name__ == '__main__':
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--fixture', type=Path)
    p.add_argument('--plan', type=Path)
    p.add_argument('--decoder', type=Path)
    p.add_argument('--scratch', type=Path)
    args = p.parse_args()
    result = dict(synthetic=synthetic_findings())
    if args.decoder:
        result['position_contracts'] = position_findings(args.decoder)
    if args.fixture:
        result['memorization_support'] = memorization_support(args.fixture)
    if args.plan:
        if args.decoder is None or args.scratch is None:
            p.error('raw audit requires private scratch and decoder snapshot')
        result['raw'] = raw_findings(args.plan, args.decoder, args.scratch)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result.get('raw', result['synthetic']).get('counts', result['synthetic']), indent=2))
