"""Read-only final-endpoint scorer for LEARN-AFK-29's paired feature study.

Run under ops/login_capped.sh. Refuses incomplete or unmatched studies; emits
JSON to stdout, without selecting a best intermediate checkpoint.
"""
import argparse
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINTS = Path('/mnt/nfs/checkpoints/lanerl-jax')


def load(path):
    return json.loads(Path(path).read_text())


def read_eval(path, update):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    selected = [r for r in rows if r['update'] == update]
    if len(selected) != 1:
        raise ValueError(f'{path}: expected exactly one evaluation at update {update}')
    r = selected[0]
    if r.get('frozen') is not True or r['games'] != 64 or r['duration_s'] != 120:
        raise ValueError('requires frozen 64-game, 120-second evaluation')
    episodes = {(e['env'], e['team']): e for e in r['episodes']}
    if len(episodes) != len(r['episodes']) or set(episodes) != {(e,t) for e in range(64) for t in (0,1)}:
        raise ValueError('missing or duplicate evaluation episodes')
    return episodes


def check_retention(initial, reference):
    for key in reference:
        for field in ('cs', 'deaths', 'kills', 'spell_selections', 'low_hp'):
            if initial[key][field] != reference[key][field]:
                raise ValueError(f'initial retention differs: {key}/{field}')
        for field in ('gold', 'gold_diff', 'tower_damage', 'hp_fraction', 'reward'):
            if not math.isclose(initial[key][field], reference[key][field], rel_tol=1e-5, abs_tol=1e-4):
                raise ValueError(f'initial numeric retention differs: {key}/{field}')


def study(experiment):
    spec = load(ROOT/'experiments'/f'{experiment}.json')
    status = load(CHECKPOINTS/experiment/'study.json')
    if status['status'] != 'complete' or status['update'] != 512 or spec['updates'] != 512:
        raise ValueError(f'{experiment}: final study incomplete')
    path = Path(status['path'])
    manifest = load(path/'manifest.json')
    if manifest['config']['scenario'] != spec:
        raise ValueError('run manifest differs from versioned experiment')
    return spec, status, read_eval(path/'evaluations.jsonl',0), read_eval(path/'evaluations.jsonl',512)


def score(control, feature):
    a, ast, ai, af = study(control)
    b, bst, bi, bf = study(feature)
    allowed = {'id','question','combat_features'}
    if {k for k in set(a)|set(b) if a.get(k)!=b.get(k)} - allowed:
        raise ValueError('experiment settings differ beyond the feature switch')
    if a.get('combat_features') is not False or b.get('combat_features') is not True:
        raise ValueError('expected original-input control and combat-input feature arm')
    ref = a['initial_eval_reference']
    reference = read_eval(ref['path'], ref['update'])
    check_retention(ai,reference);check_retention(bi,reference)
    if any(af[k]['low_hp'] != bf[k]['low_hp'] or af[k]['low_hp'] != ai[k]['low_hp'] for k in ai):
        raise ValueError('final evaluation cohorts differ')
    blue = [(e,0) for e in range(64)]
    def summary(rows):
        result = {f:mean(rows[k][f] for k in blue) for f in ('cs','deaths','kills','tower_damage','reward')}
        if not all(math.isfinite(v) for v in result.values()):
            raise ValueError('nonfinite final evaluation')
        return result
    ca,cb=summary(af),summary(bf)
    delta=[bf[k]['cs']-af[k]['cs'] for k in blue]
    passed=cb['cs']>=10.921875 and cb['cs']>=ca['cs']+1 and cb['deaths']<=.15
    return dict(control=control,feature=feature,control_job=ast['job'],feature_job=bst['job'],
        endpoint_update=512,games=64,initial_retention_passed=True,
        baseline=summary(reference),control_final=ca,feature_final=cb,
        paired_cs_delta=mean(delta),games_cs_higher=sum(x>0 for x in delta),
        games_cs_equal=sum(x==0 for x in delta),games_cs_lower=sum(x<0 for x in delta),
        primary_gate_passed=passed,
        conclusion='Promising; independent confirmation required.' if passed else
                   'No support for this bounded feature fine-tune under the predeclared gate.',
        limits='One training seed; selected AFK task; no full Tencent or C# transfer claim.')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--control',default='E75c_combat_feature_control')
    parser.add_argument('--feature',default='E76c_combat_feature_inputs')
    args=parser.parse_args()
    print(json.dumps(score(args.control,args.feature),indent=2))
