"""PERF-004: attribute GPU events to HLO source scopes, preserving mixed fusions.

No proportional attribution is invented: a kernel with multiple logical
sources is a shared bucket. Per-phase inclusive time is an upper bound and
must not be summed. GPU busy time uses the union of stream-event intervals.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
from pathlib import Path
import re


def phase_of(name):
    labels = re.findall(r'perf004_([A-Za-z0-9_]+)', name)
    if not labels:
        return 'unattributed'
    for label in reversed(labels):
        if label.startswith('tick_'):
            return label
    for label in ('apply_orders', 'post_forward', 'post_metrics', 'ppo_adam',
                  'ppo_apply', 'ppo_grad_norm', 'ppo_grad', 'gae', 'bootstrap',
                  'learner_metrics', 'batch_layout', 'ppo_shuffle', 'observe',
                  'decode', 'sampling', 'reward', 'reset_and_record',
                  'click_mask', 'policy_actor', 'simulation'):
        if label in labels:
            return label
    return labels[-1]


def family_of(phase):
    if phase.startswith('tick_') or phase == 'simulation':
        return 'simulation'
    if phase.startswith('ppo_'):
        return 'PPO'
    if phase in ('post_forward', 'post_metrics', 'learner_metrics'):
        return 'learner_diagnostics'
    return phase


def operation_kind(name, opcode=''):
    s = name.lower()
    if 'memcpy' in s or 'memset' in s:
        return 'memory_transfer'
    if 'sort' in s:
        return 'sort'
    if any(x in s for x in ('gemm', 'matmul', 'mma')) or opcode == 'dot':
        return 'matmul'
    if 'transpose' in s:
        return 'transpose'
    if 'reduce' in s or 'reduction' in s:
        return 'reduction'
    if 'scatter' in s:
        return 'scatter'
    if 'gather' in s:
        return 'gather'
    if 'fusion' in s:
        return 'other_fusion'
    return 'other'


class HloSources:
    def __init__(self, text):
        self.instructions, self.computations = {}, defaultdict(list)
        current = None
        for line in text.splitlines():
            m = re.match(r'^(?:ENTRY )?%?([\w.-]+).*\{$', line)
            if m:
                current = m[1]
                continue
            m = re.match(r'^  (?:ROOT )?%?([\w.-]+) = (.*)', line)
            if not m:
                continue
            name, body = m.groups()
            op = re.search(r'\b([a-z][a-z0-9_-]*)\(', body)
            sources = re.findall(r'op_name="((?:\\.|[^"\\])*)"', body)
            calls = re.findall(r'(?:calls|to_apply|body|condition)=%?([\w.-]+)', body)
            self.instructions[name] = dict(op=op[1] if op else '', sources=sources, calls=calls)
            self.computations[current].append(name)
        self._cache = {}

    def sources(self, name, seen=None):
        name = str(name).lstrip('%')
        if name in self._cache:
            return self._cache[name]
        info = self.instructions.get(name)
        if info is None:
            return set()
        seen = set() if seen is None else seen
        if name in seen:
            return set()
        seen.add(name)
        result = set(info['sources'])
        for computation in info['calls']:
            for child in self.computations.get(computation, ()):
                result.update(self.sources(child, seen))
        seen.remove(name)
        self._cache[name] = result
        return result

    def info(self, name, fallback=''):
        sources = self.sources(name)
        if not sources and fallback:
            sources = {fallback}
        phases = {phase_of(s) for s in sources}
        phases.discard('unattributed')
        phases = phases or {'unattributed'}
        return phases, sources


def union_ns(intervals):
    total, end = 0., float('-inf')
    for start, stop in sorted(intervals):
        total += max(0., stop - max(start, end))
        end = max(end, stop)
    return total


def _rows(times, counts, total):
    return [dict(name=k, ms=v / 1e6, fraction=v / total if total else 0.,
                 events=counts[k]) for k, v in times.most_common()]


def summarize(trace_dir, hlo_path, wall_s):
    from jax.profiler import ProfileData
    paths = sorted(Path(trace_dir).rglob('*.xplane.pb'))
    if len(paths) != 1:
        raise ValueError(f'expected one xplane in {trace_dir}, got {len(paths)}')
    with gzip.open(hlo_path, 'rt') if str(hlo_path).endswith('.gz') else open(hlo_path) as f:
        hlo = HloSources(f.read())
    pd = ProfileData.from_file(str(paths[0]))
    times = {k: Counter() for k in ('phase_exclusive_or_shared', 'phase_inclusive',
                                  'family_exclusive_or_shared', 'kernel', 'operation_kind',
                                  'network_module', 'autodiff_label')}
    counts = {k: Counter() for k in times}
    all_intervals, durations = [], []
    mapped = missing = 0
    source_examples = {}
    for plane in pd.planes:
        if 'GPU' not in plane.name:
            continue
        for line in plane.lines:
            if 'stream' not in line.name.lower():
                continue
            for event in line.events:
                stats = dict(event.stats)
                duration = event.duration_ns
                all_intervals.append((event.start_ns, event.start_ns + duration))
                durations.append(duration)
                op = str(stats.get('hlo_op', ''))
                phases, sources = hlo.info(op, str(stats.get('name', '')))
                if op in hlo.instructions:
                    mapped += 1
                else:
                    missing += 1
                source_examples.setdefault(op, dict(phases=sorted(phases), sources=sorted(sources)))
                family = {family_of(p) for p in phases}
                phase_key = '+'.join(sorted(phases))
                family_key = '+'.join(sorted(family))
                modules = set()
                labels = set()
                for s in sources:
                    if 'transpose(' in s:
                        labels.add('backward-labelled')
                    elif 'ppo_grad' in s:
                        labels.add('forward-or-residual')
                    if 'core_gru' in s:
                        modules.add('GRU')
                    elif '_Block_' in s or 'Block_' in s:
                        if 'MultiHead' in s:
                            modules.add('attention')
                        elif 'LayerNorm' in s:
                            modules.add('transformer_norm')
                        else:
                            modules.add('transformer_other')
                    elif 'value_head' in s:
                        modules.add('value_head')
                    elif 'LanePolicy' in s:
                        modules.add('policy_other')
                buckets = {'phase_exclusive_or_shared': [phase_key],
                           'phase_inclusive': phases,
                           'family_exclusive_or_shared': [family_key],
                           'kernel': [event.name],
                           'operation_kind': [operation_kind(event.name, hlo.instructions.get(op, {}).get('op', ''))],
                           'network_module': ['+'.join(sorted(modules)) or 'non_network_or_unknown'],
                           'autodiff_label': ['+'.join(sorted(labels)) or 'other']}
                for group, keys in buckets.items():
                    for key in keys:
                        times[group][key] += duration
                        counts[group][key] += 1
    total = sum(durations)
    if not durations:
        raise RuntimeError(f'no GPU stream events in {paths[0]}')
    for key in times:
        if key != 'phase_inclusive':
            assert abs(sum(times[key].values()) - total) <= max(total * 1e-9, 1.)
    start, stop = min(s for s, _ in all_intervals), max(e for _, e in all_intervals)
    busy = union_ns(all_intervals)
    ds = sorted(durations)
    result = dict(file=str(paths[0]), hlo=str(hlo_path), profiled_wall_s=wall_s,
                  stream_events=len(durations), gpu_span_s=(stop-start)/1e9,
                  gpu_busy_s=busy/1e9, gpu_busy_fraction_of_span=busy/(stop-start),
                  event_sum_s=total/1e9, hlo_mapped_events=mapped, hlo_missing_events=missing,
                  kernel_duration_us={str(q): ds[min(len(ds)-1, int(q*(len(ds)-1)))] / 1e3
                                      for q in (.5, .9, .99)},
                  events_under_5us=sum(x < 5000 for x in durations),
                  tables={k: _rows(times[k], counts[k], total) for k in times},
                  source_examples=source_examples,
                  caveat='Shared fusion buckets are not split; phase_inclusive is non-additive. Profiled wall is not unprofiled runtime.')
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    a = p.parse_args()
    manifest = json.loads((a.out / 'profile_manifest.json').read_text())
    summaries = {}
    for entry in manifest['traces']:
        tag = entry['tag']
        print('analyze', tag, flush=True)
        summaries[tag] = summarize(a.out / entry['trace'], a.out / entry['hlo'], entry['wall_s'])
        (a.out / f'{tag}_analysis.json').write_text(json.dumps(summaries[tag], indent=2))
        print(tag, 'events', summaries[tag]['stream_events'], 'busy', summaries[tag]['gpu_busy_fraction_of_span'], flush=True)
    compact = {k: {n: v for n, v in d.items() if n != 'source_examples'} for k, d in summaries.items()}
    (a.out / 'analysis.json').write_text(json.dumps(compact, indent=2))


if __name__ == '__main__':
    main()
