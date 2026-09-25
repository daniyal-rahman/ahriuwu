"""Summarize recorded source-server farming without running either simulator.

Wire observations establish proximity and outcomes; full-rate AI diagnostics
establish observed attack-state transitions. Held targets are not damage victims.
"""
import argparse
import bisect
import struct
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
import time

from .trace import INTERNAL_RE, ROW_RE, parse_internal, parse_row


def attack_transition(previous, current):
    """Observable edges, not proof of damage or a particular victim."""
    if previous is None:
        return []
    events = []
    if current.is_attacking and not previous.is_attacking:
        events.append('attack_started')
    if current.has_auto_attacked and not previous.has_auto_attacked:
        events.append('attack_cast_finished')
    if previous.is_attacking and not current.is_attacking:
        events.append('attack_ended_after_cast' if current.has_auto_attacked or
                      previous.has_auto_attacked else 'attack_ended_without_observed_cast')
    return events


class ETimeline:
    """Buff observations establish ends; input proximity alone proves no cause."""
    def __init__(self, command_times):
        self.commands = sorted(command_times)
        self.previous = None
        self.open = None
        self.episodes = []
        self.counts = Counter()
        self.active_ms = 0.

    @staticmethod
    def phase(buffs):
        if buffs is None:
            return None, False
        entry = next((b for b in buffs if b[0] == 'GarenE'), None)
        if entry is None:
            return None, True
        elapsed, duration = [struct.unpack('<f', struct.pack('<I', int(v) & 0xffffffff))[0]
                             for v in entry[1:]]
        if not all(math.isfinite(v) for v in (elapsed, duration)) or duration < 0:
            return None, False
        return (elapsed, duration), True

    def add(self, t, buffs, dead=None):
        phase, known = self.phase(buffs)
        if not known:
            self.counts['unknown_phase_samples'] += 1
            if self.open is not None:
                self.open.update(end_observed=False, censor_reason='missing_phase_instrumentation')
                self.episodes.append(self.open)
                self.open = None
            self.previous = None
            return
        prev = self.previous
        if prev is not None:
            pt, pp = prev
            if t <= pt:
                raise ValueError('E diagnostic times must increase')
            lo, hi = bisect.bisect_left(self.commands, pt), bisect.bisect_left(self.commands, t)
            inputs = self.commands[lo:hi]
            if pp is not None:
                self.active_ms += t-pt
                self.counts['e_inputs_while_observed_active'] += len(inputs)
                if pp[0] < 1.:
                    self.counts['e_inputs_during_first_second'] += len(inputs)
            if pp is not None and phase is None:
                remaining_ms = (pp[1]-pp[0])*1000.
                # Tick quantization can straddle natural expiry. Never classify
                # that edge as premature simply because last seen elapsed<3s.
                premature = remaining_ms > t-pt+2.
                row = self.open or dict(first_seen_ms=pt, start_observed=False)
                row.update(end_observed=True, end_ms=t, last_active_ms=pt,
                    last_elapsed_s=pp[0], duration_s=pp[1],
                    observed_span_ms=t-row['first_seen_ms'],
                    end_kind='premature' if premature else 'expiry_compatible',
                    dead_at_end=dead, e_inputs_in_last_interval=inputs,
                    confirmed_key_cancellation=None)
                self.episodes.append(row)
                self.counts['observed_ends'] += 1
                self.counts[row['end_kind']+'_ends'] += 1
                if premature and inputs:
                    self.counts['premature_ends_with_nearby_e_input'] += 1
                self.open = None
        if phase is not None:
            if self.open is None:
                observed = prev is not None and prev[1] is None
                self.open = dict(first_seen_ms=t, start_observed=observed,
                                 initial_elapsed_s=phase[0], duration_s=phase[1])
                self.counts['observed_starts' if observed else 'left_censored_episodes'] += 1
            self.open['last_active_ms'] = t
            self.open['last_elapsed_s'] = phase[0]
        self.previous = (t, phase)

    def report(self):
        episodes = list(self.episodes)
        if self.open is not None:
            episodes.append(dict(self.open, end_observed=False, censor_reason='recording_end'))
        return dict(counts=dict(self.counts), observed_active_seconds=self.active_ms/1000.,
            episodes=episodes, confirmed_key_cancellations=None,
            limitations=[
                'No spell-execution acknowledgement is consumed: key cancellation is unconfirmed.',
                'Nearby E input plus premature buff end is an association, not proof of cancellation.',
                'Expiry-compatible allows the final diagnostic interval and2ms clock tolerance.',
                'Active duration holds the previous buff state until the next diagnostic sample.',
                'Missing phase instrumentation censors episodes instead of inventing buff ends.'])


LEGACY_OBSERVATIONS = ('viewport-structured-v1', 'viewport-structured-v2')


def source_alive(champion, observation_interface=None):
    """Diagnostic compatibility only; production actors never use HP fallback."""
    if 'dead' in champion:
        if type(champion['dead']) is not bool:
            raise ValueError('malformed authoritative dead flag: Boolean required')
        return not champion['dead'], 'authoritative_dead'
    if observation_interface not in LEGACY_OBSERVATIONS:
        raise ValueError('missing authoritative dead flag; HP fallback requires explicit historical v1/v2 provenance')
    return champion['hp'] > 0, 'legacy_hp_proxy'


def recording_interface(recording, legacy_override=None):
    metadata = recording/'evaluation.json'
    evaluation = json.loads(metadata.read_text()) if metadata.exists() else {}
    interface = evaluation.get('observation_interface')
    candidates = [recording/'input/manifest.json']
    checkpoint = evaluation.get('checkpoint')
    if checkpoint:
        candidates.append(Path(checkpoint).parent/'manifest.json')
        # The same NFS run can be recorded under desktop's /mnt mount prefix.
        marker = 'lanerl_jax/runs/'
        if marker in checkpoint:
            candidates.append(Path(__file__).resolve().parents[2] / marker /
                              Path(checkpoint.split(marker, 1)[1]).parent / 'manifest.json')
    if interface is None:
        for path in candidates:
            if path.exists():
                interface = json.loads(path.read_text()).get('config', {}).get('train', {}).get('policy', {}).get('observation_interface')
                if interface is not None:
                    break
    if legacy_override and interface and legacy_override != interface:
        raise ValueError('legacy override conflicts with recorded observation interface')
    return interface or legacy_override


def analyze(recording, start_ms=120000, team=100, legacy_observation_interface=None):
    recording = Path(recording)
    interface = recording_interface(recording, legacy_observation_interface)
    alive_methods = Counter()
    paths = [recording/'policy_policy_actions.json', recording/'policy_obs.jsonl',
             recording/'policy/instance000.log']
    provenance = {}
    raw = paths[0].read_bytes()
    provenance[str(paths[0])] = hashlib.sha256(raw).hexdigest()
    actions = json.loads(raw)
    side = 'blue' if team == 100 else 'red'
    by_time = dict(zip(actions['t_ms'], actions[side]))
    e_timeline = ETimeline([t for t, a in by_time.items()
        if a.get('t') == 'click' and a.get('button') == 'e'])
    counts, near_buttons, all_buttons = Counter(), Counter(), Counter()
    events, previous, champion_id = [], None, None
    nearest = float('inf')
    wire_hash = hashlib.sha256()
    wire_gaps, previous_t = [], None
    with paths[1].open('rb') as f:
        for raw in f:
            wire_hash.update(raw)
            frame = json.loads(raw)
            if frame['t'] < start_ms:
                continue
            if previous_t is not None:
                wire_gaps.append(frame['t'] - previous_t)
            previous_t = frame['t']
            c = next(u for u in frame['u'] if u['k'] == 'Champion' and u['tm'] == team)
            champion_id = c['id']
            living, alive_method = source_alive(c, interface)
            alive_methods[alive_method] += 1
            enemies = [u for u in frame['u'] if u['k'] == 'Minion' and
                       u['tm'] != team and u['hp'] > 0]
            # Some collectors name the same lane unit LaneMinion.
            enemies += [u for u in frame['u'] if u['k'] == 'LaneMinion' and
                        u['tm'] != team and u['hp'] > 0]
            distance = min((math.hypot(u['x']-c['x'], u['y']-c['y']) for u in enemies), default=float('inf'))
            if living:
                nearest = min(nearest, distance)
            counts['wire_frames'] += 1
            counts['alive_frames'] += living
            counts['held_target_frames'] += bool(c.get('tgt', 0))
            counts['wire_attacking_frames'] += bool(c.get('atk', 0))
            action = by_time.get(frame['t'])
            if action is not None:
                button = action.get('button', action['t'])
                all_buttons[button] += 1
                if living and distance <= 250:
                    near_buttons[button] += 1
                if action['t'] == 'click' and button in ('move', 'attack_move'):
                    click_distance = min((math.hypot(u['x']-action['x'], u['y']-action['y']) for u in enemies), default=float('inf'))
                    counts['movement_clicks_within_35_of_enemy_minion_center'] += click_distance <= 35
            if living and distance <= 250:
                counts['alive_frames_with_enemy_minion_within_250'] += 1
            if previous:
                for key in ('cs',):
                    if c.get(key, 0) > previous.get(key, 0):
                        events.append(dict(t_ms=frame['t'], event=key, value=c[key], x=c['x'], y=c['y']))
                for slot in range(4):
                    key = f'cd{slot}'
                    if c.get(key, 0) > previous.get(key, 0) + 100:
                        counts[f'spell_{slot}_cooldown_rises'] += 1
            previous = c
    if previous is None:
        raise ValueError('No champion observations in requested interval')
    provenance[str(paths[1])] = wire_hash.hexdigest()
    counts.update(final_cs=previous.get('cs', 0))
    previous_ai, last_t, gaps = None, None, []
    log_hash = hashlib.sha256()
    needle = f'ai id={champion_id} '.encode()
    diagnostic_dead = None
    with paths[2].open('rb') as f:
        for raw in f:
            log_hash.update(raw)
            if f' Champion|{team}|'.encode() in raw and b'LANERL_STATEROW' in raw:
                row = ROW_RE.search(raw.decode())
                if row and int(row[1]) >= start_ms:
                    entity = parse_row(row[2])
                    counts['final_deaths'] = entity.champ.deaths
                    diagnostic_dead = entity.dead
            if needle not in raw:
                continue
            match = INTERNAL_RE.search(raw.decode())
            if not match or int(match[1]) < start_ms:
                continue
            t = int(match[1])
            ai = parse_internal(match[2], match[3])
            e_timeline.add(t, ai.buffs_phase, dead=diagnostic_dead)
            counts['diagnostic_frames'] += 1
            counts['diagnostic_attacking_frames'] += ai.is_attacking
            counts['diagnostic_hasaa_frames'] += ai.has_auto_attacked
            counts['diagnostic_held_target_frames'] += ai.target_net_id != 0
            if last_t is not None:
                gaps.append(t-last_t)
            for event in attack_transition(previous_ai, ai):
                counts[event] += 1
                events.append(dict(t_ms=t, event=event))
            previous_ai, last_t = ai, t
    provenance[str(paths[2])] = log_hash.hexdigest()
    return dict(recording=str(recording), start_ms=start_ms, team=team,
        observation_interface=interface,
        source_alive_method=next(iter(alive_methods)) if len(alive_methods)==1 else 'mixed',
        source_alive_methods=dict(alive_methods),
        counts=dict(counts), closest_enemy_minion_distance=nearest if math.isfinite(nearest) else None,
        commands=dict(all_buttons), commands_while_minion_within_250=dict(near_buttons),
        wire_mean_gap_ms=sum(wire_gaps) / len(wire_gaps) if wire_gaps else None,
        wire_max_gap_ms=max(wire_gaps, default=None),
        diagnostic_max_gap_ms=max(gaps, default=None), events=sorted(events, key=lambda x:x['t_ms']),
        inputs_sha256=provenance, e_activity=e_timeline.report(),
        limitations=['Historical v1/v2 HP fallback is explicitly labelled and can misclassify positive-HP corpses.',
          'Wire proximity is sampled at the reported wire intervals and includes omniscient diagnostic entities.',
          'AI transitions use every recorded champion diagnostic row; gaps are reported.',
          'hasaa rising records cast completion/launch, not damage landing or its victim.',
          'An attack ending without observed cast is consistent with cancellation, not proof of its cause.',
          'Cooldown rises are cast proxies and may include rank/reset effects; button counts are emitted commands, not suppressed raw policy choices.',
          'Clicks within 35 units of a minion center are geometric proximity only, not server hit-test results.'])


def analyze_replay(path, start_ms=0):
    """Analyze NPZ state samples; duration uses left-held recorded intervals.

    Sampling can hide complete swings or target changes between frames. The
    event list describes observable edges only, for either recorded engine.
    """
    import numpy as np
    from types import SimpleNamespace
    from ..replay import summarize

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {k: archive[k] for k in archive.files if k not in ('metadata', 'walkable')}
        metadata = json.loads(str(archive['metadata']))
    keep = data['t_ms'] >= start_ms
    data = {k: value[keep] for k, value in data.items()}
    times = data['t_ms'].astype(float)
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('replay timestamps must be finite and strictly increasing')
    dt = np.diff(times, append=times[-1])
    alive = data['alive'][:, 0].astype(bool)
    enemies = (data['kind'] == 2) & (data['team'] != data['team'][:, :1]) & data['alive']
    distances = np.hypot(data['x']-data['x'][:, :1], data['y']-data['y'][:, :1])
    nearest = np.min(np.where(enemies, distances, np.inf), axis=1)
    near = alive & (nearest <= 250)
    windows, start = [], None
    for i, enabled in enumerate(near):
        if enabled and start is None:
            start = times[i]
        if start is not None and (not enabled or i == len(times)-1):
            windows.append(dict(start_ms=float(start), end_ms=float(times[i])))
            start = None
    events, counts, previous = [], Counter(), None
    for i, t in enumerate(times):
        current = SimpleNamespace(is_attacking=bool(data['is_attacking'][i, 0]),
                                  has_auto_attacked=bool(data['has_auto_attacked'][i, 0]))
        for event in attack_transition(previous, current):
            counts[event] += 1
            events.append(dict(t_ms=float(t), event=event))
        if i:
            for name in ('cs', 'deaths'):
                value, old = int(data[name][i, 0]), int(data[name][i-1, 0])
                if value > old:
                    event = dict(t_ms=float(t), event=name, value=value, delta=value-old,
                        x=float(data['x'][i, 0]), y=float(data['y'][i, 0]),
                        enemy_minions_within_800=int((enemies[i] & (distances[i] <= 800)).sum()))
                    if 'e_active' in data:
                        event['e_active_in_sample'] = bool(data['e_active'][i, 0])
                    events.append(event)
        previous = current
    counts.update(attacking_samples=int(data['is_attacking'][:, 0].sum()),
        held_target_samples=int((data['target'][:, 0] >= 0).sum()),
        near_minion_samples=int(near.sum()))
    return dict(trace=str(path), environment=metadata.get('environment'),
        checkpoint_sha256=metadata.get('checkpoint_sha256'),
        trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        samples=len(times), start_ms=float(times[0]), end_ms=float(times[-1]),
        max_gap_ms=float(np.max(np.diff(times))) if len(times)>1 else None,
        alive_seconds=float(dt[alive].sum()/1000),
        near_minion_seconds=float(dt[near].sum()/1000),
        counts=dict(counts), summary=summarize(data), near_minion_windows=windows,
        events=sorted(events, key=lambda e:e['t_ms']),
        limitations=['Events are sampled flag/counter edges, not full-rate event logs.',
            'Duration assigns each interval its left sample; short events may be missed.',
            'Attack completion markers are not independently verified damage or victims.',
            'E active and nearby units at CS/death are context, not causal attribution.',
            'Only the recorded interval is analyzed; omitted warmup is not reconstructed.'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('recordings', nargs='+', type=Path)
    p.add_argument('--out', required=True, type=Path)
    p.add_argument('--start-ms', type=int, default=120000)
    p.add_argument('--legacy-observation-interface', choices=LEGACY_OBSERVATIONS,
                   help='Explicit provenance for unversioned historical recordings only')
    p.add_argument('--replay', action='store_true', help='Inputs are recorded NPZ traces; --start-ms filters the recorded interval')
    a = p.parse_args()
    started = time.monotonic()
    source = Path(__file__).read_bytes()
    reports = [analyze_replay(path, a.start_ms) if a.replay else analyze(path, a.start_ms, legacy_observation_interface=a.legacy_observation_interface)
               for path in a.recordings]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix('.source.py').write_bytes(source)
    a.out.write_text(json.dumps(dict(command=sys.argv, runtime_s=time.monotonic()-started,
        analyzer_sha256=hashlib.sha256(source).hexdigest(), reports=reports), indent=2)+'\n')
    for r in reports:
        print(r.get('recording', r.get('trace')), json.dumps(r['counts']), flush=True)


if __name__ == '__main__':
    main()
