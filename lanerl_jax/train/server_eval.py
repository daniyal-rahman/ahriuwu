"""Frozen-policy source-server farming evaluation with diagnostic recordings."""
import argparse
import json
import time
import math
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

from ..parity.policy_divergence import record_policy_run
from ..parity.policy_driver import scan_cast_freeze, champion_dead, validate_champion_life
from .run_manifest import file_sha256, git_provenance, git_environment


def summarize_frames(frames):
    """Behavioral evidence from real wire frames, separate from optimization."""
    out = {}
    for team in (100, 200):
        positions, closest, near = [], math.inf, 0
        deaths = attack_starts = attack_frames = minion_target_frames = 0
        prev_alive, prev_attacking = None, False
        for frame in frames:
            me = next(u for u in frame['u'] if u.get('k') == 'Champion' and u['tm'] == team)
            positions.append((me['x'], me['y']))
            live = not champion_dead(me)
            attacking = bool(me.get('atk', 0))
            deaths += prev_alive is True and not live
            attack_starts += attacking and not prev_attacking
            attack_frames += attacking
            minion_ids = {u['id'] for u in frame['u'] if u.get('k') == 'LaneMinion'
                          and u['tm'] != team and u['hp'] > 0}
            minion_target_frames += me.get('tgt', 0) in minion_ids
            prev_alive, prev_attacking = live, attacking
            if champion_dead(me):
                continue
            d = min((math.hypot(u['x']-me['x'], u['y']-me['y']) for u in frame['u']
                     if u.get('k') == 'LaneMinion' and u['tm'] != team and u['hp'] > 0), default=math.inf)
            closest = min(closest, d)
            near += d < 250
        out[str(team)] = dict(bounds=[[min(p[i] for p in positions), max(p[i] for p in positions)]
                                      for i in (0, 1)], final_position=positions[-1],
            min_enemy_minion_distance=closest if math.isfinite(closest) else None,
            frames_within_250=near, deaths=deaths,
            observed_attack_starts=attack_starts, attacking_frames=attack_frames,
            minion_target_frames=minion_target_frames,
            note='Attack flags are sampled; starts may be missed and are not confirmed hits.')
    contact = None
    for frame in frames:
        damaged = [u for u in frame['u'] if u.get('k') == 'LaneMinion' and 0 < u['hp'] < u['mhp']-1]
        if damaged:
            contact = {'t_ms': frame['t'], 'units': damaged}
            break
    return {'champions': out, 'first_minion_damage': contact, 'frames': len(frames)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--seconds', type=float, default=600.)
    p.add_argument('--port-base', type=int, default=49500)
    p.add_argument('--start-near-wave', action='store_true')
    p.add_argument('--step-ticks', type=int, default=2)
    p.add_argument('--server-dir', type=Path)
    a = p.parse_args()
    if a.server_dir is not None:
        a.server_dir = a.server_dir.resolve()
    if a.step_ticks < 1:
        p.error('step-ticks must be positive')
    # Reserve before launching: evaluation recordings must never overwrite an
    # earlier episode, and end-of-run git metadata cannot preserve loaded code.
    a.out.mkdir(parents=True, exist_ok=False)
    source = git_provenance()
    root = Path(__file__).resolve().parents[2]
    names = subprocess.check_output(
        ['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'],
        cwd=root, env=git_environment(root)).decode().split('\0')
    with tarfile.open(a.out / 'source.tar.gz', 'w:gz') as archive:
        for name in sorted(set(names)):
            path = root / name
            if name and path.is_file() and path.suffix in (
                    '.py', '.sh', '.html', '.toml', '.patch', '.md', '.json'):
                archive.add(path, arcname=name)
    source['snapshot_sha256'] = file_sha256(a.out / 'source.tar.gz')
    from lanerl_train import paths as server_paths
    vendor = server_paths.server_dir().parents[3]
    (a.out / 'vendor.patch').write_bytes(subprocess.check_output(
        ['git', 'diff', '--binary', 'HEAD'], cwd=vendor, env=git_environment(vendor)))
    source['server_binary_sha256'] = file_sha256((a.server_dir or server_paths.server_dir()) / 'GameServerLib.dll')
    source['server_config_sha256'] = file_sha256(server_paths.default_game_config())
    source['vendor_head'] = subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], cwd=vendor, env=git_environment(vendor)).decode().strip()
    (a.out / 'command.txt').write_text(shlex.join(
        [sys.executable, '-m', 'lanerl_jax.train.server_eval', *sys.argv[1:]]) + '\n')
    (a.out / 'provenance.json').write_text(json.dumps(source, indent=2))
    started = time.monotonic()
    setup_driver = None
    if a.start_near_wave:
        from .server_train import WaveStart, WAVE_START_MS
        if a.seconds <= WAVE_START_MS / 1000:
            p.error('wave-start evaluation must last beyond 120 game seconds')
        setup = WaveStart()
        finished = False
        def setup_driver(obs, i, pair):
            nonlocal finished
            validate_champion_life(obs)
            if not finished:
                me = next(u for u in obs['u'] if u.get('k') == 'Champion' and u['tm'] == 100)
                cmd = setup.order(me, obs['t'])
                if cmd is not None:
                    pair.ranks.observe(obs)
                    pair.log.append(obs['t'], cmd, {'t': 'noop'}, None, None)
                    return {'blue': cmd, 'red': {'t': 'noop'}}
                finished = True
            return pair(obs, i)
    _, actions = record_policy_run(str(a.checkpoint), a.out, seconds=a.seconds,
        red='idle', deterministic=False, seed=a.seed, port_base=a.port_base,
        setup_driver=setup_driver, step_ticks=a.step_ticks, server_dir=a.server_dir)
    with (a.out / 'policy_obs.jsonl').open() as f:
        frames = [json.loads(line) for line in f]
    detector = scan_cast_freeze(frames)
    last = {u['tm']: u for u in frames[-1]['u'] if u.get('k') == 'Champion'}
    result = dict(checkpoint=str(a.checkpoint), checkpoint_sha256=file_sha256(a.checkpoint),
        seed=a.seed, seconds=a.seconds, wall_s=time.monotonic()-started,
        freeze=detector.report(), counts=actions.meta['driver_counts'],
        cs={str(tm): u['cs'] for tm, u in last.items()}, source=source,
        opponent='idle-fountain', environment='source-server', start_near_wave=a.start_near_wave,
        step_ticks=a.step_ticks, decision_hz=60. / a.step_ticks,
        observation_interface='viewport-structured-v3', life_state='authoritative champion dead Boolean')
    result['behavior'] = summarize_frames(frames)
    (a.out / 'evaluation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    if detector.invalid:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
