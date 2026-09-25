"""Record a checkpoint through the real rollout path for client-free inspection.

Run on the login node only through ops/login_capped.sh. Records every decision
before its action, keeps both teams visible for diagnosis, and never resets.
Rendering is separate so it needs neither JAX nor a running game server.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
import subprocess
import tarfile

import numpy as np


def summarize(d):
    """Decision-weighted measures, using identity-aware swing transitions."""
    results = []
    for side in range(2):
        alive = d['alive'][:, side]
        enemy = (d['team'] != side) & (d['kind'] == 2) & d['alive']
        dist = np.hypot(d['x'] - d['x'][:, side, None],
                        d['y'] - d['y'][:, side, None])
        nearest = np.min(np.where(enemy, dist, np.inf), axis=1)
        target = d['order_target'][:, side].astype(int)
        safe = np.clip(target, 0, d['x'].shape[1] - 1)
        rows = np.arange(len(safe))
        attack = d['order_kind'][:, side] == 2  # OrderKind.ATTACK
        hostile = d['team'][rows, safe] != side
        minion = d['kind'][rows, safe] == 2
        turret = d['kind'][rows, safe] == 3
        engaged = d['is_attacking'][:, side]
        swing_target = d['aa_target'][:, side]
        swing_safe = np.clip(swing_target, 0, d['x'].shape[1]-1)
        seq = d['spawn_seq'][rows, swing_safe]
        minion_windup = (engaged & (swing_target >= 0)
                         & (d['kind'][rows, swing_safe] == 2)
                         & (d['team'][rows, swing_safe] != side))
        switched = ((d['aa_target'][1:, side] != d['aa_target'][:-1, side])
                    | (seq[1:] != seq[:-1]))
        restarts = engaged[1:] & engaged[:-1] & switched
        results.append(dict(
            side=('blue', 'red')[side], cs=int(d['cs'][-1, side]),
            deaths=int(d['deaths'][-1, side]),
            alive_fraction=float(alive.mean()),
            near_enemy_minion_fraction=float((alive & (nearest <= 250)).sum() / max(alive.sum(), 1)),
            median_nearest_minion_when_alive=float(np.median(nearest[alive & np.isfinite(nearest)]))
                if (alive & np.isfinite(nearest)).any() else None,
            minimum_nearest_minion_when_alive=float(nearest[alive].min())
                if (alive & np.isfinite(nearest)).any() else None,
            enemy_minion_windup_frames=int(minion_windup.sum()),
            attack_enemy_minion_fraction=float((attack & hostile & minion).mean()),
            attack_enemy_champion_fraction=float((attack & hostile & (d['kind'][rows, safe] == 1)).mean()),
            attack_ally_fraction=float((attack & ~hostile).mean()),
            attack_own_turret_fraction=float((attack & ~hostile & turret).mean()),
            windup_target_switches=int(restarts.sum()),
        ))
    return results


def render_metadata():
    """Static labels and camera geometry, exported alongside recorded state."""
    from lanerl_jax.obs.frame import make_lane_frame
    from lanerl_jax.sim.init import TOP_OUTER_TURRET
    from lanerl_jax.sim.profiles import PROFILES
    from lanerl_jax.train.trainer import BLUE_NEXUS
    frame = make_lane_frame(TOP_OUTER_TURRET[0], TOP_OUTER_TURRET[1], BLUE_NEXUS)
    labels = {0: 'melee', 1: 'caster', 2: 'cannon', 3: 'super'}
    return dict(camera_frame=dict(axis=np.asarray(frame.axis).tolist(),
                                  normal=np.asarray(frame.normal).tolist()),
                profiles=[dict(kind=int(k), subtype=int(sub), team=int(team),
                               label=('Garen' if k == 1 else labels.get(sub, 'minion')
                                      if k == 2 else 'turret')) for k, sub, team in PROFILES])


def state_snapshot(state, orders, action, cursor):
    """Shared diagnostic schema; caller supplies resolved orders and raw cursor.

    State is privileged replay data, never an actor observation. Works inside
    JIT or with host arrays; both farming and generic replays use this schema.
    """
    fields = ('t_ms', 'x', 'y', 'hp', 'max_hp', 'alive', 'kind', 'team', 'model',
              'spawn_seq', 'target', 'aa_target', 'is_attacking', 'has_auto_attacked',
              'aa_windup', 'aa_cooldown', 'cs', 'deaths', 'level',
              'route_status', 'waypoint_key', 'n_waypoints', 'spell_cooldown',
              'spell_level', 'r_cast_ms', 'missile_alive', 'missile_x', 'missile_y',
              'missile_tx', 'missile_source', 'missile_source_seq',
              'missile_source_model', 'missile_damage', 'missile_speed')
    out = {k: getattr(state, k) for k in fields}
    out['waypoints'] = state.waypoints[:2]
    out.update(screen_x=action[1], screen_y=action[2],
               click_x=cursor.x, click_y=cursor.y, button=action[0],
               order_kind=orders.kind, order_target=orders.target,
               order_x=orders.x, order_y=orders.y,
               q_active=state.buffs.q.active[:2], w_active=state.buffs.w.active[:2],
               e_active=state.buffs.e.active[:2])
    return out


def record(args):
    import jax
    import jax.numpy as jnp
    from flax.serialization import from_bytes
    from lanerl_jax.obs.builder import build_observation
    from lanerl_jax.obs.frame import make_lane_frame
    from lanerl_jax.sim.config import SimConfig, DEFAULT_ROUTE_ARTIFACT
    from lanerl_jax.sim.init import TOP_OUTER_TURRET
    from lanerl_jax.sim.terrain_jax import map1_terrain
    from lanerl_jax.train.actions import orders_from
    from lanerl_jax.train.policy import PolicyConfig
    from lanerl_jax.train.ppo import PPOConfig
    from lanerl_jax.train.trainer import TrainConfig, make_train, BLUE_NEXUS, RED_NEXUS
    from lanerl_jax.train.run_manifest import file_sha256, git_provenance, git_environment

    checkpoint = None
    if not args.scripted:
        checkpoint = args.checkpoint.resolve()
        manifest = json.loads((checkpoint.parent / 'manifest.json').read_text())
        conf = manifest['config']['train']
        if conf['policy'].get('action_interface') != 'screen-click-v2':
            raise ValueError('checkpoint predates screen-click-v2; use its recorded trace for legacy diagnosis')
        if conf['policy'].get('observation_interface') != 'viewport-structured-v3':
            raise ValueError('checkpoint does not declare v3 authoritative life-state observations; use its historical capture')
    # Snapshot source BEFORE running: another agent may edit the shared tree
    # while the compiled rollout is running. A final dirty hash is insufficient.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    source = git_provenance()
    root = Path(__file__).resolve().parents[1]
    source_tar = args.out.with_suffix('.source.tar.gz')
    paths = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others',
                                     '--exclude-standard'], cwd=root, env=git_environment(root)).decode().split('\0')
    with tarfile.open(source_tar, 'w:gz') as archive:
        for name in sorted(set(paths)):
            path = root / name
            if path.is_file() and path.suffix in {'.py', '.sh', '.html', '.toml', '.patch'}:
                archive.add(path, arcname=name)
    source['snapshot'] = str(source_tar.resolve())
    source['snapshot_sha256'] = file_sha256(source_tar)
    sim = SimConfig.training(route_artifact=DEFAULT_ROUTE_ARTIFACT)
    frames = [make_lane_frame(TOP_OUTER_TURRET[i], TOP_OUTER_TURRET[1-i], nexus)
              for i, nexus in enumerate((BLUE_NEXUS, RED_NEXUS))]

    def snapshot(state, orders, action):
        cursor = orders_from(action, state, None, frames[0], snap_moves=False, vision=sim.vision)
        return state_snapshot(state, orders, action, cursor)

    if args.scripted:
        from lanerl_jax.train.scripted_policy import PLAYERS
        from lanerl_jax.sim.init import init_lane
        from lanerl_jax.sim.step import env_step
        player = PLAYERS[args.scripted]
        runner = init_lane(seed=args.seed)
        def body(state, _):
            obs = [build_observation(state, i, frames[i], params=sim.params,
                                     horizon_s=args.seconds, vision=sim.vision) for i in range(2)]
            choices = [player(o, jax.random.key(args.seed)) for o in obs]
            action = tuple(jnp.stack(pair) for pair in zip(*choices))
            orders = orders_from(action, state, None, frames[0], params=sim.params, vision=sim.vision)
            return env_step(state, orders, sim), snapshot(state, orders, action)
        label = args.label or f'Scripted {args.scripted} mirror (coordinate clicks)'
    else:
        cfg = TrainConfig(n_envs=1, rollout_steps=1, n_updates=1, n_minibatches=1,
                          episode_s=conf['episode_s'], policy=PolicyConfig(**conf['policy']),
                          ppo=PPOConfig(**conf['ppo']))
        built = make_train(cfg, sim_config=sim)
        runner = built.initial_runner(jax.random.key(args.seed))
        payload = from_bytes({'params': runner.params, 'opt_state': runner.opt_state,
                              'step': runner.step}, checkpoint.read_bytes())
        runner = runner._replace(params=payload['params'], deadline_ms=jnp.full((1,), jnp.inf))
        def body(runner, _):
            state = jax.tree.map(lambda a: a[0], runner.env_state)
            runner, tr = built.rollout(runner)
            action = tuple(a[0, 0] for a in tr.action)
            orders = orders_from(action, state, None, frames[0], params=sim.params, vision=sim.vision)
            return runner, snapshot(state, orders, action)
        label = args.label or checkpoint.parent.name

    chunk = jax.jit(lambda r: jax.lax.scan(body, r, None, length=300))
    logs, started = [], time.monotonic()
    while not logs or float(logs[-1]['t_ms'][-1]) < args.seconds * 1000:
        runner, out = chunk(runner)
        logs.append(jax.tree.map(np.asarray, out))
        if len(logs) % 6 == 0 or len(logs) == 1:
            print(f"game {float(logs[-1]['t_ms'][-1])/1000:.1f}s / {args.seconds}s; "
                  f"wall {time.monotonic()-started:.1f}s", flush=True)
    data = {k: np.concatenate([r[k] for r in logs]) for k in logs[0]}
    stop = int(np.searchsorted(data['t_ms'], args.seconds * 1000)) + 1
    data = {k: v[:stop] for k, v in data.items()}
    terrain = map1_terrain()
    meta = dict(label=label, checkpoint=str(checkpoint) if checkpoint else None,
                checkpoint_sha256=file_sha256(checkpoint) if checkpoint else None,
                controller=args.scripted or "learned", action_interface="screen-click-v2", seed=args.seed,
                **render_metadata(),
                simulation=sim.describe(), source=source,
                seconds=args.seconds, view='omniscient; not the policy observation',
                action_alignment='each frame is pre-action; order arrows show that decision',
                terrain=dict(min_x=float(terrain.min_x), min_y=float(terrain.min_y),
                             cell_size=float(terrain.cell_size)))
    meta['summary'] = summarize(data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, **data, walkable=np.asarray(terrain.walkable),
                        metadata=np.asarray(json.dumps(meta)))
    args.out.with_suffix('.json').write_text(json.dumps(meta, indent=2) + '\n')
    print(json.dumps(meta['summary'], indent=2), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    control = p.add_mutually_exclusive_group(required=True)
    control.add_argument('--checkpoint', type=Path)
    control.add_argument('--scripted', choices=('lasthit', 'brawler', 'noop'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--label', default='')
    p.add_argument('--seed', type=int, default=123)
    p.add_argument('--seconds', type=float, default=600)
    args = p.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0 or args.out.suffix != '.npz':
        p.error('seconds must be positive and --out must end in .npz')
    record(args)


if __name__ == '__main__':
    main()
