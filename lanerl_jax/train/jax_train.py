"""Experimental JAX farming launcher using the source collector's PPO loop.

Same actor, reward, update code and decision budget; different dynamics.
Production routing and map visibility are mandatory. Matching this task does
not establish server parity: consult the fidelity ledger before interpreting
learning curves, including STAT-003 and automatic-rank event timing.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import shutil
import sys

from ..sim.config import DEFAULT_ROUTE_ARTIFACT, SimConfig
from .jax_farm import JaxFarmCollector, GROUND_CLICK_NORMALIZATION
from .policy import LanePolicy, PolicyConfig
from .ppo import PPOConfig
from .run_manifest import RunDir, file_sha256
from .server_train import WAVE_START_MS, run_farming_learner, snapshot_farming_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--envs', type=int, default=2)
    parser.add_argument('--rollout', type=int, default=128)
    parser.add_argument('--updates', type=int, default=2)
    parser.add_argument('--save-updates', type=int, nargs='*', default=[],
                        help='Pin checkpoints at matched decision budgets')
    parser.add_argument('--episode-s', type=float, default=600.)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--critic-lr', type=float, default=None)
    parser.add_argument('--entropy-coef', type=float, default=PPOConfig().entropy_coef)
    parser.add_argument('--epochs', type=int, default=PPOConfig().epochs)
    parser.add_argument('--minibatches', type=int, default=1)
    parser.add_argument('--normalize-advantage', action='store_true')
    parser.add_argument('--opponent', choices=('idle', 'mirror'), default='idle')
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument('--init-from', type=Path, default=None, help='params only: fresh optimizer/schedule/budget')
    parser.add_argument('--lr-anneal', action='store_true')
    parser.add_argument('--core', choices=('mlp', 'gru'), default='mlp')
    parser.add_argument('--preset', choices=('legacy', 'standard'), default='legacy')
    parser.add_argument('--ckpt-every', type=int, default=10)
    parser.add_argument('--eval-episodes', type=int, default=0)
    parser.add_argument('--batch-mode', choices=['auto', 'map', 'vmap'], default='auto')
    parser.add_argument('--start-near-wave', action='store_true')
    parser.add_argument('--step-ticks', type=int, default=2)
    parser.add_argument('--route-artifact', type=Path, default=DEFAULT_ROUTE_ARTIFACT)
    parser.add_argument('--out', type=Path,
                        default=Path('lanerl_jax/runs/server_first_20260925/jax'))
    args = parser.parse_args()
    args.route_artifact = args.route_artifact.resolve()
    if min(args.envs, args.rollout, args.updates, args.episode_s, args.step_ticks) <= 0:
        parser.error('envs, rollout, updates, episode-s and step-ticks must be positive')
    if args.start_near_wave and args.episode_s <= WAVE_START_MS / 1000:
        parser.error('wave-start episodes must end after 120 game seconds')
    teams = (0,) if args.opponent == 'idle' else (0, 1)
    hz = 60. / args.step_ticks
    if args.preset == 'standard':
        cfg = PPOConfig.standard(decision_hz=hz)
    else:
        cfg = PPOConfig(lr=args.lr, critic_lr=args.lr if args.critic_lr is None else args.critic_lr,
                        entropy_coef=args.entropy_coef, epochs=args.epochs,
                        normalize_advantage=args.normalize_advantage, decision_hz=hz,
                        gae_lambda=PPOConfig().gae_lambda ** (args.step_ticks / 2.))
    import jax, json as _json
    jax.config.update("jax_default_matmul_precision", "highest")
    resume_params_only = False
    if args.init_from is not None:
        args.resume, resume_params_only = args.init_from, True
    pcfg = PolicyConfig(core=args.core)
    if args.resume is not None and (args.resume.parent / 'manifest.json').exists():
        saved = _json.loads((args.resume.parent / 'manifest.json').read_text()).get('config', {}).get('train', {}).get('policy', {})
        if saved:
            pcfg = PolicyConfig(**saved)
    policy = LanePolicy(pcfg)
    command = shlex.join([sys.executable, '-m', 'lanerl_jax.train.jax_train', *sys.argv[1:]])
    run = RunDir(args.out, f'jax-farm-s{args.seed}', {
        'train': {'policy': policy.cfg._asdict()}, 'command': command,
        'cwd': str(Path.cwd()), 'ppo': cfg._asdict(), 'collector': vars(args),
        'environment': 'jax-experimental-farming',
        'ground_click_normalization': GROUND_CLICK_NORMALIZATION,
        'observation': 'viewport+map-fog+structured-HUD',
        'opponent': 'idle-fountain' if args.opponent == 'idle' else 'mirror-self-play',
        'reward': 'CS - 2*death + 5*(lane_potential_next - potential) + 0.005*xp',
        'initialization': 'random; no prior',
        'limitations': ['Not established source-server dynamics parity.',
                       'Native spell ranks precede extra rank-time NOOP decisions.',
                       'STAT-003: source magic-resist rune bonus is missing from JAX.']})
    snapshot_farming_source(run, command)
    collector = None
    try:
        sim = SimConfig.training(route_artifact=args.route_artifact).replace(
            step_ticks=args.step_ticks)
        if sim.route_table is None or sim.vision is None or sim.lane_path is None:
            raise RuntimeError('JAX farming requires production routes, map visibility and waves')
        route_manifest = args.route_artifact / 'manifest.json'
        shutil.copyfile(route_manifest, run.path / 'route-manifest.json')
        run.manifest['sim'] = {
            'resolved': sim.describe(), 'fingerprint': sim.fingerprint(),
            'route_manifest_sha256': file_sha256(route_manifest)}
        run.write()
        collector = JaxFarmCollector(args.envs, run.path, args.episode_s,
            args.start_near_wave, args.step_ticks, seed=args.seed, sim_config=sim,
            batch_mode=args.batch_mode, teams=teams)
    except BaseException as exc:
        if collector is not None:
            collector.close()
        run.set_results(status='failed', error=repr(exc))
        raise
    if args.eval_episodes:
        from flax.serialization import from_state_dict, msgpack_restore
        from .server_train import evaluate_frozen
        obs0, _ = collector.observe()
        params = policy.init(jax.random.key(args.seed), obs0.entities, obs0.entity_pad_mask,
                             obs0.self_vec, obs0.global_vec)
        if args.resume is not None:
            params = from_state_dict(params, msgpack_restore(Path(args.resume).read_bytes())["params"])
        run.set_results(evaluated_checkpoint=str(args.resume) if args.resume else "random",
                        checkpoint_sha256=file_sha256(args.resume) if args.resume else None)
        evaluate_frozen(collector, policy, params, run, seed=args.seed,
                        episodes_per_env=args.eval_episodes)
        return
    run_farming_learner(collector, policy, cfg, run, seed=args.seed,
                        rollout=args.rollout, updates=args.updates,
                        save_updates=args.save_updates, n_minibatches=args.minibatches,
                        resume=args.resume, ckpt_every=args.ckpt_every, lr_anneal=args.lr_anneal,
                        resume_params_only=resume_params_only)


if __name__ == '__main__':
    main()
