"""Synchronous source-server rollouts feeding the shared Flax/PPO learner.

No JAX dynamics step is called. First task: blue farms against minions while
red idles in its fountain. Random initialization; no imitation/reference loss.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import shlex
import sys
import tarfile
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from lanerl_rl.constants import BUTTONS, SCREEN_X_VALUES, SCREEN_Y_VALUES
from lanerl_rl.projection import (screen_to_world_centred, MINIMAP_X_MIN,
                                   MINIMAP_Y_MIN)
from lanerl_train.ports import PortAllocator
from lanerl_train import paths as server_paths
from lanerl_train.vec import ServerLaunchSpec, VecLaneEnv
from ..obs.builder import build_observation
from ..parity.policy_driver import (StateRebuilder, _lane_frames,
                                    wire_visibility, pending_rank_up,
                                    CastFreezeDetector, wire_own_hud, apply_own_hud,
                                    champion_dead, validate_champion_life)
from .learner import make_learner
from .policy import LanePolicy, PolicyConfig
from .ppo import (PPOConfig, gae, kl_stopped_epochs, summarise_minibatches,
                  factored_log_prob)
from .reward import lane_corridor_distance
from .run_manifest import RunDir, file_sha256, git_environment
from .trainer import _sample


def screen_order(action, champion, frame):
    """Project buttons/cursor to wire; never accepts an entity ID."""
    button, ix, iy = map(int, action)
    if not (0 <= button < len(BUTTONS) and 0 <= ix < len(SCREEN_X_VALUES)
            and 0 <= iy < len(SCREEN_Y_VALUES)):
        raise ValueError("action outside screen-click-v2 grid")
    if champion_dead(champion):
        return {"t": "noop"}
    name = BUTTONS[button]
    if name in ("noop", "recall"):
        return {"t": name}
    sx, sy = float(SCREEN_X_VALUES[ix]), float(SCREEN_Y_VALUES[iy])
    if name in ("move", "attack_move", "r") and sx >= MINIMAP_X_MIN and sy >= MINIMAP_Y_MIN:
        return {"t": "noop"}
    if name in ("q", "w", "e", "r"):
        if champion["sl"][("q", "w", "e", "r").index(name)] <= 0:
            return {"t": "noop"}
    ds, dn = screen_to_world_centred(0., 0., sx, sy)
    axis, normal = np.asarray(frame.axis), np.asarray(frame.normal)
    return {"t": "click", "button": name,
            "x": float(champion["x"] + ds * axis[0] + dn * normal[0]),
            "y": float(champion["y"] + ds * axis[1] + dn * normal[1])}


def source_farm_stats(champion, potential):
    """Reward life state comes from the server flag, never its regenerating HP."""
    return champion["cs"], not champion_dead(champion), float(potential)


def farm_reward(cs_before, cs_after, alive_before, alive_after,
                potential_before, potential_after, done, gamma):
    """Explicit task reward, shared by future matched-environment collector.

    +1 per CS, -2 per death, plus lane-approach potential shaping. Terminal
    potential is zero. No opponent term, damage bonus or ambient-gold reward.
    """
    cs = cs_after - cs_before
    death = -2. * (alive_before & ~alive_after)
    shaping = 5. * (gamma * jnp.where(done, 0., potential_after) - potential_before)
    return cs + death + shaping, {"cs": cs, "death": death, "approach": shaping}


WAVE_START_MS = 120_000
# Measured first minion contact in the untouched source-server control:
# 124.909 s, blue (2157,12474), red (2259,12573). Start behind blue's wave.
WAVE_START_POS = (1950., 12350.)


class WaveStart:
    """Fixed reset setup only; never contributes actions to the PPO batch."""
    def __init__(self):
        self.moved = False

    def order(self, champion, t_ms):
        if champion_dead(champion) or champion['cs'] != 0:
            raise RuntimeError('wave-start setup died or farmed before policy control')
        if t_ms >= WAVE_START_MS:
            if np.hypot(champion['x']-WAVE_START_POS[0], champion['y']-WAVE_START_POS[1]) > 100:
                raise RuntimeError(f"wave-start setup missed {WAVE_START_POS}: "
                                   f"position=({champion['x']}, {champion['y']}), hp={champion['hp']}")
            return None
        rank = pending_rank_up(champion)
        if rank is not None:
            return {'t': 'level', 'slot': rank}
        if not self.moved:
            self.moved = True
            return {'t': 'move', 'x': WAVE_START_POS[0], 'y': WAVE_START_POS[1]}
        return {'t': 'noop'}


class ServerCollector:
    def __init__(self, n, out, port_base, episode_s, start_near_wave=False, step_ticks=2,
                 server_dir=None):
        self.n, self.out, self.episode_s = n, Path(out), episode_s
        self.frame = _lane_frames()[0]
        self.env = VecLaneEnv(n, spec=ServerLaunchSpec(
            bot_teams="none", step_ticks=step_ticks, toponly=True,
            server_dir=server_dir,
            extra_env={"LANERL_AUTOBUY": "0"}),
            log_dir=self.out / "server", ports=PortAllocator(base=port_base).allocate(n),
            auto_restart=False)
        self.rebuilders = [StateRebuilder() for _ in range(n)]
        self.detectors = [CastFreezeDetector() for _ in range(n)]
        self.episodes = [0] * n
        self.initial_stats = [None] * n
        self.start_near_wave = start_near_wave
        params = self.rebuilders[0].params
        self.encode = jax.jit(lambda s, vis: build_observation(
            s, 0, self.frame, params=params, horizon_s=episode_s, visibility=vis))
        self.potential = jax.jit(lambda s: -lane_corridor_distance(s.x[:2], s.y[:2])[0] / 10000.)
        try:
            self.env.start()
            for raw in self.env.last_obs:
                validate_champion_life(raw)
                wire_own_hud(raw, 0)  # fail before setup if the binary lacks required HUD fields
            self._rank()
            self.initial_stats = [tuple(self.champion(i)[k] for k in ('mhp', 'ad', 'ar', 'mr'))
                                  for i in range(self.n)]
            self._prepare()
        except BaseException:
            self.env.close()
            raise

    def _prepare(self, indices=None):
        if not self.start_near_wave:
            return
        indices = list(range(self.n)) if indices is None else list(indices)
        setups = {i: WaveStart() for i in indices}
        count = 0
        # Only newly reset processes advance. Other environments remain paused.
        while setups:
            actions = [None] * self.n
            active = []
            for i in list(setups):
                cmd = setups[i].order(self.champion(i), self.env.last_obs[i]['t'])
                if cmd is None:
                    del setups[i]
                else:
                    actions[i] = {'blue': cmd, 'red': {'t': 'noop'}}
                    active.append(i)
                    if count % 300 == 0 or cmd['t'] != 'noop':
                        with (self.out / 'setup.jsonl').open('a') as f:
                            f.write(json.dumps({'env': int(i), 't': self.env.last_obs[i]['t'],
                                                'champion': self.champion(i), 'command': cmd}) + '\n')
            if active:
                for i in active:
                    self.env.handles[i].send_line(json.dumps(actions[i]))
                result = self.env._collect(active)
                if result.died:
                    raise RuntimeError(f'wave setup lost a server: {result.died}')
            count += 1

    def _rank(self):
        # Automatic fixed skill progression is task setup. These transitions
        # are not attributed to a sampled policy action.
        for _ in range(18):
            slots = [pending_rank_up(self.champion(i)) for i in range(self.n)]
            if all(s is None for s in slots):
                return
            # A level-up in one process must not silently advance its peers.
            # VecLaneEnv.step(None) advances that process with a NOOP; send
            # only to the instances whose HUD reports an available skill point.
            active = [i for i, slot in enumerate(slots) if slot is not None]
            for i in active:
                self.env.handles[i].send_line(json.dumps({
                    'blue': {'t': 'level', 'slot': slots[i]}, 'red': {'t': 'noop'}}))
            result = self.env._collect(active)
            if result.died:
                raise RuntimeError(f'skill progression lost a server: {result.died}')
        raise RuntimeError("skill progression failed to settle")

    def champion(self, i):
        return next(u for u in self.env.last_obs[i]["u"]
                    if u.get("k") == "Champion" and u["tm"] == 100)

    def spell_ranks(self):
        """Own blue ranks for sampled-action diagnostics; not actor features."""
        return np.asarray([self.champion(i)["sl"] for i in range(self.n)], dtype=np.int32)

    def observe(self):
        observations, stats = [], []
        for i, raw in enumerate(self.env.last_obs):
            self.detectors[i].observe(raw)
            if self.detectors[i].invalid:
                raise RuntimeError(self.detectors[i].reason())
            state, ids = self.rebuilders[i].rebuild(raw)
            if self.rebuilders[i].dropped_minions:
                raise RuntimeError("server observation exceeded entity capacity")
            obs = self.encode(state, wire_visibility(raw, ids, 0))
            me = self.champion(i)
            # Own HUD is authoritative, not reconstructed simulator stats.
            obs = apply_own_hud(obs, wire_own_hud(raw, 0))
            observations.append(obs)
            stats.append(source_farm_stats(me, self.potential(state)))
            if self.initial_stats[i] is None:
                self.initial_stats[i] = tuple(me[k] for k in ("mhp", "ad", "ar", "mr"))
        return jax.tree.map(lambda *v: jnp.stack(v), *observations), np.asarray(stats)

    def step(self, actions):
        self.env.step([{"blue": screen_order(a, self.champion(i), self.frame),
                        "red": {"t": "noop"}} for i, a in enumerate(actions)])
        self._rank()
        return np.asarray([o["t"] >= self.episode_s * 1000 for o in self.env.last_obs])

    def restart_done(self, done):
        for i in np.flatnonzero(done):
            # Fresh process preserves runes; the legacy in-process reset does not.
            self.episodes[i] += 1
            self.env.handles[i].close()
            h = self.env._default_factory(int(i), self.env.ports[i])
            self.env.handles[i] = h
            h.start()
            result = self.env._collect([int(i)])
            if result.died:
                raise RuntimeError(f"fresh server failed: {result.died}")
            self.rebuilders[i] = StateRebuilder()
            self.detectors[i] = CastFreezeDetector()
            validate_champion_life(self.env.last_obs[i])
            me = self.champion(i)
            stats = tuple(me[k] for k in ("mhp", "ad", "ar", "mr"))
            if stats != self.initial_stats[i]:
                raise RuntimeError(f"reset changed initial HUD stats: {stats} vs {self.initial_stats[i]}")
        self._rank()
        self._prepare(np.flatnonzero(done))

    def close(self):
        self.env.close()


def run_farming_learner(collector, policy, cfg, run, *, seed, rollout, updates,
                        save_updates=()):
    """Train either farming collector with exactly the same PPO/reward loop.

    Collectors provide n, episodes, observe(), step(actions), restart_done(),
    spell_ranks(), and close(). Observation/stat snapshots precede resets;
    spell ranks are used only for diagnostics, never as extra actor inputs.
    This function owns collector cleanup on success and failure.
    """
    rng = jax.random.key(seed)
    started = time.monotonic()
    print(run.path, flush=True)
    try:
        obs, stats = collector.observe()
        rng, init_key = jax.random.split(rng)
        params = policy.init(init_key, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec)
        tx, loss = make_learner(policy, cfg)
        opt_state = tx.init(params)
        @jax.jit
        def act(params, obs, key):
            logits = policy.apply(params, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec)
            action, lp, usage = _sample(logits, key, ~obs.entity_pad_mask)
            return action, lp, usage, logits.value
        @jax.jit
        def update(params, opt_state, batch, key):
            return kl_stopped_epochs(lambda q, b: loss(q, b, cfg), tx, params,
                opt_state, batch, key, epochs=cfg.epochs, n_minibatches=1,
                target_kl=cfg.target_kl, max_grad_norm=cfg.max_grad_norm)
        initial = jax.tree.map(lambda x: np.asarray(x).copy(), params)
        from flax.serialization import to_bytes
        (run.path / "initial.msgpack").write_bytes(to_bytes({"params": params}))
        for u in range(updates):
            rows, values, rewards, dones, reward_terms = [], [], [], [], []
            sampled_buttons = np.zeros(len(BUTTONS), dtype=np.int64)
            sampled_r_unranked = 0
            for t in range(rollout):
                rng, key = jax.random.split(rng)
                action, lp, usage, value = act(params, obs, key)
                host_actions = np.stack(jax.device_get(action), axis=-1)
                sampled_buttons += np.bincount(host_actions[:, 0], minlength=len(BUTTONS))
                sampled_r_unranked += int(np.sum(
                    (host_actions[:, 0] == BUTTONS.index("r"))
                    & (collector.spell_ranks()[:, 3] == 0)))
                done = collector.step(host_actions)
                next_obs, next_stats = collector.observe()
                reward, terms = farm_reward(stats[:, 0], next_stats[:, 0],
                    stats[:, 1].astype(bool), next_stats[:, 1].astype(bool),
                    stats[:, 2], next_stats[:, 2], done, cfg.gamma)
                rows.append({"entities": obs.entities, "mask": obs.entity_pad_mask,
                    "self": obs.self_vec, "global": obs.global_vec, "action": action,
                    "log_prob": lp, "uses_screen": usage[0], "uses_target": usage[1], "value": value})
                values.append(value); rewards.append(reward); dones.append(done)
                reward_terms.append(terms)
                for i in np.flatnonzero(done):
                    run.log({"episode": collector.episodes[i], "env": int(i),
                             "cs": float(next_stats[i, 0]), "update": u})
                if done.any():
                    collector.restart_done(done)
                    next_obs, next_stats = collector.observe()
                obs, stats = next_obs, next_stats
            rng, key = jax.random.split(rng)
            _, _, _, last_v = act(params, obs, key)
            adv, returns = gae(jnp.stack(rewards), jnp.stack(values), jnp.asarray(dones),
                               last_v, cfg.gamma, cfg.gae_lambda)
            batch = jax.tree.map(lambda *x: jnp.stack(x).reshape((-1,) + x[0].shape[1:]), *rows)
            batch["adv"] = adv.reshape(-1)
            if cfg.normalize_advantage:
                batch["adv"] = (batch["adv"] - adv.mean()) / (adv.std() + 1e-8)
            batch["returns"] = returns.reshape(-1)
            # Independent rollout/recompute equality before any optimizer step.
            logits = policy.apply(params, batch["entities"], batch["mask"], batch["self"], batch["global"])
            recomputed = factored_log_prob((logits.button, logits.screen_x, logits.screen_y),
                batch["action"], batch["uses_screen"], batch["uses_target"])
            np.testing.assert_allclose(recomputed, batch["log_prob"], atol=2e-5, rtol=2e-5)
            params, opt_state, rng, info = update(params, opt_state, batch, rng)
            metrics = {k: float(v) for k, v in summarise_minibatches(info).items()}
            if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(params)) or metrics.get("loss_nonfinite", 0):
                raise RuntimeError("nonfinite learner; refusing latest checkpoint")
            step = (u + 1) * collector.n * rollout
            metrics.update(update=u + 1, steps=step, wall_s=time.monotonic() - started,
                           mean_reward=float(jnp.stack(rewards).mean()), cs=stats[:, 0].tolist())
            # Log latent choices before screen_order collapses unavailable
            # spells/minimap clicks to NOOP. Diagnostic state is not an input.
            metrics["sampled_buttons"] = dict(zip(BUTTONS, sampled_buttons.tolist()))
            metrics["sampled_r_unranked"] = int(sampled_r_unranked)
            for name in reward_terms[0]:
                metrics["reward_" + name] = float(jnp.stack([r[name] for r in reward_terms]).mean())
            run.log(metrics)
            print(json.dumps(metrics), flush=True)
            if (u + 1) % 10 == 0 or u + 1 == updates or u + 1 in save_updates:
                saved = run.save(step, u + 1, {"params": params, "opt_state": opt_state, "step": step})
                if u + 1 in save_updates:
                    # RunDir rotates ordinary checkpoints. Explicit budget
                    # checkpoints must survive until paired evaluation.
                    import shutil
                    pinned = run.path / f"budget_{step:09d}.msgpack"
                    shutil.copyfile(saved, pinned)
                    run.set_results(**{f"budget_{step}": {
                        "file": pinned.name, "sha256": file_sha256(pinned),
                        "update": u + 1}})
        changed = any(not np.array_equal(a, b) for a, b in zip(jax.tree.leaves(initial), jax.tree.leaves(params)))
        if cfg.lr > 0 and not changed:
            raise RuntimeError("positive learning rate but no parameter changed")
        run.set_results(status="complete", parameters_changed=changed, wall_s=time.monotonic() - started)
    except BaseException as exc:
        run.set_results(status="failed", error=repr(exc))
        raise
    finally:
        collector.close()


def snapshot_farming_source(run, command):
    """Preserve the actual dirty/untracked source and launch command."""
    root = Path(__file__).resolve().parents[2]
    names = subprocess.check_output(["git", "ls-files", "-z", "--cached", "--others",
                                    "--exclude-standard"], cwd=root, env=git_environment(root)).decode().split("\0")
    with tarfile.open(run.path / "source.tar.gz", "w:gz") as tar:
        for name in sorted(set(names)):
            f = root / name
            if name and f.is_file() and f.suffix in (".py", ".sh", ".html", ".toml", ".patch", ".md", ".json"):
                tar.add(f, arcname=name)
    run.set_results(source_sha256=file_sha256(run.path / "source.tar.gz"), status="running")
    (run.path / "command.txt").write_text(command + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--envs", type=int, default=2)
    p.add_argument("--rollout", type=int, default=128)
    p.add_argument("--updates", type=int, default=2)
    p.add_argument("--save-updates", type=int, nargs="*", default=[],
                   help="Additional update numbers to checkpoint for matched-budget evaluation")
    p.add_argument("--episode-s", type=float, default=600.)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--start-near-wave", action="store_true")
    p.add_argument("--step-ticks", type=int, default=2)
    p.add_argument("--server-dir", type=Path)
    p.add_argument("--port-base", type=int, default=49300)
    p.add_argument("--out", type=Path, default=Path("lanerl_jax/runs/server_first_20260925"))
    args = p.parse_args()
    if args.server_dir is not None:
        args.server_dir = args.server_dir.resolve()
    if min(args.envs, args.rollout, args.updates, args.episode_s, args.step_ticks) <= 0:
        p.error("envs, rollout, updates, episode-s and step-ticks must be positive")
    if args.start_near_wave and args.episode_s <= WAVE_START_MS / 1000:
        p.error('wave-start episodes must end after 120 game seconds')
    cfg = PPOConfig(lr=args.lr, critic_lr=args.lr, decision_hz=60. / args.step_ticks,
                    gae_lambda=PPOConfig().gae_lambda ** (args.step_ticks / 2.))
    policy = LanePolicy(PolicyConfig())
    command = shlex.join([sys.executable, '-m', 'lanerl_jax.train.server_train', *sys.argv[1:]])
    run = RunDir(args.out, f"server-farm-s{args.seed}", {"train": {"policy": policy.cfg._asdict()},
        "command": command, "cwd": str(Path.cwd()),
        "ppo": cfg._asdict(), "collector": vars(args), "environment": "source-server",
        "observation": "viewport+server-fog+structured-HUD", "opponent": "idle-fountain",
        "reward": "CS - 2*death + 5*(gamma*terminal_zero_lane_potential_next - potential)",
        "initialization": "random; no prior"})
    snapshot_farming_source(run, command)
    vendor = server_paths.server_dir().parents[3]
    (run.path / "vendor.patch").write_bytes(subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"], cwd=vendor, env=git_environment(vendor)))
    untracked = subprocess.check_output(["git", "ls-files", "-z", "--others",
                                        "--exclude-standard"], cwd=vendor, env=git_environment(vendor)).decode().split("\0")
    with tarfile.open(run.path / "vendor-untracked.tar.gz", "w:gz") as tar:
        for name in untracked:
            if name and (vendor / name).is_file():
                tar.add(vendor / name, arcname=name)
    run.manifest["vendor"] = {
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=vendor, env=git_environment(vendor)).decode().strip(),
        "binary_sha256": file_sha256((args.server_dir or server_paths.server_dir()) / "GameServerLib.dll"),
        "config_sha256": file_sha256(server_paths.default_game_config())}
    run.write()
    try:
        collector = ServerCollector(args.envs, run.path, args.port_base, args.episode_s,
                                    args.start_near_wave, args.step_ticks, args.server_dir)
    except BaseException as exc:
        run.set_results(status='failed', error=repr(exc))
        raise
    run_farming_learner(collector, policy, cfg, run, seed=args.seed,
                        rollout=args.rollout, updates=args.updates,
                        save_updates=args.save_updates)


if __name__ == "__main__":
    main()
