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
from lanerl_jax.sim.state import Team
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


_SNAP = {}


def _snap_table_np():
    """Host copy of `actions.move_snap_table` (PATH-010): standable mask and
    nearest-standable-cell index per grid cell."""
    if not _SNAP:
        from .actions import move_snap_table
        t = move_snap_table()
        _SNAP.update(standable=np.asarray(t.standable), nearest=np.asarray(t.nearest),
                     height=t.height, width=t.width, min_x=float(t.min_x), min_y=float(t.min_y),
                     cell=float(t.cell_size))
    return _SNAP


def snap_click(x, y):
    """(x, y, snapped): the nearest standable cell centre if the click's cell is
    not standable (walls, off-map). 46-54% of a trained policy's movement clicks
    hit unwalkable ground (`runs/EVAL/replay_u1080`), and the server's click
    handler answers a null path with a straight line INTO the wall, so the
    champion hugged the map edge. Real League moves you to the closest reachable
    point; this does the same, with the table the JAX decoder snaps with."""
    t = _snap_table_np()
    ix = int(np.clip(np.floor((x - t["min_x"]) / t["cell"]), 0, t["width"] - 1))
    iy = int(np.clip(np.floor((y - t["min_y"]) / t["cell"]), 0, t["height"] - 1))
    flat = iy * t["width"] + ix
    in_grid = (t["min_x"] <= x < t["min_x"] + t["width"] * t["cell"]
               and t["min_y"] <= y < t["min_y"] + t["height"] * t["cell"])
    if in_grid and t["standable"][flat]:
        return float(x), float(y), False
    n = int(t["nearest"][flat])
    return (t["min_x"] + (n % t["width"] + 0.5) * t["cell"],
            t["min_y"] + (n // t["width"] + 0.5) * t["cell"], True)


SNAP_CLICKS = {"on": True, "count": 0, "total": 0}


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
    x = float(champion["x"] + ds * axis[0] + dn * normal[0])
    y = float(champion["y"] + ds * axis[1] + dn * normal[1])
    if SNAP_CLICKS["on"] and name in ("move", "attack_move"):
        x, y, snapped = snap_click(x, y)
        SNAP_CLICKS["total"] += 1
        SNAP_CLICKS["count"] += int(snapped)
    return {"t": "click", "button": name, "x": x, "y": y}


def source_farm_stats(champion, potential):
    """(cs, alive, potential, xp). Life state comes from the server flag, never
    its regenerating HP."""
    return champion["cs"], not champion_dead(champion), float(potential), float(champion.get("xp", 0.))


#: XP proximity reward per xp point. 0.005 (melee 77 xp -> 0.385) paid MORE per
#: step than the CS term in E01 (ratio 1.11 over updates 950-1150) and taught
#: both champions to camp the brush beside the wave; E04 runs it at 0.
XP_WEIGHT = 0.005


def farm_reward(cs_before, cs_after, alive_before, alive_after,
                potential_before, potential_after, done=None, gamma=None,
                xp_before=None, xp_after=None):
    """Explicit task reward, shared by the matched JAX collector.

    +1 per CS, -2 per death, plus lane-approach potential shaping
    ``5 * (Phi(s') - Phi(s))``. No opponent term, damage bonus or
    ambient-gold reward.

    The shaping is the UNDISCOUNTED potential difference, and the last step
    of an episode uses the real final potential (``potential_after`` is
    read before the reset). The earlier form ``gamma*Phi(s') - Phi(s)`` with
    a zero terminal potential paid ``(1-gamma)*|Phi|`` on every step spent
    standing far from the lane (+0.0033/step in the fountain at 10 Hz) plus
    ``|Phi|`` for ending the episode far away; the first lr-3e-4 run learned
    to recall and sit in the fountain within 40 updates (`REW-11`). With the
    difference form a stationary champion earns exactly zero, and only the
    endpoints of the walk are paid. ``done``/``gamma`` are accepted for
    call compatibility and unused.
    """
    cs = cs_after - cs_before
    death = -2. * (alive_before & ~alive_after)
    shaping = 5. * (potential_after - potential_before)
    # Experience is the dense half of farming: it is paid for standing near a
    # minion when it dies, killing blow or not, so it supplies the "be at the
    # wave while it dies" gradient the +1 CS term alone does not.
    xp = (XP_WEIGHT * (xp_after - xp_before)) if xp_after is not None else jnp.zeros_like(cs)
    return cs + death + shaping + xp, {"cs": cs, "death": death, "approach": shaping, "xp": xp}


WAVE_START_MS = 120_000
# Measured first minion contact in the untouched source-server control:
# 124.909 s, blue (2157,12474), red (2259,12573). Start behind blue's wave.
WAVE_START_POS = (1950., 12350.)
# Red's mirror of the same setup: the same distance (~240 u) behind ITS
# wave's contact point, along the blue->red contact direction. A single Move
# from red's fountain to that point stops at (12058,12979) -- the server's
# pathfinder gives up on that long route (measured, `runs/throughput_server_
# 20260925/redroute.log`) -- so red walks it in legs along the top lane.
RED_WAVE_START_POS = (2431., 12741.)
TEAM_WAVE_START = {0: (WAVE_START_POS,),
                   1: ((11000., 13600.), (7500., 13700.), (4500., 13600.), RED_WAVE_START_POS)}
TEAM_KEY = {0: 'blue', 1: 'red'}
TEAM_WIRE = {0: 100, 1: 200}


class WaveStart:
    """Fixed reset setup only; never contributes actions to the PPO batch.

    ``legs`` is the Move sequence; the next leg is issued once the champion
    is within ``leg_reach`` of the current one or has stopped moving.
    """
    def __init__(self, legs=(WAVE_START_POS,), tol=100., leg_reach=150.):
        self.legs = tuple(legs)
        self.pos, self.tol, self.leg_reach = self.legs[-1], tol, leg_reach
        self.leg = -1
        self.last = None
        self.still = 0

    def order(self, champion, t_ms):
        if champion_dead(champion) or champion['cs'] != 0:
            raise RuntimeError('wave-start setup died or farmed before policy control')
        if t_ms >= WAVE_START_MS:
            if np.hypot(champion['x']-self.pos[0], champion['y']-self.pos[1]) > self.tol:
                raise RuntimeError(f"wave-start setup missed {self.pos}: "
                                   f"position=({champion['x']}, {champion['y']}), hp={champion['hp']}")
            return None
        rank = pending_rank_up(champion)
        if rank is not None:
            return {'t': 'level', 'slot': rank}
        pos = (champion['x'], champion['y'])
        self.still = self.still + 1 if pos == self.last else 0
        self.last = pos
        if self.leg < 0:
            self.leg = 0
            return {'t': 'move', 'x': self.legs[0][0], 'y': self.legs[0][1]}
        if self.leg + 1 < len(self.legs):
            tx, ty = self.legs[self.leg]
            if np.hypot(pos[0]-tx, pos[1]-ty) <= self.leg_reach or self.still >= 10:
                self.leg += 1
                self.still = 0
                return {'t': 'move', 'x': self.legs[self.leg][0], 'y': self.legs[self.leg][1]}
        return {'t': 'noop'}


def _visibility_np(frame, netid, team):
    """Host copy of `wire_visibility` (same fail-closed rule, no device array)."""
    key = "vb" if int(team) == Team.BLUE else "vr"
    flags = {int(u["id"]): bool(u.get(key, False))
             for u in frame.get("u", []) if "id" in u}
    return np.asarray([flags.get(int(n), False) for n in netid], dtype=bool)


def _own_hud_np(frame, team):
    """Host copy of `wire_own_hud`: (ad, ap, ar, mr)/200, 4 slot-enable bits, dead."""
    me = next(u for u in frame["u"] if u.get("k") == "Champion" and u["tm"] == TEAM_WIRE[int(team)])
    dead = champion_dead(me)
    if len(me.get('se', [])) != 4:
        raise ValueError('source observation lacks own ability HUD enablement; use the HUD-capable server build')
    return np.asarray([me[k] / 200. for k in ("ad", "ap", "ar", "mr")] + list(me['se']) + [float(dead)], dtype=np.float32)


class ServerCollector:
    """``n`` server processes, ``teams`` policy-driven champions each.

    Agent rows are env-major then team: row ``i * len(teams) + k`` is env
    ``i``'s champion ``teams[k]``. ``teams=(0,)`` is blue farming against an
    idle red; ``teams=(0, 1)`` is mirror self-play, one policy on both sides.
    Every JAX call is batched over environments: one jitted, vmapped encode
    per team and one potential call per step, not one dispatch per env.
    """
    def __init__(self, n, out, port_base, episode_s, start_near_wave=False, step_ticks=2,
                 server_dir=None, teams=(0,)):
        self.n_envs, self.out, self.episode_s = n, Path(out), episode_s
        self.teams = tuple(int(t) for t in teams)
        self.T = len(self.teams)
        self.n = n * self.T
        self.frames = _lane_frames()
        # Four port sets per env, rotated on every fresh process: a restart
        # on the port the previous process just released hit "Address already
        # in use" (server exit 97) and killed a run at an episode reset.
        self._port_pool = PortAllocator(base=port_base).allocate(4 * n)
        self._port_turn = [0] * n
        self.env = VecLaneEnv(n, spec=ServerLaunchSpec(
            bot_teams="none", step_ticks=step_ticks, toponly=True,
            server_dir=server_dir,
            extra_env={"LANERL_AUTOBUY": "0"}),
            log_dir=self.out / "server", ports=[self._port_pool[4 * i] for i in range(n)],
            auto_restart=False)
        self.rebuilders = [StateRebuilder() for _ in range(n)]
        self.detectors = [CastFreezeDetector() for _ in range(n)]
        self.episodes = [0] * self.n
        self.initial_stats = [None] * self.n
        self.start_near_wave = start_near_wave
        params = self.rebuilders[0].params

        def encoder(team):
            frame = self.frames[team]
            def one(s, vis, hud):
                return apply_own_hud(build_observation(
                    s, team, frame, params=params, horizon_s=episode_s, visibility=vis), hud)
            return jax.jit(jax.vmap(one))
        self.encoders = {t: encoder(t) for t in self.teams}
        self.potential = jax.jit(jax.vmap(
            lambda s: -lane_corridor_distance(s.x[:2], s.y[:2]) / 10000.))
        try:
            self.env.start()
            for raw in self.env.last_obs:
                validate_champion_life(raw)
                for t in self.teams:
                    wire_own_hud(raw, t)  # fail before setup if the binary lacks required HUD fields
            self._rank()
            for i in range(n):
                for k, t in enumerate(self.teams):
                    self.initial_stats[i * self.T + k] = self._hud_stats(i, t)
            self._prepare()
        except BaseException:
            self.env.close()
            raise

    def _hud_stats(self, i, t):
        return tuple(self.champion(i, t)[k] for k in ('mhp', 'ad', 'ar', 'mr'))

    def _prepare(self, indices=None):
        if not self.start_near_wave:
            return
        indices = list(range(self.n_envs)) if indices is None else list(indices)
        setups = {i: {t: WaveStart(TEAM_WAVE_START[t], 100. if t == 0 else 250.)
                      for t in self.teams} for i in indices}
        count = 0
        # Only newly reset processes advance. Other environments remain paused.
        while setups:
            actions = [None] * self.n_envs
            active = []
            for i in list(setups):
                cmds = {}
                for t, setup in list(setups[i].items()):
                    cmd = setup.order(self.champion(i, t), self.env.last_obs[i]['t'])
                    if cmd is None:
                        del setups[i][t]
                    else:
                        cmds[TEAM_KEY[t]] = cmd
                if not setups[i]:
                    del setups[i]
                    continue
                for t in (0, 1):
                    cmds.setdefault(TEAM_KEY[t], {'t': 'noop'})
                actions[i] = cmds
                active.append(i)
                if count % 300 == 0 or any(c['t'] != 'noop' for c in cmds.values()):
                    with (self.out / 'setup.jsonl').open('a') as f:
                        f.write(json.dumps({'env': int(i), 't': self.env.last_obs[i]['t'],
                                            'champion': [self.champion(i, t) for t in self.teams],
                                            'command': cmds}) + '\n')
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
            pending = {}
            for i in range(self.n_envs):
                # A dead champion cannot rank (the server rejects live-only
                # input while dead); it ranks on respawn. Waiting on it here
                # spun the 18-try loop and killed the first mirror run.
                cmds = {TEAM_KEY[t]: {'t': 'level', 'slot': slot}
                        for t in self.teams
                        if not champion_dead(self.champion(i, t))
                        and (slot := pending_rank_up(self.champion(i, t))) is not None}
                if cmds:
                    for t in (0, 1):
                        cmds.setdefault(TEAM_KEY[t], {'t': 'noop'})
                    pending[i] = cmds
            if not pending:
                return
            # A level-up in one process must not silently advance its peers.
            for i, cmds in pending.items():
                self.env.handles[i].send_line(json.dumps(cmds))
            result = self.env._collect(list(pending))
            if result.died:
                raise RuntimeError(f'skill progression lost a server: {result.died}')
        # Not settled after 18 tries: record who is stuck and carry on; the
        # pending rank is retried on every later step. Raising here killed
        # the first two mirror runs (a dead champion, then an unknown case).
        with (self.out / 'rank_warnings.jsonl').open('a') as f:
            f.write(json.dumps({'t': [o['t'] for o in self.env.last_obs],
                'champions': [{k: self.champion(i, t).get(k) for k in ('tm', 'lvl', 'sl', 'dead', 'hp', 'xp', 'rc')}
                              for i in pending for t in self.teams],
                'pending': {int(i): c for i, c in pending.items()}}) + '\n')

    def champion(self, i, team=0):
        return next(u for u in self.env.last_obs[i]["u"]
                    if u.get("k") == "Champion" and u["tm"] == TEAM_WIRE[int(team)])

    def spell_ranks(self):
        """Own ranks per agent row for sampled-action diagnostics; not actor features."""
        return np.asarray([self.champion(i, t)["sl"] for i in range(self.n_envs)
                           for t in self.teams], dtype=np.int32)

    def observe(self):
        states, vis, huds = [], {t: [] for t in self.teams}, {t: [] for t in self.teams}
        for i, raw in enumerate(self.env.last_obs):
            self.detectors[i].observe(raw)
            if self.detectors[i].invalid:
                raise RuntimeError(self.detectors[i].reason())
            state, ids = self.rebuilders[i].rebuild(raw)
            if self.rebuilders[i].dropped_minions:
                raise RuntimeError("server observation exceeded entity capacity")
            states.append(state)
            for t in self.teams:
                vis[t].append(_visibility_np(raw, ids, t))
                # Own HUD is authoritative, not reconstructed simulator stats.
                huds[t].append(_own_hud_np(raw, t))
        # Stack on the host: the rebuilt leaves are numpy already and the
        # untouched base leaves are tiny device arrays (a jnp.stack per leaf
        # per step was 85 eager dispatches). Only the PRNG key leaf must stay
        # a device array.
        batched = jax.tree.map(
            lambda *a: jnp.stack(a) if jax.dtypes.issubdtype(a[0].dtype, jax.dtypes.prng_key)
            else np.stack([np.asarray(v) for v in a]), *states)
        pot = np.asarray(self.potential(batched))            # (n_envs, 2)
        per_team = [self.encoders[t](batched, np.stack(vis[t]), np.stack(huds[t]))
                    for t in self.teams]
        # Interleave to agent rows: env-major, then team.
        obs = jax.tree.map(
            lambda *x: jnp.stack(x, axis=1).reshape((self.n,) + x[0].shape[1:]), *per_team)
        stats = np.asarray([source_farm_stats(self.champion(i, t), pot[i, t])
                            for i in range(self.n_envs) for t in self.teams])
        return obs, stats

    def step(self, actions):
        lines = []
        for i in range(self.n_envs):
            cmds = {TEAM_KEY[t]: screen_order(actions[i * self.T + k], self.champion(i, t), self.frames[t])
                    for k, t in enumerate(self.teams)}
            for t in (0, 1):
                cmds.setdefault(TEAM_KEY[t], {"t": "noop"})
            lines.append(cmds)
        self.env.step(lines)
        self._rank()
        done = np.asarray([o["t"] >= self.episode_s * 1000 for o in self.env.last_obs])
        return np.repeat(done, self.T)

    def restart_done(self, done):
        envs = sorted({int(a) // self.T for a in np.flatnonzero(done)})
        for i in envs:
            # Fresh process preserves runes; the legacy in-process reset does not.
            self.env.handles[i].close()
            from lanerl_train.ports import is_port_free
            from lanerl_train.vec import InstanceDied
            last = None
            for attempt in range(4):
                self._port_turn[i] = (self._port_turn[i] + 1) % 4
                ports = self._port_pool[4 * i + self._port_turn[i]]
                if not (is_port_free(ports.control) and is_port_free(ports.game)):
                    last = f'ports {ports.control}/{ports.game} busy'
                    continue
                self.env.ports[i] = ports
                h = self.env._default_factory(int(i), ports)
                self.env.handles[i] = h
                try:
                    h.start()
                    result = self.env._collect([int(i)])
                    died = result.died
                except InstanceDied as exc:
                    died = {int(i): str(exc)}
                if not died:
                    break
                last = died
                with (self.out / 'restart_warnings.jsonl').open('a') as f:
                    f.write(json.dumps({'env': int(i), 'attempt': attempt, 'died': str(died)}) + '\n')
                self.env.alive[i] = True
                h.close()
            else:
                raise RuntimeError(f"fresh server failed after four attempts: {last}")
            self.rebuilders[i] = StateRebuilder()
            self.detectors[i] = CastFreezeDetector()
            validate_champion_life(self.env.last_obs[i])
            for k, t in enumerate(self.teams):
                a = i * self.T + k
                self.episodes[a] += 1
                stats = self._hud_stats(i, t)
                if stats != self.initial_stats[a]:
                    raise RuntimeError(f"reset changed initial HUD stats: {stats} vs {self.initial_stats[a]}")
        self._rank()
        self._prepare(envs)

    def close(self):
        self.env.close()


class MultiProcessCollector:
    """``workers`` processes, each owning ``n // workers`` servers and its own
    `ServerCollector`; the same host-facing API, agent rows concatenated in
    worker order (env-major then team is preserved).

    Why: one process stepping N servers in lockstep is bound by its own Python
    (observe + rebuild + JSON) at ~600 decisions/s regardless of N, while the
    servers idle. Workers parallelise that per-env work and the server ticks
    behind it; the learner process only batches the results. This is the
    pattern `lanerl_train/procactor.py` measured at 4,829 dec/s with 96
    servers on the desktop, without its staleness queue: steps stay
    synchronous, so every transition is on-policy exactly as before.
    """
    def __init__(self, n, out, port_base, episode_s, start_near_wave=False, step_ticks=2,
                 server_dir=None, teams=(0,), workers=2):
        import multiprocessing as mp
        from .server_worker import worker_main
        if n % workers:
            raise ValueError("envs must divide evenly across workers")
        self.teams, self.T = tuple(int(t) for t in teams), len(teams)
        self.n_envs, self.workers = n, workers
        self.n = n * self.T
        per = n // workers
        ctx = mp.get_context("spawn")
        self.conns, self.procs = [], []
        for w in range(workers):
            parent, child = ctx.Pipe()
            kwargs = dict(n=per, out=Path(out) / f"worker{w}", port_base=port_base + 200 * w,
                          episode_s=episode_s, start_near_wave=start_near_wave,
                          step_ticks=step_ticks, server_dir=server_dir, teams=self.teams)
            Path(kwargs["out"]).mkdir(parents=True, exist_ok=True)
            proc = ctx.Process(target=worker_main, args=(child, kwargs), daemon=True)
            proc.start()
            self.conns.append(parent); self.procs.append(proc)
        for c in self.conns:
            self._expect(c, "ready")
        self.episodes = [0] * self.n
        self._per_rows = per * self.T

    def _expect(self, conn, want="ok"):
        kind, payload = conn.recv()
        if kind == "error":
            raise RuntimeError("collector worker failed:\n" + payload)
        if kind != want:
            raise RuntimeError(f"collector worker sent {kind!r}, expected {want!r}")
        return payload

    def _all(self, cmd, args=None):
        for w, c in enumerate(self.conns):
            c.send((cmd, None if args is None else args[w]))
        return [self._expect(c) for c in self.conns]

    def observe(self):
        parts = self._all("observe")
        from ..obs.builder import Observation
        obs = Observation(*[jnp.concatenate([np.asarray(p[0][k]) for p in parts]) for k in range(5)])
        return obs, np.concatenate([p[1] for p in parts])

    def step(self, actions):
        actions = np.asarray(actions)
        chunks = [actions[w * self._per_rows:(w + 1) * self._per_rows] for w in range(self.workers)]
        return np.concatenate(self._all("step", chunks))

    def restart_done(self, done):
        done = np.asarray(done, bool)
        chunks = [done[w * self._per_rows:(w + 1) * self._per_rows] for w in range(self.workers)]
        eps = self._all("restart", chunks)
        self.episodes = [e for part in eps for e in part]

    def spell_ranks(self):
        return np.concatenate(self._all("ranks"))

    def close(self):
        for c in self.conns:
            try:
                c.send(("close", None)); c.recv()
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=30)
            if p.is_alive():
                p.kill()


def run_farming_learner(collector, policy, cfg, run, *, seed, rollout, updates,
                        save_updates=(), n_minibatches=1, resume=None, ckpt_every=10,
                        lr_anneal=False):
    """Train either farming collector with exactly the same PPO/reward loop.

    Collectors provide n, episodes, observe(), step(actions), restart_done(),
    spell_ranks(), and close(). Observation/stat snapshots precede resets;
    spell ranks are used only for diagnostics, never as extra actor inputs.
    This function owns collector cleanup on success and failure.
    """
    rng = jax.random.key(seed)
    started = time.monotonic()
    recurrent = getattr(getattr(policy, "cfg", None), "core", "mlp") == "gru"
    print(run.path, flush=True)
    try:
        obs, stats = collector.observe()
        rng, init_key = jax.random.split(rng)
        carry = policy.initial_carry((collector.n,)) if recurrent else None
        if recurrent:
            if collector.n % n_minibatches:
                raise ValueError("gru: minibatches must divide the number of agent rows (sequences)")
            params = policy.init(init_key, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec, carry)
        else:
            params = policy.init(init_key, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec)
        tx, loss = make_learner(policy, cfg, anneal_steps=(updates * cfg.epochs * n_minibatches) if lr_anneal else 0)
        opt_state = tx.init(params)
        start_update = 0
        if resume is not None:
            # Exact continuation of another run's latest checkpoint: params,
            # optimizer moments and the step counter; the RNG is folded with
            # the resumed update so the action stream does not restart.
            from flax.serialization import from_state_dict, msgpack_restore
            payload = msgpack_restore(Path(resume).read_bytes())
            params = from_state_dict(params, payload["params"])
            opt_state = from_state_dict(opt_state, payload["opt_state"])
            start_update = int(payload["step"]) // (collector.n * rollout)
            rng = jax.random.fold_in(rng, start_update)
            run.set_results(resumed_from=str(resume), resumed_update=start_update)
            print(f"resumed {resume} at update {start_update}", flush=True)
        @jax.jit
        def act(params, obs, key, carry=None):
            if recurrent:
                logits, carry = policy.apply(params, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec, carry)
            else:
                logits = policy.apply(params, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec)
            action, lp, usage = _sample(logits, key, ~obs.entity_pad_mask)
            return action, lp, usage, logits.value, carry
        @jax.jit
        def update(params, opt_state, batch, key):
            return kl_stopped_epochs(lambda q, b: loss(q, b, cfg), tx, params,
                opt_state, batch, key, epochs=cfg.epochs, n_minibatches=n_minibatches,
                target_kl=cfg.target_kl, max_grad_norm=cfg.max_grad_norm)
        initial = jax.tree.map(lambda x: np.asarray(x).copy(), params)
        from flax.serialization import to_bytes
        (run.path / "initial.msgpack").write_bytes(to_bytes({"params": params}))
        for u in range(start_update, updates):
            rows, values, rewards, dones, reward_terms = [], [], [], [], []
            sampled_buttons = np.zeros(len(BUTTONS), dtype=np.int64)
            sampled_r_unranked = 0
            carry0 = carry
            for t in range(rollout):
                rng, key = jax.random.split(rng)
                action, lp, usage, value, new_carry = act(params, obs, key, carry)
                host_actions = np.stack(jax.device_get(action), axis=-1)
                sampled_buttons += np.bincount(host_actions[:, 0], minlength=len(BUTTONS))
                sampled_r_unranked += int(np.sum(
                    (host_actions[:, 0] == BUTTONS.index("r"))
                    & (collector.spell_ranks()[:, 3] == 0)))
                done = collector.step(host_actions)
                next_obs, next_stats = collector.observe()
                xp = (stats[:, 3], next_stats[:, 3]) if stats.shape[1] > 3 else (None, None)
                reward, terms = farm_reward(stats[:, 0], next_stats[:, 0],
                    stats[:, 1].astype(bool), next_stats[:, 1].astype(bool),
                    stats[:, 2], next_stats[:, 2], done, cfg.gamma, *xp)
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
                if recurrent:
                    # The new episode starts from a zero state; the loss scan
                    # applies the same reset from the stored `done` flags.
                    carry = jnp.where(jnp.asarray(done)[:, None], 0.0, new_carry)
            rng, key = jax.random.split(rng)
            _, _, _, last_v, _ = act(params, obs, key, carry)
            adv, returns = gae(jnp.stack(rewards), jnp.stack(values), jnp.asarray(dones),
                               last_v, cfg.gamma, cfg.gae_lambda)
            if recurrent:
                # agent-major sequences [N, T, ...] so minibatches split agents
                batch = jax.tree.map(lambda *x: jnp.swapaxes(jnp.stack(x), 0, 1), *rows)
                batch["adv"] = jnp.swapaxes(adv, 0, 1)
                batch["returns"] = jnp.swapaxes(returns, 0, 1)
                batch["done"] = jnp.swapaxes(jnp.asarray(dones), 0, 1)
                batch["carry0"] = carry0
            else:
                batch = jax.tree.map(lambda *x: jnp.stack(x).reshape((-1,) + x[0].shape[1:]), *rows)
                batch["adv"] = adv.reshape(-1)
                batch["returns"] = returns.reshape(-1)
            if cfg.normalize_advantage:
                batch["adv"] = (batch["adv"] - adv.mean()) / (adv.std() + 1e-8)
            # Independent rollout/recompute equality before any optimizer step.
            logits = loss.forward(params, batch)
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
            metrics["snapped_clicks"] = SNAP_CLICKS["count"] / max(SNAP_CLICKS["total"], 1)
            SNAP_CLICKS["count"] = SNAP_CLICKS["total"] = 0
            for name in reward_terms[0]:
                metrics["reward_" + name] = float(jnp.stack([r[name] for r in reward_terms]).mean())
            run.log(metrics)
            print(json.dumps(metrics), flush=True)
            if (u + 1) % ckpt_every == 0 or u + 1 == updates or u + 1 in save_updates:
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


def evaluate_frozen(collector, policy, params, run, *, seed, episodes_per_env=1):
    """Frozen-policy episodes through the training collector itself.

    Same observation encoding, same sampling, same server binary as training;
    no learner. Each agent row's completed episodes are logged with CS and
    deaths. Returns the per-episode records.
    """
    rng = jax.random.key(seed)
    recurrent = getattr(policy.cfg, "core", "mlp") == "gru"
    @jax.jit
    def act(params, obs, key, carry=None):
        if recurrent:
            logits, carry = policy.apply(params, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec, carry)
        else:
            logits = policy.apply(params, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec)
        action, lp, usage = _sample(logits, key, ~obs.entity_pad_mask)
        return action, carry
    carry = policy.initial_carry((collector.n,)) if recurrent else None
    obs, stats = collector.observe()
    deaths = np.zeros(collector.n, np.int64)
    records = []
    done_count = np.zeros(collector.n, np.int64)
    steps = 0
    started = time.monotonic()
    try:
        while (done_count < episodes_per_env).any():
            rng, key = jax.random.split(rng)
            action, carry = act(params, obs, key, carry)
            host_actions = np.stack(jax.device_get(action), axis=-1)
            done = collector.step(host_actions)
            if recurrent:
                carry = jnp.where(jnp.asarray(done)[:, None], 0.0, carry)
            next_obs, next_stats = collector.observe()
            deaths += (stats[:, 1].astype(bool) & ~next_stats[:, 1].astype(bool))
            steps += 1
            for i in np.flatnonzero(done):
                rec = {"agent": int(i), "env": int(i) // collector.T,
                       "team": collector.teams[int(i) % collector.T],
                       "episode": collector.episodes[i], "cs": float(next_stats[i, 0]),
                       "deaths": int(deaths[i]), "wall_s": time.monotonic() - started}
                records.append(rec); run.log(rec); print(json.dumps(rec), flush=True)
                deaths[i] = 0
                done_count[i] += 1
            if done.any():
                collector.restart_done(done)
                next_obs, next_stats = collector.observe()
            obs, stats = next_obs, next_stats
        by_team = {}
        for r in records:
            by_team.setdefault(str(r["team"]), []).append(r["cs"])
        summary = {t: {"n": len(v), "mean_cs": float(np.mean(v)), "median_cs": float(np.median(v)),
                       "min_cs": float(np.min(v)), "max_cs": float(np.max(v))} for t, v in by_team.items()}
        run.set_results(status="complete", evaluation=summary, episodes=records,
                        decisions=int(steps * collector.n), wall_s=time.monotonic() - started)
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        run.set_results(status="failed", error=repr(exc))
        raise
    finally:
        collector.close()
    return records


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
    p.add_argument("--preset", choices=("legacy", "standard"), default="legacy",
                   help="standard: PPOConfig.standard() (CleanRL defaults, docs/HYPERPARAMS.md); "
                        "explicit flags below override the preset")
    p.add_argument("--core", choices=("mlp", "gru"), default="mlp",
                   help="gru: recurrent core, truncated BPTT over the rollout (memory is learned)")
    p.add_argument("--lr-anneal", action="store_true", help="linear lr decay to 0 over --updates")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--critic-lr", type=float, default=None, help="defaults to --lr")
    p.add_argument("--entropy-coef", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--minibatches", type=int, default=1)
    p.add_argument("--opponent", choices=("idle", "mirror"), default="idle",
                   help="idle: blue farms, red stays in fountain; mirror: one policy drives both champions")
    p.add_argument("--resume", type=Path, default=None, help="ckpt_latest.msgpack of a compatible run")
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--normalize-advantage", action="store_true",
                   help="per-batch advantage standardisation (the standard preset has it on)")
    p.add_argument("--xp-weight", type=float, default=XP_WEIGHT,
                   help="XP proximity reward per xp point (0 disables it)")
    p.add_argument("--no-snap-clicks", action="store_true",
                   help="send raw projected click points (pre-E04 behaviour: walls become straight-line walks)")
    p.add_argument("--workers", type=int, default=1,
                   help=">1: spawn that many collector processes, each owning envs/workers servers "
                        "(MultiProcessCollector); port blocks are base + 200*worker")
    p.add_argument("--eval-episodes", type=int, default=0,
                   help="evaluate the --resume checkpoint frozen for this many episodes per env; no learning")
    p.add_argument("--start-near-wave", action="store_true")
    p.add_argument("--step-ticks", type=int, default=2)
    p.add_argument("--server-dir", type=Path)
    p.add_argument("--port-base", type=int, default=21300,
                   help="control/game port base. Keep it BELOW 32768: ports inside the kernel's "
                        "ephemeral range (32768-60999) get taken by outgoing connections, and a "
                        "server restarted on such a port dies with exit 97 'Address already in use'")
    p.add_argument("--out", type=Path, default=Path("lanerl_jax/runs/server_first_20260925"))
    args = p.parse_args()
    if args.server_dir is not None:
        args.server_dir = args.server_dir.resolve()
    if min(args.envs, args.rollout, args.updates, args.episode_s, args.step_ticks) <= 0:
        p.error("envs, rollout, updates, episode-s and step-ticks must be positive")
    if args.start_near_wave and args.episode_s <= WAVE_START_MS / 1000:
        p.error('wave-start episodes must end after 120 game seconds')
    if (args.envs * (2 if args.opponent == "mirror" else 1) * args.rollout) % args.minibatches:
        p.error("minibatches must divide envs x agents x rollout")
    hz = 60. / args.step_ticks
    if args.preset == "standard":
        cfg = PPOConfig.standard(decision_hz=hz)
    else:
        cfg = PPOConfig(lr=1e-5, critic_lr=1e-5, decision_hz=hz,
                        gae_lambda=PPOConfig().gae_lambda ** (args.step_ticks / 2.),
                        normalize_advantage=args.normalize_advantage)
    over = {}
    if args.lr is not None: over["lr"] = args.lr; over["critic_lr"] = args.lr
    if args.critic_lr is not None: over["critic_lr"] = args.critic_lr
    if args.entropy_coef is not None: over["entropy_coef"] = args.entropy_coef
    if args.epochs is not None: over["epochs"] = args.epochs
    if args.normalize_advantage: over["normalize_advantage"] = True
    cfg = cfg._replace(**over)
    # The rollout/recompute likelihood check compares per-step and batched
    # forward passes; TF32 matmuls on the GPU would fail its 2e-5 tolerance.
    jax.config.update("jax_default_matmul_precision", "highest")
    globals()["XP_WEIGHT"] = args.xp_weight
    SNAP_CLICKS["on"] = not args.no_snap_clicks
    pcfg = PolicyConfig(core=args.core)
    if args.resume is not None and (args.resume.parent / "manifest.json").exists():
        # A checkpoint's own architecture wins over the flag: a gru checkpoint
        # loaded into an mlp policy would fail, an mlp one into a gru would be
        # a silently untrained core.
        saved = json.loads((args.resume.parent / "manifest.json").read_text()).get("config", {}).get("train", {}).get("policy", {})
        if saved:
            pcfg = PolicyConfig(**saved)
    policy = LanePolicy(pcfg)
    command = shlex.join([sys.executable, '-m', 'lanerl_jax.train.server_train', *sys.argv[1:]])
    run = RunDir(args.out, f"server-farm-s{args.seed}", {"train": {"policy": policy.cfg._asdict()},
        "command": command, "cwd": str(Path.cwd()),
        "ppo": cfg._asdict(), "collector": vars(args), "environment": "source-server",
        "observation": "viewport+server-fog+structured-HUD",
        "opponent": "idle-fountain" if args.opponent == "idle" else "mirror-self-play",
        "reward": f"CS - 2*death + 5*(lane_potential_next - potential) + {args.xp_weight}*xp",
        "click_snap": not args.no_snap_clicks,
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
        teams = (0,) if args.opponent == "idle" else (0, 1)
        if args.workers > 1:
            collector = MultiProcessCollector(args.envs, run.path, args.port_base, args.episode_s,
                                              args.start_near_wave, args.step_ticks, args.server_dir,
                                              teams=teams, workers=args.workers)
        else:
            collector = ServerCollector(args.envs, run.path, args.port_base, args.episode_s,
                                        args.start_near_wave, args.step_ticks, args.server_dir,
                                        teams=teams)
    except BaseException as exc:
        run.set_results(status='failed', error=repr(exc))
        raise
    if args.eval_episodes:
        from flax.serialization import from_state_dict, msgpack_restore
        obs0, _ = collector.observe()
        init_args = (obs0.entities, obs0.entity_pad_mask, obs0.self_vec, obs0.global_vec)
        if pcfg.core == "gru":
            init_args = init_args + (policy.initial_carry((collector.n,)),)
        params = policy.init(jax.random.key(args.seed), *init_args)
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
                        resume=args.resume, ckpt_every=args.ckpt_every, lr_anneal=args.lr_anneal)


if __name__ == "__main__":
    main()
