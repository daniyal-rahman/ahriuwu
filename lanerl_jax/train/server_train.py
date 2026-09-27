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
from .learner import make_learner, make_update
from .policy import LanePolicy, PolicyConfig
from .ppo import (PPOConfig, gae,
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
#: What a movement click onto unwalkable ground does. "resolve": the server (or
#: `snap_click`) walks to the closest reachable point -- which near a wall IS
#: the wall, so 43-58% of a diffuse policy's clicks pull it onto the map edge
#: and hold it there (INT-001, `probes/wall_click_probe.py`). "noop": the click
#: is dropped; the wall attracts nothing.
UNWALKABLE_CLICK = {"mode": "resolve", "dropped": 0, "total": 0}


class ScreenCellTable:
    """Lane-frame offsets of every screen cell centre, the minimap exclusion,
    and the standable grid: everything needed to ask "which screen cells are
    walkable from here?" (shared by `StepDiag` and `click_mask_host`)."""
    _inst = None

    @classmethod
    def get(cls):
        if cls._inst is None:
            cls._inst = cls()
        return cls._inst

    def __init__(self):
        self.t = _snap_table_np()
        ds = np.zeros((len(SCREEN_X_VALUES), len(SCREEN_Y_VALUES))); dn = ds.copy()
        for ix, sx in enumerate(SCREEN_X_VALUES):
            for iy, sy in enumerate(SCREEN_Y_VALUES):
                ds[ix, iy], dn[ix, iy] = screen_to_world_centred(0., 0., float(sx), float(sy))
        self.ds, self.dn = ds, dn
        self.minimap = (np.asarray(SCREEN_X_VALUES)[:, None] >= MINIMAP_X_MIN) & (np.asarray(SCREEN_Y_VALUES)[None, :] >= MINIMAP_Y_MIN)

    def standable(self, x, y):
        t = self.t
        ix = np.floor((x - t["min_x"]) / t["cell"]).astype(int); iy = np.floor((y - t["min_y"]) / t["cell"]).astype(int)
        inside = (ix >= 0) & (ix < t["width"]) & (iy >= 0) & (iy < t["height"])
        flat = np.clip(iy, 0, t["height"] - 1) * t["width"] + np.clip(ix, 0, t["width"] - 1)
        return inside & t["standable"][flat].astype(bool)

    def walkable_cells(self, x, y, frame):
        """(96, 54) bool: screen cells whose world point is standable and not on the minimap."""
        ax, nm = np.asarray(frame.axis), np.asarray(frame.normal)
        wx = x + self.ds * ax[0] + self.dn * nm[0]; wy = y + self.ds * ax[1] + self.dn * nm[1]
        return self.standable(wx.reshape(-1), wy.reshape(-1)).reshape(wx.shape) & ~self.minimap


def click_mask_host(collector):
    """(n, 96, 54) bool walkable-cell mask per agent row from the collector's
    positions and lane frames (PolicyConfig.click_mask). A dead or unknown
    position gets an all-True mask (no cell is impossible to name)."""
    tab = ScreenCellTable.get()
    pos = collector.positions(); n = pos.shape[0]
    out = np.ones((n, len(SCREEN_X_VALUES), len(SCREEN_Y_VALUES)), bool)
    for i in range(n):
        if np.isfinite(pos[i, 0]):
            m = tab.walkable_cells(pos[i, 0], pos[i, 1], collector.frames[collector.teams[i % collector.T]])
            if m.any():
                out[i] = m
    return out


class StepDiag:
    """Per-step, per-agent record of what the code COMPUTED, written by both
    the training rollout and frozen evaluation (`<run>/diag/steps_NNNNN.npz`).

    Columns: update, t, agent, env, team, x, y, wall_dist, alive, cs, xp,
    potential, r_cs, r_death, r_approach, r_xp (the reward function's own
    terms), value (critic), ent_button/ent_x/ent_y (policy entropies),
    p_unwalkable (click-distribution mass on unwalkable screen cells, from the
    same projection the click takes), button, sx, sy, act_unwalkable, done.

    Lesson of INT-001 / PPO-16 (2026-09-27): the wall and the KL questions
    took a day of post-hoc probes because no run logged what it saw; this
    costs ~200 rows/s and makes them a query. `--no-diag` turns it off.
    """
    COLS = ("update", "t", "agent", "env", "team", "x", "y", "wall_dist", "alive", "cs", "xp",
            "potential", "r_cs", "r_death", "r_approach", "r_xp", "value", "ent_button", "ent_x",
            "ent_y", "p_unwalkable", "button", "sx", "sy", "act_unwalkable", "done")

    @classmethod
    def create(cls, run_path, collector):
        """None for collectors without `positions()` (the JAX collector, for now)."""
        return cls(run_path, collector) if hasattr(collector, "positions") else None

    def __init__(self, run_path, collector, flush_rows=50000):
        self.dir = Path(run_path) / "diag"; self.dir.mkdir(parents=True, exist_ok=True)
        self.collector, self.flush_rows = collector, flush_rows
        self.rows, self.chunk = [], 0
        tab = ScreenCellTable.get(); t = tab.t
        standable = t["standable"].reshape(t["height"], t["width"]).astype(bool)
        try:
            from scipy import ndimage
            self.wall = ndimage.distance_transform_edt(standable) * t["cell"]
        except ImportError:
            self.wall = np.where(standable, np.nan, 0.0)
        self.t, self.ds, self.dn, self.minimap = t, tab.ds, tab.dn, tab.minimap

    def _standable_xy(self, x, y):
        t = self.t
        ix = np.floor((x - t["min_x"]) / t["cell"]).astype(int); iy = np.floor((y - t["min_y"]) / t["cell"]).astype(int)
        inside = (ix >= 0) & (ix < t["width"]) & (iy >= 0) & (iy < t["height"])
        flat = np.clip(iy, 0, t["height"] - 1) * t["width"] + np.clip(ix, 0, t["width"] - 1)
        return inside & t["standable"][flat].astype(bool), np.where(inside, self.wall.reshape(-1)[flat], 0.0)

    def add(self, update, t, stats, next_stats, terms, value, px, py, ent, actions, done):
        """stats/next_stats: (N, 4) cs, alive, potential, xp; terms: dict of (N,);
        value (N,); px (N, 96), py (N, 54) click marginals; ent (N, 3); actions (N, 3)."""
        pos = self.collector.positions()
        N = pos.shape[0]
        teams = np.asarray([self.collector.teams[i % self.collector.T] for i in range(N)])
        frames = self.collector.frames if hasattr(self.collector, "frames") else None
        stand, wall = self._standable_xy(pos[:, 0], pos[:, 1])
        p_unw = np.full(N, np.nan); act_unw = np.zeros(N, bool)
        for i in range(N):
            if not np.isfinite(pos[i, 0]) or frames is None:
                continue
            fr = frames[int(teams[i])]; ax, nm = np.asarray(fr.axis), np.asarray(fr.normal)
            wx = pos[i, 0] + self.ds * ax[0] + self.dn * nm[0]; wy = pos[i, 1] + self.ds * ax[1] + self.dn * nm[1]
            ok, _ = self._standable_xy(wx.reshape(-1), wy.reshape(-1))
            unw = ~ok.reshape(wx.shape) & ~self.minimap
            p_unw[i] = float((px[i][:, None] * py[i][None, :] * unw).sum()) if np.isfinite(px[i]).all() else np.nan
            b, sx, sy = actions[i]
            act_unw[i] = bool(unw[int(sx), int(sy)]) and BUTTONS[int(b)] in ("move", "attack_move")
        env = np.arange(N) // self.collector.T
        for i in range(N):
            self.rows.append((update, t, i, env[i], teams[i], pos[i, 0], pos[i, 1], wall[i], next_stats[i, 1], next_stats[i, 0],
                              next_stats[i, 3] if next_stats.shape[1] > 3 else np.nan, next_stats[i, 2],
                              float(terms["cs"][i]), float(terms["death"][i]), float(terms["approach"][i]), float(terms["xp"][i]),
                              float(value[i]), float(ent[i, 0]), float(ent[i, 1]), float(ent[i, 2]), p_unw[i],
                              int(actions[i, 0]), int(actions[i, 1]), int(actions[i, 2]), act_unw[i], bool(done[i])))
        if len(self.rows) >= self.flush_rows:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        arr = np.asarray(self.rows, dtype=np.float64)
        np.savez_compressed(self.dir / f"steps_{self.chunk:05d}.npz", cols=np.asarray(self.COLS), data=arr.astype(np.float32))
        self.chunk += 1; self.rows = []

    def close(self):
        self.flush()


def policy_click_marginals(logits):
    """Softmax marginals of the two click heads and the three head entropies."""
    px, py = jax.nn.softmax(logits.screen_x, -1), jax.nn.softmax(logits.screen_y, -1)
    def ent(l):
        lp = jax.nn.log_softmax(l, -1); return -(jnp.exp(lp) * lp).sum(-1)
    return px, py, jnp.stack([ent(logits.button), ent(logits.screen_x), ent(logits.screen_y)], -1)


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
    if name in ("move", "attack_move"):
        if UNWALKABLE_CLICK["mode"] == "noop":
            UNWALKABLE_CLICK["total"] += 1
            if snap_click(x, y)[2]:
                UNWALKABLE_CLICK["dropped"] += 1
                return {"t": "noop"}
        elif SNAP_CLICKS["on"]:
            x, y, snapped = snap_click(x, y)
            SNAP_CLICKS["total"] += 1
            SNAP_CLICKS["count"] += int(snapped)
    return {"t": "click", "button": name, "x": x, "y": y}


def source_farm_stats(champion, potential):
    """(cs, alive, potential, xp, gold). Life state comes from the server flag,
    never its regenerating HP. Gold is the wallet (`Stats.Gold`); the agent
    never buys, so it is earned gold plus the ambient trickle, and the trickle
    cancels in the RELATIVE reward."""
    return (champion["cs"], not champion_dead(champion), float(potential),
            float(champion.get("xp", 0.)), float(champion.get("gold", 0.)))


#: Relative reward (Dani, 2026-09-28): gold and XP are what decide fights, every
#: other stat derives from them, and a death is worth exactly its consequences
#: (the enemy's kill gold, your lost farm), not a hand-set -2. Scales: 20 gold
#: ~ one last-hit, so GOLD_SCALE 20 keeps "+1 per CS" magnitudes; XP_SCALE
#: 0.008 makes a shared minion's XP (~60) worth ~0.5.
#: `enemy_scale` weights the opponent's deltas: 1.0 = zero-sum in a mirror,
#: 0 = own gold/xp only. The XP scale is chosen so the two totals over a game are
#: roughly even: gold pays ~1.0 per own last-hit (20 g), XP pays ~0.48 per
#: enemy minion that dies nearby (60 xp), and nearby deaths are ~2x last-hits.
RELATIVE_REWARD = {"mode": "relative", "gold_scale": 20.0, "xp_scale": 0.008, "enemy_scale": 1.0}


def relative_reward(stats_before, stats_after, enemy_before, enemy_after, done=None):
    """(own gold gain - enemy gold gain)/gold_scale + xp_scale*(own xp gain -
    enemy xp gain) + lane-keep shaping, no death term. `enemy_*` rows are the
    opponent champion's stats aligned to each agent row (zeros when no
    opponent row exists: solo farming, where the terms reduce to own gold/xp).
    Terms keep the farm names so every consumer (metrics, StepDiag) reads them:
    cs -> the gold term, death -> 0, approach -> shaping, xp -> the xp term."""
    g = RELATIVE_REWARD["gold_scale"]; wx = RELATIVE_REWARD["xp_scale"]; es = RELATIVE_REWARD["enemy_scale"]
    d_gold = (stats_after[:, 4] - stats_before[:, 4]) - es * (enemy_after[:, 4] - enemy_before[:, 4])
    d_xp = (stats_after[:, 3] - stats_before[:, 3]) - es * (enemy_after[:, 3] - enemy_before[:, 3])
    shaping = 5. * (stats_after[:, 2] - stats_before[:, 2])
    gold = jnp.asarray(d_gold / g, jnp.float32); xp = jnp.asarray(wx * d_xp, jnp.float32); shaping = jnp.asarray(shaping, jnp.float32)
    zero = jnp.zeros_like(gold)
    return gold + xp + shaping, {"cs": gold, "death": zero, "approach": shaping, "xp": xp}


def enemy_rows(collector, stats):
    """Opponent champion's stats per agent row: the other team's row of the
    same env (mirror), or zeros when the collector has a single team."""
    if getattr(collector, "T", 1) == 2:
        n = stats.shape[0]; idx = np.arange(n) ^ 1
        return stats[idx]
    return np.zeros_like(stats)


def step_reward(collector, stats, next_stats, done, gamma):
    """The configured reward for one collector step (farm or relative)."""
    if RELATIVE_REWARD["mode"] == "relative":
        return relative_reward(stats, next_stats, enemy_rows(collector, stats), enemy_rows(collector, next_stats), done)
    xp = (stats[:, 3], next_stats[:, 3]) if stats.shape[1] > 3 else (None, None)
    return farm_reward(stats[:, 0], next_stats[:, 0], stats[:, 1].astype(bool), next_stats[:, 1].astype(bool),
                       stats[:, 2], next_stats[:, 2], done, gamma, *xp)


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
            for i, raw in enumerate(self.env.last_obs):
                if raw is None:
                    raise RuntimeError(f"server {i} produced no first observation (port {self.env.ports[i]}): "
                                       f"see {self.out}/server/instance{i:03d}.log; a port collision exits the server with 97")
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

    def positions(self):
        """(n, 2) world x, y per agent row (diagnostics only, never an actor input)."""
        return np.asarray([[self.champion(i, t)["x"], self.champion(i, t)["y"]] for i in range(self.n_envs)
                           for t in self.teams], dtype=np.float64)

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

    def positions(self):
        return np.concatenate(self._all("positions"))

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


class FrozenOpponentCollector:
    """Mirror servers where the learner drives ONE champion per server and a
    frozen checkpoint drives the other (a fixed sparring partner instead of
    live self-play, the remedy for self-play drift into duels). The learner's
    side alternates across servers (blue on even, red on odd) so neither side
    is favoured. Exposes the ServerCollector API with n = number of servers.
    """
    def __init__(self, inner, opp_policy, opp_params, *, seed=0):
        if inner.T != 2:
            raise ValueError("frozen opponent needs a mirror (two-team) collector")
        self.inner, self.n_envs, self.n, self.T, self.teams = inner, inner.n_envs, inner.n_envs, 1, (0,)
        self.side = np.arange(self.n_envs) % 2                 # learner's team per server
        self.rows = np.arange(self.n_envs) * 2 + self.side      # learner's agent rows in `inner`
        self.opp_rows = np.arange(self.n_envs) * 2 + (1 - self.side)
        self.opp_policy, self.opp_params = opp_policy, opp_params
        self.opp_recurrent = getattr(opp_policy.cfg, "core", "mlp") == "gru"
        self.opp_carry = opp_policy.initial_carry((self.n_envs,)) if self.opp_recurrent else None
        self.rng = jax.random.key(seed + 7919)
        @jax.jit
        def opp_act(params, obs, key, carry):
            if self.opp_recurrent:
                logits, carry = opp_policy.apply(params, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec, carry)
            else:
                logits = opp_policy.apply(params, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec)
            action, _, _ = _sample(logits, key, ~obs.entity_pad_mask)
            return action, carry
        self._opp_act = opp_act
        self._opp_obs = None
        self.episodes = [0] * self.n_envs

    def _split(self, obs, stats):
        take = lambda x: x[self.rows]
        self._opp_obs = jax.tree.map(lambda x: x[self.opp_rows], obs)
        return jax.tree.map(take, obs), stats[self.rows]

    def observe(self):
        return self._split(*self.inner.observe())

    def spell_ranks(self):
        return self.inner.spell_ranks()[self.rows]

    def positions(self):
        return self.inner.positions()[self.rows]

    def step(self, actions):
        self.rng, key = jax.random.split(self.rng)
        opp_action, carry = self._opp_act(self.opp_params, self._opp_obs, key, self.opp_carry)
        opp_host = np.stack(jax.device_get(opp_action), axis=-1)
        merged = np.zeros((self.inner.n, 3), np.int32)
        merged[self.rows] = np.asarray(actions)
        merged[self.opp_rows] = opp_host
        done = self.inner.step(merged)[self.rows]
        if self.opp_recurrent:
            self.opp_carry = jnp.where(jnp.asarray(done)[:, None], 0.0, carry)
        return done

    def restart_done(self, done):
        self.inner.restart_done(np.repeat(np.asarray(done, bool), 2))
        self.episodes = list(np.asarray(self.inner.episodes)[self.rows])

    def close(self):
        self.inner.close()


def load_checkpoint_policy(path):
    """(policy, params) for a checkpoint, architecture from its run manifest."""
    from flax.serialization import from_state_dict, msgpack_restore
    from ..obs.builder import build_observation
    from ..sim.init import init_lane, lane_params
    saved = json.loads((Path(path).parent / "manifest.json").read_text())["config"]["train"]["policy"]
    policy = LanePolicy(PolicyConfig(**saved))
    obs0 = build_observation(init_lane(), 0, _lane_frames()[0], params=lane_params())
    args = (obs0.entities, obs0.entity_pad_mask, obs0.self_vec, obs0.global_vec)
    if policy.cfg.core == "gru":
        args = args + (policy.initial_carry(()),)
    fresh = policy.init(jax.random.key(0), *args)
    params = from_state_dict(fresh, msgpack_restore(Path(path).read_bytes())["params"])
    return policy, params


def run_farming_learner(collector, policy, cfg, run, *, seed, rollout, updates,
                        save_updates=(), n_minibatches=1, resume=None, ckpt_every=10,
                        lr_anneal=False, resume_params_only=False, diag_steps=True):
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
    diag = None
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
        prior_params = None
        if cfg.kl_prior_coef > 0:
            if resume is None:
                raise ValueError("kl_prior_coef needs a prior: --init-from <checkpoint>")
            from flax.serialization import from_state_dict as _fsd, msgpack_restore as _mr
            prior_params = jax.tree.map(jnp.asarray, _fsd(params, _mr(Path(resume).read_bytes())["params"]))
            run.set_results(kl_prior=str(resume))
        cfg = cfg._replace(n_minibatches=n_minibatches)
        tx, loss = make_learner(policy, cfg, anneal_steps=(updates * cfg.epochs * n_minibatches) if lr_anneal else 0,
                                prior_params=prior_params)
        # Jitted ONCE. Calling loss.forward eagerly re-traced its scan closure
        # every update: a fresh compiled executable per update, ~10 MB each,
        # which is what walked E06 into slurm's memory limit (OOM at u1931).
        forward = jax.jit(loss.forward)
        trunk_norms = jax.jit(lambda q, b: loss.trunk_grad_norms(q, b, cfg))
        opt_state = tx.init(params)
        start_update = 0
        if resume is not None:
            # Exact continuation of another run's latest checkpoint: params,
            # optimizer moments and the step counter; the RNG is folded with
            # the resumed update so the action stream does not restart.
            from flax.serialization import from_state_dict, msgpack_restore
            payload = msgpack_restore(Path(resume).read_bytes())
            params = from_state_dict(params, payload["params"])
            if resume_params_only:
                # A NEW experiment seeded with another run's weights: fresh
                # optimizer, fresh lr schedule, its own update budget. (E07
                # inherited E06's counter and schedule and ran 767 updates at
                # a near-zero lr before hitting "update 6000".)
                run.set_results(initialised_from=str(resume))
                print(f"initialised params from {resume}; fresh optimizer and budget", flush=True)
            else:
                opt_state = from_state_dict(opt_state, payload["opt_state"])
                start_update = int(payload["step"]) // (collector.n * rollout)
                rng = jax.random.fold_in(rng, start_update)
                run.set_results(resumed_from=str(resume), resumed_update=start_update)
            print(f"resumed {resume} at update {start_update}", flush=True)
        use_mask = bool(getattr(policy.cfg, "click_mask", False))
        if use_mask and not hasattr(collector, "positions"):
            raise ValueError("click_mask needs a collector with positions()")
        @jax.jit
        def act(params, obs, key, carry=None, click_mask=None):
            if recurrent:
                logits, carry = policy.apply(params, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec, carry)
            else:
                logits = policy.apply(params, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec)
            action, lp, usage = _sample(logits, key, ~obs.entity_pad_mask, click_mask=click_mask)
            return action, lp, usage, logits.value, carry, policy_click_marginals(logits)
        diag = StepDiag.create(run.path, collector) if diag_steps else None
        update = make_update(tx, loss, cfg)
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
                click_mask = jnp.asarray(click_mask_host(collector)) if use_mask else None
                action, lp, usage, value, new_carry, (px, py, ent) = act(params, obs, key, carry, click_mask)
                host_actions = np.stack(jax.device_get(action), axis=-1)
                sampled_buttons += np.bincount(host_actions[:, 0], minlength=len(BUTTONS))
                sampled_r_unranked += int(np.sum(
                    (host_actions[:, 0] == BUTTONS.index("r"))
                    & (collector.spell_ranks()[:, 3] == 0)))
                done = collector.step(host_actions)
                next_obs, next_stats = collector.observe()
                reward, terms = step_reward(collector, stats, next_stats, done, cfg.gamma)
                rows.append({"entities": obs.entities, "mask": obs.entity_pad_mask,
                    "self": obs.self_vec, "global": obs.global_vec, "action": action,
                    "log_prob": lp, "uses_screen": usage[0], "uses_target": usage[1], "value": value,
                    **({"click_mask": click_mask} if use_mask else {})})
                if diag is not None:
                    diag.add(u, t, stats, next_stats, jax.device_get(terms), np.asarray(value),
                             np.asarray(px), np.asarray(py), np.asarray(ent), host_actions, np.asarray(done))
                values.append(value); rewards.append(reward); dones.append(done)
                reward_terms.append(terms)
                for i in np.flatnonzero(done):
                    run.log({"episode": int(collector.episodes[i]), "env": int(i),
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
            _, _, _, last_v, _, _ = act(params, obs, key, carry, jnp.asarray(click_mask_host(collector)) if use_mask else None)
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
            # Independent rollout/recompute equality before any optimizer step.
            logits = forward(params, batch)
            recomputed = factored_log_prob((logits.button, logits.screen_x, logits.screen_y),
                batch["action"], batch["uses_screen"], batch["uses_target"], click_mask=batch.get("click_mask"))
            # Head-mask mismatches are O(1) errors; a near-deterministic policy
            # (a BC clone) has log-probs of -20 and beyond whose float32
            # recomputation differs by ~1e-4, which failed the old 2e-5 gate
            # and killed E11 at update 2.
            np.testing.assert_allclose(recomputed, batch["log_prob"], atol=1e-3, rtol=1e-3,
                                       err_msg="rollout/recompute log-prob mismatch (head masks?)")
            # Which loss term steers the shared trunk (before this update's step).
            grad_diag = {k: float(v) for k, v in trunk_norms(params, batch).items()}
            # Explained variance of the critic on this batch (CleanRL's metric):
            # 1 = predicts the returns, 0 = no better than their mean, < 0 worse.
            _v = np.asarray(batch["value"]).reshape(-1); _r = np.asarray(batch["returns"]).reshape(-1)
            grad_diag["explained_variance"] = float(1.0 - np.var(_r - _v) / (np.var(_r) + 1e-8))
            params, opt_state, rng, info = update(params, opt_state, batch, rng)
            metrics = {k: float(v) for k, v in info.items()}
            metrics.update(grad_diag)
            # KL(rollout policy || updated policy) on the whole batch AFTER the
            # update. approx_kl averages pre-step minibatch measurements;
            # post_kl measures the final policy's drift over the rollout.
            logits = forward(params, batch)
            post_lp = factored_log_prob((logits.button, logits.screen_x, logits.screen_y),
                batch["action"], batch["uses_screen"], batch["uses_target"], click_mask=batch.get("click_mask"))
            metrics["post_kl"] = float(jnp.mean(batch["log_prob"] - post_lp))
            if not all(np.isfinite(np.asarray(v)).all() for v in jax.tree.leaves(params)) or metrics.get("loss_nonfinite", 0):
                raise RuntimeError("nonfinite learner; refusing latest checkpoint")
            step = (u + 1) * collector.n * rollout
            metrics.update(update=u + 1, steps=step, wall_s=time.monotonic() - started,
                           mean_reward=float(jnp.stack(rewards).mean()), cs=stats[:, 0].tolist())
            # Log latent choices before screen_order collapses unavailable
            # spells/minimap clicks to NOOP. Diagnostic state is not an input.
            metrics["sampled_buttons"] = dict(zip(BUTTONS, sampled_buttons.tolist()))
            metrics["sampled_r_unranked"] = int(sampled_r_unranked)
            # OOM watch (E06 died at slurm's 10 GB): CURRENT resident set, not
            # the monotonic peak ru_maxrss reports.
            try:
                with open("/proc/self/status") as f:
                    metrics["rss_gb"] = next(int(l.split()[1]) for l in f if l.startswith("VmRSS")) / 1e6
            except (OSError, StopIteration):
                metrics["rss_gb"] = -1.0
            metrics["snapped_clicks"] = SNAP_CLICKS["count"] / max(SNAP_CLICKS["total"], 1)
            SNAP_CLICKS["count"] = SNAP_CLICKS["total"] = 0
            metrics["dropped_clicks"] = UNWALKABLE_CLICK["dropped"] / max(UNWALKABLE_CLICK["total"], 1)
            UNWALKABLE_CLICK["dropped"] = UNWALKABLE_CLICK["total"] = 0
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
        if diag is not None:
            diag.close()
        collector.close()


def evaluate_frozen(collector, policy, params, run, *, seed, episodes_per_env=1, act_fn=None,
                    record_npz=None, deterministic=False, diag_steps=True):
    """Frozen-policy episodes through the training collector itself.

    Same observation encoding, same sampling, same server binary as training;
    no learner. Each agent row's completed episodes are logged with CS and
    deaths. Returns the per-episode records.
    """
    rng = jax.random.key(seed)
    recurrent = act_fn is None and getattr(policy.cfg, "core", "mlp") == "gru"
    use_mask = act_fn is None and bool(getattr(policy.cfg, "click_mask", False))
    @jax.jit
    def act(params, obs, key, carry=None, click_mask=None):
        if act_fn is not None:
            # A scripted player through the SAME observation/click interface
            # (`train/scripted_policy.py`): the interface oracle.
            a = jax.vmap(act_fn, in_axes=(0, None))(obs, key)
            n = obs.entities.shape[0]
            nan = jnp.full((n,), jnp.nan)
            return a, carry, (nan, jnp.full((n, len(SCREEN_X_VALUES)), jnp.nan), jnp.full((n, len(SCREEN_Y_VALUES)), jnp.nan), jnp.full((n, 3), jnp.nan))
        if recurrent:
            logits, carry = policy.apply(params, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec, carry)
        else:
            logits = policy.apply(params, obs.entities, obs.entity_pad_mask, obs.self_vec, obs.global_vec)
        if deterministic:
            # DIAGNOSTIC only (argmax clicks): the ledger records deterministic
            # evaluations misleading us before; never a gate number.
            action = (jnp.argmax(logits.button, -1), jnp.argmax(logits.screen_x, -1), jnp.argmax(logits.screen_y, -1))
        else:
            action, lp, usage = _sample(logits, key, ~obs.entity_pad_mask, click_mask=click_mask)
        return action, carry, (logits.value, *policy_click_marginals(logits))
    carry = policy.initial_carry((collector.n,)) if recurrent else None
    diag = StepDiag.create(run.path, collector) if diag_steps else None   # scripted runs too (parity timelines)
    obs, stats = collector.observe()
    deaths = np.zeros(collector.n, np.int64)
    records = []
    done_count = np.zeros(collector.n, np.int64)
    steps = 0
    started = time.monotonic()
    traj = {"entities": [], "mask": [], "self": [], "global": [], "action": [], "done": []} if record_npz else None
    try:
        while (done_count < episodes_per_env).any():
            rng, key = jax.random.split(rng)
            action, carry, extra = act(params, obs, key, carry, jnp.asarray(click_mask_host(collector)) if use_mask else None)
            host_actions = np.stack(jax.device_get(action), axis=-1)
            if traj is not None:
                # (obs, action) pairs per agent row, for behaviour-cloning diagnostics
                traj["entities"].append(np.asarray(obs.entities, np.float16)); traj["mask"].append(np.asarray(obs.entity_pad_mask))
                traj["self"].append(np.asarray(obs.self_vec, np.float32)); traj["global"].append(np.asarray(obs.global_vec, np.float32))
                traj["action"].append(host_actions.astype(np.int16))
            done = collector.step(host_actions)
            if traj is not None:
                traj["done"].append(np.asarray(done))
            if recurrent:
                carry = jnp.where(jnp.asarray(done)[:, None], 0.0, carry)
            next_obs, next_stats = collector.observe()
            deaths += (stats[:, 1].astype(bool) & ~next_stats[:, 1].astype(bool))
            if diag is not None:
                _, terms = step_reward(collector, stats, next_stats, done, None)
                value, px, py, ent = (np.asarray(v) for v in jax.device_get(extra))
                diag.add(0, steps, stats, next_stats, jax.device_get(terms), value, px, py, ent, host_actions, np.asarray(done))
            steps += 1
            for i in np.flatnonzero(done):
                rec = {"agent": int(i), "env": int(i) // collector.T,
                       "team": int(collector.teams[int(i) % collector.T]),
                       "episode": int(collector.episodes[i]), "cs": float(next_stats[i, 0]),
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
        if traj is not None:
            # time-major [T, N, ...]
            np.savez_compressed(record_npz, **{k: np.stack(v) for k, v in traj.items()})
            print("recorded", record_npz, {k: np.stack(v).shape for k, v in traj.items()}, flush=True)
        print(json.dumps(summary), flush=True)
    except BaseException as exc:
        run.set_results(status="failed", error=repr(exc))
        raise
    finally:
        if diag is not None:
            diag.close()
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
                   help="standard: PPOConfig.standard() (PureJaxRL defaults, docs/HYPERPARAMS.md); "
                        "explicit flags below override the preset")
    p.add_argument("--core", choices=("mlp", "gru"), default="mlp",
                   help="gru: recurrent core, truncated BPTT over the rollout (memory is learned)")
    p.add_argument("--core-norm", action="store_true", help="gru: LayerNorm on the GRU input")
    p.add_argument("--core-residual", action="store_true", help="gru: heads see trunk + GRU output (plain GRU heads are near-blind, ARCH-001)")
    p.add_argument("--lr-anneal", action="store_true", help="linear lr decay to 0 over --updates")
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--critic-lr", type=float, default=None, help="compatibility argument; must equal --lr (one reference Adam)")
    p.add_argument("--entropy-coef", type=float, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--target-kl", type=float, default=None, help="deprecated compatibility argument; reference PPO does not stop on KL")
    p.add_argument("--kl-prior", type=float, default=None, help="coefficient of KL(prior || policy) to the --init-from checkpoint (the league / AlphaStar anchor)")
    p.add_argument("--minibatches", type=int, default=1)
    p.add_argument("--opponent", choices=("idle", "mirror", "frozen"), default="idle",
                   help="idle: blue farms, red stays in fountain; mirror: one policy drives both champions; "
                        "frozen: the learner drives one side (alternating per server), --opponent-ckpt drives the other")
    p.add_argument("--opponent-ckpt", type=Path, default=None, help="checkpoint for --opponent frozen")
    p.add_argument("--team", type=int, choices=(0, 1), default=0,
                   help="idle mode: which side the learner/scripted player takes (1 = red, blue idles)")
    p.add_argument("--resume", type=Path, default=None, help="ckpt_latest.msgpack of a compatible run")
    p.add_argument("--init-from", type=Path, default=None,
                   help="take only the PARAMS from this checkpoint (fresh optimizer, schedule and budget)")
    p.add_argument("--ckpt-every", type=int, default=10)
    p.add_argument("--normalize-advantage", action="store_true",
                   help="per-minibatch advantage standardisation (the standard preset has it on)")
    p.add_argument("--xp-weight", type=float, default=XP_WEIGHT,
                   help="XP proximity reward per xp point (0 disables it)")
    p.add_argument("--reward", choices=("farm", "relative"), default="relative",
                   help="relative (DEFAULT from 2026-09-28): (own - enemy_scale * enemy) gold and xp deltas + shaping, no death term; farm: the pre-E23 +1 CS, -2 death, shaping, xp")
    p.add_argument("--gold-scale", type=float, default=20.0); p.add_argument("--xp-scale", type=float, default=0.008)
    p.add_argument("--enemy-scale", type=float, default=1.0, help="weight of the opponent's gold/xp deltas (1 = zero-sum mirror, 0 = own only)")
    p.add_argument("--no-diag", action="store_true", help="skip the per-step diagnostic record (<run>/diag/)")
    p.add_argument("--detach-critic", action="store_true", help="stop the critic's gradient at the shared trunk (PolicyConfig.detach_critic)")
    p.add_argument("--click-mask", action="store_true", help="sample clicks from the masked joint distribution over walkable cells (INT-001 principled fix; PolicyConfig.click_mask)")
    p.add_argument("--unwalkable-click", choices=("resolve", "noop"), default="resolve",
                   help="movement click onto unwalkable ground: resolve to the closest reachable point (server/snap) or drop it (INT-001)")
    p.add_argument("--no-snap-clicks", action="store_true",
                   help="send raw projected click points (pre-E04 behaviour: walls become straight-line walks)")
    p.add_argument("--workers", type=int, default=1,
                   help=">1: spawn that many collector processes, each owning envs/workers servers "
                        "(MultiProcessCollector); port blocks are base + 200*worker")
    p.add_argument("--eval-episodes", type=int, default=0,
                   help="evaluate the --resume checkpoint frozen for this many episodes per env; no learning")
    p.add_argument("--deterministic", action="store_true", help="eval only: argmax actions (diagnostic, never a gate number)")
    p.add_argument("--record-npz", type=Path, default=None,
                   help="eval only: save (observation, action, done) per step, time-major, for BC diagnostics")
    p.add_argument("--scripted", choices=("lasthit", "any"), default=None,
                   help="eval only: drive the learner's rows with the scripted last-hitter (interface oracle) instead of a checkpoint")
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
    if args.opponent == "frozen" and (args.opponent_ckpt is None or not args.opponent_ckpt.exists()):
        p.error("--opponent frozen needs an existing --opponent-ckpt")
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
    if args.target_kl is not None:
        print("--target-kl is ignored: reference PPO applies every minibatch", flush=True)
    if args.kl_prior is not None: over["kl_prior_coef"] = args.kl_prior
    if args.normalize_advantage: over["normalize_advantage"] = True
    cfg = cfg._replace(**over)
    # The rollout/recompute likelihood check compares per-step and batched
    # forward passes; TF32 matmuls on the GPU would fail its 2e-5 tolerance.
    jax.config.update("jax_default_matmul_precision", "highest")
    globals()["XP_WEIGHT"] = args.xp_weight
    SNAP_CLICKS["on"] = not args.no_snap_clicks
    UNWALKABLE_CLICK["mode"] = args.unwalkable_click
    RELATIVE_REWARD.update(mode=args.reward, gold_scale=args.gold_scale, xp_scale=args.xp_scale, enemy_scale=args.enemy_scale)
    pcfg = PolicyConfig(core=args.core, core_norm=args.core_norm, core_residual=args.core_residual, click_mask=args.click_mask,
                        detach_critic=args.detach_critic)
    if args.init_from is not None:
        if args.resume is not None:
            p.error("--init-from and --resume are exclusive")
        args.resume, resume_params_only = args.init_from, True
    else:
        resume_params_only = False
    if args.resume is not None and (args.resume.parent / "manifest.json").exists():
        # A checkpoint's own architecture wins over the flag: a gru checkpoint
        # loaded into an mlp policy would fail, an mlp one into a gru would be
        # a silently untrained core.
        saved = json.loads((args.resume.parent / "manifest.json").read_text()).get("config", {}).get("train", {}).get("policy", {})
        if saved:
            # Architecture from the checkpoint; the INTERFACE/training switches
            # (no parameters) stay under the flags, so a fine-tune can turn
            # them on: E21 --detach-critic from E12a, E20-style --click-mask.
            pcfg = PolicyConfig(**{**saved, "click_mask": args.click_mask or saved.get("click_mask", False),
                                   "detach_critic": args.detach_critic or saved.get("detach_critic", False)})
    policy = LanePolicy(pcfg)
    command = shlex.join([sys.executable, '-m', 'lanerl_jax.train.server_train', *sys.argv[1:]])
    run = RunDir(args.out, f"server-farm-s{args.seed}", {"train": {"policy": policy.cfg._asdict()},
        "command": command, "cwd": str(Path.cwd()),
        "ppo": cfg._asdict(), "collector": vars(args), "environment": "source-server",
        "observation": "viewport+server-fog+structured-HUD",
        "opponent": {"idle": "idle-fountain", "mirror": "mirror-self-play",
                     "frozen": f"frozen checkpoint {args.opponent_ckpt} (learner side alternates per server)"}[args.opponent],
        "reward": f"CS - 2*death + 5*(lane_potential_next - potential) + {args.xp_weight}*xp",
        "click_snap": not args.no_snap_clicks,
        "unwalkable_click": args.unwalkable_click,
        "reward": args.reward, "gold_scale": args.gold_scale, "xp_scale": args.xp_scale, "enemy_scale": args.enemy_scale,
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
        teams = (args.team,) if args.opponent == "idle" else (0, 1)
        if args.opponent == "frozen":
            opp_policy, opp_params = load_checkpoint_policy(args.opponent_ckpt)
            run.set_results(opponent_ckpt_sha256=file_sha256(args.opponent_ckpt))
        if args.workers > 1:
            collector = MultiProcessCollector(args.envs, run.path, args.port_base, args.episode_s,
                                              args.start_near_wave, args.step_ticks, args.server_dir,
                                              teams=teams, workers=args.workers)
        else:
            collector = ServerCollector(args.envs, run.path, args.port_base, args.episode_s,
                                        args.start_near_wave, args.step_ticks, args.server_dir,
                                        teams=teams)
        if args.opponent == "frozen":
            collector = FrozenOpponentCollector(collector, opp_policy, opp_params, seed=args.seed)
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
        act_fn = None
        if args.scripted:
            from .scripted_policy import scripted_act, scripted_act_any
            act_fn = scripted_act if args.scripted == "lasthit" else scripted_act_any
        run.set_results(evaluated_checkpoint=(f"scripted:{args.scripted}" if args.scripted else
                                              str(args.resume) if args.resume else "random"),
                        checkpoint_sha256=file_sha256(args.resume) if args.resume else None)
        evaluate_frozen(collector, policy, params, run, seed=args.seed,
                        episodes_per_env=args.eval_episodes, act_fn=act_fn,
                        record_npz=(run.path / args.record_npz.name) if args.record_npz else None,
                        deterministic=args.deterministic, diag_steps=not args.no_diag)
        return
    run_farming_learner(collector, policy, cfg, run, seed=args.seed,
                        rollout=args.rollout, updates=args.updates,
                        save_updates=args.save_updates, n_minibatches=args.minibatches,
                        resume=args.resume, ckpt_every=args.ckpt_every, lr_anneal=args.lr_anneal,
                        resume_params_only=resume_params_only, diag_steps=not args.no_diag)


if __name__ == "__main__":
    main()
