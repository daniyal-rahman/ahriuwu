"""Record one episode of a BC (or trainer) checkpoint in the modern world for rendering (``render.py``).

    python -m lanerl_jax.modern.bc.record CKPT [gate flags: --opponent afk|frozen|self --opponent-from ...
        --champions Garen,Jax --episode-s 600 --start-s 60 --seed 0] [--out episode.npz]

Same world and sampling as ``gate.py`` (one env, one decision per device call), but every decision's world
is kept: unit positions, health, life, visibility, both champions' level / gold / CS / cooldowns / ranks,
the sampled action and the order it decoded to (move point, attack target, cast slot and target), the
reward. Unit kind / team / tier are per decision (minion slots fill as waves spawn). Static tables
(structure positions, lane path, lane frames, the walkable mask) go in the same ``.npz`` so the renderer
needs no JAX.
"""
from __future__ import annotations

import datetime as dt
import math
import time

import jax
import jax.numpy as jnp
import numpy as np

from ..actions import MODERN_BUTTONS, modern_orders_from
from ..economy import level_for_xp
from ..rl.run_dir import file_sha256
from .gate import build_parser, setup


def main(argv=None) -> None:
    p = build_parser()
    p.set_defaults(envs=1, rollout=1, episodes=1)
    p.add_argument("--max-s", type=float, default=None, help="stop after this many game seconds (default: episode)")
    a = p.parse_args(argv)
    a.envs, a.rollout = 1, 1
    s = setup(a)
    env, cfg, built, runner, hz = s["env"], s["cfg"], s["built"], s["runner"], s["hz"]
    collect = jax.jit(built["collect"])
    wcfg = env.wcfg

    def extract(world):
        xp = world.econ.xp[:2]
        return dict(t=world.t, x=world.x, y=world.y, hp=world.hp, max_hp=world.max_hp, alive=world.alive,
                    visible=world.visible, radius=world.radius, kind=world.kind, team=world.team, sub=world.sub, cooldowns=world.champ.cooldowns[:2], ranks=world.champ.ranks[:2],
                    cs=world.champ.cs[:2], gold=world.econ.gold_total[:2], xp=xp,
                    level=level_for_xp(xp), dead=world.econ.dead[:2],
                    game_over=world.game_over)
    extract = jax.jit(extract)
    orders_of = jax.jit(lambda action, world: modern_orders_from(action, world, env.frames))
    env0 = lambda tree: jax.tree.map(lambda v: v[0], tree)           # noqa: E731

    deadline = a.max_s if a.max_s else a.episode_s
    max_steps = int(math.ceil((deadline - a.start_s) * hz)) + 2
    rows, acts, ords, rewards = [], [], [], []
    t1 = time.perf_counter()
    for step in range(max_steps):
        world = env0(runner.env_state.world)
        snap = jax.device_get(extract(world))
        runner, tr, _ = collect(runner)
        action = tuple(np.asarray(v)[0, 0] for v in tr.action)       # (2,) per head, this env
        order = jax.device_get(orders_of(tuple(jnp.asarray(v) for v in action), world))
        rows.append(snap)
        acts.append(np.stack(action, -1))                              # (2, heads)
        ords.append({k: np.asarray(v) for k, v in order._asdict().items()})
        rewards.append(np.asarray(tr.reward)[0, 0])
        if step % 600 == 0:
            print(f"t={float(snap['t']):.0f}s step {step} ({time.perf_counter() - t1:.0f}s)", flush=True)
        if bool(np.asarray(tr.done_full)[0, 0, 0]) or float(snap["t"]) >= deadline:
            break
    last = rows[-1]                      # the env resets on the last step; the final world is one decision earlier

    def stack(key, src=rows):
        return np.stack([np.asarray(r[key]) for r in src])
    terrain = wcfg.terrain[0]
    walk = np.asarray(terrain.walkable)
    if walk.ndim == 3:
        walk = walk[int(np.asarray(terrain.layer))] if terrain.layer is not None else walk[0]
    out = {f"u_{k}": stack(k) for k in ("x", "y", "hp", "max_hp", "alive", "visible", "kind", "team", "sub")}
    out.update({f"c_{k}": stack(k) for k in ("cooldowns", "ranks", "cs", "gold", "xp", "level", "dead")})
    out["t"] = stack("t")
    out["action"] = np.stack(acts)                                     # (T, 2, heads)
    out["reward"] = np.stack(rewards)                                  # (T, 2)
    for k in ords[0]:
        out[f"o_{k}"] = np.stack([o[k] for o in ords])                 # (T, 2)
    out.update(unit_kind=np.asarray(wcfg.unit_kind), unit_team=np.asarray(wcfg.unit_team),
               unit_sub=np.asarray(wcfg.unit_sub), unit_x=np.asarray(wcfg.unit_x), unit_y=np.asarray(wcfg.unit_y),
               radius=np.asarray(rows[0]["radius"]), lane_path=np.asarray(wcfg.lane_path),
               segment=np.asarray(env.segment), fountain=np.asarray(wcfg.fountain),
               walkable=walk.astype(np.uint8), terrain_min=np.asarray([terrain.min_x, terrain.min_z], np.float32),
               terrain_cell=np.asarray(float(terrain.cell_size), np.float32),
               frame_origin=np.stack([np.asarray(f.origin) for f in env.frames]),
               frame_axis=np.stack([np.asarray(f.axis) for f in env.frames]),
               frame_normal=np.stack([np.asarray(f.normal) for f in env.frames]),
               buttons=np.asarray(MODERN_BUTTONS), champions=np.asarray(s["names"]),
               decision_hz=np.asarray(hz, np.float32),
               meta=np.asarray(str(dict(checkpoint=str(a.checkpoint), sha256=file_sha256(a.checkpoint),
                                        opponent=a.opponent, opponent_from=str(s["opp_ckpt"]), seed=a.seed,
                                        episode_s=a.episode_s, start_s=a.start_s, fog=a.fog))))
    path = a.out or a.checkpoint.parent / f"episode-{a.opponent}-{dt.datetime.now(dt.UTC):%Y%m%d-%H%M%S}.npz"
    np.savez_compressed(path, **out)
    T = len(rows)
    print(f"{T} decisions ({float(rows[0]['t']):.0f}-{float(last['t']):.0f}s) in {time.perf_counter() - t1:.0f}s; "
          f"blue cs {int(last['cs'][0])} gold {float(last['gold'][0]):.0f} level {int(last['level'][0])} | "
          f"red cs {int(last['cs'][1])}; buttons "
          + " ".join(f"{MODERN_BUTTONS[b]} {c / T:.2f}" for b, c in
                     sorted(zip(*np.unique(out['action'][:, 0, 0], return_counts=True)), key=lambda bc: -bc[1])[:5])
          + f"\nwrote {path}", flush=True)


if __name__ == "__main__":
    main()
