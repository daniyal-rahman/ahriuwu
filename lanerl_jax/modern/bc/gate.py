"""In-sim gate for a behaviour-cloned policy: frozen rollouts in the modern world, no learning.

    python -m lanerl_jax.modern.bc.gate CKPT [--opponent afk|frozen|self] [--opponent-from CKPT] [--envs 64]
        [--episodes 1] [--episode-s 600] [--start-s 60] [--observation-lag 1] [--champions Garen,Jax] [--seed 0]

Builds the top-lane world the converter targets (lane 2, no jungle, no objectives, lane structures only), runs
every env through ``--episodes`` full episodes with the checkpoint's params on blue, sampling actions the way
the trainer does (same ``_env_step``), and writes ``gate-<opponent>-<utc>.json`` next to the checkpoint: one
row per episode (blue CS, gold, XP, deaths, kills, mean lane distance, button shares; red CS, gold, XP) and
the mean/median over episodes. ``--opponent self`` freezes the same checkpoint on red (use with
``--champions Garen,Garen``); ``frozen`` takes any trainer or BC checkpoint for red.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .. import train as MV
from .. import world as MS
from ..actions import MODERN_BUTTONS
from ..jax_cache import enable_compile_cache
from ..rl.policy import PolicyConfig, restore_policy_config
from ..rl.ppo import PPOConfig
from ..rl.run_dir import file_sha256


def _policy_config(ckpt: Path) -> PolicyConfig:
    manifest = ckpt.parent / "manifest.json"
    if manifest.exists():
        saved = json.loads(manifest.read_text()).get("config", {}).get("train", {}).get("policy", {})
        if saved:
            return restore_policy_config(saved)
    return PolicyConfig(core="gru")


def _load(ckpt: Path, template):
    from flax.serialization import from_state_dict, msgpack_restore
    return from_state_dict(template, msgpack_restore(ckpt.read_bytes())["params"])


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--opponent", choices=("afk", "frozen", "self"), default="afk")
    p.add_argument("--opponent-from", type=Path, default=None, help="checkpoint for --opponent frozen")
    p.add_argument("--envs", type=int, default=64)
    p.add_argument("--episodes", type=int, default=1, help="full episodes per env")
    p.add_argument("--rollout", type=int, default=128, help="decisions per device call")
    p.add_argument("--episode-s", type=float, default=600.0)
    p.add_argument("--start-s", type=float, default=60.0)
    p.add_argument("--step-ticks", type=int, default=3)
    p.add_argument("--observation-lag", type=int, default=1)
    p.add_argument("--observation-horizon-s", type=float, default=600.0,
                   help="clock normaliser of the observation (the converter used 600)")
    p.add_argument("--champions", default="Garen,Jax", help="BLUE,RED")
    p.add_argument("--fog", choices=("rays", "fast", "off"), default="rays")
    p.add_argument("--bank-size", type=int, default=16)
    p.add_argument("--buttons-off", default=",".join(MV.DEFAULT_BUTTONS_OFF))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None, help="JSON path (default: next to the checkpoint)")
    return p


def setup(a):
    """The world, config, jitted trainer pieces and initial runner for ``a`` (``build_parser`` namespace):
    the checkpoint's params on blue, the chosen opponent on red."""
    if a.opponent == "frozen" and a.opponent_from is None:
        sys.exit("--opponent frozen needs --opponent-from")
    enable_compile_cache()
    jax.config.update("jax_default_matmul_precision", "highest")
    t0 = time.perf_counter()
    names = tuple(a.champions.split(","))
    fog = False if a.fog == "off" else a.fog
    wcfg = MS.build_config(MV.default_loadouts(names), fog=fog, lanes=(2,), jungle=False, objectives=False,
                           lane_structures=True)
    env = MV.make_env(wcfg)
    hz = MV.TICK_HZ / a.step_ticks
    buttons_off = tuple(b for b in a.buttons_off.split(",") if b)
    pcfg = _policy_config(a.checkpoint)
    cfg = MV.ModernVecConfig(
        n_envs=a.envs, rollout_steps=a.rollout, n_updates=1, n_minibatches=1, episode_s=a.episode_s,
        start_s=a.start_s, step_ticks=a.step_ticks, observation_lag=a.observation_lag,
        observation_horizon_s=a.observation_horizon_s, stagger_initial=False, bank_size=a.bank_size,
        opponent="afk" if a.opponent == "afk" else "frozen", buttons_off=buttons_off,
        ppo=PPOConfig(decision_hz=hz), policy=pcfg)
    bank = MV.prepare_modern_bank(cfg, env, a.seed)
    template = MV.make_modern_vec_train(cfg._replace(opponent="mirror"), env, bank)["init_params"](
        jax.random.key(a.seed))
    params = _load(a.checkpoint, template)
    opp_ckpt = {"afk": None, "frozen": a.opponent_from, "self": a.checkpoint}[a.opponent]
    opponent_params = _load(opp_ckpt, template) if opp_ckpt else None
    built = MV.make_modern_vec_train(cfg, env, bank, opponent_params=opponent_params)
    runner = built["initial_runner"](jax.random.key(a.seed), params)
    print(f"world {names} built in {time.perf_counter() - t0:.0f}s; {a.envs} envs x {a.episodes} episodes of "
          f"{a.episode_s - a.start_s:.0f}s at {hz:.0f} Hz, opponent {a.opponent}", flush=True)
    return dict(env=env, cfg=cfg, built=built, runner=runner, names=names, opp_ckpt=opp_ckpt, hz=hz, pcfg=pcfg,
                t0=t0)


def main(argv=None) -> None:
    a = build_parser().parse_args(argv)
    s = setup(a)
    env, cfg, hz, names, pcfg, opp_ckpt, runner, t0 = (s[k] for k in ("env", "cfg", "hz", "names", "pcfg",
                                                                       "opp_ckpt", "runner", "t0"))
    collect = jax.jit(s["built"]["collect"])

    per_episode = int(math.ceil((a.episode_s - a.start_s) * hz))
    max_chunks = int(math.ceil(a.episodes * per_episode / a.rollout)) + 2
    n_b = len(MODERN_BUTTONS)
    acc = dict(deaths=np.zeros(a.envs), kills=np.zeros(a.envs), lane=np.zeros(a.envs), steps=np.zeros(a.envs),
               reward=np.zeros(a.envs), buttons=np.zeros((a.envs, n_b)), overflow=np.zeros(a.envs))
    rows: list[dict] = []
    finished = np.zeros(a.envs, int)
    t1 = time.perf_counter()
    for chunk in range(max_chunks):
        runner, tr, _ = collect(runner)
        small = jax.device_get({k: getattr(tr, k) for k in ("done_full", "cs", "gold", "xp", "deaths",
                                                              "lane_dist", "reward", "overflow")})
        button = np.asarray(tr.action[0][:, :, 0])                                   # (T, n) blue buttons
        T = button.shape[0]
        for t in range(T):
            acc["deaths"] += small["deaths"][t, :, 0]
            acc["kills"] += small["deaths"][t, :, 1]
            acc["lane"] += small["lane_dist"][t, :, 0]
            acc["reward"] += small["reward"][t, :, 0]
            acc["steps"] += 1
            acc["overflow"] = np.maximum(acc["overflow"], small["overflow"][t, :, 0])
            acc["buttons"][np.arange(a.envs), button[t]] += 1
            for e in np.flatnonzero(small["done_full"][t, :, 0]):
                if finished[e] >= a.episodes:
                    continue
                shares = acc["buttons"][e] / max(acc["steps"][e], 1)
                rows.append({
                    "env": int(e), "episode": int(finished[e]), "decisions": int(acc["steps"][e]),
                    "cs": float(small["cs"][t, e, 0]), "gold": float(small["gold"][t, e, 0]),
                    "xp": float(small["xp"][t, e, 0]), "deaths": float(acc["deaths"][e]),
                    "kills": float(acc["kills"][e]), "lane_dist": float(acc["lane"][e] / max(acc["steps"][e], 1)),
                    "reward": float(acc["reward"][e]), "overflow": float(acc["overflow"][e]),
                    "red_cs": float(small["cs"][t, e, 1]), "red_gold": float(small["gold"][t, e, 1]),
                    "red_xp": float(small["xp"][t, e, 1]),
                    "buttons": {b: round(float(shares[i]), 5) for i, b in enumerate(MODERN_BUTTONS) if shares[i]},
                })
                finished[e] += 1
                for k in acc:
                    acc[k][e] = 0
        done_n = int((finished >= a.episodes).sum())
        print(f"chunk {chunk + 1}/{max_chunks}: {len(rows)} episodes, {done_n}/{a.envs} envs finished, "
              f"{time.perf_counter() - t1:.0f}s", flush=True)
        if done_n == a.envs:
            break

    def stat(key):
        v = np.asarray([r[key] for r in rows], np.float64)
        return {"mean": float(v.mean()), "median": float(np.median(v)), "min": float(v.min()),
                "max": float(v.max())} if len(v) else {}
    keys = ("cs", "gold", "xp", "deaths", "kills", "lane_dist", "reward", "red_cs", "red_gold", "red_xp")
    summary = {k: stat(k) for k in keys}
    shares = {b: float(np.mean([r["buttons"].get(b, 0.0) for r in rows])) for b in MODERN_BUTTONS} if rows else {}
    summary["buttons"] = {b: round(v, 5) for b, v in shares.items() if v}
    summary["overflow_max"] = max((r["overflow"] for r in rows), default=0.0)
    out = a.out or a.checkpoint.parent / f"gate-{a.opponent}-{dt.datetime.now(dt.UTC):%Y%m%d-%H%M%S}.json"
    out.write_text(json.dumps({
        "checkpoint": str(a.checkpoint), "sha256": file_sha256(a.checkpoint),
        "opponent": a.opponent, "opponent_from": str(opp_ckpt) if opp_ckpt else None,
        "champions": names, "config": {k: v for k, v in cfg._asdict().items() if k not in ("ppo", "policy")},
        "policy": pcfg._asdict(), "episodes": len(rows), "summary": summary, "rows": rows,
        "wall_s": round(time.perf_counter() - t0, 1)}, indent=1))
    print(f"{len(rows)} episodes; blue cs {summary['cs'].get('mean', float('nan')):.1f} "
          f"(median {summary['cs'].get('median', float('nan')):.0f}) deaths {summary['deaths'].get('mean', 0):.2f} "
          f"kills {summary['kills'].get('mean', 0):.2f} lane {summary['lane_dist'].get('mean', 0):.0f} "
          f"| red cs {summary['red_cs'].get('mean', 0):.1f}; buttons "
          + " ".join(f"{b} {v:.3f}" for b, v in sorted(summary["buttons"].items(), key=lambda kv: -kv[1])[:6])
          + f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
