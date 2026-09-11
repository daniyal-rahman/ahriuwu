#!/usr/bin/env python
"""Evaluate a policy against the FIXED scripted bot.

Why this exists: in symmetric self-play `score` is 0.5 by construction, so the
training run cannot tell you whether the policy improved. The only honest
progress metric is performance against an opponent that does not move --
here, the frozen scripted Garen.

Setup: the control channel drives BLUE only (we send no "red" key, so
ApplyActions leaves red alone), and LANERL_BOT=purple hands red to the bot.

Reports CS@10, gold, level, deaths, damage taken, and how much of the map the
champion actually used -- a policy that never leaves the fountain and one that
farms both score 0 deaths, and only the movement stats tell them apart.

  python -m lanerl_train.eval_vs_bot --checkpoint <path|random> --episodes 4
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

_REPO = Path(__file__).resolve().parents[1]
_PROJECTS = _REPO.parent
VENDOR = _PROJECTS / "lanerl-vendor"
BIN = VENDOR / "LoLServer/GameServerConsole/bin/Release/net6.0"
CFG = _REPO / "lanerl/cfg/garen1v1.json"
sys.path.insert(0, str(_REPO))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def play_episode(policy, max_game_ms: int, step_ticks: int, log_path: Path) -> Dict:
    """One game: policy drives blue, scripted bot drives red."""
    cport, gport = free_port(), free_port()
    env = dict(os.environ)
    env.update(
        DOTNET_ROOT=str(VENDOR / "dotnet"),
        LANERL_HEADLESS="1", LANERL_FREERUN="1",
        LANERL_BOT="purple",            # red = frozen scripted bot
        LANERL_CONTROL_PORT=str(cport),
        LANERL_STEP_TICKS=str(step_ticks),
    )
    proc = subprocess.Popen(
        [str(BIN / "GameServerConsole"), "--config", str(CFG), "--port", str(gport)],
        cwd=str(BIN), env=env, stdout=log_path.open("w"), stderr=subprocess.STDOUT,
    )
    try:
        sock = None
        for _ in range(120):
            try:
                sock = socket.create_connection(("127.0.0.1", cport), timeout=2)
                break
            except OSError:
                time.sleep(1)
        if sock is None:
            raise RuntimeError(f"server never opened control port {cport}")
        f = sock.makefile("rwb")

        def step(action) -> Optional[dict]:
            f.write((json.dumps(action, separators=(",", ":")) + "\n").encode())
            f.flush()
            line = f.readline()
            return json.loads(line) if line else None

        raw = json.loads(f.readline())
        track = {"path": 0.0, "prev": None, "hp_lost": 0.0, "prev_hp": None,
                 "buttons": {}, "min_along_enemy": 1e9}
        while raw is not None and int(raw.get("t", 0)) < max_game_ms:
            act = policy(raw)
            btn = act.get("blue", {}).get("t", "noop")
            track["buttons"][btn] = track["buttons"].get(btn, 0) + 1
            raw = step({"blue": act.get("blue", {"t": "noop"})})
            if raw is None:
                break
            champs = {u["tm"]: u for u in raw.get("u", []) if u.get("k") == "Champion"}
            b = champs.get(100)
            if b is None:
                break
            if track["prev"] is not None:
                track["path"] += math.dist(track["prev"], (b["x"], b["y"]))
            track["prev"] = (b["x"], b["y"])
            if track["prev_hp"] is not None and b["hp"] < track["prev_hp"]:
                track["hp_lost"] += track["prev_hp"] - b["hp"]
            track["prev_hp"] = b["hp"]

        champs = {u["tm"]: u for u in (raw or {}).get("u", []) if u.get("k") == "Champion"}
        b, r = champs.get(100, {}), champs.get(200, {})
        return {
            "t_s": int((raw or {}).get("t", 0)) / 1000.0,
            "blue_cs": b.get("cs"), "red_cs": r.get("cs"),
            "blue_gold": b.get("gold"), "red_gold": r.get("gold"),
            "blue_lvl": b.get("lvl"), "red_lvl": r.get("lvl"),
            "blue_hp_lost": round(track["hp_lost"]),
            "distance_travelled": round(track["path"]),
            "buttons": track["buttons"],
        }
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def random_policy(raw):
    """Uniform over the semantic orders -- the 'before training' reference."""
    import random
    champs = {u["tm"]: u for u in raw.get("u", []) if u.get("k") == "Champion"}
    b = champs.get(100)
    if b is None:
        return {"blue": {"t": "noop"}}
    c = random.random()
    if c < 0.45:
        return {"blue": {"t": "move",
                         "x": b["x"] + random.uniform(-1200, 1200),
                         "y": b["y"] + random.uniform(-1200, 1200)}}
    if c < 0.75:
        foes = [u for u in raw.get("u", []) if u.get("tm") == 200 and u.get("hp", 0) > 0]
        if foes:
            tgt = min(foes, key=lambda u: math.dist((b["x"], b["y"]), (u["x"], u["y"])))
            return {"blue": {"t": "attack", "id": tgt["id"]}}
    if c < 0.9:
        return {"blue": {"t": "cast", "slot": random.randint(0, 3)}}
    return {"blue": {"t": "noop"}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="path to update_*.pt, or 'random'")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-game-ms", type=int, default=600_000)
    ap.add_argument("--step-ticks", type=int, default=2)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.checkpoint == "random":
        policy, label = random_policy, "random"
    else:
        policy, label = trained_policy(args.checkpoint), Path(args.checkpoint).stem

    rows: List[Dict] = []
    logdir = _REPO / "lanerl/logs"
    logdir.mkdir(parents=True, exist_ok=True)
    for i in range(args.episodes):
        row = play_episode(policy, args.max_game_ms, args.step_ticks,
                           logdir / f"evalbot_{label}_{i}.log")
        rows.append(row)
        print(f"  ep{i}: t={row['t_s']:.0f}s cs={row['blue_cs']} (bot {row['red_cs']}) "
              f"gold={row['blue_gold']} lvl={row['blue_lvl']} "
              f"hp_lost={row['blue_hp_lost']} dist={row['distance_travelled']} "
              f"buttons={row['buttons']}")

    def agg(k):
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        return statistics.mean(v) if v else float("nan")

    print(f"\n{label}: n={len(rows)}  CS={agg('blue_cs'):.1f} vs bot {agg('red_cs'):.1f}  "
          f"gold={agg('blue_gold'):.0f}  lvl={agg('blue_lvl'):.1f}  "
          f"hp_lost={agg('blue_hp_lost'):.0f}  dist={agg('distance_travelled'):.0f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"label": label, "rows": rows}, indent=2))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

def trained_policy(ckpt_path: str):
    """A raw-obs -> wire-order callable backed by a trained checkpoint.

    Reuses the TRAINING adapters (LaneObservationAdapter / LaneActionEncoder)
    rather than reimplementing the observation pipeline -- an eval that builds
    its observation differently from training measures the wrong thing, and
    that class of mismatch has already bitten this project twice.
    """
    import torch
    from lanerl_train.lane_wiring import make_lane_adapters
    from lanerl_rl.model import LanePolicy

    # NOTE: payload["cfg"] is the PPO config (clip_eps, gamma, lr...), NOT the
    # model config. Build the model the way __main__ does, so eval and training
    # cannot drift apart -- a mismatch here would silently measure a different
    # network from the one that was trained.
    from lanerl_rl.model import ModelConfig

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    policy_net = LanePolicy(ModelConfig())
    policy_net.load_state_dict(payload["policy"])
    policy_net.eval()

    adapters = make_lane_adapters(train_step_source=lambda: 0)
    adapter = adapters.adapter_factory(0, "blue")
    encoder = adapters.encoder
    state = None

    from lanerl_rl.infer import collate_observations

    def act(raw):
        nonlocal state
        with torch.no_grad():
            obs = adapter.build(raw, "blue")
            # (N,1,...) batch, exactly as the rollout loop builds it -- act()
            # takes a collated dict, not an AgentObservation.
            batch = collate_observations([obs], device="cpu")
            if state is None:
                state = policy_net.initial_state(1, device="cpu")
            action, _logp, _v, state = policy_net.act(batch, state, deterministic=False)
            # act() returns (B,1,...) tensors; the encoder wants scalars
            flat = {k: int(v.reshape(-1)[0]) for k, v in action.items()}
        return {"blue": encoder.encode(flat, raw, "blue")}

    return act
