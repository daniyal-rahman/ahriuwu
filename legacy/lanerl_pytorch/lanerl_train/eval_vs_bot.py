#!/usr/bin/env python
"""Evaluate a policy against the FIXED scripted bot.

Why this exists: in symmetric self-play `score` is 0.5 by construction, so the
training run cannot tell you whether the policy improved. The only honest
progress metric is performance against an opponent that does not move --
here, the frozen scripted Garen.

Setup: the control channel drives BLUE only (we send no "red" key, so
ApplyActions leaves red alone), and LANERL_BOT=purple hands red to the bot.

ONE SERVER PROCESS PER EPISODE, and that is a measurement decision, not an
implementation detail.  The server's IN-PROCESS episode reset strips the
champion's rune and mastery page: max hp 672 -> 616, attack damage 78.14 ->
57.88.  So a run of N episodes inside one process is one game played by a runed
champion and N-1 played by a weaker one, and averaging them produced this
script's headline "BC = 37.3 CS" out of three games that were not the same
game.  A process restart costs ~12 s against a ~650 s episode -- 2% -- which is
a trivial price for a homogeneous sample.  :func:`play_episode` therefore
launches, connects to, and tears down its own server, and :func:`check_homogeneous`
refuses to print a mean over episodes whose starting stats disagree, so that if
the restart ever stops happening the script fails instead of quietly blending
two populations again.  (Once the server-side reset is fixed, the restart
becomes an optimisation to revisit -- the homogeneity check is what must stay.)

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


#: Champion fields that a rune/mastery page changes and that the server sends
#: on every frame.  ``ad`` is the one that moved most (78.14 with the page,
#: 57.88 without) and ``mhp`` the one that is always present.
_START_STAT_KEYS = ("mhp", "ad", "ap", "ar", "mr", "lvl")


class HeterogeneousEpisodes(RuntimeError):
    """Episodes that must not be averaged, because they were not the same game."""


def start_stats(raw: Optional[dict]) -> Dict[str, float]:
    """The blue champion's stats on the FIRST frame of an episode.

    The fingerprint of the rune and mastery page, and therefore of whether this
    episode was played in a freshly launched server or after an in-process
    reset that stripped it (mhp 672 -> 616, ad 78.14 -> 57.88).  Absent keys are
    omitted rather than defaulted: a guessed stat is how this project ended up
    with three different wrong attack-damage constants.
    """
    champs = {u["tm"]: u for u in (raw or {}).get("u", []) if u.get("k") == "Champion"}
    b = champs.get(100, {})
    return {k: float(b[k]) for k in _START_STAT_KEYS if b.get(k) is not None}


def check_homogeneous(rows: List[Dict]) -> None:
    """Refuse to average episodes whose champion did not start the same.

    The point of this script is a single headline number, so a blended sample
    is not a caveat, it is a wrong answer.  Loud, with the offending stats, and
    fatal.
    """
    seen: Dict[str, List[int]] = {}
    for i, r in enumerate(rows):
        key = json.dumps(r.get("start_stats") or {}, sort_keys=True)
        seen.setdefault(key, []).append(i)
    if len(seen) > 1:
        detail = "; ".join(f"episodes {v}: {k}" for k, v in sorted(seen.items()))
        raise HeterogeneousEpisodes(
            "these episodes did not start from the same champion stats, so their mean "
            "is not a measurement of anything: " + detail + ". The known cause is the "
            "server's in-process episode reset stripping the rune/mastery page (mhp "
            "672 -> 616, ad 78.14 -> 57.88); every episode here is supposed to get its "
            "own server process, so either that stopped happening or the page is being "
            "applied inconsistently at launch."
        )


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def play_episode(policy, max_game_ms: int, step_ticks: int, log_path: Path,
                 red: str = "bot", bot_config: Optional[str] = None) -> Dict:
    """One game in its OWN server process: policy drives blue, scripted bot red.

    The process is launched here and torn down in the ``finally`` below, once
    per call, deliberately -- see the module docstring.  Do not hoist it out to
    amortise the ~12 s start-up across episodes: that is exactly the change
    that made episodes 2..N of every previous measurement base-stats games.
    """
    cport, gport = free_port(), free_port()
    env = dict(os.environ)
    env.update(
        DOTNET_ROOT=str(VENDOR / "dotnet"),
        LANERL_HEADLESS="1", LANERL_FREERUN="1",
        # Match TRAINING exactly: vec.ServerLaunchSpec.toponly defaults True,
        # and LANERL_TOPONLY=1 disables jungle camps (LevelScript.cs:165) and
        # every non-top minion wave (:292). Omitting it here meant the BC
        # prior was cloned on a full three-lane map WITH jungle and then
        # fine-tuned on a top-only one -- a train/deploy observation shift in
        # the very module whose docstring promises the BC set and the RL
        # rollouts come from the same pipeline.
        LANERL_TOPONLY="1",
        # red = the frozen scripted bot, or NOBODY.
        #
        # "idle" leaves red undriven: no bot, and the control channel sends no
        # red key, so the enemy champion never leaves the fountain. That is the
        # CEILING measurement -- it separates "cannot last-hit" from "is being
        # contested", which no contested game can distinguish. Without it
        # "plateau" has no denominator: we know the agent scores ~46 and not
        # whether the reachable maximum is 50 or 90.
        LANERL_BOT=("purple" if red == "bot" else "none"),
        LANERL_CONTROL_PORT=str(cport),
        LANERL_STEP_TICKS=str(step_ticks),
    )
    if bot_config:
        env["LANERL_BOT_CONFIG"] = str(bot_config)
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
        first = start_stats(raw)
        track = {"path": 0.0, "prev": None, "hp_lost": 0.0, "prev_hp": None,
                 "buttons": {}, "min_along_enemy": 1e9}
        # THE DENOMINATOR. CS alone cannot say whether 45 is good: the number
        # of enemy minions that die in a game is not fixed, and it moves with
        # how the wave is being played. Measured directly -- removing the
        # opponent made CS go DOWN (37.5 uncontested against 45.4 vs the bot),
        # because with nobody killing your minions your wave overruns theirs
        # and your own minions take the kills you wanted. So "uncontested CS"
        # is not an upper bound, and conversion share is what "plateau" has to
        # be measured against.
        seen_enemy_minions: set = set()
        alive_prev: set = set()
        died_enemy_minions: set = set()
        while raw is not None and int(raw.get("t", 0)) < max_game_ms:
            act = policy(raw)
            btn = act.get("blue", {}).get("t", "noop")
            track["buttons"][btn] = track["buttons"].get(btn, 0) + 1
            raw = step({"blue": act.get("blue", {"t": "noop"})})
            if raw is None:
                break
            champs = {u["tm"]: u for u in raw.get("u", []) if u.get("k") == "Champion"}
            # Enemy minions currently alive; anything that was alive last
            # frame and is gone now, died. Keyed by netid, so a minion that
            # merely leaves the observation is NOT counted -- this eval sees
            # the whole map, so a disappearance is a death.
            alive_now = {
                u["id"] for u in raw.get("u", [])
                if u.get("k") in ("Minion", "LaneMinion") and u.get("tm") != 100
            }
            seen_enemy_minions |= alive_now
            died_enemy_minions |= (alive_prev - alive_now)
            alive_prev = alive_now
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
            "enemy_minions_seen": len(seen_enemy_minions),
            "enemy_minions_died": len(died_enemy_minions),
            "blue_gold": b.get("gold"), "red_gold": r.get("gold"),
            "blue_lvl": b.get("lvl"), "red_lvl": r.get("lvl"),
            "blue_hp_lost": round(track["hp_lost"]),
            "distance_travelled": round(track["path"]),
            "buttons": track["buttons"],
            # Checked across episodes by check_homogeneous() before anything is
            # averaged. Carried in the row so --out records it too: a saved
            # result nobody can re-check is the same problem one step later.
            "start_stats": first,
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

    def reset():
        """Drop all cross-episode state between games.

        Neither the adapter nor the GRU was reset, so episodes 2..N carried the
        previous game's unit memory and recurrent state. The builder's own
        detector caught it -- "game clock went backwards (599979 -> 16 ms)" in
        lanerl/logs/bceval-682.out, once per boundary. With net ids reused (what
        a fresh server actually hands out) the first frame of episode 2 reported
        the enemy last seen 80% down the lane at age 0 -- a position from the
        game before. At the default --episodes 3, 2 of 3 reported games were
        contaminated, so every CS number was measured on a polluted observation.
        """
        nonlocal state, adapter
        adapter = adapters.adapter_factory(0, "blue")
        state = None

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

    act.reset = reset
    return act


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="path to update_*.pt, or 'random'")
    ap.add_argument("--episodes", type=int, default=3)
    ap.add_argument("--max-game-ms", type=int, default=600_000)
    ap.add_argument("--step-ticks", type=int, default=2)
    ap.add_argument("--out", default="")
    ap.add_argument(
        "--red", choices=("bot", "idle"), default="bot",
        help="'bot' (default) is the frozen scripted opponent. 'idle' leaves "
             "red undriven in the fountain -- an UNCONTESTED lane, which is "
             "the only way to separate 'cannot last-hit' from 'is being "
             "contested'. Uncontested CS is the denominator that makes the "
             "word 'plateau' mean anything.",
    )
    ap.add_argument(
        "--bot-config", default=None,
        help="LANERL_BOT_CONFIG for the scripted side. anchor_diamond.json is "
             "lastHitAccuracy 1.0 with an 80ms reaction and no damage error, "
             "i.e. very close to a perfect last-hitter -- run it with "
             "--red idle to measure what this map can yield at all.",
    )
    args = ap.parse_args()

    if args.checkpoint == "random":
        policy, label = random_policy, "random"
    else:
        policy, label = trained_policy(args.checkpoint), Path(args.checkpoint).stem

    rows: List[Dict] = []
    logdir = _REPO / "lanerl/logs"
    logdir.mkdir(parents=True, exist_ok=True)
    for i in range(args.episodes):
        # Each episode is a fresh server process, so the policy's own carried
        # state must be dropped to match.
        if hasattr(policy, "reset"):
            policy.reset()
        row = play_episode(policy, args.max_game_ms, args.step_ticks,
                           logdir / f"evalbot_{label}_{i}.log",
                           red=args.red, bot_config=args.bot_config)
        rows.append(row)
        died = row.get("enemy_minions_died") or 0
        conv = (100.0 * (row.get("blue_cs") or 0) / died) if died else float("nan")
        print(f"  ep{i}: t={row['t_s']:.0f}s cs={row['blue_cs']} (bot {row['red_cs']}) "
              f"died={died} conv={conv:.0f}% "
              f"gold={row['blue_gold']} lvl={row['blue_lvl']} "
              f"hp_lost={row['blue_hp_lost']} dist={row['distance_travelled']} "
              f"buttons={row['buttons']}")

    # Before the mean, not after: the whole output of this script is one
    # number, and a number averaged over two populations is worse than none.
    check_homogeneous(rows)

    def agg(k):
        v = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
        return statistics.mean(v) if v else float("nan")

    # CS@10 has sd ~7.3 (144 self-play episodes, runs/rl-bc4-0912), so the
    # default --episodes 3 carries a 95% CI of about +-18 CS. Print the spread
    # beside the mean: "BC = 37.3" was quoted as a fact off three games.
    cs = [r["blue_cs"] for r in rows if isinstance(r.get("blue_cs"), (int, float))]
    if len(cs) >= 2:
        sd = statistics.stdev(cs)
        print(f"\n  CS@10 spread: sd={sd:.1f} over n={len(cs)}; "
              f"se={sd / math.sqrt(len(cs)):.1f}. See "
              f"AnchorEvalConfig.episodes_per_anchor for what n games can resolve.")

    print(f"\n{label}: n={len(rows)}  CS={agg('blue_cs'):.1f} vs bot {agg('red_cs'):.1f}  "
          f"gold={agg('blue_gold'):.0f}  lvl={agg('blue_lvl'):.1f}  "
          f"hp_lost={agg('blue_hp_lost'):.0f}  dist={agg('distance_travelled'):.0f}")
    if args.out:
        Path(args.out).write_text(json.dumps({"label": label, "rows": rows}, indent=2))
        print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
