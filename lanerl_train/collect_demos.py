#!/usr/bin/env python
"""Collect (observation, action) demonstrations from the scripted bot.

Why behaviour cloning at all: PPO from random init provably cannot solve this
task. Travel per decision is 345 u/s / 30 Hz = 11.5 units, so reaching lane
(11,866 units) by random walk needs ~1.06M steps -- 9.9 hours of game against a
10-minute episode. The agent never saw a minion in 13,475 updates.

The observations are built with the TRAINING adapter, so the BC set and the RL
rollouts come from the same pipeline. A BC set built from a different
observation path teaches the network to imitate on inputs it will never see.

Labels come from the server's per-champion "demo" field: the order the bot
actually committed to, in the RL action space. That field is registered
server-only in the leak audit -- it is the supervised TARGET, never an input.

  python -m lanerl_train.collect_demos --games 4 --out demos/bot.npz
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

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



def move_bins_for(adapter, raw, team, goal_x, goal_y):
    """Invert decode_action's move mapping: world goal -> (move_x, move_z) bins.

    Without this, BC clones only the DECISION to move and not the direction --
    which teaches nothing about walking to lane, the single behaviour the RL
    agent could never discover on its own.

    decode_action does:  canonical bin -> unit vector -> transform.vector()
                         -> world direction -> goal = self + dir * move_distance
    The transform is its own inverse, so applying it to the world direction
    recovers the canonical one, and the bins are its nearest grid points.
    """
    import numpy as _np
    from lanerl_rl import constants as _C

    ch = next((u for u in raw.get("u", [])
               if u.get("k") == "Champion" and u.get("tm") == team), None)
    if ch is None:
        return None
    dx, dy = float(goal_x) - float(ch["x"]), float(goal_y) - float(ch["y"])
    norm = math.hypot(dx, dy)
    if norm < 1e-6:
        return None
    wx, wy = dx / norm, dy / norm
    try:
        # to_lane_vector, NOT vector. LaneTransform.vector is lane-local ->
        # WORLD (that is the direction decode_action needs); the inverse is
        # to_lane_vector, and the class docstring says so explicitly. Using
        # vector() here treated a general rotation as if it were an involution
        # -- true of the old MirrorTransform, false of this one. It made every
        # BLUE move label point the wrong way while red's were right, so BC
        # averaged the two sides to the centre bin and learned to stand still:
        # 288 units from spawn over 300 s against a lane 11,866 units away.
        tx, tz = adapter.builder.transform.to_lane_vector(wx, wy)
    except Exception:
        return None
    bins = _C.MOVE_BIN_VALUES
    return int(_np.argmin(_np.abs(bins - tx))), int(_np.argmin(_np.abs(bins - tz)))


def collect_game(max_game_ms: int, step_ticks: int, log: Path,
                 keep_noop_frac: float, seed: int = 1234) -> Dict[str, List]:
    """One bot-vs-bot game; returns per-side observation/label lists."""
    from lanerl_train.lane_wiring import make_lane_adapters

    adapters = make_lane_adapters(train_step_source=lambda: 0)
    builders = {"blue": adapters.adapter_factory(0, "blue"),
                "red": adapters.adapter_factory(0, "red")}
    team_of = {"blue": 100, "red": 200}

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
        LANERL_BOT="both",                      # both sides demonstrate
        LANERL_CONTROL_PORT=str(cport),
        LANERL_STEP_TICKS=str(step_ticks),
        # Vary the bot's seed per game. The server is fully deterministic given
        # a seed and our (constant) action stream, so two games at the default
        # seed came back BYTE-IDENTICAL -- 16,930 samples each with the same
        # action histogram. That doubles the dataset while adding no
        # information, and a BC prior trained on one repeated game clones one
        # trajectory rather than a policy.
        LANERL_BOT_SEED=str(seed),
    )
    proc = subprocess.Popen(
        [str(BIN / "GameServerConsole"), "--config", str(CFG), "--port", str(gport)],
        cwd=str(BIN), env=env, stdout=log.open("w"), stderr=subprocess.STDOUT,
    )
    out: Dict[str, List] = {"obs": [], "label": [], "side": []}
    rng = np.random.default_rng(0)
    # The bot commits its order during tick T; the control channel reports it in
    # the observation for T+1, by which time a cast it just made is on cooldown
    # and the action mask correctly forbids it. Pairing label[T+1] with obs[T+1]
    # therefore produced 787/99,655 rows whose label the policy CANNOT emit --
    # 100% of W and 100% of E casts -- and those rows dominated the BC gradient
    # (cross-entropy against a -1e9 masked logit is ~1e9). Hold the previous
    # observation and attach the label to the state the bot actually saw.
    prev_obs: Dict[str, object] = {}
    prev_netids: Dict[str, list] = {}
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
        empty = (json.dumps({}, separators=(",", ":")) + "\n").encode()

        raw = json.loads(f.readline())
        while raw is not None and int(raw.get("t", 0)) < max_game_ms:
            for side, team in team_of.items():
                ch = next((u for u in raw.get("u", [])
                           if u.get("k") == "Champion" and u.get("tm") == team), None)
                if ch is None:
                    continue
                demo = ch.get("demo")
                if not demo:
                    continue
                kind = demo.get("t", "noop")
                # build() EVERY frame, and subsample only which frames become
                # training rows. The builder is stateful -- dt, unit memory,
                # ability intel and the attack clock are all computed from the
                # frame stream it is shown -- so skipping build() on ~65% of
                # frames at irregular gaps handed BC a different observation
                # distribution than VecDriver produces at RL time, which builds
                # every frame. It showed up as global_vec.dt_norm mean 1.91 /
                # p90 5.0 / 5.6% saturated at DT_NORM_CAP against ~1.0 by
                # construction at RL time, and as enemy_ability_*_never_observed
                # stuck at 1.0 for all 99,654 rows (the intel model needs two
                # sightings <= 300 ms apart and never saw consecutive frames).
                # This is the exact mismatch this module's docstring promises to
                # prevent.
                try:
                    obs = builders[side].build(raw, side)
                except Exception:
                    continue
                slot_netids = list(builders[side].last_slot_netids)
                # Tell the builder about the swing, exactly as LaneEnv.decode
                # does at RL time. The attack clock has no server source: it
                # must be *told*, and the collector never told it -- so all four
                # attack-cycle features were identically 0 across every BC row
                # while being live at RL time. The prior was trained with them
                # dead and would meet them alive, the same train/test mismatch
                # as the dt_norm one.
                #
                # This matches RL's convention rather than fixing it: the clock
                # is driven by orders ISSUED, so an attack the engine refuses
                # (target out of range) still starts it. The real fix is the
                # server's own `atk` (Champion.IsAttacking) rising edge, which
                # is already on the wire and HUD-legal for our own champion --
                # but that changes RL's features too, so it belongs in its own
                # change, not folded in behind a running chain.
                if kind == "attack":
                    builders[side].builder.note_attack(float(raw.get("t", 0)))
                # noop dominates ~80% of frames because the bot decides on a
                # 150 ms reaction clock while we observe at 30 Hz. Keeping all
                # of them would train a policy that mostly stands still.
                drop_row = kind == "noop" and rng.random() > keep_noop_frac
                demo = dict(demo)
                # Direction, but ONLY for orders that actually carry one.
                # LanerlBot.Decide sets LastOrderX/Y in the move branch and
                # never clears them, so demo.x/y are present (and stale) on
                # attack and noop rows too -- gate on the order kind, not on
                # whether the field exists.
                if kind == "move" and demo.get("x") is not None:
                    mb = move_bins_for(builders[side], raw, team,
                                       demo["x"], demo["y"])
                    if mb is not None:
                        demo["mx"], demo["mz"] = mb
                # label THIS frame's order against the PREVIOUS frame's
                # observation -- the state the bot was looking at when it chose
                earlier = prev_obs.get(side)
                earlier_netids = prev_netids.get(side)
                # The attack REFERENT. Cloning "press attack" without which unit
                # to attack teaches nothing about a last hit: the target head
                # was left at zeros and unsupervised, so the trained policy's
                # target distribution was indistinguishable from random init and
                # put ~35% of its attack mass on its OWN minions -- a legal
                # order the server never complains about. 1,091-1,965 attack
                # orders per game, CS 0.
                #
                # Resolve against the slot map of the observation we PAIR with
                # (frame T-1), not the current frame's: the slot index only
                # means anything relative to the entity block the policy is
                # looking at.
                if kind == "attack" and demo.get("id") is not None \
                        and earlier_netids is not None:
                    try:
                        demo["slot_idx"] = earlier_netids.index(int(demo["id"]))
                    except ValueError:
                        # the bot's target is not in the paired observation's
                        # entity block (fogged, or aged out) -- unlabelable
                        demo.pop("slot_idx", None)
                if earlier is not None and not drop_row:
                    out["obs"].append(earlier)
                    out["label"].append(demo)
                    out["side"].append(side)
                prev_obs[side] = obs
                prev_netids[side] = slot_netids
            f.write(empty)
            f.flush()
            line = f.readline()
            raw = json.loads(line) if line else None
        return out
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--games", type=int, default=2)
    ap.add_argument("--max-game-ms", type=int, default=600_000)
    ap.add_argument("--step-ticks", type=int, default=2)
    ap.add_argument("--keep-noop-frac", type=float, default=0.15,
                    help="fraction of no-op frames to keep (they are ~80%% of all frames)")
    ap.add_argument("--out", default="demos/bot_demos.npz")
    args = ap.parse_args()

    logdir = _REPO / "lanerl/logs"
    logdir.mkdir(parents=True, exist_ok=True)
    all_obs, all_lab, all_side = [], [], []
    for g in range(args.games):
        got = collect_game(args.max_game_ms, args.step_ticks,
                           logdir / f"demos_{g}.log", args.keep_noop_frac,
                           seed=1234 + g * 7919)
        all_obs += got["obs"]; all_lab += got["label"]; all_side += got["side"]
        hist = Counter(d.get("t") for d in got["label"])
        print(f"  game {g}: {len(got['label'])} samples  {dict(hist)}")

    if not all_lab:
        print("NO DEMONSTRATIONS COLLECTED -- refusing to write an empty dataset")
        return 1
    if args.games > 1:
        # A duplicate-game check, because the first collection silently produced
        # two identical games and only the matching histograms gave it away.
        import hashlib
        sigs = set()
        per = len(all_obs) // args.games
        for g in range(args.games):
            chunk = all_obs[g * per : (g + 1) * per]
            h = hashlib.md5(np.stack([o.self_vec for o in chunk]).tobytes()).hexdigest()
            sigs.add(h)
        if len(sigs) < args.games:
            print(f"WARNING: only {len(sigs)} distinct games out of {args.games} -- "
                  f"the bot seed is not varying and the extra games add no information")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Save EVERY field the model consumes, not just the three actor arrays.
    # LanePolicy.forward also needs priv_entities / priv_vec / the pad masks,
    # and the action masks; a dataset missing them cannot be fed to the network
    # at all, and reconstructing them later would mean a second, divergent
    # observation path -- the exact mismatch this collector exists to avoid.
    from lanerl_rl.infer import _FLOAT_KEYS, _BOOL_KEYS, _MASK_KEYS

    arrays = {k: np.stack([getattr(o, k) for o in all_obs]) for k in _FLOAT_KEYS}
    arrays.update({k: np.stack([getattr(o, k) for o in all_obs]) for k in _BOOL_KEYS})
    arrays.update({
        f"mask_{k}": np.stack([getattr(o.action_mask, k) for o in all_obs])
        for k in _MASK_KEYS
    })
    np.savez_compressed(
        out,
        label_json=np.array([json.dumps(d) for d in all_lab]),
        side=np.array(all_side),
        **arrays,
    )
    hist = Counter(d.get("t") for d in all_lab)
    print(f"\nwrote {out}  n={len(all_lab)}  actions={dict(hist)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
