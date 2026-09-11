"""Does the repaired episode readout produce a real CS number against a real
server, or still the post-reset zero?

Runs the PRODUCTION path (VecDriver + lane_wiring.collect_rollout) against one
live instance with the scripted bot driving both champions and an episode long
enough for a wave to reach the lane, then prints every EpisodeResult next to
the server's own LANERL_CS rows for the same game time.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lanerl_rl.model import LanePolicy, ModelConfig
from lanerl_train.lane_wiring import LanePolicyActor, collect_rollout, make_lane_adapters
from lanerl_train.vec import (
    EpisodeSpec,
    ServerLaunchSpec,
    SideAssignment,
    VecDriver,
    VecLaneEnv,
)

SELF = "self"
CS_RE = re.compile(r"LANERL_CS t=(\d+) name=(\S+) team=(\d+) cs=(\d+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-game-ms", type=int, default=150_000)
    ap.add_argument("--episodes", type=int, default=2)
    #: "bot" leaves RED to the in-server script, which actually last-hits; with
    #: the policy on both sides an untrained network farms nothing and CS is a
    #: legitimate 0, which cannot tell a repaired readout from a broken one.
    ap.add_argument("--red", choices=("bot", "policy"), default="bot")
    ap.add_argument("--log-dir", default="scratchpad/cs_probe_logs")
    ap.add_argument("--out", default="scratchpad/cs_readout_probe.json")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    log_dir = Path(args.log_dir)
    env = VecLaneEnv(n=1, spec=ServerLaunchSpec(bot_teams="both"), log_dir=log_dir)
    policy = LanePolicy(
        ModelConfig(core_dim=32, d_model=32, ffn_dim=32, n_layers=1, n_heads=2, mlp_hidden=32)
    )
    actor = LanePolicyActor(policy)
    adapters = make_lane_adapters(train_step_source=lambda: 0)
    driver = VecDriver(
        env=env,
        policies={SELF: actor},
        adapter_factory=adapters.adapter_factory,
        encoder=adapters.encoder,
        assignments=[SideAssignment(blue=SELF, red=None if args.red == "bot" else SELF)],
        episode=EpisodeSpec(max_game_ms=args.max_game_ms, max_steps=10**9),
    )
    driver.start()
    eps = []
    try:
        for _ in range(60):
            r = collect_rollout(
                driver, actor, adapters.reward_contexts, SELF,
                num_steps=256, gamma=0.99, gae_lambda=0.95,
            )
            eps.extend(r.episodes)
            print(f"rollout: rows={r.steps} slots={r.parallel_envs} episodes so far={len(eps)}",
                  flush=True)
            if len(eps) >= args.episodes:
                break
    finally:
        env.close()

    text = (log_dir / "instance000.log").read_text(errors="replace")
    server_cs = [
        {"t": int(m.group(1)), "name": m.group(2), "team": int(m.group(3)), "cs": int(m.group(4))}
        for m in CS_RE.finditer(text)
    ]
    near = [r for r in server_cs if abs(r["t"] - args.max_game_ms) <= 30_000]
    out = {
        "max_game_ms": args.max_game_ms,
        "episodes": [
            {
                "reason": e.reason,
                "length_steps": e.length_steps,
                "cs_at_10": e.cs_at_10,
                "ep_return": e.ep_return,
            }
            for e in eps
        ],
        "server_resets": text.count("LANERL_RESET ms="),
        "server_cs_near_limit": near[-8:],
        "server_cs_max": max((r["cs"] for r in server_cs), default=None),
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
