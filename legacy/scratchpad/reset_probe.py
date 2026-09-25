"""Server-level probe: does the in-process reset actually rewind the clock, and
does `episode_done` re-fire on the very next observation?

Drives N REAL server instances through the real VecLaneEnv control channel with
a deliberately tiny max_game_ms, records every observation's `t`, every done and
every reset command sent, then cross-checks the Python-side reset count against
the LANERL_RESET lines the server itself printed.

Usage: python scratchpad/reset_probe.py [--n 2] [--max-game-ms 20000] [--steps 900]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lanerl_train.vec import EpisodeSpec, ServerLaunchSpec, VecLaneEnv, episode_done

RESET_RE = re.compile(r"LANERL_RESET ms=.* t_before=(\d+) t_after=(\d+)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--max-game-ms", type=int, default=20_000)
    ap.add_argument("--steps", type=int, default=900)
    ap.add_argument("--bot", default="both")
    ap.add_argument("--out", default="scratchpad/reset_probe.json")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")

    spec = EpisodeSpec(max_game_ms=args.max_game_ms, end_on_death=True, max_steps=10**9)
    launch = ServerLaunchSpec(bot_teams=args.bot)
    log_dir = Path("scratchpad/reset_probe_logs")
    env = VecLaneEnv(n=args.n, spec=launch, log_dir=log_dir)

    steps_in_ep = [0] * args.n
    ts: list[list[int]] = [[] for _ in range(args.n)]
    dones: list[list[dict]] = [[] for _ in range(args.n)]
    resets_sent = [0] * args.n
    # (step, instance, t) for each observation that came back on the step
    # immediately after we sent a reset -- the post-reset observation.
    post_reset_t: list[list[int]] = [[] for _ in range(args.n)]

    result = env.start()
    for i in range(args.n):
        if result.obs[i] is not None:
            ts[i].append(int(result.obs[i]["t"]))
            steps_in_ep[i] += 1

    step = 0
    try:
        while step < args.steps:
            step += 1
            result = env.step([None] * args.n)
            want_reset = []
            for i in range(args.n):
                raw = result.obs[i]
                if raw is None:
                    continue
                t = int(raw.get("t", 0))
                ts[i].append(t)
                steps_in_ep[i] += 1
                done, reason = episode_done(raw, spec, steps_in_ep[i])
                if done:
                    dones[i].append({"step": step, "t": t, "reason": reason,
                                     "steps_in_ep": steps_in_ep[i]})
                    want_reset.append(i)
            if not want_reset:
                continue
            for i in want_reset:
                resets_sent[i] += 1
            rr = env.reset_episodes(sorted(want_reset))
            for i in range(args.n):
                if rr.obs[i] is None:
                    continue
                t = int(rr.obs[i].get("t", 0))
                ts[i].append(t)
                if i in want_reset:
                    post_reset_t[i].append(t)
                    steps_in_ep[i] = 0
                else:
                    steps_in_ep[i] += 1
    finally:
        env.close()

    server_resets = []
    for i in range(args.n):
        p = log_dir / f"instance{i:03d}.log"
        rows = []
        if p.exists():
            for line in p.read_text(errors="replace").splitlines():
                m = RESET_RE.search(line)
                if m:
                    rows.append({"t_before": int(m.group(1)), "t_after": int(m.group(2))})
        server_resets.append(rows)

    out = {
        "n": args.n,
        "max_game_ms": args.max_game_ms,
        "steps_driven": step,
        "per_instance": [
            {
                "obs_count": len(ts[i]),
                "dones": len(dones[i]),
                "done_reasons": [d["reason"] for d in dones[i]],
                "done_steps_in_ep": [d["steps_in_ep"] for d in dones[i]],
                "resets_sent": resets_sent[i],
                "server_resets": len(server_resets[i]),
                "server_reset_rows": server_resets[i],
                "post_reset_t": post_reset_t[i],
                "max_t": max(ts[i]) if ts[i] else None,
                "first_20_t": ts[i][:20],
            }
            for i in range(args.n)
        ],
    }
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:4000])

    ok = True
    for i in range(args.n):
        d = out["per_instance"][i]
        if d["resets_sent"] != d["server_resets"]:
            print(f"MISMATCH instance {i}: sent {d['resets_sent']} resets, "
                  f"server logged {d['server_resets']}")
            ok = False
        bad = [t for t in d["post_reset_t"] if t >= args.max_game_ms]
        if bad:
            print(f"CLOCK DID NOT REWIND instance {i}: post-reset t values {bad[:5]}")
            ok = False
        # a re-fire shows up as an episode that "ended" after a handful of steps
        refires = [s for s in d["done_steps_in_ep"] if s < 5]
        if refires:
            print(f"RE-FIRE instance {i}: {len(refires)} dones within 5 steps of a reset")
            ok = False
    print("PROBE", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
