#!/usr/bin/env python
"""Do the agent's attack orders actually land on the minion it aimed at?

The unexplained observation from the behaviour-cloning work: the agent issued
essentially the SAME button counts as a much better player -- 1,224 attacks
against 1,157, comparable movement -- and scored half the CS. Same actions,
half the result. Nothing has explained that, and every investigation since has
looked at the reward or the optimiser instead.

The reward is now known to be correct (last_hit is 81% of raw return and tracks
cs_at_10 one-for-one) and the optimiser is healthy (explained variance 0.70,
kl_ref 0.06, no clipping trouble), while CS stays flat. So the remaining
candidate is that the orders do not do what they say.

That is directly checkable and never has been. ``LanerlControl`` already emits,
per champion:

    tgt   Champion.TargetUnit.NetId -- what the ENGINE thinks we are attacking
    atk   Champion.IsAttacking      -- a swing is actually in flight
    mo    Champion.MoveOrder        -- the engine's current order enum

All three are registered ``unconsumed`` in ``frame.WIRE_FIELDS``, emitted for
exactly this question, and decoded by nothing. This probe reads them.

It reports, over one real game:

    issued        attack orders we put on the wire, with a target netid
    stuck         the engine's `tgt` equals the netid we asked for, next frame
    swinging      `atk` was set at some point while that target was held
    retargeted    engine `tgt` is some OTHER live unit -- the order was
                  overridden, which is the failure that would cost CS while
                  leaving the button counts identical
    dropped       engine `tgt` is 0 -- the order went nowhere at all

    python lanerl/attack_probe.py [--seconds 420] [--port-base 47000]
"""
from __future__ import annotations

import argparse
import collections
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("attack_probe")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seconds", type=float, default=420.0)
    ap.add_argument("--port-base", type=int, default=47000)
    ap.add_argument("--policy", default="demos/bc_policy.pt")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from lanerl_rl import constants as C
    from lanerl_rl.model import LanePolicy, ModelConfig
    from lanerl_train.lane_wiring import (
        LaneActionEncoder, make_lane_adapters, LanePolicyActor,
    )
    from lanerl_train.ports import PortAllocator
    from lanerl_train.protocols import BLUE, RED
    import torch

    from lanerl_train.vec import (
        EpisodeSpec, ServerLaunchSpec, SideAssignment, VecDriver, VecLaneEnv,
    )

    ports = PortAllocator(base=args.port_base).allocate(1)
    adapters = make_lane_adapters(train_step_source=lambda: 0)
    policy = LanePolicy(ModelConfig()).to(args.device)
    blob = torch.load(args.policy, map_location=args.device, weights_only=False)
    policy.load_state_dict(blob.get("policy", blob) if isinstance(blob, dict) else blob)
    actor = LanePolicyActor(policy, device=args.device)

    env = VecLaneEnv(
        n=1, specs=[ServerLaunchSpec(bot_teams="none")], ports=ports,
        log_dir=Path("/tmp/attack_probe_logs"),
    )
    driver = VecDriver(
        env=env, policies={"self": actor},
        adapter_factory=adapters.adapter_factory, encoder=adapters.encoder,
        assignments=[SideAssignment(blue="self", red="self")],
        episode=EpisodeSpec(end_on_death=False),
    )

    # Wrap the encoder so every order we put on the wire is recorded alongside
    # the frame it was chosen on. Wrapping rather than editing lane_wiring
    # keeps this a probe: nothing in the training path changes.
    real_encode = adapters.encoder.encode
    issued: dict = {}

    def spy_encode(action, raw, side):
        order = real_encode(action, raw, side)
        if isinstance(order, dict) and order.get("t") == "attack":
            issued[side] = {"netid": order.get("id"), "t": raw.get("t")}
        else:
            issued.pop(side, None)
        return order

    adapters.encoder.encode = spy_encode

    counts = collections.Counter()
    driver.start()
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            # Ordering matters and is easy to get backwards. Inside one
            # driver.step(): _forward builds from frame N, _scatter encodes the
            # orders (filling `issued`), and env.step sends them and reads back
            # frame N+1. So AFTER step returns, `issued` holds the orders chosen
            # on frame N and last_obs is the frame those orders produced --
            # which is exactly the pairing this probe needs. Snapshotting
            # `issued` before the call would compare the previous decision's
            # order against this decision's frame.
            issued.clear()
            driver.step()
            pending = {s: dict(v) for s, v in issued.items()}
            raw = env.last_obs[0]
            if raw is None:
                continue
            units = {u["id"]: u for u in raw.get("u", [])}
            for side, want in pending.items():
                team = C.TEAM_BLUE if side == BLUE else C.TEAM_RED
                me = next((u for u in raw.get("u", [])
                           if u.get("tm") == team and u.get("k") == "Champion"), None)
                if me is None or want["netid"] is None:
                    continue
                counts["issued"] += 1
                tgt, atk = me.get("tgt"), me.get("atk")
                if tgt in (None, 0):
                    counts["dropped"] += 1
                elif int(tgt) == int(want["netid"]):
                    counts["stuck"] += 1
                    if atk:
                        counts["swinging"] += 1
                else:
                    counts["retargeted"] += 1
                    other = units.get(int(tgt))
                    counts[f"retargeted_to_{(other or {}).get('k', 'unknown')}"] += 1
    finally:
        env.close()

    n = max(1, counts["issued"])
    print("\n=== attack orders, one game ===")
    for k in ("issued", "stuck", "swinging", "dropped", "retargeted"):
        print(f"  {k:14s} {counts[k]:7d}  {100 * counts[k] / n:5.1f}%")
    for k, v in sorted(counts.items()):
        if k.startswith("retargeted_to_"):
            print(f"    {k:22s} {v:7d}")
    print("\nstuck+dropped+retargeted should equal issued; anything large in "
          "'retargeted' or 'dropped' is an order the agent believes it gave "
          "and the engine did not execute.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
