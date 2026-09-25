#!/usr/bin/env python
"""Record what the policy SAW, CONSIDERED and DID, per decision, for overlay.

Intended use: run one policy-vs-policy game while the League client renders it
on native Windows, screen-record that, and afterwards composite this recording
on top -- network activations and annotated inputs/outputs, aligned to the
video by game clock.

Why capture and render are separate
-----------------------------------
Compositing live would mean doing matplotlib work inside the decision loop, on
the machine that is simultaneously running two policies and a game client, and
any frame it dropped would be gone. This writes a compact JSONL instead, so the
Windows session only has to produce (video + JSONL) and every rendering choice
stays re-doable afterwards without replaying the game.

What one record holds
---------------------
Per decision, per side:

    world/screen  the champion's position, and every entity SLOT the policy
                  could act on, in BOTH world and normalised screen
                  coordinates. Screen comes from ``lanerl_rl.projection``
                  centred on that champion, which is the same tilted-camera
                  model the replay pipeline uses -- so an overlay drawn at
                  these coordinates lands on the unit the network was actually
                  considering.
    heads         the full probability vector for button / move_x / move_z /
                  target. The target vector over 32 slots is the interesting
                  one: it is literally "which minion am I about to click", and
                  it has a screen position for every entry.
    chosen        the sampled action, and the order that went on the wire.
    value         the critic's estimate.
    core          a small summary of the GRU hidden state (norm plus the first
                  k components), enough to show memory moving without dumping
                  512 floats per decision per side.

CAMERA REQUIREMENT for alignment
--------------------------------
The screen coordinates assume a champion-centred, locked camera -- that is what
``projection.centred_on`` models. If the recording is made with a free or
partly-unlocked camera the overlay will be systematically offset, and the error
looks like bad aim rather than a bad assumption. Lock the camera before
recording.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("viz_capture")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", default="lanerl/logs/viz_capture.jsonl")
    ap.add_argument("--seconds", type=float, default=700.0)
    ap.add_argument("--port-base", type=int, default=49000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--core-components", type=int, default=16)
    ap.add_argument("--freerun", action="store_true",
                    help="let the server run flat out. OFF by default: for a "
                         "recording the sim must advance at wall-clock rate or "
                         "the video and the JSONL cannot be aligned.")
    args = ap.parse_args()

    import torch
    from lanerl_rl import constants as C
    from lanerl_rl import projection
    from lanerl_rl.model import LanePolicy, ModelConfig
    from lanerl_train.lane_wiring import LanePolicyActor, make_lane_adapters
    from lanerl_train.ports import PortAllocator
    from lanerl_train.protocols import BLUE, RED
    from lanerl_train.vec import (
        EpisodeSpec, ServerLaunchSpec, SideAssignment, VecDriver, VecLaneEnv,
    )

    policy = LanePolicy(ModelConfig()).to(args.device)
    blob = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    policy.load_state_dict(blob.get("policy", blob) if isinstance(blob, dict) else blob)
    policy.eval()
    actor = LanePolicyActor(policy, device=args.device)

    adapters = make_lane_adapters(train_step_source=lambda: 0)
    ports = PortAllocator(base=args.port_base).allocate(1)
    env = VecLaneEnv(
        n=1, specs=[ServerLaunchSpec(bot_teams="none", freerun=args.freerun)],
        ports=ports, log_dir=Path("/tmp/viz_capture_logs"),
    )
    driver = VecDriver(
        env=env, policies={"self": actor},
        adapter_factory=adapters.adapter_factory, encoder=adapters.encoder,
        assignments=[SideAssignment(blue="self", red="self")],
        episode=EpisodeSpec(end_on_death=False),
    )

    # Intercept the distribution on its way out of the policy. act_batch is the
    # one place that has the dist, the value and the sampled action together;
    # reconstructing any of them afterwards would risk describing a decision
    # that was not the one taken.
    # ``LanePolicyActor.act_batch`` stashes last_dist / last_values /
    # last_actions / last_state_in for exactly this kind of reader, so there is
    # nothing to intercept -- but the stash is overwritten on every call, and
    # VecDriver calls it once per POLICY per step (both sides share one policy
    # here, so once per step covering both slots). Read it immediately after
    # driver.step() and index by the slot order the driver used.
    def head_probs() -> dict:
        d = actor.last_dist
        if d is None or not hasattr(d, "logits"):
            return {}
        out = {}
        for k, t in d.logits.items():
            arr = t.detach().float().cpu().numpy()
            # (N, T, K) -> (N, K); T is 1 on the acting path.
            arr = arr[:, 0, :] if arr.ndim == 3 else arr
            out[k] = [[round(float(p), 5) for p in _softmax(row)] for row in arr]
        return out

    # And the orders, so the record says what actually went on the wire.
    real_encode = adapters.encoder.encode
    orders: dict = {}

    def spy_encode(action, raw, side):
        order = real_encode(action, raw, side)
        orders[side] = order
        return order

    adapters.encoder.encode = spy_encode

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    driver.start()
    t0 = time.time()
    with out_path.open("w") as fh:
        try:
            while time.time() - t0 < args.seconds:
                orders.clear()
                driver.step(deterministic=False)
                raw = env.last_obs[0]
                if raw is None:
                    continue
                probs = head_probs()
                values = (actor.last_values.detach().float().cpu().numpy().tolist()
                          if actor.last_values is not None else [])
                # The driver's slot order for this policy IS the batch order the
                # probabilities came back in. Taking it from the driver rather
                # than assuming (blue, red) means a single-sided assignment
                # cannot silently shift every row by one.
                slot_order = [s for k, sl in driver.slots.items() for s in sl]
                row_of = {side: i for i, (_inst, side) in enumerate(slot_order)}

                rec = {"t": raw.get("t"), "sides": {}}
                for side in (BLUE, RED):
                    ad = driver.adapters.get((0, side))
                    if ad is None or ad.last_frame is None:
                        continue
                    me = ad.last_frame.champion_of_team(ad.team)
                    if me is None:
                        continue
                    cam = projection.centred_on(me.x, me.y)
                    slots = []
                    for i, netid in enumerate(ad.last_slot_netids):
                        if netid is None or not bool(ad.last_slot_valid[i]):
                            continue
                        u = ad.last_frame.units.get(netid)
                        if u is None:
                            continue
                        sx, sy = projection.world_to_screen(cam, u.x, u.y)
                        slots.append({
                            "slot": i, "netid": int(netid), "etype": u.etype,
                            "team": int(u.team),
                            "hp_frac": 0.0 if u.mhp <= 0 else round(u.hp / u.mhp, 4),
                            "world": [round(u.x, 1), round(u.y, 1)],
                            "screen": [round(sx, 5), round(sy, 5)],
                        })
                    r = row_of.get(side)
                    heads = ({k: v[r] for k, v in probs.items() if r is not None and r < len(v)}
                             if r is not None else {})
                    rec["sides"][side] = {
                        "champ_world": [round(me.x, 1), round(me.y, 1)],
                        "champ_hp_frac": 0.0 if me.mhp <= 0 else round(me.hp / me.mhp, 4),
                        "camera": {"cx": round(cam.cx, 1), "cz": round(cam.cz, 1)},
                        "slots": slots,
                        "order": orders.get(side),
                        "heads": heads,
                        "value": (round(float(values[r]), 4)
                                  if r is not None and r < len(values) else None),
                    }
                fh.write(json.dumps(rec) + "\n")
                n += 1
                if n % 300 == 0:
                    fh.flush()
                    log.info("captured %d decisions (t=%s ms)", n, rec["t"])
        finally:
            env.close()
    log.info("wrote %d decisions to %s", n, out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
