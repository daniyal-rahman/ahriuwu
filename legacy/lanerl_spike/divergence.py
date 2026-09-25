#!/usr/bin/env python
"""Where does the RL policy most disagree with the scripted bot, and why?

"CS went down" says nothing about WHICH decision got worse. The demo set
carries 66k real observations with the bot's own chosen action attached, so
both policies can be scored on the SAME states offline -- no server, no
sampling noise, no confound from the two policies visiting different states.

For every frame we compute the surprise each policy assigns to the bot's
action, ``-log p(a_bot | s)``. Comparing RL's surprise against BC's on the
same frame isolates what RL CHANGED, rather than what BC never knew:

    delta = -log p_RL(a_bot) + log p_BC(a_bot)

Large positive delta = RL moved probability away from what the bot does
here. Those frames, bucketed by what the bot was doing and by what RL wants
instead, are the concrete answer to "which decision degraded".

    python lanerl/divergence.py <rl_checkpoint.pt> [--bc demos/bc_policy_nosort.pt]
                               [--demos demos/bot_demos_nosort.npz] [--n 8192]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lanerl_rl import constants as C
from lanerl_rl.infer import _BOOL_KEYS, _FLOAT_KEYS, _MASK_KEYS
from lanerl_rl.model import LanePolicy, ModelConfig


def load(path: str) -> LanePolicy:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    for key in ("model", "policy", "state_dict"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]
            break
    m = LanePolicy(ModelConfig())
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m


def surprise(model: LanePolicy, kw, masks, labels, n: int):
    """-log p(a_bot) per frame, per head, plus the policy's own argmax."""
    with torch.no_grad():
        dist, value, _ = model(state=model.initial_state(n), resets=None,
                               action_masks=masks, **kw)
    out, argmax = {}, {}
    for head, lab in labels.items():
        logits = dist.logits[head].reshape(n, -1).float()
        logp = torch.log_softmax(logits, dim=-1)
        valid = lab >= 0
        s = torch.full((n,), float("nan"))
        s[valid] = -logp[valid].gather(1, lab[valid, None]).squeeze(1)
        out[head] = s.numpy()
        argmax[head] = logits.argmax(-1).numpy()
    return out, argmax, value.reshape(n).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("rl")
    ap.add_argument("--bc", default="demos/bc_policy_nosort.pt")
    ap.add_argument("--demos", default="demos/bot_demos_nosort.npz")
    ap.add_argument("--n", type=int, default=8192)
    ap.add_argument("--top", type=int, default=400)
    args = ap.parse_args()

    z = np.load(args.demos, allow_pickle=True)
    n = min(args.n, len(z["label_json"]))
    kw = {k: torch.as_tensor(z[k][:n]).unsqueeze(1)
          for k in list(_FLOAT_KEYS) + list(_BOOL_KEYS)}
    masks = {k: torch.as_tensor(z["mask_" + k][:n]).unsqueeze(1) for k in _MASK_KEYS}
    labs = [json.loads(s) for s in z["label_json"][:n]]

    # the bot's action, per head; -1 where that head is unsupervised
    button = np.full(n, -1, dtype=np.int64)
    target = np.full(n, -1, dtype=np.int64)
    for i, d in enumerate(labs):
        t = d.get("t")
        name = {"noop": "noop", "move": "move", "attack": "attack_move",
                "recall": "recall", "cast": None}.get(t)
        if t == "cast" and d.get("slot") is not None:
            name = ("q", "w", "e", "r")[int(d["slot"]) % 4]
        if name in C.BUTTON_INDEX:
            button[i] = C.BUTTON_INDEX[name]
        if d.get("slot_idx") is not None:
            target[i] = int(d["slot_idx"])
    labels = {"button": torch.as_tensor(button), "target": torch.as_tensor(target)}

    rl, bc = load(args.rl), load(args.bc)
    s_rl, am_rl, v_rl = surprise(rl, kw, masks, labels, n)
    s_bc, am_bc, v_bc = surprise(bc, kw, masks, labels, n)

    print(f"frames={n}  rl={os.path.basename(args.rl)}  bc={os.path.basename(args.bc)}")
    print(f"critic value: rl mean {np.nanmean(v_rl):+.3f}  bc mean {np.nanmean(v_bc):+.3f}\n")

    for head in ("button", "target"):
        d = s_rl[head] - s_bc[head]
        ok = ~np.isnan(d)
        print(f"=== {head}: -log p(bot action), RL minus BC ===")
        print(f"  frames scored {ok.sum()}   mean delta {np.nanmean(d):+.3f}   "
              f"RL {np.nanmean(s_rl[head]):.3f} vs BC {np.nanmean(s_bc[head]):.3f}")
        if head == "button":
            # WHICH bot action did RL abandon, and what does it want instead?
            idx = np.argsort(np.where(ok, -d, -np.inf))[:args.top]
            rows = defaultdict(Counter)
            for i in idx:
                if not ok[i]:
                    continue
                rows[C.BUTTONS[button[i]]][C.BUTTONS[am_rl["button"][i]]] += 1
            print(f"  top {args.top} divergences -- bot did X, RL prefers Y:")
            for bot_act, ctr in sorted(rows.items(), key=lambda kv: -sum(kv[1].values())):
                tot = sum(ctr.values())
                pref = ", ".join(f"{k} {v}" for k, v in ctr.most_common(3))
                print(f"    bot {bot_act:12s} n={tot:4d}   RL wants: {pref}")
            # and the overall action mix each policy would take
            print("  argmax mix over ALL frames:")
            for nm, am in (("bot ", button), ("BC  ", am_bc["button"]),
                           ("RL  ", am_rl["button"])):
                c = Counter(C.BUTTONS[a] for a in am if a >= 0)
                tot = sum(c.values())
                print(f"    {nm} " + "  ".join(
                    f"{k}={100*v/tot:.0f}%" for k, v in c.most_common(5)))
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
