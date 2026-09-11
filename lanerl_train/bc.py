#!/usr/bin/env python
"""Behaviour cloning: supervised pretraining from the scripted bot.

Why: PPO from random init provably cannot solve this task. Travel per decision
is 345 u/s / 30 Hz = 11.5 units, so reaching lane (11,866 units) by random walk
needs ~1.06M steps -- 9.9 hours of game against a 10-minute episode. The first
run spent 13,475 updates without the agent ever seeing a minion, which is why
entropy never fell and value_loss collapsed to a constant.

This is the AlphaStar shape: a supervised prior from demonstrations, then RL
with a KL penalty toward it. This module is the prior.

The head structure mirrors the policy's action space exactly (button / move_x /
move_z / target), so the trained weights load straight into LanePolicy and RL
can continue from them.

  python -m lanerl_train.bc --demos demos/bot_demos.npz --out demos/bc_policy.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))


def labels_to_indices(label_json: np.ndarray, obs_n: int) -> Dict[str, np.ndarray]:
    """Map the bot's semantic orders onto the policy's discrete heads.

    The mapping must match ``lanerl_rl.constants.BUTTONS`` and the move-bin
    layout, or BC teaches the network a different action space from the one RL
    then uses -- which would look like a working prior that instantly collapses
    once PPO takes over.
    """
    from lanerl_rl import constants as C

    buttons: List[str] = list(C.BUTTONS)
    idx_noop = buttons.index("noop")
    idx_move = buttons.index("move")
    idx_attack = buttons.index("attack_move")

    n = len(label_json)
    out = {
        "button": np.full(n, idx_noop, dtype=np.int64),
        "move_x": np.full(n, C.N_MOVE_BINS // 2, dtype=np.int64),
        "move_z": np.full(n, C.N_MOVE_BINS // 2, dtype=np.int64),
        "target": np.zeros(n, dtype=np.int64),
    }
    for i, raw in enumerate(label_json):
        d = json.loads(str(raw))
        kind = d.get("t", "noop")
        if kind == "move":
            out["button"][i] = idx_move
            # The demo carries an absolute goal; the policy emits a direction
            # bin. Without the champion's own position we cannot recover the
            # direction, so movement DIRECTION is not cloned here -- only the
            # decision to move. See the note in main().
        elif kind == "attack":
            out["button"][i] = idx_attack
        elif kind == "cast":
            slot = int(d.get("slot", 0))
            name = {0: "q", 1: "w", 2: "e", 3: "r"}.get(slot)
            if name in buttons:
                out["button"][i] = buttons.index(name)
        elif kind == "recall" and "recall" in buttons:
            out["button"][i] = buttons.index("recall")
    return out


def main() -> int:
    import torch
    import torch.nn.functional as F

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--demos", required=True)
    ap.add_argument("--out", default="demos/bc_policy.pt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from lanerl_rl.model import LanePolicy, ModelConfig

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    z = np.load(args.demos, allow_pickle=False)
    ent = torch.as_tensor(z["entities"], dtype=torch.float32)
    slf = torch.as_tensor(z["self_vec"], dtype=torch.float32)
    glb = torch.as_tensor(z["global_vec"], dtype=torch.float32)
    heads = labels_to_indices(z["label_json"], len(ent))
    n = len(ent)
    print(f"dataset: n={n}  entities={tuple(ent.shape)}  self={tuple(slf.shape)}")
    counts = np.bincount(heads["button"], minlength=16)
    from lanerl_rl import constants as C
    print("button distribution:",
          {b: int(c) for b, c in zip(C.BUTTONS, counts) if c})

    # A held-out split, because training accuracy on an imbalanced set where
    # one class is 80% of the data is not evidence of anything.
    g = np.random.default_rng(args.seed)
    perm = g.permutation(n)
    n_val = max(1, int(n * args.val_frac))
    val_i, tr_i = perm[:n_val], perm[n_val:]
    print(f"split: train={len(tr_i)}  val={len(val_i)}")

    policy = LanePolicy(ModelConfig())
    opt = torch.optim.Adam(policy.parameters(), lr=args.lr)

    def batch_logits(idx):
        kw = {k: torch.as_tensor(z[k][idx]).unsqueeze(1) for k in
              ("entities", "self_vec", "global_vec", "priv_entities", "priv_vec")}
        for k in ("entity_pad_mask", "priv_pad_mask"):
            kw[k] = torch.as_tensor(z[k][idx]).unsqueeze(1)
        masks = {k: torch.as_tensor(z[f"mask_{k}"][idx]).unsqueeze(1)
                 for k in ("button", "move_x", "move_z", "target")}
        state = policy.initial_state(len(idx), device="cpu")
        dist, _value, _state = policy(state=state, action_masks=masks, **kw)
        return dist

    majority = counts.max() / max(1, counts.sum())
    print(f"majority-class baseline (button): {majority:.3f} -- BC must beat this")

    for epoch in range(args.epochs):
        policy.train()
        g.shuffle(tr_i)
        tot, seen = 0.0, 0
        for s in range(0, len(tr_i), args.batch):
            idx = tr_i[s : s + args.batch]
            dist = batch_logits(idx)
            loss = F.cross_entropy(
                dist.button.logits.reshape(len(idx), -1),
                torch.as_tensor(heads["button"][idx]),
            )
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx); seen += len(idx)
        policy.eval()
        with torch.no_grad():
            dist = batch_logits(val_i)
            pred = dist.button.logits.reshape(len(val_i), -1).argmax(-1).numpy()
            acc = float((pred == heads["button"][val_i]).mean())
        print(f"  epoch {epoch}: train_loss={tot/max(1,seen):.4f}  val_button_acc={acc:.3f}")

    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"policy": policy.state_dict(), "kind": "bc"}, outp)
    print(f"wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
