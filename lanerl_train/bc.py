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
        "screen_x": np.full(n, C.N_SCREEN_X // 2, dtype=np.int64),
        "screen_y": np.full(n, C.N_SCREEN_Y // 2, dtype=np.int64),
        "target": np.zeros(n, dtype=np.int64),
        # Which heads this row actually SUPERVISES. A head with no label is not
        # the same as a head labelled "centre bin" / "slot 0": move_x/move_z sit
        # at N_MOVE_BINS // 2, whose value is MOVE_BIN_VALUES[4] = 0.0, i.e. a
        # positive "stand still" instruction. 21.3% of move rows carry no
        # direction (the tail of an order republished across ~4 frames at 30 Hz
        # against the bot's 150 ms reaction clock), and training them as
        # stand-still inflated the centre bin in the walk-to-lane zone from 2.4%
        # to 19.6% -- the same learned-to-stand-still end state as the
        # transform-inversion bug, by a different route.
        "has_dir": np.zeros(n, dtype=bool),
        "has_target": np.zeros(n, dtype=bool),
    }
    for i, raw in enumerate(label_json):
        d = json.loads(str(raw))
        kind = d.get("t", "noop")
        if kind == "move":
            out["button"][i] = idx_move
        elif kind == "attack":
            out["button"][i] = idx_attack
        elif kind == "cast":
            slot = int(d.get("slot", 0))
            name = {0: "q", 1: "w", 2: "e", 3: "r"}.get(slot)
            if name in buttons:
                out["button"][i] = buttons.index(name)
        elif kind == "recall" and "recall" in buttons:
            out["button"][i] = buttons.index("recall")
        # Direction, recovered by the collector by inverting decode_action.
        # Cloning WHERE the bot walks is the whole point: it is the behaviour
        # PPO could never discover, since a random walk needs ~9.9 hours of
        # game time to cross the map.
        if d.get("mx") is not None:
            out["screen_x"][i] = int(d["mx"])
            out["screen_y"][i] = int(d["mz"])
            out["has_dir"][i] = True
        # The attack referent -- the whole content of a last hit. Without it BC
        # clones the DECISION to attack and leaves the target head at random
        # init, which put ~35% of the policy's attack mass on its own minions.
        if d.get("slot_idx") is not None:
            out["target"][i] = int(d["slot_idx"])
            out["has_target"][i] = True
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
    print(f"supervised heads: direction {int(heads['has_dir'].sum())} rows, "
          f"target {int(heads['has_target'].sum())} rows")
    n_atk = int((heads["button"] == list(C.BUTTONS).index("attack_move")).sum())
    if n_atk and heads["has_target"].sum() == 0:
        print("NO ATTACK ROW CARRIES A TARGET SLOT -- the target head would train "
              "on all-zeros and the policy would attack an arbitrary unit. "
              "Refusing: re-collect demos with the slot_idx label.")
        return 1

    # Drop rows whose label the policy structurally cannot emit. Cross-entropy
    # against a masked (-1e9) logit is ~1e9, so a handful of them owns the whole
    # gradient: 787 of 99,655 rows drove a mean loss of 7.7 MILLION while the
    # real signal was ~2. The collector's off-by-one is fixed, but a residual
    # here must fail loudly rather than quietly dominate training again.
    mb = z["mask_button"]
    ok = mb[np.arange(n), heads["button"]].astype(bool)
    if not ok.all():
        frac = 1.0 - ok.mean()
        print(f"dropping {int((~ok).sum())} rows ({frac:.3%}) whose label is masked out")
        if frac > 0.05:
            print("MORE THAN 5% CONTRADICTORY -- the label/observation pairing is wrong, "
                  "not just noisy. Refusing to train on it.")
            return 1
        keep = np.flatnonzero(ok)
        for k in heads:
            heads[k] = heads[k][keep]
        z = {k: (z[k][keep] if getattr(z[k], "ndim", 0) and len(z[k]) == n else z[k])
             for k in z.files}
        n = len(keep)
        counts = np.bincount(heads["button"], minlength=16)

    # Gate EVERY supervised head by its OWN mask, not just the button.
    #
    # The first version of this guard checked only mask_button, and the run it
    # gated still reported train_loss = 49,946,203 falling to 57,408 -- the same
    # masked-logit catastrophe as before (cross-entropy against a -1e9 logit is
    # ~1e9), just relocated to the target head. It showed up as val_target
    # DEGRADING across epochs (0.922 -> 0.817) while val_button sat at 0.547
    # against a 0.544 majority baseline and val_move_x never moved off 0.230:
    # a handful of impossible rows owned the whole gradient.
    #
    # The cause is the same off-by-one the label pairing has to live with. The
    # bot commits an attack at frame T; we pair it with frame T-1's observation
    # and T-1's slot map. A minion that was out of range at T-1 is a legal
    # target at T and a masked one at T-1. That row is not noise to be trained
    # through -- it is unlabelable for that head, and only for that head, so
    # clear the head's gate and keep the row's button label.
    for head, gate in (("target", "has_target"),
                       ("move_x", "has_dir"), ("move_z", "has_dir")):
        m = z[f"mask_{head}"]
        allowed = m[np.arange(n), heads[head]].astype(bool)
        killed = int((heads[gate] & ~allowed).sum())
        if killed:
            print(f"  un-supervising {killed} rows on {head} "
                  f"({killed / max(1, int(heads[gate].sum())):.2%} of that head's "
                  f"labels) -- the label is masked in the paired observation")
        heads[gate] = heads[gate] & allowed
    print(f"after gating: direction {int(heads['has_dir'].sum())} rows, "
          f"target {int(heads['has_target'].sum())} rows")

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
        # The value output is DISCARDED, and has to be: a demonstration set is
        # (observation, action) pairs with no reward and no returns, so there
        # is no target to regress a value function against. The consequence --
        # 2.6M of the checkpoint's 4.6M parameters are random initialisation --
        # is recorded in the checkpoint itself; see the end of main().
        dist, _value, _state = policy(state=state, action_masks=masks, **kw)
        return dist

    idx_move_g = list(C.BUTTONS).index("move")
    majority = counts.max() / max(1, counts.sum())
    print(f"majority-class baseline (button): {majority:.3f} -- BC must beat this")

    for epoch in range(args.epochs):
        policy.train()
        g.shuffle(tr_i)
        tot, seen = 0.0, 0
        for s in range(0, len(tr_i), args.batch):
            idx = tr_i[s : s + args.batch]
            dist = batch_logits(idx)
            # Weight each head by whether THIS row supervises it. Previously
            # move_x/move_z were weighted by (button == move), which is not the
            # same question: a move order whose direction the collector could
            # not recover still counted, at full strength, as a label saying
            # "centre bin" = stand still.
            loss = F.cross_entropy(
                dist.logits["button"].reshape(len(idx), -1),
                torch.as_tensor(heads["button"][idx]),
            )
            per_head = {"move_x": "has_dir", "move_z": "has_dir",
                        "target": "has_target"}
            for head, gate in per_head.items():
                w = torch.as_tensor(heads[gate][idx].astype("float32"))
                if float(w.sum()) == 0.0:
                    continue
                ce = F.cross_entropy(
                    dist.logits[head].reshape(len(idx), -1),
                    torch.as_tensor(heads[head][idx]),
                    reduction="none",
                )
                loss = loss + (ce * w).sum() / w.sum().clamp(min=1.0)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx); seen += len(idx)
        policy.eval()
        with torch.no_grad():
            dist = batch_logits(val_i)
            pred = dist.logits["button"].reshape(len(val_i), -1).argmax(-1).numpy()
            acc = float((pred == heads["button"][val_i]).mean())
            # The target head is the one that decides a last hit, so report it
            # separately -- button accuracy stayed high while target was pure
            # noise, which is how CS 0 survived a "converged" BC run.
            tsel = val_i[heads["has_target"][val_i]]
            tacc = float("nan")
            if len(tsel):
                tp = dist.logits["target"].reshape(len(val_i), -1).argmax(-1).numpy()
                tacc = float((tp[heads["has_target"][val_i]] == heads["target"][tsel]).mean())
            dsel = val_i[heads["has_dir"][val_i]]
            dacc = float("nan")
            if len(dsel):
                dp = dist.logits["move_x"].reshape(len(val_i), -1).argmax(-1).numpy()
                dacc = float((dp[heads["has_dir"][val_i]] == heads["move_x"][dsel]).mean())
        mean_loss = tot / max(1, seen)
        if epoch == 0 and mean_loss > 1e3:
            print(f"EPOCH-0 LOSS IS {mean_loss:.0f}. A correctly gated BC loss here "
                  f"is single digits; a value this size means cross-entropy is "
                  f"being taken against a masked (-1e9) logit, so a few "
                  f"impossible rows own the entire gradient. Refusing to write a "
                  f"checkpoint that would look trained and be noise.")
            return 1
        print(f"  epoch {epoch}: train_loss={tot/max(1,seen):.4f}  "
              f"val_button={acc:.3f}  val_target={tacc:.3f}  val_move_x={dacc:.3f}")

    # HALF OF THIS CHECKPOINT IS UNTRAINED, and it does not look it.
    #
    # The whole state_dict is saved, critic included, because
    # `lanerl_train.__main__ --init-from` loads it with strict=False and then
    # REFUSES any missing key -- dropping the critic tensors would turn a BC
    # checkpoint into a startup error. But nothing in this module ever put a
    # gradient through the critic, so what gets written for it is exactly what
    # LanePolicy.__init__ produced: random.
    #
    # That is the state PPO then has to fix, at whatever learning rate it was
    # given for FINE-TUNING the actor. On rl-bc4-0912 that was 1e-5 for both,
    # and the critic never caught up (loss/value_loss 0.047 -> 0.55, max 34.3
    # over 2,690 updates). PPOConfig.critic_lr and .critic_warmup_updates
    # exist for this; the flag below is so nothing downstream has to guess.
    n_crit = sum(p.numel() for n, p in policy.named_parameters() if n.startswith("critic."))
    n_all = sum(p.numel() for p in policy.parameters())
    outp = Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"policy": policy.state_dict(), "kind": "bc", "critic_trained": False,
         "untrained_params": int(n_crit)},
        outp,
    )
    print(f"wrote {outp}")
    print(f"NOTE: the critic in this checkpoint is UNTRAINED -- {n_crit:,} of "
          f"{n_all:,} parameters ({n_crit / n_all:.0%}) are random init. "
          f"A demonstration set has no returns to regress a value function "
          f"against, so PPO has to learn it from scratch: give the critic its "
          f"own learning rate (PPOConfig.critic_lr, default 3e-4) rather than "
          f"the actor's fine-tuning rate, and watch loss/explained_variance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
