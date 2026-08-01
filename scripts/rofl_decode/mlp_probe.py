#!/usr/bin/env python3
"""Train a small MLP per (PID, entity) to predict each hero's HP from the raw
payload bytes. If a deterministic cipher exists — even byte-position-mixed
LUT + arithmetic à la the maknee blog — an MLP with one or two hidden layers
should learn it from ~1000 labeled pairs.

If the model achieves R² > 0.5 on a held-out 20% for some (PID, entity, hero),
that triple is the decoder for that hero's HP. If nothing exceeds chance,
labeled-pairs is dead for HP and we should fall back to the live pipeline."""
from __future__ import annotations

import argparse
import json
import struct
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor

from parse_rofl import parse_rofl


HERO_ENTS = list(range(0x400000AE, 0x400000B8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rofl",
                    default="/mnt/nfs/datasets/lol_replays_16_9_772/NA1_5552884026/rofl/replay.rofl")
    ap.add_argument("--raw-mem",
                    default="/mnt/nfs/datasets/lol_replays_16_9_772/NA1_5552884026/raw_mem.json")
    ap.add_argument("--pids", nargs="+", type=int,
                    default=[132, 758, 89, 828, 104, 985, 652, 876, 964, 1112, 559])
    ap.add_argument("--min-blocks", type=int, default=80)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw_mem = json.load(open(args.raw_mem))
    game_length = raw_mem[-1]["gt"]
    blocks = parse_rofl(args.rofl)
    n_frames = max(b.frame_idx for b in blocks) + 1
    frame_dur = game_length / n_frames

    fc = Counter(b.frame_idx for b in blocks)
    seen: dict[int, int] = defaultdict(int)
    block_gts = np.empty(len(blocks), dtype=np.float64)
    for i, b in enumerate(blocks):
        fi = b.frame_idx
        idx = seen[fi]
        seen[fi] = idx + 1
        block_gts[i] = fi * frame_dur + (idx + 0.5) / fc[fi] * frame_dur

    hero_hp = {}
    for hero in raw_mem[0]["heroes"]:
        s = sorted((t["gt"], t["heroes"][hero]["hp"]) for t in raw_mem if hero in t["heroes"])
        hero_hp[hero] = (
            np.array([p[0] for p in s], dtype=np.float64),
            np.array([p[1] for p in s], dtype=np.float64),
        )

    print(f"Trying {len(args.pids)} PIDs × {len(HERO_ENTS)} entities × {len(hero_hp)} heroes")
    print(f"MLP: 1 hidden layer of {args.hidden}, max {args.epochs} epochs, sklearn defaults\n")

    results = []
    for pid in args.pids:
        # Group by entity, pick modal payload size
        ent_blks: dict[int, list] = defaultdict(list)
        for i, b in enumerate(blocks):
            if b.pid == pid and b.param in HERO_ENTS:
                ent_blks[b.param].append((i, b))
        for ent, lst in ent_blks.items():
            if len(lst) < args.min_blocks:
                continue
            sizes = Counter(len(b.payload) for _, b in lst)
            modal_sz, modal_count = sizes.most_common(1)[0]
            if modal_count < args.min_blocks:
                continue
            same = [(i, b) for i, b in lst if len(b.payload) == modal_sz]

            # Build feature matrix: payload bytes / 255.0, plus block_gt as last feature
            X = np.zeros((len(same), modal_sz), dtype=np.float32)
            gts = np.zeros(len(same), dtype=np.float64)
            for j, (i, b) in enumerate(same):
                X[j] = np.frombuffer(b.payload, dtype=np.uint8) / 255.0
                gts[j] = block_gts[i]

            for hero, (gt_o, hp_o) in hero_hp.items():
                y = np.interp(gts, gt_o, hp_o).astype(np.float32)
                if y.std() < 1e-3:
                    continue
                # 80/20 random split
                X_tr, X_te, y_tr, y_te = train_test_split(
                    X, y, test_size=0.2, random_state=args.seed
                )
                mlp = MLPRegressor(
                    hidden_layer_sizes=(args.hidden,),
                    max_iter=args.epochs,
                    random_state=args.seed,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                )
                try:
                    mlp.fit(X_tr, y_tr)
                except Exception:
                    continue
                pred = mlp.predict(X_te)
                r2 = float(r2_score(y_te, pred))
                # also: fraction of test predictions within 5 HP
                acc5 = float(np.mean(np.abs(pred - y_te) <= 5.0))
                acc25 = float(np.mean(np.abs(pred - y_te) <= 25.0))
                if r2 > 0.0:
                    results.append({
                        "pid": pid, "ent": ent, "modal_sz": modal_sz,
                        "hero": hero, "n": len(same),
                        "r2": r2, "acc5": acc5, "acc25": acc25,
                    })

    results.sort(key=lambda r: -r["r2"])
    print(f"{'pid':>5} {'entity':>10} {'sz':>3} {'hero':>14} {'n':>5} "
          f"{'R²':>6} {'≤5HP':>5} {'≤25HP':>6}")
    print("-" * 70)
    for r in results[:60]:
        print(
            f"{r['pid']:5d} 0x{r['ent']:08x} {r['modal_sz']:3d} {r['hero']:>14} "
            f"{r['n']:5d} {r['r2']:6.3f} {r['acc5']:5.2f} {r['acc25']:6.2f}"
        )

    # Best (entity, hero) per PID
    print(f"\n{'='*70}\nBEST per PID")
    print(f"{'='*70}")
    by_pid: dict[int, dict] = {}
    for r in results:
        cur = by_pid.get(r["pid"])
        if cur is None or r["r2"] > cur["r2"]:
            by_pid[r["pid"]] = r
    for pid in sorted(by_pid.keys(), key=lambda p: -by_pid[p]["r2"]):
        r = by_pid[pid]
        print(
            f"  pid={r['pid']:4d}  ent=0x{r['ent']:08x}  hero={r['hero']:>14}  "
            f"R²={r['r2']:.3f}  ≤5HP={r['acc5']:.2f}  ≤25HP={r['acc25']:.2f}  n={r['n']}"
        )


if __name__ == "__main__":
    main()
