#!/usr/bin/env python3
"""
Backfill enemy-top stats (gold, gold_total, level, world coords) into the
visible_heroes block of an existing labels.json by re-deriving from the
already-shipped raw_mem.json.

Use case: pipeline.py only emitted hp/hp_max/level for visible_heroes prior
to 2026-05-08, dropping gold_total and off-screen entries entirely. The
mem scrape itself captured everything (raw_mem.json has all 10 heroes per
50Hz tick), so we can rebuild labels.json post-hoc without re-recording.

Run on the NFS host (or anywhere the dataset is mounted):
    python backfill_visible_heroes.py --root /mnt/nfs/datasets/lol_replays_16_9_772
        [--match-ids NA1_5550867234,NA1_...]      # optional subset
        [--dry-run]

For each match dir under --root that has both labels.json and raw_mem.json:
  1. Load both
  2. For each frame, find the closest mem sample by gt
  3. Replace label.visible_heroes with all 10 heroes including gold_total + world
  4. Atomically rewrite labels.json (write .tmp + os.replace)

A `_backfilled_at` ISO timestamp is added to the labels.json root so re-runs
are detectable (and skipped unless --force).
"""
import argparse
import bisect
import datetime as dt
import json
import os
import sys
import time


def _build_mem_index(raw_mem):
    """Sort mem samples by gt, return (gts, samples) for bisect."""
    sorted_samples = sorted(
        (s for s in raw_mem if s.get("gt") is not None),
        key=lambda s: s["gt"],
    )
    return [s["gt"] for s in sorted_samples], sorted_samples


def _nearest_mem(gts, samples, gt):
    """Closest mem sample by gt. None if mem coverage doesn't include this gt."""
    if not gts:
        return None
    i = bisect.bisect_left(gts, gt)
    if i <= 0:
        return samples[0]
    if i >= len(gts):
        return samples[-1]
    # Pick whichever neighbor is closer
    if (gt - gts[i - 1]) < (gts[i] - gt):
        return samples[i - 1]
    return samples[i]


def backfill_one(match_dir, force=False, dry_run=False):
    """Returns (status, msg). status ∈ {"updated", "skipped", "error"}."""
    labels_path = os.path.join(match_dir, "labels.json")
    raw_mem_path = os.path.join(match_dir, "raw_mem.json")
    if not os.path.exists(labels_path):
        return "skipped", "no labels.json"
    if not os.path.exists(raw_mem_path):
        return "skipped", "no raw_mem.json"

    with open(labels_path) as f:
        labels = json.load(f)
    if labels.get("_backfilled_at") and not force:
        return "skipped", "already backfilled (use --force to redo)"

    with open(raw_mem_path) as f:
        raw_mem = json.load(f)

    gts, samples = _build_mem_index(raw_mem)
    if not gts:
        return "error", "raw_mem.json has 0 samples with gt"

    n_frames = len(labels.get("frames", []))
    n_replaced = 0
    n_no_mem = 0
    for fr in labels["frames"]:
        gt = fr.get("gt")
        lab = fr.get("label")
        if gt is None or not lab:
            continue
        sample = _nearest_mem(gts, samples, gt)
        if sample is None:
            n_no_mem += 1
            continue
        heroes = sample.get("heroes") or {}
        if not heroes:
            n_no_mem += 1
            continue
        new_visible = []
        for name, hd in heroes.items():
            pos = hd.get("pos") or [0, 0]
            new_visible.append({
                "name": name,
                "screen": None,    # backfill can't reproject without cam — overlay's
                                   # marker draw will skip these; HUD fallback by-name still works.
                "world": pos,
                "hp": hd.get("hp", 0),
                "hp_max": hd.get("hp_max", 0),
                "gold": hd.get("gold", 0),
                "gold_total": hd.get("gold_total", 0),
                "level": hd.get("level", 0),
            })
        lab["visible_heroes"] = new_visible
        n_replaced += 1

    labels["_backfilled_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
    labels["_backfill_stats"] = {
        "n_frames": n_frames,
        "n_replaced": n_replaced,
        "n_no_mem_match": n_no_mem,
    }

    if dry_run:
        return "updated", f"would replace {n_replaced}/{n_frames} frames (dry-run)"

    tmp = labels_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(labels, f)
    os.replace(tmp, labels_path)
    return "updated", f"replaced {n_replaced}/{n_frames} frames ({n_no_mem} no-mem)"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", required=True,
                    help="Dataset root containing NA1_<gid>/ subdirs (e.g. "
                         "/mnt/nfs/datasets/lol_replays_16_9_772)")
    ap.add_argument("--match-ids", default=None,
                    help="Comma-separated match_ids to process (default: every dir under --root)")
    ap.add_argument("--force", action="store_true",
                    help="Re-backfill matches that were already backfilled")
    ap.add_argument("--dry-run", action="store_true",
                    help="Don't write — just report what would change")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        print(f"FATAL: --root {args.root!r} is not a directory", file=sys.stderr)
        sys.exit(1)

    if args.match_ids:
        targets = [m.strip() for m in args.match_ids.split(",") if m.strip()]
    else:
        targets = sorted(d for d in os.listdir(args.root)
                         if d.startswith("NA1_") and os.path.isdir(os.path.join(args.root, d)))

    print(f"[backfill] processing {len(targets)} match(es) under {args.root}",
          flush=True)
    counts = {"updated": 0, "skipped": 0, "error": 0}
    t0 = time.time()
    for i, mid in enumerate(targets, 1):
        match_dir = os.path.join(args.root, mid)
        status, msg = backfill_one(match_dir, force=args.force, dry_run=args.dry_run)
        counts[status] += 1
        print(f"  [{i}/{len(targets)}] {mid:18s} {status:8s} {msg}", flush=True)

    print(f"\n[backfill] done in {time.time()-t0:.0f}s — "
          f"updated={counts['updated']}  skipped={counts['skipped']}  errors={counts['error']}",
          flush=True)


if __name__ == "__main__":
    main()
