#!/usr/bin/env python3
"""Decompose the BC movement target into REAL player commands vs camera drift.

pipeline.py builds label.cursor.screen = project(cursor_world, camera_at_this_frame)
where cursor_world is either the current cast_target or the last movement click
(held forward). Because the camera follows the champion, a HELD cursor_world
re-projects to a MOVING screen point every frame. replay_dataset._parse_movement
then treats any screen move > 1% as a new command.

This script measures how much of the movement training signal is real vs drift,
and how much survives the 21-bin quantization.
"""
import json
from collections import Counter
from pathlib import Path

import numpy as np

CACHE = Path("/srv/nfs/projects/ahriuwu/scratchpad/audit_cache")
BINS = 21
DEADBAND = 0.01


def main():
    tot = Counter()
    click_rate = []
    drift_mag = []
    real_mag = []
    for p in sorted(CACHE.glob("*.npz")):
        z = np.load(p, allow_pickle=True)
        m = json.loads(str(z["meta"][0]))
        sw, sh = m["screen_resolution"]
        fps = m["fps"]
        gt = z["gt"]
        T = m["T"]
        sx, sy = z["cur_sx"] / sw, z["cur_sy"] / sh
        wx, wy = z["cur_wx"], z["cur_wy"]
        have = np.isfinite(sx) & np.isfinite(sy)

        # world unchanged frame-to-frame => no new command was issued
        w_same = np.zeros(T, dtype=bool)
        w_same[1:] = (wx[1:] == wx[:-1]) & (wy[1:] == wy[:-1]) & np.isfinite(wx[1:])

        # screen delta
        dsx = np.full(T, np.nan)
        dsy = np.full(T, np.nan)
        dsx[1:] = sx[1:] - sx[:-1]
        dsy[1:] = sy[1:] - sy[:-1]
        dmax = np.maximum(np.abs(dsx), np.abs(dsy))
        ok = np.isfinite(dmax)

        # raw bin index straight off cursor.screen (no dead-band) — how the
        # target moves before/after the denoise hack
        idx = np.full((T, 2), -1, dtype=int)
        idx[have, 0] = np.clip(np.round(np.clip(sx[have], 0, 1) * (BINS - 1)), 0, BINS - 1)
        idx[have, 1] = np.clip(np.round(np.clip(sy[have], 0, 1) * (BINS - 1)), 0, BINS - 1)
        bin_change = np.zeros(T, dtype=bool)
        bin_change[1:] = ((idx[1:] != idx[:-1]).any(1)) & (idx[1:, 0] >= 0) & (idx[:-1, 0] >= 0)

        tot["frames"] += T
        tot["cursor_present"] += int(have.sum())
        tot["cursor_missing"] += int((~have).sum())
        # missing DESPITE a valid cursor_world -> off-screen projection dropped
        tot["missing_but_world_ok"] += int((~have & np.isfinite(wx)).sum())

        h = ok & w_same                     # HOLD frames (no new command)
        c = ok & ~w_same & np.isfinite(wx)  # world moved (new click OR moving cast target)
        tot["hold_frames"] += int(h.sum())
        tot["cmd_frames"] += int(c.sum())
        tot["hold_over_deadband"] += int((h & (dmax > DEADBAND)).sum())
        tot["hold_bin_change"] += int((h & bin_change).sum())
        tot["cmd_over_deadband"] += int((c & (dmax > DEADBAND)).sum())
        tot["cmd_bin_change"] += int((c & bin_change).sum())
        drift_mag.append(dmax[h])
        real_mag.append(dmax[c])

        # true player command events from clicks.json casts (clicks list is not
        # cached, so use casts + world-jump detection)
        n_cast = int(np.isfinite(z["casts_t"]).sum())
        tot["casts"] += n_cast
        click_rate.append((m["match_id"], n_cast, m.get("n_click_events", 0),
                           float(gt[-1] - gt[0])))

    print(f"frames                                : {tot['frames']:,}")
    print(f"cursor.screen present                 : {tot['cursor_present']:,} "
          f"({100*tot['cursor_present']/tot['frames']:.2f}%)")
    print(f"cursor.screen None                    : {tot['cursor_missing']:,} "
          f"({100*tot['cursor_missing']/tot['frames']:.2f}%)")
    print(f"  ...of which cursor.world WAS known  : {tot['missing_but_world_ok']:,} "
          f"({100*tot['missing_but_world_ok']/max(tot['cursor_missing'],1):.1f}% of the Nones) "
          f"= command target projected OFF-SCREEN -> silently dropped, target held")
    print()
    print(f"HOLD frames (cursor_world unchanged, i.e. NO new command): {tot['hold_frames']:,}")
    print(f"  screen moved > 1% dead-band anyway  : {tot['hold_over_deadband']:,} "
          f"({100*tot['hold_over_deadband']/max(tot['hold_frames'],1):.2f}%)  <-- FABRICATED commands")
    print(f"  changed 21-bin cell anyway          : {tot['hold_bin_change']:,} "
          f"({100*tot['hold_bin_change']/max(tot['hold_frames'],1):.2f}%)")
    print()
    print(f"COMMAND frames (cursor_world changed) : {tot['cmd_frames']:,}")
    print(f"  survived the 1% dead-band           : {tot['cmd_over_deadband']:,} "
          f"({100*tot['cmd_over_deadband']/max(tot['cmd_frames'],1):.2f}%)")
    print(f"  changed 21-bin cell                 : {tot['cmd_bin_change']:,} "
          f"({100*tot['cmd_bin_change']/max(tot['cmd_frames'],1):.2f}%)  "
          f"=> {100-100*tot['cmd_bin_change']/max(tot['cmd_frames'],1):.2f}% of real target "
          f"changes are QUANTIZED AWAY")
    dm = np.concatenate(drift_mag)
    dm = dm[np.isfinite(dm)]
    rm = np.concatenate(real_mag)
    rm = rm[np.isfinite(rm)]
    for nm, a in (("drift (hold)", dm), ("real (cmd)", rm)):
        if len(a) == 0:
            continue
        print(f"|delta cursor.screen| {nm:14s}: p50 {np.percentile(a,50)*100:.3f}% "
              f"p90 {np.percentile(a,90)*100:.3f}% p99 {np.percentile(a,99)*100:.3f}% "
              f"max {a.max()*100:.1f}% of screen")
    # signal-to-noise of the target
    n_true = tot["cmd_bin_change"]
    n_fake = tot["hold_bin_change"]
    print()
    print(f"BIN-TRANSITION LABELS: {n_true:,} from real command changes, "
          f"{n_fake:,} from pure camera drift "
          f"=> {100*n_fake/max(n_true+n_fake,1):.1f}% of movement transitions are NOISE")
    print()
    tot_clicks = sum(t[2] for t in click_rate)
    tot_dur = sum(t[3] for t in click_rate)
    print(f"clicks.json: {tot_clicks:,} movement clicks over {tot_dur/60:.0f} min "
          f"= {tot_clicks/tot_dur:.2f} clicks/s  ({tot_clicks/len(click_rate):.0f}/game)")
    print(f"  vs {tot['frames']} frames at 20fps -> a real command every "
          f"{tot['frames']/max(tot_clicks,1):.1f} frames")
    print(f"casts: {tot['casts']:,} = {tot['casts']/tot_dur:.3f}/s")


if __name__ == "__main__":
    main()
