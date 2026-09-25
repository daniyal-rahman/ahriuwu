#!/usr/bin/env python3
"""LANERL_TOPONLY=1 -> spawn only top-lane waves, skip jungle camps.

Throughput decays with entity count (49x empty -> 14x with all three lanes +
jungle). A top-lane 1v1 never touches mid/bot waves or camps, so they are pure
wasted CPU. LANE_L is top (waypoints run up the left side then across the top).
"""
import pathlib
# The NFS export is mounted at /srv/nfs on danilogin and /mnt/nfs on desktop.
# Never hardcode either: resolve everything from this file's own location.
_VENDOR = pathlib.Path(__file__).resolve().parents[2] / "lanerl-vendor"
p = _VENDOR / "LoLServer/Content/LeagueSandbox-Scripts/Maps/Map1/LevelScript.cs"
s = p.read_text()
if "LANERL_TOPONLY" in s:
    print("already patched"); raise SystemExit

# skip non-top lanes in the wave spawn loop
old = """                Lane lane = barrack.Value.GetSpawnBarrackLaneID();"""
new = """                Lane lane = barrack.Value.GetSpawnBarrackLaneID();
                // top-lane-only training mode: LANE_L is top
                if (System.Environment.GetEnvironmentVariable("LANERL_TOPONLY") == "1"
                    && lane != Lane.LANE_L) continue;"""
assert old in s, "lane line not found"
s = s.replace(old, new, 1)

# skip jungle camps entirely
old2 = """            NeutralMinionSpawn.InitializeCamps();"""
new2 = """            if (System.Environment.GetEnvironmentVariable("LANERL_TOPONLY") != "1")
            {
                NeutralMinionSpawn.InitializeCamps();
            }"""
assert old2 in s, "InitializeCamps not found"
s = s.replace(old2, new2, 1)

p.write_text(s)
print("patched LevelScript.cs: top-lane-only waves + no jungle")
