"""Analyze hero trajectory against logged right-clicks to detect inflection points.

Usage:
  python analyze_trajectory_vs_clicks.py <replay_id>

Loads:
  - Replay data from /tmp/<replay_id>.rofl (parsed via emulator_v2)
  - Calibration from C:\tmp\calibration.json (wall_time -> game_time)
  - Clicks from C:\tmp\clicks.json (wall_time -> screen position)

Outputs:
  - plots/trajectory_analysis_<replay_id>.json with inflection points
  - visual trajectory plot (if matplotlib available)
"""
import json
import sys
import math
from pathlib import Path

def load_calibration(cal_file):
    """Load calibration: wall_time -> game_time offset."""
    try:
        with open(cal_file) as f:
            cal = json.load(f)
        wall_time = cal["wall_time"]
        game_time = cal["game_time"]
        return wall_time, game_time
    except Exception as e:
        print(f"ERROR: Could not load calibration from {cal_file}: {e}")
        return None, None

def load_clicks(clicks_file):
    """Load right-click log: list of {wall_time, x, y}."""
    try:
        with open(clicks_file) as f:
            clicks = json.load(f)
        print(f"Loaded {len(clicks)} clicks from {clicks_file}")
        return clicks
    except Exception as e:
        print(f"WARNING: Could not load clicks from {clicks_file}: {e}")
        return []

def load_replay_positions(replay_id):
    """Load hero positions from the parsed replay via emulator_v2 output.

    Expects: /tmp/<replay_id>_positions.json or similar from emulator_v2.
    Format: {game_time: {hero_name: {x, y, z}}} or similar.

    For now, using a placeholder — in reality, would parse the .rofl file
    using the emulator or run aimanager_inline_probe.py on the paused replay.
    """
    # TODO: integrate with emulator_v2 output or fresh probe
    print("NOTE: Using placeholder hero positions.")
    print("      In reality, load from emulator_v2 output or parse .rofl directly.")
    return {}

def dist2d(pos_a, pos_b):
    """2D distance (x, z plane, ignoring y)."""
    if not pos_a or not pos_b:
        return None
    return math.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[2] - pos_b[2])**2)

def detect_inflections(positions_over_time, threshold=100):
    """Detect direction changes (inflection points) in trajectory.

    Args:
        positions_over_time: list of {game_time, x, y, z}
        threshold: minimum distance for a "click" (user intent change)

    Returns:
        list of {game_time, direction_change_angle, distance_traveled}
    """
    inflections = []

    for i in range(2, len(positions_over_time) - 1):
        p_prev = positions_over_time[i - 2]
        p_curr = positions_over_time[i]
        p_next = positions_over_time[i + 1]

        if not all([p_prev, p_curr, p_next]):
            continue

        # Vector from prev to curr
        v1 = (p_curr[0] - p_prev[0], p_curr[2] - p_prev[2])
        # Vector from curr to next
        v2 = (p_next[0] - p_curr[0], p_next[2] - p_curr[2])

        # Skip if either vector is too small
        len1 = math.sqrt(v1[0]**2 + v1[1]**2)
        len2 = math.sqrt(v2[0]**2 + v2[1]**2)
        if len1 < 10 or len2 < 10:
            continue

        # Angle between vectors
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        cos_angle = dot / (len1 * len2)
        cos_angle = max(-1, min(1, cos_angle))  # Clamp to [-1, 1]
        angle = math.acos(cos_angle)
        angle_deg = math.degrees(angle)

        # Significant direction change (>30 degrees)?
        if angle_deg > 30:
            inflections.append({
                "game_time": p_curr.get("game_time"),
                "angle_deg": round(angle_deg, 2),
                "dist_prev": round(len1, 2),
                "dist_next": round(len2, 2),
                "pos": [round(p_curr[0], 1), round(p_curr[1], 1), round(p_curr[2], 1)],
            })

    return inflections

def wall_to_game_time(wall_time, cal_wall_time, cal_game_time):
    """Convert wall-clock time to game-clock time using calibration point."""
    if cal_wall_time is None or cal_game_time is None:
        return None
    offset = wall_time - cal_wall_time
    return cal_game_time + offset

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_trajectory_vs_clicks.py <replay_id>")
        sys.exit(1)

    replay_id = sys.argv[1]

    # Load calibration
    cal_wall, cal_game = load_calibration(r"C:\tmp\calibration.json")
    if cal_wall is None:
        print("ERROR: Cannot proceed without calibration. Exiting.")
        sys.exit(1)
    print(f"\nCalibration: wall_time={cal_wall:.4f} -> game_time={cal_game:.2f}s")

    # Load clicks
    clicks = load_clicks(r"C:\tmp\clicks.json")

    # Convert clicks to game time
    click_game_times = []
    for click in clicks:
        gt = wall_to_game_time(click["wall_time"], cal_wall, cal_game)
        if gt is not None:
            click_game_times.append({
                "game_time": round(gt, 4),
                "x": click["x"],
                "y": click["y"],
                "wall_time": click["wall_time"],
            })

    print(f"Converted {len(click_game_times)} clicks to game time:")
    for c in click_game_times[:10]:
        print(f"  game_time={c['game_time']:.2f}s pos=({c['x']},{c['y']})")
    if len(click_game_times) > 10:
        print(f"  ... and {len(click_game_times) - 10} more")

    # Load replay positions
    positions = load_replay_positions(replay_id)

    # Detect inflection points in the trajectory
    inflections = detect_inflections(positions, threshold=100)
    print(f"\nDetected {len(inflections)} inflection points (>30° direction change):")
    for inf in inflections[:15]:
        print(f"  game_time={inf['game_time']:.2f}s angle={inf['angle_deg']}° "
              f"pos=({inf['pos'][0]:.0f}, {inf['pos'][2]:.0f})")

    # Correlate: which inflections are close (in time) to clicks?
    print("\n--- Click-to-Inflection Correlation ---")
    matched = 0
    for click in click_game_times:
        closest_inf = None
        closest_dist = float('inf')
        for inf in inflections:
            if inf["game_time"] is None:
                continue
            time_dist = abs(click["game_time"] - inf["game_time"])
            if time_dist < closest_dist and time_dist < 1.5:  # Within 1.5s
                closest_dist = time_dist
                closest_inf = inf
        if closest_inf:
            print(f"  Click @ {click['game_time']:.2f}s "
                  f"-> Inflection @ {closest_inf['game_time']:.2f}s (Δt={closest_dist:.2f}s) "
                  f"angle={closest_inf['angle_deg']}°")
            matched += 1
        else:
            print(f"  Click @ {click['game_time']:.2f}s -> NO NEARBY INFLECTION")

    print(f"\nMatched {matched}/{len(click_game_times)} clicks to inflections.")

    # Save report
    report = {
        "replay_id": replay_id,
        "calibration": {"wall_time": cal_wall, "game_time": cal_game},
        "click_count": len(click_game_times),
        "clicks_game_time": click_game_times,
        "inflection_count": len(inflections),
        "inflections": inflections,
        "match_count": matched,
    }
    out_file = Path(r"C:\tmp") / f"trajectory_analysis_{replay_id}.json"
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved report to {out_file}")

    # Visual plot (if matplotlib available)
    try:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # TODO: implement actual trajectory plotting
        ax1.text(0.5, 0.5, "Trajectory plot placeholder\n(integrate with emulator_v2 output)",
                ha="center", va="center", transform=ax1.transAxes)
        ax1.set_title("Hero Trajectory with Click Markers")

        ax2.text(0.5, 0.5, f"Matched {matched}/{len(click_game_times)} clicks",
                ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Click-to-Inflection Matches")

        plt.savefig(Path(r"C:\tmp") / f"trajectory_plot_{replay_id}.png")
        print(f"Saved plot to C:\\tmp\\trajectory_plot_{replay_id}.png")
    except ImportError:
        print("\nNote: matplotlib not available; skipping visual plot.")

if __name__ == "__main__":
    main()
