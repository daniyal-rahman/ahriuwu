#!/usr/bin/env python3
"""Calibrate world-to-screen projection matrix using turret positions.

Solves for a 2x2 matrix M where:
    screen_pixel = screen_center + M @ (world_pos - garen_world_pos)

Usage:
    # Interactive mode: click turrets in a frame to calibrate
    python scripts/calibrate_projection.py \
        --frame path/to/screenshot.png \
        --garen-pos 3500 7200 \
        --output data/projection_matrix.json

    # Use movement data to auto-lookup Garen's position at a game timestamp
    python scripts/calibrate_projection.py \
        --frame path/to/screenshot.png \
        --movement-data /tmp/movement_data.pkl \
        --game-time 120.5 \
        --garen-entity 0x400000ae \
        --output data/projection_matrix.json

    # Verify calibration by overlaying turret projections on a frame
    python scripts/calibrate_projection.py \
        --verify data/projection_matrix.json \
        --frame path/to/screenshot.png \
        --garen-pos 3500 7200
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

# Summoner's Rift turret world coordinates (X, Y in game units)
# Source: https://hextechdocs.dev/map-data/
# Map bounds: min (-120, -120), max (14870, 14980)
TURRET_COORDS = {
    # Blue side
    "Blue Top T1 (outer)":  (981, 10441),
    "Blue Top T2 (inner)":  (1512, 6699),
    "Blue Top T3 (base)":   (1169, 4287),
    "Blue Mid T1 (outer)":  (5846, 6396),
    "Blue Mid T2 (inner)":  (5048, 4812),
    "Blue Mid T3 (base)":   (3651, 3696),
    "Blue Bot T1 (outer)":  (10504, 1029),
    "Blue Bot T2 (inner)":  (6919, 1483),
    "Blue Bot T3 (base)":   (4281, 1253),
    "Blue Nexus Top":       (1748, 2270),
    "Blue Nexus Bot":       (2177, 1807),
    # Red side
    "Red Top T1 (outer)":   (4318, 13875),
    "Red Top T2 (inner)":   (7943, 13411),
    "Red Top T3 (base)":    (10481, 13650),
    "Red Mid T1 (outer)":   (8955, 8510),
    "Red Mid T2 (inner)":   (9767, 10113),
    "Red Mid T3 (base)":    (11134, 11207),
    "Red Bot T1 (outer)":   (13866, 4505),
    "Red Bot T2 (inner)":   (13327, 8226),
    "Red Bot T3 (base)":    (13624, 10572),
    "Red Nexus Top":        (12611, 13084),
    "Red Nexus Bot":        (13052, 12612),
}

TURRET_NAMES = list(TURRET_COORDS.keys())


def lookup_garen_pos(movement_data_path: str, game_time: float,
                     garen_entity: int) -> tuple[float, float]:
    """Look up Garen's world position at a specific game time from movement data."""
    import pickle

    with open(movement_data_path, "rb") as f:
        movements = pickle.load(f)

    # Filter to Garen's entity, sort by time
    garen_moves = [
        m for m in movements
        if m.get("entity_id") == garen_entity or m.get("block_param") == garen_entity
    ]
    garen_moves.sort(key=lambda m: m.get("game_time") or m.get("block_timestamp", 0))

    if not garen_moves:
        print(f"Error: no movements found for entity 0x{garen_entity:x}")
        sys.exit(1)

    # Find closest movement to game_time
    best = min(garen_moves, key=lambda m: abs(
        (m.get("game_time") or m.get("block_timestamp", 0)) - game_time
    ))
    x = best.get("current_x")
    y = best.get("current_y")

    if x is None or y is None:
        print(f"Error: closest movement at t={best.get('game_time')} has no coordinates")
        sys.exit(1)

    t = best.get("game_time") or best.get("block_timestamp")
    print(f"Garen position at t={t:.1f}s (target {game_time:.1f}s): ({x:.1f}, {y:.1f})")
    return (x, y)


def click_point(frame_path: str, prompt: str) -> tuple[float, float]:
    """Show a frame and let the user click a single point."""
    frame = cv2.imread(frame_path)
    click_pos = [None]

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos[0] = (x, y)

    window_name = prompt
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(frame.shape[1], 1600), min(frame.shape[0], 900))
    cv2.setMouseCallback(window_name, on_mouse)

    print(f"\n{prompt}")
    while click_pos[0] is None:
        cv2.imshow(window_name, frame)
        cv2.waitKey(100)
    cv2.destroyAllWindows()
    return click_pos[0]


def interactive_calibrate(frame_path: str, garen_world: tuple[float, float],
                          screen_center: tuple[float, float]) -> np.ndarray:
    """Interactive calibration: user clicks on turrets in the frame.

    Returns the 2x2 projection matrix M.
    """
    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Error: cannot read frame {frame_path}")
        sys.exit(1)

    # Print turret list
    print("\n--- Turret reference list ---")
    for i, name in enumerate(TURRET_NAMES):
        wx, wy = TURRET_COORDS[name]
        print(f"  [{i:2d}] {name:30s}  ({wx:6.0f}, {wy:6.0f})")

    print(f"\nGaren world pos: ({garen_world[0]:.1f}, {garen_world[1]:.1f})")
    print(f"Screen center: ({screen_center[0]:.1f}, {screen_center[1]:.1f})")

    correspondences = []
    click_pos = [None]
    window_name = "Calibration - Click turrets (q=quit, u=undo)"

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos[0] = (x, y)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(frame.shape[1], 1600), min(frame.shape[0], 900))
    cv2.setMouseCallback(window_name, on_mouse)

    print("\nClick on a visible turret in the frame, then enter its number.")
    print("Press 'q' when done (need at least 2 points, 4+ recommended).")
    print("Press 'u' to undo the last point.\n")

    while True:
        # Draw current correspondences on frame
        display = frame.copy()
        for (sx, sy), turret_idx in correspondences:
            name = TURRET_NAMES[turret_idx]
            cv2.circle(display, (int(sx), int(sy)), 8, (0, 255, 0), 2)
            cv2.putText(display, f"[{turret_idx}]", (int(sx) + 10, int(sy) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Show point count
        cv2.putText(display, f"Points: {len(correspondences)} (need 2+, recommend 4+)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        cv2.imshow(window_name, display)
        click_pos[0] = None

        key = cv2.waitKey(100) & 0xFF
        if key == ord("q"):
            break
        if key == ord("u") and correspondences:
            removed = correspondences.pop()
            print(f"  Undid point [{removed[1]}]")
            continue

        if click_pos[0] is not None:
            sx, sy = click_pos[0]
            print(f"  Clicked at pixel ({sx}, {sy})")
            turret_idx_str = input("  Enter turret number (or 's' to skip): ").strip()
            if turret_idx_str.lower() == "s":
                continue
            try:
                turret_idx = int(turret_idx_str)
                if 0 <= turret_idx < len(TURRET_NAMES):
                    correspondences.append(((sx, sy), turret_idx))
                    name = TURRET_NAMES[turret_idx]
                    wx, wy = TURRET_COORDS[name]
                    print(f"  -> {name} at world ({wx}, {wy})")
                else:
                    print(f"  Invalid index (0-{len(TURRET_NAMES)-1})")
            except ValueError:
                print("  Invalid number")

    cv2.destroyAllWindows()

    if len(correspondences) < 2:
        print("Error: need at least 2 correspondences to solve for M")
        sys.exit(1)

    return solve_projection(correspondences, garen_world, screen_center)


def solve_projection(correspondences: list, garen_world: tuple[float, float],
                     screen_center: tuple[float, float]) -> np.ndarray:
    """Solve for 2x2 projection matrix M via least squares.

    For each correspondence (screen_pixel, turret_idx):
        screen_pixel - screen_center = M @ (turret_world - garen_world)

    Returns M as a 2x2 numpy array.
    """
    A_rows = []
    b_rows = []

    for (sx, sy), turret_idx in correspondences:
        name = TURRET_NAMES[turret_idx]
        wx, wy = TURRET_COORDS[name]

        # World offset from Garen
        dwx = wx - garen_world[0]
        dwy = wy - garen_world[1]

        # Screen offset from center
        dpx = sx - screen_center[0]
        dpy = sy - screen_center[1]

        # dpx = m00*dwx + m01*dwy
        # dpy = m10*dwx + m11*dwy
        A_rows.append([dwx, dwy, 0, 0])
        b_rows.append(dpx)
        A_rows.append([0, 0, dwx, dwy])
        b_rows.append(dpy)

    A = np.array(A_rows, dtype=np.float64)
    b = np.array(b_rows, dtype=np.float64)

    result, residuals, rank, sv = np.linalg.lstsq(A, b, rcond=None)
    m00, m01, m10, m11 = result
    M = np.array([[m00, m01], [m10, m11]])

    print(f"\n--- Projection Matrix M ---")
    print(f"  [[{m00:+.6f}, {m01:+.6f}],")
    print(f"   [{m10:+.6f}, {m11:+.6f}]]")

    if len(residuals) > 0:
        rmse = np.sqrt(residuals.sum() / len(b_rows))
        print(f"  RMSE: {rmse:.2f} pixels")

    # Print per-point errors
    print(f"\nPer-point errors:")
    for (sx, sy), turret_idx in correspondences:
        name = TURRET_NAMES[turret_idx]
        wx, wy = TURRET_COORDS[name]
        dw = np.array([wx - garen_world[0], wy - garen_world[1]])
        predicted = screen_center + M @ dw
        err = np.sqrt((sx - predicted[0]) ** 2 + (sy - predicted[1]) ** 2)
        print(f"  [{turret_idx:2d}] {name:30s}: predicted ({predicted[0]:.0f}, {predicted[1]:.0f}), "
              f"actual ({sx}, {sy}), error {err:.1f}px")

    return M


def verify_calibration(matrix_path: str, frame_path: str,
                       garen_world: tuple[float, float]):
    """Overlay projected turret positions on a frame to verify calibration."""
    with open(matrix_path) as f:
        data = json.load(f)

    M = np.array(data["matrix"])
    screen_center = np.array(data["screen_center"])

    frame = cv2.imread(frame_path)
    if frame is None:
        print(f"Error: cannot read frame {frame_path}")
        sys.exit(1)

    print(f"Matrix M:\n{M}")
    print(f"Screen center: {screen_center}")
    print(f"Garen world: {garen_world}")

    for name, (wx, wy) in TURRET_COORDS.items():
        dw = np.array([wx - garen_world[0], wy - garen_world[1]])
        sp = screen_center + M @ dw
        px, py = int(sp[0]), int(sp[1])

        # Only draw if on screen
        if 0 <= px < frame.shape[1] and 0 <= py < frame.shape[0]:
            cv2.circle(frame, (px, py), 10, (0, 0, 255), 2)
            cv2.putText(frame, name.split("(")[0].strip(), (px + 12, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw screen center
    cx, cy = int(screen_center[0]), int(screen_center[1])
    cv2.drawMarker(frame, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 20, 2)

    window_name = "Verification - Projected turrets (press any key to close)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(frame.shape[1], 1600), min(frame.shape[0], 900))
    cv2.imshow(window_name, frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Calibrate world-to-screen projection matrix"
    )
    parser.add_argument("--frame", required=True, help="Path to replay screenshot/frame")
    parser.add_argument("--output", "-o", default="data/projection_matrix.json",
                        help="Output path for matrix (default: data/projection_matrix.json)")

    # Garen position: either direct or from movement data
    pos_group = parser.add_mutually_exclusive_group()
    pos_group.add_argument("--garen-pos", nargs=2, type=float, metavar=("X", "Y"),
                           help="Garen world position (X Y)")
    parser.add_argument("--movement-data", help="Path to movement_data.pkl")
    parser.add_argument("--game-time", type=float,
                        help="Game timestamp (seconds) for position lookup")
    parser.add_argument("--garen-entity", default="0x400000ae",
                        help="Garen entity ID hex (default: 0x400000ae)")

    parser.add_argument("--verify", metavar="MATRIX_JSON",
                        help="Verify mode: overlay projections using saved matrix")
    parser.add_argument("--click-garen", action="store_true",
                        help="Click on Garen's screen position (for directed/unlocked camera)")

    args = parser.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        print(f"Error: cannot read frame {args.frame}")
        sys.exit(1)
    h, w = frame.shape[:2]
    del frame

    # Get Garen's world position
    if args.garen_pos:
        garen_world = tuple(args.garen_pos)
    elif args.movement_data and args.game_time is not None:
        entity = int(args.garen_entity, 16) if args.garen_entity.startswith("0x") \
            else int(args.garen_entity)
        garen_world = lookup_garen_pos(args.movement_data, args.game_time, entity)
    else:
        print("Error: provide --garen-pos X Y or --movement-data + --game-time")
        sys.exit(1)

    if args.verify:
        verify_calibration(args.verify, args.frame, garen_world)
        return

    if args.click_garen:
        screen_center = click_point(args.frame, "Click on Garen's feet, then press any key")
        print(f"Garen screen pos: ({screen_center[0]:.0f}, {screen_center[1]:.0f})")
    else:
        screen_center = (w / 2.0, h / 2.0)

    M = interactive_calibrate(args.frame, garen_world, screen_center)

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_data = {
        "matrix": M.tolist(),
        "screen_center": list(screen_center),
        "garen_world": list(garen_world),
        "turret_coords": {k: list(v) for k, v in TURRET_COORDS.items()},
        "notes": "screen_pixel = screen_center + M @ (world_pos - camera_world_pos)",
    }
    with open(out_path, "w") as f:
        json.dump(out_data, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
