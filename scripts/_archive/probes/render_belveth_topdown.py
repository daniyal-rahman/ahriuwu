"""Render a topdown 2D overlay video of Bel'Veth's first 10 min:
  - Summoner's Rift world grid as background
  - Bel'Veth's hero trajectory (interpolated line)
  - Click destination markers (fade in/out over ~5s around click_t)
  - Timecode + click counter HUD

Inputs:
  /tmp/belveth_tightpoll_clicks.json (clicks)
  /tmp/belveth_full_clicks.json      (hero trajectory)

Output: /tmp/belveth_first10min.mp4
"""
import json, math, os
import numpy as np
import cv2

CLICK_PATH = "/tmp/belveth_tightpoll_clicks.json"
TRAJ_PATH = "/tmp/belveth_full_clicks.json"
OUT = "/tmp/belveth_first10min.mp4"
START_GT = 0.0
END_GT = 600.0  # first 10 min
FPS = 20
W = 1080; H = 1080
WORLD_MAX = 15000.0

def world_to_screen(x, z):
    """World (x, z) → screen (sx, sy). In LoL map, (0,0) is bottom-left corner
    of blue base. Higher z is toward top of map. Our frame has y=0 at top,
    so flip z."""
    sx = int(x / WORLD_MAX * W)
    sy = int(H - (z / WORLD_MAX * H))
    return sx, sy

def draw_map_background(frame):
    # Simple SR-ish map: 3 lanes + river
    # Blue base bottom-left, red top-right
    frame[:] = (28, 35, 22)  # dark olive
    # Fountain zones
    cv2.rectangle(frame, world_to_screen(0, 2000), world_to_screen(2000, 0), (30, 60, 120), -1)
    cv2.rectangle(frame, world_to_screen(13000, 15000), world_to_screen(15000, 13000), (30, 30, 100), -1)
    # Lanes (approximate)
    # Mid (diag from blue base to red)
    cv2.line(frame,
             world_to_screen(1200, 1200), world_to_screen(13800, 13800),
             (60, 60, 50), 80)
    # Top lane (L-shape: vertical then horizontal)
    cv2.line(frame, world_to_screen(1200, 1200), world_to_screen(1200, 13000), (60, 60, 50), 80)
    cv2.line(frame, world_to_screen(1200, 13000), world_to_screen(13800, 13800), (60, 60, 50), 80)
    # Bot lane
    cv2.line(frame, world_to_screen(1200, 1200), world_to_screen(13000, 1200), (60, 60, 50), 80)
    cv2.line(frame, world_to_screen(13000, 1200), world_to_screen(13800, 13800), (60, 60, 50), 80)
    # River (diag perpendicular to mid)
    cv2.line(frame,
             world_to_screen(1200, 13800), world_to_screen(13800, 1200),
             (100, 60, 40), 60)
    # Gridlines
    for v in range(0, 15001, 3000):
        sx = int(v / WORLD_MAX * W)
        sy = int(v / WORLD_MAX * H)
        cv2.line(frame, (sx, 0), (sx, H), (50, 50, 50), 1)
        cv2.line(frame, (0, H-sy), (W, H-sy), (50, 50, 50), 1)

def main():
    clicks_d = json.load(open(CLICK_PATH))
    traj_d = json.load(open(TRAJ_PATH))
    clicks = [c for c in clicks_d["clicks"] if START_GT <= c["game_t"] < END_GT]
    # hero trajectory (in traj_d["hero_trajectory"])
    trajs = [(t["game_t"], t["pos"][0], t["pos"][2])
             for t in traj_d["hero_trajectory"] if START_GT <= t["game_t"] < END_GT]
    trajs.sort()

    print(f"clicks in window: {len(clicks)}")
    print(f"hero samples in window: {len(trajs)}")
    if not trajs:
        print("no hero data in first 10 min — using click positions as proxy")

    # Build frame sequence: 1 frame per 1/FPS game seconds
    n_frames = int((END_GT - START_GT) * FPS)
    print(f"rendering {n_frames} frames to {OUT}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(OUT, fourcc, FPS, (W, H))
    if not vw.isOpened():
        print("ERR cv2.VideoWriter failed"); return 1

    # Pre-sort clicks for fast lookup
    clicks_sorted = sorted(clicks, key=lambda c: c["game_t"])
    click_times = np.array([c["game_t"] for c in clicks_sorted])

    trail = []  # recent hero positions for trail effect
    traj_t = np.array([t[0] for t in trajs]) if trajs else np.array([])
    traj_x = np.array([t[1] for t in trajs]) if trajs else np.array([])
    traj_z = np.array([t[2] for t in trajs]) if trajs else np.array([])

    # Collect all cumulative click positions for heatmap
    all_click_positions = [(c["x"], c["z"], c["game_t"]) for c in clicks_sorted]

    for fi in range(n_frames):
        t = START_GT + fi / FPS
        frame = np.zeros((H, W, 3), dtype=np.uint8)
        draw_map_background(frame)

        # Draw cumulative faded clicks (oldest clicks dim, recent bright)
        for cx, cz, ct in all_click_positions:
            if ct > t: break
            age = t - ct
            if age > 5.0: continue  # fade window
            alpha = 1.0 - age / 5.0
            sx, sy = world_to_screen(cx, cz)
            color = (int(40 * alpha), int(120 * alpha), int(255 * alpha))
            radius = int(8 + 4 * alpha)
            cv2.circle(frame, (sx, sy), radius, color, 2)
            cv2.drawMarker(frame, (sx, sy), color, cv2.MARKER_CROSS, 12, 1)

        # All clicks so far (lightly)
        for cx, cz, ct in all_click_positions:
            if ct > t: break
            age = t - ct
            if age > 5.0:
                sx, sy = world_to_screen(cx, cz)
                cv2.circle(frame, (sx, sy), 2, (30, 60, 90), -1)

        # Bel'Veth current position (interp)
        if len(traj_t) > 0:
            if t < traj_t[0]:
                bx, bz = traj_x[0], traj_z[0]
            elif t > traj_t[-1]:
                bx, bz = traj_x[-1], traj_z[-1]
            else:
                bx = float(np.interp(t, traj_t, traj_x))
                bz = float(np.interp(t, traj_t, traj_z))
            # Trail
            trail.append((bx, bz, t))
            trail = [(x, z, tt) for x, z, tt in trail if t - tt < 10.0]
            if len(trail) > 1:
                pts = np.array([world_to_screen(x, z) for x, z, _ in trail], dtype=np.int32)
                cv2.polylines(frame, [pts], False, (255, 200, 100), 2)
            # Bel'Veth marker
            sx, sy = world_to_screen(bx, bz)
            cv2.circle(frame, (sx, sy), 14, (0, 255, 255), 3)
            cv2.circle(frame, (sx, sy), 4, (0, 255, 255), -1)

        # HUD
        mins = int(t // 60); secs = int(t % 60)
        n_clicks_so_far = sum(1 for c in clicks_sorted if c["game_t"] <= t)
        cv2.putText(frame, f"Bel'Veth  t={mins:02d}:{secs:02d}  clicks={n_clicks_so_far}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (220, 220, 220), 2)
        cv2.putText(frame, "yellow = Bel'Veth  red X = click dest",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 180, 180), 1)
        cv2.putText(frame, "BLUE",
                    world_to_screen(300, 1500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 100), 2)
        cv2.putText(frame, "RED",
                    world_to_screen(13000, 14200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 255), 2)

        vw.write(frame)
        if fi % (FPS * 30) == 0:
            print(f"  frame {fi}/{n_frames}  gt={t:.1f}  clicks_so_far={n_clicks_so_far}")

    vw.release()
    print(f"wrote {OUT}")
    # Also write a PNG snapshot of final state as a static trajectory map
    snap = np.zeros((H, W, 3), dtype=np.uint8)
    draw_map_background(snap)
    for cx, cz, ct in all_click_positions:
        sx, sy = world_to_screen(cx, cz)
        cv2.circle(snap, (sx, sy), 3, (0, 100, 200), -1)
    if len(traj_t) > 0:
        pts = np.array([world_to_screen(traj_x[i], traj_z[i]) for i in range(len(traj_t))], dtype=np.int32)
        cv2.polylines(snap, [pts], False, (255, 200, 100), 2)
    cv2.imwrite("/tmp/belveth_first10min.png", snap)
    print(f"wrote /tmp/belveth_first10min.png (static snapshot)")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
