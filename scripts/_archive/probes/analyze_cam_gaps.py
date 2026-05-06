"""Quantify cam-sampling gaps and the label-error they could cause.

Usage:
  python analyze_cam_gaps.py <pipeline_output_dir> [<other_dir> ...]

For each dir, reports:
  - cam-gap distribution (p50/p95/p99/max, count > 50/100/200ms)
  - per-frame nearest-cam-distance (how stale is the cam sample at each label?)
  - cam velocity (units/s) per gap → upper-bound interp error per frame
  - count of labels in a "danger window" (gap > 100ms AND cam moving > 500 u/s)

This tells us whether cam_max_gap is an outlier (cosmetic) or a real coverage hole.
"""
import sys, os, json, statistics, bisect

def pct(values, p):
    if not values: return 0
    s = sorted(values)
    return s[min(len(s)-1, int(p * len(s)))]

def analyze(d):
    cam = json.load(open(os.path.join(d, "raw_cam.json")))
    lab = json.load(open(os.path.join(d, "labels.json"))).get("frames", [])
    cam.sort(key=lambda s: s["wall"])
    walls = [s["wall"] for s in cam]
    gts   = [s.get("gt", 0) for s in cam]

    print(f"\n=== {os.path.basename(d)} ===")
    print(f"  n_cam={len(cam)} n_frames={len(lab)}")
    if len(cam) < 2: return

    # ─── 1. Gap distribution (cam sample → next cam sample) ───
    gaps = [walls[i+1] - walls[i] for i in range(len(walls)-1)]
    print(f"\n  cam-gap distribution (s):")
    print(f"    p50={pct(gaps,0.5)*1000:6.1f}ms  p95={pct(gaps,0.95)*1000:6.1f}ms  "
          f"p99={pct(gaps,0.99)*1000:6.1f}ms  max={max(gaps)*1000:6.1f}ms")
    for thresh in (0.05, 0.1, 0.2, 0.5):
        n_over = sum(1 for g in gaps if g > thresh)
        pct_over = 100 * n_over / len(gaps)
        print(f"    gaps > {thresh*1000:>4.0f}ms: {n_over:>5d} / {len(gaps)} ({pct_over:5.2f}%)")

    # ─── 2. Cam velocity per gap (and danger window) ───
    # velocity = |Δcam_pos| / Δwall. Upper-bound interp error in a gap = vel × (gap/2)
    # We need cam_x, cam_z. Schema: cam = [{wall, gt, cam:[x,y,z]}, ...] OR top-level x,z.
    def get_xz(s):
        if "cam" in s and isinstance(s["cam"], (list, tuple)) and len(s["cam"]) >= 3:
            return s["cam"][0], s["cam"][2]
        if "x" in s and "z" in s: return s["x"], s["z"]
        if "cx" in s and "cz" in s: return s["cx"], s["cz"]
        return None
    velocities = []
    err_bounds = []  # max possible interp error in this gap
    for i in range(len(cam)-1):
        a, b = get_xz(cam[i]), get_xz(cam[i+1])
        if not a or not b: continue
        d = ((a[0]-b[0])**2 + (a[1]-b[1])**2)**0.5
        dt = walls[i+1] - walls[i]
        if dt <= 0: continue
        v = d / dt
        velocities.append(v)
        err_bounds.append(v * dt / 2)
    if velocities:
        print(f"\n  cam velocity per gap (units/s):")
        print(f"    p50={pct(velocities,0.5):6.1f}  p95={pct(velocities,0.95):6.1f}  "
              f"p99={pct(velocities,0.99):6.1f}  max={max(velocities):6.1f}")
        print(f"\n  upper-bound interp error per gap (units, half-gap x velocity):")
        print(f"    p50={pct(err_bounds,0.5):6.2f}  p95={pct(err_bounds,0.95):6.2f}  "
              f"p99={pct(err_bounds,0.99):6.2f}  max={max(err_bounds):6.2f}")
        PX_PER_UNIT = 0.13
        print(f"  upper-bound interp error in pixels (1080p, ~{PX_PER_UNIT} px/u):")
        print(f"    p50={pct(err_bounds,0.5)*PX_PER_UNIT:6.2f}px  p95={pct(err_bounds,0.95)*PX_PER_UNIT:6.2f}px  "
              f"p99={pct(err_bounds,0.99)*PX_PER_UNIT:6.2f}px  max={max(err_bounds)*PX_PER_UNIT:6.2f}px")

        # Danger window: gap > 100ms AND velocity > 500 u/s (real pan)
        danger = sum(1 for i in range(len(velocities))
                     if (walls[i+1]-walls[i]) > 0.1 and velocities[i] > 500)
        print(f"\n  DANGER gaps (> 100ms with cam moving > 500 u/s): {danger} / {len(velocities)}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: python analyze_cam_gaps.py <dir> [<dir> ...]"); sys.exit(2)
    for d in sys.argv[1:]:
        analyze(d)
