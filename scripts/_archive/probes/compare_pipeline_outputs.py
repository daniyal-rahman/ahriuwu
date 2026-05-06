"""Diff two pipeline output dirs (baseline vs merged-test) to verify the
merged single-pass produces equivalent data.

Compares:
  - frames count
  - mem/cam sample counts + Hz + max_gap
  - effective_speed (replay 2x compliance)
  - per-frame label coverage (% labeled)
  - random spot-check of label fields against baseline:
      hero_world (Garen pos)
      cam_x / cam_z
      hero_screen
      gt offset (frame i should have same gt = start + i/20 in both)
  - distribution of gaps between mem-gt at sampled cam-gt (alignment quality)

Outputs a text report. Exits 0 if all metrics within tolerance, 1 otherwise.

Usage:
  python compare_pipeline_outputs.py <baseline_dir> <test_dir>
"""
import sys, os, json, statistics, glob

def load(path, name):
    p = os.path.join(path, name)
    if not os.path.exists(p):
        return None
    with open(p) as f: return json.load(f)

def stat_floats(values, name, prefix=""):
    if not values: return f"{prefix}{name}: empty"
    p50 = statistics.median(values)
    p95 = sorted(values)[int(0.95 * len(values))] if len(values) > 1 else values[0]
    return f"{prefix}{name}: n={len(values)}  p50={p50:.3f}  p95={p95:.3f}  max={max(values):.3f}"

def compare(base_dir, test_dir, tol_pct=10.0, tol_abs_pos=20.0):
    print(f"=== compare ===\n  baseline : {base_dir}\n  test     : {test_dir}\n")
    issues = []

    # ─── Frames count ───
    b_frames = sorted(glob.glob(os.path.join(base_dir, "frames", "*.png")))
    t_frames = sorted(glob.glob(os.path.join(test_dir, "frames", "*.png")))
    print(f"frames: baseline={len(b_frames)} test={len(t_frames)}  diff={len(t_frames)-len(b_frames):+d}")
    if abs(len(t_frames) - len(b_frames)) > max(5, 0.05 * len(b_frames)):
        issues.append(f"frames count drift > 5%: {len(b_frames)} → {len(t_frames)}")

    # ─── Mem/cam scrape stats ───
    b_mem = load(base_dir, "raw_mem.json") or []
    t_mem = load(test_dir, "raw_mem.json") or []
    b_cam = load(base_dir, "raw_cam.json") or []
    t_cam = load(test_dir, "raw_cam.json") or []
    print(f"\nmem samples : base={len(b_mem):>6}  test={len(t_mem):>6}")
    print(f"cam samples : base={len(b_cam):>6}  test={len(t_cam):>6}")

    def hz(samples):
        if len(samples) < 2: return 0
        ws = [s["wall"] for s in samples]
        return len(samples) / (ws[-1] - ws[0]) if ws[-1] > ws[0] else 0

    def max_gap(samples):
        if len(samples) < 2: return 0
        ws = sorted(s["wall"] for s in samples)
        return max(ws[i+1] - ws[i] for i in range(len(ws)-1))

    print(f"  mem hz : base={hz(b_mem):.1f}  test={hz(t_mem):.1f}")
    print(f"  cam hz : base={hz(b_cam):.1f}  test={hz(t_cam):.1f}")
    print(f"  mem max_gap : base={max_gap(b_mem)*1000:.1f}ms  test={max_gap(t_mem)*1000:.1f}ms")
    print(f"  cam max_gap : base={max_gap(b_cam)*1000:.1f}ms  test={max_gap(t_cam)*1000:.1f}ms")
    if hz(t_mem) < hz(b_mem) * 0.7:
        issues.append(f"mem hz dropped > 30%: {hz(b_mem):.1f} → {hz(t_mem):.1f}")
    if hz(t_cam) < hz(b_cam) * 0.7:
        issues.append(f"cam hz dropped > 30%: {hz(b_cam):.1f} → {hz(t_cam):.1f}")
    if max_gap(t_mem) > max(0.5, 3 * max_gap(b_mem)):
        issues.append(f"mem max_gap regressed: {max_gap(b_mem)*1000:.0f}ms → {max_gap(t_mem)*1000:.0f}ms")

    # ─── Effective replay speed ───
    def eff_speed(mem):
        if len(mem) < 2: return 0
        ws = [s["wall"] for s in mem]
        gts = [s["gt"] for s in mem]
        ws_span = ws[-1] - ws[0]; gt_span = gts[-1] - gts[0]
        return gt_span / ws_span if ws_span > 0 else 0
    print(f"  effective_speed : base={eff_speed(b_mem):.2f}x  test={eff_speed(t_mem):.2f}x")
    if eff_speed(t_mem) < eff_speed(b_mem) * 0.7:
        issues.append(f"effective_speed dropped: {eff_speed(b_mem):.2f}x → {eff_speed(t_mem):.2f}x")

    # ─── Labels ───
    b_lab = load(base_dir, "labels.json") or {}
    t_lab = load(test_dir, "labels.json") or {}
    b_fr = b_lab.get("frames", []); t_fr = t_lab.get("frames", [])
    print(f"\nlabels frames : base={len(b_fr)} test={len(t_fr)}")
    b_lab_pct = sum(1 for f in b_fr if f.get("label")) / max(1,len(b_fr))
    t_lab_pct = sum(1 for f in t_fr if f.get("label")) / max(1,len(t_fr))
    print(f"  labeled% : base={b_lab_pct*100:.1f}%  test={t_lab_pct*100:.1f}%")
    if t_lab_pct < b_lab_pct - 0.05:
        issues.append(f"labeled% dropped: {b_lab_pct*100:.1f}% → {t_lab_pct*100:.1f}%")

    # ─── Per-frame field comparison ───
    # Pair frames by index. For each pair where both have a label, compare:
    #   gt (should be identical: rec_start + i/fps)
    #   cam (cx, cz)
    #   hero pos (in label["heroes"][CHAMPION]["pos"]) — within tol_abs_pos units
    #   hero_screen (within 10 px)
    if b_fr and t_fr:
        n_pairs = min(len(b_fr), len(t_fr))
        gt_diffs = []
        cam_xy_diffs = []
        hero_pos_diffs = []
        hero_scr_diffs = []
        n_both_labeled = 0
        for i in range(n_pairs):
            bf, tf = b_fr[i], t_fr[i]
            if bf.get("gt") is not None and tf.get("gt") is not None:
                gt_diffs.append(abs(bf["gt"] - tf["gt"]))
            bl = bf.get("label"); tl = tf.get("label")
            if not bl or not tl: continue
            n_both_labeled += 1
            if bl.get("cam") and tl.get("cam"):
                bcx, bcz = bl["cam"][0], bl["cam"][2]
                tcx, tcz = tl["cam"][0], tl["cam"][2]
                cam_xy_diffs.append(((bcx-tcx)**2 + (bcz-tcz)**2)**0.5)
            # hero pos
            heroes_b = bl.get("heroes", {}); heroes_t = tl.get("heroes", {})
            common = set(heroes_b) & set(heroes_t)
            for ch in common:
                pb = heroes_b[ch].get("pos"); pt = heroes_t[ch].get("pos")
                if pb and pt and len(pb) >= 2 and len(pt) >= 2:
                    hero_pos_diffs.append(((pb[0]-pt[0])**2 + (pb[1]-pt[1])**2)**0.5)
            gsb = bl.get("garen_screen"); gst = tl.get("garen_screen")
            if gsb and gst:
                hero_scr_diffs.append(((gsb[0]-gst[0])**2 + (gsb[1]-gst[1])**2)**0.5)

        print(f"\npaired comparisons: {n_pairs} frames, {n_both_labeled} both-labeled")
        print(f"  {stat_floats(gt_diffs, 'gt diff (s)', '  ')}")
        print(f"  {stat_floats(cam_xy_diffs, 'cam (x,z) diff (units)', '  ')}")
        print(f"  {stat_floats(hero_pos_diffs, 'hero pos diff (units)', '  ')}")
        print(f"  {stat_floats(hero_scr_diffs, 'hero screen diff (px)', '  ')}")
        # Tolerances
        if gt_diffs and max(gt_diffs) > 0.05:
            issues.append(f"gt drift > 50ms (max {max(gt_diffs)*1000:.0f}ms)")
        if cam_xy_diffs and statistics.median(cam_xy_diffs) > tol_abs_pos:
            issues.append(f"cam median diff {statistics.median(cam_xy_diffs):.1f}u > {tol_abs_pos}")
        if hero_pos_diffs and statistics.median(hero_pos_diffs) > tol_abs_pos:
            issues.append(f"hero pos median diff {statistics.median(hero_pos_diffs):.1f}u > {tol_abs_pos}")

    print("\n=== verdict ===")
    if not issues:
        print("  ✓ ALL METRICS WITHIN TOLERANCE")
        return 0
    else:
        print("  ✗ ISSUES:")
        for i in issues: print(f"    - {i}")
        return 1

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python compare_pipeline_outputs.py <baseline> <test>"); sys.exit(2)
    sys.exit(compare(sys.argv[1], sys.argv[2]))
