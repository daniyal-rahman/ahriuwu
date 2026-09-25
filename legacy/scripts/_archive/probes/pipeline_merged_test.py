"""TEST: merge pass1 (mem+cam scrape) and pass2 (video recording) into a
single replay session. Compare outputs against the existing baseline.

Runs against the same game as a known-good baseline. Outputs to a SEPARATE
directory so the original baseline is untouched.

Usage (via schtasks /IT):
    pipeline_merged_test.py --game-id 5547184086 --match-id NA1_5547184086 \\
                            --team blue --slot 0 --champion Garen \\
                            --duration 600 --rec-start 1.0 --rec-end 600

Output dir: C:\\tmp\\replay_data\\<MATCH_ID>_merged\\
  - frames/*.png            (recorded by League)
  - raw_mem.json            (mem thread data)
  - raw_cam.json            (cam thread data)
  - labels.json             (post-processed)
  - merged_stats.json       (timing breakdown)
"""
import argparse, json, os, sys, time, glob, threading, shutil, ctypes
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P  # reuse all helpers
import garen_clicks_vtable as GC  # reuse identify + poll logic

FPS = P.FPS

def _click_loop(h, hero_ptr, watched, slot_real_names, stop, all_clicks, all_casts):
    """Background thread that polls click-dest vec3s + spellbook + active_spell.
    Lifted from garen_clicks_vtable.py main loop."""
    prev_vec = {a: GC.read_vec3(h, a) for a in watched}
    prev_cd = {n: None for n in GC.SLOT_NAMES}
    prev_active = None
    last_recall_t = -10.0
    last_gt = 0
    while not stop.is_set():
        try:
            hero_pos = GC.read_vec3(h, hero_ptr + GC.HERO_POS_OFF)
            cur_vecs = {a: GC.read_vec3(h, a) for a in watched}
            pb = GC.api_get("/replay/playback")
            gt = pb.get("time", last_gt) if pb else last_gt
            last_gt = gt
            for addr in watched:
                v = cur_vecs[addr]
                if not GC.valid_vec(v):
                    continue
                prev = prev_vec.get(addr)
                if prev:
                    dx, dz = v[0]-prev[0], v[2]-prev[2]
                    d = (dx*dx + dz*dz) ** 0.5
                    if d > GC.DELTA_UNITS:
                        all_clicks.append({
                            "game_t": gt, "addr": hex(addr),
                            "x": v[0], "y": v[1], "z": v[2], "delta": round(d,1),
                            "hero_x": hero_pos[0] if hero_pos else None,
                            "hero_z": hero_pos[2] if hero_pos else None,
                        })
                prev_vec[addr] = v
            slots = GC.read_spellbook_slots(h, hero_ptr)
            for name, cd_e, cd_t in slots:
                if cd_e is None or cd_t is None: continue
                prev = prev_cd[name]
                if prev is not None and cd_e - prev > max(1.0, cd_t * 0.5):
                    all_casts.append({
                        "game_t": gt, "slot": name,
                        "spell_name": slot_real_names.get(name),
                        "hero_x": hero_pos[0] if hero_pos else None,
                        "hero_z": hero_pos[2] if hero_pos else None,
                        "cd_expire": cd_e, "total_cd": cd_t,
                    })
                prev_cd[name] = cd_e
            active_name = GC.read_active_spell_name(h, hero_ptr)
            if active_name and active_name != prev_active:
                if "recall" in active_name.lower() and gt - last_recall_t > 1.0:
                    all_casts.append({
                        "game_t": gt, "slot": "B",
                        "spell_name": active_name,
                        "hero_x": hero_pos[0] if hero_pos else None,
                        "hero_z": hero_pos[2] if hero_pos else None,
                    })
                    last_recall_t = gt
            prev_active = active_name
        except Exception:
            pass
        time.sleep(GC.POLL_MS / 1000.0)

def merged_pass(game_id, cam_key, champion, duration, staging_dir,
                rec_start=1.0, rec_end=None, force_patch=False,
                fps_mult=2, py_cores=None, speed=2.0):
    """Single-replay-session: scrape mem+cam concurrently with video recording."""
    print("\n=== MERGED PASS: launch + cam-lock + (mem || cam || record) ===", flush=True)
    overall_start = time.time()

    # Override the module-level CHAMPION so init_heroes() finds the right hero.
    P.CHAMPION = champion

    P.kill_game()
    if not P.launch_replay(game_id):
        return 0, [], [], {}, {}
    time.sleep(5)

    pid = P.find_league_pid()
    if not pid:
        print("  No game PID", flush=True); P.kill_game(); return 0, [], [], {}, {}
    base, mod_size = P.find_module_base(pid)
    if not base:
        print("  No module base", flush=True); P.kill_game(); return 0, [], [], {}, {}
    if mod_size != P.EXPECTED_MOD_SIZE and not force_patch:
        print(f"  ABORT: module size 0x{mod_size:X} != expected 0x{P.EXPECTED_MOD_SIZE:X}", flush=True)
        P.kill_game(); return 0, [], [], {}, {}

    m = P.Mem(pid)
    if m.read(base, 2) != b'MZ':
        print("  RPM verify failed", flush=True); m.close(); P.kill_game(); return 0, [], [], {}, {}
    print(f"  Memory: PID={pid} base=0x{base:X}", flush=True)

    gt_rva = P.verify_game_time(m, base)
    if not gt_rva:
        print("  GameTime RVA reads garbage", flush=True); m.close(); P.kill_game(); return 0, [], [], {}, {}

    hero_ptrs = P.init_heroes(m, base)
    if champion not in hero_ptrs:
        print(f"  {champion} not found (heroes: {list(hero_ptrs.keys())})", flush=True)
        m.close(); P.kill_game(); return 0, [], [], {}, {}
    print(f"  Heroes: {list(hero_ptrs.keys())}", flush=True)

    # Pin League to cores 0..N-2 (encoder needs ~14.5 threads anyway).
    # Pin THIS Python process to the last core so mem/cam threads don't
    # compete with the encoder for cycles.
    n_cores = os.cpu_count() or 16
    if py_cores is None:
        py_cores = [n_cores - 1]
    py_set = set(py_cores)
    league_cores = [c for c in range(n_cores) if c not in py_set]
    P.pin_league(league_cores)
    try:
        import psutil
        psutil.Process().cpu_affinity(py_cores)
        print(f"  Python pinned to cores {py_cores}", flush=True)
    except Exception as e:
        print(f"  WARN: could not pin python: {e}", flush=True)
    print(f"  League pinned to cores {league_cores}", flush=True)

    # ─── Click-dest identify BEFORE recording (game must be running so
    #     candidate vec3s correlate to hero motion). 25s of running replay.
    hero_ptr_int = hero_ptrs[champion]["ptr"] if isinstance(hero_ptrs[champion], dict) else hero_ptrs[champion]
    h_click = ctypes.windll.kernel32.OpenProcess(0x0410, False, pid)
    target_vptr = base + GC.VTABLE_RVA
    print(f"\n--- Identifying click-dest candidates (25s, replay running) ---", flush=True)
    P.replay_post("/replay/playback", {"speed": speed, "paused": False})
    time.sleep(2.0)
    top, _ = GC.identify_top_k(h_click, hero_ptr_int, target_vptr, 25, k=20)
    garen_owned = [(a,*r) for (a,*r) in top
                   if GC.read_u64(h_click, a + GC.OWNER_PTR_OFF) == hero_ptr_int]
    if not garen_owned and top:
        print(f"  WARN: no candidate owned by hero — fallback to top by avg_d", flush=True)
        garen_owned = top[:1]
    watched_addrs = [a for a,*_ in garen_owned]
    print(f"  watched click addrs: {[hex(a) for a in watched_addrs]}", flush=True)

    # Resolve slot spell-names so D/F are tagged Flash/Ignite/TP/etc.
    slot_array_addr = hero_ptr_int + GC.SPELLBOOK_OFF + GC.SLOT_ARRAY_OFF
    slot_real_names = {}
    for i, name in enumerate(GC.SLOT_NAMES):
        slot_ptr = GC.read_u64(h_click, slot_array_addr + i*8)
        nm = GC.read_slot_spell_name(h_click, slot_ptr) if slot_ptr else None
        slot_real_names[name] = nm

    # Prepare staging dir
    os.makedirs(staging_dir, exist_ok=True)
    for f in glob.glob(os.path.join(staging_dir, "**", "*.png"), recursive=True):
        try: os.remove(f)
        except: pass

    # ─── Start recording FIRST (the startTime=1.0 seek would otherwise blow
    #     away the cam-lock if we engaged it earlier). Pause first so the
    #     seek lands on a paused frame, then lock cam, THEN unpause+record.
    P.replay_post("/replay/playback", {"paused": True})
    time.sleep(0.3)
    rec_resp = P.replay_post("/replay/recording", {
        "recording": True,
        "path": staging_dir.replace("\\", "/"),
        "codec": "png",
        "framesPerSecond": FPS * fps_mult,
        "startTime": rec_start,
        "endTime": rec_end if rec_end is not None else duration + 60,
        "enforceFrameRate": True,
    })
    rec_started = time.time()
    print(f"  Recording started ({rec_resp.get('width')}x{rec_resp.get('height')}) at +{rec_started-overall_start:.1f}s", flush=True)
    # Let the seek-to-startTime settle while still paused.
    time.sleep(2.0)

    # Cam-lock recipe AFTER the seek: select → unpause @ speed → focus-lock × 2
    P.replay_post("/replay/render", {"interfaceAll": True, "selectionName": champion})
    time.sleep(0.5)
    P.replay_post("/replay/playback", {"speed": speed, "paused": False})
    time.sleep(1.0)
    P.focus_game(); time.sleep(0.5)
    P.lock_camera(cam_key)
    time.sleep(0.5)
    P.focus_game(); P.lock_camera(cam_key)
    time.sleep(0.5)
    print(f"  Camera locked (key={cam_key}), {speed}x speed, post-seek", flush=True)

    # ─── Start mem + click threads. /replay/render goes stale during recording,
    #     so the cam thread would write garbage. We synthesize cam from
    #     hero pos + measured tilt offset after the run. ───
    stop = threading.Event()
    mem_data, cam_data = [], []
    all_clicks, all_casts = [], []
    mt = threading.Thread(target=P._mem_loop, args=(m, base, hero_ptrs, gt_rva, stop, mem_data), daemon=True)
    mt.start()
    if watched_addrs:
        ck = threading.Thread(target=_click_loop,
                              args=(h_click, hero_ptr_int, watched_addrs,
                                    slot_real_names, stop, all_clicks, all_casts),
                              daemon=True)
        ck.start()
    else:
        ck = None
        print(f"  WARN: no click candidates — skipping click thread", flush=True)
    threads_started = time.time()
    print(f"  mem+click threads up at +{threads_started-overall_start:.1f}s", flush=True)

    # ─── Wait for recording to finish OR stall ───
    max_wait = duration + 120
    last_gt = 0; stall = 0
    t0 = time.time()
    while time.time() - t0 < max_wait:
        time.sleep(5)
        # Check recording state
        try:
            r = P.replay_get("/replay/recording")
            if r and not r.get("recording", False):
                print(f"  Recording complete at wall={time.time()-t0:.0f}s", flush=True)
                break
        except:
            if P.find_league_pid() is None:
                print("  Game process exited", flush=True); break
        # Stall detection from mem thread
        if mem_data:
            cur_gt = mem_data[-1]["gt"]
            if cur_gt > 0 and abs(cur_gt - last_gt) < 2.0:
                stall += 1
                if stall >= 4:
                    print(f"  Game-time stalled at gt={cur_gt:.0f}s — stopping", flush=True)
                    break
            else: stall = 0
            last_gt = cur_gt
    else:
        print(f"  Recording timeout ({max_wait}s)", flush=True)

    # Stop scrape threads
    stop.set()
    mt.join(timeout=5)
    if ck: ck.join(timeout=5)
    end_t = time.time()
    wall = end_t - overall_start
    print(f"  click thread captured {len(all_clicks)} clicks, {len(all_casts)} casts", flush=True)

    # ─── Synthesize cam_data from hero pos (cam is locked to Garen, so
    #     cam_x = hero_x + dx_offset, cam_z = hero_z + dz_offset). Offsets
    #     measured from baseline pass1 cam (no recording, real cam): the
    #     cam parks ~1290u behind hero (the fixed tilt offset) and ~26u to
    #     the side. cam_y is constant when locked.
    TILT_DX = 26.0
    TILT_DZ = -1290.0
    CAM_Y   = 1911.0
    for s in mem_data:
        h = s.get("heroes", {}).get(champion) or {}
        p = h.get("pos")
        if not p or len(p) < 2: continue
        cam_data.append({
            "wall": s["wall"],
            "cx": round(p[0] + TILT_DX, 1),
            "cy": CAM_Y,
            "cz": round(p[1] + TILT_DZ, 1),
        })
    print(f"  Synthesized {len(cam_data)} cam samples from mem hero pos.", flush=True)

    # Collect stats
    pngs = sorted(glob.glob(os.path.join(staging_dir, "**", "*.png"), recursive=True))
    n_frames = len(pngs)

    mem_stats = {}
    if len(mem_data) >= 2:
        mw = [s["wall"] for s in mem_data]
        mem_span = mw[-1] - mw[0]
        mem_hz = len(mem_data) / mem_span if mem_span > 0 else 0
        mem_max_gap = max(mw[i+1]-mw[i] for i in range(len(mw)-1))
        gt_span = mem_data[-1]["gt"] - mem_data[0]["gt"]
        eff_speed = gt_span / mem_span if mem_span > 0 else 0
        mem_stats = {"mem_n": len(mem_data), "mem_hz": round(mem_hz, 1),
                     "mem_max_gap": round(mem_max_gap, 4),
                     "gt_span": round(gt_span, 1), "effective_speed": round(eff_speed, 2)}

    cam_stats = {}
    if len(cam_data) >= 2:
        cw = [s["wall"] for s in cam_data]
        cam_span = cw[-1] - cw[0]
        cam_hz = len(cam_data) / cam_span if cam_span > 0 else 0
        cam_max_gap = max(cw[i+1]-cw[i] for i in range(len(cw)-1))
        cam_stats = {"cam_n": len(cam_data), "cam_hz": round(cam_hz, 1),
                     "cam_max_gap": round(cam_max_gap, 4)}

    rec_wall = end_t - rec_started
    eff_fps = n_frames / rec_wall if rec_wall > 0 else 0

    stats = {
        "wall_total_s": round(wall, 1),
        "rec_wall_s": round(rec_wall, 1),
        "frames_recorded": n_frames,
        "effective_fps": round(eff_fps, 1),
        **mem_stats, **cam_stats,
    }
    stats["clicks_n"] = len(all_clicks)
    stats["casts_n"] = len(all_casts)
    print(f"\n  === merged pass stats ===", flush=True)
    for k, v in stats.items(): print(f"    {k} = {v}")
    click_out = {
        "champion": champion,
        "hero_addr": hex(hero_ptr_int),
        "module_base": hex(base),
        "vtable_addr": hex(target_vptr),
        "watched_addrs": [hex(a) for a in watched_addrs],
        "total_clicks": len(all_clicks),
        "clicks": all_clicks,
        "total_casts": len(all_casts),
        "casts": all_casts,
    }
    m.close()
    try: ctypes.windll.kernel32.CloseHandle(h_click)
    except: pass
    P.kill_game()
    return n_frames, mem_data, cam_data, stats, click_out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--match-id", required=True)
    ap.add_argument("--team", required=True, choices=["blue","red"])
    ap.add_argument("--slot", type=int, required=True)
    ap.add_argument("--champion", default="Garen")
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--rec-start", type=float, default=1.0)
    ap.add_argument("--rec-end", type=float, default=None)
    ap.add_argument("--force-patch", action="store_true")
    ap.add_argument("--out-suffix", default="_merged")
    ap.add_argument("--fps-mult", type=int, default=2,
                    help="encoder framesPerSecond = FPS * fps_mult (default 2 → 40fps)")
    ap.add_argument("--py-cores", default=None,
                    help="comma-separated core IDs for python (default: last core only)")
    ap.add_argument("--speed", type=float, default=2.0,
                    help="replay speed (1.0 = real-time, 2.0 = double, etc.)")
    args = ap.parse_args()

    py_cores = None
    if args.py_cores:
        py_cores = [int(x) for x in args.py_cores.split(",") if x.strip()]

    cam_key = P.cam_key_for(args.team, args.slot)
    out_dir = os.environ.get("REPLAY_OUTPUT", r"C:\tmp\replay_data")
    temp_dir = os.environ.get("REPLAY_TEMP", r"C:\tmp\_pipeline_temp")
    test_match_id = f"{args.match_id}{args.out_suffix}"
    game_dir = os.path.join(out_dir, test_match_id)
    staging = os.path.join(temp_dir, test_match_id)   # separate from final dir
    os.makedirs(game_dir, exist_ok=True)
    os.makedirs(staging, exist_ok=True)

    print(f"=== MERGED TEST: {args.match_id} {args.team} slot={args.slot} key={cam_key} ===", flush=True)

    n_frames, mem_data, cam_data, stats, click_out = merged_pass(
        args.game_id, cam_key, args.champion, args.duration, staging,
        rec_start=args.rec_start, rec_end=args.rec_end, force_patch=args.force_patch,
        fps_mult=args.fps_mult, py_cores=py_cores, speed=args.speed,
    )

    if not n_frames:
        print("FAIL: no frames recorded"); return 1

    # Save clicks alongside other data
    with open(os.path.join(game_dir, "clicks.json"), "w") as f:
        json.dump(click_out, f, indent=2)
    # Also write to the canonical location used by pipeline_to_overlay.py
    with open(r"C:\tmp\garen_clicks_vtable.json", "w") as f:
        json.dump(click_out, f, indent=2)
    print(f"  wrote clicks.json ({click_out['total_clicks']} clicks, {click_out['total_casts']} casts)", flush=True)

    # Save raw scrape data
    with open(os.path.join(game_dir, "raw_mem.json"), "w") as f:
        json.dump(mem_data, f)
    with open(os.path.join(game_dir, "raw_cam.json"), "w") as f:
        json.dump(cam_data, f)
    with open(os.path.join(game_dir, "merged_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  wrote raw_mem.json, raw_cam.json, merged_stats.json", flush=True)

    # Run post-process to build labels.json into the _merged dir (NOT baseline)
    game_info = {"match_id": test_match_id, "champion": args.champion,
                 "team": args.team, "slot": args.slot, "cam_key": cam_key}
    print("\n--- Post-Processing ---", flush=True)
    P.post_process(test_match_id, mem_data, cam_data, game_info, staging,
                   rec_start=args.rec_start)
    print(f"\nDONE merged test → {game_dir}", flush=True)
    return 0

if __name__ == "__main__":
    sys.exit(main())
