"""Validate cam-addr candidates: which ones keep updating during /replay/recording?

Reads each candidate address + /replay/render at high frequency. Starts recording
mid-run. Reports per-candidate:
  - did the value change from pre-recording values?
  - how much did it diverge from /replay/render once recording started?

The "true" cam addr is one that keeps updating even when /replay/render goes stale.

Usage:
    # Game must already be running, cam-locked, with candidates from
    # find_cam_addr.py output at C:\\tmp\\cam_addr_candidates.json
    python scripts/validate_cam_addr.py [--rec-duration 30]
"""
import os, sys, time, ctypes, json, argparse, struct
from ctypes import wintypes as wt
import http.client, ssl
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P
from find_cam_addr import find_pid, _k, PROCESS_VM_READ, PROCESS_QUERY_INFORMATION, read_at, replay_get, replay_post

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates-json", default=r"C:\tmp\cam_addr_candidates.json")
    ap.add_argument("--rec-duration", type=float, default=20.0,
                    help="seconds to record and observe candidates")
    ap.add_argument("--pre-duration", type=float, default=5.0,
                    help="seconds to observe BEFORE starting recording")
    ap.add_argument("--poll-hz", type=float, default=20.0)
    ap.add_argument("--rec-path", default=r"C:\tmp\_cam_validate_rec")
    args = ap.parse_args()

    cands = json.load(open(args.candidates_json))
    addrs = [int(c["addr"], 16) for c in cands["candidates"]]
    print(f"Validating {len(addrs)} candidate addresses")

    pid = find_pid()
    if not pid: print("League not running"); return 1
    h = _k.OpenProcess(PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid)
    if not h: print(f"OpenProcess failed: {ctypes.get_last_error()}"); return 1

    # Make sure replay is unpaused & moving (find_cam_addr leaves it paused).
    replay_post("/replay/playback", {"paused": False, "speed": 2.0})
    time.sleep(1.0)

    # Get current game time so we can correctly schedule recording start.
    pb = replay_get("/replay/playback")
    cur_gt = pb.get("currentTime", 0)
    print(f"current game time: {cur_gt:.1f}s")

    os.makedirs(args.rec_path, exist_ok=True)
    samples = []   # list of (wall, mode, api_cam, [cand_cam, ...])
    period = 1.0 / args.poll_hz

    try:
        # Phase 1: pre-recording
        print(f"\n--- phase 1: PRE-recording ({args.pre_duration}s) ---")
        t0 = time.time()
        while time.time() - t0 < args.pre_duration:
            try:
                cp = replay_get("/replay/render").get("cameraPosition", {})
                api = (cp.get("x"), cp.get("y"), cp.get("z"))
            except: api = None
            cands_now = []
            for a in addrs:
                b = read_at(h, a, 12)
                if len(b) == 12:
                    cands_now.append(struct.unpack("<fff", b))
                else:
                    cands_now.append(None)
            samples.append((time.time()-t0, "pre", api, cands_now))
            time.sleep(period)

        # Phase 2: start recording, observe
        print(f"\n--- phase 2: RECORDING ({args.rec_duration}s) ---")
        for f in os.listdir(args.rec_path):
            try: os.remove(os.path.join(args.rec_path, f))
            except: pass
        # Use current game time + tiny epsilon so recording starts immediately.
        pb_now = replay_get("/replay/playback")
        gt_now = pb_now.get("currentTime", cur_gt)
        rec_start = gt_now + 0.5
        rec_end   = gt_now + args.rec_duration * 2 + 60  # plenty of headroom @ 2x
        rec_resp = replay_post("/replay/recording", {
            "recording": True,
            "path": args.rec_path.replace("\\","/"),
            "codec": "png",
            "framesPerSecond": 40,
            "startTime": rec_start,
            "endTime": rec_end,
            "enforceFrameRate": True,
        })
        print(f"  rec resp: {rec_resp}")
        rec_t0 = time.time()
        while time.time() - rec_t0 < args.rec_duration:
            try:
                cp = replay_get("/replay/render").get("cameraPosition", {})
                api = (cp.get("x"), cp.get("y"), cp.get("z"))
            except: api = None
            cands_now = []
            for a in addrs:
                b = read_at(h, a, 12)
                if len(b) == 12:
                    cands_now.append(struct.unpack("<fff", b))
                else:
                    cands_now.append(None)
            samples.append((time.time()-t0, "rec", api, cands_now))
            time.sleep(period)
        # Stop recording
        replay_post("/replay/recording", {"recording": False})

        # ─── Analyze ───
        pre = [s for s in samples if s[1] == "pre"]
        rec = [s for s in samples if s[1] == "rec"]
        print(f"\n=== analysis: {len(pre)} pre-samples, {len(rec)} rec-samples ===")

        # Did API stale? — measure variance of api cam during pre vs rec
        def variance(vals):
            xs = [v[0] for v in vals if v]
            return (max(xs)-min(xs)) if xs else 0
        api_var_pre = variance([s[2] for s in pre])
        api_var_rec = variance([s[2] for s in rec])
        print(f"\napi cam.x range: pre={api_var_pre:.1f}u  rec={api_var_rec:.1f}u")
        if api_var_rec < 5 and api_var_pre > 50:
            print("  -> API confirmed STALE during recording")
        elif api_var_rec > 50:
            print("  -> API still varies during recording (this game might not show the bug)")

        # For each candidate: variance during recording (live cams should keep moving)
        print(f"\nper-candidate cam.x range during recording:")
        ranking = []
        for i, a in enumerate(addrs):
            xs_pre = [s[3][i][0] for s in pre if s[3][i]]
            xs_rec = [s[3][i][0] for s in rec if s[3][i]]
            v_pre = (max(xs_pre)-min(xs_pre)) if xs_pre else 0
            v_rec = (max(xs_rec)-min(xs_rec)) if xs_rec else 0
            ranking.append((v_rec, v_pre, a))
        ranking.sort(reverse=True)
        for v_rec, v_pre, a in ranking[:15]:
            print(f"  0x{a:X}  pre={v_pre:7.1f}u  rec={v_rec:7.1f}u  (rec/pre={v_rec/max(v_pre,0.001):.2f})")
        print(f"\n  ... (showing top 15 of {len(addrs)} by rec-variance)")

        # Save
        out = {
            "api_var_pre": api_var_pre, "api_var_rec": api_var_rec,
            "per_candidate": [{"addr": hex(a), "var_pre": vp, "var_rec": vr}
                              for vr, vp, a in ranking],
            "n_pre": len(pre), "n_rec": len(rec),
        }
        with open(r"C:\tmp\cam_addr_validation.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nWrote C:\\tmp\\cam_addr_validation.json")
    finally:
        _k.CloseHandle(h)
    return 0

if __name__ == "__main__":
    sys.exit(main())
