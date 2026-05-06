"""Launch a replay, lock cam, run find_cam_addr scanner.

Usage (via schtasks /IT):
    python scripts/find_cam_addr_launcher.py --game-id 5547184086 --champion Garen --slot 0 --team blue
"""
import os, sys, time, argparse, subprocess
sys.stdout.reconfigure(encoding='utf-8', errors='replace', line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game-id", required=True)
    ap.add_argument("--champion", default="Garen")
    ap.add_argument("--team", default="blue", choices=["blue", "red"])
    ap.add_argument("--slot", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="replay speed (1x makes locked cam track hero closely; higher speeds add lag)")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--motion-wait", type=float, default=5.0)
    args = ap.parse_args()

    cam_key = P.cam_key_for(args.team, args.slot)
    P.CHAMPION = args.champion

    print(f"=== launch + lock + cam-scan @ {args.speed}x ===", flush=True)
    P.kill_game()
    if not P.launch_replay(args.game_id):
        print("launch failed"); return 1
    time.sleep(5)

    pid = P.find_league_pid()
    if not pid: print("no PID"); return 1

    # Cam-lock recipe (same as pass1) at requested speed.
    P.replay_post("/replay/playback", {"paused": True})
    time.sleep(0.3)
    P.replay_post("/replay/render", {"interfaceAll": True, "selectionName": args.champion})
    time.sleep(0.5)
    P.replay_post("/replay/playback", {"speed": args.speed, "paused": False})
    time.sleep(1.0)
    P.focus_game(); P.lock_camera(cam_key)
    time.sleep(0.5)
    P.focus_game(); P.lock_camera(cam_key)
    print(f"Camera locked (key={cam_key}), {args.speed}x speed", flush=True)
    time.sleep(2.0)  # let cam settle on hero

    # Hand off to the scanner.
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, "-u", os.path.join(here, "find_cam_addr.py"),
           "--rounds", str(args.rounds),
           "--motion-wait", str(args.motion_wait)]
    print(f"Running: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    P.kill_game()
    return rc

if __name__ == "__main__":
    sys.exit(main())
