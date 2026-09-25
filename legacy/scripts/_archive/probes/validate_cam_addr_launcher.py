"""Launch replay, run find_cam_addr to get fresh candidates, then validate them
during a recording session.
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
    ap.add_argument("--speed", type=float, default=2.0,
                    help="replay speed during validation (cam should be moving)")
    args = ap.parse_args()

    cam_key = P.cam_key_for(args.team, args.slot)
    P.CHAMPION = args.champion

    print(f"=== launch + lock + find_cam_addr + validate @ {args.speed}x ===", flush=True)
    P.kill_game()
    if not P.launch_replay(args.game_id):
        print("launch failed"); return 1
    time.sleep(5)
    if not P.find_league_pid():
        print("no PID"); return 1

    # Cam-lock recipe at requested speed.
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
    time.sleep(2.0)

    here = os.path.dirname(os.path.abspath(__file__))

    # Step 1: scan for cam candidates
    print("\n>>> scanning for cam candidates")
    rc = subprocess.call([sys.executable, "-u", os.path.join(here, "find_cam_addr.py"),
                          "--rounds", "3", "--motion-wait", "3"])
    if rc != 0:
        print(f"find_cam_addr failed: rc={rc}"); P.kill_game(); return rc

    # Step 2: validate during recording
    print("\n>>> validating candidates during recording")
    rc = subprocess.call([sys.executable, "-u", os.path.join(here, "validate_cam_addr.py"),
                          "--rec-duration", "20", "--pre-duration", "5"])
    P.kill_game()
    return rc

if __name__ == "__main__":
    sys.exit(main())
