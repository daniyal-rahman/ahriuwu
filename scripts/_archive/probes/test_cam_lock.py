"""Minimal test: launch replay, set selection, trigger cam lock via schtasks /IT,
verify camera position actually moved to Bel'Veth.

No recording, no memory reads — just checks cam lock works across session 0→1.
"""
import sys, os, time, subprocess, json, ssl, urllib.request
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pipeline as P

_ctx = ssl.create_default_context(); _ctx.check_hostname=False; _ctx.verify_mode=ssl.CERT_NONE
def replay_get(ep):
    with urllib.request.urlopen(f"https://127.0.0.1:2999{ep}", context=_ctx, timeout=3) as r:
        return json.loads(r.read())
def replay_post(ep, data):
    req = urllib.request.Request(f"https://127.0.0.1:2999{ep}",
        data=json.dumps(data).encode(), headers={"Content-Type":"application/json"}, method="POST")
    with urllib.request.urlopen(req, context=_ctx, timeout=3) as r: return json.loads(r.read())

GAME_ID = "5545727197"
CAM_KEY = "2"

def cam_lock_via_schtasks(key):
    subprocess.run(['schtasks', '/End', '/TN', 'LockCam'], capture_output=True)
    subprocess.run(['schtasks', '/Delete', '/TN', 'LockCam', '/F'], capture_output=True)
    r = subprocess.run(['schtasks', '/Create', '/TN', 'LockCam',
            '/TR', f'cmd.exe /c python C:\\Users\\daniz\\lock_cam_once.py {key} > C:\\tmp\\lock_cam.log 2>&1',
            '/SC', 'ONCE', '/ST', '23:59', '/IT', '/F'], capture_output=True, text=True)
    print("schtasks create:", r.returncode, r.stdout, r.stderr)
    r = subprocess.run(['schtasks', '/Run', '/TN', 'LockCam'], capture_output=True, text=True)
    print("schtasks run:", r.returncode, r.stdout, r.stderr)

def main():
    # Check if game is alive
    pid = P.find_league_pid()
    if not pid:
        print("No game running, launching...")
        P.kill_game(); time.sleep(2)
        if not P.launch_replay(GAME_ID):
            print("launch failed"); return
        time.sleep(5)
    else:
        print(f"Game already running (PID={pid})")

    # Read cam position BEFORE lock
    st0 = replay_get("/replay/render")
    cp0 = st0["cameraPosition"]
    print(f"cam BEFORE lock: ({cp0['x']:.0f},{cp0['y']:.0f},{cp0['z']:.0f})")
    pb0 = replay_get("/replay/playback")
    print(f"playback: time={pb0['time']:.1f} paused={pb0['paused']}")

    # Make sure replay is playing (cam lock requires it)
    if pb0["paused"]:
        replay_post("/replay/playback", {"time": 5.0, "speed": 1.0, "paused": False})
        time.sleep(3)

    # Set selection
    replay_post("/replay/render", {"interfaceAll": False, "selectionName": "Belveth"})
    time.sleep(0.5)

    # Send lock via schtasks
    cam_lock_via_schtasks(CAM_KEY)
    time.sleep(3)

    # Read cam position AFTER lock
    st1 = replay_get("/replay/render")
    cp1 = st1["cameraPosition"]
    print(f"cam AFTER lock: ({cp1['x']:.0f},{cp1['y']:.0f},{cp1['z']:.0f})")

    # Wait a bit more and read again (cam should follow Bel'Veth as she moves)
    time.sleep(3)
    st2 = replay_get("/replay/render")
    cp2 = st2["cameraPosition"]
    print(f"cam after 3s : ({cp2['x']:.0f},{cp2['y']:.0f},{cp2['z']:.0f})")

    # Dump lock_cam log
    try:
        with open(r"C:\tmp\lock_cam.log") as f:
            print("\n--- lock_cam.log ---")
            print(f.read())
    except:
        print("no lock_cam.log")

    # Evaluate
    moved = abs(cp1["x"] - cp0["x"]) + abs(cp1["z"] - cp0["z"])
    print(f"\nCam moved between before/after: {moved:.0f} units")
    if moved > 500:
        print("CAM LOCK WORKED")
    else:
        print("CAM LOCK FAILED (didn't move)")

if __name__ == "__main__":
    main()
