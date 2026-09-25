"""Keylogger for live game click-destination validation.

Captures right-clicks with wall-clock timestamps during a live League game.
Writes {wall_time, key, game_time_estimate} tuples for later alignment to replay API data.
"""
import json
import os
import sys
import time
import traceback
from pynput import mouse, keyboard
from threading import Event, Thread, Lock
from datetime import datetime
import mss
from PIL import Image

CAL_SHOT_COUNT = 10
CAL_DIR = r"C:\tmp\keycal"
CAL_META = r"C:\tmp\keycal_meta.json"
os.makedirs(CAL_DIR, exist_ok=True)
# Clear previous shots so we don't confuse stale runs
for old in os.listdir(CAL_DIR):
    try: os.remove(os.path.join(CAL_DIR, old))
    except Exception: pass
_shot_lock = Lock()
_shots_taken = 0
_shot_records = []

def _grab_screen(path):
    """Capture primary monitor to `path`. Returns (ok, mean_brightness)."""
    with mss.mss() as sct:
        mon = sct.monitors[1]  # primary
        raw = sct.grab(mon)
        img = Image.frombytes("RGB", raw.size, raw.rgb)
        img.save(path, "PNG", optimize=False)
        # Sample brightness to detect black-screen (DX exclusive fullscreen failure)
        small = img.resize((40, 22))
        px = list(small.getdata())
        mean = sum(sum(p) for p in px) / (len(px) * 3)
        return mean

DEBUG_LOG = r"C:\tmp\keylog_debug.log"

def _dbg(msg):
    try:
        with open(DEBUG_LOG, "a") as f:
            f.write(f"[{time.time():.3f}] {msg}\n")
            f.flush()
    except Exception:
        pass

_dbg(f"=== startup pid={os.getpid()} argv={sys.argv} ===")

# Output file for keystroke log
LOG_FILE = r"C:\tmp\clicks.json"

clicks = []

def on_move(x, y):
    pass

def on_click(x, y, button, pressed):
    global clicks
    _dbg(f"on_click pressed={pressed} button={button} x={x} y={y}")
    if not pressed:
        return
    if button == mouse.Button.right:
        wall_time = time.time()
        clicks.append({
            "wall_time": round(wall_time, 4),
            "wall_time_iso": datetime.fromtimestamp(wall_time).isoformat(),
            "button": "right",
            "x": x,
            "y": y,
        })
        _dbg(f"RIGHT-CLICK logged count={len(clicks)} pos=({x},{y})")
        print(f"[RIGHT-CLICK] wall_time={wall_time:.4f} pos=({x},{y})")

def on_scroll(x, y, dx, dy):
    pass

def on_press(key):
    """Capture screenshot on each of the first CAL_SHOT_COUNT key presses.

    Each shot pairs a precise wall_time with the image of the screen, which
    (offline) we OCR/read for in-game clock → build wall↔game_time linear map.
    """
    global _shots_taken, _shot_records
    try:
        k = key.char if hasattr(key, 'char') else str(key)
    except Exception:
        k = "?"
    with _shot_lock:
        if _shots_taken >= CAL_SHOT_COUNT:
            return
        idx = _shots_taken
        wall_time = time.time()
        path = os.path.join(CAL_DIR, f"keycal_{idx:02d}_{wall_time:.3f}.png")
        try:
            brightness = _grab_screen(path)
        except Exception as e:
            _dbg(f"screenshot err idx={idx}: {e}")
            return
        rec = {
            "idx": idx,
            "wall_time": round(wall_time, 4),
            "wall_time_iso": datetime.fromtimestamp(wall_time).isoformat(),
            "key": k,
            "path": path,
            "mean_brightness": round(brightness, 1),
        }
        _shot_records.append(rec)
        _shots_taken += 1
        try:
            with open(CAL_META, "w") as f:
                json.dump(_shot_records, f, indent=2)
        except Exception as e:
            _dbg(f"cal meta write err: {e}")
        black = brightness < 4.0
        _dbg(f"cal shot {idx}/{CAL_SHOT_COUNT} wall={wall_time:.3f} key={k} brightness={brightness:.1f}{' BLACK!' if black else ''}")
        print(f"[CAL {idx+1}/{CAL_SHOT_COUNT}] key={k} wall={wall_time:.3f} brightness={brightness:.1f}{' BLACK!' if black else ''}")

def on_release(key):
    pass

def main():
    global clicks

    print("=" * 70)
    print("League Click-Destination Validation Keylogger")
    print("=" * 70)
    print(f"Right-clicks: {LOG_FILE}")
    print(f"Cal screenshots: {CAL_DIR}  (first {CAL_SHOT_COUNT} key presses)")
    print(f"Cal meta:     {CAL_META}")
    print()
    print("Usage:")
    print("  - Play normally. Right-clicks are logged.")
    print(f"  - After 0:10 in-game, spam any key {CAL_SHOT_COUNT}+ times, a few sec apart.")
    print("    Each press snaps a screenshot for offline clock→wall_time calibration.")
    print("  - Stop with Ctrl+C in this window OR by creating C:\\tmp\\keylogger_stop.")
    print()

    # Set up listeners
    mouse_listener = mouse.Listener(
        on_move=on_move,
        on_click=on_click,
        on_scroll=on_scroll
    )
    keyboard_listener = keyboard.Listener(
        on_press=on_press,
        on_release=on_release
    )

    mouse_listener.start()
    keyboard_listener.start()
    _dbg(f"listeners started mouse_alive={mouse_listener.is_alive()} kb_alive={keyboard_listener.is_alive()}")

    print("Logging started.")
    print()

    STOP_FILE = r"C:\tmp\keylogger_stop"
    last_saved_count = -1
    last_heartbeat = 0

    try:
        while True:
            time.sleep(0.5)
            now = time.time()
            if now - last_heartbeat > 5:
                _dbg(f"heartbeat clicks={len(clicks)} cal_shots={_shots_taken}/{CAL_SHOT_COUNT} mouse_alive={mouse_listener.is_alive()} kb_alive={keyboard_listener.is_alive()}")
                last_heartbeat = now

            # Incremental save of clicks (every 0.5s) — so logs survive abrupt kill
            if len(clicks) != last_saved_count:
                with open(LOG_FILE, "w") as f:
                    json.dump(clicks, f, indent=2)
                last_saved_count = len(clicks)

            # Clean exit when a sentinel file appears
            if os.path.exists(STOP_FILE):
                os.remove(STOP_FILE)
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        mouse_listener.stop()
        keyboard_listener.stop()

        with open(LOG_FILE, "w") as f:
            json.dump(clicks, f, indent=2)
        print(f"Saved {len(clicks)} right-clicks to {LOG_FILE}")
        print(f"Saved {_shots_taken} cal screenshots to {CAL_DIR}")
        with open(CAL_META, "w") as f:
            json.dump(_shot_records, f, indent=2)
        print(f"Cal metadata at {CAL_META}")

if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        _dbg("SystemExit")
        raise
    except BaseException:
        _dbg(f"FATAL in main():\n{traceback.format_exc()}")
        raise
    finally:
        _dbg("main() returned/exiting")
