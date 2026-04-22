"""Keylogger for live game click-destination validation.

Captures right-clicks with wall-clock timestamps during a live League game.
Writes {wall_time, key, game_time_estimate} tuples for later alignment to replay API data.
"""
import json
import time
from pynput import mouse, keyboard
from threading import Event, Thread
from datetime import datetime

# Output file for keystroke log
LOG_FILE = r"C:\tmp\clicks.json"
CALIBRATION_FILE = r"C:\tmp\calibration.json"

clicks = []
calibration = None

def on_move(x, y):
    pass

def on_click(x, y, button, pressed):
    global clicks
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
        print(f"[RIGHT-CLICK] wall_time={wall_time:.4f} pos=({x},{y})")

def on_scroll(x, y, dx, dy):
    pass

def on_press(key):
    """Capture any key press for calibration."""
    global calibration
    if calibration is not None:
        return
    try:
        k = key.char if hasattr(key, 'char') else str(key)
        wall_time = time.time()
        calibration = {
            "wall_time": round(wall_time, 4),
            "wall_time_iso": datetime.fromtimestamp(wall_time).isoformat(),
            "key": k,
        }
        print(f"[CALIBRATION] Wall-time recorded: {calibration['wall_time']}")
        print(f"Please enter the GAME CLOCK time (in seconds) when you pressed that key.")
        print("Wait for the prompt, then type and press Enter.")
    except Exception as e:
        pass

def on_release(key):
    pass

def main():
    global clicks, calibration

    print("=" * 70)
    print("League Click-Destination Validation Keylogger")
    print("=" * 70)
    print(f"Output: {LOG_FILE}")
    print(f"Calibration: {CALIBRATION_FILE}")
    print()
    print("Instructions:")
    print("1. Start this script BEFORE launching the live game")
    print("2. During the match, play normally (clicks will be logged)")
    print("3. At ANY point after 0:10 in-game, press any key once")
    print("   - This will capture wall-clock time for calibration")
    print("   - You'll be prompted to enter the in-game clock value")
    print("4. Let the script run until the match ends")
    print("5. Script will auto-save on exit (Ctrl+C)")
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

    print("Logging started. Waiting for keypress to calibrate...")
    print()

    try:
        while True:
            time.sleep(0.1)

            # If calibration was triggered, wait for user input
            if calibration and calibration.get("game_time") is None:
                try:
                    gt = input("\nEnter game clock time (seconds): ").strip()
                    if gt:
                        calibration["game_time"] = float(gt)
                        with open(CALIBRATION_FILE, "w") as f:
                            json.dump(calibration, f, indent=2)
                        print(f"Calibration saved: wall_time={calibration['wall_time']:.4f} -> game_time={calibration['game_time']:.2f}")
                        print("Continue playing normally. Right-clicks are being logged.\n")
                except ValueError:
                    print("Invalid input. Try again.")

    except KeyboardInterrupt:
        print("\n\nShutting down...")
        mouse_listener.stop()
        keyboard_listener.stop()

        # Save clicks log
        with open(LOG_FILE, "w") as f:
            json.dump(clicks, f, indent=2)
        print(f"Saved {len(clicks)} clicks to {LOG_FILE}")

        if calibration:
            with open(CALIBRATION_FILE, "w") as f:
                json.dump(calibration, f, indent=2)
            print(f"Saved calibration to {CALIBRATION_FILE}")

        print("\nNext steps:")
        print("1. Save/close the match (replay file will be .rofl)")
        print("2. Reboot into Vanguard-disabled Windows")
        print("3. Copy replay to /tmp/ via: scp windows:<path> /tmp/")
        print("4. Run: python scripts/analyze_trajectory_vs_clicks.py <replay_id>")

if __name__ == "__main__":
    main()
