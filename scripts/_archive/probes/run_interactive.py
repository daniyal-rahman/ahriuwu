#!/usr/bin/env python3
"""Launch process_replays.py in the Windows interactive desktop session.

Run this from SSH. It creates a scheduled task that executes in the
logged-in user's desktop session (where the game window lives), so
keyboard/mouse automation actually reaches the game.

Usage:
    python run_interactive.py [process_replays.py args...]

    # Example:
    python run_interactive.py --replay "C:\path\to.rofl" -o C:\output --speed 3
"""

import os
import subprocess
import sys
import time

SCRIPT_PATH = r"C:\Users\daniz\process_replays.py"
LOG_PATH = r"C:\Users\daniz\replay_pipeline.log"
TASK_NAME = "ReplayPipeline"


def main():
    # Build the command that will run in the interactive session
    # Write a .bat file to avoid quoting hell with schtasks /TR
    args_str = " ".join(f'"{a}"' if " " in a else a for a in sys.argv[1:])
    bat_path = r"C:\Users\daniz\replay_pipeline.bat"
    with open(bat_path, "w") as f:
        f.write(f'python -u "{SCRIPT_PATH}" {args_str} > "{LOG_PATH}" 2>&1\n')
    bat_cmd = bat_path

    # Delete old task if exists
    subprocess.run(
        ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
        capture_output=True,
    )

    # Delete old log
    if os.path.exists(LOG_PATH):
        os.remove(LOG_PATH)

    # Create scheduled task that runs immediately in the interactive session
    # /IT = interactive only, /RL HIGHEST = run with highest privileges
    result = subprocess.run(
        [
            "schtasks", "/Create",
            "/TN", TASK_NAME,
            "/TR", bat_cmd,
            "/SC", "ONCE",
            "/ST", "00:00",  # dummy time, we'll run it manually
            "/IT",           # interactive session only
            "/RL", "HIGHEST",
            "/F",            # force overwrite
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Failed to create task: {result.stderr}")
        sys.exit(1)

    print(f"Created task '{TASK_NAME}'")

    # Run the task now
    result = subprocess.run(
        ["schtasks", "/Run", "/TN", TASK_NAME],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"Failed to run task: {result.stderr}")
        sys.exit(1)

    print(f"Task started. Tailing log: {LOG_PATH}")
    print("=" * 60)

    # Tail the log file
    last_pos = 0
    idle_count = 0
    while True:
        try:
            if os.path.exists(LOG_PATH):
                with open(LOG_PATH, "r") as f:
                    f.seek(last_pos)
                    new_data = f.read()
                    if new_data:
                        print(new_data, end="", flush=True)
                        last_pos = f.tell()
                        idle_count = 0
                    else:
                        idle_count += 1
            else:
                idle_count += 1

            # Check if task is still running
            if idle_count > 0 and idle_count % 30 == 0:
                status = subprocess.run(
                    ["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST"],
                    capture_output=True, text=True,
                )
                if "Running" not in status.stdout:
                    # Print any remaining output
                    if os.path.exists(LOG_PATH):
                        with open(LOG_PATH, "r") as f:
                            f.seek(last_pos)
                            remaining = f.read()
                            if remaining:
                                print(remaining, end="", flush=True)
                    print("\n" + "=" * 60)
                    print("Task finished.")
                    break

            time.sleep(1)
        except KeyboardInterrupt:
            print("\nInterrupted — task may still be running in background")
            print(f"Check with: schtasks /Query /TN {TASK_NAME}")
            print(f"Kill with: schtasks /End /TN {TASK_NAME}")
            break

    # Cleanup
    subprocess.run(
        ["schtasks", "/Delete", "/TN", TASK_NAME, "/F"],
        capture_output=True,
    )


if __name__ == "__main__":
    main()
