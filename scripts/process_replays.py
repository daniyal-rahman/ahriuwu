#!/usr/bin/env python3
"""Automated replay processing pipeline.

Runs on Windows desktop. For each .rofl file:
1. Parse ROFL metadata to find Garen's player slot
2. Launch replay via League client
3. Wait for game to load (poll Replay API)
4. Select Garen via keyboard input, lock camera
5. Record video via Replay API (built-in recording)
6. Decode movement data from .rofl file
7. Generate screen-space action labels using projection matrix

Usage:
    # Process all replays in a directory
    python scripts/process_replays.py \
        --replay-dir "C:\\Users\\daniz\\Documents\\League of Legends\\Replays" \
        --output-dir data/processed_replays \
        --projection data/projection_matrix.json

    # Process a single replay
    python scripts/process_replays.py \
        --replay path/to/replay.rofl \
        --output-dir data/processed_replays \
        --projection data/projection_matrix.json

    # Just record (skip movement decoding, e.g. if PE dumps not available)
    python scripts/process_replays.py \
        --replay-dir "..." --output-dir "..." --record-only

Requirements:
    - Windows with League of Legends installed
    - League client running (for replay launch)
    - Replay API enabled: add EnableReplayApi=1 to game.cfg
    - PE section dumps in /tmp/pe_dump/ (for movement decoding, unless --record-only)

Replay API reference (localhost:2999):
    POST /replay/playback  - {paused, time, speed, seeking}
    POST /replay/render    - {cameraAttached, interfaceAll, fogOfWar, ...}
    POST /replay/recording - {recording, path, codec, width, height, framesPerSecond,
                              enforceFrameRate, replaySpeed}
    GET  /replay/playback  - current playback state
    GET  /replay/game      - process info (used to detect when game is loaded)
"""

import argparse
import ctypes
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path

# Replay API settings
REPLAY_API_BASE = "https://127.0.0.1:2999"
REPLAY_API_TIMEOUT = 5  # seconds per request

# Game settings
GAME_EXE = r"C:\Riot Games\League of Legends\Game\League of Legends.exe"
GAME_LOAD_TIMEOUT = 120  # seconds to wait for game to load
GAME_LOAD_POLL_INTERVAL = 3  # seconds between load checks

# Recording settings
RECORD_FPS = 20  # frames per game-second
REPLAY_SPEED = 8  # playback speed multiplier
RECORD_CODEC = "webm"  # webm or png-sequence

# Garen champion name (used to find player slot in metadata)
GAREN_CHAMPION = "Garen"


# ---------------------------------------------------------------------------
# Replay API client
# ---------------------------------------------------------------------------

def replay_api_get(endpoint: str) -> dict | None:
    """GET request to the Replay API."""
    import urllib.request
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    url = f"{REPLAY_API_BASE}{endpoint}"
    try:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, context=ctx, timeout=REPLAY_API_TIMEOUT) as resp:
            return json.loads(resp.read())
    except Exception:
        return None


def replay_api_post(endpoint: str, data: dict) -> dict | None:
    """POST request to the Replay API."""
    import urllib.request
    import ssl

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    url = f"{REPLAY_API_BASE}{endpoint}"
    body = json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method="POST",
                                headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, context=ctx, timeout=REPLAY_API_TIMEOUT) as resp:
            return json.loads(resp.read())
    except Exception as e:
        print(f"  API error on {endpoint}: {e}")
        return None


def wait_for_game_load(timeout: int = GAME_LOAD_TIMEOUT) -> bool:
    """Poll the Replay API until the game is loaded and responsive."""
    print("  Waiting for game to load...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        game = replay_api_get("/replay/game")
        if game and game.get("processID", 0) > 0:
            # Also check that playback is available
            playback = replay_api_get("/replay/playback")
            if playback and playback.get("length", 0) > 0:
                print(f" loaded! (game length: {playback['length']:.0f}s)")
                return True
        print(".", end="", flush=True)
        time.sleep(GAME_LOAD_POLL_INTERVAL)
    print(" TIMEOUT")
    return False


def wait_for_game_end(poll_interval: float = 5.0) -> bool:
    """Poll playback position until the game ends."""
    print("  Waiting for game to end...")
    last_time = -1
    stall_count = 0

    while True:
        playback = replay_api_get("/replay/playback")
        if not playback:
            stall_count += 1
            if stall_count > 10:
                print("  Lost connection to Replay API")
                return False
            time.sleep(poll_interval)
            continue

        stall_count = 0
        current = playback.get("time", 0)
        length = playback.get("length", 0)
        speed = playback.get("speed", 0)

        if length > 0:
            pct = 100 * current / length
            print(f"    {current:.0f}/{length:.0f}s ({pct:.1f}%) speed={speed}x    ",
                  end="\r", flush=True)

            # Game is done when we're near the end
            if current >= length - 5:
                print(f"\n  Game finished at {current:.0f}s")
                return True

        # Detect if playback stalled (paused or ended)
        if current == last_time:
            stall_count += 1
            if stall_count > 6:
                print(f"\n  Playback stalled at {current:.0f}s")
                return True
        else:
            stall_count = 0
        last_time = current

        time.sleep(poll_interval)


# ---------------------------------------------------------------------------
# Player selection via keyboard input
# ---------------------------------------------------------------------------

def select_player_slot(slot: int):
    """Select a player in the replay viewer by simulating key press.

    Slot 0-4 = blue team (keys 1-5), slot 5-9 = red team (keys Q,W,E,R,T or Ctrl+1-5).

    In the League replay viewer, pressing number keys 1-5 selects blue side players
    in tab-order. For red side, we use Ctrl+1 through Ctrl+5.

    NOTE: The exact keybinds may vary. Test with a replay first.
    """
    if sys.platform != "win32":
        print(f"  WARNING: keyboard input only works on Windows, skipping slot selection")
        return

    # Virtual key codes for 1-5
    VK_KEYS = {0: 0x31, 1: 0x32, 2: 0x33, 3: 0x34, 4: 0x35}  # 1-5
    VK_CONTROL = 0x11

    if slot < 5:
        # Blue team: press 1-5
        vk = VK_KEYS[slot]
        _send_key(vk)
    else:
        # Red team: press Ctrl + 1-5
        red_slot = slot - 5
        vk = VK_KEYS[red_slot]
        _send_key(vk, ctrl=True)

    print(f"  Selected player slot {slot} (key: {'Ctrl+' if slot >= 5 else ''}{slot % 5 + 1})")
    time.sleep(0.5)


def _send_key(vk_code: int, ctrl: bool = False):
    """Send a key press using Windows SendInput API."""
    INPUT_KEYBOARD = 1
    KEYEVENTF_KEYUP = 0x0002

    class KEYBDINPUT(ctypes.Structure):
        _fields_ = [
            ("wVk", ctypes.c_ushort),
            ("wScan", ctypes.c_ushort),
            ("dwFlags", ctypes.c_ulong),
            ("time", ctypes.c_ulong),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class INPUT(ctypes.Structure):
        class _INPUT(ctypes.Union):
            _fields_ = [("ki", KEYBDINPUT)]
        _fields_ = [
            ("type", ctypes.c_ulong),
            ("_input", _INPUT),
        ]

    def make_input(vk, flags=0):
        inp = INPUT()
        inp.type = INPUT_KEYBOARD
        inp._input.ki.wVk = vk
        inp._input.ki.dwFlags = flags
        return inp

    inputs = []
    if ctrl:
        inputs.append(make_input(0x11))  # Ctrl down
    inputs.append(make_input(vk_code))  # Key down
    inputs.append(make_input(vk_code, KEYEVENTF_KEYUP))  # Key up
    if ctrl:
        inputs.append(make_input(0x11, KEYEVENTF_KEYUP))  # Ctrl up

    arr = (INPUT * len(inputs))(*inputs)
    ctypes.windll.user32.SendInput(len(inputs), arr, ctypes.sizeof(INPUT))


# ---------------------------------------------------------------------------
# ROFL metadata parsing
# ---------------------------------------------------------------------------

def parse_rofl_metadata(rofl_path: str) -> dict | None:
    """Parse ROFL file header to extract game metadata (players, champions, etc.).

    ROFL2 format:
    - 6 bytes: magic "ROFL2\\0" or "RIOT\\0\\0"
    - Variable header with JSON metadata
    - Compressed game data chunks

    Returns dict with 'players' list (each has 'champion', 'name', 'team', 'slot').
    """
    with open(rofl_path, "rb") as f:
        magic = f.read(6)

        if magic[:4] == b"RIOT":
            # ROFL v1 format
            f.seek(0)
            header = f.read(4096)
        elif magic[:5] == b"ROFL2":
            # ROFL v2 format
            f.seek(0)
            header = f.read(8192)
        else:
            print(f"  Unknown ROFL magic: {magic!r}")
            return None

    # Find JSON metadata in the header
    # Look for the start of a JSON object containing player info
    raw = header.decode("latin-1")
    json_start = raw.find('{"')
    if json_start < 0:
        # Try finding statsJson or similar embedded JSON
        json_start = raw.find("[{")
    if json_start < 0:
        print(f"  Could not find JSON metadata in ROFL header")
        return None

    # Extract JSON (find matching bracket)
    depth = 0
    json_end = json_start
    bracket_char = raw[json_start]
    close_char = "}" if bracket_char == "{" else "]"
    for i in range(json_start, len(raw)):
        if raw[i] == bracket_char:
            depth += 1
        elif raw[i] == close_char:
            depth -= 1
            if depth == 0:
                json_end = i + 1
                break

    try:
        metadata = json.loads(raw[json_start:json_end])
    except json.JSONDecodeError as e:
        print(f"  Failed to parse ROFL JSON metadata: {e}")
        return None

    # Extract player info
    # The metadata format varies but typically has statsJson with player data
    players = []
    if isinstance(metadata, list):
        # statsJson format: list of player stats objects
        for i, p in enumerate(metadata):
            players.append({
                "slot": i,
                "name": p.get("NAME", p.get("SKIN", "?")),
                "champion": p.get("SKIN", p.get("NAME", "?")),
                "team": "blue" if i < 5 else "red",
            })
    elif isinstance(metadata, dict):
        # Top-level metadata format
        stats = metadata.get("statsJson", metadata.get("players", []))
        if isinstance(stats, str):
            stats = json.loads(stats)
        if isinstance(stats, list):
            for i, p in enumerate(stats):
                players.append({
                    "slot": i,
                    "name": p.get("NAME", p.get("summonerName", "?")),
                    "champion": p.get("SKIN", p.get("championName", "?")),
                    "team": "blue" if i < 5 else "red",
                })

    return {
        "players": players,
        "raw": metadata,
    }


def find_garen_slot(metadata: dict) -> int | None:
    """Find Garen's player slot (0-9) from ROFL metadata."""
    for p in metadata.get("players", []):
        champ = p.get("champion", "").lower()
        if champ == GAREN_CHAMPION.lower() or "garen" in champ:
            return p["slot"]
    return None


# ---------------------------------------------------------------------------
# Replay launcher
# ---------------------------------------------------------------------------

def launch_replay(rofl_path: str) -> subprocess.Popen | None:
    """Launch a replay file in the League client."""
    rofl_path = os.path.abspath(rofl_path)

    if not os.path.exists(rofl_path):
        print(f"  Replay file not found: {rofl_path}")
        return None

    print(f"  Launching replay: {os.path.basename(rofl_path)}")

    if sys.platform == "win32":
        # Use os.startfile to open with default handler (League client)
        try:
            os.startfile(rofl_path)
            return True  # startfile doesn't return a process handle
        except OSError as e:
            print(f"  Failed to launch: {e}")
            return None
    else:
        print("  ERROR: Replay launching only works on Windows")
        return None


def kill_game_process():
    """Kill the League of Legends game process."""
    if sys.platform == "win32":
        os.system('taskkill /F /IM "League of Legends.exe" 2>NUL')
    else:
        os.system("pkill -f 'League of Legends' 2>/dev/null")
    time.sleep(2)


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------

def process_single_replay(rofl_path: str, output_dir: str, projection_path: str = None,
                          record_only: bool = False, pe_dump_dir: str = None) -> dict:
    """Process a single replay file.

    Returns dict with processing stats.
    """
    rofl_path = os.path.abspath(rofl_path)
    basename = Path(rofl_path).stem  # e.g., "NA1-5489605032"
    replay_output = Path(output_dir) / basename
    replay_output.mkdir(parents=True, exist_ok=True)

    stats = {"replay": basename, "success": False, "video_path": None, "movements": 0}

    print(f"\n{'='*60}")
    print(f"Processing: {basename}")
    print(f"{'='*60}")

    # Step 1: Parse metadata to find Garen
    print("\n[1/6] Parsing ROFL metadata...")
    metadata = parse_rofl_metadata(rofl_path)
    if metadata:
        garen_slot = find_garen_slot(metadata)
        if garen_slot is not None:
            team = "blue" if garen_slot < 5 else "red"
            print(f"  Garen found at slot {garen_slot} ({team} side)")
        else:
            print(f"  WARNING: Garen not found in replay metadata")
            print(f"  Players: {[p.get('champion') for p in metadata.get('players', [])]}")
            garen_slot = None
    else:
        print("  WARNING: Could not parse metadata, will need manual player selection")
        garen_slot = None

    # Save metadata
    if metadata:
        meta_path = replay_output / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)

    # Step 2: Launch replay
    print("\n[2/6] Launching replay...")
    result = launch_replay(rofl_path)
    if not result:
        print("  FAILED to launch replay")
        return stats

    # Step 3: Wait for game to load
    print("\n[3/6] Waiting for Replay API...")
    if not wait_for_game_load():
        print("  FAILED: game did not load in time")
        kill_game_process()
        return stats

    # Step 4: Configure camera and select Garen
    print("\n[4/6] Configuring camera...")

    # Pause first to set up
    replay_api_post("/replay/playback", {"paused": True})
    time.sleep(1)

    # Select Garen player
    if garen_slot is not None:
        select_player_slot(garen_slot)
    else:
        print("  No Garen slot detected — please manually select Garen in the replay viewer")
        input("  Press Enter when ready...")

    # Lock camera, hide UI, disable fog of war
    render_settings = {
        "cameraAttached": True,
        "fogOfWar": False,
        "interfaceAll": False,      # hide all UI
        "interfaceTimeline": False,
        "interfaceMinimap": False,
        "interfaceScore": False,
        "interfaceChat": False,
        "healthBarChampions": True,
        "healthBarMinions": True,
        "healthBarStructures": True,
    }
    replay_api_post("/replay/render", render_settings)
    time.sleep(0.5)

    # Seek to start (skip loading screen)
    replay_api_post("/replay/playback", {"time": 10.0, "seeking": True})
    time.sleep(1)

    # Step 5: Start recording
    print("\n[5/6] Recording replay...")
    video_dir = str(replay_output / "video")
    os.makedirs(video_dir, exist_ok=True)

    recording_settings = {
        "recording": True,
        "path": video_dir,
        "codec": RECORD_CODEC,
        "framesPerSecond": RECORD_FPS,
        "enforceFrameRate": True,
        "replaySpeed": REPLAY_SPEED,
    }
    replay_api_post("/replay/recording", recording_settings)

    # Unpause to start playback
    replay_api_post("/replay/playback", {"paused": False, "speed": REPLAY_SPEED})

    # Wait for game to end
    game_finished = wait_for_game_end()

    # Stop recording
    print("\n  Stopping recording...")
    replay_api_post("/replay/recording", {"recording": False})
    time.sleep(2)

    stats["video_path"] = video_dir
    stats["success"] = game_finished

    # Kill game process
    print("  Closing game...")
    kill_game_process()

    # Step 6: Decode movement data (optional)
    if not record_only:
        print("\n[6/6] Decoding movement data...")
        try:
            movements = decode_replay_movements(rofl_path, pe_dump_dir)
            if movements:
                mov_path = replay_output / "movements.json"
                with open(mov_path, "w") as f:
                    json.dump(movements, f, default=str)
                stats["movements"] = len(movements)
                print(f"  Decoded {len(movements)} movements")

                # Generate action labels if projection matrix is available
                if projection_path and os.path.exists(projection_path):
                    labels = generate_action_labels(movements, projection_path)
                    labels_path = replay_output / "action_labels.json"
                    with open(labels_path, "w") as f:
                        json.dump(labels, f)
                    print(f"  Generated {len(labels)} action labels")
        except Exception as e:
            print(f"  Movement decoding failed: {e}")
    else:
        print("\n[6/6] Skipped movement decoding (--record-only)")

    return stats


def decode_replay_movements(rofl_path: str, pe_dump_dir: str = None) -> list[dict]:
    """Decode movement data from a .rofl file using the movement decoder.

    This is a wrapper around decode_replay_movement.py's MovementDecoder.
    """
    # Add scripts dir to path
    scripts_dir = str(Path(__file__).parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    try:
        from decode_replay_movement import MovementDecoder
    except ImportError:
        print("  ERROR: Could not import MovementDecoder")
        print("  Make sure scripts/decode_replay_movement.py exists")
        return []

    pe_dir = pe_dump_dir or "/tmp/pe_dump"
    text_path = os.path.join(pe_dir, "text.bin")
    rdata_path = os.path.join(pe_dir, "rdata.bin")
    data_path = os.path.join(pe_dir, "data.bin")

    for p in [text_path, rdata_path, data_path]:
        if not os.path.exists(p):
            print(f"  ERROR: PE dump not found: {p}")
            return []

    # Parse replay blocks
    import pickle
    # TODO: implement ROFL block parser here instead of relying on cached pkl
    # For now, check if a cached blocks file exists
    cache_path = rofl_path + ".blocks.pkl"
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            blocks = pickle.load(f)
    else:
        print(f"  WARNING: No cached blocks at {cache_path}")
        print(f"  TODO: implement direct ROFL block parsing")
        return []

    # Filter movement blocks
    hero_params = set(range(0x400000AE, 0x400000AE + 10))
    mov_blocks = [b for b in blocks if b["packet_id"] == 762 and b["param"] in hero_params]
    mov_blocks.sort(key=lambda b: b["timestamp"])

    decoder = MovementDecoder(text_path, rdata_path, data_path)
    movements = []
    for b in mov_blocks:
        try:
            result = decoder.decode(b["payload"])
            result["block_timestamp"] = b["timestamp"]
            result["block_param"] = b["param"]
            movements.append(result)
        except Exception:
            pass

    return movements


def generate_action_labels(movements: list[dict], projection_path: str) -> list[dict]:
    """Convert movement world coordinates to screen-space action labels.

    Uses the calibrated projection matrix:
        screen_pixel = screen_center + M @ (dest_world - garen_world)
    """
    import numpy as np

    with open(projection_path) as f:
        proj = json.load(f)

    M = np.array(proj["matrix"])
    screen_center = np.array(proj["screen_center"])

    labels = []
    for m in movements:
        if not m.get("has_destination"):
            continue

        garen_x = m.get("current_x")
        garen_y = m.get("current_y")
        dest_x = m.get("dest_x")
        dest_y = m.get("dest_y")
        game_time = m.get("game_time") or m.get("block_timestamp")

        if None in (garen_x, garen_y, dest_x, dest_y):
            continue

        # Project destination to screen space
        world_offset = np.array([dest_x - garen_x, dest_y - garen_y])
        screen_pos = screen_center + M @ world_offset

        # Compute frame index (at RECORD_FPS)
        frame_idx = int(game_time * RECORD_FPS)

        labels.append({
            "frame_idx": frame_idx,
            "game_time": game_time,
            "screen_x": float(screen_pos[0]),
            "screen_y": float(screen_pos[1]),
            "world_dest_x": dest_x,
            "world_dest_y": dest_y,
            "world_garen_x": garen_x,
            "world_garen_y": garen_y,
            "speed": m.get("speed"),
            "champion": m.get("champion_name"),
            "entity_id": m.get("entity_id"),
        })

    return labels


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Automated replay processing pipeline")

    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--replay", help="Single .rofl file to process")
    input_group.add_argument("--replay-dir", help="Directory of .rofl files to process")

    parser.add_argument("--output-dir", "-o", required=True,
                        help="Output directory for processed data")
    parser.add_argument("--projection", help="Projection matrix JSON (from calibrate_projection.py)")
    parser.add_argument("--pe-dump-dir", default=None,
                        help="PE section dumps directory (default: /tmp/pe_dump)")
    parser.add_argument("--record-only", action="store_true",
                        help="Only record video, skip movement decoding")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip replays that already have output")
    parser.add_argument("--record-fps", type=int, default=RECORD_FPS,
                        help=f"Recording FPS (default: {RECORD_FPS})")
    parser.add_argument("--replay-speed", type=float, default=REPLAY_SPEED,
                        help=f"Replay speed multiplier (default: {REPLAY_SPEED})")

    args = parser.parse_args()

    # Override globals from args
    global RECORD_FPS, REPLAY_SPEED
    RECORD_FPS = args.record_fps
    REPLAY_SPEED = args.replay_speed

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect replay files
    if args.replay:
        replays = [args.replay]
    else:
        replay_dir = Path(args.replay_dir)
        replays = sorted(str(f) for f in replay_dir.glob("*.rofl"))

    if not replays:
        print("No .rofl files found")
        sys.exit(1)

    print(f"Found {len(replays)} replay(s) to process")
    print(f"Recording at {RECORD_FPS}fps, {REPLAY_SPEED}x speed")
    print(f"Output: {output_dir}")

    # Process each replay
    all_stats = []
    for i, rofl_path in enumerate(replays):
        basename = Path(rofl_path).stem
        if args.skip_existing and (output_dir / basename / "video").exists():
            print(f"\nSkipping {basename} (already processed)")
            continue

        print(f"\n[Replay {i+1}/{len(replays)}]")
        stats = process_single_replay(
            rofl_path, str(output_dir),
            projection_path=args.projection,
            record_only=args.record_only,
            pe_dump_dir=args.pe_dump_dir,
        )
        all_stats.append(stats)

        # Brief pause between replays
        if i < len(replays) - 1:
            print("\n  Pausing 5s before next replay...")
            time.sleep(5)

    # Summary
    print(f"\n{'='*60}")
    print(f"Processing complete")
    print(f"{'='*60}")
    successes = sum(1 for s in all_stats if s["success"])
    print(f"Processed: {len(all_stats)}")
    print(f"Successful: {successes}")
    print(f"Failed: {len(all_stats) - successes}")
    total_movements = sum(s.get("movements", 0) for s in all_stats)
    if total_movements:
        print(f"Total movements decoded: {total_movements}")

    # Save summary
    summary_path = output_dir / "processing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_stats, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
