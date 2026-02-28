#!/usr/bin/env python3
"""Automated replay processing pipeline.

Runs on Windows desktop. For each .rofl file:
1. Parse ROFL metadata to find Garen's player slot
2. Launch replay via League executable
3. Wait for game to load (poll Live Client Data API)
4. Lock camera to Garen via keyboard automation (pynput)
5. Record video via screen capture (mss + cv2.VideoWriter)
6. Optionally decode movement data and generate action labels

Usage:
    # Process all replays from manifest
    python scripts/process_replays.py \
        --manifest data/replays/manifest_na1_full.json \
        --output-dir data/processed_replays

    # Process a single replay
    python scripts/process_replays.py \
        --replay path/to/replay.rofl \
        --output-dir data/processed_replays

    # With movement decoding and action labels
    python scripts/process_replays.py \
        --manifest data/replays/manifest_na1_full.json \
        --output-dir data/processed_replays \
        --projection data/projection_matrix.json

    # Dry run (show what would be processed)
    python scripts/process_replays.py --manifest data/replays/manifest.json --dry-run

Requirements:
    - Windows with League of Legends installed
    - pip install requests mss opencv-python pynput

Live Client Data API (localhost:2999, HTTPS self-signed cert):
    GET /liveclientdata/allgamedata  - full game state
    GET /liveclientdata/gamestats    - game time, map, mode
    GET /liveclientdata/playerlist   - all 10 players
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

try:
    import requests
except ImportError:
    print("Error: 'requests' required. Install with: pip install requests")
    sys.exit(1)

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LEAGUE_EXE = r"C:\Riot Games\League of Legends\Game\League of Legends.exe"
REPLAYS_DIR = r"C:\Users\daniz\Documents\League of Legends\Replays"

# Live Client Data API
LIVECLIENT_BASE = "https://127.0.0.1:2999"
API_TIMEOUT = 5

# Recording
RECORD_FPS = 20
RECORD_CODEC = "MJPG"  # cv2 fourcc

# Timeouts
GAME_LOAD_TIMEOUT = 180
GAME_LOAD_POLL = 3.0
PLAYBACK_POLL = 5.0
PROCESS_KILL_WAIT = 3

# Camera setup delays (seconds)
CAMERA_SETUP_DELAY = 2.0
KEY_PRESS_DELAY = 0.3

GAREN_CHAMPION = "Garen"
PROGRESS_FILE = "data/replays/processing_progress.json"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("process_replays")

# ---------------------------------------------------------------------------
# Live Client Data API
# ---------------------------------------------------------------------------


def liveclient_get(endpoint: str) -> dict | None:
    """GET from the Live Client Data API."""
    url = f"{LIVECLIENT_BASE}/liveclientdata/{endpoint}"
    try:
        resp = requests.get(url, verify=False, timeout=API_TIMEOUT)
        resp.raise_for_status()
        return resp.json()
    except Exception:
        return None


def get_game_time() -> float | None:
    """Get current game time in seconds."""
    data = liveclient_get("gamestats")
    if data:
        return data.get("gameTime")
    return None


def wait_for_game_load(timeout: int = GAME_LOAD_TIMEOUT) -> bool:
    """Poll Live Client Data API until the game is loaded."""
    log.info(f"Waiting for game to load (timeout {timeout}s)...")
    start = time.time()
    while time.time() - start < timeout:
        t = get_game_time()
        if t is not None:
            log.info(f"Game loaded (time: {t:.1f}s)")
            return True
        time.sleep(GAME_LOAD_POLL)
    log.error(f"Game did not load within {timeout}s")
    return False


# ---------------------------------------------------------------------------
# Keyboard automation (pynput)
# ---------------------------------------------------------------------------


def focus_game_window():
    """Find and focus the League of Legends game window.

    Windows blocks SetForegroundWindow from background processes (SSH).
    Strategy: disable the foreground lock timeout, then use multiple
    focus methods, and finally simulate a mouse click as a fallback.
    """
    if sys.platform != "win32":
        return False

    import ctypes
    import ctypes.wintypes

    user32 = ctypes.windll.user32
    kernel32 = ctypes.windll.kernel32
    result = []

    def callback(hwnd, _):
        length = user32.GetWindowTextLengthW(hwnd)
        if length > 0:
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            if "league of legends" in buf.value.lower():
                result.append((hwnd, buf.value))
        return True

    WNDENUMPROC = ctypes.WINFUNCTYPE(
        ctypes.c_bool, ctypes.wintypes.HWND, ctypes.wintypes.LPARAM)
    user32.EnumWindows(WNDENUMPROC(callback), 0)

    if not result:
        log.warning("  Could not find League window")
        return False

    hwnd = result[0][0]
    title = result[0][1]
    log.info(f"  Found window: {title!r} (hwnd={hwnd})")

    # Disable foreground lock timeout (allows SetForegroundWindow from background)
    SPI_SETFOREGROUNDLOCKTIMEOUT = 0x2001
    user32.SystemParametersInfoW(SPI_SETFOREGROUNDLOCKTIMEOUT, 0, None, 0)

    # AttachThreadInput trick
    fg_hwnd = user32.GetForegroundWindow()
    fg_thread = user32.GetWindowThreadProcessId(fg_hwnd, None)
    cur_thread = kernel32.GetCurrentThreadId()
    user32.AttachThreadInput(cur_thread, fg_thread, True)

    # Alt-key trick
    user32.keybd_event(0x12, 0, 0, 0)
    user32.keybd_event(0x12, 0, 2, 0)

    user32.ShowWindow(hwnd, 9)  # SW_RESTORE
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    user32.AttachThreadInput(cur_thread, fg_thread, False)

    time.sleep(0.5)

    # Check if focus worked
    new_fg = user32.GetForegroundWindow()
    if new_fg == hwnd:
        log.info("  Game window focused via SetForegroundWindow")
        return True

    log.info("  SetForegroundWindow failed, clicking game window...")

    # Fallback: simulate a mouse click on the game window to force focus
    from pynput.mouse import Controller as MouseController, Button
    mouse = MouseController()

    # Get window rect and click its center
    rect = ctypes.wintypes.RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(rect))
    cx = (rect.left + rect.right) // 2
    cy = (rect.top + rect.bottom) // 2
    log.info(f"  Clicking window center ({cx}, {cy})")

    mouse.position = (cx, cy)
    time.sleep(0.1)
    mouse.click(Button.left)
    time.sleep(0.5)

    new_fg = user32.GetForegroundWindow()
    if new_fg == hwnd:
        log.info("  Game window focused via mouse click")
    else:
        log.warning(f"  Focus may have failed (foreground={new_fg}, target={hwnd})")

    return True


def press_key(keyboard, key_char: str):
    """Press and release a key."""
    keyboard.press(key_char)
    time.sleep(0.05)
    keyboard.release(key_char)
    time.sleep(KEY_PRESS_DELAY)


def setup_camera(garen_slot: int):
    """Lock camera to Garen using keyboard shortcuts.

    Args:
        garen_slot: Garen's player slot (0-9). Slots 0-4 are blue team,
                    5-9 are red team. Blue: 1-5, Red: Q W E R T.
    """
    from pynput.keyboard import Controller as KbController

    _RED_KEYS = {5: 'q', 6: 'w', 7: 'e', 8: 'r', 9: 't'}
    kb = KbController()
    time.sleep(CAMERA_SETUP_DELAY)

    # Select Garen's player
    if garen_slot < 5:
        key = str(garen_slot + 1)
        log.info(f"  Selecting Garen (blue slot {garen_slot}) via key '{key}'")
        press_key(kb, key)
    else:
        key = _RED_KEYS[garen_slot]
        log.info(f"  Selecting Garen (red slot {garen_slot}) via key '{key}'")
        press_key(kb, key)

    # Center camera
    press_key(kb, ' ')
    time.sleep(0.5)

    # Lock camera (Y toggles lock)
    press_key(kb, 'y')

    # Center again to ensure lock is on the right target
    press_key(kb, ' ')

    log.info("  Camera lock sent (Y + Space)")


def set_replay_speed(speed_presses: int = 3):
    """Speed up replay playback by pressing '=' multiple times.

    Each press of '=' advances speed tier: 1x -> 2x -> 4x -> 8x.
    Press '0' first to reset to normal speed, then press '=' to reach target.

    Args:
        speed_presses: Number of times to press '=' (1=2x, 2=4x, 3=8x).
    """
    from pynput.keyboard import Controller as KbController

    kb = KbController()

    # Reset to 1x first
    press_key(kb, '0')
    time.sleep(0.3)

    # Speed up
    for i in range(speed_presses):
        press_key(kb, '=')

    speeds = {0: "1x", 1: "2x", 2: "4x", 3: "8x"}
    log.info(f"  Replay speed set to {speeds.get(speed_presses, f'{speed_presses} presses')}")


# ---------------------------------------------------------------------------
# Screen recording (mss + cv2)
# ---------------------------------------------------------------------------


class ScreenRecorder:
    """Record screen to video file using dxcam (GPU-accelerated) + cv2 writer.

    dxcam uses DXGI Desktop Duplication for fast capture.
    Frames are downscaled to target resolution before encoding.
    Call capture_frame() in a loop from the main thread.
    """

    TARGET_WIDTH = 1920
    TARGET_HEIGHT = 1080

    def __init__(self, output_path: str, fps: int = RECORD_FPS,
                 codec: str = RECORD_CODEC):
        import cv2

        self.output_path = output_path
        self.fps = fps
        self._cv2 = cv2

        # Try dxcam first, fall back to mss
        try:
            import dxcam
            # Monkey-patch comtypes to prevent COM __del__ crash (access violation
            # at 0xFFFFFFFFFFFFFFFF during GC kills the entire process)
            try:
                import comtypes._post_coinit.unknwn as _unknwn
                _unknwn._compointer_base.__del__ = lambda self: None
            except Exception:
                pass
            self._camera = dxcam.create()
            # Get native resolution from dxcam
            test_frame = self._camera.grab()
            if test_frame is not None:
                self._native_h, self._native_w = test_frame.shape[:2]
            else:
                import ctypes
                user32 = ctypes.windll.user32
                self._native_w = user32.GetSystemMetrics(0)
                self._native_h = user32.GetSystemMetrics(1)
            self._use_dxcam = True
            log.info(f"  Using dxcam capture ({self._native_w}x{self._native_h})")
        except Exception as e:
            log.warning(f"  dxcam unavailable ({e}), falling back to mss")
            import mss
            self._sct = mss.mss()
            self._monitor = self._sct.monitors[1]
            self._native_w = self._monitor["width"]
            self._native_h = self._monitor["height"]
            self._use_dxcam = False

        # Output at target resolution
        self.width = self.TARGET_WIDTH
        self.height = self.TARGET_HEIGHT
        self._need_resize = (self._native_w != self.width
                             or self._native_h != self.height)

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            output_path, fourcc, fps, (self.width, self.height))
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open VideoWriter: {output_path}")

        self.frame_times = []
        self.frame_count = 0
        self._running = True

        log.info(f"  Recorder: {self._native_w}x{self._native_h} -> "
                 f"{self.width}x{self.height} @ {fps}fps -> {output_path}")

    def capture_frame(self, game_time: float | None = None):
        """Capture one frame. Call from main thread."""
        if self._use_dxcam:
            frame = self._camera.grab()
            if frame is None:
                return  # skip dropped frame
            # dxcam returns RGB numpy array, cv2 needs BGR
            frame = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
        else:
            import numpy as np
            img = self._sct.grab(self._monitor)
            frame = np.array(img)
            frame = self._cv2.cvtColor(frame, self._cv2.COLOR_BGRA2BGR)

        if self._need_resize:
            frame = self._cv2.resize(frame, (self.width, self.height),
                                     interpolation=self._cv2.INTER_AREA)

        self.writer.write(frame)
        self.frame_times.append((self.frame_count, game_time))
        self.frame_count += 1

    def stop(self):
        """Stop recording and release resources."""
        self._running = False
        self.writer.release()
        # Don't call dxcam camera.release() — it triggers a COM access
        # violation crash. Let garbage collection handle it instead.
        if self._use_dxcam:
            self._camera = None
        log.info(f"  Recording stopped ({self.frame_count} frames)")

    def save_timestamps(self, path: str):
        """Save frame timestamps to JSON."""
        with open(path, "w") as f:
            json.dump(self.frame_times, f)
        log.info(f"  Saved {len(self.frame_times)} frame timestamps")


# ---------------------------------------------------------------------------
# ROFL metadata parsing
# ---------------------------------------------------------------------------


def parse_rofl_metadata(rofl_path: str) -> dict | None:
    """Parse ROFL file to extract game metadata (players, champions, etc.)."""
    with open(rofl_path, "rb") as f:
        data = f.read()

    if data[:4] != b"RIOT":
        log.error(f"Not a RIOT replay file (magic: {data[:6]!r})")
        return None

    json_start = data.find(b'{"gameLength')
    if json_start < 0:
        log.error("Could not find metadata JSON in ROFL file")
        return None

    depth = 0
    json_end = json_start
    for i in range(json_start, len(data)):
        if data[i:i + 1] == b"{":
            depth += 1
        elif data[i:i + 1] == b"}":
            depth -= 1
            if depth == 0:
                json_end = i + 1
                break

    try:
        meta = json.loads(data[json_start:json_end])
    except json.JSONDecodeError as e:
        log.error(f"Failed to parse ROFL JSON metadata: {e}")
        return None

    stats_raw = meta.get("statsJson", "[]")
    if isinstance(stats_raw, str):
        stats = json.loads(stats_raw)
    else:
        stats = stats_raw

    players = []
    for i, p in enumerate(stats):
        players.append({
            "slot": i,
            "name": p.get("NAME", p.get("SKIN", "?")),
            "champion": p.get("SKIN", p.get("NAME", "?")),
            "team": "blue" if i < 5 else "red",
            "position": p.get("INDIVIDUAL_POSITION", "?"),
            "win": p.get("WIN", "?"),
        })

    return {
        "players": players,
        "game_length_ms": meta.get("gameLength"),
        "raw": meta,
    }


def find_garen_slot(metadata: dict) -> int | None:
    """Find Garen's player slot (0-9) from ROFL metadata."""
    for p in metadata.get("players", []):
        if p.get("champion", "").lower() == GAREN_CHAMPION.lower():
            return p["slot"]
    return None


# ---------------------------------------------------------------------------
# Process management
# ---------------------------------------------------------------------------


def launch_replay_lcu(game_id: str) -> bool:
    """Launch a replay via the LCU API (League client must be running)."""
    import base64
    import ssl
    from urllib.request import Request, urlopen
    from urllib.error import URLError

    lockfile = r"C:\Riot Games\League of Legends\lockfile"
    if not os.path.isfile(lockfile):
        log.error(f"LCU lockfile not found: {lockfile}")
        return False

    with open(lockfile) as f:
        _, _, port, token, _ = f.read().split(":")

    auth = base64.b64encode(f"riot:{token}".encode()).decode()
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    url = f"https://127.0.0.1:{port}/lol-replays/v1/rofls/{game_id}/watch"
    body = json.dumps({"componentType": "replay"}).encode()
    req = Request(url, method="POST", data=body, headers={
        "Authorization": f"Basic {auth}",
        "Content-Type": "application/json",
    })

    try:
        urlopen(req, context=ctx)
        log.info(f"  Launched replay {game_id} via LCU API")
        return True
    except URLError as e:
        log.error(f"  LCU launch failed: {e}")
        return False


def launch_replay(rofl_path: str) -> bool:
    """Launch a replay. Tries LCU API first, then os.startfile fallback."""
    rofl_path = os.path.abspath(rofl_path)

    if not os.path.isfile(rofl_path):
        log.error(f"Replay file not found: {rofl_path}")
        return False

    basename = os.path.basename(rofl_path)
    log.info(f"Launching replay: {basename}")

    # Extract game_id from filename (e.g. "NA1-5496610100.rofl" -> "5496610100")
    stem = Path(rofl_path).stem
    parts = stem.split("-")
    game_id = parts[-1] if len(parts) >= 2 else None

    # Try LCU API first
    if game_id and sys.platform == "win32":
        if launch_replay_lcu(game_id):
            return True

    # Fallback: open with default handler
    if sys.platform == "win32":
        try:
            os.startfile(rofl_path)
            log.info("  Launched via os.startfile")
            return True
        except OSError as e:
            log.error(f"Failed to launch replay: {e}")
            return False

    log.error("Replay launching only works on Windows")
    return False


def kill_game_process():
    """Kill the League of Legends game process."""
    if sys.platform == "win32":
        os.system('taskkill /F /IM "League of Legends.exe" 2>NUL')
    else:
        os.system("pkill -f 'League of Legends' 2>/dev/null")
    time.sleep(PROCESS_KILL_WAIT)


# ---------------------------------------------------------------------------
# Progress tracking
# ---------------------------------------------------------------------------


def load_progress(path: str) -> dict:
    """Load processing progress from disk."""
    if os.path.isfile(path):
        with open(path) as f:
            return json.load(f)
    return {"completed": [], "failed": []}


def save_progress(progress: dict, path: str):
    """Save processing progress to disk."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(progress, f, indent=2)


# ---------------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------------


def process_single_replay(rofl_path: str, output_dir: str,
                          projection_path: str = None,
                          record_only: bool = False,
                          pe_dump_dir: str = None,
                          replay_speed: int = 3) -> dict:
    """Process a single replay: launch, configure camera, record, cleanup."""
    rofl_path = os.path.abspath(rofl_path)
    basename = Path(rofl_path).stem
    replay_output = Path(output_dir) / basename
    replay_output.mkdir(parents=True, exist_ok=True)

    stats = {"replay": basename, "success": False, "video_path": None, "movements": 0}

    log.info(f"{'='*60}")
    log.info(f"Processing: {basename}")
    log.info(f"{'='*60}")

    # Step 1: Parse metadata to find Garen
    log.info("[1/6] Parsing ROFL metadata...")
    metadata = parse_rofl_metadata(rofl_path)
    garen_slot = None
    if metadata:
        garen_slot = find_garen_slot(metadata)
        if garen_slot is not None:
            team = "blue" if garen_slot < 5 else "red"
            log.info(f"  Garen at slot {garen_slot} ({team} side)")
        else:
            log.warning("  Garen not found — skipping replay")
            champs = [p.get("champion") for p in metadata.get("players", [])]
            log.warning(f"  Players: {champs}")
            stats["skipped"] = True
            return stats

        meta_path = replay_output / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
    else:
        log.warning("  Could not parse metadata")

    # Step 2: Launch replay
    log.info("[2/6] Launching replay...")
    if not launch_replay(rofl_path):
        log.error("Failed to launch replay")
        return stats

    recorder = None
    try:
        # Step 3: Wait for game to load
        log.info("[3/6] Waiting for game to load...")
        if not wait_for_game_load():
            log.error("Game failed to load — skipping")
            return stats

        time.sleep(3)  # let it fully initialize

        # Step 4: Focus game window + lock camera to Garen via keyboard
        log.info("[4/6] Setting up camera...")
        focus_game_window()
        if garen_slot is not None:
            setup_camera(garen_slot)
        else:
            log.warning("  No Garen slot — skipping camera lock")

        time.sleep(1)

        # Step 4b: Speed up replay
        if replay_speed > 0:
            set_replay_speed(replay_speed)
            time.sleep(0.5)

        # Step 5: Record screen (main-thread capture loop)
        log.info("[5/6] Recording replay...")
        video_path = str(replay_output / "replay.avi")
        recorder = ScreenRecorder(video_path, fps=RECORD_FPS)

        game_length_ms = metadata.get("game_length_ms", 0) if metadata else 0
        game_length_s = game_length_ms / 1000.0 if game_length_ms else 0

        api_fail_start = None  # wall-clock when API first failed
        stall_ref_wall = time.time()  # wall-clock for stall window start
        stall_ref_game = 0.0  # game time at stall window start
        last_log_time = 0
        frame_interval = 1.0 / RECORD_FPS
        log.info("  Recording started")

        STALL_WALL_TIMEOUT = 15  # real seconds to wait before declaring stall
        STALL_GAME_THRESHOLD = 5.0  # game must advance >=5 game-sec in 15 real-sec
        API_FAIL_TIMEOUT = 10  # real seconds of no API response = game ended

        # Camera re-center: spam select key to keep Garen near screen center
        RECENTER_INTERVAL = 0.1  # real seconds between re-center presses
        from pynput.keyboard import Controller as KbController
        kb = KbController()
        # Blue side: 1-5, Red side: Q W E R T
        _RED_KEYS = {5: 'q', 6: 'w', 7: 'e', 8: 'r', 9: 't'}
        if garen_slot is not None:
            if garen_slot < 5:
                recenter_key = str(garen_slot + 1)  # blue: 1-5
            else:
                recenter_key = _RED_KEYS[garen_slot]  # red: q w e r t
        else:
            recenter_key = None
        last_recenter = 0.0

        while recorder._running:
            t0 = time.perf_counter()

            # Spam select key to re-center camera on Garen
            if recenter_key is not None and t0 - last_recenter >= RECENTER_INTERVAL:
                kb.press(recenter_key)
                kb.release(recenter_key)
                last_recenter = t0

            # Capture frame + get game time
            current = get_game_time()
            recorder.capture_frame(current)

            # Check for game end via API disconnect
            if current is None:
                if api_fail_start is None:
                    api_fail_start = time.time()
                elif time.time() - api_fail_start > API_FAIL_TIMEOUT:
                    log.info("  Game API stopped responding — game likely ended")
                    stats["success"] = True
                    break
            else:
                api_fail_start = None

                # Game length check: if we've reached/passed known length, done
                if game_length_s > 0 and current >= game_length_s - 5:
                    log.info(f"  Reached game length ({current:.0f}s >= {game_length_s:.0f}s)")
                    stats["success"] = True
                    break

                # Stall detection: check if game time advances over a window
                wall_elapsed = time.time() - stall_ref_wall
                game_elapsed = current - stall_ref_game
                if wall_elapsed > STALL_WALL_TIMEOUT:
                    if game_elapsed < STALL_GAME_THRESHOLD:
                        log.info(f"  Playback stalled at {current:.0f}s "
                                 f"(advanced {game_elapsed:.1f}s in {wall_elapsed:.0f}s)")
                        stats["success"] = True
                        break
                    # Reset window
                    stall_ref_wall = time.time()
                    stall_ref_game = current

                # Progress log every ~60s
                if current - last_log_time >= 60:
                    last_log_time = current
                    if game_length_s > 0:
                        pct = 100 * current / game_length_s
                        log.info(f"  Progress: {current:.0f}/{game_length_s:.0f}s "
                                 f"({pct:.0f}%) [{recorder.frame_count} frames]")
                    else:
                        log.info(f"  Game time: {current:.0f}s "
                                 f"[{recorder.frame_count} frames]")

            # Maintain target FPS
            elapsed = time.perf_counter() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        recorder.stop()
        timestamps_path = str(replay_output / "frame_timestamps.json")
        recorder.save_timestamps(timestamps_path)
        stats["video_path"] = video_path

        # Step 6: Movement decoding (optional)
        if not record_only:
            log.info("[6/6] Decoding movement data...")
            try:
                movements = _decode_movements(rofl_path, pe_dump_dir)
                if movements:
                    mov_path = replay_output / "movements.json"
                    with open(mov_path, "w") as f:
                        json.dump(movements, f, default=str)
                    stats["movements"] = len(movements)
                    log.info(f"  Decoded {len(movements)} movements")

                    if projection_path and os.path.isfile(projection_path):
                        labels = _generate_action_labels(
                            movements, projection_path, recorder.frame_times)
                        labels_path = replay_output / "action_labels.json"
                        with open(labels_path, "w") as f:
                            json.dump(labels, f)
                        log.info(f"  Generated {len(labels)} action labels")
            except Exception as e:
                log.error(f"  Movement decoding failed: {e}")
        else:
            log.info("[6/6] Skipped movement decoding (--record-only)")

    finally:
        if recorder and recorder._running:
            recorder.stop()
        log.info("Closing game...")
        kill_game_process()

    return stats


# ---------------------------------------------------------------------------
# Movement decoding (wraps decode_replay_movement.py)
# ---------------------------------------------------------------------------


def _decode_movements(rofl_path: str, pe_dump_dir: str = None) -> list[dict]:
    """Decode movement data from a .rofl file using the movement decoder."""
    scripts_dir = str(Path(__file__).parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)

    try:
        from decode_replay_movement import MovementDecoder
    except ImportError:
        log.warning("Could not import MovementDecoder — skipping")
        return []

    pe_dir = pe_dump_dir or "/tmp/pe_dump"
    for name in ["text.bin", "rdata.bin", "data.bin"]:
        if not os.path.isfile(os.path.join(pe_dir, name)):
            log.warning(f"PE dump not found: {pe_dir}/{name} — skipping decode")
            return []

    import pickle
    cache_path = rofl_path + ".blocks.pkl"
    if not os.path.isfile(cache_path):
        log.warning(f"No cached blocks at {cache_path}")
        return []

    with open(cache_path, "rb") as f:
        blocks = pickle.load(f)

    hero_params = set(range(0x400000AE, 0x400000AE + 10))
    # Support both 16.3 (pid=762) and 16.4 (pid=437)
    mov_blocks = [b for b in blocks
                  if b["packet_id"] in (762, 437) and b["param"] in hero_params]
    mov_blocks.sort(key=lambda b: b["timestamp"])

    decoder = MovementDecoder(
        os.path.join(pe_dir, "text.bin"),
        os.path.join(pe_dir, "rdata.bin"),
        os.path.join(pe_dir, "data.bin"),
    )

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


def _generate_action_labels(movements: list[dict], projection_path: str,
                            frame_times: list = None) -> list[dict]:
    """Convert world coordinates to screen-space action labels.

    screen_pixel = screen_center + M @ (dest_world - garen_world)
    """
    import numpy as np

    with open(projection_path) as f:
        proj = json.load(f)

    M = np.array(proj["matrix"])
    screen_center = np.array(proj["screen_center"])

    # Build game_time -> frame_idx mapping from recorded timestamps
    time_to_frame = {}
    if frame_times:
        for frame_idx, game_t in frame_times:
            if game_t is not None:
                time_to_frame[frame_idx] = game_t

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

        world_offset = np.array([dest_x - garen_x, dest_y - garen_y])
        screen_pos = screen_center + M @ world_offset

        # Find closest recorded frame to this game time
        frame_idx = None
        if frame_times:
            best_dist = float("inf")
            for fi, gt in frame_times:
                if gt is not None and abs(gt - game_time) < best_dist:
                    best_dist = abs(gt - game_time)
                    frame_idx = fi

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
# Replay collection helpers
# ---------------------------------------------------------------------------


def replays_from_manifest(manifest_path: str, replays_dir: str) -> list[tuple[str, str]]:
    """Get (rofl_path, replay_name) pairs from a manifest file."""
    with open(manifest_path) as f:
        manifest = json.load(f)

    result = []
    for m in manifest.get("matches", []):
        game_id = m.get("game_id", "")
        platform = m.get("platform", "").upper()
        name = f"{platform}-{game_id}"
        path = os.path.join(replays_dir, f"{name}.rofl")
        if os.path.isfile(path):
            result.append((path, name))

    return result


def replays_from_dir(replays_dir: str) -> list[tuple[str, str]]:
    """Get all .rofl files from a directory."""
    return [(str(f), f.stem) for f in sorted(Path(replays_dir).glob("*.rofl"))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    global RECORD_FPS  # noqa: PLW0603

    parser = argparse.ArgumentParser(
        description="Automated replay processing pipeline (keyboard automation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--manifest", help="Manifest JSON from download_replays.py")
    source.add_argument("--replay-dir", help="Directory containing .rofl files")
    source.add_argument("--replay", help="Single .rofl file to process")

    parser.add_argument("--output-dir", "-o", required=True,
                        help="Output directory for processed data")
    parser.add_argument("--replays-dir", default=REPLAYS_DIR,
                        help=f"Where .rofl files live (default: {REPLAYS_DIR})")
    parser.add_argument("--projection",
                        help="Projection matrix JSON (from calibrate_projection.py)")
    parser.add_argument("--pe-dump-dir",
                        help="PE section dumps directory (default: /tmp/pe_dump)")
    parser.add_argument("--record-only", action="store_true",
                        help="Only record video, skip movement decoding")
    parser.add_argument("--progress-file", default=PROGRESS_FILE,
                        help=f"Progress tracking file (default: {PROGRESS_FILE})")
    parser.add_argument("--skip-progress", action="store_true",
                        help="Ignore progress file — reprocess everything")
    parser.add_argument("--dry-run", action="store_true",
                        help="List replays to process without actually processing")
    parser.add_argument("--fps", type=int, default=RECORD_FPS,
                        help=f"Recording FPS (default: {RECORD_FPS})")
    parser.add_argument("--speed", type=int, default=3, choices=[0, 1, 2, 3],
                        help="Replay speed: 0=1x, 1=2x, 2=4x, 3=8x (default: 3)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        log.setLevel(logging.DEBUG)

    RECORD_FPS = args.fps

    # Build replay list
    if args.replay:
        replays = [(os.path.abspath(args.replay), Path(args.replay).stem)]
    elif args.replay_dir:
        replays = replays_from_dir(args.replay_dir)
    else:
        replays = replays_from_manifest(args.manifest, args.replays_dir)

    if not replays:
        log.error("No replay files found")
        sys.exit(1)

    log.info(f"Found {len(replays)} replay(s)")

    progress = load_progress(args.progress_file) if not args.skip_progress else {
        "completed": [], "failed": [],
    }
    done = set(progress["completed"])
    to_process = [(p, n) for p, n in replays if n not in done]

    skipped = len(replays) - len(to_process)
    if skipped:
        log.info(f"Skipping {skipped} already-processed replay(s)")

    if not to_process:
        log.info("All replays already processed!")
        return

    log.info(f"Will process {len(to_process)} replay(s) at {RECORD_FPS}fps")

    if args.dry_run:
        print("\nReplays to process:")
        for path, name in to_process:
            print(f"  {name}: {path}")
        print(f"\nTotal: {len(to_process)}")
        return

    output_dir = args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    succeeded = 0
    failed = 0

    for i, (rofl_path, replay_name) in enumerate(to_process):
        log.info(f"\n[{i+1}/{len(to_process)}] {replay_name}")

        try:
            result = process_single_replay(
                rofl_path, output_dir,
                projection_path=args.projection,
                record_only=args.record_only,
                pe_dump_dir=args.pe_dump_dir,
                replay_speed=args.speed,
            )
        except KeyboardInterrupt:
            log.info("\nInterrupted — saving progress")
            save_progress(progress, args.progress_file)
            sys.exit(1)
        except Exception as e:
            log.error(f"Unexpected error: {e}")
            result = {"success": False}

        if result.get("skipped"):
            log.info(f"  Skipped (no Garen)")
            continue
        elif result.get("success"):
            progress["completed"].append(replay_name)
            succeeded += 1
        else:
            progress["failed"].append(replay_name)
            failed += 1

        save_progress(progress, args.progress_file)

        if i < len(to_process) - 1:
            log.info("Waiting 5s before next replay...")
            time.sleep(5)

    log.info(f"\n{'='*60}")
    log.info(f"DONE — {succeeded + failed}/{len(to_process)} replays processed")
    log.info(f"  Succeeded: {succeeded}")
    log.info(f"  Failed:    {failed}")
    log.info(f"  Output:    {output_dir}")
    log.info(f"  Progress:  {args.progress_file}")

    summary_path = Path(output_dir) / "processing_summary.json"
    with open(summary_path, "w") as f:
        json.dump(progress, f, indent=2)


if __name__ == "__main__":
    main()
