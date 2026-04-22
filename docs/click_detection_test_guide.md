# Click-Destination Detection Test Guide

This guide describes how to validate whether click-destination data exists in League replays using a live game + keylogger + trajectory analysis approach.

## Overview

**Goal**: Determine if player right-clicks (and thus intent) can be recovered from replay data.

**Approach**:
1. **Live game with Vanguard enabled** to capture real clicks
2. **Keylogger** to record wall-clock timestamps of each right-click
3. **Calibration** to align wall-clock time with in-game time via Replay API
4. **Trajectory analysis** to detect inflection points in hero movement
5. **Heap scan** (optional) to search for Vec3s with click-intent signatures

## Prerequisites

- **Windows**: Booted with Vanguard **enabled** (required for live games)
- **Python 3**: Installed with `pynput` library
- **League of Legends**: Installed and playable
- **Disk space**: ~500MB for replay file

## Step 1: Reboot to Windows (with Vanguard Enabled)

Reboot the dual-boot system into Windows:
- Restart the desktop and select Windows in the boot menu
- Make sure Vanguard services are running (they should be by default)

**On macOS**: No action needed, stand by for data transfer after the game.

## Step 2: Windows Setup

Once in Windows:

1. **Install test dependencies**:
   ```
   C:\Repos\ahriuwu\scripts\setup_windows_test.bat
   ```
   This installs `pynput` (keyboard/mouse capture) and creates `C:\tmp\`.

2. **Start the test**:
   ```
   C:\Repos\ahriuwu\scripts\test_click_detection.bat
   ```
   
   Menu options:
   - **Option 1**: Start keylogger only (you open League manually)
   - **Option 2**: Start keylogger + auto-launch League
   - **Option 3**: View previous logs
   - **Option 4**: Clean logs

## Step 3: Live Game

During the game:

1. **Play normally** — all right-clicks are logged with wall-clock timestamps
2. **At any point after 0:10 game time**, press any key (e.g., spacebar)
   - Keylogger captures: wall-clock time of that key press
   - You'll be prompted: "Enter game clock time"
   - Type the in-game clock value (e.g., `45.3` for 0:45)
   - This becomes the **calibration point** to align all timestamps
3. **Make several right-clicks** throughout the game (10+ is ideal)
4. **Finish the game** — replay file is auto-saved to:
   ```
   C:\Users\daniz\Documents\League of Legends\Replays\<DATE> - <CHAMPION> - <MAP>.rofl
   ```

## Step 4: Data Collection

After the game, two files are created in `C:\tmp\`:

- **`clicks.json`**: Array of `{wall_time, wall_time_iso, button, x, y}`
- **`calibration.json`**: Single entry `{wall_time, wall_time_iso, game_time, key}`

Example:
```json
{
  "wall_time": 1713792145.2345,
  "wall_time_iso": "2026-04-22T10:22:25.234500",
  "game_time": 45.3,
  "key": "space"
}
```

## Step 5: Copy Data to macOS + Analysis

From Windows, copy replay and keylog files to macOS:

1. **Copy replay file**:
   ```bash
   scp windows:"C:\Users\daniz\Documents\League of Legends\Replays\*.rofl" /tmp/
   ```

2. **Copy keylog data**:
   ```bash
   scp windows:"C:\tmp\*.json" /tmp/
   ```

3. **Extract replay ID** from filename:
   ```
   NA1_<REPLAY_ID>.rofl  →  REPLAY_ID = <REPLAY_ID>
   ```

## Step 6: Analysis (on macOS)

### Trajectory Analysis (Primary)

Analyze hero movement for inflection points that correlate with clicks:

```bash
python scripts/analyze_trajectory_vs_clicks.py <REPLAY_ID>
```

**Output**:
- `trajectory_analysis_<REPLAY_ID>.json`: Detected inflection points + click correlations
- `trajectory_plot_<REPLAY_ID>.png`: Visual trajectory plot (if matplotlib installed)

**Interpretation**:
- **Good result**: Clicks correlate with direction changes (matched > 70%)
  → Suggests click intent is recoverable from position deltas
- **Bad result**: Few/no matches
  → Suggests either: (a) clicks aren't in position data, or (b) noise/filtering needed

### Heap Scan (Secondary, Optional)

Broad memory search for Vec3 fields matching click-destination patterns:

```bash
python scripts/heap_scan_click_intent.py
```

**What it does**:
1. Pauses replay at known hero position
2. Scans heap memory for Vec3s matching that position
3. Times-samples each match for discrete step-changes
4. Reports candidates with click-like signatures

**Note**: This requires the replay to still be running (Vanguard disabled).

## Expected Results

### If clicks ARE recoverable:
- Trajectory analysis shows 70%+ matching between clicks and inflection points
- Heap scan finds Vec3s that consistently update at click times
- Next step: Train model to predict clicks from position deltas

### If clicks are NOT recoverable:
- Trajectory analysis shows <30% matching
- Heap scan finds no meaningful patterns
- Alternative approach: Use raw position data for movement prediction (without click intent)

## Troubleshooting

**Keylogger doesn't start**:
- Check: `python -m pip install pynput`
- Check: `C:\tmp\` directory exists
- Check: Anti-virus isn't blocking pynput

**Calibration failed**:
- Make sure you press a key between game time 0:10 and match end
- Type the **in-game clock value** (visible in top-left), not wall-clock time

**Replay file not found**:
- Check: `C:\Users\daniz\Documents\League of Legends\Replays\`
- Replay only saves after you finish (FF or victory)
- Files are named: `<DATE> - <CHAMPION> - <MAP>.rofl`

**Analysis script fails**:
- Check: calibration.json has `game_time` field filled in
- Check: clicks.json has >0 clicks
- Check: replay ID is correct (from filename)

## Files

| File | Purpose |
|------|---------|
| `scripts/keylogger_test.py` | Main keylogger (runs on Windows) |
| `scripts/test_click_detection.bat` | Menu-driven test orchestration (Windows) |
| `scripts/setup_windows_test.bat` | Install dependencies (Windows) |
| `scripts/analyze_trajectory_vs_clicks.py` | Trajectory inflection analysis (macOS) |
| `scripts/heap_scan_click_intent.py` | Broad Vec3 heap search (Windows) |

## References

- **AiManager struct**: `hero+0x3E0C` (inline), ServerPos at `+0x474` (live in replay)
- **Known dead fields**: TargetPosition, PathEnd, Velocity, IsMoving, Segments (all zeroed in replay)
- **Replay API**: Port 2999, endpoints for playback/seek/render
- **Memory probe scripts**: `scripts/aimanager_inline_probe.py`, `scripts/aimanager_diag.py`
