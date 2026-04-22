# Quick Start: Click-Detection Test

## Hosts
- **Windows (Vanguard)**: `10.0.0.14` (dani@10.0.0.14)
- **macOS (this laptop)**: Analysis

## Quick Flow

### 1. Windows Side
```
Reboot to Windows
↓
Run: C:\Repos\ahriuwu\scripts\setup_windows_test.bat
↓
Run: C:\Repos\ahriuwu\scripts\test_click_detection.bat (Option 1 or 2)
↓
Play live game
- Press any key once to calibrate (enter in-game time when prompted)
- Make 10+ right-clicks
- Finish match
↓
Files saved:
  C:\tmp\clicks.json
  C:\tmp\calibration.json
  C:\Users\daniz\Documents\League of Legends\Replays\<DATE> - <CHAMP> - <MAP>.rofl
```

### 2. macOS Side
```
# Copy files from Windows
scp dani@10.0.0.14:"C:\tmp\*.json" /tmp/
scp dani@10.0.0.14:"C:\Users\daniz\Documents\League of Legends\Replays\*.rofl" /tmp/

# Find replay ID from filename: NA1_<ID>.rofl
# Run analysis
python scripts/analyze_trajectory_vs_clicks.py <ID>

# Check results in /tmp/trajectory_analysis_<ID>.json
# Success: "match_count" > 70% of total clicks
```

## Expected Output

**Success** (>70% match):
```json
{
  "click_count": 15,
  "inflection_count": 12,
  "match_count": 11,
  "...": "Matched 11/15 clicks to inflections"
}
```

**Failure** (<30% match):
```json
{
  "click_count": 15,
  "inflection_count": 8,
  "match_count": 2,
  "...": "Matched 2/15 clicks to inflections"
}
```

## Troubleshooting

**Keylogger won't start**: `pip install pynput`
**Calibration failed**: Make sure you press a key after 0:10 game time
**SCP fails**: Check `10.0.0.14` is reachable: `ping 10.0.0.14`
**Analysis fails**: Check calibration.json has `game_time` field filled in
