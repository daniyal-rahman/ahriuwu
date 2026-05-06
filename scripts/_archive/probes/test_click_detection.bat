@echo off
REM Test Click-Destination Detection in League Replays
REM
REM Prerequisites:
REM   - Windows booted with Vanguard ENABLED
REM   - Python 3 installed with pynput and urllib3
REM   - C:\tmp\ directory exists
REM   - League of Legends installed
REM
REM Usage:
REM   - Start this script
REM   - Select 1) Keylogger Only or 2) Full Test
REM   - Play a live game (Keylogger captures clicks + calibration)
REM   - Save replay
REM   - Reboot to Vanguard-disabled Linux
REM   - Run analysis scripts
REM

echo.
echo =============================================================================
echo Test: Click-Destination Detection in League Replays
echo =============================================================================
echo.
echo Prerequisites:
echo   [OK] Windows booted with Vanguard ENABLED
echo   [TODO] Python 3 with pynput installed
echo   [TODO] C:\tmp\ directory created
echo.

REM Check for Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python 3 not found
    echo Please install Python 3 from python.org and ensure python is in PATH
    pause
    exit /b 1
)

REM Create tmp directory if needed
if not exist C:\tmp mkdir C:\tmp

echo.
echo Menu:
echo   1) Start Keylogger ONLY (prep for live game)
echo   2) Start Keylogger + Open League (full test)
echo   3) View previous logs
echo   4) Clean logs and exit
echo.

set /p choice="Select option (1-4): "

if "%choice%"=="1" (
    echo.
    echo Starting keylogger...
    echo Output will be saved to:
    echo   C:\tmp\clicks.json
    echo   C:\tmp\calibration.json
    echo.
    echo Instructions:
    echo   1. This window will start the keylogger
    echo   2. Open League of Legends manually
    echo   3. Play a live game (any mode)
    echo   4. During the match, press any key once (for calibration)
    echo      - You'll be prompted to enter the in-game clock time
    echo   5. Play until you've made several right-clicks
    echo   6. End the game
    echo   7. Press Ctrl+C in this window to save logs
    echo.
    pause
    python "%~dp0keylogger_test.py"
    goto end
)

if "%choice%"=="2" (
    echo.
    echo Starting keylogger in background...
    start "Keylogger" python "%~dp0keylogger_test.py"
    timeout /t 2

    echo.
    echo Starting League of Legends...
    start "" "C:\Riot Games\League of Legends\Game\League of Legends.exe"

    echo.
    echo Instructions:
    echo   1. League will open in 10-20 seconds
    echo   2. Log in and queue for a game
    echo   3. During the match, press any key once for calibration
    echo   4. Click normally throughout the game
    echo   5. After the game, return to this window
    echo.
    timeout /t 5
    pause

    echo.
    echo Game finished? Make sure the replay file was saved.
    echo Expected location: C:\Users\daniz\Documents\League of Legends\Replays\
    echo.
    echo Keylogger window should still be open - press Ctrl+C to finalize logs.
    echo.
    goto end
)

if "%choice%"=="3" (
    echo.
    if exist C:\tmp\clicks.json (
        echo Recent clicks:
        python -m json.tool C:\tmp\clicks.json | head -30
    ) else (
        echo No clicks logged yet.
    )
    echo.
    if exist C:\tmp\calibration.json (
        echo Calibration:
        type C:\tmp\calibration.json
    ) else (
        echo No calibration yet.
    )
    echo.
    goto end
)

if "%choice%"=="4" (
    echo.
    echo Cleaning logs...
    if exist C:\tmp\clicks.json del C:\tmp\clicks.json
    if exist C:\tmp\calibration.json del C:\tmp\calibration.json
    if exist C:\tmp\trajectory_analysis_*.json del C:\tmp\trajectory_analysis_*.json
    echo Done.
    echo.
    goto end
)

echo Invalid choice.
:end
echo.
echo Next steps:
echo   1. Reboot to Vanguard-disabled Windows (or SSH to Linux)
echo   2. Copy replay to /tmp/: scp windows:path/to/replay.rofl /tmp/
echo   3. Run analysis:
echo      python scripts/analyze_trajectory_vs_clicks.py <replay_id>
echo   4. For broad heap scan:
echo      python scripts/heap_scan_click_intent.py
echo.
pause
