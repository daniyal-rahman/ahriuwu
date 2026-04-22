@echo off
REM Setup script for Windows test environment

echo.
echo Installing Python dependencies for click-detection test...
echo.

REM Install pynput (keyboard/mouse capture)
echo Installing pynput...
python -m pip install pynput --quiet
if errorlevel 1 (
    echo ERROR: Failed to install pynput
    echo Try: pip install pynput
    pause
    exit /b 1
)

REM Install matplotlib (optional, for visualization)
echo Installing matplotlib (optional for plots)...
python -m pip install matplotlib --quiet

REM Create output directory
if not exist C:\tmp mkdir C:\tmp
echo Created C:\tmp

echo.
echo Setup complete!
echo.
echo Next: run test_click_detection.bat to start the keylogger
echo.
pause
