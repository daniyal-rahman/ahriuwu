@echo off
REM Enable Vanguard anti-cheat services
REM Run as Administrator

echo.
echo Enabling Vanguard Anti-Cheat Services...
echo.

REM Check if running as admin
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script must run as Administrator
    echo Please right-click and select "Run as administrator"
    pause
    exit /b 1
)

REM Start Vanguard services
echo Starting vgc (Vanguard Client)...
net start vgc
if errorlevel 1 (
    echo Attempting to start vgc via Service Control Manager...
    sc start vgc
)

echo Starting vgkKicker (Vanguard Kernel)...
net start vgkKicker
if errorlevel 1 (
    echo Attempting to start vgkKicker via Service Control Manager...
    sc start vgkKicker
)

REM Verify status
echo.
echo Vanguard service status:
sc query vgc
echo.
sc query vgkKicker
echo.

echo Done. Vanguard should now be running.
echo You can verify in Services (services.msc) or Device Manager.
echo.
pause
