@echo off
REM Disable both Vanguard services and reboot so they stay stopped at boot
REM (needed to ReadProcessMemory on League.exe for memory probing).
sc config vgc start= disabled
sc config vgk start= disabled
shutdown /a 2>nul
shutdown /r /f /t 5 /c "Reboot: Vanguard DISABLED"
