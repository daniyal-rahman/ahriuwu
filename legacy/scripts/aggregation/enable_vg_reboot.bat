@echo off
REM Enable both Vanguard services (auto/system start) and reboot so the vgk
REM kernel driver loads at boot (post-boot starts are not trusted by League).
sc config vgc start= auto
sc config vgk start= system
shutdown /a 2>nul
shutdown /r /f /t 5 /c "Reboot: Vanguard ENABLED"
