# Rendered RL inference: our policy drives BOTH champions, the client renders.
#
# Differs from play.ps1 (two scripted bots) in exactly three ways:
#   LANERL_BOT   = none  -- the control channel drives the champions, not bots
#   LANERL_CONTROL_PORT  -- opened, loopback-only; reach it via an SSH tunnel
#   winpub_rl            -- built from CURRENT source, so the wire carries the
#                           ad/sl/mt fields the policy was trained on
#
# The server BLOCKS in LanerlControl's constructor until a trainer attaches, so
# the policy must be running (or about to be) or the game never starts.
function Log($m) { "$([DateTime]::Now.ToString('HH:mm:ss')) $m" | Out-File -Append C:\lanerl\play_rl.log -Encoding ASCII }
Remove-Item C:\lanerl\play_rl.log,C:\lanerl\server_rl.log,C:\lanerl\server_rl.err -ErrorAction SilentlyContinue
foreach ($n in @("ffmpeg","League of Legends","GameServerConsole")) {
  Get-Process $n -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3

$env:LANERL_BOT          = "none"   # champions are ours to drive
$env:LANERL_HEADLESS     = "1"      # or the server quits mid connect-retry
$env:LANERL_CONTROL_PORT = "5200"
$env:LANERL_STEP_TICKS   = "2"      # 30 Hz, matching training
Remove-Item Env:\LANERL_FREERUN -ErrorAction SilentlyContinue  # REAL TIME or the client renders nothing
# TOPONLY=1 -- MATCH TRAINING. play.ps1 clears this because it drives scripted
# bots, which do not care what map they are on. A trained policy does: every
# run in this project uses vec.ServerLaunchSpec.toponly=True, which disables
# jungle camps and every non-top minion wave.
#
# Running the policy on the full three-lane map is a train/deploy observation
# shift, and it is not subtle. Measured on the first attempt: at 180 s both
# champions were level 1 with 0 CS, parked at (4312, 10537) -- off the top-lane
# axis, which runs (574, 10220) -> (3911, 13654). They had walked somewhere
# that only exists on the map they were never trained on, and collected no
# experience at all.
$env:LANERL_TOPONLY = "1"
Remove-Item Env:\LANERL_BOT_CONFIG -ErrorAction SilentlyContinue

Start-Process -FilePath "C:\lanerl\winpub_rl\GameServerConsole.exe" `
  -ArgumentList '--config','C:\lanerl\gameinfo.json' `
  -WorkingDirectory "C:\lanerl\winpub_rl" -WindowStyle Hidden `
  -RedirectStandardOutput C:\lanerl\server_rl.log -RedirectStandardError C:\lanerl\server_rl.err
Log "server started (control port 5200, bots off)"

foreach ($i in 1..90) { Start-Sleep -Seconds 1
  if ((Test-Path C:\lanerl\server_rl.log) -and (Select-String C:\lanerl\server_rl.log -Pattern "Game is ready" -Quiet)) { break } }
Log "server ready"

Start-Process -FilePath "cmd.exe" -ArgumentList '/c','C:\lanerl\launch_client.bat' -WindowStyle Hidden
Log "client launched"
foreach ($i in 1..90) { Start-Sleep -Seconds 2
  if (Select-String C:\lanerl\server_rl.log -Pattern "Accepted client version" -Quiet) { break } }
Log "client connected"

foreach ($i in 1..300) { Start-Sleep -Seconds 1
  if (Select-String C:\lanerl\server_rl.log -Pattern "LANERL_CONTROL client attached" -Quiet) { break } }
if (Select-String C:\lanerl\server_rl.log -Pattern "LANERL_CONTROL client attached" -Quiet) {
  Log "POLICY ATTACHED -- game is live"
} else {
  Log "NO POLICY ATTACHED: the server is blocked waiting for one. Start the tunnel and play_remote.py."
}
# consoles from the scheduled task would sit on top of the game
foreach ($p in @("cmd","conhost","WindowsTerminal","powershell")) {
  Get-Process $p -ErrorAction SilentlyContinue | ForEach-Object {
    if ($_.MainWindowHandle -ne 0) { $null = (New-Object -ComObject WScript.Shell).AppActivate($_.Id) } } }
