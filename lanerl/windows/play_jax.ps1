# Bootstrap a normal real-client game, then hand the transport to JAX.
function Log($m) { "$([DateTime]::Now.ToString('HH:mm:ss')) $m" | Out-File -Append C:\lanerl\play_jax.log -Encoding ASCII }
Remove-Item C:\lanerl\play_jax.log,C:\lanerl\server_jax.log,C:\lanerl\server_jax.err -ErrorAction SilentlyContinue
foreach ($n in @("League of Legends","GameServerConsole")) {
  Get-Process $n -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue }
Start-Sleep -Seconds 3

$env:LANERL_BOT = "none"
$env:LANERL_HEADLESS = "1"
$env:LANERL_TOPONLY = "1"
$env:LANERL_JAX_RELAY_PORT = "5202"
$env:LANERL_JAX_TAKEOVER_MS = "135000"
Remove-Item Env:\LANERL_CONTROL_PORT,Env:\LANERL_FREERUN,Env:\LANERL_BOT_CONFIG -ErrorAction SilentlyContinue

Start-Process -FilePath "C:\lanerl\winpub_jax\GameServerConsole.exe" `
  -ArgumentList '--config','C:\lanerl\gameinfo.json' `
  -WorkingDirectory "C:\lanerl\winpub_jax" -WindowStyle Hidden `
  -RedirectStandardOutput C:\lanerl\server_jax.log -RedirectStandardError C:\lanerl\server_jax.err
Log "server started (JAX relay 5202, takeover 135000ms, bots off)"

foreach ($i in 1..90) { Start-Sleep -Seconds 1
  if ((Test-Path C:\lanerl\server_jax.log) -and (Select-String C:\lanerl\server_jax.log -Pattern "Game is ready" -Quiet)) { break } }
if (-not (Select-String C:\lanerl\server_jax.log -Pattern "LANERL_JAX_RELAY listening" -Quiet)) {
  Log "FATAL: server did not open JAX relay"; exit 2 }
Log "server ready and relay listening"

Start-Process -FilePath "cmd.exe" -ArgumentList '/c','C:\lanerl\launch_client.bat' -WindowStyle Hidden
Log "client launched"
foreach ($i in 1..90) { Start-Sleep -Seconds 2
  if (Select-String C:\lanerl\server_jax.log -Pattern "Accepted client version" -Quiet) { break } }
if (Select-String C:\lanerl\server_jax.log -Pattern "Accepted client version" -Quiet) {
  Log "client connected"
} else {
  Log "FATAL: client did not connect"; exit 3
}

foreach ($i in 1..360) { Start-Sleep -Seconds 1
  if (Select-String C:\lanerl\server_jax.log -Pattern "LANERL_JAX_RELAY ACTIVE" -Quiet) { break } }
if (Select-String C:\lanerl\server_jax.log -Pattern "LANERL_JAX_RELAY ACTIVE" -Quiet) {
  Log "JAX ACTIVE -- C# simulation frozen"
} else {
  Log "FATAL: JAX never took ownership"; exit 4
}
