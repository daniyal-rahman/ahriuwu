#!/usr/bin/env python3
"""Free-run patch: decouple the server tick from wall-clock.

Stock GameLoop measures REAL elapsed ms and feeds that to Update(), then sleeps
the remainder of the 16.67ms budget via NetLoop(timeout). Under LANERL_FREERUN=1
we instead feed a CONSTANT dt (identical physics, deterministic) and drop the
sleep, so the loop runs as fast as the CPU allows. This is the number that
decides whether training is weeks or months.
"""
import pathlib
# The NFS export is mounted at /srv/nfs on danilogin and /mnt/nfs on desktop.
# Never hardcode either: resolve everything from this file's own location.
_VENDOR = pathlib.Path(__file__).resolve().parents[2] / "lanerl-vendor"
game = _VENDOR / "LoLServer/GameServerLib/Game.cs"
s = game.read_text()

if "LANERL_FREERUN" in s:
    print("already patched"); raise SystemExit

# 1. constant dt instead of measured wall-clock elapsed
old_dt = """                float deltaTime = (float)lastSleepDuration;"""
new_dt = """                float deltaTime = (float)lastSleepDuration;
                // free-run: fixed logical timestep, decoupled from real time
                if (_lanerlFreeRun) deltaTime = (float)REFRESH_RATE;"""
assert old_dt in s, "deltaTime line not found"
s = s.replace(old_dt, new_dt, 1)

# 2. don't sleep the remainder of the frame budget
old_to = """                timeout = Math.Max(0, refreshRate - lastUpdateDuration - oversleep);"""
new_to = """                timeout = Math.Max(0, refreshRate - lastUpdateDuration - oversleep);
                if (_lanerlFreeRun) timeout = 0;   // never wait on the clock"""
assert old_to in s, "timeout line not found"
s = s.replace(old_to, new_to, 1)

# 3. flag + tick counter so we can measure ticks/sec
old_field = """        private System.IO.StreamWriter _lanerlWriter;"""
new_field = """        private readonly bool _lanerlFreeRun =
            System.Environment.GetEnvironmentVariable("LANERL_FREERUN") == "1";
        private long _lanerlTicks = 0;
        private System.Diagnostics.Stopwatch _lanerlClock;
        private System.IO.StreamWriter _lanerlWriter;"""
assert old_field in s
s = s.replace(old_field, new_field, 1)

# 4. report ticks/sec + game-time speedup once a second
old_rec = """        private void LanerlRecord(float diff)
        {"""
new_rec = """        private void LanerlRecord(float diff)
        {
            if (_lanerlFreeRun)
            {
                if (_lanerlClock == null) _lanerlClock = System.Diagnostics.Stopwatch.StartNew();
                _lanerlTicks++;
                if (_lanerlClock.ElapsedMilliseconds >= 2000)
                {
                    double secs = _lanerlClock.ElapsedMilliseconds / 1000.0;
                    double tps = _lanerlTicks / secs;
                    System.Console.WriteLine($"LANERL_TPS {tps:F1} ticks/s  speedup {tps / 60.0:F2}x  gametime {GameTime / 1000.0:F1}s");
                    _lanerlTicks = 0; _lanerlClock.Restart();
                }
            }"""
assert old_rec in s
s = s.replace(old_rec, new_rec, 1)

game.write_text(s)
print("patched Game.cs: free-run + tick-rate telemetry")
