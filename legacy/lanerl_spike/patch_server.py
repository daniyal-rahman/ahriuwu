#!/usr/bin/env python3
"""Patch LoLServer for headless RL/verification use.

Three changes, all env-var gated so default behaviour is untouched:
  1. CheckIfAllPlayersLeft() must not SetToExit when no client ever connected --
     it counts never-connected players as departed and kills the server.
  2. TryStart() force-start requires Any(!IsDisconnected); allow zero clients.
  3. Record whole-game state to JSONL every 100ms so we can verify wave timings,
     minion pathing, turrets, gold/XP and abilities WITHOUT a rendering client.
"""
import re, sys, pathlib

# The NFS export is mounted at /srv/nfs on danilogin and /mnt/nfs on desktop.
# Never hardcode either: resolve everything from this file's own location.
_VENDOR = pathlib.Path(__file__).resolve().parents[2] / "lanerl-vendor"
ROOT = _VENDOR / "LoLServer"
game = ROOT / "GameServerLib/Game.cs"
start = ROOT / "GameServerLib/Packets/PacketHandlers/HandleStartGame.cs"

g = game.read_text()
if "LANERL_HEADLESS" not in g:
    # 1. don't exit when nobody is connected
    g = g.replace(
        "        public bool CheckIfAllPlayersLeft()\n        {\n",
        "        public bool CheckIfAllPlayersLeft()\n        {\n"
        "            // headless: never-connected players must not count as 'left'\n"
        "            if (System.Environment.GetEnvironmentVariable(\"LANERL_HEADLESS\") == \"1\") return false;\n",
        1)
    # 3. state recorder, driven from Update()
    g = g.replace(
        "        public void Update(float diff)\n        {\n",
        "        public void Update(float diff)\n        {\n"
        "            LanerlRecord(diff);\n",
        1)
    recorder = '''
        private System.IO.StreamWriter _lanerlWriter;
        private float _lanerlAccum = 1e9f;

        /// <summary>Dump full game state as JSONL at 10Hz when LANERL_RECORD is set.</summary>
        private void LanerlRecord(float diff)
        {
            var path = System.Environment.GetEnvironmentVariable("LANERL_RECORD");
            if (string.IsNullOrEmpty(path)) return;
            _lanerlAccum += diff;
            if (_lanerlAccum < 100f) return;
            _lanerlAccum = 0f;
            if (_lanerlWriter == null)
            {
                _lanerlWriter = new System.IO.StreamWriter(path, false);
                _lanerlWriter.AutoFlush = true;
            }
            var sb = new System.Text.StringBuilder(4096);
            sb.Append("{\\"t\\":").Append(((int)GameTime).ToString()).Append(",\\"u\\":[");
            bool first = true;
            foreach (var kv in ObjectManager.GetObjects())
            {
                var au = kv.Value as GameObjects.AttackableUnits.AttackableUnit;
                if (au == null) continue;
                if (!first) sb.Append(',');
                first = false;
                sb.Append("{\\"id\\":").Append(kv.Key)
                  .Append(",\\"k\\":\\"").Append(au.GetType().Name).Append("\\"")
                  .Append(",\\"tm\\":").Append((int)au.Team)
                  .Append(",\\"x\\":").Append(((int)au.Position.X).ToString())
                  .Append(",\\"y\\":").Append(((int)au.Position.Y).ToString())
                  .Append(",\\"hp\\":").Append(((int)au.Stats.CurrentHealth).ToString())
                  .Append(",\\"mhp\\":").Append(((int)au.Stats.HealthPoints.Total).ToString());
                var ch = au as GameObjects.AttackableUnits.AI.Champion;
                if (ch != null)
                    sb.Append(",\\"gold\\":").Append(((int)ch.Stats.Gold).ToString())
                      .Append(",\\"xp\\":").Append(((int)ch.Stats.Experience).ToString())
                      .Append(",\\"lvl\\":").Append(ch.Stats.Level.ToString());
                sb.Append('}');
            }
            sb.Append("]}");
            _lanerlWriter.WriteLine(sb.ToString());
        }
'''
    g = g.replace("        public void Update(float diff)", recorder + "\n        public void Update(float diff)", 1)
    game.write_text(g)
    print("patched Game.cs (headless guard + 10Hz recorder)")
else:
    print("Game.cs already patched")

s = start.read_text()
if "LANERL_HEADLESS" not in s:
    s = s.replace(
        "                isPossibleToStart = players.Any(p => !p.IsDisconnected);",
        "                isPossibleToStart = players.Any(p => !p.IsDisconnected)\n"
        "                    || System.Environment.GetEnvironmentVariable(\"LANERL_HEADLESS\") == \"1\";",
        1)
    start.write_text(s)
    print("patched HandleStartGame.cs (force-start with zero clients)")
else:
    print("HandleStartGame.cs already patched")
