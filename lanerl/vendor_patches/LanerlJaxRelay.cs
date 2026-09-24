using GameServerCore.Enums;
using LENet;
using LeagueSandbox.GameServer.GameObjects.AttackableUnits;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Text.Json;
using Channel = GameServerCore.Packets.Enums.Channel;

namespace LeagueSandbox.GameServer.Lanerl
{
    /// <summary>
    /// Transport-only handoff from the real client bootstrap to a JAX-owned game.
    ///
    /// The ordinary server remains authoritative until Python sends {"cmd":"ready"}
    /// and GameTime reaches LANERL_JAX_TAKEOVER_MS. At that instant a single
    /// bootstrap identity map is published and Active becomes true. Game.cs then
    /// stops advancing the C# world; PacketHandlerManager suppresses its outbound
    /// gameplay and forwards decrypted client packets here. Only byte arrays sent
    /// by Python are delivered after takeover.
    /// </summary>
    public sealed class LanerlJaxRelay
    {
        private readonly TcpListener _listener;
        private readonly int _port;
        private readonly float _takeoverMs;
        private TcpClient _client;
        private NetworkStream _stream;
        private readonly byte[] _readBuffer = new byte[65536];
        private readonly StringBuilder _pending = new StringBuilder();
        private bool _pythonReady;

        public bool Enabled { get; }
        public bool Active { get; private set; }

        public LanerlJaxRelay()
        {
            var rawPort = Environment.GetEnvironmentVariable("LANERL_JAX_RELAY_PORT");
            Enabled = int.TryParse(rawPort, out _port) && _port > 0;
            if (!Enabled)
            {
                return;
            }
            var rawAt = Environment.GetEnvironmentVariable("LANERL_JAX_TAKEOVER_MS");
            _takeoverMs = float.TryParse(rawAt, NumberStyles.Float,
                CultureInfo.InvariantCulture, out var at) ? at : 95000f;
            _listener = new TcpListener(IPAddress.Loopback, _port);
            _listener.Server.SetSocketOption(
                SocketOptionLevel.Socket, SocketOptionName.ReuseAddress, true);
            _listener.Start(1);
            Console.WriteLine($"LANERL_JAX_RELAY listening on {_port}, takeover={_takeoverMs:R}ms");
        }

        private void AcceptIfPending()
        {
            if (!Enabled || _client != null || !_listener.Pending())
            {
                return;
            }
            _client = _listener.AcceptTcpClient();
            _client.NoDelay = true;
            _stream = _client.GetStream();
            Console.WriteLine("LANERL_JAX_RELAY python connected");
        }

        public void Pump(Action<int, byte[], Channel, PacketFlags> sendPacket)
        {
            if (!Enabled)
            {
                return;
            }
            AcceptIfPending();
            if (_stream == null)
            {
                return;
            }
            try
            {
                if (_client.Client.Poll(0, SelectMode.SelectRead)
                    && _client.Client.Available == 0)
                {
                    Disconnect("EOF");
                    return;
                }
                while (_stream.DataAvailable)
                {
                    int count = _stream.Read(_readBuffer, 0, _readBuffer.Length);
                    if (count <= 0)
                    {
                        Disconnect("EOF");
                        return;
                    }
                    _pending.Append(Encoding.UTF8.GetString(_readBuffer, 0, count));
                }
                while (true)
                {
                    var text = _pending.ToString();
                    int newline = text.IndexOf('\n');
                    if (newline < 0)
                    {
                        break;
                    }
                    var line = text.Substring(0, newline).TrimEnd('\r');
                    _pending.Clear();
                    _pending.Append(text.Substring(newline + 1));
                    if (line.Length > 0)
                    {
                        HandleLine(line, sendPacket);
                    }
                }
            }
            catch (Exception ex) when (ex is IOException || ex is SocketException ||
                                       ex is ObjectDisposedException)
            {
                Disconnect(ex.GetType().Name + ": " + ex.Message);
            }
        }

        private void HandleLine(string line,
            Action<int, byte[], Channel, PacketFlags> sendPacket)
        {
            using var doc = JsonDocument.Parse(line);
            var root = doc.RootElement;
            if (!root.TryGetProperty("cmd", out var cmdNode))
            {
                throw new InvalidDataException("relay line has no cmd");
            }
            var cmd = cmdNode.GetString();
            if (cmd == "ready")
            {
                _pythonReady = true;
                Console.WriteLine("LANERL_JAX_RELAY python ready");
                return;
            }
            if (cmd != "packet")
            {
                throw new InvalidDataException("unknown relay cmd: " + cmd);
            }
            if (!Active)
            {
                throw new InvalidDataException("packet received before takeover");
            }
            int clientId = root.GetProperty("client_id").GetInt32();
            var channel = (Channel)root.GetProperty("channel").GetInt32();
            var flagName = root.GetProperty("flags").GetString();
            var flags = flagName == "unsequenced"
                ? PacketFlags.UNSEQUENCED : PacketFlags.RELIABLE;
            var bytes = Convert.FromBase64String(root.GetProperty("bytes_b64").GetString());
            sendPacket(clientId, bytes, channel, flags);
        }

        public void MaybeActivate(Game game)
        {
            if (!Enabled || Active)
            {
                return;
            }
            AcceptIfPending();
            if (_stream == null || !_pythonReady || game.GameTime < _takeoverMs)
            {
                return;
            }
            Active = true;
            SendBootstrap(game);
            Console.WriteLine($"LANERL_JAX_RELAY ACTIVE at t={game.GameTime:R}; C# world frozen");
        }

        private void SendBootstrap(Game game)
        {
            var sb = new StringBuilder(16384);
            sb.Append("{\"event\":\"active\",\"t_ms\":")
              .Append(game.GameTime.ToString("R", CultureInfo.InvariantCulture))
              .Append(",\"map_center_x\":")
              .Append(game.Map.NavigationGrid.MiddleOfMap.X.ToString("R", CultureInfo.InvariantCulture))
              .Append(",\"map_center_y\":")
              .Append(game.Map.NavigationGrid.MiddleOfMap.Y.ToString("R", CultureInfo.InvariantCulture))
              .Append(",\"objects\":[");
            bool first = true;
            foreach (var pair in game.ObjectManager.GetObjects())
            {
                if (!(pair.Value is AttackableUnit unit))
                {
                    continue;
                }
                if (!first) sb.Append(',');
                first = false;
                sb.Append("{\"net_id\":").Append(unit.NetId)
                  .Append(",\"kind\":\"").Append(unit.GetType().Name)
                  .Append("\",\"team\":").Append((int)unit.Team)
                  .Append(",\"model\":").Append(JsonSerializer.Serialize(unit.Model ?? ""))
                  .Append(",\"x\":").Append(unit.Position.X.ToString("R", CultureInfo.InvariantCulture))
                  .Append(",\"y\":").Append(unit.Position.Y.ToString("R", CultureInfo.InvariantCulture))
                  .Append(",\"hp\":").Append(unit.Stats.CurrentHealth.ToString("R", CultureInfo.InvariantCulture))
                  .Append(",\"max_hp\":").Append(unit.Stats.HealthPoints.Total.ToString("R", CultureInfo.InvariantCulture))
                  .Append('}');
            }
            sb.Append("]}");
            SendLine(sb.ToString());
        }

        public void ForwardInbound(float gameTime, int clientId, Channel channel,
                                   PacketFlags flags, byte[] bytes)
        {
            if (!Active)
            {
                return;
            }
            var payload = JsonSerializer.Serialize(new Dictionary<string, object>
            {
                ["event"] = "client_packet",
                ["t_ms"] = gameTime,
                ["client_id"] = clientId,
                ["channel"] = (uint)channel,
                ["flags"] = Convert.ToUInt32(flags),
                ["bytes_b64"] = Convert.ToBase64String(bytes),
            });
            SendLine(payload);
        }

        private void SendLine(string line)
        {
            if (_stream == null)
            {
                return;
            }
            var bytes = Encoding.UTF8.GetBytes(line + "\n");
            _stream.Write(bytes, 0, bytes.Length);
            _stream.Flush();
        }

        private void Disconnect(string why)
        {
            Console.WriteLine("LANERL_JAX_RELAY disconnected: " + why);
            try { _stream?.Dispose(); } catch { }
            try { _client?.Dispose(); } catch { }
            _stream = null;
            _client = null;
            _pythonReady = false;
            if (Active)
            {
                Console.WriteLine("LANERL_FATAL JAX relay left after takeover; exiting");
                Environment.Exit(98);
            }
        }
    }
}
