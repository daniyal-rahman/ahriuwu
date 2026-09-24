using LENet;
using System;
using System.Globalization;
using System.IO;
using System.Text;
using Channel = GameServerCore.Packets.Enums.Channel;

namespace LeagueSandbox.GameServer.Lanerl
{
    /// <summary>
    /// Behaviour-neutral plaintext packet recording for the JAX emission oracle.
    ///
    /// Set LANERL_PACKET_RECORD to a JSONL path. Outbound bytes are recorded before
    /// Blowfish encryption and inbound bytes after decryption. Recording happens at
    /// PacketHandlerManager's delivery boundary, not in PacketNotifier, so a row is
    /// one packet as seen by one real client. The recorder never decodes or edits a
    /// packet and is inert when the environment variable is unset.
    /// </summary>
    public static class LanerlPacketRecorder
    {
        private static readonly object Gate = new object();
        private static StreamWriter Writer;
        private static bool Initialized;
        private static long Sequence;

        public static bool Enabled
        {
            get
            {
                EnsureInitialized();
                return Writer != null;
            }
        }

        private static void EnsureInitialized()
        {
            if (Initialized)
            {
                return;
            }

            lock (Gate)
            {
                if (Initialized)
                {
                    return;
                }

                var path = Environment.GetEnvironmentVariable("LANERL_PACKET_RECORD");
                if (!string.IsNullOrWhiteSpace(path))
                {
                    path = Path.GetFullPath(path);
                    var parent = Path.GetDirectoryName(path);
                    if (!string.IsNullOrEmpty(parent))
                    {
                        Directory.CreateDirectory(parent);
                    }
                    Writer = new StreamWriter(
                        new FileStream(path, FileMode.Create, FileAccess.Write,
                                       FileShare.Read),
                        new UTF8Encoding(false))
                    {
                        AutoFlush = true
                    };
                }
                Initialized = true;
            }
        }

        public static void Outbound(float gameTime, string route, int recipient,
                                    Channel channel, PacketFlags flags, byte[] bytes)
        {
            Write(gameTime, "out", route, recipient, channel, flags, bytes);
        }

        public static void Inbound(float gameTime, int clientId, Channel channel,
                                   PacketFlags flags, byte[] bytes)
        {
            Write(gameTime, "in", "receive", clientId, channel, flags, bytes);
        }

        private static void Write(float gameTime, string direction, string route,
                                  int clientId, Channel channel, PacketFlags flags,
                                  byte[] bytes)
        {
            EnsureInitialized();
            if (Writer == null)
            {
                return;
            }

            // All strings above are fixed literals. Bytes stay byte-exact as base64;
            // packet_id is only a cheap census hint, never a substitute for decoding.
            int rawPacketId = bytes.Length == 0 ? -1 : bytes[0];
            lock (Gate)
            {
                long seq = ++Sequence;
                Writer.Write("{\"v\":1,\"seq\":");
                Writer.Write(seq.ToString(CultureInfo.InvariantCulture));
                Writer.Write(",\"t_ms\":");
                Writer.Write(gameTime.ToString("R", CultureInfo.InvariantCulture));
                Writer.Write(",\"direction\":\"");
                Writer.Write(direction);
                Writer.Write("\",\"route\":\"");
                Writer.Write(route);
                Writer.Write("\",\"client_id\":");
                Writer.Write(clientId.ToString(CultureInfo.InvariantCulture));
                Writer.Write(",\"channel\":");
                Writer.Write(((uint)channel).ToString(CultureInfo.InvariantCulture));
                Writer.Write(",\"flags\":");
                Writer.Write(Convert.ToUInt32(flags).ToString(CultureInfo.InvariantCulture));
                Writer.Write(",\"raw_packet_id\":");
                Writer.Write(rawPacketId.ToString(CultureInfo.InvariantCulture));
                Writer.Write(",\"bytes_b64\":\"");
                Writer.Write(Convert.ToBase64String(bytes));
                Writer.WriteLine("\"}");
            }
        }
    }
}
