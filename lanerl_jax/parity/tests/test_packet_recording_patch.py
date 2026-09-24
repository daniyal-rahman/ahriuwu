from __future__ import annotations

from lanerl.patch_packet_recording import install
from lanerl.patch_jax_relay import install as install_jax_relay


PRISTINE_MANAGER = """using LeagueSandbox.GameServer;

public class PacketHandlerManager
{
        public bool SendPacket(int userId, byte[] source, Channel channelNo, PacketFlags flag = PacketFlags.RELIABLE)
        {
            // Sometimes we try to send packets to a user that doesn't exist (like in broadcast when not all players are connected).
            if (0 <= userId && userId < _peers.Length && _peers[userId] != null)
            {
                byte[] temp;
            }
            return false;
        }

        public bool BroadcastPacket(byte[] data, Channel channelNo, PacketFlags flag = PacketFlags.RELIABLE)
        {
            if (data.Length >= 8)
            {
                int failedPeers = 0;
                for(int i = 0; i < _peers.Length; i++)
                {
                    if(_peers[i] != null && _peers[i].Send((byte)channelNo, new LENet.Packet(_blowfishes[i].Encrypt(data), flag)) < 0)
                    {
                        failedPeers++;
                    }
                }
            }
            else
            {
                var packet = new LENet.Packet(data, flag);
                _server.Broadcast((byte)channelNo, packet);
                return true;
            }
        }

        public bool BroadcastPacketTeam(TeamId team, byte[] data, Channel channelNo,
            PacketFlags flag = PacketFlags.RELIABLE)
        {
            foreach (var ci in _playerManager.GetPlayers(false))
            {
                if (ci.Team == team)
                {
                    SendPacket(ci.ClientId, data, channelNo, flag);
                }
            }
            return true;
        }

        public bool BroadcastPacketVision(GameObject o, byte[] data, Channel channelNo,
            PacketFlags flag = PacketFlags.RELIABLE)
        {
            foreach (int pid in o.VisibleForPlayers)
            {
                SendPacket(pid, data, channelNo, flag);
            }
            return true;
        }

        public bool HandlePacket(Peer peer, Packet packet, Channel channelId)
        {
            var data = packet.Data;
            return HandlePacket(peer, data, channelId);
        }

        private bool HandleHandshake(Peer peer, byte[] data)
        {
            return true;
        }
}
"""


def test_packet_recorder_patch_is_complete_and_idempotent(tmp_path):
    server = tmp_path / "LoLServer-emit"
    manager = server / "GameServerLib/Packets/PacketHandlerManager.cs"
    manager.parent.mkdir(parents=True)
    (server / "GameServerLib/Lanerl").mkdir(parents=True)
    manager.write_text(PRISTINE_MANAGER)

    install(server)
    once = manager.read_text()
    install(server)
    twice = manager.read_text()

    assert twice == once
    assert once.count("LanerlPacketRecorder.Outbound") == 3
    assert once.count("LanerlPacketRecorder.Inbound") == 1
    assert 'SendPacketRecorded(ci.ClientId, data, channelNo, flag, "team")' in once
    assert 'SendPacketRecorded(pid, data, channelNo, flag, "vision")' in once
    assert (server / "GameServerLib/Lanerl/LanerlPacketRecorder.cs").is_file()


def test_jax_relay_patch_is_complete_and_idempotent_after_recorder(tmp_path):
    server = tmp_path / "LoLServer-emit"
    manager = server / "GameServerLib/Packets/PacketHandlerManager.cs"
    game = server / "GameServerLib/Game.cs"
    manager.parent.mkdir(parents=True)
    (server / "GameServerLib/Lanerl").mkdir(parents=True)
    fixture = PRISTINE_MANAGER.replace(
        "public class PacketHandlerManager\n{\n",
        """public class PacketHandlerManager
{
        private readonly Game _game;

        private readonly NetworkHandler<ICoreRequest> _netReq;

        public PacketHandlerManager()
        {
            _loadScreenConvertorTable = new Dictionary<LoadScreenPacketID, RequestConvertor>();
            InitializePacketConvertors();
        }

""",
    ).replace(
        "        public bool HandlePacket(Peer peer, Packet packet, Channel channelId)\n",
        """        public bool HandlePacket(Peer peer, byte[] data, Channel channelId)
        {
            return true;
        }

        public bool HandlePacket(Peer peer, Packet packet, Channel channelId)
""",
    )
    manager.write_text(fixture)
    game.write_text("""public class Game
{
    public void Loop()
    {
                if (IsPaused)
                {
                }
                    if (IsRunning)
                    {
                        Update(deltaTime);
                    }
    }
}
""")

    install(server)
    install_jax_relay(server)
    manager_once, game_once = manager.read_text(), game.read_text()
    install_jax_relay(server)

    assert manager.read_text() == manager_once
    assert game.read_text() == game_once
    assert "post-takeover C# gameplay is intentionally silent" in manager_once
    assert "_jaxRelay.ForwardInbound" in manager_once
    assert "SendRelayedPacket" in manager_once
    assert "JaxRelayActive" in game_once
    assert (server / "GameServerLib/Lanerl/LanerlJaxRelay.cs").is_file()
