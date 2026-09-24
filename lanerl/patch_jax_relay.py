#!/usr/bin/env python3
"""Install the opt-in JAX takeover transport in ``LoLServer-emit``.

Requires the plaintext-recorder patch because both deliberately share the
PacketHandlerManager delivery boundary. Exact anchors and per-site idempotence
make a partial install an error.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil


HERE = pathlib.Path(__file__).resolve().parent
RELAY_SOURCE = HERE / "vendor_patches/LanerlJaxRelay.cs"


def _default_server() -> pathlib.Path:
    candidates = [
        HERE.parents[1] / "lanerl-vendor/LoLServer-emit",
        pathlib.Path("/srv/nfs/projects/lanerl-vendor/LoLServer-emit"),
        pathlib.Path("/mnt/nfs/projects/lanerl-vendor/LoLServer-emit"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    raise SystemExit(
        "cannot find LoLServer-emit; pass --server /absolute/path/to/checkout"
    )


def replace_once(path: pathlib.Path, old: str, new: str, marker: str, what: str) -> None:
    text = path.read_text()
    if marker in text:
        print(f"already patched: {what}")
        return
    if text.count(old) != 1:
        raise SystemExit(f"{what}: expected one anchor, found {text.count(old)}")
    path.write_text(text.replace(old, new, 1))
    print(f"patched: {what}")


def install(server: pathlib.Path) -> None:
    server = server.resolve()
    manager = server / "GameServerLib/Packets/PacketHandlerManager.cs"
    game = server / "GameServerLib/Game.cs"
    if not manager.is_file() or not game.is_file():
        raise SystemExit(f"not a LoLServer checkout: {server}")
    if "LanerlPacketRecorder" not in manager.read_text():
        raise SystemExit("install patch_packet_recording.py first")

    target = server / "GameServerLib/Lanerl/LanerlJaxRelay.cs"
    if target.exists() and target.read_bytes() == RELAY_SOURCE.read_bytes():
        print("already installed: LanerlJaxRelay.cs")
    else:
        shutil.copyfile(RELAY_SOURCE, target)
        print("installed: LanerlJaxRelay.cs")

    replace_once(
        manager,
        """        private readonly Game _game;

        private readonly NetworkHandler<ICoreRequest> _netReq;
""",
        """        private readonly Game _game;
        private readonly LanerlJaxRelay _jaxRelay;

        private readonly NetworkHandler<ICoreRequest> _netReq;
""",
        "private readonly LanerlJaxRelay _jaxRelay;",
        "relay field",
    )
    replace_once(
        manager,
        """            _loadScreenConvertorTable = new Dictionary<LoadScreenPacketID, RequestConvertor>();
            InitializePacketConvertors();
""",
        """            _loadScreenConvertorTable = new Dictionary<LoadScreenPacketID, RequestConvertor>();
            InitializePacketConvertors();
            _jaxRelay = new LanerlJaxRelay();
""",
        "_jaxRelay = new LanerlJaxRelay();",
        "relay construction",
    )
    replace_once(
        manager,
        """        private bool SendPacketRecorded(int userId, byte[] source, Channel channelNo,
                                        PacketFlags flag, string route)
        {
            // Sometimes we try to send packets to a user that doesn't exist (like in broadcast when not all players are connected).
""",
        """        private bool SendPacketRecorded(int userId, byte[] source, Channel channelNo,
                                        PacketFlags flag, string route)
        {
            if (_jaxRelay.Active)
            {
                return true; // post-takeover C# gameplay is intentionally silent
            }
            // Sometimes we try to send packets to a user that doesn't exist (like in broadcast when not all players are connected).
""",
        "post-takeover C# gameplay is intentionally silent",
        "unicast suppression",
    )
    replace_once(
        manager,
        """        public bool BroadcastPacket(byte[] data, Channel channelNo, PacketFlags flag = PacketFlags.RELIABLE)
        {
            if (data.Length >= 8)
""",
        """        public bool BroadcastPacket(byte[] data, Channel channelNo, PacketFlags flag = PacketFlags.RELIABLE)
        {
            if (_jaxRelay.Active)
            {
                return true; // post-takeover C# gameplay is intentionally silent
            }
            if (data.Length >= 8)
""",
        "public bool BroadcastPacket(byte[] data, Channel channelNo, PacketFlags flag = PacketFlags.RELIABLE)\n"
        "        {\n            if (_jaxRelay.Active)",
        "broadcast suppression",
    )
    # Inject immediately before the byte[] overload so it remains unchanged.
    text = manager.read_text()
    if "public bool JaxRelayActive => _jaxRelay.Active;" not in text:
        anchor = "        public bool HandlePacket(Peer peer, byte[] data, Channel channelId)\n"
        api = """        public bool JaxRelayActive => _jaxRelay.Active;

        public void PumpJaxRelay()
        {
            _jaxRelay.Pump(SendRelayedPacket);
        }

        public void MaybeActivateJaxRelay()
        {
            _jaxRelay.MaybeActivate(_game);
        }

        private void SendRelayedPacket(int userId, byte[] source, Channel channelNo,
                                       PacketFlags flag)
        {
            if (userId < 0 || userId >= _peers.Length || _peers[userId] == null)
            {
                return;
            }
            LanerlPacketRecorder.Outbound(_game.GameTime, "jax_relay", userId,
                                          channelNo, flag, source);
            var wire = source.Length >= 8 ? _blowfishes[userId].Encrypt(source) : source;
            _peers[userId].Send((byte)channelNo, new LENet.Packet(wire, flag));
        }

"""
        if text.count(anchor) != 1:
            raise SystemExit("relay public API anchor moved")
        manager.write_text(text.replace(anchor, api + anchor, 1))
        print("patched: relay public API")
    else:
        print("already patched: relay public API")

    replace_once(
        manager,
        """            LanerlPacketRecorder.Inbound(_game.GameTime, recordedClientId,
                                         channelId, PacketFlags.NONE, data);
            return HandlePacket(peer, data, channelId);
""",
        """            LanerlPacketRecorder.Inbound(_game.GameTime, recordedClientId,
                                         channelId, PacketFlags.NONE, data);
            if (_jaxRelay.Active)
            {
                _jaxRelay.ForwardInbound(_game.GameTime, recordedClientId,
                                         channelId, PacketFlags.NONE, data);
                return true;
            }
            return HandlePacket(peer, data, channelId);
""",
        "_jaxRelay.ForwardInbound(_game.GameTime",
        "inbound forwarding",
    )

    replace_once(
        game,
        """                if (IsPaused)
                {
""",
        """                // Nonblocking: accepts Python and drains relay packets on the
                // same thread that owns LENet, before the client network poll.
                _packetServer.PacketHandlerManager.PumpJaxRelay();

                if (IsPaused)
                {
""",
        "_packetServer.PacketHandlerManager.PumpJaxRelay();",
        "relay pump",
    )
    replace_once(
        game,
        """                    if (IsRunning)
                    {
                        Update(deltaTime);
                    }
""",
        """                    if (IsRunning)
                    {
                        if (!_packetServer.PacketHandlerManager.JaxRelayActive)
                        {
                            Update(deltaTime);
                            _packetServer.PacketHandlerManager.MaybeActivateJaxRelay();
                        }
                        // Once active, no C# game tick runs. The loop remains alive
                        // only to pump JAX packets and the real client's ENet socket.
                    }
""",
        "Once active, no C# game tick runs.",
        "freeze C# world at takeover",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=pathlib.Path)
    args = parser.parse_args()
    install(args.server or _default_server())


if __name__ == "__main__":
    main()
