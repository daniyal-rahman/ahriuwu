#!/usr/bin/env python3
"""Install the opt-in plaintext packet recorder in an isolated server checkout.

The normal ``LoLServer`` checkout is the canonical parity oracle and deliberately
has a dirty, measured working tree. This patch therefore defaults to the sibling
``LoLServer-emit`` checkout and refuses to patch ``LoLServer`` unless an explicit
``--server`` is supplied. It is idempotent and uses exact source anchors so an
upstream source move fails loudly instead of producing a partial instrument.

After patching, build with::

    export DOTNET_ROOT=/mnt/nfs/projects/lanerl-vendor/dotnet
    export PATH="$DOTNET_ROOT:$PATH"
    dotnet build GameServerConsole/GameServerConsole.csproj -c Release \
      -o GameServerConsole/bin/Emit/net6.0 -p:SolutionDir="$PWD/"

Set ``LANERL_PACKET_RECORD=/absolute/path/packets.jsonl`` when launching. The
file contains plaintext bytes before outbound encryption and after inbound
decryption. No environment variable means no file, allocation, or packet work.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil


HERE = pathlib.Path(__file__).resolve().parent
RECORDER_SOURCE = HERE / "vendor_patches/LanerlPacketRecorder.cs"
MARKER = "LanerlPacketRecorder"


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


def _replace_once(
    path: pathlib.Path, old: str, new: str, what: str, installed: str
) -> None:
    text = path.read_text()
    if installed in text:
        print(f"already patched: {what}")
        return
    if text.count(old) != 1:
        raise SystemExit(
            f"{what}: expected exactly one patch anchor, found {text.count(old)}; "
            "refusing a partial recorder"
        )
    path.write_text(text.replace(old, new, 1))
    print(f"patched: {what}")


def install(server: pathlib.Path) -> None:
    server = server.resolve()
    manager = server / "GameServerLib/Packets/PacketHandlerManager.cs"
    if not manager.is_file():
        raise SystemExit(f"not a LoLServer checkout: {server}")

    target = server / "GameServerLib/Lanerl/LanerlPacketRecorder.cs"
    if target.exists() and target.read_bytes() == RECORDER_SOURCE.read_bytes():
        print("already installed: LanerlPacketRecorder.cs")
    else:
        shutil.copyfile(RECORDER_SOURCE, target)
        print("installed: LanerlPacketRecorder.cs")

    _replace_once(
        manager,
        "using LeagueSandbox.GameServer;\n",
        "using LeagueSandbox.GameServer;\n"
        "using LeagueSandbox.GameServer.Lanerl;\n",
        "recorder namespace",
        "using LeagueSandbox.GameServer.Lanerl;",
    )

    _replace_once(
        manager,
        """        public bool SendPacket(int userId, byte[] source, Channel channelNo, PacketFlags flag = PacketFlags.RELIABLE)
        {
            // Sometimes we try to send packets to a user that doesn't exist (like in broadcast when not all players are connected).
            if (0 <= userId && userId < _peers.Length && _peers[userId] != null)
            {
                byte[] temp;
""",
        """        public bool SendPacket(int userId, byte[] source, Channel channelNo, PacketFlags flag = PacketFlags.RELIABLE)
        {
            return SendPacketRecorded(userId, source, channelNo, flag, "send");
        }

        private bool SendPacketRecorded(int userId, byte[] source, Channel channelNo,
                                        PacketFlags flag, string route)
        {
            // Sometimes we try to send packets to a user that doesn't exist (like in broadcast when not all players are connected).
            if (0 <= userId && userId < _peers.Length && _peers[userId] != null)
            {
                LanerlPacketRecorder.Outbound(_game.GameTime, route, userId,
                                              channelNo, flag, source);
                byte[] temp;
        """,
        "SendPacket delivery boundary",
        "return SendPacketRecorded(userId, source, channelNo, flag, \"send\");",
    )

    _replace_once(
        manager,
        """                    if(_peers[i] != null && _peers[i].Send((byte)channelNo, new LENet.Packet(_blowfishes[i].Encrypt(data), flag)) < 0)
                    {
                        failedPeers++;
                    }
""",
        """                    if (_peers[i] != null)
                    {
                        LanerlPacketRecorder.Outbound(_game.GameTime, "broadcast", i,
                                                      channelNo, flag, data);
                        if (_peers[i].Send((byte)channelNo,
                                           new LENet.Packet(_blowfishes[i].Encrypt(data), flag)) < 0)
                        {
                            failedPeers++;
                        }
                    }
""",
        "encrypted BroadcastPacket delivery boundary",
        'LanerlPacketRecorder.Outbound(_game.GameTime, "broadcast", i,',
    )

    _replace_once(
        manager,
        """            else
            {
                var packet = new LENet.Packet(data, flag);
                _server.Broadcast((byte)channelNo, packet);
                return true;
            }
""",
        """            else
            {
                // LENet's host broadcast hides the recipient set. Expand it here so
                // the trace keeps the same one-row-per-client contract as long packets.
                for (int i = 0; i < _peers.Length; i++)
                {
                    if (_peers[i] != null)
                    {
                        LanerlPacketRecorder.Outbound(_game.GameTime, "broadcast", i,
                                                      channelNo, flag, data);
                    }
                }
                var packet = new LENet.Packet(data, flag);
                _server.Broadcast((byte)channelNo, packet);
                return true;
            }
""",
        "short BroadcastPacket delivery boundary",
        "LENet's host broadcast hides the recipient set",
    )

    # These two methods used to call public SendPacket, which would label every
    # row as route=send. Route is observation metadata only; delivery is unchanged.
    text = manager.read_text()
    team_old = """                if (ci.Team == team)
                {
                    SendPacket(ci.ClientId, data, channelNo, flag);
                }
"""
    team_new = """                if (ci.Team == team)
                {
                    SendPacketRecorded(ci.ClientId, data, channelNo, flag, "team");
                }
"""
    vision_old = """            foreach (int pid in o.VisibleForPlayers)
            {
                SendPacket(pid, data, channelNo, flag);
            }
"""
    vision_new = """            foreach (int pid in o.VisibleForPlayers)
            {
                SendPacketRecorded(pid, data, channelNo, flag, "vision");
            }
"""
    if 'SendPacketRecorded(ci.ClientId, data, channelNo, flag, "team")' not in text:
        if text.count(team_old) != 1 or text.count(vision_old) != 1:
            raise SystemExit("team/vision anchors moved; refusing a partial recorder")
        text = text.replace(team_old, team_new, 1).replace(vision_old, vision_new, 1)
        manager.write_text(text)
        print("patched: team and vision route labels")
    else:
        print("already patched: team and vision route labels")

    _replace_once(
        manager,
        """            return HandlePacket(peer, data, channelId);
        }

        private bool HandleHandshake(Peer peer, byte[] data)
""",
        """            int recordedClientId = ((int)peer.UserData) - 1;
            LanerlPacketRecorder.Inbound(_game.GameTime, recordedClientId,
                                         channelId, PacketFlags.NONE, data);
            return HandlePacket(peer, data, channelId);
        }

        private bool HandleHandshake(Peer peer, byte[] data)
""",
        "inbound decrypted delivery boundary",
        "LanerlPacketRecorder.Inbound(_game.GameTime, recordedClientId,",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=pathlib.Path, help="LoLServer checkout")
    args = parser.parse_args()
    install(args.server or _default_server())


if __name__ == "__main__":
    main()
