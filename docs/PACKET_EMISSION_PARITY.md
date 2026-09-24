# Packet-emission parity

## Contract

The two packet streams are independent.

- Ground truth is plaintext traffic from a real LoLServer game and every
  decrypted client input. `PacketHandlerManager` records the delivery boundary;
  it does not decode, transform, or manufacture state.
- The system under test will be a JAX-side function of
  `(previous_state, state, orders)`. It must not read the server trace except in
  the diff harness.
- A packet type is never silently discarded. Every observed type is in-scope,
  an input, explicitly out-of-scope, or `unscored`; an unscored row keeps the
  accounting gate red.

This is separate from the existing state differential. State injection asks
whether one JAX tick reaches the right next state. Packet parity asks whether
that state transition causes the same externally visible protocol events.

## Current status

The recording slice is implemented and locally verified:

- `lanerl/patch_packet_recording.py` installs an idempotent, opt-in patch in the
  isolated `LoLServer-emit` checkout.
- `lanerl/vendor_patches/LanerlPacketRecorder.cs` writes one JSONL row per real
  recipient before outbound Blowfish encryption, and writes inbound bytes after
  decryption. Team and vision paths retain their route without double-counting.
- `lanerl_jax/parity/emissions.py` strictly loads the trace, validates exact
  bytes and sequence continuity, handles extended packet IDs, reports scope by
  type, and exports the existing `LeaguePacketsSerializer` input format.
- The isolated server builds at
  `GameServerConsole/bin/Emit/net6.0/GameServerConsole.dll`.

The vendor `LeaguePackets` build does support reading as well as writing:
`BasePacket.Create(bytes, channel)` dispatches to packet-specific `ReadBody`
implementations, and `LeaguePacketsSerializer` already uses that path. Custom
binary decoders are therefore not needed for the first pass.

The first real recording is pending only because the Windows host was
unreachable at the time this slice was built (SSH to `windows`, 2026-09-24).

## Build the recorder server

On either Linux node (the NFS root may appear as `/srv/nfs` or `/mnt/nfs`):

```bash
python3 lanerl/patch_packet_recording.py
cd /mnt/nfs/projects/lanerl-vendor/LoLServer-emit
export DOTNET_ROOT=/mnt/nfs/projects/lanerl-vendor/dotnet
export PATH="$DOTNET_ROOT:$PATH"
dotnet build GameServerConsole/GameServerConsole.csproj -c Release \
  -o GameServerConsole/bin/Emit/net6.0 -p:SolutionDir="$PWD/"
```

The patch is inert unless `LANERL_PACKET_RECORD` is set. A real-game launcher
must also set `LANERL_TOPONLY=1`, retain the state dump used by the injector,
and use the same two-Garen config as the simulator.

On Windows, set the recording path inside the PowerShell launcher before
starting `GameServerConsole.exe`:

```powershell
$env:LANERL_PACKET_RECORD = "C:\lanerl\recordings\packets.jsonl"
$env:LANERL_TOPONLY = "1"
```

Use the existing checked workflow in `lanerl/windows/session.sh` for publish,
deployment, scheduled-task launch, client launch, SSH tunnel verification, and
policy/input attachment. The recorder should be added to a copy of that
launcher rather than weakening its checks.

## Validate and decode a recording

Copy the JSONL file back, then run:

```bash
python -m lanerl_jax.parity.emissions packets.jsonl \
  --client-id 0 --serializer-input client0.rlp.json
```

The command exits nonzero if any packet type is `unscored`. Decode the exported
file with the vendored serializer:

```bash
cd /mnt/nfs/projects/lanerl-vendor/LoLServer-emit/LeaguePackets
$DOTNET_ROOT/dotnet run --project LeaguePacketsSerializer/LeaguePacketsSerializer.csproj \
  -- /absolute/path/client0.rlp.json
```

The serializer writes `.serialized.json`, `.softbad.json`, and `.hardbad.json`.
The recorder acceptance check is zero hard decode failures for in-scope output;
soft failures (unparsed trailing bytes) must be listed by packet ID and cannot
be called decoded without inspecting whether the missing bytes are scored.

## Next implementation slice

1. Record one deterministic real client/server game and freeze its packet,
   state, and client-input streams together.
2. Expand the scope table from the observed census. Every new classification
   should cite the concrete packet and why it is or is not modeled.
3. Normalize LeaguePackets output into field-level events and establish the
   spawn-based NetId mapping.
4. Implement the independent JAX emitter in small packet families (spawn and
   waypoint first, then attacks/missiles, replication/death/level, spells and
   buffs), with a one-step diff after each family.
5. Only after field-level diffing works, add the byte relay for rendered client
   playtesting.
