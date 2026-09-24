# JAX simulation in the real client

## What is ready

`lanerl/windows/jax_session.sh` is the checked path for the first real-client
test. It publishes the isolated `LoLServer-emit` checkout for Windows, deploys
it, launches the existing 4.20 client, fast-forwards the JAX top-lane sim, and
hands packet ownership to JAX at 135 seconds.

Before takeover the ordinary server runs the normal game, including the
bootstrap that the real client requires: handshake, loading, heroes, turrets,
and the first two waves. JAX independently advances from time zero. At
takeover C# sends Python only the real NetIds and freezes its world. From that
point onward:

- decrypted client inputs go to Python and are not handled by the C# game;
- ordinary C# gameplay output is suppressed;
- only packet bytes produced by `lanerl_jax.emit.runtime` reach the client;
- a lost Python connection terminates the server instead of silently falling
  back to C# authority.

The smoke is interactive. A normal client move or attack order for client 0 is
decoded and applied to JAX champion slot 0. The current emitter renders paths,
basic-attack starts, health replication, and simulation-clock sync.

## Run it

From the repository root on the Linux host:

```bash
lanerl/windows/jax_session.sh
```

Useful controls:

```bash
lanerl/windows/jax_session.sh --status
lanerl/windows/jax_session.sh --stop
lanerl/windows/jax_session.sh --prepare-only
```

`--prepare-only` does not contact Windows. It installs both idempotent source
patches, performs the self-contained win-x64 publish, and verifies that
`/mnt/nfs/projects/lanerl-vendor/winpub_jax/GameServerConsole.exe` exists.

The normal smoke lasts 20 seconds after takeover. During that interval the
lane is already fighting, but the next wave has not spawned. Inspect:

- Linux driver log: `lanerl/logs/play_jax.out`
- Windows orchestration log: `C:\lanerl\play_jax.log`
- Windows server logs: `C:\lanerl\server_jax.log` and `.err`

Success is explicit in both directions: the Linux script prints
`JAX OWNS THE LIVE CLIENT`, while the Windows log records
`JAX ACTIVE -- C# simulation frozen`.

## Honest boundary of this first smoke

Post-takeover creation/destruction packets, missile visuals, champion spells,
and complete death/respawn presentation are not emitted yet. Therefore the
default run ends before the next wave at roughly 162.8 seconds. The runtime
refuses a default-window duration that crosses that boundary. This is a
client-visible integration test of JAX authority, movement, attacks, input,
and HP—not yet an unlimited replacement game server.

The Windows host was unreachable while this path was built, so the remaining
validation is the actual client run. Local checks already completed are:

- the patched C# server builds;
- the self-contained win-x64 publish succeeds;
- packet codecs have byte-layout tests;
- a mock C# relay accepts the warmed JAX driver and receives live packets at
  the 60 Hz loop without a post-takeover compile stall.
