# Vendored instrumentation for `AA-004` / `CFH-002` (2026-09-21)

Recovery artifacts for two dump changes made in the vendored server tree at
`/srv/nfs/projects/lanerl-vendor/`. Nothing is committed there, deliberately:
that tree carries several uncommitted local patches belonging to other work,
and committing over them would take someone else's changes with it.

## What the two changes are

* **`AA-004`** — `aacd=` publishes `Q(Math.Max(0f, remaining), StatQ)`. The
  clamp runs *before* the quantisation, so a still-positive sub-quantum
  residue flattens to a flat `0`, indistinguishable from a ready swing. Adds
  `LanerlAim.AutoAttackCooldownRemainingRaw` (no `Math.Max`) plus two additive
  dump fields: `aacdraw=` (quantised) and **`aacdbits=`** (the float32 bit
  pattern, the same idiom as the existing `xbits=`/`ybits=`). The bits are the
  load-bearing one — a quantised raw field does not settle the question,
  because rounding destroys exactly the residue at issue. `aacd=` is left
  untouched: every recorded corpus and every parser depends on it.
* **`CFH-002`** — `LaneMinionAI` populates `unitsAttackingAllies`, consumes it
  in `FoundNewTarget`, and clears it at the end of that same `OnUpdate`, so the
  dump only ever sees it post-clear. Emits the map contents immediately before
  the clear, as a `CallForHelpClear` decision-trace event. In the **isolated**
  `Content-trace` copy only — the canonical `Content/` must stay clean
  (`METH-003`: a script that fails to compile does not stop the server, it
  silently stops being the AI).

## Read these diffs correctly

`aa004_unclamped_cooldown.diff` is **not a minimal patch.** It is `git diff` of
`LanerlAim.cs` and `LanerlStateDump.cs` in full, so it captures the entire
uncommitted state of those two files — including a ~167-line `DescribeInternals`
hunk that predates this work and belongs to someone else. Only the
`AutoAttackCooldownRemainingRaw`, `aacdraw=` and `aacdbits=` additions are
`AA-004`'s. Do not replay this file blindly onto a clean tree and assume the
result is just `AA-004`.

`cfh002_and_trigger_split.diff` is canonical `LaneMinionAI.cs` against the
isolated copy, so it contains the `CallForHelpClear` emit **and** the
pre-existing `METH-002` behaviour-neutral trigger split. Both must be preserved
together; the trigger split's short-circuit order is load-bearing.

## Rebuild

    export DOTNET_ROOT=/srv/nfs/projects/lanerl-vendor/dotnet
    ops/login_capped.sh 8G 3 $DOTNET_ROOT/dotnet build \
      /srv/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/GameServerConsole.csproj \
      -c Release \
      -o /srv/nfs/projects/lanerl-vendor/LoLServer/GameServerConsole/bin/Trace/net6.0 \
      -p:SolutionDir=/srv/nfs/projects/lanerl-vendor/LoLServer/

Build to `bin/Trace/net6.0`, never over `bin/Release` (every parity run uses
that binary), and at the same directory *depth* because the server resolves
Content relative to its executable. `-p:SolutionDir=` is needed for the `lib/`
post-build copy. Drive it with `lanerl/cfg/garen1v1_trace.json`, whose
`CONTENT_PATH` is absolute — a relative path is silently ignored
(`Config.cs` falls back to `GetContentPath()` when `Directory.Exists` is false).

## Validation that was actually performed

* canonical `Content/` clean before and after (`git status --short Content/`).
* `Loaded all C# scripts from package`, exactly once per log, zero
  `Loaded some`.
* **Behaviour neutrality in the strong form**: `LANERL_STATEROW` digest
  `7e24091cef70d2aeb319` and row count `1,219,899` identical with
  instrumentation ON and OFF. Both fields live in `DescribeInternals`, which
  `Describe()` never calls, so neutrality holds by construction too — but it
  was measured rather than argued.
* both emit sites proven to execute, over a >200 s window with real wave
  combat (waves clash ~t=110 s, so a 120 s window would have exercised
  neither): `aacdbits=` on 456,527 internal lines, 128,901 `CallForHelpClear`
  events of which 1,325 carry a non-empty map.

Recordings kept at `lanerl_jax/runs/aa004_cfh/{aa_cfh_off,aa_cfh_on}/`.
