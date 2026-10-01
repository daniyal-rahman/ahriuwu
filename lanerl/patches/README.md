# Vendor server patches (LeagueSandbox, `/srv/nfs/projects/lanerl-vendor/LoLServer`)

The six historical patches are APPLIED to the vendor source tree (checked with `patch -R --dry-run`
on 2026-09-25) and compiled into the canonical build. They stack in this order.

| Patch | Status | What it changes | Ledger |
|---|---|---|---|
| `screen-click-v1.patch` | LIVE (base of the stack) | `click` order: button + world point, hit test on collision circles | screen-click contract 2026-09-24 |
| `server-q-cast-freeze.patch` | LIVE | Q retarget no longer freezes the champion | SERVER-001 |
| `screen-click-v2.patch` | LIVE | ground A-click = AttackMove; right-click on hostile attacks; auto-acquire needs visibility | screen-click-v2 |
| `server-hud-ability-state.patch` | LIVE | own slot-enabled bits on the wire; disabled key presses ignored | HUD correction |
| `server-dead-control.patch` | LIVE | authoritative `dead` flag; live-only input rejected while dead | dead-control defect |
| `screen-click-v3.patch` | LIVE (build `ClickV3`) | ground clicks and Moves onto unwalkable ground resolve to the closest reachable point (`GetClosestTerrainExit`) instead of a straight line into the wall | PATH-011 |

## Server builds under `GameServerConsole/bin/`

| Build | Status | Contents |
|---|---|---|
| `ClickV3/net6.0` | **CANONICAL** from 2026-09-25 22:00 (E04 onward) | all six patches |
| `DeadProbe/net6.0` | previous canonical (E01-E03 ran on it) | five patches, no click-v3 |
| `Release/net6.0` | symlink to `DeadProbe` (2026-09-25) | keeps legacy scripts resolving |
| `Trace/net6.0` | diagnostic only | 2026-09-22 build with `LANERL_SHUFFLE_ORDER` for parity floors; predates all five patches |

`HudProbe` and `ScreenClick` were deleted on 2026-09-25 with Dani's approval.


| Patch | Status | What it changes | Ledger |
|---|---|---|---|
| `server-modern-champions-26.19.patch` | ISOLATED, opt-in; not applied to shared vendor | Garen/Jax26.19 stats and Q/W/E/R lane combat, actual W mitigation/shield, dodge before on-hit, dead movement, mana/recast/target validation, per-champion skill order, modern wire state and reset | CHAMP-003 |

The modern overlay sources live in `modern-champions/`; `ops/modern_server.py`
copies the existing patched vendor into a **new** destination, replaces obsolete
champion scripts/buffs and modernizes the relevant data. It refuses an existing
destination unless `--refresh` names a tree carrying its ownership manifest.
No Git operation writes into the vendor tree. `export_patch(destination, output)`
exports the exact source/content diff; apply with `patch --binary -p1` to a copy.
The registered patch includes the C# combat self-test, inactive unless
`LANERL_MODERN_SELFTEST=1`.

Build with the vendored dotnet SDK, using one consistent canonical path prefix
(`/srv/nfs` on the login node); mixing `/srv/nfs` and `/mnt/nfs` in one MSBuild
project graph loses transitive package resolution. Use capped CPU for the build.
The isolated executable is `bin/DeadProbe/GameServerConsole.dll`, with its own
`Content`. Modern config removes historical runes/masteries; automatic item
purchases must be disabled (`LANERL_AUTOBUY=0`) for this bare-champion contract.
Shared `ClickV3` and existing runs remain historical.

CHAMP-004 adds wire schema2: own stun/silence/cast state, Jax R cast timer,
and accepted-cast counters. The collector uses counters for witnessed enemy
cast memory; W consumption/E expiry cooldown edges are not new casts.
Use `server_train --modern-champions Garen,Jax --server-dir <isolated DeadProbe>`
with `LANERL_VENDOR_ROOT` pointing at the vendor runtime. The collector writes its
own pair-specific config and validates identities/patch/schema before setup.
