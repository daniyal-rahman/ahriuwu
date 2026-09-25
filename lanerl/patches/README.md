# Vendor server patches (LeagueSandbox, `/srv/nfs/projects/lanerl-vendor/LoLServer`)

All five are APPLIED to the vendor source tree (checked with `patch -R --dry-run`
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
