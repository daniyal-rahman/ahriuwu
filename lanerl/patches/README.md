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

## Server builds under `GameServerConsole/bin/`

| Build | Status | Contents |
|---|---|---|
| `DeadProbe/net6.0` | **CANONICAL** (all training and evaluation) | all five patches |
| `HudProbe/net6.0` | superseded | through the HUD patch |
| `ScreenClick/net6.0` | superseded | through click v1 |
| `Release/net6.0` | superseded | pre-HUD click v2 |
| `Trace/net6.0` | diagnostic only | 2026-09-22 build with `LANERL_SHUFFLE_ORDER` for parity floors; predates all five patches |

Deleting the superseded builds is irreversible and needs Dani's approval.
