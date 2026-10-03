# VISION.md — 26.19 fog of war (modern world)

**Status (2026-10-02).** Fog of war is implemented in `lanerl_jax/sim/modern_vision.py` and wired through
`modern_step`. Wards, trinkets, stealth, true sight, Faelights, nearsight and dynamic terrain are deferred
(MODERN-009).

## Rules

| ID | Rule | Value | Evidence |
|---|---|---|---|
| VIS.01 | A unit is visible to a team when a live unit of that team has it within the viewer's sight radius (centre to centre) on a clear ray. | — | Wiki Sight (2026-10): "measurements use center-to-center". |
| VIS.02 | Champion sight radius. | 1350 | Wiki Sight. The champion records carry no `perceptionBubbleRadius` override. |
| VIS.03 | Melee, caster and siege minion sight radius. | 1200 | Client `perceptionBubbleRadius` 1200.0 in `SRU_{Order,Chaos}Minion{Melee,Ranged,Siege}`, and the wiki. |
| VIS.04 | Super minion and turret sight radius. | 1350 | Wiki Sight. No client override. |
| VIS.05 | Nexus sight radius. | 1350 | Client `Nexus` `perceptionBubbleRadius` 1350.0. |
| VIS.06 | Inhibitors grant no sight. | 0 | INFERRED-L: no client value and not listed on the wiki. Irrelevant to the top lane. |
| VIS.07 | Dead units grant no sight and are not visible. | — | Standard. |
| VIS.08 | Walls (navgrid bit 0x2) block sight, unless the cell is a transparent wall (0x40) or always visible (0x100). | — | Wiki Sight: impassable terrain is "always opaque". The flag names come from the NGRID v7 reader. |
| VIS.09 | Brush (navgrid bit 0x1) is opaque from the outside in, and blocks a ray passing through it. From inside a brush a unit sees out, and sees units in the same brush when the ray stays in brush. | — | Wiki Brush/Sight: "opaque towards vision when viewed from the outside inwards and not the reverse". 40 edge-connected brush patches on the 26.19 grid (wiki: 39). |
| VIS.10 | Structures (turrets, inhibitors, Nexuses) are never fogged. | — | Structures are always shown; this matches the legacy server's `IsAffectedByFoW => false`. |
| VIS.11 | A team always sees its own units. | — | Standard. |
| VIS.12 | Attack reveal: a champion hidden from the enemy team that launches a basic attack, or starts a unit-targeted ability, reveals a fixed circle at its position to the enemy team. The circle ignores walls and brush. | 300 u, 2 s | Wiki Sight ("300-unit radius … for 2 seconds after the attack completes") and Brush ("using targeted attacks and abilities will reveal a 300 radius … for 2 seconds"). The circle is fixed where the attack happened (INFERRED-M). |
| VIS.13 | Non-champion units do not aggro on what their team cannot see. Champions cannot attack, cast at or summoner-target a unit their team cannot see, and an attack order is dropped when its target enters fog. | — | Wiki Brush: "Non-champion units will not trigger aggro against an enemy unless they can see them". Standard client targeting. |

## Fog modes and known limitation

The default is `fog="rays"`, which applies VIS.08/VIS.09 as written. On the RTX 5080 it costs nothing
measurable against no fog at all (job 2203: 11.17k vs 11.21k env-ticks/s at 512 envs, 7.80k vs 7.79k at 64), because
the fused CUDA ray kernel is cheap next to the rest of the tick. The optional `fog="fast"` is the legacy VIS-FAST
lane approximation: a unit is visible when it is in a viewer's
sight radius and outside every brush, or in the viewer's own brush patch. It is a per-unit lookup with no ray
traversal. **Known limitation (VIS-FAST-M):** walls do not block sight (VIS.08 not applied), and a brush between
viewer and target does not block it. In the top lane this mostly matters for units in the jungle or river behind
the lane walls. It is kept for CPU runs, where the reference ray loop is slower.

## Unresolved

- **U-VIS-1.** The map11 constants `ca_RevealAttackerRange` = 400, `ca_RevealAttackerTimeOut` = 4.5 and
  `ca_RevealAlliesWithAttacker` have no documented trigger. They are kept as
  `CLIENT_REVEAL_ATTACKER_RANGE`/`_TIMEOUT` and not used.
- **U-VIS-2.** Whether the reveal circle follows the attacker is unconfirmed; it is modelled as fixed.
- **U-VIS-3.** Seeing from one brush into a different brush is not modelled. The ray rule allows it only when
  every crossed cell is brush.

## Implementation

- **State.** `ModernState.visible` (2, N) is the team mask and `ModernState.sight` (N, N) is each unit's own
  sight. Both are computed at the end of every tick from final positions, so the next tick's AI and the
  observation read the same mask. `ModernState.reveal` holds the attack-reveal circles. `refresh_visibility(s,
  cfg)` recomputes the mask after a state is edited by hand.
- **Rays (`fog="rays"`).** Rays use the legacy supercover caster (`obs.vision.clear_ray`) over the modern navgrid: a fused
  CUDA kernel on GPU and the reference loop on CPU. They are cast only from enemy viewers in range to the
  champion and minion slots; structures need none.
- **Consumers.**
  - lane AI `select_targets(visible=…)`;
  - champion attack, cast and summoner targets;
  - idle auto-attack;
  - rune events `sight`/`visible` (Overgrowth, Approach Velocity);
  - the `modern-world-v1` observation (only visible units are slotted);
  - `ChampionLayer.seen_cast` (an enemy cast is remembered only if it was seen);
  - click hit-testing in `train/modern_actions.py`.
- **Switch.** `build_config(…, fog="fast" | "rays" | False)`; `False` restores the all-visible world.
