# COLLISION.md — unit-vs-unit collision and avoidance (26.19)

**Status (2026-10-04):** implemented in `lanerl_jax/sim/modern_collision.py`, called from
`modern_step._move`. It replaces the legacy C# port (`collision.resolve_collisions`, which is still
used, unchanged, by the legacy `step.py` world and its parity tests).

Tags: `CLIENT` (26.19 client bins), `WIKI` (wiki.leagueoflegends.com), `RIOT` (patch notes / dev posts),
`REPLAY` (26.9 recordings), `INFERRED`. Confidence: H / M / L.

## Sources

- Wiki [Unit collision](https://wiki.leagueoflegends.com/en-us/Unit_collision?action=raw):
  "the center of units that possess a pathing radius will always attempt to avoid intersecting with
  other units' pathing radius, and collide upon meeting it". "If a unit has ghosting, all units are allowed
  to intersect with their center". Baron, the pit Rift Herald and champion-made terrain ignore ghosting.
- Wiki [Unit size](https://wiki.leagueoflegends.com/en-us/Unit_size?action=raw) and
  [Pathing radius tip](https://wiki.leagueoflegends.com/en-us/Template:Tip_data/Pathing_radius?action=raw):
  the pathing radius is "the gameplay area they occupy for unit-collision and pathfinding logic", and size
  modifiers don't change it. The gameplay radius is the hitbox and the edge for edge-to-edge range.
- Wiki [Ghosted](https://wiki.leagueoflegends.com/en-us/Ghosted), [Ghost](https://wiki.leagueoflegends.com/en-us/Ghost),
  [Minion](https://wiki.leagueoflegends.com/en-us/Minion_(League_of_Legends)): minions "respect unit collision";
  the first wave is ghosted for 28 s (side lanes) and 18 s (mid). Patch 9.4 raised the side-lane value from 18 s.
- Riot 2015, patches 5.22–5.24 ([SurrenderAt20 red-post collection](https://www.surrenderat20.net/2015/12/red-post-collection-follow-up-on.html)):
  5.22 rebuilt "one of the systems that guides pathing (particularly through minions)". In 5.23 non-champions
  treated each other as 20 % larger so minions "get out of the way" of champions. 5.24 reverted this to the old
  behaviour. Riot's [Swarm tech blog](https://www.riotgames.com/en/news/the-tech-behind-swarm) (2024) describes
  per-unit A* that checks collisions with other minions along the path.
- Client bins (CommunityDragon 16.19, `/mnt/nfs/shared/modern-world-map-research/cdragon-16.19/`):
  `garen.bin`/`jax.bin` `pathfindingCollisionRadius = 35`; minion records (MINIONS §1.1); camp records
  (`data/modern/26.19/jungle_client.json` `pathing_radius`). `map11.bin` Characters constants
  `ai_PostAvoidanceFilterDuration 0.3`, `ai_PostAvoidanceRotFilterStrength 0.2`,
  `ai_PostAvoidanceRotFilterStrengthAccel 0.125`. Their semantics are unpublished. The names suggest a 0.3 s
  low-pass filter on the heading after an avoidance turn.

## 1. Rules

| # | Rule | Tag / conf. |
|---|---|---|
| C-1 | Every live champion, lane minion and jungle monster collides with every other one: allied and enemy champions (body-blocking), champion–minion, minion–minion, champion–monster. There are no team or type exceptions. | WIKI H |
| C-2 | Collision uses the **pathing radius**, not the gameplay radius. Garen/Jax 35; melee/caster 35.74; siege 55.74; super 55.52; camps per record (Krug 85, Red 60, Gromp 40, Blue 30, Scuttle 100, ...). The gameplay radius (champions 65, minions 48/65) is only the hitbox / attack-range edge. | WIKI H, CLIENT H |
| C-3 | Contact distance: a unit's *centre* may not enter another unit's pathing disk, so a pair stops at `max(r_i, r_j)`; champions stop 35 u apart, not 70 or 130 (`modern_collision.PAIR_RULE = "max"`). Replays (§3) rule out both 2×65 and 2×35. | WIKI M, REPLAY M |
| C-4 | Ghosted units neither block nor are blocked. Sources: Ghost, Garen E, most dashes/blinks while travelling, first-wave minions (28 s side / 18 s mid after spawning), stealth while unseen, and many kit effects. Terrain still applies. | WIKI H |
| C-5 | Wards don't collide. Structures block through their navgrid pads (terrain), not as unit circles. | INFERRED M (wiki silent; pads are terrain in the client navgrid) |
| C-6 | Movers path/steer **around** blocking units. Being blocked shows up as detours and, behind a same-direction wave, as a temporary slowdown ("creep block"). Units are not teleported apart, and no source says that standing units get shoved. | WIKI H (avoidance exists), RIOT M (algorithm opaque) |
| C-7 | Creep block still exists after the first wave. No published number; its size depends on geometry. | WIKI M |
| C-8 | Baron, the pit Herald and champion-made terrain block even ghosted units. They're not modelled (no ghost-immune unit in the 1v1 top world). | WIKI H |

## 2. Implementation (`modern_collision.resolve`)

Two vectorized phases with fixed shapes and no per-unit sequential loop (O(N²) pair math). Only slots before the
wards take part (`movers=layout()["ward0"]` = 170: 2 champions, 120 minions, 48 monsters). Ghosted, dead,
ward and structure slots are excluded from both roles (C-4, C-5).

1. **Avoidance (velocity level).** Each mover's route step (`move_step` output minus its start) is tried at heading
   offsets 0, ±20, ±40, ±60, ±80 and ±100°. The preferred side turns away from the nearest blocker of the straight
   heading. Exact dead-ahead ties are mirrored by team, because the map's team symmetry flips handedness. For each
   heading, the time of first contact against every obstacle is solved over `AVOID_HORIZON_S = 0.3` s, using
   motion relative to the obstacle (moving obstacles at their own step) and the contact distance. Per side, the
   pick is the smallest clear turn, or failing that the heading whose contact comes latest. Both sides' picks
   get a terrain check (end disk at movement clearance 35 on the unit's team mask). The better walkable one is
   taken at the route step's length; if neither is walkable, the route step stands. If even the pick meets an
   obstacle within this tick, the step ends at contact (`CONTACT_STOP`, wiki "collide upon meeting"). It always
   keeps at least 25 % of the step, so a unit is never frozen. Obstacles on the mover's goal (the chase target)
   or beyond it are ignored, so a chase ends at range without swerving. Steps longer than 60 u (blinks,
   teleports) are not steered.
2. **Separation (position level).** 3 Jacobi rounds. Each overlapping pair is pushed apart along its centre line,
   split by mobility: movers 1, standing units 0.25, so a walker yields 80 % of a mutual push. Each unit moves at
   most 20 u per round. A push whose end disk is not walkable is refused, and that unit is pinned for the
   remaining rounds so its partner takes the push. A unit that already starts inside terrain isn't held by the
   check (`DTR.eject` in TIMERS handles it). Exactly coincident pairs separate along an index-derived,
   antisymmetric direction, so the result is deterministic and independent of slot order.

Cost: 11 headings × N² contact-time solves, plus 6 `is_walkable` disk checks per unit per tick.

Tests: `lanerl_jax/sim/tests/test_modern_collision.py` covers the function: radii, soft overlap resolution,
mobility split, ghosting, no wall push, determinism and permutation invariance, coincident units, steering past
a stander or a clump, head-on passing, no swerve on the chase target, and no steering into walls.
`test_modern_world_rules.py` covers the world: both champions walk base→lane, Garen walks back through his
oncoming top wave within 1.35× the straight-line time, top waves meet and fight, and minion trades stay
symmetric.

## 3. Replay evidence and sim comparison

**Replays** (`ops/modern_collision_replay.py`; 131 of the 147 games in `lol_replays_16_9_772`, patch 26.9).
raw_mem has hero positions only, with **no minion positions**. labels `waypoint` is always null, so the latest
`clicks.json` move target was used (≤ 1 s old, ≥ 300 u away). `movement.speed` is a 0.5 s look-ahead
displacement and isn't usable as instantaneous speed.

*Champion–champion* (alive, outside the fountains): enemy pair-samples under 100 u are 0.14 %, under 65 u
0.059 %, minimum 0. In 1131 enemy runs where both heroes stood still ≥ 0.5 s under 100 u apart (Yuumi and
Tahm Kench excluded), the median separation was **68 u**; 46 % were under 65 u and 20 % under 40 u. Example:
Nami and Yunara stood 33.8 u apart for 4.7 s. So champions don't collide at 2×65 or at 2×35. That is consistent
with C-3 (`max` = 35 u), with some soft overlap.

*Creep-block proxy* (recorded champion, 2–14 min, move target ≥ 300 u away, no attack/cast, speed = path length
over 0.25 s / nominal):

| set | p5 | p10 | p25 | p50 | < 0.6 | stalls / min (median, p90 s) |
|---|---|---|---|---|---|---|
| lane, clean (no HP loss 2 s, no enemy ≤ 1200 u) | 0.68 | 0.79 | 0.91 | 0.98 | 3.2 % | 2.84 (0.20, 0.40) |
| behind own outer turret, clean | 0.80 | 0.88 | 0.93 | 1.00 | 1.5 % | 0.91 |
| off-lane, clean | 0.88 | 0.88 | 0.94 | 1.00 | 0.3 % | 0.11 |

Lane walking stalls about 25× as often as off-lane walking, but the stalls are short (0.2–0.4 s) and the median
is unaffected. 30 Hz position steps add about ±10 % jitter. The excess over off-lane (about 2.7 short stalls/min)
is an upper bound on creep block: lane-only micro-stops such as last-hit hesitation count too.

**Sim** (`ops/modern_collision_creepblock.py`: Garen walks the top lane between points 1000 u inside the two outer
turrets, 90–390 s, through both teams' waves; same path-speed metric, 30 Hz):

| world | path p1 / p5 / p10 | < 0.6 | stalls / min (median, p90 s) | chord 0.5 s near minions p1 / p5 | detours (straightness < 0.9, near) |
|---|---|---|---|---|---|
| legacy `resolve_collisions` | 0.00 / 0.00 / 0.38 | 12 % | 1.61 (**2.5, 10.8**) | 0.00 / 0.00 | 8 % |
| modern (this doc) | 1.00 / 1.00 / 1.00 | 0 % | 0 | 0.60 / 0.97 | 2 % |
| variant: horizon 0.1 s | 1.00 / 1.00 / 1.00 | 0 % | 0 | 0.06 / 0.07 (jitters in place) | 8 % |

The legacy port pinned the champion against minions for seconds at a time, far beyond anything in the replays.
The modern world never stalls; minions cost small detours only. Real lane walking has about 2.7 extra short
stalls/min. The model reproduces neither those stalls nor the 0.3 s heading lag the `ai_PostAvoidance*`
constants hint at, so it's mildly optimistic. Shortening the horizon to provoke contacts doesn't produce short
stalls; it produces in-place jitter, so the default stays at 0.3 s (U-C2).

## 4. Unresolved

| ID | Question | Default | Test scenario |
|---|---|---|---|
| U-C1 | Contact distance `max(r_i, r_j)` vs `r_i + r_j` | `max` (C-3) | Practice Tool: walk Garen into a standing allied melee minion. Contact at about 36 u centre distance means `max`; about 71 u means sum. |
| U-C2 | Meaning of `ai_PostAvoidance*` (heading filter?) | not modelled (stateless steering) | Record the heading of a champion passing a standing minion and fit a first-order filter. |
| U-C3 | Do standing units get displaced by walkers (allied minions walking through an idle champion)? | Soft, 25 % share | Stand still in the path of your own wave and measure displacement. |
| U-C4 | Do minions yield to champions (Riot 5.23 asymmetry, reverted in 5.24)? | Symmetric | Same as U-C3 with a walking champion. |
| U-C5 | Champion pathing radius for champions other than Garen/Jax | 35 constant | Per-champion `pathfindingCollisionRadius` when more kits land. |
