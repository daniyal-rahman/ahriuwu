# Playtest sweep: tick-by-tick checks on full 600 s games (2026-09-24)

Uncommitted. An automated "playtest": complete 10-minute mirror games,
recording every server tick, then checked for anything a person watching the
game would notice as wrong. Each finding says what it looks like on screen,
for comparison with the streamed-client playtest.

## Headline

The sim holds up on everything a spectator reads first. Across 24 env-episodes
(144 minutes of game, 864k ticks, 16M live-unit ticks) the checks found:

- **no** unexplained speed-ups,
- **no** hp that rises or falls without a cause,
- **no** hp above max,
- **no** live unit at 0 hp,
- **no** gold, XP, CS or level-up that doesn't add up,
- **no** unit leaving the map,
- **no** dead minion that moves or keeps a target,
- **no** champion swinging at an ally,
- **no** turret shooting out of range,
- **no** E spin with more than 6 ticks.

Movement speeds, attack cadences, wave timing and spawn points all match the
content values.

**There is one clearly visible bug, (c)1, and it is real.** When a minion
slot is reused, a caster or cannon that was winding up an attack on the
minion that just died fires at the **newly spawned ally** in that slot, at
its own barracks. The missile then crosses the whole lane (7-11k units,
~10-13 s of flight) and hits its own teammate for 23-40 damage. It happens in
every condition (4-16 times per 6 games) and it reproduces deterministically
on CPU from a dumped state.

Three low-severity (c) items follow it. Everything else the eye would catch
is either a booked approximation or the reference server's own behaviour.
The most visible of those is champions grinding along walls, popping 40-100
units out of terrain about once a second. That mostly comes from the
straight-line attack chase (`PATH-009`, (a)).

## Setup

- **Policies:**
  - trained: `diag1b-20260923-231936-54371e99/ckpt_latest.msgpack`, sampled,
    both champions (mirror). Final CS 27.9 mean in training config
    (8-38), 30.8 in scripted.
  - random: uniform over every factored head.
- **Sim configs:** `SimConfig.training()` (deferred terrain repair, routed)
  and `SimConfig.scripted(route_artifact=DEFAULT_ROUTE_ARTIFACT)` (inline
  repair, same routing). So the two differ only in the collision-terrain
  mode.
- **Runs:** 6 envs x 600 s per condition. Four conditions, 24 games in total.
- **Recording:** the trainer's own obs -> policy -> `orders_from` ->
  `env_apply` wiring. Each decision advances with `cfg.replace(step_ticks=1)`
  twice (the `SimConfig.gate` construction), and both ticks are recorded.
- **Hardware and time:** RTX 5080 (slurm 1340/1341), about 2 min per
  condition. The sim, obs, policy and actions code is identical between the
  recording commit (`a4a1bd0`) and the current `HEAD` (`49e703d`).
- **Scratch:** everything lives under `/tmp/playtest/`. `data/` there is a
  symlink to `/srv/nfs/shared/playtest-20260924` (5.2 GB, so the desktop node
  could write it).
  - `data/record.py`: the recorder.
  - `analyze.py`: all checks. It writes `data/<cond>/report.json`.
  - `data/repro2.py`: search-based dumper of full pre-decision `LaneState`s.
  - `replay.py`: CPU tick replay of a dump.

"Tick" below means a 16.67 ms server tick. `t` is game time.

## Findings by check

Counts are summed over 6 games per condition, in the order
**training/trained, scripted/trained, training/random, scripted/random**.

### 1. Teleports and push-outs

- **Move component above speed x dt x 1.01, alive units:** 0 / 0 / 0 / 0.
  - Minions measured 5.4167 u/tick (325 u/s).
  - Garen measured 5.750 u/tick (345). With Q haste it measured 7.7625
    u/tick (345 x 1.35).
- **Documented teleports are exact:**
  - Every respawn and completed recall lands on the champion spawn point to
    0.00 u.
  - Minion spawns land on `MINION_SPAWN` to 0.00 u.
- **Collision push-outs** (the collision phase moving a unit before it
  walks):
  - Unit-unit push-outs: p50 0.7 u, p90 5.7, p99 22, max 111 u.
  - Terrain exits: 38-105 u, p50 42, and always a single tick.
  - Out of 3.8M mover-ticks, 13k push-outs exceed 20 u (training/trained).
- **Largest pushes:**
  - Garen popping 70-111 u sideways when an E spin ends inside a minion
    pack (ghosting stops, `SPELL-005`). Example: training/trained env5
    t=164.47 s, red Garen, 110.8 u.
  - Cannon minions squeezed in a clump, up to 101 u.
  - (b) The server's escape is a "teleport to touching", not a spring
    (`COLL-001` VERIFIED).
  - **On screen:** single-frame hops of up to a unit-width. The biggest ones
    come right after Garen's spin ends in a wave.
- **Fountain turrets are solid:** a Garen standing in his fountain gets
  shoved about 46 u by the fountain turret's 88-u body (env4 t=7.17 s). (b):
  turrets are collision obstacles (`collision.py` masks).

### 2. Overlap

Pairs closer than min(trigger sum, resolution sum) - 1 at the post-collision
position were counted, with ghosted units excluded. 90% of runs clear within
about 11 ticks.

- **Deep persistent overlaps** (>= 5 u deep, more than 3 ticks): 385 / 148
  / 490 / 270.
- **Worst cases:** 158-230 ticks (2.6-3.8 s) at 5-9 u depth. Examples:
  - training/trained env3 t=345.16 s: red melee slot3 and red caster slot26.
  - training/random env0 t=593.59 s: 230 ticks.
- **Cause:** always three or more minions wedged in a wave clump. One
  Gauss-Seidel sweep cannot satisfy every pair, so a pair stays 66-68 u apart
  against a resolution distance of 72.5, each pushed about 5-10 u per tick.
  (a) `COLL-002` ("dense packs can retain overlap").
- **On screen:** minions in a brawl visibly interpenetrate by about 10% of
  their width and jostle continuously. It is not an error the client should
  flag.
- **First occurrence:** training env0 t=92.62 s, red melees slots 2 and 4
  next to the red outer turret. This is exactly `COLL-005`'s documented
  deferred-repair case, and it is absent from the scripted runs (their first
  deep overlap is at t=125.99 s).

### 3. Terrain

- **Champion centre in an unwalkable cell:** 5404 / 4032 / 1057 / 1187
  events.
  - Every one lasts **exactly 1 tick**. The next tick's collision phase
    spirals the unit out by 38-105 u.
  - 4682 of the 5404 (training/trained) are Garen, followed by a pop of more
    than 20 u.
  - Minions: about 720-780 per condition, all at wave clumps against lane
    walls and turret bases (the `COLL-005` pattern).
- **Why Garen walks into walls:** 76% of the Garen pops happen while he
  holds `ATTACK_TO`.
  - The attack chase is a straight line to the target (`PATH-009`, (a)). A
    target across a wall corner makes him walk into it every tick.
  - The other 24% are `MOVE_TO` routes whose status is `SERVER_NULL`: the
    server's own `GetPath` null, walked as a raw line (`PATH-008`, (b)). This
    mostly happens while wedged at the fountain edge or a lane wall.
  - Of all trained Move orders, 22% are `SERVER_NULL` (37% for random).
- **On screen:** Garen "grinds" along a wall, stepping 5 ticks into it and
  snapping 40-50 u back out, about once a second in trained games. The first
  case is training/trained env1 t=0.85 s at (-120, 405): walking west off the
  fountain platform toward a click at (-1829, -69).
  - The server would route the chase through A* and not grind. **This is the
    single most visible artifact in the sweep.**
- **Corpses:** see (c)3. In training config corpses are popped out of
  terrain (713 events). In scripted config they are not, and sit inside
  terrain (935 dead ticks).
- **Units off the map:** 0.

### 4. Dead units

- **Dead minions:** 0 position changes, 0 targets, 0 `is_attacking`, 0 hp
  changes. The slot is kept with `alive=False` and frozen (the port must hide
  it).
- **Minions and turrets aiming at a corpse:** they hold a dead unit as target
  for exactly 1 tick (2,900 minion and 195 turret occurrences per condition;
  max 2 ticks). (a) `TGT-DEATHTICK` (implemented, measured, reverted).
- **Dead champions walk their route:** 7,600 corpse ticks
  (training/trained). (b) Expected: `ENT-12`.
  - They keep Q-haste speed (7.76 u/tick) if they died hasted. (b) Buffs
    survive death (`SPELL-006`).
- **Corpses dealing damage:**
  - Missiles from a dead shooter keep flying and land (170-214 per
    condition). (b) The server does not destroy a missile when its caster
    dies.
  - E corpse damage (`SPELL-006`) was not separately isolated.
- **Respawns** are at the fountain at full hp after `death_times[level]`
  seconds, **one tick early** (749 ticks for a 12.5 s timer). The timer is
  decremented on the death tick itself (`step.py` phase 26). The server's
  equivalent depends on update order (`ORDER-003`). Imperceptible.

### 5. Stuck

- **Garen:** never stuck with a route.
  - 625 cases of "moving but no net progress for 1 s" in trained games. All
    of them are the policy re-clicking back and forth (for example
    training/trained env0 t=15.8 s: alternating clicks toward base and
    toward lane).
  - That is the policy, not the sim.
- **Minions walking in place:** 112 / 87 / 183 / 107 cases lasting more than
  1 s, worst 9-10 s (training/trained env2 t=203.7 s, blue melee slot6 at
  (1199, 11200)).
  - The minion is on `ATTACK_TO` or its lane `MOVE_TO` with 300-800 u of
    route left, and allied minions stand 66-77 u ahead of it.
  - Every tick it walks 5.42 u into them and the collision phase pushes it
    back exactly 5.42 u.
  - **On screen:** a minion playing its walk animation while standing
    perfectly still behind its own wave ("moonwalking") for up to 10 s.
  - Also counted: 18 / 7 / 2 / 1 "idle, no target" cases (worst 10.1 s).
    These are the same treadmill on the lane route.
  - Classification: the mechanism is `COLL-001` (units are not A*
    obstacles, head-on escape) plus the straight chase (`PATH-009`). It has
    **not been checked against the server**. Likely (b) for the lane-route
    case; the chase case is at least partly `PATH-009` (a).

### 6. HP

- **Unexplained hp increase:** 0. Every increase coincided with one of:
  - a 500 ms regen tick,
  - Garen's passive heal,
  - a fountain pulse,
  - a level-up,
  - a respawn.
- **Hp decrease with no melee hit, missile arrival, E tick or R in the same
  tick:** 0.
- **Hp above max:** 0. **Alive with hp <= 0:** 0 (one minion sat at 7.6e-5 hp,
  alive, for a tick; float residue).
- **Max hp change without a level-up:** 0.
- The analyzer's "melee hit with no hp drop" and "missile arrived with no hp
  drop" hits (at most 6 per condition) were traced one by one. All are
  artifacts:
  - the attacker died on its hit tick,
  - a ranged attacker,
  - a level-up that raised hp in the same tick,
  - a victim at 7.6e-5 hp.

### 7. Rates

- **Attack interval**, completed swings only. The mode dominates (>95%):

  | Unit | Mode (ticks) | Period (s) | Period (ticks) |
  |---|---|---|---|
  | Melee minion | 49 | 0.8 | 48 |
  | Caster minion | 90 | 1.493 | 89.6 |
  | Cannon minion | 61 | 1.0 | 60 |
  | Outer turret | 73 | 1.2 | 72 |
  | Garen, level 1 | 97 | 1.6 | 96 |

  Garen's interval shrinks with level through the per-level attack-speed
  growth.
  - Every unit swings **one tick slower than its period**. (a)/(b)
    `AA-001`: the gate reads the cooldown before its decrement. The ledger
    records 97 ticks for 1.6 s.
  - Melee minions attack at 0.8167 s, which reads 2% slow to a stopwatch.
- **Garen intervals of 22-60 ticks** (39-56 per trained condition) all start
  under Garen Q. `cast_q` zeroes `aa_cooldown` (`orders.py:351`): Q is an
  auto-attack reset. (b)
- **Swings started and then aborted:** 8,172 / 8,279 / 2,607 / 3,549, with a
  median of 6 ticks between the aborted start and the restart. The target
  leaves `range + radius` mid-windup and the cooldown resets
  (`CancelAutoAttack(reset=true)`, `PORT_AUDIT_AI` 10.4). (b)
  - **On screen:** attack-animation stutter, most visible on casters at the
    edge of turret range.
- **Garen walks while winding up:** 12-13k ticks per trained condition,
  usually followed by the swing cancelling out of range. (b): Garen's basic
  attack has `CantCancelWhileWindingUp = 0` (`SPELL-002`), so the server's
  `ObjAIBase.CanMove` allows it (`ObjAIBase.cs:309`). **On screen:** Garen
  slides while in his attack pose.
- **Turret cadence:** 73 ticks (1.217 s) between shots while firing
  continuously.
- **E:**
  - Tick spacing is always 31 ticks (516.7 ms), 1,280-1,425 samples per
    condition.
  - Uncancelled spins deal 6 ticks, and a full spin lasts 180 ticks
    (3.00 s). Cancelled spins deal 2-5.
  - (b) `SPELL-003`: the server's ms accumulator drifts one frame per fire,
    at 0.017 / 0.533 / 1.050 / 1.567 / 2.083 / 2.600 s.
- **Waves:**
  - The first minion spawns at t=90.001 s. Minions within a wave come 800 ms
    apart.
  - Waves are 6 minions, and every third is 7 (the cannon after the three
    melees).
  - **The wave period is 36.4 s, not 30 s.** (b): `LevelScript` pushes
    `NextSpawnTime` only when the wave closes at minion 8, so the period is
    30 + 8 x 0.8 s (`waves_jax.py` comment).
  - **On screen:** waves arrive noticeably later than live League; by 10
    minutes that is 15 waves instead of 18.
- **Walk speeds:** minions 325 u/s exactly. Garen 345 (patch content, not
  live 340); hasted 465.75.

### 8. Targeting

- **Minions or turrets targeting an ally or an invisible unit:** 0. **A
  corpse:** only the 1-tick `TGT-DEATHTICK` case above.
- **Turrets holding a target beyond 750:** 0. **Any swing started out of
  `range + target radius`:** 0.
- **Champions swinging at an ally:** 0 (`ENT-01` holds).
  - Champions *hold* an allied target on 44-47% of all ticks: the trained
    policy's most frequent order is ATTACK on an allied turret, which is a
    held no-op (`ENT-01`, (b)).
  - **On screen:** nothing, unless the port draws target reticles.
- **Missiles:**
  - Missiles that vanish mid-flight with the target slot alive: 8 / 3 / 0 /
    0. These are `SLOT-001`'s intended drop of a missile aimed at a
    recycled slot.
  - Missiles in flight for more than 3 s: 16 / 7 / 2 / 11. **These are
    (c)1.**
  - Missiles whose target died: dropped, 885-1,041 per condition.
- **Friendly fire:** see (c)1.

### 9. Economy

- **Gold:** every change is explained by ambient gold (0.95 per 500 ms from
  90 s), a champion's killing blow on a minion (+gold, +1 CS), champion kills
  or turret kills.
  - The analyzer flagged 3 "unexplained" per trained condition. All three
    are a Garen hit on a minion in the same tick as a turret missile that
    crossed zero first, so the turret got the kill (correct attribution
    order).
- **CS without a last hit:** 0.
- **XP:** every change equals the XP of enemy minions that died within 1600
  u. No residual in any condition.
- **Level thresholds:** `level == level_for_xp(xp)` on every tick.

### 10. Other

- **Caps:** at most 30 of the 40 minion slots alive, so no spawn was dropped.
  At most 11 of the 24 missile slots in use.
- **Turret deaths:** 3 (training/trained) and 1 (scripted/trained) outer
  turrets died.
- **Minion jitter:** 1.0-1.4% of moving minion ticks reverse direction
  relative to the previous tick. This is the clump jostling of §2 and the
  §5 treadmill.
- **Recording artefact to know about:** GPU reruns are not bit-reproducible.
  The scripted/trained rerun diverged, by up to a whole recycled slot,
  within a 600-decision chunk. Repro states were therefore captured by
  **searching for the event class** in a rerun (`repro2.py`), then replayed
  deterministically on CPU. See `RL-007` for pinned autotune.

## Suspected bugs (c)

### (c)1 - a swing on a dead minion carries into the recycled slot: allied minions shoot their own new minion across the map

**On screen:** a caster or cannon minion in lane fires a projectile that
flies **backwards up the whole lane**, 8-11k units over about 10-13 s, and
hits a freshly spawned minion of its **own** team near its own barracks for
23 (caster) or 40 (cannon) damage. In one scripted/random game three red
casters did it together to the same new red caster.

**Counts:**

| | training/trained | scripted/trained | training/random | scripted/random |
|---|---|---|---|---|
| Recycles under a held swing | 16 | 7 | 8 | 13 |
| Friendly hits landed | 5 | 4 | 0 | 4 distinct (x3 identical envs) |
| Missiles in flight > 3 s | 16 | 7 | 2 | 11 |

The first occurrence is training/trained env2 t=130.404 s: red caster slot8,
`aa_target=3`.

**Mechanism:**

1. A ranged minion starts a swing on unit v. `aa_target = v` is fixed for the
   windup (`ENT-02`).
2. v dies to someone else. The swing keeps winding up: its `target` has
   already switched to another unit that is still in range, so the swing is
   not cancelled.
3. The next wave spawn reuses v's slot, via `spawn_minion`'s lowest free
   slot.
4. `spawn_minion` (`init.py:672`) drops *missiles* aimed at the slot
   (`SLOT-001`). It does **not** touch other units' `aa_target` (or
   `target`) that point at the slot.
5. When the windup completes, `hit_target = where(state.aa_target >= 0,
   state.aa_target, target)` (`step.py:845`) resolves to the newborn ally.
   `step_missiles` launches at it. Nothing on this path checks team or
   identity: `may_engage`/`hostile` gate only swing *starts*.

On the server a swing whose target has died resolves harmlessly: the missile
toward a dead unit is destroyed. That is also what the sim does whenever the
slot is not recycled in the meantime.

A champion variant was also seen once. In scripted/trained env0 t=384.47 s,
red Garen's `target` slot was recycled into an allied caster at the red
barracks. It is kept (the `keep_champ` team clause, `step.py:710`) and held
as an ally (`ENT-01`), so Garen silently loses his enemy target.

**Minimal reproduction (deterministic on CPU):**

- `/tmp/playtest/data/repro_training_trained_recycle_1.pkl` holds env-batched
  `pre_order` and `orders` for decision 3911 (t=130,387 ms), with
  `event=('recycle', env 2, unit 8, victim 3, tick 1 of the decision)`.
- Run `PYTHONPATH=. .venv-jax/bin/python /tmp/playtest/replay.py
  /tmp/playtest/data/repro_training_trained_recycle_1.pkl 700 training`
  (then `/tmp/playtest/show_recycle.py`). It applies the recorded orders, then
  steps one tick at a time with no-op champion orders.
- Replay output:
  - t=130,387: unit 8 (red caster) is winding up, `aa_target=3`,
    `target=5`. Slot 3 is dead (a blue minion, `spawn_seq` 27).
  - t=130,404: slot 3 is alive again, `spawn_seq` 48, **team red**, at the
    red barracks (12451, 13218). Unit 8 still has `aa_target=3` and is
    winding up.
  - t=130,537: the swing completes, and a missile launches from
    (2550, 12826) at slot 3.
  - t=141,874: the missile lands, and the red caster in slot 3 drops
    290 -> 267 hp at (9854, 13262).

**Status 2026-09-24: FIXED** as `AA-007` (a swing whose declared target
died is cancelled, `reset=true`, the server's `CastCancelCheck`) and
`SLOT-002` (`spawn_minion` drops other units' references to the slot). The
repro above now shows no friendly launch and no friendly hit.

**Fix direction** (as written before the fix): in `spawn_minion`, also clear any
`aa_target == i` (cancel that swing: `is_attacking`, `aa_windup`,
`aa_target`) and any `target == i` for other units, alongside the existing
`missile_tx == i` drop. Or, matching the server, stop resolving a swing whose
`aa_target` is dead at completion.

### (c)2 - `missile_source` is not cleared on slot recycling (attribution aliasing, low)

**On screen:** nothing visible. The projectile keeps flying and lands
correctly. A port that labels projectiles by `missile_source` will show a
missile fired by a now-dead blue caster as fired by the **red** minion that
spawned into its slot.

**Example:** training/trained env3. Blue caster slot20 launches at t=347.36 s
(tick 20841) at red slot9. Slot 20 is later reused by a red caster, and the
missile lands at t=348.11 s with `missile_source=20`.

**Consequences:** `damage_ij` rows are indexed by source, so the new
occupant can be credited:
- the kill (`killer`),
- the champion hit-flag (`hit_flag_by`, `rewards.update_hit_flag`),
- call-for-help aggro against the victim's allies. The new occupant is on
  the victim's team, so those allies see an "ally attacking an ally".

**Location:** `init.py` `spawn_minion` clears `missile_tx == i` and not
`missile_source == i`. That is the same `SLOT-001` class, the other end of
the index.

**Status 2026-09-24: FIXED** as `SLOT-003`: missiles record the shooter's
`spawn_seq`; a landing after the slot was reused still deals its damage but
is credited to no live unit.

**Frequency:** 170-214 missiles per condition land after their shooter
died. How many of those had their source slot reused before landing was not
separately counted.

### (c)3 - deferred terrain repair moves corpses; inline repair doesn't (config disagreement, low)

**On screen:**
- **Training config:** a dead Garen that is still walking its route
  (`ENT-12`) into a wall visibly **grinds**: 5 ticks into the wall, then a
  38-68 u snap back, repeated for the whole death timer. That is 839 corpse
  moves above walking speed and 713 terrain pops in training/trained.
- **Scripted config:** the same corpse walks into the wall and stays
  *inside* terrain (935 dead ticks in unwalkable cells, 0 pops).

**Location:** `step.py:336`. `repair_collision_terrain_batch(...,
eligible=arange < TU_SLICE.start)` has no `alive` mask. The inline path
(`collision.py:389`) uses `terrain_affected = alive & ...`.

**Reference behaviour:** the server probably collides corpses too
(`ENT-13` SUSPECTED: `CollisionHandler` has no `IsDead` test). If so, the
training config is the closer of the two and the inline path is the one that
is off. Either way, the two named configs disagree on a behaviour that
`SimConfig.scripted`-based parity drivers can see.

**Reproduction (CPU):** `/tmp/playtest/data/repro_training_trained_corpse_1.pkl`
(decision 9300, env0 unit1, dead). Replay it for 4 ticks with `training` and
with `scripted` (`/tmp/playtest/show_corpse.py` prints both). From the same
state at t=310,015:
- training: the corpse's collision position jumps from x=1470.96 to 1428.73
  at t=310,032.
- scripted: it continues 1476.31 -> 1481.66 into the terrain.

### (c)4 - `AA-006`'s row claims more masking than the code does (cosmetic, doc)

`AA-006` says `is_attacking`, `aa_windup` and `has_auto_attacked` are masked
by the death mask. `step.py:1452-1459` masks only `is_attacking`. A unit that
dies on the tick it starts, or finishes, a swing keeps `aa_windup` (e.g.
0.461, 3e-5) and a counting-down `aa_cooldown` on its corpse forever. That is
19-21 per condition as "swing start without target" on dead attackers.

Nothing reads a dead unit's windup, so there is no gameplay effect. A port
that draws attack state from `aa_windup` would show a corpse "mid-attack".
Either mask it or correct the row.

**Status 2026-09-24: FIXED.** The death tick now applies the whole
`CancelAutoAttack(true, true)` (no wind-up, cooldown 0 when a target was
held, `has_auto_attacked` untouched), and the `AA-006` row is corrected.

## Classification summary

| Finding | Class | Reference |
|---|---|---|
| Friendly cross-map missiles from a stale `aa_target` | **(c)1** | `step.py:845`, `init.py:672` |
| `missile_source` alias after recycle | **(c)2** | `init.py` `spawn_minion` |
| Corpse terrain repair differs by config | **(c)3** | `step.py:336` vs `collision.py:389`; `ENT-13` |
| Dead unit keeps `aa_windup` | **(c)4** | `step.py:1452`; `AA-006` text |
| Garen grinding walls on attack-chase | (a) | `PATH-009` |
| Walking raw lines into terrain on `SERVER_NULL` moves | (b) | `PATH-008` |
| Minion clumps overlap 5-10 u for seconds; 20-100 u pushes | (a)/(b) | `COLL-002`, `COLL-001` |
| First deep overlap at the red turret at 92.6 s (training only) | (a) | `COLL-005` |
| Minions "moonwalking" behind their own wave | likely (b), unverified | `COLL-001`, `PATH-009` |
| Swings one tick slower than period | (a)/(b) | `AA-001` |
| Wave period 36.4 s | (b) | `waves_jax.py` |
| E ticks 516.7 ms apart, 6 per spin | (b) | `SPELL-003` |
| Q shortens the swing gap | (b) | Q auto-attack reset |
| Garen slides during windup; swing stutter at range edge | (b) | `SPELL-002`, `ObjAIBase.CanMove` |
| Dead champion walks its route, keeps Q haste | (b) | `ENT-12`, `SPELL-006` |
| Minion/turret target held on a corpse for 1 tick | (a) | `TGT-DEATHTICK` |
| Champion holds allied targets 44-47% of ticks, never swings | (b) | `ENT-01` |
| Respawn one tick early | order-dependent | `ORDER-003` |
