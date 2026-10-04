# MECHANICS_AUDIT.md — small moment-to-moment mechanics, 26.19 League vs the modern JAX world

**Status (2026-10-04).** Report only; no code was changed. This audit covers the "small mechanics" that shape
lane play: orders, movement, attack timing, casting, minion and turret AI, collision, tick/latency, vision and
lifecycle timers. Each one is checked against `lanerl_jax/sim/modern_step.py` and its modules at HEAD `d2d35be`.
It also lists what the 26.9 replay corpus can check.

**Fixed in MODERN-023** (tests in `test_modern_world_rules.py`, `test_modern_step_helpers.py`,
`test_modern_mechanics.py`):
- **#1 Collision:** pathing radii and steering avoidance, in `modern_collision.py` (see [COLLISION.md](COLLISION.md)).
- **#2 Move orders:** they end on arrival or when the champion can make no progress (`_move`).
- **#3 Latency:** `--action-delay-ticks` in `modern_vec_train`. The default is 0; picking a value is a user decision.
- **#4 and #9:** out-of-range unit-targeted casts walk into range, and casts made during a lockout or within 0.5 s
  of a cooldown ending are buffered (`QueuedCast`, `_queue_casts`; ranges in each kit's `UNIT_TARGET_RANGE`).
- **#5 Attack windup:** a one-tick grace before launch, in `modern_mechanics.attack_step`.
- **#6 Minion Pushing:** wired up (`_minion_pushing`; recomputed every tick rather than held for 1 s).
- **#7 Stop:** a `stop` button was added (action profile `modern-world-v2`).
- **#8 Clicks:** clicks hit-test the client selection radii (`modern_actions.selection_radius`).
- **#10 Targets lost to fog:** the champion walks to the target's last-seen position (`ChampionLayer.target_seen_at`).
  Unlike League, the attack does not resume if the target reappears.

The rest of this document is the original report.

## Sources and confidence

- **CLIENT:** cdragon 16.19 character bins in `/mnt/nfs/shared/modern-world-map-research/cdragon-16.19/`
  (`garen.bin.json`, `jax.bin.json`, `sru_*minion*.bin.json`, `turret.bin.json`). These give the pathing,
  gameplay and selection radii quoted below.
- **WIKI:** League wiki pages, from the local snapshots (`wiki-damage-stats/`, `cdragon-16.19/wiki-2026-10-01/`,
  `econ-wiki/`) and from wiki.leagueoflegends.com:
  - Basic attack: windup, cancel, grace tick, attack-move, idle acquisition;
  - Tick and updates;
  - Unit size / Pathing radius;
  - Ability: cast time, out-of-range casts, targeting forgiveness;
  - Minion and Turret pages.
- **RIOT:** patch notes 26.1–26.19 (`patch-notes-26.x/`). Relevant here: the 26.10 minion-aggro change and the
  26.1 plates/first-wave changes.
- **REPLAY:** measurements on the 3-game smoketest subset `/mnt/nfs/datasets/lol_replays_16_9_772_smoketest/`
  (patch 26.9, `raw_mem.json` with all 10 heroes, `labels.json`, `clicks.json`), §M.
- **Confidence tags:**
  - **H**: client data or an explicit wiki/Riot statement, or a replay measurement.
  - **M**: a consistent community or wiki statement without a client value.
  - **L**: inferred, or nothing found.
- **Not found in any source:** path re-plan frequency, attack-move cursor radius, minion leash distance, minion
  attack-timing jitter, fog update interval.

## Prioritized table

Rank = gameplay impact on 1v1 lane / RL learning × confidence that the sim is wrong. Status words:
**wrong** = contradicts evidence; **missing** = not modelled; **approx** = modelled with a guessed or simplified
rule; **ok** = matches.

| # | Item | Sim location | Status | Impact | Conf. sim wrong | Replay-checkable |
|---|---|---|---|---|---|---|
| 1 | Collision uses gameplay radius (65/48), not pathing radius (35/35.7); no path-around-units | `modern_step.py:1057`, radii `:248` | **wrong** | High | H (client bins) | partly (§M3) |
| 2 | A move order never completes: `moving` stays True at the goal. Idle auto-acquire is dead after any move, and a pushed champion walks back to a stale goal | `modern_step.py:775`, `:945`, `:988` | **wrong** | High | H (code) | yes (§M6) |
| 3 | No action/observation latency: the agent reacts in 0–100 ms on exact state | `modern_vec_train.py:16`, `modern_actions.py:83` | **missing** | High (RL) | H | yes (§M5) |
| 4 | Out-of-range unit-targeted casts fail silently instead of walking into range and casting | `modern_champions/jax.py:166-168`, `garen.py:159` | **wrong** | Med-High | H (wiki) | partly |
| 5 | No windup grace tick (last tick before launch is uncancellable) and no 1-tick post-launch lockout | `modern_mechanics.py:61-62` | **missing** | Med | H (wiki) | yes (§M4) |
| 6 | Minion Pushing buff (level/turret advantage) not applied | `modern_step.py:1126` (no `pushing_*` args); helper `modern_minions.py:383` | **missing** | Med | H | no |
| 7 | Policy has no Stop/Hold button; with #2 fixed, it cannot stand near a wave without auto-attacking | `lanerl_rl/constants.py:596`, `modern_actions.py:135` | **missing** | Med | H | yes (S key not logged; idle-near-wave proxy) |
| 8 | Click hit-test uses gameplay radius (65/48), not selection radius (120/115/140) | `modern_actions.py:103` | **wrong** | Med | H (client bins) | no |
| 9 | Cast/ability input buffering: a cast during a cast lockout is dropped | `modern_step.py:858-861` | **missing** | Med-Low | M | no |
| 10 | Target lost to fog: the attack order is dropped and the champion goes idle; League walks to the last-seen position | `modern_step.py:764,772` | **wrong** | Med-Low | M (wiki) | partly |
| 11 | Minion death grace (HP held at 1 for 0.035 s) not applied | none (MINIONS §4.7) | **missing** | Low-Med (last-hit windows) | H | no |
| 12 | Minion→champion damage ratio: client 0.55 vs wiki "60%" | DMG.45 (README X-2) | conflict | Low-Med | L | yes, from HP drops (§M7) |
| 13 | Call-for-Help memory 2.0 s and minion re-evaluation cadence 0.25 s / 4 s give-up / 0.5 s ignore are INFERRED/legacy | `modern_lane_ai.py:64-67` | approx | Med | M | no (no minion positions) |
| 14 | Route steering: at most 16 re-plans per tick, so excess units freeze for a tick; a blocked step leaves the unit in place (no wall slide) | `modern_mechanics.py:229,277` | approx | Low-Med | M | partly (§M3) |
| 15 | Turret shot is not lost when the turret dies mid-flight | `modern_mechanics.py:142` (only target death fizzles) | **wrong** | Low | H (wiki) | no |
| 16 | Unit-targeted cast range: strict, with no targeting forgiveness (175) | `jax.py:168`, `garen.py:159` | approx | Low | M | no |
| 17 | Cost/cooldown at the end of the cast time (wiki), not at cast start | kits `cast` (Garen R) | approx | Low | M | no |
| 18 | Super-minion aura (+35 armor/MR) not wired | none | **missing** | Low (top 1v1 early) | H | no |
| 19 | Minion attack-timing jitter | none | unknown | Low | L (no source) | no |
| 20 | Fog recomputed every tick (1-tick lag) | `modern_step.py:1542` | unknown | Low | L (rate undocumented) | no |
| 21 | Instant acceleration / deceleration | `modern_mechanics.py:271-273` | **ok** | — | — | measured (§M1) |
| 22 | Instant turning; facing = last displacement | `modern_step.py:1058-1060` | **ok** | — | — | measured (§M2) |
| 23 | Attack machine: windup → launch → period, cancel resets timer to 0, per-unit timer survives target switch, attack reset, edge-to-edge range, homing missiles | `modern_mechanics.py:44-87` | **ok** | — | — | measured (§M4) |
| 24 | Chase-then-attack (move into range, start windup the same tick) | `modern_step.py:986-988`, ATTACK after MOVE | **ok** | — | — | measured (§M4) |
| 25 | Server tick 30 Hz, tick-rounded timers | `modern_world.py:87` `dt=1/30` | **ok** | — | — | — |
| 26 | Minion priority list post-26.10, CFH 500/1000, strict-priority switch, first wave 0:30 + 28 s ghost, spawn 0.8 s, MS 350 + steps | `modern_lane_ai.py:256-397`, `modern_minions.py:99-120` | **ok** (some INFERRED parts, row 13) | — | — | no |
| 27 | Turret sticky lock, champion protection 1400 (attempts count), Warming Up, plates 26.1 | `modern_lane_ai.py:399-423` | **ok** | — | — | no |
| 28 | HP regen 0.5 s pulses, ambient gold, death timers, respawn, Recall 0.5 + 8 s | `modern_step.py:1452-1455`, economy | **ok** (REPLAY_FIDELITY) | — | — | done |
| 29 | Decision rate 10 Hz vs human click cadence | `modern_vec_train.py:471` | **ok** | — | — | measured (§M5) |

## Per-item sections

### A. Movement and orders

**A1. Acceleration (ok).**
- *League:* no acceleration or deceleration; units move at full MS from the first tick. REPLAY H: in §M1, speed
  goes 0→max and max→0 within one 25 ms position step (n = 1,331 starts, 1,627 stops).
- *Sim:* full `speed·dt` on every active tick (`modern_mechanics.py:271-273`). Matches.

**A2. Turning and facing (ok).**
- *League:* champions in practice have no visible turn time. REPLAY H: heading changes complete within one
  20 Hz frame (p50 0.05 s, n = 5,242 turns), and the median click→direction-change delay is 0 s.
- *Sim:* facing is the last displacement (`modern_step.py:1058-1060`). Matches. Abilities that depend on facing
  would need an explicit facing rule; Garen and Jax have none.

**A3. A move order never completes (wrong, High).**
- *League:* when a path ends, the champion is idle, and an idle champion auto-acquires enemies within its
  acquisition range (400 for Garen/Jax, edge distance) [WIKI Basic attack H]. Collision displacement does not
  re-issue the old path.
- *Sim:* `moving` is set by a move order (`modern_step.py:775`). It is cleared only by stop, attack, attack-move,
  respawn, recall or death (`:775`, `:1513`), never on arrival. So:
  - `idle` (`:945`) requires `~moving`, which means idle auto-acquisition never fires after the first move order.
    The trainer's reset bank issues exactly such an order (`modern_vec_train.py:287`), so every episode starts
    with auto-attack disabled.
  - A champion pushed off its goal walks back to it (`:988`) indefinitely. This "anchoring" makes it harder to
    get displaced and holds positions that League would not hold.
- *Impact:* High. It changes passive minion damage, freeze/wave-management behaviour (League players fight
  idle autos with Stop/Hold), and how the policy must spend actions to attack at all.
- *Fix:* clear `moving` when within ~1 tick of travel of `move_goal` (or when the route reports arrival). Then
  add a Stop/Hold button (A6).

**A4. Click-move, path smoothing, unreachable points (approx, Low).**
- *Sim:* a direct line when the segment is clear, otherwise route-graph nodes with a cached anchor
  (`modern_pathing.py:90-107`). Non-walkable goals project to a graph node (`:76-87`, "no client parity claim").
- *League:* the client paths to the nearest reachable point. The re-plan frequency is undocumented [L].
- Lane terrain is mostly open, so the effect is small. The two artefacts that matter are in row 14:
  - at most 16 full searches per tick, so the remaining units hold position for a tick
    (`modern_mechanics.py:229,260-266`);
  - fail-closed terrain clamp: a blocked step leaves the unit in place instead of sliding along the wall (`:277`).

**A5. Order spam and repath (ok).** League has no cost to re-clicking, and neither does the sim. Replay click
cadence is about 3 distinct path-target changes per second (§M5).

**A6. Stop / Hold (missing, Med).**
- *League:* S clears the move/attack orders and the current auto-acquire target. H (hold) prevents chasing
  [WIKI H].
- *Sim:* `ModernOrders.stop` exists (`modern_step.py:176`), but the policy's button set (`noop, move,
  attack_move, q, w, e, r, recall` + modern extras; `lanerl_rl/constants.py:596`, `modern_actions.py:53`) has
  no stop. Hold does not exist.
- This matters once A3 is fixed. Without a stop button, an idle agent near a wave auto-attacks minions it might
  want to leave alone (freezing).

**A7. Attack-move (approx, Low).**
- *Sim:* nearest valid hostile to the champion within the acquisition range, edge distance; ends within 10 u of
  the point (`modern_lane_ai.py:1012-1056`).
- *League:* matches the wiki for the default setting. The cursor radius for "attack move on cursor" and the
  10 u arrival are INFERRED L. "Target champions only" is not exposed.

**A8. Chase to attack (ok).**
- *League:* walks until in range, then starts the windup immediately [WIKI H]. REPLAY: 82% of attack starts come
  straight out of movement (§M4).
- *Sim:* MOVE (`modern_step.py:986-988`) and then ATTACK in the same tick, on post-move positions. Matches.

**A9. Target enters fog (wrong, Med-Low).**
- *League:* a champion chasing a target that loses sight walks to the target's last-seen position, and the order
  resumes if the target reappears [WIKI M].
- *Sim:* the attack order is dropped on fog (`modern_step.py:764,772`). The champion becomes idle and stands
  still (or, given A3, keeps its stale move goal).
- *Impact:* brush juking in top lane, and chasing into the bushes.

### B. Attack timing

**B1. Windup, launch, period, cancel, reset, timer persistence (ok).**
- Implemented in `modern_mechanics.attack_step` (`:44-87`):
  - any cancel before launch resets the timer to 0;
  - the cooldown is per unit, so a target switch does not reset it;
  - an attack reset cancels a windup in progress;
  - edge-to-edge range (`:33`);
  - launched ranged attacks home on the target (`:136-149`).
- Windup follows DAMAGE §8.2, including Garen's modifier of 0.5.
- REPLAY: windup segment p50 0.325 s; Garen period p50 1.03 s at real attack speed (§M4). These are consistent.

**B2. Grace tick and post-launch lockout (missing, Med).**
- *League:* in the last server tick before launch, player commands cannot cancel the windup, and after launch
  there is a one-tick input lockout [WIKI Basic attack, H per research; DAMAGE §8.2].
- *Sim:* `cancel = winding & ~(same & ready)` (`modern_mechanics.py:62`). A move order on the launch tick cancels
  the attack.
- *Impact:*
  - Orb-walking at 10 Hz decisions is less forgiving in the sim than in League. A move issued one tick early
    loses the whole attack.
  - The agent learns a stricter timing than the real game requires.
- *Fix:* treat `windup_left <= dt` as uncancellable by orders (CC and death still cancel). Then hold
  move/attack orders for one tick after launch.

**B3. Attack-cancel leash (approx, Low).**
- *Sim:* a target leaving range during the windup cancels it (`:58-62`).
- *League:* the exact leash distance is unknown (DAMAGE U-08). Low impact for melee vs melee.

**B4. Missile lifetime (wrong, Low).**
- *League:* a turret shot is lost if the turret dies mid-flight [WIKI H].
- *Sim:* `advance_missiles` fizzles only on target death or slot reuse (`modern_mechanics.py:142`).
- *Fix:* also check `units.alive[src]` for turret sources.

**B5. Attack speed changing mid-swing (approx, Low).** The semantics of `gcd_AttackSpeedCatchupPercent 0.25`
are unknown (DAMAGE U-15). The sim fixes the period at swing start.

### C. Ability casting

**C1. Chase to cast (wrong, Med-High).**
- *League:* a unit-targeted spell cast out of range makes the champion walk into range and then cast; the cast is
  queued like an attack order [WIKI Ability H].
- *Sim:*
  - Jax Q requires `dist <= 700 + r_t` at the order tick (`modern_champions/jax.py:166-168`);
  - Garen R requires `<= R_RANGE + r_t` (`garen.py:159`);
  - otherwise the order is dropped with no movement.
- *Impact:* Garen R executes and Jax Q engages are the lane's kill tools. The policy must hand-time range, and a
  failed cast wastes a 100 ms decision.
- *Fix:* add `pending_cast` (slot, target, expiry) to `ChampionLayer`. While it is set, chase the target as
  `attack_order` does, and cast when in range.

**C2. Cast buffering (missing, Med-Low).**
- *League:* inputs during a cast time or lockout are queued and execute when it ends [M].
- *Sim:* `can_cast` masks the order to -1 during `cast_lock_until` and item casts (`modern_step.py:858-861`). The
  order is lost.
- *Fix:* the same `pending_cast` latch with a short expiry (~0.3–0.5 s, INFERRED).

**C3. Cast time rules (approx, Low).**
- *League:* cast times round up to whole ticks (0.25 → 0.264 s). During a cast you cannot move, attack or cast
  (Flash is allowed), and only death interrupts. Cost and cooldown usually apply at the end of the cast [WIKI H].
- *Sim:*
  - Garen R uses `R_CAST_TIME 0.435` with a lockout (`garen.py:78,197`).
  - Movement/attack lockout goes through `cast_lock_until` (`modern_step.py:1500`).
  - The cooldown starts at cast (CHAMPIONS.md).
- The cast-end versus cast-start difference only matters for interrupt-by-death edge cases.

**C4. Range metric and forgiveness (approx, Low).**
- *League:* unit-targeted range is mostly centre-to-centre (some spells edge), plus a targeting forgiveness of
  175 [WIKI M].
- *Sim:* centre-to-edge with no forgiveness.

### D. Minion behaviour

**D1. Priority, Call for Help, hysteresis (ok, with INFERRED timers).**
- Matches 26.10:
  - champion hits on minions do not aggro (`modern_lane_ai.py:295-297`);
  - CFH 500/1000 (`modern_minions.py:119-120`);
  - strict-priority switching, not mid-windup (`modern_lane_ai.py:346-354`).
- Still unsourced (row 13):
  - aggro memory of 2.0 s (`:67`);
  - re-evaluation every 0.25 s, 4 s give-up and 0.5 s ignore (`:64-66`, legacy 4.20 values).
- These timers set how long a trade costs you minion aggro, which matters for trading patterns. MINIONS U-6
  gives the test.

**D2. Spawn, schedule, speed, first wave (ok).**
- Implemented: first wave 0:30, 0.8 s spacing (client; wiki 0.79), base MS 350 plus the 5-minute steps and the
  side-lane buff, first-wave 28 s ghosting and spread (`modern_minions.py:99-113`, `modern_lane_ai.py:369-391`,
  `:913-918`).
- The wave meeting point follows from these. It is not checkable from replays, which carry no minion positions.

**D3. Minion Pushing (missing, Med).**
- `attack_packets` accepts `pushing_bonus` and `pushing_divisor` (`modern_lane_ai.py:452,490-493`), and
  `modern_minions.minion_pushing_modifiers` exists (`:383`). But `_attack` calls `LA.attack_packets(units, …,
  now=now, ai=lane_ai)` without them (`modern_step.py:1126`).
- *Effect in League:* a level lead (≥ 3:30, client `mvm_StartTime 210`) makes your wave push, up to +15%
  minion-vs-minion damage in 1v1 with no turrets down [CLIENT H, MINIONS §4.4].
- *Impact:* pushing and freezing incentives after a level lead.

**D4. Death grace (missing, Low-Med).** MINIONS §4.7: a minion below 0.35% max HP that takes lethal minion
damage is held at 1 HP for 0.035 s, which is about one tick, so champions can steal last hits. It is not
implemented. The effect is a small widening of the last-hit window.

**D5. Damage ratio (conflict, Low-Med).**
- The sim applies the client `dr_UnitToHero` 0.55 at DMG.45 (README X-2). The research summary quotes the wiki
  as "60% to champions/structures".
- 0.60 is the structure ratio in the client. Keep 0.55 for champions unless measured (§M7).

**D6. Attack jitter, siege/caster missiles (ok/unknown).**
- Missile speeds are caster 650 and siege 1200 (`modern_minions.py:110`).
- No source describes minion attack-timing randomization [L]. Sim waves are fully deterministic.

**D7. Super aura (missing, Low).** +35 armor/MR to nearby minions (MINIONS §4.5) has no code. It only matters
after an inhibitor falls.

### E. Turrets

All of these are implemented per TOWERS §3–§5 (`modern_lane_ai.py:399-423`, `modern_towers.py`), and they agree
with the research: sticky lock, champion protection within 1400 that counts zero/blocked attempts, Warming Up
+50%/stack over 5 s, edge range `750 + 88.4 + r` (`modern_towers.py:156-158`) and the 26.1 plates.

Open items:
- The CFH/protection trigger reads last tick's `damage_matrix` (1-tick lag, `modern_step.py:903`). Ranged
  attempts therefore count at impact, not at launch [L].
- B4 (shot lost on turret death).
- TOWERS U1/U3/U5.

### F. Server tick, latency, decision rate

**F1. Tick (ok).** Live servers run 30 Hz, 0.033 s [WIKI Tick and updates H]. Sim `dt = 1/30`
(`modern_world.py:87`).

**F2. Latency (missing, High for RL).**
- *Real player:* human reaction is ~200–250 ms, plus ping (NA ~30–60 ms) and server tick quantization.
- *Sim agent:*
  - It observes the exact state at the decision tick, and its order applies on that same tick
    (`modern_vec_train.py:16`; no delay anywhere in `obs/modern_builder.py` or `train/modern_actions.py`).
  - Its effective reaction time is 0–100 ms, it sees exact HP, and it has no human-style misclick or input noise.
- *Consequences:*
  - Last hits, Jax E dodges and turret-aggro timing become superhuman.
  - Policies learned this way may rely on timings that are infeasible under real latency.
- *Fix:* a FIFO action delay of 2–4 ticks (configurable). Optionally, observe state from k ticks ago.

**F3. Decision rate (ok).**
- REPLAY: about 3 distinct path-target changes per second, median inter-click gap ~260 ms, under 5% of gaps
  < 100 ms (§M5).
- 10 Hz (`modern_vec_train.py:471`, `--step-ticks 3`) is therefore enough for movement intent.
- What 10 Hz quantizes is attack timing. B2 makes that harsher than League.

### G. Collision and creep block (brief; owned by the collision redesign)

- *League:* unit-unit collision uses the **pathing radius**:
  - champions 35 [CLIENT `pathfindingCollisionRadius` 35.0 in `garen.bin.json`/`jax.bin.json`];
  - melee/caster minions 35.74, siege 55.74, super 55.52 [CLIENT];
  - non-champion units treat each other ~20% larger (wiki, patch 5.23).

  Gameplay radius (65/48/65) is for attack range only. Units steer and path around blockers.
- *Sim:* `resolve_collisions(…, s.radius, s.radius, …)` (`modern_step.py:1057`) uses **gameplay** radius
  (`:248`: 65 champion, 48 minion). Contacts are a legacy teleport-to-touching push with no avoidance. The
  champion–melee contact distance is therefore 113 u in the sim vs ~71 u in League, about 1.6× too wide.
  Lanes are more crowded, there is more creep block, and waves spread out more.
- *REPLAY (§M3):* real champions near units keep path-speed ratio ~1.0 and show minions as detours
  (straightness < 0.9 in 31% of lane-phase windows vs 14–20% later), not as slowdowns.

  That is the signature of path-around-units, not push-back. The collision redesign should reproduce:
  - the pathing radii;
  - steering around units;
  - ghosting (first wave, Ghost, dashes, Garen E; already ORed in at `:1051-1052`).

### H. Vision

Fog is recomputed every tick from final positions and read next tick (`modern_step.py:1542`). Riot does not
document the fog update interval [L], so treat this as acceptable. The attack reveal (300 u, 2 s) is
implemented (VISION.md).

### I. Lifecycle timers (ok, replay-checked)

These are all verified in REPLAY_FIDELITY.md and the economy oracle:
- HP regen in 0.5 s pulses (`modern_step.py:1452-1455`);
- ambient gold 2.04 g/s;
- death timers;
- respawn;
- Recall 0.5 s cast + 8 s channel (`modern_economy.py:45-46`);
- Homeguard;
- level-up HP;
- shop usable only in the shop area or while dead.

### J. Click hit-test (wrong, Med)

- *League:* the right-click target is resolved with the unit's **selection radius**:
  - champions 120;
  - melee/caster minions 115;
  - siege 140, super 145;
  - turret 130 [CLIENT `selectionRadius`].
- *Sim:* `pick_r = state.radius` (`modern_actions.py:103`), i.e. gameplay radius 65/48.
- *Effects:*
  - The policy must click 2–2.5× more precisely to target. On the 96×54 grid (~30×36 u cells) a minion is only
    ~3×3 cells.
  - A move click near a unit is less likely to turn into an accidental attack.
- *Fix:* use the selection radius per kind for the hit test, with nearest centre winning (already done).

## M. Measurable from our replays

The corpus has hero positions only (no minion positions). Data caveats:
- `labels.json` `movement.speed` is a 0.5 s look-ahead displacement, not an instantaneous speed;
- `waypoint` is always null;
- `raw_mem` positions are quantized to ~25 ms steps;
- `clicks.json` logs path-destination changes > 50 u, polled every 30 ms.

Smoketest runs: `ops/login_capped.sh 6G 2 .venv-jax/bin/python <script>`.

| ID | What | Recipe | Result (3 smoketest games) | Use |
|---|---|---|---|---|
| M1 | Acceleration | `raw_mem` per hero: speed = Δpos/Δt over consecutive distinct samples. Start = speed < 20 then > 0.8·nominal; count samples to reach 0.9·max | 0→max and max→0 within one step; n = 1,331 starts / 1,627 stops | A1 ok; regression test: sim step speed equals MS from tick 1 |
| M2 | Turning | heading of the displacement; events where the direction changes by > 90°; transition duration; also `clicks.json` time → first heading change | p50 0.05 s (one frame), click→turn median 0 s; n = 5,242 | A2 ok |
| M3 | Creep-block proxy | lane phase 1:05–3:00 vs later; 0.45 s windows; path-speed ratio = path length / (MS·dt); straightness = chord / path | ratio ~1.0; straightness < 0.9 in 31% (lane) vs 14–20% (later) | §G target: run the sim with the same scripted walks and compare both distributions, especially the "slowdown vs detour" split |
| M4 | Attack timing / orb-walk | `labels.action.type == attack` segments (recorded champion): segment length, start-to-start gap, movement > 50 u between autos, movement during the windup, fraction of starts preceded by movement | windup p50 0.325 s; 2–8% move during the windup; 75% of gaps include movement; 82% of starts come out of movement; Garen period p50 1.03 s | B1/B2/A8; after B2, compare the sim's cancel rate under a 10 Hz scripted orb-walk |
| M5 | Click cadence | `clicks.json` inter-event gaps | ~3/s, median 260 ms, < 5% under 100 ms | F2/F3: a delay of 2–4 ticks is compatible with human cadence |
| M6 | Idle auto-attack | recorded champion: periods with no click for ≥ 1 s while an enemy unit is within ~400 (enemy hero from `raw_mem`; minions not available) followed by `action == attack` with no new click | not run | A3: the fraction of attack starts with no preceding click. With A3 broken the sim gives 0 |
| M7 | Minion→champion damage | HP drops of the recorded champion in 0.4/0.8 s steps (minion periods) with no enemy hero nearby, vs the minion AD at that time | not run | D5: 0.55 vs 0.60 |
| M8 | Chase to cast | Garen R / Jax Q `labels.action.spell` events: distance to the nearest enemy hero at the click (`clicks.json`) vs at the cast | not run | C1: casts that start beyond the spell range after a walk-in |

## Unresolved (no source found)

- Path re-plan frequency.
- Attack-move cursor radius.
- Minion leash and give-up timers (MINIONS U-6).
- Minion attack jitter.
- Fog update interval.
- Cast-buffer window length.
- Attack-cancel leash distance (DAMAGE U-08).

All of these need a 26.19 custom-game recording at ≥ 30 fps. The tests in MINIONS §8 and TOWERS §12 apply.
