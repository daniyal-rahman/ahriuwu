# The JAX lane-sim rewrite — build plan

**Branch:** `lane-rl/jax` · **Written:** 2026-09-15 · **Scope:** the JAX rewrite only.

Other workstreams (replay harvesting, the imitation prior, human eval, the league
beyond self-play) are out of scope here and are tracked elsewhere. This document
covers one thing: replacing the C# `GameServer` in the *training* loop with a
JAX lane simulator, at parity with the server first and on a modern-League patch
table second.

---

## 0. The claim this plan is built to support

> A 1v1 Garen top lane, simulated in JAX, that reproduces the vendored server's
> mechanics closely enough that a policy trained in it plays the same way on the
> server — at enough throughput to make the experience budget stop being the
> binding constraint.

Everything below is a gate on some part of that sentence. Note the shape of the
claim: the JAX sim is **not** the product. The server stays the reference and the
evaluation venue for the whole life of the project. The JAX sim is a throughput
device, and the day it stops agreeing with the server it stops being useful.

---

## 1. What the research found

These are the facts the design rests on. Each is measured or cited to a line, not
assumed, because most of the plan's risk lives in whether they are true.

### 1.1 The current stack's throughput, and where it goes

From the last completed league run's own shutdown record
(`runs/rl-league-0915c/metrics.jsonl`, 2026-09-15):

| | |
|---|---|
| wall clock | 48,203 s (13.4 h) |
| updates | 16,800 |
| mean env steps/s | **94.1** |
| mean decisions/s | **1,129** |
| learner idle, waiting on actors | 10,554 s (22%) |
| learner compute | 6,777 s (14%) |

So a 13-hour run buys ~54M decisions. Code comments record 3,409 decisions/s at
the 2026-09-13 mirror baseline and ~1,600 for the league configuration
(`lanerl_train/procactor.py:259`, `lanerl/run_watch.py:129`), so call the current
stack **1–3.4k decisions/s**. For reference, the free-running simulator alone
manages 43× real time per instance and 688× aggregate over 16 instances
(`lanerl_train/vec.py:12-24`) — i.e. the C# server is *not* the bottleneck. The
Python `ObservationBuilder` is: it is ~55% of a decision and holds the GIL for
all of it, which is why actors had to become processes at all
(`lanerl_train/procactor.py:1-30`).

**Read:** the rewrite's throughput win comes at least as much from moving the
observation builder onto the accelerator as from moving the simulator. Plan for
both; do not treat the obs builder as an afterthought (see §4.2).

### 1.2 The server is deterministic — as of yesterday

Commit `7b6d640` ("determinism is now asserted, not tolerated", 2026-09-15
19:29): two servers, same seed, same action sequence, identical state-hash
streams, 5 runs for 5. The cause of the previous nondeterminism was a background
`Task.Run` applying Garen's passive off the game thread; it now runs from
`ICharScript.OnUpdate`.

This is the single most important fact in this document. A nondeterministic
reference cannot be differentially tested against — you can only compare
distributions, which is exactly the kind of test that let three simulation bugs
ship behind believable learning curves. A deterministic reference can be diffed.

### 1.3 The parity oracle already exists and is free

`GameServerLib/Lanerl/LanerlStateDump.cs` emits, **every server tick** — it is
called from `LanerlHooks.OnUpdate` (`LanerlHooks.cs:124`), after that tick's
actions have been applied. Measured: 7,202 snapshots over 120,010 ms of game
time, one per 16.67 ms tick, i.e. twice the decision rate:

```
LANERL_STATEHASH t=<gametime> n=<entities> h=<fnv1a64>
LANERL_STATEROW  t=<gametime> <kind>|<team>|<x>,<y>|<hp>/<maxhp>|<A|D>|...   # under LANERL_STATE_DUMP_FULL=1
```

Champion rows carry the full combat stat block, level, gold, CS, deaths, skill
points, per-slot spell level and cooldown, move order, waypoint count, sorted
buff *names*, and the cast/channel flags. Positions are quantised to 1/16 unit
and stats to 1/1024. Entities are sorted by **content**, not NetId, so the
ordering is stable across processes and across episode resets.

That is a complete, canonical, quantised dump of the whole simulation, emitted
at *finer* than the rate the JAX sim will step — which is the better granularity
to diff at, since a one-tick differential is the tightest available isolation of
a logic error from accumulated float drift. It was built to catch reset leakage;
it happens to be precisely the fixture a second implementation needs. **We do not
have to build a parity harness — we have to write a parser and a differ.**

### 1.4 The tick is fixed under training conditions

`Game.GameLoop` normally drives `Update(deltaTime)` from wall-clock sleep
duration (`Game.cs:331`) — which would make tick-accurate parity impossible.
But under `LANERL_FREERUN=1` the timestep is pinned to `REFRESH_RATE = 1000/60`
ms (`Game.cs:333`) and the loop never waits (`Game.cs:392`). Training always sets
it (`lanerl_train/vec.py:175`), as do `collect_demos`, `eval_vs_bot` and
`control_smoke`.

So the reference timestep is **exactly 1000/60 ms**, and `LANERL_STEP_TICKS=2`
gives the 30 Hz decision rate (`lanerl_rl/constants.py:85-93`). The JAX sim
scans 2 fixed 16.667 ms ticks per env step. No variable-dt modelling required.

### 1.5 Terrain is exactly extractable — and pathfinding is a table lookup

This was the risk I most expected to be a blocker, and it is not.

I parsed `Content/LeagueSandbox-Default/AIMesh/Map1/AIPath.aimesh_ngrid`
(format per `Content/Navigation/NavigationGrid.cs:95-170` and
`NavigationGridCell.ReadVersion5`, 56 bytes/cell):

```
grid 293 x 294 cells, cellSize 50.0 units, 86,142 cells
walkable (IsWalkable: NOT_PASSABLE *and* SEE_THROUGH clear): 53,135 (61.7%)
```

Rendered as ASCII it is unmistakably Summoner's Rift. **The whole map's
walkability is an 86 KB boolean array** — a static, shared device constant, not
per-env state. Terrain needs no approximation at all.

Pathing is the interesting part. The server runs A* on this grid
(`PathingHandler.GetPath`), re-pathing every 3000 ms plus on demand from
`ObjAIBase.RefreshWaypoints` (`ObjAIBase.cs:595-676`). A* does not vectorise. But
it does not have to:

```
top-lane corridor, within 1400u of the lane polyline (LanerlLane.TopLaneDefault):
  full corridor            16,339 walkable cells -> all-pairs next-hop  267.0 MB (uint8)
  laning region (mid 50%)   6,344 walkable cells -> all-pairs next-hop   40.2 MB (uint8)
```

**40 MB of static uint8 buys pathing as a table lookup, O(1) per step, with no
search on device.** Precompute once on the host, ship as a device constant.

Two corrections to that plan, both found by porting the server's pathfinder
(`lanerl_jax/data/navgrid.py`) rather than by reading it once:

- **The metric is Chebyshev, not Euclidean.** `GetPath` charges a flat `+1` per
  cell, and `ArrivalCost`/`AdditionalCost`/`Heuristic` are **zero for all 86,142
  Map1 cells** (measured). A diagonal costs the same as an orthogonal step, so a
  table baked with `sqrt(2)` diagonals is a *different pathfinder*.
- **The table must be baked from the server's own A*, not from a shortest-path
  search.** `GetPath` closes cells when it **enqueues** them, so it is not
  guaranteed optimal — and frontier order decides not just which path is found
  but whether one is. Baking shortest paths would bake a pathfinder the server
  does not have.

The alternative — straight line plus wall sliding — is not good enough, and I
measured how not-good-enough:

| goal distance | straight line is walkable |
|---|---|
| ≤ 500u | 98.4% |
| ≤ 1000u | 91.8% |
| ≤ 1800u | 81.0% |

A screen-space click can name a point up to `SCREEN_RADIUS = 1800` away, so a
straight-line approximation would path wrongly roughly one click in five. Use the
table. (Caveat to check in Stage 1: the table gives the *shortest* grid path;
the server's A* plus its path smoothing may pick a different equal-or-near-equal
path, and its tie-breaking is an implementation detail. That is a parity
measurement, not an assumption — see §5, R4.)

### 1.6 JAX runs on the hardware we have

`jax[cuda12]==0.10.2` installs and runs on the GTX 1060 (sm_61, driver
535.309.01): `jax.devices()` returns `[CudaDevice(id=0)]` and a 512×512 matmul
executes on device. sm_61 support was a real risk — recent jaxlib wheels have been
dropping old architectures — and it is not a problem at this version. Pin it.

Caveats: this box has ~1.6 GB of the 1060's 6 GB free (llama-server and another
python hold the rest), so set `XLA_PYTHON_CLIENT_PREALLOCATE=false`. The cluster
is two nodes, one GPU each — `desktop` (16 CPU / 29 GB, the 5080, currently fully
allocated to the live RL run) and `danilogin` (6 CPU / 23 GB, the 1060). The 1060
is the development box; the 5080 is the training box and is not free right now.

### 1.7 The mechanics inventory

What a top-lane 1v1 actually requires, with the source of truth for each. This is
the real scope of the rewrite.

| system | reference | vectorises as | difficulty |
|---|---|---|---|
| fixed-dt tick loop | `Game.cs:474-494` | `lax.scan` over 2 ticks/step | trivial |
| waypoint movement | `AttackableUnit.Move` (`:931`) | consume-waypoints loop, bounded | easy |
| pathing | `PathingHandler`, `RefreshWaypoints` | **next-hop table gather** (§1.5) | easy once baked |
| damage mitigation | `Stats.cs:327-349` — `dmg * 100/(100+stat)` | elementwise | trivial |
| autoattack state machine | `ObjAIBase.UpdateTarget` (`:1165-1330`) | per-unit scalars: `IsAttacking`, `HasAutoAttacked`, `_autoAttackCurrentCooldown`, windup phase | **medium-hard, and it is the whole task** |
| champion target acquisition | `ObjAIBase.cs:1295-1320` | masked argmin over distance² within `AcquisitionRange` | easy (watch tie-break order) |
| turret targeting | `AIScripts/TurretAI.cs` | priority argmin + the "champion attacking a champion" override | easy |
| minion AI | `AIScripts/LaneMinionAI.cs` (395 lines) | 250 ms re-evaluate timer; lexicographic argmin on (attackers, priority, dist²); call-for-help map; temporary-ignore map | **hardest single piece** |
| target priority list | `ObjAIBase.ClassifyTarget` (`:388`) + `ClassifyUnit` enum (14 levels) | integer table lookup | easy |
| wave spawning | `Maps/Map1/LevelScript.cs:171-200, 282-326` | 30 s interval (`MapScriptMetadata.cs:17`), one minion per 0.8 s, 3 melee + 3 caster, cannon every Nth wave, `LANERL_TOPONLY=1` spawns LANE_L only | easy, table-driven |
| minion stat modifiers | `LevelScript.cs:133-156` | constant table | trivial |
| Garen Q/W/E/R + passive | `Characters/Garen/*.cs` (425 lines total) + `Buffs/Garen/*` | 4 spells, small buff set; E is a 3 s buff ticking damage every 500 ms | medium |
| buffs | `Buff.cs` `Update` | fixed-size buff slots per unit, elapsed/duration scalars | medium |
| missiles | `Spell/Missile/SpellMissile.cs` | fixed-size missile pool, linear travel + collision | medium |
| gold / XP / levels | `LanerlEpisode.StartingGold = 475`, ambient 9.5 per 5 s after 90 s (`lanerl_rl/constants.py:346-353`) | elementwise | easy |
| fog of war | `LanerlFow.cs`, vision radii in `constants.py:247-255` | radius test + team OR-reduce | easy |
| episode reset | `LanerlEpisode.Reset` | re-init the state pytree — **free in JAX**, vs 0.23 ms in-process / 12.08 s process restart on the server | trivial |

Deliberately **not** ported (already disabled by `LANERL_TOPONLY=1`, or
irrelevant to laning): jungle camps, mid/bot lanes, inhibitors, nexus, super
minions, items/inventory beyond the starting build, recall, surrender.

### 1.8 Content data is already a swappable table

Champion and spell numbers live in JSON:
`Content/LeagueSandbox-Default/Stats/Garen/Garen.json`,
`Spells/GarenQ/GarenQ.json`, etc. — `BaseHP`, `HPPerLevel`, `BaseDamage`,
`DamagePerLevel`, `AttackRange`, `AttackSpeedPerLevel`,
`AttackDelayCastOffsetPercent`, `AcquisitionRange`, `PathfindingCollisionRadius`,
`MoveSpeed`, and so on.

This is a gift for the second half of the mandate. **The rules engine reads
numbers from a patch table; it never hardcodes them.** Parity runs against
`patch/server/` (baked from these JSON files); the modern-League update is a
second table under `patch/modern/`, plus explicit feature flags for the rules
that genuinely *changed* rather than merely being retuned.

### 1.10 Measured: the fixed-shape budget, and two gaps in the oracle

Recorded on 2026-09-16 from the real server (`LANERL_TOPONLY=1`,
`bot_teams="both"`, scripted bots playing both sides, 300 s of game time,
**18,002 tick snapshots**). Bots reached level 5 with 13 and 17 CS and were
standing in the top-lane corridor at the end, so this is lane play, not a
fountain idle.

| kind | median | p95 | p99 | **max** |
|---|---|---|---|---|
| `LaneMinion` | 15 | 25 | 27 | **28** |
| `Champion` | 2 | 2 | 2 | 2 |
| `LaneTurret` | 24 | 24 | 24 | 24 |
| `Particle` | 2 | 2 | 7 | 7 |
| `Region` | 46 | 46 | 46 | 46 |
| `Inhibitor` / `Nexus` / `LevelProp` | 6 / 2 / 4 | — | — | 6 / 2 / 4 |
| **total entities per snapshot** | | | | **121** |

Also measured: **max 5 buffs on one unit**, **max 5 waypoints on one unit**.

Two things follow.

**Suggested caps** (with headroom, since bronze-tier bots at 13 CS are not an
upper bound on a contested lane): 40 minions, 2 champions, 24 turrets, 16
particles, 8 buff slots, 8 waypoints. That is ~66 dynamic entity slots against a
measured p99 of ~31, which is the right side of the trade — a truncation bug is
silent and a few wasted lanes of a masked array are not.

**All 24 map turrets exist as objects even under `TOPONLY`.** Only two ever act.
Model all 24 as static units anyway: they cost nothing in a masked array, and it
keeps the parity diff complete rather than scoped — and "a map that lost four
turrets over a run" is one of the three bugs that motivated the state dump in
the first place.

**Gap 1 — the diff is scoped, but scoping must never hide.** A snapshot carries
~80 static objects (`Region`, `Inhibitor`, `Nexus`, `LevelProp`) irrelevant to a
1v1. `lanerl_jax.parity.diff` takes a `kinds` filter for readability but
**counts and reports everything it excluded**, precisely so a vanishing turret
cannot hide behind the filter.

**Gap 2 — the dump carries no `TargetUnit`, and this is solved without touching
the server.** `Describe` was written to catch reset leakage, so target
selection — what minion aggro and last-hitting turn on, and the mechanic with
the strictest parity target in §3 (*exact*) — is absent from it.

The vendored server is **shared with every other worktree** and had a commit
land in it on 2026-09-15, so a C# edit is not ours to make. It is also not
needed. Three existing, env-gated, behaviour-neutral outputs cover it, and
`lanerl_jax/parity/targets.py` reads all three:

| what | source | rate | identity |
|---|---|---|---|
| champion `TargetUnit`, `IsAttacking`, `MoveOrder` | control-channel obs `tgt`/`atk`/`mo` (`LanerlControl.cs:232-234`) | per decision | **full** — NetId, resolved against the same message's unit list |
| turret target | `LANERL_TURRET_TRACE=1` (`LanerlHooks.cs:553`) | on change, 250 ms poll | (type, team, distance) — near-unique against the same tick's dump |
| minion retarget | `LANERL_AGGRO_TRACE=1` (`LaneMinionAI.cs:250`) | on change | **kind only** — plus priority both sides, hold time, call-for-help flag |

Validated on a 200 s bot-driven run (2026-09-16): 689 of 689 champion targets
resolved, zero unresolvable NetIds; 153 minion retargets with priorities
matching the `ClassifyUnit` enum (`14→9` = `DEFAULT`→`MELEE_MINION`), hold time
median 1.5 s.

What genuinely remains unobservable, and is named in
`lanerl_jax.parity.diff.UNOBSERVABLE` so it appears in every report rather than
passing as agreement: **minion target identity** (which minion, as opposed to
which kind), `_autoAttackCurrentCooldown`, `HasAutoAttacked`, waypoint positions
and buff time remaining. If minion-target identity becomes the binding
constraint in J1, infer it from damage attribution before requesting a server
change.


### 1.11 Measured: movement and pathfinding parity

Ported `NavigationGrid` and `AttackableUnit.Move`, then compared the server's
per-tick champion trajectory against the port over 24 random move orders in open
lane (`lanerl_jax/parity/movement_parity.py`):

```
worst-tick position error : max 59.904   median 0.062 units
within 1/16 (the dump's own quantisation) : 12/24
within 1 unit                             : 22/24
waypoint count agrees                     : 22/24
A* returned no path                       : 0/24
```

The dump quantises to 1/16 = 0.0625 units, so **a median worst-tick error of
0.062 is parity at the resolution of the oracle**. One number covers the navgrid
parse, translation, `IsWalkable`, `CastCircle`, `GetClosestTerrainExit`, the A*
(including .NET's heap order), `SmoothPath` and the waypoint follower.

Both divergences are **waypoint-count mismatches** — route disagreements in the
pathfinder, not follower bugs. The harness reports `waypoint_count_agrees`
separately precisely so that split needs no second experiment.

Three things this established that the plan had wrong or unstated:

1. **The observation wire truncates positions to whole units**
   (`((int)au.Position.X)`); the state dump keeps 1/16. Parity fixtures must
   seed from the dump. Getting this wrong cost a factor of ~13 in apparent
   error (median 0.9 vs 0.07 units on identical runs).
2. **`GetClosestTerrainExit` is a cumulative drifting spiral**, not a
   fixed-centre polar search. Reimplementing it "sensibly" moved trajectories by
   up to 317 units.
3. **An off-grid point reports walkable at radius>0 and unwalkable at radius 0**,
   because `IsWalkable`'s loop over an empty cell range falls through to `return
   true`. The port reproduces this. It means a move order to an off-map point
   becomes a raw straight line — and the champion then walks into terrain, which
   is the one movement mechanic still unmodelled (it never arises on a pathed
   order).

Falsified along the way, recorded so it is not re-checked: the 3000 ms
`PathingHandler` re-path was **not** the cause of the divergences (divergence
ticks scattered across the cycle; server waypoint counts unchanged), and .NET's
priority-queue tie-break was **not** either (porting the exact 4-ary heap changed
nothing measurable, though it was kept since it removes a known deviation).


### 1.12 MEASURED: the J1 throughput gate, on the RTX 5080

Run 2026-09-16 on `desktop` (RTX 5080, 16 GB, sm_120, driver 580.173.02),
`jax[cuda12]==0.10.2`. The full loop is observation → policy forward → action
decode → two simulator ticks, both champions acting, with the real policy
(`lanerl_jax/train/policy.py`, production dimensions) — which is how gate 4 is
worded, because a sim-only number is the flattering one.

| envs | compile | full loop | vs baseline | sim only |
|---|---|---|---|---|
| 64 | 10.6 s | 113,958 dec/s | 101× | 171,668 |
| 512 | 12.1 s | 181,984 | 161× | 668,481 |
| **1024** | **12.8 s** | **184,783** | **164×** | **779,814** |
| 2048 | 13.0 s | 150,737 | 134× | 728,040 |
| 4096 | 12.6 s | 143,483 | 127× | 604,030 |
| 8192 | 13.3 s | 146,525 | 130× | 543,423 |

Baseline is the production stack's own logged **1,129 decisions/s**
(`runs/rl-league-0915c`, 48,203 s wall, 16,800 updates).

**Gate 4 (≥50×, single run): passed at 164×.** **Gate 5 (compile < 2 min):
passed at 12.8 s.**

Three things the numbers say that the plan could only guess at:

**Risk R8 was the right thing to worry about.** The policy costs roughly 4× the
simulator: 779,814 sim-only against 184,783 with the policy in the loop at the
same batch. Had this been measured sim-only it would have read 690× and been
wrong about where every future optimisation should go.

**The peak is at 512–1024 envs, not at the largest batch.** Throughput falls
~20% by 2048 and stays there. So "fill the device" is not the tuning rule here;
512–1024 is, and the remaining capacity is better spent on parallel *seeds*
(§1.9) than on a wider env axis.

**We are in the regime the problem needs.** A 13.4-hour run at the production
rate bought ~54M decisions. At 184,783/s the same wall clock is **~8.9 billion**
— the difference between the plan's arithmetic and OpenAI-Five-scale experience
budgets.

Caveat, stated because it is the same mistake in a different coat: this measures
the **acting** half only. There is no gradient step in it. Under an Anakin
design the update is part of the same XLA program and its cost adds, which is
why J3 re-measures end to end rather than trusting this.


### 1.9 What the prior work says, and what it says we must not do

The reference architecture is **Anakin** (Hessel et al., 2021, *Podracer
architectures for scalable Reinforcement Learning*, arXiv:2104.06272), and the
framing it gives is worth stating plainly because it names what this project is
already doing:

> Anakin — environment, action selection and learning all execute on the
> accelerator, compiled into a **single XLA program**. No host-to-device
> transfers, no latency-critical work outside XLA, no Python overhead. Requires
> the environment to be a JAX pure function.
>
> Sebulba — environments run on the CPU host; the 8 cores split into actor cores
> and learner cores; trajectories cross a queue.

**The current lanerl stack is a hand-rolled Sebulba.** Process actors, a
rollout queue, `max_staleness`, param versions, latest-wins parameter broadcast,
a GIL workaround — every one of those is a Sebulba component. And the operational
bugs this project has paid for most (undriven champions, silent instance death,
a frozen opponent's experience reaching the learner, dropped param versions) are
Sebulba failure modes. The JAX rewrite is a **move from Sebulba to Anakin**, and
that move deletes the queue, the staleness budget, the param-version plumbing
and the process-actor machinery rather than porting them.

The Anakin paper's third claim matters as much as the speed one: Anakin
experiments are *self-contained and deterministic*. This stack had to **earn**
determinism yesterday by chasing a background `Task.Run` (§1.2). Anakin gives it
by construction, which is the precondition for parity testing existing at all.

**Calibrating the speedup, honestly.** The headline numbers in this literature
are combinations, and they are routinely misread:

| claim | what it actually measures |
|---|---|
| PureJaxRL "over 4000×" | env vectorisation **combined with** training many agents at once. No single benchmark shows it. |
| PureJaxRL "over 10×" | end-to-end JIT alone, vs CleanRL, **same hyperparameters, same env count**. This is the apples-to-apples number. |
| gymnax CartPole "1000×" | env stepping only, 2k parallel envs on an A100 vs 10 envs in CPU numpy |
| SMAX "40,000× vs SMAC" | only when **multiple training runs are vectorised** |
| SMAX "up to 31×" | a **single** training run, one NVIDIA 2080 |
| SMAC 2s3z env stepping | ~6.5× at one env; 83 → 2.7×10⁶ steps/s at 10k envs |

So: **a single training run gets tens of ×, not thousands.** The thousands come
from filling the device with parallel envs and from running many seeds or
configs at once. Both of those are available to us, but they are different
claims and the plan should not conflate them (§4, J1 gate 4 is restated below in
light of this).

**The thing this buys that is not speed.** `vmap` over seeds means a
statistically meaningful number of seeds runs simultaneously on one GPU. Given
that this project's entire methodological stance is that *a learning curve
cannot falsify anything* — three simulation bugs shipped behind believable ones —
turning every ablation from one curve into a distribution over 32 seeds is
arguably worth more than the wall-clock win. Treat "N seeds per experiment" as a
first-class deliverable of J3, not a nice-to-have.

**SMAX is the cautionary tale, not the template.** JaxMARL's SMAX is the closest
published analogue to what we are doing: take a complex, engine-backed game
environment (SMAC, on StarCraft II) and reimplement it in JAX. It got its
40,000×. It also **gave up transfer to do it** — the authors drop Zerg health
regeneration and Protoss shields, remove Medivacs, Colossi and Banelings,
redesign the opponent AI, and change the reward decomposition, then state
plainly that *SMAX and SMAC are different environments, as they have different
opponent policies and dynamics*. SMAX is a fine **benchmark**. It would be a
disastrous model for us, because our deliverable is a policy that plays on the
server and later on real League, not a fast benchmark to publish numbers on.

That is exactly why parity comes before optimisation and before any new
mechanic. SMAX is what this project looks like if parity slips: fast, impressive,
and not transferable. The Tier-1 differential (§3) is the discipline that keeps
the two apart, and it is available to us in a way it was not to SMAX — because
we have a deterministic reference that dumps its full state every decision, and
SMAC does not.


---

## 2. Design decisions

**D0 — The two sides are not symmetric, and the sim must reproduce that.**
Measured from Content: blue and red cannon minions differ in `AttackRange`
(300 vs 280) and `GoldGivenOnDeath` (**35 vs 30**), and three of the four minion
types differ in attack windup. You farm the enemy's minions, so red holds a
structural income edge every cannon wave. This contradicts the mirror assumption
`lanerl_rl/obs.py` canonicalises on and that a mirror self-play setup depends
on. The JAX sim reproduces it (parity first); whether to equalise, model or
accept it is a call for the reward/canonicalisation owner. Pinned by
`lanerl_jax/parity/tests/test_patch.py`.

**D1 — Parity-first, table-driven.** The JAX sim's *logic* mirrors the C# server
line for line where behaviour is observable. Its *numbers* come from a loaded
patch table. Parity is asserted against the server; the modern patch is a table
swap plus flagged rule changes. Never hardcode a number that lives in a JSON file.

**D2 — Fixed shapes, masks, no dynamic allocation.** Struct-of-arrays pytree,
`vmap` over the env axis, `lax.scan` over time. Dead units are masked, never
removed. `LaneMinionAI`'s `temporaryIgnored` and `unitsAttackingAllies`
dictionaries become dense `(N_UNITS,)` / `(N_UNITS, N_UNITS)` arrays. Write it
this way from the first line — retrofitting masks onto branchy code is the
rewrite-inside-the-rewrite that kills these projects.

**D3 — No `lax.cond` on anything batched.** Under `vmap`, `cond` lowers to
`select`: both branches execute. With nested ability/aggro conditionals that is a
compute and compile-time blowup. Write predicates as masks and combine with
`jnp.where`. Track compile time as a first-class metric from week one; if it
crosses ~2 minutes for the full step function, stop and restructure.

The concrete instance to design around is **action dispatch**. The action space
is 8 buttons (`noop, move, attack_move, q, w, e, r, recall`), and a `switch` on
the button under `vmap` executes *all eight branches for every champion every
tick* — including all four Garen abilities with their buff and missile
machinery. This is the documented way JAX environments end up slower than
expected. So actions are never dispatched by branching: each button's effect is
computed as a **masked delta** against a shared state layout and the deltas are
combined. Same rule for the autoattack state machine and for spell effects.

**D4 — The tick is the unit of simulation, the decision is the unit of RL.**
Scan 2 ticks per env step at exactly 1000/60 ms. Everything the server does per
tick, the sim does per tick, in the server's order
(`Game.Update`: `Map.Update` → `ObjectManager.Update` → …).

**D5 — Pin a canonical entity order and match the server's.** Target acquisition
ties break on iteration order (`ObjAIBase.cs:1295-1320` takes the first strictly
closer unit; `TurretAI` notes explicitly that "the player to have been added to
the game first will always be targeted"). `argmin` takes the lowest index. These
must be the same index. This is a silent-divergence source, so it gets its own
test.

**D6 — Port the observation builder too, and test it first.** It is 55% of a
decision today and it has a *better* oracle than the sim does: the existing pure
Python `ObservationBuilder`. Diff JAX obs against Python obs on recorded frames,
offline, no server in the loop, field by field. Do this in parallel with the sim
from day one (§4.2).

**D7 — Start the policy on the MLP core, not the GRU.** `ModelConfig.core` is
already `"gru" | "mlp"`, and `lanerl_rl/model.py` already argues for the ablation
(GT Sophy reached superhuman Gran Turismo with a 4×2048 MLP and no recurrence;
the long-horizon memory in this stack lives in the observation builder, not the
core). Starting on `core="mlp"` with `frame_stack=4` removes BPTT, burn-in
staleness, stored hidden state and chunked sequence minibatching from the first
JAX trainer. Add the GRU back once the loop is trusted — the ablation is one we
wanted anyway.

**D8a — Port the arithmetic, reimplement the architecture.** The line is worth
stating because "faithful port vs reimplementation" is not one question. The
**simulator** is a faithful port: it is the thing parity is measured against,
and reimplementing it freely is the SMAX outcome (§1.9). The **trainer** is a
reimplementation: the queue, the staleness budget, the param versions and the
process actors are Sebulba machinery that Anakin makes unnecessary, and porting
them would import the failure modes the rewrite exists to delete. The PPO
*formulas* are ported literally — dual clip, GAE, the clipped value loss, the
horizon-derived gamma — because those are arithmetic from a paper, and they are
checked against the PyTorch implementation so a transcription slip cannot hide.

**D8 — Fully-compiled train loop, synchronous.** PureJaxRL-style: env, policy and
PPO update all inside one `jit`, `scan` over rollout steps and updates, zero
host round-trips. This deletes the actor/learner queue, the staleness budget, the
param-version plumbing and the process-actor machinery — which is most of where
this project's operational bugs have lived. Self-play opponents are extra
parameter sets carried on the env axis and `vmap`ped, not separate processes.

**D9 — Stack.** `jax` + `flax.linen` + `optax` + `chex`, pinned to
`jax==0.10.2`. Linen because the PureJaxRL / JaxMARL reference PPO the loop will
be lifted from is linen, and because `nn.scan` handles the recurrent core cleanly
when the GRU comes back. No Brax, no gymnax — the env is bespoke.

**D10 — The server never leaves.** It remains (a) the parity oracle, (b) the
evaluation venue for every checkpoint, (c) the venue for anchor bots and human
games. `eval_vs_bot`, `anchor_eval`, and the scripted bronze/gold/diamond anchors
keep working unchanged. Sim-to-server agreement is a headline metric, reported
every time, not a one-off validation.

**D11 — Reset is a constant write, and it must stay one.** The standard JAX
auto-reset pattern computes `where(done, reset_state, step_state)` *every* step,
so the reset path runs on every tick and over 99% of its output is discarded —
a documented cost sink for environments with expensive resets and long episodes.
Ours are long: a 120 s horizon at 30 Hz is 3,600 steps. This is affordable only
while reset is pure array initialisation from static constants (~10 KB per env),
which is cheap next to the sim. **So keep it that way**: no procedural
generation, no host round-trip, no data-dependent work in reset. Measure reset
cost separately in J1 and watch it. This becomes a live risk the moment episodes
start from imported replay states (a later workstream) — at that point resets
must be pre-materialised into a device-side pool and gathered, not computed.

---

## 3. The parity method

Three tiers, cheapest and strictest first.

### Tier 1 — one-step differential (the workhorse)

Replay a recorded server episode. At every decision, **inject the server's state
into the JAX sim**, step both by the same 2 ticks under the same actions, and diff
the resulting states field by field.

- Isolates logic errors from accumulated float drift. A one-step diff cannot drift.
- Does not require the server to reproduce itself across runs, only to have been
  recorded once.
- Runs offline against stored traces — no server process, fast, CI-able.
- Runs in **float64** on CPU, so a disagreement is a logic disagreement.

Fixture: `LANERL_STATE_DUMP=1 LANERL_STATE_DUMP_FULL=1` plus the action stream
already logged by the control channel. Parser and differ are the first thing
built (§4.1).

Per-mechanic targets, set by what actually changes a laning decision:

| quantity | target |
|---|---|
| unit position | ≤ 1/16 unit (the dump's own quantisation) after one step |
| current HP | exact to 1/1024 |
| autoattack fire tick | **exact** — this is last-hitting |
| target selection (all units) | **exact** |
| spell/buff state, cooldowns | exact to 1/1024 |
| move order, waypoint count | exact |

### Tier 2 — free-running divergence (drift, not logic)

Same initial state, same action sequence, let both run free for N decisions, and
measure *when* and *how* they separate. This is the honest number to report: "the
sim and the server agree on CS for the first X seconds, and on champion position
to within Y units for Z seconds."

Expect divergence. C# `float` arithmetic and XLA's fused/reassociated GPU float
arithmetic will not agree bit for bit, and the sim is chaotic — one minion's
target flipping cascades. Tier 2 exists to *quantify* drift, not to eliminate it.
A Tier-2 regression with Tier 1 still green means drift; a Tier-1 failure means a
bug.

### Tier 3 — behavioural / distributional

Run the same scripted policy (the existing `scripted_bronze/gold/diamond`
anchors, and an oracle last-hitter) in both, many episodes, and compare CS@10,
gold@10, HP lost, death count, wave positions. This is the test that catches
"the sim is subtly easier to farm in", which is the failure mode that produces a
great-looking agent that plays badly on the server.

### After the modern-League switch

Tier 1 and Tier 2 **only work against the server patch**. When the modern tables
land there is no oracle any more. So:

- Freeze the parity suite against `patch/server/` and keep running it forever as
  a regression gate on the rules engine.
- Validate `patch/modern/` differently — against published tables and replay
  data — and treat any rules change (not just a number change) as requiring its
  own evidence.
- Never let the modern patch be the only thing that runs in CI.

---

## 4. Stages and gates

Each stage names what "done" means. Do not start the next until the gate passes.

### Stage J0 — Fixtures and the differ (week 1)

Build the thing that will tell you whether everything else works, before building
everything else.

- Record a corpus of server episodes with `LANERL_STATE_DUMP_FULL=1` plus the
  action stream: ~50 seeds × a few thousand decisions, both a scripted-bot drive
  and a fixed scripted action sequence that exercises casts and attacks (reuse
  `test_state_hash_pairs._scripted_action`, which deliberately mixes movement,
  casts and attacks).
- Write the `LANERL_STATEROW` parser → a typed trace format on disk.
- Write the differ: two states in, a per-field, per-entity report out.
- Extract the navgrid to `sr_walk.npy` and bake the corridor next-hop table.
- Bake the patch table from the Content JSON.
- Measure the fixed-shape budget the sim needs. **Done** — see §1.10; redo it
  against a stronger opponent than the bronze bots before freezing the caps.
- Record all three streams per fixture: state dump, observation wire, and the
  two target traces (§1.10, Gap 2). **Done.**
- Corpus must include a stronger anchor than the bronze bots, and at least one
  episode that actually exercises turret targeting — 200 s of bot play produced
  zero turret acquisitions and only 141 attacking decisions in 12,000.

**Gate:** the differ can round-trip a recorded trace against itself with zero
diffs, and reports a legible failure when a field is corrupted on purpose.
The unit/missile/buff/waypoint caps are measured numbers with p100 and headroom.
Every field the §3 targets name is observable from *some* stream, and whatever
is not is named in `UNOBSERVABLE`.

**Status (2026-09-16):** `lanerl_jax/parity/{trace,diff,record,targets}.py` and
`lanerl_jax/data/extract_navgrid.py` written and green — 29 tests, two of which
boot a real server. Shape budget measured (§1.10); targeting oracle validated on
a real run. Remaining: the 50-seed corpus, the next-hop table bake, the patch
table bake.

### Stage J1 — The sim slice: Garen alone, waves, one turret (weeks 2–5)

No enemy champion. Minion waves, one turret per side, one Garen driven by
actions. This is make-or-break.

Build order, each with its own Tier-1 gate: movement and pathing → stats and
damage → autoattack state machine → champion target acquisition → wave spawning →
minion AI → turret AI → Garen Q/W/E/R and buffs.

**Gate:**
1. Tier 1 green on the whole corpus, at the §3 targets.
2. Tier 2 divergence characterised and written down (not necessarily small).
3. An oracle last-hitter — perfect timing from exact HP — scores the same CS@10
   in the sim as on the server, within noise. This is the known answer for the
   slice.
4. **Throughput measured with the real policy in the loop**, not the sim alone:
   ≥ 50× the current stack's 1,129 decisions/s, i.e. ≥ ~56k decisions/s, from a
   **single** training run.
5. Compile time for the full step function under 2 minutes.
6. Reset cost measured separately and confirmed small relative to a step (D11).

On gate 4 being reasonable: SMAX reports *up to 31×* for a single run, so 50× is
at the optimistic end of the published range and we should not be smug about it.
Two things argue we can clear it anyway. First, our baseline is unusually weak —
1,129 decisions/s is bottlenecked by a **pure-Python, GIL-held observation
builder** (§1.1), not by a fast C++ engine as SMAC's was, so J2 alone recovers a
large factor that SMAX never had available. Second, our baseline runs 24 envs;
filling the device is where the env-vectorisation factor lives.

**Gate 4 and gate 5 are MET** — 164× and 12.8 s, measured on the 5080; see
§1.12. The contingency below is kept for the record of what the decision would
have been.

*If gate 4 had failed after three weeks of honest effort: stop and reconsider —
a C/PufferLib-style CPU rewrite, or going back to optimising the existing stack,
both cheaper than a JAX sim that is only 5× faster. A 10–20× result would not
have been a failure of the idea either: PureJaxRL's apples-to-apples
end-to-end-JIT number is ~10×, and the rest of its headline comes from parallel
envs and parallel seeds, which we get regardless.*

#### J1 status, end of 2026-09-16

**Lane stability: RESOLVED.** The sim's lane used to run away to one side and
end with three blue turrets destroyed. The cause was the tick order --
`CollisionHandler.Update()` is the first call in `Map.Update`, which runs before
`ObjectManager.Update` moves anything, so the server separates the positions
units came to rest at last tick and only then moves them. We moved first and
pushed apart afterwards, letting a winning wave keep compressing into the losing
one instead of being spread out before it advanced.

    median live minions   server 21          27 -> 22
    p95 / max             server 27 / 30     39/40 -> 28/31
    mean |blue - red|     server 2.6         11.3 -> 3.3
    mean lane fraction    server .475-.533   .162-.499 -> .439-.540
    turrets destroyed     server 0           3 -> 0

and the lane now oscillates as the server's does, the lead flipping every minute
or two, rather than tipping once and never recovering.

**How it was found, which is the transferable part.** Three individually
verified corrections had each tipped the lane in an unpredictable direction
(missiles worse, turret ramp better, minion-spawn fix worse). That is the
signature of an unstable equilibrium and the sign that hunting asymmetries
one at a time cannot settle it.

What settled it was **Tier 1, the one-step injected differential** -- called for
in this plan since J0 and never built until now (`parity/inject.py`,
`parity/one_step.py`). Take the server's state at tick N, load it into the sim,
step exactly one tick, diff against the server's tick N+1. No accumulation, so
each disagreement is attributable to that tick alone. Over 14,401 predictions,
minion position was exact on 83.1% of ticks and the disagreements were **98.6%
one-sided**. One-sidedness is what separates a missing mechanic from noise, and
free-running comparison structurally cannot produce it -- after the first tiny
difference everything downstream is contaminated. **Build the Tier-1 instrument
before chasing a distributional gap, not after.**

**Green.** Terrain and pathing; wave spawn timing; movement; the auto-attack
clock; target acquisition including call-for-help; damage, kill attribution and
the gold/XP asymmetry; ranged basic-attack missiles with per-unit speeds; turret
identity, stats and time ramp; minion spawn positions; tick phase order; the
minion population and lane balance. Gate 4 (164x) and gate 5 (12.8 s).
**213 tests pass.**

**Open.**

*Gate 3 fails narrowly: 7 CS against the server's 10.* The tick reorder moved
attack opportunities 109 -> 163 without moving CS. The server gets 535. So the
remaining gap is how often a killable minion appears in reach, not what happens
once one does -- i.e. minion HP trajectories, not last-hitting. Next step is to
compare the distribution of minion HP in the band the oracle can one-shot.

*The one-step differential's own blind spot.* Missiles and minion target state
are not in the server's dump at all, so the injector cannot see them: minion
deaths were predicted late 18/18, but 13 had an in-flight missile the harness
is structurally blind to. Closing that needs a dump extension on the server
side, which means touching the vendored tree -- a decision, not a task.

*Turret tiers.* All 24 placed turrets share the outer profile. Other tiers run a
different growth schedule starting at 480 s, inside a 600 s episode, and also
gain armour and MR.

*2 of 24 pathfinder routes disagree.* Map/navgrid identity ruled out (same
inode). Deviation 0.558 units, server corners on our cells; looks like waypoint
emission, and `waypoints` is already NOT_MODELLED. Parked.

**Not built.** Garen Q/W/R, HP regen (turrets are 0 on this map; Garen is
1.568 + 0.1/level plus a separate passive heal), fog of war in `LaneState`, the
next-hop pathing table, the 50-seed corpus, the N-seeds vmap (J3 gate 5).

### Stage J2 — The observation builder in JAX (weeks 2–5, parallel to J1)

Independent of J1 and with a much better oracle, so it runs concurrently.

Port `lanerl_rl/obs.py` to a pure JAX function over the sim state. The stateful
parts — fog memory, `EnemyAbilityIntel`, `AttackClock`, last-seen positions,
staleness, reachability radii — become explicit arrays in the env state pytree.

**Watch out:** `obs.py`'s module docstring is stale. It documents `entities
(32,40)`, `self_vec (64,)`, `global_vec (48,)`; `constants.py` says
`N_SLOTS=32`, `SELF_DIM=16`, `GLOBAL_DIM=6`, `PRIV_DIM=96`, `ENTITY_DIM=_MT+3`.
**`constants.py` is the source of truth.** Action heads are
`8 buttons × 96 screen_x × 54 screen_y × 32 targets`.

**Gate:** on every frame of the recorded corpus, the JAX observation equals the
Python `ObservationBuilder`'s output field-for-field within float tolerance,
including the fog-masked and stale-entity cases, and including the red-side
mirror. The existing invariants — `test_obs_guards`, `test_fog`, `test_mirror`,
`test_no_dead_features` — are ported and green.

### Stage J3 — The compiled trainer (weeks 5–7)

PureJaxRL-shaped PPO over the J1 sim and J2 observation, `core="mlp"`.

Port dual-clip PPO (`lanerl_rl/ppo.py`) — the second clip at `c=3.0` for negative
advantages, `horizon_s`-parameterised gamma, value clipping, KL-to-reference,
advantage normalisation. Port `ZeroSumLaneReward` (`lanerl_rl/reward.py`) exactly,
including the potential-difference HP term and the ambient-gold subtraction.

**Gate:**
1. `test_actor_learner_agree`'s property holds in the JAX loop: the log-probs the
   rollout recorded equal the ones the update recomputes.
2. Loss components, advantage statistics and entropy match the PyTorch
   implementation on an identical fixed batch, to float tolerance. **Met** —
   GAE, the dual-clip surrogate, the clipped value loss and all four of
   `ppo.py`'s worked gamma values reproduce (`train/tests/test_ppo.py`).
3. The oracle last-hitter's return is reproduced by a policy trained from scratch
   in the sim on the J1 slice.
4. **RESTATED.** This said "end-to-end throughput ≥ the J1 gate". That was
   wrong, and measurement is what showed it: the J1 figure was taken on
   **acting** — the plan says so in §1.12 — and carrying it across to a loop
   that includes the gradient step was optimistic rather than demanding.

   Measured on the 5080 at 256 envs, per decision:

   | stage | cost | share |
   |---|---|---|
   | simulator | 2.07 µs | 4% |
   | observation + policy | 4.28 µs | 8% |
   | PPO update | 45.8 µs | **88%** |

   4 epochs × 4 minibatches is 16 gradient passes over every rollout, so the
   update costing ~9× the acting is **structural to PPO**, not a defect. The
   gate is therefore against the thing that actually matters — the production
   stack's 1,129 decisions/s — and the standing figure is **19,168 env-dec/s
   end to end, 17×**, sustained.

   Note what this says about where optimisation should go: the simulator is 4%
   of the cost. Further sim work buys almost nothing; epochs, minibatch shape
   and model size are the levers.
5. **N-seeds-per-experiment works**: an ablation runs ≥ 32 seeds in one `vmap`
   and reports a distribution, not a curve (§1.9). This is a deliverable, not a
   bonus — it is the methodological upgrade the rewrite is really buying.

### Stage J4 — Mirror self-play, and the transfer report (weeks 7–10)

Add the enemy Garen. Opponents are parameter sets on the env axis, `vmap`ped —
a frozen-checkpoint pool sampled on the host, stacked on device. Port the
`opponent never trains` invariant (`test_opponent_never_trains`) first, not last.

**Gate:** a policy trained in the sim beats the scripted anchors *on the server*,
and the sim-vs-server win-rate gap is measured and reported every eval. If they
diverge, stop training and fix the sim — a sim-only win rate is not a result.

### Stage J5 — The modern-League patch table (after J4)

Only once the server-parity engine is trusted and frozen as a regression gate.
Second patch table, explicit feature flags for changed *rules*, and a fresh
validation story (§3) because the oracle is gone.

---

## 5. Risk register

Fix, avoid, or accept — stated per risk, because pretending all risks get fixed
is how schedules die.

**R1 — Branchy game logic under `vmap` (compute/compile blowup).** *Avoid.*
Masks and `jnp.where` from day one (D3); never `lax.cond` on a batched predicate.
Track compile time as a metric. Escape hatch: if the step function is still ugly
and slow at the end of J1, the answer is a C rewrite, not more JAX tuning.

**R2 — The minion AI is the hardest piece and last-hitting depends on it.**
*Fix, with budget.* Its 250 ms re-evaluation timer, lexicographic
(attackers, priority, dist²) argmin, call-for-help priority map and 500 ms
temporary-ignore map are all stateful and all observable in the state dump. Give
it its own Tier-1 sub-suite. `LANERL_AGGRO_TRACE=1` already emits per-minion
retarget events (`LaneMinionAI.cs:31-37`) — use it as a second, finer oracle.
Note the deliberate server deviations already documented in that file (the
first-wave exception is not implemented; `CountUnitsAttackingUnit` is disabled) —
**match the server's behaviour, including its deviations from real League.**
Parity is against the vendored server, not against the wiki.

**R3 — Float drift between .NET float32 and XLA float32.** *Accept, and measure.*
Tier 1 in float64 on CPU to judge logic; Tier 2 to quantify drift; never treat a
long-horizon divergence as a bug until Tier 1 is checked.

**R4 — The next-hop table's path ≠ the server's A* path.** *Measure first, then
decide.* The table gives a shortest grid path; the server's A* plus smoothing may
return a different one of equal length, and its tie-breaking is incidental. Test
in J0/J1 by replaying recorded champion move orders through both and diffing the
waypoint lists. If they disagree often, the options are to bake the tie-break
into the table's construction order, or to bake the server's own `GetPath`
outputs for the corridor rather than recomputing shortest paths.

**R5 — Silent entity-ordering divergence (D5).** *Fix.* A dedicated test that
constructs exact-tie target situations and asserts both implementations pick the
same unit.

**R6 — The observation builder's hidden state diverges from the sim's.** *Avoid
by construction.* Fog memory and ability intel live in the env state pytree, not
in a side object; there is no place for them to drift to.

**R7 — The sim becomes subtly easier and the agent overfits to it.** *Fix via
Tier 3 and D10.* Every checkpoint plays the server. The transfer gap is a
headline number. Each discovered exploit becomes a regression test.

**R8 — Throughput collapses once the policy is in the loop.** *Expected; measure
early.* The entity transformer over 32 slots at large batch may well dominate the
sim. Measure with the real policy in week one of J1, before optimising anything
in the sim. Levers, in order: shrink the model, drop to 15 Hz decisions
(`STEP_TICKS=4`, already supported), then optimise the sim.

**R9 — GPU availability.** *Accept and plan around it.* One GPU per node; the
5080 is busy with the live RL run and the 1060 has ~1.6 GB free. J0, J2 and most
of J1's Tier-1 work are **CPU-only and float64 by design**, so the critical path
does not need the GPU. Only the throughput gates (J1 gate 4, J3 gate 4) do.

**R11 — Reset-under-`vmap` eats the step budget.** *Avoid by construction, then
measure.* See D11. Known failure mode in this literature for long-episode,
expensive-reset environments; ours is only safe while reset stays a constant
write.

**R12 — Action dispatch executes all eight branches.** *Avoid by construction.*
See D3. Masked deltas, never a `switch` on the button head.

**R13 — Drifting into SMAX.** *The named failure mode of this whole project.*
Every approximation taken "just for now" that is not booked as a parity failure
and paid back is a step towards a fast environment that trains an agent which
cannot play on the server. The gate is mechanical: an approximation is allowed
only when it is written down in the parity report with its measured Tier-1
disagreement, and the transfer eval (J4) is what catches the ones we did not
notice we were making.

**R10 — Scope creep into real League.** *Avoid.* Parity target is the vendored
server at its current commit, deviations included (R2). Modern League is J5 and
is a table, not a rewrite.

---

## 6. Repo layout

```
lanerl_jax/
  data/            navgrid extraction, next-hop table baking, patch tables from Content JSON
  patch/
    server/        baked from Content/LeagueSandbox-Default  <- parity target
    modern/        J5
  sim/             the rules engine: state pytree, tick step, movement, combat,
                   minion AI, turret AI, spells, buffs, missiles, waves, fog
  obs/             the JAX observation builder (port of lanerl_rl/obs.py)
  train/           compiled PPO, self-play, league sampling
  parity/          trace parser, differ, Tier 1/2/3 harnesses
  tests/
```

The server-side trees (`lanerl_rl`, `lanerl_train`, `lanerl_bot`) stay exactly as
they are. They are the oracle and the evaluation venue (D10), not legacy.

---

## 7. What to do first

Four things, in this order, none of which need a GPU:

1. **The differ and the trace corpus** (J0). Everything downstream is measured
   with it, so it exists first.
2. **Bake the static data**: the navgrid bitmap, the corridor next-hop table, the
   patch table from the Content JSON. All three are host-side one-offs and all
   three are already shown to work.
3. **Measure the fixed-shape budget** from the corpus. Guessed caps are silent
   truncation bugs.
4. **Start J1 and J2 in parallel** — they are independent and J2's oracle is far
   cheaper, so it will find design problems in the state pytree sooner.

The single most informative early measurement is **R8**: put the real policy in
the loop against even a stub sim and find out what the throughput ceiling
actually is. If the policy forward dominates, the whole shape of the J1 gate
changes, and it is better to know that in week one than in week five.

---

## 8. References

- Hessel, Kroiss, Clark, Kemaev, Quan, Keck, Viola, van Hasselt (2021).
  *Podracer architectures for scalable Reinforcement Learning.* arXiv:2104.06272.
  — Anakin vs Sebulba; the single-XLA-program design; determinism and
  self-containedness. **The architecture this rewrite is adopting.**
- Lu, Kuba, Letcher, Metz, Schroeder de Witt, Foerster (2022). *Discovered Policy
  Optimisation.* NeurIPS 35, 16455–16468. Codebase: `luchris429/purejaxrl`;
  blog: chrislu.page/blog/meta-disco. — end-to-end JIT PPO, the ~10× / 4000×
  decomposition, `vmap` over seeds and agents.
- Rutherford et al. (2023). *JaxMARL: Multi-Agent RL Environments and Algorithms
  in JAX.* arXiv:2311.10090. — SMAX. **Read as the cautionary tale (§1.9), not
  the template**: an approximate reimplementation that explicitly is not the
  same environment as the one it replaces.
- Lange. *gymnax*; Freeman et al. *Brax*; Bonnet et al. *Jumanji*; Matthews et
  al. *Craftax* — prior JAX-native environment suites; useful for step-function
  idioms, not for this domain.
