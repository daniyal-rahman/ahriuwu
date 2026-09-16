# Tier 1 after the tick reorder — gate-1 investigation, round 2

One-step injected differential: take the server's state at tick N, load it
into the sim, step exactly one tick, compare against the server's tick N+1.
No accumulation, so each disagreement is attributable to that tick's
mechanics alone.

**This entire document, both rounds, is a SINGLE-TRACE (n=1) result.** Gate
1's full text is "Tier 1 green **on the whole corpus**", and the 50-seed
corpus J0 calls for has never been built — every Tier-1 number here, before
and after, comes from one idle-lane recording (job 807's 420s trace, and this
round's fresh 420s re-recording under the fixed code). Nothing below should
be read as "the corpus passed"; there is no corpus yet.

## Correction to the first round's headline claim

The first round of this doc said:

> position/heading accuracy is currently capped by what the injector can
> reconstruct... improving the injector is now on the critical path for that
> gate

That claim does not survive the gate's own criterion. `docs/JAX_REWRITE_PLAN.md`
§3 sets the Tier-1 position target as **≤ 1/16 unit (the dump's own
quantisation), not bit-exact**. The first round measured and reported
*bit-exact* agreement only. Recomputed under the real target, on the
untouched original run:

| | bit-exact | ≤ 1/16 |
|---|---|---|
| LaneMinion.position, trustworthy injection | 81.10% | (not computed first round) |
| LaneMinion.position, untrustworthy injection | 40.61% | (not computed first round) |

and on this round's job (before any injector change, i.e. an apples-to-apples
recheck of the SAME instrument the first round measured):

| | bit-exact | ≤ 1/16 |
|---|---|---|
| trustworthy | 68.90% | **88.73%** |
| untrustworthy | 50.50% | **85.85%** |
| gap | 18.4 pts | **2.9 pts** |

The trustworthy/untrustworthy gap loses 84% of its size under the gate's real
criterion. **The injector was not the blocker it was reported to be** — it
accounts for roughly 3 points of minion position accuracy, not 18. That is
still worth closing (see below), but it is not what stands between this gate
and green. The real blocker is the tail: **11-14% of minion positions still
miss the ≤1/16 target even on the trustworthy subset**, with p95 ≈ 5.4 and
max ≈ 8.0 units — 87x and 128x the target, and not quantisation. That tail is
what the rest of this document chases.

## What changed this round

1. **`inject.py`: `reconstruct_waypoints_relaxed`.** A minion marching
   (`MOVE_TO`) off the known lane corridor (usually one that just gave up a
   chase) used to be refused outright (frozen, `movement_trustworthy=False`).
   Measured against real trace kinematics (does a guessed waypoint predict
   the ACTUAL next-tick position, no sim involved): the same corridor
   projection with no perpendicular-distance gate beats "predict no
   movement" on 87.1% of ticks (median error 0.134 units, 82.5% within 0.5
   units) on a 6,000-tick-pair sample. Wired in as a fallback, flagged
   trustworthy with an honest "measured to help, not recovered" reason
   string.
2. **`inject.py`: the ATTACK_TO chase-target guess was tried and REJECTED.**
   Symmetrical experiment for a chasing (`ATTACK_TO`) minion's target
   identity: guess the nearest strictly-best-priority enemy in range (a
   from-scratch acquisition, matching what the sim itself would compute).
   Measured, not assumed: 70.5% of guesses landed on an enemy already IN
   attack range, which is impossible for the server's true (still-chasing)
   incumbent — confirms `sim/minion_ai.py`'s documented hysteresis gap: the
   true target is usually a farther, protected unit a fresh nearest-search
   never considers. Worse, "predict no movement" already beats the guess
   outright (86.3% of ATTACK_TO minions land within 0.5 units of their
   pre-tick position one tick later — most are jammed in a collision scrum,
   functionally stationary despite nominally still closing). **This is a
   negative result, reported as one**: the code was written, measured, and
   reverted; ATTACK_TO minions are still frozen and flagged untrustworthy,
   now for a measured reason instead of an assumed one.
3. **`inject.py`: `replay_wave_states` had an inverted pairing (bug, not
   approximation).** Its own docstring claimed "the state paired with a
   snapshot at time T is the result of the previous tick's spawn decision,
   not one evaluated at T itself" — backwards. `trace[i]`'s population
   already reflects `trace[i]`'s own tick's spawn decision (the dump is
   written after that tick's `LevelScript.Update` ran), so pairing it with
   PRE-tick-i counters made `tick()` re-evaluate the identical threshold at
   the identical game time and spawn a SECOND time on top of the minion the
   dump already shows. Fixed: `replay_wave_states` now advances through each
   `trace[i].t_ms` before pairing with index i. Confirmed against the trace
   (first wave, first arrival: population unchanged 1→1 on the server, the
   unfixed pairing predicted a second spawn).
4. **`sim/step.py`: a real, general timing bug, not just a harness one.**
   `Game.Update` (`Game.cs:474-497`) increments `GameTime` BEFORE `Map.Update`
   and `ObjectManager.Update` in the same call, so every per-tick absolute-
   game-time check the server makes — wave spawning in `LevelScript.Update`,
   the turret AD/armour ramps in that same script's `LevelScriptObjects.
   OnUpdate`, `Champion.Update`'s ambient-gold gate — reads the
   POST-increment value: THIS tick's own outgoing time, not the incoming one.
   `tick()` was checking all three against `state.t_ms` (the incoming/old
   time). Fixed: a single `t_now = state.t_ms + delta_ms` computed once per
   tick, used everywhere one of these checks used to read `state.t_ms`
   directly. This is a **simulation** bug, not only a Tier-1 measurement one
   — it affects every real training/eval run, not just this differential.
5. **`one_step.py`: the gate's own ≤1/16 metric, and a missile-free bias
   split**, added as new fields (`position_le1_16[_untrustworthy_injection]`,
   `position_along_heading_missile_free`) alongside the existing bit-exact
   ones, plus an explicit trustworthy-vs-untrustworthy summary under both
   criteria in the report. Nothing removed; both readings stay available.

## Exact — unchanged

```
Champion.position      39600/39600   100.00%
Champion.hp            39600/39600   100.00%
Champion.move_order    39600/39600   100.00%
Champion.waypoints     39600/39600   100.00%
LaneTurret.position   475200/475200  100.00%
LaneTurret.move_order 475190/475200   99.998%
LaneTurret.waypoints  475200/475200  100.00%
LaneTurret.hp         475116/475200   99.98%   <- unchanged by the t_now fix,
                                                   see "still open" below
```

Champion and turret position/movement mechanics remain, on a one-step basis,
exactly correct. `LaneTurret.hp`'s 84 mismatches (all one-sided, sim's turret
ends up +13.3 HP ahead of the server's on average) are **unchanged, digit for
digit**, by the `t_now` fix — so whatever causes them is NOT the ramp-timing
bug that fix targets. Not investigated further this round; flagged for
whoever picks this up next.

## Minions: before and after, both criteria

```
                              BEFORE (bit-exact / ≤1/16)      AFTER (bit-exact / ≤1/16)
LaneMinion.position, trustworthy    68.90% / 88.73%              68.85% / 88.66%
LaneMinion.position, untrustworthy  50.50% / 85.85%              50.50% / 85.85%
```

(`BEFORE` = this round's job with only the wave-state pairing fix and the
relaxed-reconstruction fallback already in, i.e. the run the harness-blame
correction above is computed from. `AFTER` adds the `sim/step.py` `t_now`
fix.) **Essentially unchanged, as expected** — `t_now` only touches wave
spawning and the turret AD/armour/gold ramps, none of which govern ordinary
minion movement. Its effect shows up in the wave-spawning section below, not
here. The tiny (0.05-0.07 pt) drop is sampling noise from the trustworthy
population itself shifting slightly (159,341 → 159,455) because a handful of
ticks near wave-spawn boundaries got injected differently once spawning
timed out correctly.

## Wave spawning — a REAL bug found and partially fixed, not fully closed

Verdict, in one line: **it was a genuine off-by-one in the wave-spawner's
tick-relative timing (item 4 above), not purely a harness artifact — but
fixing that timing bug only closed a modest fraction of the disagreement, and
most of what remains looks like a SEPARATE harness identity-matching
artifact, still open.**

Before any fix: 1,665/1,667 spawn ticks disagreed (first round) / 1,551/1,551
(this round's baseline, before the `t_now` fix) — effectively 100% either
way. After the `replay_wave_states` pairing fix alone: still ~100% on a
3,000-tick sample. After ALSO fixing `sim/step.py`'s `t_now`: **1,447/1,551**
on the full 420s run — an improvement, but the disagreement rate barely
moved (93.3% still disagreeing, down from ~100%).

Direct evidence for the timing bug (now fixed): at the very first wave
(t≈90,000 ms), the unfixed pairing predicted a SECOND minion spawning into a
tick where the server's own population was unchanged (1→1, 0 real
deaths/arrivals) — a spurious double-spawn. Separately, at t=90,798 ms
(team 200's second arrival that wave), the unfixed check evaluated
`90,798 >= 90,800` (false, no spawn) when the server's own transition to the
NEXT tick (90,815 ms) already showed the new minion — i.e. the server's
decision used 90,815, not 90,798, exactly matching the `Game.Update`
ordering item 4 above documents. Both cases are gone after the fixes.

**Confirmed, at full-trace scale (`lanerl_jax.parity.tier1_wave_spawn_detail`,
19,800 ticks): 100% of the remaining 1,681 unmatched "new" entities have a
nearest pre-tick same-group candidate MORE than 8 units away** (median 14.4,
p90 32.3 units) — i.e. **zero** of them are a near-miss on the match radius
that a real population change would produce. All are `n_real_new > n_sim_new`
(the sim undercounts); the earliest examples are concentrated on team 200
(red) in the first few seconds after a wave starts. This rules out "the match
radius is merely a little tight" and is consistent with the harness-artifact
hypothesis above (`n_real_new`/`n_sim_new` are computed from two INDEPENDENT
procedures — greedy nearest-position identity matching vs a population-count
delta — and a marching, closely-spaced minion can make the greedy matcher
swap which individual it calls "the same one", producing a spurious
death+arrival pair that nets to zero population change but still scores as a
"1 real arrival" against the sim's correctly-computed "0 new"). **Not fully
diagnosed**: I did not pin down why this concentrates on red / early-wave
ticks specifically before this question was handed to a dedicated
content/waves audit agent (per the coordinator; I am not pursuing it
further, to avoid duplicating that work). If confirmed as identity-matching
noise, the fix is to score a group's spawn count from the RAW population-size
delta (`len(real_list) - survivors`) rather than from identity-continuity of
individuals, which is the wrong question for a count metric — but that
decision belongs to whoever settles the diagnosis.

## The position tail: collision, not movement integration

Restricted to missile-free, trustworthy ticks (the two known blind spots the
plan called out), bucketed by how many OTHER live minions/champions are
within collision-touching distance of the unit's PRE-tick position:

```
neighbours=0:  n=5101  93.4% within 1/16   mean signed +0.08
neighbours=1:  n=680   27.1% within 1/16   mean signed +0.52
neighbours=2:  n=214   33.2% within 1/16   mean signed +0.83
neighbours=3+: n=70    18.6% within 1/16   mean signed +1.18
```

**Isolated movement is essentially exact** (93.4%, mean bias +0.08 — noise).
Every point of the residual is in collision-affected movement, and it scales
with how crowded the unit is. This kills the earlier framing of the residual
as "movement integration is slightly ahead" — it's specifically the
collision pass, which is a genuinely different, more scoped question.

`sim/collision.py`'s push formula itself is right: `r1 = pathfinding_radius +
1.0`, `r2 = pathfinding_radius` exactly matches `Extensions.
GetCircleEscapePoint(Position, PathfindingRadius + 1, collider.Position,
collider.PathfindingRadius)` in `AttackableUnit.OnCollision`
(`AttackableUnit.cs:278-318`) — the escape magnitude and the choice of
`PathfindingRadius` over collision radius are correct. What differs is
**sequencing**, and it's three things, all read directly from
`GameServerLib/Handlers/CollisionHandler.cs:121-156`:

1. **Gauss-Seidel, not Jacobi.** The server's `Update()` loops `_objects` and
   calls `UpdateCollision(obj)`, which ends in `SetPosition(exit, false)` — an
   immediate in-place teleport. A later unit in the SAME pass escapes from an
   EARLIER unit's already-moved position. `resolve_collisions` computes every
   push from the same pre-tick snapshot and applies them simultaneously. This
   differs even for a single pair (explaining why the n=1 bucket is 27.1%,
   not ~93%): the second unit of a pair escapes from stale coordinates in our
   version.
2. **Multiple pushes per unit per tick.** `UpdateCollision` loops over EVERY
   neighbour `GetNearestObjects(obj)` returns and calls `OnCollision` for
   each one, each a fresh teleport. `resolve_collisions` applies exactly one
   push, from the lowest-index overlapping neighbour — a documented,
   deliberate approximation (`sim/collision.py`'s own docstring: "This
   applies one push per unit per tick"). The 2- and 3+-neighbour buckets are
   where that approximation comes due. (The "lowest-index" tie-break itself
   turns out to be a mis-transplanted rationale, not just an unverified one:
   that rule is real, but it belongs to TARGETING -- `ObjAIBase.cs:1295-1320`
   breaks distance ties by `ObjectManager.GetObjects()` order -- not to
   collision, where `UpdateCollision` does not pick "the" neighbour by index
   at all, it pushes against every one `GetNearestObjects` returns. Corrected
   in `sim/collision.py`'s own docstring alongside this finding.)
3. **Iteration order is unverified.** `resolve_collisions`'s comment claims
   "lowest-index overlapping neighbour, matching the server's iteration
   order" — but the server iterates `GetNearestObjects(obj)` (a quadtree
   query) for the inner loop and `_objects` (list/add order) for the outer
   one; neither is obviously our slot-index order. This matters more once
   (1) is fixed, since it decides who escapes from stale coordinates and who
   from fresh.

**Point 3 resolved by reading the QuadTree source, not by assuming: the
server's collision tree is BROKEN and degenerates to a flat,
insertion-ordered list for every practical position on this map.**
`CollisionHandler`'s constructor builds the tree as

```csharp
_quadDynamic = new QuadTree<GameObject>(
    _map.NavigationGrid.MinGridPosition.X,   // top
    _map.NavigationGrid.MaxGridPosition.Z,   // left  (yep, MAX -- the code says so)
    _map.NavigationGrid.MaxGridPosition.X - _map.NavigationGrid.MinGridPosition.X,
    _map.NavigationGrid.MaxGridPosition.Z - _map.NavigationGrid.MinGridPosition.Z
);
```

Map1's measured navgrid bounds (`lanerl_jax.data.navgrid.NavGrid.load()`):
`min_grid=(-328.9, ..., -110.2)`, `max_grid=(14311.8, ..., 14556.9)` (X, Y,
Z). Plugging these in: the root rect's `Left` is **14,556.9** — bigger than
every X coordinate anywhere on the map (max 14,311.8). `Quadrant.Insert`
only descends into a child quadrant when `Circle.ContainedBy(childRect)`
succeeds, and that check requires `childRect.Left <= Position.X - Radius`;
with `Left >= 14,556.9` for every one of the four children (root split in
half, both halves start at 14,556.9 or 21,877.2), **no real game object ever
satisfies it**. So `Insert`'s `while(true)` loop takes the "no child
contained it" branch on its very first iteration, every time, for every
object, and the object lands in the ROOT quadrant's own flat list. The tree
never subdivides, for anyone, ever, on this map. (The query side,
`GetIntersectingNodes`, uses a different, axis-symmetric method,
`Circle.IntersectsWith(Circle)`, which is NOT affected by the same bug — so
candidate SELECTION is still geometrically correct, just delivered by a
linear scan instead of a tree, and — because `Quadrant.Insert` links each new
node in as `n.Next = x.Next; x.Next = n`, then `GetIntersectingNodes` walks
from `last.Next` — **in true insertion order.**)

Net effect: `GetNearestObjects(obj)` returns every collision-tracked object
whose LAST-REBUILT bounds intersect `obj`'s query circle, **in the order
they were added to the tree** — which itself mirrors `CollisionHandler.
_objects`' own add order, since `UpdateQuadTree()` rebuilds the tree once
per tick by re-inserting `_objects` in that list's order. So **both the
outer loop (`_objects`) and the inner one (`GetNearestObjects`) reduce to the
SAME ordering: true object-add order** — not spatial locality, and (this
project's own `collision.py` docstring was checked against source and does
not survive it) **not slot/dump-content order either.**

**This is now confirmed, and the exact object-add order is NOT yet fully
recovered — see below for what is and is not resolved:**

* **Champions**: 2, added at match start; relative order (which team first)
  not chased down — almost certainly irrelevant here, since neither champion
  ever moves or comes near a minion in this idle fixture, so their position
  in the sequence cannot affect minion-vs-minion collision outcomes.
* **Turrets**: irrelevant regardless of order — `IsCollisionAffected`/
  `IsCollisionObject` both exclude `BaseTurret` (`CollisionHandler.cs:39-56`),
  so turrets neither push nor are pushed and never appear in this list at
  all.
* **Lane minions, across waves**: fully recoverable in principle. Spawn
  timing is deterministic and RNG-free (`sim/waves.py`, already relied on
  elsewhere in this doc), so a minion's birth tick -- and hence its rank
  among all minions ever spawned -- is knowable exactly by replaying the
  schedule and tracking identity forward tick-to-tick (the same
  nearest-position technique `parity/diff.py` already uses for death/spawn
  matching). **Not built**: this needs whole-trace state, which the current
  per-tick-independent Tier-1 injector does not carry. Proposed next step,
  not implemented this round.
* **Lane minions, within the SAME wave arrival** (`SetUpLaneMinion` spawns
  one minion per barrack per call, `LevelScript.cs:283-322`, iterating
  `LevelScriptObjects.SpawnBarracks` -- a `Dictionary<string, MapObject>`
  populated from the map's own scene-object list in
  `LevelScriptObjects.LoadSpawnBarracks`): the blue-vs-red order within one
  simultaneous arrival depends on the ORDER OBJECTS WERE PARSED OUT OF THE
  MAP'S SCENE FILE, which I did not locate in the time available (it is not
  in any script this project already reads; it would need the raw scene
  manifest, not the individual per-object `.sco.json` files this repo already
  has). **Flagged as unresolved, not guessed past.**

**Measurement taken anyway, with the ordering caveat stated up front**: as a
first, explicitly-approximate data point -- NOT a claim that this is the
server's true order -- `lanerl_jax/parity/tier1_collision_sequential.py`
implements the faithful SEQUENCING (Gauss-Seidel: each push is an immediate
in-place update visible to later checks in the same pass; multiple pushes
per unit per tick, one per overlapping neighbour, re-checked live) but uses
SLOT order (this injector's own, dump-content-based assignment) as the
stand-in for the true add order, and checks every pair directly rather than
replicating the (geometrically-equivalent, just slower) tree query. It feeds
each collision hypothesis (current Jacobi vs. this sequential reference)
through the SAME downstream one-tick movement integration, isolating
collision's effect rather than re-running the whole tick (which would also
perturb targeting on a slightly different position). An earlier version of
this measurement had a real bug (used the trace's OWN observed tick spacing
for the movement budget instead of the fixed `TICK_MS` constant `tick()`
itself always uses, which alone was enough to make even the ZERO-neighbour
"isolated" bucket read 13.9% instead of the ~93% job 822 measured through the
real `tick()` -- caught by that exact inconsistency, fixed before trusting
any number from this script):

```
                  current (Jacobi)   sequential, SLOT order   sequential, RECONSTRUCTED CREATION order
neighbours=0:  n=5884  94.4%             94.4%                    94.4%   (identical -- no collision to speak of)
neighbours=1:  n=1253  75.5%             72.5%  (-3.0)             81.3%  (+5.8 vs Jacobi, +8.8 vs slot)
neighbours=2:  n=411   70.1%             56.7% (-13.4)             73.0%  (+2.9 vs Jacobi, +16.3 vs slot)
neighbours=3+: n=78    48.7%             43.6%  (-5.1)             52.6%  (+3.9 vs Jacobi, +9.0 vs slot)
```

**Slot order is worse than Jacobi at every crowding level (as reported
initially); reconstructed CREATION order beats Jacobi at every crowding
level.** This directly confirms the source-derived prediction: `_objects`
is genuinely in creation order and genuinely load-bearing, and the earlier
slot-order result was evidence the mechanism was UNTESTED, not that it was
wrong. The creation order used here is **reconstructed, not observed** --
built from the deterministic wave schedule (exact) plus a proxy for which
of a team's currently-alive minions is oldest: cumulative progress along
that team's own lane corridor (a minion that has walked further is assumed
older; the true test is `_objects`' own preserved survivor order, not
directly visible from a single snapshot) -- and a flagged guess for
same-wave-event blue-vs-red tie-break. Both are named approximations, not
hidden ones.

**The n=1 bucket is the clean test, and it moved as predicted.** With no
inner (quadtree) order to confound it (one colliding neighbour means exactly
one escape, in any traversal order), 1,253 samples -- 72% of all crowded
samples -- isolate the OUTER order alone. It moved from 72.5% (slot) to
81.3% (creation), a real gain, though short of the ~94% isolated units reach
-- the remaining gap at n=1 is attributable to the reconstruction's own
imprecision (the arc-length proxy stalls for a minion mid-fight while a
younger one behind it keeps closing, and the blue/red tie-break is an
unverified guess), not to inner order, which cannot matter here.
**Verdict: the algorithm shape is confirmed right, in the sense that a
BETTER approximation of the true order produces a BETTER result at every
bucket including n=1's outer-order-only test — order is not just
load-bearing in theory, it moved the number in practice.**

**Where this stops, deliberately, per explicit instruction not to build a
host-side/non-JIT reference:** the natural next steps -- inverting the
server's true per-tick permutation from the k! ways a k-neighbour collision
could have resolved (to build a labelled corpus for the INNER/quadtree
order, which only n=2 and n=3+ need -- 28% of crowded samples, 6% of all),
fitting/scoring candidate inner-order models (Morton/Z-order-with-depth
correction, nearest-first, farthest-first, x-then-y) against that corpus
with a proper held-out split, and then a three-way PRODUCTION design
comparison (current Jacobi vs. an exact `lax.scan`-based Gauss-Seidel vs. a
vectorised Morton-ordered approximation, each measured on BOTH crowded-bucket
accuracy and throughput/compile-time the way J1 gate 4/5 measured them) --
is substantial engineering and benchmarking work in its own right, explicitly
flagged by the coordinator as hand-off-able rather than rushed. **Not started
this round.** A sibling audit already has the quadtree/`_objects` source
ledger; the harness in `tier1_collision_sequential.py` (in particular
`estimate_creation_order`, which a real identity tracker would replace) is
ready for whoever picks this up.

## Damage: one-sided, and a known cause, not a new one

```
HP-change disagreement:  server-only 1751, sim-only 0
deaths:                  SIM ALIVE BUT SERVER DEAD 1767, reverse 0
missile-free residual:   1.0% (50/5085)
```

Unchanged from the first round (the `t_now`/wave fixes do not touch combat
timing). The missile confound (74.3% of ticks have one in flight; 11.5%
disagreement rate with one vs 1.0% without) explains most of it. The
remaining missile-free 1.0% has an existing, source-verified explanation this
project already documented rather than a newly-discovered bug: `inject.py`'s
own docstring names the auto-attack clock (`aa_cooldown`/`aa_windup`/
`is_attacking`) as **completely unobservable** from the dump and defaults it
to "not attacking, cooldown ready" every injected tick. `ObjAIBase.Update`'s
wind-up (`Spell.cs:541`) means a hit that lands on the server mid-swing was
started on an EARLIER, un-injected tick; the injector always resets the
clock, so the sim restarts a fresh wind-up and structurally cannot land that
same hit on this one injected tick. That is one-sided by construction (the
sim can never get AHEAD of the server's swing timing this way, only behind),
which matches the observed sign exactly. Not re-verified empirically this
round (lower priority per the coordinator's own ranking) — flagged as the
standing explanation, not a new finding.

## Gate verdict

**Not green**, and the reasons changed from round 1 to round 2:

- Round 1 said the injector was the blocker. It measurably was not (the
  bit-exact/≤1/16 gap collapsed from 18.4 to 2.9 points on recheck).
- Round 2's blocker is the position tail (11-14% of minion positions missing
  ≤1/16, p95≈5.4, max≈8.0), now localised to collision handling and further
  localised to ITERATION ORDER: a slot-order sequential reference measured
  WORSE than the current Jacobi approximation, but a RECONSTRUCTED
  creation-order reference measured BETTER than Jacobi at every crowding
  level (n=1: 81.3% vs 75.5%; n=2: 73.0% vs 70.1%; n=3+: 52.6% vs 48.7%) --
  confirming the sequencing mechanism is right and the true object-add order
  is the thing to recover properly, not a dead end. Inner (quadtree) order
  and a production implementation remain open, explicitly handed off (see
  below).
- Wave spawning went from "100% disagreement, unverified" to "a real,
  confirmed, partially-fixed timing bug (1,551/1,551 → 1,447/1,551), plus a
  fully-characterised (100% of remaining mismatches are >8 units from any
  plausible match) but not-yet-root-caused harness identity-matching
  artifact, now handed to a dedicated audit rather than pursued further here.
- Auto-attack fire-tick and target-selection exactness (both **EXACT**
  targets per plan §3) are not measured by this instrument at all on a
  sustained fight — the "single biggest hole" `inject.py` already names —
  and that gap is untouched by anything in this round.
- This is one trace. The 50-seed corpus gate 1 actually asks for does not
  exist yet.

## Open, in priority order

1. **Finish recovering the true collision iteration order, then decide a
   production design.** The outer (creation) order is confirmed both from
   source (`CollisionHandler.AddObject`/`GameObject.OnAdded`, `_objects`
   preserves survivor order under removal) and empirically (the
   reconstructed-order measurement above beats Jacobi at every bucket,
   including the outer-order-only n=1 test). What is not done, in order:
   (a) replace the arc-length proxy in `estimate_creation_order` with an
   exact whole-trace identity tracker (the proxy is why n=1 reached 81.3%,
   not the ~94% isolated units achieve); (b) resolve the map scene-file
   parse order for same-wave blue-vs-red tie-breaks (still an unverified
   guess); (c) recover the INNER (quadtree) order for n=2/n=3+ (28% of
   crowded samples) — the coordinator's proposed method is inverting the
   server's true per-tick permutation from the k! ways each k-neighbour
   collision could resolve, building a labelled corpus, and scoring
   candidate models (Morton/Z-order with a depth correction, nearest-first,
   farthest-first, x-then-y) with a held-out split, not in-sample; (d) a
   three-way production comparison -- current Jacobi, an exact
   `lax.scan`-based Gauss-Seidel, and a vectorised Morton-ordered
   approximation -- each measured on BOTH crowded-bucket accuracy and
   throughput/compile-time the way J1 gate 4/5 measured them (gate 4 has
   ~3x measured margin over its 50x target, which is headroom to spend on
   this, but it needs measuring, not assuming). **(a)-(d) are explicitly
   handed off, not started this round** — (b)-(d) are substantial
   engineering/benchmarking work the coordinator offered to route to a
   fresh agent rather than have it rushed; a sibling audit already owns the
   quadtree/`_objects` source ledger. `lanerl_jax/parity/
   tier1_collision_sequential.py` (in particular `estimate_creation_order`,
   the piece a real identity tracker replaces) is the ready-made harness for
   whoever picks this up, and it must never use a host-side/non-JIT
   per-tick Python callback in whatever production design is chosen — that
   option is off the table by explicit decision.
2. **The wave-spawn matching artifact** — fully characterised (100% of
   remaining mismatches have no plausible same-group match within 8 units)
   but not root-caused; handed to the dedicated waves/content audit per the
   coordinator. If it confirms identity-matching noise, fix
   `one_step.py`'s spawn-count scoring to use population deltas directly
   rather than individual identity-continuity.
3. **`LaneTurret.hp`'s 84 mismatches** — unmoved by the `t_now` fix, so not
   the ramp-timing bug; still unexplained.
4. **The auto-attack clock** — still the single biggest hole for the plan's
   EXACT fire-tick and target-selection targets; nothing in this round
   touches it.
5. **The 50-seed corpus** (J0) — every number in this document, both rounds,
   is n=1.
