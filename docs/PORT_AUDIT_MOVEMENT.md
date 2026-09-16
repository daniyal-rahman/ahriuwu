# Port audit: movement, pathing, waypoints, collision

Source-to-port ledger for the movement/collision subsystem, read against
`/srv/nfs/projects/lanerl-vendor/LoLServer` (the tree that actually builds
`GameServerConsole`, per `lanerl_train.paths.server_binary()`). `LoLServer/`
and `GameServer/` are byte-identical for every file cited here (checked with
`diff -rq`); `RL-Learning/` is a divergent older fork and is **not**
authoritative — it was not used for any citation below.

Method: every row below was produced by reading the cited server source and
the cited port source side by side. No row is based on measurement, and no
row assumes existing port code or comments are correct because they exist,
read confidently, or have a passing test — several of the findings below are
exactly a confident-sounding comment turning out to be wrong on inspection.
`EXACT` is only used where control flow, arithmetic order/rounding/clamping,
and update-ordering were all checked, not just the end formula.

Legend: **EXACT** (demonstrated equivalent, not just plausible) / **APPROX**
(deviation stated, reason given, notes whether the port documents it) /
**WRONG** (server does X, port does something observably different in a
reachable case) / **MISSING** (mechanic not implemented at all) / **N/A**
(mechanic is unreachable in this sim's configuration or is dead code on the
server itself — reason given) / **UNVERIFIED** (not checked to the EXACT bar;
plausible but not demonstrated).

---

## 0. Tick placement

| mechanic | server | ours | verdict | evidence |
|---|---|---|---|---|
| `Map.Update` runs collision before objects update | `Game.cs:481,483` (`Map.Update(diff)` then `ObjectManager.Update(diff)`); `MapScriptHandler.cs:96-99` (`CollisionHandler.Update(); PathingHandler.Update(diff); MapScript.Update(diff);`) | `sim/step.py:190-229` runs `resolve_collisions` as step 0, before buffs/movement/spawning | **EXACT** | Verified both citations directly; `step.py`'s own comment (`step.py:196-199`) cites the same lines and is correct. |
| `PathingHandler.Update` (3 s path-revalidation sweep) | `Handlers/PathingHandler.cs:42-58` (`Update(diff)`), calls `UpdatePaths` on everything in `_pathfinders` | not implemented | **N/A — dead code on the server itself.** `AddPathfinder` (`PathingHandler.cs:27`) is never called anywhere in `LoLServer/` or `LoLServer/Content` (`grep -rn AddPathfinder` finds only the definition). `_pathfinders` is permanently empty, so `UpdatePaths` (`PathingHandler.cs:64-90`) never executes on a real object. Nothing to port. (Noted for the record: `UpdatePaths` itself looks like it would rebuild `newPath` from `obj.Waypoints[0..]`, i.e. from the *whole* list including already-passed waypoints, not from `CurrentWaypointKey` onward, and then `SetWaypoints` unconditionally resets `CurrentWaypointKey = 1` — if this ever ran on a partially-walked path it would send the unit back toward waypoint 0. It doesn't matter because it never runs.) |
| `IsWalkable(pos, radius, checkObjects: true)` | `PathingHandler.cs:100-113` | not implemented | **N/A — dead code.** `grep -rn "IsWalkable(.*true)"` and `checkObjects` in `LoLServer/` find only the definition; every call site uses the default `checkObjects=false`. The object-checking branch never executes on the server either. |

---

## 1. Collision resolution — `CollisionHandler` / `AttackableUnit.OnCollision`

This is the highest-value part of the ledger; see the priority list at the
end. Ours: `sim/collision.py` (owned by another agent — audited, not edited).

### 1a. Iteration order — the load-bearing finding

**Claim in `sim/collision.py:78`:** *"lowest-index overlapping neighbour,
matching the server's iteration order."* This is **WRONG**, on two
independent grounds, and a sibling agent's empirical Tier-1 test already
shows a slot-order sequential re-implementation performing *worse* than the
current order-independent single push at every crowding level — consistent
with what the source predicts (see below).

**What `_objects` actually is (`Handlers/CollisionHandler.cs`):**
- `AddObject` (`CollisionHandler.cs:67-79`) does `_objects.Add(obj)`.
- Called from `GameObject.OnAdded()` (`GameObjects/GameObject.cs:119-123`),
  whose own doc-comment says "Called by ObjectManager after AddObject
  (usually right after instantiation)". So insertion order = **object
  creation order**: turrets/buildings at map load, champions at game start,
  minions per wave-spawn event, missiles/regions as cast, all interleaved in
  real creation-time order.
- Removal is `GameObject.OnRemoved()` (`GameObject.cs:156-159`) →
  `_objects.Remove(obj)`, a plain `List<T>.Remove` (find + shift-down). This
  **preserves the relative order of every surviving element** — no
  swap-with-last. So among currently-alive objects, `_objects` order is
  exactly their original relative creation order, permanently.
- Conclusion: `_objects` order is, in principle, reconstructible — but only
  via a persistent monotonic creation-sequence number per unit, independent
  of array position, that also reproduces the server's true spawn sequence
  (turrets/champions first, then minions in real per-wave order).

**Why array-slot index is not that, even approximately:** `sim/state.py`'s
own docstring (`state.py:59-66`) lays out the slot scheme
`[champions | minions | turrets]`, and explicitly says the ordering choice is
there to match a *different* mechanic — target-acquisition tie-breaking
(`ObjAIBase.cs:1295-1320`, verified below) — not collision. Two independent
problems with reusing it for collision:
1. Minion slots are **recycled**: `state.py:5-9`, "A dead minion keeps its
   slot with `alive=False`; a wave spawn writes into a free slot." Once any
   recycling happens, low array index means "whichever free slot the
   recycling policy picked," not "created earliest." It is uncorrelated with
   `_objects` order, not merely a stale approximation of it.
2. Turrets are the *last* slice of the array by construction, but were
   created *first* on the real server (map load, before champions/minions
   even exist) — so slot order and creation order disagree on turrets even
   before any recycling.

**The quadtree traversal order — a second, independent order, and a harder
one.** `GetNearestObjects` (`CollisionHandler.cs:96-116`) delegates to
`_quadDynamic.GetNodesInside`, `QuadTree/QuadTree.cs:139-150`. The
implementation's own doc-comment on the recursive walk
(`QuadTree.cs:325`, on `Quadrant.GetIntersectingNodes`, `QuadTree.cs:329-361`)
says: *"The nodes are returned in pretty much random order as far as the
caller is concerned."*

Mechanically: `GetIntersectingNodes` recurses into existing children in
fixed order **topLeft, topRight, bottomLeft, bottomRight** (only if a child
exists and its rect intersects the query circle, unless the query circle
already contains the whole quadrant, in which case all existing children are
visited unconditionally), and only *after* all four child subtrees are
exhausted does it append this quadrant's own "straddles more than one child"
node list (`QuadTree.cs:361`, calling the static
`GetIntersectingNodes(QuadNode, ...)` at `QuadTree.cs:370-383`, which walks a
circular linked list oldest-to-newest — insertion order, per
`Quadrant.Insert`'s tail-pointer bookkeeping at `QuadTree.cs:232-291`). Net
result: a DFS **post-order** (children-before-self, fixed TL/TR/BL/BR) over a
tree whose *shape* — which quadrants exist, how deep, who ends up "stuck" at
a straddling level — is decided by `Insert`'s `ContainedBy` test against
progressively-halved rectangles (with a `w<1 → w=1` floor), which depends on
the spatial layout of *every* collision object on the map at insertion time,
not just the querying unit and its immediate neighbours.

Timing: `UpdateQuadTree()` (`CollisionHandler.cs:161-172`) clears and fully
rebuilds the tree **once per tick, after** the `UpdateCollision` sweep, from
`_objects` in list order. So the tree a given tick's `GetNearestObjects`
query walks reflects everyone's positions as of the **end of the previous
tick's** collision pass — one tick stale relative to the live
`obj.IsCollidingWith(obj2)` test, which uses each object's *current*
`.Position` (possibly already displaced earlier in this same tick's
Gauss-Seidel sweep).

So the neighbour-visitation order for a given unit is a deterministic (not
literally random) but **jointly state-and-history-dependent** function of
(a) every collision object's position/radius map-wide as of last tick's
rebuild, and (b) `_objects` insertion order (which determines tree-build
insertion order this tick). It is neither `_objects` order, nor
distance-sorted, nor anything a fixed per-slot index can stand in for.

**Design decision for the human, stated plainly:** the `_objects` order is
reconstructible in principle (track true creation sequence, replicate real
spawn order) but the quadtree order is not practically reconstructible in
JAX — it requires bit-exactly re-deriving a recursive, dynamically-shaped,
pointer-linked structure with data-dependent recursion depth and per-node
linked-list length, once per tick, per environment. That is not
vectorizable/traceable the way the rest of this sim is built; a per-env
quadtree inside `scan`/`vmap` is not realistically buildable. **This bounds
achievable parity on any tick where a unit has 2+ simultaneous overlapping
neighbours** — "match exactly" is not achievable there without a host-side,
non-JIT reference implementation; the practical options are (i) accept the
Jacobi approximation and document the bound, (ii) special-case a slow
non-jitted quadtree path for offline reference generation only, not for
training. Recommend (i) for training and reserve (ii), if ever needed, for
generating ground truth to validate (i)'s error against.

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Which neighbour(s) get processed, outer sweep order | `_objects` = creation order (`GameObject.cs:119-123` `OnAdded`→`AddObject`; `CollisionHandler.cs:67-79`); survivors keep relative order across removal (`GameObject.cs:156-159`, `List<T>.Remove`) | `argmax` over fixed array-slot index (`collision.py:78-81`) | **WRONG**, comment overclaims match. Slot index ≠ `_objects` order once minion slots recycle (`state.py:5-9`), and turrets are slotted last but created first. Reconstructible in principle (track true creation sequence); JAX-hostile only in that it needs plumbing, not in kind. **Flag for the human**: worth doing if the outer-sweep order alone (holding neighbour order fixed) is shown to matter; unmeasured in isolation from the quadtree-order question. |
| Which neighbour is selected first *within* one unit's candidate set | Quadtree DFS post-order (`QuadTree.cs:325-383`), itself dependent on map-wide positions and insertion order, one tick stale (`CollisionHandler.cs:161-172`) | same `argmax` over array-slot index | **WRONG**/**MISSING**, not reconstructible in JAX as built (see above). **Flag for the human**: hard bound on ordering-parity, not a bug to fix. |
| Number of pushes applied to one unit in one tick | `UpdateCollision` loops **every** quadtree candidate and calls `OnCollision` for **each** one that currently overlaps (`CollisionHandler.cs:137-155`), each call immediately mutating `Position` via `SetPosition` (`AttackableUnit.cs:317`) — so a single unit can be displaced multiple times per tick, once per neighbour found, each one recomputed from the *already-displaced* position | exactly one push per unit per tick (`collision.py:87-89`) | **APPROX**, understated by the current docstring. The existing "Gauss-Seidel, one push per unit, from lowest-index neighbour" framing (`collision.py` module docstring) describes cross-unit propagation only; it misses that a *single* unit can be pushed multiple times within its own `UpdateCollision` call, against however many neighbours the (stale) quadtree query returns. JAX-hostile: data-dependent iteration count over a candidate list whose size varies per unit per tick. |

### 1b. Escape-point geometry

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Push magnitude (non-terrain case) | `AttackableUnit.cs:278-318`, `Extensions.GetCircleEscapePoint(Position, PathfindingRadius+1, collider.Position, collider.PathfindingRadius)` at `AttackableUnit.cs:312` | `r1 = pathfinding_radius+1`, `r2 = pathfinding_radius`, `push = dj - r1 - r2[j]` (`collision.py:66-67,87`) | **EXACT** (algebraically). Derived `GetCircleEscapePoint` by hand from `GetClosestCircleEdgePoint` (`GameServerCore/Extensions.cs:282-286`) and `GetCircleEscapePoint` (`Extensions.cs:353-359`): `edgepoint1 = p1 + r1·û`, `edgepoint2 = p2 − r2·û` where `û = normalize(p2−p1)`, so `exit = p1 + (edgepoint2 − edgepoint1) = p1 + û·(d − r1 − r2) = p2 − û·(r1+r2)`. That is exactly `collision.py`'s formula and exactly what its docstring derives. |
| Direction, floating-point path | `GetClosestCircleEdgePoint` computes the direction via `Atan2` then `Cos`/`Sin` (`Extensions.cs:282-286`), twice, once per edge point | `ux, uy = dx/d, dy/d` — direct division (`collision.py:85-86`) | **APPROX**, bit-level only. Mathematically identical, not bit-identical (trig round-trip vs. direct normalize accumulate different rounding). Negligible in practice (~1 ULP-scale); not worth fixing. |
| Degenerate case: exactly-coincident centres (`d = 0`) | `Atan2(0,0) = 0` by IEEE convention on **both** `GetClosestCircleEdgePoint` calls when `p1 == p2` exactly, giving `edgepoint1 = p1+(r1,0)`, `edgepoint2 = p2+(r2,0)`, so `exit = p1 + (r2−r1, 0)` — a **small, non-zero, deterministic push** of magnitude `|r2−r1|` along +X, *not* the `r1+r2` magnitude that applies for any `d>0`, however small (worked the ε→0⁺ limit by hand: it approaches `−(r1+r2)` along the approach direction, discontinuous with the exact-`d=0` case) | `overlap` requires `(d > 0)` (`collision.py:75`); an exactly-coincident pair is masked out of `overlap` entirely, so it is never pushed at all this tick | **MISSING** for this exact edge case. Low-probability for continuously-moving champions, but **plausible for wave-spawned minions**, which can spawn multiple units at literally the same coordinate in the same tick (spawn positions are shared per team) — worth a cheap fix (`d==0` fallback: push by `|r2−r1|` along a fixed axis, e.g. +X, matching the server's convention) if that scenario is confirmed reachable by whoever owns spawn placement. Left to the owning agent since `collision.py` is not mine to edit. |

### 1c. Who collides with whom (type gating)

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| `IsCollisionObject` (who can be a static/dynamic *obstacle* others push off of, i.e. quadtree membership) | `CollisionHandler.cs:39-45`: excludes only `IsToRemove`, `LevelProp`, `Particle`, `SpellMissile`, `Region`, and requires `CollisionRadius >= 0`. **Does NOT exclude `ObjBuilding` or `BaseTurret`.** | `collides = alive & (kind != TURRET) & (kind != NONE)` used symmetrically for both axes of the overlap matrix (`collision.py:62,73`) | **WRONG.** Turrets/buildings are valid `obj2` collision partners on the server — a unit that walks into a turret's `CollisionRadius` gets pushed out of it, same escape-point mechanism as unit-unit. Our matrix excludes turrets from *both* rows and columns, so turrets neither push nor are pushed — the "pushed" half (units bouncing off turret hitboxes) is silently dropped. |
| `IsCollisionAffected` (who initiates a check on *themselves*, i.e. gets `UpdateCollision` called and can move as a result) | `CollisionHandler.cs:52-58`: excludes `IsToRemove`, `LevelProp`, `Particle`, `ObjBuilding`, `BaseTurret` | same `collides` mask used for the row axis too (`collision.py:73`) | **Correct half only.** Turrets/buildings correctly never move from collision (row exclusion is right); the bug is specifically that they're *also* excluded as static obstacles (column exclusion is wrong). The module docstring's claim "`IsCollisionAffected` excludes ... `BaseTurret`, so turrets neither push nor are pushed" (`collision.py` module docstring, "Who collides" section) is **WRONG**: `IsCollisionAffected` only governs whether a turret is pushed, not whether it pushes; that's `IsCollisionObject`, a materially different predicate that does *not* exclude turrets. |
| Ghosted / dashing gate | `AttackableUnit.cs:299-305`: skip if **either** party has `MovementParameters != null` (dashing) or `Status.HasFlag(Ghosted)` | `collides = collides & ~ghosted` (only Garen's E is modeled, `step.py:227`), ANDed across both matrix axes (`collision.py:73`) | **EXACT for what's modeled.** The AND-across-both-axes structure correctly drops a pair if either side is excluded. Dashing is **N/A** — the only champion in this sim's roster (`profiles.py:38-39`, singular `patch.champion`) is Garen, whose kit (Q/W/E/R = Decisive Strike / Courage / Judgment / Demacian Justice) has no dash, so `MovementParameters` is never set and `DashMove`/`Dash()` (`AttackableUnit.cs:867-919`, `:1520-1619`, not read in full since unreachable) never executes for any modeled unit. `pre_ghosted` correctly reads incoming (pre-`step_buffs`) buff state, matching the server's ordering where `UpdateBuffs` runs after `Map.Update`/collision (`step.py:221-227`, verified against tick order in section 0). |
| Detection radius vs. push radius | `IsCollidingWith` (`GameObject.cs:226-229`), the test that gates whether `OnCollision` fires at all, uses **`CollisionRadius`**: `DistanceSquared < (CollisionRadius+CollisionRadius)²`. The push magnitude (`AttackableUnit.cs:312`) uses **`PathfindingRadius`**, a different field. | `touching = r1+r2` built entirely from `pathfinding_radius` (`collision.py:66-67,71`), used for *both* the overlap test and the push | **APPROX**, real but low-impact. Confirmed the two radii differ in the actual data: Garen's `GameplayCollisionRadius` is `-1` → falls back to `CollisionRadius=40` (`ObjAIBase.cs:104-115`, matches `data/patch.py:211-217`'s comment exactly), while his `PathfindingCollisionRadius=35` (`data/navgrid.py:102`). So two Garens: server starts reacting to overlap at 80 units apart (`40+40`) but always resolves to exactly 71 apart (`36+35`); ours starts reacting at 71. Because the push target is always the `PathfindingRadius`-sum distance regardless of which radius triggered it, the **steady-state separation is identical** either way — the only observable difference is the tick at which the very first correction fires, roughly a 9-unit / ~1.5-tick window. Not worth prioritizing; noted for completeness since the strict bar requires citing it. |

---

## 2. Movement — `AttackableUnit.Move` / `GetMoveSpeed`

Ours: `sim/movement_jax.py` (the live path, wired via `step.py:296-299`) and
`sim/movement.py` (numpy reference, used only by `parity/movement_parity.py`
and `parity/tests/test_navgrid.py` — not on the training path).

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Gate: movement only runs if `CanMove()` and path not ended | `AttackableUnit.cs:237,253-261`: `Update` calls `Move(diff)` only inside `if (CanMove())`; `Move` itself (`:931-956`) first checks `if (CurrentWaypointKey < Waypoints.Count)` | `active0 = can_move & (k < n)` (`movement_jax.py:87-89`), `can_move` passed in from `step.py:299` via `_can_move` (`step.py:151-166`) | **EXACT for the `Move` integrator's own structure** (checked: same two-part gate, same order — `CanMove` outside, waypoint-count check first inside). `_can_move`'s definition of *what CanMove() is* is a separate, narrower row below. |
| Per-tick waypoint-consuming loop | `AttackableUnit.cs:931-956` `while(true)` loop; unbounded, data-dependent iteration count | `jax.lax.scan` over `MAX_STEPS_PER_TICK=8` (`movement_jax.py:60,90-121`), each iteration masked | **EXACT control flow, bounded by construction (documented JAX-hostility).** Checked comparison direction (`budget < dist` ↔ `maxDist < dist`, both strict-less), the "reach" branch's exact-equality checks (`nbudget != 0` ↔ `maxDist == 0`, both exact float compares, same subtraction order), and the freeze-once-inactive behavior (masked branches become no-ops once `still` goes false, matching `return`). The bound is empirically checked against 256 random synthetic paths, 11 waypoints, speeds 300–700 u/s (`sim/tests/test_movement_jax.py:114-128`, `test_the_step_bound_is_large_enough_for_random_paths`) — passes today, but the corpus is synthetic/uniform-random, not drawn from the server's own `SmoothPath` output, so it may not stress the near-coincident-waypoint case that motivates the bound in the first place. `sim/movement.py`'s docstring still calls the bound "provisional" (`movement.py:56-59`) even though `movement_jax.py`'s is asserted; the two modules' framing of confidence disagrees and should be reconciled by whoever owns them. |
| `GetMoveSpeed()` while not dashing | `AttackableUnit.cs:697-702`: returns `Stats.GetTrueMoveSpeed()` | `P("move_speed")`, a static per-profile constant from Content (`profiles.py:122`, `data/patch.py:198`) | **WRONG**, confirmed concretely reachable. `Stats.GetTrueMoveSpeed()` (`GameObjects/Stats/Stats.cs:231-233`) returns a value built by `CalculateTrueMoveSpeed()` (`Stats.cs:371-395`): base+flat, three soft-cap bands (`>490`, `415–490`, `<220`), then `×(1+PercentBonus)×(1+MultiplicativeSpeedBonus)`, then the single largest active slow. None of this is modeled; `move_speed` never varies at runtime. For *this* roster (Garen-mirror, no items/boots modeled) the concrete, currently-reachable instance is **Garen's Q haste**: `GarenQHaste.cs:34` sets `StatsModifier.MoveSpeed.PercentBonus += 0.35f` (a genuine +35% MS window). `spells.py` tracks the buff's on/off state faithfully (`BuffId.GAREN_Q_HASTE`, `qh_active`/`qh_elapsed`/`qh_expired` at `spells.py:664-668`) but `BuffResult` (`spells.py:570-580`) has **no** move-speed field at all — only `armor_pct_bonus`/`mr_pct_bonus` are threaded through to `step.py`'s `armor_eff`/`magic_resist_eff` (`step.py:169-176`). The buff's combat-relevant (silence/bonus-AD) half may be handled elsewhere, but its **movement** half is completely unwired. This directly affects kiting/chase/disengage dynamics in an all-in, which is exactly the kind of interaction a lane RL policy would need to learn correctly. |
| `CanMove()`, full definition | `ObjAIBase.cs:302-314`: `(!IsDead && MovementParameters != null) || (CanMove-flag && CanMoveEver-flag && MoveOrder not CastSpell && no casting spell && (no channel OR channel allows move) && (not attacking OR attack cancelable) && not (Netted\|Rooted\|Sleep\|Stunned\|Suppressed))` | `_can_move`: `alive & ~(move_order ∈ {CAST_SPELL, NONE, STOP, HOLD})` (`step.py:151-166`) | **APPROX, practically N/A for this roster.** The CC-status half (`Netted/Rooted/Sleep/Stunned/Suppressed`) is entirely absent, but nothing in a Garen-vs-Garen match with no items/summoners sets any of those five flags on either unit (Garen's kit has no CC on self or ally, and the mirror opponent is the same kit), so the simplification is currently sound. **Flag for the human**: this is load-bearing the moment any CC-capable unit is ever added to the roster; it is silent until then. |
| `CanChangeWaypoints()` gating a mid-cast re-target | `ObjAIBase.cs:317-322`: `!IsDead && (no MovementParameters OR following) && _castingSpell == null && (no channel OR channel cancelable)` — a unit mid-cast cannot have `SetWaypoints` succeed at all | `step.py`'s chase block (`step.py:422-432`) unconditionally overwrites `waypoints`/`waypoint_key`/`n_waypoints` whenever `chase` is true, with no cast-state gate | **APPROX**, real but likely low-impact. During a spell windup the server freezes the *target* waypoint (not just current motion — `Move` is already separately blocked by `CanMove`'s `_castingSpell == null` term), so if the chase target moves during the cast, the server keeps chasing the *old* fixed point until the cast ends and `RefreshWaypoints` runs again; ours updates the waypoint target immediately underneath the frozen position. Bounded by cast/windup duration (short); unmeasured. |
| `RefreshWaypoints` (chase re-pathing) | `ObjAIBase.cs:595-666`: for `MoveOrder==AttackTo`, if out of `idealRange` calls `_game.Map.PathingHandler.GetPath(Position, targetPos, PathfindingRadius)` (`:660`) — the **full A\*** (`NavigationGrid.cs:181-313`: priority-queue search over grid cells with a `CastCircle` line-of-sight closed-list rule) **plus `SmoothPath`** (`NavigationGrid.cs:316-333`, Theta*-style collapse of cells with mutual LOS) | direct 2-point line `[position, target position]` (`step.py:414-432`) | **APPROX, already honestly documented and independently confirmed correct as a characterization.** `step.py:417-421`'s own comment calls this a "BOOKED APPROXIMATION," cites the measured "76% of paths under 500 units are already straight," and says "chasing across terrain is where this is wrong, and it is unmeasured." I confirmed by reading `RefreshWaypoints`/`GetPath` directly: the server's chase path *is* genuinely A*+smoothing, not a line, and in fully open terrain with mutual line-of-sight `SmoothPath` will typically collapse it back to two points — so the approximation is exact in open lane and wrong near terrain features (walls, tri-bush, turret hitbox corners), exactly as documented. This is the one instance in the subsystem where an existing approximation-comment survives scrutiny; contrast with the collision-order comment above, which does not. |
| `idealRange` formula for the chase/hold boundary | `ObjAIBase.cs:1233`: `idealRange = Stats.Range.Total + TargetUnit.CollisionRadius` (attacker's own radius excluded); comparison is `DistanceSquared(...) <= idealRange²` (`ObjAIBase.cs:1198-1200,1236`) | `ideal = P("attack_range") + P("collision_radius")[tgt]`; `in_rng = d2 <= ideal*ideal` (`step.py:420-425`) | **EXACT.** Formula, operand choice (target's radius, not attacker's), and comparison operator (`<=`, not `<`) all verified against source. |
| Terrain-adjacent target clamp before pathing | `ObjAIBase.cs:658-659`: if `!IsWalkable(targetPos, PathfindingRadius)`, replace `targetPos` with `GetClosestTerrainExit(targetPos, PathfindingRadius)` before calling `GetPath` | not implemented (no terrain awareness anywhere on the live path) | **MISSING**, subsumed by the `RefreshWaypoints` row above — grouped here because it's a separate, citable branch. |

---

## 3. Terrain / navigation grid

Ours: `data/extract_navgrid.py` produces a static walkable-cell array; grepped
the entire `sim/` tree and confirmed **nothing on the live path
(`step.py`, `movement_jax.py`, `collision.py`) imports or consults it** — the
only consumer is the non-live `sim/movement.py`. So every terrain-interaction
mechanic below is simply absent at runtime, not approximated.

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Walkability flag test | `NavigationGrid.cs:493-497`: `!NOT_PASSABLE && !SEE_THROUGH` (both flags must be clear) | `walk = (flags & NOT_PASSABLE == 0) & (flags & SEE_THROUGH == 0)` (`extract_navgrid.py:105-108`) | **EXACT.** Boolean expression is identical; the module's own "over-counts by 516 cells" framing reads like an empirical discovery but the logic it lands on is a direct, correct transcription of the source condition. |
| Terrain-collision push (`UpdateCollision`'s first branch: `if (!IsWalkable(obj.Position)) obj.OnCollision(null, true)`) | `CollisionHandler.cs:139-143`; escape via `GetClosestTerrainExit(Position, PathfindingRadius+1)` (`AttackableUnit.cs:296`), skipped entirely while dashing (`AttackableUnit.cs:290-294`, N/A here — no dashes) | not implemented | **MISSING.** No unit-vs-terrain collision check exists anywhere in the live sim. Low frequency in a well-behaved open-lane scenario (waypoints are precomputed on walkable ground), but real for wave clumping against lane walls/tri-bush geometry and for any future terrain-adjacent scenario. |
| Terrain-safety fallback nested in the *object*-collision branch (`if (!IsWalkable(exit, PathfindingRadius)) exit = GetClosestTerrainExit(exit, PathfindingRadius+1)`) | `AttackableUnit.cs:313-316` | not implemented | **MISSING.** A unit-unit push that would land inside terrain is not corrected; the raw geometric escape point is used unconditionally (`collision.py:87-89`). |
| `GetClosestTerrainExit` (outward spiral search) | `NavigationGrid.cs:879-894`: unbounded `for` loop, Archimedean spiral (`r` grows every step, angle `+45°` each step) until `IsWalkable` | not implemented | **MISSING** (component of the two rows above). Also independently JAX-hostile if ever ported: unbounded, data-dependent iteration count, same category as the `Move()` loop but with no natural bound analogous to "waypoints per tick." |
| Minion lane-path construction, direction reversal for the second team | `Maps/Map1/LevelScript.cs:283-304`, `SetUpLaneMinion()`: `waypoint.Reverse()` for the non-blue team | `step.py` passes `lane_path[::-1]` for `Team.RED` (`step.py:245`) | **UNVERIFIED to the EXACT bar** — confirmed the server calls `.Reverse()` on the same waypoint list at the cited line (spot-checked only this one call site; did not read the full spawn-placement subsystem, which is out of this audit's scope and the wave-spawn owner's territory). Reversing a list is the obvious match for `.Reverse()`; flagging as unverified rather than EXACT purely because I did not trace the surrounding index/offset arithmetic in `LevelScript.cs`. |
| `TeleportTo` / `SetPosition` repath semantics after a Recall/Flash-style jump | `AttackableUnit.cs:840-858` (`TeleportTo`, clamps through `GetClosestTerrainExit` first), `AttackableUnit.cs:195-233` (`SetPosition`'s `repath` branch: either repaths via `GetPath`, calls `ResetWaypoints()`, or just patches `Waypoints[0]`) | not implemented; only `AttackableUnit.OnCollision`'s two `SetPosition(exit, false)` call sites are ported (as raw `x,y` writes with waypoints untouched) | **N/A for `TeleportTo` specifically** — nothing in this sim's modeled kit (Garen, no summoners modeled) ever calls it. The `SetPosition(_, false)` → `Waypoints[0] = Position` branch used by collision pushes is **inert in both implementations**: `CurrentWaypointKey` is always ≥1 once a path is active (`ResetWaypoints` sets it to 1, `SetWaypoints` sets it to 1), so `Waypoints[0]` is never read again by `Move()` (`CurrentWaypoint` = `Waypoints[CurrentWaypointKey]`), and our chase logic always rebuilds its 2-point waypoint list from the *current* `x,y` fresh each tick (`step.py:428-431`) rather than depending on a stored `Waypoints[0]`. Confirmed this makes the omission harmless for the paths this sim actually exercises. |
| `Champion.Respawn()` position reset | `Champion.cs:285-299`: calls `SetPosition(spawnPos)` with the **default** `repath=true` — but `IsDead` is still `true` at that exact point in the method (`IsDead = false` is set several lines *later*, `Champion.cs:296`), so `CanChangeWaypoints()` (`ObjAIBase.cs:317`, requires `!IsDead`) is false at that instant, so if the champion's path was unfinished at time of death, the internal repath's `SetWaypoints(safePath)` call **silently fails** and stale pre-death waypoints survive the respawn teleport | `x, y = spawn_x, spawn_y` on `reborn` (`step.py:641-643`); `waypoints`/`waypoint_key`/`target` are left as whatever this tick's (pre-respawn) targeting pass computed | **UNVERIFIED**, flagged rather than scored. This is a genuinely surprising, low-frequency server quirk found only by reading `Respawn()` line-by-line (a champion that dies mid-walk can, on paper, resume walking from the fountain toward its stale pre-death destination until the next order overwrites it). Whether our port's actual behavior converges to the same outcome depends on whether `target`/`chase` are also frozen correctly for a dead unit elsewhere in `step.py`'s targeting section, which is outside this audit's subsystem (combat/targeting, not movement) and was not traced to the EXACT bar. Reporting the server-side mechanism precisely so the targeting/combat owner can check it against their own code. |

---

## Priority list (WRONG / MISSING, ranked by plausible effect on lane behaviour)

1. **Collision iteration order (§1a).** Neither `_objects` order nor the
   quadtree order is reproduced by array-slot index; the quadtree order is
   not reproducible in JAX at all as this sim is built. This is the
   highest-value finding in the audit — it's already shown empirically (by
   the sibling agent's Tier-1 differential) to change which approximation is
   *better*, not just by how much. Needs a human decision on whether to
   invest in reconstructing `_objects` order (feasible) and/or accept a
   documented bound on ordering-parity for the quadtree half (likely
   unavoidable).
2. **Turret/building pushback (§1c).** Units currently pass through turret
   hitboxes with no collision response at all, in either direction. Directly
   affects any lane behaviour near a turret (turret-dive positioning, wave
   crashing into tower aggro range).
3. **`GetMoveSpeed()` has no dynamic component (§2).** Concretely: Garen's Q
   haste (+35% MS, `GarenQHaste.cs:34`) is tracked as buff-state bookkeeping
   in `spells.py` but never reaches movement. This directly changes
   engage/disengage/kiting distance calculations in any fight involving Q,
   which is a core part of Garen's kit.
4. **Missing terrain collision entirely (§3).** No unit-vs-terrain push, no
   terrain-safety fallback on unit-unit pushes. Low frequency in open lane,
   real near lane-wall/bush geometry and under wave clumping.
5. **Escape-point degenerate case at exact position coincidence (§1b).**
   Narrow, but plausibly reachable at wave spawn (same-team minions can spawn
   at literally the same coordinate). Cheap to fix if confirmed reachable.
6. **CanChangeWaypoints not gating mid-cast re-targets (§2).** Real but
   short-duration (bounded by cast/windup length); lowest priority of the
   WRONG/MISSING rows because its worst-case effect is small and transient.

Everything else above is either **EXACT** (verified to the strict bar:
`Move()`'s core loop, the `idealRange` formula, the walkability flag test,
tick placement), **N/A** (confirmed dead code on the server itself —
`PathingHandler.UpdatePaths`, `IsWalkable(checkObjects=true)` — or unreachable
in this sim's roster — dashes, `TeleportTo`), or **APPROX** with the
deviation already honestly measured and documented in the port's own
comments (`RefreshWaypoints` → straight-line chase).
