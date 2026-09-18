# JAX simulator fidelity ledger

**Purpose.** This is the durable, central record of behavior in the JAX
training simulator that is not proven identical to the vendored C# server.
Read it before interpreting an RL failure as a learning failure.  A local code
comment or an audit can contain the detailed evidence, but every behavior that
can change trajectories, observations, rewards, or legal actions must also have
an entry here.

This document is a ledger, not a backlog.  Entries remain after they are fixed;
change their status to `VERIFIED` and add the test/evidence.  New approximations
must be entered in the same commit that introduces them.

## Status vocabulary

| status | meaning |
|---|---|
| `MISSING` | Server behavior has no implementation. |
| `APPROX` | Implemented deliberately, but not server-equivalent in at least one known case. |
| `UNOBSERVABLE` | The reset/parity input does not expose enough server state to reconstruct it exactly. |
| `BOUNDED` | Equivalent inside an explicit fixed-shape bound; exceeding the bound is a diagnostic failure. |
| `VERIFIED` | Source-port and tests or differential evidence support parity for the stated scope. |

## Phase 1 gate dashboard (canonical results, 2026-09-18)

This is the short authoritative answer to “is Phase 1 done?” Detailed history
remains in `JAX_REWRITE_PLAN.md`, but stale exploratory numbers must not replace
the canonical rows below.

| gate | status | canonical evidence | remaining work |
|---|---|---|---|
| 1. whole-corpus one-step parity | **OPEN** | Stable NetIds eliminate false identity events. Correcting fractional barracks spawn coordinates makes the 300-pair raw-float pre-clash corpus 2,170/2,170 for position, HP, move order, waypoints, deaths, and spawns. The last trustworthy full exact-cache run before that fix still had minion HP 99.91%, move-order 97.40%, and waypoint-count 99.95%, so those exact gates are not silently treated as green. On a fresh fixture whose Moves are actual legal 96x54 non-minimap bin centres, all 20 lookups are READY, but waypoint count agrees 0/20 (server 3; local 6–9); the host server-algorithm port agrees 20/20. The injector restores diagnostic target/AA state, but the aggregate one-step report does not yet score post-step target identity or AA fire tick explicitly. | Re-run the full exact-cache corpus after the spawn fix and resolve its remaining HP/move/waypoint disagreements; add explicit exact target/AA-fire comparisons; make local routing reproduce server A* + `SmoothPath` waypoint emission. |
| 2. free-running divergence characterized | **PASS** | Four 600 s scenarios are recorded in `TIER2_DIVERGENCE.md`; divergence begins around wave interaction and is explicitly large. | None for Phase 1; retain as a regression corpus. |
| 3. oracle last-hitter CS@10 | **OPEN** | First **routed** measurement, 2026-09-18, 18,000 decisions: sim CS=3 / attacks=54 / approach=6,400 / deaths=2 against server CS=4 / attacks=86 / approach=3,197 / deaths=1. The gap is **-1**, down from +3 on the raw path and from the obsolete 13-vs-4. **Do not read the smaller number as convergence.** The sim spent 6,400 of its 18,000 decisions walking in against the server's 3,197, so it farmed roughly 21% less of the episode, and it died twice to the server's once. The CS figures moved closer while the thing being compared got further apart. The raw-path 7-vs-4 run is not canonical and the raw figures must not be quoted as gate evidence. | Decompose the 6,400 with the new per-walk `walks` field: "walks slower", "walks more often" and "dies mid-walk and restarts" have different fixes and the total cannot tell them apart. Re-run `isolation.py` now that it is routed (it was diagnosing the raw path while the gate ran routed) and now that sequential collision has landed. Attribute the first free-running target/AA/HP split. |
| 4. full-loop throughput >=56k decisions/s | **OPEN** | RTX 5080, canonical command (see above): 4,096 envs, 150 s warm-up, **44 live entities**, 60 timed + 5 warmup. Three runs of the identical configuration on 2026-09-18: **55,054 / 55,565 / 55,049 dec/s**; sim-only 109,024-109,047. Target 56,450. Short by **1.6-2.5%**, against a run-to-run spread of about 1%. The previously recorded 53,228 / 53,605 figures are **stale** -- the gap roughly halved with the committed routing work. No-route control 57,378 (not gate evidence). | The remaining cost is not loop control: sweeping `ROUTE_LOOP_UNROLL` over 8/16/24/4 moved nothing outside noise, so chaining more masked hop bodies is not the lever. That points at the table gathers themselves (a 231 MiB hop table) rather than XLA loop overhead. Next: measure where the ~5.5 ms routed delta goes before optimising anything, and state a repeat count and statistic for the gate, since one run cannot resolve a 2% gap at 1% noise. |
| 5. compile under two minutes | **PASS** | Routed compile 20.9-21.4 s across the three 2026-09-18 runs (earlier record: 12.9-13.6 s; it has grown with the routing work but remains far inside the 120 s gate). | None. Watch it: it moved 60% without anyone noticing, which a gate with this much headroom will not catch. |
| 6. reset cost small | **PASS** | 2.17% of a step at 512 envs and 1.74% at 2,048. | None. |

Phase 1 is therefore **not complete**: gates 1, 3, and 4 remain open.

## Canonical commands

**Read this before quoting any number below.** Every figure in the dashboard
came from one of these commands. A number produced any other way is a
diagnostic, not gate evidence, and must say so where it is written down.

This section exists because its absence is what cost the most: the routed
53,605 dec/s gate-4 figure came from a script that was never committed, so it
went stale without anyone noticing, and `benchmark.py`'s own `__main__` could
only run the no-route control. Two `slurm/*.sbatch` files pointed at worktrees
that had been deleted. `lanerl_jax/tests/test_canonical_commands.py` is the
mechanical guard against that rotting again; it runs in 0.15 s.

| gate | command | where |
|---|---|---|
| 1 | `sbatch slurm/parity.sbatch python -m lanerl_jax.parity.tier1_full` | `desktop`, CPU |
| 2 | `sbatch slurm/parity.sbatch python -m lanerl_jax.parity.tier2_batch --engine server` then `--engine sim`, then `python -m lanerl_jax.parity.tier2 compare --a ... --b ... --out ...` | `desktop`, CPU |
| 3 | `pytest lanerl_jax/parity/tests/test_last_hit_gate.py::test_oracle_scores_the_same_cs_in_sim_and_server` | boots a real server; several minutes |
| 4, 5 | `srun -p gpup --gres=gpu:1 --chdir=/mnt/nfs/projects/ahriuwu-lanerl-jax bash -lc '.venv-gpu/bin/python -m lanerl_jax.train.benchmark --envs 4096'` | **`desktop` only** — the 5080 is the gate hardware |
| 6 | same, with `--envs 512 2048 --reset-bench` | `desktop` |

Gate-4 rules that are part of the measurement, not preferences:

* **Routing is the default and is the gate.** `--no-route-table` prints
  "NOT gate evidence" in its own header. Do not quote it as a result.
* **The initial state must be warm.** `--warm-s 150` is what produces the live
  44-entity minion-bearing state; the run prints its entity count. A cold
  `init_lane` has no minions, and a 59,153 dec/s figure was once discarded for
  exactly that.
* **60 timed steps after 5 warmups.** Short 20/30-step samples above 56k were
  measured to be unstable and discarded.
* **Repeat it.** Three runs of the identical canonical configuration on
  2026-09-18 gave 55,054 / 55,565 / 55,049 dec/s. Run-to-run spread is about
  1%, which is comparable to the remaining gap, so a single run cannot settle
  the gate in either direction.

## Active fidelity entries

| ID | status | subsystem | server behavior | JAX behavior / deviation | likely symptom | evidence and resolution trigger |
|---|---|---|---|---|---|---|
| PATH-001 | `APPROX` | champion move routing | `NavigationGrid.GetPath` runs radius-aware, closed-on-enqueue A* from the champion's exact float position to the exact float click, then `SmoothPath`. | Production `run_train` loads the radius-aware local table and reconstructs a static-terrain route. Its deterministic reverse BFS and collinear compression are robust but not the server's A* tie-break and `SmoothPath`. A caller omitting the table still gets `[position, destination]`, explicitly exposed as `--no-route-table`. On the fresh 20-Move fixture generated from actual legal 96x54 non-minimap bin centres, every lookup is READY but local waypoint counts agree 0/20: server always 3, local 6–9. The host server-algorithm port agrees with the server 20/20. | A different safe side/curve around terrain changes arrival and trade timing; disabled routing causes wall entry/ejection. | `data/local_route_artifact.py`, `sim/local_pathing.py`, exact host reference in `data/navgrid.py`. A bounded device `CastCircle` now agrees with the host on the 20-click corpus (10 clear, 10 blocked, zero bound exhaustion); it still needs `SmoothPath` integration. Earlier route decomposition independently showed both missing LOS smoothing and BFS/A* path choice. |
| PATH-002 | `APPROX` | policy click geometry / HUD | The wire accepts a world destination, but the intended control surface is the locked follow-camera viewport with minimap clicks forbidden. | The 96x54 bin centres now use the same calibrated perspective equations and side-canonical lane reflection as `lanerl_rl.projection` / the real environment. Bins in the measured 352px minimap rectangle (`x>=275`, `y>=240`) decode semantic Move to NOOP, preventing a live global minimap order. The parity recorder also selects the nearest point from this exact legal grid instead of using an artificial radius clamp. Because x/y heads are factored, the joint rectangle cannot be removed from both marginal distributions; invalid pairs can still be sampled and wasted as NOOP. Other HUD regions are not yet jointly masked. | Correct local geometry, but some sampled actions are wasted; an inaccurate HUD rectangle could suppress valid floor clicks or permit HUD clicks. | `train/actions.py`, `parity/record.py`, their focused tests, and `INFERENCE_FAILURE_ANALYSIS.md` M11/H100. Resolve the remaining approximation with a joint/autoregressive spatial head or a full conditional 2-D action mask calibrated to the deployed client. |
| PATH-003 | `APPROX` | local route endpoint quantization | A* priorities and the first/final swept-circle edges use exact fractional source and goal coordinates. | The local artifact selects raw hops by source/goal cells; runtime preserves the exact terrain-projected float destination but the first branch is cell-centre-derived. Two positions in the same 50-unit cell can therefore choose the same branch when the server would not. | Rare different side around a corner, then large accumulated position divergence. | `data/local_route_artifact.py`, `data/route_artifact.py`; resolve with a server waypoint corpus and endpoint bins or an exact bounded runtime search. |
| PATH-004 | `BOUNDED` | waypoint storage | Server waypoint lists are dynamic and its `SmoothPath` corpus measured a maximum of 19 for 1800-unit clicks. | State stores at most 64 waypoints and reconstruction at most 128 raw hops. The local reverse-BFS router only removes collinear cells: among 110,890 random valid bounded Map1 routes, p99 was 24 and the maxima were 45 waypoints / 101 raw hops. Both overflows remain explicit statuses. | An exceeded bound uses the labelled two-point fallback rather than silently stopping early. | `sim/state.py`, `sim/local_pathing.py`; retain adversarial/random bound tests and raise the cap if either overflow status occurs in rollout diagnostics. |
| PATH-005 | `APPROX` | route coverage/fallback | Server attempts A* for every legal world click and uses its raw two-point fallback only if `GetPath` returns null. | The production Map1/Garen v2 artifact covers 47,477 radius-walkable source cells and +/-50-cell goal offsets in a 231.0 MiB packed-uint4 hop table. An optional v3 sidecar adds 462 MiB of uint8 same-direction runs and jumps only across collinear raw hops, retaining exact existing turns, raw-hop overflow accounting, and statuses; it is not the production default pending a throughput win. Fractional endpoints are anchored to the nearest covered centre within 3 cells while the exact float goal is retained. Table-disabled, source/goal unanchored, goal-outside-window, graph-no-route, raw-hop overflow, and waypoint overflow are distinct statuses; all use a two-point fallback and persist in `LaneState.route_status`. Training reports `route_nonready`. On the exact projected action lattice, the prior +/-44 artifact routed 98,433/115,848 unmasked spawn/lane-reference clicks (85.0%); only 37 exceeded its window, and most other non-ready cases matched Map1 edge/null-path quirks. The maximum anchored endpoint delta was 49 cells, setting the production +/-50 bound. | Terrain behavior can revert to the approximation at coverage/window boundaries; nearest-centre anchoring can pick a different first/final edge from the server. | Production artifact manifest plus `sim/local_pathing.py`; drive `route_nonready` to the server's own null-path cases on representative rollouts and inspect every nonzero class rather than averaging it away. |
| PATH-006 | `APPROX` | Gate-3 scripted drivers | Server-driven parity trajectories path every Move through `GetPath`. | `last_hit_drive`, `hp_band` and now `isolation` all load production `map1_garen_r35_o50_v2` by default; `table_disabled=True` / `--table-disabled` is the explicitly named two-point isolation control. `isolation.py` was the outstanding one and it mattered most: it is the module the gate-3 death gap was handed off to, and it drove the sim on the raw path while the gate itself ran routed, so its measured tail effect described a run that no longer happened. Its 2026-09-16 table (1.7x/1.9x isolation tails) was stale on two counts -- raw path, and measured before sequential collision landed -- and has now been **re-measured with both sides re-run the same day**: tails 1.23x/1.40x, mean nearest-ally +31%, and no-ally-alive-anywhere 11.4% sim against 10.2% server, which rules out wave survival. The medians no longer agree (395.6 vs 305.9, +29%), so that module's headline reading -- "typical position is right, only the tail is wrong" -- does not hold under the mode the gate runs; notably it was the RAW path whose median agreed. A 2,100-decision approach check was raw=1,190 versus routed=1,378 decisions (+6.27 s), so the control cannot be silently substituted for the gate. The local artifact still has the PATH-001/003 approximation, not server-exact A*. | Route choice/timing can still differ from the server; raw control results must not be called canonical, and a raw-path diagnostic cannot explain a routed gate. | Re-run routed `isolation.py` and replace its stale table. Retain raw only for causal isolations. The per-walk `walks` field on `SimRun`/`ServerRun` is what makes the approach cost attributable; use it rather than the `approach_decisions` total. |
| COLL-001 | `VERIFIED` | dynamic unit blocking | Minions/champions collide dynamically through `CollisionHandler`; they are not obstacles in terrain A*. Collision handling is sequential and can re-project an escape from terrain. | Dynamic bodies are handled in `sim/collision.py`, separate from static path selection. Garen E ghosting and frozen per-pass collision cache are represented. A champion-versus-two-minion test pins that clear static terrain still produces creep block with the server's trigger/resolution radii. | If regressed: wrong creep block, phasing, or push direction. | `sim/tests/test_collision.py::test_creep_block_is_dynamic_collision_not_a_static_route_obstacle` plus the collision source audit. Add a longer multi-minion choke/trading-pattern trajectory corpus before calling gameplay parity complete. |
| COLL-002 | `BOUNDED` | collision passes | Server collision loops are data-dependent. | Fixed pass/iteration caps are used for JAX. Exhaustion is intended to remain observable. | Dense packs can retain overlap or diverge by order. | `sim/collision.py`; measure maximum on adversarial creep packs and raise bounds rather than accepting exhaustion. |
| MOVE-001 | `BOUNDED` | waypoint consumption per tick | Server consumes waypoints until movement budget is exhausted. | At most 8 waypoint transitions are consumed per tick. | High speed or near-coincident points can end a tick short. | `sim/movement_jax.py`; `max_steps_used` and parity tests. |
| OBS-001 | `APPROX` | fog / brush visibility | Server visibility includes its complete vision, brush, and reveal rules. | The lane observation uses the simplified implementation in `obs/fog.py`; it is not a complete server vision system. | Target/action availability or enemy memory changes on brush edges. | `obs/fog.py` and observation tests. Resolve with server visibility traces around every top-lane brush boundary. |
| RESET-001 | `APPROX` | target and minion-AI internals | The canonical hash/dump intentionally omit target incumbency and private minion-AI maps. | `LANERL_STATE_DUMP_INTERNALS=1` emits diagnostic-only target NetIds, AI clocks, ignore/help maps, and the lane waypoint cursor; injector recovery is exact when that stream is present and explicitly labelled inference/default otherwise. The stream is outside the canonical hash. | A legacy/canonical-only reset can still retarget on the next tick; a diagnostic trace can isolate that rule. | `parity/trace.py`, `parity/inject.py`, injection notes. Keep canonical-only reports separate from diagnostic-internal one-step reports. |
| RESET-002 | `APPROX` | autoattack phase | The canonical hash/dump omit exact attack cooldown, windup, and attack flags. | The diagnostic stream carries those clocks/flags and the injector restores them; a canonical-only trace retains an explicit unknown/default rather than fabricating phase. Pending effects outside the fixed model remain a separate limitation. | A legacy reset can fire one tick early/late; diagnostic recovery makes the remaining simulator discrepancy measurable. | `parity/inject.py`, `parity/tests/test_inject.py`; do not call canonical-only replay exact. |
| RESET-003 | `UNOBSERVABLE` | buffs and casts | Buff callbacks, fractional phase/power, generic cast/channel metadata, and every script-private field are not fully serialized. | Fixed lanes cover implemented Garen buffs/casts; recovery cannot recreate unknown script-private state. | Divergence at expiry, spell completion, or immediately after reset. | `sim/spells.py`, `sim/buffs.py`, hidden-state audit. Extend oracle before expanding champion/content scope. |
| RESET-004 | `APPROX` | missiles and collision cache | The canonical hash/dump omit live missile integration state and the pre-move collision-cache position. | Diagnostic internals restore modelled missile owner/target/position/speed/damage and exact cached collision position; raw float32 position bits avoid injecting a rounded wire coordinate. These fields remain outside the canonical hash. Stable NetIds produced 2,400/2,400 identity-clean pairs. The full quantized run reached 28,526/28,570 (99.85%) componentwise position parity; a 300-pair raw-float pre-clash sample reached 2,160/2,170 (99.54%), with ten remaining perpendicular collision residuals. Free-running wave creation now matches the source's red/Chaos-before-blue/Order insertion order; diagnostic injection already reconstructed that order from NetIds, so this source fix correctly leaves the ten residuals unchanged. | A canonical-only/legacy trace can still schedule ranged damage or collision from the wrong state; even the diagnostic replay has not explained the last ten collision cases. | `parity/trace.py`, `parity/inject.py`, `parity/diagnostic_identity.py`, `sim/tests/test_lane.py::test_map1_top_wave_creates_red_before_blue_for_collision_order`; resolve the same-wave collision residual before closing Gate 1. |
| PROG-001 | `BOUNDED` | experience | Server exposes level but not fractional XP or level-up scheduling phase. | Injection records the exact recoverable within-level interval `[XP(level), XP(next level))` and uses a deterministic representative, never claiming the fractional value is observed. | A reset can level at a different subsequent tick. | `parity.inject.xp_bounds_for_level`; retain the interval in reports and do not use a cross-level reset as a mechanics verdict. |
| ORDER-001 | `VERIFIED` | move/target interaction | A `Move` packet does not clear `TargetUnit`; `RefreshWaypoints` can resume `AttackTo`. The real wire has no `Stop` action. | Move preserves the target; the internal `STOP` enum is not emitted and behaves as a no-op. | If regressed: the policy gains a disengage action the server lacks. | `sim/orders.py`, `sim/tests/test_orders.py`, `PORT_AUDIT_AI.md`. |
| SPELL-001 | `APPROX` | E damage snapshot AD | Server snapshots the caster's live attack damage. | Production uses profile base plus level growth, but modifiers outside the implemented buff/item slice are absent; table-free unit tests use a level-one placeholder. | Spin damage wrong after future items/buffs are added. | `_ad_placeholder` in `sim/orders.py`. Remove the fallback when all callers provide parameters and extend the live stat pipeline with content scope. |
| SCOPE-001 | `APPROX` | supported game content | The server supports the full map roster, items, runes, neutrals, objectives, all champions, and modern client rules. | The JAX scope is a 1v1 Garen top-lane training slice with lane minions and turrets. Unsupported content is absent, not approximately simulated. | Policies exploit missing pressure or fail when transferred beyond the slice. | `JAX_REWRITE_PLAN.md` scope. Add one ledger row per newly admitted content family before implementation. |
| PERF-001 | `APPROX` | routed-training throughput gate | The JAX rewrite must retain accelerator throughput high enough for RL. | Canonical command, RTX 5080, 4,096 envs, 150 s warm-up (44 live entities), 60 timed + 5 warmup: **55,054 / 55,565 / 55,049 dec/s** over three identical runs on 2026-09-18, sim-only ~109,030, compile 20.9-21.4 s. Target 56,450; short by 1.6-2.5% against ~1% run-to-run spread. The earlier 53,228/53,605 records are stale. The no-route control is 57,378, so routing is still the whole gap. **Measured negative result:** `ROUTE_LOOP_UNROLL` swept on 2026-09-18 (`--route-unroll`, canonical workload otherwise): 4 -> 55,040; 8 -> 55,565 and 55,054; 16 -> 55,049; 24 -> 55,014 dec/s. Total spread 551 dec/s (1.0%), no monotonic trend, and the two same-setting runs at unroll 8 are 511 apart -- i.e. the whole range is inside the repeat noise. Chaining more masked hop bodies is not the lever and the routed delta is not XLA loop-control overhead. (Compile does respond: 20.5 s at 4 rising to 22.1 s at 24, so a larger unroll costs compile time for nothing.) A production action-lattice audit found max 59 raw hops, but a broader reachable-cell sample found 107, so the global 128 bound cannot safely be lowered to 64 -- and the `while_loop` already exits early on the batch, so the bound is not what costs anyway. | Gate 4 remains open. Deferred terrain repair is also a separately labelled semantic approximation. | Reproduce only with the canonical command; a routed number from anything else is not gate evidence. Profile where the ~5.5 ms routed delta actually goes -- the unroll result points at the 231 MiB table's gathers, not control flow -- before optimizing. Fix the gate's repeat count and statistic: at ~1% noise, one run cannot resolve a 2% gap in either direction. |
| OPS-001 | `BOUNDED` | route asset distribution | A training checkout needs the exact artifact matching navgrid bytes, radius, and ABI. | Heavy route data live under ignored `data/jax_routes/`; production remains pinned to `map1_garen_r35_o50_v2` (231 MiB packed hops). The loader also accepts the measured `v3` same-direction-run sidecar experiment (693 MiB total), but its memory/compile cost has not yet produced a gate result, so it is not required. Unknown versions, hashes, shapes, or v3 sidecar semantics fail closed. | A fresh machine cannot start routed training until the pinned artifact is generated or distributed. | `data/local_route_artifact.py`, `train/run_train.py`; publish the pinned artifact to the project artifact store before remote training. |

## The one chain that links three open gates (2026-09-18)

`SmoothPath` is missing, and that single gap shows up as three separately-filed
problems. Worth stating in one place, because each has been worked on as though
it were independent.

1. **Gate 1, PATH-001.** On the 20-Move fixture of legal 96x54 bin centres,
   every lookup is READY but local waypoint counts agree with the server
   **0/20**: the server always emits 3, the local router 6-9. The host port of
   the server algorithm agrees 20/20, so this is not a misreading of the
   server -- the device-side reconstruction really does emit a different path.
   The reverse-BFS router only removes *collinear* cells; the server runs
   `SmoothPath`, which removes any cell the line of sight can skip.
2. **A path with 6-9 waypoints instead of 3 is a longer path.** It tracks cell
   centres instead of cutting corners. That is distance, and distance is
   decisions.
3. **Gate 3.** The first routed run has the sim spending **6,400** of its
   18,000 decisions walking in, against the server's **3,197** — while walking
   the identical scripted `APPROACH_WAYPOINTS`. It therefore farms about 21%
   less of the episode, and its champion reaches level 5 where the server's
   reaches 8. Some of that total is the extra death (2 vs 1) rather than speed;
   the new per-walk `walks` field is what separates the two, and it has not been
   run yet.

The predicate this needs already exists and is verified: the bounded device
`CastCircle` agrees with the host on the 20-click corpus (10 clear, 10 blocked,
zero bound exhaustion). What is missing is emitting waypoints through it.

**This is the highest-value single fix available.** It closes a gate-1 item
outright, and it is the leading candidate for the gate-3 lane-time deficit,
which is currently the largest unexplained term in that gate. It is *not*
expected to help gate 4 — fewer waypoints would mean less work per route, but
the measured routed delta is in the table gathers, and the `while_loop` already
exits early, so do not justify this work on throughput.

**Do not read gate 3's -1 as nearly closed.** The raw path gave +3, routing
gave -1, and the sign flipped because the sim lost farming time, not because
last-hitting converged. Fix the walk, then re-measure; a gate that agrees
because both sides are wrong in opposite directions is the failure mode this
ledger exists to prevent.

## Local-click routing design record

The routing target is intentionally local even though the map data is global:

1. The champion is the follow-camera origin.
2. The policy chooses one cell of the 96x54 screen grid. Under the calibrated
   1280x720 perspective, the farthest non-minimap bin centre is 2,281.27 world
   units from the champion. Canonical screen/lane components reach 1,892.63
   and 1,273.63 units; after rotation into the top-lane world frame, an x/y
   component can reach about 2,245 units. `SCREEN_RADIUS=1800` is a
   visibility/UI fact, not the click-projection bound.
3. Static terrain chooses a bounded waypoint route for that source/destination.
   There is no need for an unrestricted all-pairs map table.
4. Dynamic minions and champions are **not** baked into that route. They move,
   so the sequential collision pass supplies creep block on each tick.
5. Static routing and dynamic collision are both required: the former handles
   walls and alcove edges; the latter changes spacing and trading patterns.

The earlier whole-map `K x K` estimate in `JAX_REWRITE_PLAN.md` describes one
possible artifact ABI, not the required production topology. For a 50-unit
grid, the measured canonical projection reaches about 38 by 26 cells and the
rotated world-grid components reach about 45 cells before endpoint, anchoring,
and radius details; the production artifact deliberately covers a conservative
+/-50-cell square. A source-indexed local representation is
therefore proportional to map cells times local offsets, rather than all map
cells squared. The implementation still has to preserve exact endpoint,
tie-break, smoothing, overflow, and no-route semantics or label each departure
`APPROX` above.

## Maintenance rule

When debugging a policy/server mismatch, record the smallest reproducible
state/action/tick in the relevant audit and link it from the ledger row. When a
new deviation is found, add it here before optimizing around it. “Probably
irrelevant” is not a valid reason to omit a row: tiny timing and geometry
differences can alter creep block, a trade, and then the entire trajectory.
