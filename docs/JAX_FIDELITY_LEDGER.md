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
| 1. whole-corpus one-step parity | **OPEN, two named bugs** | Full exact-cache corpus re-run 2026-09-18 on `desktop` (job 882): **19,800 tick-pairs, 910,216 unit injections, 19,800/19,800 on diagnostic NetId identity, 0 legacy-proximity fallbacks.** Champion: **100% on every field, including target and AA fire.** LaneTurret: position/waypoints/aa_hit 100%. LaneMinion over 395k samples, counts not percentages: position L-inf <=1/16 **-1,742**, hp **-1,127**, waypoints **-716**, move order **-34,295**, deaths 44, spawns 6 of 120 ticks. Target identity and AA fire tick are **now scored for the first time** (`_compare_controller`, `c32d4c9`): minion target -736, aa_fire -2,880; turret target -1,508, aa_fire -1,483. Base rates make those non-vacuous -- minions fire 2,337 times on the server against **5,121** in the sim, turrets **44 against 1,527**. The stale 99.91/97.40/99.95 figures reproduce **bit-identically** on the same 2,400-pair window, so the spawn fix did not touch them; it moved position only (99.85% -> 99.93%). They were never whole-corpus numbers. | Two mechanisms are named and both are fixable. **(1) `AA-001`, the sim swings one tick early** -- see its row. **(2) move order**: 5,390 of 5,423 drilled mismatches are `sim=HOLD, server=ATTACK_TO`. The server sets Hold in `RefreshWaypoints` (`ObjAIBase.cs:651-653`), which `UpdateTarget` reaches only when `AutoAttackSpell.State == STATE_READY` (`:1239-1242`); during windup that branch is skipped and `LaneMinionAI`'s 250 ms timer re-issues AttackTo. `sim/orders.py` writes `hold = has_tgt & in_rng` every tick, gating neither. Both sides are stationary throughout, so it is a label divergence today -- but it is on the observation wire and it gates `can_move`. HP: 211 of 219 drilled minion misses are on a tick with a missile in flight (7.0% disagreement with a missile, 1.3% without), i.e. the `RESET-004` blind spot, median 12 HP. **Not settled:** the 1,508 turret target cases, all "sim holds a minion, server holds none"; range is genuinely 750, and the suspect is a one-tick blind gap (`TurretAI.OnUpdate` runs before `UpdateTarget`, whose drop path returns). Settling measurement: for each, whether the server's tick-N target was alive at N+1. |
| 2. free-running divergence characterized | **PASS** | Four 600 s scenarios are recorded in `TIER2_DIVERGENCE.md`; divergence begins around wave interaction and is explicitly large. | None for Phase 1; retain as a regression corpus. |
| 3. oracle last-hitter CS@10 | **OPEN; chain traced to one unexplained death** | Canonical routed run, 18,000 decisions: sim CS=3 / attacks=54 / deaths=2 / level 7 against server CS=4 / attacks=86 / deaths=1 / level 8. **Everything reduces to one event, and that event is `COLL-003`.** (1) 82.1% of the 3,203 excess approach decisions is the sim's second death landing mid-walk. (2) The level gap is exactly five missing melee XP grants, all in the window holding both deaths; grant sizes, the 82 red-minion deaths and the level curve all match the server exactly. (3) Splitting exposure by phase: **engaged in lane, the two engines are identical -- 9,395 attacker-decisions against 9,381 (1.001x)**; walking in, 4,210 against **60**. 100% of the exposure gap and 83% of the damage gap is one respawn walk in which the champion travels 800 units of path for **1 unit of net displacement** over 811 decisions, wedged at the exact sum of the collision radii. Standing in lane the sim is *less* crowded than the server. Ruled out with numbers: acquisition census (96.1% vs 95.5% with no alternative), acquisition radii (475/700, identical by source and table), blue attrition (blue lives *longer* in the sim, 61.5 s vs 59.4 s), clash midpoint (0.50 both), movement speed (11.49 vs 11.50 u/dec), `MOVE-001` (never binds), route fallback (324/6,400 decisions, still at full speed), `AA-001` (1.009x swing rate in rollout, not 2.19x), and `TURRET-001` (right size, **wrong sign**). | **Explain the 394.3 s death.** `COLL-003` is verified faithful, so the wedge is a consequence, not a bug -- the sim's champion is in a wave only because it died 131 s before the server's and met a different lane state. The live lead is *clumping*, not exposure: engaged decisions with >=3 attackers are 1,844 against 1,360 (1.36x) and with >=8 are 85 against 39 (2.2x), while decisions with >=1 attacker are 2,830 against 3,530 (**0.80x**). Same total exposure, fewer and denser clumps. Next measurement: the joint arrival process -- inter-acquisition intervals and overlap of concurrent holds, restricted to engaged decisions; both occupancy series are already on disk, so this is analysis-only. Still owed regardless: one identical champion-into-wave state injected into both engines, which is the only differential that can rule the collision pass in or out. Then re-run and re-attribute rather than assuming the chain unwinds. Two residuals to revisit afterwards, both currently downstream: per-hit damage +7.8% (armour, i.e. the level gap) and champion regeneration 1,869 HP against 2,661. |
| 4. full-loop throughput >=56k decisions/s | **PASSES without `SmoothPath`; 9.2% short with it** | RTX 5080, canonical command. **The routed delta was never the hop table.** Profiled 2026-09-18: routing is 4.349 ms of a 74.4 ms step, and inside it `closest_terrain_exit` is **90.3%** while the 231 MiB table's gathers are **0.3%**. Three independent refutations of the table: the route loop runs 6 trips at unroll 8 (1.74% body utilisation); 8,192 gathers cost 14.1 us from the 231 MiB table against 13.6 us from 16 KiB and an 11.8 us empty-kernel floor; and it achieves 51.5 GB/s, 5% of peak. After the spiral unroll and the `CastCircle` restructure (`d47ab44`): **no-route control 59,042 · routed without `SmoothPath` 58,255 (PASS, +3.2%) · routed with `SmoothPath` 51,270 (short 9.18%) · with `SmoothPath` at a 64-step line bound 54,741 (short 3.03%)**. The prior 55,050 and the stale 57,378 no-route figure both predate this. | **This is now a fidelity/throughput decision, not a bug.** `SmoothPath` costs about 12% and takes waypoint-count agreement from 6.0% to 55.0%. Three options, all measured: ship the control and fail gate 1's waypoint clause; ship `SmoothPath` and miss gate 4 by 9.2%; or lower `SMOOTH_CAST_LINE_STEPS` to 64 for -3.03%, which fails **closed** (less smoothing, never a wrong route) on the 8.2% of segments that exceed it and is reported through `smooth_exhausted`. Not yet tried: the line bound measured on the *real* action lattice rather than a uniform +/-50-cell corpus, which is likely far tighter than 128 since gameplay clicks are viewport-bounded. |
| 5. compile under two minutes | **PASS** | Routed compile 20.9-21.4 s across the three 2026-09-18 runs (earlier record: 12.9-13.6 s; it has grown with the routing work but remains far inside the 120 s gate). | None. Watch it: it moved 60% without anyone noticing, which a gate with this much headroom will not catch. |
| 6. reset cost small | **PASS** | 2.17% of a step at 512 envs and 1.74% at 2,048. | None. |

Phase 1 is therefore **not complete**, but as of 2026-09-18 it is no longer
open in the way it was. **Gate 4 passes** on the simulator without
`SmoothPath`, and what is left there is a stated fidelity/throughput trade, not
a missing lever. **Gate 3's chain is fully traced and its last link is
*not* a bug**: engaged in lane the two engines are identical to 0.1%, and the
deficit is one respawn walk spent wedged in a wave -- by a collision rule that
`COLL-003` now verifies as faithful. What is left is the extra death at 394.3 s
that put the champion there. **Gate 1** has two named,
fixable mechanisms and one unsettled question. What
changed on 2026-09-18 is that they stopped being open in the same way. Gates 1
and 3 now have *named mechanisms* rather than unexplained residuals: `AA-001`
(every sim swing one tick early), the `RefreshWaypoints` Hold gating, and one
remaining upstream fact -- the sim's champion takes +39% damage and is swarmed
+49% more -- which `AA-001` is a live candidate to explain on its own. Gate 4
is unchanged and independent, and remains the one gate with no identified
lever.

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
| 1 (waypoints) | `python -m lanerl_jax.parity.movement_parity` — scores the host A*+SmoothPath port **and** the production device router on the same 20 Moves; boots a real server | `desktop`, CPU |
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
| PATH-001 | `APPROX` | champion move routing | `NavigationGrid.GetPath` runs radius-aware, closed-on-enqueue A* from the champion's exact float position to the exact float click, then `SmoothPath`. | Production `run_train` loads the radius-aware local table and reconstructs a static-terrain route. `SmoothPath` is ported as of `de5cc2b` and agrees with the host port 400/400; what remains is the **A* tie-break**. Its deterministic reverse BFS reproduces the server's own cell path only 17/400 times. A caller omitting the table still gets `[position, destination]`, explicitly exposed as `--no-route-table`. On the fresh 20-Move fixture generated from actual legal 96x54 non-minimap bin centres, every lookup is READY; the host server-algorithm port agrees with the server 20/20. Local waypoint counts agreed 0/20 before smoothing (server always 3, local 6-9); on the 400-route artifact corpus smoothing takes count agreement from 6.0% to 55.0% and mean polyline length from 1.177x the server's to 1.049x. | A different safe side/curve around terrain changes arrival and trade timing; disabled routing causes wall entry/ejection. | `data/local_route_artifact.py`, `sim/local_pathing.py`, exact host reference in `data/navgrid.py`. The bounded device `CastCircle` agrees with the host on the 20-click corpus (10 clear, 10 blocked) and, checked for the first time at *smoothing* length rather than A*-neighbour length, 4,000/4,000 on segments up to 60 cells with zero bound exhaustion. Integrating `SmoothPath` exposed two float-precision port bugs on the host side; see the SmoothPath section below. Earlier route decomposition independently showed both missing LOS smoothing and BFS/A* path choice. |
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
| RESET-004 | `APPROX` | missiles and collision cache | The canonical hash/dump omit live missile integration state and the pre-move collision-cache position. | Diagnostic internals restore modelled missile owner/target/position/speed/damage and exact cached collision position; raw float32 position bits avoid injecting a rounded wire coordinate. These fields remain outside the canonical hash. Stable NetIds produced 2,400/2,400 identity-clean pairs. The full quantized run reached 28,526/28,570 (99.85%) componentwise position parity; a 300-pair raw-float pre-clash sample reached 2,160/2,170 (99.54%), with ten remaining perpendicular collision residuals. **Those ten are resolved as of 2026-09-18 and were never a collision bug** (`parity/tier1_same_wave_collision.py`): the current tree is **2,170/2,170**, and reverting *only* `MINION_SPAWN` to its rounded pre-fix value reproduces exactly 10. All ten are freshly spawned minions sitting on the rounded barracks coordinate with **zero colliding neighbours**, five spawn events by two teams, displaced by precisely the two constants' rounding deltas. No collision pass ever touched them. Free-running wave creation now matches the source's red/Chaos-before-blue/Order insertion order; diagnostic injection already reconstructed that order from NetIds, so this source fix correctly leaves the ten residuals unchanged. | A canonical-only/legacy trace can still schedule ranged damage or collision from the wrong state; the last ten collision cases are explained: an already-fixed wrong spawn constant, misattributed to collision ordering. | `parity/trace.py`, `parity/inject.py`, `parity/diagnostic_identity.py`, `sim/tests/test_lane.py::test_map1_top_wave_creates_red_before_blue_for_collision_order`; ~~resolve the same-wave collision residual before closing Gate 1~~ **done 2026-09-18**; keep `tier1_same_wave_collision.py --rounded-spawn` as the regression control, since it reproduces the failure on demand. The live blind spot is missiles: 74.3% of corpus ticks carry one, and HP disagreement is 7.0% on those against 1.3% without. |
| PROG-001 | `BOUNDED` | experience | Server exposes level but not fractional XP or level-up scheduling phase. | Injection records the exact recoverable within-level interval `[XP(level), XP(next level))` and uses a deterministic representative, never claiming the fractional value is observed. | A reset can level at a different subsequent tick. | `parity.inject.xp_bounds_for_level`; retain the interval in reports and do not use a cross-level reset as a mechanics verdict. |
| ORDER-001 | `VERIFIED` | move/target interaction | A `Move` packet does not clear `TargetUnit`; `RefreshWaypoints` can resume `AttackTo`. The real wire has no `Stop` action. | Move preserves the target; the internal `STOP` enum is not emitted and behaves as a no-op. | If regressed: the policy gains a disengage action the server lacks. | `sim/orders.py`, `sim/tests/test_orders.py`, `PORT_AUDIT_AI.md`. |
| SPELL-001 | `APPROX` | E damage snapshot AD | Server snapshots the caster's live attack damage. | Production uses profile base plus level growth, but modifiers outside the implemented buff/item slice are absent; table-free unit tests use a level-one placeholder. | Spin damage wrong after future items/buffs are added. | `_ad_placeholder` in `sim/orders.py`. Remove the fallback when all callers provide parameters and extend the live stat pipeline with content scope. |
| SCOPE-001 | `APPROX` | supported game content | The server supports the full map roster, items, runes, neutrals, objectives, all champions, and modern client rules. | The JAX scope is a 1v1 Garen top-lane training slice with lane minions and turrets. Unsupported content is absent, not approximately simulated. | Policies exploit missing pressure or fail when transferred beyond the slice. | `JAX_REWRITE_PLAN.md` scope. Add one ledger row per newly admitted content family before implementation. |
| AA-001 | `APPROX` | auto-attack cooldown ordering | `ObjAIBase.Update` calls `UpdateTarget()` (`ObjAIBase.cs:1101`) and only **then** decrements `_autoAttackCurrentCooldown` (`:1103-1105`). A unit therefore cannot swing on the tick its cooldown expires. | `sim/autoattack.py` decrements first and then gates -- and its docstring asserts the opposite order as established fact. **Every sim swing starts one tick (16.67 ms) early.** Measured on the 19,800-pair corpus: 2,824 of 2,880 minion `aa_fire` disagreements are the single shape "sim fires, server does not, targets agree, target a median 30 units *inside* range", so it is not a range-boundary effect. Worked example, minion net 1073743551: dumped `aacd` 17 -> 0 -> 0 -> **802** with `attacking` 0,0,0,1; 802/1024 = 0.7832 = 0.8 - 1/60, i.e. the server set the period and decremented in the same tick, withholding the swing for exactly the two ticks the sim fires on. The second of those is additionally unobservable: the true cooldown is a sub-quantum positive residue that `LanerlAim.AutoAttackCooldownRemaining` clamps and `Q()` rounds to 0. | **This is not a rounding-level effect.** The sim fires 5,121 minion swings against the server's 2,337 and 1,527 turret swings against 44. That changes wave attrition, trade outcomes and last-hit timing -- i.e. it is a candidate cause for Gate 3's unexplained +39% damage and +49% swarm, and for the CS gate itself. | **Reconciled against a rollout, 2026-09-18: the 2.19x is a per-pair injection artifact, not a rate error.** In a free-running 600 s episode the swing rate against the champion is **1.009x**, total HP removed from minions is within 3%, and blue minions live *longer* in the sim -- the opposite of the predicted chain. Each corpus pair re-injects true server state, and two consecutive ticks read an unobservable sub-quantum cooldown as 0, so one server swing can be counted as two sim fires. The bug is real and is a **one-tick phase error**; it is not a doubling of anyone's damage output, and it is **not** the cause of Gate 3. | `sim/autoattack.py`; the corpus comparison in `parity/one_step.py::_compare_controller` and `LaneMinion.aa_cooldown` residuals (median 0.0010 s = one quantum, p95 1.4932 s = a whole period). Deliberately **not fixed yet**: it changes every trajectory and a gate-3 attribution is mid-flight against this baseline. Fix it, then re-run the corpus AND the gate-3 driver, and compare both. |
| COLL-003 | `VERIFIED` (the port) / gate-3 cause **NOT** collision | The trigger and the resolution use **different radii**, verified at four independent source sites: `IsCollidingWith` sums `CollisionRadius` (`GameObject.cs:226-229`), `OnCollision` escapes with `GetCircleEscapePoint(Position, PathfindingRadius + 1, collider.Position, collider.PathfindingRadius)` (`AttackableUnit.cs:312`), `Champion` does **not** override `OnCollision`, and `CollisionHandler.cs:163-165` calls `obj.OnCollision(obj2)` so each object moves **itself**, sequentially. | The sim reproduces all of that. Consequence, measured with the real radii: champion (coll 30, path 35) against a **melee/caster** minion (coll 40, path 35.7) triggers at 70 and escapes to 71.7 -- a 1.8u push against 5.75u of travel per tick, so it walks through. Against a **cannon** (coll 40, path **55.7**) it triggers at 70 and escapes to **91.7** -- a **21.8u push against 5.75u of travel**. Head-on, that is a ratchet: the champion is thrown back faster than it can walk. Content confirms the asymmetry is real, not a loading error: `PathfindingCollisionRadius` 55.7437 for cannon, 35.7437 for melee, 35 for Garen, while `Minion.cs`/`Champion.cs` hard-code the 40/30 trigger radii. | **This is the mechanism of the gate-3 wedge, and it is FAITHFUL.** The sim's walk-1 shows it plainly -- 2,543 decisions at 6.17 u/decision against 11.50 on its clean walks, **686 direction reversals**, 23% of decisions under 1u, in contact 72% of the time at a median 68.1u. But the server would ratchet identically from the same geometry, so this does **not** make the sim wrong. It makes the sim *unlucky*: its champion walks into a wave and the server's does not. | **Negative result: do not look for a collision bug here.** The divergence is upstream. The server's walk-1 has a median nearest minion of **536.7u** and **zero** reversals -- it never met a wave, because it died at 525.6 s where the sim died at 394.3 s and drew a different lane state. The open question is therefore the **394.3 s death**, not the walk. The remaining collision-side check that would be decisive, and has **not** been run: inject one identical champion-into-wave state into both engines and compare per-tick displacement. Until that exists, "the collision pass is right" rests on source agreement, not on a differential. |

| TURRET-001 | `APPROX` | turret acquisition vs retention radius | Acquisition goes through the quadtree's circle-vs-circle test, `dist2 < (Range + CollisionRadius)2` (`QuadTree.cs:61-64` via `CollisionHandler.cs:119-128`, `TurretAI.cs:41`) -- **790** for a 40-radius minion. Retention is a flat centre-to-centre **750** (`TurretAI.cs:29-32`). Selection is **priority only, distance never** (`TurretAI.cs:43-62`). | The sim uses `attack_range` for both (`sim/targeting.py::turret_acquire`, `sim/step.py:663-666`), so the server's trap never fires. That trap: a best-priority minion in the 40-unit annulus is picked by `CheckForTargets` and dropped by the retention test four lines later **in the same `OnUpdate`**, every tick. Measured: a caster at 764 outranks a melee at 196, and one blue turret sat idle at cooldown 0 for **664 consecutive ticks (11 s)** with minions at 196 units. Of the 1,508 target disagreements, the server held nothing at tick N on **1,484**, and still nothing at N+2 on 1,481. The pre-tick blind-gap hypothesis (`TurretAI.OnUpdate` before `UpdateTarget`) is real but explains **24**. | Free-running, ~21 extra blue-turret swings per 330 s (**+48%**, not the per-pair 35x) at a median 168 damage, so **~18 extra red minion deaths per 600 s** -- about 15% extra red-side attrition. Sign matters: this gives the blue champion **fewer** red minions to face, so it works **against** Gate 3's signature rather than explaining it. | `sim/targeting.py`, `sim/step.py`; `parity/turret_target_drill.py` reproduces both phases. Fix by widening the candidate radius to `attack_range + target collision radius` while leaving retention at `attack_range`. **Not settled:** a second-order route runs the other way (clearing the red wave faster lets the blue wave push toward the red tower), and a one-step corpus cannot see it, because re-injecting true state every tick destroys exactly that compounding. |
| PERF-001 | `APPROX` | routed-training throughput gate | The JAX rewrite must retain accelerator throughput high enough for RL. | **The table was the wrong suspect and the unroll sweep tuned the wrong loop.** Full profile 2026-09-18 (5 repeats x 60 steps): observation 10.005 ms, policy 14.486, `apply_orders` non-routing 0.926, routing 4.349, `step_decision` 44.737. Inside routing, `closest_terrain_exit` is 90.3% and the hop-table gathers are **0.3%**. `ROUTE_LOOP_UNROLL` moved nothing because it tunes a loop that runs **6** trips; the spiral runs up to **203**, on a per-lane distribution of p50 = 0, p90 = 0, p99 = 88 -- so 97.5% of lanes need none and one straggler makes the whole `vmap` batch pay. Three semantics-preserving fixes landed in `d47ab44` (spiral unroll, `CastCircle` reducing once instead of per-step scattering, and a `scan(unroll=32)` on the line walk whose cost was *perfectly* linear in its bound at ~18 us/step, i.e. essentially all launch overhead). Result: **58,255 dec/s without `SmoothPath` -- gate 4 PASSES** -- and 51,270 with it. Negative result worth keeping: unrolling `SmoothPath`'s own greedy makes it **worse** (82.97 ms at 1, 103.81 at 16), because a masked-off chained body there is a whole wasted `CastCircle` rather than amortised launch overhead. | Gate 4 passes on the pre-`SmoothPath` simulator. What remains is a stated trade, not an unknown. | Reproduce only with the canonical command; `--no-smooth` and `--smooth-line-steps` are labelled controls. Next, if the 12% is wanted back: measure the line-step distribution on the **real action lattice** (gameplay clicks are viewport-bounded, so 128 is likely far looser than needed), and surface `smooth_exhausted` in training the way `route_nonready` is. Unattributed and larger than all of the above: `step_decision` is 60% of the step, with `input_reduce_fusion_13` at 10.80 ms and ~12.8 ms of CUB radix sorts. |
| OPS-001 | `BOUNDED` | route asset distribution | A training checkout needs the exact artifact matching navgrid bytes, radius, and ABI. | Heavy route data live under ignored `data/jax_routes/`; production remains pinned to `map1_garen_r35_o50_v2` (231 MiB packed hops). The loader also accepts the measured `v3` same-direction-run sidecar experiment (693 MiB total), but its memory/compile cost has not yet produced a gate result, so it is not required. Unknown versions, hashes, shapes, or v3 sidecar semantics fail closed. | A fresh machine cannot start routed training until the pinned artifact is generated or distributed. **Generating and loading need different environments**, measured 2026-09-18: the baker is numba-parallel and raises rather than guessing, and `.venv-gpu` is a real venv with `include-system-site-packages = false`, so it cannot see the conda env's numba. Artifacts are therefore generated in `.venv-jax` (login) and only *loaded* in `.venv-gpu` (desktop/GPU). Loading is pure numpy, so routed training and gate 4 are unaffected -- the gate-4 run loads the 231 MiB v2 table in `.venv-gpu` without numba. Left that way on purpose: numba pins numpy, and `.venv-gpu` is the environment every gate-4 number is measured in. | `data/local_route_artifact.py`, `train/run_train.py`; publish the pinned artifact to the project artifact store before remote training. |

## SmoothPath: what it fixed, and what it turned out not to explain (2026-09-18)

`SmoothPath` was missing and is now ported (`sim/local_pathing.smooth_cell_path`,
commit `de5cc2b`). It was filed as the single gap behind three open gates. One
third of that is right; the rest was an inference that the measurement does not
support, and it is worth writing down which is which.

**What it is.** The server runs A*, then a single greedy pass that deletes any
cell the last kept one can see through a swept circle of the pathfinding
radius. Our reverse-BFS router removed only *collinear* cells. Hence 6-9
waypoints where the server emits 3.

**What it fixed.** 400 routes through the production `map1_garen_r35_o50_v2`
artifact, scored against the server's own `get_path` (host self-check 400/400):

| | raw cells | before | after | server |
|---|---|---|---|---|
| mean waypoints | 37.48 | 8.97 | 4.62 | 4.49 |
| count agrees with server | - | 24/400 (6.0%) | 220/400 (55.0%) | - |
| polyline length vs server (mean / median / p90) | - | 1.177 / 1.155 / 1.337 | 1.049 / 1.004 / 1.162 | 1.000 |

**What it does not explain: Gate 3.** The argument was "more waypoints means a
longer path, and distance is decisions". The first half is true and is now
measured: **+17.7% mean, +15.5% median, 1.41x worst case**. The second half
does not carry the weight it was given. Gate 3's sim spends **6,400** approach
decisions against the server's **3,197** -- a factor of **2.0**. Path geometry
can account for at most about a fifth of that, and the worst single route in
the corpus is still only 1.41x. **Something other than pathing dominates the
approach gap**, and looking for it inside the router is looking in the wrong
place. Candidates now promoted ahead of geometry: effective movement speed /
per-tick waypoint budget (`MOVE-001` caps transitions at 8), orders re-issued
mid-walk, the `route_status` histogram over the approach legs, and walks
truncated by death.

**What it does not fix: Gate 1.** Smoothing cannot close waypoint parity by
itself, because the input path is wrong before it is smoothed. Only **17/400**
baked itineraries are the server's own A* cell path -- the local artifact is a
reverse BFS over a CastCircle-valid neighbour graph, chosen so the bake is
tractable. That caps exact waypoint agreement near 40% however the list is
smoothed, and it is what the remaining 45% of count disagreement is. This is a
bake-algorithm gap (PATH-001, PATH-003), not a smoothing gap, and it cannot be
closed by re-baking with A*: the server's search is closed-on-enqueue, so
matching it needs its exact frontier order per (source, goal) pair, which is
484M searches for this artifact.

**What it does not touch: Gate 4.** Unchanged and independent. Smoothing adds
CastCircle work per route; its cost has not yet been measured.

**Two port bugs it exposed.** Both surfaced because the device and host
disagreed on 2 of 400 routes -- not from rereading the source.

1. `navgrid.cast_circle` built the offset endpoints in float64; the server
   builds them in `Vector2`, i.e. float32. Both disagreements resolved to the
   float32 answer. The rate is below 0.05% on random segments (0 in 40,000,
   including 20,000 bake-shaped adjacent-neighbour queries, so the 231 MiB
   artifact is not implicated) -- but SmoothPath's greedy pushes every cast
   until it fails, so it probes the clear/blocked boundary *on purpose*. A
   predicate can be right everywhere that does not matter.
2. `cells_in_line` then walked with a float32 error accumulator inherited from
   those endpoints, where the source declares `double`.

Device and host SmoothPath now agree 400/400. The device `CastCircle` was
already exact -- 4,000/4,000 against the host on segments up to 60 cells, zero
bound exhaustion -- which is the first time it had been checked at smoothing
length rather than A*-neighbour length.

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
