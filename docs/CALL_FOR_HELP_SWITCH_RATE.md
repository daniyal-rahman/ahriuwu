# Call-for-help: the switch rate is faithful, and that is not the story

**Bottom line.** `1b09f16` blocked wiring `call_for_help_map` into `step.py`
because it could not show the switch RATE was faithful ("47 call-for-help
switches in 600 s" vs "1,165 sim target switches... but that counts ordinary
retargets too"). That specific number was a mis-citation (see part 1) and the
real rate check (part 2) clears easily: **1.58x server on the exact scenario
that matters**, not 25x. The switch rate was never the blocker. The real
blocker, found here, is a mechanism-level one: on a fully idle lane
call-for-help is a **reinforcement/recruitment channel that amplifies
whichever side is already ahead**, and this lane's population equilibrium is
independently, already documented as unstable
(`lanerl_jax/sim/tests/test_lane.py::test_minion_population_is_close_to_the_server`'s
own docstring: "three separate corrections tipping this lane in three
unpredictable directions... the signature of an unstable equilibrium"). Adding
a faithful, correctly-rated new feedback channel to an already-fragile
equilibrium can make it LESS like the server even though the channel itself is
right. That is a real, structural finding, not a calibration gap -- so
`enable_call_for_help` stays a toggle, default `False`. See part 6 for the
full verdict and part 7 for what would need to change before this could be
default-on.

Everything below was measured on this checkout. Every run's exact command is
given so it can be reproduced. `LANERL_VENDOR_ROOT=/srv/nfs/projects/lanerl-vendor`
is required in this worktree specifically because it is nested under
`.claude/worktrees/`, which breaks `lanerl_jax/data/paths.py`'s
walk-up-from-`__file__` inference (`projects_root()` lands on the worktree
directory instead of `/srv/nfs/projects`) -- confirmed by two `FileNotFoundError`s
in `lanerl_jax/sim/tests/test_combat.py` that disappeared once the override was
set; unrelated to anything below.

## 0. What landed in code

* `lanerl_jax/sim/targeting.py::call_for_help_map` -- unchanged, already
  correct (see `1b09f16` and `docs/TARGET_ACQUISITION_DIFF.md`).
* `lanerl_jax/sim/minion_ai.py` -- `MinionAIOut` gained a `cfh_switch` field
  (the mask `step_minion_ai` already computed internally, now exposed).
  Diagnostic only: nothing reads it, so no behaviour changes.
* `lanerl_jax/sim/step.py` -- `tick()` and `step_decision()` gained
  `enable_call_for_help: bool = False`. When `True`, `call_for_help_map` is
  computed from the tick's own damage matrix and written to
  `state.help_priority` for the NEXT tick's minion AI pass (matching the
  server's one-shot, `callsForHelpMayBeCleared`-then-wiped lifetime). When
  `False` (every existing caller, every existing test), `help_priority` is
  carried forward unchanged -- bit-for-bit the old behaviour. Judgment's
  damage (`bs.damage_dealt`, a per-victim scalar) is NOT folded into the
  broadcast -- booked, not silently dropped, and confirmed inert for both
  scenarios measured here (neither ever casts Judgment: the idle lane issues
  no champion orders at all, and `StandInWave` only moves/holds).
* `lanerl_jax/parity/perturbation.py` -- new `IdleLane` perturbation (zero
  orders, ever; distinct from `NullControl`, which camps the champion IN the
  lane) and `enable_call_for_help` threaded through `run_sim_episode` and the
  CLI (`--call-for-help`), so the champion-in-lane and fully-idle scenarios
  can be measured through the same instrument. See part 5.
* 169 tests pass unchanged (`LANERL_VENDOR_ROOT=/srv/nfs/projects/lanerl-vendor
  JAX_PLATFORMS=cpu PYTHONPATH=<worktree> .venv-jax/bin/python -m pytest
  lanerl_jax/sim/tests/ lanerl_jax/parity/tests/ -m "not slow"`).

## 1. The "47" in the task background is a mis-citation

`docs/TARGET_ACQUISITION_DIFF.md` (`1b09f16`) reports, correctly, "47 `MRT
... to=Champion` events over the 600 s episode" -- i.e. 47 total ACQUISITIONS
of the champion by a red minion, of which the table right below it says only
**1** had `cfh=1`. The very next commit's message (`1b09f16`'s own body, and
this task's background) shortens that to "the server's MRT trace records 47
call-for-help switches in 600 s," which is a different, larger set (every
`cfh=1` line anywhere in the trace, not just the ones landing on the
champion) that nobody had actually counted.

Grepping the SAME raw log that produced that table
(`/tmp/claude-1000/-srv-nfs-projects/6c72189a-2818-4135-9f1e-aaa397df05ce/scratchpad/server_target_run.log`,
the champion-oracle-driven 600 s run behind `docs/TARGET_ACQUISITION_DIFF.md`,
re-verified here to still reproduce that doc's own sanity numbers -- 18,000
decisions, `n_mrt=1211` matches its recorded methodology):

```
grep -c '^MRT ' server_target_run.log          # 1211 total switches
grep '^MRT ' server_target_run.log | grep -c 'cfh=1'   # 453
grep '^MRT ' server_target_run.log | grep -c 'cfh=0'   # 758
grep '^MRT ' server_target_run.log | grep 'to=Champion' | grep -c 'cfh=1'  # 1
grep '^MRT ' server_target_run.log | grep -c 'to=Champion'                # 47
```

**453 of 1211 switches (37.4%) on that run are genuine call-for-help
switches** (`cfh=1`), not 47. The 47 figure is real but is specifically "how
many times ANY minion acquired the champion," a number that has almost
nothing to do with `cfh=1` (46 of those 47 were plain base-priority
acquisitions, `toprio=11`, `cfh=0`). Both this task's background and
`1b09f16`'s commit message inherited the conflation. Flagged so it does not
propagate a third time.

## 2. The comparable sim-side count, isolating cfh switches from ordinary retargets

The task's actual ask: build a sim-side count that isolates `cfh=1`-equivalent
switches specifically, and compare. Definition, matched exactly to the
server's `cfh=1` (`LaneMinionAI.cs:157-274`, specifically the restricted
`FoundNewTarget(true)` scan at `:204-215` actually committing a new target at
`:266`): a minion whose CURRENT target was still valid this tick and which
nonetheless switched. This is not an approximation -- both the server source
and `step_minion_ai` (`keep = run & valid_after`, `took_new = run & ~keep &
...`, `lanerl_jax/sim/minion_ai.py:242,250`) structurally guarantee the
ordinary sweep can never displace a valid incumbent, so any such switch is
call-for-help by construction. Recovered by calling the real,
already-tested `step_minion_ai` a second time with the exact inputs `tick()`
used internally (`x, y` taken from `tick()`'s own output, since nothing after
the AI pass touches position within a tick) -- not by re-deriving the
validity logic by hand. A same-tick sanity check
(`shadow.target == tick()'s own target`, every lane minion, every tick)
holds for the full runs below, confirming the shadow call is not silently
diverging from what `tick()` actually did.

Script: `cfh_switch_rate.py` (this session's scratchpad, reproducible from
the snippet below).

**Idle lane, 600 s (18,000 ticks), `enable_call_for_help=True`, seed 0:**

| | count | per 600 s |
|---|---:|---:|
| sim, isolated cfh switches | **582** | 582 |
| server, `cfh=1` lines (fresh clean run, part 3) | **368** | 368 |
| ratio (sim/server) | | **1.58x** |

1.58x is a real gap worth naming, but it is nowhere near the "25x too many"
`1b09f16` could not rule out, and the champion-in-lane scenario (part 2b)
lands in the same order of magnitude. The switch-rate objection that blocked
wiring this in is retired.

### 2b. Champion-in-lane scenario, for completeness

`StandInWave` (10 min, seed 0, `enable_call_for_help=True`), same shadow
count: **628** isolated cfh switches over the 10-minute (18,000-decision)
episode (script: `cfh_switch_rate_champion.py`). Not directly comparable to
the server's 453 (different scenario: `StandInWave` is 10 minutes with the
champion parked at `ENGAGE_POINT` for a 5 s excursion around t=180s, not
`last_hit_drive`'s full 600 s of oracle-driven farming, and 10 minutes of
`StandInWave` is a longer window than the server's 600 s reference), but the
per-600s rate (628 x 600/1800 = 209) is the same order of magnitude as the
idle-lane number above and the server's 453, and nowhere near a 10-25x
blowup either. Reported for completeness, not leaned on for the mechanism
argument below, which does not need it.

## 3. A fresh, complete idle-lane server trace

The only idle-lane (zero champion orders) `LANERL_AGGRO_TRACE` log already on
disk (`scratchpad/idle600/idle600/instance000.log`) stops at
`t=186209` ms -- not the full 600 s -- so it was not trusted for a 600 s
total. Re-ran clean (`idle_aggro_run.py`, `ServerLaunchSpec(toponly=True,
bot_teams="none", bot_seed=4242, step_ticks=2, extra_env=TRACE_ENV)`,
18,000 decisions, `env.step([None])` every decision -- matching
`test_minion_population_is_close_to_the_server`'s own conditions exactly).
Confirmed complete: the log's own shutdown line reads
`LANERL_TPS ... gametime 596.5s`.

```
grep -c '^MRT ' instance000.log                # 1167
grep '^MRT ' instance000.log | grep -c 'cfh=1' # 368
grep '^MRT ' instance000.log | grep -c 'cfh=0' # 799
grep '^MRT ' instance000.log | grep -oP 'to=\S+' | sort | uniq -c
#   1134 to=LaneMinion
#     33 to=LaneTurret
#      0 to=Champion    -- consistent: no champion orders, so it is never
#                          in anyone's acquisition range.
```

## 4. Why the rate checks out but the aggregates still regress: total damage is basically unchanged

If the rate is close to right, why does median idle-lane population still
rise 22 -> 26? The first candidate the task names is windup/`time_since_attack`
disruption. Measured directly (`cfh_windup_interrupt.py`, idle lane, 600 s,
seed 0):

| | cfh OFF | cfh ON |
|---|---:|---:|
| switches (old target still valid) | 701 | 1,436 |
| ...of which mid-windup (`is_attacking` true) | 191 | 371 |
| total damage dealt (sum of all HP loss) | 62,686.3 | 64,414.0 (+2.8%) |
| minion deaths | 156 | 149 (-4.5%) |
| mean hp_frac of alive minions at t=600s | 0.937 | 0.893 |
| frac at full HP | 0.773 | 0.793 |

Two things this rules out:

* **Total damage output is not reduced.** cfh ON deals slightly MORE total
  damage, not less. Whatever is driving the population rise, it is not
  "minions stop landing hits."
* **The windup-interruption bug is real but pre-existing, not cfh-caused.**
  `lanerl_jax/sim/autoattack.py`'s own docstring says cancellation-on-retarget
  is "not modelled here," and the server itself locks a swing's target at
  CAST time (`Spell.cs:1010`, `CastInfo.Owner.AutoAttackHit(CastInfo.Targets[0].Unit)`)
  in a way a later `SetTargetUnit` call cannot touch (`ObjAIBase.cs:974-992`
  only writes the `TargetUnit` field and fires a network notification -- no
  windup/cooldown state). Our `step.py` instead re-reads `target` at
  hit-RESOLUTION time (`tgt = jnp.clip(target, ...)`, reused for both the
  swing gate and the damage-attribution block), so a swing that started
  against A and gets retargeted to B before it resolves lands on B, not A.
  191 of these already happen with cfh OFF (ordinary retargets on death/
  leave-range/give-up can also interrupt a windup); cfh very nearly doubles
  the count (371, 1.9x) but does not measurably change total output. **A
  real, separately-fixable bug** (capture the target at swing start, not at
  resolution), independent of the call-for-help decision, and not the driver
  of the population effect -- flagged, not fixed, out of scope here.

## 5. What actually changes: not "how much" damage, but WHO it lands on

Reproduced `test_minion_population_is_close_to_the_server`'s own methodology
exactly (same 18,000-tick loop, samples every 300 ticks, median/imbalance
over the last 80%) with the toggle (`idle_population_repro.py`), confirming
`1b09f16`'s finding on this checkout:

| | cfh OFF | cfh ON | server (`SERVER_IDLE`, `test_lane.py:42`) |
|---|---:|---:|---:|
| median total live minions | 22.0 | 26.0 | 21 |
| p95 / max | 28 / 31 | 36 / 40 | 27 / 30 |
| mean \|blue-red\| imbalance (this script's definition) | 4.21 | 11.83 | 2.6 (`1b09f16`) |
| blue outer turret destroyed by t=600s | No | **Yes** | No |
| final population (blue, red) | (17, 5) | **(1, 28)** | -- |

**This is a full reversal of which side wins, not a noisy amplification.**
Without cfh, blue ends the episode dominant (17 vs 5). With it, red
overwhelms blue almost completely (28 vs 1) and takes blue's outer turret.
And it is **completely deterministic**: reran seeds 1, 2, 3 (same script,
`init_lane(patch, seed=N)`) and got byte-identical results to seed 0 on
every field, both with and without cfh. This is not a fluke of one draw --
`init_lane`'s `seed` evidently randomizes nothing that matters once no
champion is ever ordered (spawn timing, wave composition and combat
resolution order are apparently deterministic given a fixed initial state),
so this fully idle scenario has exactly ONE trajectory per toggle setting on
this config, and reporting a "mean" or "median" from it is a summary of one
deterministic curve's own sampled time series, not a distribution over
independent draws. Worth stating plainly since the task background quotes
these as if they were population statistics.

**Why one side wins, and why cfh flips it:** `test_lane.py`'s own history
already documents this lane as sitting at an unstable, asymmetric
equilibrium independent of call-for-help -- its docstring records that
removing red's spawn-position bug ("a genuine 1.37 s head start... red then
won HARDER") tipped the SAME kind of full-population outcome the wrong way
before, from an unrelated fix. Call-for-help is, by design, a
**reinforcement mechanic**: every landed hit can raise a call
(`ObjAIBase.TakeDamage`, ported faithfully in `call_for_help_map`), and every
call recruits a nearby ally into that same fight. That is a positive-feedback
loop keyed on "whoever is already landing more hits gets more help," which
is exactly what an unstable equilibrium cannot absorb safely: whichever side
has ANY edge (here, an existing, independently-documented one favouring red)
gets that edge compounded by recruiting more allies into every fight it is
already winning, rather than each duel resolving independently. The server
has the same mechanic and the same rate (part 2), so this is not "our port
recruits too eagerly" -- it is that a genuinely faithful, correctly-rated
reinforcement channel is a bad thing to add to a system that was already
fragile for unrelated reasons. `SERVER_IDLE`'s own numbers (median 21, p95
27, max 30 -- a narrow band, no equivalent of a 1-vs-28 rout) say the real
server's lane does NOT blow up like this, which likely means some other,
still-uncorrected asymmetry in this port (position, timing, or collision
details -- `test_lane.py`'s docstring lists several already found and fixed,
and does not claim the list is closed) is what lets the reinforcement loop
find purchase here that it apparently does not find on the server.

## 6. Why the champion scenario is different

`StandInWave`'s response (perturbed − baseline, same engine, 10 min, seed 0;
cfh-OFF numbers re-verified against the pilot JSON on disk, cfh-ON numbers
freshly reproduced end-to-end with this session's toggle and matching
`9d59809`'s claimed figures exactly):

| metric | cfh OFF (rms) | cfh ON (rms) | server (rms) |
|---|---:|---:|---:|
| n_minions_blue | 14.33 | 4.05 | 1.46 |
| n_minions_red | 10.71 | 3.36 | 1.32 |
| lane_frac_all | 0.194 | 0.038 | 0.011 |
| lane_frac_blue | 0.287 | 0.061 | 0.028 |
| lane_frac_red | 0.269 | 0.062 | 0.027 |
| turret_hp_blue | 969.3 | **0.0** | 301.7 |
| turret_hp_red | 1061.0 | 384.9 | 214.2 |

Here cfh moves 6 of 7 metrics substantially TOWARD the server -- the
release-from-champion mechanism (`docs/TARGET_ACQUISITION_DIFF.md`) doing
exactly what it was built for, damping the sim's 3-10x over-reaction to a
champion standing in the wave. **The 7th, confirmed by reading the raw
samples (`sim_stand_in_wave_true_cfhon.json`): blue's outer turret sits at
1550/1550 HP at every one of the 300 samples, both baseline and perturbed.**
Not saturation (dead in both) -- untouched in both. `rms=0.0` "wins" against
the server's 301.7 only because the sim's wave never reaches the turret at
all under cfh, which is a different failure, not a fidelity improvement.

Why this scenario looks so different from part 5's rout: a single champion
is a bounded, durable "sink" -- one unit, very high HP, that does not die and
does not itself raise NEW calls for help at anywhere near a minion's rate.
Rescuing minions off it is a small number of LOW-frequency, self-limiting
events (docs/TARGET_ACQUISITION_DIFF.md: 47 champion acquisitions in a full
600 s of active farming, only 1 of them cfh-driven). It cannot snowball the
way part 5's minion-vs-minion reinforcement loop can, because there is only
one champion to rescue minions FROM, and rescuing them does not make the
champion stronger or weaker. The champion scenario mostly exercises the
bounded, corrective side of call-for-help; the idle lane mostly exercises the
unbounded, recruiting side.

## 7. Other refusals in `FoundNewTarget` -- checked, not missing

The task asks specifically whether the turret gate is the only refusal we
are missing. `IsValidTarget` (`LaneMinionAI.cs:125-136`) has one more:
`!UnitIsProtectionActive(u)` (structure "tower protection," `ApiFunctionManager.cs:1082`,
backed by `ProtectionManager`). Checked against `Content/LeagueSandbox-Scripts/Maps/Map1/LevelScriptObjects.cs:397-446`
(`LoadProtection`): protection is registered ONLY for the nexus (depends on
nexus turrets + all inhibitors), inhibitors (depend on their inhibitor
turret) and INNER/INHIBITOR/NEXUS turrets -- **`TurretType.OUTER_TURRET` has
no branch in that loop and is never added to `_protectedElements`**.
Every run measured here and in `docs/TARGET_ACQUISITION_DIFF.md` stays
confined to the top lane over a 600-1200 s window (the only turret ever destroyed or
even engaged in any of these runs is the single top-lane outer turret; a push
all the way to an inner turret, inhibitor or nexus does not happen in this
horizon), so the only turret that is ever actually a targeting candidate here
is the outer one -- which is provably never protected on
this map. `UnitIsProtectionActive` is a real refusal we do not implement,
and confirmed structurally unreachable for this config -- not a gap that
explains anything measured here.

The turret gate itself (`on_turret`, `minion_ai.py:218-221`) is confirmed
sufficient and correctly scoped: reading `LaneMinionAI.cs:173-176` and
`:317-343` together, the turret check only matters for the RESTRICTED
(`handleOnlyCallsForHelp=true`) scan, because the regular scan is only ever
reached from `ReevaluateBehavior` when `targetIsStillValid` is already
false (`:323-336`) -- the ordinary path can never reach a turret-holding
incumbent's `FoundNewTarget()` call at all. So the gate is exclusively a
call-for-help concern, exactly where this port puts it.

## 8. Verdict

* **Ship**: `enable_call_for_help` as a toggle on `tick()`/`step_decision()`/
  `run_sim_episode()`, default `False`. Existing behaviour is unchanged for
  every current caller and every existing test (169 pass).
* **Do not** flip the default on. The port is faithful -- broadcast rule,
  radius semantics, one-shot lifetime, turret gate, and now the switch RATE
  (part 2) all check out against file:line citations. What blocks it is a
  genuine, mechanism-level finding (part 5): on this lane's specific,
  independently-documented unstable equilibrium, a correctly-implemented
  reinforcement channel amplifies a pre-existing asymmetry into a rout
  instead of the server's own bounded oscillation. That is not fixed by
  tuning call-for-help itself -- the fix, if there is one short of leaving it
  off, is finding and closing whatever residual asymmetry (position, timing,
  collision) currently lets one side's advantage compound instead of
  self-correcting, which `test_lane.py`'s own docstring already says has not
  been fully run to ground ("three separate corrections... not claim the
  list is closed"). Turning call-for-help on is genuinely closer to the
  server for a champion-in-lane scenario and genuinely further for an idle
  one; per the project's own rule, a change that helps one measured scenario
  and clearly hurts another measured scenario is not a net improvement to
  ship, so it stays off.
* A real, separate, orthogonal bug was found along the way (part 4): a
  completing auto-attack swing attributes damage to whatever `target` is at
  RESOLUTION time rather than the target it was cast against, unlike the
  server. Pre-existing (191 events/600s even with cfh off), does not appear
  to move total damage output, not fixed here -- flagged for a future pass.
