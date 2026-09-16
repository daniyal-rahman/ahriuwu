# Target-acquisition diff: why the JAX champion dies and the server champion doesn't

**Bottom line.** The target-acquisition hypothesis is confirmed, but not the way
the background section framed it (minions/turret "more interested" in the
champion via some acquisition-side difference). Raw acquisition behaviour is
close to parity on both engines. What's missing is the **release**
mechanism: `lanerl_jax/sim/step.py` never emits the call-for-help event that
lets a nearby ally rescue a minion off a live target, so once a JAX minion
locks onto the standing champion, nothing short of its own death, the
champion's death, or the champion walking out of range ever takes it back off.
On the server, that rescue channel is what pulls a stray minion off the
champion and back onto the wave, usually within single-digit seconds. Absent
that channel in the sim, independently-acquired lock-ons pile up instead of
being continuously recycled, producing sustained multi-minion focus (mean 4.1,
peak 12, simultaneous attackers) that the server's population structurally
cannot sustain.

Also resolved: the turret is **not** part of the current gap. The harness
already fixed the bug that parked the champion inside enemy turret range
(`lanerl_jax/parity/last_hit_drive.py`'s `APPROACH_WAYPOINTS` module docstring);
under the current waypoint the champion sits 1,248 units from the red outer
turret, outside its 750 range, and both engines show **zero** turret-on-champion
ticks over the full 600 s episode. The "turret on 1,217 ticks" figure in the
task background predates that fix and does not apply to the current code.

## Methodology

One server run and one sim run, both driving the same scripted policy
(`lanerl_jax.parity.last_hit_drive.run_oracle_in_sim` /
`run_oracle_on_server`, `DECISIONS_600S` = 18,000 decisions @ 30 Hz = 600 s),
with `LANERL_AGGRO_TRACE=1` and `LANERL_TURRET_TRACE=1`
(`lanerl_jax/parity/targets.TRACE_ENV`) added to the server's
`ServerLaunchSpec.extra_env`. Config: `toponly=True`, `bot_teams="none"`,
`bot_seed=4242`, `step_ticks=2`, `lanerl/cfg/garen1v1.json` (`"map": 1`,
confirmed at line 132) -- the same config the whole project trains and evals
against, so every server-side citation below is reached by *this* map, not
just present somewhere in the Content tree.

Sanity check against the task's own baseline: this fresh server run reproduced
`cs=4, attacks=86` exactly; the sim run reproduced `cs=7, attacks=135, deaths=4`
exactly. The server run's death count (see below) differs from the task's
quoted "0" -- treated as data, not dismissed (see "One server death" below).

Instrumentation scripts (not committed, not part of the source tree):
`/tmp/claude-1000/-srv-nfs-projects/6c72189a-2818-4135-9f1e-aaa397df05ce/scratchpad/sim_target_trace.py`
and `server_target_trace.py`, raw outputs alongside them
(`sim_target_trace_out.json`, `server_target_run.log`,
`server_target_summary.json`, `id_team.json`).

* **Sim side**: exact ground truth. `LaneState.target` / `target_priority`
  read directly after every decision's `step_decision` call (30 Hz, same
  cadence the policy itself runs at). Champion is slot 0 (`init_lane`,
  `lanerl_jax/sim/init.py:268-272`, `for i, t in enumerate((Team.BLUE,
  Team.RED))`); the red outer turret was located at runtime by nearest-match
  to `TOP_OUTER_TURRET[Team.RED]` and landed at slot 54, position
  `(3911.7, 13654.8)`, matching the constant exactly.
* **Server side**: `LANERL_AGGRO_TRACE` (`MRT id= lt= from= to= cfh= held=
  fromprio= toprio=`, `LaneMinionAI.cs:243-251`) and `LANERL_TURRET_TRACE`
  (`LANERL_TURRET t= turret= team= target= ttype= tteam= d=`,
  `LanerlHooks.cs:553-582`), parsed with
  `lanerl_jax.parity.targets.parse_target_traces`. `MRT id=` is the *acting*
  minion's NetId, not its team, so minion team was resolved from the
  control-channel observation stream (`tm` field) collected in parallel over
  the same run and cross-referenced by NetId.

## 1. Server side

47 `MRT ... to=Champion` events over the 600 s episode, **all 47** from RED
minions acquiring the BLUE (oracle-driven) champion -- 0 events the other
direction (the RED champion never moves and is never in anyone's acquisition
range).

| fromprio | meaning | count | % |
|---|---|---:|---:|
| 14 (DEFAULT) | had no target at all | 26 | 55.3% |
| 9 (MELEE_MINION) | had a melee minion, which then became invalid | 12 | 25.5% |
| 8 (CASTER_MINION) | had a caster minion, which then became invalid | 5 | 10.6% |
| 3 (MINION_ATTACKING_MINION) | had a call-for-help-boosted minion target | 4 | 8.5% |

| toprio | meaning | count | % |
|---|---|---:|---:|
| 11 (CHAMPION, base) | plain acquisition, no call-for-help boost | 46 | 97.9% |
| 5 (CHAMPION_ATTACKING_MINION) | call-for-help: champion just hit a minion | 1 | 2.1% |

`cfh=1` (i.e. the switch came from the `FoundNewTarget(true)` trigger,
`LaneMinionAI.cs:83`, not the regular 250 ms sweep): **1 of 47** (2.1%) --
`MRT id=1073750709 lt=33149 from=LaneMinion to=Champion cfh=1 held=133
fromprio=8 toprio=5`. This is the ordinary "the champion just autoattacked a
minion, nearby allies retaliate" mechanic; it is rare here because the policy
only attacks when it can secure a kill (86 attacks in 600 s).

**Every documented departure from the champion returns to a minion.** 28 of
the 47 acquisitions have a later MRT line for the same NetId (i.e. we
observe them leaving); **28/28 (100%) leave `to_kind=LaneMinion`** -- never a
timeout/give-up, always a re-acquired minion. Hold time before that rescue
(`held_ms` on the departure line): min 283 ms, median 2,499 ms, mean 6,214 ms,
max 22,451 ms, **sum 174.0 s** across the 28 closed streaks. This is a *lower
bound* on total champion exposure time: 19 of the 47 acquisitions have no
later MRT line in the trace, either because they were still holding at
episode end or because the attacking minion itself died mid-hold (which stops
its `OnUpdate` -- `LaneMinionAI.cs:64`, `!LaneMinion.IsDead` -- and therefore
stops it from ever printing a departure line). Extrapolating the closed
streaks' rate across all 47 gives ~292 s as a rough upper estimate; the true
figure lies between 174 s and ~292+ s and is not exactly recoverable from
these traces alone -- a real limit of this observability path, reported as
such rather than forced to a point estimate.

**Turret**: 66 `LANERL_TURRET` change-lines across all 24 turrets (one
`target=none` baseline per turret at `t=233`, plus 42 further changes over the
game). **Zero** of them ever show `ttype=Champion tteam=100` (nor `tteam=200`
for the reverse) -- every turret engagement in this run is turret-vs-minion.
Confirmed by direct grep of the log (`server_target_run.log`).

**One server death.** The champion died once, at `t=528557` ms (~528.6 s into
the 600 s episode; confirmed via the champion's `hp` field in the recorded
control-channel observation stream, sampled every 10 decisions). This is the
one place this measurement disagrees with the task's quoted "server dies 0
times": under this exact scripted policy, on this exact config, with the
traces on, the server's champion *can* die -- just roughly 4x less often than
the sim's over the same 600 s (1 vs 4 here; 0 vs 4 in the task's own earlier,
untraced measurement). Reported as data: the server is not perfectly immune to
this mechanism, it is far less exposed to it, which is the more precise
version of the hypothesis than "the server never dies."

## 2. Sim side

61 acquisitions of the champion by a RED minion over the same 600 s. **All
61** have `fromprio=14` (no prior target -- the "nothing else valid nearby"
path) and **all 61** have `toprio=11` (plain base `CHAMPION` priority, never
call-for-help-boosted). Zero of the server's other three patterns
(minion-had-a-real-target-that-died-first, or a genuine call-for-help switch)
ever occur in the sim -- see the diagnosis below for why that is not a
coincidence.

| metric | value |
|---|---:|
| decision-ticks with >=1 RED minion targeting champion ("active") | 3,724 / 18,000 (20.7%) |
| ...as % of the 12,453 post-handover decisions | 29.9% |
| minion-decision-ticks targeting champion (sum, can double count concurrent) | 15,419 |
| ...as % of all RED-minion-alive decision-ticks (244,376) | 6.31% |
| mean simultaneous attackers, when active | 4.14 |
| max simultaneous attackers, any decision | **12** |
| total exposure (unit-ticks / 30 Hz) | 513.97 s (of 600 s) |
| acquisitions (switches onto champion) | 61 (6.1/min) |
| red outer turret (slot 54) targeting champion | 0 / 18,000 decisions |
| champion deaths | 4 |

## 3. Comparison (normalised)

| | sim | server | ratio (sim/server) |
|---|---:|---:|---:|
| acquisitions / min | 6.1 | 4.7 | 1.3x |
| acquisitions with no alternative available (fromprio=14) | 100% (61/61) | 55.3% (26/47) | -- |
| acquisitions that are call-for-help-driven (cfh=1) | **0%** (0/61, structurally) | 2.1% (1/47) | -- |
| documented "rescued back onto a minion" departures | **0** (mechanism does not exist) | 28 (100% of observed departures) | -- |
| mean hold before a documented departure | n/a (never happens) | 6.2 s | -- |
| total champion exposure time (minion-seconds) | 514.0 s (exact) | >=174.0 s (lower bound; ~292 s extrapolated) | ~1.8-3.0x |
| mean simultaneous attackers when active | 4.14 | not reconstructable from these traces (see caveat) | -- |
| max simultaneous attackers | 12 | not reconstructable | -- |
| turret-on-champion ticks | 0 | 0 | 1:1 (both zero) |
| deaths / 600 s | 4 | 1 (this run) / 0 (task baseline) | 4x |

The acquisition *rate* gap (1.3x) is real but modest, and does not by itself
explain a 4x death gap. The exposure-time gap (~2-3x) is larger. Neither is as
stark as the concurrency numbers the sim alone can show (mean 4.1, peak 12
simultaneous attackers) -- and concurrency, not raw exposure time, is what
actually kills a champion through burst damage rather than sustained chip
damage regen can outpace. The server-side traces cannot directly measure
concurrency (see caveat below), but the 100%-rescued departure pattern is
strong indirect evidence that the server rarely lets more than one or two
minions sit on the champion at once, because each is a candidate for rescue
the moment a nearby ally takes a hit.

## 4. Diagnosis

**Branch: the call-for-help scan (`FoundNewTarget(true)`), not the 250 ms
sweep, not the give-up timer, not the acquisition-range test.**

Ruled out first, with numbers:

* **Acquisition-range test** -- matches. `lane_params()` resolves
  `melee_red`/`cannon_red` to the 600.0 fallback and `caster_red` to 700.0
  (`lanerl_jax/sim/profiles.py:91`, `u.acquisition_range or 600.0`), the same
  defaults `_legacy_lane_params` used and the same values the wiki/Content
  data give for standard lane minions. Confirmed by direct read of the built
  `lane_params()` table (melee_red row: 600.0; caster_red row: 700.0;
  cannon_red row: 600.0). If this were the driver, raw acquisition *counts*
  would diverge far more than the observed 61 vs 47 (1.3x).
* **250 ms re-evaluation cadence** -- matches. `ACTION_TIMER_MS = 250.0`
  (`lanerl_jax/sim/minion_ai.py:107`) against `minionActionTimer >= 250.0f`
  (`LaneMinionAI.cs:83`); the port's `run = me & (just_died | cfh_switch |
  (timer >= ACTION_TIMER_MS))` (`minion_ai.py:217`) even preserves the
  server's own short-circuit -- when `keep` (still-valid incumbent) is true,
  the 250 ms sweep re-confirms `AttackTo` **without** re-scanning candidates,
  exactly mirroring `ReevaluateBehavior`'s `if(targetIsStillValid) { ...
  return OrderType.AttackTo; }` (`LaneMinionAI.cs:323,333`) short-circuiting
  before it ever reaches `FoundNewTarget()` (`LaneMinionAI.cs:342`). This is
  why a JAX minion committed to a real minion target is never pulled off it
  by the plain sweep either -- correct, and not the bug.
* **Give-up-after-4s rule** -- matches, and not implicated. A minion that is
  actively landing hits on the stationary champion has `is_attacking=True`
  every tick, which resets `time_since_attack` to 0
  (`lanerl_jax/sim/minion_ai.py:163-167`, mirroring `LaneMinionAI.cs:66-73`),
  so `GIVE_UP_MS = 4000` never fires while the champion just stands there and
  gets hit. Both engines behave identically here: none of the server's 28
  observed departures are give-ups (`to_kind` is `LaneMinion` in all 28, never
  a give-up-then-idle pattern).
* **The call-for-help scan is provably dead in `tick()`.** `LaneState.help_priority`
  is an `(N, N) int8` field (`lanerl_jax/sim/state.py:229`) initialised once
  to all-`14` (`ClassifyUnit.DEFAULT`, "no call for help registered") at
  `state.py:314`, and correctly *consumed* every tick by
  `step_minion_ai` -- `prio = jnp.where(help_priority < DEFAULT,
  help_priority, base_prio_broadcast)` (`minion_ai.py:173-174`) and the whole
  `cfh_mask`/`cfh_switch` block (`minion_ai.py:194-215`), which is the exact
  port of `FoundNewTarget(true)`'s `unitsAttackingAllies`-restricted scan
  (`LaneMinionAI.cs:157-250`, specifically the branch at `:206-210`). But
  **nothing in the codebase ever writes a value below `DEFAULT` into
  `help_priority`.** `lanerl_jax/sim/step.py:246` is the *only* other place
  the field is touched, and it's a read (`help_priority=state.help_priority`,
  passed straight through to `step_minion_ai`). The tick's damage-application
  block (`step.py:338-379`, `dmg_ij` computed at `:371-375`, the
  attacker/victim damage matrix -- exactly the data a `TakeDamage`-equivalent
  broadcast needs) does not touch `help_priority` at all, and the final
  `state.replace(...)` at `step.py:473` does not include it among the fields
  it updates, so it is carried forward unchanged, tick after tick, for the
  entire episode. `help_priority_for` (`lanerl_jax/sim/targeting.py:109`),
  the pure function that computes exactly the value the server's
  `ClassifyTarget(attacker, victim)` would (verified against
  `ObjAIBase.cs:388-440` line by line -- same six cases, same priorities), is
  defined, exported, and **never called anywhere outside its own docstring
  and the isolated unit tests** (`lanerl_jax/sim/tests/test_minion_ai.py:86,
  150, 169`, which construct `help_priority` by hand and call
  `step_minion_ai` directly, bypassing `tick()` entirely). This is not a
  booked/documented approximation -- `minion_ai.py`'s own "Booked
  approximations" section lists two unrelated ones (`WaypointReached`
  collision expansion, the `attackers` term) and says nothing about
  call-for-help, so this reads as an unfinished wiring, not a deliberate
  simplification.

  Practical consequence, confirmed by the data above: on the server,
  `ObjAIBase.TakeDamage` (`ObjAIBase.cs:1127-1155`) fires `OnCallForHelp`
  (`LaneMinionAI.cs:108-121`, `unitsAttackingAllies[attacker] =
  Math.Min(existing, ClassifyTarget(attacker, victim))`) on every landed hit
  between two units near the same wave clash -- constant, since minions are
  attacking minions throughout an active fight. That is the **only** channel
  (besides the target dying or leaving acquisition range) that can dislodge a
  minion from a valid incumbent, because `ReevaluateBehavior`'s own sweep
  short-circuits before reaching `FoundNewTarget()` whenever the incumbent is
  still valid. It is why 28/28 documented server departures from the champion
  go straight back to `LaneMinion` (median 2.5 s later): the instant a nearby
  ally takes a hit from a minion at `MINION_ATTACKING_MINION` priority (3),
  that beats the champion's plain base priority (11), and the call-for-help
  scan (which runs even while the incumbent target is valid, per the trigger
  condition at `LaneMinionAI.cs:76-93`) rips the minion back onto the wave.
  In the sim, `cfh_mask` (`minion_ai.py:194`) is `help_priority < DEFAULT`,
  which is false for every entry, every tick, by construction -- so
  `cfh_switch` (`:206`) is always false, the rescue never fires, and a minion
  that acquires the champion (via the "nothing else in range" path, which
  *does* work correctly and fires at a comparable rate to the server's, 61
  vs 47) simply keeps attacking it until the champion dies, walks away, or the
  minion itself dies. Independent acquisitions from different minions
  therefore accumulate instead of being continuously recycled back into the
  wave -- which is exactly what turns a 1.3x higher acquisition rate and a
  1.8-3x higher total exposure time into a mean-4.1/peak-12 simultaneous-
  attacker pile-on and a 4x death rate.

**Where the fix would go (not applied here):** `lanerl_jax/sim/step.py`,
inside the damage-application block, after `dmg_ij` is computed
(`step.py:371-379`) and before the final `state.replace(...)`
(`step.py:473`). It needs two things, both currently absent:

1. For each landed hit `dmg_ij[attacker, victim] > 0` (plus missile hits,
   `ms.damage_ij`, and `bs.damage_dealt` for Judgment, to match everything
   `ObjAIBase.TakeDamage` reacts to), find allies of `victim` within
   `AcquisitionRange` of *both* `victim` and `attacker` (mirroring
   `ObjAIBase.cs:1137-1145` -- note it is the *victim's own*
   `AcquisitionRange` gating both distance checks, not the ally's), and set
   `help_priority[ally, attacker] = min(existing, help_priority_for(kind[attacker],
   kind[victim]))` using the already-written, already-correct
   `lanerl_jax/sim/targeting.py:109` function.
2. Clear each minion's own `help_priority` row on its own 250 ms
   reevaluation, mirroring `callsForHelpMayBeCleared` /
   `unitsAttackingAllies.Clear()` (`LaneMinionAI.cs:99-104`). `step_minion_ai`
   currently only ever reads `help_priority`; nothing clears stale entries
   either, so populating it without also clearing it would just trade one
   parity bug for a "call-for-help memory that never expires" one.

## What the data does *not* support

* **Turret aggro is not part of the current gap** (see above) -- both engines
  show exactly zero turret-on-champion ticks under the current
  `APPROACH_WAYPOINTS`. The task background's "turret on 1,217 ticks" figure
  is from before that waypoint fix and should not be treated as current.
* **Acquisition range is not the driver** -- values match, and raw
  acquisition-count parity (61 vs 47, 1.3x) is far tighter than the 4x death
  gap it's being asked to explain.
* **The 250 ms cadence and the 4 s give-up rule are not the driver** -- both
  are faithfully ported and behave identically in the one case that matters
  here (a target that keeps landing hits never gives up, on either side).

## Caveats / what could not be measured exactly

* **Server-side simultaneous-attacker count is not reconstructable from these
  traces.** `LANERL_AGGRO_TRACE` logs transitions, not a per-tick target
  census, and `LanerlStateDump` carries no `TargetUnit` at all (the premise of
  this task). Reconstructing concurrency would require correlating every
  RED minion's own local AI clock (`lt=`, which starts at 0 at that minion's
  own spawn, not at a shared game clock) against its real spawn time from the
  observation stream, for the full ~106-minion population, then replaying
  each one's held-target timeline. Not attempted here; the 28/28
  always-rescued departure pattern and the sub-10s median hold are offered as
  strong indirect evidence of low concurrency, not a replacement for a real
  count.
* **Total server exposure time is bounded, not exact** (174.0 s to ~292 s
  extrapolated), because 19 of 47 acquisitions have no observed departure
  line in the trace -- either still active at episode end, or ended by the
  attacking minion's own death (which silently stops its `OnUpdate`, per
  `LaneMinionAI.cs:64`, and therefore stops it from ever printing a
  departure line).
* **The one server death (528.6 s)** was cross-checked only against the
  champion's `hp` field in a 10-decision-thinned observation stream (333 ms
  resolution); the exact tick of death is not pinned down further, since
  that level of precision was not needed to answer the target-acquisition
  question.
