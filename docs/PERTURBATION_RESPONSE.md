# Perturbation-response parity: pilot results

**Headline number, first as instructed: the null-control floor is exactly
zero.** Two independent runs of the identical script, same seed, same engine,
produced byte-identical metric samples at every one of 300 time-grid points,
on *both* the sim and the server. There is no re-run noise to clear at this
10-minute horizon on either side. The one real perturbation tested
(`StandInWave`, Garen stands in the enemy wave for 5 s) produced a response
that is nonzero starting at the first sample after the champion physically
arrives at the wave (not before), on both engines, comfortably above that
zero floor. **The methodology works at 10 minutes.** See "What this does not
yet show" at the end for what a 10-minute pilot cannot tell you.

## What was built

`lanerl_jax/parity/perturbation.py` (new file, not committed, not part of any
prior work). Reusable pieces:

* `lane_fraction(x, y)` -- vectorized nearest-segment projection onto
  `lanerl_jax.sim.init.TOP_LANE_PATH`, arc-length fraction in `[0, 1]`,
  direction-agnostic (a red minion on the reversed polyline and a blue minion
  on the forward one land at the same fraction for the same physical spot).
  No such helper existed anywhere in the tree already (checked
  `lanerl_jax/parity/sim_vs_server.py` and `lanerl_jax/obs/frame.py`; the
  latter's `(s, n)` lane frame is turret-to-turret and used for the
  observation encoding, not a scalar "how far down the lane" summary) -- this
  is a new, self-contained implementation.
* `UnitRecord` / `sim_units()` / `server_units()` -- a canonical
  `(kind, team, x, y, hp)` view of every live unit, built two ways (one per
  engine) into one shared shape.
* `compute_metrics()` -- the three response metrics the task specifies: mean
  lane fraction of live minions (overall and per team), per-team live minion
  counts, and both outer turrets' HP (matched to
  `lanerl_jax.sim.init.TOP_OUTER_TURRET` by nearest position; a turret absent
  from the live-unit list is reported as destroyed, HP 0 -- turrets are
  static, so "no live `LaneTurret` within 50 units of the known position" can
  only mean it died).
* Three `Perturbation`s: `NullControl`, `StandInWave`, `KillMinions` (the
  third and a fourth suggestion from the task; see "Perturbations built vs
  exercised" below for why only two of the three ran in this pilot).
* `run_sim_episode()` / `run_server_episode()` -- one decision loop per
  engine, both driven by the same `Perturbation.decide(scene, perturbed,
  script)` call, so the protocol really is identical code on both sides, not
  just "the same idea implemented twice."
* `response()` / `summarize_response()` -- `R = f(perturbed) - f(baseline)`,
  same engine, aligned by sample index, plus max-abs/RMS summaries.
* A CLI (`python -m lanerl_jax.parity.perturbation --engine ... --perturbation
  ... [--perturbed] --seed ... --minutes ... --out out.json`) that runs
  exactly one episode and writes its `ResponseCurve` to JSON. The pilot below
  was driven by shelling out to this CLI once per episode rather than
  importing the module and looping in-process, specifically so that server
  episodes are trivially run one at a time (the hard hardware constraint) and
  so a crash in one episode cannot take down a batch.

## A cleaner formulation than the task's own suggested data source

The task points at `docs/TICK_DIVERGENCE_TRACE.md` and
`lanerl_jax/parity/trace.py` for how to get server state: parse
`LANERL_STATE_DUMP_FULL=1`'s per-tick log. That machinery exists for a
different job (bit-exact per-tick diffing against the sim) and needs a
log-file round trip. This module does not use it. `LanerlControl`'s own
observation -- already read every decision to drive the champion, via
`lanerl_jax.parity.last_hit_drive.run_oracle_on_server`'s `obs["u"]` -- already
carries every live `AttackableUnit`, not just champions: `k` (kind, including
`"LaneMinion"` and `"LaneTurret"`), `tm` (team), `x`, `y`, `hp`. Dead units
are simply absent (the engine removes them from `ObjectManager`), so no
`alive` bookkeeping is needed either. That is exactly the four fields
`compute_metrics()` needs, already flowing over the channel that steps the
episode, at the decision rate the response is sampled at. Using it means one
code path per side instead of two, and it cost nothing extra to add.

## Perturbations built vs exercised in this pilot

Three perturbations are implemented, addressing the task's request for 2-3
candidates:

1. **`NullControl`** -- no perturbation, run twice. Required, and run first.
2. **`StandInWave`** -- Garen walks to `ENGAGE_POINT`
   (`TOP_LANE_PATH[5] = (2806.0, 13075.0)`, lane fraction 0.553), holds for
   `hold_s = 5` seconds starting at `trigger_ms = 180_000` (3 min), then walks
   back to `CAMP_POINT` (`TOP_LANE_PATH[2] = (861.0, 6459.0)`, lane fraction
   0.218) and holds there for the rest of the episode. **Chosen as the
   pilot's one real perturbation.**
3. **`KillMinions`** -- Garen focus-fires the 3 nearest enemy minions at a
   fixed trigger time. Implemented and smoke-tested in the sim only (140 s,
   no crash, directionally sensible: killing red minions early reduced red's
   live count and cost red's outer turret ~264 HP by t = 138 s that the
   baseline never lost) -- **not run on the server, not part of this pilot's
   headline numbers.**

Why `StandInWave` over `KillMinions` for the pilot, in order of importance:

* **Pure positioning.** Every order `StandInWave` issues is `move` or a
  `noop` hold; nothing depends on either engine's attack-damage model,
  windup timing, or hit count to kill a minion.
  `lanerl_jax/parity/last_hit_drive.py`'s own "KNOWN ASYMMETRY" notes
  document that the sim's champion attack damage does not scale with level
  while the server's does -- exactly the kind of engine-specific detail a
  kill-count-based perturbation's *timing* would inherit, and exactly what
  this test is trying to avoid depending on (the response comparison is
  still valid either way, since each engine only ever races against its own
  baseline, but a perturbation whose own mechanics are asymmetric between
  engines is a worse choice for a first methodology check).
* **Directly provokes the one confirmed real gap.**
  `docs/TARGET_ACQUISITION_DIFF.md` found the sim has no call-for-help rescue
  channel: on the server, a minion that starts attacking a standing champion
  gets pulled back onto the wave within seconds, 28/28 observed departures
  in that study; in the sim, nothing ever pulls it back. Standing a champion
  in the enemy wave is exactly the scenario that exercises that channel on
  one engine and cannot on the other. If a 10-minute response test has power
  to detect anything, this is the perturbation most likely to show it.
* **Cheapest to trust.** `lanerl_jax/parity/movement_parity.py` already
  checks move-order semantics in isolation, so `StandInWave` adds no new
  "does an order mean the same thing on both engines" surface beyond what is
  already covered elsewhere.

`KillMinions` is left in the tree, implemented, for a follow-up pass once the
response methodology itself (validated here) is trusted.

## Setup

* Config: `lanerl/cfg/garen1v1.json`, `"map": 1` (confirmed at line 132).
  Both players are Garen (`"champion": "Garen"` for both `playerId: 1` and
  `2`), matching `docs/TARGET_ACQUISITION_DIFF.md`'s methodology note.
* `ServerLaunchSpec(toponly=True, bot_teams="none", bot_seed=4242,
  step_ticks=2)` -- 30 Hz decisions, no jungle/other lanes, no in-server bot
  driving either champion. Same boot as `last_hit_drive.run_oracle_on_server`
  and `docs/TICK_DIVERGENCE_TRACE.md`'s idle-lane trace.
* Sim: `init_lane(seed=0)`, same `step_ticks=2` cadence via
  `lanerl_jax.sim.step.step_decision`.
* **Only blue (sim slot 0 / wire `tm == 100`) is ever ordered.** Red never
  receives an order, matching `docs/TICK_DIVERGENCE_TRACE.md` and
  `last_hit_drive.py`.
* Horizon: 10 minutes of game time (18,000 decisions), per the task's pilot
  scope -- not 40.
* Sampling: every 2 s of game time (every 60th decision), 300 samples per
  episode.
* Seed 0 throughout; baseline and perturbed share it, per the task's
  instructions.
* 8 episodes total: {sim, server} x {NullControl, StandInWave} x
  {baseline, perturbed}.
* Server episodes were run **strictly one at a time**, sequentially, via
  separate CLI invocations. `ps aux` and `uptime` were checked before and
  after the batch: no leftover `GameServerConsole` processes, load average
  back to the machine's idle baseline (0.14-0.77 throughout, nowhere near
  the 19.75 the task warns five concurrent servers produced).

## 1. Is each engine deterministic run-to-run under this harness?

**Yes, both, exactly.** For both `NullControl` pairs (sim and server), every
one of the 300 sampled metric dicts compared equal by value between the two
independent runs -- not "close," identically equal floats, at every sample.
`summarize_response()` over the null-control response accordingly reports
`max_abs = 0` and `rms = 0` for all 7 metrics, on both engines:

| metric | sim max_abs | sim rms | server max_abs | server rms |
|---|---:|---:|---:|---:|
| n_minions_blue | 0 | 0 | 0 | 0 |
| n_minions_red | 0 | 0 | 0 | 0 |
| lane_frac_all | 0 | 0 | 0 | 0 |
| lane_frac_blue | 0 | 0 | 0 | 0 |
| lane_frac_red | 0 | 0 | 0 | 0 |
| turret_hp_blue | 0 | 0 | 0 | 0 |
| turret_hp_red | 0 | 0 | 0 | 0 |

This confirms, rather than assumes, the task's cited earlier finding ("two
runs returned bit-identical counts") -- now checked directly against the
control-channel observation stream rather than the state-dump hash, and at a
full 10-minute / 18,000-decision horizon rather than whatever shorter check
produced that earlier claim.

## 2. Does a real perturbation produce a response clearly above the null floor?

**Yes, on both engines**, and the onset is causally where it should be: flat
zero through the trigger time (180 s) and the ~18-20 s walk to `ENGAGE_POINT`,
then nonzero starting at the very next sample after arrival (t = 198-200 s on
both engines -- Garen's move speed and the walk distance are close enough
between engines that arrival lands in the same 2-second sample bucket).
Nothing moves before that on either side, which is the sanity check that
matters most here: a spuriously "early" response would mean the two runs
were not actually in lock-step, and they are.

Response magnitude, `perturbed - baseline`, same engine, over the full
10-minute episode:

| metric | sim max_abs | sim rms | server max_abs | server rms |
|---|---:|---:|---:|---:|
| n_minions_blue | 30 | 14.33 | 5 | 1.46 |
| n_minions_red | 25 | 10.71 | 6 | 1.32 |
| lane_frac_all | 0.478 | 0.194 | 0.037 | 0.011 |
| lane_frac_blue | 0.666 | 0.287 | 0.087 | 0.028 |
| lane_frac_red | 0.655 | 0.269 | 0.100 | 0.027 |
| turret_hp_blue | 1550 | 969.3 | 804 | 301.7 |
| turret_hp_red | 1550 | 1061.0 | 814 | 214.2 |

Every one of these clears the null floor (0) by a wide margin -- there is no
"is this even distinguishable from noise" ambiguity at this horizon, for this
perturbation, on either engine.

**A secondary observation, worth flagging as preliminary and not
over-reading:** the *sim's* response is 3-10x larger than the *server's*
response to the identical protocol, on every metric (e.g. RMS turret HP
response 969-1061 on the sim vs 214-302 on the server; RMS
`n_minions_blue` response 14.3 vs 1.5). Directionally, this is consistent
with `docs/TARGET_ACQUISITION_DIFF.md`'s finding that the sim has no
call-for-help rescue channel and so cannot recycle a minion that fixates on a
standing champion the way the server does -- a missing restoring force would
show up as exactly this pattern, the sim over-reacting to the same nudge
relative to its own baseline. This is **not** a confirmation of that
mechanism on its own (one seed, one perturbation, one horizon, and part of
the sim's larger swing is plausibly the champion-vs-turret-range interaction
or ordinary chaotic amplification of a larger initial disturbance -- the sim
lets more minions reach the champion in the first place, see the population
counts below). It is the shape of evidence a longer, multi-seed run of this
harness would be well positioned to confirm or rule out.

### A raw-trajectory aside, reported as an aside and not as evidence

The two engines' **absolute, unperturbed** `NullControl` baselines look
qualitatively different from each other -- the sim's blue outer turret is
permanently destroyed by t &approx; 375 s and never recovers for the remaining
225 s, while the server's baseline oscillates (blue turret drops to 928 HP by
t = 300 s, recovers to 891 by t = 480 s, drops again; red similarly drops to
857 by t = 420 s and holds). This is exactly the kind of long-horizon,
sim-vs-server raw comparison this project's own design principle says proves
nothing on its own -- float32 non-associativity plus lane chaos can produce
qualitatively different absolute trajectories from a perfect reimplementation,
and this is genuinely two different (if correlated) seeds' worth of dynamics
by t = 300+ s. It is recorded here only as context for why the response
comparison above -- not this paragraph -- is the number to trust.

## 3. How long does one run take, and what does 40 minutes cost?

Wall-clock, all 8 pilot episodes, 18,000 decisions (10 min game time) each:

| engine | perturbation | perturbed | wall (s) |
|---|---|---|---:|
| sim | null_control | False | 11.9 |
| sim | null_control | True | 17.4 |
| sim | stand_in_wave | False | 21.4 |
| sim | stand_in_wave | True | 22.3 |
| server | null_control | False | 23.1 |
| server | null_control | True | 32.9 |
| server | stand_in_wave | False | 36.9 |
| server | stand_in_wave | True | 49.4 |

Sim mean 18.3 s / 10 min; server mean 35.6 s / 10 min. Total wall time for
the entire 8-episode pilot: **215 s (3.6 minutes)**, all on this 6-core
machine, servers strictly sequential.

Both sides trend upward across the batch (JIT warm-up amortizes the sim's
first run; the server's episodes get slower as the run goes on within each
episode too -- more live units later in an episode means more per-decision
serialization and simulation work), so a naive **linear** extrapolation to 40
minutes is a **lower bound**: sim &approx; 73 s, server &approx; 142 s per
run, call it 1-4 minutes each realistically. Either way this is minutes, not
hours -- the 40-minute version of this pilot, across several seeds and both
remaining perturbations, is cheap enough to run today without needing
`slurm`/`desktop`. Per the task's instructions, nothing was submitted to
slurm and no 40-minute run was executed as part of this pass.

## What this does not yet show

* **One seed.** Every number above is a single `(seed=0, bot_seed=4242)`
  draw. The null floor being *exactly* zero on one seed is a strong
  determinism result, but the *perturbation* response's magnitude on one seed
  says nothing about its variance across seeds -- a proper "does this clear
  the floor with margin" claim needs the response distribution over several
  seeds, not one point estimate on each side.
* **10 minutes, not 40.** The pilot's own premise is that this is the short
  test before the expensive one. Population and turret dynamics are still
  visibly ramping at t = 600 s on both engines (see the null-floor baseline
  table above) -- neither side has reached a steady state, so the 40-minute
  run is not just "more of the same," it is where turret trades and lane
  outcomes actually resolve.
* **Only `StandInWave` was exercised on the server.** `KillMinions` is built
  and sim-smoke-tested but has no server data point in this report.
* **The sim-over-reacts-relative-to-server observation is one data point,**
  offered as a reason to prioritize the 40-minute follow-up, not as a
  conclusion about the call-for-help gap.

## Where things are

* `lanerl_jax/parity/perturbation.py` -- the module (new, uncommitted).
* `docs/PERTURBATION_RESPONSE.md` -- this file.
* Raw pilot JSON (8 `ResponseCurve` dumps, one per episode) --
  `/tmp/claude-1000/-srv-nfs-projects/6c72189a-2818-4135-9f1e-aaa397df05ce/scratchpad/pilot/`
  (scratchpad, ephemeral, not part of the repo; re-run via the CLI examples
  below to regenerate).

Reproduce one episode:

```
JAX_PLATFORMS=cpu PYTHONPATH=/srv/nfs/projects/ahriuwu-lanerl-jax \
  .venv-jax/bin/python -m lanerl_jax.parity.perturbation \
  --engine server --perturbation stand_in_wave --perturbed \
  --seed 0 --minutes 10 --sample-every-s 2 --out out.json
```

---

# Isolation run: fog of war is what improved the response

Same harness (the corrected variant axis), same server baseline, only the sim
code differing — pre-fog `c60275a` against post-fog `42f3066`. Error against
the server's response, RMS, averaged over five perturbation variants:

```
metric            pre-fog   post-fog    server    |err| pre  |err| post
lane_frac_all       0.144      0.041     0.021        0.123      0.020
lane_frac_blue      0.208      0.066     0.041        0.167      0.025
lane_frac_red       0.203      0.070     0.042        0.161      0.028
n_minions_blue     11.982      5.152     1.906       10.076      3.246
n_minions_red       8.438      3.945     2.023        6.415      1.922
turret_hp_blue    761.715    163.532   332.096      429.619    168.564
turret_hp_red     709.908    288.174   231.487      478.421     56.687
```

Better on **all seven**, by 3x to 8x. Fog is the cause, not the sweep-axis fix
— which is exactly what the isolation run existed to separate, and what was
predicted before the result came back.

The mechanism is unsurprising in hindsight: without vision gating every minion
and turret in the lane could react to the champion's presence, so a positioning
perturbation propagated far more widely than the server permits.

**One wrinkle worth watching.** `turret_hp_blue` went from over-responding
(762 against 332) to **under**-responding (164). The error halved, but it
crossed the target rather than converging on it, which is not the same thing.

## Why this matters beyond fog

Fog made the idle-lane aggregate slightly *worse* — one turret lost where none
was before, mean |blue − red| 3.3 → 3.6 — and it was kept anyway because both
underlying fixes were verified against source. This is the evidence that the
judgement was right.

It is also the mirror of the call-for-help decision, where the aggregate said no
and the response test said yes, and the change was declined. The distinction
that reconciles the two:

* **fog** is correct **and** reduces response error 3–8x on every metric;
* **call-for-help** is correct **and** breaks the idle baseline outright —
  three turrets destroyed, imbalance 3.3 → 9.3.

Magnitude and direction both matter, and the aggregate alone cannot tell them
apart: it nearly passed a regression at 26 against a 26.25 bound.
