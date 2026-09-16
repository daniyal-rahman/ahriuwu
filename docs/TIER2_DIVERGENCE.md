# Tier 2: free-running divergence, characterised

**Gate 2 verdict: MET.** Tier 2 divergence is now measured, quantified against
an honest chaos floor, and written down. It is **not small** — the gate's own
parenthesis says that is allowed. Four scenarios, full 600 s episodes, sim vs.
the real server, seed 0: nothing meaningful diverges before the first wave
clash (~120–130 s); after it, minion position/HP/population and turret HP
diverge by amounts that are large relative to the game (turret HP errors of
up to 100% of max HP; population gaps of up to half the live count; in one
scenario the sim's champion dies from wave damage the server's champion takes
zero of) and these are many orders of magnitude above the measured chaos
floor, so they are real disagreement, not float32 chaos wearing a costume.
The dominant, well-evidenced cause is the already-known, deliberately-toggled
missing call-for-help channel (`docs/TARGET_ACQUISITION_DIFF.md`,
`docs/CALL_FOR_HELP_SWITCH_RATE.md`) — this document adds a new, sharper
quantification of its consequence (a ~2.6x champion damage-absorption rate,
and outright death in 2 of 4 scenarios) but does not re-litigate its cause.

Code: `lanerl_jax/parity/tier2.py` (raw capture + comparison),
`lanerl_jax/parity/tier2_batch.py` (batch driver). Data:
`lanerl_jax/runs/tier2/*.json` (raw curves and comparison reports) --
`lanerl_jax/runs/` is `.gitignore`d in this tree (same convention
`docs/PERTURBATION_RESPONSE.md` followed for its own pilot JSON), so these
are **not** committed; every number quoted below is reproducible by
re-running the two commands below and diffing against the values in this
document, which is the point of writing them down this explicitly rather
than only pointing at a file. Reproduce one scenario:

```
PYTHONPATH=$PWD ./.venv-jax/bin/python -m lanerl_jax.parity.tier2 raw \
  --engine sim --scenario idle --seed 0 --minutes 10 --sample-every-s 2 \
  --out lanerl_jax/runs/tier2/sim_idle.json
PYTHONPATH=$PWD ./.venv-jax/bin/python -m lanerl_jax.parity.tier2 raw \
  --engine server --scenario idle --seed 0 --minutes 10 --sample-every-s 2 \
  --out lanerl_jax/runs/tier2/server_idle.json
PYTHONPATH=$PWD ./.venv-jax/bin/python -m lanerl_jax.parity.tier2 compare \
  --a lanerl_jax/runs/tier2/sim_idle.json --b lanerl_jax/runs/tier2/server_idle.json \
  --out lanerl_jax/runs/tier2/cmp_idle.json
```

Server episodes ran on `desktop` via `slurm/parity_g2.sbatch` (job 814, all
four scenarios in one job, ~50 s total wall time); sim episodes ran locally
(cheap, CPU-only, no shared-resource contention). See "Where this ran" below
for one slurm/venv wrinkle found and fixed along the way.

---

## 0. Definitions and honesty checks, before any number

**What Tier 2 means here**, per the gate's own definition: both engines from
the identical initial state (server-derived geometry, stats, wave schedule —
`lanerl_jax/parity/sim_vs_server.py`'s own documented caveat that the
server's *hidden* init, e.g. rune-page staging, is never fully shared),
driven by the *same scripted action stream* issued to both, free-running, no
injection after t=0. This is weaker evidence than Tier 1's one-step
differential (nothing here is attributable to a single tick) but it is the
only measure of what actually matters for training: an agent trained
free-running on the sim has to transfer to a free-running server.

**"Several scenarios/seeds" — what actually varies, checked before trusting
any spread.** `lanerl_jax/parity/sweep.py`'s own docstring already records
finding this the hard way: a 4-seed sweep of this exact lane came back
bit-identical on both engines at every seed. Checked directly for this
report: `lanerl_jax/sim/state.py`'s `LaneState.key` (the JAX PRNG key
`init_lane(seed=...)` sets) is never read by `step.py` or anything it calls
— `grep -rn "\.key" lanerl_jax/sim/step.py lanerl_jax/sim/minion_ai.py
lanerl_jax/sim/waves.py lanerl_jax/sim/missiles.py lanerl_jax/sim/collision.py`
returns nothing outside `movement.py`'s unrelated `waypoint_key` field. The
sim's tick is a pure function of state; **there is no RNG on this path at
all**. The server side is the same story: `run_server_episode`'s own
docstring notes it has no equivalent of the sim's seed, and `bot_teams="none"`
(no in-server bot ever runs) means `bot_seed` cannot matter either. **A
"5-seed idle-lane sweep" would silently be n=1 five times over — exactly the
mistake this project has already made once and documented.** So this report
does not claim a seed-based error bar it cannot produce. What it does instead,
following `sweep.py`'s own fix for the identical problem: four **genuinely
different scenarios** (different scripted champion behaviour, different
trigger times), so the consistency (or lack of it) across them is the
evidence a spread would otherwise have provided. Where the four scenarios
agree (the timing of first divergence, §3), that agreement is meaningful
*because* nothing about the scenarios forces it to happen — each drives a
qualitatively different champion action stream.

**Entity correspondence.** Champions need no matching (one per team).
Turrets are matched to their own known, static, patch-derived position
(never move on either engine — exact identity, not a heuristic). Lane
minions are matched by **rank along the lane** (`lane_fraction`, ascending),
using **proportional rank** (quantile), not absolute index — see
`tier2.py`'s `_match_minions_by_rank` docstring. This was a deliberate
departure from `diff.py`'s nearest-position matching (right for Tier 1,
where nothing can be more than one tick apart) after `docs/ONE_STEP_
DIFFERENTIAL.md` §3 found its own fixed 8-unit match radius silently
right-censoring the worst 40x of "unmatched" entities once two runs are
allowed to actually separate — which is the normal case here by minute two,
not a failure. Quantile rank degrades gracefully under a population
mismatch (routine here: idle's population gap alone has p95 9, max 12
against typical live counts of 10–25) instead of letting a mere count
difference shift every downstream pairing by that many slots.

---

## 1. Scenarios run

All: `lanerl/cfg/garen1v1.json` (Map1, both Garen), `ServerLaunchSpec
(toponly=True, bot_teams="none", bot_seed=4242, step_ticks=2)`, seed 0, full
600 s / 18,000-decision episode, sampled every 2 s (300 samples). Only blue
is ever ordered; red never receives an order (matching every other driver in
this tree).

| scenario | script | notes |
|---|---|---|
| `idle` | zero orders, ever | the established comparison scenario (`docs/TICK_DIVERGENCE_TRACE.md`, `test_minion_population_is_close_to_the_server`) |
| `stand_early` | walk to `ENGAGE_POINT` (lane fraction 0.553) at t=100 s, hold 10 s, walk back | early-game engagement, right after the first two waves |
| `stand_late` | same, triggered at t=400 s | late-game engagement |
| `kill` | camp at `ENGAGE_POINT` **from t=0**, focus-fire the 3 nearest enemy minions once t=150 s is reached | the one scenario that exercises attack/damage; known AD-scaling asymmetry caveat (`last_hit_drive.py`) |

---

## 2. Chaos floor: perturb by the smallest representable amount

Per the task, before trusting any sim-vs-server number as "divergence," a
floor: run the **same engine against itself**, nudge one float by exactly one
float32 ULP (`numpy.nextafter`) at one point, and see how fast that spreads.

**Probe A — a marching minion, idle scenario.** `init_lane(seed=0)`, idle
script, full 600 s. At decision 2705 (t≈90.17 s, ~170 ms after the first
minion spawns), the first live blue `LaneMinion`'s x position was nudged by
one ULP: `923.2619018554688 → 923.261962890625` (Δ = 6.1×10⁻⁵ world units).
Compared against the unperturbed run over the remaining 510 s of game time,
using the exact same matching pipeline as the headline comparison:

```
champion_pos_err / champion_hp_err / turret_hp_err / population_gap : 0.0 everywhere
minion_pos_err:  median 0.0   p95 0.0   max 0.000122   (n=5419 matched pairs)
minion_hp_err:   0.0 everywhere
```

**The floor is ~1.2×10⁻⁴ world units, and it does not grow over 510 s of
subsequent game time** — a fixed, float32-rounding-scale residual, not
amplification. This is the single most important number in this report for
interpreting §3–4: **every sim-vs-server divergence reported below is
hundreds to tens of thousands of times larger than this floor.** None of it
can be dismissed as ordinary chaos.

**Probe B — the champion, `kill` scenario (a null result, explained rather
than hidden).** Same idea, nudging the blue champion's x by one ULP
(`34.89739990234375 → 34.897403717041016`) at decision 4505 (t≈150.17 s,
mid-approach to `ENGAGE_POINT`). Result: **exactly zero divergence on every
field, for the rest of the 600 s episode — including through the champion's
own death and respawn** (§4). This is *not* evidence that combat/death
dynamics are non-chaotic: `KillMinions._camp_at_engage` (and `_camp_step`,
shared by every scenario) re-issues a fresh `move` order to the same
destination on **every decision** while still approaching, and the
per-decision path replanning this triggers evidently re-derives the
champion's trajectory in a way that does not preserve a sub-quantum
perturbation to its previous position — the order-reissuance itself erases
the nudge before it can do anything, tick to tick. Probe A's minion, whose
waypoints are set once at spawn and never reissued, is the trustworthy floor
measurement; Probe B is reported because a null result from a flawed probe
is still worth knowing (it says something about order-reissuance semantics,
not about chaos), not because it proves the champion path is chaos-free.

**Floor verdict:** one clean measurement (Probe A), zero amplification over
a full 600 s episode. This is evidence the system is not exhibiting strong
sensitive-dependence-on-initial-conditions at the ULP scale in ordinary
pre-contact movement — it is not proof no chaotic regime exists anywhere
(e.g. near an exact simultaneous HP-zero tie, which Probe B could not test
for the reason above). Given the floor is this far below every measured
divergence, that gap does not change any conclusion in this report.

---

## 3. Does anything diverge before the waves meet?

**No — consistently, across all four scenarios.** From t=0 to the first wave
spawn (90 s) only champions exist, and the only disagreement in that window
is the already-documented, one-time rune-page HP/gold staging effect
(`lanerl_jax/sim/init.py`'s own docstring; `docs/TICK_DIVERGENCE_TRACE.md`
§A): champion HP reads 754.25 in the sim from t=0 and ramps up on the server
over the first ~1000 ms, producing a **transient max 83.25 HP "error" that
is gone by t=1 s and never recurs** (present, identically, in all four
scenarios' `champion_hp_err.red` — red is never ordered, so its number is
pure measurement of this one artifact). From t=90 s (first wave) to
~120–128 s (first cross-team contact), minion positions and populations
match closely on every scenario:

| scenario | t=60s minion pos median | t=60s pop (sim/srv, blue/red) |
|---|---:|---|
| idle | 15.4 | 6/6, 6/6 |
| stand_early | 15.4 | 6/6, 6/6 |
| stand_late | 41.5 | 6/6, 6/6 |
| kill | 15.4 | 6/6, 6/6 |

**First divergence, robust signal (population count — no matching heuristic
involved): t=128 s in `idle`** (blue and red both go from matched to
off-by-one simultaneously), landing exactly in the wave-clash window
(~120–130 s) this project's other documents already place the two waves'
first contact in. `docs/ONE_STEP_DIFFERENTIAL.md` and `docs/TICK_PARITY_
AUDIT.md` Gap 3 (collision-before-movement) independently point at this
exact moment as the first place the two engines' rules can disagree — this
free-running measurement is consistent with, not independent of, that.

One artifact worth naming rather than hiding: `idle`'s minion-position
metric shows a small blip at t=92 s (540 units, over only a handful of
matched pairs) — two ticks after the very first minion spawns, when one
side may have 1 live minion and the other 2. Quantile-matching a queue of
size 1–2 is not meaningful (§0); this is a measurement artifact of the
matching method at the smallest possible n, not a real divergence, and it
does not recur or propagate — the population-count signal at t=128 s is the
trustworthy "first divergence."

**Conclusion for the gate's own question ("what diverges first"): nothing,
mechanically, until the first wave clash — and then it is the combat
resolution itself (which minion fights whom, who dies when), not movement,
not wave-spawn timing, not turret geometry.** This matches, and adds a
free-running confirmation to, `docs/ONE_STEP_DIFFERENTIAL.md`'s one-step
finding that movement/collision-ordering bias and missile-mediated damage
timing are the two live gaps.

---

## 4. Whole-episode numbers, per scenario

All errors below are `sim − server` (a positive turret HP error means the
sim's turret has *more* HP than the server's, i.e. the sim under-damaged it
relative to the server, consistent with `docs/ONE_STEP_DIFFERENTIAL.md`'s
one-sided "sim under-damages" finding). `n` for champion fields excludes
samples where the champion was dead. Full distributions (not just these
summary rows) are in `lanerl_jax/runs/tier2/cmp_*.json`, regenerable but not
committed (`lanerl_jax/runs/` is `.gitignore`d, per the note above).

### idle

```
champion_hp_err (blue=red, red never ordered):  median 0.25   p95 0.25   max 83.25   (n=300; t=0 staging only)
champion_pos_err:                                exactly 0 (never ordered)
turret_hp_err   blue: median   5.3  p95/max  363.9   (of 1550 max)
turret_hp_err   red:  median 614.0  p95/max 1028.0   (66% of max)
population_gap  blue: median 1  p95 9   max 12
population_gap  red:  median 1  p95 7   max 10
minion_pos_err (quantile-matched, pooled): median 805.8  p95 5906.6  max 10683.9   (n=4607; lane length 21,888)
minion_hp_err:                              median 0.0    p95  393.0  max   638.0
first divergence (population, robust):  t=128 s
turret divergence onset:                red t=228 s, blue t=300 s
```

Time series (60 s buckets; `minion_pos_*` = quantile-matched pooled position
error in world units; populations are live counts, sim/server):

| t(s) | pos median | pos p95 | turret Δ blue | turret Δ red | blue nB(sim/srv) | red nR(sim/srv) |
|---:|---:|---:|---:|---:|---|---|
| 0 | – | – | 0 | 0 | 0/0 | 0/0 |
| 60 | 15.4 | 1051.7 | 0 | 0 | 6/6 | 6/6 |
| 120 | 405.4 | 1060.8 | 0 | 0 | 11/13 | 11/11 |
| 180 | 927.6 | 3970.0 | 0 | 600.0 | 10/11 | 11/10 |
| 240 | 517.9 | 3775.8 | 0 | 614.0 | 7/7 | 13/16 |
| 300 | 871.7 | 6001.3 | 90.0 | 614.0 | 8/12 | 16/10 |
| 360 | 1459.6 | 6087.5 | −183.1 | 1028.0 | 13/10 | 12/10 |
| 420 | 1058.3 | 5698.4 | −183.1 | 1028.0 | 15/6 | 6/13 |
| 480 | 1879.6 | 8993.3 | 363.9 | 1020.5 | 19/12 | 6/11 |
| 540 | 864.8 | 4361.1 | 363.9 | −150.0 | 12/9 | 5/6 |

Note the sign flip on `turret Δ red` (t=360→540) and on `turret Δ blue`
(t=300→360): this is the same "lead flips every minute or two" oscillation
`docs/JAX_REWRITE_PLAN.md`'s J1 status already reports internally for the
sim alone — here it is measured **directly against the server**, and the
server's own lane exhibits the analogous back-and-forth (not the same
outcome at the same tick, but the same qualitative non-monotonic pattern).

### stand_early (engage at t=100s, hold 10s)

```
champion_hp_err blue: median 132.2  p95 550.3  max 711.9   (n=287)
champion_pos_err blue: median 1824.4  p95 6276.9  max 12965.9   (n=287)
turret_hp_err blue: median 0     p95/max 395.0
turret_hp_err red:  median 499.0 p95/max 1435.0   (93% of max)
population_gap blue: median 1  p95 7   max 8
population_gap red:  median 2  p95 9   max 11
minion_pos_err: median 653.4  p95 5001.8  max 10306.2   (n=4908)
```

Champion HP, engagement window (t=120–148 s; both engines start at 754.2):

| t(s) | sim HP | server HP |
|---:|---:|---:|
| 120 | 749.3 | 754.0 |
| 124 | 586.2 | 730.0 |
| 128 | 338.1 | 573.0 |
| 132 | 185.3 | 536.0 |
| 140 | 188.7 | 583.0 |
| 148 | 225.8 | 630.0 |

Over the 12 s window t=120→132 (the table above): the sim's champion loses
**564 HP** (47.0 HP/s), the server's loses **218 HP** (18.2 HP/s).
**Ratio ≈2.6x.** Champion survives on both engines in this scenario (min HP:
sim 160.9 at t=134s / server never below ~536).

### stand_late (engage at t=400s, hold 10s)

```
champion_hp_err blue: median 0.25  p95 485.7  max 629.5   (n=296)
champion_pos_err blue: median 49.6  p95 5867.5  max 12742.5   (n=296)
turret_hp_err blue: median 478.2  p95/max 824.2
turret_hp_err red:  median 0      p95/max 857.0
population_gap blue: median 2  p95 13  max 14
population_gap red:  median 1  p95 7   max 8
minion_pos_err: median 684.4  p95 6180.7  max 10976.5   (n=4623)
```

Champion HP, engagement window:

| t(s) | sim HP | server HP |
|---:|---:|---:|
| 416 | 754.2 | 754.0 |
| 418 | 675.2 | 754.0 |
| 422 | 492.7 | 754.0 |
| 426 | 230.4 | 754.0 |
| 428 | 124.5 | 754.0 |
| 430 | **dead** | 754.0 |
| 438 | 754.2 (respawned) | 754.0 |

**The sim's champion dies here; the server's takes zero damage across the
identical 10 s hold at the identical location.** Not a rate difference this
time — a binary, saturating divergence (§6). This is the sharpest single
data point in this report for "the missing call-for-help channel changes an
outcome category, not just a number": standing in the wave is lethal on one
engine and a non-event on the other, from the same script, same seed, same
starting state.

### kill (camp at ENGAGE_POINT from t=0, focus-fire trigger at t=150s)

```
champion_hp_err blue: median 199.8  p95 580.4  max 763.7   (n=289)
champion_pos_err blue: median 11109.9  p95 13205.7  max 13206.5   (n=289)
turret_hp_err blue: median 0     p95/max 1550.0  (fully destroyed in sim from t~420s; server's untouched all game)
turret_hp_err red:  median 281.3 p95 425.0  max 949.3
population_gap blue: median 1   p95 7   max 8
population_gap red:  median 3   p95 12  max 16
minion_pos_err: median 792.4  p95 7759.4  max 11187.2   (n=4528)
```

**The champion dies here too — before its own trigger ever fires.**
`KillMinions._camp_at_engage` walks to `ENGAGE_POINT` and idles there from
t=0, same as `StandInWave` does starting at its own trigger; the intended
focus-fire behaviour only begins at t=150 s. But the sim's champion is
already dead by t≈133 s (HP 749.3 at t=120 s → 107.7 at t=132 s → dead at
t=134 s → respawned by t=142 s), **17 seconds before the scenario's own
trigger.** The server's champion survives this same camping period. **This
scenario, as configured, never exercises the focus-fire behaviour it was
built to test in the sim at this seed** — the numbers above characterise
passive over-exposure death and its cascade (red's minion count balloons to
22–23 against the server's ~10, and the sim's blue outer turret is fully
sieged down by t≈420 s while the server's is never touched), not focus-fire
combat. This is a scenario-design finding for future work
(shorten the pre-trigger camp, or trigger immediately), not a new sim bug —
recorded honestly rather than quietly re-run with a friendlier trigger time.

Time series (`kill`), showing the turret going from parity to fully
destroyed and staying there:

| t(s) | pos median | turret Δ blue | red nR (sim/srv) |
|---:|---:|---:|---|
| 180 | 342.1 | 0 | 8/12 |
| 300 | 404.9 | 0 | 17/11 |
| 360 | 2468.3 | 0 | 22/10 |
| 420 | 3367.6 | **−1550.0** | 15/9 |
| 480 | 4326.7 | −1550.0 | 23/10 |
| 540 | 4837.3 | −1550.0 | 15/6 |

---

## 5. Does divergence saturate, or grow without bound?

**Saturates / oscillates within a large-but-finite range — it does not grow
monotonically, and it does not run away to infinity, because the game itself
is bounded** (finite lane length, a live-unit cap, a turret that can only go
to 0 once). Three distinct saturation shapes, all visible above:

1. **Population and minion-position error oscillate.** `idle`'s
   `turret_hp_err_red` goes 0 → 614 → 1028 → −150 (t=180→540): it grows,
   peaks, then *reverses sign* — the sim's red turret goes from behind the
   server's to ahead of it. This is the same "lead flips every minute or
   two" pattern `docs/JAX_REWRITE_PLAN.md`'s J1 status already reports for
   the sim's internal dynamics, now confirmed as a real sim-vs-server
   pattern rather than only an internal one.
2. **Turret HP saturates one-way once destroyed.** `kill`'s blue turret
   error hits exactly −1550 (fully destroyed, 0 HP) at t≈420 s and **stays
   there** for the rest of the episode — an absorbing state, not further
   growth, because there is nothing left to destroy.
3. **Champion life/death is a hard, binary saturation.** A champion is
   either alive or dead; in `stand_late` and `kill` the sim's champion hits
   that floor while the server's does not, at seed 0.

**None of this is "small."** Turret HP errors reach 60–100% of max HP;
population gaps reach roughly half the smaller side's own count; minion
position "shape" errors reach roughly half the lane's total length; one
scenario's outcome (turret alive/dead, champion alive/dead) flips entirely.
The gate says this is allowed. What makes it a *characterisation* rather
than an alarm is: it is bounded, it is consistent in *where* it starts
(first wave contact, §3) across four independently-scripted scenarios, its
magnitude is traceable to a specific, previously-identified mechanism
(§6), and it sits many orders of magnitude above the measured chaos floor
(§2) — so it is attributable disagreement, not noise dressed up as a
finding.

---

## 6. Parity bugs and gaps found (separate from the characterisation itself)

None of these are new root causes — all trace to mechanisms this project has
already identified and deliberately deferred (`enable_call_for_help=False`
is a toggle, not an oversight; see `docs/CALL_FOR_HELP_SWITCH_RATE.md`'s
"stays a toggle, default False" verdict). What is new here is the
**quantified, absolute consequence**, which previous instruments (response
differencing, switch-rate counting) could not show directly:

1. **A champion standing in the enemy wave loses HP at roughly 2.6x the
   server's rate** (`stand_early`: 47.0 HP/s sim vs. 18.2 HP/s server over
   the same window at the same location). Consistent with, and a sharper
   number than, `docs/PERTURBATION_RESPONSE.md`'s earlier finding that the
   sim's *response* magnitude to the same perturbation was 3–10x the
   server's.
2. **That overdamage is severe enough to kill the champion outright** in 2
   of 4 scenarios at this seed (`stand_late`, `kill`) while the server's
   champion never dies in any of the four. In `stand_late` the server's
   champion takes *zero* damage across the identical 10 s hold — a
   qualitative, not just quantitative, divergence.
3. **`kill_minions`, as configured, does not test what it was built to
   test** at this seed: the champion dies from passive wave exposure 17 s
   before its own focus-fire trigger fires. A future pass should either
   shorten the pre-trigger camp or move the trigger earlier.
4. **Turret sieging is asymmetric and can be terminal**: the sim's blue
   outer turret is fully destroyed in `kill` by t≈420 s while the server's
   is never touched in the same 600 s. Consistent with the already-known
   unstable lane equilibrium; newly confirmed against the real server
   rather than only observed in the sim's own internal history.
5. **Methodology-only, not a sim bug**: quantile-rank minion matching
   produces a brief, small-n artifact (a few-hundred-unit "error" over 1–2
   matched pairs) in the first ~2 s after a wave spawns, when the two
   engines' populations differ by exactly one minion. Named so it is not
   mistaken for a real divergence; it does not recur past the initial spawn
   ramp and does not affect any headline number in §4.

---

## 7. Where this ran, and one slurm/venv wrinkle found and fixed

Server episodes: `slurm/parity_g2.sbatch` (job 814, `desktop`, all four
scenarios via `lanerl_jax.parity.tier2_batch`, ~50 s total). Copied from the
main worktree's `slurm/parity.sbatch` and repointed at
`/mnt/nfs/projects/ahriuwu-lanerl-jax-g2` per this gate's instructions.

**Found along the way**: this worktree's `.venv-jax` is a symlink to the
*main* worktree's real venv, written with an absolute `/srv/nfs/...` target.
`danilogin` has `/srv/nfs` and resolves it; `desktop` only has `/mnt/nfs`, so
the naive `./.venv-jax/bin/python` pattern the main `parity.sbatch` uses
(which works there because the main worktree's `.venv-jax` is a real
directory, not a symlink) fails on `desktop` with "No such file or
directory" for every worktree whose venv is symlinked this way. Fixed in
`slurm/parity_g2.sbatch` by resolving the symlink's literal target and
translating the `/srv/nfs` → `/mnt/nfs` mount prefix before invoking it,
rather than editing the symlink itself (correct as-is on `danilogin`; not
this job's place to change something outside this worktree's own script).

Sim episodes ran locally (CPU-only, no shared server resource, cheap enough
per `docs/PERTURBATION_RESPONSE.md`'s own cost measurement) rather than via
slurm.

**Verification, not assumption, that job 814 actually ran the work** (the
coordinator flagged, correctly, that this exact class of bug — a dangling
venv symlink making a wrapper "succeed" without ever invoking Python — killed
a sibling agent's job elsewhere, and that a job producing no rows is not a
null result, it is no result). Checked directly, after the coordinator's fix
repointed all three worktrees' `.venv-jax` at a relative target: this
worktree's script resolves the venv by reading the symlink at run time
(`readlink`) rather than hardcoding a path, so it is correct under both the
old absolute target (what job 814 actually ran against) and the new relative
one. For job 814 specifically: its `--output` log (quoted above) shows four
distinct `server <scenario>: 300 samples over 18000 decisions, wall=...s`
lines with sane, scenario-differentiated wall times (5.9–10.8 s) — not a
single "command not found" and exit — and every one of the eight raw curve
files this report uses (`server_idle.json`, `server_kill.json`,
`server_stand_early.json`, `server_stand_late.json`, and the four `sim_*`
files) was independently re-checked to contain exactly 300 `t_s` samples,
300 populated per-decision unit lists, `decisions=18000`, and a non-trivial
entity count (38–49 live objects) at the final sample — a fixture that
happened to fail silently could not produce this. `sim_kill.json` and the
separately-launched `sim_kill_base.json` (same scenario/seed, run as two
independent processes) were compared byte-for-byte and are identical, an
extra determinism cross-check that a corrupted or truncated run would not
pass. None of the numbers in this report come from an unverified job.

Two sibling agents were also active on this login node and on `desktop`
during this work (one `git worktree`'s own `pytest` run observed directly in
`ps aux`, and a slurm queue that briefly held other jobs); the server
episodes were still run strictly one at a time per `run_server_episode`'s
hard constraint, and slurm serialised the desktop-side work as intended —
consistent with this gate's instructions, not worked around.

---

## 8. Test suite

`PYTHONPATH=$PWD ./.venv-jax/bin/python -m pytest lanerl_jax -q`:
**260 passed, 1 failed, 3 warnings, 1294 s.** No sim/core code was changed
for this gate — only new, additive files (`lanerl_jax/parity/tier2.py`,
`lanerl_jax/parity/tier2_batch.py`, `slurm/parity_g2.sbatch`, this document,
and the `JAX_REWRITE_PLAN.md` status update) — so this is not a regression
introduced here. The one failure is
`lanerl_jax/parity/tests/test_last_hit_gate.py::
test_oracle_scores_the_same_cs_in_sim_and_server`, and it is a **known,
pre-existing, documented failure**: the test's own docstring says outright
"so as of 2026-09-16 it FAILS, and that failure is the deliverable" — it is
J1 gate 3 (`docs/JAX_REWRITE_PLAN.md`'s own "Open" list: "Gate 3 fails
narrowly"), not gate 2, and not introduced by this work. Notably, this run's
own numbers (`sim cs=9, server cs=4, sim deaths=5, server deaths=0`) land in
the same direction as this report's own §4/§6 finding — the sim's champion
dying to wave damage the server's does not — under the heavy, multi-agent
CPU contention this login node was under while this suite ran (see §7);
gate 3's exact CS gap is evidently sensitive to real-time contention and is
tracked separately from gate 2, not re-litigated here.

---

## 9. Gate 2, restated

> Tier 2 divergence characterised and written down (not necessarily small).

- **Characterised**: yes — four scenarios, full 600 s episodes, per-class
  (champion/minion/turret) position and HP error with median/p95/max,
  population and lane-balance tracking, time-to-first-divergence identified
  and localised to the first wave clash, saturation behaviour described with
  its three distinct shapes (§5).
- **Against a floor**: yes — a smallest-representable-perturbation
  self-divergence probe puts the chaos floor at ~10⁻⁴ world units with no
  growth over a full episode; every reported sim-vs-server number is orders
  of magnitude above it.
- **Not small**: correctly so, and said plainly — turret HP errors up to
  100% of max, population gaps up to half the count, one scenario's
  champion life/death outcome flipped entirely.
- **Written down**: this document, with every quoted number reproducible via
  the raw and comparison JSON under `lanerl_jax/runs/tier2/` (gitignored,
  regenerated by the two-command recipe near the top of this document).

**Gate 2 is met.**
