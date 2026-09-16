# Tier 1: the one-step differential, finally run

**TL;DR.** Built the injector the plan's §3 has called for since J0 and never had:
server snapshot at tick N → `LaneState` → one `tick()` → diff against the
server's real tick N+1. Ran it over all 14,401 tick-pairs of a 240 s idle-lane
recording (no orders, no bots — fully deterministic). Minion **position**
disagrees on 16.9% of trustworthy one-step predictions with a **98.6%
one-sided** bias (the sim consistently ends up further along its own heading
than the server — consistent with the collision-ordering gap already named in
`docs/TICK_PARITY_AUDIT.md` Gap 2). Minion **HP** disagrees on 0.43% of
one-step predictions, but where it does, the sim is one-sidedly *short*
(mean +19.9 HP, i.e. the sim under-damages), and an HP disagreement is **46×
more likely** on a tick where a missile the injector cannot see is in flight
(11.1% vs. 0.24% baseline). On a hand-verified
sample of all 18 minion deaths in the trace, **the sim predicted the correct
death tick 0/18 times** — always late. The dominant cause is not a logic bug
in the free-running sim; it is that the state dump does not carry missile
state or target identity, so **this specific finding says the oracle is
blind to these mechanics, not that the free-running sim mishandles them**.
See "What this instrument cannot see" below before drawing the opposite
conclusion. One position-matching artifact in this harness itself (a
too-tight correspondence radius) was found and is called out rather than
silently patched.

---

## 0. Why this run happened now, and a correction mid-flight

Two sim changes landed while this measurement was in progress:
`4a425bf` (ranged basic attacks fire real missiles; the outer-turret AD ramp
went live) and a same-day fix making lane minions spawn at the measured
barracks position rather than at their path's first/last vertex (`TOP_LANE_PATH[-1]`
was 446 units from red's true barracks). The first full run (started before
these landed) independently **rediscovered the second bug from scratch** —
before being told about it — as the single largest contributor to minion
position error in this instrument. That run was discarded and the numbers
below are from a second run against the corrected sim, using the *same*
recorded server trace (the server side of a Tier-1 comparison is unaffected
by sim-code changes; only the injected side needed redoing). The rediscovery
is worth keeping on record as evidence the instrument works: see §5.

## 1. Setup

- **Trace**: `lanerl_jax/runs/one_step/main/server/instance000.log`, recorded
  with `lanerl_train.vec.VecLaneEnv(1, ServerLaunchSpec(toponly=True,
  bot_teams="none", bot_seed=4242, step_ticks=2, extra_env={"LANERL_STATE_DUMP":
  "1", "LANERL_STATE_DUMP_FULL": "1"}))`, stepped with `env.step([None])` for
  7,200 decisions (240 s of game time). No order is ever issued and no bot
  runs, so the two champions never move or fight for the entire recording —
  the trace is fully deterministic and exercises the minion wave / minion AI /
  autoattack / turret mechanics, not champion action decode.
  14,402 snapshots parsed (one per 16.667 ms tick, `t=0` to `t=240043`),
  giving 14,401 consecutive tick-pairs.
- **Recording cost**: 240 s of game time recorded in ~17 s wall-clock,
  uncontended, on this 6-core box. No more than one server process ran at a
  time.
- **Differential cost**: the full 14,401-pair pass took 1,248 s (~20.8 min)
  single-threaded, dominated by Python-level bookkeeping (dict/list
  construction, greedy O(n·m) entity matching) rather than the JIT-compiled
  `tick()` call itself, which compiles once and dispatches fast thereafter.
- **Code**: `lanerl_jax/parity/inject.py` (the injector) and
  `lanerl_jax/parity/one_step.py` (the differential engine, aggregation,
  report). Both are new, reusable, and unaffected by this write-up's numbers
  going stale — rerun them against any trace.

## 2. The injector: what went in honestly, and what didn't

`LanerlStateDump.Describe` (`lanerl-vendor/LoLServer/GameServerLib/Lanerl/LanerlStateDump.cs`)
is exhaustively parsed by `parity/trace.py`; nothing it emits is left
unparsed, and everything parsed that `LaneState` has a slot for is injected.
But `Describe` was written to catch state leaking across a reset,
not to support a second implementation, and several fields `LaneState` needs
to step correctly simply are not there. Full accounting, field by field, is
in `inject.py`'s module docstring; the load-bearing summary:

**Injected exactly** (read straight off the dump): kind, team, alive, position,
current/max HP, move order, champion level/gold/CS/deaths, champion spell
level and cooldown per slot.

**Derived exactly, not guessed**: the wave-spawner's three scalars
(`next_spawn_ms`, `minion_number`, `cannon_count`) are not in the dump in any
form, but `LevelScript.Update` is a deterministic, RNG-free function of game
time alone, so replaying it from `t=0` with the trace's own recorded tick
times reproduces them exactly — verified directly (§4). A minion's profile
row (`model`, which drives every per-tick stat gather) is recovered from
`(kind, team, max_hp)`: melee/caster/cannon/super have unique max-HP values
(455/290/700/1500) that never collide, so this is an exact lookup against the
same patch-table-built column the sim itself uses, not a hardcoded number.

**Reconstructed with an explicit refusal case**: a marching minion's waypoint
*positions* aren't in the dump (only `Waypoints.Count`), but a minion with no
target walks the same static lane corridor (`TOP_LANE_PATH`, forward for blue
/ reversed for red) every episode, so a position that lies on that corridor
pins down which vertex is next. The injector projects the minion's position
onto the corridor and refuses (flags `movement_trustworthy=False`, injects
zero waypoints — no movement this tick, rather than a guessed direction) when
the position is more than 8 units off it. This is honest but not free: see §3.

**Defaulted, and this is the big one**: `target` (all units, always -1),
`aa_cooldown`/`aa_windup`/`is_attacking`/`has_auto_attacked` (always
"ready, not swinging"), and the minion AI's own bookkeeping
(`target_priority`, `ignore_until`, `help_priority`, `ai_local_time`,
`time_since_attack`, all reset to their fresh-unit defaults) are **not in the
dump in any form** — confirmed independently in this project's own
`parity/tests/test_autoattack.py`: `cast_spell` reads `"-"` on 100% of 15,162
champion rows in a run with 32 real swings. Missile state
(`missile_alive`/`x`/`y`/`target`/`source`/`damage`/`speed`) is likewise
always injected empty — the dump does carry a generic 2-field
`SpellMissile|x,y` row (confirmed: 19,086 of them in this trace), but with no
team, target, source, damage or speed, there is nothing in it a `LaneState`
missile slot can use.

**Consequence, stated plainly**: injecting `target=-1` plus `ai_timer=250`
(which forces an immediate re-evaluation) does not test "did the server's
held target survive this tick" — it tests "does a from-scratch acquisition
agree with the server's actual target", which conflates a wrong pick with the
server's own hysteresis (a valid incumbent is protected from a same-priority
closer unit; a fresh scan is not — `sim/minion_ai.py`). And injecting no
missiles means a missile launched on an earlier tick and still in flight is
completely invisible to the one-step prediction that should be dealing its
damage. **Neither of these is evidence the free-running sim is wrong** —
the free-running sim carries its own `ai_local_time`/`ignore_until`/missile
arrays forward correctly tick-to-tick in a real rollout; the *oracle* just
cannot expose that internal state for injection. This distinction matters
for every finding below that traces back to one of these two defaults, and
it is flagged at each one.

## 3. A methodology artifact found in this harness itself: match-radius censoring

Entity correspondence has to be recovered by nearest position (the dump
strips NetId; `parity/diff.py`'s existing method, reused here). The default
radius (8 world units, `parity.diff.DEFAULT_TOLERANCE.match_radius_q`) is
justified there by a champion's top speed (<6 units/tick). It is **too tight
for lane minions across ticks with heavy collision-driven displacement**, and
this run found that the hard way rather than assuming the borrowed constant
was fine:

- The 15 worst per-tick position errors recorded (§4) are 7.938–7.974 units —
  clustered *right at* the 8-unit ceiling. That is the signature of a radius
  that is clipping the true tail, not resolving it: a pair whose real
  separation exceeds the radius fails to match at all and silently vanishes
  from the "position" statistics as an *unmatched* entity, rather than being
  counted as a large error. **The position/HP/move-order percentages in §4
  are therefore right-censored — the true tail is worse than shown, and by an
  unmeasured amount.**
- This bit hardest in the pre-tick-position-based death/spawn matching pass
  (§6): the full run's own death-confusion table shows 785 `LaneMinion`
  slots the sim predicted alive with no matching real entity at N+1
  ("SIM ALIVE BUT SERVER DEAD" in the report). A direct radius-sensitivity
  check (same trace, same pairing method run standalone at four radii rather
  than through the full injector, so a slightly different but consistent
  785-ish baseline) swept 8/16/30/50 world units and found unmatched
  "vanished" minion counts of **767 → 350 → 103 → 29** respectively. The true
  death count in
  this trace, hand-counted from the dump's own dead-flag rows, is 18 (§6) —
  so the 8-unit radius overstates "unexplained disappearances" by roughly
  **40×**, almost all matching failures during crowded wave clashes, not
  real deaths. The bulk "wave spawning" metric this harness emits is built on
  the same pass and inherits the same problem (§7) — it is reported and then
  explicitly discarded in favour of a clean, radius-free check.

This was not tuned away. Per the ground rules, tolerances are not adjusted to
make a number look better; the radius is left at the project's existing
default, the artifact is named with its measured size, and a follow-up run at
a larger radius is recommended future work rather than done silently here.

## 4. Field-by-field results

"Exact" means the dump's own quantised integers agree bit for bit — position
to 1/16 unit, HP to 1/1024 — which is the plan's own §3 bar, not a loosened
one. `n` is the number of one-step predictions scored (subject to the
right-censoring caveat above).

| kind | field | n | exact | inexact | median &#124;err&#124; | p95 | max | mean signed | one-sided? |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Champion | hp | 28,802 | 28,798 (99.99%) | 4 | 41.20 | 80.97 | 80.97 | **−41.20** | yes — see §8 |
| Champion | position / move_order / waypoints | 28,802 each | 100.00% | 0 | — | — | — | — | n/a (idle) |
| LaneMinion | position (trustworthy injection) | 45,252 | 37,600 (83.09%) | 7,652 | 0.0625 | 7.92 | 7.955 | +1.33 | yes, see below |
| LaneMinion | position_along_heading | 7,497 | 881 (11.75%) | 6,616 | 0.0625 | 7.92 | 7.94 | **+1.16** | **98.6% one-sided** |
| LaneMinion | position (untrustworthy injection) | 118,696 | 49,790 (41.95%) | 68,906 | 0.0625 | 5.44 | 7.97 | +2.12 | flagged low-confidence, see §2/§3 |
| LaneMinion | move_order | 163,948 | 143,376 (87.45%) | 20,572 | — | — | — | — | see §9 |
| LaneMinion | waypoints (count) | 45,252 | 37,436 (82.73%) | 7,816 | — | — | — | structurally non-comparable, see below |
| LaneMinion | hp | 163,948 | 163,242 (99.57%) | 706 | 23.0 | 35.0 | 168.0 | **+19.90** | yes, see §5b |
| LaneTurret | hp | 345,624 | 345,586 (99.99%) | 38 | 14.375 | 25.0 | 39.375 | **+16.15** | yes, same cause as minion HP |
| LaneTurret | position / move_order / waypoints | 345,624 each | ≥99.999% | ≤5 | — | — | — | — | static objects, as expected |

**Position, `position_along_heading` (the collision-ordering check).** For
every matched minion that actually moved this tick (`|pred − pre-tick| >
0`), `position_along_heading` is the signed distance from the server's real
N+1 position to the sim's predicted N+1 position, projected onto the sim's
own direction of travel: positive means the sim ended up *further along its
own heading* than the server did (the sim is "ahead"/moved more), negative
means the opposite. Of the 6,616 non-exact cases, **98.6% are positive**
(0.014, i.e. 1.4%, are negative). This is not symmetric noise — it is a strong, one-sided signal
that the sim consistently travels further per tick than the server does in
disagreement cases, which is the observable signature `docs/TICK_PARITY_AUDIT.md`
Gap 2 predicts (`server pushes apart last tick's positions then moves; we
move then push apart`). **This is not an injection artifact** — position is
fully injectable ground truth, so this is direct evidence about the
free-running sim's own step order, not about the oracle's blind spots.

**Waypoints (count).** Excluded from the headline read deliberately: this
project's own `parity/sim_vs_server.NOT_MODELLED` already documents that
`Waypoints.Count` is structurally incomparable (the sim holds a fixed-size
array where a `MOVE_TO` minion always carries the *entire* remaining lane
corridor; a direct check here found the server instead reports `Count=2` for
a freshly spawned marching minion — a short, periodically-refreshed leg, not
the whole route). The 82.73% agreement number is not "the sim gets waypoint
counts right 83% of the time" — it means nothing on its own; it moves with
`position`'s gating and is listed for completeness only.

**HP, one-sidedness.** Champion HP's four misses are a one-time, sharply
localised episode-init effect: `sim/init.py`'s own comment already documents
that the champion's current HP does not jump to its rune-adjusted max
immediately (672.0 at t=0 → 754.25 once play starts). Traced directly: real
current HP stays flat at 671.85 through t=500 ms, ticks up by +1.43 at the
first 500 ms `Stats.Update` boundary (t=517 ms — ordinary, tiny HP regen,
which the sim does not model at all, `docs/TICK_PARITY_AUDIT.md` Gap 1), then
**jumps the full remaining +81.0 to max in one tick at exactly t=1000 ms**
(a one-time staging event, not a regen tick). Both champions produce this
pattern (4 = 2 champions × 2 events each), and it never recurs — the sim is
exact on **every one of the other 28,798** champion-HP one-step predictions
for the rest of the 240 s episode (no combat, no further regen headroom).
This is exactly the signature the task asked to watch for: *right on almost
every tick, wildly wrong on a handful, for one identifiable, non-recurring
reason.*

Minion and turret HP's one-sidedness (means +19.90 and +16.15 — the sim's
predicted HP is *higher* than the server's, i.e. **the sim under-damages**)
is addressed together with the missile confound in §5b, because that is what
explains it.

## 5. Ranking, two ways

**By disagreement rate** (the fair basis across fields observed at very
different sample sizes; `OneStepResult.rank_mechanics()`):

| rank | kind.field | disagreement rate | n |
|---|---|---:|---:|
| 1 | LaneMinion.position_along_heading | 88.2% | 7,497 |
| 2 | LaneMinion.position (untrustworthy injection) | 58.1% | 118,696 |
| 3 | LaneMinion.waypoints (non-comparable, §4) | 17.3% | 45,252 |
| 4 | LaneMinion.position (trustworthy injection) | 16.9% | 45,252 |
| 5 | LaneMinion.move_order | 12.5% | 163,948 |
| 6 | LaneMinion.hp | 0.43% | 163,948 |
| 7 | Champion.hp | 0.014% | 28,802 |
| 8 | LaneTurret.hp | 0.011% | 345,624 |
| — | everything else | ≤0.001% | — |

Read literally this puts *movement* ahead of *damage*, but rate alone hides
which gaps are injection artifacts vs. free-running-sim behaviour. Reordered
by **what is actually diagnostic of the sim** (excluding what is provably an
injection-only limitation, see §2):

1. **Collision-ordering position bias** (§4) — real, injectable-ground-truth,
   98.6% one-sided. The most trustworthy finding in this report about the
   free-running sim's own mechanics.
2. **Missile-mediated damage/death timing** (§5b, §6) — the largest *effect
   size* (HP under-prediction, 0/18 death-tick agreement) but confirmed to be
   substantially explained by the injector's missile blind spot, not
   necessarily a free-running-sim bug. Real but currently untestable by Tier
   1 as built.
3. **Target-acquisition hysteresis** (move_order, 12.5%) — same caveat as
   (2): the injector forces fresh re-acquisition every tick, so this rate
   is an upper bound on a real effect, not a clean measurement of it.
4. **Champion HP staging** (§4) — real, tiny, one-time, already understood.
5. **Wave spawning** — investigated and **cleared**, see §7.

### 5b. The missile confound, quantified

A `SpellMissile` row present at the *injected* tick means a missile launched
earlier is still in flight — invisible to the injector (§2). Cross-tabulated
against "did any matched entity's HP disagree this tick":

- Ticks with ≥1 missile in flight: **6,116 / 14,401 (42.5%)**.
- Of those, an HP disagreement also occurred: **680 / 6,116 (11.1%)**.
- HP-disagreement rate on missile-free ticks: **20 / 8,285 (0.24%)**.

That is a **~46× relative risk**. It is not proof of individual causation
(the flag only says *some* missile is airborne *somewhere* on the map, not
that it is aimed at the specific unit that disagreed — deliberately kept
coarse rather than guessing a target), but a mechanic that changes the
outcome rate by 46× when present is not a coincidence at this sample size.

## 6. Do minions die on the same tick?

Hand-verified, not the bulk pass-1 number (§3 explains why that one — 785
"sim alive but server shows it gone" — is inflated ~40× by match-radius
censoring and should not be read as a death count).

The dump's dead-flag rows give the **exact, small, ground-truth set**: 18
`LaneMinion` rows show `hp=0/maxhp|D` across the whole 240 s trace, each at a
distinct tick, each disappearing entirely the very next tick — a clean,
single-tick removal lag every time (`docs/TICK_PARITY_AUDIT.md` Gap 4,
confirmed with 18/18 consistency and zero counter-examples: `gap4_candidates`
= 18, exactly matching). For every one of these 18, the corresponding
pre-tick sim slot was identified by nearest position (match distance ≤5.92
units, 16/18 exactly 0.00) and the sim's one-step prediction checked:

**The sim predicted "still alive" in 18 / 18 cases (0%).** It is never early;
it is always late, because the killing blow was not applied within the
injected tick.

Follow-up, per death, on *why*: the nearest in-flight missile to the victim's
pre-death position was checked. In 13 of 18 cases the nearest missile was
within 20 units (0.0–0.9 units away in three of them, under 10 units in
ten) — strong
circumstantial evidence the missile the injector cannot see is what actually
lands the kill. In the remaining 5 cases the nearest missile was 52.5–343.9
units away (clearly unrelated), so those killing blows are more likely melee
hits that the *target-hysteresis* gap (§2) mishandled instead: with
`target=-1` forced every tick, the sim's fresh re-acquisition can pick a
different, equal-or-lower-priority victim than the one the server's
already-committed attacker was hitting, so the true victim simply never gets
attacked in the one-step prediction. Both explanations are injector
limitations, not (on this evidence) free-running-sim mechanic bugs — see the
boxed warning in §2 and the direct answer to the coordinator's question in
§8.

The final-hit magnitude (victim's own HP on the tick before death) ranges
1–103 HP across the 18 events, consistent with everything from "already at
1 HP, anything finishes it" to a single cannon/turret-class hit — not
diagnostic on its own, included for completeness.

## 7. Wave spawning: investigated and cleared

The bulk per-tick population-count check this harness emits (766 ticks with
a server-side population change, 765 flagged as a sim-count mismatch) is
**not trustworthy** — it inherits the same match-radius censoring as §6's
death count, on the same underlying position-matching pass. Rather than
report a scary, wrong number, a second, radius-free check was run: every
real tick at which the blue-side `LaneMinion` count increases was extracted
directly from the trace and compared against `sim/waves.spawn_schedule()`,
the pure, deterministic (RNG-free) function of game time this project already
uses to model `LevelScript.Update`.

**Result: 31 / 31 real blue-side spawn ticks match the predicted schedule to
the exact millisecond** (e.g. `90015, 90815, 91615, ..., 239659` — identical
lists, not just close). Wave-spawn *timing* is exact, full stop. This is
expected rather than a coincidence: both sides evaluate the identical ported
formula, and `replay_wave_states` (§2) already replays it from the trace's
own true `t=0` initial condition, so agreement here is close to a
tautology once that replay is correct — it is a check on the replay's
correctness (passed) more than a fresh discovery about the server. It also
means the sim's per-tick minion *identity* count (how many minions exist,
not which specific ones) cannot itself explain any of the population-stability
questions in §8; wave timing is not the loose bolt.

## 8. Directly for the coordinator: what this says about the missing restoring force

Three things this instrument can say with a number, and one important thing
it cannot yet say:

1. **Movement has a real, one-sided bias** (§4, §5): the sim's minions
   consistently travel further per tick than the server's in collision-heavy
   moments (98.6% one-sided, matching Gap 2's push-order difference). This is
   the one finding here that is *not* confounded by an injection blind spot —
   it is directly measured from ground-truth positions on both sides. A
   restoring force built on "the server pulls crowded minions back into
   formation slightly more than the sim does" is consistent with this data,
   though this run does not isolate the magnitude's population-level effect.
2. **Damage/death timing is dominated by a confound this instrument cannot
   remove**: missiles. 46× relative risk of an HP disagreement when a
   missile is in flight (§5b); 0/18 minion deaths landed on the correct tick,
   with 13/18 having a missile essentially on top of the victim at the
   moment of death (§6). **This cannot be read as "the free-running sim mis-times
   missile damage."** The free-running sim carries missile state forward
   correctly tick-to-tick in its own rollout (that machinery is what
   `4a425bf` wired in); the state dump this Tier-1 harness injects from
   simply does not expose missile state, so Tier 1 as built is *structurally
   blind* to whether the free-running sim's missile timing is a damping
   force, a destabilising one, or neutral. Closing this gap needs either a
   server-side dump extension (touches the shared vendor tree) or a
   same-run free-running comparison instrumented independently — not a fix
   to this injector.
3. **Target-acquisition hysteresis is a second, smaller confound of the same
   kind** (§2, §6): forcing fresh re-acquisition every injected tick cannot
   distinguish "the sim would have made the same pick with real hysteresis"
   from "the sim disagrees." The 12.5% minion move_order disagreement rate
   (§5) is an upper bound on a real effect, not a clean number.
4. **What is cleared**: wave-spawn timing (§7) is exact and is not a source
   of the instability. Champion HP staging (§4) is real but bounded to two
   ticks near t=0 and cannot matter to a multi-minute drift. The originally
   suspected minion-spawn-position bug (§0) is already fixed.

The honest summary: **Tier 1 as currently built can confirm one real,
one-sided kinematic bias (movement/collision ordering) as a damping-force
candidate, and can rule out wave-spawn timing and champion HP staging as
candidates — but it cannot currently see far enough into missile or
minion-target state to test the leading hypothesis (ranged damage timing)
at all.** That is a scope limit of the oracle, stated as plainly as the
result itself.

## 9. Answers to the original questions, briefly

- **Do minion positions agree after one step?** No, not fully: 83.1% exact
  among trustworthy predictions (right-censored, §3, so the true rate is
  somewhat lower), with a 98.6% one-sided bias consistent with the verified
  collision-ordering difference (Gap 2). See §4–§5.
- **Does health agree after one step?** Mostly (99.57% for minions, 99.99%
  for champions/turrets by count), but where it disagrees it is one-sided
  (sim under-damages) and 46× more likely on a tick with a missile in
  flight — not
  random noise, and traceable to a named, unfixed instrument gap rather than
  an arbitrary formula error. See §5b.
- **Do the same units die on the same tick?** No: 0/18 in a hand-verified,
  complete census of every minion death in the trace. Always late, and the
  delay is explained (missile invisibility, target-hysteresis) rather than
  mysterious. See §6.
- **Is there a field exactly right on most ticks and wildly wrong on a
  few?** Yes, cleanly: Champion HP, wrong on exactly 2 ticks per champion out
  of 14,401, both attributable to one documented one-time init effect (§4).
  A second, structural version of the same signature was found in the
  *matching methodology itself* (§3): position/HP/death statistics are right
  and clean except at the 8-unit match-radius boundary, where they silently
  drop the worst cases instead of counting them.

## 10. What was NOT done

- Champion action decode, spells and buffs were not exercised (the fixture
  is orderless by design — see the coordinator's specific ask this run was
  built to serve). A driven fixture (`parity/record.py`'s scripted drive) is
  a different, complementary experiment already flagged as out of scope for
  this one.
- No tolerance was loosened to make any number pass. No sim code was changed
  to make an injected assumption more convenient.
- No second, larger-match-radius rerun was done to un-censor §3/§4's tail —
  named as the most valuable immediate follow-up, not attempted here given
  the ~21-minute cost of a single full pass at the existing radius.
- The server was recorded once, run singly (never more than one boot at a
  time), and reused across both differential runs in §0 without re-recording
  — the server side of a Tier-1 comparison does not go stale when the sim
  changes.
