# Tick Divergence Trace: idle lane, sim vs server

Diagnosis only. Nothing in this document was fixed; see "Where a fix would go"
for pointers.

## Setup

Server: `lanerl_train.vec.VecLaneEnv`, one instance,
`ServerLaunchSpec(toponly=True, bot_teams="none", bot_seed=4242, step_ticks=2,
extra_env={"LANERL_STATE_DUMP": "1", "LANERL_STATE_DUMP_FULL": "1"})`. Driven
with `env.step([None])` every decision -- `None` sends a bare `{}`, so neither
champion is ever issued an order (`vec.py:643-646`). No bots, no jungle, no
other lanes. Fully deterministic on both sides.

Sim: `init_lane()` then repeated `step_decision(state, lane_params(),
step_ticks=2, lane_path=jnp.asarray(np.array(TOP_LANE_PATH, np.float32)))`,
no `Orders` ever applied (champions never move). One call = one decision = 2
ticks, matching `step_ticks=2`.

One run, 9000 decisions = 300.0 s of game time (server wall time 15.0 s,
sim wall time 31.5 s -- see "clock alignment" below for why the sim needed a
rewrite to hit that). Comparison uses the existing
`lanerl_jax.parity.{trace,diff}` machinery unchanged:
`diff_snapshots(sim_snapshot, server_snapshot, Tolerance(kinds=LANE_KINDS,
ignore_fields=NOT_MODELLED))`, i.e. exactly `sim_vs_server.py`'s tolerance,
scoped to Champion/LaneMinion/LaneTurret, with the same `NOT_MODELLED` fields
(`mr`, `attack_speed`, `buffs`, `waypoints`, `move_order`, spell
levels/cooldowns, `cast_spell`, `channel_spell`, `can_move`, `skill_points`)
excluded. The driver script lives at
`/tmp/claude-1000/-srv-nfs-projects/6c72189a-2818-4135-9f1e-aaa397df05ce/scratchpad/idle_lane_trace.py`
(scratchpad, not committed).

### Clock alignment (a harness bug I found and fixed before trusting any diff)

`LanerlStateDump.Emit` keys every row on `Q(GameTime, 1f) = (long)Math.Round(v,
MidpointRounding.AwayFromZero)` (`LanerlStateDump.cs:71,173`) -- **rounded**,
not truncated. `state_to_snapshot` takes an explicit `t_ms` from its caller, so
the caller has to reproduce that rounding to key-match the server's rows.
Truncating (`int(state.t_ms)`) instead of rounding desyncs the lookup on
roughly 2 out of every 3 ticks (`66.667` truncates to `66`, the server reports
`67`) and manufactures thousands of spurious "missing server row" entries that
are a harness bug, not a divergence. Verified bit-exact once fixed: the sim's
own tick-by-tick `state.t_ms` (float32, accumulated by repeated `+= TICK_MS`
exactly as `Game.cs:479`'s `GameTime += diff` does) matches the server's
reported `t=` on every one of the 122 ticks in a 2 s smoke trace, and on all
9001 decision boundaries in the full run (0 missing after the fix, versus
non-trivial "missing" counts before it). This means the two clocks are not
just close -- they are the same sequence of floats, tick for tick, for the
whole 300 s run. Any divergence below is real simulation content, not a
comparison artifact.

## Headline answer

**The collision-before-movement hypothesis (`TICK_PARITY_AUDIT.md` Gap 3) is
neither confirmed nor refuted by this run.** It is masked. The actual first
and, by a wide margin, dominant divergence is a completely different bug: the
lane-minion spawn point used for the **red** team is wrong by **446 world
units**, present from the first tick a red minion exists, well before any two
units are ever close enough for Gap 3's predicted push-apart to matter. Once
minions are spawning ~2 world-seconds "ahead of schedule" for the entire game,
essentially everything downstream of that (wave-clash timing, which minion
fights which, HP outcomes, and eventually which turret takes siege damage) is
a cascade from this one bug, not independent evidence either way on
collision-vs-movement ordering.

## Known/expected artifacts (not the finding -- named so they don't get
mistaken for it)

### A. Champion HP/max-HP/gold staging at t=0-1000 ms (documented, resolves)

At `t=0` (decision 0): `Champion(100)` and `Champion(200)` both show
`hp: 754.248046875 != 671.8486328125` (sim != server), `max_hp` the same pair,
and `gold: 0.0 != 475.0`. This is the effect `lanerl_jax/sim/init.py`'s own
docstring already documents: *"the champion's max HP reads 672.0 in the t=0
snapshot and 754.0 once play starts, because `LanerlEpisode` applies the [rune]
page over the first ticks rather than at construction."* `init_lane()`
constructs the champion already at the post-page value (754.248, `RUNE_HP_BONUS`
folded in); the server applies it gradually. Traced precisely in this run:
current HP sits at 671.85 through `t=500`, ticks to 673.28 at `t=517` (one
throttled `Stats.Update` regen tick, `AttackableUnit.cs:242-249`, itself listed
as `MISSING` in `TICK_PARITY_AUDIT.md` row 2), then jumps the rest of the way
to 754.248 in one step at `t=1000`. `max_hp` and the champion's starting gold
(475 -> 0) both resolve within the very first tick. **Clean from `t=1000` ms
onward until the first minion spawns at `t=90000` ms** -- 2670 consecutive
clean decisions, i.e. the harness and the two clocks agree exactly (to 1/16
unit and to the integer StatQ) for the entire 89 s before any minion exists.
That's strong independent confirmation the alignment fix above is correct and
that champion geometry/turret geometry/base regen-absence introduce no drift
on their own.

### B. An automatic item purchase around t=285 s (economy/shop, entirely unmodeled)

At `t=284002` ms both champions sit at `gold=322.20`, `move_speed=345`. By
`t=285368` ms, `gold=0.05` and `move_speed=370` (+25, a boots-tier bonus) --
simultaneously, identically, on both sides, while each champion's position is
unchanged (`(26.0, 280.0)`, the fountain, the whole 300 s -- champions are
never ordered, so this cannot be a walked purchase). The server is
auto-purchasing boots for an idle champion once it accumulates enough passive
gold; this is a shop/item mechanic, and shop/items are not modeled in the sim
at all (gold never gets spent, `move_speed` never changes). This is scope, the
same kind of gap as buffs/spells/spell cooldowns, and it happens to be why
`gold` and `move_speed` are *not yet* in `NOT_MODELLED` even though maybe they
should be -- worth a note for whoever maintains that list, not a "bug."
Champion ambient gold also disagrees far earlier and much more subtly
(`t=91065`: `gold: 1.900390625 != 2.849609375`, a ~0.95-gold gap persisting
from the first passive tick) -- the sim's ambient-income formula/cadence
(`sim/rewards.py:ambient_gold`) does not exactly match the server's, independent
of the minion-spawn bug below (it affects champions, who never move).

## THE FIRST GENUINE DIVERGENCE: red lane minions spawn 446 units from where the server spawns them

**Decision 2701, t = 90032 ms** (the first sampled decision after the wave
timer crosses 90000 ms -- both sides cross on the identical tick, per the
clock-alignment check above): `LaneMinion(team=200)` (red) is already
**438 world units** away from its nearest same-team counterpart -- outside the
differ's 8-world-unit match radius, so it is reported as two *unmatched*
entities rather than one shifted one:

```
present only in left  at (12505.8, 12777.2)   <- sim
present only in right at (12451.8, 13212.2)   <- server
```

This is the mechanic: **spawn / position**, not movement, not collision. It is
present on the tick the entity is created, before it has taken a single step,
so it cannot be Gap 3's push-apart (which requires two units already close
enough to overlap).

### Root cause (found and localized, not fixed)

`lanerl_jax/sim/step.py:166-169`:

```python
state = spawn_minion(state, Team.BLUE, _WAVE_ROW_BLUE[mi], hp_b,
                      lane_path, enabled=mtype >= 0)
state = spawn_minion(state, Team.RED, _WAVE_ROW_RED[mi], hp_r,
                      lane_path[::-1], enabled=mtype >= 0)
```

`spawn_minion` (`lanerl_jax/sim/init.py:327`) places the new minion at
`sx, sy = path[0, 0], path[0, 1]` -- the **first element of whatever path it
was handed**. For red that's `lane_path[::-1][0] == TOP_LANE_PATH[-1] ==
(12511.0, 12776.0)`.

But the file's own `MINION_SPAWN` table (`lanerl_jax/sim/init.py:134-137`,
documented as *"first full-health sighting of a new minion"* from an
independent measurement) says the true red barracks position is
`(12451.0, 13218.0)`:

```
dist(TOP_LANE_PATH[-1], MINION_SPAWN[Team.RED]) = 446.05 world units
dist(TOP_LANE_PATH[0],  MINION_SPAWN[Team.BLUE]) =   5.10 world units
```

Reversing the lane polyline to get red's path is correct for the *waypoints
after* the first one, but the polyline's endpoint is not the barracks -- it's
just the point the path happens to end at, and for blue that point is
coincidentally 5 units from the true spawn (small enough to usually stay
inside the differ's match radius) while for red it is 446 units away
(nowhere close). `MINION_SPAWN` is already computed correctly in this exact
file and is simply never used for spawning -- `grep`-ing the tree, its only
consumer is `lanerl_jax/sim/tests/test_lane.py:157`'s
`test_the_lane_path_starts_at_the_measured_barracks`, and that test checks
**only the blue side** (`abs(px - bx) < 10`), so the red-side 446-unit error
has no test that could have caught it.

### Measured effect: not a transient, a permanent ~700-unit head start

Tracking the most-advanced red minion's true (non-tolerance-clipped) distance
to its server counterpart over time:

| t (ms) | server lead minion | sim lead minion | distance |
|---|---|---|---|
| 90032 | (12451.8, 13212.2) | (12505.8, 12777.2) | 438.3 |
| 91065 | (12462.0, 13137.1) | (12178.4, 12852.4) | 401.9 |
| 93231 | (12131.3, 12754.5) | (11492.1, 13009.9) | 688.3 |
| 94831 | (11653.9, 12916.2) | (10985.3, 13126.2) | 700.8 |
| 96364 | (11177.8, 13063.6) | (10492.8, 13201.6) | 698.8 |
| 100330 | (9907.1, 13258.4) | (9208.3, 13302.9) | 700.2 |
| 105596 | (8198.8, 13366.1) | (7499.9, 13404.8) | 699.9 |
| 110862 | (6489.4, 13357.2) | (5789.9, 13327.8) | 700.1 |
| 116127 | (4779.4, 13281.6) | (4080.1, 13250.8) | 700.0 |

The gap grows from ~440 to ~700 units over the first ~4 s (the server's minion
is still walking the true, uncaptured barracks-to-lane leg that our sim skips
entirely by spawning directly on the polyline), then **holds constant at
~700 units for the rest of the measured march** -- every red minion is
permanently about 2.15 real-time seconds (700 / 325 units/s) ahead of where it
should be, for its whole lifetime, not a transient that resolves once it
reaches the shared path. The much smaller blue-side version of the same bug
(5.1 units) surfaces only occasionally as a bare-threshold "unmatched" (e.g.
`LaneMinion(team=100)` at `t=97097`/`98630`), which is why nobody noticed it on
the blue side either -- it mostly hides inside the match radius.

### This one bug is the plausible cause of nearly everything after it

Once red's clock is offset by ~2.15 s relative to blue's for the entire game,
the two waves meet at a different place and time in the sim than on the
server, so:

- **Population**: 23 live minions in the sim vs 21 on the server at `t=300000`
  (a ~10% gap, in the direction and rough magnitude the project's existing
  Tier-3 population check already tracks).
- **Matched-entity HP "disagreements"** (e.g. `t=100330`: `hp: 290.0 != 455.0`;
  `t=143896`: `hp: 340.0 != 161.0`) are not a shared damage-math bug -- they are
  the greedy nearest-position matcher pairing up minions that are not actually
  the same fight, which `diff.py`'s own docstring warns is only sound "when the
  two simulations are close" (within about one tick of travel, ~6 units); at
  hundreds of units apart the matching itself is no longer meaningful.
- **Turret sieging, by far the largest downstream symptom**: at `t=299576` the
  server's red outer turret (`(3911.7, 13654.8)`, `TOP_OUTER_TURRET[Team.RED]`)
  is at **936.25 / 1550 HP** (60%, under sustained attack) while blue's outer
  turret has taken a token 7.5 damage. The sim shows **both outer turrets still
  at full 1550 HP** at the same tick -- a completely different overall
  lane-push outcome by the 5-minute mark, not a numerical rounding difference.

## Timeline: first divergences by decision, in order encountered

Decision index is `round(t_ms / 33.333)`. Classified as **ROOT** (the bug
itself), **CASCADE** (a consequence of the root bug, not independent evidence),
**KNOWN** (already-documented modelling gap), or **INDEPENDENT** (a real,
separate small gap).

| # | decision | t (ms) | unit | field(s) | sim | server | class |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 0 | Champion(100/200) | hp, max_hp, gold | 754.25 / 754.25 / 0.0 | 671.85 / 671.85 / 475.0 | KNOWN (rune-page staging, resolves by t=1000) |
| 2 | 2701 | 90032 | LaneMinion(200) | pos (unmatched, 438 u apart) | (12505.8,12777.2) | (12451.8,13212.2) | **ROOT** -- wrong red spawn coordinate |
| 3 | 2701-2702 | 90032-90065 | LaneMinion(200) | present only in left/right | -- | -- | CASCADE of #2 (identity churn from the head start) |
| 4 | 2732 | 91065 | Champion(100/200) | gold | 1.900 | 2.850 | INDEPENDENT -- ambient gold formula/cadence off by ~1g, champions never move so unrelated to #2 |
| 5 | 2797-3484 | 93231-116127 | LaneMinion(200) | present only in left/right, stepping down-lane | -- | -- | CASCADE of #2; gap measured constant at ~700 units (table above) |
| 6 | 2913, 2959 | 97097, 98630 | LaneMinion(100) (blue) | present only in left/right | (1167.0,4013.4) | (1160.3,4008.9) | CASCADE of #2's small blue-side twin (5.1-unit constant offset, occasionally exceeds the 8-unit match radius) |
| 7 | 3010-4849 | 100330-161635 | LaneMinion(200/100) | pos + hp, on nominally matched pairs | e.g. hp 290 vs 455; 340 vs 161 | -- | CASCADE -- matching artifact once positions are hundreds of units apart, not a shared damage bug |
| 8 | 6803 | 226755 | LaneTurret(200), red outer | hp | 1550.0 (untouched) | 1535.625 | CASCADE -- server's blue wave has started sieging red's outer turret; sim's has not |
| 9 | 8562 | 285385 | Champion(100/200) | move_speed, gold | 345 / 322.2 (pre-jump) | 370 / 0.05 (post-jump) | KNOWN -- automatic boots purchase, shop/items entirely unmodeled |
| 10 | 8987 | 299576 | LaneTurret(200/100), both outer | hp | both 1550.0 (untouched) | red 936.25 (60%), blue 1542.5 | CASCADE -- by run's end the server has a real siege in progress; the sim shows a static standoff |

## Why Gap 3 (collision-before-movement) could not be tested here

Gap 3 predicts a specific, small signature: the moment two units first come
within collision range, the sim's position should show *this tick's*
push-apart while the server's shows *last tick's* -- an effect on the order of
a single tick of movement (≤ 5.75 units for a champion, ≤ 5.4 for a 325-speed
minion), appearing exactly at first contact and nowhere else.

That signature never gets a clean chance to appear in this run:

1. **Same-team minions never collide.** They spawn 800 ms apart
   (`MINION_SPACING_MS`) and all walk the same polyline at the same speed, so
   consecutive minions of one wave maintain constant separation -- there is no
   crowding or overtaking to trigger `CollisionHandler` between minions of one
   side in an idle lane.
2. **Cross-team contact is exactly what the spawn bug corrupts.** The one place
   two units *do* get close enough to test Gap 3 -- the wave clash -- happens
   at a different place and a different relative timing in the sim than on the
   server, because red's minions are a permanent ~2.15 s ahead. Any few-unit
   position residual around that clash is indistinguishable from noise
   introduced by the much larger bug already documented above.
3. Champions never move (never ordered), so they can supply no evidence either.

**To actually test Gap 3**, the red-spawn bug (or an equivalent stand-in, e.g.
patching `MINION_SPAWN` in for `path[0]` in a scratch copy, or running a
blue-vs-blue mirror so both waves share the same, correct spawn geometry) needs
to be neutralized first, then the trace re-examined at the exact tick two
opposing minions' collision radii first overlap, watching specifically for a
same-tick-vs-next-tick position offset rather than a magnitude difference.
I did not do that here because it requires changing simulation code, which is
out of scope for a diagnosis pass.

## Where a fix would go (not applied)

`lanerl_jax/sim/step.py:168-169` -- the red-team `spawn_minion` call should
seed the minion at `MINION_SPAWN[Team.RED]` (already defined,
`lanerl_jax/sim/init.py:134-137`), with the reversed `TOP_LANE_PATH` used only
for the waypoints *after* spawn, not for the spawn point itself. The matching
blue-side call is technically inconsistent too (`MINION_SPAWN[Team.BLUE]` vs
`TOP_LANE_PATH[0]`, 5.1 units) and should probably be corrected the same way
for consistency, though its effect is far smaller. `spawn_minion` itself
(`lanerl_jax/sim/init.py:304-347`) would need a spawn-position argument
distinct from `path[0]` to support this cleanly. `test_lane.py:150-157`'s
existing barracks test should be extended to cover the red side, which is
exactly the check that would have caught this.

Separately, worth flagging to whoever owns `lanerl_jax/parity/sim_vs_server.py`'s
`NOT_MODELLED` list: `gold` and `move_speed` currently are not in it, but the
sim does not model the shop/item economy at all (finding B above), so both
fields will flag as "diverging" the moment the server auto-purchases anything,
even in a scenario with zero simulation bugs.
