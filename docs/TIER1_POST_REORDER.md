# Tier 1 after the tick reorder — 19,800 minion-bearing tick-pairs

One-step injected differential: take the server's state at tick N, load it into
the sim, step exactly one tick, compare against the server's tick N+1. No
accumulation, so each disagreement is attributable to that tick alone.

Recorded on `desktop` (job 807), idle lane, no orders, 420 s of game time;
25,202 snapshots, the 19,801 at or after the first wave used. 910,216 unit
injections.

## Exact

```
Champion.position      39600/39600   100.00%
Champion.hp            39600/39600   100.00%
Champion.move_order    39600/39600   100.00%
Champion.waypoints     39600/39600   100.00%
LaneTurret.position   475200/475200  100.00%
LaneTurret.move_order 475191/475200  100.00%
LaneTurret.waypoints  475200/475200  100.00%
LaneTurret.hp         475116/475200   99.98%
```

Champion and turret mechanics are, on a one-step basis, correct.

## Minions, and the harness ceiling

```
LaneMinion.hp          388524/390275  99.55%
LaneMinion.position     88557/109200  81.10%   (trustworthy-injection subset)
LaneMinion.waypoints    89418/109200  81.88%
LaneMinion.move_order  339488/390275  86.99%
```

**Read the position number carefully.** The median error among the inexact is
**0.0625** — exactly 1/16, one quantisation step of the dump itself. p95 is
5.44 and max 7.97. So most "inexact" positions are off by a single quantum.

The mean signed error is **+1.07**: still one-sided, the sim still slightly
ahead along its own heading. That is the same signature that identified the
collision-ordering bug, reduced but not gone. Something smaller remains in the
movement/collision path.

**The ceiling.** All 19,800 ticks had at least one unit whose movement
injection was flagged untrustworthy, and the split is stark: 81.10% exact on
the trustworthy subset against 40.61% on the untrustworthy one. Minion
position/heading accuracy is currently capped by what the injector can
reconstruct, not by the simulation. **Gate 1 cannot be called green from this
run**, and improving the injector is now on the critical path for that gate.

## Damage is one-sided, and mostly a blind spot

```
HP-change disagreement:  server-only 1751, sim-only 0
deaths:                  SIM ALIVE BUT SERVER DEAD 1767, reverse 0
```

Perfectly one-sided: the sim under-damages and never over-damages. But:

```
missile in flight on 74.3% of ticks (14715/19800)
  HP-disagreement rate WITH a missile in flight : 11.5%
  HP-disagreement rate on missile-free ticks    :  1.0%
```

An 11x difference. The injector cannot see in-flight missiles — they are not in
the state dump at all — so damage the server is about to land from a missile
launched *before* the injected tick never lands in the sim. Most of the
under-damage is therefore the harness, not the model. The residual 1.0% on
missile-free ticks is the real number to chase.

## Open, in priority order

1. **Injector trustworthiness for minion movement.** It caps gate 1. Until it
   improves, 81% is a statement about the harness.
2. **Wave spawning: 1,665 of 1,667 spawn ticks disagreed on count.** Almost
   every one. This is very likely a harness artifact — the injector replays
   wave state and the sim then spawns on top of injected units — but it is
   unverified and it is alarming enough to check before trusting any
   population number from this instrument.
3. **The residual +1.07 one-sided position bias**, on missile-free trustworthy
   ticks.
4. **50 "server rows present-but-flagged-dead at N+1"** — candidates for the
   audit's Gap 4, where `Die()` fires from the victim's own next `Update` and
   so can lag the HP-zero tick by one.
