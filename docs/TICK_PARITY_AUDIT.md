# Tick Parity Audit: `lanerl_jax` vs LoLServer, top-lane 1v1 Garen

Scope: two Garens, lane minions (melee/caster/cannon), lane turrets, on the
real per-tick update loop. No jungle, items, shop, runes-as-code. Vision is
touched only where it gates the loop itself (it doesn't, for this slice).

Server commit audited: `lanerl-vendor/LoLServer` @ `be1dc8b` (branch
`lanerl/wip`, 2026-09-15). This is not vanilla LeagueSandbox/LoLServer — it is
a fork with `lanerl/`-prefixed instrumentation and several already-fixed races
(see the "checked and found nothing" section). All C# citations below are
`file:line` against that commit. All JAX citations are against
`lanerl_jax/sim/*.py` in the worktree as given (uncommitted changes included).

Every claim below is tagged **VERIFIED** (I read the code path that proves it)
or **UNCERTAIN** (control flow unclear, or rests on .NET runtime internals not
vendored in this repo — the BCL's `Dictionary<TKey,TValue>` source is not
here). Nothing is presented as verified on a guess.

> **Standing note — verify the character name, not just the mechanic.**
> `Content/` holds every map's units in one tree, and C# spell/buff scripts are
> resolved by exact character or spell **name**, not by role. Reading a real
> file under `Content/LeagueSandbox-Default/Stats/<Name>/` or
> `Content/LeagueSandbox-Scripts/Characters/<Name>/` proves nothing about this
> project unless `<Name>` is a unit `lanerl/cfg/garen1v1.json`'s configured map
> actually spawns. That config pins `"map": 1` (`garen1v1.json:132`), and
> Map1's `LevelScriptObjects.cs:77-89` names its turrets
> `OrderTurretNormal`/`ChaosTurretWorm` (outer), not the `SRUAP_Turret_*`
> family, which belongs to Map11 ("New SR") and is never spawned here. An
> earlier pass of this audit read `SRUAP_Turret_Order3`'s stats and script by
> habit (it was the first turret the filesystem search surfaced) and produced
> a wrong finding as a result — corrected in Gap 1 and Gap 2 below. Before
> trusting any Content-sourced number or script in this document: check it
> against the map's `LevelScriptObjects.cs` (turrets, nexus, inhibitors) or
> `LevelScript.cs`'s `MinionModels` (minions), the same way
> `lanerl_jax/data/patch.py:277-319` already does for turrets and
> `MINION_MODELS` does for minions — both worth reading first, since they
> encode exactly this lesson already.

---

## 1. Side-by-side ordered table

"Server phase" is the order actually executed for one Garen/minion/turret
inside one `Game.Update` call, established by reading `Game.cs` →
`MapScriptHandler.cs` → `ObjectManager.cs` → `AttackableUnit.cs` →
`ObjAIBase.cs`, not assumed from the module's own docstring.

| # | Server phase (file:line) | Our phase (file:line) | Verdict |
|---|---|---|---|
| 0 | `Map.Update(diff)` runs **before** `ObjectManager.Update(diff)` — `Game.cs:481,483`. Inside it: `CollisionHandler.Update()` → `PathingHandler.Update(diff)` → `MapScript.Update(diff)` → surrenders — `MapScriptHandler.cs:96-107` | wave spawning is `tick()` step 0, before movement — `step.py` "0. wave spawning" | **DIFFERS** (see gap 3: collision is inside this phase, and it runs before movement, not after) |
| 0a | Wave spawn timer, `LevelScript.Update` → `SetUpLaneMinion()` → `CreateLaneMinion(...)` → `ApiMapFunctionManager.CreateLaneMinion` → `ObjectManager.AddObject(m)` — **`Map1/LevelScript.cs:172-201,283,307`** (the configured map is 1, not 11 — see standing note; Map1's `LevelScript` is the `CLASSIC` class in `namespace MapScripts.Map1`, structurally near-identical to Map11's but at different line numbers and with different minion model names, confirmed same wave composition at `Map1/LevelScript.cs:83-119`); `APIMapFunctionManager.cs:135-144`. `_currentlyInUpdate` is false here (`ObjectManager.Update` hasn't started this frame), so `AddObject` takes the immediate branch and the minion lands directly in `_objects` — `ObjectManager.cs:223-236` | `step_waves_jax` + `spawn_minion`, step 0 — `step.py` | **AGREES**: a minion spawned this tick is already in the collection the per-object loop is about to run, so it moves/acts the same tick on both sides |
| 0b | `CollisionHandler.Update()` — `MapScriptHandler.cs:98`, `CollisionHandler.cs:121` — runs against positions as they stood at the **end of the previous tick** (no unit has moved yet this tick) | `resolve_collisions`, called **after** `step_move_units` in the same `tick()` call — `step.py` steps 1 then 1b | **DIFFERS** — see gap 3 |
| 1 | Per object, `ObjAIBase.Update` → `base.Update(diff)` (`AttackableUnit.Update`) runs first: `UpdateBuffs(diff)` — `AttackableUnit.cs:237,239` (buff bodies at `:805-827`, e.g. Garen's passive heal) | `step_buffs(...)` is called with **pre-tick** `state.x/state.y` even though it's written after the movement call — `step.py` "1a. buffs" | **AGREES** (data-dependency correct; see gap 6 for the misleading code layout) |
| 2 | `Stats.Update`, throttled to accumulated 500 ms — `AttackableUnit.cs:242-249`; HP/mana regen formula at `Stats.cs:242-259` | not modeled at all | **MISSING** — see gap 1 / Q&A §3 |
| 3 | `Replication.Update()` — `AttackableUnit.cs:251` (not gameplay-relevant) | n/a | n/a |
| 4 | `Move`/`DashMove`, gated by `CanMove()` (`ObjAIBase.CanMove` override, `ObjAIBase.cs:302-309`) — `AttackableUnit.cs:255-263` | `step_move_units`, gated by `_can_move` — `step.py` step 1 | **AGREES** on the blocked-order set (CastSpell/None/Stop/Hold all block movement on both sides) |
| 5 | `if (IsDead && _death != null) Die(_death)` — `AttackableUnit.cs:266-269` | death handled once, in the damage/kill section at the end of `tick()` | **DIFFERS** — see gap 4 (reward/removal can lag the HP-zero tick by one tick server-side; ours never lags) |
| 6 | Back in `ObjAIBase.Update`: `CharScript.OnUpdate(diff)` — `:1070` | not modeled (out of scope: "champion action decode") | **MISSING**, acknowledged out of scope |
| 7 | `AIScript.OnUpdate(diff)` if `!_aiPaused` — `:1081`. `LaneMinionAI.OnUpdate` — `LaneMinionAI.cs:59-97` (reevaluates on a trigger: target-just-died / call-for-help / **250 ms timer**, not every tick); `TurretAI.OnUpdate` — `TurretAI.cs:21-30` (retarget if no target, drop target if out of range) | `step_minion_ai(...)` — `step.py` step 2 | **UNCERTAIN** whether our minion AI reproduces the 250 ms reevaluation cadence exactly (it does carry an `ai_timer`/`ai_local_time` pair suggesting it's modeled, but this was not independently re-derived line-by-line against `LaneMinionAI.cs` in this pass) |
| 8 | `Spells.Values.ToList().Update(diff)` — `:1092` | not modeled (spell casts out of scope) | **MISSING**, acknowledged out of scope |
| 9 | `UpdateAssistMarkers()` — `:1100` (not gameplay-relevant to this slice) | n/a | n/a |
| 10 | `UpdateTarget()` — `:1101`, body at `:1165-1330`. Handles: drop target on death/untargetable/out-of-range, `RefreshWaypoints(idealRange)` (`:595`), and the swing gate (`_autoAttackCurrentCooldown <= 0` → `IsAttacking=true` → `AutoAttackSpell.Cast(...)`, `:1247-1261`) | target acquisition (step 3), `RefreshWaypoints` re-implementation (step 3b), autoattack gate (step 4) — three separate `step.py` sections | **AGREES** in substance (server fuses these into one method; ours splits them but consumes the same already-collision/move-updated `x,y`) |
| 11 | `_autoAttackCurrentCooldown -= diff/1000` if `>0` — **after** `UpdateTarget()`, `:1105` | `aa_cooldown` decremented inside `step_autoattack`, called after target/RefreshWaypoints — `step.py` step 4 | **AGREES** (both decrement after the gate) |
| 12 | Damage lands **inside `TakeDamage`**, called synchronously from whichever code path applied it (melee: inside this same `UpdateTarget`/spell-cast call; ranged: inside `SpellMissile.CheckFlagsForUnit` on a **later tick's** `Update`, since a missile created this tick is deferred — see row 13). `IsDead`/killer transition: `AttackableUnit.cs:589-598` | damage + kill attribution, `tick()` "5. apply damage" section | **AGREES** on the kill rule itself (first hit to cross zero wins); see gap 5 for the "index" that rule is keyed on |
| 13 | `SpellMissile` is a `GameObject` updated through the *same* `foreach (var obj in _objects.Values) obj.Update(diff)` loop — `ObjectManager.cs:79-82`, `SpellMissile.cs:70-80`. A missile created **during** this tick's loop (`_currentlyInUpdate==true`) goes into `_objectsToAdd` (`ObjectManager.cs:229-231`) and isn't merged into `_objects` until after the loop (`:121-129`) — so it does not move until the **next** tick | `step_missiles`: "advance-then-spawn... a missile created at the end of a cast does not also travel on the tick it was created" — `missiles.py` module docstring + code | **AGREES**, already correctly modeled |
| 14 | Object removal sweep (dead non-turret/building objects) — `ObjectManager.cs:86-115`; `_objectsToAdd` merge — `:121-129`; vision + `LateUpdate` pass over pre-existing objects only — `:131-149` | n/a (fixed-shape state, no add/remove bookkeeping needed) | n/a — architectural, not a behavior gap |
| 15 | `ProtectionManager.Update`, chat commands, game-script timers — `Game.cs:485-488` | n/a | n/a, out of scope for this slice |

---

## 2. Gap list, ranked by potential to change observable behaviour

### Gap 1 — HP regen is not modeled at all; real for Garen, zero for turrets *on this map* (HIGH for Garen, VOID for turrets). **VERIFIED — corrected.**

*Correction: the original pass of this gap claimed turret regen = 3 HP/s,
sourced from `SRUAP_Turret_Order3`'s stats. That is a Map11 turret. This
project's config spawns Map1, whose outer turrets are `OrderTurretNormal`
(blue) / `ChaosTurretWorm` (red) — see the standing note. Both have
`"BaseStaticHPRegen": "0"` (`Content/LeagueSandbox-Default/Stats/OrderTurretNormal/OrderTurretNormal.json:25`,
`.../ChaosTurretWorm/ChaosTurretWorm.json:25`), and `lanerl_jax/data/patch.py:277-311`
already documents this — independently, by a 600 s idle-recording measurement
(turret HP monotone across 36,001 snapshots; single-hit HP drops quantized to
armour 60, not the 67 that `SRUAP_Turret_Order3` would give) — reaching the
same "map 1 turrets, zero regen" conclusion given to me as a correction. So
there is no turret regen on this map, full stop; it is not something our
omission gets wrong, and turret regen is not why a real turret outlasts ours
in any recorded difference. The arithmetic in the surviving finding below
(the 500 ms clock, `Total * diff * 0.001f`) was independently verified and is
correct; only the "which turret" input was wrong.*

Server: every `AttackableUnit.Update` accumulates `diff` into `_statUpdateTimer`
and calls `Stats.Update` once per **500 ms** (not every tick) —
`AttackableUnit.cs:242-249`. `Stats.Update` applies
`CurrentHealth += HealthRegeneration.Total * diff * 0.001f` whenever
`HealthRegeneration.Total > 0` — `Stats.cs:242-249`, with
`HealthRegeneration.BaseValue = charData.BaseStaticHpRegen` — `Stats.cs:137`
(so `HealthRegeneration.Total` is already in **HP per second**, and the
`diff * 0.001f` term is just converting the accumulated milliseconds).
- **Minions** (the ones this map actually spawns —
  `Blue_Minion_Basic`/`Blue_Minion_Wizard`/`Blue_Minion_MechCannon` and their
  Red equivalents, per `lanerl_jax/data/patch.py`'s `MINION_MODELS` and
  confirmed directly against
  `Content/LeagueSandbox-Default/Stats/Blue_Minion_{Basic,Wizard,MechCannon}/*.json:26`):
  `BaseStaticHPRegen = 0` and no per-level regen ⇒ `HealthRegeneration.Total =
  0` ⇒ genuinely zero regen. Omitting regen for minions costs nothing.
- **Turrets** (`OrderTurretNormal`/`ChaosTurretWorm`, confirmed above):
  `BaseStaticHPRegen = 0` ⇒ genuinely zero regen. **Not a gap** — nothing to
  model, nothing our omission gets wrong. (`lanerl_jax/data/patch.py` also
  flags that all 24 placed turrets currently share this one outer-turret
  profile, so inner/inhibitor/nexus tiers "shoot like an outer turret" —
  a separate, already-self-documented approximation, low-relevance for a
  laning-phase 1v1 where the outer turret is what's in play.)
- **Garen** has *two* separate healing mechanisms, on two different clocks,
  neither modeled (Garen is a champion, not a map object — his stats are not
  affected by which map hosts the match, so this part needed no correction):
  1. Ordinary `HealthRegeneration`: base **1.568** HP/s + `HPRegenPerLevel =
     0.1`/level (`Content/LeagueSandbox-Default/Stats/Garen/Garen.json:26,44`)
     on the same 500 ms cadence as everyone else.
  2. His passive "Perseverance", implemented as buff script
     `GarenPassiveHeal` with its **own independent ~1000 ms accumulator**
     (`healingTimer > 1000f`) that heals 0.4%/0.8%/2% of max HP per second
     (level-bracketed, `HEALTH_PERCENTAGES`) once he's been out of combat for
     9 s, and is cancelled the instant he takes damage
     (`ShouldPassiveTurnOff`, wired via `ApiEventManager.OnTakeDamage` —
     `GarenPassiveHeal.cs`). This is buff-script logic, ticked from
     `UpdateBuffs(diff)` every real tick (`AttackableUnit.cs:239` calls
     `buff.Update(diff)` for every active buff at `:805-827`), with the
     script doing its own internal throttling.

  In a lane with long stretches between trades, (2) alone can restore a large
  fraction of Garen's bar. Omitting all regen doesn't just miss a few HP per
  tick — it removes a mechanic that materially decides whether a trade is
  worth taking, whether poke matters, and whether an all-in is lethal. This
  is now the entire content of Gap 1: it's a Garen-only finding, not a
  turret one.

### Gap 2 — On this map, a lane turret's basic attack should be a missile with no minion damage discount; the sim currently fires it as an instant, undiscounted hit but still marks it as never a missile (HIGH). **VERIFIED, and found while re-checking Gap 1's mistake — this is live in the sim, not just in this document.**

Same root cause as Gap 1's correction: `Content/LeagueSandbox-Scripts/Characters/SRUAP_Turret_Order3/BasicAttack.cs`
(and `Chaos3`'s equivalent) is a Map11-only script implementing a custom
`OnSpellPreCast` that deals instant damage and multiplies it by 0.7× against
minions. It does not exist for Map1's actual turrets. Verified three ways:
1. No C# class named `OrderTurretNormalBasicAttack` or `ChaosTurretWormBasicAttack`
   exists anywhere in `Content/LeagueSandbox-Scripts` (repo-wide grep, no
   hits) — and scripts are resolved by exact spell/class name
   (`Spell.cs:149`: `CreateObjectStatic<ISpellScript>("Spells", SpellName) ??
   new SpellScriptEmpty()`), not by turret role, so this spell falls back to
   `SpellScriptEmpty` and `HasEmptyScript = true` (`Spell.cs:124`).
2. `OrderTurretNormal.json:38`: `"IsMelee": "false"` — the turret is ranged.
3. `Spell.FinishCasting`: `if (!Owner.IsMelee) { if (HasEmptyScript) {
   CreateSpellMissile(...) } }` — `Spell.cs:997-1002`. Ranged + empty script
   ⇒ **a missile is created**, at `MissileSpeed: 1200`
   (`Content/LeagueSandbox-Default/Spells/OrderTurretNormalBasicAttack/OrderTurretNormalBasicAttack.json:165`)
   over a ~750-range outer turret — real, non-zero flight time (~0.6 s), just
   faster than a caster minion's 500 u/s missile. Damage lands later, at
   `SpellMissile.CheckFlagsForUnit` → `ApplyEffects`/`AutoAttackHit`
   (`SpellMissile.cs:172-186`), through the *generic* auto-attack damage path
   — full AD, no minion discount, since there is no custom script left to
   apply one.

`lanerl_jax/sim/combat.py:66-92` already found and fixed the damage-discount
half of this: `TURRET_DAMAGE_VS_MINION = 1.0`, with a long comment reaching
the same conclusion as above (no `Characters/` folder for this map's
turrets ⇒ `SpellScriptEmpty` ⇒ full AD, no 0.7×) and citing the same
`SRUAP_Turret_Order3/BasicAttack.cs` as the source of the *wrong* 0.7× that
"does not apply here." **But `lanerl_jax/sim/profiles.py:96-108`'s
`fires_missile` column was not updated to match**:

```python
# `Spell.FinishCasting`: a basic attack becomes a missile only when the
# attacker is ranged AND its BasicAttack script is empty. Both are
# needed -- a lane turret is ranged and must NOT get one, because its
# own script applies the damage and leaves MissileParameters null.
cols["fires_missile"][row] = float(
    (not u.is_melee) and kind != Kind.TURRET)
```

The comment's premise ("its own script applies the damage") is exactly the
Map11 assumption `combat.py`'s comment, two files over, already refutes for
this map — Map1's turret has no "own script." The `kind != Kind.TURRET`
exclusion should not be there; on this config a lane turret's basic attack
satisfies both of `FinishCasting`'s conditions for a missile (ranged +
empty script) exactly like a caster/cannon minion does, and
`sim/missiles.py`'s existing machinery (already correctly handling
casters/cannons) would handle it the same way if the flag were set.

**Consequence, concretely**: our sim currently applies every turret
autoattack's (already-correct, full-AD) damage on the same tick as the swing.
On the real server it should land ~0.6 s (turret) later, as a missile that
can whiff entirely if the target dies, leaves range, or becomes untargetable
before it arrives (`SpellMissile.Update`, `SpellMissile.cs:70-80` — the same
"wasted damage" dynamic `missiles.py`'s docstring already describes for
casters). This matters most exactly where a top-lane 1v1 audit should care:
turret-dive and last-hit-under-tower timing. A Garen who kills his target and
turns to leave right as a tower shot is in flight should take zero damage
from that shot on the server; our sim would have already landed it. Ranked
above the collision-ordering gap because it is a binary behavior difference
(instant vs. has flight time, always-lands vs. can-whiff) on the single unit
type a top-lane dive scenario revolves around, not a positional nudge.

### Gap 3 — Collision is resolved before movement server-side, after movement in ours, and lags by a tick either way (HIGH). **VERIFIED.**
`Game.Update` runs `Map.Update(diff)` **before** `ObjectManager.Update(diff)`
— `Game.cs:481,483`. Inside `Map.Update`, `CollisionHandler.Update()` is the
*first* thing called, before `PathingHandler`/`MapScript` — `MapScriptHandler.cs:96-100`.
Since no unit has moved yet this tick when `Map.Update` runs, this collision
pass pushes apart the **positions left over from the end of the previous
tick**; movement for the current tick then happens afterward, in
`ObjectManager.Update`.

Our `tick()` does the reverse and in the same call: `step_move_units` first,
then `resolve_collisions` on the just-moved positions — `step.py` steps 1 and
1b. The comment there ("`CollisionHandler.Update` runs from `Map.Update`,
i.e. after the objects have moved") is only true across tick boundaries
(this-tick's collision uses *last* tick's move), not true within the same
`Game.Update` call the way the code computes it.

Consequence: our sim's target-acquisition and attack-range checks later in
the *same* tick see a position that has already had *this tick's* collision
push-apart baked in; the server's equivalent checks see a position that is
still one tick "behind" on collision (it reflects last tick's push, not this
tick's). This is exactly the class of bug the module docstring warns about
generically for movement-vs-buffs — it also applies to collision-vs-movement,
and this pass found it wasn't caught by that warning. Melee positioning
(Garen wedging into a minion wave, minions crowding a choke) is central to a
lane fight, so this is ranked high despite each individual push being small.

### Gap 4 — Kill rewards/removal can lag the HP-zero tick by one tick, keyed on iteration order (MEDIUM-HIGH). **VERIFIED.**
`TakeDamage` sets `IsDead = true` and records `_death` (with `Killer =
attacker`) the instant cumulative damage crosses zero —
`AttackableUnit.cs:589-598`. But the consequence of death — `Die(_death)`,
which grants the killer's gold/XP (`AttackableUnit.cs:634-664`) and calls
`SetToRemove()` — fires only from the **dying unit's own** `Update()` call,
specifically `if (IsDead && _death != null) Die(_death)` at
`AttackableUnit.cs:266-269`. If the victim's turn in the `_objects.Values`
iteration comes *before* the attacker's turn in the same tick, the victim's
own `Update()` has already run this tick (with `IsDead` still false at that
point) and won't run again until next tick — so `Die()`, and the gold/XP
grant, doesn't fire until one tick after the HP actually crossed zero.

Our sim computes death, kill attribution, and `death_rewards` all inside one
`tick()` call unconditionally (`step.py` "5. apply damage, and attribute the
kill" through the reward block) — there is no path in our sim where a reward
is delayed a tick. The skew is small in absolute time (≤16.7 ms) but it is a
real, verified structural difference in exactly the signal (gold/XP, and
hence level via `level_for_xp`) that feeds rewards.

### Gap 5 — "Lowest-index-wins" is real, but the index is dictionary iteration order, not a stable per-unit ID (MEDIUM). **VERIFIED the mechanism, UNCERTAIN the exact reuse policy.**
`TakeDamage`'s `if (!IsDead && Stats.CurrentHealth <= 0)` guard
(`AttackableUnit.cs:589`) is exactly "first `Update()` call this tick whose
damage crosses zero wins" — confirmed. "First" means first in
`_objects.Values` enumeration order, i.e., insertion order of a
`Dictionary<uint, GameObject>` (`ObjectManager.cs:27,79`). NetIds come from
one global, monotonically increasing counter (`NetworkIdManager.cs:9-16`), so
**absent any removals**, enumeration order = NetId order = spawn order.

But minions *are* removed from `_objects` on death (`RemoveObject`, exempting
only turrets/buildings — `ObjectManager.cs:97-108`), and .NET's
`Dictionary<TKey,TValue>` is documented (CoreCLR implementation, not vendored
in this repo, hence **UNCERTAIN** here specifically) to reuse a freed slot's
position for the *next* inserted entry rather than always appending. If that
holds, a newly spawned minion can land in enumeration order at the position
of an earlier-dying minion rather than strictly after every currently-alive
unit — i.e., "index" is not a permanent, monotonically-assigned priority once
minions have died and respawned into the collection.

`lanerl_jax/sim/init.py:299-312`'s `spawn_minion` already documents choosing
"lowest free slot" as the deliberate matching convention, so this is a
known, accepted approximation, not an oversight — flagging it here only
because "stable across ticks" was asked explicitly and the honest answer is
"stable for champions and turrets (never removed); not independently
verified for minions once wave-clearing has been running a while." Practical
impact in a 1v1 is narrow: it only affects tie-breaking among simultaneous
lethal hits on a target from *multiple different-wave minions*, never
Garen-vs-Garen or Garen-vs-turret (whose relative order is fixed for the
whole episode, champions/turrets being created once, before any minion).

### Gap 6 — Buff/movement code order is misleading but not a bug (LOW, informational). **VERIFIED.**
`step.py` computes `step_move_units` (step 1) before `step_buffs` (step 1a)
textually, and step 1's comment says "before anything else," which reads as
"movement precedes buffs." Functionally it doesn't: `step_buffs` is called
with the **pre-tick** `state.x`/`state.y`, not the freshly computed
post-movement `x,y` — so the actual data dependency matches the server
(`UpdateBuffs(diff)` at `AttackableUnit.cs:239` runs strictly before
`Move(...)` at `:262`). No behavior gap today. Recorded so a future refactor
that "cleans up" the statement order by feeding buffs the post-move `x,y`
doesn't silently introduce the bug the comment already describes as
dangerous.

### Gap 7 — Dead units don't get an extra swing this tick, but do still run their AI/char scripts (LOW, informational). **VERIFIED.**
`ObjAIBase.Update` calls `CharScript.OnUpdate` and (if not `_aiPaused`)
`AIScript.OnUpdate` **unconditionally** — `ObjAIBase.cs:1070,1081` — death
does not set `_aiPaused` (confirmed: the only call sites for `PauseAI(true)`
are the chat `/spawn` debug command and Map8/Ascension capture points —
`ApiFunctionManager.cs:464`, `SpawnCommand.cs:137`,
`Map8/LevelScriptObjects*.cs:106/109` — never a death path). So a unit killed
earlier in the tick still has its scripts invoked later the same tick. But
the actual swing gate, `UpdateTarget`, explicitly early-returns on `IsDead`
(`ObjAIBase.cs:1165-1173`), and `LaneMinionAI.OnUpdate` separately self-guards
on `!LaneMinion.IsDead` (`LaneMinionAI.cs:61`) before doing anything
meaningful. `TurretAI.OnUpdate` has **no** explicit `IsDead` guard of its own
(`TurretAI.cs:21-30`) — a dying turret's `CheckForTargets`/`SetTargetUnit`
calls still run this tick — but this is inert because the shared
`UpdateTarget` swing gate blocks any actual attack regardless. Net: no unit
gets a "free" attack off after dying in the same tick, on either side. This
matches the implicit assumption in our `tick()` (a unit's `alive` mask is
applied before the *next* `tick()` call's targeting/attack logic runs).

### Gap 8 — Missile-vs-movement ordering: already correct (not a gap). **VERIFIED.**
Documented here only to close out an explicit question in the task. A
missile created mid-tick (during the caster's own `Update`, i.e. while
`_currentlyInUpdate == true`) is queued into `_objectsToAdd`
(`ObjectManager.cs:229-231`) and only merged into `_objects` after the main
per-object loop finishes (`:121-129`), so it gets no `Update()`/movement until
the *next* tick — `SpellMissile.Update`, `SpellMissile.cs:70-80`.
`lanerl_jax/sim/missiles.py` already implements and documents exactly this
("advance-then-spawn... a missile created at the end of a cast does not also
travel on the tick it was created"). No discrepancy found.

---

## 3. Specific questions

**Is there HP regen? Does the server apply a regen tick, on what period, and to whom? Does Garen regen, and does his passive interact with it? Does it matter that we don't model it?**

Yes for Garen, no for anything else on this map. `AttackableUnit.Update`
accumulates `diff` and calls `Stats.Update` once per **500 ms of accumulated
time**, not every 16.67 ms tick — `AttackableUnit.cs:242-249`. `Stats.Update`
adds `HealthRegeneration.Total * diff * 0.001f` to `CurrentHealth`, clamped to
max, whenever `HealthRegeneration.Total > 0` and the unit isn't dead/full —
`Stats.cs:242-249`. `HealthRegeneration.BaseValue` comes straight from
`charData.BaseStaticHpRegen` (`Stats.cs:137`). On this map (map 1, per the
standing note) that's **0 for minions** (`Blue_Minion_{Basic,Wizard,MechCannon}.json`
and Red equivalents) *and* **0 for turrets** (`OrderTurretNormal.json`,
`ChaosTurretWorm.json`, both `:25`) — so minions and turrets genuinely never
regen here; see Gap 1's correction for the earlier, wrong "3 for turrets"
claim that was sourced from a Map11 turret this config never spawns.

Garen regens on **two independent clocks**: the ordinary 500 ms
`HealthRegeneration` stat (base **1.568** HP/s + `HPRegenPerLevel = 0.1`/level,
`Garen.json:26,44`), plus his unique passive `GarenPassiveHeal` buff, which
runs its own ~1000 ms internal timer and heals a level-scaled percentage of
max HP per second once 9 s out of combat, cancelled immediately on taking
damage. This is a *separate* mechanism from the stat-based regen (a direct
`TakeHeal` call, not a `HealthRegeneration` modifier), so it is not something
that would show up "for free" if generic HP regen were added later — it
needs its own implementation. **This matters**: we do not model any of it,
and in a 1v1 lane with real gaps between trades, Garen's passive alone can
restore a large fraction of his HP bar, directly affecting which trades are
winnable and whether all-ins are lethal. Turret and minion regen, by
contrast, do not matter — both are genuinely zero on this map, not just
unmodeled.

**What is `ObjectManager`'s iteration order, and is it stable across ticks?**

`ObjectManager.Update`'s main loop is `foreach (var obj in _objects.Values)
obj.Update(diff)` over a `Dictionary<uint, GameObject>` —
`ObjectManager.cs:27,79-82`. NetIds are assigned from one single
monotonically increasing counter shared by every `GameObject` in the game
(`NetworkIdManager.cs:6-16`), so in the absence of any removals, enumeration
order equals NetId order equals creation order — and it is stable, since
nothing reorders an already-inserted, never-removed entry. For this scenario,
**both champions and all turrets are created once, at match/map init, and
never removed** (`ObjectManager.cs:97-108` explicitly exempts
`BaseTurret`/`ObjBuilding` from the death-removal sweep) — so their relative
iteration order is fixed for the whole episode and verified stable.
**Minions** are removed on death (same exemption check), and whether a newly
spawned minion's enumeration position is strictly "after everything currently
alive" or can land at an earlier freed slot depends on .NET's
`Dictionary<TKey,TValue>` internal slot-reuse policy, which is not vendored
in this repo — **UNCERTAIN**, not independently verified here. Practically:
stable and verified for champions/turrets (which is most of what a 1v1's
kill-attribution tie-breaks will ever involve), unverified in the general
minion-vs-minion case.

**Does a unit that dies during a tick still act later in that same tick?**

No new attack, but its scripts still run. See Gap 7 above: `CharScript.OnUpdate`
and `AIScript.OnUpdate` are called unconditionally regardless of `IsDead`
(`ObjAIBase.cs:1070,1081`, and death never sets `_aiPaused`), but the shared
`UpdateTarget` swing gate hard-returns on `IsDead` before any attack logic
(`ObjAIBase.cs:1165-1173`), so no unit gets an attack off after dying earlier
in the same tick, on either side.

**Are buffs ticked before or after the unit's own update?**

Buffs are the *first* thing inside the unit's own `Update`:
`AttackableUnit.Update` calls `UpdateBuffs(diff)` at `:239`, before stat
regen, movement, or (via the `ObjAIBase` override) any of `CharScript`,
`AIScript`, spells, or `UpdateTarget`. Our sim's `step_buffs` consumes
pre-tick position, matching this (see Gap 6). "Before or after the unit's own
update" as literally asked doesn't quite parse — buffs are part of the unit's
own update, and they're the first part of it.

**How is `delta` (frame time) derived, and is it exactly 1000/60 ms per tick or variable?**

Variable by default. `Game.GameLoop` measures real wall-clock elapsed time
with a `Stopwatch` and uses that as `deltaTime` —
`Game.cs:326-331` (`lastMapDurationWatch.Elapsed.TotalMilliseconds`) — so it
jitters with OS scheduling, GC, and the previous tick's own work. It is
forced to exactly `REFRESH_RATE = 1000.0/60.0` in exactly two cases: the very
first tick of the game, to avoid `Update(0)` (`Game.cs:335-339`), and
whenever the environment variable `LANERL_FREERUN=1` is set
(`Game.cs:333`,`403-405`) — a mode this fork added specifically for headless,
non-realtime data collection/training ("free-run: fixed logical timestep,
decoupled from real time"), which also disables the inter-tick sleep
(`Game.cs:392`) so the loop runs flat-out while still reporting a fixed
16.667 ms logical `diff`. `lanerl_jax`'s `TICK_MS = 1000.0/60.0`
(`movement_jax.py:54`) matches the free-run value exactly and will **not**
match a server run outside free-run mode, where `diff` is measured and
variable tick-to-tick. Worth confirming that whatever reference traces this
project trains/validates against were collected with `LANERL_FREERUN=1` —
if any weren't, the fixed-`TICK_MS` sim is being compared against a jittered
ground truth.

---

## 4. Checked and found nothing

- **`LaneMinion.cs` / `LaneTurret.cs` override `Update`?** No — grepped both
  files; neither declares an `Update(float diff)` override, so lane minions
  and lane turrets run exactly the `ObjAIBase.Update` sequence in the table
  above. `Champion.cs:226` is the only override among the four, and it calls
  `base.Update(diff)` first (`:228`) before adding ambient gold/XP timers and
  the respawn countdown — consistent with `tick()`'s section 6.
- **Missile-vs-movement ordering** — already correctly modeled
  (advance-then-spawn); see Gap 8. No fix needed.
- **Turret's 0.7× damage multiplier vs minions — retracted, this was not
  "checked and found nothing," it was checked and found wrong.** The first
  pass of this section said the 0.7× rule was already correctly documented by
  `missiles.py`'s and `step.py`'s existing docstrings and didn't need
  re-deriving. That table (`Spell.FinishCasting`, `IsMelee`/`HasEmptyScript`,
  "lane turret ... instant, at `OnSpellPreCast`") describes
  `SRUAP_Turret_Order3`/`Chaos3` — Map11 turrets, not spawned on this map (see
  standing note). Re-derived against the actual turret and it does not hold:
  see Gap 2. `LaneTurret.AutoAttackHit` (`LaneTurret.cs`, ~lines 86-95) itself
  has only the fountain-turret special case and defers everything else to
  `base.AutoAttackHit`, which is map-agnostic and gave no signal either way.
- **`ProtectionManager.Update`, chat commands, game-script timers**
  (`Game.cs:485-488`) — confirmed these run *after* `ObjectManager.Update`,
  i.e. outside the movement/AI/combat pipeline being audited; not traced
  further (protection is a spawn-protection-style mechanic not relevant past
  very early game for a top-lane 1v1).
- **Whether `_aiPaused` is ever set on death** — grepped every `PauseAI(`
  call site (`ApiFunctionManager.cs:464`, `SpawnCommand.cs:137`,
  `Map8/LevelScriptObjects*.cs:106,109`); none is a death path. See Gap 7.
- **CoreCLR `Dictionary<TKey,TValue>` free-slot reuse semantics** —
  deliberately *not* claimed as verified; the BCL implementation is not
  vendored in this repo, so Gap 5 / the iteration-order answer marks this
  UNCERTAIN rather than asserting a specific reuse policy from memory of the
  runtime rather than from this codebase.
- **Neutral-camp minion spawning (`NeutralMinionSpawn.*`)** and Map8
  (Ascension)-specific scripts — out of scope (no jungle in this scenario);
  found but not read.
- **Items, shop, runes-as-code, vision-as-code** — out of scope per the task;
  `ObjectManager`'s vision machinery (`TeamHasVisionOn`, `UpdateTeamsVision`,
  the backwards-iteration fog fix and its extensive in-repo audit comments at
  `ObjectManager.cs:277-345`) was read only far enough to confirm it runs in
  the vision/`LateUpdate` phase, after the combat-relevant per-object loop,
  and does not gate movement/targeting/damage for this scenario.
- **`Minion.cs` (83 lines)** — confirmed it declares no `Update` override and
  is a thin shared base; not read further since `LaneMinion`/`LaneMinionAI`
  are the operative files for this scenario.

---

## Summary for the impatient

Highest-value fixes, in order: **(1)** flip `profiles.py`'s `fires_missile`
so lane turrets fire one, like any other ranged unit with an empty script —
`combat.py` already did the hard part (identifying that this map's turrets
have no custom script and get no 0.7×-vs-minion discount) and this is the one
line that didn't get updated to match; right now turret damage lands a tick
early and can never whiff, when on the server it travels for ~0.6 s and can
miss a target that dies or leaves range first — exactly the dynamic that
matters for turret-dive timing in a top-lane 1v1. **(2)** model regen for
Garen specifically — his passive heal is a distinct ~1 s-cadence mechanism
(a direct heal, not a `HealthRegeneration` modifier), not something generic
regen support would give for free; turret and minion regen do not need
modeling, both are genuinely zero on this map. **(3)** swap the
collision/movement order (collide-on-last-tick's-rest-position, then move) or
at minimum verify the current move-then-collide order doesn't matter enough
to bother — right now it's an unverified "probably small" that a HIGH-ranked,
VERIFIED ordering difference doesn't deserve. **(4)** be aware that gold/XP/
level can be off by one tick relative to the server around kills, which
matters more for reward-signal debugging than for the physics sim itself.
Missile-vs-movement ordering (for missiles that already exist) and the
dead-unit-can't-swing-again property are already correct — don't spend time
re-verifying either. And: before trusting any other Content-sourced number in
this codebase, check it against the map's `LevelScriptObjects.cs`/
`LevelScript.cs` first — that's how both mistakes in this document's first
pass got made, and how both got caught.
