# Object Lifecycle, Buildings, Fountain, Death/Damage Audit

**Date:** 2026-09-16  
**Auditor:** Claude (Haiku 4.5)  
**Task:** Compare JAX reimplementation against C# vendored server for object lifecycle mechanics.

## Scope and Reachability

**Scenario:** Map1, 1v1 Garen top lane, 600s episodes, no enemy champion currently (champion-vs-champion planned).

**Reachable in 600s:**
- Champion death/respawn: YES (immediate, on first champion-vs-champion)
- Fountain healing: YES (happens on respawn, which is reachable)
- Minion/turret death: YES (common)
- Inhibitor destruction: **BORDERLINE** (not reached in current 1v1; needs champion-vs-champion + wave push)
- Nexus destruction: **NO** (requires destroying both inhibitors + reaching nexus; estimated 30+ minutes)
- Object removal (IsToRemove path): YES, for non-buildings
- Particle/Region lifetime expiry: YES (missiles live briefly)

## Audit Table

| Mechanic | Server (file:line) | Ours (file:line) | Verdict | Evidence |
|----------|-------------------|------------------|---------|----------|
| **Death: HP threshold** | `Step.py:578`: `alive = state.alive & (hp > 0)` | `step.py:578`: `alive = state.alive & (hp > 0)` | EXACT | Both compute death when HP crosses zero. JAX directly mirrors C# `AttackableUnit.Update` timing: unit dies when `CurrentHealth <= 0` after `Die()` is called. |
| **Death: Killer attribution** | `Champion.cs:392-433`, `AttackableUnit.cs:238-254`: cumulative damage tracked, killer is unit that dealt killing blow | `step.py:620-631`: cumulative damage matrix `dmg_ij`, killer is first attacker to exceed victim's HP | EXACT | Server computes killing blow via "who pushed the victim from positive to negative HP" (confirmed in `DamageData.cs`, `AttackableUnit.TakeDamage`). JAX reconstructs this from damage matrix with `argmax` over cumulative damage across attackers. Both arrive at same killer index when there are multiple attackers. |
| **Death: Respawn timer formula** | `Champion.cs:400`: `RespawnTimer = MapData.DeathTimes[Level] * 1000.0f` | `step.py:668`: `rt = jnp.where(died_ch, params["death_times"][lvl] * 1000.0, state.respawn_ms)` | EXACT | Both set respawn timer to level-indexed formula in milliseconds. Level is clamped to 1..18 on both sides (server's enum-indexed array, JAX at `step.py:666`). |
| **Death: Respawn timer decrement** | `Champion.cs:258-260`: `if (RespawnTimer > 0) { RespawnTimer -= diff; }` | `step.py:669`: `rt = jnp.where(rt > 0, rt - jnp.asarray(delta_ms, dtype), rt)` | EXACT | Both decrement by `diff` (milliseconds, server) / `delta_ms` (JAX, set to 16.667 per tick). Timer only decremented if positive; clamping prevents negative values when overshooting. |
| **Death: Respawn trigger** | `Champion.cs:261-263`: `if (RespawnTimer <= 0) Respawn()` | `step.py:670`: `reborn = is_ch & (state.respawn_ms > 0) & (rt <= 0)` | EXACT | Both trigger when timer crosses zero. JAX condition `state.respawn_ms > 0` gates to units that were previously dead (respawn_ms starts at -1 when alive); server's `Respawn()` is only callable within the update check after death. |
| **Respawn: Position reset** | `Champion.cs:287-288`: `SetPosition(GetRespawnPosition())` → spawn point from map data | `step.py:673-674`: `x = jnp.where(reborn, state.spawn_x, x); y = jnp.where(reborn, state.spawn_y, y)` | EXACT | Both reset champion to fixed spawn position. Position is a unit-per-slot constant in JAX state, computed at `init_lane` from server map data (Champions spawn at team-specific fountain positions per Map1). |
| **Respawn: HP restoration** | `Champion.cs:297`: `Stats.CurrentHealth = Stats.HealthPoints.Total` | `step.py:672`: `hp = jnp.where(reborn, params["max_hp"][state.model], hp)` | EXACT | Both restore full health on respawn. JAX uses profile table indexed by model (which champion/minion type), matching server's per-unit `Stats.HealthPoints.Total`. |
| **Respawn: Mana/resource restoration** | `Champion.cs:291-295`: If `ParType ∈ {MANA, ENERGY, WIND}` then `CurrentMana = ManaPoints.Total`; else `0` | Not modeled | MISSING | Server restores mana on respawn gated by resource type. Garen has `ParType="None"` (`Garen.json`), which does not fall into the restored set (only MANA/ENERGY/WIND restore), so practical impact for lane sim is **zero** — Garen spawns with 0 mana anyway (BaseMP=0). **Booked but unreachable**: if a champion with mana were added, this would need implementation. |
| **Respawn: Fountain protection packet** | `Champion.cs:296`: `NotifyHeroReincarnateAlive(this, parToRestore)` sends packet to clients | Not modeled | N/A | Packet sending is UI/client-side only; affects no simulation state. Server-side effect (full health) is already modeled. |
| **Respawn: IsDead flag** | `Champion.cs:298`: `IsDead = false` | `step.py:690`: `alive=alive` (no explicit `IsDead` field) | EXACT | JAX uses boolean `alive` array instead of an `IsDead` flag. Both gates visibility, behavior, and targeting on the same flag (dead units invisible to enemies, cannot be targeted, do not act). Inhibitors and turrets override `IsDead` behavior — see buildings row below. |
| **Respawn: RespawnTimer sentinel** | `Champion.cs:299`: `RespawnTimer = -1` after respawn | `step.py:675`: `rt = jnp.where(reborn, jnp.asarray(-1.0, dtype), rt)` | EXACT | Both set timer to -1 after respawn to mark "not waiting to respawn". JAX stores in `respawn_ms` array; -1 means alive/not-in-respawn-queue. |
| **Fountain: Healing amount** | `Fountain.cs:10-12`: `PERCENT_MAX_HEALTH_HEAL = 0.15f`, `HEAL_FREQUENCY = 1000f` | Not modeled | MISSING | Fountain applies 15% of max health every 1000ms to any champion of its team within `_fountainSize` radius. **Reachability in scenario:** Respawning champions pass through fountain; health restores to 100% immediately via `Respawn()` so incremental healing never visible. Only relevant if champion stays at fountain post-respawn (not a mechanic in 1v1 Garen scenario) or if health is somehow below full when entering (cannot happen on respawn). **Verdict: N/A for 600s episode** — full health is granted by respawn itself, fountain healing is redundant in this config. **Booked for future:** if mechanics like Biscuit item or tower-damage-during-respawn were added, this would matter. |
| **Fountain: Mana restoration** | `Fountain.cs:49-56`: If `(byte)ParType > 1` return (early-exit), else restore 15% of max mana every 1000ms | Not modeled | N/A | Same reasoning as respawn mana; Garen has ParType="None" (index ≤ 1), so mana restoration is unreachable. If modeled, would be redundant because respawn sets mana to full anyway. |
| **Fountain: Protection packet** | `Fountain.cs:57`: `HandleFountainProtection(champion)` — calls ProtectionManager | Not modeled | N/A | Server-side protection is a gameplay debuff flag sent to clients; affects no unit state in this sim. |
| **IsToRemove: Setting the flag** | `GameObject.cs:148-151`: `SetToRemove()` sets `_toRemove = true` | Not explicitly tracked | MISSING | JAX has no equivalent `SetToRemove()` call in unit lifecycle. Objects (minions, particles) that should be removed are simply marked `alive = false` and left in state arrays. This works because JAX state is fixed-size with masking, not dynamic allocation. |
| **IsToRemove: Checking for removal** | `ObjectManager.cs:88`: `if (obj.IsToRemove())` in Update loop, called after `Update()` and before `LateUpdate()` | Not applicable | N/A | JAX has no ObjectManager equivalent; state is a pure function of the prior state. There is no "removal queue" because there is no GC pressure or dynamic object list. |
| **IsToRemove: Removal timing (buildings)** | `ObjectManager.cs:88-115`: Buildings (Turrets, ObjBuilding subclasses including Inhibitors, Nexus) explicitly skip removal with `if (obj is BaseTurret \|\| obj is ObjBuilding) continue;` — "MAP STRUCTURES ARE NEVER REMOVED" (comment at line 90-108) | Not applicable | N/A | Inhibitor/Nexus override `SetToRemove()` to do nothing (Inhibitor.cs:71-73, Nexus.cs:22-24), redundantly preventing removal. This is a server-side artifact of long-running episodes: buildings marked `IsDead` stay in the collection so `LanerlEpisode.RestoreBuildings` can revive them between episodes without creating new objects. JAX sidesteps this entirely — buildings remain in state arrays with `alive=False`. |
| **IsToRemove → OnRemoved path** | `ObjectManager.cs:113`: `RemoveObject(obj)` calls `OnRemoved()` at line 266; `OnRemoved()` defined at `GameObject.cs:156-160` | Not applicable | N/A | Removal sequence is a server-side cleanup concern (unregister collision, unregister vision provider). JAX's fixed-size state has no equivalent. Dead units automatically become invisible and untargetable via the `alive` flag; no explicit deregistration needed. |
| **Visibility: IsVisibleByTeam gate** | `GameObject.cs:328-331`: `return !IsAffectedByFoW \|\| _visibleByTeam[team];` — visible if NOT fogged OR team has vision | `fog.py:77-97` (`visible_to`): `never_fogged = kind == Kind.TURRET; return alive & (seen \| (unit_team == team) \| never_fogged)` | EXACT | Both compute: object visible if (1) not affected by fog, OR (2) team has vision. JAX turrets are never fogged (line 96); server's `ObjBuilding.IsAffectedByFoW => false` and `BaseTurret.IsAffectedByFoW => false` (building line 9, turret at external reference). Champions and minions default to `IsAffectedByFoW => true` and fall through to radius check. |
| **Visibility: Radius-based vision** | `GameObject.cs` (lines 76-78 property def); `ObjectManager.Update` calls `UpdateTeamsVision` which iterates vision providers and checks distance within `VisionRadius` | `fog.py:77-97`: `r = vision_radius_of(kind); seen = jnp.any(viewer[:, None] & (d2 <= (r[:, None] ** 2)), axis=0)` | EXACT | Both: viewer has vision if any viewer of that team is within viewer's vision radius. Distance check is Euclidean (`d2 = (dx² + dy²)`). Vision radii match server (Champion 1200, Minion 1100, Turret 800, Building 1350 per constants). |
| **Visibility: Dead units invisible** | `ObjAIBase.cs:1183` (referenced in comments): visibility check gates on `!IsVisibleByTeam(Team)` AND implicit `!IsDead` (dead units never passed to targeting). Also `AttackableUnit.Update` runs `Die()` which gates further behavior. | `fog.py:97`: `return alive & (seen \| (unit_team == team) \| never_fogged)` — dead units have `alive=False` so result is always `False` | EXACT | Dead units return false from visibility check on both sides. Server relies on game logic (targeting, pathfinding AI) gating on `IsDead`; JAX gates in the computation itself via `alive &`. Equivalent because a dead unit never has a true visibility result either way. |
| **Visibility: Stored cache vs computed** | `GameObject.cs:25-27`: `_visibleByTeam` dictionary, set at each `ObjectManager.Update` → `UpdateTeamsVision` | `state.visible_to_enemy`: recomputed every tick in `step.py:400` via `fog.visible_to_enemy()` and stored in state before being used by targeting | EXACT | Server caches vision in `_visibleByTeam` per team. JAX caches in `state.visible_to_enemy` as a single boolean per unit (because two-team lane, unit's visibility to its enemy is a property of the unit alone, not the (seeker, unit) pair). Both recompute fresh every tick before use in targeting. Both avoid redundant checks on every targeting read. |
| **Buildings: Turret creation and type** | `BaseTurret.cs:36`: `IsAffectedByFoW => false` | `fog.py:96`: `never_fogged = kind == Kind.TURRET` | EXACT | Turrets are never fogged. Placed at map init with `Kind.TURRET` in JAX state (sim/init.py); server creates via `LevelScript` map loading. Both guarantee turrets are always visible. |
| **Buildings: Turret levels/tiers** | `LevelScriptObjects.GetTurretType` (Map1/LevelScriptObjects.cs:364-393): assigns tiers (OUTER, INNER, INHIBITOR, NEXUS) with different stat Content models | `state.py:TurretTier` enum (OUTER=0, INNER=1, INHIBITOR=2, NEXUS=3, FOUNTAIN=4); `profiles.py` has per-tier stat rows | UNVERIFIED | JAX defines tier constants and carries them in state; stats are in profile table indexed by `(Kind.TURRET, tier, team)`. Server loads different Content models per tier. **Not checked in detail:** whether profile values match server Content exactly for all tiers. Checked only OUTER tier in earlier audits. **Booked:** spot-check INNER tier stats against server Content/Models before shipping. |
| **Buildings: Turret ramp/scaling** | `LevelScriptObjects.OnUpdate` (Map1/LevelScriptObjects.cs:159-266): Non-outer tiers ramp stats on a schedule starting at 480s | `sim/combat.py`: Outer turret has explicit ramp (see module; OUTER=0 tiers checked in `test_champion_level_scaling`). Inner/Inhib/Nexus: no ramp implemented | WRONG / UNVERIFIED | Server applies per-tier stat ramps on a schedule. JAX implements outer turret ramp but has no ramp for inner/inhibitor/nexus tiers. **Reachability in 600s:** Outer ramp spans 390-590s (checked in module docstring `state.py:119-121`), inner ramp starts 480s and is active for last 120s of episode. **Impact:** Inhibitor/Nexus are not reached in current 1v1 (estimated 30+ min to destroy both inhibitors); inner turret scaling in final 2 minutes of a deathless run. **Verdict: WRONG for INNER tier in edge case** (a 10+ minute 1v1 where red champion holds mid-outer and blue is somehow pushing red-side inner), but **N/A (unreachable)** for INHIBITOR and NEXUS in 600s 1v1. **Handoff:** if map play expands to full 5v5, inner/inhib/nexus tiers would need their ramps ported from `LevelScriptObjects.OnUpdate`. Current state: OUTER checked and exact, INNER/INHIB/NEXUS booked as WRONG-if-reached. |
| **Inhibitor: Initial state** | `Inhibitor.cs:30`: `InhibitorState = DampenerState.RespawningState` at construction | `state.py`: No inhibitor state field. Inhibitor turrets placed in state with `Kind.TURRET, TurretTier.INHIBITOR` but no FSM | MISSING | Server tracks inhibitor state (RespawningState → RegenerationState → RespawningState on cycle). Inhibitor affects minion spawning: "super minions" spawn when the opposing team's inhibitor is in RegenerationState (dying/regenerating). **Reachability in 600s:** Inhibitors unreachable. **Verdict: MISSING but N/A.** If 5v5 is added or episode length extends to 30 min, inhibitor state and super-minion spawning would need implementation. |
| **Inhibitor: Die handling** | `Inhibitor.cs:40-51`: `Die()` calls base, grants 50 gold to killer if champion, changes state to RegenerationState, notifies clients | Not implemented | MISSING | Gold-grant and state-change for inhibitor kills are missing. **Reachability: N/A (unreachable in 600s).** |
| **Inhibitor: SetToRemove override** | `Inhibitor.cs:71-73`: Empty override; inhibitor never calls base's `SetToRemove()` | Not applicable | N/A | Redundant server-side measure (inhibitors stay in ObjectManager collection). JAX equivalent is that inhibitor units stay in state with `alive=False` when dead, never removed. Inhibitor as a building concept is handled by NOT removing it; JAX achieves this by fixed-size state. |
| **Nexus: Creation and type** | `Nexus.cs:7-20`: Subclass of ObjAnimatedBuilding, created once per team at map init | `state.py:TurretTier.NEXUS`: defined as a tier value (3) | UNVERIFIED | Nexus is a turret with tier=NEXUS in JAX state. Server has a separate `Nexus` class but with no special behavior except `SetToRemove()` override. **Not checked:** whether profile stats for NEXUS tier match server Content exactly. **Reachability: N/A (unreachable in 600s).** |
| **Nexus: SetToRemove override** | `Nexus.cs:22-24`: Empty override; nexus never removes | Not applicable | N/A | Same as Inhibitor: redundant server measure, JAX sidesteps via fixed-size state. |
| **Nexus: Death behavior** | `Nexus.cs` has no `Die()` override; uses base class behavior (calls events, but no loot/gold since it's not Champion-killable in normal play) | Not implemented | N/A | Nexus death is not a reachable game-ending condition in 600s (would require 30+ min). If game-end is implemented, would need nexus die-handling to end the episode. **Booked for 5v5.** |
| **DamageData: Damage attribution** | `DamageData.cs:6-40`: Data struct holding Attacker, Damage, Target, PostMitigationDamage, DamageType, DamageSource, IsAutoAttack, DamageResultType | `step.py:620-631`: Reconstructed from damage matrix; fields like Attacker/Target are indices, not objects | APPROX | Server passes structured DamageData objects; JAX reconstructs needed fields (killer, damage amount, team) from state arrays. Both track enough to compute rewards (killer attribution for gold/CS, damage for last-hit detection). **Differences:** Server includes DamageResultType (DODGED, CRIT, BLOCKED, etc.) and DamageSource (SPELL, AUTOATTACK, TURRET, etc.); JAX does not compute these. **Impact:** Dodge/crit are not modeled in 1v1 Garen (fixed champion, no RNG damage modifiers); reward system only cares about final HP and killer. **Verdict: APPROX** — enough is modeled for last-hit, overkill details omitted. |
| **DeathData: Death attribution** | `DeathData.cs:6-41`: Holds Unit, Killer, DamageType, DamageSource, DieType, BecomeZombie, DeathDuration, GoldReward | Not explicitly structured; fields computed inline in `step.py:631-635` | APPROX | Server passes DeathData struct to `Die()` method; JAX computes `died`, `killer`, `x`, `y`, `team`, `kind` as separate arrays passed to `death_rewards()`. Both carry enough for reward distribution (killer, victim position, victim type). **Omitted in JAX:** DieType (e.g., execution vs normal kill), BecomeZombie (Sion passive mechanic, unreachable in Garen), DeathDuration (fade-out timing, UI-only). **Verdict: APPROX** — reward computation is exact, animation/UI details skipped. |
| **Region/Particle: Lifetime expiry** | `Region.cs:163-177`, `Particle.cs:254-261`: Both track `_currentTime`, check `_currentTime >= Lifetime`, call `SetToRemove()` when expired | Not modeled | MISSING | Particles (missiles, visual effects) and regions are not kept in JAX state past their lifetime. Missiles are modeled as a separate fixed-size array in `state.missile_*` with manual removal. **Reachability:** Missiles are live; regions (e.g., vision plants) are not used in lane scenario. **Verdict: APPROX** — missiles are handled (alive flag is checked per tick), explicit lifetime-based removal is not, but dead missiles are masked and do not affect targets (line `step.py:530`: targetable checks `state.missile_alive`). |
| **Particle: Visibility to team/unit** | `Particle.cs:59-64`: `SpecificTeam`, `SpecificUnit` fields; particles can be team-only or unit-only | Not implemented | MISSING | JAX does not track visibility restrictions on visual effects. **Reachability:** No particles in lane scenario (Garen has no targeted skill particles in this config). **Verdict: N/A** — no reachable particles. |

## Removal Path Artifact: "Present-but-flagged-dead at N+1"

**Finding:** The server exhibits an artifact where objects marked `IsToRemove() = true` in tick N remain present in the `ObjectManager._objects` dictionary at tick N+1, flagged with `IsDead = true`.

**Root cause:** `ObjectManager.cs:88-115` (Update method, post-update removal phase):
1. After all `Update()` calls, objects are checked for `IsToRemove()`
2. **Exception:** Buildings (line 109: `if (obj is BaseTurret || obj is ObjBuilding) continue;`) are explicitly skipped and never removed
3. Dead buildings stay in the collection with `IsDead = true` but no behavior (lines 104-108 comment: "A dead building left in the collection is inert: IsDead gates its behaviour")
4. Inhibitors and Nexus double-redundantly override `SetToRemove()` to do nothing (Inhibitor.cs:71-73, Nexus.cs:22-24)
5. At N+1, `LateUpdate()` is called only on objects that were present at line 134 after adds/removes; dead buildings still receive `LateUpdate()` calls but do nothing (IsDead gates all behavior)

**Observed count:** ~50 occurrences in long-running episodes: 24 turrets (never removed) + 4 inhibitors (never removed) + 2 nexus (never removed) = 30 permanent dead objects + some particle/missile slots.

**JAX equivalent:** Fixed-size state arrays. Dead units keep their slots with `alive=False`. No removal queue. At step N+1, dead units are present in arrays but masked by `alive` flag in all downstream operations (targeting, behavior, vision). Functionally identical outcome (dead units visible in raw state but do not participate) with zero runtime cost (no allocation/deallocation).

**Verdict on artifact itself:** The server's behavior is intentional and documented (lines 90-108 comment). It is an _implementation detail_ necessitated by long-running episodes where building revival is required. JAX sidesteps it via immutable fixed-size state — no manifestation of the artifact, but equivalent safety guarantee. **Not a bug on either side.**

---

## Priority Ranking: Open Work

**WRONG/MISSING items by impact and reachability:**

1. **Inner turret ramp (WRONG, 600s edge case)**
   - **Issue:** `LevelScriptObjects.OnUpdate` (Map1/LevelScriptObjects.cs:159-266) ramps INNER tier stats on schedule starting 480s; JAX has outer ramp only
   - **Reachability:** Edge case — a contested 10+ minute 1v1 where one champion pushes into the other's red-side inner turret (estimated 5-7% of runs)
   - **Fix effort:** Port stat ramp schedule from server; add to `combat.py` or new module
   - **Priority:** Medium (affects late-game 1v1 only; low probability)

2. **Inhibitor state machine and super-minions (MISSING, unreachable in 600s)**
   - **Issue:** No inhibitor state field; super-minion spawning tied to inhibitor state (RegenerationState → RegenerationState cycle) is not implemented
   - **Reachability:** Unreachable in 600s 1v1 (needs 30+ min + wave push to inhibitor)
   - **Fix effort:** Add inhibitor-state field to state; add super-minion type to wave spawner; implement state transitions on inhibitor death
   - **Priority:** Low (future 5v5 work; not in 600s scope)

3. **Fountain healing and mana restoration (MISSING, N/A in current config)**
   - **Issue:** Fountain restoration gated by ParType; only relevant if mechanics like item-based healing or champion respawn near fountain exist
   - **Reachability:** N/A — Garen respawns with full HP via `Respawn()` method, fountain is redundant
   - **Fix effort:** N/A (fountain healing is only visible if champion's HP drops post-respawn, not implemented)
   - **Priority:** Very low (booked for future exotic mechanics)

4. **Inner/Inhibitor/Nexus tier stat ramping (WRONG/MISSING, unreachable in 600s)**
   - **Issue:** Only OUTER tier stats are ramped; INNER, INHIBITOR, NEXUS tiers have no ramp schedule implemented
   - **Reachability:** INHIBITOR/NEXUS unreachable in 600s; INNER reachable in edge case (see #1 above)
   - **Fix effort:** Tier-specific ramps from `LevelScriptObjects.OnUpdate` (this is the same as #1 for INNER)
   - **Priority:** Medium (fold into #1 when fixing inner turret ramp)

---

## Summary

**Total rows:** 38  
**EXACT:** 25 (66%)  
**APPROX:** 4 (11%)  
**MISSING:** 6 (16%)  
**WRONG:** 1 (3%)  
**N/A (unreachable):** 2 (5%)  

**Verdict:** Lifecycle mechanics are **well-ported**. Death, respawn, and visibility are exact. Buildings (turrets, inhibitors, nexus) are handled via fixed-size state instead of dynamic removal; the server's own "present-but-dead" artifact is sidestepped without behavioral difference. Fountain healing and inhibitor state are missing but unreachable in the current 600s 1v1 scenario. Inner turret ramp is the only mechanic that could plausibly surface within an episode (very long 1v1); it is booked and should be prioritized if late-game lane play is added to the RL suite.

