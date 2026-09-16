# Port Audit: Items, API Events, and Scripting Infrastructure

## Question 1: LANERL_AUTOBUY and Doran's Shield

### What is granted
**Item 1054 (Doran's Shield)** is automatically granted to the server's champion when `LANERL_AUTOBUY != "0"` (default enabled).

- **Env var control**: `LANERL_AUTOBUY` (default: enabled)
- **Location in build path**: First item in default BuildPath
- **Build path**: `{ 1054, 2003, 2003, 1001, 3047, 3134, 3068, 3035 }` (LanerlConfig.cs:327)
- **Purchase mechanism**: `ch.Shop.HandleItemBuyRequest(_cfg.BuildPath[st.NextItem])` (LanerlHooks.cs:421)

### Stats conferred by Doran's Shield

From `Content/LeagueSandbox-Default/Items/1054/1054.json`:
- **FlatHPPoolMod**: 80 (line 41)
- **FlatHPRegenMod**: 1.2 (line 42) — note: this is the flat field, not percent
- **FlatArmorMod**: 0 (line 34)

From `Content/LeagueSandbox-Scripts/Items/Passives/DoransShield.cs` (ItemID_1054):
- **OnActivate** adds 1.2 to `HealthRegeneration.BaseBonus` (line 13)
- Registers the StatsModifier with the owner (line 14)

### Total asymmetry summary

**Doran's Shield grants:**
- **+80 HP** (flat, from FlatHPPoolMod)
- **+1.2 HP/sec** (flat, from FlatHPRegenMod + ItemPassive script's BaseBonus)

**Is the ItemPassive script live on Map1?**
YES. ItemPassives are loaded for ALL champions, regardless of map, whenever an item is added to inventory (Inventory.cs:80-91). The script loading does not check map.

**CORRECTED — there is NO HP deficit.** An earlier revision of this document claimed the server's champion runs at 834 effective HP against our 754, a 10.6% deficit confounding every server-side measurement. That is **wrong**. `sim/init.RUNE_HP_BONUS` is defined as `754.248046875 - 616.28`: the gap between Content's base HP and the value the state dump reports *during play*, which already includes everything the server has bought, Doran's Shield included. Verified against the built profile table: our champion's `max_hp` is 754.248047, matching the dump to its own 1/1024 quantisation. The original claim was arrived at by adding the item's +80 to a number that already contained it; acting on it would have given us 834 against the server's 754 and inverted the comparison while appearing to fix it.

**What WAS missing is the regen**, and it is now fixed (`44ddfa9`). `ItemPassives/DoransShield.cs` does `HealthRegeneration.BaseBonus += 1.2f`; our champion's `hp_regen` was Garen's Content 1.568 and is now 2.768. Regen was missed precisely because the state dump does NOT expose it, so unlike max HP, AD and armour it was taken from Content and never checked against the oracle. Unmodelled this was up to 1.2 * 600 = 720 HP of healing per episode. Note also that `FlatHPRegenMod` in `1054.json` is read by no C# at all (`ItemData.cs:81` reads only `FlatHPPoolMod`) — the regen comes solely from the passive script.

**Do not attribute gate-3's death gap to this.** With max HP already matching, the gap is unexplained; the regen fix pushes our champion toward MORE survivability, not less, so if the gap persists that is a finding in its own right.

---

## Question 2: Event System — Live Subscribers on Map1

### Events WITH live subscribers on our roster

The following events have ACTIVE listeners attached to units in a typical Map1 lane game (Garen + minions + turrets):

| Event | Subscriber | Unit | File | Line | Purpose |
|-------|-----------|------|------|------|---------|
| `OnUpdateStats` | GarenPassive buff | Garen | `Buffs/Garen/GarenPassive.cs` | 40 | Updates tooltip on level-up |
| `OnTakeDamage` | GarenPassiveCooldown buff | Garen (when passive running) | `Buffs/Garen/GarenPassiveCooldown.cs` | (see file) | Resets passive out-of-combat timer on damage taken |
| `OnTakeDamage` | GarenPassiveHeal buff | Garen (when passive healing) | `Buffs/Garen/GarenPassiveHeal.cs` | (see file) | Tracks damage taken during healing phase |
| `OnPreAttack` | GarenQ buff | Garen (when Q buff active) | `Buffs/Garen/GarenQ.cs` | (see file) | Overrides next auto-attack mechanics |
| `OnPreTakeDamage` | GarenW buff | Garen (when W buff active) | `Buffs/Garen/GarenW.cs` | (see file) | Applies damage reduction |

**Minion AI (LaneMinionAI.cs)**: No event listeners. Purely update-driven (OnUpdate loop).

**Turret AI (TurretAI.cs)**: No event listeners. Purely update-driven (OnUpdate loop).

### Events with NO subscribers on Map1

The following events ARE published by `ApiEventManager` but have **zero live listeners** on Map1 with our roster:
- `OnCollision`
- `OnCollisionTerrain`
- `OnDeath` (no listeners except for nexus/inhibitor on death, not for lane minions)
- `OnHitUnit`
- `OnDealDamage`
- `OnKill`
- `OnKillUnit`
- `OnLaunchAttack`
- `OnLaunchMissile`
- `OnMoveEnd`
- `OnMoveSuccess`
- `OnMoveFailure`
- `OnPreAttack` (except GarenQ buff)
- `OnSpellHit`
- `OnSpellCast`
- `OnSpellChannel`
- And 20+ others listed in ApiEventManager.cs comments

**These events are "DEAD" — they publish to zero listeners, so the cost of publishing them is wasted CPU but the game logic does not depend on them.**

---

## Question 3: Scripting Infrastructure — Script Resolution and Map Specificity

### Resolution Mechanism

Scripts resolve by **CHARACTER MODEL NAME** at unit spawn time, via the following mechanism (ObjAIBase.cs):

```csharp
CharScript = CSharpScriptEngine.CreateObjectStatic<ICharScript>("CharScripts", $"CharScript{Model}") ?? new CharScriptEmpty();
```

Example instantiations:
- Garen (model="Garen") → looks for `CharScripts.CharScriptGaren`
- SRUAP_Turret_Order1 (model="SRUAP_Turret_Order1") → looks for `CharScripts.CharScriptSRUAP_Turret_Order1`
- OrderTurretNormal (model="OrderTurretNormal") → looks for `CharScripts.CharScriptOrderTurretNormal`

### Proof that Map1 Turrets do NOT use SRUAP scripts

**Map1 turret models** (LevelScriptObjects.cs:77-89):
- **Blue**: OrderTurretShrine, OrderTurretAngel, OrderTurretDragon, OrderTurretNormal2, OrderTurretNormal
- **Red**: ChaosTurretShrine, ChaosTurretNormal, ChaosTurretGiant, ChaosTurretWorm2, ChaosTurretWorm

**SRUAP_Turret_* scripts location**: `Characters/SRUAP_Turret_Order4/`, `Characters/SRUAP_Turret_Chaos4/`, etc. — these only exist for Ascension (Map11).

**Verdict**: SRUAP_Turret_* scripts are **NEVER instantiated on Map1** because the model names do not match. The script resolution mechanism is sound and map-specific. This confirms that the revert of the prior "fix" was correct: that fix applied a 0.7 damage multiplier from SRUAP_Turret_* scripts to Map1 turrets, which must not happen because those scripts never instantiate on Map1 in the first place. Script isolation is verified.

### Additional Notes on Script Resolution

- Item scripts also resolve similarly by item ID: `CSharpScriptEngine.CreateObjectStatic<IItemScript>("ItemPassives", $"ItemID_{item.ItemId}")` (Inventory.cs:83)
- If a script does not exist, `?? new ItemScriptEmpty()` or `?? new CharScriptEmpty()` provides a safe fallback
- Buff scripts resolve by buff name: resolved at buff registration time

---

## Audit Table: Mechanics Modelled vs. Server

| Mechanic | Server (file:line) | Ours (file:line or "not modelled") | Verdict | Evidence |
|----------|---|---|---|---|
| Item: Doran's Shield HP bonus | 1054.json:41 (FlatHPPoolMod: 80) | not modelled | MISSING | Sim has 0 items |
| Item: Doran's Shield regen bonus | DoransShield.cs:13 (BaseBonus += 1.2) | not modelled | MISSING | Sim has 0 items |
| Event: OnUpdateStats (Garen passive) | GarenPassive.cs:40 | not modelled | MISSING | Sim does not track buff events |
| Event: OnTakeDamage (Garen passive cooldown) | GarenPassiveCooldown.cs | not modelled | MISSING | Sim does not track buff events |
| Event: OnPreTakeDamage (Garen W shield) | GarenW.cs | not modelled | MISSING | Sim does not model W buff |
| Event: OnPreAttack (Garen Q mechanics) | GarenQ.cs | not modelled | MISSING | Sim does not model Q buff |
| Event: OnCollision (terrain/minion) | ApiEventManager.cs:117-120 | not modelled | DEAD | No subscribers on Map1 |
| Event: OnCollisionTerrain | ApiEventManager.cs:119 | not modelled | DEAD | No subscribers on Map1 |
| Event: OnDeath (minion/champion) | ApiEventManager.cs:125-126 | not modelled | DEAD (for minions) | Lane minions have no OnDeath listeners |
| Event: OnHitUnit | ApiEventManager.cs:127-128 | not modelled | DEAD | No subscribers on Map1 |
| Event: OnDealDamage | ApiEventManager.cs:123-124 | not modelled | DEAD | No subscribers on Map1 |
| Event: OnKill | ApiEventManager.cs:131-132 | not modelled | DEAD | No subscribers on Map1 |
| Script: Garen character script | CharScriptGaren.cs:35-81 | lanerl_jax/sim/step.py (partial) | APPROX | Only passive apply modelled, not via script system |
| Script: SRUAP_Turret_* | Characters/SRUAP_Turret_*/BasicAttack.cs | N/A | N/A | Scripts never instantiate on Map1; different models |

---

## Prioritized Impact List

### CRITICAL ASYMMETRY (parity-blocking)
1. **Doran's Shield**: server grants +80 max HP and +1.2 HP/s. The **HP is already accounted for** in our port — `RUNE_HP_BONUS` is measured from the dump's in-play value, which includes it; our champion's max HP already equals the server's. The **regen was genuinely missing and is now fixed** (`44ddfa9`, champion `hp_regen` 1.568 -> 2.768). The earlier "10.6% deficit confounds every measurement" claim in this document was wrong and has been retracted above. Gate-3's death gap remains unexplained.

### HIGH (likely measured in test results)
2. **Garen passive mechanics**: OnUpdateStats event listener (for scaling regen) and OnTakeDamage listeners (for cooldown reset). Sim does not model event-driven mechanics.
3. **Garen W (Courage) buff**: OnPreTakeDamage listener for damage reduction. Sim does not model this.
4. **Garen Q (Decisive Strike) buff**: OnPreAttack listener for swing override. Sim does not model this.

### MEDIUM (unlikely to matter for lane parity)
5. Dead events (OnCollision, OnDeath, OnHitUnit, OnKill, etc.) that have zero subscribers on Map1. Correct to ignore.

### MAP-SPECIFICITY VERIFIED ✓
- Script resolution by character model name is sound
- SRUAP_Turret_* scripts do not and cannot run on Map1
- No script-loading bugs exist around map specificity
- Prior revert of the 0.7 damage multiplier was correct

---

## Test Status Baseline

Run before further changes:
```bash
JAX_PLATFORMS=cpu PYTHONPATH=$PWD ./.venv-jax/bin/python -m pytest lanerl_jax -q -rf
```

Expected: **1 failed, 269 passed** (test_last_hit_gate.py::test_oracle_scores_the_same_cs_in_sim_and_server is the expected failure)

Any other failure is an audit finding or regression.
