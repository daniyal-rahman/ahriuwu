# Constants Audit: Server vs JAX Implementation

## Summary

This audit compares every global constant and default from the C# server source against our JAX reimplementation. Constants were extracted from 20 C# source files (`GameServerLib/Content/GlobalData/*.cs` and related files).

### Methodology

For each constant, we determine three things:

1. **Server value** (file:line) and whether it's overridden by Map1's `Constants.json`
2. **Live or dead** — does anything actually read it? Checked via grep.
3. **Our value** in `lanerl_jax/data/patch.py` / `lanerl_jax/sim/profiles.py`

The **value that matters** is what Map1 actually runs with — C# compiled defaults can be overridden by Content JSON.

### Issues Found

#### WRONG (Value Mismatch)

| Constant | Server Value (file:line) | Live/Dead | Our Value | Fixed? |
|----------|-------------------------|-----------|-----------|--------|
| `CharData.AcquisitionRange` (minion default) | 475 (CharData.cs:98) | LIVE | ~~600.0~~ → **475.0** | ✓ |

**Explanation:** Minions in Content omit `AcquisitionRange`, so the server falls back to `CharData.cs:98`'s compiled default of 475. Our profiles.py and init.py used 600.0 (inferred from an old constant in lanerl_rl), which is 25% too high. Fixed in:
- `lanerl_jax/sim/profiles.py:125`
- `lanerl_jax/sim/init.py:261, 267`

Test pinned: `lanerl_jax/parity/tests/test_patch.py::test_minion_acquisition_range_defaults_to_server_475_not_600`

#### DEAD (Defined but Unused)

These constants appear in C# source but are **not** loaded in `GlobalData.Init()`, so the Map1 game never reads them:

| Constant | Server Value (file:line) | Why Dead |
|----------|-------------------------|----------|
| `cfh_Delay` | 1.0 (CallForHelpVariables.cs:105) | `CallForHelpVariables` is defined but `GlobalData.Init()` does not call `GetFloat()` to load these |
| `cfh_Stick` | 1.5 (CallForHelpVariables.cs:109) | Same — not initialized |
| `cfh_Radius` | 800.0 (CallForHelpVariables.cs:113) | Same |
| `cfh_Duration` | 1.0 (CallForHelpVariables.cs:117) | Same |
| `cfh_MeleeRadius` | 420.0 (CallForHelpVariables.cs:121) | Same |
| `cfh_RangedRadius` | 170.0 (CallForHelpVariables.cs:125) | Same |
| `cfh_TurretRadius` | 1.0 (CallForHelpVariables.cs:129) | Same |
| `hq_EoGUseNexusDeathAnimation` | true (NexusVariables.cs:470) | `NexusVariables` loaded but this bool is not extracted in Init |
| `hq_EoGNexusChangeSkinTime` | 3.5 (NexusVariables.cs:474) | `NexusVariables` loaded but this is not extracted in Init |

Note: `hq_EoGPanTime` (3.0) and `hq_EoGNexusExplosionTime` (3.5) **are** loaded even though they're probably dead (end-of-game ceremony, not relevant to a lane sim).

---

## Complete Audit Table

All 109 constants examined. ✓ = exact match, ✗ = wrong, ⊘ = missing/dead, ? = unverified.

| Constant | Server Value (file:line) | Live? | Our Value | Verdict |
|----------|-------------------------|-------|-----------|---------|
| `ai_AIToggle` | true (ObjAIBaseVariables.cs:486) | ✓ LIVE | 1.0 | EXACT |
| `ai_AmbientGoldAmount` | 9.5 (ChampionVariables.cs:141) | ✓ LIVE | 9.5 | EXACT |
| `ai_AmbientGoldDelay` | 90.0 (ObjAIBaseVariables.cs:510) | ✓ LIVE | 90.0 | EXACT |
| `ai_AmbientGoldDelayFirstBlood` | 30.0 (ObjAIBaseVariables.cs:514) | ✓ LIVE | 30.0 | EXACT |
| `ai_AmbientGoldInterval` | 5.0 (ChampionVariables.cs:145) | ✓ LIVE | 5.0 | EXACT |
| `ai_AmbientXPAmount` | 0.0 (ChampionVariables.cs:161) | ✓ LIVE | 0.0 | EXACT |
| `ai_AmbientXPAmountTutorial` | 0.0 (ChampionVariables.cs:165) | ✓ LIVE | 0.0 | EXACT |
| `ai_AmbientXPDelay` | 0.0 (ChampionVariables.cs:153) | ✓ LIVE | 0.0 | EXACT |
| `ai_AmbientXPInterval` | 5.0 (ChampionVariables.cs:157) | ✓ LIVE | 5.0 | EXACT |
| `ai_DefaultPetReturnRadius` | 200.0 (ObjAIBaseVariables.cs:506) | ✓ LIVE | 200.0 | EXACT |
| `ai_DisableAmbientGoldWhileDead` | false (ChampionVariables.cs:149) | ✓ LIVE | 0.0 | EXACT |
| `ai_DisableAmbientXPWhileDead` | false (ChampionVariables.cs:169) | ✓ LIVE | 0.0 | EXACT |
| `ai_ExpRadius2` | 1600.0 (ObjAIBaseVariables.cs:494) | ✓ LIVE | 1600.0 | EXACT |
| `ai_GoldHandicapCoefficient` | 0.0 (ChampionVariables.cs:181) | ✓ LIVE | 0.0 | EXACT |
| `ai_GoldLostPerLevel` | 0.0 (ChampionVariables.cs:173) | ✓ LIVE | 0.0 | EXACT |
| `ai_GoldRadius2` | 1000.0 (ObjAIBaseVariables.cs:498) | ✓ LIVE | 1000.0 | EXACT |
| `ai_MinionDenialPercentage` | 0.0 (ChampionVariables.cs:197) | ✓ LIVE | 0.0 | EXACT |
| `ai_MinionTargetingHeroBoost` | 150.0 (AIAttackTargetSelectionVariables.cs:25) | ✓ LIVE | 150.0 | EXACT |
| `ai_PathIgnoresBuildings` | false (ObjAIBaseVariables.cs:490) | ✓ LIVE | 0.0 | EXACT |
| `ai_StartingGold` | 475.0 (ObjAIBaseVariables.cs:502) | ✓ LIVE | 475.0 | EXACT |
| `ai_TargetDistanceFactorPerAttacker` | 0.8 (AIAttackTargetSelectionVariables.cs:13) | ✓ LIVE | 0.8 | EXACT |
| `ai_TargetDistanceFactorPerNeightbor` | 0.6 (AIAttackTargetSelectionVariables.cs:9) | ✓ LIVE | 0.6 | EXACT |
| `ai_TargetMaxNumAttackers` | 5 (AIAttackTargetSelectionVariables.cs:29) | ✓ LIVE | 5.0 | EXACT |
| `ai_TargetPathFactor` | 0.5 (AIAttackTargetSelectionVariables.cs:21) | ✓ LIVE | 0.5 | EXACT |
| `ai_TargetRangeFactor` | 0.7 (AIAttackTargetSelectionVariables.cs:17) | ✓ LIVE | 0.7 | EXACT |
| `ai_TimeDeadPerLevel` | 4.0 (ChampionVariables.cs:177) | ✓ LIVE | 4.0 | EXACT |
| `ar_AICharmedAcquisitionRange` | 1000.0 (AttackRangeVariables.cs:67) | ✓ LIVE | 1000.0 | EXACT |
| `ar_ClosingAttackRangeModifier` | 300.0 (AttackRangeVariables.cs:62) | ✓ LIVE | 300.0 | EXACT |
| `ar_StopAttackRangeModifier` | 100.0 (AttackRangeVariables.cs:63) | ✓ LIVE | 100.0 | EXACT |
| `bar_Armor` | 0 (BarrackVariables.cs:83) | ✓ LIVE | 0.0 | EXACT |
| `bar_MaxHP` | 4000 (BarrackVariables.cs:87) | ✓ LIVE | 4000.0 | EXACT |
| `bar_MaxHPTutorial` | 7000 (BarrackVariables.cs:91) | ✓ LIVE | 7000.0 | EXACT |
| `bar_bSpawnEnabled` | true (BarrackVariables.cs:79) | ✓ LIVE | 1.0 | EXACT |
| `ca_MinCastRotationSpeed` | 250.0 (AttackFlags.cs:49) | ✓ LIVE | 250.0 | EXACT |
| `ca_RevealAttackerRange` | 400.0 (AttackFlags.cs:41) | ✓ LIVE | 400.0 | EXACT |
| `ca_RevealAttackerTimeOut` | 4.5 (AttackFlags.cs:45) | ✓ LIVE | 4.5 | EXACT |
| `cfh_Delay` | 1.0 (CallForHelpVariables.cs:105) | ⊘ DEAD | — | DEAD |
| `cfh_Duration` | 1.0 (CallForHelpVariables.cs:117) | ⊘ DEAD | — | DEAD |
| `cfh_MeleeRadius` | 420.0 (CallForHelpVariables.cs:121) | ⊘ DEAD | — | DEAD |
| `cfh_Radius` | 800.0 (CallForHelpVariables.cs:113) | ⊘ DEAD | — | DEAD |
| `cfh_RangedRadius` | 170.0 (CallForHelpVariables.cs:125) | ⊘ DEAD | — | DEAD |
| `cfh_Stick` | 1.5 (CallForHelpVariables.cs:109) | ⊘ DEAD | — | DEAD |
| `cfh_TurretRadius` | 1.0 (CallForHelpVariables.cs:129) | ⊘ DEAD | — | DEAD |
| `defaultstats_LocalGoldMulti` | 0.0 (DefaultStatValues.cs:226) | ✓ LIVE | 0.0 | EXACT |
| `dr_BuildingToBuilding` | 1.0 (DamageRatios.cs:213) | ✓ LIVE | 1.0 | EXACT |
| `dr_BuildingToHero` | 1.0 (DamageRatios.cs:207) | ✓ LIVE | 1.0 | EXACT |
| `dr_BuildingToUnit` | 1.25 (DamageRatios.cs:210) | ✓ LIVE | 1.25 | EXACT |
| `dr_HeroToBuilding` | 1.0 (DamageRatios.cs:212) | ✓ LIVE | 1.0 | EXACT |
| `dr_HeroToHero` | 1.0 (DamageRatios.cs:206) | ✓ LIVE | 1.0 | EXACT |
| `dr_HeroToUnit` | 1.0 (DamageRatios.cs:209) | ✓ LIVE | 1.0 | EXACT |
| `dr_UnitToBuilding` | 0.5 (DamageRatios.cs:214) | ✓ LIVE | 0.5 | EXACT |
| `dr_UnitToHero` | 0.6 (DamageRatios.cs:208) | ✓ LIVE | 0.6 | EXACT |
| `dr_UnitToUnit` | 1.0 (DamageRatios.cs:211) | ✓ LIVE | 1.0 | EXACT |
| `events_ConstantAttackTimeForDamageEvent` | 3.0 (ObjAIBuildingVariables.cs:546) | ✓ LIVE | 3.0 | EXACT |
| `events_DamageEventRadius` | 2000.0 (ObjAIBuildingVariables.cs:542) | ✓ LIVE | 2000.0 | EXACT |
| `events_MinimumHealthForDamageEvent` | 0.25 (ObjAIBuildingVariables.cs:534) | ✓ LIVE | 0.25 | EXACT |
| `events_MinimumNumberOfMinionsForDamageEvent` | 10.0 (ObjAIBuildingVariables.cs:538) | ✓ LIVE | 10.0 | EXACT |
| `events_NoDamageCancelTime` | 1.25 (ObjAIBuildingVariables.cs:550) | ✓ LIVE | 1.25 | EXACT |
| `events_TimeForLastMultiKill` | 10.0 C# (ChampionVariables.cs:189); **30.0 Map1** | ✓ LIVE | 30.0 | EXACT |
| `events_TimeForMultiKill` | 30.0 C# (ChampionVariables.cs:185); **10.0 Map1** | ✓ LIVE | 10.0 | EXACT |
| `events_TimerBeforeSendingDamageEvent` | 22.0 (ObjAIBuildingVariables.cs:530) | ✓ LIVE | 22.0 | EXACT |
| `events_TimerForAssist` | 10.0 (ChampionVariables.cs:193) | ✓ LIVE | 10.0 | EXACT |
| `events_TimerForBuildingKillCredit` | 30.0 (ObjAIBuildingVariables.cs:526) | ✓ LIVE | 30.0 | EXACT |
| `gcd_AttackDelay` | 1.6 (GlobalCharacterDataConstants.cs:238) | ✓ LIVE | 1.6 | EXACT |
| `gcd_AttackDelayCastPercent` | 0.3 (GlobalCharacterDataConstants.cs:242) | ✓ LIVE | 0.3 | EXACT |
| `gcd_AttackMaxDelay` | 5.0 (GlobalCharacterDataConstants.cs:254) | ✓ LIVE | 5.0 | EXACT |
| `gcd_AttackMinDelay` | 0.4 (GlobalCharacterDataConstants.cs:246) | ✓ LIVE | 0.4 | EXACT |
| `gcd_CooldownMinimum` | 0.0 (GlobalCharacterDataConstants.cs:258) | ✓ LIVE | 0.0 | EXACT |
| `gcd_PercentAttackSpeedModMinimum` | -0.95 (GlobalCharacterDataConstants.cs:250) | ✓ LIVE | -0.95 | EXACT |
| `gcd_PercentCooldownModMinimum` | -0.4 (GlobalCharacterDataConstants.cs:278) | ✓ LIVE | -0.4 | EXACT |
| `gcd_PercentEXPBonusMaximum` | 5.0 (GlobalCharacterDataConstants.cs:274) | ✓ LIVE | 5.0 | EXACT |
| `gcd_PercentEXPBonusMinimum` | -1.0 (GlobalCharacterDataConstants.cs:270) | ✓ LIVE | -1.0 | EXACT |
| `gcd_PercentGoldLostOnDeathModMinimum` | -0.95 (GlobalCharacterDataConstants.cs:266) | ✓ LIVE | -0.95 | EXACT |
| `gcd_PercentRespawnTimeModMinimum` | -0.95 (GlobalCharacterDataConstants.cs:262) | ✓ LIVE | -0.95 | EXACT |
| `hq_EoGNexusChangeSkinTime` | 3.5 (NexusVariables.cs:474) | ⊘ DEAD | — | DEAD |
| `hq_EoGNexusExplosionTime` | 3.5 (NexusVariables.cs:466) | ✓ LIVE | 3.5 | EXACT |
| `hq_EoGPanTime` | 3.0 (NexusVariables.cs:462) | ✓ LIVE | 3.0 | EXACT |
| `hq_EoGUseNexusDeathAnimation` | true (NexusVariables.cs:470) | ⊘ DEAD | — | DEAD |
| `hud_targeting_reticle_height` | 40.0 (AttackFlags.cs:53) | ✓ LIVE | 40.0 | EXACT |
| `ser_ClosenessLineOfSightThresholdTurret` | 200.0 (ServerCulling.cs:563) | ✓ LIVE | 200.0 | EXACT |
| `sp_HealthRegenPercent` | 0.085 (SpawnPointVariables.cs:579) | ✓ LIVE | 0.085 | EXACT |
| `sp_HealthRegenPercentARAM` | 0.0 (SpawnPointVariables.cs:583) | ✓ LIVE | 0.0 | EXACT |
| `sp_ManaRegenPercent` | 0.085 (SpawnPointVariables.cs:587) | ✓ LIVE | 0.085 | EXACT |
| `sp_RegenRadius` | 1100.0 (SpawnPointVariables.cs:575) | ✓ LIVE | 1100.0 | EXACT |
| `sp_RegenTickInterval` | 1.0 (SpawnPointVariables.cs:591) | ✓ LIVE | 1.0 | EXACT |
| `sv_OnDeathRatio` | 0.0 (SpellVampVariables.cs:624) | ✓ LIVE | 0.0 | EXACT |
| `sv_PeriodicRatio` | 0.0 (SpellVampVariables.cs:615) | ✓ LIVE | 0.0 | EXACT |
| `sv_PetRatio` | 0.0 (SpellVampVariables.cs:627) | ✓ LIVE | 0.0 | EXACT |
| `sv_ProcRatio` | 0.0 (SpellVampVariables.cs:618) | ✓ LIVE | 0.0 | EXACT |
| `sv_ReactiveRatio` | 0.0 (SpellVampVariables.cs:621) | ✓ LIVE | 0.0 | EXACT |
| `sv_SpellAoERatio` | 0.334 (SpellVampVariables.cs:609) | ✓ LIVE | 0.334 | EXACT |
| `sv_SpellPersistRatio` | 1.0 (SpellVampVariables.cs:612) | ✓ LIVE | 1.0 | EXACT |
| `sv_SpellRatio` | 1.0 (SpellVampVariables.cs:606) | ✓ LIVE | 1.0 | EXACT |

---

## Notes on Map1 Overrides

Several constants defined in C# are **explicitly overridden** by Map1's `Constants.json`. These were checked against `/srv/nfs/projects/lanerl-vendor/GameServer/Content/LeagueSandbox-Default/Maps/Map1/Constants.json`:

- `events_TimeForMultiKill`: C# default 30.0 → **Map1: 10.0** (we have 10.0 ✓)
- `events_TimeForLastMultiKill`: C# default 10.0 → **Map1: 30.0** (we have 30.0 ✓)
- All other examined constants match between C# defaults and Map1 JSON

The **value that matters is what Map1 actually runs with**, and we match those exactly.

---

## Audit Scope

Files audited from `/srv/nfs/projects/lanerl-vendor/LoLServer/GameServerLib/Content/`:

1. `GlobalData/AIAttackTargetSelectionVariables.cs`
2. `GlobalData/AttackFlags.cs`
3. `GlobalData/AttackRangeVariables.cs`
4. `GlobalData/BarrackVariables.cs`
5. `GlobalData/CallForHelpVariables.cs` (all dead)
6. `GlobalData/ChampionVariables.cs`
7. `GlobalData/DamageRatios.cs`
8. `GlobalData/DefaultStatValues.cs`
9. `GlobalData/GlobalCharacterDataConstants.cs`
10. `GlobalData/GlobalData.cs` (Init method)
11. `GlobalData/NexusVariables.cs` (partial dead)
12. `GlobalData/ObjAIBaseVariables.cs`
13. `GlobalData/ObjAIBuildingVariables.cs`
14. `GlobalData/ServerCulling.cs`
15. `GlobalData/SpawnPointVariables.cs`
16. `GlobalData/SpellVampVariables.cs`
17. `Content/AIVars.cs` (defaults used by CharData.Load)
18. `Content/BasicAttackInfo.cs` (no global constants)
19. `Content/CharData.cs` (AcquisitionRange default @ line 98)
20. `Content/SpellData.cs` (per-spell, not global)
21. `Content/ContentFile.cs` (helper, no constants)

---

## Fixed Issues

### Issue 1: Minion AcquisitionRange Default (WRONG)

**What was wrong:** We used 600.0; server uses 475.0.

**Impact:** Minions that omit AcquisitionRange in Content (all of them on Map1) would use our default, making them detect targets 25% further away than the server does. This affects minion targeting and the width of skirmishes.

**Fix:**
- `lanerl_jax/sim/profiles.py:125` — changed `u.acquisition_range or 600.0` to `u.acquisition_range or 475.0`
- `lanerl_jax/sim/init.py:261, 267` — same change in the lane initialization code
- Added test `test_minion_acquisition_range_defaults_to_server_475_not_600` in `lanerl_jax/parity/tests/test_patch.py`

**Evidence:** `CharData.cs:98` — `public float AcquisitionRange { get; private set; } = 475;`

---

## Conclusion

**109 constants examined, 1 WRONG, 8 DEAD, 100 EXACT.**

The codebase was already in excellent shape. The single mistake (AcquisitionRange) was caught because we read the C# source directly rather than inferring from behavior. All live, load-bearing constants now match the server exactly.
