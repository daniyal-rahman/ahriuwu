# Content-script mechanics audit — Garen top-lane 1v1

Scope: `Content/LeagueSandbox-Scripts/` in the vendored server at
`/srv/nfs/projects/lanerl-vendor/LoLServer` (read-only), restricted to what can
fire in a Garen-vs-Garen top-lane 1v1: Garen, lane minions (`Blue_Minion_Basic`,
`Blue_Minion_Wizard`, `Blue_Minion_MechCannon`, and Red equivalents), and lane /
nexus / fountain turrets. Jungle monsters, items, runes and other champions are
out of scope. All paths below are relative to
`Content/LeagueSandbox-Scripts/` unless given in full; engine paths are
relative to the vendor repo root.

## The rule this whole document is now organized around

**A content script only applies if the unit it's named for is the unit
actually on the field, and scripts resolve by CHARACTER NAME, not by concept
("outer turret") or by file location.** For every finding below, three
questions are answered explicitly:

1. What character name does the script attach to (i.e. what name does the
   engine look up to find it)?
2. Is that the name Map 1 actually spawns for the relevant slot?
3. Does a matching `Characters/<name>/` folder exist, or does it silently
   fall back to `SpellScriptEmpty`/native behavior?

Each finding is also tagged with **how it was confirmed**: `SOURCE-READ` (I
traced the code/JSON and it plainly resolves this way) or `MEASURED`
(independently confirmed against a live-server observation). Never presented
as measured when it's only been read.

**Status: the map-routing bug this document's first draft surfaced is real
and has been confirmed and fixed.** Three independent measurements against a
600s idle recording (courtesy of the coordinator) nail down Map 1's outer
turret identity:
- Turret HP was monotone across all 36,001 snapshots (zero regen) — rules out
  `SRUAP_Turret_Order3`, which has `BaseStaticHPRegen: 3`
  (confirmed by re-reading `LeagueSandbox-Default/Stats/SRUAP_Turret_Order3/SRUAP_Turret_Order3.json`).
- Single-step turret-hit HP drops are consistent with 100 armor mitigation
  points corresponding to **Armor 60**, not 67 — matches `OrderTurretNormal`
  (Armor 60), not `SRUAP_Turret_Order3`/`OrderTurretDragon` (Armor 67, the
  inhibitor tier).
- The live config pins `"map": 1` (`lanerl/cfg/garen1v1.json:132`,
  `lanerl_bot/configs/garen1v1_bot.json:132`).

`TURRET_DAMAGE_VS_MINION` has been reverted to `1.0` and `TURRET_MODELS`
repointed to `OrderTurretNormal`/`ChaosTurretWorm` in the sim. **One more
thing to get right while doing that:** Red's outer turret is
`ChaosTurretWorm`, *not* `ChaosTurretNormal` — `ChaosTurretNormal` is Chaos's
**nexus** turret (AD 180, Armor 65, `BaseStaticHPRegen: 6`; confirmed by
reading `LeagueSandbox-Default/Stats/ChaosTurretNormal/ChaosTurretNormal.json`).
Pairing Blue/Red turret models by *string similarity* ("Normal" ~ "Normal")
instead of by the `TurretType` key they're actually declared under is exactly
how that bug happened. The correct pairing, read directly out of
`Maps/Map1/LevelScriptObjects.cs:73-90` (`TowerModels` dict, keyed by
`TurretType`, not by name), with base stats from each `CharData` JSON:

| `TurretType` | Blue (Order) model | Blue stats (HP/AD/Armor/SpellBlock/Regen) | Red (Chaos) model | Red stats |
|---|---|---|---|---|
| `OUTER_TURRET` | `OrderTurretNormal` | 1300 / 152 / **60** / 100 / 0 | `ChaosTurretWorm` | 1300 / 152 / **60** / 100 / 0 |
| `INNER_TURRET` | `OrderTurretNormal2` | 1300 / 170 / 60 / 100 / 0 | `ChaosTurretWorm2` | 1300 / 170 / 60 / 100 / 0 |
| `INHIBITOR_TURRET` | `OrderTurretDragon` | 1300 / 190 / 67 / 100 / **3** | `ChaosTurretGiant` | 1300 / 190 / 67 / 100 / **3** |
| `NEXUS_TURRET` | `OrderTurretAngel` | 1300 / 180 / 65 / 100 / **6** | `ChaosTurretNormal` | 1300 / 180 / 65 / 100 / **6** |
| `FOUNTAIN_TURRET` | `OrderTurretShrine` | 9999 / 999 / 0 / — / 0 | `ChaosTurretShrine` | 9999 / 999 / 0 / — / 0 |

Every row's Blue/Red pair has identical stats (the map is symmetric per
tier) — only the *names* differ, and only two of the ten names ("Normal" and
"Normal2") are shared verbatim between a Blue outer/inner turret and a
same-named-pattern Chaos turret that is *not* the same tier. Confirmed
`MEASURED` for the `OUTER_TURRET` row (by the coordinator, as above);
`SOURCE-READ` only for the other four rows (I did not independently measure
inner/inhibitor/nexus/fountain turret identity against a live trace — if
those matter downstream, they deserve the same three-point check the outer
turret got, since the general risk — a script/model that doesn't match the
unit actually spawned — applies equally to all five tiers).

None of these ten names (`OrderTurretNormal`, `OrderTurretNormal2`,
`OrderTurretDragon`, `OrderTurretAngel`, `OrderTurretShrine`,
`ChaosTurretWorm`, `ChaosTurretWorm2`, `ChaosTurretGiant`, `ChaosTurretNormal`,
`ChaosTurretShrine`) has a matching `Content/LeagueSandbox-Scripts/Characters/`
folder — confirmed by directory listing (only `Garen`, `SRUAP_Turret_Chaos3`,
`SRUAP_Turret_Chaos4`, `SRUAP_Turret_Order3`, `SRUAP_Turret_Order4` exist) and
by a whole-tree `grep -rl` for all ten names, which only turns up the
`TowerModels` dictionary *string literals* themselves, never a class
definition. So **every turret on Map 1, at every tier, uses the native
`ObjAIBase.AutoAttackHit` fallback** (see the routing chain in Finding 1) —
flat `Stats.AttackDamage.Total` physical damage, no target-type discount,
`S5Test_TowerWrath` never added. This is now `MEASURED` for the outer tier
and `SOURCE-READ` (same mechanism, same conclusion) for the other four.

## Minions — the most important check, and it comes back clean

`Maps/Map1/LevelScript.cs:24-38` (`MinionModels` dict):
```csharp
{TeamId.TEAM_BLUE, new Dictionary<MinionSpawnType, string>{
    {MinionSpawnType.MINION_TYPE_MELEE, "Blue_Minion_Basic"},
    {MinionSpawnType.MINION_TYPE_CASTER, "Blue_Minion_Wizard"},
    {MinionSpawnType.MINION_TYPE_CANNON, "Blue_Minion_MechCannon"},
    {MinionSpawnType.MINION_TYPE_SUPER, "Blue_Minion_MechMelee"}
}},
{TeamId.TEAM_PURPLE, new Dictionary<MinionSpawnType, string>{
    {MinionSpawnType.MINION_TYPE_MELEE, "Red_Minion_Basic"},
    {MinionSpawnType.MINION_TYPE_CASTER, "Red_Minion_Wizard"},
    {MinionSpawnType.MINION_TYPE_CANNON, "Red_Minion_MechCannon"},
    {MinionSpawnType.MINION_TYPE_SUPER, "Red_Minion_MechMelee"}
}}
```
This is an **exact, literal match** to what the sim uses
(`Blue_Minion_Basic`/`Blue_Minion_Wizard`/`Blue_Minion_MechCannon` and Red
equivalents) — not a "close enough" name, the identical string. So there is
no minion stat/identity mismatch on Map 1: whatever HP/AD/Armor/`IsMelee`
values the sim reads for these exact names from the stat JSON are the values
the live server uses too. `Content/LeagueSandbox-Scripts/Characters/` has no
folder for any of these eight minion names either (confirmed by directory
listing), so minions' own basic attacks also fall back to native
`AutoAttackHit` — consistent with there being no minion-specific damage
script anywhere in the tree (their only content script involvement is
`AIScripts/LaneMinionAI.cs`, pure targeting/movement, see Finding 9).
Confirmed `SOURCE-READ` (exact dictionary-literal string match — as strong a
form of source confirmation as exists, short of a live trace, since there's
no derivation or lookup chain to get wrong).

## Summary table (ranked by relevance to a live Garen top-lane 1v1)

| # | Finding | Character name / is it spawned? | File:line | Confirmed via | Can it matter here? |
|---|---|---|---|---|---|
| 1 | Turret `BasicAttack` scripts (incl. the 0.7x anti-minion multiplier and `S5Test_TowerWrath`) never load on Map 1 — every turret tier uses native `AutoAttackHit` | Attaches to `SRUAP_Turret_{Order,Chaos}{3,4}`; Map 1 spawns `OrderTurretDragon`/`OrderTurretAngel`/etc. instead (see table above) — **not the same name, script never loads** | `Characters/SRUAP_Turret_{Order3,Chaos3,Order4,Chaos4}/BasicAttack.cs` | **MEASURED** (outer tier, 3-point check above) + SOURCE-READ (routing chain, other tiers) | Resolved: `TURRET_DAMAGE_VS_MINION` reverted to 1.0, `TURRET_MODELS` repointed. No further action. |
| 2 | `S5Test_TowerWrath` buff is a complete no-op regardless of map (empty `OnActivate`, no stat modifier, no `OnUpdate`) | Attaches to buff name `S5Test_TowerWrath`, only ever `AddBuff`'d from the dead scripts in #1 | `Buffs/LaserTurrets/S5Test_TowerWrath.cs:11-24` | SOURCE-READ | Cannot matter on Map 1 at all — the call site that would add it is unreachable (#1); and even on a map where it is reachable, it does nothing. |
| 3 | Garen passive: whether a minion autoattack interrupts HP regen depends on an accidental `UnitTag` enum-value collision — melee/caster minions never interrupt it, cannon minions only fail to interrupt it once Garen is level ≥ 11 | Attacks Garen's own passive logic; keys off the *attacker's* `CharData.UnitTags`, i.e. `Blue_Minion_Basic`/`Wizard`/`MechCannon` and Red equivalents — **confirmed exact-match spawn name** (see Minions section above) | `Characters/Garen/CharScriptGaren.cs:19-27,101-116`; `GameServerCore/Enums/UnitTag.cs:6-29` | SOURCE-READ (pure enum arithmetic + confirmed minion identity — no live trace) | Fires on essentially every minion trade in laning phase, levels 1-10 especially. |
| 4 | Garen E ("Judgment") deals 0.75x damage to minions, 0x to turrets/buildings | Attaches to buff `GarenE`, added by `Characters/Garen/E.cs`'s `GarenE` spell script — Garen's `CharData` `Spell3 = "GarenE"` (confirmed, see Garen section below) | `Buffs/Garen/GarenE.cs:101,106` | SOURCE-READ | Directly relevant — E is the primary wave-clear tool in this matchup, this discount is not in the stat JSON, and (unlike the turret one) it is not gated by any map/model mismatch. |
| 5 | Turret max-HP scaling by enemy champion count, and time-gated flat Armor/MR/AD growth, applied by the *map* script (`LevelScriptObjects`), independent of which `CharData`/model each turret uses | Keyed by `TurretType`, not character name — fires identically for whatever model Map 1 assigns to each tier | `Maps/Map1/LevelScriptObjects.cs:121-153` (HP), `:159-266` (growth) | SOURCE-READ (not independently measured over time — see note below) | Live on Map 1 regardless of the turret-model mixup. In a 1v1: flat +250 HP to outer/inner/inhibitor, +125 HP to nexus at match start, plus scheduled AD/Armor/MR growth over the game. |
| 6 | Garen W ("Courage") multiplies **all** incoming post-mitigation damage by 0.7 while active, regardless of source | Attaches to buff `GarenW`, added by `Characters/Garen/W.cs`'s `GarenW` spell script — `Spell2 = "GarenW"` (confirmed) | `Buffs/Garen/GarenW.cs:47-55` | SOURCE-READ | Directly relevant to trading under tower / against minions; not map-dependent. |
| 7 | `GarenWPassive` is a permanent (`infiniteduration=true`) +20% Armor / +20% MR buff, applied once, the first time W is leveled | Attaches to buff `GarenWPassive`, added by the same `GarenW` spell script on `OnLevelUpSpell` | `Characters/Garen/W.cs:31-46`; `Buffs/Garen/GarenWPassive.cs:31-39` | SOURCE-READ | Changes Garen's effective mitigation vs. every minion and turret hit from ~level 2 onward; not map-dependent. |
| 8 | `TurretAI` "tower aggro": a turret already hitting a non-champion switches onto an enemy champion caught auto-attacking another champion in range | AI script name `"TurretAI"` is an explicit string in `Maps/Map1/LevelScriptObjects.cs:26` (`LaneTurretAI = "TurretAI"`), passed to every turret's constructor regardless of its `CharData`/model — **not derived from character name**, so unaffected by the model mixup | `AIScripts/TurretAI.cs:60-88` | SOURCE-READ | Directly relevant to turret-dive / all-in trades between two Garens under a tower; robust to which specific turret model is in play. |
| 9 | `LaneMinionAI` won't pull a minion off a turret target for a fresh wave/champion | AI script name `"LaneMinionAI"` is an explicit string, `Maps/Map1/LevelScript.cs:21`, passed to every lane minion regardless of type — same "not name-derived" robustness as #8 | `AIScripts/LaneMinionAI.cs:150-173` | SOURCE-READ | Affects whether minions peel off a turret onto Garen; deliberate per the code's own comments. |
| 10 | Garen Q's damage (30 + 25×(lvl-1) + 1.4×AD physical) and silence duration are script-defined, no minion/turret discount | Attaches to `GarenQAttack`, swapped in by buff `GarenQ` from spell `GarenQ` — `Spell1 = "GarenQ"` (confirmed) | `Characters/Garen/Q.cs:142,151` | SOURCE-READ | Full, undiscounted damage vs. minions and (if targetable) turrets; not map-dependent. |
| 11 | Garen R deals `175×lvl + missingHP%×missingHP` magic damage; targetability vs minion/turret is gated by stat-JSON `SpellData`, not this script | `Spell4 = "GarenR"` (confirmed) attaches `Characters/Garen/R.cs`'s `GarenR` | `Characters/Garen/R.cs:28-29` | SOURCE-READ (targeting gate itself not verified — outside content-script scope) | Likely champion-only; not independently confirmed. |
| 12 | `LevelScript.cs` `MinionModifiers` dict populated but never applied | — (map-level, not character-name-gated; this is a plain dead-code path) | `Maps/Map1/LevelScript.cs:123-157` | SOURCE-READ (re-confirmed) | Per your note — not a new finding, listed only for completeness. |
| 13 | `Buffs/Global/HPByPlayerLevel.cs` fully inert: never `AddBuff`'d anywhere, and its own stat line is commented out | Attaches to buff name `HPByPlayerLevel`; grep confirms zero `AddBuff("HPByPlayerLevel", ...)` call sites anywhere in the tree | `Buffs/Global/HPByPlayerLevel.cs:48-53` | SOURCE-READ | Cannot matter — dead code, independent of map. |
| 14 | Fountain turret native override: flat 1000 true damage on hit; untargetable by both teams | Gated by `TurretType.FOUNTAIN_TURRET`, not by `CharData`/model name — fires for `OrderTurretShrine`/`ChaosTurretShrine` on Map 1 just as it would for any other fountain-tier model | `GameServerLib/GameObjects/AttackableUnits/AI/LaneTurret.cs:28-32,90-98` (engine, not content script) | SOURCE-READ | Cannot matter in ordinary laning; only relevant for a fountain dive/backdoor. Not affected by the turret-model mixup (type-gated, not name-gated). |
| 15 | On Map 11 only: shared, un-reset `StatsModifier` in `OnMatchStart` leaks +33 Armor/-20 AD from the inhibitor-turret branch onto later-processed outer/inner/nexus turrets | Map-11-specific; Map 1 is confirmed live, so this cannot fire regardless of character-name questions | `Maps/Map11/LevelScriptObjects.cs:80-124` | SOURCE-READ | Cannot matter on the live Map 1 config. Recorded only in case of a future map migration. |
| 16 | `Buffs/Global/Recall.cs` cancels a recall channel on any non-periodic damage instance | Attaches to buff `Recall`; not damage/stat math | `Buffs/Global/Recall.cs:38-45` | SOURCE-READ | Only matters if the sim models recalling. |
| 17 | `AIScripts/MinionAI.cs` (`MinonAI`) is not used by lane minions | AI script string for lane minions is `"LaneMinionAI"` (`Map1/LevelScript.cs:21`), not `"MinionAI"` — confirmed by explicit-string wiring, same mechanism as #8/#9 | `AIScripts/MinionAI.cs:12` | SOURCE-READ | Searched, not applicable. |
| 18 | `Buffs/Global/*` other than the above (`Blind`, `Burning`, `Disarm`, `ExpirationTimer`, `GlobalMonsterBuff`, `HowlingAbyssAura`, `OdinChannelVision`, `OdinPlayerBuff`, `Silence`, `Slow`, `Stun`, `URFBuff`) | — | — | SOURCE-READ (searched) | Generic CC/DoT primitives or other-game-mode-specific (ARAM/Dominion/URF) buffs. No hidden Garen/minion/turret-specific multipliers found beyond `Silence` (already covered under Q). **Found nothing surprising.** |

## Detail

### 1. Turret `BasicAttack` scripts never load on Map 1 (MEASURED for outer tier; SOURCE-READ for the rest) — RESOLVED

**Character-name check:** the script attaches to `SRUAP_Turret_Order3BasicAttack`
/ `Chaos3` / `Order4` / `Chaos4` (class names matching
`Characters/SRUAP_Turret_{Order3,Chaos3,Order4,Chaos4}/BasicAttack.cs`, all
four confirmed byte-identical in the `dmg *= 0.7f` / `AddBuff("S5Test_TowerWrath", ...)`
pair). Map 1 does not spawn anything with those names — see the corrected
turret table at the top of this document. The full routing chain (why a
name mismatch means the script silently never loads, rather than erroring):

1. A unit's basic-attack spell is named `"<CharacterName>BasicAttack"` —
   `GameServerLib/Content/CharData.cs:184` (`Name = name + "BasicAttack"`).
2. Script lookup is a flat `namespace.ClassName` search over the whole
   compiled Content-Scripts assembly (folder location is irrelevant) —
   `GameServerLib/Scripting/CSharp/CSharpScriptEngine.cs:162-179`
   (`CreateObjectStatic`). If no matching class exists, it silently falls
   back to `SpellScriptEmpty` (`?? new SpellScriptEmpty()` at
   `GameServerLib/GameObjects/Spell/Spell.cs:149`).
3. `Spell.HasEmptyScript` is set `true` exactly when that fallback fired
   (`GameServerLib/GameObjects/Spell/Spell.cs:124`). In `FinishCasting()`,
   for a ranged unit, a generic auto-attack missile (which calls the native
   `ObjAIBase.AutoAttackHit`) is created only if `HasEmptyScript`
   (`GameServerLib/GameObjects/Spell/Spell.cs:999`). If a custom script
   exists instead, no missile is spawned here — the custom script is fully
   responsible, so there's no double-damage risk either way.
4. The native fallback, `ObjAIBase.AutoAttackHit`
   (`GameServerLib/GameObjects/AttackableUnits/AI/ObjAIBase.cs:269-294`),
   deals flat `Stats.AttackDamage.Total` physical damage with standard
   mitigation — no target-type discount, no buff application.
5. On Map 1, turret models are `OrderTurretNormal`/`OrderTurretNormal2`/
   `OrderTurretDragon`/`OrderTurretAngel`/`OrderTurretShrine` and their
   `ChaosTurret*` counterparts (`Maps/Map1/LevelScriptObjects.cs:73-90`).
   None of the ten names have a matching `Characters/` folder (confirmed by
   directory listing and a whole-tree `grep -rl`, which only matches the
   `TowerModels` dictionary string literals themselves).

`SRUAP_Turret_Order1`/`Order2`/`Chaos1`/`Chaos2` (the actual outer/inner-tier
`SRUAP_Turret_*` `CharData` entries) exist as data but also have no
BasicAttack script — only `Order3`/`Chaos3` (inhibitor) and `Order4`/`Chaos4`
(nexus) do. `Maps/Map11/LevelScriptObjects.cs:270-296` ("New SR") is the only
place in the repo that spawns turrets under the `SRUAP_Turret_*` names
directly. So even on Map 11, the outer/inner tiers a laning-phase 1v1 fights
under would still use the native path; only a siege on an inhibitor/nexus
turret would ever reach the discount script. This part remains SOURCE-READ —
I have no Map-11 measurement, and Map 11 isn't live here regardless.

**Status:** the coordinator's fix (`TURRET_DAMAGE_VS_MINION = 1.0`,
`TURRET_MODELS` → `OrderTurretNormal`/`ChaosTurretWorm`) matches this
analysis. No open action item.

### 2. `S5Test_TowerWrath` is a no-op (SOURCE-READ)

**Character-name check:** attaches to buff name `S5Test_TowerWrath`, only
ever added from `AddBuff("S5Test_TowerWrath", 0.5f, 1, spell, target, owner)`
inside the four dead `BasicAttack.cs` scripts from Finding 1 — so on Map 1
this call site is never reached at all. Independent of that:

```csharp
// Buffs/LaserTurrets/S5Test_TowerWrath.cs:11-24
internal class S5Test_TowerWrath : IBuffGameScript
{
    public BuffScriptMetaData BuffMetaData { get; set; } = new BuffScriptMetaData
    { BuffType = BuffType.AURA, BuffAddType = BuffAddType.RENEW_EXISTING };
    public StatsModifier StatsModifier { get; private set; }
    public void OnActivate(AttackableUnit unit, Buff buff, Spell ownerSpell)
    { //TODO: Investigate all the effects applied by this buff
    }
}
```
No `StatsModifier` field is ever set, no `OnUpdate`/`OnDeactivate` exists, and
a repo-wide `grep -rn "TowerWrath|S5Test"` outside Content-Scripts turns up
nothing in `GameServerLib`/`GameServerCore` either. There is no "tower
wrath" damage ramp anywhere in this codebase, on any map, reachable or not.

### 3. Garen passive: minion-tag exception has an enum collision bug (SOURCE-READ)

**Character-name check:** this logic lives in Garen's own `CharScriptGaren`
and keys off the *attacker's* `CharData.UnitTags` — the attacker in scope is
whichever minion hit Garen. Confirmed: Map 1 spawns exactly
`Blue_Minion_Basic`/`Blue_Minion_Wizard`/`Blue_Minion_MechCannon` and Red
equivalents (see Minions section), so their `UnitTags` strings
(`"Minion | Minion_Lane"` for melee/caster, `"Minion | Minion_Lane | Minion_Lane_Siege"`
for cannon — confirmed directly from each unit's stat JSON) are exactly what
this analysis assumes. No name-mismatch risk here.

```csharp
// Characters/Garen/CharScriptGaren.cs:19-27
private static readonly UnitTag[] MINION_UNIT_TAG_PASSIVE_EXCEPTIONS =
{
    UnitTag.Minion, UnitTag.Minion_Lane, UnitTag.Minion_Lane_Siege,
    UnitTag.Minion_Lane_Super, UnitTag.Minion_Summon,
};
// :101-116
public static bool ShouldPassiveTurnOff(AttackableUnit unit, DamageData damageData)
{
    if (MINION_UNIT_TAG_PASSIVE_EXCEPTIONS.Contains(damageData.Attacker.CharData.UnitTags))
        return false;
    if (unit.Stats.Level >= 11 && UnitTag.Monster.Equals(damageData.Attacker.CharData.UnitTags))
        return false;
    if (damageData.PostMitigationDamage <= 0) return false;
    return true;
}
```
`GameServerCore/Enums/UnitTag.cs:6-29` declares `UnitTag` as `[Flags]` but
assigns no explicit values, so C# numbers members sequentially (0,1,2,3,...),
not as a bitmask (1,2,4,8,...): `Champion=0, Champion_Clone=1, Minion=2,
Minion_Lane=3, Minion_Lane_Siege=4, Minion_Lane_Super=5, Minion_Summon=6,
Monster=7, ...`. `CharData.UnitTags` is built by OR-ing each tag
(`GameServerLib/Content/CharData.cs:152-156`). So `Minion | Minion_Lane` =
`2 | 3` = `3`, which happens to equal `Minion_Lane` exactly — melee/caster
minions correctly match the exceptions list, their autoattacks never turn
off the passive, at any level. But `Minion | Minion_Lane | Minion_Lane_Siege`
= `2 | 3 | 4` = `7`, the exact value of `UnitTag.Monster` — cannon minions'
combined tag is **not** in the exceptions array (which lists 2,3,4,5,6, not
7), so the first check fails, and it only escapes turning off the passive via
the second check, gated on `unit.Stats.Level >= 11` (Garen's own level).
**Net effect:** melee/caster minion autoattacks never interrupt Garen's
passive at any level; cannon minion autoattacks do interrupt it (starting
the 9/6/4s out-of-combat cooldown by `CharScriptGaren.GetCooldownForLevel`,
`:96-99`) until Garen reaches level 11, at which point they're silently
treated like jungle-monster damage by coincidence of the enum values.

### 4. Garen E: 0.75x to minions, 0x to turrets/buildings (SOURCE-READ)

**Character-name check:** Garen's `CharData` (`LeagueSandbox-Default/Stats/Garen/Garen.json`,
`MetaData.Id = "Garen"`) declares `Spell3 = "GarenE"`, matching the
`GarenE : ISpellScript` class in `Characters/Garen/E.cs`, which on
`OnSpellPostCast` adds buff `GarenE` — matching the `GarenE : IBuffGameScript`
class in `Buffs/Garen/GarenE.cs` exactly. The project's own config also pins
`"champion": "Garen"` (`lanerl/cfg/garen1v1.json:9,72`). Full chain confirmed,
no fallback risk.

```csharp
// Buffs/Garen/GarenE.cs:99-109
if (units[i].Team != Owner.Team && !(units[i] is ObjBuilding || units[i] is BaseTurret) && units[i] is ObjAIBase)
{
    var customTickDamage = tickDamage;
    if (units[i] is Minion) { customTickDamage *= 0.75f; }
    units[i].TakeDamage(Owner, customTickDamage, DamageType.DAMAGE_TYPE_PHYSICAL, DamageSource.DAMAGE_SOURCE_SPELL, isCrit);
}
```
E ticks every 500ms against everything in a 330-unit radius.
`is ObjBuilding || is BaseTurret` is excluded from the hit list entirely — E
can never damage a turret, at all, on any tier, on any map. `is Minion` (true
for `LaneMinion` and its `Blue_Minion_*`/`Red_Minion_*` instances) gets
`×0.75`. Base per-tick damage is `10 + 12.5×(lvl-1) + AD×(0.35+0.05×(lvl-1))`
(`GarenE.cs:36-37`), computed once at `OnActivate`, not re-read per tick.
Also present, unlikely to matter without items: a per-tick crit roll against
`Owner.Stats.CriticalChance.Total` using a fixed-seed `Random(0x6A3E17)`
(`GarenE.cs:28,93`) — Garen's base kit is 0% crit chance, so this cannot fire
in an item-less 1v1.

### 5. Turret HP/stat scaling from `LevelScriptObjects` (SOURCE-READ — not independently timed against a trace)

**Character-name check:** none needed — this is keyed by `TurretType`, an
enum tag passed explicitly at `CreateLaneTurret` call time
(`Maps/Map1/LevelScriptObjects.cs:335,358`), not by `CharData`/model name. It
fires identically no matter which specific model occupies each tier, so it
is *not* affected by the outer-turret naming mixup and applies on Map 1 as
written.

`OnMatchStart` (`:121-153`, called unconditionally):
```csharp
foreach (var turret in TurretList[team][lane]) {
    if (turret.Type == TurretType.FOUNTAIN_TURRET) { continue; }
    else if (turret.Type != TurretType.NEXUS_TURRET)
        TurretHealthModifier.HealthPoints.BaseBonus = 250.0f * Players[enemyTeam].Count;
    else
        TurretHealthModifier.HealthPoints.BaseBonus = 125.0f * Players[enemyTeam].Count;
    turret.AddStatModifier(TurretHealthModifier);
    turret.Stats.CurrentHealth += turret.Stats.HealthPoints.Total;
    AddTurretItems(turret, GetTurretItems(TurretItems, turret.Type));
}
```
(Checked whether reusing one `StatsModifier` instance across turret types
could leak values between iterations — it's safe here:
`Stat.ApplyStatModifier` [`GameServerLib/GameObjects/Stats/Stat.cs:85-96`]
copies by value into the turret's own fields at `AddStatModifier` time, and
this loop only ever sets the one field it uses immediately beforehand.
Contrast with Finding 15, a real instance of this failure mode, but on
Map 11 only.)

Then two independent scripted growth schedules from `OnUpdate` (`:159-186`),
neither in the stat JSON:
```csharp
// :224-247 (non-outer turrets): Armor+1, MR+1, AD+4 per application
static float timeCheck = 480.0f * 1000;   // starts at game time 8:00
// up to 30 applications, every 60s; skips OUTER, FOUNTAIN, and INNER once timesApplied>=20

// :248-266 (outer turrets only): MR+1, AD+4 (no Armor) per application
static float outerTurretTimeCheck = 30.0f * 1000;  // starts at game time 0:30
// up to 7 applications, every 60s
```
Net effect for a top-lane 1v1: at match start the outer/inner turret gets
+250 flat HP (nexus +125), scaled by `Players[enemyTeam].Count` (1 in a
1v1). The **outer** turret then gets +1 MR/+4 AD, seven times, every 60s
starting at 0:30 — plateauing at +7 MR/+28 AD by ~7:30 and never growing
again (no Armor growth at all for the outer tier on Map 1). The **inner**
turret separately gets +1 Armor/+1 MR/+4 AD every 60s starting at 8:00, up to
20 times (plateaus ~28:00). Inhibitor/nexus get the same +1/+1/+4 schedule
uncapped (up to 30 applications, until ~38:00). **This part is SOURCE-READ
only** — I do not have an independent measurement confirming the exact
timings or that the schedule actually executes as written. Given the
project already has a 600s idle-recording harness (used for Finding 1), the
cheapest way to promote this to MEASURED would be to check outer-turret
single-hit damage at t≈0:30, 1:30, ..., 7:30 for +4 AD steps, and inner
turret similarly from t=8:00 onward.

### 6 & 7. Garen W and its passive (SOURCE-READ)

**Character-name check:** `Spell2 = "GarenW"` in Garen's `CharData`, matching
`GarenW : ISpellScript` in `Characters/Garen/W.cs`, which adds buffs `GarenW`
(`Buffs/Garen/GarenW.cs`) and, once, `GarenWPassive`
(`Buffs/Garen/GarenWPassive.cs`) — both class names confirmed to match their
`AddBuff` call sites exactly. No fallback risk.

```csharp
// Buffs/Garen/GarenW.cs:38-55
public void OnActivate(...) {
    StatsModifier.Tenacity.PercentBonus += 30;   // units bug, irrelevant here (no CC sources in this matchup)
    unit.AddStatModifier(StatsModifier);
    ApiEventManager.OnPreTakeDamage.AddListener(this, unit, PreTakeDamage, false);
}
public void PreTakeDamage(DamageData dmg) { dmg.PostMitigationDamage *= 0.7f; }
```
Duration = `2 + spellLevel - 1` seconds (`Characters/Garen/W.cs:51`), i.e.
2/3/4/5/6s by rank. While active, every instance of post-mitigation damage
Garen takes — minion autoattack, turret shot, champion hit — is cut by 30%,
with no source filter.

```csharp
// Characters/Garen/W.cs:31-46 — fires once, first time W is leveled to rank 1
AddBuff("GarenWPassive", 1, 1, spell, owner, owner, true);   // infiniteduration=true
```
The 7th `AddBuff` parameter is `infiniteduration`
(`GameServerLib/API/ApiFunctionManager.cs:273`); `true` means the buff never
expires on its own regardless of the `1`-second duration argument.
`GarenWPassive` (`Buffs/Garen/GarenWPassive.cs:31-39`) applies
`Armor.PercentBonus += 0.2f` and `MagicResist.PercentBonus += 0.2f`. So from
the moment W is first leveled (normally level 2) onward, Garen has a
permanent +20%/+20% mitigation boost against every minion and turret hit for
the rest of the game.

### 8. Turret targeting / "tower aggro" (SOURCE-READ)

**Character-name check:** not applicable — `Maps/Map1/LevelScriptObjects.cs:26`
sets `LaneTurretAI = "TurretAI"` as a literal string, passed to every
`CreateLaneTurret` call regardless of which `CharData`/model that turret
uses. `ObjAIBase`'s AI script is loaded the same name-lookup way as spell
scripts (`game.ScriptEngine.CreateObject<IAIScript>("AIScripts", aiScript)`,
`GameServerLib/GameObjects/AttackableUnits/AI/ObjAIBase.cs:230`) but the
*input* string here comes from the map script's explicit assignment, not
from the turret's character name — so this is robust to the outer-turret
model mixup in a way the `BasicAttack` script (Finding 1) is not.

```csharp
// AIScripts/TurretAI.cs:60-88
if (baseTurret.TargetUnit == null) { /* pick lowest-priority unit in range */ }
else {
    if (baseTurret.TargetUnit is Champion) { continue; }  // never re-targets off a champion
    if (!(u is Champion enemyChamp) || enemyChamp.TargetUnit == null) { continue; }
    if (!(enemyChamp.TargetUnit is Champion enemyChampTarget) || out of either's range) { continue; }
    nextTarget = enemyChamp; break;   // switch onto the aggressor
}
```
Once a turret locks onto a champion it won't re-evaluate away from it until
death/range/vision loss. If the turret is hitting something else (e.g. a
minion) and an enemy champion in range is caught autoattacking any other
champion also in range of both, the turret switches onto the aggressor — the
classic tower-aggro punish for diving/all-in trades. The code's own comment
also flags a tie-break quirk when two champions are in range simultaneously
(`TurretAI.cs:52-54`) — not exploitable in a 1v1 (only one enemy champion
exists).

### 9. Minions won't leave a turret target for a fresh wave/champion (SOURCE-READ)

**Character-name check:** `Maps/Map1/LevelScript.cs:21` sets
`LaneMinionAI = "LaneMinionAI"`, passed to every `CreateLaneMinion` call
(`:307`) regardless of minion type — same explicit-string robustness as
Finding 8.

```csharp
// AIScripts/LaneMinionAI.cs:150-173 (comments original to the file)
// Structures are not in the published 6-entry priority list at all, so nothing
// outranks a turret the minion has already committed to... It still leaves
// the turret by the normal exits - the turret dies, it falls out of
// acquisition range or vision, or the 4s failed-to-attack Ignore() fires.
if(targetIsStillValid && LaneMinion.TargetUnit is BaseTurret) { return false; }
```
Documented as deliberate (with a wiki citation in the source). A minion wave
already engaged with a turret will not divert onto Garen just because he (or
an enemy wave) arrives — it takes the turret's death, a range/vision break,
or 4 seconds of the minion failing to land a hit.

### 10. Garen Q: script-defined damage, no type discount (SOURCE-READ)

**Character-name check:** `Spell1 = "GarenQ"` in Garen's `CharData`, matching
the `GarenQ : ISpellScript` class (`Characters/Garen/Q.cs`) which adds buff
`GarenQ` (`Buffs/Garen/GarenQ.cs`); that buff swaps Garen's auto-attack spell
to `"GarenQAttack"` (`ObjAIBase.SetAutoAttackSpell(string, bool)`,
`GameServerLib/GameObjects/AttackableUnits/AI/ObjAIBase.cs:800`, does the
same namespace-lookup as any other spell name), matching `GarenQAttack :
ISpellScript`, also in `Characters/Garen/Q.cs`. Full chain confirmed.

```csharp
// Characters/Garen/Q.cs:142,151
float silenceDuration = 1.5f + 0.25f * (spellLevel - 1);
var damage = 30 + (25 * (spellLevel - 1)) + owner.Stats.AttackDamage.Total * 1.4f;
target.TakeDamage(owner, damage, DamageType.DAMAGE_TYPE_PHYSICAL, DamageSource.DAMAGE_SOURCE_ATTACK, false);
```
Both numbers are hardcoded in script, not read from the stat JSON. No
`is Minion`/`is BaseTurret` branch anywhere in `GarenQAttack` — full damage
regardless of target type (contrast with E's 0.75x/0x). Q cancels Garen's
in-flight swing and skips the next auto-attack slot, then routes the *next*
swing through `GarenQAttack` — an empowered next-auto-attack, matching real
Decisive Strike, not an instant separate hit.

### 11. Garen R (SOURCE-READ; targetability not verified)

**Character-name check:** `Spell4 = "GarenR"` in Garen's `CharData`, matching
`GarenR : ISpellScript` (`Characters/Garen/R.cs`). Confirmed.

```csharp
// Characters/Garen/R.cs:28-29
var percentMissingHP = new[] { 0.2857f, 0.3333f, 0.4f }[spell.CastInfo.SpellLevel - 1];
var damage = 175f * spell.CastInfo.SpellLevel + percentMissingHP * (Target.Stats.HealthPoints.Total - Target.Stats.CurrentHealth);
```
No type check in the script itself. Whether R can even be cast on a minion
or turret is controlled by `SpellData`'s targeting flags in the stat JSON —
out of this audit's content-script scope, not verified here.

## Answers to the specific questions

**1. Does anything modify damage based on attacker/target TYPE pair?**
Yes, three found, all in content scripts (not stat JSON):
- Turret → Minion: ×0.7, but **confirmed unreachable on the live Map 1
  config** (Finding 1) — already corrected in the sim.
- Garen (Champion) → Minion: ×0.75, from his E (`Buffs/Garen/GarenE.cs:106`)
  — live, confirmed via the character-name chain, not map-dependent.
- Garen (Champion) → Turret/Building: ×0, E cannot hit them at all
  (`GarenE.cs:101`) — live, same confirmation.
No minion→turret, minion→champion, or turret→champion discount was found
anywhere (`LaneMinionAI` and the native `AutoAttackHit` path both deal flat,
type-agnostic AD with standard mitigation).

**2. Do turrets have escalating damage against champions ("tower wrath")?**
No. `S5Test_TowerWrath` is inert on every map, and on the live map the call
site that would even add it is unreachable. Separately, turrets *do* get
scripted, non-JSON flat AD/Armor/MR growth over time via
`LevelScriptObjects.UpdateTowerStats`/`UpdateOuterTurretStats` (Finding 5,
SOURCE-READ, TurretType-gated so unaffected by the model mixup) — the real,
if much less dramatic, analogue of a "towers get stronger" mechanic here.

**3. Minion damage falloff / level scaling / time-based growth by script?**
None found for minions. `LaneMinionAI.cs` only does targeting/movement.
`LevelScript.cs`'s `MinionModifiers` dict is populated but never applied
(already known). No other minion-related script touches AD/Armor/HP by time
or wave count. For turrets, in contrast, extensive script-driven,
non-JSON, time- and enemy-count-based stat growth does exist (Finding 5).

**4. Garen's passive: what does it do, and from where is it applied?**
`CharScriptGaren.OnUpdate` (`Characters/Garen/CharScriptGaren.cs:74-82`),
called every tick via `ICharScript.OnUpdate` (confirmed to load: Garen's
`CharData.MetaData.Id = "Garen"` → `LoadCharScript` builds
`$"CharScript{Model}"` = `"CharScriptGaren"`, matching the class exactly),
waits until `GameTime() > 1f` then, exactly once, calls
`AddBuff("GarenPassive", 99999f, 1, ...)`. This replaced a prior `Task.Run`
background-thread implementation (documented in-line, `:41-70`) that was
nondeterministic, wall-clock-speed-dependent, and mutated state off the game
thread. `GarenPassive` (`Buffs/Garen/GarenPassive.cs`) does no healing
itself; it `AddBuff`s `GarenPassiveHeal`, which ticks every 1000ms healing
`HEALTH_PERCENTAGES[level bracket] × HealthPoints.Total` (0.4%/0.8%/2% of max
HP/second for levels 1-10/11-15/16-18, `GarenPassiveHeal.cs:33`), and listens
for `OnTakeDamage` via `ShouldPassiveTurnOff`; when triggered it swaps to
`GarenPassiveCooldown` (waits `GetCooldownForLevel` = 9/6/4s by the same
brackets) before swapping back. The turn-off condition has the enum-collision
quirk in Finding 3.

## Files searched with no additional findings worth reporting

- `AIScripts/BasicJungleMonsterAI.cs`, `AIScripts/Pet.cs` — jungle
  monster/summon-specific, out of scope.
- `AIScripts/MinionAI.cs` (`MinonAI`) — confirmed not wired to lane minions
  on Map 1 (explicit-string check, Finding 17); no numeric modifiers in it
  regardless.
- `Buffs/Global/*` other than `HPByPlayerLevel` and `Recall` — searched, all
  generic CC (`Blind`, `Disarm`, `Silence`, `Slow`, `Stun`), a DoT
  (`Burning`), infrastructure (`ExpirationTimer`), or other-game-mode-only
  (`GlobalMonsterBuff` [jungle], `HowlingAbyssAura`/`URFBuff` [ARAM/URF],
  `OdinChannelVision`/`OdinPlayerBuff` [Dominion]). None reference Garen,
  lane minions, or lane/nexus/fountain turrets specifically.
- `Maps/Map1/LevelScript.cs` `MinionModifiers` — re-confirmed declared and
  populated in `Init()` (`:123-157`) but never read/applied anywhere; not
  re-reported as new per your instruction.
- Garen's `E.cs`/`W.cs` cast wrappers (`Characters/Garen/E.cs`,
  `Characters/Garen/W.cs`) carry no damage numbers themselves — those live in
  the corresponding `Buffs/Garen/*.cs` files, detailed above.

## Open items / suggested next measurements

- **Finding 5 (turret time-based growth)** is the highest-value remaining
  SOURCE-READ-only claim: it directly changes turret AD/Armor/MR over the
  course of a match. Cheapest confirmation: reuse the existing 600s
  idle-recording harness and check outer-turret single-hit damage at
  t≈0:30, 1:30, ..., 7:30 for `+4` AD steps (7 of them), then inner-turret
  damage from t=8:00 for the `+1/+1/+4` schedule.
- **Inner/inhibitor/nexus/fountain turret identity on Map 1** (the four rows
  in the corrected table marked SOURCE-READ only) have not individually been
  put through the same 3-point measurement the outer turret got. Given how
  costly the outer-turret name mixup already turned out to be, the same
  check is cheap insurance if any of those tiers matter to a scenario later
  (e.g. inhibitor sieges).
