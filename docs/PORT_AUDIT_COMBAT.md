# Port Audit: Stats, Damage, Buffs, Spells, Missiles, Death/Gold/XP

Scope: the subsystem covering `Stats.cs`/`AttackableUnit.cs`/`Spell.cs`/missiles/
sectors/buffs/Garen's kit/XP+gold-on-death/autoattack, for the top-lane 1v1
Garen-vs-Garen slice on Map1 that `lanerl_jax` actually simulates.

Server commit audited: `lanerl-vendor/LoLServer` @ `be1dc8b` (branch
`lanerl/wip`, 2026-09-15). **This is not vanilla LeagueSandbox** — it is a
fork carrying (a) `GameServerLib/Lanerl/*.cs`, ~5,000 lines of RL-harness glue
(episode reset, bot AI, wire protocol) with no vanilla equivalent, and (b)
several already-fixed Content-script bugs (e.g. `Buffs/Garen/GarenE.cs`'s
`OrderBy(unit => DistanceSquared(unit.Position, unit.Position))` self-compare
and its unseeded per-tick `new Random()`, both fixed in-place with comments
explaining the original bug). Verified identical to `lanerl-vendor/GameServer`
(`diff` empty on `Stats.cs`); **do not** cite `lanerl-vendor/RL-Learning`,
which differs by 380 lines on `Stats.cs` alone and is a different, older fork.

JAX citations are against `lanerl_jax/sim/*.py` and `lanerl_jax/data/patch.py`
in worktree `ahriuwu-lanerl-jax-a3` @ `024e19c` (`lane-rl/jax-a3`), including
uncommitted state at audit time.

**Method.** Every C# citation below was opened and read at the cited lines in
this session (or by one of four parallel sub-audits scoped to a single file
set each, briefed with the same bar) — not inferred from an existing
docstring or test. Where an existing `lanerl_jax` comment already asserted a
claim, it is marked as *(re-verified)* if the source was independently
re-read and confirmed, or flagged explicitly where re-reading found the
comment wrong, stale, or unverifiable from the cited files.

**Verdict bar (per the coordinator's standing instruction).** EXACT requires
the same control flow (guard order, boundary cases: zero, negative, dead,
first/last tick), the same arithmetic (operation order, rounding, clamp
placement), and the same position in the tick, demonstrated — not "looks
similar". APPROX states the deviation, whether it's already documented in our
code, and whether fixing it is JAX-hostile. UNVERIFIED is used in preference
to a confident-but-unchecked EXACT.

---

## 0. Headline findings (read this first)

1. **Garen's W ("Courage") active does not reduce damage taken on the real
   server — it only changes what number the floating combat text shows.**
   `AttackableUnit.TakeDamage` (`AttackableUnit.cs:551`) copies
   `damageData.PostMitigationDamage` into a local `float` **before**
   `OnPreTakeDamage` fires (`:558`), and the line that actually subtracts HP
   (`:585`) and the line that computes lifesteal (`:615`) both read that
   *stale* local, never the field `GarenW.PreTakeDamage`
   (`Buffs/Garen/GarenW.cs:49-54`) mutates. `lanerl_jax` implements the
   *intended* 30% reduction as a real, effective multiplier
   (`spells.py:415` `W_DAMAGE_MULT=0.7`, applied in `step.py:513`). This is a
   confirmed **WRONG**, not a judgment call — see §5.2. It is the single
   highest-leverage row in this document: a policy trained against the sim
   currently believes W absorbs 30% of incoming damage, and the server never
   pays that out.
2. **Champion-kill gold and XP are entirely unimplemented**, and **turret
   destruction pays zero gold/XP in the sim** (reads always-zero fields). See
   §8. Both are real, not merely dormant, the moment a champion or a tower
   dies in this 1v1.
3. Two more real, load-bearing gaps confirmed by direct reading, not
   inherited from any existing comment:
   - `UnitTag`'s enum-collision bug (used to gate Garen's out-of-combat
     regen) also affects **super minions**, not only cannon minions as our
     code currently special-cases, and the **level ≥ 11 exemption** for both
     is missing entirely from `step.py`. See §3.3.
   - A missile's damage payload freezes the firing unit's raw AD at launch;
     the server re-reads it live at impact (`SpellMissile.cs:190-192`
     calls `AutoAttackHit`, which reads `Stats.AttackDamage.Total` fresh).
     Silent for minions (AD never changes) and small/rare for turrets
     (matters only if an AD-ramp tick lands mid-flight). See §6.3.
4. Garen's Q empowered-attack damage+silence is confirmed **still not wired**
   into `autoattack.py` (casting Q currently opens the window and starts the
   cooldown, but the "next attack" deals ordinary damage) — matches the
   brief's description exactly. See §7 and §9.2.
5. One thing that turned out **already fixed** and worth not re-flagging:
   `spells.py`'s own docstring for R and W says three `step.py` integration
   hooks are "not made here" — re-reading current `step.py` (`:280-281`,
   `:513`) shows all three are wired. The docstring is stale, the code is
   ahead of it. R's damage is correctly mitigated against `magic_resist`, not
   `armor` (`step.py:262` passes `magic_resist=P("magic_resist")` into
   `step_buffs`).

---

## 1. Base stats (numeric table, Garen, `LeagueSandbox-Default/Stats/Garen/Garen.json`)

| field | value | consumed by (ours) |
|---|---|---|
| `BaseHP` | 616.28 | `data/patch.py` `UnitStats.base_hp` |
| `HPPerLevel` | 96 | `ad_per_level`-style column, `profiles.py` |
| `BaseStaticHPRegen` | 1.568 | `hp_regen` profile column |
| `HPRegenPerLevel` | 0.1 | not applied per-level anywhere found (see §2.4) |
| `BaseMP` / `MPPerLevel` | 0 / 0 | N/A, `PARType: None` |
| `BaseStaticMPRegen` / `MPRegenPerLevel` | 0 / 0 | N/A |
| `BaseDamage` (AD) | 57.88 | `patch.champion.base_ad` |
| `DamagePerLevel` | 3.5 | `ad_per_level` (fixed today, `sim/step.py`) |
| `Armor` | 27.536 | `patch.champion.base_armor` |
| `ArmorPerLevel` | 2.7 | growth curve, `profiles.py` |
| `SpellBlock` (MR) | 32.1 | `patch.champion.base_mr` |
| `SpellBlockPerLevel` | 1.25 | growth curve — **no per-level MR growth found wired for champions in `profiles.py`'s column list**, see §2.4 |
| `AttackRange` | 125 (melee) | `fires_missile = False` |
| `MoveSpeed` | 345 | movement subsystem (audit-only, not owned here) |
| `AttackSpeedPerLevel` (`GrowthAttackSpeed`) | 2.9 | see §2.5 |
| `AttackDelayOffsetPercent` | 0 | `combat.attack_speed_flat` |
| `AttackDelayCastOffsetPercent` | -0.091666667 | `combat.attack_windup`, matches the `-0.0917` cited in `combat.py`'s docstring **exactly** (re-verified against JSON, not just trusted) |
| `IsMelee` | true | `fires_missile` |

**The rune/mastery baseline** (`lanerl/cfg/garen1v1.json`'s fixed 30-rune
page + 18 talent points, applied once at spawn by `Champion.OnAdded` and
preserved across in-process episode resets by `LanerlEpisode.RestoreBaseline`,
`GameServerLib/Lanerl/LanerlEpisode.cs:196-224`) is **not re-derived
analytically** — correctly so, since that would mean re-implementing the full
S4 rune/mastery system for one config. It is measured against a
`LANERL_STATEROW` dump and stored as three named deltas in
`sim/init.py:183-224`:

| stat | Content base | measured (rune+mastery) | delta |
|---|---|---|---|
| Max HP @ L1 | 616.28 | 754.248046875 | `RUNE_HP_BONUS` = +137.968 |
| Attack damage @ L1 | 57.88 | 78.134765625 | `RUNE_AD_BONUS` = +20.2548 |
| Armor | 27.536 (≈27.5361328125 at wire precision) | 36.5361328125 | `RUNE_ARMOR_BONUS` = +9.0 |

Verdict: **APPROX by design, documented, not JAX-hostile to fix better** — it
is exact for the three stats it covers, and it is a measured constant rather
than a derivation, which is the right call given the alternative is porting
old-style runes/masteries wholesale. **Gap**: MR (`SpellBlock`), mana/mana
regen (N/A, Garen has none), and HP/mana regen-per-level bonuses from the
same page are not measured or modelled at all — the rune page in this config
almost certainly includes MR/level glyphs (glyphs are MR/level in every
S3/S4-era page), so Garen's simulated MR is very likely under the server's by
some untracked constant, at every level. **Design decision for the human**:
either measure the MR delta the same way (one more idle-dump diff) or accept
the gap; low practical impact in an all-physical Garen mirror but not zero
(affects any true/magic damage source, including Garen's own R, which is
magical — see §9.4).

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Champion base-stat baseline (rune/mastery page) | `LanerlEpisode.cs:104-224` (`Capture`/`RestoreBaseline`); page itself applied by `Champion.OnAdded` (not read this session — outside file list, inferred from the comment at `LanerlEpisode.cs:97-104`) | `sim/init.py:183-224`, `sim/profiles.py:178-184` | APPROX (measured, documented, HP/AD/Armor only) | See table above. MR/mana-regen/HP-regen page deltas: MISSING, low-priority given no magic damage source is reachable except Garen's own R. |

---

## 2. Per-level growth (`Stats.LevelUp`, `Stats.cs:270-286`)

Full method, re-read fresh this session:

```csharp
public void LevelUp()
{
    Level++;
    StatsModifier statsLevelUp = new StatsModifier();
    statsLevelUp.HealthPoints.BaseValue = GetLevelUpStatValue(HealthPerLevel);
    statsLevelUp.ManaPoints.BaseValue = GetLevelUpStatValue(ManaPerLevel);
    statsLevelUp.AttackDamage.BaseValue = GetLevelUpStatValue(AttackDamagePerLevel.BaseValue);
    statsLevelUp.AttackDamage.FlatBonus = GetLevelUpStatValue(AttackDamagePerLevel.FlatBonus);
    statsLevelUp.Armor.BaseValue = GetLevelUpStatValue(ArmorPerLevel);
    statsLevelUp.MagicResist.BaseValue = GetLevelUpStatValue(MagicResistPerLevel);
    statsLevelUp.HealthRegeneration.BaseValue = GetLevelUpStatValue(HealthRegenerationPerLevel);
    statsLevelUp.ManaRegeneration.BaseValue = GetLevelUpStatValue(ManaRegenerationPerLevel);
    statsLevelUp.AttackSpeed.PercentBaseBonus = GetLevelUpStatValue(GrowthAttackSpeed / 100.0f);
    AddModifier(statsLevelUp);
    CurrentHealth += statsLevelUp.HealthPoints.BaseValue;
    CurrentMana += statsLevelUp.ManaPoints.BaseValue;
}
public float GetLevelUpStatValue(float value) => value * (0.65f + 0.035f * Level);
```

`Level++` runs **first**, so `GetLevelUpStatValue` at the transition to level
`N` uses the **post-increment** `Level == N`. `AddModifier` → `Stat.
ApplyStatModifier` does `BaseValue += statModifier.BaseValue` (`Stat.cs:84`),
so repeated `LevelUp()` calls accumulate additively. Closed form for the
total growth from level 1 to level `N`:

```
growth_sum(N) = sum_{L=2..N} (0.65 + 0.035*L) = 0.65*(N-1) + 0.035*(N*(N+1)/2 - 1)
```

Independently cross-checked against `Maps/Map1/StatsProgression.json`'s
`PerLevelStatsFactor` table, which is the **per-step** coefficient
`0.65+0.035*L` at each `L` (not the cumulative sum — re-derived by hand this
session, since the existing `combat.py` docstring's phrasing is easy to
misread as the cumulative value): Level2=0.72, Level10=1.0, Level11=1.035,
Level18=1.28, all match `0.65+0.035*L` exactly for `L`=2,10,11,18. (Nothing
in `Stats.cs` reads this JSON — `GetLevelUpStatValue` computes the constant
inline — so this table is independent corroborating data, not the mechanism
itself; likely consumed by a client-side tooltip preview, outside this
subsystem.)

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Per-level growth formula (shape + rounding) | `Stats.cs:270-286` | `combat.py:181-206` (`growth_sum`, `level_up_factor`, `stat_at_level`) | EXACT | Closed-form algebraically equal to the additive accumulation; independently cross-checked against `StatsProgression.json` (above), not merely against the docstring. |
| AD grows per level (the fact fixed today) | `Stats.cs:274` (`AttackDamage.BaseValue` gets `GetLevelUpStatValue`) | `sim/step.py:456` `ad_now = P("attack_damage") + P("ad_per_level")*growth_sum(...)`; `profiles.py` `ad_per_level` column; test `sim/tests/test_champion_level_scaling.py` | EXACT | Re-verified against `Stats.cs` directly this session (not re-derived). `DamagePerLevel=3.5` from Garen.json now flows through. |
| Max HP grows per level | `Stats.cs:273` | `profiles.py` HP growth via same `stat_at_level` path (pre-existing, not part of today's fix) | EXACT | Same mechanism confirmed. |
| Armor grows per level | `Stats.cs:275` (`ArmorPerLevel=2.7`) | `profiles.py`/`combat.py` growth path — column exists (`armor` gathered with `stat_at_level`-equivalent in `build_profile_tables`, confirmed by reading `profiles.py:117-160`) | EXACT | Re-verified this session by reading `profiles.py` end to end, not assumed from the AD fix alone. |
| **Magic Resist grows per level** | `Stats.cs:276` (`MagicResist.BaseValue` gets `GetLevelUpStatValue(MagicResistPerLevel)`, `MagicResistPerLevel=1.25` from `SpellBlockPerLevel`) | **not found**: `profiles.py`'s `build_profile_tables` column list (`sim/profiles.py:117-120`) has no `magic_resist_per_level` and no per-level growth applied to `magic_resist` anywhere in `step.py` | **MISSING** | Grepped `lanerl_jax/sim/*.py` for `mr_per_level`/`magic_resist_per_level`/any `growth_sum` call on a magic-resist column: none. Garen's simulated MR is frozen at the (rune-augmented, but not level-augmented) L1 value for the whole episode. Reachable and observable the moment R (magical) or any future magic source is used against him — currently the only magic-damage path in the kit (§9.4), so the effect is confined to how much R hurts Garen's mirror, but it is real at every level ≥ 2. |
| **HP regen grows per level** | `Stats.cs:277` (`HealthRegenerationPerLevel=0.1` from `HpRegenPerLevel`) | **not found** in `regen.py`/`profiles.py` — `hp_regen` is a flat per-profile constant (`regen.py:33`: "Garen is 1.568 + 0.1 per level" **in a comment**, but no code applies the `+0.1*growth` term) | **WRONG** (comment claims it, code doesn't do it) | `regen.py:33`'s own docstring states the per-level number but `profiles.py`'s `hp_regen` column (`build_profile_tables`) reads only `patch.champion.hp_regen` (the base `BaseStaticHpRegen`), with no `growth_sum` term added — confirmed by reading the full column-build loop, no second term. Low magnitude (0.1/level × 0.65-1.28 factor ≈ 0.065-0.13 HP/s at high levels) but it is the same class of bug as the AD-per-level one that motivated this audit, and it is currently **mis-documented as done** in the very comment sitting next to the gap. |
| Mana / mana regen per level | `Stats.cs:271,278` | N/A | N/A | Garen has `PARType: None`, `ManaPerLevel=0`, `ManaRegenerationPerLevel=0` — no-op on both sides regardless. |
| Attack speed per level | `Stats.cs:279` (`AttackSpeedMultiplier.PercentBaseBonus` gets `GetLevelUpStatValue(GrowthAttackSpeed/100)`, `GrowthAttackSpeed=2.9`) | see §2.5 below | see §2.5 | |
| `CurrentHealth`/`CurrentMana` increment on level-up uses the **raw** `BaseValue` delta, not the resulting `Stat.Total` delta | `Stats.cs:284-285`: `CurrentHealth += statsLevelUp.HealthPoints.BaseValue;` (not `HealthPoints.Total`'s delta) | not checked against this specific formula — see §2.6 | UNVERIFIED | Exact only when `HealthPoints.PercentBaseBonus`/`PercentBonus` are both 0 at the moment of leveling (true for this project's no-%-HP-item config); would silently under-heal a level-up the instant a %-max-HP item entered the build. Flag for whoever owns the level-up HP-credit code in `step.py`, not independently located this session. |
| `Stats.RemoveModifier` omits `AttackDamagePerLevel.RemoveStatModifier` | `Stats.cs:189-206` (compare `AddModifier`'s `AttackDamagePerLevel.ApplyStatModifier(modifier.AttackDamagePerLevel)` at `:154`, absent from `RemoveModifier`) | N/A | **N/A (server bug, unreachable)** | Genuine, confirmed-by-diff asymmetry in the server itself: a buff/item that raises `AttackDamagePerLevel.FlatBonus` would leak on removal. BUG-COMPAT candidate *if* anything in this project's item/buff set ever touches `AttackDamagePerLevel` — grepped Garen's kit and the fixed `BuildPath` item list (`LanerlConfig.cs:327`, IDs 1054/2003/2003/1001/3047/3134/3068/3035 — Doran's Shield, 2×Health Potion, Boots, and four late items): none is a per-level-AD-scaling item. Not reachable today; recorded for completeness per the exhaustiveness requirement. |

### 2.4 Stat composition (`Stat.Total`, `Stat.cs:70`)

```csharp
public virtual float Total => ((BaseValue + BaseBonus) * (1 + PercentBaseBonus) + FlatBonus) * (1 + PercentBonus);
```

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Stat composition order | `Stat.cs:70` | `combat.py:196-199` (`stat_total`) | EXACT | Textually identical formula, re-read fresh (not assumed). |
| Modifier apply/remove semantics | `Stat.cs:84-107` (`ApplyStatModifier`/`RemoveStatModifier`, both `+=`/`-=` on all five fields, gated by `StatModifier.StatModified` — an epsilon-based dirty check with no numeric effect since it only skips a no-op `+=0`) | Not modelled as a generic five-field Stat object anywhere in `lanerl_jax` — stats are flat arrays recomputed from profile+level+explicit per-effect terms each tick (confirmed by the buff-engine sub-audit, §4) | N/A (architecture mismatch, not a bug) | Our side has no direct analogue to `AddModifier`/`RemoveModifier`; each stat-affecting effect (W passive's armor/MR %, W active's damage multiplier) is hand-wired in `step.py`/`spells.py` instead. Reasonable JAX-shaped design; flagged as a standing fact, not scored per-mechanic beyond the individual effects already covered elsewhere in this table. |

### 2.5 Attack speed

```csharp
// Stats.cs:130 (LoadStats)
AttackSpeedFlat = 1.0f / GlobalCharacterDataConstants.AttackDelay / (1.0f + charData.AttackDelayOffsetPercent);
// Stats.cs:227-230
public float GetTotalAttackSpeed() => AttackSpeedFlat * AttackSpeedMultiplier.Total;
```

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| `AttackSpeedFlat` formula | `Stats.cs:130` | `combat.py:213-224` (`attack_speed_flat`) | EXACT | Re-verified against `Stats.cs`, including the fountain's `AttackDelayOffsetPercent=-1` → `1/0` → `+Infinity` edge case, which `combat.py` reproduces via `np.errstate(divide="ignore")` rather than crashing — a genuinely non-obvious C#-float-semantics match. |
| Attack speed grows per level | `Stats.cs:279` (`AttackSpeedMultiplier.PercentBaseBonus += GetLevelUpStatValue(GrowthAttackSpeed/100)`) | **not found**: no `growth_sum` applied to an attack-speed multiplier anywhere in `combat.py`/`profiles.py`/`step.py` (grepped `AttackSpeedMultiplier`, `attack_speed_mult`, `as_per_level`: none) | **MISSING** | Same class of omission as HP-regen-per-level (§2 table): `GrowthAttackSpeed=2.9` (i.e. +2.9% AS per `GetLevelUpStatValue` factor, compounding through the same non-linear curve) is not applied anywhere. At level 18 this is `2.9% * growth_sum(18)/... ` — using `level_up_factor` cumulative sum, `growth_sum(18) ≈ 0.65*17+0.035*(18*19/2-1) = 11.05+0.035*170=11.05+5.95=17.0`, so `PercentBaseBonus ≈ 0.029*17.0 ≈ 0.493`, i.e. **attack speed multiplier should be ~1.49x by level 18**, applied to `AttackSpeedFlat` — currently the sim's attack period is flat across all levels. This directly changes DPS and last-hit timing at every level past 1, compounding with every other AD-scaling error already fixed. High priority — see §10. |

### 2.6 HP regen tick (`Stats.Update`, `Stats.cs:243-259`)

```csharp
public void Update(float diff)
{
    if (HealthRegeneration.Total > 0 && CurrentHealth < HealthPoints.Total && CurrentHealth > 0)
    {
        var newHealth = CurrentHealth + HealthRegeneration.Total * diff * 0.001f;
        CurrentHealth = Math.Min(HealthPoints.Total, newHealth);
    }
    if ((byte)ParType > 1) return;
    if (ManaRegeneration.Total > 0 && CurrentMana < ManaPoints.Total) { ... }
}
```

Called from `AttackableUnit.Update` (`AttackableUnit.cs:237-267`), confirmed
fresh this session:

```csharp
public override void Update(float diff)
{
    UpdateBuffs(diff);
    _statUpdateTimer += diff;
    while (_statUpdateTimer >= 500)
    {
        Stats.Update(_statUpdateTimer);     // NOTE: passed the un-decremented accumulator
        _statUpdateTimer -= 500;
        ...
    }
    ...
    if (CanMove()) { ... Move(...) ... }
    if (IsDead && _death != null) { Die(_death); _death = null; }
}
```

This confirms the brief's ground-truth tick order **exactly**:
`UpdateBuffs → Stats.Update (regen) → Move → Die`, all inside one
`AttackableUnit.Update` call.

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Per-unit tick order (buffs → stat tick → move → die) | `AttackableUnit.cs:237-267` | `step.py:251-296` (comments at `:251-253`, `:281-282` cite this exact order) | EXACT | Re-read both sides fresh; order matches line for line. |
| Base regen guard (`Total>0 && hp<max && hp>0`) placed *inside* the update, clamp to max placed *inside* the same guard | `Stats.cs:245-250` | `regen.py:100,127-135` (`can_regen = alive & (hp>0) & (hp<max_hp)`, clamp only applied `where(gain>0, ...)`) | EXACT | This is the exact clamp-inside-guard pattern the coordinator flagged as a past bug class (a fixture minion died when a clamp sat outside this guard); re-read `Stats.cs` fresh and confirmed the current `regen.py` code puts it back inside, with a comment (`regen.py:120-131`) explaining the earlier failure mode. Mathematically proved (not merely asserted) that combining base+passive into one `gain>0`-gated clamp equals two sequential non-negative clamped updates, given every reachable `hp_regen` in this config is ≥ 0 (verified: Garen 1.568+0.1/lvl, minions/turrets 0) — see derivation below. |
| `_statUpdateTimer` accumulator re-fires with the **un-decremented** value each loop iteration (`Stats.Update(_statUpdateTimer)`, not `Stats.Update(500)`) — a real over-count if `diff` ever exceeds ~500ms in one server frame | `AttackableUnit.cs:242-248` | `regen.py:110-115`: single `if fires` check (no `while`), always uses the full accumulated `st` | EXACT for both sides' actual operating envelope | The server's own per-frame `diff` is small (tens of ms) so the `while` loop fires 0-1 times per call in practice; `lanerl_jax`'s tick is `TICK_MS=1000/60≈16.7ms` (`movement_jax.py:54`), so `stat_timer+delta_ms` never reaches 1000 and a single fire always suffices. The multi-fire compounding case is unreachable on **both** sides at these tick granularities — noted, not scored as a gap. |
| `Stats.Update` skips mana regen when `(byte)ParType > 1` | `Stats.cs:251-254` | N/A (`ManaRegenerationPerLevel=0`, `ManaPerLevel=0` for Garen; no mana modelled in `lanerl_jax` at all) | N/A | Garen's `PARType=None`; moot regardless of the guard's outcome. |
| `TakeHeal` clamps on write to `[0, Max]` | `AttackableUnit.cs:478-481`: `Stats.CurrentHealth = Math.Clamp(Stats.CurrentHealth + amount, 0, Stats.HealthPoints.Total)` | `regen.py:132-135` clamps Garen's passive heal the same way, combined with base regen | EXACT | Confirms no server-side "overheal stored internally, revealed later" behavior exists for the heal path (unlike the bare `CurrentHealth` setter, which does NOT clamp — see next row) — so the combined single-clamp approach in `regen.py` cannot diverge from two sequential clamped writes. |
| `Stats.CurrentHealth`/`CurrentMana` **getters** clamp to max, but the **setter** does not (`_currentHealth = value` unclamped in `Stats.cs:65-73`) | `Stats.cs:65-73` | `regen.py`/`combat.py` always clamp on write (`jnp.minimum(hp+gain, max_hp)`, `jnp.maximum(hp-dealt, 0)`) | APPROX (documented gap, very low impact) | Any code path that sets `CurrentHealth` directly without an explicit clamp (e.g. `Stats.LevelUp`'s `CurrentHealth += statsLevelUp.HealthPoints.BaseValue`, §2 table) relies on the getter's clamp as a safety net, meaning the *internal* stored value can silently exceed max HP and only become externally visible again if max HP later shrinks without an intervening write. `lanerl_jax` stores a single clamped-on-write `hp` array with no such hidden headroom. Unreachable in this project's no-%-HP-item config (see §2 table's level-up row); flagged for completeness, not actionable today. |

**Regen guard-equivalence proof** (referenced above): for non-negative
`base, passive` and constant `max_hp` within one tick,
`min(hp+base+passive, max_hp) == min(min(hp+base, max_hp)+passive, max_hp)`
— case `hp+base ≤ max_hp`: both sides equal `hp+base+passive`. Case
`hp+base > max_hp`: LHS `= max_hp` (since `passive ≥ 0` keeps the sum
`> max_hp`); RHS `= min(max_hp+passive, max_hp) = max_hp`. Equal in both
cases, so the single-clamp implementation is exact whenever both deltas are
non-negative — true for every profile in this config (§2.6 table).

### 2.7 UnitTag enum collision (Garen's passive interruption)

`GameServerCore/Enums/UnitTag.cs:6-31` declares `[Flags] enum UnitTag` with
**no explicit values**, so C# numbers members sequentially: `Champion=0,
Champion_Clone=1, Minion=2, Minion_Lane=3, Minion_Lane_Siege=4,
Minion_Lane_Super=5, Minion_Summon=6, Monster=7, ...`. `CharData.cs:152-155`
OR-combines a unit's `UnitTags` string from Content directly into this
non-bitmask-safe enum.

Measured combinations (verified against the actual Content JSON files, not
assumed):

| unit | `UnitTags` string (Content) | raw value |
|---|---|---|
| `SRU_OrderMinionMelee`/`Ranged` | `"Minion \| Minion_Lane"` | 2\|3 = **3** (= `Minion_Lane`) |
| `SRU_ChaosMinionSiege` (cannon) | `"Minion \| Minion_Lane \| Minion_Lane_Siege"` | 2\|3\|4 = **7** (= `Monster`) |
| `SRU_OrderMinionSuper` (super) | `"Minion \| Minion_Lane \| Minion_Lane_Super"` | 2\|3\|5 = **7** (= `Monster`) |

`CharScriptGaren.ShouldPassiveTurnOff` (`Characters/Garen/CharScriptGaren.cs:101-116`):

```csharp
public static bool ShouldPassiveTurnOff(AttackableUnit unit, DamageData damageData)
{
    if (MINION_UNIT_TAG_PASSIVE_EXCEPTIONS.Contains(damageData.Attacker.CharData.UnitTags))
        return false;                                              // exact-value membership: {2,3,4,5,6}
    if (unit.Stats.Level >= 11 && UnitTag.Monster.Equals(damageData.Attacker.CharData.UnitTags))
        return false;                                              // level-gated exemption for raw value 7
    if (damageData.PostMitigationDamage <= 0)
        return false;
    return true;
}
```

Because both cannon **and** super minions collide to raw value 7, they are
**identical** under this logic: below Garen's level 11 they interrupt his
passive regen (fall through the first check, fail the level gate, hit the
final `return true`); at level ≥ 11 the second check exempts *both* (since
`UnitTag.Monster.Equals(7)` is true for either). Melee/caster minions (raw
value 3, in the exceptions list) never interrupt it at any level.

`step.py:517-546`'s actual code:

```python
_cannon = (state.kind == Kind.LANE_MINION) & (_minion_type_of(state) == MinionType.CANNON)
breaks_combat = ~((state.kind == Kind.LANE_MINION) & ~_cannon)
```

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Melee/caster minion damage never interrupts Garen's passive | `CharScriptGaren.cs:104-105` (raw value 3 ∈ exceptions) | `step.py:521-522` (`breaks_combat=False` for non-cannon lane minions) | EXACT | Both raw values collapse to the same case. |
| **Cannon minion damage interrupts the passive only below level 11** | `CharScriptGaren.cs:106-109` (raw value 7, exempted only when `unit.Stats.Level>=11`) | `step.py:521-522` (`breaks_combat=True` for cannon **unconditionally**, no level check anywhere in the expression) | **WRONG** | Confirmed by direct bit arithmetic and direct reading of `ShouldPassiveTurnOff`, not by trusting `step.py`'s own comment (which *describes* the level-11 exemption correctly at `step.py:508` but the code below it does not implement it). Cannon hits at level ≥11 currently wrongly reset Garen's out-of-combat clock. |
| **Super minion damage should behave identically to cannon** (raw value also 7) | Same code path — `Minion_Lane_Super` collides with `Monster` exactly as `Minion_Lane_Siege` does | `step.py:521-522`: `_cannon` only flags `MinionType.CANNON`; super minions fall into `~_cannon`, so `breaks_combat=False` for them **always**, at every level | **WRONG** | Independently derived the collision (2\|3\|5=7) and cross-checked both Content JSON files myself; the existing `step.py` comment (`:508`) never mentions super minions at all, i.e. this half of the bug was not previously known to this project. Super-minion waves (post-inhibitor push) currently never interrupt Garen's regen at any level, when the server interrupts it below level 11. |

---

## 3. Damage, mitigation, death (`AttackableUnit.cs`)

### 3.1 `GetPostMitigationDamage` (`Stats.cs:307-330`)

```csharp
public float GetPostMitigationDamage(float damage, DamageType type, AttackableUnit attacker)
{
    if (damage <= 0f) return 0.0f;
    float stat = type switch {
        DAMAGE_TYPE_PHYSICAL => Armor.Total,
        DAMAGE_TYPE_MAGICAL => MagicResist.Total,
        DAMAGE_TYPE_TRUE => return damage,   // early return, no stat lookup
        _ => throw ...
    };
    float mitigationPercent = 100 / (100 + stat);
    if (stat < 0) mitigationPercent = 2 - mitigationPercent;
    return damage * mitigationPercent;
}
```

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Physical/magical mitigation formula + negative-resist branch + `damage<=0` guard | `Stats.cs:307-330` | `combat.py:158-168` (`post_mitigation_damage`) | EXACT | Re-read fresh. Order differs cosmetically (server checks `damage<=0` first and short-circuits before computing `stat`; ours computes `pct` unconditionally then gates the final `where` on `damage<=0`) but `pct`'s computation never depends on `damage`'s sign, so the two are numerically identical for every input, including `stat<0` (armor/MR shred past zero, made *finite and amplifying* rather than a `100/0` division). No minimum-damage floor on either side (confirmed: no "damage ≥ 1" rule anywhere in `GetPostMitigationDamage`). |
| True damage: early return, no mitigation, no resist read at all | `Stats.cs:317` (`case DAMAGE_TYPE_TRUE: return damage;`) | `combat.py:158-159` docstring explicitly warns callers not to route true damage through `post_mitigation_damage` with `resist=0` ("same number by luck, not by contract") | EXACT (by discipline) | Confirmed no true-damage source in the Garen kit calls `post_mitigation_damage` — R is magical (§9.4), not true; nothing in this kit deals true damage. N/A in practice but the contract is correctly stated. |

### 3.2 `TakeDamage` (`AttackableUnit.cs:544-623`, full method re-read)

```csharp
public virtual void TakeDamage(DamageData damageData, DamageResultType damageText, IEventSource sourceScript = null)
{
    float healRatio = 0.0f;
    var attacker = damageData.Attacker;
    var attackerStats = damageData.Attacker.Stats;
    var type = damageData.DamageType;
    var source = damageData.DamageSource;
    var postMitigationDamage = damageData.PostMitigationDamage;      // (1) captured HERE

    if (!CanTakeDamage(type)) return;

    ApiEventManager.OnPreTakeDamage.Publish(damageData.Target, damageData);   // (2) W's PreTakeDamage mutates damageData.PostMitigationDamage HERE

    if (... lifesteal/spellvamp ratio switch ...) { healRatio = ...; }
    if (this is Champion c && damageData.Attacker is Champion cAttacker) c.AddAssistMarker(cAttacker, 10.0f, damageData);

    Stats.CurrentHealth = Math.Max(0.0f, Stats.CurrentHealth - postMitigationDamage);   // (3) uses the LOCAL from (1), NOT the mutated field from (2)

    ApiEventManager.OnTakeDamage.Publish(damageData.Target, damageData);
    if (!IsDead && Stats.CurrentHealth <= 0) { IsDead = true; _death = new DeathData {...}; }
    if (attacker.Team != Team) _game.PacketNotifier.NotifyUnitApplyDamage(damageData, ...);   // (4) reads damageData.PostMitigationDamage, the MUTATED value

    if (healRatio > 0) attackerStats.CurrentHealth = Math.Min(attackerStats.HealthPoints.Total, attackerStats.CurrentHealth + healRatio * postMitigationDamage);  // (5) ALSO the stale local from (1)
}
```

`DamageData` is a `class` (`AttackableUnits/DamageData.cs:6`), so (2)'s
mutation genuinely changes the shared object's `PostMitigationDamage`
*property* — but (3) and (5) never re-read that property; they read the
`float` local captured before (2) ran. C# floats are value types, so this is
airtight regardless of `DamageData`'s reference-vs-value-type status. Line
(4) is the one place that *does* see the post-(2) value, meaning the
client-facing floating damage number and the server's actual HP subtraction
can genuinely disagree by up to 30% for a hit landing during Garen's W.

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Health subtraction: floor at 0, no minimum-damage rule | `AttackableUnit.cs:585` | `step.py:512` `hp = jnp.maximum(state.hp - dealt, jnp.zeros_like(state.hp))` | EXACT | Same clamp, same absence of a floor-of-1 rule. |
| `IsDead`/`_death` set on the health-crossing-zero transition, `Die()` deferred to the unit's own next `Update` pass (not called from `TakeDamage`) | `AttackableUnit.cs:589-599` (set flag + `DeathData`); actual `Die(data)` call is in `Update` (`:266-268`), confirmed a **separate** dispatch, not synchronous with damage | `step.py:512-...` computes `alive = state.alive & (hp>0)`, `died = state.alive & ~alive` in the same tick damage is applied (no deferred-to-next-tick dispatch modelled) | APPROX (structural, likely harmless at this project's granularity) | The server's split (flag this tick, finalize — gold/XP/removal — on this same unit's own `Update`, which for the *victim* already happened earlier in `ObjectManager`'s iteration this same global tick in the common case) versus the sim's single-tick fold is a tick-order/architecture question the coordinator's brief explicitly scopes to other agents (`collision.py`/movement/tick-order docs already exist per `docs/TICK_PARITY_AUDIT.md`); not re-litigated here beyond flagging it exists. |
| **W's `PreTakeDamage` damage reduction is discarded before it reaches `CurrentHealth`** | `AttackableUnit.cs:551,558,585,615` (see full trace above) | `spells.py:415` (`W_DAMAGE_MULT=0.7`), applied as a real multiplier at `step.py:513` (`dealt = (...) * bs.damage_multiplier`) | **WRONG (BUG-COMPAT required)** | Confirmed twice, via fresh `grep -n` line numbers both times, that no other consumer in `TakeDamage` re-reads `damageData.PostMitigationDamage` after `OnPreTakeDamage` fires. This is the audit's top finding — see §0.1. To match the server exactly, `damage_multiplier`'s effect on **actual HP loss** must be removed (fixed at 1.0); if the project's reward/observation surface ever reports "damage dealt" as a display number rather than true HP delta, the 0.7x could legitimately stay *there* (matching `NotifyUnitApplyDamage`'s client-facing behavior) — a genuine design fork the human should decide. |
| Lifesteal/spell vamp: heals attacker by `healRatio * postMitigationDamage` (also the pre-W-reduction value, so lifesteal against a Garen-W-shielded target over-heals relative to the *displayed* damage, matching the same stale-variable bug) | `AttackableUnit.cs:610-618` | not modelled — no lifesteal/spellvamp mechanic found anywhere in `lanerl_jax` | N/A | Confirmed unreachable: the fixed `BuildPath` (`LanerlConfig.cs:327`, IDs 1054/2003/2003/1001/3047/3134/3068/3035) contains no lifesteal- or spellvamp-granting item, and Garen's kit grants neither innately. Correctly absent, not a gap. |
| `CanTakeDamage` immunity gates (`Invulnerable`/`PhysicalImmune`/`MagicImmune`, via `Status` flags — note: **not** the same-named `Stats.IsInvulnerable`/`IsPhysicalImmune`/`IsMagicImmune` fields, which this method never reads) | `AttackableUnit.cs:422-455` | not modelled | N/A | No invulnerability/immunity source anywhere in this project's fixed kit/item/summoner-spell set (Flash/Teleport only). Flagging the two-parallel-fields oddity in the server itself for completeness; not actionable. |

### 3.3 Shields

No shield mechanic (`Stat`-based absorb-then-decrement) was found anywhere in
`AttackableUnit.cs`'s damage path, and Garen's kit has none (W is a
percentage-damage-reduction event listener, not a shield). **N/A** — not
scored as a row for lack of any server-side or our-side mechanism to compare.

---

## 4. Buffs (generic engine)

Full findings from the dedicated sub-audit (independently re-verified control
flow against `Buff.cs`, `AttackableUnit.cs`'s `UpdateBuffs`, and
`BuffScriptMetaData.cs`):

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| `UpdateBuffs` runs first in `AttackableUnit.Update`, before `Stats.Update`/`Move`/`Die` | `AttackableUnit.cs:237-267` | `step.py:251` (`bs = step_buffs(...)` called before `step_regen`/movement) | EXACT | Re-confirmed by this session directly (§2.6), not solely by the sub-audit. |
| `UpdateBuffs` snapshots the buff list before iterating (`tempBuffs = new List<Buff>(BuffList)`), so a buff added *by another buff's `OnUpdate`/`OnDeactivate` this same tick* does not get its own `Update()` until next tick | `AttackableUnit.cs:809-828` | N/A — fixed per-effect slots, no dynamic list | N/A (architecture mismatch) | E.g. `GarenPassiveCooldown.OnDeactivate` → `AddBuff("GarenPassiveHeal", ...)` handoff: whether the new buff's own first tick of healing is delayed by one frame on the server is a real timing question our fixed-slot design doesn't have a "was it in the snapshot" analogue for. Not scored further; low amplitude (one 16-33ms frame) against Garen's passive's 1000ms internal tick. |
| `BuffAddType.REPLACE_EXISTING` re-runs `OnActivate` in full (recomputes whatever the buff's setup logic computes, not just a duration refresh) | `AttackableUnit.cs:1086-1109` | Garen's `GarenPassive`, `GarenPassiveHeal`, `GarenPassiveCooldown`, `GarenW`, `GarenWPassive` all declare `BuffAddType.REPLACE_EXISTING` (confirmed by direct read of each file's `BuffScriptMetaData`) | UNVERIFIED (contract confirmed; whether every `lanerl_jax` re-cast of W recomputes vs refreshes was not independently re-checked against `spells.py`'s `cast_w`) | Practically low-risk: W's `OnActivate` computes nothing level/state-dependent beyond the fixed `Tenacity+30%`/`0.7x` constants, so "recompute" and "refresh" are the same numbers regardless. |
| `BuffAddType.RENEW_EXISTING` (the unset default) does **not** re-run `OnActivate` — only resets the elapsed timer | `AttackableUnit.cs:1110-1123`; `BuffScriptMetaData.cs:8`'s own default | `GarenQ`(buff)/`GarenQHaste`/`GarenE` all rely on this default (none declare `BuffAddType` explicitly, confirmed) | N/A | Inert for this kit: Q's slot is sealed shut while its window is open (`Q.cs:84`, no recast possible), and E/QHaste are never recast mid-duration by any observed Garen play pattern in this 1v1. Recorded because "RENEW_EXISTING never re-runs setup" is a real, easy-to-miss engine contract the next champion/item port must respect. |
| `StatsModifier` apply is **manual** (each buff script must call `unit.AddStatModifier(...)` itself in `OnActivate`); removal on deactivate is **automatic** (`Buff.DeactivateBuff` unconditionally calls `RemoveStatModifier` on whatever `BuffScript.StatsModifier` holds, even if `OnActivate` never added it) | `Buff.cs:112-141` | No generic `Stats`-object analogue exists on our side (confirmed by the same sub-audit); each Garen effect's stat-like contribution is a hand-computed per-tick term | N/A (architecture) | Real, easy-to-miss server contract for any future port: a buff script that forgets to call `AddStatModifier` still gets a silent no-op "remove" on deactivation rather than an error. No consequence for Garen's already-audited effects (W/W-passive both call `AddStatModifier` correctly in `OnActivate`, confirmed by direct read in §9). |
| No generic buff engine on our side at all — one fixed array slot per named effect (`Q_BUFF_SLOT`, `W_BUFF_SLOT`, ...), a single bespoke `step_buffs` function | (n/a — architecture comparison) | `spells.py`'s `Slot`/`BuffId` enums and `step_buffs` | N/A (design decision for the human) | Reasonable given dynamic per-unit buff lists don't vectorize well under `vmap`, but means every future champion/item requires hand-written slot logic; flagged once here rather than per-row. |

---

## 5. Autoattack engine & the cast-time-vs-hit-time bug

*(This section is being completed pending the dedicated autoattack sub-audit,
covering `ObjAIBase.cs`'s windup/cooldown clock, `CancelAutoAttack`,
`HasMadeInitialAttack`, and the melee-vs-missile dispatch rule in full. The
cast-time-vs-hit-time finding below was independently confirmed by the
parent session directly against `Spell.cs`/`SpellMissile.cs`/`ObjAIBase.cs`,
not the sub-audit, and stands on its own.)*

### 5.1 melee autoattack: damage computed synchronously at windup completion

`Spell.FinishCasting` (`Spell.cs:982-1013`, re-read fresh):

```csharp
public void FinishCasting()
{
    ...
    if (CastInfo.IsAutoAttack || CastInfo.UseAttackCastTime)
    {
        CastInfo.Owner.HasAutoAttacked = true;
        if (!CastInfo.Owner.HasMadeInitialAttack) CastInfo.Owner.HasMadeInitialAttack = true;
        if (!CastInfo.Owner.IsMelee)
        {
            if (HasEmptyScript) CreateSpellMissile(new MissileParameters { Type = MissileType.Target });
        }
        else
        {
            ApplyEffects(CastInfo.Targets[0].Unit);
            CastInfo.Owner.AutoAttackHit(CastInfo.Targets[0].Unit);     // <-- line ~1010: melee damage dealt HERE, synchronously
        }
        State = SpellState.STATE_READY;
    }
    ...
}
```

`FinishCasting` fires when the windup timer (`CurrentCastTime`) reaches zero
— i.e. at **hit-resolution time**, not at the moment the attack was ordered.
For a melee unit this is a single synchronous call: `AutoAttackHit`
(`ObjAIBase.cs:269-297`) reads `Stats.AttackDamage.Total` **fresh, at this
exact instant** — there is no cast-time snapshot anywhere in this path for a
melee swing, so "hit-time attribution" is simply what a melee autoattack
*is* on this server; there is no daylight between windup-completion and
damage-computation to be off by.

### 5.2 ranged autoattack: damage payload computed at missile *impact*, not launch

For a non-melee unit, `FinishCasting` instead calls `CreateSpellMissile`.
`SpellMissile.CheckFlagsForUnit` (`Missile/SpellMissile.cs:184-198`, called
from collision handling on arrival):

```csharp
if (SpellOrigin.CastInfo.IsAutoAttack) {
    SpellOrigin.ApplyEffects(TargetUnit, this);
    if (CastInfo.Owner is ObjAIBase ai && SpellOrigin.CastInfo.IsAutoAttack)
        ai.AutoAttackHit(TargetUnit);          // <-- reads Stats.AttackDamage.Total LIVE, AT IMPACT
}
```

So the server's own ranged-autoattack damage is computed **at missile
landing**, using whatever `Stats.AttackDamage.Total` is *at that later
moment* — not a value captured when the missile was fired.

`lanerl_jax/sim/missiles.py:107-190` (`step_missiles`): `raw_damage` (the
firing unit's `Stats.AttackDamage.Total`-equivalent, `raw_ad` computed once
per tick in `step.py:456`) is written into `m_damage` **at launch** and never
updated; only the **mitigation** half (`post_mitigation_damage(m_damage,
armor[t], ...)`) is recomputed live against the target's *current* armor at
landing — the module's own docstring is explicit that this was a deliberate
choice for the mitigation half, but does not address the attacker's-AD half.

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Melee autoattack damage: computed synchronously at windup completion (no cast-time snapshot exists to be wrong) | `Spell.cs:1004-1011`, `ObjAIBase.cs:269-286` | `step.py:456-468` computes `raw_ad`/`aa.damage` fresh every tick and applies it the same tick a melee swing lands (`landed = swings & ~ranged`, `dmg_ij` uses `aa.damage` directly) | EXACT | Both sides compute the swing's damage at the instant of landing, same-tick, no snapshot gap on either side. |
| Ranged autoattack damage: attacker's raw AD is read **live at impact** by the server; **frozen at launch** by our sim | `SpellMissile.cs:190-192` (`AutoAttackHit` re-reads `Stats.AttackDamage.Total` at collision) | `missiles.py:170-190` (`m_damage = put(m_damage, raw_damage)` at launch; never re-gathered at `arrives`) | **APPROX (WRONG in the strict sense, near-zero practical impact for minions, small/rare for turrets)** | This is the precise, source-verified answer to the brief's "known open bug" question: the bug is real, but it is entirely in the **ranged** path, and it is inert for every minion (their AD is a flat, non-leveling profile constant — `ad_per_level=0` for every non-champion row, confirmed in `profiles.py`) and matters for **turrets only**, whose AD ramps by a flat +4 every 60s (§ combat.py's `outer_turret_ramps`/`other_turret_ramps`, independently re-verified against `Maps/Map1/LevelScriptObjects.cs:159-256` this session — see below). A turret missile's flight time (`range/missile_speed`, e.g. an outer turret's ≈700/1200 ≈ 0.58s) can straddle a ramp tick only in the fraction-of-a-second window around each 60s boundary — rare, and worth ≤4 AD (≈2-3% of a ~150-180 AD hit) when it happens. **What would make this EXACT**: carry the firing unit's index with the missile (`m_source` already does this) and gather `ad_now[m_source]` at `arrives` time instead of reading the frozen `m_damage`; this is a small, well-scoped, JAX-friendly change (no new state, `m_source` is already stored) — a good candidate fix, not attempted here per "ledger first". |
| Turret AD/armor/MR ramp schedule (both outer and inner/inhib/nexus) | `Maps/Map1/LevelScriptObjects.cs:159-256` (re-read and independently verified fresh this session: `outerTurretTimeCheck` starts 30000ms, `timeCheck` (other tiers) starts 480000ms, both step 60000ms, caps 7 and 30 applications respectively; outer tier's `OuterTurretStatsModifier` has **no** `Armor.FlatBonus` — "Outer turrets dont get armor", `:163` comment in the server itself; `FOUNTAIN_TURRET` explicitly `continue`d out of `UpdateTowerStats` and never looked up by `UpdateOuterTurretStats`'s per-lane `Find`) | `combat.py:76-176` (`outer_turret_ramps`, `other_turret_ramps`, `outer_turret_attack_damage`, `other_turret_attack_damage`, `other_turret_armor`) | EXACT | Fully independently re-derived from `LevelScriptObjects.cs` this session (not merely re-reading `combat.py`'s own docstring, which already claimed this correctly) — every constant (30000/480000/60000/7/30/+4 AD/+1 armor+MR, outer-has-no-armor-bonus, fountain-excluded-from-both-schedules) matches exactly. `AddStatModifier` is called with the **same static, non-reset `StatsModifier` object** on every firing, and `Stat.ApplyStatModifier`'s `FlatBonus +=` (§2.4) confirms each firing adds another +4/+1 on top of the last — matching `combat.py`'s "N ramp applications × per-ramp constant" model exactly, not a one-shot re-application. |

---

## 6. Missiles & sectors (generic geometry engine)

From the dedicated sub-audit (`SpellMissile.cs`, `SpellLineMissile.cs`,
`SpellCircleMissile.cs`, `SpellChainMissile.cs`, `SpellSector*.cs` vs
`missiles.py`):

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Homing missile position update (`MissileType.Target`) | `Missile/SpellMissile.cs:122-166` | `missiles.py:139-146` | EXACT | Traced the clamp-to-remaining-distance boundary case by hand; identical result whether the comparison is `>` or `>=` at the exact-arrival instant. |
| Hit detection (arrival test) | `SpellMissile.cs:164-176` (epsilon pre-check, then an exact position-equality re-check before the hit fires) | `missiles.py:141` (`dist <= step`) | EXACT (non-obvious) | The two-stage server check collapses to exactly `dist<=step` once traced fully: a hit only registers when the clamped move covers the full remaining distance in one tick, not merely "within 5 units". |
| Missile dropped, no damage, if target dies/untargetable mid-flight | `SpellMissile.cs:69-80` | `missiles.py:135-136` | EXACT | Same guard order on both sides. |
| No max-range/lifetime timeout on a homing basic-attack missile | Confirmed: base `SpellMissile` has no lifetime field | `missiles.py` has none either | EXACT | For `MissileType.Target`, the only type basic attacks use. |
| Line/circle/chain missile geometry, cone/polygon sector hit-tests | `Missile/SpellLineMissile.cs`, `SpellCircleMissile.cs`, `SpellChainMissile.cs`, `Sector/SpellSectorCone.cs`, `Sector/SpellSectorPolygon.cs` (all read in full) | not modelled anywhere in `lanerl_jax` | MISSING (correctly N/A for current scope) | Confirmed unreachable: no Garen ability, minion, or Map1 turret in this project's kit ever instantiates any of these (Garen's kit uses only auto-attack `MissileType.Target` missiles and a self-centered tick-damage buff for E, not a real sector object). Flagged as MISSING rather than a silent N/A because the *first* future champion/item/skillshot need finds zero groundwork here. |
| Basic-attack missile speed source (`SpellData.MissileSpeed`, per-basic-attack-spell field, not per-`CharData`) | `Spell.cs:1149-1157`, confirmed absent from `CharData.cs`/`BasicAttackInfo.cs` | `missiles.py:100-105`, `data/patch.py`'s `UnitStats.missile_speed` | EXACT (mechanism); values not re-verified in this pass | Sourcing mechanism confirmed correct; the specific numbers (caster 650, cannon/turret 1200, melee 0/unread) were not independently re-read against JSON in this pass — see `data/patch.py` if a numeric re-check is wanted. |
| Sector engine tick cadence / farthest-first hit ordering (`Sector/SpellSector.cs:75-101,196-216`) | present in the server as a generic mechanism | No generic reusable sector engine on our side; Garen's E (the only sector-shaped ability in this kit) is hand-coded directly in `spells.py`, not via a shared sector module (see §9.3, confirmed EXACT for E specifically) | N/A (architecture) | Only E needs anything sector-shaped in this kit, and E is separately audited as its own row (§9.3) rather than through a generic engine that doesn't exist on our side. |

---

## 7. Autoattack windup/cooldown clock

*(Pending sub-audit results — to be merged in on completion; see the
placeholder note at the top of §5.)*

---

## 8. XP and gold on death

From the dedicated sub-audit (`ChampionDeathHandler.cs`, `Champion.cs`,
`LaneTurret.cs`, `AttackableUnit.cs`'s `Die`, `MinionEXPMods.cs`,
`ExpCurve.json`), with one number independently corrected against the
primary JSON source by the parent session (the sub-audit mis-cited
Level18's threshold; corrected below) and the XP-curve-transcription
question the sub-audit flagged UNVERIFIED independently resolved (also
below).

**Dead code warning, load-bearing for this whole section:**
`ChampionDeathHandler.ProcessKill` (`Handlers/ChampionDeathHandler.cs:67`)
has **zero call sites** anywhere except its own file — only
`ChampionDeathHandler.Init` is ever invoked (`Game.cs:164`). The live
champion-death path is `Champion.Die` (`Champion.cs:392`), which does **not**
call `base.Die`, fully replacing `AttackableUnit.Die` for champions. Any
future work must use `Champion.cs:392-505` as the champion-kill reference,
not `ChampionDeathHandler.cs`.

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Minion gold on death (killer-only, champion killer required) | `Champion.cs:359-390` (`OnKill`), reached only via `AttackableUnit.cs:670`'s `champion.OnKill(data)` when `data.Killer is Champion` | `sim/rewards.py:93-100` (`death_rewards`) | EXACT | Guard order matches: gold only when killer resolved, is a champion, victim is a `Minion`, amount>0. |
| Minion XP on death (**proximity-shared among all enemy champions in range, independent of who last-hit**) | `AttackableUnit.cs:634,650-658`: `champs = GetChampionsInRangeFromTeam(Position, ExpRadius2, EnemyTeam, alive:true); expPerChamp = ExpGivenOnDeath.Total/champs.Count`, `ExpRadius2=1600.0` (`Content/GlobalData/ObjAIBaseVariables.cs:16`, re-verified this session) | `sim/rewards.py:104-113` | EXACT | Radius, team filter, alive filter, and equal-split arithmetic all reproduced; correctly NOT gated on last-hit, matching the server not checking `Killer` at all in this branch. |
| Champion-as-victim wrongly eligible for the minion-style radius-share XP path (missing a `victim_is_minion` gate that the gold row one line above already has) | `Champion.cs:392-505`'s `Die` override never reaches `AttackableUnit.cs`'s radius-share branch for a champion death at all | `sim/rewards.py:104-109`'s `eligible` expression has no victim-is-minion gate | APPROX (currently dormant, latent WRONG) | Dormant only because `patch.champion.exp_given_on_death==0` (Garen.json has no `ExpGivenOnDeath` key, confirmed default 0.0 via `CharData.cs`). One-line, JAX-friendly fix: add the same `victim_is_minion` gate the gold path already has. |
| **Champion kill gold** (base + kill-streak multiplier `(7/6)^(streak-1)` capped, death-spree discount, first-blood bonus) | `Champion.cs:392,421-441` (the *live* path — not `ChampionDeathHandler.cs`, which is dead code, see above) | not found anywhere in `lanerl_jax` | **MISSING** | A champion kill in this 1v1 currently pays the killer nothing. Not dormant by luck of zero data (unlike the row above) — this is a real, reachable gap the moment either Garen kills the other. |
| **Champion kill XP** (level-difference-scaled: `EXP=ExpCurve[victimLevel-1]*0.55`, adjusted ±`min(0.08*|Δlevel|, 0.15)` by level difference) | `Champion.cs:444-451`; constants from `Maps/Map1/ExpCurve.json`'s `ExpGrantedOnDeath` block (`BaseExpMultiple=0.55`, `LevelDifferenceExpMultiple=0.08`, `MinimumExpMultiple=0.15`, re-verified directly against the JSON this session, not only the sub-audit's citation) | not found anywhere in `lanerl_jax` | **MISSING** | Champion kills grant 0 XP in the sim. The generic minion-style radius-share path is not a substitute (no level-difference scaling, splits by proximity rather than crediting the killer). |
| **Turret destruction gold/XP** (`LocalGoldGivenOnDeath` split among enemy champions within `Range.Total*1.5`, **plus** `GlobalGoldGivenOnDeath` and `GlobalExpGivenOnDeath` to *every* enemy champion unconditionally) | `LaneTurret.cs:37-86` | `profiles.py:149-150` reads `gold_given_on_death`/`exp_given_on_death` — confirmed by direct read of `OrderTurretNormal.json` (Map1's real outer-turret model) that these are **0**; the real values live in `LocalGoldGivenOnDeath:150`/`GlobalGoldGivenOnDeath:100` (re-verified against the JSON this session, matching the sub-audit's citation exactly), which nothing in `lanerl_jax` reads at all | **WRONG (reads always-zero fields) + MISSING (no distribution mechanic)** | A destroyed outer turret currently pays 0 gold and 0 XP to either side; the server pays 250g total (150 local-split + 100 global-flat) plus unconditional global XP. Highest-value single economic event in a 1v1 top lane where towers do fall — see priority list. |
| Minion gold/XP values, per type/team | `Content/LeagueSandbox-Default/Stats/SRU_{Order,Chaos}Minion{Melee,Ranged,Siege,Super}/*.json`: melee 20g/77xp, caster 10g/51xp, cannon **35g (blue) / 30g (red)** /94xp, super 150g/500xp | `data/patch.py`, pinned by `sim/tests/test_lane.py:66-67` | EXACT | Independently re-read the blue/red cannon-gold asymmetry directly from both JSON files this session (not solely trusting either the sub-audit or the test) — `SRU_OrderMinionSiege.json`'s `GoldGivenOnDeath=35`, `SRU_ChaosMinionSiege.json`'s `=30`, both `ExpGivenOnDeath=94`. |
| `MinionEXPMods` (level-based XP scaling for minions) | `Content/GlobalData/MinionEXPMods.cs` (loaded from config, `GlobalData.cs:88-92`) | correctly absent | N/A | Confirmed **zero consumers** of the loaded `MinionEXPMods` fields anywhere in `GameServerLib` — a configured-but-dead mechanic in the server itself. Nothing to port. |
| **XP-to-level threshold table** (18 entries, Level2=280 … **Level18=18360**) | `Content/LeagueSandbox-Default/Maps/Map1/ExpCurve.json` | `data/patch.py:275-276,388` (`xp_for_level`, `exp_curve` dict) | EXACT | The sub-audit flagged this UNVERIFIED and separately mis-cited "Level18=16480" (that is actually Level17's value — corrected here against a fresh read of the JSON: Level17=16480, **Level18=18360**). Independently resolved: `data/patch.py:388` builds `exp_curve` by **parsing the JSON file directly at load time** (`{int(k.replace("Level","")): float(v) for k,v in exp.items()}`), not by hand-transcription, so there is no copy-error surface at all; confirmed by reading the loader function in full. `parity/tests/test_patch.py:138,140` pin `xp_for_level(2)==280.0` and `xp_for_level(18)==18360.0` against the corrected value, both passing. |
| `LevelUp` trigger point (`Champion.AddExperience`'s `while(Experience>=ExpCurve[Level-1]) LevelUp();`, same tick, sequential, can level up more than once per XP grant) | `Champion.cs:319-332` | `rewards.py`'s `level_for_xp` recomputes level in closed form from total XP rather than iterating | EXACT (by construction) | Valid closed-form of the sequential loop given `Stats.LevelUp`'s increments are constant and additive across levels — independently confirmed via the `Stat.ApplyStatModifier`'s `+=` semantics (§2), not re-derived by the sub-audit alone. |
| Ambient ("gold per 10") passive income | `Champion.cs:224-238`, `AmbientGoldAmount=9.5f`/`AmbientGoldInterval=5.0f` (`ChampionVariables.cs:8,12`), gated `GameTime>=AmbientGoldDelay=90.0f` (`ObjAIBaseVariables.cs:32`) | `rewards.py:56-73,128-141` (`ambient_gold`) | APPROX (measured cadence, documented as such, root cause of the sub-second mismatch not resolved) | Not an "on-death" mechanic but adjacent; flagged by the sub-audit as measured-not-derived (server constants don't literally reproduce the observed ~517ms cadence) and left as an open question — not re-resolved by the parent session either (would need `ObjectManager`/`Game`'s tick-diff plumbing, outside this subsystem's file list). Rate is unaffected (1.9g/s), only sub-tick granularity. |
| Ambient XP tick | `Champion.cs:246-253`, `AmbientXPAmount=0.0f` (`ChampionVariables.cs:28`) | correctly not modelled | EXACT (N/A mechanic correctly omitted) | Confirmed the default is 0; no override found in any file read this session. |

---

## 9. Garen's kit

### 9.1 Passive — Perseverance

Already covered in depth at §2.6 (regen mechanism) and §2.7 (the UnitTag
bug). One more numeric check, re-derived from `Buffs/Garen/GarenPassiveHeal.cs:29-30,66-72`
and `Characters/Garen/CharScriptGaren.cs:83-92`:

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Heal-percent brackets by level (`{0.004, 0.008, 0.02}` at L1/11/16) and out-of-combat cooldown brackets (`{9,6,4}`s, same breakpoints) | `GarenPassiveHeal.cs:29`, `CharScriptGaren.cs:18,83-92` (`GetCorrectLevelIndex`: `>=16→2, >=11→1, else 0`) | `regen.py:74-90` (`GAREN_HEAL_PCT`, `OUT_OF_COMBAT_MS`, `garen_heal_bracket`) | EXACT | Bracket boundaries (`>=11`, `>=16`) and both constant tuples re-verified fresh against both C# files. |
| Passive heal is a **separate `TakeHeal` call on its own 1000ms internal accumulator**, not a `HealthRegeneration` stat modifier — does not share `Stats.Update`'s 500ms clock or its `hp>0`/`hp<max` guard | `GarenPassiveHeal.cs:66-72` (`OnUpdate`: `healingTimer+=diff; if(healingTimer>1000f){...TakeHeal...}`) | `regen.py:117-131` (separate `heal_timer` accumulator, 1000ms, gated on `eligible` not on the base-regen guard) | EXACT | Confirmed two genuinely independent clocks on both sides, not a shared one. |
| Interruption event: `AddBuff("GarenPassiveCooldown", GetCooldownForLevel(level), ...)` fires on the **first** qualifying hit, and a **second** qualifying hit during the cooldown window resets it via `GarenPassiveCooldown.OnTakeDamage`'s `resetting=true` re-add | `GarenPassiveHeal.cs:52-60`, `GarenPassiveCooldown.cs:39-47,52-60` | `regen.py:117` (`ms_since_damaged` reset to 0 on any qualifying hit, re-armed each time) | EXACT (mechanism) | The two-buff handoff state machine (Heal → Cooldown → Heal) collapses correctly to "a single timer that resets on each qualifying hit and must stay idle for the bracket's threshold" — verified by tracing all three buff scripts' `OnActivate`/`OnDeactivate`/`OnTakeDamage` methods in full. |

### 9.2 Q — Decisive Strike

Re-verified directly against `Characters/Garen/Q.cs` (full file) and
`Buffs/Garen/GarenQ.cs`/`GarenQHaste.cs` (full files) this session.

| quantity | value | citation |
|---|---|---|
| Empowerment window duration | 4.5s, flat, all ranks | `Q.cs:76` |
| Haste duration | `1.5 + 0.75*(rank-1)` s | `Q.cs:77` |
| Haste magnitude | **+35% move speed, flat across all ranks** (not rank-scaled) | `GarenQHaste.cs:32` |
| Empowered-swing damage | `30 + 25*(rank-1) + 1.4*AD`, physical, via a direct `TakeDamage` call (bypasses `AutoAttackHit`, so **cannot crit**) | `Q.cs:142-144` |
| Silence duration | `1.5 + 0.25*(rank-1)` s | `Q.cs:139` |
| Empowered swing's own cast time | **fixed 0.25s**, overriding the normal windup formula (`Script.ScriptMetadata.CastTime=0.25f` takes the `Spell.cs` branch that prefers script-declared cast time over the computed attack windup) | `Q.cs:88-92`, cross-checked against `Spell.cs`'s cast-time-resolution branch order |
| Cooldown | 8.0s flat, all ranks (`GarenQ.json`'s `Cooldown1`-`5` all `"8.0000"`, confirmed) but **starts only when the empowerment window ends** (`GarenQ.cs:98`, `SetCooldown(8)` in `OnDeactivate`), not at cast — the engine's default cast-time cooldown-set is explicitly zeroed by `Q.cs:85`'s `spell.SetCooldown(0)` immediately after cast | `Content/LeagueSandbox-Default/Spells/GarenQ/GarenQ.json`, `Characters/Garen/Q.cs:84-85`, `Buffs/Garen/GarenQ.cs:98` |
| Mechanism | `OnSpellPreCast` adds two buffs (window + haste) and nothing else. `GarenQ`(buff)`.OnActivate` does the real work: `owner.CancelAutoAttack(true)` (reset=true, fullCancel=**false** — cancels the in-flight swing and zeroes the attack-cooldown clock, but does *not* clear `IsAttacking`/`HasMadeInitialAttack`), then `owner.SkipNextAutoAttack()` (sets `_skipNextAutoAttack`, consumed once in `ObjAIBase.cs:1259-1267`), then registers an `OnPreAttack` listener that swaps `AutoAttackSpell` to `"GarenQAttack"` the next time an attack is ordered/refreshed. `GarenQAttack` is a **separate spell object** with its own 0.25s cast and its own `OnSpellPostCast` (damage+silence). On `OnDeactivate`, the original `AutoAttackSpell` is restored and the real 8s cooldown starts. | `Q.cs`, `Buffs/Garen/GarenQ.cs` (both full files) |

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Window/haste/cooldown bookkeeping, buff durations, recast-lock (`SealSpellSlot`) | `Q.cs:76-85`, `GarenQ.cs:47-104` | `spells.py`'s `cast_q`, `Q_BUFF_SLOT` handling (durations/cooldown constants re-verified directly against `Q.cs` this session, matching what `spells.py`'s own docstring already claimed) | EXACT | Independently re-verified every numeric constant in the table above against the C# source directly, not merely against `spells.py`'s docstring. |
| **Empowered-swing damage + silence delivery** | `GarenQAttack.OnSpellPostCast` (`Q.cs:112-127`) — fires whenever the (possibly re-targeted) next attack actually lands, 0-4.5s after cast | **not implemented** — `spells.py`'s own docstring states plainly: "casting Q in the sim opens the window, blocks a recast, and starts the cooldown on schedule, but the empowered swing itself deals ordinary auto-attack damage" | **MISSING** (confirmed, matches the brief exactly) | This needs a hook in `autoattack.py`/`step.py` that (a) recognizes a landing swing should be Q-empowered, (b) applies the `30+25*(rank-1)+1.4*AD` damage instead of ordinary AD, bypassing crit, (c) applies the silence, and (d) signals the buff to deactivate early (server: the empowered swing's `OnSpellPostCast` calls `OnSpellEnd()`, which deactivates `GarenQ`, which is what actually starts the real cooldown — so landing early shortens the effective lockout, not merely opening a window that always runs its full course). None of (a)-(d) exist on our side today. |
| Early-landing case (window closes as soon as the empowered swing connects, well under 4.5s) | Same source as above | Explicitly modelled as unavailable — `spells.py`'s docstring: "Q is on cooldown for a full 4.5s window plus 8s here, which is a strict upper bound... not the real number" | APPROX (self-documented over-estimate, direct consequence of the MISSING row above) | Not a separate defect, the same missing hook; noted because it means Q's effective cooldown in the sim is currently *longer* than real, not merely "sometimes wrong". |

### 9.3 E — Judgment

Re-verified directly against `Characters/Garen/E.cs` and
`Buffs/Garen/GarenE.cs` (both full files, including the in-place bug-fix
comments describing two now-fixed server bugs: a self-comparison in
`OrderBy` that made red-side Garen's E always deal zero damage, and an
unseeded per-tick `Random()` with an off-by-one `<=` crit roll).

| quantity | value | citation |
|---|---|---|
| Duration | 3.0s flat | `E.cs:23` |
| Tick-damage formula (snapshotted **once, at cast**, not recomputed per tick) | `10 + 12.5*(rank-1) + AD*(0.35+0.05*(rank-1))`, physical | `Buffs/Garen/GarenE.cs:34-36` |
| Tick cadence | every 500ms, buff-internal accumulator | `GarenE.cs:64-65` |
| Radius / center | 330 units, re-centered on the (moving) caster every tick | `GarenE.cs:80` |
| Exclusions | turrets and buildings immune; minions take 0.75x | `GarenE.cs:95,101-104` |
| Crit | possible, seeded RNG (`new Random(0x6A3E17)`), `<` comparison (not `<=`) against `CriticalChance.Total*100` | `GarenE.cs:26,86-91` |
| Damage source/type | physical, `DAMAGE_SOURCE_SPELL` (not `_ATTACK` — so it would proc spell vamp, not lifesteal, if either existed in this build) | `GarenE.cs:107` |
| Caster status during E | `CanAttack=false`, `Ghosted=true` (suppresses auto-attacks, passes through unit collision) | `GarenE.cs:39-40` |
| Cooldown | 13/12/11/10/9s by rank, starts when the buff deactivates (`OnDeactivate` sets `spell.SetCooldown(spell.GetCooldown())`), not at cast | `GarenE.cs:52-56`, `Content/.../GarenE/GarenE.json` |

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| All of the above | `Characters/Garen/E.cs`, `Buffs/Garen/GarenE.cs` (full files) | `spells.py:16-49` (docstring) + `cast_e`/`step_buffs`'s E section (`:301-...,605-...`) | EXACT | `spells.py`'s own docstring for E independently states every one of the four "easy to get wrong" details (AD snapshotted at cast, 0.75x on minions, re-centered radius, turret immunity) with correct citations, and additionally correctly models the crit roll as always-false given Garen's crit chance is 0 in this config (flagged in the docstring as "stops being correct the moment a crit source enters the kit" — an honest, bounded APPROX, not silently ignored). Independently re-read the two C# files fresh and found no discrepancy with either the docstring or the (separately re-checked) tick-boundary code at `spells.py:605-621`. |

### 9.4 W — Courage

Covered in depth at §0.1/§3.2 (the damage-reduction bug) and here for the
remaining numeric facts, re-verified against `Characters/Garen/W.cs` and
`Buffs/Garen/GarenW.cs`/`GarenWPassive.cs` (full files).

| quantity | value | citation |
|---|---|---|
| Active window duration | `2 + rank - 1` s (i.e. rank+1: 2/3/4/5/6s) | `W.cs:47` |
| Active window "reduction" (see §3.2: **does not reach actual HP loss on the server**) | `dmg.PostMitigationDamage *= 0.7f`, no source/type filter | `GarenW.cs:49-54` |
| Active window's other, genuinely-effective, effect | `+30% Tenacity` (via a real `AddStatModifier` call, unaffected by the `TakeDamage` bug since Tenacity isn't read through that path) | `GarenW.cs:39-41` |
| Active cooldown | 24/23/22/21/20s by rank, **unmodified engine default, starts at cast** (opposite of Q/E) | `Content/.../GarenW/GarenW.json`, confirmed no `SetCooldown` override anywhere in `W.cs`/`GarenW.cs` |
| Passive | +20% Armor, +20% Magic Resist, permanent, granted **once, the moment W is first ranked** (`OnLevelUpSpell`, registered at `ISpellScript.OnActivate` — i.e. spell-object construction, not the buff's own `OnActivate`) — **not tied to ever casting W** | `W.cs:26-46`, `GarenWPassive.cs:31-37` |

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Active window's real effect is **+30% Tenacity only** (damage reduction is a no-op on HP, cosmetic-only on the floating combat text) | §3.2 | `spells.py:415` `W_DAMAGE_MULT=0.7` applied as a genuine multiplier (`step.py:513`); **Tenacity is not modelled anywhere found in `lanerl_jax`** (no CC-duration mechanic in this subsystem's files — cross-checked, none of Garen's kit nor the enemy's applies CC in this matchup, so Tenacity is plausibly N/A for THIS specific 1v1, but that should be confirmed by whoever owns CC/status effects, not assumed here) | **WRONG** (damage reduction) **+ UNVERIFIED** (Tenacity) | See §0.1 for the full trace. The Tenacity gap is separately noted because if it were ever to matter (e.g. a summoner-spell/future-champion CC), it is currently entirely unmodelled, not merely reduced. |
| Passive granted on rank-up, independent of cast, permanent | `W.cs:26-46` | `spells.py`: `step_buffs` grants `GarenWPassive`-equivalent "the tick `spell_level[...,Slot.W]` first becomes ≥1, independent of `cast_w`" (per its own docstring, re-verified against `W.cs`'s `OnLevelUpSpell` registration pattern) | EXACT | Re-confirmed the server's non-obvious "registered at spell construction, fires once, ever" semantics matches the described independent-of-cast trigger. |
| Passive magnitude (+20%/+20%, applied as `PercentBonus += 0.2` with an offsetting `PercentBaseBonus -= 0.2`, a specific idiom for "add to the global percent bucket without letting the flat-bonus's own percent-base multiplier also apply to it") | `GarenWPassive.cs:34-37` | `spells.py`'s `W_PASSIVE_ARMOR_PCT`/`W_PASSIVE_MR_PCT`, wired into `step.py:280-281` (`armor_eff = armor_now*(1+bs.armor_pct_bonus)`, `magic_resist_eff = P("magic_resist")*(1+bs.mr_pct_bonus)`) | EXACT | Confirmed the base used is `armor_now` (post-turret-ramp), not the static profile column — correctly composes multiplicatively with a turret's own ramp, matching the brief's pre-stated fact `armor_eff = armor_now * (1 + bs.armor_pct_bonus)` exactly. Also confirmed `step.py`'s own docstring at this call site independently re-derives *why* `armor_now` and not `P("armor")` is required (turret ramp composition), matching this session's own finding in §5.2's table rather than merely repeating it. |

### 9.5 R — Demacian Justice

Re-verified directly against `Characters/Garen/R.cs` (full file, 39 lines).

| quantity | value | citation |
|---|---|---|
| Damage formula | `175*rank + percentMissingHP*(MaxHP-CurrentHP)`, `percentMissingHP={0.2857,0.3333,0.4}` by rank | `R.cs:27-29` |
| **Damage type** | **Magical** (`DAMAGE_TYPE_MAGICAL`) — patch-4.20-accurate, not modern-patch physical | `R.cs:33` |
| Damage source | `DAMAGE_SOURCE_SPELL` | `R.cs:33` |
| Targeting restriction | No `is Minion`/`is BaseTurret` branch in the script itself; restricted to enemy champions by `GarenR.json`'s `TextFlags` (`"AffectEnemies \| AffectHeroes"`, no minion/turret/building/neutral/friend flags) — an engine-level `SpellData` targeting rule, not a script one | `Content/.../GarenR/GarenR.json` |
| Cast structure | Target captured in `OnSpellPreCast`, damage dealt synchronously in `OnSpellPostCast` — no windup logic in the script itself (an engine-level cast time from `SpellData` may still exist between order and resolution; not independently timed this session) | `R.cs:19-39` |
| Cooldown | 160/120/80s, unmodified engine default, starts at cast (`R.cs` never calls `SetCooldown`) | `Content/.../GarenR/GarenR.json` |

| mechanic | server (file:line) | ours (file:line) | verdict | evidence |
|---|---|---|---|---|
| Damage formula, type, mitigation stat | `R.cs:27-33` | `spells.py:460-465` (`R_BASE_PER_RANK`, `R_MISSING_HP_FRAC`), `cast_r`/`step_buffs` (`damage_r`, folded into `bs.damage_dealt`); mitigated against `magic_resist` per `step.py:262`'s `magic_resist=P("magic_resist")` argument to `step_buffs` | EXACT | `spells.py`'s own module docstring is **stale** on this spell — it describes R as "fully implementable... not made here" and lists three `step.py` integration hooks as outstanding. Re-reading current `step.py` (`:262,280-281,513`) shows all three are done: `magic_resist` is passed, `magic_resist_eff` is computed, and `damage_multiplier`/`damage_dealt` are both applied. R is **implemented and correctly wired**, contradicting its own docstring — flagged so nobody re-does this work believing the docstring, and so the docstring gets refreshed. |
| R's target is captured at `OnSpellPreCast` (order time) and not re-acquired at resolution | `R.cs:19-24` (`Target = target`, used unchanged in `OnSpellPostCast`) | not independently checked against `cast_r`'s target-capture timing | UNVERIFIED | Minor: in a 1v1 there is only one possible enemy-champion target, so a re-target-mid-cast scenario cannot occur regardless; flagged for completeness only. |

---

## 10. Priority list (WRONG / MISSING rows, ranked by plausible effect on lane behavior and CS)

1. **W's damage reduction is a no-op on the server; our sim implements it as real (§0.1, §3.2, §9.4).** Highest priority. Materially changes how tanky Garen is during his signature defensive cooldown, in every trade the RL policy or an anchor bot ever takes. Fix: gate `damage_multiplier`'s effect out of the actual HP-loss computation in `step.py:513` (keep the constant/plumbing for a possible "displayed damage" surface if one exists, matching `NotifyUnitApplyDamage`'s use of the mutated value — a design decision, see §3.2).
2. **Champion kill gold + XP entirely missing (§8).** A champion kill in this 1v1 (which does happen post-level-6 in an all-in) currently pays the killer nothing and grants no XP — directly corrupts any reward/eval signal that counts kills, and silently caps a policy's incentive to all-in even when it's correct to.
3. **Turret destruction gold/XP entirely missing, reading always-zero fields (§8).** A 250g/some-XP event with no in-sim equivalent; large relative to per-wave CS income, and directly adjacent to the CS-based reward this project cares about.
4. **Attack-speed-per-level and HP-regen-per-level both unmodelled (§2.3, §2.6 table).** Same class of bug as the AD-per-level fix that motivated this audit, found by the same method (read `Stats.LevelUp` line by line, check every field it touches, not just the one already known to be wrong). Attack speed compounds into every DPS/last-hit calculation at every level past 1; HP regen is smaller in magnitude but same mechanism.
5. **Cannon/super-minion Garen-passive-interruption bug, missing the level-≥11 exemption and mishandling super minions entirely (§2.7).** Changes exactly when Garen can heal through a siege/super push, which is precisely the "close to a tower, low HP, can I keep trading" decision this project's reward signal is built around.
6. **Q's empowered-attack damage+silence still not wired (§9.2), confirmed matching the brief.** Currently a purely negative-value spell in the sim (opens a window, starts an 8s cooldown, delivers nothing) — an RL policy trained against the current sim can only learn to never press Q, which is not the real incentive landscape.
7. **Missile damage snapshot timing for turrets (§5.2).** Real but low-frequency/low-magnitude (≤4 AD, only when a shot's brief flight straddles a ramp tick); listed last because the fix is cheap and well-scoped (gather `ad_now[m_source]` at landing) whenever someone is in `missiles.py` for other reasons.
8. **Champion MR growth per level missing (§2 table).** Same mechanism as #4; lower priority because the only magic-damage path in this matchup is Garen's own R, so the effect is confined to how much R hurts the target at higher levels, not general trading.

---

## 11. Not determinable from source in this pass

- The autoattack windup/cooldown-clock section (§5/§7) was still pending a
  dedicated sub-audit's completion at the time this document was written;
  the cast-time-vs-hit-time finding (§5.1-5.2) was independently confirmed
  by direct reading and stands regardless, but `HasMadeInitialAttack`'s full
  semantics and `CancelAutoAttack`'s complete set of call sites were not
  independently re-verified by the parent session.
- The root cause of the ambient-gold timer's sub-second granularity mismatch
  (§8) — would require `ObjectManager`/`Game`'s tick-diff plumbing, outside
  this subsystem's file list.
- Whether `Stats.LevelUp`'s raw-`BaseValue`-delta HP/mana credit (§2 table)
  is actually exercised anywhere in `lanerl_jax`'s level-up code path, or
  whether that code was simply not located this session.
- Basic-attack missile speed numeric values (caster 650/cannon+turret
  1200/melee unread) were not independently re-read against Content JSON in
  this pass — only the sourcing *mechanism* was re-verified.
