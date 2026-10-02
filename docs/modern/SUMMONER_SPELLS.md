# SUMMONER_SPELLS.md — Summoner spell spec for the modern JAX simulator (patch 26.19)

**Scope.** These are the summoner spells selectable on normal PC Summoner's Rift (`CLASSIC`, map
11) at patch **26.19**, client build **16.19.8230722**:

- Flash
- Teleport / Unleashed Teleport, including the top-lane role-quest interaction
- Ignite
- Exhaust
- Barrier
- Heal
- Ghost
- Cleanse
- Smite (jungle; listed and **DEFERRED**)
- Hexflash, the Hextech Flashtraption replacement for Flash

The spec also covers shared rules: summoner spell haste, start-of-game cooldown, casting under CC,
and rune/item interactions.

**Excluded:**

- Clarity (13) and Mark/Dash (32/39): ARAM/URF only. CLIENT `mGameModes` has no `CLASSIC`.
- Jade, Arena (`CHERRY`), Poro King and Ultimate Spellbook variants.

**Patch pin:** 26.19 / 16.19.8230722. **Retrieval:** 2026-10-01.

**Confidence tags** are as in `RUNES.md` §0.5:

- **CLIENT**: the 16.19 bin or the client stringtable
- **RIOT**: patch notes
- **WIKI**: wiki at the cited oldid
- **INFERRED**: researcher inference, at confidence H, M or L

## 0. Sources

**Client data.** These files are in `/mnt/nfs/shared/modern-world-map-research/cdragon-16.19/`,
with sha256 values in `SHA256SUMS`:

- `shared.cdtb.bin.json` (sha256 `34f68553ab38cfe344936473fcb48d99574d5fd51e4feb42994fa0c4fe50769e`). It contains these
  objects:
  - `Shared/Spells/SummonerFlash`, `SummonerTeleport`, `S12_SummonerTeleportUpgrade`
    (Unleashed), `S12_SummonerTeleportUpgradeOld`, `TeleportCancel`
  - `SummonerDot` (Ignite), `SummonerExhaust`, `SummonerBarrier`, `SummonerHeal`,
    `SummonerHaste` (Ghost), `SummonerBoost` (Cleanse), `SummonerSmite`
  - `SummonerFlashPerksHextechFlashtraptionV2`
  - `SR_2026_S1_RoleBound_EnhancedTeleportBonus`
- `cdragon-summoner-spells.json`, which holds spell ids, cooldowns and `gameModes`. Its sha256 is
  `5449cbad8e20fdfd841dab4fac0241063d66386f06d5a31252d0ce56f113dafa`.
- `en_us-lol.stringtable.json`, which holds the `generatedtip_summonerspell_*` tooltips. Its
  sha256 is `8c051cb2a24b31f3fa9af95d39085b832b0da2b8cf51093f98c4ab0620ecb8e8`.
- `cdragon-items.json`, for Ionian Boots of Lucidity (3158) and Crimson Lucidity (3171). Its
  sha256 is `f126341e102bf33968afece5ea213b81f61086a5488e94eda4f22940fd82da09`.

**Riot patch notes 26.1–26.19.** Text is archived at `cdragon-16.19/riot-patchnotes-26.x/`.
Summoner-relevant lines:

| Patch | Change |
|---|---|
| 26.1 | Top quest Teleport reward: Unleashed Teleport on a 7-minute cooldown if not taken; if taken, a shield of 30% max HP for 30 s. Smite 600/900/1200 → 600/1000/1400. |
| 26.4 | "Omnivamp will no longer apply to Smite and Ignite damage." |
| 26.12 | Top-quest TP shield 30%/30 s → **35%/10 s**. |
| 26.19 | Top quest free TP 420 → **390 s**. Upgraded Unleashed TP for a top laner who took TP: 330–240 → **300–210 s**. Non-top Unleashed TP unchanged. |

No other summoner spell changed in 26.1–26.19. Flash, Ignite (except omnivamp), Exhaust,
Barrier, Heal, Ghost and Cleanse are unchanged.

**Wiki permalinks** (`index.php?oldid=N`), all fetched 2026-10-01 and archived in
`cdragon-16.19/wiki-2026-10-01/`:

| Page | oldid |
|---|---|
| Module:SpellData/data | **4070517** (numbers for every spell) |
| Flash | 4008665 |
| Teleport | 4065272 |
| Ignite | 4053158 |
| Exhaust | 4053233 |
| Barrier | 3959683 |
| Heal | 3958758 |
| Ghost | 3991260 |
| Cleanse | 4037451 |
| Smite | 4053300 |
| Summoner spell | 4068655 |
| Haste | 4070414 |
| Role Quests | 4064833 |
| Blink | 4047228 |
| Grievous Wounds | 3969410 |
| V6.19 | 4014063 |

**Cross-reference.** `docs/modern/ROLE_QUESTS.md` §4.2–4.3 owns quest completion. This document
reuses its Teleport findings, and they agree.

## 1. Shared rules

### 1.1 Loadout

- Each champion has exactly **2 distinct** summoner spells, chosen before the game.
- **Smite is required for jungle** and is deferred.
- A completed top quest can add a **third**, bonus Unleashed Teleport in the role-quest slot if
  Teleport was not equipped (§3.6).
- Validation rejects:
  - duplicate spells
  - spells whose `mGameModes` lacks `CLASSIC`
  - Clarity, Mark, Jade, Arena and Poro variants
- Teleport/Hexflash substitutions are applied by the rune layer: Flashtraption becomes Cash Back
  without Flash (`RUNES.md` §2.2).

### 1.2 Cooldown and summoner spell haste

```
effective_cd = base_cd * 100 / (100 + summoner_haste)
```

This is WIKI Haste 4070414. Summoner spell haste is a **separate stat bucket** from ability haste.
Ability haste does **not** affect summoner spells.

Sources of summoner spell haste on SR at 26.19:

| Source | Amount | Notes |
|---|---|---|
| Cosmic Insight | **+18** | CLIENT |
| Ionian Boots of Lucidity (3158) | **+10** | CLIENT item text "Gain 10 Summoner Spell Haste" |
| Crimson Lucidity (3171) | **+20** | Mid-quest boots only, so it does not affect a top laner |
| Grisly Mementos | 0 on SR | Its summoner-haste variant applies only in modes without wards |

**Smite** (deferred): haste affects only its **recharge** (90 s), not the 15 s cast cooldown
(WIKI Summoner spell). CLIENT has `mCooldownNotAffectedByCDR` on Smite.

**Haste changes apply to the remaining cooldown.** **Default (INFERRED-M):** when haste changes,
the remaining cooldown is rescaled proportionally, so the elapsed fraction is preserved. That is
the engine's general behavior for haste changes. See U-S1.

**Teleport and Unleashed Teleport** use summoner haste (WIKI V12.15 hotfix).

### 1.3 Start-of-game cooldown

On SR, **all summoner spells start on a 15 s cooldown** at match start. This matches the
spawn-gate opening. Sources: WIKI Summoner spell 4068655, Trivia; introduced in V6.19 (WIKI
4014063), with no removal found.

- **Smite**'s charge timing is unaffected.
- Whether summoner haste shortens this initial 15 s: **default no** (U-S2).
- In simulator terms: `cd_remaining[slot] = 15.0` at game time 0. Fountain spawn is at t=0, and
  spells are castable from t=15 s.

### 1.4 Casting under crowd control, channels and stealth

| Spell | Castable while stunned/silenced etc. ("cast-inhibiting" CC) | Blocked by root/ground | Blocked by stasis/suppression | Breaks own channels? | Breaks stealth |
|---|---|---|---|---|---|
| Flash | **No** | **No (cannot cast)** | No | No (`mDoesntBreakChannels`) | n/a (camouflage rules deferred) |
| Teleport / Unleashed | **No** | **No (cannot cast)** | No | — (it *is* a channel) | — |
| Hexflash | **No** | **No** | No | — | yes (`mCastingBreaksStealth`) |
| Ignite, Exhaust | Yes (`canCastWhileDisabled`) | Yes | **No** | No | **Yes** |
| Barrier, Heal, Ghost, Cleanse | Yes | Yes | **No** | No | No |

Sources: CLIENT flags `canCastWhileDisabled`, `cantCastWhileRooted`, `mDoesntBreakChannels` and
`mCastingBreaksStealth`; WIKI Summoner spell: "stasis and suppression disable all summoner
spells". Nearsight additionally blocks Teleport (WIKI Teleport).

**Cleanse** cannot be cast under suppression or stasis (WIKI Cleanse).

**Cast time.** All of these spells are **instant**, with no cast time. Do not model the
`castFrame`/`spellCastTime` animation fields as gameplay delays (INFERRED-M; WIKI Teleport "has
no cast time"). Ignite's `spellCastTime 0.25` is visual: the first tick lands 0–0.264 s after the
cast, per the next stat-update tick (WIKI Ignite). See U-S3.

### 1.5 Rune and item interactions owned by other specs

| Interaction | Effect | Spec |
|---|---|---|
| Nimbus Cloak | MS bracket by hasted cooldown: <100 s gives 15%, 100–250 s gives 35%, >250 s gives 45%. Teleport always gets 45%. | `RUNES.md` §5.5 |
| Sudden Impact | Armed by Flash, Hexflash and TP arrival | `RUNES.md` §4.6 |
| Conqueror | +2 stacks on Ignite activation | `RUNES.md` §3.1 |
| Electrocute | +1 stack on Ignite activation | `RUNES.md` §4.1 |
| Aery | Triggered by Ignite's first tick | `RUNES.md` §5.2 |
| Hextech Flashtraption | Hexflash, §8 below | `RUNES.md` §7.2 |
| Unsealed Spellbook | Swaps summoner spells | deferred |
| Crimson Lucidity | Noxian Haste: casting a summoner spell grants MS for 4 s | items spec |

`lin(a,b) = a + (b-a)*(L-1)/17`. All level-scaled summoner values below use this
`ByCharLevelInterpolation` form unless noted (CLIENT). For L19–20, extrapolation is the default,
as in `RUNES.md` U-01.

## 2. Flash (id 4, `SummonerFlash`)

Sources: CLIENT `mEffectAmount[0] = 400`, `cooldownTime 300`, `castRange 25000` (cursor
anywhere), `castRangeDisplayOverride 425`, `cantCastWhileRooted`, `mDoesntBreakChannels`;
WIKI Flash 4008665, SpellData.

| Property | Value |
|---|---|
| Cooldown | **300 s** (CLIENT) |
| Blink distance | **400** units toward the cursor (CLIENT effect amount). If the cursor is beyond 400, blink exactly 400 in that direction; if closer, blink to the cursor (WIKI). The 425 display ring is cosmetic. |
| Cast | Instant, quick-cast. The champion's facing is set to the blink direction (WIKI). |
| Restrictions | Cannot be cast while rooted, grounded, stunned/silenced/other cast-inhibiting CC, suppressed or in stasis. Sealed during movement lockouts such as being in another champion's dash lock (WIKI V25.04). |
| Interaction with the current attack/cast | Does not interrupt channels (`mDoesntBreakChannels`). INFERRED-M: it does not cancel an in-progress attack windup. Flash during windup is a known engine behavior; deferred to the champion/attack-state spec (U-S4). |

**Wall rule.** CLIENT gives the target point; the resolution is WIKI Blink 4047228:

- If the destination lies in impassable terrain, the unit is placed at the **nearest passable
  location to the destination**. This can effectively exceed 400 units across a thin wall, or be
  shorter.
- **Implementation:** `dest = start + dir*min(|cursor - start|, 400)`. If `navgrid.blocked(dest,
  radius)`, search outward from `dest` for the nearest open cell center, then clamp to the
  disk-contact contract of `MODERN-011`. Ties go to the cell nearest `start` (INFERRED-M).
- No pathing check happens between start and dest: blinks ignore intervening terrain and units.

**Post-effects:** arms Sudden Impact (4 s) and triggers Nimbus Cloak (300 s cooldown gives the
45% bracket).

## 3. Teleport (id 12, `SummonerTeleport`) and Unleashed Teleport (`S12_SummonerTeleportUpgrade`)

Sources: CLIENT DataValues. Base TP: `ChannelDuration 3, TravelSpeed 1000, MaxTravelTime 8,
UpgradeMinute 10, cooldownTime 300`. Unleashed: `ChannelDuration 3, TravelSpeed 4500,
MaxTravelTime 7, MSAmount 0.5, MSDuration 4`, plus `UpgradedCooldown` as below. Both have
`mRequiredUnitTags [minion, structure, Ward]`, `mChannelIsInterruptedByAttacking false`,
`cantCastWhileRooted`, `mDoesNotConsumeCooldown` (the script starts the cooldown), and targeting
forgiveness 400 beyond distance 2000. WIKI Teleport 4065272, SpellData, Role Quests 4064833;
RIOT 26.1/26.12/26.19.

### 3.1 Phases

1. **Cast.** Instant cast, no cast time, on a target allied unit at any distance (`castRange
   25000`, global).
   - Valid targets: allied **minions**, allied **structures** (turrets; INFERRED-M that
     inhibitors and Nexus are also valid via the `structure` tag, U-S5), and allied **wards**.
   - Also valid (WIKI): certain allied summoned units and traps.
   - Invalid: untargetable units, clones, Farsight wards.
   - **Destination** = the target's position **at cast time**; later target movement is ignored.
   - Travel distance `d = |caster_pos_at_cast - target_pos_at_cast|`.
   - Casting removes homeguard / homestart buffs.
   - **Forgiveness:** if the target is ≥ 2000 away (CLIENT; the wiki says 3000), targeting snaps
     to valid units within 400 of the click.
2. **Channel.** Lasts **3.0 s**. The champion cannot move, attack, cast, use items or use other
   summoner spells.
   - **Interrupted by:** death, root, ground, silence, and every other cast-inhibiting CC (stun,
     knockup, suppression…) (WIKI "interrupts: death, root, ground, silence"; INFERRED-H for the
     rest).
   - **Not interrupted by** taking damage (`mChannelIsInterruptedByAttacking false`).
   - The caster **cannot cancel** the channel (V8.23 "Commit", still current).
   - The channel continues if the target dies or the ward expires.
   - **On interrupt, cooldown:** no client/wiki source. **Default (INFERRED-L):** the cooldown
     starts at its full value on interrupt (U-S6).
3. **Dash.** The champion is **untargetable** and **cannot act**, moving to the destination over
   the travel time:
   - **Teleport:** `t_dash = 0.5 + 4.5 * min(d, 5000)/5000` → 0.5 to 5.0 s (WIKI formula, V25.S1.1).
   - **Unleashed:** `t_dash = 0.5 + 3.5 * min(d, 18000)/18000` → 0.5 to 4.0 s (WIKI).
   - CLIENT gives `TravelSpeed 1000 / MaxTravelTime 8` and `4500 / 7`. Pure `d/speed` disagrees
     with the wiki at short range: the wiki has a 0.5 s floor, and at 5000 units `d/1000 = 5.0`
     matches the wiki's 5.0. **Default:** the wiki formula, because it is measurement-derived and
     consistent at both endpoints (U-S7).
   - The dash ignores collision and terrain (orb travel).
   - Untargetability does not destroy projectiles already in flight. The minimap icon is hidden
     from enemies.
   - On arrival, if the space is occupied or blocked, land at the nearest open space in the
     direction of travel.
4. **Cooldown start.** The cooldown starts **when the dash ends**, not when the channel ends.
5. **On arrival:**
   - **Unleashed** grants **+50% bonus total MS for 4 s**. CLIENT is `MSDuration 4`; the wiki
     says 3 s. **Default: 4 s from client data** (U-S8).
   - Sudden Impact is armed.
   - Nimbus Cloak triggers when the channel completes.

### 3.2 Cooldowns

| Case | Cooldown | Tag |
|---|---|---|
| Teleport (before 10:00) | **300 s** flat | CLIENT |
| Unleashed (from 10:00), no top quest | `330 - 10*(L-1)` for L ≤ 10, i.e. **330 → 240** at L10, then flat 240 | CLIENT breakpoints: `level1 330, -10/level, breakpoint L10 (+additional -10, then 0/level)`; WIKI "330 - 10 × level up to level 10" |
| Unleashed, top quest completed **and** TP equipped | Same − **30** → **300 → 210** | CLIENT `BuffCounterByCoefficient(top-quest buff, -30)`; RIOT 26.19 |
| Top-quest free Unleashed TP (TP **not** equipped; role-quest slot) | **390 s** flat (26.1–26.18: 420 s). Not in client data; it comes from a server script. | RIOT 26.19; WIKI; see ROLE_QUESTS.md §4.2 |

Unleashed cooldown fixtures by level: L1 330, L2 320, L5 290, L6 280, L9 250, L10–20 240.
With the quest: L1 300, L6 250, L10+ 210.

### 3.3 Transformation at 10:00 game time (`UpgradeMinute 10`; RIOT/WIKI)

At 10:00, Teleport becomes Unleashed Teleport (WIKI):

- Unleashed is placed on a **2 s** cooldown.
- If the current remaining cooldown exceeds Unleashed's maximum cooldown, it is reduced to that
  maximum.
- **Default:** "maximum" = the level-1 Unleashed value, 330 (300 with the quest), after haste
  (U-S9).

INFERRED-M reading of the 2 s rule: the effective remaining cooldown becomes
`max(2, min(remaining, unleashed_cd_max))`.

The base `SummonerTeleport.UpgradedCooldown = 240` field is a legacy tooltip value; do not use it.

### 3.4 Top-lane role quest Teleport (owned by ROLE_QUESTS.md; summarized)

The spell-relevant parts of the quest reward:

- **TP not equipped:** gain Unleashed Teleport as a bonus third spell with a 390 s cooldown. It
  is Unleashed immediately. Its initial cooldown at grant is U-RQ-6, defaulting to ready.
- **TP equipped:**
  - Unleashed cooldown −30 s.
  - After the channel completes and the dash ends, gain a **shield = 35% max HP for 10 s**
    (RIOT 26.12).
  - No shield if the channel was interrupted (WIKI V26.03).
- **Shield timing:** "on arrival" (RIOT 26.1). **Default:** applied when the dash ends.

### 3.5 Teleport state

Teleport needs these state fields:

- `tp_phase ∈ {idle, channel, dash}`
- `tp_t_end`
- `tp_dest_xy`
- `tp_target_id`
- `tp_is_unleashed`
- `tp_cd_remaining`
- `tp_arrive_ms_until`
- the quest flags

## 4. Ignite (id 14, `SummonerDot`)

Sources: CLIENT `TooltipTrueDamageCalculation = Breakpoints(70, +20/level, at L6 +25/level)`,
`DotDuration 5, InverseDuration 0.2, GrievousAmount 0.4, SightRadius 100, castRange 600,
cooldownTime 180, canCastWhileDisabled, mCastingBreaksStealth`; WIKI Ignite 4053158, SpellData.

| Property | Value |
|---|---|
| Cooldown | **180 s** |
| Range | **600**, center-to-center (WIKI "Center range 600") |
| Target | One **enemy champion** (`mAffectsTypeFlags` champion only). Unit-targeted and instant. |
| Total true damage | 70 at L1. +20/level through L5 (150). +25/level from L6. |
| Ticks | **5 ticks** of `total/5`. The first lands 0–0.264 s after the cast. **Default:** the next 0.25 s sim boundary, or the same tick if dt ≥ 0.25 (INFERRED-M). Later ticks are every **1.056 s** (WIKI). The last tick is at ≈ +4.22 s. |
| Grievous Wounds | **40%** healing reduction for **5 s** |
| Vision | Reveals the target. Standard sight, radius 100 around it; not true sight, so it does not reveal stealth. DEFERRED-VISION. |
| Damage properties | True, proc and periodic. **Not amplified by damage amps** (V25.15). **No omnivamp** (RIOT 26.4). Exempt from amps, so §1.4 of RUNES.md does not apply (Coup de Grace, PTA and similar don't amplify it). |
| Re-cast | Overrides and refreshes. Does not stack. |
| Cleanse | Removes the DoT and the reveal, **not** the Grievous Wounds (WIKI) |
| Rune triggers | Conqueror +2 on activation; Electrocute +1; Aery on the first tick; combat events on each tick |

Per-level total damage, from CLIENT breakpoints, L1–20: 70, 90, 110, 130, 150, 175, 200, 225,
250, 275, 300, 325, 350, 375, 400, 425, 450, **475** (L18), 500, 525.

**Grievous Wounds rule** (WIKI GW 3969410): healing and regeneration received are multiplied by
`(1 - 0.40)`. Multiple GW sources don't stack; take the max. It is unified at 40% for all sources
since V13.3.

## 5. Exhaust (id 3, `SummonerExhaust`)

Sources: CLIENT `DebuffDuration 3, Slow 40, DamageReduction 35, castRange 650, cooldownTime 240,
canCastWhileDisabled`; WIKI Exhaust 4053233.

- **Cast.** Target enemy champion, range **650**, instant. Cooldown **240 s**.
- **Effect** for **3 s**: two separate debuffs.
  - A **40% slow** (a CC debuff; reduced by slow resist).
  - **"Damage dealt −35%"**. It does **not** reduce true damage. It is a dealt modifier on the
    target (RUNES.md §1.4).
- **Interactions:**
  - Cleanse removes both debuffs.
  - Tenacity does not shorten either; the wiki wording "only the slow can be affected by
    tenacity" predates slow-resist separation.
  - **Default (INFERRED-M):** slow resist reduces the slow's magnitude; tenacity does nothing.
- **Re-cast** refreshes the duration without stacking.
- It counts as a slow, so it triggers Cheap Shot, Approach Velocity, Font of Life and Glacial
  (no immobilize).

## 6. Barrier (id 21)

Sources: CLIENT `ShieldStrength lin(100,460)`, `ShieldDuration 2.5`, `cooldownTime 180`,
`canCastWhileDisabled`; WIKI Barrier 3959683.

- Self-cast, instant.
- **Shield** `lin(100,460)` for **2.5 s**.
- Scaled by heal and shield power, and by Revitalize (RUNES.md §6.7).
- Triggers Shield Bash.
- Cooldown **180 s**.
- **Fixtures:** L1 100, L6 205.88, L9 269.41, L13 354.12, L18 460.

## 7. Heal (id 7)

Sources: CLIENT `TotalHeal lin(80,318)`, `AllyRange 900`, `MoveSpeed 0.30`, `MoveSpeedDuration
1`, `StackingHealDebuff 0.5`, `DebuffDuration 30`, `CursorForgiveness 200`, `cooldownTime 240`;
WIKI Heal 3958758.

- Self plus one allied champion, either the one nearest the cursor within 200 units or the most
  wounded (lowest %HP) within 900. Both get:
  - a heal of `lin(80,318)`
  - **+30% bonus total MS for 1 s**
- **Repeat debuff.** Targets that were affected by Heal within the debuff window get **50%**
  healing.
  - Window: CLIENT 30 s vs WIKI 35 s. **Default: 30 s from client data** (U-S10).
- Reduced by Grievous Wounds. Scaled by HSP and Revitalize.
- Cooldown **240 s**.
- **Fixtures:** L1 80, L6 150, L9 192, L13 248, L18 318.

## 8. Ghost (id 6, `SummonerHaste`)

Sources: CLIENT `MoveSpeedMod lin(0.24,0.48)`, `Duration 10`, `cooldownTime 240`; WIKI Ghost
3991260, SpellData.

- Self, instant.
- **+`lin(24%,48%)` bonus MS** (percent, additive bucket) and **Ghosted**, which ignores unit
  collision, for **10 s**.
- Not interrupted by CC.
- Cooldown 240 s.
- **Unused data.** The bin also contains a second calc `{7bcbbf47} = lin(4,7)` that no tooltip
  references. It is unused, possibly a legacy duration; ignore it (U-S11).
- Celerity multiplies this bonus by 1.07.
- **Fixtures:** L1 24%, L6 31.06%, L9 35.29%, L18 48%.

## 9. Cleanse (id 1, `SummonerBoost`)

Sources: CLIENT `TenacityValue 0.75, TenacityDuration 3, cooldownTime 240`; WIKI Cleanse 4037451.

**Cast.** Self, instant. Cannot be cast under suppression or stasis.

**Removes:**

- blind, charm, flee, taunt, berserk, cripple, disarm, drowsy and sleep, ground, root, silence,
  slow, stun
- summoner-spell debuffs: Ignite DoT and reveal (not GW), Exhaust (both parts)

**Does not remove:** airborne, nearsight, suppression.

**Then** grants **75% tenacity for 3 s**.

Cooldown 240 s.

## 10. Smite (id 11) — DEFERRED (jungle)

Sources: CLIENT `SmiteBaseDamage 600, SmiteUpgradedDamage 1000, Smite2ndUpgradedDamage 1400,
FirstPVPDamage 40, SmiteSlowAmount 0.2, SmiteSlowDuration 2, mMaxAmmo 2, mAmmoRechargeTime 90,
cooldownTime 15, castRange 500` (edge range); WIKI Smite 4053300; RIOT 26.1.

- Target: large/medium monster or lane minion, 600 true damage. Pets take 40.
- Charges: 2 charges with a 90 s recharge and a 15 s gap between casts. The game starts with 1
  charge and gains more from 0:48 (WIKI).
- Upgrades: Unleashed Smite (15 pet treats) deals 1000, and can hit champions for 40 true + 20%
  slow for 2 s. Primal Smite (35 treats) deals 1400 AoE.
- **Not needed for the top-lane milestone.** The loadout validator should allow it but mark it
  deferred.

## 11. Hexflash (Hextech Flashtraption; spell `SummonerFlashPerksHextechFlashtraptionV2`)

Sources: CLIENT spell `mChannelDuration 2, mCastRangeGrowthMax 400, mCastRangeGrowthDuration 2,
cooldownTime 20, mCanMoveWhileChanneling true, mUseChargeChanneling`; rune
`MinimumChannelDuration 1, ChampionCombatCooldown 10`; WIKI T:Hextech Flashtraption 4011828.

- **Availability.** Replaces Flash while Flash's remaining cooldown is > 2 s.
- **Channel.** Charge for up to **2 s**. MS is set to a static **0**, and the range grows from
  **200 to 400**. The wiki formula: 200 + 40 per 0.3 s, capped at 400; it reaches the cap at
  1.5 s.
- **Release** at ≥ **1 s**; it auto-releases at 2 s. It blinks to the target within the current
  range, then gives +50% bonus MS for about 0.25 s (WIKI estimate).
- **Cooldown** 20 s after the channel.
- **Early stop.** Releasing before 1 s, or entering **champion combat**, puts Hexflash on a 10 s
  cooldown with no blink.
- Arms Sudden Impact. Nimbus Cloak triggers on blink, or on interrupt only after channeling long
  enough to blink.

## 12. State the simulator must carry (per champion)

| Field | Notes |
|---|---|
| `summ_ids[2]` (static), `bonus_slot_id` (quest TP) | Loadout |
| `summ_cd_remaining[3]` | Initialized to 15 s at t=0 (§1.3). The quest slot is initialized on grant. |
| `summoner_haste` | Derived from Cosmic Insight and Lucidity |
| `tp_phase`, `tp_t_end`, `tp_dest`, `tp_target`, `tp_unleashed:bool`, `tp_quest_shield:bool`, `unleashed_ms_until` | §3 |
| `ignite_debuff_until[enemy]`, `ignite_next_tick_t`, `ignite_per_tick`, `ignite_ticks_left`, `ignite_source` | §4 |
| `gw_until`, `gw_amount` (on the target) | Grievous Wounds |
| `exhaust_slow_until`, `exhaust_dmg_until` (on the target) | §5 |
| `shield_barrier_amount`, `shield_barrier_until` | §6; shield stack ordering is owned by the damage spec |
| `heal_debuff_until` | §7 |
| `ghost_until`, `ghost_ms_pct` | §8 |
| `cleanse_ten_until` | §9 |
| `hexflash_phase`, `hexflash_start_t`, `hexflash_cd` | §11 |
| `game_time` | For the 10:00 Unleashed transformation |

## 13. Events and hooks

1. **Tick start:**
   - cooldowns −= dt
   - the 10:00 transform (§3.3)
   - TP phase transitions: channel end → dash; dash end → arrive, set the cooldown, apply MS and
     the quest shield
   - Ignite ticks
   - debuff expiries
2. **Action phase.** Summoner casts resolve **before** attacks and abilities issued in the same
   tick (INFERRED-M: summoners are instant, and Flash during windup is common, U-S4). Then:
   - Nimbus Cloak and Sudden Impact arming (rune hooks)
   - Crimson Lucidity Noxian Haste
   - Conqueror and Electrocute stacks from Ignite
3. **Damage pipeline.**
   - Exhaust's −35% applies to the exhausted unit's outgoing non-true damage (dealt modifier).
   - Barrier is a shield layer.
   - Heal and Ignite-tick healing are modified by GW.
4. **CC application.** Any cast-inhibiting CC, root, ground or death during the TP channel
   cancels it; this also covers Hexflash (§3.1). Cleanse removes CC and summoner debuffs in its
   cast tick.
5. **Movement.** Ghost MS and collision ignore. TP dash sets `untargetable & cannot_act` with a
   scripted position path. Flash/Hexflash teleport the position with the nearest-passable
   resolution.

## 14. Diff vs current implementation

| ID | Location | Current | Spec |
|---|---|---|---|
| S-D1 | `lanerl_jax/sim/` (modern) | **No summoner spell code exists.** grep finds no Flash/Teleport/Ignite/Exhaust/Barrier/Heal/Ghost/Cleanse in `modern_*.py`. ROLE_QUESTS.md also notes `state.py` has no summoner/Teleport state. | New: loadout validation, cooldown state, the §12 fields, the §13 hooks |
| S-D2 | `lanerl_jax/sim/modern_items.py:22-38` `ItemStats` | No `summoner_haste` field. Lucidity's "Gain 10 Summoner Spell Haste" text is not parsed (the regex at `:80-85` only maps ability haste, health regen and tenacity). | Add `summoner_haste` and `item_haste`; map Lucidity (3158) +10 and Crimson Lucidity (3171) +20 |
| S-D3 | `lanerl_jax/sim/modern_stats.py:115-128` | Dealt reductions are multiplicative with amps | Exhaust is in the additive dealt-modifier sum (RUNES.md U-19, D-6) |
| S-D4 | Legacy (4.20) Teleport in the old sim, if any | 4.20 values: 300 s, 3.5 s channel, instant blink, 240 s turret refund | All replaced by §3. The legacy ruleset is preserved separately per MODERN-001. |
| S-D5 | `docs/MODERN_PATCH_DELTA.md` §11.4 | "Unleashed… +50% MS for 3s"; base dash 0.5–5 s | MS duration 3 s (wiki) vs 4 s (client 26.19); default 4 s (U-S8). The rest is confirmed. It omits the 26.19 quest values (390 s; −30 s) and the 26.12 shield (35%/10 s). |

## 15. Unresolved / needs live measurement

| ID | Question | Default | Test (Practice Tool 26.19 unless noted) |
|---|---|---|---|
| U-S1 | Remaining-cooldown rescale when summoner haste changes (buying Lucidity mid-cooldown) | Proportional rescale | Flash, then buy Lucidity; read the remaining time |
| U-S2 | Is the 15 s start-of-game cooldown reduced by haste, and is it still present (wiki trivia)? | 15 s, unhasted | Custom SR game, Cosmic Insight: read the summoner cooldowns at 0:00 |
| U-S3 | Ignite first-tick delay and tick spacing (1.056 s) at the sim tick rate | +0.25 s first, then 1.056 s | Log HP |
| U-S4 | Flash during attack windup / ability cast: cancel, continue or reposition? | Out of scope; champion attack-state spec | Frame capture |
| U-S5 | Teleport onto inhibitors and the Nexus | Allowed (`structure` tag) | Try it |
| U-S6 | Teleport cooldown after the channel is interrupted by CC | Full cooldown from interrupt | Get stunned mid-channel; read the cooldown |
| U-S7 | TP dash time: wiki linear formula vs client `TravelSpeed`/`MaxTravelTime` | Wiki formula | Teleport at 1000/5000/10000 units; time the arrival |
| U-S8 | Unleashed MS duration: 4 s (client) vs 3 s (wiki) | 4 s | Watch the buff timer |
| U-S9 | The 10:00 transform cooldown cap, "maximum Unleashed cooldown" | Level-1 value, after haste; a floor of 2 s | Cast TP at 9:00; read at 10:00 |
| U-S10 | Heal repeat-debuff window: 30 s (client) vs 35 s (wiki) | 30 s | Two Heals 32 s apart (needs 2 champions) |
| U-S11 | Ghost `{7bcbbf47} = lin(4,7)` | Unused | — |
| U-S12 | Flash wall-resolution tie-break and the exact "nearest passable" metric on the navgrid | Nearest open cell to the destination | Flash into known walls in Practice Tool; compare landing coordinates |
| U-S13 | Quest free-TP initial cooldown on grant (ROLE_QUESTS U-RQ-6) | Ready | Complete the quest; check the slot |

## 16. Test fixtures

| # | Setup | Expected |
|---|---|---|
| S-F1 | t=0 | All summoner cooldowns at 15.0; Flash castable at t ≥ 15 |
| S-F2 | Flash from (0,0), cursor (1000,0), open ground | Land at (400,0); cooldown 300 |
| S-F3 | Flash, cursor (200,0) | Land at (200,0) |
| S-F4 | Flash with 28 summoner haste (Cosmic + Lucidity) | Cooldown `300*100/128 = 234.375` |
| S-F5 | Flash while rooted, or while stunned | Rejected |
| S-F6 | Ignite at L1 / L5 / L6 / L9 / L18 | Total 70 / 150 / 175 / 250 / 475 true. Per tick 14 / 30 / 35 / 50 / 95. 5 ticks. GW 40% for 5 s. |
| S-F7 | Ignite plus a target healing 100 HP | Receives 60 |
| S-F8 | Exhaust on an attacker dealing 200 physical pre-mitigation (no other amps) | 130 pre-mitigation; true damage unchanged; target slowed 40% for 3 s |
| S-F9 | Barrier L1 / L9 / L18 | 100 / 269.41 / 460 for 2.5 s |
| S-F10 | Heal L9 on self | +192 HP, +30% MS for 1 s. A second Heal within 30 s: +96 |
| S-F11 | Ghost L1 / L18 | +24% / +48% MS for 10 s, no unit collision |
| S-F12 | Cleanse while stunned, Ignited and Exhausted | Stun, Ignite DoT and Exhaust removed; GW remains; 75% tenacity for 3 s |
| S-F13 | Teleport at 5:00, target 5000 units away | Channel 3.0 s + dash 5.0 s; cooldown starts at +8.0 s; 300 s |
| S-F14 | Teleport at 5:00, target 2500 away | Dash `0.5 + 4.5*0.5 = 2.75` s |
| S-F15 | Unleashed at L6, target 9000 away | Dash `0.5 + 3.5*0.5 = 2.25` s; +50% MS for 4 s on arrival; cooldown 280 |
| S-F16 | Unleashed at L12 with top quest complete (TP equipped) | Cooldown 210; 35% max HP shield for 10 s on arrival |
| S-F17 | Top quest complete, TP not equipped | Bonus Unleashed TP in the quest slot, cooldown 390 |
| S-F18 | TP used at 8:00 (cooldown until 13:00) at L7 | At 10:00, remaining 180 → Unleashed; min(180, 330) = 180 remains (§3.3 default) |
| S-F19 | TP channel at t, enemy stun at t+1.5 | Channel cancelled, no dash; cooldown per U-S6 default |
| S-F20 | TP channel, taking 300 damage | Not interrupted |
| S-F21 | Nimbus Cloak: Flash 300 s / Ignite 180 s / Exhaust 240 s / TP | 45% / 35% / 35% / 45% (per RUNES.md §5.5) |
