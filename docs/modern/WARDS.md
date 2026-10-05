# WARDS.md — 26.19 wards, trinkets, stealth and true sight (modern world)

**Status (2026-10-02).** The rules are implemented in `lanerl_jax/modern/wards.py` (ward slots, trinket
charges, placement, hits, rewards, Oracle sweeps, Deep Ward and Sixth Sense) and `lanerl_jax/modern/vision.py`
(ward sight radii and the optional stealth / true sight / unobstructed / exposed inputs). Tests:
`lanerl_jax/modern/tests/test_wards.py`. The world tick (`world.tick`) does not call them yet. Integration
is the lead's job (see "Integration contract").

Evidence levels:
- **CLIENT**: CommunityDragon 16.19 (client 16.19.8230722): `items_client.json`, `items.cdtb.bin.json` spells,
  character records `yellowtrinket` / `jammerdevice` / `bluetrinket` / `sightward` (fetched 2026-10-02 from
  `raw.communitydragon.org/16.19/game/data/characters/<name>/<name>.bin.json`), and `lol.stringtable.json`
  tooltips.
- **WIKI**: wiki.leagueoflegends.com raw pages (2026-10-02): Ward, Stealth_Ward, Control_Ward, Oracle_Lens,
  Farsight_Alteration, Sight, Stealth, Turret, Deep_Ward, Sixth_Sense, Grisly_Mementos, and the templates
  `Tip data/{Totem,Control,Farsight,Stealth} ward`, `Disabled ward`, `True sight` and `Ward timer info`.
- **PATCH**: `/mnt/nfs/shared/modern-world-map-research/patch-notes-26.x/p26-*.txt`.
- **INFERRED-M / INFERRED-L**: our own call, with medium or low confidence.

Naming: in 26.19 the trinket item 3340 is called "Stealth Ward", but the unit it places is the **Totem Ward**
(client `YellowTrinket`). The 150 s / 30 g "Stealth Ward" unit only comes from support quest items, so it does
not exist in the lane world. The code calls the trinket ward `WardType.TOTEM` (0). The other types are
`CONTROL` (1) and `FARSIGHT` (2).

## Rules

### Units

| ID | Rule | Value | Evidence |
|---|---|---|---|
| WRD.01 | Totem Ward health, sight radius and gameplay radius. | 3 HP, 900, 1 | CLIENT `YellowTrinket` `baseHP 3`, `perceptionBubbleRadius 900`, `overrideGameplayCollisionRadius 1`. |
| WRD.02 | Control Ward health and sight radius. | 4 HP, 900 | CLIENT `JammerDevice` `baseHP 4`, `perceptionBubbleRadius 900`. |
| WRD.03 | Farsight Ward health and sight radius. | 1 HP, 500 | CLIENT `BlueTrinket` `baseHP 1`, `perceptionBubbleRadius 500`. |
| WRD.04 | Wards take 1 damage per basic-attack hit, whatever the attack's damage. | 1 | WIKI Ward. |
| WRD.05 | Only enemy champion basic attacks damage wards. Turrets ignore wards. Minions do not target them. Abilities cannot target wards. | — | CLIENT unit tags `Ward \| Special \| Special_TurretIgnores`. WIKI Ward ("cannot be targeted by most abilities"). Minions: INFERRED-M. |
| WRD.06 | Gold bounty for the killer. | Totem 10, Control 30, Farsight 15 | WIKI tip data. Only the killer is paid. In a 1v1 lane there is never a second ally (WIKI Ward). |
| WRD.07 | Early detection: an enemy's first hit within 10 s of placement pays 5 g. That 5 g comes out of the bounty. | 5 g, 10 s | WIKI Ward. Pings are not modelled. |
| WRD.08 | Killing a ward gives no XP. | 0 | WIKI V14.19 (Totem, Stealth and Farsight). Control Ward: U-W-5. |
| WRD.09 | Control Ward regeneration: 1 HP every 3 s once it has gone 6 s without damage. The first tick comes 9 s after the last hit. | 1 / 3 s after 6 s | WIKI Control Ward. The first-tick timing is INFERRED-L. |
| WRD.10 | Wards do not collide and do not count as minions for targeting. | — | WIKI V1.0.0.120. CLIENT `pathfindingCollisionRadius 5`. |

### Stealth Ward trinket (3340) → Totem Ward

| ID | Rule | Value | Evidence |
|---|---|---|---|
| TOT.01 | Charges. | max 2 | CLIENT `Effect5Amount 2`, `mMaxAmmo 2`. |
| TOT.02 | Recharge per charge, by average champion level (linear from level 1 to 18). | 210 → 90 s | CLIENT `StartingSingleChargeTime 210`, `EndingSingleChargeTime 90`. PATCH 26.3 ("170–90 ⇒ 210–90 (by level)"). WIKI ("average champion level"). Linear interpolation: INFERRED-M. |
| TOT.03 | Ward duration, by average champion level at placement. | 90 → 120 s | CLIENT `Effect1Amount 90`, `Effect3Amount 120`. WIKI. |
| TOT.04 | Totem Wards placed per player. Placing past the cap replaces the oldest one. | 3 | CLIENT `MaxWardsPlaced 3`. WIKI Ward ("replace your earliest"). |
| TOT.05 | Cast range, measured from the champion's centre to the point. | 625 | CLIENT `castRange 625`. The centre-to-point measurement is INFERRED-M. |
| TOT.06 | The ward is visible for its first 2 s, then stealthed. | 2 s | WIKI Totem Ward tip data. |
| TOT.07 | Lockout between activations. | 1.25 s | CLIENT `cooldownTime 1.25`. The wiki says 2 s. |
| TOT.08 | Starting charges at game start. | 1 | INFERRED-L (U-W-2). |

### Control Ward (2055)

| ID | Rule | Value | Evidence |
|---|---|---|---|
| CTL.01 | Cost; inventory stack. | 75 g; 2 | CLIENT `price 75`, `max_stack 2` (the shop already enforces both). The 40 g quest price is support-only (PATCH 26.1/26.3). |
| CTL.02 | Placing a Control Ward consumes one from the inventory. One may be placed per player; a second replaces the first. | 1 | CLIENT `consumed`, `maxNumberOfUnits 1`. WIKI. |
| CTL.03 | Cast range. | 625 | CLIENT `castRange 625`. The wiki says 600. |
| CTL.04 | It lasts until killed. It is visible (not stealthed) and cannot be disabled. Control Wards do not disable each other. | — | WIKI tip data. CLIENT tooltip ("Control Wards do not disable other Control Wards"). |
| CTL.05 | It has true sight over its 900 sight radius: it reveals stealthed enemy wards and disables every enemy ward (Totem and Farsight) in that radius. A disabled ward grants no sight. | 900 | WIKI tip data / Sight ("true sight of wards and traps within the area"). The radius is the ward's sight: INFERRED-M. |
| CTL.06 | The Control Ward is exposed to the enemy (visible through fog) while it reveals a stealthed enemy ward. | — | WIKI tip data. |
| CTL.07 | Placement is allowed only while alive. | — | PATCH 26.16 bugfix (Control Wards were being consumed while dead). |

### Farsight Alteration (3363) → Farsight Ward

| ID | Rule | Value | Evidence |
|---|---|---|---|
| FAR.01 | Purchase requires level 9; the shop enforces it. | 9 | CLIENT `required_level 9`. |
| FAR.02 | Cast range. | 4000 | CLIENT `castRange 4000`, `Effect1Amount 4000`. |
| FAR.03 | One charge; recharge by average level. | 198 → 99 s | CLIENT `Effect10Amount 198`, `Effect11Amount 99`. WIKI. |
| FAR.04 | Sight 800 for 2 s after placement, then 500. | 800 / 2 s / 500 | CLIENT `ScryerVisionRange 800`, `Effect2Amount 2`, `PersistVisionRange 500`. |
| FAR.05 | The Farsight Ward sees over terrain and into brush (unobstructed). | — | CLIENT tooltip ("Can see into Terrain and Brush"). |
| FAR.06 | Once it spots an enemy champion inside its current radius, its sight becomes 800, and it destroys itself 3 s later. A disabled Farsight Ward does not trigger. | 3 s | WIKI V13.10. Ignoring the trigger while disabled is our choice: the wiki lists the disabled-ward trigger as a bug. |
| FAR.07 | It is visible and lasts until killed or triggered. It has no per-player limit. | — | WIKI tip data. |

### Oracle Lens (3364)

| ID | Rule | Value | Evidence |
|---|---|---|---|
| ORA.01 | Charges; recharge by average level. | 2; 160 → 100 s | CLIENT `MaxAmmo 2`, `Starting/EndingSingleChargeTime`. WIKI. |
| ORA.02 | Sweep duration. The sweep follows the user. | 8 s | CLIENT `Duration 8`. PATCH 26.1. |
| ORA.03 | Sweep radius by the user's level, edge range: 600 at levels 1–4, 630 at 5, +30 every 3 levels, 750 from 17. | 600–750 | CLIENT `StartingRadius 600`, `EndingRadius 750`, `castRadius` table. WIKI V13.10 breakpoints. |
| ORA.04 | The sweep reveals stealthed enemy wards in its radius, through brush and over terrain, and disables them. A ward stays disabled 2 s after it leaves the radius. | 2 s | WIKI Oracle Lens / Ward ("drone can see into brush and over terrain"). CLIENT tooltip ("Revealed Stealth Wards are disabled while revealed"). |
| ORA.05 | While the sweep is active, wards the user hits are revealed for 2 s. | 2 s | WIKI. |
| ORA.06 | Lockout between activations. | 5 s | CLIENT `cooldownTime 5`. WIKI. |
| ORA.07 | The Sweeper Drone's obscured vision of non-ward units (silhouettes) is not modelled, because it does not make units targetable. | — | WIKI Sight ("Obscured vision"). |

### Trinkets in general

| ID | Rule | Value | Evidence |
|---|---|---|---|
| TRK.01 | Item haste and trinket haste shorten the recharge: `R · 100 / (100 + haste)`. Sources are Grisly Mementos (+6 trinket haste per memento) and Cosmic Insight (+10 item haste). | — | WIKI ("Trinkets count as active items and their cooldowns will be affected by item haste"). CLIENT Grisly `TrinketAH 6`. The formula is INFERRED-M; the client sets `mAmmoNotAffectedByCDR`. |
| TRK.02 | Swapping trinkets (in the shop, free) keeps the time-equivalent of the old trinket's charges and progress. | — | WIKI V9.24 ("Charges are now converted into its value in cooldown"). Our implementation: `(charges + progress) · R_old / R_new`. INFERRED-M. |
| TRK.03 | Using a trinket does not break recall. | — | PATCH 26.17 bugfix ("placing Red Trinket, Sight Ward, or Vision Ward canceled Recall"). |
| TRK.04 | Placement must be on walkable terrain; team gates count as closed. Requests out of range or on terrain are rejected; the champion does not walk into range. | — | WIKI V13.4 (invalid-location indicator). PATCH 26.11 (no clamp casting for ward trinkets). Rejecting instead of walking: INFERRED-M. |
| TRK.05 | The world has a slot budget of 8 wards per team. When it is full, the team's oldest ward is replaced. | 8 | Project contract (`MAX_WARDS_PER_TEAM`). With one champion per team, only Farsight spam can reach it (3 Totem + 1 Control + Farsight Wards). |

### Vision

| ID | Rule | Value | Evidence |
|---|---|---|---|
| VW.01 | A ward is a viewer with its own radius and follows the normal fog rules. In particular, a ward inside a brush sees that brush. | — | WIKI Ward ("Placing a ward inside of the brush will grant vision of the brush"). |
| VW.02 | A stealthed unit is hidden from enemies, except inside an enemy true-sight radius: Control Ward 900, the Oracle sweep, or a turret's 1100. Attack-reveal circles do not show it. | 1100 | WIKI Stealth / Sight / Turret ("True Sight … within 1100 range"). |
| VW.03 | True sight reveals a stealthed target centre to centre, ignoring walls and brush. | — | INFERRED-M (U-W-3). WIKI true-sight tip: "reveals units through Fog of War, brush, and stealth". |
| VW.04 | Exposed wards are visible to the enemy team through fog. These are Sixth Sense and Oracle-hit reveals, and a Control Ward that is revealing a ward. | — | WIKI. |
| VW.05 | A disabled ward grants no sight but can still be seen. | — | WIKI Disabled ward. |
| VW.06 | Wards are fogged units; they are not always visible like structures. | — | Standard. |

### Vision runes (26.19 catalogue)

The retired runes Zombie Ward, Ghost Poro and Eyeball Collection are not in the 26.19 catalogue. Their
replacements are Sixth Sense, Grisly Mementos and Deep Ward (WIKI V25.S1.1).

| ID | Rule | Value | Evidence |
|---|---|---|---|
| RUN.01 | **Deep Ward 8141.** A trinket Totem Ward placed in the *enemy* jungle (and, from level 9, in the river) is *Deep*: +1 HP and +45→150 s duration, scaled by average champion level. Control and Farsight Wards are never Deep. The non-trinket +30→45 s has no ward to apply to in the lane world. | +1, 45→150 s, lvl 9 | CLIENT `ExtraHealth 1`, `LevelThreshold 9`, `TTTrinketDurationIncreaseMin/Max 45/150`. WIKI V25.S1.2. |
| RUN.02 | Deep Ward regions come from the navgrid region bytes. MainRegion 5/6 is jungle and 7/8 is river. Jungle quadrant 3/4 (west/south) is blue's jungle and 1/2 (north/east) is red's. | — | NGRID v7 enums (FrankTheBoxMonster `NavGridCell.cs` at 92943ed). Spot checks on 26.19 camps: INFERRED-M. |
| RUN.03 | **Sixth Sense 8137.** When the holder is alive and off cooldown, it senses the nearest enemy ward within 900 that is untracked and unseen by its team, and tracks it. From level 11 the ward is also revealed (exposed) for 10 s. Cooldown 250 s. | 900, 11, 10 s, 250 s | CLIENT `{d3bd04a2} 900`, `LevelThreshold 11`, `RevealDuration 10`, `MeleeItemCalcValue 250`. WIKI V25.05. "Tracked" is the ward-timer information (`WardView.tracked`, for observations). |
| RUN.04 | **Grisly Mementos 8140.** +6 trinket haste per memento (already in `domination.stats`), applied to trinket recharge by TRK.01. The summoner-haste variant is for modes without wards, so it does not apply on SR. | 6 | CLIENT. |

## Unresolved

- **U-W-1.** Stealth Ward and Oracle Lens carry `GameTimeThreshold 14`, and Oracle also has `MaxAmmoPre 1`, but
  no tooltip uses them. The patch notes (26.1/26.3) and the wiki say the cooldown scales "by level". We follow
  them and ignore the game-time keys. A possible reading is that Oracle has 1 charge before 14:00.
- **U-W-2.** The number of charges at game start is undocumented. We assume 1 for Stealth Ward and Oracle
  (`INIT_CHARGES`) and 0 for Farsight, which is unreachable at level 1 anyway.
- **U-W-3.** Whether Control Ward true sight respects brush and walls. The wiki Sight page says "true sight
  follows the same rules as standard sight", but the true-sight tip says it works "through brush". We reveal
  stealthed wards anywhere within the radius. Oracle and turret true sight are modelled the same way.
- **U-W-4.** Placement delay: Totem and Control Ward spells have `missileSpeed 1450` and `spellTotalTime 1.4`.
  We treat placement as instant and stationary.
- **U-W-5.** Control Ward XP. V6.22 gave 40 XP; V14.19 removed ward XP but lists only Totem, Stealth and Farsight
  Wards. We use 0 (`CONTROL_WARD_XP_HISTORIC = 40` is kept as a named alternative).
- **U-W-6.** Deep Ward's client calculation is `ByCharLevelInterpolation` (holder level), but the wiki says
  average champion level. We use the average.
- **U-W-7.** Not modelled: ward timers from pings or from seeing the placement (only Sixth Sense tracks); the
  disabled-ward attack reveal; Teleport or Jax Q targeting wards (exposes them for 2 s); "Stealth Ward is added
  if you leave base without a trinket"; Faelights; Scryer's Bloom; nearsight; Umbral Glaive Blackout.

## Integration contract (for the world tick)

See the `wards` docstring and `ward_step` / `ward_view` / `vision_kwargs` signatures. Summary:
- `cfg.ward_grid = ward_grid(map_grid)` on the host.
- `Wards` (`init_wards(C, trinket_ids)`) is carried in the state.
- `ward_step` runs once per tick after DEATH/economy, on final positions.
- Its `gold` goes to the champions' gold, and `consumed_control` removes one 2055.
- `ward_view` is written onto the `KIND_WARD` unit slots.
- `vision_kwargs` goes into `vision.visibility`. `n_fogged` must include the ward slots.
- Champion attack packets whose target is a ward slot become `hits` / `hitter` and are removed from the damage
  pipeline.
- Lane AI, idle auto-attack and kits must skip `KIND_WARD`.
