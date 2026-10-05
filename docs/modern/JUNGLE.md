# JUNGLE.md — 26.19 regular camps, Rift Scuttler, monster AI, rewards, Smite, jungle pets

Code: `lanerl_jax/modern/jungle/camps.py` (rules), `lanerl_jax/modern/items/effects/jungle.py` (item 1101–1103
damage modifier), `lanerl_jax/modern/data/build_jungle.py` → `lanerl_jax/modern/data/26.19/jungle_client.json`
(client values only). Tests: `lanerl_jax/modern/tests/test_jungle.py`.

Evidence levels: **CLIENT** (16.19.8230722 bins), **WIKI** (wiki.leagueoflegends.com, revisions below),
**PATCH** (Riot notes 26.1–26.19, `patch-notes-26.x/p26-*.txt`), **INFERRED-M** (reasoned, medium
confidence), **INFERRED-L** (placeholder, low confidence). If client and wiki disagree, the client wins.

Sources (cached in `/mnt/nfs/shared/modern-world-map-research/jungle-16.19/`):
- CLIENT: `sru_{blue,red,gromp,murkwolf,murkwolfmini,razorbeak,razorbeakmini,krug,krugmini,krugminimini,crab}.bin.json`
  (Root character records), `geometry-decoded.json` (`NeutralCampGeComponentDef` markers + monster
  placements, Team 300), `map11-decoded.json` (`CampName`, `JungleLocationMapInformation`),
  `cdragon-16.19/shared.cdtb.bin.json` (`SummonerSmite`, `Monster_Heal_Mis`, `CrestoftheAncientGolem`,
  `BlessingoftheLizardElder`), `items_client.json` (1101–1103 data values), en_US string table.
- WIKI: Blue Sentinel 4002539, Red Brambleback 4002538, Gromp 4069561, Murk Wolf camp 4002532, Raptor camp
  4002531, Krug camp 4045076, Rift Scuttler camp 4058481, Monster 4070733, Jungling 3990919, Smite 4053300,
  Template:Jungle monster stat 3709342, Template:Jungle pet info 4051359, Buff data Crest of Insight 4048128 /
  Crest of Cinders 4048126 / Speed Shrine 4058482, Experience (champion) (econ-wiki cache).
- PATCH 26.1 (spawn times, Smite 600/1000/1400, pets, quest 35 treats, crest changes, Scuttler level),
  26.12 (Smite second-charge fix), 26.14 (Blue AH 10/15/20), 26.16 (pet damage ratios), 26.4 (no omnivamp on
  Smite). 26.1's "270 s / 120 s respawn" and "30% gold to allies" are **Swiftplay-only**; 26.16 "Maximum
  Health Scaling 250% ⇒ 315%" is **League Classic** — neither applies to SR.

## 1. Camps and slots

14 camps (12 jungle camps + 2 Scuttlers), positions = client `NeutralCamp` markers; each monster's home =
its client placement (client X, Z = sim x, y). Monster slots (38 ≤ 40 regular budget; 8 left for epics):

| Camp (per side) | Members (Monster type) | Slots |
|---|---|---|
| Blue Sentinel | BLUE | 1 |
| Red Brambleback | RED | 1 |
| Gromp | GROMP | 1 |
| Murk Wolves | WOLF + 2 WOLF_MINI | 3 |
| Raptors | RAPTOR + 5 RAPTOR_MINI | 6 |
| Krugs | KRUG (Ancient) + KRUG_MEDIUM + 4 reserve | 6 |
| Rift Scuttler (each river) | SCUTTLE | 1 |

Krug split (WIKI): Ancient → 4 Mini Krugs, Krug → 2 Mini Krugs, 8 units total; the dying parent's slot is
reused by one mini, so 4 reserves suffice. Minis spawn 1.0 s after the death (WIKI "over 1 second"), one
level below the parent (WIKI), at the death position + 60-unit offsets (INFERRED-L).

| Camp | First spawn | Respawn after last member dies | Leash |
|---|---|---|---|
| Blue, Red | 0:55 PATCH 26.1 | 300 s WIKI | 650 WIKI |
| Wolves, Raptors | 0:55 PATCH 26.1 | 135 s WIKI | 650 WIKI |
| Gromp | 1:07 PATCH 26.1 | 135 s WIKI | 450 WIKI |
| Krugs | 1:07 PATCH 26.1 | 135 s WIKI | 650 WIKI |
| Scuttler | 2:55 PATCH 26.1 | 150 s CLIENT tooltip (wiki infobox 2:30) | — |

Respawn timer starts when every member, including Krug splits and pending splits, is dead (WIKI). When a
camp's large monster dies, remaining members are *marked for death* and despawn (no reward) after 10 s
without champion combat (WIKI).

Camp level = average champion level, rounded, at spawn (WIKI Monster "Camp level").

## 2. Monster stats (level-1 CLIENT; scaling WIKI)

| Type | HP | AD | Armor/MR | MS | AS | Range | Windup | Missile | Radius | Gold | XP | Bonus on attack |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Blue | 2300 | 66 | 42 | 275 | 0.493 | 100 INFERRED-M | 0.279 | melee | 131 | 90 | 95 | 5% current HP phys WIKI |
| Red | 2300 | 66 | 42 | 275 | 0.493 | 200 | 0.261 | melee | 120 | 90 | 95 | 5% current HP phys WIKI |
| Gromp | 2050 | 70 | 42 | 330 | 0.425 | 150 | 0.18 | 1800 | 120 | 80 | 120 | 5% current HP magic WIKI |
| Greater Murk Wolf | 1600 | 30 | 42 | 525 | 0.625 | 175 | 0.3125 | melee | 80 | 55 | 50 | 3% current HP phys |
| Murk Wolf | 630 | 10 WIKI | 20 | 525 | 0.625 | 175 | 0.3125 | melee | 50 | 15 | 15 | — |
| Crimson Raptor | 1200 | 17 | 42 | 450 | 0.667 | 200 | 0.2 | 750 | 75 | 35 | 20 | 3% current HP phys |
| Raptor | 500 | 7 | 20 | 525 | 1.0 | 125 | 0.4 | melee | 50 | 8 | 10 | — |
| Ancient Krug | 1400 (wiki 1350) | 57 | 42 | 250 | 0.613 | 150 | 0.368 | melee | 100 | 15 | 15 | 3% current HP phys |
| Krug | 650 | 20 | 20 | 400 | 0.613 | 110 | 0.41 | melee | 50 | 10 | 10 | — |
| Mini Krug | 60 | 13 | 20 | 400 | 0.613 | 110 | 0.41 | melee | 25 | 14 | 16 | — |
| Rift Scuttler | 1550 | 35 | 42 | 255 | 0.638 | 300 | — | never attacks | 50 | 55 | 100 | — |

Windup fraction = client `mAttackCastTime / mAttackTotalTime`, else `0.3 + attackDelayCastOffsetPercent`
(INFERRED-M); windup seconds = fraction / AS.

Level scaling (WIKI Template:Jungle monster stat, camp infoboxes): AD × (1, 1, 1.1, 1.15, 1.2, 1.25, 1.35,
1.45, 1.55, 1.65, 1.8, 1.95, 2.1, 2.25, 2.4, 2.6, 2.8, 3.0) by level 1–18. HP: template ×1 (L1–2),
1.2 + 0.1·(L−3) (L3–11), 2.0 + 0.05·(L−11) (L12+); Blue/Red 1 + 0.1·(L−1) from L3; Scuttler 1 + 0.1·(L−1)
to L9, 2.0 + 0.2·(L−10) to L17 (L18 = L17, INFERRED-L). XP × (1, 1, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5),
capped at monster level 8. Gold flat, except Scuttler × (1, 1.05, …, 1.4 at L9, 1.5, …, 2.2 at L17).
First Scuttlers: 35% less HP, 80% less XP (WIKI); 26.1 removed the −1 spawn level (PATCH).

Monster attacks: PHYSICAL `TAG_BASIC_ATTACK` (no life steal), raw = AD + bonus; Gromp's magic bonus is a
second packet emitted at launch (INFERRED-M; its main hit rides the missile).

## 3. Monster AI (WIKI Monster "behavior & Patience"; rates INFERRED)

- **Aggro**: any champion damage on any camp member aggroes the whole camp (WIKI). Lane minion/turret damage
  does not (INFERRED-M). Targets: the nearest champion that damaged the camp during this aggro episode,
  regardless of vision (WIKI "nearest champion"; restriction to attackers INFERRED-M; straight-line instead of
  path distance INFERRED-M). Re-evaluated every 0.25 s (INFERRED-M); each switch costs 1/7 patience (WIKI
  "up to 6 target changes").
- **Patience** P ∈ [0, 1] drains while the monster or its target is beyond the camp leash, or no target can be
  found: 0.35/s × (1 + excess/leash), halved while hit in the last 1 s (INFERRED-L). Smalls share the large
  monster's patience and reset while it is alive and within 700 (WIKI).
- **Soft reset** (P = 0, 6 s WIKI): walk home at +20% MS (WIKI), heal 6% max HP/s (WIKI), ignore attackers
  until back inside the leash; a hit inside the leash ends it, restores 0.5 patience (INFERRED-L) and grants
  1.5 s without patience loss (WIKI).
- **Hard reset** (after 6 s soft): ignore everything, heal 25%/s (INFERRED-L "much faster"), 2× MS
  (INFERRED-L). Arriving within 50 of its home: full heal (INFERRED-M), patience refills after 2 s over 2 s
  (WIKI).
- **Rift Scuttler**: never attacks; patrols ±700 along the river axis (1, −1)/√2 at base MS − 100 (WIKI −100;
  path INFERRED-L); champion damage makes it flee away from the attacker for 3 s (INFERRED-L) at full MS,
  clamped to ±1400 along the river. 100% slow resist while not fleeing, −100% tenacity (CC ×2) (WIKI).
  Untargetable 1.5 s after spawning (CLIENT `untargetableSpawnTime`).
- Movement terrain/collision are the world's (`mechanics.move_step`).

## 4. Rewards (to the champion landing the killing blow, WIKI Experience/Gold)

Gold and XP per §2; no level-difference penalty in 26.x (removed, WIKI). Jungle-item holders additionally
(WIKI Template:Jungle pet info, CLIENT):
- +80 XP per large monster, +150 XP on the first; comeback +50 XP × round(levels behind) when more than 1.1
  levels behind the average (CLIENT `SmiteComebackXP` 50).
- Treats: +1 per large monster, champion takedown and epic takedown (CLIENT `MonsterKillAmount`,
  `ChampKillAmount`, `EpicKillAmount` = 1). One bonus treat stored per 60 s (90 s adult); a large kill with a
  stored bonus feeds one extra treat and pays 20 gold (adult: consumes up to 2, 20 gold each).
- Kill heal (large): `min(70 + 20·L̄, 250) · min(1.25 · missing%, 1)` (CLIENT `Monster_Heal_Mis`; minimum 0 per
  PATCH 26.1; formula shape INFERRED-M), L̄ = average champion level; mana `(15 + 4·L̄) · min(1 + 1.25 ·
  missing%, 2.25)`; energy 15 (PATCH 26.1). 0.3 s delay not modelled (INFERRED-M).
- Jungle quest (35 treats): +10 gold +10 XP per large monster; +4% MS in jungle/river, 8% out of combat
  (PATCH 26.1) — `quest_jungle_ms` needs a jungle mask.
- Lane-minion penalties: −70% XP at 0:00 shrinking to 0 at 20:00 unless ≥1.5 levels behind; Monster Hunter
  (minion gold > 40% of monster gold, before 14:00): −13 gold and −50% XP per minion (WIKI).
- Holders take 50% damage from non-epic monsters and deal +10% non-true damage to them (PATCH 26.1).

## 5. Buffs

- **Crest of Insight** (Blue, 120 s WIKI): AH 10/15/20 at 1/6/11 (CLIENT, PATCH 26.14), 5 + 1% max mana
  (or energy) per second (CLIENT FlatRestore/PercentRestore).
- **Crest of Cinders** (Red, 120 s): on-hit slow 10/15/25% at 1/6/11 (CLIENT SlowPotency; the 26.1 note's
  "10/15/20" is superseded by client data), ranged ×0.5, 3 s; burn 15 + 3/level from 6 true over 3 instances
  (on hit, +1 s, +2 s), re-hits only refresh (WIKI); regen 0.5/1/1.5/3% max HP per 5 s at 1/4/6/11 when not
  recently damaged by champions, turrets or epics (CLIENT).
- Both transfer to an enemy champion killer with a fresh duration and are lost on any other death (WIKI,
  string table "If the buff holder is slain, this buff is transferred to their killer").
- Global/Voidborn crests removed in 26.1 (PATCH). Draconic wisps (Elemental Rift) not modelled (DEFERRED).
- **Speed Shrine** (Scuttler kill): 90 s for the killer's team (CLIENT), 500-radius area, 525 sight (WIKI
  estimates), +30% MS for 1.5 s for champions that dealt no damage in 5 s (WIKI). Placed at the Scuttler camp
  marker (INFERRED-M; real: in front of the pit). Voidborn Scuttler reveal (post-20:00) DEFERRED.

## 6. Smite and jungle pets

**Smite** (CLIENT `SummonerSmite`): 600 / 1000 (15 treats) / 1400 (35 treats) true damage (PATCH 26.1) to
large/medium monsters (incl. epic) and enemy lane minions; not small monsters (CLIENT tags, WIKI). Range 500
edge-to-edge (`castRangeUseBoundingBoxes`). Forgiveness: nearest smiteable monster within 125 of the cursor.
Max 2 charges, 90 s recharge (summoner haste applies to the recharge only, WIKI), 15 s between casts (not
hasted, `mCooldownNotAffectedByCDR`). Starts with 1 charge; the second-charge recharge starts at 0:48 (WIKI,
PATCH 26.12 fix) → 2 charges at 2:18. First cast at 0:15 (summoner start cooldown, INFERRED-M). Proc, no
damage modifiers, no omnivamp (PATCH 26.4), castable while disabled. Unleashed (15 treats): champions take
40 true + 20% slow 2 s. Primal (35 treats): also 1400 to every other monster within 210 of the target (CLIENT
`castRadius`, `AoESmiteRatio` 1). Smite itself no longer heals (removed V12.22, WIKI); healing comes from the
jungle-item kill heal. Smite on champion pets (40) is not modelled (no pets in the sim).

**Pets** (CLIENT `PetDPS`/`PetHPS`, WIKI): once per second while a jungle monster within 650 targets the
holder, the pet hits every such monster for 20–150 (by level) + 10% bonus AD + 16% AP + 4% bonus HP + 25%
bonus armor + 25% bonus MR true damage (AoE/pet tags, no omnivamp; PATCH 26.16 ratios) and heals the holder
6 (+2/level from 4). "Two more attacks after you stop" and "cannot kill epics" are not modelled (epic slots
are not jungle slots). Evolutions at 15 / 35 treats (WIKI/PATCH 26.1; client `Breakpoint1` still 20). At 35
the egg is consumed (effects persist) and the jungle quest completes. Evolution buffs (CLIENT item data):
- Scorchclaw's Slash: 3 embers per 0.5 s up to 100, large kills fill; next damage to an enemy champion burns
  it and enemies within 250 for 5% max HP true over 4 s (1 tick/s) and slows 30% for 3 s (decay not modelled).
- Gustwalker's Gait: 30% MS decaying over 1.5 s on entering brush, 45% on large kills (needs `in_brush`).
- Mosstomper's Courage: 200 (+20/level from 11) shield on evolution, on large kills and after 10 s out of
  combat (if a fight happened since the last grant).
- Jungle-zone mana regen (8% + L·0.615% of missing mana per s) via `pet_mana_regen` (needs a jungle mask).

## 7. Unresolved / deferred
- Path distance for targeting/leash, exact patience drain rates and reset speeds/heal (INFERRED).
- Scuttler river path, flee duration, shrine position; immobilize "weaken defenses" (Elusive) not modelled.
- Elemental Rift Draconic camps and wisps, Voidborn Scuttler reveal, King of the Jungle (not found in notes).
- Gromp bonus magic timing for ranged hits; Red/Scorch slows without decay.
