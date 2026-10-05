# Epic objectives and the Elemental Rift (26.19, client 16.19.8230722)

Code: `lanerl_jax/modern/jungle/objectives.py` (rules, AI, buffs, rewards),
`lanerl_jax/modern/map/rift.py` (terrain variants).
Data: `lanerl_jax/modern/data/26.19/objectives_client.json`, built by
`python -m lanerl_jax.modern.data.build_objectives`. Tests:
`lanerl_jax/modern/tests/test_objectives.py`.

Evidence tags: **CLIENT** (16.19 bins, map scripts, navgrid overlays), **WIKI**
(wiki.leagueoflegends.com, fetched 2026-10-02, archived with sha256 in
`/mnt/nfs/shared/modern-world-map-research/objectives-16.19/wiki/`), **PATCH** (Riot notes
26.1–26.19), **INFERRED-M** (a reasoned reading of the evidence), **INFERRED-L** (a placeholder).
Every value in the JSON has its tag and source. The JSON also pins the sha256 of every input file.

## 1. Which objectives exist on 26.19

| Objective | 26.19 | Evidence |
|---|---|---|
| Voidgrubs (3, one group) | yes | WIKI, CLIENT `sru_horde` |
| Rift Herald and the Mercenary | yes | WIKI, CLIENT |
| Elemental Drakes (6 elements), Dragon Soul, Elder Dragon | yes | CLIENT `SR_DragonLevelScript` |
| Baron Nashor (3 forms) | yes | PATCH 26.1, CLIENT |
| Atakhan, Blood Roses, Feats of Strength | **removed** | PATCH 26.1 |
| Objective bounties | exist on SR, **disabled** in the sim | They depend on hidden team-advantage weights (ECONOMY_PROGRESSION §6.7). |

26.16 "League Classic" is a separate mode, not SR, and is ignored here.

## 2. Timeline

| Event | Time | Tag |
|---|---|---|
| First drake | 5:00 | WIKI; CLIENT formula `InitialCountdown + 235` |
| Drake respawn | 300 s | CLIENT `DragonSpawnRate` |
| Voidgrubs | 8:00; despawn 14:45 (14:55 in combat); no respawn | WIKI (V25.09); CLIENT `StopSpawnTimeSecs 885` |
| Rift Herald | 15:00 once; despawn 19:45 (19:55 in combat) | WIKI; CLIENT 1185 |
| Baron Nashor | 20:00; respawn 360 s | PATCH 26.1; WIKI |
| Elder Dragon | 360 s after the soul drake; respawn 360 s | CLIENT `ElderDragonSpawnRate` |

**Drake elements.** The first two drakes are distinct random elements out of six. The third
element is distinct from both. It is decided when the 2nd drake dies (CLIENT
`DRAGONS_TO_TERRAINCHANGE 2`) and becomes the rift element for every later drake. The team that
first reaches 4 Dragon Slayer stacks gets the Soul of the rift element (CLIENT `DRAGONS_TO_ELDER
4`). After that only the Elder spawns. The sim pre-rolls the 3 elements from the init key.

**Rift transformation time.** The sim transforms the rift at the moment of the 2nd kill: wiki
wording, `rift_transform_delay = 0`, INFERRED-M. The client variable
`dragonAtTerrainChangeMomentIndex = 3` hints that it may instead coincide with the 3rd spawn
(open question Q1).

## 3. Monsters

All monsters use the 26.1 champion-style growth, `base + g·(L−1)(0.7025+0.0175(L−1))`. This
reproduces the patch-note ranges exactly: Baron 17,792 at level 11 and 19,190 at 18; drake
5,106 at level 6. Every monster has armor `34+8g` and MR `32+4g` (CLIENT).

**Monster level** is the champions' average level, rounded up, and at least the minimum level:
grubs 7, drakes 6, Herald and Mercenary 9, Baron 11, Elder 13 (PATCH/WIKI). A monster levels up
only after 30 s out of combat (WIKI "Delayed Evolution"; the sim applies this to every
objective). On a level-up its HP keeps the same fraction of max HP.

| Monster | HP | AD | AS | Range | Notes |
|---|---|---|---|---|---|
| Voidgrub | 1300+200g | 12+2g | 1.0 (CLIENT; wiki says 0.5) | 500 | Every 12 s in combat it spawns 4 camp Voidmites (5.5 s life). When a grub dies, the others heal 30% max + 30% missing HP, then take that much back as true damage over 10 s. |
| Rift Herald | 7000+700g | 60+10g | 0.4 | 250 | Attacks deal +20% of the target's current HP. She opens with a Charge (2.5 s windup, 200% AD on her path, knock-aside 0.5 s INFERRED-L). Swipes at 65.75% and 32.75% HP (1.5 s, 125% AD, 350 radius INFERRED-L). Eye: a champion basic attack from behind deals 12% of her max HP (CLIENT) as true damage, then a 6 s cooldown. |
| Mercenary | 2500+270g | 48+8g | 0.5 | 250 | Attacks deal +1.75% of her own current HP. Leap: 2.5 s windup, then 3000 true damage to the structure (PATCH 26.1), ×2/3 for each later charge. She loses 66% of her current HP (CLIENT). |
| Elemental drake | 3625+375g (Mountain 4200+430g) | 35–105 by element | 0.25–1.0 by element | 500 | Attacks deal +4–7% of the target's current HP by element (CLIENT). Infernal splashes 350. Ancient Grudge: −15% damage taken from champions per Dragon Slayer stack of their team (PATCH). |
| Elder | 11500+575g | 150 | 0.5 | 500 | |
| Baron | 16300+170g | 175+20g | 0.625 | 900 | Stationary. See below. |

The monsters marked in the table deal **×1.5 damage to non-champions** (PATCH 26.1). The
Mercenary and the Voidmites do not.

**Baron.**
- Each hit applies Void Corruption: −0.5 armor and MR per stack, max 100 stacks, lasts 8 s
  (WIKI). The basic attack adds 2 extra stacks.
- Corrosion fires with each attack: 35% AD magic damage to the nearest unit with the fewest
  stacks.
- Every 6th attack casts the next ability in the rotation Acid Pool → Form → Acid Shot →
  Tentacle. The first ability is random. The non-form abilities deal 100% AD magic damage; Acid
  Pool also slows 60% for 2.5 s and Tentacle knocks up for 1.25 s.
- Form abilities: Hunting deals 20% of current HP. Territorial deals 100% AD (PATCH; the wiki
  says 140%). All-Seeing deals 140% AD to the 2 furthest champions within 2200.
- Baron's Gaze: his current target deals 50% less damage to him.
- Approximations: abilities resolve instantly at the 6th attack, with no windup, and their
  footprints are simplified (INFERRED-L). The 26.1 "0.5 armor/MR per second" shred is not
  modelled: the wiki reports it does not exist (Q2).

**AI (WIKI semantics, INFERRED-L numbers).**
- A monster aggroes on damage. It then targets the nearest champion within its leash,
  regardless of sight. The Herald instead charges the first unit that damaged her. Baron
  attacks the nearest unit, champion or minion, in his range.
- Patience drains at 0.25/s (more when far outside the leash) whenever the monster has no
  valid in-leash target. Hits inside the leash restore 0.25.
- At 0 patience the monster soft-resets: it walks home for 6 s, heals 6% of max HP per second,
  and ignores attackers outside the leash. Then it hard-resets: it ignores all attackers and
  heals 25%/s. Reset movement is ×1.2 speed.
- Leash radius: Herald 1200 (WIKI); drakes 1200 and grubs 1000 (INFERRED-L).
- Team-owned units: the Mercenary walks to the nearest enemy structure and attacks minions on
  the way; Hunger Voidmites attack structures only.

## 4. Team buffs (CLIENT values)

| Buff | Effect |
|---|---|
| Dragon Slayer per stack | Infernal +3% AD and AP. Mountain +5% armor and MR. Ocean restores 2% of missing HP every 5 s. Cloud +5% slow resist and +5% out-of-combat MS. Hextech +5 AH and +5% AS. Chemtech +6% tenacity and heal/shield power. |
| Infernal Soul | 100 + 22.5% bonus AD + 13.5% AP + 2.75% bonus HP adaptive damage, 250 radius, 3 s cooldown |
| Mountain Soul | After 5 s without damage: shield 220 + 16% bonus AD + 12% AP + 12% bonus HP |
| Ocean Soul | Heal 150 + 26% bonus AD + 17% AP + 7% bonus HP and mana 100 + 3.5% max mana, over 4 s; 30% effect vs minions and monsters |
| Cloud Soul | +15% MS; 60% for 6 s after casting R (30 s cooldown) |
| Hextech Soul | 25–50 true damage plus a slow of 45%/35% + scalings, chaining to 4 champions within 600, 8 s cooldown |
| Chemtech Soul | Below 50% HP: 13% more damage dealt and 13% damage reduction |
| Aspect of the Dragon (150 s, lost on death) | Burn of 75–225 true damage by minutes 25–45, in 3 ticks. Damaging a champion below 20% max HP executes them after 0.5 s (max-HP true damage, `PROP_EXECUTE`), with a 2 s lockout per target. |
| Hand of Baron (180 s, lost on death) | 12–48 AD and 20–80 AP over minutes 20–40, latched at the kill (WIKI table, interpolated). Empowered Recall (4 s). Homeguard +50%. Empowers nearby minions (below). |
| Touch of the Void (per grub, max 3) | Champion non-proc damage to structures burns them for 4 s: melee 4/12/16 and ranged 2/6/8 true damage every 0.5 s (CLIENT, the 26.11 values). At 3 stacks, Hunger of the Void summons 1 allied Voidmite per 15 s while you hit structures. The Voidmite has melee-minion HP (26.11) and AD (INFERRED-L) and lives 20 s (INFERRED-L). |
| Glimpse of the Void (Herald) | The killer gets the Eye for 300 s (CLIENT). The pickup is automatic in the sim; the client gives a 20 s window. Every participant gets an Empowered Recall charge. |

Implementation notes for the buffs:
- Infernal stacks become flat AD/AP from the STATS-phase totals, applied in one pass.
- The Hextech slow is approximated as a constant half-strength slow for 2 s.
- Elder burns, executes and the Hextech chain target enemy champions only (INFERRED-M).
- The Infernal Soul procs on champions and epic monsters (INFERRED-M).
- Soul procs land one tick after the triggering hit.

**Hand of Baron minion empowerment (WIKI).**
- Trigger: an unempowered allied minion within 600 of a buffed champion. Then every allied
  minion within 1450 of that champion is empowered.
- A minion loses the empowerment when no buffed champion is within 1500.
- Melee: +75 range, damage reduction 50→70% from champions (by minutes 20→40), 85% from
  minions, 15% from AoE, periodic and proc damage.
- Caster: +100 range, +20 AD, missile speed 900, the same champion and AoE reductions.
- Siege: +750 range, +50 AD, ×0.5 AS, 200% base + 300% bonus AD vs structures, 200 splash.
- Super: ×1.25 AS.
- All: MS floor of 92.5% of nearby champions' average MS, capped at 500.

## 5. Rewards (PATCH 26.1 unless noted)

| Kill | Gold | XP |
|---|---|---|
| Grub | 30 to the killer | 65 to each team champion within 2000 |
| Herald | 100 to the killer | 240 within 2000 |
| Drake | 75 to the killer | 160–400 (levels 6–18) within 2000 |
| Elder / Baron | 100 to the killer + 150 to every team member | 650 to every team member |
| Mercenary | 25 | 200 within 600 (CLIENT) |
| Camp Voidmite | 1 | 0 |

- Comeback XP (drakes, Elder, Baron): +25% per level below the enemy team's average
  (INFERRED-M reading), capped at 2×.
- `epic_takedown` goes to the killer and to team champions that damaged the monster in the last
  10 s. For grubs, only the first grub counts (WIKI).
- `large_monster_kill` counts kills of grubs, the Herald and the Mercenary.
- A non-champion final blow credits the last champion that damaged the monster (INFERRED-M).

## 6. Elemental Rift and Baron-pit terrain

**Real client data.** `map11.bin` `MapNavGridOverlays` lists navgrid overlays. We extracted
them from the pinned 16.19 manifest with `ops/modern/fetch_map.py`:

| Overlay | sha256 prefix | Rects |
|---|---|---|
| `navgrid_infernal_seasonal` | 3be8d74c | 2 |
| `navgrid_mountain_seasonal` | 5162cfbf | 6 |
| `navgrid_ocean_seasonal` | b07fb383 | 11 |
| `navgrid_chemtech` | 5308b610 | 1 |
| `navgrid_cup_base` (Territorial pit) | a5c8167f | 2 |
| `navgrid_tunnel_base` (All-Seeing pit) | dadc1b17 | 7 |

The files are in `/mnt/nfs/shared/modern-world-map-research/{rift,baronpit}-overlays-16.19.8230722/`.
Cloud and Hextech have no navgrid overlay: their speed zones and Hex-gates are interactive
objects, deferred (Q3).

**Overlay format (reverse-engineered).**
- Header: `u8 version=1, u8 rect_count`.
- Per rect: `u32 x, z, w, h` in cells, then `w·h` `u16` flags.
- One trailing byte of unknown meaning (0 or 1).
- The cells **replace** the base flags. Unchanged cells repeat the base values, and new cells
  use flag combinations that only a replacement reading yields.

**Element ids.** They are the client `MapFlagIndexOverride` ids of the `SR_<Element>Terrain`
mutators: 1 Infernal, 2 Mountain, 3 Ocean, 4 Cloud, 5 Hextech, 6 Chemtech.

**Baron forms.** 0 Hunting (no change), 1 Territorial (Cup), 2 All-Seeing (Tunnel). Mapping Cup
to Territorial is INFERRED-M, from the wiki's "crescent wall". The form is rolled at the first
Baron spawn.

**Variant artifact.** `/mnt/nfs/shared/modern-world-map-research/rift-variants-26.19/`
(`variants.npz`, `manifest.json`; sha256 pinned in the JSON):
- Indexing: `variant = element·3 + form`, so 21 variants.
- Contents: per-variant flags, per-team walkable masks (gates resolved as in
  `ModernMapGrid.walkable`) and per-variant brush labels.
- Overlay order: pit first, then element. One overlap exists (Ocean × Tunnel, 5 cells) and is
  harmless: the Tunnel keeps the base values there.
- Size of the change vs the base: Mountain changes 635 walkable cells, Ocean changes 538 brush
  cells, Infernal opens the dragon-pit walls, and the All-Seeing tunnel changes 411 cells.
- The 0x80 bit, whose meaning is unknown, follows the overlay. Movement and vision do not
  read it.

## 7. Deferred / approximations

- Rodeo (champion-driven Herald charge).
- Hex-gates and Cloud speed zones; Infernal cinders, Chemtech plants, Faelights.
- Corrupted and Draconic jungle camps (owned by `jungle.camps`).
- The 20 s Eye pickup window.
- Ocean, Cloud, Hextech and Mountain drake attack riders.
- Monstrous Toughness caps.
- Smite (not in the 2-champion loadouts).
- The 8-slot budget caps camp Voidmites at 3 live at once; the client allows up to 12.
