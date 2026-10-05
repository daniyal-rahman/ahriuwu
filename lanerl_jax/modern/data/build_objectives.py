"""TOOL: build the 26.19 epic-objective table and the Elemental Rift terrain variants.

``python -m lanerl_jax.modern.data.build_objectives [--variants-out DIR]``

Inputs (all pinned by sha256 in the output; nothing is fetched here):

* CommunityDragon 16.19 character bins fetched 2026-10-02 into
  ``/mnt/nfs/shared/modern-world-map-research/objectives-16.19/`` (``sru_*.bin.json``)
  and the wiki revisions archived next to them (``wiki/``).
* ``cdragon-16.19/shared.cdtb.bin.json`` (buff spells), ``items.cdtb.bin.json`` (Eye of
  the Herald 3513), ``globals.cdtb.bin.json`` (terrain mutators), ``map11.bin.json``
  (navgrid overlay list).
* ``geometry-decoded.json`` (``NeutralCampGeComponentDef`` camp transforms) and
  ``map11-decoded.json`` (``SR_DragonLevelScript`` constants).
* Navgrid overlays extracted with ``ops/modern/fetch_map.py`` from the pinned 16.19.8230722
  manifest into ``rift-overlays-16.19.8230722`` and ``baronpit-overlays-16.19.8230722``.

Outputs:

* ``lanerl_jax/modern/data/26.19/objectives_client.json``: every value with an evidence tag
  (CLIENT / WIKI / PATCH / INFERRED-M / INFERRED-L) and the variant-artifact pin.
* ``--variants-out`` (default ``/mnt/nfs/shared/modern-world-map-research/rift-variants-26.19``,
  must not exist): ``variants.npz`` with per-variant navgrid flags, per-team walkable masks
  and brush ids, plus ``manifest.json``. Not in git; the JSON pins its sha256.

Overlay format (reverse-engineered here, checked against the base grid): ``u8 version (=1),
u8 rect_count``, then per rect ``u32 x, u32 z, u32 w, u32 h`` (cells) and ``w*h`` little-endian
``u16`` flags, row-major in z; one trailing byte (0 or 1, meaning unknown). Overlay cells
**replace** the base flags of their rectangle: unchanged cells repeat the base values
(including the 0x80 bit), and new cells use known flag combinations (e.g. 3 = brush|wall in
the Ocean overlay where the base is 0x42), which an OR/mask reading would not produce.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import struct
from pathlib import Path

import numpy as np
from . import PATCH_DIR

RESEARCH = Path("/mnt/nfs/shared/modern-world-map-research")
OBJ = RESEARCH / "objectives-16.19"
CD = RESEARCH / "cdragon-16.19"
OUT_JSON = PATCH_DIR / "objectives_client.json"
DEFAULT_VARIANTS = RESEARCH / "rift-variants-26.19"
RIFT_OVERLAYS = RESEARCH / "rift-overlays-16.19.8230722" / "assets" / "assets" / "maps" / "navgrid"
PIT_OVERLAYS = RESEARCH / "baronpit-overlays-16.19.8230722" / "assets" / "assets" / "maps" / "navgrid"

# Elemental terrain ids = client ``MapFlagIndexOverride(n)`` of the ``SR_<Element>Terrain``
# mutators (globals.cdtb.bin.json): 1 Infernal, 2 Mountain, 3 Ocean, 4 Cloud, 5 Hextech,
# 6 Chemtech. 0 = no transformation (base map).
ELEMENTS = ("none", "infernal", "mountain", "ocean", "cloud", "hextech", "chemtech")
ELEMENT_OVERLAY = {1: "navgrid_infernal_seasonal.ngrid_overlay", 2: "navgrid_mountain_seasonal.ngrid_overlay",
                   3: "navgrid_ocean_seasonal.ngrid_overlay", 6: "navgrid_chemtech.ngrid_overlay"}
# Baron forms: 0 Hunting (no terrain change), 1 Territorial (crescent wall: "Cup"),
# 2 All-Seeing (lateral tunnel: "Tunnel"). Client mutators SR_BaronPitWalled -> MapBaronPitOverride(1),
# SR_BaronPitTunnel -> (2). Cup = Territorial is INFERRED-M (wiki: "small crescent wall").
BARON_FORMS = ("hunting", "territorial", "all_seeing")
PIT_OVERLAY = {1: "navgrid_cup_base.ngrid_overlay", 2: "navgrid_tunnel_base.ngrid_overlay"}

CHARACTERS = ("sru_baron", "sru_dragon_air", "sru_dragon_chemtech", "sru_dragon_earth", "sru_dragon_elder",
              "sru_dragon_fire", "sru_dragon_hextech", "sru_dragon_water", "sru_horde", "sru_horde_mini",
              "sru_riftherald", "sru_riftherald_mercenary", "sru_atakhan")
SHARED_SPELLS = ("SRX_DragonBuffInfernal", "SRX_DragonBuffMountain", "SRX_DragonBuffOcean", "SRX_DragonBuffCloud",
                 "SRX_DragonBuffHextech", "SRX_DragonBuffChemTech", "SRX_DragonSoulBuffInfernal",
                 "SRX_DragonSoulBuffMountain", "SRX_DragonSoulBuffOcean", "SRX_DragonSoulBuffCloud",
                 "SRX_DragonSoulBuffHextech", "SRX_DragonSoulBuffChemTech", "ElderDragonBuff", "SRT_2024_Horde_DoT",
                 "SRT_2024_Horde_Summoner", "SRT_2024_Horde_SpawnMinis", "SRT_2024_Horde_Shield",
                 "SRU_RiftHerald_ControlBattleSled")


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parse_overlay(raw: bytes) -> list[tuple[int, int, np.ndarray]]:
    """``[(x, z, flags[h, w])]``; strict: the byte count must match exactly."""
    if len(raw) < 2 or raw[0] != 1:
        raise ValueError("unsupported navgrid overlay version")
    n, off, out = raw[1], 2, []
    for _ in range(n):
        x, z, w, h = struct.unpack_from("<4I", raw, off)
        off += 16
        a = np.frombuffer(raw, "<u2", w * h, off).reshape(h, w).copy()
        off += 2 * w * h
        out.append((x, z, a))
    if len(raw) - off != 1:
        raise ValueError("unexpected overlay trailer")
    return out


def apply_overlay(flags: np.ndarray, rects) -> np.ndarray:
    f = flags.copy()
    for x, z, a in rects:
        h, w = a.shape
        if z + h > f.shape[0] or x + w > f.shape[1]:
            raise ValueError("overlay rect outside the grid")
        f[z:z + h, x:x + w] = a
    return f


def _val(v):
    return v["baseValue"] if isinstance(v, dict) and "baseValue" in v else v


def _data_values(spell: dict) -> dict:
    out = {}
    for dv in spell.get("mSpell", {}).get("DataValues", []):
        vals = dv.get("values")
        if vals is None:
            continue
        out[dv["name"]] = vals[0] if len(set(vals)) == 1 else vals
    return out


def character_records(sources: dict) -> dict:
    keys = {"baseHPModifiable": "hp", "hpPerLevelModifiable": "hp_per_level", "baseDamageModifiable": "ad",
            "damagePerLevelModifiable": "ad_per_level", "baseArmorModifiable": "armor",
            "armorPerLevelModifiable": "armor_per_level", "baseMR": "mr", "mrPerLevel": "mr_per_level",
            "baseMoveSpeedModifiable": "move_speed", "attackRangeModifiable": "attack_range",
            "attackSpeedModifiable": "attack_speed", "acquisitionRange": "acquisition_range",
            "expGivenOnDeath": "xp", "goldGivenOnDeath": "gold", "experienceRadius": "xp_radius",
            "globalGoldGivenOnDeath": "global_gold", "globalExpGivenOnDeath": "global_xp",
            "overrideGameplayCollisionRadius": "radius", "unitTagsString": "tags",
            "towerTargetingPriorityBoost": "tower_priority_boost"}
    out = {}
    for name in CHARACTERS:
        path = OBJ / f"{name}.bin.json"
        sources[str(path)] = sha256(path)
        d = json.loads(path.read_text())
        root = next(v for k, v in d.items() if k.lower().endswith("characterrecords/root"))
        rec = {dst: _val(root[src]) for src, dst in keys.items() if src in root}
        ba = root.get("basicAttack", {})
        rec["attack_total_time"] = ba.get("mAttackTotalTime")
        rec["attack_cast_time"] = ba.get("mAttackCastTime")
        spells = {k.split("/")[-1]: _data_values(v) for k, v in d.items()
                  if "/Spells/" in k and isinstance(v, dict) and _data_values(v)}
        rec["spells"] = spells
        out[name] = rec
    return out


def shared_spells(sources: dict) -> dict:
    path = CD / "shared.cdtb.bin.json"
    sources[str(path)] = sha256(path)
    d = json.loads(path.read_text())
    return {n: _data_values(d["Shared/Spells/" + n]) for n in SHARED_SPELLS}


def eye_of_the_herald(sources: dict) -> dict:
    path = CD / "items.cdtb.bin.json"
    sources[str(path)] = sha256(path)
    d = json.loads(path.read_text())
    return {v["mName"]: v["mValue"] for v in d["Items/3513"]["mDataValues"]}


def camps(sources: dict) -> dict:
    path = RESEARCH / "geometry-decoded.json"
    sources[str(path)] = sha256(path)
    d = json.loads(path.read_text())
    out = {}

    def walk(o):
        if isinstance(o, dict):
            nc = o.get("NeutralCamp")
            if isinstance(nc, dict) and nc.get("MinimapIcon") in ("Horde", "SRU_RiftHerald", "Baron", "Dragon"):
                t = o["transform"][3]
                out[nc["MinimapIcon"]] = {"x": t[0], "y": t[2], "stop_spawn_time": nc.get("StopSpawnTimeSecs")}
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d)
    return out


def dragon_script(sources: dict) -> dict:
    """Constants of ``SR_DragonLevelScript`` (LevelControlScript {02a3f5cc}); first branch = SR."""
    path = RESEARCH / "map11-decoded.json"
    sources[str(path)] = sha256(path)
    d = json.loads(path.read_text())
    seen: dict = {}
    formulas: list = []

    def walk(o):
        if isinstance(o, dict):
            dest, src = o.get("Dest"), o.get("Src")
            if isinstance(dest, dict) and isinstance(src, dict) and "value" in src and dest.get("Var"):
                seen.setdefault(dest["Var"], []).append(src["value"])
            if "Formula" in o:
                formulas.append(o["Formula"])
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(d["{02a3f5cc}"])
    keep = ("DragonSpawnRate", "ElderDragonSpawnRate", "DRAGONS_TO_ELDER", "DRAGONS_TO_TERRAINCHANGE",
            "NUM_ELEMENTAL_FLAVORS", "CampSpawnLeadupSoft", "CampSpawnLeadupBright")
    return {"values": {k: seen[k] for k in keep if k in seen},
            "spawn_formulas": [f for f in formulas if "InitialCountdown" in f],
            "note": "first value of each list is the SR branch; the second (150/300, +175) is Swiftplay"}


def rift_variants(sources: dict):
    from .navgrid import ModernMapGrid, load_patch_map
    grid, manifest = load_patch_map(RESEARCH / "grid-26.19-base")
    rects = {}
    for e, fn in ELEMENT_OVERLAY.items():
        p = RIFT_OVERLAYS / fn
        sources[str(p)] = sha256(p)
        rects[("e", e)] = parse_overlay(p.read_bytes())
    for f, fn in PIT_OVERLAY.items():
        p = PIT_OVERLAYS / fn
        sources[str(p)] = sha256(p)
        rects[("p", f)] = parse_overlay(p.read_bytes())

    def cells(rs):
        m = np.zeros(grid.flags.shape, bool)
        for x, z, a in rs:
            m[z:z + a.shape[0], x:x + a.shape[1]] = True
        return m
    base = np.asarray(grid.flags)
    # Overlapping cells are allowed only where the pit overlay keeps the base value (then the
    # element overlay, applied last, decides). 16.19: Ocean x Tunnel share 5 cells, all base in Tunnel.
    overlaps = {f"{e}+{f}": int(np.sum(cells(rects[("e", e)]) & cells(rects[("p", f)])
                                       & (apply_overlay(base, rects[("p", f)]) != base)))
                for e in ELEMENT_OVERLAY for f in PIT_OVERLAY}
    flags, walk, changed = [], [], {}
    for e in range(len(ELEMENTS)):
        for f in range(len(BARON_FORMS)):
            fl = np.asarray(grid.flags)
            if f in PIT_OVERLAY:
                fl = apply_overlay(fl, rects[("p", f)])
            if e in ELEMENT_OVERLAY:
                fl = apply_overlay(fl, rects[("e", e)])
            g = ModernMapGrid(fl.astype(np.uint16), grid.regions, grid.heights, grid.height_spacing,
                              grid.cell_size, grid.min_bounds, grid.max_bounds)
            flags.append(fl.astype(np.uint16))
            walk.append(np.stack([g.walkable(0), g.walkable(1)]))
            base_w = np.stack([grid.walkable(0), grid.walkable(1)])
            changed[f"{ELEMENTS[e]}/{BARON_FORMS[f]}"] = {
                "walkable_cells_changed": int(np.sum(walk[-1][0] != base_w[0])),
                "brush_cells_changed": int(np.sum(((fl & 1) != 0) != ((np.asarray(grid.flags) & 1) != 0)))}
    from scipy.ndimage import label
    bush = np.stack([label((f & 1) != 0)[0].astype(np.int32) for f in flags])
    return (np.stack(flags), np.stack(walk), bush, grid, manifest, overlaps, changed,
            {k: [[int(x), int(z), int(a.shape[1]), int(a.shape[0])] for x, z, a in v] for k, v in
             {f"element_{ELEMENTS[e]}": rects[("e", e)] for e in ELEMENT_OVERLAY}.items()}
            | {f"pit_{BARON_FORMS[f]}": [[int(x), int(z), int(a.shape[1]), int(a.shape[0])] for x, z, a in rects[("p", f)]]
               for f in PIT_OVERLAY})


def rules() -> dict:
    """Spawn/reward/buff rules not stored as client data, each with its evidence tag."""
    E = lambda v, tag, src: {"value": v, "evidence": tag, "source": src}
    return {
        "atakhan_present": E(False, "PATCH", "26.1: Atakhan removed, no longer spawns"),
        "feats_of_strength_present": E(False, "PATCH", "26.1: Feats removed"),
        "objective_bounties": E(False, "INFERRED-M",
                                "exist on SR (wiki) but depend on hidden team-advantage weights; disabled like ECONOMY_PROGRESSION §6.7"),
        "grubs_spawn": E(480.0, "WIKI", "Voidgrub camp: spawn 8:00 (V25.09), one group of 3, no respawn; PATCH 26.1 'spawn times unchanged'"),
        "grubs_count": E(3, "WIKI", "Voidgrub camp"),
        "grubs_despawn": E(885.0, "CLIENT", "NeutralCamp Horde StopSpawnTimeSecs 885 (14:45)"),
        "grubs_despawn_in_combat": E(895.0, "WIKI", "14:55 if in combat"),
        "grub_mites_per_wave": E(4, "CLIENT", "SRT_2024_Horde_SpawnMinis SpawnCount 4"),
        "grub_mite_wave_period": E(12.0, "CLIENT", "SpawnMinisCDMax 12"),
        "grub_mite_lifetime": E(5.5, "CLIENT", "SpawnMaxLifetime 5.5 (wiki: decay over 6 s)"),
        "grub_death_heal_max": E(0.30, "WIKI", "Defensive Measures 30% max + 30% missing, self true damage over 10 s"),
        "grub_death_heal_missing": E(0.30, "WIKI", "Defensive Measures"),
        "grub_death_self_damage_duration": E(10.0, "WIKI", "Defensive Measures"),
        "herald_spawn": E(900.0, "WIKI", "Rift Herald spawntime 15:00 (V25.09), once per game"),
        "herald_despawn": E(1185.0, "CLIENT", "NeutralCamp SRU_RiftHerald StopSpawnTimeSecs 1185 (19:45)"),
        "herald_despawn_in_combat": E(1195.0, "WIKI", "19:55 if in combat"),
        "herald_leash": E(1200.0, "WIKI", "Rift Herald infobox leash 1200"),
        "herald_charge_ad_ratio": E(2.0, "WIKI", "Charge 200% AD physical after 2.5 s windup at fight start"),
        "herald_charge_windup": E(2.5, "WIKI", "Charge"),
        "herald_swipe_thresholds": E([0.6575, 0.3275], "WIKI", "Swipe at 65.75%/32.75% max HP"),
        "herald_swipe_ad_ratio": E(1.25, "WIKI", "Swipe 125% AD cone"),
        "herald_swipe_windup": E(1.5, "WIKI", "Swipe"),
        "herald_on_hit_current_hp": E(0.20, "WIKI", "Colossal Strength 20% current HP bonus physical (V25.09)"),
        "herald_eye_cooldown": E(6.0, "CLIENT", "RiftHeraldMercenary EncounterEyeCD 6 (PATCH 26.1 8 -> 6)"),
        "herald_eye_max_hp_damage": E(0.12, "CLIENT", "RiftHeraldMercenary EncounterEyeMaxHPDmg 0.12 (true damage INFERRED-M)"),
        "eye_pickup_duration": E(20.0, "CLIENT", "item 3513 EyePickUpDuration"),
        "eye_duration": E(300.0, "CLIENT", "item 3513 GlimpseAndTrinketDuration"),
        "merc_leap_damage": E(3000.0, "PATCH", "26.1 mercenary charge damage 3000 (client HeraldLeapAttack 1500+125/lvl is unreconciled)"),
        "merc_leap_decay": E(2.0 / 3.0, "WIKI", "each later charge 66.67% of the previous"),
        "merc_leap_self_damage": E(0.66, "CLIENT", "HeraldLeapAttack CurrentHealthRatioToSelf 0.66"),
        "merc_leap_windup": E(2.5, "WIKI", "Leap Attack windup"),
        "merc_leap_range": E(1000.0, "INFERRED-L", "distance at which she starts the leap (not documented)"),
        "merc_on_hit_current_hp": E(0.0175, "WIKI", "1.75% of her current HP bonus physical"),
        "merc_kill_gold": E(25.0, "WIKI", "Mercenary 25 gold for the slayer"),
        "baron_spawn": E(1200.0, "PATCH", "26.1 Baron 25:00 -> 20:00; wiki 20:00"),
        "baron_respawn": E(360.0, "WIKI", "respawntime 6:00"),
        "baron_buff_duration": E(180.0, "CLIENT", "BaronAttackMelee BaronBuffDuration 180"),
        "baron_ability_every": E(6, "WIKI", "an ability every 6th basic attack, cyclic"),
        "baron_corrosion_ad_ratio": E(0.35, "PATCH", "26.1 Corrosion secondary 35% total AD (magic, WIKI)"),
        "baron_ability_ad_ratio": E(1.0, "PATCH", "26.1 Acid Pool/Acid Shot/Tentacle 100% AD (magic)"),
        "baron_pull_ad_ratio": E(1.4, "WIKI", "Territorial pull 140% AD (patch notes say 100%)"),
        "baron_void_corruption_per_stack": E(0.5, "WIKI", "Void Corruption -0.5 armor/MR per stack, 8 s"),
        "baron_void_corruption_max": E(100, "WIKI", "100 stacks (26.13 fix lowered an over-cap)"),
        "baron_void_corruption_duration": E(8.0, "WIKI", "8 s"),
        "baron_gaze_reduction": E(0.5, "CLIENT", "string game_buff_tooltip_sru_baron_target: target deals 50% less to Baron"),
        "baron_tentacle_knockup": E(1.25, "WIKI", "Tentacle Knockup 1.25 s"),
        "baron_acid_slow": E([0.6, 2.5], "WIKI", "Acid Pool field 60% slow 2.5 s"),
        "baron_aoe_radius": E(300.0, "INFERRED-L", "ability footprint around the target"),
        "hand_of_baron_ad": E([[20, 12], [22, 14], [24, 16], [26, 19], [28, 22], [30, 26], [32, 30], [34, 34],
                               [36, 39], [38, 43], [40, 48]], "WIKI", "Buff data Hand of Baron (minute, AD), latched at kill"),
        "hand_of_baron_ap": E([[20, 20], [22, 23], [24, 27], [26, 32], [28, 37], [30, 43], [32, 50], [34, 57],
                               [36, 65], [38, 72], [40, 80]], "WIKI", "Buff data Hand of Baron (minute, AP)"),
        "baron_minion_acquire_radius": E(600.0, "WIKI", "Hand of Baron notes: unempowered minion within 600"),
        "baron_minion_empower_radius": E(1450.0, "WIKI", "empower all minions within 1450"),
        "baron_minion_lose_radius": E(1500.0, "WIKI", "lose empowerment beyond 1500"),
        "baron_minion_champion_dr": E([[20, 0.50], [40, 0.70]], "WIKI", "melee/caster DR from champions 50-70% by minute (V9.2: 50% + 1%/min)"),
        "baron_minion_melee_minion_dr": E(0.85, "WIKI", "melee 85% DR from minions"),
        "baron_minion_aoe_dr": E(0.15, "WIKI", "15% DR from AoE/persistent/proc (melee/caster)"),
        "baron_minion_range": E([75.0, 100.0, 750.0, 0.0], "WIKI", "bonus attack range melee/caster/siege/super"),
        "baron_minion_ad": E([0.0, 20.0, 50.0, 0.0], "WIKI", "bonus AD melee/caster/siege/super"),
        "baron_minion_missile_speed": E([0.0, 900.0, 1600.0, 0.0], "WIKI", "missile speed caster/siege"),
        "baron_minion_attack_speed_mult": E([1.0, 1.0, 0.5, 1.25], "WIKI", "siege 50% reduced, super +25%"),
        "baron_minion_ms_floor": E([0.925, 500.0], "WIKI", "min MS 92.5% of nearby champions' average, cap 500"),
        "baron_minion_siege_structure": E([2.0, 3.0], "WIKI", "siege vs structures 200% base AD + 300% bonus AD"),
        "baron_minion_siege_splash": E(200.0, "WIKI", "siege splash radius 200"),
        "empowered_recall_time": E(4.0, "WIKI", "Hand of Baron / Glimpse: channel halved (8 -> 4)"),
        "baron_homeguard_bonus": E(0.5, "WIKI", "+50% bonus MS from homeguard"),
        "dragon_first_spawn": E(300.0, "WIKI", "5:00; client SR_DragonLevelScript DragonSpawnTime = InitialCountdown + 235"),
        "dragon_respawn": E(300.0, "CLIENT", "DragonSpawnRate 300"),
        "elder_respawn": E(360.0, "CLIENT", "ElderDragonSpawnRate 360 (first Elder 360 s after the soul drake, wiki 14.3)"),
        "dragons_to_terrain_change": E(2, "CLIENT", "DRAGONS_TO_TERRAINCHANGE 2"),
        "dragons_to_soul": E(4, "CLIENT", "DRAGONS_TO_ELDER 4"),
        "rift_transform_delay": E(0.0, "INFERRED-M",
                                  "wiki: transforms after the 2nd drake is slain; client dragonAtTerrainChangeMomentIndex = 3 hints at the 3rd spawn"),
        "dragon_vengeance_per_stack": E(0.15, "PATCH", "26.1 Ancient Grudge 15/30/45%(/60%) DR vs champions per slayer-team stack"),
        "dragon_leash": E(1200.0, "INFERRED-L", "not documented"),
        "grub_leash": E(1000.0, "INFERRED-L", "not documented"),
        "dragon_local_xp": E([160.0, 400.0], "PATCH", "26.1 local XP 160-400 for levels 6-18 (linear INFERRED-M)"),
        "comeback_xp": E([0.25, 2.0], "PATCH", "26.1: +25% per level behind (enemy-team average, INFERRED-M), cap 2x; dragons/Elder/Baron"),
        "dragon_min_level": E(6, "PATCH", "26.1"),
        "elder_min_level": E(13, "PATCH", "26.1"),
        "baron_min_level": E(11, "PATCH", "26.1"),
        "herald_min_level": E(9, "PATCH", "26.1"),
        "grub_min_level": E(7, "PATCH", "26.1"),
        "monster_level_rounding": E("ceil", "WIKI", "average champion level rounded up (Voidgrub history)"),
        "baron_evolution_delay": E(30.0, "WIKI", "Delayed Evolution: levels only after 30 s out of combat"),
        "non_champion_damage_mult": E(1.5, "PATCH", "26.1: grubs/Herald deal 50% more to non-champions, matching Dragon and Baron"),
        "elder_buff_duration": E(150.0, "WIKI", "Aspect of the Dragon 150 s"),
        "elder_execute_delay": E(0.5, "WIKI", "Elder Immolation 0.5 s delay, 2 s per-target lockout"),
        "elder_burn_ticks": E([0.25, 1.25, 2.25], "WIKI", "burn over 2.25 s, 3 ticks"),
        "touch_dot_duration": E(4.0, "CLIENT", "SRT_2024_Horde_DoT DurationOfDoT"),
        "touch_dot_tick": E(0.5, "CLIENT", "SecondsPerTick"),
        "hunger_voidmite_hp_melee_minion_ratio": E(1.0, "PATCH", "26.11 summoned Voidmite HP = 100% melee minion"),
        "hunger_voidmite_lifetime": E(20.0, "INFERRED-L", "summoned Voidmite lifetime not documented"),
        "merc_voidmites": E(5, "WIKI", "Rodeo impact spawns 5 (+Touch stacks) Voidmites (Rodeo itself deferred)"),
        "patience_soft_reset": E(6.0, "WIKI", "Monster: soft reset 6 s, heal 6% max HP/s, then hard reset"),
        "patience_soft_heal": E(0.06, "WIKI", "6% max HP per second"),
        "patience_hard_heal": E(0.25, "INFERRED-L", "'much faster' healing"),
        "patience_reset_ms_mult": E(1.2, "WIKI", "Template:Monster patience +20% MS when out of patience"),
        "patience_drain": E(0.25, "INFERRED-L", "patience fraction lost per second outside leash / without target (+ distance term)"),
        "monster_target": E("nearest champion", "WIKI", "aggroed monsters acquire the nearest champion regardless of sight; Baron the nearest unit"),
    }


def build(variants_out: Path) -> dict:
    sources: dict = {}
    data = {"schema": "lanerl-modern-objectives-v1", "patch": "26.19", "client_build": "16.19.8230722",
            "evidence_tags": "CLIENT (16.19 bins/maps), WIKI (wiki.leagueoflegends.com 2026-10-02), PATCH (26.1-26.19 notes), INFERRED-M/-L",
            "objectives_present": ["voidgrubs", "rift_herald", "elemental_drakes", "dragon_soul", "elder_dragon", "baron_nashor"],
            "objectives_absent": {"atakhan": "PATCH 26.1 removed", "blood_roses": "PATCH 26.1 removed",
                                  "feats_of_strength": "PATCH 26.1 removed"},
            "characters": character_records(sources), "shared_spells": shared_spells(sources),
            "eye_of_the_herald": eye_of_the_herald(sources), "camps": camps(sources),
            "dragon_script": dragon_script(sources), "rules": rules(),
            "elements": list(ELEMENTS), "baron_forms": list(BARON_FORMS)}
    flags, walk, bush, grid, manifest, overlaps, changed, rects = rift_variants(sources)
    if any(overlaps.values()):
        raise ValueError(f"element and Baron-pit overlays conflict: {overlaps}")
    variants_out = Path(variants_out)
    variants_out.mkdir(parents=True, exist_ok=False)
    buf = io.BytesIO()
    np.savez_compressed(buf, flags=flags, walkable=walk, bush_ids=bush)
    (variants_out / "variants.npz").write_bytes(buf.getvalue())
    arrays_sha = hashlib.sha256(buf.getvalue()).hexdigest()
    vman = {"schema": "lanerl-rift-variants-v1", "patch": "26.19", "client_build": "16.19.8230722",
            "base_grid_arrays_sha256": manifest["arrays_sha256"], "arrays_sha256": arrays_sha,
            "index": "variant = element * 3 + baron_form", "elements": list(ELEMENTS),
            "baron_forms": list(BARON_FORMS), "shape": list(flags.shape),
            "cell_size": grid.cell_size, "min_bounds": list(grid.min_bounds), "max_bounds": list(grid.max_bounds),
            "coordinates": "world-xz; arrays[z,x]", "overlay_rects_xzwh": rects, "changed_vs_base": changed}
    (variants_out / "manifest.json").write_text(json.dumps(vman, indent=2) + "\n")
    data["rift_terrain"] = {"artifact": str(variants_out), "arrays_sha256": arrays_sha,
                            "manifest_sha256": sha256(variants_out / "manifest.json"),
                            "index": vman["index"], "changed_vs_base": changed, "overlay_rects_xzwh": rects,
                            "evidence": "CLIENT navgrid overlays (map11.bin MapNavGridOverlays); Cloud/Hextech ship no navgrid overlay"}
    data["sources_sha256"] = sources
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variants-out", type=Path, default=DEFAULT_VARIANTS)
    p.add_argument("--json-out", type=Path, default=OUT_JSON)
    a = p.parse_args()
    data = build(a.variants_out)
    a.json_out.write_text(json.dumps(data, indent=1, sort_keys=False) + "\n")
    print(json.dumps(data["rift_terrain"]["changed_vs_base"], indent=1))


if __name__ == "__main__":
    main()
