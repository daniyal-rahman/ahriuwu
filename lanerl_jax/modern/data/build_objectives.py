"""Build ``26.19/objectives_client.json`` (epic objectives) and the Elemental Rift terrain variants.

    python -m lanerl_jax.modern.data.build_objectives [--variants-out DIR] [--json-out PATH]

Inputs (sha256-pinned in the output; nothing is fetched): CommunityDragon 16.19 character bins in
``RESEARCH/objectives-16.19/`` (wiki revisions archived next to them), ``cdragon-16.19`` shared/items/globals/
map11 bins, ``geometry-decoded.json`` (camp transforms), ``map11-decoded.json`` (``SR_DragonLevelScript``) and
the navgrid overlays extracted with ``ops/modern/fetch_map.py`` into ``rift-overlays-16.19.8230722`` and
``baronpit-overlays-16.19.8230722``. Every rule carries an evidence tag (CLIENT / WIKI / PATCH / INFERRED-M/-L).
``--variants-out`` (must not exist) receives ``variants.npz`` (per-variant flags, per-team walkable masks,
brush ids) and ``manifest.json``; it is not in git, the JSON pins its sha256.

Overlay format (reverse-engineered, checked against the base grid): ``u8 version (=1), u8 rect_count``, per
rect ``u32 x, z, w, h`` (cells) and ``w*h`` little-endian ``u16`` flags row-major in z, then one trailing byte
(0 or 1, meaning unknown). Overlay cells *replace* the base flags of their rectangle: unchanged cells repeat
the base values (incl. 0x80) and new cells use flag combinations an OR/mask reading would not produce
(e.g. 3 = brush|wall in the Ocean overlay where the base is 0x42).
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import json
import struct
from pathlib import Path

import numpy as np

from . import CLIENT_BUILD, PATCH, PATCH_DIR, RESEARCH, sha256, walk_dicts
from .navgrid import load_patch_map

OBJ = RESEARCH / "objectives-16.19"
CD = RESEARCH / "cdragon-16.19"
RIFT_OVERLAYS = RESEARCH / "rift-overlays-16.19.8230722" / "assets" / "assets" / "maps" / "navgrid"
PIT_OVERLAYS = RESEARCH / "baronpit-overlays-16.19.8230722" / "assets" / "assets" / "maps" / "navgrid"

# Element id = client ``MapFlagIndexOverride(n)`` of the ``SR_<Element>Terrain`` mutators (globals bin);
# 0 = base map. Cloud and Hextech ship no navgrid overlay.
ELEMENTS = ("none", "infernal", "mountain", "ocean", "cloud", "hextech", "chemtech")
ELEMENT_OVERLAY = {1: "navgrid_infernal_seasonal.ngrid_overlay", 2: "navgrid_mountain_seasonal.ngrid_overlay",
                   3: "navgrid_ocean_seasonal.ngrid_overlay", 6: "navgrid_chemtech.ngrid_overlay"}
# Baron form id = ``MapBaronPitOverride(n)``: SR_BaronPitWalled -> 1 ("Cup"), SR_BaronPitTunnel -> 2.
# Cup = Territorial is INFERRED-M (wiki: "small crescent wall").
BARON_FORMS = ("hunting", "territorial", "all_seeing")
PIT_OVERLAY = {1: "navgrid_cup_base.ngrid_overlay", 2: "navgrid_tunnel_base.ngrid_overlay"}

CHARACTERS = ("sru_baron", "sru_dragon_air", "sru_dragon_chemtech", "sru_dragon_earth", "sru_dragon_elder",
              "sru_dragon_fire", "sru_dragon_hextech", "sru_dragon_water", "sru_horde", "sru_horde_mini",
              "sru_riftherald", "sru_riftherald_mercenary", "sru_atakhan")
CHARACTER_FIELDS = {
    "baseHPModifiable": "hp", "hpPerLevelModifiable": "hp_per_level", "baseDamageModifiable": "ad",
    "damagePerLevelModifiable": "ad_per_level", "baseArmorModifiable": "armor",
    "armorPerLevelModifiable": "armor_per_level", "baseMR": "mr", "mrPerLevel": "mr_per_level",
    "baseMoveSpeedModifiable": "move_speed", "attackRangeModifiable": "attack_range",
    "attackSpeedModifiable": "attack_speed", "acquisitionRange": "acquisition_range",
    "expGivenOnDeath": "xp", "goldGivenOnDeath": "gold", "experienceRadius": "xp_radius",
    "globalGoldGivenOnDeath": "global_gold", "globalExpGivenOnDeath": "global_xp",
    "overrideGameplayCollisionRadius": "radius", "unitTagsString": "tags",
    "towerTargetingPriorityBoost": "tower_priority_boost"}
SHARED_SPELLS = ("SRX_DragonBuffInfernal", "SRX_DragonBuffMountain", "SRX_DragonBuffOcean", "SRX_DragonBuffCloud",
                 "SRX_DragonBuffHextech", "SRX_DragonBuffChemTech", "SRX_DragonSoulBuffInfernal",
                 "SRX_DragonSoulBuffMountain", "SRX_DragonSoulBuffOcean", "SRX_DragonSoulBuffCloud",
                 "SRX_DragonSoulBuffHextech", "SRX_DragonSoulBuffChemTech", "ElderDragonBuff", "SRT_2024_Horde_DoT",
                 "SRT_2024_Horde_Summoner", "SRT_2024_Horde_SpawnMinis", "SRT_2024_Horde_Shield",
                 "SRU_RiftHerald_ControlBattleSled")
DRAGON_SCRIPT_VARS = ("DragonSpawnRate", "ElderDragonSpawnRate", "DRAGONS_TO_ELDER", "DRAGONS_TO_TERRAINCHANGE",
                      "NUM_ELEMENTAL_FLAVORS", "CampSpawnLeadupSoft", "CampSpawnLeadupBright")


def parse_overlay(raw: bytes) -> list[tuple[int, int, np.ndarray]]:
    """``[(x, z, flags[h, w])]``; the byte count must match exactly."""
    if len(raw) < 2 or raw[0] != 1:
        raise ValueError("unsupported navgrid overlay version")
    off, out = 2, []
    for _ in range(raw[1]):
        x, z, w, h = struct.unpack_from("<4I", raw, off)
        out.append((x, z, np.frombuffer(raw, "<u2", w * h, off + 16).reshape(h, w).copy()))
        off += 16 + 2 * w * h
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


def _data_values(spell: dict) -> dict:
    return {dv["name"]: vals[0] if len(set(vals)) == 1 else vals
            for dv in spell.get("mSpell", {}).get("DataValues", []) if (vals := dv.get("values")) is not None}


def _load(path: Path, sources: dict):
    sources[str(path)] = sha256(path)
    return json.loads(path.read_text())


def character_records(sources: dict) -> dict:
    out = {}
    for name in CHARACTERS:
        d = _load(OBJ / f"{name}.bin.json", sources)
        root = next(v for k, v in d.items() if k.lower().endswith("characterrecords/root"))
        rec = {dst: v["baseValue"] if isinstance(v := root[src], dict) and "baseValue" in v else v
               for src, dst in CHARACTER_FIELDS.items() if src in root}
        ba = root.get("basicAttack", {})
        rec["attack_total_time"] = ba.get("mAttackTotalTime")
        rec["attack_cast_time"] = ba.get("mAttackCastTime")
        rec["spells"] = {k.split("/")[-1]: _data_values(v) for k, v in d.items()
                         if "/Spells/" in k and isinstance(v, dict) and _data_values(v)}
        out[name] = rec
    return out


def camps(sources: dict) -> dict:
    out = {}
    for o in walk_dicts(_load(RESEARCH / "geometry-decoded.json", sources)):
        nc = o.get("NeutralCamp")
        if isinstance(nc, dict) and nc.get("MinimapIcon") in ("Horde", "SRU_RiftHerald", "Baron", "Dragon"):
            t = o["transform"][3]
            out[nc["MinimapIcon"]] = {"x": t[0], "y": t[2], "stop_spawn_time": nc.get("StopSpawnTimeSecs")}
    return out


def dragon_script(sources: dict) -> dict:
    """Constants of ``SR_DragonLevelScript`` (LevelControlScript {02a3f5cc}); first branch = SR."""
    seen, formulas = {}, []
    for o in walk_dicts(_load(RESEARCH / "map11-decoded.json", sources)["{02a3f5cc}"]):
        dest, src = o.get("Dest"), o.get("Src")
        if isinstance(dest, dict) and isinstance(src, dict) and "value" in src and dest.get("Var"):
            seen.setdefault(dest["Var"], []).append(src["value"])
        if "Formula" in o:
            formulas.append(o["Formula"])
    return {"values": {k: seen[k] for k in DRAGON_SCRIPT_VARS if k in seen},
            "spawn_formulas": [f for f in formulas if "InitialCountdown" in f],
            "note": "first value of each list is the SR branch; the second (150/300, +175) is Swiftplay"}


def rift_variants(sources: dict, out: Path) -> dict:
    """Write the ``ELEMENTS x BARON_FORMS`` terrain artifact to ``out``; returns its pin for the JSON."""
    grid, manifest = load_patch_map(RESEARCH / "grid-26.19-base")

    def overlay(path: Path):
        sources[str(path)] = sha256(path)
        return parse_overlay(path.read_bytes())
    element = {e: overlay(RIFT_OVERLAYS / fn) for e, fn in ELEMENT_OVERLAY.items()}
    pit = {f: overlay(PIT_OVERLAYS / fn) for f, fn in PIT_OVERLAY.items()}

    def cells(rs):
        m = np.zeros(grid.flags.shape, bool)
        for x, z, a in rs:
            m[z:z + a.shape[0], x:x + a.shape[1]] = True
        return m
    base = np.asarray(grid.flags)
    # Overlaps are allowed only where the pit overlay keeps the base value (the element overlay, applied last,
    # decides). 16.19: Ocean x Tunnel share 5 cells, all base in Tunnel.
    overlaps = {f"{e}+{f}": int(np.sum(cells(element[e]) & cells(pit[f]) & (apply_overlay(base, pit[f]) != base)))
                for e in ELEMENT_OVERLAY for f in PIT_OVERLAY}
    if any(overlaps.values()):
        raise ValueError(f"element and Baron-pit overlays conflict: {overlaps}")
    base_walk = grid.walkable(0)
    flags, walk, changed = [], [], {}
    for e, element_name in enumerate(ELEMENTS):
        for f, form in enumerate(BARON_FORMS):
            fl = apply_overlay(base, pit[f]) if f in pit else base
            fl = apply_overlay(fl, element[e]) if e in element else fl
            g = dataclasses.replace(grid, flags=fl.astype(np.uint16))
            flags.append(g.flags)
            walk.append(np.stack([g.walkable(0), g.walkable(1)]))
            changed[f"{element_name}/{form}"] = {
                "walkable_cells_changed": int(np.sum(walk[-1][0] != base_walk)),
                "brush_cells_changed": int(np.sum(((fl & 1) != 0) != ((base & 1) != 0)))}
    from scipy.ndimage import label
    bush = np.stack([label((f & 1) != 0)[0].astype(np.int32) for f in flags])
    flags = np.stack(flags)

    def xzwh(rs):
        return [[int(x), int(z), int(a.shape[1]), int(a.shape[0])] for x, z, a in rs]
    rects = ({f"element_{ELEMENTS[e]}": xzwh(rs) for e, rs in element.items()}
             | {f"pit_{BARON_FORMS[f]}": xzwh(rs) for f, rs in pit.items()})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    buf = io.BytesIO()
    np.savez_compressed(buf, flags=flags, walkable=np.stack(walk), bush_ids=bush)
    (out / "variants.npz").write_bytes(buf.getvalue())
    arrays_sha = sha256(out / "variants.npz")
    index = "variant = element * 3 + baron_form"
    vman = {"schema": "lanerl-rift-variants-v1", "patch": PATCH, "client_build": CLIENT_BUILD,
            "base_grid_arrays_sha256": manifest["arrays_sha256"], "arrays_sha256": arrays_sha,
            "index": index, "elements": list(ELEMENTS),
            "baron_forms": list(BARON_FORMS), "shape": list(flags.shape),
            "cell_size": grid.cell_size, "min_bounds": list(grid.min_bounds), "max_bounds": list(grid.max_bounds),
            "coordinates": "world-xz; arrays[z,x]", "overlay_rects_xzwh": rects, "changed_vs_base": changed}
    (out / "manifest.json").write_text(json.dumps(vman, indent=2) + "\n")
    return {"artifact": str(out), "arrays_sha256": arrays_sha, "manifest_sha256": sha256(out / "manifest.json"),
            "index": index, "changed_vs_base": changed, "overlay_rects_xzwh": rects,
            "evidence": "CLIENT navgrid overlays (map11.bin MapNavGridOverlays); Cloud/Hextech ship no navgrid overlay"}


def rules() -> dict:
    """Spawn/reward/buff rules not stored as client data, each with its evidence tag."""
    E = lambda v, tag, src: {"value": v, "evidence": tag, "source": src}
    return {
        "atakhan_present": E(False, "PATCH", "26.1: Atakhan removed, no longer spawns"),
        "feats_of_strength_present": E(False, "PATCH", "26.1: Feats removed"),
        "objective_bounties": E(False, "INFERRED-M",
                                "exist on SR (wiki) but depend on hidden team-advantage weights; disabled like "
                                "ECONOMY_PROGRESSION §6.7"),
        "grubs_spawn": E(480.0, "WIKI", "Voidgrub camp: spawn 8:00 (V25.09), one group of 3, no respawn; PATCH 26.1 "
                         "'spawn times unchanged'"),
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
        "herald_eye_max_hp_damage": E(0.12, "CLIENT",
                                      "RiftHeraldMercenary EncounterEyeMaxHPDmg 0.12 (true damage INFERRED-M)"),
        "eye_pickup_duration": E(20.0, "CLIENT", "item 3513 EyePickUpDuration"),
        "eye_duration": E(300.0, "CLIENT", "item 3513 GlimpseAndTrinketDuration"),
        "merc_leap_damage": E(3000.0, "PATCH", "26.1 mercenary charge damage 3000 (client HeraldLeapAttack "
                              "1500+125/lvl is unreconciled)"),
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
        "baron_gaze_reduction": E(0.5, "CLIENT",
                                  "string game_buff_tooltip_sru_baron_target: target deals 50% less to Baron"),
        "baron_tentacle_knockup": E(1.25, "WIKI", "Tentacle Knockup 1.25 s"),
        "baron_acid_slow": E([0.6, 2.5], "WIKI", "Acid Pool field 60% slow 2.5 s"),
        "baron_aoe_radius": E(300.0, "INFERRED-L", "ability footprint around the target"),
        "hand_of_baron_ad": E([[20, 12], [22, 14], [24, 16], [26, 19], [28, 22], [30, 26], [32, 30], [34, 34],
                               [36, 39], [38, 43], [40, 48]], "WIKI",
                              "Buff data Hand of Baron (minute, AD), latched at kill"),
        "hand_of_baron_ap": E([[20, 20], [22, 23], [24, 27], [26, 32], [28, 37], [30, 43], [32, 50], [34, 57],
                               [36, 65], [38, 72], [40, 80]], "WIKI", "Buff data Hand of Baron (minute, AP)"),
        "baron_minion_acquire_radius": E(600.0, "WIKI", "Hand of Baron notes: unempowered minion within 600"),
        "baron_minion_empower_radius": E(1450.0, "WIKI", "empower all minions within 1450"),
        "baron_minion_lose_radius": E(1500.0, "WIKI", "lose empowerment beyond 1500"),
        "baron_minion_champion_dr": E([[20, 0.50], [40, 0.70]], "WIKI",
                                      "melee/caster DR from champions 50-70% by minute (V9.2: 50% + 1%/min)"),
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
        "dragon_first_spawn": E(300.0, "WIKI",
                                "5:00; client SR_DragonLevelScript DragonSpawnTime = InitialCountdown + 235"),
        "dragon_respawn": E(300.0, "CLIENT", "DragonSpawnRate 300"),
        "elder_respawn": E(360.0, "CLIENT",
                           "ElderDragonSpawnRate 360 (first Elder 360 s after the soul drake, wiki 14.3)"),
        "dragons_to_terrain_change": E(2, "CLIENT", "DRAGONS_TO_TERRAINCHANGE 2"),
        "dragons_to_soul": E(4, "CLIENT", "DRAGONS_TO_ELDER 4"),
        "rift_transform_delay": E(0.0, "INFERRED-M",
                                  "wiki: transforms after the 2nd drake is slain; client "
                                  "dragonAtTerrainChangeMomentIndex = 3 hints at the 3rd spawn"),
        "dragon_vengeance_per_stack": E(0.15, "PATCH",
                                        "26.1 Ancient Grudge 15/30/45%(/60%) DR vs champions per slayer-team stack"),
        "dragon_leash": E(1200.0, "INFERRED-L", "not documented"),
        "grub_leash": E(1000.0, "INFERRED-L", "not documented"),
        "dragon_local_xp": E([160.0, 400.0], "PATCH", "26.1 local XP 160-400 for levels 6-18 (linear INFERRED-M)"),
        "comeback_xp": E([0.25, 2.0], "PATCH",
                         "26.1: +25% per level behind (enemy-team average, INFERRED-M), cap 2x; dragons/Elder/Baron"),
        "dragon_min_level": E(6, "PATCH", "26.1"),
        "elder_min_level": E(13, "PATCH", "26.1"),
        "baron_min_level": E(11, "PATCH", "26.1"),
        "herald_min_level": E(9, "PATCH", "26.1"),
        "grub_min_level": E(7, "PATCH", "26.1"),
        "monster_level_rounding": E("ceil", "WIKI", "average champion level rounded up (Voidgrub history)"),
        "baron_evolution_delay": E(30.0, "WIKI", "Delayed Evolution: levels only after 30 s out of combat"),
        "non_champion_damage_mult": E(1.5, "PATCH",
                                      "26.1: grubs/Herald deal 50% more to non-champions, matching Dragon and Baron"),
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
        "patience_drain": E(0.25, "INFERRED-L",
                            "patience fraction lost per second outside leash / without target (+ distance term)"),
        "monster_target": E("nearest champion", "WIKI", "aggroed monsters acquire the nearest champion regardless of "
                            "sight; Baron the nearest unit"),
    }


def build(variants_out: Path) -> dict:
    sources: dict = {}                                  # filled in read order
    characters = character_records(sources)
    shared = _load(CD / "shared.cdtb.bin.json", sources)
    eye = _load(CD / "items.cdtb.bin.json", sources)["Items/3513"]["mDataValues"]
    data = {"schema": "lanerl-modern-objectives-v1", "patch": PATCH, "client_build": CLIENT_BUILD,
            "evidence_tags": "CLIENT (16.19 bins/maps), WIKI (wiki.leagueoflegends.com 2026-10-02), "
                             "PATCH (26.1-26.19 notes), INFERRED-M/-L",
            "objectives_present": ["voidgrubs", "rift_herald", "elemental_drakes", "dragon_soul", "elder_dragon",
                                   "baron_nashor"],
            "objectives_absent": {"atakhan": "PATCH 26.1 removed", "blood_roses": "PATCH 26.1 removed",
                                  "feats_of_strength": "PATCH 26.1 removed"},
            "characters": characters,
            "shared_spells": {n: _data_values(shared["Shared/Spells/" + n]) for n in SHARED_SPELLS},
            "eye_of_the_herald": {v["mName"]: v["mValue"] for v in eye},
            "camps": camps(sources), "dragon_script": dragon_script(sources), "rules": rules(),
            "elements": list(ELEMENTS), "baron_forms": list(BARON_FORMS)}
    data["rift_terrain"] = rift_variants(sources, variants_out)
    data["sources_sha256"] = sources
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--variants-out", type=Path, default=RESEARCH / "rift-variants-26.19")
    p.add_argument("--json-out", type=Path, default=PATCH_DIR / "objectives_client.json")
    a = p.parse_args()
    data = build(a.variants_out)
    a.json_out.write_text(json.dumps(data, indent=1) + "\n")
    print(json.dumps(data["rift_terrain"]["changed_vs_base"], indent=1))


if __name__ == "__main__":
    main()
