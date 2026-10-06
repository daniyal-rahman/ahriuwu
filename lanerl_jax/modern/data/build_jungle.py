"""Build ``26.19/jungle_client.json`` (SR jungle) from 16.19.8230722 client data.

Reads the CommunityDragon 16.19 monster character bins (``RESEARCH/jungle-16.19/``), the decoded Map11
materials bin (``NeutralCampGeComponentDef`` camp markers and monster placements, Team 300 = neutral), the
decoded ``map11.bin`` (``CampName`` objects, ``JungleLocationMapInformation`` indices) and
``shared.cdtb.bin.json`` (Smite, jungle-item kill heal, Crest of Insight/Cinders). Output is CLIENT data
only; wiki/patch-note values live in ``lanerl_jax/modern/jungle/camps.py``. Client (X, Z) = sim (x, y).

    python -m lanerl_jax.modern.data.build_jungle [--research DIR] [--out PATH]
"""
from __future__ import annotations

import json
from pathlib import Path

from . import CLIENT_BUILD, PATCH, PATCH_DIR, build_main, sha256, walk_dicts

CHARACTERS = ("SRU_Blue", "SRU_Red", "SRU_Gromp", "SRU_Murkwolf", "SRU_MurkwolfMini", "SRU_Razorbeak",
              "SRU_RazorbeakMini", "SRU_Krug", "SRU_KrugMini", "SRU_KrugMiniMini", "Sru_Crab")
REGULAR_CAMPS = ("Order Blue", "Order Red", "Order OwlBear", "Order Wolves", "Order Wraiths",
                 "Order Small Golems", "Chaos Blue", "Chaos Red", "Chaos OwlBear", "Chaos Wolves",
                 "Chaos Wraiths", "Chaos Small Golems", "Baron Crab", "Dragon Crab")
ROOT_FIELDS = {"hp": "baseHPModifiable", "attack_damage": "baseDamageModifiable", "armor": "baseArmorModifiable",
               "magic_resist": "baseMR", "move_speed": "baseMoveSpeedModifiable",
               "attack_speed": "attackSpeedModifiable", "attack_range": "attackRangeModifiable"}


def _base(v):
    return v.get("baseValue") if isinstance(v, dict) else v


def character_record(path: Path, name: str) -> dict:
    d = json.loads(path.read_text())
    r = d[next(k for k in d if k.lower() == f"characters/{name}/characterrecords/root".lower())]
    ba = r.get("basicAttack", {})
    attack_spells = {k.rsplit("/", 1)[-1]: {"missile_speed": v.get("mSpell", {}).get("missileSpeed"),
                                            "cast_frame": v.get("mSpell", {}).get("castFrame")}
                     for k, v in d.items()
                     if isinstance(v, dict) and v.get("__type") == "SpellObject" and "BasicAttack" in k}
    return {
        "character": r["mCharacterName"],
        **{dst: _base(r.get(src)) for dst, src in ROOT_FIELDS.items()},
        "attack_total_time": ba.get("mAttackTotalTime"),
        "attack_cast_time": ba.get("mAttackCastTime"),
        "attack_delay_cast_offset_percent": r.get("attackDelayCastOffsetPercent",
                                                  ba.get("mAttackDelayCastOffsetPercent")),
        "gold": r.get("goldGivenOnDeath"),
        "xp": r.get("expGivenOnDeath"),
        "gameplay_radius": r.get("overrideGameplayCollisionRadius"),
        "pathing_radius": r.get("pathfindingCollisionRadius"),
        "acquisition_range": r.get("acquisitionRange"),
        "untargetable_spawn_time": r.get("untargetableSpawnTime"),
        "minion_score": r.get("minionScoreValue"),
        "unit_tags": r.get("unitTagsString"),
        "attack_spells": attack_spells,
    }


def camps_and_placements(geometry: dict, map11: dict) -> list:
    names = {k: v["CampName"] for k, v in map11.items() if isinstance(v, dict) and "CampName" in v}
    index = {loc["name"].lower(): loc.get("Index") for v in map11.values()
             if isinstance(v, dict) and "JungleLocationInformation" in v for loc in v["JungleLocationInformation"]}
    camps, mons = [], []
    for o in walk_dicts(geometry):
        if "transform" not in o:
            continue
        t = o["transform"][3]
        if "NeutralCamp" in o:
            nc = o["NeutralCamp"]
            name = names.get(nc.get("{5a4ef4e7}"))
            if name in REGULAR_CAMPS:
                camps.append({"name": name, "camp_hash": nc.get("{5a4ef4e7}"), "x": t[0], "y": t[2],
                              "minimap_icon": nc.get("MinimapIcon"), "camp_level": nc.get("CampLevel"),
                              "scoreboard_timer": nc.get("ScoreboardTimer"),
                              "spawn_group": nc.get("{1f2e5fd0}", {}).get("{752ff961}"),
                              "index": index.get(name.lower())})
        elif "Character" in o:
            rec = o["Character"].get("CharacterRecord", "")
            ch = rec.split("/")[1] if rec.startswith("Characters/") else ""
            if ch in CHARACTERS:
                f = o["transform"]
                mons.append({"character": ch, "x": t[0], "y": t[2], "facing": [f[2][0], f[2][2]],
                             "name_hash": o.get("name")})
    camps.sort(key=lambda c: REGULAR_CAMPS.index(c["name"]))
    for c in camps:
        c["members"] = []
    for m in mons:
        best = min(camps, key=lambda c: (c["x"] - m["x"]) ** 2 + (c["y"] - m["y"]) ** 2)
        d = ((best["x"] - m["x"]) ** 2 + (best["y"] - m["y"]) ** 2) ** 0.5
        if d > 600.0:
            raise RuntimeError(f"placement {m} is {d:.0f} units from the nearest camp")
        best["members"].append(m)
    for c in camps:
        c["members"].sort(key=lambda m: (CHARACTERS.index(m["character"]), m["x"], m["y"]))
    return camps


def shared_values(shared: dict) -> dict:
    def dvs(name):
        sp = shared["Shared/Spells/" + name].get("mSpell", {})
        return {x["name"]: x["values"][0] for x in sp.get("DataValues", []) if isinstance(x, dict) and "values" in x}

    def crest(name):
        return {"data_values": dvs(name),
                "calculations": shared["Shared/Spells/" + name].get("mSpell", {}).get("mSpellCalculations", {})}

    sp = shared["Shared/Spells/SummonerSmite"]["mSpell"]
    return {
        "smite": {"data_values": dvs("SummonerSmite"), "cooldown": sp["cooldownTime"][0],
                  "max_ammo": sp["mMaxAmmo"][0], "ammo_recharge_time": sp["mAmmoRechargeTime"][0],
                  "cooldown_not_affected_by_cdr": sp.get("mCooldownNotAffectedByCDR", False),
                  "cast_range": sp["castRange"][0], "cast_radius": sp["castRadius"][0],
                  "cast_range_use_bounding_boxes": sp.get("castRangeUseBoundingBoxes", False),
                  "can_cast_while_disabled": sp.get("canCastWhileDisabled", False),
                  "forgiveness_range": sp["TargetingForgivenessDefinitions"][0]["ForgivenessRange"],
                  "required_unit_tags": sp["mRequiredUnitTags"]["mObjectTagList"],
                  "pet_dps": sp["mSpellCalculations"]["PetDPS"], "pet_hps": sp["mSpellCalculations"]["PetHPS"]},
        "monster_kill_heal": dvs("Monster_Heal_Mis"),
        "crest_of_insight": crest("CrestoftheAncientGolem"),
        "crest_of_cinders": crest("BlessingoftheLizardElder"),
    }


def build(research: Path) -> dict:
    char_paths = {c: research / "jungle-16.19" / f"{c.lower()}.bin.json" for c in CHARACTERS}
    geometry_path = research / "geometry-decoded.json"
    map_path = research / "map11-decoded.json"
    shared_path = research / "cdragon-16.19" / "shared.cdtb.bin.json"
    crab = json.loads(char_paths["Sru_Crab"].read_text())
    shrine = crab["Characters/Sru_Crab/Spells/Sru_CrabShrineStats"]["mSpell"]["mEffectAmount"]
    return {
        "schema": "modern-jungle-client/1", "patch": PATCH, "client_build": CLIENT_BUILD, "map_id": 11,
        "coordinates": "client (X, Z) = simulator (x, y)",
        "monsters": {c: character_record(p, c) for c, p in char_paths.items()},
        "camps": camps_and_placements(json.loads(geometry_path.read_text()), json.loads(map_path.read_text())),
        "scuttle_shrine_effect_amounts": [e.get("value", [None])[0] for e in shrine if "value" in e],
        **shared_values(json.loads(shared_path.read_text())),
        "sources": {p.name: sha256(p) for p in (*char_paths.values(), geometry_path, map_path, shared_path)},
    }


if __name__ == "__main__":
    build_main(__doc__, build, PATCH_DIR / "jungle_client.json",
               lambda p, out: f"wrote {out} ({len(p['camps'])} camps)", sort_keys=True, indent=1)
