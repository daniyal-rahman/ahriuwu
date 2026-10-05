"""Build the patch-pinned Summoner's Rift jungle table from 16.19.8230722 client data.

Host-side tool, not imported by the simulator. Reads

* the CommunityDragon ``16.19`` monster character bins (``Root`` character
  records of SRU_Blue, SRU_Red, SRU_Gromp, SRU_Murkwolf(+Mini),
  SRU_Razorbeak(+Mini), SRU_Krug/KrugMini/KrugMiniMini, Sru_Crab) cached in
  ``RESEARCH/jungle-16.19/`` (``curl`` from raw.communitydragon.org/16.19);
* the decoded Map11 materials bin (``geometry-decoded.json``: the
  ``NeutralCampGeComponentDef`` camp markers and the monster
  ``SkinCharacterGeComponentDef`` placements, Team 300 = neutral);
* the decoded ``map11.bin`` (``CampName`` objects keyed by the camp hash the
  markers reference, and the ``JungleLocationMapInformation`` camp indices);
* ``shared.cdtb.bin.json`` (``SummonerSmite`` data values and ammo, the
  jungle-item kill heal ``Monster_Heal_Mis``, the Crest of Insight / Crest of
  Cinders calculations);

and writes ``lanerl_jax/modern/data/26.19/jungle_client.json``. Everything in
the output is CLIENT data; wiki/patch-note values (respawn timers, level
scaling tables, leash radii, evolution thresholds) live in
``lanerl_jax/modern/jungle/camps.py`` with their evidence level.

World coordinates: client (X, Z) = simulator (x, y).

    python -m lanerl_jax.modern.data.build_jungle [--research DIR] [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import PATCH_DIR

RESEARCH = Path("/mnt/nfs/shared/modern-world-map-research")
OUT = PATCH_DIR / "jungle_client.json"
CHARACTERS = ("SRU_Blue", "SRU_Red", "SRU_Gromp", "SRU_Murkwolf", "SRU_MurkwolfMini", "SRU_Razorbeak",
              "SRU_RazorbeakMini", "SRU_Krug", "SRU_KrugMini", "SRU_KrugMiniMini", "Sru_Crab")
REGULAR_CAMPS = ("Order Blue", "Order Red", "Order OwlBear", "Order Wolves", "Order Wraiths",
                 "Order Small Golems", "Chaos Blue", "Chaos Red", "Chaos OwlBear", "Chaos Wolves",
                 "Chaos Wraiths", "Chaos Small Golems", "Baron Crab", "Dragon Crab")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _base(v):
    return v.get("baseValue") if isinstance(v, dict) else v


def character_record(path: Path, name: str) -> dict:
    d = json.loads(path.read_text())
    key = next(k for k in d if k.lower() == f"characters/{name}/characterrecords/root".lower())
    r = d[key]
    ba = r.get("basicAttack", {})
    attack_spells = {}
    for k, v in d.items():
        if isinstance(v, dict) and v.get("__type") == "SpellObject" and "BasicAttack" in k:
            sp = v.get("mSpell", {})
            attack_spells[k.rsplit("/", 1)[-1]] = {"missile_speed": sp.get("missileSpeed"),
                                                   "cast_frame": sp.get("castFrame")}
    return {
        "character": r["mCharacterName"],
        "hp": _base(r.get("baseHPModifiable")),
        "attack_damage": _base(r.get("baseDamageModifiable")),
        "armor": _base(r.get("baseArmorModifiable")),
        "magic_resist": _base(r.get("baseMR")),
        "move_speed": _base(r.get("baseMoveSpeedModifiable")),
        "attack_speed": _base(r.get("attackSpeedModifiable")),
        "attack_range": _base(r.get("attackRangeModifiable")),
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


def _walk_placements(o):
    if isinstance(o, dict):
        if "transform" in o:
            yield o
        for v in o.values():
            yield from _walk_placements(v)
    elif isinstance(o, list):
        for v in o:
            yield from _walk_placements(v)


def camps_and_placements(geometry: dict, map11: dict) -> list:
    names = {k: v["CampName"] for k, v in map11.items() if isinstance(v, dict) and "CampName" in v}
    index = {}
    for k, v in map11.items():
        if isinstance(v, dict) and "JungleLocationInformation" in v:
            for loc in v["JungleLocationInformation"]:
                index[loc["name"].lower()] = loc.get("Index")
    camps, mons = [], []
    for o in _walk_placements(geometry):
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
    def dvs(key):
        sp = shared[key].get("mSpell", {})
        return {x["name"]: x["values"][0] for x in sp.get("DataValues", []) if isinstance(x, dict) and "values" in x}

    def calcs(key):
        return shared[key].get("mSpell", {}).get("mSpellCalculations", {})

    sp = shared["Shared/Spells/SummonerSmite"]["mSpell"]
    return {
        "smite": {"data_values": dvs("Shared/Spells/SummonerSmite"), "cooldown": sp["cooldownTime"][0],
                  "max_ammo": sp["mMaxAmmo"][0], "ammo_recharge_time": sp["mAmmoRechargeTime"][0],
                  "cooldown_not_affected_by_cdr": sp.get("mCooldownNotAffectedByCDR", False),
                  "cast_range": sp["castRange"][0], "cast_radius": sp["castRadius"][0],
                  "cast_range_use_bounding_boxes": sp.get("castRangeUseBoundingBoxes", False),
                  "can_cast_while_disabled": sp.get("canCastWhileDisabled", False),
                  "forgiveness_range": sp["TargetingForgivenessDefinitions"][0]["ForgivenessRange"],
                  "required_unit_tags": sp["mRequiredUnitTags"]["mObjectTagList"],
                  "pet_dps": sp["mSpellCalculations"]["PetDPS"], "pet_hps": sp["mSpellCalculations"]["PetHPS"]},
        "monster_kill_heal": dvs("Shared/Spells/Monster_Heal_Mis"),
        "crest_of_insight": {"data_values": dvs("Shared/Spells/CrestoftheAncientGolem"),
                             "calculations": calcs("Shared/Spells/CrestoftheAncientGolem")},
        "crest_of_cinders": {"data_values": dvs("Shared/Spells/BlessingoftheLizardElder"),
                             "calculations": calcs("Shared/Spells/BlessingoftheLizardElder")},
    }


def build(research: Path) -> dict:
    jdir = research / "jungle-16.19"
    char_paths = {c: jdir / f"{c.lower()}.bin.json" for c in CHARACTERS}
    geometry_path = research / "geometry-decoded.json"
    map_path = research / "map11-decoded.json"
    shared_path = research / "cdragon-16.19" / "shared.cdtb.bin.json"
    geometry = json.loads(geometry_path.read_text())
    map11 = json.loads(map_path.read_text())
    shared = json.loads(shared_path.read_text())
    crab = json.loads(char_paths["Sru_Crab"].read_text())
    shrine = crab["Characters/Sru_Crab/Spells/Sru_CrabShrineStats"]["mSpell"]["mEffectAmount"]
    sources = {p.name: sha256(p) for p in char_paths.values()}
    sources.update({geometry_path.name: sha256(geometry_path), map_path.name: sha256(map_path),
                    shared_path.name: sha256(shared_path)})
    return {
        "schema": "modern-jungle-client/1",
        "patch": "26.19",
        "client_build": "16.19.8230722",
        "map_id": 11,
        "coordinates": "client (X, Z) = simulator (x, y)",
        "monsters": {c: character_record(p, c) for c, p in char_paths.items()},
        "camps": camps_and_placements(geometry, map11),
        "scuttle_shrine_effect_amounts": [e.get("value", [None])[0] for e in shrine if "value" in e],
        **shared_values(shared),
        "sources": sources,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--research", type=Path, default=RESEARCH)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    payload = build(args.research)
    args.out.write_text(json.dumps(payload, sort_keys=True, indent=1) + "\n")
    print(f"wrote {args.out} ({len(payload['camps'])} camps)")


if __name__ == "__main__":
    main()
