"""Build the patch-pinned SR item table from 16.19.8230722 client data.

Host-side tool, not imported by the simulator. Reads the CommunityDragon
``16.19`` item bin (exactly client build 16.19.8230722), the decoded Map11
``map11.bin`` CLASSIC item lists and the en_US item names, and writes
``lanerl_jax/modern/data/26.19/items_client.json``.

Pool = CLASSIC ``GameModeMapData {0b03bf5a}`` item lists, restricted to
in-store items plus the four transform/quest-distributed items the runtime
needs (Seraph's 3040, Muramana 3042, Fimbulwinter 3121, Diadem 2530).
Jungle pets are included as data (they are starters/group members) but their
effects are deferred by MODERN-009.

    python -m lanerl_jax.modern.data.build_items [--research DIR] [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from . import PATCH_DIR

RESEARCH = Path("/mnt/nfs/shared/modern-world-map-research")
OUT = PATCH_DIR / "items_client.json"
CLASSIC_MAP_DATA = "{0b03bf5a}"
TRANSFORMS = (3040, 3042, 3121, 2530)
# Rune-distributed items (RUNES.md §7): Total Biscuit (Biscuit Delivery),
# Slightly Magical Boots (Magical Footwear), Elixirs of Skill/Avarice/Force
# (Triple Tonic). Not in store; granted into the inventory by the rune.
RUNE_GRANTED = (2010, 2422, 2150, 2151, 2152)

# Client ItemData stat field -> runtime stat name (docs/modern/DAMAGE_AND_STATS.md §3.4).
STAT_FIELDS = {
    "mFlatHPPoolMod": "health",
    "mFlatPhysicalDamageMod": "attack_damage",
    "mFlatMagicDamageMod": "ability_power",
    "mFlatArmorMod": "armor",
    "mFlatSpellBlockMod": "magic_resist",
    "mPercentAttackSpeedMod": "attack_speed",
    "mPercentMultiplicativeAttackSpeedMod": "multiplicative_attack_speed",
    "mFlatCritChanceMod": "crit_chance",
    "mFlatCritDamageMod": "crit_damage",
    "mPercentLifeStealMod": "life_steal",
    "PercentOmnivampMod": "omnivamp",
    "mFlatMovementSpeedMod": "move_speed",
    "mPercentMovementSpeedMod": "percent_move_speed",
    "mAbilityHasteMod": "ability_haste",
    "mFlatHPRegenMod": "health_regen",
    "mPercentBaseHPRegenMod": "percent_base_health_regen",
    "flatMPPoolMod": "mana",
    "percentBaseMPRegenMod": "percent_base_mana_regen",
    "PhysicalLethality": "lethality",
    "mPercentArmorPenetrationMod": "percent_armor_pen",
    "mFlatMagicPenetrationMod": "magic_pen",
    "mPercentMagicPenetrationMod": "percent_magic_pen",
    "mPercentTenacityItemMod": "tenacity",
    "mPercentSlowResistMod": "slow_resist",
    "mPercentHealingAmountMod": "heal_shield_power",
}
SPELL_FIELDS = ("mCastTime", "castConeDistance", "mCantCancelWhileWindingUp",
                "mCanMoveWhileChanneling", "mSpellTags", "mAffectsTypeFlags")


def fnv1a(text: str) -> str:
    h = 0x811C9DC5
    for b in text.lower().encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return "{%08x}" % h


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(research: Path) -> dict:
    cache = research / "cdragon-16.19"
    items_path = cache / "items.cdtb.bin.json"
    map_path = research / "map11-decoded.json"
    names_path = cache / "cdragon-items.json"
    bins = json.loads(items_path.read_text())
    map11 = json.loads(map_path.read_text())
    names = {row["id"]: row["name"] for row in json.loads(names_path.read_text())}

    by_hash = {}
    for key, rec in bins.items():
        if isinstance(rec, dict) and rec.get("__type") == "ItemData":
            by_hash[fnv1a(key)] = key
            by_hash[key] = key
    groups_by_key = {}
    for key, rec in bins.items():
        if isinstance(rec, dict) and rec.get("__type") == "ItemGroup":
            groups_by_key[key] = rec
            groups_by_key[fnv1a(key)] = rec

    pool = set()
    for list_key in map11[CLASSIC_MAP_DATA]["itemLists"]:
        for ref in map11[list_key]["mItems"]:
            if ref not in by_hash:
                raise RuntimeError(f"unresolved CLASSIC item reference {ref}")
            pool.add(by_hash[ref])

    def item_id(key) -> int:
        if isinstance(key, int):
            return key
        return int(bins[by_hash[key]]["itemID"])

    selected = {}
    for key in sorted(pool):
        rec = bins[key]
        iid = int(rec["itemID"])
        in_store = bool(rec.get("mItemDataAvailability", {}).get("mInStore", False))
        if in_store or iid in TRANSFORMS:
            selected[iid] = key
    for iid in RUNE_GRANTED:
        key = f"Items/{iid}"
        if bins.get(key, {}).get("itemID") != iid:
            raise RuntimeError(f"rune-granted item {iid} missing from the item bin")
        selected[iid] = key
    missing = [t for t in TRANSFORMS if t not in selected]
    if missing:
        raise RuntimeError(f"transform items missing from CLASSIC pool: {missing}")
    # Quest-distributed recipe components (support World Atlas line) are not
    # in store but must exist for recipe consumption and total cost.
    frontier = list(selected.values())
    while frontier:
        for ref in bins[frontier.pop()].get("recipeItemLinks", []):
            key = by_hash[ref]
            iid = int(bins[key]["itemID"])
            if iid not in selected:
                if key not in pool:
                    raise RuntimeError(f"recipe component {iid} outside CLASSIC pool")
                selected[iid] = key
                frontier.append(key)

    group_table = {}
    items = {}
    for iid, key in sorted(selected.items()):
        rec = bins[key]
        groups = []
        for ref in rec.get("mItemGroups", []):
            g = groups_by_key.get(ref)
            if g is None:
                raise RuntimeError(f"item {iid}: unresolved item group {ref}")
            gid = str(g["mItemGroupID"])
            if gid == "Default":
                continue
            groups.append(gid)
            entry = {"max_ownable": int(g.get("mMaxGroupOwnable", -1))}
            for field, dest in (("mPurchaseCooldown", "purchase_cooldown"),
                                ("mInventorySlotMin", "slot_min"),
                                ("mInventorySlotMax", "slot_max")):
                if field in g:
                    entry[dest] = g[field]
            group_table[gid] = entry
        stats = {dest: float(rec[src]) for src, dest in STAT_FIELDS.items() if src in rec}
        data_values = {dv["mName"]: float(dv.get("mValue", 0.0)) for dv in rec.get("mDataValues", [])}
        spell = None
        if rec.get("spellName"):
            for prefix in (f"Items/{iid}/Spells/", "Items/Spells/", "Shared/Spells/"):
                s = bins.get(prefix + rec["spellName"])
                if s and isinstance(s.get("mSpell"), dict):
                    sd = s["mSpell"]
                    spell = {"name": rec["spellName"]}
                    spell.update({f: sd[f] for f in SPELL_FIELDS if f in sd})
                    spell["uses_autoattack_cast_time"] = "mUseAutoattackCastTimeData" in sd
                    break
        avail = rec.get("mItemDataAvailability", {})
        items[str(iid)] = {
            "name": names.get(iid, rec.get("mDisplayName", str(iid))),
            "in_store": bool(avail.get("mInStore", False)),
            "price": int(rec.get("price", 0)),
            "recipe": [item_id(r) for r in rec.get("recipeItemLinks", [])],
            "sell_modifier": float(rec.get("sellBackModifier", 0.7)),
            "can_be_sold": bool(rec.get("mCanBeSold", False)),
            "max_stack": int(rec.get("maxStack", 1)),
            "consumed": bool(rec.get("consumed", False)),
            "consume_on_acquire": bool(rec.get("consumeOnAcquire", False)),
            "usable_in_store": bool(rec.get("usableInStore", False)),
            "clickable": bool(rec.get("clickable", False)),
            "epicness": int(rec.get("epicness", 0)),
            "groups": groups,
            "required_level": int(rec.get("mRequiredLevel", 0)),
            "required_champion": rec.get("mRequiredChampion", ""),
            "required_spell": rec.get("mRequiredSpellName", ""),
            "required_buff": rec.get("mRequiredBuffCurrencyName", ""),
            "required_identities": list(rec.get("mRequiredPurchaseIdentities", [])),
            "special_recipe": item_id(rec["specialRecipe"]) if rec.get("specialRecipe") else 0,
            "sidegrades": [item_id(r) for r in rec.get("sidegradeItemLinks", [])],
            "categories": list(rec.get("mCategories", [])),
            "stats": stats,
            "data_values": data_values,
            "effect_amount": [float(x) for x in rec.get("mEffectAmount", [])],
            "calculations": rec.get("mItemCalculations", {}),
            "spell": spell,
        }

    def total(iid: int) -> int:
        row = items[str(iid)]
        return row["price"] + sum(total(c) for c in row["recipe"])

    for iid, row in items.items():
        for c in row["recipe"]:
            if str(c) not in items:
                raise RuntimeError(f"item {iid}: recipe component {c} outside SR pool")
        row["total"] = total(int(iid))

    return {
        "schema": "lanerl-client-sr-items-v1",
        "patch": "26.19",
        "client_build": "16.19.8230722",
        "mode": "CLASSIC",
        "map_id": 11,
        "sources": {
            "items.cdtb.bin.json": sha256(items_path),
            "map11-decoded.json": sha256(map_path),
            "cdragon-items.json": sha256(names_path),
        },
        "groups": dict(sorted(group_table.items())),
        "items": items,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--research", type=Path, default=RESEARCH)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    payload = build(args.research)
    args.out.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"wrote {len(payload['items'])} items, {len(payload['groups'])} groups -> {args.out}")


if __name__ == "__main__":
    main()
