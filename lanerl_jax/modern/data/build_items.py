"""Build ``26.19/items_client.json`` (SR items) from 16.19.8230722 client data.

Reads the CommunityDragon 16.19 item bin, the decoded Map11 CLASSIC item lists and the en_US names. Pool =
CLASSIC ``GameModeMapData {0b03bf5a}`` item lists, restricted to in-store items plus the transform items
(Seraph's 3040, Muramana 3042, Fimbulwinter 3121, Diadem 2530), the rune-granted items and their recipe
components. Jungle pets are kept as data; their effects are deferred (MODERN-009).

    python -m lanerl_jax.modern.data.build_items [--research DIR] [--out PATH]
"""
from __future__ import annotations

import json
from pathlib import Path

from . import CLIENT_BUILD, PATCH, PATCH_DIR, build_main, sha256

CLASSIC_MAP_DATA = "{0b03bf5a}"
TRANSFORMS = (3040, 3042, 3121, 2530)
# Granted by runes, not in store (RUNES.md §7): Total Biscuit, Slightly Magical Boots, Elixirs of
# Skill/Avarice/Force (Triple Tonic).
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
GROUP_FIELDS = {"mPurchaseCooldown": "purchase_cooldown", "mInventorySlotMin": "slot_min",
                "mInventorySlotMax": "slot_max"}
SPELL_FIELDS = ("mCastTime", "castConeDistance", "mCantCancelWhileWindingUp",
                "mCanMoveWhileChanneling", "mSpellTags", "mAffectsTypeFlags")


def fnv1a(text: str) -> str:
    h = 0x811C9DC5
    for b in text.lower().encode():
        h = ((h ^ b) * 0x01000193) & 0xFFFFFFFF
    return "{%08x}" % h


def spell_record(bins: dict, iid: int, name: str) -> dict | None:
    for prefix in (f"Items/{iid}/Spells/", "Items/Spells/", "Shared/Spells/"):
        s = bins.get(prefix + name)
        if s and isinstance(s.get("mSpell"), dict):
            sd = s["mSpell"]
            return {"name": name, **{f: sd[f] for f in SPELL_FIELDS if f in sd},
                    "uses_autoattack_cast_time": "mUseAutoattackCastTimeData" in sd}
    return None


def build(research: Path) -> dict:
    items_path = research / "cdragon-16.19" / "items.cdtb.bin.json"
    map_path = research / "map11-decoded.json"
    names_path = research / "cdragon-16.19" / "cdragon-items.json"
    bins = json.loads(items_path.read_text())
    map11 = json.loads(map_path.read_text())
    names = {row["id"]: row["name"] for row in json.loads(names_path.read_text())}

    def of_type(t):
        return {k: rec for k, rec in bins.items() if isinstance(rec, dict) and rec.get("__type") == t}
    by_hash = {h: key for key in of_type("ItemData") for h in (fnv1a(key), key)}
    groups_by_key = {h: rec for key, rec in of_type("ItemGroup").items() for h in (key, fnv1a(key))}

    def item_id(key) -> int:
        return key if isinstance(key, int) else int(bins[by_hash[key]]["itemID"])

    pool = set()
    for list_key in map11[CLASSIC_MAP_DATA]["itemLists"]:
        for ref in map11[list_key]["mItems"]:
            if ref not in by_hash:
                raise RuntimeError(f"unresolved CLASSIC item reference {ref}")
            pool.add(by_hash[ref])
    selected = {}
    for key in sorted(pool):
        iid = int(bins[key]["itemID"])
        if bins[key].get("mItemDataAvailability", {}).get("mInStore", False) or iid in TRANSFORMS:
            selected[iid] = key
    for iid in RUNE_GRANTED:
        key = f"Items/{iid}"
        if bins.get(key, {}).get("itemID") != iid:
            raise RuntimeError(f"rune-granted item {iid} missing from the item bin")
        selected[iid] = key
    missing = [t for t in TRANSFORMS if t not in selected]
    if missing:
        raise RuntimeError(f"transform items missing from CLASSIC pool: {missing}")
    # Quest-distributed recipe components (support World Atlas line) are not in store but must exist for
    # recipe consumption and total cost.
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

    group_table, items = {}, {}
    for iid, key in sorted(selected.items()):
        rec = bins[key]
        groups = []
        for ref in rec.get("mItemGroups", []):
            g = groups_by_key.get(ref)
            if g is None:
                raise RuntimeError(f"item {iid}: unresolved item group {ref}")
            gid = str(g["mItemGroupID"])
            if gid != "Default":
                groups.append(gid)
                group_table[gid] = {"max_ownable": int(g.get("mMaxGroupOwnable", -1)),
                                    **{dest: g[src] for src, dest in GROUP_FIELDS.items() if src in g}}
        items[str(iid)] = {
            "name": names.get(iid, rec.get("mDisplayName", str(iid))),
            "in_store": bool(rec.get("mItemDataAvailability", {}).get("mInStore", False)),
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
            "stats": {dest: float(rec[src]) for src, dest in STAT_FIELDS.items() if src in rec},
            "data_values": {dv["mName"]: float(dv.get("mValue", 0.0)) for dv in rec.get("mDataValues", [])},
            "effect_amount": [float(x) for x in rec.get("mEffectAmount", [])],
            "calculations": rec.get("mItemCalculations", {}),
            "spell": spell_record(bins, iid, rec["spellName"]) if rec.get("spellName") else None,
        }

    def total(iid: int) -> int:
        row = items[str(iid)]
        return row["price"] + sum(total(c) for c in row["recipe"])

    for iid, row in items.items():
        for c in row["recipe"]:
            if str(c) not in items:
                raise RuntimeError(f"item {iid}: recipe component {c} outside SR pool")
        row["total"] = total(int(iid))
    return {"schema": "lanerl-client-sr-items-v1", "patch": PATCH, "client_build": CLIENT_BUILD, "mode": "CLASSIC",
            "map_id": 11, "sources": {p.name: sha256(p) for p in (items_path, map_path, names_path)},
            "groups": dict(sorted(group_table.items())), "items": items}


if __name__ == "__main__":
    build_main(__doc__, build, PATCH_DIR / "items_client.json",
               lambda p, out: f"wrote {len(p['items'])} items, {len(p['groups'])} groups -> {out}",
               sort_keys=True, separators=(",", ":"))
