"""Build ``26.19/runes_client.json`` (SR runes) from 16.19.8230722 client data.

Reads the CommunityDragon 16.19 perk bin and the en_US perk names. Holds the five PerkStyles (rows from
``mSlots``, allowed secondaries, default stat-mod sets), the three stat-shard slots, and per selectable perk
its style, row, CLASSIC ``mEffectAmount`` (``mEffectAmountGameMode`` overrides dropped) and ``mCalculations``.
Hash-named entries (e.g. ``{3ecd47e5}`` = Legend: Haste 9105) resolve through ``mPerkId``.

    python -m lanerl_jax.modern.data.build_runes [--research DIR] [--out PATH]
"""
from __future__ import annotations

import json
from pathlib import Path

from . import CLIENT_BUILD, PATCH, PATCH_DIR, build_main, sha256

STYLES = (8000, 8100, 8200, 8300, 8400)
SHARD_SLOTS = ("Perks/StatMods/Slots/OffensiveStats", "Perks/StatMods/Slots/FlexStats",
               "Perks/StatMods/Slots/DefensiveStats")


def build(research: Path) -> dict:
    perks_path = research / "cdragon-16.19" / "perks.cdtb.bin.json"
    names_path = research / "cdragon-16.19" / "cdragon-perks.json"
    bins = json.loads(perks_path.read_text())
    names = {row["id"]: row["name"] for row in json.loads(names_path.read_text())}

    def perk_ids(keys) -> list[int]:
        return [int(bins[k]["mPerkId"]) for k in keys]

    styles, perks = {}, {}
    for rec in bins.values():
        if not isinstance(rec, dict) or rec.get("__type") != "PerkStyle" or int(rec["mPerkStyleId"]) not in STYLES:
            continue
        sid = int(rec["mPerkStyleId"])
        rows = [perk_ids(slot["mPerks"]) for slot in rec["mSlots"]]
        styles[str(sid)] = {
            "name": rec["mPerkStyleName"], "rows": rows,
            "allowed_sub_styles": [int(s) for s in rec["mAllowedSubStyles"]],
            "default_stat_mods": {str(s["mStyleId"]): perk_ids(s["mPerks"])
                                  for s in rec.get("mDefaultStatModsPerSubStyle", [])},
        }
        for r, row in enumerate(rows):
            for pid in row:
                perks[str(pid)] = {"style": sid, "row": r}
    shard_slots = [perk_ids(bins[slot]["mPerks"]) for slot in SHARD_SLOTS]
    for slot, ids in enumerate(shard_slots):
        for pid in ids:
            perks.setdefault(str(pid), {"style": 0, "row": -1, "shard_slots": []})["shard_slots"].append(slot)

    by_id = {int(rec["mPerkId"]): rec for rec in bins.values()
             if isinstance(rec, dict) and rec.get("__type") == "Perk" and "mPerkId" in rec}
    for key, entry in perks.items():
        rec = by_id[int(key)]
        data = rec.get("mScript", {}).get("mSpellScriptData", {})
        entry.update({
            "name": names.get(int(key), rec["mPerkName"]),
            "bin_name": rec["mPerkName"],
            "enabled": bool(rec.get("mEnabled", True)),
            "stackable": bool(rec.get("mStackable", False)),
            "script": rec.get("mScript", {}).get("mSpellScriptName", ""),
            "effect_amount": {k: float(v) for k, v in data.get("mEffectAmount", {}).items()},
            "calculations": data.get("mCalculations", {}),
        })
    return {
        "schema": "lanerl-client-sr-runes-v1", "patch": PATCH, "client_build": CLIENT_BUILD,
        "sources": {p.name: sha256(p) for p in (perks_path, names_path)},
        "styles": styles,
        "shard_slots": shard_slots,
        "perks": dict(sorted(perks.items(), key=lambda kv: int(kv[0]))),
        "not_selectable": sorted(pid for pid in by_id if str(pid) not in perks),
    }


if __name__ == "__main__":
    build_main(__doc__, build, PATCH_DIR / "runes_client.json",
               lambda p, out: (f"wrote {out}: {sum(1 for q in p['perks'].values() if q['row'] >= 0)} runes, "
                               f"{len(p['shard_slots'])} shard slots"), indent=1)
