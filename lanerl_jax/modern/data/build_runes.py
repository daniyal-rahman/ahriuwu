"""Build the patch-pinned SR rune table from 16.19.8230722 client data.

Host-side tool, not imported by the simulator. Reads the CommunityDragon
``16.19`` perk bin (exactly client build 16.19.8230722) and the en_US perk
names, and writes ``lanerl_jax/modern/data/26.19/runes_client.json``.

The table holds the five PerkStyles (rows from ``mSlots``, allowed
secondaries, default stat-mod sets), the three stat-shard slots, and for
every selectable perk its style, row, ``mEffectAmount`` (CLASSIC values;
``mEffectAmountGameMode`` mode overrides are dropped) and ``mCalculations``.
Hash-named entries (e.g. ``{3ecd47e5}`` = Legend: Haste 9105) are resolved
through ``mPerkId``.

    python -m lanerl_jax.modern.data.build_runes [--research DIR] [--out PATH]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from . import PATCH_DIR

RESEARCH = Path("/mnt/nfs/shared/modern-world-map-research")
OUT = PATCH_DIR / "runes_client.json"
STYLES = (8000, 8100, 8200, 8300, 8400)
SHARD_SLOTS = ("Perks/StatMods/Slots/OffensiveStats", "Perks/StatMods/Slots/FlexStats",
               "Perks/StatMods/Slots/DefensiveStats")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _clean(value):
    """Drop ``__type`` noise below the calculation root but keep part types."""
    if isinstance(value, dict):
        return {k: _clean(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_clean(v) for v in value]
    return value


def build(research: Path) -> dict:
    cache = research / "cdragon-16.19"
    perks_path = cache / "perks.cdtb.bin.json"
    names_path = cache / "cdragon-perks.json"
    bins = json.loads(perks_path.read_text())
    names = {row["id"]: row["name"] for row in json.loads(names_path.read_text())}

    def perk_id(key: str) -> int:
        return int(bins[key]["mPerkId"])

    styles, perks = {}, {}
    for key, rec in bins.items():
        if not isinstance(rec, dict) or rec.get("__type") != "PerkStyle":
            continue
        sid = int(rec["mPerkStyleId"])
        if sid not in STYLES:
            continue
        rows = [[perk_id(p) for p in slot["mPerks"]] for slot in rec["mSlots"]]
        styles[str(sid)] = {
            "name": rec["mPerkStyleName"],
            "rows": rows,
            "allowed_sub_styles": [int(s) for s in rec["mAllowedSubStyles"]],
            "default_stat_mods": {str(s["mStyleId"]): [perk_id(p) for p in s["mPerks"]]
                                  for s in rec.get("mDefaultStatModsPerSubStyle", [])},
        }
        for r, row in enumerate(rows):
            for pid in row:
                perks[str(pid)] = {"style": sid, "row": r}
    shard_slots = [[perk_id(p) for p in bins[slot]["mPerks"]] for slot in SHARD_SLOTS]
    for slot, ids in enumerate(shard_slots):
        for pid in ids:
            perks.setdefault(str(pid), {"style": 0, "row": -1, "shard_slots": []})
            perks[str(pid)]["shard_slots"].append(slot)

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
            "calculations": _clean(data.get("mCalculations", {})),
        })
    retired = sorted(pid for pid in by_id if str(pid) not in perks)
    return {
        "schema": "lanerl-client-sr-runes-v1",
        "patch": "26.19",
        "client_build": "16.19.8230722",
        "sources": {
            "perks.cdtb.bin.json": sha256(perks_path),
            "cdragon-perks.json": sha256(names_path),
        },
        "styles": styles,
        "shard_slots": shard_slots,
        "perks": dict(sorted(perks.items(), key=lambda kv: int(kv[0]))),
        "not_selectable": retired,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--research", type=Path, default=RESEARCH)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    payload = build(args.research)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=False) + "\n")
    n = sum(1 for p in payload["perks"].values() if p["row"] >= 0)
    print(f"wrote {args.out}: {n} runes, {len(payload['shard_slots'])} shard slots")


if __name__ == "__main__":
    main()
