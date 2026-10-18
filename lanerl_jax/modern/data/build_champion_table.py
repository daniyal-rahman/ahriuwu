"""Build ``26.19/champion_table.json``: one row per champion of the 16.19 build with the identity index, base
stats and per-ability kit descriptors the ``modern-world-v2`` observation reads (docs/modern/MULTICHAMP.md §3-4).

    python -m lanerl_jax.modern.data.build_champion_table [--catalogs DIR] [--ddragon FILE] [--out FILE]

Sources: the decoder's saved-client champion catalogs (``<DIR>/<Name>.json``, schema ``rofl_champion_assets_v1``)
for ability definitions and Data Dragon 16.19.1 for base stats, Riot keys and resource type. Index 0 is reserved
for "unknown"; champions are numbered 1.. in Riot-key order. A champion in Data Dragon without a catalog gets a row
with ``catalog = null`` and zero descriptors, so the index space is complete either way.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import PATCH_DIR, sha256

CATALOGS = Path("/mnt/nfs/datasets/rofl_champion_catalogs_2619_20261018/champions")
DDRAGON = Path("/mnt/nfs/shared/build-research/raw/ddragon/champion.json")
OUT = PATCH_DIR / "champion_table.json"
SLOTS = "QWER"
CAST_TYPES = ("self", "unit", "point", "direction")
#: ``cast_mode`` values the catalog uses -> our four cast types (``None`` = unknown, treated as point).
CAST_MODE = {"self": "self", "unit": "unit", "point": "point", "direction": "direction", None: "point"}
TAG_FLAGS = dict(dash=("PositiveEffect_MoveBlock", "Trait_PlayerSelectedDashDirection"),
                 cc=("Trait_ImmobilizingCCSpell", "Trait_ImmobilizingCCAbility"),
                 knockback=("Trait_KnockBack",), shield=("Trait_Shield",), attack_reset=("Trait_AttackReset",),
                 aoe=("Trait_AoE",), recast=("Trait_RecastOrReplaceSpell",))
FLAGS = tuple(TAG_FLAGS)
RESOURCES = ("mana", "energy", "other")
#: Data Dragon ``partype`` -> resource class.
RESOURCE = {"Mana": "mana", "Energy": "energy"}
MAX_RIOT_KEY = 1024
RANGE_SENTINEL = 5000.0      # cast ranges at or above this are the client's "no bound" value, not a range
DEFAULT_COOLDOWN = 10.0      # a spell without a Cooldown field has the client default (modern.data.champions)


def _value(field):
    return field.get("value") if isinstance(field, dict) else field


def root_spell(catalog: dict, slot: int) -> dict | None:
    """The slot's root spell row: the one whose slot provenance names the slot's ability object."""
    entry = _value(catalog["slots"][slot])
    hits = [sp for sp in catalog["spells"] if _value(sp["slot"]) == slot
            and any(p.get("entry") == entry for p in sp["slot"].get("provenance", []))]
    if len(hits) != 1:
        raise ValueError(f"{_value(catalog['champion'])}: slot {slot} has {len(hits)} root spells")
    return hits[0]


def slot_descriptor(catalog: dict, slot: int) -> dict:
    root = root_spell(catalog, slot)
    children = [sp for sp in catalog["spells"] if _value(sp["slot"]) == slot]
    tags = set()
    for sp in children:
        raw = _value(sp.get("raw_spell_definition")) or {}
        tags.update(raw.get("mSpellTags") or [])
    cd = _value(root["base_cooldown_seconds_by_rank"]) or {}
    ranks = sorted(int(k) for k in cd) if cd else []
    rng = _value(root["configured_range_rank_table"]) or []
    mode = _value(root["cast_mode"])
    if mode not in CAST_MODE:
        raise ValueError(f"{_value(catalog['champion'])}: slot {slot} cast mode {mode!r}")
    rng0 = float(rng[0]) if rng else 0.0
    cooldown = [float(cd[str(r)]) for r in ranks]
    return dict(name=_value(root["name"]), cast_type=CAST_MODE[mode], cast_mode_raw=mode,
                range=0.0 if rng0 >= RANGE_SENTINEL else rng0,     # 25000 = "unbounded" on directional spells
                cooldown_by_rank=cooldown or [DEFAULT_COOLDOWN] * 5,
                cooldown_source="catalog" if cooldown else "default",
                cast_time=float(_value(root["cast_time_seconds"]) or 0.0),
                missile_speed=float(_value(root["missile_speed_world_units_per_second"]) or 0.0),
                **{f: bool(tags & set(TAG_FLAGS[f])) for f in FLAGS}, tags=sorted(tags))


def build(catalogs: Path, ddragon: Path) -> dict:
    dd = json.loads(ddragon.read_text())
    wads = {}
    for p in catalogs.glob("*.json"):
        wads[p.stem.lower()] = p
    rows = []
    for ddragon_id, rec in sorted(dd["data"].items(), key=lambda kv: int(kv[1]["key"])):
        st = rec["stats"]
        row = dict(ddragon_id=ddragon_id, name=rec["name"], riot_key=int(rec["key"]), catalog=None,
                   resource=RESOURCE.get(rec["partype"], "other"), partype=rec["partype"], tags=rec["tags"],
                   stats=dict(attack_range=st["attackrange"], move_speed=st["movespeed"], hp=st["hp"],
                              ad=st["attackdamage"], armor=st["armor"], mr=st["spellblock"],
                              attack_speed=st["attackspeed"]),
                   slots=[])
        path = wads.get(ddragon_id.lower())
        if path is not None:
            cat = json.loads(path.read_text())
            row.update(catalog=dict(file=path.name, sha256=sha256(path), wad_name=_value(cat["champion"]),
                                    build=cat.get("build")),
                       slots=[slot_descriptor(cat, s) for s in range(4)])
        rows.append(row)
    for i, row in enumerate(rows, start=1):
        row["index"] = i
    if any(r["riot_key"] >= MAX_RIOT_KEY for r in rows):
        raise ValueError("a Riot key exceeds the lookup size")
    return dict(schema="modern_champion_table_v1", ddragon=dict(file=str(ddragon), version=dd["version"],
                                                                 sha256=sha256(ddragon)),
                catalogs=str(catalogs), n_champions=len(rows), with_catalog=sum(r["catalog"] is not None for r in rows),
                cast_types=list(CAST_TYPES), flags=list(FLAGS), resources=list(RESOURCES),
                max_riot_key=MAX_RIOT_KEY, champions=rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--catalogs", type=Path, default=CATALOGS)
    p.add_argument("--ddragon", type=Path, default=DDRAGON)
    p.add_argument("--out", type=Path, default=OUT)
    a = p.parse_args(argv)
    table = build(a.catalogs, a.ddragon)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(table, indent=1))
    print(f"{table['n_champions']} champions, {table['with_catalog']} with a catalog -> {a.out}")


if __name__ == "__main__":
    main()
