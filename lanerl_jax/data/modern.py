"""Pinned 26.19 champion data, extracted from Riot's shipped BIN records.

The small JSON snapshots are checked in so neither training nor tests fetch
`latest`. Data Dragon incorrectly reports zero AD growth in this patch; use
the BIN's damagePerLevelModifiable instead. Map/items/runes are separate scope.
"""
from functools import lru_cache
import json
from pathlib import Path

PATCH = "26.19"
IDS = {"Garen": 86, "Jax": 24}


@lru_cache(None)
def champion(name):
    if name not in IDS:
        raise ValueError(f"unsupported modern champion: {name}")
    return json.loads((Path(__file__).parent / "modern_26_19" / f"{name.lower()}.json").read_text())


def stat(name, key):
    return champion(name)["character"][key]["baseValue"]


def spell(name, slot):
    suffix = f"/{name}{slot}"
    if slot == "Passive":
        suffix = f"/{name}Passive"
    return next(v for k, v in champion(name)["spells"].items() if k.endswith(suffix))


def values(name, slot, key):
    return next(v["values"] for v in spell(name, slot)["values"] if v["name"] == key)


def cooldowns(name, slot):
    return spell(name, slot)["cooldown"]["values"][1:4 if slot == "R" else 6]
