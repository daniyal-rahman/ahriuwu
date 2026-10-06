"""Pinned 26.19 Garen/Jax records extracted from Riot's shipped BIN files (no network at runtime).

Data Dragon reports zero AD growth in this patch; the BIN's ``damagePerLevelModifiable`` is used instead.
"""
import json
from functools import lru_cache

from . import PATCH_DIR

IDS = {"Garen": 86, "Jax": 24}


@lru_cache(None)
def champion(name):
    if name not in IDS:
        raise ValueError(f"unsupported modern champion: {name}")
    return json.loads((PATCH_DIR / "champions" / f"{name.lower()}.json").read_text())


def stat(name, key):
    return champion(name)["character"][key]["baseValue"]


def spell(name, slot):
    """Spell record of ``slot`` (Q/W/E/R/Passive)."""
    return next(v for k, v in champion(name)["spells"].items() if k.endswith(f"/{name}{slot}"))


def values(name, slot, key):
    return next(v["values"] for v in spell(name, slot)["values"] if v["name"] == key)


def cooldowns(name, slot):
    return spell(name, slot)["cooldown"]["values"][1:4 if slot == "R" else 6]
