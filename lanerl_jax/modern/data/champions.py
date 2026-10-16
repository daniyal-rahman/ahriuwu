"""Pinned 26.19 champion records extracted from Riot's shipped BIN files (no network at runtime; regenerate with
``build_champions``).

Data Dragon reports zero AD growth in this patch; the BIN's ``damagePerLevelModifiable`` is used instead.
"""
import json
from functools import lru_cache

from . import PATCH_DIR

IDS = {"Garen": 86, "Jax": 24, "Nasus": 75, "Renekton": 58, "Darius": 122, "Malphite": 54, "Aatrox": 266, "Fiora": 114, "Yone": 777, "Ambessa": 799, "Camille": 164, "KSante": 897,
       "Jayce": 126, "Ezreal": 81}


@lru_cache(None)
def champion(name):
    if name not in IDS:
        raise ValueError(f"unsupported modern champion: {name}")
    return json.loads((PATCH_DIR / "champions" / f"{name.lower()}.json").read_text())


def stat(name, key):
    return champion(name)["character"][key]["baseValue"]


def spell(name, slot):
    """Spell record of ``slot`` (Q/W/E/R/Passive)."""
    rec = champion(name)
    return rec["spells"][rec["slots"][slot]]


def values(name, slot, key):
    return next(v["values"] for v in spell(name, slot)["values"] if v["name"] == key)


# Ranks per slot where the client's rank-up table differs from 5/5/5/3 (Jayce: 6 per basic, an innate R).
MAX_RANKS = {"Jayce": (6, 6, 6, 1)}


def max_ranks(name):
    return MAX_RANKS.get(name, (5, 5, 5, 3))


# BINs omit fields left at their default; a spell without ``Cooldown`` has the 10 s default (Jayce's Lightning Field,
# wiki 10).
DEFAULT_COOLDOWN = 10.0


def cooldowns(name, slot):
    """Base cooldowns of ranks 1.. (a second-form slot ``Q2`` has its base slot's ranks)."""
    ranks = max_ranks(name)["QWER".index(slot[0])]
    rec = spell(name, slot).get("cooldown")
    return [DEFAULT_COOLDOWN] * ranks if rec is None else rec["values"][1:1 + ranks]
