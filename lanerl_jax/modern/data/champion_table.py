"""The 26.19 champion table (``build_champion_table``): identity index, base stats and per-ability descriptors
for every champion of the build, as arrays the ``modern-world-v2`` observation gathers by champion index.

Row 0 of every array is "unknown" (zeros; cooldowns at the client default). Champion ``i`` (1..N) is
``champions()[i - 1]``; ``index_of(name)`` resolves a Data Dragon id, display name or WAD name; ``RIOT_LOOKUP``
maps a Riot key (the sim kits' ``ID``) to the index.
"""
from __future__ import annotations

import json
from functools import lru_cache

import numpy as np

from . import PATCH_DIR
from .champions import IDS as KIT_IDS, cooldowns as kit_cooldowns

TABLE = PATCH_DIR / "champion_table.json"
CAST_TYPES = ("self", "unit", "point", "direction")
FLAGS = ("dash", "cc", "knockback", "shield", "attack_reset", "aoe", "recast")
RESOURCES = ("mana", "energy", "other")
#: Per-ability descriptor columns (16): cast type one-hot, range/1000, cooldown rank 1 /60, cooldown max rank /60,
#: cast time, 7 flags, missile speed/2000.
SLOT_COLUMNS = (*(f"cast_{c}" for c in CAST_TYPES), "range", "cd_rank1", "cd_max", "cast_time", *FLAGS, "missile")
#: Per-side base columns (9).
BASE_COLUMNS = ("attack_range", "move_speed", "hp", "ad", "armor", "mr", *(f"resource_{r}" for r in RESOURCES))
SIDE_DIM = len(BASE_COLUMNS) + 4 * len(SLOT_COLUMNS)        # 73
MAX_RANKS = 6                                               # cooldown table width (Jayce has 6 ranks)
#: Ranks per slot; the client's 7-wide tables carry values past the last real rank (Jax R: 110/100/90 then 8/7/7).
RANK_CAP = {"Jayce": (6, 6, 6, 1)}
DEFAULT_RANK_CAP = (5, 5, 5, 3)
DEFAULT_COOLDOWN = 10.0
#: Skill tier the 16.19 archive is recorded as (apex-sourced lobbies; the exact tier per player is unknown).
ARCHIVE_TIER = 8                                            # MASTER
N_TIERS = 10                                                # 0 unknown, 1 Iron .. 9 Challenger
TIERS = ("unknown", "IRON", "BRONZE", "SILVER", "GOLD", "PLATINUM", "EMERALD", "DIAMOND", "MASTER", "GRANDMASTER",
         "CHALLENGER")[:N_TIERS + 1]


def side_columns(prefix: str) -> list[str]:
    return [f"{prefix}_{c}" for c in BASE_COLUMNS] + [f"{prefix}_{s}_{c}" for s in "qwer" for c in SLOT_COLUMNS]


@lru_cache(maxsize=None)
def table() -> dict:
    if not TABLE.exists():
        raise FileNotFoundError(f"{TABLE} missing: run lanerl_jax.modern.data.build_champion_table")
    return json.loads(TABLE.read_text())


def champions() -> list[dict]:
    return table()["champions"]


def n_champions() -> int:
    return len(champions())


@lru_cache(maxsize=None)
def _names() -> dict[str, int]:
    out = {}
    for row in champions():
        for key in (row["ddragon_id"], row["name"], (row["catalog"] or {}).get("wad_name")):
            if key:
                out[key.lower().replace("'", "").replace(" ", "").replace(".", "")] = row["index"]
    return out


def index_of(name: str) -> int:
    """Champion index of a Data Dragon id / display name / WAD name; 0 when unknown."""
    return _names().get(name.lower().replace("'", "").replace(" ", "").replace(".", ""), 0)


@lru_cache(maxsize=None)
def riot_lookup() -> np.ndarray:
    """(max_riot_key,) int32: Riot key -> champion index (0 where no champion has the key)."""
    out = np.zeros((table()["max_riot_key"],), np.int32)
    for row in champions():
        out[row["riot_key"]] = row["index"]
    return out


def tier_index(tier: str | None) -> int:
    t = (tier or "").upper()
    return TIERS.index(t) if t in TIERS else 0


def rank_cap(row: dict) -> tuple:
    return RANK_CAP.get(row["ddragon_id"], DEFAULT_RANK_CAP)


def _cooldown_ranks(d: dict | None, cap: int, kit: tuple | None = None) -> list[float]:
    """Cooldown at ranks 1..cap (the last known value repeats), then repeated to MAX_RANKS. ``kit`` =
    (champion name, slot) reads the sim kit's own table for the 14 implemented champions, so the sim and the
    table agree (two catalogs lack a root Cooldown field: Camille W, Ezreal R)."""
    if kit is not None and kit[0] in KIT_IDS:
        cd = [float(v) for v in kit_cooldowns(kit[0], kit[1])]
    else:
        cd = list((d or {}).get("cooldown_by_rank") or []) or [DEFAULT_COOLDOWN]
    cd = (cd + [cd[-1]] * MAX_RANKS)[:cap]
    return (cd + [cd[-1]] * MAX_RANKS)[:MAX_RANKS]


def _slot_row(d: dict | None, cap: int, kit: tuple | None = None) -> list[float]:
    if d is None:
        return [0.0] * len(SLOT_COLUMNS)
    cd = _cooldown_ranks(d, cap, kit)
    # Clamped: the client keeps sentinel values on a few spells (Swain Q missile speed 1e9, Smolder R range 4200).
    return [*(float(d["cast_type"] == c) for c in CAST_TYPES), min(d["range"], 3000.0) / 1000.0,
            min(cd[0], 300.0) / 60.0, min(cd[cap - 1], 300.0) / 60.0, min(d["cast_time"], 3.0),
            *(float(d[f]) for f in FLAGS), min(d["missile_speed"], 5000.0) / 2000.0]


@lru_cache(maxsize=None)
def descriptors() -> np.ndarray:
    """(N + 1, SIDE_DIM) float32 descriptor block per champion index; row 0 zeros."""
    rows = [np.zeros((SIDE_DIM,), np.float32)]
    for row in champions():
        st = row["stats"]
        base = [st["attack_range"] / 600.0, st["move_speed"] / 500.0, st["hp"] / 1000.0, st["ad"] / 100.0,
                st["armor"] / 100.0, st["mr"] / 100.0, *(float(row["resource"] == r) for r in RESOURCES)]
        slots = row["slots"] or [None] * 4
        rows.append(np.asarray(base + sum((_slot_row(d, c, (row["ddragon_id"], s))
                                           for s, (d, c) in zip("QWER", zip(slots, rank_cap(row)))), []), np.float32))
    out = np.stack(rows)
    assert out.shape == (n_champions() + 1, SIDE_DIM)
    return out


@lru_cache(maxsize=None)
def cooldowns() -> np.ndarray:
    """(N + 1, 4, MAX_RANKS) float32 base cooldown by rank (the last known rank repeats); row 0 the default."""
    out = np.full((n_champions() + 1, 4, MAX_RANKS), DEFAULT_COOLDOWN, np.float32)
    for row in champions():
        for s, (d, cap) in enumerate(zip(row["slots"] or [], rank_cap(row))):
            out[row["index"], s] = _cooldown_ranks(d, cap, (row["ddragon_id"], "QWER"[s]))
    return out


@lru_cache(maxsize=None)
def cast_types() -> np.ndarray:
    """(N + 1, 4) int8 cast type per slot (index into CAST_TYPES; unknown champions: point)."""
    out = np.full((n_champions() + 1, 4), CAST_TYPES.index("point"), np.int8)
    for row in champions():
        for s, d in enumerate(row["slots"] or []):
            out[row["index"], s] = CAST_TYPES.index(d["cast_type"])
    return out
