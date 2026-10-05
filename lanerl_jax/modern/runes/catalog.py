"""Patch-26.19 SR rune catalog, page legality and automatic substitutions.

Host-side loading of ``runes_client.json`` (built by
``lanerl_jax.modern.data.build_runes`` from client build 16.19.8230722).
Rune *rows* (0..n-1, sorted by perk id; the 62 selectable runes and the 7
stat shards) index the (C, R) page-count matrix that effect kernels close
over, so the tick never dispatches on ids. Kernels key on perk **id**, never
on the Data Dragon key (8230 is Stormraider's Surge, DDragon key
``PhaseRush``; RUNES.md D-10).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..data import PATCH, PATCH_DIR

CLIENT_BUILD = "16.19.8230722"
DATA_PATH = PATCH_DIR / "runes_client.json"

PRECISION, DOMINATION, SORCERY, INSPIRATION, RESOLVE = 8000, 8100, 8200, 8300, 8400
STYLES = (PRECISION, DOMINATION, SORCERY, INSPIRATION, RESOLVE)

# Substitution targets (RUNES.md §2.2 rule 4).
AFTERSHOCK, GRASP = 8439, 8437
GLACIAL, FIRST_STRIKE = 8351, 8369
MANAFLOW, AXIOM = 8226, 8224
PRESENCE_OF_MIND, TRIUMPH = 8009, 9111
FLASHTRAPTION, CASH_BACK = 8306, 8321
ULTIMATE_HUNTER, RELENTLESS, TREASURE = 8106, 8105, 8135
NIMBUS = 8275

SHARD_ADAPTIVE, SHARD_AS, SHARD_AH = 5008, 5005, 5007
SHARD_MS, SHARD_HEALTH_SCALING = 5010, 5001
SHARD_HEALTH, SHARD_TENACITY = 5011, 5013
SHARD_NAMES = {"adaptive": SHARD_ADAPTIVE, "attack_speed": SHARD_AS, "ability_haste": SHARD_AH,
               "move_speed": SHARD_MS, "health_scaling": SHARD_HEALTH_SCALING,
               "health_flat": SHARD_HEALTH, "tenacity": SHARD_TENACITY}


@dataclass(frozen=True)
class RuneSpec:
    perk_id: int
    row: int
    name: str
    bin_name: str
    style: int              # 0 for stat shards
    slot_row: int           # 0 keystone, 1..3 minor rows, -1 shard
    shard_slots: tuple[int, ...]
    effect_amount: dict
    calculations: dict

    def ea(self, name: str, default: float | None = None) -> float:
        if name in self.effect_amount:
            return self.effect_amount[name]
        if default is None:
            raise KeyError(f"rune {self.perk_id} {self.name} has no effect amount {name!r}")
        return default


class RuneCatalog:
    def __init__(self, payload: dict):
        if payload.get("schema") != "lanerl-client-sr-runes-v1" or payload.get("patch") != PATCH \
                or payload.get("client_build") != CLIENT_BUILD:
            raise RuntimeError("modern rune catalog has wrong schema, patch or client build")
        self.sources = dict(payload["sources"])
        self.styles = {int(k): v for k, v in payload["styles"].items()}
        self.shard_slots = tuple(tuple(s) for s in payload["shard_slots"])
        self.not_selectable = frozenset(payload["not_selectable"])
        rows = sorted(payload["perks"].items(), key=lambda kv: int(kv[0]))
        self.ids = tuple(int(k) for k, _ in rows)
        self.row_of = {pid: r for r, pid in enumerate(self.ids)}
        self.specs = tuple(RuneSpec(
            perk_id=int(k), row=r, name=rec["name"], bin_name=rec["bin_name"], style=int(rec["style"]),
            slot_row=int(rec["row"]), shard_slots=tuple(rec.get("shard_slots", ())),
            effect_amount=dict(rec["effect_amount"]), calculations=rec["calculations"])
            for r, (k, rec) in enumerate(rows))
        self.by_id = {s.perk_id: s for s in self.specs}
        self.runes = tuple(s.perk_id for s in self.specs if s.slot_row >= 0)
        self.shards = tuple(s.perk_id for s in self.specs if s.slot_row < 0)

    def __getitem__(self, perk_id: int) -> RuneSpec:
        try:
            return self.by_id[int(perk_id)]
        except KeyError:
            raise KeyError(f"perk {perk_id} is not a selectable patch-{PATCH} SR rune or shard") from None

    def __contains__(self, perk_id: int) -> bool:
        return int(perk_id) in self.by_id

    def row(self, perk_id: int) -> int:
        return self.row_of[int(perk_id)]

    def ea(self, perk_id: int, name: str, default: float | None = None) -> float:
        return self[perk_id].ea(name, default)

    def style_row(self, style: int, slot_row: int) -> tuple[int, ...]:
        return tuple(self.styles[style]["rows"][slot_row])


@lru_cache(maxsize=1)
def rune_catalog() -> RuneCatalog:
    return RuneCatalog(json.loads(DATA_PATH.read_text()))


def ea(perk_id: int, name: str, default: float | None = None) -> float:
    return rune_catalog().ea(perk_id, name, default)


# ---- level scaling primitives (RUNES.md §1.1) --------------------------------

def lin(start: Any, end: Any, level: Any, *, scale_past_18: bool = True) -> Any:
    """``ByCharLevelInterpolation``: linear over 1..18, extrapolating past 18
    unless ``mScalePastDefaultMaxLevel=false`` (README X-1, RUNES U-01)."""
    lv = jnp.maximum(jnp.asarray(level, jnp.float32), 1.0)
    if not scale_past_18:
        lv = jnp.minimum(lv, 18.0)
    return start + (end - start) * (lv - 1.0) / 17.0


def lin_growth(start: Any, end: Any, level: Any) -> Any:
    """Interpolation with ``mScaleByStatProgressionMultiplier``: the fraction
    is the champion stat-growth curve ``g(L) = n(0.7025 + 0.0175 n)/17``."""
    n = jnp.maximum(jnp.asarray(level, jnp.float32) - 1.0, 0.0)
    return start + (end - start) * n * (0.7025 + 0.0175 * n) / 17.0


def breakpoints(level1: float, initial_per_level: float, points: tuple[tuple[int, float], ...],
                level: Any) -> Any:
    """``ByCharLevelBreakpoints``: per-level increments that change at breakpoints."""
    lv = jnp.asarray(level, jnp.float32)
    total = jnp.asarray(level1, jnp.float32)
    marks = [(2, initial_per_level)] + [(int(k), float(v)) for k, v in points]
    for i, (start, per) in enumerate(marks):
        stop = marks[i + 1][0] if i + 1 < len(marks) else 10 ** 6
        total = total + per * jnp.clip(lv - start + 1.0, 0.0, stop - start)
    return total


def level_table(values, level: Any) -> Any:
    """``ByCharLevelFormula``: client table indexed by level (index 0 unused)."""
    table = jnp.asarray(values, jnp.float32)
    return table[jnp.clip(jnp.asarray(level, jnp.int32), 0, table.shape[0] - 1)]


# ---- pages ------------------------------------------------------------------

@dataclass(frozen=True)
class RunePage:
    """One SR page: keystone + 3 primary minors, 2 secondary minors, 3 shards."""
    primary_style: int
    keystone: int
    primary: tuple[int, int, int]
    secondary_style: int
    secondary: tuple[int, int]
    shards: tuple[int, int, int]

    @property
    def perks(self) -> tuple[int, ...]:
        return (self.keystone, *self.primary, *self.secondary, *self.shards)

    @classmethod
    def from_perk_ids(cls, perk_ids, primary_style: int, secondary_style: int) -> "RunePage":
        """Client/DDragon order: keystone, 3 primary, 2 secondary, 3 shards."""
        ids = tuple(int(p) for p in perk_ids)
        if len(ids) != 9:
            raise ValueError("a rune page lists exactly 9 perks (4 primary, 2 secondary, 3 shards)")
        return cls(int(primary_style), ids[0], ids[1:4], int(secondary_style), ids[4:6], ids[6:9])


@dataclass(frozen=True)
class ChampionTraits:
    """Champion facts the client substitutions read (RUNES.md §2.2 rule 4)."""
    has_immobilize: bool
    resource: str = "mana"          # "mana" | "energy" | "none"
    special: str = ""               # Yorick, Bel'Veth, Samira, Elise, Jayce, Nidalee, Zoe
    flash_equipped: bool = True
    adaptive_physical: bool = True  # champion adaptive type for ties


def validate_page(page: RunePage) -> None:
    """Reject illegal pages (RUNES.md §2.2 rules 1–3); never silently fix."""
    cat = rune_catalog()
    if page.primary_style not in cat.styles:
        raise ValueError(f"unknown primary style {page.primary_style}")
    if page.secondary_style not in cat.styles:
        raise ValueError(f"unknown secondary style {page.secondary_style}")
    if page.secondary_style not in cat.styles[page.primary_style]["allowed_sub_styles"]:
        raise ValueError("secondary style must differ from the primary and be an allowed sub-style")
    for pid in page.perks:
        if pid in cat.not_selectable or pid not in cat:
            raise ValueError(f"perk {pid} is not selectable on SR at {PATCH}")
    if page.keystone not in cat.style_row(page.primary_style, 0):
        raise ValueError(f"keystone {page.keystone} is not in the primary tree's keystone row")
    for r, pid in enumerate(page.primary, start=1):
        if pid not in cat.style_row(page.primary_style, r):
            raise ValueError(f"primary rune {pid} is not in row {r} of style {page.primary_style}")
    rows = []
    for pid in page.secondary:
        spec = cat[pid]
        if spec.style != page.secondary_style:
            raise ValueError(f"secondary rune {pid} is not in style {page.secondary_style}")
        if spec.slot_row == 0:
            raise ValueError("a secondary tree cannot provide a keystone")
        rows.append(spec.slot_row)
    if len(set(rows)) != 2:
        raise ValueError("the two secondary runes must come from different rows")
    for slot, pid in enumerate(page.shards):
        if pid not in cat.shard_slots[slot]:
            raise ValueError(f"shard {pid} is not allowed in shard slot {slot + 1}")


def substitute(page: RunePage, traits: ChampionTraits) -> RunePage:
    """Client game-start substitutions, applied identically (rule 4)."""
    swap = {}
    if not traits.has_immobilize or traits.special == "Yorick":
        swap[AFTERSHOCK], swap[GLACIAL] = GRASP, FIRST_STRIKE
    if traits.resource != "mana":
        swap[MANAFLOW] = AXIOM
    if traits.resource == "none":
        swap[PRESENCE_OF_MIND] = TRIUMPH
    if not traits.flash_equipped:
        swap[FLASHTRAPTION] = CASH_BACK
    if traits.special == "Bel'Veth":
        swap[ULTIMATE_HUNTER] = RELENTLESS
    if traits.special == "Samira":
        swap[ULTIMATE_HUNTER] = TREASURE
    if traits.special in ("Elise", "Jayce", "Nidalee", "Zoe"):
        swap[AXIOM] = NIMBUS
    s = lambda pid: swap.get(pid, pid)
    return replace(page, keystone=s(page.keystone), primary=tuple(s(p) for p in page.primary),
                   secondary=tuple(s(p) for p in page.secondary))


def prepare_page(page: RunePage, traits: ChampionTraits) -> RunePage:
    """Validate the chosen page, then apply game-start substitutions."""
    validate_page(page)
    return substitute(page, traits)


def page_counts(pages) -> np.ndarray:
    """(C, R) int32 perk counts for prepared pages (``None`` = explicit empty
    page, the legacy no-runes ruleset)."""
    cat = rune_catalog()
    out = np.zeros((len(pages), len(cat.ids)), np.int32)
    for c, page in enumerate(pages):
        if page is None:
            continue
        for pid in page.perks:
            out[c, cat.row(pid)] += 1
    return out


def has_rune(page: Any, perk_id: int) -> Any:
    """(C,) bool from a (C, R) page-count matrix."""
    return page[:, rune_catalog().row(perk_id)] > 0


def rune_count(page: Any, perk_id: int) -> Any:
    return page[:, rune_catalog().row(perk_id)]


GAREN_DEFAULT_PAGE = RunePage(PRECISION, 8010, (9111, 9105, 8299), SORCERY, (8224, 8234),
                              (SHARD_ADAPTIVE, SHARD_ADAPTIVE, SHARD_HEALTH_SCALING))
