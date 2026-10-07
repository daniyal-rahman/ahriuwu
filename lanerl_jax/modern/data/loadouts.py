"""Per-champion allow-lists: the items a champion may buy and its recommended rune pages.

Built outside the repo from LoLalytics 16.19 Emerald+ item sets (6-8 completed items per champion and role, top-2
boots, top-2 starting sets, their components, a global consumable list) and Riot's client rune recommendations
(``champion-rune-recommendations.json``, client 16.19); method and coverage in the file's ``meta`` and
``/mnt/nfs/shared/build-research/LOADOUTS.md``.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from ..runes import catalog as R

DEFAULT_LOADOUTS = Path("/mnt/nfs/shared/build-research/allowed_loadouts_16.19.json")


@lru_cache(None)
def _load(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def allowed_items(champion: str, role: str = "top", path: Path = DEFAULT_LOADOUTS) -> tuple[int, ...]:
    """Item ids ``champion`` may buy in ``role``: completed items, boots, starters, components and consumables."""
    data = _load(path)
    e = data["champions"][f"{champion.lower()}:{role}"]
    groups = (e["completed"], e["boots"], e["starters"], e["components"], e["transforms"], data["global_consumables"])
    return tuple(sorted({int(x[0]) for g in groups for x in g}))


def rune_pages(champion: str, role: str = "top", path: Path = DEFAULT_LOADOUTS) -> tuple[R.RunePage, ...]:
    """The 2-3 recommended pages of ``champion`` in ``role``, most played first."""
    first = lambda xs: tuple(int(x[0]) for x in xs)                                        # noqa: E731
    return tuple(R.RunePage(p["primary_tree"][0], p["keystone"][0], first(p["primary_minors"]),
                            p["secondary_tree"][0], first(p["secondary_minors"]), first(p["shards"]))
                 for p in _load(path)["champions"][f"{champion.lower()}:{role}"]["rune_pages"])
