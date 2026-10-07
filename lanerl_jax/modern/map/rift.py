"""Elemental Rift and Baron-pit terrain variants (client navgrid overlays, OBJECTIVES §6) as gatherable arrays.

``variant = element * 3 + baron_form``: element 0 none, 1 Infernal, 2 Mountain, 3 Ocean, 4 Cloud, 5 Hextech,
6 Chemtech; baron_form 0 Hunting, 1 Territorial, 2 All-Seeing. Built by ``data/build_modern_objectives.py``.
"""
from __future__ import annotations

import hashlib
import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import jax.numpy as jnp
import numpy as np

from ..data import PATCH_DIR
from .terrain import StaticTerrain, row_gaps

TABLE = PATCH_DIR / "objectives_client.json"
N_ELEMENTS = 7
N_BARON_FORMS = 3


class RiftTerrain(NamedTuple):
    walkable: Any       # (V, 2, H, W) bool per team
    gaps: Any           # row_gaps(walkable)
    flags: Any          # (V, H, W) int32 navgrid flags
    bush_ids: Any       # (V, H, W) int32 brush labels per variant (0 = none)
    cell_size: float
    min_x: float
    min_z: float
    max_x: float
    max_z: float


def variant_index(element, baron_form) -> Any:
    return (jnp.clip(jnp.asarray(element, jnp.int32), 0, N_ELEMENTS - 1) * N_BARON_FORMS
            + jnp.clip(jnp.asarray(baron_form, jnp.int32), 0, N_BARON_FORMS - 1)).astype(jnp.int32)


@lru_cache(maxsize=2)
def load_rift_terrain(artifact: str | None = None) -> RiftTerrain:
    """Load the variant artifact and check it against the pin in ``objectives_client.json``."""
    pin = json.loads(TABLE.read_text())["rift_terrain"]
    path = Path(artifact or pin["artifact"])
    raw = (path / "variants.npz").read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin["arrays_sha256"]:
        raise ValueError("rift variant arrays checksum mismatch")
    if hashlib.sha256((path / "manifest.json").read_bytes()).hexdigest() != pin["manifest_sha256"]:
        raise ValueError("rift variant manifest checksum mismatch")
    m = json.loads((path / "manifest.json").read_text())
    with np.load(io.BytesIO(raw), allow_pickle=False) as a:
        walk, flags, bush = a["walkable"], a["flags"], a["bush_ids"]
    if walk.shape[0] != N_ELEMENTS * N_BARON_FORMS:
        raise ValueError("unexpected variant count")
    return RiftTerrain(jnp.asarray(walk), row_gaps(walk), jnp.asarray(flags.astype(np.int32)), jnp.asarray(bush),
                       float(m["cell_size"]), float(m["min_bounds"][0]), float(m["min_bounds"][2]),
                       float(m["max_bounds"][0]), float(m["max_bounds"][2]))


def terrain_for(rt: RiftTerrain, variant, team: int) -> StaticTerrain:
    """``StaticTerrain`` of one team for a (traced) variant index."""
    return StaticTerrain(rt.walkable[variant, team], rt.cell_size, rt.min_x, rt.min_z, rt.max_x, rt.max_z,
                         gaps=rt.gaps[variant, team])


def terrain_pair(rt: RiftTerrain, variant) -> tuple:
    """Per-team tuple, drop-in for ``WorldConfig.terrain``."""
    return tuple(terrain_for(rt, variant, t) for t in (0, 1))


def vision_for(rt: RiftTerrain, variant, grid) -> Any:
    """The base ``VisionGrid`` with the variant's flags and brush labels."""
    out = grid._replace(flags=rt.flags[variant])
    return out._replace(bush_ids=rt.bush_ids[variant]) if grid.bush_ids is not None else out
