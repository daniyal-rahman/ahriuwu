"""Elemental Rift and Baron-pit terrain variants as swappable JAX arrays (26.19).

The client ships the transformations as navgrid *overlays* (``map11.bin``
``MapNavGridOverlays``): rectangles of replacement cell flags applied over the
base navgrid ``AIPath_SRX_2``. ``data/build_modern_objectives.py`` applies them
host-side and stores every variant (docs/modern/OBJECTIVES.md §6):

    variant = element * 3 + baron_form
    element    0 none, 1 Infernal, 2 Mountain, 3 Ocean, 4 Cloud, 5 Hextech, 6 Chemtech
               (client MapFlagIndexOverride ids; Cloud and Hextech ship no navgrid overlay,
               so their terrain equals the base map)
    baron_form 0 Hunting (no change), 1 Territorial (Cup overlay), 2 All-Seeing (Tunnel overlay)

``jungle.objectives.terrain_variant(state, now)`` gives the current index; the
step swaps ``walkable[v, team]`` into ``StaticTerrain`` (movement, Flash) and
``flags[v]``/``bush_ids[v]`` into the ``obs.vision.VisionGrid`` (fog). All
selectors are fixed-shape gathers, safe under jit/vmap.
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

from .terrain import StaticTerrain
from ..data import PATCH_DIR

TABLE = PATCH_DIR / "objectives_client.json"
N_ELEMENTS = 7
N_BARON_FORMS = 3


class RiftTerrain(NamedTuple):
    walkable: Any       # (V, 2, H, W) bool per team (gates resolved like ModernMapGrid.walkable)
    flags: Any          # (V, H, W) int32 navgrid flags (brush bit 1, wall bit 2, ...)
    bush_ids: Any       # (V, H, W) int32 edge-connected brush labels (0 = no brush), per variant
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
    return RiftTerrain(jnp.asarray(walk), jnp.asarray(flags.astype(np.int32)), jnp.asarray(bush),
                       float(m["cell_size"]), float(m["min_bounds"][0]), float(m["min_bounds"][2]),
                       float(m["max_bounds"][0]), float(m["max_bounds"][2]))


def terrain_for(rt: RiftTerrain, variant, team: int) -> StaticTerrain:
    """``StaticTerrain`` of one team for a (traced) variant index."""
    return StaticTerrain(rt.walkable[variant, team], rt.cell_size, rt.min_x, rt.min_z, rt.max_x, rt.max_z)


def terrain_pair(rt: RiftTerrain, variant) -> tuple:
    """Per-team tuple, drop-in for ``WorldConfig.terrain``."""
    return tuple(terrain_for(rt, variant, t) for t in (0, 1))


def vision_for(rt: RiftTerrain, variant, grid) -> Any:
    """``obs.vision.VisionGrid`` with the variant's flags and brush labels.

    ``grid`` is the base ``VisionGrid`` (``cfg.vision``); the fast fog uses
    ``bush_ids``, the ray fog uses ``flags``. Brush ids are per-variant labels,
    so equal ids mean "same brush patch" only within one variant.
    """
    out = grid._replace(flags=rt.flags[variant])
    return out._replace(bush_ids=rt.bush_ids[variant]) if grid.bush_ids is not None else out
