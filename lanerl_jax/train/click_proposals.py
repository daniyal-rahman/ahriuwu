"""Observation-only candidate screen cells and an exact categorical mixture.

Candidates are projected visible hostile entity centres, without HP ranking,
unit IDs, hit testing, or hidden simulator state. The executed action remains
the same button/x/y screen click, resolved by the ordinary game decoder.
"""
import math

import jax
import jax.numpy as jnp

from lanerl_rl.projection import (DEFAULT_CAM_Y, DEFAULT_FOV_V_DEG,
    DEFAULT_RESOLUTION, DEFAULT_TILT_DEG, FLOOR_Y, MINIMAP_X_MIN,
    MINIMAP_Y_MIN, target_on_screen)
from lanerl_jax.obs.builder import NORM_DIST


def proposal_cells(entities, pad_mask, nx=96, ny=54):
    """Flattened X-major screen bins containing projected entity centres."""
    ds, dn = entities[..., 1] * NORM_DIST, entities[..., 2] * NORM_DIST
    tilt = math.radians(DEFAULT_TILT_DEG)
    ct, st = math.cos(tilt), math.sin(tilt)
    dy = FLOOR_Y - DEFAULT_CAM_Y
    dz = dn - dy * ct / st
    vy, vz = dy * ct + dz * st, -dy * st + dz * ct
    tv = math.tan(math.radians(DEFAULT_FOV_V_DEG) / 2)
    th = tv * DEFAULT_RESOLUTION[0] / DEFAULT_RESOLUTION[1]
    z = jnp.maximum(vz, 1e-6)
    sx, sy = .5 * (1 + ds / (z * th)), .5 * (1 - vy / (z * tv))
    ix = jnp.clip(jnp.floor(sx * nx), 0, nx-1).astype(jnp.int32)
    iy = jnp.clip(jnp.floor(sy * ny), 0, ny-1).astype(jnp.int32)
    in_minimap = ((ix+.5)/nx >= MINIMAP_X_MIN) & ((iy+.5)/ny >= MINIMAP_Y_MIN)
    # Layout: valid,ds,dn,hp,6 kind bits,3 team bits,3 subtype bits.
    valid = (~pad_mask & (entities[..., 0] > .5) & (entities[..., 11] > .5)
             & target_on_screen(ds, dn) & ~in_minimap)
    return ix * ny + iy, valid


def mixture_click_logits(lg_x, lg_y, scores, gate_logit, cells, valid):
    """Exact physical-click distribution, summing duplicate candidate cells.

    The learned gate is proposal mass; absent candidates fall back to the
    old factored click distribution. Return finite log weights, X-major.
    """
    nx, ny = lg_x.shape[-1], lg_y.shape[-1]
    base = (jax.nn.softmax(lg_x)[..., :, None] * jax.nn.softmax(lg_y)[..., None, :])
    leading = base.shape[:-2]
    maximum = jnp.max(jnp.where(valid, scores, -jnp.inf), axis=-1, keepdims=True)
    maximum = jnp.where(jnp.any(valid, axis=-1, keepdims=True), maximum, 0.)
    weights = jnp.exp(jnp.where(valid, scores-maximum, -1e4)) * valid
    weights = weights / jnp.maximum(weights.sum(-1, keepdims=True), 1e-30)
    # Explicit flatten/vmap supports unbatched, rollout and sequence layouts.
    def scatter(index, weight):
        return jnp.zeros(nx*ny, weight.dtype).at[index].add(weight)
    proposed = jax.vmap(scatter)(cells.reshape((-1, cells.shape[-1])),
                                 weights.reshape((-1, weights.shape[-1])))
    proposed = proposed.reshape(leading + (nx*ny,))
    gate = jax.nn.sigmoid(gate_logit) * jnp.any(valid, axis=-1)
    probability = (1-gate[..., None]) * base.reshape(leading + (nx*ny,)) + gate[..., None] * proposed
    # A tiny finite floor keeps entropy gradients defined for underflowed cells.
    return jnp.log(jnp.maximum(probability, 1e-30))
