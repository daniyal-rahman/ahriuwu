"""Opt-in visible health/position history, associated without simulator IDs.

Inputs are exclusively the existing visible entity rows, their padding mask,
and the existing self HUD position. Mutual nearest-neighbour association uses
type/team/subtype and camera-compensated positions. Ambiguous, absent or distant
matches lose their history; an unseen unit never contributes a current row.

Fifteen past 10 Hz samples plus the current row cover 1.5 seconds. Each past
sample supplies rounded HP, displacement from current position, and a known
bit. The 64-unit association radius and 8-unit ambiguity margin are engineering
choices, not claims from the literature. Raw IDs/targets/attack clocks are not
accepted by this API. Callers must reset the memory at episode boundaries.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

from .builder import ENTITY_DIM, N_SLOTS, NORM_DIST, NORM_XY

VISIBLE_HISTORY_INTERFACE = 'viewport-structured-v5-visible-history'
PAST_SAMPLES = 15
HISTORY_FEATURES = 4 * PAST_SAMPLES
HISTORY_ENTITY_DIM = ENTITY_DIM + HISTORY_FEATURES
MATCH_RADIUS = 64.0
MATCH_MARGIN = 8.0


class VisibleHistory(NamedTuple):
    # Samples are newest-first [HP fraction, absolute lane x/y / NORM_DIST].
    samples: jax.Array
    known: jax.Array
    types: jax.Array


def empty_visible_history(batch_shape=(), dtype=jnp.float32):
    return VisibleHistory(
        jnp.zeros((*batch_shape, N_SLOTS, PAST_SAMPLES, 3), dtype),
        jnp.zeros((*batch_shape, N_SLOTS, PAST_SAMPLES), bool),
        jnp.zeros((*batch_shape, N_SLOTS, ENTITY_DIM-4), dtype))


def append_visible_history(entities, pad_mask, self_vec, past):
    """One observer/frame -> augmented rows and next memory; vmap for batches.

No observation object or LaneState is accepted, so slot_unit cannot accidentally
become an association key. Matching tolerates arbitrary row permutations.
Unknown histories are zero with known=0, distinct from an observed zero HP bar.
"""
    if entities.shape != (N_SLOTS, ENTITY_DIM):
        raise ValueError('visible history requires one original v3 entity table')
    valid = ~pad_mask & (entities[:, 0] > .5)
    position = entities[:, 1:3] + self_vec[None, :2] * (NORM_XY/NORM_DIST)
    position = jnp.where(valid[:, None], position, 0.)
    types = entities[:, 4:]
    compatible = (valid[:, None] & past.known[None, :, 0]
                  & jnp.all(types[:, None, :] == past.types[None, :, :], axis=-1))
    distance = jnp.linalg.norm(position[:, None, :]-past.samples[None, :, 0, 1:], axis=-1) * NORM_DIST
    distance = jnp.where(compatible, distance, jnp.inf)
    best = jnp.argmin(distance, axis=1)
    reverse = jnp.argmin(distance, axis=0)
    near = jnp.take_along_axis(distance, best[:, None], axis=1)[:, 0]
    row_two = -jax.lax.top_k(-distance, 2)[0]
    col_two = -jax.lax.top_k(-distance.T, 2)[0]
    matched = (valid & (near <= MATCH_RADIUS)
               & (reverse[best] == jnp.arange(N_SLOTS))
               & (row_two[:, 1]-row_two[:, 0] >= MATCH_MARGIN)
               & ((col_two[:, 1]-col_two[:, 0])[best] >= MATCH_MARGIN))
    known = past.known[best] & matched[:, None]
    samples = jnp.where(known[..., None], past.samples[best], 0.)
    displacement = samples[..., 1:]-position[:, None, :]
    features = jnp.concatenate([samples[..., :1], displacement,
                                known[..., None].astype(entities.dtype)], axis=-1)
    features = jnp.where(known[..., None], features, 0.)
    augmented = jnp.concatenate([entities, features.reshape(N_SLOTS, HISTORY_FEATURES)], axis=-1)
    current = jnp.concatenate([entities[:, 3:4], position], axis=-1)
    current = jnp.where(valid[:, None], current, 0.)
    next_memory = VisibleHistory(
        jnp.concatenate([current[:, None, :], samples[:, :-1, :]], axis=1),
        jnp.concatenate([valid[:, None], known[:, :-1]], axis=1),
        jnp.where(valid[:, None], types, 0.))
    return augmented, next_memory
