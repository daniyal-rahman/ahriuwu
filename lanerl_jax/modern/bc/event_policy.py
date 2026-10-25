"""Button-conditioned event policy; Lane BC v2, BC.md section 15."""
from __future__ import annotations

from typing import NamedTuple

import flax.linen as nn
import jax
import jax.numpy as jnp

from ..rl.policy import _Block, TRUNK, HEAD
from .event_inputs import ENTITY_DIM, SELF_DIM, GLOBAL_DIM, ABILITY_FIELDS
from .events import BUTTONS, BUTTON_INDEX as B


class EventPolicyConfig(NamedTuple):
    entity_dim: int = ENTITY_DIM
    self_dim: int = SELF_DIM
    global_dim: int = GLOBAL_DIM
    metadata_dim: int = len(ABILITY_FIELDS) + 4
    n_slots: int = 32
    d_model: int = 128
    n_layers: int = 3
    n_heads: int = 4
    ffn_dim: int = 256
    ctx_dim: int = 256
    mlp_hidden: int = 1024
    mlp_layers: int = 3
    core_dim: int = 512
    champion_embed: int = 16
    button_embed: int = 32
    target_dim: int = 256
    pointer_dim: int = 64
    n_champions: int = 256
    n_patches: int = 2
    patch_embed: int = 4
    n_angles: int = 72
    grid_x: int = 96
    grid_y: int = 54
    conditional_aim: bool = True
    history: bool = True


class EventLogits(NamedTuple):
    button: jax.Array
    pointer: jax.Array
    grid: jax.Array
    angle: jax.Array
    co_move: jax.Array
    co_move_grid: jax.Array
    co_move_order: jax.Array
    own_future: jax.Array
    enemy_future: jax.Array
    own_hp_future: jax.Array


class EventPolicy(nn.Module):
    cfg: EventPolicyConfig = EventPolicyConfig()

    def setup(self):
        c = self.cfg
        self.champion = nn.Embed(c.n_champions, c.champion_embed)
        self.patch = nn.Embed(c.n_patches, c.patch_embed)
        self.entity = nn.Dense(c.d_model, **TRUNK)
        self.entity_missing = nn.Dense(c.d_model, use_bias=False, **TRUNK)
        self.entity_identity = nn.Dense(c.d_model, use_bias=False, **TRUNK)
        self.self_token = nn.Dense(c.d_model, **TRUNK)
        self.self_missing = nn.Dense(c.d_model, use_bias=False, **TRUNK)
        self.blocks = [_Block(c) for _ in range(c.n_layers)]
        self.context = nn.Dense(c.ctx_dim, **TRUNK)
        self.mlp = [nn.Dense(c.mlp_hidden, **TRUNK) for _ in range(c.mlp_layers)]
        self.core_projection = nn.Dense(c.core_dim, **TRUNK)
        self.core_norm = nn.LayerNorm()
        self.gru = nn.GRUCell(c.core_dim)
        self.button_head = nn.Dense(len(BUTTONS), **HEAD)
        self.button_embedding = nn.Embed(len(BUTTONS), c.button_embed)
        self.condition = nn.Dense(c.target_dim, **TRUNK)
        self.pointer_query = nn.Dense(c.pointer_dim, use_bias=False, **HEAD)
        self.pointer_key = nn.Dense(c.pointer_dim, use_bias=False, **TRUNK)
        self.grid_head = nn.Dense(c.grid_x * c.grid_y, **HEAD)
        self.angle_head = nn.Dense(c.n_angles, **HEAD)
        self.co_move_head = nn.Dense(2, **HEAD)
        self.co_move_order_head = nn.Dense(2, **HEAD)
        self.own_future_head = nn.Dense(24 * 14, **HEAD)
        self.enemy_future_head = nn.Dense(24 * 14, **HEAD)
        self.hp_future_head = nn.Dense(1, **HEAD)

    def encode(self, entities, pad, self_vec, global_vec, entity_ids, own_id, enemy_id,
               own_metadata, enemy_metadata, patch, carry):
        c = self.cfg
        if entities.shape[-2:] != (c.n_slots, c.entity_dim) or self_vec.shape[-1] != c.self_dim:
            raise ValueError('v2 observation dimensions differ from the saved schema')
        if not c.history:
            # Keep identical parameter shapes in the history ablation.
            entities = entities.at[..., 20:].set(jnp.nan)
            self_vec = self_vec.at[..., 24:].set(jnp.nan)
        em = ~jnp.isfinite(entities)
        values = jnp.where(em, 0., entities)
        tokens = self.entity(values) + self.entity_missing(em.astype(jnp.float32))
        eid = jnp.clip(entity_ids, 0, c.n_champions - 1)
        tokens += self.entity_identity(self.champion(eid)) * (eid > 0)[..., None]
        own = self.champion(jnp.clip(own_id, 0, c.n_champions - 1))
        enemy = self.champion(jnp.clip(enemy_id, 0, c.n_champions - 1))
        patch_emb = self.patch(jnp.clip(patch, 0, c.n_patches - 1))
        metadata_shape = self_vec.shape[:-1] + (4 * c.metadata_dim,)
        context = jnp.concatenate([self_vec, global_vec, own_metadata.reshape(metadata_shape),
                                   enemy_metadata.reshape(metadata_shape), own, enemy, patch_emb], -1)
        missing = ~jnp.isfinite(context)
        self_token = self.self_token(jnp.where(missing, 0., context)) + self.self_missing(missing.astype(jnp.float32))
        tokens = jnp.concatenate([self_token[..., None, :], tokens], -2)
        padded = jnp.concatenate([jnp.zeros(pad.shape[:-1] + (1,), bool), pad], -1)
        mask = nn.make_attention_mask(~padded, ~padded)
        for block in self.blocks:
            tokens = block(tokens, mask)
        entity_tokens = tokens[..., 1:, :]
        keep = (~pad)[..., None]
        mean = jnp.sum(jnp.where(keep, entity_tokens, 0.), -2) / jnp.maximum(keep.sum(-2), 1)
        maximum = jnp.max(jnp.where(keep, entity_tokens, -1e30), -2)
        maximum = jnp.where(keep.any(-2), maximum, 0.)
        h = nn.relu(self.context(jnp.concatenate([tokens[..., 0, :], mean, maximum, own, enemy, patch_emb], -1)))
        for layer in self.mlp:
            h = nn.relu(layer(h))
        h = self.core_projection(h)
        carry, recurrent = self.gru(carry, self.core_norm(h))
        return h + recurrent, carry, entity_tokens

    def buttons(self, hidden):
        return self.button_head(hidden)

    def targets(self, hidden, entity_tokens, pad, own_metadata, button):
        c = self.cfg
        selected = jnp.take_along_axis(own_metadata, jnp.clip(button - B['q'], 0, 3)[..., None, None], -2)[..., 0, :]
        cast = (button >= B['q']) & (button <= B['r'])
        selected = jnp.where(cast[..., None] & jnp.isfinite(selected), selected, 0.)
        embedding = self.button_embedding(button)
        if not c.conditional_aim:
            selected, embedding = jnp.zeros_like(selected), jnp.zeros_like(embedding)
        query = nn.relu(self.condition(jnp.concatenate([hidden, embedding, selected], -1)))
        pointer = jnp.einsum('...d,...nd->...n', self.pointer_query(query),
                             self.pointer_key(entity_tokens)) / c.pointer_dim ** .5
        pointer = jnp.where(pad, -1e4, pointer)
        return pointer, self.grid_head(query), self.angle_head(query)

    def __call__(self, entities, pad, self_vec, global_vec, entity_ids, own_id, enemy_id,
                 own_metadata, enemy_metadata, patch, carry, *, target_button=None):
        h, carry, tokens = self.encode(entities, pad, self_vec, global_vec, entity_ids, own_id, enemy_id,
                                       own_metadata, enemy_metadata, patch, carry)
        button = self.buttons(h)
        chosen = button.argmax(-1) if target_button is None else target_button
        pointer, grid, angle = self.targets(h, tokens, pad, own_metadata, chosen)
        _, co_grid, _ = self.targets(h, tokens, pad, own_metadata, jnp.full_like(chosen, B['move']))
        condition = jnp.concatenate([h, self.button_embedding(chosen)], -1)
        return EventLogits(button, pointer, grid, angle, self.co_move_head(condition), co_grid,
                           self.co_move_order_head(condition), self.own_future_head(h),
                           self.enemy_future_head(h), self.hp_future_head(h)[..., 0]), carry

    def initial_carry(self, batch_shape=()):
        return jnp.zeros(tuple(batch_shape) + (self.cfg.core_dim,), jnp.float32)
