"""Qualified target eligibility shared by loss and emission (BC.md section15)."""
import jax.numpy as jnp

from ..data import champion_table as CT
from .events import BUTTON_INDEX as B
from ..screen import MINIMAP_X_MIN, MINIMAP_Y_MIN

GAREN = CT.index_of('Garen')


def pointer_candidates(entities, pad, own_id, button):
    enemy = entities[..., 11] > .5
    champion = entities[..., 4] > .5
    attackable = champion | (entities[..., 5] > .5) | (entities[..., 6] > .5)
    attack = button == B['attack']
    garen_r = (own_id == GAREN) & (button == B['r'])
    # Other abilities retain unknown eligibility rather than inventing restrictions.
    qualified = jnp.where(attack[..., None], enemy & attackable, True)
    qualified &= jnp.where(garen_r[..., None], enemy & champion, True)
    return ~pad & qualified


def emission_logits(logits, entities, pad, own_id):
    for b in (B['attack'], B['r']):
        button = jnp.full(logits.shape[:-1], b, jnp.int32)
        available = pointer_candidates(entities, pad, own_id, button).any(-1)
        requires_pointer = (b == B['attack']) | (own_id == GAREN)
        logits = logits.at[..., b].set(jnp.where(requires_pointer & ~available, -jnp.inf, logits[..., b]))
    return logits


def grid_candidates(nx=96, ny=54):
    x = (jnp.arange(nx) + .5)/nx
    y = (jnp.arange(ny) + .5)/ny
    return ~((x[None, :] >= MINIMAP_X_MIN) & (y[:, None] >= MINIMAP_Y_MIN))
