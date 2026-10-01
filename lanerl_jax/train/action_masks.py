"""Button legality from observed own readiness; no tactical last-hit rules.

Attack-move remains legal on AA cooldown: it can move or queue an attack.
R here checks own cast readiness, not target choice/range; target-conditioned
screen masks are a separate interface change. NOOP always remains available.
"""
import jax.numpy as jnp
from lanerl_rl.constants import BUTTONS
from ..obs.combat_features import SELF_FEATURES


def available_buttons(self_vec):
    alive = self_vec[..., 14] < .5
    def ready(name):
        return alive & (self_vec[..., 16 + SELF_FEATURES.index(name + '_available')] > .5)
    values = {'noop': jnp.ones_like(alive), 'move': ready('move'),
              'attack_move': ready('move'), 'recall': ready('recall')}
    values.update({name: ready(name) for name in ('q', 'w', 'e', 'r')})
    return jnp.stack([values[name] for name in BUTTONS], axis=-1)
