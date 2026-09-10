"""``lanerl_rl`` -- RL scaffolding for a 1v1 Garen mirror match on a headless
LeagueSandbox-derived server.

Modules
-------
``constants``  decision rate, map geometry, unit stats, every tensor layout
``frame``      frame decoding, the lane frame, the fog gate, per-agent memory
``obs``        the observation builder (actor) and privileged builder (critic)
``audit``      static + differential proof that no server-only field reaches
               the actor observation
``model``      transformer-over-entities policy, swappable GRU/MLP core, and an
               asymmetric critic that sees the actor's history as well as the
               privileged state
``ppo``        dual-clip recurrent PPO with burn-in minibatching
``reward``     JueWu-shaped zero-sum reward, potential-based shaping, and the
               absolute evaluation metrics that self-play cannot fake
``infer``      batched inference across parallel environments
``env``        two-agent environment scaffolding and server backends
"""

from . import constants  # noqa: F401

__all__ = [
    "constants",
    "frame",
    "obs",
    "audit",
    "model",
    "ppo",
    "reward",
    "infer",
    "env",
]
__version__ = "0.2.0"
