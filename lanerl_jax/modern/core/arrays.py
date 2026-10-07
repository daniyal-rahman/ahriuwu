"""Shape-static array helpers shared by the world subsystems."""
from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def first_true(mask: Any, size: int) -> tuple[Any, Any]:
    """``(idx, count)``: the first ``size`` true positions of (n,) ``mask`` in order, padded with n, and the true
    count. Same as ``jnp.nonzero(mask, size=size, fill_value=n)``, which lowers to a much slower GPU kernel."""
    n = mask.shape[0]
    slot = jnp.cumsum(mask, dtype=jnp.int32) - 1
    idx = jnp.full((size,), n, jnp.int32).at[jnp.where(mask, slot, size)].set(jnp.arange(n, dtype=jnp.int32),
                                                                               mode="drop")
    return idx, (slot[-1] + 1 if n else jnp.int32(0))
