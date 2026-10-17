"""The click part of the behaviour-cloning loss.

The click heads are a discretised screen: a column of 96 and a row of 54,
whose logits add (`rl/ppo.py`). Two targets for them:

* plain: the clicked cell. Cross-entropy is ``-ppo.click_log_prob``, the
  same term PPO uses.
* sloped (``sigma_cells > 0``): a Gaussian over the cells around the clicked
  one, standard deviation `sigma_cells` cells. A click one cell off then costs
  little more than the exact cell, as it should: the label is itself rounded
  to a cell and the replay's click point is the server's path end. Cells are
  square on screen (1280 / 96 = 720 / 54 pixels), so an isotropic Gaussian is
  a product of one over columns and one over rows, and its cross-entropy with
  the factorised heads is the sum of the two heads' cross-entropies.

Only the decisions whose click is a real target count (`labels.Labels.click_valid`).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

__all__ = ["gaussian_target", "click_cross_entropy"]


def gaussian_target(index, n: int, sigma: float):
    """``(..., n)``: a Gaussian over ``0..n-1`` centred on `index`, normalised; one-hot
    when `sigma` is 0."""
    cells = jnp.arange(n)
    if sigma <= 0:
        return (cells == index[..., None]).astype(jnp.float32)
    w = jnp.exp(-0.5 * ((cells - index[..., None]) / sigma) ** 2)
    return w / jnp.sum(w, axis=-1, keepdims=True)


def click_cross_entropy(screen_x_logits, screen_y_logits, screen_x, screen_y,
                        sigma_cells: float = 0.0):
    """``(...)``: cross-entropy of the click heads against the plain or sloped target."""
    def ce(logits, index):
        target = gaussian_target(index, logits.shape[-1], sigma_cells)
        return -jnp.sum(target * jax.nn.log_softmax(logits, axis=-1), axis=-1)
    return ce(screen_x_logits, screen_x) + ce(screen_y_logits, screen_y)
