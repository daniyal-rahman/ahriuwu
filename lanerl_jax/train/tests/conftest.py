"""Skip the sim suite where JAX is absent, loudly enough to notice.

JAX deliberately does **not** live in the shared ``ml`` conda env: that env is
used by every other worktree and by the live training runs, and adding a package
to it is a change to someone else's environment. The JAX work uses
``.venv-jax`` (a venv layered over ``ml`` with ``--system-site-packages``, so it
sees torch and the server-side packages without duplicating them).

Consequence: ``pytest`` under ``ml`` collects this directory and must skip it
rather than error. Run the sim suite with ``.venv-jax/bin/python -m pytest``.
"""
import os

import pytest

_jax = pytest.importorskip(
    "jax",
    reason="JAX is not installed in this interpreter; run the sim suite with "
           ".venv-jax/bin/python -m pytest lanerl_jax",
)


# x64 is enabled ONCE, here, for the whole suite.
#
# Three test modules used to call `jax.config.update("jax_enable_x64", ...)` at
# import time with different values. pytest imports every module before running
# anything, so the last import won and six movement tests failed depending on
# collection order -- a real order-dependent failure, and the kind that looks
# like a flake.
#
# Enabling x64 does not *force* float64: arrays created with an explicit
# float32 dtype stay float32, which is what the sim does everywhere
# (`empty_state(dtype=jnp.float32)`). It only lets the movement reference
# compare in double precision, which is the one place that needs it.
_jax.config.update("jax_enable_x64", True)

# Never preallocate the GPU. `.venv-jax` is CUDA-enabled, and on the dev box the
# only card has ~1.6 GB free (another python and a llama-server hold the rest),
# so the default 75% preallocation fails outright. This makes the suite run
# wherever it lands instead of depending on which machine picked it up.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
