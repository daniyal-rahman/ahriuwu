"""Use the dedicated JAX environment; share x64 and compilation-cache settings across tests."""
import os

import pytest

_jax = pytest.importorskip(
    "jax",
    reason="JAX is not installed in this interpreter; run the sim suite with "
           ".venv-jax/bin/python -m pytest lanerl_jax",
)


# Configure once so test import order cannot change movement-reference precision.
_jax.config.update("jax_enable_x64", True)

# Leave device memory available to other processes.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

# Persistent compilation cache: full-tick tests compile for minutes; reruns load them.
from lanerl_jax.modern.jax_cache import enable_compile_cache  # noqa: E402

enable_compile_cache()
