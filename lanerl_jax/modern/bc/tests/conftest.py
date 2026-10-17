"""The BC tests run on the CPU and never preallocate the GPU (the converter is a CPU tool)."""
import os

import pytest

pytest.importorskip("jax", reason="run the modern suite with the JAX environment")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
